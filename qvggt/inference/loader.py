import os

import tempfile

from collections import OrderedDict

from pathlib import Path

import torch

import torch.nn as nn

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ..common.runtime_env import configure_quiet_runtime, ensure_runtime_paths, suppress_output

configure_quiet_runtime()

with suppress_output():

    ensure_runtime_paths()

with suppress_output():

    from ..runtime.awq import WQLinear

    from transformers import VGGTConfig, VGGTForMultiTask

def _convert_floating_tensors(value, dtype):

    if torch.is_tensor(value):

        if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:

            return value

        if value.dtype in [torch.float32, torch.float64, torch.bfloat16]:

            return value.to(dtype=dtype)

        return value

    if isinstance(value, tuple):

        return tuple(_convert_floating_tensors(item, dtype) for item in value)

    if isinstance(value, list):

        return [_convert_floating_tensors(item, dtype) for item in value]

    if isinstance(value, dict):

        return {key: _convert_floating_tensors(item, dtype) for key, item in value.items()}

    return value


class FP16InferenceWrapper(nn.Module):

    def __init__(self, model: nn.Module, dtype: torch.dtype):

        super().__init__()

        self.model = model

        self.dtype = dtype

    def forward(self, *args, **kwargs):

        args = _convert_floating_tensors(args, self.dtype)

        kwargs = _convert_floating_tensors(kwargs, self.dtype)

        output = self.model(*args, **kwargs)

        return _convert_floating_tensors(output, self.dtype)

    def __getattr__(self, name):

        if name in {"model", "dtype"}:

            return super().__getattr__(name)

        return getattr(self.model, name)

class VGGTAWQLoader:

    def __init__(self, model_path: str | None = None, device: str = "cuda", dtype=torch.float16, disable_cict_inference: bool = False):

        self.model_path = model_path

        self.device = device

        self.dtype = dtype

        self.disable_cict_inference = disable_cict_inference

        self.config = None

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:

        raw_path = Path(checkpoint_path).expanduser()

        candidates = []

        if raw_path.is_absolute():

            candidates.append(raw_path)

        else:

            candidates.append((Path.cwd() / raw_path))

            candidates.append((Path.home() / raw_path))

        checked = []

        for candidate in candidates:

            candidate = candidate.resolve()

            checked.append(str(candidate))

            if candidate.is_file():

                return str(candidate)

            if candidate.is_dir():

                model_v2 = candidate / "model-v2.pt"

                checked.append(str(model_v2))

                if model_v2.is_file():

                    return str(model_v2)

        raise FileNotFoundError(

            "Quantized checkpoint not found. Please provide a valid path to `model-v2.pt` "

            f"or to a directory containing it. Checked: {checked}"

        )

    def _load_config(self):

        if self.config is None:

            if self.model_path is not None:

                config_path = os.path.dirname(self.model_path)

                self.config = VGGTConfig.from_pretrained(config_path, trust_remote_code=True)

            else:

                self.config = VGGTConfig()

            self.config.use_cache = False

        return self.config

    def _checkpoint_is_self_contained(self, state_dict):

        required_keys = (

            "aggregator.patch_embed.patch_embed.proj.weight",

            "aggregator.camera_token",

            "camera_head.pose_branch.fc1.weight",

        )

        return all(key in state_dict for key in required_keys)

    def _infer_quantization_layout(self, state_dict):

        frame_blocks = set()

        global_blocks = set()

        quantize_camera_head = False

        layer_types = set()

        for key in state_dict.keys():

            if key.endswith(".qweight"):

                if ".frame_blocks." in key:

                    block_idx = int(key.split(".frame_blocks.", 1)[1].split(".", 1)[0])

                    frame_blocks.add(block_idx)

                elif ".global_blocks." in key:

                    block_idx = int(key.split(".global_blocks.", 1)[1].split(".", 1)[0])

                    global_blocks.add(block_idx)

                elif "camera_head.trunk." in key:

                    quantize_camera_head = True

                for layer_type in ("qkv", "proj", "fc1", "fc2"):

                    if f".{layer_type}." in key:

                        layer_types.add(layer_type)

        return {

            "frame_blocks": sorted(frame_blocks) or None,

            "global_blocks": sorted(global_blocks) or None,

            "quantize_camera_head": quantize_camera_head,

            "layer_types": sorted(layer_types) or None,

        }

    def _extract_cict_token_from_state_dict(self, state_dict):

        for key in ("camera_head.cict_token", "model.camera_head.cict_token", "cict_token"):

            if key in state_dict and torch.is_tensor(state_dict[key]):

                return state_dict[key].detach().cpu().view(-1)

        return None

    def _apply_cict_token_to_model(self, model, cict_token):

        if self.disable_cict_inference or cict_token is None:

            return

        if not hasattr(model, "camera_head") or not hasattr(model.camera_head, "set_cict_token"):

            return

        model.camera_head.set_cict_token(cict_token.to(dtype=self.dtype))

    def _create_empty_model(self):

        config = self._load_config()

        def skip(*args, **kwargs):

            return None

        original_reset_linear = torch.nn.Linear.reset_parameters

        original_reset_layernorm = torch.nn.LayerNorm.reset_parameters

        original_kaiming_uniform = torch.nn.init.kaiming_uniform_

        original_kaiming_normal = torch.nn.init.kaiming_normal_

        original_uniform = torch.nn.init.uniform_

        original_normal = torch.nn.init.normal_

        torch.nn.Linear.reset_parameters = lambda self: None

        torch.nn.LayerNorm.reset_parameters = lambda self: None

        torch.nn.init.kaiming_uniform_ = skip

        torch.nn.init.kaiming_normal_ = skip

        torch.nn.init.uniform_ = skip

        torch.nn.init.normal_ = skip

        try:

            with init_empty_weights():

                model = VGGTForMultiTask(config)

        finally:

            torch.nn.Linear.reset_parameters = original_reset_linear

            torch.nn.LayerNorm.reset_parameters = original_reset_layernorm

            torch.nn.init.kaiming_uniform_ = original_kaiming_uniform

            torch.nn.init.kaiming_normal_ = original_kaiming_normal

            torch.nn.init.uniform_ = original_uniform

            torch.nn.init.normal_ = original_normal

        return model

    def _find_linear_layers(self, module, prefix=""):

        layers = {}

        for name, child in module.named_children():

            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):

                layers[full_name] = child

            else:

                layers.update(self._find_linear_layers(child, full_name))

        return layers

    def _should_quantize_layer(

        self,

        layer_name: str,

        quantize_type: str,

        frame_blocks,

        global_blocks,

        layer_types,

        quantize_camera_head: bool,

    ) -> bool:

        if quantize_camera_head and "camera_head" in layer_name:

            return "trunk." in layer_name and not any(

                non_linear in layer_name for non_linear in ["norm", "bn", "ln", "layernorm", "batchnorm", "groupnorm"]

            )

        if not layer_name.startswith("aggregator.") or "patch_embed" in layer_name:

            return False

        def get_layer_type_from_name(name):

            for key in ["qkv", "proj", "fc1", "fc2"]:

                if f".{key}" in name:

                    return key

            return "unknown"

        layer_type = get_layer_type_from_name(layer_name)

        if layer_types is not None and layer_type not in layer_types:

            return False

        if ".frame_blocks." in layer_name:

            block_idx = int(layer_name.split(".frame_blocks.", 1)[1].split(".", 1)[0])

            if quantize_type == "all":

                return True

            if quantize_type == "specific_blocks":

                return frame_blocks is not None and block_idx in frame_blocks

            return False

        if ".global_blocks." in layer_name:

            block_idx = int(layer_name.split(".global_blocks.", 1)[1].split(".", 1)[0])

            if quantize_type == "all":

                return True

            if quantize_type == "specific_blocks":

                return global_blocks is not None and block_idx in global_blocks

            return False

        return False

    def _quantize_specified_blocks(

        self,

        model,

        w_bit: int,

        group_size: int,

        quantize_type: str,

        frame_blocks,

        global_blocks,

        layer_types,

        quantize_camera_head: bool,

    ):

        all_linear_layers = self._find_linear_layers(model)

        quantized_layers = []

        for layer_name, linear_module in all_linear_layers.items():

            if not self._should_quantize_layer(

                layer_name,

                quantize_type=quantize_type,

                frame_blocks=frame_blocks,

                global_blocks=global_blocks,

                layer_types=layer_types,

                quantize_camera_head=quantize_camera_head,

            ):

                continue

            *parent_names, attr_name = layer_name.split(".")

            parent_module = model

            for parent_name in parent_names:

                parent_module = getattr(parent_module, parent_name)

            wq_linear = WQLinear.from_linear(linear_module, w_bit=w_bit, group_size=group_size, init_only=True)

            setattr(parent_module, attr_name, wq_linear)

            quantized_layers.append(layer_name)

        return quantized_layers

    def _merge_weights_with_selective_quantization(

        self,

        quantized_checkpoint_path: str,

        quantize_type: str,

        specific_frame_blocks,

        specific_global_blocks,

        quantized_layers,

        quantize_camera_head: bool,

    ) -> tuple[OrderedDict, torch.Tensor | None]:

        quantized_state_dict = torch.load(quantized_checkpoint_path, map_location="cpu")

        cict_token = self._extract_cict_token_from_state_dict(quantized_state_dict)

        for aux_key in ("cict_token", "cict_topk", "unquantized_patch_start_idx"):

            if aux_key in quantized_state_dict:

                del quantized_state_dict[aux_key]

        if self._checkpoint_is_self_contained(quantized_state_dict):

            merged_state_dict = OrderedDict(quantized_state_dict)

        else:

            if self.model_path is None:

                raise ValueError(

                    "Checkpoint is not self-contained. Please provide an original FP16 model checkpoint via model_path."

                )

            original_state_dict = torch.load(self.model_path, map_location="cpu")

            merged_state_dict = OrderedDict(original_state_dict)

        def get_layer_type_from_key(key):

            for layer_type in ["qkv", "proj", "fc1", "fc2"]:

                if f".{layer_type}" in key:

                    return layer_type

            if "camera_head" in key:

                return "linear"

            return "unknown"

        def block_selected(key):

            if quantize_camera_head and "camera_head" in key:

                return True

            if ".frame_blocks." in key:

                block_idx = int(key.split(".frame_blocks.", 1)[1].split(".", 1)[0])

                return quantize_type == "all" or (

                    quantize_type == "specific_blocks"

                    and specific_frame_blocks is not None

                    and block_idx in specific_frame_blocks

                )

            if ".global_blocks." in key:

                block_idx = int(key.split(".global_blocks.", 1)[1].split(".", 1)[0])

                return quantize_type == "all" or (

                    quantize_type == "specific_blocks"

                    and specific_global_blocks is not None

                    and block_idx in specific_global_blocks

                )

            return False

        keys_to_remove = set()

        for key, value in quantized_state_dict.items():

            if not block_selected(key):

                continue

            layer_type = get_layer_type_from_key(key)

            if quantized_layers is not None and layer_type not in quantized_layers:

                continue

            merged_state_dict[key] = value

            for suffix in [".qweight", ".scales", ".scaled_zeros"]:

                if key.endswith(suffix):

                    original_key = key.replace(suffix, ".weight")

                    if original_key in merged_state_dict and original_key != key:

                        keys_to_remove.add(original_key)

        for key in keys_to_remove:

            merged_state_dict.pop(key, None)

        return merged_state_dict, cict_token

    def load_quantized_model(

        self,

        quantized_checkpoint_path: str,

        w_bit: int = 4,

        group_size: int = 128,

        quantize_type: str = "specific_blocks",

        specific_frame_blocks=None,

        specific_global_blocks=None,

        layer_types=None,

        quantize_camera_head: bool = True,

    ):

        quantized_checkpoint_path = self._resolve_checkpoint_path(quantized_checkpoint_path)

        raw_quantized_state_dict = torch.load(quantized_checkpoint_path, map_location="cpu")

        if self._checkpoint_is_self_contained(raw_quantized_state_dict):

            inferred_layout = self._infer_quantization_layout(raw_quantized_state_dict)

            specific_frame_blocks = inferred_layout["frame_blocks"]

            specific_global_blocks = inferred_layout["global_blocks"]

            quantize_camera_head = inferred_layout["quantize_camera_head"]

            layer_types = inferred_layout["layer_types"]

            quantize_type = "specific_blocks"

        merged_state_dict, cict_token = self._merge_weights_with_selective_quantization(

            quantized_checkpoint_path=quantized_checkpoint_path,

            quantize_type=quantize_type,

            specific_frame_blocks=specific_frame_blocks,

            specific_global_blocks=specific_global_blocks,

            quantized_layers=layer_types,

            quantize_camera_head=quantize_camera_head,

        )

        model = self._create_empty_model()

        self._quantize_specified_blocks(

            model=model,

            w_bit=w_bit,

            group_size=group_size,

            quantize_type=quantize_type,

            frame_blocks=specific_frame_blocks,

            global_blocks=specific_global_blocks,

            layer_types=layer_types,

            quantize_camera_head=quantize_camera_head,

        )

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:

            torch.save(merged_state_dict, tmp.name)

            with suppress_output():

                load_checkpoint_and_dispatch(

                    model,

                    tmp.name,

                    device_map={"": self.device},

                    no_split_module_classes=[],

                    offload_folder=None,

                    offload_state_dict=False,

                )

        for _, module in model.named_modules():

            if "WQLinear" in str(type(module)):

                if hasattr(module, "qweight"):

                    module.qweight.data = module.qweight.data.to(self.device)

                if hasattr(module, "scales"):

                    module.scales.data = module.scales.data.to(self.device).to(self.dtype)

                if hasattr(module, "scaled_zeros"):

                    module.scaled_zeros.data = module.scaled_zeros.data.to(self.device).to(self.dtype)

                if hasattr(module, "qzeros"):

                    module.qzeros.data = module.qzeros.data.to(self.device)

                if hasattr(module, "g_idx"):

                    module.g_idx.data = module.g_idx.data.to(self.device)

                if hasattr(module, "bias") and module.bias is not None:

                    module.bias.data = module.bias.data.to(self.device).to(self.dtype)

        if self.dtype == torch.float16:

            model = model.half()

        self._apply_cict_token_to_model(model, cict_token)

        model = FP16InferenceWrapper(model, self.dtype)

        model.eval()

        return model
