import argparse

import json

from dataclasses import dataclass

from pathlib import Path

import numpy as np

import torch

from ..common.geometry import unproject_depth_map_to_point_map

from ..common.pose_enc import pose_encoding_to_extri_intri

from .io import list_image_files, load_and_preprocess_images

from .loader import VGGTAWQLoader

from .visualize import predictions_to_glb

DEFAULT_W_BIT = 4

DEFAULT_GROUP_SIZE = 128

DEFAULT_QUANTIZE_TYPE = "specific_blocks"

DEFAULT_PREPROCESS_MODE = "crop"

DEFAULT_PREDICTION_MODE = "pointmap"

DEFAULT_CONF_THRES = 3.0

DEFAULT_FILTER_BY_FRAMES = "all"

@dataclass

class InferenceConfig:

    quantized_checkpoint_path: str

    input_dir: str

    output_dir: str

    export_glb: bool = False

def resolve_device(device: str) -> str:

    if device == "auto":

        return "cuda" if torch.cuda.is_available() else "cpu"

    return device

def load_model_for_inference(quantized_checkpoint_path: str):

    device = resolve_device("auto")

    loader = VGGTAWQLoader(

        device=device,

        dtype=torch.float16,

    )

    return loader.load_quantized_model(

        quantized_checkpoint_path=quantized_checkpoint_path,

        w_bit=DEFAULT_W_BIT,

        group_size=DEFAULT_GROUP_SIZE,

        quantize_type=DEFAULT_QUANTIZE_TYPE,

        specific_frame_blocks=None,

        specific_global_blocks=None,

        layer_types=None,

        quantize_camera_head=True,

    )

def run_inference_with_model(model, input_dir: str) -> dict:

    device = resolve_device("auto")

    input_images = list_image_files(input_dir)

    images = load_and_preprocess_images(input_images, mode=DEFAULT_PREPROCESS_MODE).to(device).half().unsqueeze(0)

    with torch.no_grad():

        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    predictions["extrinsic"] = extrinsic

    predictions["intrinsic"] = intrinsic

    np_predictions = {}

    for key, value in predictions.items():

        if isinstance(value, torch.Tensor):

            np_predictions[key] = value.detach().cpu().numpy().squeeze(0)

    np_predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(

        np_predictions["depth"],

        np_predictions["extrinsic"],

        np_predictions["intrinsic"],

    )

    return np_predictions

def run_inference(config: InferenceConfig) -> dict:

    model = load_model_for_inference(config.quantized_checkpoint_path)

    return run_inference_with_model(model, config.input_dir)

def save_outputs(

    predictions: dict,

    output_dir: str,

    config: InferenceConfig,

    conf_thres: float = DEFAULT_CONF_THRES,

) -> tuple[Path, Path | None]:

    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.npz"

    np.savez(predictions_path, **predictions)

    metadata = {

        "quantized_checkpoint_path": config.quantized_checkpoint_path,

        "input_dir": config.input_dir,

        "quantize_type": DEFAULT_QUANTIZE_TYPE,

        "quantization_layout": "inferred_from_checkpoint",

        "quantize_camera_head": "inferred_from_checkpoint",

        "w_bit": DEFAULT_W_BIT,

        "group_size": DEFAULT_GROUP_SIZE,

        "layer_types": "inferred_from_checkpoint",

        "preprocess_mode": DEFAULT_PREPROCESS_MODE,

        "export_glb": config.export_glb,

        "confidence_threshold_percent": conf_thres,

    }

    (output_path / "run_config.json").write_text(json.dumps(metadata, indent=2))

    glb_path = None

    if config.export_glb:

        scene = predictions_to_glb(

            predictions,

            conf_thres=conf_thres,

            filter_by_frames=DEFAULT_FILTER_BY_FRAMES,

            mask_black_bg=False,

            mask_white_bg=False,

            show_cam=True,

            prediction_mode=DEFAULT_PREDICTION_MODE,

        )

        glb_path = output_path / "scene.glb"

        scene.export(file_obj=str(glb_path))

    return predictions_path, glb_path

def run_and_save(config: InferenceConfig) -> tuple[dict, Path, Path | None]:

    predictions = run_inference(config)

    predictions_path, glb_path = save_outputs(predictions, config.output_dir, config)

    return predictions, predictions_path, glb_path

def build_argparser():

    parser = argparse.ArgumentParser(description="Run QVGGT inference from one quantized checkpoint and an image directory.")

    parser.add_argument("--quantized-checkpoint-path", required=True, help="Path to the quantized checkpoint, e.g. model-v2.pt")

    parser.add_argument("--input-dir", required=True, help="Directory containing input images")

    parser.add_argument("--output-dir", required=True, help="Directory to save predictions.npz and optional visualization")

    parser.add_argument("--export-glb", action="store_true")

    return parser

def main():

    parser = build_argparser()

    args = parser.parse_args()

    config = InferenceConfig(

        quantized_checkpoint_path=args.quantized_checkpoint_path,

        input_dir=args.input_dir,

        output_dir=args.output_dir,

        export_glb=args.export_glb,

    )

    print(f"Input images: {config.input_dir}")

    print(f"Output directory: {config.output_dir}")

    _, predictions_path, glb_path = run_and_save(config)

    print(f"Saved predictions to {predictions_path}")

    if glb_path is not None:

        print(f"Saved visualization to {glb_path}")

if __name__ == "__main__":

    main()
