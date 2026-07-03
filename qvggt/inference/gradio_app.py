import shutil

import uuid

from pathlib import Path

from .cli import InferenceConfig, load_model_for_inference, run_inference_with_model, save_outputs

_MODEL_CACHE = {}

def _repo_root() -> Path:

    return Path(__file__).resolve().parents[2]

def get_default_example_dir() -> str:

    return str(_repo_root() / "examples" / "demo_room" / "images")

def _get_or_load_model(checkpoint_path: str):

    checkpoint_path = str(Path(checkpoint_path).expanduser())

    if checkpoint_path not in _MODEL_CACHE:

        _MODEL_CACHE[checkpoint_path] = load_model_for_inference(checkpoint_path)

    return _MODEL_CACHE[checkpoint_path]

def _prepare_input_dir(uploaded_files, example_dir: str, run_dir: Path) -> tuple[Path, list[str]]:

    input_dir = run_dir / "images"

    input_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = []

    if uploaded_files:

        for idx, source in enumerate(uploaded_files, start=1):

            source_path = Path(source)

            target_name = f"{idx:02d}{source_path.suffix.lower()}"

            target_path = input_dir / target_name

            shutil.copy2(source_path, target_path)

            copied_paths.append(str(target_path))

    else:

        example_root = Path(example_dir).expanduser().resolve()

        if not example_root.is_dir():

            raise FileNotFoundError(f"Example image directory not found: {example_root}")

        for idx, source_path in enumerate(sorted(example_root.iterdir()), start=1):

            if source_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:

                continue

            target_name = f"{idx:02d}{source_path.suffix.lower()}"

            target_path = input_dir / target_name

            shutil.copy2(source_path, target_path)

            copied_paths.append(str(target_path))

    if not copied_paths:

        raise ValueError("Please upload at least one image, or provide a valid example image directory.")

    return input_dir, copied_paths

def run_demo(checkpoint_path: str, uploaded_files, example_dir: str, export_glb: bool, conf_thres: float):

    if not checkpoint_path or not checkpoint_path.strip():

        raise ValueError("Please provide a quantized checkpoint path.")

    run_id = uuid.uuid4().hex[:8]

    run_dir = _repo_root() / "outputs" / "gradio_demo" / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    input_dir, gallery_paths = _prepare_input_dir(uploaded_files, example_dir, run_dir)

    model = _get_or_load_model(checkpoint_path)

    predictions = run_inference_with_model(model, str(input_dir))

    config = InferenceConfig(

        quantized_checkpoint_path=checkpoint_path,

        input_dir=str(input_dir),

        output_dir=str(run_dir),

        export_glb=export_glb,

    )

    predictions_path, glb_path = save_outputs(predictions, str(run_dir), config, conf_thres=conf_thres)

    status_lines = [

        f"Images: {len(gallery_paths)}",

        f"Predictions: {predictions_path}",

    ]

    if glb_path is not None:

        status_lines.append(f"Visualization: {glb_path}")

        status_lines.append(f"Confidence Threshold: {conf_thres:.1f}%")

    model3d_value = str(glb_path) if glb_path is not None else None

    glb_file_value = str(glb_path) if glb_path is not None else None

    return "\n".join(status_lines), gallery_paths, str(predictions_path), glb_file_value, model3d_value

def build_demo(default_checkpoint_path: str = ""):

    import gradio as gr

    with gr.Blocks(title="QVGGT Demo", theme=gr.themes.Soft()) as demo:

        gr.Markdown(

            """
            # QVGGT Demo
            Run a complete quantized inference pass from a checkpoint, preview the input images, and export the reconstructed 3D scene.
            If no images are uploaded, the demo will fall back to the bundled example images.
            """

        )

        example_dir = gr.State(value=get_default_example_dir())

        with gr.Row(equal_height=False):

            with gr.Column(scale=7):

                checkpoint_path = gr.Textbox(

                    label="Quantized Checkpoint",

                    placeholder="/path/to/model-v2.pt or a directory containing model-v2.pt",

                    value=default_checkpoint_path,

                    info="You can provide either the full path to model-v2.pt or the folder that contains it.",

                )

                uploaded_files = gr.Files(

                    label="Upload Input Images",

                    file_count="multiple",

                    file_types=["image"],

                )

            with gr.Column(scale=20):

                export_glb = gr.Checkbox(value=True, label="Export scene.glb")

                conf_thres = gr.Slider(

                    minimum=0.0,

                    maximum=100.0,

                    value=20.0,

                    step=0.5,

                    label="Confidence Threshold (%)",

                    info="Higher values keep only higher-confidence points in the exported point cloud.",

                )

                run_button = gr.Button("Run Inference", variant="primary", size="lg")

        status = gr.Textbox(label="Run Status", lines=4)

        with gr.Row(equal_height=False):

            with gr.Column(scale=4):

                input_gallery = gr.Gallery(label="Input Images", columns=2, height=520, object_fit="contain")

                predictions_file = gr.File(label="predictions.npz")

                glb_file = gr.File(label="scene.glb")

            with gr.Column(scale=6):

                model3d = gr.Model3D(label="3D Visualization", clear_color=[1.0, 1.0, 1.0, 1.0], height=760)

        run_button.click(

            fn=run_demo,

            inputs=[checkpoint_path, uploaded_files, example_dir, export_glb, conf_thres],

            outputs=[status, input_gallery, predictions_file, glb_file, model3d],

        )

    return demo
