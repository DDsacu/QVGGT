# QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer

---

[Zhizhen Pan](https://ddsacu.github.io/)1,2, [Hesong Wang](https://viridisgreen.github.io/)1, [Huan Wang](https://huanwang.tech/)1*

1Westlake University    2Beijing University of Posts and Telecommunications

*Corresponding author

## 🔥 News

- **[2026.07.03]** Code and checkpoints released.
- **[2026.05.30]** Project page released.
- **[2026.02.21]** The paper has been accepted by CVPR 2026!

## 📖 Overview

QVGGT is a post-training quantization framework tailored for the Visual Geometry Grounded Transformer (VGGT).

## 🛠️ Installation

We recommend cloning **QVGGT** and **vggt-hf** into the same parent directory:

```bash
git clone https://github.com/DDsacu/QVGGT.git
git clone https://github.com/DDsacu/vggt-hf.git
```

The expected layout is:

```text
workspace/
  QVGGT/
  vggt-hf/
```

### 0. Create and activate an environment

```bash
conda create -n qvggt python=3.11 -y
conda activate qvggt
pip install --upgrade pip
```

### 1. Install PyTorch

Install a CUDA-enabled PyTorch build that matches your local CUDA toolkit and GPU driver.

Please follow the official instructions from PyTorch:

- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

After installation, verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Install this repository's Python dependencies

From the `QVGGT_CVPR26` repository root:

```bash
cd QVGGT
pip install -r requirements.txt
```

This installs the common runtime dependencies for QVGGT inference.

### 3. Install `vggt-hf`

From the parent workspace directory:

```bash
cd ../vggt-hf
pip install -e .
```

This step is required because QVGGT loads the VGGT model definition from `vggt-hf`.

### 4. Build `awq_inference_engine`

From the repository root:

```bash
cd qvggt/runtime/awq_inference_engine
pip install -e .
cd ../../..
```

This step builds the CUDA extension required by the AWQ inference runtime. This will speed up our reasoning process!

## ⚡ Quick Start

From the `QVGGT_CVPR26` repository root, run inference with a quantized checkpoint:

```bash
python scripts/run_inference.py \
  --quantized-checkpoint-path /path/to/model-v2.pt \
  --input-dir /path/to/input_images \
  --output-dir /path/to/output_dir
```

To additionally export a 3D visualization:

```bash
python scripts/run_inference.py \
  --quantized-checkpoint-path /path/to/model-v2.pt \
  --input-dir /path/to/input_images \
  --output-dir /path/to/output_dir \
  --export-glb
```

## 🎬 Demo

We also provide a Gradio demo that runs one complete inference pass and visualizes the result.

Bundled example input images are included in:

```text
examples/cat/images
```

Launch the demo from the repository root:

```bash
python scripts/run_gradio_demo.py \
  --quantized-checkpoint-path /path/to/model-v2.pt \
  --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

## 🛣️ TODO Roadmap

Planned next-stage public release items:

- quantization pipeline
- calibration data preparation
- evaluation on geometry benchmarks
- example checkpoints and download instructions
- demo assets and visualization examples

## Acknowledgements

This codebase builds on [VGGT](https://github.com/facebookresearch/vggt) and [AWQ](https://github.com/mit-han-lab/llm-awq). Thanks for their awesome work!

## Contact

For questions or collaboration requests, please feel free to open an issue or contact me at `zhizhenpan9@gmail.com`

## Citation

If you find QVGGT useful, please consider citing:

```bibtex
@article{pan2026qvggt,
        title={QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer},
        author={Pan, Zhizhen and Wang, Hesong and Wang, Huan},
        journal={arXiv preprint arXiv:2605.31124},
        year={2026}
      }
```

