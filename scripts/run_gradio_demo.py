import argparse

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

from qvggt.inference.gradio_app import build_demo

def build_argparser():

    parser = argparse.ArgumentParser(description="Launch the QVGGT Gradio demo.")

    parser.add_argument("--quantized-checkpoint-path", default="", help="Optional default checkpoint path shown in the UI")

    parser.add_argument("--port", type=int, default=7860)

    parser.add_argument("--server-name", default="127.0.0.1")

    parser.add_argument("--share", action="store_true")

    return parser

def main():

    args = build_argparser().parse_args()

    demo = build_demo(default_checkpoint_path=args.quantized_checkpoint_path)

    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)

if __name__ == "__main__":

    main()
