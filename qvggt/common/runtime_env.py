import importlib.util

import io

import logging

import os

import sys

import warnings

from contextlib import contextmanager, redirect_stderr, redirect_stdout

from pathlib import Path

def _prepend_if_exists(path: Path) -> None:

    resolved = path.resolve()

    if resolved.is_dir():

        path_str = str(resolved)

        if path_str not in sys.path:

            sys.path.insert(0, path_str)

def ensure_runtime_paths() -> None:

    repo_root = Path(__file__).resolve().parents[1]

    project_root = repo_root.parent

    hf_src = os.environ.get("QVGGT_HF_SRC")

    if hf_src:

        _prepend_if_exists(Path(hf_src))

    else:

        _prepend_if_exists(project_root / "vggt-hf" / "src")

    if importlib.util.find_spec("transformers") is None:

        raise ImportError(

            "transformers is not available. Install the runtime dependencies or "

            "set QVGGT_HF_SRC to a local vggt-hf/src checkout."

        )

@contextmanager

def suppress_output():

    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):

        yield

def configure_quiet_runtime() -> None:

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    warnings.filterwarnings("ignore", category=UserWarning)

    warnings.filterwarnings("ignore", category=FutureWarning)

    for logger_name in ("transformers", "accelerate", "awq"):

        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:

        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        transformers_logging.disable_progress_bar()

    except Exception:

        pass
