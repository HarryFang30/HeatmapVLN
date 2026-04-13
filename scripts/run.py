#!/usr/bin/env python3
"""
Unified script entrypoint.

Usage:
    python scripts/run.py train [train-args]
    python scripts/run.py evaluate [evaluate-args]
    python scripts/run.py evaluate heatmap [heatmap-args]
    python scripts/run.py evaluate r2r [r2r-args]
    python scripts/run.py visualize heatmap [visualize-args]
    python scripts/run.py visualize trajectory [visualize-args]
    python scripts/run.py inference [inference-args]

Notes:
    - Existing direct entrypoints (train.py / evaluate.py / visualize.py /
      inference.py) are kept for compatibility.
    - Dispatch resets sys.argv before importing the target module so legacy
      scripts that inspect argv at import time still work.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COMMAND_MODULES = {
    "train": "scripts.train",
    "evaluate": "scripts.evaluate",
    "visualize": "scripts.visualize",
    "inference": "scripts.inference",
}


def _print_help() -> None:
    print(
        "用法:\n"
        "  python scripts/run.py train [train-args]\n"
        "  python scripts/run.py evaluate [evaluate-args]\n"
        "  python scripts/run.py evaluate heatmap [heatmap-args]\n"
        "  python scripts/run.py evaluate r2r [r2r-args]\n"
        "  python scripts/run.py visualize heatmap [visualize-args]\n"
        "  python scripts/run.py visualize trajectory [visualize-args]\n"
        "  python scripts/run.py inference [inference-args]\n\n"
        "子命令:\n"
        "  train       训练主入口\n"
        "  evaluate    评估总入口\n"
        "  visualize   可视化总入口\n"
        "  inference   推理入口"
    )


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    command = argv.pop(0)
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        _print_help()
        return 2

    sys.argv = [sys.argv[0], *argv]
    module = importlib.import_module(module_name)
    target_main = getattr(module, "main", None)
    if target_main is None:
        raise RuntimeError(f"Entrypoint `{module_name}` does not define main()")

    result = target_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
