#!/usr/bin/env python3
"""
Unified evaluation entrypoint.

Usage examples:
    python scripts/evaluate.py --config configs/train_config_internnav.yaml --checkpoint /path/to/ckpt --split val_unseen
    python scripts/evaluate.py heatmap --config configs/train_heatmap_config.yaml --checkpoint /path/to/ckpt
    python scripts/evaluate.py r2r --config configs/train_config_internnav.yaml --base_checkpoint /path/to/stage1.pth --checkpoint /path/to/stage2.pth --gpu_id 0 --sim_gpu_id 0
    python scripts/evaluate.py r2r --config configs/train_config_internnav.yaml --base_checkpoint /path/to/stage1.pth --gpu_id 0 --sim_gpu_id 0
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _print_help() -> None:
    print(
        "用法:\n"
        "  python scripts/evaluate.py [general-args]\n"
        "  python scripts/evaluate.py heatmap [heatmap-args]\n"
        "  python scripts/evaluate.py r2r [r2r-args]\n\n"
        "说明:\n"
        "  不带子命令时，默认执行通用评估（原 scripts/evaluate.py 语义）。\n"
        "  子命令 `heatmap` 对应热力图专项评估。\n"
        "  子命令 `r2r` 对应 VLN-CE / R2R val_unseen 评估。"
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    dispatch = "general"
    if argv and argv[0] in {"general", "heatmap", "r2r"}:
        dispatch = argv.pop(0)

    if dispatch == "general":
        from scripts.evaluation.general import main as target_main
    elif dispatch == "heatmap":
        from scripts.evaluation.heatmap import main as target_main
    else:
        from scripts.evaluation.r2r_val_unseen import main as target_main

    sys.argv = [sys.argv[0], *argv]
    result = target_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
