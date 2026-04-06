#!/usr/bin/env python3
"""
Unified visualization entrypoint.

Usage examples:
    python scripts/visualize.py heatmap --checkpoint /path/to/ckpt --num-samples 10 --output-dir ./vis_heatmap
    python scripts/visualize.py trajectory --checkpoint /path/to/ckpt --num-clips 3 --output-dir ./vis_trajectory
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _print_help() -> None:
    print(
        "用法:\n"
        "  python scripts/visualize.py heatmap [args]\n"
        "  python scripts/visualize.py trajectory [args]\n\n"
        "子命令:\n"
        "  heatmap     4 视角热力图对比可视化\n"
        "  trajectory  轨迹热力图时序 / BEV 可视化"
    )


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    subcommand = argv.pop(0)
    if subcommand == "heatmap":
        from scripts.visualization.heatmap import main as target_main
    elif subcommand in {"trajectory", "traj"}:
        from scripts.visualization.trajectory_heatmaps import main as target_main
    else:
        _print_help()
        return 2

    sys.argv = [sys.argv[0], *argv]
    result = target_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
