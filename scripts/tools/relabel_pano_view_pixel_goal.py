#!/usr/bin/env python3
"""Relabel panoramic view_id + pixel goals for r2r_paronamic_data clips.

Writes ``pano_view_labels.json`` into each clip directory (in-place sidecar).
Does not modify RGB/depth/pose data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.pano_view_pixel_goal import (  # noqa: E402
    LABEL_VERSION,
    label_clip_frames,
    write_clip_labels,
)


def _discover_clips(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    clips: list[Path] = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for clip_dir in sorted(scene_dir.iterdir()):
            if clip_dir.is_dir() and (clip_dir / "meta.json").exists():
                clips.append(clip_dir)
    return clips


def main() -> None:
    p = argparse.ArgumentParser(description="Relabel pano view_id + pixel goals (C3 policy)")
    p.add_argument(
        "--root",
        default="/home/intern/zhr/fjl/r2r_paronamic_data",
        help="Dataset root containing train/ split",
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=256, help="Projection resolution (square)")
    p.add_argument("--min-goal-len", type=int, default=3)
    p.add_argument("--max-side-dist-m", type=float, default=6.0)
    p.add_argument("--depth-tolerance", type=float, default=0.5)
    p.add_argument("--min-history", type=int, default=5)
    p.add_argument("--limit-clips", type=int, default=0, help="0 = all clips")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--summary-json",
        default="",
        help="Optional path to write aggregate stats JSON",
    )
    args = p.parse_args()

    clips = _discover_clips(Path(args.root), args.split)
    if args.limit_clips > 0:
        clips = clips[: args.limit_clips]

    print(f"Relabeling {len(clips)} clips under {args.root}/{args.split}")
    print(
        f"  policy=C3  image_size={args.image_size}  min_goal_len={args.min_goal_len}  "
        f"max_side_dist_m={args.max_side_dist_m}"
    )

    kind_counter: Counter[str] = Counter()
    view_counter: Counter[str] = Counter()
    legacy_mismatch = 0
    pixel_total = 0
    errors: list[str] = []

    for i, clip_dir in enumerate(clips):
        try:
            frames = label_clip_frames(
                clip_dir,
                img_size=args.image_size,
                min_goal_len=args.min_goal_len,
                max_side_dist_m=args.max_side_dist_m,
                depth_tolerance=args.depth_tolerance,
                min_history=args.min_history,
            )
            if not args.dry_run:
                write_clip_labels(clip_dir, frames)

            for entry in frames.values():
                kind_counter[entry["sample_kind"]] += 1
                view_counter[entry["pano_view_id"]] += 1
                if entry["sample_kind"] == "pixel":
                    pixel_total += 1
                    legacy = entry.get("legacy_front_pixel_goal")
                    pano = entry.get("pano_pixel_goal")
                    if legacy is not None and pano is not None and entry["pano_view_id"] != "front":
                        legacy_mismatch += 1
                    elif legacy is None and pano is not None:
                        legacy_mismatch += 1

            if (i + 1) % 50 == 0 or i + 1 == len(clips):
                print(f"  processed {i + 1}/{len(clips)} clips", flush=True)
        except Exception as exc:
            errors.append(f"{clip_dir}: {exc}")
            print(f"  ERROR {clip_dir}: {exc}", flush=True)

    summary = {
        "version": LABEL_VERSION,
        "policy": "C3",
        "root": str(args.root),
        "split": args.split,
        "num_clips": len(clips),
        "num_errors": len(errors),
        "sample_kind_counts": dict(kind_counter),
        "pano_view_id_counts": dict(view_counter),
        "pixel_samples": pixel_total,
        "pixel_non_front_or_legacy_miss": legacy_mismatch,
        "errors": errors[:20],
    }

    print("\n=== Relabel summary ===")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {out}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
