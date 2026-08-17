#!/usr/bin/env python3
"""Build a deterministic, frame-balanced AMB3R cache plan for eight workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cache_contract import ROW_POLICY, atomic_write_json, endpoint_frame_ids


PLAN_SCHEMA = "heatmapvln-amb3r-endpoint-pose-cache-plan-v2"


def _under(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay below {root}: {path}") from exc
    return path


def _flat_split(scene_name: str) -> str:
    """Reproduce the existing VLNSlidingWindowDataset scene split exactly."""

    value = int(hashlib.md5(scene_name.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if value < 10 else "train"


def _clip_rows(
    dataset_root: Path,
    splits: list[str],
    max_clips_per_split: int = 0,
    ignored_empty_clip_placeholders: list[str] | None = None,
) -> list[dict]:
    explicit_split_layout = (dataset_root / "train").is_dir()
    rows: list[dict] = []
    seen: set[str] = set()
    for split in splits:
        selected_for_split = 0
        if explicit_split_layout:
            scene_dirs = sorted(
                path for path in (dataset_root / split).iterdir() if path.is_dir()
            )
        else:
            scene_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
            scene_dirs = [
                path for path in scene_dirs if _flat_split(path.name) == split
            ]
        for scene_dir in scene_dirs:
            for clip_dir in sorted(scene_dir.glob("clip_*")):
                if not clip_dir.is_dir():
                    continue
                clip_key = f"{scene_dir.name}/{clip_dir.name}"
                if clip_key in seen:
                    raise ValueError(f"Duplicate clip key: {clip_key}")
                seen.add(clip_key)
                meta_path = clip_dir / "meta.json"
                chunks = sorted((clip_dir / "chunks").glob("chunk_*.npz"))
                # The collection contains one abandoned allocation placeholder:
                # an empty clip_* directory with an empty chunks/ child.  It was
                # never counted as a successful clip and the training dataset
                # cannot index it.  Skip only this exact no-payload state.  A
                # partial payload remains a hard error because silently dropping
                # it would hide real dataset corruption.
                if not meta_path.is_file() and not chunks:
                    if ignored_empty_clip_placeholders is not None:
                        ignored_empty_clip_placeholders.append(clip_key)
                    continue
                if not meta_path.is_file() or not chunks:
                    raise FileNotFoundError(
                        f"Expected meta.json and chunks/chunk_*.npz under {clip_dir}"
                    )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                frame_count = int(meta.get("num_frames", -1))
                if frame_count < 1:
                    raise ValueError(f"Invalid num_frames in {meta_path}: {frame_count}")
                rows.append(
                    {
                        "clip_key": clip_key,
                        "split": split,
                        "clip_path": str(clip_dir.resolve()),
                        "frame_count": frame_count,
                    }
                )
                selected_for_split += 1
                if (
                    max_clips_per_split > 0
                    and selected_for_split >= max_clips_per_split
                ):
                    break
            if (
                max_clips_per_split > 0
                and selected_for_split >= max_clips_per_split
            ):
                break
    return sorted(rows, key=lambda row: row["clip_key"])


def _endpoint_count(
    frame_count: int, map_init_window: int, map_every: int
) -> int:
    return len(
        endpoint_frame_ids(
            frame_count,
            map_init_window=map_init_window,
            map_every=map_every,
        )
    )


def _assign_shards(
    rows: list[dict],
    num_shards: int,
    map_init_window: int,
    map_every: int,
) -> list[dict]:
    bins = [
        {"shard_id": index, "clips": [], "frame_count": 0, "query_rows": 0}
        for index in range(num_shards)
    ]
    # Longest-processing-time greedy assignment balances continuous input frames.
    for row in sorted(rows, key=lambda item: (-item["frame_count"], item["clip_key"])):
        target = min(
            bins,
            key=lambda item: (
                item["frame_count"],
                len(item["clips"]),
                item["shard_id"],
            ),
        )
        output = dict(row)
        output["query_rows"] = _endpoint_count(
            row["frame_count"], map_init_window, map_every
        )
        target["clips"].append(output)
        target["frame_count"] += int(row["frame_count"])
        target["query_rows"] += int(output["query_rows"])
    for shard in bins:
        shard["clips"].sort(key=lambda item: item["clip_key"])
        shard["clip_count"] = len(shard["clips"])
    return bins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--map-init-window", type=int, default=20)
    parser.add_argument("--map-every", type=int, default=8)
    parser.add_argument("--max-clips-per-split", type=int, default=0)
    parser.add_argument(
        "--allowed-root", default="/mnt/afs/lixiaoou/intern/fjl"
    )
    args = parser.parse_args()

    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=True)
    dataset_root = _under(Path(args.dataset_root), allowed_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    cache_root = _under(Path(args.cache_root), allowed_root)
    plan_path = _under(Path(args.plan), allowed_root)
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    if args.num_history < 1 or args.min_history < 1:
        raise ValueError("num-history and min-history must be positive")
    if args.map_init_window < 2 or args.map_every < 1:
        raise ValueError("map-init-window must be >=2 and map-every must be positive")
    if args.map_init_window - 1 < args.min_history:
        raise ValueError(
            "The first stateful endpoint must have at least min-history frames"
        )
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits or any(split not in {"train", "val"} for split in splits):
        raise ValueError("splits must be a comma-separated subset of train,val")

    ignored_empty_clip_placeholders: list[str] = []
    rows = _clip_rows(
        dataset_root,
        splits,
        max_clips_per_split=args.max_clips_per_split,
        ignored_empty_clip_placeholders=ignored_empty_clip_placeholders,
    )
    if not rows:
        raise RuntimeError("No clips selected")
    if any(row["frame_count"] < args.map_init_window for row in rows):
        raise ValueError(
            "Every clip must contain at least map_init_window frames"
        )

    shards = _assign_shards(
        rows,
        args.num_shards,
        args.map_init_window,
        args.map_every,
    )
    by_split = {
        split: {
            "clips": sum(row["split"] == split for row in rows),
            "frames": sum(
                row["frame_count"] for row in rows if row["split"] == split
            ),
            "query_rows": sum(
                _endpoint_count(
                    row["frame_count"], args.map_init_window, args.map_every
                )
                for row in rows
                if row["split"] == split
            ),
        }
        for split in splits
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "splits": splits,
        "num_shards": int(args.num_shards),
        "num_history": int(args.num_history),
        "min_history": int(args.min_history),
        "map_init_window": int(args.map_init_window),
        "map_every": int(args.map_every),
        "translation_scale": 1.0,
        "causal": True,
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame": False,
        "query_every_frame_from_min_history": False,
        "future_pose_revisions_used": False,
        "ignored_empty_clip_placeholder_count": len(
            ignored_empty_clip_placeholders
        ),
        "ignored_empty_clip_placeholders": ignored_empty_clip_placeholders,
        "clip_count": len(rows),
        "frame_count": sum(row["frame_count"] for row in rows),
        "query_rows": sum(
            _endpoint_count(
                row["frame_count"], args.map_init_window, args.map_every
            )
            for row in rows
        ),
        "by_split": by_split,
        "assignment": "longest-frame-count-first greedy",
        "shards": shards,
    }
    atomic_write_json(plan_path, plan)
    print(json.dumps({key: plan[key] for key in (
        "schema", "dataset_root", "cache_root", "splits", "clip_count",
        "frame_count", "query_rows", "by_split",
        "ignored_empty_clip_placeholder_count",
        "ignored_empty_clip_placeholders")}, indent=2))
    print(
        json.dumps(
            {
                "shard_frames": [shard["frame_count"] for shard in shards],
                "shard_queries": [shard["query_rows"] for shard in shards],
                "shard_clips": [shard["clip_count"] for shard in shards],
                "plan": str(plan_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
