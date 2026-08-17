#!/usr/bin/env python3
"""Cheap CPU-only shard preflight used before loading the 6.7 GB DA3 model."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from build_training_cache_plan import PLAN_SCHEMA
from cache_contract import (
    HISTORY_POSE_CONVENTION,
    POSE_CONVENTION,
    ROW_POLICY,
    SHARD_READY_SCHEMA,
    atomic_write_json,
    cache_path_for,
    validate_clip_cache,
)


def _under(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay below {root}: {path}") from exc
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--write-ready", action="store_true")
    parser.add_argument("--allowed-root", default="/mnt/afs/lixiaoou/intern/fjl")
    args = parser.parse_args()

    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=True)
    plan_path = _under(Path(args.plan), allowed_root)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported plan schema: {plan.get('schema')}")
    if not 0 <= args.shard_id < int(plan["num_shards"]):
        raise ValueError(f"Invalid shard-id {args.shard_id}")
    shard = plan["shards"][args.shard_id]
    if int(shard["shard_id"]) != args.shard_id:
        raise ValueError("Plan shard order is inconsistent")
    cache_root = _under(Path(plan["cache_root"]), allowed_root)
    marker = cache_root / "_control" / f"shard_{args.shard_id:02d}.ready.json"
    errors = []
    verified = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                validate_clip_cache,
                cache_path_for(cache_root, entry["clip_key"]),
                expected_clip_key=entry["clip_key"],
                expected_frame_count=int(entry["frame_count"]),
                num_history=int(plan["num_history"]),
                min_history=int(plan["min_history"]),
                map_init_window=int(plan["map_init_window"]),
                map_every=int(plan["map_every"]),
            ): entry
            for entry in shard["clips"]
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                verified.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "clip_key": entry["clip_key"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if errors or len(verified) != int(shard["clip_count"]):
        if args.write_ready:
            marker.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "complete": False,
                    "shard_id": args.shard_id,
                    "verified": len(verified),
                    "assigned": shard["clip_count"],
                    "error_count": len(errors),
                    "first_errors": errors[:20],
                },
                indent=2,
            )
        )
        return 1
    ready = {
        "schema": SHARD_READY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "shard_id": args.shard_id,
        "clips_total": len(verified),
        "frames_total": sum(item["frame_count"] for item in verified),
        "query_rows_total": sum(item["query_rows"] for item in verified),
        "failures": 0,
        "num_history": int(plan["num_history"]),
        "min_history": int(plan["min_history"]),
        "translation_scale": 1.0,
        "pose_convention": POSE_CONVENTION,
        "history_pose_convention": HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame": False,
        "query_every_frame_from_min_history": False,
        "map_init_window": int(plan["map_init_window"]),
        "map_every": int(plan["map_every"]),
        "snapshot_timing": (
            "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
        ),
        "future_pose_revisions_used": False,
        "causal": True,
    }
    if args.write_ready:
        atomic_write_json(marker, ready)
    print(json.dumps(ready, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
