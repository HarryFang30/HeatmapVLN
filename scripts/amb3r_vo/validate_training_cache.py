#!/usr/bin/env python3
"""Validate a full plan and atomically publish the root cache-ready marker."""

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
    ROOT_READY_SCHEMA,
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


def _validate_shard_ready(path: Path, shard: dict, plan: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": SHARD_READY_SCHEMA,
        "complete": True,
        "shard_id": int(shard["shard_id"]),
        "clips_total": int(shard["clip_count"]),
        "frames_total": int(shard["frame_count"]),
        "query_rows_total": int(shard["query_rows"]),
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
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"{path}: {key}={value.get(key)!r}, expected {expected_value!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--write-ready", action="store_true")
    parser.add_argument("--require-shard-ready", action="store_true")
    parser.add_argument("--allowed-root", default="/mnt/afs/liwenhao/agent/370910109")
    args = parser.parse_args()

    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=True)
    plan_path = _under(Path(args.plan), allowed_root)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported plan schema: {plan.get('schema')}")
    expected_plan_policy = {
        "causal": True,
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame": False,
        "query_every_frame_from_min_history": False,
        "future_pose_revisions_used": False,
    }
    for key, expected in expected_plan_policy.items():
        if plan.get(key) != expected:
            raise ValueError(
                f"Plan {key}={plan.get(key)!r}, expected {expected!r}"
            )
    cache_root = _under(Path(plan["cache_root"]), allowed_root)
    control_root = cache_root / "_control"
    ready_path = control_root / "cache.ready.json"
    entries = [entry for shard in plan["shards"] for entry in shard["clips"]]
    if len(entries) != int(plan["clip_count"]):
        raise ValueError("Plan clip_count does not match shard entries")
    if len({entry["clip_key"] for entry in entries}) != len(entries):
        raise ValueError("Plan contains duplicate clip keys")

    errors: list[dict[str, str]] = []
    verified: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_entry = {
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
            for entry in entries
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                verified.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "clip_key": entry["clip_key"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    if args.require_shard_ready:
        for shard in plan["shards"]:
            marker = control_root / f"shard_{int(shard['shard_id']):02d}.ready.json"
            try:
                _validate_shard_ready(marker, shard, plan)
            except Exception as exc:
                errors.append(
                    {
                        "clip_key": f"_control/shard_{int(shard['shard_id']):02d}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    observed_frames = sum(item["frame_count"] for item in verified)
    observed_queries = sum(item["query_rows"] for item in verified)
    if len(verified) != int(plan["clip_count"]):
        errors.append(
            {
                "clip_key": "_plan",
                "error": (
                    f"verified clips {len(verified)} != planned {plan['clip_count']}"
                ),
            }
        )
    if observed_frames != int(plan["frame_count"]):
        errors.append(
            {
                "clip_key": "_plan",
                "error": (
                    f"verified frames {observed_frames} != planned {plan['frame_count']}"
                ),
            }
        )
    if observed_queries != int(plan["query_rows"]):
        errors.append(
            {
                "clip_key": "_plan",
                "error": (
                    f"verified query rows {observed_queries} != planned "
                    f"{plan['query_rows']}"
                ),
            }
        )

    if errors:
        if args.write_ready:
            ready_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "complete": False,
                    "verified_clips": len(verified),
                    "planned_clips": plan["clip_count"],
                    "error_count": len(errors),
                    "first_errors": errors[:50],
                },
                indent=2,
            )
        )
        return 1

    ready = {
        "schema": ROOT_READY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "dataset_root": plan["dataset_root"],
        "cache_root": str(cache_root),
        "splits": plan["splits"],
        "clips_total": len(verified),
        "frames_total": observed_frames,
        "query_rows_total": observed_queries,
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
        "plan_path": str(plan_path),
    }
    if args.write_ready:
        atomic_write_json(ready_path, ready)
    print(json.dumps(ready | {"ready_path": str(ready_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
