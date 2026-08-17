#!/usr/bin/env python3
"""Persistent-model worker for causal AMB3R map-endpoint caches.

Each clip is ingested exactly once. Rows are snapshotted only at the official
stateful map endpoints (initialization, every ``map_every`` frames, and the
final tail). A snapshot is materialized before any later RGB is ingested, so
later trajectory revisions cannot leak into an earlier training row.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cache_contract import (
    CACHE_SCHEMA,
    HISTORY_POSE_CONVENTION,
    POSE_CONVENTION,
    ROW_POLICY,
    SHARD_READY_SCHEMA,
    atomic_write_json,
    cache_path_for,
    endpoint_frame_ids,
    history_indices,
    sidecar_path,
    validate_clip_cache,
)
from build_training_cache_plan import PLAN_SCHEMA


def _under(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay below {root}: {path}") from exc
    return path


def _decode_rgb(raw: object) -> np.ndarray:
    import cv2

    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        image = np.asarray(raw)[..., :3]
        return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
    if isinstance(raw, np.ndarray):
        encoded = np.asarray(raw, dtype=np.uint8).reshape(-1)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(raw, dtype=np.uint8)
    else:
        encoded = np.asarray(raw, dtype=np.uint8).reshape(-1)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode an rgb_front frame")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_rgb_only(clip_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only frame IDs and continuous front RGB; never read pose arrays."""

    meta_path = clip_dir / "meta.json"
    chunk_paths = sorted((clip_dir / "chunks").glob("chunk_*.npz"))
    if not meta_path.is_file() or not chunk_paths:
        raise FileNotFoundError(f"Missing meta/chunks under {clip_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows: list[tuple[int, np.ndarray]] = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=True) as payload:
            missing = {"frame_ids", "rgb_front"}.difference(payload.files)
            if missing:
                raise KeyError(f"{chunk_path} is missing {sorted(missing)}")
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
            rgb = payload["rgb_front"]
            if len(frame_ids) != len(rgb):
                raise ValueError(f"Chunk field lengths differ in {chunk_path}")
            rows.extend(
                (int(frame_id), _decode_rgb(raw_rgb))
                for frame_id, raw_rgb in zip(frame_ids, rgb)
            )
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise ValueError(f"No RGB frames in {clip_dir}")
    frame_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    expected = np.arange(len(rows), dtype=np.int64)
    if not np.array_equal(frame_ids, expected):
        raise ValueError(
            "AMB3R requires continuous IDs 0..T-1; got "
            f"first={frame_ids[0]} last={frame_ids[-1]} count={len(frame_ids)}"
        )
    declared = int(meta.get("num_frames", -1))
    if declared != len(rows):
        raise ValueError(f"meta num_frames={declared}, decoded frames={len(rows)}")
    return frame_ids, np.stack([row[1] for row in rows]), meta


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _export_clip(
    *,
    entry: dict[str, Any],
    cache_root: Path,
    session: Any,
    num_history: int,
    min_history: int,
    map_init_window: int,
    map_every: int,
    resolution: tuple[int, int],
    checkpoint_path: Path,
) -> dict[str, Any]:
    clip_key = str(entry["clip_key"])
    clip_dir = Path(entry["clip_path"]).expanduser().resolve(strict=True)
    output = cache_path_for(cache_root, clip_key)
    manifest_output = sidecar_path(output)
    frame_ids, images_rgb, meta = _load_rgb_only(clip_dir)
    frame_count = len(frame_ids)
    if frame_count != int(entry["frame_count"]):
        raise ValueError(
            f"Plan frame_count={entry['frame_count']} but decoded {frame_count}"
        )
    current_frame_ids = endpoint_frame_ids(
        frame_count,
        map_init_window=map_init_window,
        map_every=map_every,
    )
    query_rows = len(current_frame_ids)
    history_frame_ids = np.full(
        (query_rows, num_history), -1, dtype=np.int64
    )
    history_counts = np.zeros(query_rows, dtype=np.int64)
    history_rel_poses = np.zeros(
        (query_rows, num_history, 4), dtype=np.float32
    )
    phase_counts: Counter[str] = Counter()

    session_id = "offline-cache-" + clip_key.replace("/", "-")
    session.reset(session_id, max_frames=frame_count)
    started = time.monotonic()
    endpoint_rows = {
        int(current): row for row, current in enumerate(current_frame_ids)
    }
    for current in range(frame_count):
        session.ingest(
            session_id,
            frame_id=current,
            frame_rgb=images_rgb[current],
            capture_step=current,
        )
        row = endpoint_rows.get(current)
        if row is None:
            continue
        history = history_indices(current, num_history)
        result = session.query(
            session_id,
            current_frame_id=current,
            history_frame_ids=history.tolist(),
            translation_scale=1.0,
        )
        prediction = np.asarray(result.history_rel_poses, dtype=np.float32)
        if not result.ready or prediction.shape != (len(history), 4):
            raise RuntimeError(
                f"Query t={current} not ready or wrong shape {prediction.shape}"
            )
        if result.last_mapped_frame_id != current:
            raise RuntimeError(
                "Endpoint query did not commit the current frame: "
                f"current={current}, last_mapped={result.last_mapped_frame_id}"
            )
        count = len(history)
        history_frame_ids[row, :count] = history
        history_counts[row] = count
        history_rel_poses[row, :count] = prediction
        phase_counts[str(result.provider_phase)] += 1
    inference_seconds = time.monotonic() - started
    final_tail_length = (frame_count - map_init_window) % map_every
    forced_final_tail = final_tail_length != 0
    endpoint_kind_counts = {
        "initialization": 1,
        "periodic": (frame_count - map_init_window) // map_every,
        "forced_final_tail": int(forced_final_tail),
    }

    # Remove a stale commit marker before replacing a previously invalid cache.
    manifest_output.unlink(missing_ok=True)
    _atomic_save_npz(
        output,
        {
            "current_frame_ids": current_frame_ids,
            "history_frame_ids": history_frame_ids,
            "history_counts": history_counts,
            "history_rel_poses": history_rel_poses,
        },
    )
    manifest = {
        "schema": CACHE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_key": clip_key,
        "split": entry["split"],
        "source_clip": str(clip_dir),
        "scene_id": meta.get("scene_id", clip_key.split("/")[0]),
        "episode_id": meta.get("episode_id"),
        "frame_count": frame_count,
        "query_rows": query_rows,
        "current_frame_start": int(current_frame_ids[0]),
        "current_frame_end_inclusive": int(current_frame_ids[-1]),
        "num_history": num_history,
        "min_history": min_history,
        "history_sampling": (
            "arange(0,current) when current<=K else "
            "linspace(0,current-1,K,dtype=int64)"
        ),
        "pose_convention": POSE_CONVENTION,
        "history_pose_convention": HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "source_pose_provider": "amb3r_vo_da3",
        "amb3r_source_c2w": "opencv",
        "conversion": (
            "opencv_c2w_to_habitat_c2w_then_compute_history_rel_poses"
        ),
        "translation_scale": 1.0,
        "per_episode_gt_scale_used": False,
        "gt_pose_read_by_exporter": False,
        "causal": True,
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame": False,
        "query_every_frame_from_min_history": False,
        "query_timing": "only_at_official_map_update_endpoints_plus_final",
        "snapshot_timing": (
            "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
        ),
        "future_pose_revisions_used": False,
        "forced_final_tail": forced_final_tail,
        "final_tail_length": final_tail_length,
        "endpoint_kind_counts": endpoint_kind_counts,
        "warmup_provider": "not_used",
        "warmup_rows": 0,
        "stateful_provider_start_frame": map_init_window - 1,
        "provider_phase_counts": dict(sorted(phase_counts.items())),
        "map_init_window": map_init_window,
        "map_every": map_every,
        "resolution_wh": [resolution[0], resolution[1]],
        "da3_checkpoint_path": str(checkpoint_path),
        "checkpoint_digest_enforced": False,
        "inference_seconds": inference_seconds,
    }
    # The sidecar is the commit marker and is always renamed last.
    atomic_write_json(manifest_output, manifest)
    return validate_clip_cache(
        output,
        expected_clip_key=clip_key,
        expected_frame_count=frame_count,
        num_history=num_history,
        min_history=min_history,
        map_init_window=map_init_window,
        map_every=map_every,
    ) | {"inference_seconds": inference_seconds, "status": "generated"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amb3r-root", required=True)
    parser.add_argument("--da3-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--map-init-window", type=int, default=20)
    parser.add_argument("--map-every", type=int, default=8)
    parser.add_argument("--resolution", nargs=2, type=int, default=(518, 392))
    parser.add_argument("--clip-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--allowed-root", default="/mnt/afs/lixiaoou/intern/fjl")
    args = parser.parse_args()

    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=True)
    plan_path = _under(Path(args.plan), allowed_root)
    repo = _under(Path(args.repo), allowed_root)
    amb3r_root = _under(Path(args.amb3r_root), allowed_root)
    checkpoint = _under(Path(args.da3_checkpoint), allowed_root)
    if not repo.is_dir() or not amb3r_root.is_dir() or not checkpoint.exists():
        raise FileNotFoundError("repo, amb3r-root, or DA3 checkpoint is missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported plan schema: {plan.get('schema')}")
    cache_root = _under(Path(plan["cache_root"]), allowed_root)
    num_history = int(plan["num_history"])
    min_history = int(plan["min_history"])
    plan_map_init_window = int(plan["map_init_window"])
    plan_map_every = int(plan["map_every"])
    if args.map_init_window != plan_map_init_window:
        raise ValueError(
            "--map-init-window must match the cache plan: "
            f"argument={args.map_init_window}, plan={plan_map_init_window}"
        )
    if args.map_every != plan_map_every:
        raise ValueError(
            "--map-every must match the cache plan: "
            f"argument={args.map_every}, plan={plan_map_every}"
        )
    if float(plan.get("translation_scale", -1)) != 1.0:
        raise ValueError("Only native AMB3R translation_scale=1.0 is supported")
    if not 0 <= args.shard_id < int(plan["num_shards"]):
        raise ValueError(f"Invalid shard-id {args.shard_id}")
    shard = plan["shards"][args.shard_id]
    if int(shard["shard_id"]) != args.shard_id:
        raise ValueError("Plan shard order is inconsistent")

    # Invalidate a prior successful run before importing or loading the heavy
    # model. Otherwise an import/model-load failure can leave a stale ready
    # marker that incorrectly tells orchestration this shard is complete.
    control_root = cache_root / "_control"
    control_root.mkdir(parents=True, exist_ok=True)
    shard_ready = control_root / f"shard_{args.shard_id:02d}.ready.json"
    shard_ready.unlink(missing_ok=True)
    failures_path = control_root / f"shard_{args.shard_id:02d}.failures.jsonl"

    sys.path[:0] = [str(repo), str(amb3r_root), str(amb3r_root / "thirdparty")]
    from amb3r.model_zoo import load_model
    from src.vo.online_amb3r import OnlineAMB3RSession, StatefulAMB3RBackend

    print(
        json.dumps(
            {
                "event": "model_load_start",
                "shard_id": args.shard_id,
                "device": args.device,
                "clips": len(shard["clips"]),
            }
        ),
        flush=True,
    )
    load_started = time.monotonic()
    model = load_model("da3", ckpt_path=str(checkpoint))
    backend = StatefulAMB3RBackend(
        model,
        cfg_path=amb3r_root / "slam" / "slam_config.yaml",
        device=args.device,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
    )
    session = OnlineAMB3RSession(
        backend,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
        max_history=num_history,
        resolution=tuple(args.resolution),
    )
    print(
        json.dumps(
            {
                "event": "model_load_complete",
                "shard_id": args.shard_id,
                "seconds": time.monotonic() - load_started,
            }
        ),
        flush=True,
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shard_started = time.monotonic()
    for index, entry in enumerate(shard["clips"], start=1):
        clip_key = str(entry["clip_key"])
        output = cache_path_for(cache_root, clip_key)
        try:
            result = validate_clip_cache(
                output,
                expected_clip_key=clip_key,
                expected_frame_count=int(entry["frame_count"]),
                num_history=num_history,
                min_history=min_history,
                map_init_window=plan_map_init_window,
                map_every=plan_map_every,
            )
            result["status"] = "resumed_existing"
            completed.append(result)
        except Exception:
            last_error: BaseException | None = None
            for attempt in range(1, max(1, args.clip_retries) + 1):
                try:
                    result = _export_clip(
                        entry=entry,
                        cache_root=cache_root,
                        session=session,
                        num_history=num_history,
                        min_history=min_history,
                        map_init_window=args.map_init_window,
                        map_every=args.map_every,
                        resolution=tuple(args.resolution),
                        checkpoint_path=checkpoint,
                    )
                    result["attempt"] = attempt
                    completed.append(result)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        json.dumps(
                            {
                                "event": "clip_attempt_failed",
                                "shard_id": args.shard_id,
                                "clip_key": clip_key,
                                "attempt": attempt,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        ),
                        flush=True,
                    )
                    traceback.print_exc()
                    gc.collect()
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            if last_error is not None:
                failure = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "shard_id": args.shard_id,
                    "clip_key": clip_key,
                    "error": f"{type(last_error).__name__}: {last_error}",
                }
                failures.append(failure)
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                    handle.flush()
        if index % max(1, args.progress_every) == 0 or index == len(shard["clips"]):
            elapsed = time.monotonic() - shard_started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "shard_id": args.shard_id,
                        "done": index,
                        "total": len(shard["clips"]),
                        "valid": len(completed),
                        "failures": len(failures),
                        "elapsed_seconds": elapsed,
                        "clips_per_hour": index * 3600.0 / max(elapsed, 1e-6),
                    }
                ),
                flush=True,
            )

    # Revalidate every assigned clip before exposing the shard-ready marker.
    verified = []
    for entry in shard["clips"]:
        try:
            verified.append(
                validate_clip_cache(
                    cache_path_for(cache_root, entry["clip_key"]),
                    expected_clip_key=entry["clip_key"],
                    expected_frame_count=int(entry["frame_count"]),
                    num_history=num_history,
                    min_history=min_history,
                    map_init_window=plan_map_init_window,
                    map_every=plan_map_every,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "shard_id": args.shard_id,
                    "clip_key": entry["clip_key"],
                    "error": f"post_validation: {type(exc).__name__}: {exc}",
                }
            )
    if failures or len(verified) != len(shard["clips"]):
        print(
            json.dumps(
                {
                    "event": "shard_incomplete",
                    "shard_id": args.shard_id,
                    "verified": len(verified),
                    "assigned": len(shard["clips"]),
                    "failures": len(failures),
                }
            ),
            flush=True,
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
        "num_history": num_history,
        "min_history": min_history,
        "translation_scale": 1.0,
        "pose_convention": POSE_CONVENTION,
        "history_pose_convention": HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame": False,
        "query_every_frame_from_min_history": False,
        "map_init_window": plan_map_init_window,
        "map_every": plan_map_every,
        "snapshot_timing": (
            "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
        ),
        "future_pose_revisions_used": False,
        "causal": True,
        "elapsed_seconds": time.monotonic() - shard_started,
    }
    atomic_write_json(shard_ready, ready)
    print(json.dumps({"event": "shard_ready", **ready}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
