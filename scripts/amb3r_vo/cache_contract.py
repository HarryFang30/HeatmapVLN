#!/usr/bin/env python3
"""Shared schema and validation for causal AMB3R endpoint caches.

One row is captured immediately after an official AMB3R map update.  Earlier
rows are never recomputed from a later trajectory revision.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA = "heatmapvln-amb3r-causal-endpoint-training-cache-v2"
SHARD_READY_SCHEMA = "heatmapvln-amb3r-endpoint-pose-cache-shard-ready-v2"
ROOT_READY_SCHEMA = "heatmapvln-amb3r-endpoint-pose-cache-ready-v2"
ROW_POLICY = "official_map_update_endpoints_plus_final"
POSE_CONVENTION = "forward_m,left_m,cos_relative_yaw,sin_relative_yaw"
HISTORY_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)
CACHE_FILENAME = "amb3r_pose_cache.npz"
REQUIRED_ARRAY_KEYS = {
    "current_frame_ids",
    "history_frame_ids",
    "history_counts",
    "history_rel_poses",
}


def history_indices(current: int, num_history: int) -> np.ndarray:
    """Match VLNSlidingWindowDataset._sample_history_indices(0, current, K)."""

    current = int(current)
    num_history = int(num_history)
    if current <= 0:
        return np.empty(0, dtype=np.int64)
    if current <= num_history:
        return np.arange(current, dtype=np.int64)
    return np.linspace(0, current - 1, num_history, dtype=np.int64)


def endpoint_frame_ids(
    frame_count: int,
    map_init_window: int = 20,
    map_every: int = 8,
) -> np.ndarray:
    """Return official causal map endpoints ``init-1, +every, ..., final``.

    The final frame is appended only when it is not already a regular update
    point.  A clip shorter than the initialization window cannot yield a
    stateful AMB3R pose and is rejected instead of silently using warmup poses.
    """

    count = int(frame_count)
    init = int(map_init_window)
    every = int(map_every)
    if init < 2:
        raise ValueError("map_init_window must be at least two")
    if every < 1:
        raise ValueError("map_every must be positive")
    if count < init:
        raise ValueError(
            f"frame_count={count} is smaller than map_init_window={init}"
        )
    endpoints = np.arange(init - 1, count, every, dtype=np.int64)
    final = count - 1
    if int(endpoints[-1]) != final:
        endpoints = np.concatenate(
            [endpoints, np.asarray([final], dtype=np.int64)]
        )
    return endpoints


def cache_path_for(cache_root: str | Path, clip_key: str) -> Path:
    clip_key = str(clip_key).strip("/")
    pieces = Path(clip_key).parts
    if len(pieces) != 2 or any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError(f"clip_key must be '<scene>/<clip>', got {clip_key!r}")
    return Path(cache_root).expanduser().resolve() / pieces[0] / pieces[1] / CACHE_FILENAME


def sidecar_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return Path(str(path) + ".json")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Commit JSON with a same-directory rename; no lock or digest is used."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_manifest_value(
    manifest: dict[str, Any], key: str, expected: Any
) -> None:
    observed = manifest.get(key)
    if observed != expected:
        raise ValueError(
            f"manifest[{key!r}]={observed!r}, expected {expected!r}"
        )


def validate_clip_cache(
    cache_path: str | Path,
    *,
    expected_clip_key: str | None = None,
    expected_frame_count: int | None = None,
    num_history: int = 8,
    min_history: int = 5,
    map_init_window: int = 20,
    map_every: int = 8,
) -> dict[str, Any]:
    """Fail closed on schema, coverage, padding, causality, and pose tokens."""

    path = Path(cache_path)
    manifest_path = sidecar_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_manifest_value(manifest, "schema", CACHE_SCHEMA)
    _require_manifest_value(manifest, "causal", True)
    _require_manifest_value(manifest, "num_history", int(num_history))
    _require_manifest_value(manifest, "min_history", int(min_history))
    _require_manifest_value(manifest, "pose_convention", POSE_CONVENTION)
    _require_manifest_value(
        manifest, "history_pose_convention", HISTORY_POSE_CONVENTION
    )
    _require_manifest_value(manifest, "translation_scale", 1.0)
    _require_manifest_value(manifest, "per_episode_gt_scale_used", False)
    _require_manifest_value(manifest, "gt_pose_read_by_exporter", False)
    _require_manifest_value(manifest, "pose_provider", "amb3r_vo_da3")
    _require_manifest_value(manifest, "endpoint_only", True)
    _require_manifest_value(manifest, "row_policy", ROW_POLICY)
    _require_manifest_value(manifest, "query_only_at_map_endpoints", True)
    _require_manifest_value(manifest, "query_every_frame", False)
    _require_manifest_value(manifest, "query_every_frame_from_min_history", False)
    _require_manifest_value(manifest, "map_init_window", int(map_init_window))
    _require_manifest_value(manifest, "map_every", int(map_every))
    _require_manifest_value(
        manifest,
        "snapshot_timing",
        "immediately_after_endpoint_mapping_before_ingesting_later_rgb",
    )
    _require_manifest_value(manifest, "future_pose_revisions_used", False)
    if expected_clip_key is not None:
        _require_manifest_value(manifest, "clip_key", expected_clip_key)

    frame_count = int(manifest.get("frame_count", -1))
    if expected_frame_count is not None and frame_count != int(expected_frame_count):
        raise ValueError(
            f"frame_count={frame_count}, expected {int(expected_frame_count)}"
        )
    expected_current = endpoint_frame_ids(
        frame_count,
        map_init_window=map_init_window,
        map_every=map_every,
    )
    expected_rows = len(expected_current)
    final_tail_length = int(
        (frame_count - int(map_init_window)) % int(map_every)
    )
    forced_final_tail = final_tail_length != 0
    periodic_count = int(
        (frame_count - int(map_init_window)) // int(map_every)
    )
    expected_kind_counts = {
        "initialization": 1,
        "periodic": periodic_count,
        "forced_final_tail": int(forced_final_tail),
    }
    _require_manifest_value(manifest, "forced_final_tail", forced_final_tail)
    _require_manifest_value(manifest, "final_tail_length", final_tail_length)
    _require_manifest_value(
        manifest, "endpoint_kind_counts", expected_kind_counts
    )
    _require_manifest_value(
        manifest, "provider_phase_counts", {"stateful_backend": expected_rows}
    )
    _require_manifest_value(
        manifest, "current_frame_start", int(expected_current[0])
    )
    _require_manifest_value(
        manifest, "current_frame_end_inclusive", int(expected_current[-1])
    )

    with np.load(path, allow_pickle=False) as payload:
        observed_keys = set(payload.files)
        if observed_keys != REQUIRED_ARRAY_KEYS:
            raise ValueError(
                f"cache keys={sorted(observed_keys)}, expected "
                f"{sorted(REQUIRED_ARRAY_KEYS)}"
            )
        current_ids = np.asarray(payload["current_frame_ids"])
        history_ids = np.asarray(payload["history_frame_ids"])
        history_counts = np.asarray(payload["history_counts"])
        rel_poses = np.asarray(payload["history_rel_poses"])

    if current_ids.dtype != np.int64 or current_ids.shape != (expected_rows,):
        raise ValueError(
            "current_frame_ids must be int64 [N], got "
            f"dtype={current_ids.dtype} shape={current_ids.shape}"
        )
    if history_ids.dtype != np.int64 or history_ids.shape != (
        expected_rows,
        int(num_history),
    ):
        raise ValueError(
            "history_frame_ids must be int64 [N,K], got "
            f"dtype={history_ids.dtype} shape={history_ids.shape}"
        )
    if history_counts.dtype != np.int64 or history_counts.shape != (expected_rows,):
        raise ValueError(
            "history_counts must be int64 [N], got "
            f"dtype={history_counts.dtype} shape={history_counts.shape}"
        )
    if rel_poses.dtype != np.float32 or rel_poses.shape != (
        expected_rows,
        int(num_history),
        4,
    ):
        raise ValueError(
            "history_rel_poses must be float32 [N,K,4], got "
            f"dtype={rel_poses.dtype} shape={rel_poses.shape}"
        )
    if not np.isfinite(rel_poses).all():
        raise ValueError("history_rel_poses contain NaN or infinite values")

    if not np.array_equal(current_ids, expected_current):
        raise ValueError(
            "current_frame_ids do not exactly match official map-update "
            "endpoints plus the final frame"
        )
    for row, current in enumerate(expected_current.tolist()):
        expected_history = history_indices(current, num_history)
        count = len(expected_history)
        if int(history_counts[row]) != count:
            raise ValueError(
                f"row {row} history_count={history_counts[row]}, expected {count}"
            )
        if not np.array_equal(history_ids[row, :count], expected_history):
            raise ValueError(f"row {row} history IDs do not match dataset sampling")
        if np.any(history_ids[row, :count] >= current):
            raise ValueError(f"row {row} contains a non-causal history frame")
        if not np.all(history_ids[row, count:] == -1):
            raise ValueError(f"row {row} frame-ID padding must be -1")
        if not np.all(rel_poses[row, count:] == 0.0):
            raise ValueError(f"row {row} pose padding must be exactly zero")
        yaw_norm = np.linalg.norm(rel_poses[row, :count, 2:4], axis=-1)
        if not np.allclose(yaw_norm, 1.0, rtol=0.0, atol=2e-3):
            raise ValueError(f"row {row} has invalid cosine/sine yaw tokens")

    _require_manifest_value(manifest, "query_rows", expected_rows)
    return {
        "clip_key": manifest["clip_key"],
        "frame_count": frame_count,
        "query_rows": expected_rows,
        "first_endpoint": int(expected_current[0]),
        "last_endpoint": int(expected_current[-1]),
        "forced_final_tail": forced_final_tail,
        "final_tail_length": final_tail_length,
        "cache_path": str(path),
        "manifest_path": str(manifest_path),
    }


__all__ = [
    "CACHE_FILENAME",
    "CACHE_SCHEMA",
    "HISTORY_POSE_CONVENTION",
    "POSE_CONVENTION",
    "REQUIRED_ARRAY_KEYS",
    "ROW_POLICY",
    "ROOT_READY_SCHEMA",
    "SHARD_READY_SCHEMA",
    "atomic_write_json",
    "cache_path_for",
    "endpoint_frame_ids",
    "history_indices",
    "sidecar_path",
    "validate_clip_cache",
]
