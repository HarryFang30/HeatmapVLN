"""Strict reader for offline causal AMB3R pose-token caches.

The cache is an input-sidecar only.  It never contains or computes heatmap
targets, and it deliberately has no fallback to Habitat/GT poses.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


AMB3R_POSE_CACHE_SCHEMA = "heatmapvln-amb3r-causal-endpoint-training-cache-v2"
AMB3R_POSE_CACHE_FILENAME = "amb3r_pose_cache.npz"
AMB3R_POSE_CONVENTION = "forward_m,left_m,cos_relative_yaw,sin_relative_yaw"
AMB3R_HISTORY_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)
AMB3R_POSE_PROVIDER = "amb3r_vo_cache"
AMB3R_ENDPOINT_ROW_POLICY = "official_map_update_endpoints_plus_final"
AMB3R_ENDPOINT_SNAPSHOT_TIMING = (
    "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
)


class AMB3RPoseCacheError(RuntimeError):
    """A cache is absent, malformed, non-causal, or identity-mismatched."""


class AMB3RPoseCache:
    """Lazy per-clip AMB3R pose cache with bounded worker-local LRU state."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_root: str | Path,
        num_history: int,
        min_history: int,
        max_cached_clips: int = 16,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.num_history = int(num_history)
        self.min_history = int(min_history)
        self.max_cached_clips = max(1, int(max_cached_clips))
        if not self.root.is_dir():
            raise AMB3RPoseCacheError(
                f"AMB3R pose cache root does not exist: {self.root}"
            )
        if self.num_history < 1 or self.min_history < 1:
            raise ValueError("num_history and min_history must be positive")
        self._clips: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def clip_key(self, clip_dir: str | Path) -> str:
        clip = Path(clip_dir).expanduser().resolve()
        try:
            relative = clip.relative_to(self.dataset_root)
        except ValueError as exc:
            raise AMB3RPoseCacheError(
                f"Clip {clip} is outside dataset root {self.dataset_root}"
            ) from exc
        parts = relative.parts
        # The endpoint-v2 exporter deliberately writes the same
        # ``<scene>/clip_*`` key for both supported dataset layouts.  The
        # random-walk corpus is flat, while R2R expert data may be rooted at
        # ``<root>/<train|val>/<scene>/clip_*``.  Normalize only those two
        # explicit split names; all other nesting remains an identity error.
        if len(parts) == 3 and parts[0] in {"train", "val"}:
            parts = parts[1:]
        if len(parts) != 2 or not parts[1].startswith("clip_"):
            raise AMB3RPoseCacheError(
                "AMB3R cache identity requires dataset-relative "
                f"<scene>/clip_*; got {relative.as_posix()!r}"
            )
        return "/".join(parts)

    def _load_clip(self, clip_key: str) -> dict[str, np.ndarray]:
        cached = self._clips.get(clip_key)
        if cached is not None:
            self._clips.move_to_end(clip_key)
            return cached

        cache_path = self.root / clip_key / AMB3R_POSE_CACHE_FILENAME
        manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
        if not cache_path.is_file() or not manifest_path.is_file():
            raise AMB3RPoseCacheError(
                "Missing required AMB3R pose cache sidecar: "
                f"npz={cache_path}, manifest={manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AMB3RPoseCacheError(
                f"Invalid AMB3R cache manifest: {manifest_path}: {exc}"
            ) from exc
        expected_manifest = {
            "schema": AMB3R_POSE_CACHE_SCHEMA,
            "clip_key": clip_key,
            "causal": True,
            "num_history": self.num_history,
            "min_history": self.min_history,
            "pose_convention": AMB3R_POSE_CONVENTION,
            "history_pose_convention": AMB3R_HISTORY_POSE_CONVENTION,
            "pose_provider": "amb3r_vo_da3",
            "per_episode_gt_scale_used": False,
            "gt_pose_read_by_exporter": False,
            "endpoint_only": True,
            "row_policy": AMB3R_ENDPOINT_ROW_POLICY,
            "query_only_at_map_endpoints": True,
            "query_every_frame_from_min_history": False,
            "query_every_frame": False,
            "snapshot_timing": AMB3R_ENDPOINT_SNAPSHOT_TIMING,
            "future_pose_revisions_used": False,
        }
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        translation_scale = manifest.get("translation_scale")
        if (
            isinstance(translation_scale, bool)
            or not isinstance(translation_scale, (int, float))
            or not np.isfinite(float(translation_scale))
            or not np.isclose(
                float(translation_scale), 1.0, rtol=0.0, atol=1e-8
            )
        ):
            mismatches["translation_scale"] = (translation_scale, 1.0)
        if mismatches:
            raise AMB3RPoseCacheError(
                f"AMB3R cache manifest mismatch for {clip_key}: {mismatches}"
            )

        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                required = {
                    "current_frame_ids",
                    "history_frame_ids",
                    "history_counts",
                    "history_rel_poses",
                }
                missing = required.difference(payload.files)
                if missing:
                    raise AMB3RPoseCacheError(
                        f"{cache_path} is missing arrays {sorted(missing)}"
                    )
                current = np.asarray(payload["current_frame_ids"])
                history = np.asarray(payload["history_frame_ids"])
                counts = np.asarray(payload["history_counts"])
                poses = np.asarray(payload["history_rel_poses"])
        except (OSError, ValueError) as exc:
            if isinstance(exc, AMB3RPoseCacheError):
                raise
            raise AMB3RPoseCacheError(
                f"Unable to read AMB3R pose cache {cache_path}: {exc}"
            ) from exc

        n_rows = int(current.shape[0]) if current.ndim == 1 else -1
        expected_shapes = {
            "current_frame_ids": (n_rows,),
            "history_frame_ids": (n_rows, self.num_history),
            "history_counts": (n_rows,),
            "history_rel_poses": (n_rows, self.num_history, 4),
        }
        arrays = {
            "current_frame_ids": current,
            "history_frame_ids": history,
            "history_counts": counts,
            "history_rel_poses": poses,
        }
        shape_errors = {
            key: (tuple(array.shape), shape)
            for key, (array, shape) in (
                (key, (arrays[key], expected_shapes[key]))
                for key in arrays
            )
            if tuple(array.shape) != shape
        }
        if n_rows < 1 or shape_errors:
            raise AMB3RPoseCacheError(
                f"AMB3R cache shape mismatch for {clip_key}: {shape_errors}"
            )
        if current.dtype != np.int64 or history.dtype != np.int64 or counts.dtype != np.int64:
            raise AMB3RPoseCacheError(
                "AMB3R cache ID/count arrays must be exactly int64: "
                f"current={current.dtype}, history={history.dtype}, counts={counts.dtype}"
            )
        if poses.dtype != np.float32:
            raise AMB3RPoseCacheError(
                f"AMB3R history_rel_poses must be exactly float32, got {poses.dtype}"
            )
        frame_count = manifest.get("frame_count")
        map_init_window = manifest.get("map_init_window")
        map_every = manifest.get("map_every")
        query_rows = manifest.get("query_rows")
        integer_fields = {
            "frame_count": frame_count,
            "map_init_window": map_init_window,
            "map_every": map_every,
            "query_rows": query_rows,
        }
        invalid_integer_fields = {
            key: value
            for key, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_integer_fields:
            raise AMB3RPoseCacheError(
                f"AMB3R cache manifest integer fields are invalid for "
                f"{clip_key}: {invalid_integer_fields}"
            )
        frame_count = int(frame_count)
        map_init_window = int(map_init_window)
        map_every = int(map_every)
        query_rows = int(query_rows)
        if frame_count < 1 or map_init_window < 1 or map_every < 1:
            raise AMB3RPoseCacheError(
                "AMB3R endpoint cache requires positive frame_count, "
                f"map_init_window, and map_every for {clip_key}"
            )
        first_endpoint = map_init_window - 1
        if first_endpoint < self.min_history or first_endpoint >= frame_count:
            raise AMB3RPoseCacheError(
                "AMB3R endpoint cache has no valid initialized endpoint: "
                f"clip={clip_key}, frame_count={frame_count}, "
                f"map_init_window={map_init_window}, min_history={self.min_history}"
            )
        expected_current_list = list(range(first_endpoint, frame_count, map_every))
        final_frame = frame_count - 1
        if expected_current_list[-1] != final_frame:
            expected_current_list.append(final_frame)
        expected_current = np.asarray(expected_current_list, dtype=np.int64)
        if query_rows != n_rows or query_rows != len(expected_current):
            raise AMB3RPoseCacheError(
                "AMB3R endpoint cache row-count mismatch: "
                f"clip={clip_key}, manifest={query_rows}, arrays={n_rows}, "
                f"policy_expected={len(expected_current)}"
            )
        if not np.array_equal(current, expected_current):
            raise AMB3RPoseCacheError(
                "current_frame_ids must exactly match the official causal "
                "map-update endpoints plus final frame for "
                f"{clip_key}: observed={current.tolist()}, "
                f"expected={expected_current.tolist()}"
            )
        if np.any((counts < self.min_history) | (counts > self.num_history)):
            raise AMB3RPoseCacheError(
                f"history_counts outside [{self.min_history},{self.num_history}] for {clip_key}"
            )
        for row, count in enumerate(counts.tolist()):
            valid_ids = history[row, :count]
            if np.any(valid_ids < 0) or np.any(valid_ids >= current[row]):
                raise AMB3RPoseCacheError(
                    f"Non-causal history IDs in {clip_key}, row={row}"
                )
            if count > 1 and np.any(np.diff(valid_ids) < 0):
                raise AMB3RPoseCacheError(
                    f"Non-chronological history IDs in {clip_key}, row={row}"
                )
            if count < self.num_history and np.any(history[row, count:] != -1):
                raise AMB3RPoseCacheError(
                    f"History-ID padding must be -1 in {clip_key}, row={row}"
                )
            expected_history = (
                np.arange(current[row], dtype=np.int64)
                if current[row] <= self.num_history
                else np.linspace(
                    0,
                    current[row] - 1,
                    self.num_history,
                    dtype=np.int64,
                )
            )
            if count != len(expected_history) or not np.array_equal(
                valid_ids, expected_history
            ):
                raise AMB3RPoseCacheError(
                    "AMB3R cache history IDs do not exactly match dataset "
                    f"sampling in {clip_key}, row={row}, "
                    f"observed={valid_ids.tolist()}, "
                    f"expected={expected_history.tolist()}"
                )
            valid_poses = poses[row, :count]
            if not np.isfinite(valid_poses).all():
                raise AMB3RPoseCacheError(
                    f"Non-finite AMB3R poses in {clip_key}, row={row}"
                )
            yaw_norm = np.linalg.norm(valid_poses[:, 2:4], axis=1)
            if not np.allclose(yaw_norm, 1.0, atol=2e-3, rtol=0.0):
                raise AMB3RPoseCacheError(
                    f"Non-unit AMB3R yaw encoding in {clip_key}, row={row}"
                )

        loaded = {
            "current_frame_ids": current,
            "history_frame_ids": history,
            "history_counts": counts,
            "history_rel_poses": poses,
            "_frame_count": np.asarray(frame_count, dtype=np.int64),
        }
        self._clips[clip_key] = loaded
        while len(self._clips) > self.max_cached_clips:
            self._clips.popitem(last=False)
        return loaded

    def current_frame_ids(
        self,
        clip_dir: str | Path,
        *,
        expected_frame_count: int | None = None,
    ) -> np.ndarray:
        """Return the only current-frame IDs eligible for dataset sampling.

        Loading is deliberately fail-closed: a missing/legacy cache cannot
        silently widen the index back to GT-pose sliding-window frames.
        """

        clip_key = self.clip_key(clip_dir)
        arrays = self._load_clip(clip_key)
        if expected_frame_count is not None:
            observed = int(arrays["_frame_count"])
            if observed != int(expected_frame_count):
                raise AMB3RPoseCacheError(
                    "AMB3R cache/data frame-count identity mismatch: "
                    f"clip={clip_key}, dataset={int(expected_frame_count)}, "
                    f"cache={observed}"
                )
        return np.array(arrays["current_frame_ids"], dtype=np.int64, copy=True)

    def lookup(
        self,
        clip_dir: str | Path,
        *,
        current_frame_id: int,
        history_frame_ids: np.ndarray,
    ) -> np.ndarray:
        """Return the exact cached ``[K,4]`` input or fail closed."""

        clip_key = self.clip_key(clip_dir)
        arrays = self._load_clip(clip_key)
        current = int(current_frame_id)
        history = np.asarray(history_frame_ids, dtype=np.int64).reshape(-1)
        rows = np.searchsorted(arrays["current_frame_ids"], current)
        if rows >= len(arrays["current_frame_ids"]) or int(
            arrays["current_frame_ids"][rows]
        ) != current:
            raise AMB3RPoseCacheError(
                f"Missing AMB3R cache row: clip={clip_key}, current={current}"
            )
        count = int(arrays["history_counts"][rows])
        cached_history = arrays["history_frame_ids"][rows, :count]
        if count != len(history) or not np.array_equal(cached_history, history):
            raise AMB3RPoseCacheError(
                "AMB3R cache history identity mismatch: "
                f"clip={clip_key}, current={current}, "
                f"dataset={history.tolist()}, cache={cached_history.tolist()}"
            )
        return np.array(
            arrays["history_rel_poses"][rows, :count],
            dtype=np.float32,
            copy=True,
        )
