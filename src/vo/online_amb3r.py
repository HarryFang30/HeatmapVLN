"""Causal, session-scoped AMB3R-VO pose provider.

The public :class:`OnlineAMB3RSession` API is intentionally small enough to
sit behind an RPC boundary: reset one episode, ingest every continuous front
RGB frame, and query the relative pose tokens for the heatmap head's selected
history frame IDs.  It never accepts or stores Habitat/GT poses.

Frames before map initialization are explicitly unavailable: callers must
keep the native navigation path and must not inject provisional pose control.
At ``map_init_window`` frames the official AMB3R-VO map is initialized once.
Subsequent frames are committed exactly once in incremental tails:
automatically every ``map_every`` frames, or earlier when a query needs the
newest pose.  The mapped path therefore never reruns the full episode prefix.

This module contains no file/checkpoint hashes and no synchronization locks.
An RPC wrapper must serialize calls for a session (for example with one gRPC
worker) because both AMB3R's map and this state machine are mutable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.vo.amb3r_pose import history_rel_poses_from_amb3r


class IncrementalAMB3RBackend(Protocol):
    """Small backend contract used by the online state machine.

    Implementations own AMB3R's mutable map.  ``map_increment`` receives the
    frame store for random access to active keyframes, but its model forward
    must contain only active keyframes plus ``[start_index, end_index]``.
    """

    def reset(self, *, max_frames: int) -> None: ...

    def predict_direct(self, frames_rgb: np.ndarray) -> np.ndarray: ...

    def initialize(self, frames_rgb: np.ndarray) -> np.ndarray: ...

    def map_increment(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        start_index: int,
        end_index: int,
    ) -> np.ndarray: ...

    def poses(self, *, frame_count: int) -> np.ndarray: ...


@dataclass(frozen=True)
class OnlinePoseQuery:
    """Result returned by :meth:`OnlineAMB3RSession.query`."""

    session_id: str
    current_frame_id: int
    history_frame_ids: tuple[int, ...]
    history_rel_poses: np.ndarray
    ready: bool
    provider_phase: str
    frame_count: int
    trajectory_revision: int
    last_mapped_frame_id: int | None

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-safe, non-privileged RPC payload."""

        return {
            "schema": "heatmapvln-amb3r-online-query-v1",
            "session_id": self.session_id,
            "current_frame_id": self.current_frame_id,
            "history_frame_ids": list(self.history_frame_ids),
            "history_rel_poses": self.history_rel_poses.tolist(),
            "ready": self.ready,
            "provider_phase": self.provider_phase,
            "frame_count": self.frame_count,
            "trajectory_revision": self.trajectory_revision,
            "last_mapped_frame_id": self.last_mapped_frame_id,
            "pose_provider": "amb3r_vo_da3",
        }


def _validate_pose_prefix(value: Any, expected_frames: int) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float32)
    expected_shape = (int(expected_frames), 4, 4)
    if poses.shape != expected_shape:
        raise RuntimeError(
            f"AMB3R backend returned poses with shape {poses.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(poses).all():
        raise RuntimeError("AMB3R backend returned non-finite poses")
    expected_bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        poses[:, 3, :].shape,
    )
    if not np.allclose(poses[:, 3, :], expected_bottom, rtol=0.0, atol=1e-3):
        raise RuntimeError("AMB3R backend returned invalid homogeneous poses")
    return np.ascontiguousarray(poses)


class StatefulAMB3RBackend:
    """Adapter around the released ``slam.pipeline.AMB3R_VO`` backend.

    Imports of AMB3R and Torch are deliberately lazy, so pure state-machine
    tests do not require the external repository or a GPU.
    """

    def __init__(
        self,
        model: Any,
        *,
        cfg_path: str | Path,
        device: str = "cuda:0",
        map_init_window: int = 20,
        map_every: int = 8,
    ) -> None:
        if int(map_init_window) < 2:
            raise ValueError("map_init_window must be at least two")
        if int(map_every) < 1:
            raise ValueError("map_every must be positive")

        from slam.pipeline import AMB3R_VO

        self.device = str(device)
        self.map_init_window = int(map_init_window)
        self.map_every = int(map_every)
        self.pipeline = AMB3R_VO(model, cfg_path=str(Path(cfg_path).expanduser()))
        self.pipeline.cfg.device = self.device
        self.pipeline.cfg.map_init_window = self.map_init_window
        self.pipeline.cfg.map_every = self.map_every
        self.pipeline.model = self.pipeline.model.to(self.device).eval()
        if hasattr(self.pipeline.model, "device"):
            self.pipeline.model.device = self.device
        self._max_frames = 0
        self._initialized = False

    def reset(self, *, max_frames: int) -> None:
        self._max_frames = int(max_frames)
        if self._max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.pipeline.keyframe_memory = None
        self._initialized = False

    @staticmethod
    def _as_model_batch(frames_rgb: np.ndarray):
        import torch

        frames = np.asarray(frames_rgb)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                "frames_rgb must be uint8 [T,H,W,3], "
                f"got dtype={frames.dtype} shape={frames.shape}"
            )
        return (
            torch.from_numpy(np.ascontiguousarray(frames))
            .permute(0, 3, 1, 2)
            .float()
            .div_(127.5)
            .sub_(1.0)
            .unsqueeze(0)
        )

    def predict_direct(self, frames_rgb: np.ndarray) -> np.ndarray:
        import torch

        images = self._as_model_batch(frames_rgb).to(self.device)
        with torch.inference_mode():
            poses = self.pipeline.model.predict_camera_poses(images)[0]
        return poses.detach().float().cpu().numpy()

    def initialize(self, frames_rgb: np.ndarray) -> np.ndarray:
        from slam.memory import SLAMemory

        frames = np.asarray(frames_rgb)
        if len(frames) != self.map_init_window:
            raise ValueError(
                "Map initialization must receive exactly "
                f"{self.map_init_window} frames, got {len(frames)}"
            )
        _, height, width, _ = frames.shape
        self.pipeline.keyframe_memory = SLAMemory(
            self.pipeline.cfg,
            self._max_frames,
            int(height),
            int(width),
        )
        self.pipeline.initialize_map(
            {"images": self._as_model_batch(frames).to(self.device)},
            self.pipeline.cfg,
        )
        self._initialized = True
        return self.poses(frame_count=len(frames))

    def map_increment(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        if not self._initialized or self.pipeline.keyframe_memory is None:
            raise RuntimeError("AMB3R map has not been initialized")
        start, end = int(start_index), int(end_index)
        if not 0 <= start <= end < len(frames_rgb):
            raise IndexError(
                f"Invalid incremental AMB3R range [{start}, {end}] for "
                f"{len(frames_rgb)} frames"
            )
        active = (
            self.pipeline.keyframe_memory.cur_kf_idx.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        if active.size == 0 or np.any(active >= start):
            raise RuntimeError(
                "AMB3R active keyframes must be mapped frames preceding the new tail"
            )
        selected_indices = [int(index) for index in active] + list(
            range(start, end + 1)
        )
        selected = np.stack(
            [np.asarray(frames_rgb[index]) for index in selected_indices],
            axis=0,
        )
        # Only active keyframes plus the never-before-mapped tail enter the
        # expensive model forward. ``frames_rgb`` is merely the CPU frame store.
        self.pipeline.mapping(
            {
                "images": self._as_model_batch(selected).to(self.device),
                "start_idx": start,
                "end_idx": end,
            },
            self.pipeline.cfg,
        )
        return self.poses(frame_count=end + 1)

    def poses(self, *, frame_count: int) -> np.ndarray:
        if not self._initialized or self.pipeline.keyframe_memory is None:
            raise RuntimeError("AMB3R map has not been initialized")
        count = int(frame_count)
        return (
            self.pipeline.keyframe_memory.poses[:count]
            .detach()
            .float()
            .cpu()
            .numpy()
        )


class OnlineAMB3RSession:
    """One causal AMB3R trajectory, reset at every Habitat episode."""

    def __init__(
        self,
        backend: IncrementalAMB3RBackend,
        *,
        map_init_window: int = 20,
        map_every: int = 8,
        max_history: int = 8,
        resolution: tuple[int, int] = (518, 392),
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if int(map_init_window) < 2:
            raise ValueError("map_init_window must be at least two")
        if int(map_every) < 1:
            raise ValueError("map_every must be positive")
        if int(max_history) < 1:
            raise ValueError("max_history must be positive")
        if len(resolution) != 2 or any(int(value) <= 0 for value in resolution):
            raise ValueError(f"Invalid AMB3R resolution: {resolution}")
        self.backend = backend
        self.map_init_window = int(map_init_window)
        self.map_every = int(map_every)
        self.max_history = int(max_history)
        self.resolution = tuple(int(value) for value in resolution)
        self.frame_processor = frame_processor
        self._session_id: str | None = None
        self._max_frames = 0
        self._frames: list[np.ndarray] = []
        self._capture_steps: list[int] = []
        self._frame_shape: tuple[int, int, int] | None = None
        self._last_mapped_index: int | None = None
        self._trajectory_revision = 0

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def reset(self, session_id: str, *, max_frames: int) -> dict[str, Any]:
        identifier = str(session_id).strip()
        if not identifier:
            raise ValueError("session_id must be non-empty")
        limit = int(max_frames)
        if limit < 1:
            raise ValueError("max_frames must be positive")
        self.backend.reset(max_frames=limit)
        self._session_id = identifier
        self._max_frames = limit
        self._frames = []
        self._capture_steps = []
        self._frame_shape = None
        self._last_mapped_index = None
        self._trajectory_revision = 0
        return {
            "schema": "heatmapvln-amb3r-online-reset-v1",
            "session_id": identifier,
            "max_frames": limit,
            "map_init_window": self.map_init_window,
            "map_every": self.map_every,
        }

    def _require_session(self, session_id: str) -> str:
        identifier = str(session_id)
        if self._session_id is None:
            raise RuntimeError("AMB3R session has not been reset")
        if identifier != self._session_id:
            raise ValueError(
                f"Stale AMB3R session {identifier!r}; active session is {self._session_id!r}"
            )
        return identifier

    def _process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                "frame_rgb must be uint8 [H,W,3], "
                f"got dtype={frame.dtype} shape={frame.shape}"
            )
        if self.frame_processor is None:
            from src.vo.clip_io import center_crop_resize_for_amb3r

            processed = center_crop_resize_for_amb3r(
                frame[None],
                resolution=self.resolution,
            )[0]
        else:
            processed = np.asarray(self.frame_processor(frame))
        if (
            processed.dtype != np.uint8
            or processed.ndim != 3
            or processed.shape[-1] != 3
        ):
            raise ValueError(
                "frame_processor must return uint8 [H,W,3], "
                f"got dtype={processed.dtype} shape={processed.shape}"
            )
        shape = tuple(int(value) for value in processed.shape)
        if self._frame_shape is not None and shape != self._frame_shape:
            raise ValueError(
                f"Processed frame shape changed from {self._frame_shape} to {shape}"
            )
        self._frame_shape = shape
        return np.ascontiguousarray(processed).copy()

    def _frames_array(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("AMB3R session has no frames")
        return np.stack(self._frames, axis=0)

    def ingest(
        self,
        session_id: str,
        *,
        frame_id: int,
        frame_rgb: np.ndarray,
        capture_step: int,
    ) -> dict[str, Any]:
        identifier = self._require_session(session_id)
        if isinstance(frame_id, (bool, np.bool_)):
            raise TypeError("frame_id must be an integer, not bool")
        index = int(frame_id)
        expected = len(self._frames)
        if index != expected:
            raise ValueError(
                f"frame_id must be strictly contiguous: expected {expected}, got {index}"
            )
        if expected >= self._max_frames:
            raise RuntimeError(
                f"AMB3R session exceeded max_frames={self._max_frames}"
            )
        if isinstance(capture_step, (bool, np.bool_)):
            raise TypeError("capture_step must be an integer, not bool")
        step = int(capture_step)
        if step < 0:
            raise ValueError("capture_step must be non-negative")
        if self._capture_steps and step < self._capture_steps[-1]:
            raise ValueError("capture_step must be monotonic non-decreasing")

        self._frames.append(self._process_frame(frame_rgb))
        self._capture_steps.append(step)

        if len(self._frames) == self.map_init_window:
            poses = self.backend.initialize(self._frames_array())
            _validate_pose_prefix(poses, len(self._frames))
            self._last_mapped_index = len(self._frames) - 1
            self._trajectory_revision += 1
        elif (
            self._last_mapped_index is not None
            and len(self._frames) - 1 - self._last_mapped_index >= self.map_every
        ):
            self._flush_pending()

        return {
            "schema": "heatmapvln-amb3r-online-ingest-v1",
            "session_id": identifier,
            "frame_id": index,
            "capture_step": step,
            "frame_count": len(self._frames),
            "map_initialized": self._last_mapped_index is not None,
            "last_mapped_frame_id": self._last_mapped_index,
            "trajectory_revision": self._trajectory_revision,
        }

    def _flush_pending(self) -> np.ndarray:
        if self._last_mapped_index is None:
            raise RuntimeError("Cannot flush before AMB3R map initialization")
        newest = len(self._frames) - 1
        if newest <= self._last_mapped_index:
            return _validate_pose_prefix(
                self.backend.poses(frame_count=len(self._frames)),
                len(self._frames),
            )
        start = self._last_mapped_index + 1
        poses = self.backend.map_increment(
            self._frames,
            start_index=start,
            end_index=newest,
        )
        poses = _validate_pose_prefix(poses, len(self._frames))
        self._last_mapped_index = newest
        self._trajectory_revision += 1
        return poses

    def query(
        self,
        session_id: str,
        *,
        current_frame_id: int,
        history_frame_ids: Sequence[int],
        translation_scale: float = 1.0,
    ) -> OnlinePoseQuery:
        identifier = self._require_session(session_id)
        if not self._frames:
            raise RuntimeError("Cannot query before ingesting a frame")
        if isinstance(current_frame_id, (bool, np.bool_)):
            raise TypeError("current_frame_id must be an integer, not bool")
        current = int(current_frame_id)
        latest = len(self._frames) - 1
        if current != latest:
            raise ValueError(
                "Causal online queries must target the latest ingested frame: "
                f"latest={latest}, requested={current}"
            )
        history: list[int] = []
        for raw_index in history_frame_ids:
            if isinstance(raw_index, (bool, np.bool_)):
                raise TypeError("history_frame_ids must contain integers, not bool")
            index = int(raw_index)
            if index < 0 or index > current:
                raise ValueError(
                    f"History frame {index} must be in [0, {current}]"
                )
            history.append(index)
        if any(right < left for left, right in zip(history, history[1:])):
            raise ValueError(
                "history_frame_ids must be chronological (non-decreasing)"
            )
        if len(history) > self.max_history:
            raise ValueError(
                f"At most {self.max_history} history frames are supported, got {len(history)}"
            )
        if not np.isfinite(translation_scale) or float(translation_scale) <= 0.0:
            raise ValueError("translation_scale must be a finite positive scalar")

        if not history:
            return OnlinePoseQuery(
                session_id=identifier,
                current_frame_id=current,
                history_frame_ids=(),
                history_rel_poses=np.empty((0, 4), dtype=np.float32),
                ready=False,
                provider_phase="insufficient_history",
                frame_count=len(self._frames),
                trajectory_revision=self._trajectory_revision,
                last_mapped_frame_id=self._last_mapped_index,
            )

        if len(self._frames) < self.map_init_window:
            # The Heatmap Head is adapted to official AMB3R map endpoints,
            # not to repeatedly re-estimated pose-only prefixes.  Expose an
            # explicit unavailable state so deployment retains native
            # InternNav until the first map is initialized.
            return OnlinePoseQuery(
                session_id=identifier,
                current_frame_id=current,
                history_frame_ids=tuple(history),
                history_rel_poses=np.empty((0, 4), dtype=np.float32),
                ready=False,
                provider_phase="map_warmup",
                frame_count=len(self._frames),
                trajectory_revision=self._trajectory_revision,
                last_mapped_frame_id=None,
            )

        poses = self._flush_pending()
        phase = "stateful_backend"
        relative = history_rel_poses_from_amb3r(
            poses,
            history,
            current,
            translation_scale=float(translation_scale),
        )
        return OnlinePoseQuery(
            session_id=identifier,
            current_frame_id=current,
            history_frame_ids=tuple(history),
            history_rel_poses=relative,
            ready=True,
            provider_phase=phase,
            frame_count=len(self._frames),
            trajectory_revision=self._trajectory_revision,
            last_mapped_frame_id=self._last_mapped_index,
        )


def build_online_amb3r_session(
    model: Any,
    *,
    cfg_path: str | Path,
    device: str = "cuda:0",
    map_init_window: int = 20,
    map_every: int = 8,
    max_history: int = 8,
    resolution: tuple[int, int] = (518, 392),
) -> OnlineAMB3RSession:
    """Construct the released AMB3R backend and its online state machine."""

    backend = StatefulAMB3RBackend(
        model,
        cfg_path=cfg_path,
        device=device,
        map_init_window=map_init_window,
        map_every=map_every,
    )
    return OnlineAMB3RSession(
        backend,
        map_init_window=map_init_window,
        map_every=map_every,
        max_history=max_history,
        resolution=resolution,
    )


__all__ = [
    "IncrementalAMB3RBackend",
    "OnlineAMB3RSession",
    "OnlinePoseQuery",
    "StatefulAMB3RBackend",
    "build_online_amb3r_session",
]
