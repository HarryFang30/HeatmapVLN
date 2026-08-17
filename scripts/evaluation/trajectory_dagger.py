"""Trajectory DAgger collection primitives for heatmap-conditioned NextDiT.

The module is intentionally independent from r2r_val_unseen. It records exact
learner observations and oracle trajectory targets, never predicted heatmaps.
Training reconstructs heatmaps online from the stored RGB and relative poses.

Capacity uses decimal GB: the hard logical-file limit is exactly 300 GB and
episodes are committed only while the collection remains at or below 295 GB.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


def _load_trajectory_utils_module():
    """Load the pure NumPy utility without executing src.data package imports."""
    path = Path(__file__).resolve().parents[2] / "src" / "data" / "trajectory_utils.py"
    spec = importlib.util.spec_from_file_location("_heatmapvln_trajectory_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trajectory utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAJECTORY_UTILS = _load_trajectory_utils_module()
get_trajectory_relative_to_frame = _TRAJECTORY_UTILS.get_trajectory_relative_to_frame
interpolate_and_resample_trajectory = _TRAJECTORY_UTILS.interpolate_and_resample_trajectory


VIEW_NAMES = ("front", "right", "back", "left")
COLLECTION_SCHEMA = "heatmapvln-trajectory-dagger-collection-v1"
SAMPLE_SCHEMA = "heatmapvln-trajectory-dagger-sample-v1"
COMMIT_SCHEMA = "heatmapvln-trajectory-dagger-episode-commit-v1"

HARD_CAPACITY_BYTES = 300_000_000_000
COMMIT_CEILING_BYTES = 295_000_000_000

_SAFE_EPISODE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_SAFE_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,99}$")


class CapacityExceededError(RuntimeError):
    """Raised before a collection write would violate its byte contract."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validated_pose(pose: Any, *, name: str = "pose") -> np.ndarray:
    array = np.asarray(pose, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{name} is not a homogeneous transform")
    return array.astype(np.float32)


def _validated_pose_sequence(poses: Any, *, name: str) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N,4,4], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def encode_rgb_to_jpeg(image: Any, *, quality: int = 75) -> bytes:
    """Encode PIL, NumPy, or Torch RGB input to JPEG bytes.

    Existing bytes are returned unchanged, allowing the evaluator to reuse the
    exact bytes sent over RPC.
    """
    if isinstance(quality, bool) or not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be an integer in [1, 100]")
    if isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
        if not payload:
            raise ValueError("JPEG payload is empty")
        return payload

    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        value = image
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"RGB image must have three dimensions, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise ValueError(f"RGB image must end in three channels, got {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("RGB image contains non-finite values")
            if array.size and float(array.max()) <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array, mode="RGB")

    output = io.BytesIO()
    pil_image.save(output, format="JPEG", quality=int(quality), optimize=False)
    payload = output.getvalue()
    if not payload:
        raise RuntimeError("PIL produced an empty JPEG payload")
    return payload


def encode_views(views: Mapping[str, Any], *, quality: int = 75) -> dict[str, bytes]:
    missing = [name for name in VIEW_NAMES if name not in views]
    extra = sorted(set(views) - set(VIEW_NAMES))
    if missing or extra:
        raise ValueError(f"Panorama views mismatch: missing={missing} extra={extra}")
    return {name: encode_rgb_to_jpeg(views[name], quality=quality) for name in VIEW_NAMES}


@dataclass(frozen=True)
class HistoryObservation:
    """One exact observation from the learner's executed prefix."""

    frame_id: int
    pose: np.ndarray
    view_jpegs: Mapping[str, bytes]
    primitive_step: int
    system2_call_index: int | None = None
    lookdown_jpeg: bytes | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or int(self.frame_id) < 0:
            raise ValueError("frame_id must be an integer >= 0")
        if isinstance(self.primitive_step, bool) or int(self.primitive_step) < 0:
            raise ValueError("primitive_step must be an integer >= 0")
        if self.system2_call_index is not None and (
            isinstance(self.system2_call_index, bool) or int(self.system2_call_index) < 0
        ):
            raise ValueError("system2_call_index must be None or an integer >= 0")
        pose = _validated_pose(self.pose)
        view_jpegs = dict(self.view_jpegs)
        if set(view_jpegs) != set(VIEW_NAMES):
            raise ValueError(f"view_jpegs must contain exactly {VIEW_NAMES}")
        for name, payload in view_jpegs.items():
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"view_jpegs[{name!r}] must be non-empty bytes")
        if self.lookdown_jpeg is not None and (
            not isinstance(self.lookdown_jpeg, bytes) or not self.lookdown_jpeg
        ):
            raise ValueError("lookdown_jpeg must be None or non-empty bytes")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "primitive_step", int(self.primitive_step))
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "view_jpegs", view_jpegs)


def encode_history_observation(
    *,
    frame_id: int,
    pose: Any,
    views: Mapping[str, Any],
    primitive_step: int,
    system2_call_index: int | None = None,
    lookdown_image: Any | None = None,
    jpeg_quality: int = 75,
) -> HistoryObservation:
    return HistoryObservation(
        frame_id=frame_id,
        pose=pose,
        view_jpegs=encode_views(views, quality=jpeg_quality),
        primitive_step=primitive_step,
        system2_call_index=system2_call_index,
        lookdown_jpeg=(
            encode_rgb_to_jpeg(lookdown_image, quality=jpeg_quality)
            if lookdown_image is not None
            else None
        ),
    )


def sample_history_indices(available: int, num_history: int) -> list[int]:
    """Match the evaluator's endpoint-preserving linspace history sampler."""
    if available <= 0 or num_history <= 0:
        return []
    if available <= num_history:
        return list(range(available))
    return np.unique(np.linspace(0, available - 1, num_history, dtype=np.int32)).tolist()


@dataclass(frozen=True)
class RouteObservation:
    offpath_m: float
    raw_progress_m: float
    progress_m: float
    progress_delta_m: float


class MonotonicRouteTracker:
    """Project positions onto a reference route without progress regression."""

    def __init__(
        self,
        reference_path: Sequence[Sequence[float]],
        *,
        max_advance_per_observation_m: float | None = 2.0,
        max_update_offpath_m: float | None = 1.0,
    ) -> None:
        points = np.asarray(reference_path, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (2, 3):
            raise ValueError("reference_path must have shape [N,2] or [N,3]")
        if not np.isfinite(points).all():
            raise ValueError("reference_path contains non-finite values")
        if points.shape[1] == 2:
            points = np.column_stack([points[:, 0], np.zeros(len(points)), points[:, 1]])
        if max_advance_per_observation_m is not None and (
            not math.isfinite(float(max_advance_per_observation_m))
            or float(max_advance_per_observation_m) <= 0.0
        ):
            raise ValueError("max_advance_per_observation_m must be positive or None")
        if max_update_offpath_m is not None and (
            not math.isfinite(float(max_update_offpath_m))
            or float(max_update_offpath_m) <= 0.0
        ):
            raise ValueError("max_update_offpath_m must be positive or None")
        self.points = points
        # R2R contains multi-floor trajectories whose XZ projections can
        # overlap. Full XYZ projection prevents a state on one floor from
        # jumping progress to a vertically separated route segment.
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        self.max_advance_per_observation_m = max_advance_per_observation_m
        self.max_update_offpath_m = max_update_offpath_m
        self._progress_m = 0.0
        self._observation_count = 0
        self.reference_fingerprint = sha256_bytes(points.astype("<f8").tobytes())

    @property
    def progress_m(self) -> float:
        return float(self._progress_m)

    @property
    def total_length_m(self) -> float:
        return float(self.cumulative[-1])

    def project(self, position: Sequence[float]) -> tuple[float, float]:
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.size not in (2, 3) or not np.isfinite(pos).all():
            raise ValueError("position must be a finite [x,z] or [x,y,z] vector")
        query = (
            np.asarray([pos[0], 0.0, pos[1]], dtype=np.float64)
            if pos.size == 2
            else pos
        )
        if len(self.points) == 1 or self.total_length_m <= 1e-12:
            return float(np.linalg.norm(query - self.points[0])), 0.0

        best_distance = float("inf")
        best_progress = 0.0
        for index in range(len(self.points) - 1):
            start = self.points[index]
            delta = self.points[index + 1] - start
            length_sq = float(np.dot(delta, delta))
            if length_sq <= 1e-12:
                continue
            fraction = float(np.dot(query - start, delta) / length_sq)
            fraction = min(1.0, max(0.0, fraction))
            projected = start + fraction * delta
            distance = float(np.linalg.norm(query - projected))
            progress = float(self.cumulative[index] + fraction * math.sqrt(length_sq))
            if distance < best_distance - 1e-12 or (
                abs(distance - best_distance) <= 1e-12 and progress < best_progress
            ):
                best_distance, best_progress = distance, progress
        return best_distance, best_progress

    def observe(
        self,
        position: Sequence[float],
        *,
        max_advance_m: float | None = None,
    ) -> RouteObservation:
        offpath, raw_progress = self.project(position)
        previous = self._progress_m
        if self.max_update_offpath_m is not None and offpath > self.max_update_offpath_m:
            progress = previous
        else:
            progress = max(previous, raw_progress)
            limit = self.max_advance_per_observation_m if max_advance_m is None else max_advance_m
            if limit is not None and self._observation_count > 0:
                if not math.isfinite(float(limit)) or float(limit) <= 0.0:
                    raise ValueError("max_advance_m must be positive or None")
                progress = min(progress, previous + float(limit))
        progress = min(progress, self.total_length_m)
        self._progress_m = progress
        self._observation_count += 1
        return RouteObservation(offpath, raw_progress, progress, progress - previous)

    def point_at_progress(self, progress_m: float) -> np.ndarray:
        progress = min(self.total_length_m, max(0.0, float(progress_m)))
        if len(self.points) == 1 or progress <= 0.0:
            return self.points[0].astype(np.float32).copy()
        if progress >= self.total_length_m:
            return self.points[-1].astype(np.float32).copy()
        index = int(np.searchsorted(self.cumulative, progress, side="right") - 1)
        index = min(max(index, 0), len(self.points) - 2)
        length = max(float(self.cumulative[index + 1] - self.cumulative[index]), 1e-12)
        fraction = (progress - float(self.cumulative[index])) / length
        return (self.points[index] + fraction * (self.points[index + 1] - self.points[index])).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "reference_fingerprint": self.reference_fingerprint,
            "progress_m": self.progress_m,
            "observation_count": self._observation_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("reference_fingerprint") != self.reference_fingerprint:
            raise ValueError("Route tracker reference fingerprint mismatch")
        progress = float(state.get("progress_m", -1.0))
        count = int(state.get("observation_count", -1))
        if not 0.0 <= progress <= self.total_length_m or count < 0:
            raise ValueError("Invalid route tracker resume state")
        self._progress_m = progress
        self._observation_count = count


def poses_to_nextdit_target(
    future_poses: Sequence[Any] | np.ndarray,
    *,
    predict_horizon: int = 32,
    action_scale: float = 4.0,
    camera_forward_axis: str = "-z",
) -> tuple[np.ndarray, float]:
    """Convert world poses to the exact trajectory_dataset target convention."""
    if isinstance(predict_horizon, bool) or int(predict_horizon) < 1:
        raise ValueError("predict_horizon must be an integer >= 1")
    if not math.isfinite(float(action_scale)) or float(action_scale) <= 0.0:
        raise ValueError("action_scale must be positive")
    poses = _validated_pose_sequence(future_poses, name="future_poses")
    if len(poses) == 0:
        raise ValueError("future_poses may not be empty")
    horizon = int(predict_horizon)
    if len(poses) < 2:
        return np.zeros((horizon, 3), dtype=np.float32), 1.0
    try:
        relative = get_trajectory_relative_to_frame(
            poses,
            camera_deg=0.0,
            camera_forward_axis=camera_forward_axis,
        )
        _, target = interpolate_and_resample_trajectory(
            relative,
            predict_step_num=horizon,
            action_scale=float(action_scale),
        )
    except (ValueError, np.linalg.LinAlgError, IndexError):
        return np.zeros((horizon, 3), dtype=np.float32), 0.0
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (horizon, 3) or not np.isfinite(target).all():
        return np.zeros((horizon, 3), dtype=np.float32), 0.0
    return target, 1.0


class ShadowOracleBackend(Protocol):
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def reset(self, pose: np.ndarray) -> None: ...
    def get_pose(self) -> np.ndarray: ...
    def next_action(self, goal_position: np.ndarray) -> int | None: ...
    def step(self, action: int) -> np.ndarray | None: ...


class HabitatShadowBackend:
    """Save, shadow-step, restore, and verify one existing Habitat simulator."""

    def __init__(
        self,
        simulator: Any,
        *,
        agent_id: int = 0,
        follower: Any | None = None,
        goal_radius: float = 0.25,
        restore_atol: float = 1e-5,
    ) -> None:
        self.simulator = simulator
        self.agent_id = int(agent_id)
        self.restore_atol = float(restore_atol)
        if follower is None:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            follower = ShortestPathFollower(
                simulator,
                goal_radius=float(goal_radius),
                return_one_hot=False,
                stop_on_error=False,
            )
        self.follower = follower

    def _agent(self) -> Any:
        return self.simulator.get_agent(self.agent_id)

    @staticmethod
    def _state_pose(state: Any) -> np.ndarray:
        import quaternion

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
        pose[:3, 3] = np.asarray(state.position, dtype=np.float64)
        return _validated_pose(pose, name="Habitat AgentState pose")

    def snapshot(self) -> Any:
        return {
            "schema": "habitat-shadow-snapshot-v1",
            "agent_state": copy.deepcopy(self._agent().get_state()),
            "has_prev_sim_obs": hasattr(self.simulator, "_prev_sim_obs"),
            "prev_sim_obs": copy.deepcopy(getattr(self.simulator, "_prev_sim_obs", None)),
        }

    def _set_state(self, state: Any, *, reset_sensors: bool) -> None:
        try:
            self._agent().set_state(state, reset_sensors=reset_sensors)
        except TypeError:
            self._agent().set_state(state)

    def restore(self, snapshot: Any) -> None:
        if not isinstance(snapshot, Mapping) or snapshot.get("schema") != "habitat-shadow-snapshot-v1":
            raise TypeError("Invalid Habitat shadow snapshot")
        state = snapshot["agent_state"]
        expected = self._state_pose(state)
        self._set_state(copy.deepcopy(state), reset_sensors=False)
        if bool(snapshot.get("has_prev_sim_obs")):
            # HabitatSim.step mutates this wrapper-level collision/observation
            # cache even when the AgentState is later restored. Restore it as
            # part of the transaction so oracle actions cannot leak into the
            # learner rollout or its diagnostics.
            self.simulator._prev_sim_obs = copy.deepcopy(snapshot.get("prev_sim_obs"))
        actual = self.get_pose()
        if not np.allclose(actual, expected, atol=self.restore_atol, rtol=0.0):
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            raise RuntimeError(f"Habitat shadow restore verification failed: max_error={error:g}")

    def reset(self, pose: np.ndarray) -> None:
        import quaternion

        matrix = _validated_pose(pose, name="shadow reset pose")
        state = copy.deepcopy(self._agent().get_state())
        state.position = matrix[:3, 3].astype(np.float64)
        state.rotation = quaternion.from_rotation_matrix(matrix[:3, :3].astype(np.float64))
        self._set_state(state, reset_sensors=True)
        if not np.allclose(self.get_pose(), matrix, atol=self.restore_atol, rtol=0.0):
            raise RuntimeError("Habitat shadow reset pose verification failed")

    def get_pose(self) -> np.ndarray:
        return self._state_pose(self._agent().get_state())

    def next_action(self, goal_position: np.ndarray) -> int | None:
        action = self.follower.get_next_action(np.asarray(goal_position, dtype=np.float32))
        return None if action is None else int(action)

    def step(self, action: int) -> np.ndarray:
        self.simulator.step(int(action))
        return self.get_pose()

    def simulate_actions(
        self,
        actions: Sequence[int],
        *,
        start_pose: Any | None = None,
        stop_action: int = 0,
        max_actions: int = 128,
    ) -> np.ndarray:
        """Simulate a native action prefix and always restore the real state."""
        if len(actions) > int(max_actions):
            raise ValueError(f"Action sequence exceeds max_actions={max_actions}")
        snapshot = self.snapshot()
        try:
            if start_pose is not None:
                self.reset(_validated_pose(start_pose, name="native start pose"))
            poses = [self.get_pose().copy()]
            for action in actions:
                action = int(action)
                if action == int(stop_action):
                    break
                poses.append(self.step(action).copy())
            return np.stack(poses).astype(np.float32)
        finally:
            self.restore(snapshot)


@dataclass(frozen=True)
class OracleRelabelConfig:
    predict_horizon: int = 32
    action_scale: float = 4.0
    target_path_length_m: float = 3.2
    anchor_lookahead_m: float = 1.0
    anchor_spacing_m: float = 1.0
    goal_tolerance_m: float = 0.3
    max_actions: int = 128
    stop_action: int = 0
    camera_forward_axis: str = "-z"

    def __post_init__(self) -> None:
        if self.predict_horizon < 1 or self.max_actions < 1:
            raise ValueError("predict_horizon and max_actions must be positive")
        for name in (
            "action_scale",
            "target_path_length_m",
            "anchor_lookahead_m",
            "anchor_spacing_m",
            "goal_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class OracleRelabelResult:
    trajectory: np.ndarray
    trajectory_valid: float
    future_poses: np.ndarray
    actions: tuple[int, ...]
    oracle_kind: str
    terminal: bool
    route_progress_m: float
    travelled_m: float
    fallback_reason: str | None = None

    @property
    def valid(self) -> bool:
        return bool(self.trajectory_valid > 0.5)


def _xz_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(math.hypot(float(a[0] - b[0]), float(a[2] - b[2])))


def _route_anchor_targets(
    tracker: MonotonicRouteTracker,
    progress_m: float,
    goal_position: np.ndarray,
    config: OracleRelabelConfig,
) -> list[np.ndarray]:
    start = min(tracker.total_length_m, max(0.0, progress_m) + config.anchor_lookahead_m)
    targets: list[np.ndarray] = []
    cursor = start
    while cursor < tracker.total_length_m - 1e-6:
        targets.append(tracker.point_at_progress(cursor))
        cursor += config.anchor_spacing_m
    if tracker.total_length_m > 0.0:
        targets.append(tracker.point_at_progress(tracker.total_length_m))
    goal = np.asarray(goal_position, dtype=np.float32).reshape(3)
    if not targets or _xz_distance(targets[-1], goal) > config.goal_tolerance_m:
        targets.append(goal)
    return targets


def _run_shadow_targets(
    backend: ShadowOracleBackend,
    *,
    current_pose: np.ndarray,
    targets: Sequence[np.ndarray],
    final_goal: np.ndarray,
    config: OracleRelabelConfig,
) -> tuple[np.ndarray, tuple[int, ...], float, bool, str | None]:
    try:
        backend.reset(current_pose)
    except Exception as exc:
        return (
            np.stack([current_pose]),
            (),
            0.0,
            False,
            f"shadow reset {type(exc).__name__}: {exc}",
        )
    try:
        poses = [backend.get_pose().copy()]
    except Exception as exc:
        return (
            np.stack([current_pose]),
            (),
            0.0,
            False,
            f"shadow pose {type(exc).__name__}: {exc}",
        )
    actions: list[int] = []
    travelled = 0.0
    terminal = False
    failure: str | None = None

    for target_index, target in enumerate(targets):
        target = np.asarray(target, dtype=np.float32).reshape(3)
        while len(actions) < config.max_actions:
            try:
                current = backend.get_pose()
            except Exception as exc:
                failure = f"shadow pose {type(exc).__name__}: {exc}"
                break
            distance = _xz_distance(current[:3, 3], target)
            if distance <= config.goal_tolerance_m:
                if target_index == len(targets) - 1 and _xz_distance(target, final_goal) <= config.goal_tolerance_m:
                    terminal = True
                break
            try:
                action = backend.next_action(target)
            except Exception as exc:
                failure = f"oracle planner {type(exc).__name__}: {exc}"
                break
            if action is None:
                failure = f"oracle returned None for target {target_index}"
                break
            action = int(action)
            if action == int(config.stop_action):
                if distance <= config.goal_tolerance_m * 2.0:
                    if target_index == len(targets) - 1:
                        terminal = True
                    break
                failure = f"premature STOP for target {target_index} at {distance:.3f}m"
                break
            before = current[:3, 3].copy()
            try:
                stepped = backend.step(action)
                after_pose = backend.get_pose() if stepped is None else _validated_pose(stepped, name="shadow step pose")
            except Exception as exc:
                failure = f"shadow step {type(exc).__name__}: {exc}"
                break
            travelled += _xz_distance(before, after_pose[:3, 3])
            poses.append(after_pose.copy())
            actions.append(action)
            if travelled >= config.target_path_length_m:
                return np.stack(poses), tuple(actions), travelled, terminal, None
        if failure is not None:
            break
        if len(actions) >= config.max_actions:
            failure = "oracle action budget exhausted"
            break

    if failure is None and (terminal or travelled >= config.target_path_length_m):
        return np.stack(poses), tuple(actions), travelled, terminal, None
    if failure is None:
        failure = "oracle route ended before target path length"
    return np.stack(poses), tuple(actions), travelled, terminal, failure


def relabel_with_shadow_oracle(
    backend: ShadowOracleBackend,
    *,
    route_tracker: MonotonicRouteTracker,
    current_pose: Any,
    route_progress_m: float,
    goal_position: Sequence[float],
    config: OracleRelabelConfig | None = None,
) -> OracleRelabelResult:
    """Generate a route-aware suffix and restore the simulator in finally."""
    cfg = config or OracleRelabelConfig()
    pose = _validated_pose(current_pose, name="current_pose")
    goal = np.asarray(goal_position, dtype=np.float32).reshape(-1)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("goal_position must be a finite xyz vector")
    progress = min(route_tracker.total_length_m, max(0.0, float(route_progress_m)))
    snapshot = backend.snapshot()
    try:
        route_targets = _route_anchor_targets(route_tracker, progress, goal, cfg)
        future, actions, travelled, terminal, failure = _run_shadow_targets(
            backend,
            current_pose=pose,
            targets=route_targets,
            final_goal=goal,
            config=cfg,
        )
        kind = "route_recovery"
        fallback_reason = None
        if failure is not None:
            fallback_reason = failure
            future, actions, travelled, terminal, failure = _run_shadow_targets(
                backend,
                current_pose=pose,
                targets=[goal],
                final_goal=goal,
                config=cfg,
            )
            kind = "goal_fallback"
        trajectory, valid = poses_to_nextdit_target(
            future,
            predict_horizon=cfg.predict_horizon,
            action_scale=cfg.action_scale,
            camera_forward_axis=cfg.camera_forward_axis,
        )
        if failure is not None:
            valid = 0.0
        return OracleRelabelResult(
            trajectory=trajectory,
            trajectory_valid=float(valid),
            future_poses=future.astype(np.float32),
            actions=actions,
            oracle_kind=kind,
            terminal=bool(terminal),
            route_progress_m=progress,
            travelled_m=float(travelled),
            fallback_reason=fallback_reason if failure is None else f"{fallback_reason}; {failure}",
        )
    finally:
        backend.restore(snapshot)


@dataclass(frozen=True)
class CandidateThresholds:
    hard_offpath_m: float = 1.0
    normal_offpath_m: float = 0.5
    hard_disagreement_m: float = 0.5
    normal_disagreement_m: float = 0.25
    history_overlap_threshold: float = 0.5
    oracle_nonoverlap_threshold: float = 0.25
    history_radius_m: float = 0.5
    wrong_branch_progress_regression_m: float = 0.25
    wrong_branch_offpath_growth_m: float = 0.1
    turn_only_translation_m: float = 0.05
    normal_heading_disagreement_deg: float = 15.0
    hard_heading_disagreement_deg: float = 45.0


@dataclass(frozen=True)
class CandidateSignals:
    native_kind: str
    offpath_m: float
    route_progress_delta_m: float
    native_oracle_disagreement: float
    native_history_overlap: float = 0.0
    oracle_history_overlap: float = 0.0
    native_endpoint_offpath_m: float = 0.0
    native_route_progress_delta_m: float = 0.0
    native_oracle_heading_disagreement_deg: float = 0.0
    loop_detected: bool = False
    oscillation_detected: bool = False
    collision_or_stuck: bool = False
    wrong_branch: bool = False
    failure_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "offpath_m",
            "route_progress_delta_m",
            "native_oracle_disagreement",
            "native_history_overlap",
            "oracle_history_overlap",
            "native_endpoint_offpath_m",
            "native_route_progress_delta_m",
            "native_oracle_heading_disagreement_deg",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class CandidateSelection:
    bucket: str
    tags: tuple[str, ...]
    reasons: tuple[str, ...]
    hardness_score: float

    @property
    def accepted(self) -> bool:
        return self.bucket in {"dagger_normal", "dagger_hard"}


def _future_history_overlap(
    future_poses: np.ndarray,
    history_poses: np.ndarray,
    *,
    radius_m: float,
) -> float:
    if len(future_poses) <= 1 or len(history_poses) == 0:
        return 0.0
    future = future_poses[1:, :3, 3][:, [0, 2]]
    history = history_poses[:, :3, 3][:, [0, 2]]
    distances = np.linalg.norm(future[:, None, :] - history[None, :, :], axis=-1)
    return float(np.mean(np.min(distances, axis=1) <= float(radius_m)))


def _path_disagreement(native_poses: np.ndarray, oracle_poses: np.ndarray) -> float:
    """Compare native and oracle action-result poses over the same primitive-step horizon."""
    if len(native_poses) == 0 or len(oracle_poses) == 0:
        return float("inf")
    steps = min(16, len(native_poses) - 1, len(oracle_poses) - 1)
    if steps <= 0:
        return 0.0 if len(native_poses) == len(oracle_poses) == 1 else float("inf")
    native_xz = native_poses[1 : steps + 1, :3, 3][:, [0, 2]]
    oracle_xz = oracle_poses[1 : steps + 1, :3, 3][:, [0, 2]]
    mean_distance = float(np.linalg.norm(native_xz - oracle_xz, axis=1).mean())
    endpoint_distance = float(np.linalg.norm(native_xz[-1] - oracle_xz[-1]))
    return 0.5 * (mean_distance + endpoint_distance)


def _heading_disagreement_degrees(native_pose: np.ndarray, oracle_pose: np.ndarray) -> float:
    native_forward = -native_pose[:3, 2][[0, 2]]
    oracle_forward = -oracle_pose[:3, 2][[0, 2]]
    native_norm = float(np.linalg.norm(native_forward))
    oracle_norm = float(np.linalg.norm(oracle_forward))
    if native_norm <= 1e-8 or oracle_norm <= 1e-8:
        return 180.0
    cosine = float(
        np.clip(
            np.dot(native_forward, oracle_forward) / (native_norm * oracle_norm),
            -1.0,
            1.0,
        )
    )
    return float(math.degrees(math.acos(cosine)))


def build_candidate_signals(
    route_tracker: MonotonicRouteTracker,
    current_pose: Any,
    history_poses: Any,
    native_future_poses: Any,
    oracle: OracleRelabelResult,
    *,
    native_kind: str = "trajectory",
    route_progress_delta_m: float = 0.0,
    loop_detected: bool = False,
    oscillation_detected: bool = False,
    collision_or_stuck: bool = False,
    wrong_branch: bool = False,
    failure_tags: Sequence[str] = (),
    thresholds: CandidateThresholds | None = None,
) -> CandidateSignals:
    """Compute all geometry-derived candidate signals in one shared helper."""
    cfg = thresholds or CandidateThresholds()
    current = _validated_pose(current_pose, name="candidate current_pose")
    history = _validated_pose_sequence(history_poses, name="history_poses")
    native = _validated_pose_sequence(native_future_poses, name="native_future_poses")
    offpath_m, current_route_progress_m = route_tracker.project(current[:3, 3])
    if len(native) > 0:
        native_endpoint_offpath_m, native_endpoint_progress_m = route_tracker.project(
            native[-1, :3, 3]
        )
    else:
        native_endpoint_offpath_m = offpath_m
        native_endpoint_progress_m = current_route_progress_m
    native_route_progress_delta_m = native_endpoint_progress_m - current_route_progress_m
    offpath_recovery = bool(
        native_endpoint_offpath_m
        < offpath_m - cfg.wrong_branch_offpath_growth_m
    )
    geometric_wrong_branch = bool(
        (
            native_route_progress_delta_m < -cfg.wrong_branch_progress_regression_m
            and not offpath_recovery
        )
        or (
            native_endpoint_offpath_m >= cfg.hard_offpath_m
            and native_endpoint_offpath_m
            > offpath_m + cfg.wrong_branch_offpath_growth_m
        )
    )
    native_overlap = _future_history_overlap(native, history, radius_m=cfg.history_radius_m)
    oracle_future = _validated_pose_sequence(
        oracle.future_poses, name="oracle.future_poses"
    )
    oracle_overlap = _future_history_overlap(
        oracle_future,
        history,
        radius_m=cfg.history_radius_m,
    )
    disagreement = _path_disagreement(native, oracle_future)
    if not math.isfinite(disagreement):
        disagreement = max(cfg.hard_disagreement_m, 1.0)
    comparable_steps = min(len(native) - 1, len(oracle_future) - 1)
    heading_disagreement_deg = 0.0
    if comparable_steps > 0:
        native_prefix = native[: comparable_steps + 1, :3, 3][:, [0, 2]]
        oracle_prefix = oracle_future[: comparable_steps + 1, :3, 3][:, [0, 2]]
        native_travel_m = float(np.linalg.norm(np.diff(native_prefix, axis=0), axis=1).sum())
        oracle_travel_m = float(np.linalg.norm(np.diff(oracle_prefix, axis=0), axis=1).sum())
        if max(native_travel_m, oracle_travel_m) < cfg.turn_only_translation_m:
            heading_disagreement_deg = _heading_disagreement_degrees(
                native[comparable_steps], oracle_future[comparable_steps]
            )
    return CandidateSignals(
        native_kind=str(native_kind),
        offpath_m=float(offpath_m),
        route_progress_delta_m=float(route_progress_delta_m),
        native_oracle_disagreement=float(disagreement),
        native_history_overlap=native_overlap,
        oracle_history_overlap=oracle_overlap,
        native_endpoint_offpath_m=float(native_endpoint_offpath_m),
        native_route_progress_delta_m=float(native_route_progress_delta_m),
        native_oracle_heading_disagreement_deg=float(heading_disagreement_deg),
        loop_detected=bool(loop_detected),
        oscillation_detected=bool(oscillation_detected),
        collision_or_stuck=bool(collision_or_stuck),
        wrong_branch=bool(wrong_branch or geometric_wrong_branch),
        failure_tags=tuple(str(tag) for tag in failure_tags),
    )


def classify_candidate(
    signals: CandidateSignals,
    *,
    thresholds: CandidateThresholds | None = None,
) -> CandidateSelection:
    cfg = thresholds or CandidateThresholds()
    if str(signals.native_kind) != "trajectory":
        return CandidateSelection("discard", (), ("native decision bypasses NextDiT",), 0.0)

    tags = set(str(tag) for tag in signals.failure_tags if str(tag))
    if signals.wrong_branch:
        tags.add("wrong_branch")
    if signals.loop_detected:
        tags.add("loop")
    if signals.oscillation_detected:
        tags.add("oscillation")
    if signals.collision_or_stuck:
        tags.add("collision_or_stuck")
    if (
        signals.native_history_overlap >= cfg.history_overlap_threshold
        and signals.oracle_history_overlap <= cfg.oracle_nonoverlap_threshold
    ):
        tags.add("avoidable_revisit")
    if signals.oracle_history_overlap >= cfg.history_overlap_threshold:
        tags.add("necessary_backtrack")
    if signals.offpath_m >= cfg.hard_offpath_m:
        tags.add("off_route")
    if signals.native_oracle_disagreement >= cfg.hard_disagreement_m:
        tags.add("native_oracle_disagreement")
    if (
        signals.native_oracle_heading_disagreement_deg
        >= cfg.hard_heading_disagreement_deg
    ):
        tags.add("heading_disagreement")

    score = max(
        signals.offpath_m / max(cfg.hard_offpath_m, 1e-6),
        signals.native_oracle_disagreement / max(cfg.hard_disagreement_m, 1e-6),
        signals.native_oracle_heading_disagreement_deg
        / max(cfg.hard_heading_disagreement_deg, 1e-6),
        signals.native_history_overlap,
        signals.oracle_history_overlap,
    ) + 0.25 * len(tags)
    if tags:
        reason = "hard signals: " + ",".join(sorted(tags))
        return CandidateSelection("dagger_hard", tuple(sorted(tags)), (reason,), float(score))

    if (
        signals.offpath_m <= cfg.normal_offpath_m
        and signals.native_oracle_disagreement <= cfg.normal_disagreement_m
        and signals.native_oracle_heading_disagreement_deg
        <= cfg.normal_heading_disagreement_deg
        and signals.route_progress_delta_m >= -1e-6
    ):
        return CandidateSelection(
            "dagger_normal",
            (),
            ("on-route native/oracle agreement",),
            float(score),
        )
    return CandidateSelection("discard", (), ("ambiguous middle-band state",), float(score))


def logical_usage_bytes(root: str | Path) -> int:
    """Return exact logical regular-file bytes without following symlinks."""
    root_path = Path(root)
    if not root_path.exists():
        return 0
    total = 0
    for directory, dirnames, filenames in os.walk(root_path, followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise ValueError(f"Collection root contains a directory symlink: {base / name}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"Collection root contains a file symlink: {path}")
            total += path.stat().st_size
    return int(total)


@dataclass(frozen=True)
class CapacityGuard:
    root: Path
    hard_capacity_bytes: int = HARD_CAPACITY_BYTES
    commit_ceiling_bytes: int = COMMIT_CEILING_BYTES

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        hard = int(self.hard_capacity_bytes)
        ceiling = int(self.commit_ceiling_bytes)
        if hard <= 0 or ceiling <= 0 or ceiling >= hard:
            raise ValueError("Capacity contract requires 0 < commit ceiling < hard capacity")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "hard_capacity_bytes", hard)
        object.__setattr__(self, "commit_ceiling_bytes", ceiling)

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".capacity.lock").open("a+b") as handle:
            deadline = time.monotonic() + 300.0
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    break
                except (BlockingIOError, InterruptedError) as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring collection lock: {self.root / '.capacity.lock'}"
                        ) from exc
                    # Some AFS clients surface lock contention as EAGAIN even
                    # for a blocking flock. Retry explicitly on the same host.
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def ensure_can_commit(self, additional_bytes: int) -> int:
        additional = int(additional_bytes)
        if additional < 0:
            raise ValueError("additional_bytes must be non-negative")
        used = logical_usage_bytes(self.root)
        projected = used + additional
        if projected > self.hard_capacity_bytes:
            raise CapacityExceededError(
                f"Hard collection capacity exceeded: used={used} additional={additional} "
                f"projected={projected} hard={self.hard_capacity_bytes}"
            )
        if projected > self.commit_ceiling_bytes:
            raise CapacityExceededError(
                f"Episode commit ceiling exceeded: used={used} additional={additional} "
                f"projected={projected} ceiling={self.commit_ceiling_bytes}"
            )
        return projected

    def ensure_hard_capacity(self, additional_bytes: int) -> int:
        """Guard crash-safe maintenance writes against the absolute 300 GB cap."""
        additional = int(additional_bytes)
        if additional < 0:
            raise ValueError("additional_bytes must be non-negative")
        used = logical_usage_bytes(self.root)
        projected = used + additional
        if projected > self.hard_capacity_bytes:
            raise CapacityExceededError(
                f"Hard collection capacity exceeded: used={used} additional={additional} "
                f"projected={projected} hard={self.hard_capacity_bytes}"
            )
        return projected


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class CollectionState:
    root: Path
    manifest: Mapping[str, Any]
    fingerprint: str
    guard: CapacityGuard
    committed_episode_keys: frozenset[str]
    incomplete_staging: tuple[str, ...]


def _manifest_identity(
    contract: Mapping[str, Any],
    *,
    hard_capacity_bytes: int,
    commit_ceiling_bytes: int,
) -> tuple[dict[str, Any], str]:
    normalized_contract = json.loads(canonical_json_bytes(contract).decode("utf-8"))
    identity = {
        "schema": COLLECTION_SCHEMA,
        "contract": normalized_contract,
        "capacity": {
            "hard_capacity_bytes": int(hard_capacity_bytes),
            "commit_ceiling_bytes": int(commit_ceiling_bytes),
        },
    }
    return identity, sha256_bytes(canonical_json_bytes(identity))


def scan_committed_episodes(
    root: str | Path,
    *,
    manifest_fingerprint: str,
    verify_hashes: bool = False,
) -> dict[str, Mapping[str, Any]]:
    episodes_root = Path(root) / "episodes"
    if not episodes_root.exists():
        return {}
    commits: dict[str, Mapping[str, Any]] = {}
    for episode_dir in sorted(path for path in episodes_root.iterdir() if path.is_dir()):
        commit_path = episode_dir / "commit.json"
        tar_path = episode_dir / "episode.tar"
        if not commit_path.is_file() or not tar_path.is_file():
            raise RuntimeError(f"Incomplete committed episode directory: {episode_dir}")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        key = str(commit.get("episode_key") or "")
        if commit.get("schema") != COMMIT_SCHEMA or key != episode_dir.name:
            raise RuntimeError(f"Invalid episode commit marker: {commit_path}")
        if commit.get("manifest_fingerprint") != manifest_fingerprint:
            raise RuntimeError(f"Episode manifest fingerprint mismatch: {episode_dir}")
        if int(commit.get("tar_bytes", -1)) != tar_path.stat().st_size:
            raise RuntimeError(f"Episode tar size mismatch: {tar_path}")
        if verify_hashes and commit.get("tar_sha256") != sha256_file(tar_path):
            raise RuntimeError(f"Episode tar SHA256 mismatch: {tar_path}")
        if key in commits:
            raise RuntimeError(f"Duplicate committed episode key: {key}")
        commits[key] = commit
    return commits


def _reconcile_progress_ledger(
    root: Path,
    commits: Mapping[str, Mapping[str, Any]],
    guard: CapacityGuard,
) -> None:
    """Atomically rebuild the derived progress ledger from commit markers.

    Episode directories plus ``commit.json`` are authoritative. Reconstructing
    the ledger closes the crash window between the episode-directory rename and
    the append/fsync of ``collection_progress.jsonl``.
    """
    ledger_path = root / "collection_progress.jsonl"
    expected = b"".join(
        canonical_json_bytes(commits[key], newline=True) for key in sorted(commits)
    )
    if ledger_path.exists() and ledger_path.read_bytes() == expected:
        return
    if not expected and not ledger_path.exists():
        return

    guard.ensure_hard_capacity(len(expected))
    temporary = root / f".collection_progress.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        _write_new_file(temporary, expected)
        os.replace(temporary, ledger_path)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_collection(
    root: str | Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
    hard_capacity_bytes: int = HARD_CAPACITY_BYTES,
    commit_ceiling_bytes: int = COMMIT_CEILING_BYTES,
    verify_commits: bool = False,
) -> CollectionState:
    """Create or strictly resume a fingerprinted collection root."""
    root_path = Path(root).resolve()
    guard = CapacityGuard(root_path, hard_capacity_bytes, commit_ceiling_bytes)
    root_path.mkdir(parents=True, exist_ok=True)
    identity, fingerprint = _manifest_identity(
        contract,
        hard_capacity_bytes=hard_capacity_bytes,
        commit_ceiling_bytes=commit_ceiling_bytes,
    )
    manifest_path = root_path / "collection_manifest.json"
    with guard.locked():
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Collection manifest already exists: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != COLLECTION_SCHEMA:
                raise RuntimeError(f"Unsupported collection schema: {manifest.get('schema')!r}")
            if not isinstance(manifest.get("ready"), bool):
                raise RuntimeError("Collection manifest ready flag must be boolean")
            if manifest.get("fingerprint") != fingerprint:
                raise RuntimeError("Collection manifest fingerprint does not match resume contract")
            recorded = {name: manifest.get(name) for name in ("schema", "contract", "capacity")}
            if canonical_json_bytes(recorded) != canonical_json_bytes(identity):
                raise RuntimeError("Collection identity changed despite matching fingerprint")
        else:
            if resume:
                raise FileNotFoundError(f"Cannot resume without collection manifest: {manifest_path}")
            unexpected = [path for path in root_path.iterdir() if path.name != ".capacity.lock"]
            if unexpected:
                raise FileExistsError(f"Refusing non-empty collection root without manifest: {unexpected[:3]}")
            manifest = {
                **identity,
                "fingerprint": fingerprint,
                "created_at": _utc_now(),
                "ready": False,
            }
            payload = canonical_json_bytes(manifest, newline=True)
            guard.ensure_can_commit(len(payload))
            temporary = root_path / f".collection_manifest.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            _write_new_file(temporary, payload)
            os.replace(temporary, manifest_path)
            _fsync_directory(root_path)
        commits = scan_committed_episodes(
            root_path,
            manifest_fingerprint=fingerprint,
            verify_hashes=verify_commits,
        )
        if manifest["ready"]:
            expected_progress = b"".join(
                canonical_json_bytes(commits[key], newline=True)
                for key in sorted(commits)
            )
            progress_path = root_path / "collection_progress.jsonl"
            actual_progress = progress_path.read_bytes() if progress_path.exists() else b""
            if actual_progress != expected_progress:
                raise RuntimeError(
                    "Sealed collection progress ledger disagrees with commit markers; "
                    "refusing to repair immutable state"
                )
        else:
            _reconcile_progress_ledger(root_path, commits, guard)
    staging_root = root_path / ".staging"
    incomplete = tuple(sorted(path.name for path in staging_root.iterdir())) if staging_root.exists() else ()
    return CollectionState(
        root=root_path,
        manifest=manifest,
        fingerprint=fingerprint,
        guard=guard,
        committed_episode_keys=frozenset(commits),
        incomplete_staging=incomplete,
    )


@dataclass(frozen=True)
class EpisodeCommit:
    episode_key: str
    tar_path: Path
    tar_sha256: str
    tar_bytes: int
    sample_count: int
    frame_count: int
    already_committed: bool = False


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def _array_npy_bytes(value: Any, *, name: str) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"Array {name!r} may not use object dtype")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"Array {name!r} contains non-finite values")
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


class EpisodeTarRecorder:
    """Atomically persist one deduplicated episode package."""

    def __init__(self, state: CollectionState) -> None:
        self.state = state
        self.root = state.root
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_episode_key(episode_key: str) -> str:
        key = str(episode_key)
        if not _SAFE_EPISODE_KEY.fullmatch(key):
            raise ValueError(f"Unsafe episode key: {episode_key!r}")
        return key

    def _existing_commit(self, episode_key: str) -> EpisodeCommit | None:
        episode_dir = self.root / "episodes" / episode_key
        if not episode_dir.exists():
            return None
        commits = scan_committed_episodes(
            self.root,
            manifest_fingerprint=self.state.fingerprint,
            verify_hashes=False,
        )
        commit = commits.get(episode_key)
        if commit is None:
            raise RuntimeError(f"Episode destination exists without valid commit: {episode_dir}")
        return EpisodeCommit(
            episode_key=episode_key,
            tar_path=episode_dir / "episode.tar",
            tar_sha256=str(commit["tar_sha256"]),
            tar_bytes=int(commit["tar_bytes"]),
            sample_count=int(commit["sample_count"]),
            frame_count=int(commit["frame_count"]),
            already_committed=True,
        )

    def record_episode(
        self,
        *,
        episode_key: str,
        episode_metadata: Mapping[str, Any],
        observations: Sequence[HistoryObservation],
        samples: Sequence[Mapping[str, Any]],
        arrays: Mapping[str, Any] | None = None,
    ) -> EpisodeCommit:
        key = self.validate_episode_key(episode_key)
        if not samples:
            raise ValueError("Refusing to commit an episode without retained samples")

        observation_by_id: dict[int, HistoryObservation] = {}
        for observation in observations:
            if observation.frame_id in observation_by_id:
                raise ValueError(f"Duplicate observation frame_id: {observation.frame_id}")
            observation_by_id[observation.frame_id] = observation

        trajectories: list[np.ndarray] = []
        future_pose_chunks: list[np.ndarray] = []
        future_offsets = [0]
        normalized_samples: list[dict[str, Any]] = []
        sample_keys: set[str] = set()
        for index, raw_sample in enumerate(samples):
            sample = dict(raw_sample)
            sample_key = str(sample.get("key") or "")
            if not sample_key or sample_key in sample_keys:
                raise ValueError(f"Missing or duplicate sample key: {sample_key!r}")
            sample_keys.add(sample_key)
            if sample.get("native_kind") != "trajectory":
                raise ValueError(f"Sample {sample_key} bypasses NextDiT")
            if sample.get("source_type") not in {"dagger_normal", "dagger_hard"}:
                raise ValueError(f"Invalid source_type for {sample_key}")
            current_id = int(sample["current_frame_id"])
            history_ids = [int(value) for value in sample.get("history_frame_ids", [])]
            missing = [
                frame_id
                for frame_id in [current_id, *history_ids]
                if frame_id not in observation_by_id
            ]
            if missing:
                raise ValueError(f"Sample {sample_key} references missing frames: {missing}")
            mask = [int(value) for value in sample.get("history_valid_mask", [1] * len(history_ids))]
            ages = [int(value) for value in sample.get("history_age_steps", [])]
            if len(mask) != len(history_ids) or any(value not in (0, 1) for value in mask):
                raise ValueError(f"Invalid history_valid_mask for {sample_key}")
            if len(ages) != len(history_ids) or any(value < 0 for value in ages):
                raise ValueError(f"Invalid history_age_steps for {sample_key}")
            trajectory = np.asarray(sample.pop("trajectory"), dtype=np.float32)
            if trajectory.shape != (32, 3) or not np.isfinite(trajectory).all():
                raise ValueError(f"Sample {sample_key} trajectory must be finite [32,3]")
            trajectories.append(trajectory)
            future = _validated_pose_sequence(
                sample.pop("oracle_future_poses", np.empty((0, 4, 4))),
                name=f"{sample_key}.oracle_future_poses",
            )
            future_pose_chunks.append(future)
            future_offsets.append(future_offsets[-1] + len(future))
            sample.update(
                {
                    "schema": SAMPLE_SCHEMA,
                    "trajectory_index": index,
                    "future_pose_start": future_offsets[-2],
                    "future_pose_end": future_offsets[-1],
                    "history_valid_mask": mask,
                    "history_age_steps": ages,
                }
            )
            normalized_samples.append(sample)

        frame_rows: list[dict[str, Any]] = []
        tar_buffer = io.BytesIO()
        reserved_episode_fields = {"schema", "episode_key", "manifest_fingerprint"}
        episode_metadata_dict = dict(episode_metadata)
        collisions = reserved_episode_fields & set(episode_metadata_dict)
        if collisions:
            raise ValueError(
                f"Episode metadata overwrites reserved fields: {sorted(collisions)}"
            )
        with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            episode_payload = {
                **episode_metadata_dict,
                "schema": COLLECTION_SCHEMA,
                "episode_key": key,
                "manifest_fingerprint": self.state.fingerprint,
            }
            _tar_add_bytes(archive, "episode.json", canonical_json_bytes(episode_payload, newline=True))
            for frame_id, observation in sorted(observation_by_id.items()):
                view_paths: dict[str, str] = {}
                for view_name in VIEW_NAMES:
                    member = f"frames/{frame_id:06d}_{view_name}.jpg"
                    _tar_add_bytes(archive, member, observation.view_jpegs[view_name])
                    view_paths[view_name] = member
                lookdown_path = None
                if observation.lookdown_jpeg is not None:
                    lookdown_path = f"lookdown/{frame_id:06d}.jpg"
                    _tar_add_bytes(archive, lookdown_path, observation.lookdown_jpeg)
                frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "primitive_step": observation.primitive_step,
                        "system2_call_index": observation.system2_call_index,
                        "pose": observation.pose,
                        "views": view_paths,
                        "lookdown": lookdown_path,
                    }
                )
            _tar_add_bytes(
                archive,
                "frames.jsonl",
                b"".join(canonical_json_bytes(row, newline=True) for row in frame_rows),
            )
            _tar_add_bytes(
                archive,
                "samples.jsonl",
                b"".join(canonical_json_bytes(row, newline=True) for row in normalized_samples),
            )
            combined_future = (
                np.concatenate(future_pose_chunks, axis=0)
                if any(len(chunk) for chunk in future_pose_chunks)
                else np.empty((0, 4, 4), dtype=np.float32)
            )
            payload_arrays: dict[str, Any] = {
                "trajectories": np.stack(trajectories).astype(np.float32),
                "oracle_future_poses": combined_future,
                "oracle_future_offsets": np.asarray(future_offsets, dtype=np.int64),
            }
            for name, value in (arrays or {}).items():
                if not _SAFE_ARRAY_NAME.fullmatch(str(name)) or name in payload_arrays:
                    raise ValueError(f"Invalid or duplicate episode array name: {name!r}")
                payload_arrays[str(name)] = value
            for name, value in sorted(payload_arrays.items()):
                _tar_add_bytes(archive, f"arrays/{name}.npy", _array_npy_bytes(value, name=name))

        tar_payload = tar_buffer.getvalue()
        tar_sha = sha256_bytes(tar_payload)
        commit_payload = {
            "schema": COMMIT_SCHEMA,
            "episode_key": key,
            "manifest_fingerprint": self.state.fingerprint,
            "tar_file": "episode.tar",
            "tar_sha256": tar_sha,
            "tar_bytes": len(tar_payload),
            "sample_count": len(normalized_samples),
            "frame_count": len(frame_rows),
            "sample_keys": sorted(sample_keys),
            "created_at": _utc_now(),
        }
        commit_bytes = canonical_json_bytes(commit_payload, newline=True)
        progress_bytes = canonical_json_bytes(commit_payload, newline=True)
        additional = len(tar_payload) + len(commit_bytes) + len(progress_bytes)

        destination = self.root / "episodes" / key
        staging = self.root / ".staging" / f"{key}.{os.getpid()}.{uuid.uuid4().hex}.partial"
        with self.state.guard.locked():
            manifest_path = self.root / "collection_manifest.json"
            try:
                live_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot verify live collection manifest before commit: {manifest_path}"
                ) from exc
            if live_manifest.get("fingerprint") != self.state.fingerprint:
                raise RuntimeError("Live collection manifest fingerprint changed before commit")
            if not isinstance(live_manifest.get("ready"), bool):
                raise RuntimeError("Live collection manifest ready flag is not boolean")
            live_identity = {
                name: live_manifest.get(name) for name in ("schema", "contract", "capacity")
            }
            state_identity = {
                name: self.state.manifest.get(name)
                for name in ("schema", "contract", "capacity")
            }
            if canonical_json_bytes(live_identity) != canonical_json_bytes(state_identity):
                raise RuntimeError("Live collection manifest identity changed before commit")
            if live_manifest["ready"]:
                raise RuntimeError(
                    f"Collection is sealed and rejects episode commits: {self.root}"
                )
            existing = self._existing_commit(key)
            if existing is not None:
                if existing.tar_sha256 != tar_sha:
                    raise RuntimeError(
                        f"Stable episode key collision for {key!r}: "
                        f"committed={existing.tar_sha256} candidate={tar_sha}"
                    )
                return existing
            self.state.guard.ensure_can_commit(additional)
            staging.mkdir(parents=False, exist_ok=False)
            try:
                _write_new_file(staging / "episode.tar", tar_payload)
                _write_new_file(staging / "commit.json", commit_bytes)
                _fsync_directory(staging)
                os.replace(staging, destination)
                _fsync_directory(destination.parent)
                with (self.root / "collection_progress.jsonl").open("ab") as handle:
                    handle.write(progress_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(self.root)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise

        return EpisodeCommit(
            episode_key=key,
            tar_path=destination / "episode.tar",
            tar_sha256=tar_sha,
            tar_bytes=len(tar_payload),
            sample_count=len(normalized_samples),
            frame_count=len(frame_rows),
        )


@dataclass
class _PendingCandidate:
    selection: CandidateSelection
    sample: dict[str, Any]
    observations: tuple[HistoryObservation, ...]


class TrajectoryDaggerCollector:
    """Evaluator-facing begin, consider, finalize wrapper."""

    def __init__(
        self,
        recorder: EpisodeTarRecorder,
        *,
        max_normal_per_episode: int = 1,
        max_hard_per_episode: int = 2,
        thresholds: CandidateThresholds | None = None,
    ) -> None:
        if max_normal_per_episode < 0 or max_hard_per_episode < 0:
            raise ValueError("Per-episode candidate limits must be non-negative")
        self.recorder = recorder
        self.max_normal = int(max_normal_per_episode)
        self.max_hard = int(max_hard_per_episode)
        self.thresholds = thresholds or CandidateThresholds()
        self._episode_key: str | None = None
        self._episode_metadata: dict[str, Any] = {}
        self._pending: list[_PendingCandidate] = []

    def begin_episode(self, episode_key: str, episode_metadata: Mapping[str, Any]) -> None:
        if self._episode_key is not None:
            raise RuntimeError("An episode is already active")
        self._episode_key = self.recorder.validate_episode_key(episode_key)
        self._episode_metadata = dict(episode_metadata)
        self._pending = []

    def build_signals(
        self,
        route_tracker: MonotonicRouteTracker,
        current_pose: Any,
        history_poses: Any,
        native_future_poses: Any,
        oracle: OracleRelabelResult,
        **kwargs: Any,
    ) -> CandidateSignals:
        return build_candidate_signals(
            route_tracker,
            current_pose,
            history_poses,
            native_future_poses,
            oracle,
            thresholds=self.thresholds,
            **kwargs,
        )

    def consider(
        self,
        *,
        sample_key: str,
        current: HistoryObservation,
        history: Sequence[HistoryObservation],
        instruction: str,
        native_response: Mapping[str, Any],
        oracle: OracleRelabelResult,
        signals: CandidateSignals,
        metadata: Mapping[str, Any] | None = None,
    ) -> CandidateSelection:
        if self._episode_key is None:
            raise RuntimeError("begin_episode must be called before consider")
        response_kind = str(native_response.get("kind") or "")
        if response_kind != signals.native_kind:
            raise ValueError("native_response.kind and CandidateSignals.native_kind disagree")
        selection = classify_candidate(signals, thresholds=self.thresholds)
        if not selection.accepted:
            return selection
        if not oracle.valid:
            return CandidateSelection(
                "discard",
                selection.tags,
                ("oracle relabel is invalid",),
                selection.hardness_score,
            )
        history_tuple = tuple(history)
        history_ids = [observation.frame_id for observation in history_tuple]
        if len(history_ids) != len(set(history_ids)):
            raise ValueError("History contains duplicate frame IDs")
        ages = [max(0, current.primitive_step - observation.primitive_step) for observation in history_tuple]
        reserved = {
            "key",
            "source_type",
            "native_kind",
            "current_frame_id",
            "history_frame_ids",
            "history_valid_mask",
            "history_age_steps",
            "trajectory",
            "oracle_future_poses",
        }
        extras = dict(metadata or {})
        collision = reserved & set(extras)
        if collision:
            raise ValueError(f"Candidate metadata overwrites reserved fields: {sorted(collision)}")
        native_fields = {
            name: native_response.get(name)
            for name in (
                "llm_output",
                "native_first_output",
                "native_lookdown_turns",
                "native_front_only",
                "native_checkpoint_only",
                "system2_source",
                "system1_source",
                "policy_backend",
                "policy_fingerprint",
                "native_protocol",
                "pano_goal_view",
                "pixel_goal",
                "actions",
                "trajectory_summary",
                "trajectory_metrics",
                "trajectory_x_sign",
                "trajectory_heading_alignment",
                "anti_deadlock",
            )
            if name in native_response
        }
        sample = {
            "key": str(sample_key),
            "source_type": selection.bucket,
            "instruction": str(instruction),
            "native_kind": response_kind,
            "native": native_fields,
            "current_frame_id": current.frame_id,
            "history_frame_ids": history_ids,
            "history_valid_mask": [1] * len(history_ids),
            "history_age_steps": ages,
            "trajectory": oracle.trajectory,
            "trajectory_valid": float(oracle.trajectory_valid),
            "oracle_future_poses": oracle.future_poses,
            "oracle": {
                "kind": oracle.oracle_kind,
                "actions": list(oracle.actions),
                "terminal": oracle.terminal,
                "route_progress_m": oracle.route_progress_m,
                "travelled_m": oracle.travelled_m,
                "fallback_reason": oracle.fallback_reason,
            },
            "failure_tags": list(selection.tags),
            "candidate_signals": dataclasses.asdict(signals),
            **extras,
        }
        self._pending.append(_PendingCandidate(selection, sample, (current, *history_tuple)))
        return selection

    @staticmethod
    def _same_observation(left: HistoryObservation, right: HistoryObservation) -> bool:
        return (
            left.frame_id == right.frame_id
            and left.primitive_step == right.primitive_step
            and left.system2_call_index == right.system2_call_index
            and np.array_equal(left.pose, right.pose)
            and dict(left.view_jpegs) == dict(right.view_jpegs)
            and left.lookdown_jpeg == right.lookdown_jpeg
        )

    def finalize_episode(self) -> EpisodeCommit | None:
        if self._episode_key is None:
            raise RuntimeError("No active episode to finalize")
        hard = sorted(
            (item for item in self._pending if item.selection.bucket == "dagger_hard"),
            key=lambda item: (-item.selection.hardness_score, str(item.sample["key"])),
        )[: self.max_hard]
        normal = sorted(
            (item for item in self._pending if item.selection.bucket == "dagger_normal"),
            key=lambda item: str(item.sample["key"]),
        )[: self.max_normal]
        retained = [*normal, *hard]
        episode_key = self._episode_key
        episode_metadata = self._episode_metadata
        self.abort_episode()
        if not retained:
            return None

        observations: dict[int, HistoryObservation] = {}
        for item in retained:
            for observation in item.observations:
                existing = observations.get(observation.frame_id)
                if existing is not None and not self._same_observation(existing, observation):
                    raise ValueError(
                        f"Frame ID {observation.frame_id} has conflicting observation payloads"
                    )
                observations[observation.frame_id] = observation
        return self.recorder.record_episode(
            episode_key=episode_key,
            episode_metadata=episode_metadata,
            observations=list(observations.values()),
            samples=[item.sample for item in retained],
        )

    def abort_episode(self) -> None:
        self._episode_key = None
        self._episode_metadata = {}
        self._pending = []


__all__ = [
    "COLLECTION_SCHEMA",
    "COMMIT_CEILING_BYTES",
    "HARD_CAPACITY_BYTES",
    "CandidateSelection",
    "CandidateSignals",
    "CandidateThresholds",
    "CapacityExceededError",
    "CapacityGuard",
    "CollectionState",
    "EpisodeCommit",
    "EpisodeTarRecorder",
    "HabitatShadowBackend",
    "HistoryObservation",
    "MonotonicRouteTracker",
    "OracleRelabelConfig",
    "OracleRelabelResult",
    "RouteObservation",
    "TrajectoryDaggerCollector",
    "build_candidate_signals",
    "canonical_json_bytes",
    "classify_candidate",
    "encode_history_observation",
    "encode_rgb_to_jpeg",
    "encode_views",
    "logical_usage_bytes",
    "poses_to_nextdit_target",
    "prepare_collection",
    "relabel_with_shadow_oracle",
    "sample_history_indices",
    "scan_committed_episodes",
    "sha256_file",
]
