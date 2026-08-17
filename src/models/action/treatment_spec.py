"""Canonical local action post-processing for the Stage-0 PPA gate.

This module is deliberately independent of Habitat and the model runtime.  A
raw bank of diffusion deltas is converted into the exact local queue consumed
by the R2R evaluator, while preserving every decision needed for an auditable
baseline-vs-treatment comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.utils.trajectory_direction import align_trajectory_endpoint_heading


ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

TRAJECTORY_SELECTION_CHOICES = (
    "mean",
    "endpoint_medoid",
    "path_medoid",
    "median_endpoint_nearest",
    "forward_or_medoid",
    "longest_forward",
)


@dataclass(frozen=True)
class TrajectoryPostprocessConfig:
    """All deployment knobs that can change a local action treatment."""

    num_sample_trajs: int = 32
    action_scale: float = 4.0
    trajectory_selection: str = "mean"
    trajectory_x_sign: float = 1.0
    target_heading_deg: float | None = None
    step_size_m: float = 0.25
    turn_angle_deg: float = 15.0
    lookahead: int = 4
    local_pad_to: int = 8
    local_execute_cap: int = 4
    first_stop_anti_deadlock_action: int = ACTION_LEFT

    def validate(self) -> None:
        if self.num_sample_trajs < 1:
            raise ValueError("num_sample_trajs must be positive")
        if not np.isfinite(self.action_scale) or self.action_scale <= 0:
            raise ValueError("action_scale must be positive and finite")
        if self.trajectory_selection not in TRAJECTORY_SELECTION_CHOICES:
            raise ValueError(
                f"unsupported trajectory_selection={self.trajectory_selection!r}"
            )
        if self.trajectory_x_sign not in (-1.0, 1.0):
            raise ValueError("trajectory_x_sign must be -1 or 1")
        if self.target_heading_deg is not None and not np.isfinite(
            self.target_heading_deg
        ):
            raise ValueError("target_heading_deg must be finite or None")
        if self.step_size_m <= 0 or self.turn_angle_deg <= 0 or self.lookahead < 1:
            raise ValueError("discretizer geometry must be positive")
        if self.local_pad_to < 1 or self.local_execute_cap < 1:
            raise ValueError("local queue lengths must be positive")
        if self.local_execute_cap > self.local_pad_to:
            raise ValueError("local_execute_cap cannot exceed local_pad_to")
        if self.first_stop_anti_deadlock_action not in (
            ACTION_LEFT,
            ACTION_RIGHT,
        ):
            raise ValueError("anti-deadlock action must be LEFT or RIGHT")


@dataclass(frozen=True)
class TreatmentSpec:
    """Exact local treatment represented at every post-processing boundary.

    ``response_actions`` is the queue returned by the model RPC server.
    ``habitat_actions`` excludes a local STOP because the client interprets it
    as a replan marker and never sends that STOP to Habitat.  A first-action
    STOP is replaced by one deterministic turn to break an identical-view
    System2/System1 loop.
    """

    schema: str
    trajectory_selection: str
    selected_trajectory_index: int | None
    action_scale: float
    trajectory_x_sign: float
    target_heading_deg: float | None
    heading_rotation_deg: float
    raw_discrete_actions: tuple[int, ...]
    padded_capped_actions: tuple[int, ...]
    response_actions: tuple[int, ...]
    habitat_actions: tuple[int, ...]
    execute_len: int
    end_reason: str
    replan_after: bool
    trigger_anti_deadlock: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reconstruct_xy_from_delta(delta_xyt: np.ndarray) -> np.ndarray:
    values = np.asarray(delta_xyt)
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError(f"expected [N,T,>=2] deltas, got {values.shape}")
    # Match the deployed legacy path exactly: scaling/cumsum happen in
    # float32, then assignment into the default-float64 path buffer upcasts.
    cumulative = np.cumsum(values[:, :, :2], axis=1)
    output = np.zeros((values.shape[0], values.shape[1] + 1, 2))
    output[:, 1:] = cumulative
    return output


def trajectory_xy_path_len(trajectory: np.ndarray) -> float:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(value[:, :2], axis=0), axis=1).sum())


def trajectory_to_discrete_actions(
    trajectory: np.ndarray,
    *,
    step_size_m: float = 0.25,
    turn_angle_deg: float = 15.0,
    lookahead: int = 4,
) -> list[int]:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
        raise ValueError(f"expected [T,>=2] trajectory, got {value.shape}")
    actions: list[int] = []
    yaw = 0.0
    pos = value[0, :2].copy()
    goal = value[-1, :2]
    turn_angle_rad = np.deg2rad(float(turn_angle_deg))

    def normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(value[:, :2] - pos, axis=1)
        nearest_idx = int(np.argmin(dists))
        target_idx = min(nearest_idx + int(lookahead), len(value) - 1)
        target_dir = value[target_idx, :2] - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break
        target_yaw = float(np.arctan2(target_dir[1], target_dir[0]))
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = round(delta_yaw / turn_angle_rad)
        if n_turns > 0:
            actions.extend([ACTION_LEFT] * n_turns)
        elif n_turns < 0:
            actions.extend([ACTION_RIGHT] * (-n_turns))
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)
        next_pos = pos + float(step_size_m) * np.array(
            [np.cos(yaw), np.sin(yaw)], dtype=np.float64
        )
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(ACTION_FORWARD)
        pos = next_pos
    return actions


def _endpoint_medoid_index(paths: np.ndarray) -> int:
    endpoints = paths[:, -1, :2]
    distances = np.linalg.norm(
        endpoints[:, None, :] - endpoints[None, :, :], axis=-1
    )
    return int(np.argmin(distances.sum(axis=1)))


def _path_medoid_index(paths: np.ndarray) -> int:
    flat = paths[:, :, :2].reshape(paths.shape[0], -1)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


def select_trajectory_xy(
    paths: np.ndarray,
    selection: str,
    *,
    discretizer: TrajectoryPostprocessConfig,
) -> tuple[np.ndarray, int | None]:
    value = np.asarray(paths, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] == 0:
        raise ValueError(f"expected non-empty [N,T,2] paths, got {value.shape}")
    if selection == "mean":
        return np.mean(value, axis=0), None
    if selection == "endpoint_medoid":
        index = _endpoint_medoid_index(value)
        return value[index], index
    if selection == "path_medoid":
        index = _path_medoid_index(value)
        return value[index], index
    if selection == "median_endpoint_nearest":
        endpoints = value[:, -1, :2]
        median_endpoint = np.median(endpoints, axis=0)
        index = int(
            np.argmin(np.linalg.norm(endpoints - median_endpoint[None, :], axis=-1))
        )
        return value[index], index

    candidates: list[tuple[int, int, float]] = []
    for index, path in enumerate(value):
        actions = trajectory_to_discrete_actions(
            path,
            step_size_m=discretizer.step_size_m,
            turn_angle_deg=discretizer.turn_angle_deg,
            lookahead=discretizer.lookahead,
        )
        forward_count = actions.count(ACTION_FORWARD)
        if forward_count:
            candidates.append((index, forward_count, trajectory_xy_path_len(path)))

    if selection == "forward_or_medoid":
        if candidates:
            medoid_endpoint = value[_endpoint_medoid_index(value), -1, :2]
            median_path_len = float(
                np.median([trajectory_xy_path_len(path) for path in value])
            )

            def score(item: tuple[int, int, float]) -> tuple[float, int, float]:
                index, forward_count, path_len = item
                endpoint_distance = float(
                    np.linalg.norm(value[index, -1, :2] - medoid_endpoint)
                )
                return endpoint_distance, -forward_count, abs(path_len - median_path_len)

            index = min(candidates, key=score)[0]
        else:
            index = _endpoint_medoid_index(value)
        return value[index], index
    if selection == "longest_forward":
        index = (
            max(candidates, key=lambda item: (item[2], item[1]))[0]
            if candidates
            else _endpoint_medoid_index(value)
        )
        return value[index], index
    raise ValueError(f"unsupported selection={selection!r}")


def build_treatment_spec(
    dp_actions: Any,
    config: TrajectoryPostprocessConfig,
) -> TreatmentSpec:
    """Create the exact RPC/client local treatment from raw NextDiT deltas."""

    config.validate()
    value = (
        dp_actions.detach().float().cpu().numpy()
        if hasattr(dp_actions, "detach")
        else np.asarray(dp_actions, dtype=np.float32)
    )
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] < 2:
        raise ValueError(f"expected [N,T,>=2] diffusion samples, got {value.shape}")
    if value.shape[0] < config.num_sample_trajs:
        raise ValueError(
            f"need {config.num_sample_trajs} diffusion samples, got {value.shape[0]}"
        )
    if not np.isfinite(value).all():
        raise ValueError("diffusion samples contain non-finite values")

    deltas = value[: config.num_sample_trajs].copy()
    deltas[:, :, :2] /= float(config.action_scale)
    deltas[:, :, 0] *= float(config.trajectory_x_sign)
    paths = reconstruct_xy_from_delta(deltas)
    selected, selected_index = select_trajectory_xy(
        paths,
        config.trajectory_selection,
        discretizer=config,
    )
    heading_rotation_deg = 0.0
    if config.target_heading_deg is not None:
        selected, heading_rotation_deg = align_trajectory_endpoint_heading(
            selected,
            target_angle_deg=float(config.target_heading_deg),
        )

    raw = trajectory_to_discrete_actions(
        selected,
        step_size_m=config.step_size_m,
        turn_angle_deg=config.turn_angle_deg,
        lookahead=config.lookahead,
    )
    if not raw:
        raw = [ACTION_STOP]
    padded = (raw + [ACTION_STOP] * max(config.local_pad_to - len(raw), 0))[
        : config.local_execute_cap
    ]

    anti_deadlock = bool(padded and padded[0] == ACTION_STOP)
    if anti_deadlock:
        response = [int(config.first_stop_anti_deadlock_action)]
        habitat = list(response)
        end_reason = "anti_deadlock_replan"
    else:
        response = [int(action) for action in padded]
        stop_index = response.index(ACTION_STOP) if ACTION_STOP in response else len(response)
        habitat = response[:stop_index]
        end_reason = (
            "local_stop_replan"
            if stop_index < len(response)
            else "queue_exhausted_replan"
        )

    return TreatmentSpec(
        schema="heatmapvln-local-treatment-v1",
        trajectory_selection=config.trajectory_selection,
        selected_trajectory_index=selected_index,
        action_scale=float(config.action_scale),
        trajectory_x_sign=float(config.trajectory_x_sign),
        target_heading_deg=(
            None
            if config.target_heading_deg is None
            else float(config.target_heading_deg)
        ),
        heading_rotation_deg=float(heading_rotation_deg),
        raw_discrete_actions=tuple(int(action) for action in raw),
        padded_capped_actions=tuple(int(action) for action in padded),
        response_actions=tuple(response),
        habitat_actions=tuple(habitat),
        execute_len=len(habitat),
        end_reason=end_reason,
        replan_after=True,
        trigger_anti_deadlock=anti_deadlock,
    )


def assert_exact_treatment_spec_equal(
    baseline: TreatmentSpec,
    treatment: TreatmentSpec,
) -> None:
    if baseline != treatment:
        left = baseline.to_dict()
        right = treatment.to_dict()
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise RuntimeError(
            "Stage-0 TreatmentSpec mismatch at fields "
            f"{differing}: baseline={left} treatment={right}"
        )


__all__ = [
    "ACTION_FORWARD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STOP",
    "TRAJECTORY_SELECTION_CHOICES",
    "TrajectoryPostprocessConfig",
    "TreatmentSpec",
    "assert_exact_treatment_spec_equal",
    "build_treatment_spec",
    "reconstruct_xy_from_delta",
    "select_trajectory_xy",
    "trajectory_to_discrete_actions",
    "trajectory_xy_path_len",
]
