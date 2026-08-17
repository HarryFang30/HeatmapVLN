"""Direction-response diagnostics for local trajectory predictions."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np


VIEW_TARGET_ANGLE_DEG: dict[str, float] = {
    # Local trajectory angles are counter-clockwise: +Y turns left.
    "front": 0.0,
    "left": 90.0,
    "back": 180.0,
    "right": -90.0,
}

_PANO_RECENTER_TURN: dict[str, tuple[str | None, float]] = {
    "front": (None, 0.0),
    "right": ("right", 90.0),
    # Either direction is equivalent for 180 degrees. Keep right deterministic.
    "back": ("right", 180.0),
    "left": ("left", 90.0),
}


def normalize_angle_deg(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def angular_error_deg(angle: float | np.ndarray, target: float) -> float | np.ndarray:
    return np.abs(normalize_angle_deg(np.asarray(angle) - float(target)))


def view_pixel_target_angle_deg(
    view_id: str,
    pixel_xy: tuple[float, float] | list[float],
    image_size: tuple[int, int] | list[int],
    *,
    horizontal_fov_deg: float = 90.0,
) -> float:
    view = str(view_id).lower()
    if view not in VIEW_TARGET_ANGLE_DEG:
        raise ValueError(f"Unsupported view_id={view_id!r}")
    width = float(image_size[0])
    if width <= 0.0:
        raise ValueError(f"image width must be positive, got {image_size}")
    u = float(pixel_xy[0])
    pixel_offset_deg = -((u / width) - 0.5) * float(horizontal_fov_deg)
    return float(normalize_angle_deg(VIEW_TARGET_ANGLE_DEG[view] + pixel_offset_deg))


def pano_recenter_turn(
    view_id: str,
    *,
    turn_angle_deg: float,
    atol: float = 1.0e-6,
) -> tuple[str | None, int]:
    """Return the exact discrete turn needed to make a pano view the new front.

    ``right`` is negative yaw and ``left`` is positive yaw. The caller must
    execute the returned real Habitat turns before capturing the System1
    front/lookdown observation. Refuse non-divisible turn angles instead of
    silently leaving a residual camera-heading error.
    """
    view = str(view_id).lower()
    if view not in _PANO_RECENTER_TURN:
        raise ValueError(f"Unsupported view_id={view_id!r}")
    if not np.isfinite(turn_angle_deg) or float(turn_angle_deg) <= 0.0:
        raise ValueError(f"turn_angle_deg must be positive and finite, got {turn_angle_deg!r}")
    direction, total_deg = _PANO_RECENTER_TURN[view]
    if direction is None:
        return None, 0
    raw_count = total_deg / float(turn_angle_deg)
    count = int(round(raw_count))
    if count <= 0 or not np.isclose(raw_count, count, rtol=0.0, atol=float(atol)):
        raise ValueError(
            f"Pano recenter angle {total_deg:g} is not divisible by "
            f"Habitat TURN_ANGLE={float(turn_angle_deg):g}"
        )
    return direction, count


def align_trajectory_endpoint_heading(
    trajectory_xy: np.ndarray,
    *,
    target_angle_deg: float,
    min_endpoint_distance: float = 1.0e-6,
) -> tuple[np.ndarray, float]:
    """Rotate a local path so its endpoint bearing matches a pano goal ray."""
    trajectory = np.asarray(trajectory_xy, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
        raise ValueError(f"Expected trajectory [T,>=2], got {trajectory.shape}")
    aligned = trajectory.copy()
    origin = trajectory[0, :2]
    endpoint_delta = trajectory[-1, :2] - origin
    endpoint_distance = float(np.linalg.norm(endpoint_delta))
    if endpoint_distance <= float(min_endpoint_distance):
        return aligned, 0.0

    current_angle_deg = float(np.degrees(np.arctan2(endpoint_delta[1], endpoint_delta[0])))
    rotation_deg = float(normalize_angle_deg(float(target_angle_deg) - current_angle_deg))
    rotation_rad = np.deg2rad(rotation_deg)
    rotation = np.array(
        [
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)],
        ],
        dtype=np.float64,
    )
    aligned[:, :2] = (trajectory[:, :2] - origin) @ rotation.T + origin
    return aligned, rotation_deg


def reconstruct_delta_xy(
    delta_xyt: np.ndarray,
    *,
    action_scale: float,
    trajectory_x_sign: float = 1.0,
) -> np.ndarray:
    deltas = np.asarray(delta_xyt, dtype=np.float64).copy()
    if deltas.ndim != 3 or deltas.shape[-1] < 2:
        raise ValueError(f"Expected [N,T,>=2] trajectory deltas, got {deltas.shape}")
    if action_scale <= 0.0:
        raise ValueError(f"action_scale must be positive, got {action_scale}")
    if trajectory_x_sign not in (-1.0, 1.0):
        raise ValueError(f"trajectory_x_sign must be -1 or 1, got {trajectory_x_sign}")
    deltas[:, :, :2] /= float(action_scale)
    deltas[:, :, 0] *= float(trajectory_x_sign)
    cumulative = np.cumsum(deltas[:, :, :2], axis=1)
    origin = np.zeros((deltas.shape[0], 1, 2), dtype=np.float64)
    return np.concatenate([origin, cumulative], axis=1)


def summarize_direction_response(
    delta_xyt: np.ndarray,
    *,
    view_id: str,
    action_scale: float,
    trajectory_x_sign: float = 1.0,
    target_angle_deg: float | None = None,
) -> dict[str, object]:
    view = str(view_id).lower()
    if view not in VIEW_TARGET_ANGLE_DEG:
        raise ValueError(f"Unsupported view_id={view_id!r}")
    expected_deg = (
        VIEW_TARGET_ANGLE_DEG[view]
        if target_angle_deg is None
        else float(normalize_angle_deg(target_angle_deg))
    )
    paths = reconstruct_delta_xy(
        delta_xyt,
        action_scale=action_scale,
        trajectory_x_sign=trajectory_x_sign,
    )
    endpoints = paths[:, -1, :2]
    direct = np.linalg.norm(endpoints, axis=1)
    angles = np.degrees(np.arctan2(endpoints[:, 1], endpoints[:, 0]))
    valid = direct > 1.0e-6
    errors = np.full_like(direct, 180.0)
    errors[valid] = angular_error_deg(angles[valid], expected_deg)

    target_rad = np.deg2rad(expected_deg)
    target_unit = np.array([np.cos(target_rad), np.sin(target_rad)], dtype=np.float64)
    progress = endpoints @ target_unit
    alignment = np.full_like(direct, -1.0)
    alignment[valid] = progress[valid] / direct[valid]

    mean_path = paths.mean(axis=0)
    mean_endpoint = mean_path[-1, :2]
    mean_direct = float(np.linalg.norm(mean_endpoint))
    mean_angle = (
        float(np.degrees(np.arctan2(mean_endpoint[1], mean_endpoint[0])))
        if mean_direct > 1.0e-6
        else None
    )
    mean_error = (
        float(angular_error_deg(mean_angle, expected_deg))
        if mean_angle is not None
        else 180.0
    )

    return {
        "view": view,
        "expected_angle_deg": float(expected_deg),
        "num_candidates": int(paths.shape[0]),
        "candidate_endpoint_angles_deg": [float(v) for v in angles.tolist()],
        "candidate_endpoint_xy_m": [[float(x), float(y)] for x, y in endpoints.tolist()],
        "candidate_angle_error_mean_deg": float(errors.mean()),
        "candidate_angle_error_median_deg": float(np.median(errors)),
        "candidate_within_45_rate": float(np.mean(errors <= 45.0)),
        "candidate_within_90_rate": float(np.mean(errors <= 90.0)),
        "candidate_positive_progress_rate": float(np.mean(progress > 0.0)),
        "candidate_alignment_mean": float(alignment.mean()),
        "candidate_progress_mean_m": float(progress.mean()),
        "candidate_direct_mean_m": float(direct.mean()),
        "mean_endpoint_xy_m": [float(mean_endpoint[0]), float(mean_endpoint[1])],
        "mean_endpoint_direct_m": mean_direct,
        "mean_endpoint_angle_deg": mean_angle,
        "mean_endpoint_angle_error_deg": mean_error,
        "mean_endpoint_progress_m": float(mean_endpoint @ target_unit),
    }


def pairwise_representation_stats(
    representations: Mapping[str, np.ndarray],
) -> dict[str, object]:
    pairs: list[dict[str, float | str]] = []
    for left, right in combinations(sorted(representations), 2):
        a = np.asarray(representations[left], dtype=np.float64).reshape(-1)
        b = np.asarray(representations[right], dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"Representation shape mismatch: {left}={a.shape}, {right}={b.shape}")
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        denom = max(a_norm * b_norm, 1.0e-12)
        cosine = float(np.dot(a, b) / denom)
        relative_l2 = float(np.linalg.norm(a - b) / max(0.5 * (a_norm + b_norm), 1.0e-12))
        pairs.append({
            "left": left,
            "right": right,
            "cosine": cosine,
            "relative_l2": relative_l2,
        })
    return {
        "pairs": pairs,
        "cosine_mean": float(np.mean([p["cosine"] for p in pairs])) if pairs else 1.0,
        "relative_l2_mean": (
            float(np.mean([p["relative_l2"] for p in pairs])) if pairs else 0.0
        ),
    }
