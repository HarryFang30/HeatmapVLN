"""No-depth four-bin future-trajectory heatmap label geometry.

The renderer consumes the exact 32-step expert action target used by
System-1.  GT camera poses are consulted only to recover relative height (so
stairs are retained); neither poses nor depth are returned as model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .heatmap_geometry import compute_adaptive_sigma_pinhole, draw_gaussian_point


FUTURE_HEATMAP_SCHEMA = "heatmapvln-future-trajectory-4x4-v1"
FUTURE_HEATMAP_TARGET_SOURCE = "expert_system1_action_target"
FUTURE_TIME_RANGES = ((1, 8), (9, 16), (17, 24), (25, 32))
FUTURE_DIRECTION_ORDER = ("front", "right", "back", "left")
_VIEW_YAWS_DEG = (0.0, -90.0, 180.0, 90.0)


class FutureTrajectoryHeatmapError(ValueError):
    """Raised when future-label geometry violates the fixed contract."""


@dataclass(frozen=True)
class FutureTrajectoryHeatmapTarget:
    heatmap: np.ndarray
    visibility: np.ndarray
    view5: np.ndarray
    time_mask: np.ndarray
    anchor_heatmap: np.ndarray
    anchor_uv: np.ndarray
    schema: str = FUTURE_HEATMAP_SCHEMA
    target_source: str = FUTURE_HEATMAP_TARGET_SOURCE


def action_deltas_to_camera_points(
    trajectory_delta_xyt: np.ndarray,
    *,
    action_scale: float,
    relative_heights_m: np.ndarray | None = None,
) -> np.ndarray:
    """Convert the native scaled action target to current-front camera XYZ."""

    value = np.asarray(trajectory_delta_xyt, dtype=np.float32)
    if value.shape != (32, 3):
        raise FutureTrajectoryHeatmapError(
            f"trajectory_delta_xyt must be [32,3], got {value.shape}"
        )
    if (
        not np.isfinite(value).all()
        or not np.isfinite(action_scale)
        or float(action_scale) <= 0.0
    ):
        raise FutureTrajectoryHeatmapError(
            "trajectory_delta_xyt/action_scale must be finite and valid"
        )

    # Native fields are incremental (forward, left, delta-yaw), with XY
    # multiplied by action_scale.  Habitat camera coordinates are +X right,
    # +Y up, -Z forward.
    forward_left = np.cumsum(value[:, :2] / float(action_scale), axis=0)
    if relative_heights_m is None:
        heights = np.zeros(32, dtype=np.float32)
    else:
        heights = np.asarray(relative_heights_m, dtype=np.float32)
        if heights.shape != (32,) or not np.isfinite(heights).all():
            raise FutureTrajectoryHeatmapError(
                "relative_heights_m must be finite [32]"
            )
    return np.column_stack(
        (-forward_left[:, 1], heights, -forward_left[:, 0])
    ).astype(np.float32)


def relative_future_centers_from_world(
    current_camera_c2w: np.ndarray,
    future_camera_c2w: np.ndarray,
) -> np.ndarray:
    """Transform expert camera centers into the current front-camera frame."""

    current = np.asarray(current_camera_c2w, dtype=np.float32)
    future = np.asarray(future_camera_c2w, dtype=np.float32)
    if current.shape != (4, 4) or future.ndim != 3 or future.shape[1:] != (4, 4):
        raise FutureTrajectoryHeatmapError(
            "current/future camera poses must be [4,4] and [L,4,4]"
        )
    if len(future) < 1 or not np.isfinite(current).all() or not np.isfinite(future).all():
        raise FutureTrajectoryHeatmapError(
            "future camera-pose sequence must be finite and non-empty"
        )
    centers = future[:, :3, 3]
    homogeneous = np.concatenate(
        (centers, np.ones((len(centers), 1), dtype=np.float32)), axis=1
    )
    try:
        relative = (np.linalg.inv(current) @ homogeneous.T).T[:, :3]
    except np.linalg.LinAlgError as exc:
        raise FutureTrajectoryHeatmapError(
            "current_camera_c2w must be invertible"
        ) from exc
    if not np.isfinite(relative).all():
        raise FutureTrajectoryHeatmapError(
            "world-to-current transform produced non-finite points"
        )
    return relative.astype(np.float32)


def interpolate_action_aligned_heights(
    raw_relative_camera_points: np.ndarray,
    action_flat_camera_points: np.ndarray,
) -> np.ndarray:
    """Interpolate expert height onto the exact 32 System-1 XY waypoints."""

    raw = np.asarray(raw_relative_camera_points, dtype=np.float32)
    action = np.asarray(action_flat_camera_points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 1:
        raise FutureTrajectoryHeatmapError(
            "raw_relative_camera_points must be [L,3]"
        )
    if action.shape != (32, 3):
        raise FutureTrajectoryHeatmapError(
            "action_flat_camera_points must be [32,3]"
        )
    if not np.isfinite(raw).all() or not np.isfinite(action).all():
        raise FutureTrajectoryHeatmapError(
            "height interpolation inputs must be finite"
        )

    raw_with_origin = np.concatenate(
        (np.zeros((1, 3), dtype=np.float32), raw), axis=0
    )
    raw_arc = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(
                    np.diff(raw_with_origin[:, (0, 2)], axis=0), axis=1
                )
            ),
        )
    )
    # Preserve the final height at duplicate planar positions.  This keeps a
    # vertical stair/elevator segment rather than silently selecting its lower
    # endpoint.
    keep = np.concatenate((np.diff(raw_arc) > 1.0e-6, [True]))
    raw_arc = raw_arc[keep]
    raw_height = raw_with_origin[:, 1][keep]
    if len(raw_arc) == 1:
        return np.full(32, raw_height[0], dtype=np.float32)

    action_with_origin = np.concatenate(
        (np.zeros((1, 2), dtype=np.float32), action[:, (0, 2)]), axis=0
    )
    action_arc = np.cumsum(
        np.concatenate(
            (
                [0.0],
                np.linalg.norm(np.diff(action_with_origin, axis=0), axis=1),
            )
        )
    )[1:]
    return np.interp(
        np.minimum(action_arc, raw_arc[-1]), raw_arc, raw_height
    ).astype(np.float32)


def _rotation_y(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(
        ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)),
        dtype=np.float32,
    )


def _project_unique_view(
    point_front: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[int, float, float, float] | None:
    width, height = int(image_size[0]), int(image_size[1])
    candidates: list[tuple[float, int, float, float, float]] = []
    for view_index, yaw in enumerate(_VIEW_YAWS_DEG):
        point_view = _rotation_y(yaw).T @ point_front
        x, y, z = map(float, point_view)
        if z >= -0.1:
            continue
        depth = -z
        u = float(intrinsics[0, 0]) * x / depth + float(intrinsics[0, 2])
        v = float(intrinsics[1, 1]) * (-y) / depth + float(intrinsics[1, 2])
        if 0.0 <= u < width and 0.0 <= v < height:
            candidates.append((abs(x / depth), view_index, u, v, depth))
    if not candidates:
        return None
    _, view_index, u, v, depth = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return view_index, u, v, depth


def render_future_trajectory_heatmaps(
    relative_camera_points: np.ndarray,
    *,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (384, 384),
    heatmap_size: tuple[int, int] = (64, 64),
    time_mask: np.ndarray | None = None,
) -> FutureTrajectoryHeatmapTarget:
    """Render four temporal tubes (eight waypoints each) without depth."""

    points = np.asarray(relative_camera_points, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if points.shape != (32, 3) or not np.isfinite(points).all():
        raise FutureTrajectoryHeatmapError(
            "relative_camera_points must be finite [32,3]"
        )
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
    ):
        raise FutureTrajectoryHeatmapError(
            "intrinsics must be a valid finite [3,3]"
        )
    width, height = int(image_size[0]), int(image_size[1])
    hm_width, hm_height = int(heatmap_size[0]), int(heatmap_size[1])
    if min(width, height, hm_width, hm_height) <= 0:
        raise FutureTrajectoryHeatmapError(
            "image and heatmap sizes must be positive"
        )
    if time_mask is None:
        mask = np.ones(4, dtype=bool)
    else:
        mask = np.asarray(time_mask, dtype=bool)
        if mask.shape != (4,):
            raise FutureTrajectoryHeatmapError("time_mask must be [4]")

    heatmap = np.zeros((4, 4, hm_height, hm_width), dtype=np.float32)
    anchor_heatmap = np.zeros_like(heatmap)
    visibility = np.zeros((4, 4), dtype=np.float32)
    view5 = np.zeros(4, dtype=np.int64)
    anchor_uv = np.full((4, 2), np.nan, dtype=np.float32)

    for time_index, (start_one, end_one) in enumerate(FUTURE_TIME_RANGES):
        if not bool(mask[time_index]):
            continue
        endpoint_projection = None
        for point_index in range(start_one - 1, end_one):
            projection = _project_unique_view(
                points[point_index], K, (width, height)
            )
            if projection is None:
                continue
            view_index, u, v, depth = projection
            sigma = compute_adaptive_sigma_pinhole(
                z_depth=depth,
                fx=float(K[0, 0]),
                object_size_3d=1.5,
                heatmap_width=hm_width,
                img_width=width,
                min_sigma=4.0,
                max_sigma=8.0,
            )
            center = (u * hm_width / width, v * hm_height / height)
            draw_gaussian_point(heatmap[time_index, view_index], center, sigma)
            visibility[time_index, view_index] = 1.0
            if point_index == end_one - 1:
                endpoint_projection = (view_index, center, sigma)

        if endpoint_projection is not None:
            view_index, center, sigma = endpoint_projection
            draw_gaussian_point(
                anchor_heatmap[time_index, view_index], center, sigma
            )
            anchor_uv[time_index] = np.asarray(center, dtype=np.float32)
            view5[time_index] = view_index + 1

    return FutureTrajectoryHeatmapTarget(
        heatmap=heatmap,
        visibility=visibility,
        view5=view5,
        time_mask=mask,
        anchor_heatmap=anchor_heatmap,
        anchor_uv=anchor_uv,
    )


def build_future_target_from_system1_action(
    system1_trajectory_delta_xyt: np.ndarray,
    *,
    action_scale: float,
    current_camera_c2w: np.ndarray,
    expert_future_camera_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (384, 384),
    heatmap_size: tuple[int, int] = (64, 64),
) -> FutureTrajectoryHeatmapTarget:
    """Build a target from the exact action label; GT pose supplies height only."""

    flat_points = action_deltas_to_camera_points(
        system1_trajectory_delta_xyt,
        action_scale=action_scale,
    )
    raw_relative = relative_future_centers_from_world(
        current_camera_c2w,
        expert_future_camera_c2w,
    )
    points = flat_points.copy()
    points[:, 1] = interpolate_action_aligned_heights(raw_relative, flat_points)
    return render_future_trajectory_heatmaps(
        points,
        intrinsics=intrinsics,
        image_size=image_size,
        heatmap_size=heatmap_size,
    )
