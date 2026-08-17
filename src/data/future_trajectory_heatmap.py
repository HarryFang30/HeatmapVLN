"""Four-horizon future-trajectory heatmap supervision.

This module contains label geometry only. Model inputs remain RGB plus the
existing history-pose provider; no depth or future pose enters the forward
path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .heatmap_geometry import compute_adaptive_sigma_pinhole, draw_gaussian_point

FUTURE_HEATMAP_SCHEMA = "heatmapvln-future-trajectory-4x4-v1"
FUTURE_TIME_RANGES = ((1, 8), (9, 16), (17, 24), (25, 32))
FUTURE_DIRECTION_ORDER = ("front", "right", "back", "left")
_VIEW_YAWS_DEG = (0.0, -90.0, 180.0, 90.0)


class FutureTrajectoryHeatmapError(ValueError):
    """Raised on a malformed future-label input."""


@dataclass(frozen=True)
class FutureTrajectoryHeatmapTarget:
    heatmap: np.ndarray
    visibility: np.ndarray
    view5: np.ndarray
    time_mask: np.ndarray
    anchor_heatmap: np.ndarray
    anchor_uv: np.ndarray
    schema: str = FUTURE_HEATMAP_SCHEMA


def action_deltas_to_camera_points(
    trajectory_delta_xyt: np.ndarray,
    *,
    action_scale: float,
    relative_heights_m: np.ndarray | None = None,
) -> np.ndarray:
    """Convert 32 scaled action deltas to current-front camera XYZ.

    Action fields are ``forward,left,delta_yaw`` after undoing
    ``action_scale``. Habitat camera coordinates are ``+X right,+Y up,-Z
    forward``. Height is supplied from expert/oracle poses so stairs survive;
    it defaults to the flat agent-height plane (relative Y=0).
    """

    value = np.asarray(trajectory_delta_xyt, dtype=np.float32)
    if value.shape != (32, 3):
        raise FutureTrajectoryHeatmapError(
            f"trajectory_delta_xyt must be [32,3], got {value.shape}"
        )
    if (
        not np.isfinite(value).all()
        or not np.isfinite(action_scale)
        or action_scale <= 0
    ):
        raise FutureTrajectoryHeatmapError(
            "trajectory/action_scale must be finite and valid"
        )
    xy = np.cumsum(value[:, :2] / float(action_scale), axis=0)
    if relative_heights_m is None:
        heights = np.zeros(32, dtype=np.float32)
    else:
        heights = np.asarray(relative_heights_m, dtype=np.float32)
        if heights.shape != (32,) or not np.isfinite(heights).all():
            raise FutureTrajectoryHeatmapError(
                "relative_heights_m must be finite [32]"
            )
    return np.column_stack((-xy[:, 1], heights, -xy[:, 0])).astype(np.float32)


def relative_future_centers_from_world(
    current_camera_c2w: np.ndarray,
    future_c2w: np.ndarray,
    *,
    future_poses_are_agent_base: bool = False,
    agent_camera_height_m: float = 1.25,
) -> np.ndarray:
    """Transform future centers into the current Habitat front-camera frame."""

    current = np.asarray(current_camera_c2w, dtype=np.float32)
    future = np.asarray(future_c2w, dtype=np.float32)
    if current.shape != (4, 4) or future.ndim != 3 or future.shape[1:] != (4, 4):
        raise FutureTrajectoryHeatmapError(
            "current/future poses must be [4,4] and [L,4,4]"
        )
    if (
        len(future) < 1
        or not np.isfinite(current).all()
        or not np.isfinite(future).all()
    ):
        raise FutureTrajectoryHeatmapError(
            "future pose sequence must be finite and non-empty"
        )
    if not np.isfinite(agent_camera_height_m):
        raise FutureTrajectoryHeatmapError(
            "agent_camera_height_m must be finite"
        )
    centers = future[:, :3, 3].copy()
    if future_poses_are_agent_base:
        # Habitat world Y is up. DAgger oracle poses are agent-base poses;
        # expert R2R pose_front entries are already camera-center poses.
        centers[:, 1] += float(agent_camera_height_m)
    homogeneous = np.concatenate(
        (centers, np.ones((len(centers), 1), dtype=np.float32)), axis=1
    )
    try:
        current_inv = np.linalg.inv(current)
    except np.linalg.LinAlgError as exc:
        raise FutureTrajectoryHeatmapError(
            "current_camera_c2w must be invertible"
        ) from exc
    relative = (current_inv @ homogeneous.T).T[:, :3]
    if not np.isfinite(relative).all():
        raise FutureTrajectoryHeatmapError(
            "world-to-current pose transform produced non-finite points"
        )
    return relative.astype(np.float32)


def interpolate_action_aligned_heights(
    raw_relative_camera_points: np.ndarray,
    action_flat_camera_points: np.ndarray,
) -> np.ndarray:
    """Interpolate expert height onto the 32 action-equivalent XY points."""

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

    # Prefix the current camera origin when the source begins at the first
    # future pose. Repeated/turn-only points contribute zero horizontal arc.
    raw_with_origin = np.concatenate(
        (np.zeros((1, 3), dtype=np.float32), raw), axis=0
    )
    raw_planar = raw_with_origin[:, (0, 2)]
    raw_arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(raw_planar, axis=0), axis=1)))
    )
    # Retain the last height at duplicate planar coordinates so a vertical
    # stair/elevator segment is not silently collapsed to its lower endpoint.
    keep = np.concatenate((np.diff(raw_arc) > 1.0e-6, [True]))
    raw_arc = raw_arc[keep]
    raw_height = raw_with_origin[:, 1][keep]
    if len(raw_arc) == 1:
        return np.full(32, raw_height[0], dtype=np.float32)

    action_planar = np.concatenate(
        (np.zeros((1, 2), dtype=np.float32), action[:, (0, 2)]), axis=0
    )
    action_arc = np.cumsum(
        np.concatenate(
            ([0.0], np.linalg.norm(np.diff(action_planar, axis=0), axis=1))
        )
    )[1:]
    return np.interp(
        np.minimum(action_arc, raw_arc[-1]), raw_arc, raw_height
    ).astype(np.float32)


def _rotation_y(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(
        ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float32
    )


def _project_unique_view(
    point_front: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[int, float, float, float] | None:
    width, height = (int(image_size[0]), int(image_size[1]))
    candidates: list[tuple[float, int, float, float, float]] = []
    for view_idx, yaw in enumerate(_VIEW_YAWS_DEG):
        point_view = _rotation_y(yaw).T @ point_front
        x, y, z = (
            float(point_view[0]),
            float(point_view[1]),
            float(point_view[2]),
        )
        if z >= -0.1:
            continue
        depth = -z
        u = float(intrinsics[0, 0]) * x / depth + float(intrinsics[0, 2])
        v = float(intrinsics[1, 1]) * (-y) / depth + float(intrinsics[1, 2])
        if 0.0 <= u < width and 0.0 <= v < height:
            # Prefer the view closest to its optical axis. View index is a
            # deterministic boundary tie-break in front/right/back/left order.
            candidates.append((abs(x / depth), view_idx, u, v, depth))
    if not candidates:
        return None
    _, view_idx, u, v, depth = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return view_idx, u, v, depth


def render_future_trajectory_heatmaps(
    relative_camera_points: np.ndarray,
    *,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (384, 384),
    heatmap_size: tuple[int, int] = (64, 64),
    time_mask: np.ndarray | None = None,
) -> FutureTrajectoryHeatmapTarget:
    """Render four 8-waypoint temporal tubes plus endpoint-anchor maps."""

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
    width, height = (int(image_size[0]), int(image_size[1]))
    hm_width, hm_height = (int(heatmap_size[0]), int(heatmap_size[1]))
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

    for time_idx, (start_one, end_one) in enumerate(FUTURE_TIME_RANGES):
        if not bool(mask[time_idx]):
            continue
        endpoint_projection = None
        for point_idx in range(start_one - 1, end_one):
            projection = _project_unique_view(points[point_idx], K, (width, height))
            if projection is None:
                continue
            view_idx, u, v, depth = projection
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
            draw_gaussian_point(heatmap[time_idx, view_idx], center, sigma)
            visibility[time_idx, view_idx] = 1.0
            if point_idx == end_one - 1:
                endpoint_projection = (view_idx, center, sigma)

        if endpoint_projection is not None:
            view_idx, center, sigma = endpoint_projection
            draw_gaussian_point(anchor_heatmap[time_idx, view_idx], center, sigma)
            anchor_uv[time_idx] = np.asarray(center, dtype=np.float32)
            view5[time_idx] = view_idx + 1

    return FutureTrajectoryHeatmapTarget(
        heatmap=heatmap,
        visibility=visibility,
        view5=view5,
        time_mask=mask,
        anchor_heatmap=anchor_heatmap,
        anchor_uv=anchor_uv,
    )


def build_future_target_from_action_and_poses(
    trajectory_delta_xyt: np.ndarray,
    *,
    action_scale: float,
    current_camera_c2w: np.ndarray,
    raw_future_poses: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (384, 384),
    heatmap_size: tuple[int, int] = (64, 64),
    trajectory_valid: bool = True,
    future_poses_are_agent_base: bool = False,
    agent_camera_height_m: float = 1.25,
) -> FutureTrajectoryHeatmapTarget:
    """Build the Future target without exposing pose to the model.

    The metric action target defines the exact 32 horizontal waypoints. Raw
    expert/oracle poses contribute only relative camera height, interpolated by
    horizontal arc length so flat floors stay at zero and stairs remain
    vertical. A valid all-zero STOP target remains four supervised none bins.
    """

    flat_points = action_deltas_to_camera_points(
        trajectory_delta_xyt,
        action_scale=action_scale,
    )
    if not bool(trajectory_valid):
        return render_future_trajectory_heatmaps(
            flat_points,
            intrinsics=intrinsics,
            image_size=image_size,
            heatmap_size=heatmap_size,
            time_mask=np.zeros(4, dtype=bool),
        )

    raw_relative = relative_future_centers_from_world(
        current_camera_c2w,
        raw_future_poses,
        future_poses_are_agent_base=future_poses_are_agent_base,
        agent_camera_height_m=agent_camera_height_m,
    )
    heights = interpolate_action_aligned_heights(raw_relative, flat_points)
    points = flat_points.copy()
    points[:, 1] = heights
    return render_future_trajectory_heatmaps(
        points,
        intrinsics=intrinsics,
        image_size=image_size,
        heatmap_size=heatmap_size,
    )
