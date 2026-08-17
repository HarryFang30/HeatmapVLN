"""
Trajectory processing utilities — coordinate transforms, resampling,
and augmentation.

Reference: InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def _drop_duplicate_trajectory_points(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Remove non-finite and consecutive duplicate 2D points.

    ``scipy.interpolate.CubicSpline`` requires the distance coordinate to be
    strictly increasing. Stationary frames can otherwise create zero-length
    segments and make the whole trajectory target invalid.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = points.reshape(-1, points.shape[-1])[:, :2]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) <= 1:
        return points.astype(np.float32, copy=False)

    eps_sq = float(eps * eps)
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    last = points[0]
    for idx in range(1, len(points)):
        if float(np.sum((points[idx] - last) ** 2)) > eps_sq:
            keep[idx] = True
            last = points[idx]
    return points[keep].astype(np.float32, copy=False)


def compute_history_rel_poses(
    history_poses: list[np.ndarray],
    current_pose: np.ndarray,
    camera_deg: float = 0,
    camera_forward_axis: str = "+z",
) -> np.ndarray:
    """Compute (dx, dy, cos_yaw, sin_yaw) for each history pose relative to current.

    Returns:
        rel_poses: [K, 4]
    """
    if len(history_poses) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    all_poses = np.stack(
        [np.array(current_pose, dtype=np.float32)]
        + [np.array(p, dtype=np.float32) for p in history_poses],
        axis=0,
    )
    rel_xyyaw = get_trajectory_relative_to_frame(
        all_poses,
        camera_deg=camera_deg,
        camera_forward_axis=camera_forward_axis,
    )
    hist_rel = rel_xyyaw[1:]
    return np.column_stack([
        hist_rel[:, :2],
        np.cos(hist_rel[:, 2]),
        np.sin(hist_rel[:, 2]),
    ]).astype(np.float32)


def get_trajectory_relative_to_frame(
    extrinsics: np.ndarray,
    camera_deg: float = 0,
    camera_forward_axis: str = "+z",
) -> np.ndarray:
    """Compute trajectory poses (x, y, yaw) relative to the first frame.

    Args:
        extrinsics: Pose matrices in the convention used by InternNav's
            trajectory transform, shape (n, 4, 4).
        camera_deg: camera pitch in degrees
        camera_forward_axis: Interpretation applied by this HeatmapVLN
            wrapper. The legacy call path kept its historical ``+z``
            semantics. The random-walk poses are Habitat camera-to-world
            matrices whose RGB camera faces local ``-z``; selecting ``-z``
            makes ``x`` forward, ``y`` left, and positive yaw a left turn.

    Returns:
        relative_xyyaw: (n, 3)
    """
    if camera_forward_axis not in {"+z", "-z"}:
        raise ValueError(
            "camera_forward_axis must be '+z' or '-z', "
            f"got {camera_forward_axis!r}"
        )

    T_camera2robot = np.array([
        [[0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ])

    T_robot2camera = np.array([
        [[0.0, 0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ])

    if camera_deg is not None and camera_deg != 0:
        camera_rad = np.radians(camera_deg)
        T_deg = np.array([
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
             [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ], dtype=np.float32)
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera[0])[np.newaxis]

    extrinsics_robot = np.matmul(extrinsics, T_camera2robot[0])

    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)

    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)

    relative_translations = relative_to_ref[:, :2, 3]
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])

    relative_xyyaw = np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)

    if camera_forward_axis == "-z":
        # Convert the legacy HeatmapVLN wrapper fields to the physical Habitat
        # c2w convention (forward, left, left-positive yaw). The reflection
        # changes forward and yaw signs while preserving lateral left.
        relative_xyyaw[:, 0] *= -1.0
        relative_xyyaw[:, 2] *= -1.0

    return relative_xyyaw


def smooth_and_resample_trajectory(points: np.ndarray, sample_length: int = 25, interval: float = 0.1) -> np.ndarray:
    """Smooth and resample a 2D trajectory to fixed length via cubic spline.

    Args:
        points: (n, 2) trajectory points
        sample_length: output length
        interval: sampling interval in metres

    Returns:
        resampled: (sample_length, 2)
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        logger.warning("scipy not available, using linear interpolation")
        if len(points) == 0:
            return np.zeros((sample_length, 2))
        if len(points) == 1:
            return np.tile(points[0], (sample_length, 1))
        indices = np.linspace(0, len(points) - 1, sample_length)
        return np.array([points[int(i)] for i in indices])

    total_distance = sample_length * interval
    points = _drop_duplicate_trajectory_points(points)

    if len(points) == 0:
        return np.zeros((sample_length, 2))

    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))

    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    if cumulative_distances[-1] <= 1e-6:
        return np.tile(points[0], (sample_length, 1))

    if len(points) > 3:
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])

        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))

        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
        smoothed_points = _drop_duplicate_trajectory_points(smoothed_points)
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances

    if len(smoothed_points) == 0:
        return np.zeros((sample_length, 2))
    if len(smoothed_points) == 1 or smooth_cumulative_distances[-1] <= 1e-6:
        return np.tile(smoothed_points[0], (sample_length, 1))

    target_distances = np.linspace(0, total_distance, sample_length)
    resampled = np.zeros((sample_length, 2))

    for i, target_dist in enumerate(target_distances):
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue

        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1
        segment_idx = max(0, min(segment_idx, len(smooth_cumulative_distances) - 2))

        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]

        if end_dist > start_dist:
            t = (target_dist - start_dist) / (end_dist - start_dist)
        else:
            t = 0

        resampled[i] = smoothed_points[segment_idx] + t * (
            smoothed_points[min(segment_idx + 1, len(smoothed_points) - 1)] - smoothed_points[segment_idx]
        )

    return resampled


def xy_to_delta_xyt(xy_actions: np.ndarray) -> np.ndarray:
    """Convert absolute (x, y) positions to incremental (dx, dy, delta_yaw).

    Returns:
        delta_xyt: (N-1, 3)
    """
    if len(xy_actions) < 2:
        return np.zeros((max(0, len(xy_actions) - 1), 3), dtype=np.float32)

    vectors = np.diff(xy_actions, axis=0)
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])

    if len(yaw) < 2:
        delta_yaw = yaw.copy() if len(yaw) > 0 else np.array([0.0])
    else:
        delta_yaw = np.diff(yaw)
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        delta_yaw = np.concatenate([[yaw[0]], delta_yaw])

    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt.astype(np.float32)


def interpolate_and_resample_trajectory(
    absolute_trajectories: np.ndarray,
    predict_step_num: int = 24,
    action_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, smooth, resample trajectory and convert to delta actions.

    Returns:
        resampled_trajectories: (predict_step_num+1, 2) absolute positions
        resampled_relative_poses: (predict_step_num, 3) delta (dx, dy, dyaw)
    """
    start_point = np.array([[0.0, 0.0]])

    traj = absolute_trajectories[..., :2]

    if len(traj) > 1:
        steps = traj[1:] - traj[:-1]
        steps_sq = (steps**2).sum(axis=-1)
        mask = steps_sq > 0.05

        filtered_traj = traj[1:][mask]
        filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)
    else:
        filtered_traj = start_point

    resampled_trajectories = smooth_and_resample_trajectory(
        filtered_traj, sample_length=predict_step_num + 1
    )
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)

    resampled_relative_poses[:, 0:2] *= action_scale

    return resampled_trajectories, resampled_relative_poses


def apply_trajectory_augmentation(
    trajectory: np.ndarray,
    rotation_range: float = 0.3,
    scale_range: tuple[float, float] = (0.8, 1.2),
    p: float = 0.5,
) -> np.ndarray:
    """Random rotation and scaling augmentation on (dx, dy, delta_yaw) trajectories."""
    if random.random() > p:
        return trajectory

    augmented = trajectory.copy()

    if random.random() > 0.5:
        angle = random.uniform(-rotation_range, rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        dx = augmented[:, 0].copy()
        dy = augmented[:, 1].copy()
        augmented[:, 0] = dx * cos_a - dy * sin_a
        augmented[:, 1] = dx * sin_a + dy * cos_a

        augmented[:, 2] += angle
        augmented[:, 2] = (augmented[:, 2] + np.pi) % (2 * np.pi) - np.pi

    if random.random() > 0.5:
        scale = random.uniform(*scale_range)
        augmented[:, 0:2] *= scale

    return augmented
