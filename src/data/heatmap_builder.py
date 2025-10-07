"""
Geometric Tools for Heatmap Generation (heatmap_builder.py)
=========================================================

Pure mathematical/geometric implementation for dataset collection.
No external simulation dependencies - only numpy/cv2/torch.

Key Functions:
- Camera intrinsics handling (fx/fy/cx/cy or hfov)
- SO(3)/SE(3) pose distance metrics (for FPS)
- Deterministic keyframe sampling (Uniform/FPS)
- 3D point projection and occlusion filtering
- Gaussian heatmap generation

Following dataset.md section 4 specifications exactly.
"""

import numpy as np
import cv2
import torch
from typing import Union, Optional, Tuple, List, Dict
import math
import random
import logging

logger = logging.getLogger(__name__)


# =====================================
# 4.1 Camera Intrinsics
# =====================================

def build_intrinsics(width: int, height: int, fx: float = None, fy: float = None,
                     cx: float = None, cy: float = None, hfov_deg: float = None) -> dict:
    """
    Build camera intrinsics from explicit fx/fy/cx/cy or derive from hfov.

    Args:
        width, height: Image dimensions
        fx, fy, cx, cy: Explicit intrinsic parameters (optional)
        hfov_deg: Horizontal field of view in degrees (optional)

    Returns:
        dict: {K: 3x3 matrix, Kinv: 3x3 inverse, fx, fy, cx, cy}
    """
    # Case 1: Explicit parameters provided
    if fx is not None and fy is not None:
        if cx is None:
            cx = width / 2.0
        if cy is None:
            cy = height / 2.0

    # Case 2: Derive from HFOV
    elif hfov_deg is not None:
        fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        fy = fx  # Assume square pixels
        cx = width / 2.0
        cy = height / 2.0

    else:
        raise ValueError("Must provide either (fx, fy) or hfov_deg")

    # Build intrinsic matrix K
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ], dtype=np.float32)

    # Compute inverse
    Kinv = np.linalg.inv(K)

    return {
        'K': K,
        'Kinv': Kinv,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


# =====================================
# 4.2 SO(3)/SE(3) Metrics for FPS
# =====================================

def so3_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    Compute SO(3) angle between two rotation matrices.

    Args:
        Ra, Rb: 3x3 rotation matrices

    Returns:
        float: Angle in radians
    """
    # R_relative = Ra.T @ Rb
    R_rel = Ra.T @ Rb

    # angle = arccos((trace(R_rel) - 1) / 2)
    trace = np.trace(R_rel)
    # Clamp to avoid numerical issues
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(angle)


def se3_pose_distance(Ta: np.ndarray, Tb: np.ndarray, alpha: float, beta: float) -> float:
    """
    Compute SE(3) pose distance between two 4x4 transformation matrices.

    Args:
        Ta, Tb: 4x4 transformation matrices [R|t; 0|1]
        alpha: Weight for translation distance
        beta: Weight for rotation distance

    Returns:
        float: Weighted pose distance
    """
    # Extract rotation and translation
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]

    # Translation distance (Euclidean)
    trans_dist = np.linalg.norm(ta - tb)

    # Rotation distance (SO(3) angle)
    rot_dist = so3_angle(Ra, Rb)

    # Weighted combination
    distance = alpha * trans_dist + beta * rot_dist

    return float(distance)


# =====================================
# 4.3 Deterministic Keyframe Sampling
# =====================================

def uniform_keyframe_indices(T: int, K: int, ref_idx: int) -> List[int]:
    """
    Select K keyframes using uniform sampling from frames [0, ref_idx].

    Args:
        T: Total sequence length
        K: Number of keyframes to select
        ref_idx: Reference frame index (inclusive upper bound)

    Returns:
        List[int]: Selected keyframe indices
    """
    if ref_idx >= T:
        ref_idx = T - 1

    available_frames = list(range(ref_idx + 1))  # [0, 1, ..., ref_idx]

    if K >= len(available_frames):
        return available_frames

    # Uniform sampling
    step = len(available_frames) / K
    indices = []
    for i in range(K):
        idx = int(i * step)
        indices.append(available_frames[idx])

    return indices


def fps_keyframe_indices(poses: List[np.ndarray], K: int, ref_idx: int,
                         alpha: float, beta: float, seed: int) -> List[int]:
    """
    Select K keyframes using Farthest Point Sampling (FPS) in SE(3) pose space.

    Args:
        poses: List of 4x4 transformation matrices
        K: Number of keyframes to select
        ref_idx: Reference frame index (inclusive upper bound)
        alpha: Translation weight in pose distance
        beta: Rotation weight in pose distance
        seed: Random seed for deterministic results

    Returns:
        List[int]: Selected keyframe indices
    """
    # Set seed for deterministic results
    random.seed(seed)
    np.random.seed(seed)

    available_indices = list(range(min(ref_idx + 1, len(poses))))

    if K >= len(available_indices):
        return available_indices

    selected = []
    remaining = available_indices.copy()

    # Start with random frame
    first_idx = remaining.pop(random.randint(0, len(remaining) - 1))
    selected.append(first_idx)

    # Iteratively select farthest frames
    for _ in range(K - 1):
        if not remaining:
            break

        max_min_dist = -1
        best_idx = None
        best_remaining_idx = None

        for i, candidate_idx in enumerate(remaining):
            # Find minimum distance to already selected frames
            min_dist = float('inf')
            for selected_idx in selected:
                dist = se3_pose_distance(
                    poses[candidate_idx], poses[selected_idx], alpha, beta
                )
                min_dist = min(min_dist, dist)

            # Select frame with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = candidate_idx
                best_remaining_idx = i

        if best_idx is not None:
            selected.append(best_idx)
            remaining.pop(best_remaining_idx)

    return sorted(selected)


# =====================================
# 4.4 Point Projection Pipeline
# =====================================

def unproject_depth_to_points(depth: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
    """
    Unproject depth map to 3D camera coordinates.

    Args:
        depth: [H, W] depth map
        Kinv: 3x3 inverse intrinsic matrix

    Returns:
        np.ndarray: [N, 3] 3D points in camera coordinates
    """
    H, W = depth.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten and create homogeneous coordinates
    u_flat = u.flatten()
    v_flat = v.flatten()
    d_flat = depth.flatten()

    # Filter valid depths (> 0)
    valid_mask = d_flat > 0
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    d_valid = d_flat[valid_mask]

    if len(d_valid) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Homogeneous pixel coordinates
    pixels_hom = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # [3, N]

    # Unproject to camera coordinates
    camera_coords = Kinv @ pixels_hom  # [3, N]
    camera_coords = camera_coords * d_valid[np.newaxis, :]  # Scale by depth

    return camera_coords.T  # [N, 3]


def world_from_cam(points_c: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """
    Transform points from camera to world coordinates.

    Args:
        points_c: [N, 3] points in camera coordinates
        T_w_c: 4x4 camera-to-world transformation matrix

    Returns:
        np.ndarray: [N, 3] points in world coordinates
    """
    if len(points_c) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Convert to homogeneous coordinates
    points_c_hom = np.hstack([points_c, np.ones((len(points_c), 1))])  # [N, 4]

    # Transform to world coordinates
    points_w_hom = (T_w_c @ points_c_hom.T).T  # [N, 4]

    return points_w_hom[:, :3]  # [N, 3]


def cam_from_world(points_w: np.ndarray, T_c_w: np.ndarray) -> np.ndarray:
    """
    Transform points from world to camera coordinates.

    Args:
        points_w: [N, 3] points in world coordinates
        T_c_w: 4x4 world-to-camera transformation matrix

    Returns:
        np.ndarray: [N, 3] points in camera coordinates
    """
    if len(points_w) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Convert to homogeneous coordinates
    points_w_hom = np.hstack([points_w, np.ones((len(points_w), 1))])  # [N, 4]

    # Transform to camera coordinates
    points_c_hom = (T_c_w @ points_w_hom.T).T  # [N, 4]

    return points_c_hom[:, :3]  # [N, 3]


def project_cam_points_to_pixels(points_c: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                                 w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D camera points to 2D pixels, filtering invalid points.

    Args:
        points_c: [N, 3] points in camera coordinates
        fx, fy, cx, cy: Camera intrinsic parameters
        w, h: Image dimensions

    Returns:
        tuple: (uv[M, 2], z[M]) - pixel coordinates and depths for valid points
    """
    if len(points_c) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Extract coordinates
    x, y, z = points_c[:, 0], points_c[:, 1], points_c[:, 2]

    # Filter points behind camera (z <= 0)
    valid_depth = z > 0
    if not np.any(valid_depth):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x_valid = x[valid_depth]
    y_valid = y[valid_depth]
    z_valid = z[valid_depth]

    # Project to image plane
    u = fx * (x_valid / z_valid) + cx
    v = fy * (y_valid / z_valid) + cy

    # Filter points outside image bounds
    valid_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u_final = u[valid_bounds]
    v_final = v[valid_bounds]
    z_final = z_valid[valid_bounds]

    # Stack u, v coordinates
    uv = np.stack([u_final, v_final], axis=1)  # [M, 2]

    return uv, z_final


# =====================================
# 4.5 Occlusion Filtering (Optional)
# =====================================

def occlusion_filter(uv: np.ndarray, z: np.ndarray, depth_ref: np.ndarray, eps: float) -> np.ndarray:
    """
    Filter occluded points by comparing projected depth with reference depth.

    Args:
        uv: [N, 2] pixel coordinates (u, v)
        z: [N] projected depths
        depth_ref: [H, W] reference depth map
        eps: Depth tolerance for occlusion check

    Returns:
        np.ndarray: [N] boolean mask - True for visible points
    """
    if len(uv) == 0:
        return np.empty((0,), dtype=bool)

    H, W = depth_ref.shape

    # Round pixel coordinates to integers
    u_int = np.round(uv[:, 0]).astype(int)
    v_int = np.round(uv[:, 1]).astype(int)

    # Check bounds
    valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    visible = np.zeros(len(uv), dtype=bool)

    if np.any(valid_bounds):
        u_valid = u_int[valid_bounds]
        v_valid = v_int[valid_bounds]
        z_valid = z[valid_bounds]

        # Get reference depths at projected locations
        ref_depths = depth_ref[v_valid, u_valid]

        # Points are visible if their depth is close to reference depth
        # (within eps tolerance)
        depth_check = np.abs(z_valid - ref_depths) <= eps

        # Also check that reference depth is valid (> 0)
        ref_valid = ref_depths > 0

        visible[valid_bounds] = depth_check & ref_valid

    return visible


# =====================================
# 4.5.5 Visibility Ratio for Keyframe Selection
# =====================================

def project_to_ref_pixels(pts_ci: np.ndarray, Ki: dict):
    """
    Project points in reference camera frame to pixel coordinates using reference intrinsics.

    This helper ensures consistent projection to reference frame across visibility evaluation
    and heatmap generation.

    Args:
        pts_ci: Points in reference camera frame [N, 3] (x, y, z)
        Ki: Reference camera intrinsics dict with 'fx', 'fy', 'cx', 'cy'

    Returns:
        xi, yi, zi: Pixel x coords, pixel y coords, depths
    """
    zi = pts_ci[:, 2]
    xi = Ki['fx'] * (pts_ci[:, 0] / zi) + Ki['cx']
    yi = Ki['fy'] * (pts_ci[:, 1] / zi) + Ki['cy']
    return xi, yi, zi


def visible_ratio_for_keyframe(depth_j: np.ndarray, Kj: dict, Ki: dict,
                               T_w_c_j: np.ndarray, T_c_ref_w: np.ndarray,
                               depth_ref: np.ndarray, ref_w: int, ref_h: int,
                               occl_eps: float, subsample: int = 4) -> float:
    """
    Compute visibility ratio of keyframe j in reference frame i.

    Returns the fraction of keyframe j's points that are visible in the reference frame.
    Uses subsampling for efficiency.

    CRITICAL FIX: Uses Ki (reference frame intrinsics) for projection to reference frame,
    not Kj (keyframe intrinsics).

    Args:
        depth_j: [H, W] depth map of keyframe j
        Kj: Intrinsics dict for keyframe j {Kinv, fx, fy, cx, cy}
        Ki: Intrinsics dict for reference frame i {fx, fy, cx, cy}
        T_w_c_j: 4x4 camera-to-world transform for keyframe j
        T_c_ref_w: 4x4 world-to-camera transform for reference frame i
        depth_ref: [H, W] reference depth map (for occlusion check)
        ref_w, ref_h: Reference frame dimensions
        occl_eps: Occlusion tolerance
        subsample: Subsample stride (e.g., 4 means sample every 4th pixel)

    Returns:
        float: Visibility ratio in [0, 1]
    """
    # Step 1: Generate sparse grid of sample points
    H, W = depth_j.shape
    ys = np.arange(0, H, subsample, dtype=np.int32)
    xs = np.arange(0, W, subsample, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')

    # Get depths at grid points
    z = depth_j[grid_y, grid_x]
    valid = z > 0

    if valid.sum() == 0:
        return 0.0

    total_sampled = valid.sum()

    # Step 2: Unproject grid points to camera coordinates
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    Kinv = Kj['Kinv']
    ones = np.ones_like(u)
    pix = np.stack([u, v, ones], axis=-1)  # [N, 3]

    rays = (Kinv @ pix.T).T  # [N, 3]
    pts_cj = rays * z[:, None]  # [N, 3]

    # Step 3: Transform to world → reference camera
    pts_w = world_from_cam(pts_cj, T_w_c_j)
    pts_ci = cam_from_world(pts_w, T_c_ref_w)

    # Step 4: Project to reference image using Ki (reference frame intrinsics)
    xi, yi, zi = project_to_ref_pixels(pts_ci, Ki)
    infront = zi > 0

    if infront.sum() == 0:
        return 0.0

    # Check bounds
    inb = (xi >= 0) & (xi < ref_w) & (yi >= 0) & (yi < ref_h)
    valid_proj = infront & inb

    if valid_proj.sum() == 0:
        return 0.0

    # Step 5: Occlusion check
    if depth_ref is not None:
        # Sample nearest pixel depth
        x0 = np.clip(np.round(xi[valid_proj]).astype(int), 0, ref_w - 1)
        y0 = np.clip(np.round(yi[valid_proj]).astype(int), 0, ref_h - 1)

        # Check occlusion: point is visible if its depth is close to reference depth
        ref_depth_at_proj = depth_ref[y0, x0]
        z_proj = zi[valid_proj]

        # Point is occluded if reference depth + eps < projected depth
        occluded = ref_depth_at_proj + occl_eps < z_proj

        # Update valid mask
        visible_count = (~occluded).sum()
    else:
        visible_count = valid_proj.sum()

    # Return visibility ratio
    return float(visible_count) / float(total_sampled)


# =====================================
# 4.6 Keyframe → Reference Integration
# =====================================

def project_keyframe_to_ref(depth_j: np.ndarray, Kj: dict, T_w_c_j: np.ndarray,
                            T_c_ref_w: np.ndarray, depth_ref: Optional[np.ndarray],
                            ref_w: int, ref_h: int, occl_eps: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project keyframe j's 3D points to reference frame coordinates.

    Args:
        depth_j: [H, W] depth map of keyframe j
        Kj: Intrinsics dict for keyframe j {Kinv, fx, fy, cx, cy}
        T_w_c_j: 4x4 world-to-camera transform for keyframe j
        T_c_ref_w: 4x4 camera-to-world transform for reference frame (inverse)
        depth_ref: [H, W] reference depth map (for occlusion check, optional)
        ref_w, ref_h: Reference frame dimensions
        occl_eps: Occlusion tolerance (optional)

    Returns:
        tuple: (uv_ref[N, 2], valid_mask[N]) - pixel coords and visibility mask
    """
    # Step 1: Unproject keyframe depth to camera coordinates
    points_c_j = unproject_depth_to_points(depth_j, Kj['Kinv'])

    if len(points_c_j) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    # Step 2: Transform to world coordinates
    points_w = world_from_cam(points_c_j, T_w_c_j)

    # Step 3: Transform to reference camera coordinates
    points_c_ref = cam_from_world(points_w, T_c_ref_w)

    # Step 4: Project to reference image coordinates
    uv_ref, z_ref = project_cam_points_to_pixels(
        points_c_ref, Kj['fx'], Kj['fy'], Kj['cx'], Kj['cy'], ref_w, ref_h
    )

    # Step 5: Apply occlusion filtering if enabled
    if depth_ref is not None and occl_eps is not None:
        visible_mask = occlusion_filter(uv_ref, z_ref, depth_ref, occl_eps)
    else:
        visible_mask = np.ones(len(uv_ref), dtype=bool)

    return uv_ref, visible_mask


# =====================================
# 4.7 Heatmap Generation
# =====================================

def heatmap_from_points(uv_ref: np.ndarray, ref_size: Tuple[int, int], hm_size: Tuple[int, int],
                        sigma_px: float) -> np.ndarray:
    """
    Generate Gaussian heatmap from 2D points.

    Args:
        uv_ref: [N, 2] pixel coordinates (u, v)
        ref_size: (width, height) of reference image
        hm_size: (width, height) of output heatmap
        sigma_px: Gaussian sigma in pixels

    Returns:
        np.ndarray: [Hm, Wm] normalized probability heatmap (sum=1, or all zeros)
    """
    Hm, Wm = hm_size[1], hm_size[0]  # Note: hm_size is (W, H), output is [H, W]

    if len(uv_ref) == 0:
        # Return all-zeros heatmap
        return np.zeros((Hm, Wm), dtype=np.float32)

    # Scale coordinates from reference size to heatmap size
    ref_w, ref_h = ref_size
    scale_u = Wm / ref_w
    scale_v = Hm / ref_h

    u_scaled = uv_ref[:, 0] * scale_u
    v_scaled = uv_ref[:, 1] * scale_v

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing='ij')

    # Initialize heatmap
    heatmap = np.zeros((Hm, Wm), dtype=np.float32)

    # Add Gaussian for each point
    for u, v in zip(u_scaled, v_scaled):
        # Gaussian centered at (u, v)
        gaussian = np.exp(-((x_grid - u)**2 + (y_grid - v)**2) / (2 * sigma_px**2))
        heatmap += gaussian

    # Normalize to probability distribution (sum = 1)
    total_sum = heatmap.sum()
    if total_sum > 0:
        heatmap = heatmap / total_sum

    return heatmap


# =====================================
# Testing and Validation
# =====================================

def test_geometric_functions():
    """Basic unit tests for geometric functions."""
    logger.info("Testing geometric functions...")

    # Test intrinsics building
    intrinsics = build_intrinsics(640, 480, hfov_deg=60.0)
    assert 'K' in intrinsics
    assert intrinsics['K'].shape == (3, 3)
    logger.info("✓ Intrinsics building test passed")

    # Test pose distance
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [1, 0, 0]  # 1m translation
    dist = se3_pose_distance(T1, T2, alpha=1.0, beta=0.1)
    assert dist > 0
    logger.info("✓ Pose distance test passed")

    # Test uniform sampling
    indices = uniform_keyframe_indices(T=10, K=3, ref_idx=8)
    assert len(indices) == 3
    assert all(0 <= idx <= 8 for idx in indices)
    logger.info("✓ Uniform sampling test passed")

    # Test depth unprojection
    depth = np.ones((10, 10), dtype=np.float32)
    Kinv = np.linalg.inv(intrinsics['K'])
    points = unproject_depth_to_points(depth, Kinv)
    assert points.shape[1] == 3
    logger.info("✓ Depth unprojection test passed")

    # Test heatmap generation
    uv = np.array([[5.0, 5.0], [15.0, 15.0]], dtype=np.float32)
    heatmap = heatmap_from_points(uv, ref_size=(32, 32), hm_size=(16, 16), sigma_px=2.0)
    assert heatmap.shape == (16, 16)
    assert abs(heatmap.sum() - 1.0) < 1e-5  # Should sum to 1
    logger.info("✓ Heatmap generation test passed")

    logger.info("All geometric function tests passed! ✅")


if __name__ == "__main__":
    # Run tests when executed directly
    logging.basicConfig(level=logging.INFO)
    test_geometric_functions()