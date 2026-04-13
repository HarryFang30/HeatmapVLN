"""
Heatmap geometry utilities — pinhole projection and Gaussian rendering.

Pure-numpy functions with no internal project dependencies; used by
both CPU-side dataset heatmap generation and GPU heatmap verification.
"""

import math
from typing import List, Optional, Tuple

import numpy as np


def project_point_pinhole(
    p_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int
) -> Optional[Tuple[float, float, float]]:
    """Project a camera-space 3D point to pixel coordinates.

    Habitat convention: X right, Y up, -Z forward.

    Returns:
        (u, v, z_depth) or None if behind camera / out of frame.
    """
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

    if z >= -0.1:
        return None

    z_depth = -z

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x / z_depth + cx
    v = fy * (-y) / z_depth + cy

    if not (0 <= u < width and 0 <= v < height):
        return None

    return float(u), float(v), float(z_depth)


def compute_adaptive_sigma_pinhole(
    z_depth: float,
    fx: float,
    object_size_3d: float = 0.5,
    heatmap_width: int = 64,
    img_width: int = 640,
    min_sigma: float = 0.8,
    max_sigma: float = 6.0
) -> float:
    """Compute depth-adaptive Gaussian sigma under perspective projection."""
    if z_depth <= 0.1:
        return float(max_sigma)

    projected_size_img = object_size_3d * fx / z_depth
    scale = heatmap_width / img_width
    projected_size_hm = projected_size_img * scale
    sigma = projected_size_hm / 3.0
    sigma = np.clip(sigma, min_sigma, max_sigma)

    return float(sigma)


def draw_gaussian_point(
    heatmap: np.ndarray,
    center: Tuple[float, float],
    sigma: float,
    peak_value: float = 1.0,
    use_max: bool = True,
) -> None:
    """Draw a Gaussian blob on *heatmap* in-place."""
    H, W = heatmap.shape
    u, v = center

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x_min = max(0, int(np.floor(u - radius)))
    x_max = min(W, int(np.ceil(u + radius)))
    y_min = max(0, int(np.floor(v - radius)))
    y_max = min(H, int(np.ceil(v + radius)))

    if x_min >= x_max or y_min >= y_max:
        return

    xs = np.arange(x_min, x_max, dtype=np.float32)
    ys = np.arange(y_min, y_max, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)

    blob = peak_value * np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

    roi = heatmap[y_min:y_max, x_min:x_max]
    if use_max:
        np.maximum(roi, blob, out=roi)
    else:
        np.add(roi, blob, out=roi)


def compute_history_heatmap(
    history_poses: List[np.ndarray],
    current_pose: np.ndarray,
    current_depth: Optional[np.ndarray],
    hm_size: Tuple[int, int] = (64, 64),
    img_size: Tuple[int, int] = (640, 480),
    K: Optional[np.ndarray] = None,
    depth_normalize: bool = True,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    occlusion_tolerance: float = 0.5,
    max_visible_distance: float = 15.0,
    use_max_merge: bool = True,
    use_distance_decay: bool = True,
    distance_decay_ref: float = 5.0,
    min_peak_value: float = 0.7,
) -> Tuple[np.ndarray, int]:
    """Render history-camera positions as a Gaussian heatmap on the current frame.

    Returns:
        heatmap: [Hm, Wm] float32 in [0, 1]
        visibility_count: number of visible history frames
    """
    Hm, Wm = hm_size
    img_w, img_h = img_size

    heatmap = np.zeros((Hm, Wm), dtype=np.float32)
    visibility_count = 0

    if K is None:
        hfov_rad = math.radians(90.0)
        fx = img_w / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    else:
        K = np.array(K, dtype=np.float32)

    fx = K[0, 0]

    T_current = np.array(current_pose, dtype=np.float32)
    T_current_inv = np.linalg.inv(T_current)

    depth_plane = None
    if current_depth is not None:
        if current_depth.ndim == 3 and current_depth.shape[-1] == 1:
            depth_plane = current_depth[:, :, 0]
        else:
            depth_plane = current_depth

    for hist_pose in history_poses:
        T_hist = np.array(hist_pose, dtype=np.float32)

        hist_center_world = np.array([
            T_hist[0, 3], T_hist[1, 3], T_hist[2, 3], 1.0
        ], dtype=np.float32)

        p_cam = T_current_inv @ hist_center_world

        distance = float(np.linalg.norm(p_cam[:3]))
        if distance < 1e-4 or distance > max_visible_distance:
            continue

        projection = project_point_pinhole(p_cam, K, img_w, img_h)
        if projection is None:
            continue
        u, v, z_depth = projection

        if depth_plane is not None:
            depth_h, depth_w = depth_plane.shape
            u_d = u * (depth_w / img_w)
            v_d = v * (depth_h / img_h)
            u_int = int(np.clip(u_d, 0, depth_w - 1))
            v_int = int(np.clip(v_d, 0, depth_h - 1))
            observed_depth = float(depth_plane[v_int, u_int])

            if depth_normalize:
                observed_depth = depth_min + observed_depth * (depth_max - depth_min)

            if not np.isfinite(observed_depth) or observed_depth <= 0:
                continue
            if observed_depth < z_depth - occlusion_tolerance:
                continue

        sigma = compute_adaptive_sigma_pinhole(
            z_depth=z_depth,
            fx=fx,
            object_size_3d=1.5,
            heatmap_width=Wm,
            img_width=img_w,
            min_sigma=4.0,
            max_sigma=8.0,
        )

        if use_distance_decay:
            decay_factor = 1.0 / (1.0 + distance / distance_decay_ref)
            peak_value = min_peak_value + (1.0 - min_peak_value) * decay_factor
        else:
            peak_value = 1.0

        u_hm = u * Wm / img_w
        v_hm = v * Hm / img_h

        draw_gaussian_point(
            heatmap, (u_hm, v_hm), sigma,
            peak_value=peak_value,
            use_max=use_max_merge
        )
        visibility_count += 1

    return heatmap, visibility_count
