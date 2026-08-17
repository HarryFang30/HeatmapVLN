"""Convert AMB3R/DA3 camera trajectories into HeatmapVLN pose tokens.

AMB3R's DA3 wrapper returns camera-to-world matrices whose local camera frame
uses the OpenCV convention (right, down, forward).  The frozen HeatmapVLN head
was trained from Habitat/OpenGL camera-to-world matrices (right, up, back) and
expects ``[forward_m, left_m, cos(yaw), sin(yaw)]``.  This module performs only
the fixed convention change and then deliberately delegates the final pose
encoding to :func:`src.data.trajectory_utils.compute_history_rel_poses`, the
same function used for GT training data.

No GT pose is read by these conversion functions.  An optional translation
scale is a single train-split constant and defaults to one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.data.trajectory_utils import compute_history_rel_poses


_CV_TO_HABITAT = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _validated_c2w(poses: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float32)
    if array.ndim < 3 or array.shape[-2:] != (4, 4):
        raise ValueError(f"Expected c2w poses with shape (..., 4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Camera trajectory contains NaN or infinite values")
    bottom = array[..., 3, :]
    expected = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        bottom.shape,
    )
    if not np.allclose(bottom, expected, rtol=0.0, atol=1e-4):
        raise ValueError("Camera trajectory contains non-rigid homogeneous matrices")
    return array


def opencv_c2w_to_habitat_c2w(
    poses_c2w: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    """Change the local camera basis from OpenCV to Habitat/OpenGL.

    AMB3R's global world frame is an arbitrary gauge and must not be aligned
    with GT.  Right multiplication changes only the camera basis; that global
    gauge then cancels inside ``current_c2w^-1 @ history_c2w``.  The input and
    output shapes are identical.
    """

    poses = _validated_c2w(poses_c2w)
    return np.matmul(poses, _CV_TO_HABITAT).astype(np.float32, copy=False)


def history_rel_poses_from_amb3r(
    poses_c2w_opencv: np.ndarray | Sequence[np.ndarray],
    history_indices: Sequence[int] | np.ndarray,
    current_index: int,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Build the frozen heatmap head's ``[K, 4]`` trajectory input.

    Args:
        poses_c2w_opencv: Continuous AMB3R c2w trajectory ``[T,4,4]``.
        history_indices: The existing HeatmapVLN K-history frame indices.
        current_index: Current frame index in the same continuous trajectory.
        translation_scale: Optional fixed train-split scalar.  It is never
            fitted per episode and defaults to native AMB3R scale.
    """

    poses_cv = _validated_c2w(poses_c2w_opencv)
    if poses_cv.ndim != 3:
        raise ValueError(f"Expected one trajectory [T,4,4], got {poses_cv.shape}")
    if not np.isfinite(translation_scale) or float(translation_scale) <= 0.0:
        raise ValueError("translation_scale must be a finite positive scalar")

    total = int(poses_cv.shape[0])
    current = int(current_index)
    indices = np.asarray(history_indices, dtype=np.int64).reshape(-1)
    if current < 0 or current >= total:
        raise IndexError(f"current_index={current} is outside trajectory length {total}")
    if indices.size and ((indices < 0).any() or (indices >= total).any()):
        raise IndexError("history_indices contain an index outside the trajectory")
    if indices.size and (indices > current).any():
        raise ValueError(
            "Every heatmap history index must be no later than current_index"
        )

    poses_habitat = opencv_c2w_to_habitat_c2w(poses_cv)
    rel = compute_history_rel_poses(
        [poses_habitat[int(index)] for index in indices],
        poses_habitat[current],
        camera_forward_axis="-z",
    )
    rel[:, :2] *= float(translation_scale)
    return rel.astype(np.float32, copy=False)


def fit_global_translation_scale(
    predicted_rel_poses: np.ndarray,
    gt_rel_poses: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    eps: float = 1e-8,
) -> float:
    """Fit one least-squares translation scale over a train-set collection.

    Both pose arrays use the heatmap ``[...,4]`` contract.  Only the first two
    translation fields participate; rotation is intentionally untouched.
    """

    predicted = np.asarray(predicted_rel_poses, dtype=np.float64)
    target = np.asarray(gt_rel_poses, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 4:
        raise ValueError(
            "predicted_rel_poses and gt_rel_poses must share shape (..., 4); "
            f"got {predicted.shape} and {target.shape}"
        )
    finite = np.isfinite(predicted).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != finite.shape:
            raise ValueError(f"valid_mask shape {mask.shape} does not match {finite.shape}")
        finite &= mask
    pred_xy = predicted[..., :2][finite]
    gt_xy = target[..., :2][finite]
    denominator = float(np.sum(pred_xy * pred_xy))
    if denominator <= float(eps):
        raise ValueError("Cannot fit scale: predicted translations have zero energy")
    scale = float(np.sum(pred_xy * gt_xy) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Fitted translation scale is not positive: {scale}")
    return scale
