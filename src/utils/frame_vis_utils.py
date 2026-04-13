#!/usr/bin/env python3
"""
Frame-by-frame visualization utilities for VLN heatmap evaluation.

Provides interpolation and overlay functions for generating frame-by-frame
comparisons of GT vs predicted heatmaps.
"""


import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d


def interpolate_keyframe_predictions(
    pred_keyframes: torch.Tensor,
    keyframe_indices: np.ndarray,
    total_frames: int,
    method: str = 'linear'
) -> torch.Tensor:
    """
    Interpolate K keyframe predictions to T total frames.

    Args:
        pred_keyframes: Predicted heatmaps at keyframes [K, H, W]
        keyframe_indices: Indices of keyframes in range [0, T-1], shape [K]
        total_frames: Total number of frames T
        method: Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
        Interpolated heatmaps for all T frames [T, H, W]
    """
    K, H, W = pred_keyframes.shape

    # Convert to numpy for scipy interpolation
    pred_np = pred_keyframes.cpu().numpy()

    # Initialize output
    interpolated = np.zeros((total_frames, H, W), dtype=np.float32)

    # Handle edge cases
    if K == 1:
        # Single keyframe: replicate to all frames
        interpolated[:] = pred_np[0]
        return torch.from_numpy(interpolated).to(pred_keyframes.device)

    if total_frames == K:
        # Already have all frames
        return pred_keyframes

    # Interpolate each pixel's temporal evolution
    for h in range(H):
        for w in range(W):
            pixel_values = pred_np[:, h, w]  # [K]

            # Create interpolation function
            if method == 'cubic' and K >= 4:
                kind = 'cubic'
            elif method in ('cubic', 'linear'):
                kind = 'linear'
            else:
                kind = 'nearest'
            f = interp1d(keyframe_indices, pixel_values, kind=kind,
                         bounds_error=False, fill_value='extrapolate')

            # Interpolate to all frames
            target_indices = np.arange(total_frames)
            interpolated[:, h, w] = f(target_indices)

    # Ensure non-negative (heatmaps should be non-negative)
    interpolated = np.clip(interpolated, 0, None)

    # Renormalize each frame to maintain probability distribution property
    for t in range(total_frames):
        frame_sum = interpolated[t].sum()
        if frame_sum > 0:
            interpolated[t] /= frame_sum

    return torch.from_numpy(interpolated).to(pred_keyframes.device)


def resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target image size.

    Args:
        heatmap: Original heatmap (typically 64x64)
        target_size: Target size as (width, height)

    Returns:
        Resized heatmap
    """
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply color mapping to heatmap.

    Args:
        heatmap: Normalized heatmap (values 0-1)
        colormap: OpenCV colormap constant

    Returns:
        Colored heatmap as BGR image
    """
    # Normalize to 0-255 range
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    return colored_heatmap


def create_single_heatmap_overlay(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    threshold: float = 0.05,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay single heatmap on RGB image.

    Args:
        rgb_image: RGB image (H, W, 3) in BGR format
        heatmap: Heatmap array (Hm, Wm)
        alpha: Transparency of heatmap overlay (0-1)
        threshold: Minimum heatmap value to display (0-1)
        colormap: OpenCV colormap to use

    Returns:
        Overlaid image in BGR format
    """
    # Get image dimensions
    h, w = rgb_image.shape[:2]

    # Resize heatmap if needed
    if heatmap.shape != (h, w):
        heatmap_resized = resize_heatmap(heatmap, (w, h))
    else:
        heatmap_resized = heatmap.copy()

    # Apply colormap to the heatmap (already in 0-1 range)
    colored_heatmap = apply_colormap(heatmap_resized, colormap)

    # Create a mask for significant heatmap values only
    mask = heatmap_resized > threshold
    mask_3d = np.stack([mask, mask, mask], axis=-1)  # Expand to 3 channels

    # Blend entire images first
    blended = cv2.addWeighted(rgb_image, 1 - alpha, colored_heatmap, alpha, 0)

    # Use mask to selectively apply blended regions
    overlaid = np.where(mask_3d, blended, rgb_image)

    return overlaid.astype(np.uint8)


def create_frame_by_frame_overlay(
    rgb_frame: np.ndarray,
    heatmap_history: np.ndarray | None,
    heatmap_future: np.ndarray | None,
    alpha: float = 0.5,
    threshold: float = 0.05,
    colormap_history: int = cv2.COLORMAP_WINTER,  # Cyan
    colormap_future: int = cv2.COLORMAP_HOT       # Red/Yellow
) -> np.ndarray:
    """
    Create dual-heatmap overlay on single RGB frame.

    Overlays both history (cyan) and future (red) heatmaps on the same frame.
    Based on visualize_heatmaps.py:164-218 implementation.

    Args:
        rgb_frame: RGB image (H, W, 3) in BGR format
        heatmap_history: History heatmap (Hm, Wm), can be None
        heatmap_future: Future heatmap (Hm, Wm), can be None
        alpha: Transparency of heatmap overlay (0-1)
        threshold: Minimum heatmap value to display (0-1)
        colormap_history: Colormap for history (default: WINTER/cyan)
        colormap_future: Colormap for future (default: HOT/red-yellow)

    Returns:
        Overlaid image in BGR format
    """
    h, w = rgb_frame.shape[:2]
    result = rgb_frame.copy()

    # Overlay future first (background layer)
    if heatmap_future is not None:
        if heatmap_future.shape != (h, w):
            heatmap_future_resized = resize_heatmap(heatmap_future, (w, h))
        else:
            heatmap_future_resized = heatmap_future.copy()

        # Apply colormap
        colored_future = apply_colormap(heatmap_future_resized, colormap_future)

        # Create mask
        mask_future = heatmap_future_resized > threshold

        if mask_future.any():
            mask_future_3d = np.stack([mask_future, mask_future, mask_future], axis=-1)
            blended_future = cv2.addWeighted(result, 1 - alpha, colored_future, alpha, 0)
            result = np.where(mask_future_3d, blended_future, result)

    # Overlay history on top (foreground layer)
    if heatmap_history is not None:
        if heatmap_history.shape != (h, w):
            heatmap_history_resized = resize_heatmap(heatmap_history, (w, h))
        else:
            heatmap_history_resized = heatmap_history.copy()

        # Apply colormap
        colored_history = apply_colormap(heatmap_history_resized, colormap_history)

        # Create mask
        mask_history = heatmap_history_resized > threshold

        if mask_history.any():
            mask_history_3d = np.stack([mask_history, mask_history, mask_history], axis=-1)
            blended_history = cv2.addWeighted(result, 1 - alpha, colored_history, alpha, 0)
            result = np.where(mask_history_3d, blended_history, result)

    return result.astype(np.uint8)


def create_difference_map(
    pred_hm: np.ndarray,
    gt_hm: np.ndarray,
    colormap: int = cv2.COLORMAP_HOT
) -> np.ndarray:
    """
    Generate absolute difference heatmap between prediction and ground truth.

    Args:
        pred_hm: Predicted heatmap (Hm, Wm)
        gt_hm: Ground truth heatmap (Hm, Wm)
        colormap: OpenCV colormap to use (default: REDS)

    Returns:
        Colored difference map as BGR image
    """
    # Compute absolute difference
    diff = np.abs(pred_hm - gt_hm)

    # Normalize to 0-1 range
    if diff.max() > 0:
        diff = diff / diff.max()

    # Apply colormap
    colored_diff = apply_colormap(diff, colormap)

    return colored_diff
