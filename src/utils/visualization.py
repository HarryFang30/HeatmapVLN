#!/usr/bin/env python3
"""
Visualization utilities for VLN heatmap evaluation.

Provides grid generation and thumbnail creation for frame-by-frame comparisons.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from .frame_vis_utils import (
    create_frame_by_frame_overlay,
    create_single_heatmap_overlay,
    create_difference_map
)


def annotate_frame(
    image: np.ndarray,
    frame_idx: int,
    validity_hist: bool,
    validity_fut: bool,
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """
    Annotate frame with index and validity status.

    Args:
        image: Image to annotate (BGR format)
        frame_idx: Frame index
        validity_hist: History validity (True/False)
        validity_fut: Future validity (True/False)
        font_scale: Font scale for text
        thickness: Text thickness

    Returns:
        Annotated image
    """
    annotated = image.copy()

    # Prepare text
    frame_text = f"Frame {frame_idx}"
    valid_hist_icon = "Y" if validity_hist else "N"
    valid_fut_icon = "Y" if validity_fut else "N"
    valid_text = f"H:{valid_hist_icon} F:{valid_fut_icon}"

    # Position
    text_pos = (10, 30)
    valid_pos = (10, 60)

    # Draw text with outline (white outline, black text)
    cv2.putText(annotated, frame_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, frame_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.putText(annotated, valid_text, valid_pos, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale * 0.8, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, valid_text, valid_pos, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale * 0.8, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated


def create_comparison_grid(
    frames: torch.Tensor,
    gt_hm_hist: torch.Tensor,
    gt_hm_fut: torch.Tensor,
    pred_hm_hist: torch.Tensor,
    pred_hm_fut: torch.Tensor,
    gt_val_hist: torch.Tensor,
    gt_val_fut: torch.Tensor,
    save_dir: Path,
    overlay_mode: str = 'dual',
    max_frames: int = 16,
    alpha: float = 0.5,
    threshold: float = 0.05
) -> Dict[str, str]:
    """
    Generate comparison grids based on overlay mode.

    Args:
        frames: RGB frames [T, 3, H, W]
        gt_hm_hist: GT history heatmaps [T, Hm, Wm]
        gt_hm_fut: GT future heatmaps [T, Hm, Wm]
        pred_hm_hist: Predicted history heatmaps [T, Hm, Wm] (interpolated)
        pred_hm_fut: Predicted future heatmaps [T, Hm, Wm] (interpolated)
        gt_val_hist: GT history validity [T]
        gt_val_fut: GT future validity [T]
        save_dir: Directory to save grids
        overlay_mode: 'dual', 'separate', or 'full-separate'
        max_frames: Maximum frames to show
        alpha: Heatmap overlay transparency
        threshold: Minimum heatmap value to display

    Returns:
        Dictionary of generated file paths (relative to eval_vis root)
    """
    T = min(frames.shape[0], max_frames)
    paths = {}

    # Convert to numpy and BGR
    frames_np = []
    for t in range(T):
        frame = frames[t].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames_np.append(frame_bgr)

    gt_hm_hist_np = gt_hm_hist[:T].cpu().numpy()
    gt_hm_fut_np = gt_hm_fut[:T].cpu().numpy()
    pred_hm_hist_np = pred_hm_hist[:T].cpu().numpy()
    pred_hm_fut_np = pred_hm_fut[:T].cpu().numpy()
    gt_val_hist_np = gt_val_hist[:T].cpu().numpy() > 0.5
    gt_val_fut_np = gt_val_fut[:T].cpu().numpy() > 0.5

    if overlay_mode == 'dual':
        # Generate 4-column grid (RGB | GT dual | Pred dual | Diff)
        grid_path = save_dir / 'grid_comparison.png'
        _create_dual_grid(
            frames_np, gt_hm_hist_np, gt_hm_fut_np,
            pred_hm_hist_np, pred_hm_fut_np,
            gt_val_hist_np, gt_val_fut_np,
            grid_path, alpha, threshold
        )
        paths['dual'] = str(grid_path.relative_to(save_dir.parent.parent))

    elif overlay_mode == 'separate':
        # Generate two separate grids
        hist_path = save_dir / 'grid_comparison_history.png'
        fut_path = save_dir / 'grid_comparison_future.png'

        _create_single_type_grid(
            frames_np, gt_hm_hist_np, pred_hm_hist_np,
            gt_val_hist_np, hist_path, 'History',
            alpha, threshold, cv2.COLORMAP_WINTER
        )
        _create_single_type_grid(
            frames_np, gt_hm_fut_np, pred_hm_fut_np,
            gt_val_fut_np, fut_path, 'Future',
            alpha, threshold, cv2.COLORMAP_HOT
        )

        paths['history'] = str(hist_path.relative_to(save_dir.parent.parent))
        paths['future'] = str(fut_path.relative_to(save_dir.parent.parent))

    elif overlay_mode == 'full-separate':
        # Generate 7-column grid (RGB | Hist GT | Hist Pred | Hist Diff | Fut GT | Fut Pred | Fut Diff)
        grid_path = save_dir / 'grid_comparison_full.png'
        _create_full_separate_grid(
            frames_np, gt_hm_hist_np, gt_hm_fut_np,
            pred_hm_hist_np, pred_hm_fut_np,
            gt_val_hist_np, gt_val_fut_np,
            grid_path, alpha, threshold
        )
        paths['full'] = str(grid_path.relative_to(save_dir.parent.parent))

    return paths


def _create_dual_grid(
    frames_np, gt_hm_hist, gt_hm_fut, pred_hm_hist, pred_hm_fut,
    gt_val_hist, gt_val_fut, save_path, alpha, threshold
):
    """Create 4-column grid with dual overlays."""
    T = len(frames_np)
    rows = []

    for t in range(T):
        frame = frames_np[t]

        # Column 1: Annotated RGB frame
        col1 = annotate_frame(frame, t, gt_val_hist[t], gt_val_fut[t])

        # Column 2: GT dual overlay
        col2 = create_frame_by_frame_overlay(
            frame, gt_hm_hist[t], gt_hm_fut[t],
            alpha, threshold
        )

        # Column 3: Pred dual overlay
        col3 = create_frame_by_frame_overlay(
            frame, pred_hm_hist[t], pred_hm_fut[t],
            alpha, threshold
        )

        # Column 4: Difference map (average of history and future diffs)
        # Resize difference maps to match frame dimensions
        h, w = frame.shape[:2]
        diff_hist = create_difference_map(pred_hm_hist[t], gt_hm_hist[t])
        diff_hist = cv2.resize(diff_hist, (w, h), interpolation=cv2.INTER_LINEAR)
        diff_fut = create_difference_map(pred_hm_fut[t], gt_hm_fut[t])
        diff_fut = cv2.resize(diff_fut, (w, h), interpolation=cv2.INTER_LINEAR)
        col4 = cv2.addWeighted(diff_hist, 0.5, diff_fut, 0.5, 0)

        # Concatenate columns
        row = np.hstack([col1, col2, col3, col4])
        rows.append(row)

    # Concatenate rows
    grid = np.vstack(rows)
    cv2.imwrite(str(save_path), grid)


def _create_single_type_grid(
    frames_np, gt_hm, pred_hm, gt_val, save_path, label,
    alpha, threshold, colormap
):
    """Create 4-column grid for single heatmap type (history or future)."""
    T = len(frames_np)
    rows = []

    for t in range(T):
        frame = frames_np[t]

        # Column 1: Annotated RGB frame
        col1 = annotate_frame(frame, t, gt_val[t], gt_val[t])

        # Column 2: GT overlay
        col2 = create_single_heatmap_overlay(
            frame, gt_hm[t], alpha, threshold, colormap
        )

        # Column 3: Pred overlay
        col3 = create_single_heatmap_overlay(
            frame, pred_hm[t], alpha, threshold, colormap
        )

        # Column 4: Difference map (resize to match frame dimensions)
        h, w = frame.shape[:2]
        col4 = create_difference_map(pred_hm[t], gt_hm[t])
        col4 = cv2.resize(col4, (w, h), interpolation=cv2.INTER_LINEAR)

        # Concatenate columns
        row = np.hstack([col1, col2, col3, col4])
        rows.append(row)

    # Concatenate rows
    grid = np.vstack(rows)
    cv2.imwrite(str(save_path), grid)


def _create_full_separate_grid(
    frames_np, gt_hm_hist, gt_hm_fut, pred_hm_hist, pred_hm_fut,
    gt_val_hist, gt_val_fut, save_path, alpha, threshold
):
    """Create 7-column grid with full separation."""
    T = len(frames_np)
    rows = []

    for t in range(T):
        frame = frames_np[t]

        # Column 1: Annotated RGB frame
        col1 = annotate_frame(frame, t, gt_val_hist[t], gt_val_fut[t])

        # Get frame dimensions for resizing difference maps
        h, w = frame.shape[:2]

        # Columns 2-4: History (GT | Pred | Diff)
        col2_hist = create_single_heatmap_overlay(
            frame, gt_hm_hist[t], alpha, threshold, cv2.COLORMAP_WINTER
        )
        col3_hist = create_single_heatmap_overlay(
            frame, pred_hm_hist[t], alpha, threshold, cv2.COLORMAP_WINTER
        )
        col4_hist = create_difference_map(pred_hm_hist[t], gt_hm_hist[t])
        col4_hist = cv2.resize(col4_hist, (w, h), interpolation=cv2.INTER_LINEAR)

        # Columns 5-7: Future (GT | Pred | Diff)
        col5_fut = create_single_heatmap_overlay(
            frame, gt_hm_fut[t], alpha, threshold, cv2.COLORMAP_HOT
        )
        col6_fut = create_single_heatmap_overlay(
            frame, pred_hm_fut[t], alpha, threshold, cv2.COLORMAP_HOT
        )
        col7_fut = create_difference_map(pred_hm_fut[t], gt_hm_fut[t])
        col7_fut = cv2.resize(col7_fut, (w, h), interpolation=cv2.INTER_LINEAR)

        # Concatenate columns
        row = np.hstack([col1, col2_hist, col3_hist, col4_hist,
                        col5_fut, col6_fut, col7_fut])
        rows.append(row)

    # Concatenate rows
    grid = np.vstack(rows)
    cv2.imwrite(str(save_path), grid)


def create_thumbnail(
    rgb_frame: np.ndarray,
    gt_hm_hist: Optional[np.ndarray],
    gt_hm_fut: Optional[np.ndarray],
    save_path: Path,
    size: Tuple[int, int] = (256, 256),
    alpha: float = 0.5,
    threshold: float = 0.05
) -> None:
    """
    Generate thumbnail from first frame with GT overlay.

    Args:
        rgb_frame: RGB frame [H, W, 3] (RGB format, not BGR)
        gt_hm_hist: GT history heatmap [Hm, Wm], can be None
        gt_hm_fut: GT future heatmap [Hm, Wm], can be None
        save_path: Path to save thumbnail
        size: Thumbnail size (width, height)
        alpha: Overlay transparency
        threshold: Minimum heatmap value to display
    """
    # Convert RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Create dual overlay
    overlay = create_frame_by_frame_overlay(
        frame_bgr, gt_hm_hist, gt_hm_fut, alpha, threshold
    )

    # Resize to thumbnail size
    thumbnail = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)

    # Save
    cv2.imwrite(str(save_path), thumbnail)
