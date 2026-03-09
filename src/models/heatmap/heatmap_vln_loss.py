"""
HeatmapVLN Loss
=================

Simplified loss for navigation-oriented heatmap prediction:
  1. Visibility BCE       — classify whether a history view is visible
  2. Coordinate loss      — directly supervise peak location (soft-argmax)
  3. Quality Focal Loss   — per-pixel heatmap regression with focal weighting

The model outputs sigmoid-activated heatmaps in (0, 1).  QFL provides:
  - 5.5× stronger false-positive gradient than L2  (BCE vs L2 at pred=0.9)
  - Automatic hard-example focus  (focal scale → 0 for correct pixels)
  - Unified push-up + push-down in a single per-pixel loss

Reference: Quality Focal Loss — Generalized Focal Loss (Li et al., NeurIPS 2020)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapVLNLoss(nn.Module):
    """
    Combined loss for HeatmapVLN.

    Args:
        lambda_vis: weight for visibility BCE.
        lambda_coord: weight for positive-sample coordinate loss.
        lambda_kl: (unused, kept for config compatibility).
        lambda_neg: (unused, kept for config compatibility).
        lambda_peak: weight for Quality Focal Loss on heatmaps.
        temperature: soft-argmax temperature (fixed, no annealing recommended).
        heatmap_size: expected heatmap resolution.
        qfl_beta: focal exponent for QFL (default 2.0).
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_coord: float = 1.0,
        lambda_kl: float = 0.0,
        lambda_neg: float = 0.0,
        lambda_peak: float = 1.0,
        temperature: float = 1.0,
        heatmap_size: Tuple[int, int] = (64, 64),
        qfl_beta: float = 2.0,
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_peak = lambda_peak
        self.temperature = temperature
        self.qfl_beta = qfl_beta
        self.heatmap_size = tuple(int(v) for v in heatmap_size)

        height, width = self.heatmap_size
        coords_y, coords_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("coords_x", coords_x, persistent=False)
        self.register_buffer("coords_y", coords_y, persistent=False)

    def _validate_heatmap_shape(self, heatmaps: torch.Tensor) -> None:
        if heatmaps.ndim not in (4, 5):
            raise ValueError(
                f"Expected heatmaps with shape (N_hist, 4, H, W) or (B, N_hist, 4, H, W), got {tuple(heatmaps.shape)}"
            )
        if heatmaps.shape[-3] != 4:
            raise ValueError(
                f"Expected 4 view channels before spatial dims, got shape {tuple(heatmaps.shape)}"
            )
        actual_size = tuple(int(v) for v in heatmaps.shape[-2:])
        if actual_size != self.heatmap_size:
            raise ValueError(
                f"Heatmap size mismatch: expected {self.heatmap_size}, got {actual_size}"
            )

    def _flatten_inputs(
        self,
        pred_vis: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_heatmap_shape(pred_heatmaps)
        self._validate_heatmap_shape(gt_heatmaps)

        pred_vis = pred_vis.reshape(-1, pred_vis.shape[-1])
        gt_vis = gt_vis.reshape(-1, gt_vis.shape[-1])
        pred_heatmaps = pred_heatmaps.reshape(-1, pred_heatmaps.shape[-3], pred_heatmaps.shape[-2], pred_heatmaps.shape[-1])
        gt_heatmaps = gt_heatmaps.reshape(-1, gt_heatmaps.shape[-3], gt_heatmaps.shape[-2], gt_heatmaps.shape[-1])

        if pred_vis.shape != gt_vis.shape:
            raise ValueError(
                f"Visibility shape mismatch: pred {tuple(pred_vis.shape)} vs gt {tuple(gt_vis.shape)}"
            )
        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(
                f"Heatmap shape mismatch: pred {tuple(pred_heatmaps.shape)} vs gt {tuple(gt_heatmaps.shape)}"
            )
        if pred_heatmaps.shape[:2] != pred_vis.shape:
            raise ValueError(
                f"Visibility/heatmap leading shape mismatch: vis {tuple(pred_vis.shape)} vs heatmaps {tuple(pred_heatmaps.shape)}"
            )

        return pred_vis, pred_heatmaps, gt_vis, gt_heatmaps

    def set_temperature(self, temperature: float) -> None:
        """Update soft-argmax temperature during training."""
        self.temperature = float(temperature)

    def soft_argmax_coord_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract differentiable peak coordinates with soft-argmax and
        penalize Euclidean distance in pixel space.

        pred/target: [K, H, W]
        """
        pred = pred.float()
        target = target.float()
        num_samples = pred.shape[0]

        pred_logits = (pred * self.temperature).reshape(num_samples, -1)
        target_logits = (target * self.temperature).reshape(num_samples, -1)

        pred_weights = F.softmax(pred_logits, dim=-1).reshape_as(pred)
        target_weights = F.softmax(target_logits, dim=-1).reshape_as(target)

        pred_cx = (pred_weights * self.coords_x).sum(dim=(-2, -1))
        pred_cy = (pred_weights * self.coords_y).sum(dim=(-2, -1))
        target_cx = (target_weights * self.coords_x).sum(dim=(-2, -1))
        target_cy = (target_weights * self.coords_y).sum(dim=(-2, -1))

        coord_dist = torch.sqrt(
            (pred_cx - target_cx).square()
            + (pred_cy - target_cy).square()
            + 1e-6
        )
        return coord_dist.mean()

    def quality_focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quality Focal Loss for continuous-valued heatmap GT.

        QFL(p, y) = |p - y|^β × BCE(p, y)

        - False positive  (p=0.9, y=0):  scale=0.81, loss=1.87, grad≈12
        - Missed detection (p=0.01, y=0.9): scale=0.79, loss=3.27, grad≈71
        - Correct pred     (p≈y):          scale≈0,    loss≈0             ← no wasted gradient
        - Easy negative    (p=0.01, y=0):  scale≈0,    loss≈0             ← no wasted gradient

        pred/target: arbitrary shape, values in (0, 1).
        Returns: scalar loss normalized per positive pixel.
        """
        pred = pred.float().clamp(1e-6, 1 - 1e-6)
        target = target.float()

        scale = (pred - target).abs().pow(self.qfl_beta)

        # AMP-safe BCE: convert sigmoid output to logits
        pred_logits = torch.logit(pred)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none"
        )

        focal = scale * bce

        num_pos = (target >= 0.01).float().sum().clamp(min=1)
        return focal.sum() / num_pos

    def forward(
        self,
        pred_vis: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_vis:      ``(N_hist, 4)`` or ``(B, N_hist, 4)``.
            pred_heatmaps: ``(N_hist, 4, H, W)`` or ``(B, N_hist, 4, H, W)``.
            gt_vis:        ``(N_hist, 4)`` or ``(B, N_hist, 4)``.
            gt_heatmaps:   ``(N_hist, 4, H, W)`` or ``(B, N_hist, 4, H, W)``.
        """
        device = pred_vis.device
        pred_vis, pred_heatmaps, gt_vis, gt_heatmaps = self._flatten_inputs(
            pred_vis,
            pred_heatmaps,
            gt_vis,
            gt_heatmaps,
        )

        # (1) Visibility loss
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_vis.float())

        # (2) Coordinate loss for position refinement
        pos_mask = gt_vis.bool()
        if pos_mask.any() and self.lambda_coord > 0:
            pred_pos = pred_heatmaps[pos_mask]
            gt_pos = gt_heatmaps[pos_mask]
            coord_loss = self.soft_argmax_coord_loss(pred_pos, gt_pos)
        else:
            coord_loss = torch.tensor(0.0, device=device)

        # (3) Quality Focal Loss — unified per-pixel heatmap regression.
        #     Replaces the old separate neg_loss (L2) + peak_loss (L1).
        qfl_loss = self.quality_focal_loss(pred_heatmaps, gt_heatmaps)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_coord * coord_loss
            + self.lambda_peak * qfl_loss
        )

        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "kl_loss": torch.tensor(0.0, device=device),
            "neg_loss": torch.tensor(0.0, device=device),
            "peak_loss": qfl_loss.detach(),
        }
