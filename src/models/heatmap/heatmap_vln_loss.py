"""
HeatmapVLN Loss
=================

Loss for navigation-oriented heatmap prediction:
  1. Visibility BCE        — classify whether a history view is visible
  2. Softmax Cross-Entropy — pixel-space classification on visible views
  3. Negative BCE          — push invisible-view heatmaps toward zero
  4. Coordinate loss       — low-weight auxiliary for peak position refinement

Design rationale — two distinct sub-problems require different losses:

  "Where is it?"  (visible views, gt_vis > 0)
      Softmax CE over H×W = 4096 pixel classes.  GT L1-normalized to a
      probability distribution.  Pixel competition prevents false positives
      *within* visible views; no-collapse guarantee; clean gradients.

  "It's NOT here." (invisible views, gt_vis = 0)
      Per-pixel BCE pushing all sigmoid outputs toward 0.
      Gradient = sigmoid(z) = pred: strong push-down on false positives
      (grad 0.9 at pred=0.9), negligible on already-correct pixels
      (grad 0.01 at pred=0.01).

Softmax CE alone cannot suppress false positives on invisible views because
those views are skipped (GT is all-zero, cannot be normalised).  The negative
BCE provides direct h_loc supervision on these views — defense in depth beyond
the visibility gate.
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
        lambda_coord: weight for soft-argmax coordinate loss (auxiliary).
        lambda_kl: (unused, kept for config compatibility).
        lambda_neg: weight for invisible-view negative suppression BCE.
        lambda_peak: weight for softmax cross-entropy on visible-view heatmaps.
        temperature: soft-argmax temperature (only affects coord_loss).
        heatmap_size: expected heatmap resolution (H, W).
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_coord: float = 0.2,
        lambda_kl: float = 0.0,
        lambda_neg: float = 1.0,
        lambda_peak: float = 1.0,
        temperature: float = 1.0,
        heatmap_size: Tuple[int, int] = (64, 64),
        **kwargs,
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_neg = lambda_neg
        self.lambda_peak = lambda_peak
        self.temperature = temperature
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

    def softmax_ce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Softmax cross-entropy treating each H×W heatmap as a 4096-class
        classification problem.

        L = -Σ_x  gt_prob(x) · log( softmax(logits)(x) )

        where gt_prob = gt / Σgt  (L1-normalised to a probability distribution)
        and logits = logit(sigmoid_output) = raw network output before sigmoid.

        pred:   [K, H, W]  sigmoid-activated values in (0, 1)
        target: [K, H, W]  GT heatmap values in [0, 1]
        Returns: scalar CE averaged over K views.
        """
        pred = pred.float()
        target = target.float()
        K = pred.shape[0]

        logits = torch.logit(pred.clamp(1e-6, 1 - 1e-6)).reshape(K, -1)
        log_probs = F.log_softmax(logits, dim=-1)

        gt_flat = target.reshape(K, -1)
        gt_sum = gt_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gt_prob = gt_flat / gt_sum

        ce = -(gt_prob * log_probs).sum(dim=-1)
        return ce.mean()

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

        # (1) Visibility loss — all views
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_vis.float())

        pos_mask = gt_vis.bool()
        neg_mask = ~pos_mask
        has_pos = pos_mask.any()
        has_neg = neg_mask.any()
        pred_pos = pred_heatmaps[pos_mask] if has_pos else None
        gt_pos = gt_heatmaps[pos_mask] if has_pos else None

        # (2) Softmax CE — pixel-space classification on visible views
        if has_pos and self.lambda_peak > 0:
            ce_loss = self.softmax_ce_loss(pred_pos, gt_pos)
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # (3) Negative BCE — push invisible-view h_loc toward zero
        if has_neg and self.lambda_neg > 0:
            pred_neg = pred_heatmaps[neg_mask]
            neg_logits = torch.logit(pred_neg.float().clamp(1e-6, 1 - 1e-6))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="mean",
            )
        else:
            neg_loss = torch.tensor(0.0, device=device)

        # (4) Coordinate loss — auxiliary peak-position refinement
        if has_pos and self.lambda_coord > 0:
            coord_loss = self.soft_argmax_coord_loss(pred_pos, gt_pos)
        else:
            coord_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_peak * ce_loss
            + self.lambda_neg * neg_loss
            + self.lambda_coord * coord_loss
        )

        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "kl_loss": torch.tensor(0.0, device=device),
            "neg_loss": neg_loss.detach(),
            "peak_loss": ce_loss.detach(),
        }
