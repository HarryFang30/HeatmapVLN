"""
HeatmapVLN Loss
=================

Task-priority loss for navigation-oriented heatmap prediction:
  1. Visibility BCE       — classify whether a history view is visible
  2. Coordinate loss      — directly supervise peak location
  3. KL distribution loss — shape matching after normalization
  4. Negative suppression — invisible views should stay near zero

Reference: HeatmapVLN设计文档 Section 8
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
        lambda_kl: weight for positive-sample KL shape loss.
        lambda_neg: weight for negative-sample suppression.
        temperature: soft-argmax temperature.
        heatmap_size: expected heatmap resolution.
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_coord: float = 5.0,
        lambda_kl: float = 1.0,
        lambda_neg: float = 0.1,
        temperature: float = 3.0,
        heatmap_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_kl = lambda_kl
        self.lambda_neg = lambda_neg
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
        if heatmaps.ndim != 4:
            raise ValueError(
                f"Expected pred_heatmaps with shape (N_hist, 4, H, W), got {tuple(heatmaps.shape)}"
            )
        actual_size = tuple(int(v) for v in heatmaps.shape[-2:])
        if actual_size != self.heatmap_size:
            raise ValueError(
                f"Heatmap size mismatch: expected {self.heatmap_size}, got {actual_size}"
            )

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

    @staticmethod
    def normalized_kl_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare normalized heatmaps as probability distributions.

        pred/target: [K, H, W]
        """
        eps = 1e-8
        pred = pred.float().reshape(pred.shape[0], -1)
        target = target.float().reshape(target.shape[0], -1)

        pred_prob = pred + eps
        pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)

        target_prob = target + eps
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True)

        kl = target_prob * (target_prob.log() - pred_prob.log())
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        pred_vis: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_vis:      ``(N_hist, 4)``          — predicted visibility (raw logits).
            pred_heatmaps: ``(N_hist, 4, 64, 64)``  — predicted heatmaps (after sigmoid).
            gt_vis:        ``(N_hist, 4)``           — GT visibility (0 or 1).
            gt_heatmaps:   ``(N_hist, 4, 64, 64)``  — GT heatmaps.
        """
        device = pred_vis.device
        self._validate_heatmap_shape(pred_heatmaps)

        # (1) Visibility loss
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_vis.float())

        # (2) + (3) Positive-sample coordinate and distribution losses
        pos_mask = gt_vis.bool()
        if pos_mask.any():
            pred_pos = pred_heatmaps[pos_mask]
            gt_pos = gt_heatmaps[pos_mask]
            coord_loss = self.soft_argmax_coord_loss(pred_pos, gt_pos)
            kl_loss = self.normalized_kl_loss(pred_pos, gt_pos)
        else:
            coord_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)

        # (4) Negative-sample suppression (invisible views should be all-zero)
        neg_mask = ~pos_mask
        if neg_mask.any():
            neg_loss = pred_heatmaps[neg_mask].float().square().mean()
        else:
            neg_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_coord * coord_loss
            + self.lambda_kl * kl_loss
            + self.lambda_neg * neg_loss
        )

        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "neg_loss": neg_loss.detach(),
        }
