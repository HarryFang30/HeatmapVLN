"""
HeatmapVLN Loss
=================

Three-component loss for the Coarse-to-Fine localisation system:
  1. Visibility loss  — BCE on per-view visibility logits (~75% are negative)
  2. Positive heatmap loss — Adaptive Wing Loss on visible views
  3. Negative suppression  — L2 penalty on heatmap output for invisible views

Reference: HeatmapVLN设计文档 Section 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class HeatmapVLNLoss(nn.Module):
    """
    Combined loss for HeatmapVLN.

    Args:
        lambda_vis:  weight for visibility BCE.
        lambda_pos:  weight for positive-sample Adaptive Wing Loss.
        lambda_neg:  weight for negative-sample suppression.
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_pos: float = 1.0,
        lambda_neg: float = 0.1,
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    @staticmethod
    def adaptive_wing_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
    ) -> torch.Tensor:
        """
        Adaptive Wing Loss (Wang et al., ICCV 2019).

        Foreground pixels (near Gaussian peak): large gradient -> precise localisation.
        Background pixels: small gradient -> do not disturb foreground learning.
        """
        delta = (pred - target).abs()
        A = (
            omega
            * (1.0 / (1.0 + (theta / epsilon) ** (omega - target)))
            * (omega - target)
            * ((theta / epsilon) ** (omega - target - 1))
            / epsilon
        )
        C = theta * A - omega * torch.log(
            1.0 + (theta / epsilon) ** (omega - target)
        )
        loss = torch.where(
            delta < theta,
            omega * torch.log(1.0 + (delta / epsilon) ** (omega - target)),
            A * delta - C,
        )
        return loss.mean()

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

        # (1) Visibility loss
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_vis.float())

        # (2) Positive-sample heatmap loss (only for visible views)
        pos_mask = gt_vis.bool()
        if pos_mask.any():
            pos_loss = self.adaptive_wing_loss(
                pred_heatmaps[pos_mask], gt_heatmaps[pos_mask],
            )
        else:
            pos_loss = torch.tensor(0.0, device=device)

        # (3) Negative-sample suppression (invisible views should be all-zero)
        neg_mask = ~pos_mask
        if neg_mask.any():
            neg_loss = (pred_heatmaps[neg_mask] ** 2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_pos * pos_loss
            + self.lambda_neg * neg_loss
        )

        return {
            "total": total,
            "vis_loss": vis_loss.detach(),
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
        }
