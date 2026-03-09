"""
HeatmapVLN Loss
=================

Task-priority loss for navigation-oriented heatmap prediction:
  1. Visibility BCE       — classify whether a history view is visible
  2. Coordinate loss      — directly supervise peak location (soft-argmax)
  3. KL distribution loss — shape matching after normalization
  4. Background suppression — pixel-level L2 on all GT-dark pixels (visible + invisible)
  5. Peak region reconstruction — dense Smooth-L1 on GT's brightest pixels (anti-collapse)

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
        lambda_neg: weight for background suppression (pixel-level L2).
        lambda_peak: weight for peak region reconstruction (Smooth-L1).
        temperature: soft-argmax temperature (fixed, no annealing recommended).
        heatmap_size: expected heatmap resolution.
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_coord: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_neg: float = 1.0,
        lambda_peak: float = 1.0,
        temperature: float = 1.0,
        heatmap_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_kl = lambda_kl
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

        # (4) Background suppression: pixel-level L2 on ALL GT-dark pixels.
        #     Unlike the old view-level approach (~pos_mask), this also
        #     suppresses false activation in visible views' dark regions
        #     (e.g., stripe artifacts that pass through a visible view).
        gt_dark = gt_heatmaps.float() < 0.01
        if gt_dark.any():
            neg_loss = pred_heatmaps[gt_dark].float().square().mean()
        else:
            neg_loss = torch.tensor(0.0, device=device)

        # (5) Peak region reconstruction: L1 on ALL GT-bright pixels.
        #     Covers the full Gaussian blob (~250 pixels for sigma≈3) instead
        #     of just top-40.  L1 (not Smooth-L1) gives constant gradient
        #     magnitude regardless of error size, crucial for escaping the
        #     near-zero collapse state.
        if pos_mask.any():
            bright = gt_pos >= 0.01
            if bright.any():
                peak_loss = F.l1_loss(
                    pred_pos[bright].float(), gt_pos[bright].float()
                )
            else:
                peak_loss = torch.tensor(0.0, device=device)
        else:
            peak_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_coord * coord_loss
            + self.lambda_kl * kl_loss
            + self.lambda_neg * neg_loss
            + self.lambda_peak * peak_loss
        )

        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "peak_loss": peak_loss.detach(),
        }
