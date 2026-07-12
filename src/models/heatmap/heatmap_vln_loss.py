"""
HeatmapVLN Loss
=================

Loss for navigation-oriented heatmap prediction:
  1. Visibility BCE (weighted) — classify whether a history view is visible
  2. Softmax Cross-Entropy     — pixel-space classification on visible views
  3. Coordinate loss           — low-weight auxiliary for peak position refinement
  4. Neg loss (per-pixel BCE)  — push invisible-view heatmaps toward zero

Design principle:
  - Heatmap backbone answers "WHERE" (peak CE) and "suppress invisible" (neg BCE).
    These operate on disjoint view sets (pos_mask vs neg_mask), no gradient conflict.
  - vis_head answers "WHETHER" (weighted BCE), gradient detached from backbone.
"""



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
        lambda_peak: weight for softmax cross-entropy on visible-view heatmaps.
        temperature: soft-argmax temperature (only affects coord_loss).
        heatmap_size: expected heatmap resolution (H, W).
        vis_pos_weight: pos_weight for visibility BCE to handle class
                        imbalance (recommended ~7.0 for 87%/13% neg/pos ratio).
    """

    def __init__(
        self,
        lambda_vis: float = 1.0,
        lambda_coord: float = 0.2,
        lambda_kl: float = 0.0,
        lambda_peak: float = 1.0,
        lambda_neg: float = 0.0,
        temperature: float = 1.0,
        heatmap_size: tuple[int, int] = (64, 64),
        vis_pos_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_vis = lambda_vis
        self.lambda_coord = lambda_coord
        self.lambda_peak = lambda_peak
        self.lambda_neg = lambda_neg
        self.temperature = temperature
        self.heatmap_size = tuple(int(v) for v in heatmap_size)
        self.vis_pos_weight = vis_pos_weight

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Differentiable soft-argmax for predictions.
        pred_weights = F.softmax(pred_logits, dim=-1).reshape_as(pred)
        pred_cx = (pred_weights * self.coords_x).sum(dim=(-2, -1))
        pred_cy = (pred_weights * self.coords_y).sum(dim=(-2, -1))

        # Hard argmax for target (no gradient needed, saves one materialization).
        target_flat = target.reshape(num_samples, -1)
        target_peak = target_flat.argmax(dim=-1)
        target_cx = self.coords_x.reshape(-1)[target_peak]
        target_cy = self.coords_y.reshape(-1)[target_peak]

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

        Uses F.cross_entropy with soft targets for a fused log_softmax+nll
        kernel, reducing intermediate tensors and autograd nodes.

        pred:   [K, H, W]  sigmoid-activated values in (0, 1)
        target: [K, H, W]  GT heatmap values in [0, 1]
        Returns: scalar CE averaged over K views.
        """
        pred = pred.float()
        target = target.float()
        K = pred.shape[0]

        logits = torch.logit(pred.clamp(1e-6, 1 - 1e-6)).reshape(K, -1)

        gt_flat = target.reshape(K, -1)
        gt_prob = gt_flat / gt_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return F.cross_entropy(logits, gt_prob)

    def forward(
        self,
        pred_vis: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_vis:      ``(N_hist, 4)`` or ``(B, N_hist, 4)``.
            pred_heatmaps: ``(N_hist, 4, H, W)`` or ``(B, N_hist, 4, H, W)``.
            gt_vis:        ``(N_hist, 4)`` or ``(B, N_hist, 4)``.
            gt_heatmaps:   ``(N_hist, 4, H, W)`` or ``(B, N_hist, 4, H, W)``.
            history_mask:  ``(N_hist,)`` or ``(B, N_hist)``.  Optional mask to
                           exclude padded history positions from all losses.
        """
        device = pred_vis.device
        pred_vis, pred_heatmaps, gt_vis, gt_heatmaps = self._flatten_inputs(
            pred_vis,
            pred_heatmaps,
            gt_vis,
            gt_heatmaps,
        )

        # Flatten history_mask to (N,) matching pred_vis leading dim.  A
        # mismatch is unsafe: discarding the mask would silently turn padded
        # panoramic slots into invisible negative examples.
        if history_mask is not None:
            valid = history_mask.reshape(-1).bool()
            if valid.shape[0] != pred_vis.shape[0]:
                raise ValueError(
                    "history_mask shape mismatch: mask has "
                    f"{valid.shape[0]} entries but flattened pred_vis has "
                    f"{pred_vis.shape[0]} rows. Refusing to treat padded "
                    "history slots as negative samples."
                )
            valid_4 = valid.unsqueeze(-1).expand_as(pred_vis)
        else:
            valid = None
            valid_4 = None

        # (1) Visibility loss — weighted BCE on real (non-padded) views
        pw = torch.tensor([self.vis_pos_weight], device=device) if self.vis_pos_weight != 1.0 else None
        vis_bce = F.binary_cross_entropy_with_logits(
            pred_vis, gt_vis.float(), reduction="none", pos_weight=pw,
        )
        if valid_4 is not None:
            vis_loss = (vis_bce * valid_4).sum() / valid_4.float().sum().clamp(min=1)
        else:
            vis_loss = vis_bce.mean()

        # Build pos mask, excluding padding
        pos_mask = gt_vis.bool()
        if valid is not None:
            real_view = valid.unsqueeze(-1).expand_as(pos_mask)
            pos_mask = pos_mask & real_view
        has_pos = pos_mask.any()
        pred_pos = pred_heatmaps[pos_mask] if has_pos else None
        gt_pos = gt_heatmaps[pos_mask] if has_pos else None

        # (2) Softmax CE — pixel-space classification on visible views
        if has_pos and self.lambda_peak > 0:
            ce_loss = self.softmax_ce_loss(pred_pos, gt_pos)
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # (3) Coordinate loss — auxiliary peak-position refinement
        if has_pos and self.lambda_coord > 0:
            coord_loss = self.soft_argmax_coord_loss(pred_pos, gt_pos)
        else:
            coord_loss = torch.tensor(0.0, device=device)

        # (4) Neg loss — per-pixel BCE on invisible views
        if self.lambda_neg > 0:
            neg_mask = (~gt_vis.bool())
            if valid is not None:
                real_view = valid.unsqueeze(-1).expand_as(neg_mask)
                neg_mask = neg_mask & real_view

            if neg_mask.any():
                pred_neg = pred_heatmaps[neg_mask].float().clamp(max=1 - 1e-6)
                neg_loss = -torch.log1p(-pred_neg).mean()
            else:
                neg_loss = torch.tensor(0.0, device=device)
        else:
            neg_loss = torch.tensor(0.0, device=device)

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_peak * ce_loss
            + self.lambda_coord * coord_loss
            + self.lambda_neg * neg_loss
        )

        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "peak_loss": ce_loss.detach(),
            "neg_loss": neg_loss.detach(),
        }
