"""Loss and metrics for four-bin future trajectory *tube* heatmaps.

Each time bin can contain eight waypoints and can cross a view boundary.  It
is therefore deliberately not evaluated with point PCK, soft-argmax, or a
forced single-view class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .heatmap.heatmap_vln_loss import HeatmapVLNLoss


class FutureTrajectoryObjectiveError(ValueError):
    """Raised when a future-tube tensor violates the v1 contract."""


def _validate_contract(
    visibility: torch.Tensor,
    heatmaps: torch.Tensor,
    time_mask: torch.Tensor,
) -> None:
    if heatmaps.ndim != 5 or tuple(heatmaps.shape[1:]) != (4, 4, 64, 64):
        raise FutureTrajectoryObjectiveError(
            "future heatmaps must be [B,4_time,4_view,64,64], got "
            f"{tuple(heatmaps.shape)}"
        )
    if tuple(visibility.shape) != tuple(heatmaps.shape[:3]):
        raise FutureTrajectoryObjectiveError(
            "future visibility must match [B,4_time,4_view]"
        )
    if time_mask.dtype != torch.bool or tuple(time_mask.shape) != tuple(
        heatmaps.shape[:2]
    ):
        raise FutureTrajectoryObjectiveError("future_time_mask must be bool [B,4]")


class FutureTrajectoryHeatmapObjective(nn.Module):
    """Reuse the existing map loss with point-only objectives disabled."""

    def __init__(
        self,
        *,
        lambda_vis: float = 1.0,
        lambda_peak: float = 1.0,
        lambda_neg: float = 0.25,
        lambda_view_macro: float = 0.5,
        vis_pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = HeatmapVLNLoss(
            lambda_vis=lambda_vis,
            lambda_peak=lambda_peak,
            lambda_coord=0.0,
            lambda_neg=lambda_neg,
            lambda_view_macro=lambda_view_macro,
            lambda_panoramic_view=0.0,
            lambda_direction_macro=0.0,
            heatmap_size=(64, 64),
            vis_pos_weight=vis_pos_weight,
            allow_probability_fallback=False,
        )

    def forward(
        self,
        *,
        pred_visibility_logits: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        pred_heatmap_logits: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        future_time_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_contract(pred_visibility_logits, pred_heatmaps, future_time_mask)
        _validate_contract(gt_visibility, gt_heatmaps, future_time_mask)
        if tuple(pred_heatmap_logits.shape) != tuple(pred_heatmaps.shape):
            raise FutureTrajectoryObjectiveError(
                "future raw logits/probability shapes must match"
            )
        return self.base(
            pred_visibility_logits,
            pred_heatmaps,
            gt_visibility,
            gt_heatmaps,
            history_mask=future_time_mask,
            pred_heatmap_logits=pred_heatmap_logits,
        )


@dataclass(frozen=True)
class FutureTubeMetrics:
    soft_iou: float
    topk_support_recall: float
    visibility_f1: float
    valid_time_bins: int
    supported_view_bins: int
    per_view_soft_iou: tuple[float | None, float | None, float | None, float | None]
    per_view_support: tuple[int, int, int, int]


def future_tube_sufficient_statistics(
    *,
    pred_visibility_logits: torch.Tensor,
    pred_heatmaps: torch.Tensor,
    gt_visibility: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    future_time_mask: torch.Tensor,
    target_support_threshold: float = 0.1,
) -> torch.Tensor:
    """Return additive FP64 statistics suitable for DDP SUM reduction.

    Layout: IoU sum/count, top-k recall sum/count, visibility TP/FP/FN,
    valid time bins, supported view bins, then four per-view IoU sums/counts.
    """

    _validate_contract(pred_visibility_logits, pred_heatmaps, future_time_mask)
    _validate_contract(gt_visibility, gt_heatmaps, future_time_mask)
    if not 0.0 < target_support_threshold < 1.0:
        raise ValueError("target_support_threshold must be in (0,1)")

    device = pred_heatmaps.device
    stats = torch.zeros(17, dtype=torch.float64, device=device)
    valid = future_time_mask.unsqueeze(-1).expand_as(gt_visibility)
    gt_visible = gt_visibility.bool() & valid
    pred_visible = pred_visibility_logits.sigmoid() >= 0.5
    stats[4] = (pred_visible & gt_visible).sum()
    stats[5] = (pred_visible & ~gt_visible & valid).sum()
    stats[6] = (~pred_visible & gt_visible).sum()
    stats[7] = future_time_mask.sum()

    prediction = pred_heatmaps.float().clamp(0.0, 1.0)
    target = gt_heatmaps.float().clamp(0.0, 1.0)
    supported = gt_visible & (target.sum(dim=(-2, -1)) > 0)
    stats[8] = supported.sum()
    if not bool(supported.any()):
        return stats

    pred_rows = prediction[supported]
    target_rows = target[supported]
    intersection = torch.minimum(pred_rows, target_rows).sum(dim=(-2, -1))
    union = torch.maximum(pred_rows, target_rows).sum(dim=(-2, -1)).clamp_min(1e-8)
    stats[0] = (intersection / union).sum()
    stats[1] = pred_rows.shape[0]

    recall_sum = torch.zeros((), device=device, dtype=torch.float64)
    recall_count = 0
    target_support = target_rows >= float(target_support_threshold)
    for pred_row, support_row in zip(pred_rows, target_support, strict=True):
        k = int(support_row.sum().item())
        if k <= 0:
            continue
        topk = pred_row.reshape(-1).topk(k=min(k, pred_row.numel())).indices
        recall_sum += support_row.reshape(-1).index_select(0, topk).double().mean()
        recall_count += 1
    stats[2] = recall_sum
    stats[3] = recall_count

    for view in range(4):
        view_mask = supported[..., view]
        count = int(view_mask.sum().item())
        if count:
            p = prediction[..., view, :, :][view_mask]
            g = target[..., view, :, :][view_mask]
            inter = torch.minimum(p, g).sum(dim=(-2, -1))
            uni = torch.maximum(p, g).sum(dim=(-2, -1)).clamp_min(1e-8)
            stats[9 + 2 * view] = (inter / uni).sum()
            stats[10 + 2 * view] = count
    return stats


def future_tube_metrics_from_statistics(stats: torch.Tensor) -> FutureTubeMetrics:
    if stats.numel() != 17 or not torch.isfinite(stats).all():
        raise FutureTrajectoryObjectiveError("future tube statistics must be 17 finite values")
    values = stats.detach().double().cpu()
    tp, fp, fn = (values[index].item() for index in (4, 5, 6))
    visibility_denom = 2.0 * tp + fp + fn
    per_view_iou: list[float | None] = []
    per_view_support: list[int] = []
    for view in range(4):
        count = int(values[10 + 2 * view].item())
        per_view_support.append(count)
        per_view_iou.append(
            float(values[9 + 2 * view].item() / count) if count else None
        )
    iou_count = values[1].item()
    recall_count = values[3].item()
    return FutureTubeMetrics(
        soft_iou=float(values[0].item() / iou_count) if iou_count else 0.0,
        topk_support_recall=(
            float(values[2].item() / recall_count) if recall_count else 0.0
        ),
        visibility_f1=float(2.0 * tp / visibility_denom) if visibility_denom else 1.0,
        valid_time_bins=int(values[7].item()),
        supported_view_bins=int(values[8].item()),
        per_view_soft_iou=tuple(per_view_iou),
        per_view_support=tuple(per_view_support),
    )


@torch.no_grad()
def compute_future_tube_metrics(**kwargs) -> FutureTubeMetrics:
    return future_tube_metrics_from_statistics(
        future_tube_sufficient_statistics(**kwargs)
    )
