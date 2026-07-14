"""Target-grounded identity supervision for strict multi-history diagnostics.

The loss in this module consumes only model predictions and ground-truth
heatmap tensors.  Pose, frame index, temporal slot, and trajectory metadata
are deliberately absent from the API.  Four ground-truth targets are used as
within-sample negatives so that four independent history predictions cannot
all satisfy the objective with the same location prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PANORAMIC_VIEWS = 4


@dataclass(frozen=True)
class PrimaryPanoramaTargets:
    """Primary visible target coordinates derived solely from GT tensors.

    Every tensor has shape ``[B, K]``.  ``panorama_x`` uses the circular view
    order ``front | right | back | left`` and therefore lies in
    ``[0, panorama_width)``.
    """

    view_indices: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    panorama_x: torch.Tensor
    panorama_width: int


def _validate_ground_truth(
    gt_visibility: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    *,
    expected_num_targets: int,
) -> tuple[int, int, int, int]:
    if gt_visibility.ndim != 3:
        raise ValueError(f"gt_visibility must have shape [B,K,4], got {tuple(gt_visibility.shape)}")
    if gt_heatmaps.ndim != 5:
        raise ValueError(f"gt_heatmaps must have shape [B,K,4,H,W], got {tuple(gt_heatmaps.shape)}")
    batch_size, num_targets, num_views = gt_visibility.shape
    if num_targets != expected_num_targets:
        raise ValueError(f"Expected exactly K={expected_num_targets} targets, got K={num_targets}")
    if num_views != NUM_PANORAMIC_VIEWS:
        raise ValueError(f"Expected {NUM_PANORAMIC_VIEWS} views, got {num_views}")
    if tuple(gt_heatmaps.shape[:3]) != tuple(gt_visibility.shape):
        raise ValueError(
            f"GT visibility/heatmap leading shape mismatch: {tuple(gt_visibility.shape)} vs {tuple(gt_heatmaps.shape)}"
        )
    height, width = (int(value) for value in gt_heatmaps.shape[-2:])
    if height <= 0 or width <= 0:
        raise ValueError(f"Heatmap dimensions must be positive, got {(height, width)}")
    if not torch.isfinite(gt_visibility).all() or not torch.isfinite(gt_heatmaps).all():
        raise ValueError("Ground-truth visibility and heatmaps must be finite")
    return int(batch_size), int(num_targets), height, width


def extract_primary_panorama_targets(
    gt_visibility: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    *,
    expected_num_targets: int = 4,
) -> PrimaryPanoramaTargets:
    """Extract one primary target per history from visibility and heatmaps.

    Among visible views, the largest GT heatmap value wins.  Flattened argmax
    deterministically resolves exact ties in favour of the lower view index,
    matching the selection-manifest primary-target rule.  GT extraction is a
    label operation and is intentionally non-differentiable.
    """

    batch_size, num_targets, height, width = _validate_ground_truth(
        gt_visibility,
        gt_heatmaps,
        expected_num_targets=expected_num_targets,
    )
    visible = gt_visibility > 0.5
    missing = ~visible.any(dim=-1)
    if missing.any():
        indices = torch.nonzero(missing, as_tuple=False).detach().cpu().tolist()
        raise ValueError(f"Every history requires a visible GT target; missing at {indices}")

    masked = (
        gt_heatmaps.detach()
        .float()
        .masked_fill(
            ~visible[..., None, None],
            -torch.inf,
        )
    )
    flat = masked.reshape(batch_size, num_targets, -1)
    peak_values, peak_indices = flat.max(dim=-1)
    if not torch.isfinite(peak_values).all() or (peak_values <= 0).any():
        indices = (
            torch.nonzero(
                ~torch.isfinite(peak_values) | (peak_values <= 0),
                as_tuple=False,
            )
            .detach()
            .cpu()
            .tolist()
        )
        raise ValueError(f"Visible GT targets must have positive finite peaks; invalid at {indices}")

    pixels_per_view = height * width
    view_indices = torch.div(peak_indices, pixels_per_view, rounding_mode="floor")
    within_view = torch.remainder(peak_indices, pixels_per_view)
    y = torch.div(within_view, width, rounding_mode="floor")
    x = torch.remainder(within_view, width)
    panorama_x = view_indices * width + x
    return PrimaryPanoramaTargets(
        view_indices=view_indices,
        x=x,
        y=y,
        panorama_x=panorama_x,
        panorama_width=NUM_PANORAMIC_VIEWS * width,
    )


def circular_pairwise_distances(
    panorama_x: torch.Tensor,
    y: torch.Tensor,
    panorama_width: int,
) -> torch.Tensor:
    """Return circular panorama distances with shape ``[B, K, K]``."""

    if panorama_x.ndim != 2 or y.ndim != 2 or panorama_x.shape != y.shape:
        raise ValueError(
            f"panorama_x and y must have the same [B,K] shape, got {tuple(panorama_x.shape)} and {tuple(y.shape)}"
        )
    if panorama_width <= 0:
        raise ValueError(f"panorama_width must be positive, got {panorama_width}")
    x_float = torch.remainder(panorama_x.float(), float(panorama_width))
    y_float = y.float()
    delta_x = (x_float.unsqueeze(-1) - x_float.unsqueeze(-2)).abs()
    delta_x = torch.minimum(delta_x, float(panorama_width) - delta_x)
    delta_y = y_float.unsqueeze(-1) - y_float.unsqueeze(-2)
    return torch.sqrt(delta_x.square() + delta_y.square())


def target_grounded_score_matrix(
    pred_heatmap_logits: torch.Tensor,
    targets: PrimaryPanoramaTargets,
) -> torch.Tensor:
    """Sample every history prediction at every GT target location.

    Spatial logits are normalized independently inside each view before the
    four views are stitched into a circular panorama.  Consequently this
    score measures conditional localization rather than allowing a query to
    win identity classification using only a view-level bias.  Bilinear
    ``grid_sample`` keeps the score differentiable with respect to all model
    logits; the hard GT coordinates themselves need no gradient.
    """

    if pred_heatmap_logits.ndim != 5:
        raise ValueError(f"pred_heatmap_logits must have shape [B,K,4,H,W], got {tuple(pred_heatmap_logits.shape)}")
    batch_size, num_queries, num_views, height, width = pred_heatmap_logits.shape
    if num_views != NUM_PANORAMIC_VIEWS:
        raise ValueError(f"Expected {NUM_PANORAMIC_VIEWS} views, got {num_views}")
    if tuple(targets.panorama_x.shape) != (batch_size, num_queries):
        raise ValueError(
            f"Target/query shape mismatch: expected {(batch_size, num_queries)}, got {tuple(targets.panorama_x.shape)}"
        )
    if targets.panorama_width != num_views * width:
        raise ValueError(
            f"Target panorama width does not match prediction: {targets.panorama_width} vs {num_views * width}"
        )
    if not torch.isfinite(pred_heatmap_logits).all():
        raise ValueError("Predicted heatmap logits must be finite")

    target_x = targets.panorama_x.to(device=pred_heatmap_logits.device, dtype=torch.float32)
    target_y = targets.y.to(device=pred_heatmap_logits.device, dtype=torch.float32)
    if (target_x < 0).any() or (target_x >= targets.panorama_width).any():
        raise ValueError("Target panorama x coordinates are outside the circular panorama")
    if (target_y < 0).any() or (target_y >= height).any():
        raise ValueError("Target y coordinates are outside the heatmap")

    logits = pred_heatmap_logits.float()
    spatial_log_prob = F.log_softmax(
        logits.reshape(batch_size, num_queries, num_views, height * width),
        dim=-1,
    ).reshape(batch_size, num_queries, num_views, height, width)
    panorama_log_prob = spatial_log_prob.permute(0, 1, 3, 2, 4).reshape(
        batch_size,
        num_queries,
        height,
        num_views * width,
    )

    wrapped_x = torch.remainder(target_x, float(targets.panorama_width))
    grid_x = 2.0 * (wrapped_x + 0.5) / float(targets.panorama_width) - 1.0
    grid_y = 2.0 * (target_y + 0.5) / float(height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        panorama_log_prob,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(-1)


def target_grounded_panorama_losses(
    pred_heatmap_logits: torch.Tensor,
    gt_visibility: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Supervise the target as one class over the complete panorama.

    The target distribution is the non-negative visible GT heatmap normalized
    over all ``4*H*W`` panorama pixels.  The global cross-entropy decomposes
    exactly into a soft categorical view-marginal loss and a target-view-
    weighted conditional pixel loss.  Unlike the legacy visibility MLP, all
    three quantities are computed directly from raw query-to-current match
    logits.
    """

    if pred_heatmap_logits.ndim != 5:
        raise ValueError(f"pred_heatmap_logits must have shape [B,K,4,H,W], got {tuple(pred_heatmap_logits.shape)}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    batch_size, num_queries, num_views, height, width = pred_heatmap_logits.shape
    if num_views != NUM_PANORAMIC_VIEWS:
        raise ValueError(f"Expected {NUM_PANORAMIC_VIEWS} views, got {num_views}")
    if tuple(gt_visibility.shape) != (batch_size, num_queries, num_views):
        raise ValueError(
            "Prediction/GT visibility shape mismatch: "
            f"{tuple(pred_heatmap_logits.shape[:3])} vs {tuple(gt_visibility.shape)}"
        )
    if tuple(gt_heatmaps.shape) != tuple(pred_heatmap_logits.shape):
        raise ValueError(
            f"Prediction/GT heatmap shape mismatch: {tuple(pred_heatmap_logits.shape)} vs {tuple(gt_heatmaps.shape)}"
        )
    if not (
        torch.isfinite(pred_heatmap_logits).all()
        and torch.isfinite(gt_visibility).all()
        and torch.isfinite(gt_heatmaps).all()
    ):
        raise ValueError("Predicted logits and GT panorama tensors must be finite")
    if (gt_heatmaps < 0).any():
        raise ValueError("GT panorama heatmaps must be non-negative")

    device = pred_heatmap_logits.device
    scaled_logits = pred_heatmap_logits.float() / float(temperature)
    pixels_per_view = height * width
    visible = gt_visibility.to(device=device).float() > 0.5
    nonnegative_gt = gt_heatmaps.to(device=device).float()
    target_mass = nonnegative_gt * visible[..., None, None]
    panorama_target_mass = target_mass.reshape(batch_size, num_queries, -1).sum(dim=-1)
    if (panorama_target_mass <= 0).any():
        missing = torch.nonzero(panorama_target_mass <= 0, as_tuple=False).detach().cpu().tolist()
        raise ValueError(f"Every query requires positive visible GT heatmap mass; missing at {missing}")
    target_distribution = target_mass / panorama_target_mass[..., None, None, None]
    panorama_log_probs = F.log_softmax(
        scaled_logits.reshape(batch_size, num_queries, -1),
        dim=-1,
    ).reshape_as(scaled_logits)
    panorama_loss = -(target_distribution * panorama_log_probs).sum(dim=(2, 3, 4)).mean()

    per_view_flat = scaled_logits.reshape(
        batch_size,
        num_queries,
        num_views,
        pixels_per_view,
    )
    view_logits = torch.logsumexp(per_view_flat, dim=-1)
    target_view_distribution = target_distribution.sum(dim=(-2, -1))
    view_loss = -(target_view_distribution * F.log_softmax(view_logits, dim=-1)).sum(dim=-1).mean()
    within_view_log_probs = F.log_softmax(per_view_flat, dim=-1).reshape_as(scaled_logits)
    within_view_loss = -(target_distribution * within_view_log_probs).sum(dim=(2, 3, 4)).mean()
    decomposition_residual = panorama_loss - view_loss - within_view_loss
    return {
        "panorama_loss": panorama_loss,
        "view_loss": view_loss,
        "within_view_loss": within_view_loss,
        "view_logits": view_logits,
        "target_view_distribution": target_view_distribution,
        "decomposition_residual": decomposition_residual,
    }


class TargetGroundedIdentityLoss(nn.Module):
    """Symmetric target identity InfoNCE plus primary-view cross-entropy."""

    def __init__(
        self,
        *,
        identity_weight: float = 2.0,
        view_weight: float = 1.0,
        temperature: float = 1.0,
        min_target_separation: float = 12.0,
        expected_num_targets: int = 4,
    ) -> None:
        super().__init__()
        if not all(
            math.isfinite(value)
            for value in (
                identity_weight,
                view_weight,
                temperature,
                min_target_separation,
            )
        ):
            raise ValueError("identity_weight, view_weight, temperature, and min_target_separation must be finite")
        if identity_weight < 0 or view_weight < 0:
            raise ValueError("identity_weight and view_weight must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if min_target_separation < 0:
            raise ValueError("min_target_separation must be non-negative")
        if expected_num_targets != 4:
            raise ValueError("The strict diagnostic contract requires exactly four targets")
        self.identity_weight = float(identity_weight)
        self.view_weight = float(view_weight)
        self.temperature = float(temperature)
        self.min_target_separation = float(min_target_separation)
        self.expected_num_targets = int(expected_num_targets)

    def forward(
        self,
        pred_visibility_logits: torch.Tensor,
        pred_heatmap_logits: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_heatmaps: torch.Tensor,
    ) -> dict[str, torch.Tensor | PrimaryPanoramaTargets]:
        if pred_visibility_logits.ndim != 3:
            raise ValueError(
                f"pred_visibility_logits must have shape [B,K,4], got {tuple(pred_visibility_logits.shape)}"
            )
        if tuple(pred_visibility_logits.shape) != tuple(gt_visibility.shape):
            raise ValueError(
                "Prediction/GT visibility shape mismatch: "
                f"{tuple(pred_visibility_logits.shape)} vs {tuple(gt_visibility.shape)}"
            )
        if tuple(pred_heatmap_logits.shape) != tuple(gt_heatmaps.shape):
            raise ValueError(
                "Prediction/GT heatmap shape mismatch: "
                f"{tuple(pred_heatmap_logits.shape)} vs {tuple(gt_heatmaps.shape)}"
            )
        if not torch.isfinite(pred_visibility_logits).all():
            raise ValueError("Predicted visibility logits must be finite")

        targets = extract_primary_panorama_targets(
            gt_visibility,
            gt_heatmaps,
            expected_num_targets=self.expected_num_targets,
        )
        pairwise_distances = circular_pairwise_distances(
            targets.panorama_x,
            targets.y,
            targets.panorama_width,
        )
        num_targets = self.expected_num_targets
        off_diagonal = ~torch.eye(
            num_targets,
            dtype=torch.bool,
            device=pairwise_distances.device,
        ).unsqueeze(0)
        minimum_separation = pairwise_distances.masked_select(off_diagonal).min()
        if float(minimum_separation.detach().cpu()) + 1e-12 < self.min_target_separation:
            raise ValueError(
                "GT target separation violates the strict circular-panorama contract: "
                f"minimum={float(minimum_separation.detach().cpu()):.6f} "
                f"required={self.min_target_separation:.6f}"
            )

        score_matrix = target_grounded_score_matrix(pred_heatmap_logits, targets)
        batch_size = int(score_matrix.shape[0])
        labels = (
            torch.arange(
                num_targets,
                device=score_matrix.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
        )
        scaled_scores = score_matrix / self.temperature
        row_identity_loss = F.cross_entropy(
            scaled_scores.reshape(batch_size * num_targets, num_targets),
            labels,
        )
        column_identity_loss = F.cross_entropy(
            scaled_scores.transpose(1, 2).reshape(batch_size * num_targets, num_targets),
            labels,
        )
        identity_loss = 0.5 * (row_identity_loss + column_identity_loss)
        view_loss = F.cross_entropy(
            pred_visibility_logits.float().reshape(batch_size * num_targets, NUM_PANORAMIC_VIEWS),
            targets.view_indices.to(device=pred_visibility_logits.device).reshape(-1),
        )
        total = self.identity_weight * identity_loss + self.view_weight * view_loss
        return {
            "total": total,
            "identity_loss": identity_loss,
            "row_identity_loss": row_identity_loss,
            "column_identity_loss": column_identity_loss,
            "view_loss": view_loss,
            "score_matrix": score_matrix,
            "pairwise_target_distances": pairwise_distances,
            "minimum_target_separation": minimum_separation,
            "targets": targets,
        }


class TargetGroundedPanoramaIdentityLoss(nn.Module):
    """K-way visual identity plus global panoramic pixel supervision.

    No visibility-head output is accepted by this API.  View selection is the
    marginal of the same raw heatmap logits used for pixel localization, so a
    learned visibility readout cannot absorb the auxiliary task while the VLM
    remains unchanged.
    """

    def __init__(
        self,
        *,
        identity_weight: float = 2.0,
        panorama_weight: float = 1.0,
        temperature: float = 1.0,
        min_target_separation: float = 12.0,
        expected_num_targets: int = 4,
        require_unique_visible_view: bool = True,
    ) -> None:
        super().__init__()
        values = (
            identity_weight,
            panorama_weight,
            temperature,
            min_target_separation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("identity_weight, panorama_weight, temperature, and min_target_separation must be finite")
        if identity_weight < 0 or panorama_weight < 0:
            raise ValueError("identity_weight and panorama_weight must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if min_target_separation < 0:
            raise ValueError("min_target_separation must be non-negative")
        if expected_num_targets != 4:
            raise ValueError("The strict diagnostic contract requires exactly four targets")
        self.identity_weight = float(identity_weight)
        self.panorama_weight = float(panorama_weight)
        self.temperature = float(temperature)
        self.min_target_separation = float(min_target_separation)
        self.expected_num_targets = int(expected_num_targets)
        self.require_unique_visible_view = bool(require_unique_visible_view)

    def forward(
        self,
        pred_heatmap_logits: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_heatmaps: torch.Tensor,
    ) -> dict[str, torch.Tensor | PrimaryPanoramaTargets]:
        if tuple(pred_heatmap_logits.shape) != tuple(gt_heatmaps.shape):
            raise ValueError(
                "Prediction/GT heatmap shape mismatch: "
                f"{tuple(pred_heatmap_logits.shape)} vs {tuple(gt_heatmaps.shape)}"
            )
        targets = extract_primary_panorama_targets(
            gt_visibility,
            gt_heatmaps,
            expected_num_targets=self.expected_num_targets,
        )
        if self.require_unique_visible_view:
            visible_counts = (gt_visibility > 0.5).sum(dim=-1)
            if not torch.equal(visible_counts, torch.ones_like(visible_counts)):
                raise ValueError(
                    "Strict panorama identity requires exactly one visible GT view per query; "
                    f"got {visible_counts.detach().cpu().tolist()}"
                )
        pairwise_distances = circular_pairwise_distances(
            targets.panorama_x,
            targets.y,
            targets.panorama_width,
        )
        off_diagonal = ~torch.eye(
            self.expected_num_targets,
            dtype=torch.bool,
            device=pairwise_distances.device,
        ).unsqueeze(0)
        minimum_separation = pairwise_distances.masked_select(off_diagonal).min()
        if float(minimum_separation.detach().cpu()) + 1e-12 < self.min_target_separation:
            raise ValueError(
                "GT target separation violates the strict circular-panorama contract: "
                f"minimum={float(minimum_separation.detach().cpu()):.6f} "
                f"required={self.min_target_separation:.6f}"
            )

        score_matrix = target_grounded_score_matrix(pred_heatmap_logits, targets)
        batch_size = int(score_matrix.shape[0])
        labels = (
            torch.arange(self.expected_num_targets, device=score_matrix.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
        )
        scaled_scores = score_matrix / self.temperature
        row_identity_loss = F.cross_entropy(
            scaled_scores.reshape(batch_size * self.expected_num_targets, self.expected_num_targets),
            labels,
        )
        column_identity_loss = F.cross_entropy(
            scaled_scores.transpose(1, 2).reshape(
                batch_size * self.expected_num_targets,
                self.expected_num_targets,
            ),
            labels,
        )
        identity_loss = 0.5 * (row_identity_loss + column_identity_loss)
        panorama = target_grounded_panorama_losses(
            pred_heatmap_logits,
            gt_visibility,
            gt_heatmaps,
            temperature=self.temperature,
        )
        total = self.identity_weight * identity_loss + self.panorama_weight * panorama["panorama_loss"]
        return {
            "total": total,
            "identity_loss": identity_loss,
            "row_identity_loss": row_identity_loss,
            "column_identity_loss": column_identity_loss,
            **panorama,
            "score_matrix": score_matrix,
            "pairwise_target_distances": pairwise_distances,
            "minimum_target_separation": minimum_separation,
            "targets": targets,
        }


__all__ = [
    "PrimaryPanoramaTargets",
    "TargetGroundedIdentityLoss",
    "TargetGroundedPanoramaIdentityLoss",
    "circular_pairwise_distances",
    "extract_primary_panorama_targets",
    "target_grounded_panorama_losses",
    "target_grounded_score_matrix",
]
