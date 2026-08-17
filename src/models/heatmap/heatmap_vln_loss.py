"""
HeatmapVLN loss.

The spatial objectives consume the decoder's raw logits.  Probability
heatmaps remain part of the public inference API, but are not round-tripped
through sigmoid -> clamp -> logit during training.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class HeatmapVLNLoss(nn.Module):
    """Task-priority loss for panoramic history localization.

    The primary objective is visibility BCE plus conditional spatial
    cross-entropy on visible views.  Three optional auxiliaries are available:

    * ``lambda_panoramic_view`` adds a hierarchical 5-way view loss with
      classes ``[none, front, right, back, left]``.  The ``none`` logit is the
      fixed zero reference and the four existing visibility logits are reused,
      so no checkpoint parameters are added.
    * ``lambda_view_macro`` averages conditional spatial loss per view before
      averaging the four supported directions.  This improves minority-view
      gradients without resampling or duplicating data.
    * ``lambda_direction_macro`` macro-averages 5-way classification loss over
      the supported front/right/back/left GT directions.  It excludes ``none``
      and leaves the natural-distribution visibility/view losses untouched.

    ``pred_heatmap_logits`` should be supplied by new training code.  The
    probability-only fallback is retained solely so old tools/checkpoints keep
    running; it uses normalized log-probabilities rather than reconstructing
    clipped logits.
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
        lambda_view_macro: float = 0.0,
        lambda_direction_macro: float = 0.0,
        lambda_panoramic_view: float = 0.0,
        panoramic_detach_visibility: bool = False,
        coord_smooth_l1_beta: float = 0.1,
        allow_probability_fallback: bool = True,
        **kwargs,
    ):
        super().__init__()
        del lambda_kl, kwargs  # Retained in the signature for config compatibility.

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if coord_smooth_l1_beta < 0:
            raise ValueError(
                "coord_smooth_l1_beta must be non-negative, got "
                f"{coord_smooth_l1_beta}"
            )

        self.lambda_vis = float(lambda_vis)
        self.lambda_coord = float(lambda_coord)
        self.lambda_peak = float(lambda_peak)
        self.lambda_neg = float(lambda_neg)
        self.lambda_view_macro = float(lambda_view_macro)
        self.lambda_direction_macro = float(lambda_direction_macro)
        self.lambda_panoramic_view = float(lambda_panoramic_view)
        self.temperature = float(temperature)
        self.heatmap_size = tuple(int(v) for v in heatmap_size)
        self.vis_pos_weight = float(vis_pos_weight)
        self.panoramic_detach_visibility = bool(panoramic_detach_visibility)
        self.coord_smooth_l1_beta = float(coord_smooth_l1_beta)
        self.allow_probability_fallback = bool(allow_probability_fallback)
        self._warned_probability_fallback = False

        height, width = self.heatmap_size
        if height <= 0 or width <= 0:
            raise ValueError(f"heatmap_size must be positive, got {self.heatmap_size}")
        coords_y, coords_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("coords_x", coords_x, persistent=False)
        self.register_buffer("coords_y", coords_y, persistent=False)

    def _validate_heatmap_shape(self, heatmaps: torch.Tensor) -> None:
        if heatmaps.ndim not in (4, 5):
            raise ValueError(
                "Expected heatmaps with shape (N_hist, 4, H, W) or "
                f"(B, N_hist, 4, H, W), got {tuple(heatmaps.shape)}"
            )
        if heatmaps.shape[-3] != 4:
            raise ValueError(
                "Expected 4 view channels before spatial dims, got shape "
                f"{tuple(heatmaps.shape)}"
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
        pred_heatmaps = pred_heatmaps.reshape(
            -1,
            pred_heatmaps.shape[-3],
            pred_heatmaps.shape[-2],
            pred_heatmaps.shape[-1],
        )
        gt_heatmaps = gt_heatmaps.reshape(
            -1,
            gt_heatmaps.shape[-3],
            gt_heatmaps.shape[-2],
            gt_heatmaps.shape[-1],
        )

        if pred_vis.shape != gt_vis.shape:
            raise ValueError(
                "Visibility shape mismatch: "
                f"pred {tuple(pred_vis.shape)} vs gt {tuple(gt_vis.shape)}"
            )
        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(
                "Heatmap shape mismatch: "
                f"pred {tuple(pred_heatmaps.shape)} vs gt {tuple(gt_heatmaps.shape)}"
            )
        if pred_heatmaps.shape[:2] != pred_vis.shape:
            raise ValueError(
                "Visibility/heatmap leading shape mismatch: "
                f"vis {tuple(pred_vis.shape)} vs heatmaps {tuple(pred_heatmaps.shape)}"
            )
        return pred_vis, pred_heatmaps, gt_vis, gt_heatmaps

    def _flatten_raw_logits(
        self,
        pred_heatmap_logits: torch.Tensor,
        expected_shape: torch.Size,
    ) -> torch.Tensor:
        self._validate_heatmap_shape(pred_heatmap_logits)
        flattened = pred_heatmap_logits.reshape(
            -1,
            pred_heatmap_logits.shape[-3],
            pred_heatmap_logits.shape[-2],
            pred_heatmap_logits.shape[-1],
        )
        if flattened.shape != expected_shape:
            raise ValueError(
                "Raw/probability heatmap shape mismatch: "
                f"logits {tuple(flattened.shape)} vs heatmaps {tuple(expected_shape)}"
            )
        return flattened

    def _fallback_spatial_logits(self, pred_heatmaps: torch.Tensor) -> torch.Tensor:
        if not self.allow_probability_fallback:
            raise ValueError(
                "pred_heatmap_logits is required when "
                "allow_probability_fallback=False"
            )
        if not self._warned_probability_fallback:
            logger.warning(
                "HeatmapVLNLoss received probability heatmaps without raw logits. "
                "Using a compatibility log-probability objective; new training "
                "must pass pred_heatmap_logits explicitly."
            )
            self._warned_probability_fallback = True
        probabilities = pred_heatmaps.float()
        if bool(((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError(
                "Probability fallback expects pred_heatmaps in [0, 1]; "
                "pass raw values through pred_heatmap_logits instead"
            )
        return probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()

    def set_temperature(self, temperature: float) -> None:
        """Update the DSNT softmax temperature during training."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    @staticmethod
    def _normalise_targets(target: torch.Tensor) -> torch.Tensor:
        target_flat = target.float().reshape(target.shape[0], -1)
        return target_flat / target_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def softmax_ce_per_sample(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional pixel CE from raw decoder logits, returned per view."""
        logits_flat = pred_logits.float().reshape(pred_logits.shape[0], -1)
        target_prob = self._normalise_targets(target)
        return -(target_prob * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1)

    def softmax_ce_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional pixel CE from raw decoder logits, averaged over views."""
        return self.softmax_ce_per_sample(pred_logits, target).mean()

    def soft_argmax_coord_per_sample(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """DSNT-style normalized-coordinate SmoothL1, returned per view."""
        num_samples = pred_logits.shape[0]
        pred_weights = F.softmax(
            pred_logits.float().reshape(num_samples, -1) / self.temperature,
            dim=-1,
        ).reshape_as(pred_logits)
        target_weights = self._normalise_targets(target).reshape_as(target)

        pred_coords = torch.stack(
            (
                (pred_weights * self.coords_x).sum(dim=(-2, -1)),
                (pred_weights * self.coords_y).sum(dim=(-2, -1)),
            ),
            dim=-1,
        )
        target_coords = torch.stack(
            (
                (target_weights * self.coords_x).sum(dim=(-2, -1)),
                (target_weights * self.coords_y).sum(dim=(-2, -1)),
            ),
            dim=-1,
        )
        return F.smooth_l1_loss(
            pred_coords,
            target_coords,
            beta=self.coord_smooth_l1_beta,
            reduction="none",
        ).sum(dim=-1)

    def soft_argmax_coord_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """DSNT-style normalized-coordinate SmoothL1 averaged over views."""
        return self.soft_argmax_coord_per_sample(pred_logits, target).mean()

    @staticmethod
    def _view_macro_average(
        per_sample_loss: torch.Tensor,
        view_indices: torch.Tensor,
    ) -> torch.Tensor:
        supported = [
            per_sample_loss[view_indices == view].mean()
            for view in range(4)
            if bool((view_indices == view).any())
        ]
        if not supported:
            return per_sample_loss.sum() * 0.0
        return torch.stack(supported).mean()

    def panoramic_view_loss(
        self,
        pred_vis: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hierarchical 5-way ``[none=0, four view logits]`` cross-entropy."""
        view_logits = (
            pred_vis.detach() if self.panoramic_detach_visibility else pred_vis
        ).float()
        none_logits = torch.zeros(
            view_logits.shape[0],
            1,
            dtype=view_logits.dtype,
            device=view_logits.device,
        )
        logits = torch.cat((none_logits, view_logits), dim=-1)

        view_mass = gt_heatmaps.float().sum(dim=(-2, -1))
        total_mass = view_mass.sum(dim=-1, keepdim=True)
        fallback_mass = gt_vis.float().clamp_min(0)
        fallback_total = fallback_mass.sum(dim=-1, keepdim=True)
        target_views = torch.where(
            total_mass > 0,
            view_mass / total_mass.clamp_min(1e-8),
            fallback_mass / fallback_total.clamp_min(1e-8),
        )
        has_visible = (total_mass.squeeze(-1) > 0) | (
            fallback_total.squeeze(-1) > 0
        )
        target_none = (~has_visible).to(target_views.dtype).unsqueeze(-1)
        target = torch.cat((target_none, target_views), dim=-1)

        per_history = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if valid is not None:
            return (
                per_history * valid.to(per_history.dtype)
            ).sum() / valid.sum().clamp_min(1)
        return per_history.mean()

    def direction_macro_loss(
        self,
        pred_vis: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Macro-average direction CE without replacing the natural prior.

        A history with multiple visible directions contributes once to each
        supported direction. ``none`` examples remain represented by the
        primary visibility/panoramic losses, but do not dilute this auxiliary.
        """
        view_logits = (
            pred_vis.detach() if self.panoramic_detach_visibility else pred_vis
        ).float()
        logits = torch.cat(
            (
                torch.zeros(
                    view_logits.shape[0],
                    1,
                    dtype=view_logits.dtype,
                    device=view_logits.device,
                ),
                view_logits,
            ),
            dim=-1,
        )
        negative_log_probs = -F.log_softmax(logits, dim=-1)[:, 1:]
        supported_mask = gt_vis.bool() & (
            gt_heatmaps.float().sum(dim=(-2, -1)) > 0
        )
        if valid is not None:
            supported_mask = supported_mask & valid.unsqueeze(-1)
        per_direction = [
            negative_log_probs[supported_mask[:, view], view].mean()
            for view in range(4)
            if bool(supported_mask[:, view].any())
        ]
        if not per_direction:
            return view_logits.sum() * 0.0
        return torch.stack(per_direction).mean()

    def forward(
        self,
        pred_vis: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_vis: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        pred_heatmap_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for batched or unbatched panoramic histories."""
        raw_logits_supplied = pred_heatmap_logits is not None
        pred_vis, pred_heatmaps, gt_vis, gt_heatmaps = self._flatten_inputs(
            pred_vis,
            pred_heatmaps,
            gt_vis,
            gt_heatmaps,
        )
        if pred_heatmap_logits is None:
            spatial_logits = self._fallback_spatial_logits(pred_heatmaps)
        else:
            spatial_logits = self._flatten_raw_logits(
                pred_heatmap_logits,
                pred_heatmaps.shape,
            ).float()
            if spatial_logits.device != pred_heatmaps.device:
                raise ValueError(
                    "Raw logits and probability heatmaps must be on the same device"
                )

        device = pred_vis.device
        zero = pred_vis.float().sum() * 0.0

        if history_mask is not None:
            valid = history_mask.reshape(-1).bool()
            if valid.shape[0] != pred_vis.shape[0]:
                logger.warning(
                    "history_mask shape mismatch: mask has %d entries but "
                    "flattened pred_vis has %d rows; mask discarded.",
                    valid.shape[0],
                    pred_vis.shape[0],
                )
                valid = None
                valid_4 = None
            else:
                valid = valid.to(device=device)
                valid_4 = valid.unsqueeze(-1).expand_as(pred_vis)
        else:
            valid = None
            valid_4 = None

        pos_weight = (
            torch.tensor([self.vis_pos_weight], device=device)
            if self.vis_pos_weight != 1.0
            else None
        )
        vis_bce = F.binary_cross_entropy_with_logits(
            pred_vis.float(),
            gt_vis.float(),
            reduction="none",
            pos_weight=pos_weight,
        )
        if valid_4 is not None:
            vis_loss = (
                vis_bce * valid_4.to(vis_bce.dtype)
            ).sum() / valid_4.sum().clamp_min(1)
        else:
            vis_loss = vis_bce.mean()

        target_has_mass = gt_heatmaps.float().sum(dim=(-2, -1)) > 0
        pos_mask = gt_vis.bool() & target_has_mass
        if valid is not None:
            pos_mask = pos_mask & valid.unsqueeze(-1)
        pos_indices = pos_mask.nonzero(as_tuple=False)
        has_pos = pos_indices.shape[0] > 0

        peak_per_sample = None
        coord_per_sample = None
        if has_pos:
            pred_pos_logits = spatial_logits[pos_mask]
            gt_pos = gt_heatmaps[pos_mask]
            view_indices = pos_indices[:, 1]
        else:
            pred_pos_logits = None
            gt_pos = None
            view_indices = None

        if has_pos and (self.lambda_peak > 0 or self.lambda_view_macro > 0):
            peak_per_sample = self.softmax_ce_per_sample(pred_pos_logits, gt_pos)
            peak_loss = peak_per_sample.mean()
        else:
            peak_loss = zero

        if has_pos and self.lambda_coord > 0:
            coord_per_sample = self.soft_argmax_coord_per_sample(
                pred_pos_logits,
                gt_pos,
            )
            coord_loss = coord_per_sample.mean()
        else:
            coord_loss = zero

        if has_pos and self.lambda_view_macro > 0:
            if peak_per_sample is None:
                peak_per_sample = self.softmax_ce_per_sample(
                    pred_pos_logits,
                    gt_pos,
                )
            macro_peak = self._view_macro_average(
                peak_per_sample,
                view_indices,
            )
            if self.lambda_coord > 0:
                if coord_per_sample is None:
                    coord_per_sample = self.soft_argmax_coord_per_sample(
                        pred_pos_logits,
                        gt_pos,
                    )
                macro_coord = self._view_macro_average(
                    coord_per_sample,
                    view_indices,
                )
            else:
                macro_coord = zero
            view_macro_loss = macro_peak + self.lambda_coord * macro_coord
        else:
            view_macro_loss = zero

        if self.lambda_neg > 0:
            neg_mask = ~gt_vis.bool()
            if valid is not None:
                neg_mask = neg_mask & valid.unsqueeze(-1)
            if bool(neg_mask.any()):
                if raw_logits_supplied:
                    neg_loss = F.softplus(spatial_logits[neg_mask]).mean()
                else:
                    max_probability = 1.0 - torch.finfo(torch.float32).eps
                    neg_loss = -torch.log1p(
                        -pred_heatmaps[neg_mask].float().clamp_max(max_probability)
                    ).mean()
            else:
                neg_loss = zero
        else:
            neg_loss = zero

        if self.lambda_panoramic_view > 0:
            pano_view_loss = self.panoramic_view_loss(
                pred_vis,
                gt_vis,
                gt_heatmaps,
                valid,
            )
        else:
            pano_view_loss = zero

        if self.lambda_direction_macro > 0:
            direction_macro_loss = self.direction_macro_loss(
                pred_vis,
                gt_vis,
                gt_heatmaps,
                valid,
            )
        else:
            direction_macro_loss = zero

        total = (
            self.lambda_vis * vis_loss
            + self.lambda_peak * peak_loss
            + self.lambda_coord * coord_loss
            + self.lambda_neg * neg_loss
            + self.lambda_view_macro * view_macro_loss
            + self.lambda_panoramic_view * pano_view_loss
            + self.lambda_direction_macro * direction_macro_loss
        )
        return {
            "total": total,
            "monitor_total": total.detach(),
            "vis_loss": vis_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "peak_loss": peak_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "view_macro_loss": view_macro_loss.detach(),
            "panoramic_view_loss": pano_view_loss.detach(),
            "direction_macro_loss": direction_macro_loss.detach(),
        }
