"""
Validation loop.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from src.models.pipeline import VLNPipeline
from src.models.future_trajectory_objective import (
    future_tube_metrics_from_statistics,
    future_tube_sufficient_statistics,
)
from src.models.past_plan_action import compute_shared_plan_action_losses
from src.models.past_plan_action_loss import (
    PastPlanActionLossWeights,
    compose_past_plan_action_loss,
)
from src.data.future_trajectory_batch import assert_no_future_teacher_inputs
from src.utils.gpu_heatmap import GPUHeatmapComputer

from .distributed import DistributedContext, _dist_all_reduce_in_place
from .utils import (
    _unwrap_model,
    build_future_heatmap_loss_fn,
    build_heatmap_loss_fn,
    make_autocast_context,
)
from .visualization import (
    _should_use_gpu_gt,
    visualize_heatmap_predictions,
)
from .train_loop import _prepare_trajectory_sequence_inputs

logger = logging.getLogger(__name__)

_HEATMAP_VIEW_NAMES = ("front", "right", "back", "left")
_HEATMAP_COMPONENT_KEYS = (
    "peak_loss",
    "vis_loss",
    "coord_loss",
    "neg_loss",
    "view_macro_loss",
    "direction_macro_loss",
    "panoramic_view_loss",
)


class _HeatmapJointMetricAccumulator:
    """Additive validation statistics for end-to-end panoramic localization.

    Each real history slot contributes one 5-way target:
    ``none/front/right/back/left``.  Visible slots additionally contribute one
    conditional spatial error in their primary GT view.  Joint PCK requires
    both the correct view and a spatial error within the threshold.

    The histogram stores squared integer-pixel errors.  It makes median/P90
    exactly mergeable across DDP ranks with SUM all-reduce, unlike averaging
    rank-local quantiles.
    """

    _VALID = 0
    _VISIBLE = 1
    _NONE = 2
    _VIEW5_CORRECT = 3
    _JOINT_PCK4_CORRECT = 4
    _JOINT_PCK8_CORRECT = 5
    _PER_VIEW_PCK8_START = 6
    _PER_VIEW_COUNT_START = 10
    _NUM_COUNTS = 14

    def __init__(
        self,
        *,
        heatmap_size: tuple[int, int],
        device: torch.device,
    ):
        height, width = (int(value) for value in heatmap_size)
        if height <= 0 or width <= 0:
            raise ValueError(f"heatmap_size must be positive, got {heatmap_size}")
        self.heatmap_size = (height, width)
        self.max_squared_error = (height - 1) ** 2 + (width - 1) ** 2
        self.counts = torch.zeros(
            self._NUM_COUNTS,
            dtype=torch.float64,
            device=device,
        )
        self.pixel_error_histogram = torch.zeros(
            self.max_squared_error + 1,
            dtype=torch.float64,
            device=device,
        )

    @staticmethod
    def _flatten_heatmap(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.ndim not in (4, 5) or tensor.shape[-3] != 4:
            raise ValueError(
                f"{name} must be [N,4,H,W] or [B,N,4,H,W], got "
                f"{tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, 4, tensor.shape[-2], tensor.shape[-1])

    @staticmethod
    def _flatten_visibility(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.ndim not in (2, 3) or tensor.shape[-1] != 4:
            raise ValueError(
                f"{name} must be [N,4] or [B,N,4], got {tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, 4)

    def update(
        self,
        *,
        pred_visibility_logits: torch.Tensor,
        pred_heatmaps: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> None:
        pred_vis = self._flatten_visibility(
            pred_visibility_logits.detach().float(),
            name="pred_visibility_logits",
        )
        gt_vis = self._flatten_visibility(
            gt_visibility.detach().float(),
            name="gt_visibility",
        )
        pred_hm = self._flatten_heatmap(
            pred_heatmaps.detach().float(),
            name="pred_heatmaps",
        )
        gt_hm = self._flatten_heatmap(
            gt_heatmaps.detach().float(),
            name="gt_heatmaps",
        )
        if pred_vis.shape != gt_vis.shape:
            raise ValueError(
                f"Visibility shape mismatch: {tuple(pred_vis.shape)} != "
                f"{tuple(gt_vis.shape)}"
            )
        if pred_hm.shape[:2] != pred_vis.shape or gt_hm.shape[:2] != gt_vis.shape:
            raise ValueError("Visibility and heatmap history/view shapes differ")
        if pred_hm.shape[0] != gt_hm.shape[0]:
            raise ValueError(
                f"Prediction/target history count differs: "
                f"{pred_hm.shape[0]} != {gt_hm.shape[0]}"
            )
        if pred_hm.shape[-2:] != gt_hm.shape[-2:]:
            original = pred_hm.shape
            pred_hm = F.interpolate(
                pred_hm.reshape(-1, 1, *original[-2:]),
                size=gt_hm.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(*original[:-2], *gt_hm.shape[-2:])

        num_histories = pred_vis.shape[0]
        if history_mask is None:
            valid = torch.ones(
                num_histories,
                dtype=torch.bool,
                device=pred_vis.device,
            )
        else:
            valid = history_mask.detach().reshape(-1).to(
                device=pred_vis.device,
                dtype=torch.bool,
            )
            if valid.numel() != num_histories:
                raise ValueError(
                    "history_mask does not match flattened heatmap histories: "
                    f"{valid.numel()} != {num_histories}"
                )
        if not bool(valid.any()):
            return

        pred_vis = pred_vis[valid]
        gt_vis = gt_vis[valid]
        pred_hm = pred_hm[valid]
        gt_hm = gt_hm[valid]

        # A primary view is needed for a strict 5-way accuracy and per-view
        # accounting. Select the highest target peak among declared visible
        # views; slots without valid positive heatmap mass map to "none".
        gt_peak_per_view = gt_hm.amax(dim=(-2, -1))
        eligible_view = (gt_vis > 0.5) & (gt_peak_per_view > 0)
        masked_gt_peak = gt_peak_per_view.masked_fill(
            ~eligible_view,
            -torch.inf,
        )
        visible = eligible_view.any(dim=-1)
        gt_view = masked_gt_peak.argmax(dim=-1)
        gt_class = torch.where(visible, gt_view + 1, torch.zeros_like(gt_view))

        none_logit = torch.zeros(
            pred_vis.shape[0],
            1,
            dtype=pred_vis.dtype,
            device=pred_vis.device,
        )
        pred_class = torch.cat((none_logit, pred_vis), dim=-1).argmax(dim=-1)
        view5_correct = pred_class == gt_class

        self.counts[self._VALID] += float(valid.sum().item())
        self.counts[self._VISIBLE] += visible.sum()
        self.counts[self._NONE] += (~visible).sum()
        self.counts[self._VIEW5_CORRECT] += view5_correct.sum()
        if not bool(visible.any()):
            return

        visible_rows = torch.nonzero(visible, as_tuple=False).flatten()
        target_views = gt_view[visible_rows]
        pred_maps_in_target_view = pred_hm[visible_rows, target_views]
        gt_maps_in_target_view = gt_hm[visible_rows, target_views]
        width = int(gt_hm.shape[-1])

        pred_peak = pred_maps_in_target_view.reshape(
            visible_rows.numel(),
            -1,
        ).argmax(dim=-1)
        gt_peak = gt_maps_in_target_view.reshape(
            visible_rows.numel(),
            -1,
        ).argmax(dim=-1)
        pred_y = torch.div(pred_peak, width, rounding_mode="floor")
        pred_x = torch.remainder(pred_peak, width)
        gt_y = torch.div(gt_peak, width, rounding_mode="floor")
        gt_x = torch.remainder(gt_peak, width)
        squared_error = (pred_x - gt_x).square() + (pred_y - gt_y).square()
        if squared_error.numel() and int(squared_error.max().item()) > self.max_squared_error:
            raise ValueError(
                "Observed pixel error exceeds the configured heatmap histogram. "
                f"configured_size={self.heatmap_size}, target_size="
                f"{tuple(gt_hm.shape[-2:])}"
            )
        self.pixel_error_histogram += torch.bincount(
            squared_error,
            minlength=self.pixel_error_histogram.numel(),
        ).to(dtype=self.pixel_error_histogram.dtype)

        visible_view_correct = view5_correct[visible_rows]
        within4 = squared_error <= 4 ** 2
        within8 = squared_error <= 8 ** 2
        joint4 = visible_view_correct & within4
        joint8 = visible_view_correct & within8
        self.counts[self._JOINT_PCK4_CORRECT] += joint4.sum()
        self.counts[self._JOINT_PCK8_CORRECT] += joint8.sum()

        for view in range(4):
            view_rows = target_views == view
            self.counts[self._PER_VIEW_COUNT_START + view] += view_rows.sum()
            self.counts[self._PER_VIEW_PCK8_START + view] += (
                joint8 & view_rows
            ).sum()

    def all_reduce(self) -> None:
        """SUM sufficient statistics across all initialized DDP ranks."""
        _dist_all_reduce_in_place(self.counts)
        _dist_all_reduce_in_place(self.pixel_error_histogram)

    @staticmethod
    def _histogram_quantile(histogram: torch.Tensor, quantile: float) -> float:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0,1], got {quantile}")
        histogram = histogram.detach().to(dtype=torch.float64, device="cpu")
        total = int(histogram.sum().item())
        if total == 0:
            return 0.0

        position = (total - 1) * quantile
        lower_rank = math.floor(position)
        upper_rank = math.ceil(position)
        cumulative = histogram.cumsum(dim=0)

        def value_at(rank: int) -> float:
            index = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(float(rank + 1), dtype=cumulative.dtype),
                    right=False,
                ).item()
            )
            return math.sqrt(index)

        lower = value_at(lower_rank)
        upper = value_at(upper_rank)
        return lower + (position - lower_rank) * (upper - lower)

    def compute(self) -> dict[str, float]:
        valid = self.counts[self._VALID].item()
        visible = self.counts[self._VISIBLE].item()
        none = self.counts[self._NONE].item()
        metrics = {
            "val_heatmap_joint_pck4": (
                self.counts[self._JOINT_PCK4_CORRECT].item() / visible
                if visible > 0
                else 0.0
            ),
            "val_heatmap_joint_pck8": (
                self.counts[self._JOINT_PCK8_CORRECT].item() / visible
                if visible > 0
                else 0.0
            ),
            "val_heatmap_pixel_error_median": self._histogram_quantile(
                self.pixel_error_histogram,
                0.5,
            ),
            "val_heatmap_pixel_error_p90": self._histogram_quantile(
                self.pixel_error_histogram,
                0.9,
            ),
            "val_heatmap_view5_accuracy": (
                self.counts[self._VIEW5_CORRECT].item() / valid
                if valid > 0
                else 0.0
            ),
            "val_heatmap_valid_count": valid,
            "val_heatmap_visible_count": visible,
            "val_heatmap_none_count": none,
        }
        supported_direction_pck8 = []
        for view, name in enumerate(_HEATMAP_VIEW_NAMES):
            count = self.counts[self._PER_VIEW_COUNT_START + view].item()
            correct = self.counts[self._PER_VIEW_PCK8_START + view].item()
            per_view_pck8 = correct / count if count > 0 else 0.0
            metrics[f"val_heatmap_{name}_pck8"] = per_view_pck8
            metrics[f"val_heatmap_{name}_count"] = count
            if count > 0:
                supported_direction_pck8.append(per_view_pck8)
        metrics["val_heatmap_macro_joint_pck8"] = (
            sum(supported_direction_pck8) / len(supported_direction_pck8)
            if supported_direction_pck8
            else 0.0
        )
        metrics["val_heatmap_supported_direction_count"] = float(
            len(supported_direction_pck8)
        )
        return metrics


def _select_stop_hysteresis_thresholds(
    thresholds: torch.Tensor,
    confusion: torch.Tensor,
    *,
    max_add_false_positive_rate: float,
    min_veto_recall: float,
    min_add_threshold: float = 0.0,
    max_veto_threshold: float = 1.0,
) -> dict[str, float]:
    """Select asymmetric STOP thresholds from aggregate validation counts.

    ``confusion`` columns are TP, FP, TN, FN for each ascending threshold.
    Adding STOP is FPR-constrained; accepting an original STOP is
    recall-constrained. Conservative floor/ceiling values protect the frozen
    InternNav prior under closed-loop distribution shift.
    """
    thresholds = torch.as_tensor(thresholds, dtype=torch.float64).flatten().cpu()
    confusion = torch.as_tensor(confusion, dtype=torch.float64).cpu()
    if confusion.shape != (thresholds.numel(), 4):
        raise ValueError(
            "STOP threshold confusion must have shape (num_thresholds, 4), got "
            f"{tuple(confusion.shape)}"
        )
    if thresholds.numel() < 2 or bool((thresholds[1:] < thresholds[:-1]).any()):
        raise ValueError("STOP thresholds must be an ascending grid")
    if not 0.0 <= max_add_false_positive_rate < 1.0:
        raise ValueError("max_add_false_positive_rate must be in [0, 1)")
    if not 0.0 < min_veto_recall <= 1.0:
        raise ValueError("min_veto_recall must be in (0, 1]")
    if not 0.0 <= min_add_threshold <= 1.0:
        raise ValueError("min_add_threshold must be in [0, 1]")
    if not 0.0 <= max_veto_threshold <= 1.0:
        raise ValueError("max_veto_threshold must be in [0, 1]")

    tp, fp, tn, fn = confusion.unbind(dim=1)
    positive_count = tp + fn
    negative_count = fp + tn
    if positive_count.max().item() <= 0 or negative_count.max().item() <= 0:
        raise ValueError("STOP threshold calibration requires both classes")
    recall = tp / positive_count.clamp_min(1.0)
    false_positive_rate = fp / negative_count.clamp_min(1.0)
    precision = tp / (tp + fp).clamp_min(1.0)
    threshold_tolerance = 1e-7

    add_candidates = torch.nonzero(
        (false_positive_rate <= max_add_false_positive_rate)
        & (thresholds >= min_add_threshold - threshold_tolerance),
        as_tuple=False,
    ).flatten()
    add_index = int(add_candidates[0]) if add_candidates.numel() else thresholds.numel() - 1

    veto_candidates = torch.nonzero(
        (recall >= min_veto_recall)
        & (thresholds <= max_veto_threshold + threshold_tolerance),
        as_tuple=False,
    ).flatten()
    veto_index = int(veto_candidates[-1]) if veto_candidates.numel() else 0

    return {
        "add_stop_threshold": float(thresholds[add_index].item()),
        "veto_stop_threshold": float(thresholds[veto_index].item()),
        "add_false_positive_rate": float(false_positive_rate[add_index].item()),
        "add_recall": float(recall[add_index].item()),
        "add_precision": float(precision[add_index].item()),
        "veto_false_positive_rate": float(false_positive_rate[veto_index].item()),
        "veto_recall": float(recall[veto_index].item()),
        "veto_precision": float(precision[veto_index].item()),
        "positive_count": float(positive_count[0].item()),
        "negative_count": float(negative_count[0].item()),
    }


@torch.inference_mode()
def validate(
    model: VLNPipeline,
    val_loader: DataLoader,
    cfg: dict,
    logger,
    stage_cfg: dict,
    tb_writer: SummaryWriter | None = None,
    epoch: int = 0,
    vis_dir: Path | None = None,
    max_batches: int | None = None,
    gpu_heatmap_computer: GPUHeatmapComputer | None = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    heatmap_temperature: float | None = None,
    dist_context: DistributedContext | None = None,
) -> dict[str, float]:
    """Validation with optional visualization."""
    dist_context = dist_context or DistributedContext(
        enabled=False,
        device=torch.device(cfg['model'].get('device', 'cuda')),
    )
    model_module = _unwrap_model(model)
    model.eval()

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    total_lm_loss = 0.0
    total_stop_loss = 0.0
    total_future_heatmap_loss = 0.0
    total_preserve_loss = 0.0
    total_delta_z_loss = 0.0
    total_heatmap_mse = 0.0
    num_heatmap_mse_batches = 0
    num_batches = 0
    vis_tp = vis_tn = vis_fp = vis_fn = 0
    total_peak_loss = 0.0
    total_vis_loss = 0.0
    total_coord_loss = 0.0

    loss_cfg = cfg['loss']
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
    train_stop_head = bool(stage_cfg.get('train_system2_stop_head', False))
    past_plan_action_stage = stage_cfg.get('past_plan_action_stage')
    train_past_plan_action = past_plan_action_stage is not None
    need_heatmap_targets = train_history
    if train_past_plan_action and (train_lm or train_stop_head):
        raise ValueError(
            "Past→Plan→Action v1 cannot be combined with LM or STOP-head validation"
        )

    device = dist_context.device
    model_heatmap_cfg = cfg.get('model', {}).get('heatmap', {})
    if not isinstance(model_heatmap_cfg, dict):
        model_heatmap_cfg = {}
    metric_heatmap_size = model_heatmap_cfg.get(
        'heatmap_size',
        cfg.get('data', {}).get('init_hm_size', (64, 64)),
    )
    heatmap_joint_metrics = _HeatmapJointMetricAccumulator(
        heatmap_size=tuple(metric_heatmap_size),
        device=device,
    )
    total_heatmap_components = torch.zeros(
        len(_HEATMAP_COMPONENT_KEYS),
        device=device,
        dtype=torch.float64,
    )

    hm_loss_fn = build_heatmap_loss_fn(
        cfg, device,
        temperature=heatmap_temperature,
    )
    future_hm_loss_fn = (
        build_future_heatmap_loss_fn(cfg, device) if train_future else None
    )
    future_tube_stats = torch.zeros(17, device=device, dtype=torch.float64)
    ppa_weights_cfg = loss_cfg.get('past_plan_action', {}) or {}
    ppa_weights = PastPlanActionLossWeights(
        action=float(ppa_weights_cfg.get('action', 1.0)),
        history=float(ppa_weights_cfg.get('history', 0.3)),
        future=float(ppa_weights_cfg.get('future', 0.3)),
        preserve=float(ppa_weights_cfg.get('preserve', 0.5)),
        delta_z=float(ppa_weights_cfg.get('delta_z', 0.01)),
    )

    val_inference_batches = cfg.get('validation', {}).get('val_inference_batches', 10)
    validation_cfg = cfg.get('validation', {})
    stop_grid_steps = max(int(validation_cfg.get('stop_threshold_grid_steps', 200)), 2)
    stop_thresholds = torch.linspace(
        0.0,
        1.0,
        stop_grid_steps + 1,
        device=device,
        dtype=torch.float64,
    )
    stop_confusion_grid = torch.zeros(
        (stop_thresholds.numel(), 4),
        device=device,
        dtype=torch.float64,
    )

    total_val_batches = len(val_loader)
    if max_batches is not None:
        total_val_batches = min(total_val_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式(验证): 只处理 {total_val_batches} batches")

    logger.info(f"  📊 验证: {total_val_batches} batches (training loss), "
                f"{val_inference_batches} batches (推理 MSE)")
    logger.info(f"  🌡️ Heatmap temperature: {hm_loss_fn.temperature:.3f}")

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating", total=total_val_batches, disable=not dist_context.is_main)):
            if max_batches is not None and i >= max_batches:
                break
            history_frames = batch['history_frames']
            current_frame = batch['current_frame']
            _B, _K, _C, _H, _W = history_frames.shape

            gt_action = batch['action'].to(device, non_blocking=True)
            action_valid = batch['action_valid'].to(device, non_blocking=True)
            is_stop = batch['is_stop'].to(device, non_blocking=True)
            stop_target = batch.get('system2_stop_target')
            stop_predictor_positions = batch.get('system2_stop_predictor_position')
            if train_stop_head:
                if stop_target is None or stop_predictor_positions is None:
                    raise RuntimeError(
                        "train_system2_stop_head=True requires STOP validation targets "
                        "and predictor positions"
                    )
                stop_target = stop_target.to(device, non_blocking=True)
            text = batch['text']

            gt_heatmap = None
            if need_heatmap_targets:
                if _should_use_gpu_gt(batch, gpu_heatmap_computer):
                    history_poses = batch['history_poses'].to(device, non_blocking=True)
                    current_poses = batch['current_pose'].to(device, non_blocking=True)
                    current_depths = batch['current_depth'].to(device) if gpu_has_depth and 'current_depth' in batch else None
                    intrinsics = batch['intrinsics'].to(device) if 'intrinsics' in batch else None
                    gt_heatmap = gpu_heatmap_computer.compute_batch(
                        history_poses=history_poses,
                        current_poses=current_poses,
                        current_depths=current_depths,
                        intrinsics=intrinsics,
                        depth_normalized=gpu_depth_normalized,
                    )
                else:
                    gt_heatmap = batch['heatmap'].to(device, non_blocking=True)

            with make_autocast_context(device, cfg.get('optim', {}).get('amp', 'bf16')):
                if text and len(text) > 0:
                    instruction_text = list(text)
                else:
                    instruction_text = None
                current_views_batch = batch.get('current_views')
                history_panoramas_batch = batch.get('history_panoramas')
                panoramic_inputs_batch = batch.get('pano_inputs')
                panoramic_num_histories = batch.get('pano_num_histories')
                panoramic_text_anchor_positions = batch.get('pano_text_anchor_positions')
                history_rel_poses = batch.get('history_rel_poses')
                if history_rel_poses is not None:
                    history_rel_poses = history_rel_poses.to(device, non_blocking=True)
                if panoramic_inputs_batch is not None and not train_action:
                    video_frames = current_frame.unsqueeze(1)
                else:
                    video_frames = torch.cat([
                        history_frames,
                        history_frames[:, -1:],
                    ], dim=1)

                if train_future:
                    assert_no_future_teacher_inputs(batch)
                output = model(
                    video_frames=video_frames,
                    instruction_text=instruction_text,
                    current_observation=current_frame,
                    current_views=current_views_batch,
                    history_panoramas=history_panoramas_batch,
                    panoramic_inputs=panoramic_inputs_batch,
                    panoramic_num_histories=panoramic_num_histories,
                    panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                    stop_predictor_positions=(
                        stop_predictor_positions if train_stop_head else None
                    ),
                    history_rel_poses=history_rel_poses,
                    return_heatmaps=train_history,
                    return_future_heatmaps=train_future,
                    return_heatmap_logits=train_history,
                    return_actions=train_action,
                    return_lm_loss=train_lm,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=(
                        stop_target
                        if train_stop_head
                        else (is_stop if train_action else None)
                    ),
                )

            heatmap_loss = torch.tensor(0.0, device=device)
            loss_dict = None
            if train_history and 'visibility' in output and 'heatmaps' in output and gt_heatmap is not None:
                if 'gt_visibility' in batch:
                    gt_vis = batch['gt_visibility'].to(device)
                else:
                    gt_vis = gt_heatmap.amax(dim=(-2, -1)).clamp(0, 1).to(device)
                hm_history_mask = batch.get('history_mask')
                if hm_history_mask is not None:
                    hm_history_mask = hm_history_mask.to(device)
                loss_dict = hm_loss_fn(
                    output['visibility'],
                    output['heatmaps'],
                    gt_vis=gt_vis,
                    gt_heatmaps=gt_heatmap.to(device),
                    history_mask=hm_history_mask,
                    pred_heatmap_logits=output.get('heatmap_logits'),
                )
                heatmap_loss = loss_dict['total']
                total_peak_loss += loss_dict.get('peak_loss', torch.tensor(0.0)).item()
                total_vis_loss += loss_dict.get('vis_loss', torch.tensor(0.0)).item()
                total_coord_loss += loss_dict.get('coord_loss', torch.tensor(0.0)).item()
                for component_index, component_key in enumerate(
                    _HEATMAP_COMPONENT_KEYS
                ):
                    component = loss_dict.get(component_key)
                    if torch.is_tensor(component):
                        total_heatmap_components[component_index] += (
                            component.detach().to(dtype=torch.float64)
                        )
                heatmap_joint_metrics.update(
                    pred_visibility_logits=output['visibility'],
                    pred_heatmaps=output.get('heatmap_logits', output['heatmaps']),
                    gt_visibility=gt_vis,
                    gt_heatmaps=gt_heatmap,
                    history_mask=hm_history_mask,
                )

            future_loss_dict = None
            future_heatmap_loss = torch.tensor(0.0, device=device)
            if train_future:
                required_batch = (
                    'future_trajectory_heatmap',
                    'future_trajectory_visibility',
                    'future_trajectory_time_mask',
                )
                required_output = (
                    'future_visibility',
                    'future_heatmaps',
                    'future_heatmap_logits',
                )
                missing_batch = [key for key in required_batch if key not in batch]
                missing_output = [key for key in required_output if key not in output]
                if missing_batch or missing_output:
                    raise RuntimeError(
                        "Future validation supervision is incomplete: "
                        f"batch_missing={missing_batch} output_missing={missing_output}"
                    )
                assert future_hm_loss_fn is not None
                gt_future_visibility = batch['future_trajectory_visibility'].to(device)
                gt_future_heatmap = batch['future_trajectory_heatmap'].to(device)
                future_time_mask = batch['future_trajectory_time_mask'].to(device)
                future_loss_dict = future_hm_loss_fn(
                    pred_visibility_logits=output['future_visibility'],
                    pred_heatmaps=output['future_heatmaps'],
                    pred_heatmap_logits=output['future_heatmap_logits'],
                    gt_visibility=gt_future_visibility,
                    gt_heatmaps=gt_future_heatmap,
                    future_time_mask=future_time_mask,
                )
                future_heatmap_loss = future_loss_dict['total']
                future_tube_stats += future_tube_sufficient_statistics(
                    pred_visibility_logits=output['future_visibility'],
                    pred_heatmaps=output['future_heatmaps'],
                    gt_visibility=gt_future_visibility,
                    gt_heatmaps=gt_future_heatmap,
                    future_time_mask=future_time_mask,
                )

            if 'visibility' in output and output['visibility'] is not None:
                pred_vis_logits = output['visibility'].detach()
                gt_vis_batch = batch.get('gt_visibility')
                if gt_vis_batch is None and gt_heatmap is not None:
                    gt_vis_batch = (gt_heatmap.amax(dim=(-2, -1)) > 0).float()
                if gt_vis_batch is not None:
                    pv = (torch.sigmoid(pred_vis_logits.float()).reshape(-1) > 0.5).float()
                    gv = (gt_vis_batch.to(pred_vis_logits.device).reshape(-1) > 0.5).float()
                    vis_tp += ((pv == 1) & (gv == 1)).sum().item()
                    vis_tn += ((pv == 0) & (gv == 0)).sum().item()
                    vis_fp += ((pv == 1) & (gv == 0)).sum().item()
                    vis_fn += ((pv == 0) & (gv == 1)).sum().item()

            trajectory_loss = torch.tensor(0.0, device=device)
            action_plan_losses = None

            if train_action:
                if hasattr(model_module, 'nextdit_action_head') and model_module.nextdit_action_head is not None:
                    if 'trajectory' not in batch:
                        raise RuntimeError(
                            "train_action=True but validation batch has no trajectory target."
                        )
                    if 'traj_hidden_states' not in output:
                        raise RuntimeError(
                            "train_action=True but validation output has no traj_hidden_states."
                        )
                    gt_trajectory = batch['trajectory'].to(device)
                    trajectory_valid = batch['trajectory_valid'].to(device)
                    traj_images = batch.get('traj_images')
                    if traj_images is not None:
                        traj_images = traj_images.to(device)
                    gt_trajectory, trajectory_valid, traj_images = (
                        _prepare_trajectory_sequence_inputs(
                            gt_trajectory,
                            trajectory_valid,
                            traj_images,
                            mode=str(
                                stage_cfg.get('trajectory_sequence_mode', 'all')
                            ),
                        )
                    )
                    if train_past_plan_action:
                        action_plan_losses = compute_shared_plan_action_losses(
                            action_head=model_module.nextdit_action_head,
                            plan_z0=output['plan_z0'],
                            plan_z=output['plan_z'],
                            gt_trajectory=gt_trajectory,
                            trajectory_valid=trajectory_valid,
                            traj_images=traj_images,
                            preserve_weight=ppa_weights.preserve,
                            delta_weight=ppa_weights.delta_z,
                        )
                        trajectory_loss = action_plan_losses['action']
                    else:
                        traj_hidden_states = model_module.adapt_traj_hidden_states(
                            output['traj_hidden_states']
                        )
                        traj_result = model_module.nextdit_action_head.compute_loss(
                            traj_hidden_states,
                            gt_trajectory,
                            traj_images=traj_images,
                            trajectory_valid=trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']

            lm_loss = torch.tensor(0.0, device=device)
            if train_lm:
                if 'lm_loss' not in output or output['lm_loss'] is None:
                    raise RuntimeError(
                        "train_lm=True but validation model output has no lm_loss."
                    )
                lm_loss = output['lm_loss']

            stop_loss = torch.tensor(0.0, device=device)
            if train_stop_head:
                if 'stop_loss' not in output or output['stop_loss'] is None:
                    raise RuntimeError(
                        "train_system2_stop_head=True but validation output has no stop_loss"
                    )
                stop_loss = output['stop_loss']
                stop_probabilities = output['stop_probability'].detach().float()
                targets = stop_target.detach() >= 0.5
                predictions = (
                    stop_probabilities.unsqueeze(0)
                    >= stop_thresholds.to(dtype=stop_probabilities.dtype).unsqueeze(1)
                )
                stop_confusion_grid += torch.stack(
                    [
                        (predictions & targets.unsqueeze(0)).sum(dim=1),
                        (predictions & ~targets.unsqueeze(0)).sum(dim=1),
                        (~predictions & ~targets.unsqueeze(0)).sum(dim=1),
                        (~predictions & targets.unsqueeze(0)).sum(dim=1),
                    ],
                    dim=1,
                ).to(dtype=torch.float64)

            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 0.0)
            lm_weight = loss_cfg.get('lm_weight', stage_cfg.get('lm_weight', 1.0))
            stop_weight = loss_cfg.get('stop_weight', 1.0)

            ppa_loss_dict = None
            if train_past_plan_action:
                ppa_loss_dict = compose_past_plan_action_loss(
                    stage=past_plan_action_stage,
                    history_loss=loss_dict if train_history else None,
                    future_loss=future_loss_dict,
                    action_plan_losses=action_plan_losses,
                    weights=ppa_weights,
                )
                loss = ppa_loss_dict['total']
            else:
                loss = (
                    heatmap_weight * heatmap_loss
                    + trajectory_weight * trajectory_loss
                    + lm_weight * lm_loss
                    + stop_weight * stop_loss
                )

            total_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_action_loss += trajectory_loss.item()
            total_lm_loss += lm_loss.item()
            total_stop_loss += stop_loss.item()
            total_future_heatmap_loss += future_heatmap_loss.item()
            if ppa_loss_dict is not None:
                total_preserve_loss += ppa_loss_dict['preserve'].item()
                total_delta_z_loss += ppa_loss_dict['delta_z_l2'].item()
            num_batches += 1

            # Reuse current output for inference MSE + visualization
            num_vis_batches = cfg['log'].get('val_vis_batches', 2)
            if num_batches <= val_inference_batches:
                try:
                    vis_output = output
                    if train_history and gt_heatmap is not None and 'heatmaps' in vis_output:
                        infer_pred_hm = vis_output.get('heatmaps_gated', vis_output['heatmaps']).to(device)
                        gt_hm_eval = gt_heatmap.to(device)
                        if infer_pred_hm.shape[-2:] != gt_hm_eval.shape[-2:]:
                            orig = infer_pred_hm.shape
                            infer_pred_hm = F.interpolate(
                                infer_pred_hm.reshape(-1, 1, *orig[-2:]),
                                size=gt_hm_eval.shape[-2:],
                                mode='bilinear', align_corners=False,
                            ).reshape(*orig[:-2], *gt_hm_eval.shape[-2:])
                        hm_mask = batch.get('history_mask')
                        mask_usable = (
                            hm_mask is not None
                            and infer_pred_hm.dim() >= 4
                            and tuple(hm_mask.shape) == tuple(infer_pred_hm.shape[:2])
                        )
                        if mask_usable:
                            m = hm_mask.to(device).float()
                            while m.dim() < infer_pred_hm.dim():
                                m = m.unsqueeze(-1)
                            m = m.expand_as(infer_pred_hm)
                            sq_err = (infer_pred_hm - gt_hm_eval).square()
                            batch_mse = (sq_err * m).sum() / m.sum().clamp(min=1)
                        else:
                            batch_mse = F.mse_loss(infer_pred_hm, gt_hm_eval)
                        total_heatmap_mse += batch_mse.item()
                        num_heatmap_mse_batches += 1

                    if (
                        dist_context.is_main
                        and train_history
                        and 'heatmaps' in vis_output
                        and num_batches <= num_vis_batches
                        and vis_dir is not None
                    ):
                        vis_path = visualize_heatmap_predictions(
                            model=model_module,
                            batch=batch,
                            output=vis_output,
                            epoch=epoch,
                            step=num_batches,
                            output_dir=vis_dir,
                            num_samples=4,
                            gt_heatmap_override=gt_heatmap if _should_use_gpu_gt(batch, gpu_heatmap_computer) else None,
                        )

                        if vis_path is not None:
                            if tb_writer is not None:
                                import cv2
                                vis_img = cv2.imread(str(vis_path))
                                if vis_img is not None:
                                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                    vis_img = vis_img.transpose(2, 0, 1)
                                    tb_writer.add_image(f'val/heatmap_viz_batch{num_batches}', vis_img, epoch)

                            logger.info(f"[VAL-VIS] Epoch {epoch}, Batch {num_batches} visualization saved")
                except Exception as e:
                    logger.warning("Validation inference/visualization failed: %s", e, exc_info=True)

            del output, gt_heatmap

    totals = torch.tensor(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_heatmap_mse,
            float(num_batches),
            float(num_heatmap_mse_batches),
            float(vis_tp),
            float(vis_tn),
            float(vis_fp),
            float(vis_fn),
            total_peak_loss,
            total_vis_loss,
            total_coord_loss,
            total_lm_loss,
            total_stop_loss,
            total_future_heatmap_loss,
            total_preserve_loss,
            total_delta_z_loss,
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)
    _dist_all_reduce_in_place(stop_confusion_grid)
    _dist_all_reduce_in_place(total_heatmap_components)
    _dist_all_reduce_in_place(future_tube_stats)
    heatmap_joint_metrics.all_reduce()
    val_heatmap_joint_metrics = heatmap_joint_metrics.compute()

    reduced_num_batches = max(int(totals[4].item()), 1)
    reduced_num_heatmap_mse_batches = int(totals[5].item())
    avg_loss = (totals[0] / reduced_num_batches).item()
    avg_hm = (totals[1] / reduced_num_batches).item()
    avg_act = (totals[2] / reduced_num_batches).item()
    avg_hm_mse = (totals[3] / max(reduced_num_heatmap_mse_batches, 1)).item() if reduced_num_heatmap_mse_batches > 0 else 0.0
    avg_peak_loss = (totals[10] / reduced_num_batches).item()
    avg_vis_loss = (totals[11] / reduced_num_batches).item()
    avg_coord_loss = (totals[12] / reduced_num_batches).item()
    avg_lm_loss = (totals[13] / reduced_num_batches).item()
    avg_stop_loss = (totals[14] / reduced_num_batches).item()
    avg_future_heatmap_loss = (totals[15] / reduced_num_batches).item()
    avg_preserve_loss = (totals[16] / reduced_num_batches).item()
    avg_delta_z_l2 = (totals[17] / reduced_num_batches).item()
    val_heatmap_component_metrics = {
        f"val_hm_{key}": (
            total_heatmap_components[index] / reduced_num_batches
        ).item()
        for index, key in enumerate(_HEATMAP_COMPONENT_KEYS)
    }

    r_tp, r_tn, r_fp, r_fn = totals[6].item(), totals[7].item(), totals[8].item(), totals[9].item()
    vis_total = r_tp + r_tn + r_fp + r_fn
    val_vis_metrics = {}
    if vis_total > 0:
        val_vis_metrics['val_vis_accuracy'] = (r_tp + r_tn) / vis_total
        val_vis_metrics['val_vis_precision'] = r_tp / max(r_tp + r_fp, 1)
        val_vis_metrics['val_vis_recall'] = r_tp / max(r_tp + r_fn, 1)
        val_vis_metrics['val_vis_tnr'] = r_tn / max(r_tn + r_fp, 1)
        p, r = val_vis_metrics['val_vis_precision'], val_vis_metrics['val_vis_recall']
        val_vis_metrics['val_vis_f1'] = 2 * p * r / max(p + r, 1e-8)
        val_vis_metrics['val_vis_gt_pos_ratio'] = (r_tp + r_fn) / vis_total
        logger.info(
            f"  📊 Visibility gate: acc={val_vis_metrics['val_vis_accuracy']:.3f} "
            f"prec={val_vis_metrics['val_vis_precision']:.3f} "
            f"recall={val_vis_metrics['val_vis_recall']:.3f} "
            f"TNR={val_vis_metrics['val_vis_tnr']:.3f} "
            f"F1={val_vis_metrics['val_vis_f1']:.3f} "
            f"(gt_pos={val_vis_metrics['val_vis_gt_pos_ratio']:.2f})"
        )

    logger.info(
        f"  [HM] peak={avg_peak_loss:.4f} "
        f"vis={avg_vis_loss:.4f} "
        f"coord={avg_coord_loss:.4f} "
        f"neg={val_heatmap_component_metrics['val_hm_neg_loss']:.4f} "
        f"view_macro={val_heatmap_component_metrics['val_hm_view_macro_loss']:.4f} "
        f"direction_macro={val_heatmap_component_metrics['val_hm_direction_macro_loss']:.4f} "
        f"panoramic_view={val_heatmap_component_metrics['val_hm_panoramic_view_loss']:.4f}"
    )
    if reduced_num_heatmap_mse_batches > 0:
        logger.info(f"  📊 Heatmap 推理 MSE (采样 {reduced_num_heatmap_mse_batches} batches): {avg_hm_mse:.6f}")
    if val_heatmap_joint_metrics["val_heatmap_valid_count"] > 0:
        logger.info(
            "  [HM joint] PCK@4=%.4f PCK@8=%.4f macro-PCK@8=%.4f "
            "pixel median=%.2f P90=%.2f view5_acc=%.4f visible=%d none=%d",
            val_heatmap_joint_metrics["val_heatmap_joint_pck4"],
            val_heatmap_joint_metrics["val_heatmap_joint_pck8"],
            val_heatmap_joint_metrics["val_heatmap_macro_joint_pck8"],
            val_heatmap_joint_metrics["val_heatmap_pixel_error_median"],
            val_heatmap_joint_metrics["val_heatmap_pixel_error_p90"],
            val_heatmap_joint_metrics["val_heatmap_view5_accuracy"],
            int(val_heatmap_joint_metrics["val_heatmap_visible_count"]),
            int(val_heatmap_joint_metrics["val_heatmap_none_count"]),
        )
        logger.info(
            "  [HM joint/view] %s",
            " ".join(
                f"{name}:PCK@8="
                f"{val_heatmap_joint_metrics[f'val_heatmap_{name}_pck8']:.4f}"
                f"(n={int(val_heatmap_joint_metrics[f'val_heatmap_{name}_count'])})"
                for name in _HEATMAP_VIEW_NAMES
            ),
        )

    result = {
        'val_loss': avg_loss,
        'val_heatmap_loss': avg_hm,
        'val_trajectory_loss': avg_act,
        'val_lm_loss': avg_lm_loss,
        'val_stop_loss': avg_stop_loss,
        'val_future_heatmap_loss': avg_future_heatmap_loss,
        'val_preserve_loss': avg_preserve_loss,
        'val_delta_z_l2': avg_delta_z_l2,
        'val_heatmap_mse': avg_hm_mse,
        'val_total_loss': avg_loss,
        'val_hm_peak_loss': avg_peak_loss,
        'val_hm_vis_loss': avg_vis_loss,
        'val_hm_coord_loss': avg_coord_loss,
    }
    result.update(val_heatmap_component_metrics)
    result.update(val_heatmap_joint_metrics)
    if train_future:
        future_metrics = future_tube_metrics_from_statistics(future_tube_stats)
        result.update(
            {
                'val_future_tube_soft_iou': future_metrics.soft_iou,
                'val_future_topk_support_recall': future_metrics.topk_support_recall,
                'val_future_visibility_f1': future_metrics.visibility_f1,
                'val_future_valid_time_bins': future_metrics.valid_time_bins,
                'val_future_supported_view_bins': future_metrics.supported_view_bins,
            }
        )
        for view_index, view_name in enumerate(_HEATMAP_VIEW_NAMES):
            view_iou = future_metrics.per_view_soft_iou[view_index]
            result[f'val_future_{view_name}_soft_iou'] = (
                0.0 if view_iou is None else view_iou
            )
            result[f'val_future_{view_name}_support'] = (
                future_metrics.per_view_support[view_index]
            )
        logger.info(
            "  [Future tube] IoU=%.4f top-k-support=%.4f vis-F1=%.4f "
            "valid_time=%d supported_views=%d",
            future_metrics.soft_iou,
            future_metrics.topk_support_recall,
            future_metrics.visibility_f1,
            future_metrics.valid_time_bins,
            future_metrics.supported_view_bins,
        )
    if avg_act > 0:
        logger.info(f"  📊 Trajectory loss: {avg_act:.6f}")
    if train_lm:
        logger.info(f"  📊 LM loss: {avg_lm_loss:.6f}")
    if train_stop_head:
        calibrated = _select_stop_hysteresis_thresholds(
            stop_thresholds,
            stop_confusion_grid,
            max_add_false_positive_rate=float(
                validation_cfg.get('stop_add_max_false_positive_rate', 0.01)
            ),
            min_veto_recall=float(
                validation_cfg.get('stop_veto_min_recall', 0.98)
            ),
            min_add_threshold=float(
                validation_cfg.get('stop_add_min_threshold', 0.9)
            ),
            max_veto_threshold=float(
                validation_cfg.get('stop_veto_max_threshold', 0.5)
            ),
        )
        result.update({f"val_stop_{key}": value for key, value in calibrated.items()})
        logger.info(
            "  STOP calibrated: add>=%.3f (FPR=%.4f recall=%.4f), "
            "veto accept>=%.3f (recall=%.4f FPR=%.4f), positives=%d negatives=%d",
            calibrated['add_stop_threshold'],
            calibrated['add_false_positive_rate'],
            calibrated['add_recall'],
            calibrated['veto_stop_threshold'],
            calibrated['veto_recall'],
            calibrated['veto_false_positive_rate'],
            int(calibrated['positive_count']),
            int(calibrated['negative_count']),
        )
    result.update(val_vis_metrics)
    return result
