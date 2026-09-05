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

from src.data.future_trajectory_batch import assert_no_future_teacher_inputs
from src.models.future_trajectory_objective import (
    future_tube_metrics_from_statistics,
    future_tube_sufficient_statistics,
)
from src.models.past_plan_action import compute_shared_plan_action_losses
from src.models.past_plan_action_loss import (
    PastPlanActionLossWeights,
    compose_past_plan_action_loss,
)
from src.models.pipeline import VLNPipeline
from src.utils.gpu_heatmap import GPUHeatmapComputer

from .distributed import DistributedContext, _dist_all_reduce_in_place
from .pose_adaptation import assert_required_history_pose_provider
from .train_loop import _prepare_trajectory_sequence_inputs
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

logger = logging.getLogger(__name__)


def _accumulate_ppa_rollout_stats(
    *,
    action_head,
    plan_z: torch.Tensor,
    plan_z0: torch.Tensor,
    gt_trajectory: torch.Tensor,
    trajectory_valid: torch.Tensor | None,
    traj_images: torch.Tensor | None,
    postprocess_config,
    batch_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample bridged vs native banks under shared noise and score them.

    Returns ``[pairs, endpoint_err, endpoint_err_native, endpoint_gap,
    agreement]`` sums for one batch.  Noise is seeded by the batch index only,
    so the metric is comparable across epochs and the native/bridged rollouts
    differ exclusively through the Plan delta.
    """

    from src.models.action.rollout_metrics import compute_rollout_pair_metrics

    stats = torch.zeros(5, device=device, dtype=torch.float64)
    num_samples = int(action_head.config.num_sample_trajs)
    predict_steps = int(action_head.config.predict_steps)
    action_dim = int(action_head.config.action_dim)
    for b in range(plan_z.shape[0]):
        if trajectory_valid is not None and float(trajectory_valid[b]) <= 0.0:
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(20260826 + 1_000_003 * int(batch_index) + b)
        noise = torch.randn(
            (num_samples, predict_steps, action_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        images = None if traj_images is None else traj_images[b : b + 1]
        bank_bridged = action_head.get_trajectory_from_projected(
            plan_z[b : b + 1],
            traj_images=images,
            initial_noise=noise.clone(),
        )
        bank_native = action_head.get_trajectory_from_projected(
            plan_z0[b : b + 1],
            traj_images=images,
            initial_noise=noise.clone(),
        )
        pair_metrics = compute_rollout_pair_metrics(
            bank_bridged=bank_bridged,
            bank_native=bank_native,
            gt_trajectory=gt_trajectory[b],
            config=postprocess_config,
        )
        stats += torch.tensor(
            [
                1.0,
                pair_metrics["endpoint_error"],
                pair_metrics["endpoint_error_native"],
                pair_metrics["endpoint_gap_to_native"],
                pair_metrics["action_agreement"],
            ],
            device=device,
            dtype=torch.float64,
        )
    return stats


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
    """DDP-mergeable panoramic view+localization validation metrics."""

    _VALID = 0
    _VISIBLE = 1
    _NONE = 2
    _VIEW5_CORRECT = 3
    _JOINT_PCK4_CORRECT = 4
    _JOINT_PCK8_CORRECT = 5
    _PER_VIEW_PCK4_START = 6
    _PER_VIEW_PCK8_START = 10
    _PER_VIEW_COUNT_START = 14
    _NUM_COUNTS = 18

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
        # Integer squared-error bins make median/P90 exactly SUM-reducible.
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
            raise ValueError("Prediction and target history counts differ")
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

        gt_peak_per_view = gt_hm.amax(dim=(-2, -1))
        eligible_view = (gt_vis > 0.5) & (gt_peak_per_view > 0)
        visible = eligible_view.any(dim=-1)
        gt_view = gt_peak_per_view.masked_fill(
            ~eligible_view,
            -torch.inf,
        ).argmax(dim=-1)
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
        pred_maps = pred_hm[visible_rows, target_views]
        gt_maps = gt_hm[visible_rows, target_views]
        width = int(gt_hm.shape[-1])
        pred_peak = pred_maps.reshape(visible_rows.numel(), -1).argmax(dim=-1)
        gt_peak = gt_maps.reshape(visible_rows.numel(), -1).argmax(dim=-1)
        pred_y = torch.div(pred_peak, width, rounding_mode="floor")
        pred_x = torch.remainder(pred_peak, width)
        gt_y = torch.div(gt_peak, width, rounding_mode="floor")
        gt_x = torch.remainder(gt_peak, width)
        squared_error = (pred_x - gt_x).square() + (pred_y - gt_y).square()
        if (
            squared_error.numel()
            and int(squared_error.max().item()) > self.max_squared_error
        ):
            raise ValueError(
                "Observed pixel error exceeds configured heatmap histogram: "
                f"configured_size={self.heatmap_size}, "
                f"target_size={tuple(gt_hm.shape[-2:])}"
            )
        self.pixel_error_histogram += torch.bincount(
            squared_error,
            minlength=self.pixel_error_histogram.numel(),
        ).to(dtype=self.pixel_error_histogram.dtype)

        correct_view = view5_correct[visible_rows]
        joint4 = correct_view & (squared_error <= 4 ** 2)
        joint8 = correct_view & (squared_error <= 8 ** 2)
        self.counts[self._JOINT_PCK4_CORRECT] += joint4.sum()
        self.counts[self._JOINT_PCK8_CORRECT] += joint8.sum()
        for view in range(4):
            view_rows = target_views == view
            self.counts[self._PER_VIEW_COUNT_START + view] += view_rows.sum()
            self.counts[self._PER_VIEW_PCK4_START + view] += (
                joint4 & view_rows
            ).sum()
            self.counts[self._PER_VIEW_PCK8_START + view] += (
                joint8 & view_rows
            ).sum()

    def all_reduce(self) -> None:
        _dist_all_reduce_in_place(self.counts)
        _dist_all_reduce_in_place(self.pixel_error_histogram)

    @staticmethod
    def _histogram_quantile(histogram: torch.Tensor, quantile: float) -> float:
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
        supported_direction_pck4 = []
        supported_direction_pck8 = []
        for view, name in enumerate(_HEATMAP_VIEW_NAMES):
            count = self.counts[self._PER_VIEW_COUNT_START + view].item()
            correct4 = self.counts[self._PER_VIEW_PCK4_START + view].item()
            correct = self.counts[self._PER_VIEW_PCK8_START + view].item()
            per_view_pck4 = correct4 / count if count > 0 else 0.0
            per_view_pck8 = correct / count if count > 0 else 0.0
            metrics[f"val_heatmap_{name}_pck4"] = per_view_pck4
            metrics[f"val_heatmap_{name}_pck8"] = per_view_pck8
            metrics[f"val_heatmap_{name}_count"] = count
            if count > 0:
                supported_direction_pck4.append(per_view_pck4)
                supported_direction_pck8.append(per_view_pck8)
        metrics["val_heatmap_macro_joint_pck4"] = (
            sum(supported_direction_pck4) / len(supported_direction_pck4)
            if supported_direction_pck4
            else 0.0
        )
        metrics["val_heatmap_macro_joint_pck8"] = (
            sum(supported_direction_pck8) / len(supported_direction_pck8)
            if supported_direction_pck8
            else 0.0
        )
        metrics["val_heatmap_supported_direction_count"] = float(
            len(supported_direction_pck8)
        )
        return metrics


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
    total_heatmap_components = torch.zeros(
        len(_HEATMAP_COMPONENT_KEYS),
        device=dist_context.device,
        dtype=torch.float64,
    )

    loss_cfg = cfg['loss']
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
    past_plan_action_stage = stage_cfg.get('past_plan_action_stage')
    train_past_plan_action = past_plan_action_stage is not None
    need_heatmap_targets = train_history
    if train_past_plan_action and train_lm:
        raise ValueError(
            "Past→Plan→Action v1 cannot be combined with LM validation"
        )
    heatmap_control_cfg = (
        cfg.get('model', {})
        .get('action_head', {})
        .get('nextdit', {})
        .get('heatmap_control', {})
    )
    system2_memory_enabled = bool(
        cfg.get('model', {}).get('system2_memory', {}).get('enabled', False)
    )
    heatmap_control_enabled = bool(heatmap_control_cfg.get('enabled', False))
    if train_past_plan_action and heatmap_control_enabled:
        raise ValueError(
            "Past→Plan→Action v1 requires legacy heatmap_control disabled"
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

    hm_loss_fn = build_heatmap_loss_fn(
        cfg,
        device,
        temperature=heatmap_temperature,
    )
    future_hm_loss_fn = (
        build_future_heatmap_loss_fn(cfg, device) if train_future else None
    )
    future_tube_stats = torch.zeros(17, device=device, dtype=torch.float64)
    ppa_weights = PastPlanActionLossWeights(
        action=float(loss_cfg.get('trajectory_weight', 1.0)),
        history=float(loss_cfg.get('history_weight', 0.3)),
        future=float(loss_cfg.get('future_weight', 0.3)),
        preserve=float(loss_cfg.get('preserve_weight', 0.5)),
        delta_z=float(loss_cfg.get('delta_z_weight', 0.01)),
    )
    ppa_delta_relative = bool(loss_cfg.get('delta_z_relative', False))
    ppa_advantage_reference = (
        float(loss_cfg.get('action_advantage_reference_mse', 0.125))
        if bool(loss_cfg.get('action_advantage_enabled', False))
        else None
    )
    ppa_advantage_max_weight = float(
        loss_cfg.get('action_advantage_max_weight', 4.0)
    )

    val_inference_batches = cfg.get('validation', {}).get('val_inference_batches', 10)
    val_rollout_batches = int(
        cfg.get('validation', {}).get('val_rollout_batches', 0)
    )
    rollout_postprocess_config = None
    if train_past_plan_action and val_rollout_batches > 0:
        from src.models.action.treatment_spec import TrajectoryPostprocessConfig

        action_head_for_rollout = getattr(
            model_module, 'nextdit_action_head', None
        )
        if action_head_for_rollout is None:
            raise RuntimeError(
                "val_rollout_batches > 0 requires the NextDiT action head"
            )
        # Deployment-default post-processing, matching the certified
        # closed-loop evaluation (selection=mean, x_sign=1, no heading fix).
        rollout_postprocess_config = TrajectoryPostprocessConfig(
            num_sample_trajs=int(
                action_head_for_rollout.config.num_sample_trajs
            ),
            action_scale=float(
                cfg.get('data', {})
                .get('trajectory', {})
                .get('action_scale', 4.0)
            ),
        )
    # [count, endpoint_err_z, endpoint_err_z0, endpoint_gap, agreement]
    rollout_stats = torch.zeros(5, device=device, dtype=torch.float64)

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
            assert_required_history_pose_provider(batch, stage_cfg)
            if train_future:
                assert_no_future_teacher_inputs(batch)
            single_view_heatmap_batch = "pixel_values" in batch
            if single_view_heatmap_batch:
                required_single_view_keys = {
                    "image_grid_thw",
                    "num_histories",
                    "history_rel_poses",
                }
                missing_single_view = sorted(required_single_view_keys - set(batch))
                if missing_single_view:
                    raise RuntimeError(
                        "internnav_single_view validation batch is incomplete: "
                        f"missing={missing_single_view}"
                    )
                if train_action or train_lm:
                    raise RuntimeError(
                        "worker-preprocessed internnav_single_view validation "
                        "batches are heatmap-only"
                    )
                history_frames = None
                current_frame = None
            else:
                history_frames = batch['history_frames']
                current_frame = batch['current_frame']

            gt_action = batch['action'].to(device, non_blocking=True)
            action_valid = batch['action_valid'].to(device, non_blocking=True)
            is_stop = batch['is_stop'].to(device, non_blocking=True)
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
                if single_view_heatmap_batch:
                    single_view_inputs_batch = {
                        "pixel_values": batch["pixel_values"],
                        "image_grid_thw": batch["image_grid_thw"],
                    }
                    single_view_num_histories = batch["num_histories"]
                else:
                    single_view_inputs_batch = batch.get('single_view_inputs')
                    single_view_num_histories = batch.get('single_view_num_histories')
                history_rel_poses = batch.get('history_rel_poses')
                heatmap_single_view_inputs = batch.get('heatmap_single_view_inputs')
                heatmap_single_view_num_histories = batch.get(
                    'heatmap_single_view_num_histories'
                )
                heatmap_control_history_mask = batch.get('heatmap_control_history_mask')
                history_valid_mask = batch.get('history_valid_mask')
                history_age_steps = batch.get('history_age_steps')
                traj_images_batch = batch.get('traj_images')
                if traj_images_batch is not None:
                    traj_images_batch = traj_images_batch.to(device, non_blocking=True)
                if history_rel_poses is not None:
                    history_rel_poses = history_rel_poses.to(device, non_blocking=True)
                if history_valid_mask is not None:
                    history_valid_mask = history_valid_mask.to(
                        device,
                        non_blocking=True,
                    )
                if single_view_heatmap_batch:
                    video_frames = None
                elif panoramic_inputs_batch is not None:
                    video_frames = current_frame.unsqueeze(1)
                else:
                    video_frames = torch.cat([
                        history_frames,
                        history_frames[:, -1:],
                    ], dim=1)

                output = model(
                    video_frames=video_frames,
                    instruction_text=instruction_text,
                    current_observation=current_frame,
                    current_views=current_views_batch,
                    history_panoramas=history_panoramas_batch,
                    panoramic_inputs=panoramic_inputs_batch,
                    panoramic_num_histories=panoramic_num_histories,
                    panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                    single_view_inputs=single_view_inputs_batch,
                    single_view_num_histories=single_view_num_histories,
                    heatmap_single_view_inputs=heatmap_single_view_inputs,
                    heatmap_single_view_num_histories=heatmap_single_view_num_histories,
                    heatmap_control_history_mask=heatmap_control_history_mask,
                    history_valid_mask=history_valid_mask,
                    history_age_steps=history_age_steps,
                    history_rel_poses=history_rel_poses,
                    traj_images=traj_images_batch,
                    sample_trajectory=False,
                    return_heatmaps=train_history or heatmap_control_enabled,
                    return_heatmap_logits=train_history or heatmap_control_enabled,
                    return_future_heatmaps=train_future,
                    return_actions=train_action,
                    return_lm_loss=train_lm,
                    inject_system2_memory=system2_memory_enabled,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
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
                    output['visibility'].float(),
                    output['heatmaps'].float(),
                    gt_vis=gt_vis.float(),
                    gt_heatmaps=gt_heatmap.to(device, dtype=torch.float32),
                    history_mask=hm_history_mask,
                    pred_heatmap_logits=output['heatmap_logits'].float(),
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
                    pred_heatmaps=output['heatmaps'],
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
                        f"batch_missing={missing_batch} "
                        f"output_missing={missing_output}"
                    )
                assert future_hm_loss_fn is not None
                gt_future_visibility = batch[
                    'future_trajectory_visibility'
                ].to(device, non_blocking=True)
                gt_future_heatmap = batch['future_trajectory_heatmap'].to(
                    device,
                    non_blocking=True,
                )
                future_time_mask = batch['future_trajectory_time_mask'].to(
                    device,
                    non_blocking=True,
                )
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
                    pv_full = torch.sigmoid(pred_vis_logits.float()) > 0.5
                    gv_full = gt_vis_batch.to(pred_vis_logits.device) > 0.5
                    history_mask = batch.get('history_mask')
                    if history_mask is not None:
                        valid = history_mask.to(
                            device=pred_vis_logits.device,
                            dtype=torch.bool,
                        )
                        while valid.ndim < pv_full.ndim:
                            valid = valid.unsqueeze(-1)
                        valid = valid.expand_as(pv_full)
                        pv = pv_full[valid].float()
                        gv = gv_full[valid].float()
                    else:
                        pv = pv_full.reshape(-1).float()
                        gv = gv_full.reshape(-1).float()
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
                    traj_images = traj_images_batch
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
                        missing_plan = [
                            key
                            for key in ('plan_z0', 'plan_z', 'delta_z')
                            if key not in output
                        ]
                        if missing_plan:
                            raise RuntimeError(
                                "Past→Plan→Action validation output is incomplete: "
                                f"missing={missing_plan}"
                            )
                        action_plan_losses = compute_shared_plan_action_losses(
                            action_head=model_module.nextdit_action_head,
                            plan_z0=output['plan_z0'],
                            plan_z=output['plan_z'],
                            gt_trajectory=gt_trajectory,
                            trajectory_valid=trajectory_valid,
                            traj_images=traj_images,
                            preserve_weight=ppa_weights.preserve,
                            delta_weight=ppa_weights.delta_z,
                            delta_relative=ppa_delta_relative,
                            advantage_reference_mse=ppa_advantage_reference,
                            advantage_max_weight=ppa_advantage_max_weight,
                        )
                        trajectory_loss = action_plan_losses['action']
                        if (
                            rollout_postprocess_config is not None
                            and num_batches < val_rollout_batches
                        ):
                            rollout_stats += _accumulate_ppa_rollout_stats(
                                action_head=model_module.nextdit_action_head,
                                plan_z=output['plan_z'],
                                plan_z0=output['plan_z0'],
                                gt_trajectory=gt_trajectory,
                                trajectory_valid=trajectory_valid,
                                traj_images=traj_images,
                                postprocess_config=rollout_postprocess_config,
                                batch_index=i,
                                device=device,
                            )
                    else:
                        traj_hidden_states = model_module.adapt_traj_hidden_states(
                            output['traj_hidden_states']
                        )
                        heatmap_tokens = None
                        heatmap_mask = None
                        heatmap_valid = None
                        if heatmap_control_enabled:
                            required_control = {
                                'heatmap_control_tokens',
                                'heatmap_control_mask',
                                'heatmap_control_valid',
                            }
                            missing_control = sorted(required_control - set(output))
                            if missing_control:
                                raise RuntimeError(
                                    "heatmap control validation output is incomplete: "
                                    f"missing={missing_control}"
                                )
                            heatmap_tokens = output['heatmap_control_tokens']
                            heatmap_mask = output['heatmap_control_mask']
                            heatmap_valid = output['heatmap_control_valid']
                        traj_result = model_module.nextdit_action_head.compute_loss(
                            traj_hidden_states,
                            gt_trajectory,
                            traj_images=traj_images,
                            heatmap_tokens=heatmap_tokens,
                            heatmap_mask=heatmap_mask,
                            heatmap_valid=heatmap_valid,
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

            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 0.0)
            lm_weight = loss_cfg.get('lm_weight', stage_cfg.get('lm_weight', 1.0))

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
                )

            total_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_action_loss += trajectory_loss.item()
            total_lm_loss += lm_loss.item()
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
                        and not single_view_heatmap_batch
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
            total_future_heatmap_loss,
            total_preserve_loss,
            total_delta_z_loss,
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)
    _dist_all_reduce_in_place(total_heatmap_components)
    _dist_all_reduce_in_place(future_tube_stats)
    _dist_all_reduce_in_place(rollout_stats)
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
    avg_future_heatmap_loss = (totals[14] / reduced_num_batches).item()
    avg_preserve_loss = (totals[15] / reduced_num_batches).item()
    avg_delta_z_l2 = (totals[16] / reduced_num_batches).item()
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
        # Explicit name makes it clear this is globally reduced TP/FP/FN
        # (micro averaging), after history_mask filtering above.
        val_vis_metrics['val_heatmap_visibility_micro_f1'] = (
            val_vis_metrics['val_vis_f1']
        )
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
                f"{name}:PCK@4="
                f"{val_heatmap_joint_metrics[f'val_heatmap_{name}_pck4']:.4f},"
                "PCK@8="
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
        'val_future_heatmap_loss': avg_future_heatmap_loss,
        'val_preserve_loss': avg_preserve_loss,
        'val_delta_z_l2': avg_delta_z_l2,
        'val_heatmap_mse': avg_hm_mse,
        'val_total_loss': avg_loss,
        'val_hm_peak_loss': avg_peak_loss,
        'val_hm_vis_loss': avg_vis_loss,
        'val_hm_coord_loss': avg_coord_loss,
    }
    if rollout_postprocess_config is not None:
        rollout_pairs = rollout_stats[0].item()
        if rollout_pairs <= 0:
            raise RuntimeError(
                "val_rollout_batches > 0 but no valid PPA rollout pair was "
                "evaluated; sampled-rollout checkpoint selection would be blind"
            )
        result.update(
            {
                'val_rollout_pairs': rollout_pairs,
                'val_rollout_endpoint_error': (
                    rollout_stats[1] / rollout_pairs
                ).item(),
                'val_rollout_endpoint_error_native': (
                    rollout_stats[2] / rollout_pairs
                ).item(),
                'val_rollout_endpoint_gap': (
                    rollout_stats[3] / rollout_pairs
                ).item(),
                'val_rollout_action_agreement': (
                    rollout_stats[4] / rollout_pairs
                ).item(),
            }
        )
        logger.info(
            "  🎯 PPA rollout (%d pairs, shared noise): endpoint bridged=%.3fm "
            "native=%.3fm gap=%.3fm action_agreement=%.3f",
            int(rollout_pairs),
            result['val_rollout_endpoint_error'],
            result['val_rollout_endpoint_error_native'],
            result['val_rollout_endpoint_gap'],
            result['val_rollout_action_agreement'],
        )
    result.update(val_heatmap_component_metrics)
    result.update(val_heatmap_joint_metrics)
    if train_future:
        future_metrics = future_tube_metrics_from_statistics(future_tube_stats)
        result.update(
            {
                'val_future_soft_iou': future_metrics.soft_iou,
                # Keep the explicit tube alias for manifests produced by the
                # initial Past→Plan→Action implementation.
                'val_future_tube_soft_iou': future_metrics.soft_iou,
                'val_future_topk_support_recall': (
                    future_metrics.topk_support_recall
                ),
                'val_future_visibility_f1': future_metrics.visibility_f1,
                'val_future_valid_time_bins': future_metrics.valid_time_bins,
                'val_future_supported_view_bins': (
                    future_metrics.supported_view_bins
                ),
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
            "  [Future tube] soft-IoU=%.4f top-k-support=%.4f "
            "vis-F1=%.4f valid_time=%d supported_views=%d | %s",
            future_metrics.soft_iou,
            future_metrics.topk_support_recall,
            future_metrics.visibility_f1,
            future_metrics.valid_time_bins,
            future_metrics.supported_view_bins,
            " ".join(
                f"{view_name}:IoU="
                f"{result[f'val_future_{view_name}_soft_iou']:.4f}"
                f"(n={result[f'val_future_{view_name}_support']})"
                for view_name in _HEATMAP_VIEW_NAMES
            ),
        )
    if avg_act > 0:
        logger.info(f"  📊 Trajectory loss: {avg_act:.6f}")
    if train_lm:
        logger.info(f"  📊 LM loss: {avg_lm_loss:.6f}")
    result.update(val_vis_metrics)
    return result
