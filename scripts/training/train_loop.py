"""
Core training loop: ``train_one_epoch``.
"""

from __future__ import annotations

import gc
import logging
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pipeline import VLNPipeline
from src.data.future_trajectory_batch import assert_no_future_teacher_inputs
from src.models.past_plan_action import compute_shared_plan_action_losses
from src.models.past_plan_action_loss import (
    PastPlanActionLossWeights,
    compose_past_plan_action_loss,
)
from src.utils.gpu_heatmap import GPUHeatmapComputer

from .distributed import (
    DistributedContext,
    _dist_all_reduce_in_place,
    _get_supported_trainable_sync_modules,
    synchronize_trainable_module_gradients,
)
from .ema import EMAModel
from .manifest import _append_jsonl
from .memory import _CG_LIMIT_GB, _cgroup_mem_usage_gb, _drop_page_cache, _malloc_trim
from .model_builder import end_nextdit_warmup
from .optimizer import get_heatmap_temperature
from .pose_adaptation import assert_required_history_pose_provider
from .utils import (
    _format_decode_internal_timing,
    _format_qwen_internal_timing,
    _get_trainable_params,
    _mean_timing,
    _unwrap_model,
    build_future_heatmap_loss_fn,
    build_heatmap_loss_fn,
    compute_l2_sp_loss,
    make_autocast_context,
)
from .visualization import (
    _select_primary_heatmap_slice,
    _should_use_gpu_gt,
    visualize_heatmap_predictions,
)

logger = logging.getLogger(__name__)

_TRAJECTORY_VIEW_ORDER = ("front", "right", "back", "left")
_HEATMAP_COMPONENT_KEYS = (
    "peak_loss",
    "vis_loss",
    "coord_loss",
    "neg_loss",
    "view_macro_loss",
    "direction_macro_loss",
    "panoramic_view_loss",
)


def _prepare_trajectory_sequence_inputs(
    gt_trajectory: torch.Tensor,
    trajectory_valid: torch.Tensor | None,
    traj_images: torch.Tensor | None,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Select the System1 supervision layout used by the current stage.

    Closed-loop evaluation predicts one trajectory from the goal-freeze
    lookdown frame, with that frame in both the anchor and current slots.
    ``first_only`` reproduces that layout and avoids repeating one initial
    pano-goal latent against later egocentric trajectory targets.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "all":
        return gt_trajectory, trajectory_valid, traj_images
    if normalized_mode != "first_only":
        raise ValueError(
            "training stage trajectory_sequence_mode must be all or first_only, "
            f"got {mode!r}"
        )

    if gt_trajectory.ndim == 3:
        return gt_trajectory, trajectory_valid, traj_images
    if gt_trajectory.ndim != 4:
        raise RuntimeError(
            "first_only trajectory supervision expects [B,N,T,D], got "
            f"{tuple(gt_trajectory.shape)}"
        )
    if traj_images is None or traj_images.ndim != 5:
        raise RuntimeError(
            "first_only trajectory supervision requires traj_images [B,N,H,W,C]"
        )
    if gt_trajectory.shape[:2] != traj_images.shape[:2]:
        raise RuntimeError(
            "trajectory/traj_images sequence shape mismatch: "
            f"trajectory={tuple(gt_trajectory.shape)} images={tuple(traj_images.shape)}"
        )

    first_image = traj_images[:, 0]
    eval_matched_pair = torch.stack([first_image, first_image], dim=1)
    first_valid = trajectory_valid
    if trajectory_valid is not None:
        if trajectory_valid.ndim != 2:
            raise RuntimeError(
                "first_only trajectory_valid must be [B,N], got "
                f"{tuple(trajectory_valid.shape)}"
            )
        first_valid = trajectory_valid[:, 0]
    return gt_trajectory[:, 0], first_valid, eval_matched_pair


def _trajectory_view_sample_weights(
    view_ids,
    cfg: dict,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Build finite positive per-sample weights for panoramic goal views."""
    if not bool(cfg.get("enabled", False)):
        return None
    if view_ids is None:
        raise RuntimeError(
            "trajectory_view_weights is enabled but the batch has no pano_view_id"
        )

    configured = cfg.get("weights") or {}
    allowed = set(_TRAJECTORY_VIEW_ORDER)
    unknown = sorted(set(configured) - allowed)
    if unknown:
        raise ValueError(f"Unknown trajectory view weights: {unknown}")

    values: list[float] = []
    for raw_view in view_ids:
        view = str(raw_view).lower()
        if view not in allowed:
            raise RuntimeError(
                "trajectory_view_weights requires pixel-goal views, got "
                f"{raw_view!r}"
            )
        value = float(configured.get(view, 1.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"trajectory view weight for {view!r} must be finite and > 0, got {value}"
            )
        values.append(value)
    return torch.tensor(values, device=device, dtype=torch.float32)


def _apply_bridge_only_train_mode(model_module: VLNPipeline, stage_cfg: dict, logger) -> None:
    """Keep frozen Stage2 components in eval mode while selected bridge params train.

    ``model.train()`` recursively switches every child module to training mode.
    Stage2 variants often train only a subset of the action condition stack
    while Qwen/LoRA and most of System1 stay frozen.  Leaving frozen modules in
    train mode can enable dropout/stochastic-depth noise even though their
    weights are frozen.
    """
    from .pose_adaptation import is_pose_adaptation_stage

    trainable = set(stage_cfg.get("trainable_modules", []))
    pose_adaptation = is_pose_adaptation_stage(stage_cfg)
    ppa_chain = getattr(model_module, "past_plan_action", None)
    ppa_enabled = ppa_chain is not None
    selective_action_modules = {
        "latent_queries",
        "cond_projector",
        "memory_encoder",
        "rgb_resampler",
        "traj_dit",
        "action_encoder",
        "action_decoder",
        "pano_latent_adapter",
        "heatmap_tokenizer",
        "heatmap_control",
    }
    is_selective_stage2 = (
        stage_cfg.get("bridge_only", False)
        or (
            stage_cfg.get("train_action", False)
            and bool(trainable & selective_action_modules)
            and "nextdit_action_head" not in trainable
        )
    )
    is_heatmap_only = (
        trainable == {"heatmap_vln"}
        and not stage_cfg.get("train_action", False)
        and not stage_cfg.get(
            "train_lm",
            stage_cfg.get("train_system2_sft", False),
        )
    )
    is_heatmap_control = (
        trainable == {"heatmap_tokenizer", "heatmap_control"}
        and stage_cfg.get("train_action", False)
    )
    is_selective_stage2 = (
        is_selective_stage2 or is_heatmap_only or is_heatmap_control or ppa_enabled
    )
    if not is_selective_stage2:
        return

    if getattr(model_module, "heatmap_vln", None) is not None:
        if "heatmap_vln" in trainable:
            if ppa_enabled:
                # The Past bottleneck/shared decoder is selectively trained;
                # frozen visual conditioners must stay deterministic.
                model_module.heatmap_vln.eval()
                coarse = model_module.heatmap_vln.coarse
                for module_name in (
                    "proj_history",
                    "proj_traj",
                    "self_attn",
                    "vis_head",
                    "heatmap_head",
                ):
                    getattr(coarse, module_name).train()
                model_module.heatmap_vln.fine.train()
            elif pose_adaptation:
                # Recursive head.train() would reactivate dropout/stochastic
                # behavior in frozen DPT/Fine branches.  Only the four
                # whitelisted coarse modules are allowed training behavior.
                model_module.heatmap_vln.eval()
                coarse = model_module.heatmap_vln.coarse
                for module_name in (
                    "proj_traj",
                    "self_attn",
                    "vis_head",
                    "heatmap_head",
                ):
                    getattr(coarse, module_name).train()
                logger.info(
                    "  ✓ AMB3R pose-adaptation mode: frozen Head branches eval; "
                    "four whitelisted coarse modules train"
                )
            else:
                model_module.heatmap_vln.train()
        else:
            model_module.heatmap_vln.eval()

    # HeatmapVLN keeps the same Qwen module as ``vlm_backbone.model`` under a
    # second parent. Apply the frozen-backbone mode *after* head.train(), or
    # recursive train() would silently switch the shared Qwen back to train.
    vlm_backbone = getattr(model_module, "vlm_backbone", getattr(model_module, "qwen2_5_vl", None))
    if vlm_backbone is not None and not ({"lora", "vlm_lora"} & trainable):
        if is_heatmap_only or is_heatmap_control or ppa_enabled:
            # Heatmap-only training consumes deterministic frozen Qwen features
            # while keeping only the localization head in train mode.
            vlm_backbone.eval()
        else:
            # Keep the VLM itself in train mode so Transformers gradient
            # checkpointing remains active, but disable frozen dropout noise.
            for submodule in vlm_backbone.modules():
                if isinstance(submodule, torch.nn.Dropout):
                    submodule.eval()

    if "llm_projector" not in trainable and getattr(model_module, "llm_projector", None) is not None:
        model_module.llm_projector.eval()

    if "pano_latent_adapter" in trainable and getattr(model_module, "pano_latent_adapter", None) is not None:
        model_module.pano_latent_adapter.train()
    elif getattr(model_module, "pano_latent_adapter", None) is not None:
        model_module.pano_latent_adapter.eval()

    tokenizer = getattr(model_module, "heatmap_tokenizer", None)
    if tokenizer is not None:
        tokenizer.train("heatmap_tokenizer" in trainable)

    nah = getattr(model_module, "nextdit_action_head", None)
    if nah is not None:
        nah.eval()
        submodules = {
            "cond_projector": "cond_projector",
            "memory_encoder": "memory_encoder",
            "rgb_resampler": "rgb_resampler",
            "traj_dit": "traj_dit",
            "action_encoder": "action_encoder",
            "action_decoder": "action_decoder",
        }
        for cfg_name, attr_name in submodules.items():
            if cfg_name in trainable:
                submodule = getattr(nah, attr_name, None)
                if submodule is not None:
                    submodule.train()

        if "heatmap_control" in trainable:
            adapters_fn = getattr(nah, "heatmap_control_adapters", None)
            adapters = tuple(adapters_fn()) if callable(adapters_fn) else ()
            for adapter in adapters:
                adapter.train()

    if ppa_enabled:
        ppa_chain.eval()
        ppa_chain.future_head.train()
        if str(stage_cfg.get("past_plan_action_stage", "stage2_joint")) == "stage2_joint":
            ppa_chain.bridge.train()

    logger.info(
        "  Selective train mode: frozen modules eval; trainable modules=%s",
        sorted(trainable),
    )


def train_one_epoch(
    model: VLNPipeline,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
    cfg: dict,
    epoch: int,
    logger,
    tb_writer: SummaryWriter | None = None,
    global_step_offset: int = 0,
    stage_idx: int = 0,
    stage_name: str = "",
    stage_cfg: dict | None = None,
    max_batches: int | None = None,
    vis_dir: Path | None = None,
    gpu_heatmap_computer: GPUHeatmapComputer | None = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    ema: EMAModel | None = None,
    metrics_jsonl_path: Path | None = None,
    total_train_steps: int = 1,
    dist_context: DistributedContext | None = None,
    ckpt_manager=None,
    mid_epoch_save_every: int = 500,
    nextdit_warmup_steps: int = 0,
    skip_first_n_batches: int | None = None,
    l2_sp_reference: dict[str, torch.Tensor] | None = None,
    checkpoint_selection_state: dict | None = None,
    train_sampler=None,
    actual_batch_observer: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Train one epoch.

    Parameters
    ----------
    skip_first_n_batches:
        When set (mid-epoch resume), fast-forward the dataloader iterator by
        this many batches before beginning the actual training loop.
    actual_batch_observer:
        Optional audit-only callback invoked for each batch that survives both
        skip/max filtering, immediately before the provider contract and model
        forward. It must not mutate the batch.
    """
    dist_context = dist_context or DistributedContext(
        enabled=False,
        device=torch.device(cfg['model'].get('device', 'cuda')),
    )
    model_module = _unwrap_model(model)
    model.train()
    _apply_bridge_only_train_mode(model_module, stage_cfg or {}, logger)
    synced_trainable_modules = (
        _get_supported_trainable_sync_modules(model_module, stage_cfg)
        if dist_context.enabled
        else []
    )
    total_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_heatmap_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_action_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_lm_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_l2_sp_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_future_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_preserve_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_delta_z_loss = torch.zeros((), device=dist_context.device, dtype=torch.float64)
    total_heatmap_components = torch.zeros(
        len(_HEATMAP_COMPONENT_KEYS),
        device=dist_context.device,
        dtype=torch.float64,
    )
    total_trajectory_view_counts = torch.zeros(
        len(_TRAJECTORY_VIEW_ORDER),
        device=dist_context.device,
        dtype=torch.float64,
    )
    num_batches = 0

    optim_cfg = cfg['optim']
    loss_cfg = cfg['loss']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)
    l2_sp_cfg = loss_cfg.get('l2_sp', {})
    l2_sp_weight = (
        float(l2_sp_cfg.get('weight', 0.0) or 0.0)
        if bool(l2_sp_cfg.get('enabled', False))
        else 0.0
    )
    l2_sp_normalization = str(
        l2_sp_cfg.get('normalization', 'mean_parameter_mse')
    )
    trajectory_sequence_mode = str(
        (stage_cfg or {}).get('trajectory_sequence_mode', 'all')
    )
    trajectory_view_weights_cfg = loss_cfg.get('trajectory_view_weights', {})

    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
    need_heatmap_targets = train_history or train_future
    heatmap_control_cfg = (
        cfg.get('model', {})
        .get('action_head', {})
        .get('nextdit', {})
        .get('heatmap_control', {})
    )
    heatmap_control_enabled = bool(heatmap_control_cfg.get('enabled', False))
    ppa_enabled = getattr(model_module, 'past_plan_action', None) is not None
    ppa_stage = str(
        stage_cfg.get(
            'past_plan_action_stage',
            'stage2_joint' if train_action else 'stage1_map_pretrain',
        )
    )
    device = dist_context.device
    future_loss_fn = (
        build_future_heatmap_loss_fn(cfg, device) if ppa_enabled else None
    )
    ppa_loss_weights = PastPlanActionLossWeights(
        action=float(loss_cfg.get('trajectory_weight', 1.0)),
        history=float(loss_cfg.get('history_weight', 0.3)),
        future=float(loss_cfg.get('future_weight', 0.3)),
        preserve=float(loss_cfg.get('preserve_weight', 0.5)),
        delta_z=float(loss_cfg.get('delta_z_weight', 0.01)),
    )

    l2_sp_device_reference = None
    if l2_sp_weight > 0.0 and l2_sp_reference:
        l2_sp_device_reference = {
            name: value.to(device=device, dtype=torch.float32)
            for name, value in l2_sp_reference.items()
        }

    if dist_context.is_main:
        logger.info(
            "  Trajectory supervision: sequence_mode=%s view_weights=%s",
            trajectory_sequence_mode,
            (
                trajectory_view_weights_cfg.get('weights', {})
                if trajectory_view_weights_cfg.get('enabled', False)
                else "disabled"
            ),
        )

    hm_loss_fn = build_heatmap_loss_fn(cfg, device)
    hm_loss_fn.set_temperature(
        get_heatmap_temperature(cfg, global_step_offset, total_train_steps)
    )

    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式: 只处理 {total_batches} batches")

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{stage_cfg['epochs']}",
        total=total_batches,
        ncols=cfg['log'].get('tqdm_ncols', 120),
        disable=not dist_context.is_main,
    )

    global_step = 0
    valid_batch_count = 0
    enable_timing = cfg.get('log', {}).get('enable_timing', False)
    timing_stats = {
        'data_wait_s': 0.0,
        'gt_s': 0.0,
        'forward_s': 0.0,
        'backward_s': 0.0,
        'optimizer_s': 0.0,
        'prepare_inputs_s': 0.0,
        'qwen_forward_s': 0.0,
        'heatmap_decode_s': 0.0,
        'pipeline_qwen_total_s': 0.0,
    } if enable_timing else {}
    profiled_steps = 0
    prev_step_end = time.perf_counter() if enable_timing else 0.0

    diag_interval = cfg['log'].get('diag_interval', 100)
    log_interval = max(1, int(cfg['log'].get('log_interval', 10)))
    tensorboard_interval = max(1, int(cfg['log'].get('tensorboard_interval', log_interval)))
    progress_interval = int(cfg['log'].get('progress_interval', log_interval))
    page_cache_drop_enabled = bool(cfg['log'].get('page_cache_drop_enabled', True))
    page_cache_drop_interval = int(cfg['log'].get('page_cache_drop_interval', 25))
    page_cache_drop_threshold = float(cfg['log'].get('page_cache_drop_threshold', 0.80))
    aligned_interval = grad_accum_steps * diag_interval
    for head_attr in ['heatmap_vln']:
        head = getattr(model, head_attr, None)
        if head is not None and hasattr(head, '_training_step_counter'):
            head._training_step_counter = 0
            head._inference_interval = aligned_interval

    trainable_params = _get_trainable_params(model_module)

    import psutil
    _mem_log_proc = psutil.Process()
    _cg_limit = _CG_LIMIT_GB

    for i, batch in enumerate(pbar):
        # Fast-forward dataloader on mid-epoch resume — the checkpoint was
        # saved *after* completing `skip_first_n_batches` batches (counter
        # incremented post-step), so we skip index 0 through
        # skip_first_n_batches-1 and resume from index skip_first_n_batches.
        if skip_first_n_batches is not None and i < skip_first_n_batches:
            continue
        if max_batches is not None and i >= max_batches:
            break

        if actual_batch_observer is not None:
            actual_batch_observer(i, batch)
        assert_required_history_pose_provider(batch, stage_cfg)
        if ppa_enabled:
            assert_no_future_teacher_inputs(batch)

        if (i <= 5 or i % 25 == 0) and cfg['log'].get('show_gpu_memory', False):
            main_rss = _mem_log_proc.memory_info().rss / (1024 * 1024)
            children = _mem_log_proc.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children) / (1024 * 1024)
            cg_used = _cgroup_mem_usage_gb()
            cg_info = f"cgroup: {cg_used:.1f}/{_cg_limit:.0f}GB" if _cg_limit > 0 else f"cgroup: {cg_used:.1f}GB(no limit)"
            logger.debug(
                "[MAIN batch=%d] main_rss=%.0fMB children(%d)=%.0fMB total=%.0fMB | %s",
                i, main_rss, len(children), child_rss, main_rss + child_rss, cg_info,
            )
        if (
            page_cache_drop_enabled
            and page_cache_drop_interval > 0
            and (i <= 5 or i % page_cache_drop_interval == 0)
        ):
            _drop_page_cache(threshold=page_cache_drop_threshold)

        if enable_timing:
            loop_start = time.perf_counter()
            timing_stats['data_wait_s'] += max(loop_start - prev_step_end, 0.0)

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
                    "internnav_single_view batch is incomplete: "
                    f"missing={missing_single_view}"
                )
            if train_action or train_lm:
                raise RuntimeError(
                    "worker-preprocessed internnav_single_view batches are "
                    "heatmap-only; native System1/System2 must remain frozen"
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
            if enable_timing:
                gt_start = time.perf_counter()
            if _should_use_gpu_gt(batch, gpu_heatmap_computer):
                history_poses = batch['history_poses'].to(device, non_blocking=True)
                current_poses = batch['current_pose'].to(device, non_blocking=True)
                current_depths = batch['current_depth'].to(device, non_blocking=True) if gpu_has_depth and 'current_depth' in batch else None
                intrinsics = batch['intrinsics'].to(device, non_blocking=True) if 'intrinsics' in batch else None
                gt_heatmap = gpu_heatmap_computer.compute_batch(
                    history_poses=history_poses,
                    current_poses=current_poses,
                    current_depths=current_depths,
                    intrinsics=intrinsics,
                    depth_normalized=gpu_depth_normalized,
                )
                if 'is_flipped' in batch:
                    flip_mask = batch['is_flipped']
                    if flip_mask.any():
                        for b_idx in range(gt_heatmap.shape[0]):
                            if flip_mask[b_idx]:
                                gt_heatmap[b_idx] = gt_heatmap[b_idx].flip(dims=[-1])
            else:
                gt_heatmap = batch['heatmap'].to(device, non_blocking=True)
            if enable_timing:
                timing_stats['gt_s'] += time.perf_counter() - gt_start

        if enable_timing:
            forward_start = time.perf_counter()
        with make_autocast_context(device, optim_cfg.get('amp', 'bf16')):
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
            heatmap_single_view_inputs = batch.get('heatmap_single_view_inputs')
            heatmap_single_view_num_histories = batch.get(
                'heatmap_single_view_num_histories'
            )
            heatmap_control_history_mask = batch.get('heatmap_control_history_mask')
            history_valid_mask = batch.get('history_valid_mask')
            history_age_steps = batch.get('history_age_steps')
            history_rel_poses = batch.get('history_rel_poses')
            if history_rel_poses is not None:
                history_rel_poses = history_rel_poses.to(device, non_blocking=True)
            if single_view_heatmap_batch:
                # The frozen Qwen vision tower consumes worker-preprocessed
                # images directly.  No raw RGB tensor crosses the worker
                # boundary and the language model is not executed.
                video_frames = None
            elif panoramic_inputs_batch is not None:
                # Worker-tokenized Qwen inputs already contain every panorama.
                # The pipeline only needs this tensor for batch/shape metadata.
                video_frames = current_frame.unsqueeze(1)
            else:
                # Pre-allocate to avoid implicit copy from torch.cat on
                # potentially non-contiguous tensors.
                B, K, C, H, W = history_frames.shape
                video_frames = torch.empty(
                    B, K + 1, C, H, W,
                    dtype=history_frames.dtype, device=history_frames.device,
                )
                video_frames[:, :K] = history_frames
                video_frames[:, -1] = current_frame

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
                sample_trajectory=False,
                return_heatmaps=train_history or train_future or heatmap_control_enabled,
                return_heatmap_logits=train_history or train_future or heatmap_control_enabled,
                return_actions=train_action,
                return_future_heatmaps=train_future and ppa_enabled,
                return_lm_loss=train_lm,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,
                action_valid=action_valid if train_action else None,
                gt_stop=is_stop if train_action else None,
                gt_history_heatmap=gt_heatmap if train_history else None,
            )

            heatmap_loss = torch.tensor(0.0, device=device)
            loss_dict = None

            if train_history and 'visibility' in output and 'heatmaps' in output and gt_heatmap is not None:
                if 'gt_visibility' in batch:
                    gt_vis = batch['gt_visibility'].to(device, non_blocking=True)
                else:
                    gt_vis = gt_heatmap.amax(dim=(-2, -1)).clamp(0, 1).to(device)
                hm_history_mask = batch.get('history_mask')
                if hm_history_mask is not None:
                    hm_history_mask = hm_history_mask.to(device, non_blocking=True)
                # Keep the numerically sensitive heatmap objective identical
                # to validation and outside the surrounding AMP policy.
                with torch.autocast(device_type=device.type, enabled=False):
                    loss_dict = hm_loss_fn(
                        output['visibility'].float(),
                        output['heatmaps'].float(),
                        gt_vis=gt_vis.float(),
                        gt_heatmaps=gt_heatmap.to(
                            device, dtype=torch.float32, non_blocking=True
                        ),
                        history_mask=hm_history_mask,
                        pred_heatmap_logits=output['heatmap_logits'].float(),
                    )
                heatmap_loss = loss_dict['total']

            trajectory_loss = torch.tensor(0.0, device=device)
            action_plan_losses = None

            if train_action:
                if hasattr(model_module, 'nextdit_action_head') and model_module.nextdit_action_head is not None:
                    if 'trajectory' not in batch:
                        raise RuntimeError(
                            "train_action=True but batch has no trajectory target. "
                            "Check the trajectory dataset/collator configuration."
                        )
                    if 'traj_hidden_states' not in output:
                        raise RuntimeError(
                            "train_action=True but model output has no traj_hidden_states. "
                            "Check that the panoramic tokenized collator is enabled and "
                            "that TRAJ latent queries are being passed to Qwen."
                        )
                    gt_trajectory = batch['trajectory'].to(device, non_blocking=True)
                    trajectory_valid = batch['trajectory_valid'].to(device, non_blocking=True)
                    traj_images = batch.get('traj_images')
                    if traj_images is not None:
                        traj_images = traj_images.to(device, non_blocking=True)
                    gt_trajectory, trajectory_valid, traj_images = (
                        _prepare_trajectory_sequence_inputs(
                            gt_trajectory,
                            trajectory_valid,
                            traj_images,
                            mode=trajectory_sequence_mode,
                        )
                    )
                    view_weights = _trajectory_view_sample_weights(
                        batch.get('pano_view_id'),
                        trajectory_view_weights_cfg,
                        device=device,
                    )
                    for raw_view in batch.get('pano_view_id') or []:
                        view = str(raw_view).lower()
                        if view in _TRAJECTORY_VIEW_ORDER:
                            total_trajectory_view_counts[
                                _TRAJECTORY_VIEW_ORDER.index(view)
                            ] += 1
                    if view_weights is not None:
                        if trajectory_valid is None:
                            trajectory_valid = view_weights
                        elif trajectory_valid.ndim == 1:
                            trajectory_valid = trajectory_valid.float() * view_weights
                        elif trajectory_valid.ndim == 2:
                            trajectory_valid = (
                                trajectory_valid.float() * view_weights.unsqueeze(1)
                            )
                        else:
                            raise RuntimeError(
                                "Unsupported trajectory_valid shape for view weighting: "
                                f"{tuple(trajectory_valid.shape)}"
                            )
                    if ppa_enabled:
                        for key in ('plan_z0', 'plan_z'):
                            if key not in output:
                                raise RuntimeError(
                                    f"Past->Plan->Action output lacks {key}"
                                )
                        action_plan_losses = compute_shared_plan_action_losses(
                            action_head=model_module.nextdit_action_head,
                            plan_z0=output['plan_z0'],
                            plan_z=output['plan_z'],
                            gt_trajectory=gt_trajectory,
                            trajectory_valid=trajectory_valid,
                            traj_images=traj_images,
                            preserve_weight=0.0,
                            delta_weight=0.0,
                        )
                        trajectory_loss = action_plan_losses['action']
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
                                "heatmap control output is incomplete: "
                                f"missing={missing_control}"
                            )
                        heatmap_tokens = output['heatmap_control_tokens']
                        heatmap_mask = output['heatmap_control_mask']
                        heatmap_valid = output['heatmap_control_valid']
                    if not ppa_enabled:
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

            future_loss = torch.tensor(0.0, device=device)
            future_loss_dict = None
            ppa_losses = None
            if ppa_enabled and train_future:
                required_future = {
                    'future_visibility_logits',
                    'future_heatmaps',
                    'future_heatmap_logits',
                }
                missing_future = sorted(required_future - set(output))
                if missing_future:
                    raise RuntimeError(
                        f"Future Head output is incomplete: {missing_future}"
                    )
                future_loss_dict = future_loss_fn(
                    pred_visibility_logits=output['future_visibility_logits'].float(),
                    pred_heatmaps=output['future_heatmaps'].float(),
                    pred_heatmap_logits=output['future_heatmap_logits'].float(),
                    gt_visibility=batch['future_trajectory_visibility'].to(
                        device, dtype=torch.float32, non_blocking=True
                    ),
                    gt_heatmaps=batch['future_trajectory_heatmap'].to(
                        device, dtype=torch.float32, non_blocking=True
                    ),
                    future_time_mask=batch['future_trajectory_time_mask'].to(
                        device, dtype=torch.bool, non_blocking=True
                    ),
                )
                future_loss = future_loss_dict['total']

            lm_loss = torch.tensor(0.0, device=device)
            if train_lm:
                if 'lm_loss' not in output or output['lm_loss'] is None:
                    raise RuntimeError(
                        "train_lm=True but model output has no lm_loss. "
                        "Check PanoramicTokenizedCollator labels and Qwen forward wiring."
                    )
                lm_loss = output['lm_loss']

            l2_sp_loss = torch.tensor(0.0, device=device)
            if l2_sp_weight > 0.0 and l2_sp_reference:
                l2_sp_loss = compute_l2_sp_loss(
                    model_module,
                    l2_sp_device_reference,
                    device=device,
                    normalization=l2_sp_normalization,
                )

            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 0.0)
            lm_weight = loss_cfg.get('lm_weight', stage_cfg.get('lm_weight', 1.0))

            if ppa_enabled:
                ppa_losses = compose_past_plan_action_loss(
                    stage=ppa_stage,
                    history_loss=loss_dict if train_history else None,
                    future_loss=future_loss_dict,
                    action_plan_losses=action_plan_losses,
                    weights=ppa_loss_weights,
                )
                loss = ppa_losses['total'] + lm_weight * lm_loss + l2_sp_weight * l2_sp_loss
            else:
                loss = (
                    heatmap_weight * heatmap_loss
                    + trajectory_weight * trajectory_loss
                    + lm_weight * lm_loss
                    + l2_sp_weight * l2_sp_loss
                )
            loss = loss / grad_accum_steps
        if enable_timing:
            timing_stats['forward_s'] += time.perf_counter() - forward_start
            profiled_steps += 1

            metadata = output.get('processing_metadata', {}) if isinstance(output, dict) else {}
            model_timings = metadata.get('timings') or {}
            for key, value in model_timings.items():
                if isinstance(value, (int, float)):
                    timing_stats[key] = timing_stats.get(key, 0.0) + float(value)

        if enable_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if enable_timing:
            timing_stats['backward_s'] += time.perf_counter() - backward_start
        valid_batch_count += 1

        if valid_batch_count % grad_accum_steps == 0:
            if enable_timing:
                opt_start = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
                synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
                optimizer.step()
            _next_step = global_step + 1
            if tb_writer is not None and _next_step % diag_interval == 0:
                _cached_lora_grads = _collect_lora_grad_norms(model_module)
            else:
                _cached_lora_grads = None
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update()
            if enable_timing:
                timing_stats['optimizer_s'] += time.perf_counter() - opt_start
            global_step += 1

            abs_step = global_step_offset + global_step
            if nextdit_warmup_steps > 0 and abs_step == nextdit_warmup_steps:
                end_nextdit_warmup(model_module, logger, stage_cfg=stage_cfg)
                trainable_params = _get_trainable_params(model_module)
                if dist_context.enabled:
                    synced_trainable_modules = _get_supported_trainable_sync_modules(
                        model_module, stage_cfg
                    )

            current_heatmap_temperature = get_heatmap_temperature(
                cfg,
                abs_step,
                total_train_steps,
            )
            hm_loss_fn.set_temperature(current_heatmap_temperature)

            show_gpu_memory = cfg['log'].get('show_gpu_memory', False)
            log_this_step = global_step % log_interval == 0 or global_step <= 3
            tb_this_step = (
                tb_writer is not None
                and (global_step % tensorboard_interval == 0 or global_step <= 3)
            )
            step_scalars = None
            if log_this_step or tb_this_step:
                step_scalars = {
                    "loss": float((loss.detach() * grad_accum_steps).item()),
                    "heatmap_loss": float(heatmap_loss.detach().item()),
                    "trajectory_loss": float(trajectory_loss.detach().item()),
                    "lm_loss": float(lm_loss.detach().item()),
                    "l2_sp_loss": float(l2_sp_loss.detach().item()),
                }

            if log_this_step:
                all_lrs = scheduler.get_last_lr()
                lr_strs = []
                for gi, lr_val in enumerate(all_lrs):
                    gname = optimizer.param_groups[gi].get('name', f'g{gi}')
                    lr_strs.append(f"{gname}={lr_val:.2e}")
                lr_display = ", ".join(lr_strs)
                gpu_mem_str = ""
                if show_gpu_memory:
                    gpu_mem_str = (
                        f" | GPU: alloc={torch.cuda.memory_allocated() / 1024**3:.1f}GB"
                        f" reserved={torch.cuda.memory_reserved() / 1024**3:.1f}GB"
                        f" max={torch.cuda.max_memory_allocated() / 1024**3:.1f}GB"
                    )
                traj_str = (
                    f", traj: {step_scalars['trajectory_loss']:.4f}"
                    if step_scalars["trajectory_loss"] > 0 else ""
                )
                lm_str = f", lm: {step_scalars['lm_loss']:.4f}" if train_lm else ""
                l2_sp_str = (
                    f", l2sp: {step_scalars['l2_sp_loss']:.6f}"
                    if l2_sp_weight > 0.0 and step_scalars["l2_sp_loss"] > 0
                    else ""
                )
                metadata = output.get('processing_metadata', {}) if isinstance(output, dict) else {}
                input_stats_str = ""
                if metadata.get('pano_seq_len') is not None:
                    input_stats_str = (
                        f" | In: L={metadata.get('pano_seq_len')} "
                        f"img_groups={metadata.get('pano_image_groups', 0)} "
                        f"vid_groups={metadata.get('pano_video_groups', 0)} "
                        f"img_tok={metadata.get('num_image_tokens', 0)} "
                        f"hist_max={metadata.get('pano_history_max', 0)}"
                    )
                logger.info(
                    f"[{stage_name}] "
                    f"Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Step {global_step} | "
                    f"Loss: {step_scalars['loss']:.4f} "
                    f"(hm: {step_scalars['heatmap_loss']:.4f}{traj_str}{lm_str}{l2_sp_str}) | "
                    f"LR: [{lr_display}]"
                    + gpu_mem_str
                    + input_stats_str
                    + (
                        (
                            f" | T[s] data={_mean_timing(timing_stats, profiled_steps, 'data_wait_s'):.3f} "
                            f"gt={_mean_timing(timing_stats, profiled_steps, 'gt_s'):.3f} "
                            f"fwd={_mean_timing(timing_stats, profiled_steps, 'forward_s'):.3f} "
                            f"(prep={_mean_timing(timing_stats, profiled_steps, 'prepare_inputs_s'):.3f} "
                            f"qwen={_mean_timing(timing_stats, profiled_steps, 'qwen_forward_s'):.3f} "
                            f"decode={_mean_timing(timing_stats, profiled_steps, 'heatmap_decode_s'):.3f}) "
                            f"bwd={_mean_timing(timing_stats, profiled_steps, 'backward_s'):.3f} "
                            f"opt={_mean_timing(timing_stats, max(global_step, 1), 'optimizer_s'):.3f}"
                            f"{' | ' + _format_decode_internal_timing(timing_stats, profiled_steps) if _format_decode_internal_timing(timing_stats, profiled_steps) else ''}"
                            f"{' | ' + _format_qwen_internal_timing(timing_stats, profiled_steps) if _format_qwen_internal_timing(timing_stats, profiled_steps) else ''}"
                        ) if enable_timing else ""
                    )
                )
                if isinstance(loss_dict, dict):
                    auxiliary = " ".join(
                        f"{key.removesuffix('_loss')}="
                        f"{float(loss_dict[key]):.4f}"
                        for key in (
                            "neg_loss",
                            "view_macro_loss",
                            "direction_macro_loss",
                            "panoramic_view_loss",
                        )
                        if key in loss_dict and float(loss_dict[key]) > 0
                    )
                    logger.info(
                        f"  [HM] peak={loss_dict.get('peak_loss', 0):.4f} "
                        f"vis={loss_dict.get('vis_loss', 0):.4f} "
                        f"coord={loss_dict.get('coord_loss', 0):.4f}"
                        f"{' ' + auxiliary if auxiliary else ''}"
                    )
                if metrics_jsonl_path is not None:
                    step_record = {
                        "record_type": "train_step",
                        "stage": stage_name,
                        "epoch": epoch,
                        "batch": i + 1,
                        "global_step": global_step,
                        "loss": step_scalars["loss"],
                        "heatmap_loss": step_scalars["heatmap_loss"],
                        "trajectory_loss": step_scalars["trajectory_loss"],
                            "lm_loss": step_scalars["lm_loss"],
                            "l2_sp_loss": step_scalars["l2_sp_loss"],
                            "lrs": {
                            optimizer.param_groups[gi].get("name", f"g{gi}"): lr_val
                            for gi, lr_val in enumerate(all_lrs)
                        },
                    }
                    if isinstance(loss_dict, dict):
                        for k in _HEATMAP_COMPONENT_KEYS:
                            if k in loss_dict:
                                step_record[f'hm_{k}'] = loss_dict[k].item()
                    if show_gpu_memory:
                        step_record["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
                    _append_jsonl(metrics_jsonl_path, step_record)

            if tb_this_step:
                actual_step = global_step_offset + global_step
                tb_writer.add_scalar('train/loss', step_scalars["loss"], actual_step)
                tb_writer.add_scalar('train/heatmap_loss', step_scalars["heatmap_loss"], actual_step)
                if step_scalars["trajectory_loss"] > 0:
                    tb_writer.add_scalar('train/trajectory_loss', step_scalars["trajectory_loss"], actual_step)
                if train_lm:
                    tb_writer.add_scalar('train/lm_loss', step_scalars["lm_loss"], actual_step)
                if l2_sp_weight > 0.0 and step_scalars["l2_sp_loss"] > 0:
                    tb_writer.add_scalar('train/l2_sp_loss', step_scalars["l2_sp_loss"], actual_step)
                if isinstance(loss_dict, dict):
                    for k in _HEATMAP_COMPONENT_KEYS:
                        if k in loss_dict:
                            tb_writer.add_scalar(f'train/hm_{k}', loss_dict[k].item(), actual_step)
                if enable_timing:
                    tb_writer.add_scalar('timing/data_wait_s', _mean_timing(timing_stats, profiled_steps, 'data_wait_s'), actual_step)
                    tb_writer.add_scalar('timing/gt_s', _mean_timing(timing_stats, profiled_steps, 'gt_s'), actual_step)
                    tb_writer.add_scalar('timing/forward_s', _mean_timing(timing_stats, profiled_steps, 'forward_s'), actual_step)
                    tb_writer.add_scalar('timing/backward_s', _mean_timing(timing_stats, profiled_steps, 'backward_s'), actual_step)
                    tb_writer.add_scalar('timing/optimizer_s', _mean_timing(timing_stats, max(global_step, 1), 'optimizer_s'), actual_step)
                    if profiled_steps > 0:
                        tb_writer.add_scalar('timing/prepare_inputs_s', _mean_timing(timing_stats, profiled_steps, 'prepare_inputs_s'), actual_step)
                        tb_writer.add_scalar('timing/qwen_forward_s', _mean_timing(timing_stats, profiled_steps, 'qwen_forward_s'), actual_step)
                        tb_writer.add_scalar('timing/heatmap_decode_s', _mean_timing(timing_stats, profiled_steps, 'heatmap_decode_s'), actual_step)
                        for key in ('decode_vit_fusion_s', 'decode_llm_fusion_s', 'decode_coarse_s', 'decode_fine_s', 'decode_post_s'):
                            if key in timing_stats:
                                tb_writer.add_scalar(f'timing/{key}', _mean_timing(timing_stats, profiled_steps, key), actual_step)
                        for key in sorted(timing_stats.keys()):
                            if key.startswith('qwen_') and key not in {
                                'qwen_forward_s',
                            }:
                                tb_writer.add_scalar(f'timing/{key}', _mean_timing(timing_stats, profiled_steps, key), actual_step)
                for gi, lr_val in enumerate(scheduler.get_last_lr()):
                    gname = optimizer.param_groups[gi].get('name', f'g{gi}')
                    tb_writer.add_scalar(f'lr/{gname}', lr_val, actual_step)

                diag_interval = cfg['log'].get('diag_interval', 100)
                if global_step % diag_interval == 0:
                    _log_heatmap_diagnostics(
                        output, gt_heatmap, batch, tb_writer, actual_step,
                        cfg, logger,
                    )
                    _log_lora_diagnostics(
                        model_module, tb_writer, actual_step, cfg, logger,
                        cached_grad_norms=_cached_lora_grads,
                    )

        vis_interval = cfg['log'].get('vis_every_steps', 500)
        if (
            tb_writer is not None
            and train_history
            and 'heatmaps' in output
            and not single_view_heatmap_batch
            and global_step % vis_interval == 0
            and global_step > 0
        ):
            vis_path = visualize_heatmap_predictions(
                model=model_module,
                batch=batch,
                output=output,
                epoch=epoch,
                step=global_step,
                output_dir=vis_dir if vis_dir else Path('.'),
                num_samples=2,
                gt_heatmap_override=gt_heatmap if _should_use_gpu_gt(batch, gpu_heatmap_computer) else None,
            )
            if vis_path:
                try:
                    import cv2
                    vis_img = cv2.imread(str(vis_path))
                    if vis_img is not None:
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        vis_img = vis_img.transpose(2, 0, 1)
                        tb_writer.add_image('train/heatmap_viz', vis_img, global_step_offset + global_step)
                except Exception:
                    logger.debug("Failed to write visualization to TensorBoard", exc_info=True)

        iter_loss_t = (loss.detach() * grad_accum_steps).to(dtype=torch.float64)
        iter_hm_t = heatmap_loss.detach().to(dtype=torch.float64)
        iter_traj_t = trajectory_loss.detach().to(dtype=torch.float64)
        iter_lm_t = lm_loss.detach().to(dtype=torch.float64)
        iter_l2_sp_t = l2_sp_loss.detach().to(dtype=torch.float64)
        iter_future_t = future_loss.detach().to(dtype=torch.float64)
        if ppa_enabled and ppa_losses is not None:
            iter_preserve_t = ppa_losses['preserve'].detach().to(dtype=torch.float64)
            iter_delta_z_t = ppa_losses['delta_z_l2'].detach().to(dtype=torch.float64)
        else:
            iter_preserve_t = torch.zeros_like(iter_future_t)
            iter_delta_z_t = torch.zeros_like(iter_future_t)

        total_loss += iter_loss_t
        total_heatmap_loss += iter_hm_t
        total_action_loss += iter_traj_t
        total_lm_loss += iter_lm_t
        total_l2_sp_loss += iter_l2_sp_t
        total_future_loss += iter_future_t
        total_preserve_loss += iter_preserve_t
        total_delta_z_loss += iter_delta_z_t
        if isinstance(loss_dict, dict):
            for component_index, component_key in enumerate(
                _HEATMAP_COMPONENT_KEYS
            ):
                component = loss_dict.get(component_key)
                if torch.is_tensor(component):
                    total_heatmap_components[component_index] += (
                        component.detach().to(dtype=torch.float64)
                    )
        num_batches += 1

        if progress_interval > 0 and (num_batches <= 3 or num_batches % progress_interval == 0):
            _iter_loss = float(iter_loss_t.item())
            _iter_hm = float(iter_hm_t.item())
            _iter_traj = float(iter_traj_t.item())
            _iter_lm = float(iter_lm_t.item())
            _avg_hm = float((total_heatmap_loss / max(num_batches, 1)).item())
            pbar.set_postfix({
                'loss': f"{_iter_loss:.4f}",
                'hm': f"{_avg_hm:.4f}",
                'traj': f"{_iter_traj:.4f}",
                'lm': f"{_iter_lm:.4f}",
            })

        del output, loss, heatmap_loss, gt_heatmap
        del trajectory_loss, lm_loss, l2_sp_loss
        loss_dict = None
        del video_frames
        del current_views_batch, history_panoramas_batch
        del single_view_inputs_batch, single_view_num_histories
        del panoramic_inputs_batch, panoramic_num_histories, panoramic_text_anchor_positions
        del history_frames, current_frame, gt_action, action_valid, is_stop, text
        del batch

        if num_batches % 50 == 0:
            gc.collect()
            _malloc_trim()
            if cfg['log'].get('show_gpu_memory', False):
                post_rss = _mem_log_proc.memory_info().rss / (1024 * 1024)
                gc_stats = gc.get_stats()
                cg_now = _cgroup_mem_usage_gb()
                cg_str = f"cgroup={cg_now:.1f}/{_cg_limit:.0f}GB" if _cg_limit > 0 else f"cgroup={cg_now:.1f}GB"
                logger.debug(
                    "[MAIN batch=%d post-gc] rss=%.0fMB | %s | gc: gen0=%d gen1=%d gen2=%d",
                    i, post_rss, cg_str,
                    gc_stats[0]['collected'], gc_stats[1]['collected'], gc_stats[2]['collected'],
                )
            if page_cache_drop_enabled:
                _drop_page_cache(threshold=page_cache_drop_threshold)

            if num_batches % 200 == 0:
                torch.cuda.empty_cache()
                if tb_writer is not None:
                    tb_writer.flush()

        completed_epoch_batches = i + 1
        if (
            ckpt_manager is not None
            and mid_epoch_save_every > 0
            and completed_epoch_batches > 0
            and completed_epoch_batches % mid_epoch_save_every == 0
            and completed_epoch_batches < len(train_loader)
            and dist_context.is_main
        ):
            model_module_for_save = _unwrap_model(model)
            mid_metrics = {
                'total_loss': float((total_loss / num_batches).item()),
                'heatmap_loss': float((total_heatmap_loss / num_batches).item()),
                'lm_loss': float((total_lm_loss / num_batches).item()),
                'l2_sp_loss': float((total_l2_sp_loss / num_batches).item()),
            }
            mid_extra_state = {}
            if l2_sp_reference:
                mid_extra_state["l2_sp_reference_state"] = l2_sp_reference
            if checkpoint_selection_state is not None:
                mid_extra_state["checkpoint_selection_state"] = (
                    checkpoint_selection_state
                )
            if train_sampler is not None and hasattr(train_sampler, "state_dict"):
                sampler_state = train_sampler.state_dict()
                if sampler_state.get("schema") == (
                    "heatmapvln-deterministic-mixture-sampler-v1"
                ):
                    mid_extra_state["mixture_sampler_state"] = sampler_state
            ckpt_manager.save(
                model=model_module_for_save,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stage_idx=stage_idx,
                stage_name=stage_name,
                metrics=mid_metrics,
                cfg=cfg,
                is_best=False,
                scaler=scaler,
                batch=completed_epoch_batches,
                extra_state=mid_extra_state or None,
                ema=ema,
            )
            logger.info(
                f"  Mid-epoch checkpoint saved at batch {completed_epoch_batches} "
                f"(loss={mid_metrics['total_loss']:.4f})"
            )

        prev_step_end = time.perf_counter()

    # Handle remaining gradients
    remaining = valid_batch_count % grad_accum_steps
    if remaining > 0:
        # Every microbatch loss was divided by grad_accum_steps above.  A
        # partial final window therefore needs to be renormalized to the mean
        # of its actual `remaining` microbatches before synchronization and
        # clipping.  Without this, e.g. a 1/2 tail step has half-sized grads.
        tail_grad_scale = grad_accum_steps / remaining
        if scaler is not None:
            scaler.unscale_(optimizer)
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.mul_(tail_grad_scale)
            synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.mul_(tail_grad_scale)
            synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if ema is not None:
            ema.update()
        global_step += 1

        abs_step2 = global_step_offset + global_step
        if nextdit_warmup_steps > 0 and abs_step2 == nextdit_warmup_steps:
            end_nextdit_warmup(model_module, logger, stage_cfg=stage_cfg)
            trainable_params = _get_trainable_params(model_module)
            if dist_context.enabled:
                synced_trainable_modules = _get_supported_trainable_sync_modules(
                    model_module, stage_cfg
                )

        hm_loss_fn.set_temperature(
            get_heatmap_temperature(
                cfg,
                abs_step2,
                total_train_steps,
            )
        )

    totals = torch.stack(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_lm_loss,
            total_l2_sp_loss,
            total_future_loss,
            total_preserve_loss,
            total_delta_z_loss,
            torch.tensor(float(num_batches), device=device, dtype=torch.float64),
        ]
    )
    _dist_all_reduce_in_place(totals)
    _dist_all_reduce_in_place(total_heatmap_components)
    _dist_all_reduce_in_place(total_trajectory_view_counts)

    reduced_num_batches = max(int(totals[8].item()), 1)
    view_count_metrics = {
        f'trajectory_view_count_{view}': int(total_trajectory_view_counts[idx].item())
        for idx, view in enumerate(_TRAJECTORY_VIEW_ORDER)
    }
    if dist_context.is_main and total_trajectory_view_counts.sum().item() > 0:
        logger.info("  Trajectory view samples: %s", view_count_metrics)
    heatmap_component_metrics = {
        f"hm_{key}": (
            total_heatmap_components[index] / reduced_num_batches
        ).item()
        for index, key in enumerate(_HEATMAP_COMPONENT_KEYS)
    }
    return {
        'total_loss': (totals[0] / reduced_num_batches).item(),
        'heatmap_loss': (totals[1] / reduced_num_batches).item(),
        'trajectory_loss': (totals[2] / reduced_num_batches).item(),
        'lm_loss': (totals[3] / reduced_num_batches).item(),
        'l2_sp_loss': (totals[4] / reduced_num_batches).item(),
        'future_heatmap_loss': (totals[5] / reduced_num_batches).item(),
        'preserve_loss': (totals[6] / reduced_num_batches).item(),
        'delta_z_loss': (totals[7] / reduced_num_batches).item(),
        'optimizer_steps': global_step,
        **heatmap_component_metrics,
        **view_count_metrics,
    }


# ---------------------------------------------------------------------------
# Internal: LoRA diagnostics logged to TensorBoard
# ---------------------------------------------------------------------------

_LORA_PARAM_RE = re.compile(r'layers\.(\d+)\..*?(\w+_proj)\.lora_([AB])\.')


def _collect_lora_grad_norms(model_module) -> dict | None:
    """Snapshot LoRA gradient norms before optimizer.zero_grad() destroys them.

    Returns ``{(layer_idx, module, 'A'|'B'): float}`` or *None* when no
    LoRA parameters carry gradients.
    """
    qwen_model = getattr(
        getattr(model_module, 'qwen2_5_vl', None), 'model', None,
    )
    if qwen_model is None:
        return None

    keys: list[tuple[int, str, str]] = []
    gpu_norms: list[torch.Tensor] = []
    for name, param in qwen_model.named_parameters():
        m = _LORA_PARAM_RE.search(name)
        if m and param.grad is not None:
            keys.append((int(m.group(1)), m.group(2), m.group(3)))
            gpu_norms.append(param.grad.float().norm())  # stays on GPU

    if not keys:
        return None

    # Single GPU→CPU sync instead of O(n_lora_params) sequential syncs.
    norms_cpu = torch.stack(gpu_norms).cpu().tolist()
    return {key: float(norm) for key, norm in zip(keys, norms_cpu)}


def _log_lora_diagnostics(
    model_module,
    tb_writer: SummaryWriter,
    actual_step: int,
    cfg: dict,
    logger,
    cached_grad_norms: dict | None = None,
):
    """Log LoRA weight norms, gradient norms, and delta_W to TensorBoard.

    Metrics written
    ---------------
    lora/B_norm_layer{L}        per-layer ||B||_F  (movement from init)
    lora/deltaW_layer{L}        per-layer (alpha/r)||BA||_F  (effective weight change)
    lora/grad_norm_layer{L}     per-layer gradient Frobenius norm (from cache)
    lora/grad_decay_L20_vs_L5   ratio of layer-20 / layer-5 gradient norms
    lora/total_B_norm           aggregate across all layers
    lora/total_deltaW_norm      aggregate across all layers
    lora/total_grad_norm        aggregate across all layers
    lora_detail/...             per-(layer, module) breakdown
    """
    qwen_model = getattr(
        getattr(model_module, 'qwen2_5_vl', None), 'model', None,
    )
    if qwen_model is None:
        return

    lora_params: dict[tuple[int, str, str], torch.nn.Parameter] = {}
    for name, param in qwen_model.named_parameters():
        m = _LORA_PARAM_RE.search(name)
        if m:
            lora_params[(int(m.group(1)), m.group(2), m.group(3))] = param

    if not lora_params:
        return

    lora_rank = cfg['model'].get('llm', {}).get('lora_rank', 16)
    lora_alpha = cfg['model'].get('llm', {}).get('lora_alpha', 32)
    scaling = lora_alpha / lora_rank

    layers = sorted({k[0] for k in lora_params})
    modules = sorted({k[1] for k in lora_params})

    total_B_sq = 0.0
    total_dW_sq = 0.0
    total_grad_sq = 0.0

    for layer_idx in layers:
        layer_B_sq = 0.0
        layer_grad_sq = 0.0
        layer_dW_sq = 0.0

        for module in modules:
            A = lora_params.get((layer_idx, module, 'A'))
            B = lora_params.get((layer_idx, module, 'B'))
            if A is None or B is None:
                continue

            B_norm = B.data.float().norm().item()
            layer_B_sq += B_norm ** 2

            with torch.no_grad():
                dW_norm = (B.data.float() @ A.data.float()).norm().item() * scaling
            layer_dW_sq += dW_norm ** 2

            tb_writer.add_scalar(
                f'lora_detail/B_norm_L{layer_idx}_{module}', B_norm, actual_step,
            )
            tb_writer.add_scalar(
                f'lora_detail/deltaW_L{layer_idx}_{module}', dW_norm, actual_step,
            )

            if cached_grad_norms:
                for ab in ('A', 'B'):
                    gn = cached_grad_norms.get((layer_idx, module, ab), 0.0)
                    if gn > 0:
                        tb_writer.add_scalar(
                            f'lora_detail/grad_{ab}_L{layer_idx}_{module}',
                            gn, actual_step,
                        )
                        if ab == 'B':
                            layer_grad_sq += gn ** 2

        layer_B = layer_B_sq ** 0.5
        layer_grad = layer_grad_sq ** 0.5
        layer_dW = layer_dW_sq ** 0.5

        tb_writer.add_scalar(f'lora/B_norm_layer{layer_idx}', layer_B, actual_step)
        tb_writer.add_scalar(f'lora/deltaW_layer{layer_idx}', layer_dW, actual_step)
        tb_writer.add_scalar(f'lora/grad_norm_layer{layer_idx}', layer_grad, actual_step)

        total_B_sq += layer_B_sq
        total_dW_sq += layer_dW_sq
        total_grad_sq += layer_grad_sq

    tb_writer.add_scalar('lora/total_B_norm', total_B_sq ** 0.5, actual_step)
    tb_writer.add_scalar('lora/total_deltaW_norm', total_dW_sq ** 0.5, actual_step)
    tb_writer.add_scalar('lora/total_grad_norm', total_grad_sq ** 0.5, actual_step)

    # Layer 20 vs Layer 5 gradient decay ratio
    if cached_grad_norms:
        g20_sq = sum(cached_grad_norms.get((20, m, 'B'), 0.0) ** 2 for m in modules)
        g5_sq = sum(cached_grad_norms.get((5, m, 'B'), 0.0) ** 2 for m in modules)
        g20, g5 = g20_sq ** 0.5, g5_sq ** 0.5
        tb_writer.add_scalar('lora/grad_L20', g20, actual_step)
        tb_writer.add_scalar('lora/grad_L5', g5, actual_step)
        if g5 > 1e-12:
            tb_writer.add_scalar('lora/grad_decay_L20_vs_L5', g20 / g5, actual_step)

    total_dW = total_dW_sq ** 0.5
    if total_dW < 1e-7 and actual_step > 100:
        logger.warning(
            f"[LoRA-DIAG] ⚠️ ||delta_W||_F = {total_dW:.2e} < 1e-7, "
            f"LoRA signal may be too weak!"
        )


# ---------------------------------------------------------------------------
# Internal: heatmap quality diagnostics logged to TensorBoard
# ---------------------------------------------------------------------------

def _heatmap_diagnostic_distribution(
    output: dict,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return sigmoid heatmaps and their spatial diagnostic distribution.

    New HeatmapVLN paths expose the decoder's raw logits.  Using those logits
    directly preserves values outside the numerically invertible sigmoid
    range.  The logit(sigmoid) reconstruction remains only for legacy outputs.
    """
    pred_probabilities = _select_primary_heatmap_slice(
        output["heatmaps"].detach()
    ).unsqueeze(1)
    raw_logits = output.get("heatmap_logits")
    used_raw_logits = raw_logits is not None
    if used_raw_logits:
        spatial_logits = _select_primary_heatmap_slice(
            raw_logits.detach()
        ).unsqueeze(1).float()
        if spatial_logits.shape != pred_probabilities.shape:
            raise ValueError(
                "Heatmap diagnostic raw-logit shape mismatch: "
                f"logits={tuple(spatial_logits.shape)} "
                f"probabilities={tuple(pred_probabilities.shape)}"
            )
    else:
        spatial_logits = torch.logit(
            pred_probabilities.float().clamp(1e-6, 1 - 1e-6)
        )

    batch_size, channels, height, width = spatial_logits.shape
    distribution = torch.softmax(
        spatial_logits.reshape(batch_size, channels, -1),
        dim=-1,
    ).reshape(batch_size, channels, height, width)
    return pred_probabilities, distribution, used_raw_logits


def _log_heatmap_diagnostics(
    output: dict,
    gt_heatmap: torch.Tensor | None,
    batch: dict,
    tb_writer: SummaryWriter,
    actual_step: int,
    cfg: dict,
    logger,
):
    """Log per-step heatmap quality diagnostics to TensorBoard."""
    show_gpu_memory = cfg['log'].get('show_gpu_memory', False)

    if (
        'heatmaps' in output
        and output['heatmaps'] is not None
        and gt_heatmap is not None
    ):
        pred_hm_raw, pred_hm, used_raw_logits = (
            _heatmap_diagnostic_distribution(output)
        )
        gt_hm_for_diag = gt_heatmap
        gt_hm_for_diag = _select_primary_heatmap_slice(gt_hm_for_diag)

        _B, _C, _H, _W = pred_hm_raw.shape

        pred_mean = pred_hm.mean().item()
        pred_max = pred_hm.max().item()
        pred_std = pred_hm.std().item()
        sig_max = pred_hm_raw.max().item()

        tb_writer.add_scalar('diag/pred_heatmap_mean', pred_mean, actual_step)
        tb_writer.add_scalar('diag/pred_heatmap_max', pred_max, actual_step)
        tb_writer.add_scalar('diag/pred_heatmap_std', pred_std, actual_step)
        tb_writer.add_scalar('diag/pred_sigmoid_max', sig_max, actual_step)

        gt_mean = gt_hm_for_diag.mean().item()
        gt_max = gt_hm_for_diag.max().item()

        uniform_baseline = 1.0 / (_H * _W)
        peak_ratio = pred_max / uniform_baseline if uniform_baseline > 0 else 0
        if show_gpu_memory:
            source = "raw_logits" if used_raw_logits else "legacy_logit_sigmoid"
            logger.info(
                f"[DIAG-HM] softmax({source}): max={pred_max:.6f} "
                f"({peak_ratio:.1f}× uniform), sig_max={sig_max:.4f}"
            )
            logger.info(f"[DIAG-HM] gt:      mean={gt_mean:.4f}, max={gt_max:.4f}")
            if peak_ratio < 2.0:
                logger.warning(f"[DIAG-HM] ⚠️ softmax 分布接近均匀！peak_ratio={peak_ratio:.1f}×")

        B, C, H, W = pred_hm.shape
        gt_hm_diag = gt_hm_for_diag.to(pred_hm.device)
        if gt_hm_diag.dim() == 3:
            gt_hm_diag = gt_hm_diag.unsqueeze(1)

        pred_flat = pred_hm.view(B, -1)
        gt_flat = gt_hm_diag.view(B, -1)

        pred_peak_idx = pred_flat.argmax(dim=1)
        gt_peak_idx = gt_flat.argmax(dim=1)

        pred_peak_y = (pred_peak_idx // W).float()
        pred_peak_x = (pred_peak_idx % W).float()
        gt_peak_y = (gt_peak_idx // W).float()
        gt_peak_x = (gt_peak_idx % W).float()

        dx = torch.abs(pred_peak_x - gt_peak_x)
        dx = torch.min(dx, W - dx)
        dy = torch.abs(pred_peak_y - gt_peak_y)

        peak_distance = torch.sqrt(dx**2 + dy**2).mean().item()
        tb_writer.add_scalar('diag/hm_peak_distance', peak_distance, actual_step)
        tb_writer.add_scalar('diag/hm_peak_dx', dx.mean().item(), actual_step)
        tb_writer.add_scalar('diag/hm_peak_dy', dy.mean().item(), actual_step)

        pred_threshold = pred_hm.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] * 0.5
        gt_threshold = gt_hm_diag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] * 0.5

        pred_mask = (pred_hm > pred_threshold).float()
        gt_mask = (gt_hm_diag > gt_threshold).float()

        intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        union = ((pred_mask + gt_mask) > 0).float().sum(dim=(1, 2, 3))

        iou = (intersection / (union + 1e-6)).mean().item()
        tb_writer.add_scalar('diag/hm_peak_iou', iou, actual_step)

        pred_peak_conf = pred_hm.max(dim=-1)[0].max(dim=-1)[0].mean().item()
        gt_peak_conf = gt_hm_diag.max(dim=-1)[0].max(dim=-1)[0].mean().item()

        tb_writer.add_scalar('diag/hm_pred_peak_conf', pred_peak_conf, actual_step)
        tb_writer.add_scalar('diag/hm_gt_peak_conf', gt_peak_conf, actual_step)

        if gt_peak_conf > 0:
            conf_ratio = pred_peak_conf / gt_peak_conf
            tb_writer.add_scalar('diag/hm_peak_conf_ratio', conf_ratio, actual_step)

        try:
            nms_kernel = 5
            pad = nms_kernel // 2
            gt_padded = F.pad(gt_hm_diag, [pad] * 4, mode='replicate')
            local_max = F.max_pool2d(gt_padded, kernel_size=nms_kernel, stride=1, padding=0)
            is_gt_peak = (gt_hm_diag == local_max) & (gt_hm_diag > 0.1)

            pred_padded = F.pad(pred_hm, [pad] * 4, mode='replicate')
            pred_local_max = F.max_pool2d(pred_padded, kernel_size=nms_kernel, stride=1, padding=0)
            is_pred_peak = (pred_hm == pred_local_max) & (pred_hm > pred_hm.max() * 0.2)

            total_gt_peaks = 0
            total_matched = 0
            total_multi_dist = 0.0
            multi_peak_count = 0

            for bi in range(B):
                gt_peaks_bi = is_gt_peak[bi, 0].nonzero(as_tuple=False)
                pred_peaks_bi = is_pred_peak[bi, 0].nonzero(as_tuple=False)

                n_gt = len(gt_peaks_bi)
                total_gt_peaks += n_gt

                if n_gt == 0 or len(pred_peaks_bi) == 0:
                    continue

                for gi in range(min(n_gt, 8)):
                    gt_y, gt_x = gt_peaks_bi[gi].float()

                    pred_y = pred_peaks_bi[:, 0].float()
                    pred_x = pred_peaks_bi[:, 1].float()
                    dx_mp = torch.abs(pred_x - gt_x)
                    dx_mp = torch.min(dx_mp, W - dx_mp)
                    dy_mp = torch.abs(pred_y - gt_y)
                    dists = torch.sqrt(dx_mp**2 + dy_mp**2)

                    min_dist = dists.min().item()
                    total_multi_dist += min_dist
                    multi_peak_count += 1

                    if min_dist < 5.0:
                        total_matched += 1

            if multi_peak_count > 0:
                avg_multi_peak_dist = total_multi_dist / multi_peak_count
                tb_writer.add_scalar('diag/hm_multi_peak_distance', avg_multi_peak_dist, actual_step)

            if total_gt_peaks > 0:
                peak_recall = total_matched / total_gt_peaks
                tb_writer.add_scalar('diag/hm_peak_recall_5px', peak_recall, actual_step)

            tb_writer.add_scalar('diag/hm_avg_gt_peaks', total_gt_peaks / B, actual_step)

        except Exception as e:
            logger.debug(f"Multi-peak eval error (non-critical): {e}")

    if 'visibility' in output and output['visibility'] is not None:
        pred_vis_logits = output['visibility'].detach()
        gt_vis_batch = batch.get('gt_visibility')
        if gt_vis_batch is None and gt_heatmap is not None:
            gt_vis_batch = (gt_heatmap.amax(dim=(-2, -1)) > 0).float()
        if gt_vis_batch is not None:
            gt_vis_flat = gt_vis_batch.to(pred_vis_logits.device).reshape(-1)
            pred_vis_prob = torch.sigmoid(pred_vis_logits.float()).reshape(-1)
            pred_vis_bin = (pred_vis_prob > 0.5).float()
            gt_vis_bin = (gt_vis_flat > 0.5).float()
            total_n = gt_vis_bin.numel()
            if total_n > 0:
                tp = ((pred_vis_bin == 1) & (gt_vis_bin == 1)).sum().item()
                tn = ((pred_vis_bin == 0) & (gt_vis_bin == 0)).sum().item()
                fp = ((pred_vis_bin == 1) & (gt_vis_bin == 0)).sum().item()
                fn = ((pred_vis_bin == 0) & (gt_vis_bin == 1)).sum().item()
                accuracy = (tp + tn) / total_n
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                tnr = tn / max(tn + fp, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                tb_writer.add_scalar('diag/vis_accuracy', accuracy, actual_step)
                tb_writer.add_scalar('diag/vis_precision', precision, actual_step)
                tb_writer.add_scalar('diag/vis_recall', recall, actual_step)
                tb_writer.add_scalar('diag/vis_tnr', tnr, actual_step)
                tb_writer.add_scalar('diag/vis_f1', f1, actual_step)
                tb_writer.add_scalar('diag/vis_pred_mean', pred_vis_prob.mean().item(), actual_step)
                pos_ratio = gt_vis_bin.mean().item()
                tb_writer.add_scalar('diag/vis_gt_pos_ratio', pos_ratio, actual_step)
                if show_gpu_memory:
                    logger.info(
                        f"[DIAG-VIS] acc={accuracy:.3f} prec={precision:.3f} "
                        f"recall={recall:.3f} TNR={tnr:.3f} F1={f1:.3f} "
                        f"(gt_pos={pos_ratio:.2f})"
                    )

    if cfg['log'].get('show_gpu_memory', False):
        tb_writer.add_scalar('diag/gpu_memory_gb', torch.cuda.memory_allocated() / 1024**3, actual_step)
        tb_writer.add_scalar('diag/gpu_memory_reserved_gb', torch.cuda.memory_reserved() / 1024**3, actual_step)
