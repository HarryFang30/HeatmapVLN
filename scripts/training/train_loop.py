"""
Core training loop: ``train_one_epoch``.
"""

from __future__ import annotations

import gc
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pipeline import VLNPipeline
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
from .utils import (
    _format_decode_internal_timing,
    _format_qwen_internal_timing,
    _get_trainable_params,
    _mean_timing,
    _unwrap_model,
    build_heatmap_loss_fn,
    make_autocast_context,
)
from .visualization import (
    _select_primary_heatmap_slice,
    _should_use_gpu_gt,
    visualize_heatmap_predictions,
)

logger = logging.getLogger(__name__)


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
) -> dict[str, float]:
    """Train one epoch."""
    dist_context = dist_context or DistributedContext(
        enabled=False,
        device=torch.device(cfg['model'].get('device', 'cuda')),
    )
    model_module = _unwrap_model(model)
    model.train()
    synced_trainable_modules = (
        _get_supported_trainable_sync_modules(model_module, stage_cfg)
        if dist_context.enabled
        else []
    )
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    total_lm_loss = 0.0
    num_batches = 0

    optim_cfg = cfg['optim']
    loss_cfg = cfg['loss']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)

    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))

    device = dist_context.device

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
        if max_batches is not None and i >= max_batches:
            break

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
        if i <= 5 or i % 25 == 0:
            _drop_page_cache()

        if enable_timing:
            loop_start = time.perf_counter()
            timing_stats['data_wait_s'] += max(loop_start - prev_step_end, 0.0)

        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        _B, _K, _C, _H, _W = history_frames.shape

        gt_action = batch['action'].to(device, non_blocking=True)
        action_valid = batch['action_valid'].to(device, non_blocking=True)
        is_stop = batch['is_stop'].to(device, non_blocking=True)
        text = batch['text']

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

            output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame,
                current_views=current_views_batch,
                history_panoramas=history_panoramas_batch,
                panoramic_inputs=panoramic_inputs_batch,
                panoramic_num_histories=panoramic_num_histories,
                panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                history_rel_poses=history_rel_poses,
                return_heatmaps=train_history or train_future,
                return_actions=train_action,
                return_lm_loss=train_lm,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,
                action_valid=action_valid if train_action else None,
                gt_stop=is_stop if train_action else None,
                gt_history_heatmap=gt_heatmap if train_history else None,
                gt_future_heatmap=gt_heatmap if train_future else None,
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
                loss_dict = hm_loss_fn(
                    output['visibility'],
                    output['heatmaps'],
                    gt_vis=gt_vis,
                    gt_heatmaps=gt_heatmap.to(device, non_blocking=True),
                    history_mask=hm_history_mask,
                )
                heatmap_loss = loss_dict['total']

            trajectory_loss = torch.tensor(0.0, device=device)

            if train_action:
                if hasattr(model_module, 'nextdit_action_head') and model_module.nextdit_action_head is not None:
                    if 'trajectory' in batch and 'traj_hidden_states' in output:
                        gt_trajectory = batch['trajectory'].to(device, non_blocking=True)
                        trajectory_valid = batch['trajectory_valid'].to(device, non_blocking=True)
                        traj_images = batch.get('traj_images')
                        if traj_images is not None:
                            traj_images = traj_images.to(device, non_blocking=True)
                        traj_result = model_module.nextdit_action_head.compute_loss(
                            output['traj_hidden_states'],
                            gt_trajectory,
                            traj_images=traj_images,
                            trajectory_valid=trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']

            lm_loss = torch.tensor(0.0, device=device)
            if train_lm:
                if 'lm_loss' not in output or output['lm_loss'] is None:
                    raise RuntimeError(
                        "train_lm=True but model output has no lm_loss. "
                        "Check PanoramicTokenizedCollator labels and Qwen forward wiring."
                    )
                lm_loss = output['lm_loss']

            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 0.0)
            lm_weight = loss_cfg.get('lm_weight', stage_cfg.get('lm_weight', 1.0))

            loss = (
                heatmap_weight * heatmap_loss
                + trajectory_weight * trajectory_loss
                + lm_weight * lm_loss
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

            log_interval = cfg['log'].get('log_interval', 10)
            show_gpu_memory = cfg['log'].get('show_gpu_memory', False)
            if global_step % log_interval == 0 or global_step <= 3:
                all_lrs = scheduler.get_last_lr()
                lr_strs = []
                for gi, lr_val in enumerate(all_lrs):
                    gname = optimizer.param_groups[gi].get('name', f'g{gi}')
                    lr_strs.append(f"{gname}={lr_val:.2e}")
                lr_display = ", ".join(lr_strs)
                gpu_mem_str = f" | GPU: {torch.cuda.memory_allocated() / 1024**3:.1f}GB" if show_gpu_memory else ""
                traj_str = f", traj: {trajectory_loss.item():.4f}" if trajectory_loss.item() > 0 else ""
                lm_str = f", lm: {lm_loss.item():.4f}" if train_lm else ""
                logger.info(
                    f"[{stage_name}] "
                    f"Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item()*grad_accum_steps:.4f} "
                    f"(hm: {heatmap_loss.item():.4f}{traj_str}{lm_str}) | "
                    f"LR: [{lr_display}]"
                    + gpu_mem_str
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
                    neg_str = f" neg={loss_dict.get('neg_loss', 0):.4f}" if loss_dict.get('neg_loss', 0) > 0 else ""
                    logger.info(
                        f"  [HM] peak={loss_dict.get('peak_loss', 0):.4f} "
                        f"vis={loss_dict.get('vis_loss', 0):.4f} "
                        f"coord={loss_dict.get('coord_loss', 0):.4f}"
                        f"{neg_str}"
                    )
                if metrics_jsonl_path is not None:
                    step_record = {
                        "record_type": "train_step",
                        "stage": stage_name,
                        "epoch": epoch,
                        "batch": i + 1,
                        "global_step": global_step,
                        "loss": loss.item() * grad_accum_steps,
                        "heatmap_loss": heatmap_loss.item(),
                        "trajectory_loss": trajectory_loss.item(),
                        "lm_loss": lm_loss.item(),
                        "lrs": {
                            optimizer.param_groups[gi].get("name", f"g{gi}"): lr_val
                            for gi, lr_val in enumerate(all_lrs)
                        },
                    }
                    if isinstance(loss_dict, dict):
                        for k in ('peak_loss', 'vis_loss', 'coord_loss', 'neg_loss'):
                            if k in loss_dict:
                                step_record[f'hm_{k}'] = loss_dict[k].item()
                    if show_gpu_memory:
                        step_record["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
                    _append_jsonl(metrics_jsonl_path, step_record)

            if tb_writer is not None:
                actual_step = global_step_offset + global_step
                tb_writer.add_scalar('train/loss', loss.item()*grad_accum_steps, actual_step)
                tb_writer.add_scalar('train/heatmap_loss', heatmap_loss.item(), actual_step)
                if trajectory_loss.item() > 0:
                    tb_writer.add_scalar('train/trajectory_loss', trajectory_loss.item(), actual_step)
                if train_lm:
                    tb_writer.add_scalar('train/lm_loss', lm_loss.item(), actual_step)
                if isinstance(loss_dict, dict):
                    for k in ('vis_loss', 'coord_loss', 'peak_loss', 'neg_loss'):
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
        if tb_writer is not None and global_step % vis_interval == 0 and global_step > 0:
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

        _iter_loss = loss.item() * grad_accum_steps
        _iter_hm = heatmap_loss.item()
        _iter_traj = trajectory_loss.item()
        _iter_lm = lm_loss.item()

        total_loss += _iter_loss
        total_heatmap_loss += _iter_hm
        total_action_loss += _iter_traj
        total_lm_loss += _iter_lm
        num_batches += 1

        del output, loss, heatmap_loss, gt_heatmap
        del trajectory_loss, lm_loss
        loss_dict = None
        del video_frames
        del current_views_batch, history_panoramas_batch
        del panoramic_inputs_batch, panoramic_num_histories, panoramic_text_anchor_positions
        del history_frames, current_frame, gt_action, action_valid, is_stop, text
        del batch

        pbar.set_postfix({
            'loss': f"{_iter_loss:.4f}",
            'hm': f"{total_heatmap_loss / num_batches:.4f}",
            'traj': f"{_iter_traj:.4f}",
            'lm': f"{_iter_lm:.4f}",
        })

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
            _drop_page_cache()

            if num_batches % 200 == 0:
                torch.cuda.empty_cache()
                if tb_writer is not None:
                    tb_writer.flush()

        if (
            ckpt_manager is not None
            and mid_epoch_save_every > 0
            and num_batches > 0
            and num_batches % mid_epoch_save_every == 0
            and dist_context.is_main
        ):
            model_module_for_save = _unwrap_model(model)
            mid_metrics = {
                'total_loss': total_loss / num_batches,
                'heatmap_loss': total_heatmap_loss / num_batches,
                'lm_loss': total_lm_loss / num_batches,
            }
            if ema is not None:
                with ema.apply():
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
                        batch=num_batches,
                    )
            else:
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
                    batch=num_batches,
                )
            logger.info(
                f"  Mid-epoch checkpoint saved at batch {num_batches} "
                f"(loss={mid_metrics['total_loss']:.4f})"
            )

        prev_step_end = time.perf_counter()

    # Handle remaining gradients
    remaining = valid_batch_count % grad_accum_steps
    if remaining > 0:
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

    totals = torch.tensor(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_lm_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)

    reduced_num_batches = max(int(totals[4].item()), 1)
    return {
        'total_loss': (totals[0] / reduced_num_batches).item(),
        'heatmap_loss': (totals[1] / reduced_num_batches).item(),
        'trajectory_loss': (totals[2] / reduced_num_batches).item(),
        'lm_loss': (totals[3] / reduced_num_batches).item(),
        'optimizer_steps': global_step,
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

    grad_norms: dict[tuple[int, str, str], float] = {}
    for name, param in qwen_model.named_parameters():
        m = _LORA_PARAM_RE.search(name)
        if m and param.grad is not None:
            key = (int(m.group(1)), m.group(2), m.group(3))
            grad_norms[key] = param.grad.float().norm().item()

    return grad_norms or None


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

def _log_heatmap_diagnostics(
    output: dict,
    gt_heatmap: torch.Tensor,
    batch: dict,
    tb_writer: SummaryWriter,
    actual_step: int,
    cfg: dict,
    logger,
):
    """Log per-step heatmap quality diagnostics to TensorBoard."""
    show_gpu_memory = cfg['log'].get('show_gpu_memory', False)

    if 'heatmaps' in output and output['heatmaps'] is not None:
        pred_hm_raw = output['heatmaps'].detach()

        pred_hm_raw = _select_primary_heatmap_slice(pred_hm_raw).unsqueeze(1)
        gt_hm_for_diag = gt_heatmap
        gt_hm_for_diag = _select_primary_heatmap_slice(gt_hm_for_diag)

        _B, _C, _H, _W = pred_hm_raw.shape
        _logits = torch.logit(pred_hm_raw.float().clamp(1e-6, 1 - 1e-6))
        pred_hm = torch.softmax(_logits.reshape(_B, _C, -1), dim=-1).reshape(_B, _C, _H, _W)

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
            logger.info(f"[DIAG-HM] softmax: max={pred_max:.6f} ({peak_ratio:.1f}× uniform), sig_max={sig_max:.4f}")
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
