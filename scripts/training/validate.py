"""
Validation loop.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.pipeline import VLNPipeline
from src.utils.gpu_heatmap import GPUHeatmapComputer

from .utils import _unwrap_model
from .distributed import DistributedContext, _dist_all_reduce_in_place
from .visualization import (
    visualize_heatmap_predictions,
    _should_use_gpu_gt,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def validate(
    model: VLNPipeline,
    val_loader: DataLoader,
    cfg: Dict,
    logger,
    stage_cfg: Dict,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    vis_dir: Optional[Path] = None,
    max_batches: int = None,
    gpu_heatmap_computer: Optional[GPUHeatmapComputer] = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    heatmap_temperature: Optional[float] = None,
    dist_context: Optional[DistributedContext] = None,
) -> Dict[str, float]:
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
    total_stop_loss = 0.0
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
    loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')

    device = dist_context.device

    from src.models.heatmap import HeatmapVLNLoss
    hm_loss_fn = HeatmapVLNLoss(
        lambda_vis=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_vis', 1.0),
        lambda_coord=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_coord', 1.0),
        lambda_kl=cfg.get('loss', {}).get('heatmap_vln', {}).get(
            'lambda_kl',
            cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_pos', 1.0),
        ),
        lambda_peak=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_peak', 1.0),
        lambda_neg=0.0,
        temperature=heatmap_temperature if heatmap_temperature is not None else cfg.get('loss', {}).get('heatmap_vln', {}).get('temperature', 1.0),
        heatmap_size=tuple(cfg['model'].get('heatmap', {}).get('heatmap_size', cfg['data']['init_hm_size'])),
        vis_pos_weight=cfg.get('loss', {}).get('heatmap_vln', {}).get('vis_pos_weight', 1.0),
    ).to(device)

    val_inference_batches = cfg.get('validation', {}).get('val_inference_batches', 10)

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
            B, K, C, H, W = history_frames.shape

            gt_action = batch['action'].to(device)
            action_valid = batch['action_valid'].to(device)
            is_stop = batch['is_stop'].to(device)
            text = batch['text']

            if _should_use_gpu_gt(batch, gpu_heatmap_computer):
                history_poses = batch['history_poses'].to(device)
                current_poses = batch['current_pose'].to(device)
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
                gt_heatmap = batch['heatmap'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                )

            heatmap_loss = torch.tensor(0.0, device=device)
            if train_history and 'visibility' in output and 'heatmaps' in output:
                if gt_heatmap is not None:
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
                    )
                    heatmap_loss = loss_dict['total']
                    total_peak_loss += loss_dict.get('peak_loss', torch.tensor(0.0)).item()
                    total_vis_loss += loss_dict.get('vis_loss', torch.tensor(0.0)).item()
                    total_coord_loss += loss_dict.get('coord_loss', torch.tensor(0.0)).item()

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

            action_loss = torch.tensor(0.0, device=device)
            trajectory_loss = torch.tensor(0.0, device=device)

            if train_action:
                if hasattr(model_module, 'nextdit_action_head') and model_module.nextdit_action_head is not None:
                    if 'trajectory' in batch and 'traj_hidden_states' in output:
                        gt_trajectory = batch['trajectory'].to(device)
                        trajectory_valid = batch['trajectory_valid'].to(device)
                        traj_images = batch.get('traj_images')
                        if traj_images is not None:
                            traj_images = traj_images.to(device)
                        traj_result = model_module.nextdit_action_head.compute_loss(
                            output['traj_hidden_states'],
                            gt_trajectory,
                            traj_images=traj_images,
                            trajectory_valid=trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                elif hasattr(model_module, 'transformer_action_head') and model_module.transformer_action_head is not None:
                    if 'trajectory' in batch:
                        gt_trajectory = batch['trajectory'].to(device)
                        trajectory_valid = batch['trajectory_valid'].to(device)
                        traj_result = model_module.transformer_action_head.compute_loss(
                            output['llm_tokens'],
                            gt_trajectory,
                            trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                elif hasattr(model_module, 'action_head') and model_module.action_head is not None and 'action_cond' in output:
                    action_result = model_module.action_head.compute_loss(
                        output['action_cond'],
                        gt_action.unsqueeze(1),
                        action_valid
                    )
                    action_loss = action_result['loss']

            stop_loss = torch.tensor(0.0, device=device)
            progress_loss = torch.tensor(0.0, device=device)

            if train_action:
                if hasattr(model_module, 'progress_head') and model_module.progress_head is not None:
                    if 'progress' in batch:
                        gt_progress = batch['progress'].to(device)
                        progress_valid = batch.get('trajectory_valid', action_valid).to(device)
                        progress_result = model_module.progress_head(
                            output['llm_tokens'],
                            gt_progress=gt_progress,
                            action_valid=progress_valid,
                            return_loss=True,
                        )
                        progress_loss = progress_result['loss']
                elif hasattr(model_module, 'stop_head') and model_module.stop_head is not None and 'stop_logits' in output:
                    stop_loss = model_module.stop_head.compute_loss(
                        output['stop_logits'],
                        is_stop,
                        action_valid
                    )

            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 1.0)
            progress_weight = loss_cfg.get('progress_weight', 0.5)

            action_total_loss = trajectory_loss if trajectory_loss.item() > 0 else action_loss
            stop_total_loss = progress_loss if progress_loss.item() > 0 else stop_loss

            loss = heatmap_weight * heatmap_loss + trajectory_weight * action_total_loss + progress_weight * stop_total_loss

            total_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_action_loss += action_total_loss.item()
            total_stop_loss += stop_total_loss.item()
            num_batches += 1

            # Reuse current output for inference MSE + visualization
            num_vis_batches = cfg['log'].get('val_vis_batches', 2)
            if num_batches <= val_inference_batches:
                try:
                    vis_output = output
                    if train_history and 'heatmaps' in vis_output:
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

                    if dist_context.is_main and num_batches <= num_vis_batches and vis_dir is not None:
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
                                vis_img = cv2.imread(str(vis_path))
                                if vis_img is not None:
                                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                    vis_img = vis_img.transpose(2, 0, 1)
                                    tb_writer.add_image(f'val/heatmap_viz_batch{num_batches}', vis_img, epoch)

                            logger.info(f"[VAL-VIS] Epoch {epoch}, Batch {num_batches} visualization saved")
                except Exception as e:
                    logger.warning(f"Validation inference/visualization failed: {e}")

            del output, gt_heatmap

    totals = torch.tensor(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_stop_loss,
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
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)

    reduced_num_batches = max(int(totals[5].item()), 1)
    reduced_num_heatmap_mse_batches = int(totals[6].item())
    avg_loss = (totals[0] / reduced_num_batches).item()
    avg_hm = (totals[1] / reduced_num_batches).item()
    avg_act = (totals[2] / reduced_num_batches).item()
    avg_stop = (totals[3] / reduced_num_batches).item()
    avg_hm_mse = (totals[4] / max(reduced_num_heatmap_mse_batches, 1)).item() if reduced_num_heatmap_mse_batches > 0 else 0.0
    avg_peak_loss = (totals[11] / reduced_num_batches).item()
    avg_vis_loss = (totals[12] / reduced_num_batches).item()
    avg_coord_loss = (totals[13] / reduced_num_batches).item()

    r_tp, r_tn, r_fp, r_fn = totals[7].item(), totals[8].item(), totals[9].item(), totals[10].item()
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
        f"coord={avg_coord_loss:.4f}"
    )
    if reduced_num_heatmap_mse_batches > 0:
        logger.info(f"  📊 Heatmap 推理 MSE (采样 {reduced_num_heatmap_mse_batches} batches): {avg_hm_mse:.6f}")

    result = {
        'val_loss': avg_loss,
        'val_heatmap_loss': avg_hm,
        'val_heatmap_mse': avg_hm_mse,
        'val_total_loss': avg_loss,
        'val_hm_peak_loss': avg_peak_loss,
        'val_hm_vis_loss': avg_vis_loss,
        'val_hm_coord_loss': avg_coord_loss,
    }
    result.update(val_vis_metrics)
    return result
