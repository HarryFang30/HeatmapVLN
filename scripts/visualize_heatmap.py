#!/usr/bin/env python3
"""
Heatmap 推理可视化
==================

加载训练好的模型权重，使用与训练完全相同的数据加载流程，
进行完整扩散推理，生成 predicted heatmap vs GT heatmap 的对比可视化。

用法:
    python scripts/visualize_heatmap.py \
        --checkpoint /root/autodl-tmp/heatmap_training_outputs/run_20260209_025529/ckpts/best.pth \
        --num-samples 10 \
        --output-dir ./vis_heatmap_12epoch
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.utils.gpu_heatmap import GPUHeatmapComputer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("visualize")


# ==================== 与 train.py 完全一致的 build_model ====================
def build_model(cfg: Dict) -> VLNPipeline:
    """构建 VLN Pipeline（复制自 train.py）"""
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    progress_cfg = model_cfg.get('progress_head', {})

    action_head_type = action_cfg.get('type', 'transformer')
    legacy_action_cfg = action_cfg.get('legacy', {})
    transformer_action_cfg = action_cfg.get('transformer', {})

    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 2048),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'flash_attention_2'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        # 推理时关闭 packing
        enable_packing=False,
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        device=model_cfg.get('device', 'cuda'),
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', False),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        image_size=cfg['data']['image_size'][0],
        heatmap_use_image_encoder=heatmap_cfg.get('use_image_encoder', True),
        heatmap_pool_method=heatmap_cfg.get('pool_method', 'attention'),
        heatmap_pool_num_heads=heatmap_cfg.get('pool_num_heads', 4),
        heatmap_use_circular_padding=heatmap_cfg.get('use_circular_padding', False),
        heatmap_dropout=heatmap_cfg.get('dropout', 0.1),
        heatmap_block_out_channels=tuple(heatmap_cfg.get('block_out_channels', [64, 128, 256])),
        heatmap_layers_per_block=heatmap_cfg.get('layers_per_block', 2),
        heatmap_attention_levels=tuple(heatmap_cfg.get('attention_levels', [2])),
        heatmap_num_train_timesteps=heatmap_cfg.get('num_train_timesteps', 100),
        heatmap_cfg_drop_prob=heatmap_cfg.get('cfg_drop_prob', 0.1),
        heatmap_cfg_scale=heatmap_cfg.get('cfg_scale', 3.0),
        heatmap_use_sequence_conditioning=heatmap_cfg.get('use_sequence_conditioning', False),
        heatmap_seq_cross_attn_heads=heatmap_cfg.get('seq_cross_attn_heads', 8),
        heatmap_seq_cross_attn_head_dim=heatmap_cfg.get('seq_cross_attn_head_dim', 64),
        heatmap_use_spatial_injection=heatmap_cfg.get('use_spatial_injection', False),
        heatmap_image_encoder_use_pretrained=heatmap_cfg.get('image_encoder_use_pretrained', False),
        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        action_head_type=action_head_type,
        enable_action_head=action_cfg.get('enable', False),  # 推理只看热力图
        enable_stop_head=False,
        enable_progress_head=False,
        verbose=False,
    )

    return VLNPipeline(config)


# ==================== 与 train.py 完全一致的 collate_fn ====================
def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """滑动窗口数据集的 collate 函数（复制自 train.py）"""
    max_K = max(s['history_frames'].shape[0] for s in batch)

    history_frames_padded = []
    history_mask = []

    for s in batch:
        frames = s['history_frames']
        K = frames.shape[0]
        if K < max_K:
            pad_size = max_K - K
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([torch.ones(K), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            mask = torch.ones(K)
        history_frames_padded.append(frames_padded)
        history_mask.append(mask)

    history_frames = torch.stack(history_frames_padded, dim=0)
    history_mask = torch.stack(history_mask, dim=0)
    current_frame = torch.stack([s['current_frame'] for s in batch], dim=0)
    heatmap = torch.stack([s['heatmap'] for s in batch], dim=0)
    action = torch.stack([s['action'] for s in batch], dim=0)
    action_valid = torch.tensor([s['action_valid'] for s in batch])
    is_stop = torch.tensor([s.get('is_stop', 0.0) for s in batch])
    text = [s['text'] for s in batch]

    result = {
        'history_frames': history_frames,
        'history_mask': history_mask,
        'current_frame': current_frame,
        'heatmap': heatmap,
        'action': action,
        'action_valid': action_valid,
        'is_stop': is_stop,
        'text': text,
    }

    # GPU 热力图所需字段
    if 'history_poses' in batch[0]:
        result['history_poses'] = torch.stack([s['history_poses'] for s in batch], dim=0)
        result['current_pose'] = torch.stack([s['current_pose'] for s in batch], dim=0)
        result['has_depth'] = batch[0].get('has_depth', False)
        if result['has_depth']:
            result['current_depth'] = torch.stack([s['current_depth'] for s in batch], dim=0)
        result['has_intrinsics'] = batch[0].get('has_intrinsics', False)
        if result['has_intrinsics']:
            result['intrinsics'] = torch.stack([s['intrinsics'] for s in batch], dim=0)

    return result


# ==================== 可视化函数 ====================
def visualize_sample(
    sample_idx: int,
    current_frame: np.ndarray,
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    instruction: str,
    output_path: str,
    metrics: Dict = None,
):
    """生成对比可视化：当前帧 | GT 热力图 | 预测热力图 (与 train.py 布局类似)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 0: 当前帧 | GT | 预测
    rgb = np.clip(current_frame, 0, 1)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Current Frame", fontsize=13)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_heatmap, cmap='inferno', vmin=0, vmax=1)
    axes[0, 1].set_title(f"GT Heatmap (max={gt_heatmap.max():.3f})", fontsize=13)
    axes[0, 1].axis('off')

    pred_max = max(pred_heatmap.max(), 0.001)
    axes[0, 2].imshow(pred_heatmap, cmap='inferno', vmin=0, vmax=max(pred_max, 0.1))
    axes[0, 2].set_title(f"Pred Heatmap (max={pred_heatmap.max():.3f})", fontsize=13)
    axes[0, 2].axis('off')

    # Row 1: 叠加对比 | 差异图 | 指标面板
    # 叠加: 热力图 overlay 在帧上
    H, W = rgb.shape[:2]
    hm_h, hm_w = pred_heatmap.shape
    pred_up = cv2.resize(pred_heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    gt_up = cv2.resize(gt_heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    # 热力图叠加
    overlay = rgb.copy()
    pred_color = plt.cm.inferno(pred_up / max(pred_up.max(), 0.01))[:, :, :3]
    alpha = np.clip(pred_up * 3, 0, 0.7)[:, :, None]
    overlay = overlay * (1 - alpha) + pred_color * alpha
    overlay = np.clip(overlay, 0, 1)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Pred Overlay on Frame", fontsize=13)
    axes[1, 0].axis('off')

    # 差异图
    diff = np.abs(gt_heatmap - pred_heatmap)
    im_diff = axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=max(diff.max(), 0.01))
    axes[1, 1].set_title(f"|GT - Pred| (max={diff.max():.3f})", fontsize=13)
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046)

    # 在 GT 和 Pred 上标注峰值位置
    if gt_heatmap.max() > 0.01:
        gt_peak_y, gt_peak_x = np.unravel_index(gt_heatmap.argmax(), gt_heatmap.shape)
        axes[0, 1].plot(gt_peak_x, gt_peak_y, 'g+', markersize=15, markeredgewidth=2)
    if pred_heatmap.max() > 0.01:
        pred_peak_y, pred_peak_x = np.unravel_index(pred_heatmap.argmax(), pred_heatmap.shape)
        axes[0, 2].plot(pred_peak_x, pred_peak_y, 'r+', markersize=15, markeredgewidth=2)

    # 指标面板
    info_lines = [f"Instruction: {instruction[:100]}"]
    if metrics:
        info_lines.append("")
        for k, v in metrics.items():
            info_lines.append(f"{k}: {v}")
    info_text = "\n".join(info_lines)
    axes[1, 2].text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top',
                    transform=axes[1, 2].transAxes, wrap=True, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title("Metrics", fontsize=13)

    fig.suptitle(f"Sample {sample_idx}", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Heatmap 推理可视化")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--num-samples', type=int, default=10, help='可视化样本数')
    parser.add_argument('--output-dir', type=str, default='./vis_heatmap_output', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inference-steps', type=int, default=None,
                        help='Override num_inference_steps (default: use checkpoint config)')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 加载 checkpoint ====================
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    epoch = ckpt.get('epoch', '?')
    best_val = ckpt.get('best_val_loss', '?')
    logger.info(f"  Epoch: {epoch}, Best val loss: {best_val}")

    # ==================== 构建数据集（与 train.py 一致）====================
    logger.info("Loading dataset...")
    sw_cfg = cfg['data']['sliding_window']
    defer_heatmap_to_gpu = sw_cfg.get('defer_heatmap_to_gpu', False)

    val_dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split=cfg['data'].get('val_split', 'val'),
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
        sample_stride=sw_cfg.get('sample_stride', 2),
        clip_level_sampling=sw_cfg.get('clip_level_sampling', True),
        samples_per_clip=sw_cfg.get('val_samples_per_clip', 2),
        defer_heatmap_to_gpu=defer_heatmap_to_gpu,
    )
    if hasattr(val_dataset, 'set_epoch'):
        val_dataset.set_epoch(0)

    logger.info(f"  Val dataset: {len(val_dataset)} samples")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ==================== GPU 热力图计算器 ====================
    gpu_heatmap_computer = None
    if defer_heatmap_to_gpu:
        hm_size = tuple(cfg['data']['init_hm_size'])
        gpu_heatmap_computer = GPUHeatmapComputer(
            hm_size=hm_size,
            img_size=(640, 480),
            device=device,
        )
        logger.info(f"  GPU heatmap computer enabled (hm_size={hm_size})")

    # ==================== 构建模型 ====================
    # Override inference steps if specified
    if args.inference_steps is not None:
        cfg['model']['heatmap_head']['num_inference_steps'] = args.inference_steps
        logger.info(f"  Override num_inference_steps = {args.inference_steps}")

    logger.info("Building model...")
    model = build_model(cfg)

    # 加载权重
    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', {}))
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Loaded: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        logger.warning("  No trainable weights in checkpoint!")

    model = model.to(device)
    model.eval()
    logger.info("Model ready.")

    # ==================== 推理 + 可视化 ====================
    logger.info(f"Running inference on {args.num_samples} samples...")
    all_peak_dists = []
    all_ious = []
    count = 0

    for i, batch in enumerate(val_loader):
        if count >= args.num_samples:
            break

        # 和 train.py 验证循环完全一样的流程
        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        text = batch['text']
        B, K, C, H, W = history_frames.shape

        # GT 热力图
        if gpu_heatmap_computer is not None and 'history_poses' in batch:
            history_poses = batch['history_poses'].to(device)
            current_poses = batch['current_pose'].to(device)
            has_depth = batch.get('has_depth', False)
            current_depths = batch['current_depth'].to(device) if has_depth else None
            has_intrinsics = batch.get('has_intrinsics', False)
            intrinsics = batch['intrinsics'].to(device) if has_intrinsics else None
            gt_heatmap = gpu_heatmap_computer.compute_batch(
                history_poses=history_poses,
                current_poses=current_poses,
                current_depths=current_depths,
                intrinsics=intrinsics,
            )  # [B, Hm, Wm]
        else:
            gt_heatmap = batch['heatmap'].to(device)  # [B, Hm, Wm]

        # 跳过空 GT
        if gt_heatmap[0].max() < 0.01:
            continue

        # 完整扩散推理（与 train.py 验证可视化一致）
        video_frames = torch.cat([history_frames, current_frame.unsqueeze(1)], dim=1)
        instruction_text = list(text) if text and len(text) > 0 else None

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            vis_output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame.to(device),
                return_heatmaps=True,
                return_actions=False,
            )

        # 提取预测热力图
        if 'history_heatmaps' not in vis_output or vis_output['history_heatmaps'] is None:
            logger.warning(f"  Sample {i}: no heatmap output")
            continue

        pred_hm = vis_output['history_heatmaps'][:, -1, :, :]  # [B, Hm, Wm]
        # 如果尺寸不匹配，resize
        if pred_hm.shape[-2:] != gt_heatmap.shape[-2:]:
            pred_hm = F.interpolate(
                pred_hm.unsqueeze(1), size=gt_heatmap.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(1)

        # 转 numpy
        gt_hm_np = gt_heatmap[0].float().cpu().numpy()
        pred_hm_np = pred_hm[0].detach().float().cpu().numpy()
        pred_hm_np = np.clip(pred_hm_np, 0, 1)
        current_frame_np = current_frame[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC

        # 计算指标
        gt_peak = np.unravel_index(gt_hm_np.argmax(), gt_hm_np.shape)
        if pred_hm_np.max() > 0.01:
            pred_peak = np.unravel_index(pred_hm_np.argmax(), pred_hm_np.shape)
            peak_dist = np.sqrt((gt_peak[0] - pred_peak[0])**2 + (gt_peak[1] - pred_peak[1])**2)
        else:
            pred_peak = None
            peak_dist = float('inf')

        # IoU（threshold=0.5*max）
        gt_mask = gt_hm_np > 0.5 * gt_hm_np.max()
        pred_mask = pred_hm_np > 0.5 * max(pred_hm_np.max(), 0.01)
        intersection = (gt_mask & pred_mask).sum()
        union = (gt_mask | pred_mask).sum()
        iou = intersection / max(union, 1)

        mse = np.mean((gt_hm_np - pred_hm_np) ** 2)

        all_peak_dists.append(peak_dist)
        all_ious.append(iou)

        metrics = {
            'peak_distance': f"{peak_dist:.1f}px",
            'peak_iou': f"{iou:.3f}",
            'mse': f"{mse:.4f}",
            'pred_max': f"{pred_hm_np.max():.3f}",
            'gt_max': f"{gt_hm_np.max():.3f}",
            'gt_peak': f"({gt_peak[1]}, {gt_peak[0]})",
            'pred_peak': f"({pred_peak[1]}, {pred_peak[0]})" if pred_peak else "N/A",
        }

        instr = text[0] if text else "N/A"
        logger.info(f"  [{count+1:>2}] peak_dist={peak_dist:>5.1f}px, iou={iou:.3f}, "
                     f"pred_max={pred_hm_np.max():.3f}, gt_max={gt_hm_np.max():.3f}")

        # 保存可视化
        out_path = output_dir / f"sample_{count:03d}.png"
        visualize_sample(
            count + 1, current_frame_np,
            pred_hm_np, gt_hm_np,
            instr, str(out_path),
            metrics=metrics,
        )

        count += 1
        torch.cuda.empty_cache()

    # ==================== 汇总 ====================
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUMMARY (Epoch {epoch}, {count} samples)")
    logger.info("=" * 60)

    finite_dists = [d for d in all_peak_dists if d != float('inf')]
    if finite_dists:
        logger.info(f"Peak Distance ({cfg['data']['init_hm_size'][0]}x{cfg['data']['init_hm_size'][1]}):")
        logger.info(f"  Mean:   {np.mean(finite_dists):.1f} px")
        logger.info(f"  Median: {np.median(finite_dists):.1f} px")
        logger.info(f"  Min:    {np.min(finite_dists):.1f} px")
        logger.info(f"  Max:    {np.max(finite_dists):.1f} px")
        logger.info(f"  < 5px:  {sum(1 for d in finite_dists if d < 5)}/{len(finite_dists)}")
        logger.info(f"  < 10px: {sum(1 for d in finite_dists if d < 10)}/{len(finite_dists)}")

    if all_ious:
        logger.info(f"Peak IoU:")
        logger.info(f"  Mean:   {np.mean(all_ious):.3f}")
        logger.info(f"  Median: {np.median(all_ious):.3f}")

    inf_count = sum(1 for d in all_peak_dists if d == float('inf'))
    if inf_count:
        logger.info(f"Empty predictions: {inf_count}/{len(all_peak_dists)}")

    logger.info(f"\nOutput: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
