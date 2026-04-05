#!/usr/bin/env python3
"""
VLN Pipeline 评估脚本
======================

使用 Qwen3.5 评估视觉语言导航模型。

支持评估：
- 历史热力图头 (History Heatmap)
- 轨迹预测头 (Trajectory - NextDiTActionHead)
- 进度预测头 (Progress)
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.data.vln_sliding_window_dataset import VLNTrajectoryDataset

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def flatten_heatmap_slices(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() <= 3:
        return heatmaps
    return heatmaps.reshape(-1, heatmaps.shape[-2], heatmaps.shape[-1])


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for VLNTrajectoryDataset (non-packing mode)."""
    result = {
        'history_frames': torch.stack([s['history_frames'] for s in batch], dim=0),
        'current_frame': torch.stack([s['current_frame'] for s in batch], dim=0),
        'heatmap': torch.stack([s['heatmap'] for s in batch], dim=0),
        'action': torch.stack([s['action'] for s in batch], dim=0),
        'action_valid': torch.tensor([s['action_valid'] for s in batch]),
        'is_stop': torch.tensor([s.get('is_stop', 0.0) for s in batch]),
        'text': [s['text'] for s in batch],
    }
    
    # Trajectory data
    if 'trajectory' in batch[0]:
        result['trajectory'] = torch.stack([s['trajectory'] for s in batch], dim=0)
        result['trajectory_valid'] = torch.tensor([s.get('trajectory_valid', 0.0) for s in batch])
        result['progress'] = torch.tensor([s.get('progress', 0.0) for s in batch])
    if 'current_views' in batch[0]:
        result['current_views'] = torch.stack([s['current_views'] for s in batch], dim=0)
    if 'history_panoramas' in batch[0]:
        result['history_panoramas'] = torch.stack([s['history_panoramas'] for s in batch], dim=0)
    
    return result


def build_model(cfg: Dict, device: str = 'cuda:0') -> VLNPipeline:
    """Build VLN pipeline for evaluation (与 train.py 保持一致)."""
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
    nextdit_cfg = action_cfg.get('nextdit', {})
    
    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3.5'),
        llm_backbone_type=llm_cfg.get('backbone_type', 'auto'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 4096),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        max_video_frames=llm_cfg.get('max_video_frames', -1),
        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 8192),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        internnav_system1_path=nextdit_cfg.get('internnav_system1_path', ''),
        device=device,
        
        enable_heatmap=heatmap_cfg.get('enable', True),
        heatmap_c_vit=heatmap_cfg.get('c_vit', 1152),
        heatmap_c_llm=heatmap_cfg.get('c_llm', 4096),
        heatmap_c_fused=heatmap_cfg.get('c_fused', 256),
        heatmap_vit_layer_indices=heatmap_cfg.get('vit_layer_indices', [6, 12, 18, 24]),
        heatmap_llm_layer_indices=heatmap_cfg.get('llm_layer_indices', [7, 15, 23]),
        heatmap_size=tuple(heatmap_cfg.get('heatmap_size', data_cfg['init_hm_size'])),
        image_size=heatmap_cfg.get('image_size', data_cfg['image_size'][0]),
        heatmap_trajectory_config=heatmap_cfg.get('trajectory', None),
        
        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_layer_indices=llm_cfg.get('lora_layer_indices', None),
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        
        enable_action_head=action_cfg.get('enable', True),
        nextdit_enabled=nextdit_cfg.get('enabled', False),
        nextdit_vlm_hidden_dim=nextdit_cfg.get('vlm_hidden_dim', 4096),
        nextdit_latent_emb_size=nextdit_cfg.get('latent_emb_size', 768),
        nextdit_n_query=nextdit_cfg.get('n_query', 4),
        nextdit_dit_dim=nextdit_cfg.get('dit_dim', 384),
        nextdit_dit_layers=nextdit_cfg.get('dit_layers', 12),
        nextdit_dit_heads=nextdit_cfg.get('dit_heads', 6),
        nextdit_dit_kv_heads=nextdit_cfg.get('dit_kv_heads', 6),
        nextdit_dit_ffn_dim_multiplier=nextdit_cfg.get('dit_ffn_dim_multiplier', None),
        nextdit_predict_steps=nextdit_cfg.get('predict_steps', 32),
        nextdit_action_dim=nextdit_cfg.get('action_dim', 3),
        nextdit_num_inference_steps=nextdit_cfg.get('num_inference_steps', 10),
        nextdit_guidance_scale=nextdit_cfg.get('guidance_scale', 1.0),
        nextdit_num_sample_trajs=nextdit_cfg.get('num_sample_trajs', 32),
        nextdit_dav2_ckpt_path=nextdit_cfg.get('dav2_ckpt_path', ''),
        nextdit_enable_gradient_checkpointing=nextdit_cfg.get('enable_gradient_checkpointing', True),
        
        verbose=False,
    )
    
    return VLNPipeline(config)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint with partial loading support."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('trainable_state_dict', ckpt))
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Epoch: {ckpt.get('epoch', 'N/A')}  Stage: {ckpt.get('stage_name', 'N/A')}")
    if missing_keys:
        logger.info(f"  Missing keys (using pretrained): {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"  Unexpected keys: {len(unexpected_keys)}")

    return model


def build_dataloader(
    cfg: Dict,
    split: str = 'val',
) -> DataLoader:
    """Build dataloader using VLNTrajectoryDataset."""
    sw_cfg = cfg['data']['sliding_window']
    traj_cfg = cfg['data'].get('trajectory', {})
    
    # Use trajectory dataset
    base_dataset = VLNTrajectoryDataset(
        root=cfg['data']['root'],
        split=split,
        min_history=sw_cfg['min_history'],
        num_history_sample=traj_cfg.get('num_history_sample', sw_cfg['num_history_sample']),
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        sample_stride=sw_cfg.get('sample_stride', 5),
        clip_level_sampling=False,  # Disable for evaluation
        enable_augmentation=False,  # No augmentation for evaluation
    )
    
    dataset = base_dataset
    actual_collate_fn = collate_fn
    num_workers = 4
    
    return DataLoader(
        dataset,
        batch_size=cfg['optim'].get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=actual_collate_fn,
    )


def compute_spatial_metrics(pred_hm: np.ndarray, gt_hm: np.ndarray) -> Dict[str, float]:
    """Compute spatial accuracy metrics for heatmaps."""
    # Normalize
    if pred_hm.max() > 0:
        pred_hm = pred_hm / pred_hm.max()
    if gt_hm.max() > 0:
        gt_hm = gt_hm / gt_hm.max()
    
    # Peak location error
    pred_peak = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
    gt_peak = np.unravel_index(np.argmax(gt_hm), gt_hm.shape)
    peak_error = np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    # IoU at threshold
    thresh = 0.3
    pred_mask = pred_hm > thresh
    gt_mask = gt_hm > thresh
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    iou = intersection / (union + 1e-6)

    # Cosine similarity
    cos_sim = np.dot(pred_hm.flatten(), gt_hm.flatten()) / (
        np.linalg.norm(pred_hm) * np.linalg.norm(gt_hm) + 1e-6
    )

    # MAE
    mae = np.abs(pred_hm - gt_hm).mean()

    return {
        'peak_error': float(peak_error),
        'iou': float(iou),
        'cosine_sim': float(cos_sim),
        'mae': float(mae)
    }


def compute_trajectory_metrics(pred_traj: np.ndarray, gt_traj: np.ndarray, valid: float = 1.0) -> Dict[str, float]:
    """Compute trajectory prediction metrics.
    
    Args:
        pred_traj: Predicted trajectory [T, 3] (dx, dy, dyaw)
        gt_traj: Ground truth trajectory [T, 3]
        valid: Validity flag
        
    Returns:
        Dictionary with ADE, FDE metrics
    """
    if valid < 0.5:
        return {'ade': 0.0, 'fde': 0.0, 'valid': False}
    
    # Only use position (dx, dy), ignore yaw
    pred_pos = pred_traj[:, :2]  # [T, 2]
    gt_pos = gt_traj[:, :2]  # [T, 2]
    
    # Compute cumulative positions
    pred_cum = np.cumsum(pred_pos, axis=0)
    gt_cum = np.cumsum(gt_pos, axis=0)
    
    # Average Displacement Error (ADE)
    displacements = np.sqrt(np.sum((pred_cum - gt_cum) ** 2, axis=1))
    ade = displacements.mean()
    
    # Final Displacement Error (FDE)
    fde = displacements[-1]
    
    return {
        'ade': float(ade),
        'fde': float(fde),
        'valid': True
    }


def compute_progress_metrics(pred_progress: np.ndarray, gt_progress: np.ndarray) -> Dict[str, float]:
    """Compute progress prediction metrics."""
    mae = np.abs(pred_progress - gt_progress).mean()
    
    # Accuracy at threshold
    thresh = 0.1
    accuracy = (np.abs(pred_progress - gt_progress) < thresh).mean()
    
    # Boundary accuracy (progress near 0 or 1)
    boundary_mask = (gt_progress < 0.1) | (gt_progress > 0.9)
    if boundary_mask.sum() > 0:
        boundary_acc = (np.abs(pred_progress[boundary_mask] - gt_progress[boundary_mask]) < thresh).mean()
    else:
        boundary_acc = 0.0
    
    return {
        'progress_mae': float(mae),
        'progress_accuracy': float(accuracy),
        'progress_boundary_acc': float(boundary_acc),
    }


@torch.no_grad()
def evaluate(
    model: VLNPipeline,
    dataloader: DataLoader,
    cfg: Dict,
    device: torch.device,
    save_dir: Path = None,
    num_vis: int = 20,
    eval_heatmap: bool = True,
    eval_trajectory: bool = True,
    eval_progress: bool = True,
    args=None
) -> Dict[str, float]:
    """Run evaluation."""
    model.eval()

    totals = {
        'hm_peak_error': 0.0,
        'hm_iou': 0.0,
        'hm_cosine_sim': 0.0,
        'hm_mae': 0.0,
        'traj_ade': 0.0,
        'traj_fde': 0.0,
        'progress_mae': 0.0,
        'progress_accuracy': 0.0,
        'progress_boundary_acc': 0.0,
    }
    counts = {'hm': 0, 'traj': 0, 'progress': 0}

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args and args.max_samples is not None and idx >= args.max_samples:
            logger.info(f"Reached max_samples={args.max_samples}, stopping")
            break

        gt_heatmap = batch['heatmap'].to(device)
        current_frame = batch['current_frame']
        B = current_frame.shape[0]
        
        history_frames = batch['history_frames']
        video_frames = torch.cat([
            history_frames,
            history_frames[:, -1:]
        ], dim=1).to(device)
        text = batch['text']
        instruction = list(text) if text and len(text) > 0 else None
        current_views = batch.get('current_views')
        history_panoramas = batch.get('history_panoramas')
        if current_views is not None:
            current_views = current_views.to(device)
        if history_panoramas is not None:
            history_panoramas = history_panoramas.to(device)

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                video_frames=video_frames,
                instruction_text=instruction,
                current_observation=current_frame.to(device),
                current_views=current_views,
                history_panoramas=history_panoramas,
                return_heatmaps=eval_heatmap,
                return_actions=eval_trajectory,
            )
        
        # Evaluate heatmap
        if eval_heatmap and 'heatmaps' in outputs:
            pred_hm = flatten_heatmap_slices(outputs['heatmaps'].detach().cpu()).numpy()
            gt_hm = flatten_heatmap_slices(gt_heatmap.cpu()).numpy()

            if pred_hm.shape != gt_hm.shape:
                logger.warning(
                    "Skip heatmap metrics due to shape mismatch: pred=%s gt=%s",
                    pred_hm.shape, gt_hm.shape,
                )
            else:
                for hm_idx in range(pred_hm.shape[0]):
                    if gt_hm[hm_idx].sum() > 0:
                        metrics = compute_spatial_metrics(pred_hm[hm_idx], gt_hm[hm_idx])
                        totals['hm_peak_error'] += metrics['peak_error']
                        totals['hm_iou'] += metrics['iou']
                        totals['hm_cosine_sim'] += metrics['cosine_sim']
                        totals['hm_mae'] += metrics['mae']
                        counts['hm'] += 1

        # Evaluate trajectory
        if eval_trajectory and 'trajectory' in outputs:
            pred_traj = outputs['trajectory']
            if pred_traj is not None and 'trajectory' in batch:
                gt_traj = batch['trajectory'].cpu().numpy()
                traj_valid = batch['trajectory_valid'].cpu().numpy()
                pred_traj = pred_traj.cpu().numpy()
                
                for b in range(B):
                    metrics = compute_trajectory_metrics(pred_traj[b], gt_traj[b], traj_valid[b])
                    if metrics['valid']:
                        totals['traj_ade'] += metrics['ade']
                        totals['traj_fde'] += metrics['fde']
                        counts['traj'] += 1
        
        # Visualization
        if save_dir is not None and idx < num_vis:
            save_dir.mkdir(parents=True, exist_ok=True)
            visualize_sample(
                idx, batch, outputs, save_dir,
                eval_heatmap, eval_trajectory, model
            )

    # Average metrics
    results = {}
    
    if counts['hm'] > 0:
        results['hm_peak_error'] = totals['hm_peak_error'] / counts['hm']
        results['hm_iou'] = totals['hm_iou'] / counts['hm']
        results['hm_cosine_sim'] = totals['hm_cosine_sim'] / counts['hm']
        results['hm_mae'] = totals['hm_mae'] / counts['hm']
        results['num_hm_samples'] = counts['hm']
    
    if counts['traj'] > 0:
        results['traj_ade'] = totals['traj_ade'] / counts['traj']
        results['traj_fde'] = totals['traj_fde'] / counts['traj']
        results['num_traj_samples'] = counts['traj']
    
    # progress head removed — no progress metrics
        results['num_progress_samples'] = counts['progress']
    
    return results


def visualize_sample(
    idx: int,
    batch: Dict,
    outputs: Dict,
    save_dir: Path,
    eval_heatmap: bool,
    eval_trajectory: bool,
    model: VLNPipeline,
):
    """Visualize a single sample."""
    current_frame = batch['current_frame'][0].permute(1, 2, 0).cpu().numpy()
    current_frame = np.clip(current_frame, 0, 1)
    gt_heatmap = batch['heatmap'][0].cpu().numpy()
    if gt_heatmap.ndim == 4:
        gt_heatmap = gt_heatmap[0, 0]
    elif gt_heatmap.ndim == 3:
        gt_heatmap = gt_heatmap[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Frame and Heatmaps
    axes[0, 0].imshow(current_frame)
    axes[0, 0].set_title("Current Frame")
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(gt_heatmap, cmap='inferno', vmin=0, vmax=1)
    axes[0, 1].set_title(f"GT Heatmap (max={gt_heatmap.max():.2f})")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    if eval_heatmap and 'heatmaps' in outputs:
        pred_hm = outputs['heatmaps'][0, 0, 0].cpu().numpy()
        pred_hm = np.clip(pred_hm, 0, 1)
        im = axes[0, 2].imshow(pred_hm, cmap='inferno', vmin=0, vmax=1)
        axes[0, 2].set_title(f"Pred Heatmap (max={pred_hm.max():.2f})")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    else:
        axes[0, 2].axis('off')
    
    # Row 2: Trajectory and Progress
    if eval_trajectory and 'trajectory' in batch:
        gt_traj = batch['trajectory'][0].cpu().numpy()
        
        # Plot GT trajectory
        gt_cum = np.cumsum(gt_traj[:, :2], axis=0)
        axes[1, 0].plot(gt_cum[:, 0], gt_cum[:, 1], 'b-o', label='GT', markersize=3)
        axes[1, 0].scatter([0], [0], c='green', s=100, marker='*', label='Start')
        axes[1, 0].scatter([gt_cum[-1, 0]], [gt_cum[-1, 1]], c='red', s=100, marker='X', label='End')
        
        pred_traj = outputs.get('trajectory')
        if pred_traj is not None:
            pred_traj = pred_traj[0].cpu().numpy()
            pred_cum = np.cumsum(pred_traj[:, :2], axis=0)
            axes[1, 0].plot(pred_cum[:, 0], pred_cum[:, 1], 'r--s', label='Pred', markersize=3)
        
        axes[1, 0].set_title("Trajectory (cumulative)")
        axes[1, 0].legend()
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    axes[1, 1].axis('off')
    
    # Info text
    info_text = f"Sample {idx}\n"
    if 'text' in batch and batch['text'][0]:
        info_text += f"Instruction: {batch['text'][0][:50]}..."
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    transform=axes[1, 2].transAxes, wrap=True)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / f"sample_{idx:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='VLN Pipeline Evaluation')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    parser.add_argument('--num-vis', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--eval-heatmap', action='store_true', help='Evaluate heatmap')
    parser.add_argument('--eval-trajectory', action='store_true', help='Evaluate trajectory')
    parser.add_argument('--eval-progress', action='store_true', help='Evaluate progress')
    parser.add_argument('--use-packing', action='store_true', help='Use sequence packing')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    cfg = load_config(args.config)

    # Default: evaluate all
    if not args.eval_heatmap and not args.eval_trajectory and not args.eval_progress:
        args.eval_heatmap = True
        args.eval_trajectory = True
        args.eval_progress = True

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model = build_model(cfg, device=args.device)
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    # Build dataloader
    packing_enabled = args.use_packing or cfg['model']['llm'].get('enable_packing', False)
    if packing_enabled:
        raise ValueError("Qwen3.5 v2 评估路径不支持 sequence packing，请关闭 enable_packing。")
    dataloader = build_dataloader(cfg, split=args.split)
    logger.info(f"Dataset: {len(dataloader.dataset)} samples")

    # Save directory
    save_dir = None
    if args.save_vis:
        save_dir = Path(cfg['log']['out_dir']) / 'eval_vis'
        save_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    logger.info("=" * 60)
    logger.info(f"Evaluating on {args.split} split...")
    logger.info(f"  Heatmap: {args.eval_heatmap}")
    logger.info(f"  Trajectory: {args.eval_trajectory}")
    logger.info(f"  Progress: {args.eval_progress}")
    logger.info("  Packing: False")
    logger.info("=" * 60)

    metrics = evaluate(
        model, dataloader, cfg, device, save_dir,
        args.num_vis, args.eval_heatmap, args.eval_trajectory, args.eval_progress,
        args=args
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    if args.eval_heatmap and 'num_hm_samples' in metrics:
        logger.info(f"Heatmap ({metrics['num_hm_samples']} samples):")
        logger.info(f"  Peak Error: {metrics['hm_peak_error']:.2f} pixels")
        logger.info(f"  IoU:        {metrics['hm_iou']:.4f}")
        logger.info(f"  Cosine Sim: {metrics['hm_cosine_sim']:.4f}")
        logger.info(f"  MAE:        {metrics['hm_mae']:.4f}")
    
    if args.eval_trajectory and 'num_traj_samples' in metrics:
        logger.info(f"Trajectory ({metrics['num_traj_samples']} samples):")
        logger.info(f"  ADE:        {metrics['traj_ade']:.4f}")
        logger.info(f"  FDE:        {metrics['traj_fde']:.4f}")
    
    if args.eval_progress and 'num_progress_samples' in metrics:
        logger.info(f"Progress ({metrics['num_progress_samples']} samples):")
        logger.info(f"  MAE:        {metrics['progress_mae']:.4f}")
        logger.info(f"  Accuracy:   {metrics['progress_accuracy']:.4f}")
        logger.info(f"  Boundary:   {metrics['progress_boundary_acc']:.4f}")

    # Save metrics
    metrics_file = Path(cfg['log']['out_dir']) / f'metrics_{args.split}.yaml'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
