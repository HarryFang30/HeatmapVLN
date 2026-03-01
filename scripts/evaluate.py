#!/usr/bin/env python3
"""
VLN Pipeline 评估脚本
======================

使用 Qwen3-VL 评估视觉语言导航模型。

支持评估：
- 历史热力图头 (History Heatmap)
- 轨迹预测头 (Trajectory - TransformerActionHead)
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
from src.data.tokenized_dataset import TokenizedVLNDataset, FlattenedCollatorForVLN

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    
    return result


def build_model(cfg: Dict, device: str = 'cuda:0') -> VLNPipeline:
    """Build VLN pipeline for evaluation (与 train.py 保持一致)."""
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    stop_cfg = model_cfg.get('stop_head', {})
    progress_cfg = model_cfg.get('progress_head', {})
    
    action_head_type = action_cfg.get('type', 'transformer')
    legacy_action_cfg = action_cfg.get('legacy', {})
    transformer_action_cfg = action_cfg.get('transformer', {})
    
    config = VLNPipelineConfig(
        # Qwen3-VL
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 4096),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'flash_attention_2'),
        max_video_frames=llm_cfg.get('max_video_frames', -1),
        
        # Sequence Packing
        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 8192),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        
        # Device
        device=device,
        
        # Heatmap
        heatmap_size=tuple(data_cfg['init_hm_size']),
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', False),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        image_size=data_cfg['image_size'][0],
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
        # Sequence cross-attention conditioning
        heatmap_use_sequence_conditioning=heatmap_cfg.get('use_sequence_conditioning', False),
        heatmap_seq_cross_attn_heads=heatmap_cfg.get('seq_cross_attn_heads', 8),
        heatmap_seq_cross_attn_head_dim=heatmap_cfg.get('seq_cross_attn_head_dim', 64),
        # Spatial feature injection
        heatmap_use_spatial_injection=heatmap_cfg.get('use_spatial_injection', False),
        heatmap_image_encoder_use_pretrained=heatmap_cfg.get('image_encoder_use_pretrained', False),
        
        # Multi-layer feature extraction
        multi_layer_features=llm_cfg.get('multi_layer_features', False),
        feature_layer_indices=llm_cfg.get('feature_layer_indices', None),
        feature_fusion_method=llm_cfg.get('feature_fusion_method', 'weighted_sum'),
        
        # LoRA
        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        
        # Action Head
        action_head_type=action_head_type,
        enable_action_head=action_cfg.get('enable', True),
        action_dim=legacy_action_cfg.get('action_dim', 2),
        action_pred_horizon=legacy_action_cfg.get('pred_horizon', 1),
        action_encoding_size=legacy_action_cfg.get('encoding_size', 256),
        action_down_dims=legacy_action_cfg.get('down_dims', None),
        action_num_diffusion_iters=legacy_action_cfg.get('num_diffusion_iters', 10),
        action_stats_min=legacy_action_cfg.get('action_stats_min', [-0.17, -0.03]),
        action_stats_max=legacy_action_cfg.get('action_stats_max', [0.19, 0.31]),
        transformer_action_dim=transformer_action_cfg.get('action_dim', 3),
        transformer_predict_size=transformer_action_cfg.get('predict_size', 24),
        transformer_n_emb=transformer_action_cfg.get('n_emb', 384),
        transformer_n_layer=transformer_action_cfg.get('n_layer', 16),
        transformer_n_head=transformer_action_cfg.get('n_head', 6),
        transformer_n_cond_layers=transformer_action_cfg.get('n_cond_layers', 4),
        transformer_num_train_timesteps=transformer_action_cfg.get('num_train_timesteps', 20),
        transformer_p_drop_emb=transformer_action_cfg.get('p_drop_emb', 0.1),
        transformer_p_drop_attn=transformer_action_cfg.get('p_drop_attn', 0.1),
        transformer_causal_attn=transformer_action_cfg.get('causal_attn', True),
        
        # Stop / Progress
        enable_stop_head=stop_cfg.get('enable', False),
        stop_hidden_dim=stop_cfg.get('hidden_dim', 512),
        stop_focal_gamma=stop_cfg.get('focal_gamma', 3.0),
        stop_focal_alpha=stop_cfg.get('focal_alpha', 0.9),
        enable_progress_head=progress_cfg.get('enable', False),
        progress_hidden_dim=progress_cfg.get('hidden_dim', 512),
        
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
    model: Optional[VLNPipeline] = None,
    packing_enabled: bool = False,
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
    
    if packing_enabled and model is not None:
        # Ensure model is loaded for processor
        if not model.qwen3_vl._model_loaded:
            model.qwen3_vl._load_model()
        
        spatial_merge_size = cfg['model']['llm'].get('spatial_merge_size', 2)
        dataset = TokenizedVLNDataset(
            base_dataset=base_dataset,
            processor=model.qwen3_vl.processor,
            spatial_merge_size=spatial_merge_size,
        )
        actual_collate_fn = FlattenedCollatorForVLN()
        num_workers = 0  # Packing mode uses main process
    else:
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
    packing_enabled: bool = False,
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
        
        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if packing_enabled:
                outputs = model.forward_packed(
                    packed_batch=batch,
                    return_heatmaps=eval_heatmap,
                    return_actions=eval_trajectory,
                )
            else:
                history_frames = batch['history_frames']
                video_frames = torch.cat([
                    history_frames,
                    current_frame.unsqueeze(1)
                ], dim=1).to(device)
                
                text = batch['text']
                instruction = list(text) if text and len(text) > 0 else None
                
                outputs = model(
                    video_frames=video_frames,
                    instruction_text=instruction,
                    current_observation=current_frame.to(device),
                    return_heatmaps=eval_heatmap,
                    return_actions=eval_trajectory,
                )
        
        # Evaluate heatmap
        if eval_heatmap and 'history_heatmaps' in outputs:
            pred_hm = outputs['history_heatmaps'][:, -1].cpu().numpy()
            gt_hm = gt_heatmap.cpu().numpy()
            
            for b in range(B):
                if gt_hm[b].sum() > 0:
                    metrics = compute_spatial_metrics(pred_hm[b], gt_hm[b])
                    totals['hm_peak_error'] += metrics['peak_error']
                    totals['hm_iou'] += metrics['iou']
                    totals['hm_cosine_sim'] += metrics['cosine_sim']
                    totals['hm_mae'] += metrics['mae']
                    counts['hm'] += 1

        # Evaluate trajectory
        if eval_trajectory and 'trajectory' in batch:
            gt_traj = batch['trajectory'].cpu().numpy()
            traj_valid = batch['trajectory_valid'].cpu().numpy()
            
            # Get predicted trajectory
            if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
                action_cond = outputs.get('action_cond')
                if action_cond is not None:
                    if action_cond.dim() == 2:
                        action_cond = action_cond.unsqueeze(1)
                    pred_traj = model.transformer_action_head.get_trajectory(action_cond)
                    pred_traj = pred_traj.cpu().numpy()
                    
                    for b in range(B):
                        metrics = compute_trajectory_metrics(pred_traj[b], gt_traj[b], traj_valid[b])
                        if metrics['valid']:
                            totals['traj_ade'] += metrics['ade']
                            totals['traj_fde'] += metrics['fde']
                            counts['traj'] += 1
        
        # Evaluate progress
        if eval_progress and 'progress' in batch:
            gt_progress = batch['progress'].cpu().numpy()
            
            if hasattr(model, 'progress_head') and model.progress_head is not None:
                pred_progress = outputs.get('progress')
                if pred_progress is not None:
                    pred_progress = pred_progress.cpu().numpy()
                    
                    metrics = compute_progress_metrics(pred_progress, gt_progress)
                    totals['progress_mae'] += metrics['progress_mae'] * B
                    totals['progress_accuracy'] += metrics['progress_accuracy'] * B
                    totals['progress_boundary_acc'] += metrics['progress_boundary_acc'] * B
                    counts['progress'] += B
        
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
    
    if counts['progress'] > 0:
        results['progress_mae'] = totals['progress_mae'] / counts['progress']
        results['progress_accuracy'] = totals['progress_accuracy'] / counts['progress']
        results['progress_boundary_acc'] = totals['progress_boundary_acc'] / counts['progress']
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Frame and Heatmaps
    axes[0, 0].imshow(current_frame)
    axes[0, 0].set_title("Current Frame")
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(gt_heatmap, cmap='inferno', vmin=0, vmax=1)
    axes[0, 1].set_title(f"GT Heatmap (max={gt_heatmap.max():.2f})")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    if eval_heatmap and 'history_heatmaps' in outputs:
        pred_hm = outputs['history_heatmaps'][0, -1].cpu().numpy()
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
        
        # Plot predicted trajectory
        if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
            action_cond = outputs.get('action_cond')
            if action_cond is not None:
                if action_cond.dim() == 2:
                    action_cond = action_cond.unsqueeze(1)
                pred_traj = model.transformer_action_head.get_trajectory(action_cond)
                pred_traj = pred_traj[0].cpu().numpy()
                pred_cum = np.cumsum(pred_traj[:, :2], axis=0)
                axes[1, 0].plot(pred_cum[:, 0], pred_cum[:, 1], 'r--s', label='Pred', markersize=3)
        
        axes[1, 0].set_title("Trajectory (cumulative)")
        axes[1, 0].legend()
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # Progress
    if 'progress' in batch:
        gt_progress = batch['progress'][0].item()
        pred_progress = outputs.get('progress')
        if pred_progress is not None:
            pred_progress = pred_progress[0].item()
            axes[1, 1].bar(['GT', 'Pred'], [gt_progress, pred_progress], color=['blue', 'orange'])
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title(f"Progress (GT={gt_progress:.2f}, Pred={pred_progress:.2f})")
        else:
            axes[1, 1].bar(['GT'], [gt_progress], color=['blue'])
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title(f"Progress (GT={gt_progress:.2f})")
    else:
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
    dataloader = build_dataloader(cfg, split=args.split, model=model, packing_enabled=packing_enabled)
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
    logger.info(f"  Packing: {packing_enabled}")
    logger.info("=" * 60)

    metrics = evaluate(
        model, dataloader, cfg, device, save_dir,
        args.num_vis, args.eval_heatmap, args.eval_trajectory, args.eval_progress,
        packing_enabled, args
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
