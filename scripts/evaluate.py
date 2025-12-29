#!/usr/bin/env python3
"""
VLN Pipeline 评估脚本
======================

使用 Qwen3-VL 评估视觉语言导航模型。

支持评估：
- 历史热力图头 (History Heatmap)
- 未来热力图头 (Future Heatmap)
- 动作头 (Action Head)
- 停止预测头 (Stop Head)
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.utils.loss import SimplifiedHeatmapLoss, NeRFRippleHeatmapLoss

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
    """Collate function for VLNSlidingWindowDataset."""
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
    
    return {
        'history_frames': torch.stack(history_frames_padded, dim=0),
        'history_mask': torch.stack(history_mask, dim=0),
        'current_frame': torch.stack([s['current_frame'] for s in batch], dim=0),
        'heatmap': torch.stack([s['heatmap'] for s in batch], dim=0),
        'action': torch.stack([s['action'] for s in batch], dim=0),
        'action_valid': torch.tensor([s['action_valid'] for s in batch]),
        'is_stop': torch.tensor([s.get('is_stop', 0.0) for s in batch]),
        'text': [s['text'] for s in batch],
    }


def build_model(cfg: Dict, device: str = 'cuda:0') -> VLNPipeline:
    """Build VLN pipeline for evaluation."""
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    stop_cfg = model_cfg.get('stop_head', {})
    
    config = VLNPipelineConfig(
        # Qwen3-VL
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 2048),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'flash_attention_2'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        
        # Device
        device=device,
        
        # Heatmap
        heatmap_size=tuple(data_cfg['init_hm_size']),
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', True),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        image_size=data_cfg['image_size'][0],
        
        # Action
        enable_action_head=action_cfg.get('enable', True),
        action_dim=action_cfg.get('action_dim', 2),
        action_pred_horizon=action_cfg.get('pred_horizon', 1),
        action_encoding_size=action_cfg.get('encoding_size', 256),
        action_num_diffusion_iters=action_cfg.get('num_diffusion_iters', 10),
        
        # Stop
        enable_stop_head=stop_cfg.get('enable', True),
        stop_hidden_dim=stop_cfg.get('hidden_dim', 512),
        
        verbose=True,
    )
    
    return VLNPipeline(config)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint with partial loading support."""
    ckpt = torch.load(checkpoint_path, map_location=device)
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


def build_dataloader(cfg: Dict, split: str = 'val') -> DataLoader:
    """Build dataloader using VLNSlidingWindowDataset."""
    sw_cfg = cfg['data']['sliding_window']
    
    dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split=split,
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
    )
    
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def compute_spatial_metrics(pred_hm: np.ndarray, gt_hm: np.ndarray) -> Dict[str, float]:
    """Compute spatial accuracy metrics."""
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


def compute_action_metrics(pred_action: np.ndarray, gt_action: np.ndarray) -> Dict[str, float]:
    """Compute action prediction metrics."""
    l2_error = np.sqrt(((pred_action - gt_action) ** 2).sum(axis=-1)).mean()
    dx_error = np.abs(pred_action[..., 0] - gt_action[..., 0]).mean()
    dy_error = np.abs(pred_action[..., 1] - gt_action[..., 1]).mean()
    
    return {
        'action_l2_error': float(l2_error),
        'action_dx_error': float(dx_error),
        'action_dy_error': float(dy_error),
    }


def compute_stop_metrics(pred_stop: np.ndarray, gt_stop: np.ndarray) -> Dict[str, float]:
    """Compute stop prediction metrics."""
    pred_binary = (pred_stop > 0.5).astype(float)
    accuracy = (pred_binary == gt_stop).mean()
    
    # Precision, Recall, F1
    tp = ((pred_binary == 1) & (gt_stop == 1)).sum()
    fp = ((pred_binary == 1) & (gt_stop == 0)).sum()
    fn = ((pred_binary == 0) & (gt_stop == 1)).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'stop_accuracy': float(accuracy),
        'stop_precision': float(precision),
        'stop_recall': float(recall),
        'stop_f1': float(f1),
    }


@torch.no_grad()
def evaluate(
    model: VLNPipeline,
    dataloader: DataLoader,
    cfg: Dict,
    device: torch.device,
    save_dir: Path = None,
    num_vis: int = 20,
    eval_history: bool = True,
    eval_future: bool = True,
    eval_action: bool = True,
    eval_stop: bool = True,
    amp_dtype=None,
    args=None
) -> Dict[str, float]:
    """Run evaluation."""
    model.eval()

    totals = {
        'hist_peak_error': 0.0,
        'hist_iou': 0.0,
        'hist_cosine_sim': 0.0,
        'hist_mae': 0.0,
        'fut_peak_error': 0.0,
        'fut_iou': 0.0,
        'fut_cosine_sim': 0.0,
        'fut_mae': 0.0,
        'action_l2_error': 0.0,
        'action_dx_error': 0.0,
        'action_dy_error': 0.0,
        'stop_accuracy': 0.0,
        'stop_precision': 0.0,
        'stop_recall': 0.0,
        'stop_f1': 0.0,
    }
    counts = {'hist': 0, 'fut': 0, 'action': 0, 'stop': 0}

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args and args.max_samples is not None and idx >= args.max_samples:
            logger.info(f"Reached max_samples={args.max_samples}, stopping")
            break

        # Prepare data
        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        B, K, C, H, W = history_frames.shape
        
        video_frames = torch.cat([
            history_frames,
            current_frame.unsqueeze(1)
        ], dim=1).to(device)
        
        gt_heatmap = batch['heatmap'].to(device)
        gt_action = batch['action'].to(device)
        action_valid = batch['action_valid'].to(device)
        gt_stop = batch['is_stop'].to(device)
        
        # Get instruction
        text = batch['text']
        instruction = text[0] if text and len(text) > 0 and text[0] else None
        
        # Forward pass
        if amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                outputs = model(
                    video_frames=video_frames,
                    instruction_text=instruction,
                    current_observation=current_frame.to(device),
                    return_heatmaps=True,
                    return_actions=eval_action,
                )
        else:
            outputs = model(
                video_frames=video_frames,
                instruction_text=instruction,
                current_observation=current_frame.to(device),
                return_heatmaps=True,
                return_actions=eval_action,
            )
        
        # Evaluate history heatmap
        if eval_history and 'history_heatmaps' in outputs:
            pred_hm = outputs['history_heatmaps'][:, -1].cpu().numpy()
            gt_hm = gt_heatmap.cpu().numpy()
            
            for b in range(B):
                if gt_hm[b].sum() > 0:
                    metrics = compute_spatial_metrics(pred_hm[b], gt_hm[b])
                    totals['hist_peak_error'] += metrics['peak_error']
                    totals['hist_iou'] += metrics['iou']
                    totals['hist_cosine_sim'] += metrics['cosine_sim']
                    totals['hist_mae'] += metrics['mae']
                    counts['hist'] += 1

        # Evaluate future heatmap
        if eval_future and 'future_heatmaps' in outputs:
            pred_hm = outputs['future_heatmaps'][:, -1].cpu().numpy()
            gt_hm = gt_heatmap.cpu().numpy()
            
            for b in range(B):
                if gt_hm[b].sum() > 0:
                    metrics = compute_spatial_metrics(pred_hm[b], gt_hm[b])
                    totals['fut_peak_error'] += metrics['peak_error']
                    totals['fut_iou'] += metrics['iou']
                    totals['fut_cosine_sim'] += metrics['cosine_sim']
                    totals['fut_mae'] += metrics['mae']
                    counts['fut'] += 1

        # Evaluate action
        if eval_action and 'actions' in outputs and outputs['actions'] is not None:
            pred_act = outputs['actions'].cpu().numpy()
            gt_act = gt_action.cpu().numpy()
            
            for b in range(B):
                if action_valid[b]:
                    metrics = compute_action_metrics(pred_act[b], gt_act[b])
                    totals['action_l2_error'] += metrics['action_l2_error']
                    totals['action_dx_error'] += metrics['action_dx_error']
                    totals['action_dy_error'] += metrics['action_dy_error']
                    counts['action'] += 1
        
        # Evaluate stop prediction
        if eval_stop and 'stop_prob' in outputs:
            pred_stop = outputs['stop_prob'].cpu().numpy()
            gt_stop_np = gt_stop.cpu().numpy()
            
            for b in range(B):
                if action_valid[b]:
                    metrics = compute_stop_metrics(pred_stop[b:b+1], gt_stop_np[b:b+1])
                    totals['stop_accuracy'] += metrics['stop_accuracy']
                    totals['stop_precision'] += metrics['stop_precision']
                    totals['stop_recall'] += metrics['stop_recall']
                    totals['stop_f1'] += metrics['stop_f1']
                    counts['stop'] += 1
        
        # Visualization
        if save_dir is not None and idx < num_vis:
            save_dir.mkdir(parents=True, exist_ok=True)
            visualize_sample(
                idx, batch, outputs, save_dir,
                eval_history, eval_future, eval_action
            )

    # Average metrics
    results = {}
    for k, v in totals.items():
        if k.startswith('hist_'):
            results[k] = v / max(counts['hist'], 1)
        elif k.startswith('fut_'):
            results[k] = v / max(counts['fut'], 1)
        elif k.startswith('action_'):
            results[k] = v / max(counts['action'], 1)
        elif k.startswith('stop_'):
            results[k] = v / max(counts['stop'], 1)
    
    results['num_hist_samples'] = counts['hist']
    results['num_fut_samples'] = counts['fut']
    results['num_action_samples'] = counts['action']
    results['num_stop_samples'] = counts['stop']
    
    return results


def visualize_sample(
    idx: int,
    batch: Dict,
    outputs: Dict,
    save_dir: Path,
    eval_history: bool,
    eval_future: bool,
    eval_action: bool
):
    """Visualize a single sample."""
    current_frame = batch['current_frame'][0].permute(1, 2, 0).cpu().numpy()
    gt_heatmap = batch['heatmap'][0].cpu().numpy()
    gt_action = batch['action'][0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Frame and GT
    axes[0, 0].imshow(current_frame)
    axes[0, 0].set_title("Current Frame")
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(gt_heatmap, cmap='hot')
    axes[0, 1].set_title("GT Heatmap")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    axes[0, 2].text(0.5, 0.5, f"GT Action:\ndx={gt_action[0]:.4f}\ndy={gt_action[1]:.4f}",
                    ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("GT Action")
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    if eval_history and 'history_heatmaps' in outputs:
        pred_hist = outputs['history_heatmaps'][0, -1].cpu().numpy()
        im = axes[1, 0].imshow(pred_hist, cmap='hot')
        axes[1, 0].set_title("Pred History Heatmap")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].axis('off')
    
    if eval_future and 'future_heatmaps' in outputs:
        pred_fut = outputs['future_heatmaps'][0, -1].cpu().numpy()
        im = axes[1, 1].imshow(pred_fut, cmap='hot')
        axes[1, 1].set_title("Pred Future Heatmap")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].axis('off')
    
    if eval_action and 'actions' in outputs and outputs['actions'] is not None:
        pred_act = outputs['actions'][0].cpu().numpy()
        axes[1, 2].text(0.5, 0.5, f"Pred Action:\ndx={pred_act[0]:.4f}\ndy={pred_act[1]:.4f}",
                        ha='center', va='center', fontsize=14, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("Pred Action")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / f"sample_{idx:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='VLN Pipeline Evaluation')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    parser.add_argument('--num-vis', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--use-history', action='store_true', help='Evaluate history heatmap')
    parser.add_argument('--use-future', action='store_true', help='Evaluate future heatmap')
    parser.add_argument('--use-action', action='store_true', help='Evaluate action')
    parser.add_argument('--use-stop', action='store_true', help='Evaluate stop prediction')
    parser.add_argument('--amp', type=str, default='bf16', choices=['none', 'fp16', 'bf16'])
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
    if not args.use_history and not args.use_future and not args.use_action and not args.use_stop:
        args.use_history = True
        args.use_future = True
        args.use_action = True
        args.use_stop = True

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
    dataloader = build_dataloader(cfg, split=args.split)
    logger.info(f"Dataset: {len(dataloader.dataset)} samples")

    # Save directory
    save_dir = None
    if args.save_vis:
        save_dir = Path(cfg['log']['out_dir']) / 'eval_vis'
        save_dir.mkdir(parents=True, exist_ok=True)

    # AMP dtype
    amp_dtype = None
    if args.amp == 'bf16':
        amp_dtype = torch.bfloat16
    elif args.amp == 'fp16':
        amp_dtype = torch.float16

    # Run evaluation
    logger.info("=" * 60)
    logger.info(f"Evaluating on {args.split} split...")
    logger.info(f"  History: {args.use_history}")
    logger.info(f"  Future: {args.use_future}")
    logger.info(f"  Action: {args.use_action}")
    logger.info(f"  Stop: {args.use_stop}")
    logger.info("=" * 60)

    metrics = evaluate(
        model, dataloader, cfg, device, save_dir,
        args.num_vis, args.use_history, args.use_future, args.use_action, args.use_stop,
        amp_dtype, args
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    if args.use_history:
        logger.info(f"History Heatmap ({metrics['num_hist_samples']} samples):")
        logger.info(f"  Peak Error: {metrics['hist_peak_error']:.2f} pixels")
        logger.info(f"  IoU:        {metrics['hist_iou']:.4f}")
        logger.info(f"  Cosine Sim: {metrics['hist_cosine_sim']:.4f}")
        logger.info(f"  MAE:        {metrics['hist_mae']:.4f}")
    
    if args.use_future:
        logger.info(f"Future Heatmap ({metrics['num_fut_samples']} samples):")
        logger.info(f"  Peak Error: {metrics['fut_peak_error']:.2f} pixels")
        logger.info(f"  IoU:        {metrics['fut_iou']:.4f}")
        logger.info(f"  Cosine Sim: {metrics['fut_cosine_sim']:.4f}")
        logger.info(f"  MAE:        {metrics['fut_mae']:.4f}")
    
    if args.use_action:
        logger.info(f"Action ({metrics['num_action_samples']} samples):")
        logger.info(f"  L2 Error:   {metrics['action_l2_error']:.4f}")
        logger.info(f"  dx Error:   {metrics['action_dx_error']:.4f}")
        logger.info(f"  dy Error:   {metrics['action_dy_error']:.4f}")
    
    if args.use_stop:
        logger.info(f"Stop Prediction ({metrics['num_stop_samples']} samples):")
        logger.info(f"  Accuracy:   {metrics['stop_accuracy']:.4f}")
        logger.info(f"  Precision:  {metrics['stop_precision']:.4f}")
        logger.info(f"  Recall:     {metrics['stop_recall']:.4f}")
        logger.info(f"  F1:         {metrics['stop_f1']:.4f}")

    # Save metrics
    metrics_file = Path(cfg['log']['out_dir']) / f'metrics_{args.split}.yaml'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
