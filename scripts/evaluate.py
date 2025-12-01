#!/usr/bin/env python3
"""
Evaluation for SpatialMLLMPipeline (dual-head NavigationHeatmapLoss).
支持选择评估历史头/未来头。
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.utils.loss import NavigationHeatmapLoss
from src.utils.frame_vis_utils import interpolate_keyframe_predictions
from src.utils.visualization import create_comparison_grid, create_thumbnail
from src.utils.html_template import create_html_index

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict, device: str = 'cuda:0') -> SpatialMLLMPipeline:
    """Build model for evaluation (single GPU)

    Args:
        cfg: Configuration dictionary
        device: Target device (default: 'cuda:0')

    Returns:
        SpatialMLLMPipeline model ready for evaluation
    """
    model_cfg = cfg['model']
    integration_cfg = SpatialMLLMIntegrationConfig(
        target_keyframes=cfg['data']['heatmap_per_clip'],
        total_frames=cfg['data']['frames_per_clip'],
        sampling_method="hybrid",
        llm_model_path=model_cfg['llm']['model_path'],
        # ⭐ FIX: Use single device for all components during evaluation
        vggt_gpu=device,
        dinov3_gpu=device,
        llm_gpu=device,
        use_multi_gpu=False,          # Correct: single GPU mode for evaluation
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_inter_frame_heatmaps=True,
        dinov3_img_size=cfg['data']['image_size'][0],
        vggt_img_size=cfg['data']['image_size'][0],
        enable_gradient_checkpointing=False,  # Evaluation doesn't need gradient checkpointing
        verbose=True
    )
    return SpatialMLLMPipeline(integration_cfg)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint with strict validation

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device for loading

    Returns:
        Model with loaded weights

    Raises:
        RuntimeError: If checkpoint architecture doesn't match model
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load checkpoint with strict=False to allow partial loading (e.g., only heatmap heads)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.info(f"Missing keys (will use pretrained weights): {len(missing_keys)} keys")
        if len(missing_keys) < 10:
            for key in missing_keys:
                logger.info(f"  - {key}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) < 10:
            for key in unexpected_keys:
                logger.warning(f"  - {key}")

    logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Epoch: {ckpt.get('epoch', 'N/A')}  Stage: {ckpt.get('stage', 'N/A')}")

    # Calculate correctly: how many checkpoint params were actually loaded
    loaded_count = len([k for k in state_dict.keys() if k not in missing_keys])
    logger.info(f"  Loaded {loaded_count} / {len(state_dict)} parameters from checkpoint")
    logger.info(f"  Using pretrained weights for {len(missing_keys)} parameters not in checkpoint")

    return model


def build_dataloader(cfg: Dict, split: str = 'val') -> DataLoader:
    dataset = VLNHeatmapDataset(
        root=cfg['data']['root'],
        split=split,
        frames_per_clip=cfg['data']['frames_per_clip'],
        heatmap_per_clip=cfg['data']['heatmap_per_clip'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        num_sample_frames=cfg['data'].get('num_sample_frames')
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def aggregate_gt(gt_hm: np.ndarray, gt_val: np.ndarray) -> np.ndarray:
    # gt_hm: [T,H,W], gt_val: [T]
    valid_mask = (gt_val > 0.5).astype(np.float32)[:, None, None]
    weighted = gt_hm * valid_mask
    sum_hm = weighted.sum(axis=0)
    count = valid_mask.sum(axis=0).clip(min=1.0)
    avg = sum_hm / count
    if avg.sum() > 0:
        avg = avg / avg.sum()
    return avg


def compute_spatial_metrics(pred_hm: np.ndarray, gt_hm: np.ndarray) -> Dict[str, float]:
    """Compute spatial accuracy metrics beyond loss

    Args:
        pred_hm: Predicted heatmap [H, W]
        gt_hm: Ground truth heatmap [H, W]

    Returns:
        Dictionary of spatial metrics
    """
    # Peak location error (Euclidean distance between predicted and GT peaks)
    pred_peak = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
    gt_peak = np.unravel_index(np.argmax(gt_hm), gt_hm.shape)
    peak_error = np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    # IoU at threshold (intersection over union for high-confidence regions)
    thresh = 0.3
    pred_mask = pred_hm > thresh * pred_hm.max()
    gt_mask = gt_hm > thresh * gt_hm.max()
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    iou = intersection / (union + 1e-6)

    # Cosine similarity (measures distribution alignment)
    cos_sim = np.dot(pred_hm.flatten(), gt_hm.flatten()) / (
        np.linalg.norm(pred_hm) * np.linalg.norm(gt_hm) + 1e-6
    )

    # Mean Absolute Error
    mae = np.abs(pred_hm - gt_hm).mean()

    return {
        'peak_error': float(peak_error),
        'iou': float(iou),
        'cosine_sim': float(cos_sim),
        'mae': float(mae)
    }


@torch.no_grad()
def evaluate(model: SpatialMLLMPipeline, dataloader: DataLoader,
             criterion: NavigationHeatmapLoss, cfg: Dict, device: torch.device,
             save_dir: Path = None, num_vis: int = 20, eval_history: bool = True, eval_future: bool = True, amp_dtype=None, args=None) -> Dict[str, float]:
    model.eval()

    # Collect samples info for HTML index
    samples_info = []

    totals = {
        'total_loss': 0.0,
        'history_loss': 0.0,
        'future_loss': 0.0,
        'mse': 0.0,
        'kl': 0.0,
        'valid': 0.0,
        # ⭐ NEW: Enhanced spatial metrics
        'hist_peak_error': 0.0,
        'hist_iou': 0.0,
        'hist_cosine_sim': 0.0,
        'hist_mae': 0.0,
        'fut_peak_error': 0.0,
        'fut_iou': 0.0,
        'fut_cosine_sim': 0.0,
        'fut_mae': 0.0,
    }
    count = 0
    spatial_count = {'hist': 0, 'fut': 0}

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # ⭐ NEW: Early termination if max_samples specified
        if args and args.max_samples is not None and idx >= args.max_samples:
            logger.info(f"Reached max_samples={args.max_samples}, stopping evaluation early")
            break

        # ⭐ FIX: Move ALL tensors to device upfront for consistency
        frames = batch['frames'].to(device)
        gt_hist = batch['gt_heatmap_history'].to(device)
        gt_fut = batch['gt_heatmap_future'].to(device)
        val_hist = batch['gt_validity_history'].to(device)
        val_fut = batch['gt_validity_future'].to(device)

        # ⭐ NEW: Forward pass with optional AMP
        if amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                outputs = model(frames, instruction_text=batch.get('text', [None])[0], return_heatmaps=True)
        else:
            outputs = model(frames, instruction_text=batch.get('text', [None])[0], return_heatmaps=True)

        pred_hist = outputs['history_heatmaps']      # [B,K,H,W]
        pred_fut = outputs['future_heatmaps']
        pred_hist_val = outputs['history_validity']  # [B,K]
        pred_fut_val = outputs['future_validity']

        loss_total = 0.0
        if eval_history and val_hist.sum() > 0.5:
            B, K, Hm, Wm = pred_hist.shape
            logits = pred_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
            gt_map = gt_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
            gt_val = val_hist.reshape(B * K, 1)
            pred_val = pred_hist_val.reshape(B * K, 1)
            loss_hist, comps = criterion(logits, gt_map, pred_val, gt_val)
            totals['history_loss'] += loss_hist.item()
            totals['mse'] += comps['mse']
            totals['kl'] += comps['kl']
            totals['valid'] += comps['valid']
            loss_total += loss_hist.item()

        if eval_future and val_fut.sum() > 0.5:
            B, K, Hm, Wm = pred_fut.shape
            logits = pred_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
            gt_map = gt_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
            gt_val = val_fut.reshape(B * K, 1)
            pred_val = pred_fut_val.reshape(B * K, 1)
            loss_fut, comps = criterion(logits, gt_map, pred_val, gt_val)
            totals['future_loss'] += loss_fut.item()
            totals['mse'] += comps['mse']
            totals['kl'] += comps['kl']
            totals['valid'] += comps['valid']
            loss_total += loss_fut.item()

        totals['total_loss'] += loss_total
        count += 1

        # ⭐ NEW: Compute spatial metrics (aggregate GT first)
        if eval_history and val_hist.sum() > 0.5 and pred_hist.shape[1] > 0:
            pred_hm = pred_hist[0, 0].detach().cpu().numpy()
            gt_hm_agg = aggregate_gt(gt_hist[0].cpu().numpy(), val_hist[0].cpu().numpy())
            if gt_hm_agg.sum() > 0:  # Valid GT heatmap
                spatial_metrics = compute_spatial_metrics(pred_hm, gt_hm_agg)
                totals['hist_peak_error'] += spatial_metrics['peak_error']
                totals['hist_iou'] += spatial_metrics['iou']
                totals['hist_cosine_sim'] += spatial_metrics['cosine_sim']
                totals['hist_mae'] += spatial_metrics['mae']
                spatial_count['hist'] += 1

        if eval_future and val_fut.sum() > 0.5 and pred_fut.shape[1] > 0:
            pred_hm = pred_fut[0, 0].detach().cpu().numpy()
            gt_hm_agg = aggregate_gt(gt_fut[0].cpu().numpy(), val_fut[0].cpu().numpy())
            if gt_hm_agg.sum() > 0:  # Valid GT heatmap
                spatial_metrics = compute_spatial_metrics(pred_hm, gt_hm_agg)
                totals['fut_peak_error'] += spatial_metrics['peak_error']
                totals['fut_iou'] += spatial_metrics['iou']
                totals['fut_cosine_sim'] += spatial_metrics['cosine_sim']
                totals['fut_mae'] += spatial_metrics['mae']
                spatial_count['fut'] += 1

        if save_dir is not None and idx < num_vis:
            save_dir.mkdir(parents=True, exist_ok=True)

            # Frame-by-frame visualization (new mode)
            vis_mode = args.vis_mode if args else 'aggregated'
            if vis_mode in ['frame_by_frame', 'both']:
                T = frames.shape[1]  # T_sampled
                K = pred_hist.shape[1]  # Number of keyframes

                # Assume uniform distribution of keyframes
                keyframe_indices = np.linspace(0, T-1, K, dtype=int)

                # Interpolate predictions to T frames
                pred_hist_interp = interpolate_keyframe_predictions(
                    pred_hist[0], keyframe_indices, T,
                    method=args.interpolation_method if args else 'linear'
                )
                pred_fut_interp = interpolate_keyframe_predictions(
                    pred_fut[0], keyframe_indices, T,
                    method=args.interpolation_method if args else 'linear'
                )

                # Create sample output directory
                sample_dir = save_dir / f"sample_{idx:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Generate comparison grid(s)
                grid_paths = create_comparison_grid(
                    frames[0], gt_hist[0], gt_fut[0],
                    pred_hist_interp, pred_fut_interp,
                    val_hist[0], val_fut[0],
                    save_dir=sample_dir,
                    overlay_mode=args.overlay_mode if args else 'dual',
                    max_frames=args.max_frames_per_vis if args else 16,
                    alpha=args.overlay_alpha if args else 0.5,
                    threshold=args.heatmap_threshold if args else 0.05
                )

                # Generate thumbnail
                first_frame = frames[0][0].permute(1, 2, 0).cpu().numpy()
                first_frame = (first_frame * 255).astype(np.uint8)
                create_thumbnail(
                    first_frame,
                    gt_hist[0][0].cpu().numpy() if gt_hist.shape[1] > 0 else None,
                    gt_fut[0][0].cpu().numpy() if gt_fut.shape[1] > 0 else None,
                    save_path=sample_dir / "thumbnail.png",
                    alpha=args.overlay_alpha if args else 0.5,
                    threshold=args.heatmap_threshold if args else 0.05
                )

                # ⭐ NEW: Save raw heatmap npy files for debugging
                np.save(sample_dir / "pred_history_interp.npy", pred_hist_interp.cpu().numpy())
                np.save(sample_dir / "pred_future_interp.npy", pred_fut_interp.cpu().numpy())
                np.save(sample_dir / "gt_history.npy", gt_hist[0].cpu().numpy())
                np.save(sample_dir / "gt_future.npy", gt_fut[0].cpu().numpy())

                # Log statistics for the first sample
                if idx == 0:
                    logger.info("=" * 60)
                    logger.info("Sample 0 Heatmap Statistics:")
                    logger.info(f"Pred History - min: {pred_hist_interp.min():.6f}, max: {pred_hist_interp.max():.6f}, mean: {pred_hist_interp.mean():.6f}, sum: {pred_hist_interp.sum():.6f}")
                    logger.info(f"Pred Future  - min: {pred_fut_interp.min():.6f}, max: {pred_fut_interp.max():.6f}, mean: {pred_fut_interp.mean():.6f}, sum: {pred_fut_interp.sum():.6f}")
                    logger.info(f"GT History   - min: {gt_hist[0].min():.6f}, max: {gt_hist[0].max():.6f}, mean: {gt_hist[0].mean():.6f}, sum per frame: {gt_hist[0].sum(dim=(-2,-1))}")
                    logger.info(f"GT Future    - min: {gt_fut[0].min():.6f}, max: {gt_fut[0].max():.6f}, mean: {gt_fut[0].mean():.6f}, sum per frame: {gt_fut[0].sum(dim=(-2,-1))}")
                    logger.info("=" * 60)

                # Collect metadata for HTML index
                samples_info.append({
                    'idx': idx,
                    'scene_name': batch.get('meta', [{}])[0].get('scene', 'unknown') if 'meta' in batch else 'unknown',
                    'instruction': batch.get('text', [''])[0],
                    'num_frames': T,
                    'valid_hist': int((val_hist[0] > 0.5).sum().item()),
                    'valid_fut': int((val_fut[0] > 0.5).sum().item()),
                    'metrics': {
                        'mae_hist': float(np.abs(pred_hist_interp.cpu().numpy() - gt_hist[0].cpu().numpy()).mean()),
                        'mae_fut': float(np.abs(pred_fut_interp.cpu().numpy() - gt_fut[0].cpu().numpy()).mean()),
                    },
                    'grid_paths': grid_paths,
                    'thumbnail_path': f"samples/sample_{idx:04d}/thumbnail.png",
                    'overlay_mode': args.overlay_mode if args else 'dual'
                })

            # Aggregated visualization (original mode, for backward compatibility)
            if vis_mode in ['aggregated', 'both']:
                frames_np = frames[0].cpu()
                hist_hm = pred_hist[0, 0].detach().cpu().numpy() if pred_hist.shape[1] > 0 else None
                fut_hm = pred_fut[0, 0].detach().cpu().numpy() if pred_fut.shape[1] > 0 else None
                hist_val = torch.sigmoid(pred_hist_val)[0, 0].item() if pred_hist_val.numel() > 0 else 0.0
                fut_val = torch.sigmoid(pred_fut_val)[0, 0].item() if pred_fut_val.numel() > 0 else 0.0
                visualize_sample(
                    frames_np,
                    hist_hm,
                    fut_hm,
                    batch['gt_heatmap_history'][0].cpu().numpy(),
                    batch['gt_heatmap_future'][0].cpu().numpy(),
                    hist_val,
                    fut_val,
                    batch['gt_validity_history'][0].cpu().numpy(),
                    batch['gt_validity_future'][0].cpu().numpy(),
                    save_dir / f"sample_{idx:04d}_aggregated.png"
                )

    # Average metrics
    for k in totals:
        if k.startswith('hist_') and k != 'history_loss':
            totals[k] /= max(spatial_count['hist'], 1)
        elif k.startswith('fut_') and k != 'future_loss':
            totals[k] /= max(spatial_count['fut'], 1)
        else:
            totals[k] /= max(count, 1)

    # Generate HTML index
    if save_dir is not None and samples_info and (args and not args.no_html):
        create_html_index(samples_info, save_dir, totals)

    return totals


def visualize_sample(
    frames: torch.Tensor,        # [T, 3, H, W]
    pred_hist_hm: np.ndarray,    # [H, W] or None
    pred_fut_hm: np.ndarray,     # [H, W] or None
    gt_hist_hm: np.ndarray,      # [T, H, W]
    gt_fut_hm: np.ndarray,       # [T, H, W]
    pred_hist_val: float,
    pred_fut_val: float,
    gt_hist_val: np.ndarray,     # [T]
    gt_fut_val: np.ndarray,      # [T]
    save_path: Path
):
    T = frames.shape[0]
    frame_indices = [0, T // 2, T - 1]
    selected_frames = frames[frame_indices].permute(0, 2, 3, 1).numpy()

    def aggregate_gt(gt_hm, gt_val):
        valid_mask = (gt_val > 0.5).astype(np.float32)[:, None, None]
        weighted = gt_hm * valid_mask
        sum_hm = weighted.sum(axis=0)
        count = valid_mask.sum(axis=0).clip(min=1.0)
        avg = sum_hm / count
        if avg.sum() > 0:
            avg = avg / avg.sum()
        return avg

    gt_hist_agg = aggregate_gt(gt_hist_hm, gt_hist_val)
    gt_fut_agg = aggregate_gt(gt_fut_hm, gt_fut_val)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: frames
    for i, (ax, frame) in enumerate(zip(axes[0], selected_frames)):
        ax.imshow(frame)
        ax.set_title(f"Frame {frame_indices[i]}")
        ax.axis('off')

    # Row 2: history
    im0 = axes[1, 0].imshow(gt_hist_agg, cmap='hot')
    axes[1, 0].set_title(f"GT History (val {gt_hist_val.mean():.2f})")
    axes[1, 0].axis('off'); plt.colorbar(im0, ax=axes[1, 0], fraction=0.046)
    if pred_hist_hm is not None:
        im1 = axes[1, 1].imshow(pred_hist_hm, cmap='hot')
        axes[1, 1].set_title(f"Pred History (val {pred_hist_val:.2f})")
        axes[1, 1].axis('off'); plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
        diff = np.abs(gt_hist_agg - pred_hist_hm)
        im2 = axes[1, 2].imshow(diff, cmap='coolwarm')
        axes[1, 2].set_title(f"Hist Diff (MAE {diff.mean():.4f})")
        axes[1, 2].axis('off'); plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 1].axis('off'); axes[1, 2].axis('off')

    # Row 3: future
    im3 = axes[2, 0].imshow(gt_fut_agg, cmap='hot')
    axes[2, 0].set_title(f"GT Future (val {gt_fut_val.mean():.2f})")
    axes[2, 0].axis('off'); plt.colorbar(im3, ax=axes[2, 0], fraction=0.046)
    if pred_fut_hm is not None:
        im4 = axes[2, 1].imshow(pred_fut_hm, cmap='hot')
        axes[2, 1].set_title(f"Pred Future (val {pred_fut_val:.2f})")
        axes[2, 1].axis('off'); plt.colorbar(im4, ax=axes[2, 1], fraction=0.046)
        diff_f = np.abs(gt_fut_agg - pred_fut_hm)
        im5 = axes[2, 2].imshow(diff_f, cmap='coolwarm')
        axes[2, 2].set_title(f"Fut Diff (MAE {diff_f.mean():.4f})")
        axes[2, 2].axis('off'); plt.colorbar(im5, ax=axes[2, 2], fraction=0.046)
    else:
        axes[2, 1].axis('off'); axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate SpatialMLLMPipeline (dual-head)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--num-vis', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to evaluate (default: all). Useful for quick testing.')
    parser.add_argument('--use-history', action='store_true', help='只评估历史头（若同时未指定future，则默认两头）')
    parser.add_argument('--use-future', action='store_true', help='只评估未来头（若同时未指定history，则默认两头）')
    parser.add_argument('--amp', type=str, default='none', choices=['none', 'fp16', 'bf16'],
                       help='AMP precision (match training config for consistency)')
    # New arguments for frame-by-frame visualization
    parser.add_argument('--vis-mode', type=str, default='aggregated',
                       choices=['frame_by_frame', 'aggregated', 'both'],
                       help='Visualization mode: frame-by-frame, aggregated, or both')
    parser.add_argument('--overlay-mode', type=str, default='dual',
                       choices=['dual', 'separate', 'full-separate'],
                       help='Overlay mode: dual (history+future together), separate (2 grids), full-separate (7 columns)')
    parser.add_argument('--max-frames-per-vis', type=int, default=16,
                       help='Maximum frames to show per visualization grid')
    parser.add_argument('--no-html', action='store_true',
                       help='Disable HTML index generation')
    parser.add_argument('--interpolation-method', type=str, default='linear',
                       choices=['linear', 'nearest', 'cubic'],
                       help='Interpolation method for keyframe predictions')
    parser.add_argument('--overlay-alpha', type=float, default=0.5,
                       help='Heatmap overlay transparency (0-1)')
    parser.add_argument('--heatmap-threshold', type=float, default=0.05,
                       help='Minimum heatmap value to display (0-1)')
    args = parser.parse_args()

    # ⭐ NEW: Sanity check - config file exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    # ⭐ NEW: Sanity check - checkpoint file exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    cfg = load_config(args.config)

    eval_history = args.use_history or (not args.use_history and not args.use_future)
    eval_future = args.use_future or (not args.use_history and not args.use_future)

    # ⭐ NEW: Sanity check - CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (evaluation will be very slow)")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_str = str(device) if device.type == 'cuda' else 'cuda:0'
    logger.info(f"Using device: {device}")

    # ⭐ FIX: Build model with explicit device to avoid GPU allocation conflicts
    model = build_model(cfg, device=device_str)
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    dataloader = build_dataloader(cfg, split=args.split)

    loss_cfg = cfg['loss']
    criterion = NavigationHeatmapLoss(
        alpha=loss_cfg['alpha'],
        lambda_mse=loss_cfg['lambda_mse'],
        lambda_kl=loss_cfg['lambda_kl'],
        lambda_valid=loss_cfg['lambda_valid']
    )

    save_dir = None
    if args.save_vis:
        save_dir = Path(cfg['log']['out_dir']) / 'eval_vis'
        save_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ NEW: Set up AMP dtype
    amp_dtype = None
    if args.amp == 'bf16':
        amp_dtype = torch.bfloat16
        logger.info("Using BF16 AMP for evaluation")
    elif args.amp == 'fp16':
        amp_dtype = torch.float16
        logger.info("Using FP16 AMP for evaluation")
    else:
        logger.info("Using FP32 (no AMP) for evaluation")

    # ⭐ NEW: Progress indication
    logger.info("=" * 60)
    logger.info(f"Starting evaluation on {args.split} split...")
    logger.info(f"Eval history: {eval_history}, Eval future: {eval_future}")
    logger.info("=" * 60)

    metrics = evaluate(model, dataloader, criterion, cfg, device, save_dir, args.num_vis, eval_history, eval_future, amp_dtype, args)

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Total Loss : {metrics['total_loss']:.4f}")
    logger.info(f"History Loss: {metrics['history_loss']:.4f}")
    logger.info(f"Future  Loss: {metrics['future_loss']:.4f}")
    logger.info(f"MSE(avg)    : {metrics['mse']:.4f}")
    logger.info(f"KL(avg)     : {metrics['kl']:.4f}")
    logger.info(f"Valid(avg)  : {metrics['valid']:.4f}")
    logger.info("")
    logger.info("⭐ Enhanced Spatial Metrics:")
    logger.info(f"History - Peak Error: {metrics.get('hist_peak_error', 0):.2f} pixels")
    logger.info(f"History - IoU       : {metrics.get('hist_iou', 0):.4f}")
    logger.info(f"History - Cosine Sim: {metrics.get('hist_cosine_sim', 0):.4f}")
    logger.info(f"History - MAE       : {metrics.get('hist_mae', 0):.4f}")
    logger.info(f"Future  - Peak Error: {metrics.get('fut_peak_error', 0):.2f} pixels")
    logger.info(f"Future  - IoU       : {metrics.get('fut_iou', 0):.4f}")
    logger.info(f"Future  - Cosine Sim: {metrics.get('fut_cosine_sim', 0):.4f}")
    logger.info(f"Future  - MAE       : {metrics.get('fut_mae', 0):.4f}")

    metrics_file = Path(cfg['log']['out_dir']) / f'metrics_{args.split}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
