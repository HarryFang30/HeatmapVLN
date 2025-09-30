"""
Evaluation Script for VLN Heatmap Model

Evaluates trained model and outputs:
- NLL/KL metrics (consistent with training loss)
- Peak localization error (argmax pixel distance)
- Visualization overlays

Usage:
    python scripts/eval_heatmap.py --config configs/training_config.yaml --ckpt path/to/checkpoint.pt
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import logging

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: str, model: nn.Module, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Checkpoint loaded from {ckpt_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Stage: {checkpoint.get('stage', 'N/A')}")
    logger.info(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    return model


def compute_peak_error(pred_hm: torch.Tensor, target_hm: torch.Tensor) -> float:
    """
    Compute peak localization error (argmax pixel distance).

    Args:
        pred_hm: Predicted heatmap [Hm, Wm]
        target_hm: Target heatmap [Hm, Wm]

    Returns:
        float: Euclidean distance between peaks in pixels
    """
    # Find argmax positions
    pred_flat = pred_hm.view(-1)
    target_flat = target_hm.view(-1)

    pred_idx = torch.argmax(pred_flat).item()
    target_idx = torch.argmax(target_flat).item()

    # Convert to 2D coordinates
    H, W = pred_hm.shape
    pred_y, pred_x = pred_idx // W, pred_idx % W
    target_y, target_x = target_idx // W, target_idx % W

    # Euclidean distance
    distance = np.sqrt((pred_y - target_y)**2 + (pred_x - target_x)**2)

    return distance


def visualize_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay visualization of heatmap on frame.

    Args:
        frame: RGB frame [H, W, 3] in range [0, 255]
        heatmap: Heatmap [Hm, Wm] in range [0, 1]
        alpha: Overlay transparency

    Returns:
        np.ndarray: Overlay image [H, W, 3]
    """
    H, W = frame.shape[:2]

    # Resize heatmap to frame size
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Convert heatmap to color (jet colormap)
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Blend with frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                  vis_dir: Path = None, max_vis: int = 20):
    """
    Evaluate model on dataset.

    Args:
        model: VLNHeatmapModel
        dataloader: Evaluation dataloader
        device: Device to run on
        vis_dir: Directory to save visualizations (optional)
        max_vis: Maximum number of visualizations to save

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    # Metrics
    total_nll = 0.0
    total_peak_errors = []
    num_samples = 0
    num_valid_heatmaps = 0

    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)        # [B, T, 3, H, W]
            targets = batch['gt_heatmaps'].to(device)  # [B, K, Hm, Wm]
            mask = batch['mask'].to(device)            # [B, K]

            # Forward pass
            preds, aux_outputs = model(frames)

            # Compute NLL loss
            nll = kl_ce_loss(preds, targets, mask=mask)
            total_nll += nll.item()

            # Compute peak errors for valid heatmaps
            B, K = mask.shape
            for b in range(B):
                for k in range(K):
                    if mask[b, k] > 0.5:  # Valid heatmap
                        pred_hm = preds[b, k]
                        target_hm = targets[b, k]

                        # Convert target to probability distribution
                        target_prob = torch.softmax(target_hm.view(-1), dim=0).view_as(target_hm)

                        # Compute peak error
                        peak_error = compute_peak_error(pred_hm, target_prob)
                        total_peak_errors.append(peak_error)
                        num_valid_heatmaps += 1

            num_samples += 1

            # Visualize samples
            if vis_dir is not None and vis_count < max_vis:
                for b in range(min(B, max_vis - vis_count)):
                    # Use middle frame for visualization
                    mid_frame_idx = frames.shape[1] // 2
                    frame = frames[b, mid_frame_idx].cpu().numpy()  # [3, H, W]
                    frame = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]

                    # Visualize each heatmap
                    for k in range(K):
                        if mask[b, k] > 0.5:
                            pred_hm = preds[b, k].cpu().numpy()
                            target_hm = torch.softmax(targets[b, k].view(-1), dim=0).view_as(targets[b, k]).cpu().numpy()

                            # Create side-by-side visualization
                            overlay_pred = visualize_heatmap(frame.copy(), pred_hm, alpha=0.5)
                            overlay_target = visualize_heatmap(frame.copy(), target_hm, alpha=0.5)

                            # Concatenate
                            vis_img = np.hstack([frame, overlay_target, overlay_pred])

                            # Add text labels
                            cv2.putText(vis_img, 'Frame', (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(vis_img, 'Target', (frame.shape[1] + 10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(vis_img, 'Prediction', (frame.shape[1] * 2 + 10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # Save
                            vis_path = vis_dir / f"vis_{vis_count}_heatmap_{k}.png"
                            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

                    vis_count += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Aggregate metrics
    avg_nll = total_nll / max(num_samples, 1)
    avg_peak_error = np.mean(total_peak_errors) if total_peak_errors else 0.0
    std_peak_error = np.std(total_peak_errors) if total_peak_errors else 0.0
    median_peak_error = np.median(total_peak_errors) if total_peak_errors else 0.0

    metrics = {
        'nll': avg_nll,
        'peak_error_mean': avg_peak_error,
        'peak_error_std': std_peak_error,
        'peak_error_median': median_peak_error,
        'num_samples': num_samples,
        'num_valid_heatmaps': num_valid_heatmaps
    }

    return metrics


def main(args):
    """Main evaluation function"""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("VLN Heatmap Model Evaluation")
    logger.info("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataset and dataloader
    logger.info("\nCreating dataset...")
    # Use highest resolution from config
    hm_size = tuple(config['training']['stages'][-1]['hm_size'])

    dataset = VLNHeatmapDataset(
        root=config['data']['root'],
        split='val',
        frames_per_clip=config['data']['frames_per_clip'],
        heatmap_per_clip=config['data']['heatmap_per_clip'],
        image_size=tuple(config['data']['image_size']),
        hm_size=hm_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['optim']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples, hm_size={hm_size}")

    # Create model
    logger.info("\nCreating model...")
    model = VLNHeatmapModel(
        k_heatmaps=config['data']['heatmap_per_clip'],
        hm_size=hm_size,
        vision_dim=config['model']['vision_dim'],
        agg=config['model']['agg'],
        models_root=config['model']['models_root']
    ).to(device)

    # Load checkpoint
    logger.info(f"\nLoading checkpoint: {args.ckpt}")
    model = load_checkpoint(args.ckpt, model, device)

    # Create visualization directory
    vis_dir = None
    if args.save_vis:
        vis_dir = Path(config['log']['out_dir']) / 'eval_vis'
        vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {vis_dir}")

    # Evaluate
    logger.info("\nEvaluating model...")
    metrics = evaluate_model(model, dataloader, device, vis_dir=vis_dir, max_vis=args.max_vis)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"NLL (KL-CE Loss):           {metrics['nll']:.6f}")
    logger.info(f"Peak Error (mean ± std):    {metrics['peak_error_mean']:.2f} ± {metrics['peak_error_std']:.2f} pixels")
    logger.info(f"Peak Error (median):        {metrics['peak_error_median']:.2f} pixels")
    logger.info(f"Samples evaluated:          {metrics['num_samples']}")
    logger.info(f"Valid heatmaps:             {metrics['num_valid_heatmaps']}")
    logger.info("=" * 80)

    if vis_dir:
        logger.info(f"\nVisualizations saved to: {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLN heatmap model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--save-vis', action='store_true',
                       help='Save visualization overlays')
    parser.add_argument('--max-vis', type=int, default=20,
                       help='Maximum number of visualizations to save')

    args = parser.parse_args()
    main(args)