#!/usr/bin/env python3
"""
Inference for SpatialMLLMPipeline (dual-head).
支持通过参数选择输出历史头/未来头/同时输出。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting_config import configure_matplotlib_fonts
from src.utils.logger import setup_logger
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig

configure_matplotlib_fonts()
logger = logging.getLogger("inference")


def load_video_frames(video_path: str, max_frames: int = 32, target_size: tuple = (224, 224)) -> torch.Tensor:
    """Load and preprocess video frames with robust error handling.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_size: Target resolution (H, W)

    Returns:
        Tensor of shape [T, 3, H, W] with T = max_frames

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video can't be opened
        ValueError: If video has no valid frames
    """
    # ⭐ FIX: Validate input before processing
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get total frame count for logging
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    logger.info(f"Video has {total_frames} frames, will sample up to {max_frames}")

    # Read frames
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC -> CHW
        frames.append(frame)

    cap.release()

    # ⭐ FIX: Validate we got frames
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # ⭐ FIX: Handle insufficient frames (duplicate last frame instead of zeros)
    if len(frames) < max_frames:
        logger.warning(f"Video only has {len(frames)} frames, duplicating last frame to reach {max_frames}")
        while len(frames) < max_frames:
            frames.append(frames[-1].clone())  # Duplicate last valid frame

    return torch.stack(frames)


def visualize_heatmaps(heatmaps: torch.Tensor, output_dir: str, name: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    if heatmaps.dim() == 4:
        heatmaps = heatmaps[0]
    num_views = heatmaps.shape[0]
    cols = min(num_views, 4)
    rows = (num_views + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(-1, 1)
    for i in range(num_views):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        hm = heatmaps[i].cpu().numpy()
        im = ax.imshow(hm, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'{prefix} {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(num_views, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_{prefix}_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close()


def build_model(cfg: Dict, device: str = 'cuda:0') -> SpatialMLLMPipeline:
    """Build model for inference (single GPU)

    Args:
        cfg: Configuration dictionary
        device: Target device (default: 'cuda:0')

    Returns:
        SpatialMLLMPipeline model ready for inference
    """
    model_cfg = cfg['model']
    integration_cfg = SpatialMLLMIntegrationConfig(
        target_keyframes=cfg['data']['heatmap_per_clip'],
        total_frames=cfg['data']['frames_per_clip'],
        sampling_method="hybrid",
        llm_model_path=model_cfg['llm']['model_path'],
        # ⭐ FIX: Use single device for all components during inference
        vggt_gpu=device,
        dinov3_gpu=device,
        llm_gpu=device,
        use_multi_gpu=False,          # Correct: single GPU mode for inference
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_inter_frame_heatmaps=True,
        dinov3_img_size=cfg['data']['image_size'][0],
        vggt_img_size=cfg['data']['image_size'][0],
        enable_gradient_checkpointing=False,  # Inference doesn't need gradient checkpointing
        verbose=True
    )
    return SpatialMLLMPipeline(integration_cfg)


@torch.no_grad()
def run_inference(model: SpatialMLLMPipeline, frames: torch.Tensor, instruction: str,
                  use_history: bool, use_future: bool, amp_dtype=None):
    """Run inference with optional AMP

    Args:
        model: SpatialMLLMPipeline model
        frames: Input frames [1, T, 3, H, W]
        instruction: Navigation instruction text
        use_history: Whether to return history heatmaps
        use_future: Whether to return future heatmaps
        amp_dtype: AMP dtype (None, torch.float16, or torch.bfloat16)

    Returns:
        Dictionary with heatmaps and validity scores
    """
    # ⭐ NEW: AMP support for faster inference
    if amp_dtype is not None:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            outputs = model(frames, instruction_text=instruction, return_heatmaps=True)
    else:
        outputs = model(frames, instruction_text=instruction, return_heatmaps=True)

    results = {}
    if use_history and 'history_heatmaps' in outputs:
        results['history_heatmaps'] = outputs['history_heatmaps']
        results['history_validity'] = torch.sigmoid(outputs['history_validity'])
    if use_future and 'future_heatmaps' in outputs:
        results['future_heatmaps'] = outputs['future_heatmaps']
        results['future_validity'] = torch.sigmoid(outputs['future_validity'])
    return results


def main():
    parser = argparse.ArgumentParser(description="SpatialMLLMPipeline Inference")
    parser.add_argument('--config', type=str, default='configs/training_config_full_model.yaml')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--instruction', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./outputs_inference')
    parser.add_argument('--use-history', action='store_true', help='输出历史头')
    parser.add_argument('--use-future', action='store_true', help='输出未来头')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--amp', type=str, default='none', choices=['none', 'fp16', 'bf16'],
                       help='AMP precision (match training config for consistency)')
    args = parser.parse_args()

    # ⭐ NEW: Enhanced logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ⭐ NEW: Sanity check - config file exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    cfg = yaml.safe_load(open(args.config))

    if not args.use_history and not args.use_future:
        args.use_history = True
        args.use_future = True

    # ⭐ NEW: Sanity check - CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (this will be slow)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = str(device) if device.type == 'cuda' else 'cuda:0'
    logger.info(f"Using device: {device_str}")

    # ⭐ FIX: Build model with explicit device to avoid GPU allocation conflicts
    model = build_model(cfg, device=device_str)

    # ⭐ NEW: Sanity check - checkpoint file exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    # ⭐ FIX: Use strict checkpoint loading to catch architecture mismatches
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        logger.info(f"Successfully loaded checkpoint: {args.checkpoint}")
    except RuntimeError as e:
        logger.error(f"Checkpoint loading failed: {e}")
        logger.error("This usually means architecture mismatch between checkpoint and model.")
        raise

    model = model.to(device)
    model.eval()

    frames = load_video_frames(args.video, max_frames=cfg['data']['frames_per_clip'],
                               target_size=tuple(cfg['data']['image_size']))
    frames = frames.unsqueeze(0).to(device)
    instruction = args.instruction or "Analyze spatial relationships between frames"

    # ⭐ NEW: Set up AMP dtype
    amp_dtype = None
    if args.amp == 'bf16':
        amp_dtype = torch.bfloat16
        logger.info("Using BF16 AMP for inference")
    elif args.amp == 'fp16':
        amp_dtype = torch.float16
        logger.info("Using FP16 AMP for inference")
    else:
        logger.info("Using FP32 (no AMP) for inference")

    # ⭐ NEW: Progress indication
    logger.info("=" * 60)
    logger.info("Running inference...")
    logger.info("=" * 60)
    results = run_inference(model, frames, instruction, args.use_history, args.use_future, amp_dtype)

    # ⭐ NEW: Validate heatmap outputs
    name = Path(args.video).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_history and 'history_heatmaps' in results:
        hm = results['history_heatmaps']
        logger.info(f"Generated history heatmaps: shape={hm.shape}")
        visualize_heatmaps(hm, str(out_dir), name, 'history')
        logger.info(f"Saved history heatmaps to {out_dir}/{name}_history_heatmaps.png")

    if args.use_future and 'future_heatmaps' in results:
        hm = results['future_heatmaps']
        logger.info(f"Generated future heatmaps: shape={hm.shape}")
        visualize_heatmaps(hm, str(out_dir), name, 'future')
        logger.info(f"Saved future heatmaps to {out_dir}/{name}_future_heatmaps.png")

    logger.info("=" * 60)
    logger.info(f"Inference complete! All results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
