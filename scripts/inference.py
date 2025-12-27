#!/usr/bin/env python3
"""
Inference for SpatialMLLMPipeline (dual-head + action).
支持通过参数选择输出历史头/未来头/动作头。

更新：
- 适配新的双 DiffusionHeatmapHead 架构
- 支持 DiffusionActionHead 动作输出
- 移除已删除的 validity head 引用
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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
from src.models.pipeline import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig

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
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    logger.info(f"Video has {total_frames} frames, will sample up to {max_frames}")

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

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    if len(frames) < max_frames:
        logger.warning(f"Video only has {len(frames)} frames, duplicating last frame to reach {max_frames}")
        while len(frames) < max_frames:
            frames.append(frames[-1].clone())

    return torch.stack(frames)


def load_clip_frames(clip_dir: str, max_frames: int = 32, target_size: tuple = (224, 224)) -> torch.Tensor:
    """Load frames from a dataset clip directory.

    Args:
        clip_dir: Path to clip directory (contains rgb/ subfolder)
        max_frames: Maximum number of frames to load
        target_size: Target resolution (H, W)

    Returns:
        Tensor of shape [T, 3, H, W]
    """
    clip_path = Path(clip_dir)
    rgb_dir = clip_path / "rgb"
    
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    # Find all PNG files
    png_files = sorted(rgb_dir.glob("*.png"))
    if len(png_files) == 0:
        raise ValueError(f"No PNG files found in: {rgb_dir}")
    
    logger.info(f"Found {len(png_files)} frames in clip")
    
    frames = []
    for i, png_path in enumerate(png_files[:max_frames]):
        frame = cv2.imread(str(png_path))
        if frame is None:
            logger.warning(f"Failed to load frame: {png_path}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
    
    if len(frames) == 0:
        raise ValueError(f"Could not load any frames from: {rgb_dir}")
    
    # Pad if needed
    while len(frames) < max_frames:
        frames.append(frames[-1].clone())
    
    return torch.stack(frames[:max_frames])


def load_instruction_from_clip(clip_dir: str) -> Optional[str]:
    """Load instruction from clip's meta.json."""
    meta_path = Path(clip_dir) / "meta.json"
    if not meta_path.exists():
        return None
    
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    
    return meta.get("instruction", None)


def visualize_heatmaps(heatmaps: torch.Tensor, output_dir: str, name: str, prefix: str):
    """Visualize heatmaps and save to file."""
    os.makedirs(output_dir, exist_ok=True)
    if heatmaps.dim() == 4:
        heatmaps = heatmaps[0]  # Remove batch dim
    
    num_views = heatmaps.shape[0]
    cols = min(num_views, 4)
    rows = (num_views + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
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
    save_path = os.path.join(output_dir, f"{name}_{prefix}_heatmaps.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {prefix} heatmaps to {save_path}")


def build_model(cfg: Dict, device: str = 'cuda:0') -> SpatialMLLMPipeline:
    """Build model for inference.

    Args:
        cfg: Configuration dictionary
        device: Target device

    Returns:
        SpatialMLLMPipeline model ready for inference
    """
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    
    # 获取热力图头配置
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    
    # 使用 sliding_window 配置（新格式）
    sw_cfg = data_cfg.get('sliding_window', {})
    num_history = sw_cfg.get('num_history_sample', 8)
    
    integration_cfg = SpatialMLLMIntegrationConfig(
        target_keyframes=num_history,
        total_frames=num_history + 1,  # history + current
        sampling_method="hybrid",
        llm_model_path=model_cfg['llm']['model_path'],
        # Single GPU mode for inference
        vggt_gpu=device,
        dinov3_gpu=device,
        llm_gpu=device,
        use_multi_gpu=False,
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,
        heatmap_size=tuple(data_cfg['init_hm_size']),
        enable_inter_frame_heatmaps=True,
        # 启用双热力图头
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', True),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        # 启用动作头
        enable_action_head=action_cfg.get('enable', True),
        action_dim=action_cfg.get('action_dim', 2),
        action_pred_horizon=action_cfg.get('pred_horizon', 1),
        action_encoding_size=action_cfg.get('encoding_size', 768),
        action_num_diffusion_iters=action_cfg.get('num_diffusion_iters', 10),
        # 图像尺寸
        dinov3_img_size=data_cfg['image_size'][0],
        vggt_img_size=data_cfg['image_size'][0],
        enable_gradient_checkpointing=False,
        verbose=True
    )
    return SpatialMLLMPipeline(integration_cfg)


@torch.no_grad()
def run_inference(
    model: SpatialMLLMPipeline,
    frames: torch.Tensor,
    instruction: str,
    current_observation: Optional[torch.Tensor] = None,
    use_history: bool = True,
    use_future: bool = True,
    use_actions: bool = True,
    amp_dtype=None
) -> Dict[str, Any]:
    """Run inference.

    Args:
        model: SpatialMLLMPipeline model
        frames: Input frames [1, T, 3, H, W]
        instruction: Navigation instruction text
        current_observation: Current observation [1, 3, H, W], uses last frame if None
        use_history: Whether to return history heatmaps
        use_future: Whether to return future heatmaps
        use_actions: Whether to return predicted actions
        amp_dtype: AMP dtype (None, torch.float16, or torch.bfloat16)

    Returns:
        Dictionary with heatmaps and actions
    """
    # 使用最后一帧作为当前观测（如果未指定）
    if current_observation is None:
        current_observation = frames[:, -1]  # [1, 3, H, W]
    
    if amp_dtype is not None:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            outputs = model(
                video_frames=frames,
                instruction_text=instruction,
                current_observation=current_observation,
                return_heatmaps=True,
                return_actions=use_actions,
            )
    else:
        outputs = model(
            video_frames=frames,
            instruction_text=instruction,
            current_observation=current_observation,
            return_heatmaps=True,
            return_actions=use_actions,
        )

    results = {}
    
    # 热力图输出
    if use_history and 'history_heatmaps' in outputs:
        results['history_heatmaps'] = outputs['history_heatmaps']
    
    if use_future and 'future_heatmaps' in outputs:
        results['future_heatmaps'] = outputs['future_heatmaps']
    
    # 动作输出
    if use_actions and 'actions' in outputs:
        results['actions'] = outputs['actions']
        logger.info(f"Predicted actions: {outputs['actions'].cpu().numpy()}")
    
    # 元数据
    if 'processing_metadata' in outputs:
        results['metadata'] = outputs['processing_metadata']
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SpatialMLLMPipeline Inference (Dual-Head + Action)")
    parser.add_argument('--config', type=str, default='configs/training_config_full_model.yaml')
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--clip', type=str, default=None, help='Path to dataset clip directory')
    parser.add_argument('--instruction', type=str, default=None, help='Navigation instruction')
    parser.add_argument('--output-dir', type=str, default='./outputs_inference')
    parser.add_argument('--use-history', action='store_true', help='Output history heatmaps')
    parser.add_argument('--use-future', action='store_true', help='Output future heatmaps')
    parser.add_argument('--use-actions', action='store_true', help='Output predicted actions')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--amp', type=str, default='bf16', choices=['none', 'fp16', 'bf16'],
                       help='AMP precision')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Validate inputs
    if args.video is None and args.clip is None:
        logger.error("Either --video or --clip must be specified")
        return
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    cfg = yaml.safe_load(open(args.config))

    # Default: output all heads
    if not args.use_history and not args.use_future and not args.use_actions:
        args.use_history = True
        args.use_future = True
        args.use_actions = True

    # Device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (this will be slow)")
        args.device = 'cpu'
    logger.info(f"Using device: {args.device}")

    # Build model
    logger.info("Building model...")
    model = build_model(cfg, device=args.device)

    # Load checkpoint (optional)
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt.get('trainable_state_dict', ckpt))
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.info(f"Missing keys (using pretrained): {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        state = ckpt.get('model_state_dict', ckpt.get('trainable_state_dict', ckpt))
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}

        # Load with strict=False to allow partial loading
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.info(f"Missing keys (using pretrained): {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.info("Checkpoint loaded successfully")
    elif args.checkpoint:
        logger.warning(f"Checkpoint not found: {args.checkpoint}, using pretrained weights only")

    model = model.to(args.device)
    model.eval()

    # Load frames
    data_cfg = cfg['data']
    sw_cfg = data_cfg.get('sliding_window', {})
    max_frames = sw_cfg.get('num_history_sample', 8) + 1
    target_size = tuple(data_cfg['image_size'])
    
    if args.video:
        frames = load_video_frames(args.video, max_frames=max_frames, target_size=target_size)
        name = Path(args.video).stem
        instruction = args.instruction
    else:
        frames = load_clip_frames(args.clip, max_frames=max_frames, target_size=target_size)
        name = Path(args.clip).name
        instruction = args.instruction or load_instruction_from_clip(args.clip)
    
    if instruction is None:
        instruction = "Navigate to the destination"
        logger.info(f"No instruction provided, using default: '{instruction}'")
    else:
        logger.info(f"Instruction: '{instruction}'")
    
    frames = frames.unsqueeze(0).to(args.device)  # [1, T, 3, H, W]

    # AMP dtype
    amp_dtype = None
    if args.amp == 'bf16':
        amp_dtype = torch.bfloat16
        logger.info("Using BF16 AMP")
    elif args.amp == 'fp16':
        amp_dtype = torch.float16
        logger.info("Using FP16 AMP")

    # Run inference
    logger.info("=" * 60)
    logger.info("Running inference...")
    logger.info("=" * 60)
    
    results = run_inference(
        model, frames, instruction,
        use_history=args.use_history,
        use_future=args.use_future,
        use_actions=args.use_actions,
        amp_dtype=amp_dtype
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if 'history_heatmaps' in results:
        hm = results['history_heatmaps']
        logger.info(f"Generated history heatmaps: shape={hm.shape}")
        visualize_heatmaps(hm, str(out_dir), name, 'history')

    if 'future_heatmaps' in results:
        hm = results['future_heatmaps']
        logger.info(f"Generated future heatmaps: shape={hm.shape}")
        visualize_heatmaps(hm, str(out_dir), name, 'future')

    if 'actions' in results:
        actions = results['actions'].cpu().numpy()
        logger.info(f"Predicted actions shape: {actions.shape}")
        # Save actions to file
        np.save(out_dir / f"{name}_actions.npy", actions)
        logger.info(f"Saved actions to {out_dir}/{name}_actions.npy")

    logger.info("=" * 60)
    logger.info(f"Inference complete! Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
