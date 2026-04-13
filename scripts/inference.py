#!/usr/bin/env python3
"""
VLN Pipeline 推理脚本
======================

推荐通过统一入口调用：

    python scripts/run.py inference ...

也兼容直接调用：

    python scripts/inference.py ...
"""

import argparse
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lora_utils import resolve_lora_layer_indices
from src.models.pipeline import VLNPipeline, VLNPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("inference")


def load_video_frames(video_path: str, max_frames: int = 32, target_size: tuple = (224, 224)) -> torch.Tensor:
    """Load and preprocess video frames.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_size: Target resolution (H, W)

    Returns:
        Tensor of shape [T, 3, H, W]
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

    png_files = sorted(rgb_dir.glob("*.png"))
    if len(png_files) == 0:
        raise ValueError(f"No PNG files found in: {rgb_dir}")

    logger.info(f"Found {len(png_files)} frames in clip")

    frames = []
    for _i, png_path in enumerate(png_files[:max_frames]):
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

    while len(frames) < max_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames[:max_frames])


def load_instruction_from_clip(clip_dir: str) -> str | None:
    """Load instruction from clip's meta.json."""
    meta_path = Path(clip_dir) / "meta.json"
    if not meta_path.exists():
        return None

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    return meta.get("instruction", None)


def build_model(cfg: dict, device: str = 'cuda:0') -> VLNPipeline:
    """Build VLN pipeline for inference.

    Args:
        cfg: Configuration dictionary
        device: Target device

    Returns:
        VLNPipeline model ready for inference
    """
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
    nextdit_cfg = action_cfg.get('nextdit', {})
    resolved_lora_layers = resolve_lora_layer_indices(llm_cfg, heatmap_cfg, logger=logger)

    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get('model_path', './models/internnav_backbone'),
        llm_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 3584),
        llm_token_dim=llm_cfg.get('token_dim', 896),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        max_video_frames=llm_cfg.get('max_video_frames', -1),
        llm_enable_internal_profiling=llm_cfg.get('enable_internal_profiling', False),
        llm_enable_compile=llm_cfg.get('enable_compile', False),
        llm_compile_mode=llm_cfg.get('compile_mode', 'reduce-overhead'),
        llm_compile_backend=llm_cfg.get('compile_backend', 'inductor'),
        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        internnav_system1_path=nextdit_cfg.get('internnav_system1_path', ''),
        device=device,

        enable_heatmap=heatmap_cfg.get('enable', True),
        heatmap_c_vit=heatmap_cfg.get('c_vit', 1280),
        heatmap_c_llm=heatmap_cfg.get('c_llm', 3584),
        heatmap_c_fused=heatmap_cfg.get('c_fused', 256),
        heatmap_vit_layer_indices=heatmap_cfg.get('vit_layer_indices', [7, 15, 23, 31]),
        heatmap_llm_layer_indices=heatmap_cfg.get('llm_layer_indices', [6, 13, 20]),
        heatmap_size=tuple(heatmap_cfg.get('heatmap_size', data_cfg['init_hm_size'])),
        image_size=heatmap_cfg.get('image_size', data_cfg['image_size'][0]),
        heatmap_lambda_vis=heatmap_cfg.get('lambda_vis', 1.0),
        heatmap_lambda_coord=heatmap_cfg.get('lambda_coord', 1.0),
        heatmap_lambda_kl=heatmap_cfg.get('lambda_kl', heatmap_cfg.get('lambda_pos', 1.0)),
        heatmap_lambda_peak=heatmap_cfg.get('lambda_peak', 1.0),
        heatmap_trajectory_config=heatmap_cfg.get('trajectory', None),

        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_layer_indices=resolved_lora_layers,
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),

        enable_action_head=action_cfg.get('enable', True),
        nextdit_enabled=nextdit_cfg.get('enabled', False),
        nextdit_vlm_hidden_dim=nextdit_cfg.get('vlm_hidden_dim', 3584),
        nextdit_latent_emb_size=nextdit_cfg.get('latent_emb_size', 768),
        nextdit_n_query=nextdit_cfg.get('n_query', 4),
        nextdit_dit_dim=nextdit_cfg.get('dit_dim', 384),
        nextdit_dit_layers=nextdit_cfg.get('dit_layers', 12),
        nextdit_dit_heads=nextdit_cfg.get('dit_heads', 6),
        nextdit_dit_kv_heads=nextdit_cfg.get('dit_kv_heads', 6),
        nextdit_dit_ffn_dim_multiplier=nextdit_cfg.get('dit_ffn_dim_multiplier', 2 / 3),
        nextdit_predict_steps=nextdit_cfg.get('predict_steps', 32),
        nextdit_action_dim=nextdit_cfg.get('action_dim', 3),
        nextdit_num_inference_steps=nextdit_cfg.get('num_inference_steps', 10),
        nextdit_guidance_scale=nextdit_cfg.get('guidance_scale', 1.0),
        nextdit_num_sample_trajs=nextdit_cfg.get('num_sample_trajs', 32),
        nextdit_dav2_ckpt_path=nextdit_cfg.get('dav2_ckpt_path', ''),
        nextdit_enable_gradient_checkpointing=nextdit_cfg.get('enable_gradient_checkpointing', True),

        verbose=True,
    )

    return VLNPipeline(config)


def visualize_heatmap(heatmap: np.ndarray, output_path: str, title: str = "Heatmap"):
    """Visualize and save a single heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
    plt.colorbar(label='Probability')
    plt.title(f"{title} (max={heatmap.max():.3f})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved heatmap to {output_path}")


def visualize_trajectory(trajectory: np.ndarray, output_path: str, title: str = "Trajectory"):
    """Visualize and save trajectory prediction."""
    # Cumulative sum to get positions
    positions = np.cumsum(trajectory[:, :2], axis=0)

    plt.figure(figsize=(10, 8))

    # Plot trajectory
    plt.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, linewidth=2, label='Trajectory')

    # Mark start and end
    plt.scatter([0], [0], c='green', s=200, marker='*', zorder=5, label='Start')
    plt.scatter([positions[-1, 0]], [positions[-1, 1]], c='red', s=200, marker='X', zorder=5, label='End')

    # Add step numbers
    for i, (x, y) in enumerate(positions[::4]):  # Every 4th point
        plt.annotate(str(i*4), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel('X displacement')
    plt.ylabel('Y displacement')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved trajectory to {output_path}")


def visualize_combined(
    current_frame: np.ndarray,
    heatmap: np.ndarray | None,
    trajectory: np.ndarray | None,
    progress: float | None,
    output_path: str,
    instruction: str = "",
):
    """Create combined visualization."""
    _fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Current frame
    axes[0, 0].imshow(current_frame)
    axes[0, 0].set_title("Current Observation")
    axes[0, 0].axis('off')

    # Heatmap
    if heatmap is not None:
        im = axes[0, 1].imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
        axes[0, 1].set_title(f"History Heatmap (max={heatmap.max():.3f})")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    else:
        axes[0, 1].text(0.5, 0.5, "No heatmap", ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')

    # Trajectory
    if trajectory is not None:
        positions = np.cumsum(trajectory[:, :2], axis=0)
        axes[1, 0].plot(positions[:, 0], positions[:, 1], 'b-o', markersize=3, linewidth=2)
        axes[1, 0].scatter([0], [0], c='green', s=150, marker='*', zorder=5, label='Start')
        axes[1, 0].scatter([positions[-1, 0]], [positions[-1, 1]], c='red', s=150, marker='X', zorder=5, label='End')
        axes[1, 0].set_title(f"Predicted Trajectory ({len(trajectory)} steps)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal')
    else:
        axes[1, 0].text(0.5, 0.5, "No trajectory", ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')

    # Info panel
    info_text = f"Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"Instruction: {instruction}"
    if progress is not None:
        info_text += f"\n\nProgress: {progress:.2%}"

    axes[1, 1].text(0.1, 0.7, info_text, fontsize=12, verticalalignment='top',
                    transform=axes[1, 1].transAxes, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Inference Info")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined visualization to {output_path}")


@torch.no_grad()
def run_inference(
    model: VLNPipeline,
    frames: torch.Tensor,
    instruction: str,
    current_observation: torch.Tensor | None = None,
    output_heatmap: bool = True,
    output_trajectory: bool = True,
    output_progress: bool = True,
) -> dict[str, Any]:
    """Run inference.

    Args:
        model: VLNPipeline model
        frames: Input frames [1, T, 3, H, W]
        instruction: Navigation instruction text
        current_observation: Current observation [1, 3, H, W], uses last frame if None
        output_heatmap: Whether to generate heatmap
        output_trajectory: Whether to generate trajectory
        output_progress: Whether to predict progress

    Returns:
        Dictionary with predictions
    """
    if current_observation is None:
        current_observation = frames[:, -1]
    if output_heatmap:
        raise ValueError(
            "当前 `scripts/run.py inference` 只接受单路视频/clip 帧，无法为 HeatmapVLN v2 "
            "构造 `current_views` 和 `history_panoramas`。请改用 `scripts/run.py visualize heatmap` "
            "或基于数据集批次进行热力图推理。"
        )

    device = next(model.parameters()).device

    autocast_context = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == 'cuda'
        else nullcontext()
    )
    with autocast_context:
        outputs = model(
            video_frames=frames.to(device),
            instruction_text=instruction,
            current_observation=current_observation.to(device),
            return_heatmaps=output_heatmap,
            return_actions=output_trajectory,
        )

    results = {}

    # Trajectory
    if output_trajectory:
        trajectory = outputs.get('trajectory')
        if trajectory is not None:
            results['trajectory'] = trajectory[0].cpu().numpy()
            logger.info(f"Generated trajectory: shape={results['trajectory'].shape}")
    if 'processing_metadata' in outputs:
        results['metadata'] = outputs['processing_metadata']

    return results


def main():
    parser = argparse.ArgumentParser(description="VLN Pipeline Inference")
    parser.add_argument('--config', type=str, default='configs/train_config_internnav.yaml')
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--clip', type=str, default=None, help='Path to dataset clip directory')
    parser.add_argument('--instruction', type=str, default=None, help='Navigation instruction')
    parser.add_argument('--output-dir', type=str, default='./outputs_inference')
    parser.add_argument('--output-heatmap', action='store_true', help='Output heatmap (需要全景 dataset 输入，当前脚本默认不支持)')
    parser.add_argument('--output-trajectory', action='store_true', help='Output predicted trajectory')
    parser.add_argument('--output-progress', action='store_true', help='Output progress prediction')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()

    if args.video is None and args.clip is None:
        logger.error("Either --video or --clip must be specified")
        return

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    cfg = yaml.safe_load(open(args.config))

    # Default: output trajectory + progress
    if not args.output_heatmap and not args.output_trajectory and not args.output_progress:
        args.output_trajectory = True
        args.output_progress = True

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
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('trainable_state_dict', ckpt))
        if next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.info(f"Missing keys (using pretrained): {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        logger.info("Checkpoint loaded successfully")
    elif args.checkpoint:
        logger.warning(f"Checkpoint not found: {args.checkpoint}, using pretrained weights only")

    model = model.to(args.device)
    model.eval()

    # Load frames
    data_cfg = cfg['data']
    sw_cfg = data_cfg.get('sliding_window', {})
    traj_cfg = data_cfg.get('trajectory', {})
    max_frames = traj_cfg.get('num_history_sample', sw_cfg.get('num_history_sample', 32)) + 1
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

    # Get current frame for visualization
    current_frame_np = frames[-1].permute(1, 2, 0).numpy()
    current_frame_np = np.clip(current_frame_np, 0, 1)

    frames = frames.unsqueeze(0)  # [1, T, 3, H, W]

    # Run inference
    logger.info("=" * 60)
    logger.info("Running inference...")
    logger.info("=" * 60)

    results = run_inference(
        model, frames, instruction,
        output_heatmap=args.output_heatmap,
        output_trajectory=args.output_trajectory,
        output_progress=args.output_progress,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save individual outputs
    if 'heatmap' in results:
        visualize_heatmap(results['heatmap'], str(out_dir / f"{name}_heatmap.png"), "History Heatmap")
        np.save(out_dir / f"{name}_heatmap.npy", results['heatmap'])

    if 'trajectory' in results:
        visualize_trajectory(results['trajectory'], str(out_dir / f"{name}_trajectory.png"), "Predicted Trajectory")
        np.save(out_dir / f"{name}_trajectory.npy", results['trajectory'])

        # Also save as readable format
        with open(out_dir / f"{name}_trajectory.txt", 'w') as f:
            f.write(f"# Predicted trajectory ({len(results['trajectory'])} steps)\n")
            f.write("# dx, dy, dyaw\n")
            for step in results['trajectory']:
                f.write(f"{step[0]:.6f}, {step[1]:.6f}, {step[2]:.6f}\n")

    # Save combined visualization
    visualize_combined(
        current_frame_np,
        results.get('heatmap'),
        results.get('trajectory'),
        results.get('progress'),
        str(out_dir / f"{name}_combined.png"),
        instruction,
    )

    # Save summary
    summary = {
        'instruction': instruction,
        'progress': results.get('progress'),
        'heatmap_max': float(results['heatmap'].max()) if 'heatmap' in results else None,
        'trajectory_steps': len(results['trajectory']) if 'trajectory' in results else None,
    }
    with open(out_dir / f"{name}_summary.yaml", 'w') as f:
        yaml.dump(summary, f)

    logger.info("=" * 60)
    logger.info(f"Inference complete! Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
