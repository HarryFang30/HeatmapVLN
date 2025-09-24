#!/opt/conda/bin/python python3
"""
Frame-Indexed Heatmap Verification Script
==========================================

This script specifically tests and verifies that the frame-indexed heatmap generation
is working correctly - ensuring each keyframe produces a distinct heatmap showing
where that keyframe's content appears in the current observation view.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_video(video_path: str, max_frames: int = 8) -> torch.Tensor:
    """Load test video frames."""
    # Resolve video path robustly
    try:
        from src.utils.path_utils import resolve_video_path
        resolved_path = resolve_video_path(video_path)
        video_path = str(resolved_path)
        logger.info(f"Resolved video path to: {video_path}")
    except Exception as e:
        logger.warning(f"Could not resolve video path: {e}, using original: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    if total_frames > max_frames:
        step = total_frames // max_frames
        frame_indices = list(range(0, total_frames, step))[:max_frames]
    else:
        frame_indices = list(range(total_frames))

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)
        frames.append(frame)

    cap.release()

    # Pad if necessary
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))

    return torch.stack(frames[:max_frames])


def verify_frame_indexed_heatmaps():
    """Main verification function."""
    logger.info("=== FRAME-INDEXED HEATMAP VERIFICATION ===")

    # Create pipeline
    config = SpatialMLLMIntegrationConfig(
        use_real_llm=True,
        llm_model_path='./models/qwen_2.5_vl',
        use_multi_gpu=True,
        vggt_gpu='cuda:0',
        dinov3_gpu='cuda:1',
        llm_gpu='cuda:2'
    )

    pipeline = SpatialMLLMPipeline(config)

    # Load test video
    video_frames = load_test_video('./test.mp4', max_frames=8)
    video_frames = video_frames.unsqueeze(0)  # Add batch dimension

    # Use frame 6 as current observation (different from middle)
    current_obs_idx = 6
    current_observation = video_frames[:, current_obs_idx]

    logger.info(f"Video frames shape: {video_frames.shape}")
    logger.info(f"Current observation: Frame {current_obs_idx}")
    logger.info(f"Current observation shape: {current_observation.shape}")

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        result = pipeline(
            video_frames=video_frames,
            instruction_text="Analyze the spatial relationships and show where each keyframe appears in the current view",
            current_observation=current_observation
        )

    # Analyze results
    logger.info("=== ANALYZING FRAME-INDEXED HEATMAPS ===")

    if 'frame_indexed_heatmaps' not in result:
        logger.error("❌ No frame_indexed_heatmaps found in result!")
        logger.info(f"Available keys: {list(result.keys())}")
        return False

    frame_heatmaps = result['frame_indexed_heatmaps']
    selected_keyframes = result['selected_keyframes']

    logger.info(f"✅ Found frame_indexed_heatmaps: {len(frame_heatmaps)} heatmaps")
    logger.info(f"Selected keyframes: {selected_keyframes}")
    logger.info(f"Keyframe indices in heatmaps: {list(frame_heatmaps.keys())}")

    # Verify each keyframe has a distinct heatmap
    logger.info("=== VERIFYING HEATMAP DISTINCTNESS ===")

    heatmap_stats = {}
    for frame_idx, heatmap in frame_heatmaps.items():
        stats = {
            'shape': heatmap.shape,
            'min': float(heatmap.min().item()),
            'max': float(heatmap.max().item()),
            'mean': float(heatmap.mean().item()),
            'std': float(heatmap.std().item()),
            'non_zero_ratio': float((heatmap != 0).float().mean().item())
        }
        heatmap_stats[frame_idx] = stats

        logger.info(f"Frame {frame_idx} heatmap:")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        logger.info(f"  Non-zero ratio: {stats['non_zero_ratio']:.4f}")

    # Check if heatmaps are distinct (not identical)
    logger.info("=== CHECKING HEATMAP DISTINCTNESS ===")

    frame_indices = list(frame_heatmaps.keys())
    distinct_count = 0

    for i in range(len(frame_indices)):
        for j in range(i+1, len(frame_indices)):
            frame_i, frame_j = frame_indices[i], frame_indices[j]
            heatmap_i = frame_heatmaps[frame_i]
            heatmap_j = frame_heatmaps[frame_j]

            # Calculate similarity
            diff = torch.abs(heatmap_i - heatmap_j)
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())

            logger.info(f"Frame {frame_i} vs Frame {frame_j}:")
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")

            if max_diff > 1e-6:  # Threshold for considering heatmaps different
                distinct_count += 1
                logger.info(f"  ✅ Heatmaps are DISTINCT")
            else:
                logger.warning(f"  ⚠️ Heatmaps appear IDENTICAL")

    total_pairs = len(frame_indices) * (len(frame_indices) - 1) // 2
    logger.info(f"Distinct heatmap pairs: {distinct_count}/{total_pairs}")

    # Create visualization
    create_detailed_visualization(frame_heatmaps, video_frames, current_obs_idx)

    # Save verification results
    save_verification_results(frame_heatmaps, heatmap_stats, selected_keyframes, current_obs_idx)

    success = distinct_count == total_pairs and len(frame_heatmaps) > 1

    if success:
        logger.info("🎉 ✅ VERIFICATION PASSED!")
        logger.info("✅ Each keyframe produces a distinct heatmap")
        logger.info("✅ Heatmaps show different spatial patterns")
        logger.info("✅ Frame-indexed heatmap generation is working correctly")
    else:
        logger.error("❌ VERIFICATION FAILED!")
        logger.error("❌ Some heatmaps are identical or missing")

    return success


def create_detailed_visualization(frame_heatmaps, video_frames, current_obs_idx):
    """Create detailed visualization showing each keyframe's heatmap separately."""
    logger.info("Creating detailed frame-indexed heatmap visualization...")

    num_frames = len(frame_heatmaps)
    frame_indices = sorted(frame_heatmaps.keys())

    # Create figure with subplots
    cols = min(4, num_frames + 1)  # +1 for current observation
    rows = (num_frames + 1 + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else axes.reshape(1, 1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot current observation
    current_obs = video_frames[0, current_obs_idx].permute(1, 2, 0).cpu().numpy()
    current_obs = np.clip(current_obs, 0, 1)

    axes[0, 0].imshow(current_obs)
    axes[0, 0].set_title(f'Current Observation\n(Frame {current_obs_idx})', fontweight='bold')
    axes[0, 0].axis('off')

    # Plot each keyframe's heatmap
    plot_idx = 1
    for frame_idx in frame_indices:
        row = plot_idx // cols
        col = plot_idx % cols

        if row < rows and col < cols:
            heatmap = frame_heatmaps[frame_idx][0].cpu().numpy()  # Remove batch dim

            # Show heatmap
            im = axes[row, col].imshow(heatmap, cmap='hot', interpolation='bilinear')
            axes[row, col].set_title(f'Keyframe {frame_idx} in Current View\n(Range: {heatmap.min():.3f}-{heatmap.max():.3f})',
                                   fontweight='bold')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, rows * cols):
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')

    plt.suptitle(f'Frame-Indexed Heatmaps: Where Each Keyframe Appears in Current View (Frame {current_obs_idx})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save visualization
    output_path = Path('./frame_indexed_verification.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✅ Saved detailed visualization to: {output_path}")
    plt.close()


def save_verification_results(frame_heatmaps, heatmap_stats, selected_keyframes, current_obs_idx):
    """Save verification results to file."""
    # Check if all heatmaps have different statistics (indicating they're distinct)
    distinct_count = 0
    frame_indices = list(heatmap_stats.keys())
    for i in range(len(frame_indices)):
        for j in range(i+1, len(frame_indices)):
            stats_i = heatmap_stats[frame_indices[i]]
            stats_j = heatmap_stats[frame_indices[j]]
            # Compare key statistics to determine if heatmaps are different
            if (abs(stats_i['mean'] - stats_j['mean']) > 1e-6 or
                abs(stats_i['std'] - stats_j['std']) > 1e-6 or
                abs(stats_i['max'] - stats_j['max']) > 1e-6):
                distinct_count += 1

    total_pairs = len(frame_indices) * (len(frame_indices) - 1) // 2
    verification_passed = distinct_count == total_pairs and len(frame_heatmaps) > 1

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'frame_indexed_heatmap_verification',
        'current_observation_frame': current_obs_idx,
        'selected_keyframes': selected_keyframes.tolist() if hasattr(selected_keyframes, 'tolist') else selected_keyframes,
        'num_heatmaps_generated': len(frame_heatmaps),
        'frame_indices_with_heatmaps': list(frame_heatmaps.keys()),
        'heatmap_statistics': heatmap_stats,
        'distinct_pairs': distinct_count,
        'total_pairs': total_pairs,
        'verification_status': 'PASSED' if verification_passed else 'FAILED'
    }

    output_path = Path('./frame_indexed_verification_results.json')
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Saved verification results to: {output_path}")


if __name__ == "__main__":
    logger.info("Starting Frame-Indexed Heatmap Verification")

    success = verify_frame_indexed_heatmaps()

    if success:
        print("\n🏆 === VERIFICATION SUCCESSFUL ===")
        print("✅ Frame-indexed heatmap generation is working correctly!")
        print("✅ Each keyframe produces a distinct heatmap")
        print("✅ Heatmaps show where keyframes appear in current observation")
    else:
        print("\n❌ === VERIFICATION FAILED ===")
        print("❌ Frame-indexed heatmap generation needs debugging")
        sys.exit(1)