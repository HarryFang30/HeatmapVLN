"""
Dataset Packing Script (pack_dataset.py)
========================================

Convert raw sequences to standard training format following dataset.md specifications.

Process:
1. Read raw sequences (RGB/Depth/Poses/Intrinsics)
2. Apply frame sampling and keyframe selection
3. Generate frame-indexed heatmaps via geometric projection
4. Pack into standard training dataset format

Usage:
    python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
    python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
    python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split test
"""

import os
import sys
import json
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.heatmap_builder import (
    build_intrinsics, uniform_keyframe_indices, fps_keyframe_indices,
    project_keyframe_to_ref, heatmap_from_points
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_clip_data(clip_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict]:
    """
    Load raw clip data: RGB images, depth maps, poses, and intrinsics.

    Args:
        clip_dir: Path to raw clip directory

    Returns:
        tuple: (rgb_frames, depth_maps, poses, intrinsics)
    """
    rgb_dir = clip_dir / "rgb"
    depth_dir = clip_dir / "depth"
    poses_file = clip_dir / "poses.json"
    intrinsics_file = clip_dir / "intrinsics.json"

    # Check required files exist
    if not all([rgb_dir.exists(), depth_dir.exists(), poses_file.exists(), intrinsics_file.exists()]):
        raise FileNotFoundError(f"Missing required files in {clip_dir}")

    # Load poses
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    poses = [np.array(pose, dtype=np.float32) for pose in poses_data]

    # Load intrinsics
    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)

    # Load RGB frames
    rgb_files = sorted(rgb_dir.glob("*.png"))
    rgb_frames = []
    for rgb_file in rgb_files:
        rgb = cv2.imread(str(rgb_file))
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb)

    # Load depth maps
    depth_files = sorted(depth_dir.glob("*.npy"))
    depth_maps = []
    for depth_file in depth_files:
        depth = np.load(depth_file).astype(np.float32)
        depth_maps.append(depth)

    if len(rgb_frames) != len(depth_maps) or len(rgb_frames) != len(poses):
        logger.warning(f"Mismatched data lengths in {clip_dir}: "
                      f"RGB={len(rgb_frames)}, Depth={len(depth_maps)}, Poses={len(poses)}")

    return rgb_frames, depth_maps, poses, intrinsics


def extract_subsequences(rgb_frames: List[np.ndarray], depth_maps: List[np.ndarray],
                        poses: List[np.ndarray], frames_per_clip: int, stride: int) -> List[Tuple]:
    """
    Extract subsequences using sliding window.

    Args:
        rgb_frames, depth_maps, poses: Input data sequences
        frames_per_clip: Length of each subsequence (T)
        stride: Step size for sliding window

    Returns:
        List of (rgb_sub, depth_sub, poses_sub) subsequences
    """
    subsequences = []
    total_frames = min(len(rgb_frames), len(depth_maps), len(poses))

    start_idx = 0
    while start_idx + frames_per_clip <= total_frames:
        end_idx = start_idx + frames_per_clip

        rgb_sub = rgb_frames[start_idx:end_idx]
        depth_sub = depth_maps[start_idx:end_idx]
        poses_sub = poses[start_idx:end_idx]

        subsequences.append((rgb_sub, depth_sub, poses_sub))
        start_idx += stride

    return subsequences


def select_keyframes(poses: List[np.ndarray], config: Dict, ref_idx: int) -> List[int]:
    """
    Select keyframes using configured sampling strategy.

    Args:
        poses: List of 4x4 transformation matrices
        config: Pack configuration
        ref_idx: Reference frame index

    Returns:
        List[int]: Selected keyframe indices
    """
    sampler = config['sampler']
    K = config['keyframes']

    if sampler == 'uniform':
        return uniform_keyframe_indices(len(poses), K, ref_idx)
    elif sampler == 'fps':
        return fps_keyframe_indices(
            poses, K, ref_idx,
            alpha=config['fps_alpha'],
            beta=config['fps_beta'],
            seed=config.get('seed', 42)
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def generate_heatmaps(depth_maps: List[np.ndarray], poses: List[np.ndarray], intrinsics: Dict,
                     keyframe_indices: List[int], ref_idx: int, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate frame-indexed heatmaps from keyframes to reference frame.

    Args:
        depth_maps: List of depth maps
        poses: List of poses (camera-to-world)
        intrinsics: Camera intrinsics
        keyframe_indices: Selected keyframe indices
        ref_idx: Reference frame index
        config: Heatmap configuration

    Returns:
        tuple: (heatmaps[K, Hm, Wm], mask[K])
    """
    K = len(keyframe_indices)
    Hm, Wm = config['heatmap']['size']
    ref_h, ref_w = depth_maps[ref_idx].shape

    heatmaps = np.zeros((K, Hm, Wm), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)

    # Reference frame transformation
    T_w_c_ref = poses[ref_idx]
    T_c_ref_w = np.linalg.inv(T_w_c_ref)

    # Build intrinsics dict
    if 'K' in intrinsics:
        # Intrinsics matrix format
        K_matrix = np.array(intrinsics['K'], dtype=np.float32)
        intrinsics_dict = {
            'K': K_matrix,
            'Kinv': np.linalg.inv(K_matrix),
            'fx': K_matrix[0, 0],
            'fy': K_matrix[1, 1],
            'cx': K_matrix[0, 2],
            'cy': K_matrix[1, 2]
        }
    else:
        # Individual parameters format
        intrinsics_dict = build_intrinsics(
            ref_w, ref_h,
            fx=intrinsics['fx'], fy=intrinsics['fy'],
            cx=intrinsics['cx'], cy=intrinsics['cy']
        )

    # Process each keyframe
    for k, j in enumerate(keyframe_indices):
        if j >= len(depth_maps) or j >= len(poses):
            logger.warning(f"Keyframe index {j} out of range, skipping")
            continue

        depth_j = depth_maps[j]
        T_w_c_j = poses[j]

        # Project keyframe to reference
        depth_ref = depth_maps[ref_idx] if config['heatmap']['occlusion_check'] else None
        occl_eps = config['heatmap']['occlusion_eps'] if config['heatmap']['occlusion_check'] else None

        uv_ref, valid_mask = project_keyframe_to_ref(
            depth_j, intrinsics_dict, T_w_c_j, T_c_ref_w,
            depth_ref, ref_w, ref_h, occl_eps
        )

        # Generate heatmap
        if len(uv_ref) > 0 and np.any(valid_mask):
            uv_valid = uv_ref[valid_mask]
            heatmap = heatmap_from_points(
                uv_valid, (ref_w, ref_h), (Wm, Hm),
                config['heatmap']['gaussian_sigma_px']
            )
            heatmaps[k] = heatmap
            mask[k] = 1.0
        else:
            # Empty heatmap
            mask[k] = 0.0

        logger.debug(f"Keyframe {j} -> ref {ref_idx}: {np.sum(valid_mask)}/{len(uv_ref)} valid points, "
                    f"mask={mask[k]}")

    return heatmaps, mask


def save_packed_clip(output_dir: Path, rgb_sub: List[np.ndarray], depth_sub: List[np.ndarray],
                    poses_sub: List[np.ndarray], intrinsics: Dict, heatmaps: np.ndarray,
                    mask: np.ndarray, meta: Dict, config: Dict):
    """
    Save packed clip in standard training format.

    Args:
        output_dir: Output directory for this clip
        rgb_sub, depth_sub, poses_sub: Subsequence data
        intrinsics: Camera intrinsics
        heatmaps: Generated heatmaps [K, Hm, Wm]
        mask: Validity mask [K]
        meta: Metadata dictionary
        config: Export configuration
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    # Save RGB frames
    for t, rgb in enumerate(rgb_sub):
        if config['export']['rgb_format'] == 'png':
            rgb_path = rgb_dir / f"{t:06d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Save depth maps
    for t, depth in enumerate(depth_sub):
        if config['export']['depth_format'] == 'npy':
            depth_path = depth_dir / f"{t:06d}.npy"
            np.save(depth_path, depth.astype(np.float32))

    # Save poses
    poses_path = output_dir / "poses.json"
    poses_data = [pose.tolist() for pose in poses_sub]
    with open(poses_path, 'w') as f:
        json.dump(poses_data, f, indent=2)

    # Save intrinsics
    intrinsics_path = output_dir / "intrinsics.json"
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)

    # Save heatmaps
    if config['export']['heatmap_format'] == 'npy':
        heatmaps_path = output_dir / "heatmaps.npy"
        np.save(heatmaps_path, heatmaps)

    # Save mask
    mask_path = output_dir / "mask.npy"
    np.save(mask_path, mask)

    # Save metadata
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def determine_ref_index(T: int, ref_policy: str, ref_index: int) -> int:
    """Determine reference frame index based on policy."""
    if ref_policy == 'last':
        return T - 1
    elif ref_policy == 'middle':
        return T // 2
    elif ref_policy == 'index':
        if ref_index < 0:
            return T + ref_index  # Negative indexing
        else:
            return min(ref_index, T - 1)
    else:
        raise ValueError(f"Unknown ref_policy: {ref_policy}")


def pack_split(config: Dict, split: str):
    """Pack one split (train/val/test) of the dataset."""
    logger.info(f"📦 Packing split: {split}")

    raw_root = Path(config['data']['raw_root'])
    save_root = Path(config['data']['save_root'])
    split_config = config['data']['splits'][split]

    # Set random seed
    np.random.seed(config['seed'])

    # Statistics
    total_clips = 0
    empty_heatmap_count = 0
    processed_clips = 0

    # Process each scene
    for scene_name in split_config['scenes']:
        logger.info(f"  Processing scene: {scene_name}")

        scene_dir = raw_root / split / scene_name
        if not scene_dir.exists():
            logger.warning(f"Scene directory not found: {scene_dir}")
            continue

        # Find all clips in scene
        clip_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')])
        logger.info(f"    Found {len(clip_dirs)} clips")

        for clip_dir in clip_dirs:
            try:
                # Load raw clip data
                rgb_frames, depth_maps, poses, intrinsics = load_raw_clip_data(clip_dir)

                if len(rgb_frames) == 0:
                    logger.warning(f"Empty clip: {clip_dir}")
                    continue

                # Extract subsequences
                subsequences = extract_subsequences(
                    rgb_frames, depth_maps, poses,
                    config['pack']['frames_per_clip'],
                    config['pack']['stride']
                )

                logger.debug(f"    Clip {clip_dir.name}: {len(rgb_frames)} frames -> {len(subsequences)} subsequences")

                # Process each subsequence
                for subseq_idx, (rgb_sub, depth_sub, poses_sub) in enumerate(subsequences):
                    T = len(rgb_sub)
                    ref_idx = determine_ref_index(T, config['pack']['ref_policy'], config['pack']['ref_index'])

                    # Select keyframes
                    keyframe_indices = select_keyframes(poses_sub, config['pack'], ref_idx)

                    # Generate heatmaps
                    heatmaps, mask = generate_heatmaps(
                        depth_sub, poses_sub, intrinsics,
                        keyframe_indices, ref_idx, config
                    )

                    # Create metadata
                    meta = {
                        'scene': scene_name,
                        'episode_id': int(clip_dir.name.split('_')[-1]),
                        'subsequence_id': subseq_idx,
                        'T': T,
                        'K': len(keyframe_indices),
                        'ref_idx': ref_idx,
                        'key_indices': keyframe_indices,
                        'sampler': {
                            'type': config['pack']['sampler'],
                            'alpha': config['pack'].get('fps_alpha', 1.0),
                            'beta': config['pack'].get('fps_beta', 0.7),
                            'seed': config['seed']
                        },
                        'image_size': [rgb_sub[0].shape[1], rgb_sub[0].shape[0]],  # [W, H]
                        'heatmap_size': config['heatmap']['size'],
                        'gaussian_sigma_px': config['heatmap']['gaussian_sigma_px'],
                        'occlusion_check': config['heatmap']['occlusion_check'],
                        'occlusion_eps': config['heatmap']['occlusion_eps']
                    }

                    # Create output path
                    output_clip_dir = save_root / split / scene_name / f"clip_{total_clips+1:06d}"

                    # Save packed clip
                    save_packed_clip(
                        output_clip_dir, rgb_sub, depth_sub, poses_sub,
                        intrinsics, heatmaps, mask, meta, config
                    )

                    total_clips += 1
                    processed_clips += 1

                    # Count empty heatmaps
                    empty_count = np.sum(mask == 0)
                    empty_heatmap_count += empty_count

                    # Progress logging
                    if total_clips % 50 == 0:
                        empty_ratio = empty_heatmap_count / (total_clips * config['pack']['keyframes'])
                        logger.info(f"    Progress: {total_clips} clips processed, "
                                   f"empty heatmap ratio: {empty_ratio:.1%}")

            except Exception as e:
                logger.error(f"Error processing clip {clip_dir}: {e}")
                continue

        # Scene statistics
        logger.info(f"  Scene {scene_name} completed: {processed_clips} clips processed")

    # Split statistics
    if total_clips > 0:
        empty_ratio = empty_heatmap_count / (total_clips * config['pack']['keyframes'])
        logger.info(f"✅ Split {split} completed:")
        logger.info(f"   Total clips: {total_clips}")
        logger.info(f"   Empty heatmap ratio: {empty_ratio:.1%}")
    else:
        logger.warning(f"No clips processed for split {split}")


def main():
    parser = argparse.ArgumentParser(description="Pack raw sequences into standard training format")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to dataset packing configuration')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Split to process')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    config = load_config(args.config)
    logger.info(f"📋 Loaded configuration: {args.config}")
    logger.info(f"🎯 Processing split: {args.split}")

    # Process split
    pack_split(config, args.split)

    logger.info(f"🎉 Dataset packing completed for split: {args.split}")
    logger.info(f"📁 Output saved to: {config['data']['save_root']}/{args.split}")


if __name__ == "__main__":
    main()