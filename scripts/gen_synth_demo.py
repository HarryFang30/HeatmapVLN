"""
Synthetic Demo Data Generator (gen_synth_demo.py)
================================================

Generate synthetic VLN sequences with consistent geometry for testing the full pipeline.
No external simulation dependencies - creates controlled 3D scenes and camera trajectories.

Key Features:
- 3D checkerboard/plane point cloud generation
- Circular camera trajectory around scene
- Consistent depth map rendering from 3D geometry
- Reference frame can "see" keyframe points (validates projection)
- Saves in raw_sequences format for pack_dataset.py

Usage:
    python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 1 --T 8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.heatmap_builder import build_intrinsics

logger = logging.getLogger(__name__)


class SyntheticSceneGenerator:
    """Generate synthetic 3D scenes with controlled geometry."""

    def __init__(self, scene_size: float = 5.0, checkerboard_size: int = 8):
        self.scene_size = scene_size
        self.checkerboard_size = checkerboard_size
        self.points_3d = self._generate_3d_points()

    def _generate_3d_points(self) -> np.ndarray:
        """Generate 3D checkerboard points."""
        points = []

        # Ground plane checkerboard
        step = self.scene_size / self.checkerboard_size
        for i in range(self.checkerboard_size + 1):
            for j in range(self.checkerboard_size + 1):
                x = -self.scene_size/2 + i * step
                z = -self.scene_size/2 + j * step
                y = 0.0  # Ground level
                points.append([x, y, z])

        # Add some elevated points (walls/objects)
        wall_height = 2.0
        for i in range(0, self.checkerboard_size + 1, 2):
            # Back wall
            x = -self.scene_size/2 + i * step
            z = self.scene_size/2
            for h in [0.5, 1.0, 1.5, 2.0]:
                points.append([x, h, z])

            # Side wall
            z = -self.scene_size/2 + i * step
            x = self.scene_size/2
            for h in [0.5, 1.0, 1.5, 2.0]:
                points.append([x, h, z])

        return np.array(points, dtype=np.float32)

    def generate_camera_trajectory(self, T: int, radius: float = 3.0, height: float = 1.5) -> List[np.ndarray]:
        """Generate circular camera trajectory around scene center."""
        poses = []

        for t in range(T):
            # Circular trajectory
            angle = 2 * np.pi * t / T

            # Camera position
            cam_x = radius * np.cos(angle)
            cam_z = radius * np.sin(angle)
            cam_y = height

            # Look at scene center
            look_at = np.array([0.0, 0.5, 0.0])  # Slightly above ground
            camera_pos = np.array([cam_x, cam_y, cam_z])

            # Build camera coordinate system
            forward = look_at - camera_pos
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)

            up = np.cross(right, forward)

            # Build rotation matrix (camera to world)
            R = np.column_stack([right, up, -forward])  # Note: -forward for camera convention

            # Build 4x4 transformation matrix (camera to world)
            T_w_c = np.eye(4)
            T_w_c[:3, :3] = R
            T_w_c[:3, 3] = camera_pos

            poses.append(T_w_c)

        return poses

    def render_depth_map(self, T_w_c: np.ndarray, intrinsics: Dict, width: int, height: int) -> np.ndarray:
        """Render synthetic depth map from 3D points."""
        # Transform points to camera coordinates
        T_c_w = np.linalg.inv(T_w_c)
        points_hom = np.hstack([self.points_3d, np.ones((len(self.points_3d), 1))])
        points_cam = (T_c_w @ points_hom.T).T[:, :3]

        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0.1
        points_cam = points_cam[valid_depth]

        if len(points_cam) == 0:
            return np.zeros((height, width), dtype=np.float32)

        # Project to image coordinates
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        # Filter points within image bounds
        valid_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_valid = u[valid_bounds]
        v_valid = v[valid_bounds]
        z_valid = z[valid_bounds]

        # Create sparse depth map
        depth_map = np.zeros((height, width), dtype=np.float32)

        for ui, vi, zi in zip(u_valid, v_valid, z_valid):
            u_int, v_int = int(round(ui)), int(round(vi))
            if 0 <= u_int < width and 0 <= v_int < height:
                # Keep closest depth if multiple points project to same pixel
                if depth_map[v_int, u_int] == 0 or zi < depth_map[v_int, u_int]:
                    depth_map[v_int, u_int] = zi

        # Apply Gaussian blur to create dense depth map
        if np.any(depth_map > 0):
            # Create mask of valid depths
            valid_mask = depth_map > 0

            # Apply mild Gaussian blur
            depth_blurred = cv2.GaussianBlur(depth_map, (15, 15), 3.0)

            # Blend original and blurred based on proximity to valid points
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            valid_dilated = cv2.dilate(valid_mask.astype(np.uint8), kernel)

            # Use original depths where available, blurred elsewhere
            depth_final = np.where(valid_mask, depth_map, depth_blurred * valid_dilated)
        else:
            depth_final = depth_map

        return depth_final

    def render_rgb_image(self, T_w_c: np.ndarray, intrinsics: Dict, width: int, height: int) -> np.ndarray:
        """Render simple RGB image with checkerboard pattern."""
        # Create simple checkerboard-like pattern
        rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        for y in range(height):
            intensity = int(50 + 100 * (1 - y / height))  # Darker at top
            rgb[y, :] = [intensity, intensity, intensity + 20]  # Slightly blue sky

        # Render 3D points as colored circles
        T_c_w = np.linalg.inv(T_w_c)
        points_hom = np.hstack([self.points_3d, np.ones((len(self.points_3d), 1))])
        points_cam = (T_c_w @ points_hom.T).T[:, :3]

        # Filter and project points
        valid_depth = points_cam[:, 2] > 0.1
        points_cam = points_cam[valid_depth]
        original_points = self.points_3d[valid_depth]

        if len(points_cam) > 0:
            fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

            x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy

            for ui, vi, zi, pt_3d in zip(u, v, z, original_points):
                u_int, v_int = int(round(ui)), int(round(vi))
                if 0 <= u_int < width and 0 <= v_int < height:
                    # Color based on height
                    height_ratio = max(0, min(1, pt_3d[1] / 2.0))  # Y coordinate
                    color = [
                        int(255 * (1 - height_ratio)),  # Red decreases with height
                        int(255 * height_ratio),        # Green increases with height
                        100                             # Constant blue
                    ]

                    # Draw circle
                    cv2.circle(rgb, (u_int, v_int), 3, color, -1)

        return rgb


def generate_synthetic_clip(output_root: str, scene_name: str, clip_id: int, T: int = 8,
                           width: int = 384, height: int = 384, hfov_deg: float = 60.0):
    """Generate one synthetic clip with T frames."""

    # Create output directories
    clip_dir = Path(output_root) / "train" / scene_name / f"clip_{clip_id:06d}"
    rgb_dir = clip_dir / "rgb"
    depth_dir = clip_dir / "depth"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic clip: {clip_dir}")

    # Initialize scene generator
    scene = SyntheticSceneGenerator()

    # Generate camera trajectory
    poses = scene.generate_camera_trajectory(T, radius=3.0, height=1.5)

    # Build camera intrinsics
    intrinsics = build_intrinsics(width, height, hfov_deg=hfov_deg)

    # Generate frames
    rgb_frames = []
    depth_maps = []
    pose_matrices = []

    for t in range(T):
        logger.info(f"  Rendering frame {t+1}/{T}")

        T_w_c = poses[t]

        # Render RGB and depth
        rgb = scene.render_rgb_image(T_w_c, intrinsics, width, height)
        depth = scene.render_depth_map(T_w_c, intrinsics, width, height)

        # Save RGB
        rgb_path = rgb_dir / f"{t:06d}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Save depth
        depth_path = depth_dir / f"{t:06d}.npy"
        np.save(depth_path, depth.astype(np.float32))

        rgb_frames.append(rgb)
        depth_maps.append(depth)
        pose_matrices.append(T_w_c.tolist())  # Convert to list for JSON serialization

    # Save poses.json
    poses_path = clip_dir / "poses.json"
    with open(poses_path, 'w') as f:
        json.dump(pose_matrices, f, indent=2)

    # Save intrinsics.json
    intrinsics_path = clip_dir / "intrinsics.json"
    intrinsics_data = {
        "fx": float(intrinsics['fx']),
        "fy": float(intrinsics['fy']),
        "cx": float(intrinsics['cx']),
        "cy": float(intrinsics['cy']),
        "K": intrinsics['K'].tolist()
    }
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics_data, f, indent=2)

    logger.info(f"✅ Synthetic clip generated: {T} frames")

    # Print some statistics
    total_points = len(scene.points_3d)
    avg_depth_points = np.mean([np.sum(depth > 0) for depth in depth_maps])
    logger.info(f"   Scene: {total_points} 3D points")
    logger.info(f"   Avg visible points per frame: {avg_depth_points:.0f}")

    return clip_dir


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VLN demo data")
    parser.add_argument('--root', type=str, default='./raw_sequences',
                        help='Root directory for raw sequences')
    parser.add_argument('--scene', type=str, default='RoomA',
                        help='Scene name')
    parser.add_argument('--clips', type=int, default=1,
                        help='Number of clips to generate')
    parser.add_argument('--T', type=int, default=8,
                        help='Number of frames per clip')
    parser.add_argument('--W', type=int, default=384,
                        help='Image width')
    parser.add_argument('--H', type=int, default=384,
                        help='Image height')
    parser.add_argument('--hfov', type=float, default=60.0,
                        help='Horizontal field of view in degrees')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("🚀 Starting synthetic demo data generation")
    logger.info(f"Parameters: {args.clips} clips, {args.T} frames, {args.W}x{args.H}, hfov={args.hfov}°")

    # Generate clips
    generated_clips = []
    for clip_id in range(1, args.clips + 1):
        clip_dir = generate_synthetic_clip(
            output_root=args.root,
            scene_name=args.scene,
            clip_id=clip_id,
            T=args.T,
            width=args.W,
            height=args.H,
            hfov_deg=args.hfov
        )
        generated_clips.append(clip_dir)

    logger.info("🎉 Synthetic demo generation completed!")
    logger.info("Generated clips:")
    for clip_dir in generated_clips:
        logger.info(f"  📁 {clip_dir}")

    # Print next steps
    logger.info("\n📋 Next steps:")
    logger.info("1. Run pack_dataset.py to convert to training format:")
    logger.info("   python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train")
    logger.info("2. Run inspect_dataset.py to visualize results:")
    logger.info("   python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 1")


if __name__ == "__main__":
    main()