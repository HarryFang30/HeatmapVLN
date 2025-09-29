"""
Dataset Inspection Tool (inspect_dataset.py)
===========================================

Quality inspection tool for visualizing frame-indexed heatmaps.

Features:
- Upsample heatmaps to RGB resolution
- Color-coded overlay on reference frames
- Save visualization images
- Statistics on empty heatmaps

Usage:
    python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class DatasetInspector:
    """Dataset quality inspection and visualization."""

    def __init__(self, root: str, split: str):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.clips = self._enumerate_clips()
        logger.info(f"Found {len(self.clips)} clips in {self.split_dir}")

    def _enumerate_clips(self) -> List[Path]:
        """Enumerate all clip directories."""
        clips = []

        # Find all scene directories
        scene_dirs = [d for d in self.split_dir.iterdir() if d.is_dir()]

        for scene_dir in scene_dirs:
            # Find all clip directories in scene
            clip_dirs = sorted([d for d in scene_dir.iterdir()
                              if d.is_dir() and d.name.startswith('clip_')])
            clips.extend(clip_dirs)

        return clips

    def inspect_clip(self, clip_dir: Path, output_dir: Path) -> dict:
        """
        Inspect one clip and generate visualization.

        Args:
            clip_dir: Path to clip directory
            output_dir: Output directory for visualizations

        Returns:
            dict: Inspection statistics
        """
        logger.info(f"Inspecting clip: {clip_dir}")

        try:
            # Load clip data
            meta = self._load_metadata(clip_dir)
            ref_idx = meta.get('ref_idx', -1)
            K = meta.get('K', 4)

            # Load reference frame
            ref_frame = self._load_reference_frame(clip_dir, ref_idx)

            # Load heatmaps and mask
            heatmaps, mask = self._load_heatmaps(clip_dir)

            # Generate overlay visualization
            overlay_image = self._create_overlay_visualization(ref_frame, heatmaps, mask)

            # Save visualization
            scene_name = clip_dir.parent.name
            clip_name = clip_dir.name
            output_scene_dir = output_dir / scene_name
            output_scene_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_scene_dir / f"{clip_name}_overlay.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

            # Compute statistics
            valid_heatmaps = int(np.sum(mask > 0.5))
            empty_heatmaps = K - valid_heatmaps

            stats = {
                'clip_dir': str(clip_dir),
                'scene': scene_name,
                'clip': clip_name,
                'ref_idx': ref_idx,
                'total_heatmaps': K,
                'valid_heatmaps': valid_heatmaps,
                'empty_heatmaps': empty_heatmaps,
                'empty_ratio': empty_heatmaps / K if K > 0 else 0.0,
                'output_path': str(output_path)
            }

            logger.info(f"  ✅ Clip inspected: {valid_heatmaps}/{K} valid heatmaps, "
                       f"empty ratio: {stats['empty_ratio']:.1%}")

            return stats

        except Exception as e:
            logger.error(f"  ❌ Error inspecting clip {clip_dir}: {e}")
            return {
                'clip_dir': str(clip_dir),
                'error': str(e),
                'valid_heatmaps': 0,
                'total_heatmaps': 0,
                'empty_ratio': 1.0
            }

    def _load_metadata(self, clip_dir: Path) -> dict:
        """Load clip metadata."""
        meta_file = clip_dir / "meta.json"

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with open(meta_file, 'r') as f:
            return json.load(f)

    def _load_reference_frame(self, clip_dir: Path, ref_idx: int) -> np.ndarray:
        """Load reference RGB frame."""
        rgb_dir = clip_dir / "rgb"

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

        # Find RGB file for reference frame
        ref_file = rgb_dir / f"{ref_idx:06d}.png"

        if not ref_file.exists():
            # Try to find any available frame
            rgb_files = sorted(rgb_dir.glob("*.png"))
            if len(rgb_files) == 0:
                raise FileNotFoundError(f"No RGB files found in {rgb_dir}")
            ref_file = rgb_files[-1]  # Use last frame
            logger.warning(f"Reference frame {ref_idx} not found, using {ref_file.name}")

        # Load and convert to RGB
        image = cv2.imread(str(ref_file))
        if image is None:
            raise ValueError(f"Failed to load image: {ref_file}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_heatmaps(self, clip_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load heatmaps and mask."""
        heatmaps_file = clip_dir / "heatmaps.npy"
        mask_file = clip_dir / "mask.npy"

        if not heatmaps_file.exists():
            raise FileNotFoundError(f"Heatmaps file not found: {heatmaps_file}")
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        heatmaps = np.load(heatmaps_file).astype(np.float32)
        mask = np.load(mask_file).astype(np.float32)

        return heatmaps, mask

    def _create_overlay_visualization(self, ref_frame: np.ndarray, heatmaps: np.ndarray,
                                    mask: np.ndarray) -> np.ndarray:
        """
        Create overlay visualization of heatmaps on reference frame.

        Args:
            ref_frame: Reference RGB frame [H, W, 3]
            heatmaps: Heatmaps [K, Hm, Wm]
            mask: Validity mask [K]

        Returns:
            np.ndarray: Overlay image [H_out, W_out, 3]
        """
        H, W = ref_frame.shape[:2]
        K = len(heatmaps)

        # Create grid layout: 2x(K//2+1) for K heatmaps + original
        cols = min(3, K + 1)  # Maximum 3 columns
        rows = (K + 1 + cols - 1) // cols  # Ceiling division

        # Create output image
        cell_h, cell_w = H, W
        output_h = rows * cell_h
        output_w = cols * cell_w
        output_image = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        # Color maps for different heatmaps
        colormaps = [cm.Reds, cm.Blues, cm.Greens, cm.Purples]

        # Place original image in top-left
        output_image[:cell_h, :cell_w] = ref_frame

        # Add "Original" text
        cv2.putText(output_image, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Place heatmap overlays
        for k in range(K):
            # Calculate grid position
            cell_idx = k + 1  # Skip first cell (original)
            row = cell_idx // cols
            col = cell_idx % cols

            if row >= rows:
                break

            # Calculate output position
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w

            if mask[k] > 0.5:  # Valid heatmap
                # Upsample heatmap to image resolution
                heatmap = heatmaps[k]  # [Hm, Wm]
                heatmap_upsampled = cv2.resize(heatmap, (W, H))

                # Normalize for visualization (handle potential numerical issues)
                hm_max = heatmap_upsampled.max()
                if hm_max > 0:
                    heatmap_norm = heatmap_upsampled / hm_max
                else:
                    heatmap_norm = heatmap_upsampled

                # Apply colormap
                colormap = colormaps[k % len(colormaps)]
                heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # Drop alpha
                heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

                # Create overlay (blend with original image)
                alpha = 0.6
                overlay = (alpha * heatmap_colored + (1 - alpha) * ref_frame).astype(np.uint8)

                # Add text indicating keyframe index
                overlay_with_text = overlay.copy()
                cv2.putText(overlay_with_text, f"Keyframe {k}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                output_image[y_start:y_end, x_start:x_end] = overlay_with_text

            else:  # Empty heatmap
                # Show reference frame with "Empty" label
                empty_image = ref_frame.copy()
                cv2.putText(empty_image, f"Empty {k}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

                output_image[y_start:y_end, x_start:x_end] = empty_image

        return output_image

    def inspect_dataset(self, num_clips: Optional[int] = None, output_dir: str = "./outputs/inspect") -> dict:
        """
        Inspect multiple clips and generate visualizations.

        Args:
            num_clips: Number of clips to inspect (None for all)
            output_dir: Output directory for visualizations

        Returns:
            dict: Overall statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        clips_to_inspect = self.clips[:num_clips] if num_clips else self.clips

        logger.info(f"🔍 Inspecting {len(clips_to_inspect)} clips...")

        # Inspect clips
        all_stats = []
        total_heatmaps = 0
        total_empty = 0

        for i, clip_dir in enumerate(clips_to_inspect):
            logger.info(f"Progress: {i+1}/{len(clips_to_inspect)}")

            stats = self.inspect_clip(clip_dir, output_path)
            all_stats.append(stats)

            total_heatmaps += stats.get('total_heatmaps', 0)
            total_empty += stats.get('empty_heatmaps', 0)

        # Overall statistics
        overall_empty_ratio = total_empty / total_heatmaps if total_heatmaps > 0 else 0.0

        overall_stats = {
            'total_clips_inspected': len(all_stats),
            'total_heatmaps': total_heatmaps,
            'total_empty_heatmaps': total_empty,
            'overall_empty_ratio': overall_empty_ratio,
            'output_directory': str(output_path),
            'clip_stats': all_stats
        }

        logger.info(f"✅ Dataset inspection completed!")
        logger.info(f"   Clips inspected: {len(all_stats)}")
        logger.info(f"   Total heatmaps: {total_heatmaps}")
        logger.info(f"   Empty heatmaps: {total_empty} ({overall_empty_ratio:.1%})")
        logger.info(f"   Visualizations saved to: {output_path}")

        return overall_stats


def main():
    parser = argparse.ArgumentParser(description="Inspect and visualize VLN heatmap dataset")
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of packed dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to inspect')
    parser.add_argument('--num', type=int, default=None,
                        help='Number of clips to inspect (default: all)')
    parser.add_argument('--output', type=str, default='./outputs/inspect',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info(f"🔍 Starting dataset inspection")
    logger.info(f"Dataset: {args.root}/{args.split}")
    logger.info(f"Clips to inspect: {args.num if args.num else 'all'}")
    logger.info(f"Output: {args.output}")

    try:
        # Create inspector
        inspector = DatasetInspector(args.root, args.split)

        # Run inspection
        stats = inspector.inspect_dataset(args.num, args.output)

        # Save statistics
        stats_file = Path(args.output) / f"{args.split}_inspection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"📊 Statistics saved to: {stats_file}")

        # Print summary
        logger.info("\n📋 INSPECTION SUMMARY:")
        logger.info(f"Empty heatmap ratio: {stats['overall_empty_ratio']:.1%}")
        if stats['overall_empty_ratio'] > 0.2:
            logger.warning("⚠️  High empty heatmap ratio (>20%) - check geometry/projection settings")
        else:
            logger.info("✅ Empty heatmap ratio looks reasonable (<20%)")

    except Exception as e:
        logger.error(f"❌ Inspection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()