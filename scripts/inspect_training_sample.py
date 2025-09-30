"""
Inspect Training Sample

Check if the training data is actually learnable.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.vln_heatmap_adapter import VLNHeatmapDataset


def main():
    print("=" * 80)
    print("Inspecting Training Sample")
    print("=" * 80)

    # Load data
    ds = VLNHeatmapDataset(
        root='./data/habitat_vln', split='train',
        frames_per_clip=8, heatmap_per_clip=4,
        image_size=(384, 384), hm_size=(64, 64)
    )

    print(f"\nDataset size: {len(ds)}")

    if len(ds) == 0:
        print("ERROR: Dataset is empty!")
        return

    # Get first sample
    sample = ds[0]
    frames = sample['frames']  # [T, 3, H, W]
    heatmaps = sample['gt_heatmaps']  # [K, Hm, Wm]
    mask = sample['mask']  # [K]

    print(f"\nSample 0:")
    print(f"  Frames: {frames.shape}")
    print(f"  Heatmaps: {heatmaps.shape}")
    print(f"  Mask: {mask.tolist()}")

    # Check each heatmap
    print(f"\nHeatmap statistics:")
    for k in range(heatmaps.size(0)):
        hm = heatmaps[k]
        valid = mask[k] > 0.5

        print(f"  Heatmap {k}: valid={valid}, "
              f"range=[{hm.min():.6f}, {hm.max():.6f}], "
              f"mean={hm.mean():.6f}, "
              f"sum={hm.sum():.6f}, "
              f"nonzero={(hm > 0).sum()}/{hm.numel()}")

        if valid:
            # This is the heatmap we're trying to learn from
            print(f"    ↑ This is the ONLY valid heatmap we can learn from!")

            # Check if it has structure
            entropy = -(hm * torch.log(hm.clamp(min=1e-8))).sum()
            max_entropy = np.log(64 * 64)
            print(f"    Entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")

            if entropy / max_entropy > 0.99:
                print(f"    ⚠️  Heatmap is nearly uniform! No structure to learn!")

    # Summary
    num_valid = (mask > 0.5).sum().item()
    print(f"\nSummary:")
    print(f"  Valid heatmaps: {num_valid}/{len(mask)} ({num_valid/len(mask)*100:.1f}%)")
    print(f"  Invalid heatmaps: {len(mask) - num_valid}/{len(mask)}")

    if num_valid == 0:
        print(f"\n✗ PROBLEM: NO valid heatmaps to learn from!")
    elif num_valid / len(mask) < 0.5:
        print(f"\n⚠️  WARNING: < 50% valid heatmaps. Training will be inefficient.")


if __name__ == '__main__':
    main()