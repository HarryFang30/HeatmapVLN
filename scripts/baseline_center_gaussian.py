"""
Center Gaussian Baseline

Purpose: Evaluate a trivial baseline that always predicts a Gaussian centered
at the image center. This baseline does NOT use any input information.

This serves as a lower bound - any trained model should significantly outperform
this baseline, proving it learned meaningful spatial relationships.

Usage:
    python scripts/baseline_center_gaussian.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.vln_heatmap_adapter import VLNHeatmapDataset


def center_gaussian(Hm: int, Wm: int, sigma: float = 0.12) -> torch.Tensor:
    """
    Create a Gaussian heatmap centered at the image center.

    Args:
        Hm: Heatmap height
        Wm: Heatmap width
        sigma: Gaussian sigma as fraction of max(Hm, Wm)

    Returns:
        torch.Tensor: Normalized Gaussian heatmap [Hm, Wm] with sum=1
    """
    y, x = np.mgrid[0:Hm, 0:Wm]
    cy, cx = (Hm - 1) / 2, (Wm - 1) / 2

    # Squared distance from center
    d2 = (y - cy) ** 2 + (x - cx) ** 2

    # Gaussian
    sigma_pixels = sigma * max(Hm, Wm)
    hm = np.exp(-d2 / (2 * sigma_pixels ** 2)).astype(np.float32)

    # Normalize to sum=1
    hm_sum = hm.sum()
    hm = hm / hm_sum

    return torch.from_numpy(hm)


def main():
    """Evaluate center Gaussian baseline on validation set"""
    print("=" * 80)
    print("Center Gaussian Baseline Evaluation")
    print("=" * 80)
    print("\nThis baseline always predicts a Gaussian at image center,")
    print("regardless of input. Any trained model should beat this!\n")

    # Configuration
    data_root = './data/habitat_vln'
    hm_size = (64, 64)
    max_samples = 200  # Limit evaluation for speed

    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Heatmap size: {hm_size}")
    print(f"  Max samples: {max_samples}")

    # Create dataset
    print("\nLoading validation dataset...")
    try:
        ds = VLNHeatmapDataset(
            root=data_root,
            split='val',
            frames_per_clip=8,
            heatmap_per_clip=4,
            image_size=(384, 384),
            hm_size=hm_size
        )
        print(f"✓ Dataset loaded: {len(ds)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("Note: Run pack_dataset.py to create validation data first")
        return

    if len(ds) == 0:
        print("✗ Validation dataset is empty")
        print("Note: Run pack_dataset.py with --split val")
        return

    # Create center Gaussian prediction
    print("\nGenerating center Gaussian baseline...")
    hm_baseline = center_gaussian(hm_size[0], hm_size[1], sigma=0.12)
    print(f"✓ Baseline heatmap created: {hm_baseline.shape}, sum={hm_baseline.sum():.6f}")

    # Evaluate on validation set
    print("\n" + "-" * 80)
    print("Evaluating baseline...")
    print("-" * 80)

    total_nll = 0.0
    num_samples = 0
    num_evaluated = min(len(ds), max_samples)

    for i in range(num_evaluated):
        sample = ds[i]
        gt_heatmaps = sample['gt_heatmaps']  # [K, Hm, Wm]
        mask = sample['mask']  # [K]

        K = gt_heatmaps.size(0)

        # Compute NLL for each valid heatmap
        for k in range(K):
            if mask[k] > 0.5:  # Valid heatmap
                # Convert GT to probability distribution
                gt_flat = gt_heatmaps[k].view(-1)
                q = torch.softmax(gt_flat, dim=0)  # Target distribution

                # Prediction (always center Gaussian)
                p = hm_baseline.view(-1).clamp(min=1e-8)  # Predicted distribution

                # KL divergence: -sum(q * log(p))
                nll = -(q * torch.log(p)).sum()

                total_nll += float(nll)
                num_samples += 1

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_evaluated} samples...")

    # Compute average
    avg_nll = total_nll / max(num_samples, 1)

    # Results
    print("-" * 80)
    print("\nBaseline Results:")
    print("=" * 80)
    print(f"Samples evaluated:        {num_evaluated}")
    print(f"Valid heatmaps:           {num_samples}")
    print(f"Average NLL (KL-CE Loss): {avg_nll:.6f}")
    print()
    print("Interpretation:")
    print(f"  - Random uniform baseline: ~{np.log(hm_size[0] * hm_size[1]):.3f}")
    print(f"  - Center Gaussian:          {avg_nll:.3f}")
    print(f"  - Your model should be:     < {avg_nll:.3f} (significantly better)")
    print("=" * 80)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)

    main()