"""
Dataset Mean Heatmap Baseline

Purpose: Compute the mean heatmap from training data and use it as a constant
prediction for all test samples. This baseline captures dataset-level biases
but does not adapt to individual inputs.

A good trained model should significantly outperform this baseline, proving it
learned input-specific spatial relationships rather than just dataset statistics.

Usage:
    python scripts/baseline_mean_heatmap.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.vln_heatmap_adapter import VLNHeatmapDataset


def compute_mean_heatmap(dataset, max_samples: int = 100) -> torch.Tensor:
    """
    Compute mean heatmap from training dataset.

    Args:
        dataset: Training dataset
        max_samples: Maximum number of batches to process

    Returns:
        torch.Tensor: Mean heatmap [Hm, Wm] with sum=1
    """
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    mean_hm = None
    num_batches = 0

    print("Computing mean heatmap from training data...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_samples:
            break

        gt_heatmaps = batch['gt_heatmaps']  # [B, K, Hm, Wm]
        mask = batch['mask']  # [B, K]

        # Convert to probability distributions
        B, K, Hm, Wm = gt_heatmaps.shape
        gt_flat = gt_heatmaps.view(B, K, -1)
        gt_probs = F.softmax(gt_flat, dim=-1).view(B, K, Hm, Wm)

        # Accumulate mean (only from valid heatmaps)
        for b in range(B):
            for k in range(K):
                if mask[b, k] > 0.5:  # Valid heatmap
                    if mean_hm is None:
                        mean_hm = gt_probs[b, k].clone()
                        num_batches = 1
                    else:
                        mean_hm = mean_hm + gt_probs[b, k]
                        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1} batches...")

    # Average
    if mean_hm is not None and num_batches > 0:
        mean_hm = mean_hm / num_batches
        # Renormalize to sum=1
        mean_hm = mean_hm / mean_hm.sum()

    print(f"✓ Mean heatmap computed from {num_batches} valid heatmaps")

    return mean_hm


def main():
    """Evaluate dataset mean baseline"""
    print("=" * 80)
    print("Dataset Mean Heatmap Baseline Evaluation")
    print("=" * 80)
    print("\nThis baseline predicts the training set mean heatmap for all inputs.")
    print("It captures dataset bias but not input-specific patterns.\n")

    # Configuration
    data_root = './data/habitat_vln'
    hm_size = (64, 64)
    max_train_samples = 100  # Batches for computing mean
    max_val_samples = 200    # Samples for evaluation

    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Heatmap size: {hm_size}")

    # Load training dataset
    print("\nLoading training dataset...")
    try:
        train_ds = VLNHeatmapDataset(
            root=data_root,
            split='train',
            frames_per_clip=8,
            heatmap_per_clip=4,
            image_size=(384, 384),
            hm_size=hm_size
        )
        print(f"✓ Train dataset: {len(train_ds)} samples")
    except Exception as e:
        print(f"✗ Failed to load training dataset: {e}")
        return

    if len(train_ds) == 0:
        print("✗ Training dataset is empty")
        print("Note: Run pack_dataset.py to create training data first")
        return

    # Compute mean heatmap from training data
    print("\n" + "-" * 80)
    mean_hm = compute_mean_heatmap(train_ds, max_samples=max_train_samples)
    print("-" * 80)

    if mean_hm is None:
        print("✗ Failed to compute mean heatmap")
        return

    print(f"Mean heatmap shape: {mean_hm.shape}, sum={mean_hm.sum():.6f}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    try:
        val_ds = VLNHeatmapDataset(
            root=data_root,
            split='val',
            frames_per_clip=8,
            heatmap_per_clip=4,
            image_size=(384, 384),
            hm_size=hm_size
        )
        print(f"✓ Validation dataset: {len(val_ds)} samples")
    except Exception as e:
        print(f"✗ Failed to load validation dataset: {e}")
        print("Note: Run pack_dataset.py with --split val")
        return

    if len(val_ds) == 0:
        print("✗ Validation dataset is empty")
        return

    # Evaluate on validation set
    print("\n" + "-" * 80)
    print("Evaluating mean heatmap baseline...")
    print("-" * 80)

    total_nll = 0.0
    num_samples = 0
    num_evaluated = min(len(val_ds), max_val_samples)

    for i in range(num_evaluated):
        sample = val_ds[i]
        gt_heatmaps = sample['gt_heatmaps']  # [K, Hm, Wm]
        mask = sample['mask']  # [K]

        K = gt_heatmaps.size(0)

        # Compute NLL for each valid heatmap
        for k in range(K):
            if mask[k] > 0.5:  # Valid heatmap
                # Convert GT to probability distribution
                gt_flat = gt_heatmaps[k].view(-1)
                q = torch.softmax(gt_flat, dim=0)  # Target distribution

                # Prediction (always mean heatmap)
                p = mean_hm.view(-1).clamp(min=1e-8)  # Predicted distribution

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
    print(f"  - Dataset Mean baseline:    {avg_nll:.3f}")
    print(f"  - Your model should be:     < {avg_nll:.3f} (significantly better)")
    print()
    print("If your model is worse than this baseline, it means:")
    print("  - Model is not learning from inputs")
    print("  - Model is outputting worse predictions than training set average")
    print("=" * 80)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)

    main()