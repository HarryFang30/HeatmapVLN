"""
Test with Synthetic Data - ALL heatmaps valid

Create simple synthetic data where we KNOW the model should be able to learn.
This isolates whether the problem is the model or the data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss


def create_synthetic_batch(batch_size=1, frames_per_clip=8, k_heatmaps=4,
                           image_size=(384, 384), hm_size=(64, 64)):
    """Create synthetic training batch with ALL valid heatmaps"""
    # Random frames
    frames = torch.randn(batch_size, frames_per_clip, 3, *image_size)

    # Create structured heatmaps (Gaussians at different positions)
    heatmaps = []
    for k in range(k_heatmaps):
        # Random Gaussian center
        cy = np.random.randint(10, hm_size[0] - 10)
        cx = np.random.randint(10, hm_size[1] - 10)

        y, x = np.mgrid[0:hm_size[0], 0:hm_size[1]]
        d2 = (y - cy)**2 + (x - cx)**2
        hm = np.exp(-d2 / (2 * 100))  # sigma=10 pixels
        hm = hm / hm.sum()  # Normalize

        heatmaps.append(torch.from_numpy(hm).float())

    heatmaps = torch.stack(heatmaps, dim=0)  # [K, H, W]
    heatmaps = heatmaps.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, K, H, W]

    # ALL masks valid
    mask = torch.ones(batch_size, k_heatmaps)

    return frames, heatmaps, mask


def main():
    print("=" * 80)
    print("Test with Synthetic Data - ALL Heatmaps Valid")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = VLNHeatmapModel(
        k_heatmaps=4, hm_size=(64, 64), vision_dim=1024,
        agg='mean', use_lora=False
    ).to(device).train()

    print(f"Model parameters: {model.get_trainable_parameters()['total']:,}")

    # Create synthetic data
    frames, targets, mask = create_synthetic_batch()
    frames, targets, mask = frames.to(device), targets.to(device), mask.to(device)

    print(f"\nSynthetic data:")
    print(f"  Frames: {frames.shape}")
    print(f"  Targets: {targets.shape}, range=[{targets.min():.6f}, {targets.max():.6f}]")
    print(f"  Mask: {mask[0].tolist()} (ALL VALID)")
    print(f"  Target entropy: {-(targets * torch.log(targets.clamp(min=1e-8))).sum(dim=(-1,-2))[0].mean():.3f}")

    # Optimizer with higher LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)

    print(f"\nTraining for 400 steps (higher LR=5e-3)...")
    print("-" * 80)

    for step in range(400):
        # Forward
        preds, _ = model(frames)
        loss = kl_ce_loss(preds, targets, mask=mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradient norms
        if step == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Step {step}: loss={float(loss):.6f}, grad_norm={total_norm:.6f}")

        optimizer.step()

        if step % 50 == 0 and step > 0:
            print(f"Step {step}: loss={float(loss):.6f}")

    print("-" * 80)
    print(f"\nFinal loss: {float(loss):.6f}")

    if float(loss) < 5.0:
        print("✓✓ EXCELLENT: Model learned from synthetic data!")
        print("   This proves the model CAN learn.")
        print("   Problem is likely:")
        print("   - Real data has only 25% valid heatmaps")
        print("   - Need more training samples")
        print("   - Or need to generate better training data")
    elif float(loss) < 7.0:
        print("✓ GOOD: Model is learning (loss < 7.0)")
    else:
        print("✗ FAIL: Model cannot learn even from synthetic data")
        print("   This indicates a fundamental model/training issue")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()