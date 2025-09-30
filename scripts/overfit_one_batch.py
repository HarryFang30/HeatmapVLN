"""
Overfit One Batch Test

Purpose: Verify the model can LEARN by overfitting 1-2 training samples.

This is the critical sanity check that proves:
1. Model has sufficient capacity
2. Loss function is correctly implemented
3. Gradients flow properly
4. Optimization works

Expected result: Loss should drop significantly (ideally < 1.0) within 200-800 steps.

Usage:
    python scripts/overfit_one_batch.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss


def main():
    """Run overfit test on 1-2 samples"""
    print("=" * 80)
    print("Overfit One Batch Test - Sanity Check #1")
    print("=" * 80)
    print("\nPurpose: Verify model can LEARN by overfitting small dataset")
    print("Expected: Loss drops from ~8 to < 3.0 (ideally < 1.0) in 200-800 steps\n")

    # Configuration
    data_root = './data/habitat_vln'
    frames_per_clip = 8
    heatmap_per_clip = 4
    image_size = (384, 384)
    hm_size = (64, 64)
    num_samples = 2  # Overfit on just 2 samples
    max_steps = 800
    log_every = 50

    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Samples to overfit: {num_samples}")
    print(f"  Max steps: {max_steps}")
    print(f"  Heatmap size: {hm_size}")

    # Create dataset
    print("\nCreating dataset...")
    try:
        ds = VLNHeatmapDataset(
            root=data_root,
            split='train',
            frames_per_clip=frames_per_clip,
            heatmap_per_clip=heatmap_per_clip,
            image_size=image_size,
            hm_size=hm_size
        )
        print(f"✓ Full dataset: {len(ds)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Take only first N samples
    actual_samples = min(num_samples, len(ds))
    small_ds = Subset(ds, list(range(actual_samples)))
    print(f"✓ Overfit subset: {actual_samples} samples")

    # Create dataloader (repeat same samples)
    dl = DataLoader(small_ds, batch_size=1, shuffle=True)

    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VLNHeatmapModel(
        k_heatmaps=heatmap_per_clip,
        hm_size=hm_size,
        vision_dim=1024,
        agg='mean',
        use_lora=False
    ).to(device)
    model.train()

    # Print parameter count
    param_count = model.get_trainable_parameters()['total']
    print(f"✓ Model created: {param_count:,} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )

    # Training loop
    print("\n" + "-" * 80)
    print("Starting overfit training...")
    print("-" * 80)

    initial_loss = None
    final_loss = None
    step = 0

    # Infinite loop over small dataset
    import itertools
    for step, batch in zip(range(max_steps), itertools.cycle(dl)):
        # Move to device
        frames = batch['frames'].to(device)
        targets = batch['gt_heatmaps'].to(device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        # Forward pass
        preds, _ = model(frames)
        loss = kl_ce_loss(preds, targets, mask=mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record losses
        loss_value = float(loss)
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value

        # Log progress
        if step % log_every == 0:
            print(f"Step {step:4d}: loss={loss_value:.6f}")

    # Final results
    print("-" * 80)
    print("\nOverfit Test Results:")
    print("=" * 80)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {initial_loss - final_loss:.6f} ({(1 - final_loss/initial_loss)*100:.1f}% decrease)")
    print()

    # Evaluate success
    success = final_loss < 3.0
    excellent = final_loss < 1.0

    if excellent:
        print("✓✓ EXCELLENT: Model overfits perfectly (loss < 1.0)")
        print("   This proves the model has sufficient capacity and can learn!")
    elif success:
        print("✓ PASS: Model can learn (loss < 3.0)")
        print("   Model is learning, but could potentially improve more.")
    else:
        print("✗ FAIL: Model is NOT learning properly")
        print("   Possible issues:")
        print("   - Check renderer parameters (τ, σ, α)")
        print("   - Verify heatmap normalization (sum=1)")
        print("   - Check softmax dimensions")
        print("   - Increase learning rate or training steps")

    print("=" * 80)

    return success


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    success = main()
    sys.exit(0 if success else 1)