"""
Overfit One Batch Test - UPDATED with ResNet + Logits Loss + Relative Threshold

Purpose: Verify the model can LEARN by overfitting 1-2 training samples.

Changes from previous version:
- Uses ResNet18 pretrained backbone
- Trains in logits space (heatmap_ce_from_logits)
- Higher learning rate (3e-3)
- No AMP during overfit test
- Relative entropy threshold (H(q) + 0.3) instead of absolute <1.0

Expected result: Loss should drop to near H(q) (target entropy) within 200-800 steps.
Per fix_visibility_and_effectiveK.md: Optimal CE is H(q), not absolute <1.0.

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
import itertools

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import heatmap_ce_from_logits
from src.data.quality_metrics import heatmap_entropy


def grad_norm(params):
    """Compute gradient norm for parameters"""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def main():
    """Run overfit test on 1-2 samples"""
    print("=" * 80)
    print("Overfit One Batch Test - ResNet + Logits Loss + Relative Threshold")
    print("=" * 80)
    print("\nPurpose: Verify model can LEARN by overfitting small dataset")
    print("Expected: Loss drops to near H(q) (target entropy) in 200-800 steps")
    print("Note: Optimal CE is H(q), NOT absolute <1.0\n")

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
    print(f"  Backbone: ResNet18 (pretrained)")
    print(f"  Loss: heatmap_ce_from_logits (logits space)")
    print(f"  Learning rate: 3e-3")

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

    # Compute target entropy for relative threshold (per fix_visibility_and_effectiveK.md § 5)
    print("\nComputing target entropy for relative threshold...")
    try:
        first_batch = next(iter(dl))
        with torch.no_grad():
            H = heatmap_entropy(first_batch['gt_heatmaps'])  # [B, K]
            mask = first_batch.get('mask')
            if mask is not None and mask.sum() > 0:
                H_masked = H[mask.bool()]
                H_med = float(torch.median(H_masked)) if len(H_masked) > 0 else 4.0
            else:
                H_med = float(torch.median(H))

        pass_threshold = H_med + 0.3  # Relative threshold: median(H(q)) + 0.3
        print(f"✓ Target entropy (median): {H_med:.4f}")
        print(f"✓ PASS threshold (H(q) + 0.3): {pass_threshold:.4f}")
        print(f"  Explanation: CE optimal value is H(q), not absolute <1.0")
    except Exception as e:
        print(f"⚠ Could not compute target entropy: {e}")
        print(f"  Using fallback threshold: 4.3")
        pass_threshold = 4.3
        H_med = 4.0

    # Create model with ResNet backbone
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VLNHeatmapModel(
        k_heatmaps=heatmap_per_clip,
        hm_size=hm_size,
        vision_dim=1024,
        agg='mean',
        backbone='resnet18',  # Use pretrained ResNet18
        use_lora=False
    ).to(device)
    model.train()

    # Print parameter count
    param_count = model.get_trainable_parameters()
    print(f"✓ Model created:")
    for component, count in param_count.items():
        print(f"    {component}: {count:,}")

    # Create optimizer with higher LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,  # Higher LR for faster convergence
        weight_decay=1e-4  # Lower WD
    )

    # Training loop
    print("\n" + "-" * 80)
    print("Starting overfit training (NO AMP)...")
    print("-" * 80)

    initial_loss = None
    final_loss = None
    step = 0

    # Infinite loop over small dataset
    for step, batch in zip(range(max_steps), itertools.cycle(dl)):
        # Move to device
        frames = batch['frames'].to(device)
        targets = batch['gt_heatmaps'].to(device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        # Forward pass (return logits, not probabilities)
        logits, aux = model(frames, return_logits=True)

        # Compute loss in logits space
        loss = heatmap_ce_from_logits(logits, targets, mask=mask)

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
            with torch.no_grad():
                logits_std = logits.flatten(1).std(dim=1).mean().item()
                head_grad = grad_norm(model.head.parameters())

            print(f"Step {step:4d}: loss={loss_value:.6f}, "
                  f"logits_std={logits_std:.4f}, head_grad={head_grad:.4f}")

    # Final results
    print("-" * 80)
    print("\nOverfit Test Results:")
    print("=" * 80)
    print(f"Target entropy (median H(q)): {H_med:.6f}")
    print(f"PASS threshold (H(q) + 0.3):  {pass_threshold:.6f}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {initial_loss - final_loss:.6f} ({(1 - final_loss/initial_loss)*100:.1f}% decrease)")
    print()

    # Evaluate success using RELATIVE threshold
    success = final_loss <= pass_threshold
    excellent = final_loss <= H_med + 0.1  # Very close to optimal

    if excellent:
        print(f"✓✓ EXCELLENT: Model reaches optimal CE (loss ≤ {H_med + 0.1:.2f})")
        print("   This proves the model has sufficient capacity and can learn!")
        print("   Ready to proceed with full training.")
    elif success:
        print(f"✓ PASS: Model can learn (loss ≤ {pass_threshold:.2f})")
        print("   Model is learning and approaching the entropy limit.")
        print("   Consider:")
        print("   - Longer training to reach H(q)")
        print("   - Check if logits_std stays healthy (> 0.1)")
    else:
        print(f"✗ FAIL: Model is NOT learning properly (loss > {pass_threshold:.2f})")
        print("   Possible issues:")
        print("   - Check if heatmap_ce_from_logits is being called")
        print("   - Verify logits_std doesn't collapse to 0")
        print("   - Check head_grad is non-zero")
        print("   - Try even higher learning rate (5e-3)")
        print("   Note: Optimal CE is H(q), NOT absolute <1.0")

    print("=" * 80)

    return success


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)  # Show model init messages

    success = main()
    sys.exit(0 if success else 1)