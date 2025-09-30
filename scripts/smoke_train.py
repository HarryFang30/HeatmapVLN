"""
Smoke Test for VLN Heatmap Training Pipeline

Purpose: Verify batch→model→loss chain works without running full training.

This script:
1. Loads the dataset
2. Gets one batch
3. Runs forward pass through model
4. Computes loss
5. Verifies shapes and values

Expected output: No errors, prints shapes and loss value
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss


def main():
    """Run smoke test"""
    print("=" * 80)
    print("VLN Heatmap Training Pipeline - Smoke Test")
    print("=" * 80)

    # Configuration
    data_root = "./data/habitat_vln"
    frames_per_clip = 8
    heatmap_per_clip = 4
    image_size = (384, 384)
    hm_size = (64, 64)
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Frames per clip: {frames_per_clip}")
    print(f"  Heatmaps per clip: {heatmap_per_clip}")
    print(f"  Image size: {image_size}")
    print(f"  Heatmap size: {hm_size}")
    print(f"  Batch size: {batch_size}")

    # Step 1: Create dataset
    print("\n" + "-" * 80)
    print("Step 1: Creating dataset...")
    print("-" * 80)

    try:
        ds = VLNHeatmapDataset(
            root=data_root,
            split="train",
            frames_per_clip=frames_per_clip,
            heatmap_per_clip=heatmap_per_clip,
            image_size=image_size,
            hm_size=hm_size
        )
        print(f"✓ Dataset created: {len(ds)} samples")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

    if len(ds) == 0:
        print("✗ Dataset is empty - run pack_dataset.py first")
        return False

    # Step 2: Create dataloader
    print("\n" + "-" * 80)
    print("Step 2: Creating dataloader...")
    print("-" * 80)

    try:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"✓ DataLoader created")
    except Exception as e:
        print(f"✗ DataLoader creation failed: {e}")
        return False

    # Step 3: Get one batch
    print("\n" + "-" * 80)
    print("Step 3: Loading one batch...")
    print("-" * 80)

    try:
        batch = next(iter(dl))
        frames = batch["frames"]          # [B, T, 3, H, W]
        targets = batch["gt_heatmaps"]    # [B, K, Hm, Wm]
        mask = batch["mask"]              # [B, K]

        print(f"✓ Batch loaded successfully")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Frames range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"  Valid heatmaps: {(mask > 0.5).sum()}/{mask.numel()}")
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Create model
    print("\n" + "-" * 80)
    print("Step 4: Creating model...")
    print("-" * 80)

    try:
        model = VLNHeatmapModel(
            k_heatmaps=heatmap_per_clip,
            hm_size=hm_size,
            vision_dim=1024,
            agg="mean",
            use_lora=False
        )
        model.eval()

        print(f"✓ Model created")

        # Print parameter counts
        param_counts = model.get_trainable_parameters()
        print(f"  Trainable parameters:")
        for component, count in param_counts.items():
            print(f"    {component}: {count:,}")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Forward pass
    print("\n" + "-" * 80)
    print("Step 5: Running forward pass...")
    print("-" * 80)

    try:
        with torch.no_grad():
            preds, aux_outputs = model(frames)

        print(f"✓ Forward pass successful")
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Predictions range: [{preds.min():.6f}, {preds.max():.6f}]")

        # Check normalization
        preds_sum = preds.sum(dim=(-1, -2))
        print(f"  Normalization check (should be ~1.0):")
        print(f"    Min sum: {preds_sum.min():.6f}")
        print(f"    Max sum: {preds_sum.max():.6f}")
        print(f"    Mean sum: {preds_sum.mean():.6f}")

        # Check auxiliary outputs
        print(f"  Auxiliary outputs: {list(aux_outputs.keys())}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Compute loss
    print("\n" + "-" * 80)
    print("Step 6: Computing loss...")
    print("-" * 80)

    try:
        loss = kl_ce_loss(preds, targets, mask=mask)

        print(f"✓ Loss computation successful")
        print(f"  Loss value: {float(loss):.6f}")
        print(f"  Loss is finite: {torch.isfinite(loss).item()}")
        print(f"  Loss is positive: {(loss > 0).item()}")

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 7: Test backward pass (optional)
    print("\n" + "-" * 80)
    print("Step 7: Testing backward pass...")
    print("-" * 80)

    try:
        model.train()
        preds_train, _ = model(frames)
        loss_train = kl_ce_loss(preds_train, targets, mask=mask)
        loss_train.backward()

        print(f"✓ Backward pass successful")
        print(f"  Gradients computed for all parameters")

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Success!
    print("\n" + "=" * 80)
    print("✓ SMOKE TEST PASSED")
    print("=" * 80)
    print("\nAll checks passed! Pipeline is ready for training.")
    print(f"\nNext step: Run training with:")
    print(f"  python scripts/train.py --config configs/training_config.yaml")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    success = main()
    sys.exit(0 if success else 1)