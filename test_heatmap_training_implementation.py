"""
Test script to validate the complete heatmap training implementation.

This script tests all components required by train.md:
1. HeatmapDatasetAdapter
2. VLNHeatmapModel with MultiHeatmapHead and GaussianRenderer
3. KL-CE loss function
4. Training script integration

Run this to verify everything works before full training.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components to test
from src.data.vln_dataset import VLNTrainingDataset, HeatmapDatasetAdapter, create_heatmap_dataloader
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.models.heatmap.multi_head import MultiHeatmapHead
from src.models.heatmap.renderer import GaussianRenderer
from src.utils.losses import kl_ce_loss, create_heatmap_loss_fn, HeatmapLossConfig


def test_dataset_adapter():
    """Test HeatmapDatasetAdapter functionality"""
    print("🧪 Testing HeatmapDatasetAdapter...")

    try:
        # Create dummy base dataset
        class DummyBaseDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                # Return dummy data matching VLNTrainingDataset format
                return {
                    'video_frames': torch.randn(16, 3, 224, 224),  # N_m frames
                    'instruction': f'Navigate to target {idx}',
                    'target_heatmap': torch.randn(224, 224).abs(),  # Positive values
                    'target_points': [(100, 100), (150, 120)],
                    'metadata': {'sample_id': idx},
                    'sample_id': idx
                }

        base_dataset = DummyBaseDataset()

        # Test adapter
        adapter = HeatmapDatasetAdapter(
            base_dataset=base_dataset,
            frames_per_clip=8,      # T
            heatmap_per_clip=4,     # K
            hm_size=(64, 64)
        )

        print(f"  ✓ Adapter created: {len(adapter)} samples")

        # Test sample
        sample = adapter[0]
        expected_keys = {"frames", "text", "gt_heatmaps", "mask", "meta"}
        assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

        frames = sample['frames']
        gt_heatmaps = sample['gt_heatmaps']
        mask = sample['mask']

        print(f"  ✓ Sample structure correct")
        print(f"    frames: {frames.shape}")
        print(f"    gt_heatmaps: {gt_heatmaps.shape}")
        print(f"    mask: {mask.shape}")

        # Verify shapes
        assert frames.shape == (8, 3, 224, 224), f"Frames shape mismatch: {frames.shape}"
        assert gt_heatmaps.shape == (4, 64, 64), f"GT heatmaps shape mismatch: {gt_heatmaps.shape}"
        assert mask.shape == (4,), f"Mask shape mismatch: {mask.shape}"

        # Verify normalization
        for k in range(4):
            hm_sum = gt_heatmaps[k].sum().item()
            if mask[k] > 0.5:  # Valid heatmap
                assert abs(hm_sum - 1.0) < 1e-3, f"Heatmap {k} not normalized: sum={hm_sum}"

        print(f"  ✓ Heatmap normalization verified")
        print("✅ HeatmapDatasetAdapter test passed!\n")
        return True

    except Exception as e:
        print(f"❌ HeatmapDatasetAdapter test failed: {e}")
        return False


def test_model_components():
    """Test VLNHeatmapModel and its components"""
    print("🧪 Testing VLNHeatmapModel components...")

    try:
        # Test MultiHeatmapHead
        print("  Testing MultiHeatmapHead...")
        head = MultiHeatmapHead(in_dim=1024, k_heatmaps=4, hm_size=(64, 64))
        x = torch.randn(2, 1024)
        logits = head(x)
        assert logits.shape == (2, 4, 64, 64), f"Head output shape mismatch: {logits.shape}"
        print(f"    ✓ MultiHeatmapHead: {x.shape} -> {logits.shape}")

        # Test GaussianRenderer
        print("  Testing GaussianRenderer...")
        renderer = GaussianRenderer(hm_size=(64, 64))
        probs = renderer(logits)
        assert probs.shape == (2, 4, 64, 64), f"Renderer output shape mismatch: {probs.shape}"

        # Verify normalization
        probs_sum = probs.sum(dim=(-1, -2))
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), "Probs not normalized"
        print(f"    ✓ GaussianRenderer: normalized probs, sum={probs_sum.mean():.6f}")

        # Test full VLNHeatmapModel
        print("  Testing VLNHeatmapModel...")
        model = VLNHeatmapModel(
            k_heatmaps=4,
            hm_size=(64, 64),
            vision_dim=1024,
            agg='mean'
        )

        frames = torch.randn(2, 8, 3, 224, 224)  # [B, T, C, H, W]
        probs, aux_outputs = model(frames)

        assert probs.shape == (2, 4, 64, 64), f"Model output shape mismatch: {probs.shape}"
        assert 'logits' in aux_outputs, "Missing logits in aux_outputs"
        assert 'renderer_params' in aux_outputs, "Missing renderer_params in aux_outputs"

        print(f"    ✓ VLNHeatmapModel: {frames.shape} -> {probs.shape}")

        # Test curriculum learning
        model.update_hm_size((128, 128))
        probs_new, _ = model(frames)
        assert probs_new.shape == (2, 4, 128, 128), f"Curriculum update failed: {probs_new.shape}"
        print(f"    ✓ Curriculum learning: {(64, 64)} -> {(128, 128)}")

        print("✅ Model components test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss functions"""
    print("🧪 Testing loss functions...")

    try:
        # Create test data
        batch_size = 2
        k_heatmaps = 4
        hm_size = (32, 32)

        # Predicted probabilities (normalized)
        logits = torch.randn(batch_size, k_heatmaps, *hm_size)
        logits_flat = logits.view(batch_size * k_heatmaps, -1)
        pred_probs_flat = torch.softmax(logits_flat, dim=1)
        pred_probs = pred_probs_flat.view(batch_size, k_heatmaps, *hm_size)

        # Target heatmaps (raw values)
        target_maps = torch.randn(batch_size, k_heatmaps, *hm_size)

        # Mask (some invalid)
        mask = torch.ones(batch_size, k_heatmaps)
        mask[0, -1] = 0  # Last heatmap of first batch is invalid

        print(f"  Test shapes: pred_probs={pred_probs.shape}, target_maps={target_maps.shape}, mask={mask.shape}")

        # Test KL-CE loss
        loss_val = kl_ce_loss(pred_probs, target_maps, mask)
        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.item() > 0, f"Loss should be positive: {loss_val.item()}"
        print(f"    ✓ KL-CE loss: {loss_val.item():.4f}")

        # Test without mask
        loss_no_mask = kl_ce_loss(pred_probs, target_maps)
        assert torch.isfinite(loss_no_mask), f"Loss without mask is not finite: {loss_no_mask}"
        print(f"    ✓ KL-CE loss (no mask): {loss_no_mask.item():.4f}")

        # Test loss factory
        config = HeatmapLossConfig(loss_type='kl_ce', focal_enabled=False)
        loss_fn = create_heatmap_loss_fn(config)
        factory_loss = loss_fn(pred_probs, target_maps, mask)
        assert torch.isfinite(factory_loss), f"Factory loss is not finite: {factory_loss}"
        print(f"    ✓ Factory loss: {factory_loss.item():.4f}")

        # Test focal loss
        config_focal = HeatmapLossConfig(loss_type='kl_ce', focal_enabled=True)
        focal_loss_fn = create_heatmap_loss_fn(config_focal)
        focal_loss_val = focal_loss_fn(pred_probs, target_maps, mask)
        assert torch.isfinite(focal_loss_val), f"Focal loss is not finite: {focal_loss_val}"
        print(f"    ✓ Focal loss: {focal_loss_val.item():.4f}")

        print("✅ Loss functions test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_training_step():
    """Test a complete training step"""
    print("🧪 Testing end-to-end training step...")

    try:
        # Create dummy dataset and adapter
        class DummyBaseDataset:
            def __len__(self):
                return 4  # Small batch

            def __getitem__(self, idx):
                return {
                    'video_frames': torch.randn(12, 3, 224, 224),
                    'instruction': f'Task {idx}',
                    'target_heatmap': torch.randn(224, 224).abs(),
                    'target_points': [(100 + idx*10, 100 + idx*10)],
                    'metadata': {},
                    'sample_id': idx
                }

        base_dataset = DummyBaseDataset()

        # Create data loader
        dataloader = create_heatmap_dataloader(
            base_dataset=base_dataset,
            frames_per_clip=8,
            heatmap_per_clip=4,
            hm_size=(64, 64),
            batch_size=2,
            shuffle=False,
            num_workers=0  # No multiprocessing for testing
        )

        print(f"  ✓ DataLoader created with {len(dataloader)} batches")

        # Create model and optimizer
        model = VLNHeatmapModel(
            k_heatmaps=4,
            hm_size=(64, 64),
            vision_dim=1024,
            agg='mean'
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_config = HeatmapLossConfig(loss_type='kl_ce')
        loss_fn = create_heatmap_loss_fn(loss_config)

        print(f"  ✓ Model and optimizer created")

        # Training step
        model.train()
        batch = next(iter(dataloader))

        frames = batch['frames']           # [B, T, 3, H, W]
        gt_heatmaps = batch['gt_heatmaps'] # [B, K, Hm, Wm]
        mask = batch['mask']               # [B, K]

        print(f"  ✓ Batch loaded: frames={frames.shape}, gt_heatmaps={gt_heatmaps.shape}, mask={mask.shape}")

        # Forward pass
        preds, aux_outputs = model(frames)
        loss = loss_fn(preds, gt_heatmaps, mask)

        print(f"  ✓ Forward pass: preds={preds.shape}, loss={loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  ✓ Backward pass completed")

        # Test with different heatmap size (curriculum learning)
        model.update_hm_size((96, 96))

        dataloader_new = create_heatmap_dataloader(
            base_dataset=base_dataset,
            frames_per_clip=8,
            heatmap_per_clip=4,
            hm_size=(96, 96),
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        batch_new = next(iter(dataloader_new))
        frames_new = batch_new['frames']
        gt_heatmaps_new = batch_new['gt_heatmaps']
        mask_new = batch_new['mask']

        preds_new, _ = model(frames_new)
        loss_new = loss_fn(preds_new, gt_heatmaps_new, mask_new)

        assert preds_new.shape == (2, 4, 96, 96), f"Curriculum learning failed: {preds_new.shape}"

        print(f"  ✓ Curriculum learning: {preds_new.shape}, loss={loss_new.item():.4f}")

        print("✅ End-to-end training step test passed!\n")
        return True

    except Exception as e:
        print(f"❌ End-to-end training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test configuration file loading"""
    print("🧪 Testing configuration loading...")

    try:
        import yaml

        config_path = PROJECT_ROOT / "configs" / "heatmap_training_config.yaml"

        if not config_path.exists():
            print(f"  ⚠️ Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required keys
        required_keys = ['data', 'training', 'optim', 'loss', 'log']
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"

        # Check training stages
        assert 'stages' in config['training'], "Missing training.stages"
        assert len(config['training']['stages']) > 0, "No training stages defined"

        for stage in config['training']['stages']:
            required_stage_keys = ['name', 'epochs', 'freeze_llm', 'hm_size']
            for key in required_stage_keys:
                assert key in stage, f"Missing stage key: {key}"

        print(f"  ✓ Configuration file valid: {len(config['training']['stages'])} stages")
        print(f"    Stages: {[s['name'] for s in config['training']['stages']]}")

        print("✅ Configuration loading test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting heatmap training implementation tests...\n")

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing

    tests = [
        test_dataset_adapter,
        test_model_components,
        test_loss_functions,
        test_end_to_end_training_step,
        test_configuration_loading
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_fn.__name__} crashed: {e}")
            failed += 1

    print("=" * 60)
    print(f"🏁 TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 ALL TESTS PASSED! Implementation is ready for training.")
        print("\nNext steps:")
        print("1. Prepare your dataset and update config['data']['root']")
        print("2. Run training with: python scripts/train_heatmap.py --config configs/heatmap_training_config.yaml")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)