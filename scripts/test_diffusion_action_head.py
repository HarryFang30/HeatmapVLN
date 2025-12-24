#!/usr/bin/env python3
"""
Test script for Diffusion Action Head

Verifies the diffusion-based action generation module works correctly.
Tests:
1. Module initialization
2. Forward pass (inference mode)
3. Training mode with loss computation
4. Action normalization/unnormalization
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict

def test_action_config():
    """Test DiffusionActionConfig initialization."""
    print("\n" + "="*60)
    print("Test 1: DiffusionActionConfig")
    print("="*60)
    
    from src.models.action import DiffusionActionConfig
    
    # Default config
    config = DiffusionActionConfig()
    print(f"  Default config created:")
    print(f"    action_dim: {config.action_dim}")
    print(f"    pred_horizon: {config.pred_horizon}")
    print(f"    cond_dim: {config.cond_dim}")
    print(f"    encoding_size: {config.encoding_size}")
    print(f"    num_diffusion_iters: {config.num_diffusion_iters}")
    print(f"    down_dims: {config.down_dims}")
    
    # Custom config (update action_stats to match action_dim)
    custom_config = DiffusionActionConfig(
        action_dim=3,
        pred_horizon=4,
        cond_dim=1024,
        num_diffusion_iters=20,
        action_stats_min=[-1.0, -1.0, -1.0],  # Match action_dim=3
        action_stats_max=[1.0, 1.0, 1.0],
    )
    print(f"\n  Custom config created:")
    print(f"    action_dim: {custom_config.action_dim}")
    print(f"    pred_horizon: {custom_config.pred_horizon}")
    
    print("  ✓ Config test passed")
    return True


def test_action_utils():
    """Test action normalization utilities."""
    print("\n" + "="*60)
    print("Test 2: Action Normalization Utilities")
    print("="*60)
    
    from src.models.action import ActionStats, normalize_actions, unnormalize_actions
    
    # Create stats
    stats = ActionStats(min=[-1.0, -1.0], max=[1.0, 1.0])
    print(f"  ActionStats: min={stats.min}, max={stats.max}")
    
    # Test numpy
    actions_np = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0]])
    normalized_np = normalize_actions(actions_np, stats)
    recovered_np = unnormalize_actions(normalized_np, stats)
    
    print(f"\n  NumPy test:")
    print(f"    Original: {actions_np}")
    print(f"    Normalized: {normalized_np}")
    print(f"    Recovered: {recovered_np}")
    
    np_error = np.abs(actions_np - recovered_np).max()
    print(f"    Max error: {np_error:.6f}")
    assert np_error < 1e-5, "NumPy normalization error too large"
    
    # Test torch
    actions_t = torch.tensor([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0]])
    normalized_t = normalize_actions(actions_t, stats)
    recovered_t = unnormalize_actions(normalized_t, stats)
    
    print(f"\n  Torch test:")
    print(f"    Original: {actions_t}")
    print(f"    Normalized: {normalized_t}")
    print(f"    Recovered: {recovered_t}")
    
    torch_error = (actions_t - recovered_t).abs().max().item()
    print(f"    Max error: {torch_error:.6f}")
    assert torch_error < 1e-5, "Torch normalization error too large"
    
    print("  ✓ Normalization test passed")
    return True


def test_diffusion_components():
    """Test diffusion core components."""
    print("\n" + "="*60)
    print("Test 3: Diffusion Core Components")
    print("="*60)
    
    from src.models.action.diffusion import (
        ConditionalUnet1D, 
        SinusoidalPosEmb,
        Conv1dBlock
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Test SinusoidalPosEmb
    pos_emb = SinusoidalPosEmb(dim=256).to(device)
    timesteps = torch.tensor([0, 5, 9], device=device)
    emb = pos_emb(timesteps)
    print(f"\n  SinusoidalPosEmb:")
    print(f"    Input: {timesteps.shape}")
    print(f"    Output: {emb.shape}")
    assert emb.shape == (3, 256), "SinusoidalPosEmb output shape mismatch"
    
    # Test Conv1dBlock
    conv_block = Conv1dBlock(64, 128, kernel_size=3).to(device)
    x = torch.randn(2, 64, 16, device=device)
    y = conv_block(x)
    print(f"\n  Conv1dBlock:")
    print(f"    Input: {x.shape}")
    print(f"    Output: {y.shape}")
    assert y.shape == (2, 128, 16), "Conv1dBlock output shape mismatch"
    
    # Test ConditionalUnet1D
    unet = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=768,
        down_dims=[192, 384, 768],
    ).to(device)
    
    sample = torch.randn(2, 1, 2, device=device)  # (B, horizon, action_dim)
    timestep = torch.tensor([5, 5], device=device)
    global_cond = torch.randn(2, 768, device=device)
    
    noise_pred = unet(sample, timestep, global_cond=global_cond)
    print(f"\n  ConditionalUnet1D:")
    print(f"    Sample: {sample.shape}")
    print(f"    Timestep: {timestep.shape}")
    print(f"    Global cond: {global_cond.shape}")
    print(f"    Noise pred: {noise_pred.shape}")
    assert noise_pred.shape == sample.shape, "ConditionalUnet1D output shape mismatch"
    
    print("  ✓ Diffusion components test passed")
    return True


def test_diffusion_action_head():
    """Test DiffusionActionHead initialization and forward pass."""
    print("\n" + "="*60)
    print("Test 4: DiffusionActionHead")
    print("="*60)
    
    from src.models.action import DiffusionActionHead, DiffusionActionConfig
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Create config
    config = DiffusionActionConfig(
        action_dim=2,
        pred_horizon=1,
        cond_dim=2048,
        encoding_size=768,
        num_diffusion_iters=10,
    )
    
    # Create model
    model = DiffusionActionHead(config).to(device)
    print(f"\n  Model created:")
    print(f"    Action dim: {model.action_dim}")
    print(f"    Pred horizon: {model.pred_horizon}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total params: {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    
    # Test inference
    print("\n  Testing inference mode:")
    model.eval()
    batch_size = 2
    seq_len = 16
    llm_features = torch.randn(batch_size, seq_len, 2048, device=device)
    
    with torch.no_grad():
        actions = model(llm_features)
    
    print(f"    LLM features: {llm_features.shape}")
    print(f"    Actions output: {actions.shape}")
    print(f"    Actions sample: {actions[0]}")
    
    expected_shape = (batch_size, config.pred_horizon, config.action_dim)
    assert actions.shape == expected_shape, f"Action shape mismatch: {actions.shape} != {expected_shape}"
    
    # Test training mode
    print("\n  Testing training mode:")
    model.train()
    gt_actions = torch.randn(batch_size, config.pred_horizon, config.action_dim, device=device)
    
    result = model(llm_features, gt_actions=gt_actions, return_loss=True)
    
    print(f"    GT actions: {gt_actions.shape}")
    print(f"    Loss: {result['loss'].item():.4f}")
    print(f"    Pred actions: {result['actions'].shape}")
    
    assert 'loss' in result, "Loss not in result"
    assert result['loss'].requires_grad, "Loss should require grad"
    
    print("  ✓ DiffusionActionHead test passed")
    return True


def test_pipeline_integration():
    """Test action head integration with SpatialMLLMPipeline config."""
    print("\n" + "="*60)
    print("Test 5: Pipeline Integration (Config Only)")
    print("="*60)
    
    from src.models.spatial_mllm_compat import SpatialMLLMIntegrationConfig
    
    # Test config with action head enabled
    config = SpatialMLLMIntegrationConfig(
        enable_action_head=True,
        action_dim=2,
        action_pred_horizon=1,
        action_encoding_size=768,
        action_num_diffusion_iters=10,
    )
    
    print(f"  Config created with action head settings:")
    print(f"    enable_action_head: {config.enable_action_head}")
    print(f"    action_dim: {config.action_dim}")
    print(f"    action_pred_horizon: {config.action_pred_horizon}")
    print(f"    action_encoding_size: {config.action_encoding_size}")
    print(f"    action_num_diffusion_iters: {config.action_num_diffusion_iters}")
    
    print("  ✓ Pipeline integration config test passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  DIFFUSION ACTION HEAD TEST SUITE")
    print("="*60)
    
    tests = [
        ("Action Config", test_action_config),
        ("Action Utils", test_action_utils),
        ("Diffusion Components", test_diffusion_components),
        ("Diffusion Action Head", test_diffusion_action_head),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    results: Dict[str, bool] = {}
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())

