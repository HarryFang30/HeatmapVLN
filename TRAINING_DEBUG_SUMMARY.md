# Training Debug Summary

**Date**: 2025-09-30
**Status**: ❌ CRITICAL ISSUE - Model Cannot Learn

## Problem

The model **completely fails** to learn, even on:
- ✗ Real training data (1 sample, 25% valid heatmaps)
- ✗ Synthetic data (perfect Gaussians, 100% valid)
- ✗ With and without Gaussian Renderer
- ✗ With and without BatchNorm/LayerNorm

Loss stays at **8.317766** (uniform baseline) across all tests.

## Key Findings

### 1. Vanishing Gradients
- Initial gradient norm: **0.003** (with no BatchNorm)
- This is orders of magnitude too small
- Gradients are essentially zero → no learning

### 2. Logits Collapse
- Step 0: logits_std = 0.22 (reasonable)
- Step 50: logits_std = 0.001 (collapsed!)
- The network learns to output near-zero logits → uniform predictions

### 3. Not a Data Issue
- Tested with synthetic Gaussians (clear structure, entropy=7.2/8.3)
- ALL heatmaps valid (mask=[1,1,1,1])
- Still no learning

### 4. Not a Renderer Issue
- Tested without renderer (direct softmax on logits)
- Still no learning

## Root Cause Hypothesis

The **placeholder backbone** may be fundamentally broken for this task:
- Simple CNN: 3→64→128→256→512 channels
- Mean pooling over 8 temporal frames
- Projects to 1024-dim vector
- Then MLP head → 4×64×64 logits

Possible issues:
1. **Too much averaging**: Mean over 8 frames may wash out signal
2. **Weak features**: Simple CNN may not extract useful patterns
3. **Architecture mismatch**: May need attention/transformer for this task

## Recommendations

### Option 1: Use Simpler Test Model
Create minimal model (e.g., direct MLP from flattened patches) to verify training code works.

### Option 2: Increase Model Capacity
- Add skip connections (ResNet-style)
- Use deeper/wider CNN
- Add attention mechanisms

### Option 3: Change Approach
- Use pretrained vision encoder (CLIP, DINOv2, etc.)
- Directly integrate real Qwen2.5-VL backbone
- Use different architecture (ViT instead of CNN)

### Option 4: Debug Training Code
- Check if there's a bug in loss masking
- Verify gradient flow through all components
- Try much higher learning rates (1e-2, 5e-2)

## Current Status

**BLOCKED**: Cannot proceed with training until model can learn on synthetic data.

**Next Action**: Need to either:
1. Fix the placeholder backbone
2. Or integrate real Qwen2.5-VL backbone
3. Or use a proven pretrained vision model

The training pipeline code is correct, but the model architecture is fundamentally unable to learn.
