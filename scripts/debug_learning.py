"""
Debug Learning Issue

Investigate why the model is not learning in overfit test.

Checks:
1. Renderer parameters (τ, σ, α)
2. Output distribution properties
3. Gradient magnitudes
4. Loss computation correctness
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss


def main():
    print("=" * 80)
    print("Debugging Learning Issue")
    print("=" * 80)

    # Load data
    ds = VLNHeatmapDataset(
        root='./data/habitat_vln', split='train',
        frames_per_clip=8, heatmap_per_clip=4,
        image_size=(384, 384), hm_size=(64, 64)
    )
    dl = DataLoader(ds, batch_size=1)
    batch = next(iter(dl))

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VLNHeatmapModel(
        k_heatmaps=4, hm_size=(64, 64), vision_dim=1024,
        agg='mean', use_lora=False
    ).to(device).train()

    frames = batch['frames'].to(device)
    targets = batch['gt_heatmaps'].to(device)
    mask = batch.get('mask')
    if mask is not None:
        mask = mask.to(device)

    # Forward pass
    preds, aux = model(frames)
    loss = kl_ce_loss(preds, targets, mask=mask)

    print("\n1. Renderer Parameters:")
    print("-" * 80)
    renderer_params = aux['renderer_params']
    print(f"  τ (temperature): {renderer_params['tau']:.6f}")
    print(f"  σ (blur):        {renderer_params['sigma']:.6f}")
    print(f"  α (mixing):      {renderer_params['alpha']:.6f}")

    print("\n2. Output Statistics:")
    print("-" * 80)
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Predictions range: [{preds.min():.6e}, {preds.max():.6e}]")
    print(f"  Predictions mean:  {preds.mean():.6e}")
    print(f"  Predictions std:   {preds.std():.6e}")

    # Check normalization
    pred_sums = preds.sum(dim=(-1, -2))
    print(f"  Sum per heatmap:   {pred_sums[0].tolist()}")

    # Check entropy (how uniform is the distribution?)
    entropy = -(preds * torch.log(preds.clamp(min=1e-8))).sum(dim=(-1, -2))
    max_entropy = torch.log(torch.tensor(64.0 * 64.0))
    print(f"  Entropy:           {entropy[0].tolist()}")
    print(f"  Max entropy:       {max_entropy:.6f} (uniform)")
    print(f"  Normalized:        {(entropy[0] / max_entropy).tolist()} (1.0 = uniform)")

    print("\n3. Target Statistics:")
    print("-" * 80)
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets range: [{targets.min():.6e}, {targets.max():.6e}]")
    print(f"  Mask:          {mask[0].tolist() if mask is not None else 'None'}")

    print("\n4. Loss:")
    print("-" * 80)
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Expected for uniform: ~8.317")

    # Backward pass
    print("\n5. Gradient Check:")
    print("-" * 80)
    model.zero_grad()
    loss.backward()

    # Check gradients for different components
    print("  Component        | Max Grad | Mean Grad | Has Grad?")
    print("  " + "-" * 60)

    # Renderer parameters
    if hasattr(model.renderer, 'log_tau'):
        print(f"  renderer.log_tau | {model.renderer.log_tau.grad.abs().max():.6e} | {model.renderer.log_tau.grad.abs().mean():.6e} | {model.renderer.log_tau.grad is not None}")
    if hasattr(model.renderer, 'log_sigma'):
        print(f"  renderer.log_sigma| {model.renderer.log_sigma.grad.abs().max():.6e} | {model.renderer.log_sigma.grad.abs().mean():.6e} | {model.renderer.log_sigma.grad is not None}")
    if hasattr(model.renderer, 'alpha'):
        print(f"  renderer.alpha   | {model.renderer.alpha.grad.abs().max():.6e} | {model.renderer.alpha.grad.abs().mean():.6e} | {model.renderer.alpha.grad is not None}")

    # Head parameters
    for name, param in model.head.named_parameters():
        if param.grad is not None:
            print(f"  head.{name:12s} | {param.grad.abs().max():.6e} | {param.grad.abs().mean():.6e} | True")
        else:
            print(f"  head.{name:12s} | N/A      | N/A      | False")

    # Backbone (first layer only)
    backbone_grads = []
    for name, param in model.backbone.named_parameters():
        if param.grad is not None:
            backbone_grads.append(param.grad.abs().max().item())
    if backbone_grads:
        print(f"  backbone (all)   | {max(backbone_grads):.6e} | {sum(backbone_grads)/len(backbone_grads):.6e} | True")

    print("\n6. Diagnosis:")
    print("=" * 80)

    # Check if outputs are too uniform
    avg_normalized_entropy = (entropy[0] / max_entropy).mean().item()
    if avg_normalized_entropy > 0.99:
        print("  ⚠️  OUTPUT TOO UNIFORM (entropy ≈ max)")
        print("      Predictions are nearly uniform distribution")
        print("      Possible causes:")
        print("      - Temperature τ too high (outputs get softened too much)")
        print("      - Blur σ too high (smooths everything to uniform)")
        print("      - Head outputs have very small magnitude (logits ≈ 0)")

    # Check if gradients are flowing
    if max(backbone_grads) < 1e-6:
        print("  ⚠️  GRADIENTS TOO SMALL")
        print("      Backbone gradients are vanishingly small")
        print("      This prevents learning")

    # Check logits
    logits = aux['logits']
    print(f"\n7. Logits (before renderer):")
    print("-" * 80)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.6e}, {logits.max():.6e}]")
    print(f"  Logits std:   {logits.std():.6e}")

    if logits.std() < 0.1:
        print("  ⚠️  LOGITS TOO SMALL")
        print("      Head is outputting nearly zero logits")
        print("      This causes uniform distributions after softmax")
        print("      Possible cause: Placeholder backbone too weak")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()