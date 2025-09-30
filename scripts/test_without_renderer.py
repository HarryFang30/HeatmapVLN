"""
Test learning WITHOUT the Gaussian Renderer

This bypasses the renderer to see if the head alone can learn.
If this works, the problem is in the renderer.
If this still doesn't work, the problem is more fundamental.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import itertools

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss


def main():
    print("=" * 80)
    print("Test Learning WITHOUT Renderer")
    print("=" * 80)

    # Load data
    ds = VLNHeatmapDataset(
        root='./data/habitat_vln', split='train',
        frames_per_clip=8, heatmap_per_clip=4,
        image_size=(384, 384), hm_size=(64, 64)
    )
    small_ds = Subset(ds, [0])
    dl = DataLoader(small_ds, batch_size=1, shuffle=True)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VLNHeatmapModel(
        k_heatmaps=4, hm_size=(64, 64), vision_dim=1024,
        agg='mean', use_lora=False
    ).to(device).train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    print(f"Model parameters: {model.get_trainable_parameters()['total']:,}")
    print("Training for 800 steps WITHOUT using renderer...")
    print("(Using head logits directly + softmax)\n")

    for step, batch in zip(range(800), itertools.cycle(dl)):
        frames = batch['frames'].to(device)
        targets = batch['gt_heatmaps'].to(device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        # Forward through backbone and head ONLY (skip renderer)
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        feats_flat = model.backbone(frames_flat)
        feats = feats_flat.view(B, T, model.vision_dim)
        fused = feats.mean(dim=1)  # [B, vision_dim]
        logits = model.head(fused)  # [B, K, Hm, Wm]

        # Simple softmax (no temperature, no blur)
        B, K, Hm, Wm = logits.shape
        logits_flat = logits.view(B, K, -1)
        preds_flat = F.softmax(logits_flat, dim=-1)
        preds = preds_flat.view(B, K, Hm, Wm)

        # Compute loss
        loss = kl_ce_loss(preds, targets, mask=mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d}: loss={float(loss):.6f}, logits_std={logits.std():.6f}")

    print(f"\nFinal loss: {float(loss):.6f}")
    if float(loss) < 3.0:
        print("✓ SUCCESS: Model CAN learn without renderer!")
        print("   Problem is in the Gaussian Renderer")
    else:
        print("✗ FAIL: Model still cannot learn")
        print("   Problem is more fundamental (backbone/head/data)")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()