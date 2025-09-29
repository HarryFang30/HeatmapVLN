"""
MultiHeatmapHead for generating K heatmaps from fused features.

This module implements the MLP head that converts spatial features to
K frame-indexed heatmaps as specified in train.md.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MultiHeatmapHead(nn.Module):
    """
    Multi-heatmap head that generates K heatmaps from input features.

    Architecture:
        input_features -> MLP -> K×Hm×Wm logits -> reshape -> [K, Hm, Wm]
    """

    def __init__(self, in_dim: int, k_heatmaps: int, hm_size: Tuple[int, int]):
        """
        Initialize MultiHeatmapHead.

        Args:
            in_dim: Input feature dimension
            k_heatmaps: Number of heatmaps to generate (K)
            hm_size: Heatmap spatial size (Hm, Wm)
        """
        super().__init__()

        self.k = k_heatmaps
        self.hm_size = hm_size
        Hm, Wm = hm_size

        # Output dimension: K heatmaps × spatial size
        out_dim = k_heatmaps * Hm * Wm

        # Simple MLP as specified in train.md
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape [B, in_dim]

        Returns:
            torch.Tensor: Logits of shape [B, K, Hm, Wm]
        """
        B = x.size(0)

        # MLP forward
        y = self.mlp(x)  # [B, K×Hm×Wm]

        # Reshape to [B, K, Hm, Wm]
        y = y.view(B, self.k, *self.hm_size)

        return y  # logits

    def update_hm_size(self, new_hm_size: Tuple[int, int]):
        """
        Update heatmap size for curriculum learning.

        Args:
            new_hm_size: New heatmap size (Hm, Wm)
        """
        if new_hm_size != self.hm_size:
            Hm, Wm = new_hm_size
            new_out_dim = self.k * Hm * Wm

            # Get current MLP
            old_mlp = self.mlp
            in_dim = old_mlp[0].in_features

            # Create new MLP with updated output dimension
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, new_out_dim)
            )

            # Transfer weights from old MLP (first layer)
            with torch.no_grad():
                self.mlp[0].weight.copy_(old_mlp[0].weight)
                self.mlp[0].bias.copy_(old_mlp[0].bias)
                # Second layer is randomly initialized for new size
                nn.init.xavier_uniform_(self.mlp[2].weight)
                nn.init.zeros_(self.mlp[2].bias)

            # Update size
            self.hm_size = new_hm_size

            # Move to same device as old model
            if next(old_mlp.parameters()).is_cuda:
                self.mlp = self.mlp.to(next(old_mlp.parameters()).device)

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return f'k_heatmaps={self.k}, hm_size={self.hm_size}'


# Test and example usage
if __name__ == "__main__":
    # Test MultiHeatmapHead
    batch_size = 4
    in_dim = 1024
    k_heatmaps = 4
    hm_size = (64, 64)

    # Create head
    head = MultiHeatmapHead(in_dim=in_dim, k_heatmaps=k_heatmaps, hm_size=hm_size)

    # Test forward pass
    x = torch.randn(batch_size, in_dim)
    logits = head(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: [B={batch_size}, K={k_heatmaps}, Hm={hm_size[0]}, Wm={hm_size[1]}]")

    # Test curriculum learning (size update)
    print(f"\nTesting curriculum learning:")
    print(f"Original hm_size: {head.hm_size}")

    new_size = (128, 128)
    head.update_hm_size(new_size)
    print(f"Updated hm_size: {head.hm_size}")

    # Test with new size
    logits_new = head(x)
    print(f"New output shape: {logits_new.shape}")
    print(f"Expected shape: [B={batch_size}, K={k_heatmaps}, Hm={new_size[0]}, Wm={new_size[1]}]")

    print("\nMultiHeatmapHead test completed successfully!")