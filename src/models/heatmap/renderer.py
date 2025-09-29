"""
GaussianRenderer for converting logits to probability distributions with learnable parameters.

This module implements the renderer that applies:
- Temperature scaling (τ)
- Gaussian blur (σ)
- Mixing coefficient (α)

As specified in train.md section 5(b).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


def gaussian_separable_blur(input_tensor: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Apply separable Gaussian blur (two 1D convolutions).

    Args:
        input_tensor: Input tensor of shape [B, K, H, W]
        sigma: Blur sigma parameter

    Returns:
        torch.Tensor: Blurred tensor of same shape
    """
    if sigma <= 1e-4:  # Skip blur if sigma too small
        return input_tensor

    B, K, H, W = input_tensor.shape

    # Calculate kernel size (odd number)
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    kernel_size = max(kernel_size, 3)  # Minimum size 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    half_size = kernel_size // 2
    x = torch.arange(-half_size, half_size + 1, dtype=input_tensor.dtype, device=input_tensor.device)
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Reshape for conv1d
    input_reshaped = input_tensor.view(B * K, 1, H, W)

    # Apply horizontal blur (along W dimension)
    # Reshape to [B*K*H, 1, W] for 1D conv
    h_input = input_reshaped.permute(0, 2, 1, 3).contiguous().view(B * K * H, 1, W)
    h_kernel = gaussian_kernel.view(1, 1, -1)
    h_blurred = F.conv1d(h_input, h_kernel, padding=half_size)
    h_blurred = h_blurred.view(B * K, H, 1, W).permute(0, 2, 1, 3)

    # Apply vertical blur (along H dimension)
    # Reshape to [B*K*W, 1, H] for 1D conv
    v_input = h_blurred.permute(0, 3, 1, 2).contiguous().view(B * K * W, 1, H)
    v_kernel = gaussian_kernel.view(1, 1, -1)
    v_blurred = F.conv1d(v_input, v_kernel, padding=half_size)
    v_blurred = v_blurred.view(B * K, W, 1, H).permute(0, 2, 3, 1)

    # Reshape back to original
    output = v_blurred.permute(0, 2, 3, 1).contiguous().view(B, K, H, W)

    return output


class GaussianRenderer(nn.Module):
    """
    Gaussian renderer with learnable temperature, blur, and mixing parameters.

    Converts logits to probability distributions using:
    1. Temperature scaling: softmax(logits / τ)
    2. Gaussian blur: σ parameter
    3. Mixing: α * blurred + (1-α) * unblurred
    """

    def __init__(self, hm_size: Tuple[int, int]):
        """
        Initialize GaussianRenderer.

        Args:
            hm_size: Heatmap size (H, W)
        """
        super().__init__()

        self.hm_size = hm_size

        # Learnable parameters (as specified in train.md)
        self.log_tau = nn.Parameter(torch.zeros(1))      # Temperature: exp(log_tau)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.5)))  # Blur: exp(log_sigma)
        self.alpha = nn.Parameter(torch.tensor(0.6))     # Mixing coefficient

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probability distributions.

        Args:
            logits: Input logits of shape [B, K, Hm, Wm]

        Returns:
            torch.Tensor: Probability distributions of shape [B, K, Hm, Wm]
                         Each heatmap sums to 1
        """
        # Clamp temperature to reasonable range
        tau = torch.exp(self.log_tau).clamp(1e-2, 10.0)

        # Initial softmax with temperature
        # Flatten spatial dimensions for softmax, then reshape
        B, K, Hm, Wm = logits.shape
        logits_flat = (logits / tau).view(B, K, -1)  # [B, K, Hm*Wm]
        probs_flat = torch.softmax(logits_flat, dim=-1)  # Softmax over spatial dimension
        probs_unblurred = probs_flat.view(B, K, Hm, Wm)  # Reshape back

        # Gaussian blur
        sigma = torch.exp(self.log_sigma)
        probs_blurred = gaussian_separable_blur(probs_unblurred, sigma)

        # Renormalize after blur to ensure sum=1
        probs_blurred = probs_blurred / (probs_blurred.sum(dim=(-1, -2), keepdim=True) + 1e-8)

        # Mix blurred and unblurred
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha in [0, 1]

        # Recompute unblurred for mixing (ensure same normalization)
        logits_flat_final = logits.view(B, K, -1) / tau
        probs_unblurred_final = torch.softmax(logits_flat_final, dim=-1).view(B, K, Hm, Wm)

        probs = alpha * probs_blurred + (1 - alpha) * probs_unblurred_final

        # Final normalization to ensure each heatmap sums to 1
        probs = probs / (probs.sum(dim=(-1, -2), keepdim=True) + 1e-8)

        return probs

    def update_hm_size(self, new_hm_size: Tuple[int, int]):
        """
        Update heatmap size for curriculum learning.

        Args:
            new_hm_size: New heatmap size (H, W)
        """
        self.hm_size = new_hm_size
        # Note: GaussianRenderer doesn't have size-dependent parameters,
        # so no parameter updates needed

    def get_parameters_info(self) -> dict:
        """
        Get current parameter values for monitoring.

        Returns:
            dict: Dictionary with current parameter values
        """
        return {
            'tau': torch.exp(self.log_tau).item(),
            'sigma': torch.exp(self.log_sigma).item(),
            'alpha': torch.sigmoid(self.alpha).item()
        }

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        params = self.get_parameters_info()
        return f"hm_size={self.hm_size}, tau={params['tau']:.3f}, sigma={params['sigma']:.3f}, alpha={params['alpha']:.3f}"


# Test and example usage
if __name__ == "__main__":
    # Test GaussianRenderer
    batch_size = 2
    k_heatmaps = 4
    hm_size = (64, 64)

    # Create renderer
    renderer = GaussianRenderer(hm_size=hm_size)

    # Create test logits
    logits = torch.randn(batch_size, k_heatmaps, *hm_size)

    print(f"Input logits shape: {logits.shape}")

    # Forward pass
    probs = renderer(logits)

    print(f"Output probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")

    # Check normalization (each heatmap should sum to 1)
    probs_sum = probs.sum(dim=(-1, -2))
    print(f"Normalization check (should be close to 1.0):")
    print(f"  Min sum: {probs_sum.min().item():.6f}")
    print(f"  Max sum: {probs_sum.max().item():.6f}")
    print(f"  Mean sum: {probs_sum.mean().item():.6f}")

    # Check parameter values
    params = renderer.get_parameters_info()
    print(f"\nLearnable parameters:")
    print(f"  τ (temperature): {params['tau']:.3f}")
    print(f"  σ (blur): {params['sigma']:.3f}")
    print(f"  α (mixing): {params['alpha']:.3f}")

    # Test curriculum learning
    print(f"\nTesting curriculum learning:")
    print(f"Original hm_size: {renderer.hm_size}")

    new_size = (128, 128)
    renderer.update_hm_size(new_size)
    print(f"Updated hm_size: {renderer.hm_size}")

    # Test with new size
    logits_new = torch.randn(batch_size, k_heatmaps, *new_size)
    probs_new = renderer(logits_new)
    print(f"New output shape: {probs_new.shape}")

    # Check normalization with new size
    probs_new_sum = probs_new.sum(dim=(-1, -2))
    print(f"New size normalization check:")
    print(f"  Mean sum: {probs_new_sum.mean().item():.6f}")

    print("\nGaussianRenderer test completed successfully!")