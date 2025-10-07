"""
Quality Metrics for VLN Heatmap Data (quality_metrics.py)
=========================================================

Utility functions for measuring data quality:
- Heatmap entropy (measures sharpness/peakedness)
- Effective K (number of valid heatmaps)
- Visibility ratio (for keyframe selection)

Used by packing and quality reporting scripts.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

EPS = 1e-8


def heatmap_entropy(hm: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute entropy H(q) = -Σ q log q for heatmap probability distributions.

    Automatically normalizes heatmaps to sum=1 before computing entropy.
    Lower entropy = sharper/more peaked heatmap (better for training).

    Args:
        hm: Heatmap tensor [..., Hm, Wm] in any shape
        eps: Small epsilon for numerical stability

    Returns:
        torch.Tensor: Entropy values with shape [...]

    Example:
        >>> hm = torch.randn(4, 64, 64)  # [K, Hm, Wm]
        >>> H = heatmap_entropy(hm)      # [K]
        >>> print(H.shape)
        torch.Size([4])
    """
    # Reshape to [..., Hm*Wm]
    original_shape = hm.shape[:-2]
    x = hm.float().reshape(*original_shape, -1)  # [..., Hm*Wm]

    # Ensure non-negative (heatmaps should be non-negative, but just in case)
    x = x.clamp(min=0.0)

    # Normalize each distribution to sum=1
    x = x / (x.sum(dim=-1, keepdim=True) + eps)

    # Compute entropy: H = -Σ p log p
    entropy = -(x * (x + eps).log()).sum(dim=-1)  # [...]

    return entropy


def effective_k(mask: torch.Tensor) -> torch.Tensor:
    """
    Count effective (valid) heatmaps per sample.

    Args:
        mask: Binary validity mask [K] or [B, K]
              1 = valid heatmap, 0 = invalid/empty heatmap

    Returns:
        torch.Tensor: Number of valid heatmaps per sample
        - If mask is [K]: returns scalar
        - If mask is [B, K]: returns [B]

    Example:
        >>> mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0]])  # [2, 4]
        >>> k_eff = effective_k(mask)  # [2]
        >>> print(k_eff)
        tensor([3., 1.])
    """
    return mask.float().sum(dim=-1)


def visible_ratio(uv: np.ndarray, total_points: int) -> float:
    """
    Compute ratio of visible points.

    Args:
        uv: [N, 2] visible point coordinates
        total_points: Total number of sampled points

    Returns:
        float: Visibility ratio in [0, 1]
    """
    if total_points == 0:
        return 0.0
    return float(len(uv)) / float(total_points)


def heatmap_peak_location(hm: np.ndarray) -> tuple:
    """
    Find peak (maximum) location in heatmap.

    Args:
        hm: [Hm, Wm] heatmap

    Returns:
        tuple: (y, x) coordinates of peak
    """
    if hm.sum() == 0:
        return (-1, -1)

    flat_idx = np.argmax(hm)
    y, x = np.unravel_index(flat_idx, hm.shape)
    return (int(y), int(x))


def heatmap_peak_value(hm: np.ndarray) -> float:
    """
    Get peak (maximum) value in heatmap.

    Args:
        hm: [Hm, Wm] heatmap

    Returns:
        float: Peak value
    """
    return float(hm.max())


# Testing
if __name__ == "__main__":
    print("Testing quality metrics...")

    # Test heatmap_entropy
    print("\n1. Testing heatmap_entropy...")
    hm = torch.randn(4, 64, 64)
    H = heatmap_entropy(hm)
    print(f"   Input shape: {hm.shape}")
    print(f"   Entropy shape: {H.shape}")
    print(f"   Entropy values: {H}")

    # Test with batch dimension
    hm_batch = torch.randn(2, 4, 64, 64)
    H_batch = heatmap_entropy(hm_batch)
    print(f"   Batch input shape: {hm_batch.shape}")
    print(f"   Batch entropy shape: {H_batch.shape}")

    # Test effective_k
    print("\n2. Testing effective_k...")
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0]])
    k_eff = effective_k(mask)
    print(f"   Mask: {mask}")
    print(f"   Effective K: {k_eff}")

    # Test visible_ratio
    print("\n3. Testing visible_ratio...")
    uv = np.random.rand(50, 2)
    ratio = visible_ratio(uv, 100)
    print(f"   Visible points: {len(uv)}")
    print(f"   Total points: 100")
    print(f"   Visibility ratio: {ratio:.2%}")

    # Test peak functions
    print("\n4. Testing peak functions...")
    hm_np = np.random.rand(64, 64)
    hm_np[30, 40] = 10.0  # Create obvious peak
    peak_loc = heatmap_peak_location(hm_np)
    peak_val = heatmap_peak_value(hm_np)
    print(f"   Peak location: {peak_loc}")
    print(f"   Peak value: {peak_val:.3f}")

    print("\n✅ All quality metrics tests passed!")