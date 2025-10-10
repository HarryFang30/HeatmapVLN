"""
Loss functions for heatmap training.

Implements:
- heatmap_ce_from_logits: Training loss in logits space (RECOMMENDED)
- kl_ce_loss: Original KL-CE loss in probability space
- focal_kl_ce_loss: Focal loss variant
- mse_heatmap_loss: MSE alternative
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def heatmap_ce_from_logits(logits: torch.Tensor, target_maps: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           tau: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Cross-Entropy loss in logits space (RECOMMENDED for training).

    This avoids double softmax and preserves gradient signal better than
    computing loss in probability space.

    **NEW**: Supports dynamic K - handles cases where logits.shape[1] != target_maps.shape[1]
    by taking min(K_pred, K_target) to handle variable-length reference paths.

    Args:
        logits: Raw logits from model [B, K_pred, Hm, Wm]
        target_maps: Target heatmaps [B, K_target, Hm, Wm] (raw values, will be normalized)
        mask: Optional validity mask [B, K_target] where 1=valid, 0=invalid
        tau: Temperature parameter (default 1.0)
        eps: Numerical stability epsilon

    Returns:
        torch.Tensor: Scalar loss value
    """
    B_pred, K_pred, Hm_pred, Wm_pred = logits.shape
    B_target, K_target, Hm_target, Wm_target = target_maps.shape

    # Handle K mismatch (variable reference path lengths)
    if K_pred != K_target:
        K_min = min(K_pred, K_target)
        logits = logits[:, :K_min].contiguous()  # Truncate to min K and make contiguous
        target_maps = target_maps[:, :K_min].contiguous()
        if mask is not None:
            mask = mask[:, :K_min].contiguous()
        K = K_min
    else:
        K = K_pred

    # Flatten spatial dimensions (use reshape for safety after slicing)
    logits_flat = logits.reshape(B_pred * K, -1) / tau  # [B*K, Hm*Wm]
    target_flat = target_maps.reshape(B_target * K, -1)   # [B*K, Hm*Wm]

    # Compute log probabilities (stable)
    log_p = F.log_softmax(logits_flat, dim=-1)  # [B*K, Hm*Wm]

    # Normalize targets (don't apply softmax, just normalize)
    q = target_flat / (target_flat.sum(dim=-1, keepdim=True) + eps)  # [B*K, Hm*Wm]

    # Cross-entropy: -sum(q * log(p))
    loss = -(q * log_p).sum(dim=-1)  # [B*K]

    # Apply mask
    if mask is not None:
        m = mask.view(B_target * K).float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)
    else:
        return loss.mean()


def kl_ce_loss(pred_probs: torch.Tensor, target_maps: torch.Tensor,
               mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    KL-CE loss for heatmap training as specified in train.md.

    This is the main loss function that:
    1. Converts target_maps to probability distributions via softmax
    2. Computes KL divergence (equivalent to cross-entropy)
    3. Applies mask to skip invalid channels
    4. Returns mean loss

    Args:
        pred_probs: Predicted probability distributions [B, K, Hm, Wm]
        target_maps: Target heatmaps [B, K, Hm, Wm] (raw values, will be softmax normalized)
        mask: Optional mask [B, K] where 1=valid, 0=invalid. If None, all are valid.
        eps: Small epsilon for numerical stability

    Returns:
        torch.Tensor: Scalar loss value
    """
    B, K, Hm, Wm = pred_probs.shape

    # Ensure target_maps has same shape
    assert target_maps.shape == pred_probs.shape, f"Shape mismatch: pred {pred_probs.shape} vs target {target_maps.shape}"

    # Convert target maps to probability distributions
    # Flatten spatial dimensions for softmax
    target_flat = target_maps.view(B * K, -1)  # [B*K, Hm*Wm]
    q = torch.softmax(target_flat, dim=1)  # Normalize each heatmap to sum=1

    # Flatten predicted probabilities
    p = pred_probs.view(B * K, -1).clamp_min(eps)  # [B*K, Hm*Wm]

    # Compute KL divergence: -∑(q * log(p))
    # This is equivalent to cross-entropy between q and p
    loss = -(q * torch.log(p)).sum(dim=-1)  # [B*K]

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.view(B * K).float()  # [B*K]
        # Only compute loss for valid heatmaps (mask=1)
        masked_loss = loss * mask_flat
        # Return mean over valid samples
        return masked_loss.sum() / mask_flat.sum().clamp_min(1.0)
    else:
        # No mask, return mean over all samples
        return loss.mean()


def focal_kl_ce_loss(pred_probs: torch.Tensor, target_maps: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Focal loss variant of KL-CE loss for handling hard examples.

    Args:
        pred_probs: Predicted probability distributions [B, K, Hm, Wm]
        target_maps: Target heatmaps [B, K, Hm, Wm]
        mask: Optional mask [B, K]
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter (focusing parameter)
        eps: Numerical stability epsilon

    Returns:
        torch.Tensor: Scalar focal loss value
    """
    B, K, Hm, Wm = pred_probs.shape

    # Convert target maps to probability distributions
    target_flat = target_maps.view(B * K, -1)
    q = torch.softmax(target_flat, dim=1)

    # Flatten predicted probabilities
    p = pred_probs.view(B * K, -1).clamp_min(eps)

    # Compute base CE loss
    ce_loss = -(q * torch.log(p)).sum(dim=-1)  # [B*K]

    # Compute confidence (max probability for each heatmap)
    p_max = p.max(dim=-1)[0]  # [B*K]

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - p_max) ** gamma

    # Apply focal loss
    focal_loss = alpha * focal_weight * ce_loss

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.view(B * K).float()
        masked_loss = focal_loss * mask_flat
        return masked_loss.sum() / mask_flat.sum().clamp_min(1.0)
    else:
        return focal_loss.mean()


def mse_heatmap_loss(pred_probs: torch.Tensor, target_maps: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    MSE loss alternative for heatmap training.

    Args:
        pred_probs: Predicted probability distributions [B, K, Hm, Wm]
        target_maps: Target heatmaps [B, K, Hm, Wm]
        mask: Optional mask [B, K]

    Returns:
        torch.Tensor: Scalar MSE loss value
    """
    B, K, Hm, Wm = pred_probs.shape

    # Convert target maps to probability distributions
    target_flat = target_maps.view(B * K, -1)
    q = torch.softmax(target_flat, dim=1).view(B, K, Hm, Wm)

    # MSE loss
    mse_loss = F.mse_loss(pred_probs, q, reduction='none').mean(dim=(-1, -2))  # [B, K]

    # Apply mask if provided
    if mask is not None:
        masked_loss = mse_loss * mask
        return masked_loss.sum() / mask.sum().clamp_min(1.0)
    else:
        return mse_loss.mean()


class HeatmapLossConfig:
    """Configuration class for heatmap losses"""

    def __init__(self, loss_type: str = 'kl_ce', focal_enabled: bool = False,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        self.loss_type = loss_type
        self.focal_enabled = focal_enabled
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma


def create_heatmap_loss_fn(config: HeatmapLossConfig):
    """
    Factory function to create loss function based on configuration.

    Args:
        config: HeatmapLossConfig instance

    Returns:
        Callable loss function
    """
    if config.loss_type == 'kl_ce':
        if config.focal_enabled:
            def loss_fn(pred_probs, target_maps, mask=None):
                return focal_kl_ce_loss(
                    pred_probs, target_maps, mask,
                    alpha=config.focal_alpha, gamma=config.focal_gamma
                )
        else:
            loss_fn = kl_ce_loss
    elif config.loss_type == 'mse':
        loss_fn = mse_heatmap_loss
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")

    return loss_fn


# Test and example usage
if __name__ == "__main__":
    # Test loss functions
    batch_size = 2
    k_heatmaps = 4
    hm_size = (32, 32)

    # Create test data
    pred_probs = torch.softmax(torch.randn(batch_size, k_heatmaps, *hm_size), dim=(-1, -2))
    target_maps = torch.randn(batch_size, k_heatmaps, *hm_size)  # Raw values
    mask = torch.ones(batch_size, k_heatmaps)  # All valid
    mask[0, -1] = 0  # Make last heatmap of first batch invalid

    print(f"Test data shapes:")
    print(f"  pred_probs: {pred_probs.shape}")
    print(f"  target_maps: {target_maps.shape}")
    print(f"  mask: {mask.shape}")

    # Check that pred_probs are normalized
    pred_sums = pred_probs.sum(dim=(-1, -2))
    print(f"\nPredicted probs normalization check:")
    print(f"  Min sum: {pred_sums.min().item():.6f}")
    print(f"  Max sum: {pred_sums.max().item():.6f}")

    # Test KL-CE loss
    loss_kl_ce = kl_ce_loss(pred_probs, target_maps, mask)
    print(f"\nKL-CE loss: {loss_kl_ce.item():.6f}")

    # Test without mask
    loss_no_mask = kl_ce_loss(pred_probs, target_maps)
    print(f"KL-CE loss (no mask): {loss_no_mask.item():.6f}")

    # Test Focal loss
    focal_loss = focal_kl_ce_loss(pred_probs, target_maps, mask)
    print(f"Focal KL-CE loss: {focal_loss.item():.6f}")

    # Test MSE loss
    mse_loss = mse_heatmap_loss(pred_probs, target_maps, mask)
    print(f"MSE loss: {mse_loss.item():.6f}")

    # Test loss factory
    print(f"\nTesting loss factory:")
    config = HeatmapLossConfig(loss_type='kl_ce', focal_enabled=False)
    loss_fn = create_heatmap_loss_fn(config)
    factory_loss = loss_fn(pred_probs, target_maps, mask)
    print(f"Factory KL-CE loss: {factory_loss.item():.6f}")

    config_focal = HeatmapLossConfig(loss_type='kl_ce', focal_enabled=True, focal_gamma=2.0)
    focal_loss_fn = create_heatmap_loss_fn(config_focal)
    factory_focal_loss = focal_loss_fn(pred_probs, target_maps, mask)
    print(f"Factory Focal loss: {factory_focal_loss.item():.6f}")

    print("\nHeatmap loss functions test completed successfully!")