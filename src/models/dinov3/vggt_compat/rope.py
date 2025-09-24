# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# 2D Rotary Position Embeddings (RoPE) for DINOv3 integration
# Compatible with VGGT's RoPE interface

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.
    
    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        key = (height, width)
        
        if key not in self.position_cache:
            # Create grid coordinates
            y_coords = torch.arange(height, dtype=torch.float32)
            x_coords = torch.arange(width, dtype=torch.float32)
            
            # Create meshgrid and flatten
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            positions = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)  # [H*W, 2]
            
            self.position_cache[key] = positions
        
        positions = self.position_cache[key].to(device)
        # Expand for batch size
        return positions.unsqueeze(0).expand(batch_size, -1, -1)

    def clear_cache(self):
        """Clears the position cache."""
        self.position_cache.clear()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for spatial transformers.
    
    Extends the original RoPE concept to handle 2D spatial positions,
    applying rotary embeddings separately to different dimensions of the
    feature space based on spatial coordinates.
    """

    def __init__(
        self,
        frequency: int = 100,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        temperature: float = 10000.0,
    ):
        """Initialize 2D RoPE.
        
        Args:
            frequency: Base frequency for position encoding
            embed_dim: Embedding dimension (if provided, used for head_dim calculation)
            num_heads: Number of attention heads (if provided, used for head_dim calculation) 
            temperature: Temperature parameter for frequency calculation
        """
        super().__init__()
        self.frequency = frequency
        self.temperature = temperature
        
        # Calculate head dimension if both embed_dim and num_heads are provided
        if embed_dim is not None and num_heads is not None:
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = None
        
        self._cached_freqs = {}

    def _compute_freqs(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute rotation frequencies for the given dimension."""
        cache_key = (dim, device, dtype)
        if cache_key not in self._cached_freqs:
            # Create frequency tensor
            freqs = torch.arange(0, dim, 2, dtype=dtype, device=device)
            freqs = freqs / dim
            freqs = self.frequency ** (-freqs)
            self._cached_freqs[cache_key] = freqs
        return self._cached_freqs[cache_key]

    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor."""
        # Split the last dimension into pairs
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)  
            positions: Position tensor of shape (batch, seq_len, 2) with [y, x] coordinates
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Get device and dtype
        device = q.device
        dtype = q.dtype
        
        # Compute frequencies
        freqs = self._compute_freqs(head_dim // 2, device, dtype)  # Only use half dimensions
        
        # Extract y and x positions
        y_pos = positions[..., 0]  # [batch, seq_len]
        x_pos = positions[..., 1]  # [batch, seq_len]
        
        # Create position encodings for both dimensions
        # We'll use half the head_dim for y and half for x
        half_dim = head_dim // 4  # Quarter for each cos/sin of each dimension
        
        if half_dim > 0:
            # Y dimension encoding
            y_freqs = freqs[:half_dim].unsqueeze(0).unsqueeze(0)  # [1, 1, half_dim]
            y_pos_expanded = y_pos.unsqueeze(-1)  # [batch, seq_len, 1]
            y_angles = y_pos_expanded * y_freqs  # [batch, seq_len, half_dim]
            y_cos = torch.cos(y_angles)
            y_sin = torch.sin(y_angles)
            
            # X dimension encoding  
            x_freqs = freqs[:half_dim].unsqueeze(0).unsqueeze(0)  # [1, 1, half_dim]
            x_pos_expanded = x_pos.unsqueeze(-1)  # [batch, seq_len, 1]
            x_angles = x_pos_expanded * x_freqs  # [batch, seq_len, half_dim]
            x_cos = torch.cos(x_angles)
            x_sin = torch.sin(x_angles)
            
            # Combine cos and sin for both dimensions
            cos_combined = torch.cat([y_cos, x_cos], dim=-1)  # [batch, seq_len, half_dim*2]
            sin_combined = torch.cat([y_sin, x_sin], dim=-1)  # [batch, seq_len, half_dim*2]
            
            # Expand for heads
            cos_combined = cos_combined.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [batch, heads, seq_len, half_dim*2]
            sin_combined = sin_combined.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [batch, heads, seq_len, half_dim*2]
            
            # Only apply to the rotary dimensions
            rotary_dim = half_dim * 2
            
            if rotary_dim < head_dim:
                # Split q and k into rotary and non-rotary parts
                q_rotary, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
                k_rotary, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
                
                # Apply rotation to rotary parts
                q_rotary = self._apply_rotary_pos_emb(q_rotary, cos_combined, sin_combined)
                k_rotary = self._apply_rotary_pos_emb(k_rotary, cos_combined, sin_combined)
                
                # Concatenate back
                q = torch.cat([q_rotary, q_pass], dim=-1)
                k = torch.cat([k_rotary, k_pass], dim=-1)
            else:
                # Apply rotation to entire tensors
                q = self._apply_rotary_pos_emb(q, cos_combined, sin_combined)
                k = self._apply_rotary_pos_emb(k, cos_combined, sin_combined)
        
        return q, k

    def clear_cache(self):
        """Clear the frequency cache."""
        self._cached_freqs.clear()