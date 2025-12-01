# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeAwareConnector(nn.Module):
    """
    Time-Aware Multimodal Connector for VLA Navigation Agent.

    Fuses 2D features (DINOv3), 3D features (VGGT), and learnable time embeddings,
    then projects to LLM input space (Qwen2.5-VL).

    Architecture:
        1. Internal learnable time embeddings (nn.Embedding)
        2. Concatenation of [x_2d, x_3d, time_embed]
        3. MLP projection: Linear -> GELU -> Dropout -> Linear

    Args:
        dim_2d: Dimension of DINOv3 features (e.g., 1024)
        dim_3d: Dimension of VGGT features (e.g., 768)
        llm_input_dim: Target output dimension for Qwen2.5-VL (e.g., 3584)
        time_embed_dim: Dimension of time embedding vectors (e.g., 64)
        max_frames: Maximum number of frames (vocabulary size for embeddings)
        hidden_multiplier: Hidden layer size multiplier (default: 2.0)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)

    Input:
        x_2d: [Batch, Nk, dim_2d] - DINOv3 features for Nk keyframes
        x_3d: [Batch, Nk, dim_3d] - VGGT features for Nk keyframes
        frame_indices: [Batch, Nk] - Integer frame indices (e.g., [0, 5, 10, ...])

    Output:
        [Batch, Nk, llm_input_dim] - Projected features for LLM input
    """

    def __init__(
        self,
        dim_2d: int,
        dim_3d: int,
        llm_input_dim: int,
        time_embed_dim: int = 64,
        max_frames: int = 100,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Store dimensions
        self.dim_2d = dim_2d
        self.dim_3d = dim_3d
        self.llm_input_dim = llm_input_dim
        self.time_embed_dim = time_embed_dim
        self.max_frames = max_frames

        # A. Internal Time Embedding (Learnable)
        # Maps frame_indices [0, max_frames-1] to learnable vectors
        self.time_embed = nn.Embedding(
            num_embeddings=max_frames,
            embedding_dim=time_embed_dim
        )

        # B. Dynamic Dimension Calculation
        # Input = concatenation of [x_2d, x_3d, time_embedding]
        actual_input_width = dim_2d + dim_3d + time_embed_dim

        # Hidden layer dimension
        hidden_features = int(actual_input_width * hidden_multiplier)

        # C. MLP Projection Layers
        # Linear -> GELU -> Dropout -> Linear -> Dropout
        self.fc1 = nn.Linear(actual_input_width, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, llm_input_dim, bias=bias)
        self.drop2 = nn.Dropout(dropout)

        # Initialize embeddings with small values
        nn.init.normal_(self.time_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_2d: Tensor,
        x_3d: Tensor,
        frame_indices: Tensor
    ) -> Tensor:
        """
        Forward pass: Fuse multimodal features with time information.

        Args:
            x_2d: [Batch, Nk, dim_2d] - DINOv3 2D semantic features
            x_3d: [Batch, Nk, dim_3d] - VGGT 3D geometric features
            frame_indices: [Batch, Nk] - Integer indices (e.g., [0, 5, 10, ...])

        Returns:
            [Batch, Nk, llm_input_dim] - Projected features for LLM
        """
        # Validate input shapes
        batch_size, num_keyframes, _ = x_2d.shape
        assert x_3d.shape == (batch_size, num_keyframes, self.dim_3d), \
            f"x_3d shape mismatch: expected {(batch_size, num_keyframes, self.dim_3d)}, got {x_3d.shape}"
        assert frame_indices.shape == (batch_size, num_keyframes), \
            f"frame_indices shape mismatch: expected {(batch_size, num_keyframes)}, got {frame_indices.shape}"

        # Step 1: Lookup time embeddings
        # frame_indices: [Batch, Nk] -> time_vec: [Batch, Nk, time_embed_dim]
        time_vec = self.time_embed(frame_indices)

        # Step 2: Concatenate all features along last dimension
        # [Batch, Nk, dim_2d] + [Batch, Nk, dim_3d] + [Batch, Nk, time_embed_dim]
        # -> [Batch, Nk, dim_2d + dim_3d + time_embed_dim]
        fused = torch.cat([x_2d, x_3d, time_vec], dim=-1)

        # Step 3: Project through MLP
        # Linear -> GELU -> Dropout -> Linear -> Dropout
        x = self.fc1(fused)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        # Output: [Batch, Nk, llm_input_dim]
        return x

    def get_time_embedding(self, frame_idx: int) -> Tensor:
        """
        Utility method to retrieve a specific time embedding.

        Args:
            frame_idx: Frame index (0 to max_frames-1)

        Returns:
            [time_embed_dim] - Time embedding vector
        """
        assert 0 <= frame_idx < self.max_frames, \
            f"frame_idx {frame_idx} out of range [0, {self.max_frames})"
        return self.time_embed.weight[frame_idx]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  dim_2d={self.dim_2d},\n"
            f"  dim_3d={self.dim_3d},\n"
            f"  llm_input_dim={self.llm_input_dim},\n"
            f"  time_embed_dim={self.time_embed_dim},\n"
            f"  max_frames={self.max_frames},\n"
            f"  actual_input_width={self.dim_2d + self.dim_3d + self.time_embed_dim}\n"
            f")"
        )
