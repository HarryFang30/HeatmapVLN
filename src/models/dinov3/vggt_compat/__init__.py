# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# VGGT compatibility layer for DINOv3 integration

from .layers import (
    PatchEmbed,
    Block,
    MemEffAttention,
    Mlp,
    NestedTensorBlock,
    SwiGLUFFN,
    SwiGLUFFNFused,
)
from .rope import (
    RotaryPositionEmbedding2D,
    PositionGetter,
)

__all__ = [
    'PatchEmbed',
    'Block', 
    'MemEffAttention',
    'Mlp',
    'NestedTensorBlock',
    'SwiGLUFFN',
    'SwiGLUFFNFused',
    'RotaryPositionEmbedding2D',
    'PositionGetter',
]