"""
NextDiT-based action generation module ported from DualVLN (InternNav).

Architecture: NextDiT + Flow Matching + Async Visual Memory (DepthAnythingV2)
"""

from .components import MemoryEncoder, QFormer, SinusoidalPositionalEncoding
from .nextdit_crossattn import NextDiTCrossAttn, NextDiTCrossAttnConfig

__all__ = [
    'MemoryEncoder',
    'NextDiTCrossAttn',
    'NextDiTCrossAttnConfig',
    'QFormer',
    'SinusoidalPositionalEncoding',
]
