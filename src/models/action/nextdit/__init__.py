"""
NextDiT-based action generation module ported from DualVLN (InternNav).

Architecture: NextDiT + Flow Matching + Async Visual Memory (DepthAnythingV2)
"""

from .nextdit_crossattn import NextDiTCrossAttn, NextDiTCrossAttnConfig
from .components import SinusoidalPositionalEncoding, MemoryEncoder, QFormer

__all__ = [
    'NextDiTCrossAttn',
    'NextDiTCrossAttnConfig',
    'SinusoidalPositionalEncoding',
    'MemoryEncoder',
    'QFormer',
]
