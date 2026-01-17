"""
Qwen3-VL Integration for VLN Pipeline
======================================

This module provides Qwen3-VL model integration for the simplified VLN pipeline.
It replaces VGGT and DINOv3 with Qwen3-VL's native vision encoder.

Key Components:
- Qwen3VLIntegration: Main integration class for video processing
- Qwen3VLConfig: Configuration dataclass
- Sequence Packing: 基于官方 fine-tuning 框架的高效批量训练支持
"""

from .integration import Qwen3VLIntegration, Qwen3VLConfig

# Sequence packing utilities
try:
    from .sequence_packing import (
        FlattenedDataCollatorForVLN,
        split_packed_hidden_states,
        split_packed_vision_hidden_states,
        replace_attention_with_varlen,
        get_rope_index_3,
        PackedSequenceProcessor,
    )
    PACKING_AVAILABLE = True
except ImportError:
    PACKING_AVAILABLE = False

__all__ = [
    "Qwen3VLIntegration",
    "Qwen3VLConfig",
    "PACKING_AVAILABLE",
]

if PACKING_AVAILABLE:
    __all__.extend([
        "FlattenedDataCollatorForVLN",
        "split_packed_hidden_states",
        "split_packed_vision_hidden_states",
        "replace_attention_with_varlen",
        "get_rope_index_3",
        "PackedSequenceProcessor",
    ])

