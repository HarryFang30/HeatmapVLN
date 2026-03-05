"""
Qwen3.5 Integration for VLN Pipeline
=====================================

This module provides Qwen3.5 model integration for the VLN pipeline.

Key Components:
- Qwen3_5Integration: Main integration class for video processing
- Qwen3_5Config: Configuration dataclass
- Sequence Packing: (disabled for Qwen3.5 due to hybrid attention)
"""

from .integration import Qwen3_5Integration, Qwen3_5Config

# Sequence packing utilities (kept for reference, disabled by default)
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
    "Qwen3_5Integration",
    "Qwen3_5Config",
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
