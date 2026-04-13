"""
Qwen2.5-VL integration for the VLN Pipeline.

Key Components:
- `Qwen2_5VLIntegration`: main integration wrapper
- `Qwen2_5VLConfig`: configuration dataclass
- Sequence packing utilities: kept as legacy helpers
"""

from .integration import Qwen2_5VLConfig, Qwen2_5VLIntegration

# Sequence packing utilities (kept for reference, disabled by default)
try:
    from .sequence_packing import (
        FlattenedDataCollatorForVLN,
        PackedSequenceProcessor,
        get_rope_index_3,
        replace_attention_with_varlen,
        split_packed_hidden_states,
        split_packed_vision_hidden_states,
    )
    PACKING_AVAILABLE = True
except ImportError:
    PACKING_AVAILABLE = False

__all__ = [
    "PACKING_AVAILABLE",
    "Qwen2_5VLConfig",
    "Qwen2_5VLIntegration",
]

if PACKING_AVAILABLE:
    __all__.extend([
        "FlattenedDataCollatorForVLN",
        "PackedSequenceProcessor",
        "get_rope_index_3",
        "replace_attention_with_varlen",
        "split_packed_hidden_states",
        "split_packed_vision_hidden_states",
    ])
