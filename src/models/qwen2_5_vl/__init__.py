"""
Qwen2.5-VL integration for the VLN Pipeline.

Key Components:
- `Qwen2_5VLIntegration`: main integration wrapper
- `Qwen2_5VLConfig`: configuration dataclass
- Sequence packing utilities: kept as legacy helpers
"""

from .integration import Qwen2_5VLIntegration, Qwen2_5VLConfig

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
    "Qwen2_5VLIntegration",
    "Qwen2_5VLConfig",
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
