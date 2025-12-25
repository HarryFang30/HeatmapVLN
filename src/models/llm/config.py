"""
LLM Configuration
=================

Configuration dataclass for LLM integration.
"""

from dataclasses import dataclass


@dataclass
class RealLLMConfig:
    """Configuration for real LLM integration."""
    model_path: str = "./models/qwen_2.5_vl"  # Local model path (relative to project root)
    use_vggt_model: bool = False  # Whether to use VGGT-integrated model or standard Qwen2.5-VL
    vggt_model_path: str = "Diankun/Spatial-MLLM-subset-sft"  # VGGT-integrated model (HuggingFace)
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"  # Use eager attention (compatible with older GPUs)
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.1
    use_cache: bool = True
    extract_hidden_states: bool = True
    hidden_layer_for_heatmap: int = -1  # Use last layer hidden states

