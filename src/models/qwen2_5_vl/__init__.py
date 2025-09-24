"""
Qwen2.5-VL Integration for VLN Project
====================================

This module provides real Qwen2.5-VL model integration for the VLN pipeline,
replacing the fake LLM token generation with actual LLM processing.

Key Components:
- Qwen2_5_VL_VGGTForConditionalGeneration: VGGT-integrated Qwen2.5-VL model
- Qwen2_5_VLProcessor: Text and video input processor
- Real LLM hidden state extraction for spatial reasoning
"""

from .modeling_qwen2_5_vl import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from .processing_qwen2_5_vl import Qwen2_5_VLProcessor
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig

__all__ = [
    "Qwen2_5_VL_VGGTForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLProcessor",
    "Qwen2_5_VLConfig",
]