# Spatial-MLLM LLM Module
# Copyright 2025 - Integration for Project

from .configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from .modeling_qwen2_5_vl import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)

from .processing_qwen2_5_vl import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLProcessorKwargs,
)

# Original modular version (base Qwen2.5-VL without spatial capabilities)
from .modular_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as BaseQwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor as BaseQwen2_5_VLProcessor,
)

__all__ = [
    # Main Spatial-MLLM classes
    "Qwen2_5_VL_VGGTForConditionalGeneration",  # Main Spatial-MLLM model
    "Qwen2_5_VLProcessor",                      # Spatial-aware processor
    
    # Base Qwen2.5-VL classes
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel", 
    "Qwen2_5_VLPreTrainedModel",
    
    # Configuration classes
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    
    # Processing classes
    "Qwen2_5_VLProcessorKwargs",
    
    # Original base classes (renamed)
    "BaseQwen2_5_VLForConditionalGeneration",
    "BaseQwen2_5_VLProcessor",
]