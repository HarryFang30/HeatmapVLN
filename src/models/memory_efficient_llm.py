"""
Memory-Efficient LLM Integration
===============================

This module provides memory-efficient loading of large LLMs like Qwen2.5-VL
that can work alongside other large models (DINOv3) by dynamically loading
and unloading models to manage GPU memory.

Key Features:
1. Sequential model loading (not parallel)
2. Dynamic model unloading after processing
3. Memory optimization with torch.cuda.empty_cache()
4. Minimal memory footprint
"""

import torch
import torch.nn as nn
import logging
import gc
from typing import Dict, Any, Optional
from contextlib import contextmanager

from .real_llm_integration import RealLLMIntegration, RealLLMConfig

logger = logging.getLogger(__name__)


class MemoryEfficientLLM(nn.Module):
    """
    Memory-efficient wrapper for large LLM models.

    This class loads LLM models on-demand and immediately unloads them
    after processing to minimize GPU memory usage.
    """

    def __init__(self, config: RealLLMConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Don't load model immediately - load on demand
        self.model_loaded = False
        self.llm_integration = None

        logger.info("Memory-efficient LLM wrapper initialized (model not loaded)")

    @contextmanager
    def load_model_temporarily(self):
        """Context manager that loads model, yields it, then unloads it."""
        try:
            # Free up memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            logger.info("Loading LLM model temporarily...")
            self.llm_integration = RealLLMIntegration(self.config)
            self.model_loaded = True

            # Log memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"GPU memory after LLM loading: {memory_gb:.2f} GB")

            yield self.llm_integration

        finally:
            # Unload model and free memory
            if self.llm_integration is not None:
                del self.llm_integration.model
                del self.llm_integration.processor
                del self.llm_integration
                self.llm_integration = None
                self.model_loaded = False

                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

                logger.info("LLM model unloaded and memory freed")

                if torch.cuda.is_available():
                    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(f"GPU memory after cleanup: {memory_gb:.2f} GB")

    def forward(
        self,
        fused_features: torch.Tensor,
        instruction_text: str,
        current_observation: torch.Tensor,
        video_frames: torch.Tensor,
        return_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs through LLM with memory-efficient loading.
        """
        try:
            with self.load_model_temporarily() as llm:
                logger.info("Processing through memory-efficient LLM...")
                result = llm(
                    fused_features=fused_features,
                    instruction_text=instruction_text,
                    current_observation=current_observation,
                    video_frames=video_frames,
                    return_hidden_states=return_hidden_states
                )
                return result

        except Exception as e:
            logger.error(f"Memory-efficient LLM processing failed: {e}")

            # Return fallback result
            return {
                'llm_tokens': fused_features.clone(),
                'llm_output': f"Memory-efficient LLM failed: {str(e)}",
                'hidden_states': None,
                'attention_weights': None,
            }


def create_memory_efficient_llm(
    model_path: str = "./models/qwen_2.5_vl",
    use_vggt_model: bool = False,
    device: str = "cuda",
    torch_dtype: str = "bfloat16"
) -> MemoryEfficientLLM:
    """
    Factory function to create memory-efficient LLM integration.

    Args:
        model_path: Local model path for Qwen2.5-VL
        use_vggt_model: Whether to use VGGT-integrated model
        device: Computing device
        torch_dtype: Model precision

    Returns:
        Configured MemoryEfficientLLM instance
    """
    config = RealLLMConfig(
        model_path=model_path,
        use_vggt_model=use_vggt_model,
        device=device,
        torch_dtype=torch_dtype
    )

    return MemoryEfficientLLM(config)


# Testing and validation
if __name__ == "__main__":
    # Test memory-efficient LLM
    memory_llm = create_memory_efficient_llm(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Memory-efficient LLM created successfully!")

    # Test with dummy inputs
    batch_size, num_keyframes, num_patches, feature_dim = 1, 4, 196, 2048
    dummy_features = torch.randn(batch_size, num_keyframes, num_patches, feature_dim)
    dummy_observation = torch.randn(1, 3, 224, 224)
    dummy_video = torch.randn(1, 4, 3, 224, 224)

    with torch.no_grad():
        output = memory_llm(
            fused_features=dummy_features,
            instruction_text="Navigate through this space",
            current_observation=dummy_observation,
            video_frames=dummy_video
        )

    print(f"Output keys: {list(output.keys())}")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")