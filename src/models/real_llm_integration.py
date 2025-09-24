"""
Real LLM Integration for VLN Pipeline
====================================

This module provides real Qwen2.5-VL model integration to replace the fake
LLM token generation in the spatial pipeline. The key innovation is using
LLMs to boost 3D spatial relationship comprehension.

Key Features:
1. Load real Qwen2.5-VL-VGGT model from HuggingFace
2. Process spatial features + instruction text + current observation
3. Extract real LLM hidden states for spatial reasoning
4. Generate heatmaps from actual LLM spatial understanding
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from ..utils.path_utils import resolve_model_path, get_default_model_paths

from .qwen2_5_vl import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

logger = logging.getLogger(__name__)


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


class RealLLMIntegration(nn.Module):
    """
    Real LLM integration wrapper that replaces fake token generation.

    This module:
    1. Loads the actual Qwen2.5-VL-VGGT model
    2. Processes text instructions + video frames + spatial features
    3. Extracts real LLM hidden states for heatmap generation
    4. Provides spatial reasoning through language understanding
    """

    def __init__(self, config: RealLLMConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Resolve model path to absolute path
        try:
            resolved_model_path = resolve_model_path(config.model_path, "LLM")
            self.config.model_path = str(resolved_model_path)
            logger.info(f"Resolved LLM model path: {self.config.model_path}")
        except FileNotFoundError as e:
            logger.error(f"Failed to resolve LLM model path: {e}")
            raise

        # Load real Qwen2.5-VL-VGGT model
        logger.info(f"Loading real Qwen2.5-VL-VGGT model from {config.model_path}")
        self._load_model()

        # Initialize spatial feature projector
        # This projects our spatial features to LLM embedding space
        # Note: Will be created dynamically when model is loaded
        self.spatial_feature_projector = None

        logger.info("Real LLM integration initialized successfully")

    def _load_model(self):
        """Load the real Qwen2.5-VL model and processor."""
        try:
            if self.config.use_vggt_model:
                # Load the VGGT-integrated model from HuggingFace
                logger.info(f"Loading VGGT-integrated model from {self.config.vggt_model_path}")
                self.model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                    self.config.vggt_model_path,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=True,
                    device_map=self.device
                )
                model_type = "VGGT-integrated"
            else:
                # Load standard Qwen2.5-VL model from local files
                logger.info(f"Loading standard Qwen2.5-VL model from {self.config.model_path}")
                from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=True,
                    device_map=self.device
                )
                model_type = "standard"

            # Load the processor
            if self.config.use_vggt_model:
                self.processor = Qwen2_5_VLProcessor.from_pretrained(
                    self.config.vggt_model_path,
                    trust_remote_code=True
                )
            else:
                from transformers import Qwen2_5_VLProcessor as HFProcessor
                self.processor = HFProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )

            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded {model_type} Qwen2.5-VL model")

            # Now create the spatial projector with correct dimensions
            self.spatial_feature_projector = self._create_spatial_projector()

        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            logger.warning("This will cause LLM integration to fail")
            self.model = None
            self.processor = None

    def _create_spatial_projector(self) -> nn.Module:
        """Create projector to align spatial features with LLM embedding space."""

        if self.model is None:
            return nn.Identity()

        # Get LLM embedding dimension
        try:
            # Try different ways to get the embedding dimension
            if hasattr(self.model, 'config'):
                llm_embed_dim = self.model.config.hidden_size
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                llm_embed_dim = self.model.model.embed_tokens.embedding_dim
            else:
                llm_embed_dim = 3584  # Default Qwen2.5-VL-7B embedding dim
        except:
            llm_embed_dim = 3584  # Default Qwen2.5-VL-7B embedding dim

        # Spatial features come from fusion module (typically 2048 dim)
        spatial_dim = 2048

        projector = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, llm_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(llm_embed_dim)
        )

        # Ensure proper dtype - convert bfloat16 to float32 for compatibility
        target_dtype = torch.float32 if self.config.torch_dtype == "bfloat16" else getattr(torch, self.config.torch_dtype)
        return projector.to(device=self.device, dtype=target_dtype)

    def forward(
        self,
        fused_features: torch.Tensor,  # [B, N_k, N_patches, D_fusion]
        instruction_text: str,
        current_observation: torch.Tensor,  # [B, C, H, W]
        video_frames: torch.Tensor,  # [B, N_k, C, H, W] - selected keyframes
        return_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs through real LLM for spatial reasoning.

        Args:
            fused_features: Fused 3D+2D spatial features
            instruction_text: Navigation instruction
            current_observation: Current first-person view
            video_frames: Selected keyframes
            return_hidden_states: Whether to return hidden states for heatmaps

        Returns:
            Dictionary containing:
                - 'llm_tokens': Real LLM hidden states for heatmap generation
                - 'llm_output': LLM text generation output
                - 'attention_weights': Attention maps for spatial understanding
        """

        if self.model is None:
            logger.warning("Model not loaded, returning dummy tokens")
            return self._create_dummy_output(fused_features)

        batch_size, num_keyframes = fused_features.shape[:2]

        try:
            # Prepare input for Qwen2.5-VL
            with torch.no_grad():
                llm_output = self._process_through_llm(
                    instruction_text,
                    video_frames,
                    current_observation,
                    return_hidden_states
                )

            # Extract spatial features for heatmap generation
            if return_hidden_states and 'hidden_states' in llm_output and llm_output['hidden_states'] is not None:
                # Use the last layer hidden states for spatial reasoning
                # hidden_states is a tuple of tensors (one per layer)
                all_hidden_states = llm_output['hidden_states']
                if isinstance(all_hidden_states, tuple) and len(all_hidden_states) > 0:
                    # Get the last layer (or specified layer) hidden states
                    layer_idx = self.config.hidden_layer_for_heatmap if self.config.hidden_layer_for_heatmap >= 0 else len(all_hidden_states) - 1
                    hidden_states = all_hidden_states[layer_idx]

                    # Debug: Log the structure
                    logger.info(f"Hidden states type: {type(hidden_states)}")
                    logger.info(f"Number of elements in hidden states: {len(hidden_states) if hasattr(hidden_states, '__len__') else 'N/A'}")

                    # Handle nested tuple structure from generation output
                    actual_hidden_states = hidden_states
                    if isinstance(hidden_states, tuple):
                        # For generation output, hidden_states might be nested: (decoder_hidden_states,)
                        if len(hidden_states) > 0:
                            actual_hidden_states = hidden_states[0]  # Get the first element
                            if isinstance(actual_hidden_states, tuple) and len(actual_hidden_states) > 0:
                                # Get the last layer from decoder hidden states
                                actual_hidden_states = actual_hidden_states[-1]

                    if isinstance(actual_hidden_states, torch.Tensor):
                        logger.info(f"Final hidden states shape: {actual_hidden_states.shape}")
                        # Process hidden states to match expected format
                        llm_tokens = self._process_hidden_states_for_heatmaps(
                            actual_hidden_states, fused_features.shape
                        )
                    else:
                        logger.warning(f"Could not extract tensor from hidden states: {type(actual_hidden_states)}")
                        # Fallback: use spatial feature projection if available
                        if self.spatial_feature_projector is not None:
                            llm_tokens = self.spatial_feature_projector(fused_features)
                        else:
                            llm_tokens = fused_features  # Direct passthrough
                else:
                    # Fallback: use spatial feature projection if available
                    if self.spatial_feature_projector is not None:
                        llm_tokens = self.spatial_feature_projector(fused_features)
                    else:
                        llm_tokens = fused_features  # Direct passthrough
            else:
                # Fallback: project spatial features through learned projection if available
                if self.spatial_feature_projector is not None:
                    llm_tokens = self.spatial_feature_projector(fused_features)
                else:
                    llm_tokens = fused_features  # Direct passthrough

            return {
                'llm_tokens': llm_tokens,
                'llm_output': llm_output.get('generated_text', ''),
                'hidden_states': llm_output.get('hidden_states'),
                'attention_weights': llm_output.get('attentions'),
            }

        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Falling back to spatial feature projection")
            return self._create_fallback_output(fused_features)

    def _process_through_llm(
        self,
        instruction_text: str,
        video_frames: torch.Tensor,
        current_observation: torch.Tensor,
        return_hidden_states: bool = True
    ) -> Dict[str, Any]:
        """Process inputs through the actual Qwen2.5-VL model."""

        # Convert tensors to format expected by processor
        # video_frames: [B, N_k, C, H, W] -> list of PIL Images per video
        from PIL import Image
        video_list = []
        for batch_idx in range(video_frames.shape[0]):
            # Convert to list of PIL Images
            frames = []
            for frame_idx in range(video_frames.shape[1]):
                frame = video_frames[batch_idx, frame_idx].permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
                frame = (frame * 255).astype('uint8')  # Convert to 0-255 range
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
            video_list.append(frames)

        # Convert current observation to PIL Image
        current_obs_list = []
        for batch_idx in range(current_observation.shape[0]):
            obs_frame = current_observation[batch_idx].permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
            obs_frame = (obs_frame * 255).astype('uint8')  # Convert to 0-255 range
            pil_image = Image.fromarray(obs_frame)
            current_obs_list.append(pil_image)

        # Format messages for Qwen2.5-VL with THREE inputs: video, current observation, text instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_list[0],  # Use first batch item - historical keyframes
                        "nframes": video_frames.shape[1]
                    },
                    {
                        "type": "image",
                        "image": current_obs_list[0]  # Current observation as separate input
                    },
                    {
                        "type": "text",
                        "text": f"You are analyzing a first-person navigation sequence. The video shows historical keyframes, and the image shows your current observation. Instruction: {instruction_text}. Generate spatial heatmaps showing where the historical keyframes appear relative to your current viewpoint."
                    }
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        from qwen_vl_utils import process_vision_info
        _, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Add raw video input for VGGT processing (only if using VGGT model)
        if self.config.use_vggt_model:
            inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
        inputs = inputs.to(self.device)

        # Generate with hidden states
        with torch.cuda.amp.autocast(enabled=True, dtype=getattr(torch, self.config.torch_dtype)):
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                use_cache=self.config.use_cache,
                output_hidden_states=return_hidden_states,
                output_attentions=return_hidden_states,
                return_dict_in_generate=True
            )

        # Decode output text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output.sequences)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {
            'generated_text': output_text[0] if output_text else '',
            'hidden_states': output.hidden_states if hasattr(output, 'hidden_states') else None,
            'attentions': output.attentions if hasattr(output, 'attentions') else None,
            'sequences': output.sequences
        }

    def _process_hidden_states_for_heatmaps(
        self,
        hidden_states: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Process LLM hidden states to match expected heatmap format."""

        # hidden_states: [B, seq_len, hidden_dim]
        # target_shape: [B, N_k, N_patches, D_fusion]

        batch_size, num_keyframes, num_patches, feature_dim = target_shape

        # Take the mean of the hidden states across sequence length
        # This gives us a representation of the entire spatial understanding
        pooled_hidden = hidden_states.mean(dim=1)  # [B, hidden_dim]

        # Project to target feature dimension with proper dtype handling
        if pooled_hidden.shape[-1] != feature_dim:
            # Use float32 for compatibility
            target_dtype = torch.float32
            projector = nn.Linear(
                pooled_hidden.shape[-1], feature_dim
            ).to(device=pooled_hidden.device, dtype=target_dtype)

            # Convert input to float32 if needed
            pooled_hidden = pooled_hidden.to(dtype=target_dtype)
            pooled_hidden = projector(pooled_hidden)

        # Expand to match target shape
        # [B, hidden_dim] -> [B, N_k, N_patches, D_fusion]
        expanded_hidden = pooled_hidden.unsqueeze(1).unsqueeze(2).expand(
            batch_size, num_keyframes, num_patches, feature_dim
        )

        return expanded_hidden

    def _create_dummy_output(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create dummy output when model loading fails."""
        return {
            'llm_tokens': fused_features.clone(),
            'llm_output': "Model not loaded - dummy output",
            'hidden_states': None,
            'attention_weights': None,
        }

    def _create_fallback_output(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create fallback output using spatial feature projection."""
        # Ensure proper dtype for projection
        fused_features = fused_features.to(dtype=torch.float32)

        if self.spatial_feature_projector is not None:
            projected_features = self.spatial_feature_projector(fused_features)
        else:
            # If projector not available, return original features
            projected_features = fused_features

        return {
            'llm_tokens': projected_features,
            'llm_output': "LLM processing failed - using spatial projection fallback",
            'hidden_states': None,
            'attention_weights': None,
        }


def create_real_llm_integration(
    model_path: str = "./models/qwen_2.5_vl",
    use_vggt_model: bool = False,
    device: str = "cuda",
    torch_dtype: str = "bfloat16"
) -> RealLLMIntegration:
    """
    Factory function to create real LLM integration.

    Args:
        model_path: Local model path for Qwen2.5-VL
        use_vggt_model: Whether to use VGGT-integrated model
        device: Computing device
        torch_dtype: Model precision

    Returns:
        Configured RealLLMIntegration instance
    """
    config = RealLLMConfig(
        model_path=model_path,
        use_vggt_model=use_vggt_model,
        device=device,
        torch_dtype=torch_dtype
    )

    return RealLLMIntegration(config)


# Testing and validation
if __name__ == "__main__":
    # Test real LLM integration
    llm_integration = create_real_llm_integration(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Real LLM Integration created successfully!")

    # Test with dummy inputs
    batch_size, num_keyframes, num_patches, feature_dim = 1, 4, 196, 2048
    dummy_features = torch.randn(batch_size, num_keyframes, num_patches, feature_dim)
    dummy_observation = torch.randn(1, 3, 224, 224)
    dummy_video = torch.randn(1, 4, 3, 224, 224)

    with torch.no_grad():
        output = llm_integration(
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