"""
Real LLM Integration for VLN Pipeline
=====================================

This module provides real Qwen2.5-VL model integration to replace the fake
LLM token generation in the spatial pipeline.

Key Features:
1. Load real Qwen2.5-VL-VGGT model from HuggingFace or local
2. Process spatial features + instruction text + current observation
3. Extract real LLM hidden states for spatial reasoning
4. Generate heatmaps from actual LLM spatial understanding
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from PIL import Image

from ...utils.path_utils import resolve_model_path
from .config import RealLLMConfig

logger = logging.getLogger(__name__)


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

        # Initialize spatial feature projector (created after model loading)
        self.spatial_feature_projector = None
        
        # Cache for hidden state projector (避免每次推理创建新层)
        self._hidden_projector_cache: Dict[int, nn.Linear] = {}

        logger.info("Real LLM integration initialized successfully")

    def _load_model(self):
        """Load the real Qwen2.5-VL model and processor."""
        try:
            if self.config.use_vggt_model:
                # Load the VGGT-integrated model from HuggingFace
                from ..qwen2_5_vl import (
                    Qwen2_5_VL_VGGTForConditionalGeneration,
                    Qwen2_5_VLProcessor,
                )
                logger.info(f"Loading VGGT-integrated model from {self.config.vggt_model_path}")
                self.model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                    self.config.vggt_model_path,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=True,
                    device_map=self.device
                )
                self.processor = Qwen2_5_VLProcessor.from_pretrained(
                    self.config.vggt_model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                model_type = "VGGT-integrated"
            else:
                # Load standard Qwen2.5-VL model from local files
                from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
                logger.info(f"Loading standard Qwen2.5-VL model from {self.config.model_path}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=True,
                    device_map=self.device
                )
                self.processor = Qwen2_5_VLProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                model_type = "standard"

            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded {model_type} Qwen2.5-VL model")

            # Create the spatial projector with correct dimensions
            self.spatial_feature_projector = self._create_spatial_projector()

        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            logger.warning("This will cause LLM integration to fail")
            self.model = None
            self.processor = None

    def _get_llm_embed_dim(self) -> int:
        """Get LLM embedding dimension safely."""
        if self.model is None:
            return 3584  # Default Qwen2.5-VL-7B embedding dim
        
        try:
            if hasattr(self.model, 'config'):
                return self.model.config.hidden_size
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens.embedding_dim
        except AttributeError:
            pass
        
        return 3584  # Default Qwen2.5-VL-7B embedding dim

    def _create_spatial_projector(self) -> nn.Module:
        """Create projector to align spatial features with LLM embedding space."""
        if self.model is None:
            return nn.Identity()

        llm_embed_dim = self._get_llm_embed_dim()
        spatial_dim = 2048  # Spatial features from fusion module

        projector = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, llm_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(llm_embed_dim)
        )

        # Use float32 for compatibility
        target_dtype = torch.float32
        return projector.to(device=self.device, dtype=target_dtype)

    def _get_or_create_hidden_projector(self, input_dim: int, output_dim: int) -> nn.Linear:
        """Get cached projector or create new one (cached to avoid memory leak)."""
        cache_key = (input_dim, output_dim)
        
        if cache_key not in self._hidden_projector_cache:
            projector = nn.Linear(input_dim, output_dim)
            projector = projector.to(device=self.device, dtype=torch.float32)
            self._hidden_projector_cache[cache_key] = projector
            
        return self._hidden_projector_cache[cache_key]

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
            llm_tokens = self._extract_llm_tokens(llm_output, fused_features, return_hidden_states)

            return {
                'llm_tokens': llm_tokens,
                'llm_output': llm_output.get('generated_text', ''),
                'hidden_states': llm_output.get('hidden_states'),
                'attention_weights': llm_output.get('attentions'),
            }

        except Exception as e:
            logger.error(f"Real LLM integration failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_fallback_output(fused_features)

    def _extract_llm_tokens(
        self,
        llm_output: Dict,
        fused_features: torch.Tensor,
        return_hidden_states: bool
    ) -> torch.Tensor:
        """Extract LLM tokens from output."""
        if return_hidden_states and 'hidden_states' in llm_output and llm_output['hidden_states'] is not None:
            all_hidden_states = llm_output['hidden_states']
            if isinstance(all_hidden_states, tuple) and len(all_hidden_states) > 0:
                layer_idx = self.config.hidden_layer_for_heatmap if self.config.hidden_layer_for_heatmap >= 0 else len(all_hidden_states) - 1
                hidden_states = all_hidden_states[layer_idx]

                # Handle nested tuple structure from generation output
                actual_hidden_states = hidden_states
                if isinstance(hidden_states, tuple):
                    if len(hidden_states) > 0:
                        actual_hidden_states = hidden_states[0]
                        if isinstance(actual_hidden_states, tuple) and len(actual_hidden_states) > 0:
                            actual_hidden_states = actual_hidden_states[-1]

                if isinstance(actual_hidden_states, torch.Tensor):
                    return self._process_hidden_states_for_heatmaps(
                        actual_hidden_states, fused_features.shape
                    )

        # Fallback: use spatial feature projection
        if self.spatial_feature_projector is not None:
            return self.spatial_feature_projector(fused_features.to(dtype=torch.float32))
        return fused_features

    def _process_through_llm(
        self,
        instruction_text: str,
        video_frames: torch.Tensor,
        current_observation: torch.Tensor,
        return_hidden_states: bool = True
    ) -> Dict[str, Any]:
        """Process inputs through the actual Qwen2.5-VL model."""
        # Calculate actual total pixels
        _, T, _, H, W = video_frames.shape
        obs_pixels = int(current_observation.shape[-2] * current_observation.shape[-1]) if current_observation is not None else 0
        actual_total_pixels = int(T * H * W + obs_pixels)

        # Override Qwen's hardcoded pixel budget
        try:
            import qwen_vl_utils.vision_process as vision_process
            original_preset = getattr(vision_process, 'VIDEO_TOTAL_PIXELS', 90316800)
            vision_process.VIDEO_TOTAL_PIXELS = actual_total_pixels

            qwen_logger = logging.getLogger('qwen_vl_utils')
            original_level = qwen_logger.level
            qwen_logger.setLevel(logging.WARNING)
        except Exception as e:
            logger.warning(f"Failed to set VIDEO_TOTAL_PIXELS: {e}")

        # Convert tensors to PIL Images
        video_list = []
        for batch_idx in range(video_frames.shape[0]):
            frames = []
            for frame_idx in range(video_frames.shape[1]):
                frame = video_frames[batch_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype('uint8')
                frames.append(Image.fromarray(frame))
            video_list.append(frames)

        current_obs_list = []
        for batch_idx in range(current_observation.shape[0]):
            obs_frame = current_observation[batch_idx].permute(1, 2, 0).cpu().numpy()
            obs_frame = (obs_frame * 255).astype('uint8')
            current_obs_list.append(Image.fromarray(obs_frame))

        # Format messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_list[0],
                        "nframes": video_frames.shape[1],
                        "total_pixels": actual_total_pixels
                    },
                    {
                        "type": "image",
                        "image": current_obs_list[0]
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

        # Process vision info
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Restore qwen_vl_utils logger level
        try:
            qwen_logger.setLevel(original_level)
        except:
            pass

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )

        if self.config.use_vggt_model:
            inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
        inputs = inputs.to(self.device)

        # Generate with hidden states
        with torch.amp.autocast('cuda', enabled=True, dtype=getattr(torch, self.config.torch_dtype)):
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
        batch_size, num_keyframes, num_patches, feature_dim = target_shape

        # Pool hidden states across sequence length
        pooled_hidden = hidden_states.mean(dim=1).to(dtype=torch.float32)

        # Project to target feature dimension using cached projector
        if pooled_hidden.shape[-1] != feature_dim:
            projector = self._get_or_create_hidden_projector(pooled_hidden.shape[-1], feature_dim)
            pooled_hidden = projector(pooled_hidden)

        # Expand to match target shape
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
        fused_features = fused_features.to(dtype=torch.float32)

        if self.spatial_feature_projector is not None:
            projected_features = self.spatial_feature_projector(fused_features)
        else:
            projected_features = fused_features

        return {
            'llm_tokens': projected_features,
            'llm_output': "FALLBACK: LLM processing failed - using spatial projection only",
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

