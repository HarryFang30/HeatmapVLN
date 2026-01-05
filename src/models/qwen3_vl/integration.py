"""
Qwen3-VL Integration Module
===========================

This module provides the Qwen3-VL integration for the VLN pipeline.
It handles video processing and hidden state extraction for downstream heads.

Features:
- Load Qwen3-VL model with flash attention support
- Process video frames + current observation + instruction text
- Extract hidden states for heatmap/action/stop heads
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLConfig:
    """Configuration for Qwen3-VL integration."""
    
    # Model path
    model_path: str = "./models/qwen_3_vl"
    
    # Device and dtype
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Attention implementation (flash_attention_2, sdpa, or eager)
    attn_implementation: str = "sdpa"  # Default to sdpa (works without flash_attn)
    
    # Generation settings (for inference mode)
    max_new_tokens: int = 128
    temperature: float = 0.7
    
    # Hidden state extraction
    hidden_layer_for_features: int = -1  # -1 = last layer
    
    # Video processing
    max_video_frames: int = 16  # Maximum frames to process
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class Qwen3VLIntegration(nn.Module):
    """
    Qwen3-VL Integration for VLN Pipeline.
    
    This class wraps the Qwen3-VL model to:
    1. Process video frames and text instructions
    2. Extract hidden states for downstream heads
    3. Provide a clean interface for the pipeline
    
    Args:
        config: Qwen3VLConfig with model settings
    """
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Model and processor (lazy loading)
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        # Special token IDs (from tokenizer_config.json)
        self.video_token_id = 151656  # <|video_pad|>
        self.image_token_id = 151655  # <|image_pad|>
        self.vision_start_id = 151652  # <|vision_start|>
        self.vision_end_id = 151653  # <|vision_end|>
        
        logger.info(f"Qwen3VLIntegration initialized (model will be loaded on first forward)")
    
    def _load_model(self):
        """Load the Qwen3-VL model and processor."""
        if self._model_loaded:
            return
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading Qwen3-VL from {self.config.model_path}")
            
            # Load model
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.get_torch_dtype(),
                attn_implementation=self.config.attn_implementation,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Freeze all Qwen3-VL parameters (never train the backbone)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )
            
            self._model_loaded = True
            logger.info(f"Qwen3-VL loaded successfully on {self.device}")
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {total_params:,} (all frozen, trainable: {trainable_params})")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            raise
    
    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor frames to PIL Images.
        
        Args:
            tensor: (N, C, H, W) tensor with values in [0, 1]
            
        Returns:
            List of PIL Images
        """
        images = []
        tensor = tensor.cpu()
        
        for i in range(tensor.shape[0]):
            frame = tensor[i]  # (C, H, W)
            # Convert to numpy (H, W, C) and scale to [0, 255]
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame_np))
        
        return images
    
    def _prepare_messages(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """
        Prepare messages for Qwen3-VL chat template.
        
        Args:
            history_frames: (B, K, C, H, W) history video frames
            current_frame: (B, C, H, W) current observation
            instruction: Navigation instruction text
            
        Returns:
            List of message dicts for chat template
        """
        # For batch processing, we only use the first sample
        # (Qwen3-VL processes one conversation at a time)
        batch_idx = 0
        
        # Convert history frames to PIL images
        history_pil = self._tensor_to_pil_images(history_frames[batch_idx])
        
        # Limit number of frames
        if len(history_pil) > self.config.max_video_frames:
            # Uniform sampling
            indices = np.linspace(0, len(history_pil) - 1, self.config.max_video_frames, dtype=int)
            history_pil = [history_pil[i] for i in indices]
        
        # Convert current frame to PIL
        current_pil = self._tensor_to_pil_images(current_frame[batch_idx:batch_idx+1])[0]
        
        # Build instruction text
        if instruction is None or instruction == "":
            instruction = "Analyze the spatial relationships in this navigation sequence."
        
        prompt_text = (
            f"You are a navigation assistant. "
            f"The video shows the historical trajectory, and the image shows your current view. "
            f"Instruction: {instruction}. "
            f"Understand the spatial layout and identify where you came from."
        )
        
        # Build message content
        content = [
            {"type": "video", "video": history_pil},
            {"type": "image", "image": current_pil},
            {"type": "text", "text": prompt_text},
        ]
        
        messages = [{"role": "user", "content": content}]
        
        return messages, history_pil, current_pil
    
    def forward(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
        return_hidden_states: bool = True,
        generate_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through Qwen3-VL.
        
        Args:
            history_frames: (B, K, C, H, W) history video frames
            current_frame: (B, C, H, W) current observation
            instruction: Navigation instruction text
            return_hidden_states: Whether to return hidden states
            generate_text: Whether to generate text output
            
        Returns:
            Dict containing:
                - hidden_states: (B, seq_len, hidden_dim) LLM hidden states
                - vision_hidden_states: (B, num_vision_tokens, hidden_dim) vision token hidden states
                - generated_text: Generated text (if generate_text=True)
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        batch_size = history_frames.shape[0]
        
        # Prepare messages
        messages, history_pil, current_pil = self._prepare_messages(
            history_frames, current_frame, instruction
        )
        
        # Apply chat template and get inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store input_ids for vision token extraction
        input_ids = inputs["input_ids"]
        
        # Forward pass
        with torch.no_grad():
            if generate_text:
                # Generation mode
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    output_hidden_states=return_hidden_states,
                    return_dict_in_generate=True,
                )
                
                # Decode generated text
                generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                
                # Get hidden states from generation
                if return_hidden_states and hasattr(outputs, 'hidden_states'):
                    # outputs.hidden_states is a tuple of tuples
                    # Each inner tuple has hidden states for each layer
                    # We take the last generation step, last layer
                    hidden_states = self._extract_hidden_from_generation(outputs.hidden_states)
                else:
                    hidden_states = None
            else:
                # Encoder-only mode (for training)
                outputs = self.model(
                    **inputs,
                    output_hidden_states=return_hidden_states,
                    return_dict=True,
                )
                
                generated_text = None
                
                if return_hidden_states:
                    # Get hidden states from the specified layer
                    layer_idx = self.config.hidden_layer_for_features
                    if layer_idx == -1:
                        layer_idx = len(outputs.hidden_states) - 1
                    hidden_states = outputs.hidden_states[layer_idx]
                else:
                    hidden_states = None
        
        # Extract vision token hidden states
        vision_hidden_states = None
        if hidden_states is not None:
            vision_hidden_states = self._extract_vision_hidden_states(
                hidden_states, input_ids
            )
        
        # Expand to batch size if needed
        if hidden_states is not None and hidden_states.shape[0] == 1 and batch_size > 1:
            hidden_states = hidden_states.expand(batch_size, -1, -1)
        if vision_hidden_states is not None and vision_hidden_states.shape[0] == 1 and batch_size > 1:
            vision_hidden_states = vision_hidden_states.expand(batch_size, -1, -1)
        
        return {
            "hidden_states": hidden_states,
            "vision_hidden_states": vision_hidden_states,
            "generated_text": generated_text,
            "input_ids": input_ids,
        }
    
    def _extract_hidden_from_generation(
        self,
        hidden_states_tuple: Tuple,
    ) -> torch.Tensor:
        """
        Extract hidden states from generation output.
        
        Generation output has structure:
        hidden_states_tuple[step][layer] = (batch, seq, hidden)
        
        We concatenate all steps and take the last layer.
        """
        all_hidden = []
        layer_idx = self.config.hidden_layer_for_features
        
        for step_hidden in hidden_states_tuple:
            if isinstance(step_hidden, tuple) and len(step_hidden) > 0:
                if layer_idx == -1:
                    layer_idx = len(step_hidden) - 1
                if layer_idx < len(step_hidden):
                    all_hidden.append(step_hidden[layer_idx])
        
        if all_hidden:
            return torch.cat(all_hidden, dim=1)
        return None
    
    def _extract_vision_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states corresponding to vision tokens.
        
        Args:
            hidden_states: (B, seq_len, hidden_dim)
            input_ids: (B, seq_len)
            
        Returns:
            (B, num_vision_tokens, hidden_dim)
        """
        batch_size = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        
        # Find vision token positions (video_pad and image_pad tokens)
        video_mask = input_ids == self.video_token_id
        image_mask = input_ids == self.image_token_id
        vision_mask = video_mask | image_mask
        
        # Get number of vision tokens per sample
        num_vision_tokens = vision_mask.sum(dim=1)
        max_vision_tokens = num_vision_tokens.max().item()
        
        if max_vision_tokens == 0:
            # No vision tokens found, return pooled hidden states
            logger.warning("No vision tokens found, returning mean-pooled hidden states")
            return hidden_states.mean(dim=1, keepdim=True)
        
        # Extract vision hidden states
        vision_hidden = torch.zeros(
            batch_size, max_vision_tokens, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        for b in range(batch_size):
            mask = vision_mask[b]
            vision_indices = mask.nonzero(as_tuple=True)[0]
            n_tokens = len(vision_indices)
            if n_tokens > 0:
                vision_hidden[b, :n_tokens] = hidden_states[b, vision_indices]
        
        return vision_hidden
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        if self._model_loaded and self.model is not None:
            return self.model.config.hidden_size
        # Default for Qwen3-VL-2B
        return 2048
    
    def freeze(self):
        """Freeze all model parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all model parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = True

