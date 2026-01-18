"""
Qwen3-VL Integration Module
===========================

This module provides the Qwen3-VL integration for the VLN pipeline.
It handles video processing and hidden state extraction for downstream heads.

Features:
- Load Qwen3-VL model with flash attention support
- Process video frames + current observation + instruction text
- Extract hidden states for heatmap/action/stop heads
- Sequence Packing support for efficient batch training (based on official implementation)

Sequence Packing (New):
- 基于 Qwen3-VL 官方 fine-tuning 框架实现
- 使用 FlattenedDataCollator 将多个样本拼接成一个长序列
- 使用 flash_attn_varlen_func 处理变长序列
- 显著提高显存利用率，减少 padding 浪费
"""

import warnings
# Suppress Qwen3-VL internal warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*fps.*frames per second.*")
warnings.filterwarnings("ignore", message=".*video_metadata.*")

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Import sequence packing utilities
try:
    from .sequence_packing import (
        FlattenedDataCollatorForVLN,
        split_packed_hidden_states,
        split_packed_vision_hidden_states,
        replace_attention_with_varlen,
        get_rope_index_3,
        IMAGE_TOKEN_ID,
        VIDEO_TOKEN_ID,
    )
    PACKING_AVAILABLE = True
except ImportError:
    PACKING_AVAILABLE = False
    logger.warning("Sequence packing module not available")


@dataclass
class Qwen3VLConfig:
    """Configuration for Qwen3-VL integration."""
    
    # Model path
    model_path: str = "./models/qwen_3_vl"
    
    # Device and dtype
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Attention implementation (flash_attention_2, sdpa, or eager)
    # 推荐使用 flash_attention_2 以获得最佳性能
    attn_implementation: str = "flash_attention_2"
    
    # Generation settings (for inference mode)
    max_new_tokens: int = 128
    temperature: float = 0.7
    
    # Hidden state extraction
    hidden_layer_for_features: int = -1  # -1 = last layer
    
    # Video processing
    max_video_frames: int = 16  # Maximum frames to process
    
    # Sequence Packing settings (based on official Qwen3-VL fine-tuning)
    enable_packing: bool = False  # Whether to use sequence packing
    max_seq_length: int = 4096    # Maximum packed sequence length
    spatial_merge_size: int = 2   # Vision spatial merge size for position IDs
    
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
        
        # Sequence packing state
        self._packing_enabled = config.enable_packing
        self._varlen_attention_replaced = False
        
        logger.info(f"Qwen3VLIntegration initialized (model will be loaded on first forward)")
        if self._packing_enabled:
            logger.info(f"Sequence packing enabled (max_seq_length={config.max_seq_length})")
    
    def _load_model(self):
        """Load the Qwen3-VL model and processor."""
        if self._model_loaded:
            return
        
        try:
            # 如果启用 packing，需要在导入模型前替换 attention
            # 这是官方实现的做法，确保 varlen attention 正确生效
            if self._packing_enabled and not self._varlen_attention_replaced:
                if PACKING_AVAILABLE:
                    try:
                        from flash_attn.flash_attn_interface import flash_attn_varlen_func
                        replace_attention_with_varlen(None)  # 替换类方法，不需要模型实例
                        self._varlen_attention_replaced = True
                        logger.info("Pre-replaced attention with varlen version before model loading")
                    except ImportError:
                        logger.warning("flash_attn not available, skipping varlen attention replacement")
            
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
            
            # 设置 padding_side 为 left，用于批量处理
            self.processor.tokenizer.padding_side = 'left'
            
            logger.info(f"Qwen3-VL loaded successfully on {self.device}")
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {total_params:,} (all frozen, trainable: {trainable_params})")
            logger.info(f"Batch processing enabled (padding_side='left')")
            
            # 注意：varlen attention 已经在模型导入前替换过了
            # 不需要再次调用 enable_sequence_packing()
            if self._packing_enabled and self._varlen_attention_replaced:
                logger.info("Sequence packing with varlen attention is active")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            raise
    
    def enable_sequence_packing(self) -> bool:
        """
        启用 Sequence Packing 模式
        
        基于 Qwen3-VL 官方 fine-tuning 框架：
        1. 替换 attention forward 函数以支持 flash_attn_varlen_func
        2. 使用 cumulative sequence lengths 作为 attention mask
        
        Returns:
            bool: 是否成功启用
        """
        if not PACKING_AVAILABLE:
            logger.warning("Sequence packing not available (missing dependencies)")
            return False
        
        if self._varlen_attention_replaced:
            logger.info("Varlen attention already enabled")
            return True
        
        if not self._model_loaded:
            logger.warning("Model not loaded yet, packing will be enabled after loading")
            self._packing_enabled = True
            return True
        
        # Check if flash_attn is available
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
            has_flash_attn = True
        except ImportError:
            has_flash_attn = False
            logger.warning(
                "flash_attn not installed. Sequence packing will use standard attention. "
                "For best performance, install with: pip install flash-attn --no-build-isolation"
            )
        
        if has_flash_attn:
            # Replace attention with varlen version
            replace_attention_with_varlen(self.model)
            self._varlen_attention_replaced = True
            logger.info("Sequence packing enabled with flash_attn_varlen_func")
        else:
            # Fallback: still enable packing but without varlen attention
            self._packing_enabled = True
            logger.info("Sequence packing enabled (without varlen attention)")
        
        return True
    
    def forward_packed(
        self,
        packed_batch: Dict[str, Any],
        return_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """
        处理 packed batch 的 forward pass
        
        这是 Sequence Packing 模式的核心方法，处理由 FlattenedDataCollatorForVLN 
        生成的 packed batch。
        
        Args:
            packed_batch: 由 FlattenedDataCollatorForVLN 生成的 packed batch，包含：
                - input_ids: (1, total_seq_len)
                - attention_mask: cumsum_seq_lens, shape (num_samples + 1,)
                - position_ids: (3, 1, total_seq_len)
                - pixel_values: optional
                - image_grid_thw: optional
                - pixel_values_videos: optional
                - video_grid_thw: optional
                - seq_lens: List[int], 每个样本的序列长度
                - num_samples: int
            return_hidden_states: 是否返回 hidden states
        
        Returns:
            Dict containing:
                - hidden_states: (num_samples, hidden_dim) 每个样本的表示
                - vision_hidden_states: (num_samples, max_vision_tokens, hidden_dim)
                - seq_lens: List[int], 每个样本的序列长度
        """
        if not self._model_loaded:
            self._load_model()
        
        # 准备 inputs
        input_ids = packed_batch["input_ids"].to(self.device)
        attention_mask = packed_batch["attention_mask"].to(self.device)
        position_ids = packed_batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        
        seq_lens = packed_batch["seq_lens"]
        num_samples = packed_batch["num_samples"]
        
        # 处理视觉数据
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
        
        if "pixel_values" in packed_batch and packed_batch["pixel_values"] is not None:
            model_inputs["pixel_values"] = packed_batch["pixel_values"].to(self.device)
        if "image_grid_thw" in packed_batch and packed_batch["image_grid_thw"] is not None:
            model_inputs["image_grid_thw"] = packed_batch["image_grid_thw"].to(self.device)
        if "pixel_values_videos" in packed_batch and packed_batch["pixel_values_videos"] is not None:
            model_inputs["pixel_values_videos"] = packed_batch["pixel_values_videos"].to(self.device)
        if "video_grid_thw" in packed_batch and packed_batch["video_grid_thw"] is not None:
            model_inputs["video_grid_thw"] = packed_batch["video_grid_thw"].to(self.device)
        
        # Forward pass
        # 注意：不能使用 torch.no_grad()！
        # 虽然 Qwen3-VL 参数被冻结 (requires_grad=False)，但需要保留计算图
        # 以便梯度可以回传到下游的 llm_projector 和 heads
        outputs = self.model(
            **model_inputs,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        
        if return_hidden_states:
            layer_idx = self.config.hidden_layer_for_features
            if layer_idx == -1:
                layer_idx = len(outputs.hidden_states) - 1
            packed_hidden_states = outputs.hidden_states[layer_idx]  # (1, total_seq_len, hidden_dim)
        else:
            packed_hidden_states = None
        
        # 拆分 packed hidden states
        if packed_hidden_states is not None and PACKING_AVAILABLE:
            # 提取每个样本的表示（使用 last token pooling）
            sample_hidden_states = split_packed_hidden_states(
                packed_hidden_states, seq_lens, pool_method="last"
            )
            
            # 提取视觉 token hidden states
            vision_hidden_states = split_packed_vision_hidden_states(
                packed_hidden_states, input_ids, seq_lens
            )
        else:
            sample_hidden_states = packed_hidden_states
            vision_hidden_states = None
        
        return {
            "hidden_states": sample_hidden_states,  # (num_samples, hidden_dim)
            "vision_hidden_states": vision_hidden_states,  # (num_samples, max_vision_tokens, hidden_dim)
            "packed_hidden_states": packed_hidden_states,  # (1, total_seq_len, hidden_dim) 原始输出
            "seq_lens": seq_lens,
            "num_samples": num_samples,
        }
    
    def get_data_collator(self) -> "FlattenedDataCollatorForVLN":
        """
        获取用于 Sequence Packing 的 Data Collator
        
        Returns:
            FlattenedDataCollatorForVLN instance
        """
        if not self._model_loaded:
            self._load_model()
        
        if not PACKING_AVAILABLE:
            raise ImportError("Sequence packing module not available")
        
        return FlattenedDataCollatorForVLN(tokenizer=self.processor.tokenizer)
    
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
    
    def _prepare_messages_single(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Image.Image], Image.Image]:
        """
        Prepare messages for a single sample.
        
        Args:
            history_frames: (K, C, H, W) history video frames for ONE sample
            current_frame: (C, H, W) current observation for ONE sample
            instruction: Navigation instruction text
            
        Returns:
            Tuple of (messages, history_pil, current_pil)
        """
        # Convert history frames to PIL images
        history_pil = self._tensor_to_pil_images(history_frames)
        
        # Limit number of frames (max_video_frames == -1 means no limit)
        if self.config.max_video_frames > 0 and len(history_pil) > self.config.max_video_frames:
            # Uniform sampling
            indices = np.linspace(0, len(history_pil) - 1, self.config.max_video_frames, dtype=int)
            history_pil = [history_pil[i] for i in indices]
        
        # Convert current frame to PIL
        current_pil = self._tensor_to_pil_images(current_frame.unsqueeze(0))[0]
        
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
    
    def _prepare_conversations_batch(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
    ) -> List[List[Dict]]:
        """
        Prepare conversations for batch processing.
        
        Args:
            history_frames: (B, K, C, H, W) history video frames
            current_frame: (B, C, H, W) current observation
            instruction: Navigation instruction (str for all, or List[str] per sample)
            
        Returns:
            List of conversations, each conversation is a list of messages
        """
        batch_size = history_frames.shape[0]
        conversations = []
        
        for b in range(batch_size):
            # Get instruction for this sample
            if instruction is None:
                sample_instruction = None
            elif isinstance(instruction, list):
                sample_instruction = instruction[b] if b < len(instruction) else instruction[0]
            else:
                sample_instruction = instruction
            
            # Prepare single sample messages
            messages, _, _ = self._prepare_messages_single(
                history_frames[b],  # (K, C, H, W)
                current_frame[b],   # (C, H, W)
                sample_instruction,
            )
            conversations.append(messages)
        
        return conversations
    
    def _forward_batch(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Batch forward pass through Qwen3-VL.
        
        This processes all samples in a single forward pass for efficiency.
        
        Args:
            history_frames: (B, K, C, H, W) history video frames
            current_frame: (B, C, H, W) current observation
            instruction: Navigation instruction text
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Tuple of (hidden_states, vision_hidden_states)
        """
        batch_size = history_frames.shape[0]
        
        # Prepare conversations for all samples
        conversations = self._prepare_conversations_batch(
            history_frames, current_frame, instruction
        )
        
        # Apply chat template with padding for batch processing
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # 关键：启用 padding 用于批量处理
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]  # (B, seq_len)
        
        # Batch forward pass
        # 注意：不能使用 torch.no_grad()！
        # 虽然 Qwen3-VL 参数被冻结，但需要保留计算图以便梯度回传到下游模块
        outputs = self.model(
            **inputs,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        
        if return_hidden_states:
            layer_idx = self.config.hidden_layer_for_features
            if layer_idx == -1:
                layer_idx = len(outputs.hidden_states) - 1
            hidden_states = outputs.hidden_states[layer_idx]  # (B, seq_len, hidden_dim)
        else:
            hidden_states = None
        
        # Extract vision token hidden states for each sample
        vision_hidden_states = None
        if hidden_states is not None:
            vision_hidden_states = self._extract_vision_hidden_states(
                hidden_states, input_ids
            )
        
        return hidden_states, vision_hidden_states
    
    def _forward_single(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
        return_hidden_states: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for a single sample.
        
        Args:
            history_frames: (K, C, H, W) history video frames
            current_frame: (C, H, W) current observation
            instruction: Navigation instruction text
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Tuple of (hidden_states, vision_hidden_states)
        """
        # Prepare messages for single sample
        messages, _, _ = self._prepare_messages_single(
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
        
        input_ids = inputs["input_ids"]
        
        # Forward pass
        # 注意：不能使用 torch.no_grad()！需要保留计算图以便梯度回传
        outputs = self.model(
            **inputs,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        
        if return_hidden_states:
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
        
        return hidden_states, vision_hidden_states
    
    def forward(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
        generate_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through Qwen3-VL with true batch processing.
        
        Uses native batch processing with padding for efficiency.
        All samples are processed in a single forward pass.
        
        Args:
            history_frames: (B, K, C, H, W) history video frames
            current_frame: (B, C, H, W) current observation
            instruction: Navigation instruction text (str for all samples, or List[str] per sample)
            return_hidden_states: Whether to return hidden states
            generate_text: Whether to generate text output (only for first sample)
            
        Returns:
            Dict containing:
                - hidden_states: (B, seq_len, hidden_dim) LLM hidden states
                - vision_hidden_states: (B, num_vision_tokens, hidden_dim) vision token hidden states
                - generated_text: Generated text (if generate_text=True, only first sample)
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        batch_size = history_frames.shape[0]
        
        # Use batch forward for efficiency
        hidden_states, vision_hidden_states = self._forward_batch(
            history_frames,
            current_frame,
            instruction,
            return_hidden_states,
        )
        
        # Generate text only for first sample (if requested)
        generated_text = None
        if generate_text:
            # Get instruction for first sample
            if instruction is None:
                sample_instruction = None
            elif isinstance(instruction, list):
                sample_instruction = instruction[0] if len(instruction) > 0 else None
            else:
                sample_instruction = instruction
                
            generated_text = self._generate_text_single(
                history_frames[0], current_frame[0], sample_instruction
            )
        
        return {
            "hidden_states": hidden_states,
            "vision_hidden_states": vision_hidden_states,
            "generated_text": generated_text,
        }
    
    def _generate_text_single(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
    ) -> str:
        """Generate text for a single sample."""
        messages, _, _ = self._prepare_messages_single(
            history_frames, current_frame, instruction
        )
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return generated_text
    
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

