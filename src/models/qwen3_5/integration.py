"""
Qwen3.5 Integration Module
===========================

This module provides the Qwen3.5 integration for the VLN pipeline.
It handles video processing and hidden state extraction for downstream heads.

Features:
- Load Qwen3.5 model with flash attention support
- Process video frames + current observation + instruction text
- Extract hidden states for heatmap/action/stop heads

Note: Sequence packing is disabled for Qwen3.5 due to the hybrid
linear+full attention architecture (GatedDeltaNet does not support varlen).
"""

import warnings
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
class Qwen3_5Config:
    """Configuration for Qwen3.5 integration."""
    
    # Model path
    model_path: str = "./models/qwen_3.5"
    
    # Device and dtype
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Attention implementation (sdpa, flash_attention_2, or eager)
    # Blackwell GPU (RTX 5090) 需使用 sdpa，flash_attention_2 不支持 sm_120
    attn_implementation: str = "sdpa"
    
    # Generation settings (for inference mode)
    max_new_tokens: int = 128
    temperature: float = 0.7
    
    # Hidden state extraction
    hidden_layer_for_features: int = -1  # -1 = last layer (deprecated when multi_layer_features=True)
    
    # Multi-layer feature extraction (CVPR 2025 best practice)
    # 从 LLM 的不同深度提取特征并融合，保留空间+语义信息
    multi_layer_features: bool = False
    feature_layer_indices: Optional[List[int]] = None  # e.g. [3, 11, 19, 27] for 32-layer LLM
    
    # Video processing
    max_video_frames: int = 16  # Maximum frames to process
    
    # Sequence Packing settings (based on official Qwen3-VL fine-tuning)
    enable_packing: bool = False  # Whether to use sequence packing
    max_seq_length: int = 4096    # Maximum packed sequence length
    spatial_merge_size: int = 2   # Vision spatial merge size for position IDs
    
    # LoRA configuration
    use_lora: bool = False        # Enable LoRA adapters
    lora_rank: int = 16           # LoRA rank
    lora_alpha: int = 32          # LoRA alpha
    lora_num_layers: int = 4      # Number of last LLM layers to apply LoRA
    lora_dropout: float = 0.05    # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Target modules (default: ["q_proj", "v_proj"])
    
    # ViT pre-merge feature extraction for spatial-semantic fusion
    vit_hook_layers: Optional[List[int]] = None  # ViT block indices (e.g. [6, 13, 20, 26])
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class Qwen3_5Integration(nn.Module):
    """
    Qwen3.5 Integration for VLN Pipeline.
    
    This class wraps the Qwen3.5 model to:
    1. Process video frames and text instructions
    2. Extract hidden states for downstream heads
    3. Provide a clean interface for the pipeline
    
    Args:
        config: Qwen3_5Config with model settings
    """
    
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Model and processor (lazy loading)
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        # Special token IDs (Qwen3.5 tokenizer)
        self.video_token_id = 248057  # <|video_pad|>
        self.image_token_id = 248056  # <|image_pad|>
        self.vision_start_id = 248053  # <|vision_start|>
        self.vision_end_id = 248054  # <|vision_end|>
        
        # Sequence packing state
        self._packing_enabled = config.enable_packing
        self._varlen_attention_replaced = False
        
        # ViT pre-merge feature hooks
        self._vit_hook_indices = config.vit_hook_layers or []
        self._vit_hook_handles: List = []
        self._vit_block_features: Dict[int, List[torch.Tensor]] = {}
        
        logger.info(f"Qwen3_5Integration initialized (model will be loaded on first forward)")
    
    def _load_model(self):
        """Load the Qwen3.5 model and processor."""
        if self._model_loaded:
            return
        
        try:
            from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading Qwen3.5 from {self.config.model_path}")
            
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.get_torch_dtype(),
                attn_implementation=self.config.attn_implementation,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Freeze all parameters (never train the backbone)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Apply LoRA if configured (after freezing base model)
            if self.config.use_lora:
                self._apply_lora()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )
            
            self._model_loaded = True
            
            # 设置 padding_side 为 left，用于批量处理
            self.processor.tokenizer.padding_side = 'left'
            
            logger.info(f"Qwen3.5 loaded successfully on {self.device}")
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {total_params:,} (all frozen, trainable: {trainable_params})")
            logger.info(f"Batch processing enabled (padding_side='left')")
            
            # Register ViT block hooks for pre-merge spatial features
            if self._vit_hook_indices:
                self._register_vit_hooks()
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3.5: {e}")
            raise
    
    # ------------------------------------------------------------------
    # ViT pre-merge feature hooks
    # ------------------------------------------------------------------

    def _get_visual_module(self):
        """Resolve the Qwen3_5VisionModel regardless of LoRA wrapping."""
        model = self.model
        # PeftModel wraps: peft_model.base_model (LoraModel) .model (Qwen3_5ForConditionalGeneration)
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            inner = model.model
            # inner may be Qwen3_5ForConditionalGeneration or Qwen3_5Model
            if hasattr(inner, 'visual'):
                return inner.visual
            if hasattr(inner, 'model') and hasattr(inner.model, 'visual'):
                return inner.model.visual
        if hasattr(model, 'visual'):
            return model.visual
        raise RuntimeError("Cannot locate vision model in model hierarchy")

    def _register_vit_hooks(self):
        """Register forward hooks on ViT blocks to capture pre-merge features."""
        visual = self._get_visual_module()
        num_blocks = len(visual.blocks)
        for idx in self._vit_hook_indices:
            if idx >= num_blocks:
                logger.warning(
                    f"ViT hook block {idx} out of range (max {num_blocks - 1}), skipping"
                )
                continue

            def _make_hook(layer_idx: int):
                def _hook(module, _input, output):
                    self._vit_block_features[layer_idx].append(output.detach())
                return _hook

            handle = visual.blocks[idx].register_forward_hook(_make_hook(idx))
            self._vit_hook_handles.append(handle)
        logger.info(f"Registered ViT pre-merge hooks on blocks {self._vit_hook_indices}")

    def _clear_vit_hooks(self):
        """Reset captured features before each forward pass."""
        self._vit_block_features = {idx: [] for idx in self._vit_hook_indices}

    def _extract_image_vit_features(
        self,
        image_grid_thw: Optional[torch.Tensor],
    ) -> Optional[List[torch.Tensor]]:
        """
        Extract per-image pre-merge ViT features from hook captures,
        un-shuffle the pixel-shuffle ordering, and reshape to 2-D feature maps.

        Returns:
            List of ``len(vit_hook_indices)`` tensors, each
            ``(num_images, vit_dim, h_pre, w_pre)``, or *None*.
        """
        if not self._vit_hook_indices or image_grid_thw is None:
            return None

        merge_s = self.config.spatial_merge_size
        num_images = image_grid_thw.shape[0]

        # Per-image pre-merge patch counts
        per_image_sizes = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ).tolist()

        result_layers: List[torch.Tensor] = []
        for idx in self._vit_hook_indices:
            captures = self._vit_block_features.get(idx, [])
            if not captures:
                return None
            # First capture = images (get_image_features is called before get_video_features)
            features = captures[0]  # (total_image_patches, vit_dim)

            per_image = torch.split(features, [int(s) for s in per_image_sizes])
            images_2d = []
            for img_i in range(num_images):
                patches = per_image[img_i]  # (t*h*w, dim)
                t_val, h_val, w_val = (int(v) for v in image_grid_thw[img_i].tolist())
                dim = patches.shape[-1]
                h_m = h_val // merge_s
                w_m = w_val // merge_s
                # Un-shuffle: (h_m, w_m, merge, merge, dim) → (h_pre, w_pre, dim)
                patches = patches.view(t_val, h_m, w_m, merge_s, merge_s, dim)
                patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
                patches = patches.view(t_val, h_m * merge_s, w_m * merge_s, dim)
                patches = patches.squeeze(0)          # (h_pre, w_pre, dim)
                patches = patches.permute(2, 0, 1)    # (dim, h_pre, w_pre)
                images_2d.append(patches)

            result_layers.append(torch.stack(images_2d, dim=0))

        return result_layers  # List[(num_images, vit_dim, h_pre, w_pre)]

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora(self):
        """
        Apply LoRA adapters to the last N layers of Qwen3.5's language model.
        
        Uses PEFT library to add low-rank adapters to q_proj and v_proj
        in the specified layers. LoRA parameters are trainable while
        the base model remains frozen.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.error("peft not installed. Install with: pip install peft")
            raise ImportError("peft is required for LoRA. Install with: pip install peft")
        
        num_layers = None
        if hasattr(self.model, 'model'):
            m = self.model.model
            if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers'):
                num_layers = len(m.language_model.layers)
            elif hasattr(m, 'layers'):
                num_layers = len(m.layers)
        if num_layers is None and hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            num_layers = len(self.model.language_model.model.layers)
        
        if num_layers is None:
            num_layers = 32  # Qwen3.5 7B default
            logger.warning(f"Could not detect layer count, using default: {num_layers}")
        else:
            logger.info(f"Detected {num_layers} LLM layers")
        
        # Apply LoRA to the last N layers
        lora_layers = list(range(
            num_layers - self.config.lora_num_layers,
            num_layers
        ))
        
        # 使用配置中的 target_modules，默认 ["q_proj", "v_proj"]
        target_modules = self.config.lora_target_modules or ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            layers_to_transform=lora_layers,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Log LoRA info
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"LoRA applied to layers {lora_layers} ({target_modules}), "
            f"rank={self.config.lora_rank}, alpha={self.config.lora_alpha}"
        )
        logger.info(
            f"LoRA trainable: {lora_params:,} / {total_params:,} "
            f"({100 * lora_params / total_params:.4f}%)"
        )
    
    def enable_sequence_packing(self) -> bool:
        """Sequence packing is disabled for Qwen3.5 (hybrid attention)."""
        logger.warning(
            "Sequence packing is not supported for Qwen3.5 due to hybrid "
            "linear+full attention architecture. Use standard batching instead."
        )
        return False
    
    def forward_packed(self, packed_batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Disabled for Qwen3.5 — use forward() with standard batching."""
        raise NotImplementedError(
            "Sequence packing is not supported for Qwen3.5. "
            "Set enable_packing=false in config and use standard forward()."
        )
    
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
        
        is_panoramic = (current_frame.shape[-1] != current_frame.shape[-2] or
                        current_frame.shape[-1] > 300)
        if is_panoramic:
            prompt_text = (
                "You are a navigation assistant. "
                "The video shows the historical trajectory from a forward-facing camera. "
                "The image shows your current panoramic observation in a 2x2 grid: "
                "top-left=Front, top-right=Right, bottom-left=Back, bottom-right=Left. "
                f"Instruction: {instruction}. "
                "Understand the full 360-degree spatial layout and identify where you came from."
            )
        else:
            prompt_text = (
                f"You are a navigation assistant. "
                f"The video shows the historical trajectory, and the image shows your current view. "
                f"Instruction: {instruction}. "
                f"Understand the spatial layout and identify where you came from."
            )
        
        # Build message content
        # 使用 nframes 明确指定帧数，避免 fps 采样警告
        content = [
            {"type": "video", "video": history_pil, "nframes": len(history_pil)},
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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Batch forward by processing each sample individually and stacking.
        Qwen3.5's internal position ID computation does not handle
        padded batches correctly, so we loop over samples.
        """
        batch_size = history_frames.shape[0]

        all_hidden = []
        all_vision = []
        all_vit_pre = []
        max_image_tokens = 0

        for b in range(batch_size):
            instr_b = (
                instruction[b] if isinstance(instruction, list) else instruction
            )
            hs, vis, n_img, vit_pre = self._forward_single(
                history_frames[b], current_frame[b], instr_b, return_hidden_states,
            )
            all_hidden.append(hs)
            all_vision.append(vis)
            all_vit_pre.append(vit_pre)
            max_image_tokens = max(max_image_tokens, n_img)

        # Stack results
        def _pad_and_stack(tensors, pad_dim=1):
            """Pad variable-length dim and stack into batch."""
            max_len = max(t.shape[pad_dim] for t in tensors)
            padded = []
            for t in tensors:
                diff = max_len - t.shape[pad_dim]
                if diff > 0:
                    pad_shape = list(t.shape)
                    pad_shape[pad_dim] = diff
                    t = torch.cat([t, torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)], dim=pad_dim)
                padded.append(t)
            return torch.cat(padded, dim=0)

        if return_hidden_states:
            if isinstance(all_hidden[0], list):
                # Multi-layer: list of tensors per layer
                n_layers = len(all_hidden[0])
                hidden_states = []
                for li in range(n_layers):
                    layer_tensors = [h[li] for h in all_hidden]
                    hidden_states.append(_pad_and_stack(layer_tensors))
                vision_hidden_states = []
                for li in range(n_layers):
                    layer_vis = [v[li] for v in all_vision]
                    vision_hidden_states.append(_pad_and_stack(layer_vis))
            else:
                hidden_states = _pad_and_stack(all_hidden)
                vision_hidden_states = _pad_and_stack(all_vision) if all_vision[0] is not None else None
        else:
            hidden_states = None
            vision_hidden_states = None

        # Stack ViT pre-merge features
        vit_pre_merge = None
        if all_vit_pre[0] is not None:
            n_vit_layers = len(all_vit_pre[0])
            vit_pre_merge = []
            for li in range(n_vit_layers):
                layer_feats = [vp[li] for vp in all_vit_pre]
                vit_pre_merge.append(torch.cat(layer_feats, dim=0))

        return hidden_states, vision_hidden_states, max_image_tokens, vit_pre_merge
    
    def _forward_single(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[str] = None,
        return_hidden_states: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Forward pass for a single sample.
        
        Args:
            history_frames: (K, C, H, W) history video frames
            current_frame: (C, H, W) current observation
            instruction: Navigation instruction text
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Tuple of (hidden_states, vision_hidden_states, num_image_tokens)
        """
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
        
        input_ids = inputs["input_ids"]
        
        image_mask = input_ids == self.image_token_id
        num_image_tokens = int(image_mask.sum().item())
        
        # Expand video_grid_thw temporal dimension to match mm_token_type_ids groups
        # [[t, h, w]] → t copies of [[1, h, w]] since the processor splits
        # multi-temporal video tokens into separate groups in mm_token_type_ids
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            vgt = inputs["video_grid_thw"]
            if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                inputs["video_grid_thw"] = torch.repeat_interleave(
                    vgt, vgt[:, 0], dim=0
                )
                inputs["video_grid_thw"][:, 0] = 1

        if self._vit_hook_indices:
            self._clear_vit_hooks()
        
        outputs = self.model(
            **inputs,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        
        vit_pre_merge = None
        if self._vit_hook_indices:
            vit_pre_merge = self._extract_image_vit_features(
                inputs.get("image_grid_thw")
            )
        
        if return_hidden_states:
            if self.config.multi_layer_features and self.config.feature_layer_indices:
                multi_hidden = []
                for li in self.config.feature_layer_indices:
                    idx = li if li >= 0 else len(outputs.hidden_states) + li
                    idx = min(idx, len(outputs.hidden_states) - 1)
                    multi_hidden.append(outputs.hidden_states[idx])
                hidden_states = multi_hidden
            else:
                layer_idx = self.config.hidden_layer_for_features
                if layer_idx == -1:
                    layer_idx = len(outputs.hidden_states) - 1
                hidden_states = outputs.hidden_states[layer_idx]
        else:
            hidden_states = None
        
        vision_hidden_states = None
        if hidden_states is not None:
            if isinstance(hidden_states, list):
                vision_hidden_list = []
                for hs in hidden_states:
                    vis_hs = self._extract_vision_hidden_states(hs, input_ids)
                    vision_hidden_list.append(vis_hs)
                vision_hidden_states = vision_hidden_list
            else:
                vision_hidden_states = self._extract_vision_hidden_states(
                    hidden_states, input_ids
                )
        
        return hidden_states, vision_hidden_states, num_image_tokens, vit_pre_merge
    
    def forward(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
        generate_text: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Qwen3.5 with batch processing."""
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        batch_size = history_frames.shape[0]
        
        # Use batch forward for efficiency
        hidden_states, vision_hidden_states, num_image_tokens, vit_pre_merge = (
            self._forward_batch(
                history_frames, current_frame, instruction, return_hidden_states,
            )
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
            "num_image_tokens": num_image_tokens,
            "vit_pre_merge_features": vit_pre_merge,
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
    
    def is_multi_layer(self) -> bool:
        """Check if multi-layer feature extraction is enabled."""
        return self.config.multi_layer_features and bool(self.config.feature_layer_indices)
    
    def get_num_feature_layers(self) -> int:
        """Get number of feature layers being extracted."""
        if self.is_multi_layer():
            return len(self.config.feature_layer_indices)
        return 1
    
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
        # Default for Qwen3.5 7B
        return 4096
    
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

