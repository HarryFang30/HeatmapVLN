"""
Qwen2.5-VL integration module.

Features:
- Load the Qwen2.5-VL backbone
- Process video frames + current observation + instruction text
- Extract hidden states for downstream heads

Sequence packing is currently disabled on the shared stack.
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*fps.*frames per second.*")
warnings.filterwarnings("ignore", message=".*video_metadata.*")

import os
import time
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

from ..heatmap.input_constructor import find_text_anchor_positions
from ..runtime_compat import ensure_transformers_runtime_compat

logger = logging.getLogger(__name__)
VIEW_NAMES = ("front", "right", "back", "left")

# Aligned with InternNav: special token ID for trajectory query placeholders.
# These positions in input_ids are replaced by learnable latent_queries before
# the LLM forward pass.  The token ID matches InternNav's vocabulary entry so
# that the same backbone weights are compatible.
TRAJ_TOKEN_INDEX = 151667

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
class Qwen2_5VLConfig:
    """Configuration for the Qwen2.5-VL integration wrapper."""
    
    # Model path
    model_path: str = "./models/internnav_backbone"
    
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
    
    # Video processing
    max_video_frames: int = 16  # Maximum frames to process
    
    # Sequence Packing settings (legacy; disabled on the shared stack)
    enable_packing: bool = False  # Whether to use sequence packing
    max_seq_length: int = 4096    # Maximum packed sequence length
    spatial_merge_size: int = 2   # Vision spatial merge size for position IDs
    
    # LoRA configuration
    use_lora: bool = False        # Enable LoRA adapters
    lora_rank: int = 16           # LoRA rank
    lora_alpha: int = 32          # LoRA alpha
    lora_num_layers: int = 4      # Number of last LLM layers to apply LoRA
    lora_layer_indices: Optional[List[int]] = None  # Exact layer indices (overrides lora_num_layers)
    lora_dropout: float = 0.05    # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Target modules (default: ["q_proj", "v_proj"])
    gradient_checkpointing: bool = False
    enable_internal_profiling: bool = False
    enable_runtime_timing: bool = False
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_backend: str = "inductor"
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


class _ModuleTimingProfiler:
    """Lightweight forward hook profiler for selected Qwen submodules."""

    def __init__(self, device: torch.device):
        self.device = device
        self._handles: List[Any] = []
        self._starts: Dict[int, float] = {}
        self._totals: Dict[str, float] = {}
        self._registered: set[Tuple[int, str]] = set()

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def register(self, module: Optional[nn.Module], key: str) -> None:
        if module is None or not isinstance(module, nn.Module):
            return
        reg_key = (id(module), key)
        if reg_key in self._registered:
            return
        self._registered.add(reg_key)

        def _pre_hook(mod: nn.Module, _inputs: Tuple[Any, ...]) -> None:
            self._sync()
            self._starts[id(mod)] = time.perf_counter()

        def _post_hook(mod: nn.Module, _inputs: Tuple[Any, ...], _output: Any) -> None:
            start = self._starts.pop(id(mod), None)
            self._sync()
            if start is None:
                return
            self._totals[key] = self._totals.get(key, 0.0) + (time.perf_counter() - start)

        self._handles.append(module.register_forward_pre_hook(_pre_hook))
        self._handles.append(module.register_forward_hook(_post_hook))

    def reset(self) -> None:
        self._starts.clear()
        self._totals = {}

    def snapshot(self) -> Dict[str, float]:
        totals = dict(self._totals)
        self.reset()
        return totals

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._starts.clear()
        self._totals = {}
        self._registered.clear()


class Qwen2_5VLIntegration(nn.Module):
    """Qwen2.5-VL integration wrapper for VLN Pipeline."""
    
    def __init__(self, config: Qwen2_5VLConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Model and processor (lazy loading)
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        self._runtime_compat_state = None
        
        # Token IDs are set during model loading based on backbone type
        self.video_token_id = None
        self.image_token_id = None
        self.vision_start_id = None
        self.vision_end_id = None
        
        # Sequence packing state
        self._packing_enabled = config.enable_packing
        self._varlen_attention_replaced = False
        self._internal_profiler: Optional[_ModuleTimingProfiler] = None
        self._last_internal_timings: Dict[str, float] = {}
        
        logger.info("VLM Integration initialized (model will be loaded on first forward)")
    
    def _load_model(self):
        """Load the Qwen2.5-VL backbone and processor."""
        if self._model_loaded:
            return

        self._runtime_compat_state = ensure_transformers_runtime_compat(
            model_path=self.config.model_path,
            requested_backbone_type="qwen2_5_vl",
            requested_attn_implementation=self.config.attn_implementation,
            logger=logger,
        )
        logger.info("Detected backbone type: qwen2_5_vl")

        try:
            from transformers import AutoProcessor

            self._load_qwen25vl()

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            if self.config.use_lora:
                self._apply_lora()

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )
            if self.config.gradient_checkpointing:
                base = getattr(self.model, "base_model", self.model)
                if hasattr(base, "gradient_checkpointing_enable"):
                    base.gradient_checkpointing_enable()
                    logger.info("VLM gradient checkpointing enabled (saves ~60%% activation memory)")

            self._model_loaded = True
            self.processor.tokenizer.padding_side = "left"

            logger.info(
                "%s loaded on %s (attn=%s)",
                "qwen2_5_vl", self.device, self.config.attn_implementation,
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info("Parameters: %s (frozen, trainable: %d)", f"{total_params:,}", trainable_params)
            self._setup_internal_profiler()
            self._maybe_enable_compile()

        except Exception as e:
            logger.error("Failed to load Qwen2.5-VL backbone: %s", e)
            raise

    def _load_qwen25vl(self):
        """Load a Qwen2.5-VL backbone."""
        from transformers import Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Qwen2.5-VL from %s", self.config.model_path)
        self.model = self._load_with_attn_fallback(
            Qwen2_5_VLForConditionalGeneration, self.config.model_path,
        )
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", 151655)
        self.video_token_id = getattr(cfg, "video_token_id", 151656)
        self.vision_start_id = getattr(cfg, "vision_start_token_id", 151652)
        self.vision_end_id = getattr(cfg, "vision_end_token_id", 151653)

    def _load_with_attn_fallback(self, model_cls, model_path: str):
        """Try loading with the requested attention impl, fall back to sdpa."""
        requested = self.config.attn_implementation
        candidates: List[str] = []
        if requested == "flash_attention_2":
            flash_available = bool(
                self._runtime_compat_state and self._runtime_compat_state.flash_attn_available
            )
            if flash_available:
                candidates.append(requested)
            else:
                logger.warning("Skipping flash_attention_2 because flash_attn is unavailable in this environment")
        else:
            candidates.append(requested)
        if "sdpa" not in candidates:
            candidates.append("sdpa")

        for attn_impl in candidates:
            try:
                logger.info("Trying attention implementation: %s", attn_impl)
                model = model_cls.from_pretrained(
                    model_path,
                    torch_dtype=self.config.get_torch_dtype(),
                    attn_implementation=attn_impl,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                if attn_impl != requested:
                    logger.warning("Attention fallback: requested `%s`, using `%s`", requested, attn_impl)
                self.config.attn_implementation = attn_impl
                return model
            except Exception as exc:
                logger.warning("Failed with attention `%s`: %s", attn_impl, exc)
        raise RuntimeError(f"Failed to load model from {model_path}")

    @staticmethod
    def _get_nested_module(root: Any, path: str) -> Optional[nn.Module]:
        module = root
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module if isinstance(module, nn.Module) else None

    def _setup_internal_profiler(self) -> None:
        if not self.config.enable_internal_profiling:
            return
        if self._internal_profiler is not None:
            return

        base_model = getattr(self.model, "model", self.model)
        profiler = _ModuleTimingProfiler(self.device)

        def register_first(root: Any, key: str, candidates: List[str]) -> Optional[nn.Module]:
            for path in candidates:
                module = self._get_nested_module(root, path)
                if module is not None:
                    profiler.register(module, key)
                    return module
            return None

        visual_root = register_first(base_model, "qwen_visual_encode_s", ["visual", "visual_module"])
        if visual_root is not None:
            register_first(visual_root, "qwen_visual_patch_embed_s", ["patch_embed"])
            register_first(
                visual_root,
                "qwen_visual_pos_embed_s",
                ["pos_embed", "position_embedding", "positional_embedding"],
            )
            register_first(
                visual_root,
                "qwen_visual_rotary_s",
                ["rot_pos_emb", "rotary_pos_emb"],
            )
            register_first(visual_root, "qwen_visual_merger_s", ["merger", "proj", "projector"])

            visual_blocks = getattr(visual_root, "blocks", None)
            if isinstance(visual_blocks, (nn.ModuleList, list, tuple)):
                for block in visual_blocks:
                    if not isinstance(block, nn.Module):
                        continue
                    profiler.register(block, "qwen_visual_blocks_s")
                    for attr in ("attn", "self_attn"):
                        profiler.register(getattr(block, attr, None), "qwen_visual_attn_s")
                    profiler.register(getattr(block, "mlp", None), "qwen_visual_mlp_s")
                    for attr in ("norm", "norm1", "norm2"):
                        profiler.register(getattr(block, attr, None), "qwen_visual_norm_s")

        language_root = register_first(base_model, "qwen_language_model_s", ["language_model"])
        if language_root is not None:
            language_layers = getattr(language_root, "layers", None)
            if language_layers is None:
                language_layers = getattr(getattr(language_root, "model", None), "layers", None)

            if isinstance(language_layers, (nn.ModuleList, list, tuple)):
                for layer in language_layers:
                    if not isinstance(layer, nn.Module):
                        continue
                    profiler.register(layer, "qwen_llm_layers_s")
                    for attr in ("self_attn", "attention", "full_attn"):
                        profiler.register(getattr(layer, attr, None), "qwen_llm_full_attn_s")
                    for attr in ("linear_attn", "delta_net"):
                        profiler.register(getattr(layer, attr, None), "qwen_llm_linear_attn_s")
                    profiler.register(getattr(layer, "mlp", None), "qwen_llm_mlp_s")
                    for attr in ("input_layernorm", "post_attention_layernorm", "norm", "norm1", "norm2"):
                        profiler.register(getattr(layer, attr, None), "qwen_llm_norm_s")

        if profiler._handles:
            self._internal_profiler = profiler
            logger.info("Enabled Qwen internal profiling hooks")
        else:
            logger.warning("Qwen internal profiling requested, but no matching modules were found")

    def _consume_internal_timings(self) -> Dict[str, float]:
        timings = dict(self._last_internal_timings)
        self._last_internal_timings = {}
        return timings

    def _compile_attr(self, root: Any, attr_name: str, label: str) -> bool:
        module = getattr(root, attr_name, None)
        if module is None or not isinstance(module, nn.Module):
            return False
        try:
            compiled = torch.compile(
                module,
                mode=self.config.compile_mode,
                backend=self.config.compile_backend,
            )
            setattr(root, attr_name, compiled)
            logger.info(
                "Compiled Qwen submodule `%s` with torch.compile(mode=%s, backend=%s)",
                label,
                self.config.compile_mode,
                self.config.compile_backend,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to compile Qwen submodule `%s`: %s", label, exc)
            return False

    def _compile_backend_available(self) -> bool:
        if self.config.compile_backend != "inductor":
            return True
        try:
            from triton.compiler.compiler import triton_key  # noqa: F401
            return True
        except Exception as exc:
            logger.warning(
                "Skip torch.compile backend `%s` because runtime dependency check failed: %s",
                self.config.compile_backend,
                exc,
            )
            return False

    def _maybe_enable_compile(self) -> None:
        if not self.config.enable_compile:
            return
        if self.config.enable_internal_profiling:
            logger.info("Skip torch.compile because internal profiling is enabled")
            return
        if self.config.use_lora:
            logger.info("Skip torch.compile because LoRA is enabled")
            return
        if not self._compile_backend_available():
            return
        logger.info(
            "Skip torch.compile for Qwen2.5-VL on this stack: "
            "visual submodules are unstable with dynamic shapes and "
            "the language model depends on FLA Triton kernels that are not "
            "reliably compatible with Inductor here"
        )
    
    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora(self):
        """
        Apply LoRA adapters to the last N layers of Qwen2.5-VL's language model.
        
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
            num_layers = 28  # InternNav Qwen2.5-VL default
            logger.warning(f"Could not detect layer count, using default: {num_layers}")
        else:
            logger.info(f"Detected {num_layers} LLM layers")
        
        if self.config.lora_layer_indices is not None:
            lora_layers = list(self.config.lora_layer_indices)
        else:
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
        """Sequence packing is disabled on the current Qwen2.5-VL stack."""
        logger.warning(
            "Sequence packing is not supported on the current Qwen2.5-VL "
            "training stack. Use standard batching instead."
        )
        return False
    
    def forward_packed(self, packed_batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Disabled for Qwen2.5-VL on the current training stack."""
        raise NotImplementedError(
            "Sequence packing is not supported for Qwen2.5-VL. "
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
        
        prompt_text = (
            "You are a navigation assistant. "
            "The video shows the historical trajectory from a forward-facing camera. "
            "The image shows your current front view. "
            f"Instruction: {instruction}. "
            "Understand the spatial layout and identify where you came from."
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

    @staticmethod
    def _views_tensor_to_dict(views: torch.Tensor) -> Dict[str, torch.Tensor]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(VIEW_NAMES)}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(
                f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}"
            )
        return [
            self._views_tensor_to_dict(history_panoramas[idx])
            for idx in range(history_panoramas.shape[0])
        ]

    @staticmethod
    def _pad_and_stack(tensors: List[torch.Tensor], pad_dim: int = 1) -> torch.Tensor:
        """Pad a variable-length dimension and stack into a batch tensor."""
        max_len = max(t.shape[pad_dim] for t in tensors)
        padded = []
        for t in tensors:
            diff = max_len - t.shape[pad_dim]
            if diff > 0:
                pad_shape = list(t.shape)
                pad_shape[pad_dim] = diff
                t = torch.cat(
                    [t, torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)],
                    dim=pad_dim,
                )
            padded.append(t)
        return torch.cat(padded, dim=0)

    def _forward_model_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        return_hidden_states: bool,
        skip_lm_head: bool = False,
        latent_queries: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
        """Run Qwen on already prepared multimodal inputs.

        Aligned with InternNav's traj-token mechanism:
        - Training: ``input_ids`` already contains ``TRAJ_TOKEN_INDEX``
          placeholders appended by the collator (after pixel-goal text).
        - Inference: if no ``TRAJ_TOKEN_INDEX`` found, they are appended here.
        - In both cases a forward pre-hook replaces the TRAJ embeddings
          with ``latent_queries`` so they attend to the full context
          (including pixel-goal coordinates).

        Args:
            skip_lm_head: bypass the LM head matmul (heatmap-only training).
            latent_queries: (B, n_query, hidden_dim) learnable trajectory
                condition vectors injected at TRAJ_TOKEN_INDEX positions.
        """
        raw_input_ids = inputs["input_ids"]
        num_image_tokens = int((raw_input_ids == self.image_token_id).sum().item())

        n_query = 0
        hook_handle = None
        need_hidden = return_hidden_states or (latent_queries is not None)

        # input_ids to use for vision-feature extraction (without TRAJ tokens)
        vision_input_ids = raw_input_ids

        if latent_queries is not None:
            B, n_query, D = latent_queries.shape
            device = raw_input_ids.device

            has_traj_tokens = (raw_input_ids == TRAJ_TOKEN_INDEX).any().item()

            if has_traj_tokens:
                # Training path: TRAJ tokens placed by collator after
                # pixel-goal assistant text.  Strip them from the copy used
                # for vision-feature extraction so lengths stay aligned.
                vision_input_ids = raw_input_ids[:, :-n_query]
            else:
                # Inference / fallback: append TRAJ placeholder tokens.
                traj_ids = torch.full(
                    (B, n_query), TRAJ_TOKEN_INDEX,
                    device=device, dtype=raw_input_ids.dtype,
                )
                inputs["input_ids"] = torch.cat([raw_input_ids, traj_ids], dim=1)

                if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                    mask_ext = torch.ones(
                        B, n_query, device=device,
                        dtype=inputs["attention_mask"].dtype,
                    )
                    inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"], mask_ext], dim=1,
                    )

                if "mm_token_type_ids" in inputs and inputs["mm_token_type_ids"] is not None:
                    mm_ext = torch.zeros(
                        B, n_query, device=device,
                        dtype=inputs["mm_token_type_ids"].dtype,
                    )
                    inputs["mm_token_type_ids"] = torch.cat(
                        [inputs["mm_token_type_ids"], mm_ext], dim=1,
                    )

            # Pre-hook: replace TRAJ_TOKEN_INDEX embeddings with latent_queries
            # (InternNav-style mask-based replacement)
            lq = latent_queries
            _traj_token_id = TRAJ_TOKEN_INDEX
            _input_ids_ref = inputs["input_ids"]

            def _replace_traj_embeds_hook(module, args, kwargs):
                embeds = kwargs.get('inputs_embeds')
                if embeds is not None:
                    traj_mask = _input_ids_ref == _traj_token_id
                    if traj_mask.any():
                        embeds = embeds.clone()
                        flat_lq = lq.to(dtype=embeds.dtype, device=embeds.device)
                        embeds[traj_mask] = flat_lq.reshape(-1, flat_lq.shape[-1])
                        kwargs = dict(kwargs)
                        kwargs['inputs_embeds'] = embeds
                return args, kwargs

            language_model_root = (
                self._get_nested_module(self.model, "model.language_model")
                or self._get_nested_module(self.model, "language_model")
                or self._get_nested_module(self.model, "model")
            )
            if language_model_root is None:
                raise RuntimeError("Could not locate the language model module for latent query injection")

            hook_handle = language_model_root.register_forward_pre_hook(
                _replace_traj_embeds_hook, with_kwargs=True,
            )

        fwd_kwargs = dict(
            **inputs,
            output_hidden_states=need_hidden,
            return_dict=True,
            use_cache=False,
        )
        if self._internal_profiler is not None:
            self._internal_profiler.reset()

        try:
            if skip_lm_head:
                inner_model = getattr(self.model, "model", None)
                if inner_model is None:
                    outputs = self.model(**fwd_kwargs)
                else:
                    try:
                        outputs = inner_model(**fwd_kwargs)
                    except TypeError as exc:
                        if "unexpected keyword argument" not in str(exc):
                            raise
                        outputs = self.model(**fwd_kwargs)
            else:
                outputs = self.model(**fwd_kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if self._internal_profiler is not None:
            self._last_internal_timings = self._internal_profiler.snapshot()
        else:
            self._last_internal_timings = {}

        traj_hidden_states = None
        hidden_states = None

        if need_hidden:
            layer_idx = self.config.hidden_layer_for_features
            if layer_idx == -1:
                layer_idx = len(outputs.hidden_states) - 1
            hidden_states = outputs.hidden_states[layer_idx]

            if n_query > 0:
                last_hs = outputs.hidden_states[-1]
                traj_hidden_states = last_hs[:, -n_query:, :].contiguous()
                hidden_states = hidden_states[:, :-n_query, :].contiguous()

            if not return_hidden_states:
                hidden_states = None

        vision_hidden_states = None
        if hidden_states is not None:
            vision_hidden_states = self._extract_vision_hidden_states(
                hidden_states, vision_input_ids,
            )

        return hidden_states, vision_hidden_states, num_image_tokens, traj_hidden_states

    def _forward_single_panorama(
        self,
        current_views: torch.Tensor,
        history_panoramas: torch.Tensor,
        instruction: Optional[str] = None,
        return_hidden_states: bool = True,
        heatmap_vln: Optional[nn.Module] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[Dict[str, torch.Tensor]]]:
        """Forward a single panoramic sample through one Qwen pass.

        When ``return_hidden_states`` is False (heatmap-only training), the
        Qwen forward is wrapped in ``torch.no_grad()`` to avoid storing
        intermediate activations for the frozen backbone, saving ~4-8 GB VRAM.
        """
        current_views_dict = self._views_tensor_to_dict(current_views)
        history_panoramas_list = self._history_tensor_to_list(history_panoramas)

        if heatmap_vln is not None:
            inputs, num_history = heatmap_vln.prepare_qwen_inputs(
                current_views=current_views_dict,
                history_panoramas=history_panoramas_list,
                instruction=instruction,
                device=self.device,
            )
            heatmap_vln.feat_extractor.clear()
        else:
            raise RuntimeError("Panoramic forward requires a HeatmapVLN instance for single-chain decoding.")

        if return_hidden_states:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                inputs, return_hidden_states,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True,
                )

        heatmap_output = heatmap_vln.decode_from_inputs(inputs, num_history)
        return hidden_states, vision_hidden_states, num_image_tokens, heatmap_output

    def _forward_batch_panorama(
        self,
        current_views: torch.Tensor,
        history_panoramas: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
        heatmap_vln: Optional[nn.Module] = None,
        history_rel_poses: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[Dict[str, torch.Tensor]]]:
        """Batch forward for panoramic input using one batched Qwen pass."""
        if heatmap_vln is None:
            raise RuntimeError("Panoramic forward requires a HeatmapVLN instance for batched decoding.")

        t0 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        inputs, num_histories = heatmap_vln.prepare_qwen_inputs_batch(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=instruction,
            device=self.device,
        )
        image_positions_batch = [
            heatmap_vln._find_image_positions_from_ids(inputs["input_ids"][b])
            for b in range(inputs["input_ids"].shape[0])
        ]
        text_anchors_batch = [
            find_text_anchor_positions(
                inputs["input_ids"][b:b + 1],
                heatmap_vln.processor.tokenizer,
                num_history=num_histories[b],
            )
            for b in range(inputs["input_ids"].shape[0])
        ]
        t1 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        heatmap_vln.feat_extractor.clear()
        heatmap_vln.feat_extractor.prepare_batch_capture(
            image_token_positions_batch=image_positions_batch,
            text_anchor_positions_batch=text_anchors_batch,
            image_grid_thw=inputs.get("image_grid_thw"),
        )

        if return_hidden_states:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                inputs, return_hidden_states,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True,
                )
        t2 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        internal_timings = self._consume_internal_timings() if self.config.enable_runtime_timing else {}

        heatmap_output = heatmap_vln.decode_from_inputs_batch(
            inputs,
            num_histories,
            image_positions_batch=image_positions_batch,
            text_anchors_batch=text_anchors_batch,
            history_rel_poses=history_rel_poses,
        )
        if self.config.enable_runtime_timing:
            t3 = time.perf_counter()
            decode_timings = dict(heatmap_output.get("timings", {}) or {})
            heatmap_output["timings"] = {
                "prepare_inputs_s": t1 - t0,
                "qwen_forward_s": t2 - t1,
                "heatmap_decode_s": t3 - t2,
                "panorama_total_s": t3 - t0,
            }
            heatmap_output["timings"].update(decode_timings)
            heatmap_output["timings"].update(internal_timings)
        return hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hs

    def _forward_batch_panorama_tokenized(
        self,
        panoramic_inputs: Dict[str, torch.Tensor],
        num_histories: List[int],
        text_anchor_positions_batch: Optional[List[Dict[int, int]]] = None,
        return_hidden_states: bool = True,
        heatmap_vln: Optional[nn.Module] = None,
        history_rel_poses: Optional[torch.Tensor] = None,
        latent_queries: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward already-tokenized batch through one Qwen pass.

        When ``heatmap_vln`` is None the heatmap hook/decode pipeline is
        skipped entirely — only the VLM forward (with optional TRAJ latent
        query injection) is executed.  This is the Stage 2 InternNav path
        where VLM input is front-view + lookdown (no panoramic history anchors).
        """
        t0 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in panoramic_inputs.items()}

        if heatmap_vln is not None:
            heatmap_vln._normalize_multimodal_inputs(inputs)
            image_positions_batch = [
                heatmap_vln._find_image_positions_from_ids(inputs["input_ids"][b])
                for b in range(inputs["input_ids"].shape[0])
            ]
            if text_anchor_positions_batch is None:
                text_anchor_positions_batch = [
                    find_text_anchor_positions(
                        inputs["input_ids"][b:b + 1],
                        heatmap_vln.processor.tokenizer,
                        num_history=num_histories[b],
                    )
                    for b in range(inputs["input_ids"].shape[0])
                ]
            t1 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
            heatmap_vln.feat_extractor.clear()
            heatmap_vln.feat_extractor.prepare_batch_capture(
                image_token_positions_batch=image_positions_batch,
                text_anchor_positions_batch=text_anchor_positions_batch,
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        else:
            # Lightweight normalisation (same as HeatmapVLN._normalize_multimodal_inputs)
            if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
                vgt = inputs["video_grid_thw"]
                if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                    inputs["video_grid_thw"] = torch.repeat_interleave(
                        vgt, vgt[:, 0], dim=0,
                    )
                    inputs["video_grid_thw"][:, 0] = 1
            t1 = time.perf_counter() if self.config.enable_runtime_timing else 0.0

        if return_hidden_states:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                inputs, return_hidden_states, latent_queries=latent_queries,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True, latent_queries=latent_queries,
                )
        t2 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        internal_timings = self._consume_internal_timings() if self.config.enable_runtime_timing else {}

        heatmap_output = None
        if heatmap_vln is not None:
            heatmap_output = heatmap_vln.decode_from_inputs_batch(
                inputs,
                num_histories,
                image_positions_batch=image_positions_batch,
                text_anchors_batch=text_anchor_positions_batch,
                history_rel_poses=history_rel_poses,
            )
        if self.config.enable_runtime_timing:
            t3 = time.perf_counter()
            if heatmap_output is not None:
                decode_timings = dict(heatmap_output.get("timings", {}) or {})
                heatmap_output["timings"] = {
                    "prepare_inputs_s": t1 - t0,
                    "qwen_forward_s": t2 - t1,
                    "heatmap_decode_s": t3 - t2,
                    "panorama_total_s": t3 - t0,
                }
                heatmap_output["timings"].update(decode_timings)
                heatmap_output["timings"].update(internal_timings)
        return hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hs
    
    def _forward_batch(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Batch forward by processing each sample individually and stacking.
        Qwen2.5-VL's internal position ID computation does not handle
        padded batches correctly, so we loop over samples.
        """
        batch_size = history_frames.shape[0]

        all_hidden = []
        all_vision = []
        max_image_tokens = 0

        for b in range(batch_size):
            instr_b = (
                instruction[b] if isinstance(instruction, list) else instruction
            )
            hs, vis, n_img, _ = self._forward_single(
                history_frames[b], current_frame[b], instr_b, return_hidden_states,
            )
            all_hidden.append(hs)
            all_vision.append(vis)
            max_image_tokens = max(max_image_tokens, n_img)

        if return_hidden_states:
            hidden_states = self._pad_and_stack(all_hidden)
            vision_hidden_states = self._pad_and_stack(all_vision) if all_vision[0] is not None else None
        else:
            hidden_states = None
            vision_hidden_states = None

        return hidden_states, vision_hidden_states, max_image_tokens, None
    
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

        hidden_states, vision_hidden_states, num_image_tokens, traj_hs = self._forward_model_inputs(
            inputs, return_hidden_states,
        )
        return hidden_states, vision_hidden_states, num_image_tokens, None
    
    def forward(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Optional[Union[str, List[str]]] = None,
        return_hidden_states: bool = True,
        generate_text: bool = False,
        current_views: Optional[torch.Tensor] = None,
        history_panoramas: Optional[torch.Tensor] = None,
        panoramic_inputs: Optional[Dict[str, torch.Tensor]] = None,
        panoramic_num_histories: Optional[List[int]] = None,
        panoramic_text_anchor_positions: Optional[List[Dict[int, int]]] = None,
        heatmap_vln: Optional[nn.Module] = None,
        history_rel_poses: Optional[torch.Tensor] = None,
        latent_queries: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through Qwen2.5-VL with batch processing."""
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        batch_size = current_views.shape[0] if current_views is not None else history_frames.shape[0]
        traj_hidden_states = None

        if panoramic_inputs is not None:
            if panoramic_num_histories is None:
                raise ValueError("panoramic_num_histories is required with panoramic_inputs")
            hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hidden_states = (
                self._forward_batch_panorama_tokenized(
                    panoramic_inputs=panoramic_inputs,
                    num_histories=panoramic_num_histories,
                    text_anchor_positions_batch=panoramic_text_anchor_positions,
                    return_hidden_states=return_hidden_states,
                    heatmap_vln=heatmap_vln,
                    history_rel_poses=history_rel_poses,
                    latent_queries=latent_queries,
                )
            )
        elif current_views is not None and history_panoramas is not None:
            hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hs = (
                self._forward_batch_panorama(
                    current_views=current_views,
                    history_panoramas=history_panoramas,
                    instruction=instruction,
                    return_hidden_states=return_hidden_states,
                    heatmap_vln=heatmap_vln,
                    history_rel_poses=history_rel_poses,
                )
            )
            traj_hidden_states = traj_hs
        else:
            hidden_states, vision_hidden_states, num_image_tokens, heatmap_output = (
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
        
        result = {
            "hidden_states": hidden_states,
            "vision_hidden_states": vision_hidden_states,
            "generated_text": generated_text,
            "num_image_tokens": num_image_tokens,
        }
        if traj_hidden_states is not None:
            result["traj_hidden_states"] = traj_hidden_states
        if heatmap_output is not None:
            result.update(heatmap_output)
        else:
            timings = self._consume_internal_timings()
            if timings:
                result["timings"] = timings
        return result
    
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

    def generate_latents(
        self,
        output_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        latent_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Two-step inference aligned with InternNav ``generate_latents``.

        After the model has auto-regressively generated pixel-goal text,
        this method takes the **full** output sequence (prompt + generated
        tokens), appends ``TRAJ_TOKEN_INDEX`` placeholders, replaces them
        with ``latent_queries``, and runs a single forward pass.  The
        hidden states at the TRAJ positions are the trajectory conditions.

        Args:
            output_ids: (1, L) full token sequence including generated text.
            pixel_values: preprocessed image/video tensors from the processor.
            image_grid_thw: grid layout tensor for vision tokens.
            latent_queries: (1, n_query, hidden_dim) learnable queries.

        Returns:
            traj_hidden_states: (1, n_query, hidden_dim)
        """
        if not self._model_loaded:
            self._load_model()

        n_query = latent_queries.shape[1]
        device = output_ids.device

        traj_suffix = torch.full(
            (1, n_query), TRAJ_TOKEN_INDEX,
            device=device, dtype=output_ids.dtype,
        )
        extended_ids = torch.cat([output_ids, traj_suffix], dim=1)

        with torch.no_grad():
            text_embeds = self.model.model.embed_tokens(extended_ids)

        image_mask = extended_ids == self.image_token_id
        if pixel_values is not None and image_mask.any():
            pixel_values = pixel_values.to(
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )
            image_embeds = self.model.visual(
                pixel_values, grid_thw=image_grid_thw,
            )
            text_embeds[image_mask] = image_embeds.to(
                device=text_embeds.device,
            )[:image_mask.sum(), :]

        lq = latent_queries.to(dtype=text_embeds.dtype, device=text_embeds.device)
        text_embeds[:, -n_query:, :] = lq

        rope_fn = getattr(self.model, 'get_rope_index', None)
        position_ids = None
        if rope_fn is not None:
            position_ids, _ = rope_fn(extended_ids, image_grid_thw)
            position_ids = position_ids.to(device)

        with torch.no_grad():
            outputs = self.model.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        traj_hidden_states = outputs.hidden_states[-1][:, -n_query:, :]
        return traj_hidden_states.contiguous()

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
        # Default for InternNav Qwen2.5-VL
        return 3584
    
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

