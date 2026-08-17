"""
Qwen2.5-VL integration module.

Features:
- Load the Qwen2.5-VL backbone
- Process video frames + current observation + instruction text
- Extract hidden states for downstream heads

Sequence packing is currently disabled on the shared stack.
"""

import hashlib
import inspect
import json
import logging
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*fps.*frames per second.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*video_metadata.*")
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..heatmap.input_constructor import find_text_anchor_positions
from ..runtime_compat import ensure_transformers_runtime_compat

logger = logging.getLogger(__name__)
VIEW_NAMES = ("front", "right", "back", "left")
STRUCTURED_VIEW_CLASSES = ("stop", "front", "right", "back", "left", "turn")
DEFAULT_LORA_ADAPTER_NAME = "default"
STOP_DECISION_ADAPTER_NAME = "stop_decision"

# Aligned with InternNav: special token ID for trajectory query placeholders.
# These positions in input_ids are replaced by learnable latent_queries before
# the LLM forward pass.  The token ID matches InternNav's vocabulary entry so
# that the same backbone weights are compatible.
TRAJ_TOKEN_INDEX = 151667

# Import sequence packing utilities
try:
    from .sequence_packing import (
        IMAGE_TOKEN_ID,
        VIDEO_TOKEN_ID,
        FlattenedDataCollatorForVLN,
        get_rope_index_3,
        replace_attention_with_varlen,
        split_packed_hidden_states,
        split_packed_vision_hidden_states,
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
    lora_layer_indices: list[int] | None = None  # Exact layer indices (overrides lora_num_layers)
    lora_dropout: float = 0.05    # LoRA dropout
    lora_target_modules: list[str] | None = None  # Target modules (default: ["q_proj", "v_proj"])
    heatmap_trains_backbone: bool = False  # Allow heatmap loss to backprop through backbone
    gradient_checkpointing: bool = False
    enable_internal_profiling: bool = False
    enable_runtime_timing: bool = False
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_backend: str = "inductor"
    frozen_traj_inference_mode: bool = False
    traj_last_hidden_state_only: bool = False

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
        self._handles: list[Any] = []
        self._starts: dict[int, float] = {}
        self._totals: dict[str, float] = {}
        self._registered: set[tuple[int, str]] = set()

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def register(self, module: nn.Module | None, key: str) -> None:
        if module is None or not isinstance(module, nn.Module):
            return
        reg_key = (id(module), key)
        if reg_key in self._registered:
            return
        self._registered.add(reg_key)

        def _pre_hook(mod: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._sync()
            self._starts[id(mod)] = time.perf_counter()

        def _post_hook(mod: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
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

    def snapshot(self) -> dict[str, float]:
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
        self._internal_profiler: _ModuleTimingProfiler | None = None
        self._last_internal_timings: dict[str, float] = {}
        self._lm_head_bypass_lock = threading.RLock()

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
                    try:
                        base.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False},
                        )
                    except TypeError as exc:
                        if self.config.heatmap_trains_backbone:
                            raise RuntimeError(
                                "Heatmap-to-backbone training requires non-reentrant "
                                "gradient checkpointing because heatmap features are "
                                "captured by forward hooks. Upgrade transformers or "
                                "disable llm.gradient_checkpointing."
                            ) from exc
                        base.gradient_checkpointing_enable()
                        logger.warning(
                            "VLM gradient checkpointing enabled with the legacy "
                            "reentrant implementation; hook-based auxiliary losses "
                            "must remain detached"
                        )
                    else:
                        logger.info(
                            "VLM non-reentrant gradient checkpointing enabled "
                            "(hook-based auxiliary gradients preserved)"
                        )
                # Frozen-backbone LoRA training needs the input embeddings to
                # require gradients; otherwise checkpointed layers can detach
                # the graph and return an LM loss without grad_fn.
                if self.config.use_lora and hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                    logger.info("VLM input gradients enabled for LoRA + gradient checkpointing")

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
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Qwen2.5-VL from %s", self.config.model_path)
        model_config = self._build_qwen25vl_config_for_path(Qwen2_5_VLConfig)
        self.model = self._load_with_attn_fallback(
            Qwen2_5_VLForConditionalGeneration, self.config.model_path, model_config,
        )
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", 151655)
        self.video_token_id = getattr(cfg, "video_token_id", 151656)
        self.vision_start_id = getattr(cfg, "vision_start_token_id", 151652)
        self.vision_end_id = getattr(cfg, "vision_end_token_id", 151653)

    def _build_qwen25vl_config_for_path(self, config_cls):
        """Reuse full InternNav checkpoints by loading their backbone with Qwen2.5-VL config."""
        cfg_path = Path(self.config.model_path) / "config.json"
        if not cfg_path.is_file():
            return None

        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        if raw_cfg.get("model_type") != "internvla_n1":
            return None

        qwen_cfg = dict(raw_cfg)
        qwen_cfg["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        qwen_cfg["model_type"] = "qwen2_5_vl"
        qwen_cfg["auto_map"] = {
            "AutoConfig": "transformers.Qwen2_5_VLConfig",
            "AutoModelForCausalLM": "transformers.Qwen2_5_VLForConditionalGeneration",
        }
        for key in ("n_query", "system1", "model_cfg"):
            qwen_cfg.pop(key, None)

        logger.info("Detected InternNav full checkpoint; loading backbone with Qwen2.5-VL config")
        return config_cls.from_dict(qwen_cfg)

    def _load_with_attn_fallback(self, model_cls, model_path: str, model_config=None):
        """Try loading with the requested attention impl, fall back to sdpa."""
        requested = self.config.attn_implementation
        candidates: list[str] = []
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
                load_kwargs = dict(
                    torch_dtype=self.config.get_torch_dtype(),
                    attn_implementation=attn_impl,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                if model_config is not None:
                    load_kwargs["config"] = model_config
                model = model_cls.from_pretrained(model_path, **load_kwargs)
                if attn_impl != requested:
                    logger.warning("Attention fallback: requested `%s`, using `%s`", requested, attn_impl)
                self.config.attn_implementation = attn_impl
                return model
            except Exception as exc:
                logger.warning("Failed with attention `%s`: %s", attn_impl, exc)
        raise RuntimeError(f"Failed to load model from {model_path}")

    @staticmethod
    def _get_nested_module(root: Any, path: str) -> nn.Module | None:
        module = root
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module if isinstance(module, nn.Module) else None

    def _locate_conditional_generation_lm_head(
        self,
    ) -> tuple[str, nn.Module, nn.Module]:
        """Find the physical conditional-generation module and its LM head.

        PEFT wraps Qwen as ``PeftModel -> LoraModel ->
        Qwen2_5_VLForConditionalGeneration``. Attribute proxying makes
        ``hasattr(wrapper, 'lm_head')`` ambiguous, so only direct registered
        child modules named ``lm_head`` are accepted. There must be exactly one
        physical head; otherwise sparse predictor alignment is not safe.
        """
        candidates: list[tuple[str, nn.Module, nn.Module]] = []
        for name, module in self.model.named_modules():
            lm_head = module._modules.get("lm_head")
            if isinstance(lm_head, nn.Module):
                candidates.append((name, module, lm_head))
        unique_heads = {id(lm_head) for _name, _owner, lm_head in candidates}
        if len(unique_heads) != 1 or not candidates:
            paths = [f"{name}.lm_head" if name else "lm_head" for name, *_ in candidates]
            raise RuntimeError(
                "Correct-label sparse logits require exactly one physical "
                f"conditional-generation lm_head, found paths={paths}"
            )
        # ``named_modules`` de-duplicates module instances, so multiple owners
        # of the same physical head would imply non-standard aliasing. Refuse
        # that topology rather than guessing which forward actually calls it.
        if len(candidates) != 1:
            raise RuntimeError(
                "Correct-label sparse logits found ambiguous lm_head owners: "
                f"{[name for name, _owner, _head in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _forward_explicitly_accepts(module: nn.Module, argument: str) -> bool:
        """Return true only for a named forward parameter, never bare kwargs."""
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return False
        return argument in parameters

    def _setup_internal_profiler(self) -> None:
        if not self.config.enable_internal_profiling:
            return
        if self._internal_profiler is not None:
            return

        base_model = getattr(self.model, "model", self.model)
        profiler = _ModuleTimingProfiler(self.device)

        def register_first(root: Any, key: str, candidates: list[str]) -> nn.Module | None:
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

    def _consume_internal_timings(self) -> dict[str, float]:
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
            from triton.compiler.compiler import triton_key
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
        except ImportError as err:
            logger.error("peft not installed. Install with: pip install peft")
            raise ImportError("peft is required for LoRA. Install with: pip install peft") from err

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

    @staticmethod
    def _adapter_parameter_marker(adapter_name: str) -> tuple[str, ...]:
        return tuple(
            f".{family}.{adapter_name}."
            for family in (
                "lora_A",
                "lora_B",
                "lora_embedding_A",
                "lora_embedding_B",
            )
        )

    def available_lora_adapters(self) -> tuple[str, ...]:
        """Return PEFT adapter names with a strict, deterministic order."""
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Qwen must be loaded before inspecting LoRA adapters")
        configs = getattr(self.model, "peft_config", None)
        if not isinstance(configs, dict) or not configs:
            raise RuntimeError("Loaded Qwen model is not a PEFT model with LoRA adapters")
        return tuple(sorted(str(name) for name in configs))

    def active_lora_adapters(self) -> tuple[str, ...]:
        """Return the adapters currently summed by every PEFT LoRA layer."""
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Qwen must be loaded before inspecting active adapters")
        base_model = getattr(self.model, "base_model", None)
        if base_model is None:
            raise RuntimeError("Loaded Qwen PEFT model has no base_model tuner")
        active = getattr(base_model, "active_adapters", None)
        if active is None:
            active = getattr(base_model, "active_adapter", None)
        if isinstance(active, str):
            return (active,)
        if isinstance(active, (list, tuple)):
            return tuple(str(name) for name in active)
        raise RuntimeError(f"Could not inspect active PEFT adapters: {active!r}")

    def activate_lora_adapters(
        self,
        adapter_names: list[str] | tuple[str, ...],
        *,
        trainable_adapters: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Activate an explicit LoRA sum and independently control gradients.

        ``PeftModel.set_adapter`` accepts only one adapter in PEFT 0.19.  The
        underlying tuner supports a list, which makes the frozen navigation
        LoRA and a small STOP-only delta additive.  Gradients are then narrowed
        to the named STOP adapter so the original 224 tensors remain immutable.
        """
        names = tuple(dict.fromkeys(str(name) for name in adapter_names))
        trainable = set(str(name) for name in trainable_adapters)
        if not names:
            raise ValueError("At least one LoRA adapter must remain active")
        available = set(self.available_lora_adapters())
        unknown = sorted(set(names) - available)
        unknown_trainable = sorted(trainable - set(names))
        if unknown or unknown_trainable:
            raise ValueError(
                "Invalid LoRA adapter activation: "
                f"unknown={unknown} trainable_not_active={unknown_trainable} "
                f"available={sorted(available)}"
            )
        base_model = getattr(self.model, "base_model", None)
        set_adapter = getattr(base_model, "set_adapter", None)
        if not callable(set_adapter):
            raise RuntimeError("Loaded PEFT tuner does not support adapter stacking")
        set_adapter(list(names), inference_mode=False)

        markers_by_adapter = {
            name: self._adapter_parameter_marker(name) for name in available
        }
        seen_trainable: set[str] = set()
        for parameter_name, parameter in self.model.named_parameters():
            owner = next(
                (
                    name
                    for name, markers in markers_by_adapter.items()
                    if any(marker in parameter_name for marker in markers)
                ),
                None,
            )
            if owner is None:
                continue
            should_train = owner in trainable
            parameter.requires_grad_(should_train)
            if should_train:
                seen_trainable.add(owner)
        if seen_trainable != trainable:
            raise RuntimeError(
                "Failed to locate every trainable LoRA adapter: "
                f"requested={sorted(trainable)} found={sorted(seen_trainable)}"
            )
        if self.active_lora_adapters() != names:
            raise RuntimeError(
                "PEFT adapter stack did not activate exactly as requested: "
                f"requested={names} active={self.active_lora_adapters()}"
            )

    def add_stop_decision_adapter(
        self,
        *,
        adapter_name: str = STOP_DECISION_ADAPTER_NAME,
        rank: int = 8,
        alpha: int = 16,
        layer_indices: list[int] | tuple[int, ...] = tuple(range(20, 28)),
        target_modules: list[str] | tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
    ) -> int:
        """Add a zero-initialized LoRA delta used only for STOP scoring."""
        if not adapter_name or adapter_name == DEFAULT_LORA_ADAPTER_NAME:
            raise ValueError(f"Invalid STOP-decision adapter name: {adapter_name!r}")
        if rank <= 0 or alpha <= 0:
            raise ValueError("STOP-decision LoRA rank and alpha must be positive")
        layers = sorted({int(index) for index in layer_indices})
        modules = tuple(dict.fromkeys(str(name) for name in target_modules))
        if not layers or not modules:
            raise ValueError("STOP-decision LoRA requires layers and target modules")
        available = set(self.available_lora_adapters())
        if adapter_name in available:
            raise ValueError(f"LoRA adapter already exists: {adapter_name}")
        try:
            from peft import LoraConfig
        except ImportError as err:
            raise ImportError("peft is required for STOP-decision LoRA") from err
        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            target_modules=list(modules),
            layers_to_transform=layers,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        add_adapter = getattr(self.model, "add_adapter", None)
        if not callable(add_adapter):
            raise RuntimeError("Loaded Qwen PEFT model cannot add another adapter")
        add_adapter(adapter_name, config)
        self.activate_lora_adapters(
            (DEFAULT_LORA_ADAPTER_NAME, adapter_name),
            trainable_adapters=(adapter_name,),
        )
        parameter_count = sum(
            parameter.numel()
            for _name, parameter in self.lora_adapter_named_parameters(adapter_name)
        )
        if parameter_count <= 0:
            raise RuntimeError("STOP-decision adapter was added without parameters")
        logger.info(
            "STOP-decision LoRA added: name=%s rank=%d alpha=%d layers=%s "
            "targets=%s params=%d",
            adapter_name,
            rank,
            alpha,
            layers,
            list(modules),
            parameter_count,
        )
        return parameter_count

    def lora_adapter_named_parameters(
        self,
        adapter_name: str,
    ) -> list[tuple[str, nn.Parameter]]:
        markers = self._adapter_parameter_marker(adapter_name)
        parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if any(marker in name for marker in markers)
        ]
        if not parameters:
            raise RuntimeError(f"No LoRA parameters found for adapter {adapter_name!r}")
        return parameters

    @staticmethod
    def _canonical_adapter_parameter_name(name: str, adapter_name: str) -> str:
        for family in (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
        ):
            marker = f".{family}.{adapter_name}."
            if marker in name:
                return name.replace(marker, f".{family}.", 1)
        raise ValueError(
            f"Parameter {name!r} does not belong to adapter {adapter_name!r}"
        )

    def lora_adapter_state_dict(
        self,
        adapter_name: str,
        *,
        cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.lora_adapter_named_parameters(adapter_name):
            canonical = self._canonical_adapter_parameter_name(name, adapter_name)
            value = parameter.detach().clone()
            if cpu:
                value = value.cpu()
            if canonical in state:
                raise RuntimeError(f"Duplicate canonical LoRA key: {canonical}")
            state[canonical] = value
        return dict(sorted(state.items()))

    def load_lora_adapter_state_dict(
        self,
        adapter_name: str,
        state_dict: dict[str, torch.Tensor],
    ) -> int:
        expected = {
            self._canonical_adapter_parameter_name(name, adapter_name): parameter
            for name, parameter in self.lora_adapter_named_parameters(adapter_name)
        }
        if set(state_dict) != set(expected):
            raise RuntimeError(
                "STOP-decision adapter checkpoint key mismatch: "
                f"missing={sorted(set(expected) - set(state_dict))[:5]} "
                f"unexpected={sorted(set(state_dict) - set(expected))[:5]}"
            )
        for name, parameter in expected.items():
            value = state_dict[name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"STOP-decision adapter tensor mismatch for {name}: "
                    f"checkpoint={getattr(value, 'shape', None)} model={tuple(parameter.shape)}"
                )
            if not bool(torch.isfinite(value.float()).all()):
                raise RuntimeError(f"STOP-decision adapter contains non-finite tensor: {name}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        return len(expected)

    def lora_adapter_fingerprint(self, adapter_name: str) -> str:
        digest = hashlib.sha256()
        digest.update(b"heatmapvln-lora-adapter-fp32-v1\0")
        for name, value in self.lora_adapter_state_dict(adapter_name).items():
            tensor = value.float().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def structured_view_token_contract(self) -> dict[str, Any]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen processor has no tokenizer")
        patterns = {
            name: tokenizer.encode(f"view: {name}", add_special_tokens=False)
            for name in STRUCTURED_VIEW_CLASSES
        }
        prefixes = {tuple(pattern[:-1]) for pattern in patterns.values()}
        class_ids = [int(patterns[name][-1]) for name in STRUCTURED_VIEW_CLASSES]
        if (
            any(len(pattern) != 3 for pattern in patterns.values())
            or len(prefixes) != 1
            or len(set(class_ids)) != len(class_ids)
        ):
            raise RuntimeError(
                "STOP-decision adapter requires one shared two-token `view:` "
                f"prefix and six distinct class tokens, got {patterns}"
            )
        return {
            "schema": "heatmapvln-structured-view-token-contract-v1",
            "classes": list(STRUCTURED_VIEW_CLASSES),
            "prefix_token_ids": list(next(iter(prefixes))),
            "class_token_ids": class_ids,
            "patterns": patterns,
        }

    def structured_view_class_logits(
        self,
        sequence_hidden: torch.Tensor,
        predictor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Project only the six structured class tokens, never the vocabulary."""
        if sequence_hidden.ndim != 3:
            raise ValueError(
                f"Expected [B,S,H] sequence hidden states, got {tuple(sequence_hidden.shape)}"
            )
        positions = predictor_positions.to(
            device=sequence_hidden.device,
            dtype=torch.long,
        )
        if positions.shape != (sequence_hidden.shape[0],):
            raise ValueError(
                f"Predictor positions must have shape ({sequence_hidden.shape[0]},), "
                f"got {tuple(positions.shape)}"
            )
        if bool((positions < 0).any()) or bool((positions >= sequence_hidden.shape[1]).any()):
            raise ValueError("Structured view predictor position is out of bounds")
        rows = torch.arange(sequence_hidden.shape[0], device=sequence_hidden.device)
        predictors = sequence_hidden[rows, positions]
        _owner_path, _owner, lm_head = self._locate_conditional_generation_lm_head()
        weight = getattr(lm_head, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise RuntimeError("Physical Qwen lm_head has no rank-2 weight")
        contract = self.structured_view_token_contract()
        token_ids = torch.tensor(
            contract["class_token_ids"],
            device=weight.device,
            dtype=torch.long,
        )
        selected_weight = weight.index_select(0, token_ids)
        bias = getattr(lm_head, "bias", None)
        selected_bias = (
            bias.index_select(0, token_ids) if torch.is_tensor(bias) else None
        )
        logits = F.linear(
            predictors.to(dtype=selected_weight.dtype),
            selected_weight,
            selected_bias,
        )
        if logits.shape != (sequence_hidden.shape[0], len(STRUCTURED_VIEW_CLASSES)):
            raise RuntimeError(f"Unexpected structured view logits shape: {tuple(logits.shape)}")
        return logits.float()

    def merge_lora_for_frozen_forward(self, *, safe_merge: bool = True) -> int:
        """Merge a loaded LoRA adapter into a frozen Qwen backbone.

        Stage3 never updates Qwen or LoRA.  Merging removes the extra low-rank
        matmuls from every attention projection while preserving the exact eval
        function of the loaded adapter.
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Qwen must be loaded before merging LoRA")

        lora_tensors = [
            name for name, _param in self.model.named_parameters()
            if "lora_" in name
        ]
        if not lora_tensors:
            raise RuntimeError(
                "merge_lora_for_frozen_forward requested but the loaded Qwen "
                "model has no LoRA tensors"
            )

        merge_and_unload = getattr(self.model, "merge_and_unload", None)
        if not callable(merge_and_unload):
            raise RuntimeError(
                "Loaded Qwen LoRA model does not support merge_and_unload; "
                "refusing to silently keep the slower unmerged path"
            )

        if self._internal_profiler is not None:
            self._internal_profiler.close()
            self._internal_profiler = None

        merged_model = merge_and_unload(
            progressbar=False,
            safe_merge=safe_merge,
        )
        merged_model.requires_grad_(False)
        merged_model.eval()
        self.model = merged_model
        self.config.use_lora = False

        remaining_lora = [
            name for name, _param in self.model.named_parameters()
            if "lora_" in name
        ]
        if remaining_lora:
            raise RuntimeError(
                "LoRA merge reported success but adapter tensors remain: "
                f"{remaining_lora[:5]}"
            )

        if self.config.enable_internal_profiling:
            self._setup_internal_profiler()
        logger.info(
            "Merged %d frozen LoRA tensors into Qwen for Stage3 forward",
            len(lora_tensors),
        )
        return len(lora_tensors)

    def enable_sequence_packing(self) -> bool:
        """Sequence packing is disabled on the current Qwen2.5-VL stack."""
        logger.warning(
            "Sequence packing is not supported on the current Qwen2.5-VL "
            "training stack. Use standard batching instead."
        )
        return False

    def forward_packed(self, packed_batch: dict[str, Any], **kwargs) -> dict[str, Any]:
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

    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> list[Image.Image]:
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
        instruction: str | None = None,
    ) -> tuple[list[dict], list[Image.Image], Image.Image]:
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
        instruction: Union[str, list[str]] | None = None,
    ) -> list[list[dict]]:
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
    def _views_tensor_to_dict(views: torch.Tensor) -> dict[str, torch.Tensor]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(VIEW_NAMES)}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(
                f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}"
            )
        return [
            self._views_tensor_to_dict(history_panoramas[idx])
            for idx in range(history_panoramas.shape[0])
        ]

    @staticmethod
    def _pad_and_stack(tensors: list[torch.Tensor], pad_dim: int = 1) -> torch.Tensor:
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

    def _model_has_trainable_parameters(self) -> bool:
        if self.model is None:
            return False
        return any(param.requires_grad for param in self.model.parameters())

    @staticmethod
    def _last_hidden_state_from_outputs(outputs) -> torch.Tensor | None:
        last_hs = getattr(outputs, "last_hidden_state", None)
        if last_hs is not None:
            return last_hs
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if torch.is_tensor(first) and first.ndim == 3:
                return first
        return None

    def _forward_model_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        return_hidden_states: bool,
        skip_lm_head: bool = False,
        latent_queries: torch.Tensor | None = None,
        return_lm_loss: bool = False,
        return_lm_correct_logprobs: bool = False,
        structured_class_token_ids: tuple[int, ...] | None = None,
        return_last_hidden_state_only: bool = False,
        extract_vision_hidden_states: bool = True,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        torch.Tensor | None,
        torch.Tensor | dict[str, Any] | None,
    ]:
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
        if return_lm_loss and return_lm_correct_logprobs:
            raise ValueError(
                "return_lm_loss and return_lm_correct_logprobs are mutually exclusive"
            )
        if structured_class_token_ids is not None:
            if not return_lm_correct_logprobs:
                raise ValueError(
                    "structured_class_token_ids requires "
                    "return_lm_correct_logprobs=True"
                )
            if (
                len(structured_class_token_ids) < 2
                or len(set(structured_class_token_ids))
                != len(structured_class_token_ids)
            ):
                raise ValueError(
                    "structured_class_token_ids must contain distinct class tokens"
                )
        if return_last_hidden_state_only and not return_hidden_states:
            raise ValueError(
                "return_last_hidden_state_only requires return_hidden_states=True"
            )
        raw_input_ids = inputs["input_ids"]
        lm_labels = (
            inputs.get("labels")
            if (return_lm_loss or return_lm_correct_logprobs)
            else None
        )
        num_image_tokens = int((raw_input_ids == self.image_token_id).sum().item())

        n_query = 0
        traj_hook_handle = None
        sparse_lm_head_hook_handle = None
        sparse_lm_head = None
        sparse_lm_head_hook = None
        sparse_lm_head_hook_state: dict[str, Any] | None = None
        need_traj_hidden = latent_queries is not None

        # input_ids to use for vision-feature extraction (without TRAJ tokens)
        vision_input_ids = raw_input_ids

        if latent_queries is not None:
            B, n_query, _D = latent_queries.shape
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

            traj_hook_handle = language_model_root.register_forward_pre_hook(
                _replace_traj_embeds_hook, with_kwargs=True,
            )

        inner_model = getattr(self.model, "model", None) if skip_lm_head else None
        bypass_lm_head_owner = None
        bypass_lm_head = None
        if skip_lm_head:
            try:
                (
                    _bypass_owner_path,
                    bypass_lm_head_owner,
                    bypass_lm_head,
                ) = self._locate_conditional_generation_lm_head()
            except RuntimeError:
                # Lightweight test doubles and non-Qwen wrappers can expose a
                # real base model directly. Keep that compatibility fallback.
                bypass_lm_head_owner = None
                bypass_lm_head = None
        use_physical_lm_head_bypass = bool(
            skip_lm_head
            and bypass_lm_head_owner is not None
            and bypass_lm_head is not None
        )
        use_inner_model_for_skip = bool(
            skip_lm_head
            and (use_physical_lm_head_bypass or inner_model is not None)
        )
        use_sequence_last_hidden_state_only = bool(
            return_hidden_states
            and return_last_hidden_state_only
            and use_inner_model_for_skip
        )
        if return_last_hidden_state_only and not use_sequence_last_hidden_state_only:
            raise RuntimeError(
                "Last-hidden-only sequence features require the skip-LM-head inner model path"
            )
        # TRAJ latent queries only need the final sequence state. The optimized
        # frozen path calls the inner model, which exposes last_hidden_state;
        # legacy/wrapper paths retain the all-hidden-states fallback.
        use_last_hidden_state_only = bool(
            need_traj_hidden
            and self.config.traj_last_hidden_state_only
            and use_inner_model_for_skip
        )
        need_all_hidden_states = (
            (return_hidden_states and not use_sequence_last_hidden_state_only)
            or (need_traj_hidden and not use_last_hidden_state_only)
        )
        fwd_kwargs = dict(
            **{k: v for k, v in inputs.items() if k != "labels"},
            output_hidden_states=need_all_hidden_states,
            return_dict=True,
            use_cache=False,
        )
        if return_lm_loss:
            if lm_labels is None:
                raise ValueError("return_lm_loss=True requires `labels` in panoramic_inputs")
            fwd_kwargs["labels"] = lm_labels
            skip_lm_head = False
            inner_model = None
            if need_traj_hidden:
                fwd_kwargs["output_hidden_states"] = True
        correct_logprob_alignment = None
        if return_lm_correct_logprobs:
            if lm_labels is None:
                raise ValueError(
                    "return_lm_correct_logprobs=True requires `labels` in panoramic_inputs"
                )
            if lm_labels.ndim != 2 or lm_labels.shape != raw_input_ids.shape:
                raise ValueError(
                    "Correct-label log-prob labels must match input_ids: "
                    f"labels={tuple(lm_labels.shape)} input_ids={tuple(raw_input_ids.shape)}"
                )
            shifted_valid = lm_labels[:, 1:] != -100
            sample_predictor_positions = [
                torch.nonzero(shifted_valid[row], as_tuple=False).flatten()
                for row in range(shifted_valid.shape[0])
            ]
            if not any(positions.numel() for positions in sample_predictor_positions):
                raise ValueError(
                    "Correct-label log-prob forward has no non-ignored shifted labels"
                )
            sample_correct_token_ids = [
                lm_labels[row, positions + 1]
                for row, positions in enumerate(sample_predictor_positions)
            ]
            predictor_union = torch.unique(
                torch.cat(sample_predictor_positions), sorted=True,
            )
            (
                conditional_generation_path,
                conditional_generation_model,
                sparse_lm_head,
            ) = self._locate_conditional_generation_lm_head()
            native_sparse_logits = self._forward_explicitly_accepts(
                conditional_generation_model,
                "logits_to_keep",
            )
            if native_sparse_logits:
                # New Transformers versions accept a tensor of sequence
                # positions and apply the LM head only to those positions.
                fwd_kwargs["logits_to_keep"] = predictor_union
                sparse_backend = "hf_logits_to_keep_tensor_predictor_union_v1"
            else:
                # Transformers 4.51 Qwen2.5-VL always calls
                # `lm_head(hidden_states)` and exposes no logits_to_keep. Slice
                # the *input* to the physical conditional-generation LM head;
                # index_select remains differentiable, so LoRA gradients are
                # preserved without ever materialising [B, S, vocab] logits.
                sparse_backend = "lm_head_pre_hook_predictor_union_v1"
                sparse_lm_head_hook_state = {
                    "call_count": 0,
                    "input_shape_before": None,
                    "input_shape_after": None,
                    "removed": False,
                }

                def _slice_lm_head_input_hook(module, args, kwargs):
                    del module
                    assert sparse_lm_head_hook_state is not None
                    sparse_lm_head_hook_state["call_count"] += 1
                    if sparse_lm_head_hook_state["call_count"] != 1:
                        raise RuntimeError(
                            "Conditional-generation lm_head was called more than "
                            "once during one sparse correct-label forward"
                        )
                    if args:
                        hidden_states = args[0]
                        input_location = "args"
                    elif "input" in kwargs:
                        hidden_states = kwargs["input"]
                        input_location = "kwargs"
                    else:
                        raise RuntimeError(
                            "Could not locate hidden_states input to conditional-generation lm_head"
                        )
                    if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                        raise RuntimeError(
                            "Conditional-generation lm_head input must be [B,S,H], "
                            f"got {type(hidden_states).__name__} "
                            f"shape={getattr(hidden_states, 'shape', None)}"
                        )
                    if (
                        hidden_states.shape[0] != raw_input_ids.shape[0]
                        or hidden_states.shape[1] != raw_input_ids.shape[1]
                    ):
                        raise RuntimeError(
                            "Conditional-generation lm_head input is not aligned "
                            "with tokenized SFT inputs: "
                            f"hidden={tuple(hidden_states.shape)} "
                            f"input_ids={tuple(raw_input_ids.shape)}"
                        )
                    selected = hidden_states.index_select(
                        1,
                        predictor_union.to(device=hidden_states.device),
                    )
                    sparse_lm_head_hook_state["input_shape_before"] = list(
                        hidden_states.shape
                    )
                    sparse_lm_head_hook_state["input_shape_after"] = list(
                        selected.shape
                    )
                    if input_location == "args":
                        return (selected, *args[1:]), kwargs
                    new_kwargs = dict(kwargs)
                    new_kwargs["input"] = selected
                    return args, new_kwargs

                sparse_lm_head_hook = _slice_lm_head_input_hook
            skip_lm_head = False
            inner_model = None
            correct_logprob_alignment = {
                "schema": "shifted_correct_label_predictors_v1",
                "ignore_index": -100,
                "batch_size": int(lm_labels.shape[0]),
                "sequence_length": int(lm_labels.shape[1]),
                "sample_predictor_positions": [
                    positions.detach().cpu().tolist()
                    for positions in sample_predictor_positions
                ],
                "sample_correct_token_ids": [
                    token_ids.detach().cpu().tolist()
                    for token_ids in sample_correct_token_ids
                ],
                "predictor_position_union": predictor_union.detach().cpu().tolist(),
                "sample_label_tokens": [
                    int(positions.numel())
                    for positions in sample_predictor_positions
                ],
                "label_tokens": int(shifted_valid.sum().item()),
                "backend": sparse_backend,
                "conditional_generation_module": conditional_generation_path,
                "lm_head_module": (
                    f"{conditional_generation_path}.lm_head"
                    if conditional_generation_path
                    else "lm_head"
                ),
                "native_logits_to_keep_explicit_signature": native_sparse_logits,
            }
        if self._internal_profiler is not None:
            self._internal_profiler.reset()

        qwen_needs_grad = (
            return_lm_loss
            or return_lm_correct_logprobs
            or self._model_has_trainable_parameters()
            or (latent_queries is not None and latent_queries.requires_grad)
        )

        def _filter_kwargs_for_forward(module: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
            try:
                signature = inspect.signature(module.forward)
            except (TypeError, ValueError):
                return kwargs
            parameters = signature.parameters
            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
                return kwargs
            return {key: value for key, value in kwargs.items() if key in parameters}

        def _run_model_forward():
            if skip_lm_head:
                if use_physical_lm_head_bypass:
                    assert bypass_lm_head_owner is not None
                    assert bypass_lm_head is not None
                    # PEFT wraps the multimodal conditional-generation model;
                    # calling ``PeftModel.model`` does not reach the Qwen base
                    # model and therefore cannot expose last_hidden_state. Run
                    # the normal PEFT path so all LoRA modules and vision input
                    # handling remain identical, but replace the one physical
                    # vocabulary projection with Identity for this forward.
                    with self._lm_head_bypass_lock:
                        registered_head = bypass_lm_head_owner._modules.get("lm_head")
                        if registered_head is not bypass_lm_head:
                            raise RuntimeError(
                                "Physical Qwen lm_head changed before frozen-forward bypass"
                            )
                        identity = nn.Identity()
                        bypass_lm_head_owner._modules["lm_head"] = identity
                        try:
                            return self.model(**fwd_kwargs)
                        finally:
                            bypass_lm_head_owner._modules["lm_head"] = bypass_lm_head
                if inner_model is None:
                    return self.model(**fwd_kwargs)
                try:
                    return inner_model(**fwd_kwargs)
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    if need_traj_hidden and not fwd_kwargs.get("output_hidden_states", False):
                        fwd_kwargs["output_hidden_states"] = True
                    filtered_kwargs = _filter_kwargs_for_forward(inner_model, fwd_kwargs)
                    if filtered_kwargs.keys() == fwd_kwargs.keys():
                        raise
                    return inner_model(**filtered_kwargs)
            return self.model(**fwd_kwargs)

        use_inference_context = bool(
            not qwen_needs_grad
            and not (
                need_traj_hidden
                and not self.config.frozen_traj_inference_mode
            )
        )
        outputs_are_inference_tensors = use_inference_context

        def _run_with_runtime_context():
            if qwen_needs_grad:
                return _run_model_forward()
            if use_inference_context:
                with torch.inference_mode():
                    return _run_model_forward()
            with torch.no_grad():
                return _run_model_forward()

        if sparse_lm_head_hook is not None:
            assert sparse_lm_head is not None
            sparse_lm_head_hook_handle = sparse_lm_head.register_forward_pre_hook(
                sparse_lm_head_hook,
                with_kwargs=True,
            )
        try:
            try:
                outputs = _run_with_runtime_context()
            except TypeError as exc:
                if (
                    return_lm_correct_logprobs
                    and correct_logprob_alignment is not None
                    and correct_logprob_alignment["backend"]
                    == "hf_logits_to_keep_tensor_predictor_union_v1"
                    and "logits_to_keep" in str(exc)
                ):
                    raise RuntimeError(
                        "The loaded Qwen wrapper does not support tensor "
                        "`logits_to_keep`; refusing an implicit full-logits "
                        "fallback for correct-label preservation"
                    ) from exc
                raise
            bypassed_sequence_hidden = None
            if use_physical_lm_head_bypass:
                bypassed_sequence_hidden = getattr(outputs, "logits", None)
                expected_hidden_dim = getattr(bypass_lm_head, "in_features", None)
                bypass_weight = getattr(bypass_lm_head, "weight", None)
                if expected_hidden_dim is None and torch.is_tensor(bypass_weight):
                    expected_hidden_dim = int(bypass_weight.shape[-1])
                if (
                    not torch.is_tensor(bypassed_sequence_hidden)
                    or bypassed_sequence_hidden.ndim != 3
                    or (
                        expected_hidden_dim is not None
                        and bypassed_sequence_hidden.shape[-1] != expected_hidden_dim
                    )
                ):
                    raise RuntimeError(
                        "Physical Qwen lm_head bypass did not return sequence hidden states: "
                        f"shape={getattr(bypassed_sequence_hidden, 'shape', None)} "
                        f"expected_hidden_dim={expected_hidden_dim}"
                    )
            if (
                use_last_hidden_state_only
                and bypassed_sequence_hidden is None
                and self._last_hidden_state_from_outputs(outputs) is None
            ):
                logger.warning(
                    "Qwen output %s from %s did not expose last_hidden_state; "
                    "disabling traj_last_hidden_state_only and retrying with "
                    "output_hidden_states=true",
                    type(outputs).__name__,
                    type(inner_model).__name__ if inner_model is not None else "None",
                )
                self.config.traj_last_hidden_state_only = False
                fwd_kwargs["output_hidden_states"] = True
                outputs = _run_with_runtime_context()
        finally:
            cleanup_errors = []
            if sparse_lm_head_hook_handle is not None:
                hook_id = sparse_lm_head_hook_handle.id
                try:
                    sparse_lm_head_hook_handle.remove()
                except Exception as exc:  # pragma: no cover - PyTorch handle failure
                    cleanup_errors.append(f"lm_head hook remove failed: {exc!r}")
                if (
                    sparse_lm_head is not None
                    and hook_id in sparse_lm_head._forward_pre_hooks
                ):
                    cleanup_errors.append("lm_head hook remains registered after remove")
                if sparse_lm_head_hook_state is not None:
                    sparse_lm_head_hook_state["removed"] = not cleanup_errors
            if traj_hook_handle is not None:
                try:
                    traj_hook_handle.remove()
                except Exception as exc:  # pragma: no cover - PyTorch handle failure
                    cleanup_errors.append(f"trajectory hook remove failed: {exc!r}")
            if cleanup_errors:
                raise RuntimeError(
                    "Failed to clean temporary Qwen forward hooks: "
                    + "; ".join(cleanup_errors)
                )

        if self._internal_profiler is not None:
            self._last_internal_timings = self._internal_profiler.snapshot()
        else:
            self._last_internal_timings = {}

        traj_hidden_states = None
        hidden_states = None
        lm_output: torch.Tensor | dict[str, Any] | None = (
            getattr(outputs, "loss", None) if return_lm_loss else None
        )
        if return_lm_correct_logprobs:
            logits = getattr(outputs, "logits", None)
            if logits is None or logits.ndim != 3:
                raise RuntimeError(
                    "Correct-label log-prob forward requires rank-3 `outputs.logits`"
                )
            assert correct_logprob_alignment is not None
            if sparse_lm_head_hook_state is not None:
                if sparse_lm_head_hook_state["call_count"] != 1:
                    raise RuntimeError(
                        "Sparse lm_head pre-hook did not execute exactly once: "
                        f"{sparse_lm_head_hook_state}"
                    )
                if not sparse_lm_head_hook_state["removed"]:
                    raise RuntimeError("Sparse lm_head pre-hook was not removed")
                correct_logprob_alignment.update(
                    {
                        "lm_head_hook_call_count": sparse_lm_head_hook_state[
                            "call_count"
                        ],
                        "lm_head_input_shape_before": sparse_lm_head_hook_state[
                            "input_shape_before"
                        ],
                        "lm_head_input_shape_after": sparse_lm_head_hook_state[
                            "input_shape_after"
                        ],
                        "lm_head_hook_removed": sparse_lm_head_hook_state[
                            "removed"
                        ],
                    }
                )
            predictor_union = torch.tensor(
                correct_logprob_alignment["predictor_position_union"],
                device=logits.device,
                dtype=torch.long,
            )
            if (
                logits.shape[0] != lm_labels.shape[0]
                or logits.shape[1] != predictor_union.numel()
            ):
                raise RuntimeError(
                    "Qwen sparse LM-head backend returned an unexpected shape: "
                    f"logits={tuple(logits.shape)} expected_batch={lm_labels.shape[0]} "
                    f"expected_kept_positions={predictor_union.numel()}"
                )
            flat_correct_logprobs = []
            flat_correct_rejection_log_odds = []
            sample_structured_class_logits = []
            sample_structured_class_targets = []
            structured_ids = (
                torch.tensor(
                    structured_class_token_ids,
                    device=logits.device,
                    dtype=torch.long,
                )
                if structured_class_token_ids is not None
                else None
            )
            for row, (positions_list, token_ids_list) in enumerate(
                zip(
                    correct_logprob_alignment["sample_predictor_positions"],
                    correct_logprob_alignment["sample_correct_token_ids"],
                )
            ):
                positions = torch.tensor(
                    positions_list, device=logits.device, dtype=torch.long,
                )
                token_ids = torch.tensor(
                    token_ids_list, device=logits.device, dtype=torch.long,
                )
                kept_columns = torch.searchsorted(predictor_union, positions)
                selected_logits = logits[row, kept_columns].float()
                correct_logits = selected_logits.gather(
                    dim=-1, index=token_ids.unsqueeze(-1),
                ).squeeze(-1)
                competing_logits = selected_logits.clone()
                competing_logits.scatter_(
                    dim=-1,
                    index=token_ids.unsqueeze(-1),
                    value=float("-inf"),
                )
                flat_correct_logprobs.append(
                    correct_logits - torch.logsumexp(selected_logits, dim=-1)
                )
                flat_correct_rejection_log_odds.append(
                    correct_logits - torch.logsumexp(competing_logits, dim=-1)
                )
                if structured_ids is not None:
                    matches = token_ids.unsqueeze(-1) == structured_ids.unsqueeze(0)
                    occurrences = torch.nonzero(matches, as_tuple=False)
                    if occurrences.shape != (1, 2):
                        raise RuntimeError(
                            "Each structured SFT row must expose exactly one class "
                            "token, got "
                            f"row={row} token_ids={token_ids_list} "
                            f"class_ids={list(structured_class_token_ids)}"
                        )
                    labelled_position = int(occurrences[0, 0].item())
                    class_target = int(occurrences[0, 1].item())
                    sample_structured_class_logits.append(
                        selected_logits[labelled_position]
                        .index_select(0, structured_ids)
                        .float()
                    )
                    sample_structured_class_targets.append(class_target)
            correct_logprobs = torch.cat(flat_correct_logprobs, dim=0)
            correct_rejection_log_odds = torch.cat(
                flat_correct_rejection_log_odds,
                dim=0,
            )
            if correct_logprobs.dtype != torch.float32:
                raise RuntimeError(
                    "Correct-label log probabilities must be accumulated in FP32"
                )
            if correct_rejection_log_odds.dtype != torch.float32:
                raise RuntimeError(
                    "Correct-label rejection log odds must be accumulated in FP32"
                )
            correct_logprob_alignment["returned_logits_shape"] = list(logits.shape)
            correct_logprob_alignment["returned_logprob_dtype"] = str(
                correct_logprobs.dtype
            )
            correct_logprob_alignment["returned_rejection_log_odds_dtype"] = str(
                correct_rejection_log_odds.dtype
            )
            lm_output = {
                "correct_label_logprobs": correct_logprobs,
                "correct_label_rejection_log_odds": correct_rejection_log_odds,
                "alignment": correct_logprob_alignment,
            }
            if structured_ids is not None:
                structured_logits = torch.stack(
                    sample_structured_class_logits,
                    dim=0,
                )
                if structured_logits.dtype != torch.float32:
                    raise RuntimeError(
                        "Structured class logits must be accumulated in FP32"
                    )
                lm_output["structured_class_logits"] = structured_logits
                lm_output["structured_class_targets"] = (
                    sample_structured_class_targets
                )

        if return_hidden_states:
            if use_sequence_last_hidden_state_only:
                hidden_states = bypassed_sequence_hidden
                if hidden_states is None:
                    hidden_states = self._last_hidden_state_from_outputs(outputs)
                if hidden_states is None:
                    raise RuntimeError(
                        "Inner Qwen model did not expose last_hidden_state for STOP features"
                    )
            else:
                layer_idx = self.config.hidden_layer_for_features
                if layer_idx == -1:
                    layer_idx = len(outputs.hidden_states) - 1
                hidden_states = outputs.hidden_states[layer_idx]

        if need_traj_hidden:
            last_hs = bypassed_sequence_hidden
            if last_hs is None:
                last_hs = self._last_hidden_state_from_outputs(outputs)
            if last_hs is None:
                raise RuntimeError(
                    "Failed to extract last hidden state for TRAJ latent queries. "
                    "Use an inner/base model output or enable output_hidden_states."
                )
            traj_hidden_states = last_hs[:, -n_query:, :].contiguous()
            if outputs_are_inference_tensors:
                # A normal clone can be saved by the trainable adapter's
                # backward pass; inference tensors cannot.
                traj_hidden_states = traj_hidden_states.clone()
            if hidden_states is not None:
                hidden_states = hidden_states[:, :-n_query, :].contiguous()

        if not return_hidden_states:
            hidden_states = None

        vision_hidden_states = None
        if hidden_states is not None and extract_vision_hidden_states:
            vision_hidden_states = self._extract_vision_hidden_states(
                hidden_states, vision_input_ids,
            )

        return hidden_states, vision_hidden_states, num_image_tokens, traj_hidden_states, lm_output

    def _forward_single_panorama(
        self,
        current_views: torch.Tensor,
        history_panoramas: torch.Tensor,
        instruction: str | None = None,
        return_hidden_states: bool = True,
        heatmap_vln: nn.Module | None = None,
        history_rel_poses: torch.Tensor | None = None,
        return_heatmap_memory_tokens: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, dict[str, torch.Tensor] | None]:
        """Forward a single panoramic sample through one Qwen pass.

        When ``return_hidden_states`` is False (heatmap-only training), the
        Qwen forward is wrapped in ``torch.inference_mode()`` to avoid storing
        intermediate activations for the frozen backbone, saving ~4-8 GB VRAM.

        When ``heatmap_trains_backbone`` is True, the inference_mode wrapper is
        skipped so that heatmap loss gradients can flow back through the backbone.
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

        need_grad = return_hidden_states or self.config.heatmap_trains_backbone
        if need_grad:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs, _lm_loss = self._forward_model_inputs(
                inputs, return_hidden_states,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, _traj_hs, _lm_loss = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True,
                )

        heatmap_decode_kwargs = {"history_rel_poses": history_rel_poses}
        if return_heatmap_memory_tokens:
            heatmap_decode_kwargs["return_memory_tokens"] = True
        heatmap_output = heatmap_vln.decode_from_inputs(
            inputs,
            num_history,
            **heatmap_decode_kwargs,
        )
        return hidden_states, vision_hidden_states, num_image_tokens, heatmap_output

    def _forward_batch_panorama(
        self,
        current_views: torch.Tensor,
        history_panoramas: torch.Tensor,
        instruction: Union[str, list[str]] | None = None,
        return_hidden_states: bool = True,
        heatmap_vln: nn.Module | None = None,
        history_rel_poses: torch.Tensor | None = None,
        return_heatmap_memory_tokens: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, dict[str, torch.Tensor] | None]:
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

        need_grad = return_hidden_states or self.config.heatmap_trains_backbone
        if need_grad:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs, _lm_loss = self._forward_model_inputs(
                inputs, return_hidden_states,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, traj_hs, _lm_loss = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True,
                )
        t2 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        internal_timings = self._consume_internal_timings() if self.config.enable_runtime_timing else {}

        heatmap_decode_kwargs = {
            "image_positions_batch": image_positions_batch,
            "text_anchors_batch": text_anchors_batch,
            "history_rel_poses": history_rel_poses,
        }
        if return_heatmap_memory_tokens:
            heatmap_decode_kwargs["return_memory_tokens"] = True
        heatmap_output = heatmap_vln.decode_from_inputs_batch(
            inputs,
            num_histories,
            **heatmap_decode_kwargs,
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
        panoramic_inputs: dict[str, torch.Tensor],
        num_histories: list[int],
        text_anchor_positions_batch: list[dict[int, int]] | None = None,
        return_hidden_states: bool = True,
        heatmap_vln: nn.Module | None = None,
        history_rel_poses: torch.Tensor | None = None,
        latent_queries: torch.Tensor | None = None,
        return_lm_loss: bool = False,
        return_lm_correct_logprobs: bool = False,
        sequence_last_hidden_state_only: bool = False,
        return_heatmap_memory_tokens: bool = False,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | dict[str, Any] | None,
    ]:
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

        need_grad = (
            return_lm_loss
            or return_lm_correct_logprobs
            or return_hidden_states
            or latent_queries is not None
            or (heatmap_vln is not None and self.config.heatmap_trains_backbone)
        )
        skip_lm = (
            (heatmap_vln is None)
            and not return_lm_loss
            and not return_lm_correct_logprobs
        )
        if need_grad:
            hidden_states, vision_hidden_states, num_image_tokens, traj_hs, lm_output = self._forward_model_inputs(
                inputs, return_hidden_states,
                skip_lm_head=skip_lm, latent_queries=latent_queries,
                return_lm_loss=return_lm_loss,
                return_lm_correct_logprobs=return_lm_correct_logprobs,
                return_last_hidden_state_only=sequence_last_hidden_state_only,
                extract_vision_hidden_states=not sequence_last_hidden_state_only,
            )
        else:
            with torch.inference_mode():
                hidden_states, vision_hidden_states, num_image_tokens, traj_hs, lm_output = self._forward_model_inputs(
                    inputs, False, skip_lm_head=True, latent_queries=latent_queries,
                )
        t2 = time.perf_counter() if self.config.enable_runtime_timing else 0.0
        internal_timings = self._consume_internal_timings() if self.config.enable_runtime_timing else {}

        heatmap_output = None
        if heatmap_vln is not None:
            heatmap_decode_kwargs = {
                "image_positions_batch": image_positions_batch,
                "text_anchors_batch": text_anchor_positions_batch,
                "history_rel_poses": history_rel_poses,
            }
            if return_heatmap_memory_tokens:
                heatmap_decode_kwargs["return_memory_tokens"] = True
            heatmap_output = heatmap_vln.decode_from_inputs_batch(
                inputs,
                num_histories,
                **heatmap_decode_kwargs,
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
        return hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hs, lm_output

    def _forward_batch(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Union[str, list[str]] | None = None,
        return_hidden_states: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, None]:
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
        instruction: str | None = None,
        return_hidden_states: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, None]:
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

        hidden_states, vision_hidden_states, num_image_tokens, _traj_hs, _lm_loss = self._forward_model_inputs(
            inputs, return_hidden_states,
        )
        return hidden_states, vision_hidden_states, num_image_tokens, None

    def forward(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: Union[str, list[str]] | None = None,
        return_hidden_states: bool = True,
        generate_text: bool = False,
        current_views: torch.Tensor | None = None,
        history_panoramas: torch.Tensor | None = None,
        panoramic_inputs: dict[str, torch.Tensor] | None = None,
        panoramic_num_histories: list[int] | None = None,
        panoramic_text_anchor_positions: list[dict[int, int]] | None = None,
        heatmap_vln: nn.Module | None = None,
        history_rel_poses: torch.Tensor | None = None,
        latent_queries: torch.Tensor | None = None,
        return_lm_loss: bool = False,
        return_lm_correct_logprobs: bool = False,
        sequence_last_hidden_state_only: bool = False,
        return_heatmap_memory_tokens: bool = False,
    ) -> dict[str, Any]:
        """Forward pass through Qwen2.5-VL with batch processing."""
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()

        traj_hidden_states = None
        lm_output: torch.Tensor | dict[str, Any] | None = None

        if panoramic_inputs is not None:
            if panoramic_num_histories is None:
                raise ValueError("panoramic_num_histories is required with panoramic_inputs")
            hidden_states, vision_hidden_states, num_image_tokens, heatmap_output, traj_hidden_states, lm_output = (
                self._forward_batch_panorama_tokenized(
                    panoramic_inputs=panoramic_inputs,
                    num_histories=panoramic_num_histories,
                    text_anchor_positions_batch=panoramic_text_anchor_positions,
                    return_hidden_states=return_hidden_states,
                    heatmap_vln=heatmap_vln,
                    history_rel_poses=history_rel_poses,
                    latent_queries=latent_queries,
                    return_lm_loss=return_lm_loss,
                    return_lm_correct_logprobs=return_lm_correct_logprobs,
                    sequence_last_hidden_state_only=sequence_last_hidden_state_only,
                    return_heatmap_memory_tokens=return_heatmap_memory_tokens,
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
                    return_heatmap_memory_tokens=return_heatmap_memory_tokens,
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
        if return_lm_loss and lm_output is not None:
            result["lm_loss"] = lm_output
        if return_lm_correct_logprobs:
            if not isinstance(lm_output, dict):
                raise RuntimeError(
                    "Correct-label log-prob forward returned no structured LM output"
                )
            result["lm_correct_label_logprobs"] = lm_output[
                "correct_label_logprobs"
            ]
            result["lm_correct_label_alignment"] = lm_output["alignment"]
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
        instruction: str | None = None,
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

    @staticmethod
    def _normalize_multimodal_forward_inputs(inputs: dict[str, torch.Tensor]) -> None:
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            vgt = inputs["video_grid_thw"]
            if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                inputs["video_grid_thw"] = torch.repeat_interleave(
                    vgt, vgt[:, 0], dim=0,
                )
                inputs["video_grid_thw"][:, 0] = 1

    def extract_traj_hidden_states(
        self,
        output_ids: torch.Tensor,
        latent_queries: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract trajectory conditions via the training-aligned forward path.

        Appends ``TRAJ_TOKEN_INDEX`` placeholders when absent, then runs
        ``_forward_model_inputs`` with ``latent_queries`` injection (same as
        ``PanoramicTokenizedCollator`` + Stage2 training).
        """
        if not self._model_loaded:
            self._load_model()

        n_query = latent_queries.shape[1]
        batch_size = output_ids.shape[0]
        device = self.device
        ids = output_ids.to(device)

        # ``output_ids`` from ``model.generate(...)`` includes both the prompt
        # and the autoregressively generated text tokens, while ``attention_mask``
        # and ``mm_token_type_ids`` provided by the caller usually correspond to
        # the *prompt only* (the original ``apply_chat_template`` output).  Pad
        # them up to the pre-suffix length of ``ids`` so the eventual
        # ``get_rope_index`` call sees matching shapes; generated tokens are
        # real text tokens (mask=1, mm_token_type=0).
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            gap = ids.shape[1] - attention_mask.shape[1]
            if gap > 0:
                fill = torch.ones(
                    batch_size, gap,
                    device=device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, fill], dim=1)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)
            gap_mm = ids.shape[1] - mm_token_type_ids.shape[1]
            if gap_mm > 0:
                fill_mm = torch.zeros(
                    batch_size, gap_mm,
                    device=device,
                    dtype=mm_token_type_ids.dtype,
                )
                mm_token_type_ids = torch.cat([mm_token_type_ids, fill_mm], dim=1)

        has_traj_tokens = (ids == TRAJ_TOKEN_INDEX).any().item()
        if not has_traj_tokens:
            traj_suffix = torch.full(
                (batch_size, n_query),
                TRAJ_TOKEN_INDEX,
                device=device,
                dtype=ids.dtype,
            )
            ids = torch.cat([ids, traj_suffix], dim=1)
            if attention_mask is not None:
                ext = torch.ones(
                    batch_size, n_query,
                    device=device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, ext], dim=1)
            if mm_token_type_ids is not None:
                mm_ext = torch.zeros(
                    batch_size, n_query,
                    device=device,
                    dtype=mm_token_type_ids.dtype,
                )
                mm_token_type_ids = torch.cat([mm_token_type_ids, mm_ext], dim=1)

        inputs: dict[str, torch.Tensor] = {"input_ids": ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if mm_token_type_ids is not None:
            inputs["mm_token_type_ids"] = mm_token_type_ids
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(device)
        if image_grid_thw is not None:
            inputs["image_grid_thw"] = image_grid_thw.to(device)

        self._normalize_multimodal_forward_inputs(inputs)

        lq = latent_queries.to(device=device, dtype=self.config.get_torch_dtype())
        with torch.no_grad():
            _hidden, _vision, _n_img, traj_hidden_states, _lm_loss = self._forward_model_inputs(
                inputs,
                return_hidden_states=False,
                latent_queries=lq,
            )

        if traj_hidden_states is None:
            raise RuntimeError("extract_traj_hidden_states: forward returned no traj_hidden_states")
        return traj_hidden_states.contiguous()

    def generate_latents(
        self,
        output_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        latent_queries: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract traj_hidden_states (delegates to training-aligned forward)."""
        return self.extract_traj_hidden_states(
            output_ids=output_ids,
            latent_queries=latent_queries,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

    def _extract_hidden_from_generation(
        self,
        hidden_states_tuple: tuple,
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
