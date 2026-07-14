"""
VLN Pipeline — v2 (Coarse-to-Fine) + InternNav System 1
=========================================================

Architecture:
    Current panorama (4 views) + N history panoramas + text
        |
    Qwen2.5-VL backbone (frozen + LoRA)
        |
    ├── HeatmapVLN (Coarse-to-Fine, ~2M trainable)
    │       → visibility (N_hist, 4)
    │       → heatmaps   (N_hist, 4, 64, 64)
    │
    └── NextDiT System 1 (InternNav action head)
            → trajectory (B, T, 3)
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .heatmap import HeatmapVLN
from .qwen2_5_vl import Qwen2_5VLConfig, Qwen2_5VLIntegration

logger = logging.getLogger(__name__)
VIEW_NAMES = ("front", "right", "back", "left")

INTERNNAV_SYSTEM1_PREFIXES = (
    "model.latent_queries",
    "model.cond_projector.",
    "model.traj_dit.",
    "model.memory_encoder.",
    "model.rgb_model.",
    "model.rgb_resampler.",
    "model.action_encoder.",
    "model.action_decoder.",
)


def _is_internnav_system1_key(key: str) -> bool:
    return any(key.startswith(prefix) or key == prefix.rstrip(".") for prefix in INTERNNAV_SYSTEM1_PREFIXES)


def _remap_internnav_system1_key(key: str) -> str:
    return key[len("model.") :] if key.startswith("model.") else key


@dataclass
class VLNPipelineConfig:
    """Configuration for VLN pipeline."""

    # Backbone configuration
    llm_model_path: str = "./models/internnav_backbone"
    llm_backbone_type: str = "qwen2_5_vl"
    llm_hidden_dim: int = 3584
    llm_token_dim: int = 896
    llm_torch_dtype: str = "bfloat16"
    llm_attn_implementation: str = "sdpa"
    max_video_frames: int = 16
    llm_gradient_checkpointing: bool = False
    llm_enable_internal_profiling: bool = False
    enable_runtime_timing: bool = False
    llm_enable_compile: bool = False
    llm_compile_mode: str = "reduce-overhead"
    llm_compile_backend: str = "inductor"
    llm_frozen_traj_inference_mode: bool = False
    llm_traj_last_hidden_state_only: bool = False

    # Sequence packing configuration (currently disabled on the shared stack)
    enable_packing: bool = False
    max_seq_length: int = 4096
    spatial_merge_size: int = 2

    # InternNav System 1 weights (pre-trained NextDiT + latent_queries)
    internnav_system1_path: str = ""
    internnav_model_path: str = ""

    # Device configuration
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # LoRA configuration for the VLM backbone fine-tuning
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_num_layers: int = 4
    lora_layer_indices: list[int] | None = None
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # ==================== HeatmapVLN v2 (Coarse-to-Fine) ====================
    enable_heatmap: bool = True
    heatmap_c_vit: int = 1280
    heatmap_c_llm: int = 3584
    heatmap_c_fused: int = 256
    heatmap_vit_layer_indices: list[int] | None = None  # e.g. [6, 12, 18, 24]
    heatmap_llm_layer_indices: list[int] | None = None  # e.g. [7, 15, 23] (full_attention)
    heatmap_size: tuple[int, int] = (64, 64)
    heatmap_trajectory_config: dict[str, Any] | None = None
    heatmap_decoder_mode: str = "legacy"
    heatmap_pose_free_config: dict[str, Any] | None = None

    # Allow heatmap loss gradients to flow back through the VLM backbone (LoRA).
    # When False (default), hooked features are detached and the backbone runs
    # in inference_mode during heatmap-only training — zero VRAM overhead from
    # backbone activations.  When True, the computation graph is preserved so
    # that LoRA parameters receive gradients from the heatmap loss (+4-8 GB).
    heatmap_trains_backbone: bool = False

    # HeatmapVLNLoss weights
    heatmap_lambda_vis: float = 1.0
    heatmap_lambda_coord: float = 1.0
    heatmap_lambda_kl: float = 1.0
    heatmap_lambda_peak: float = 1.0

    # ==================== Action Head (NextDiT System 1) ====================
    enable_action_head: bool = True
    nextdit_enabled: bool = False
    nextdit_vlm_hidden_dim: int = 3584
    nextdit_latent_emb_size: int = 768
    nextdit_n_query: int = 4
    nextdit_dit_dim: int = 384
    nextdit_dit_layers: int = 12
    nextdit_dit_heads: int = 6
    nextdit_dit_kv_heads: int = 6
    nextdit_dit_ffn_dim_multiplier: float | None = 2 / 3
    nextdit_predict_steps: int = 32
    nextdit_action_dim: int = 3
    nextdit_num_inference_steps: int = 10
    nextdit_guidance_scale: float = 1.0
    nextdit_num_sample_trajs: int = 32
    nextdit_dav2_ckpt_path: str = ""
    nextdit_enable_gradient_checkpointing: bool = True

    # Optional pano-student -> InternNav-latent adapter.  This sits before the
    # NextDiT cond_projector and preserves the original System1 visual path.
    pano_latent_adapter_enabled: bool = False
    pano_latent_adapter_hidden_dim: int = 1024
    pano_latent_adapter_dropout: float = 0.0
    pano_latent_adapter_checkpoint_path: str = ""
    pano_latent_adapter_strict_load: bool = True

    # Performance settings
    enable_gradient_checkpointing: bool = False
    verbose: bool = False

    # Image size
    image_size: int = 256


class VLNPipeline(nn.Module):
    """
    VLN Pipeline with Qwen2.5-VL and HeatmapVLN v2 (Coarse-to-Fine).
    """

    def __init__(self, config: VLNPipelineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        logger.info("=" * 60)
        logger.info("Initializing VLN Pipeline (Qwen2.5-VL, HeatmapVLN v2 Coarse-to-Fine)")
        logger.info("=" * 60)

        # ==================== Backbone ====================
        qwen_config = Qwen2_5VLConfig(
            model_path=config.llm_model_path,
            device=config.device,
            torch_dtype=config.llm_torch_dtype,
            attn_implementation=config.llm_attn_implementation,
            max_video_frames=config.max_video_frames,
            gradient_checkpointing=config.llm_gradient_checkpointing,
            enable_internal_profiling=config.llm_enable_internal_profiling,
            enable_runtime_timing=config.enable_runtime_timing,
            enable_compile=config.llm_enable_compile,
            compile_mode=config.llm_compile_mode,
            compile_backend=config.llm_compile_backend,
            frozen_traj_inference_mode=config.llm_frozen_traj_inference_mode,
            traj_last_hidden_state_only=config.llm_traj_last_hidden_state_only,
            enable_packing=config.enable_packing,
            max_seq_length=config.max_seq_length,
            spatial_merge_size=config.spatial_merge_size,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_num_layers=config.lora_num_layers,
            lora_layer_indices=config.lora_layer_indices,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            heatmap_trains_backbone=config.heatmap_trains_backbone,
        )
        self.qwen2_5_vl = Qwen2_5VLIntegration(qwen_config)
        self.vlm_backbone = self.qwen2_5_vl
        logger.info("Qwen2.5-VL integration initialized")

        # ==================== LLM Projector ====================
        self.llm_projector = nn.Sequential(
            nn.LayerNorm(config.llm_hidden_dim),
            nn.Linear(config.llm_hidden_dim, config.llm_token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.llm_token_dim, config.llm_token_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.LayerNorm(config.llm_token_dim),
        ).to(device=self.device, dtype=config.dtype)
        logger.info(
            "LLM projector: %d -> %d",
            config.llm_hidden_dim,
            config.llm_token_dim,
        )

        # ==================== HeatmapVLN v2 ====================
        # HeatmapVLN is constructed lazily after the VLM backbone is loaded,
        # because it needs the actual model instance for hook registration.
        self.heatmap_vln: HeatmapVLN | None = None
        self._heatmap_enabled = config.enable_heatmap

        # ==================== Action Head (NextDiT System 1) ====================
        self.nextdit_action_head = None
        self.latent_queries = None
        self.pano_latent_adapter = None
        self._internnav_system1_load_audit: dict[str, Any] | None = None

        if config.enable_action_head and config.nextdit_enabled:
            from .action import NextDiTActionConfig, NextDiTActionHead

            nextdit_cfg = NextDiTActionConfig(
                vlm_hidden_dim=config.nextdit_vlm_hidden_dim,
                latent_emb_size=config.nextdit_latent_emb_size,
                n_query=config.nextdit_n_query,
                dit_dim=config.nextdit_dit_dim,
                dit_layers=config.nextdit_dit_layers,
                dit_heads=config.nextdit_dit_heads,
                dit_kv_heads=config.nextdit_dit_kv_heads,
                dit_ffn_dim_multiplier=config.nextdit_dit_ffn_dim_multiplier,
                predict_steps=config.nextdit_predict_steps,
                action_dim=config.nextdit_action_dim,
                num_inference_steps=config.nextdit_num_inference_steps,
                guidance_scale=config.nextdit_guidance_scale,
                num_sample_trajs=config.nextdit_num_sample_trajs,
                dav2_ckpt_path=config.nextdit_dav2_ckpt_path,
                enable_gradient_checkpointing=config.nextdit_enable_gradient_checkpointing,
            )
            self.nextdit_action_head = NextDiTActionHead(nextdit_cfg).to(
                device=self.device,
                dtype=config.dtype,
            )
            self.latent_queries = nn.Parameter(
                torch.randn(1, config.nextdit_n_query, config.nextdit_vlm_hidden_dim) * 0.02
            )
            logger.info(
                "NextDiTActionHead: dit_layers=%d, predict_steps=%d, n_query=%d",
                config.nextdit_dit_layers,
                config.nextdit_predict_steps,
                config.nextdit_n_query,
            )
            if config.internnav_system1_path:
                self._load_internnav_system1(config.internnav_system1_path)
            elif config.internnav_model_path:
                self._load_internnav_system1_from_model_dir(config.internnav_model_path)

            if config.pano_latent_adapter_enabled:
                from .adapters import PanoLatentSpaceAdapter

                self.pano_latent_adapter = PanoLatentSpaceAdapter(
                    dim=config.nextdit_vlm_hidden_dim,
                    hidden_dim=config.pano_latent_adapter_hidden_dim,
                    dropout=config.pano_latent_adapter_dropout,
                ).to(device=self.device, dtype=config.dtype)
                logger.info(
                    "PanoLatentSpaceAdapter enabled: dim=%d hidden_dim=%d",
                    config.nextdit_vlm_hidden_dim,
                    config.pano_latent_adapter_hidden_dim,
                )
                if config.pano_latent_adapter_checkpoint_path:
                    self._load_pano_latent_adapter(
                        config.pano_latent_adapter_checkpoint_path,
                        strict=config.pano_latent_adapter_strict_load,
                    )
        elif config.nextdit_enabled:
            logger.info("NextDiT action head disabled by config.enable_action_head=False")

        logger.info("=" * 60)
        logger.info("Pipeline initialization complete")
        logger.info("=" * 60)

    @staticmethod
    def _torch_load_checkpoint(path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _extract_pano_adapter_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
        if not isinstance(ckpt, dict):
            raise TypeError("Pano adapter checkpoint must be a dict")

        state = ckpt.get("adapter_state_dict")
        if isinstance(state, dict):
            return state

        for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
            candidate = ckpt.get(key)
            if not isinstance(candidate, dict):
                continue
            prefixed = {
                name.removeprefix("module.").removeprefix("pano_latent_adapter."): value
                for name, value in candidate.items()
                if name.removeprefix("module.").startswith("pano_latent_adapter.")
            }
            if prefixed:
                return prefixed

        if all(torch.is_tensor(value) for value in ckpt.values()):
            return {
                name.removeprefix("module.").removeprefix("pano_latent_adapter."): value for name, value in ckpt.items()
            }

        raise KeyError("No adapter_state_dict or pano_latent_adapter.* trainable_state_dict found in checkpoint")

    def _load_pano_latent_adapter(self, ckpt_path: str, *, strict: bool = True) -> None:
        if self.pano_latent_adapter is None:
            raise RuntimeError("Cannot load pano adapter before it is constructed")
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Pano latent adapter checkpoint not found: {path}")
        ckpt = self._torch_load_checkpoint(str(path))
        state_dict = self._extract_pano_adapter_state_dict(ckpt)
        missing, unexpected = self.pano_latent_adapter.load_state_dict(
            state_dict,
            strict=strict,
        )
        if missing:
            logger.warning("Pano adapter missing keys when loading %s: %s", path, missing)
        if unexpected:
            logger.warning("Pano adapter unexpected keys when loading %s: %s", path, unexpected)
        logger.info(
            "Loaded pano latent adapter from %s (%d tensors)",
            path,
            len(state_dict),
        )

    def adapt_traj_hidden_states(self, traj_hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the optional pano latent adapter before System1 cond_projector."""
        if self.pano_latent_adapter is None:
            return traj_hidden_states
        return self.pano_latent_adapter(traj_hidden_states)

    def _load_internnav_system1(self, ckpt_path: str):
        """Load InternNav System 1 weights into NextDiTActionHead + latent_queries.

        This loads pre-trained weights for the entire NextDiT pipeline from an
        InternNav checkpoint (produced by convert_internnav_backbone.py).
        After loading, System 1 modules are frozen and only latent_queries +
        cond_projector remain trainable (Plan A adaptation strategy).
        """
        from safetensors import safe_open

        logger.info("Loading InternNav System 1 from %s", ckpt_path)
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            ckpt_keys = list(f.keys())
            ckpt_sd = {k: f.get_tensor(k) for k in ckpt_keys}

        self._load_system1_state_dict(ckpt_sd, source=ckpt_path)

    def _load_internnav_system1_from_model_dir(self, model_dir: str):
        """Load System 1 directly from a full InternNav HF checkpoint directory."""
        from safetensors import safe_open

        model_path = Path(model_dir)
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            single_path = model_path / "model.safetensors"
            if not single_path.is_file():
                raise FileNotFoundError(
                    f"InternNav model weights not found in {model_path}: expected "
                    "model.safetensors.index.json or model.safetensors"
                )
            logger.info("Loading InternNav System 1 from full model shard %s", single_path)
            with safe_open(str(single_path), framework="pt", device="cpu") as f:
                ckpt_sd = {_remap_internnav_system1_key(k): f.get_tensor(k) for k in f if _is_internnav_system1_key(k)}
            self._load_system1_state_dict(ckpt_sd, source=str(single_path))
            return

        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        target_keys = {k: shard for k, shard in weight_map.items() if _is_internnav_system1_key(k)}
        if not target_keys:
            raise RuntimeError(f"No InternNav System 1 tensors found in {index_path}")

        keys_by_shard: dict[str, list[str]] = {}
        for key, shard in target_keys.items():
            keys_by_shard.setdefault(shard, []).append(key)

        ckpt_sd = {}
        logger.info(
            "Loading InternNav System 1 from full model %s (%d tensors across %d shards)",
            model_path,
            len(target_keys),
            len(keys_by_shard),
        )
        for shard_name, keys in sorted(keys_by_shard.items()):
            shard_path = model_path / shard_name
            if not shard_path.is_file():
                cache_path = model_path / ".cache" / "huggingface" / "download" / shard_name
                if cache_path.is_file():
                    shard_path = cache_path
                else:
                    raise FileNotFoundError(f"Missing InternNav model shard: {shard_path}")

            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                available = set(f.keys())
                for key in keys:
                    if key not in available:
                        raise KeyError(f"Tensor {key} listed in {index_path} is missing from {shard_path}")
                    ckpt_sd[_remap_internnav_system1_key(key)] = f.get_tensor(key)

        self._load_system1_state_dict(ckpt_sd, source=str(model_path))

    def _load_system1_state_dict(self, ckpt_sd: dict[str, torch.Tensor], source: str):
        """Apply remapped System 1 weights to the action head and latent queries."""
        latent_queries_loaded = False
        if "latent_queries" in ckpt_sd:
            lq = ckpt_sd.pop("latent_queries")
            if self.latent_queries.shape == lq.shape:
                self.latent_queries.data.copy_(lq)
                latent_queries_loaded = True
                logger.info("  Loaded latent_queries %s", tuple(lq.shape))
            else:
                logger.warning(
                    "  latent_queries shape mismatch: ckpt %s vs model %s, skipped",
                    tuple(lq.shape),
                    tuple(self.latent_queries.shape),
                )

        head_sd = self.nextdit_action_head.state_dict()
        loaded, skipped = [], []
        for key, val in ckpt_sd.items():
            if key in head_sd:
                if head_sd[key].shape == val.shape:
                    head_sd[key] = val
                    loaded.append(key)
                else:
                    skipped.append(f"{key}: ckpt {tuple(val.shape)} vs model {tuple(head_sd[key].shape)}")
            else:
                skipped.append(f"{key}: not in model")

        missing = [k for k in head_sd if k not in ckpt_sd and k != "latent_queries"]
        self.nextdit_action_head.load_state_dict(head_sd, strict=False)
        self._internnav_system1_load_audit = {
            "source": source,
            "latent_queries_loaded": latent_queries_loaded,
            "loaded_keys": tuple(loaded),
            "skipped": tuple(skipped),
            "missing_keys": tuple(missing),
        }

        rgb_loaded = sum(1 for key in loaded if key.startswith("rgb_model."))
        traj_loaded = sum(1 for key in loaded if key.startswith("traj_dit."))
        logger.info("  Loaded %d/%d System 1 params from %s", len(loaded), len(ckpt_sd), source)
        logger.info(
            "  DepthAnythingV2 encoder (rgb_model): %d tensors, traj_dit: %d tensors",
            rgb_loaded,
            traj_loaded,
        )
        if rgb_loaded == 0:
            logger.warning(
                "  rgb_model (DepthAnythingV2) received 0 tensors from %s — "
                "set paths.internnav_model_path / INTERNNAV_MODEL_PATH or nextdit.dav2_ckpt_path",
                source,
            )
        if skipped:
            logger.warning("  Skipped: %s", skipped[:10])
        if missing:
            logger.info("  Missing in ckpt (will use random init): %d keys", len(missing))

        self._freeze_system1_core()

    def _freeze_system1_core(self):
        """Freeze System 1 core modules, keep bridge layers trainable.

        Frozen: traj_dit, rgb_model, memory_encoder, rgb_resampler,
                action_encoder, action_decoder, pos_encoding
        Trainable: latent_queries, cond_projector
        """
        head = self.nextdit_action_head
        frozen_modules = [
            head.traj_dit,
            head.rgb_model,
            head.memory_encoder,
            head.rgb_resampler,
            head.action_encoder,
            head.action_decoder,
            head.pos_encoding,
        ]
        frozen_count = 0
        for mod in frozen_modules:
            for p in mod.parameters():
                p.requires_grad = False
                frozen_count += 1

        trainable_count = 0
        for p in head.cond_projector.parameters():
            p.requires_grad = True
            trainable_count += 1
        self.latent_queries.requires_grad = True
        trainable_count += 1

        logger.info(
            "System 1 freeze: %d params frozen, %d bridge params trainable (latent_queries + cond_projector)",
            frozen_count,
            trainable_count,
        )

    def _ensure_heatmap_vln(self):
        """Lazily construct HeatmapVLN after the VLM backbone is loaded."""
        if self.heatmap_vln is not None or not self._heatmap_enabled:
            return

        if not self.qwen2_5_vl._model_loaded:
            self.qwen2_5_vl._load_model()

        cfg = self.config
        default_vit_indices = [7, 15, 23, 31]
        default_llm_indices = [6, 13, 20]
        decoder_mode = getattr(cfg, "heatmap_decoder_mode", "legacy")
        vit_indices = default_vit_indices if cfg.heatmap_vit_layer_indices is None else cfg.heatmap_vit_layer_indices
        if decoder_mode == "pose_free_matcher":
            vit_indices = []
        llm_indices = cfg.heatmap_llm_layer_indices or default_llm_indices

        trajectory_config = getattr(cfg, "heatmap_trajectory_config", None)
        pose_free_config = dict(getattr(cfg, "heatmap_pose_free_config", None) or {})
        pose_free_config.setdefault("heatmap_size", cfg.heatmap_size)

        self.heatmap_vln = HeatmapVLN(
            qwen_model=self.qwen2_5_vl.model,
            processor=self.qwen2_5_vl.processor,
            c_vit=cfg.heatmap_c_vit,
            c_llm=cfg.heatmap_c_llm,
            c_fused=cfg.heatmap_c_fused,
            vit_layer_indices=vit_indices,
            llm_layer_indices=llm_indices,
            spatial_merge_size=getattr(cfg, "spatial_merge_size", 2),
            enable_runtime_timing=cfg.enable_runtime_timing,
            trajectory_config=trajectory_config,
            heatmap_trains_backbone=cfg.heatmap_trains_backbone,
            decoder_mode=decoder_mode,
            pose_free_config=pose_free_config,
        )

        # Move all trainable parts to correct device/dtype
        for module in [
            self.heatmap_vln.vit_dpt_fusion,
            self.heatmap_vln.llm_dpt_fusion,
            self.heatmap_vln.fine,
            self.heatmap_vln.coarse,
            self.heatmap_vln.pose_free_matcher,
        ]:
            if module is not None:
                module.to(device=self.device, dtype=cfg.dtype)

        logger.info(
            "HeatmapVLN v2 constructed (decoder_mode=%s, LLM layers=%s)",
            decoder_mode,
            llm_indices,
        )

    @staticmethod
    def _views_tensor_to_dict(views: torch.Tensor) -> dict[str, Any]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(VIEW_NAMES)}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> list[dict[str, Any]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}")
        return [self._views_tensor_to_dict(history_panoramas[idx]) for idx in range(history_panoramas.shape[0])]

    def _forward_heatmap_batch(
        self,
        current_views: Any,
        history_panoramas: Any,
        instruction_text: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.heatmap_vln is None:
            raise RuntimeError("HeatmapVLN has not been constructed")

        if torch.is_tensor(current_views):
            if current_views.dim() == 4:
                instruction = instruction_text if isinstance(instruction_text, str) else None
                return self.heatmap_vln(
                    self._views_tensor_to_dict(current_views),
                    self._history_tensor_to_list(history_panoramas),
                    instruction=instruction,
                )

            if current_views.dim() != 5:
                raise ValueError(f"Expected current_views [B, 4, C, H, W], got {tuple(current_views.shape)}")
            if not torch.is_tensor(history_panoramas) or history_panoramas.dim() != 6:
                raise ValueError("Batched history_panoramas must be a tensor of shape [B, N, 4, C, H, W]")

            all_visibility = []
            all_heatmaps = []
            for b in range(current_views.shape[0]):
                if isinstance(instruction_text, list):
                    instruction = instruction_text[b] if b < len(instruction_text) else instruction_text[0]
                else:
                    instruction = instruction_text
                sample_output = self.heatmap_vln(
                    self._views_tensor_to_dict(current_views[b]),
                    self._history_tensor_to_list(history_panoramas[b]),
                    instruction=instruction,
                )
                all_visibility.append(sample_output["visibility"])
                all_heatmaps.append(sample_output["heatmaps"])

            result = {
                "visibility": torch.stack(all_visibility, dim=0),
                "heatmaps": torch.stack(all_heatmaps, dim=0),
            }
            if not self.training:
                from .heatmap.heatmap_vln import HeatmapVLN

                result["heatmaps_gated"] = HeatmapVLN._gated_softmax_heatmaps(
                    result["heatmaps"],
                    result["visibility"],
                )
            return result

        return self.heatmap_vln(
            current_views,
            history_panoramas,
            instruction=instruction_text if isinstance(instruction_text, str) else None,
        )

    def forward(
        self,
        video_frames: torch.Tensor,
        instruction_text: str | None = None,
        current_observation: torch.Tensor | None = None,
        return_intermediate: bool = False,
        return_heatmaps: bool = True,
        return_actions: bool = True,
        gt_actions: torch.Tensor | None = None,
        action_valid: torch.Tensor | None = None,
        gt_stop: torch.Tensor | None = None,
        gt_history_heatmap: torch.Tensor | None = None,
        gt_future_heatmap: torch.Tensor | None = None,
        current_views: dict[str, Any] | None = None,
        history_panoramas: list[dict[str, Any]] | None = None,
        panoramic_inputs: dict[str, torch.Tensor] | None = None,
        panoramic_num_histories: list[int] | None = None,
        panoramic_text_anchor_positions: list[dict[int, int]] | None = None,
        history_rel_poses: torch.Tensor | None = None,
        return_lm_loss: bool = False,
        return_lm_correct_logprobs: bool = False,
        return_heatmap_logits: bool = False,
    ) -> dict[str, Any]:
        """Forward pass."""
        del gt_stop  # Reserved for optional stop-head training compatibility.
        batch_size, num_frames = video_frames.shape[:2]

        if current_observation is None:
            current_observation = video_frames[:, -1]

        history_frames = video_frames[:, :-1] if num_frames > 1 else video_frames
        use_panoramic_chain = panoramic_inputs is not None or (
            current_views is not None and history_panoramas is not None
        )
        # Only construct and pass heatmap_vln when panoramic history anchors
        # are present.  In Stage 2 InternNav mode (pano_num_histories all
        # zeros / no panoramic views), the VLM forward runs without hooks.
        has_standard_panoramic_views = current_views is not None and history_panoramas is not None
        has_tokenized_panoramic_history = panoramic_num_histories is not None and any(
            n > 0 for n in panoramic_num_histories
        )
        need_heatmap = (
            use_panoramic_chain
            and return_heatmaps
            and (has_standard_panoramic_views or has_tokenized_panoramic_history)
        )
        if return_heatmap_logits and not need_heatmap:
            raise ValueError(
                "return_heatmap_logits=True requires return_heatmaps=True and "
                "an active panoramic heatmap path with at least one history"
            )
        if need_heatmap:
            self._ensure_heatmap_vln()

        # NextDiT action training only consumes TRAJ latent-query states.
        # Full projected image-token features are needed for intermediate
        # inspection / legacy feature consumers, not for bridge-only Stage2.
        need_projected_sequence_features = return_intermediate
        need_traj_query_features = (
            return_actions and self.nextdit_action_head is not None and self.latent_queries is not None
        )

        qwen_output = None
        raw_hidden_states = None
        llm_tokens = None
        heatmap_output = None
        qwen_timings = None
        qwen_input_stats: dict[str, Any] = {}
        if panoramic_inputs is not None:
            input_ids = panoramic_inputs.get("input_ids")
            if input_ids is not None:
                qwen_input_stats["pano_batch_size"] = int(input_ids.shape[0])
                qwen_input_stats["pano_seq_len"] = int(input_ids.shape[1])
            image_grid = panoramic_inputs.get("image_grid_thw")
            if image_grid is not None:
                qwen_input_stats["pano_image_groups"] = int(image_grid.shape[0])
            video_grid = panoramic_inputs.get("video_grid_thw")
            if video_grid is not None:
                qwen_input_stats["pano_video_groups"] = int(video_grid.shape[0])
            pixel_values = panoramic_inputs.get("pixel_values")
            if pixel_values is not None:
                qwen_input_stats["pano_pixel_values"] = int(pixel_values.shape[0])
            if panoramic_num_histories is not None:
                qwen_input_stats["pano_history_max"] = int(max(panoramic_num_histories))
                qwen_input_stats["pano_history_avg"] = float(sum(panoramic_num_histories)) / max(
                    len(panoramic_num_histories), 1
                )
        should_run_qwen = (
            need_projected_sequence_features
            or need_traj_query_features
            or use_panoramic_chain
            or return_lm_loss
            or return_lm_correct_logprobs
        )
        if should_run_qwen:
            # ==================== Step 1: VLM backbone processing ====================
            qwen_start = time.perf_counter() if self.config.enable_runtime_timing else 0.0
            lq = None
            if need_traj_query_features:
                lq = self.latent_queries.expand(batch_size, -1, -1).to(
                    device=self.device,
                    dtype=self.config.dtype,
                )
            qwen_output = self.qwen2_5_vl(
                history_frames=history_frames,
                current_frame=current_observation,
                instruction=instruction_text,
                return_hidden_states=need_projected_sequence_features,
                generate_text=False,
                current_views=current_views,
                history_panoramas=history_panoramas,
                panoramic_inputs=panoramic_inputs,
                panoramic_num_histories=panoramic_num_histories,
                panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                heatmap_vln=self.heatmap_vln if need_heatmap else None,
                history_rel_poses=history_rel_poses,
                latent_queries=lq,
                return_lm_loss=return_lm_loss,
                return_lm_correct_logprobs=return_lm_correct_logprobs,
            )
            if self.config.enable_runtime_timing:
                qwen_end = time.perf_counter()
                qwen_timings = dict(qwen_output.get("timings", {}) or {})
                qwen_total_s = qwen_end - qwen_start
                qwen_timings.setdefault("qwen_forward_s", qwen_total_s)
                qwen_timings.setdefault("pipeline_qwen_total_s", qwen_total_s)

            if need_projected_sequence_features:
                raw_hidden_states = qwen_output.get("vision_hidden_states")
                if raw_hidden_states is None:
                    raw_hidden_states = qwen_output.get("hidden_states")
                if raw_hidden_states is None:
                    raise RuntimeError("Failed to extract hidden states from VLM backbone")

                if isinstance(raw_hidden_states, list):
                    raw_hidden_states = raw_hidden_states[-1]
                raw_hidden_states = raw_hidden_states.to(
                    device=self.device,
                    dtype=self.config.dtype,
                )

                # ==================== Step 2: Project Hidden States ====================
                llm_tokens = self.llm_projector(raw_hidden_states)

            if return_heatmaps and need_heatmap:
                if "visibility" not in qwen_output or "heatmaps" not in qwen_output:
                    raise RuntimeError("Panoramic Qwen path did not return HeatmapVLN outputs")
                heatmap_output = {
                    "visibility": qwen_output["visibility"],
                    "heatmaps": qwen_output["heatmaps"],
                }
                if return_heatmap_logits:
                    if "heatmap_logits" not in qwen_output:
                        raise RuntimeError(
                            "Raw heatmap logits were explicitly requested but "
                            "the active heatmap decoder did not return them"
                        )
                    heatmap_output["heatmap_logits"] = qwen_output["heatmap_logits"]
                if "heatmaps_gated" in qwen_output:
                    heatmap_output["heatmaps_gated"] = qwen_output["heatmaps_gated"]

        # ==================== Step 4: Action Generation ====================
        trajectory = None
        traj_hidden_states = qwen_output.get("traj_hidden_states") if qwen_output is not None else None

        if return_actions and self.nextdit_action_head is not None and traj_hidden_states is not None:
            if not self.training:
                condition_hidden_states = self.adapt_traj_hidden_states(traj_hidden_states)
                trajectory = self.nextdit_action_head.get_trajectory(
                    condition_hidden_states,
                )

        # ==================== Build Output ====================
        output: dict[str, Any] = {
            "processing_metadata": {
                "num_input_frames": num_frames,
                "batch_size": batch_size,
                "llm_token_shape": None if llm_tokens is None else llm_tokens.shape,
                "timings": qwen_timings,
                "num_image_tokens": (qwen_output.get("num_image_tokens") if isinstance(qwen_output, dict) else None),
                **qwen_input_stats,
            },
        }

        if llm_tokens is not None:
            output["llm_tokens"] = llm_tokens

        if heatmap_output is not None:
            output["visibility"] = heatmap_output["visibility"]
            output["heatmaps"] = heatmap_output["heatmaps"]
            if "heatmap_logits" in heatmap_output:
                output["heatmap_logits"] = heatmap_output["heatmap_logits"]
            if "heatmaps_gated" in heatmap_output:
                output["heatmaps_gated"] = heatmap_output["heatmaps_gated"]

        if traj_hidden_states is not None:
            output["traj_hidden_states"] = traj_hidden_states
        if qwen_output is not None and qwen_output.get("lm_loss") is not None:
            output["lm_loss"] = qwen_output["lm_loss"]
        if qwen_output is not None and qwen_output.get("lm_correct_label_logprobs") is not None:
            output["lm_correct_label_logprobs"] = qwen_output["lm_correct_label_logprobs"]
            output["lm_correct_label_alignment"] = qwen_output["lm_correct_label_alignment"]

        if self.nextdit_action_head is not None:
            output["has_nextdit_action_head"] = True
            if not self.training and trajectory is not None:
                output["trajectory"] = trajectory

        if return_intermediate and raw_hidden_states is not None and qwen_output is not None:
            output["intermediate_features"] = {
                "raw_hidden_states": raw_hidden_states,
                "qwen_output": qwen_output,
            }

        return output

    @torch.no_grad()
    def generate_trajectory(
        self,
        panoramic_inputs: dict[str, torch.Tensor],
        panoramic_num_histories: list[int],
        traj_images: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Two-step inference aligned with InternNav.

        Step 1: auto-regressive pixel-goal text generation.
        Step 2: ``generate_latents`` — forward with generated text + TRAJ
                tokens to extract ``traj_hidden_states``.
        Step 3: NextDiT trajectory generation from conditions.

        Returns the predicted trajectory tensor, or ``None`` on failure.
        """
        if self.nextdit_action_head is None or self.latent_queries is None:
            return None

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in panoramic_inputs.items()}

        with torch.no_grad():
            output_ids = self.qwen2_5_vl.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            ).sequences

        lq = self.latent_queries.expand(1, -1, -1).to(
            device=self.device,
            dtype=self.config.dtype,
        )
        traj_hidden_states = self.qwen2_5_vl.generate_latents(
            output_ids=output_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            latent_queries=lq,
        )
        traj_hidden_states = self.adapt_traj_hidden_states(traj_hidden_states)

        trajectory = self.nextdit_action_head.get_trajectory(
            traj_hidden_states,
            traj_images=traj_images,
        )
        return trajectory

    def forward_packed(
        self,
        packed_batch: dict[str, Any],
        return_intermediate: bool = False,
        return_heatmaps: bool = True,
        return_actions: bool = True,
    ) -> dict[str, Any]:
        """Forward pass with packed batch."""
        batch_size = packed_batch["num_samples"]
        seq_lens = packed_batch["seq_lens"]
        packed_batch["current_frame"] = packed_batch["current_frame"].to(self.device)

        # ==================== Step 1: VLM backbone processing (Packed) ====================
        qwen_output = self.qwen2_5_vl.forward_packed(
            packed_batch=packed_batch,
            return_hidden_states=True,
        )

        raw_hidden_states = qwen_output.get("hidden_states")
        if raw_hidden_states is None:
            raise RuntimeError("Failed to extract hidden states (packed mode)")

        if isinstance(raw_hidden_states, list):
            raw_hidden_states = raw_hidden_states[-1]
        raw_hidden_states = raw_hidden_states.to(
            device=self.device,
            dtype=self.config.dtype,
        )
        if raw_hidden_states.dim() == 2:
            raw_hidden_states = raw_hidden_states.unsqueeze(1)

        # ==================== Step 2: Project Hidden States ====================
        llm_tokens = self.llm_projector(raw_hidden_states)

        # ==================== Step 3: HeatmapVLN v2 ====================
        # In packed mode, heatmap generation through current_views /
        # history_panoramas should be done via forward() instead.
        # This path does not generate heatmaps.

        # ==================== Build Output ====================
        output: dict[str, Any] = {
            "llm_tokens": llm_tokens,
            "processing_metadata": {
                "num_samples": batch_size,
                "seq_lens": seq_lens,
                "total_seq_len": packed_batch["input_ids"].shape[1],
                "llm_token_shape": llm_tokens.shape,
                "mode": "packed",
            },
        }

        if return_intermediate:
            output["intermediate_features"] = {
                "raw_hidden_states": raw_hidden_states,
                "qwen_output": qwen_output,
            }

        return output


def create_vln_pipeline(
    llm_model_path: str = "./models/internnav_backbone",
    heatmap_size: tuple[int, int] = (64, 64),
    device: str = "cuda",
    verbose: bool = True,
    **kwargs,
) -> VLNPipeline:
    """Factory function to create the VLN pipeline."""
    config = VLNPipelineConfig(
        llm_model_path=llm_model_path,
        heatmap_size=heatmap_size,
        device=device,
        verbose=verbose,
        **kwargs,
    )
    return VLNPipeline(config)
