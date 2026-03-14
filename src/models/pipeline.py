"""
VLN Pipeline with Qwen3.5 — v2 (Coarse-to-Fine)
===================================================

Architecture:
    Current panorama (4 views) + N history panoramas + text
        |
    Qwen3.5 (Vision Encoder + LLM, frozen)
        |
    HeatmapVLN (Coarse-to-Fine, ~2M trainable)
        |
    Output:
        - visibility (N_hist, 4)
        - heatmaps   (N_hist, 4, 64, 64)
        - trajectory  (optional, from TransformerActionHead)
        - progress    (optional, from ProgressPredictionHead)
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
import logging
from dataclasses import dataclass

from .qwen3_5 import Qwen3_5Integration, Qwen3_5Config
from .action import (
    DiffusionActionHead,
    DiffusionActionConfig,
    StopPredictionHead,
    TransformerActionHead,
    ProgressPredictionHead,
)
from .heatmap import HeatmapVLN, HeatmapVLNLoss

logger = logging.getLogger(__name__)
VIEW_NAMES = ("front", "right", "back", "left")


@dataclass
class VLNPipelineConfig:
    """Configuration for VLN pipeline."""

    # Qwen3.5 configuration
    llm_model_path: str = "./models/qwen_3.5"
    llm_hidden_dim: int = 4096
    llm_token_dim: int = 1024
    llm_torch_dtype: str = "bfloat16"
    llm_attn_implementation: str = "sdpa"
    max_video_frames: int = 16
    llm_enable_internal_profiling: bool = False
    enable_runtime_timing: bool = False
    llm_enable_compile: bool = False
    llm_compile_mode: str = "reduce-overhead"
    llm_compile_backend: str = "inductor"

    # Sequence Packing configuration (disabled for Qwen3.5)
    enable_packing: bool = False
    max_seq_length: int = 4096
    spatial_merge_size: int = 2

    # Device configuration
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # LoRA configuration for Qwen3.5 fine-tuning
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_num_layers: int = 4
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # ==================== HeatmapVLN v2 (Coarse-to-Fine) ====================
    enable_heatmap: bool = True
    heatmap_c_vit: int = 1152
    heatmap_c_llm: int = 4096
    heatmap_c_fused: int = 256
    heatmap_vit_layer_indices: Optional[List[int]] = None   # e.g. [6, 12, 18, 24]
    heatmap_llm_layer_indices: Optional[List[int]] = None  # e.g. [7, 15, 23] (full_attention)
    heatmap_size: Tuple[int, int] = (64, 64)

    # HeatmapVLNLoss weights
    heatmap_lambda_vis: float = 1.0
    heatmap_lambda_coord: float = 1.0
    heatmap_lambda_kl: float = 1.0
    heatmap_lambda_peak: float = 1.0

    # ==================== Action Head ====================
    action_head_type: str = "transformer"
    enable_action_head: bool = True
    action_dim: int = 2
    action_pred_horizon: int = 1
    action_encoding_size: int = 256
    action_down_dims: List[int] = None
    action_num_diffusion_iters: int = 10
    action_stats_min: List[float] = None
    action_stats_max: List[float] = None

    # Transformer action head
    transformer_action_dim: int = 3
    transformer_predict_size: int = 24
    transformer_n_emb: int = 384
    transformer_n_layer: int = 16
    transformer_n_head: int = 6
    transformer_n_cond_layers: int = 4
    transformer_num_train_timesteps: int = 20
    transformer_p_drop_emb: float = 0.1
    transformer_p_drop_attn: float = 0.1
    transformer_causal_attn: bool = True

    # Stop/Progress prediction
    enable_stop_head: bool = False
    stop_hidden_dim: int = 512
    stop_focal_gamma: float = 2.0
    stop_focal_alpha: float = 0.75
    enable_progress_head: bool = True
    progress_hidden_dim: int = 512

    # Performance settings
    enable_gradient_checkpointing: bool = False
    verbose: bool = False

    # Image size
    image_size: int = 256


class VLNPipeline(nn.Module):
    """
    VLN Pipeline with Qwen3.5 and HeatmapVLN v2 (Coarse-to-Fine).
    """

    def __init__(self, config: VLNPipelineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        logger.info("=" * 60)
        logger.info("Initializing VLN Pipeline with Qwen3.5 (v2 Coarse-to-Fine)")
        logger.info("=" * 60)

        # ==================== Qwen3.5 Integration ====================
        qwen_config = Qwen3_5Config(
            model_path=config.llm_model_path,
            device=config.device,
            torch_dtype=config.llm_torch_dtype,
            attn_implementation=config.llm_attn_implementation,
            max_video_frames=config.max_video_frames,
            enable_internal_profiling=config.llm_enable_internal_profiling,
            enable_runtime_timing=config.enable_runtime_timing,
            enable_compile=config.llm_enable_compile,
            compile_mode=config.llm_compile_mode,
            compile_backend=config.llm_compile_backend,
            enable_packing=config.enable_packing,
            max_seq_length=config.max_seq_length,
            spatial_merge_size=config.spatial_merge_size,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_num_layers=config.lora_num_layers,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
        )
        self.qwen3_5 = Qwen3_5Integration(qwen_config)
        logger.info("Qwen3.5 integration initialized")

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
            config.llm_hidden_dim, config.llm_token_dim,
        )

        # ==================== HeatmapVLN v2 ====================
        # HeatmapVLN is constructed lazily after Qwen3.5 model is loaded,
        # because it needs the actual model instance for hook registration.
        self.heatmap_vln: Optional[HeatmapVLN] = None
        self._heatmap_enabled = config.enable_heatmap

        # ==================== Action Head ====================
        self.action_head = None
        self.transformer_action_head = None

        if config.enable_action_head:
            if config.action_head_type == "transformer":
                self.transformer_action_head = TransformerActionHead(
                    vlm_token_dim=config.llm_token_dim,
                    n_emb=config.transformer_n_emb,
                    predict_size=config.transformer_predict_size,
                    n_layer=config.transformer_n_layer,
                    n_head=config.transformer_n_head,
                    n_cond_layers=config.transformer_n_cond_layers,
                    p_drop_emb=config.transformer_p_drop_emb,
                    p_drop_attn=config.transformer_p_drop_attn,
                    action_dim=config.transformer_action_dim,
                    num_train_timesteps=config.transformer_num_train_timesteps,
                    causal_attn=config.transformer_causal_attn,
                ).to(device=self.device, dtype=config.dtype)
                logger.info(
                    "TransformerActionHead: predict_size=%d, n_layer=%d, action_dim=%d",
                    config.transformer_predict_size,
                    config.transformer_n_layer,
                    config.transformer_action_dim,
                )
            else:
                action_config_kwargs = {
                    'action_dim': config.action_dim,
                    'pred_horizon': config.action_pred_horizon,
                    'cond_dim': config.llm_token_dim,
                    'encoding_size': config.action_encoding_size,
                    'num_diffusion_iters': config.action_num_diffusion_iters,
                    'action_stats_min': config.action_stats_min or [-0.17, -0.03],
                    'action_stats_max': config.action_stats_max or [0.19, 0.31],
                    'device': str(self.device),
                }
                if config.action_down_dims is not None:
                    action_config_kwargs['down_dims'] = config.action_down_dims
                action_config = DiffusionActionConfig(**action_config_kwargs)
                self.action_head = DiffusionActionHead(action_config).to(
                    device=self.device, dtype=config.dtype,
                )
                logger.info("DiffusionActionHead (legacy) initialized")

        # ==================== Stop Head (Legacy) ====================
        if config.enable_stop_head:
            self.stop_head = StopPredictionHead(
                input_dim=config.llm_token_dim,
                hidden_dim=config.stop_hidden_dim,
                dropout=0.1,
                focal_gamma=config.stop_focal_gamma,
                focal_alpha=config.stop_focal_alpha,
            ).to(device=self.device, dtype=config.dtype)
            logger.info("Stop Head initialized")
        else:
            self.stop_head = None

        # ==================== Progress Head ====================
        if config.enable_progress_head:
            self.progress_head = ProgressPredictionHead(
                input_dim=config.llm_token_dim,
                hidden_dim=config.progress_hidden_dim,
                dropout=0.1,
                concat_state_txt=True,
            ).to(device=self.device, dtype=config.dtype)
            logger.info("Progress Head initialized")
        else:
            self.progress_head = None

        logger.info("=" * 60)
        logger.info("Pipeline initialization complete")
        logger.info("=" * 60)

    def _ensure_heatmap_vln(self):
        """Lazily construct HeatmapVLN after Qwen3.5 model is loaded."""
        if self.heatmap_vln is not None or not self._heatmap_enabled:
            return

        if not self.qwen3_5._model_loaded:
            self.qwen3_5._load_model()

        cfg = self.config
        vit_indices = cfg.heatmap_vit_layer_indices or [6, 12, 18, 24]
        llm_indices = cfg.heatmap_llm_layer_indices or [7, 15, 23]

        self.heatmap_vln = HeatmapVLN(
            qwen_model=self.qwen3_5.model,
            processor=self.qwen3_5.processor,
            c_vit=cfg.heatmap_c_vit,
            c_llm=cfg.heatmap_c_llm,
            c_fused=cfg.heatmap_c_fused,
            vit_layer_indices=vit_indices,
            llm_layer_indices=llm_indices,
            enable_runtime_timing=cfg.enable_runtime_timing,
        )

        # Move all trainable parts to correct device/dtype
        for module in [
            self.heatmap_vln.vit_dpt_fusion,
            self.heatmap_vln.llm_dpt_fusion,
            self.heatmap_vln.fine,
            self.heatmap_vln.coarse,
        ]:
            module.to(device=self.device, dtype=cfg.dtype)

        logger.info(
            "HeatmapVLN v2 constructed (Coarse-to-Fine, LLM layers=%s)",
            llm_indices,
        )

    @staticmethod
    def _views_tensor_to_dict(views: torch.Tensor) -> Dict[str, Any]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(VIEW_NAMES)}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> List[Dict[str, Any]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(
                f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}"
            )
        return [
            self._views_tensor_to_dict(history_panoramas[idx])
            for idx in range(history_panoramas.shape[0])
        ]

    def _forward_heatmap_batch(
        self,
        current_views: Any,
        history_panoramas: Any,
        instruction_text: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
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
                raise ValueError(
                    f"Expected current_views [B, 4, C, H, W], got {tuple(current_views.shape)}"
                )
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
                from src.models.heatmap.heatmap_vln import HeatmapVLN
                result["heatmaps_gated"] = HeatmapVLN._gated_softmax_heatmaps(
                    result["heatmaps"], result["visibility"],
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
        instruction_text: Optional[str] = None,
        current_observation: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        return_heatmaps: bool = True,
        return_actions: bool = True,
        gt_actions: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        gt_stop: Optional[torch.Tensor] = None,
        gt_history_heatmap: Optional[torch.Tensor] = None,
        gt_future_heatmap: Optional[torch.Tensor] = None,
        current_views: Optional[Dict[str, Any]] = None,
        history_panoramas: Optional[List[Dict[str, Any]]] = None,
        panoramic_inputs: Optional[Dict[str, torch.Tensor]] = None,
        panoramic_num_histories: Optional[List[int]] = None,
        panoramic_text_anchor_positions: Optional[List[Dict[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        For the v2 Coarse-to-Fine heatmap path, provide ``current_views``
        and ``history_panoramas`` directly.  The legacy ``video_frames``
        path still works for action / progress heads.
        """
        batch_size, num_frames = video_frames.shape[:2]

        if current_observation is None:
            current_observation = video_frames[:, -1]

        history_frames = video_frames[:, :-1] if num_frames > 1 else video_frames
        use_panoramic_chain = panoramic_inputs is not None or (
            current_views is not None and history_panoramas is not None
        )
        if use_panoramic_chain:
            self._ensure_heatmap_vln()

        need_sequence_features = return_intermediate or return_actions

        qwen_output = None
        raw_hidden_states = None
        llm_tokens = None
        heatmap_output = None
        qwen_timings = None
        should_run_qwen = need_sequence_features or use_panoramic_chain
        if should_run_qwen:
            # ==================== Step 1: Qwen3.5 Processing ====================
            qwen_start = time.perf_counter() if self.config.enable_runtime_timing else 0.0
            qwen_output = self.qwen3_5(
                history_frames=history_frames,
                current_frame=current_observation,
                instruction=instruction_text,
                return_hidden_states=need_sequence_features,
                generate_text=False,
                current_views=current_views,
                history_panoramas=history_panoramas,
                panoramic_inputs=panoramic_inputs,
                panoramic_num_histories=panoramic_num_histories,
                panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                heatmap_vln=self.heatmap_vln if use_panoramic_chain else None,
            )
            if self.config.enable_runtime_timing:
                qwen_end = time.perf_counter()
                qwen_timings = dict(qwen_output.get('timings', {}) or {})
                qwen_timings.setdefault('pipeline_qwen_total_s', qwen_end - qwen_start)

            if need_sequence_features:
                raw_hidden_states = qwen_output.get('vision_hidden_states')
                if raw_hidden_states is None:
                    raw_hidden_states = qwen_output.get('hidden_states')
                if raw_hidden_states is None:
                    raise RuntimeError("Failed to extract hidden states from Qwen3.5")

                if isinstance(raw_hidden_states, list):
                    raw_hidden_states = raw_hidden_states[-1]
                raw_hidden_states = raw_hidden_states.to(
                    device=self.device, dtype=self.config.dtype,
                )

                # ==================== Step 2: Project Hidden States ====================
                llm_tokens = self.llm_projector(raw_hidden_states)

            if return_heatmaps and use_panoramic_chain:
                if 'visibility' not in qwen_output or 'heatmaps' not in qwen_output:
                    raise RuntimeError("Panoramic Qwen path did not return HeatmapVLN outputs")
                heatmap_output = {
                    'visibility': qwen_output['visibility'],
                    'heatmaps': qwen_output['heatmaps'],
                }
                if 'heatmaps_gated' in qwen_output:
                    heatmap_output['heatmaps_gated'] = qwen_output['heatmaps_gated']

        # ==================== Step 4: Action Generation ====================
        actions = None
        trajectory = None
        action_cond = llm_tokens.mean(dim=1) if llm_tokens is not None else None

        if return_actions and llm_tokens is not None:
            if self.transformer_action_head is not None:
                if not self.training:
                    trajectory = self.transformer_action_head.get_trajectory(llm_tokens)
            elif self.action_head is not None:
                if not self.training:
                    actions = self.action_head(action_cond)

        # ==================== Step 5: Stop/Progress ====================
        progress = None
        stop_logits = None
        stop_prob = None

        if self.progress_head is not None and llm_tokens is not None:
            progress = self.progress_head.get_progress(llm_tokens)

        if self.stop_head is not None and llm_tokens is not None:
            stop_cond = llm_tokens.mean(dim=1)
            stop_logits = self.stop_head.classifier(stop_cond).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)

        # ==================== Build Output ====================
        output: Dict[str, Any] = {
            'processing_metadata': {
                'num_input_frames': num_frames,
                'batch_size': batch_size,
                'llm_token_shape': None if llm_tokens is None else llm_tokens.shape,
                'timings': qwen_timings,
            },
        }

        if llm_tokens is not None:
            output['llm_tokens'] = llm_tokens

        if heatmap_output is not None:
            output['visibility'] = heatmap_output['visibility']
            output['heatmaps'] = heatmap_output['heatmaps']
            if 'heatmaps_gated' in heatmap_output:
                output['heatmaps_gated'] = heatmap_output['heatmaps_gated']

        if action_cond is not None:
            output['action_cond'] = action_cond

        if self.transformer_action_head is not None:
            output['has_transformer_action_head'] = True
            if not self.training and trajectory is not None:
                output['trajectory'] = trajectory
        elif self.action_head is not None:
            output['has_action_head'] = True
            if not self.training and actions is not None:
                output['actions'] = actions

        if progress is not None:
            output['progress'] = progress
        if stop_logits is not None:
            output['stop_logits'] = stop_logits
            output['stop_prob'] = stop_prob

        if return_intermediate and raw_hidden_states is not None and qwen_output is not None:
            output['intermediate_features'] = {
                'raw_hidden_states': raw_hidden_states,
                'qwen_output': qwen_output,
            }

        return output

    def forward_packed(
        self,
        packed_batch: Dict[str, Any],
        return_intermediate: bool = False,
        return_heatmaps: bool = True,
        return_actions: bool = True,
        gt_actions: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        gt_stop: Optional[torch.Tensor] = None,
        gt_history_heatmap: Optional[torch.Tensor] = None,
        gt_future_heatmap: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with packed batch.

        Note: HeatmapVLN v2 uses its own multi-image input construction
        (construct_input), so the packed path primarily serves action/progress
        heads. Heatmap generation should use forward() with current_views /
        history_panoramas.
        """
        batch_size = packed_batch["num_samples"]
        seq_lens = packed_batch["seq_lens"]
        current_observation = packed_batch["current_frame"].to(self.device)

        # ==================== Step 1: Qwen3.5 Processing (Packed) ====================
        qwen_output = self.qwen3_5.forward_packed(
            packed_batch=packed_batch,
            return_hidden_states=True,
        )

        raw_hidden_states = qwen_output.get('hidden_states')
        if raw_hidden_states is None:
            raise RuntimeError("Failed to extract hidden states (packed mode)")

        if isinstance(raw_hidden_states, list):
            raw_hidden_states = raw_hidden_states[-1]
        raw_hidden_states = raw_hidden_states.to(
            device=self.device, dtype=self.config.dtype,
        )
        if raw_hidden_states.dim() == 2:
            raw_hidden_states = raw_hidden_states.unsqueeze(1)

        # ==================== Step 2: Project Hidden States ====================
        llm_tokens = self.llm_projector(raw_hidden_states)

        # ==================== Step 3: HeatmapVLN v2 ====================
        # In packed mode, heatmap generation through current_views /
        # history_panoramas should be done via forward() instead.
        # This path does not generate heatmaps.

        # ==================== Step 4: Action Generation ====================
        actions = None
        trajectory = None
        action_cond = llm_tokens.mean(dim=1)

        if return_actions:
            if self.transformer_action_head is not None:
                if not self.training:
                    trajectory = self.transformer_action_head.get_trajectory(llm_tokens)
            elif self.action_head is not None:
                if not self.training:
                    actions = self.action_head(action_cond)

        # ==================== Step 5: Stop/Progress ====================
        progress = None
        stop_logits = None
        stop_prob = None

        if self.progress_head is not None:
            progress = self.progress_head.get_progress(llm_tokens)

        if self.stop_head is not None:
            stop_cond = llm_tokens.mean(dim=1)
            stop_logits = self.stop_head.classifier(stop_cond).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)

        # ==================== Build Output ====================
        output: Dict[str, Any] = {
            'llm_tokens': llm_tokens,
            'processing_metadata': {
                'num_samples': batch_size,
                'seq_lens': seq_lens,
                'total_seq_len': packed_batch["input_ids"].shape[1],
                'llm_token_shape': llm_tokens.shape,
                'mode': 'packed',
            },
        }

        output['action_cond'] = action_cond

        if self.transformer_action_head is not None:
            output['has_transformer_action_head'] = True
            if not self.training and trajectory is not None:
                output['trajectory'] = trajectory
        elif self.action_head is not None:
            output['has_action_head'] = True
            if not self.training and actions is not None:
                output['actions'] = actions

        if progress is not None:
            output['progress'] = progress
        if stop_logits is not None:
            output['stop_logits'] = stop_logits
            output['stop_prob'] = stop_prob

        if return_intermediate:
            output['intermediate_features'] = {
                'raw_hidden_states': raw_hidden_states,
                'qwen_output': qwen_output,
            }

        return output


def create_vln_pipeline(
    llm_model_path: str = "./models/qwen_3.5",
    heatmap_size: Tuple[int, int] = (64, 64),
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
