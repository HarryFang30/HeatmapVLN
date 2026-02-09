"""
VLN Pipeline with Qwen3-VL
==========================

This module provides a VLN pipeline that uses Qwen3-VL directly
for video understanding.

Architecture:
    Input: history_frames + current_frame + instruction
        ↓
    Qwen3-VL (Vision Encoder + LLM)
        ↓
    Hidden States Projection (2048 → 1024)
        ↓
    Output Heads:
        - History Heatmap Head (Diffusion)
        - Future Heatmap Head (Diffusion)
        - Action Head (Diffusion Policy)
        - Stop Head (Binary Classifier)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
import logging
from dataclasses import dataclass

from .qwen3_vl import Qwen3VLIntegration, Qwen3VLConfig
from .action import (
    DiffusionActionHead, 
    DiffusionActionConfig, 
    StopPredictionHead,
    TransformerActionHead,
    ProgressPredictionHead,
)
from .heatmap import DiffusionHeatmapHead, DiffusionHeatmapConfig

logger = logging.getLogger(__name__)


@dataclass
class VLNPipelineConfig:
    """Configuration for VLN pipeline."""
    
    # Qwen3-VL configuration
    llm_model_path: str = "./models/qwen_3_vl"
    llm_hidden_dim: int = 4096  # Qwen3-VL 7B hidden size
    llm_token_dim: int = 1024   # Projected dimension for output heads
    llm_torch_dtype: str = "bfloat16"
    llm_attn_implementation: str = "sdpa"  # sdpa works without flash_attn
    max_video_frames: int = 16
    
    # Sequence Packing configuration (based on official Qwen3-VL fine-tuning)
    enable_packing: bool = False   # Whether to use sequence packing
    max_seq_length: int = 4096     # Maximum packed sequence length
    spatial_merge_size: int = 2    # Vision spatial merge size for position IDs
    
    # Device configuration
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    
    # Heatmap generation
    heatmap_size: Tuple[int, int] = (64, 64)
    enable_history_heatmap_head: bool = True
    enable_future_heatmap_head: bool = True
    diffusion_heatmap_cond_dim: int = 512
    diffusion_heatmap_num_inference_steps: int = 10
    
    # Image size for heatmap encoder
    image_size: int = 224
    
    # Heatmap head ablation settings
    heatmap_use_image_encoder: bool = True  # Set to False to disable CNN encoder (LLM-only mode)
    heatmap_pool_method: str = "attention"  # 'attention', 'mean', 'first', 'last', 'max'
    heatmap_pool_num_heads: int = 4  # Number of attention heads for attention pooling
    heatmap_use_circular_padding: bool = False  # 360° panorama: circular padding for horizontal edges
    heatmap_dropout: float = 0.1  # Dropout rate for heatmap head
    
    # Heatmap UNet architecture (controls model capacity)
    heatmap_block_out_channels: Tuple[int, ...] = (64, 128, 256)  # UNet channel dims per level
    heatmap_layers_per_block: int = 2  # ResBlocks per level
    heatmap_attention_levels: Tuple[int, ...] = (2,)  # Which levels have attention
    heatmap_num_train_timesteps: int = 100  # Diffusion training steps
    heatmap_cfg_drop_prob: float = 0.1  # CFG: drop condition probability during training
    heatmap_cfg_scale: float = 3.0  # CFG: guidance scale during inference
    
    # Sequence cross-attention conditioning (dual-path: FiLM + sequence cross-attn)
    heatmap_use_sequence_conditioning: bool = False  # Enable sequence-level cross-attention
    heatmap_seq_cross_attn_heads: int = 8            # Cross-attention heads
    heatmap_seq_cross_attn_head_dim: int = 64        # Head dimension
    
    # Visibility head (suppress false positives for invisible history points)
    heatmap_use_visibility_head: bool = False         # Enable visibility prediction head
    heatmap_visibility_loss_weight: float = 0.5       # Visibility BCE loss weight
    heatmap_visibility_threshold: float = 0.5         # Inference visibility threshold
    
    # Spatial feature injection (CNN multi-scale features -> UNet skip connections)
    heatmap_use_spatial_injection: bool = False       # Enable spatial feature injection
    
    # LoRA configuration for Qwen3-VL fine-tuning
    use_lora: bool = False           # Enable LoRA on Qwen3-VL
    lora_rank: int = 16              # LoRA rank
    lora_alpha: int = 32             # LoRA alpha (typically 2x rank)
    lora_num_layers: int = 4         # Number of last LLM layers to apply LoRA
    lora_dropout: float = 0.05       # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Target modules (default: ["q_proj", "v_proj"])
    
    # Action generation - Mode selection
    # 'legacy': DiffusionActionHead (UNet1D)
    # 'transformer': TransformerActionHead (InternNav style)
    action_head_type: str = "transformer"  # 'legacy' or 'transformer'
    
    # Legacy action head settings (DiffusionActionHead)
    enable_action_head: bool = True
    action_dim: int = 2
    action_pred_horizon: int = 1
    action_encoding_size: int = 256
    action_down_dims: List[int] = None
    action_num_diffusion_iters: int = 10
    action_stats_min: List[float] = None
    action_stats_max: List[float] = None
    
    # Transformer action head settings (TransformerActionHead, InternNav style)
    transformer_action_dim: int = 3  # (dx, dy, delta_yaw)
    transformer_predict_size: int = 24  # 24 step trajectory
    transformer_n_emb: int = 384  # Internal embedding dimension
    transformer_n_layer: int = 16  # Transformer decoder layers
    transformer_n_head: int = 6  # 对齐 InternNav: n_emb // head_dim = 384 // 64 = 6
    transformer_n_cond_layers: int = 4  # Condition encoder layers
    transformer_num_train_timesteps: int = 20
    transformer_p_drop_emb: float = 0.1  # Embedding dropout
    transformer_p_drop_attn: float = 0.1  # Attention dropout
    transformer_causal_attn: bool = True  # Use causal attention
    
    # Stop/Progress prediction
    enable_stop_head: bool = False  # Deprecated, use progress_head instead
    stop_hidden_dim: int = 512
    stop_focal_gamma: float = 2.0
    stop_focal_alpha: float = 0.75
    
    # Progress prediction (replaces stop prediction)
    enable_progress_head: bool = True
    progress_hidden_dim: int = 512
    
    # Performance settings
    enable_gradient_checkpointing: bool = False
    verbose: bool = False


class VLNPipeline(nn.Module):
    """
    VLN Pipeline with Qwen3-VL.
    
    This pipeline uses Qwen3-VL for video understanding,
    extracting hidden states for downstream prediction heads.
    """
    
    def __init__(self, config: VLNPipelineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info("=" * 60)
        logger.info("Initializing VLN Pipeline with Qwen3-VL")
        logger.info("=" * 60)
        
        # ==================== Qwen3-VL Integration ====================
        qwen_config = Qwen3VLConfig(
            model_path=config.llm_model_path,
            device=config.device,
            torch_dtype=config.llm_torch_dtype,
            attn_implementation=config.llm_attn_implementation,
            max_video_frames=config.max_video_frames,
            # Sequence packing settings
            enable_packing=config.enable_packing,
            max_seq_length=config.max_seq_length,
            spatial_merge_size=config.spatial_merge_size,
            # LoRA settings
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_num_layers=config.lora_num_layers,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
        )
        self.qwen3_vl = Qwen3VLIntegration(qwen_config)
        if config.enable_packing:
            logger.info(f"✓ Qwen3-VL integration initialized (packing enabled, max_seq={config.max_seq_length})")
        else:
            logger.info(f"✓ Qwen3-VL integration initialized")
        
        # ==================== LLM Projector ====================
        self.llm_projector = nn.Sequential(
            nn.LayerNorm(config.llm_hidden_dim),
            nn.Linear(config.llm_hidden_dim, config.llm_token_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # 增加dropout防止过拟合
            nn.Linear(config.llm_token_dim, config.llm_token_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(config.llm_token_dim),
        ).to(device=self.device, dtype=config.dtype)
        logger.info(f"✓ LLM projector: {config.llm_hidden_dim} → {config.llm_token_dim}")
        
        # ==================== Heatmap Heads ====================
        heatmap_device = self.device
        diffusion_heatmap_config = DiffusionHeatmapConfig(
            llm_dim=config.llm_token_dim,
            cond_dim=config.diffusion_heatmap_cond_dim,
            heatmap_size=config.heatmap_size,
            num_inference_steps=config.diffusion_heatmap_num_inference_steps,
            image_size=(config.image_size, config.image_size),
            # Ablation settings - now correctly passed from config
            use_image_encoder=config.heatmap_use_image_encoder,
            llm_pool_method=config.heatmap_pool_method,
            llm_pool_num_heads=config.heatmap_pool_num_heads,
            # 360° panorama support
            use_circular_padding=config.heatmap_use_circular_padding,
            # Regularization
            dropout=config.heatmap_dropout,
            # UNet architecture (model capacity)
            block_out_channels=config.heatmap_block_out_channels,
            layers_per_block=config.heatmap_layers_per_block,
            attention_levels=config.heatmap_attention_levels,
            num_train_timesteps=config.heatmap_num_train_timesteps,
            # Classifier-Free Guidance (CFG)
            cfg_drop_prob=config.heatmap_cfg_drop_prob,
            cfg_scale=config.heatmap_cfg_scale,
            # Sequence cross-attention conditioning
            use_sequence_conditioning=config.heatmap_use_sequence_conditioning,
            seq_cross_attn_heads=config.heatmap_seq_cross_attn_heads,
            seq_cross_attn_head_dim=config.heatmap_seq_cross_attn_head_dim,
            # Visibility head (suppress false positives)
            use_visibility_head=config.heatmap_use_visibility_head,
            visibility_loss_weight=config.heatmap_visibility_loss_weight,
            visibility_threshold=config.heatmap_visibility_threshold,
            # Spatial feature injection
            use_spatial_injection=config.heatmap_use_spatial_injection,
        )
        
        # History Heatmap Head
        if config.enable_history_heatmap_head:
            self.history_heatmap_head = DiffusionHeatmapHead(diffusion_heatmap_config).to(
                device=heatmap_device, dtype=config.dtype
            )
            logger.info(
                f"✓ History Heatmap Head initialized "
                f"(use_image_encoder={config.heatmap_use_image_encoder}, "
                f"pool_method={config.heatmap_pool_method})"
            )
        else:
            self.history_heatmap_head = None
        
        # Future Heatmap Head
        if config.enable_future_heatmap_head:
            self.future_heatmap_head = DiffusionHeatmapHead(diffusion_heatmap_config).to(
                device=heatmap_device, dtype=config.dtype
            )
            logger.info(
                f"✓ Future Heatmap Head initialized "
                f"(use_image_encoder={config.heatmap_use_image_encoder}, "
                f"pool_method={config.heatmap_pool_method})"
            )
        else:
            self.future_heatmap_head = None
        
        # ==================== Action Head ====================
        self.action_head = None
        self.transformer_action_head = None
        
        if config.enable_action_head:
            if config.action_head_type == "transformer":
                # New: TransformerActionHead (InternNav style)
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
                    f"✓ TransformerActionHead initialized: "
                    f"predict_size={config.transformer_predict_size}, "
                    f"n_layer={config.transformer_n_layer}, "
                    f"n_cond_layers={config.transformer_n_cond_layers}, "
                    f"action_dim={config.transformer_action_dim}"
                )
            else:
                # Legacy: DiffusionActionHead
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
                    device=self.device, dtype=config.dtype
                )
                logger.info(f"✓ DiffusionActionHead (legacy) initialized")
        
        # ==================== Stop Head (Legacy) ====================
        if config.enable_stop_head:
            self.stop_head = StopPredictionHead(
                input_dim=config.llm_token_dim,
                hidden_dim=config.stop_hidden_dim,
                dropout=0.1,
                focal_gamma=config.stop_focal_gamma,
                focal_alpha=config.stop_focal_alpha,
            ).to(device=self.device, dtype=config.dtype)
            logger.info(f"✓ Stop Head initialized")
        else:
            self.stop_head = None
        
        # ==================== Progress Head (New) ====================
        if config.enable_progress_head:
            self.progress_head = ProgressPredictionHead(
                input_dim=config.llm_token_dim,
                hidden_dim=config.progress_hidden_dim,
                dropout=0.1,
                concat_state_txt=True,  # 对齐 InternNav: state + text_embed 拼接
            ).to(device=self.device, dtype=config.dtype)
            logger.info(f"✓ Progress Head initialized (InternNav concat_state_txt mode)")
        else:
            self.progress_head = None
        
        logger.info("=" * 60)
        logger.info("Pipeline initialization complete")
        logger.info("=" * 60)
    
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
    ) -> Dict[str, Any]:
        """
        Forward pass of the pipeline.
        
        Args:
            video_frames: Input video [B, N, C, H, W]
            instruction_text: Navigation instruction
            current_observation: Current view [B, C, H, W] (optional, uses last frame if None)
            return_intermediate: Return intermediate features
            return_heatmaps: Generate heatmaps
            return_actions: Generate actions
            gt_actions: Ground truth actions for training
            action_valid: Action validity mask
            gt_stop: Ground truth stop labels
            gt_history_heatmap: GT history heatmap for training
            gt_future_heatmap: GT future heatmap for training
            
        Returns:
            Dictionary containing predictions and losses
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Set current observation (use last frame if not provided)
        if current_observation is None:
            current_observation = video_frames[:, -1]  # [B, C, H, W]
        
        # History frames are all except the last
        history_frames = video_frames[:, :-1] if num_frames > 1 else video_frames
        
        if self.config.verbose:
            logger.info(f"Processing: {num_frames} frames, batch_size={batch_size}")
        
        # ==================== Step 1: Qwen3-VL Processing ====================
        qwen_output = self.qwen3_vl(
            history_frames=history_frames,
            current_frame=current_observation,
            instruction=instruction_text,
            return_hidden_states=True,
            generate_text=False,
        )
        
        # Get hidden states (prefer vision-specific, fallback to full)
        raw_hidden_states = qwen_output.get('vision_hidden_states')
        if raw_hidden_states is None:
            raw_hidden_states = qwen_output.get('hidden_states')
        
        if raw_hidden_states is None:
            raise RuntimeError("Failed to extract hidden states from Qwen3-VL")
        
        # Move to device
        raw_hidden_states = raw_hidden_states.to(device=self.device, dtype=self.config.dtype)
        
        # ==================== Step 2: Project Hidden States ====================
        llm_tokens = self.llm_projector(raw_hidden_states)  # [B, seq_len, llm_token_dim]
        
        if self.config.verbose:
            logger.info(f"LLM tokens shape: {llm_tokens.shape}")
        
        # ==================== Step 3: Heatmap Generation ====================
        history_heatmap = None
        future_heatmap = None
        history_heatmap_loss = None
        future_heatmap_loss = None
        history_heatmap_noise_std = None
        history_heatmap_noise_pred_std = None
        future_heatmap_noise_std = None
        future_heatmap_noise_pred_std = None
        history_heatmap_base_loss = None
        history_heatmap_focal_loss = None
        history_heatmap_visibility_loss = None
        
        if return_heatmaps:
            observation_for_heatmap = current_observation.to(device=self.device, dtype=self.config.dtype)
            llm_tokens_for_heatmap = llm_tokens.to(dtype=self.config.dtype)
            
            # History Heatmap
            if self.history_heatmap_head is not None:
                if gt_history_heatmap is not None:
                    # 有 GT 时用噪声预测（训练和验证均可用，只需 1 次 UNet 前向）
                    gt_history_hm = gt_history_heatmap.to(self.device)
                    result = self.history_heatmap_head(
                        llm_tokens=llm_tokens_for_heatmap,
                        observation=observation_for_heatmap,
                        gt_heatmap=gt_history_hm,
                        return_loss=True,
                        skip_inference=not self.training,  # eval 模式跳过推理（验证加速）
                    )
                    history_heatmap_loss = result['loss']
                    history_heatmap = result.get('heatmap')
                    history_heatmap_noise_std = result.get('noise_std')
                    history_heatmap_noise_pred_std = result.get('noise_pred_std')
                    history_heatmap_base_loss = result.get('base_loss')
                    history_heatmap_focal_loss = result.get('focal_loss')
                    history_heatmap_visibility_loss = result.get('visibility_loss')
                else:
                    # 无 GT 时走完整扩散推理（纯推理/可视化）
                    history_heatmap = self.history_heatmap_head(
                        llm_tokens=llm_tokens_for_heatmap,
                        observation=observation_for_heatmap,
                    )
            
            # Future Heatmap
            if self.future_heatmap_head is not None:
                if gt_future_heatmap is not None:
                    # 有 GT 时用噪声预测
                    gt_future_hm = gt_future_heatmap.to(self.device)
                    result = self.future_heatmap_head(
                        llm_tokens=llm_tokens_for_heatmap,
                        observation=observation_for_heatmap,
                        gt_heatmap=gt_future_hm,
                        return_loss=True,
                        skip_inference=not self.training,  # eval 模式跳过推理
                    )
                    future_heatmap_loss = result['loss']
                    future_heatmap = result.get('heatmap')
                    future_heatmap_noise_std = result.get('noise_std')
                    future_heatmap_noise_pred_std = result.get('noise_pred_std')
                else:
                    future_heatmap = self.future_heatmap_head(
                        llm_tokens=llm_tokens_for_heatmap,
                        observation=observation_for_heatmap,
                    )
        
        # ==================== Step 4: Action Generation ====================
        actions = None
        trajectory = None
        action_cond = llm_tokens.mean(dim=1)  # [B, llm_token_dim]
        
        if return_actions:
            # New: TransformerActionHead (trajectory prediction)
            if self.transformer_action_head is not None:
                if not self.training:
                    trajectory = self.transformer_action_head.get_trajectory(llm_tokens)
            # Legacy: DiffusionActionHead
            elif self.action_head is not None:
                if not self.training:
                    actions = self.action_head(action_cond)
        
        # ==================== Step 5: Stop/Progress Prediction ====================
        stop_logits = None
        stop_prob = None
        progress = None
        
        # Progress prediction (new)
        if self.progress_head is not None:
            progress = self.progress_head.get_progress(llm_tokens)
        
        # Stop prediction (legacy)
        if self.stop_head is not None:
            stop_cond = llm_tokens.mean(dim=1)
            stop_logits = self.stop_head.classifier(stop_cond).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)
        
        # ==================== Build Output ====================
        output = {
            'llm_tokens': llm_tokens,
            'processing_metadata': {
                'num_input_frames': num_frames,
                'batch_size': batch_size,
                'llm_token_shape': llm_tokens.shape,
            }
        }
        
        # Heatmaps
        if history_heatmap is not None:
            output['history_heatmaps'] = history_heatmap.unsqueeze(1)
        if future_heatmap is not None:
            output['future_heatmaps'] = future_heatmap.unsqueeze(1)
        
        # Heatmap losses
        if history_heatmap_loss is not None:
            output['history_heatmap_loss'] = history_heatmap_loss
            if history_heatmap_noise_std is not None:
                output['history_heatmap_noise_std'] = history_heatmap_noise_std
                output['history_heatmap_noise_pred_std'] = history_heatmap_noise_pred_std
            if history_heatmap_base_loss is not None:
                output['history_heatmap_base_loss'] = history_heatmap_base_loss
                output['history_heatmap_focal_loss'] = history_heatmap_focal_loss
            if history_heatmap_visibility_loss is not None:
                output['history_heatmap_visibility_loss'] = history_heatmap_visibility_loss
        if future_heatmap_loss is not None:
            output['future_heatmap_loss'] = future_heatmap_loss
            if future_heatmap_noise_std is not None:
                output['future_heatmap_noise_std'] = future_heatmap_noise_std
                output['future_heatmap_noise_pred_std'] = future_heatmap_noise_pred_std
        
        # Actions / Trajectory
        output['action_cond'] = action_cond
        
        if self.transformer_action_head is not None:
            output['has_transformer_action_head'] = True
            if not self.training and trajectory is not None:
                output['trajectory'] = trajectory
        elif self.action_head is not None:
            output['has_action_head'] = True
            if not self.training and actions is not None:
                output['actions'] = actions
        
        # Progress prediction
        if progress is not None:
            output['progress'] = progress
        
        # Stop prediction (legacy)
        if stop_logits is not None:
            output['stop_logits'] = stop_logits
            output['stop_prob'] = stop_prob
        
        if return_intermediate:
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
        Forward pass with packed batch (Sequence Packing mode).
        
        基于 Qwen3-VL 官方 fine-tuning 框架的 Sequence Packing 实现。
        所有样本被打包成一个长序列，使用 flash_attn_varlen_func 处理。
        
        Args:
            packed_batch: Dict from PackingCollatorForVLN, containing:
                - input_ids: (1, total_seq_len)
                - attention_mask: cumsum_seq_lens (num_samples + 1,)
                - position_ids: (3, 1, total_seq_len)
                - pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
                - seq_lens: List[int]
                - num_samples: int
                - current_frame: (B, C, H, W) for heatmap heads
                - heatmap, action, etc.
            
        Returns:
            Dictionary containing predictions and losses (same format as forward)
        """
        batch_size = packed_batch["num_samples"]
        seq_lens = packed_batch["seq_lens"]
        
        current_observation = packed_batch["current_frame"].to(self.device)
        
        if self.config.verbose:
            total_seq_len = packed_batch["input_ids"].shape[1]
            logger.info(f"[PACKED] Processing {batch_size} samples, total_seq_len={total_seq_len}")
        
        # ==================== Step 1: Qwen3-VL Processing (Packed) ====================
        qwen_output = self.qwen3_vl.forward_packed(
            packed_batch=packed_batch,
            return_hidden_states=True,
        )
        
        # Get hidden states: (num_samples, hidden_dim) after pooling
        raw_hidden_states = qwen_output.get('hidden_states')
        vision_hidden_states = qwen_output.get('vision_hidden_states')
        
        if raw_hidden_states is None:
            raise RuntimeError("Failed to extract hidden states from Qwen3-VL (packed mode)")
        
        # Move to device
        raw_hidden_states = raw_hidden_states.to(device=self.device, dtype=self.config.dtype)
        
        # 对于 packed mode，hidden_states 是 (B, hidden_dim)，需要 unsqueeze 成 (B, 1, hidden_dim)
        # 以匹配下游 heads 的期望输入格式
        if raw_hidden_states.dim() == 2:
            raw_hidden_states = raw_hidden_states.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # ==================== Step 2: Project Hidden States ====================
        llm_tokens = self.llm_projector(raw_hidden_states)  # [B, 1, llm_token_dim]
        
        if self.config.verbose:
            logger.info(f"[PACKED] LLM tokens shape: {llm_tokens.shape}")
        
        # ==================== Step 3: Heatmap Generation ====================
        # 使用 vision_hidden_states 如果可用（更丰富的视觉信息）
        if vision_hidden_states is not None:
            vision_hidden_states = vision_hidden_states.to(device=self.device, dtype=self.config.dtype)
            llm_tokens_for_heatmap = self.llm_projector(vision_hidden_states)
        else:
            llm_tokens_for_heatmap = llm_tokens
        
        history_heatmap = None
        future_heatmap = None
        history_heatmap_loss = None
        future_heatmap_loss = None
        history_heatmap_noise_std = None
        history_heatmap_noise_pred_std = None
        future_heatmap_noise_std = None
        future_heatmap_noise_pred_std = None
        history_heatmap_base_loss = None
        history_heatmap_focal_loss = None
        history_heatmap_visibility_loss = None
        
        if return_heatmaps:
            observation_for_heatmap = current_observation.to(dtype=self.config.dtype)
            llm_tokens_hm = llm_tokens_for_heatmap.to(dtype=self.config.dtype)
            
            # History Heatmap
            if self.history_heatmap_head is not None:
                if gt_history_heatmap is not None:
                    # 有 GT 时用噪声预测（训练和验证均可用，只需 1 次 UNet 前向）
                    gt_history_hm = gt_history_heatmap.to(self.device)
                    result = self.history_heatmap_head(
                        llm_tokens=llm_tokens_hm,
                        observation=observation_for_heatmap,
                        gt_heatmap=gt_history_hm,
                        return_loss=True,
                        skip_inference=not self.training,  # eval 模式跳过推理（验证加速）
                    )
                    history_heatmap_loss = result['loss']
                    history_heatmap = result.get('heatmap')
                    history_heatmap_noise_std = result.get('noise_std')
                    history_heatmap_noise_pred_std = result.get('noise_pred_std')
                    history_heatmap_base_loss = result.get('base_loss')
                    history_heatmap_focal_loss = result.get('focal_loss')
                    history_heatmap_visibility_loss = result.get('visibility_loss')
                else:
                    # 无 GT 时走完整扩散推理（纯推理/可视化）
                    history_heatmap = self.history_heatmap_head(
                        llm_tokens=llm_tokens_hm,
                        observation=observation_for_heatmap,
                    )
            
            # Future Heatmap
            if self.future_heatmap_head is not None:
                if gt_future_heatmap is not None:
                    # 有 GT 时用噪声预测
                    gt_future_hm = gt_future_heatmap.to(self.device)
                    result = self.future_heatmap_head(
                        llm_tokens=llm_tokens_hm,
                        observation=observation_for_heatmap,
                        gt_heatmap=gt_future_hm,
                        return_loss=True,
                        skip_inference=not self.training,  # eval 模式跳过推理
                    )
                    future_heatmap_loss = result['loss']
                    future_heatmap = result.get('heatmap')
                    future_heatmap_noise_std = result.get('noise_std')
                    future_heatmap_noise_pred_std = result.get('noise_pred_std')
                else:
                    future_heatmap = self.future_heatmap_head(
                        llm_tokens=llm_tokens_hm,
                        observation=observation_for_heatmap,
                    )
        
        # ==================== Step 4: Action Generation ====================
        actions = None
        trajectory = None
        action_cond = llm_tokens.mean(dim=1)  # [B, llm_token_dim]
        
        if return_actions:
            if self.transformer_action_head is not None:
                if not self.training:
                    trajectory = self.transformer_action_head.get_trajectory(llm_tokens)
            elif self.action_head is not None:
                if not self.training:
                    actions = self.action_head(action_cond)
        
        # ==================== Step 5: Stop/Progress Prediction ====================
        stop_logits = None
        stop_prob = None
        progress = None
        
        if self.progress_head is not None:
            progress = self.progress_head.get_progress(llm_tokens)
        
        if self.stop_head is not None:
            stop_cond = llm_tokens.mean(dim=1)
            stop_logits = self.stop_head.classifier(stop_cond).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)
        
        # ==================== Build Output ====================
        output = {
            'llm_tokens': llm_tokens,
            'processing_metadata': {
                'num_samples': batch_size,
                'seq_lens': seq_lens,
                'total_seq_len': packed_batch["input_ids"].shape[1],
                'llm_token_shape': llm_tokens.shape,
                'mode': 'packed',
            }
        }
        
        # Heatmaps
        if history_heatmap is not None:
            output['history_heatmaps'] = history_heatmap.unsqueeze(1)
        if future_heatmap is not None:
            output['future_heatmaps'] = future_heatmap.unsqueeze(1)
        
        # Heatmap losses
        if history_heatmap_loss is not None:
            output['history_heatmap_loss'] = history_heatmap_loss
            if history_heatmap_noise_std is not None:
                output['history_heatmap_noise_std'] = history_heatmap_noise_std
                output['history_heatmap_noise_pred_std'] = history_heatmap_noise_pred_std
            if history_heatmap_base_loss is not None:
                output['history_heatmap_base_loss'] = history_heatmap_base_loss
                output['history_heatmap_focal_loss'] = history_heatmap_focal_loss
            if history_heatmap_visibility_loss is not None:
                output['history_heatmap_visibility_loss'] = history_heatmap_visibility_loss
        if future_heatmap_loss is not None:
            output['future_heatmap_loss'] = future_heatmap_loss
            if future_heatmap_noise_std is not None:
                output['future_heatmap_noise_std'] = future_heatmap_noise_std
                output['future_heatmap_noise_pred_std'] = future_heatmap_noise_pred_std
        
        # Actions / Trajectory
        output['action_cond'] = action_cond
        
        if self.transformer_action_head is not None:
            output['has_transformer_action_head'] = True
            if not self.training and trajectory is not None:
                output['trajectory'] = trajectory
        elif self.action_head is not None:
            output['has_action_head'] = True
            if not self.training and actions is not None:
                output['actions'] = actions
        
        # Progress prediction
        if progress is not None:
            output['progress'] = progress
        
        # Stop prediction (legacy)
        if stop_logits is not None:
            output['stop_logits'] = stop_logits
            output['stop_prob'] = stop_prob
        
        if return_intermediate:
            output['intermediate_features'] = {
                'raw_hidden_states': raw_hidden_states,
                'qwen_output': qwen_output,
            }
        
        return output
    
    def update_heatmap_size(self, new_size: Tuple[int, int]):
        """Update heatmap size configuration (for curriculum training).
        
        This updates both the config and the actual heatmap heads to ensure
        the model generates heatmaps at the correct resolution.
        
        Args:
            new_size: New heatmap size as (H, W) tuple
        """
        old_size = self.config.heatmap_size
        self.config.heatmap_size = new_size
        
        # Update history heatmap head
        if self.history_heatmap_head is not None:
            self.history_heatmap_head.heatmap_size = new_size
            self.history_heatmap_head.config.heatmap_size = new_size
            logger.info(f"  ✓ History heatmap head: {old_size} → {new_size}")
        
        # Update future heatmap head
        if self.future_heatmap_head is not None:
            self.future_heatmap_head.heatmap_size = new_size
            self.future_heatmap_head.config.heatmap_size = new_size
            logger.info(f"  ✓ Future heatmap head: {old_size} → {new_size}")
        
        logger.info(f"Updated heatmap size: {old_size} → {new_size}")


def create_vln_pipeline(
    llm_model_path: str = "./models/qwen_3_vl",
    heatmap_size: Tuple[int, int] = (64, 64),
    device: str = "cuda",
    verbose: bool = True,
    **kwargs,
) -> VLNPipeline:
    """
    Factory function to create the VLN pipeline.
    
    Returns:
        Configured VLNPipeline instance
    """
    config = VLNPipelineConfig(
        llm_model_path=llm_model_path,
        heatmap_size=heatmap_size,
        device=device,
        verbose=verbose,
        **kwargs,
    )
    
    return VLNPipeline(config)
