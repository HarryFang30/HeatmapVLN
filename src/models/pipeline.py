"""
Spatial-MLLM Integration Compatibility Layer
==========================================

This module provides the final compatibility layer that integrates all
components of the VLN pipeline with the Spatial-MLLM architecture.

It bridges:
1. Space-aware frame sampling → Keyframe selection
2. VGGT 3D features + DINOv3 2D features → Feature fusion  
3. Fused features → Qwen2.5-VL LLM processing
4. LLM hidden states → First-person inter-frame heatmaps

Architecture Overview:
N_m frames → VGGT (all frames) → Space-aware sampling → N_k indices
N_k indices → VGGT features (3D path) + DINOv3 features (2D path)
→ Feature fusion → LLM → First-person heatmaps

This ensures the complete VLN pipeline works end-to-end with proper
compatibility across all components.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List, Union
import logging
from dataclasses import dataclass

from ..data import create_keyframe_selector
from .dinov3 import create_dinov3_compatibility_layer
from .vggt.models.vggt import VGGT
from .llm import create_real_llm_integration, create_memory_efficient_llm
from .action import DiffusionActionHead, DiffusionActionConfig, StopPredictionHead
from .heatmap import DiffusionHeatmapHead, DiffusionHeatmapConfig

logger = logging.getLogger(__name__)


@dataclass
class SpatialMLLMIntegrationConfig:
    """Configuration for complete Spatial-MLLM integration."""
    # Frame sampling configuration
    target_keyframes: int = 16  # N_k
    total_frames: int = 128  # N_m
    sampling_method: str = "hybrid"  # greedy_coverage, novelty_weighted, hybrid
    
    # Model configurations
    # NOTE: DINOv3 defaults aligned with models/dinov3/config.json
    dinov3_model_size: str = "7b"  # base, large, giant, 7b (local model is 7b)
    dinov3_patch_size: int = 16  # From config.json patch_size
    dinov3_img_size: int = 224  # From config.json image_size
    
    # VGGT configuration
    vggt_img_size: int = 518  # VGGT uses 518
    vggt_patch_size: int = 14  # VGGT uses 14
    vggt_embed_dim: int = 1024
    
    # Feature fusion configuration  
    feature_fusion_dim: int = 2048  # Dimension for fused 2D + 3D features
    fusion_method: str = "concatenate"  # concatenate, attention, mlp
    
    # Real LLM integration (Qwen2.5-VL)
    use_real_llm: bool = True  # ENABLED with multi-GPU distribution
    llm_model_path: str = "./models/qwen_2.5_vl"  # Local model path
    llm_use_vggt_model: bool = False  # Use standard Qwen2.5-VL (not VGGT-integrated)
    llm_memory_efficient: bool = True  # Load/unload LLM dynamically to save memory
    llm_token_dim: int = 1024  # Token dimension for Qwen2.5-VL
    llm_torch_dtype: str = "bfloat16"
    llm_attn_implementation: str = "flash_attention_2"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    enable_llm_projection: bool = True  # Fallback if real LLM fails
    
    # Heatmap generation
    heatmap_size: Tuple[int, int] = (224, 224)
    enable_inter_frame_heatmaps: bool = True
    
    # Action generation (Diffusion Policy)
    enable_action_head: bool = False  # Enable diffusion action head
    action_dim: int = 2  # 2D navigation (dx, dy)
    action_pred_horizon: int = 1  # Number of action steps to predict
    action_encoding_size: int = 256  # Condition projection dimension (简化：768 → 256)
    action_down_dims: List[int] = None  # UNet channel dims (默认使用 action_config.py 中的值)
    
    # Stop prediction (binary classifier)
    enable_stop_head: bool = False  # Enable stop prediction head
    stop_hidden_dim: int = 512  # Hidden dimension for stop classifier
    stop_focal_gamma: float = 2.0  # Focal loss gamma (focus on hard examples)
    stop_focal_alpha: float = 0.75  # Focal loss alpha (weight for STOP class)
    action_num_diffusion_iters: int = 10  # Diffusion denoising steps
    action_stats_min: List[float] = None  # Will use defaults if None
    action_stats_max: List[float] = None  # Will use defaults if None
    
    # Diffusion Heatmap Generation (dual-head architecture)
    enable_history_heatmap_head: bool = False  # Enable history diffusion heatmap head
    enable_future_heatmap_head: bool = False   # Enable future diffusion heatmap head
    diffusion_heatmap_cond_dim: int = 512  # Condition dimension for diffusion
    diffusion_heatmap_num_inference_steps: int = 10  # Diffusion inference steps
    
    # Performance settings
    device: str = "cuda"
    dtype: torch.dtype = torch.float32  # Match DINOv3 config.json torch_dtype
    dinov3_dtype: torch.dtype = torch.float32  # Separate dtype for DINOv3 (from config.json)
    vggt_compute_dtype: torch.dtype = torch.float32
    enable_gradient_checkpointing: bool = False

    # Multi-GPU settings
    use_multi_gpu: bool = True  # Distribute models across GPUs
    vggt_gpu: str = "cuda:0"  # VGGT on GPU 0
    dinov3_gpu: str = "cuda:1"  # DINOv3 on GPU 1
    llm_gpu: str = "cuda:2"  # LLM on GPU 2
    
    # Debug and logging
    verbose: bool = True
    save_intermediate_features: bool = False
    device_allocation: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.device_allocation:
            self.vggt_gpu = self.device_allocation.get('vggt', self.vggt_gpu)
            self.dinov3_gpu = self.device_allocation.get('dinov3', self.dinov3_gpu)
            self.llm_gpu = self.device_allocation.get('llm', self.llm_gpu)
            if 'device' in self.device_allocation:
                self.device = self.device_allocation['device']


class SpatialMLLMPipeline(nn.Module):
    """
    Complete Spatial-MLLM pipeline with VLN integration.
    
    This class orchestrates the entire pipeline from video input to
    first-person inter-frame heatmap generation, ensuring compatibility
    across all components.
    """
    
    def __init__(self, config: SpatialMLLMIntegrationConfig):
        super().__init__()
        self.config = config
        # In DDP mode (use_multi_gpu=False), use rank-specific device from config
        # In multi-GPU mode (use_multi_gpu=True), use explicit GPU assignments
        self.device = torch.device(config.device)

        # 🔍 DEBUG: Print device configuration
        print(f"[DEBUG] SpatialMLLMPipeline.__init__")
        print(f"[DEBUG] config.device: {config.device}")
        print(f"[DEBUG] config.use_multi_gpu: {config.use_multi_gpu}")
        print(f"[DEBUG] config.vggt_gpu: {config.vggt_gpu}")
        print(f"[DEBUG] config.dinov3_gpu: {config.dinov3_gpu}")
        print(f"[DEBUG] config.llm_gpu: {config.llm_gpu}")
        print(f"[DEBUG] self.device: {self.device}")

        # Initialize VGGT for 3D geometry processing (all N_m frames)
        # In DDP mode: all modules on same device (self.device)
        # In multi-GPU mode: on dedicated GPU
        vggt_device = torch.device(config.vggt_gpu if config.use_multi_gpu else config.device)
        print(f"[DEBUG] vggt_device will be: {vggt_device}")
        try:
            # Try to load pretrained VGGT from local model directory
            from ..utils.path_utils import resolve_model_path
            vggt_model_path = "./models/vggt"
            try:
                resolved_vggt_path = resolve_model_path(vggt_model_path, "VGGT")
                self.vggt = VGGT.from_pretrained(str(resolved_vggt_path)).to(device=vggt_device)
                print(f"Loaded pretrained VGGT from {resolved_vggt_path} on {vggt_device}")
            except FileNotFoundError:
                print(f"VGGT model not found at {vggt_model_path}, will use random initialization")
                raise  # Re-raise to trigger fallback
        except Exception as e:
            print(f"Could not load pretrained VGGT from local path: {e}")
            # Fallback to random initialization
            self.vggt = VGGT(
                img_size=config.vggt_img_size,
                patch_size=config.vggt_patch_size,
                embed_dim=config.vggt_embed_dim
            ).to(device=vggt_device)
            print(f"Using randomly initialized VGGT weights on {vggt_device}")
        
        # Initialize keyframe selector with space-aware sampling
        self.keyframe_selector = create_keyframe_selector(
            target_keyframes=config.target_keyframes,
            total_frames=config.total_frames,
            sampling_method=config.sampling_method,
            device=config.device,
            verbose=config.verbose
        )
        
        # Initialize DINOv3 for 2D semantic processing (N_k keyframes only) on dedicated GPU
        dinov3_device = config.dinov3_gpu if config.use_multi_gpu else config.device
        # Note: The actual model in /models/dinov3 is 7B size (4096 hidden, 40 layers)
        # The compatibility layer will automatically load from the local safetensors
        # Config values are aligned with models/dinov3/config.json
        self.dinov3_compat = create_dinov3_compatibility_layer(
            model_size=config.dinov3_model_size,  # "7b" by default
            patch_size=config.dinov3_patch_size,  # 16 (from config.json)
            img_size=config.dinov3_img_size,  # 224 (from config.json)
            target_embed_dim=config.vggt_embed_dim,  # Match VGGT dimensions
            align_with_vggt=True,
            device=dinov3_device,
            dtype=config.dinov3_dtype  # Use separate DINOv3 dtype (float32 from config.json)
        )
        
        # Initialize feature fusion module (放在vggt_gpu上，因为VGGT占用小，llm_gpu已被Qwen占满)
        fusion_device = torch.device(config.vggt_gpu if config.use_multi_gpu else config.device)
        
        # VGGT feature projection layer (registered as model parameter)
        # Projects 3D world_points (3 channels) to target dimension (2 * embed_dim)
        # This MUST be registered as nn.Module to participate in gradient updates
        vggt_world_points_dim = 3  # world_points has 3 channels (x, y, z)
        vggt_target_dim = config.vggt_embed_dim * 2  # 2048 for embed_dim=1024
        self.vggt_feature_projection = nn.Sequential(
            nn.Linear(vggt_world_points_dim, config.vggt_embed_dim),
            nn.LayerNorm(config.vggt_embed_dim),
            nn.GELU(),
            nn.Linear(config.vggt_embed_dim, vggt_target_dim),
            nn.LayerNorm(vggt_target_dim),
        ).to(device=fusion_device, dtype=config.dtype)
        logger.info(f"Registered VGGT feature projection: {vggt_world_points_dim} -> {vggt_target_dim}")
        
        self.feature_fusion = self._create_feature_fusion_module().to(device=fusion_device, dtype=config.dtype)
        
        # Initialize REAL LLM integration (Qwen2.5-VL)
        if config.use_real_llm:
            if config.llm_memory_efficient:
                logger.info("Initializing MEMORY-EFFICIENT Qwen2.5-VL integration")
                llm_device = config.llm_gpu if config.use_multi_gpu else config.device
                self.llm_integration = create_memory_efficient_llm(
                    model_path=config.llm_model_path,
                    use_vggt_model=config.llm_use_vggt_model,
                    device=llm_device,
                    torch_dtype=config.llm_torch_dtype
                )
            else:
                logger.info("Initializing REAL Qwen2.5-VL integration")
                llm_device = config.llm_gpu if config.use_multi_gpu else config.device
                self.llm_integration = create_real_llm_integration(
                    model_path=config.llm_model_path,
                    use_vggt_model=config.llm_use_vggt_model,
                    device=llm_device,
                    torch_dtype=config.llm_torch_dtype
                )
            # Keep projector as fallback (放在fusion同一GPU上，即vggt_gpu)
            projector_device = fusion_device  # 与fusion module在同一GPU
            if config.enable_llm_projection:
                self.llm_projector = self._create_llm_projection_module().to(device=projector_device, dtype=config.dtype)
            else:
                self.llm_projector = nn.Identity().to(device=projector_device, dtype=config.dtype)
        else:
            logger.warning("Using FAKE LLM projection - not real LLM processing!")
            self.llm_integration = None
            projector_device = fusion_device  # 与fusion module在同一GPU
            if config.enable_llm_projection:
                self.llm_projector = self._create_llm_projection_module().to(device=projector_device, dtype=config.dtype)
            else:
                self.llm_projector = nn.Identity().to(device=projector_device, dtype=config.dtype)
            
        # Initialize Diffusion Action Head (parallel with heatmap output)
        if config.enable_action_head:
            logger.info("Initializing Diffusion Action Head for navigation actions")
            action_device = torch.device(config.llm_gpu if config.use_multi_gpu else config.device)
            
            # 构建 action config 参数
            action_config_kwargs = {
                'action_dim': config.action_dim,
                'pred_horizon': config.action_pred_horizon,
                'cond_dim': config.llm_token_dim,  # Use projected LLM tokens as condition
                'encoding_size': config.action_encoding_size,
                'num_diffusion_iters': config.action_num_diffusion_iters,
                # 🔧 使用实际数据集统计值（加 10% 余量）
                'action_stats_min': config.action_stats_min or [-0.17, -0.03],
                'action_stats_max': config.action_stats_max or [0.19, 0.31],
                'device': str(action_device),
            }
            # 可选：传递 down_dims（如果在 config 中指定）
            if config.action_down_dims is not None:
                action_config_kwargs['down_dims'] = config.action_down_dims
            
            action_config = DiffusionActionConfig(**action_config_kwargs)
            self.action_head = DiffusionActionHead(action_config).to(device=action_device, dtype=config.dtype)
            logger.info(f"Diffusion Action Head initialized on {action_device}, down_dims={action_config.down_dims}")
        else:
            self.action_head = None
        
        # Initialize Stop Prediction Head (binary classifier for STOP action)
        if config.enable_stop_head:
            logger.info("Initializing Stop Prediction Head")
            stop_device = torch.device(config.llm_gpu if config.use_multi_gpu else config.device)
            self.stop_head = StopPredictionHead(
                input_dim=config.llm_token_dim,
                hidden_dim=config.stop_hidden_dim,
                dropout=0.1,
                focal_gamma=config.stop_focal_gamma,
                focal_alpha=config.stop_focal_alpha,
            ).to(device=stop_device, dtype=config.dtype)
            logger.info(f"Stop Prediction Head initialized on {stop_device}")
        else:
            self.stop_head = None
        
        # Initialize Dual Diffusion Heatmap Heads (history and future)
        heatmap_device = torch.device(config.llm_gpu if config.use_multi_gpu else config.device)
        diffusion_heatmap_config = DiffusionHeatmapConfig(
            llm_dim=config.llm_token_dim,
            cond_dim=config.diffusion_heatmap_cond_dim,
            heatmap_size=config.heatmap_size,
            num_inference_steps=config.diffusion_heatmap_num_inference_steps,
            image_size=(config.dinov3_img_size, config.dinov3_img_size),
        )
        
        # History Heatmap Head
        if config.enable_history_heatmap_head:
            logger.info("Initializing History Diffusion Heatmap Head")
            self.history_heatmap_head = DiffusionHeatmapHead(diffusion_heatmap_config).to(
                device=heatmap_device, dtype=config.dtype
            )
            logger.info(f"History Heatmap Head initialized on {heatmap_device}")
        else:
            self.history_heatmap_head = None
            
        # Future Heatmap Head
        if config.enable_future_heatmap_head:
            logger.info("Initializing Future Diffusion Heatmap Head")
            self.future_heatmap_head = DiffusionHeatmapHead(diffusion_heatmap_config).to(
                device=heatmap_device, dtype=config.dtype
            )
            logger.info(f"Future Heatmap Head initialized on {heatmap_device}")
        else:
            self.future_heatmap_head = None
            
        # Performance optimization
        if config.enable_gradient_checkpointing:
            self.vggt.gradient_checkpointing_enable()
            
    def _create_feature_fusion_module(self) -> nn.Module:
        """Create module for fusing 3D VGGT and 2D DINOv3 features."""
        
        vggt_dim = self.config.vggt_embed_dim * 2  # VGGT outputs 2*embed_dim
        dinov3_dim = self.config.vggt_embed_dim * 2  # DINOv3 compat aligns to this
        fusion_dim = self.config.feature_fusion_dim
        
        if self.config.fusion_method == "concatenate":
            return nn.Sequential(
                nn.LayerNorm(vggt_dim + dinov3_dim),
                nn.Linear(vggt_dim + dinov3_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )
        elif self.config.fusion_method == "attention":
            return SpatialAttentionFusion(
                vggt_dim=vggt_dim,
                dinov3_dim=dinov3_dim,
                output_dim=fusion_dim
            )
        elif self.config.fusion_method == "mlp":
            return SpatialMLPFusion(
                vggt_dim=vggt_dim,
                dinov3_dim=dinov3_dim,
                output_dim=fusion_dim
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
    
    def _create_llm_projection_module(self) -> nn.Module:
        """Create projection layer for LLM token compatibility."""
        
        fusion_dim = self.config.feature_fusion_dim
        llm_dim = self.config.llm_token_dim
        
        return nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(llm_dim)
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
        action_valid: Optional[torch.Tensor] = None,  # action mask for continuous actions
        gt_stop: Optional[torch.Tensor] = None,  # 🆕 ground truth stop labels (0/1)
    ) -> Dict[str, Any]:
        """
        Complete forward pass of the Spatial-MLLM pipeline processing three inputs.
        
        This method processes:
        1. Current observation (first-person view)
        2. Feature tokens from VGGT and DINOv3 (spatial understanding)  
        3. Language instructions (navigation commands)
        
        Args:
            video_frames: Input video [B, N_m, C, H, W]
            instruction_text: VLN instruction text
            current_observation: Current first-person view [B, C, H, W] (optional, uses first frame if None)
            return_intermediate: Return intermediate features
            return_heatmaps: Generate single first-person inter-frame heatmap
            
        Returns:
            Dictionary containing:
                - 'selected_keyframes': Selected keyframe indices [N_k]
                - 'fused_features': Fused spatial features [B, N_k, D]
                - 'llm_tokens': LLM-compatible tokens [B, N_k, D_llm]
                - 'inter_frame_heatmap': Single first-person heatmap [B, 1, H, W] (if enabled)
                - 'actions': Predicted navigation actions [B, pred_horizon, action_dim] (if enabled)
                - 'action_loss': Action prediction loss (if gt_actions provided)
                - 'intermediate_features': Debug information (if requested)
                - 'processing_metadata': Pipeline statistics
        """
        
        if video_frames.shape[1] > self.config.total_frames:
            video_frames = video_frames[:, :self.config.total_frames]
        total_frames = video_frames.shape[1]
        
        # Set current observation (use last frame if not provided - represents current view)
        if current_observation is None:
            last_idx = total_frames - 1
            current_observation = video_frames[:, last_idx]  # [B, C, H, W]
            if self.config.verbose:
                logger.info(f"Using last frame (index {last_idx}) as current observation")
        
        if self.config.verbose:
            logger.info(f"Processing video: {video_frames.shape}")
            logger.info(f"Current observation: {current_observation.shape}")
            logger.info(f"Instruction: {instruction_text}")
            
        # Step 1: Process ALL frames through VGGT for geometry extraction
        logger.info("Step 1: VGGT processing for geometry extraction")
        vggt_device = torch.device(self.config.vggt_gpu if self.config.use_multi_gpu else self.config.device)
        video_frames_vggt = video_frames.to(device=vggt_device)
        vggt_predictions = self._process_all_frames_vggt(video_frames_vggt)
        
        # Step 2: Apply space-aware keyframe selection  
        logger.info("Step 2: Space-aware keyframe selection")
        keyframe_result = self.keyframe_selector(
            vggt_predictions=vggt_predictions,
            original_frames=video_frames
        )
        selected_indices = keyframe_result['keyframe_indices']
        selected_frames = keyframe_result.get('original_frames')
        
        if self.config.verbose:
            # Dynamic count handling for list or list[list] structures
            idx = selected_indices
            if isinstance(idx, (list, tuple)) and idx and isinstance(idx[0], (list, tuple)):
                count = len(idx[0])
            else:
                count = len(idx)
            logger.info("Selected %d keyframes: %s", count, idx.tolist() if hasattr(idx, 'tolist') else idx)
        
        # Step 3: Dual-path feature extraction
        logger.info("Step 3: Dual-path feature extraction")
        
        # 3D path: Index-selected VGGT features (pre-computed)
        vggt_features = keyframe_result['vggt_features']
        # Move VGGT features to main device for fusion (handle dict of tensors)
        if isinstance(vggt_features, dict):
            vggt_features = {k: v.to(device=self.device, dtype=self.config.dtype) if isinstance(v, torch.Tensor) else v
                           for k, v in vggt_features.items()}
        else:
            vggt_features = vggt_features.to(device=self.device, dtype=self.config.dtype)
        vggt_spatial_tokens = self._extract_vggt_spatial_features(vggt_features)
        
        # 2D path: Process selected frames through DINOv3 on dedicated GPU
        if selected_frames is not None:
            dinov3_device = torch.device(self.config.dinov3_gpu if self.config.use_multi_gpu else self.config.device)
            selected_frames_dinov3 = selected_frames.to(device=dinov3_device, dtype=self.config.dtype)
            dinov3_result = self.dinov3_compat(selected_frames_dinov3, return_features=True)
            dinov3_features = dinov3_result['vggt_aligned_features']  # Already aligned
            # Move features back to main device for fusion
            dinov3_features = dinov3_features.to(device=self.device)
        else:
            # Fallback: use VGGT features only
            logger.warning("No original frames available, using VGGT features only")
            dinov3_features = vggt_spatial_tokens
            
        # Step 4: Feature fusion (3D + 2D)
        logger.info("Step 4: Spatial feature fusion")
        fused_features = self._fuse_spatial_features(vggt_spatial_tokens, dinov3_features)
        
        # Step 5: REAL LLM processing (Qwen2.5-VL)
        logger.info("Step 5: REAL LLM spatial reasoning")
        if self.llm_integration is not None and self.config.use_real_llm:
            # Use REAL Qwen2.5-VL model for spatial reasoning
            logger.info("Processing through REAL Qwen2.5-VL model")
            try:
                # Move data to LLM device if using multi-GPU
                if self.config.use_multi_gpu:
                    llm_device = torch.device(self.config.llm_gpu)
                    fused_features_llm = fused_features.to(device=llm_device)
                    current_obs_llm = current_observation.to(device=llm_device) if current_observation is not None else None
                    video_frames_llm = (selected_frames if selected_frames is not None
                                      else video_frames[:, selected_indices]).to(device=llm_device)
                else:
                    fused_features_llm = fused_features
                    current_obs_llm = current_observation
                    video_frames_llm = selected_frames if selected_frames is not None else video_frames[:, selected_indices]

                llm_result = self.llm_integration(
                    fused_features=fused_features_llm,
                    instruction_text=instruction_text or "Analyze spatial relationships between video frames",
                    current_observation=current_obs_llm if current_obs_llm is not None else video_frames_llm[:, -1],
                    video_frames=video_frames_llm,
                    return_hidden_states=True
                )
                # IMPORTANT: Apply projection to get correct dimensions for heatmap converter
                raw_llm_tokens = llm_result['llm_tokens']
                # Move LLM tokens back to main device and apply projection
                raw_llm_tokens = raw_llm_tokens.to(device=self.device)
                llm_tokens = self.llm_projector(raw_llm_tokens)  # 2048 -> 1024 for compatibility

                # 🎯 SUCCESS: Real LLM processing completed
                generated_text = llm_result.get('llm_output', '')
                text_preview = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                logger.info(f"🎯 SUCCESS: Real Qwen2.5-VL spatial reasoning completed")
                logger.info(f"   📝 Generated text: {text_preview}")
                logger.info(f"   🧠 Using authentic LLM hidden states for heatmap generation")

            except Exception as e:
                # ⚠️ FAILURE: Real LLM processing failed, using projection fallback
                logger.error(f"⚠️ FAILURE: Real LLM processing failed: {e}")
                logger.warning(f"   🔄 Falling back to spatial feature projection (no genuine LLM reasoning)")
                logger.warning(f"   ⚡ Heatmaps will be generated from VGGT+DINOv3 features only")
                llm_tokens = self.llm_projector(fused_features)
        else:
            # 🔧 FALLBACK: No real LLM integration available
            logger.warning("🔧 FALLBACK: No real LLM integration available")
            logger.warning("   📊 Using spatial feature projection only (no language-enhanced reasoning)")
            logger.warning("   🎯 Heatmaps based on pure VGGT+DINOv3 spatial features")
            llm_tokens = self.llm_projector(fused_features)
        
        # Step 6: Generate heatmaps using Dual Diffusion Heatmap Heads
        history_heatmap = None
        future_heatmap = None
        
        if return_heatmaps:
            # Prepare inputs for heatmap heads
            if self.history_heatmap_head is not None or self.future_heatmap_head is not None:
                # Get device from whichever head is available
                if self.history_heatmap_head is not None:
                    heatmap_device = next(self.history_heatmap_head.parameters()).device
                else:
                    heatmap_device = next(self.future_heatmap_head.parameters()).device
                    
                llm_tokens_for_heatmap = llm_tokens.to(heatmap_device)
                observation_for_heatmap = current_observation.to(heatmap_device)
            
            # Generate History Heatmap
            if self.history_heatmap_head is not None:
                logger.info("Step 6a: History heatmap generation")
                history_heatmap = self.history_heatmap_head(
                    llm_tokens=llm_tokens_for_heatmap,
                    observation=observation_for_heatmap,
                )  # [B, Hm, Wm]
                logger.info(f"History heatmap generated: {history_heatmap.shape}")
            
            # Generate Future Heatmap
            if self.future_heatmap_head is not None:
                logger.info("Step 6b: Future heatmap generation")
                future_heatmap = self.future_heatmap_head(
                    llm_tokens=llm_tokens_for_heatmap,
                    observation=observation_for_heatmap,
                )  # [B, Hm, Wm]
                logger.info(f"Future heatmap generated: {future_heatmap.shape}")
        
        # Prepare output
        output = {
            'selected_keyframes': selected_indices,
            'fused_features': fused_features,
            'llm_tokens': llm_tokens,
            'processing_metadata': {
                'num_input_frames': total_frames,
                'num_selected_keyframes': len(selected_indices),
                'current_observation_shape': current_observation.shape,
                'current_observation_frame_idx': total_frames - 1,
                'current_observation_method': 'last_frame',
                'instruction_provided': instruction_text is not None,
                'sampling_method': self.config.sampling_method,
                'fusion_method': self.config.fusion_method,
                'vggt_dimensions': vggt_spatial_tokens.shape,
                'dinov3_dimensions': dinov3_features.shape,
                'fused_dimensions': fused_features.shape,
                'llm_token_dimensions': llm_tokens.shape
            }
        }
        
        # Add heatmaps to output
        if history_heatmap is not None:
            output['history_heatmaps'] = history_heatmap.unsqueeze(1)  # [B, 1, Hm, Wm]
        if future_heatmap is not None:
            output['future_heatmaps'] = future_heatmap.unsqueeze(1)    # [B, 1, Hm, Wm]
            
        # Step 7: Generate navigation actions using Diffusion Policy (parallel with heatmaps)
        if return_actions and self.action_head is not None:
            logger.info("Step 7: Diffusion Policy action generation")
            try:
                # Use LLM tokens as condition for action generation (same as heatmap heads)
                # This ensures actions are based on language-enhanced spatial reasoning
                # Pool over spatial and temporal dimensions: [B, N_k, L, D] -> [B, D]
                action_cond = llm_tokens.mean(dim=(1, 2))  # [B, D]
                
                # Move to action head device
                action_device = next(self.action_head.parameters()).device
                action_cond = action_cond.to(device=action_device)
                
                if gt_actions is not None:
                    # Training mode: compute action loss with masking
                    gt_actions_device = gt_actions.to(device=action_device)
                    # 🆕 传递 action_valid mask 给 action_head
                    action_valid_device = action_valid.to(device=action_device) if action_valid is not None else None
                    action_result = self.action_head(
                        action_cond, 
                        gt_actions=gt_actions_device,
                        action_valid=action_valid_device,  # 🆕 传递 mask
                        return_loss=True
                    )
                    output['actions'] = action_result['actions']
                    output['action_loss'] = action_result['loss']
                    output['normalized_actions'] = action_result['normalized_actions']
                    logger.info(f"Action loss: {action_result['loss'].item():.4f}")
                else:
                    # Inference mode: generate actions
                    actions = self.action_head(action_cond)
                    output['actions'] = actions
                    logger.info(f"Generated actions shape: {actions.shape}")
                    
            except Exception as e:
                logger.error(f"Action generation failed: {e}")
                output['actions'] = None
                output['action_loss'] = None
        
        # Step 8: Stop prediction (binary classification)
        if self.stop_head is not None:
            logger.info("Step 8: Stop prediction")
            try:
                # Use pooled LLM tokens as condition
                stop_cond = llm_tokens.mean(dim=(1, 2))  # [B, D]
                stop_device = next(self.stop_head.parameters()).device
                stop_cond = stop_cond.to(device=stop_device)
                
                if gt_stop is not None:
                    # Training mode: compute stop loss
                    gt_stop_device = gt_stop.to(device=stop_device)
                    action_valid_device = action_valid.to(device=stop_device) if action_valid is not None else None
                    stop_result = self.stop_head(
                        stop_cond,
                        gt_stop=gt_stop_device,
                        action_valid=action_valid_device,
                        return_loss=True,
                    )
                    output['stop_prob'] = stop_result['stop_prob']
                    output['stop_loss'] = stop_result['loss']
                    logger.info(f"Stop loss: {stop_result['loss'].item():.4f}")
                else:
                    # Inference mode: predict stop
                    stop_prob = self.stop_head(stop_cond)
                    output['stop_prob'] = stop_prob
                    logger.info(f"Stop probabilities: mean={stop_prob.mean().item():.4f}")
                    
            except Exception as e:
                logger.error(f"Stop prediction failed: {e}")
                output['stop_prob'] = None
                output['stop_loss'] = None
        
        if return_intermediate:
            output['intermediate_features'] = {
                'vggt_predictions': vggt_predictions,
                'keyframe_selection_result': keyframe_result,
                'vggt_spatial_tokens': vggt_spatial_tokens,
                'dinov3_features': dinov3_features,
                'raw_fused_features': fused_features
            }
            
        if self.config.verbose:
            logger.info("Spatial-MLLM pipeline completed successfully")
            
        return output
    
    def _process_all_frames_vggt(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process all N_m frames through VGGT for geometry extraction."""
        
        batch_size, num_frames = video_frames.shape[:2]
        
        # Reshape for VGGT processing: [B*N_m, C, H, W]
        frames_flat = video_frames.view(-1, *video_frames.shape[2:])
        
        autocast_enabled = self.config.vggt_compute_dtype in (torch.float16, torch.bfloat16)
        with torch.amp.autocast('cuda', enabled=autocast_enabled, dtype=self.config.vggt_compute_dtype):
            # VGGT processes all frames for geometry
            vggt_output = self.vggt(frames_flat)
            
        # VGGT outputs are already correctly structured with batch-frame dimensions
        # Check each tensor and only reshape if necessary
        for key, tensor in vggt_output.items():
            if self._tensor_needs_reshaping(tensor, batch_size, num_frames, key):
                try:
                    reshaped_tensor = self._reshape_vggt_tensor(tensor, batch_size, num_frames, key)
                    vggt_output[key] = reshaped_tensor
                    if self.config.verbose:
                        print(f"INFO: Reshaped {key}: {tensor.shape} → {reshaped_tensor.shape}")
                except RuntimeError as e:
                    if self.config.verbose:
                        print(f"INFO: Failed to reshape {key}: {tensor.shape} - {e}")
            else:
                if self.config.verbose:
                    print(f"INFO: {key} already correctly structured: {tensor.shape}")
                
        return vggt_output
    
    def _tensor_needs_reshaping(self, tensor: torch.Tensor, batch_size: int, num_frames: int, key: str) -> bool:
        """
        Determine if a VGGT output tensor needs reshaping.
        
        VGGT processes flattened frames [B*N_frames, C, H, W] and may output tensors
        in [1, B*N_frames, ...] format that need to be reshaped to [B, N_frames, ...].
        """
        
        # Check if tensor already has the expected batch-frame structure
        if len(tensor.shape) >= 2:
            # Expected: first dim = batch_size, second dim = num_frames
            if tensor.shape[0] == batch_size and tensor.shape[1] == num_frames:
                return False  # Already correctly structured
            
            # Check for VGGT's common output pattern: [1, B*N_frames, ...]
            # This happens when VGGT treats B*N_frames as a sequence in a single batch
            if tensor.shape[0] == 1 and tensor.shape[1] == batch_size * num_frames:
                return True  # Needs reshaping from [1, B*N_frames, ...] to [B, N_frames, ...]
        
        # Check for tensors that might need special handling
        return True
    
    def _reshape_vggt_tensor(self, tensor: torch.Tensor, batch_size: int, num_frames: int, key: str) -> torch.Tensor:
        """
        Reshape a VGGT tensor that needs reshaping.
        
        Handles the common VGGT output pattern: [1, B*N_frames, ...] → [B, N_frames, ...]
        """
        
        total_expected_frames = batch_size * num_frames
        
        # Handle VGGT's common pattern: [1, B*N_frames, ...] 
        if (len(tensor.shape) >= 2 and 
            tensor.shape[0] == 1 and 
            tensor.shape[1] == total_expected_frames):
            
            # Remove the first dimension and reshape: [B*N_frames, ...] → [B, N_frames, ...]
            tensor_flat = tensor.squeeze(0)  # [B*N_frames, ...]
            remaining_shape = tensor_flat.shape[1:]
            new_shape = (batch_size, num_frames) + remaining_shape
            return tensor_flat.view(new_shape)
        
        # Handle case where tensor is already flattened: [B*N_frames, ...]
        if tensor.shape[0] == total_expected_frames:
            # Tensor is in [B*N_frames, ...] format, reshape to [B, N_frames, ...]
            remaining_shape = tensor.shape[1:]
            new_shape = (batch_size, num_frames) + remaining_shape
            return tensor.view(new_shape)
        
        # If we can't determine how to reshape, return original
        return tensor
    
    def _extract_vggt_spatial_features(self, vggt_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract spatial features from VGGT predictions.
        
        Uses the registered vggt_feature_projection layer (not dynamically created).
        This ensures the projection weights are trained and consistent across calls.
        
        Args:
            vggt_features: Dict containing 'world_points' with shape [B, N_k, H, W, 3]
            
        Returns:
            Tensor of shape [B, N_k, H*W, 2*embed_dim] with projected spatial features
        """
        # Use world points as primary spatial features
        # Shape: [B, N_k, H, W, 3] → [B, N_k, H*W, 3]
        world_points = vggt_features['world_points']
        batch_size, num_keyframes, height, width = world_points.shape[:4]
        
        # Store original spatial dimensions for later alignment
        self._vggt_spatial_size = (height, width)
        
        # Flatten spatial dimensions: [B, N_k, H, W, 3] -> [B, N_k, H*W, 3]
        spatial_features = world_points.view(batch_size, num_keyframes, height * width, -1)
        
        # Move to projection layer device and project using REGISTERED layer
        projection_device = next(self.vggt_feature_projection.parameters()).device
        projection_dtype = next(self.vggt_feature_projection.parameters()).dtype
        spatial_features = spatial_features.to(device=projection_device, dtype=projection_dtype)
        
        # Apply the registered projection layer (trainable parameters!)
        # [B, N_k, H*W, 3] -> [B, N_k, H*W, 2*embed_dim]
        spatial_features = self.vggt_feature_projection(spatial_features)
        
        if self.config.verbose:
            logger.info(f"VGGT spatial features: {spatial_features.shape} (from {height}x{width} grid)")
            
        return spatial_features
    
    def _fuse_spatial_features(
        self,
        vggt_features: torch.Tensor,
        dinov3_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse 3D VGGT and 2D DINOv3 features with proper spatial alignment.
        
        Handles the spatial resolution mismatch between VGGT (typically 37×37) 
        and DINOv3 (typically 14×14) by aligning to a common resolution.
        
        Args:
            vggt_features: [B, N_k, L_vggt, D_vggt] - VGGT spatial features
            dinov3_features: [B, N_k, L_dinov3, D_dinov3] - DINOv3 spatial features
            
        Returns:
            Fused features [B, N_k, L_fusion, D_fusion]
        """
        import torch.nn.functional as F  # Local import to avoid global namespace clutter

        # Move features to fusion device
        fusion_device = next(self.feature_fusion.parameters()).device
        if vggt_features.device != fusion_device:
            vggt_features = vggt_features.to(fusion_device, dtype=self.config.dtype)
        if dinov3_features.device != fusion_device:
            dinov3_features = dinov3_features.to(fusion_device, dtype=self.config.dtype)

        # Extract dimensions
        batch_size, num_keyframes = vggt_features.shape[:2]
        vggt_spatial_dim = vggt_features.shape[2]
        dinov3_spatial_dim = dinov3_features.shape[2]
        vggt_feat_dim = vggt_features.shape[-1]
        dinov3_feat_dim = dinov3_features.shape[-1]
        
        if self.config.verbose:
            logger.info(f"Fusion input - VGGT: {vggt_features.shape}, DINOv3: {dinov3_features.shape}")

        if self.config.fusion_method == "concatenate":
            # Align spatial resolutions to DINOv3's resolution (typically smaller, e.g., 14×14)
            # This preserves DINOv3's semantic features at full resolution
            # while downsampling VGGT's denser geometric features
            
            # Calculate grid sizes
            vggt_side = int(vggt_spatial_dim ** 0.5)
            dinov3_side = int(dinov3_spatial_dim ** 0.5)
            
            # Use DINOv3's resolution as target (or cap at 14×14 for memory safety)
            SAFE_MAX_SIDE = 14
            fusion_side = min(dinov3_side, SAFE_MAX_SIDE)
            
            if self.config.verbose:
                logger.info(f"Spatial alignment: VGGT {vggt_side}x{vggt_side} + DINOv3 {dinov3_side}x{dinov3_side} -> {fusion_side}x{fusion_side}")

            # Reshape VGGT to spatial grid and downsample
            # [B, N_k, L, D] -> [B*N_k, D, H, W] -> pool -> [B*N_k, D, fusion_side, fusion_side]
            vggt_img = vggt_features.view(batch_size * num_keyframes, vggt_side, vggt_side, vggt_feat_dim)
            vggt_img = vggt_img.permute(0, 3, 1, 2).contiguous()  # [B*N_k, D, H, W]
            vggt_resized = F.adaptive_avg_pool2d(vggt_img, (fusion_side, fusion_side))

            # Reshape DINOv3 to spatial grid and resize if needed
            # [B, N_k, L, D] -> [B*N_k, D, H, W]
            dinov3_img = dinov3_features.view(batch_size * num_keyframes, dinov3_side, dinov3_side, dinov3_feat_dim)
            dinov3_img = dinov3_img.permute(0, 3, 1, 2).contiguous()  # [B*N_k, D, H, W]
            
            if dinov3_side != fusion_side:
                dinov3_resized = F.interpolate(
                    dinov3_img, size=(fusion_side, fusion_side),
                    mode='bilinear', align_corners=False
                )
            else:
                dinov3_resized = dinov3_img

            # Concatenate channel-wise: [B*N_k, D_vggt + D_dinov3, H, W]
            fused_img = torch.cat([vggt_resized, dinov3_resized], dim=1)
            
            # Reshape back to sequence format for fusion MLP
            # [B*N_k, D_total, H, W] -> [B*N_k, H, W, D_total] -> [B, N_k, H*W, D_total]
            fused_flat = fused_img.permute(0, 2, 3, 1).contiguous()
            fused_flat = fused_flat.view(batch_size, num_keyframes, fusion_side * fusion_side, -1)
            
            if self.config.verbose:
                logger.info(f"Fused features before MLP: {fused_flat.shape}")

            return self.feature_fusion(fused_flat)
        else:
            # Use specialized fusion module (attention or mlp)
            # These modules need to handle spatial alignment internally
            return self.feature_fusion(vggt_features, dinov3_features)
    
    def update_heatmap_size(self, new_size: Tuple[int, int]):
        """Update heatmap size configuration (for curriculum training)."""
        self.config.heatmap_size = new_size
        logger.info(f"Updated heatmap size to {new_size}")


class SpatialAttentionFusion(nn.Module):
    """Attention-based fusion of 3D VGGT and 2D DINOv3 spatial features.
    
    Uses cross-attention where DINOv3 features query VGGT features,
    allowing semantic features to attend to geometric information.
    """
    
    def __init__(self, vggt_dim: int, dinov3_dim: int, output_dim: int):
        super().__init__()
        self.vggt_dim = vggt_dim
        self.dinov3_dim = dinov3_dim
        self.output_dim = output_dim
        
        # Common dimension for cross-attention
        self.attn_dim = min(vggt_dim, dinov3_dim)
        
        # Project both to common dimension for attention
        self.vggt_proj = nn.Linear(vggt_dim, self.attn_dim) if vggt_dim != self.attn_dim else nn.Identity()
        self.dinov3_proj = nn.Linear(dinov3_dim, self.attn_dim) if dinov3_dim != self.attn_dim else nn.Identity()
        
        # Cross-attention: DINOv3 (query) attends to VGGT (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm after attention
        self.norm = nn.LayerNorm(self.attn_dim)
        
        # Final projection to output dimension
        # Combines attention output with original DINOv3 features (residual)
        self.output_projection = nn.Sequential(
            nn.Linear(self.attn_dim + dinov3_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, vggt_features: torch.Tensor, dinov3_features: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention fusion: DINOv3 semantic features query VGGT geometric features.
        
        Handles spatial dimension mismatch by aligning to common resolution.
        
        Args:
            vggt_features: [B, N_k, L_vggt, D_vggt] - 3D geometric features
            dinov3_features: [B, N_k, L_dinov3, D_dinov3] - 2D semantic features
            
        Returns:
            Fused features [B, N_k, L_aligned, output_dim]
        """
        import torch.nn.functional as F
        
        batch_size, num_keyframes = vggt_features.shape[:2]
        vggt_spatial_dim = vggt_features.shape[2]
        dinov3_spatial_dim = dinov3_features.shape[2]
        
        # Handle spatial dimension mismatch
        if vggt_spatial_dim != dinov3_spatial_dim:
            # Calculate spatial grid sizes
            vggt_side = int(vggt_spatial_dim ** 0.5)
            dinov3_side = int(dinov3_spatial_dim ** 0.5)
            
            # Use the smaller resolution as target
            target_side = min(vggt_side, dinov3_side)
            
            # Reshape VGGT to spatial grid for pooling
            vggt_img = vggt_features.view(batch_size * num_keyframes, vggt_side, vggt_side, -1)
            vggt_img = vggt_img.permute(0, 3, 1, 2).contiguous()
            
            # Reshape DINOv3 to spatial grid
            dinov3_img = dinov3_features.view(batch_size * num_keyframes, dinov3_side, dinov3_side, -1)
            dinov3_img = dinov3_img.permute(0, 3, 1, 2).contiguous()
            
            # Resize to target resolution
            if vggt_side != target_side:
                vggt_img = F.adaptive_avg_pool2d(vggt_img, (target_side, target_side))
            if dinov3_side != target_side:
                dinov3_img = F.interpolate(dinov3_img, size=(target_side, target_side),
                                           mode='bilinear', align_corners=False)
            
            # Reshape back: [B*N_k, D, H, W] -> [B, N_k, L, D]
            vggt_features = vggt_img.permute(0, 2, 3, 1).contiguous()
            vggt_features = vggt_features.view(batch_size, num_keyframes, target_side * target_side, -1)
            
            dinov3_features = dinov3_img.permute(0, 2, 3, 1).contiguous()
            dinov3_features = dinov3_features.view(batch_size, num_keyframes, target_side * target_side, -1)
        
        # Now both have aligned spatial dimensions
        seq_len = dinov3_features.shape[2]
        
        # Flatten batch and keyframes for attention: [B*N_k, L, D]
        vggt_flat = vggt_features.view(batch_size * num_keyframes, seq_len, -1)
        dinov3_flat = dinov3_features.view(batch_size * num_keyframes, seq_len, -1)
        
        # Project to common dimension
        vggt_proj = self.vggt_proj(vggt_flat)  # [B*N_k, L, attn_dim]
        dinov3_proj = self.dinov3_proj(dinov3_flat)  # [B*N_k, L, attn_dim]
        
        # Cross-attention: DINOv3 queries, VGGT keys and values
        attn_output, _ = self.cross_attention(
            query=dinov3_proj,
            key=vggt_proj,
            value=vggt_proj
        )
        attn_output = self.norm(attn_output)  # [B*N_k, L, attn_dim]
        
        # Combine attention output with original DINOv3 features (residual connection)
        combined = torch.cat([attn_output, dinov3_flat], dim=-1)  # [B*N_k, L, attn_dim + dinov3_dim]
        
        # Project to output dimension
        output = self.output_projection(combined)  # [B*N_k, L, output_dim]
        
        # Reshape back to [B, N_k, L, output_dim]
        output = output.view(batch_size, num_keyframes, seq_len, -1)
        
        return output


class SpatialMLPFusion(nn.Module):
    """MLP-based fusion of 3D and 2D spatial features with automatic spatial alignment."""
    
    def __init__(self, vggt_dim: int, dinov3_dim: int, output_dim: int):
        super().__init__()
        self.vggt_dim = vggt_dim
        self.dinov3_dim = dinov3_dim
        self.output_dim = output_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vggt_dim + dinov3_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, vggt_features: torch.Tensor, dinov3_features: torch.Tensor) -> torch.Tensor:
        """
        MLP fusion with automatic spatial alignment.
        
        Args:
            vggt_features: [B, N_k, L_vggt, D_vggt] - VGGT spatial features
            dinov3_features: [B, N_k, L_dinov3, D_dinov3] - DINOv3 spatial features
            
        Returns:
            Fused features [B, N_k, L_aligned, output_dim]
        """
        import torch.nn.functional as F
        
        batch_size, num_keyframes = vggt_features.shape[:2]
        vggt_spatial_dim = vggt_features.shape[2]
        dinov3_spatial_dim = dinov3_features.shape[2]
        
        # If spatial dimensions differ, align them
        if vggt_spatial_dim != dinov3_spatial_dim:
            # Calculate spatial grid sizes
            vggt_side = int(vggt_spatial_dim ** 0.5)
            dinov3_side = int(dinov3_spatial_dim ** 0.5)
            
            # Use the smaller resolution as target (typically DINOv3's 14x14)
            target_side = min(vggt_side, dinov3_side)
            
            # Reshape to spatial grid for pooling/interpolation
            # VGGT: [B, N_k, L, D] -> [B*N_k, D, H, W]
            vggt_img = vggt_features.view(batch_size * num_keyframes, vggt_side, vggt_side, -1)
            vggt_img = vggt_img.permute(0, 3, 1, 2).contiguous()
            
            # DINOv3: [B, N_k, L, D] -> [B*N_k, D, H, W]
            dinov3_img = dinov3_features.view(batch_size * num_keyframes, dinov3_side, dinov3_side, -1)
            dinov3_img = dinov3_img.permute(0, 3, 1, 2).contiguous()
            
            # Resize both to target resolution
            if vggt_side != target_side:
                vggt_img = F.adaptive_avg_pool2d(vggt_img, (target_side, target_side))
            if dinov3_side != target_side:
                dinov3_img = F.interpolate(dinov3_img, size=(target_side, target_side), 
                                           mode='bilinear', align_corners=False)
            
            # Reshape back to sequence format: [B*N_k, D, H, W] -> [B, N_k, L, D]
            vggt_features = vggt_img.permute(0, 2, 3, 1).contiguous()
            vggt_features = vggt_features.view(batch_size, num_keyframes, target_side * target_side, -1)
            
            dinov3_features = dinov3_img.permute(0, 2, 3, 1).contiguous()
            dinov3_features = dinov3_features.view(batch_size, num_keyframes, target_side * target_side, -1)
        
        # Now both have same spatial dimensions, safe to concatenate
        concatenated = torch.cat([vggt_features, dinov3_features], dim=-1)
        return self.fusion_mlp(concatenated)


def create_spatial_mllm_pipeline(
    target_keyframes: int = 16,
    total_frames: int = 128,
    sampling_method: str = "hybrid",
    dinov3_model_size: str = "7b",  # Match local model
    fusion_method: str = "concatenate",
    dinov3_img_size: int = 224,  # From config.json (DINOv3 native resolution)
    vggt_img_size: int = 518,  # VGGT uses 518
    device: str = "cuda",
    verbose: bool = True
) -> SpatialMLLMPipeline:
    """
    Factory function to create complete Spatial-MLLM pipeline.
    
    Returns:
        Configured SpatialMLLMPipeline instance
    """
    config = SpatialMLLMIntegrationConfig(
        target_keyframes=target_keyframes,
        total_frames=total_frames,
        sampling_method=sampling_method,
        dinov3_model_size=dinov3_model_size,
        dinov3_img_size=dinov3_img_size,  # 224 for DINOv3 (from config.json)
        vggt_img_size=vggt_img_size,  # 518 for VGGT
        fusion_method=fusion_method,
        device=device,
        verbose=verbose
    )
    
    return SpatialMLLMPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    pipeline = create_spatial_mllm_pipeline(
        target_keyframes=8,  # Reduced for testing
        total_frames=32,     # Reduced for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Spatial-MLLM Integration Pipeline created successfully!")
    print(f"Configuration: {pipeline.config}")
    
    # Test forward pass
    batch_size, total_frames = 1, 32
    test_video = torch.randn(batch_size, total_frames, 3, 518, 518)
    
    with torch.no_grad():
        result = pipeline(test_video, return_intermediate=True)
        
    print(f"Pipeline output keys: {list(result.keys())}")
    print(f"Processing metadata: {result['processing_metadata']}")
