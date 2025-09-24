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
from typing import Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..data import create_keyframe_selector
from .dinov3_compatibility import create_dinov3_compatibility_layer
from .vggt.models.vggt import VGGT
from .heatmap.converter import LLMToHeatmapConverter as LLMHiddenStateConverter
from .real_llm_integration import create_real_llm_integration
from .memory_efficient_llm import create_memory_efficient_llm

logger = logging.getLogger(__name__)


@dataclass
class SpatialMLLMIntegrationConfig:
    """Configuration for complete Spatial-MLLM integration."""
    # Frame sampling configuration
    target_keyframes: int = 16  # N_k
    total_frames: int = 128  # N_m
    sampling_method: str = "hybrid"  # greedy_coverage, novelty_weighted, hybrid
    
    # Model configurations
    dinov3_model_size: str = "large"  # base, large, giant
    dinov3_patch_size: int = 14
    dinov3_img_size: int = 518
    
    # VGGT configuration
    vggt_img_size: int = 518
    vggt_patch_size: int = 14
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
    
    # Performance settings
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    enable_gradient_checkpointing: bool = False

    # Multi-GPU settings
    use_multi_gpu: bool = True  # Distribute models across GPUs
    vggt_gpu: str = "cuda:0"  # VGGT on GPU 0
    dinov3_gpu: str = "cuda:1"  # DINOv3 on GPU 1
    llm_gpu: str = "cuda:2"  # LLM on GPU 2
    
    # Debug and logging
    verbose: bool = True
    save_intermediate_features: bool = False


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
        self.device = torch.device(config.device)
        
        # Initialize VGGT for 3D geometry processing (all N_m frames) on dedicated GPU
        vggt_device = torch.device(config.vggt_gpu if config.use_multi_gpu else config.device)
        try:
            # Try to load pretrained VGGT from local model directory
            from ..utils.path_utils import resolve_model_path
            vggt_model_path = "./models/vggt"
            try:
                resolved_vggt_path = resolve_model_path(vggt_model_path, "VGGT")
                self.vggt = VGGT.from_pretrained(str(resolved_vggt_path)).to(device=vggt_device, dtype=config.dtype)
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
            ).to(device=vggt_device, dtype=config.dtype)
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
        actual_dinov3_model_size = "7b"  # Based on actual local model analysis
        self.dinov3_compat = create_dinov3_compatibility_layer(
            model_size=actual_dinov3_model_size,
            patch_size=config.dinov3_patch_size,
            img_size=config.dinov3_img_size,
            target_embed_dim=config.vggt_embed_dim,  # Match VGGT dimensions
            align_with_vggt=True,
            device=dinov3_device,
            dtype=config.dtype
        )
        
        # Initialize feature fusion module
        self.feature_fusion = self._create_feature_fusion_module().to(device=self.device, dtype=config.dtype)
        
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
            # Keep projector as fallback
            if config.enable_llm_projection:
                self.llm_projector = self._create_llm_projection_module().to(device=self.device, dtype=config.dtype)
            else:
                self.llm_projector = nn.Identity().to(device=self.device, dtype=config.dtype)
        else:
            logger.warning("Using FAKE LLM projection - not real LLM processing!")
            self.llm_integration = None
            if config.enable_llm_projection:
                self.llm_projector = self._create_llm_projection_module().to(device=self.device, dtype=config.dtype)
            else:
                self.llm_projector = nn.Identity().to(device=self.device, dtype=config.dtype)
            
        # Initialize heatmap converter for inter-frame heatmaps
        if config.enable_inter_frame_heatmaps:
            self.heatmap_converter = LLMHiddenStateConverter(
                vlm_dim=config.llm_token_dim,
                target_size=config.heatmap_size[0]  # Use single dimension
            ).to(device=self.device, dtype=config.dtype)
        else:
            self.heatmap_converter = None
            
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
        return_heatmaps: bool = True
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
                - 'intermediate_features': Debug information (if requested)
                - 'processing_metadata': Pipeline statistics
        """
        
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
        video_frames_vggt = video_frames.to(device=vggt_device, dtype=self.config.dtype)
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
            logger.info(f"Selected {len(selected_indices)} keyframes: {selected_indices.tolist()}")
        
        # Step 3: Dual-path feature extraction
        logger.info("Step 3: Dual-path feature extraction")
        
        # 3D path: Index-selected VGGT features (pre-computed)
        vggt_features = keyframe_result['vggt_features']
        # Move VGGT features to main device for fusion (handle dict of tensors)
        if isinstance(vggt_features, dict):
            vggt_features = {k: v.to(device=self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in vggt_features.items()}
        else:
            vggt_features = vggt_features.to(device=self.device)
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
                logger.info(f"✅ REAL LLM processing successful! Generated text: {llm_result.get('llm_output', '')[:100]}...")
            except Exception as e:
                logger.error(f"❌ REAL LLM processing failed: {e}")
                logger.warning("Falling back to fake LLM projection")
                llm_tokens = self.llm_projector(fused_features)
        else:
            # Fallback to fake projector
            logger.warning("❌ Using FAKE LLM projection - no real spatial reasoning!")
            llm_tokens = self.llm_projector(fused_features)
        
        # Step 6: Generate frame-indexed inter-frame heatmaps
        frame_indexed_heatmaps = None
        if return_heatmaps and self.heatmap_converter is not None:
            logger.info("Step 6: Frame-indexed inter-frame heatmap generation")
            # Use last frame index as reference for spatial relationships (current observation)
            current_frame_idx = total_frames - 1
            frame_indexed_heatmaps = self._generate_inter_frame_heatmaps(
                llm_tokens, selected_indices, keyframe_result['geometry_data'], current_frame_idx=current_frame_idx
            )
        
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
        
        if frame_indexed_heatmaps is not None:
            output['frame_indexed_heatmaps'] = frame_indexed_heatmaps
            # For backward compatibility, also provide the first heatmap as 'inter_frame_heatmap'
            if frame_indexed_heatmaps:
                first_frame_idx = min(frame_indexed_heatmaps.keys())
                output['inter_frame_heatmap'] = frame_indexed_heatmaps[first_frame_idx].unsqueeze(1)  # Add channel dim
            
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
        
        with torch.amp.autocast('cuda', enabled=True, dtype=self.config.dtype):
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
        """Extract spatial features from VGGT predictions."""
        
        # Use world points as primary spatial features
        # Shape: [B, N_k, H, W, 3] → [B, N_k, H*W, 3]
        world_points = vggt_features['world_points']
        batch_size, num_keyframes, height, width = world_points.shape[:4]
        
        # Flatten spatial dimensions
        spatial_features = world_points.view(batch_size, num_keyframes, height * width, -1)
        
        # Project to target dimension (2 * embed_dim to match VGGT head outputs)
        target_dim = self.config.vggt_embed_dim * 2
        if spatial_features.shape[-1] != target_dim:
            projection = nn.Linear(spatial_features.shape[-1], target_dim).to(device=spatial_features.device, dtype=spatial_features.dtype)
            spatial_features = projection(spatial_features)
            
        return spatial_features
    
    def _fuse_spatial_features(
        self,
        vggt_features: torch.Tensor,
        dinov3_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse 3D VGGT and 2D DINOv3 features."""
        
        if self.config.verbose:
            print(f"INFO: VGGT features shape: {vggt_features.shape}")
            print(f"INFO: DINOv3 features shape: {dinov3_features.shape}")
        
        if self.config.fusion_method == "concatenate":
            # Handle spatial dimension mismatch by projecting to common space
            batch_size, num_keyframes = vggt_features.shape[:2]
            vggt_spatial_dim = vggt_features.shape[2]
            dinov3_spatial_dim = dinov3_features.shape[2]
            feature_dim = vggt_features.shape[-1]
            
            if vggt_spatial_dim != dinov3_spatial_dim:
                # Use interpolation to resize DINOv3 features to match VGGT spatial dimensions
                # This is more memory-efficient than creating a huge linear layer
                
                # Determine spatial layout for DINOv3 features
                dinov3_h = int(dinov3_spatial_dim ** 0.5)
                dinov3_w = dinov3_spatial_dim // dinov3_h
                vggt_h = int(vggt_spatial_dim ** 0.5)
                vggt_w = vggt_spatial_dim // vggt_h
                
                # Reshape DINOv3 features to [B, N_k, C, H, W] format
                dinov3_spatial = dinov3_features.view(batch_size, num_keyframes, feature_dim, dinov3_h, dinov3_w)
                dinov3_spatial = dinov3_spatial.permute(0, 1, 2, 3, 4).contiguous()
                
                # Interpolate to match VGGT spatial dimensions
                dinov3_resized = torch.nn.functional.interpolate(
                    dinov3_spatial.view(batch_size * num_keyframes, feature_dim, dinov3_h, dinov3_w),
                    size=(vggt_h, vggt_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Reshape back to [B, N_k, spatial_dim, feature_dim]
                dinov3_features = dinov3_resized.view(batch_size, num_keyframes, feature_dim, vggt_h * vggt_w)
                dinov3_features = dinov3_features.permute(0, 1, 3, 2).contiguous()
                
                if self.config.verbose:
                    print(f"INFO: Resized DINOv3 features to shape: {dinov3_features.shape}")
            
            # Now concatenate along feature dimension
            fused = torch.cat([vggt_features, dinov3_features], dim=-1)
            return self.feature_fusion(fused)
        else:
            # Use specialized fusion module
            return self.feature_fusion(vggt_features, dinov3_features)
    
    def _generate_inter_frame_heatmaps(
        self,
        llm_tokens: torch.Tensor,
        selected_indices: torch.Tensor,
        geometry_data: Dict[str, torch.Tensor],
        current_frame_idx: int = 0
    ) -> Dict[int, torch.Tensor]:
        """Generate frame-indexed heatmaps showing spatial relationships between keyframes and current observation."""

        # llm_tokens shape: [B, N_frames, N_patches, D]
        batch_size, num_frames, num_patches, token_dim = llm_tokens.shape

        # Convert selected indices to list of original frame numbers
        if len(selected_indices.shape) == 1:
            # Single batch case: [N_k] -> List[int]
            keyframe_indices = selected_indices.cpu().tolist()
        else:
            # Multi-batch case: [B, N_k] -> use first batch for now
            keyframe_indices = selected_indices[0].cpu().tolist()

        logger.info(f"Generating frame-indexed heatmaps for keyframes: {keyframe_indices}")
        logger.info(f"LLM tokens shape: {llm_tokens.shape}")

        # Generate frame-specific heatmaps
        frame_indexed_heatmaps = {}

        for i, original_frame_idx in enumerate(keyframe_indices):
            if i >= num_frames:
                # Skip if we don't have enough frames
                logger.warning(f"Skipping frame {original_frame_idx}: index {i} >= num_frames {num_frames}")
                continue

            # Extract tokens for this specific frame: [B, N_patches, D]
            frame_tokens = llm_tokens[:, i]  # [B, N_patches, D]

            # Reshape to 2D spatial layout for heatmap generation
            # Assuming square patch layout: N_patches = H_patches * W_patches
            patch_h = patch_w = int(num_patches ** 0.5)
            if patch_h * patch_w != num_patches:
                # Handle non-square case by finding closest square or use default
                patch_h = patch_w = int(num_patches ** 0.5)
                logger.warning(f"Non-square patch count {num_patches}, using {patch_h}x{patch_w}")

            # Reshape to spatial layout: [B, D, H_patch, W_patch]
            try:
                spatial_tokens = frame_tokens.permute(0, 2, 1).view(batch_size, token_dim, patch_h, patch_w)

                # Generate heatmap using the converter's upsampling
                frame_heatmap = self.heatmap_converter.generate_heatmap(spatial_tokens)  # [B, 1, H, W]

                # Store with original frame index as key, remove channel dimension
                frame_indexed_heatmaps[original_frame_idx] = frame_heatmap.squeeze(1)  # [B, H, W]

                logger.info(f"Generated heatmap for frame {original_frame_idx}: shape {frame_heatmap.squeeze(1).shape}")

            except Exception as e:
                logger.error(f"Failed to generate heatmap for frame {original_frame_idx}: {e}")
                # Create zero heatmap as fallback
                frame_indexed_heatmaps[original_frame_idx] = torch.zeros(
                    batch_size, self.config.heatmap_size[0], self.config.heatmap_size[1],
                    device=llm_tokens.device, dtype=llm_tokens.dtype
                )

        logger.info(f"Generated {len(frame_indexed_heatmaps)} frame-indexed heatmaps")
        return frame_indexed_heatmaps  # Dict[int, torch.Tensor] - heatmaps indexed by original frame numbers


class SpatialAttentionFusion(nn.Module):
    """Attention-based fusion of 3D and 2D spatial features."""
    
    def __init__(self, vggt_dim: int, dinov3_dim: int, output_dim: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=min(vggt_dim, dinov3_dim),
            num_heads=8,
            batch_first=True
        )
        self.output_projection = nn.Linear(vggt_dim + dinov3_dim, output_dim)
        
    def forward(self, vggt_features: torch.Tensor, dinov3_features: torch.Tensor) -> torch.Tensor:
        # Simplified attention fusion
        concatenated = torch.cat([vggt_features, dinov3_features], dim=-1)
        return self.output_projection(concatenated)


class SpatialMLPFusion(nn.Module):
    """MLP-based fusion of 3D and 2D spatial features."""
    
    def __init__(self, vggt_dim: int, dinov3_dim: int, output_dim: int):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vggt_dim + dinov3_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, vggt_features: torch.Tensor, dinov3_features: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([vggt_features, dinov3_features], dim=-1)
        return self.fusion_mlp(concatenated)


def create_spatial_mllm_pipeline(
    target_keyframes: int = 16,
    total_frames: int = 128,
    sampling_method: str = "hybrid",
    dinov3_model_size: str = "7b",  # Updated to match local model
    fusion_method: str = "concatenate",
    img_size: int = 518,  # Add img_size parameter
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
        dinov3_img_size=img_size,
        vggt_img_size=img_size,
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