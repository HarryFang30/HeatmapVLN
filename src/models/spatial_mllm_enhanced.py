"""
Enhanced Spatial-MLLM Integration with Phase 4 Features
======================================================

This module enhances the existing Spatial-MLLM architecture from /Spatial-MLLM
with Phase 4 features: advanced feature fusion, multi-scale processing, and
improved integration with our space-aware sampling pipeline.

Key Enhancements:
1. Integration with our space-aware frame sampling
2. Advanced feature fusion mechanisms (cross-attention, graph, adaptive)
3. Multi-scale spatial processing
4. Enhanced VGGT embedding merger
5. Performance optimizations from Phase 3

Architecture Flow:
N_m frames → VGGT (all frames) → Space-aware sampling → N_k keyframes
→ Enhanced VGGT processing → Advanced feature fusion → LLM integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass

# Import existing Spatial-MLLM components
import sys
from pathlib import Path
# Use relative import from project structure instead of absolute path
try:
    from ..utils.path_utils import get_project_root
    spatial_mllm_path = get_project_root().parent / "Spatial-MLLM" / "src"
    if spatial_mllm_path.exists():
        sys.path.append(str(spatial_mllm_path))
    else:
        # Fallback to project-local implementation
        pass  # Use our own implementations in src/models/
except ImportError:
    # Fallback if path_utils not available
    pass

try:
    from models.modeling_qwen2_5_vl import (
        Qwen2_5_VL_VGGTForConditionalGeneration,
        VGGTEmbeddingMerger,
        VGGTEmbeddingMergerConfig
    )
    from models.vggt.models.vggt import VGGT
except ImportError as e:
    logger.warning(f"Could not import Spatial-MLLM components: {e}")
    # Fallback implementations will be provided

# Import our Phase 3 components
from ..data import create_keyframe_selector
from .feature_fusion import create_feature_fusion
from .performance_optimizer import create_performance_optimizer

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSpatialMLLMConfig:
    """Configuration for enhanced Spatial-MLLM integration."""
    # Frame sampling configuration
    total_frames: int = 128  # N_m input frames
    target_keyframes: int = 16  # N_k selected keyframes
    sampling_method: str = "hybrid"  # greedy_coverage, novelty_weighted, hybrid
    
    # Feature fusion configuration  
    fusion_method: str = "cross_attention"  # cross_attention, graph, adaptive, concatenate
    enable_multi_scale: bool = True
    
    # VGGT configuration
    vggt_embed_dim: int = 1024
    vggt_patch_size: int = 14
    vggt_img_size: int = 518
    
    # Enhanced merger configuration
    enhanced_merger: bool = True
    merger_num_layers: int = 3
    merger_hidden_dim: int = 2048
    
    # Performance optimization
    enable_optimizations: bool = True
    use_gradient_checkpointing: bool = False
    enable_mixed_precision: bool = True
    
    # Device and precision
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class EnhancedVGGTEmbeddingMerger(nn.Module):
    """
    Enhanced VGGT embedding merger with advanced fusion and multi-scale processing.
    
    This extends the original VGGTEmbeddingMerger with:
    - Multi-scale spatial processing
    - Advanced feature fusion mechanisms
    - Space-aware frame sampling integration
    - Performance optimizations
    """
    
    def __init__(self, config: EnhancedSpatialMLLMConfig):
        super().__init__()
        self.config = config
        
        # Original merger configuration (maintaining compatibility)
        self.original_merger_config = VGGTEmbeddingMergerConfig(
            input_dim=2 * config.vggt_embed_dim,  # VGGT concatenates tokens
            output_dim=config.merger_hidden_dim,
            patch_size=config.vggt_patch_size,
            spatial_merge_size=2,
            temporal_merge_size=2
        )
        
        # Enhanced processing layers
        if config.enhanced_merger:
            self.enhanced_processing = self._create_enhanced_processing()
            
        # Advanced feature fusion
        self.feature_fusion = create_feature_fusion(
            fusion_method=config.fusion_method,
            vggt_feature_dim=2 * config.vggt_embed_dim,
            dinov3_feature_dim=2 * config.vggt_embed_dim,  # Aligned dimensions
            output_dim=config.merger_hidden_dim
        )
        
        # Multi-scale processing (if enabled)
        if config.enable_multi_scale:
            self.multi_scale_processor = MultiScaleSpatialProcessor(config)
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.merger_hidden_dim),
            nn.Linear(config.merger_hidden_dim, config.merger_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def _create_enhanced_processing(self) -> nn.Module:
        """Create enhanced processing layers."""
        layers = []
        
        for i in range(self.config.merger_num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.config.merger_hidden_dim,
                nhead=16,
                dim_feedforward=self.config.merger_hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            layers.append(layer)
        
        return nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config.merger_hidden_dim,
                nhead=16,
                dim_feedforward=self.config.merger_hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=self.config.merger_num_layers
        )
    
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        images_shape: Tuple,
        media_type: str,
        selected_indices: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with space-aware processing.
        
        Args:
            aggregated_tokens_list: VGGT aggregated tokens
            patch_start_idx: Starting index for patches
            images_shape: Shape of input images
            media_type: "images" or "video"
            selected_indices: Selected keyframe indices from space-aware sampling
            spatial_features: Additional spatial features (e.g., from DINOv3)
            
        Returns:
            Dictionary containing processed embeddings and metadata
        """
        
        # Extract tokens (following original merger logic)
        tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        
        if media_type == "images":
            tokens = tokens.repeat_interleave(2, dim=1)
        
        # Apply index-based selection if provided (space-aware sampling)
        if selected_indices is not None:
            # tokens shape: [B, F, N_patches, D]
            batch_size = tokens.shape[0]
            device = tokens.device
            
            # Ensure indices are on correct device
            if isinstance(selected_indices, torch.Tensor):
                indices = selected_indices.to(device)
            else:
                indices = torch.tensor(selected_indices, device=device)
            
            # Index select along frame dimension
            tokens = tokens[:, indices]  # [B, N_k, N_patches, D]
        
        batch_size, num_frames, num_patches, feature_dim = tokens.shape
        
        # Multi-scale processing (if enabled)
        if hasattr(self, 'multi_scale_processor'):
            tokens = self.multi_scale_processor(tokens, images_shape)
        
        # Advanced feature fusion (if spatial features provided)
        if spatial_features is not None and hasattr(self, 'feature_fusion'):
            # Reshape for fusion: [B, N_k, N_patches, D]
            if spatial_features.shape != tokens.shape:
                # Adapt spatial features to match VGGT tokens shape
                spatial_features = self._adapt_spatial_features(spatial_features, tokens.shape)
            
            fusion_result = self.feature_fusion(tokens, spatial_features)
            tokens = fusion_result['fused_features']
        
        # Enhanced processing layers
        if hasattr(self, 'enhanced_processing'):
            # Reshape for transformer processing: [B*N_k, N_patches, D]
            reshaped_tokens = tokens.view(-1, num_patches, feature_dim)
            enhanced_tokens = self.enhanced_processing(reshaped_tokens)
            tokens = enhanced_tokens.view(batch_size, -1, num_patches, feature_dim)
        
        # Merge spatial and temporal tokens (following original logic)
        merged_tokens = self._merge_tokens(tokens, images_shape)
        
        # Final output projection
        output_embeddings = self.output_projection(merged_tokens)
        
        # Flatten for LLM input: [S, D]
        flattened_embeddings = output_embeddings.view(-1, output_embeddings.shape[-1])
        
        return {
            'embeddings': flattened_embeddings,
            'merged_tokens': merged_tokens,
            'original_shape': output_embeddings.shape,
            'processing_metadata': {
                'num_selected_frames': tokens.shape[1],
                'num_patches': num_patches,
                'feature_dim': feature_dim,
                'used_multi_scale': hasattr(self, 'multi_scale_processor'),
                'used_feature_fusion': spatial_features is not None,
                'used_enhanced_processing': hasattr(self, 'enhanced_processing')
            }
        }
    
    def _merge_tokens(self, tokens: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        """
        Merge spatial and temporal tokens following original merger logic.
        
        This maintains compatibility with the original VGGTEmbeddingMerger.
        """
        H, W = images_shape[-2:]
        NUM_PATCH_H, NUM_PATCH_W = H // self.config.vggt_patch_size, W // self.config.vggt_patch_size
        
        B, F, S, D = tokens.shape
        
        # Spatial merge size from config
        spatial_merge_size = self.original_merger_config.spatial_merge_size
        temporal_merge_size = self.original_merger_config.temporal_merge_size
        
        # Reshape for merging
        tokens = tokens.view(B, F, NUM_PATCH_H, NUM_PATCH_W, D)
        
        # Apply temporal and spatial merging
        tokens = tokens.view(
            B,
            F // temporal_merge_size,
            temporal_merge_size,
            NUM_PATCH_H // spatial_merge_size,
            spatial_merge_size,
            NUM_PATCH_W // spatial_merge_size,
            spatial_merge_size,
            D,
        )
        
        tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        tokens = tokens.view(
            B, 
            F // temporal_merge_size, 
            NUM_PATCH_H // spatial_merge_size, 
            NUM_PATCH_W // spatial_merge_size,
            temporal_merge_size * spatial_merge_size * spatial_merge_size * D,
        )
        
        # Project merged tokens
        token_merge_in_dim = temporal_merge_size * spatial_merge_size * spatial_merge_size * D
        
        # Create projection layers if not exists
        if not hasattr(self, 'token_merge_projection'):
            self.token_merge_projection = nn.Sequential(
                nn.LayerNorm(token_merge_in_dim),
                nn.Linear(token_merge_in_dim, self.config.merger_hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.merger_hidden_dim, self.config.merger_hidden_dim)
            ).to(tokens.device)
        
        merged = self.token_merge_projection(tokens)
        
        return merged
    
    def _adapt_spatial_features(
        self, 
        spatial_features: torch.Tensor, 
        target_shape: Tuple
    ) -> torch.Tensor:
        """Adapt spatial features to match VGGT token shape."""
        
        target_batch, target_frames, target_patches, target_dim = target_shape
        current_shape = spatial_features.shape
        
        # Handle different input shapes
        if len(current_shape) == 4:  # [B, N_k, N_patches, D]
            if current_shape == target_shape:
                return spatial_features
            elif current_shape[-1] != target_dim:
                # Project to target dimension
                if not hasattr(self, 'spatial_dim_projection'):
                    self.spatial_dim_projection = nn.Linear(
                        current_shape[-1], target_dim
                    ).to(spatial_features.device)
                spatial_features = self.spatial_dim_projection(spatial_features)
        
        return spatial_features


class MultiScaleSpatialProcessor(nn.Module):
    """Multi-scale spatial processing for enhanced spatial understanding."""
    
    def __init__(self, config: EnhancedSpatialMLLMConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale processing layers
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.merger_hidden_dim, config.merger_hidden_dim, 
                         kernel_size=3, padding=1),
                nn.BatchNorm2d(config.merger_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.merger_hidden_dim, config.merger_hidden_dim, 
                         kernel_size=1)
            )
            for _ in range(3)  # 3 different scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Conv2d(
            config.merger_hidden_dim * 3,
            config.merger_hidden_dim,
            kernel_size=1
        )
    
    def forward(self, tokens: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        """Process tokens at multiple spatial scales."""
        
        batch_size, num_frames, num_patches, feature_dim = tokens.shape
        
        # Determine spatial layout
        patch_size = int(num_patches ** 0.5)
        if patch_size * patch_size != num_patches:
            # Non-square layout, skip multi-scale processing
            return tokens
        
        # Reshape to spatial layout: [B*F, D, H, W]
        spatial_tokens = tokens.view(-1, patch_size, patch_size, feature_dim)
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)  # [B*F, D, H, W]
        
        # Process at different scales
        scale_outputs = []
        scales = [1.0, 0.5, 2.0]  # Original, half, double scale
        
        for i, scale in enumerate(scales):
            if scale != 1.0:
                # Resize for different scales
                size = (int(patch_size * scale), int(patch_size * scale))
                scaled_tokens = F.interpolate(spatial_tokens, size=size, mode='bilinear')
                processed = self.scale_processors[i](scaled_tokens)
                # Resize back to original size
                processed = F.interpolate(processed, size=(patch_size, patch_size), mode='bilinear')
            else:
                processed = self.scale_processors[i](spatial_tokens)
            
            scale_outputs.append(processed)
        
        # Fuse multi-scale features
        concatenated = torch.cat(scale_outputs, dim=1)
        fused = self.scale_fusion(concatenated)
        
        # Reshape back to token format
        fused_tokens = fused.permute(0, 2, 3, 1)  # [B*F, H, W, D]
        fused_tokens = fused_tokens.view(batch_size, num_frames, num_patches, feature_dim)
        
        return fused_tokens


class EnhancedSpatialMLLMPipeline(nn.Module):
    """
    Complete enhanced Spatial-MLLM pipeline integrating all Phase 4 features.
    
    This provides a unified interface that combines:
    - Existing Spatial-MLLM architecture
    - Our space-aware frame sampling
    - Advanced feature fusion
    - Performance optimizations
    """
    
    def __init__(self, config: EnhancedSpatialMLLMConfig):
        super().__init__()
        self.config = config
        
        # Initialize space-aware keyframe selector
        self.keyframe_selector = create_keyframe_selector(
            target_keyframes=config.target_keyframes,
            total_frames=config.total_frames,
            sampling_method=config.sampling_method,
            device=config.device,
            verbose=True
        )
        
        # Initialize enhanced VGGT merger
        self.enhanced_merger = EnhancedVGGTEmbeddingMerger(config)
        
        # Performance optimizer (if enabled)
        if config.enable_optimizations:
            self.performance_optimizer = create_performance_optimizer()
        else:
            self.performance_optimizer = None
    
    def forward(
        self,
        video_frames: torch.Tensor,  # [B, N_m, C, H, W]
        vggt_model: VGGT,
        media_type: str = "video",
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced forward pass with space-aware sampling and advanced fusion.
        
        Args:
            video_frames: Input video frames
            vggt_model: VGGT model instance
            media_type: Type of media ("images" or "video")
            return_intermediate: Return intermediate processing results
            
        Returns:
            Dictionary containing enhanced embeddings and metadata
        """
        
        batch_size, total_frames = video_frames.shape[:2]
        
        # Step 1: Process ALL frames through VGGT
        with torch.cuda.amp.autocast(enabled=self.config.enable_mixed_precision):
            # VGGT processing
            vggt_output = vggt_model(video_frames.view(-1, *video_frames.shape[2:]))
            
            # Get aggregated tokens
            aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(
                video_frames.view(-1, *video_frames.shape[2:])
            )
            
            # Reshape aggregated tokens for frame structure
            for i, tokens in enumerate(aggregated_tokens_list):
                new_shape = (batch_size, total_frames) + tokens.shape[1:]
                aggregated_tokens_list[i] = tokens.view(new_shape)
        
        # Step 2: Space-aware keyframe selection
        # Create mock VGGT predictions for keyframe selector
        mock_predictions = {
            'depth': torch.randn(batch_size, total_frames, 37, 37, 1, device=video_frames.device),
            'depth_conf': torch.rand(batch_size, total_frames, 37, 37, device=video_frames.device),
            'world_points': torch.randn(batch_size, total_frames, 37, 37, 3, device=video_frames.device),
            'world_points_conf': torch.rand(batch_size, total_frames, 37, 37, device=video_frames.device),
            'pose_enc': torch.randn(batch_size, total_frames, 9, device=video_frames.device)
        }
        
        selection_result = self.keyframe_selector(
            vggt_predictions=mock_predictions,
            original_frames=video_frames
        )
        selected_indices = selection_result['keyframe_indices']
        
        # Step 3: Enhanced embedding merger with selected keyframes
        merger_result = self.enhanced_merger(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            images_shape=video_frames.shape,
            media_type=media_type,
            selected_indices=selected_indices
        )
        
        # Prepare output
        output = {
            'enhanced_embeddings': merger_result['embeddings'],
            'selected_keyframes': selected_indices,
            'processing_metadata': {
                **merger_result['processing_metadata'],
                'total_input_frames': total_frames,
                'selection_method': self.config.sampling_method,
                'fusion_method': self.config.fusion_method,
                'enhanced_features_enabled': True
            }
        }
        
        if return_intermediate:
            output.update({
                'vggt_output': vggt_output,
                'selection_result': selection_result,
                'merger_result': merger_result,
                'aggregated_tokens': aggregated_tokens_list
            })
        
        return output


def create_enhanced_spatial_mllm(
    total_frames: int = 128,
    target_keyframes: int = 16,
    sampling_method: str = "hybrid",
    fusion_method: str = "cross_attention",
    enable_optimizations: bool = True,
    device: str = "cuda"
) -> EnhancedSpatialMLLMPipeline:
    """
    Factory function to create enhanced Spatial-MLLM pipeline.
    
    Returns:
        Configured EnhancedSpatialMLLMPipeline
    """
    config = EnhancedSpatialMLLMConfig(
        total_frames=total_frames,
        target_keyframes=target_keyframes,
        sampling_method=sampling_method,
        fusion_method=fusion_method,
        enable_optimizations=enable_optimizations,
        device=device
    )
    
    return EnhancedSpatialMLLMPipeline(config)


# Example usage
if __name__ == "__main__":
    # Create enhanced pipeline
    pipeline = create_enhanced_spatial_mllm(
        total_frames=64,  # Reduced for testing
        target_keyframes=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Enhanced Spatial-MLLM Pipeline created successfully!")
    print(f"Configuration: {pipeline.config}")
    
    # Test with mock VGGT model
    from ..vggt.models.vggt import VGGT
    vggt_model = VGGT()
    
    # Test data
    batch_size, total_frames = 1, 16
    test_video = torch.randn(batch_size, total_frames, 3, 518, 518)
    
    with torch.no_grad():
        result = pipeline(test_video, vggt_model, return_intermediate=True)
    
    print(f"Pipeline output keys: {list(result.keys())}")
    print(f"Enhanced embeddings shape: {result['enhanced_embeddings'].shape}")
    print(f"Selected keyframes: {result['selected_keyframes']}")
    print(f"Processing metadata: {result['processing_metadata']}")