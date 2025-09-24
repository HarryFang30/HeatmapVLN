"""
VGGT Integration and Compatibility Layer
=======================================

This module provides a comprehensive integration layer for VGGT (Visual Geometry 
and Geometry Transformer) within the VLN pipeline. It ensures compatibility with
the space-aware frame sampling and dual-encoder architecture.

Key Features:
- VGGT model initialization and configuration
- Batch processing optimization for N_m frames
- Geometry feature extraction and caching
- Integration with space-aware sampling algorithms
- Performance monitoring and debugging
- Memory efficient processing for large video sequences

VGGT Role in Pipeline:
1. Process ALL N_m input frames for complete geometry understanding
2. Extract camera poses, depth maps, and 3D world points
3. Provide geometry information to space-aware sampling algorithm  
4. Supply pre-computed features for efficient index-based retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from dataclasses import dataclass
import numpy as np

from .vggt.models.vggt import VGGT

logger = logging.getLogger(__name__)


@dataclass
class VGGTIntegrationConfig:
    """Configuration for VGGT integration in VLN pipeline."""
    # Model architecture
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    
    # Processing configuration
    batch_size_limit: int = 8  # Maximum frames to process in single batch
    enable_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    
    # Memory optimization
    enable_feature_caching: bool = True
    cache_geometry_features: bool = True
    clear_cache_after_sampling: bool = False
    
    # Geometry processing
    depth_confidence_threshold: float = 0.1
    world_points_confidence_threshold: float = 0.1
    enable_pose_refinement: bool = True
    
    # Performance settings
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    
    # Debug and monitoring
    verbose: bool = True
    profile_performance: bool = False


class VGGTProcessor(nn.Module):
    """
    Integrated VGGT processor for VLN pipeline.
    
    This class handles VGGT model management, batch processing of video frames,
    geometry extraction, and integration with downstream components.
    """
    
    def __init__(self, config: VGGTIntegrationConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize VGGT model
        self.vggt = VGGT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim
        ).to(device=self.device, dtype=config.dtype)
        
        # Setup performance optimization
        if config.enable_gradient_checkpointing:
            # Enable gradient checkpointing if VGGT supports it
            if hasattr(self.vggt, 'gradient_checkpointing_enable'):
                self.vggt.gradient_checkpointing_enable()
        
        # Initialize caching system
        self._geometry_cache = {}
        self._feature_cache = {}
        
        # Performance tracking
        self._processing_stats = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'batch_counts': {}
        }
        
        logger.info(f"VGGT Processor initialized: {config.img_size}x{config.img_size}, "
                   f"patch_size={config.patch_size}, embed_dim={config.embed_dim}")
    
    def forward(
        self,
        video_frames: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
        return_raw_features: bool = False,
        enable_caching: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process video frames through VGGT for geometry extraction.
        
        Args:
            video_frames: Input frames [B, N_m, C, H, W]
            frame_indices: Optional frame indices for partial processing
            return_raw_features: Include raw VGGT features in output
            enable_caching: Override caching setting
            
        Returns:
            Dictionary containing VGGT predictions and geometry information
        """
        
        if enable_caching is None:
            enable_caching = self.config.enable_feature_caching
            
        batch_size, total_frames = video_frames.shape[:2]
        
        if self.config.verbose:
            logger.info(f"Processing {total_frames} frames through VGGT")
        
        start_time = time.time()
        
        # Generate cache key if caching enabled
        cache_key = None
        if enable_caching:
            cache_key = self._generate_cache_key(video_frames, frame_indices)
            if cache_key in self._geometry_cache:
                logger.info("Using cached VGGT geometry features")
                self._processing_stats['cache_hits'] += 1
                return self._geometry_cache[cache_key]
        
        # Prepare frame indices
        if frame_indices is None:
            frame_indices = list(range(total_frames))
        
        # Process frames in batches for memory efficiency
        all_predictions = self._process_frames_in_batches(
            video_frames, frame_indices, batch_size
        )
        
        # Post-process and enhance geometry features
        enhanced_predictions = self._enhance_geometry_features(all_predictions)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        enhanced_predictions['processing_metadata'] = {
            'total_frames': total_frames,
            'processed_frames': len(frame_indices),
            'processing_time_seconds': processing_time,
            'frames_per_second': len(frame_indices) / processing_time,
            'batch_size_used': self.config.batch_size_limit,
            'device_used': str(self.device)
        }
        
        # Update performance statistics
        self._update_performance_stats(len(frame_indices), processing_time)
        
        # Cache results if enabled
        if enable_caching and cache_key:
            self._geometry_cache[cache_key] = enhanced_predictions
        
        if self.config.verbose:
            logger.info(f"VGGT processing completed in {processing_time:.3f}s "
                       f"({len(frame_indices)/processing_time:.1f} FPS)")
        
        return enhanced_predictions
    
    def _process_frames_in_batches(
        self,
        video_frames: torch.Tensor,
        frame_indices: List[int],
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Process frames in batches for memory efficiency."""
        
        total_frames = len(frame_indices)
        batch_limit = self.config.batch_size_limit
        
        # Collect all batch results
        batch_results = []
        
        # Process in chunks
        for i in range(0, total_frames, batch_limit):
            end_idx = min(i + batch_limit, total_frames)
            batch_frame_indices = frame_indices[i:end_idx]
            
            # Extract batch frames
            batch_frames = video_frames[:, batch_frame_indices]  # [B, batch_size, C, H, W]
            batch_frames_flat = batch_frames.view(-1, *batch_frames.shape[2:])  # [B*batch_size, C, H, W]
            
            if self.config.verbose and total_frames > batch_limit:
                logger.info(f"Processing batch {i//batch_limit + 1}/{(total_frames + batch_limit - 1)//batch_limit}: "
                           f"frames {i}-{end_idx-1}")
            
            # Process batch through VGGT
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                batch_predictions = self.vggt(batch_frames_flat)
            
            # Reshape back to batch structure
            for key, tensor in batch_predictions.items():
                if len(tensor.shape) >= 2:
                    new_shape = (batch_size, len(batch_frame_indices)) + tensor.shape[1:]
                    batch_predictions[key] = tensor.view(new_shape)
            
            batch_results.append(batch_predictions)
        
        # Concatenate all batches along frame dimension
        concatenated_results = {}
        for key in batch_results[0].keys():
            concatenated_tensors = []
            for batch_result in batch_results:
                concatenated_tensors.append(batch_result[key])
            concatenated_results[key] = torch.cat(concatenated_tensors, dim=1)
        
        return concatenated_results
    
    def _enhance_geometry_features(
        self,
        vggt_predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Enhance and post-process geometry features."""
        
        enhanced = vggt_predictions.copy()
        
        # Apply confidence filtering to depth maps
        if 'depth' in enhanced and 'depth_conf' in enhanced:
            depth = enhanced['depth']
            depth_conf = enhanced['depth_conf']
            
            # Mask low-confidence depth values
            conf_mask = depth_conf > self.config.depth_confidence_threshold
            filtered_depth = depth.clone()
            filtered_depth[~conf_mask.unsqueeze(-1)] = 0.0
            
            enhanced['filtered_depth'] = filtered_depth
            enhanced['depth_valid_ratio'] = conf_mask.float().mean(dim=[-2, -1])  # [B, S]
        
        # Filter world points by confidence
        if 'world_points' in enhanced and 'world_points_conf' in enhanced:
            world_points = enhanced['world_points']
            world_points_conf = enhanced['world_points_conf']
            
            # Create validity mask
            points_mask = world_points_conf > self.config.world_points_confidence_threshold
            filtered_points = world_points.clone()
            filtered_points[~points_mask.unsqueeze(-1)] = 0.0
            
            enhanced['filtered_world_points'] = filtered_points
            enhanced['world_points_valid_ratio'] = points_mask.float().mean(dim=[-2, -1])  # [B, S]
        
        # Compute spatial statistics for sampling algorithm
        enhanced['spatial_statistics'] = self._compute_spatial_statistics(enhanced)
        
        return enhanced
    
    def _compute_spatial_statistics(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute spatial statistics for space-aware sampling."""
        
        stats = {}
        
        if 'world_points' in predictions and 'world_points_conf' in predictions:
            world_points = predictions['world_points']  # [B, S, H, W, 3]
            world_points_conf = predictions['world_points_conf']  # [B, S, H, W]
            
            # Compute scene extents per frame
            valid_mask = world_points_conf > self.config.world_points_confidence_threshold
            
            # Scene bounding box per frame
            frame_extents = []
            for s in range(world_points.shape[1]):
                frame_points = world_points[:, s]  # [B, H, W, 3]
                frame_mask = valid_mask[:, s]  # [B, H, W]
                
                if frame_mask.sum() > 0:
                    valid_points = frame_points[frame_mask]  # [N_valid, 3]
                    min_coords = valid_points.min(dim=0)[0]  # [3]
                    max_coords = valid_points.max(dim=0)[0]  # [3]
                    extent = max_coords - min_coords  # [3]
                else:
                    extent = torch.zeros(3, device=world_points.device)
                
                frame_extents.append(extent)
            
            stats['scene_extents'] = torch.stack(frame_extents, dim=0)  # [S, 3]
            stats['scene_volumes'] = torch.prod(stats['scene_extents'], dim=-1)  # [S]
        
        # Camera pose statistics
        if 'pose_enc' in predictions:
            pose_enc = predictions['pose_enc']  # [B, S, 9]
            
            # Extract translation and rotation components
            translations = pose_enc[:, :, :3]  # [B, S, 3]
            quaternions = pose_enc[:, :, 3:7]  # [B, S, 4]
            
            # Translation distances between consecutive frames
            if translations.shape[1] > 1:
                trans_diffs = torch.diff(translations, dim=1)  # [B, S-1, 3]
                trans_distances = torch.norm(trans_diffs, dim=-1)  # [B, S-1]
                stats['translation_distances'] = trans_distances
                
                # Rotation differences (simplified)
                quat_diffs = torch.diff(quaternions, dim=1)  # [B, S-1, 4]
                rotation_changes = torch.norm(quat_diffs, dim=-1)  # [B, S-1]
                stats['rotation_changes'] = rotation_changes
        
        return stats
    
    def _generate_cache_key(
        self,
        video_frames: torch.Tensor,
        frame_indices: Optional[List[int]]
    ) -> str:
        """Generate cache key for geometry features."""
        
        # Create hash based on frame tensor properties and indices
        frame_hash = hash((
            video_frames.shape,
            video_frames.device,
            video_frames.dtype,
            str(frame_indices) if frame_indices else "all"
        ))
        
        return f"vggt_geometry_{frame_hash}"
    
    def _update_performance_stats(self, num_frames: int, processing_time: float):
        """Update performance tracking statistics."""
        
        self._processing_stats['total_frames_processed'] += num_frames
        self._processing_stats['total_processing_time'] += processing_time
        
        # Track batch size usage
        batch_key = f"batch_{min(num_frames, self.config.batch_size_limit)}"
        if batch_key not in self._processing_stats['batch_counts']:
            self._processing_stats['batch_counts'][batch_key] = 0
        self._processing_stats['batch_counts'][batch_key] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        stats = self._processing_stats.copy()
        
        # Compute derived statistics
        if stats['total_frames_processed'] > 0:
            stats['average_fps'] = stats['total_frames_processed'] / stats['total_processing_time']
            stats['average_processing_time_per_frame'] = stats['total_processing_time'] / stats['total_frames_processed']
        else:
            stats['average_fps'] = 0.0
            stats['average_processing_time_per_frame'] = 0.0
        
        # Cache statistics
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['cache_hits'] + len(self._geometry_cache), 1)
        )
        
        return stats
    
    def clear_cache(self):
        """Clear all cached features."""
        self._geometry_cache.clear()
        self._feature_cache.clear()
        logger.info("VGGT processor cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the VGGT model."""
        
        total_params = sum(p.numel() for p in self.vggt.parameters())
        trainable_params = sum(p.numel() for p in self.vggt.parameters() if p.requires_grad)
        
        return {
            'model_type': 'VGGT',
            'img_size': self.config.img_size,
            'patch_size': self.config.patch_size,
            'embed_dim': self.config.embed_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'dtype': str(self.config.dtype)
        }


def create_vggt_processor(
    img_size: int = 518,
    patch_size: int = 14,
    embed_dim: int = 1024,
    batch_size_limit: int = 8,
    device: str = "cuda",
    verbose: bool = True
) -> VGGTProcessor:
    """
    Factory function to create VGGT processor.
    
    Args:
        img_size: Input image size
        patch_size: Vision transformer patch size
        embed_dim: Embedding dimension
        batch_size_limit: Maximum batch size for processing
        device: Computing device
        verbose: Enable detailed logging
        
    Returns:
        Configured VGGTProcessor instance
    """
    config = VGGTIntegrationConfig(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        batch_size_limit=batch_size_limit,
        device=device,
        verbose=verbose
    )
    
    return VGGTProcessor(config)


# Testing and validation
if __name__ == "__main__":
    # Create VGGT processor
    processor = create_vggt_processor(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("VGGT Processor created successfully!")
    print(f"Model info: {processor.get_model_info()}")
    
    # Test processing
    batch_size, total_frames = 1, 16
    test_video = torch.randn(batch_size, total_frames, 3, 518, 518)
    
    with torch.no_grad():
        result = processor(test_video)
    
    print(f"Processing result keys: {list(result.keys())}")
    print(f"Performance stats: {processor.get_performance_stats()}")
    
    # Test with frame indices
    selected_indices = [0, 4, 8, 12]
    with torch.no_grad():
        partial_result = processor(test_video, frame_indices=selected_indices)
    
    print(f"Partial processing completed for {len(selected_indices)} frames")