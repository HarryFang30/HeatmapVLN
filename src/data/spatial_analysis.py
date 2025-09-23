"""
Geometry-based Spatial Novelty Detection
========================================

This module provides advanced spatial analysis capabilities for VLN tasks,
focusing on detecting spatial novelty and geometric changes between frames.
It works in conjunction with the space-aware frame sampler to provide
intelligent frame selection based on 3D spatial understanding.

Key Features:
- Camera pose change analysis
- 3D scene geometry novelty detection  
- Spatial overlap and occlusion analysis
- Multi-scale spatial feature comparison
- Temporal spatial dynamics tracking

This implementation extends beyond the basic greedy coverage algorithm
to provide richer spatial understanding for frame selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpatialAnalysisConfig:
    """Configuration for spatial novelty detection."""
    # Pose analysis parameters
    translation_threshold: float = 0.5  # Minimum translation for pose novelty (meters)
    rotation_threshold: float = 15.0  # Minimum rotation for pose novelty (degrees)
    
    # Geometric novelty parameters
    depth_novelty_threshold: float = 0.3  # Depth change threshold for novelty
    scene_overlap_threshold: float = 0.7  # Maximum overlap for spatial novelty
    
    # Multi-scale analysis
    spatial_grid_sizes: List[int] = None  # Grid sizes for multi-scale analysis
    
    # Temporal analysis
    temporal_window: int = 8  # Window size for temporal dynamics
    novelty_decay: float = 0.9  # Decay factor for temporal novelty
    
    def __post_init__(self):
        if self.spatial_grid_sizes is None:
            self.spatial_grid_sizes = [4, 7, 14]  # Multi-scale grid analysis


class SpatialNoveltyDetector(nn.Module):
    """
    Geometry-based spatial novelty detector for intelligent frame selection.
    
    This module analyzes 3D spatial relationships and camera dynamics to
    identify frames with high spatial novelty - frames that provide new
    geometric information not captured by previously selected frames.
    """
    
    def __init__(self, config: SpatialAnalysisConfig):
        super().__init__()
        self.config = config
        
        # Initialize spatial analysis components
        self._pose_analyzer = CameraPoseAnalyzer(config)
        self._geometry_analyzer = GeometryNoveltyAnalyzer(config)
        self._temporal_analyzer = TemporalSpatialAnalyzer(config)
        
    def forward(
        self,
        geometry_data: Dict[str, torch.Tensor],
        selected_frames: Optional[List[int]] = None,
        frame_history: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze spatial novelty for frame selection.
        
        Args:
            geometry_data: Extracted geometry from VGGT
            selected_frames: Previously selected frame indices
            frame_history: Historical analysis data
            
        Returns:
            Dictionary containing spatial novelty scores and analysis
        """
        
        if selected_frames is None:
            selected_frames = []
            
        frame_indices = geometry_data['frame_indices']
        num_frames = len(frame_indices)
        
        # Initialize novelty scores
        novelty_scores = torch.zeros(num_frames, device=geometry_data['depth_maps'].device)
        
        # Component 1: Camera pose novelty analysis
        pose_novelty = self._pose_analyzer(geometry_data, selected_frames)
        
        # Component 2: Geometric scene novelty analysis  
        geometry_novelty = self._geometry_analyzer(geometry_data, selected_frames)
        
        # Component 3: Temporal spatial dynamics analysis
        temporal_novelty = self._temporal_analyzer(
            geometry_data, selected_frames, frame_history
        )
        
        # Combine novelty components with learned weights
        alpha_pose = 0.4      # Pose change importance
        alpha_geometry = 0.4  # Geometric novelty importance  
        alpha_temporal = 0.2  # Temporal dynamics importance
        
        novelty_scores = (
            alpha_pose * pose_novelty +
            alpha_geometry * geometry_novelty +
            alpha_temporal * temporal_novelty
        )
        
        # Apply softmax for probabilistic interpretation
        novelty_probs = F.softmax(novelty_scores, dim=0)
        
        return {
            'novelty_scores': novelty_scores,
            'novelty_probabilities': novelty_probs,
            'pose_novelty': pose_novelty,
            'geometry_novelty': geometry_novelty,
            'temporal_novelty': temporal_novelty,
            'analysis_metadata': {
                'num_frames_analyzed': num_frames,
                'num_selected_frames': len(selected_frames),
                'pose_weight': alpha_pose,
                'geometry_weight': alpha_geometry,
                'temporal_weight': alpha_temporal
            }
        }
    
    def get_top_novel_frames(
        self,
        novelty_analysis: Dict[str, torch.Tensor],
        k: int = 5
    ) -> List[int]:
        """Get indices of top-k most novel frames."""
        novelty_scores = novelty_analysis['novelty_scores']
        top_k_indices = torch.topk(novelty_scores, k=min(k, len(novelty_scores)))
        return top_k_indices.indices.cpu().tolist()


class CameraPoseAnalyzer(nn.Module):
    """Analyzes camera pose changes for spatial novelty detection."""
    
    def __init__(self, config: SpatialAnalysisConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        geometry_data: Dict[str, torch.Tensor],
        selected_frames: List[int]
    ) -> torch.Tensor:
        """
        Analyze camera pose novelty.
        
        Computes spatial novelty based on camera translation and rotation
        changes relative to previously selected frames.
        """
        
        pose_enc = geometry_data['pose_enc']  # [S, 9] - absT_quaR_FoV format
        frame_indices = geometry_data['frame_indices']
        num_frames = len(frame_indices)
        
        novelty_scores = torch.ones(num_frames, device=pose_enc.device)
        
        if not selected_frames:
            # No frames selected yet - uniform novelty
            return novelty_scores
        
        # Extract pose components (absT_quaR_FoV format: [tx, ty, tz, qx, qy, qz, qw, fx, fy])
        translations = pose_enc[:, :3]  # [S, 3] - translation vectors
        quaternions = pose_enc[:, 3:7]  # [S, 4] - rotation quaternions
        
        # Analyze novelty for each frame
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in selected_frames:
                novelty_scores[i] = 0.0  # Already selected
                continue
                
            current_trans = translations[frame_idx]
            current_quat = quaternions[frame_idx]
            
            max_similarity = 0.0
            
            # Compare against all selected frames
            for selected_idx in selected_frames:
                if selected_idx < len(translations):
                    selected_trans = translations[selected_idx]
                    selected_quat = quaternions[selected_idx]
                    
                    # Translation similarity
                    trans_dist = torch.norm(current_trans - selected_trans).item()
                    trans_novelty = 1.0 - min(trans_dist / self.config.translation_threshold, 1.0)
                    
                    # Rotation similarity (quaternion dot product)
                    quat_similarity = torch.abs(torch.dot(current_quat, selected_quat)).item()
                    rot_novelty = 1.0 - quat_similarity
                    
                    # Combined pose similarity
                    pose_similarity = 0.6 * (1.0 - trans_novelty) + 0.4 * (1.0 - rot_novelty)
                    max_similarity = max(max_similarity, pose_similarity)
            
            # Novelty is inverse of maximum similarity
            novelty_scores[i] = 1.0 - max_similarity
        
        return novelty_scores


class GeometryNoveltyAnalyzer(nn.Module):
    """Analyzes 3D scene geometry for spatial novelty detection."""
    
    def __init__(self, config: SpatialAnalysisConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        geometry_data: Dict[str, torch.Tensor],
        selected_frames: List[int]
    ) -> torch.Tensor:
        """
        Analyze geometric novelty based on 3D scene structure.
        
        Uses depth maps and 3D points to detect novel geometric content
        not visible in previously selected frames.
        """
        
        depth_maps = geometry_data['depth_maps']  # [S, H, W, 1]
        depth_conf = geometry_data['depth_conf']  # [S, H, W]
        world_points = geometry_data['world_points']  # [S, H, W, 3]
        frame_indices = geometry_data['frame_indices']
        
        num_frames = len(frame_indices)
        novelty_scores = torch.ones(num_frames, device=depth_maps.device)
        
        if not selected_frames:
            return novelty_scores
        
        # Multi-scale geometric analysis
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in selected_frames:
                novelty_scores[i] = 0.0
                continue
                
            current_depth = depth_maps[frame_idx].squeeze(-1)  # [H, W]
            current_conf = depth_conf[frame_idx]  # [H, W]
            
            geometric_novelty = 0.0
            
            # Analyze against each selected frame
            for selected_idx in selected_frames:
                if selected_idx < len(depth_maps):
                    selected_depth = depth_maps[selected_idx].squeeze(-1)
                    selected_conf = depth_conf[selected_idx]
                    
                    # Multi-scale depth comparison
                    scale_novelties = []
                    for grid_size in self.config.spatial_grid_sizes:
                        novelty = self._compute_depth_novelty(
                            current_depth, selected_depth,
                            current_conf, selected_conf,
                            grid_size
                        )
                        scale_novelties.append(novelty)
                    
                    # Combine multi-scale novelties
                    frame_novelty = torch.mean(torch.stack(scale_novelties))
                    geometric_novelty = max(geometric_novelty, frame_novelty.item())
            
            novelty_scores[i] = geometric_novelty
        
        return novelty_scores
    
    def _compute_depth_novelty(
        self,
        depth1: torch.Tensor,
        depth2: torch.Tensor, 
        conf1: torch.Tensor,
        conf2: torch.Tensor,
        grid_size: int
    ) -> torch.Tensor:
        """Compute depth-based geometric novelty at specific scale."""
        
        # Get spatial dimensions for processing
        
        # Downsample to grid resolution
        depth1_grid = F.adaptive_avg_pool2d(
            depth1.unsqueeze(0).unsqueeze(0), (grid_size, grid_size)
        ).squeeze()
        depth2_grid = F.adaptive_avg_pool2d(
            depth2.unsqueeze(0).unsqueeze(0), (grid_size, grid_size)  
        ).squeeze()
        
        conf1_grid = F.adaptive_avg_pool2d(
            conf1.unsqueeze(0).unsqueeze(0), (grid_size, grid_size)
        ).squeeze()
        conf2_grid = F.adaptive_avg_pool2d(
            conf2.unsqueeze(0).unsqueeze(0), (grid_size, grid_size)
        ).squeeze()
        
        # Compute confidence-weighted depth differences
        valid_mask = (conf1_grid > 0.1) & (conf2_grid > 0.1)
        
        if valid_mask.sum() == 0:
            return torch.tensor(1.0)  # Maximum novelty if no valid comparison
            
        depth_diff = torch.abs(depth1_grid - depth2_grid)
        weighted_diff = depth_diff * (conf1_grid + conf2_grid) / 2.0
        
        # Normalize by depth scale
        depth_scale = torch.max(depth1_grid[valid_mask].max(), depth2_grid[valid_mask].max())
        normalized_diff = weighted_diff[valid_mask] / (depth_scale + 1e-8)
        
        # Novelty based on mean normalized difference
        novelty = torch.mean(normalized_diff).clamp(0, 1)
        
        return novelty


class TemporalSpatialAnalyzer(nn.Module):
    """Analyzes temporal spatial dynamics for frame selection."""
    
    def __init__(self, config: SpatialAnalysisConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        geometry_data: Dict[str, torch.Tensor],
        selected_frames: List[int],
        frame_history: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Analyze temporal spatial dynamics for novelty detection.
        
        Considers spatial changes over time and temporal coherence
        for intelligent frame selection.
        """
        
        frame_indices = geometry_data['frame_indices']
        num_frames = len(frame_indices)
        
        # Initialize with temporal position bias
        # Prefer frames that are temporally distributed
        temporal_novelty = torch.zeros(num_frames, device=geometry_data['depth_maps'].device)
        
        if not selected_frames:
            # Uniform temporal novelty for first selection
            temporal_novelty.fill_(1.0)
            return temporal_novelty
        
        # Compute temporal distribution novelty
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in selected_frames:
                temporal_novelty[i] = 0.0
                continue
                
            # Distance to nearest selected frame in temporal domain
            min_temporal_dist = min(abs(frame_idx - sel_idx) for sel_idx in selected_frames)
            
            # Normalize temporal distance
            max_possible_dist = max(frame_indices) - min(frame_indices)
            normalized_dist = min_temporal_dist / max(max_possible_dist, 1)
            
            # Temporal novelty increases with temporal distance
            temporal_novelty[i] = normalized_dist
        
        # Apply decay for temporal coherence
        if self.config.novelty_decay < 1.0:
            temporal_novelty *= self.config.novelty_decay
        
        return temporal_novelty


def create_spatial_analyzer(
    translation_threshold: float = 0.5,
    rotation_threshold: float = 15.0,
    depth_novelty_threshold: float = 0.3,
    scene_overlap_threshold: float = 0.7,
    temporal_window: int = 8,
    spatial_grid_sizes: Optional[List[int]] = None
) -> SpatialNoveltyDetector:
    """
    Factory function to create a spatial novelty detector.
    
    Returns:
        Configured SpatialNoveltyDetector instance
    """
    config = SpatialAnalysisConfig(
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold,
        depth_novelty_threshold=depth_novelty_threshold,
        scene_overlap_threshold=scene_overlap_threshold,
        temporal_window=temporal_window,
        spatial_grid_sizes=spatial_grid_sizes
    )
    
    return SpatialNoveltyDetector(config)


# Example usage
if __name__ == "__main__":
    analyzer = create_spatial_analyzer()
    print("Spatial Novelty Detector created successfully!")
    print(f"Configuration: {analyzer.config}")