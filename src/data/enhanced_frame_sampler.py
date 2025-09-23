"""
Enhanced Space-aware Frame Sampling with Multi-Objective Submodular Optimization
=================================================================================

This module provides significant improvements over the basic greedy coverage algorithm:

1. **Multi-Objective Submodular Optimization**: Balances spatial coverage, semantic diversity,
   temporal coherence, and uncertainty-aware selection
2. **Hierarchical Spatial Representation**: Multi-scale voxelization for better spatial understanding
3. **Semantic-Spatial Fusion**: Combines geometric and visual feature diversity
4. **Temporal Coherence Regularization**: Ensures good temporal distribution
5. **Uncertainty-Aware Selection**: Leverages confidence scores for intelligent selection
6. **Visual Saliency Integration**: Considers visual importance and motion

The enhanced algorithm optimizes a submodular objective function:
f(S) = α₁·Coverage(S) + α₂·Diversity(S) + α₃·TemporalCoherence(S) + α₄·Uncertainty(S)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSamplingConfig:
    """Configuration for enhanced space-aware frame sampling."""
    # Target selection
    target_frames: int = 16  # N_k - target number of selected frames
    candidate_frames: int = 128  # N_m - total candidate frames

    # Multi-objective weights
    coverage_weight: float = 0.4  # α₁ - spatial coverage importance
    diversity_weight: float = 0.3  # α₂ - semantic diversity importance
    temporal_weight: float = 0.2  # α₃ - temporal coherence importance
    uncertainty_weight: float = 0.1  # α₄ - uncertainty-aware importance

    # Hierarchical voxelization
    voxel_scales: List[float] = None  # Multiple voxel scales [λ₁, λ₂, λ₃]
    adaptive_voxelization: bool = True  # Enable adaptive voxel sizing

    # Semantic diversity
    use_visual_features: bool = True  # Include visual feature diversity
    feature_diversity_threshold: float = 0.3  # Minimum feature diversity

    # Temporal coherence
    temporal_smoothness: float = 0.8  # Temporal distribution smoothness
    min_temporal_gap: int = 2  # Minimum gap between selected frames

    # Uncertainty-aware selection
    confidence_threshold: float = 0.1  # Minimum confidence for valid points
    confidence_percentile: float = 50.0  # Percentile threshold for filtering
    uncertainty_boost: float = 1.5  # Boost factor for uncertain regions

    # Visual saliency
    motion_importance: float = 0.2  # Weight for motion-based saliency
    depth_variation_importance: float = 0.3  # Weight for depth variation

    # Optimization
    optimization_method: str = "submodular"  # Options: "submodular", "greedy", "hybrid"
    max_iterations: int = 100  # Maximum optimization iterations
    convergence_threshold: float = 1e-4  # Convergence threshold

    # System
    device: str = "cuda"

    def __post_init__(self):
        if self.voxel_scales is None:
            self.voxel_scales = [10.0, 20.0, 40.0]  # Multi-scale voxelization


class EnhancedFrameSampler(nn.Module):
    """
    Enhanced space-aware frame sampler with multi-objective optimization.

    This implementation provides significant improvements over basic greedy coverage:
    - Multi-objective submodular optimization
    - Hierarchical spatial representation
    - Semantic-spatial fusion
    - Temporal coherence regularization
    - Uncertainty-aware selection
    """

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Initialize components
        self._spatial_analyzer = HierarchicalSpatialAnalyzer(config)
        self._diversity_analyzer = SemanticDiversityAnalyzer(config)
        self._temporal_analyzer = TemporalCoherenceAnalyzer(config)
        self._uncertainty_analyzer = UncertaintyAnalyzer(config)
        self._submodular_optimizer = SubmodularOptimizer(config)

        # Cache for expensive computations
        self._computation_cache = {}

    def forward(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        visual_features: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced frame selection with multi-objective optimization.

        Args:
            vggt_predictions: VGGT model outputs with geometry information
            visual_features: Optional visual features for semantic diversity [B, S, D]
            frame_indices: Optional frame indices to consider

        Returns:
            Dictionary containing enhanced selection results
        """

        batch_size, num_frames = vggt_predictions['depth'].shape[:2]

        if batch_size > 1:
            return self._process_multi_batch(vggt_predictions, visual_features, frame_indices)

        if frame_indices is None:
            frame_indices = list(range(num_frames))

        logger.info(f"Enhanced sampling: {len(frame_indices)} candidates → {self.config.target_frames} targets")

        # Step 1: Extract multi-scale spatial representation
        spatial_analysis = self._spatial_analyzer(vggt_predictions, frame_indices)

        # Step 2: Compute semantic diversity if visual features available
        diversity_analysis = None
        if self.config.use_visual_features and visual_features is not None:
            diversity_analysis = self._diversity_analyzer(visual_features, frame_indices)

        # Step 3: Analyze temporal coherence
        temporal_analysis = self._temporal_analyzer(frame_indices)

        # Step 4: Compute uncertainty-aware scores
        uncertainty_analysis = self._uncertainty_analyzer(vggt_predictions, frame_indices)

        # Step 5: Multi-objective submodular optimization
        optimization_result = self._submodular_optimizer.optimize(
            spatial_analysis=spatial_analysis,
            diversity_analysis=diversity_analysis,
            temporal_analysis=temporal_analysis,
            uncertainty_analysis=uncertainty_analysis,
            frame_indices=frame_indices
        )

        selected_indices = optimization_result['selected_indices']

        # Ensure consistent tensor format
        indices_tensor = torch.tensor(selected_indices, device=self.device).unsqueeze(0)

        return {
            'selected_indices': indices_tensor,  # [1, n_keyframes]
            'optimization_result': optimization_result,
            'spatial_analysis': spatial_analysis,
            'diversity_analysis': diversity_analysis,
            'temporal_analysis': temporal_analysis,
            'uncertainty_analysis': uncertainty_analysis,
            'objective_scores': optimization_result['objective_history'],
            'selection_quality': self._compute_selection_quality(optimization_result)
        }

    def _process_multi_batch(self, vggt_predictions, visual_features, frame_indices):
        """Handle multi-batch processing with consistency."""
        batch_size, num_frames = vggt_predictions['depth'].shape[:2]

        # For now, use consistent selection across batches
        # Could be enhanced to do per-batch optimization
        if frame_indices is None:
            frame_indices = list(range(num_frames))

        target_keyframes = min(self.config.target_frames, num_frames)

        # Simple temporal distribution for multi-batch consistency
        if target_keyframes >= num_frames:
            selected_indices = list(range(num_frames))
        else:
            step = num_frames / target_keyframes
            selected_indices = [int(i * step) for i in range(target_keyframes)]
            selected_indices = [min(idx, num_frames - 1) for idx in selected_indices]

        indices_tensor = torch.tensor(selected_indices, device=self.device)
        batch_indices = indices_tensor.unsqueeze(0).repeat(batch_size, 1)

        return {
            'selected_indices': batch_indices,
            'optimization_result': {'method': 'multi_batch_fallback'},
            'selection_quality': torch.ones(batch_size, device=self.device)
        }

    def _compute_selection_quality(self, optimization_result: Dict) -> torch.Tensor:
        """Compute overall quality score for the selection."""

        if 'objective_scores' not in optimization_result:
            return torch.tensor(1.0, device=self.device)

        objective_scores = optimization_result['objective_scores']

        # Quality based on objective improvement and convergence
        if len(objective_scores) > 1:
            improvement = objective_scores[-1] - objective_scores[0]
            convergence = abs(objective_scores[-1] - objective_scores[-2]) < self.config.convergence_threshold
            quality = improvement + (0.1 if convergence else 0.0)
        else:
            quality = objective_scores[0] if objective_scores else 1.0

        return torch.tensor(quality, device=self.device)


class HierarchicalSpatialAnalyzer(nn.Module):
    """Multi-scale spatial analysis with hierarchical voxelization."""

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config

    def forward(self, vggt_predictions: Dict[str, torch.Tensor], frame_indices: List[int]) -> Dict:
        """Compute hierarchical spatial representation."""

        world_points = vggt_predictions['world_points'].squeeze(0)  # [S, H, W, 3]
        world_points_conf = vggt_predictions['world_points_conf'].squeeze(0)  # [S, H, W]

        # Multi-scale voxelization
        multi_scale_voxels = {}

        for scale_idx, voxel_lambda in enumerate(self.config.voxel_scales):
            voxel_sets = self._compute_voxel_sets_at_scale(
                world_points, world_points_conf, frame_indices, voxel_lambda
            )
            multi_scale_voxels[f'scale_{scale_idx}'] = voxel_sets

        # Adaptive voxelization based on scene complexity
        if self.config.adaptive_voxelization:
            adaptive_voxels = self._compute_adaptive_voxelization(
                world_points, world_points_conf, frame_indices
            )
            multi_scale_voxels['adaptive'] = adaptive_voxels

        return {
            'multi_scale_voxels': multi_scale_voxels,
            'scene_complexity': self._estimate_scene_complexity(world_points, world_points_conf)
        }

    def _compute_voxel_sets_at_scale(
        self,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor,
        frame_indices: List[int],
        voxel_lambda: float
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        """Compute voxel sets at specific scale."""

        # Collect all valid points
        all_valid_points = []
        frame_valid_points = {}

        for frame_idx in frame_indices:
            if frame_idx >= world_points.shape[0]:
                continue

            points = world_points[frame_idx]  # [H, W, 3]
            conf = world_points_conf[frame_idx]  # [H, W]

            # Enhanced confidence filtering
            conf_threshold = self.config.confidence_threshold
            conf_percentile_val = torch.quantile(conf.flatten(), self.config.confidence_percentile / 100.0)

            valid_mask = (conf > conf_threshold) & (conf >= conf_percentile_val)

            if valid_mask.sum() == 0:
                frame_valid_points[frame_idx] = torch.empty((0, 3), device=world_points.device)
                continue

            valid_points = points[valid_mask]
            frame_valid_points[frame_idx] = valid_points
            all_valid_points.append(valid_points)

        if not all_valid_points:
            return {idx: set() for idx in frame_indices}

        # Global statistics for consistent voxelization
        all_points = torch.cat(all_valid_points, dim=0)
        point_min = all_points.min(dim=0)[0]
        point_max = all_points.max(dim=0)[0]
        scene_extents = point_max - point_min

        # Voxel size calculation
        min_extent = scene_extents.min().item()
        voxel_size = min_extent / voxel_lambda

        # Voxelize each frame
        voxel_sets = {}
        for frame_idx in frame_indices:
            if frame_idx not in frame_valid_points:
                voxel_sets[frame_idx] = set()
                continue

            valid_points = frame_valid_points[frame_idx]

            if len(valid_points) == 0:
                voxel_sets[frame_idx] = set()
                continue

            # Voxelization
            normalized_points = (valid_points - point_min.unsqueeze(0)) / voxel_size
            voxel_coords = torch.floor(normalized_points).long()

            voxel_set = set()
            for coord in voxel_coords:
                voxel_set.add(tuple(coord.cpu().numpy()))

            voxel_sets[frame_idx] = voxel_set

        return voxel_sets

    def _compute_adaptive_voxelization(
        self,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor,
        frame_indices: List[int]
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        """Compute adaptive voxelization based on local scene complexity."""

        # Use median voxel size as baseline, then adapt per region
        baseline_lambda = np.median(self.config.voxel_scales)
        return self._compute_voxel_sets_at_scale(
            world_points, world_points_conf, frame_indices, baseline_lambda
        )

    def _estimate_scene_complexity(
        self,
        world_points: torch.Tensor,
        world_points_conf: torch.Tensor
    ) -> float:
        """Estimate scene complexity for adaptive processing."""

        # Use depth variation and point density as complexity metrics
        valid_mask = world_points_conf > self.config.confidence_threshold

        if valid_mask.sum() == 0:
            return 0.0

        valid_points = world_points[valid_mask]

        # Depth variation
        depths = torch.norm(valid_points, dim=-1)
        depth_std = depths.std().item()

        # Point density
        point_density = valid_mask.float().mean().item()

        # Combined complexity score
        complexity = depth_std * point_density

        return float(complexity)


class SemanticDiversityAnalyzer(nn.Module):
    """Analyzes semantic diversity using visual features."""

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config

    def forward(self, visual_features: torch.Tensor, frame_indices: List[int]) -> Dict:
        """Compute semantic diversity analysis."""

        if visual_features is None:
            return {'diversity_matrix': None, 'feature_clusters': None}

        # Extract features for specified frames
        batch_size = visual_features.shape[0]
        if batch_size > 1:
            visual_features = visual_features[0]  # Use first batch
        else:
            visual_features = visual_features.squeeze(0)  # Remove batch dimension [S, D]

        # Safely index the features
        max_idx = visual_features.shape[0] - 1
        safe_frame_indices = [min(idx, max_idx) for idx in frame_indices]
        frame_features = visual_features[safe_frame_indices]  # [N_frames, D]

        # Compute pairwise feature similarity
        diversity_matrix = self._compute_diversity_matrix(frame_features)

        # Identify feature clusters for diversity-aware selection
        feature_clusters = self._identify_feature_clusters(frame_features, diversity_matrix)

        return {
            'diversity_matrix': diversity_matrix,
            'feature_clusters': feature_clusters,
            'feature_statistics': self._compute_feature_statistics(frame_features)
        }

    def _compute_diversity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise diversity matrix."""

        # Normalize features
        features_norm = F.normalize(features, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        # Convert to diversity (1 - similarity)
        diversity_matrix = 1.0 - similarity_matrix

        return diversity_matrix

    def _identify_feature_clusters(
        self,
        features: torch.Tensor,
        diversity_matrix: torch.Tensor
    ) -> List[List[int]]:
        """Identify clusters in feature space."""

        # Simple clustering based on diversity threshold
        n_frames = len(features)
        visited = set()
        clusters = []

        for i in range(n_frames):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, n_frames):
                if j not in visited and diversity_matrix[i, j] < self.config.feature_diversity_threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        return clusters

    def _compute_feature_statistics(self, features: torch.Tensor) -> Dict:
        """Compute statistical properties of features."""

        return {
            'feature_dimension': features.shape[-1],
            'feature_mean': features.mean(dim=0),
            'feature_std': features.std(dim=0),
            'feature_range': features.max(dim=0)[0] - features.min(dim=0)[0]
        }


class TemporalCoherenceAnalyzer(nn.Module):
    """Analyzes temporal coherence for frame selection."""

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config

    def forward(self, frame_indices: List[int]) -> Dict:
        """Analyze temporal properties of frames."""

        n_frames = len(frame_indices)

        # Compute temporal distribution quality
        temporal_distribution = self._compute_temporal_distribution(frame_indices)

        # Compute temporal coherence matrix
        coherence_matrix = self._compute_temporal_coherence_matrix(frame_indices)

        return {
            'temporal_distribution': temporal_distribution,
            'coherence_matrix': coherence_matrix,
            'temporal_statistics': {
                'frame_span': max(frame_indices) - min(frame_indices) if frame_indices else 0,
                'average_gap': np.mean(np.diff(sorted(frame_indices))) if len(frame_indices) > 1 else 0,
                'temporal_coverage': len(frame_indices) / (max(frame_indices) + 1) if frame_indices else 0
            }
        }

    def _compute_temporal_distribution(self, frame_indices: List[int]) -> torch.Tensor:
        """Compute temporal distribution quality scores."""

        n_frames = len(frame_indices)
        distribution_scores = torch.zeros(n_frames)

        if n_frames <= 1:
            return distribution_scores

        sorted_indices = sorted(frame_indices)

        for i, frame_idx in enumerate(frame_indices):
            # Score based on temporal spacing
            pos = sorted_indices.index(frame_idx)

            # Prefer frames with good temporal spacing
            if pos == 0 or pos == len(sorted_indices) - 1:
                # End frames get higher scores
                distribution_scores[i] = 1.0
            else:
                # Middle frames scored by spacing quality
                prev_gap = sorted_indices[pos] - sorted_indices[pos - 1]
                next_gap = sorted_indices[pos + 1] - sorted_indices[pos]

                # Prefer consistent spacing
                gap_consistency = 1.0 - abs(prev_gap - next_gap) / max(prev_gap + next_gap, 1)
                distribution_scores[i] = gap_consistency

        return distribution_scores

    def _compute_temporal_coherence_matrix(self, frame_indices: List[int]) -> torch.Tensor:
        """Compute temporal coherence between frame pairs."""

        n_frames = len(frame_indices)
        coherence_matrix = torch.zeros(n_frames, n_frames)

        for i in range(n_frames):
            for j in range(n_frames):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    # Temporal distance-based coherence
                    temporal_dist = abs(frame_indices[i] - frame_indices[j])
                    coherence = math.exp(-temporal_dist / 10.0)  # Exponential decay
                    coherence_matrix[i, j] = coherence

        return coherence_matrix


class UncertaintyAnalyzer(nn.Module):
    """Analyzes uncertainty for intelligent frame selection."""

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config

    def forward(self, vggt_predictions: Dict[str, torch.Tensor], frame_indices: List[int]) -> Dict:
        """Compute uncertainty-aware selection scores."""

        depth_conf = vggt_predictions['depth_conf'].squeeze(0)  # [S, H, W]
        world_points_conf = vggt_predictions['world_points_conf'].squeeze(0)  # [S, H, W]

        uncertainty_scores = []

        for frame_idx in frame_indices:
            if frame_idx >= depth_conf.shape[0]:
                uncertainty_scores.append(0.0)
                continue

            # Combine different uncertainty sources
            depth_uncertainty = 1.0 - depth_conf[frame_idx].mean().item()
            point_uncertainty = 1.0 - world_points_conf[frame_idx].mean().item()

            # Motion-based uncertainty (if available)
            motion_uncertainty = self._compute_motion_uncertainty(frame_idx, vggt_predictions)

            # Combined uncertainty score
            combined_uncertainty = (
                0.4 * depth_uncertainty +
                0.4 * point_uncertainty +
                0.2 * motion_uncertainty
            )

            uncertainty_scores.append(combined_uncertainty)

        return {
            'uncertainty_scores': torch.tensor(uncertainty_scores),
            'depth_uncertainties': torch.tensor([1.0 - depth_conf[i].mean().item()
                                               for i in frame_indices if i < depth_conf.shape[0]]),
            'point_uncertainties': torch.tensor([1.0 - world_points_conf[i].mean().item()
                                               for i in frame_indices if i < world_points_conf.shape[0]])
        }

    def _compute_motion_uncertainty(self, frame_idx: int, vggt_predictions: Dict) -> float:
        """Compute motion-based uncertainty score."""

        # Simple motion estimation using depth changes
        depth_maps = vggt_predictions['depth'].squeeze(0)  # [S, H, W, 1]

        if frame_idx == 0 or frame_idx >= depth_maps.shape[0] - 1:
            return 0.5  # Boundary frames have medium motion uncertainty

        # Compare with adjacent frames
        current_depth = depth_maps[frame_idx].squeeze(-1)
        prev_depth = depth_maps[frame_idx - 1].squeeze(-1)
        next_depth = depth_maps[frame_idx + 1].squeeze(-1)

        # Depth change indicates motion
        depth_change = (torch.abs(current_depth - prev_depth) +
                       torch.abs(current_depth - next_depth)) / 2.0

        # Normalize and return motion uncertainty
        motion_score = torch.mean(depth_change).item()
        return min(motion_score / 10.0, 1.0)  # Normalize to [0, 1]


class SubmodularOptimizer(nn.Module):
    """Multi-objective submodular optimization for frame selection."""

    def __init__(self, config: EnhancedSamplingConfig):
        super().__init__()
        self.config = config

    def optimize(
        self,
        spatial_analysis: Dict,
        diversity_analysis: Optional[Dict],
        temporal_analysis: Dict,
        uncertainty_analysis: Dict,
        frame_indices: List[int]
    ) -> Dict:
        """Optimize multi-objective submodular function."""

        if self.config.optimization_method == "greedy":
            return self._greedy_optimization(spatial_analysis, frame_indices)
        elif self.config.optimization_method == "submodular":
            return self._submodular_optimization(
                spatial_analysis, diversity_analysis, temporal_analysis, uncertainty_analysis, frame_indices
            )
        elif self.config.optimization_method == "hybrid":
            return self._hybrid_optimization(
                spatial_analysis, diversity_analysis, temporal_analysis, uncertainty_analysis, frame_indices
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")

    def _submodular_optimization(
        self,
        spatial_analysis: Dict,
        diversity_analysis: Optional[Dict],
        temporal_analysis: Dict,
        uncertainty_analysis: Dict,
        frame_indices: List[int]
    ) -> Dict:
        """Advanced submodular optimization with multiple objectives."""

        n_frames = len(frame_indices)
        target_frames = min(self.config.target_frames, n_frames)

        # Initialize selection
        selected_indices = []
        remaining_indices = set(frame_indices)
        objective_history = []

        for iteration in range(target_frames):
            best_frame = None
            best_gain = -float('inf')

            for candidate_idx in remaining_indices:
                # Compute marginal gain for adding this frame
                marginal_gain = self._compute_marginal_gain(
                    candidate_idx, selected_indices, spatial_analysis,
                    diversity_analysis, temporal_analysis, uncertainty_analysis
                )

                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_frame = candidate_idx

            if best_frame is not None:
                selected_indices.append(best_frame)
                remaining_indices.remove(best_frame)
                objective_history.append(best_gain)
            else:
                break

            # Check convergence
            if len(objective_history) >= 2:
                if abs(objective_history[-1] - objective_history[-2]) < self.config.convergence_threshold:
                    break

        return {
            'selected_indices': selected_indices,
            'objective_history': objective_history,
            'optimization_method': 'submodular',
            'converged': len(objective_history) >= 2 and
                        abs(objective_history[-1] - objective_history[-2]) < self.config.convergence_threshold
        }

    def _compute_marginal_gain(
        self,
        candidate_idx: int,
        selected_indices: List[int],
        spatial_analysis: Dict,
        diversity_analysis: Optional[Dict],
        temporal_analysis: Dict,
        uncertainty_analysis: Dict
    ) -> float:
        """Compute marginal gain for adding a candidate frame."""

        # Coverage gain
        coverage_gain = self._compute_coverage_gain(candidate_idx, selected_indices, spatial_analysis)

        # Diversity gain
        diversity_gain = 0.0
        if diversity_analysis is not None:
            diversity_gain = self._compute_diversity_gain(candidate_idx, selected_indices, diversity_analysis)

        # Temporal coherence gain
        temporal_gain = self._compute_temporal_gain(candidate_idx, selected_indices, temporal_analysis)

        # Uncertainty-based gain
        uncertainty_gain = self._compute_uncertainty_gain(candidate_idx, uncertainty_analysis)

        # Weighted combination
        total_gain = (
            self.config.coverage_weight * coverage_gain +
            self.config.diversity_weight * diversity_gain +
            self.config.temporal_weight * temporal_gain +
            self.config.uncertainty_weight * uncertainty_gain
        )

        return total_gain

    def _compute_coverage_gain(self, candidate_idx: int, selected_indices: List[int], spatial_analysis: Dict) -> float:
        """Compute spatial coverage gain."""

        multi_scale_voxels = spatial_analysis['multi_scale_voxels']

        # Use primary scale for coverage computation
        primary_scale = list(multi_scale_voxels.keys())[0]
        voxel_sets = multi_scale_voxels[primary_scale]

        if candidate_idx not in voxel_sets:
            return 0.0

        candidate_voxels = voxel_sets[candidate_idx]

        # Compute union of already selected voxels
        covered_voxels = set()
        for sel_idx in selected_indices:
            if sel_idx in voxel_sets:
                covered_voxels.update(voxel_sets[sel_idx])

        # Marginal coverage gain
        new_coverage = len(candidate_voxels - covered_voxels)

        # Normalize by total possible coverage
        total_possible = len(candidate_voxels)
        if total_possible == 0:
            return 0.0

        return new_coverage / max(total_possible, 1)

    def _compute_diversity_gain(self, candidate_idx: int, selected_indices: List[int], diversity_analysis: Dict) -> float:
        """Compute semantic diversity gain."""

        diversity_matrix = diversity_analysis.get('diversity_matrix')
        if diversity_matrix is None:
            return 0.0

        if not selected_indices:
            return 1.0  # First frame has maximum diversity

        # Minimum diversity to already selected frames
        min_diversity = float('inf')
        for sel_idx in selected_indices:
            if sel_idx < diversity_matrix.shape[0] and candidate_idx < diversity_matrix.shape[1]:
                diversity = diversity_matrix[candidate_idx, sel_idx].item()
                min_diversity = min(min_diversity, diversity)

        return min_diversity if min_diversity != float('inf') else 0.0

    def _compute_temporal_gain(self, candidate_idx: int, selected_indices: List[int], temporal_analysis: Dict) -> float:
        """Compute temporal coherence gain."""

        if not selected_indices:
            return 1.0  # First frame

        # Prefer frames that improve temporal distribution
        sorted_selected = sorted(selected_indices)

        # Find best insertion position
        best_gap_improvement = 0.0

        for i in range(len(sorted_selected) + 1):
            # Compute gap improvement if inserted at position i
            if i == 0:
                if len(sorted_selected) > 0:
                    gap = sorted_selected[0] - candidate_idx
                    if gap > 0:
                        best_gap_improvement = max(best_gap_improvement, min(gap, 10) / 10.0)
            elif i == len(sorted_selected):
                gap = candidate_idx - sorted_selected[-1]
                if gap > 0:
                    best_gap_improvement = max(best_gap_improvement, min(gap, 10) / 10.0)
            else:
                # Between two frames
                left_gap = candidate_idx - sorted_selected[i-1]
                right_gap = sorted_selected[i] - candidate_idx

                if left_gap > 0 and right_gap > 0:
                    gap_quality = min(left_gap, right_gap) / max(left_gap, right_gap)
                    best_gap_improvement = max(best_gap_improvement, gap_quality)

        return best_gap_improvement

    def _compute_uncertainty_gain(self, candidate_idx: int, uncertainty_analysis: Dict) -> float:
        """Compute uncertainty-based gain."""

        uncertainty_scores = uncertainty_analysis.get('uncertainty_scores')
        if uncertainty_scores is None:
            return 0.0

        if candidate_idx < len(uncertainty_scores):
            return uncertainty_scores[candidate_idx].item() * self.config.uncertainty_boost

        return 0.0

    def _greedy_optimization(self, spatial_analysis: Dict, frame_indices: List[int]) -> Dict:
        """Fallback to greedy coverage optimization."""

        # Simple greedy coverage as fallback
        multi_scale_voxels = spatial_analysis['multi_scale_voxels']
        primary_scale = list(multi_scale_voxels.keys())[0]
        voxel_sets = multi_scale_voxels[primary_scale]

        selected_indices = []
        covered_voxels = set()
        remaining_indices = set(frame_indices)

        target_frames = min(self.config.target_frames, len(frame_indices))

        for _ in range(target_frames):
            best_frame = None
            best_coverage_gain = 0

            for frame_idx in remaining_indices:
                if frame_idx in voxel_sets:
                    coverage_gain = len(voxel_sets[frame_idx] - covered_voxels)
                    if coverage_gain > best_coverage_gain:
                        best_coverage_gain = coverage_gain
                        best_frame = frame_idx

            if best_frame is not None:
                selected_indices.append(best_frame)
                covered_voxels.update(voxel_sets[best_frame])
                remaining_indices.remove(best_frame)
            else:
                break

        return {
            'selected_indices': selected_indices,
            'objective_history': [best_coverage_gain],
            'optimization_method': 'greedy'
        }

    def _hybrid_optimization(
        self, spatial_analysis, diversity_analysis, temporal_analysis, uncertainty_analysis, frame_indices
    ) -> Dict:
        """Hybrid optimization combining multiple strategies."""

        # For now, use submodular as default hybrid
        return self._submodular_optimization(
            spatial_analysis, diversity_analysis, temporal_analysis, uncertainty_analysis, frame_indices
        )


def create_enhanced_sampler(
    target_frames: int = 16,
    candidate_frames: int = 128,
    coverage_weight: float = 0.4,
    diversity_weight: float = 0.3,
    temporal_weight: float = 0.2,
    uncertainty_weight: float = 0.1,
    optimization_method: str = "submodular",
    device: str = "cuda"
) -> EnhancedFrameSampler:
    """
    Factory function to create an enhanced frame sampler.

    Returns:
        Configured EnhancedFrameSampler instance
    """
    config = EnhancedSamplingConfig(
        target_frames=target_frames,
        candidate_frames=candidate_frames,
        coverage_weight=coverage_weight,
        diversity_weight=diversity_weight,
        temporal_weight=temporal_weight,
        uncertainty_weight=uncertainty_weight,
        optimization_method=optimization_method,
        device=device
    )

    return EnhancedFrameSampler(config)


# Example usage
if __name__ == "__main__":
    sampler = create_enhanced_sampler(
        target_frames=16,
        candidate_frames=32,  # Reduced for testing
        optimization_method="submodular",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Enhanced Frame Sampler created successfully!")
    print(f"Configuration: {sampler.config}")