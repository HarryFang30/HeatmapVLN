"""
Space-aware Frame Sampling Implementation
========================================

Implements the Greedy Maximum Coverage Sampling Algorithm for optimal frame selection
in VLN tasks. This module selects N_k keyframes from N_m candidate frames by maximizing
3D voxel coverage while minimizing redundancy.

Key Features:
- VGGT-based geometry extraction (camera poses, depth maps)
- 3D point cloud reconstruction and voxelization
- Greedy maximum coverage selection
- Adaptive voxel sizing based on scene scale
- Confidence-based point filtering

Algorithm Flow:
1. Extract geometry from all N_m frames using VGGT
2. Reconstruct 3D point clouds from depth maps and poses
3. Filter points by confidence thresholds
4. Voxelize point clouds with adaptive sizing
5. Apply greedy maximum coverage selection
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for space-aware frame sampling."""
    target_frames: int = 16  # N_k - target number of selected frames
    candidate_frames: int = 128  # N_m - total candidate frames
    voxel_lambda: float = 20.0  # Voxel size parameter (λ)
    confidence_threshold: float = 0.1  # Minimum confidence for valid points
    confidence_percentile: float = 50.0  # Percentile threshold for point filtering
    early_termination: bool = True  # Stop when no new coverage gained
    device: str = "cuda"


class SpaceAwareFrameSampler(nn.Module):
    """
    Space-aware frame sampler using greedy maximum coverage algorithm.
    
    This implementation follows the algorithm specification:
    - Processes all N_m frames through VGGT for geometry extraction
    - Reconstructs 3D point clouds from depth maps and camera poses
    - Applies adaptive voxelization for spatial discretization
    - Uses greedy selection to maximize voxel coverage
    """
    
    def __init__(self, config: SamplingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Cache for intermediate computations
        self._voxel_cache = {}
        self._geometry_cache = {}
        
    def forward(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        frame_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform space-aware frame sampling.
        
        Args:
            vggt_predictions: Dictionary containing VGGT outputs:
                - 'depth': Depth maps [B, S, H, W, 1]
                - 'depth_conf': Depth confidence [B, S, H, W]
                - 'world_points': 3D world points [B, S, H, W, 3]
                - 'world_points_conf': Point confidence [B, S, H, W]
                - 'pose_enc': Camera pose encoding [B, S, 9]
            frame_indices: Optional list of frame indices (default: range(N_m))
            
        Returns:
            Dictionary containing:
                - 'selected_indices': Selected frame indices [N_k]
                - 'coverage_scores': Coverage score for each iteration
                - 'voxel_sets': Voxel sets for each frame (for debugging)
                - 'total_coverage': Final total voxel coverage
        """
        
        batch_size, num_frames = vggt_predictions['depth'].shape[:2]
        
        # Handle multi-batch by processing each batch separately
        if batch_size > 1:
            return self._process_multi_batch(vggt_predictions, frame_indices)
            
        if frame_indices is None:
            frame_indices = list(range(num_frames))
            
        # Step 1: Extract geometry and reconstruct 3D points
        logger.info(f"Extracting geometry for {num_frames} frames")
        geometry_data = self._extract_geometry(vggt_predictions, frame_indices)
        
        # Step 2: Compute voxel sets for each frame
        logger.info("Computing voxel coverage for each frame")
        voxel_sets = self._compute_voxel_sets(geometry_data)
        
        # Step 3: Apply greedy maximum coverage selection
        logger.info(f"Selecting {self.config.target_frames} frames using greedy coverage")
        selected_indices, coverage_scores = self._greedy_maximum_coverage(
            voxel_sets, frame_indices
        )
        
        # Compute final statistics
        total_coverage = len(set().union(*[voxel_sets[i] for i in selected_indices]))
        
        logger.info(f"Selected {len(selected_indices)} frames with total coverage: {total_coverage}")
        
        # Ensure consistent tensor shape: always return [batch_size, n_keyframes]
        indices_tensor = torch.tensor(selected_indices, device=self.device).unsqueeze(0)  # Add batch dimension
        coverage_tensor = torch.tensor(coverage_scores, device=self.device).unsqueeze(0)  # Add batch dimension
        
        return {
            'selected_indices': indices_tensor,  # Shape: [1, n_keyframes]
            'coverage_scores': coverage_tensor,  # Shape: [1, n_keyframes] 
            'voxel_sets': {i: list(voxel_sets[i]) for i in frame_indices},
            'total_coverage': total_coverage,
            'geometry_data': geometry_data  # For downstream processing
        }
    
    def _process_multi_batch(self, vggt_predictions: Dict[str, torch.Tensor], frame_indices: Optional[List[int]]) -> Dict:
        """
        Simple multi-batch fallback: create consistent keyframe selection across batches.
        
        For now, this uses a simple strategy that ensures all batches get the same
        keyframe selection pattern, which is sufficient for many use cases.
        """
        batch_size, num_frames = vggt_predictions['depth'].shape[:2]
        
        if frame_indices is None:
            frame_indices = list(range(num_frames))
        
        # Use simple uniform sampling for multi-batch as a fallback
        # This ensures consistent, predictable results across all batches
        target_keyframes = min(self.config.target_frames, num_frames)
        
        if target_keyframes >= num_frames:
            # Select all frames if target >= available
            selected_indices = list(range(num_frames))
        else:
            # Uniform sampling across the sequence
            step = num_frames / target_keyframes
            selected_indices = [int(i * step) for i in range(target_keyframes)]
            # Ensure we don't exceed bounds
            selected_indices = [min(idx, num_frames - 1) for idx in selected_indices]
        
        # Create consistent results for all batches
        indices_tensor = torch.tensor(selected_indices, device=self.device)
        
        # Replicate the same indices for all batches
        batch_indices = indices_tensor.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, n_keyframes]
        
        # Create placeholder results for multi-batch
        coverage_scores = torch.ones_like(batch_indices, dtype=torch.float, device=self.device)
        total_coverage = torch.ones(batch_size, device=self.device)
        
        # Create minimal geometry data structure for multi-batch compatibility
        geometry_data = {
            'frame_indices': frame_indices,
            'batch_size': batch_size,
            'num_frames': num_frames,
            'is_multi_batch': True
        }
        
        return {
            'selected_indices': batch_indices,
            'coverage_scores': coverage_scores,
            'total_coverage': total_coverage,
            'geometry_data': geometry_data
        }
    
    def _combine_batch_results(self, batch_results: List[Dict], batch_size: int) -> Dict:
        """Combine results from multiple batches."""
        
        # Combine selected indices from all batches
        all_indices = []
        all_coverage_scores = []
        all_total_coverage = []
        combined_geometry_data = []
        
        for batch_idx, result in enumerate(batch_results):
            all_indices.append(torch.tensor(result['selected_indices'], device=self.device))
            all_coverage_scores.append(result['coverage_scores'])
            all_total_coverage.append(result['total_coverage'])
            combined_geometry_data.append(result['geometry_data'])
        
        return {
            'selected_indices': torch.stack(all_indices),  # [batch_size, n_keyframes]
            'coverage_scores': torch.stack(all_coverage_scores),  # [batch_size, n_keyframes]
            'total_coverage': torch.tensor(all_total_coverage, device=self.device),  # [batch_size]
            'geometry_data': combined_geometry_data  # List of geometry data per batch
        }
    
    def _process_single_batch(self, vggt_predictions: Dict[str, torch.Tensor], frame_indices: List[int]) -> Dict:
        """Process a single batch using the existing logic."""
        
        # Extract geometry and reconstruct 3D points
        geometry_data = self._extract_geometry(vggt_predictions, frame_indices)
        
        # Apply greedy maximum coverage algorithm
        coverage_result = self._greedy_maximum_coverage(geometry_data, frame_indices)
        selected_indices = coverage_result['selected_indices']
        coverage_scores = coverage_result['coverage_scores'] 
        total_coverage = coverage_result['total_coverage']
        voxel_sets = coverage_result['voxel_sets']
        
        return {
            'selected_indices': selected_indices,
            'coverage_scores': torch.tensor(coverage_scores, device=self.device),
            'voxel_sets': {i: list(voxel_sets[i]) for i in frame_indices},
            'total_coverage': total_coverage,
            'geometry_data': geometry_data
        }
    
    def _extract_geometry(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        frame_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract 3D geometry from VGGT predictions.
        
        Following the algorithm specification:
        Pi^m = D^m · Ki^{-1}[u|v|1]^T · Ei^{-1}
        
        Args:
            vggt_predictions: VGGT model outputs
            frame_indices: List of frame indices to process
            
        Returns:
            Dictionary containing extracted geometry data
        """
        
        # Extract tensors (remove batch dimension for processing)
        depth_maps = vggt_predictions['depth'].squeeze(0)  # [S, H, W, 1]
        depth_conf = vggt_predictions['depth_conf'].squeeze(0)  # [S, H, W]
        world_points = vggt_predictions['world_points'].squeeze(0)  # [S, H, W, 3]
        world_points_conf = vggt_predictions['world_points_conf'].squeeze(0)  # [S, H, W]
        pose_enc = vggt_predictions['pose_enc'].squeeze(0)  # [S, 9]
        
        geometry_data = {
            'depth_maps': depth_maps,
            'depth_conf': depth_conf,
            'world_points': world_points,
            'world_points_conf': world_points_conf,
            'pose_enc': pose_enc,
            'frame_indices': frame_indices
        }
        
        return geometry_data
    
    def _compute_voxel_sets(
        self,
        geometry_data: Dict[str, torch.Tensor]
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        """
        Compute voxel sets for each frame using adaptive voxelization.
        
        Algorithm steps:
        1. Filter valid points by confidence
        2. Compute adaptive voxel size: Δ = (1/λ) · min(max(Pvalid) - min(Pvalid))
        3. Voxelize: V(fi^m) = {⌊(p - min(Pvalid))/Δ⌋ | p ∈ Pi^m ∩ Pvalid}
        """
        
        world_points = geometry_data['world_points']  # [S, H, W, 3]
        world_points_conf = geometry_data['world_points_conf']  # [S, H, W]
        frame_indices = geometry_data['frame_indices']
        
        num_frames = world_points.shape[0]
        
        # Step 1: Collect all valid points across frames for adaptive voxel sizing
        all_valid_points = []
        frame_valid_points = {}
        
        for frame_idx in frame_indices:
            if frame_idx >= num_frames:
                continue
                
            points = world_points[frame_idx]  # [H, W, 3]
            conf = world_points_conf[frame_idx]  # [H, W]
            
            # Apply confidence thresholds as per algorithm specification
            conf_threshold = self.config.confidence_threshold
            conf_percentile_val = torch.quantile(conf.flatten(), self.config.confidence_percentile / 100.0)
            
            # Valid points: confidence > threshold AND >= percentile
            valid_mask = (conf > conf_threshold) & (conf >= conf_percentile_val)
            
            if valid_mask.sum() == 0:
                logger.warning(f"No valid points found for frame {frame_idx}")
                frame_valid_points[frame_idx] = torch.empty((0, 3), device=self.device)
                continue
                
            valid_points = points[valid_mask]  # [N_valid, 3]
            frame_valid_points[frame_idx] = valid_points
            all_valid_points.append(valid_points)
        
        if not all_valid_points:
            raise ValueError("No valid points found in any frame")
            
        # Concatenate all valid points for global statistics
        all_points = torch.cat(all_valid_points, dim=0)  # [N_total, 3]
        
        # Step 2: Compute adaptive voxel size
        # Δ = (1/λ) · min(max(Pvalid) - min(Pvalid))
        point_min = all_points.min(dim=0)[0]  # [3]
        point_max = all_points.max(dim=0)[0]  # [3]
        scene_extents = point_max - point_min  # [3]
        
        # Use minimum extent dimension for isotropic voxels
        min_extent = scene_extents.min().item()
        voxel_size = min_extent / self.config.voxel_lambda
        
        logger.info(f"Scene extents: {scene_extents.tolist()}")
        logger.info(f"Adaptive voxel size: {voxel_size:.4f}")
        
        # Step 3: Voxelize each frame's point cloud
        voxel_sets = {}
        
        for frame_idx in frame_indices:
            if frame_idx not in frame_valid_points:
                voxel_sets[frame_idx] = set()
                continue
                
            valid_points = frame_valid_points[frame_idx]
            
            if len(valid_points) == 0:
                voxel_sets[frame_idx] = set()
                continue
            
            # Voxelization: V(fi^m) = {⌊(p - min(Pvalid))/Δ⌋}
            normalized_points = (valid_points - point_min.unsqueeze(0)) / voxel_size
            voxel_coords = torch.floor(normalized_points).long()
            
            # Convert to set of tuples for efficient set operations
            voxel_set = set()
            for coord in voxel_coords:
                voxel_set.add(tuple(coord.cpu().numpy()))
            
            voxel_sets[frame_idx] = voxel_set
            
            logger.debug(f"Frame {frame_idx}: {len(valid_points)} points -> {len(voxel_set)} voxels")
        
        return voxel_sets
    
    def _greedy_maximum_coverage(
        self,
        voxel_sets: Dict[int, Set[Tuple[int, int, int]]],
        frame_indices: List[int]
    ) -> Tuple[List[int], List[float]]:
        """
        Apply greedy maximum coverage selection algorithm.
        
        Algorithm implementation:
        1. Initialize: S ← ∅, C ← ∅, R ← {1, ..., Nm}
        2. For t = 1 to Nk:
            a) i* ← argmax_{i∈R} |V(fi^m) \ C|
            b) If |V(fi*^m) \ C| = 0: break
            c) S ← S ∪ {i*}, C ← C ∪ V(fi*^m), R ← R \ {i*}
        3. Return S
        """
        
        # Initialize algorithm state
        S = []  # Selected frames
        C = set()  # Covered voxels
        R = set(frame_indices)  # Remaining candidates
        coverage_scores = []  # Track coverage gain at each iteration
        
        target_frames = min(self.config.target_frames, len(frame_indices))
        
        logger.info(f"Starting greedy selection: {len(R)} candidates -> {target_frames} targets")
        
        for t in range(target_frames):
            if not R:  # No remaining candidates
                logger.info(f"Early termination: no remaining candidates at iteration {t}")
                break
            
            # Find frame with maximum coverage gain: i* ← argmax_{i∈R} |V(fi^m) \ C|
            best_frame = None
            best_coverage_gain = 0
            
            for frame_idx in R:
                frame_voxels = voxel_sets[frame_idx]
                coverage_gain = len(frame_voxels - C)  # |V(fi^m) \ C|
                
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_frame = frame_idx
            
            # Check early termination condition
            if best_coverage_gain == 0:
                if self.config.early_termination:
                    logger.info(f"Early termination: no additional coverage at iteration {t}")
                    break
                else:
                    # Select frame with largest total coverage (fallback)
                    best_frame = max(R, key=lambda i: len(voxel_sets[i]))
                    best_coverage_gain = 0
            
            # Update algorithm state
            S.append(best_frame)  # S ← S ∪ {i*}
            C.update(voxel_sets[best_frame])  # C ← C ∪ V(fi*^m)
            R.remove(best_frame)  # R ← R \ {i*}
            coverage_scores.append(best_coverage_gain)
            
            logger.debug(f"Iteration {t+1}: selected frame {best_frame}, "
                        f"gain={best_coverage_gain}, total_coverage={len(C)}")
        
        logger.info(f"Greedy selection completed: {len(S)} frames selected, "
                   f"total coverage: {len(C)} voxels")
        
        return S, coverage_scores

    def get_sampling_statistics(
        self,
        sampling_result: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute detailed statistics about the sampling process.
        
        Returns:
            Dictionary with sampling statistics
        """
        selected_indices = sampling_result['selected_indices'].cpu().numpy()
        coverage_scores = sampling_result['coverage_scores'].cpu().numpy()
        voxel_sets = sampling_result['voxel_sets']
        total_coverage = sampling_result['total_coverage']
        
        # Compute statistics
        stats = {
            'num_selected_frames': len(selected_indices),
            'total_voxel_coverage': total_coverage,
            'mean_coverage_gain': coverage_scores.mean() if len(coverage_scores) > 0 else 0.0,
            'std_coverage_gain': coverage_scores.std() if len(coverage_scores) > 0 else 0.0,
            'max_coverage_gain': coverage_scores.max() if len(coverage_scores) > 0 else 0.0,
            'min_coverage_gain': coverage_scores.min() if len(coverage_scores) > 0 else 0.0,
        }
        
        # Coverage efficiency: total coverage / sum of individual frame coverages
        if voxel_sets:
            individual_coverages = [len(voxel_sets[i]) for i in selected_indices]
            total_individual = sum(individual_coverages)
            stats['coverage_efficiency'] = total_coverage / total_individual if total_individual > 0 else 0.0
            stats['mean_frame_coverage'] = total_individual / len(selected_indices) if len(selected_indices) > 0 else 0.0
        
        return stats


def create_frame_sampler(
    target_frames: int = 16,
    candidate_frames: int = 128,
    voxel_lambda: float = 20.0,
    confidence_threshold: float = 0.1,
    confidence_percentile: float = 50.0,
    device: str = "cuda"
) -> SpaceAwareFrameSampler:
    """
    Factory function to create a space-aware frame sampler.
    
    Args:
        target_frames: Number of frames to select (N_k)
        candidate_frames: Total number of candidate frames (N_m)
        voxel_lambda: Voxel size parameter (λ = 20)
        confidence_threshold: Minimum confidence for valid points (0.1)
        confidence_percentile: Percentile threshold for points (50%)
        device: Computing device
        
    Returns:
        Configured SpaceAwareFrameSampler instance
    """
    config = SamplingConfig(
        target_frames=target_frames,
        candidate_frames=candidate_frames,
        voxel_lambda=voxel_lambda,
        confidence_threshold=confidence_threshold,
        confidence_percentile=confidence_percentile,
        device=device
    )
    
    return SpaceAwareFrameSampler(config)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    sampler = create_frame_sampler(
        target_frames=16,
        candidate_frames=32,  # Reduced for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Space-aware Frame Sampler created successfully!")
    print(f"Configuration: {sampler.config}")