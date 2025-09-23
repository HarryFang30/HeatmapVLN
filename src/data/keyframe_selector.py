"""
Keyframe Selector with Index-based Retrieval  
============================================

This module provides the high-level interface for space-aware keyframe selection
in the VLN pipeline. It orchestrates the sampling process and implements efficient
index-based retrieval for the dual-encoder architecture.

Architecture Flow:
1. VGGT Pre-processing: ALL N_m frames → VGGT → Feature vectors + Geometry
2. Space-aware Sampling: Geometry → Sampling algorithm → Select N_k indices  
3. Index-based Retrieval: 
   - 3D Path: Selected indices → Pre-computed VGGT features
   - 2D Path: Selected indices → Original frames → DINOv3

Key Features:
- Efficient index-based feature retrieval
- Integration with space-aware sampling algorithms
- Support for multiple sampling strategies
- Caching and optimization for repeated selections
- Comprehensive logging and analysis
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from .frame_sampler import SpaceAwareFrameSampler, SamplingConfig, create_frame_sampler
from .spatial_analysis import SpatialNoveltyDetector, create_spatial_analyzer

logger = logging.getLogger(__name__)


@dataclass 
class KeyframeSelectionConfig:
    """Configuration for keyframe selection pipeline."""
    # Frame selection parameters
    target_keyframes: int = 16  # N_k - target number of keyframes
    total_frames: int = 128  # N_m - total input frames
    
    # Sampling strategy
    sampling_method: str = "greedy_coverage"  # Options: greedy_coverage, novelty_weighted, hybrid
    use_spatial_analysis: bool = True  # Enable spatial novelty analysis
    
    # Hybrid sampling parameters (when sampling_method="hybrid")
    coverage_weight: float = 0.7  # Weight for coverage-based selection
    novelty_weight: float = 0.3  # Weight for novelty-based selection
    
    # Performance optimization
    enable_caching: bool = True  # Cache intermediate results
    cache_dir: Optional[str] = None  # Directory for persistent caching
    
    # Device configuration
    device: str = "cuda"
    
    # Logging and debugging
    verbose: bool = True  # Enable detailed logging
    save_debug_info: bool = False  # Save debugging information


class KeyframeSelector(nn.Module):
    """
    High-level keyframe selector for VLN pipeline.
    
    This class orchestrates the space-aware frame sampling process and provides
    efficient index-based retrieval for the dual-encoder architecture.
    
    Usage:
        selector = KeyframeSelector(config)
        result = selector(vggt_predictions, original_frames)
        selected_indices = result['keyframe_indices']
    """
    
    def __init__(self, config: KeyframeSelectionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize sampling components
        sampling_config = SamplingConfig(
            target_frames=config.target_keyframes,
            candidate_frames=config.total_frames,
            device=config.device
        )
        self.frame_sampler = SpaceAwareFrameSampler(sampling_config)
        
        # Initialize spatial analysis if enabled
        if config.use_spatial_analysis:
            self.spatial_analyzer = create_spatial_analyzer()
        else:
            self.spatial_analyzer = None
            
        # Initialize caching system
        self._cache = {}
        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
        # Performance tracking
        self._selection_stats = {
            'total_selections': 0,
            'cache_hits': 0,
            'average_selection_time': 0.0
        }
        
    def forward(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        original_frames: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
        selection_hint: Optional[str] = None
    ) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        Select keyframes using space-aware sampling.
        
        Args:
            vggt_predictions: VGGT model outputs containing geometry information
            original_frames: Original video frames [B, N_m, C, H, W] (optional)
            frame_indices: Specific frame indices to consider (optional)
            selection_hint: Hint for selection strategy (optional)
            
        Returns:
            Dictionary containing:
                - 'keyframe_indices': Selected keyframe indices [N_k]
                - 'vggt_features': Index-selected VGGT features for 3D path
                - 'original_frames': Index-selected frames for 2D path (if provided)
                - 'geometry_data': Extracted geometry information
                - 'selection_metadata': Selection statistics and analysis
                - 'sampling_result': Detailed sampling algorithm results
        """
        
        start_time = time.time()
        
        # Input validation and preprocessing
        batch_size = vggt_predictions['depth'].shape[0]
        # Multi-batch processing is now supported
            
        # Generate cache key for optimization
        cache_key = self._generate_cache_key(vggt_predictions, frame_indices, selection_hint)
        
        # Check cache first
        if self.config.enable_caching and cache_key in self._cache:
            logger.info("Using cached keyframe selection")
            self._selection_stats['cache_hits'] += 1
            return self._cache[cache_key].copy()
            
        # Prepare frame indices
        if frame_indices is None:
            total_frames = vggt_predictions['depth'].shape[1]
            frame_indices = list(range(total_frames))
        
        logger.info(f"Selecting {self.config.target_keyframes} keyframes from "
                   f"{len(frame_indices)} candidate frames")
        
        # Step 1: Apply sampling algorithm based on method
        if self.config.sampling_method == "greedy_coverage":
            selection_result = self._greedy_coverage_selection(vggt_predictions, frame_indices)
        elif self.config.sampling_method == "novelty_weighted":
            selection_result = self._novelty_weighted_selection(vggt_predictions, frame_indices)  
        elif self.config.sampling_method == "hybrid":
            selection_result = self._hybrid_selection(vggt_predictions, frame_indices)
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
            
        selected_indices = selection_result['selected_indices']
        
        # Step 2: Index-based feature retrieval
        retrieval_result = self._index_based_retrieval(
            vggt_predictions, original_frames, selected_indices
        )
        
        # Step 3: Combine results and create output
        output = {
            'keyframe_indices': selected_indices,
            'vggt_features': retrieval_result['vggt_features'],
            'geometry_data': selection_result['geometry_data'],
            'selection_metadata': self._create_selection_metadata(
                selection_result, len(frame_indices), time.time() - start_time
            ),
            'sampling_result': selection_result
        }
        
        # Add original frames if provided
        if retrieval_result['original_frames'] is not None:
            output['original_frames'] = retrieval_result['original_frames']
            
        # Update performance tracking
        self._update_performance_stats(time.time() - start_time)
        
        # Cache result if enabled
        if self.config.enable_caching:
            self._cache[cache_key] = output.copy()
            
        if self.config.verbose:
            logger.info(f"Keyframe selection completed in {time.time() - start_time:.3f}s")
            
        return output
    
    def _greedy_coverage_selection(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        frame_indices: List[int]
    ) -> Dict:
        """Apply greedy maximum coverage selection."""
        return self.frame_sampler(vggt_predictions, frame_indices)
    
    def _novelty_weighted_selection(
        self,
        vggt_predictions: Dict[str, torch.Tensor], 
        frame_indices: List[int]
    ) -> Dict:
        """Apply novelty-weighted frame selection."""
        if self.spatial_analyzer is None:
            logger.warning("Spatial analyzer not available, falling back to greedy coverage")
            return self._greedy_coverage_selection(vggt_predictions, frame_indices)
            
        # Extract geometry for spatial analysis
        sampling_result = self.frame_sampler(vggt_predictions, frame_indices)
        geometry_data = sampling_result['geometry_data']
        
        # Check if this is multi-batch processing - fallback to greedy method
        if geometry_data and geometry_data.get('is_multi_batch', False):
            logger.info("Multi-batch detected, falling back to greedy coverage for consistency")
            return self._greedy_coverage_selection(vggt_predictions, frame_indices)
        
        # Iterative selection based on novelty
        selected_indices = []
        remaining_indices = frame_indices.copy()
        
        for _ in range(self.config.target_keyframes):
            if not remaining_indices:
                break
                
            # Analyze spatial novelty
            novelty_analysis = self.spatial_analyzer(geometry_data, selected_indices)
            novelty_scores = novelty_analysis['novelty_scores']
            
            # Select frame with highest novelty from remaining candidates
            remaining_mask = torch.zeros_like(novelty_scores, dtype=torch.bool)
            for idx in remaining_indices:
                if idx < len(remaining_mask):
                    remaining_mask[idx] = True
                    
            masked_scores = novelty_scores * remaining_mask.float()
            best_idx = torch.argmax(masked_scores).item()
            
            if best_idx in remaining_indices:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # Fallback: select first remaining
                if remaining_indices:
                    selected_indices.append(remaining_indices.pop(0))
                    
        # Update sampling result
        sampling_result['selected_indices'] = torch.tensor(selected_indices, device=self.device)
        sampling_result['novelty_analysis'] = novelty_analysis
        
        return sampling_result
    
    def _hybrid_selection(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        frame_indices: List[int]
    ) -> Dict:
        """Apply hybrid coverage + novelty selection."""
        if self.spatial_analyzer is None:
            logger.warning("Spatial analyzer not available, using pure coverage selection")
            return self._greedy_coverage_selection(vggt_predictions, frame_indices)
        
        # Get coverage-based selection
        coverage_result = self._greedy_coverage_selection(vggt_predictions, frame_indices)
        
        # Check if coverage result indicates multi-batch - use it directly
        coverage_indices_tensor = coverage_result['selected_indices']
        if len(coverage_indices_tensor.shape) > 1:  # Multi-batch case: [batch_size, n_keyframes]
            logger.info("Multi-batch detected in hybrid selection, using coverage-only result")
            return coverage_result
        
        # Get novelty-based selection for single-batch case
        novelty_result = self._novelty_weighted_selection(vggt_predictions, frame_indices)
        
        # Combine selections with weighting
        coverage_indices = set(coverage_result['selected_indices'].cpu().numpy())
        novelty_indices = set(novelty_result['selected_indices'].cpu().numpy())
        
        # Weighted selection: prioritize overlap, then add diverse choices
        overlap_indices = coverage_indices.intersection(novelty_indices)
        coverage_only = coverage_indices - overlap_indices  
        novelty_only = novelty_indices - overlap_indices
        
        # Build final selection
        final_indices = list(overlap_indices)  # Start with consensus choices
        
        # Add weighted choices from remaining
        remaining_slots = self.config.target_keyframes - len(final_indices)
        coverage_slots = int(remaining_slots * self.config.coverage_weight)
        novelty_slots = remaining_slots - coverage_slots
        
        final_indices.extend(list(coverage_only)[:coverage_slots])
        final_indices.extend(list(novelty_only)[:novelty_slots])
        
        # Fill any remaining slots
        if len(final_indices) < self.config.target_keyframes:
            all_candidates = set(frame_indices)
            unused = all_candidates - set(final_indices)
            final_indices.extend(list(unused)[:self.config.target_keyframes - len(final_indices)])
        
        # Update result
        hybrid_result = coverage_result.copy()
        hybrid_result['selected_indices'] = torch.tensor(final_indices[:self.config.target_keyframes], 
                                                        device=self.device)
        hybrid_result['novelty_analysis'] = novelty_result.get('novelty_analysis')
        hybrid_result['hybrid_metadata'] = {
            'coverage_weight': self.config.coverage_weight,
            'novelty_weight': self.config.novelty_weight,
            'overlap_frames': len(overlap_indices),
            'coverage_only_frames': len(coverage_only),
            'novelty_only_frames': len(novelty_only)
        }
        
        return hybrid_result
    
    def _index_based_retrieval(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        original_frames: Optional[torch.Tensor],
        selected_indices: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Perform index-based retrieval for dual-encoder architecture.
        
        Efficiently retrieves:
        - Pre-computed VGGT features for 3D processing path
        - Original frames for 2D processing path (DINOv3)
        """
        
        device = selected_indices.device
        
        # Handle both 1D and 2D selected_indices
        if len(selected_indices.shape) == 1:
            # Legacy 1D case: [n_keyframes]
            indices = selected_indices.cpu().numpy()
            batch_size = 1
        else:
            # New 2D case: [batch_size, n_keyframes]
            indices = selected_indices.cpu().numpy()
            batch_size = indices.shape[0]
        
        # Extract VGGT features using index selection
        vggt_features = {}
        for key, tensor in vggt_predictions.items():
            if len(tensor.shape) >= 2:  # Has sequence dimension
                if len(selected_indices.shape) == 1:
                    # Legacy 1D indexing
                    selected_tensor = tensor[:, indices]  # [B, N_k, ...]
                else:
                    # 2D indexing: handle batch-wise selection
                    selected_tensors = []
                    for b in range(batch_size):
                        batch_indices = indices[b]  # [n_keyframes] for batch b
                        selected_tensors.append(tensor[b:b+1, batch_indices])  # [1, n_keyframes, ...]
                    selected_tensor = torch.cat(selected_tensors, dim=0)  # [batch_size, n_keyframes, ...]
                vggt_features[key] = selected_tensor
        
        # Extract original frames if provided
        selected_frames = None
        if original_frames is not None:
            if len(selected_indices.shape) == 1:
                # Legacy 1D indexing
                if len(original_frames.shape) == 5:  # [B, S, C, H, W]
                    selected_frames = original_frames[:, indices]  # [B, N_k, C, H, W]
                elif len(original_frames.shape) == 4:  # [S, C, H, W]
                    selected_frames = original_frames[indices]  # [N_k, C, H, W]
                else:
                    logger.warning(f"Unexpected original_frames shape: {original_frames.shape}")
            else:
                # 2D indexing: handle batch-wise selection
                if len(original_frames.shape) == 5:  # [B, S, C, H, W]
                    selected_frame_tensors = []
                    for b in range(batch_size):
                        batch_indices = indices[b]  # [n_keyframes] for batch b
                        selected_frame_tensors.append(original_frames[b:b+1, batch_indices])  # [1, n_keyframes, C, H, W]
                    selected_frames = torch.cat(selected_frame_tensors, dim=0)  # [batch_size, n_keyframes, C, H, W]
                elif len(original_frames.shape) == 4:  # [S, C, H, W] - shouldn't happen with batch processing
                    logger.warning("4D original_frames with 2D indices - unexpected configuration")
                    selected_frames = original_frames[indices[0]]  # Use first batch indices as fallback
                else:
                    logger.warning(f"Unexpected original_frames shape: {original_frames.shape}")
                
        return {
            'vggt_features': vggt_features,
            'original_frames': selected_frames
        }
    
    def _create_selection_metadata(
        self,
        selection_result: Dict,
        total_candidates: int,
        selection_time: float
    ) -> Dict:
        """Create comprehensive metadata about the selection process."""
        
        selected_indices = selection_result['selected_indices']
        num_selected = len(selected_indices)
        
        metadata = {
            'selection_method': self.config.sampling_method,
            'num_selected_frames': num_selected,
            'num_candidate_frames': total_candidates,
            'selection_ratio': num_selected / total_candidates if total_candidates > 0 else 0.0,
            'selection_time_seconds': selection_time,
            'device_used': str(self.device),
            'config': self.config.__dict__.copy()
        }
        
        # Add sampling-specific metadata
        if 'total_coverage' in selection_result:
            metadata['total_voxel_coverage'] = selection_result['total_coverage']
            
        if 'coverage_scores' in selection_result:
            coverage_scores = selection_result['coverage_scores'].cpu().numpy()
            metadata['coverage_statistics'] = {
                'mean_coverage_gain': float(coverage_scores.mean()),
                'std_coverage_gain': float(coverage_scores.std()),
                'max_coverage_gain': float(coverage_scores.max()),
                'min_coverage_gain': float(coverage_scores.min())
            }
            
        if 'hybrid_metadata' in selection_result:
            metadata.update(selection_result['hybrid_metadata'])
            
        return metadata
    
    def _generate_cache_key(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        frame_indices: Optional[List[int]],
        selection_hint: Optional[str]
    ) -> str:
        """Generate cache key for selection results."""
        
        # Create hash from prediction tensor shapes and key parameters
        shape_info = []
        for key, tensor in vggt_predictions.items():
            shape_info.append(f"{key}:{tensor.shape}")
            
        indices_str = str(frame_indices) if frame_indices else "None"
        hint_str = selection_hint or "None"
        
        cache_components = [
            self.config.sampling_method,
            str(self.config.target_keyframes),
            "|".join(shape_info),
            indices_str,
            hint_str
        ]
        
        return "|".join(cache_components)
    
    def _update_performance_stats(self, selection_time: float):
        """Update performance tracking statistics."""
        
        self._selection_stats['total_selections'] += 1
        
        # Update running average
        current_avg = self._selection_stats['average_selection_time']
        total_selections = self._selection_stats['total_selections']
        
        new_avg = ((current_avg * (total_selections - 1)) + selection_time) / total_selections
        self._selection_stats['average_selection_time'] = new_avg
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the selector."""
        stats = self._selection_stats.copy()
        
        if stats['total_selections'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_selections']
        else:
            stats['cache_hit_rate'] = 0.0
            
        return stats
    
    def clear_cache(self):
        """Clear the selection cache."""
        self._cache.clear()
        logger.info("Selection cache cleared")


def create_keyframe_selector(
    target_keyframes: int = 16,
    total_frames: int = 128,
    sampling_method: str = "greedy_coverage",
    use_spatial_analysis: bool = True,
    device: str = "cuda",
    enable_caching: bool = True,
    verbose: bool = True
) -> KeyframeSelector:
    """
    Factory function to create a keyframe selector.
    
    Args:
        target_keyframes: Number of keyframes to select (N_k)
        total_frames: Total number of input frames (N_m)
        sampling_method: Selection strategy ("greedy_coverage", "novelty_weighted", "hybrid")
        use_spatial_analysis: Enable spatial novelty analysis
        device: Computing device
        enable_caching: Enable result caching
        verbose: Enable detailed logging
        
    Returns:
        Configured KeyframeSelector instance
    """
    config = KeyframeSelectionConfig(
        target_keyframes=target_keyframes,
        total_frames=total_frames,
        sampling_method=sampling_method,
        use_spatial_analysis=use_spatial_analysis,
        device=device,
        enable_caching=enable_caching,
        verbose=verbose
    )
    
    return KeyframeSelector(config)


# Example usage and testing
if __name__ == "__main__":
    selector = create_keyframe_selector(
        target_keyframes=16,
        total_frames=64,  # Reduced for testing
        sampling_method="hybrid",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Keyframe Selector created successfully!")
    print(f"Configuration: {selector.config}")
    print(f"Performance stats: {selector.get_performance_stats()}")