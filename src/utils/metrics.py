"""
Evaluation metrics for VLN Spatial-MLLM Pipeline
Comprehensive metrics for spatial accuracy, temporal consistency, and inter-frame understanding
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

logger = logging.getLogger(__name__)


class VLNMetrics:
    """
    Comprehensive metrics suite for VLN evaluation
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def calculate_spatial_accuracy(self, predicted_heatmaps: torch.Tensor, 
                                 ground_truth_heatmaps: torch.Tensor,
                                 method: str = "mse") -> Dict[str, float]:
        """
        Calculate spatial accuracy between predicted and ground truth heatmaps
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, N_views, H, W]
            ground_truth_heatmaps: Ground truth heatmaps [B, N_views, H, W]
            method: Metric method ("mse", "mae", "cosine", "correlation")
            
        Returns:
            Dictionary of spatial accuracy metrics
        """
        pred = self._normalize_heatmaps(predicted_heatmaps)
        gt = self._normalize_heatmaps(ground_truth_heatmaps)
        
        metrics = {}
        
        if method == "mse" or method == "all":
            mse = F.mse_loss(pred, gt)
            metrics["spatial_mse"] = mse.item()
            metrics["spatial_accuracy_mse"] = 1.0 - torch.clamp(mse, 0, 1).item()
            
        if method == "mae" or method == "all":
            mae = F.l1_loss(pred, gt)
            metrics["spatial_mae"] = mae.item()
            metrics["spatial_accuracy_mae"] = 1.0 - torch.clamp(mae, 0, 1).item()
            
        if method == "cosine" or method == "all":
            # Flatten heatmaps for cosine similarity
            pred_flat = pred.view(pred.shape[0], -1)
            gt_flat = gt.view(gt.shape[0], -1)
            
            cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)
            metrics["spatial_cosine_similarity"] = cosine_sim.mean().item()
            
        if method == "correlation" or method == "all":
            # Calculate pixel-wise correlation
            correlations = []
            for b in range(pred.shape[0]):
                pred_np = pred[b].cpu().numpy().flatten()
                gt_np = gt[b].cpu().numpy().flatten()
                
                if np.std(pred_np) > 1e-6 and np.std(gt_np) > 1e-6:
                    corr, _ = pearsonr(pred_np, gt_np)
                    correlations.append(corr)
                else:
                    correlations.append(0.0)
                    
            metrics["spatial_correlation"] = np.mean(correlations)
            
        return metrics
    
    def calculate_temporal_consistency(self, heatmaps: torch.Tensor) -> Dict[str, float]:
        """
        Calculate temporal consistency of heatmaps across views
        
        Args:
            heatmaps: Heatmaps tensor [B, N_views, H, W]
            
        Returns:
            Dictionary of temporal consistency metrics
        """
        if heatmaps.shape[1] < 2:
            return {"temporal_consistency": 1.0, "temporal_smoothness": 1.0}
        
        batch_size, num_views = heatmaps.shape[:2]
        
        # 1. Adjacent view consistency
        adjacent_consistency = []
        for b in range(batch_size):
            for v in range(num_views - 1):
                corr = self._normalized_cross_correlation(
                    heatmaps[b, v], heatmaps[b, v + 1]
                )
                adjacent_consistency.append(corr)
        
        # 2. Overall temporal smoothness
        temporal_gradients = []
        for b in range(batch_size):
            batch_heatmaps = heatmaps[b]
            for v in range(num_views - 1):
                grad = torch.mean(torch.abs(batch_heatmaps[v] - batch_heatmaps[v + 1]))
                temporal_gradients.append(grad.item())
        
        # 3. Global consistency (all pairs)
        global_consistency = []
        for b in range(batch_size):
            batch_consistencies = []
            for i in range(num_views):
                for j in range(i + 1, num_views):
                    corr = self._normalized_cross_correlation(
                        heatmaps[b, i], heatmaps[b, j]
                    )
                    batch_consistencies.append(corr)
            global_consistency.extend(batch_consistencies)
        
        return {
            "temporal_consistency_adjacent": np.mean(adjacent_consistency),
            "temporal_consistency_global": np.mean(global_consistency),
            "temporal_smoothness": 1.0 - np.mean(temporal_gradients),  # Invert for consistency
            "temporal_variation": np.std(temporal_gradients)
        }
    
    def calculate_inter_frame_accuracy(self, results: Dict[str, torch.Tensor],
                                     ground_truth: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Calculate inter-frame spatial understanding accuracy
        
        Args:
            results: Pipeline results containing heatmaps, geometry_info, etc.
            ground_truth: Optional ground truth data for comparison
            
        Returns:
            Dictionary of inter-frame accuracy metrics
        """
        heatmaps = results['first_person_heatmaps']
        selected_indices = results['selected_indices']
        geometry_info = results.get('geometry_info', {})
        
        metrics = {}
        
        # 1. Cross-frame spatial diversity
        # Good inter-frame understanding should show different patterns for different viewpoints
        spatial_diversity = []
        for b in range(heatmaps.shape[0]):
            batch_heatmaps = heatmaps[b]
            pairwise_diffs = []
            
            for i in range(batch_heatmaps.shape[0]):
                for j in range(i + 1, batch_heatmaps.shape[0]):
                    diff = torch.mean(torch.abs(batch_heatmaps[i] - batch_heatmaps[j]))
                    pairwise_diffs.append(diff.item())
            
            if pairwise_diffs:
                spatial_diversity.append(np.mean(pairwise_diffs))
        
        metrics["inter_frame_spatial_diversity"] = np.mean(spatial_diversity) if spatial_diversity else 0.0
        
        # 2. Geometry-heatmap alignment
        # Heatmaps should correlate with geometric changes
        if 'camera_poses' in geometry_info:
            geometry_alignment = self._calculate_geometry_heatmap_alignment(
                heatmaps, geometry_info['camera_poses']
            )
            metrics.update(geometry_alignment)
        
        # 3. Keyframe selection quality
        if selected_indices:
            selection_metrics = self._calculate_keyframe_selection_quality(selected_indices)
            metrics.update(selection_metrics)
        
        # 4. Spatial coverage consistency
        coverage_metrics = self._calculate_spatial_coverage_metrics(heatmaps)
        metrics.update(coverage_metrics)
        
        return metrics
    
    def calculate_heatmap_quality(self, heatmaps: torch.Tensor) -> Dict[str, float]:
        """
        Calculate overall heatmap quality metrics
        
        Args:
            heatmaps: Heatmaps tensor [B, N_views, H, W]
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        batch_size, num_views = heatmaps.shape[:2]
        
        # 1. Peak clarity (ratio of max to mean)
        peak_clarity_scores = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                max_val = torch.max(heatmap)
                mean_val = torch.mean(heatmap)
                clarity = (max_val / (mean_val + 1e-8)).clamp(0, 10) / 10
                peak_clarity_scores.append(clarity.item())
        
        metrics["peak_clarity"] = np.mean(peak_clarity_scores)
        
        # 2. Dynamic range
        dynamic_ranges = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                dynamic_range = (torch.max(heatmap) - torch.min(heatmap)).clamp(0, 1)
                dynamic_ranges.append(dynamic_range.item())
        
        metrics["dynamic_range"] = np.mean(dynamic_ranges)
        
        # 3. Spatial coherence (smoothness)
        coherence_scores = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                
                # Calculate spatial gradients
                grad_x = torch.diff(heatmap, dim=1)
                grad_y = torch.diff(heatmap, dim=0)
                
                # Smoothness is inverse of gradient magnitude
                gradient_magnitude = torch.mean(torch.abs(grad_x) + torch.abs(grad_y[:-1, :]))
                smoothness = 1.0 - gradient_magnitude.clamp(0, 1)
                coherence_scores.append(smoothness.item())
        
        metrics["spatial_coherence"] = np.mean(coherence_scores)
        
        # 4. Focus quality (concentration of attention)
        focus_scores = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                
                # Calculate entropy (lower entropy = more focused)
                heatmap_norm = heatmap / (torch.sum(heatmap) + 1e-8)
                entropy = -torch.sum(heatmap_norm * torch.log(heatmap_norm + 1e-8))
                
                # Normalize entropy (lower is better for focus)
                max_entropy = np.log(heatmap.numel())
                focus = 1.0 - (entropy / max_entropy).clamp(0, 1)
                focus_scores.append(focus.item())
        
        metrics["focus_quality"] = np.mean(focus_scores)
        
        # 5. Overall quality score (weighted combination)
        overall_quality = (
            0.3 * metrics["peak_clarity"] +
            0.2 * metrics["dynamic_range"] +
            0.25 * metrics["spatial_coherence"] +
            0.25 * metrics["focus_quality"]
        )
        metrics["overall_quality"] = overall_quality
        
        return metrics
    
    def calculate_success_rate(self, predicted_actions: torch.Tensor,
                             ground_truth_actions: torch.Tensor,
                             threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate task success rate metrics
        
        Args:
            predicted_actions: Predicted action probabilities or coordinates
            ground_truth_actions: Ground truth actions
            threshold: Success threshold for coordinate-based tasks
            
        Returns:
            Dictionary of success metrics
        """
        if predicted_actions.shape != ground_truth_actions.shape:
            logger.warning(f"Shape mismatch: pred {predicted_actions.shape} vs gt {ground_truth_actions.shape}")
            return {"success_rate": 0.0}
        
        metrics = {}
        
        # For coordinate-based tasks (e.g., navigation waypoints)
        if len(predicted_actions.shape) >= 2 and predicted_actions.shape[-1] >= 2:
            # Calculate Euclidean distances
            distances = torch.norm(predicted_actions - ground_truth_actions, dim=-1)
            successes = (distances <= threshold).float()
            
            metrics["success_rate"] = torch.mean(successes).item()
            metrics["mean_error_distance"] = torch.mean(distances).item()
            metrics["median_error_distance"] = torch.median(distances).item()
            
        # For classification-based tasks
        else:
            if predicted_actions.dtype != ground_truth_actions.dtype:
                # Convert to same type for comparison
                if predicted_actions.dtype == torch.float:
                    predicted_actions = (predicted_actions > 0.5).long()
                
            accuracy = (predicted_actions == ground_truth_actions).float().mean()
            metrics["success_rate"] = accuracy.item()
            
            # Additional classification metrics if multi-class
            if len(torch.unique(ground_truth_actions)) > 2:
                pred_np = predicted_actions.cpu().numpy().flatten()
                gt_np = ground_truth_actions.cpu().numpy().flatten()
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt_np, pred_np, average='weighted', zero_division=0
                )
                
                metrics["precision"] = precision
                metrics["recall"] = recall  
                metrics["f1_score"] = f1
        
        return metrics
    
    def _normalize_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Normalize heatmaps to [0, 1] range"""
        batch_size, num_views = heatmaps.shape[:2]
        normalized = torch.zeros_like(heatmaps)
        
        for b in range(batch_size):
            for v in range(num_views):
                hmap = heatmaps[b, v]
                min_val = torch.min(hmap)
                max_val = torch.max(hmap)
                
                if max_val > min_val:
                    normalized[b, v] = (hmap - min_val) / (max_val - min_val)
                else:
                    normalized[b, v] = hmap
        
        return normalized
    
    def _normalized_cross_correlation(self, heatmap1: torch.Tensor, 
                                    heatmap2: torch.Tensor) -> float:
        """Calculate normalized cross-correlation between two heatmaps"""
        h1_flat = heatmap1.flatten()
        h2_flat = heatmap2.flatten()
        
        # Remove mean (normalize)
        h1_norm = h1_flat - torch.mean(h1_flat)
        h2_norm = h2_flat - torch.mean(h2_flat)
        
        # Calculate correlation
        correlation = torch.dot(h1_norm, h2_norm) / (
            torch.norm(h1_norm) * torch.norm(h2_norm) + 1e-8
        )
        
        return abs(correlation.item()) if not torch.isnan(correlation) else 0.0
    
    def _calculate_geometry_heatmap_alignment(self, heatmaps: torch.Tensor,
                                           camera_poses: torch.Tensor) -> Dict[str, float]:
        """Calculate alignment between heatmaps and geometric changes"""
        batch_size = heatmaps.shape[0]
        alignments = []
        
        for b in range(batch_size):
            batch_heatmaps = heatmaps[b]
            batch_poses = camera_poses[b] if camera_poses.dim() > 2 else camera_poses
            
            # Calculate pose differences
            pose_diffs = []
            heatmap_diffs = []
            
            for i in range(batch_heatmaps.shape[0] - 1):
                # Pose difference (Frobenius norm)
                pose_diff = torch.norm(batch_poses[i] - batch_poses[i + 1])
                pose_diffs.append(pose_diff.item())
                
                # Heatmap difference  
                heatmap_diff = torch.mean(torch.abs(batch_heatmaps[i] - batch_heatmaps[i + 1]))
                heatmap_diffs.append(heatmap_diff.item())
            
            if len(pose_diffs) > 1 and np.std(pose_diffs) > 1e-6 and np.std(heatmap_diffs) > 1e-6:
                corr, _ = pearsonr(pose_diffs, heatmap_diffs)
                alignments.append(abs(corr) if not np.isnan(corr) else 0.0)
            else:
                alignments.append(0.0)
        
        return {
            "geometry_heatmap_alignment": np.mean(alignments) if alignments else 0.0
        }
    
    def _calculate_keyframe_selection_quality(self, selected_indices: List[List[int]]) -> Dict[str, float]:
        """Calculate quality of keyframe selection"""
        all_metrics = []
        
        for indices in selected_indices:
            if len(indices) < 2:
                continue
                
            # 1. Temporal distribution (uniformity)
            spacings = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            spacing_uniformity = 1.0 / (1.0 + np.var(spacings)) if spacings else 0.0
            
            # 2. Coverage (span of selected frames)
            total_span = max(indices) - min(indices) if indices else 0
            coverage_ratio = total_span / max(max(indices), 1) if indices else 0.0
            
            # 3. Efficiency (avoid clustering)
            min_spacing = min(spacings) if spacings else 0
            efficiency = min_spacing / (sum(spacings) / len(spacings)) if spacings else 0.0
            
            all_metrics.append({
                "spacing_uniformity": spacing_uniformity,
                "coverage_ratio": coverage_ratio,  
                "selection_efficiency": efficiency
            })
        
        # Average across batches
        if all_metrics:
            return {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0].keys()
            }
        else:
            return {
                "spacing_uniformity": 0.0,
                "coverage_ratio": 0.0,
                "selection_efficiency": 0.0
            }
    
    def _calculate_spatial_coverage_metrics(self, heatmaps: torch.Tensor) -> Dict[str, float]:
        """Calculate spatial coverage quality of heatmaps"""
        batch_size, num_views, H, W = heatmaps.shape
        
        # 1. Spatial entropy (diversity of attention)
        entropies = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                heatmap_norm = heatmap / (torch.sum(heatmap) + 1e-8)
                entropy = -torch.sum(heatmap_norm * torch.log(heatmap_norm + 1e-8))
                entropies.append(entropy.item())
        
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(H * W)
        normalized_entropy = mean_entropy / max_entropy
        
        # 2. Coverage area (percentage of image with significant attention)
        coverage_ratios = []
        for b in range(batch_size):
            for v in range(num_views):
                heatmap = heatmaps[b, v]
                threshold = 0.1 * torch.max(heatmap)
                coverage = (heatmap > threshold).float().mean()
                coverage_ratios.append(coverage.item())
        
        return {
            "spatial_entropy": normalized_entropy,
            "spatial_coverage_ratio": np.mean(coverage_ratios)
        }


def calculate_comprehensive_metrics(results: Dict[str, torch.Tensor],
                                  ground_truth: Optional[Dict[str, torch.Tensor]] = None,
                                  device: str = "cuda") -> Dict[str, float]:
    """
    Convenience function to calculate all VLN metrics at once
    
    Args:
        results: Pipeline results dictionary
        ground_truth: Optional ground truth data
        device: Device for computations
        
    Returns:
        Dictionary containing all computed metrics
    """
    metrics_calculator = VLNMetrics(device=device)
    all_metrics = {}
    
    # Spatial accuracy (if ground truth available)
    if ground_truth and 'heatmaps' in ground_truth:
        spatial_metrics = metrics_calculator.calculate_spatial_accuracy(
            results['first_person_heatmaps'], 
            ground_truth['heatmaps'],
            method="all"
        )
        all_metrics.update(spatial_metrics)
    
    # Temporal consistency
    temporal_metrics = metrics_calculator.calculate_temporal_consistency(
        results['first_person_heatmaps']
    )
    all_metrics.update(temporal_metrics)
    
    # Inter-frame accuracy
    inter_frame_metrics = metrics_calculator.calculate_inter_frame_accuracy(
        results, ground_truth
    )
    all_metrics.update(inter_frame_metrics)
    
    # Heatmap quality
    quality_metrics = metrics_calculator.calculate_heatmap_quality(
        results['first_person_heatmaps']
    )
    all_metrics.update(quality_metrics)
    
    # Success rate (if ground truth actions available)
    if ground_truth and 'actions' in ground_truth and 'predicted_actions' in results:
        success_metrics = metrics_calculator.calculate_success_rate(
            results['predicted_actions'],
            ground_truth['actions']
        )
        all_metrics.update(success_metrics)
    
    return all_metrics