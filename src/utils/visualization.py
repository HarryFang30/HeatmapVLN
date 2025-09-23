"""
Visualization utilities for VLN Project
Heatmap visualization, 3D spatial plotting, and result analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import torch
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class HeatmapVisualizer:
    """
    Comprehensive heatmap visualization for VLN spatial understanding
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom colormap for VLN heatmaps
        self.vln_colormap = LinearSegmentedColormap.from_list(
            'vln_heatmap',
            ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffffff'],
            N=256
        )
    
    def visualize_inter_frame_heatmaps(self, heatmaps: torch.Tensor, 
                                     selected_indices: List[int],
                                     video_name: str = "video",
                                     save: bool = True) -> plt.Figure:
        """
        Visualize first-person inter-frame heatmaps
        
        Args:
            heatmaps: Tensor of shape [num_views, H, W] or [B, num_views, H, W]
            selected_indices: List of selected keyframe indices
            video_name: Name for saving files
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if heatmaps.dim() == 4:
            heatmaps = heatmaps[0]  # Take first batch if batched
        
        heatmaps = heatmaps.detach().cpu().numpy()
        num_views = heatmaps.shape[0]
        
        # Create figure with subplots
        cols = min(num_views, 4)
        rows = (num_views + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        for i in range(num_views):
            ax = axes[i] if num_views > 1 else axes
            heatmap = heatmaps[i]
            
            # Normalize heatmap
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Display heatmap
            im = ax.imshow(heatmap_norm, cmap=self.vln_colormap, interpolation='bilinear')
            
            # Add title with frame index
            frame_idx = selected_indices[i] if i < len(selected_indices) else i
            ax.set_title(f'Inter-Frame Heatmap\nKeyframe {frame_idx}', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Spatial Attention', rotation=270, labelpad=15)
            
            # Add grid for better spatial reference
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
        # Hide unused subplots
        for i in range(num_views, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'First-Person Inter-Frame Spatial Heatmaps: {video_name}', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{video_name}_inter_frame_heatmaps.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Inter-frame heatmaps saved to {save_path}")
        
        return fig
    
    def visualize_spatial_sampling(self, geometry_info: Dict[str, torch.Tensor],
                                 selected_indices: List[int],
                                 total_frames: int,
                                 video_name: str = "video") -> plt.Figure:
        """
        Visualize space-aware frame sampling decisions
        
        Args:
            geometry_info: Dictionary containing camera poses, depths, etc.
            selected_indices: Selected keyframe indices
            total_frames: Total number of frames
            video_name: Video name for title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Frame selection timeline
        ax1 = axes[0, 0]
        frame_indices = list(range(total_frames))
        selection_mask = [i in selected_indices for i in frame_indices]
        
        colors = ['red' if selected else 'lightgray' for selected in selection_mask]
        ax1.scatter(frame_indices, [1] * total_frames, c=colors, s=50, alpha=0.7)
        
        for idx in selected_indices:
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
            ax1.text(idx, 1.1, str(idx), ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Selected')
        ax1.set_title('Space-Aware Frame Sampling')
        ax1.set_ylim(0.5, 1.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Camera pose changes (if available)
        ax2 = axes[0, 1]
        if 'camera_poses' in geometry_info:
            poses = geometry_info['camera_poses'].cpu().numpy()
            # Calculate pose differences
            pose_diffs = np.linalg.norm(np.diff(poses.reshape(poses.shape[0], -1), axis=0), axis=1)
            
            ax2.plot(range(1, len(pose_diffs) + 1), pose_diffs, 'b-', alpha=0.7)
            
            # Highlight selected frames
            for idx in selected_indices:
                if idx > 0 and idx <= len(pose_diffs):
                    ax2.plot(idx, pose_diffs[idx-1], 'ro', markersize=8)
            
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Pose Change')
            ax2.set_title('Camera Pose Changes')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Camera poses not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Camera Pose Changes')
        
        # 3. Depth complexity (if available)
        ax3 = axes[1, 0]
        if 'depth_maps' in geometry_info:
            depths = geometry_info['depth_maps'].cpu().numpy()
            # Calculate depth variance as complexity measure
            depth_complexity = np.var(depths.reshape(depths.shape[0], -1), axis=1)
            
            ax3.plot(range(len(depth_complexity)), depth_complexity, 'g-', alpha=0.7)
            
            # Highlight selected frames
            for idx in selected_indices:
                if idx < len(depth_complexity):
                    ax3.plot(idx, depth_complexity[idx], 'ro', markersize=8)
            
            ax3.set_xlabel('Frame Index')
            ax3.set_ylabel('Depth Complexity')
            ax3.set_title('Scene Depth Complexity')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Depth maps not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Scene Depth Complexity')
        
        # 4. Spatial coverage (visualization of sampling efficiency)
        ax4 = axes[1, 1]
        
        # Create a heatmap showing sampling efficiency
        coverage_matrix = np.zeros((8, 8))  # 8x8 grid for visualization
        
        # Simulate spatial coverage (in real implementation, use actual voxel data)
        for i, idx in enumerate(selected_indices):
            # Map frame index to spatial grid position
            row = (idx * 7) // total_frames
            col = (idx * 7) % 8
            coverage_matrix[row, col] += 1
        
        im4 = ax4.imshow(coverage_matrix, cmap='Blues', interpolation='nearest')
        ax4.set_title('Spatial Coverage Grid')
        ax4.set_xlabel('Spatial Grid X')
        ax4.set_ylabel('Spatial Grid Y')
        
        # Add text annotations
        for i in range(8):
            for j in range(8):
                if coverage_matrix[i, j] > 0:
                    ax4.text(j, i, int(coverage_matrix[i, j]), 
                            ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im4, ax=ax4)
        
        plt.suptitle(f'Space-Aware Sampling Analysis: {video_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        save_path = self.save_dir / f"{video_name}_sampling_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Sampling analysis saved to {save_path}")
        
        return fig
    
    def visualize_feature_fusion(self, vggt_features: torch.Tensor,
                                dinov3_features: torch.Tensor,
                                fused_features: torch.Tensor,
                                video_name: str = "video") -> plt.Figure:
        """
        Visualize feature fusion between VGGT (3D) and DINOv3 (2D) features
        
        Args:
            vggt_features: 3D features from VGGT
            dinov3_features: 2D features from DINOv3  
            fused_features: Combined features
            video_name: Video name for title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert to numpy for visualization
        vggt_np = vggt_features.detach().cpu().numpy()
        dinov3_np = dinov3_features.detach().cpu().numpy()
        fused_np = fused_features.detach().cpu().numpy()
        
        # Take first frame and first 16x16 spatial dimensions for visualization
        if len(vggt_np.shape) > 2:
            vggt_vis = vggt_np[0][:256].reshape(16, 16)  # Assume first 256 dims
        else:
            vggt_vis = vggt_np[:256].reshape(16, 16)
            
        if len(dinov3_np.shape) > 2:
            dinov3_vis = dinov3_np[0][:256].reshape(16, 16)
        else:
            dinov3_vis = dinov3_np[:256].reshape(16, 16)
            
        if len(fused_np.shape) > 2:
            fused_vis = fused_np[0][:256].reshape(16, 16)
        else:
            fused_vis = fused_np[:256].reshape(16, 16)
        
        # VGGT features (3D geometry)
        im1 = axes[0, 0].imshow(vggt_vis, cmap='viridis')
        axes[0, 0].set_title('VGGT Features\n(3D Geometry)', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # DINOv3 features (2D semantics)
        im2 = axes[0, 1].imshow(dinov3_vis, cmap='plasma')
        axes[0, 1].set_title('DINOv3 Features\n(2D Semantics)', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Fused features
        im3 = axes[0, 2].imshow(fused_vis, cmap='coolwarm')
        axes[0, 2].set_title('Fused Features\n(3D + 2D)', fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Feature statistics
        axes[1, 0].hist(vggt_np.flatten(), bins=50, alpha=0.7, color='green', density=True)
        axes[1, 0].set_title('VGGT Feature Distribution')
        axes[1, 0].set_xlabel('Feature Value')
        axes[1, 0].set_ylabel('Density')
        
        axes[1, 1].hist(dinov3_np.flatten(), bins=50, alpha=0.7, color='orange', density=True)
        axes[1, 1].set_title('DINOv3 Feature Distribution')
        axes[1, 1].set_xlabel('Feature Value')
        axes[1, 1].set_ylabel('Density')
        
        axes[1, 2].hist(fused_np.flatten(), bins=50, alpha=0.7, color='purple', density=True)
        axes[1, 2].set_title('Fused Feature Distribution')
        axes[1, 2].set_xlabel('Feature Value')
        axes[1, 2].set_ylabel('Density')
        
        plt.suptitle(f'Feature Fusion Visualization: {video_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        save_path = self.save_dir / f"{video_name}_feature_fusion.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Feature fusion visualization saved to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, results: Dict[str, Any], 
                               video_name: str = "video") -> plt.Figure:
        """
        Create a comprehensive dashboard summarizing VLN pipeline results
        
        Args:
            results: Dictionary containing all pipeline outputs
            video_name: Video name for title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main heatmap display
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        if 'first_person_heatmaps' in results:
            heatmaps = results['first_person_heatmaps']
            if heatmaps.dim() == 4:
                heatmaps = heatmaps[0]  # Take first batch
            
            # Show average heatmap
            avg_heatmap = torch.mean(heatmaps, dim=0).detach().cpu().numpy()
            im_main = ax_main.imshow(avg_heatmap, cmap=self.vln_colormap)
            ax_main.set_title('Average Inter-Frame Heatmap', fontsize=14, fontweight='bold')
            ax_main.axis('off')
            plt.colorbar(im_main, ax=ax_main)
        
        # Selected frames timeline
        ax_timeline = fig.add_subplot(gs[0, 2:])
        if 'selected_indices' in results:
            indices = results['selected_indices'][0] if isinstance(results['selected_indices'][0], list) else results['selected_indices']
            total_frames = max(indices) + 5 if indices else 32
            
            all_frames = list(range(total_frames))
            colors = ['red' if i in indices else 'lightgray' for i in all_frames]
            ax_timeline.scatter(all_frames, [1] * len(all_frames), c=colors, s=30)
            
            ax_timeline.set_xlabel('Frame Index')
            ax_timeline.set_title('Selected Keyframes')
            ax_timeline.set_ylim(0.5, 1.5)
            ax_timeline.grid(True, alpha=0.3)
        
        # Processing metrics
        ax_metrics = fig.add_subplot(gs[1, 2:])
        if 'processing_time' in results:
            metrics = {
                'Processing Time (s)': results.get('processing_time', 0),
                'Selected Frames': len(results.get('selected_indices', [0])),
                'Heatmap Views': results['first_person_heatmaps'].shape[1] if 'first_person_heatmaps' in results else 0
            }
            
            bars = ax_metrics.bar(range(len(metrics)), list(metrics.values()), 
                                color=['skyblue', 'lightgreen', 'coral'])
            ax_metrics.set_xticks(range(len(metrics)))
            ax_metrics.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
            ax_metrics.set_title('Processing Metrics')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.2f}' if isinstance(value, float) else str(value),
                              ha='center', va='bottom')
        
        # Individual heatmap views
        if 'first_person_heatmaps' in results:
            heatmaps = results['first_person_heatmaps']
            if heatmaps.dim() == 4:
                heatmaps = heatmaps[0]
            
            num_views = min(heatmaps.shape[0], 4)
            for i in range(num_views):
                ax = fig.add_subplot(gs[2, i])
                heatmap = heatmaps[i].detach().cpu().numpy()
                
                im = ax.imshow(heatmap, cmap=self.vln_colormap, interpolation='bilinear')
                ax.set_title(f'View {i+1}', fontsize=10)
                ax.axis('off')
        
        # Add overall title and summary text
        plt.suptitle(f'VLN Pipeline Results Dashboard: {video_name}', 
                     fontsize=18, fontweight='bold')
        
        # Add text summary
        if 'geometry_info' in results or 'llm_outputs' in results:
            summary_text = f"Video: {video_name}\n"
            summary_text += f"Status: {'Success' if results.get('success', False) else 'Failed'}\n"
            summary_text += f"Keyframes: {len(results.get('selected_indices', []))}\n"
            summary_text += f"Processing: {results.get('processing_time', 0):.2f}s"
            
            fig.text(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Save dashboard
        save_path = self.save_dir / f"{video_name}_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Results dashboard saved to {save_path}")
        
        return fig


def create_quick_visualization(heatmaps: torch.Tensor, save_path: str = None) -> plt.Figure:
    """
    Quick utility function for basic heatmap visualization
    
    Args:
        heatmaps: Heatmap tensor
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if heatmaps.dim() == 4:
        heatmaps = heatmaps[0]  # Take first batch
    
    num_views = heatmaps.shape[0]
    cols = min(num_views, 4)
    rows = (num_views + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_views):
        heatmap = heatmaps[i].detach().cpu().numpy()
        ax = axes[i] if num_views > 1 else axes[0]
        
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_title(f'Heatmap {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_views, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Quick visualization saved to {save_path}")
    
    return fig