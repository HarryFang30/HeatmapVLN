# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# Adapted from BridgeVLA heatmap visualization utilities

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Union
from itertools import cycle


def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply softmax only on non-zero regions of the heatmap.
    
    Args:
        heatmap (torch.Tensor): Input heatmap tensor
        eps (float): Numerical stability epsilon
        
    Returns:
        torch.Tensor: Softmax-normalized heatmap
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)
    
    mask = (heatmap != 0).float()
    stable_input = heatmap * mask 
    exp_vals = torch.exp(stable_input) * mask
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)
    soft_heatmap = exp_vals / (sum_exp + eps)
    return soft_heatmap


def visualize_points_and_heatmap(
    image: Image.Image,
    points: List[Tuple[float, float]], 
    heatmap: torch.Tensor,
    save_path: str,
    point_radius: int = 3,
    heatmap_alpha: float = 0.7,
    colormap: str = 'viridis',
    title: Optional[str] = None
) -> None:
    """
    Visualize image with point annotations and corresponding heatmap overlay.
    
    Args:
        image (PIL.Image): Original RGB image
        points (List[Tuple[float, float]]): Normalized coordinates [(x, y), ...]
        heatmap (torch.Tensor): Heatmap tensor of shape [1, H, W] or [H, W]
        save_path (str): Path to save visualization
        point_radius (int): Radius for drawing points (default: 3)
        heatmap_alpha (float): Transparency of heatmap overlay (default: 0.7)
        colormap (str): Matplotlib colormap name (default: 'viridis')
        title (Optional[str]): Optional title for the visualization
    """
    img_width, img_height = image.size
    scaled_points = [(x * img_width, y * img_height) for (x, y) in points]
    
    # Create image with point annotations
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    
    # Draw points on image
    for i, (x, y) in enumerate(scaled_points):
        bbox = [
            (x - point_radius, y - point_radius),
            (x + point_radius, y + point_radius)
        ]
        color = 'red' if i == 0 else 'green'  # First point in red, others in green
        draw.ellipse(bbox, fill=color, outline=color)
    
    # Process heatmap for visualization
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.squeeze().cpu().numpy()
    else:
        heatmap_np = heatmap.squeeze()
    
    # Normalize heatmap for better visualization
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image with points
    ax1.imshow(drawable_image)
    ax1.set_title('Image with Point Annotations')
    ax1.axis('off')
    
    # Pure heatmap
    im2 = ax2.imshow(heatmap_np, cmap=colormap)
    ax2.set_title('Generated Heatmap')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay visualization
    ax3.imshow(drawable_image, alpha=1.0 - heatmap_alpha)
    im3 = ax3.imshow(heatmap_np, cmap=colormap, alpha=heatmap_alpha)
    ax3.set_title('Heatmap Overlay')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_bboxes_and_heatmap(
    image: Image.Image,
    bboxes_norm: List[Tuple[float, float, float, float]], 
    heatmap_tensor: torch.Tensor,
    save_path: str,
    bbox_colors: List[str] = ['red', 'lime', 'cyan', 'yellow'],
    bbox_width: int = 2,
    heatmap_alpha: float = 0.7,
    colormap: str = 'viridis',
    title: Optional[str] = None
) -> None:
    """
    Visualize image with bounding box annotations and heatmap overlay.
    
    Args:
        image (PIL.Image): Original RGB image
        bboxes_norm (List[Tuple[float, float, float, float]]): Normalized bboxes [(cx, cy, w, h), ...]
        heatmap_tensor (torch.Tensor): Heatmap tensor of shape [1, H, W] or [H, W]
        save_path (str): Path to save visualization
        bbox_colors (List[str]): Colors for bounding boxes (default: ['red', 'lime', 'cyan', 'yellow'])
        bbox_width (int): Width of bounding box lines (default: 2)
        heatmap_alpha (float): Transparency of heatmap overlay (default: 0.7)
        colormap (str): Matplotlib colormap name (default: 'viridis')
        title (Optional[str]): Optional title for the visualization
    """
    # Resize image to match heatmap resolution
    target_size = heatmap_tensor.shape[-1]  # Assume square heatmap
    resized_img = image.resize((target_size, target_size))
    draw = ImageDraw.Draw(resized_img)
    
    # Create color cycle
    color_cycle = cycle(bbox_colors)
    
    # Draw bounding boxes
    for bbox in bboxes_norm:
        cx, cy, w, h = bbox
        # Convert normalized coordinates to pixels
        x0 = max(0, int((cx - w/2) * target_size))
        y0 = max(0, int((cy - h/2) * target_size))
        x1 = min(target_size-1, int((cx + w/2) * target_size))
        y1 = min(target_size-1, int((cy + h/2) * target_size))
        
        current_color = next(color_cycle)
        draw.rectangle([x0, y0, x1, y1], outline=current_color, width=bbox_width)
    
    # Process heatmap
    if isinstance(heatmap_tensor, torch.Tensor):
        heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    else:
        heatmap_np = heatmap_tensor.squeeze()
    
    # Normalize for visualization
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image with bounding boxes
    ax1.imshow(resized_img)
    ax1.set_title(f'Image with {len(bboxes_norm)} Bounding Boxes')
    ax1.axis('off')
    
    # Pure heatmap
    im2 = ax2.imshow(heatmap_np, cmap=colormap)
    ax2.set_title('Generated Heatmap')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay visualization  
    ax3.imshow(resized_img, alpha=1.0 - heatmap_alpha)
    im3 = ax3.imshow(heatmap_np, cmap=colormap, alpha=heatmap_alpha)
    ax3.set_title('Heatmap Overlay')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_multi_view_heatmaps(
    images: List[Image.Image],
    heatmaps: torch.Tensor,
    save_path: str,
    view_names: Optional[List[str]] = None,
    colormap: str = 'viridis',
    heatmap_alpha: float = 0.7,
    title: Optional[str] = None
) -> None:
    """
    Visualize multi-view heatmaps from different camera angles.
    
    Args:
        images (List[PIL.Image]): List of RGB images from different views
        heatmaps (torch.Tensor): Multi-view heatmaps [num_views, H, W]
        save_path (str): Path to save visualization
        view_names (Optional[List[str]]): Names for each view (default: View 0, View 1, ...)
        colormap (str): Matplotlib colormap name (default: 'viridis')
        heatmap_alpha (float): Transparency of heatmap overlay (default: 0.7)
        title (Optional[str]): Optional title for the visualization
    """
    num_views = len(images)
    assert heatmaps.shape[0] == num_views, "Number of images and heatmaps must match"
    
    if view_names is None:
        view_names = [f'View {i}' for i in range(num_views)]
    
    # Create subplot grid
    fig, axes = plt.subplots(2, num_views, figsize=(5*num_views, 10))
    if num_views == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_views):
        # Resize image to match heatmap resolution
        target_size = heatmaps.shape[-1]
        resized_img = images[i].resize((target_size, target_size))
        
        # Process heatmap
        heatmap_np = heatmaps[i].cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        # Top row: original images
        axes[0, i].imshow(resized_img)
        axes[0, i].set_title(f'{view_names[i]} - Original')
        axes[0, i].axis('off')
        
        # Bottom row: heatmap overlays
        axes[1, i].imshow(resized_img, alpha=1.0 - heatmap_alpha)
        im = axes[1, i].imshow(heatmap_np, cmap=colormap, alpha=heatmap_alpha)
        axes[1, i].set_title(f'{view_names[i]} - Heatmap')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_attention_comparison(
    image: Image.Image,
    predicted_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    save_path: str,
    colormap: str = 'viridis',
    title: Optional[str] = None
) -> None:
    """
    Visualize comparison between predicted and target heatmaps.
    
    Args:
        image (PIL.Image): Original RGB image
        predicted_heatmap (torch.Tensor): Model-predicted heatmap [H, W]
        target_heatmap (torch.Tensor): Ground truth target heatmap [H, W]
        save_path (str): Path to save visualization
        colormap (str): Matplotlib colormap name (default: 'viridis')
        title (Optional[str]): Optional title for the visualization
    """
    # Process heatmaps
    pred_np = predicted_heatmap.squeeze().cpu().numpy()
    target_np = target_heatmap.squeeze().cpu().numpy()
    
    # Normalize for visualization
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
    
    # Calculate difference
    diff_np = np.abs(pred_np - target_np)
    
    # Resize image
    target_size = pred_np.shape[0]
    resized_img = image.resize((target_size, target_size))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: overlays
    axes[0, 0].imshow(resized_img, alpha=0.3)
    im1 = axes[0, 0].imshow(pred_np, cmap=colormap, alpha=0.7)
    axes[0, 0].set_title('Predicted Heatmap Overlay')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    axes[0, 1].imshow(resized_img, alpha=0.3)
    im2 = axes[0, 1].imshow(target_np, cmap=colormap, alpha=0.7)
    axes[0, 1].set_title('Target Heatmap Overlay')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(resized_img, alpha=0.3)
    im3 = axes[0, 2].imshow(diff_np, cmap='Reds', alpha=0.7)
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Bottom row: pure heatmaps
    im4 = axes[1, 0].imshow(pred_np, cmap=colormap)
    axes[1, 0].set_title('Predicted Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im5 = axes[1, 1].imshow(target_np, cmap=colormap)
    axes[1, 1].set_title('Target Heatmap')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im6 = axes[1, 2].imshow(diff_np, cmap='Reds')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Add metrics
    mse = np.mean((pred_np - target_np)**2)
    mae = np.mean(diff_np)
    axes[1, 2].text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                    transform=axes[1, 2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_heatmap_animation_frames(
    image: Image.Image,
    heatmap_sequence: torch.Tensor,
    output_dir: str,
    colormap: str = 'viridis',
    heatmap_alpha: float = 0.7
) -> List[str]:
    """
    Create individual frames for heatmap animation.
    
    Args:
        image (PIL.Image): Base image
        heatmap_sequence (torch.Tensor): Sequence of heatmaps [seq_len, H, W]
        output_dir (str): Directory to save frames
        colormap (str): Matplotlib colormap name (default: 'viridis')
        heatmap_alpha (float): Transparency of heatmap overlay (default: 0.7)
        
    Returns:
        List[str]: List of generated frame paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    seq_len = heatmap_sequence.shape[0]
    target_size = heatmap_sequence.shape[-1]
    resized_img = image.resize((target_size, target_size))
    
    for t in range(seq_len):
        heatmap_np = heatmap_sequence[t].cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(resized_img, alpha=1.0 - heatmap_alpha)
        im = ax.imshow(heatmap_np, cmap=colormap, alpha=heatmap_alpha)
        ax.set_title(f'Heatmap Evolution - Frame {t+1}/{seq_len}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        frame_path = os.path.join(output_dir, f'frame_{t:04d}.png')
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        frame_paths.append(frame_path)
    
    return frame_paths


if __name__ == "__main__":
    print("Testing heatmap visualization utilities...")
    
    # Create test data
    image = Image.new('RGB', (224, 224), color='lightblue')
    points = [(0.3, 0.4), (0.7, 0.6)]
    heatmap = torch.randn(224, 224)
    heatmap = torch.sigmoid(heatmap)  # Normalize to [0, 1]
    
    # Test point visualization
    visualize_points_and_heatmap(
        image, points, heatmap, 
        save_path='test_point_heatmap.png',
        title='Test Point Heatmap Visualization'
    )
    print("Point heatmap visualization saved")
    
    # Test bbox visualization
    bboxes = [(0.5, 0.5, 0.3, 0.2)]  # (cx, cy, w, h)
    visualize_bboxes_and_heatmap(
        image, bboxes, heatmap,
        save_path='test_bbox_heatmap.png',
        title='Test BBox Heatmap Visualization'
    )
    print("BBox heatmap visualization saved")
    
    # Test multi-view visualization
    images = [image, image, image]
    multi_heatmaps = torch.stack([heatmap, heatmap*0.5, heatmap*1.5])
    visualize_multi_view_heatmaps(
        images, multi_heatmaps,
        save_path='test_multiview_heatmaps.png',
        title='Test Multi-view Heatmap Visualization'
    )
    print("Multi-view heatmap visualization saved")
    
    print("Heatmap visualization utilities test completed successfully!")