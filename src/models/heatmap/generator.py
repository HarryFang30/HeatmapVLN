# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# Adapted from BridgeVLA heatmap generation utilities

import torch
import numpy as np
import ast
from typing import Tuple, Union, List, Optional


def generate_hm_from_pt(
    pt: torch.Tensor, 
    res: Union[int, Tuple[int, int]], 
    sigma: float, 
    thres_sigma_times: int = 3
) -> torch.Tensor:
    """
    Generate Gaussian heatmap from point coordinates.
    
    This function creates a 2D Gaussian heatmap centered at the given points,
    which is used as training targets for action prediction in BridgeVLA.
    
    Args:
        pt (torch.Tensor): Point coordinates of shape [num_points, 2]
        res (Union[int, Tuple[int, int]]): Output resolution (height, width)
        sigma (float): Gaussian kernel standard deviation
        thres_sigma_times (int): Threshold coefficient (default: 3)
        
    Returns:
        torch.Tensor: Generated heatmap [num_points, height, width]
    """
    num_pt, x = pt.shape
    assert x == 2, f"Point coordinates must be 2D, got {x}"
    
    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res
    
    # Create coordinate meshgrid
    _hmx = torch.arange(0, resy, device=pt.device, dtype=pt.dtype)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    
    _hmy = torch.arange(0, resx, device=pt.device, dtype=pt.dtype)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)
    
    # Calculate Gaussian distribution
    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    
    # Apply threshold
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0
    
    # Normalize each heatmap to sum to 1
    hm_sum = torch.sum(hm, (1, 2), keepdim=True)
    hm = hm / (hm_sum + 1e-6)
    
    return hm


def masked_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted average across the first dimension for non-zero regions.
    
    Args:
        tensor (torch.Tensor): Input tensor [num_heatmaps, H, W]
        
    Returns:
        torch.Tensor: Averaged tensor [1, H, W]
    """
    mask = (tensor != 0).float()
    count = mask.sum(dim=0, keepdim=True)
    count = torch.where(count == 0, torch.ones_like(count), count)
    summed = tensor.sum(dim=0, keepdim=True) / count
    return summed


def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply softmax only on non-zero regions of the heatmap.
    
    Args:
        heatmap (torch.Tensor): Input heatmap [1, H, W] or [H, W]
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


def convert_xyxy_to_cxcywh(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x1, y1, x2, y2) to (cx, cy, w, h) format.
    
    Args:
        bbox (List[float]): Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple[float, float, float, float]: Converted bbox (cx, cy, w, h)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return (cx, cy, w, h)


def generate_target_heatmap_from_annotation(
    annotation: str, 
    flag: str, 
    image_size: int = 224, 
    sigma: float = 2
) -> torch.Tensor:
    """
    Generate target heatmap from annotation data for training.
    
    This function converts different types of annotations (bounding boxes, points)
    into target heatmaps that can be used for training the BridgeVLA model.
    
    Args:
        annotation (str): Annotation string (coordinates or bounding box)
        flag (str): Annotation type flag ('detection_1', 'detection_2', etc.)
        image_size (int): Target image size (default: 224)
        sigma (float): Gaussian kernel standard deviation (default: 2)
        
    Returns:
        torch.Tensor: Target heatmap for training [1, image_size, image_size]
        
    Raises:
        ValueError: If annotation flag is not supported
    """
    
    if flag == "detection_1":
        # Single bounding box -> center point heatmap
        answer_points = ast.literal_eval(annotation)
        assert len(answer_points) == 4, f"Expected 4 bbox coordinates, got {len(answer_points)}"
        
        bbox = convert_xyxy_to_cxcywh(answer_points)
        label = torch.tensor([[bbox[0], bbox[1]]], dtype=torch.float32)  # Center point
        
        target_heatmap = generate_hm_from_pt(
            label.reshape(-1, 2) * image_size,
            (image_size, image_size),
            sigma=sigma
        )
        return target_heatmap
        
    elif flag == "detection_2":
        # Multiple points -> fused heatmap
        answer_points = ast.literal_eval(annotation)
        labels = torch.tensor(
            [[point[0], point[1]] for point in answer_points], 
            dtype=torch.float32
        )
        
        target_heatmaps = generate_hm_from_pt(
            labels.reshape(-1, 2) * image_size,
            (image_size, image_size),
            sigma=sigma
        )
        
        # Fuse multiple heatmaps
        target_heatmap = masked_mean(target_heatmaps)
        target_heatmap = masked_softmax(target_heatmap)
        return target_heatmap
        
    elif flag == "single_point":
        # Single point annotation
        point = ast.literal_eval(annotation)
        assert len(point) == 2, f"Expected 2 coordinates, got {len(point)}"
        
        label = torch.tensor([point], dtype=torch.float32)
        target_heatmap = generate_hm_from_pt(
            label * image_size,
            (image_size, image_size),
            sigma=sigma
        )
        return target_heatmap
        
    elif flag == "grasp_point":
        # Grasping point for robotic manipulation
        grasp_coords = ast.literal_eval(annotation)
        if isinstance(grasp_coords[0], list):
            # Multiple grasp candidates
            labels = torch.tensor(grasp_coords, dtype=torch.float32)
            target_heatmaps = generate_hm_from_pt(
                labels * image_size,
                (image_size, image_size),
                sigma=sigma
            )
            target_heatmap = masked_mean(target_heatmaps)
        else:
            # Single grasp point
            label = torch.tensor([grasp_coords], dtype=torch.float32)
            target_heatmap = generate_hm_from_pt(
                label * image_size,
                (image_size, image_size),
                sigma=sigma
            )
        
        return masked_softmax(target_heatmap)
    
    else:
        raise ValueError(f"Unsupported annotation flag: {flag}")


def create_multi_scale_heatmap(
    points: torch.Tensor,
    image_size: int = 224,
    scales: List[float] = [2.0, 5.0, 10.0]
) -> torch.Tensor:
    """
    Create multi-scale heatmap by combining different Gaussian scales.
    
    Args:
        points (torch.Tensor): Point coordinates [num_points, 2]
        image_size (int): Target image size
        scales (List[float]): List of sigma values for different scales
        
    Returns:
        torch.Tensor: Multi-scale heatmap [1, image_size, image_size]
    """
    heatmaps = []
    
    for sigma in scales:
        hm = generate_hm_from_pt(points, (image_size, image_size), sigma)
        heatmaps.append(hm)
    
    # Stack and take maximum across scales
    all_heatmaps = torch.stack(heatmaps, dim=0)
    multi_scale_hm = torch.max(all_heatmaps, dim=0)[0]
    
    # Normalize
    multi_scale_hm = masked_softmax(multi_scale_hm)
    
    return multi_scale_hm


def apply_heatmap_augmentation(
    heatmap: torch.Tensor,
    noise_std: float = 0.01,
    blur_sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Apply augmentation to heatmap for robust training.
    
    Args:
        heatmap (torch.Tensor): Input heatmap
        noise_std (float): Standard deviation of Gaussian noise
        blur_sigma (Optional[float]): Gaussian blur sigma (if None, no blur)
        
    Returns:
        torch.Tensor: Augmented heatmap
    """
    augmented = heatmap.clone()
    
    # Add Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
        augmented = torch.clamp(augmented, min=0)
    
    # Apply Gaussian blur (simplified version)
    if blur_sigma is not None:
        from torch.nn import functional as F
        # Simple box blur approximation
        kernel_size = int(6 * blur_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        
        # Create simple averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.to(augmented.device)
        
        augmented = F.conv2d(
            augmented.unsqueeze(1), kernel, 
            padding=padding, groups=1
        ).squeeze(1)
    
    # Renormalize
    augmented = masked_softmax(augmented)
    
    return augmented


if __name__ == "__main__":
    print("Testing heatmap generation utilities...")
    
    # Test basic heatmap generation
    points = torch.tensor([[112.0, 112.0], [56.0, 56.0]], dtype=torch.float32)
    heatmap = generate_hm_from_pt(points, res=224, sigma=10.0)
    print(f"Generated heatmap shape: {heatmap.shape}")
    print(f"Heatmap value range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    # Test annotation to heatmap conversion
    bbox_annotation = "[0.3, 0.4, 0.7, 0.6]"
    target_hm = generate_target_heatmap_from_annotation(
        bbox_annotation, "detection_1", sigma=5.0
    )
    print(f"Target heatmap shape: {target_hm.shape}")
    
    # Test multi-scale heatmap
    multi_scale_hm = create_multi_scale_heatmap(points[:1])
    print(f"Multi-scale heatmap shape: {multi_scale_hm.shape}")
    
    # Test augmentation
    augmented_hm = apply_heatmap_augmentation(heatmap[0:1], noise_std=0.01)
    print(f"Augmented heatmap shape: {augmented_hm.shape}")
    
    print("Heatmap generation utilities test completed successfully!")