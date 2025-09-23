"""
Data transforms for VLN preprocessing and augmentation
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import random
from PIL import Image
import cv2

from ..utils.exceptions import ValidationError, VideoProcessingError
from ..utils.validation import InputValidator


class VLNTransforms:
    """Base class for VLN transforms"""
    
    def __init__(self, apply_probability: float = 1.0, seed: Optional[int] = None):
        """
        Initialize transform
        
        Args:
            apply_probability: Probability of applying transform
            seed: Random seed for reproducibility
        """
        self.apply_probability = apply_probability
        self.seed = seed
        self.validator = InputValidator()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def should_apply(self) -> bool:
        """Check if transform should be applied based on probability"""
        return random.random() < self.apply_probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to sample"""
        if self.should_apply():
            return self.transform(sample)
        return sample
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform implementation - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement transform method")


class VideoResize(VLNTransforms):
    """Resize video frames to target size"""
    
    def __init__(
        self, 
        size: Union[int, Tuple[int, int]], 
        interpolation_mode: str = "bilinear",
        **kwargs
    ):
        """
        Initialize video resize transform
        
        Args:
            size: Target size (height, width) or single int for square
            interpolation_mode: Interpolation method
        """
        super().__init__(**kwargs)
        
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation_mode = interpolation_mode
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Resize video frames"""
        try:
            frames = sample['frames']  # [T, C, H, W]
            
            # Validate input
            self.validator.validate_video_tensor(frames)
            
            # Resize frames
            T, C, H, W = frames.shape
            frames_resized = F.interpolate(
                frames,
                size=self.size,
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode in ['bilinear', 'bicubic'] else None
            )
            
            sample['frames'] = frames_resized
            
            # Update target data if present
            if 'target_heatmap' in sample:
                heatmap = sample['target_heatmap']
                if isinstance(heatmap, torch.Tensor) and heatmap.dim() >= 2:
                    # Resize heatmap to match new frame size
                    if heatmap.dim() == 2:
                        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                        heatmap_resized = F.interpolate(heatmap, size=self.size, mode='bilinear', align_corners=False)
                        sample['target_heatmap'] = heatmap_resized.squeeze(0).squeeze(0)
                    else:
                        heatmap_resized = F.interpolate(heatmap, size=self.size, mode='bilinear', align_corners=False)
                        sample['target_heatmap'] = heatmap_resized
            
            return sample
            
        except Exception as e:
            raise VideoProcessingError(
                f"Error in video resize transform: {str(e)}",
                video_path=sample.get('video_path', 'unknown')
            ) from e


class VideoNormalize(VLNTransforms):
    """Normalize video frames"""
    
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        **kwargs
    ):
        """
        Initialize video normalization
        
        Args:
            mean: Channel means for normalization
            std: Channel stds for normalization
        """
        super().__init__(**kwargs)
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize video frames"""
        frames = sample['frames']  # [T, C, H, W]
        
        # Move mean and std to same device
        if frames.device != self.mean.device:
            self.mean = self.mean.to(frames.device)
            self.std = self.std.to(frames.device)
        
        # Normalize
        frames_normalized = (frames - self.mean) / self.std
        sample['frames'] = frames_normalized
        
        return sample


class VideoAugmentation(VLNTransforms):
    """Video-specific augmentations for VLN tasks"""
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        gaussian_blur_prob: float = 0.2,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
        **kwargs
    ):
        """
        Initialize video augmentation
        
        Args:
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            saturation_range: Saturation adjustment range
            hue_range: Hue adjustment range
            gaussian_blur_prob: Probability of applying Gaussian blur
            gaussian_blur_sigma: Gaussian blur sigma range
        """
        super().__init__(**kwargs)
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_sigma = gaussian_blur_sigma
        
        # Color jitter transform
        self.color_jitter = T.ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_range
        )
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply video augmentations"""
        frames = sample['frames']  # [T, C, H, W]
        T, C, H, W = frames.shape
        
        # Apply same augmentation to all frames for temporal consistency
        augmented_frames = []
        
        # Sample augmentation parameters once
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        saturation_factor = random.uniform(*self.saturation_range)
        hue_factor = random.uniform(*self.hue_range)
        
        apply_blur = random.random() < self.gaussian_blur_prob
        if apply_blur:
            blur_sigma = random.uniform(*self.gaussian_blur_sigma)
            kernel_size = int(2 * blur_sigma * 3) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            
            # Apply color augmentation
            frame = T.functional.adjust_brightness(frame, brightness_factor)
            frame = T.functional.adjust_contrast(frame, contrast_factor)
            frame = T.functional.adjust_saturation(frame, saturation_factor)
            frame = T.functional.adjust_hue(frame, hue_factor)
            
            # Apply Gaussian blur if selected
            if apply_blur:
                frame = T.functional.gaussian_blur(frame, kernel_size, [blur_sigma, blur_sigma])
            
            augmented_frames.append(frame)
        
        sample['frames'] = torch.stack(augmented_frames, dim=0)
        return sample


class SpatialAugmentation(VLNTransforms):
    """Spatial augmentations that preserve spatial relationships"""
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.1,
        rotation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        **kwargs
    ):
        """
        Initialize spatial augmentation
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_range: Rotation angle range in degrees
            scale_range: Scale factor range
            translation_range: Translation range as fraction of image size
        """
        super().__init__(**kwargs)
        
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply spatial augmentations"""
        frames = sample['frames']  # [T, C, H, W]
        T, C, H, W = frames.shape
        
        # Sample transformation parameters once for consistency
        do_hflip = random.random() < self.horizontal_flip_prob
        do_vflip = random.random() < self.vertical_flip_prob
        rotation_angle = random.uniform(*self.rotation_range)
        scale_factor = random.uniform(*self.scale_range)
        tx = random.uniform(*self.translation_range) * W
        ty = random.uniform(*self.translation_range) * H
        
        # Create affine transformation matrix
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Transformation matrix: scale, rotate, translate
        transform_matrix = torch.tensor([
            [scale_factor * cos_a, -scale_factor * sin_a, tx],
            [scale_factor * sin_a, scale_factor * cos_a, ty]
        ], dtype=torch.float32)
        
        # Apply transformations
        transformed_frames = []
        
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            
            # Apply flips
            if do_hflip:
                frame = torch.flip(frame, dims=[2])  # Flip width dimension
            if do_vflip:
                frame = torch.flip(frame, dims=[1])  # Flip height dimension
            
            # Apply affine transformation
            if abs(rotation_angle) > 0.1 or abs(scale_factor - 1.0) > 0.01 or abs(tx) > 1 or abs(ty) > 1:
                # Add batch dimension for grid_sample
                frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
                
                # Create sampling grid
                grid = F.affine_grid(
                    transform_matrix.unsqueeze(0),
                    frame_batch.shape,
                    align_corners=False
                )
                
                # Apply transformation
                frame_transformed = F.grid_sample(
                    frame_batch,
                    grid,
                    mode='bilinear',
                    padding_mode='reflection',
                    align_corners=False
                )
                
                frame = frame_transformed.squeeze(0)
            
            transformed_frames.append(frame)
        
        sample['frames'] = torch.stack(transformed_frames, dim=0)
        
        # Transform target data if present
        if 'target_heatmap' in sample:
            heatmap = sample['target_heatmap']
            if isinstance(heatmap, torch.Tensor):
                # Apply same transformations to heatmap
                if do_hflip:
                    heatmap = torch.flip(heatmap, dims=[-1])
                if do_vflip:
                    heatmap = torch.flip(heatmap, dims=[-2])
                
                # Apply affine transformation to heatmap
                if heatmap.dim() == 2:
                    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif heatmap.dim() == 3:
                    heatmap = heatmap.unsqueeze(0)  # [1, C, H, W]
                
                grid = F.affine_grid(
                    transform_matrix.unsqueeze(0),
                    heatmap.shape,
                    align_corners=False
                )
                
                heatmap_transformed = F.grid_sample(
                    heatmap,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )
                
                sample['target_heatmap'] = heatmap_transformed.squeeze()
        
        if 'target_coordinates' in sample:
            coords = sample['target_coordinates']
            if isinstance(coords, torch.Tensor) and coords.shape[-1] >= 2:
                # Transform coordinates
                coords_transformed = coords.clone()
                
                # Apply flips to coordinates
                if do_hflip:
                    coords_transformed[..., 0] = W - coords_transformed[..., 0]
                if do_vflip:
                    coords_transformed[..., 1] = H - coords_transformed[..., 1]
                
                # Apply affine transformation
                # Convert to homogeneous coordinates
                ones = torch.ones(*coords_transformed.shape[:-1], 1)
                coords_homo = torch.cat([coords_transformed[..., :2], ones], dim=-1)
                
                # Apply transformation
                transform_3x3 = torch.cat([
                    transform_matrix,
                    torch.tensor([[0., 0., 1.]], dtype=torch.float32)
                ], dim=0)
                
                coords_transformed_homo = torch.matmul(coords_homo, transform_3x3.T)
                coords_transformed[..., :2] = coords_transformed_homo[..., :2]
                
                sample['target_coordinates'] = coords_transformed
        
        return sample


class TemporalAugmentation(VLNTransforms):
    """Temporal augmentations for video sequences"""
    
    def __init__(
        self,
        frame_drop_prob: float = 0.1,
        frame_repeat_prob: float = 0.1,
        temporal_jitter_prob: float = 0.2,
        temporal_jitter_range: int = 2,
        **kwargs
    ):
        """
        Initialize temporal augmentation
        
        Args:
            frame_drop_prob: Probability of dropping random frames
            frame_repeat_prob: Probability of repeating random frames
            temporal_jitter_prob: Probability of temporal order jittering
            temporal_jitter_range: Maximum frames to jitter
        """
        super().__init__(**kwargs)
        
        self.frame_drop_prob = frame_drop_prob
        self.frame_repeat_prob = frame_repeat_prob
        self.temporal_jitter_prob = temporal_jitter_prob
        self.temporal_jitter_range = temporal_jitter_range
    
    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal augmentations"""
        frames = sample['frames']  # [T, C, H, W]
        T, C, H, W = frames.shape
        
        frame_indices = list(range(T))
        
        # Apply temporal jittering
        if random.random() < self.temporal_jitter_prob:
            jitter_amount = random.randint(1, min(self.temporal_jitter_range, T // 4))
            
            # Randomly swap adjacent frames
            for _ in range(jitter_amount):
                i = random.randint(0, T - 2)
                frame_indices[i], frame_indices[i + 1] = frame_indices[i + 1], frame_indices[i]
        
        # Apply frame dropping
        if random.random() < self.frame_drop_prob:
            num_drop = random.randint(1, max(1, T // 8))
            drop_indices = random.sample(range(T), num_drop)
            frame_indices = [i for i in frame_indices if i not in drop_indices]
        
        # Apply frame repetition
        if random.random() < self.frame_repeat_prob:
            num_repeat = random.randint(1, max(1, T // 8))
            for _ in range(num_repeat):
                repeat_idx = random.choice(frame_indices)
                insert_pos = random.randint(0, len(frame_indices))
                frame_indices.insert(insert_pos, repeat_idx)
        
        # Ensure we maintain target length
        while len(frame_indices) < T:
            frame_indices.append(frame_indices[-1])  # Repeat last frame
        
        frame_indices = frame_indices[:T]  # Truncate if too long
        
        # Apply temporal transformations
        sample['frames'] = frames[frame_indices]
        
        return sample


def create_vln_transforms(
    train: bool = True,
    image_size: Union[int, Tuple[int, int]] = 224,
    normalize: bool = True,
    augment: bool = True
) -> T.Compose:
    """
    Create standard VLN transform pipeline
    
    Args:
        train: Whether this is for training (affects augmentation)
        image_size: Target image size
        normalize: Whether to normalize frames
        augment: Whether to apply augmentations
        
    Returns:
        Composed transform pipeline
    """
    transforms = []
    
    # Always resize
    transforms.append(VideoResize(image_size))
    
    # Add augmentations for training
    if train and augment:
        transforms.extend([
            VideoAugmentation(apply_probability=0.7),
            SpatialAugmentation(apply_probability=0.5),
            TemporalAugmentation(apply_probability=0.3)
        ])
    
    # Normalize if requested
    if normalize:
        transforms.append(VideoNormalize())
    
    return T.Compose(transforms) if len(transforms) > 1 else transforms[0] if transforms else lambda x: x