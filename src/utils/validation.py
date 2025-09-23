"""
Validation utilities for VLN Project
Provides comprehensive input validation and error checking
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

from .exceptions import (
    ValidationError, ConfigurationError, VideoProcessingError, 
    ResourceError, GeometryError
)

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validation for VLN pipeline components
    """
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                            name: str = "tensor", allow_batch: bool = True) -> torch.Tensor:
        """
        Validate tensor shape
        
        Args:
            tensor: Input tensor
            expected_shape: Expected shape (can use -1 for any size)
            name: Name for error reporting
            allow_batch: Whether to allow batch dimension
            
        Returns:
            Validated tensor
            
        Raises:
            ValidationError: If shape doesn't match
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                f"{name} must be a torch.Tensor",
                parameter_name=name,
                received_value=type(tensor),
                expected_type=torch.Tensor
            )
        
        actual_shape = tensor.shape
        
        # Handle batch dimension
        if allow_batch and len(actual_shape) == len(expected_shape) + 1:
            actual_shape = actual_shape[1:]  # Skip batch dimension
        
        if len(actual_shape) != len(expected_shape):
            raise ValidationError(
                f"{name} has wrong number of dimensions: got {len(actual_shape)}, expected {len(expected_shape)}",
                parameter_name=name,
                received_value=actual_shape,
                expected_type=expected_shape
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValidationError(
                    f"{name} dimension {i} mismatch: got {actual}, expected {expected}",
                    parameter_name=f"{name}_dim_{i}",
                    received_value=actual,
                    expected_type=expected
                )
        
        return tensor
    
    @staticmethod
    def validate_tensor_range(tensor: torch.Tensor, min_val: float = None, max_val: float = None,
                            name: str = "tensor") -> torch.Tensor:
        """
        Validate tensor value range
        
        Args:
            tensor: Input tensor
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error reporting
            
        Returns:
            Validated tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch.Tensor", parameter_name=name)
        
        if min_val is not None and torch.any(tensor < min_val):
            raise ValidationError(
                f"{name} contains values below minimum {min_val}",
                parameter_name=name,
                received_value=f"min={tensor.min().item()}"
            )
        
        if max_val is not None and torch.any(tensor > max_val):
            raise ValidationError(
                f"{name} contains values above maximum {max_val}",
                parameter_name=name,
                received_value=f"max={tensor.max().item()}"
            )
        
        # Check for NaN and Inf
        if torch.any(torch.isnan(tensor)):
            raise ValidationError(f"{name} contains NaN values", parameter_name=name)
        
        if torch.any(torch.isinf(tensor)):
            raise ValidationError(f"{name} contains infinite values", parameter_name=name)
        
        return tensor
    
    @staticmethod
    def validate_video_tensor(video_tensor: torch.Tensor, name: str = "video_tensor") -> torch.Tensor:
        """
        Validate video tensor format and content
        
        Args:
            video_tensor: Video tensor [B, T, C, H, W] or [T, C, H, W]
            name: Name for error reporting
            
        Returns:
            Validated video tensor
        """
        if not isinstance(video_tensor, torch.Tensor):
            raise VideoProcessingError(f"{name} must be a torch.Tensor")
        
        # Check dimensions
        if video_tensor.dim() not in [4, 5]:
            raise VideoProcessingError(
                f"{name} must be 4D [T, C, H, W] or 5D [B, T, C, H, W], got {video_tensor.dim()}D"
            )
        
        # Check minimum requirements
        if video_tensor.dim() == 4:  # [T, C, H, W]
            T, C, H, W = video_tensor.shape
            if T < 1:
                raise VideoProcessingError(f"{name} must have at least 1 frame, got {T}")
            if C not in [1, 3]:
                raise VideoProcessingError(f"{name} must have 1 or 3 channels, got {C}")
            if H < 32 or W < 32:
                raise VideoProcessingError(f"{name} frames too small: {H}x{W}, minimum 32x32")
        
        elif video_tensor.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = video_tensor.shape
            if B < 1:
                raise VideoProcessingError(f"{name} must have at least 1 sample in batch, got {B}")
            if T < 1:
                raise VideoProcessingError(f"{name} must have at least 1 frame, got {T}")
            if C not in [1, 3]:
                raise VideoProcessingError(f"{name} must have 1 or 3 channels, got {C}")
            if H < 32 or W < 32:
                raise VideoProcessingError(f"{name} frames too small: {H}x{W}, minimum 32x32")
        
        # Validate value range (assuming normalized to [0, 1] or [-1, 1])
        min_val, max_val = video_tensor.min().item(), video_tensor.max().item()
        if not ((-1.5 <= min_val <= 1.5) and (-1.5 <= max_val <= 1.5)):
            logger.warning(f"{name} values outside expected range [0,1] or [-1,1]: [{min_val:.3f}, {max_val:.3f}]")
        
        return video_tensor
    
    @staticmethod
    def validate_heatmap_tensor(heatmap_tensor: torch.Tensor, name: str = "heatmap") -> torch.Tensor:
        """
        Validate heatmap tensor format
        
        Args:
            heatmap_tensor: Heatmap tensor [B, N_views, H, W] or [N_views, H, W]
            name: Name for error reporting
            
        Returns:
            Validated heatmap tensor
        """
        if not isinstance(heatmap_tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch.Tensor", parameter_name=name)
        
        if heatmap_tensor.dim() not in [3, 4]:
            raise ValidationError(
                f"{name} must be 3D [N_views, H, W] or 4D [B, N_views, H, W]",
                parameter_name=f"{name}_dimensions",
                received_value=heatmap_tensor.dim()
            )
        
        # Validate heatmap values (should be non-negative)
        if torch.any(heatmap_tensor < 0):
            raise ValidationError(
                f"{name} contains negative values",
                parameter_name=name,
                received_value=f"min={heatmap_tensor.min().item()}"
            )
        
        # Check for valid probability distribution (values should sum to reasonable range)
        if heatmap_tensor.dim() == 3:
            for i in range(heatmap_tensor.shape[0]):
                heatmap_sum = torch.sum(heatmap_tensor[i])
                if heatmap_sum < 1e-6:
                    logger.warning(f"{name} view {i} is nearly empty (sum={heatmap_sum:.6f})")
        
        return heatmap_tensor
    
    @staticmethod
    def validate_geometry_info(geometry_info: Dict[str, torch.Tensor], 
                             expected_frames: int = None, 
                             name: str = "geometry_info") -> Dict[str, torch.Tensor]:
        """
        Validate geometry information dictionary
        
        Args:
            geometry_info: Dictionary containing geometry tensors
            expected_frames: Expected number of frames
            name: Name for error reporting
            
        Returns:
            Validated geometry info
        """
        if not isinstance(geometry_info, dict):
            raise GeometryError(f"{name} must be a dictionary")
        
        required_keys = ['camera_poses', 'depth_maps']
        for key in required_keys:
            if key not in geometry_info:
                raise GeometryError(f"{name} missing required key: {key}")
        
        # Validate camera poses
        poses = geometry_info['camera_poses']
        if not isinstance(poses, torch.Tensor):
            raise GeometryError(f"{name}.camera_poses must be a torch.Tensor")
        
        if poses.dim() not in [3, 4]:  # [T, 4, 4] or [B, T, 4, 4]
            raise GeometryError(f"{name}.camera_poses must be 3D or 4D, got {poses.dim()}D")
        
        if poses.shape[-2:] != (4, 4):
            raise GeometryError(f"{name}.camera_poses last dimensions must be [4, 4]")
        
        # Validate depth maps
        depths = geometry_info['depth_maps']
        if not isinstance(depths, torch.Tensor):
            raise GeometryError(f"{name}.depth_maps must be a torch.Tensor")
        
        if depths.dim() not in [3, 4]:  # [T, H, W] or [B, T, H, W]
            raise GeometryError(f"{name}.depth_maps must be 3D or 4D, got {depths.dim()}D")
        
        # Check frame consistency
        if expected_frames is not None:
            pose_frames = poses.shape[-3] if poses.dim() == 4 else poses.shape[0]
            depth_frames = depths.shape[-3] if depths.dim() == 4 else depths.shape[0]
            
            if pose_frames != expected_frames:
                raise GeometryError(
                    f"{name}.camera_poses frame count mismatch: got {pose_frames}, expected {expected_frames}"
                )
            
            if depth_frames != expected_frames:
                raise GeometryError(
                    f"{name}.depth_maps frame count mismatch: got {depth_frames}, expected {expected_frames}"
                )
        
        return geometry_info
    
    @staticmethod
    def validate_frame_indices(indices: List[int], total_frames: int, 
                             name: str = "frame_indices") -> List[int]:
        """
        Validate frame indices
        
        Args:
            indices: List of frame indices
            total_frames: Total number of available frames
            name: Name for error reporting
            
        Returns:
            Validated indices
        """
        if not isinstance(indices, (list, tuple)):
            raise ValidationError(f"{name} must be a list or tuple", parameter_name=name)
        
        if len(indices) == 0:
            raise ValidationError(f"{name} cannot be empty", parameter_name=name)
        
        for i, idx in enumerate(indices):
            if not isinstance(idx, int):
                raise ValidationError(
                    f"{name}[{i}] must be an integer",
                    parameter_name=f"{name}[{i}]",
                    received_value=type(idx)
                )
            
            if idx < 0 or idx >= total_frames:
                raise ValidationError(
                    f"{name}[{i}] out of range: {idx}, valid range [0, {total_frames-1}]",
                    parameter_name=f"{name}[{i}]",
                    received_value=idx
                )
        
        # Check for duplicates
        if len(set(indices)) != len(indices):
            raise ValidationError(f"{name} contains duplicate indices", parameter_name=name)
        
        return list(indices)
    
    @staticmethod
    def validate_configuration(config: Dict[str, Any], schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary
            schema: Optional schema for validation
            
        Returns:
            Validated configuration
        """
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Basic required sections
        required_sections = ['video', 'frame_sampling', 'heatmap', 'system']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate video configuration
        video_config = config['video']
        if 'total_frames' not in video_config:
            raise ConfigurationError("video.total_frames is required")
        if 'keyframes' not in video_config:
            raise ConfigurationError("video.keyframes is required")
        
        total_frames = video_config['total_frames']
        keyframes = video_config['keyframes']
        
        if not isinstance(total_frames, int) or total_frames <= 0:
            raise ConfigurationError(
                "video.total_frames must be a positive integer",
                config_key="video.total_frames",
                expected_value="> 0"
            )
        
        if not isinstance(keyframes, int) or keyframes <= 0:
            raise ConfigurationError(
                "video.keyframes must be a positive integer",
                config_key="video.keyframes",
                expected_value="> 0"
            )
        
        if keyframes > total_frames:
            raise ConfigurationError(
                f"video.keyframes ({keyframes}) cannot be greater than video.total_frames ({total_frames})",
                config_key="video.keyframes"
            )
        
        # Validate frame_size
        if 'frame_size' in video_config:
            frame_size = video_config['frame_size']
            if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
                raise ConfigurationError(
                    "video.frame_size must be a list/tuple of 2 integers",
                    config_key="video.frame_size"
                )
            
            h, w = frame_size
            if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
                raise ConfigurationError(
                    "video.frame_size values must be positive integers",
                    config_key="video.frame_size"
                )
        
        # Validate heatmap configuration
        heatmap_config = config['heatmap']
        if 'target_size' not in heatmap_config:
            raise ConfigurationError("heatmap.target_size is required")
        
        target_size = heatmap_config['target_size']
        if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
            raise ConfigurationError(
                "heatmap.target_size must be a list/tuple of 2 integers",
                config_key="heatmap.target_size"
            )
        
        return config
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], must_exist: bool = True, 
                         file_type: str = None, name: str = "file") -> Path:
        """
        Validate file path
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            file_type: Expected file extension (e.g., '.mp4', '.json')
            name: Name for error reporting
            
        Returns:
            Validated Path object
        """
        if not isinstance(file_path, (str, Path)):
            raise ValidationError(
                f"{name} must be a string or Path object",
                parameter_name=name,
                received_value=type(file_path)
            )
        
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(
                f"{name} does not exist: {path}",
                parameter_name=name,
                received_value=str(path)
            )
        
        if file_type and path.suffix.lower() != file_type.lower():
            raise ValidationError(
                f"{name} must have extension {file_type}",
                parameter_name=name,
                received_value=path.suffix,
                expected_type=file_type
            )
        
        return path
    
    @staticmethod
    def validate_system_resources(required_memory_gb: float = None, 
                                required_gpu_memory_gb: float = None) -> Dict[str, Any]:
        """
        Validate system resources
        
        Args:
            required_memory_gb: Required RAM in GB
            required_gpu_memory_gb: Required GPU memory in GB
            
        Returns:
            System resource information
        """
        import psutil
        
        resources = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1e9,
            'available_memory_gb': psutil.virtual_memory().available / 1e9,
        }
        
        # Check memory requirements
        if required_memory_gb and resources['available_memory_gb'] < required_memory_gb:
            raise ResourceError(
                f"Insufficient memory: need {required_memory_gb}GB, have {resources['available_memory_gb']:.1f}GB",
                resource_type="memory",
                required_amount=f"{required_memory_gb}GB",
                available_amount=f"{resources['available_memory_gb']:.1f}GB"
            )
        
        # Check GPU resources
        if torch.cuda.is_available():
            resources['gpu_count'] = torch.cuda.device_count()
            resources['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            resources['gpu_name'] = torch.cuda.get_device_name(0)
            
            if required_gpu_memory_gb and resources['gpu_memory_gb'] < required_gpu_memory_gb:
                raise ResourceError(
                    f"Insufficient GPU memory: need {required_gpu_memory_gb}GB, have {resources['gpu_memory_gb']:.1f}GB",
                    resource_type="gpu_memory",
                    required_amount=f"{required_gpu_memory_gb}GB",
                    available_amount=f"{resources['gpu_memory_gb']:.1f}GB"
                )
        else:
            resources['gpu_count'] = 0
            resources['gpu_memory_gb'] = 0
            resources['gpu_name'] = "None"
            
            if required_gpu_memory_gb:
                raise ResourceError(
                    "GPU required but not available",
                    resource_type="gpu",
                    required_amount="1 GPU",
                    available_amount="0 GPUs"
                )
        
        return resources


def validate_pipeline_inputs(video_frames: torch.Tensor,
                           text_instructions: List[str],
                           geometry_info: Dict[str, torch.Tensor] = None,
                           config: Dict[str, Any] = None) -> None:
    """
    Validate complete pipeline inputs
    
    Args:
        video_frames: Video tensor
        text_instructions: List of text instructions
        geometry_info: Optional geometry information
        config: Optional configuration
    """
    validator = InputValidator()
    
    # Validate video frames
    video_frames = validator.validate_video_tensor(video_frames, "video_frames")
    
    # Validate text instructions
    if not isinstance(text_instructions, list):
        raise ValidationError("text_instructions must be a list")
    
    if len(text_instructions) == 0:
        raise ValidationError("text_instructions cannot be empty")
    
    for i, instruction in enumerate(text_instructions):
        if not isinstance(instruction, str):
            raise ValidationError(f"text_instructions[{i}] must be a string")
        
        if len(instruction.strip()) == 0:
            raise ValidationError(f"text_instructions[{i}] cannot be empty")
    
    # Validate geometry info if provided
    if geometry_info is not None:
        expected_frames = video_frames.shape[-4] if video_frames.dim() == 5 else video_frames.shape[0]
        validator.validate_geometry_info(geometry_info, expected_frames)
    
    # Validate configuration if provided
    if config is not None:
        validator.validate_configuration(config)


def create_validation_decorator(validation_func: Callable):
    """
    Create a decorator for input validation
    
    Args:
        validation_func: Function to validate inputs
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Apply validation
            validation_func(*args, **kwargs)
            # Call original function
            return func(*args, **kwargs)
        return wrapper
    return decorator