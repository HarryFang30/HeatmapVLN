"""
VLN Dataset implementations for various benchmark datasets
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import logging

from ..utils.exceptions import DataLoaderError, ValidationError, ConfigurationError
from ..utils.validation import InputValidator


class VLNDataset(Dataset):
    """Base dataset class for VLN tasks"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        target_frames: int = 16,
        total_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        max_instruction_length: int = 512,
        cache_frames: bool = True,
        transform: Optional[callable] = None,
        **kwargs
    ):
        """
        Initialize VLN dataset
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split (train/val/test)
            target_frames: Target keyframes to select (N_k)
            total_frames: Total frames to process (N_m)  
            frame_size: Target frame size (height, width)
            max_instruction_length: Maximum instruction token length
            cache_frames: Whether to cache processed frames
            transform: Optional transform function
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.target_frames = target_frames
        self.total_frames = total_frames
        self.frame_size = frame_size
        self.max_instruction_length = max_instruction_length
        self.cache_frames = cache_frames
        self.transform = transform
        
        # Validation
        self.validator = InputValidator()
        self._validate_config()
        
        # Initialize data structures
        self.samples = []
        self.frame_cache = {} if cache_frames else None
        
        # Load dataset metadata
        self._load_dataset()
        
        logging.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _validate_config(self):
        """Validate dataset configuration"""
        if not self.data_path.exists():
            raise ConfigurationError(
                f"Dataset path does not exist: {self.data_path}",
                config_key="data_path"
            )
        
        if self.target_frames <= 0 or self.total_frames <= 0:
            raise ValidationError(
                "Frame counts must be positive",
                parameter_name="target_frames/total_frames"
            )
        
        if self.target_frames > self.total_frames:
            raise ValidationError(
                "Target frames cannot exceed total frames",
                parameter_name="target_frames",
                received_value=self.target_frames,
                expected_type=f"<= {self.total_frames}"
            )
    
    def _load_dataset(self):
        """Load dataset metadata - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load video frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            torch.Tensor: Video frames [T, C, H, W]
        """
        try:
            # Check cache first
            if self.cache_frames and video_path in self.frame_cache:
                return self.frame_cache[video_path]
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise DataLoaderError(
                    f"Failed to open video: {video_path}",
                    dataset_path=video_path
                )
            
            frames = []
            frame_count = 0
            
            while frame_count < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                raise DataLoaderError(
                    f"No frames extracted from video: {video_path}",
                    dataset_path=video_path
                )
            
            # Pad if necessary
            while len(frames) < self.total_frames:
                frames.append(frames[-1])  # Repeat last frame
            
            # Convert to tensor
            frames = np.stack(frames[:self.total_frames], axis=0)  # [T, H, W, C]
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
            
            # Cache if enabled
            if self.cache_frames:
                self.frame_cache[video_path] = frames
            
            return frames
            
        except Exception as e:
            raise DataLoaderError(
                f"Error loading video frames: {str(e)}",
                dataset_path=video_path
            ) from e
    
    def _load_instruction(self, instruction: Union[str, Dict]) -> Dict[str, Any]:
        """
        Process instruction data
        
        Args:
            instruction: Raw instruction data
            
        Returns:
            Dict containing processed instruction
        """
        if isinstance(instruction, str):
            return {
                'text': instruction[:self.max_instruction_length],
                'length': min(len(instruction), self.max_instruction_length)
            }
        elif isinstance(instruction, dict):
            text = instruction.get('text', '')
            return {
                'text': text[:self.max_instruction_length],
                'length': min(len(text), self.max_instruction_length),
                'metadata': instruction
            }
        else:
            raise ValidationError(
                "Invalid instruction format",
                parameter_name="instruction",
                received_value=type(instruction),
                expected_type="str or dict"
            )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        
        try:
            sample = self.samples[idx]
            
            # Load video frames
            frames = self._load_video_frames(sample['video_path'])
            
            # Process instruction
            instruction = self._load_instruction(sample['instruction'])
            
            # Create sample dict
            result = {
                'frames': frames,  # [T, C, H, W]
                'instruction': instruction,
                'sample_id': sample.get('sample_id', idx),
                'metadata': sample.get('metadata', {})
            }
            
            # Add target data if available
            if 'target_heatmap' in sample:
                result['target_heatmap'] = torch.tensor(sample['target_heatmap'])
            
            if 'target_coordinates' in sample:
                result['target_coordinates'] = torch.tensor(sample['target_coordinates'])
            
            # Apply transforms
            if self.transform:
                result = self.transform(result)
            
            return result
            
        except Exception as e:
            raise DataLoaderError(
                f"Error loading sample {idx}: {str(e)}",
                dataset_path=self.data_path,
                batch_index=idx
            ) from e


class RLBenchDataset(VLNDataset):
    """RLBench dataset for VLN tasks"""
    
    def __init__(self, data_path: str, **kwargs):
        """Initialize RLBench dataset"""
        super().__init__(data_path, **kwargs)
    
    def _load_dataset(self):
        """Load RLBench dataset metadata"""
        split_file = self.data_path / f"{self.split}_episodes.json"
        
        if not split_file.exists():
            raise ConfigurationError(
                f"RLBench split file not found: {split_file}",
                config_key="split_file"
            )
        
        try:
            with open(split_file, 'r') as f:
                episodes = json.load(f)
            
            for episode in episodes:
                # RLBench specific structure
                video_path = self.data_path / "videos" / f"{episode['episode_id']}.mp4"
                
                if video_path.exists():
                    sample = {
                        'video_path': str(video_path),
                        'instruction': episode['instruction'],
                        'sample_id': episode['episode_id'],
                        'metadata': {
                            'task_name': episode.get('task_name', ''),
                            'success': episode.get('success', False),
                            'episode_length': episode.get('episode_length', 0)
                        }
                    }
                    
                    # Add target data if available
                    if 'target_locations' in episode:
                        sample['target_coordinates'] = episode['target_locations']
                    
                    self.samples.append(sample)
                    
        except Exception as e:
            raise DataLoaderError(
                f"Error loading RLBench dataset: {str(e)}",
                dataset_path=str(self.data_path)
            ) from e


class ColosseumDataset(VLNDataset):
    """COLOSSEUM dataset for VLN tasks"""
    
    def __init__(self, data_path: str, **kwargs):
        """Initialize COLOSSEUM dataset"""
        super().__init__(data_path, **kwargs)
    
    def _load_dataset(self):
        """Load COLOSSEUM dataset metadata"""
        annotations_file = self.data_path / "annotations" / f"{self.split}.json"
        
        if not annotations_file.exists():
            raise ConfigurationError(
                f"COLOSSEUM annotations not found: {annotations_file}",
                config_key="annotations_file"
            )
        
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                video_path = self.data_path / "videos" / ann['video_file']
                
                if video_path.exists():
                    sample = {
                        'video_path': str(video_path),
                        'instruction': ann['navigation_instruction'],
                        'sample_id': ann['sample_id'],
                        'metadata': {
                            'environment': ann.get('environment', ''),
                            'path_length': ann.get('path_length', 0),
                            'difficulty': ann.get('difficulty', 'medium')
                        }
                    }
                    
                    # Add spatial targets
                    if 'spatial_targets' in ann:
                        sample['target_heatmap'] = ann['spatial_targets']
                    
                    self.samples.append(sample)
                    
        except Exception as e:
            raise DataLoaderError(
                f"Error loading COLOSSEUM dataset: {str(e)}",
                dataset_path=str(self.data_path)
            ) from e


class CustomVLNDataset(VLNDataset):
    """Custom VLN dataset with flexible structure"""
    
    def __init__(
        self,
        data_path: str,
        video_dir: str = "videos",
        annotations_file: str = "annotations.json",
        **kwargs
    ):
        """
        Initialize custom VLN dataset
        
        Args:
            data_path: Root dataset directory
            video_dir: Subdirectory containing video files
            annotations_file: JSON file with annotations
        """
        self.video_dir = video_dir
        self.annotations_file = annotations_file
        super().__init__(data_path, **kwargs)
    
    def _load_dataset(self):
        """Load custom dataset metadata"""
        annotations_path = self.data_path / self.annotations_file
        
        if not annotations_path.exists():
            raise ConfigurationError(
                f"Custom annotations not found: {annotations_path}",
                config_key="annotations_file"
            )
        
        try:
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            
            # Support different JSON structures
            samples_data = data.get(self.split, data)  # Try split-specific, fallback to full data
            
            if isinstance(samples_data, dict):
                samples_data = samples_data.values()
            
            for item in samples_data:
                video_file = item.get('video_file') or item.get('video_path')
                if not video_file:
                    continue
                
                video_path = self.data_path / self.video_dir / video_file
                
                if video_path.exists():
                    sample = {
                        'video_path': str(video_path),
                        'instruction': item.get('instruction', ''),
                        'sample_id': item.get('id', len(self.samples)),
                        'metadata': {k: v for k, v in item.items() 
                                   if k not in ['video_file', 'video_path', 'instruction', 'id']}
                    }
                    
                    # Flexible target data handling
                    for target_key in ['target_heatmap', 'heatmap', 'spatial_target']:
                        if target_key in item:
                            sample['target_heatmap'] = item[target_key]
                            break
                    
                    for coord_key in ['target_coordinates', 'coordinates', 'target_points']:
                        if coord_key in item:
                            sample['target_coordinates'] = item[coord_key]
                            break
                    
                    self.samples.append(sample)
                    
        except Exception as e:
            raise DataLoaderError(
                f"Error loading custom dataset: {str(e)}",
                dataset_path=str(self.data_path)
            ) from e


def create_vln_dataset(
    dataset_type: str,
    data_path: str,
    split: str = "train",
    **kwargs
) -> VLNDataset:
    """
    Factory function to create VLN datasets
    
    Args:
        dataset_type: Type of dataset ("rlbench", "colosseum", "custom")
        data_path: Path to dataset
        split: Dataset split
        **kwargs: Additional arguments for dataset
        
    Returns:
        VLNDataset: Configured dataset instance
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type == "rlbench":
        return RLBenchDataset(data_path, split=split, **kwargs)
    elif dataset_type == "colosseum":
        return ColosseumDataset(data_path, split=split, **kwargs)
    elif dataset_type == "custom":
        return CustomVLNDataset(data_path, split=split, **kwargs)
    else:
        raise ConfigurationError(
            f"Unknown dataset type: {dataset_type}",
            config_key="dataset_type",
            expected_value="rlbench, colosseum, or custom"
        )