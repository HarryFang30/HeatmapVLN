"""
Data preprocessing script for VLN Spatial-MLLM Pipeline
Handles video format standardization, frame extraction, and data validation
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from tqdm import tqdm
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


def extract_video_frames(video_path: str, output_dir: str, max_frames: int = 32,
                        target_size: tuple = (224, 224)) -> Dict[str, Any]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        max_frames: Maximum frames to extract
        target_size: Target frame size (H, W)
        
    Returns:
        Dictionary with extraction metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract (uniform sampling)
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
    
    extracted_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        extracted_frames.append({
            'frame_idx': i,
            'original_idx': frame_idx,
            'path': frame_path
        })
    
    cap.release()
    
    metadata = {
        'video_path': video_path,
        'output_dir': output_dir,
        'total_original_frames': total_frames,
        'extracted_frames': len(extracted_frames),
        'fps': fps,
        'target_size': target_size,
        'frames': extracted_frames
    }
    
    return metadata


def validate_video_file(video_path: str) -> Dict[str, Any]:
    """
    Validate video file and extract basic information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information and validation status
    """
    if not os.path.exists(video_path):
        return {'valid': False, 'error': 'File does not exist'}
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'valid': False, 'error': 'Cannot open video file'}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'valid': True,
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'file_size': os.path.getsize(video_path)
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def process_annotation_file(annotation_path: str) -> List[Dict[str, Any]]:
    """
    Process annotation file for VLN tasks.
    
    Args:
        annotation_path: Path to annotation file
        
    Returns:
        List of processed annotations
    """
    if not os.path.exists(annotation_path):
        return []
    
    # Load annotations (assuming JSON format)
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        # Validate and standardize annotations
        processed = []
        for ann in annotations:
            if isinstance(ann, dict) and 'instruction' in ann:
                processed_ann = {
                    'instruction': ann['instruction'],
                    'video_path': ann.get('video_path', ''),
                    'target_locations': ann.get('target_locations', []),
                    'spatial_relationships': ann.get('spatial_relationships', []),
                    'metadata': ann.get('metadata', {})
                }
                processed.append(processed_ann)
        
        return processed
        
    except Exception as e:
        print(f"Error processing annotation file {annotation_path}: {e}")
        return []


def create_dataset_split(data_list: List[Dict[str, Any]], 
                        train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data_list: List of data items
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Dictionary with train/val/test splits
    """
    np.random.shuffle(data_list)
    
    n_total = len(data_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    splits = {
        'train': data_list[:n_train],
        'val': data_list[n_train:n_train + n_val],
        'test': data_list[n_train + n_val:]
    }
    
    return splits


def preprocess_data(config: Dict[str, Any], args):
    """
    Main preprocessing function called from main.py
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("preprocess", config['logging']['level'])
    logger.info("Starting VLN data preprocessing pipeline")
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).rglob(f"*{ext}"))
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process each video
    processed_data = []
    failed_videos = []
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Validate video
            validation_info = validate_video_file(str(video_path))
            if not validation_info['valid']:
                logger.warning(f"Invalid video {video_path}: {validation_info['error']}")
                failed_videos.append({
                    'path': str(video_path),
                    'error': validation_info['error']
                })
                continue
            
            # Extract frames
            video_name = video_path.stem
            frames_output_dir = os.path.join(output_dir, 'frames', video_name)
            
            extraction_metadata = extract_video_frames(
                str(video_path),
                frames_output_dir,
                max_frames=config['video']['total_frames'],
                target_size=tuple(config['video']['frame_size'])
            )
            
            # Look for corresponding annotation file
            annotation_path = video_path.with_suffix('.json')
            annotations = process_annotation_file(str(annotation_path))
            
            # Create data entry
            data_entry = {
                'video_name': video_name,
                'original_video_path': str(video_path),
                'frames_dir': frames_output_dir,
                'video_info': validation_info,
                'extraction_metadata': extraction_metadata,
                'annotations': annotations
            }
            
            processed_data.append(data_entry)
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {str(e)}")
            failed_videos.append({
                'path': str(video_path),
                'error': str(e)
            })
    
    # Create dataset splits
    if processed_data:
        splits = create_dataset_split(processed_data)
        
        # Save splits to files
        for split_name, split_data in splits.items():
            split_file = os.path.join(output_dir, f"{split_name}_data.json")
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"Saved {len(split_data)} items to {split_file}")
    
    # Save processing summary
    summary = {
        'input_directory': input_dir,
        'output_directory': output_dir,
        'total_videos_found': len(video_files),
        'successfully_processed': len(processed_data),
        'failed_videos': len(failed_videos),
        'processing_config': {
            'max_frames': config['video']['total_frames'],
            'target_size': config['video']['frame_size']
        },
        'dataset_splits': {
            'train': len(splits['train']) if processed_data else 0,
            'val': len(splits['val']) if processed_data else 0,
            'test': len(splits['test']) if processed_data else 0
        } if processed_data else {},
        'failed_videos': failed_videos
    }
    
    summary_file = os.path.join(output_dir, 'preprocessing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Videos found: {len(video_files)}")
    print(f"Successfully processed: {len(processed_data)}")
    print(f"Failed: {len(failed_videos)}")
    
    if processed_data:
        print(f"Dataset splits:")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Validation: {len(splits['val'])}")
        print(f"  Test: {len(splits['test'])}")
    
    print(f"Summary saved to: {summary_file}")
    print("="*50)
    
    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    print("VLN Preprocessing Script - Run via main.py for full functionality")