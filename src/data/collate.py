"""
Custom collate functions for VLN batch processing
"""

import torch
from typing import Dict, List, Any, Optional, Union
import numpy as np
from collections import defaultdict

from ..utils.exceptions import DataLoaderError, ValidationError


def vln_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard collate function for VLN batches
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dict: Collated batch
    """
    if not batch:
        raise DataLoaderError("Empty batch provided to collate function")
    
    try:
        collated = {}
        
        # Handle frames - pad to consistent length if needed
        frames_list = [sample['frames'] for sample in batch]
        max_frames = max(frames.shape[0] for frames in frames_list)
        
        padded_frames = []
        frame_masks = []
        
        for frames in frames_list:
            T, C, H, W = frames.shape
            if T < max_frames:
                # Pad with last frame
                padding = frames[-1:].repeat(max_frames - T, 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
            
            padded_frames.append(frames)
            
            # Create mask for valid frames
            mask = torch.ones(max_frames, dtype=torch.bool)
            if T < max_frames:
                mask[T:] = False
            frame_masks.append(mask)
        
        collated['frames'] = torch.stack(padded_frames, dim=0)  # [B, T, C, H, W]
        collated['frame_masks'] = torch.stack(frame_masks, dim=0)  # [B, T]
        
        # Handle instructions
        instructions = []
        instruction_lengths = []
        
        for sample in batch:
            instr = sample['instruction']
            if isinstance(instr, dict):
                instructions.append(instr['text'])
                instruction_lengths.append(instr['length'])
            else:
                instructions.append(str(instr))
                instruction_lengths.append(len(str(instr)))
        
        collated['instructions'] = instructions
        collated['instruction_lengths'] = torch.tensor(instruction_lengths, dtype=torch.long)
        
        # Handle sample IDs
        collated['sample_ids'] = [sample.get('sample_id', i) for i, sample in enumerate(batch)]
        
        # Handle target data if present
        if 'target_heatmap' in batch[0]:
            target_heatmaps = []
            for sample in batch:
                heatmap = sample['target_heatmap']
                if not isinstance(heatmap, torch.Tensor):
                    heatmap = torch.tensor(heatmap, dtype=torch.float32)
                target_heatmaps.append(heatmap)
            
            collated['target_heatmaps'] = torch.stack(target_heatmaps, dim=0)
        
        if 'target_coordinates' in batch[0]:
            target_coords = []
            for sample in batch:
                coords = sample['target_coordinates']
                if not isinstance(coords, torch.Tensor):
                    coords = torch.tensor(coords, dtype=torch.float32)
                target_coords.append(coords)
            
            # Handle variable number of coordinates
            max_coords = max(coords.shape[0] for coords in target_coords)
            padded_coords = []
            coord_masks = []
            
            for coords in target_coords:
                num_coords = coords.shape[0]
                if num_coords < max_coords:
                    # Pad with zeros
                    padding_shape = (max_coords - num_coords,) + coords.shape[1:]
                    padding = torch.zeros(padding_shape, dtype=coords.dtype)
                    coords = torch.cat([coords, padding], dim=0)
                
                padded_coords.append(coords)
                
                # Create mask for valid coordinates
                mask = torch.zeros(max_coords, dtype=torch.bool)
                mask[:num_coords] = True
                coord_masks.append(mask)
            
            collated['target_coordinates'] = torch.stack(padded_coords, dim=0)
            collated['coordinate_masks'] = torch.stack(coord_masks, dim=0)
        
        # Handle metadata
        metadata_list = []
        for sample in batch:
            metadata = sample.get('metadata', {})
            metadata_list.append(metadata)
        collated['metadata'] = metadata_list
        
        return collated
        
    except Exception as e:
        raise DataLoaderError(
            f"Error in collate function: {str(e)}",
            batch_index=len(batch)
        ) from e


def adaptive_batch_collate(
    batch: List[Dict[str, Any]], 
    max_frames: int = 512
) -> Dict[str, Any]:
    """
    Adaptive collate function that limits total frames per batch
    
    Args:
        batch: List of sample dictionaries
        max_frames: Maximum total frames across all samples in batch
        
    Returns:
        Dict: Collated batch with potentially reduced samples
    """
    if not batch:
        raise DataLoaderError("Empty batch provided to adaptive collate function")
    
    # Calculate total frames needed
    total_frames = sum(sample['frames'].shape[0] for sample in batch)
    
    # If within limit, use standard collate
    if total_frames <= max_frames:
        return vln_collate_fn(batch)
    
    # Otherwise, reduce batch size to fit within frame limit
    selected_samples = []
    current_frames = 0
    
    for sample in batch:
        sample_frames = sample['frames'].shape[0]
        if current_frames + sample_frames <= max_frames:
            selected_samples.append(sample)
            current_frames += sample_frames
        else:
            break
    
    # Ensure we have at least one sample
    if not selected_samples:
        selected_samples = [batch[0]]
    
    return vln_collate_fn(selected_samples)


def multi_modal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhanced collate function for multi-modal VLN data
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dict: Collated batch with multi-modal support
    """
    collated = vln_collate_fn(batch)
    
    # Handle additional modalities if present
    modalities = ['depth', 'rgb', 'segmentation', 'pose', 'action']
    
    for modality in modalities:
        if modality in batch[0]:
            modality_data = []
            
            for sample in batch:
                data = sample[modality]
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                modality_data.append(data)
            
            # Stack if all have same shape, otherwise store as list
            try:
                if len(set(data.shape for data in modality_data)) == 1:
                    collated[modality] = torch.stack(modality_data, dim=0)
                else:
                    collated[modality] = modality_data
            except:
                collated[modality] = modality_data
    
    return collated


def temporal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function optimized for temporal consistency
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dict: Collated batch with temporal alignment
    """
    collated = vln_collate_fn(batch)
    
    # Add temporal indices for frame relationships
    batch_size, max_frames = collated['frames'].shape[:2]
    
    # Create temporal position embeddings
    temporal_positions = torch.arange(max_frames).unsqueeze(0).repeat(batch_size, 1)
    collated['temporal_positions'] = temporal_positions
    
    # Create frame-to-frame relationship matrix
    frame_relationships = torch.zeros(batch_size, max_frames, max_frames)
    for b in range(batch_size):
        valid_frames = collated['frame_masks'][b].sum().item()
        # Mark adjacent frame relationships
        for i in range(valid_frames - 1):
            frame_relationships[b, i, i + 1] = 1  # Next frame
            frame_relationships[b, i + 1, i] = 1  # Previous frame
    
    collated['frame_relationships'] = frame_relationships
    
    return collated


def hierarchical_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for hierarchical spatial reasoning
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dict: Collated batch with hierarchical structure
    """
    collated = vln_collate_fn(batch)
    
    # Group samples by spatial hierarchy levels if metadata available
    spatial_levels = defaultdict(list)
    
    for i, sample in enumerate(batch):
        metadata = sample.get('metadata', {})
        level = metadata.get('spatial_level', 'room')  # default to room level
        spatial_levels[level].append(i)
    
    # Add spatial level information
    collated['spatial_levels'] = dict(spatial_levels)
    
    # Create hierarchical masks
    level_masks = {}
    for level, indices in spatial_levels.items():
        mask = torch.zeros(len(batch), dtype=torch.bool)
        mask[indices] = True
        level_masks[level] = mask
    
    collated['level_masks'] = level_masks
    
    return collated


def get_collate_fn(collate_type: str = "standard") -> callable:
    """
    Factory function to get appropriate collate function
    
    Args:
        collate_type: Type of collate function to use
        
    Returns:
        callable: Collate function
    """
    collate_functions = {
        "standard": vln_collate_fn,
        "adaptive": adaptive_batch_collate,
        "multi_modal": multi_modal_collate_fn,
        "temporal": temporal_collate_fn,
        "hierarchical": hierarchical_collate_fn
    }
    
    if collate_type not in collate_functions:
        raise ValidationError(
            f"Unknown collate type: {collate_type}",
            parameter_name="collate_type",
            expected_type="one of: " + ", ".join(collate_functions.keys())
        )
    
    return collate_functions[collate_type]