"""
VLN DataLoader utilities with optimized batch processing
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import Dict, Any, List, Optional, Union
import numpy as np
import random
from collections import defaultdict
import logging

from ..utils.exceptions import DataLoaderError, ValidationError
from .dataset import VLNDataset
from .collate import vln_collate_fn, adaptive_batch_collate


class VLNDataLoader(DataLoader):
    """Specialized DataLoader for VLN tasks"""
    
    def __init__(
        self,
        dataset: VLNDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[callable] = None,
        sampler: Optional[Sampler] = None,
        adaptive_batching: bool = True,
        max_frames_per_batch: int = 512,
        **kwargs
    ):
        """
        Initialize VLN DataLoader
        
        Args:
            dataset: VLNDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function
            sampler: Custom sampler
            adaptive_batching: Whether to use adaptive batch sizing
            max_frames_per_batch: Maximum total frames per batch
            **kwargs: Additional DataLoader arguments
        """
        # Set default collate function
        if collate_fn is None:
            if adaptive_batching:
                collate_fn = lambda batch: adaptive_batch_collate(
                    batch, max_frames=max_frames_per_batch
                )
            else:
                collate_fn = vln_collate_fn
        
        # Custom sampler for balanced sampling
        if sampler is None and shuffle and hasattr(dataset, 'get_balanced_sampler'):
            try:
                sampler = dataset.get_balanced_sampler()
                shuffle = False  # Disable shuffle when using custom sampler
            except:
                pass  # Fall back to default shuffling
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            sampler=sampler,
            **kwargs
        )
        
        self.adaptive_batching = adaptive_batching
        self.max_frames_per_batch = max_frames_per_batch
        
        logging.info(
            f"Initialized VLNDataLoader: batch_size={batch_size}, "
            f"adaptive_batching={adaptive_batching}, "
            f"max_frames_per_batch={max_frames_per_batch}"
        )


class BalancedTaskSampler(Sampler):
    """Sampler that balances samples across different tasks"""
    
    def __init__(
        self,
        dataset: VLNDataset,
        samples_per_task: int = None,
        shuffle: bool = True,
        seed: int = None
    ):
        """
        Initialize balanced task sampler
        
        Args:
            dataset: VLNDataset instance
            samples_per_task: Number of samples per task (None for auto)
            shuffle: Whether to shuffle samples within tasks
            seed: Random seed for reproducibility
        """
        super().__init__(dataset)
        
        self.dataset = dataset
        self.samples_per_task = samples_per_task
        self.shuffle = shuffle
        self.seed = seed
        
        # Group samples by task
        self.task_groups = self._group_by_task()
        
        # Determine samples per task
        if self.samples_per_task is None:
            min_task_size = min(len(samples) for samples in self.task_groups.values())
            self.samples_per_task = max(1, min_task_size)
        
        logging.info(f"BalancedTaskSampler: {len(self.task_groups)} tasks, {self.samples_per_task} samples per task")
    
    def _group_by_task(self) -> Dict[str, List[int]]:
        """Group sample indices by task"""
        task_groups = defaultdict(list)
        
        for idx, sample in enumerate(self.dataset.samples):
            # Extract task identifier from metadata
            task_name = "default"
            
            if 'metadata' in sample:
                metadata = sample['metadata']
                # Try different task identifier keys
                for key in ['task_name', 'task', 'environment', 'category']:
                    if key in metadata:
                        task_name = metadata[key]
                        break
            
            task_groups[task_name].append(idx)
        
        return dict(task_groups)
    
    def __iter__(self):
        """Iterate through balanced samples"""
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None
        
        indices = []
        
        for task_name, task_indices in self.task_groups.items():
            # Sample from this task
            if len(task_indices) >= self.samples_per_task:
                # Sample without replacement
                if self.shuffle:
                    if generator:
                        sampled = torch.randperm(len(task_indices), generator=generator)[:self.samples_per_task]
                    else:
                        sampled = torch.randperm(len(task_indices))[:self.samples_per_task]
                    sampled_indices = [task_indices[i] for i in sampled]
                else:
                    sampled_indices = task_indices[:self.samples_per_task]
            else:
                # Sample with replacement if needed
                if self.shuffle:
                    sampled_indices = random.choices(task_indices, k=self.samples_per_task)
                else:
                    # Repeat indices to reach target count
                    sampled_indices = (task_indices * ((self.samples_per_task // len(task_indices)) + 1))[:self.samples_per_task]
            
            indices.extend(sampled_indices)
        
        # Final shuffle of all selected indices
        if self.shuffle:
            if generator:
                perm = torch.randperm(len(indices), generator=generator)
                indices = [indices[i] for i in perm]
            else:
                random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return len(self.task_groups) * self.samples_per_task


class DynamicBatchSampler(Sampler):
    """Sampler that creates batches with similar frame counts for efficiency"""
    
    def __init__(
        self,
        dataset: VLNDataset,
        batch_size: int = 4,
        max_frames_per_batch: int = 512,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize dynamic batch sampler
        
        Args:
            dataset: VLNDataset instance
            batch_size: Target batch size
            max_frames_per_batch: Maximum total frames per batch
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop last incomplete batch
        """
        super().__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_frames_per_batch = max_frames_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Pre-compute frame counts for efficiency
        self.frame_counts = self._get_frame_counts()
        
        logging.info(f"DynamicBatchSampler: target_batch_size={batch_size}, max_frames_per_batch={max_frames_per_batch}")
    
    def _get_frame_counts(self) -> List[int]:
        """Get frame counts for all samples"""
        frame_counts = []
        
        for sample in self.dataset.samples:
            # Use total_frames from dataset as estimate
            frame_count = getattr(self.dataset, 'total_frames', 32)
            frame_counts.append(frame_count)
        
        return frame_counts
    
    def __iter__(self):
        """Create dynamic batches"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        batches = []
        current_batch = []
        current_frame_count = 0
        
        for idx in indices:
            frame_count = self.frame_counts[idx]
            
            # Check if adding this sample would exceed limits
            if (len(current_batch) >= self.batch_size or 
                current_frame_count + frame_count > self.max_frames_per_batch):
                
                if current_batch:  # Don't add empty batches
                    batches.append(current_batch)
                    current_batch = []
                    current_frame_count = 0
            
            current_batch.append(idx)
            current_frame_count += frame_count
        
        # Handle last batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        return iter(batches)
    
    def __len__(self):
        """Estimate number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_vln_dataloader(
    dataset: VLNDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler_type: str = "default",
    adaptive_batching: bool = True,
    max_frames_per_batch: int = 512,
    balanced_sampling: bool = False,
    **kwargs
) -> VLNDataLoader:
    """
    Factory function to create optimized VLN dataloaders
    
    Args:
        dataset: VLNDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        sampler_type: Type of sampler ("default", "balanced", "dynamic")
        adaptive_batching: Whether to use adaptive batch sizing
        max_frames_per_batch: Maximum total frames per batch
        balanced_sampling: Whether to balance across tasks
        **kwargs: Additional DataLoader arguments
        
    Returns:
        VLNDataLoader: Configured dataloader
    """
    # Select sampler
    sampler = None
    
    if sampler_type == "balanced" or balanced_sampling:
        try:
            sampler = BalancedTaskSampler(dataset, shuffle=shuffle)
            shuffle = False  # Disable shuffle when using custom sampler
        except Exception as e:
            logging.warning(f"Failed to create balanced sampler: {e}, falling back to default")
    
    elif sampler_type == "dynamic":
        try:
            sampler = DynamicBatchSampler(
                dataset,
                batch_size=batch_size,
                max_frames_per_batch=max_frames_per_batch,
                shuffle=shuffle
            )
            # Dynamic sampler returns batches, so we need batch_size=1 and batch_sampler
            kwargs['batch_sampler'] = sampler
            kwargs.pop('batch_size', None)
            kwargs.pop('shuffle', None)
            kwargs.pop('drop_last', None)
            
            return DataLoader(
                dataset,
                collate_fn=vln_collate_fn,
                num_workers=num_workers,
                **kwargs
            )
        except Exception as e:
            logging.warning(f"Failed to create dynamic sampler: {e}, falling back to default")
    
    # Create standard VLN dataloader
    return VLNDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        adaptive_batching=adaptive_batching,
        max_frames_per_batch=max_frames_per_batch,
        **kwargs
    )


def create_train_val_dataloaders(
    train_dataset: VLNDataset,
    val_dataset: VLNDataset,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> tuple[VLNDataLoader, VLNDataLoader]:
    """
    Create train and validation dataloaders with appropriate settings
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of worker processes
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Training dataloader with data augmentation and shuffling
    train_loader = create_vln_dataloader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        balanced_sampling=True,
        adaptive_batching=True,
        drop_last=True,
        **kwargs
    )
    
    # Validation dataloader with deterministic order
    val_loader = create_vln_dataloader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        adaptive_batching=False,
        drop_last=False,
        **kwargs
    )
    
    return train_loader, val_loader