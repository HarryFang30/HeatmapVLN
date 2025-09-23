"""
Checkpoint management utilities for VLN Project
Handles saving, loading, and resuming training checkpoints
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Comprehensive checkpoint management for VLN training
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 keep_last_n: int = 5, keep_best: bool = True):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Whether to keep the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        
        # Track checkpoint history
        self.history_file = self.checkpoint_dir / "checkpoint_history.json"
        self.history = self._load_history()
        
        # Best checkpoint tracking
        self.best_metric = float('inf')
        self.best_checkpoint = None
    
    def save_checkpoint(self, state_dict: Dict[str, Any], 
                       epoch: int, 
                       metrics: Dict[str, float] = None,
                       is_best: bool = False,
                       filename: Optional[str] = None) -> str:
        """
        Save a training checkpoint
        
        Args:
            state_dict: Model and training state
            epoch: Current epoch
            metrics: Evaluation metrics
            is_best: Whether this is the best checkpoint
            filename: Optional custom filename
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch:04d}_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'state_dict': state_dict,
            'metrics': metrics or {},
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Update history
            self.history.append({
                'epoch': epoch,
                'filename': filename,
                'path': str(checkpoint_path),
                'timestamp': checkpoint_data['timestamp'],
                'metrics': metrics or {},
                'is_best': is_best
            })
            
            # Save best checkpoint if applicable
            if is_best or self._is_new_best(metrics):
                self._save_best_checkpoint(checkpoint_path)
                is_best = True
                self.history[-1]['is_best'] = True
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            # Save history
            self._save_history()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            load_best: Load the best checkpoint instead
            
        Returns:
            Loaded checkpoint data
        """
        if load_best:
            checkpoint_path = self._get_best_checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError("No best checkpoint found")
        
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
            # Validate checkpoint structure
            required_keys = ['epoch', 'timestamp', 'state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"Missing key in checkpoint: {key}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def resume_from_checkpoint(self, models: Dict[str, torch.nn.Module],
                             optimizers: Dict[str, torch.optim.Optimizer] = None,
                             schedulers: Dict[str, Any] = None,
                             checkpoint_path: Optional[str] = None) -> int:
        """
        Resume training from checkpoint
        
        Args:
            models: Dictionary of models to load
            optimizers: Dictionary of optimizers to load
            schedulers: Dictionary of schedulers to load
            checkpoint_path: Specific checkpoint to resume from
            
        Returns:
            Epoch to resume from
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Load model states
        for name, model in models.items():
            if name in checkpoint['state_dict']:
                try:
                    model.load_state_dict(checkpoint['state_dict'][name])
                    logger.info(f"Loaded {name} from checkpoint")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
        
        # Load optimizer states
        if optimizers:
            for name, optimizer in optimizers.items():
                opt_key = f"{name}_optimizer"
                if opt_key in checkpoint['state_dict']:
                    try:
                        optimizer.load_state_dict(checkpoint['state_dict'][opt_key])
                        logger.info(f"Loaded {name} optimizer from checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load {name} optimizer: {e}")
        
        # Load scheduler states
        if schedulers:
            for name, scheduler in schedulers.items():
                sched_key = f"{name}_scheduler"
                if sched_key in checkpoint['state_dict']:
                    try:
                        scheduler.load_state_dict(checkpoint['state_dict'][sched_key])
                        logger.info(f"Loaded {name} scheduler from checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load {name} scheduler: {e}")
        
        resume_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resuming training from epoch {resume_epoch}")
        
        return resume_epoch
    
    def get_checkpoint_info(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the full model
        
        Args:
            checkpoint_path: Path to checkpoint (latest if None)
            
        Returns:
            Checkpoint information
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint_path()
            if checkpoint_path is None:
                return {}
        
        try:
            # Load only metadata (not the full state dict)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'metrics': checkpoint.get('metrics', {}),
                'path': checkpoint_path
            }
            
            # Add state dict keys without loading full tensors
            if 'state_dict' in checkpoint:
                info['model_keys'] = list(checkpoint['state_dict'].keys())
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {}
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        for entry in self.history:
            checkpoint_path = Path(entry['path'])
            if checkpoint_path.exists():
                checkpoints.append(entry)
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load checkpoint history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint history: {e}")
        return []
    
    def _save_history(self):
        """Save checkpoint history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint history: {e}")
    
    def _is_new_best(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if current metrics represent a new best"""
        if not metrics or not self.keep_best:
            return False
        
        # Default metric for "best" - can be configured
        metric_key = 'val_loss'  # Lower is better
        if metric_key not in metrics:
            # Try alternative metrics
            for alt_key in ['spatial_accuracy', 'success_rate', 'overall_quality']:
                if alt_key in metrics:
                    metric_key = alt_key
                    current_metric = metrics[metric_key]
                    return current_metric > self.best_metric  # Higher is better
            return False
        
        current_metric = metrics[metric_key]
        if current_metric < self.best_metric:  # Lower loss is better
            self.best_metric = current_metric
            return True
        
        return False
    
    def _save_best_checkpoint(self, checkpoint_path: Path):
        """Save a copy as best checkpoint"""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        try:
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint = str(best_path)
            logger.info(f"Best checkpoint updated: {best_path}")
        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")
    
    def _get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to the latest checkpoint"""
        if not self.history:
            return None
        
        # Find the latest checkpoint that still exists
        for entry in reversed(self.history):
            checkpoint_path = Path(entry['path'])
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        return None
    
    def _get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to the best checkpoint"""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if best_path.exists():
            return str(best_path)
        
        # Fallback: find best in history
        best_entry = None
        for entry in self.history:
            if entry.get('is_best', False):
                checkpoint_path = Path(entry['path'])
                if checkpoint_path.exists():
                    best_entry = entry
                    break
        
        return best_entry['path'] if best_entry else None
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints keeping only the most recent N"""
        if len(self.history) <= self.keep_last_n:
            return
        
        # Sort by epoch
        sorted_history = sorted(self.history, key=lambda x: x['epoch'])
        
        # Keep the last N checkpoints and any marked as best
        to_keep = set()
        
        # Keep last N
        for entry in sorted_history[-self.keep_last_n:]:
            to_keep.add(entry['path'])
        
        # Keep best checkpoint
        if self.keep_best:
            for entry in sorted_history:
                if entry.get('is_best', False):
                    to_keep.add(entry['path'])
        
        # Remove old checkpoints
        to_remove = []
        for entry in sorted_history:
            if entry['path'] not in to_keep:
                checkpoint_path = Path(entry['path'])
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                        logger.info(f"Removed old checkpoint: {checkpoint_path}")
                        to_remove.append(entry)
                    except Exception as e:
                        logger.error(f"Failed to remove checkpoint {checkpoint_path}: {e}")
        
        # Update history
        for entry in to_remove:
            self.history.remove(entry)


def create_state_dict(models: Dict[str, torch.nn.Module],
                     optimizers: Dict[str, torch.optim.Optimizer] = None,
                     schedulers: Dict[str, Any] = None,
                     **kwargs) -> Dict[str, Any]:
    """
    Utility function to create a state dictionary for checkpointing
    
    Args:
        models: Dictionary of models
        optimizers: Dictionary of optimizers
        schedulers: Dictionary of schedulers
        **kwargs: Additional items to include in state dict
        
    Returns:
        Complete state dictionary
    """
    state_dict = {}
    
    # Add model states
    for name, model in models.items():
        if hasattr(model, 'module'):  # Handle DDP
            state_dict[name] = model.module.state_dict()
        else:
            state_dict[name] = model.state_dict()
    
    # Add optimizer states
    if optimizers:
        for name, optimizer in optimizers.items():
            state_dict[f"{name}_optimizer"] = optimizer.state_dict()
    
    # Add scheduler states
    if schedulers:
        for name, scheduler in schedulers.items():
            state_dict[f"{name}_scheduler"] = scheduler.state_dict()
    
    # Add any additional items
    state_dict.update(kwargs)
    
    return state_dict