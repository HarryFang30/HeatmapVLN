"""
Utility functions for Diffusion Action Head

Provides action normalization/unnormalization and statistics management.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import numpy as np
import torch
import json
from pathlib import Path


@dataclass
class ActionStats:
    """
    Statistics for action normalization.
    
    Actions are normalized to [-1, 1] range using min/max statistics.
    
    Attributes:
        min: Minimum values for each action dimension
        max: Maximum values for each action dimension
    """
    min: List[float] = field(default_factory=lambda: [-1.07, -1.05])
    max: List[float] = field(default_factory=lambda: [1.07, 1.03])
    
    @property
    def action_dim(self) -> int:
        return len(self.min)
    
    def to_dict(self) -> Dict[str, List[float]]:
        return {'min': self.min, 'max': self.max}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ActionStats':
        return cls(min=d['min'], max=d['max'])
    
    def save(self, path: Union[str, Path]):
        """Save statistics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ActionStats':
        """Load statistics from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def from_data(cls, actions: np.ndarray) -> 'ActionStats':
        """
        Compute statistics from action data.
        
        Args:
            actions: (N, action_dim) array of actions
            
        Returns:
            ActionStats with computed min/max
        """
        return cls(
            min=actions.min(axis=0).tolist(),
            max=actions.max(axis=0).tolist()
        )


def normalize_actions(
    actions: Union[np.ndarray, torch.Tensor],
    stats: ActionStats
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize actions to [-1, 1] range.
    
    Args:
        actions: (..., action_dim) actions to normalize
        stats: ActionStats with min/max values
        
    Returns:
        Normalized actions in [-1, 1] range
    """
    is_tensor = isinstance(actions, torch.Tensor)
    
    if is_tensor:
        device = actions.device
        dtype = actions.dtype
        min_val = torch.tensor(stats.min, device=device, dtype=dtype)
        max_val = torch.tensor(stats.max, device=device, dtype=dtype)
    else:
        min_val = np.array(stats.min)
        max_val = np.array(stats.max)
    
    # Normalize to [0, 1] then scale to [-1, 1]
    normalized = (actions - min_val) / (max_val - min_val + 1e-8)
    normalized = normalized * 2.0 - 1.0
    
    # Clip to [-1, 1]
    if is_tensor:
        normalized = torch.clamp(normalized, -1.0, 1.0)
    else:
        normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def unnormalize_actions(
    normalized_actions: Union[np.ndarray, torch.Tensor],
    stats: ActionStats
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unnormalize actions from [-1, 1] range to original scale.
    
    Args:
        normalized_actions: (..., action_dim) normalized actions in [-1, 1]
        stats: ActionStats with min/max values
        
    Returns:
        Actions in original scale
    """
    is_tensor = isinstance(normalized_actions, torch.Tensor)
    
    if is_tensor:
        device = normalized_actions.device
        dtype = normalized_actions.dtype
        min_val = torch.tensor(stats.min, device=device, dtype=dtype)
        max_val = torch.tensor(stats.max, device=device, dtype=dtype)
    else:
        min_val = np.array(stats.min)
        max_val = np.array(stats.max)
    
    # Scale from [-1, 1] to [0, 1] then to original range
    actions = (normalized_actions + 1.0) / 2.0
    actions = actions * (max_val - min_val) + min_val
    
    return actions


def cumsum_actions(
    delta_actions: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert delta actions to cumulative positions.
    
    Args:
        delta_actions: (B, horizon, action_dim) delta actions
        
    Returns:
        (B, horizon, action_dim) cumulative positions
    """
    if isinstance(delta_actions, torch.Tensor):
        return torch.cumsum(delta_actions, dim=1)
    else:
        return np.cumsum(delta_actions, axis=1)


def get_action_from_diffusion_output(
    diffusion_output: torch.Tensor,
    stats: ActionStats,
    cumulative: bool = True
) -> torch.Tensor:
    """
    Process diffusion model output to get final actions.
    
    Args:
        diffusion_output: (B, pred_horizon, action_dim) normalized delta actions
        stats: Action statistics for unnormalization
        cumulative: If True, return cumulative positions; else return deltas
        
    Returns:
        (B, pred_horizon, action_dim) actions
    """
    # Unnormalize
    actions = unnormalize_actions(diffusion_output, stats)
    
    # Convert to cumulative if requested
    if cumulative:
        actions = cumsum_actions(actions)
    
    return actions


# Default action statistics
# 🔧 扩大边界，覆盖更多动作场景（避免测试时超出范围）
DEFAULT_ACTION_STATS = ActionStats(
    min=[-0.5, -0.2],
    max=[0.5, 1.0]
)

