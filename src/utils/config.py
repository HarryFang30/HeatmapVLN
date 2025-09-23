"""
Configuration utilities for VLN Project
Based on BridgeVLA configuration patterns
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easy access
    config = OmegaConf.create(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = [
        'dinov3', 'vggt', 'llm', 'video', 'frame_sampling',
        'training', 'heatmap', 'system', 'logging'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate critical parameters
    if config.video.total_frames < config.video.keyframes:
        raise ValueError("total_frames must be >= keyframes")
    
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
        
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = OmegaConf.merge(base_config, override_config)
    return merged