"""
Algorithm Registry and Selection System
======================================

This module provides a centralized registry for different frame sampling algorithms
with flexible configuration and runtime selection capabilities.

Features:
- Algorithm registration and discovery
- Configuration-driven algorithm selection
- Runtime algorithm switching
- Performance benchmarking
- Extensible architecture for new algorithms

Supported Algorithms:
1. Original Greedy Coverage (baseline)
2. Enhanced Multi-Objective Submodular
3. Hybrid approaches
4. Custom algorithms (extensible)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Type, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import yaml
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Enumeration of available algorithm types."""
    GREEDY_COVERAGE = "greedy_coverage"
    ENHANCED_SUBMODULAR = "enhanced_submodular"
    HYBRID_COVERAGE_DIVERSITY = "hybrid_coverage_diversity"
    TEMPORAL_FIRST = "temporal_first"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    CUSTOM = "custom"


@dataclass
class AlgorithmConfig:
    """Base configuration for all algorithms."""
    algorithm_type: AlgorithmType
    target_frames: int = 16
    candidate_frames: int = 128
    device: str = "cuda"

    # Performance settings
    timeout_seconds: Optional[float] = None
    memory_limit_gb: Optional[float] = None

    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    name: str = ""
    description: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_type': self.algorithm_type.value,
            'target_frames': self.target_frames,
            'candidate_frames': self.candidate_frames,
            'device': self.device,
            'timeout_seconds': self.timeout_seconds,
            'memory_limit_gb': self.memory_limit_gb,
            'algorithm_params': self.algorithm_params,
            'name': self.name,
            'description': self.description,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """Create from dictionary."""
        algorithm_type = AlgorithmType(data['algorithm_type'])
        return cls(
            algorithm_type=algorithm_type,
            target_frames=data.get('target_frames', 16),
            candidate_frames=data.get('candidate_frames', 128),
            device=data.get('device', 'cuda'),
            timeout_seconds=data.get('timeout_seconds'),
            memory_limit_gb=data.get('memory_limit_gb'),
            algorithm_params=data.get('algorithm_params', {}),
            name=data.get('name', ''),
            description=data.get('description', ''),
            version=data.get('version', '1.0')
        )


class BaseFrameSampler(nn.Module, ABC):
    """Abstract base class for all frame sampling algorithms."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._performance_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'last_call_time': 0.0
        }

    @abstractmethod
    def _core_sampling(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        visual_features: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Core sampling logic to be implemented by subclasses."""
        pass

    def forward(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        visual_features: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Main forward method with performance tracking."""

        start_time = time.time()

        try:
            # Timeout handling
            if self.config.timeout_seconds:
                # In real implementation, would use threading/async
                pass

            # Core sampling
            result = self._core_sampling(
                vggt_predictions, visual_features, frame_indices, **kwargs
            )

            # Add algorithm metadata
            result['algorithm_metadata'] = {
                'algorithm_type': self.config.algorithm_type.value,
                'algorithm_name': self.config.name,
                'version': self.config.version,
                'config': self.config.to_dict()
            }

            return result

        finally:
            # Update performance stats
            call_time = time.time() - start_time
            self._update_performance_stats(call_time)

    def _update_performance_stats(self, call_time: float):
        """Update performance statistics."""
        self._performance_stats['total_calls'] += 1
        self._performance_stats['total_time'] += call_time
        self._performance_stats['last_call_time'] = call_time
        self._performance_stats['average_time'] = (
            self._performance_stats['total_time'] / self._performance_stats['total_calls']
        )

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self._performance_stats.copy()

    def reset_performance_stats(self):
        """Reset performance tracking."""
        self._performance_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'last_call_time': 0.0
        }


class AlgorithmRegistry:
    """
    Central registry for frame sampling algorithms.

    Provides registration, discovery, and instantiation of algorithms
    with flexible configuration management.
    """

    def __init__(self):
        self._algorithms: Dict[AlgorithmType, Type[BaseFrameSampler]] = {}
        self._configs: Dict[str, AlgorithmConfig] = {}
        self._instances: Dict[str, BaseFrameSampler] = {}

        # Register built-in algorithms
        self._register_builtin_algorithms()

    def register_algorithm(
        self,
        algorithm_type: AlgorithmType,
        algorithm_class: Type[BaseFrameSampler],
        default_config: Optional[AlgorithmConfig] = None
    ):
        """Register a new algorithm."""

        if not issubclass(algorithm_class, BaseFrameSampler):
            raise ValueError(f"Algorithm class must inherit from BaseFrameSampler")

        self._algorithms[algorithm_type] = algorithm_class

        if default_config is not None:
            config_name = f"{algorithm_type.value}_default"
            self._configs[config_name] = default_config

        logger.info(f"Registered algorithm: {algorithm_type.value}")

    def get_available_algorithms(self) -> List[AlgorithmType]:
        """Get list of available algorithm types."""
        return list(self._algorithms.keys())

    def get_algorithm_class(self, algorithm_type: AlgorithmType) -> Type[BaseFrameSampler]:
        """Get algorithm class by type."""
        if algorithm_type not in self._algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        return self._algorithms[algorithm_type]

    def create_algorithm(
        self,
        algorithm_type: AlgorithmType,
        config: Optional[AlgorithmConfig] = None,
        instance_name: Optional[str] = None
    ) -> BaseFrameSampler:
        """Create algorithm instance."""

        if algorithm_type not in self._algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

        if config is None:
            # Use default config
            default_config_name = f"{algorithm_type.value}_default"
            if default_config_name in self._configs:
                config = self._configs[default_config_name]
            else:
                config = AlgorithmConfig(algorithm_type=algorithm_type)

        algorithm_class = self._algorithms[algorithm_type]
        instance = algorithm_class(config)

        # Cache instance if name provided
        if instance_name:
            self._instances[instance_name] = instance

        return instance

    def get_algorithm_instance(self, instance_name: str) -> BaseFrameSampler:
        """Get cached algorithm instance."""
        if instance_name not in self._instances:
            raise ValueError(f"No algorithm instance named: {instance_name}")
        return self._instances[instance_name]

    def register_config(self, config_name: str, config: AlgorithmConfig):
        """Register a configuration."""
        self._configs[config_name] = config
        logger.info(f"Registered config: {config_name}")

    def get_config(self, config_name: str) -> AlgorithmConfig:
        """Get configuration by name."""
        if config_name not in self._configs:
            raise ValueError(f"Unknown config: {config_name}")
        return self._configs[config_name]

    def get_available_configs(self) -> List[str]:
        """Get list of available configuration names."""
        return list(self._configs.keys())

    def save_config(self, config: AlgorithmConfig, filepath: Path):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)

    def load_config(self, filepath: Path) -> AlgorithmConfig:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return AlgorithmConfig.from_dict(data)

    def _register_builtin_algorithms(self):
        """Register built-in algorithms."""

        # Import here to avoid circular imports
        from .frame_sampler import SpaceAwareFrameSampler, SamplingConfig
        from .enhanced_frame_sampler import EnhancedFrameSampler, EnhancedSamplingConfig

        # Wrapper classes for built-in algorithms
        class GreedyCoverageSampler(BaseFrameSampler):
            def __init__(self, config: AlgorithmConfig):
                super().__init__(config)

                # Filter parameters for SamplingConfig
                valid_params = {}
                sampling_config_params = ['voxel_lambda', 'confidence_threshold', 'confidence_percentile', 'early_termination']
                for param in sampling_config_params:
                    if param in config.algorithm_params:
                        valid_params[param] = config.algorithm_params[param]

                sampling_config = SamplingConfig(
                    target_frames=config.target_frames,
                    candidate_frames=config.candidate_frames,
                    device=config.device,
                    **valid_params
                )
                self.sampler = SpaceAwareFrameSampler(sampling_config)

            def _core_sampling(self, vggt_predictions, visual_features=None, frame_indices=None, **kwargs):
                return self.sampler(vggt_predictions, frame_indices)

        class EnhancedSubmodularSampler(BaseFrameSampler):
            def __init__(self, config: AlgorithmConfig):
                super().__init__(config)

                # Filter parameters for EnhancedSamplingConfig
                valid_params = {}
                enhanced_config_params = [
                    'coverage_weight', 'diversity_weight', 'temporal_weight', 'uncertainty_weight',
                    'voxel_scales', 'adaptive_voxelization', 'use_visual_features', 'feature_diversity_threshold',
                    'temporal_smoothness', 'min_temporal_gap', 'confidence_threshold', 'confidence_percentile',
                    'uncertainty_boost', 'motion_importance', 'depth_variation_importance',
                    'optimization_method', 'max_iterations', 'convergence_threshold'
                ]
                for param in enhanced_config_params:
                    if param in config.algorithm_params:
                        valid_params[param] = config.algorithm_params[param]

                enhanced_config = EnhancedSamplingConfig(
                    target_frames=config.target_frames,
                    candidate_frames=config.candidate_frames,
                    device=config.device,
                    **valid_params
                )
                self.sampler = EnhancedFrameSampler(enhanced_config)

            def _core_sampling(self, vggt_predictions, visual_features=None, frame_indices=None, **kwargs):
                return self.sampler(vggt_predictions, visual_features, frame_indices)

        # Register algorithms
        self.register_algorithm(
            AlgorithmType.GREEDY_COVERAGE,
            GreedyCoverageSampler,
            AlgorithmConfig(
                algorithm_type=AlgorithmType.GREEDY_COVERAGE,
                name="Original Greedy Coverage",
                description="Original greedy maximum coverage algorithm",
                algorithm_params={
                    'voxel_lambda': 20.0,
                    'confidence_threshold': 0.1,
                    'confidence_percentile': 50.0
                }
            )
        )

        self.register_algorithm(
            AlgorithmType.ENHANCED_SUBMODULAR,
            EnhancedSubmodularSampler,
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ENHANCED_SUBMODULAR,
                name="Enhanced Multi-Objective Submodular",
                description="Advanced multi-objective submodular optimization",
                algorithm_params={
                    'coverage_weight': 0.4,
                    'diversity_weight': 0.3,
                    'temporal_weight': 0.2,
                    'uncertainty_weight': 0.1,
                    'optimization_method': 'submodular'
                }
            )
        )


# Global registry instance
_global_registry = AlgorithmRegistry()


def get_registry() -> AlgorithmRegistry:
    """Get the global algorithm registry."""
    return _global_registry


def register_algorithm(
    algorithm_type: AlgorithmType,
    algorithm_class: Type[BaseFrameSampler],
    default_config: Optional[AlgorithmConfig] = None
):
    """Register algorithm in global registry."""
    _global_registry.register_algorithm(algorithm_type, algorithm_class, default_config)


def create_algorithm(
    algorithm_type: Union[AlgorithmType, str],
    config: Optional[AlgorithmConfig] = None,
    instance_name: Optional[str] = None
) -> BaseFrameSampler:
    """Create algorithm instance from global registry."""

    if isinstance(algorithm_type, str):
        algorithm_type = AlgorithmType(algorithm_type)

    return _global_registry.create_algorithm(algorithm_type, config, instance_name)


def get_available_algorithms() -> List[str]:
    """Get available algorithm names."""
    return [alg.value for alg in _global_registry.get_available_algorithms()]


# Predefined configurations
def create_predefined_configs():
    """Create predefined algorithm configurations."""

    registry = get_registry()

    # Fast greedy configuration
    fast_greedy = AlgorithmConfig(
        algorithm_type=AlgorithmType.GREEDY_COVERAGE,
        target_frames=8,
        candidate_frames=32,
        name="Fast Greedy",
        description="Fast greedy coverage for quick testing",
        algorithm_params={
            'voxel_lambda': 30.0,  # Coarser voxels for speed
            'early_termination': True
        }
    )
    registry.register_config("fast_greedy", fast_greedy)

    # High quality submodular configuration
    high_quality = AlgorithmConfig(
        algorithm_type=AlgorithmType.ENHANCED_SUBMODULAR,
        target_frames=16,
        candidate_frames=128,
        name="High Quality Submodular",
        description="High quality multi-objective optimization",
        algorithm_params={
            'coverage_weight': 0.3,
            'diversity_weight': 0.4,  # Higher diversity emphasis
            'temporal_weight': 0.2,
            'uncertainty_weight': 0.1,
            'optimization_method': 'submodular',
            'voxel_scales': [5.0, 15.0, 30.0],  # Finer scales
            'use_visual_features': True
        }
    )
    registry.register_config("high_quality", high_quality)

    # Balanced configuration
    balanced = AlgorithmConfig(
        algorithm_type=AlgorithmType.ENHANCED_SUBMODULAR,
        target_frames=12,
        candidate_frames=64,
        name="Balanced",
        description="Balanced performance and quality",
        algorithm_params={
            'coverage_weight': 0.4,
            'diversity_weight': 0.3,
            'temporal_weight': 0.2,
            'uncertainty_weight': 0.1,
            'optimization_method': 'hybrid'
        }
    )
    registry.register_config("balanced", balanced)

    # Temporal-focused configuration
    temporal_focus = AlgorithmConfig(
        algorithm_type=AlgorithmType.ENHANCED_SUBMODULAR,
        target_frames=16,
        candidate_frames=128,
        name="Temporal Focus",
        description="Emphasizes temporal coherence",
        algorithm_params={
            'coverage_weight': 0.2,
            'diversity_weight': 0.2,
            'temporal_weight': 0.5,  # High temporal emphasis
            'uncertainty_weight': 0.1,
            'temporal_smoothness': 0.9,
            'min_temporal_gap': 3
        }
    )
    registry.register_config("temporal_focus", temporal_focus)


# Initialize predefined configurations
create_predefined_configs()


# Example usage and testing
if __name__ == "__main__":
    # Test algorithm registry
    registry = get_registry()

    print("Available algorithms:")
    for alg_type in registry.get_available_algorithms():
        print(f"  - {alg_type.value}")

    print("\nAvailable configurations:")
    for config_name in registry.get_available_configs():
        print(f"  - {config_name}")

    # Create algorithm instances
    print("\nCreating algorithm instances...")

    greedy_alg = create_algorithm("greedy_coverage")
    enhanced_alg = create_algorithm("enhanced_submodular")

    print(f"Greedy algorithm: {type(greedy_alg).__name__}")
    print(f"Enhanced algorithm: {type(enhanced_alg).__name__}")

    # Test configuration loading
    balanced_config = registry.get_config("balanced")
    balanced_alg = create_algorithm(AlgorithmType.ENHANCED_SUBMODULAR, balanced_config)

    print(f"Balanced algorithm config: {balanced_config.name}")

    print("Algorithm registry test completed successfully!")