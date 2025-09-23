"""
Configuration-Driven Algorithm Factory
======================================

This module provides a high-level factory for creating and managing frame sampling
algorithms with configuration-driven initialization and runtime management.

Features:
- YAML/JSON configuration file support
- Environment variable configuration
- Runtime algorithm switching
- Batch algorithm creation
- Configuration validation
- Auto-configuration based on hardware/constraints
"""

import torch
import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from .algorithm_registry import (
    AlgorithmRegistry, AlgorithmConfig, AlgorithmType, BaseFrameSampler,
    get_registry, create_algorithm
)

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source types."""
    FILE_YAML = "yaml"
    FILE_JSON = "json"
    ENVIRONMENT = "env"
    DICT = "dict"
    AUTO = "auto"


@dataclass
class HardwareProfile:
    """Hardware profile for auto-configuration."""
    gpu_memory_gb: float
    cpu_cores: int
    total_memory_gb: float
    gpu_count: int = 1
    gpu_type: str = "unknown"

    @classmethod
    def detect_hardware(cls) -> 'HardwareProfile':
        """Auto-detect hardware profile."""
        gpu_memory_gb = 0.0
        gpu_count = 0
        gpu_type = "cpu"

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_type = torch.cuda.get_device_name(0)

        cpu_cores = os.cpu_count() or 1

        # Estimate total memory (simplified)
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_memory_gb = 8.0  # Default assumption

        return cls(
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            total_memory_gb=total_memory_gb,
            gpu_count=gpu_count,
            gpu_type=gpu_type
        )


class AlgorithmFactory:
    """
    Configuration-driven factory for creating frame sampling algorithms.

    Supports multiple configuration sources and provides intelligent
    auto-configuration based on hardware and performance constraints.
    """

    def __init__(self, registry: Optional[AlgorithmRegistry] = None):
        self.registry = registry or get_registry()
        self.hardware_profile = HardwareProfile.detect_hardware()
        self._config_cache: Dict[str, AlgorithmConfig] = {}
        self._instance_cache: Dict[str, BaseFrameSampler] = {}

        logger.info(f"AlgorithmFactory initialized with hardware: "
                   f"{self.hardware_profile.gpu_count} GPUs, "
                   f"{self.hardware_profile.gpu_memory_gb:.1f}GB GPU memory")

    def create_from_config(
        self,
        config_source: Union[str, Path, Dict, AlgorithmConfig],
        source_type: ConfigSource = ConfigSource.AUTO,
        instance_name: Optional[str] = None,
        cache_instance: bool = True
    ) -> BaseFrameSampler:
        """
        Create algorithm from various configuration sources.

        Args:
            config_source: Configuration file path, dict, or AlgorithmConfig
            source_type: Type of configuration source
            instance_name: Name for caching the instance
            cache_instance: Whether to cache the created instance

        Returns:
            Configured algorithm instance
        """

        # Load configuration
        if isinstance(config_source, AlgorithmConfig):
            config = config_source
        else:
            config = self._load_config(config_source, source_type)

        # Apply hardware optimizations
        config = self._optimize_config_for_hardware(config)

        # Create algorithm instance
        instance = self.registry.create_algorithm(
            config.algorithm_type, config, instance_name
        )

        # Cache if requested
        if cache_instance and instance_name:
            self._instance_cache[instance_name] = instance

        logger.info(f"Created algorithm: {config.algorithm_type.value} "
                   f"({config.name}) -> {instance_name or 'unnamed'}")

        return instance

    def create_from_preset(
        self,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None
    ) -> BaseFrameSampler:
        """Create algorithm from predefined preset configuration."""

        if preset_name not in self.registry.get_available_configs():
            raise ValueError(f"Unknown preset: {preset_name}")

        config = self.registry.get_config(preset_name)

        # Apply overrides
        if overrides:
            config = self._apply_config_overrides(config, overrides)

        return self.create_from_config(config, instance_name=instance_name)

    def create_auto_configured(
        self,
        performance_target: str = "balanced",
        constraints: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None
    ) -> BaseFrameSampler:
        """
        Create auto-configured algorithm based on hardware and constraints.

        Args:
            performance_target: "fast", "balanced", "quality"
            constraints: Optional constraints (memory_limit, time_limit, etc.)
            instance_name: Name for the created instance

        Returns:
            Auto-configured algorithm instance
        """

        config = self._auto_configure(performance_target, constraints)
        return self.create_from_config(config, instance_name=instance_name)

    def create_batch(
        self,
        config_specs: List[Dict[str, Any]],
        prefix: str = "algo"
    ) -> Dict[str, BaseFrameSampler]:
        """
        Create multiple algorithms from specifications.

        Args:
            config_specs: List of configuration specifications
            prefix: Prefix for instance names

        Returns:
            Dictionary of created algorithm instances
        """

        instances = {}

        for i, spec in enumerate(config_specs):
            instance_name = f"{prefix}_{i}"

            if 'preset' in spec:
                # Create from preset
                instance = self.create_from_preset(
                    spec['preset'],
                    spec.get('overrides'),
                    instance_name
                )
            elif 'config' in spec:
                # Create from config
                instance = self.create_from_config(
                    spec['config'],
                    ConfigSource.DICT,
                    instance_name
                )
            elif 'auto_config' in spec:
                # Create auto-configured
                instance = self.create_auto_configured(
                    spec['auto_config'],
                    spec.get('constraints'),
                    instance_name
                )
            else:
                raise ValueError(f"Invalid config spec: {spec}")

            instances[instance_name] = instance

        logger.info(f"Created batch of {len(instances)} algorithms")
        return instances

    def get_cached_instance(self, instance_name: str) -> BaseFrameSampler:
        """Get cached algorithm instance."""
        if instance_name not in self._instance_cache:
            raise ValueError(f"No cached instance: {instance_name}")
        return self._instance_cache[instance_name]

    def list_cached_instances(self) -> List[str]:
        """List names of cached algorithm instances."""
        return list(self._instance_cache.keys())

    def clear_cache(self):
        """Clear instance cache."""
        self._instance_cache.clear()
        self._config_cache.clear()
        logger.info("Algorithm factory cache cleared")

    def save_config_template(self, filepath: Path, algorithm_type: AlgorithmType):
        """Save configuration template for an algorithm type."""

        # Get default config for algorithm type
        default_config_name = f"{algorithm_type.value}_default"
        if default_config_name in self.registry.get_available_configs():
            config = self.registry.get_config(default_config_name)
        else:
            config = AlgorithmConfig(algorithm_type=algorithm_type)

        # Save as YAML template
        template_data = config.to_dict()
        template_data['_template_info'] = {
            'algorithm_type': algorithm_type.value,
            'description': f'Configuration template for {algorithm_type.value}',
            'usage': 'Modify the parameters below and save as your custom config'
        }

        with open(filepath, 'w') as f:
            yaml.dump(template_data, f, default_flow_style=False)

        logger.info(f"Saved config template: {filepath}")

    def _load_config(
        self,
        config_source: Union[str, Path, Dict],
        source_type: ConfigSource
    ) -> AlgorithmConfig:
        """Load configuration from various sources."""

        # Auto-detect source type
        if source_type == ConfigSource.AUTO:
            if isinstance(config_source, dict):
                source_type = ConfigSource.DICT
            elif isinstance(config_source, (str, Path)):
                path = Path(config_source)
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    source_type = ConfigSource.FILE_YAML
                elif path.suffix.lower() == '.json':
                    source_type = ConfigSource.FILE_JSON
                else:
                    source_type = ConfigSource.ENVIRONMENT
            else:
                raise ValueError(f"Cannot auto-detect config source type for: {config_source}")

        # Load based on source type
        if source_type == ConfigSource.DICT:
            return AlgorithmConfig.from_dict(config_source)

        elif source_type == ConfigSource.FILE_YAML:
            cache_key = f"yaml:{config_source}"
            if cache_key not in self._config_cache:
                with open(config_source, 'r') as f:
                    data = yaml.safe_load(f)
                self._config_cache[cache_key] = AlgorithmConfig.from_dict(data)
            return self._config_cache[cache_key]

        elif source_type == ConfigSource.FILE_JSON:
            cache_key = f"json:{config_source}"
            if cache_key not in self._config_cache:
                with open(config_source, 'r') as f:
                    data = json.load(f)
                self._config_cache[cache_key] = AlgorithmConfig.from_dict(data)
            return self._config_cache[cache_key]

        elif source_type == ConfigSource.ENVIRONMENT:
            return self._load_config_from_env(str(config_source))

        else:
            raise ValueError(f"Unsupported config source type: {source_type}")

    def _load_config_from_env(self, env_prefix: str) -> AlgorithmConfig:
        """Load configuration from environment variables."""

        env_data = {}

        # Standard environment variables
        env_mappings = {
            f'{env_prefix}_ALGORITHM_TYPE': 'algorithm_type',
            f'{env_prefix}_TARGET_FRAMES': 'target_frames',
            f'{env_prefix}_CANDIDATE_FRAMES': 'candidate_frames',
            f'{env_prefix}_DEVICE': 'device',
            f'{env_prefix}_NAME': 'name',
            f'{env_prefix}_DESCRIPTION': 'description'
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Type conversion
                if config_key in ['target_frames', 'candidate_frames']:
                    value = int(value)
                elif config_key == 'algorithm_type':
                    value = AlgorithmType(value)

                env_data[config_key] = value

        # Algorithm-specific parameters
        algorithm_params = {}
        param_prefix = f'{env_prefix}_PARAM_'

        for env_var, value in os.environ.items():
            if env_var.startswith(param_prefix):
                param_name = env_var[len(param_prefix):].lower()
                # Try to convert to appropriate type
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                algorithm_params[param_name] = value

        if algorithm_params:
            env_data['algorithm_params'] = algorithm_params

        return AlgorithmConfig.from_dict(env_data)

    def _apply_config_overrides(
        self,
        config: AlgorithmConfig,
        overrides: Dict[str, Any]
    ) -> AlgorithmConfig:
        """Apply overrides to configuration."""

        config_dict = config.to_dict()

        # Apply direct overrides
        for key, value in overrides.items():
            if key == 'algorithm_params':
                # Merge algorithm parameters
                config_dict['algorithm_params'].update(value)
            else:
                config_dict[key] = value

        return AlgorithmConfig.from_dict(config_dict)

    def _optimize_config_for_hardware(self, config: AlgorithmConfig) -> AlgorithmConfig:
        """Optimize configuration based on hardware profile."""

        optimized_params = config.algorithm_params.copy()

        # GPU memory-based optimizations
        if self.hardware_profile.gpu_memory_gb < 8:
            # Low memory optimizations
            optimized_params['voxel_lambda'] = optimized_params.get('voxel_lambda', 20.0) * 1.5
            config.candidate_frames = min(config.candidate_frames, 64)
            config.device = "cpu"  # Force CPU for very low memory
        elif self.hardware_profile.gpu_memory_gb > 24:
            # High memory optimizations
            optimized_params['voxel_scales'] = optimized_params.get('voxel_scales', [5.0, 15.0, 45.0])

        # CPU vs GPU device selection
        if self.hardware_profile.gpu_count == 0:
            config.device = "cpu"
        elif config.device == "cuda" and self.hardware_profile.gpu_memory_gb < 4:
            config.device = "cpu"

        # Update config
        config.algorithm_params = optimized_params

        return config

    def _auto_configure(
        self,
        performance_target: str,
        constraints: Optional[Dict[str, Any]]
    ) -> AlgorithmConfig:
        """Auto-configure algorithm based on target and constraints."""

        constraints = constraints or {}

        # Base configuration based on performance target
        if performance_target == "fast":
            algorithm_type = AlgorithmType.GREEDY_COVERAGE
            base_params = {
                'voxel_lambda': 30.0,
                'early_termination': True
            }
            target_frames = 8
            candidate_frames = 32

        elif performance_target == "quality":
            algorithm_type = AlgorithmType.ENHANCED_SUBMODULAR
            base_params = {
                'coverage_weight': 0.3,
                'diversity_weight': 0.4,
                'temporal_weight': 0.2,
                'uncertainty_weight': 0.1,
                'optimization_method': 'submodular',
                'voxel_scales': [5.0, 15.0, 30.0],
                'use_visual_features': True
            }
            target_frames = 16
            candidate_frames = 128

        else:  # balanced
            algorithm_type = AlgorithmType.ENHANCED_SUBMODULAR
            base_params = {
                'coverage_weight': 0.4,
                'diversity_weight': 0.3,
                'temporal_weight': 0.2,
                'uncertainty_weight': 0.1,
                'optimization_method': 'hybrid'
            }
            target_frames = 12
            candidate_frames = 64

        # Apply constraints
        if 'memory_limit_gb' in constraints:
            memory_limit = constraints['memory_limit_gb']
            if memory_limit < 4:
                algorithm_type = AlgorithmType.GREEDY_COVERAGE
                candidate_frames = min(candidate_frames, 32)

        if 'time_limit_seconds' in constraints:
            time_limit = constraints['time_limit_seconds']
            if time_limit < 1.0:
                algorithm_type = AlgorithmType.GREEDY_COVERAGE
                base_params['early_termination'] = True

        # Device selection
        device = "cuda" if self.hardware_profile.gpu_count > 0 else "cpu"
        if constraints.get('force_cpu', False):
            device = "cpu"

        config = AlgorithmConfig(
            algorithm_type=algorithm_type,
            target_frames=target_frames,
            candidate_frames=candidate_frames,
            device=device,
            algorithm_params=base_params,
            name=f"Auto-configured {performance_target}",
            description=f"Auto-configured for {performance_target} performance"
        )

        return config


# Factory functions for common use cases
def create_fast_algorithm(
    target_frames: int = 8,
    candidate_frames: int = 32,
    instance_name: Optional[str] = None
) -> BaseFrameSampler:
    """Create fast algorithm for quick testing."""
    factory = AlgorithmFactory()
    return factory.create_auto_configured("fast", instance_name=instance_name)


def create_quality_algorithm(
    target_frames: int = 16,
    candidate_frames: int = 128,
    instance_name: Optional[str] = None
) -> BaseFrameSampler:
    """Create high-quality algorithm."""
    factory = AlgorithmFactory()
    return factory.create_auto_configured("quality", instance_name=instance_name)


def create_from_config_file(
    config_path: Union[str, Path],
    instance_name: Optional[str] = None
) -> BaseFrameSampler:
    """Create algorithm from configuration file."""
    factory = AlgorithmFactory()
    return factory.create_from_config(config_path, instance_name=instance_name)


# Global factory instance
_global_factory = AlgorithmFactory()


def get_factory() -> AlgorithmFactory:
    """Get global algorithm factory."""
    return _global_factory


# Example usage and testing
if __name__ == "__main__":
    # Test algorithm factory
    factory = AlgorithmFactory()

    print("Hardware profile:")
    print(f"  GPU count: {factory.hardware_profile.gpu_count}")
    print(f"  GPU memory: {factory.hardware_profile.gpu_memory_gb:.1f}GB")
    print(f"  CPU cores: {factory.hardware_profile.cpu_cores}")

    # Test auto-configuration
    print("\nTesting auto-configuration:")
    fast_algo = factory.create_auto_configured("fast", instance_name="fast_test")
    quality_algo = factory.create_auto_configured("quality", instance_name="quality_test")

    print(f"Fast algorithm: {fast_algo.config.algorithm_type.value}")
    print(f"Quality algorithm: {quality_algo.config.algorithm_type.value}")

    # Test preset creation
    print("\nTesting preset creation:")
    balanced_algo = factory.create_from_preset("balanced", instance_name="balanced_test")
    print(f"Balanced algorithm: {balanced_algo.config.name}")

    # Test batch creation
    print("\nTesting batch creation:")
    batch_specs = [
        {'preset': 'fast_greedy'},
        {'preset': 'balanced', 'overrides': {'target_frames': 10}},
        {'auto_config': 'quality'}
    ]
    batch_algos = factory.create_batch(batch_specs)
    print(f"Created batch: {list(batch_algos.keys())}")

    print("\nAlgorithm factory test completed successfully!")