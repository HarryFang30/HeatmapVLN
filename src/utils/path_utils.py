"""
Path utilities for robust file handling across different environments.
"""

import os
from pathlib import Path
from typing import Union, Optional


def get_project_root() -> Path:
    """
    Get the project root directory in a robust way.

    Returns:
        Path: The project root directory
    """
    # Start from this file and go up to find the project root
    current_file = Path(__file__).resolve()

    # Look for markers that indicate the project root
    root_markers = [
        'requirements.txt',
        'main.py',
        'configs',
        'src'
    ]

    current_dir = current_file.parent
    while current_dir != current_dir.parent:  # Not at filesystem root
        # Check if we have project markers
        if any((current_dir / marker).exists() for marker in root_markers):
            # Additional validation - ensure we have src and configs
            if (current_dir / 'src').exists() and (current_dir / 'configs').exists():
                return current_dir
        current_dir = current_dir.parent

    # Fallback: assume we're in the Project directory
    return Path.cwd()


def resolve_model_path(path: Union[str, Path], model_type: str = "model") -> Path:
    """
    Resolve model path to an absolute path, handling relative paths robustly.

    Args:
        path: The model path (can be relative or absolute)
        model_type: Type of model for better error messages

    Returns:
        Path: Resolved absolute path

    Raises:
        FileNotFoundError: If the resolved path doesn't exist
    """
    path = Path(path)

    # If already absolute and exists, return it
    if path.is_absolute():
        if path.exists():
            return path
        else:
            raise FileNotFoundError(f"{model_type} path does not exist: {path}")

    # For relative paths, resolve relative to project root
    project_root = get_project_root()
    resolved_path = project_root / path

    if resolved_path.exists():
        return resolved_path.resolve()
    else:
        # Try common alternative locations
        alternatives = [
            project_root / "models" / path.name,
            project_root / path.name,
            Path.home() / "VLN" / "Project" / path,  # Fallback to old structure
            Path.home() / "VLN" / path.name if path.name else None
        ]

        for alt_path in alternatives:
            if alt_path and alt_path.exists():
                print(f"Warning: Using alternative path for {model_type}: {alt_path}")
                return alt_path.resolve()

        raise FileNotFoundError(
            f"{model_type} not found at {resolved_path} or any alternative locations"
        )


def resolve_video_path(path: Union[str, Path]) -> Path:
    """
    Resolve video path robustly, checking multiple common locations.

    Args:
        path: The video path (can be relative or absolute)

    Returns:
        Path: Resolved absolute path

    Raises:
        FileNotFoundError: If the video cannot be found
    """
    path = Path(path)

    # If already absolute and exists, return it
    if path.is_absolute():
        if path.exists():
            return path
        else:
            raise FileNotFoundError(f"Video file does not exist: {path}")

    # For relative paths, check multiple locations
    search_locations = [
        Path.cwd() / path,  # Current directory
        get_project_root() / path,  # Project root
        get_project_root() / "test_data" / path,  # Test data directory
        Path.home() / "VLN" / path,  # Legacy VLN directory
        Path.home() / "VLN" / "vggt" / "examples" / "videos" / path.name,  # VGGT examples
    ]

    for location in search_locations:
        if location.exists():
            return location.resolve()

    raise FileNotFoundError(
        f"Video file '{path}' not found in any of the search locations: "
        f"{[str(loc) for loc in search_locations]}"
    )


def get_config_path(config_name: str = "default_config.yaml") -> Path:
    """
    Get the path to a configuration file.

    Args:
        config_name: Name of the config file

    Returns:
        Path: Path to the config file
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path


def ensure_output_dir(output_path: Union[str, Path]) -> Path:
    """
    Ensure an output directory exists, creating it if necessary.

    Args:
        output_path: Path to the output directory

    Returns:
        Path: Resolved path to the output directory
    """
    output_path = Path(output_path)

    # If relative, make it relative to project root
    if not output_path.is_absolute():
        output_path = get_project_root() / output_path

    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path.resolve()


def get_default_model_paths() -> dict:
    """
    Get default model paths relative to project root.

    Returns:
        dict: Dictionary of model paths
    """
    return {
        'dinov3': './models/dinov3',
        'vggt': './models/vggt',
        'llm': './models/qwen_2.5_vl'
    }


def validate_environment() -> dict:
    """
    Validate the current environment and return status information.

    Returns:
        dict: Environment validation results
    """
    project_root = get_project_root()

    status = {
        'project_root': str(project_root),
        'project_root_exists': project_root.exists(),
        'required_dirs': {},
        'model_paths': {},
        'errors': []
    }

    # Check required directories
    required_dirs = ['src', 'configs', 'models']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        status['required_dirs'][dir_name] = {
            'path': str(dir_path),
            'exists': dir_path.exists()
        }
        if not dir_path.exists():
            status['errors'].append(f"Required directory missing: {dir_path}")

    # Check model paths
    default_paths = get_default_model_paths()
    for model_name, rel_path in default_paths.items():
        try:
            resolved_path = resolve_model_path(rel_path, model_name)
            status['model_paths'][model_name] = {
                'configured_path': rel_path,
                'resolved_path': str(resolved_path),
                'exists': True
            }
        except FileNotFoundError as e:
            status['model_paths'][model_name] = {
                'configured_path': rel_path,
                'resolved_path': None,
                'exists': False,
                'error': str(e)
            }
            status['errors'].append(f"Model path issue for {model_name}: {e}")

    return status