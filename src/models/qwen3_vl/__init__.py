"""
Qwen3-VL Integration for VLN Pipeline
======================================

This module provides Qwen3-VL model integration for the simplified VLN pipeline.
It replaces VGGT and DINOv3 with Qwen3-VL's native vision encoder.

Key Components:
- Qwen3VLIntegration: Main integration class for video processing
- Qwen3VLConfig: Configuration dataclass
"""

from .integration import Qwen3VLIntegration, Qwen3VLConfig

__all__ = [
    "Qwen3VLIntegration",
    "Qwen3VLConfig",
]

