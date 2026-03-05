"""
VLN Models Module
=================

This module provides a VLN pipeline that uses Qwen3.5 for video understanding.

Architecture:
```
Video Frames + Instruction
        |
    Qwen3.5 (Vision Encoder + LLM)
        |
    Hidden States Projection (4096 -> 1024)
        |
    Output Heads:
        - History Heatmap (Spatial-Semantic Fusion)
        - Action Head
```
"""

# === Qwen3.5 Integration ===
from .qwen3_5 import (
    Qwen3_5Integration,
    Qwen3_5Config,
)

# === VLN Pipeline ===
from .pipeline import (
    VLNPipeline,
    VLNPipelineConfig,
    create_vln_pipeline,
)

# === Heatmap Components (Diffusion-based) ===
from .heatmap import (
    DiffusionHeatmapHead,
    DiffusionHeatmapConfig,
    create_diffusion_heatmap_head,
)

# === Action Components ===
from .action import (
    DiffusionActionHead,
    DiffusionActionConfig,
    StopPredictionHead,
)

# === Other Components ===
try:
    from .mlp import MLP
except ImportError:
    MLP = None


__all__ = [
    # Qwen3.5 Integration
    'Qwen3_5Integration',
    'Qwen3_5Config',
    
    # VLN Pipeline
    'VLNPipeline',
    'VLNPipelineConfig',
    'create_vln_pipeline',
    
    # Heatmap Components
    'DiffusionHeatmapHead',
    'DiffusionHeatmapConfig',
    'create_diffusion_heatmap_head',
    
    # Action Components
    'DiffusionActionHead',
    'DiffusionActionConfig',
    'StopPredictionHead',
    
    # Other
    'MLP',
]
