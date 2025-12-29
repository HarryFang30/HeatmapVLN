"""
VLN Models Module
=================

This module provides a VLN pipeline that uses Qwen3-VL for video understanding.

Architecture:
```
Video Frames + Instruction
        ↓
    Qwen3-VL (Vision Encoder + LLM)
        ↓
    Hidden States Projection (2048 → 1024)
        ↓
    Output Heads:
        - History Heatmap (Diffusion)
        - Future Heatmap (Diffusion)
        - Action Head (Diffusion Policy)
        - Stop Head (Binary Classifier)
```
"""

# === Qwen3-VL Integration ===
from .qwen3_vl import (
    Qwen3VLIntegration,
    Qwen3VLConfig,
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
    # Qwen3-VL Integration
    'Qwen3VLIntegration',
    'Qwen3VLConfig',
    
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
