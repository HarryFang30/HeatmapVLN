"""
VLN Models Module
=================

This module provides a VLN pipeline that uses Qwen3.5 for video understanding.

Architecture (v2 — Coarse-to-Fine):
    Current panorama (4 views) + N history panoramas (N*4 views) + text
        |
    Qwen3.5 (Vision Encoder + LLM, frozen)
        |
    ViT features (16x16) + LLM features (8x8) + text hidden states
        |
    Coarse Localisation (zero params) -> visibility + 8x8 heatmap
        |
    Fine Localisation (~2M params) -> 64x64 heatmap
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

# === Heatmap Components (v2 Coarse-to-Fine) ===
from .heatmap import (
    HeatmapVLN,
    HeatmapVLNLoss,
    CoarseLocalization,
    DPTLiteFusion,
    FineLocalization,
    FeatureExtractor,
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

    # Heatmap Components (v2)
    'HeatmapVLN',
    'HeatmapVLNLoss',
    'CoarseLocalization',
    'DPTLiteFusion',
    'FineLocalization',
    'FeatureExtractor',

    # Action Components
    'DiffusionActionHead',
    'DiffusionActionConfig',
    'StopPredictionHead',

    # Other
    'MLP',
]
