"""
VLN Models Module
=================

Architecture (v2 — Coarse-to-Fine + InternNav System 1):
    Current panorama (4 views) + N history panoramas (N*4 views) + text
        |
    Qwen2.5-VL (Vision Encoder + LLM, frozen + LoRA)
        |
    ├── HeatmapVLN (Coarse-to-Fine)
    │       → visibility + 64x64 heatmap
    │
    └── NextDiT System 1 (InternNav action head)
            → trajectory (B, T, 3)
"""

# === VLM Integration ===
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

# === Action Components (NextDiT System 1) ===
from .action import (
    NextDiTActionHead,
    NextDiTActionConfig,
)

# === Other Components ===
try:
    from .mlp import MLP
except ImportError:
    MLP = None


__all__ = [
    'Qwen3_5Integration',
    'Qwen3_5Config',
    'VLNPipeline',
    'VLNPipelineConfig',
    'create_vln_pipeline',
    'HeatmapVLN',
    'HeatmapVLNLoss',
    'CoarseLocalization',
    'DPTLiteFusion',
    'FineLocalization',
    'FeatureExtractor',
    'NextDiTActionHead',
    'NextDiTActionConfig',
    'MLP',
]
