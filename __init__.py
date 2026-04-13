"""
HeatmapVLN — Vision-Language Navigation with Coarse-to-Fine Heatmap Generation

Architecture (v2 — Qwen2.5-VL + HeatmapVLN + NextDiT System 1):
    Current panorama (4 views) + N history panoramas + text instruction
        |
    Qwen2.5-VL (Vision Encoder + LLM, frozen + LoRA)
        |
    ├── HeatmapVLN (Coarse-to-Fine localization)
    │       → visibility prediction + 64×64 heatmap
    │
    └── NextDiT System 1 (InternNav action head)
            → trajectory (B, T, 3)
"""

__version__ = "0.1.0"
__author__ = "Jialei Fang"
__description__ = "First-Person Inter-Frame Spatial Understanding for VLN"
