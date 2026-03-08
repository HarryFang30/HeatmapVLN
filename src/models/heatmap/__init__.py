# Heatmap Module — HeatmapVLN v2
#
# Coarse-to-Fine spatial projection system:
#   - CoarseLocalization: zero-param dot-product matching (visibility + 8x8 heatmap)
#   - DPTLiteFusion: multi-layer ViT feature fusion (16x16)
#   - FineLocalization: FiLM modulation + CNN upsample (64x64)
#   - HeatmapVLN: complete model assembly
#   - HeatmapVLNLoss: four-component task-priority loss

from .heatmap_vln import HeatmapVLN
from .heatmap_vln_loss import HeatmapVLNLoss
from .coarse_localization import CoarseLocalization
from .dpt_lite_fusion import DPTLiteFusion
from .fine_localization import FineLocalization
from .feature_extractor import FeatureExtractor
from .input_constructor import construct_input, find_text_anchor_positions

__version__ = "5.0.0"

__all__ = [
    "HeatmapVLN",
    "HeatmapVLNLoss",
    "CoarseLocalization",
    "DPTLiteFusion",
    "FineLocalization",
    "FeatureExtractor",
    "construct_input",
    "find_text_anchor_positions",
    "__version__",
]
