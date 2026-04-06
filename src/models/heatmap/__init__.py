# Heatmap Module — HeatmapVLN v2
#
# Current default coarse-to-fine system:
#   - TrajectoryGuidedAttention: history query + rel pose + spatial token fusion
#   - DPTLiteFusion: multi-layer ViT / LLM feature fusion
#   - FineLocalization: spatial_out-guided decoder (no FiLM, no fine-stage query_proj)
#   - HeatmapVLN: complete model assembly
#   - HeatmapVLNLoss: four-component task-priority loss

from .heatmap_vln import HeatmapVLN
from .heatmap_vln_loss import HeatmapVLNLoss
from .coarse_localization import CoarseLocalization
from .trajectory_attention import TrajectoryGuidedAttention
from .dpt_lite_fusion import DPTLiteFusion
from .fine_localization import FineLocalization
from .feature_extractor import FeatureExtractor
from .input_constructor import construct_input, find_text_anchor_positions

__version__ = "5.0.0"

__all__ = [
    "HeatmapVLN",
    "HeatmapVLNLoss",
    "CoarseLocalization",
    "TrajectoryGuidedAttention",
    "DPTLiteFusion",
    "FineLocalization",
    "FeatureExtractor",
    "construct_input",
    "find_text_anchor_positions",
    "__version__",
]
