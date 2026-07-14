# Heatmap Module — HeatmapVLN v2
#
# Current default coarse-to-fine system:
#   - TrajectoryGuidedAttention: history query + rel pose + spatial token fusion
#   - DPTLiteFusion: multi-layer ViT / LLM feature fusion
#   - FineLocalization: spatial_out-guided decoder (no FiLM, no fine-stage query_proj)
#   - HeatmapVLN: complete model assembly
#   - HeatmapVLNLoss: four-component task-priority loss

from .coarse_localization import CoarseLocalization
from .dpt_lite_fusion import DPTLiteFusion
from .feature_extractor import FeatureExtractor
from .fine_localization import FineLocalization
from .heatmap_vln import HeatmapVLN
from .heatmap_vln_loss import HeatmapVLNLoss
from .input_constructor import construct_input, find_text_anchor_positions
from .pose_free_matching import PoseFreeHistoryMatcher, pad_history_queries
from .trajectory_attention import TrajectoryGuidedAttention

__version__ = "5.0.0"

__all__ = [
    "CoarseLocalization",
    "DPTLiteFusion",
    "FeatureExtractor",
    "FineLocalization",
    "HeatmapVLN",
    "HeatmapVLNLoss",
    "PoseFreeHistoryMatcher",
    "TrajectoryGuidedAttention",
    "__version__",
    "construct_input",
    "find_text_anchor_positions",
    "pad_history_queries",
]
