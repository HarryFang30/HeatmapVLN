"""
Coarse Localization Module
============================

Zero trainable parameters.  Relies entirely on Qwen3.5's frozen features.

Computes dot-product matching between text query vectors (history position
hidden states) and current view LLM features (8x8 spatial grids) to produce:
  - visibility logits  (scalar per view, 4 total)
  - coarse heatmaps    (8x8 per view)

Reference: HeatmapVLN设计文档 Section 5
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseLocalization(nn.Module):
    """
    Zero-parameter coarse localisation via cosine-similarity response maps.

    Inputs:
        current_llm:     ``{0: (H,W,C), 1: (H,W,C), 2: (H,W,C), 3: (H,W,C)}``
        history_queries: list of ``(C,)`` tensors (text token hidden states)

    Outputs:
        list of dicts, each containing:
            - ``visibility``: ``(4,)``  — per-view visibility logits
            - ``coarse_heatmap``: ``(4, H, W)`` — per-view coarse heatmaps
    """

    def __init__(self, visibility_scale: float = 4.0):
        super().__init__()
        self.visibility_scale = visibility_scale

    def forward(
        self,
        current_llm: Dict[int, torch.Tensor],
        history_queries: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []

        for q in history_queries:
            q_norm = F.normalize(q, dim=-1)  # (C,)

            view_vis = []
            view_heatmaps = []

            for view_idx in range(4):
                v_feat = current_llm[view_idx]  # (H, W, C)
                v_feat_norm = F.normalize(v_feat, dim=-1)

                heatmap = torch.einsum("c, hwc -> hw", q_norm, v_feat_norm)

                visibility = heatmap.max() * self.visibility_scale

                view_vis.append(visibility)
                view_heatmaps.append(heatmap)

            results.append({
                "visibility": torch.stack(view_vis),          # (4,)
                "coarse_heatmap": torch.stack(view_heatmaps), # (4, H, W)
            })

        return results
