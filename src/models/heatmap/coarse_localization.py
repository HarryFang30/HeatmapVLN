"""
Coarse Localization Module
============================

Computes dot-product matching between text query vectors (history position
hidden states) and current view LLM features (8x8 spatial grids) to produce:
  - visibility logits  (scalar per view, 4 total)  — via trainable MLP
  - coarse heatmaps    (8x8 per view)              — zero-parameter cosine sim

The coarse heatmaps remain zero-parameter (pure cosine similarity), preserving
the design principle that spatial correspondence comes from Qwen3.5.

The visibility prediction adds a lightweight trainable MLP (~66K params) so
that the visibility BCE loss can provide gradient signal during training.

Reference: HeatmapVLN设计文档 Section 5
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseLocalization(nn.Module):
    """
    Coarse localisation with trainable visibility head.

    The coarse heatmap (8x8 cosine-similarity response map) is zero-parameter.
    The visibility head is a small MLP that takes the heatmap statistics and
    the query-view dot-product features to produce a trainable visibility logit,
    enabling the visibility BCE loss to propagate gradients.

    Args:
        c_llm: LLM hidden dimension (for the trainable visibility head).
               Set to 0 to disable the trainable head and fall back to
               the original zero-parameter ``heatmap.max() * scale`` mode.
        visibility_scale: scale factor for the zero-parameter fallback.
    """

    def __init__(self, c_llm: int = 4096, visibility_scale: float = 4.0):
        super().__init__()
        self.visibility_scale = visibility_scale
        self.c_llm = c_llm

        if c_llm > 0:
            self.vis_head = nn.Sequential(
                nn.Linear(c_llm + 3, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
        else:
            self.vis_head = None

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
                view_heatmaps.append(heatmap)

                if self.vis_head is not None:
                    hm_max = heatmap.max()
                    hm_mean = heatmap.mean()
                    hm_std = heatmap.std()
                    stats = torch.stack([hm_max, hm_mean, hm_std])  # (3,)
                    vis_input = torch.cat([q, stats])  # (C+3,)
                    visibility = self.vis_head(vis_input).squeeze(-1)  # scalar
                else:
                    visibility = heatmap.max() * self.visibility_scale

                view_vis.append(visibility)

            results.append({
                "visibility": torch.stack(view_vis),          # (4,)
                "coarse_heatmap": torch.stack(view_heatmaps), # (4, H, W)
            })

        return results
