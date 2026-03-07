"""
Coarse Localization Module
============================

Computes dot-product matching between text query vectors and multi-layer
fused LLM visual features (8x8 spatial grids) to produce:
  - visibility logits  (scalar per view, 4 total)  — via trainable MLP
  - coarse heatmaps    (8x8 per view)              — cosine similarity

The multi-layer LLM features are fused upstream (LLM DPT-Lite) into a
c_fused-dimensional representation.  The text query (c_llm-dimensional)
is projected to c_fused via a learned Linear before the dot product.

Reference: HeatmapVLN设计文档 Section 5
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseLocalization(nn.Module):
    """
    Coarse localisation with trainable query projection and visibility head.

    Args:
        c_llm:   LLM hidden dimension (text query input dim).
        c_fused: Fused visual feature dimension after LLM DPT-Lite fusion.
                 The dot product operates in this space.
        visibility_scale: scale factor for the zero-parameter fallback
                          (used only when c_llm=0).
    """

    def __init__(
        self,
        c_llm: int = 4096,
        c_fused: int = 256,
        visibility_scale: float = 4.0,
    ):
        super().__init__()
        self.visibility_scale = visibility_scale
        self.c_llm = c_llm
        self.c_fused = c_fused

        # Project text query from c_llm → c_fused for cosine similarity
        self.query_proj = nn.Linear(c_llm, c_fused)

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
        """
        Args:
            current_llm:     ``{view_idx: (H, W, C_fused)}`` — fused multi-layer features.
            history_queries: list of ``(C_llm,)`` tensors (original LLM dim).

        Returns:
            list of dicts with ``visibility`` (4,) and ``coarse_heatmap`` (4, H, W).
        """
        results: List[Dict[str, torch.Tensor]] = []

        for q in history_queries:
            q_proj = self.query_proj(q)                   # (C_fused,)
            q_norm = F.normalize(q_proj, dim=-1)          # (C_fused,)

            view_vis = []
            view_heatmaps = []

            for view_idx in range(4):
                v_feat = current_llm[view_idx]            # (H, W, C_fused)
                v_feat_norm = F.normalize(v_feat, dim=-1)

                heatmap = torch.einsum("c, hwc -> hw", q_norm, v_feat_norm)
                view_heatmaps.append(heatmap)

                if self.vis_head is not None:
                    hm_max = heatmap.max()
                    hm_mean = heatmap.mean()
                    hm_std = heatmap.std()
                    stats = torch.stack([hm_max, hm_mean, hm_std])  # (3,)
                    vis_input = torch.cat([q, stats])     # (C_llm + 3,)
                    visibility = self.vis_head(vis_input).squeeze(-1)
                else:
                    visibility = heatmap.max() * self.visibility_scale

                view_vis.append(visibility)

            results.append({
                "visibility": torch.stack(view_vis),          # (4,)
                "coarse_heatmap": torch.stack(view_heatmaps), # (4, H, W)
            })

        return results
