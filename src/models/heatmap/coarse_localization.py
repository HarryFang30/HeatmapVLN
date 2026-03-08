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
        batch = self.forward_batched(current_llm, history_queries)
        return [
            {
                "visibility": batch["visibility"][hist_idx],
                "coarse_heatmap": batch["coarse_heatmap"][hist_idx],
            }
            for hist_idx in range(batch["visibility"].shape[0])
        ]

    def forward_batched(self, current_llm, history_queries) -> Dict[str, torch.Tensor]:
        """Batched coarse localization."""
        if isinstance(current_llm, dict):
            current_llm_tensor = torch.stack([current_llm[view_idx] for view_idx in range(4)], dim=0)
        else:
            current_llm_tensor = current_llm

        if isinstance(history_queries, list):
            if len(history_queries) == 0:
                device = current_llm_tensor.device
                num_views = current_llm_tensor.shape[0]
                height, width = current_llm_tensor.shape[1:3]
                return {
                    "visibility": torch.empty(0, num_views, device=device),
                    "coarse_heatmap": torch.empty(0, num_views, height, width, device=device),
                }
            history_queries_tensor = torch.stack(history_queries, dim=0)
        else:
            history_queries_tensor = history_queries

        q_proj = self.query_proj(history_queries_tensor)
        q_norm = F.normalize(q_proj, dim=-1)
        v_feat_norm = F.normalize(current_llm_tensor, dim=-1)
        heatmaps = torch.einsum("nc,vhwc->nvhw", q_norm, v_feat_norm)

        if self.vis_head is not None:
            hm_max = heatmaps.amax(dim=(-2, -1))
            hm_mean = heatmaps.mean(dim=(-2, -1))
            hm_std = heatmaps.std(dim=(-2, -1))
            stats = torch.stack([hm_max, hm_mean, hm_std], dim=-1)
            query_expand = history_queries_tensor[:, None, :].expand(-1, current_llm_tensor.shape[0], -1)
            vis_input = torch.cat([query_expand, stats], dim=-1)
            visibility = self.vis_head(vis_input.reshape(-1, vis_input.shape[-1])).reshape(
                history_queries_tensor.shape[0], current_llm_tensor.shape[0]
            )
        else:
            visibility = heatmaps.amax(dim=(-2, -1)) * self.visibility_scale

        return {
            "visibility": visibility,
            "coarse_heatmap": heatmaps,
        }
