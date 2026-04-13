"""
Trajectory-Guided Attention Module
=====================================

Replaces CoarseLocalization with a self-attention mechanism that allows
three types of tokens to interact:

  1. **history_query** (N, c_llm) — visual identity of historical locations
  2. **trajectory PE** (N, pe_dim) — sinusoidal positional encoding of
     relative egocentric poses (dx, dy, cos_yaw, sin_yaw)
  3. **spatial features** (V*H*W, c_fused) — DPT-fused LLM features for
     current 4-view panorama (flattened to 256 tokens)

Outputs:
  - coarse_heatmap (N, 4, 8, 8)       — per-history-location heatmap logits
  - visibility     (N, 4)             — per-view visibility logits
  - spatial_out    (N, V*H*W, d_attn) — enriched spatial features for Fine
"""

import math
from typing import Union

import torch
import torch.nn as nn


def sinusoidal_pe(
    x: torch.Tensor,
    num_freqs: int = 16,
    max_spatial_range: float = 10.0,
) -> torch.Tensor:
    """
    Sinusoidal positional encoding for continuous values.

    Args:
        x: (..., D) continuous input (D=4 for dx, dy, cos_yaw, sin_yaw)
        num_freqs: number of frequency bands
        max_spatial_range: normalization constant for spatial coordinates

    Returns:
        (..., D * (1 + 2*num_freqs)) — original values + sin/cos encodings
    """
    x_norm = x / max_spatial_range
    freqs = torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    freqs = (2.0 * math.pi * (2.0 ** freqs))  # (num_freqs,)

    # (..., D, 1) * (num_freqs,) -> (..., D, num_freqs)
    angles = x_norm.unsqueeze(-1) * freqs
    pe = torch.cat([
        x_norm,                      # (..., D)
        angles.sin().flatten(-2),    # (..., D * num_freqs)
        angles.cos().flatten(-2),    # (..., D * num_freqs)
    ], dim=-1)
    return pe


class TrajectoryGuidedAttention(nn.Module):
    """
    Self-attention fusion of history_query, trajectory PE, and spatial features.

    Token layout per history sample:
        [hist_token(1), traj_token(1), spatial_0 ... spatial_255]
    Total: 258 tokens, all projected to d_attn dimensions.

    Args:
        c_llm:     LLM hidden dimension for history_query (default 4096)
        c_fused:   Fused spatial feature dimension (default 256, = d_attn)
        num_freqs: Frequency bands for sinusoidal PE (default 16)
        d_attn:    Attention dimension (default 256)
        num_heads: Attention heads (default 4)
        num_layers: Transformer encoder layers (default 1)
        max_spatial_range: PE normalization range (default 10.0)
    """

    def __init__(
        self,
        c_llm: int = 4096,
        c_fused: int = 256,
        num_freqs: int = 16,
        d_attn: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        max_spatial_range: float = 10.0,
    ):
        super().__init__()
        self.c_llm = c_llm
        self.c_fused = c_fused
        self.d_attn = d_attn
        self.num_freqs = num_freqs
        self.max_spatial_range = max_spatial_range

        pe_dim = 4 * (1 + 2 * num_freqs)  # 4 * 33 = 132 by default

        self.proj_history = nn.Linear(c_llm, d_attn)
        self.proj_traj = nn.Linear(pe_dim, d_attn)

        # Learnable positional embedding for the 258 token positions
        # [0] = history, [1] = trajectory, [2:258] = 256 spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(258, d_attn) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_attn,
            nhead=num_heads,
            dim_feedforward=d_attn * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.heatmap_head = nn.Sequential(
            nn.Linear(d_attn, d_attn // 2),
            nn.GELU(),
            nn.Linear(d_attn // 2, 1),
        )

        self.vis_head = nn.Sequential(
            nn.Linear(d_attn, d_attn // 2),
            nn.GELU(),
            nn.Linear(d_attn // 2, 4),
        )

    def _build_pe(self, x: torch.Tensor) -> torch.Tensor:
        return sinusoidal_pe(x, self.num_freqs, self.max_spatial_range)

    def forward(
        self,
        current_llm: Union[dict[int, torch.Tensor], torch.Tensor],
        history_queries: Union[list[torch.Tensor], torch.Tensor],
        history_rel_poses: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            current_llm:
                Non-batch: dict {view_idx: (H, W, C_fused)} or (4, H, W, C_fused)
                Batch:     (B, 4, H, W, C_fused)
            history_queries:
                Non-batch: list of (C_llm,) or (N, C_llm)
                Batch:     (B, N, C_llm)
            history_rel_poses:
                Non-batch: (N, 4) or None
                Batch:     (B, N, 4) or None

        Returns:
            dict with "visibility" and "coarse_heatmap"
        """
        # Normalize inputs to tensor form
        if isinstance(current_llm, dict):
            current_llm_tensor = torch.stack(
                [current_llm[i] for i in range(4)], dim=0
            )  # (4, H, W, C)
        else:
            current_llm_tensor = current_llm

        if isinstance(history_queries, list):
            if len(history_queries) == 0:
                device = current_llm_tensor.device
                V = current_llm_tensor.shape[-4]
                H, W = current_llm_tensor.shape[-3], current_llm_tensor.shape[-2]
                return {
                    "visibility": torch.empty(0, 4, device=device),
                    "coarse_heatmap": torch.empty(0, 4, H, W, device=device),
                    "spatial_out": torch.empty(0, V * H * W, self.d_attn, device=device),
                }
            history_queries_tensor = torch.stack(history_queries, dim=0)
        else:
            history_queries_tensor = history_queries

        is_batched = current_llm_tensor.dim() == 5

        if is_batched:
            return self._forward_batch(
                current_llm_tensor, history_queries_tensor, history_rel_poses
            )
        else:
            return self._forward_single(
                current_llm_tensor, history_queries_tensor, history_rel_poses
            )

    def _forward_single(
        self,
        spatial: torch.Tensor,       # (4, H, W, C)
        hist_q: torch.Tensor,        # (N, C_llm)
        rel_poses: torch.Tensor | None,  # (N, 4) or None
    ) -> dict[str, torch.Tensor]:
        N = hist_q.shape[0]
        V, H, W, C = spatial.shape
        device = spatial.device

        spatial_flat = spatial.reshape(V * H * W, C)  # (256, C)

        hist_proj = self.proj_history(hist_q)  # (N, d_attn)

        if rel_poses is not None:
            traj_pe = self._build_pe(rel_poses)
            traj_proj = self.proj_traj(traj_pe)  # (N, d_attn)
        else:
            traj_proj = torch.zeros(N, self.d_attn, device=device)

        results_vis = []
        results_hm = []
        results_spatial = []

        for i in range(N):
            tokens = torch.cat([
                hist_proj[i:i+1],       # (1, d_attn)
                traj_proj[i:i+1],       # (1, d_attn)
                spatial_flat,           # (256, d_attn)
            ], dim=0)  # (258, d_attn)

            tokens = tokens + self.pos_embed[:tokens.shape[0]]
            tokens = tokens.unsqueeze(0)  # (1, 258, d_attn)

            out = self.self_attn(tokens).squeeze(0)  # (258, d_attn)

            sp_out = out[2:]  # (256, d_attn)
            hm_logits = self.heatmap_head(sp_out).squeeze(-1)  # (256,)
            hm = hm_logits.reshape(V, H, W)  # (4, 8, 8)

            hist_out = out[0]  # (d_attn,)
            vis = self.vis_head(hist_out)  # (4,)

            results_hm.append(hm)
            results_vis.append(vis)
            results_spatial.append(sp_out)

        return {
            "coarse_heatmap": torch.stack(results_hm, dim=0),       # (N, 4, H, W)
            "visibility": torch.stack(results_vis, dim=0),           # (N, 4)
            "spatial_out": torch.stack(results_spatial, dim=0),      # (N, V*H*W, d_attn)
        }

    def _forward_batch(
        self,
        spatial: torch.Tensor,       # (B, 4, H, W, C)
        hist_q: torch.Tensor,        # (B, N, C_llm)
        rel_poses: torch.Tensor | None,  # (B, N, 4) or None
    ) -> dict[str, torch.Tensor]:
        B, V, H, W, C = spatial.shape
        N = hist_q.shape[1]
        device = spatial.device

        spatial_flat = spatial.reshape(B, V * H * W, C)  # (B, 256, C)

        hist_proj = self.proj_history(hist_q)  # (B, N, d_attn)

        if rel_poses is not None:
            traj_pe = self._build_pe(rel_poses)
            traj_proj = self.proj_traj(traj_pe)  # (B, N, d_attn)
        else:
            traj_proj = torch.zeros(B, N, self.d_attn, device=device)

        spatial_expanded = spatial_flat.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, V * H * W, C)

        hist_tokens = hist_proj.reshape(B * N, 1, self.d_attn)
        traj_tokens = traj_proj.reshape(B * N, 1, self.d_attn)

        tokens = torch.cat([hist_tokens, traj_tokens, spatial_expanded], dim=1)  # (B*N, 258, d_attn)
        tokens = tokens + self.pos_embed[:tokens.shape[1]]

        out = self.self_attn(tokens)  # (B*N, 258, d_attn)

        sp_out = out[:, 2:]  # (B*N, 256, d_attn)
        hm_logits = self.heatmap_head(sp_out).squeeze(-1)  # (B*N, 256)
        heatmaps = hm_logits.reshape(B, N, V, H, W)  # (B, N, 4, 8, 8)

        hist_out = out[:, 0]  # (B*N, d_attn)
        vis = self.vis_head(hist_out)  # (B*N, 4)
        visibility = vis.reshape(B, N, V)  # (B, N, 4)

        return {
            "coarse_heatmap": heatmaps,
            "visibility": visibility,
            "spatial_out": sp_out.reshape(B, N, V * H * W, self.d_attn),  # (B, N, 256, d_attn)
        }
