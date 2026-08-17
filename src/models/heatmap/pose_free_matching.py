"""Pose-free multi-history matching head for causal grounding diagnostics.

This module deliberately has a much narrower contract than the production
``HeatmapVLN`` decoder.  It receives only VLM features:

* current panoramic patch tokens, ``[B, 4, Hc, Wc, C_current]``;
* one VLM-produced query per historical observation, ``[B, K, C_query]``;
* an optional padding mask, ``[B, K]``.

It does *not* accept relative poses, trajectory coordinates, temporal-slot
embeddings, or per-history parameters.  Every historical query is matched to
the same current-patch bank with shared projections.  Consequently, permuting
the history dimension must permute the outputs in exactly the same way, and
changing one query cannot alter another query's heatmap.  Those properties are
useful causal contracts for the multi-history visual-grounding pilot.

The head is intentionally lightweight.  Localization is a normalized bilinear
match followed only by bilinear spatial upsampling; visibility is read from
three shared statistics of the same match map.  This makes it substantially
harder for the decoder to learn a standalone coordinate predictor while the
VLM features remain uninformative.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PANORAMIC_VIEWS = 4


def pad_history_queries(
    history_queries_batch: Sequence[Sequence[torch.Tensor]],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-sample VLM history queries without detaching their gradients.

    Args:
        history_queries_batch: Batch of variable-length sequences.  Every
            query must be a rank-1 tensor with a common channel dimension.
        device: Optional output device.  By default the first query's device.

    Returns:
        ``(queries, mask)`` with shapes ``[B, Kmax, C]`` and ``[B, Kmax]``.

    ``torch.cat``/``torch.stack`` are used rather than copying detached values
    into a preallocated buffer, so heatmap gradients keep an explicit path to
    the hooked VLM hidden states.
    """

    batch_size = len(history_queries_batch)
    first = next(
        (query for sample in history_queries_batch for query in sample),
        None,
    )
    if first is None:
        raise ValueError("At least one history query is required to infer the feature dimension")
    if first.ndim != 1:
        raise ValueError(f"Each history query must be rank 1, got {tuple(first.shape)}")

    target_device = first.device if device is None else device
    query_dim = int(first.shape[0])
    max_histories = max((len(sample) for sample in history_queries_batch), default=0)
    padded_samples: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for batch_idx, sample in enumerate(history_queries_batch):
        checked: list[torch.Tensor] = []
        for history_idx, query in enumerate(sample):
            if query.ndim != 1 or int(query.shape[0]) != query_dim:
                raise ValueError(
                    "History query shape mismatch at "
                    f"batch={batch_idx}, history={history_idx}: expected ({query_dim},), "
                    f"got {tuple(query.shape)}"
                )
            checked.append(query.to(device=target_device))

        if checked:
            sample_tensor = torch.stack(checked, dim=0)
        else:
            sample_tensor = first.new_empty((0, query_dim), device=target_device)
        padding = max_histories - len(checked)
        if padding:
            sample_tensor = torch.cat(
                [sample_tensor, sample_tensor.new_zeros((padding, query_dim))],
                dim=0,
            )
        padded_samples.append(sample_tensor)
        masks.append(torch.arange(max_histories, device=target_device) < len(checked))

    if batch_size == 0:
        # This branch is unreachable when ``first`` exists, but keeping it
        # explicit makes the output contract obvious to static readers.
        raise ValueError("History query batch must not be empty")
    return torch.stack(padded_samples, dim=0), torch.stack(masks, dim=0)


class PoseFreeHistoryMatcher(nn.Module):
    """Shared lightweight ``history-query x current-patch`` heatmap head.

    Relative pose is absent from the API by design.  The only route from a
    historical observation to its output is its VLM query vector.

    Args:
        current_dim: Channel dimension of current panoramic patch tokens.
        query_dim: Channel dimension of VLM history queries.
        match_dim: Low-dimensional shared matching space.
        heatmap_size: Output ``(height, width)`` per panoramic view.
        visibility_hidden_dim: Width of the tiny shared visibility readout.
        logit_temperature: Initial inverse temperature applied to cosine
            similarities.  It is trainable but bounded during forward.
    """

    uses_relative_pose = False
    num_views = NUM_PANORAMIC_VIEWS

    def __init__(
        self,
        current_dim: int,
        query_dim: int,
        match_dim: int = 64,
        heatmap_size: tuple[int, int] = (64, 64),
        visibility_hidden_dim: int = 16,
        logit_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if current_dim <= 0 or query_dim <= 0 or match_dim <= 0:
            raise ValueError("current_dim, query_dim, and match_dim must be positive")
        if visibility_hidden_dim <= 0:
            raise ValueError("visibility_hidden_dim must be positive")
        if len(heatmap_size) != 2 or min(int(v) for v in heatmap_size) <= 0:
            raise ValueError(f"heatmap_size must contain two positive values, got {heatmap_size}")
        if logit_temperature <= 0:
            raise ValueError("logit_temperature must be positive")

        self.current_dim = int(current_dim)
        self.query_dim = int(query_dim)
        self.match_dim = int(match_dim)
        self.heatmap_size = tuple(int(v) for v in heatmap_size)

        # Separate LayerNorm + bias-free projections keep the matching head
        # small and prevent either side from contributing a learned spatial or
        # history-slot bias by itself.
        self.current_norm = nn.LayerNorm(self.current_dim)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.current_projection = nn.Linear(
            self.current_dim,
            self.match_dim,
            bias=False,
        )
        self.query_projection = nn.Linear(
            self.query_dim,
            self.match_dim,
            bias=False,
        )

        # Visibility is derived only from the query-conditioned match map.  It
        # never sees a raw query, slot id, pose, or absolute trajectory value.
        self.visibility_readout = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, visibility_hidden_dim),
            nn.GELU(),
            nn.Linear(visibility_hidden_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(float(logit_temperature)), dtype=torch.float32))

    @property
    def trainable_parameter_count(self) -> int:
        """Number of decoder parameters, useful for attribution reports."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_inputs(
        self,
        current_patches: torch.Tensor,
        history_queries: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if current_patches.ndim != 5:
            raise ValueError(f"current_patches must have shape [B, 4, Hc, Wc, C], got {tuple(current_patches.shape)}")
        if int(current_patches.shape[1]) != self.num_views:
            raise ValueError(f"Expected {self.num_views} panoramic views, got {int(current_patches.shape[1])}")
        if int(current_patches.shape[-1]) != self.current_dim:
            raise ValueError(f"Expected current feature dim {self.current_dim}, got {int(current_patches.shape[-1])}")
        if history_queries.ndim != 3:
            raise ValueError(f"history_queries must have shape [B, K, Cq], got {tuple(history_queries.shape)}")
        if int(history_queries.shape[0]) != int(current_patches.shape[0]):
            raise ValueError(
                "Batch mismatch between current_patches and history_queries: "
                f"{int(current_patches.shape[0])} vs {int(history_queries.shape[0])}"
            )
        if int(history_queries.shape[-1]) != self.query_dim:
            raise ValueError(f"Expected history query dim {self.query_dim}, got {int(history_queries.shape[-1])}")

        expected_mask_shape = history_queries.shape[:2]
        if history_mask is None:
            return torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=history_queries.device,
            )
        if tuple(history_mask.shape) != tuple(expected_mask_shape):
            raise ValueError(
                f"history_mask must have shape {tuple(expected_mask_shape)}, got {tuple(history_mask.shape)}"
            )
        return history_mask.to(device=history_queries.device, dtype=torch.bool)

    @staticmethod
    def _visibility_features(coarse_logits: torch.Tensor) -> torch.Tensor:
        """Return shared match-strength statistics ``[B, K, 4, 3]``."""

        flat = coarse_logits.flatten(-2)
        maximum = flat.amax(dim=-1)
        mean = flat.mean(dim=-1)
        smooth_max = torch.logsumexp(flat, dim=-1) - math.log(flat.shape[-1])
        return torch.stack([maximum, mean, smooth_max], dim=-1)

    def forward(
        self,
        current_patches: torch.Tensor,
        history_queries: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict one independent panoramic heatmap per history query.

        Returns:
            A dictionary containing:

            * ``heatmaps``: sigmoid probabilities ``[B, K, 4, H, W]``;
            * ``visibility``: per-view logits ``[B, K, 4]``;
            * ``heatmap_logits``: pre-sigmoid logits at output resolution;
            * ``coarse_heatmap_logits``: native patch-grid match logits;
            * ``history_mask``: normalized boolean padding mask.
        """

        mask = self._validate_inputs(current_patches, history_queries, history_mask)
        batch_size, _, coarse_h, coarse_w, _ = current_patches.shape
        num_histories = history_queries.shape[1]

        if num_histories == 0:
            output_h, output_w = self.heatmap_size
            return {
                "heatmaps": current_patches.new_empty((batch_size, 0, self.num_views, output_h, output_w)),
                "visibility": current_patches.new_empty((batch_size, 0, self.num_views)),
                "heatmap_logits": current_patches.new_empty((batch_size, 0, self.num_views, output_h, output_w)),
                "coarse_heatmap_logits": current_patches.new_empty((batch_size, 0, self.num_views, coarse_h, coarse_w)),
                "history_mask": mask,
            }

        current_keys = self.current_projection(self.current_norm(current_patches))
        history_keys = self.query_projection(self.query_norm(history_queries))
        current_keys = F.normalize(current_keys, dim=-1)
        history_keys = F.normalize(history_keys, dim=-1)

        # Capped exactly as in common contrastive matching heads.  This is the
        # sole operation that couples a historical query to a current patch.
        scale = self.logit_scale.exp().clamp(max=100.0).to(current_keys.dtype)
        coarse_logits = scale * torch.einsum(
            "bkf,bvhwf->bkvhw",
            history_keys,
            current_keys,
        )

        visibility_features = self._visibility_features(coarse_logits)
        visibility = self.visibility_readout(visibility_features).squeeze(-1)

        output_h, output_w = self.heatmap_size
        heatmap_logits = F.interpolate(
            coarse_logits.reshape(
                batch_size * num_histories * self.num_views,
                1,
                coarse_h,
                coarse_w,
            ),
            size=(output_h, output_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size,
            num_histories,
            self.num_views,
            output_h,
            output_w,
        )
        heatmaps = torch.sigmoid(heatmap_logits)

        # Preserve the established HeatmapVLN padded-output convention.  The
        # loss must still receive the same mask so padded zero logits are not
        # interpreted as negative visibility examples.
        mask_values = mask.to(dtype=heatmaps.dtype)
        heatmaps = heatmaps * mask_values[:, :, None, None, None]
        heatmap_logits = heatmap_logits * mask_values[:, :, None, None, None]
        coarse_logits = coarse_logits * mask_values[:, :, None, None, None]
        visibility = visibility * mask_values[:, :, None]

        return {
            "heatmaps": heatmaps,
            "visibility": visibility,
            "heatmap_logits": heatmap_logits,
            "coarse_heatmap_logits": coarse_logits,
            "history_mask": mask,
        }


__all__ = [
    "NUM_PANORAMIC_VIEWS",
    "PoseFreeHistoryMatcher",
    "pad_history_queries",
]
