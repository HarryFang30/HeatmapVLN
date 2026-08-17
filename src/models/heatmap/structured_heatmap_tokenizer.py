"""Structured tokenization of frozen four-view heatmap predictions.

The tokenizer consumes only raw outputs from the frozen heatmap branch and
explicit history metadata. It intentionally does not accept RGB features: the
control path should describe where each historical observation is in the
current panorama, how confident that localization is, and when it occurred.

Input contract:
- heatmap_logits: raw spatial logits [B, K, 4, 64, 64].
- visibility_logits: raw front/right/back/left logits [B, K, 4].
- history_mask: boolean or binary numeric validity mask [B, K].
- history_age_steps: non-negative primitive-step ages [B, K].

Output token order is history-major, then view-major within each history:
h0/front, h0/right, h0/back, h0/left, h1/front, and so on. All probability and
moment calculations run in FP32 even when called under AMP.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


HEATMAP_DIRECTION_ORDER = ("front", "right", "back", "left")
HEATMAP_SIZE = (64, 64)
COARSE_SIZE = (8, 8)
SPATIAL_STATISTIC_NAMES = (
    "mean_x",
    "mean_y",
    "var_x",
    "var_y",
    "cov_xy",
    "normalized_entropy",
    "peak_probability",
)

# 64 coarse probabilities + 7 spatial statistics + 7 state/geometry values:
# visibility logit, p(view), p(none), sin(yaw), cos(yaw), age, rank.
STRUCTURED_FEATURE_DIM = 64 + len(SPATIAL_STATISTIC_NAMES) + 7


def _autocast_disabled(tensor: torch.Tensor):
    """Return a disabled-autocast context for supported device types."""

    if tensor.device.type in {"cpu", "cuda", "xpu", "mps"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class StructuredHeatmapTokenizer(nn.Module):
    """Convert frozen heatmap predictions into trainable control tokens.

    Spatial logits are normalized independently inside each view. The 8x8
    representation uses sum pooling rather than average pooling, preserving
    unit probability mass for every valid view. Visibility uses the same
    operational five-way distribution as HeatmapVLN inference: a fixed zero
    none logit followed by the four predicted view logits.

    Args:
        token_dim: Output control-token width.
        mlp_hidden_dim: Hidden width of the shared per-token MLP.
        temporal_num_heads: Attention heads in the one-layer temporal encoder.
        temporal_ffn_dim: Feed-forward width in the temporal encoder.
        dropout: Dropout in the trainable tokenizer.
        age_scale_steps: Fixed saturation scale for logarithmic age encoding.
        probability_epsilon: Clamp used only inside entropy logarithms.
    """

    direction_order = HEATMAP_DIRECTION_ORDER
    spatial_statistic_names = SPATIAL_STATISTIC_NAMES
    structured_feature_dim = STRUCTURED_FEATURE_DIM

    def __init__(
        self,
        *,
        token_dim: int = 128,
        mlp_hidden_dim: int = 256,
        temporal_num_heads: int = 4,
        temporal_ffn_dim: int = 512,
        dropout: float = 0.0,
        age_scale_steps: float = 32.0,
        probability_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if int(mlp_hidden_dim) <= 0:
            raise ValueError("mlp_hidden_dim must be positive")
        if int(temporal_num_heads) <= 0 or int(token_dim) % int(temporal_num_heads):
            raise ValueError("temporal_num_heads must divide token_dim")
        if int(temporal_ffn_dim) <= 0:
            raise ValueError("temporal_ffn_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(float(age_scale_steps)) or float(age_scale_steps) <= 0.0:
            raise ValueError("age_scale_steps must be finite and positive")
        if (
            not math.isfinite(float(probability_epsilon))
            or not 0.0 < float(probability_epsilon) < 1.0
        ):
            raise ValueError("probability_epsilon must be finite and in (0, 1)")

        self.token_dim = int(token_dim)
        self.age_scale_steps = float(age_scale_steps)
        self.probability_epsilon = float(probability_epsilon)

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(STRUCTURED_FEATURE_DIM),
            nn.Linear(STRUCTURED_FEATURE_DIM, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        # Exactly one layer. Full attention is intentional because every
        # history token is already observed; no causal mask is needed.
        self.temporal_transformer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(temporal_num_heads),
            dim_feedforward=int(temporal_ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        coordinate = torch.linspace(-1.0, 1.0, HEATMAP_SIZE[0], dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.register_buffer(
            "_grid_x",
            grid_x.reshape(1, 1, 1, *HEATMAP_SIZE),
            persistent=False,
        )
        self.register_buffer(
            "_grid_y",
            grid_y.reshape(1, 1, 1, *HEATMAP_SIZE),
            persistent=False,
        )
        # Exact cardinal values avoid tiny sin(pi) residuals in diagnostics.
        self.register_buffer(
            "_view_yaw_sin_cos",
            torch.tensor(
                (
                    (0.0, 1.0),   # front:   0 deg
                    (-1.0, 0.0),  # right: -90 deg
                    (0.0, -1.0),  # back:  180 deg
                    (1.0, 0.0),   # left:   90 deg
                ),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _normalize_history_mask(history_mask: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(history_mask) or history_mask.ndim != 2:
            shape = tuple(history_mask.shape) if torch.is_tensor(history_mask) else None
            raise ValueError(f"history_mask must be [B,K], got {shape}")
        if history_mask.dtype == torch.bool:
            return history_mask
        if history_mask.is_floating_point() and not torch.isfinite(history_mask).all():
            raise ValueError("history_mask contains non-finite values")
        if not torch.all((history_mask == 0) | (history_mask == 1)):
            raise ValueError("numeric history_mask must contain only 0 and 1")
        return history_mask.bool()

    @staticmethod
    def _validate_inputs(
        heatmap_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
        history_mask: torch.Tensor,
        history_age_steps: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor]:
        if not torch.is_tensor(heatmap_logits) or heatmap_logits.ndim != 5:
            shape = tuple(heatmap_logits.shape) if torch.is_tensor(heatmap_logits) else None
            raise ValueError(
                "heatmap_logits must be [B,K,4,64,64], "
                f"got {shape}"
            )
        if tuple(heatmap_logits.shape[-3:]) != (4, *HEATMAP_SIZE):
            raise ValueError(
                "heatmap_logits must end in [4,64,64], got "
                f"{tuple(heatmap_logits.shape)}"
            )
        if not heatmap_logits.is_floating_point():
            raise TypeError("heatmap_logits must be floating point")
        batch_size, num_history = heatmap_logits.shape[:2]
        expected_visibility = (batch_size, num_history, 4)
        if (
            not torch.is_tensor(visibility_logits)
            or tuple(visibility_logits.shape) != expected_visibility
        ):
            shape = tuple(visibility_logits.shape) if torch.is_tensor(visibility_logits) else None
            raise ValueError(
                "visibility_logits must be [B,K,4] aligned with heatmap_logits, "
                f"got {shape}"
            )
        if not visibility_logits.is_floating_point():
            raise TypeError("visibility_logits must be floating point")
        expected_history = (batch_size, num_history)
        if (
            not torch.is_tensor(history_age_steps)
            or tuple(history_age_steps.shape) != expected_history
        ):
            shape = tuple(history_age_steps.shape) if torch.is_tensor(history_age_steps) else None
            raise ValueError(
                "history_age_steps must be [B,K] aligned with heatmap_logits, "
                f"got {shape}"
            )
        if history_age_steps.dtype == torch.bool:
            raise TypeError("history_age_steps must be numeric, not bool")
        mask = StructuredHeatmapTokenizer._normalize_history_mask(history_mask)
        if tuple(mask.shape) != expected_history:
            raise ValueError(
                "history_mask must align with heatmap_logits: "
                f"expected {expected_history}, got {tuple(mask.shape)}"
            )
        devices = {
            heatmap_logits.device,
            visibility_logits.device,
            mask.device,
            history_age_steps.device,
        }
        if len(devices) != 1:
            raise ValueError("all StructuredHeatmapTokenizer inputs must share one device")
        if not torch.isfinite(heatmap_logits).all():
            raise ValueError("heatmap_logits contains non-finite values")
        if not torch.isfinite(visibility_logits).all():
            raise ValueError("visibility_logits contains non-finite values")
        age_float = history_age_steps.float()
        if not torch.isfinite(age_float).all() or torch.any(age_float < 0):
            raise ValueError("history_age_steps must be finite and non-negative")
        return int(batch_size), int(num_history), mask

    @staticmethod
    def _stable_history_rank(
        history_age_steps: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return oldest=0, newest=1 ranks, ignoring invalid slots.

        Stable sorting makes equal-age histories retain slot order. Rank is
        metadata rather than a differentiable feature source.
        """

        ranks = torch.zeros_like(history_age_steps, dtype=torch.float32)
        with torch.no_grad():
            for batch_index in range(history_mask.shape[0]):
                valid_indices = torch.nonzero(
                    history_mask[batch_index], as_tuple=False
                ).flatten()
                count = int(valid_indices.numel())
                if count <= 1:
                    continue
                valid_ages = history_age_steps[batch_index].index_select(
                    0, valid_indices
                )
                age_order = torch.argsort(
                    valid_ages,
                    descending=True,
                    stable=True,
                )
                chronological_indices = valid_indices.index_select(0, age_order)
                chronological_rank = torch.linspace(
                    0.0,
                    1.0,
                    count,
                    device=history_age_steps.device,
                    dtype=torch.float32,
                )
                ranks[batch_index].scatter_(
                    0,
                    chronological_indices,
                    chronological_rank,
                )
        return ranks

    def _empty_output(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.zeros(
                batch_size, 0, self.token_dim, device=device, dtype=torch.float32
            ),
            "token_mask": torch.zeros(
                batch_size, 0, device=device, dtype=torch.bool
            ),
            "coarse_probabilities": torch.zeros(
                batch_size, 0, 4, *COARSE_SIZE, device=device, dtype=torch.float32
            ),
            "spatial_statistics": torch.zeros(
                batch_size,
                0,
                4,
                len(SPATIAL_STATISTIC_NAMES),
                device=device,
                dtype=torch.float32,
            ),
            "view_probabilities": torch.zeros(
                batch_size, 0, 4, device=device, dtype=torch.float32
            ),
            "none_probability": torch.zeros(
                batch_size, 0, device=device, dtype=torch.float32
            ),
            "normalized_age": torch.zeros(
                batch_size, 0, device=device, dtype=torch.float32
            ),
            "history_rank": torch.zeros(
                batch_size, 0, device=device, dtype=torch.float32
            ),
            "structured_features": torch.zeros(
                batch_size,
                0,
                4,
                STRUCTURED_FEATURE_DIM,
                device=device,
                dtype=torch.float32,
            ),
        }

    def forward(
        self,
        heatmap_logits: torch.Tensor,
        visibility_logits: torch.Tensor,
        history_mask: torch.Tensor,
        history_age_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_history, mask = self._validate_inputs(
            heatmap_logits,
            visibility_logits,
            history_mask,
            history_age_steps,
        )
        if num_history == 0:
            return self._empty_output(batch_size, heatmap_logits.device)

        with _autocast_disabled(heatmap_logits):
            logits = heatmap_logits.float()
            visibility = visibility_logits.float()
            ages = history_age_steps.float()
            mask_f = mask.float()
            view_mask = mask[:, :, None]
            spatial_mask = view_mask[:, :, :, None, None]

            spatial_probability = torch.softmax(
                logits.reshape(batch_size, num_history, 4, -1),
                dim=-1,
            ).reshape_as(logits)

            flat_probability = spatial_probability.reshape(
                batch_size * num_history * 4,
                1,
                *HEATMAP_SIZE,
            )
            coarse_probability = F.avg_pool2d(
                flat_probability,
                kernel_size=(8, 8),
                stride=(8, 8),
            ) * 64.0
            coarse_probability = coarse_probability.reshape(
                batch_size,
                num_history,
                4,
                *COARSE_SIZE,
            )

            grid_x = self._grid_x.to(device=logits.device)
            grid_y = self._grid_y.to(device=logits.device)
            mean_x = (spatial_probability * grid_x).sum(dim=(-2, -1))
            mean_y = (spatial_probability * grid_y).sum(dim=(-2, -1))
            delta_x = grid_x - mean_x[..., None, None]
            delta_y = grid_y - mean_y[..., None, None]
            var_x = (
                spatial_probability * delta_x.square()
            ).sum(dim=(-2, -1))
            var_y = (
                spatial_probability * delta_y.square()
            ).sum(dim=(-2, -1))
            cov_xy = (
                spatial_probability * delta_x * delta_y
            ).sum(dim=(-2, -1))
            entropy = -(
                spatial_probability
                * spatial_probability.clamp_min(
                    self.probability_epsilon
                ).log()
            ).sum(dim=(-2, -1)) / math.log(
                HEATMAP_SIZE[0] * HEATMAP_SIZE[1]
            )
            peak = spatial_probability.amax(dim=(-2, -1))
            spatial_statistics = torch.stack(
                (mean_x, mean_y, var_x, var_y, cov_xy, entropy, peak),
                dim=-1,
            )

            none_logit = torch.zeros(
                batch_size,
                num_history,
                1,
                device=visibility.device,
                dtype=torch.float32,
            )
            view_none_probability = torch.softmax(
                torch.cat((none_logit, visibility), dim=-1),
                dim=-1,
            )
            none_probability = view_none_probability[..., 0]
            view_probability = view_none_probability[..., 1:]

            normalized_age = torch.log1p(
                ages.clamp(max=self.age_scale_steps)
            ) / math.log1p(self.age_scale_steps)
            history_rank = self._stable_history_rank(ages, mask)

            # Diagnostics use semantic neutral values for invalid histories.
            coarse_probability = coarse_probability * spatial_mask
            spatial_statistics = (
                spatial_statistics * view_mask[..., None]
            )
            view_probability = view_probability * view_mask
            none_probability = (
                none_probability * mask_f + (1.0 - mask_f)
            )
            normalized_age = normalized_age * mask_f
            history_rank = history_rank * mask_f

            yaw = self._view_yaw_sin_cos.to(
                device=logits.device
            ).reshape(1, 1, 4, 2)
            yaw = yaw.expand(batch_size, num_history, -1, -1)
            age_feature = normalized_age[:, :, None, None].expand(
                -1, -1, 4, -1
            )
            rank_feature = history_rank[:, :, None, None].expand(
                -1, -1, 4, -1
            )
            none_feature = none_probability[:, :, None, None].expand(
                -1, -1, 4, -1
            )

            structured_features = torch.cat(
                (
                    coarse_probability.flatten(-2),
                    spatial_statistics,
                    visibility.unsqueeze(-1),
                    view_probability.unsqueeze(-1),
                    none_feature,
                    yaw,
                    age_feature,
                    rank_feature,
                ),
                dim=-1,
            )
            if structured_features.shape[-1] != STRUCTURED_FEATURE_DIM:
                raise AssertionError(
                    "internal structured feature width mismatch: "
                    f"{structured_features.shape[-1]} != "
                    f"{STRUCTURED_FEATURE_DIM}"
                )
            structured_features = (
                structured_features * view_mask[..., None]
            )

            # Direct reshape from [B,K,4,D] locks history-major order.
            token_mask = mask.repeat_interleave(4, dim=1)
            tokens = self.shared_mlp(
                structured_features.reshape(
                    batch_size,
                    num_history * 4,
                    STRUCTURED_FEATURE_DIM,
                )
            )
            tokens = tokens * token_mask.unsqueeze(-1)

            key_padding_mask = ~token_mask
            all_invalid = ~token_mask.any(dim=1)
            if all_invalid.any():
                # MultiheadAttention returns NaN when every key is masked. Give
                # such rows one zero sentinel, then mask the row again.
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_invalid, 0] = False
            tokens = self.temporal_transformer(
                tokens,
                src_key_padding_mask=key_padding_mask,
            )
            tokens = tokens * token_mask.unsqueeze(-1)

        return {
            "tokens": tokens.float(),
            "token_mask": token_mask,
            "coarse_probabilities": coarse_probability.float(),
            "spatial_statistics": spatial_statistics.float(),
            "view_probabilities": view_probability.float(),
            "none_probability": none_probability.float(),
            "normalized_age": normalized_age.float(),
            "history_rank": history_rank.float(),
            "structured_features": structured_features.float(),
        }


__all__ = [
    "COARSE_SIZE",
    "HEATMAP_DIRECTION_ORDER",
    "HEATMAP_SIZE",
    "SPATIAL_STATISTIC_NAMES",
    "STRUCTURED_FEATURE_DIM",
    "StructuredHeatmapTokenizer",
]
