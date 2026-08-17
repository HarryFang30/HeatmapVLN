"""Low-dimensional, zero-initialized heatmap control for frozen NextDiT.

The adapter deliberately owns no native InternNav parameter. It receives the
post-native-attention hidden state as its query and structured heatmap tokens as
key/value memory. A per-head zero gate makes the augmented block an exact no-op
before control training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0


class HeatmapControlAdapter(nn.Module):
    """One independent FP32 heatmap cross-attention branch for a NextDiT block."""

    def __init__(
        self,
        *,
        model_dim: int = 384,
        control_dim: int = 128,
        num_heads: int = 4,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or control_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim, control_dim, and num_heads must be positive")
        if control_dim % num_heads != 0:
            raise ValueError(
                f"control_dim must be divisible by num_heads, got {control_dim} and {num_heads}"
            )

        self.model_dim = int(model_dim)
        self.control_dim = int(control_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.control_dim // self.num_heads

        self.query_norm = nn.LayerNorm(self.model_dim, eps=norm_eps)
        self.context_norm = nn.LayerNorm(self.control_dim, eps=norm_eps)
        self.attention = Attention(
            query_dim=self.model_dim,
            cross_attention_dim=self.control_dim,
            dim_head=self.head_dim,
            heads=self.num_heads,
            kv_heads=self.num_heads,
            qk_norm="layer_norm_across_heads",
            eps=norm_eps,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.gate = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.float()

    def _apply(self, fn, recurse: bool = True):
        """Follow device moves while keeping every floating control tensor FP32.

        NextDiTActionHead is normally moved wholesale to bf16. Overriding
        _apply is necessary because a parent module's to(dtype=...) otherwise
        casts children without calling their public to method.
        """

        super()._apply(fn, recurse=recurse)
        for parameter in self.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.float()
        for buffer in self.buffers():
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                buffer.data = buffer.data.float()
        return self

    def _prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        heatmap_tokens: torch.Tensor,
        heatmap_mask: torch.Tensor | None,
        heatmap_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.model_dim:
            raise ValueError(
                f"hidden_states must be [B,T,{self.model_dim}], got {tuple(hidden_states.shape)}"
            )
        if heatmap_tokens.ndim != 3 or heatmap_tokens.shape[-1] != self.control_dim:
            raise ValueError(
                f"heatmap_tokens must be [B,S,{self.control_dim}], got {tuple(heatmap_tokens.shape)}"
            )
        batch_size = hidden_states.shape[0]
        if heatmap_tokens.shape[0] != batch_size:
            raise ValueError("hidden_states and heatmap_tokens batch sizes disagree")
        if heatmap_tokens.device != hidden_states.device:
            raise ValueError("hidden_states and heatmap_tokens must be on the same device")

        token_count = heatmap_tokens.shape[1]
        if heatmap_mask is not None:
            if heatmap_mask.dtype != torch.bool:
                raise TypeError("heatmap_mask must be bool")
            if heatmap_mask.shape != (batch_size, token_count):
                raise ValueError(
                    f"heatmap_mask must be {(batch_size, token_count)}, got {tuple(heatmap_mask.shape)}"
                )
            if heatmap_mask.device != hidden_states.device:
                raise ValueError("heatmap_mask must be on the hidden-state device")
        if heatmap_valid is not None:
            if heatmap_valid.dtype != torch.bool:
                raise TypeError("heatmap_valid must be bool")
            if heatmap_valid.shape != (batch_size,):
                raise ValueError(
                    f"heatmap_valid must be {(batch_size,)}, got {tuple(heatmap_valid.shape)}"
                )
            if heatmap_valid.device != hidden_states.device:
                raise ValueError("heatmap_valid must be on the hidden-state device")

        if token_count == 0:
            safe_tokens = heatmap_tokens.new_zeros((batch_size, 1, self.control_dim))
            safe_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=hidden_states.device)
            effective_valid = torch.zeros((batch_size,), dtype=torch.bool, device=hidden_states.device)
        else:
            original_mask = (
                heatmap_mask
                if heatmap_mask is not None
                else torch.ones((batch_size, token_count), dtype=torch.bool, device=hidden_states.device)
            )
            has_token = original_mask.any(dim=1)
            effective_valid = has_token if heatmap_valid is None else (has_token & heatmap_valid)

            safe_mask = original_mask.clone()
            all_padding = ~has_token
            if all_padding.any():
                safe_mask[all_padding, 0] = True

            safe_tokens = heatmap_tokens.masked_fill(~original_mask.unsqueeze(-1), 0)
            safe_tokens = safe_tokens * effective_valid[:, None, None].to(dtype=safe_tokens.dtype)

        return (
            hidden_states.float(),
            safe_tokens.float(),
            safe_mask,
            effective_valid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        heatmap_tokens: torch.Tensor,
        *,
        heatmap_mask: torch.Tensor | None = None,
        heatmap_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return an FP32 [B,T,model_dim] residual control update."""

        query, context, safe_mask, effective_valid = self._prepare_inputs(
            hidden_states,
            heatmap_tokens,
            heatmap_mask,
            heatmap_valid,
        )
        query = self.query_norm(query)
        context = self.context_norm(context)
        per_head = self.attention(
            hidden_states=query,
            encoder_hidden_states=context,
            attention_mask=safe_mask,
            query_rotary_emb=None,
            key_rotary_emb=None,
        )
        expected = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_heads,
            self.head_dim,
        )
        if per_head.shape != expected:
            raise RuntimeError(
                f"heatmap attention returned {tuple(per_head.shape)}, expected {expected}"
            )

        per_head = per_head * self.gate.tanh().view(1, 1, self.num_heads, 1)
        delta = self.attention.to_out[0](per_head.flatten(-2))
        return delta * effective_valid[:, None, None].to(dtype=delta.dtype)
