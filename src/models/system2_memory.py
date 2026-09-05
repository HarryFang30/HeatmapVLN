"""Project the Past Head memory into System2's token embedding space.

Every EXP-01..EXP-12 result describes a model in which the history memory
``M_t`` reaches only the *execution* layer: it is injected into ``Z``, which
conditions a local controller that walks toward a pixel goal System2 already
chose.  EXP-05/EXP-07 measured the consequence (closed-loop parity), and EXP-12
found the states where a memory would have mattered are exactly the ones where
System2 itself answers "front" and is wrong.

This module moves the injection point.  ``M_t`` becomes ``num_tokens`` extra
token embeddings inside System2's own prompt, so the language model can attend
to where the robot has been while it is still deciding where to go.  Nothing
else about the released prompt changes.

``mode`` selects the arm, and the arms are constructed to be exactly
comparable:

``memory``
    the real thing: each history slot is projected into an embedding.
``constant``
    the control.  Identical token count, identical position, identical
    trainable-parameter budget, but the embeddings do not depend on ``M_t`` at
    all.  Any gain the ``memory`` arm shows over this one cannot be explained by
    "the prompt got longer" or "System2 was fine-tuned on DAgger data", which
    is the confound that would otherwise sink the claim.
``off``
    no memory tokens are emitted; the caller must not place placeholders.

Padded history slots (``mask == 0``) get a learned ``absent`` embedding rather
than a zero vector, so "I have no eighth memory" is representable instead of
being silently indistinguishable from "my eighth memory is the zero vector".
"""

from __future__ import annotations

import torch
import torch.nn as nn

MEMORY_MODES = ("memory", "constant", "off")


class System2MemoryTokens(nn.Module):
    """Turn ``[B, K, memory_dim]`` history memory into ``[B, K, embed_dim]``."""

    def __init__(
        self,
        *,
        memory_dim: int = 256,
        embed_dim: int = 3584,
        num_tokens: int = 8,
        mode: str = "memory",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if mode not in MEMORY_MODES:
            raise ValueError(f"mode must be one of {MEMORY_MODES}, got {mode!r}")
        if memory_dim <= 0 or embed_dim <= 0 or num_tokens <= 0:
            raise ValueError("memory_dim, embed_dim and num_tokens must be positive")
        self.mode = str(mode)
        self.memory_dim = int(memory_dim)
        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)

        self.memory_norm = nn.LayerNorm(memory_dim)
        self.projection = nn.Linear(memory_dim, embed_dim)
        self.slot_embedding = nn.Parameter(torch.empty(num_tokens, embed_dim))
        self.absent_embedding = nn.Parameter(torch.empty(embed_dim))
        # Present in every mode so the two arms have the same parameter count
        # and the same optimizer state shape; only ``constant`` reads it.
        self.constant_embedding = nn.Parameter(torch.empty(num_tokens, embed_dim))

        nn.init.normal_(self.projection.weight, std=init_std)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.slot_embedding, std=init_std)
        nn.init.normal_(self.absent_embedding, std=init_std)
        nn.init.normal_(self.constant_embedding, std=init_std)

    def forward(
        self,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if self.mode == "off":
            raise RuntimeError(
                "System2MemoryTokens.mode='off' emits no tokens; the collator "
                "must not place memory placeholders in the prompt"
            )

        if self.mode == "constant":
            if batch_size is None:
                if memory is None:
                    raise ValueError(
                        "the constant arm needs either a batch_size or a memory "
                        "tensor to size its output"
                    )
                batch_size = int(memory.shape[0])
            tokens = self.constant_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            return tokens.contiguous()

        if memory is None or memory_mask is None:
            raise ValueError("mode='memory' requires both memory and memory_mask")
        if memory.ndim != 3 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory must be [B,K,{self.memory_dim}], got {tuple(memory.shape)}"
            )
        if memory.shape[1] != self.num_tokens:
            raise ValueError(
                f"memory carries {memory.shape[1]} history slots but the prompt "
                f"reserves {self.num_tokens} memory tokens; they must match"
            )
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError(
                f"memory_mask must be [B,K], got {tuple(memory_mask.shape)}"
            )

        weight_dtype = self.projection.weight.dtype
        projected = self.projection(self.memory_norm(memory.to(dtype=weight_dtype)))
        valid = memory_mask.to(device=projected.device, dtype=torch.bool).unsqueeze(-1)
        absent = self.absent_embedding.to(dtype=projected.dtype).view(1, 1, -1)
        tokens = torch.where(valid, projected, absent.expand_as(projected))
        return tokens + self.slot_embedding.to(dtype=projected.dtype).unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, num_tokens={self.num_tokens}, "
            f"memory_dim={self.memory_dim}, embed_dim={self.embed_dim}"
        )


__all__ = ["MEMORY_MODES", "System2MemoryTokens"]
