"""Project the Past Head memory -- or the raw history geometry -- into System2's token embedding space.

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
comparable -- every mode owns the same parameter tensors, only the ones it
reads receive gradient:

``memory``
    each history slot's ``M_t`` vector is projected into an embedding.
``geometry``
    each history slot's relative pose (forward, left, cos yaw, sin yaw) from the
    odometry module is sinusoidally encoded and projected.  EXP-13-A/EXP-15/EXP-16
    showed the decision-relevant content of ``M_t`` is this pose, so the arm
    gives System2 the odometry directly and lets the fine-tune do the cognition.
    ``pose_dropout`` blanks all K pose tokens for a fraction of training samples
    (the ``absent`` embedding), so the model can also be evaluated without
    odometry -- that reading is the "how much geometry did the VLA internalise"
    number.  ``pose_noise_*`` perturb the training poses the way EXP-15 did
    (Gaussian metres on the offsets, a Gaussian yaw rotation of the (cos, sin)
    pair, optionally scaled by sqrt(1 + age) so older slots drift more), because
    the sealed training poses are simulator truth while deployment reads AMB3R.
``constant``
    the control.  Identical token count, identical position, identical
    trainable-parameter budget, but the embeddings do not depend on ``M_t`` or
    the poses at all.  Any gain the other arms show over this one cannot be
    explained by "the prompt got longer" or "System2 was fine-tuned on DAgger
    data", which is the confound that would otherwise sink the claim.
``off``
    no memory tokens are emitted; the caller must not place placeholders.

Padded history slots (``mask == 0``) get a learned ``absent`` embedding rather
than a zero vector, so "I have no eighth memory" is representable instead of
being silently indistinguishable from "my eighth memory is the zero vector".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

MEMORY_MODES = ("memory", "constant", "geometry", "off")
POSE_DIM = 4  # forward_m, left_m, cos_yaw, sin_yaw


def pose_pe_dim(num_freqs: int) -> int:
    return POSE_DIM * (1 + 2 * int(num_freqs))


def sinusoidal_pose_encoding(
    poses: torch.Tensor, *, num_freqs: int, max_spatial_range: float
) -> torch.Tensor:
    """The Past Head's own pose encoding, kept identical so the two arms read the same numbers.

    ``poses`` is ``[..., 4]``; the result is ``[..., 4 * (1 + 2 * num_freqs)]``:
    the normalised values followed by their sines and cosines at ``num_freqs``
    octave-spaced frequencies (see ``src/models/heatmap/trajectory_attention.py``).
    """
    if poses.shape[-1] != POSE_DIM:
        raise ValueError(f"poses must end with {POSE_DIM} channels, got {tuple(poses.shape)}")
    x_norm = poses / float(max_spatial_range)
    freqs = torch.arange(int(num_freqs), device=poses.device, dtype=poses.dtype)
    freqs = 2.0 * math.pi * (2.0 ** freqs)
    angles = x_norm.unsqueeze(-1) * freqs
    return torch.cat([x_norm, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


def perturb_rel_poses(
    poses: torch.Tensor,
    *,
    translation_m: float,
    rotation_deg: float,
    ages: torch.Tensor | None,
    drift: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """EXP-15's pose-noise model in torch: additive metres, a yaw rotation of (cos, sin).

    ``poses`` is ``[B, K, 4]``.  With ``drift`` the per-slot sigma is scaled by
    ``sqrt(1 + age_steps)``.  The yaw pair is *rotated*, never perturbed
    component-wise, so its unit norm survives and a reader cannot detect the
    corruption instead of the pose error it stands for.
    """
    out = poses.clone()
    batch, slots, _ = out.shape
    if drift:
        if ages is None:
            raise ValueError("drift noise needs history_age_steps")
        scale = torch.sqrt(1.0 + ages.to(device=out.device, dtype=out.dtype).reshape(batch, slots))
    else:
        scale = torch.ones((batch, slots), device=out.device, dtype=out.dtype)
    if translation_m > 0.0:
        noise = torch.randn((batch, slots, 2), device=out.device, dtype=out.dtype, generator=generator)
        out[:, :, :2] = out[:, :, :2] + noise * float(translation_m) * scale.unsqueeze(-1)
    if rotation_deg > 0.0:
        delta = math.radians(float(rotation_deg)) * scale * torch.randn(
            (batch, slots), device=out.device, dtype=out.dtype, generator=generator
        )
        cos_d, sin_d = torch.cos(delta), torch.sin(delta)
        cos_y, sin_y = out[:, :, 2].clone(), out[:, :, 3].clone()
        out[:, :, 2] = cos_y * cos_d - sin_y * sin_d
        out[:, :, 3] = sin_y * cos_d + cos_y * sin_d
    return out


class System2MemoryTokens(nn.Module):
    """Turn ``[B, K, memory_dim]`` history memory or ``[B, K, 4]`` poses into ``[B, K, embed_dim]``."""

    def __init__(
        self,
        *,
        memory_dim: int = 256,
        embed_dim: int = 3584,
        num_tokens: int = 8,
        mode: str = "memory",
        init_std: float = 0.02,
        pose_num_freqs: int = 16,
        pose_max_range: float = 10.0,
        pose_dropout: float = 0.0,
        pose_noise_translation_m: float = 0.0,
        pose_noise_rotation_deg: float = 0.0,
        pose_noise_drift: bool = True,
    ) -> None:
        super().__init__()
        if mode not in MEMORY_MODES:
            raise ValueError(f"mode must be one of {MEMORY_MODES}, got {mode!r}")
        if memory_dim <= 0 or embed_dim <= 0 or num_tokens <= 0:
            raise ValueError("memory_dim, embed_dim and num_tokens must be positive")
        if pose_num_freqs <= 0 or pose_max_range <= 0:
            raise ValueError("pose_num_freqs and pose_max_range must be positive")
        if not 0.0 <= float(pose_dropout) < 1.0:
            raise ValueError("pose_dropout must be in [0, 1)")
        if float(pose_noise_translation_m) < 0.0 or float(pose_noise_rotation_deg) < 0.0:
            raise ValueError("pose noise sigmas must be >= 0")
        self.mode = str(mode)
        self.memory_dim = int(memory_dim)
        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)
        self.pose_num_freqs = int(pose_num_freqs)
        self.pose_max_range = float(pose_max_range)
        self.pose_dropout = float(pose_dropout)
        self.pose_noise_translation_m = float(pose_noise_translation_m)
        self.pose_noise_rotation_deg = float(pose_noise_rotation_deg)
        self.pose_noise_drift = bool(pose_noise_drift)

        self.memory_norm = nn.LayerNorm(memory_dim)
        self.projection = nn.Linear(memory_dim, embed_dim)
        self.slot_embedding = nn.Parameter(torch.empty(num_tokens, embed_dim))
        self.absent_embedding = nn.Parameter(torch.empty(embed_dim))
        # Present in every mode so the arms have the same parameter set and the
        # same optimizer state shape; only ``constant`` reads it.
        self.constant_embedding = nn.Parameter(torch.empty(num_tokens, embed_dim))
        # Likewise present in every mode; only ``geometry`` reads it.
        self.geometry_projection = nn.Linear(pose_pe_dim(self.pose_num_freqs), embed_dim)

        nn.init.normal_(self.projection.weight, std=init_std)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.geometry_projection.weight, std=init_std)
        nn.init.zeros_(self.geometry_projection.bias)
        nn.init.normal_(self.slot_embedding, std=init_std)
        nn.init.normal_(self.absent_embedding, std=init_std)
        nn.init.normal_(self.constant_embedding, std=init_std)

    # ------------------------------------------------------------------ helpers
    def _slot(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        return self.slot_embedding.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _absent(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        absent = self.absent_embedding.to(dtype=dtype).view(1, 1, -1)
        return absent.expand(batch_size, self.num_tokens, -1)

    def _check_mask(self, mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if mask.shape != (batch_size, self.num_tokens):
            raise ValueError(
                f"history mask must be [B,{self.num_tokens}], got {tuple(mask.shape)}"
            )
        return mask.to(dtype=torch.bool)

    @property
    def pose_noise_enabled(self) -> bool:
        return self.pose_noise_translation_m > 0.0 or self.pose_noise_rotation_deg > 0.0

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        *,
        batch_size: int | None = None,
        history_rel_poses: torch.Tensor | None = None,
        history_age_steps: torch.Tensor | None = None,
        force_no_pose: bool = False,
    ) -> torch.Tensor:
        if self.mode == "off":
            raise RuntimeError(
                "System2MemoryTokens.mode='off' emits no tokens; the collator "
                "must not place memory placeholders in the prompt"
            )

        if self.mode == "constant":
            if batch_size is None:
                if memory is None and history_rel_poses is None:
                    raise ValueError(
                        "the constant arm needs a batch_size, a memory tensor or a "
                        "pose tensor to size its output"
                    )
                batch_size = int((memory if memory is not None else history_rel_poses).shape[0])
            tokens = self.constant_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            return tokens.contiguous()

        if self.mode == "geometry":
            return self.forward_geometry(
                history_rel_poses,
                memory_mask,
                history_age_steps=history_age_steps,
                force_no_pose=force_no_pose,
            )

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
        tokens = torch.where(valid, projected, self._absent(projected.shape[0], projected.dtype))
        return tokens + self._slot(projected.shape[0], projected.dtype)

    def forward_geometry(
        self,
        history_rel_poses: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        *,
        history_age_steps: torch.Tensor | None = None,
        force_no_pose: bool = False,
    ) -> torch.Tensor:
        """Pose tokens: sinusoidal encoding -> projection, absent where the slot is padded.

        ``force_no_pose`` blanks every slot (evaluation without odometry).  During
        training, ``pose_dropout`` does the same for a Bernoulli fraction of the
        batch so that evaluation state is in-distribution, and ``pose_noise_*``
        perturb the (simulator-true) training poses towards deployment error.
        """
        if self.mode != "geometry":
            raise RuntimeError("forward_geometry is only valid in mode='geometry'")
        if history_rel_poses is None or history_mask is None:
            raise ValueError("mode='geometry' requires history_rel_poses and a history mask")
        if history_rel_poses.ndim != 3 or history_rel_poses.shape[-1] != POSE_DIM:
            raise ValueError(
                f"history_rel_poses must be [B,K,{POSE_DIM}], got {tuple(history_rel_poses.shape)}"
            )
        if history_rel_poses.shape[1] != self.num_tokens:
            raise ValueError(
                f"history_rel_poses carries {history_rel_poses.shape[1]} slots but the "
                f"prompt reserves {self.num_tokens} memory tokens; they must match"
            )
        batch_size = int(history_rel_poses.shape[0])
        weight_dtype = self.geometry_projection.weight.dtype
        device = self.geometry_projection.weight.device
        poses = history_rel_poses.to(device=device, dtype=torch.float32)
        if not torch.isfinite(poses).all():
            raise ValueError("history_rel_poses contains non-finite values")
        if self.training and self.pose_noise_enabled:
            poses = perturb_rel_poses(
                poses,
                translation_m=self.pose_noise_translation_m,
                rotation_deg=self.pose_noise_rotation_deg,
                ages=history_age_steps,
                drift=self.pose_noise_drift,
            )
        encoded = sinusoidal_pose_encoding(
            poses, num_freqs=self.pose_num_freqs, max_spatial_range=self.pose_max_range
        ).to(dtype=weight_dtype)
        projected = self.geometry_projection(encoded)
        valid = self._check_mask(history_mask.to(device=device), batch_size)
        if force_no_pose:
            valid = torch.zeros_like(valid)
        elif self.training and self.pose_dropout > 0.0:
            keep = torch.rand(batch_size, device=device) >= self.pose_dropout
            valid = valid & keep.unsqueeze(1)
        tokens = torch.where(
            valid.unsqueeze(-1), projected, self._absent(batch_size, projected.dtype)
        )
        return tokens + self._slot(batch_size, projected.dtype)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, num_tokens={self.num_tokens}, "
            f"memory_dim={self.memory_dim}, embed_dim={self.embed_dim}, "
            f"pose_pe_dim={pose_pe_dim(self.pose_num_freqs)}, pose_dropout={self.pose_dropout}, "
            f"pose_noise=({self.pose_noise_translation_m} m, {self.pose_noise_rotation_deg} deg, drift={self.pose_noise_drift})"
        )
