"""Geometry-aware panoramic latent adapter for InternNav System 1.

The adapter translates frozen panoramic System-2 TRAJ hidden states into the
768-dim condition tokens consumed directly by NextDiT.  A compact geometry
token tells the translator how the structured panoramic goal maps to egocentric
heading, instead of asking the latent tokens to rediscover that mapping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

VIEW_ID_TO_INDEX: dict[str, int] = {
    "front": 0,
    "right": 1,
    "back": 2,
    "left": 3,
}

GEOMETRY_CONVENTION_TRAJECTORY = "trajectory_left_positive_v2"
GEOMETRY_CONVENTION_LEGACY_CAMERA = "camera_right_positive_v1"

_VIEW_CENTER_YAW_TRAJECTORY_RAD = torch.tensor(
    [0.0, -math.pi / 2.0, math.pi, math.pi / 2.0],
    dtype=torch.float32,
)
_VIEW_CENTER_YAW_LEGACY_CAMERA_RAD = -_VIEW_CENTER_YAW_TRAJECTORY_RAD


def view_ids_to_indices(
    view_ids: Sequence[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert canonical pano view ids to a LongTensor."""
    indices: list[int] = []
    for view_id in view_ids:
        key = str(view_id).lower()
        if key not in VIEW_ID_TO_INDEX:
            raise ValueError(f"Unsupported panoramic goal view_id: {view_id!r}")
        indices.append(VIEW_ID_TO_INDEX[key])
    return torch.tensor(indices, dtype=torch.long, device=device)


class PanoLatentSpaceAdapter(nn.Module):
    """Simple MLP that maps student VLM traj_hidden_states into teacher latent space.

    The adapter sits **before** ``cond_projector``::

        traj_hs (B, Q, 3584) → MLP → adapted_hs (B, Q, 3584)
            → cond_projector (frozen) → 768 → NextDiT

    This preserves InternNav's pre-trained cond_projector knowledge and only
    learns the student→teacher latent-space translation (pure style transfer).
    No geometry, no cross-attention — just a 2-layer MLP with residual.
    """

    def __init__(
        self,
        *,
        dim: int = 3584,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.dim),
        )
        # Small init so the adapter starts close to identity.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, student_latents: torch.Tensor) -> torch.Tensor:
        """Return adapted latents with the same shape as the input."""
        if student_latents.ndim != 3:
            raise ValueError(
                f"student_latents must be [B,Q,D], got {tuple(student_latents.shape)}"
            )
        if student_latents.shape[-1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {student_latents.shape[-1]}"
            )
        adapter_dtype = next(self.mlp[0].parameters()).dtype
        residual = student_latents.to(dtype=adapter_dtype)
        out = self.mlp(residual) + residual  # residual connection
        return out


class GeometryAwarePanoToNextDiTAdapter(nn.Module):
    """Decoder-style translator from pano student latents to NextDiT latents.

    Args:
        student_dim: Hidden size of the frozen Pano-System2 TRAJ tokens.
        adapter_dim: Internal Transformer width.
        output_dim: Condition width consumed by NextDiT.  This is 768 for the
            direct adapter path that skips ``cond_projector``.
        num_query: Number of output trajectory condition tokens.
        num_layers: Number of decoder layers.  Keep this configurable so v1 can
            choose either a one-layer or two-layer translator without changing
            checkpoint format.
    """

    def __init__(
        self,
        *,
        student_dim: int = 3584,
        adapter_dim: int = 768,
        output_dim: int = 768,
        num_query: int = 4,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        geometry_embed_dim: int = 64,
        horizontal_fov_deg: float = 90.0,
        geometry_convention: str = GEOMETRY_CONVENTION_TRAJECTORY,
    ) -> None:
        super().__init__()
        if num_query <= 0:
            raise ValueError("num_query must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.student_dim = int(student_dim)
        self.adapter_dim = int(adapter_dim)
        self.output_dim = int(output_dim)
        self.num_query = int(num_query)
        self.num_layers = int(num_layers)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.geometry_convention = str(geometry_convention)
        if self.geometry_convention not in {
            GEOMETRY_CONVENTION_TRAJECTORY,
            GEOMETRY_CONVENTION_LEGACY_CAMERA,
        }:
            raise ValueError(f"Unsupported geometry_convention={geometry_convention!r}")

        self.student_proj = nn.Linear(self.student_dim, self.adapter_dim)
        self.view_embedding = nn.Embedding(len(VIEW_ID_TO_INDEX), geometry_embed_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(geometry_embed_dim + 4, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
        )
        self.output_queries = nn.Parameter(torch.empty(self.num_query, self.adapter_dim))
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.adapter_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.adapter_dim)
        self.output_proj = nn.Linear(self.adapter_dim, self.output_dim)

    @staticmethod
    def geometry_scalars(
        view_indices: torch.Tensor,
        pixel_xy: torch.Tensor,
        image_hw: torch.Tensor,
        *,
        horizontal_fov_deg: float = 90.0,
        geometry_convention: str = GEOMETRY_CONVENTION_TRAJECTORY,
    ) -> torch.Tensor:
        """Return ``[x_norm, y_norm, sin(theta), cos(theta)]`` per sample.

        ``image_hw`` is ``[height, width]``.  The horizontal angle follows the
        local trajectory convention: front=0, right=-90, back=180,
        left=+90 degrees. Positive yaw points left because local trajectory
        coordinates are ``x=forward, y=left``.
        """
        if view_indices.ndim != 1:
            raise ValueError(f"view_indices must be [B], got {tuple(view_indices.shape)}")
        if pixel_xy.ndim != 2 or pixel_xy.shape[-1] != 2:
            raise ValueError(f"pixel_xy must be [B,2], got {tuple(pixel_xy.shape)}")
        if image_hw.ndim == 1:
            image_hw = image_hw.unsqueeze(0).expand(pixel_xy.shape[0], -1)
        if image_hw.ndim != 2 or image_hw.shape[-1] != 2:
            raise ValueError(f"image_hw must be [B,2] or [2], got {tuple(image_hw.shape)}")

        dtype = pixel_xy.dtype
        device = pixel_xy.device
        image_hw = image_hw.to(device=device, dtype=dtype).clamp_min(1.0)
        height = image_hw[:, 0]
        width = image_hw[:, 1]
        x_norm = pixel_xy[:, 0].to(dtype=dtype) / width
        y_norm = pixel_xy[:, 1].to(dtype=dtype) / height

        if geometry_convention == GEOMETRY_CONVENTION_TRAJECTORY:
            centers = _VIEW_CENTER_YAW_TRAJECTORY_RAD.to(device=device, dtype=dtype)
            pixel_sign = -1.0
        elif geometry_convention == GEOMETRY_CONVENTION_LEGACY_CAMERA:
            centers = _VIEW_CENTER_YAW_LEGACY_CAMERA_RAD.to(device=device, dtype=dtype)
            pixel_sign = 1.0
        else:
            raise ValueError(f"Unsupported geometry_convention={geometry_convention!r}")
        center_yaw = centers[view_indices.to(device=device)]
        fov_rad = torch.as_tensor(
            math.radians(horizontal_fov_deg),
            device=device,
            dtype=dtype,
        )
        # Image ``u`` grows to the right, which is negative local yaw under
        # the trajectory convention. The legacy convention is retained only
        # for loading old geometry-aware checkpoints without changing meaning.
        theta = center_yaw + pixel_sign * (x_norm - 0.5) * fov_rad
        return torch.stack([x_norm, y_norm, torch.sin(theta), torch.cos(theta)], dim=-1)

    def geometry_token(
        self,
        view_indices: torch.Tensor,
        pixel_xy: torch.Tensor,
        image_hw: torch.Tensor,
    ) -> torch.Tensor:
        scalars = self.geometry_scalars(
            view_indices,
            pixel_xy,
            image_hw,
            horizontal_fov_deg=self.horizontal_fov_deg,
            geometry_convention=self.geometry_convention,
        )
        view_emb = self.view_embedding(view_indices.to(device=pixel_xy.device))
        geom = torch.cat([view_emb.to(dtype=scalars.dtype), scalars], dim=-1)
        return self.geometry_mlp(geom).unsqueeze(1)

    def forward(
        self,
        student_latents: torch.Tensor,
        view_indices: torch.Tensor,
        pixel_xy: torch.Tensor,
        image_hw: torch.Tensor,
        *,
        goal_text_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if student_latents.ndim != 3:
            raise ValueError(
                f"student_latents must be [B,Q,D], got {tuple(student_latents.shape)}"
            )
        if student_latents.shape[-1] != self.student_dim:
            raise ValueError(
                f"Expected student_dim={self.student_dim}, got {student_latents.shape[-1]}"
            )

        adapter_dtype = self.output_queries.dtype
        student_memory = self.student_proj(student_latents.to(dtype=adapter_dtype))
        geom_memory = self.geometry_token(
            view_indices=view_indices,
            pixel_xy=pixel_xy.to(dtype=adapter_dtype),
            image_hw=image_hw.to(dtype=adapter_dtype),
        )
        memory_parts = [student_memory, geom_memory]
        if goal_text_hidden is not None:
            if goal_text_hidden.shape[-1] == self.student_dim:
                memory_parts.append(self.student_proj(goal_text_hidden.to(dtype=adapter_dtype)))
            elif goal_text_hidden.shape[-1] == self.adapter_dim:
                memory_parts.append(goal_text_hidden.to(dtype=adapter_dtype))
            else:
                raise ValueError(
                    "goal_text_hidden last dim must match student_dim or adapter_dim, "
                    f"got {goal_text_hidden.shape[-1]}"
                )
        memory = torch.cat(memory_parts, dim=1)

        queries = self.output_queries.unsqueeze(0).expand(student_latents.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(tgt=queries, memory=memory)
        out = self.output_proj(self.output_norm(queries))
        return out
