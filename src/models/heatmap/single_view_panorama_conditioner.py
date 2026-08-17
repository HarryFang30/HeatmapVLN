"""Convert one observed front-view feature map into four decoder slots.

This is not an image-to-panorama reconstruction network.  Side/back pixels are
unobserved, so pretending that a learned warp can recover their true scene
content creates a misleading shortcut.  The conditioner instead preserves the
front feature map exactly and represents the three unobserved directions with
trainable canonical query maps.  A zero-initialized global-context path can
later condition those query maps on what is visible in front.

Direction convention is fixed to the corrected InternNav/heatmap geometry:

    front = 0 degrees, right = -90, back = 180, left = +90
"""

from __future__ import annotations

import torch
import torch.nn as nn

VIEW_NAMES = ("front", "right", "back", "left")
VIEW_ANGLES_DEGREES = (0.0, -90.0, 180.0, 90.0)


class SingleViewPanoramaConditioner(nn.Module):
    """Build four directional feature slots from a true front feature map.

    Args:
        channels: Input/output channel count.
        spatial_size: Expected square feature-map size (16 for ViT fusion or
            8 for LLM fusion in the existing heatmap decoder).
        use_global_context: Add a front-image pooled context path to the three
            unobserved directions.  Its gate is initialized to zero so warm
            started panoramic decoder weights initially see stable canonical
            side/back inputs.
        query_init_std: Initialization scale for canonical query maps.

    Input:
        ``front``: ``[B,C,H,W]``.

    Output:
        ``[B,4,C,H,W]`` in ``front/right/back/left`` order.  View zero is
        bit-for-bit the input tensor (subject only to ``torch.stack``); it is
        never projected or spatially warped.
    """

    def __init__(
        self,
        channels: int,
        spatial_size: int,
        *,
        use_global_context: bool = True,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if channels <= 0 or spatial_size <= 0:
            raise ValueError("channels and spatial_size must be positive")
        if query_init_std < 0:
            raise ValueError("query_init_std must be non-negative")

        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.use_global_context = bool(use_global_context)

        # There are only three unobserved slots.  The front slot is always the
        # real feature map and intentionally has no learned replacement.
        self.canonical_queries = nn.Parameter(torch.empty(3, self.channels, self.spatial_size, self.spatial_size))
        nn.init.normal_(self.canonical_queries, std=float(query_init_std))

        # sin/cos at first and second harmonics keeps the 180-degree direction
        # distinct and makes the angular convention explicit in the state.
        self.direction_encoder = nn.Sequential(
            nn.Linear(4, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, self.channels),
        )

        angles = torch.tensor(VIEW_ANGLES_DEGREES, dtype=torch.float32)
        radians = torch.deg2rad(angles)
        angular_basis = torch.stack(
            [
                torch.sin(radians),
                torch.cos(radians),
                torch.sin(2.0 * radians),
                torch.cos(2.0 * radians),
            ],
            dim=-1,
        )
        self.register_buffer("direction_angles_degrees", angles, persistent=True)
        self.register_buffer("direction_angular_basis", angular_basis, persistent=False)

        if self.use_global_context:
            self.context_projector = nn.Sequential(
                nn.LayerNorm(self.channels),
                nn.Linear(self.channels, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, self.channels),
            )
            # One per unobserved direction and channel.  Zero initialization
            # makes context opt-in through training, avoiding a noisy initial
            # perturbation when the legacy heatmap head is warm started.
            self.context_gate = nn.Parameter(torch.zeros(3, self.channels))
        else:
            self.context_projector = None
            self.register_parameter("context_gate", None)

    def forward(self, front: torch.Tensor) -> torch.Tensor:
        if front.ndim != 4:
            raise ValueError(f"front must be [B,C,H,W], got {tuple(front.shape)}")
        batch_size, channels, height, width = front.shape
        expected = (self.channels, self.spatial_size, self.spatial_size)
        if (channels, height, width) != expected:
            raise ValueError(f"front has [C,H,W]={(channels, height, width)}, expected {expected}")

        direction_features = self.direction_encoder(
            self.direction_angular_basis.to(device=front.device, dtype=front.dtype)
        )
        unobserved = self.canonical_queries.to(dtype=front.dtype)
        unobserved = unobserved + direction_features[1:, :, None, None]
        unobserved = unobserved.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        if self.context_projector is not None:
            pooled_front = front.mean(dim=(-2, -1))
            context = self.context_projector(pooled_front)
            gated = context[:, None, :] * torch.tanh(self.context_gate)[None, :, :]
            unobserved = unobserved + gated[:, :, :, None, None]

        return torch.cat([front.unsqueeze(1), unobserved], dim=1)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, spatial_size={self.spatial_size}, "
            f"views={VIEW_NAMES}, angles={VIEW_ANGLES_DEGREES}, "
            f"use_global_context={self.use_global_context}"
        )
