"""Single-view input, four-direction heatmap decoder.

The module deliberately keeps the reusable panoramic heatmap submodule names:

``vit_dpt_fusion``, ``coarse``, and ``fine``.

Consequently those tensors can be warm started from a panoramic heatmap-only
checkpoint. The prior ``llm_dpt_fusion`` consumed language-layer states and is
intentionally not reused. Current-image spatial tensors come only from restored
ViT rasters; the native visual merger output is used only as the 3584-d history
query. The two panorama conditioners are new. No Qwen, language-model, LoRA,
System1, or navigation-adapter parameter is owned by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heatmap.dpt_lite_fusion import DPTLiteFusion
from src.models.heatmap.fine_localization import FineLocalization
from src.models.heatmap.trajectory_attention import TrajectoryGuidedAttention

from .native_single_view_feature_extractor import NativeSingleViewFeatures
from .single_view_panorama_conditioner import SingleViewPanoramaConditioner

LEGACY_REUSABLE_PREFIXES = (
    "vit_dpt_fusion.",
    "coarse.",
    "fine.",
)
DEFAULT_RESET_LEGACY_KEYS = (
    "coarse.proj_history.weight",
    "coarse.proj_history.bias",
)
OUTPUT_DIRECTION_ORDER = ("front", "right", "back", "left")
ARCHITECTURE_ID = "internnav_single_view_vision_only_four_direction_v2"
HISTORY_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)


class SingleViewFourDirectionHeatmapHead(nn.Module):
    """Decode native single-view features into panoramic heatmap supervision.

    Input geometry follows the existing trajectory-aware heatmap head:
    ``history_rel_poses[..., :] = (forward_m, left_m,
    cos(relative_yaw), sin(relative_yaw))`` in the current Habitat front
    camera frame, with camera ``-Z`` forward and positive yaw turning left.

    Output contract is unchanged:

    - ``visibility``: ``[B,K,4]`` logits
    - ``heatmaps``: ``[B,K,4,64,64]`` probabilities
    - ``heatmap_logits``: ``[B,K,4,64,64]`` logits

    Direction index order is front/right/back/left with corrected yaw angles
    0/-90/180/+90 degrees.
    """

    def __init__(
        self,
        *,
        c_vit: int = 1280,
        c_merged: int = 3584,
        c_fused: int = 256,
        vit_layer_indices: Sequence[int] = (7, 15, 23, 31),
        trajectory_num_freqs: int = 16,
        trajectory_num_heads: int = 4,
        trajectory_num_layers: int = 1,
        max_spatial_range: float = 10.0,
        conditioner_global_context: bool = True,
        coarse_logit_residual: bool = False,
        joint_panorama_inference: bool = True,
    ) -> None:
        super().__init__()
        self.architecture_id = ARCHITECTURE_ID
        self.output_direction_order = OUTPUT_DIRECTION_ORDER
        self.history_pose_convention = HISTORY_POSE_CONVENTION
        self.c_vit = int(c_vit)
        self.c_merged = int(c_merged)
        self.c_fused = int(c_fused)
        self.vit_layer_indices = tuple(int(i) for i in vit_layer_indices)
        self.joint_panorama_inference = bool(joint_panorama_inference)
        if not self.vit_layer_indices:
            raise ValueError("ViT layer list must not be empty")

        # Names and tensor shapes match the legacy panoramic head.
        self.vit_dpt_fusion = DPTLiteFusion(
            c_vit=self.c_vit,
            c_fused=self.c_fused,
            n_layers=len(self.vit_layer_indices),
        )
        # These are the only architectural additions. Expansion happens after
        # the one restored-raster ViT fusion, leaving legacy DPT tensors
        # reusable. The coarse branch is a deterministic 16->8 average pool;
        # no current spatial feature comes from the language model or visual
        # merger output.
        self.vit_panorama_conditioner = SingleViewPanoramaConditioner(
            channels=self.c_fused,
            spatial_size=16,
            use_global_context=conditioner_global_context,
        )
        self.coarse_panorama_conditioner = SingleViewPanoramaConditioner(
            channels=self.c_fused,
            spatial_size=8,
            use_global_context=conditioner_global_context,
        )

        self.coarse = TrajectoryGuidedAttention(
            c_llm=self.c_merged,
            c_fused=self.c_fused,
            num_freqs=int(trajectory_num_freqs),
            d_attn=self.c_fused,
            num_heads=int(trajectory_num_heads),
            num_layers=int(trajectory_num_layers),
            max_spatial_range=float(max_spatial_range),
        )
        self.fine = FineLocalization(
            c_fused=self.c_fused,
            coarse_logit_residual=bool(coarse_logit_residual),
        )

    def forward(
        self,
        features: NativeSingleViewFeatures,
        history_rel_poses: torch.Tensor,
        *,
        return_coarse: bool = False,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._validate_features(features, history_rel_poses)
        batch_size, num_history = features.history_mask.shape
        template = next(iter(features.current_vit.values()))

        if num_history == 0 and not return_memory_tokens:
            empty_visibility = template.new_empty((batch_size, 0, 4))
            empty_heatmaps = template.new_empty((batch_size, 0, 4, 64, 64))
            result = {
                "visibility": empty_visibility,
                "heatmaps": empty_heatmaps,
                "heatmap_logits": empty_heatmaps.clone(),
                "history_mask": features.history_mask,
                "heatmap_direction_order": OUTPUT_DIRECTION_ORDER,
            }
            if not self.training:
                result.update(self._build_inference_heatmaps(result))
            return result

        front_vit = self._fuse_current_layers(
            features.current_vit,
            self.vit_layer_indices,
            self.vit_dpt_fusion,
            expected_channels=self.c_vit,
            expected_spatial=16,
            label="ViT",
        )
        panoramic_vit = self.vit_panorama_conditioner(front_vit)
        front_coarse = F.adaptive_avg_pool2d(front_vit, output_size=(8, 8))
        panoramic_coarse_nchw = self.coarse_panorama_conditioner(front_coarse)
        panoramic_coarse = panoramic_coarse_nchw.permute(0, 1, 3, 4, 2).contiguous()

        if num_history == 0:
            empty_visibility = template.new_zeros((batch_size, 0, 4))
            empty_heatmaps = template.new_zeros((batch_size, 0, 4, 64, 64))
            result = {
                "visibility": empty_visibility,
                "heatmaps": empty_heatmaps,
                "heatmap_logits": empty_heatmaps.clone(),
                "history_mask": features.history_mask,
                "heatmap_direction_order": OUTPUT_DIRECTION_ORDER,
                "history_memory": template.new_zeros(
                    (batch_size, 0, self.c_fused)
                ),
                "history_memory_mask": features.history_mask,
                "history_spatial_memory": template.new_zeros(
                    (batch_size, 0, 4 * 8 * 8, self.c_fused)
                ),
                "panoramic_vit_features": panoramic_vit,
            }
            if not self.training:
                result.update(self._build_inference_heatmaps(result))
            return result

        coarse_results = self.coarse(
            panoramic_coarse,
            features.history_queries,
            history_rel_poses=history_rel_poses,
        )
        heatmaps, heatmap_logits = self.fine(
            vit_fused=panoramic_vit,
            coarse_heatmap=coarse_results["coarse_heatmap"],
            spatial_out=coarse_results["spatial_out"],
            return_logits=True,
        )

        # Padded history slots are neutralized at the decoder boundary.  Loss
        # code should still use history_mask; zeros here prevent diagnostics
        # and metrics from accidentally treating padding as a prediction.
        mask = features.history_mask.to(dtype=heatmaps.dtype)
        visibility = coarse_results["visibility"] * mask.unsqueeze(-1)
        spatial_mask = mask[:, :, None, None, None]
        heatmaps = heatmaps * spatial_mask
        heatmap_logits = heatmap_logits * spatial_mask

        result = {
            "visibility": visibility,
            "heatmaps": heatmaps,
            "heatmap_logits": heatmap_logits,
            "history_mask": features.history_mask,
            "heatmap_direction_order": OUTPUT_DIRECTION_ORDER,
        }
        if return_coarse:
            result["coarse_heatmap"] = coarse_results["coarse_heatmap"] * spatial_mask
            result["spatial_out"] = coarse_results["spatial_out"] * mask[:, :, None, None]
            result["panoramic_vit_features"] = panoramic_vit
            result["panoramic_coarse_features"] = panoramic_coarse
        if return_memory_tokens:
            result["history_memory"] = (
                coarse_results["history_memory"] * mask.unsqueeze(-1)
            )
            result["history_memory_mask"] = features.history_mask
            result["history_spatial_memory"] = (
                coarse_results["spatial_out"] * mask[:, :, None, None]
            )
            result["panoramic_vit_features"] = panoramic_vit
        if not self.training:
            result.update(self._build_inference_heatmaps(result))
        return result

    def trainable_head_modules(self) -> tuple[nn.Module, ...]:
        """Explicit optimizer whitelist for the single-view heatmap stage."""

        return (
            self.vit_dpt_fusion,
            self.vit_panorama_conditioner,
            self.coarse_panorama_conditioner,
            self.coarse,
            self.fine,
        )

    def _build_inference_heatmaps(
        self,
        result: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits = result["heatmap_logits"].float()
        visibility = result["visibility"].float()
        height, width = logits.shape[-2:]
        spatial = torch.softmax(
            logits.reshape(*logits.shape[:-2], height * width),
            dim=-1,
        ).reshape_as(logits)
        history_mask = result["history_mask"].to(dtype=spatial.dtype)
        if self.joint_panorama_inference:
            none_logit = torch.zeros(
                *visibility.shape[:-1],
                1,
                device=visibility.device,
                dtype=visibility.dtype,
            )
            view_none = torch.softmax(
                torch.cat((none_logit, visibility), dim=-1),
                dim=-1,
            )
            gated = spatial * view_none[..., 1:, None, None]
            return {
                "heatmaps_gated": gated * history_mask[:, :, None, None, None],
                "none_probability": (
                    view_none[..., 0] * history_mask + (1.0 - history_mask)
                ),
            }
        gated = spatial * torch.sigmoid(visibility)[..., None, None]
        return {
            "heatmaps_gated": gated * history_mask[:, :, None, None, None]
        }

    @staticmethod
    def _fuse_current_layers(
        layer_features: Mapping[int, torch.Tensor],
        layer_indices: Sequence[int],
        fusion: nn.Module,
        *,
        expected_channels: int,
        expected_spatial: int,
        label: str,
    ) -> torch.Tensor:
        inputs = []
        batch_size = None
        for layer_index in layer_indices:
            tensor = layer_features.get(layer_index)
            if tensor is None:
                raise RuntimeError(f"missing {label} feature layer {layer_index}")
            if tensor.ndim != 4:
                raise ValueError(f"{label} layer {layer_index} must be [B,C,H,W], got {tuple(tensor.shape)}")
            expected = (expected_channels, expected_spatial, expected_spatial)
            if tuple(tensor.shape[1:]) != expected:
                raise ValueError(
                    f"{label} layer {layer_index} has [C,H,W]={tuple(tensor.shape[1:])}, expected {expected}"
                )
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(f"{label} layers disagree on batch size")
            inputs.append(tensor)
        return fusion(inputs)

    def _validate_features(
        self,
        features: NativeSingleViewFeatures,
        history_rel_poses: torch.Tensor,
    ) -> None:
        if features.history_mask.ndim != 2:
            raise ValueError(f"history_mask must be [B,K], got {tuple(features.history_mask.shape)}")
        if features.history_mask.dtype != torch.bool:
            raise TypeError("history_mask must be bool")
        batch_size, num_history = features.history_mask.shape
        if features.history_queries.shape != (batch_size, num_history, self.c_merged):
            raise ValueError(
                "history_queries must be "
                f"{(batch_size, num_history, self.c_merged)}, got {tuple(features.history_queries.shape)}"
            )
        if history_rel_poses.shape != (batch_size, num_history, 4):
            raise ValueError(
                f"history_rel_poses must be {(batch_size, num_history, 4)}, got {tuple(history_rel_poses.shape)}"
            )
        if not torch.is_floating_point(history_rel_poses):
            raise TypeError("history_rel_poses must be floating point")
        if history_rel_poses.device != features.history_queries.device:
            raise ValueError("history_rel_poses and history_queries must be on the same device")

    def legacy_reusable_state_keys(
        self,
        *,
        include_text_anchor_history_projection: bool = False,
    ) -> tuple[str, ...]:
        """Return exact keys eligible for a heatmap-only legacy warm start.

        By default the two ``coarse.proj_history`` tensors are excluded. Their
        shape still matches, but the old model projected a text-anchor state
        while this model projects a visual-merger per-image mean.
        """

        return tuple(
            key
            for key in self.state_dict()
            if any(key.startswith(prefix) for prefix in LEGACY_REUSABLE_PREFIXES)
            and (include_text_anchor_history_projection or key not in DEFAULT_RESET_LEGACY_KEYS)
        )

    def new_single_view_state_keys(self) -> tuple[str, ...]:
        """Return conditioner tensors that have no panoramic counterpart."""

        return tuple(
            key
            for key in self.state_dict()
            if key.startswith("vit_panorama_conditioner.")
            or key.startswith("coarse_panorama_conditioner.")
            or key in DEFAULT_RESET_LEGACY_KEYS
        )
