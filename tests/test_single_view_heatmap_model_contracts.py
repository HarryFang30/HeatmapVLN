"""Canonical-import regression locks for the single-view heatmap model path."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.heatmap.native_single_view_feature_extractor import (
    NativeSingleViewFeatureExtractor,
    NativeSingleViewFeatures,
)
from src.models.heatmap.single_view_heatmap_decoder import (
    DEFAULT_RESET_LEGACY_KEYS,
    SingleViewFourDirectionHeatmapHead,
)
from src.models.heatmap.single_view_panorama_conditioner import (
    VIEW_ANGLES_DEGREES,
    VIEW_NAMES,
    SingleViewPanoramaConditioner,
)


class _PassBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


class _FakeVisual(nn.Module):
    """Minimal native-Qwen visual surface with explicit window packing."""

    def __init__(
        self,
        *,
        num_layers: int = 2,
        window_order: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_PassBlock() for _ in range(num_layers)])
        self.window_order = tuple(window_order)

    def get_window_index(
        self,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        indices: list[int] = []
        offset = 0
        for t, h, w in grid_thw.detach().cpu().tolist():
            group_count = int(t) * (int(h) // 2) * (int(w) // 2)
            if group_count != len(self.window_order):
                raise AssertionError(
                    "fake window order only supports four groups per image"
                )
            indices.extend(offset + index for index in self.window_order)
            offset += group_count
        return torch.tensor(indices, device=grid_thw.device), None

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)[0]
        merged = hidden.reshape(-1, 4, hidden.shape[-1]).mean(dim=1)
        window_index = self.get_window_index(grid_thw)[0]
        # Native Qwen returns merger tokens in restored image-raster order.
        return merged.index_select(0, torch.argsort(window_index))


class _FakeQwen(nn.Module):
    def __init__(
        self,
        *,
        window_order: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        inner = nn.Module()
        inner.visual = _FakeVisual(window_order=window_order)
        self.model = inner


def test_native_extractor_restores_window_packed_vit_raster() -> None:
    window_order = (2, 0, 3, 1)
    model = _FakeQwen(window_order=window_order).eval()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    extractor = NativeSingleViewFeatureExtractor(
        model,
        vit_layer_indices=(0,),
        vit_output_spatial=4,
        merged_output_spatial=2,
    )

    raster = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)
    raster_groups = raster.reshape(2, 2, 2, 2, 1)
    raster_groups = raster_groups.permute(0, 2, 1, 3, 4).reshape(4, 4, 1)
    packed_pixels = raster_groups[list(window_order)].reshape(16, 1)
    features = extractor.extract_from_pixels(
        pixel_values=packed_pixels,
        image_grid_thw=grid,
        num_histories=(0,),
    )

    restored = features.current_vit[0][0].permute(1, 2, 0)
    assert torch.equal(restored, raster)
    assert not restored.requires_grad
    extractor.remove_hooks()


def test_native_extractor_rejects_frozen_lora_parameter() -> None:
    model = _FakeQwen().eval()
    model.register_parameter(
        "lora_A",
        nn.Parameter(torch.zeros(1), requires_grad=False),
    )
    with pytest.raises(RuntimeError, match="LoRA/PEFT"):
        NativeSingleViewFeatureExtractor(model, vit_layer_indices=(0,))


def test_conditioner_locks_front_right_back_left_contract() -> None:
    torch.manual_seed(4)
    conditioner = SingleViewPanoramaConditioner(
        channels=8,
        spatial_size=5,
        use_global_context=False,
    )
    front = torch.randn(2, 8, 5, 5)
    output = conditioner(front)

    assert output.shape == (2, 4, 8, 5, 5)
    assert torch.equal(output[:, 0], front)
    assert VIEW_NAMES == ("front", "right", "back", "left")
    assert VIEW_ANGLES_DEGREES == (0.0, -90.0, 180.0, 90.0)
    assert tuple(conditioner.direction_angles_degrees.tolist()) == (
        0.0,
        -90.0,
        180.0,
        90.0,
    )


def _features(
    *,
    batch_size: int = 2,
    num_history: int = 3,
) -> NativeSingleViewFeatures:
    c_vit = 12
    c_merged = 16
    vit_layers = (0, 1)
    current_vit = {
        layer: torch.randn(batch_size, c_vit, 16, 16)
        for layer in vit_layers
    }
    current_merged = torch.randn(batch_size, c_merged, 8, 8)
    history_vit = {
        layer: torch.randn(batch_size, num_history, c_vit, 16, 16)
        for layer in vit_layers
    }
    history_merged = torch.randn(
        batch_size,
        num_history,
        c_merged,
        8,
        8,
    )
    history_queries = history_merged.mean(dim=(-2, -1))
    history_mask = torch.ones(batch_size, num_history, dtype=torch.bool)
    if batch_size > 1 and num_history > 1:
        history_mask[1, 1:] = False
        history_queries[1, 1:] = 0
    return NativeSingleViewFeatures(
        current_vit=current_vit,
        current_merged=current_merged,
        history_vit=history_vit,
        history_merged=history_merged,
        history_queries=history_queries,
        history_mask=history_mask,
    )


def _head() -> SingleViewFourDirectionHeatmapHead:
    return SingleViewFourDirectionHeatmapHead(
        c_vit=12,
        c_merged=16,
        c_fused=8,
        vit_layer_indices=(0, 1),
        trajectory_num_freqs=2,
        trajectory_num_heads=4,
        trajectory_num_layers=1,
        conditioner_global_context=True,
    )


def test_decoder_shape_padding_and_backward_contract() -> None:
    torch.manual_seed(12)
    features = _features()
    head = _head()
    relative_poses = torch.randn(2, 3, 4)
    result = head(features, relative_poses, return_coarse=True)

    assert result["visibility"].shape == (2, 3, 4)
    assert result["heatmaps"].shape == (2, 3, 4, 64, 64)
    assert result["heatmap_logits"].shape == (2, 3, 4, 64, 64)
    assert result["coarse_heatmap"].shape == (2, 3, 4, 8, 8)
    assert result["panoramic_vit_features"].shape == (2, 4, 8, 16, 16)
    assert result["panoramic_coarse_features"].shape == (2, 4, 8, 8, 8)
    assert torch.count_nonzero(result["visibility"][1, 1:]) == 0
    assert torch.count_nonzero(result["heatmap_logits"][1, 1:]) == 0

    loss = result["heatmap_logits"][features.history_mask].square().mean()
    loss = loss + result["visibility"][features.history_mask].square().mean()
    loss.backward()

    assert head.vit_dpt_fusion.align[0].weight.grad is not None
    assert head.coarse.proj_history.weight.grad is not None
    assert head.coarse.proj_traj.weight.grad is not None
    assert head.fine.refine[0].weight.grad is not None
    assert head.vit_panorama_conditioner.canonical_queries.grad is not None
    assert head.coarse_panorama_conditioner.canonical_queries.grad is not None
    assert all(not value.requires_grad for value in features.current_vit.values())
    assert not features.history_queries.requires_grad


def test_decoder_warmstart_state_excludes_old_text_anchor_projection() -> None:
    head = _head()
    reusable = set(head.legacy_reusable_state_keys())
    reset = set(DEFAULT_RESET_LEGACY_KEYS)
    new = set(head.new_single_view_state_keys())

    assert reusable
    assert reset.isdisjoint(reusable)
    assert reset <= new
    assert not any("llm_dpt_fusion" in key for key in head.state_dict())
    assert not any("lora" in key.lower() for key in head.state_dict())
    assert not any("adapter" in key.lower() for key in head.state_dict())
