from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heatmap.feature_extractor import FeatureExtractor

MERGE_SIZE = 2
WINDOW_GROUP_SIZE = 4


def _reference_window_index(
    grid_thw: torch.Tensor,
    *,
    merge_size: int = MERGE_SIZE,
    window_group_size: int = WINDOW_GROUP_SIZE,
) -> torch.Tensor:
    """Independent CPU reference for Qwen's valid window-cell traversal."""

    indices: list[int] = []
    group_offset = 0
    for grid_t, grid_h, grid_w in grid_thw.detach().cpu().tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        group_h = grid_h // merge_size
        group_w = grid_w // merge_size
        windows_h = math.ceil(group_h / window_group_size)
        windows_w = math.ceil(group_w / window_group_size)
        for temporal_idx in range(grid_t):
            temporal_offset = temporal_idx * group_h * group_w
            for window_h in range(windows_h):
                for window_w in range(windows_w):
                    for inner_h in range(window_group_size):
                        group_y = window_h * window_group_size + inner_h
                        if group_y >= group_h:
                            continue
                        for inner_w in range(window_group_size):
                            group_x = window_w * window_group_size + inner_w
                            if group_x >= group_w:
                                continue
                            indices.append(
                                group_offset
                                + temporal_offset
                                + group_y * group_w
                                + group_x
                            )
        group_offset += grid_t * group_h * group_w
    return torch.tensor(indices, dtype=torch.long, device=grid_thw.device)


class _FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.window_index_calls: list[torch.Tensor] = []

    def get_window_index(self, grid_thw: torch.Tensor):
        self.window_index_calls.append(grid_thw.detach().cpu().clone())
        return _reference_window_index(grid_thw), [0]


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])


def _extractor(
    *,
    restore_vit_spatial_layout: bool | None = True,
) -> tuple[FeatureExtractor, _FakeBackbone]:
    backbone = _FakeBackbone()
    restore_kwargs = (
        {}
        if restore_vit_spatial_layout is None
        else {
            "restore_vit_spatial_layout": restore_vit_spatial_layout,
        }
    )
    extractor = FeatureExtractor(
        backbone,
        vit_layer_indices=[0],
        llm_layer_indices=[0],
        spatial_merge_size=MERGE_SIZE,
        detach_features=False,
        **restore_kwargs,
    )
    return extractor, backbone


def _coordinate_raster(image_idx: int, side: int) -> torch.Tensor:
    coord_y, coord_x = torch.meshgrid(
        torch.arange(side, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.full_like(coord_y, float(image_idx))
    return torch.stack([image, coord_y, coord_x], dim=-1)


def _to_qwen_block_order(
    raster: torch.Tensor,
    *,
    merge_size: int = MERGE_SIZE,
) -> torch.Tensor:
    """Pack raster patches exactly as seen by a Qwen visual block hook."""

    grid_h, grid_w, channels = raster.shape
    grouped = raster.reshape(
        grid_h // merge_size,
        merge_size,
        grid_w // merge_size,
        merge_size,
        channels,
    )
    grouped = grouped.permute(0, 2, 1, 3, 4).reshape(
        -1,
        merge_size**2,
        channels,
    )
    grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    window_index = _reference_window_index(grid)
    return grouped.index_select(0, window_index).reshape(
        grid_h * grid_w,
        channels,
    )


def _llm_positions(sides: list[int]) -> tuple[dict[int, tuple[int, int]], int]:
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    for image_idx, side in enumerate(sides):
        token_count = (side // MERGE_SIZE) ** 2
        positions[image_idx] = (cursor, cursor + token_count)
        cursor += token_count
    return positions, cursor


@pytest.mark.parametrize("side", [18, 16])
def test_coordinate_coded_vit_tokens_restore_exact_native_raster(side: int) -> None:
    extractor, backbone = _extractor()
    grid = torch.tensor([[1, side, side]], dtype=torch.long)
    layouts = extractor._build_vit_spatial_layouts(
        grid,
        expected_images=1,
    )

    raster = _coordinate_raster(image_idx=7, side=side)
    block_tokens = _to_qwen_block_order(raster)
    restored = extractor._restore_vit_tokens(
        block_tokens,
        layouts[0],
        layer_idx=0,
        image_idx=0,
    )

    assert len(backbone.visual.window_index_calls) == 1
    assert torch.equal(backbone.visual.window_index_calls[0], grid)
    assert restored.shape == (1, side, side, 3)
    torch.testing.assert_close(restored[0], raster, rtol=0, atol=0)


def test_vit_spatial_restore_is_explicitly_opt_in_for_legacy_checkpoints() -> None:
    extractor, backbone = _extractor(restore_vit_spatial_layout=None)
    sides = [16, 16, 16, 16]
    positions, _sequence_length = _llm_positions(sides)
    grid = torch.tensor([[1, side, side] for side in sides], dtype=torch.long)

    extractor.prepare_batch_capture([positions], [{}], grid)

    assert extractor.restore_vit_spatial_layout is False
    assert backbone.visual.window_index_calls == []
    assert extractor._batch_capture_plan["vit_layouts_batch"] == [{}]


def test_compact_multi_image_capture_restores_mixed_grids_and_autograd() -> None:
    extractor, backbone = _extractor()
    sides_batch = [
        [18, 16, 18, 16],
        [16, 18, 16, 18],
    ]
    positions_batch = []
    sequence_lengths = []
    for sides in sides_batch:
        positions, sequence_length = _llm_positions(sides)
        positions_batch.append(positions)
        sequence_lengths.append(sequence_length)

    flat_sides = [side for sides in sides_batch for side in sides]
    grid = torch.tensor(
        [[1, side, side] for side in flat_sides],
        dtype=torch.long,
    )
    extractor.prepare_batch_capture(
        positions_batch,
        [{}, {}],
        grid,
    )

    rasters = [
        _coordinate_raster(image_idx=image_idx, side=side)
        for image_idx, side in enumerate(flat_sides)
    ]
    block_output = torch.cat(
        [_to_qwen_block_order(raster) for raster in rasters],
        dim=0,
    ).requires_grad_()
    backbone.visual.blocks[0](block_output)

    llm_hidden = torch.zeros(
        len(sides_batch),
        max(sequence_lengths),
        5,
    )
    backbone.model.layers[0](llm_hidden)

    vit_tensors, _llm_tensors, history_queries = (
        extractor.extract_batch_compact_tensors()
    )
    actual = vit_tensors[0]

    expected_samples = []
    raster_idx = 0
    for sides in sides_batch:
        expected_views = []
        for side in sides:
            raster = rasters[raster_idx]
            raster_idx += 1
            if side != 16:
                raster = F.interpolate(
                    raster.permute(2, 0, 1).unsqueeze(0),
                    size=(16, 16),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            expected_views.append(raster)
        expected_samples.append(torch.stack(expected_views, dim=0))
    expected = torch.stack(expected_samples, dim=0)

    assert len(backbone.visual.window_index_calls) == 1
    assert torch.equal(backbone.visual.window_index_calls[0], grid)
    assert actual.shape == (2, 4, 16, 16, 3)
    assert history_queries == [[], []]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.sum().backward()
    assert block_output.grad is not None
    assert torch.isfinite(block_output.grad).all()
    assert torch.count_nonzero(block_output.grad) == block_output.numel()
