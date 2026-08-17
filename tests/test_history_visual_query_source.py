from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from src.config_schema import HeatmapPoseFreeConfig
from src.models.heatmap.feature_extractor import (
    HISTORY_QUERY_TEXT_ANCHOR,
    HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
    FeatureExtractor,
)
from src.models.heatmap.heatmap_vln import HeatmapVLN


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])


def _extractor(
    source: str = HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
    *,
    detach_features: bool = False,
) -> FeatureExtractor:
    return FeatureExtractor(
        _FakeBackbone(),
        vit_layer_indices=[],
        llm_layer_indices=[0],
        spatial_merge_size=2,
        detach_features=detach_features,
        history_query_source=source,
    )


def _layout(
    *,
    num_histories: int = 1,
    token_counts: list[int] | None = None,
) -> tuple[dict[int, tuple[int, int]], dict[int, int], torch.Tensor, int]:
    occurrence_count = 4 + 4 * num_histories
    if token_counts is None:
        token_counts = [4] * occurrence_count
    assert len(token_counts) == occurrence_count

    positions: dict[int, tuple[int, int]] = {}
    grids = []
    cursor = 1
    for occurrence, count in enumerate(token_counts):
        positions[occurrence] = (cursor, cursor + count)
        cursor += count + 1
        side = int(count**0.5)
        assert side * side == count
        grids.append([1, side * 2, side * 2])

    anchors = {history_idx: positions[4 + history_idx * 4 + 3][1] for history_idx in range(num_histories)}
    return positions, anchors, torch.tensor(grids, dtype=torch.long), cursor + 2


def _hidden_with_occurrence_values(
    positions: dict[int, tuple[int, int]],
    anchors: dict[int, int],
    sequence_length: int,
    *,
    channels: int = 3,
) -> torch.Tensor:
    hidden = torch.zeros(1, sequence_length, channels)
    for occurrence, (start, end) in positions.items():
        hidden[:, start:end, :] = float(occurrence + 1)
    for anchor in anchors.values():
        hidden[:, anchor, :] = 999.0
    return hidden


def _compact_query(
    extractor: FeatureExtractor,
    hidden: torch.Tensor,
    positions: dict[int, tuple[int, int]],
    anchors: dict[int, int],
    grid: torch.Tensor,
) -> torch.Tensor:
    extractor.prepare_batch_capture([positions], [anchors], grid)
    extractor._make_llm_hook(0)(None, None, (hidden,))
    assert extractor._captured_batch_queries is not None
    return extractor._captured_batch_queries[0][0]


def test_default_text_anchor_source_is_unchanged_and_visual_source_is_explicit() -> None:
    positions, anchors, grid, sequence_length = _layout()
    hidden = _hidden_with_occurrence_values(positions, anchors, sequence_length)

    default = _extractor(HISTORY_QUERY_TEXT_ANCHOR)
    default_query = _compact_query(default, hidden, positions, anchors, grid)
    torch.testing.assert_close(
        default_query,
        hidden[0, anchors[0]],
        rtol=0,
        atol=0,
    )
    assert default.history_query_source == HISTORY_QUERY_TEXT_ANCHOR

    visual = _extractor()
    visual_query = _compact_query(visual, hidden, positions, anchors, grid)
    torch.testing.assert_close(
        visual_query,
        torch.full_like(visual_query, 6.5),
        rtol=0,
        atol=0,
    )
    assert visual.history_query_source == HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1


def test_visual_query_gives_each_history_view_equal_weight() -> None:
    token_counts = [4, 4, 4, 4, 4, 16, 4, 16]
    positions, anchors, grid, sequence_length = _layout(token_counts=token_counts)
    hidden = torch.zeros(1, sequence_length, 2)
    for view_idx in range(4):
        start, end = positions[4 + view_idx]
        hidden[:, start:end, :] = float(view_idx + 1)

    query = _compact_query(_extractor(), hidden, positions, anchors, grid)
    torch.testing.assert_close(query, torch.full_like(query, 2.5), rtol=0, atol=0)


def test_compact_and_noncompact_visual_query_semantics_are_bitwise_equal() -> None:
    positions, anchors, grid, sequence_length = _layout()
    hidden = torch.randn(1, sequence_length, 5)

    compact_query = _compact_query(_extractor(), hidden, positions, anchors, grid)

    noncompact = _extractor()
    noncompact.llm_hidden_states[0] = hidden
    _current_vit, _current_llm, noncompact_queries, _history_views = noncompact.extract(
        positions,
        anchors,
        grid,
    )
    assert len(noncompact_queries) == 1
    torch.testing.assert_close(
        compact_query,
        noncompact_queries[0],
        rtol=0,
        atol=0,
    )

    second_hidden = hidden + torch.randn_like(hidden)
    second_compact_query = _compact_query(
        _extractor(),
        second_hidden,
        positions,
        anchors,
        grid,
    )
    batched_noncompact = _extractor()
    batched_noncompact.llm_hidden_states[0] = torch.cat(
        [hidden, second_hidden],
        dim=0,
    )
    extracted = batched_noncompact.extract_batch(
        [positions, positions],
        [anchors, anchors],
        torch.cat([grid, grid], dim=0),
    )
    torch.testing.assert_close(extracted[0][2][0], compact_query, rtol=0, atol=0)
    torch.testing.assert_close(
        extracted[1][2][0],
        second_compact_query,
        rtol=0,
        atol=0,
    )


def test_visual_query_ignores_anchor_hidden_and_changes_with_history_pixels() -> None:
    positions, anchors, grid, sequence_length = _layout()
    hidden = torch.randn(1, sequence_length, 4)
    baseline = _compact_query(_extractor(), hidden, positions, anchors, grid)

    anchor_changed = hidden.clone()
    anchor_changed[:, anchors[0], :] += 1000
    anchor_query = _compact_query(
        _extractor(),
        anchor_changed,
        positions,
        anchors,
        grid,
    )
    assert torch.equal(anchor_query, baseline)

    history_changed = hidden.clone()
    start, end = positions[6]
    history_changed[:, start:end, :] += 1
    history_query = _compact_query(
        _extractor(),
        history_changed,
        positions,
        anchors,
        grid,
    )
    assert not torch.equal(history_query, baseline)


def test_visual_query_pooling_preserves_autograd_only_through_history_spans() -> None:
    positions, anchors, grid, sequence_length = _layout()
    hidden = torch.randn(1, sequence_length, 6, requires_grad=True)
    query = _compact_query(_extractor(detach_features=False), hidden, positions, anchors, grid)
    query.square().sum().backward()

    assert hidden.grad is not None
    for occurrence in range(4, 8):
        start, end = positions[occurrence]
        assert hidden.grad[:, start:end].abs().sum() > 0
    for occurrence in range(4):
        start, end = positions[occurrence]
        assert torch.count_nonzero(hidden.grad[:, start:end]) == 0
    assert torch.count_nonzero(hidden.grad[:, anchors[0]]) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_occurrence", "requires exactly 8 image occurrences"),
        ("overlap", "preceding occurrence ends"),
        ("wrong_span_length", "span has 3 tokens"),
        ("anchor_before_images", "must follow all four visual spans"),
        ("missing_grid_row", "expected 8 image-grid rows"),
    ],
)
def test_visual_query_capture_fails_closed_on_invalid_mapping(
    mutation: str,
    message: str,
) -> None:
    positions, anchors, grid, _sequence_length = _layout()
    positions = deepcopy(positions)
    anchors = dict(anchors)
    grid = grid.clone()

    if mutation == "missing_occurrence":
        positions.pop(7)
    elif mutation == "overlap":
        start, _end = positions[4]
        positions[4] = (start - 2, start + 2)
    elif mutation == "wrong_span_length":
        start, _end = positions[7]
        positions[7] = (start, start + 3)
    elif mutation == "anchor_before_images":
        anchors[0] = positions[7][1] - 1
    elif mutation == "missing_grid_row":
        grid = grid[:-1]
    else:  # pragma: no cover - parametrization exhaustiveness
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=message):
        _extractor().prepare_batch_capture([positions], [anchors], grid)


def test_visual_query_capture_rejects_anchor_after_next_history_images() -> None:
    positions, anchors, grid, _sequence_length = _layout(num_histories=2)
    anchors[0] = positions[8][0]
    with pytest.raises(RuntimeError, match="must precede the next history images"):
        _extractor().prepare_batch_capture([positions], [anchors], grid)


def test_visual_query_hook_rejects_span_outside_hidden_sequence() -> None:
    positions, anchors, grid, sequence_length = _layout()
    extractor = _extractor()
    extractor.prepare_batch_capture([positions], [anchors], grid)
    too_short = torch.zeros(1, sequence_length - 8, 3)
    with pytest.raises(RuntimeError, match="outside sequence length"):
        extractor._make_llm_hook(0)(None, None, (too_short,))


def test_pose_free_config_and_heatmap_propagate_query_source() -> None:
    assert HeatmapPoseFreeConfig().history_query_source == HISTORY_QUERY_TEXT_ANCHOR
    with pytest.raises(ValidationError):
        HeatmapPoseFreeConfig(history_query_source="unknown")
    with pytest.raises(ValidationError):
        HeatmapPoseFreeConfig(
            history_query_soruce=HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
        )

    heatmap = HeatmapVLN(
        qwen_model=_FakeBackbone(),
        processor=object(),
        c_vit=6,
        c_llm=8,
        c_fused=4,
        vit_layer_indices=[],
        llm_layer_indices=[0],
        spatial_merge_size=4,
        decoder_mode="pose_free_matcher",
        pose_free_config={
            "history_query_source": HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
            "match_dim": 4,
        },
    )
    assert heatmap.history_query_source == HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1
    assert heatmap.feat_extractor.history_query_source == HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1
    assert heatmap.feat_extractor.spatial_merge_size == 4

    with pytest.raises(ValueError, match="Unknown pose_free configuration keys"):
        HeatmapVLN(
            qwen_model=_FakeBackbone(),
            processor=object(),
            c_vit=6,
            c_llm=8,
            c_fused=4,
            vit_layer_indices=[],
            llm_layer_indices=[0],
            decoder_mode="pose_free_matcher",
            pose_free_config={
                "history_query_soruce": HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
            },
        )


def test_unknown_feature_extractor_query_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="history_query_source"):
        _extractor("unknown")
