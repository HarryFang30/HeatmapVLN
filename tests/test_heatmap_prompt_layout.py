from __future__ import annotations

import re
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.heatmap.feature_extractor import FeatureExtractor
from src.models.heatmap.heatmap_vln import HeatmapVLN
from src.models.heatmap.input_constructor import (
    VIEW_NAMES,
    _build_history_anchor_text,
    construct_input,
    find_text_anchor_positions,
)
from src.models.qwen2_5_vl.integration import Qwen2_5VLIntegration


def _panorama(base: int) -> dict[str, Image.Image]:
    return {
        name: Image.new("RGB", (2, 2), color=(base + view_idx, 0, 0))
        for view_idx, name in enumerate(VIEW_NAMES)
    }


def _image_ids(messages: list[dict]) -> list[int]:
    return [
        item["image"].getpixel((0, 0))[0]
        for message in messages
        for item in message["content"]
        if item["type"] == "image"
    ]


def test_history_projection_layout_matches_feature_occurrence_contract() -> None:
    current = _panorama(10)
    histories = [_panorama(20), _panorama(30)]

    messages = construct_input(
        current_views=current,
        history_panoramas=histories,
        instruction="find the kitchen",
        history_projection_layout=True,
    )

    # FeatureExtractor's contract is occurrence 0..3=current, followed by
    # chronological history panoramas in four-view groups.
    assert _image_ids(messages) == [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]

    content = messages[0]["content"]
    for hist_idx in range(len(histories)):
        anchor_item_idx = next(
            idx
            for idx, item in enumerate(content)
            if item.get("text") == _build_history_anchor_text(hist_idx)
        )
        preceding_images = [
            item for item in content[:anchor_item_idx]
            if item["type"] == "image"
        ]
        # The anchor is causal-after current[4] and every history group up to
        # and including the panorama it represents.
        assert len(preceding_images) == 4 + (hist_idx + 1) * 4


def test_default_system2_layout_remains_history_first() -> None:
    current = _panorama(10)
    histories = [_panorama(20), _panorama(30)]

    messages = construct_input(
        current_views=current,
        history_panoramas=histories,
        instruction="find the kitchen",
        pixel_goal=[12, 34],
        assistant_text="view: front\npixel: 12 34",
        structured_pano_output=True,
    )

    # Existing Stage1-S2 checkpoints were trained with history-first prompts;
    # the explicit Stage-1 switch must not change the default contract.
    assert _image_ids(messages) == [20, 21, 22, 23, 30, 31, 32, 33, 10, 11, 12, 13]


class _AnchorTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        match = re.fullmatch(r"Historical observation (\d+) .*", text)
        if match is None:
            raise AssertionError(f"Unexpected anchor text: {text}")
        return [9000 + int(match.group(1))]


def test_anchor_token_positions_are_after_required_visual_context() -> None:
    tokenizer = _AnchorTokenizer()
    image_positions: dict[int, tuple[int, int]] = {}
    input_ids: list[int] = []
    occurrence = 0

    # current[4]
    for _ in range(4):
        start = len(input_ids)
        input_ids.extend([77] * 4)
        image_positions[occurrence] = (start, len(input_ids))
        occurrence += 1

    # history[k][4] -> anchor[k]
    expected_anchor_positions = {}
    for hist_idx in range(2):
        for _ in range(4):
            start = len(input_ids)
            input_ids.extend([77] * 4)
            image_positions[occurrence] = (start, len(input_ids))
            occurrence += 1
        anchor_id = tokenizer.encode(_build_history_anchor_text(hist_idx))[0]
        input_ids.append(anchor_id)
        expected_anchor_positions[hist_idx] = len(input_ids) - 1

    anchors = find_text_anchor_positions(
        torch.tensor([input_ids]),
        tokenizer,
        num_history=2,
    )
    assert anchors == expected_anchor_positions
    FeatureExtractor._validate_history_projection_layout(
        image_positions,
        anchors,
        tag="unit test",
    )


def test_feature_extractor_rejects_pre_image_history_anchor() -> None:
    image_positions = {
        occurrence: (occurrence * 4 + 10, occurrence * 4 + 14)
        for occurrence in range(8)
    }
    with pytest.raises(RuntimeError, match=r"occurs .* before its required visual context"):
        FeatureExtractor._validate_history_projection_layout(
            image_positions,
            {0: 5},
            tag="legacy broken layout",
        )


def test_feature_extractor_maps_current_and_history_occurrences() -> None:
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.llm_layer_indices = [0]
    extractor.vit_layer_indices = [0]
    extractor._batch_capture_plan = None
    extractor._llm_resize_logged = False
    extractor._vit_resize_logged = False

    image_positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    for occurrence in range(8):
        image_positions[occurrence] = (cursor, cursor + 64)
        cursor += 64
    anchor_position = cursor
    hidden = torch.zeros(1, cursor + 1, 3)
    for occurrence, (start, end) in image_positions.items():
        hidden[0, start:end] = float(occurrence + 1)
    hidden[0, anchor_position] = 99.0
    extractor.llm_hidden_states = {0: hidden}

    vit = torch.cat(
        [torch.full((256, 2), float(occurrence + 1)) for occurrence in range(8)],
        dim=0,
    )
    extractor.vit_features = {0: vit}
    image_grid_thw = torch.tensor([[1, 16, 16]] * 8)

    current_vit, current_llm, history_queries, history_llm = extractor.extract(
        image_positions,
        {0: anchor_position},
        image_grid_thw,
    )

    for view_idx in range(4):
        assert torch.all(current_llm[view_idx][0] == float(view_idx + 1))
        assert torch.all(current_vit[view_idx][0] == float(view_idx + 1))
        assert torch.all(history_llm[0][view_idx] == float(view_idx + 5))
    assert torch.all(history_queries[0] == 99.0)


class _RecordingProcessor:
    def __init__(self) -> None:
        self.messages_batch = None

    def apply_chat_template(self, messages_batch, **kwargs):
        del kwargs
        self.messages_batch = messages_batch
        return {"input_ids": torch.ones(len(messages_batch), 1, dtype=torch.long)}


def test_batched_prompt_slices_padded_history_to_real_lengths() -> None:
    model = HeatmapVLN.__new__(HeatmapVLN)
    nn.Module.__init__(model)
    processor = _RecordingProcessor()
    model.processor = processor

    current_views = torch.zeros(2, 4, 3, 2, 2)
    padded_histories = torch.zeros(2, 3, 4, 3, 2, 2)
    _inputs, resolved_lengths = model.prepare_qwen_inputs_batch(
        current_views=current_views,
        history_panoramas=padded_histories,
        instruction=["one", "two"],
        device=torch.device("cpu"),
        num_histories=[1, 2],
    )

    assert resolved_lengths == [1, 2]
    assert processor.messages_batch is not None
    assert [_image_ids(messages) for messages in processor.messages_batch] == [
        [0] * 8,   # current[4] + one real history panorama
        [0] * 12,  # current[4] + two real history panoramas
    ]


def test_standard_panorama_forward_propagates_real_history_lengths() -> None:
    """The non-tokenized Qwen path must not lose collate's real-K metadata."""
    integration = Qwen2_5VLIntegration.__new__(Qwen2_5VLIntegration)
    nn.Module.__init__(integration)
    integration._model_loaded = True
    observed: dict[str, object] = {}

    def fake_forward_batch_panorama(self, **kwargs):
        del self
        observed.update(kwargs)
        return None, None, 0, {}, None

    integration._forward_batch_panorama = types.MethodType(
        fake_forward_batch_panorama,
        integration,
    )

    integration(
        history_frames=torch.zeros(2, 1, 3, 2, 2),
        current_frame=torch.zeros(2, 3, 2, 2),
        current_views=torch.zeros(2, 4, 3, 2, 2),
        history_panoramas=torch.zeros(2, 3, 4, 3, 2, 2),
        panoramic_num_histories=[1, 2],
        heatmap_vln=object(),
        return_hidden_states=False,
    )

    assert observed["num_histories"] == [1, 2]
