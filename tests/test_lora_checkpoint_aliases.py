from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.training.utils import (
    _load_normalized_state_dict,
    assert_complete_lora_checkpoint_match,
)


class _SharedLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 3))
        self.lora_B = nn.Parameter(torch.zeros(4, 2))


class _ModelWithHeatmapQwenAlias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen2_5_vl = nn.Module()
        self.qwen2_5_vl.model = _SharedLoRA()
        self.vlm_backbone = self.qwen2_5_vl

        self.heatmap_vln = nn.Module()
        self.heatmap_vln.qwen = self.qwen2_5_vl.model


def _canonical_checkpoint() -> dict[str, torch.Tensor]:
    return {
        "qwen2_5_vl.model.lora_A": torch.ones(2, 3),
        "qwen2_5_vl.model.lora_B": torch.ones(4, 2),
    }


def test_complete_lora_match_ignores_shared_heatmap_qwen_alias() -> None:
    model = _ModelWithHeatmapQwenAlias()

    assert len([key for key in model.state_dict() if "lora_" in key]) == 6
    assert len([key for key, _ in model.named_parameters() if "lora_" in key]) == 2
    assert assert_complete_lora_checkpoint_match(model, _canonical_checkpoint()) == 2


def test_complete_lora_match_still_rejects_missing_physical_tensor() -> None:
    model = _ModelWithHeatmapQwenAlias()
    checkpoint = _canonical_checkpoint()
    checkpoint.pop("qwen2_5_vl.model.lora_B")

    with pytest.raises(
        RuntimeError,
        match=r"model_lora=2 checkpoint_lora=1 matched=1 missing=1",
    ):
        assert_complete_lora_checkpoint_match(model, checkpoint)


def test_complete_lora_match_does_not_ignore_an_independent_heatmap_lora() -> None:
    model = _ModelWithHeatmapQwenAlias()
    model.heatmap_vln.qwen = _SharedLoRA()

    with pytest.raises(
        RuntimeError,
        match=r"model_lora=4 checkpoint_lora=2 matched=2 missing=2",
    ):
        assert_complete_lora_checkpoint_match(model, _canonical_checkpoint())


def test_complete_lora_match_still_rejects_shape_mismatch() -> None:
    model = _ModelWithHeatmapQwenAlias()
    checkpoint = _canonical_checkpoint()
    checkpoint["qwen2_5_vl.model.lora_A"] = torch.ones(3, 3)

    with pytest.raises(
        RuntimeError,
        match=r"matched=1 missing=0 unexpected=0 shape_mismatches=1",
    ):
        assert_complete_lora_checkpoint_match(model, checkpoint)


def test_complete_lora_match_accepts_ddp_and_backbone_checkpoint_aliases() -> None:
    model = _ModelWithHeatmapQwenAlias()
    checkpoint = {
        "module.vlm_backbone.model.lora_A": torch.ones(2, 3),
        "module.vlm_backbone.model.lora_B": torch.ones(4, 2),
    }

    assert assert_complete_lora_checkpoint_match(model, checkpoint) == 2


def test_normalized_loader_updates_the_shared_lora_parameters() -> None:
    model = _ModelWithHeatmapQwenAlias()
    checkpoint = _canonical_checkpoint()

    _missing, unexpected, loaded = _load_normalized_state_dict(model, checkpoint)

    assert unexpected == []
    assert loaded == 2
    assert torch.equal(model.qwen2_5_vl.model.lora_A, checkpoint["qwen2_5_vl.model.lora_A"])
    assert torch.equal(model.heatmap_vln.qwen.lora_B, checkpoint["qwen2_5_vl.model.lora_B"])
