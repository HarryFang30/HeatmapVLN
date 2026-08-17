"""Tests for the fail-closed frozen heatmap checkpoint dependency loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import scripts.training.frozen_heatmap_checkpoint as loader_module
from scripts.training import (
    FrozenHeatmapCheckpointError,
    load_frozen_heatmap_checkpoint,
)
from scripts.training.frozen_heatmap_checkpoint import compute_file_sha256


class _TinyHeatmap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra_scale = nn.Parameter(torch.zeros(2))
        self.projection = nn.Linear(3, 2)


class _TinyPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_vln = _TinyHeatmap()
        self.nextdit = nn.Linear(2, 2)


class _ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


def _full_state(
    model: _TinyPipeline,
    *,
    prefix: str = "heatmap_vln.",
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}{name}": torch.full_like(parameter, float(index + 1))
        for index, (name, parameter) in enumerate(
            model.heatmap_vln.named_parameters()
        )
    }


def _save(
    path: Path,
    payload: object,
) -> str:
    torch.save(payload, path)
    return compute_file_sha256(path)


def _assert_model_unchanged(
    model: _TinyPipeline,
    before: dict[str, torch.Tensor],
) -> None:
    for name, parameter in model.heatmap_vln.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])


def test_loads_complete_module_prefixed_state_and_returns_dependency_metadata(
    tmp_path: Path,
) -> None:
    pipeline = _TinyPipeline()
    wrapped = _ModuleWrapper(pipeline)
    expected = _full_state(pipeline, prefix="module.heatmap_vln.")
    checkpoint = tmp_path / "heatmap.pth"
    digest = _save(
        checkpoint,
        {
            "metadata": {"producer": "unit-test"},
            "trainable_state_dict": expected,
        },
    )

    dependency = load_frozen_heatmap_checkpoint(
        wrapped,
        checkpoint,
        digest,
    )

    for name, parameter in pipeline.heatmap_vln.named_parameters():
        torch.testing.assert_close(
            parameter.detach(),
            expected[f"module.heatmap_vln.{name}"],
        )
        assert parameter.requires_grad is False
    assert pipeline.heatmap_vln.training is False
    assert dependency == {
        "schema_version": "frozen-heatmap-checkpoint-v1",
        "dependency_type": "frozen_heatmap",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "state_key": "trainable_state_dict",
        "target_module": "heatmap_vln",
        "tensor_count": 3,
        "parameter_names": [
            "extra_scale",
            "projection.bias",
            "projection.weight",
        ],
        "parameter_shapes": {
            "extra_scale": [2],
            "projection.bias": [2],
            "projection.weight": [2, 3],
        },
        "frozen": True,
    }


def test_sha_mismatch_fails_before_deserialization_or_model_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "heatmap.pth"
    _save(checkpoint, {"trainable_state_dict": {"anything": torch.ones(1)}})

    def _must_not_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load must not run before SHA verification")

    monkeypatch.setattr(loader_module.torch, "load", _must_not_load)
    with pytest.raises(FrozenHeatmapCheckpointError, match="SHA-256 mismatch"):
        load_frozen_heatmap_checkpoint(
            object(),  # type: ignore[arg-type]
            checkpoint,
            "0" * 64,
        )


def test_rejects_invalid_expected_digest_before_touching_path(tmp_path: Path) -> None:
    with pytest.raises(FrozenHeatmapCheckpointError, match="64 lowercase"):
        load_frozen_heatmap_checkpoint(
            _TinyPipeline(),
            tmp_path / "missing.pth",
            "ABC",
        )


def test_accepts_only_trainable_state_dict(tmp_path: Path) -> None:
    model = _TinyPipeline()
    checkpoint = tmp_path / "wrong-key.pth"
    digest = _save(checkpoint, {"model_state_dict": _full_state(model)})

    with pytest.raises(
        FrozenHeatmapCheckpointError,
        match="payload.trainable_state_dict",
    ):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)


def test_rejects_partial_heatmap_state_before_copy(tmp_path: Path) -> None:
    model = _TinyPipeline()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.heatmap_vln.named_parameters()
    }
    state = _full_state(model)
    state.pop("heatmap_vln.projection.bias")
    checkpoint = tmp_path / "partial.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="missing="):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)
    _assert_model_unchanged(model, before)


def test_rejects_unexpected_heatmap_parameter(tmp_path: Path) -> None:
    model = _TinyPipeline()
    state = _full_state(model)
    state["heatmap_vln.unexpected.weight"] = torch.ones(1)
    checkpoint = tmp_path / "unexpected.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="unexpected="):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)


def test_rejects_shape_mismatch_before_copy(tmp_path: Path) -> None:
    model = _TinyPipeline()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.heatmap_vln.named_parameters()
    }
    state = _full_state(model)
    state["heatmap_vln.projection.weight"] = torch.ones(3, 2)
    checkpoint = tmp_path / "shape.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="shape mismatch"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)
    _assert_model_unchanged(model, before)


def test_rejects_duplicate_keys_after_module_normalization(tmp_path: Path) -> None:
    model = _TinyPipeline()
    state = _full_state(model)
    state["module.heatmap_vln.extra_scale"] = torch.ones(2)
    checkpoint = tmp_path / "duplicate.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="Duplicate checkpoint"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "heatmap_vln.lora_A.weight",
        "qwen.layers.0.weight",
        "system1.weight",
        "system2.weight",
        "nextdit.blocks.0.weight",
        "heatmap_vln.pano_adapter.weight",
        "heatmap_vln.tokenizer.weight",
        "heatmap_vln.heatmap_control.weight",
    ],
)
def test_rejects_forbidden_parameter_families(
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    model = _TinyPipeline()
    state = _full_state(model)
    state[forbidden_name] = torch.ones(1)
    checkpoint = tmp_path / "forbidden.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="Forbidden"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)


def test_rejects_non_tensor_values(tmp_path: Path) -> None:
    model = _TinyPipeline()
    state: dict[str, object] = dict(_full_state(model))
    state["heatmap_vln.extra_scale"] = [1.0, 1.0]
    checkpoint = tmp_path / "non-tensor.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="not a tensor"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)


def test_rejects_non_finite_tensor_without_mutating_model(tmp_path: Path) -> None:
    model = _TinyPipeline()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.heatmap_vln.named_parameters()
    }
    state = _full_state(model)
    state["heatmap_vln.extra_scale"][0] = torch.nan
    checkpoint = tmp_path / "non-finite.pth"
    digest = _save(checkpoint, {"trainable_state_dict": state})

    with pytest.raises(FrozenHeatmapCheckpointError, match="non-finite"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)
    _assert_model_unchanged(model, before)


def test_post_copy_verification_failure_rolls_back_all_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPipeline()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.heatmap_vln.named_parameters()
    }
    checkpoint = tmp_path / "verify.pth"
    digest = _save(
        checkpoint,
        {"trainable_state_dict": _full_state(model)},
    )
    monkeypatch.setattr(loader_module.torch, "equal", lambda left, right: False)

    with pytest.raises(FrozenHeatmapCheckpointError, match="Post-copy"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)
    _assert_model_unchanged(model, before)


def test_requires_existing_parameterized_heatmap_module(tmp_path: Path) -> None:
    class _NoHeatmap(nn.Module):
        pass

    checkpoint = tmp_path / "empty-target.pth"
    digest = _save(
        checkpoint,
        {"trainable_state_dict": {"heatmap_vln.weight": torch.ones(1)}},
    )
    with pytest.raises(FrozenHeatmapCheckpointError, match="heatmap_vln must exist"):
        load_frozen_heatmap_checkpoint(_NoHeatmap(), checkpoint, digest)

    model = _NoHeatmap()
    model.heatmap_vln = nn.Identity()
    with pytest.raises(FrozenHeatmapCheckpointError, match="has no named parameters"):
        load_frozen_heatmap_checkpoint(model, checkpoint, digest)
