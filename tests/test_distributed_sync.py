import pytest
import torch
import torch.nn as nn
from scripts.training import distributed
from scripts.training.distributed import (
    _all_reduce_trainable_grad,
    _get_supported_trainable_sync_modules,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 2))
        self.frozen_lora_A = nn.Parameter(torch.ones(2, 2), requires_grad=False)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_backbone = DummyBackbone()
        self.heatmap_vln = None
        self.nextdit_action_head = None
        self.latent_queries = None


def test_lora_trainable_modules_are_supported_for_manual_sync():
    model = DummyModel()
    model.vlm_backbone.base.weight.requires_grad_(False)
    model.vlm_backbone.base.bias.requires_grad_(False)

    sync_modules = _get_supported_trainable_sync_modules(
        model,
        {"trainable_modules": ["lora"]},
    )

    assert [name for name, _ in sync_modules] == ["vlm_lora"]
    params = list(sync_modules[0][1].parameters())
    assert len(params) == 2
    assert params[0] is model.vlm_backbone.lora_A
    assert params[1] is model.vlm_backbone.lora_B


def test_lora_sync_requires_loaded_trainable_lora_params():
    model = DummyModel()
    for param in model.vlm_backbone.parameters():
        param.requires_grad_(False)

    with pytest.raises(RuntimeError, match="no trainable lora_"):
        _get_supported_trainable_sync_modules(
            model,
            {"trainable_modules": ["lora"]},
        )


def test_manual_sync_materializes_missing_gradient_and_still_reduces(monkeypatch):
    param = nn.Parameter(torch.tensor([2.0, -3.0]))
    reduced = []

    def fake_all_reduce(tensor):
        reduced.append(tensor)
        tensor.add_(torch.tensor([4.0, 6.0]))
        return tensor

    monkeypatch.setattr(
        distributed,
        "_dist_all_reduce_in_place",
        fake_all_reduce,
    )

    assert param.grad is None
    _all_reduce_trainable_grad(param, world_size=2)

    assert reduced == [param.grad]
    assert torch.equal(param.grad, torch.tensor([2.0, 3.0]))


def test_manual_sync_ignores_only_frozen_parameters(monkeypatch):
    param = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    called = False

    def fake_all_reduce(tensor):
        nonlocal called
        called = True
        return tensor

    monkeypatch.setattr(
        distributed,
        "_dist_all_reduce_in_place",
        fake_all_reduce,
    )

    _all_reduce_trainable_grad(param, world_size=2)

    assert param.grad is None
    assert called is False
