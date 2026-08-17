from __future__ import annotations

import inspect
import importlib
import logging

import torch
import torch.nn as nn

from scripts.train import (
    _dataset_uses_dynamic_sampling,
    _install_baseline_best_threshold,
)
from scripts.training.checkpoint import CheckpointManager
from scripts.training.model_builder import set_trainable_modules
from scripts.training.optimizer import (
    build_optimizer,
    ensure_heatmap_optimizer_state_fp32,
)
from scripts.training.train_loop import train_one_epoch
from scripts.training.utils import build_heatmap_loss_fn
from src.data.sliding_window_dataset import VLNSlidingWindowDataset


class _Coarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.vis_head = nn.Linear(2, 1)


class _Heatmap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = nn.Linear(2, 2).to(torch.bfloat16)
        self.vit_dpt_fusion = nn.Linear(2, 2).to(torch.bfloat16)
        self.llm_dpt_fusion = nn.Linear(2, 2).to(torch.bfloat16)
        self.coarse = _Coarse().to(torch.bfloat16)
        self.fine = nn.Linear(2, 2).to(torch.bfloat16)
        self.pose_free_matcher = None


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_vln = _Heatmap()


def _stage_cfg(**overrides):
    cfg = {
        "trainable_modules": ["heatmap_vln"],
        "strict_trainable_modules": True,
    }
    cfg.update(overrides)
    return cfg


def test_trainable_heatmap_is_fp32_without_casting_qwen() -> None:
    model = _Model()

    set_trainable_modules(model, _stage_cfg(), logging.getLogger("test"))

    assert model.heatmap_vln.qwen.weight.dtype == torch.bfloat16
    head_modules = (
        model.heatmap_vln.vit_dpt_fusion,
        model.heatmap_vln.llm_dpt_fusion,
        model.heatmap_vln.coarse,
        model.heatmap_vln.fine,
    )
    assert all(
        param.dtype == torch.float32
        for module in head_modules
        for param in module.parameters()
    )


def test_heatmap_optimizer_state_is_normalized_to_fp32() -> None:
    model = _Model()
    stage_cfg = _stage_cfg()
    set_trainable_modules(model, stage_cfg, logging.getLogger("test"))
    optimizer = build_optimizer(
        model,
        {"optim": {"weight_decay": 0.01}},
        stage_cfg,
    )
    heatmap_group = next(
        group
        for group in optimizer.param_groups
        if group["name"].startswith("heatmap_")
    )
    param = heatmap_group["params"][0]
    optimizer.state[param]["exp_avg"] = torch.zeros_like(param, dtype=torch.bfloat16)
    optimizer.state[param]["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.bfloat16)

    converted = ensure_heatmap_optimizer_state_fp32(optimizer)

    assert converted == 2
    assert optimizer.state[param]["exp_avg"].dtype == torch.float32
    assert optimizer.state[param]["exp_avg_sq"].dtype == torch.float32


def test_training_forward_explicitly_disables_trajectory_sampling() -> None:
    source = inspect.getsource(train_one_epoch)
    assert "sample_trajectory=False" in source.replace(" ", "")


class _DynamicDataset:
    dynamic_sampling_enabled = True

    def set_epoch(self, _epoch: int) -> None:
        pass


class _StaticDataset(_DynamicDataset):
    dynamic_sampling_enabled = False


def test_dynamic_sampling_marker_controls_worker_refresh() -> None:
    assert _dataset_uses_dynamic_sampling(_DynamicDataset())
    assert not _dataset_uses_dynamic_sampling(_StaticDataset())


def test_baseline_can_guard_max_metric_best_selection(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path))
    manager.configure_best_metric("val_heatmap_joint_pck8", "max")

    assert _install_baseline_best_threshold(manager, 0.61, enabled=True)
    assert manager.best_metric_value == 0.61
    assert not manager.is_better(0.60)
    assert manager.is_better(0.62)


def test_heatmap_loss_factory_wires_strict_raw_logits_mode() -> None:
    loss = build_heatmap_loss_fn(
        {
            "model": {"heatmap": {"heatmap_size": [4, 4]}},
            "data": {"init_hm_size": [4, 4]},
            "loss": {
                "heatmap_vln": {
                    "allow_probability_fallback": False,
                    "lambda_view_macro": 0.3,
                    "lambda_direction_macro": 0.2,
                    "lambda_panoramic_view": 0.1,
                }
            },
        },
        torch.device("cpu"),
    )

    assert loss.allow_probability_fallback is False
    assert loss.lambda_view_macro == 0.3
    assert loss.lambda_direction_macro == 0.2
    assert loss.lambda_panoramic_view == 0.1


def test_sliding_window_augmentation_default_is_opt_in() -> None:
    default = inspect.signature(VLNSlidingWindowDataset).parameters[
        "enable_augmentation"
    ].default
    assert default is False


class _EmptyLoader:
    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())


class _OneBatchLoader:
    def __init__(self, batch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter((self.batch,))


class _LogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.zeros(1, 2, 4, 4, 4, dtype=torch.bfloat16)
        return {
            "visibility": torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            "heatmaps": torch.sigmoid(logits),
            "heatmap_logits": logits,
        }


def test_validation_uses_the_same_heatmap_loss_configuration(monkeypatch) -> None:
    validate_module = importlib.import_module("scripts.training.validate")
    captured = {}

    class _Loss:
        temperature = 1.0

        def __call__(self, *_args, **kwargs):
            captured["loss_kwargs"] = kwargs
            zero = torch.tensor(0.0)
            return {
                "total": zero,
                "peak_loss": zero,
                "vis_loss": zero,
                "coord_loss": zero,
            }

    def fake_build(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Loss()

    monkeypatch.setattr(validate_module, "build_heatmap_loss_fn", fake_build)
    model = _LogitModel()
    batch = {
        "history_frames": torch.zeros(1, 2, 3, 4, 4),
        "current_frame": torch.zeros(1, 3, 4, 4),
        "action": torch.zeros(1, 2),
        "action_valid": torch.ones(1),
        "is_stop": torch.zeros(1),
        "text": ["go"],
        "heatmap": torch.zeros(1, 2, 4, 4, 4),
        "gt_visibility": torch.zeros(1, 2, 4),
        "history_mask": torch.ones(1, 2),
    }
    validate_module.validate(
        model,
        _OneBatchLoader(batch),
        {
            "model": {"device": "cpu"},
            "optim": {"amp": "none"},
            "loss": {},
            "validation": {"val_inference_batches": 0},
            "log": {},
        },
        logging.getLogger("test"),
        {
            "train_history": True,
            "train_future": False,
            "train_action": False,
        },
    )

    assert "lambda_neg_override" not in captured["kwargs"]
    assert model.calls[0]["return_heatmap_logits"] is True
    assert captured["loss_kwargs"]["pred_heatmap_logits"].dtype == torch.float32
