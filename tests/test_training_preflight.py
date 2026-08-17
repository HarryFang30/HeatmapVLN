import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.training.preflight as preflight


class _EpochAwareDataset:
    def __init__(self) -> None:
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _EpochAwareSampler:
    def __init__(self) -> None:
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _OneBatchLoader:
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        raise AssertionError("preflight must not preview the DataLoader")


def _finite_metrics() -> dict[str, float]:
    return {
        'total_loss': 1.25,
        'heatmap_loss': 0.0,
        'trajectory_loss': 1.25,
        'lm_loss': 0.0,
        'l2_sp_loss': 0.0,
        'optimizer_steps': 1,
    }


class _ToyHeatmapPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_vln = nn.Module()
        self.heatmap_vln.decoder = nn.Linear(3, 2, bias=False)
        self.vlm_backbone = nn.Module()
        self.vlm_backbone.lora_A = nn.Parameter(
            torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            requires_grad=False,
        )


class _ToyControlBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_control = nn.Module()
        self.heatmap_control.gate = nn.Parameter(torch.zeros(4))
        self.heatmap_control.projection = nn.Linear(3, 3, bias=False)


class _ToyControlPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_tokenizer = nn.Linear(3, 3, bias=False)
        self.nextdit_action_head = nn.Module()
        self.nextdit_action_head.traj_dit = nn.Module()
        self.nextdit_action_head.traj_dit.model = nn.Module()
        self.nextdit_action_head.traj_dit.model.layers = nn.ModuleList(
            [_ToyControlBlock(), _ToyControlBlock()]
        )


def _run(monkeypatch, metrics, *, model=None, stage_cfg=None, mutation=None):
    captured = {}

    def fake_train_one_epoch(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        if mutation is not None:
            mutation(args[0])
        return metrics

    monkeypatch.setattr(preflight, 'train_one_epoch', fake_train_one_epoch)
    dataset = _EpochAwareDataset()
    sampler = _EpochAwareSampler()
    model = object() if model is None else model
    stage_cfg = {'epochs': 2} if stage_cfg is None else stage_cfg
    result = preflight.run_training_preflight(
        model,
        _OneBatchLoader(),
        object(),
        object(),
        None,
        {},
        logging.getLogger('test.preflight'),
        stage_name='stage3',
        stage_cfg=stage_cfg,
        train_dataset=dataset,
        train_sampler=sampler,
    )
    return result, captured, dataset, sampler


def test_real_preflight_runs_one_complete_training_step(monkeypatch):
    result, captured, dataset, sampler = _run(monkeypatch, _finite_metrics())

    assert result['optimizer_steps'] == 1
    assert dataset.epoch == 1
    assert sampler.epoch == 1
    assert captured['kwargs']['max_batches'] == 1
    assert captured['kwargs']['ckpt_manager'] is None
    assert captured['kwargs']['mid_epoch_save_every'] == 0
    assert captured['kwargs']['tb_writer'] is None
    assert captured['kwargs']['metrics_jsonl_path'] is None
    assert captured['kwargs']['actual_batch_observer'] is None


def test_strict_smoke_observes_the_actual_train_loop_batch_without_preview(
    monkeypatch,
):
    actual_batch = {
        "sample_identity": [
            "scene/clip_000001@000007",
            "scene/clip_000002@000009",
        ],
        "history_pose_provider": "amb3r_vo_cache",
    }
    captured: dict[str, object] = {}

    def fake_train_one_epoch(*args, **kwargs):
        observer = kwargs["actual_batch_observer"]
        assert observer is not None
        observer(0, actual_batch)
        metrics = _finite_metrics()
        metrics["heatmap_loss"] = 1.0
        return metrics

    def fake_build_local_rank_audit(**kwargs):
        captured.update(kwargs)
        return {"rank": 0}

    monkeypatch.setattr(preflight, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(preflight, "smoke_audit_enabled", lambda stage: True)
    monkeypatch.setattr(
        preflight,
        "install_gradient_hooks",
        lambda model: ({"fake": {"seen": True}}, []),
    )
    monkeypatch.setattr(
        preflight,
        "build_local_rank_audit",
        fake_build_local_rank_audit,
    )
    monkeypatch.setattr(
        preflight,
        "gather_and_validate_local_audit",
        lambda local, **kwargs: {
            "global_unique_identity_count": 16,
            "gradient_hook_tensors_by_rank": [34] * 8,
            "post_parameter_digest": "model-digest",
            "ema_digest": "ema-digest",
        },
    )
    dataset = _EpochAwareDataset()
    sampler = _EpochAwareSampler()

    result = preflight.run_training_preflight(
        object(),
        _OneBatchLoader(),
        object(),
        object(),
        None,
        {"optim": {"batch_size": 2}},
        logging.getLogger("test.preflight.strict-smoke"),
        stage_name="amb3r_pose_domain_adaptation",
        stage_cfg={"epochs": 1},
        train_dataset=dataset,
        train_sampler=sampler,
        ema=object(),
        dist_context=SimpleNamespace(
            enabled=True,
            world_size=8,
            rank=0,
        ),
    )

    assert result["pose_adaptation_8gpu_smoke"]["global_unique_identity_count"] == 16
    assert captured["identities"] == actual_batch["sample_identity"]
    assert captured["providers"] == ["amb3r_vo_cache", "amb3r_vo_cache"]


def _strict_smoke_context() -> SimpleNamespace:
    return SimpleNamespace(enabled=True, world_size=8, rank=0)


def _actual_smoke_batch() -> dict[str, object]:
    return {
        "sample_identity": [
            "scene/clip_000001@000007",
            "scene/clip_000002@000009",
        ],
        "history_pose_provider": "amb3r_vo_cache",
    }


def test_strict_smoke_removes_gradient_hooks_when_training_raises(monkeypatch):
    class _Handle:
        removed = False

        def remove(self) -> None:
            self.removed = True

    handle = _Handle()

    def failing_train(*args, **kwargs):
        kwargs["actual_batch_observer"](0, _actual_smoke_batch())
        raise RuntimeError("deliberate train failure")

    monkeypatch.setattr(preflight, "train_one_epoch", failing_train)
    monkeypatch.setattr(preflight, "smoke_audit_enabled", lambda stage: True)
    monkeypatch.setattr(
        preflight,
        "install_gradient_hooks",
        lambda model: ({}, [handle]),
    )

    with pytest.raises(RuntimeError, match="deliberate train failure"):
        preflight.run_training_preflight(
            object(),
            _OneBatchLoader(),
            object(),
            object(),
            None,
            {"optim": {"batch_size": 2}},
            logging.getLogger("test.preflight.strict-smoke-finally"),
            stage_name="amb3r_pose_domain_adaptation",
            stage_cfg={"epochs": 1},
            train_dataset=_EpochAwareDataset(),
            train_sampler=_EpochAwareSampler(),
            ema=object(),
            dist_context=_strict_smoke_context(),
        )

    assert handle.removed is True


def test_strict_smoke_requires_positive_heatmap_loss(monkeypatch):
    def zero_heatmap_train(*args, **kwargs):
        kwargs["actual_batch_observer"](0, _actual_smoke_batch())
        return _finite_metrics()

    monkeypatch.setattr(preflight, "train_one_epoch", zero_heatmap_train)
    monkeypatch.setattr(preflight, "smoke_audit_enabled", lambda stage: True)
    monkeypatch.setattr(
        preflight,
        "install_gradient_hooks",
        lambda model: ({}, []),
    )

    with pytest.raises(RuntimeError, match="positive heatmap_loss"):
        preflight.run_training_preflight(
            object(),
            _OneBatchLoader(),
            object(),
            object(),
            None,
            {"optim": {"batch_size": 2}},
            logging.getLogger("test.preflight.strict-smoke-heatmap-loss"),
            stage_name="amb3r_pose_domain_adaptation",
            stage_cfg={"epochs": 1},
            train_dataset=_EpochAwareDataset(),
            train_sampler=_EpochAwareSampler(),
            ema=object(),
            dist_context=_strict_smoke_context(),
        )


def test_real_preflight_rejects_nonfinite_loss(monkeypatch):
    metrics = _finite_metrics()
    metrics['trajectory_loss'] = float('nan')

    with pytest.raises(RuntimeError, match='invalid metrics'):
        _run(monkeypatch, metrics)


def test_real_preflight_requires_optimizer_step(monkeypatch):
    metrics = _finite_metrics()
    metrics['optimizer_steps'] = 0

    with pytest.raises(RuntimeError, match='optimizer_steps=0'):
        _run(monkeypatch, metrics)


def test_heatmap_preflight_reports_real_parameter_delta_and_frozen_lora(monkeypatch):
    model = _ToyHeatmapPipeline()

    def mutate_trainable(module):
        with torch.no_grad():
            module.heatmap_vln.decoder.weight.add_(0.125)

    result, *_ = _run(
        monkeypatch,
        _finite_metrics(),
        model=model,
        stage_cfg={
            'epochs': 2,
            'trainable_modules': ['heatmap_vln'],
        },
        mutation=mutate_trainable,
    )

    assert result['heatmap_changed_tensors'] == 1
    assert result['heatmap_changed_elements'] == 6
    assert result['heatmap_param_delta_max_abs'] == pytest.approx(0.125)
    assert result['heatmap_param_delta_l2'] > 0
    assert result['heatmap_param_delta_relative_l2'] > 0
    assert result['frozen_lora_sampled_tensors'] == 1
    assert result['frozen_lora_sampled_elements'] == 8
    assert result['frozen_lora_sample_max_abs_delta'] == 0


def test_heatmap_preflight_rejects_zero_parameter_update(monkeypatch):
    with pytest.raises(RuntimeError, match='did not change any trainable heatmap'):
        _run(
            monkeypatch,
            _finite_metrics(),
            model=_ToyHeatmapPipeline(),
            stage_cfg={
                'epochs': 2,
                'trainable_modules': ['heatmap_vln'],
            },
        )


def test_heatmap_preflight_rejects_frozen_lora_mutation(monkeypatch):
    model = _ToyHeatmapPipeline()

    def mutate_trainable_and_frozen(module):
        with torch.no_grad():
            module.heatmap_vln.decoder.weight.add_(0.125)
            module.vlm_backbone.lora_A.add_(1)

    with pytest.raises(RuntimeError, match='mutated sampled values from frozen LoRA'):
        _run(
            monkeypatch,
            _finite_metrics(),
            model=model,
            stage_cfg={
                'epochs': 2,
                'trainable_modules': ['heatmap_vln'],
            },
            mutation=mutate_trainable_and_frozen,
        )


def _control_stage_cfg() -> dict:
    return {
        'epochs': 2,
        'trainable_modules': ['heatmap_tokenizer', 'heatmap_control'],
    }


def test_control_preflight_requires_and_reports_real_zero_gate_delta(monkeypatch):
    model = _ToyControlPipeline()

    def mutate_gate(module):
        with torch.no_grad():
            module.nextdit_action_head.traj_dit.model.layers[
                0
            ].heatmap_control.gate[1].add_(0.01)

    result, *_ = _run(
        monkeypatch,
        _finite_metrics(),
        model=model,
        stage_cfg=_control_stage_cfg(),
        mutation=mutate_gate,
    )

    assert result['heatmap_control_changed_gates'] == 1
    assert result['heatmap_control_changed_gate_elements'] == 1
    assert result['heatmap_control_gate_delta_max_abs'] == pytest.approx(0.01)


def test_control_preflight_rejects_weight_decay_only_false_positive(monkeypatch):
    model = _ToyControlPipeline()

    def mutate_non_gate_parameter(module):
        with torch.no_grad():
            module.heatmap_tokenizer.weight.add_(0.01)

    with pytest.raises(RuntimeError, match='did not update any zero-initialized gate'):
        _run(
            monkeypatch,
            _finite_metrics(),
            model=model,
            stage_cfg=_control_stage_cfg(),
            mutation=mutate_non_gate_parameter,
        )


def test_control_preflight_requires_positive_trajectory_loss(monkeypatch):
    model = _ToyControlPipeline()
    metrics = _finite_metrics()
    metrics['trajectory_loss'] = 0.0

    def mutate_gate(module):
        with torch.no_grad():
            module.nextdit_action_head.traj_dit.model.layers[
                0
            ].heatmap_control.gate[0].add_(0.01)

    with pytest.raises(RuntimeError, match='strictly positive'):
        _run(
            monkeypatch,
            metrics,
            model=model,
            stage_cfg=_control_stage_cfg(),
            mutation=mutate_gate,
        )


def test_control_preflight_rejects_nonzero_initial_gate(monkeypatch):
    model = _ToyControlPipeline()
    with torch.no_grad():
        model.nextdit_action_head.traj_dit.model.layers[
            0
        ].heatmap_control.gate[0] = 0.5

    with pytest.raises(RuntimeError, match='exactly zero gates'):
        _run(
            monkeypatch,
            _finite_metrics(),
            model=model,
            stage_cfg=_control_stage_cfg(),
        )
