import logging

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
