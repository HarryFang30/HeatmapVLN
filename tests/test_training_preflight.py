import logging

import pytest
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


def _run(monkeypatch, metrics):
    captured = {}

    def fake_train_one_epoch(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return metrics

    monkeypatch.setattr(preflight, 'train_one_epoch', fake_train_one_epoch)
    dataset = _EpochAwareDataset()
    sampler = _EpochAwareSampler()
    result = preflight.run_training_preflight(
        object(),
        _OneBatchLoader(),
        object(),
        object(),
        None,
        {},
        logging.getLogger('test.preflight'),
        stage_name='stage3',
        stage_cfg={'epochs': 2},
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
