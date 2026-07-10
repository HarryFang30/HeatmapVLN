"""Real one-batch training preflight used by ``scripts/train.py``."""

from __future__ import annotations

import math
from typing import Any

from .train_loop import train_one_epoch


def run_training_preflight(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    cfg: dict,
    logger,
    *,
    stage_name: str,
    stage_cfg: dict,
    train_dataset,
    train_sampler=None,
    gpu_heatmap_computer=None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    ema=None,
    total_train_steps: int = 1,
    dist_context=None,
    nextdit_warmup_steps: int = 0,
    l2_sp_reference: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Execute one complete train batch and require a finite optimizer step.

    The optimizer/EMA mutation is process-local: the caller exits without
    saving a checkpoint after this function succeeds.
    """
    if len(train_loader) < 1:
        raise RuntimeError('Training preflight requires at least one full batch')

    preflight_epoch = 1
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(preflight_epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(preflight_epoch)

    logger.info(
        'Running real training preflight: one batch with forward, loss, backward, '
        'DDP gradient synchronization, and an in-memory optimizer step'
    )
    metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        cfg,
        preflight_epoch,
        logger,
        tb_writer=None,
        global_step_offset=0,
        stage_idx=0,
        stage_name=stage_name,
        stage_cfg=stage_cfg,
        max_batches=1,
        skip_first_n_batches=None,
        vis_dir=None,
        gpu_heatmap_computer=gpu_heatmap_computer,
        gpu_has_depth=gpu_has_depth,
        gpu_depth_normalized=gpu_depth_normalized,
        ema=ema,
        metrics_jsonl_path=None,
        total_train_steps=total_train_steps,
        dist_context=dist_context,
        ckpt_manager=None,
        mid_epoch_save_every=0,
        nextdit_warmup_steps=nextdit_warmup_steps,
        l2_sp_reference=l2_sp_reference,
    )

    optimizer_steps = int(metrics.get('optimizer_steps', 0))
    if optimizer_steps != 1:
        raise RuntimeError(
            'Training preflight did not complete exactly one optimizer step: '
            f'optimizer_steps={optimizer_steps}'
        )

    metric_names = (
        'total_loss',
        'heatmap_loss',
        'trajectory_loss',
        'lm_loss',
        'l2_sp_loss',
    )
    invalid = {
        name: metrics.get(name)
        for name in metric_names
        if name not in metrics or not math.isfinite(float(metrics[name]))
    }
    if invalid:
        raise RuntimeError(f'Training preflight produced invalid metrics: {invalid}')

    logger.info(
        'Real training preflight passed: loss=%.6f trajectory_loss=%.6f optimizer_steps=%d',
        metrics['total_loss'],
        metrics['trajectory_loss'],
        optimizer_steps,
    )
    return metrics
