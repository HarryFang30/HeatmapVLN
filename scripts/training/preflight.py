"""Real one-batch training preflight used by ``scripts/train.py``."""

from __future__ import annotations

import math
from typing import Any

import torch

from .train_loop import train_one_epoch


def _unwrap_model(model):
    """Return the underlying module without depending on a DDP import."""
    return getattr(model, "module", model)


def _snapshot_trainable_heatmap_parameters(
    model,
    stage_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Copy only the trainable heatmap head to CPU for an exact delta check.

    A heatmap-only preflight has a relatively small trainable head, while the
    frozen Qwen backbone is orders of magnitude larger.  Keeping the snapshot
    scoped to ``heatmap_vln`` makes the check exact without cloning the full
    model on every rank.
    """
    if "heatmap_vln" not in set(stage_cfg.get("trainable_modules", ())):
        return {}

    module = _unwrap_model(model)
    if not hasattr(module, "named_parameters"):
        raise RuntimeError(
            "Heatmap training preflight cannot inspect model parameters"
        )

    snapshot = {
        name: param.detach().to(device="cpu", copy=True)
        for name, param in module.named_parameters()
        if name.startswith("heatmap_vln.") and param.requires_grad
    }
    if not snapshot:
        raise RuntimeError(
            "Heatmap training preflight found no trainable heatmap_vln parameters"
        )
    return snapshot


def _snapshot_past_plan_action_parameters(
    model,
    stage_cfg: dict,
) -> dict[str, torch.Tensor]:
    if stage_cfg.get('past_plan_action_stage') is None:
        return {}
    module = _unwrap_model(model)
    snapshot = {
        name: parameter.detach().to(device='cpu', copy=True)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith('past_plan_action.')
            or name.startswith('heatmap_vln.coarse.')
            or name.startswith('heatmap_vln.fine.')
        )
    }
    if not snapshot:
        raise RuntimeError('PPA preflight found no approved trainable parameters')
    return snapshot


def _verify_past_plan_action_parameter_delta(
    model,
    before: dict[str, torch.Tensor],
    stage_cfg: dict,
) -> dict[str, float]:
    if not before:
        return {}
    parameters = dict(_unwrap_model(model).named_parameters())
    changed_by_family = {'future': 0, 'bridge': 0, 'shared_map': 0}
    for name, original in before.items():
        current = parameters[name].detach().cpu()
        if not bool(torch.isfinite(current.float()).all()):
            raise RuntimeError(f'PPA smoke produced non-finite parameter: {name}')
        if torch.equal(current, original):
            continue
        if name.startswith('past_plan_action.future_head.'):
            changed_by_family['future'] += 1
        elif name.startswith('past_plan_action.bridge.'):
            changed_by_family['bridge'] += 1
        else:
            changed_by_family['shared_map'] += 1
    if changed_by_family['future'] == 0:
        raise RuntimeError('PPA smoke did not update the Future Head')
    if (
        stage_cfg.get('past_plan_action_stage') == 'stage2_joint'
        and changed_by_family['bridge'] == 0
    ):
        raise RuntimeError('PPA Stage-2 smoke did not update the zero-init bridge')
    return {
        f'ppa_changed_{family}_tensors': float(count)
        for family, count in changed_by_family.items()
    }


def _evenly_spaced_indices(numel: int, sample_count: int) -> list[int]:
    if numel <= sample_count:
        return list(range(numel))
    if sample_count == 1:
        return [0]
    return [
        round(index * (numel - 1) / (sample_count - 1))
        for index in range(sample_count)
    ]


def _snapshot_frozen_lora_samples(
    model,
    stage_cfg: dict,
    *,
    samples_per_tensor: int = 8,
) -> dict[str, tuple[list[int], torch.Tensor]]:
    """Take a tiny deterministic sample from each frozen LoRA tensor.

    This is an integrity sentinel, not a replacement for the strict trainable
    scope check.  Sampling every frozen LoRA tensor avoids a full backbone
    clone while still catching accidental optimizer/forward mutations across
    layers and projections.
    """
    trainable_modules = set(stage_cfg.get("trainable_modules", ()))
    if (
        "heatmap_vln" not in trainable_modules
        or trainable_modules & {"lora", "vlm_lora"}
    ):
        return {}

    module = _unwrap_model(model)
    if not hasattr(module, "named_parameters"):
        return {}

    snapshot: dict[str, tuple[list[int], torch.Tensor]] = {}
    for name, param in module.named_parameters():
        if "lora_" not in name.lower() or param.requires_grad or param.numel() == 0:
            continue
        indices = _evenly_spaced_indices(
            param.numel(),
            min(samples_per_tensor, param.numel()),
        )
        device_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=param.device,
        )
        values = (
            param.detach()
            .reshape(-1)
            .index_select(0, device_indices)
            .to(device="cpu", copy=True)
        )
        snapshot[name] = (indices, values)
    return snapshot


def _verify_trainable_heatmap_delta(
    model,
    before: dict[str, torch.Tensor],
) -> dict[str, float]:
    if not before:
        return {}

    module = _unwrap_model(model)
    parameters = dict(module.named_parameters())
    missing = sorted(set(before) - set(parameters))
    if missing:
        raise RuntimeError(
            "Heatmap training preflight lost snapshotted parameters: "
            f"{missing[:5]}"
        )

    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    delta_sq_sum = 0.0
    reference_sq_sum = 0.0

    for name, original in before.items():
        current = parameters[name].detach().to(device="cpu", copy=True)
        if not bool(torch.isfinite(current.float()).all()):
            raise RuntimeError(
                "Heatmap training preflight produced non-finite parameters: "
                f"{name}"
            )
        if current.shape != original.shape or current.dtype != original.dtype:
            raise RuntimeError(
                "Heatmap training preflight changed parameter metadata: "
                f"{name} {tuple(original.shape)}/{original.dtype} -> "
                f"{tuple(current.shape)}/{current.dtype}"
            )

        changed = current.ne(original)
        tensor_changed_elements = int(changed.sum().item())
        if tensor_changed_elements:
            changed_tensors += 1
            changed_elements += tensor_changed_elements

        delta = current.float() - original.float()
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        delta_sq_sum += float(
            torch.sum(delta.square(), dtype=torch.float64).item()
        )
        reference_sq_sum += float(
            torch.sum(original.float().square(), dtype=torch.float64).item()
        )

    delta_l2 = math.sqrt(delta_sq_sum)
    reference_l2 = math.sqrt(reference_sq_sum)
    relative_l2 = delta_l2 / max(reference_l2, 1e-12)
    if (
        changed_elements == 0
        or not math.isfinite(delta_l2)
        or not math.isfinite(relative_l2)
        or max_abs_delta <= 0.0
    ):
        raise RuntimeError(
            "Training preflight completed an optimizer step but did not change "
            "any trainable heatmap parameters. This commonly indicates a "
            "zero/underflowed update or a disconnected loss."
        )

    return {
        "heatmap_changed_tensors": float(changed_tensors),
        "heatmap_changed_elements": float(changed_elements),
        "heatmap_param_delta_max_abs": max_abs_delta,
        "heatmap_param_delta_l2": delta_l2,
        "heatmap_param_delta_relative_l2": relative_l2,
    }


def _verify_frozen_lora_samples(
    model,
    before: dict[str, tuple[list[int], torch.Tensor]],
) -> dict[str, float]:
    if not before:
        return {}

    module = _unwrap_model(model)
    parameters = dict(module.named_parameters())
    max_abs_delta = 0.0
    sampled_elements = 0
    changed: list[str] = []

    for name, (indices, original) in before.items():
        param = parameters.get(name)
        if param is None:
            changed.append(f"{name} (missing)")
            continue
        device_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=param.device,
        )
        current = (
            param.detach()
            .reshape(-1)
            .index_select(0, device_indices)
            .to(device="cpu", copy=True)
        )
        sampled_elements += current.numel()
        delta = current.float() - original.float()
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        if not torch.equal(current, original):
            changed.append(name)

    if changed:
        raise RuntimeError(
            "Training preflight mutated sampled values from frozen LoRA "
            f"parameters: {changed[:5]}"
        )
    return {
        "frozen_lora_sampled_tensors": float(len(before)),
        "frozen_lora_sampled_elements": float(sampled_elements),
        "frozen_lora_sample_max_abs_delta": max_abs_delta,
    }


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
    heatmap_before = _snapshot_trainable_heatmap_parameters(model, stage_cfg)
    ppa_before = _snapshot_past_plan_action_parameters(model, stage_cfg)
    frozen_lora_before = _snapshot_frozen_lora_samples(model, stage_cfg)
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
    if stage_cfg.get('past_plan_action_stage') is not None:
        metric_names += (
            'future_heatmap_loss',
            'preserve_loss',
            'delta_z_l2',
            'ppa_bridge_grad_norm',
            'ppa_future_grad_norm',
            'ppa_shared_map_grad_norm',
        )
    invalid = {
        name: metrics.get(name)
        for name in metric_names
        if name not in metrics or not math.isfinite(float(metrics[name]))
    }
    if invalid:
        raise RuntimeError(f'Training preflight produced invalid metrics: {invalid}')
    if stage_cfg.get('verify_stage0_equivalence', False) and not bool(
        metrics.get('stage0_equivalence_passed', False)
    ):
        raise RuntimeError(
            'Past→Plan→Action preflight did not complete the Stage-0 exact '
            'native-equivalence audit'
        )
    if stage_cfg.get('past_plan_action_stage') == 'stage2_joint':
        if metrics.get('ppa_bridge_grad_norm', 0.0) <= 0.0:
            raise RuntimeError('Stage-2 bridge received no gradient in the real smoke')
        if metrics.get('ppa_future_grad_norm', 0.0) <= 0.0:
            raise RuntimeError('Stage-2 Future Head received no gradient in the real smoke')

    delta_metrics = _verify_trainable_heatmap_delta(model, heatmap_before)
    ppa_delta_metrics = _verify_past_plan_action_parameter_delta(
        model, ppa_before, stage_cfg
    )
    if stage_cfg.get('past_plan_action_stage') is not None:
        from src.models.past_plan_action_training import (
            assert_native_frozen_and_gradient_free,
        )

        raw_model = _unwrap_model(model)
        assert_native_frozen_and_gradient_free(
            raw_model.nextdit_action_head,
            raw_model.nextdit_action_head.cond_projector,
        )
    frozen_lora_metrics = _verify_frozen_lora_samples(model, frozen_lora_before)
    metrics.update(delta_metrics)
    metrics.update(ppa_delta_metrics)
    metrics.update(frozen_lora_metrics)

    if delta_metrics:
        logger.info(
            "Heatmap parameter delta: changed=%d tensors/%d elements "
            "max_abs=%.6e l2=%.6e relative_l2=%.6e",
            int(delta_metrics["heatmap_changed_tensors"]),
            int(delta_metrics["heatmap_changed_elements"]),
            delta_metrics["heatmap_param_delta_max_abs"],
            delta_metrics["heatmap_param_delta_l2"],
            delta_metrics["heatmap_param_delta_relative_l2"],
        )
    if frozen_lora_metrics:
        logger.info(
            "Frozen LoRA integrity samples unchanged: %d tensors/%d elements",
            int(frozen_lora_metrics["frozen_lora_sampled_tensors"]),
            int(frozen_lora_metrics["frozen_lora_sampled_elements"]),
        )

    logger.info(
        'Real training preflight passed: loss=%.6f trajectory_loss=%.6f optimizer_steps=%d',
        metrics['total_loss'],
        metrics['trajectory_loss'],
        optimizer_steps,
    )
    return metrics
