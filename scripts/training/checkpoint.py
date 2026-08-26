"""
Checkpoint management: save, load, cleanup, and resume.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

from .utils import (
    _load_normalized_state_dict,
    _normalize_state_key,
    _normalized_model_state_dict,
    _normalized_trainable_param_names,
    safe_torch_load,
)

logger = logging.getLogger(__name__)


def _checkpoint_stage_config(
    cfg: dict,
    stage_idx: int,
    stage_name: str,
) -> dict:
    """Resolve the stage responsible for the checkpoint without mutating cfg."""
    stages = cfg.get('training', {}).get('stages', [])
    if 0 <= stage_idx < len(stages):
        candidate = stages[stage_idx]
        if not stage_name or candidate.get('name') == stage_name:
            return candidate
    for candidate in stages:
        if candidate.get('name') == stage_name:
            return candidate
    return {}


def _should_include_frozen_lora_in_deployment(
    cfg: dict,
    stage_idx: int,
    stage_name: str,
) -> bool:
    """Whether a dependent stage should emit a self-contained LoRA payload.

    Stages that merge LoRA into a frozen dense backbone cannot export those
    adapters separately.  They retain their existing base-chain semantics.
    Other stages may explicitly opt in/out; otherwise a declared base
    dependency enables the safer, self-contained deployment representation.
    """
    stage_cfg = _checkpoint_stage_config(cfg, stage_idx, stage_name)
    explicit = stage_cfg.get('deployment_include_frozen_lora')
    if explicit is not None:
        return bool(explicit)
    requires_base = bool(
        stage_cfg.get('requires_base_checkpoint', False)
        or stage_cfg.get('bridge_only', False)
    )
    return requires_base and not bool(stage_cfg.get('merge_frozen_lora', False))


def _normalized_frozen_lora_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Export each physical frozen LoRA parameter exactly once.

    ``state_dict()`` contains aliases for the shared Qwen module
    (``qwen2_5_vl``, ``vlm_backbone``, and ``heatmap_vln.qwen``).
    ``named_parameters()`` de-duplicates shared Parameter objects, which keeps
    the deployment payload canonical and compatible with strict evaluation.
    """
    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        normalized_name = _normalize_state_key(name)
        if 'lora_' not in normalized_name or parameter.requires_grad:
            continue
        if normalized_name in result:
            raise RuntimeError(
                "Multiple physical frozen LoRA parameters normalize to the "
                f"same checkpoint key: {normalized_name}"
            )
        result[normalized_name] = parameter.detach()
    return result


def _normalized_lora_parameter_names(model: nn.Module) -> set[str]:
    return {
        _normalize_state_key(name)
        for name, _parameter in model.named_parameters()
        if 'lora_' in _normalize_state_key(name)
    }


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup."""

    def __init__(self, out_dir: str, max_ckpts: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_ckpts = max_ckpts
        self.best_val_loss = float('inf')
        self.best_metric_name = 'val_loss'
        self.best_metric_mode = 'min'
        self.best_metric_value = float('inf')
        self.best_ckpt_path = None
        self.ckpt_history = []

    def configure_best_metric(self, name: str, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {'min', 'max'}:
            raise ValueError(
                f"validation.save_best_mode must be 'min' or 'max', got {mode!r}"
            )
        self.best_metric_name = str(name)
        self.best_metric_mode = mode
        self.best_metric_value = (
            float('inf') if mode == 'min' else float('-inf')
        )

    def is_better(self, value: float) -> bool:
        value = float(value)
        if not math.isfinite(value):
            return False
        if self.best_metric_mode == 'min':
            return value < self.best_metric_value
        return value > self.best_metric_value

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        stage_idx: int,
        stage_name: str,
        metrics: dict,
        cfg: dict,
        is_best: bool = False,
        scaler: GradScaler | None = None,
        batch: int | None = None,
        extra_state: dict | None = None,
        ema=None,
        best_only: bool = False,
    ) -> Path:
        """Save checkpoint. ``batch`` being not None produces a mid-epoch save.

        With EMA enabled, ``trainable_state_dict`` remains the deployment/init
        weight entry for backwards compatibility and contains EMA weights.
        ``online_trainable_state_dict`` stores the matching optimizer weights
        used for an exact training resume.  A dependent, non-merged-LoRA stage
        additionally exports frozen LoRA tensors in the deployment entry while
        keeping them out of the optimizer-matched online entry.

        ``best_only`` is intended for a pre-training validation incumbent.  It
        writes only ``best.pth`` and deliberately leaves ``latest.pth`` and the
        epoch checkpoint history untouched.
        """
        if best_only and not is_best:
            raise ValueError("best_only=True requires is_best=True")

        trainable_params = _normalized_trainable_param_names(model)
        normalized_state_dict = _normalized_model_state_dict(model)
        online_trainable_state_dict = {
            k: v for k, v in normalized_state_dict.items()
            if k in trainable_params
        }
        trainable_state_dict = online_trainable_state_dict

        ema_state_dict = None
        if ema is not None:
            ema_state_dict = ema.state_dict()
            normalized_ema_shadow = {
                _normalize_state_key(name): value
                for name, value in ema_state_dict['shadow'].items()
            }
            missing_ema = sorted(trainable_params - set(normalized_ema_shadow))
            if missing_ema:
                raise RuntimeError(
                    "EMA checkpoint is missing trainable parameters: "
                    f"{missing_ema[:5]}"
                )
            trainable_state_dict = {
                name: normalized_ema_shadow[name]
                for name in trainable_params
            }

        deployment_trainable_tensor_count = len(trainable_state_dict)
        stage_cfg = _checkpoint_stage_config(cfg, stage_idx, stage_name)
        ppa_stage = stage_cfg.get('past_plan_action_stage')
        include_complete_heatmap_head = bool(
            stage_cfg.get('heatmap_trainable_parameter_prefixes')
            or ppa_stage is not None
        )
        frozen_heatmap_head_tensor_count = 0
        complete_heatmap_head_tensor_count = 0
        if include_complete_heatmap_head:
            from .pose_adaptation import (
                EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS,
                complete_heatmap_head_state,
            )

            complete_head = complete_heatmap_head_state(model)
            complete_heatmap_head_tensor_count = len(complete_head)
            frozen_heatmap_head_tensor_count = len(
                set(complete_head) - set(trainable_state_dict)
            )
            trainable_state_dict = {
                **complete_head,
                **trainable_state_dict,
            }
            if ppa_stage is None and (
                len(trainable_state_dict) != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS
            ):
                raise RuntimeError(
                    "AMB3R pose-adaptation deployment state is not the exact "
                    f"{EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS}-tensor Head"
                )
            if not set(complete_head).issubset(trainable_state_dict):
                raise RuntimeError(
                    "Deployment state is missing learned Heatmap Head tensors"
                )
        frozen_lora_state_dict: dict[str, torch.Tensor] = {}
        include_frozen_lora = _should_include_frozen_lora_in_deployment(
            cfg,
            stage_idx,
            stage_name,
        )
        if include_frozen_lora:
            frozen_lora_state_dict = _normalized_frozen_lora_state_dict(model)
            all_lora_names = _normalized_lora_parameter_names(model)
            if not all_lora_names:
                raise RuntimeError(
                    "Self-contained deployment checkpoint requested for a "
                    "base-dependent stage, but the model has no LoRA parameters"
                )
            # Deployment uses EMA for trainable tensors and the exact frozen
            # tensors loaded from the declared base.  A trainable LoRA tensor,
            # if present, intentionally wins over the frozen-state seed.
            trainable_state_dict = {
                **frozen_lora_state_dict,
                **trainable_state_dict,
            }
            missing_lora = sorted(
                all_lora_names - set(trainable_state_dict)
            )
            if missing_lora:
                raise RuntimeError(
                    "Self-contained deployment checkpoint is missing LoRA "
                    f"parameters: {missing_lora[:5]}"
                )

        val_loss = metrics.get('val_loss', float('inf'))
        if math.isfinite(float(val_loss)):
            self.best_val_loss = min(self.best_val_loss, float(val_loss))
        if is_best:
            if self.best_metric_name not in metrics:
                raise KeyError(
                    "Best-checkpoint metric is absent from validation output: "
                    f"{self.best_metric_name!r}; available={sorted(metrics)}"
                )
            selected_value = float(metrics[self.best_metric_name])
            if not math.isfinite(selected_value):
                raise ValueError(
                    f"Best-checkpoint metric {self.best_metric_name!r} is not finite: "
                    f"{selected_value}"
                )
            self.best_metric_value = selected_value

        ckpt = {
            'epoch': epoch,
            'batch': batch,
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'trainable_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': cfg,
            'best_val_loss': self.best_val_loss,
            'best_metric_name': self.best_metric_name,
            'best_metric_mode': self.best_metric_mode,
            'best_metric_value': self.best_metric_value,
        }
        if ema_state_dict is not None:
            ckpt.update({
                'online_trainable_state_dict': online_trainable_state_dict,
                'ema_state_dict': ema_state_dict,
                'weight_semantics': {
                    'trainable_state_dict': (
                        'ema_trainable_plus_frozen_lora'
                        if include_frozen_lora
                        else (
                            'ema_trainable_plus_frozen_heatmap_head'
                            if include_complete_heatmap_head
                            else 'ema'
                        )
                    ),
                    'online_trainable_state_dict': 'optimizer_matched_online',
                },
            })
        elif include_frozen_lora:
            ckpt['weight_semantics'] = {
                'trainable_state_dict': 'online_trainable_plus_frozen_lora',
            }
        elif include_complete_heatmap_head:
            ckpt['weight_semantics'] = {
                'trainable_state_dict': (
                    'online_trainable_plus_frozen_heatmap_head'
                ),
            }
        if include_frozen_lora:
            ckpt['deployment_state_manifest'] = {
                'requires_base_checkpoint': True,
                'base_checkpoint': cfg.get('runtime', {}).get('base_checkpoint'),
                'included_frozen_lora': True,
                'frozen_lora_tensor_count': len(frozen_lora_state_dict),
                'deployment_trainable_tensor_count': (
                    deployment_trainable_tensor_count
                ),
                'deployment_tensor_count': len(trainable_state_dict),
                'online_trainable_tensor_count': len(
                    online_trainable_state_dict
                ),
            }
        if include_complete_heatmap_head:
            ckpt['deployment_state_manifest'] = {
                **ckpt.get('deployment_state_manifest', {}),
                'self_contained_heatmap_head': True,
                'heatmap_head_tensor_count': complete_heatmap_head_tensor_count,
                'frozen_heatmap_head_tensor_count': (
                    frozen_heatmap_head_tensor_count
                ),
                'online_trainable_tensor_count': len(
                    online_trainable_state_dict
                ),
            }
        if ppa_stage is not None:
            future_names = {
                _normalize_state_key(name)
                for name, _parameter in model.named_parameters()
                if _normalize_state_key(name).startswith(
                    'past_plan_action.future_head.'
                )
            }
            complete_future = {
                name: normalized_state_dict[name]
                for name in future_names
            }
            # Bridge-only action refinement freezes the Future Head, but its
            # deployment checkpoint must remain self-contained.  Trainable
            # EMA tensors override this frozen snapshot in ordinary stages.
            trainable_state_dict = {
                **complete_future,
                **trainable_state_dict,
            }
            # ``ckpt`` was constructed above.  Rebind its deployment entry to
            # the self-contained mapping after adding frozen Future tensors;
            # otherwise it retains the pre-merge dict object.
            ckpt['trainable_state_dict'] = trainable_state_dict
            missing_future = sorted(future_names - set(trainable_state_dict))
            if not future_names or missing_future:
                raise RuntimeError(
                    "Past->Plan->Action deployment state lacks the complete "
                    f"Future Head: missing={missing_future[:8]}"
                )
            ckpt['past_plan_action_contract'] = {
                'schema': 'past-plan-action-checkpoint-v1',
                'stage': str(ppa_stage),
                'complete_heatmap_head_tensors': (
                    complete_heatmap_head_tensor_count
                ),
                'complete_future_head_tensors': len(future_names),
                'bridge_in_deployment_state': any(
                    name.startswith('past_plan_action.bridge.')
                    for name in trainable_state_dict
                ),
                'stage1_to_stage2_fresh_optimizer': True,
                'checkpoint_digest_enforced': False,
                'file_lock_used': False,
            }

        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        if extra_state:
            ckpt.update(extra_state)

        if best_only:
            best_path = self.out_dir / "best.pth"
            torch.save(ckpt, best_path)
            self.best_ckpt_path = best_path
            file_size_mb = best_path.stat().st_size / (1024**2)
            logger.info(
                "⭐ Saved baseline incumbent: %s=%.6f (%s, %.1f MB)",
                self.best_metric_name,
                self.best_metric_value,
                self.best_metric_mode,
                file_size_mb,
            )
            return best_path

        if batch is not None:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}_batch_{batch:05d}.pth"
        else:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        file_size_mb = ckpt_path.stat().st_size / (1024**2)
        logger.info("💾 Saved: %s (%.1f MB)", ckpt_path.name, file_size_mb)

        self.ckpt_history.append((ckpt_path, val_loss, epoch))

        if is_best:
            best_path = self.out_dir / "best.pth"
            torch.save(ckpt, best_path)
            self.best_ckpt_path = best_path
            logger.info(
                "⭐ Best model: %s=%.6f (%s)",
                self.best_metric_name,
                self.best_metric_value,
                self.best_metric_mode,
            )

        latest_path = self.out_dir / "latest.pth"
        torch.save(ckpt, latest_path)

        self._cleanup_old_ckpts()

        return ckpt_path

    def _cleanup_old_ckpts(self):
        ckpts = sorted(self.out_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.max_ckpts:
            old_ckpt = ckpts.pop(0)
            old_ckpt.unlink()
            logger.info("🗑️  Removed old checkpoint: %s", old_ckpt.name)

    def load(self, ckpt_path: str) -> dict:
        ckpt = safe_torch_load(ckpt_path)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.best_metric_name = ckpt.get('best_metric_name', 'val_loss')
        self.best_metric_mode = ckpt.get('best_metric_mode', 'min')
        self.best_metric_value = ckpt.get(
            'best_metric_value',
            self.best_val_loss,
        )
        return ckpt

    def get_latest(self) -> Path | None:
        latest = self.out_dir / "latest.pth"
        if latest.exists():
            return latest
        legacy_latest = self.out_dir.parent / "ckpts" / "latest.pth"
        return legacy_latest if legacy_latest.exists() else None

    def get_best(self) -> Path | None:
        best = self.out_dir / "best.pth"
        if best.exists():
            return best
        legacy_best = self.out_dir.parent / "ckpts" / "best.pth"
        return legacy_best if legacy_best.exists() else None


def load_checkpoint_for_resume(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: GradScaler | None = None,
    ema=None,
    logger=None,
    strict_state_restore: bool = False,
    metadata_only: bool = False,
) -> dict:
    """Load checkpoint for resume training."""
    if logger:
        logger.info(f"📂 Loading checkpoint: {ckpt_path}")

    ckpt = safe_torch_load(ckpt_path)

    deployment_manifest = ckpt.get('deployment_state_manifest') or {}
    if not metadata_only and deployment_manifest.get('self_contained_heatmap_head'):
        from .pose_adaptation import EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS

        deployment_state = ckpt.get('trainable_state_dict') or {}
        if len(deployment_state) != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS:
            raise RuntimeError(
                "Self-contained pose-adaptation resume expected exact complete "
                f"Heatmap Head ({EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS} "
                f"tensors), got {len(deployment_state)}"
            )
        _missing, _unexpected, deployment_loaded = _load_normalized_state_dict(
            model,
            deployment_state,
        )
        if deployment_loaded != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS:
            raise RuntimeError(
                "Failed to restore the complete frozen+trainable Heatmap Head "
                f"before exact resume: loaded={deployment_loaded}"
            )
        if logger:
            logger.info(
                "  ✓ Restored self-contained complete Heatmap Head (%d tensors)",
                deployment_loaded,
            )

    state_key = (
        'online_trainable_state_dict'
        if ckpt.get('online_trainable_state_dict')
        else 'trainable_state_dict'
    )
    state_dict = ckpt.get(state_key, {})
    if not metadata_only and state_dict:
        _missing, _unexpected, loaded_count = _load_normalized_state_dict(model, state_dict)
        if logger:
            logger.info(
                "  ✓ Loaded %d trainable parameters for resume from %s",
                loaded_count,
                state_key,
            )

    def _restore_runtime_state(target, state_key: str, label: str) -> None:
        if target is None:
            return
        if state_key not in ckpt:
            if strict_state_restore:
                raise RuntimeError(
                    f"Strict resume checkpoint is missing {state_key}"
                )
            return
        try:
            target.load_state_dict(ckpt[state_key])
            if logger:
                logger.info("  ✓ %s state restored", label)
        except Exception as exc:
            if strict_state_restore:
                raise RuntimeError(
                    f"Strict resume failed to restore {label} state"
                ) from exc
            if logger:
                logger.warning("  ⚠ Failed to restore %s: %s", label, exc)

    _restore_runtime_state(
        None if metadata_only else optimizer,
        'optimizer_state_dict',
        'Optimizer',
    )
    _restore_runtime_state(
        None if metadata_only else scheduler,
        'scheduler_state_dict',
        'Scheduler',
    )
    _restore_runtime_state(
        None if metadata_only else scaler,
        'scaler_state_dict',
        'GradScaler',
    )

    if ema is not None and not metadata_only:
        if 'ema_state_dict' in ckpt:
            try:
                ema.load_state_dict(ckpt['ema_state_dict'])
            except Exception as exc:
                if strict_state_restore:
                    raise RuntimeError(
                        "Strict resume failed to restore EMA state"
                    ) from exc
                raise
            if logger:
                logger.info(
                    "  ✓ EMA state restored (step=%d)",
                    ema.step_count,
                )
        else:
            if strict_state_restore:
                raise RuntimeError(
                    "Strict resume checkpoint is missing ema_state_dict"
                )
            # Legacy checkpoints stored EMA weights directly in
            # trainable_state_dict but did not record an EMA shadow/step.
            # Start both online and shadow from the loaded weights so the
            # resumed optimizer cannot be paired with a random pre-load EMA.
            ema.reset_from_model()
            if logger:
                logger.warning(
                    "  ⚠ Legacy checkpoint has no EMA state; initialized EMA "
                    "from the loaded model weights"
                )

    return {
        'epoch': ckpt.get('epoch', 0),
        'batch': ckpt.get('batch'),
        'stage_idx': ckpt.get('stage_idx', 0),
        'stage_name': ckpt.get('stage_name', ''),
        'metrics': ckpt.get('metrics', {}),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
        'best_metric_name': ckpt.get('best_metric_name', 'val_loss'),
        'best_metric_mode': ckpt.get('best_metric_mode', 'min'),
        'best_metric_value': ckpt.get(
            'best_metric_value',
            ckpt.get('best_val_loss', float('inf')),
        ),
        'l2_sp_reference_state': ckpt.get('l2_sp_reference_state'),
        'checkpoint_selection_state': ckpt.get('checkpoint_selection_state'),
        'mixture_sampler_state': ckpt.get('mixture_sampler_state'),
    }
