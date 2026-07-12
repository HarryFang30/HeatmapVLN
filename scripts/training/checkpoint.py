"""
Checkpoint management: save, load, cleanup, and resume.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

from .utils import (
    _load_normalized_state_dict,
    _normalize_state_key,
    _normalized_model_state_dict,
    _normalized_trainable_param_names,
    _unwrap_model,
    safe_torch_load,
)

logger = logging.getLogger(__name__)


def _materialize_and_validate_heatmap_checkpoint_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    checkpoint_path: str | None = None,
) -> int:
    """Materialize a lazy HeatmapVLN head and validate every saved head tensor.

    ``VLNPipeline`` cannot construct ``heatmap_vln`` until its lazy Qwen
    backbone exists.  Loading a checkpoint before the first forward therefore
    used to make all ``heatmap_vln.*`` entries look unexpected; with
    ``strict=False`` they were silently discarded.  Detect those entries before
    loading, construct the head through the pipeline's public lazy initializer,
    and require every saved heatmap tensor to have a shape-compatible target.
    """
    heatmap_state = {
        _normalize_state_key(name): value
        for name, value in state_dict.items()
        if _normalize_state_key(name).startswith("heatmap_vln.")
    }
    if not heatmap_state:
        return 0

    raw_model = _unwrap_model(model)
    if getattr(raw_model, "heatmap_vln", None) is None:
        ensure_heatmap = getattr(raw_model, "_ensure_heatmap_vln", None)
        if not callable(ensure_heatmap):
            raise RuntimeError(
                "Checkpoint contains HeatmapVLN tensors, but the target model "
                "does not provide _ensure_heatmap_vln()"
            )
        ensure_heatmap()

    if getattr(raw_model, "heatmap_vln", None) is None:
        source = f" from {checkpoint_path}" if checkpoint_path else ""
        raise RuntimeError(
            "Checkpoint contains HeatmapVLN tensors"
            f"{source}, but the heatmap head is disabled or could not be constructed"
        )

    model_state = _normalized_model_state_dict(model)
    missing = sorted(name for name in heatmap_state if name not in model_state)
    shape_mismatches = sorted(
        name
        for name, value in heatmap_state.items()
        if name in model_state and tuple(model_state[name].shape) != tuple(value.shape)
    )
    if missing or shape_mismatches:
        source = f" from {checkpoint_path}" if checkpoint_path else ""
        missing_preview = ", ".join(missing[:5])
        mismatch_preview = ", ".join(
            f"{name}: ckpt {tuple(heatmap_state[name].shape)} vs model {tuple(model_state[name].shape)}"
            for name in shape_mismatches[:5]
        )
        raise RuntimeError(
            "Incomplete HeatmapVLN checkpoint load refused"
            f"{source}: saved={len(heatmap_state)} missing={len(missing)} "
            f"shape_mismatches={len(shape_mismatches)}"
            + (f" missing_preview=[{missing_preview}]" if missing_preview else "")
            + (f" shape_mismatch_preview=[{mismatch_preview}]" if mismatch_preview else "")
        )

    return len(heatmap_state)


def load_checkpoint_model_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    checkpoint_path: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[str], list[str], int]:
    """Load model weights, including shape-checked lazy HeatmapVLN tensors."""
    heatmap_count = _materialize_and_validate_heatmap_checkpoint_state(
        model,
        state_dict,
        checkpoint_path=checkpoint_path,
    )
    missing, unexpected, loaded_count = _load_normalized_state_dict(model, state_dict)
    if heatmap_count:
        target_logger = logger or globals()["logger"]
        target_logger.info(
            "  ✓ Verified and loaded %d HeatmapVLN checkpoint tensors",
            heatmap_count,
        )
    return missing, unexpected, loaded_count


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup."""

    def __init__(self, out_dir: str, max_ckpts: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_ckpts = max_ckpts
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None
        self.ckpt_history = []

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
    ) -> Path:
        """Save checkpoint. ``batch`` being not None produces a mid-epoch save."""
        trainable_params = _normalized_trainable_param_names(model)
        normalized_state_dict = _normalized_model_state_dict(model)
        trainable_state_dict = {
            k: v for k, v in normalized_state_dict.items()
            if k in trainable_params
        }

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
        }

        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        if extra_state:
            ckpt.update(extra_state)

        if batch is not None:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}_batch_{batch:05d}.pth"
        else:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        file_size_mb = ckpt_path.stat().st_size / (1024**2)
        logger.info("💾 Saved: %s (%.1f MB)", ckpt_path.name, file_size_mb)

        val_loss = metrics.get('val_loss', float('inf'))
        self.ckpt_history.append((ckpt_path, val_loss, epoch))

        if is_best:
            self.best_val_loss = val_loss
            best_path = self.out_dir / "best.pth"
            torch.save(ckpt, best_path)
            self.best_ckpt_path = best_path
            logger.info("⭐ Best model: val_loss=%.4f", val_loss)

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
    logger=None,
) -> dict:
    """Load checkpoint for resume training."""
    if logger:
        logger.info(f"📂 Loading checkpoint: {ckpt_path}")

    ckpt = safe_torch_load(ckpt_path)

    state_dict = (
        ckpt.get('trainable_state_dict')
        or ckpt.get('model_state_dict')
        or ckpt.get('state_dict')
        or {}
    )
    if state_dict:
        _missing, _unexpected, loaded_count = load_checkpoint_model_state(
            model,
            state_dict,
            checkpoint_path=ckpt_path,
            logger=logger,
        )
        if logger:
            logger.info(f"  ✓ Loaded {loaded_count} trainable parameters")

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if logger:
                logger.info("  ✓ Optimizer state restored")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠ Failed to restore optimizer: {e}")

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if logger:
                logger.info("  ✓ Scheduler state restored")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠ Failed to restore scheduler: {e}")

    if scaler is not None and 'scaler_state_dict' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            if logger:
                logger.info("  ✓ GradScaler state restored")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠ Failed to restore scaler: {e}")

    return {
        'epoch': ckpt.get('epoch', 0),
        'batch': ckpt.get('batch'),
        'stage_idx': ckpt.get('stage_idx', 0),
        'stage_name': ckpt.get('stage_name', ''),
        'metrics': ckpt.get('metrics', {}),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
        'l2_sp_reference_state': ckpt.get('l2_sp_reference_state'),
    }
