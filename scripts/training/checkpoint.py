"""
Checkpoint management: save, load, cleanup, and resume.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from .utils import (
    _normalized_model_state_dict,
    _normalized_trainable_param_names,
    _load_normalized_state_dict,
)

logger = logging.getLogger(__name__)


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
        metrics: Dict,
        cfg: Dict,
        is_best: bool = False,
        scaler: GradScaler = None,
        batch: int = None,
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

        if batch is not None:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}_batch_{batch:05d}.pth"
        else:
            ckpt_path = self.out_dir / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        file_size_mb = ckpt_path.stat().st_size / (1024**2)
        print(f"💾 Saved: {ckpt_path.name} ({file_size_mb:.1f} MB)")

        val_loss = metrics.get('val_loss', float('inf'))
        self.ckpt_history.append((ckpt_path, val_loss, epoch))

        if is_best:
            self.best_val_loss = val_loss
            best_path = self.out_dir / "best.pth"
            torch.save(ckpt, best_path)
            self.best_ckpt_path = best_path
            print(f"⭐ Best model: val_loss={val_loss:.4f}")

        latest_path = self.out_dir / "latest.pth"
        torch.save(ckpt, latest_path)

        self._cleanup_old_ckpts()

        return ckpt_path

    def _cleanup_old_ckpts(self):
        ckpts = sorted(self.out_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.max_ckpts:
            old_ckpt = ckpts.pop(0)
            old_ckpt.unlink()
            print(f"🗑️  Removed old checkpoint: {old_ckpt.name}")

    def load(self, ckpt_path: str) -> Dict:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        return ckpt

    def get_latest(self) -> Optional[Path]:
        latest = self.out_dir / "latest.pth"
        if latest.exists():
            return latest
        legacy_latest = self.out_dir.parent / "ckpts" / "latest.pth"
        return legacy_latest if legacy_latest.exists() else None

    def get_best(self) -> Optional[Path]:
        best = self.out_dir / "best.pth"
        if best.exists():
            return best
        legacy_best = self.out_dir.parent / "ckpts" / "best.pth"
        return legacy_best if legacy_best.exists() else None


def load_checkpoint_for_resume(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler: GradScaler = None,
    logger=None,
) -> Dict:
    """Load checkpoint for resume training."""
    if logger:
        logger.info(f"📂 Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = ckpt.get('trainable_state_dict', {})
    if state_dict:
        missing, unexpected, loaded_count = _load_normalized_state_dict(model, state_dict)
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
        'stage_idx': ckpt.get('stage_idx', 0),
        'stage_name': ckpt.get('stage_name', ''),
        'metrics': ckpt.get('metrics', {}),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
    }
