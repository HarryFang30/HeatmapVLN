#!/usr/bin/env python3
"""
Stage 1: History头预热 (64x64)

目标：让history头学会基本的热力图生成
训练：history_head, history_validity
冻结：所有其他模块

运行方式：
  # 单GPU
  python train_stage1_history_warmup.py --config configs/training_config_full_model.yaml

  # 多GPU
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_stage1_history_warmup.py --config configs/training_config_full_model.yaml
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
from datetime import datetime

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import functools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_utils_spatial import *
from src.utils.loss import NavigationHeatmapLoss

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] [%(levelname)s] %(message)s') ##修改debug级别
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Stage 1: History头预热')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='从当前阶段checkpoint恢复')
    args = parser.parse_args()

    # Setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    cfg = load_config(args.config)

    # 获取Stage 1配置
    stage_cfg = cfg['training']['stages'][0]  # stage1_history_warmup
    stage_name = stage_cfg['name']

    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 1: History头预热 (64x64)")
    logger.info(f"{'='*80}")
    logger.info(f"Epochs: {stage_cfg['epochs']}")
    logger.info(f"Heatmap size: {stage_cfg['hm_size']}")
    logger.info(f"Train history: {stage_cfg.get('train_history', True)}")
    logger.info(f"Train future: {stage_cfg.get('train_future', False)}")
    logger.info(f"{'='*80}\n")

    # Build data
    train_loader, val_loader = build_dataloaders(cfg, rank, world_size)

    # ⭐ FIX: Build model with local_rank for DDP compatibility
    model = build_model(cfg, local_rank)

    # Update hm_size
    hm_size = tuple(stage_cfg['hm_size'])
    if hasattr(model, 'update_heatmap_size'):
        model.update_heatmap_size(hm_size)
    train_loader.dataset.hm_size = hm_size
    val_loader.dataset.hm_size = hm_size

    # Freeze/unfreeze
    freeze_unfreeze_modules(model, stage_cfg)

    # ⭐ FSDP: Use Fully Sharded Data Parallel for large model compatibility
    if world_size > 1:
        target_device = torch.device(f"cuda:{local_rank}")
        logger.info(f"[Rank {rank}] Preparing model for FSDP on {target_device}...")

        # Move model to correct device
        model = model.to(target_device)
        logger.info(f"[Rank {rank}] ✅ Model moved to {target_device}")

        # Configure FSDP with mixed precision for bfloat16
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Use size-based auto wrap policy (wrap modules > 100M parameters)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=100_000_000  # 100M parameters threshold
        )

        # Wrap with FSDP - uses FULL_SHARD strategy (ZeRO-3 equivalent)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3: shard params, grads, optimizer states
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=local_rank,
            limit_all_gathers=True,  # Memory optimization
            use_orig_params=True,  # Compatibility with optimizer
        )
        logger.info(f"[Rank {rank}] ✅ FSDP wrapper applied successfully (FULL_SHARD strategy)")

    # Optimizer & Scheduler
    # ⭐ FIX: Use math.ceil to align scheduler with actual optimizer steps (including tail gradients)
    total_steps = math.ceil(len(train_loader) / cfg['optim']['grad_accum_steps']) * stage_cfg['epochs']
    optimizer = build_optimizer(model, stage_cfg, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ⭐ FIX: Conditional scaler creation based on AMP mode
    if cfg['optim']['amp'] == 'fp16':
        scaler = GradScaler()
    else:
        scaler = None  # bf16 doesn't need scaler, none means FP32

    # Loss
    criterion = NavigationHeatmapLoss(
        alpha=cfg['loss']['alpha'],
        lambda_mse=cfg['loss']['lambda_mse'],
        lambda_kl=cfg['loss']['lambda_kl'],
        lambda_valid=cfg['loss']['lambda_valid']
    )

    # TensorBoard setup (only on rank 0)
    tb_writer = None
    if rank == 0 and cfg['log'].get('use_tensorboard', False):
        tb_dir = Path(cfg['log'].get('tensorboard_dir', '/root/tf-logs'))
        tb_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_writer = SummaryWriter(log_dir=str(tb_dir / f'stage1_{timestamp}'))
        logger.info(f"📊 TensorBoard enabled: {tb_dir / f'stage1_{timestamp}'}")

    # Training loop
    best_val_loss = float('inf')
    stage_idx = 0  # ⭐ Stage 1 对应 idx=0

    start_epoch = 1
    if args.resume:
        resumed = load_training_state(model, optimizer, scheduler, args.resume, device, scaler)
        if resumed:
            start_epoch = resumed[0]

    for epoch in range(start_epoch, stage_cfg['epochs'] + 1):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # ⭐ 增强：Epoch 开始日志（仅 rank 0）
        if rank == 0:
            logger.info("=" * 80)
            logger.info(
                f"[Stage {stage_idx+1}/{len(cfg['training']['stages'])}: {stage_name}] "
                f"Epoch {epoch}/{stage_cfg['epochs']} Started"
            )
            lr_list = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            logger.info(
                f"  Heatmap Size: {hm_size} | Batch Size: {cfg['optim']['batch_size']} | "
                f"Grad Accum: {cfg['optim']['grad_accum_steps']} | "
                f"Learning Rates: {lr_list}"
            )
            logger.info("=" * 80)

        # ⭐ 传递 stage_idx 和 stage_name 到 train_one_epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            stage_cfg, cfg, epoch, rank, device, criterion,
            stage_idx=stage_idx,
            stage_name=stage_name,
            tb_writer=tb_writer
        )

        # ⭐ FIX Bug 1: All ranks execute validation (metrics aggregated via all-reduce)
        val_metrics = validate(model, val_loader, criterion, stage_cfg, cfg, device, world_size, rank, tb_writer=tb_writer, epoch=epoch)

        # Only rank 0 logs and saves checkpoints
        if rank == 0:
            # ⭐ 增强：Epoch 完成日志
            logger.info("=" * 80)
            logger.info(
                f"[Stage {stage_idx+1}: {stage_name}] "
                f"Epoch {epoch}/{stage_cfg['epochs']} Completed"
            )
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['total_loss']:.4f}")

            is_best = val_metrics['total_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['total_loss']
                logger.info(f"  ⭐ New Best Val Loss: {best_val_loss:.4f}")

            if epoch % cfg['log']['save_every_epochs'] == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_name,
                    {'train': train_metrics, 'val': val_metrics},
                    cfg, is_best, scaler
                )
            logger.info("=" * 80)

    if world_size > 1:
        torch.distributed.destroy_process_group()

    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
            logger.info("📊 TensorBoard writer closed")
        logger.info(f"\n✅ Stage 1 completed! Best val loss: {best_val_loss:.4f}")
        logger.info(f"下一步: python train_stage2_history_mlp.py --config {args.config}\n")


if __name__ == '__main__':
    main()
