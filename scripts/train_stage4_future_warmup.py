#!/usr/bin/env python3
"""
Stage 4: Future头预热 (128)
训练：future_head, future_validity
冻结：所有其他（包括已训练好的history）
"""

import os, sys, argparse, logging, math
from pathlib import Path
from datetime import datetime
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_utils_spatial import *
from src.utils.loss import NavigationHeatmapLoss

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Future头预热')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='从当前阶段checkpoint恢复')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    cfg = load_config(args.config)

    stage_cfg = cfg['training']['stages'][3]  # stage4_future_warmup
    stage_name = stage_cfg['name']

    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 4: Future头预热 (128) - 在已训练backbone上训练future")
    logger.info(f"{'='*80}\n")

    train_loader, val_loader = build_dataloaders(cfg, rank, world_size)
    # ⭐ FIX: Build model with local_rank for DDP compatibility
    model = build_model(cfg, local_rank)

    # 动态获取上一阶段checkpoint，避免硬编码
    try:
        stage_idx = next(i for i, s in enumerate(cfg['training']['stages']) if s['name'] == stage_name)
        prev_stage_name = cfg['training']['stages'][stage_idx - 1]['name']
    except Exception:
        prev_stage_name = "stage3_history_full"

    prev_ckpt = Path(cfg['log']['out_dir']) / f"{prev_stage_name}_best.pth"
    if prev_ckpt.exists():
        load_checkpoint(model, str(prev_ckpt), device)
    else:
        logger.warning(f"⚠️ Stage 3 checkpoint not found: {prev_ckpt}")

    hm_size = tuple(stage_cfg['hm_size'])
    if hasattr(model, 'update_heatmap_size'):
        model.update_heatmap_size(hm_size)
    train_loader.dataset.hm_size = hm_size
    val_loader.dataset.hm_size = hm_size

    freeze_unfreeze_modules(model, stage_cfg)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ⭐ FIX: Use math.ceil to align scheduler with actual optimizer steps
    total_steps = math.ceil(len(train_loader) / cfg['optim']['grad_accum_steps']) * stage_cfg['epochs']
    optimizer = build_optimizer(model, stage_cfg, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ⭐ FIX: Conditional scaler based on AMP mode
    if cfg['optim']['amp'] == 'fp16':
        scaler = GradScaler()
    else:
        scaler = None

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
        tb_writer = SummaryWriter(log_dir=str(tb_dir / f'stage4_{timestamp}'))
        logger.info(f"📊 TensorBoard enabled: {tb_dir / f'stage4_{timestamp}'}")

    best_val_loss = float('inf')
    stage_idx = 3  # ⭐ Stage 4 对应 idx=3

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
            lr_list = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            logger.info("=" * 80)
            logger.info(
                f"[Stage {stage_idx+1}/{len(cfg['training']['stages'])}: {stage_name}] "
                f"Epoch {epoch}/{stage_cfg['epochs']} Started"
            )
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

        # ⭐ FIX Bug 1: All ranks execute validation
        val_metrics = validate(model, val_loader, criterion, stage_cfg, cfg, device, world_size, rank, tb_writer=tb_writer, epoch=epoch)

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
                save_checkpoint(model, optimizer, scheduler, epoch, stage_name, {'train': train_metrics, 'val': val_metrics}, cfg, is_best, scaler)
            logger.info("=" * 80)

    if world_size > 1:
        torch.distributed.destroy_process_group()

    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
            logger.info("📊 TensorBoard writer closed")
        logger.info(f"\n✅ Stage 4 completed! Best val loss: {best_val_loss:.4f}")
        logger.info(f"下一步: python train_stage5_joint_training.py --config {args.config}\n")


if __name__ == '__main__':
    main()
