"""
训练工具函数 - 适配 SpatialMLLMPipeline 的双头（history/future）流程
沿用旧五阶段脚本的调用风格：build_dataloaders/build_model/freeze_unfreeze_modules/
build_optimizer/build_scheduler/train_one_epoch/validate。
"""

import os
import sys
import math
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.utils.loss import NavigationHeatmapLoss

logger = logging.getLogger(__name__)


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    return rank, world_size, local_rank


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = VLNHeatmapDataset(
        root=cfg['data']['root'],
        split='train',
        frames_per_clip=cfg['data']['frames_per_clip'],
        heatmap_per_clip=cfg['data']['heatmap_per_clip'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        num_sample_frames=cfg['data'].get('num_sample_frames')
    )

    val_dataset = VLNHeatmapDataset(
        root=cfg['data']['root'],
        split='val',
        frames_per_clip=cfg['data']['frames_per_clip'],
        heatmap_per_clip=cfg['data']['heatmap_per_clip'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        num_sample_frames=cfg['data'].get('num_sample_frames')
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['optim']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        prefetch_factor=cfg['data'].get('prefetch_factor', 2),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['optim']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        prefetch_factor=cfg['data'].get('prefetch_factor', 2),
        drop_last=False
    )

    return train_loader, val_loader


def build_model(cfg: Dict, local_rank: int = 0) -> nn.Module:
    """构建SpatialMLLMPipeline模型（兼容DDP/多卡拆分）"""
    model_cfg = cfg['model']
    use_multi_gpu = model_cfg.get('use_multi_gpu', False)

    # Critical fix for DDP: set device to rank-specific GPU
    if use_multi_gpu:
        # Multi-GPU mode: use explicit device assignments from config
        vggt_gpu = model_cfg['vggt_gpu']
        dinov3_gpu = model_cfg['dinov3_gpu']
        llm_gpu = model_cfg['llm_gpu']
        device_str = "cuda"  # Generic device for multi-GPU
    else:
        # DDP mode: all modules on same device (rank's GPU)
        device_str = f"cuda:{local_rank}"
        vggt_gpu = dinov3_gpu = llm_gpu = device_str

    integration_cfg = SpatialMLLMIntegrationConfig(
        target_keyframes=cfg['data']['heatmap_per_clip'],
        total_frames=cfg['data']['frames_per_clip'],
        sampling_method="hybrid",
        llm_model_path=model_cfg['llm']['model_path'],
        device=device_str,  # ✅ Pass rank-specific device to config
        vggt_gpu=vggt_gpu,  # ✅ In DDP mode, this equals device_str (cuda:local_rank)
        dinov3_gpu=dinov3_gpu,  # ✅ In DDP mode, this equals device_str
        llm_gpu=llm_gpu,  # ✅ In DDP mode, this equals device_str
        use_multi_gpu=use_multi_gpu,  # ✅ Should be False for DDP
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_inter_frame_heatmaps=True,
        dinov3_img_size=cfg['data']['image_size'][0],
        vggt_img_size=cfg['data']['image_size'][0],
        enable_gradient_checkpointing=cfg['optim'].get('gradient_checkpointing', False),
        verbose=True
    )

    # ✅ Model is initialized with correct device in __init__
    # All sub-modules should be on device_str (cuda:local_rank in DDP mode)
    model = SpatialMLLMPipeline(integration_cfg)

    # ⚠️ CRITICAL: Do NOT call model.to() after init - modules are already on correct devices
    # Calling .to() again might break LLM checkpoint loading or move modules incorrectly
    # model = model.to(device_str)  # REMOVED - causes issues with already-placed modules

    return model


def freeze_unfreeze_modules(model: nn.Module, stage_cfg: Dict):
    """冻结/解冻模块（兼容DDP/FSDP）"""
    # Unwrap FSDP or DDP wrapper to access actual model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        model_unwrapped = model  # FSDP handles parameter access differently
    elif isinstance(model, DDP):
        model_unwrapped = model.module
    else:
        model_unwrapped = model

    for param in model_unwrapped.parameters():
        param.requires_grad = False

    trainable = stage_cfg.get('trainable_modules', [])

    module_map = {
        'history_heatmap_converter': 'history_heatmap_converter',
        'future_heatmap_converter': 'future_heatmap_converter',
        'feature_fusion': 'feature_fusion',
        'llm_projector': 'llm_projector',
        'vggt': 'vggt',
        'dinov3_compat': 'dinov3_compat',
        # validity heads跟随各自converter
        'history_validity_head': 'history_validity_head',
        'future_validity_head': 'future_validity_head',
    }

    logger.info("Trainable modules:")
    for name in trainable:
        attr = module_map.get(name)
        if attr and hasattr(model_unwrapped, attr):
            module = getattr(model_unwrapped, attr)
            for p in module.parameters():
                p.requires_grad = True
            logger.info(f"  ✓ {name}")
        # 如果选择converter，也解冻对应的validity head
        if name == 'history_heatmap_converter' and hasattr(model_unwrapped, 'history_validity_head'):
            for p in model_unwrapped.history_validity_head.parameters():
                p.requires_grad = True
        if name == 'future_heatmap_converter' and hasattr(model_unwrapped, 'future_validity_head'):
            for p in model_unwrapped.future_validity_head.parameters():
                p.requires_grad = True


def build_optimizer(model: nn.Module, stage_cfg: Dict, cfg: Dict) -> torch.optim.Optimizer:
    """构建分组学习率优化器"""
    model_unwrapped = model.module if isinstance(model, DDP) else model
    optim_cfg = cfg['optim']
    lr_override = stage_cfg.get('learning_rates', {})

    def lr_for(key, default):
        return lr_override.get(key, optim_cfg.get(key, default))

    param_groups = []

    # 历史/未来 converter + validity
    if hasattr(model_unwrapped, 'history_heatmap_converter'):
        params = [p for p in model_unwrapped.history_heatmap_converter.parameters() if p.requires_grad]
        if hasattr(model_unwrapped, 'history_validity_head'):
            params += [p for p in model_unwrapped.history_validity_head.parameters() if p.requires_grad]
        if params:
            lr = lr_for('history_heatmap_lr', optim_cfg['heatmap_lr'])
            param_groups.append({'params': params, 'lr': lr, 'name': 'history_heatmap_converter'})
            logger.info(f"  history_heatmap_converter lr={lr:.2e}, params={sum(p.numel() for p in params):,}")

    if hasattr(model_unwrapped, 'future_heatmap_converter'):
        params = [p for p in model_unwrapped.future_heatmap_converter.parameters() if p.requires_grad]
        if hasattr(model_unwrapped, 'future_validity_head'):
            params += [p for p in model_unwrapped.future_validity_head.parameters() if p.requires_grad]
        if params:
            lr = lr_for('future_heatmap_lr', optim_cfg['heatmap_lr'])
            param_groups.append({'params': params, 'lr': lr, 'name': 'future_heatmap_converter'})
            logger.info(f"  future_heatmap_converter lr={lr:.2e}, params={sum(p.numel() for p in params):,}")

    # 融合 + 投影
    fusion_params = []
    if hasattr(model_unwrapped, 'feature_fusion'):
        fusion_params += [p for p in model_unwrapped.feature_fusion.parameters() if p.requires_grad]
    if hasattr(model_unwrapped, 'llm_projector'):
        fusion_params += [p for p in model_unwrapped.llm_projector.parameters() if p.requires_grad]
    if fusion_params:
        lr = lr_for('fusion_lr', optim_cfg['fusion_lr'])
        param_groups.append({'params': fusion_params, 'lr': lr, 'name': 'fusion_projector'})
        logger.info(f"  fusion_projector lr={lr:.2e}, params={sum(p.numel() for p in fusion_params):,}")

    # 编码器
    encoder_params = []
    if hasattr(model_unwrapped, 'vggt'):
        encoder_params += [p for p in model_unwrapped.vggt.parameters() if p.requires_grad]
    if hasattr(model_unwrapped, 'dinov3_compat'):
        encoder_params += [p for p in model_unwrapped.dinov3_compat.parameters() if p.requires_grad]
    if encoder_params:
        lr = lr_for('encoder_lr', optim_cfg['encoder_lr'])
        param_groups.append({'params': encoder_params, 'lr': lr, 'name': 'encoders'})
        logger.info(f"  encoders lr={lr:.2e}, params={sum(p.numel() for p in encoder_params):,}")

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=optim_cfg['weight_decay'])
    return optimizer


def build_scheduler(optimizer, cfg: Dict, total_steps: int):
    warmup_steps = int(total_steps * cfg['optim']['warmup_ratio'])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(progress * math.pi))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_dual_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                     criterion: NavigationHeatmapLoss, stage_cfg: Dict, cfg: Dict,
                     device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', True)

    gt_hist = batch['gt_heatmap_history'].to(device)
    gt_fut = batch['gt_heatmap_future'].to(device)
    val_hist = batch['gt_validity_history'].to(device)
    val_fut = batch['gt_validity_future'].to(device)

    hist_pred = outputs['history_heatmaps']
    fut_pred = outputs['future_heatmaps']
    hist_val_pred = outputs['history_validity']
    fut_val_pred = outputs['future_validity']

    total_loss = 0.0
    metrics = {}

    if train_history:
        B, K, Hm, Wm = hist_pred.shape
        pred_logits = hist_pred.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
        gt_map = gt_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
        gt_val = val_hist.reshape(B * K, 1)
        pred_val = hist_val_pred.reshape(B * K, 1)
        loss_hist, comps = criterion(pred_logits, gt_map, pred_val, gt_val)
        total_loss += cfg['loss'].get('history_weight', 1.0) * loss_hist
        metrics.update({f'hist_{k}': v for k, v in comps.items()})
        metrics['history_loss'] = loss_hist.item()

    if train_future:
        B, K, Hm, Wm = fut_pred.shape
        pred_logits = fut_pred.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
        gt_map = gt_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
        gt_val = val_fut.reshape(B * K, 1)
        pred_val = fut_val_pred.reshape(B * K, 1)
        loss_fut, comps = criterion(pred_logits, gt_map, pred_val, gt_val)
        total_loss += cfg['loss'].get('future_weight', 1.0) * loss_fut
        metrics.update({f'fut_{k}': v for k, v in comps.items()})
        metrics['future_loss'] = loss_fut.item()

    metrics['total_loss'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
    return total_loss, metrics


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer, scheduler,
                    scaler: GradScaler, stage_cfg: Dict, cfg: Dict,
                    epoch: int, rank: int, device: torch.device,
                    criterion: NavigationHeatmapLoss,
                    stage_idx: int = 0,
                    stage_name: str = "",
                    tb_writer = None) -> Dict[str, float]:
    """
    训练一个epoch，支持 tqdm 进度条和详细日志输出

    Args:
        model: 训练模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: AMP 梯度缩放器
        stage_cfg: 阶段配置
        cfg: 全局配置
        epoch: 当前 epoch 编号
        rank: DDP rank
        device: 训练设备
        criterion: 损失函数
        stage_idx: 阶段索引 (0-4)
        stage_name: 阶段名称

    Returns:
        metrics: 平均训练指标字典
    """
    model.train()
    total_metrics = {}
    grad_accum_steps = cfg['optim']['grad_accum_steps']
    log_interval = cfg['log'].get('log_interval', 10)
    show_gpu_memory = cfg['log'].get('show_gpu_memory', True)
    show_lr_all_groups = cfg['log'].get('show_lr_all_groups', False)
    tqdm_ncols = cfg['log'].get('tqdm_ncols', 120)

    # ⭐ 确定 AMP dtype
    amp_dtype = None
    if cfg['optim']['amp'] == 'bf16':
        amp_dtype = torch.bfloat16
    elif cfg['optim']['amp'] == 'fp16':
        amp_dtype = torch.float16

    # ⭐ 统一日志函数（在 train_one_epoch 内部定义）
    def log_training_progress(batch_idx: int, total_batches: int, global_step: int,
                             loss: float, lrs: list, gpu_mem_gb: float):
        """统一的训练进度日志输出"""
        progress_pct = (batch_idx + 1) / total_batches * 100

        # 构建学习率字符串
        if show_lr_all_groups:
            lr_str = " | ".join([f"LR{i}: {lr:.2e}" for i, lr in enumerate(lrs)])
        else:
            lr_str = f"LR: {lrs[0]:.2e}"

        # 构建完整日志
        log_parts = [
            f"[Stage {stage_idx+1}: {stage_name}]",
            f"Epoch {epoch}",
            f"Batch {batch_idx+1}/{total_batches} ({progress_pct:.1f}%)",
            f"Step {global_step}",
            f"Loss: {loss:.4f}",
            lr_str
        ]

        if show_gpu_memory:
            log_parts.append(f"GPU: {gpu_mem_gb:.1f}GB")

        logger.info(" | ".join(log_parts))

    # ⭐ 仅在 rank 0 显示 tqdm 进度条
    if rank == 0:
        total_epochs = stage_cfg.get('epochs', 1)
        pbar = tqdm(
            train_loader,
            desc=f"[Stage {stage_idx+1}] Epoch {epoch}/{total_epochs}",
            ncols=tqdm_ncols
        )
        iterator = pbar
    else:
        iterator = train_loader

    global_step = 0

    for batch_idx, batch in enumerate(iterator):
        frames = batch['frames'].to(device)

        # 前向传播（条件 AMP）
        if amp_dtype is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(frames, instruction_text=None, return_heatmaps=True)
                loss, metrics = compute_dual_loss(outputs, batch, criterion, stage_cfg, cfg, device)
                loss = loss / grad_accum_steps
        else:
            outputs = model(frames, instruction_text=None, return_heatmaps=True)
            loss, metrics = compute_dual_loss(outputs, batch, criterion, stage_cfg, cfg, device)
            loss = loss / grad_accum_steps

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积 + 优化器步进
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            # TensorBoard logging (only rank 0)
            if rank == 0 and tb_writer is not None:
                tb_writer.add_scalar('train/loss', loss.item() * grad_accum_steps, global_step)
                tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # 累积指标
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

        # ⭐ 更新 tqdm postfix（仅 rank 0）
        if rank == 0:
            avg_loss = total_metrics['total_loss'] / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}',
                'step': global_step
            })

            # ⭐ 定期输出详细日志（根据 log_interval）
            if (batch_idx + 1) % log_interval == 0:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                log_training_progress(
                    batch_idx, len(train_loader), global_step,
                    avg_loss, lrs, gpu_mem_gb
                )

    # ⭐ 尾部梯度处理（改为 DEBUG 级别）
    if (batch_idx + 1) % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['grad_clip'])
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if rank == 0:
            logger.debug(f"Epoch {epoch}: Applied tail gradients from {(batch_idx + 1) % grad_accum_steps} remaining batch(es)")

    # 关闭 tqdm
    if rank == 0:
        pbar.close()

    # 计算平均指标
    num_batches = len(train_loader)
    for k in total_metrics:
        total_metrics[k] /= num_batches

    return total_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: NavigationHeatmapLoss,
    stage_cfg: Dict,
    cfg: Dict,
    device: torch.device,
    world_size: int = 1,
    rank: int = 0,
    tb_writer = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Validate model on validation set.

    ⭐ All ranks execute validation and aggregate metrics via all-reduce.
    """
    model.eval()
    total_metrics = {}

    for batch in val_loader:
        frames = batch['frames'].to(device)
        outputs = model(frames, instruction_text=None, return_heatmaps=True)
        loss, metrics = compute_dual_loss(outputs, batch, criterion, stage_cfg, cfg, device)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

    # Calculate averages
    num_batches = len(val_loader)
    for k in total_metrics:
        total_metrics[k] /= num_batches

    # ⭐ FIX: Aggregate metrics across all ranks
    if world_size > 1:
        for key in total_metrics:
            tensor = torch.tensor(total_metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_metrics[key] = (tensor / world_size).item()

    # TensorBoard logging (only rank 0)
    if rank == 0 and tb_writer is not None:
        tb_writer.add_scalar('val/total_loss', total_metrics.get('total_loss', 0.0), epoch)
        for k, v in total_metrics.items():
            if k != 'total_loss':
                tb_writer.add_scalar(f'val/{k}', v, epoch)

    return total_metrics


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, stage_name: str,
                    metrics: Dict[str, float], cfg: Dict, is_best: bool = False, scaler: GradScaler = None):
    out_dir = Path(cfg['log']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model

    # 🔥 只保存可训练的参数（requires_grad=True），避免保存冻结的大模型权重
    full_state_dict = model_to_save.state_dict()
    trainable_state_dict = {
        k: v for k, v in full_state_dict.items()
        if any(p.requires_grad and p.data_ptr() == v.data_ptr()
               for p in model_to_save.parameters())
    }

    # 记录保存的参数数量和大小
    total_params = sum(p.numel() for p in full_state_dict.values())
    trainable_params = sum(p.numel() for p in trainable_state_dict.values())
    logger.info(f"Saving checkpoint: {trainable_params:,} / {total_params:,} trainable params "
                f"({100*trainable_params/max(total_params,1):.1f}%)")

    checkpoint = {
        'epoch': epoch,
        'stage': stage_name,
        'model_state_dict': trainable_state_dict,  # 🔥 只保存可训练参数
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': cfg
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    ckpt_path = out_dir / f"{stage_name}_epoch_{epoch}.pth"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    if is_best:
        best_path = out_dir / f"{stage_name}_best.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved: {best_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """加载checkpoint（兼容DDP，支持部分参数加载）"""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = model.module if isinstance(model, DDP) else model

    # 处理DDP prefix
    state_dict = checkpoint['model_state_dict']
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 🔥 使用 strict=False 加载，允许只加载部分参数（可训练参数）
    missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)

    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    logger.info(f"  Previous epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Previous stage: {checkpoint.get('stage', 'N/A')}")
    if missing_keys:
        logger.info(f"  Missing keys (frozen params, expected): {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"  Unexpected keys: {unexpected_keys[:5]}...")  # 显示前5个

    return checkpoint


def load_training_state(model: nn.Module, optimizer, scheduler, checkpoint_path: str,
                        device: torch.device, scaler: GradScaler = None):
    """加载模型+优化器+调度器（可选scaler），返回起始epoch"""
    ckpt = load_checkpoint(model, checkpoint_path, device)
    if ckpt is None:
        return None

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    start_epoch = ckpt.get('epoch', 0) + 1
    logger.info(f"Resumed training from epoch {start_epoch} (checkpoint: {checkpoint_path})")
    return start_epoch, ckpt
