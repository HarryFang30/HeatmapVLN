#!/usr/bin/env python3
"""
完整模型训练脚本 - SpatialMLLMPipeline (VGGT + DINOv3 + Qwen2.5-VL)
三阶段训练，全程冻结LLM
"""

import sys
import os
from pathlib import Path

# =============================================================================
# 🔥 GPU配置模块 - 由配置文件控制（单卡 / 多卡拆分）
# =============================================================================
# 🔥 启用expandable_segments以减少显存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print(f"🔥 GPU配置: 由 training_config_full_model.yaml 控制")
print("="*80)

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional
import warnings
import gc
warnings.filterwarnings("ignore")

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.utils.losses import heatmap_ce_from_logits
from src.utils.loss import NavigationHeatmapLoss
from src.utils.logger import setup_logger


def variable_length_collate_fn(batch):
    """
    Custom collate function to handle variable-length frame sequences and heatmaps.
    Pads all sequences to the maximum length in the batch.
    """
    # Find max frame count and max heatmap count in this batch
    max_T = max(sample['frames'].shape[0] for sample in batch)
    max_K = max(sample['gt_heatmap_history'].shape[0] for sample in batch)

    # Prepare batched tensors
    batch_frames = []
    batch_text = []
    batch_gt_hist = []
    batch_gt_fut = []
    batch_val_hist = []
    batch_val_fut = []
    batch_meta = []
    batch_frame_mask = []  # mask indicating valid frames

    for sample in batch:
        frames = sample['frames']  # [T, 3, H, W]
        T = frames.shape[0]

        # Pad frames to max_T if needed
        if T < max_T:
            pad_size = max_T - T
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)  # Repeat last frame
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            frame_mask = torch.cat([torch.ones(T), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            frame_mask = torch.ones(T)

        # Pad heatmaps to max_K if needed
        gt_hist = sample['gt_heatmap_history']  # [K, Hm, Wm]
        gt_fut = sample['gt_heatmap_future']    # [K, Hm, Wm]
        val_hist = sample['gt_validity_history']  # [K]
        val_fut = sample['gt_validity_future']    # [K]
        K = gt_hist.shape[0]

        if K < max_K:
            pad_size = max_K - K
            Hm, Wm = gt_hist.shape[1], gt_hist.shape[2]
            pad_heatmaps = torch.zeros(pad_size, Hm, Wm)
            pad_mask = torch.zeros(pad_size)
            gt_hist_padded = torch.cat([gt_hist, pad_heatmaps], dim=0)
            gt_fut_padded = torch.cat([gt_fut, pad_heatmaps.clone()], dim=0)
            val_hist_padded = torch.cat([val_hist, pad_mask], dim=0)
            val_fut_padded = torch.cat([val_fut, pad_mask.clone()], dim=0)
        else:
            gt_hist_padded = gt_hist
            gt_fut_padded = gt_fut
            val_hist_padded = val_hist
            val_fut_padded = val_fut

        batch_frames.append(frames_padded)
        batch_text.append(sample['text'])
        batch_gt_hist.append(gt_hist_padded)
        batch_gt_fut.append(gt_fut_padded)
        batch_val_hist.append(val_hist_padded)
        batch_val_fut.append(val_fut_padded)
        batch_meta.append(sample['meta'])
        batch_frame_mask.append(frame_mask)

    return {
        'frames': torch.stack(batch_frames, dim=0),  # [B, max_T, 3, H, W]
        'text': batch_text,                          # List of strings
        'gt_heatmap_history': torch.stack(batch_gt_hist, dim=0),   # [B, max_K, Hm, Wm]
        'gt_heatmap_future': torch.stack(batch_gt_fut, dim=0),     # [B, max_K, Hm, Wm]
        'gt_validity_history': torch.stack(batch_val_hist, dim=0), # [B, max_K]
        'gt_validity_future': torch.stack(batch_val_fut, dim=0),   # [B, max_K]
        'meta': batch_meta,                          # List of dicts
        'frame_mask': torch.stack(batch_frame_mask, dim=0)  # [B, max_T]
    }


def load_config(config_path: str) -> Dict:
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def build_model(cfg: Dict) -> SpatialMLLMPipeline:
    """构建完整模型"""
    model_cfg = cfg['model']
    use_multi_gpu = model_cfg.get('use_multi_gpu', False)

    # 构建SpatialMLLMIntegrationConfig
    config = SpatialMLLMIntegrationConfig(
        # Frame sampling
        target_keyframes=cfg['data']['heatmap_per_clip'],
        total_frames=cfg['data']['frames_per_clip'],
        sampling_method="hybrid",

        # Model paths and GPU allocation
        llm_model_path=model_cfg['llm']['model_path'],
        vggt_gpu=model_cfg['vggt_gpu'],
        dinov3_gpu=model_cfg['dinov3_gpu'],
        llm_gpu=model_cfg['llm_gpu'],
        use_multi_gpu=use_multi_gpu,

        # LLM settings
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,  # 🔥 FIX: 训练时保持模型加载，避免重复加载导致OOM

        # Heatmap settings (使用data配置中的初始尺寸)
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_inter_frame_heatmaps=True,

        # Image sizes
        dinov3_img_size=cfg['data']['image_size'][0],
        vggt_img_size=cfg['data']['image_size'][0],

        # Memory optimization
        enable_gradient_checkpointing=cfg['optim'].get('gradient_checkpointing', False),

        # Debug
        verbose=True
    )

    model = SpatialMLLMPipeline(config)
    print(f"✅ 完整模型已构建：SpatialMLLMPipeline")
    print(f"   use_multi_gpu = {use_multi_gpu}")
    print(f"   VGGT → {model_cfg['vggt_gpu']}")
    print(f"   DINOv3 → {model_cfg['dinov3_gpu']}")
    print(f"   Qwen2.5-VL → {model_cfg['llm_gpu']} (frozen)")

    return model


def freeze_module(module: nn.Module, freeze: bool = True):
    """冻结/解冻模块"""
    for param in module.parameters():
        param.requires_grad = not freeze


def set_trainable_modules(model: SpatialMLLMPipeline, stage_cfg: Dict):
    """根据阶段配置设置可训练模块"""
    # 1) 先全部冻结
    freeze_module(model, freeze=True)

    # 2) 根据配置解冻特定模块
    trainable = stage_cfg.get('trainable_modules', [])

    # Heatmap converters
    if 'history_heatmap_converter' in trainable or 'heatmap_converter' in trainable:
        if hasattr(model, 'history_heatmap_converter'):
            freeze_module(model.history_heatmap_converter, freeze=False)
            print("  ✓ Unfrozen: history_heatmap_converter")
    if 'future_heatmap_converter' in trainable or 'heatmap_converter' in trainable:
        if hasattr(model, 'future_heatmap_converter'):
            freeze_module(model.future_heatmap_converter, freeze=False)
            print("  ✓ Unfrozen: future_heatmap_converter")

    if 'feature_fusion' in trainable:
        freeze_module(model.feature_fusion, freeze=False)
        print("  ✓ Unfrozen: feature_fusion")

    if 'llm_projector' in trainable:
        freeze_module(model.llm_projector, freeze=False)
        print("  ✓ Unfrozen: llm_projector")

    if 'vggt' in trainable:
        freeze_module(model.vggt, freeze=False)
        print("  ✓ Unfrozen: vggt")

    if 'dinov3_compat' in trainable:
        freeze_module(model.dinov3_compat, freeze=False)
        print("  ✓ Unfrozen: dinov3_compat")

    # 3) LLM始终冻结
    if hasattr(model, 'llm_integration') and model.llm_integration is not None:
        freeze_module(model.llm_integration, freeze=True)


def build_optimizer(model: SpatialMLLMPipeline, cfg: Dict, stage_cfg: Dict) -> torch.optim.Optimizer:
    """构建分层学习率优化器"""
    optim_cfg = cfg['optim']

    # 参数分组
    param_groups = []

    # 1) 热力图转换器（高学习率，可分头设置）
    hist_lr = optim_cfg.get('history_heatmap_lr', optim_cfg['heatmap_lr'])
    fut_lr = optim_cfg.get('future_heatmap_lr', optim_cfg['heatmap_lr'])

    if hasattr(model, 'history_heatmap_converter'):
        hist_params = [p for p in model.history_heatmap_converter.parameters() if p.requires_grad]
        if hist_params:
            param_groups.append({
                'params': hist_params,
                'lr': hist_lr,
                'name': 'history_heatmap_converter'
            })
            print(f"  Param group: history_heatmap_converter (lr={hist_lr})")

    if hasattr(model, 'future_heatmap_converter'):
        fut_params = [p for p in model.future_heatmap_converter.parameters() if p.requires_grad]
        if fut_params:
            param_groups.append({
                'params': fut_params,
                'lr': fut_lr,
                'name': 'future_heatmap_converter'
            })
            print(f"  Param group: future_heatmap_converter (lr={fut_lr})")

    # 2) 融合模块+投影器（中等学习率）
    fusion_params = []
    if hasattr(model, 'feature_fusion'):
        fusion_params.extend([p for p in model.feature_fusion.parameters() if p.requires_grad])
    if hasattr(model, 'llm_projector'):
        fusion_params.extend([p for p in model.llm_projector.parameters() if p.requires_grad])
    if fusion_params:
        param_groups.append({
            'params': fusion_params,
            'lr': optim_cfg['fusion_lr'],
            'name': 'fusion'
        })
        print(f"  Param group: fusion (lr={optim_cfg['fusion_lr']})")

    # 3) 编码器（低学习率）
    encoder_params = []
    if hasattr(model, 'vggt'):
        encoder_params.extend([p for p in model.vggt.parameters() if p.requires_grad])
    if hasattr(model, 'dinov3_compat'):
        encoder_params.extend([p for p in model.dinov3_compat.parameters() if p.requires_grad])
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': optim_cfg['encoder_lr'],
            'name': 'encoders'
        })
        print(f"  Param group: encoders (lr={optim_cfg['encoder_lr']})")

    if not param_groups:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=optim_cfg['weight_decay']
    )

    return optimizer


def build_scheduler(optimizer, cfg: Dict, total_steps: int):
    """构建学习率调度器"""
    optim_cfg = cfg['optim']
    warmup_steps = int(total_steps * optim_cfg['warmup_ratio'])

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )

    return scheduler


def train_one_epoch(
    model: SpatialMLLMPipeline,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: Dict,
    criterion: NavigationHeatmapLoss,
    epoch: int,
    logger,
    tb_writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    stage_idx: int = 0,
    stage_name: str = "",
    stage_cfg: Dict = None
) -> float:
    """训练一个epoch"""

    # ⭐ 统一的训练进度日志输出函数
    def log_training_progress(
        batch_idx: int,
        total_batches: int,
        global_step: int,
        loss: float,
        lrs: list,
        gpu_mem_gb: float,
        actual_step: int = 0
    ):
        """统一的训练进度日志输出"""
        progress_pct = (batch_idx + 1) / total_batches * 100

        # 控制台日志
        logger.info(
            f"[Stage {stage_idx+1}: {stage_name}] "
            f"Epoch {epoch}/{stage_cfg['epochs']} | "
            f"Batch {batch_idx+1}/{total_batches} ({progress_pct:.1f}%) | "
            f"Step {global_step} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lrs[0]:.2e} | "
            f"GPU: {gpu_mem_gb:.1f}GB"
        )

        # TensorBoard 日志（如果启用）
        if tb_writer is not None and actual_step > 0:
            tb_writer.add_scalar('train/loss_step', loss, actual_step)
            for idx, lr in enumerate(lrs):
                tb_writer.add_scalar(f'train/lr_group_{idx}', lr, actual_step)
            tb_writer.add_scalar('system/gpu_memory_gb', gpu_mem_gb, actual_step)
            tb_writer.flush()

    model.train()
    total_loss = 0.0
    num_batches = 0

    optim_cfg = cfg['optim']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)

    # 🔥 调试：记录optimizer step计数
    global_step = 0  # 当前epoch内的optimizer step数
    valid_batch_count = 0  # 🔥 有效batch计数（只在实际backward后递增）

    # ⭐ 优化 tqdm 进度条：显示更多上下文信息
    total_epochs = stage_cfg['epochs']
    tqdm_ncols = cfg['log'].get('tqdm_ncols', 120)
    pbar = tqdm(
        train_loader,
        desc=f"[Stage {stage_idx+1}] Epoch {epoch}/{total_epochs}",
        ncols=tqdm_ncols
    )

    for i, batch in enumerate(pbar):
        frames = batch['frames']  # [B, T, 3, H, W]
        text = batch.get('text', None)  # List[str] or None
        gt_hist = batch['gt_heatmap_history']  # [B, K, Hm, Wm]
        gt_fut = batch['gt_heatmap_future']    # [B, K, Hm, Wm]
        mask_hist = batch['gt_validity_history']  # [B, K]
        mask_fut = batch['gt_validity_future']    # [B, K]
        train_history = stage_cfg.get('train_history', True)
        train_future = stage_cfg.get('train_future', True)

        # Multi-GPU数据移动
        # frames会在模型内部自动分配到各个GPU
        gt_hist = gt_hist.to(cfg['model']['llm_gpu'])  # 与输出同GPU
        gt_fut = gt_fut.to(cfg['model']['llm_gpu'])
        mask_hist = mask_hist.to(cfg['model']['llm_gpu'])
        mask_fut = mask_fut.to(cfg['model']['llm_gpu'])

        # 前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # SpatialMLLMPipeline返回字典
            # 注意：模型的forward只接受单个文本，使用批次第一个
            output = model(
                video_frames=frames,
                instruction_text=text[0] if (text and len(text) > 0) else None
            )

            # 提取双头热力图（已按keyframe顺序堆叠）
            pred_hist = output['history_heatmaps']  # [B, K, H, W]
            pred_fut = output['future_heatmaps']    # [B, K, H, W]

            # 确保在同一设备
            if pred_hist.device != gt_hist.device:
                pred_hist = pred_hist.to(gt_hist.device)
                pred_fut = pred_fut.to(gt_fut.device)

            # 🔥 检查mask是否全为0（跳过无效batch）
            if (mask_hist is not None and mask_hist.sum() < 0.5) and (mask_fut is not None and mask_fut.sum() < 0.5):
                logger.debug(f"  ⚠️ Skipping batch {i+1}: all masks are zero (no valid GT)")
                continue  # 跳过这个batch

            # 计算双头损失（NavigationHeatmapLoss）
            hist_loss = torch.tensor(0.0, device=gt_hist.device)
            fut_loss = torch.tensor(0.0, device=gt_hist.device)

            if train_history and (mask_hist is None or mask_hist.sum() > 0.5):
                # 展平 [B,K,H,W] -> [B*K,1,H,W]
                B, K, Hm, Wm = pred_hist.shape
                pred_hist_logits = pred_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_hist_map = gt_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_hist_val = mask_hist.reshape(B * K, 1)
                # 简单有效性预测：来自模型输出 validity [B, K]
                pred_hist_val = output['history_validity'].reshape(B * K, 1)

                hist_loss, _ = criterion(
                    pred_logits=pred_hist_logits,
                    gt_heatmap_raw=gt_hist_map,
                    pred_validity=pred_hist_val,
                    gt_validity=gt_hist_val
                )

            if train_future and (mask_fut is None or mask_fut.sum() > 0.5):
                B, K, Hm, Wm = pred_fut.shape
                pred_fut_logits = pred_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_fut_map = gt_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_fut_val = mask_fut.reshape(B * K, 1)
                pred_fut_val = output['future_validity'].reshape(B * K, 1)

                fut_loss, _ = criterion(
                    pred_logits=pred_fut_logits,
                    gt_heatmap_raw=gt_fut_map,
                    pred_validity=pred_fut_val,
                    gt_validity=gt_fut_val
                )

            loss = (hist_loss + fut_loss) / grad_accum_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 🔥 FIX: 只在实际backward后递增有效batch计数
        valid_batch_count += 1

        # 🔥 FIX: 显式删除大tensor释放显存
        del output, pred_hist, pred_fut
        if 'frames' in locals():
            del frames, gt_hist, gt_fut, mask_hist, mask_fut

        # 🔥 梯度累积：使用valid_batch_count而非batch索引i
        if valid_batch_count % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                optim_cfg['grad_clip']
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # ⭐ 统一日志输出（替代原来的分散日志）
            global_step += 1
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]

            # 使用统一日志函数
            log_interval = cfg['log'].get('log_interval', 10)
            if global_step % log_interval == 0 or global_step <= 5:
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                log_training_progress(
                    batch_idx=i,
                    total_batches=len(train_loader),
                    global_step=global_step,
                    loss=loss.item() * grad_accum_steps,
                    lrs=current_lrs,
                    gpu_mem_gb=mem_alloc,
                    actual_step=global_step_offset + global_step
                )

        # 🔥 FIX: 每个iteration后清理显存，防止累积（尤其是大features）
        if (i + 1) % 2 == 0:  # 每2个iteration清理一次（平衡速度和显存）
            gc.collect()
            torch.cuda.empty_cache()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        # ⭐ 移除单独的 GPU 内存日志（已整合到统一日志函数中）

        # ⭐ 优化 tqdm postfix：添加 GPU 内存和 step 信息
        mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
        pbar.set_postfix({
            'loss': f"{loss.item() * grad_accum_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            'gpu': f"{mem_alloc:.1f}GB",
            'step': global_step
        })

    # 🔥 处理epoch结束时剩余的梯度（如果有）
    remaining_batches = valid_batch_count % grad_accum_steps
    if remaining_batches > 0:
        logger.debug(f"  ⚠️ Processing remaining {remaining_batches} batches at epoch end")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1
        logger.debug(f"  [Final Optimizer Step {global_step}] processed {remaining_batches}/{grad_accum_steps} batches")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: SpatialMLLMPipeline,
    val_loader: DataLoader,
    cfg: Dict,
    logger,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    stage_cfg: Dict = None,
    criterion: NavigationHeatmapLoss = None
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        frames = batch['frames']
        text = batch.get('text', None)
        gt_hist = batch['gt_heatmap_history'].to(cfg['model']['llm_gpu'])
        gt_fut = batch['gt_heatmap_future'].to(cfg['model']['llm_gpu'])
        mask_hist = batch['gt_validity_history'].to(cfg['model']['llm_gpu'])
        mask_fut = batch['gt_validity_future'].to(cfg['model']['llm_gpu'])
        val_history = True if stage_cfg is None else stage_cfg.get('train_history', True)
        val_future = True if stage_cfg is None else stage_cfg.get('train_future', True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                video_frames=frames,
                instruction_text=text[0] if (text and len(text) > 0) else None
            )

            pred_hist = output['history_heatmaps']
            pred_fut = output['future_heatmaps']

            # 确保在同一设备
            if pred_hist.device != gt_hist.device:
                pred_hist = pred_hist.to(gt_hist.device)
                pred_fut = pred_fut.to(gt_fut.device)

            # 🔥 跳过mask全为0的batch
            if (mask_hist is not None and mask_hist.sum() < 0.5) and (mask_fut is not None and mask_fut.sum() < 0.5):
                continue

            hist_loss = torch.tensor(0.0, device=gt_hist.device)
            fut_loss = torch.tensor(0.0, device=gt_hist.device)

            if val_history and (mask_hist is None or mask_hist.sum() > 0.5):
                B, K, Hm, Wm = pred_hist.shape
                pred_hist_logits = pred_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_hist_map = gt_hist.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_hist_val = mask_hist.reshape(B * K, 1)
                pred_hist_val = output['history_validity'].reshape(B * K, 1)

                hist_loss, _ = criterion(
                    pred_logits=pred_hist_logits,
                    gt_heatmap_raw=gt_hist_map,
                    pred_validity=pred_hist_val,
                    gt_validity=gt_hist_val
                )
            if val_future and (mask_fut is None or mask_fut.sum() > 0.5):
                B, K, Hm, Wm = pred_fut.shape
                pred_fut_logits = pred_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_fut_map = gt_fut.unsqueeze(2).reshape(B * K, 1, Hm, Wm)
                gt_fut_val = mask_fut.reshape(B * K, 1)
                pred_fut_val = output['future_validity'].reshape(B * K, 1)

                fut_loss, _ = criterion(
                    pred_logits=pred_fut_logits,
                    gt_heatmap_raw=gt_fut_map,
                    pred_validity=pred_fut_val,
                    gt_validity=gt_fut_val
                )
            loss = hist_loss + fut_loss

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # TensorBoard logging
    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', avg_loss, epoch)
        tb_writer.flush()  # 🔥 立即刷新验证指标

    return {'val_loss': avg_loss}


def save_checkpoint(
    model: SpatialMLLMPipeline,
    optimizer: torch.optim.Optimizer,
    scheduler,  # 🔥 新增scheduler参数
    epoch: int,
    stage_name: str,
    metrics: Dict,
    cfg: Dict
):
    """保存检查点 - 只保存可训练参数以节省磁盘空间"""
    out_dir = Path(cfg['log']['out_dir']) / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / f"epoch_{epoch}.pth"

    # 🔥 只保存可训练的参数（requires_grad=True）
    trainable_state_dict = {
        k: v for k, v in model.state_dict().items()
        if any(p.requires_grad for p in model.parameters() if id(p) == id(v))
    }

    # 🔥 更安全的方法：遍历named_parameters找到可训练的
    trainable_params = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.add(name)

    trainable_state_dict = {
        k: v for k, v in model.state_dict().items()
        if k in trainable_params
    }

    torch.save({
        'epoch': epoch,
        'stage': stage_name,
        'trainable_state_dict': trainable_state_dict,  # 🔥 只保存可训练参数
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': cfg
    }, ckpt_path)

    # 显示文件大小
    file_size_gb = ckpt_path.stat().st_size / (1024**3)
    print(f"💾 Checkpoint saved: {ckpt_path} ({file_size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config_full_model.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    set_seed(cfg['seed'])

    # 设置日志
    log_dir = Path(cfg['log']['out_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(log_dir / 'train.log'))

    # 设置TensorBoard
    tb_writer = None
    if cfg['log'].get('use_tensorboard', False):
        tb_dir = Path(cfg['log'].get('tensorboard_dir', './runs_full_model'))
        tb_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_writer = SummaryWriter(log_dir=str(tb_dir / f'run_{timestamp}'))
        logger.info(f"📊 TensorBoard enabled: {tb_dir / f'run_{timestamp}'}")

    # Loss function (NavigationHeatmapLoss)
    loss_cfg = cfg['loss']
    criterion = NavigationHeatmapLoss(
        alpha=loss_cfg['alpha'],
        lambda_mse=loss_cfg['lambda_mse'],
        lambda_kl=loss_cfg['lambda_kl'],
        lambda_valid=loss_cfg['lambda_valid']
    )

    logger.info("="*60)
    logger.info("完整模型训练 - SpatialMLLMPipeline")
    logger.info("="*60)

    # 构建数据集
    logger.info("📂 Loading datasets...")
    train_dataset = VLNHeatmapDataset(
        root=cfg['data']['root'],
        split='train',
        frames_per_clip=cfg['data']['frames_per_clip'],
        heatmap_per_clip=cfg['data']['heatmap_per_clip'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size'])
    )

    val_dataset = VLNHeatmapDataset(
        root=cfg['data']['root'],
        split='val',
        frames_per_clip=cfg['data']['frames_per_clip'],
        heatmap_per_clip=cfg['data']['heatmap_per_clip'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size'])
    )

    logger.info(f"  Train: {len(train_dataset)} clips")
    logger.info(f"  Val: {len(val_dataset)} clips")

    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg)

    # 🔥 加载checkpoint (如果提供了--resume参数)
    start_stage = 0
    start_epoch = 1
    if args.resume:
        logger.info(f"📥 Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        # 加载模型权重（支持新旧格式）
        if 'trainable_state_dict' in checkpoint:
            # 新格式：只保存了可训练参数
            model.load_state_dict(checkpoint['trainable_state_dict'], strict=False)
            logger.info(f"  ✅ Trainable weights loaded (partial checkpoint)")
        elif 'model_state_dict' in checkpoint:
            # 旧格式：保存了完整模型
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  ✅ Model weights loaded (full checkpoint)")

        # 确定从哪个stage和epoch继续
        resume_stage_name = checkpoint.get('stage', '')
        resume_epoch = checkpoint.get('epoch', 0)

        # 找到对应的stage索引
        for idx, stage_cfg in enumerate(cfg['training']['stages']):
            if stage_cfg['name'] == resume_stage_name:
                start_stage = idx
                start_epoch = resume_epoch + 1  # 从下一个epoch开始
                break

        # 如果当前stage已完成，移到下一个stage
        if start_epoch > cfg['training']['stages'][start_stage]['epochs']:
            start_stage += 1
            start_epoch = 1

        logger.info(f"  📍 Resume from Stage {start_stage + 1} ({cfg['training']['stages'][start_stage]['name']}), Epoch {start_epoch}")

    # 三阶段训练
    for stage_idx, stage_cfg in enumerate(cfg['training']['stages']):
        # 跳过已完成的stages
        if stage_idx < start_stage:
            logger.info(f"⏭️  Skipping Stage {stage_idx + 1} (already completed)")
            continue
        stage_name = stage_cfg['name']
        logger.info("="*60)
        logger.info(f"🚀 Stage {stage_idx + 1}: {stage_name}")
        logger.info("="*60)

        # 更新热力图分辨率
        hm_size = tuple(stage_cfg['hm_size'])
        train_dataset.hm_size = hm_size
        val_dataset.hm_size = hm_size
        if hasattr(model, 'update_heatmap_size'):
            model.update_heatmap_size(hm_size)
        logger.info(f"  Heatmap size: {hm_size}")

        # 构建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=True,
            num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=variable_length_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=False,
            num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=variable_length_collate_fn
        )

        # 设置可训练模块
        logger.info("🔧 Setting trainable modules...")
        set_trainable_modules(model, stage_cfg)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # 构建优化器和调度器
        optimizer = build_optimizer(model, cfg, stage_cfg)
        # 🔥 FIX: total_steps应该是optimizer step数，需要除以grad_accum_steps
        grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
        total_batches = len(train_loader) * stage_cfg['epochs']
        total_steps = total_batches // grad_accum_steps  # 实际optimizer step数
        logger.info(f"  📊 Total batches: {total_batches}, Grad accum: {grad_accum_steps}, Optimizer steps: {total_steps}")
        scheduler = build_scheduler(optimizer, cfg, total_steps)
        scaler = GradScaler()

        # 🔥 打印初始scheduler配置
        logger.info(f"  📊 Scheduler configuration:")
        logger.info(f"     Warmup steps: {int(total_steps * cfg['optim']['warmup_ratio'])}")
        logger.info(f"     Total steps: {total_steps}")
        logger.info(f"     Initial LRs: {[pg['lr'] for pg in optimizer.param_groups]}")
        logger.info(f"     Base LRs: {[pg.get('initial_lr', pg['lr']) for pg in optimizer.param_groups]}")

        # 🔥 Resume时加载优化器和调度器状态
        if args.resume and stage_idx == start_stage:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"  ✅ Optimizer state loaded")
            else:
                logger.warning(f"  ⚠️ No optimizer state in checkpoint")

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"  ✅ Scheduler state loaded (last_epoch={scheduler.last_epoch})")
            else:
                logger.warning(f"  ⚠️ No scheduler state in checkpoint - scheduler will start from epoch 0")
                logger.warning(f"     Attempting manual fix based on completed epochs...")

                # 🔥 手动修复：根据已完成的epoch数，推算应该执行的step数
                completed_epochs = start_epoch - 1  # Epoch 2 means 1 epoch completed
                steps_per_epoch = len(train_loader) // grad_accum_steps
                expected_step = completed_epochs * steps_per_epoch

                logger.info(f"     Completed epochs: {completed_epochs}, Steps per epoch: {steps_per_epoch}")
                logger.info(f"     Expected step: {expected_step}, Manually advancing scheduler...")

                # 手动step scheduler到正确位置
                for _ in range(expected_step):
                    scheduler.step()

                logger.info(f"     ✅ Scheduler manually advanced to step {scheduler.last_epoch}")

            # 🔥 打印当前调度器状态
            logger.info(f"  📊 Scheduler debug:")
            logger.info(f"     last_epoch: {scheduler.last_epoch}")
            logger.info(f"     Current LRs: {[pg['lr'] for pg in optimizer.param_groups]}")
            if hasattr(scheduler, '_schedulers'):  # SequentialLR
                for i, sub_sched in enumerate(scheduler._schedulers):
                    logger.info(f"     Sub-scheduler {i}: {type(sub_sched).__name__}")
                    if hasattr(sub_sched, 'eta_min'):
                        logger.info(f"       eta_min: {sub_sched.eta_min}")
                    if hasattr(sub_sched, 'T_max'):
                        logger.info(f"       T_max: {sub_sched.T_max}")

        # 训练循环
        best_val_loss = float('inf')

        # 确定epoch起始点（resume时使用start_epoch）
        epoch_start = start_epoch if stage_idx == start_stage else 1

        # TensorBoard: 计算当前阶段之前的总步数
        steps_per_epoch = len(train_loader) // cfg['optim'].get('grad_accum_steps', 1)
        global_step_offset = 0
        for prev_stage_idx in range(stage_idx):
            prev_epochs = cfg['training']['stages'][prev_stage_idx]['epochs']
            global_step_offset += prev_epochs * steps_per_epoch
        # 加上当前阶段已完成的epoch
        if stage_idx == start_stage and start_epoch > 1:
            global_step_offset += (start_epoch - 1) * steps_per_epoch

        for epoch in range(epoch_start, stage_cfg['epochs'] + 1):
            # ⭐ 增强 Epoch 开始日志
            logger.info("=" * 80)
            logger.info(
                f"[Stage {stage_idx+1}/{len(cfg['training']['stages'])}: {stage_name}] "
                f"Epoch {epoch}/{stage_cfg['epochs']} Started"
            )
            logger.info(
                f"  Heatmap Size: {hm_size} | Batch Size: {cfg['optim']['batch_size']} | "
                f"Grad Accum: {grad_accum_steps} | Learning Rates: {[f'{pg['lr']:.2e}' for pg in optimizer.param_groups]}"
            )
            logger.info("=" * 80)

            # 训练
            # 计算当前epoch的global_step_offset
            epoch_offset = global_step_offset + (epoch - epoch_start) * steps_per_epoch
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                cfg, criterion, epoch, logger, tb_writer, epoch_offset,
                stage_idx=stage_idx, stage_name=stage_name, stage_cfg=stage_cfg
            )

            # 验证
            val_metrics = validate(model, val_loader, cfg, logger, tb_writer, epoch, stage_cfg, criterion)
            val_loss = val_metrics['val_loss']

            # ⭐ 增强 Epoch 结束日志
            logger.info("=" * 80)
            logger.info(
                f"[Stage {stage_idx+1}: {stage_name}] Epoch {epoch}/{stage_cfg['epochs']} Completed"
            )
            logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info("=" * 80)

            # TensorBoard: 记录epoch级别的指标
            if tb_writer is not None:
                tb_writer.add_scalar('epoch/train_loss', train_loss, epoch)
                tb_writer.add_scalar('epoch/val_loss', val_loss, epoch)
                tb_writer.add_scalar('epoch/stage', stage_idx, epoch)
                tb_writer.flush()  # 🔥 每个epoch后刷新以便实时查看

            # 保存检查点
            if epoch % cfg['log']['save_every_epochs'] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_name,  # 🔥 传入scheduler
                    {'train_loss': train_loss, **val_metrics},
                    cfg
                )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  ✨ New best val_loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_name + '_best',  # 🔥 传入scheduler
                    {'train_loss': train_loss, **val_metrics},
                    cfg
                )

    logger.info("="*60)
    logger.info("✅ 训练完成！")
    logger.info("="*60)

    # 关闭TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
        logger.info("📊 TensorBoard writer closed")


if __name__ == '__main__':
    main()
