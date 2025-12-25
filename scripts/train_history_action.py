#!/usr/bin/env python3
"""
History + Action 训练脚本
==========================

两阶段训练流程：
- 阶段 A：训练 History 热力图头 + 动作头
- 阶段 B：训练 Future 热力图头 + 动作头

使用 VLNSlidingWindowDataset 进行数据加载。
"""

import sys
import os
from pathlib import Path

# 启用 expandable_segments 减少显存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Any
import warnings
import gc
import logging

warnings.filterwarnings("ignore")

from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.models.action import DiffusionActionHead, DiffusionActionConfig
from src.utils.loss import NavigationHeatmapLoss
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


# ============================================
# 配置加载与工具函数
# ============================================

def load_config(config_path: str) -> Dict:
    """加载 YAML 配置"""
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


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    滑动窗口数据集的 collate 函数
    处理可变长度的历史帧
    """
    # 找到最大历史帧数
    max_K = max(s['history_frames'].shape[0] for s in batch)
    
    # 对历史帧进行 padding
    history_frames_padded = []
    history_mask = []
    
    for s in batch:
        frames = s['history_frames']  # [K, 3, H, W]
        K = frames.shape[0]
        
        if K < max_K:
            # Padding：用最后一帧填充
            pad_size = max_K - K
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([torch.ones(K), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            mask = torch.ones(K)
        
        history_frames_padded.append(frames_padded)
        history_mask.append(mask)
    
    history_frames = torch.stack(history_frames_padded, dim=0)  # [B, max_K, 3, H, W]
    history_mask = torch.stack(history_mask, dim=0)              # [B, max_K]
    current_frame = torch.stack([s['current_frame'] for s in batch], dim=0)    # [B, 3, H, W]
    heatmap = torch.stack([s['heatmap'] for s in batch], dim=0)                # [B, Hm, Wm]
    action = torch.stack([s['action'] for s in batch], dim=0)                  # [B, 2]
    action_valid = torch.tensor([s['action_valid'] for s in batch])            # [B]
    text = [s['text'] for s in batch]
    
    return {
        'history_frames': history_frames,
        'history_mask': history_mask,
        'current_frame': current_frame,
        'heatmap': heatmap,
        'action': action,
        'action_valid': action_valid,
        'text': text,
    }


# ============================================
# 模型构建
# ============================================

def build_model(cfg: Dict) -> nn.Module:
    """
    构建完整模型（SpatialMLLMPipeline + Dual Heatmap Heads + ActionHead）
    """
    model_cfg = cfg['model']
    
    # 获取热力图头配置
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    enable_history = heatmap_cfg.get('enable_history', True)
    enable_future = heatmap_cfg.get('enable_future', True)
    
    # 构建 SpatialMLLMPipeline 配置
    pipeline_config = SpatialMLLMIntegrationConfig(
        # Frame sampling
        target_keyframes=cfg['data']['sliding_window']['num_history_sample'],
        total_frames=cfg['data']['sliding_window']['num_history_sample'] + 1,  # history + current
        sampling_method="hybrid",

        # Model paths and GPU
        llm_model_path=model_cfg['llm']['model_path'],
        vggt_gpu=model_cfg['vggt_gpu'],
        dinov3_gpu=model_cfg['dinov3_gpu'],
        llm_gpu=model_cfg['llm_gpu'],
        use_multi_gpu=model_cfg.get('use_multi_gpu', False),

        # LLM settings
        use_real_llm=model_cfg['llm']['use_real_llm'],
        llm_memory_efficient=False,

        # Heatmap settings
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_inter_frame_heatmaps=True,
        
        # Dual Heatmap Heads (Diffusion-based)
        enable_history_heatmap_head=enable_history,
        enable_future_heatmap_head=enable_future,
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),

        # Image sizes
        dinov3_img_size=cfg['data']['image_size'][0],
        vggt_img_size=cfg['data']['image_size'][0],

        # Memory optimization
        enable_gradient_checkpointing=cfg['optim'].get('gradient_checkpointing', False),

        # Diffusion Action Head
        enable_action_head=model_cfg['action_head']['enable'],
        action_dim=model_cfg['action_head']['action_dim'],
        action_pred_horizon=model_cfg['action_head']['pred_horizon'],
        action_encoding_size=model_cfg['action_head']['encoding_size'],
        action_num_diffusion_iters=model_cfg['action_head']['num_diffusion_iters'],

        verbose=True
    )
    
    model = SpatialMLLMPipeline(pipeline_config)
    
    print(f"✅ 模型已构建：SpatialMLLMPipeline + Dual Heatmap Heads + ActionHead")
    print(f"   VGGT → {model_cfg['vggt_gpu']}")
    print(f"   DINOv3 → {model_cfg['dinov3_gpu']}")
    print(f"   Qwen2.5-VL → {model_cfg['llm_gpu']} (frozen)")
    print(f"   HistoryHeatmapHead → enabled={enable_history}")
    print(f"   FutureHeatmapHead → enabled={enable_future}")
    print(f"   ActionHead → enabled={model_cfg['action_head']['enable']}")
    
    return model


def freeze_module(module: nn.Module, freeze: bool = True):
    """冻结/解冻模块"""
    for param in module.parameters():
        param.requires_grad = not freeze


def set_trainable_modules(model: SpatialMLLMPipeline, stage_cfg: Dict, logger):
    """根据阶段配置设置可训练模块"""
    # 1) 先全部冻结
    freeze_module(model, freeze=True)
    
    # 2) 根据配置解冻特定模块
    trainable = stage_cfg.get('trainable_modules', [])
    
    # History Heatmap Head (Diffusion)
    if 'history_heatmap_head' in trainable:
        if hasattr(model, 'history_heatmap_head') and model.history_heatmap_head is not None:
            freeze_module(model.history_heatmap_head, freeze=False)
            logger.info("  ✓ Unfrozen: history_heatmap_head")
            
    # Future Heatmap Head (Diffusion)
    if 'future_heatmap_head' in trainable:
        if hasattr(model, 'future_heatmap_head') and model.future_heatmap_head is not None:
            freeze_module(model.future_heatmap_head, freeze=False)
            logger.info("  ✓ Unfrozen: future_heatmap_head")
    
    # Action head
    if 'action_head' in trainable:
        if hasattr(model, 'action_head') and model.action_head is not None:
            freeze_module(model.action_head, freeze=False)
            logger.info("  ✓ Unfrozen: action_head")
    
    # Feature fusion & projector
    if 'feature_fusion' in trainable:
        if hasattr(model, 'feature_fusion'):
            freeze_module(model.feature_fusion, freeze=False)
            logger.info("  ✓ Unfrozen: feature_fusion")
    
    if 'llm_projector' in trainable:
        if hasattr(model, 'llm_projector'):
            freeze_module(model.llm_projector, freeze=False)
            logger.info("  ✓ Unfrozen: llm_projector")
    
    # Encoders
    if 'vggt' in trainable:
        if hasattr(model, 'vggt'):
            freeze_module(model.vggt, freeze=False)
            logger.info("  ✓ Unfrozen: vggt")
    
    if 'dinov3_compat' in trainable:
        if hasattr(model, 'dinov3_compat'):
            freeze_module(model.dinov3_compat, freeze=False)
            logger.info("  ✓ Unfrozen: dinov3_compat")
    
    # 3) LLM 始终冻结
    if hasattr(model, 'llm_integration') and model.llm_integration is not None:
        freeze_module(model.llm_integration, freeze=True)


def build_optimizer(model: SpatialMLLMPipeline, cfg: Dict, stage_cfg: Dict) -> torch.optim.Optimizer:
    """构建分层学习率优化器"""
    optim_cfg = cfg['optim']
    param_groups = []
    
    # 1) 热力图头 (Diffusion-based)
    hist_lr = optim_cfg.get('history_heatmap_lr', optim_cfg.get('heatmap_lr', 1e-3))
    fut_lr = optim_cfg.get('future_heatmap_lr', optim_cfg.get('heatmap_lr', 1e-3))
    
    if hasattr(model, 'history_heatmap_head') and model.history_heatmap_head is not None:
        hist_params = [p for p in model.history_heatmap_head.parameters() if p.requires_grad]
        if hist_params:
            param_groups.append({
                'params': hist_params,
                'lr': hist_lr,
                'name': 'history_heatmap_head'
            })
            print(f"  Param group: history_heatmap_head (lr={hist_lr})")
    
    if hasattr(model, 'future_heatmap_head') and model.future_heatmap_head is not None:
        fut_params = [p for p in model.future_heatmap_head.parameters() if p.requires_grad]
        if fut_params:
            param_groups.append({
                'params': fut_params,
                'lr': fut_lr,
                'name': 'future_heatmap_head'
            })
            print(f"  Param group: future_heatmap_head (lr={fut_lr})")
    
    # 2) 动作头
    action_lr = optim_cfg.get('action_lr', 1e-3)
    if hasattr(model, 'action_head') and model.action_head is not None:
        action_params = [p for p in model.action_head.parameters() if p.requires_grad]
        if action_params:
            param_groups.append({
                'params': action_params,
                'lr': action_lr,
                'name': 'action_head'
            })
            print(f"  Param group: action_head (lr={action_lr})")
    
    # 3) 融合模块+投影器
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
    
    # 4) 编码器
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
        T_max=max(1, total_steps - warmup_steps)
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )
    
    return scheduler


# ============================================
# 训练与验证
# ============================================

def train_one_epoch(
    model: SpatialMLLMPipeline,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: Dict,
    heatmap_criterion: NavigationHeatmapLoss,
    epoch: int,
    logger,
    tb_writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    stage_idx: int = 0,
    stage_name: str = "",
    stage_cfg: Dict = None
) -> Dict[str, float]:
    """训练一个 epoch"""
    
    model.train()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    
    optim_cfg = cfg['optim']
    loss_cfg = cfg['loss']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)
    
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    
    device = torch.device(cfg['model']['llm_gpu'])
    
    # 进度条
    pbar = tqdm(
        train_loader,
        desc=f"[Stage {stage_idx+1}] Epoch {epoch}/{stage_cfg['epochs']}",
        ncols=cfg['log'].get('tqdm_ncols', 120)
    )
    
    global_step = 0
    valid_batch_count = 0
    
    for i, batch in enumerate(pbar):
        # 准备数据
        # 将 history_frames 和 current_frame 合并为视频帧序列
        history_frames = batch['history_frames']  # [B, K, 3, H, W]
        current_frame = batch['current_frame']    # [B, 3, H, W]
        
        B, K, C, H, W = history_frames.shape
        
        # 合并为 [B, K+1, 3, H, W]
        video_frames = torch.cat([
            history_frames,
            current_frame.unsqueeze(1)
        ], dim=1)
        
        # GT 热力图和动作
        gt_heatmap = batch['heatmap'].to(device)        # [B, Hm, Wm]
        gt_action = batch['action'].to(device)          # [B, 2]
        action_valid = batch['action_valid'].to(device) # [B]
        text = batch['text']
        
        # 前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 模型前向
            output = model(
                video_frames=video_frames,
                instruction_text=text[0] if (text and len(text) > 0) else None,
                return_heatmaps=True,
                return_actions=train_action,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,  # [B, 1, 2]
            )
            
            # ========== 热力图损失 ==========
            heatmap_loss = torch.tensor(0.0, device=device)
            
            if train_history:
                # 使用 history_heatmaps 的第一帧（当前帧中历史帧的位置）
                pred_heatmap = output.get('history_heatmaps')  # [B, K, H, W]
                if pred_heatmap is not None:
                    # 取最后一帧的预测（对应当前帧）
                    pred_hm = pred_heatmap[:, -1, :, :]  # [B, H, W]
                    
                    # 调整尺寸如果需要
                    if pred_hm.shape[-2:] != gt_heatmap.shape[-2:]:
                        pred_hm = torch.nn.functional.interpolate(
                            pred_hm.unsqueeze(1),
                            size=gt_heatmap.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    
                    # 准备 NavigationHeatmapLoss 输入
                    pred_logits = pred_hm.unsqueeze(1)  # [B, 1, H, W]
                    gt_hm_input = gt_heatmap.unsqueeze(1)  # [B, 1, H, W]
                    
                    # 有效性标签（热力图非空则有效）
                    gt_validity = (gt_heatmap.view(B, -1).sum(dim=1) > 0.1).float().unsqueeze(1)  # [B, 1]
                    pred_validity = torch.ones_like(gt_validity)  # 假设预测总是有效
                    
                    heatmap_loss, _ = heatmap_criterion(
                        pred_logits=pred_logits,
                        gt_heatmap_raw=gt_hm_input,
                        pred_validity=pred_validity,
                        gt_validity=gt_validity
                    )
            
            if train_future:
                pred_fut = output.get('future_heatmaps')
                if pred_fut is not None:
                    # 类似处理（这里暂时使用相同 GT，实际应该有不同的 GT）
                    pred_hm = pred_fut[:, -1, :, :]
                    if pred_hm.shape[-2:] != gt_heatmap.shape[-2:]:
                        pred_hm = torch.nn.functional.interpolate(
                            pred_hm.unsqueeze(1),
                            size=gt_heatmap.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    
                    pred_logits = pred_hm.unsqueeze(1)
                    gt_hm_input = gt_heatmap.unsqueeze(1)
                    gt_validity = (gt_heatmap.view(B, -1).sum(dim=1) > 0.1).float().unsqueeze(1)
                    pred_validity = torch.ones_like(gt_validity)
                    
                    fut_loss, _ = heatmap_criterion(
                        pred_logits=pred_logits,
                        gt_heatmap_raw=gt_hm_input,
                        pred_validity=pred_validity,
                        gt_validity=gt_validity
                    )
                    heatmap_loss = heatmap_loss + fut_loss
            
            # ========== 动作损失 ==========
            action_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # 动作损失已经在模型前向中计算
                action_loss_from_model = output.get('action_loss')
                if action_loss_from_model is not None:
                    # 只对有效动作计算损失
                    if action_valid.sum() > 0:
                        action_loss = action_loss_from_model
            
            # ========== 总损失 ==========
            heatmap_weight = loss_cfg.get('history_weight', 1.0) if train_history else loss_cfg.get('future_weight', 1.0)
            action_weight = loss_cfg.get('action_weight', 0.5)
            
            loss = heatmap_weight * heatmap_loss + action_weight * action_loss
            loss = loss / grad_accum_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        valid_batch_count += 1
        
        # 清理
        del output
        
        # 梯度累积
        if valid_batch_count % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            
            # 日志
            log_interval = cfg['log'].get('log_interval', 10)
            if global_step % log_interval == 0 or global_step <= 3:
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(
                    f"[Stage {stage_idx+1}: {stage_name}] "
                    f"Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item()*grad_accum_steps:.4f} "
                    f"(hm: {heatmap_loss.item():.4f}, act: {action_loss.item():.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"GPU: {mem_alloc:.1f}GB"
                )
                
                if tb_writer is not None:
                    actual_step = global_step_offset + global_step
                    tb_writer.add_scalar('train/loss', loss.item()*grad_accum_steps, actual_step)
                    tb_writer.add_scalar('train/heatmap_loss', heatmap_loss.item(), actual_step)
                    tb_writer.add_scalar('train/action_loss', action_loss.item(), actual_step)
                    tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], actual_step)
        
        # 定期清理显存
        if (i + 1) % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        total_loss += loss.item() * grad_accum_steps
        total_heatmap_loss += heatmap_loss.item()
        total_action_loss += action_loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item()*grad_accum_steps:.4f}",
            'hm': f"{heatmap_loss.item():.4f}",
            'act': f"{action_loss.item():.4f}",
        })
    
    # 处理剩余梯度
    remaining = valid_batch_count % grad_accum_steps
    if remaining > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    
    return {
        'total_loss': total_loss / max(num_batches, 1),
        'heatmap_loss': total_heatmap_loss / max(num_batches, 1),
        'action_loss': total_action_loss / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: SpatialMLLMPipeline,
    val_loader: DataLoader,
    cfg: Dict,
    heatmap_criterion: NavigationHeatmapLoss,
    logger,
    stage_cfg: Dict,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    
    loss_cfg = cfg['loss']
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    
    device = torch.device(cfg['model']['llm_gpu'])
    
    for batch in tqdm(val_loader, desc="Validating"):
        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        B, K, C, H, W = history_frames.shape
        
        video_frames = torch.cat([
            history_frames,
            current_frame.unsqueeze(1)
        ], dim=1)
        
        gt_heatmap = batch['heatmap'].to(device)
        gt_action = batch['action'].to(device)
        action_valid = batch['action_valid'].to(device)
        text = batch['text']
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                video_frames=video_frames,
                instruction_text=text[0] if (text and len(text) > 0) else None,
                return_heatmaps=True,
                return_actions=train_action,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,
            )
            
            heatmap_loss = torch.tensor(0.0, device=device)
            if train_history and 'history_heatmaps' in output:
                pred_hm = output['history_heatmaps'][:, -1, :, :]
                if pred_hm.shape[-2:] != gt_heatmap.shape[-2:]:
                    pred_hm = torch.nn.functional.interpolate(
                        pred_hm.unsqueeze(1),
                        size=gt_heatmap.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                pred_logits = pred_hm.unsqueeze(1)
                gt_hm_input = gt_heatmap.unsqueeze(1)
                gt_validity = (gt_heatmap.view(B, -1).sum(dim=1) > 0.1).float().unsqueeze(1)
                pred_validity = torch.ones_like(gt_validity)
                
                heatmap_loss, _ = heatmap_criterion(
                    pred_logits=pred_logits,
                    gt_heatmap_raw=gt_hm_input,
                    pred_validity=pred_validity,
                    gt_validity=gt_validity
                )
            
            action_loss = torch.tensor(0.0, device=device)
            if train_action and 'action_loss' in output:
                if action_valid.sum() > 0:
                    action_loss = output['action_loss']
            
            heatmap_weight = loss_cfg.get('history_weight', 1.0)
            action_weight = loss_cfg.get('action_weight', 0.5)
            loss = heatmap_weight * heatmap_loss + action_weight * action_loss
        
        total_loss += loss.item()
        total_heatmap_loss += heatmap_loss.item()
        total_action_loss += action_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_hm = total_heatmap_loss / max(num_batches, 1)
    avg_act = total_action_loss / max(num_batches, 1)
    
    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', avg_loss, epoch)
        tb_writer.add_scalar('val/heatmap_loss', avg_hm, epoch)
        tb_writer.add_scalar('val/action_loss', avg_act, epoch)
        tb_writer.flush()
    
    return {
        'val_loss': avg_loss,
        'val_heatmap_loss': avg_hm,
        'val_action_loss': avg_act,
    }


def save_checkpoint(
    model: SpatialMLLMPipeline,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    stage_name: str,
    metrics: Dict,
    cfg: Dict
):
    """保存检查点"""
    out_dir = Path(cfg['log']['out_dir']) / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = out_dir / f"epoch_{epoch}.pth"
    
    # 只保存可训练参数
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
        'trainable_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': cfg
    }, ckpt_path)
    
    file_size_mb = ckpt_path.stat().st_size / (1024**2)
    print(f"💾 Checkpoint saved: {ckpt_path} ({file_size_mb:.1f} MB)")


# ============================================
# 主函数
# ============================================

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
    
    # TensorBoard
    tb_writer = None
    if cfg['log'].get('use_tensorboard', False):
        tb_dir = Path(cfg['log'].get('tensorboard_dir', './runs'))
        tb_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_writer = SummaryWriter(log_dir=str(tb_dir / f'history_action_{timestamp}'))
        logger.info(f"📊 TensorBoard: {tb_dir / f'history_action_{timestamp}'}")
    
    # 损失函数
    loss_cfg = cfg['loss']
    heatmap_criterion = NavigationHeatmapLoss(
        alpha=loss_cfg['alpha'],
        lambda_mse=loss_cfg['lambda_mse'],
        lambda_kl=loss_cfg['lambda_kl'],
        lambda_valid=loss_cfg['lambda_valid']
    )
    
    logger.info("=" * 60)
    logger.info("History + Action 训练")
    logger.info("=" * 60)
    
    # 构建数据集
    logger.info("📂 Loading datasets (VLNSlidingWindowDataset)...")
    sw_cfg = cfg['data']['sliding_window']
    
    train_dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split='train',
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
    )
    
    val_dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split='val',
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
    )
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    
    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg)
    
    # 多阶段训练
    for stage_idx, stage_cfg in enumerate(cfg['training']['stages']):
        stage_name = stage_cfg['name']
        logger.info("=" * 60)
        logger.info(f"🚀 Stage {stage_idx + 1}: {stage_name}")
        logger.info("=" * 60)
        
        # 更新热力图分辨率
        hm_size = tuple(stage_cfg['hm_size'])
        train_dataset.hm_size = hm_size
        val_dataset.hm_size = hm_size
        if hasattr(model, 'update_heatmap_size'):
            model.update_heatmap_size(hm_size)
        logger.info(f"  Heatmap size: {hm_size}")
        logger.info(f"  Train history: {stage_cfg.get('train_history', True)}")
        logger.info(f"  Train future: {stage_cfg.get('train_future', False)}")
        logger.info(f"  Train action: {stage_cfg.get('train_action', True)}")
        
        # 构建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=True,
            num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=collate_fn,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=False,
            num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=collate_fn,
        )
        
        # 设置可训练模块
        logger.info("🔧 Setting trainable modules...")
        set_trainable_modules(model, stage_cfg, logger)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # 构建优化器和调度器
        optimizer = build_optimizer(model, cfg, stage_cfg)
        grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
        total_batches = len(train_loader) * stage_cfg['epochs']
        total_steps = total_batches // grad_accum_steps
        logger.info(f"  Total steps: {total_steps}")
        scheduler = build_scheduler(optimizer, cfg, total_steps)
        scaler = GradScaler()
        
        # 训练循环
        best_val_loss = float('inf')
        steps_per_epoch = len(train_loader) // grad_accum_steps
        global_step_offset = 0
        
        for prev_idx in range(stage_idx):
            prev_epochs = cfg['training']['stages'][prev_idx]['epochs']
            global_step_offset += prev_epochs * steps_per_epoch
        
        for epoch in range(1, stage_cfg['epochs'] + 1):
            logger.info("=" * 80)
            logger.info(f"[Stage {stage_idx+1}: {stage_name}] Epoch {epoch}/{stage_cfg['epochs']}")
            logger.info("=" * 80)
            
            # 训练
            epoch_offset = global_step_offset + (epoch - 1) * steps_per_epoch
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                cfg, heatmap_criterion, epoch, logger, tb_writer, epoch_offset,
                stage_idx=stage_idx, stage_name=stage_name, stage_cfg=stage_cfg
            )
            
            # 验证
            val_metrics = validate(
                model, val_loader, cfg, heatmap_criterion, logger, stage_cfg, tb_writer, epoch
            )
            
            logger.info(
                f"  Train Loss: {train_metrics['total_loss']:.4f} "
                f"(hm: {train_metrics['heatmap_loss']:.4f}, act: {train_metrics['action_loss']:.4f})"
            )
            logger.info(
                f"  Val Loss: {val_metrics['val_loss']:.4f} "
                f"(hm: {val_metrics['val_heatmap_loss']:.4f}, act: {val_metrics['val_action_loss']:.4f})"
            )
            
            # 保存检查点
            if epoch % cfg['log']['save_every_epochs'] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_name,
                    {**train_metrics, **val_metrics}, cfg
                )
            
            # 保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                logger.info(f"  ✨ New best val_loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, stage_name + '_best',
                    {**train_metrics, **val_metrics}, cfg
                )
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成！")
    logger.info("=" * 60)
    
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()

