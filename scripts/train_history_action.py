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

# ============================================
# CUDA 性能优化 (针对大显存 GPU 如 H100/A100)
# ============================================
torch.backends.cudnn.benchmark = True  # 自动选择最优卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 矩阵乘法（A100/H100）
torch.backends.cudnn.allow_tf32 = True  # 启用 TF32 卷积
torch.set_float32_matmul_precision('high')  # 矩阵乘法精度（high 更快，medium 更精确）
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import warnings
import gc
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.models.pipeline import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.models.action import DiffusionActionHead, DiffusionActionConfig
from src.utils.loss import (
    NavigationHeatmapLoss,
    NeRFRippleHeatmapLoss,
    SimplifiedHeatmapLoss,
)
from src.utils.logger import setup_logger
from src.utils.notifier import FeishuNotifier, create_notifier

logger = logging.getLogger(__name__)


# ============================================
# 训练 ETA 估算器
# ============================================

class TrainingTimer:
    """训练时间和 ETA 估算器"""
    
    def __init__(self, total_epochs: int, total_stages: int = 1):
        self.total_epochs = total_epochs
        self.total_stages = total_stages
        self.start_time = None
        self.epoch_times = []
        self.stage_start_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.stage_start_time = time.time()
        
    def start_stage(self):
        """开始新阶段"""
        self.stage_start_time = time.time()
        self.epoch_times = []
        
    def end_epoch(self):
        """记录 epoch 结束"""
        if self.stage_start_time is None:
            return
        elapsed = time.time() - self.stage_start_time
        self.epoch_times.append(elapsed)
        self.stage_start_time = time.time()
        
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """获取预估剩余时间"""
        if not self.epoch_times:
            return "计算中..."
        
        avg_epoch_time = np.mean(self.epoch_times[-5:])  # 最近 5 个 epoch 的平均
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}秒"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}分钟"
        else:
            return f"{eta_seconds/3600:.1f}小时"
    
    def get_epoch_time(self) -> str:
        """获取上一个 epoch 的用时"""
        if not self.epoch_times:
            return "N/A"
        last_time = self.epoch_times[-1]
        if last_time < 60:
            return f"{last_time:.1f}s"
        else:
            return f"{last_time/60:.1f}min"
    
    def get_total_elapsed(self) -> str:
        """获取总用时"""
        if self.start_time is None:
            return "N/A"
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))


# ============================================
# 训练曲线绘制器
# ============================================

class TrainingPlotter:
    """训练曲线绘制器，每个 epoch 后更新保存图表"""
    
    def __init__(self, out_dir: Path, figsize: Tuple[int, int] = (14, 10)):
        """
        初始化绘制器
        
        Args:
            out_dir: 输出目录
            figsize: 图表尺寸
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # 存储历史数据
        self.history = {
            'epoch': [],
            'stage': [],
            'train_loss': [],
            'val_loss': [],
            'train_heatmap_loss': [],
            'val_heatmap_loss': [],
            'train_action_loss': [],
            'val_action_loss': [],
            'lr': [],
            'is_best': [],
        }
        
        # 阶段边界（用于绘制竖线）
        self.stage_boundaries = []
        self.current_stage = None
        
    def update(
        self,
        epoch: int,
        stage_name: str,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float = None,
        is_best: bool = False,
    ):
        """
        更新历史数据并保存图表
        
        Args:
            epoch: 当前 epoch（全局编号）
            stage_name: 阶段名称
            train_metrics: 训练指标
            val_metrics: 验证指标
            lr: 当前学习率
            is_best: 是否是最佳模型
        """
        # 记录阶段边界
        if stage_name != self.current_stage:
            if self.current_stage is not None:
                self.stage_boundaries.append(len(self.history['epoch']))
            self.current_stage = stage_name
        
        # 追加数据
        self.history['epoch'].append(epoch)
        self.history['stage'].append(stage_name)
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['train_heatmap_loss'].append(train_metrics.get('heatmap_loss', 0))
        self.history['val_heatmap_loss'].append(val_metrics.get('val_heatmap_loss', 0))
        self.history['train_action_loss'].append(train_metrics.get('action_loss', 0))
        self.history['val_action_loss'].append(val_metrics.get('val_action_loss', 0))
        self.history['lr'].append(lr or 0)
        self.history['is_best'].append(is_best)
        
        # 保存图表
        self.save_plot()
        
    def save_plot(self):
        """生成并保存训练曲线图"""
        if len(self.history['epoch']) == 0:
            return
        
        epochs = self.history['epoch']
        
        # 创建图表：2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('VLN Training Progress', fontsize=14, fontweight='bold')
        
        # 绘制阶段边界竖线的辅助函数
        def draw_stage_lines(ax):
            for idx in self.stage_boundaries:
                if idx < len(epochs):
                    ax.axvline(x=epochs[idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # ========== 子图 1: Total Loss ==========
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
        
        # 标注最佳点
        best_indices = [i for i, is_best in enumerate(self.history['is_best']) if is_best]
        if best_indices:
            best_epochs = [epochs[i] for i in best_indices]
            best_vals = [self.history['val_loss'][i] for i in best_indices]
            ax1.scatter(best_epochs, best_vals, c='gold', marker='*', s=100, zorder=5, label='Best Model')
        
        draw_stage_lines(ax1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # ========== 子图 2: Heatmap Loss ==========
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_heatmap_loss'], 'b-', label='Train Heatmap', linewidth=1.5)
        ax2.plot(epochs, self.history['val_heatmap_loss'], 'r-', label='Val Heatmap', linewidth=1.5)
        draw_stage_lines(ax2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Heatmap Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # ========== 子图 3: Action Loss ==========
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['train_action_loss'], 'b-', label='Train Action', linewidth=1.5)
        ax3.plot(epochs, self.history['val_action_loss'], 'r-', label='Val Action', linewidth=1.5)
        draw_stage_lines(ax3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Action Loss')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # ========== 子图 4: Learning Rate ==========
        ax4 = axes[1, 1]
        if any(lr > 0 for lr in self.history['lr']):
            ax4.plot(epochs, self.history['lr'], 'g-', linewidth=1.5)
            ax4.set_yscale('log')
        draw_stage_lines(ax4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        
        # 添加阶段标签
        if self.stage_boundaries:
            # 在底部添加阶段标注
            unique_stages = []
            seen = set()
            for s in self.history['stage']:
                if s not in seen:
                    unique_stages.append(s)
                    seen.add(s)
            stage_text = " → ".join(unique_stages)
            fig.text(0.5, 0.02, f"Stages: {stage_text}", ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # 保存
        save_path = self.out_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # 同时保存 JSON 数据（方便后续分析）
        json_path = self.out_dir / 'training_history.json'
        import json
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.history['epoch']:
            return {}
        
        best_idx = None
        best_val = float('inf')
        for i, (val, is_best) in enumerate(zip(self.history['val_loss'], self.history['is_best'])):
            if is_best and val < best_val:
                best_val = val
                best_idx = i
        
        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['epoch'][best_idx] if best_idx else None,
            'best_val_loss': best_val if best_idx else None,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
        }


# ============================================
# 热力图可视化
# ============================================

def visualize_heatmap_predictions(
    model: nn.Module,
    batch: Dict,
    output: Dict,
    epoch: int,
    step: int,
    output_dir: Path,
    num_samples: int = 2,
):
    """
    可视化热力图预测结果
    
    Args:
        model: 模型
        batch: 输入 batch
        output: 模型输出
        epoch: 当前 epoch
        step: 当前 step
        output_dir: 输出目录
        num_samples: 可视化样本数
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 获取数据
        current_frames = batch['current_frame']  # [B, 3, H, W]
        gt_heatmaps = batch['heatmap']           # [B, Hm, Wm]
        
        pred_heatmaps = output.get('history_heatmaps')  # [B, K, Hm, Wm]
        if pred_heatmaps is None:
            pred_heatmaps = output.get('future_heatmaps')
        
        if pred_heatmaps is None:
            return
        
        # 取最后一个预测（对应当前帧）
        if pred_heatmaps.dim() == 4:
            pred_heatmaps = pred_heatmaps[:, -1]  # [B, Hm, Wm]
        
        B = min(num_samples, current_frames.shape[0])
        
        fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(B):
            # RGB 图像
            rgb = current_frames[i].cpu().numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb, 0, 1)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f"Input Frame")
            axes[i, 0].axis('off')
            
            # GT 热力图
            gt_hm = gt_heatmaps[i].cpu().numpy()
            axes[i, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=1)
            axes[i, 1].set_title(f"GT Heatmap (max={gt_hm.max():.2f})")
            axes[i, 1].axis('off')
            
            # 预测热力图
            pred_hm = pred_heatmaps[i].detach().cpu().numpy()
            pred_hm = np.clip(pred_hm, 0, 1)
            axes[i, 2].imshow(pred_hm, cmap='inferno', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Pred Heatmap (max={pred_hm.max():.2f})")
            axes[i, 2].axis('off')
        
        plt.suptitle(f"Epoch {epoch}, Step {step}")
        plt.tight_layout()
        
        save_path = vis_dir / f"epoch_{epoch:03d}_step_{step:05d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        return None


def build_heatmap_loss(loss_cfg: Dict, loss_type: str = None) -> nn.Module:
    """
    根据配置构建热力图损失函数。
    
    Args:
        loss_cfg: 完整的 loss 配置字典
        loss_type: 损失类型，覆盖配置中的默认值
        
    Returns:
        nn.Module: 热力图损失函数
    """
    # 优先使用传入的 loss_type，否则使用配置默认值
    loss_type = loss_type or loss_cfg.get('heatmap_loss_type', 'simplified')
    
    if loss_type == 'simplified':
        params = loss_cfg.get('simplified', {})
        return SimplifiedHeatmapLoss(
            lambda_mse=params.get('lambda_mse', 1.0),
            lambda_grad=params.get('lambda_grad', 0.3),
            peak_weight=params.get('peak_weight', 5.0),
        )
    
    elif loss_type == 'nerf_ripple':
        params = loss_cfg.get('nerf_ripple', {})
        return NeRFRippleHeatmapLoss(
            lambda_mse=params.get('lambda_mse', 1.0),
            lambda_fft=params.get('lambda_fft', 0.3),
            lambda_radial=params.get('lambda_radial', 0.2),
            lambda_peak=params.get('lambda_peak', 1.0),
            fft_weight_decay=params.get('fft_weight_decay', 0.5),
        )
    
    elif loss_type == 'navigation':
        params = loss_cfg.get('navigation', {})
        return NavigationHeatmapLoss(
            alpha=params.get('alpha', 20.0),
            lambda_mse=params.get('lambda_mse', 1.0),
            lambda_kl=params.get('lambda_kl', 0.2),
            lambda_valid=params.get('lambda_valid', 0.1),
        )
    
    else:
        raise ValueError(f"Unknown heatmap loss type: {loss_type}")


def compute_heatmap_loss(
    heatmap_criterion: nn.Module,
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """
    统一的热力图损失计算函数，适配不同 Loss 类的接口。
    
    Args:
        heatmap_criterion: 热力图损失函数
        pred_heatmap: [B, H, W] 或 [B, 1, H, W] 预测热力图
        gt_heatmap: [B, H, W] 或 [B, 1, H, W] GT 热力图
        loss_type: 损失类型
        
    Returns:
        torch.Tensor: 损失值
    """
    # 确保维度一致
    if pred_heatmap.dim() == 3:
        pred_hm = pred_heatmap.unsqueeze(1)
    else:
        pred_hm = pred_heatmap
    
    if gt_heatmap.dim() == 3:
        gt_hm = gt_heatmap.unsqueeze(1)
    else:
        gt_hm = gt_heatmap
    
    if loss_type == 'navigation':
        # NavigationHeatmapLoss 需要额外的 validity 参数
        B = gt_hm.shape[0]
        gt_validity = (gt_hm.view(B, -1).sum(dim=1) > 0.1).float().unsqueeze(1)
        pred_validity = torch.ones_like(gt_validity)
        
        loss, _ = heatmap_criterion(
            pred_logits=pred_hm,
            gt_heatmap_raw=gt_hm,
            pred_validity=pred_validity,
            gt_validity=gt_validity,
            smooth_gt=False,  # 保留波纹
        )
    else:
        # SimplifiedHeatmapLoss 和 NeRFRippleHeatmapLoss 使用简单接口
        loss, _ = heatmap_criterion(pred_hm, gt_hm)
    
    return loss


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

        verbose=False  # 关闭冗余日志，加快训练
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
    heatmap_criterion: nn.Module,
    epoch: int,
    logger,
    tb_writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    stage_idx: int = 0,
    stage_name: str = "",
    stage_cfg: Dict = None,
    max_batches: int = None,
) -> Dict[str, float]:
    """训练一个 epoch
    
    Args:
        max_batches: 最大 batch 数（用于快速调试，None 表示不限制）
    """
    
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
    
    # 确定实际的 batch 数量
    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式: 只处理 {total_batches} batches")
    
    # 进度条
    pbar = tqdm(
        train_loader,
        desc=f"[Stage {stage_idx+1}] Epoch {epoch}/{stage_cfg['epochs']}",
        total=total_batches,
        ncols=cfg['log'].get('tqdm_ncols', 120)
    )
    
    global_step = 0
    valid_batch_count = 0
    
    for i, batch in enumerate(pbar):
        # 快速调试模式：限制 batch 数量
        if max_batches is not None and i >= max_batches:
            logger.info(f"  ⚡ 达到 max_batches={max_batches}，提前结束 epoch")
            break
        
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
        
        # 处理导航指令
        # 注意：Qwen2.5-VL 当前只支持单一指令输入
        # 如果 batch 中有不同指令，合并为一个（用分隔符连接）
        if text and len(text) > 0:
            unique_texts = list(set(t for t in text if t))  # 去重非空指令
            if len(unique_texts) == 0:
                instruction_text = None
            elif len(unique_texts) == 1:
                instruction_text = unique_texts[0]
            else:
                # 多个不同指令：合并（实际场景中同一 batch 的指令通常相同）
                instruction_text = unique_texts[0]  # 使用第一个非空指令
                if i == 0:  # 只在第一个 batch 警告一次
                    logger.warning(f"Batch contains {len(unique_texts)} different instructions, using first one")
        else:
            instruction_text = None
        
        # 前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 模型前向
            output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame.to(device),  # 显式传递当前观测
                return_heatmaps=True,
                return_actions=train_action,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,  # [B, 1, 2]
            )
            
            # ========== 热力图损失 ==========
            heatmap_loss = torch.tensor(0.0, device=device)
            loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')
            
            if train_history:
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
                    
                    # 使用统一的损失计算函数
                    heatmap_loss = compute_heatmap_loss(
                        heatmap_criterion, pred_hm, gt_heatmap, loss_type
                    )
            
            if train_future:
                pred_fut = output.get('future_heatmaps')
                if pred_fut is not None:
                    pred_hm = pred_fut[:, -1, :, :]
                    if pred_hm.shape[-2:] != gt_heatmap.shape[-2:]:
                        pred_hm = torch.nn.functional.interpolate(
                            pred_hm.unsqueeze(1),
                            size=gt_heatmap.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    
                    fut_loss = compute_heatmap_loss(
                        heatmap_criterion, pred_hm, gt_heatmap, loss_type
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
            
            # 定期可视化热力图预测
            vis_interval = cfg['log'].get('vis_every_steps', 500)
            if global_step % vis_interval == 0 and global_step > 0:
                vis_path = visualize_heatmap_predictions(
                    model=model,
                    batch=batch,
                    output=output,
                    epoch=epoch,
                    step=global_step,
                    output_dir=Path(cfg['log']['out_dir']),
                    num_samples=2,
                )
                if vis_path and tb_writer is not None:
                    # 读取可视化图像并写入 TensorBoard
                    try:
                        import cv2
                        vis_img = cv2.imread(str(vis_path))
                        if vis_img is not None:
                            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                            vis_img = vis_img.transpose(2, 0, 1)  # HWC -> CHW
                            tb_writer.add_image('train/heatmap_viz', vis_img, actual_step)
                    except Exception as e:
                        pass  # 忽略可视化写入错误
        
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


@torch.inference_mode()  # 比 no_grad 更高效，禁用所有梯度跟踪
def validate(
    model: SpatialMLLMPipeline,
    val_loader: DataLoader,
    cfg: Dict,
    heatmap_criterion: nn.Module,
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
    loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')
    
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
        
        # 处理导航指令（与训练一致）
        if text and len(text) > 0:
            unique_texts = list(set(t for t in text if t))
            instruction_text = unique_texts[0] if unique_texts else None
        else:
            instruction_text = None
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame.to(device),
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
                
                heatmap_loss = compute_heatmap_loss(
                    heatmap_criterion, pred_hm, gt_heatmap, loss_type
                )
            
            action_loss = torch.tensor(0.0, device=device)
            if train_action and 'action_loss' in output:
                action_loss_val = output.get('action_loss')
                if action_loss_val is not None and action_valid.sum() > 0:
                    action_loss = action_loss_val
            
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


class CheckpointManager:
    """管理检查点的保存、加载和清理"""
    
    def __init__(self, out_dir: str, max_ckpts: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_ckpts = max_ckpts
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None
        self.ckpt_history = []  # [(path, val_loss, epoch)]
    
    def save(
        self,
        model: SpatialMLLMPipeline,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        stage_idx: int,
        stage_name: str,
        metrics: Dict,
        cfg: Dict,
        is_best: bool = False,
        scaler: GradScaler = None,
    ) -> Path:
        """保存检查点
        
        Args:
            is_best: 是否是当前最佳模型
            scaler: GradScaler 用于 AMP 断点续训
        
        Returns:
            保存的检查点路径
        """
        stage_dir = self.out_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集可训练参数名
        trainable_params = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.add(name)
        
        # 只保存可训练参数（节省空间）
        trainable_state_dict = {
            k: v for k, v in model.state_dict().items()
            if k in trainable_params
        }
        
        # 构建检查点
        ckpt = {
            'epoch': epoch,
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'trainable_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': cfg,
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存 GradScaler 状态（AMP）
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        
        # 保存定期检查点
        ckpt_path = stage_dir / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        file_size_mb = ckpt_path.stat().st_size / (1024**2)
        print(f"💾 Checkpoint: {ckpt_path.name} ({file_size_mb:.1f} MB)")
        
        # 记录历史
        val_loss = metrics.get('val_loss', float('inf'))
        self.ckpt_history.append((ckpt_path, val_loss, epoch))
        
        # 保存最佳模型（全局最佳，放在根目录）
        if is_best:
            self.best_val_loss = val_loss
            best_path = self.out_dir / "best_model.pth"
            torch.save(ckpt, best_path)
            self.best_ckpt_path = best_path
            print(f"⭐ Best model updated: val_loss={val_loss:.4f}")
        
        # 保存最新检查点（用于恢复训练）
        latest_path = self.out_dir / "latest.pth"
        torch.save(ckpt, latest_path)
        
        # 清理旧检查点（每个阶段保留 max_ckpts 个）
        self._cleanup_old_ckpts(stage_dir)
        
        return ckpt_path
    
    def _cleanup_old_ckpts(self, stage_dir: Path):
        """清理旧的检查点，只保留最新的 max_ckpts 个"""
        ckpts = sorted(stage_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.max_ckpts:
            old_ckpt = ckpts.pop(0)
            old_ckpt.unlink()
            print(f"🗑️  Removed old checkpoint: {old_ckpt.name}")
    
    def load(self, ckpt_path: str) -> Dict:
        """加载检查点"""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        return ckpt
    
    def get_latest(self) -> Optional[Path]:
        """获取最新检查点路径"""
        latest = self.out_dir / "latest.pth"
        return latest if latest.exists() else None
    
    def get_best(self) -> Optional[Path]:
        """获取最佳检查点路径"""
        best = self.out_dir / "best_model.pth"
        return best if best.exists() else None


def load_checkpoint_for_resume(
    ckpt_path: str,
    model: SpatialMLLMPipeline,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: GradScaler = None,
    logger = None,
) -> Dict:
    """加载检查点用于断点续训
    
    Args:
        ckpt_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 调度器（可选）
        scaler: GradScaler（可选，用于 AMP）
        logger: 日志器
    
    Returns:
        包含 epoch, stage_idx, stage_name, metrics, best_val_loss 的字典
    """
    if logger:
        logger.info(f"📂 Loading checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 加载模型参数
    state_dict = ckpt.get('trainable_state_dict', {})
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if logger:
            logger.info(f"  ✓ Loaded {len(state_dict)} trainable parameters")
            if missing:
                logger.info(f"  Missing keys: {len(missing)} (using pretrained)")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if logger:
                logger.info("  ✓ Optimizer state restored")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠ Failed to restore optimizer: {e}")
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if logger:
                logger.info("  ✓ Scheduler state restored")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠ Failed to restore scheduler: {e}")
    
    # 加载 GradScaler 状态（AMP）
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


# Legacy function for compatibility
def save_checkpoint(
    model: SpatialMLLMPipeline,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    stage_name: str,
    metrics: Dict,
    cfg: Dict
):
    """保存检查点（旧接口，保持兼容）"""
    out_dir = Path(cfg['log']['out_dir']) / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = out_dir / f"epoch_{epoch}.pth"
    
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
    parser = argparse.ArgumentParser(description="History + Action 训练脚本")
    parser.add_argument('--config', type=str, default='configs/training_config_full_model.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                        help='从检查点恢复（路径或 "latest"）')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动从最新检查点恢复（如果存在）')
    
    # 阶段控制参数
    parser.add_argument('--stage', type=str, default=None,
                        help='指定运行的阶段名称（如 warmup_64, mid_128, full_224）')
    parser.add_argument('--stage-index', type=int, default=None,
                        help='指定运行的阶段索引（0, 1, 2...）')
    parser.add_argument('--stage-only', action='store_true',
                        help='只运行指定的单个阶段，不继续后续阶段')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='从指定 epoch 开始训练（默认 1）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的 epoch 数量')
    
    # 调试参数
    parser.add_argument('--dry-run', action='store_true',
                        help='只构建模型和数据，不实际训练（用于测试配置）')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='每个 epoch 最多处理的 batch 数（用于快速调试）')
    
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
    
    # 损失函数配置（每个阶段可能不同）
    loss_cfg = cfg['loss']
    default_loss_type = loss_cfg.get('heatmap_loss_type', 'simplified')
    logger.info(f"📊 Default heatmap loss type: {default_loss_type}")
    
    logger.info("=" * 60)
    logger.info("History + Action 训练")
    logger.info("=" * 60)
    
    # 构建数据集
    logger.info("📂 Loading datasets (VLNSlidingWindowDataset)...")
    sw_cfg = cfg['data']['sliding_window']
    
    # 采样步长：控制训练速度（stride=5 表示样本数减少 5 倍）
    sample_stride = sw_cfg.get('sample_stride', 1)
    
    train_dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split='train',
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
        sample_stride=sample_stride,
    )
    
    val_split = cfg['data'].get('val_split', 'val')  # 支持 val_unseen 等
    val_dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split=val_split,
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
        sample_stride=sample_stride,  # 验证集也使用相同步长
    )
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    
    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg)
    
    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        out_dir=cfg['log']['out_dir'],
        max_ckpts=cfg['log'].get('max_ckpts', 3)
    )
    
    # 创建通知器
    notifier = create_notifier(cfg)
    if notifier:
        logger.info("📢 飞书通知已启用")
    
    # 创建训练曲线绘制器
    plotter = TrainingPlotter(out_dir=log_dir)
    logger.info(f"📈 训练曲线将保存至: {log_dir / 'training_curves.png'}")
    
    # 断点续训
    resume_epoch = 0
    resume_stage_idx = 0
    resume_path = None
    
    # 确定恢复路径
    if args.resume:
        # 显式指定检查点路径
        if args.resume == 'latest':
            resume_path = ckpt_manager.get_latest()
            if resume_path is None:
                logger.warning("⚠️ No latest checkpoint found, starting from scratch")
        else:
            resume_path = Path(args.resume)
    elif args.auto_resume:
        # 自动检测最新检查点
        resume_path = ckpt_manager.get_latest()
        if resume_path:
            logger.info(f"🔄 Auto-resume: Found checkpoint {resume_path}")
        else:
            logger.info("🆕 Auto-resume: No checkpoint found, starting fresh")
    
    if resume_path and Path(resume_path).exists():
        resume_info = load_checkpoint_for_resume(
            str(resume_path), model, optimizer=None, scheduler=None, logger=logger
        )
        resume_epoch = resume_info['epoch']
        resume_stage_idx = resume_info['stage_idx']
        ckpt_manager.best_val_loss = resume_info['best_val_loss']
        logger.info(f"  📌 Resuming from epoch {resume_epoch}, stage {resume_stage_idx}")
        logger.info(f"  📊 Best val_loss so far: {ckpt_manager.best_val_loss:.4f}")
    
    # Dry run 模式
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("🧪 Dry run 模式：模型和数据构建成功，退出")
        logger.info("=" * 60)
        return
    
    # 确定要运行的阶段
    all_stages = cfg['training']['stages']
    stage_names = [s['name'] for s in all_stages]
    
    # 确定起始阶段
    start_stage_idx = resume_stage_idx  # 默认使用恢复的阶段
    if args.stage_index is not None:
        start_stage_idx = args.stage_index
        logger.info(f"📌 使用阶段索引: {start_stage_idx}")
    elif args.stage is not None:
        if args.stage in stage_names:
            start_stage_idx = stage_names.index(args.stage)
            logger.info(f"📌 使用阶段名称: {args.stage} (index={start_stage_idx})")
        else:
            logger.error(f"❌ 未知阶段名称: {args.stage}")
            logger.error(f"   可用阶段: {stage_names}")
            return
    
    # 确定结束阶段
    if args.stage_only:
        end_stage_idx = start_stage_idx + 1
        logger.info(f"📌 只运行单个阶段: {stage_names[start_stage_idx]}")
    else:
        end_stage_idx = len(all_stages)
        if start_stage_idx > 0:
            logger.info(f"📌 从阶段 {start_stage_idx} 运行到最后")
    
    # 打印阶段信息
    logger.info("=" * 60)
    logger.info("📋 训练阶段计划:")
    for i, s in enumerate(all_stages):
        marker = "▶" if start_stage_idx <= i < end_stage_idx else "○"
        logger.info(f"  {marker} Stage {i}: {s['name']} ({s['epochs']} epochs, hm={s['hm_size']})")
    logger.info("=" * 60)
    
    # 计算总 epoch 数
    total_all_epochs = sum(s['epochs'] for s in all_stages[start_stage_idx:end_stage_idx])
    global_epoch_counter = 0  # 全局 epoch 计数器
    
    # 发送训练开始通知
    if notifier:
        try:
            notifier.send_training_start(
                config_name=Path(args.config).stem,
                stages=all_stages[start_stage_idx:end_stage_idx],
                total_epochs=total_all_epochs,
            )
        except Exception as e:
            logger.warning(f"发送开始通知失败: {e}")
    
    # 多阶段训练
    for stage_idx in range(start_stage_idx, end_stage_idx):
        stage_cfg = all_stages[stage_idx]
        stage_name = stage_cfg['name']
        
        # 覆盖 epoch 数量
        if args.epochs is not None:
            stage_cfg = stage_cfg.copy()
            stage_cfg['epochs'] = args.epochs
            logger.info(f"📌 覆盖 epochs: {args.epochs}")
        
        logger.info("=" * 60)
        logger.info(f"🚀 Stage {stage_idx + 1}/{len(all_stages)}: {stage_name}")
        logger.info("=" * 60)
        
        # 更新热力图分辨率
        hm_size = tuple(stage_cfg['hm_size'])
        train_dataset.hm_size = hm_size
        val_dataset.hm_size = hm_size
        if hasattr(model, 'update_heatmap_size'):
            model.update_heatmap_size(hm_size)
        
        # 构建本阶段的热力图损失函数
        stage_loss_type = stage_cfg.get('heatmap_loss_type', default_loss_type)
        heatmap_criterion = build_heatmap_loss(loss_cfg, stage_loss_type)
        logger.info(f"  🎯 Heatmap loss: {stage_loss_type} ({type(heatmap_criterion).__name__})")
        
        logger.info(f"  Heatmap size: {hm_size}")
        logger.info(f"  Train history: {stage_cfg.get('train_history', True)}")
        logger.info(f"  Train future: {stage_cfg.get('train_future', False)}")
        logger.info(f"  Train action: {stage_cfg.get('train_action', True)}")
        
        # 构建数据加载器
        # DataLoader 性能优化参数
        num_workers = cfg['data']['num_workers']
        prefetch_factor = cfg['data'].get('prefetch_factor', 2)
        persistent_workers = num_workers > 0  # 保持 worker 进程，减少启动开销
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
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
        
        # 断点续训：加载优化器、调度器和 scaler 状态
        if resume_path and stage_idx == resume_stage_idx:
            if Path(resume_path).exists():
                load_checkpoint_for_resume(
                    str(resume_path), model, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    scaler=scaler,  # 恢复 AMP 状态
                    logger=logger
                )
        
        # 训练循环
        best_val_loss = ckpt_manager.best_val_loss  # 使用全局最佳
        steps_per_epoch = len(train_loader) // grad_accum_steps
        global_step_offset = 0
        
        for prev_idx in range(stage_idx):
            prev_epochs = all_stages[prev_idx]['epochs']
            global_step_offset += prev_epochs * steps_per_epoch
        
        # 确定起始 epoch
        if stage_idx == start_stage_idx and resume_epoch > 0:
            start_epoch = resume_epoch + 1  # 从恢复的下一个 epoch 开始
        elif stage_idx == start_stage_idx:
            start_epoch = args.start_epoch
        else:
            start_epoch = 1
        total_epochs = stage_cfg['epochs']
        
        # Early stopping
        patience = cfg['validation'].get('patience', 5)
        no_improve_count = 0
        
        # 训练计时器
        timer = TrainingTimer(total_epochs=total_epochs)
        timer.start()
        
        # 可视化配置
        vis_interval = cfg['log'].get('vis_every_steps', 500)
        vis_dir = Path(cfg['log']['out_dir'])
        
        for epoch in range(start_epoch, total_epochs + 1):
            timer.start_stage()  # 开始计时
            
            logger.info("=" * 80)
            logger.info(f"[Stage {stage_idx+1}: {stage_name}] Epoch {epoch}/{total_epochs}")
            if epoch > start_epoch:
                eta = timer.get_eta(epoch - 1, total_epochs)
                logger.info(f"  ⏱️  Last epoch: {timer.get_epoch_time()} | ETA: {eta} | Total: {timer.get_total_elapsed()}")
            logger.info("=" * 80)
            
            # 训练
            epoch_offset = global_step_offset + (epoch - 1) * steps_per_epoch
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                cfg, heatmap_criterion, epoch, logger, tb_writer, epoch_offset,
                stage_idx=stage_idx, stage_name=stage_name, stage_cfg=stage_cfg,
                max_batches=args.max_batches
            )
            
            timer.end_epoch()  # 记录 epoch 结束
            
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
            
            # 显示 ETA
            eta = timer.get_eta(epoch, total_epochs)
            logger.info(f"  ⏱️  Epoch time: {timer.get_epoch_time()} | ETA: {eta}")
            
            # 检查是否为最佳模型
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                no_improve_count = 0
                logger.info(f"  ⭐ New best val_loss: {best_val_loss:.4f}")
            else:
                no_improve_count += 1
                logger.info(f"  No improvement for {no_improve_count}/{patience} epochs")
            
            # 更新全局 epoch 计数器
            global_epoch_counter += 1
            
            # 获取当前学习率
            current_lr = scheduler.get_last_lr()[0] if scheduler else 0
            
            # 更新训练曲线图
            plotter.update(
                epoch=global_epoch_counter,
                stage_name=stage_name,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                is_best=is_best,
            )
            logger.info(f"  📈 训练曲线已更新: {log_dir / 'training_curves.png'}")
            
            # 发送飞书通知
            if notifier:
                try:
                    notifier.send_epoch_report(
                        epoch=epoch,
                        total_epochs=total_epochs,
                        stage_name=stage_name,
                        stage_idx=stage_idx,
                        total_stages=len(all_stages),
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        eta=eta,
                        epoch_time=timer.get_epoch_time(),
                        is_best=is_best,
                        best_val_loss=best_val_loss,
                    )
                except Exception as e:
                    logger.warning(f"发送通知失败: {e}")
            
            # 保存检查点（使用 CheckpointManager，包含 scaler 状态）
            if epoch % cfg['log']['save_every_epochs'] == 0 or is_best:
                ckpt_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    metrics={**train_metrics, **val_metrics},
                    cfg=cfg,
                    is_best=is_best,
                    scaler=scaler,  # 保存 AMP 状态
                )
            
            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"  🛑 Early stopping triggered after {patience} epochs without improvement")
                break
        
        # 阶段结束日志
        logger.info(f"  📊 Stage {stage_name} completed in {timer.get_total_elapsed()}")
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成！")
    logger.info("=" * 60)
    
    # 输出训练摘要
    summary = plotter.get_summary()
    if summary:
        logger.info(f"📊 训练摘要:")
        logger.info(f"   总 Epochs: {summary.get('total_epochs', 'N/A')}")
        logger.info(f"   最佳 Epoch: {summary.get('best_epoch', 'N/A')}")
        logger.info(f"   最佳 val_loss: {summary.get('best_val_loss', 'N/A'):.4f}" if summary.get('best_val_loss') else "")
        logger.info(f"   最终 train_loss: {summary.get('final_train_loss', 0):.4f}")
        logger.info(f"   最终 val_loss: {summary.get('final_val_loss', 0):.4f}")
    
    # 发送训练完成通知
    if notifier:
        try:
            notifier.send_training_complete(
                total_time=timer.get_total_elapsed() if timer else "N/A",
                best_val_loss=best_val_loss,
                final_stage=stage_name if 'stage_name' in dir() else "完成",
            )
        except Exception as e:
            logger.warning(f"发送完成通知失败: {e}")
    
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()

