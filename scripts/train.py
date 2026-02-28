#!/usr/bin/env python3
"""
VLN 训练脚本
==============

使用 Qwen3-VL 进行视觉语言导航训练。
单阶段训练：History 热力图头 + Action Head + Stop Head
"""

import sys
import os
from pathlib import Path

# 启用 expandable_segments 减少显存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 禁用 tokenizers 并行，避免多进程 fork 冲突导致死锁
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch

# ============================================
# CUDA 性能优化
# ============================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
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
import psutil
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")
# 特别抑制 Qwen-VL 的 fps 警告（我们使用 nframes 而不是 fps 采样）
warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
warnings.filterwarnings("ignore", message="Asked to sample")

from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset, VLNTrajectoryDataset
from src.data.packing_collator import PackingCollatorForVLN
from src.data.tokenized_dataset import TokenizedVLNDataset, FlattenedCollatorForVLN
from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.utils.logger import setup_logger
from src.utils.gpu_heatmap import GPUHeatmapComputer
from src.utils.notifier import FeishuNotifier, create_notifier

logger = logging.getLogger(__name__)


# ============================================
# Worker 初始化函数（模块级别，支持 spawn 多进程）
# ============================================

def _malloc_trim():
    """强制 glibc 将释放的内存归还操作系统（解决 Python 内存碎片化问题）"""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _worker_init_fn(worker_id):
    """Worker 进程初始化函数 - 抑制警告 + 内存管理"""
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
    warnings.filterwarnings("ignore", message="Asked to sample")
    
    # 设置 worker 级别的内存管理：降低 glibc 的 mmap 阈值
    # 使得大块内存分配使用 mmap，释放后可以直接归还 OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        # M_MMAP_THRESHOLD = -3, 设置为 32KB，让更多分配走 mmap
        libc.mallopt(-3, 32 * 1024)
        # M_TRIM_THRESHOLD = -1, 设置为 64KB，更积极地归还内存
        libc.mallopt(-1, 64 * 1024)
    except Exception:
        pass


# ============================================
# EMA (Exponential Moving Average)
# ============================================

class EMAModel:
    """
    Exponential Moving Average for model parameters (with warmup).
    
    扩散模型标准技术：用参数的滑动平均做推理，
    避免训练末期的参数波动，显著提升泛化性能。
    
    ⚠️ 关键实现细节：shadow 参数必须使用 float32 存储和计算。
    bfloat16 只有 7 位尾数（精度 ~0.78%），而 EMA alpha=0.001，
    远低于 bfloat16 精度阈值，会导致 add_(param, alpha=0.001) 完全无效。
    
    Warmup 策略：前 warmup_steps 步使用自适应 decay = 1 - 1/(step+1)，
    避免初始随机权重污染 EMA shadow。
    
    用法：
        ema = EMAModel(model, decay=0.999)
        # 每次 optimizer.step() 后调用
        ema.update()
        # 验证时用 EMA 参数
        with ema.apply():
            validate(model, ...)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 2000):
        self.model = model
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        
        # 只追踪需要训练的参数（节省内存）
        # ⚠️ 必须使用 float32！bfloat16 精度不足，EMA 更新会完全无效
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().clone()
    
    def _get_decay(self) -> float:
        """获取当前 decay 值（含 warmup）"""
        # Warmup: decay 从 0 逐步增加到 target_decay
        # 公式: min(target_decay, 1 - 1/(step+1))
        # step=0 → decay=0 (直接复制当前参数)
        # step=999 → decay=0.999 (达到目标)
        warmup_decay = 1.0 - 1.0 / (self.step_count + 1)
        return min(self.target_decay, warmup_decay)
    
    @torch.no_grad()
    def update(self):
        """更新 EMA 参数：shadow = decay * shadow + (1 - decay) * param（float32 精度）"""
        decay = self._get_decay()
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                # param 可能是 bfloat16，必须转 float32 再做 EMA 运算
                self.shadow[name].mul_(decay).add_(param.data.float(), alpha=1.0 - decay)
        self.step_count += 1
    
    def apply(self):
        """Context manager：临时将模型参数替换为 EMA 参数"""
        return _EMAContext(self)
    
    @property
    def decay(self):
        return self._get_decay()
    
    def state_dict(self):
        return {'shadow': self.shadow, 'target_decay': self.target_decay, 'step_count': self.step_count}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        # 确保加载的 shadow 也是 float32
        for name in self.shadow:
            if self.shadow[name].dtype != torch.float32:
                self.shadow[name] = self.shadow[name].float()
        self.target_decay = state_dict.get('target_decay', self.target_decay)
        self.step_count = state_dict.get('step_count', 0)


class _EMAContext:
    """EMA 上下文管理器：进入时替换为 EMA 参数，退出时恢复原始参数"""
    
    def __init__(self, ema: EMAModel):
        self.ema = ema
    
    def __enter__(self):
        self.ema.backup = {}
        for name, param in self.ema.model.named_parameters():
            if name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                # EMA shadow 是 float32，需要转回模型原始 dtype
                param.data.copy_(self.ema.shadow[name].to(dtype=param.dtype))
        return self.ema.model
    
    def __exit__(self, *args):
        for name, param in self.ema.model.named_parameters():
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}


# ============================================
# 训练 ETA 估算器
# ============================================

class TrainingTimer:
    """训练时间和 ETA 估算器"""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
    def start_epoch(self):
        """开始新 epoch"""
        self.epoch_start_time = time.time()
        
    def end_epoch(self):
        """记录 epoch 结束"""
        if self.epoch_start_time is None:
            return
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        self.epoch_start_time = time.time()
        
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """获取预估剩余时间"""
        if not self.epoch_times:
            return "计算中..."
        
        avg_epoch_time = np.mean(self.epoch_times[-5:])
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
    """训练曲线绘制器"""
    
    def __init__(self, out_dir: Path, figsize: Tuple[int, int] = (14, 10)):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
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
        """更新历史数据并保存图表"""
        if stage_name != self.current_stage:
            if self.current_stage is not None:
                self.stage_boundaries.append(len(self.history['epoch']))
            self.current_stage = stage_name
        
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
        
        self.save_plot()
        
    def save_plot(self):
        """生成并保存训练曲线图"""
        if len(self.history['epoch']) == 0:
            return
        
        epochs = self.history['epoch']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('VLN Training Progress', fontsize=14, fontweight='bold')
        
        def draw_stage_lines(ax):
            for idx in self.stage_boundaries:
                if idx < len(epochs):
                    ax.axvline(x=epochs[idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Total Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
        
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
        
        # Heatmap Loss
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_heatmap_loss'], 'b-', label='Train Heatmap', linewidth=1.5)
        ax2.plot(epochs, self.history['val_heatmap_loss'], 'r-', label='Val Heatmap', linewidth=1.5)
        draw_stage_lines(ax2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Heatmap Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Action Loss
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['train_action_loss'], 'b-', label='Train Action', linewidth=1.5)
        ax3.plot(epochs, self.history['val_action_loss'], 'r-', label='Val Action', linewidth=1.5)
        draw_stage_lines(ax3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Action Loss')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        ax4 = axes[1, 1]
        if any(lr > 0 for lr in self.history['lr']):
            ax4.plot(epochs, self.history['lr'], 'g-', linewidth=1.5)
            ax4.set_yscale('log')
        draw_stage_lines(ax4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        
        if self.stage_boundaries:
            unique_stages = []
            seen = set()
            for s in self.history['stage']:
                if s not in seen:
                    unique_stages.append(s)
                    seen.add(s)
            stage_text = " → ".join(unique_stages)
            fig.text(0.5, 0.02, f"Stages: {stage_text}", ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        save_path = self.out_dir / 'curves.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # 保存 JSON 数据
        json_path = self.out_dir / 'history.json'
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
    gt_heatmap_override: Optional[torch.Tensor] = None,
):
    """可视化热力图预测结果
    
    Args:
        gt_heatmap_override: 当使用 defer_heatmap_to_gpu 时，传入 GPU 计算的 GT 热力图，
                            替代 batch['heatmap']（后者是零占位符）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        current_frames = batch['current_frame']
        gt_heatmaps = gt_heatmap_override if gt_heatmap_override is not None else batch['heatmap']
        
        pred_heatmaps = output.get('history_heatmaps')
        if pred_heatmaps is None:
            pred_heatmaps = output.get('future_heatmaps')
        
        if pred_heatmaps is None:
            return
        
        # 提取 2D 热力图用于可视化
        # 多视角 [B, 4, H, W]: 取 front (index 0)
        # 单视角 [B, 1, H, W]: 取唯一通道 (index 0 或 -1 均可)
        if pred_heatmaps.dim() == 4 and pred_heatmaps.shape[1] == 4:
            pred_heatmaps = pred_heatmaps[:, 0]  # front view
        elif pred_heatmaps.dim() == 4:
            pred_heatmaps = pred_heatmaps[:, -1]  # single channel
        
        if gt_heatmaps.dim() == 4 and gt_heatmaps.shape[1] == 4:
            gt_heatmaps = gt_heatmaps[:, 0]  # front view
        
        # 全景网格：裁剪左上角作为 front view 显示
        if current_frames.shape[-1] > 300:
            h = current_frames.shape[-2] // 2
            w = current_frames.shape[-1] // 2
            current_frames = current_frames[:, :, :h, :w]
        elif 'current_views' in batch:
            current_frames = batch['current_views'][:, 0]  # front view
        
        B = min(num_samples, current_frames.shape[0])
        
        fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(B):
            rgb = current_frames[i].cpu().numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb, 0, 1)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f"Input Frame (Front)")
            axes[i, 0].axis('off')
            
            gt_hm = gt_heatmaps[i].cpu().numpy()
            axes[i, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=1)
            axes[i, 1].set_title(f"GT Heatmap (max={gt_hm.max():.2f})")
            axes[i, 1].axis('off')
            
            pred_hm = pred_heatmaps[i].detach().float().cpu().numpy()
            pred_hm = np.clip(pred_hm, 0, 1)
            axes[i, 2].imshow(pred_hm, cmap='inferno', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Pred Heatmap (max={pred_hm.max():.2f})")
            axes[i, 2].axis('off')
        
        plt.suptitle(f"Epoch {epoch}, Step {step}")
        plt.tight_layout()
        
        # 简洁命名: e001_s00100.png
        save_path = output_dir / f"e{epoch:03d}_s{step:05d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        return None


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
    """滑动窗口/轨迹数据集的 collate 函数"""
    max_K = max(s['history_frames'].shape[0] for s in batch)
    
    history_frames_padded = []
    history_mask = []
    
    for s in batch:
        frames = s['history_frames']
        K = frames.shape[0]
        
        if K < max_K:
            pad_size = max_K - K
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([torch.ones(K), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            mask = torch.ones(K)
        
        history_frames_padded.append(frames_padded)
        history_mask.append(mask)
    
    history_frames = torch.stack(history_frames_padded, dim=0)
    history_mask = torch.stack(history_mask, dim=0)
    current_frame = torch.stack([s['current_frame'] for s in batch], dim=0)
    heatmap = torch.stack([s['heatmap'] for s in batch], dim=0)
    action = torch.stack([s['action'] for s in batch], dim=0)
    action_valid = torch.tensor([s['action_valid'] for s in batch])
    discrete_action = torch.tensor([s.get('discrete_action', 1) for s in batch])
    is_stop = torch.tensor([s.get('is_stop', 0.0) for s in batch])
    text = [s['text'] for s in batch]
    
    result = {
        'history_frames': history_frames,
        'history_mask': history_mask,
        'current_frame': current_frame,
        'heatmap': heatmap,
        'action': action,
        'action_valid': action_valid,
        'discrete_action': discrete_action,
        'is_stop': is_stop,
        'text': text,
    }
    
    # 多视角数据
    if 'current_views' in batch[0]:
        result['current_views'] = torch.stack([s['current_views'] for s in batch], dim=0)
    
    # 水平翻转标记
    if 'is_flipped' in batch[0]:
        result['is_flipped'] = torch.tensor([s.get('is_flipped', False) for s in batch], dtype=torch.bool)
    
    # 轨迹数据集的额外字段
    if 'trajectory' in batch[0]:
        result['trajectory'] = torch.stack([s['trajectory'] for s in batch], dim=0)
        result['trajectory_valid'] = torch.tensor([s.get('trajectory_valid', 0.0) for s in batch])
        result['progress'] = torch.tensor([s.get('progress', 0.0) for s in batch])
    
    return result


# ============================================
# 模型构建
# ============================================

def build_model(cfg: Dict) -> nn.Module:
    """构建 VLN Pipeline"""
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    stop_cfg = model_cfg.get('stop_head', {})
    progress_cfg = model_cfg.get('progress_head', {})
    
    # 确定动作头类型
    action_head_type = action_cfg.get('type', 'transformer')
    
    # 获取 Legacy 和 Transformer 配置
    legacy_action_cfg = action_cfg.get('legacy', {})
    transformer_action_cfg = action_cfg.get('transformer', {})
    
    config = VLNPipelineConfig(
        # Qwen3-VL
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 2048),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'flash_attention_2'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        
        # Sequence Packing (based on official Qwen3-VL fine-tuning)
        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        
        # Device
        device=model_cfg.get('device', 'cuda'),
        
        # Heatmap
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', True),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        image_size=cfg['data']['image_size'][0],
        # Heatmap head ablation settings
        heatmap_use_image_encoder=heatmap_cfg.get('use_image_encoder', True),
        heatmap_pool_method=heatmap_cfg.get('pool_method', 'attention'),
        heatmap_pool_num_heads=heatmap_cfg.get('pool_num_heads', 4),
        # 360° panorama support: circular padding for horizontal edges
        heatmap_use_circular_padding=heatmap_cfg.get('use_circular_padding', False),
        # Regularization
        heatmap_dropout=heatmap_cfg.get('dropout', 0.1),
        # UNet architecture (model capacity) - ~20M params with [128, 256, 512, 512]
        heatmap_block_out_channels=tuple(heatmap_cfg.get('block_out_channels', [64, 128, 256])),
        heatmap_layers_per_block=heatmap_cfg.get('layers_per_block', 2),
        heatmap_attention_levels=tuple(heatmap_cfg.get('attention_levels', [2])),
        heatmap_num_train_timesteps=heatmap_cfg.get('num_train_timesteps', 100),
        # Classifier-Free Guidance (CFG) for heatmap
        heatmap_cfg_drop_prob=heatmap_cfg.get('cfg_drop_prob', 0.1),
        heatmap_cfg_scale=heatmap_cfg.get('cfg_scale', 3.0),
        # Sequence cross-attention conditioning
        heatmap_use_sequence_conditioning=heatmap_cfg.get('use_sequence_conditioning', False),
        heatmap_seq_cross_attn_heads=heatmap_cfg.get('seq_cross_attn_heads', 8),
        heatmap_seq_cross_attn_head_dim=heatmap_cfg.get('seq_cross_attn_head_dim', 64),
        # Visibility head (suppress false positives)
        heatmap_use_visibility_head=heatmap_cfg.get('use_visibility_head', False),
        heatmap_visibility_loss_weight=heatmap_cfg.get('visibility_loss_weight', 0.5),
        heatmap_visibility_threshold=heatmap_cfg.get('visibility_threshold', 0.5),
        # Spatial feature injection
        heatmap_use_spatial_injection=heatmap_cfg.get('use_spatial_injection', False),
        # Image encoder backbone (pretrained ResNet-18 vs lightweight CNN)
        heatmap_image_encoder_use_pretrained=heatmap_cfg.get('image_encoder_use_pretrained', False),
        # Inference sharpening (spatial softmax with temperature)
        heatmap_sharpen_temperature=heatmap_cfg.get('sharpen_temperature', 0.1),
        # x0 reconstruction loss (direct output quality supervision)
        heatmap_x0_loss_weight=heatmap_cfg.get('x0_loss_weight', 1.0),
        # L1 sparsity regularization
        heatmap_sparsity_loss_weight=heatmap_cfg.get('sparsity_loss_weight', 0.5),
        # Negative sample diffusion loss weight
        heatmap_negative_sample_weight=heatmap_cfg.get('negative_sample_weight', 0.3),
        # Peak distance loss (differentiable soft-argmax)
        heatmap_peak_distance_loss_weight=heatmap_cfg.get('peak_distance_loss_weight', 2.0),
        heatmap_peak_loss_temperature=heatmap_cfg.get('peak_loss_temperature', 0.1),
        
        # Multi-layer feature extraction
        multi_layer_features=llm_cfg.get('multi_layer_features', False),
        feature_layer_indices=llm_cfg.get('feature_layer_indices', None),
        feature_fusion_method=llm_cfg.get('feature_fusion_method', 'weighted_sum'),
        
        # LoRA configuration
        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        
        # Action Head Type
        action_head_type=action_head_type,
        enable_action_head=action_cfg.get('enable', True),
        
        # Legacy Action (DiffusionActionHead)
        action_dim=legacy_action_cfg.get('action_dim', 2),
        action_pred_horizon=legacy_action_cfg.get('pred_horizon', 1),
        action_encoding_size=legacy_action_cfg.get('encoding_size', 256),
        action_down_dims=legacy_action_cfg.get('down_dims', None),
        action_num_diffusion_iters=legacy_action_cfg.get('num_diffusion_iters', 10),
        action_stats_min=legacy_action_cfg.get('action_stats_min', [-0.17, -0.03]),
        action_stats_max=legacy_action_cfg.get('action_stats_max', [0.19, 0.31]),
        
        # Transformer Action (TransformerActionHead, InternNav style)
        transformer_action_dim=transformer_action_cfg.get('action_dim', 3),
        transformer_predict_size=transformer_action_cfg.get('predict_size', 24),
        transformer_n_emb=transformer_action_cfg.get('n_emb', 384),
        transformer_n_layer=transformer_action_cfg.get('n_layer', 16),
        transformer_n_head=transformer_action_cfg.get('n_head', 6),  # 对齐 InternNav: 384 // 64 = 6
        transformer_n_cond_layers=transformer_action_cfg.get('n_cond_layers', 4),
        transformer_num_train_timesteps=transformer_action_cfg.get('num_train_timesteps', 20),
        transformer_p_drop_emb=transformer_action_cfg.get('p_drop_emb', 0.1),
        transformer_p_drop_attn=transformer_action_cfg.get('p_drop_attn', 0.1),
        transformer_causal_attn=transformer_action_cfg.get('causal_attn', True),
        
        # Stop (Legacy)
        enable_stop_head=stop_cfg.get('enable', False),
        stop_hidden_dim=stop_cfg.get('hidden_dim', 512),
        stop_focal_gamma=stop_cfg.get('focal_gamma', 3.0),
        stop_focal_alpha=stop_cfg.get('focal_alpha', 0.9),
        
        # Progress (New)
        enable_progress_head=progress_cfg.get('enable', True),
        progress_hidden_dim=progress_cfg.get('hidden_dim', 512),
        
        verbose=True,
    )
    
    model = VLNPipeline(config)
    
    packing_enabled = llm_cfg.get('enable_packing', False)
    print(f"✅ VLN Pipeline 已构建")
    print(f"   Qwen3-VL → {llm_cfg.get('model_path', './models/qwen_3_vl')}")
    if packing_enabled:
        print(f"   SequencePacking → enabled=True, max_seq_length={llm_cfg.get('max_seq_length', 4096)}")
    else:
        print(f"   SequencePacking → enabled=False (使用传统 padding)")
    print(f"   HistoryHeatmapHead → enabled={heatmap_cfg.get('enable_history', True)}, "
          f"use_image_encoder={heatmap_cfg.get('use_image_encoder', True)}, "
          f"pool_method={heatmap_cfg.get('pool_method', 'attention')}")
    print(f"   FutureHeatmapHead → enabled={heatmap_cfg.get('enable_future', True)}")
    print(f"   ActionHead → type={action_head_type}, enabled={action_cfg.get('enable', True)}")
    print(f"   ProgressHead → enabled={progress_cfg.get('enable', True)}")
    print(f"   StopHead (legacy) → enabled={stop_cfg.get('enable', False)}")
    
    return model


def freeze_module(module: nn.Module, freeze: bool = True):
    """冻结/解冻模块"""
    for param in module.parameters():
        param.requires_grad = not freeze


def set_trainable_modules(model: VLNPipeline, stage_cfg: Dict, logger):
    """根据阶段配置设置可训练模块"""
    # 先全部冻结
    freeze_module(model, freeze=True)
    
    trainable = stage_cfg.get('trainable_modules', [])
    
    # History Heatmap Head
    if 'history_heatmap_head' in trainable:
        if hasattr(model, 'history_heatmap_head') and model.history_heatmap_head is not None:
            freeze_module(model.history_heatmap_head, freeze=False)
            logger.info("  ✓ Unfrozen: history_heatmap_head")
            
    # Future Heatmap Head
    if 'future_heatmap_head' in trainable:
        if hasattr(model, 'future_heatmap_head') and model.future_heatmap_head is not None:
            freeze_module(model.future_heatmap_head, freeze=False)
            logger.info("  ✓ Unfrozen: future_heatmap_head")
    
    # Action head (Legacy)
    if 'action_head' in trainable:
        if hasattr(model, 'action_head') and model.action_head is not None:
            freeze_module(model.action_head, freeze=False)
            logger.info("  ✓ Unfrozen: action_head (legacy)")
    
    # Transformer Action Head (New)
    if 'transformer_action_head' in trainable:
        if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
            freeze_module(model.transformer_action_head, freeze=False)
            logger.info("  ✓ Unfrozen: transformer_action_head")
    
    # Stop head (Legacy)
    if 'stop_head' in trainable:
        if hasattr(model, 'stop_head') and model.stop_head is not None:
            freeze_module(model.stop_head, freeze=False)
            logger.info("  ✓ Unfrozen: stop_head")
    
    # Progress Head (New)
    if 'progress_head' in trainable:
        if hasattr(model, 'progress_head') and model.progress_head is not None:
            freeze_module(model.progress_head, freeze=False)
            logger.info("  ✓ Unfrozen: progress_head")
    
    # LLM Projector
    if 'llm_projector' in trainable:
        if hasattr(model, 'llm_projector'):
            freeze_module(model.llm_projector, freeze=False)
            logger.info("  ✓ Unfrozen: llm_projector")
    
    # Multi-layer Fusion module
    if 'multi_layer_fusion' in trainable:
        if hasattr(model, 'multi_layer_fusion') and model.multi_layer_fusion is not None:
            freeze_module(model.multi_layer_fusion, freeze=False)
            logger.info("  ✓ Unfrozen: multi_layer_fusion")
    
    # Qwen3-VL: 冻结基座，但保留 LoRA 参数可训练
    if hasattr(model, 'qwen3_vl') and model.qwen3_vl is not None:
        freeze_module(model.qwen3_vl, freeze=True)
        # 如果配置了 LoRA 且在 trainable_modules 中，解冻 LoRA 参数
        # 兼容 'lora' 和 'qwen3_vl_lora' 两种写法
        if 'lora' in trainable or 'qwen3_vl_lora' in trainable:
            lora_count = 0
            for name, param in model.qwen3_vl.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    lora_count += 1
            if lora_count > 0:
                logger.info(f"  ✓ Unfrozen: qwen3_vl LoRA ({lora_count} parameter tensors)")
            else:
                logger.warning("  ⚠️ LoRA in trainable_modules but no LoRA params found (model loaded?)")


def build_optimizer(model: VLNPipeline, cfg: Dict, stage_cfg: Dict) -> torch.optim.Optimizer:
    """构建分层学习率优化器，支持分组 weight_decay"""
    optim_cfg = cfg['optim']
    param_groups = []
    
    default_wd = optim_cfg.get('weight_decay', 1e-2)
    projector_wd = optim_cfg.get('projector_weight_decay', default_wd)
    
    def get_param_groups_with_wd(module, lr, name, wd):
        """分离需要和不需要 weight_decay 的参数"""
        decay_params = []
        no_decay_params = []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # bias 和 LayerNorm 不使用 weight_decay
            if 'bias' in n or 'norm' in n.lower() or 'ln' in n.lower():
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        
        groups = []
        if decay_params:
            groups.append({
                'params': decay_params,
                'lr': lr,
                'weight_decay': wd,
                'name': f'{name}_decay'
            })
        if no_decay_params:
            groups.append({
                'params': no_decay_params,
                'lr': lr,
                'weight_decay': 0.0,
                'name': f'{name}_no_decay'
            })
        return groups
    
    # History Heatmap Head — split ResNet backbone (low lr) from rest (normal lr)
    hist_lr = optim_cfg.get('history_heatmap_lr', optim_cfg.get('heatmap_lr', 1e-4))
    resnet_backbone_lr = optim_cfg.get('resnet_backbone_lr', 1e-5)
    if hasattr(model, 'history_heatmap_head') and model.history_heatmap_head is not None:
        head = model.history_heatmap_head
        # Check if this head uses a pretrained ResNet encoder
        has_resnet = (hasattr(head, 'condition_encoder') and 
                      hasattr(head.condition_encoder, 'image_encoder') and
                      hasattr(head.condition_encoder.image_encoder, 'img_mean'))  # ResNet marker
        
        if has_resnet:
            resnet_encoder = head.condition_encoder.image_encoder
            resnet_param_ids = set(id(p) for p in resnet_encoder.parameters())
            
            # ResNet backbone params (low lr)
            resnet_groups = get_param_groups_with_wd(
                resnet_encoder, resnet_backbone_lr, 'resnet_backbone', default_wd)
            if resnet_groups:
                param_groups.extend(resnet_groups)
                n_resnet = sum(len(g['params']) for g in resnet_groups)
                print(f"  Param group: resnet_backbone (lr={resnet_backbone_lr}, wd={default_wd}, params={n_resnet})")
            
            # Rest of heatmap head (normal lr) — exclude ResNet params
            rest_decay, rest_no_decay = [], []
            for name, p in head.named_parameters():
                if not p.requires_grad or id(p) in resnet_param_ids:
                    continue
                if p.dim() <= 1 or name.endswith('.bias'):
                    rest_no_decay.append(p)
                else:
                    rest_decay.append(p)
            if rest_decay:
                param_groups.append({'params': rest_decay, 'lr': hist_lr, 
                                     'weight_decay': default_wd, 'name': 'history_heatmap_head'})
            if rest_no_decay:
                param_groups.append({'params': rest_no_decay, 'lr': hist_lr,
                                     'weight_decay': 0.0, 'name': 'history_heatmap_head_no_decay'})
            n_rest = len(rest_decay) + len(rest_no_decay)
            print(f"  Param group: history_heatmap_head (lr={hist_lr}, wd={default_wd}, params={n_rest})")
        else:
            groups = get_param_groups_with_wd(head, hist_lr, 'history_heatmap_head', default_wd)
            if groups:
                param_groups.extend(groups)
                print(f"  Param group: history_heatmap_head (lr={hist_lr}, wd={default_wd})")
    
    # Future Heatmap Head
    fut_lr = optim_cfg.get('future_heatmap_lr', optim_cfg.get('heatmap_lr', 1e-4))
    if hasattr(model, 'future_heatmap_head') and model.future_heatmap_head is not None:
        groups = get_param_groups_with_wd(model.future_heatmap_head, fut_lr, 'future_heatmap_head', default_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: future_heatmap_head (lr={fut_lr}, wd={default_wd})")
    
    # Action Head (Legacy)
    action_lr = optim_cfg.get('action_lr', 1e-4)
    if hasattr(model, 'action_head') and model.action_head is not None:
        groups = get_param_groups_with_wd(model.action_head, action_lr, 'action_head', default_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: action_head (lr={action_lr}, wd={default_wd})")
    
    # Transformer Action Head (New)
    transformer_action_lr = optim_cfg.get('transformer_action_lr', action_lr)
    if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
        groups = get_param_groups_with_wd(model.transformer_action_head, transformer_action_lr, 'transformer_action_head', default_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: transformer_action_head (lr={transformer_action_lr}, wd={default_wd})")
    
    # Stop Head (Legacy)
    stop_lr = optim_cfg.get('stop_lr', action_lr)
    if hasattr(model, 'stop_head') and model.stop_head is not None:
        groups = get_param_groups_with_wd(model.stop_head, stop_lr, 'stop_head', default_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: stop_head (lr={stop_lr}, wd={default_wd})")
    
    # Progress Head (New)
    progress_lr = optim_cfg.get('progress_lr', action_lr)
    if hasattr(model, 'progress_head') and model.progress_head is not None:
        groups = get_param_groups_with_wd(model.progress_head, progress_lr, 'progress_head', default_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: progress_head (lr={progress_lr}, wd={default_wd})")
    
    # LLM Projector (使用更小的 weight_decay)
    proj_lr = optim_cfg.get('llm_projector_lr', 3e-5)
    if hasattr(model, 'llm_projector'):
        groups = get_param_groups_with_wd(model.llm_projector, proj_lr, 'llm_projector', projector_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: llm_projector (lr={proj_lr}, wd={projector_wd})")
    
    # Multi-layer Fusion module
    fusion_lr = optim_cfg.get('multi_layer_fusion_lr', 1e-4)
    if hasattr(model, 'multi_layer_fusion') and model.multi_layer_fusion is not None:
        groups = get_param_groups_with_wd(model.multi_layer_fusion, fusion_lr, 'multi_layer_fusion', projector_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: multi_layer_fusion (lr={fusion_lr}, wd={projector_wd})")
    
    # Qwen3-VL LoRA parameters (very low LR for backbone fine-tuning)
    lora_lr = optim_cfg.get('lora_lr', 1e-5)
    if hasattr(model, 'qwen3_vl') and model.qwen3_vl is not None:
        lora_params = [p for n, p in model.qwen3_vl.named_parameters() 
                       if p.requires_grad and 'lora_' in n]
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'weight_decay': 0.0,  # LoRA 通常不使用 weight_decay
                'name': 'qwen3_vl_lora'
            })
            print(f"  Param group: qwen3_vl_lora (lr={lora_lr}, wd=0.0, params={len(lora_params)})")
    
    if not param_groups:
        raise ValueError("No trainable parameters found!")
    
    # 创建优化器（不在这里设置 weight_decay，因为已经在参数组中设置）
    optimizer = torch.optim.AdamW(param_groups)
    
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
        T_max=max(1, total_steps - warmup_steps),
        eta_min=optim_cfg.get('min_lr', 1e-6)  # 最小学习率，避免降到 0
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
    model: VLNPipeline,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: Dict,
    epoch: int,
    logger,
    tb_writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    stage_idx: int = 0,
    stage_name: str = "",
    stage_cfg: Dict = None,
    max_batches: int = None,
    packing_enabled: bool = False,
    vis_dir: Optional[Path] = None,
    gpu_heatmap_computer: Optional[GPUHeatmapComputer] = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    ema: Optional[EMAModel] = None,
) -> Dict[str, float]:
    """训练一个 epoch"""
    
    model.train()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_diffusion_loss = 0.0      # 纯扩散 loss（不含 visibility）
    total_visibility_loss = 0.0     # visibility head BCE loss
    total_action_loss = 0.0
    total_stop_loss = 0.0
    num_batches = 0
    
    optim_cfg = cfg['optim']
    loss_cfg = cfg['loss']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)
    
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    
    device = torch.device(cfg['model'].get('device', 'cuda'))
    
    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式: 只处理 {total_batches} batches")
    
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{stage_cfg['epochs']}",
        total=total_batches,
        ncols=cfg['log'].get('tqdm_ncols', 120)
    )
    
    global_step = 0
    valid_batch_count = 0
    
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        
        # 准备数据 - Packing 模式和传统模式的 batch 结构不同
        if packing_enabled:
            # Packing 模式: batch 来自 FlattenedCollatorForVLN
            # 没有 history_frames，直接使用 packed batch
            B = batch['num_samples']
            current_frame = batch['current_frame']
        else:
            # 传统模式: batch 来自普通 collate_fn
            history_frames = batch['history_frames']
            current_frame = batch['current_frame']
            B, K, C, H, W = history_frames.shape
        
        gt_action = batch['action'].to(device)
        action_valid = batch['action_valid'].to(device)
        is_stop = batch['is_stop'].to(device)
        text = batch['text']
        
        # GPU 热力图计算（如果启用）
        if gpu_heatmap_computer is not None and 'history_poses' in batch:
            history_poses = batch['history_poses'].to(device)  # [B, K, 4, 4]
            current_poses = batch['current_pose'].to(device)   # [B, 4, 4]
            
            current_depths = batch['current_depth'].to(device) if gpu_has_depth and 'current_depth' in batch else None
            intrinsics = batch['intrinsics'].to(device) if 'intrinsics' in batch else None
            
            gt_heatmap = gpu_heatmap_computer.compute_batch(
                history_poses=history_poses,
                current_poses=current_poses,
                current_depths=current_depths,
                intrinsics=intrinsics,
                depth_normalized=gpu_depth_normalized,
            )  # [B, Hm, Wm]
            
            # 水平翻转增强：对翻转的样本也翻转 GT 热力图
            if 'is_flipped' in batch:
                flip_mask = batch['is_flipped']  # [B] bool tensor
                if flip_mask.any():
                    for b_idx in range(gt_heatmap.shape[0]):
                        if flip_mask[b_idx]:
                            gt_heatmap[b_idx] = gt_heatmap[b_idx].flip(dims=[-1])
        else:
            gt_heatmap = batch['heatmap'].to(device)
        
        # 前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if packing_enabled:
                # Packing 模式: 直接传递 packed batch
                output = model.forward_packed(
                    packed_batch=batch,
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                    gt_future_heatmap=gt_heatmap if train_future else None,
                )
            else:
                # 传统模式: 构建 video_frames
                video_frames = torch.cat([
                    history_frames,
                    current_frame.unsqueeze(1)
                ], dim=1)
                
                # 处理导航指令
                if text and len(text) > 0:
                    instruction_text = list(text)
                else:
                    instruction_text = None
                
                output = model(
                    video_frames=video_frames,
                    instruction_text=instruction_text,
                    current_observation=current_frame.to(device),
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                    gt_future_heatmap=gt_heatmap if train_future else None,
                )
            
            # Heatmap Loss
            heatmap_loss = torch.tensor(0.0, device=device)
            loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')
            
            if train_history and 'history_heatmap_loss' in output:
                heatmap_loss = output['history_heatmap_loss']
            
            if train_future and 'future_heatmap_loss' in output:
                heatmap_loss = heatmap_loss + output['future_heatmap_loss']
            
            # Action Loss / Trajectory Loss
            action_loss = torch.tensor(0.0, device=device)
            trajectory_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Transformer Action Head (new) - 使用 trajectory
                if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
                    if 'trajectory' in batch:
                        gt_trajectory = batch['trajectory'].to(device)
                        trajectory_valid = batch['trajectory_valid'].to(device)
                        # 传入完整 llm_tokens 序列
                        traj_result = model.transformer_action_head.compute_loss(
                            output['llm_tokens'],
                            gt_trajectory,
                            trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                # Legacy Action Head - 使用单步动作
                elif hasattr(model, 'action_head') and model.action_head is not None and 'action_cond' in output:
                    action_result = model.action_head.compute_loss(
                        output['action_cond'], 
                        gt_action.unsqueeze(1),
                        action_valid
                    )
                    action_loss = action_result['loss']
            
            # Progress Loss / Stop Loss
            stop_loss = torch.tensor(0.0, device=device)
            progress_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Progress Head (new)
                # 修复：传入完整 llm_tokens 序列，确保 concat_state_txt 中
                # last_token 和 mean_pool 是不同的表示（对齐 InternNav）
                if hasattr(model, 'progress_head') and model.progress_head is not None:
                    if 'progress' in batch:
                        gt_progress = batch['progress'].to(device)
                        # 使用 trajectory_valid 作为 mask（更准确）或 action_valid 作为备选
                        progress_valid = batch.get('trajectory_valid', action_valid).to(device)
                        progress_result = model.progress_head(
                            output['llm_tokens'],  # 传入完整序列 (B, seq_len, D)
                            gt_progress=gt_progress,
                            action_valid=progress_valid,
                            return_loss=True,
                        )
                        progress_loss = progress_result['loss']
                # Legacy Stop Head
                elif hasattr(model, 'stop_head') and model.stop_head is not None and 'stop_logits' in output:
                    stop_loss = model.stop_head.compute_loss(
                        output['stop_logits'],
                        is_stop,
                        action_valid
                    )
            
            # 总损失
            heatmap_weight = loss_cfg.get('history_weight', 1.0) if train_history else loss_cfg.get('future_weight', 1.0)
            action_weight = loss_cfg.get('action_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 1.0)
            stop_weight = loss_cfg.get('stop_weight', 0.5)
            progress_weight = loss_cfg.get('progress_weight', 0.5)
            
            # 使用 trajectory_loss 或 action_loss（根据哪个有效）
            action_total_loss = trajectory_loss if trajectory_loss.item() > 0 else action_loss
            stop_total_loss = progress_loss if progress_loss.item() > 0 else stop_loss
            
            loss = heatmap_weight * heatmap_loss + trajectory_weight * action_total_loss + progress_weight * stop_total_loss
            loss = loss / grad_accum_steps
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        valid_batch_count += 1
        
        # 梯度累积
        if valid_batch_count % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update()
            global_step += 1
            
            # 日志
            log_interval = cfg['log'].get('log_interval', 10)
            if global_step % log_interval == 0 or global_step <= 3:
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(
                    f"[{stage_name}] "
                    f"Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item()*grad_accum_steps:.4f} "
                    f"(hm: {heatmap_loss.item():.4f}, traj: {trajectory_loss.item():.4f}, prog: {progress_loss.item():.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"GPU: {mem_alloc:.1f}GB"
                )
                
            if tb_writer is not None:
                actual_step = global_step_offset + global_step
                tb_writer.add_scalar('train/loss', loss.item()*grad_accum_steps, actual_step)
                tb_writer.add_scalar('train/heatmap_loss', heatmap_loss.item(), actual_step)
                tb_writer.add_scalar('train/trajectory_loss', trajectory_loss.item(), actual_step)
                tb_writer.add_scalar('train/progress_loss', progress_loss.item(), actual_step)
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], actual_step)
                
                # 🔧 修复：优先使用 trajectory_valid（trajectory 数据集），否则使用 action_valid
                # 监控有效样本比例
                if 'trajectory_valid' in batch:
                    valid_ratio = batch['trajectory_valid'].float().mean().item()
                else:
                    valid_ratio = action_valid.float().mean().item()
                tb_writer.add_scalar('train/action_valid_ratio', valid_ratio, actual_step)
                
                # 诊断信息记录（固定间隔）
                diag_interval = cfg['log'].get('diag_interval', 100)
                if global_step % diag_interval == 0:
                    # 多层融合权重诊断
                    if hasattr(model, 'multi_layer_fusion') and model.multi_layer_fusion is not None:
                        fusion_weights = model.multi_layer_fusion.get_layer_weights()
                        layer_indices = cfg['model']['llm'].get('feature_layer_indices', [])
                        for i, (li, w) in enumerate(zip(layer_indices, fusion_weights)):
                            tb_writer.add_scalar(f'diag/fusion_weight_layer{li}', w, actual_step)
                        logger.info(f"[DIAG-FUSION] layer_weights: {dict(zip(layer_indices, [f'{w:.3f}' for w in fusion_weights]))}")
                    
                    # 热力图输出诊断 - 检查是否坍缩为全黑
                    if 'history_heatmaps' in output and output['history_heatmaps'] is not None:
                        pred_hm = output['history_heatmaps'].detach()
                        
                        # 多视角 [B, 4, H, W]: 取 front 视角；单视角 [B, 1, H, W]: 不变
                        if pred_hm.dim() == 4 and pred_hm.shape[1] == 4:
                            pred_hm = pred_hm[:, 0].unsqueeze(1)  # [B, 1, H, W] front view
                        gt_hm_for_diag = gt_heatmap
                        if gt_hm_for_diag.dim() == 4 and gt_hm_for_diag.shape[1] == 4:  # [B, 4, H, W]
                            gt_hm_for_diag = gt_hm_for_diag[:, 0]  # [B, H, W] front view
                        
                        pred_mean = pred_hm.mean().item()
                        pred_max = pred_hm.max().item()
                        pred_std = pred_hm.std().item()
                        
                        tb_writer.add_scalar('diag/pred_heatmap_mean', pred_mean, actual_step)
                        tb_writer.add_scalar('diag/pred_heatmap_max', pred_max, actual_step)
                        tb_writer.add_scalar('diag/pred_heatmap_std', pred_std, actual_step)
                        
                        gt_mean = gt_hm_for_diag.mean().item()
                        gt_max = gt_hm_for_diag.max().item()
                        
                        logger.info(f"[DIAG-HM] pred: mean={pred_mean:.4f}, max={pred_max:.4f}, std={pred_std:.4f}")
                        logger.info(f"[DIAG-HM] gt:   mean={gt_mean:.4f}, max={gt_max:.4f}")
                        
                        if pred_max < 0.1:
                            logger.warning(f"[DIAG-HM] ⚠️ 热力图输出疑似坍缩！pred_max={pred_max:.4f} < 0.1")
                        
                        non_zero_ratio = (pred_hm > 0.01).float().mean().item()
                        tb_writer.add_scalar('diag/pred_heatmap_nonzero_ratio', non_zero_ratio, actual_step)
                        if non_zero_ratio < 0.05:
                            logger.warning(f"[DIAG-HM] ⚠️ 热力图几乎全黑！non_zero_ratio={non_zero_ratio*100:.2f}%")
                        
                        # ==================== 热力图质量指标 ====================
                        B, C, H, W = pred_hm.shape
                        gt_hm_diag = gt_hm_for_diag.to(pred_hm.device)
                        if gt_hm_diag.dim() == 3:
                            gt_hm_diag = gt_hm_diag.unsqueeze(1)
                        
                        # ==================== 热力图质量指标 ====================
                        # 1. Peak 位置误差 (像素距离)
                        # 找到 pred 和 gt 的最大值位置
                        pred_flat = pred_hm.view(B, -1)
                        gt_flat = gt_hm_diag.view(B, -1)
                        
                        pred_peak_idx = pred_flat.argmax(dim=1)  # (B,)
                        gt_peak_idx = gt_flat.argmax(dim=1)      # (B,)
                        
                        # 转换为 (y, x) 坐标
                        pred_peak_y = (pred_peak_idx // W).float()
                        pred_peak_x = (pred_peak_idx % W).float()
                        gt_peak_y = (gt_peak_idx // W).float()
                        gt_peak_x = (gt_peak_idx % W).float()
                        
                        # 考虑全景图的环形连续性计算 x 距离
                        dx = torch.abs(pred_peak_x - gt_peak_x)
                        dx = torch.min(dx, W - dx)  # 取环形最短距离
                        dy = torch.abs(pred_peak_y - gt_peak_y)
                        
                        peak_distance = torch.sqrt(dx**2 + dy**2).mean().item()
                        tb_writer.add_scalar('diag/hm_peak_distance', peak_distance, actual_step)
                        tb_writer.add_scalar('diag/hm_peak_dx', dx.mean().item(), actual_step)
                        tb_writer.add_scalar('diag/hm_peak_dy', dy.mean().item(), actual_step)
                        
                        # 2. Peak IoU (交并比) - 使用阈值化后的区域计算
                        # 阈值取 max 的 50%
                        pred_threshold = pred_hm.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] * 0.5
                        gt_threshold = gt_hm_diag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] * 0.5
                        
                        pred_mask = (pred_hm > pred_threshold).float()
                        gt_mask = (gt_hm_diag > gt_threshold).float()
                        
                        intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
                        union = ((pred_mask + gt_mask) > 0).float().sum(dim=(1, 2, 3))
                        
                        # 避免除零
                        iou = (intersection / (union + 1e-6)).mean().item()
                        tb_writer.add_scalar('diag/hm_peak_iou', iou, actual_step)
                        
                        # 3. 峰值置信度对比
                        pred_peak_conf = pred_hm.max(dim=-1)[0].max(dim=-1)[0].mean().item()  # pred 峰值
                        gt_peak_conf = gt_hm_diag.max(dim=-1)[0].max(dim=-1)[0].mean().item()  # gt 峰值
                        
                        tb_writer.add_scalar('diag/hm_pred_peak_conf', pred_peak_conf, actual_step)
                        tb_writer.add_scalar('diag/hm_gt_peak_conf', gt_peak_conf, actual_step)
                        
                        # 置信度比值 (越接近 1 越好)
                        if gt_peak_conf > 0:
                            conf_ratio = pred_peak_conf / gt_peak_conf
                            tb_writer.add_scalar('diag/hm_peak_conf_ratio', conf_ratio, actual_step)
                        
                        # 4. 多峰评估指标 —— 检测 GT 中所有峰并逐一匹配
                        # 弥补 global argmax 只看最强峰的缺陷
                        try:
                            # NMS 检测 GT 峰值
                            nms_kernel = 5
                            pad = nms_kernel // 2
                            gt_padded = F.pad(gt_hm_diag, [pad] * 4, mode='replicate')
                            local_max = F.max_pool2d(gt_padded, kernel_size=nms_kernel, stride=1, padding=0)
                            is_gt_peak = (gt_hm_diag == local_max) & (gt_hm_diag > 0.1)
                            
                            # 同样检测 pred 峰值
                            pred_padded = F.pad(pred_hm, [pad] * 4, mode='replicate')
                            pred_local_max = F.max_pool2d(pred_padded, kernel_size=nms_kernel, stride=1, padding=0)
                            is_pred_peak = (pred_hm == pred_local_max) & (pred_hm > pred_hm.max() * 0.2)
                            
                            total_gt_peaks = 0
                            total_matched = 0
                            total_multi_dist = 0.0
                            multi_peak_count = 0
                            
                            for bi in range(B):
                                gt_peaks_bi = is_gt_peak[bi, 0].nonzero(as_tuple=False)  # (N_gt, 2) yx
                                pred_peaks_bi = is_pred_peak[bi, 0].nonzero(as_tuple=False)  # (N_pred, 2)
                                
                                n_gt = len(gt_peaks_bi)
                                total_gt_peaks += n_gt
                                
                                if n_gt == 0 or len(pred_peaks_bi) == 0:
                                    continue
                                
                                # 贪心匹配：对每个 GT 峰找最近的 pred 峰
                                for gi in range(min(n_gt, 8)):  # 最多 8 个峰
                                    gt_y, gt_x = gt_peaks_bi[gi].float()
                                    
                                    # 计算到所有 pred 峰的距离（考虑全景环形）
                                    pred_y = pred_peaks_bi[:, 0].float()
                                    pred_x = pred_peaks_bi[:, 1].float()
                                    dx_mp = torch.abs(pred_x - gt_x)
                                    dx_mp = torch.min(dx_mp, W - dx_mp)
                                    dy_mp = torch.abs(pred_y - gt_y)
                                    dists = torch.sqrt(dx_mp**2 + dy_mp**2)
                                    
                                    min_dist = dists.min().item()
                                    total_multi_dist += min_dist
                                    multi_peak_count += 1
                                    
                                    if min_dist < 5.0:  # 5px 内算匹配成功
                                        total_matched += 1
                            
                            if multi_peak_count > 0:
                                avg_multi_peak_dist = total_multi_dist / multi_peak_count
                                tb_writer.add_scalar('diag/hm_multi_peak_distance', avg_multi_peak_dist, actual_step)
                            
                            if total_gt_peaks > 0:
                                peak_recall = total_matched / total_gt_peaks
                                tb_writer.add_scalar('diag/hm_peak_recall_5px', peak_recall, actual_step)
                            
                            # 记录平均 GT 峰数量
                            tb_writer.add_scalar('diag/hm_avg_gt_peaks', total_gt_peaks / B, actual_step)
                            
                        except Exception as e:
                            logger.debug(f"Multi-peak eval error (non-critical): {e}")
                    
                    # Focal Loss 诊断
                    if 'history_heatmap_base_loss' in output and output['history_heatmap_base_loss'] is not None:
                        base_loss_val = output['history_heatmap_base_loss']
                        focal_loss_val = output['history_heatmap_focal_loss']
                        tb_writer.add_scalar('diag/heatmap_base_loss', base_loss_val, actual_step)
                        tb_writer.add_scalar('diag/heatmap_focal_loss', focal_loss_val, actual_step)
                        # 比值：如果 focal_loss > base_loss，说明峰值区域预测更差
                        if base_loss_val > 0:
                            focal_ratio = focal_loss_val / base_loss_val
                            tb_writer.add_scalar('diag/heatmap_focal_ratio', focal_ratio, actual_step)
                    
                    # Visibility Loss 诊断
                    if 'history_heatmap_visibility_loss' in output and output['history_heatmap_visibility_loss'] is not None:
                        vis_loss_val = output['history_heatmap_visibility_loss']
                        tb_writer.add_scalar('diag/heatmap_visibility_loss', vis_loss_val, actual_step)
                    
                    # x0 Reconstruction Loss 诊断
                    if 'history_heatmap_x0_loss' in output and output['history_heatmap_x0_loss'] is not None:
                        tb_writer.add_scalar('diag/heatmap_x0_loss', output['history_heatmap_x0_loss'], actual_step)
                    
                    # Sparsity Loss 诊断
                    if 'history_heatmap_sparsity_loss' in output and output['history_heatmap_sparsity_loss'] is not None:
                        tb_writer.add_scalar('diag/heatmap_sparsity_loss', output['history_heatmap_sparsity_loss'], actual_step)
                    
                    # Negative Zero Loss 诊断
                    if 'history_heatmap_neg_zero_loss' in output and output['history_heatmap_neg_zero_loss'] is not None:
                        tb_writer.add_scalar('diag/heatmap_neg_zero_loss', output['history_heatmap_neg_zero_loss'], actual_step)
                    
                    # Peak Distance Loss 诊断
                    if 'history_heatmap_peak_dist_loss' in output and output['history_heatmap_peak_dist_loss'] is not None:
                        tb_writer.add_scalar('diag/heatmap_peak_dist_loss', output['history_heatmap_peak_dist_loss'], actual_step)
                    
                    # Progress prediction 诊断
                    if 'progress' in output and output['progress'] is not None:
                        pred_progress = output['progress'].detach()
                        gt_progress = batch.get('progress')
                        if gt_progress is not None:
                            gt_progress = gt_progress.to(pred_progress.device)
                            progress_mae = (pred_progress - gt_progress).abs().mean().item()
                            tb_writer.add_scalar('diag/progress_mae', progress_mae, actual_step)
                            tb_writer.add_scalar('diag/progress_pred_mean', pred_progress.mean().item(), actual_step)
                            tb_writer.add_scalar('diag/progress_gt_mean', gt_progress.mean().item(), actual_step)
                            # 边界检测准确率 (progress < 0.1 或 > 0.9)
                            boundary_mask = (gt_progress < 0.1) | (gt_progress > 0.9)
                            if boundary_mask.sum() > 0:
                                boundary_error = (pred_progress[boundary_mask] - gt_progress[boundary_mask]).abs().mean().item()
                                tb_writer.add_scalar('diag/progress_boundary_error', boundary_error, actual_step)
                    
                    # 轨迹预测诊断
                    if 'trajectory' in output and output['trajectory'] is not None:
                        pred_traj = output['trajectory'].detach()
                        gt_traj = batch.get('trajectory')
                        if gt_traj is not None:
                            gt_traj = gt_traj.to(pred_traj.device)
                            # ADE (Average Displacement Error)
                            displacement = torch.sqrt(((pred_traj[..., :2] - gt_traj[..., :2]) ** 2).sum(dim=-1))
                            ade = displacement.mean().item()
                            tb_writer.add_scalar('diag/trajectory_ade', ade, actual_step)
                            # FDE (Final Displacement Error)
                            fde = displacement[:, -1].mean().item()
                            tb_writer.add_scalar('diag/trajectory_fde', fde, actual_step)
                    
                    # GPU 显存监控
                    tb_writer.add_scalar('diag/gpu_memory_gb', torch.cuda.memory_allocated(0) / 1024**3, actual_step)
                    tb_writer.add_scalar('diag/gpu_memory_reserved_gb', torch.cuda.memory_reserved(0) / 1024**3, actual_step)
                
                # 轨迹分布直方图（每 100 步记录一次，避免日志过大）
                if global_step % 100 == 0:
                    if 'trajectory' in output and output['trajectory'] is not None:
                        tb_writer.add_histogram('train/pred_trajectory_dx', output['trajectory'][..., 0].flatten().cpu(), actual_step)
                        tb_writer.add_histogram('train/pred_trajectory_dy', output['trajectory'][..., 1].flatten().cpu(), actual_step)
                    if 'trajectory' in batch and batch['trajectory'] is not None:
                        tb_writer.add_histogram('train/gt_trajectory_dx', batch['trajectory'][..., 0].flatten().cpu(), actual_step)
                        tb_writer.add_histogram('train/gt_trajectory_dy', batch['trajectory'][..., 1].flatten().cpu(), actual_step)
        
        # 定期可视化热力图预测并记录到 TensorBoard
        vis_interval = cfg['log'].get('vis_every_steps', 500)
        if tb_writer is not None and global_step % vis_interval == 0 and global_step > 0:
            vis_path = visualize_heatmap_predictions(
                model=model,
                batch=batch,
                output=output,
                epoch=epoch,
                step=global_step,
                output_dir=vis_dir if vis_dir else Path('.'),
                num_samples=2,
                gt_heatmap_override=gt_heatmap if gpu_heatmap_computer is not None else None,
            )
            if vis_path:
                try:
                    vis_img = cv2.imread(str(vis_path))
                    if vis_img is not None:
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        vis_img = vis_img.transpose(2, 0, 1)  # HWC -> CHW
                        tb_writer.add_image('train/heatmap_viz', vis_img, global_step_offset + global_step)
                except Exception as e:
                    pass  # 忽略可视化写入错误
        
        # 拆分记录：扩散 loss 和 visibility loss（必须在 del output 之前）
        total_loss += loss.item() * grad_accum_steps
        total_heatmap_loss += heatmap_loss.item()
        if 'history_heatmap_visibility_loss' in output and output['history_heatmap_visibility_loss'] is not None:
            total_visibility_loss += output['history_heatmap_visibility_loss']
            vis_weight = cfg['model']['heatmap_head'].get('visibility_loss_weight', 0.5)
            total_diffusion_loss += heatmap_loss.item() - vis_weight * output['history_heatmap_visibility_loss']
        else:
            total_diffusion_loss += heatmap_loss.item()
        total_action_loss += action_total_loss.item()
        total_stop_loss += stop_total_loss.item()
        num_batches += 1
        
        del output
        
        # 🔧 显式释放 batch 引用（尤其是 packing 模式下 pixel_values_videos 很大）
        if (i + 1) % 4 == 0:
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            _malloc_trim()
        
        # 进度条显示拆分的 loss：diff=扩散, vis=可见性
        diff_loss_display = total_diffusion_loss / num_batches
        vis_loss_display = total_visibility_loss / num_batches
        pbar.set_postfix({
            'loss': f"{loss.item()*grad_accum_steps:.4f}",
            'diff': f"{diff_loss_display:.4f}",
            'vis': f"{vis_loss_display:.4f}",
            'traj': f"{action_total_loss.item():.4f}",
        })
    
    # 处理剩余梯度
    remaining = valid_batch_count % grad_accum_steps
    if remaining > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if ema is not None:
            ema.update()
    
    return {
        'total_loss': total_loss / max(num_batches, 1),
        'heatmap_loss': total_heatmap_loss / max(num_batches, 1),
        'diffusion_loss': total_diffusion_loss / max(num_batches, 1),       # 纯扩散 loss
        'visibility_loss': total_visibility_loss / max(num_batches, 1),     # visibility BCE
        'action_loss': total_action_loss / max(num_batches, 1),
        'stop_loss': total_stop_loss / max(num_batches, 1),
    }


@torch.inference_mode()
def validate(
    model: VLNPipeline,
    val_loader: DataLoader,
    cfg: Dict,
    logger,
    stage_cfg: Dict,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    packing_enabled: bool = False,
    vis_dir: Optional[Path] = None,
    max_batches: int = None,
    gpu_heatmap_computer: Optional[GPUHeatmapComputer] = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
) -> Dict[str, float]:
    """验证（带可视化）"""
    model.eval()
    
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_diffusion_loss = 0.0      # 纯扩散 loss（不含 visibility）
    total_visibility_loss = 0.0     # visibility head BCE loss
    total_action_loss = 0.0
    total_stop_loss = 0.0
    total_heatmap_mse = 0.0      # 完整推理的 heatmap MSE（采样）
    num_heatmap_mse_batches = 0  # MSE 采样 batch 计数
    num_batches = 0
    
    loss_cfg = cfg['loss']
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')
    
    device = torch.device(cfg['model'].get('device', 'cuda'))
    
    # 验证推理 batch 数限制：只对前 N 个 batch 做完整扩散推理计算 heatmap MSE
    val_inference_batches = cfg.get('validation', {}).get('val_inference_batches', 10)
    
    total_val_batches = len(val_loader)
    if max_batches is not None:
        total_val_batches = min(total_val_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式(验证): 只处理 {total_val_batches} batches")
    
    logger.info(f"  📊 验证: {total_val_batches} batches (噪声预测 loss), "
                f"{val_inference_batches} batches (完整推理 MSE)")
    
    for i, batch in enumerate(tqdm(val_loader, desc="Validating", total=total_val_batches)):
        if max_batches is not None and i >= max_batches:
            break
        # Packing 模式和传统模式的 batch 结构不同
        if packing_enabled:
            B = batch['num_samples']
            current_frame = batch['current_frame']
        else:
            history_frames = batch['history_frames']
            current_frame = batch['current_frame']
            B, K, C, H, W = history_frames.shape
        
        gt_action = batch['action'].to(device)
        action_valid = batch['action_valid'].to(device)
        is_stop = batch['is_stop'].to(device)
        text = batch['text']
        
        # GPU 热力图计算（如果启用）
        if gpu_heatmap_computer is not None and 'history_poses' in batch:
            history_poses = batch['history_poses'].to(device)
            current_poses = batch['current_pose'].to(device)
            
            current_depths = batch['current_depth'].to(device) if gpu_has_depth and 'current_depth' in batch else None
            intrinsics = batch['intrinsics'].to(device) if 'intrinsics' in batch else None
            
            gt_heatmap = gpu_heatmap_computer.compute_batch(
                history_poses=history_poses,
                current_poses=current_poses,
                current_depths=current_depths,
                intrinsics=intrinsics,
                depth_normalized=gpu_depth_normalized,
            )
        else:
            gt_heatmap = batch['heatmap'].to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if packing_enabled:
                # Packing 模式 — 传入 GT 热力图用噪声预测 loss（快速，1 次 UNet）
                output = model.forward_packed(
                    packed_batch=batch,
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                )
            else:
                # 传统模式
                video_frames = torch.cat([
                    history_frames,
                    current_frame.unsqueeze(1)
                ], dim=1)
                
                if text and len(text) > 0:
                    instruction_text = list(text)
                else:
                    instruction_text = None
                
                output = model(
                    video_frames=video_frames,
                    instruction_text=instruction_text,
                    current_observation=current_frame.to(device),
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                )
            
            # 使用噪声预测 loss（与训练一致，且速度快 ~40x）
            heatmap_loss = torch.tensor(0.0, device=device)
            if train_history and 'history_heatmap_loss' in output:
                heatmap_loss = output['history_heatmap_loss']
            
            # Action Loss / Trajectory Loss (验证)
            action_loss = torch.tensor(0.0, device=device)
            trajectory_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Transformer Action Head (new) - 使用 trajectory
                if hasattr(model, 'transformer_action_head') and model.transformer_action_head is not None:
                    if 'trajectory' in batch:
                        gt_trajectory = batch['trajectory'].to(device)
                        trajectory_valid = batch['trajectory_valid'].to(device)
                        # 传入完整 llm_tokens 序列
                        traj_result = model.transformer_action_head.compute_loss(
                            output['llm_tokens'],
                            gt_trajectory,
                            trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                # Legacy Action Head
                elif hasattr(model, 'action_head') and model.action_head is not None and 'action_cond' in output:
                    action_result = model.action_head.compute_loss(
                        output['action_cond'],
                        gt_action.unsqueeze(1),
                        action_valid
                    )
                    action_loss = action_result['loss']
            
            # Progress Loss / Stop Loss (验证)
            stop_loss = torch.tensor(0.0, device=device)
            progress_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Progress Head (new)
                # 修复：传入完整 llm_tokens 序列（对齐 InternNav）
                if hasattr(model, 'progress_head') and model.progress_head is not None:
                    if 'progress' in batch:
                        gt_progress = batch['progress'].to(device)
                        # 使用 trajectory_valid 作为 mask（更准确）或 action_valid 作为备选
                        progress_valid = batch.get('trajectory_valid', action_valid).to(device)
                        progress_result = model.progress_head(
                            output['llm_tokens'],  # 传入完整序列 (B, seq_len, D)
                            gt_progress=gt_progress,
                            action_valid=progress_valid,
                            return_loss=True,
                        )
                        progress_loss = progress_result['loss']
                # Legacy Stop Head
                elif hasattr(model, 'stop_head') and model.stop_head is not None and 'stop_logits' in output:
                    stop_loss = model.stop_head.compute_loss(
                        output['stop_logits'],
                        is_stop,
                        action_valid
                    )
            
            heatmap_weight = loss_cfg.get('history_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 1.0)
            progress_weight = loss_cfg.get('progress_weight', 0.5)
            
            action_total_loss = trajectory_loss if trajectory_loss.item() > 0 else action_loss
            stop_total_loss = progress_loss if progress_loss.item() > 0 else stop_loss
            
            loss = heatmap_weight * heatmap_loss + trajectory_weight * action_total_loss + progress_weight * stop_total_loss
        
        total_loss += loss.item()
        total_heatmap_loss += heatmap_loss.item()
        # 拆分记录：扩散 loss 和 visibility loss
        if 'history_heatmap_visibility_loss' in output and output['history_heatmap_visibility_loss'] is not None:
            total_visibility_loss += output['history_heatmap_visibility_loss']
            vis_weight = cfg['model']['heatmap_head'].get('visibility_loss_weight', 0.5)
            total_diffusion_loss += heatmap_loss.item() - vis_weight * output['history_heatmap_visibility_loss']
        else:
            total_diffusion_loss += heatmap_loss.item()
        total_action_loss += action_total_loss.item()  # trajectory or action
        total_stop_loss += stop_total_loss.item()      # progress or stop
        num_batches += 1
        
        # ==================== 完整推理 + 可视化（前几个 batch）====================
        # 对前 val_inference_batches 个 batch 做完整扩散推理，计算真实 heatmap MSE
        # 其中前 val_vis_batches 个 batch 额外保存可视化图片
        num_vis_batches = cfg['log'].get('val_vis_batches', 2)
        if num_batches <= val_inference_batches:
            try:
                # 使用纯推理模式生成热力图（不传 gt_heatmap，触发完整扩散推理）
                if packing_enabled:
                    vis_output = model.forward_packed(
                        packed_batch=batch,
                        return_heatmaps=True,
                        return_actions=False,
                    )
                else:
                    video_frames = torch.cat([
                        history_frames,
                        current_frame.unsqueeze(1)
                    ], dim=1)
                    vis_output = model(
                        video_frames=video_frames,
                        instruction_text=list(text) if text else None,
                        current_observation=current_frame.to(device),
                        return_heatmaps=True,
                        return_actions=False,
                    )
                
                # 计算完整推理的 heatmap MSE（真实生成质量指标）
                if train_history and 'history_heatmaps' in vis_output:
                    infer_pred_hm = vis_output['history_heatmaps']
                    # 多视角 [B, 4, H, W]: 取 front；单视角 [B, 1, H, W]: 取唯一通道
                    if infer_pred_hm.dim() == 4 and infer_pred_hm.shape[1] == 4:
                        infer_pred_hm = infer_pred_hm[:, 0, :, :]   # front view
                    elif infer_pred_hm.dim() == 4:
                        infer_pred_hm = infer_pred_hm[:, -1, :, :]  # single channel
                    gt_hm_eval = gt_heatmap
                    if gt_hm_eval.dim() == 4 and gt_hm_eval.shape[1] == 4:
                        gt_hm_eval = gt_hm_eval[:, 0]
                    if infer_pred_hm.shape[-2:] != gt_hm_eval.shape[-2:]:
                        infer_pred_hm = F.interpolate(
                            infer_pred_hm.unsqueeze(1),
                            size=gt_hm_eval.shape[-2:],
                            mode='bilinear', align_corners=False
                        ).squeeze(1)
                    batch_mse = F.mse_loss(infer_pred_hm, gt_hm_eval).item()
                    total_heatmap_mse += batch_mse
                    num_heatmap_mse_batches += 1
                
                # 前几个 batch 保存可视化图片
                if num_batches <= num_vis_batches and vis_dir is not None:
                    vis_path = visualize_heatmap_predictions(
                        model=model,
                        batch=batch,
                        output=vis_output,
                        epoch=epoch,
                        step=num_batches,
                        output_dir=vis_dir,
                        num_samples=4,
                        gt_heatmap_override=gt_heatmap if gpu_heatmap_computer is not None else None,
                    )
                    
                    if vis_path is not None:
                        new_path = vis_dir / f"e{epoch:03d}_b{num_batches:02d}.png"
                        import shutil
                        shutil.copy(vis_path, new_path)
                        
                        if tb_writer is not None:
                            vis_img = cv2.imread(str(new_path))
                            if vis_img is not None:
                                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                vis_img = vis_img.transpose(2, 0, 1)
                                tb_writer.add_image(f'val/heatmap_viz_batch{num_batches}', vis_img, epoch)
                        
                        logger.info(f"[VAL-VIS] Epoch {epoch}, Batch {num_batches} visualization saved")
            except Exception as e:
                logger.warning(f"Validation inference/visualization failed: {e}")
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_hm = total_heatmap_loss / max(num_batches, 1)
    avg_act = total_action_loss / max(num_batches, 1)
    avg_stop = total_stop_loss / max(num_batches, 1)
    avg_hm_mse = total_heatmap_mse / max(num_heatmap_mse_batches, 1) if num_heatmap_mse_batches > 0 else 0.0
    
    # 注意：TensorBoard 记录移至主循环中使用 global_epoch_counter
    # 避免多阶段训练时 epoch 重复导致数据覆盖
    
    if num_heatmap_mse_batches > 0:
        logger.info(f"  📊 Heatmap 推理 MSE (采样 {num_heatmap_mse_batches} batches): {avg_hm_mse:.6f}")
    
    avg_diff = total_diffusion_loss / max(num_batches, 1)
    avg_vis = total_visibility_loss / max(num_batches, 1)
    
    return {
        'val_loss': avg_loss,
        'val_heatmap_loss': avg_hm,           # 噪声预测 loss（全量 batch）
        'val_diffusion_loss': avg_diff,       # 纯扩散 loss（不含 visibility）
        'val_visibility_loss': avg_vis,       # visibility BCE loss
        'val_heatmap_mse': avg_hm_mse,        # 完整推理 MSE（采样 batch）
        'val_action_loss': avg_act,
        'val_stop_loss': avg_stop,
        'val_total_loss': avg_loss,            # 兼容 save_best_metric
    }


class CheckpointManager:
    """管理检查点的保存、加载和清理"""
    
    def __init__(self, out_dir: str, max_ckpts: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_ckpts = max_ckpts
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None
        self.ckpt_history = []
    
    def save(
        self,
        model: VLNPipeline,
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
        """保存检查点"""
        stage_dir = self.out_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        trainable_params = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.add(name)
        
        trainable_state_dict = {
            k: v for k, v in model.state_dict().items()
            if k in trainable_params
        }
        
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
        
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        
        ckpt_path = stage_dir / f"e{epoch:03d}.pth"
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
        
        self._cleanup_old_ckpts(stage_dir)
        
        return ckpt_path
    
    def _cleanup_old_ckpts(self, stage_dir: Path):
        """清理旧的检查点"""
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
        best = self.out_dir / "best.pth"
        return best if best.exists() else None


def load_checkpoint_for_resume(
    ckpt_path: str,
    model: VLNPipeline,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: GradScaler = None,
    logger = None,
) -> Dict:
    """加载检查点用于断点续训"""
    if logger:
        logger.info(f"📂 Loading checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = ckpt.get('trainable_state_dict', {})
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if logger:
            logger.info(f"  ✓ Loaded {len(state_dict)} trainable parameters")
    
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


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="VLN 训练脚本（单阶段）")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                        help='从检查点恢复（路径或 "latest"）')
    parser.add_argument('--auto-resume', action='store_true',
                        help='自动从最新检查点恢复（继续使用之前的输出目录）')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='从指定 epoch 开始训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的 epoch 数量')
    parser.add_argument('--dry-run', action='store_true',
                        help='只构建模型和数据，不实际训练')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='每个 epoch 最多处理的 batch 数')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    
    # ==================== 输出目录结构（每次训练独立文件夹）====================
    base_out_dir = Path(cfg['log']['out_dir'])
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    latest_link = base_out_dir / 'latest'
    is_resuming = False
    
    # 判断是否断点续训
    if args.auto_resume and latest_link.exists():
        # 断点续训：使用之前的运行目录
        if latest_link.is_symlink():
            run_dir = base_out_dir / latest_link.resolve().name
        else:
            run_dir = latest_link
        
        # 检查是否有可恢复的检查点
        ckpt_dir_check = run_dir / 'ckpts'
        latest_ckpt = ckpt_dir_check / 'latest.pth' if ckpt_dir_check.exists() else None
        if latest_ckpt and latest_ckpt.exists():
            is_resuming = True
            print(f"🔄 断点续训: 继续使用 {run_dir.name}")
        else:
            # 没有可恢复的检查点，创建新目录
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = base_out_dir / f'run_{run_timestamp}'
            print(f"📁 新训练: {run_dir.name}")
    else:
        # 新训练：创建新的运行目录
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = base_out_dir / f'run_{run_timestamp}'
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 更新 'latest' 符号链接
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    
    # 子目录
    ckpt_dir = run_dir / 'ckpts'
    vis_train_dir = run_dir / 'vis' / 'train'
    vis_val_dir = run_dir / 'vis' / 'val'
    plots_dir = run_dir / 'plots'
    
    for d in [ckpt_dir, vis_train_dir, vis_val_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(str(run_dir / 'train.log'))
    logger.info(f"📁 Output: {run_dir}")
    logger.info(f"   Latest: {latest_link} → {run_dir.name}")
    
    # ==================== TensorBoard ====================
    tb_writer = None
    if cfg['log'].get('use_tensorboard', False):
        tb_base = Path(cfg['log'].get('tensorboard_dir', './runs'))
        tb_base.mkdir(parents=True, exist_ok=True)
        
        tb_latest_link = tb_base / 'latest'
        
        # 从 run_dir 名字中提取时间戳 (run_YYYYMMDD_HHMMSS -> YYYYMMDD_HHMMSS)
        run_dir_name = run_dir.name
        if run_dir_name.startswith('run_'):
            tb_timestamp = run_dir_name[4:]  # 去掉 "run_" 前缀
        else:
            tb_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if is_resuming:
            # 断点续训：使用之前的 TensorBoard 目录
            tb_run_dir = tb_base / tb_timestamp
            if not tb_run_dir.exists():
                # 如果不存在，创建新目录
                tb_run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🔄 TensorBoard: 继续使用 {tb_run_dir}")
        else:
            # 新训练：归档旧日志
            tb_archive = Path(__file__).parent.parent / 'tf-logs-archive'
            tb_archive.mkdir(parents=True, exist_ok=True)
            
            for old_dir in tb_base.iterdir():
                if old_dir.is_dir() and old_dir.name != 'latest':
                    # 移动旧目录到归档
                    dest = tb_archive / old_dir.name
                    if not dest.exists():
                        import shutil
                        shutil.move(str(old_dir), str(dest))
                        logger.info(f"📦 归档旧TensorBoard: {old_dir.name} → tf-logs-archive/")
            
            # 创建新的运行目录
            tb_run_dir = tb_base / tb_timestamp
        
        tb_writer = SummaryWriter(log_dir=str(tb_run_dir))
        
        # 更新 'latest' 符号链接
        if tb_latest_link.is_symlink() or tb_latest_link.exists():
            tb_latest_link.unlink()
        tb_latest_link.symlink_to(tb_run_dir.name)
        
        logger.info(f"📊 TensorBoard: {tb_run_dir}")
        logger.info(f"   只看当前: tensorboard --logdir {tb_latest_link}")
    
    loss_cfg = cfg['loss']
    default_loss_type = loss_cfg.get('heatmap_loss_type', 'simplified')
    
    logger.info("=" * 60)
    logger.info("VLN 训练 (Qwen3-VL)")
    logger.info("=" * 60)
    
    # 构建数据集
    logger.info("📂 Loading datasets...")
    dataset_type = cfg['data'].get('dataset_type', 'sliding_window')
    logger.info(f"  Dataset type: {dataset_type}")
    
    if dataset_type == 'trajectory':
        # 使用新的轨迹数据集（支持 24 步预测）
        traj_cfg = cfg['data'].get('trajectory', cfg['data'].get('sliding_window', {}))
        sample_stride = traj_cfg.get('sample_stride', 1)
        clip_level_sampling = traj_cfg.get('clip_level_sampling', True)
        samples_per_clip = traj_cfg.get('samples_per_clip', 8)
        
        # 随机子序列采样配置
        random_subsequence = traj_cfg.get('random_subsequence', False)
        min_subsequence_length = traj_cfg.get('min_subsequence_length', 30)
        subsequence_samples_per_clip = traj_cfg.get('subsequence_samples_per_clip', 3)
        
        train_dataset = VLNTrajectoryDataset(
            root=cfg['data']['root'],
            split='train',
            min_history=traj_cfg.get('min_history', 5),
            num_history_sample=traj_cfg.get('num_history_sample', 8),
            image_size=tuple(cfg['data']['image_size']),
            hm_size=tuple(cfg['data']['init_hm_size']),
            load_depth=traj_cfg.get('load_depth', True),
            cache_poses=traj_cfg.get('cache_poses', True),
            sample_stride=sample_stride,
            clip_level_sampling=clip_level_sampling,
            samples_per_clip=samples_per_clip,
            random_subsequence=random_subsequence,
            min_subsequence_length=min_subsequence_length,
            subsequence_samples_per_clip=subsequence_samples_per_clip,
            predict_horizon=traj_cfg.get('predict_horizon', 24),
            action_scale=traj_cfg.get('action_scale', 4.0),
            enable_trajectory_augmentation=traj_cfg.get('enable_trajectory_augmentation', True),
            # FGR2R 子指令配置
            use_subinstruction=traj_cfg.get('use_subinstruction', False),
            fgr2r_subinstr_path=traj_cfg.get('fgr2r_subinstr_path', None),
        )
        
        val_split = cfg['data'].get('val_split', 'val')
        val_samples_per_clip = traj_cfg.get('val_samples_per_clip', 2)
        val_dataset = VLNTrajectoryDataset(
            root=cfg['data']['root'],
            split=val_split,
            min_history=traj_cfg.get('min_history', 5),
            num_history_sample=traj_cfg.get('num_history_sample', 8),
            image_size=tuple(cfg['data']['image_size']),
            hm_size=tuple(cfg['data']['init_hm_size']),
            load_depth=traj_cfg.get('load_depth', True),
            cache_poses=traj_cfg.get('cache_poses', True),
            sample_stride=sample_stride,
            clip_level_sampling=clip_level_sampling,
            samples_per_clip=val_samples_per_clip,
            random_subsequence=False,  # 验证集不使用随机子序列
            predict_horizon=traj_cfg.get('predict_horizon', 24),
            action_scale=traj_cfg.get('action_scale', 4.0),
            enable_trajectory_augmentation=False,  # 验证集不增强
            use_subinstruction=False,  # 验证集使用完整指令
        )
    else:
        # 使用原始滑动窗口数据集
        sw_cfg = cfg['data']['sliding_window']
        sample_stride = sw_cfg.get('sample_stride', 1)
        clip_level_sampling = sw_cfg.get('clip_level_sampling', True)
        samples_per_clip = sw_cfg.get('samples_per_clip', 2)
        defer_heatmap_to_gpu = sw_cfg.get('defer_heatmap_to_gpu', False)
        
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
            clip_level_sampling=clip_level_sampling,
            samples_per_clip=samples_per_clip,
            defer_heatmap_to_gpu=defer_heatmap_to_gpu,
        )
        
        val_split = cfg['data'].get('val_split', 'val')
        val_samples_per_clip = sw_cfg.get('val_samples_per_clip', 2)
        val_dataset = VLNSlidingWindowDataset(
            root=cfg['data']['root'],
            split=val_split,
            min_history=sw_cfg['min_history'],
            num_history_sample=sw_cfg['num_history_sample'],
            image_size=tuple(cfg['data']['image_size']),
            hm_size=tuple(cfg['data']['init_hm_size']),
            load_depth=sw_cfg.get('load_depth', True),
            cache_poses=sw_cfg.get('cache_poses', True),
            sample_stride=sample_stride,
            clip_level_sampling=clip_level_sampling,
            samples_per_clip=val_samples_per_clip,
            defer_heatmap_to_gpu=defer_heatmap_to_gpu,
        )
    
    # 验证集使用固定 epoch=0，确保每次验证样本一致
    if hasattr(val_dataset, 'set_epoch'):
        val_dataset.set_epoch(0)
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    
    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg)
    
    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        out_dir=str(ckpt_dir),
        max_ckpts=cfg['log'].get('max_ckpts', 3)
    )
    
    # 创建通知器
    notifier = create_notifier(cfg)
    
    # 创建训练曲线绘制器
    plotter = TrainingPlotter(out_dir=plots_dir)
    
    # 断点续训
    resume_epoch = 0
    resume_path = None
    
    if args.resume:
        if args.resume == 'latest':
            resume_path = ckpt_manager.get_latest()
        else:
            resume_path = Path(args.resume)
    elif args.auto_resume:
        resume_path = ckpt_manager.get_latest()
    
    if resume_path and Path(resume_path).exists():
        resume_info = load_checkpoint_for_resume(
            str(resume_path), model, optimizer=None, scheduler=None, logger=logger
        )
        resume_epoch = resume_info['epoch']
        ckpt_manager.best_val_loss = resume_info['best_val_loss']
    
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("🧪 Dry run 模式：模型和数据构建成功")
        logger.info("=" * 60)
        return
    
    # 获取训练配置（单阶段）
    all_stages = cfg['training']['stages']
    if not all_stages:
        logger.error("❌ 配置文件中没有定义训练阶段")
        return
    
    stage_cfg = all_stages[0]  # 使用第一个（也是唯一的）阶段
    stage_name = stage_cfg['name']
    
    if args.epochs is not None:
        stage_cfg = stage_cfg.copy()
        stage_cfg['epochs'] = args.epochs
    
    total_epochs = stage_cfg['epochs']
    
    logger.info("=" * 60)
    logger.info(f"📋 训练配置: {stage_name}")
    logger.info(f"   Epochs: {total_epochs}, Heatmap Size: {stage_cfg['hm_size']}")
    logger.info("=" * 60)
    
    # 发送训练开始通知
    if notifier:
        try:
            notifier.send_training_start(
                config_name=Path(args.config).stem,
                stages=[stage_cfg],
                total_epochs=total_epochs,
            )
            logger.info("📢 飞书通知已发送: 训练开始")
        except Exception as e:
            logger.warning(f"飞书通知发送失败: {e}")
        
    # 更新热力图分辨率
    hm_size = tuple(stage_cfg['hm_size'])
    train_dataset.hm_size = hm_size
    val_dataset.hm_size = hm_size
    if hasattr(model, 'update_heatmap_size'):
        model.update_heatmap_size(hm_size)
    
    logger.info(f"  Heatmap size: {hm_size}")
    
    # 构建数据加载器
    num_workers = cfg['data']['num_workers']
    prefetch_factor = cfg['data'].get('prefetch_factor', 2)
    
    # 检查是否启用 Sequence Packing
    packing_enabled = cfg['model']['llm'].get('enable_packing', False)
    
    if packing_enabled:
        # Packing 模式：符合官方实现
        # Tokenization 在 Dataset.__getitem__() 中完成，可以利用 num_workers 并行
        logger.info("📦 Sequence Packing enabled (official implementation)")
        
        # 必须先加载模型以获取 processor
        if not model.qwen3_vl._model_loaded:
            model.qwen3_vl._load_model()
        
        # 包装数据集，在 __getitem__ 中做 tokenization
        spatial_merge_size = cfg['model']['llm'].get('spatial_merge_size', 2)
        train_dataset = TokenizedVLNDataset(
            base_dataset=train_dataset,
            processor=model.qwen3_vl.processor,
            spatial_merge_size=spatial_merge_size,
        )
        val_dataset = TokenizedVLNDataset(
            base_dataset=val_dataset,
            processor=model.qwen3_vl.processor,
            spatial_merge_size=spatial_merge_size,
        )
        
        # 使用官方的 FlattenedCollator（只做拼接，不做 tokenization）
        actual_collate_fn = FlattenedCollatorForVLN()
        
        logger.info("   Tokenization in Dataset.__getitem__() - can use num_workers")
        logger.info(f"   num_workers: {num_workers}")
    else:
        actual_collate_fn = collate_fn
    
    # 🔧 使用 fork 模式（而非 spawn），利用 copy-on-write 共享内存
    # TOKENIZERS_PARALLELISM=false 已设置，fork 安全
    # spawn 模式每个 worker 独立复制 processor+dataset ≈ 10GB/worker
    # fork 模式 workers 共享父进程内存（COW），仅 ~1-2GB/worker
    # 在 90GB Docker 容器下这是关键差异
    mp_context = 'fork' if num_workers > 0 else None
    
    uses_dynamic_sampling = hasattr(train_dataset, 'set_epoch')
    
    # 🔧 内存优化：packing 模式下禁用 persistent_workers
    # 原因：packing 模式每个 __getitem__ 执行 tokenization（分配/释放 ~20MB），
    # Python pymalloc 不归还内存给 OS，persistent_workers 下内存碎片化持续增长。
    # 禁用后每个 epoch 重建 workers，回收碎片化内存。
    if packing_enabled:
        persistent_workers = False
        logger.info("   ⚠️ Packing mode: persistent_workers=False (防止 tokenization 内存碎片化)")
    else:
        persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['optim']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=actual_collate_fn,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    
    # 验证集也需要 workers 加速 tokenization（否则主进程串行处理会极慢）
    # 使用与训练相同的 workers 配置，persistent_workers=False 保证验证结束后释放内存
    val_num_workers = min(num_workers, 4)  # 验证用 4 workers 足够
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['optim']['batch_size'],
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=actual_collate_fn,
        prefetch_factor=prefetch_factor if val_num_workers > 0 else None,
        persistent_workers=False,  # 验证结束后释放内存
    )
    logger.info(f"   📊 验证 DataLoader: num_workers={val_num_workers}, prefetch={prefetch_factor}")
    
    if uses_dynamic_sampling:
        if persistent_workers:
            logger.info("   ✅ Dynamic sampling enabled with persistent_workers")
        else:
            logger.info("   ✅ Dynamic sampling enabled (workers rebuilt each epoch to reclaim memory)")
    logger.info(f"   🧠 Memory config: num_workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    
    # ⚠️ 强制加载 Qwen3-VL（含 LoRA），确保所有参数在 set_trainable + build_optimizer 之前就位
    # 否则 LoRA 参数（懒加载，首次前向才创建）不会被 optimizer 捕获
    if hasattr(model, 'qwen3_vl') and hasattr(model.qwen3_vl, '_load_model'):
        if model.qwen3_vl.model is None:
            logger.info("🔄 Pre-loading Qwen3-VL (ensure LoRA params available for optimizer)...")
            model.qwen3_vl._load_model()
    
    # 设置可训练模块
    logger.info("🔧 Setting trainable modules...")
    set_trainable_modules(model, stage_cfg, logger)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 构建优化器和调度器
    optimizer = build_optimizer(model, cfg, stage_cfg)
    grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
    total_batches = len(train_loader) * total_epochs
    total_steps = total_batches // grad_accum_steps
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    # GradScaler 仅用于 fp16，bf16 不需要（动态范围更大）
    amp_type = cfg['optim'].get('amp', 'bf16')
    scaler = GradScaler() if amp_type == 'fp16' else None
    
    if resume_path and Path(resume_path).exists():
        load_checkpoint_for_resume(
            str(resume_path), model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            scaler=scaler,
            logger=logger
        )
    
    best_val_loss = ckpt_manager.best_val_loss
    steps_per_epoch = len(train_loader) // grad_accum_steps
    
    if resume_epoch > 0:
        start_epoch = resume_epoch + 1
        global_epoch_counter = resume_epoch  # 断点续训时从 resume_epoch 开始计数
    else:
        start_epoch = args.start_epoch
        global_epoch_counter = start_epoch - 1  # 确保第一个 epoch 记录为 start_epoch
    
    patience = cfg['validation'].get('patience', 5)
    no_improve_count = 0
    
    # GPU 热力图计算器（减少 CPU 瓶颈）
    data_cfg = cfg['data']
    sliding_cfg = data_cfg.get('sliding_window', {})
    defer_heatmap_to_gpu = sliding_cfg.get('defer_heatmap_to_gpu', False)
    
    if defer_heatmap_to_gpu:
        hm_size = tuple(data_cfg.get('init_hm_size', [64, 64]))
        # 从 intrinsics 或默认值获取图像尺寸
        img_size = (640, 480)  # 默认 Pinhole 尺寸
        
        gpu_heatmap_computer = GPUHeatmapComputer(
            hm_size=hm_size,
            img_size=img_size,
            device='cuda',
        )
        gpu_depth_normalized = not getattr(train_dataset, 'depth_is_meters', False)
        gpu_has_depth = getattr(train_dataset, 'load_depth', True)
        logger.info(f"🚀 GPU heatmap computation enabled (hm_size={hm_size}, "
                     f"depth_normalized={gpu_depth_normalized}, has_depth={gpu_has_depth})")
    else:
        gpu_heatmap_computer = None
        gpu_depth_normalized = True
        gpu_has_depth = False
    
    # EMA (Exponential Moving Average) — 扩散模型标准技术
    # 用参数的滑动平均做推理，减少训练波动，提升泛化
    ema_decay = cfg.get('optim', {}).get('ema_decay', 0.999)
    ema_warmup = cfg.get('optim', {}).get('ema_warmup_steps', 2000)
    ema = EMAModel(model, decay=ema_decay, warmup_steps=ema_warmup)
    logger.info(f"📐 EMA enabled: decay={ema_decay}, warmup_steps={ema_warmup}")
    
    timer = TrainingTimer(total_epochs=total_epochs)
    timer.start()
    
    for epoch in range(start_epoch, total_epochs + 1):
        timer.start_epoch()
            
        # Clip-level 采样：每个 epoch 重新采样，减少样本相关性
        # 🔧 修复：不再重建 DataLoader，直接修改 sample_index
        # persistent_workers 下 workers 共享 dataset 对象的引用，
        # 修改 sample_index 后，下一次迭代会自动使用新的索引
        if uses_dynamic_sampling:
            train_dataset.set_epoch(epoch)
            logger.info(f"   🔄 Resampled {len(train_dataset)} samples for epoch {epoch} (persistent workers)")
        
        logger.info("=" * 80)
        logger.info(f"[{stage_name}] Epoch {epoch}/{total_epochs}")
        logger.info("=" * 80)
        
        epoch_offset = (epoch - 1) * steps_per_epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, logger, tb_writer, epoch_offset,
            stage_idx=0, stage_name=stage_name, stage_cfg=stage_cfg,
            max_batches=args.max_batches,
            packing_enabled=packing_enabled,
            vis_dir=vis_train_dir,
            gpu_heatmap_computer=gpu_heatmap_computer,
            gpu_has_depth=gpu_has_depth,
            gpu_depth_normalized=gpu_depth_normalized,
            ema=ema,
        )
        
        timer.end_epoch()
        
        # 🧹 强制内存清理，防止内存泄漏
        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()  # 归还碎片化内存给 OS
        
        # 使用 EMA 参数进行验证（滑动平均参数更稳定，泛化更好）
        with ema.apply():
            val_metrics = validate(
                model, val_loader, cfg, logger, stage_cfg, tb_writer, epoch,
                packing_enabled=packing_enabled,
                vis_dir=vis_val_dir,
                max_batches=args.max_batches,
                gpu_heatmap_computer=gpu_heatmap_computer,
                gpu_has_depth=gpu_has_depth,
                gpu_depth_normalized=gpu_depth_normalized,
            )
        
        # 🧹 验证后再次清理内存
        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()  # 归还碎片化内存给 OS
        
        # 📊 内存使用监控
        process = psutil.Process()
        mem_info = process.memory_info()
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"  🧠 Memory: CPU={mem_info.rss / (1024**3):.2f}GB, GPU={gpu_mem:.2f}GB (reserved={gpu_reserved:.2f}GB)")
        
        logger.info(
            f"  Train Loss: {train_metrics['total_loss']:.4f} "
            f"(diff: {train_metrics.get('diffusion_loss', 0):.4f}, "
            f"vis: {train_metrics.get('visibility_loss', 0):.4f}, "
            f"traj: {train_metrics['action_loss']:.4f})"
        )
        val_hm_mse_str = f", infer_mse: {val_metrics['val_heatmap_mse']:.6f}" if val_metrics.get('val_heatmap_mse', 0) > 0 else ""
        logger.info(
            f"  Val Loss: {val_metrics['val_loss']:.4f} "
            f"(diff: {val_metrics.get('val_diffusion_loss', 0):.4f}, "
            f"vis: {val_metrics.get('val_visibility_loss', 0):.4f}, "
            f"traj: {val_metrics['val_action_loss']:.4f}{val_hm_mse_str})"
        )
        
        eta = timer.get_eta(epoch, total_epochs)
        logger.info(f"  ⏱️  Epoch time: {timer.get_epoch_time()} | ETA: {eta}")
        
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            no_improve_count = 0
            logger.info(f"  ⭐ New best val_loss: {best_val_loss:.4f}")
        else:
            no_improve_count += 1
        
        global_epoch_counter += 1
        current_lr = scheduler.get_last_lr()[0] if scheduler else 0
        
        plotter.update(
            epoch=global_epoch_counter,
            stage_name=stage_name,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
            is_best=is_best,
        )
        
        # 记录 epoch 级别的 loss 到 TensorBoard
        if tb_writer is not None:
            # 总损失对比
            tb_writer.add_scalars('loss/total', {
                'train': train_metrics['total_loss'],
                'val': val_metrics['val_loss'],
            }, global_epoch_counter)
            
            # 热力图损失对比（含 visibility）
            tb_writer.add_scalars('loss/heatmap', {
                'train': train_metrics['heatmap_loss'],
                'val': val_metrics['val_heatmap_loss'],
            }, global_epoch_counter)
            
            # 🔑 纯扩散损失对比（不含 visibility，与旧版本可比）
            tb_writer.add_scalars('loss/diffusion_only', {
                'train': train_metrics.get('diffusion_loss', train_metrics['heatmap_loss']),
                'val': val_metrics.get('val_diffusion_loss', val_metrics['val_heatmap_loss']),
            }, global_epoch_counter)
            
            # Visibility Loss
            if train_metrics.get('visibility_loss', 0) > 0:
                tb_writer.add_scalar('loss/visibility_bce', train_metrics['visibility_loss'], global_epoch_counter)
            
            # 轨迹损失对比
            tb_writer.add_scalars('loss/trajectory', {
                'train': train_metrics['action_loss'],
                'val': val_metrics['val_action_loss'],
            }, global_epoch_counter)
            
            # 进度损失对比
            tb_writer.add_scalars('loss/progress', {
                'train': train_metrics.get('stop_loss', 0),
                'val': val_metrics.get('val_stop_loss', 0),
            }, global_epoch_counter)
            
            # 完整推理 heatmap MSE（真实生成质量）
            if val_metrics.get('val_heatmap_mse', 0) > 0:
                tb_writer.add_scalar('loss/heatmap_inference_mse', val_metrics['val_heatmap_mse'], global_epoch_counter)
            
            # 学习率
            tb_writer.add_scalar('train/lr', current_lr, global_epoch_counter)
            
            # 单独的指标（方便筛选）
            tb_writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], global_epoch_counter)
            tb_writer.add_scalar('epoch/val_loss', val_metrics['val_loss'], global_epoch_counter)
            tb_writer.add_scalar('epoch/train_diffusion_loss', train_metrics.get('diffusion_loss', 0), global_epoch_counter)
            tb_writer.add_scalar('epoch/visibility_loss', train_metrics.get('visibility_loss', 0), global_epoch_counter)
            
            tb_writer.flush()
        
        # 发送飞书通知
        if notifier:
            try:
                notifier.send_epoch_report(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    stage_name=stage_name,
                    stage_idx=0,
                    total_stages=1,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    eta=eta,
                    epoch_time=timer.get_epoch_time(),
                    is_best=is_best,
                    best_val_loss=best_val_loss,
                )
            except Exception as e:
                logger.warning(f"飞书通知发送失败: {e}")
        
        if epoch % cfg['log']['save_every_epochs'] == 0 or is_best:
            # 使用 EMA 参数保存（推理时直接使用，无需额外处理）
            with ema.apply():
                ckpt_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage_idx=0,
                    stage_name=stage_name,
                    metrics={**train_metrics, **val_metrics},
                    cfg=cfg,
                    is_best=is_best,
                    scaler=scaler,
                )
        
        if no_improve_count >= patience:
            logger.info(f"  🛑 Early stopping")
            break
    
    logger.info(f"  📊 训练完成，耗时: {timer.get_total_elapsed()}")
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成！")
    logger.info("=" * 60)
    
    summary = plotter.get_summary()
    if summary:
        logger.info(f"📊 训练摘要:")
        logger.info(f"   总 Epochs: {summary.get('total_epochs', 'N/A')}")
        logger.info(f"   最佳 Epoch: {summary.get('best_epoch', 'N/A')}")
        if summary.get('best_val_loss'):
            logger.info(f"   最佳 val_loss: {summary.get('best_val_loss'):.4f}")
    
    # 发送训练完成通知
    if notifier:
        try:
            notifier.send_training_complete(
                total_time=timer.get_total_elapsed() if timer else "N/A",
                best_val_loss=best_val_loss if 'best_val_loss' in dir() else 0.0,
                final_stage=stage_name if 'stage_name' in dir() else "完成",
            )
            logger.info("📢 飞书通知已发送: 训练完成")
        except Exception as e:
            logger.warning(f"飞书通知发送失败: {e}")
    
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()
