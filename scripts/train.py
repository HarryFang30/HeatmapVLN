#!/usr/bin/env python3
"""
VLN 训练脚本
==============

使用 Qwen3.5 进行视觉语言导航训练。
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
import json
import platform
import socket
import subprocess
import shutil

# ============================================
# CUDA 性能优化
# ============================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import warnings
import gc
import logging
import time
import math
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

from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset, VLNTrajectoryDataset
from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.utils.logger import setup_logger
from src.utils.gpu_heatmap import GPUHeatmapComputer
from src.utils.notifier import FeishuNotifier, create_notifier

logger = logging.getLogger(__name__)


class DistributedContext:
    """Minimal distributed runtime context."""

    def __init__(
        self,
        enabled: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _dist_backend() -> Optional[str]:
    return dist.get_backend() if _dist_is_initialized() else None


def _normalize_state_key(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module."):]
    return name.replace(".module.", ".")


def _normalized_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        _normalize_state_key(name): value
        for name, value in model.state_dict().items()
    }


def _normalized_trainable_param_names(model: nn.Module) -> set[str]:
    return {
        _normalize_state_key(name)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _load_normalized_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], int]:
    current_state = model.state_dict()
    normalized_to_actual = {
        _normalize_state_key(name): name
        for name in current_state.keys()
    }
    remapped_state_dict = {}
    for name, value in state_dict.items():
        actual_name = normalized_to_actual.get(_normalize_state_key(name))
        if actual_name is not None:
            remapped_state_dict[actual_name] = value
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    return missing, unexpected, len(remapped_state_dict)


def _get_supported_trainable_sync_modules(
    model: VLNPipeline,
    stage_cfg: Dict[str, Any],
) -> List[Tuple[str, nn.Module]]:
    trainable = set(stage_cfg.get("trainable_modules", []))
    supported_trainable = {"heatmap_vln", "llm_projector"}
    unsupported = sorted(trainable - supported_trainable)
    if unsupported:
        raise RuntimeError(
            "Current multi-GPU trainable-module sync only supports trainable_modules "
            f"{sorted(supported_trainable)}. Unsupported entries: {unsupported}. "
            "Please keep other heads/backbone frozen in distributed mode."
        )

    sync_modules: List[Tuple[str, nn.Module]] = []

    if "llm_projector" in trainable and getattr(model, "llm_projector", None) is not None:
        if any(param.requires_grad for param in model.llm_projector.parameters()):
            sync_modules.append(("llm_projector", model.llm_projector))

    if "heatmap_vln" in trainable:
        if model.heatmap_vln is None:
            raise RuntimeError("heatmap_vln is trainable but has not been constructed before distributed sync.")
        for attr_name in ["vit_dpt_fusion", "llm_dpt_fusion", "coarse", "fine"]:
            module = getattr(model.heatmap_vln, attr_name, None)
            if module is not None and any(param.requires_grad for param in module.parameters()):
                sync_modules.append((f"heatmap_vln.{attr_name}", module))

    if not sync_modules:
        raise RuntimeError(
            "Distributed mode is enabled, but no supported trainable submodules were found for synchronization."
        )

    return sync_modules


def initialize_trainable_module_sync(
    model: VLNPipeline,
    stage_cfg: Dict[str, Any],
    dist_context: "DistributedContext",
    logger: logging.Logger,
) -> List[Tuple[str, nn.Module]]:
    sync_modules = _get_supported_trainable_sync_modules(model, stage_cfg)
    for module_name, module in sync_modules:
        logger.info("🔄 Broadcasting trainable module: %s", module_name)
        for _, param in module.named_parameters():
            if param.requires_grad:
                _dist_broadcast_in_place(param.data, src=0)
    _dist_barrier()
    logger.info(
        "🔗 Distributed trainable-module sync enabled for: %s",
        ", ".join(name for name, _ in sync_modules),
    )
    return sync_modules


def synchronize_trainable_module_gradients(
    sync_modules: List[Tuple[str, nn.Module]],
    dist_context: "DistributedContext",
) -> None:
    if not dist_context.enabled or dist_context.world_size <= 1:
        return
    for _, module in sync_modules:
        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                _dist_all_reduce_in_place(param.grad)
                param.grad.div_(dist_context.world_size)


def _dist_barrier() -> None:
    if _dist_is_initialized():
        dist.barrier()


def _dist_broadcast_in_place(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not _dist_is_initialized():
        return tensor
    backend = _dist_backend()
    if tensor.is_cuda and backend != "nccl":
        cpu_tensor = tensor.detach().cpu()
        dist.broadcast(cpu_tensor, src=src)
        tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def _dist_all_reduce_in_place(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        backend = _dist_backend()
        if tensor.is_cuda and backend != "nccl":
            cpu_tensor = tensor.detach().cpu()
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM)
            tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


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


def _cgroup_mem_usage_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ):
        try:
            with open(path, "r") as f:
                return int(f.read().strip()) / (1024 ** 3)
        except Exception:
            continue
    return -1.0


def _cgroup_mem_limit_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(path, "r") as f:
                val = f.read().strip()
                if val == "max":
                    return -1.0
                v = int(val)
                if v > 1 << 60:
                    return -1.0
                return v / (1024 ** 3)
        except Exception:
            continue
    return -1.0


_CG_LIMIT_GB = _cgroup_mem_limit_gb()


def _drop_page_cache(force: bool = False, threshold: float = 0.80):
    """Drop page cache when cgroup memory usage exceeds threshold of limit.

    Args:
        force: If True, always drop regardless of threshold.
        threshold: Fraction of cgroup limit above which to drop (default 80%).
    """
    if _CG_LIMIT_GB <= 0:
        return
    usage = _cgroup_mem_usage_gb()
    if not force and usage <= _CG_LIMIT_GB * threshold:
        return

    # Method 1: /proc/sys/vm/drop_caches (needs root / SYS_ADMIN)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")
        after = _cgroup_mem_usage_gb()
        print(
            f"[PAGE_CACHE] drop_caches: {usage:.1f}GB → {after:.1f}GB "
            f"(limit={_CG_LIMIT_GB:.0f}GB)",
            file=sys.stderr, flush=True,
        )
        return
    except PermissionError:
        pass
    except Exception as e:
        print(f"[PAGE_CACHE] drop_caches failed: {e}", file=sys.stderr, flush=True)
        return

    # Method 2: cgroup v1 force_empty (works without SYS_ADMIN in some Docker setups)
    try:
        with open("/sys/fs/cgroup/memory/memory.force_empty", "w") as f:
            f.write("0\n")
        after = _cgroup_mem_usage_gb()
        print(
            f"[PAGE_CACHE] force_empty: {usage:.1f}GB → {after:.1f}GB "
            f"(limit={_CG_LIMIT_GB:.0f}GB)",
            file=sys.stderr, flush=True,
        )
        return
    except Exception:
        pass

    # Method 3: POSIX_FADV_DONTNEED via madvise on cached training data
    # This is a last resort - we advise the kernel that we don't need the data
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        # malloc_trim at least returns heap memory
        libc.malloc_trim(0)
        print(
            f"[PAGE_CACHE] WARNING: cannot drop page cache "
            f"(no permission for drop_caches or force_empty). "
            f"cgroup={usage:.1f}/{_CG_LIMIT_GB:.0f}GB. "
            f"Consider running: chmod 666 /proc/sys/vm/drop_caches",
            file=sys.stderr, flush=True,
        )
    except Exception:
        pass


def _worker_init_fn(worker_id):
    """Worker 进程初始化函数 - 抑制警告 + 内存管理"""
    import gc as _gc
    import os as _os
    import sys as _sys
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
    warnings.filterwarnings("ignore", message="Asked to sample")

    _gc.set_threshold(700, 10, 999_999_999)

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 32 * 1024)   # M_MMAP_THRESHOLD  → 32 KB
        libc.mallopt(-1, 64 * 1024)   # M_TRIM_THRESHOLD  → 64 KB
        libc.mallopt(-8, 2)           # M_ARENA_MAX → 2
    except Exception:
        pass

    try:
        with open("/proc/self/statm", "rb") as f:
            pages = int(f.read().split()[1])
        rss_mb = pages * _os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        print(
            f"[WORKER init] worker_id={worker_id} pid={_os.getpid()} "
            f"rss={rss_mb:.0f}MB gc_threshold={_gc.get_threshold()}",
            file=_sys.stderr, flush=True,
        )
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
            'best_epoch': self.history['epoch'][best_idx] if best_idx is not None else None,
            'best_val_loss': best_val if best_idx is not None else None,
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
    VIEW_LABELS = ["Front", "Right", "Back", "Left"]
    
    try:
        gt_heatmaps = gt_heatmap_override if gt_heatmap_override is not None else batch['heatmap']
        pred_heatmaps = output.get('heatmaps')
        
        if pred_heatmaps is None:
            return

        has_panoramic = 'current_views' in batch
        B = min(num_samples, batch['current_frame'].shape[0])

        if has_panoramic:
            pred_vis_raw = output.get('visibility')       # (B, N_hist, 4) or None
            gated_heatmaps = output.get('heatmaps_gated') # (B, N_hist, 4, H, W) or None
            batch_gt_vis = batch.get('gt_visibility')     # (B, N_hist, 4) or None

            total_rows = B * 4
            fig, axes = plt.subplots(total_rows, 4, figsize=(16, 4 * total_rows))
            if total_rows == 1:
                axes = axes[np.newaxis, :]

            for b in range(B):
                views = batch['current_views'][b]  # (4, C, H, W)
                row_offset = b * 4

                # --- GT heatmap: aggregate N_hist via max → (4, Hm, Wm) ---
                gt_b = gt_heatmaps[b]
                if gt_b.dim() == 4:
                    gt_4 = gt_b.max(dim=0).values
                elif gt_b.dim() == 3 and gt_b.shape[0] == 4:
                    gt_4 = gt_b
                else:
                    gt_4 = gt_b.unsqueeze(0).expand(4, -1, -1)

                # --- Pred gated heatmap: aggregate N_hist via max → (4, H, W) ---
                gated_4 = None
                if gated_heatmaps is not None:
                    if gated_heatmaps.dim() == 5:
                        gated_4 = gated_heatmaps[b].max(dim=0).values
                    elif gated_heatmaps.dim() == 4:
                        gated_4 = gated_heatmaps[b]

                if gated_4 is None:
                    if pred_heatmaps.dim() == 5:
                        pred_b = pred_heatmaps[b]
                    elif pred_heatmaps.dim() == 4 and pred_heatmaps.shape[1] == 4:
                        pred_b = pred_heatmaps[b].unsqueeze(0)
                    else:
                        pred_b = pred_heatmaps[b].unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1)
                    N_h, _, Hm, Wm = pred_b.shape
                    sig = pred_b.detach().float().clamp(1e-6, 1 - 1e-6)
                    logits = torch.logit(sig)
                    probs = torch.softmax(logits.reshape(N_h, 4, -1), dim=-1).reshape(N_h, 4, Hm, Wm)
                    if pred_vis_raw is not None:
                        if pred_vis_raw.dim() == 3:
                            vis_gate = torch.sigmoid(pred_vis_raw[b].detach().float())
                        else:
                            vis_gate = torch.sigmoid(pred_vis_raw[b].detach().float()).unsqueeze(0)
                        probs = probs * vis_gate[:, :, None, None]
                    gated_4 = probs.max(dim=0).values

                # --- Visibility: aggregate via max across N_hist → (4,) ---
                if pred_vis_raw is not None:
                    if pred_vis_raw.dim() == 3:
                        vis_scores = torch.sigmoid(pred_vis_raw[b].detach().float()).max(dim=0).values.cpu().numpy()
                    else:
                        vis_scores = torch.sigmoid(pred_vis_raw[b].detach().float()).cpu().numpy()
                else:
                    vis_scores = np.ones(4)

                if batch_gt_vis is not None:
                    if batch_gt_vis.dim() == 3:
                        gt_vis_4 = batch_gt_vis[b].float().max(dim=0).values.cpu().numpy()
                    else:
                        gt_vis_4 = batch_gt_vis[b].float().cpu().numpy()
                else:
                    gt_vis_4 = (gt_4.float().amax(dim=(-2, -1)).cpu().numpy() > 0).astype(float)

                N_hist_count = gt_b.shape[0] if gt_b.dim() == 4 else 1

                for v in range(4):
                    r = row_offset + v
                    rgb = views[v].cpu().numpy().transpose(1, 2, 0)
                    rgb = np.clip(rgb, 0, 1)
                    axes[r, 0].imshow(rgb)
                    label = f"S{b} {VIEW_LABELS[v]}" if v == 0 else VIEW_LABELS[v]
                    axes[r, 0].set_title(label, fontweight='bold')
                    axes[r, 0].axis('off')

                    gt_hm = gt_4[v].float().cpu().numpy()
                    axes[r, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=max(gt_hm.max(), 0.01))
                    axes[r, 1].set_title(f"GT (max={gt_hm.max():.2f})")
                    axes[r, 1].axis('off')

                    gated_v = gated_4[v].detach().float().cpu().numpy()
                    gated_vmax = max(gated_v.max(), 1e-8)
                    axes[r, 2].imshow(gated_v, cmap='inferno', vmin=0, vmax=gated_vmax)
                    axes[r, 2].set_title(f"Gated (max={gated_v.max():.4f})")
                    axes[r, 2].axis('off')

                    pred_v = vis_scores[v]
                    gt_v = gt_vis_4[v]
                    correct = (pred_v > 0.5) == (gt_v > 0.5)
                    bg_color = [0.85, 0.95, 0.85] if correct else [0.95, 0.85, 0.85]
                    axes[r, 3].set_facecolor(bg_color)
                    axes[r, 3].text(
                        0.5, 0.55,
                        f"Pred vis: {pred_v:.2f}\nGT vis: {gt_v:.0f}",
                        ha='center', va='center', fontsize=14, fontfamily='monospace',
                        transform=axes[r, 3].transAxes,
                    )
                    status = "OK" if correct else "WRONG"
                    axes[r, 3].text(
                        0.5, 0.15, status,
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        color='green' if correct else 'red',
                        transform=axes[r, 3].transAxes,
                    )
                    axes[r, 3].set_title("Visibility")
                    axes[r, 3].set_xticks([])
                    axes[r, 3].set_yticks([])

                    if v == 0:
                        axes[r, 0].set_ylabel(
                            f"Sample {b}\n(N={N_hist_count})",
                            fontsize=12, fontweight='bold', rotation=0,
                            labelpad=60, va='center',
                        )

            plt.suptitle(f"Epoch {epoch}, Step {step} — {B} samples, max-agg", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            save_path = output_dir / f"e{epoch:03d}_s{step:05d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path

        else:
            # Legacy single-view fallback
            current_frames = batch['current_frame']
            pred_heatmaps_2d = _select_primary_heatmap_slice(pred_heatmaps)
            gt_heatmaps_2d = _select_primary_heatmap_slice(gt_heatmaps)

            fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            for i in range(B):
                rgb = current_frames[i].cpu().numpy().transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1)
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title("Input Frame")
                axes[i, 0].axis('off')
                gt_hm = gt_heatmaps_2d[i].cpu().numpy()
                axes[i, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=1)
                axes[i, 1].set_title(f"GT (max={gt_hm.max():.2f})")
                axes[i, 1].axis('off')
                pred_sig = pred_heatmaps_2d[i].detach().float()
                _lg = torch.logit(pred_sig.clamp(1e-6, 1 - 1e-6))
                pred_prob = torch.softmax(_lg.reshape(-1), dim=0).reshape_as(pred_sig).cpu().numpy()
                pred_vmax = max(pred_prob.max(), 1e-6)
                axes[i, 2].imshow(pred_prob, cmap='inferno', vmin=0, vmax=pred_vmax)
                pr = pred_prob.max() / (1.0 / (pred_prob.shape[0] * pred_prob.shape[1]))
                axes[i, 2].set_title(f"Pred ({pr:.0f}× unif)")
                axes[i, 2].axis('off')

            plt.suptitle(f"Epoch {epoch}, Step {step}")
            plt.tight_layout()
            save_path = output_dir / f"e{epoch:03d}_s{step:05d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        return None


def _should_use_gpu_gt(batch: Dict[str, Any], gpu_heatmap_computer: Optional[GPUHeatmapComputer]) -> bool:
    if gpu_heatmap_computer is None or 'history_poses' not in batch:
        return False
    # Panoramic v2 GT is produced per history position and per view in the dataset.
    # The GPU helper only supports the legacy single-view layout.
    return 'current_views' not in batch


def _select_primary_heatmap_slice(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() == 5:
        return heatmaps[:, 0, 0]
    if heatmaps.dim() == 4 and heatmaps.shape[1] == 4:
        return heatmaps[:, 0]
    if heatmaps.dim() == 4:
        return heatmaps[:, -1]
    return heatmaps


def _get_trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _mean_timing(stats: Dict[str, float], count: int, key: str) -> float:
    if count <= 0:
        return 0.0
    return stats.get(key, 0.0) / count


def _format_qwen_internal_timing(stats: Dict[str, float], count: int) -> str:
    if count <= 0:
        return ""

    def avg(key: str) -> float:
        return _mean_timing(stats, count, key)

    qwen_vis = avg('qwen_visual_encode_s')
    qwen_lm = avg('qwen_language_model_s')
    qwen_layers = avg('qwen_llm_layers_s')
    qwen_full = avg('qwen_llm_full_attn_s')
    qwen_linear = avg('qwen_llm_linear_attn_s')
    qwen_mlp = avg('qwen_llm_mlp_s')
    qwen_norm = avg('qwen_llm_norm_s')
    qwen_patch = avg('qwen_visual_patch_embed_s')
    qwen_pos = avg('qwen_visual_pos_embed_s')
    qwen_rot = avg('qwen_visual_rotary_s')
    qwen_blocks = avg('qwen_visual_blocks_s')
    qwen_attn = avg('qwen_visual_attn_s')
    qwen_vmlp = avg('qwen_visual_mlp_s')
    qwen_vnorm = avg('qwen_visual_norm_s')
    qwen_merger = avg('qwen_visual_merger_s')

    sections = []
    if any(v > 0 for v in [qwen_vis, qwen_lm, qwen_layers, qwen_full, qwen_linear, qwen_mlp, qwen_norm]):
        qwen_nonlayer = max(qwen_lm - qwen_layers, 0.0)
        qwen_lres = max(qwen_layers - qwen_full - qwen_linear - qwen_mlp - qwen_norm, 0.0)
        qwen_residual = max(avg('qwen_forward_s') - qwen_vis - qwen_lm, 0.0)
        sections.append(
            f"Q[s] vis={qwen_vis:.3f} lm={qwen_lm:.3f} layers={qwen_layers:.3f} "
            f"full={qwen_full:.3f} linear={qwen_linear:.3f} mlp={qwen_mlp:.3f} "
            f"norm={qwen_norm:.3f} lres={qwen_lres:.3f} nonlayer={qwen_nonlayer:.3f} "
            f"residual={qwen_residual:.3f}"
        )

    if any(v > 0 for v in [qwen_patch, qwen_pos, qwen_rot, qwen_blocks, qwen_attn, qwen_vmlp, qwen_vnorm, qwen_merger]):
        qwen_vres = max(qwen_blocks - qwen_attn - qwen_vmlp - qwen_vnorm, 0.0)
        qwen_vnon = max(qwen_vis - qwen_patch - qwen_pos - qwen_rot - qwen_blocks - qwen_merger, 0.0)
        sections.append(
            f"QV[s] patch={qwen_patch:.3f} pos={qwen_pos:.3f} rot={qwen_rot:.3f} "
            f"blocks={qwen_blocks:.3f} attn={qwen_attn:.3f} mlp={qwen_vmlp:.3f} "
            f"norm={qwen_vnorm:.3f} merger={qwen_merger:.3f} vres={qwen_vres:.3f} "
            f"vnon={qwen_vnon:.3f}"
        )

    return " | ".join(sections)


def _format_decode_internal_timing(stats: Dict[str, float], count: int) -> str:
    if count <= 0:
        return ""

    def avg(key: str) -> float:
        return _mean_timing(stats, count, key)

    vit = avg('decode_vit_fusion_s')
    llm = avg('decode_llm_fusion_s')
    coarse = avg('decode_coarse_s')
    fine = avg('decode_fine_s')
    post = avg('decode_post_s')
    if not any(v > 0 for v in [vit, llm, coarse, fine, post]):
        return ""
    return (
        f"D[s] vit={vit:.3f} llm={llm:.3f} coarse={coarse:.3f} "
        f"fine={fine:.3f} post={post:.3f}"
    )


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
    if 'history_panoramas' in batch[0]:
        result['history_panoramas'] = torch.stack([s['history_panoramas'] for s in batch], dim=0)
    if 'gt_visibility' in batch[0]:
        result['gt_visibility'] = torch.stack([s['gt_visibility'] for s in batch], dim=0)
    
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

def build_model(cfg: Dict, verbose: bool = True) -> nn.Module:
    """构建 VLN Pipeline"""
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
    stop_cfg = model_cfg.get('stop_head', {})
    progress_cfg = model_cfg.get('progress_head', {})
    
    # 确定动作头类型
    action_head_type = action_cfg.get('type', 'transformer')
    
    # 获取 Legacy 和 Transformer 配置
    legacy_action_cfg = action_cfg.get('legacy', {})
    transformer_action_cfg = action_cfg.get('transformer', {})
    
    config = VLNPipelineConfig(
        # Qwen3.5
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3.5'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 4096),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        llm_enable_internal_profiling=llm_cfg.get('enable_internal_profiling', False),
        enable_runtime_timing=cfg.get('log', {}).get('enable_timing', False),
        llm_enable_compile=llm_cfg.get('enable_compile', False),
        llm_compile_mode=llm_cfg.get('compile_mode', 'reduce-overhead'),
        llm_compile_backend=llm_cfg.get('compile_backend', 'inductor'),
        
        # Sequence Packing (disabled for Qwen3.5)
        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        
        # Device
        device=model_cfg.get('device', 'cuda'),
        
        # HeatmapVLN v2 (Coarse-to-Fine)
        enable_heatmap=heatmap_cfg.get('enable', True),
        heatmap_c_vit=heatmap_cfg.get('c_vit', 1152),
        heatmap_c_llm=heatmap_cfg.get('c_llm', 4096),
        heatmap_c_fused=heatmap_cfg.get('c_fused', 256),
        heatmap_vit_layer_indices=heatmap_cfg.get('vit_layer_indices', [6, 12, 18, 24]),
        heatmap_llm_layer_indices=heatmap_cfg.get('llm_layer_indices', [7, 15, 23]),
        heatmap_size=tuple(heatmap_cfg.get('heatmap_size', cfg['data']['init_hm_size'])),
        image_size=heatmap_cfg.get('image_size', cfg['data']['image_size'][0]),
        heatmap_lambda_vis=heatmap_cfg.get('lambda_vis', 1.0),
        heatmap_lambda_coord=heatmap_cfg.get('lambda_coord', 1.0),
        heatmap_lambda_kl=heatmap_cfg.get('lambda_kl', heatmap_cfg.get('lambda_pos', 1.0)),
        heatmap_lambda_neg=heatmap_cfg.get('lambda_neg', 1.0),
        
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
    if verbose:
        print(f"✅ VLN Pipeline 已构建")
        print(f"   Qwen3.5 → {llm_cfg.get('model_path', './models/qwen_3.5')}")
        print(f"   SequencePacking → enabled={packing_enabled} (Qwen3.5 不支持启用)")
        print(
            "   HeatmapVLN → "
            f"enabled={heatmap_cfg.get('enable', True)}, "
            f"c_vit={heatmap_cfg.get('c_vit', 1152)}, "
            f"c_llm={heatmap_cfg.get('c_llm', 4096)}, "
            f"c_fused={heatmap_cfg.get('c_fused', 256)}, "
            f"vit_layers={heatmap_cfg.get('vit_layer_indices', [6, 12, 18, 24])}, "
            f"llm_layers={heatmap_cfg.get('llm_layer_indices', [7, 15, 23])}"
        )
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
    
    # HeatmapVLN v2 (DPT fusions + CoarseLocalization + FineLocalization)
    if 'heatmap_vln' in trainable:
        if hasattr(model, 'heatmap_vln') and model.heatmap_vln is not None:
            freeze_module(model.heatmap_vln.vit_dpt_fusion, freeze=False)
            freeze_module(model.heatmap_vln.llm_dpt_fusion, freeze=False)
            freeze_module(model.heatmap_vln.coarse, freeze=False)
            freeze_module(model.heatmap_vln.fine, freeze=False)
            logger.info("  ✓ Unfrozen: heatmap_vln (vit_dpt + llm_dpt + coarse + fine)")
    
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
    
    
    # Qwen3.5: 冻结基座，但保留 LoRA 参数可训练
    if hasattr(model, 'qwen3_5') and model.qwen3_5 is not None:
        freeze_module(model.qwen3_5, freeze=True)
        # 如果配置了 LoRA 且在 trainable_modules 中，解冻 LoRA 参数
        # 兼容 'lora' 和 'qwen3_5_lora' 两种写法
        if 'lora' in trainable or 'qwen3_5_lora' in trainable:
            lora_count = 0
            for name, param in model.qwen3_5.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    lora_count += 1
            if lora_count > 0:
                logger.info(f"  ✓ Unfrozen: qwen3_5 LoRA ({lora_count} parameter tensors)")
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
            # bias、LayerNorm、以及 softmax 前的混合权重不使用 weight_decay
            if 'bias' in n or 'norm' in n.lower() or 'ln' in n.lower() or n == 'layer_weights':
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
    # HeatmapVLN v2 param groups
    heatmap_lr = optim_cfg.get('heatmap_lr', 2e-4)
    if hasattr(model, 'heatmap_vln') and model.heatmap_vln is not None:
        for name, submodule in [
            ('vit_dpt_fusion', model.heatmap_vln.vit_dpt_fusion),
            ('llm_dpt_fusion', model.heatmap_vln.llm_dpt_fusion),
            ('coarse',         model.heatmap_vln.coarse),
            ('fine',           model.heatmap_vln.fine),
        ]:
            groups = get_param_groups_with_wd(submodule, heatmap_lr, f'heatmap_{name}', default_wd)
            if groups:
                param_groups.extend(groups)
                print(f"  Param group: heatmap_{name} (lr={heatmap_lr}, wd={default_wd})")
    
    
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
    
    
    # Qwen3.5 LoRA parameters (very low LR for backbone fine-tuning)
    lora_lr = optim_cfg.get('lora_lr', 1e-5)
    if hasattr(model, 'qwen3_5') and model.qwen3_5 is not None:
        lora_params = [p for n, p in model.qwen3_5.named_parameters() 
                       if p.requires_grad and 'lora_' in n]
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'weight_decay': 0.0,  # LoRA 通常不使用 weight_decay
                'name': 'qwen3_5_lora'
            })
            print(f"  Param group: qwen3_5_lora (lr={lora_lr}, wd=0.0, params={len(lora_params)})")
    
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


def get_heatmap_temperature(cfg: Dict, step: int, total_steps: int) -> float:
    """按优化步数返回当前 soft-argmax temperature。"""
    heatmap_loss_cfg = cfg.get('loss', {}).get('heatmap_vln', {})
    base_temperature = float(heatmap_loss_cfg.get('temperature', 1.0))
    schedule_cfg = heatmap_loss_cfg.get('temperature_schedule', {})

    if not schedule_cfg or not schedule_cfg.get('enabled', False):
        return base_temperature

    start_temp = float(schedule_cfg.get('start', base_temperature))
    end_temp = float(schedule_cfg.get('end', start_temp))
    mode = str(schedule_cfg.get('mode', 'cosine')).lower()

    if total_steps <= 1:
        return end_temp

    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)

    if mode == 'linear':
        interp = progress
    else:
        # 默认用 cosine 插值，与学习率主调度节奏一致。
        interp = 0.5 * (1.0 - math.cos(math.pi * progress))

    return start_temp + (end_temp - start_temp) * interp


def init_distributed_context(cfg: Dict) -> DistributedContext:
    """Initialize optional DDP runtime from config + torchrun env."""
    gpu_cfg = cfg.get("gpu", {})
    multi_gpu_cfg = gpu_cfg.get("multi_gpu", {})
    enabled = bool(multi_gpu_cfg.get("enabled", False))
    configured_devices = list(gpu_cfg.get("devices", [0]))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))

    if not enabled:
        if world_size_env > 1:
            raise RuntimeError(
                "Detected torchrun distributed environment, but gpu.multi_gpu.enabled is false. "
                "Please enable the multi-GPU switch or launch with plain python for single-card training."
            )
        if torch.cuda.is_available():
            device_id = int(configured_devices[0]) if configured_devices else 0
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        return DistributedContext(enabled=False, device=device)

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA, but CUDA is unavailable.")

    world_size = world_size_env
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        requested = len(configured_devices)
        raise RuntimeError(
            "Detected gpu.multi_gpu.enabled=true but no torchrun distributed environment. "
            f"Please launch with torchrun --nproc_per_node={requested} scripts/train.py ..."
        )

    if configured_devices and world_size != len(configured_devices):
        raise RuntimeError(
            f"WORLD_SIZE={world_size} does not match configured gpu.devices={configured_devices}."
        )

    if local_rank >= len(configured_devices):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds configured gpu.devices={configured_devices}."
        )

    device_id = int(configured_devices[local_rank])
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend=gpu_cfg.get("backend", "nccl"),
        init_method="env://",
    )
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device(f"cuda:{device_id}"),
    )


def cleanup_distributed() -> None:
    if _dist_is_initialized():
        dist.destroy_process_group()


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
    vis_dir: Optional[Path] = None,
    gpu_heatmap_computer: Optional[GPUHeatmapComputer] = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    ema: Optional[EMAModel] = None,
    metrics_jsonl_path: Optional[Path] = None,
    total_train_steps: int = 1,
    dist_context: Optional[DistributedContext] = None,
    ckpt_manager: Optional['CheckpointManager'] = None,
    mid_epoch_save_every: int = 500,
) -> Dict[str, float]:
    """训练一个 epoch"""
    dist_context = dist_context or DistributedContext(
        enabled=False,
        device=torch.device(cfg['model'].get('device', 'cuda')),
    )
    model_module = _unwrap_model(model)
    model.train()
    synced_trainable_modules = (
        _get_supported_trainable_sync_modules(model_module, stage_cfg)
        if dist_context.enabled
        else []
    )
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    total_stop_loss = 0.0
    num_batches = 0
    
    optim_cfg = cfg['optim']
    loss_cfg = cfg['loss']
    grad_accum_steps = optim_cfg.get('grad_accum_steps', 1)
    
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    
    device = dist_context.device
    
    from src.models.heatmap import HeatmapVLNLoss
    hm_loss_fn = HeatmapVLNLoss(
        lambda_vis=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_vis', 1.0),
        lambda_coord=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_coord', 1.0),
        lambda_kl=cfg.get('loss', {}).get('heatmap_vln', {}).get(
            'lambda_kl',
            cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_pos', 1.0),
        ),
        lambda_neg=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_neg', 1.0),
        lambda_peak=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_peak', 1.0),
        temperature=cfg.get('loss', {}).get('heatmap_vln', {}).get('temperature', 1.0),
        heatmap_size=tuple(cfg['model'].get('heatmap', {}).get('heatmap_size', cfg['data']['init_hm_size'])),
    ).to(device)
    hm_loss_fn.set_temperature(
        get_heatmap_temperature(cfg, global_step_offset, total_train_steps)
    )
    
    total_batches = len(train_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式: 只处理 {total_batches} batches")
    
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{stage_cfg['epochs']}",
        total=total_batches,
        ncols=cfg['log'].get('tqdm_ncols', 120),
        disable=not dist_context.is_main,
    )
    
    global_step = 0
    valid_batch_count = 0
    enable_timing = cfg.get('log', {}).get('enable_timing', False)
    timing_stats = {
        'data_wait_s': 0.0,
        'gt_s': 0.0,
        'forward_s': 0.0,
        'backward_s': 0.0,
        'optimizer_s': 0.0,
        'prepare_inputs_s': 0.0,
        'qwen_forward_s': 0.0,
        'heatmap_decode_s': 0.0,
        'pipeline_qwen_total_s': 0.0,
    } if enable_timing else {}
    profiled_steps = 0
    prev_step_end = time.perf_counter() if enable_timing else 0.0
    
    # 同步 diffusion head 的推理计数器，确保与 global_step 对齐
    # _training_step_counter 每 batch +1, global_step 每 grad_accum_steps batch +1
    # 设 _inference_interval = grad_accum * diag_interval, 并每 epoch 重置计数器
    diag_interval = cfg['log'].get('diag_interval', 100)
    aligned_interval = grad_accum_steps * diag_interval
    for head_attr in ['heatmap_vln']:
        head = getattr(model, head_attr, None)
        if head is not None and hasattr(head, '_training_step_counter'):
            head._training_step_counter = 0
            head._inference_interval = aligned_interval
    
    trainable_params = _get_trainable_params(model_module)

    _mem_log_proc = psutil.Process()
    _cg_limit = _CG_LIMIT_GB

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        if i <= 5 or i % 25 == 0:
            main_rss = _mem_log_proc.memory_info().rss / (1024 * 1024)
            children = _mem_log_proc.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children) / (1024 * 1024)
            cg_used = _cgroup_mem_usage_gb()
            cg_info = f"cgroup: {cg_used:.1f}/{_cg_limit:.0f}GB" if _cg_limit > 0 else f"cgroup: {cg_used:.1f}GB(no limit)"
            print(
                f"[MAIN batch={i}] main_rss={main_rss:.0f}MB "
                f"children({len(children)})={child_rss:.0f}MB "
                f"total={main_rss + child_rss:.0f}MB | "
                f"{cg_info}",
                file=sys.stderr,
                flush=True,
            )
            _drop_page_cache()

        if enable_timing:
            loop_start = time.perf_counter()
            timing_stats['data_wait_s'] += max(loop_start - prev_step_end, 0.0)
        
        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        B, K, C, H, W = history_frames.shape
        
        gt_action = batch['action'].to(device, non_blocking=True)
        action_valid = batch['action_valid'].to(device, non_blocking=True)
        is_stop = batch['is_stop'].to(device, non_blocking=True)
        text = batch['text']
        
        if enable_timing:
            gt_start = time.perf_counter()
        # GPU 热力图计算（如果启用）
        if _should_use_gpu_gt(batch, gpu_heatmap_computer):
            history_poses = batch['history_poses'].to(device, non_blocking=True)  # [B, K, 4, 4]
            current_poses = batch['current_pose'].to(device, non_blocking=True)   # [B, 4, 4]
            
            current_depths = batch['current_depth'].to(device, non_blocking=True) if gpu_has_depth and 'current_depth' in batch else None
            intrinsics = batch['intrinsics'].to(device, non_blocking=True) if 'intrinsics' in batch else None
            
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
            gt_heatmap = batch['heatmap'].to(device, non_blocking=True)
        if enable_timing:
            timing_stats['gt_s'] += time.perf_counter() - gt_start
        
        # 前向传播
        if enable_timing:
            forward_start = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if text and len(text) > 0:
                instruction_text = list(text)
            else:
                instruction_text = None
            current_views_batch = batch.get('current_views')
            history_panoramas_batch = batch.get('history_panoramas')
            panoramic_inputs_batch = batch.get('pano_inputs')
            panoramic_num_histories = batch.get('pano_num_histories')
            panoramic_text_anchor_positions = batch.get('pano_text_anchor_positions')
            if panoramic_inputs_batch is not None and not train_action:
                video_frames = current_frame.unsqueeze(1)
            else:
                video_frames = torch.cat([
                    history_frames,
                    history_frames[:, -1:],
                ], dim=1)
            
            output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame,
                current_views=current_views_batch,
                history_panoramas=history_panoramas_batch,
                panoramic_inputs=panoramic_inputs_batch,
                panoramic_num_histories=panoramic_num_histories,
                panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                return_heatmaps=True,
                return_actions=train_action,
                gt_actions=gt_action.unsqueeze(1) if train_action else None,
                action_valid=action_valid if train_action else None,
                gt_stop=is_stop if train_action else None,
                gt_history_heatmap=gt_heatmap if train_history else None,
                gt_future_heatmap=gt_heatmap if train_future else None,
            )
            
            # Heatmap Loss (v2: HeatmapVLNLoss computed externally)
            heatmap_loss = torch.tensor(0.0, device=device)
            loss_dict = None
            
            if train_history and 'visibility' in output and 'heatmaps' in output:
                if gt_heatmap is not None:
                    if 'gt_visibility' in batch:
                        gt_vis = batch['gt_visibility'].to(device, non_blocking=True)
                    else:
                        gt_vis = gt_heatmap.amax(dim=(-2, -1)).clamp(0, 1).to(device)
                    hm_history_mask = batch.get('history_mask')
                    if hm_history_mask is not None:
                        hm_history_mask = hm_history_mask.to(device, non_blocking=True)
                    loss_dict = hm_loss_fn(
                        output['visibility'],
                        output['heatmaps'],
                        gt_vis=gt_vis,
                        gt_heatmaps=gt_heatmap.to(device, non_blocking=True),
                        history_mask=hm_history_mask,
                    )
                    heatmap_loss = loss_dict['total']
            
            # Action Loss / Trajectory Loss
            action_loss = torch.tensor(0.0, device=device)
            trajectory_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Transformer Action Head (new) - 使用 trajectory
                if hasattr(model_module, 'transformer_action_head') and model_module.transformer_action_head is not None:
                    if 'trajectory' in batch:
                        gt_trajectory = batch['trajectory'].to(device, non_blocking=True)
                        trajectory_valid = batch['trajectory_valid'].to(device, non_blocking=True)
                        # 传入完整 llm_tokens 序列
                        traj_result = model_module.transformer_action_head.compute_loss(
                            output['llm_tokens'],
                            gt_trajectory,
                            trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                # Legacy Action Head - 使用单步动作
                elif hasattr(model_module, 'action_head') and model_module.action_head is not None and 'action_cond' in output:
                    action_result = model_module.action_head.compute_loss(
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
                if hasattr(model_module, 'progress_head') and model_module.progress_head is not None:
                    if 'progress' in batch:
                        gt_progress = batch['progress'].to(device, non_blocking=True)
                        # 使用 trajectory_valid 作为 mask（更准确）或 action_valid 作为备选
                        progress_valid = batch.get('trajectory_valid', action_valid).to(device)
                        progress_result = model_module.progress_head(
                            output['llm_tokens'],  # 传入完整序列 (B, seq_len, D)
                            gt_progress=gt_progress,
                            action_valid=progress_valid,
                            return_loss=True,
                        )
                        progress_loss = progress_result['loss']
                # Legacy Stop Head
                elif hasattr(model_module, 'stop_head') and model_module.stop_head is not None and 'stop_logits' in output:
                    stop_loss = model_module.stop_head.compute_loss(
                        output['stop_logits'],
                        is_stop,
                        action_valid
                    )
            
            # 总损失
            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            action_weight = loss_cfg.get('action_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 1.0)
            stop_weight = loss_cfg.get('stop_weight', 0.5)
            progress_weight = loss_cfg.get('progress_weight', 0.5)
            
            # 使用 trajectory_loss 或 action_loss（根据哪个有效）
            action_total_loss = trajectory_loss if trajectory_loss.item() > 0 else action_loss
            stop_total_loss = progress_loss if progress_loss.item() > 0 else stop_loss
            
            loss = heatmap_weight * heatmap_loss + trajectory_weight * action_total_loss + progress_weight * stop_total_loss
            loss = loss / grad_accum_steps
        if enable_timing:
            timing_stats['forward_s'] += time.perf_counter() - forward_start
            profiled_steps += 1

            metadata = output.get('processing_metadata', {}) if isinstance(output, dict) else {}
            model_timings = metadata.get('timings') or {}
            for key, value in model_timings.items():
                if isinstance(value, (int, float)):
                    timing_stats[key] = timing_stats.get(key, 0.0) + float(value)
        
        # 反向传播
        if enable_timing:
            backward_start = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if enable_timing:
            timing_stats['backward_s'] += time.perf_counter() - backward_start
        valid_batch_count += 1
        
        # 梯度累积
        if valid_batch_count % grad_accum_steps == 0:
            if enable_timing:
                opt_start = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
                synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                synchronize_trainable_module_gradients(synced_trainable_modules, dist_context)
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update()
            if enable_timing:
                timing_stats['optimizer_s'] += time.perf_counter() - opt_start
            global_step += 1
            current_heatmap_temperature = get_heatmap_temperature(
                cfg,
                global_step_offset + global_step,
                total_train_steps,
            )
            hm_loss_fn.set_temperature(current_heatmap_temperature)
            
            # 日志
            log_interval = cfg['log'].get('log_interval', 10)
            if global_step % log_interval == 0 or global_step <= 3:
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                all_lrs = scheduler.get_last_lr()
                lr_strs = []
                for gi, lr_val in enumerate(all_lrs):
                    gname = optimizer.param_groups[gi].get('name', f'g{gi}')
                    lr_strs.append(f"{gname}={lr_val:.2e}")
                lr_display = ", ".join(lr_strs)
                logger.info(
                    f"[{stage_name}] "
                    f"Epoch {epoch}/{stage_cfg['epochs']} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item()*grad_accum_steps:.4f} "
                    f"(hm: {heatmap_loss.item():.4f}, traj: {trajectory_loss.item():.4f}, prog: {progress_loss.item():.4f}) | "
                    f"Temp: {current_heatmap_temperature:.3f} | "
                    f"LR: [{lr_display}] | "
                    f"GPU: {mem_alloc:.1f}GB"
                    + (
                        (
                            f" | T[s] data={_mean_timing(timing_stats, profiled_steps, 'data_wait_s'):.3f} "
                            f"gt={_mean_timing(timing_stats, profiled_steps, 'gt_s'):.3f} "
                            f"fwd={_mean_timing(timing_stats, profiled_steps, 'forward_s'):.3f} "
                            f"(prep={_mean_timing(timing_stats, profiled_steps, 'prepare_inputs_s'):.3f} "
                            f"qwen={_mean_timing(timing_stats, profiled_steps, 'qwen_forward_s'):.3f} "
                            f"decode={_mean_timing(timing_stats, profiled_steps, 'heatmap_decode_s'):.3f}) "
                            f"bwd={_mean_timing(timing_stats, profiled_steps, 'backward_s'):.3f} "
                            f"opt={_mean_timing(timing_stats, max(global_step, 1), 'optimizer_s'):.3f}"
                            f"{' | ' + _format_decode_internal_timing(timing_stats, profiled_steps) if _format_decode_internal_timing(timing_stats, profiled_steps) else ''}"
                            f"{' | ' + _format_qwen_internal_timing(timing_stats, profiled_steps) if _format_qwen_internal_timing(timing_stats, profiled_steps) else ''}"
                        ) if enable_timing else ""
                    )
                )
                if metrics_jsonl_path is not None:
                    _append_jsonl(
                        metrics_jsonl_path,
                        {
                            "record_type": "train_step",
                            "stage": stage_name,
                            "epoch": epoch,
                            "batch": i + 1,
                            "global_step": global_step,
                            "loss": loss.item() * grad_accum_steps,
                            "heatmap_loss": heatmap_loss.item(),
                            "trajectory_loss": trajectory_loss.item(),
                            "progress_loss": progress_loss.item(),
                            "heatmap_temperature": current_heatmap_temperature,
                            "gpu_memory_gb": mem_alloc,
                            "lrs": {
                                optimizer.param_groups[gi].get("name", f"g{gi}"): lr_val
                                for gi, lr_val in enumerate(all_lrs)
                            },
                        },
                    )
                
            if tb_writer is not None:
                actual_step = global_step_offset + global_step
                tb_writer.add_scalar('train/loss', loss.item()*grad_accum_steps, actual_step)
                tb_writer.add_scalar('train/heatmap_loss', heatmap_loss.item(), actual_step)
                if isinstance(loss_dict, dict):
                    for k in ('vis_loss', 'coord_loss', 'kl_loss', 'neg_loss', 'peak_loss'):
                        if k in loss_dict:
                            tb_writer.add_scalar(f'train/hm_{k}', loss_dict[k].item(), actual_step)
                tb_writer.add_scalar('train/trajectory_loss', trajectory_loss.item(), actual_step)
                tb_writer.add_scalar('train/progress_loss', progress_loss.item(), actual_step)
                tb_writer.add_scalar('train/heatmap_temperature', current_heatmap_temperature, actual_step)
                if enable_timing:
                    tb_writer.add_scalar('timing/data_wait_s', _mean_timing(timing_stats, profiled_steps, 'data_wait_s'), actual_step)
                    tb_writer.add_scalar('timing/gt_s', _mean_timing(timing_stats, profiled_steps, 'gt_s'), actual_step)
                    tb_writer.add_scalar('timing/forward_s', _mean_timing(timing_stats, profiled_steps, 'forward_s'), actual_step)
                    tb_writer.add_scalar('timing/backward_s', _mean_timing(timing_stats, profiled_steps, 'backward_s'), actual_step)
                    tb_writer.add_scalar('timing/optimizer_s', _mean_timing(timing_stats, max(global_step, 1), 'optimizer_s'), actual_step)
                    if profiled_steps > 0:
                        tb_writer.add_scalar('timing/prepare_inputs_s', _mean_timing(timing_stats, profiled_steps, 'prepare_inputs_s'), actual_step)
                        tb_writer.add_scalar('timing/qwen_forward_s', _mean_timing(timing_stats, profiled_steps, 'qwen_forward_s'), actual_step)
                        tb_writer.add_scalar('timing/heatmap_decode_s', _mean_timing(timing_stats, profiled_steps, 'heatmap_decode_s'), actual_step)
                        for key in ('decode_vit_fusion_s', 'decode_llm_fusion_s', 'decode_coarse_s', 'decode_fine_s', 'decode_post_s'):
                            if key in timing_stats:
                                tb_writer.add_scalar(f'timing/{key}', _mean_timing(timing_stats, profiled_steps, key), actual_step)
                        for key in sorted(timing_stats.keys()):
                            if key.startswith('qwen_') and key not in {
                                'qwen_forward_s',
                            }:
                                tb_writer.add_scalar(f'timing/{key}', _mean_timing(timing_stats, profiled_steps, key), actual_step)
                for gi, lr_val in enumerate(scheduler.get_last_lr()):
                    gname = optimizer.param_groups[gi].get('name', f'g{gi}')
                    tb_writer.add_scalar(f'lr/{gname}', lr_val, actual_step)
                
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
                    # 热力图输出诊断 — 在 softmax 概率空间评估
                    if 'heatmaps' in output and output['heatmaps'] is not None:
                        pred_hm_raw = output['heatmaps'].detach()
                        
                        pred_hm_raw = _select_primary_heatmap_slice(pred_hm_raw).unsqueeze(1)
                        gt_hm_for_diag = gt_heatmap
                        gt_hm_for_diag = _select_primary_heatmap_slice(gt_hm_for_diag)
                        
                        # Convert to softmax probability (aligned with CE training objective)
                        _B, _C, _H, _W = pred_hm_raw.shape
                        _logits = torch.logit(pred_hm_raw.float().clamp(1e-6, 1 - 1e-6))
                        pred_hm = torch.softmax(_logits.reshape(_B, _C, -1), dim=-1).reshape(_B, _C, _H, _W)
                        
                        pred_mean = pred_hm.mean().item()
                        pred_max = pred_hm.max().item()
                        pred_std = pred_hm.std().item()
                        # Sigmoid-space stats for reference
                        sig_max = pred_hm_raw.max().item()
                        
                        tb_writer.add_scalar('diag/pred_heatmap_mean', pred_mean, actual_step)
                        tb_writer.add_scalar('diag/pred_heatmap_max', pred_max, actual_step)
                        tb_writer.add_scalar('diag/pred_heatmap_std', pred_std, actual_step)
                        tb_writer.add_scalar('diag/pred_sigmoid_max', sig_max, actual_step)
                        
                        gt_mean = gt_hm_for_diag.mean().item()
                        gt_max = gt_hm_for_diag.max().item()
                        
                        # softmax max 的健康基线 ≈ 1/4096 = 0.00024（均匀分布）
                        # 正常训练的模型应显著高于此值
                        uniform_baseline = 1.0 / (_H * _W)
                        peak_ratio = pred_max / uniform_baseline if uniform_baseline > 0 else 0
                        logger.info(f"[DIAG-HM] softmax: max={pred_max:.6f} ({peak_ratio:.1f}× uniform), sig_max={sig_max:.4f}")
                        logger.info(f"[DIAG-HM] gt:      mean={gt_mean:.4f}, max={gt_max:.4f}")
                        
                        if peak_ratio < 2.0:
                            logger.warning(f"[DIAG-HM] ⚠️ softmax 分布接近均匀！peak_ratio={peak_ratio:.1f}×")
                        
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
                    
                    # Visibility gate 诊断
                    if 'visibility' in output and output['visibility'] is not None:
                        pred_vis_logits = output['visibility'].detach()
                        gt_vis_batch = batch.get('gt_visibility')
                        if gt_vis_batch is None and gt_heatmap is not None:
                            gt_vis_batch = (gt_heatmap.amax(dim=(-2, -1)) > 0).float()
                        if gt_vis_batch is not None:
                            gt_vis_flat = gt_vis_batch.to(pred_vis_logits.device).reshape(-1)
                            pred_vis_prob = torch.sigmoid(pred_vis_logits.float()).reshape(-1)
                            pred_vis_bin = (pred_vis_prob > 0.5).float()
                            gt_vis_bin = (gt_vis_flat > 0.5).float()
                            total_n = gt_vis_bin.numel()
                            if total_n > 0:
                                tp = ((pred_vis_bin == 1) & (gt_vis_bin == 1)).sum().item()
                                tn = ((pred_vis_bin == 0) & (gt_vis_bin == 0)).sum().item()
                                fp = ((pred_vis_bin == 1) & (gt_vis_bin == 0)).sum().item()
                                fn = ((pred_vis_bin == 0) & (gt_vis_bin == 1)).sum().item()
                                accuracy = (tp + tn) / total_n
                                precision = tp / max(tp + fp, 1)
                                recall = tp / max(tp + fn, 1)
                                tnr = tn / max(tn + fp, 1)
                                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                                tb_writer.add_scalar('diag/vis_accuracy', accuracy, actual_step)
                                tb_writer.add_scalar('diag/vis_precision', precision, actual_step)
                                tb_writer.add_scalar('diag/vis_recall', recall, actual_step)
                                tb_writer.add_scalar('diag/vis_tnr', tnr, actual_step)
                                tb_writer.add_scalar('diag/vis_f1', f1, actual_step)
                                tb_writer.add_scalar('diag/vis_pred_mean', pred_vis_prob.mean().item(), actual_step)
                                pos_ratio = gt_vis_bin.mean().item()
                                tb_writer.add_scalar('diag/vis_gt_pos_ratio', pos_ratio, actual_step)
                                logger.info(
                                    f"[DIAG-VIS] acc={accuracy:.3f} prec={precision:.3f} "
                                    f"recall={recall:.3f} TNR={tnr:.3f} F1={f1:.3f} "
                                    f"(gt_pos={pos_ratio:.2f})"
                                )

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
                    tb_writer.add_scalar('diag/gpu_memory_gb', torch.cuda.memory_allocated() / 1024**3, actual_step)
                    tb_writer.add_scalar('diag/gpu_memory_reserved_gb', torch.cuda.memory_reserved() / 1024**3, actual_step)
                
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
                model=model_module,
                batch=batch,
                output=output,
                epoch=epoch,
                step=global_step,
                output_dir=vis_dir if vis_dir else Path('.'),
                num_samples=2,
                gt_heatmap_override=gt_heatmap if _should_use_gpu_gt(batch, gpu_heatmap_computer) else None,
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
        
        _iter_loss = loss.item() * grad_accum_steps
        _iter_hm = heatmap_loss.item()
        _iter_traj = action_total_loss.item()
        _iter_stop = stop_total_loss.item()

        total_loss += _iter_loss
        total_heatmap_loss += _iter_hm
        total_action_loss += _iter_traj
        total_stop_loss += _iter_stop
        num_batches += 1

        del output, loss, heatmap_loss, gt_heatmap
        del action_total_loss, stop_total_loss
        del trajectory_loss, action_loss, progress_loss, stop_loss
        loss_dict = None
        del video_frames
        del current_views_batch, history_panoramas_batch
        del panoramic_inputs_batch, panoramic_num_histories, panoramic_text_anchor_positions
        del history_frames, current_frame, gt_action, action_valid, is_stop, text
        del batch

        pbar.set_postfix({
            'loss': f"{_iter_loss:.4f}",
            'hm': f"{total_heatmap_loss / num_batches:.4f}",
            'traj': f"{_iter_traj:.4f}",
        })

        if num_batches % 50 == 0:
            gc.collect()
            _malloc_trim()
            post_rss = _mem_log_proc.memory_info().rss / (1024 * 1024)
            gc_stats = gc.get_stats()
            cg_now = _cgroup_mem_usage_gb()
            cg_str = f"cgroup={cg_now:.1f}/{_cg_limit:.0f}GB" if _cg_limit > 0 else f"cgroup={cg_now:.1f}GB"
            print(
                f"[MAIN batch={i} post-gc] rss={post_rss:.0f}MB | {cg_str} | "
                f"gc: gen0={gc_stats[0]['collected']} gen1={gc_stats[1]['collected']} "
                f"gen2={gc_stats[2]['collected']}",
                file=sys.stderr,
                flush=True,
            )

            _drop_page_cache()

            if num_batches % 200 == 0:
                torch.cuda.empty_cache()
                if tb_writer is not None:
                    tb_writer.flush()

        if (
            ckpt_manager is not None
            and mid_epoch_save_every > 0
            and num_batches > 0
            and num_batches % mid_epoch_save_every == 0
            and dist_context.is_main
        ):
            model_module_for_save = _unwrap_model(model)
            mid_metrics = {
                'total_loss': total_loss / num_batches,
                'heatmap_loss': total_heatmap_loss / num_batches,
                'action_loss': total_action_loss / num_batches,
                'stop_loss': total_stop_loss / num_batches,
            }
            if ema is not None:
                with ema.apply():
                    ckpt_manager.save(
                        model=model_module_for_save,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        stage_idx=stage_idx,
                        stage_name=stage_name,
                        metrics=mid_metrics,
                        cfg=cfg,
                        is_best=False,
                        scaler=scaler,
                        batch=num_batches,
                    )
            else:
                ckpt_manager.save(
                    model=model_module_for_save,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    metrics=mid_metrics,
                    cfg=cfg,
                    is_best=False,
                    scaler=scaler,
                    batch=num_batches,
                )
            logger.info(
                f"  Mid-epoch checkpoint saved at batch {num_batches} "
                f"(loss={mid_metrics['total_loss']:.4f})"
            )

        prev_step_end = time.perf_counter()
    
    # 处理剩余梯度
    remaining = valid_batch_count % grad_accum_steps
    if remaining > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg['grad_clip'])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if ema is not None:
            ema.update()
        global_step += 1
        hm_loss_fn.set_temperature(
            get_heatmap_temperature(
                cfg,
                global_step_offset + global_step,
                total_train_steps,
            )
        )
    
    totals = torch.tensor(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_stop_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)

    reduced_num_batches = max(int(totals[4].item()), 1)
    return {
        'total_loss': (totals[0] / reduced_num_batches).item(),
        'heatmap_loss': (totals[1] / reduced_num_batches).item(),
        'action_loss': (totals[2] / reduced_num_batches).item(),
        'stop_loss': (totals[3] / reduced_num_batches).item(),
        'optimizer_steps': global_step,
        'heatmap_temperature': hm_loss_fn.temperature,
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
    vis_dir: Optional[Path] = None,
    max_batches: int = None,
    gpu_heatmap_computer: Optional[GPUHeatmapComputer] = None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    heatmap_temperature: Optional[float] = None,
    dist_context: Optional[DistributedContext] = None,
) -> Dict[str, float]:
    """验证（带可视化）"""
    dist_context = dist_context or DistributedContext(
        enabled=False,
        device=torch.device(cfg['model'].get('device', 'cuda')),
    )
    model_module = _unwrap_model(model)
    model.eval()
    
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_action_loss = 0.0
    total_stop_loss = 0.0
    total_heatmap_mse = 0.0      # 完整推理的 heatmap MSE（采样）
    num_heatmap_mse_batches = 0  # MSE 采样 batch 计数
    num_batches = 0
    vis_tp = vis_tn = vis_fp = vis_fn = 0
    
    loss_cfg = cfg['loss']
    train_history = stage_cfg.get('train_history', True)
    train_future = stage_cfg.get('train_future', False)
    train_action = stage_cfg.get('train_action', True)
    loss_type = stage_cfg.get('heatmap_loss_type', 'simplified')
    
    device = dist_context.device
    
    from src.models.heatmap import HeatmapVLNLoss
    hm_loss_fn = HeatmapVLNLoss(
        lambda_vis=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_vis', 1.0),
        lambda_coord=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_coord', 1.0),
        lambda_kl=cfg.get('loss', {}).get('heatmap_vln', {}).get(
            'lambda_kl',
            cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_pos', 1.0),
        ),
        lambda_neg=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_neg', 1.0),
        lambda_peak=cfg.get('loss', {}).get('heatmap_vln', {}).get('lambda_peak', 1.0),
        temperature=heatmap_temperature if heatmap_temperature is not None else cfg.get('loss', {}).get('heatmap_vln', {}).get('temperature', 1.0),
        heatmap_size=tuple(cfg['model'].get('heatmap', {}).get('heatmap_size', cfg['data']['init_hm_size'])),
    ).to(device)
    
    # 验证推理 batch 数限制：只对前 N 个 batch 做完整推理计算 heatmap MSE
    val_inference_batches = cfg.get('validation', {}).get('val_inference_batches', 10)
    
    total_val_batches = len(val_loader)
    if max_batches is not None:
        total_val_batches = min(total_val_batches, max_batches)
        logger.info(f"  ⚡ 快速调试模式(验证): 只处理 {total_val_batches} batches")
    
    logger.info(f"  📊 验证: {total_val_batches} batches (training loss), "
                f"{val_inference_batches} batches (推理 MSE)")
    logger.info(f"  🌡️ Heatmap temperature: {hm_loss_fn.temperature:.3f}")
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating", total=total_val_batches, disable=not dist_context.is_main)):
            if max_batches is not None and i >= max_batches:
                break
            history_frames = batch['history_frames']
            current_frame = batch['current_frame']
            B, K, C, H, W = history_frames.shape
            
            gt_action = batch['action'].to(device)
            action_valid = batch['action_valid'].to(device)
            is_stop = batch['is_stop'].to(device)
            text = batch['text']
            
            # GPU 热力图计算（如果启用）
            if _should_use_gpu_gt(batch, gpu_heatmap_computer):
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
                if text and len(text) > 0:
                    instruction_text = list(text)
                else:
                    instruction_text = None
                current_views_batch = batch.get('current_views')
                history_panoramas_batch = batch.get('history_panoramas')
                panoramic_inputs_batch = batch.get('pano_inputs')
                panoramic_num_histories = batch.get('pano_num_histories')
                panoramic_text_anchor_positions = batch.get('pano_text_anchor_positions')
                if panoramic_inputs_batch is not None and not train_action:
                    video_frames = current_frame.unsqueeze(1)
                else:
                    video_frames = torch.cat([
                        history_frames,
                        history_frames[:, -1:],
                    ], dim=1)
                
                output = model(
                    video_frames=video_frames,
                    instruction_text=instruction_text,
                    current_observation=current_frame,
                    current_views=current_views_batch,
                    history_panoramas=history_panoramas_batch,
                    panoramic_inputs=panoramic_inputs_batch,
                    panoramic_num_histories=panoramic_num_histories,
                    panoramic_text_anchor_positions=panoramic_text_anchor_positions,
                    return_heatmaps=True,
                    return_actions=train_action,
                    gt_actions=gt_action.unsqueeze(1) if train_action else None,
                    action_valid=action_valid if train_action else None,
                    gt_stop=is_stop if train_action else None,
                    gt_history_heatmap=gt_heatmap if train_history else None,
                )
            
            # Heatmap loss (v2)
            heatmap_loss = torch.tensor(0.0, device=device)
            if train_history and 'visibility' in output and 'heatmaps' in output:
                if gt_heatmap is not None:
                    if 'gt_visibility' in batch:
                        gt_vis = batch['gt_visibility'].to(device)
                    else:
                        gt_vis = gt_heatmap.amax(dim=(-2, -1)).clamp(0, 1).to(device)
                    hm_history_mask = batch.get('history_mask')
                    if hm_history_mask is not None:
                        hm_history_mask = hm_history_mask.to(device)
                    loss_dict = hm_loss_fn(
                        output['visibility'],
                        output['heatmaps'],
                        gt_vis=gt_vis,
                        gt_heatmaps=gt_heatmap.to(device),
                        history_mask=hm_history_mask,
                    )
                    heatmap_loss = loss_dict['total']

            # Visibility gate 累积统计
            if 'visibility' in output and output['visibility'] is not None:
                pred_vis_logits = output['visibility'].detach()
                gt_vis_batch = batch.get('gt_visibility')
                if gt_vis_batch is None and gt_heatmap is not None:
                    gt_vis_batch = (gt_heatmap.amax(dim=(-2, -1)) > 0).float()
                if gt_vis_batch is not None:
                    pv = (torch.sigmoid(pred_vis_logits.float()).reshape(-1) > 0.5).float()
                    gv = (gt_vis_batch.to(pred_vis_logits.device).reshape(-1) > 0.5).float()
                    vis_tp += ((pv == 1) & (gv == 1)).sum().item()
                    vis_tn += ((pv == 0) & (gv == 0)).sum().item()
                    vis_fp += ((pv == 1) & (gv == 0)).sum().item()
                    vis_fn += ((pv == 0) & (gv == 1)).sum().item()

            # Action Loss / Trajectory Loss (验证)
            action_loss = torch.tensor(0.0, device=device)
            trajectory_loss = torch.tensor(0.0, device=device)
            
            if train_action:
                # Transformer Action Head (new) - 使用 trajectory
                if hasattr(model_module, 'transformer_action_head') and model_module.transformer_action_head is not None:
                    if 'trajectory' in batch:
                        gt_trajectory = batch['trajectory'].to(device)
                        trajectory_valid = batch['trajectory_valid'].to(device)
                        # 传入完整 llm_tokens 序列
                        traj_result = model_module.transformer_action_head.compute_loss(
                            output['llm_tokens'],
                            gt_trajectory,
                            trajectory_valid,
                        )
                        trajectory_loss = traj_result['loss']
                # Legacy Action Head
                elif hasattr(model_module, 'action_head') and model_module.action_head is not None and 'action_cond' in output:
                    action_result = model_module.action_head.compute_loss(
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
                if hasattr(model_module, 'progress_head') and model_module.progress_head is not None:
                    if 'progress' in batch:
                        gt_progress = batch['progress'].to(device)
                        # 使用 trajectory_valid 作为 mask（更准确）或 action_valid 作为备选
                        progress_valid = batch.get('trajectory_valid', action_valid).to(device)
                        progress_result = model_module.progress_head(
                            output['llm_tokens'],  # 传入完整序列 (B, seq_len, D)
                            gt_progress=gt_progress,
                            action_valid=progress_valid,
                            return_loss=True,
                        )
                        progress_loss = progress_result['loss']
                # Legacy Stop Head
                elif hasattr(model_module, 'stop_head') and model_module.stop_head is not None and 'stop_logits' in output:
                    stop_loss = model_module.stop_head.compute_loss(
                        output['stop_logits'],
                        is_stop,
                        action_valid
                    )
            
            heatmap_weight = loss_cfg.get('heatmap_weight', 1.0)
            trajectory_weight = loss_cfg.get('trajectory_weight', 1.0)
            progress_weight = loss_cfg.get('progress_weight', 0.5)
            
            action_total_loss = trajectory_loss if trajectory_loss.item() > 0 else action_loss
            stop_total_loss = progress_loss if progress_loss.item() > 0 else stop_loss
            
            loss = heatmap_weight * heatmap_loss + trajectory_weight * action_total_loss + progress_weight * stop_total_loss
        
            total_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_action_loss += action_total_loss.item()  # trajectory or action
            total_stop_loss += stop_total_loss.item()      # progress or stop
            num_batches += 1
            
            # ==================== 复用当前 output 做推理 MSE + 可视化 ====================
            num_vis_batches = cfg['log'].get('val_vis_batches', 2)
            if num_batches <= val_inference_batches:
                try:
                    vis_output = output
                    if train_history and 'heatmaps' in vis_output:
                        infer_pred_hm = vis_output.get('heatmaps_gated', vis_output['heatmaps']).to(device)
                        gt_hm_eval = gt_heatmap.to(device)
                        if infer_pred_hm.shape[-2:] != gt_hm_eval.shape[-2:]:
                            orig = infer_pred_hm.shape
                            infer_pred_hm = F.interpolate(
                                infer_pred_hm.reshape(-1, 1, *orig[-2:]),
                                size=gt_hm_eval.shape[-2:],
                                mode='bilinear', align_corners=False,
                            ).reshape(*orig[:-2], *gt_hm_eval.shape[-2:])
                        hm_mask = batch.get('history_mask')
                        mask_usable = (
                            hm_mask is not None
                            and infer_pred_hm.dim() >= 4
                            and tuple(hm_mask.shape) == tuple(infer_pred_hm.shape[:2])
                        )
                        if mask_usable:
                            m = hm_mask.to(device).float()
                            while m.dim() < infer_pred_hm.dim():
                                m = m.unsqueeze(-1)
                            m = m.expand_as(infer_pred_hm)
                            sq_err = (infer_pred_hm - gt_hm_eval).square()
                            batch_mse = (sq_err * m).sum() / m.sum().clamp(min=1)
                        else:
                            batch_mse = F.mse_loss(infer_pred_hm, gt_hm_eval)
                        total_heatmap_mse += batch_mse.item()
                        num_heatmap_mse_batches += 1
                    
                    if dist_context.is_main and num_batches <= num_vis_batches and vis_dir is not None:
                        vis_path = visualize_heatmap_predictions(
                            model=model_module,
                            batch=batch,
                            output=vis_output,
                            epoch=epoch,
                            step=num_batches,
                            output_dir=vis_dir,
                            num_samples=4,
                            gt_heatmap_override=gt_heatmap if _should_use_gpu_gt(batch, gpu_heatmap_computer) else None,
                        )
                        
                        if vis_path is not None:
                            if tb_writer is not None:
                                vis_img = cv2.imread(str(vis_path))
                                if vis_img is not None:
                                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                    vis_img = vis_img.transpose(2, 0, 1)
                                    tb_writer.add_image(f'val/heatmap_viz_batch{num_batches}', vis_img, epoch)
                            
                            logger.info(f"[VAL-VIS] Epoch {epoch}, Batch {num_batches} visualization saved")
                except Exception as e:
                    logger.warning(f"Validation inference/visualization failed: {e}")

            del output, gt_heatmap
    
    totals = torch.tensor(
        [
            total_loss,
            total_heatmap_loss,
            total_action_loss,
            total_stop_loss,
            total_heatmap_mse,
            float(num_batches),
            float(num_heatmap_mse_batches),
            float(vis_tp),
            float(vis_tn),
            float(vis_fp),
            float(vis_fn),
        ],
        device=device,
        dtype=torch.float64,
    )
    _dist_all_reduce_in_place(totals)

    reduced_num_batches = max(int(totals[5].item()), 1)
    reduced_num_heatmap_mse_batches = int(totals[6].item())
    avg_loss = (totals[0] / reduced_num_batches).item()
    avg_hm = (totals[1] / reduced_num_batches).item()
    avg_act = (totals[2] / reduced_num_batches).item()
    avg_stop = (totals[3] / reduced_num_batches).item()
    avg_hm_mse = (totals[4] / max(reduced_num_heatmap_mse_batches, 1)).item() if reduced_num_heatmap_mse_batches > 0 else 0.0

    r_tp, r_tn, r_fp, r_fn = totals[7].item(), totals[8].item(), totals[9].item(), totals[10].item()
    vis_total = r_tp + r_tn + r_fp + r_fn
    val_vis_metrics = {}
    if vis_total > 0:
        val_vis_metrics['val_vis_accuracy'] = (r_tp + r_tn) / vis_total
        val_vis_metrics['val_vis_precision'] = r_tp / max(r_tp + r_fp, 1)
        val_vis_metrics['val_vis_recall'] = r_tp / max(r_tp + r_fn, 1)
        val_vis_metrics['val_vis_tnr'] = r_tn / max(r_tn + r_fp, 1)
        p, r = val_vis_metrics['val_vis_precision'], val_vis_metrics['val_vis_recall']
        val_vis_metrics['val_vis_f1'] = 2 * p * r / max(p + r, 1e-8)
        val_vis_metrics['val_vis_gt_pos_ratio'] = (r_tp + r_fn) / vis_total
        logger.info(
            f"  📊 Visibility gate: acc={val_vis_metrics['val_vis_accuracy']:.3f} "
            f"prec={val_vis_metrics['val_vis_precision']:.3f} "
            f"recall={val_vis_metrics['val_vis_recall']:.3f} "
            f"TNR={val_vis_metrics['val_vis_tnr']:.3f} "
            f"F1={val_vis_metrics['val_vis_f1']:.3f} "
            f"(gt_pos={val_vis_metrics['val_vis_gt_pos_ratio']:.2f})"
        )
    
    # 注意：TensorBoard 记录移至主循环中使用 global_epoch_counter
    # 避免多阶段训练时 epoch 重复导致数据覆盖
    
    if reduced_num_heatmap_mse_batches > 0:
        logger.info(f"  📊 Heatmap 推理 MSE (采样 {reduced_num_heatmap_mse_batches} batches): {avg_hm_mse:.6f}")
    
    result = {
        'val_loss': avg_loss,
        'val_heatmap_loss': avg_hm,
        'val_heatmap_mse': avg_hm_mse,
        'val_action_loss': avg_act,
        'val_stop_loss': avg_stop,
        'val_total_loss': avg_loss,
    }
    result.update(val_vis_metrics)
    return result


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
        batch: int = None,
    ) -> Path:
        """保存检查点。batch 不为 None 时保存为 mid-epoch checkpoint。"""
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
        """清理旧的检查点"""
        ckpts = sorted(self.out_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
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
        if latest.exists():
            return latest
        legacy_latest = self.out_dir.parent / "ckpts" / "latest.pth"
        return legacy_latest if legacy_latest.exists() else None
    
    def get_best(self) -> Optional[Path]:
        """获取最佳检查点路径"""
        best = self.out_dir / "best.pth"
        if best.exists():
            return best
        legacy_best = self.out_dir.parent / "ckpts" / "best.pth"
        return legacy_best if legacy_best.exists() else None


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


def _make_json_safe(value: Any) -> Any:
    """Convert common training objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(payload), f, indent=2, ensure_ascii=False)


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **_make_json_safe(payload)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_symlink(link_path: Path, target: Any) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _run_git_command(project_dir: Path, args: List[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _capture_git_state(project_dir: Path) -> Dict[str, Any]:
    commit = _run_git_command(project_dir, ["rev-parse", "HEAD"])
    short_commit = _run_git_command(project_dir, ["rev-parse", "--short", "HEAD"])
    branch = _run_git_command(project_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_short = _run_git_command(project_dir, ["status", "--short"])
    return {
        "commit": commit or None,
        "short_commit": short_commit or None,
        "branch": branch or None,
        "is_dirty": bool(status_short),
        "status_short": status_short.splitlines() if status_short else [],
    }


def _capture_env_state(
    args: argparse.Namespace,
    run_dir: Path,
    cfg: Dict[str, Any],
    is_resuming: bool,
) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "is_resuming": is_resuming,
        "argv": sys.argv,
        "config_path": args.config,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "configured_device": cfg.get("model", {}).get("device", "cuda"),
        "timestamp": datetime.now().isoformat(),
    }


def _find_resume_checkpoint(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "checkpoints" / "latest.pth",
        run_dir / "ckpts" / "latest.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="VLN 训练脚本（单阶段）")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                        help='从检查点恢复（路径或 "latest"）')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='仅加载模型权重（不恢复 optimizer/scheduler/epoch），用于加载预训练权重后从头训练')
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
    parser.add_argument('--distributed', action='store_true',
                        help='启用 DDP（需配合 torchrun；也可在配置 gpu.multi_gpu.enabled 中开启）')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    if args.distributed:
        cfg.setdefault('gpu', {}).setdefault('multi_gpu', {})['enabled'] = True

    dist_context = init_distributed_context(cfg)
    cfg.setdefault('model', {})['device'] = str(dist_context.device)
    set_seed(cfg['seed'])
    
    # ==================== 输出目录结构（每次训练独立文件夹）====================
    base_out_dir = Path(cfg['log']['out_dir'])
    if dist_context.is_main:
        base_out_dir.mkdir(parents=True, exist_ok=True)

    latest_link = base_out_dir / 'latest'
    run_dir = None
    is_resuming = False
    if dist_context.is_main:
        if args.auto_resume and latest_link.exists():
            run_dir = latest_link.resolve() if latest_link.is_symlink() else latest_link
            if _find_resume_checkpoint(run_dir) is not None:
                is_resuming = True
                print(f"🔄 断点续训: 继续使用 {run_dir.name}")
            else:
                run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_dir = base_out_dir / f'run_{run_timestamp}'
                print(f"📁 新训练: {run_dir.name}")
        else:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = base_out_dir / f'run_{run_timestamp}'
    if dist_context.enabled:
        shared = [str(run_dir) if run_dir is not None else "", bool(is_resuming)]
        dist.broadcast_object_list(shared, src=0)
        run_dir = Path(shared[0])
        is_resuming = bool(shared[1])
    elif run_dir is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = base_out_dir / f'run_{run_timestamp}'

    manifest_dir = run_dir / 'manifest'
    logs_dir = run_dir / 'logs'
    ckpt_dir = run_dir / 'checkpoints'
    vis_train_dir = run_dir / 'visualizations' / 'train'
    vis_val_dir = run_dir / 'visualizations' / 'val'
    plots_dir = run_dir / 'plots'
    tb_run_dir = run_dir / 'tensorboard'
    metrics_jsonl_path = logs_dir / 'metrics.jsonl'

    if dist_context.is_main:
        for d in [manifest_dir, logs_dir, ckpt_dir, vis_train_dir, vis_val_dir, plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        _safe_symlink(latest_link, run_dir.name)
    _dist_barrier()

    logger = setup_logger(
        name=f"train.{run_dir.name}.r{dist_context.rank}",
        level=cfg['log'].get('log_level', 'INFO') if dist_context.is_main else 'WARNING',
        log_file=str(logs_dir / 'train.log') if dist_context.is_main else None,
    )
    if dist_context.is_main:
        logger.info(f"📁 Output: {run_dir}")
        logger.info(f"   Latest: {latest_link} → {run_dir.name}")
        logger.info(f"   Logs: {logs_dir}")
        logger.info(f"   Checkpoints: {ckpt_dir}")
        if dist_context.enabled:
            logger.info(
                f"   DDP: enabled=True, world_size={dist_context.world_size}, local_rank={dist_context.local_rank}, device={dist_context.device}"
            )

        _write_yaml(manifest_dir / "config.yaml", cfg)
        _write_json(manifest_dir / "args.json", vars(args))
        _write_json(manifest_dir / "git.json", _capture_git_state(project_root))
        _write_json(
            manifest_dir / "env.json",
            _capture_env_state(args=args, run_dir=run_dir, cfg=cfg, is_resuming=is_resuming),
        )
        _append_jsonl(
            metrics_jsonl_path,
            {
                "record_type": "run_start",
                "run_name": run_dir.name,
                "is_resuming": is_resuming,
                "output_dir": str(run_dir),
                "distributed": dist_context.enabled,
                "world_size": dist_context.world_size,
            },
        )

    # ==================== TensorBoard ====================
    tb_writer = None
    if dist_context.is_main and cfg['log'].get('use_tensorboard', False):
        tb_base_cfg = cfg['log'].get('tensorboard_dir')
        live_tb_dir = Path(tb_base_cfg) if tb_base_cfg else tb_run_dir
        if not is_resuming:
            _clear_directory(live_tb_dir)
        else:
            live_tb_dir.mkdir(parents=True, exist_ok=True)

        _safe_symlink(tb_run_dir, live_tb_dir)
        tb_writer = SummaryWriter(log_dir=str(live_tb_dir))
        logger.info(f"📊 TensorBoard: {tb_run_dir}")
        logger.info(f"   实时监控目录: {live_tb_dir}")
        logger.info(f"   autodl入口: tensorboard --logdir {live_tb_dir}")
    if not dist_context.is_main:
        metrics_jsonl_path = None
    
    loss_cfg = cfg['loss']
    default_loss_type = loss_cfg.get('heatmap_loss_type', 'simplified')
    
    logger.info("=" * 60)
    logger.info("VLN 训练 (Qwen3.5)")
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
        load_history_frames = sw_cfg.get('load_history_frames', True)
        
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
            load_history_frames=load_history_frames,
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
            load_history_frames=load_history_frames,
        )
    
    # 验证集使用固定 epoch=0，确保每次验证样本一致
    if hasattr(val_dataset, 'set_epoch'):
        val_dataset.set_epoch(0)
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    
    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg, verbose=dist_context.is_main)
    
    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        out_dir=str(ckpt_dir),
        max_ckpts=cfg['log'].get('max_ckpts', 3)
    )
    
    # 创建通知器
    notifier = create_notifier(cfg) if dist_context.is_main else None
    
    # 创建训练曲线绘制器
    plotter = TrainingPlotter(out_dir=plots_dir) if dist_context.is_main else None
    
    # 仅加载模型权重（不恢复训练状态）
    if args.load_weights:
        weights_path = Path(args.load_weights)
        if weights_path.exists():
            ckpt = torch.load(str(weights_path), map_location='cpu')
            state_dict = ckpt.get('trainable_state_dict', {})
            if state_dict:
                missing, unexpected, loaded_count = _load_normalized_state_dict(model, state_dict)
                logger.info(f"✓ Loaded {loaded_count} params from {weights_path.name} (weights only, fresh optimizer/scheduler)")
                if missing:
                    logger.info(f"  Missing keys: {len(missing)}")
                if unexpected:
                    logger.info(f"  Unexpected keys: {len(unexpected)}")
            else:
                logger.warning(f"⚠ No trainable_state_dict found in {weights_path}")
            del ckpt
            torch.cuda.empty_cache()
        else:
            logger.error(f"✗ Weights file not found: {weights_path}")
    
    # 断点续训
    resume_epoch = 0
    resume_path = None
    
    if args.resume:
        if args.resume == 'latest':
            resume_path = _find_resume_checkpoint(run_dir) or ckpt_manager.get_latest()
        else:
            resume_path = Path(args.resume)
    elif args.auto_resume:
        resume_path = _find_resume_checkpoint(run_dir) or ckpt_manager.get_latest()
    
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
        raise ValueError(
            "Qwen3.5 路径已移除 Sequence Packing 兼容代码，请在配置中设置 "
            "model.llm.enable_packing=false。"
        )
    actual_collate_fn = collate_fn
    use_panoramic_tokenized_collator = (
        cfg['model'].get('heatmap', {}).get('enable', True)
        and getattr(train_dataset, '_is_panoramic', False)
        and getattr(val_dataset, '_is_panoramic', False)
    )
    if use_panoramic_tokenized_collator:
        from transformers import AutoProcessor

        llm_model_path = cfg['model'].get('llm', {}).get('model_path', './models/qwen_3.5')
        logger.info("🔄 Loading Qwen processor for panoramic worker-side tokenization...")
        pano_processor = AutoProcessor.from_pretrained(llm_model_path, trust_remote_code=True)
        actual_collate_fn = PanoramicTokenizedCollator(pano_processor)
        logger.info("   ✅ Panoramic tokenized collator enabled")
    
    # 🔧 使用 fork 模式（而非 spawn），利用 copy-on-write 共享内存
    # TOKENIZERS_PARALLELISM=false 已设置，fork 安全
    # spawn 模式每个 worker 独立复制 processor+dataset ≈ 10GB/worker
    # fork 模式 workers 共享父进程内存（COW），仅 ~1-2GB/worker
    # 在 90GB Docker 容器下这是关键差异
    mp_context = 'fork' if num_workers > 0 else None
    
    uses_dynamic_sampling = hasattr(train_dataset, 'set_epoch')
    
    # Disable persistent workers so each epoch gets a fresh worker pool.
    # This avoids worker-side memory accumulation across epoch boundaries.
    persistent_workers = False
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist_context.world_size,
        rank=dist_context.rank,
        shuffle=True,
        drop_last=True,
    ) if dist_context.enabled else None
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist_context.world_size,
        rank=dist_context.rank,
        shuffle=False,
        drop_last=False,
    ) if dist_context.enabled else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['optim']['batch_size'],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=actual_collate_fn,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        in_order=False if num_workers > 0 else True,
    )
    
    # 验证集也需要 workers 加速 tokenization（否则主进程串行处理会极慢）
    # 使用与训练相同的 workers 配置，persistent_workers=False 保证验证结束后释放内存
    val_num_workers = min(num_workers, 4)  # 验证用 4 workers 足够
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['optim']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
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
    if dist_context.enabled:
        logger.info(
            f"   🔀 DistributedSampler enabled: world_size={dist_context.world_size}, rank={dist_context.rank}"
        )
    
    # ⚠️ 强制加载 Qwen3.5（含 LoRA），确保所有参数在 set_trainable + build_optimizer 之前就位
    # 否则 LoRA 参数（懒加载，首次前向才创建）不会被 optimizer 捕获
    raw_model = model
    if hasattr(raw_model, 'qwen3_5') and hasattr(raw_model.qwen3_5, '_load_model'):
        if raw_model.qwen3_5.model is None:
            logger.info("🔄 Pre-loading Qwen3.5 (ensure LoRA params available for optimizer)...")
            raw_model.qwen3_5._load_model()
        logger.info(
            "   🧠 Qwen attention implementation: %s",
            getattr(raw_model.qwen3_5.config, 'attn_implementation', 'unknown'),
        )
    if getattr(raw_model.config, 'enable_heatmap', False):
        logger.info("🔄 Constructing HeatmapVLN before optimizer setup...")
        raw_model._ensure_heatmap_vln()
    
    # 设置可训练模块
    logger.info("🔧 Setting trainable modules...")
    set_trainable_modules(raw_model, stage_cfg, logger)
    
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 构建优化器和调度器
    optimizer = build_optimizer(raw_model, cfg, stage_cfg)
    grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
    total_batches = len(train_loader) * total_epochs
    total_steps = total_batches // grad_accum_steps
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    # GradScaler 仅用于 fp16，bf16 不需要（动态范围更大）
    amp_type = cfg['optim'].get('amp', 'bf16')
    scaler = GradScaler() if amp_type == 'fp16' else None
    
    if resume_path and Path(resume_path).exists():
        load_checkpoint_for_resume(
            str(resume_path), raw_model, 
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
    epoch_boundary_cooldown_s = float(cfg.get('log', {}).get('epoch_boundary_cooldown_s', 0.0) or 0.0)
    
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
            device=str(dist_context.device),
        )
        gpu_depth_normalized = not getattr(train_dataset, 'depth_is_meters', False)
        gpu_has_depth = getattr(train_dataset, 'load_depth', True)
        logger.info(f"🚀 GPU heatmap computation enabled (hm_size={hm_size}, "
                     f"depth_normalized={gpu_depth_normalized}, has_depth={gpu_has_depth})")
    else:
        gpu_heatmap_computer = None
        gpu_depth_normalized = True
        gpu_has_depth = False
    
    if dist_context.enabled:
        initialize_trainable_module_sync(
            raw_model,
            stage_cfg=stage_cfg,
            dist_context=dist_context,
            logger=logger,
        )

    # EMA (Exponential Moving Average) — 扩散模型标准技术
    # 用参数的滑动平均做推理，减少训练波动，提升泛化
    ema_decay = cfg.get('optim', {}).get('ema_decay', 0.999)
    ema_warmup = cfg.get('optim', {}).get('ema_warmup_steps', 2000)
    ema = EMAModel(raw_model, decay=ema_decay, warmup_steps=ema_warmup)
    logger.info(f"📐 EMA enabled: decay={ema_decay}, warmup_steps={ema_warmup}")
    
    timer = TrainingTimer(total_epochs=total_epochs)
    timer.start()

    _drop_page_cache(force=True)
    cg_init = _cgroup_mem_usage_gb()
    logger.info(f"  cgroup memory after initial page cache drop: {cg_init:.1f}/{_CG_LIMIT_GB:.0f}GB")

    for epoch in range(start_epoch, total_epochs + 1):
        timer.start_epoch()
            
        # Clip-level 采样：每个 epoch 重新采样，减少样本相关性
        # 🔧 修复：不再重建 DataLoader，直接修改 sample_index
        # persistent_workers 下 workers 共享 dataset 对象的引用，
        # 修改 sample_index 后，下一次迭代会自动使用新的索引
        if uses_dynamic_sampling:
            train_dataset.set_epoch(epoch)
            logger.info(f"   🔄 Resampled {len(train_dataset)} samples for epoch {epoch} (persistent workers)")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        logger.info("=" * 80)
        logger.info(f"[{stage_name}] Epoch {epoch}/{total_epochs}")
        logger.info("=" * 80)
        
        epoch_offset = (epoch - 1) * steps_per_epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, logger, tb_writer, epoch_offset,
            stage_idx=0, stage_name=stage_name, stage_cfg=stage_cfg,
            max_batches=args.max_batches,
            vis_dir=vis_train_dir,
            gpu_heatmap_computer=gpu_heatmap_computer,
            gpu_has_depth=gpu_has_depth,
            gpu_depth_normalized=gpu_depth_normalized,
            ema=ema,
            metrics_jsonl_path=metrics_jsonl_path,
            total_train_steps=total_steps,
            dist_context=dist_context,
            ckpt_manager=ckpt_manager,
            mid_epoch_save_every=cfg['log'].get('mid_epoch_save_every', 500),
        )
        
        timer.end_epoch()
        
        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()
        _drop_page_cache()

        # 使用 EMA 参数进行验证（滑动平均参数更稳定，泛化更好）
        with ema.apply():
            val_metrics = validate(
                model, val_loader, cfg, logger, stage_cfg, tb_writer, epoch,
                vis_dir=vis_val_dir,
                max_batches=args.max_batches,
                gpu_heatmap_computer=gpu_heatmap_computer,
                gpu_has_depth=gpu_has_depth,
                gpu_depth_normalized=gpu_depth_normalized,
                heatmap_temperature=train_metrics.get('heatmap_temperature'),
                dist_context=dist_context,
            )
        
        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()
        _drop_page_cache()
        
        # 📊 内存使用监控
        process = psutil.Process()
        mem_info = process.memory_info()
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"  🧠 Memory: CPU={mem_info.rss / (1024**3):.2f}GB, GPU={gpu_mem:.2f}GB (reserved={gpu_reserved:.2f}GB)")
        
        logger.info(
            f"  Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f}, "
            f"traj: {train_metrics['action_loss']:.4f})"
        )
        val_hm_mse_str = f", infer_mse: {val_metrics['val_heatmap_mse']:.6f}" if val_metrics.get('val_heatmap_mse', 0) > 0 else ""
        logger.info(
            f"  Val Loss: {val_metrics['val_loss']:.4f} "
            f"(hm: {val_metrics['val_heatmap_loss']:.4f}, "
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

        if dist_context.is_main:
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "record_type": "epoch_summary",
                    "epoch": epoch,
                    "global_epoch": global_epoch_counter,
                    "stage": stage_name,
                    "is_best": is_best,
                    "learning_rate": current_lr,
                    "train": train_metrics,
                    "val": val_metrics,
                    "epoch_time": timer.get_epoch_time(),
                    "eta": eta,
                },
            )
        
        if plotter is not None:
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
            
            # 热力图损失对比
            tb_writer.add_scalars('loss/heatmap', {
                'train': train_metrics['heatmap_loss'],
                'val': val_metrics['val_heatmap_loss'],
            }, global_epoch_counter)
            
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

            for vk in ('val_vis_accuracy', 'val_vis_precision', 'val_vis_recall',
                        'val_vis_tnr', 'val_vis_f1', 'val_vis_gt_pos_ratio'):
                if vk in val_metrics:
                    tb_writer.add_scalar(f'epoch/{vk}', val_metrics[vk], global_epoch_counter)
            
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
            if dist_context.is_main:
                with ema.apply():
                    ckpt_manager.save(
                        model=raw_model,
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
            _dist_barrier()
        
        if no_improve_count >= patience:
            logger.info(f"  🛑 Early stopping")
            break

        if epoch < total_epochs and epoch_boundary_cooldown_s > 0:
            logger.info(
                f"  💤 Epoch boundary cooldown: sleep {epoch_boundary_cooldown_s:.1f}s "
                "to let workers exit and memory settle"
            )
            gc.collect()
            torch.cuda.empty_cache()
            _malloc_trim()
            time.sleep(epoch_boundary_cooldown_s)
            gc.collect()
            torch.cuda.empty_cache()
            _malloc_trim()
    
    logger.info(f"  📊 训练完成，耗时: {timer.get_total_elapsed()}")
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成！")
    logger.info("=" * 60)
    
    summary = plotter.get_summary() if plotter is not None else {}
    if summary:
        logger.info(f"📊 训练摘要:")
        logger.info(f"   总 Epochs: {summary.get('total_epochs', 'N/A')}")
        logger.info(f"   最佳 Epoch: {summary.get('best_epoch', 'N/A')}")
        if summary.get('best_val_loss'):
            logger.info(f"   最佳 val_loss: {summary.get('best_val_loss'):.4f}")

    final_summary = {
        **summary,
        "elapsed_time": timer.get_total_elapsed(),
        "best_val_loss_runtime": best_val_loss,
        "run_dir": str(run_dir),
    }
    if dist_context.is_main:
        _write_json(manifest_dir / "summary.json", final_summary)
        _append_jsonl(
            metrics_jsonl_path,
            {
                "record_type": "run_complete",
                "summary": final_summary,
                "elapsed_time": timer.get_total_elapsed(),
                "best_val_loss": best_val_loss,
            },
        )
    
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
    cleanup_distributed()


if __name__ == '__main__':
    main()
