#!/usr/bin/env python3
"""
VLN 训练脚本
==============

使用共享 Habitat/InternNav 环境进行视觉语言导航训练。
单阶段训练：History 热力图头 + Action Head + Progress Head
"""

import os
import sys
from pathlib import Path


def _apply_early_gpu_arg_for_monitor() -> None:
    """
    monitor_gpu_idle 占卡时会执行: python train.py --gpu 0,1 ...
    必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES，否则会与物理卡号不一致。
    从 sys.argv 中移除 --gpu，避免下方 argparse 报 unknown argument。
    """
    if "--gpu" not in sys.argv:
        return
    try:
        i = sys.argv.index("--gpu")
    except ValueError:
        return
    if i + 1 >= len(sys.argv):
        return
    val = sys.argv[i + 1].strip()
    if val:
        os.environ["CUDA_VISIBLE_DEVICES"] = val
    del sys.argv[i : i + 2]


_apply_early_gpu_arg_for_monitor()

# 启用 expandable_segments 减少显存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 禁用 tokenizers 并行，避免多进程 fork 冲突导致死锁
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.multiprocessing as _mp

# /dev/shm 只有 64 MB（Docker 默认值），PyTorch 默认的 file_descriptor 策略
# 底层调用 shm_open() 在 /dev/shm 创建临时文件，num_workers>0 时必然溢出。
# 切换到 file_system 策略，让 PyTorch 在 /tmp（2.6 TB）上用普通文件做 IPC。
_mp.set_sharing_strategy('file_system')

import argparse
import gc
import logging
import time
import warnings
from datetime import datetime

import psutil

# ============================================
# CUDA 性能优化
# ============================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
warnings.filterwarnings("ignore", message="Asked to sample")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

# --- All training utilities from the modular scripts.training package ---
from scripts.training import (
    _CG_LIMIT_GB,
    CheckpointManager,
    EMAModel,
    ShmBypassCollate,
    ShmBypassDataset,
    TrainingPlotter,
    TrainingTimer,
    _append_jsonl,
    _capture_env_state,
    _capture_git_state,
    _cgroup_mem_usage_gb,
    _clear_directory,
    _dist_barrier,
    _drop_page_cache,
    _find_resume_checkpoint,
    _load_normalized_state_dict,
    _malloc_trim,
    _safe_symlink,
    _worker_init_fn,
    _write_json,
    _write_yaml,
    apply_nextdit_warmup_freeze,
    build_model,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    collate_fn,
    init_distributed_context,
    initialize_trainable_module_sync,
    load_checkpoint_for_resume,
    load_config,
    make_grad_scaler,
    safe_torch_load,
    set_seed,
    set_trainable_modules,
    train_one_epoch,
)
from scripts.training.validate import validate

from src.data.factory import build_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.runtime_compat import ensure_transformers_runtime_compat
from src.utils.gpu_heatmap import GPUHeatmapComputer
from src.utils.logger import setup_logger
from src.utils.notifier import create_notifier

logger = logging.getLogger(__name__)


def _infer_base_checkpoint_from_resume(resume_path: Path, logger) -> str | None:
    """Recover the Stage 1/base checkpoint path recorded in a bridge checkpoint."""
    try:
        ckpt = safe_torch_load(str(resume_path))
    except Exception as exc:
        logger.warning("Could not inspect resume checkpoint for base weights: %s", exc)
        return None

    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    base_path = cfg.get('runtime', {}).get('base_checkpoint') if isinstance(cfg, dict) else None
    del ckpt
    if not base_path:
        return None

    resolved = Path(base_path)
    if not resolved.exists():
        logger.warning("Resume checkpoint records missing base weights: %s", resolved)
        return None
    return str(resolved)


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="VLN 训练脚本（共享 Habitat/InternNav 环境）")
    parser.add_argument('--config', type=str, default='configs/train_config_internnav.yaml',
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
    parser.add_argument('--num-workers', type=int, default=None,
                        help='覆盖 data.num_workers，适合在小 /dev/shm 环境下做 smoke test')
    parser.add_argument('--prefetch-factor', type=int, default=None,
                        help='覆盖 data.prefetch_factor')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None,
                        help='覆盖 data.pin_memory')

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    if args.distributed:
        cfg.setdefault('gpu', {}).setdefault('multi_gpu', {})['enabled'] = True
    if args.num_workers is not None:
        cfg.setdefault('data', {})['num_workers'] = args.num_workers
    if args.prefetch_factor is not None:
        cfg.setdefault('data', {})['prefetch_factor'] = args.prefetch_factor
    if args.pin_memory is not None:
        cfg.setdefault('data', {})['pin_memory'] = args.pin_memory

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

    cfg['loss']

    logger.info("=" * 60)
    logger.info("VLN 训练 (shared Habitat/InternNav env)")
    logger.info("=" * 60)

    # 构建数据集
    logger.info("📂 Loading datasets...")
    dataset_type = cfg['data'].get('dataset_type', 'sliding_window')
    logger.info(f"  Dataset type: {dataset_type}")
    val_root_cfg = cfg['data'].get('val_root')
    if val_root_cfg:
        logger.info(f"  Validation from separate root: {val_root_cfg} (split={cfg['data'].get('val_split', 'val')})")
    train_dataset = build_dataset(cfg, split='train')

    if dataset_type == 'trajectory':
        val_root = cfg['data'].get('val_root')
        if val_root:
            val_samples = cfg['data'].get('trajectory', cfg['data'].get('sliding_window', {})).get('val_samples_per_clip', 2)
            val_dataset = build_dataset(
                cfg, split=cfg['data'].get('val_split', 'val'), root=val_root,
                samples_per_clip=val_samples,
                random_subsequence=False,
                enable_trajectory_augmentation=False,
                use_subinstruction=False,
            )
        else:
            val_dataset = None
            logger.info("  No val_root configured, skipping validation dataset")
    else:
        val_root = cfg['data'].get('val_root') or cfg['data']['root']
        val_samples = cfg['data']['sliding_window'].get('val_samples_per_clip', 2)
        val_dataset = build_dataset(
            cfg, split=cfg['data'].get('val_split', 'val'), root=val_root,
            samples_per_clip=val_samples,
        )

    if val_dataset is not None and hasattr(val_dataset, 'set_epoch'):
        val_dataset.set_epoch(0)

    logger.info(f"  Train: {len(train_dataset)} samples")
    if val_dataset is not None:
        logger.info(f"  Val: {len(val_dataset)} samples")
    else:
        logger.info("  Val: disabled (no val_root)")

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

    if resume_path and Path(resume_path).exists() and not args.load_weights:
        inferred_base = _infer_base_checkpoint_from_resume(Path(resume_path), logger)
        if inferred_base:
            args.load_weights = inferred_base
            logger.info("🔗 Inferred base checkpoint from resume metadata: %s", inferred_base)

    if resume_path and Path(resume_path).exists():
        resume_info = load_checkpoint_for_resume(
            str(resume_path), model, optimizer=None, scheduler=None, logger=logger
        )
        resume_epoch = resume_info['epoch']
        ckpt_manager.best_val_loss = resume_info['best_val_loss']

    # 获取训练配置（单阶段）
    all_stages = cfg['training']['stages']
    if not all_stages:
        logger.error("❌ 配置文件中没有定义训练阶段")
        return

    stage_cfg = all_stages[0]
    stage_name = stage_cfg['name']

    if args.epochs is not None:
        stage_cfg = stage_cfg.copy()
        stage_cfg['epochs'] = args.epochs

    total_epochs = stage_cfg['epochs']

    requires_base_checkpoint = bool(
        stage_cfg.get('requires_base_checkpoint', False)
        or stage_cfg.get('bridge_only', False)
    )
    if requires_base_checkpoint and not args.load_weights:
        raise ValueError(
            "Bridge-only training requires the Stage1-S2 panoramic System2 SFT checkpoint. "
            "Pass it with --load-weights so the frozen panoramic LoRA/System2 base "
            "is loaded before the bridge checkpoint."
        )
    if requires_base_checkpoint and not Path(args.load_weights).exists():
        raise FileNotFoundError(
            f"Bridge-only base checkpoint does not exist: {args.load_weights}"
        )

    logger.info("=" * 60)
    logger.info(f"📋 训练配置: {stage_name}")
    logger.info(f"   Epochs: {total_epochs}, Heatmap Size: {stage_cfg['hm_size']}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("🧪 Dry run 模式：模型和数据构建成功")
        logger.info("=" * 60)
        return

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
    if val_dataset is not None:
        val_dataset.hm_size = hm_size
    if hasattr(model, 'update_heatmap_size'):
        model.update_heatmap_size(hm_size)

    logger.info(f"  Heatmap size: {hm_size}")

    # 构建数据加载器
    num_workers = cfg['data']['num_workers']
    prefetch_factor = cfg['data'].get('prefetch_factor', 2)

    packing_enabled = cfg['model']['llm'].get('enable_packing', False)

    if packing_enabled:
        raise ValueError(
            "当前共享环境训练路径已移除 Sequence Packing 兼容代码，请在配置中设置 "
            "model.llm.enable_packing=false。"
        )
    actual_collate_fn = collate_fn
    use_worker_tokenized_collator = stage_cfg.get(
        'use_worker_tokenized_collator',
        cfg['data'].get(
            'use_worker_tokenized_collator',
            stage_cfg.get('train_action', True)
            or stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)),
        ),
    )
    use_panoramic_tokenized_collator = (
        use_worker_tokenized_collator
        and (
            cfg['model'].get('heatmap', {}).get('enable', True)
            or stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False))
        )
        and getattr(train_dataset, '_is_panoramic', False)
        and (val_dataset is None or getattr(val_dataset, '_is_panoramic', False))
    )
    if use_panoramic_tokenized_collator:
        from transformers import AutoProcessor

        llm_cfg = cfg['model'].get('llm', {})
        llm_model_path = llm_cfg.get('model_path', './models/internnav_backbone')
        ensure_transformers_runtime_compat(
            model_path=llm_model_path,
            requested_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
            requested_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
            logger=logger,
        )
        logger.info("🔄 Loading Qwen processor for panoramic worker-side tokenization...")
        pano_processor = AutoProcessor.from_pretrained(llm_model_path, trust_remote_code=True)
        n_traj_query = cfg.get('model', {}).get('action_head', {}).get('nextdit', {}).get('n_query', 0)
        if not cfg.get('model', {}).get('action_head', {}).get('nextdit', {}).get('enabled', False):
            n_traj_query = 0
        if not stage_cfg.get('train_action', False):
            n_traj_query = 0
        train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
        actual_collate_fn = PanoramicTokenizedCollator(
            pano_processor,
            n_traj_query=n_traj_query,
            sft_mode=train_lm,
            sft_include_turns=stage_cfg.get('sft_include_turns', True),
            sft_include_forward=stage_cfg.get('sft_include_forward', True),
        )
        logger.info(
            "   ✅ Panoramic tokenized collator enabled (n_traj_query=%d, sft_mode=%s)",
            n_traj_query, train_lm,
        )
    elif getattr(train_dataset, '_is_panoramic', False) and not stage_cfg.get('train_action', True):
        logger.info("   ✅ Heatmap-only stage: using standard panoramic collate path (skip AutoProcessor worker tokenization)")

    mp_context = 'fork' if num_workers > 0 else None

    # -- /dev/shm bypass: wrap datasets + collate when workers are used ----
    # PyTorch DataLoader workers transfer tensors via shm_open() which
    # lives in /dev/shm (only 64 MB in this container).  By converting
    # tensors → numpy in the worker and back → tensors before collation,
    # data travels through the regular pickle pipe and never touches shm.
    if num_workers > 0:
        train_dataset = ShmBypassDataset(train_dataset)
        if val_dataset is not None:
            val_dataset = ShmBypassDataset(val_dataset)
        actual_collate_fn = ShmBypassCollate(actual_collate_fn)
        logger.info("   🔀 ShmBypass enabled: tensor↔numpy IPC (bypassing 64 MB /dev/shm)")

    uses_dynamic_sampling = hasattr(train_dataset, 'set_epoch')

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
    ) if (dist_context.enabled and val_dataset is not None) else None

    os.environ["HEATMAPVLN_LOG_MEMORY"] = "1" if cfg["log"].get("show_gpu_memory", False) else "0"
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
        in_order=not num_workers > 0,
    )

    if val_dataset is not None:
        val_num_workers = min(num_workers, 4)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['optim']['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_num_workers,
            pin_memory=cfg['data']['pin_memory'],
            collate_fn=actual_collate_fn,
            prefetch_factor=prefetch_factor if val_num_workers > 0 else None,
            persistent_workers=False,
        )
        logger.info(f"   📊 验证 DataLoader: num_workers={val_num_workers}, prefetch={prefetch_factor}")
    else:
        val_loader = None
        logger.info("   📊 验证 DataLoader: disabled")

    if uses_dynamic_sampling:
        if persistent_workers:
            logger.info("   ✅ Dynamic sampling enabled with persistent_workers")
        else:
            logger.info("   ✅ Dynamic sampling enabled (workers rebuilt each epoch to reclaim memory)")
    if cfg["log"].get("show_gpu_memory", False):
        logger.info(f"   🧠 Memory config: num_workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    if dist_context.enabled:
        logger.info(
            f"   🔀 DistributedSampler enabled: world_size={dist_context.world_size}, rank={dist_context.rank}"
        )

    # ⚠️ 强制加载 VLM backbone（含 LoRA），确保所有参数在 set_trainable + build_optimizer 之前就位
    raw_model = model
    vlm_backbone = getattr(raw_model, 'vlm_backbone', getattr(raw_model, 'qwen2_5_vl', None))
    if vlm_backbone is not None and hasattr(vlm_backbone, '_load_model'):
        if vlm_backbone.model is None:
            logger.info("🔄 Pre-loading VLM backbone (ensure LoRA params available for optimizer)...")
            vlm_backbone._load_model()
        logger.info(
            "   🧠 Qwen attention implementation: %s",
            getattr(vlm_backbone.config, 'attn_implementation', 'unknown'),
        )
    if getattr(raw_model.config, 'enable_heatmap', False):
        logger.info("🔄 Constructing HeatmapVLN before optimizer setup...")
        raw_model._ensure_heatmap_vln()

    if args.load_weights:
        weights_path = Path(args.load_weights)
        if weights_path.exists():
            cfg.setdefault('runtime', {})['base_checkpoint'] = str(weights_path.resolve())
            ckpt = safe_torch_load(str(weights_path))
            state_dict = ckpt.get('trainable_state_dict', {})
            loaded_count = 0
            if state_dict:
                missing, unexpected, loaded_count = _load_normalized_state_dict(raw_model, state_dict)
                logger.info(f"✓ Loaded {loaded_count} params from {weights_path.name} (weights only, fresh optimizer/scheduler)")
                if loaded_count < len(state_dict):
                    logger.warning(f"  ⚠ Only {loaded_count}/{len(state_dict)} checkpoint params matched!")
                if missing:
                    logger.info(f"  Missing keys (in model but not checkpoint): {len(missing)}")
                if unexpected:
                    logger.info(f"  Unexpected keys (in checkpoint but not model): {len(unexpected)}")
            else:
                logger.warning(f"⚠ No trainable_state_dict found in {weights_path}")
            if requires_base_checkpoint and loaded_count == 0:
                raise RuntimeError(
                    "Bridge-only training did not load any parameters from the Stage1-S2 "
                    f"base checkpoint: {weights_path}"
                )
            del ckpt
            torch.cuda.empty_cache()
        else:
            logger.error(f"✗ Weights file not found: {weights_path}")

    # 设置可训练模块
    logger.info("🔧 Setting trainable modules...")
    set_trainable_modules(raw_model, stage_cfg, logger)

    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 构建优化器和调度器
    optimizer = build_optimizer(raw_model, cfg, stage_cfg)

    nextdit_warmup_steps = apply_nextdit_warmup_freeze(raw_model, cfg, logger)

    grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
    total_batches = len(train_loader) * total_epochs
    total_steps = total_batches // grad_accum_steps
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    amp_type = cfg['optim'].get('amp', 'bf16')
    scaler = make_grad_scaler(dist_context.device, amp_type)

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
        global_epoch_counter = resume_epoch
    else:
        start_epoch = args.start_epoch
        global_epoch_counter = start_epoch - 1

    patience = cfg['validation'].get('patience', 5)
    eval_every_epochs = max(1, int(cfg.get('validation', {}).get('eval_every_epochs', 1)))
    no_improve_count = 0
    val_metrics: dict = {}
    epoch_boundary_cooldown_s = float(cfg.get('log', {}).get('epoch_boundary_cooldown_s', 0.0) or 0.0)

    # GPU 热力图计算器
    data_cfg = cfg['data']
    sliding_cfg = data_cfg.get('sliding_window', {})
    defer_heatmap_to_gpu = sliding_cfg.get('defer_heatmap_to_gpu', False)

    if defer_heatmap_to_gpu:
        hm_size = tuple(data_cfg.get('init_hm_size', [64, 64]))
        img_size = (640, 480)

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

    # EMA
    ema_decay = cfg.get('optim', {}).get('ema_decay', 0.999)
    ema_warmup = cfg.get('optim', {}).get('ema_warmup_steps', 2000)
    ema = EMAModel(raw_model, decay=ema_decay, warmup_steps=ema_warmup)
    logger.info(f"📐 EMA enabled: decay={ema_decay}, warmup_steps={ema_warmup}")

    timer = TrainingTimer(total_epochs=total_epochs)
    timer.start()

    _drop_page_cache(force=True)
    if cfg['log'].get('show_gpu_memory', False):
        cg_init = _cgroup_mem_usage_gb()
        logger.info(f"  cgroup memory after initial page cache drop: {cg_init:.1f}/{_CG_LIMIT_GB:.0f}GB")

    for epoch in range(start_epoch, total_epochs + 1):
        timer.start_epoch()

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
            nextdit_warmup_steps=nextdit_warmup_steps,
        )

        timer.end_epoch()

        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()
        _drop_page_cache()

        do_eval = val_loader is not None and ((epoch % eval_every_epochs == 0) or (epoch == total_epochs))

        if do_eval:
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
        else:
            logger.info(f"  ⏭️  跳过验证（eval_every_epochs={eval_every_epochs}，将在 epoch {epoch + eval_every_epochs - (epoch % eval_every_epochs)} 验证）")

        if cfg['log'].get('show_gpu_memory', False):
            process = psutil.Process()
            mem_info = process.memory_info()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"  🧠 Memory: CPU={mem_info.rss / (1024**3):.2f}GB, GPU={gpu_mem:.2f}GB (reserved={gpu_reserved:.2f}GB)")

        train_traj_str = f", traj: {train_metrics['trajectory_loss']:.4f}" if train_metrics.get('trajectory_loss', 0) > 0 else ""
        train_lm_str = f", lm: {train_metrics['lm_loss']:.4f}" if train_metrics.get('lm_loss', 0) > 0 else ""
        logger.info(
            f"  Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f}{train_traj_str}{train_lm_str})"
        )

        eta = timer.get_eta(epoch, total_epochs)
        logger.info(f"  ⏱️  Epoch time: {timer.get_epoch_time()} | ETA: {eta}")

        if do_eval and val_metrics:
            val_hm_mse_str = f", infer_mse: {val_metrics['val_heatmap_mse']:.6f}" if val_metrics.get('val_heatmap_mse', 0) > 0 else ""
            val_traj_str = f", traj: {val_metrics['val_trajectory_loss']:.4f}" if val_metrics.get('val_trajectory_loss', 0) > 0 else ""
            val_lm_str = f", lm: {val_metrics['val_lm_loss']:.4f}" if val_metrics.get('val_lm_loss', 0) > 0 else ""
            logger.info(
                f"  Val Loss: {val_metrics['val_loss']:.4f} "
                f"(hm: {val_metrics['val_heatmap_loss']:.4f}{val_traj_str}{val_lm_str}{val_hm_mse_str})"
            )
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                no_improve_count = 0
                logger.info(f"  ⭐ New best val_loss: {best_val_loss:.4f}")
            else:
                no_improve_count += 1
        else:
            is_best = False

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
                    "val_evaluated": do_eval,
                    "learning_rate": current_lr,
                    "train": train_metrics,
                    "val": val_metrics if do_eval else None,
                    "epoch_time": timer.get_epoch_time(),
                    "eta": eta,
                },
            )

        if plotter is not None:
            plotter.update(
                epoch=global_epoch_counter,
                stage_name=stage_name,
                train_metrics=train_metrics,
                val_metrics=val_metrics if do_eval else {},
                lr=current_lr,
                is_best=is_best,
            )

        if tb_writer is not None:
            tb_writer.add_scalar('train/lr', current_lr, global_epoch_counter)
            tb_writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], global_epoch_counter)

            if do_eval and val_metrics:
                tb_writer.add_scalars('loss/total', {
                    'train': train_metrics['total_loss'],
                    'val': val_metrics['val_loss'],
                }, global_epoch_counter)

                tb_writer.add_scalars('loss/heatmap', {
                    'train': train_metrics['heatmap_loss'],
                    'val': val_metrics['val_heatmap_loss'],
                }, global_epoch_counter)

                train_traj = train_metrics.get('trajectory_loss', 0)
                val_traj = val_metrics.get('val_trajectory_loss', 0)
                if train_traj > 0 or val_traj > 0:
                    tb_writer.add_scalars('loss/trajectory', {
                        'train': train_traj,
                        'val': val_traj,
                    }, global_epoch_counter)
                train_lm = train_metrics.get('lm_loss', 0)
                val_lm = val_metrics.get('val_lm_loss', 0)
                if train_lm > 0 or val_lm > 0:
                    tb_writer.add_scalars('loss/lm', {
                        'train': train_lm,
                        'val': val_lm,
                    }, global_epoch_counter)

                for hm_key in ('peak_loss', 'vis_loss', 'coord_loss', 'neg_loss'):
                    val_key = f'val_hm_{hm_key}'
                    if val_key in val_metrics:
                        tb_writer.add_scalar(f'epoch/val_hm_{hm_key}', val_metrics[val_key], global_epoch_counter)

                if val_metrics.get('val_heatmap_mse', 0) > 0:
                    tb_writer.add_scalar('loss/heatmap_inference_mse', val_metrics['val_heatmap_mse'], global_epoch_counter)

                tb_writer.add_scalar('epoch/val_loss', val_metrics['val_loss'], global_epoch_counter)

                for vk in ('val_vis_accuracy', 'val_vis_precision', 'val_vis_recall',
                            'val_vis_tnr', 'val_vis_f1', 'val_vis_gt_pos_ratio'):
                    if vk in val_metrics:
                        tb_writer.add_scalar(f'epoch/{vk}', val_metrics[vk], global_epoch_counter)

            tb_writer.flush()

        if notifier and do_eval:
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
            logger.info("  🛑 Early stopping")
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
        logger.info("📊 训练摘要:")
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

    if notifier:
        try:
            notifier.send_training_complete(
                total_time=timer.get_total_elapsed() if timer else "N/A",
                best_val_loss=best_val_loss,
                final_stage=stage_name,
            )
            logger.info("📢 飞书通知已发送: 训练完成")
        except Exception as e:
            logger.warning(f"飞书通知发送失败: {e}")

    if tb_writer is not None:
        tb_writer.close()
    cleanup_distributed()


if __name__ == '__main__':
    main()
