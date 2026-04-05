#!/usr/bin/env python3
"""
VLN 训练脚本
==============

使用共享 Habitat/InternNav 环境进行视觉语言导航训练。
单阶段训练：History 热力图头 + Action Head + Progress Head
"""

import sys
import os
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
import gc
import time
import logging
import argparse
import warnings
import psutil
from datetime import datetime

# ============================================
# CUDA 性能优化
# ============================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
warnings.filterwarnings("ignore", message="Asked to sample")

from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset, VLNTrajectoryDataset
from src.models.runtime_compat import ensure_transformers_runtime_compat
from src.utils.logger import setup_logger
from src.utils.gpu_heatmap import GPUHeatmapComputer
from src.utils.notifier import FeishuNotifier, create_notifier

# --- All training utilities from the modular training/ package ---
from training import (
    load_config,
    set_seed,
    DistributedContext,
    init_distributed_context,
    cleanup_distributed,
    initialize_trainable_module_sync,
    _dist_barrier,
    _malloc_trim,
    _drop_page_cache,
    _cgroup_mem_usage_gb,
    _CG_LIMIT_GB,
    _worker_init_fn,
    _load_normalized_state_dict,
    EMAModel,
    TrainingTimer,
    TrainingPlotter,
    collate_fn,
    build_model,
    set_trainable_modules,
    apply_nextdit_warmup_freeze,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    validate,
    CheckpointManager,
    load_checkpoint_for_resume,
    _write_json,
    _write_yaml,
    _append_jsonl,
    _safe_symlink,
    _clear_directory,
    _capture_git_state,
    _capture_env_state,
    _find_resume_checkpoint,
)

logger = logging.getLogger(__name__)


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
    logger.info("VLN 训练 (shared Habitat/InternNav env)")
    logger.info("=" * 60)
    
    # 构建数据集
    logger.info("📂 Loading datasets...")
    dataset_type = cfg['data'].get('dataset_type', 'sliding_window')
    logger.info(f"  Dataset type: {dataset_type}")
    val_root_cfg = cfg['data'].get('val_root')
    if val_root_cfg:
        logger.info(f"  Validation from separate root: {val_root_cfg} (split={cfg['data'].get('val_split', 'val')})")
    if dataset_type == 'trajectory':
        traj_cfg = cfg['data'].get('trajectory', cfg['data'].get('sliding_window', {}))
        sample_stride = traj_cfg.get('sample_stride', 1)
        clip_level_sampling = traj_cfg.get('clip_level_sampling', True)
        samples_per_clip = traj_cfg.get('samples_per_clip', 8)
        
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
            load_traj_images=traj_cfg.get('load_traj_images', False),
            traj_image_size=tuple(traj_cfg.get('traj_image_size', [224, 224])),
            use_subinstruction=traj_cfg.get('use_subinstruction', False),
            fgr2r_subinstr_path=traj_cfg.get('fgr2r_subinstr_path', None),
        )
        
        val_root = cfg['data'].get('val_root') or cfg['data']['root']
        val_split = cfg['data'].get('val_split', 'val')
        val_samples_per_clip = traj_cfg.get('val_samples_per_clip', 2)
        val_dataset = VLNTrajectoryDataset(
            root=val_root,
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
            random_subsequence=False,
            predict_horizon=traj_cfg.get('predict_horizon', 24),
            action_scale=traj_cfg.get('action_scale', 4.0),
            enable_trajectory_augmentation=False,
            load_traj_images=traj_cfg.get('load_traj_images', False),
            traj_image_size=tuple(traj_cfg.get('traj_image_size', [224, 224])),
            use_subinstruction=False,
        )
    else:
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
        
        val_root = cfg['data'].get('val_root') or cfg['data']['root']
        val_split = cfg['data'].get('val_split', 'val')
        val_samples_per_clip = sw_cfg.get('val_samples_per_clip', 2)
        val_dataset = VLNSlidingWindowDataset(
            root=val_root,
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
    
    stage_cfg = all_stages[0]
    stage_name = stage_cfg['name']
    
    if args.epochs is not None:
        stage_cfg = stage_cfg.copy()
        stage_cfg['epochs'] = args.epochs
    
    total_epochs = stage_cfg['epochs']
    
    logger.info("=" * 60)
    logger.info(f"📋 训练配置: {stage_name}")
    logger.info(f"   Epochs: {total_epochs}, Heatmap Size: {stage_cfg['hm_size']}")
    logger.info("=" * 60)
    
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
    
    packing_enabled = cfg['model']['llm'].get('enable_packing', False)
    
    if packing_enabled:
        raise ValueError(
            "当前共享环境训练路径已移除 Sequence Packing 兼容代码，请在配置中设置 "
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
        actual_collate_fn = PanoramicTokenizedCollator(pano_processor)
        logger.info("   ✅ Panoramic tokenized collator enabled")
    
    mp_context = 'fork' if num_workers > 0 else None
    
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
    ) if dist_context.enabled else None
    
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
        in_order=False if num_workers > 0 else True,
    )
    
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
    if hasattr(raw_model, 'qwen3_5') and hasattr(raw_model.qwen3_5, '_load_model'):
        if raw_model.qwen3_5.model is None:
            logger.info("🔄 Pre-loading VLM backbone (ensure LoRA params available for optimizer)...")
            raw_model.qwen3_5._load_model()
        logger.info(
            "   🧠 Qwen attention implementation: %s",
            getattr(raw_model.qwen3_5.config, 'attn_implementation', 'unknown'),
        )
    if getattr(raw_model.config, 'enable_heatmap', False):
        logger.info("🔄 Constructing HeatmapVLN before optimizer setup...")
        raw_model._ensure_heatmap_vln()
    
    if args.load_weights:
        weights_path = Path(args.load_weights)
        if weights_path.exists():
            ckpt = torch.load(str(weights_path), map_location='cpu')
            state_dict = ckpt.get('trainable_state_dict', {})
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

        do_eval = (epoch % eval_every_epochs == 0) or (epoch == total_epochs)

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
        
        logger.info(
            f"  Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f})"
        )
        
        eta = timer.get_eta(epoch, total_epochs)
        logger.info(f"  ⏱️  Epoch time: {timer.get_epoch_time()} | ETA: {eta}")

        if do_eval and val_metrics:
            val_hm_mse_str = f", infer_mse: {val_metrics['val_heatmap_mse']:.6f}" if val_metrics.get('val_heatmap_mse', 0) > 0 else ""
            logger.info(
                f"  Val Loss: {val_metrics['val_loss']:.4f} "
                f"(hm: {val_metrics['val_heatmap_loss']:.4f}{val_hm_mse_str})"
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
