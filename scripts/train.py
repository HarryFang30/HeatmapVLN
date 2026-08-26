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
import math
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
    assert_complete_internnav_system1_load,
    assert_complete_lora_checkpoint_match,
    build_l2_sp_reference,
    build_model,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    collate_fn,
    extract_lora_checkpoint_state,
    init_distributed_context,
    initialize_trainable_module_sync,
    ensure_heatmap_optimizer_state_fp32,
    heatmap_control_enabled,
    load_checkpoint_for_resume,
    load_frozen_heatmap_checkpoint,
    load_past_plan_action_initialization,
    load_pose_adaptation_initialization,
    load_config,
    make_grad_scaler,
    reject_heatmap_control_load_weights,
    run_training_preflight,
    safe_torch_load,
    set_seed,
    set_trainable_modules,
    train_one_epoch,
    validate_heatmap_control_resume_checkpoint,
    validate_heatmap_warmstart_contract,
    verify_heatmap_warmstart_loaded,
)
from scripts.training.selection import BestCheckpointSelector
from scripts.training.preflight import assert_single_view_training_contract
from scripts.training.single_view_heatmap_warmstart import (
    WARMSTART_POLICY as SINGLE_VIEW_WARMSTART_POLICY,
    file_sha256 as single_view_artifact_sha256,
    load_artifact_into_model as load_single_view_heatmap_artifact,
)
from scripts.training.native_internnav_dependency import (
    inject_native_internnav_dependency_from_env,
)
from scripts.training.formal_heatmap_control_contract import (
    FormalHeatmapControlContractError,
    assert_formal_heatmap_control_no_training_eval,
)
from scripts.training.validate import validate

from src.data.factory import build_dataset, build_trajectory_dataset
from src.data.amb3r_pose_cache import AMB3R_POSE_PROVIDER
from src.data.internnav_heatmap_control_collator import (
    InternNavHeatmapControlCollator,
)
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.single_view_heatmap_collator import SingleViewHeatmapCollator
from src.data.trajectory_dagger_dataset import (
    DeterministicMixtureSampler,
    SourceMixtureDataset,
)
from src.models.runtime_compat import ensure_transformers_runtime_compat
from src.utils.gpu_heatmap import GPUHeatmapComputer
from src.utils.logger import setup_logger
from src.utils.notifier import create_notifier

logger = logging.getLogger(__name__)


def _dataset_uses_dynamic_sampling(dataset) -> bool:
    """Return whether ``set_epoch`` rebuilds this dataset's sample index."""
    set_epoch = getattr(dataset, 'set_epoch', None)
    if not callable(set_epoch):
        return False
    explicit = getattr(dataset, 'dynamic_sampling_enabled', None)
    return True if explicit is None else bool(explicit)


def _install_baseline_best_threshold(
    checkpoint_manager,
    value: float,
    *,
    enabled: bool,
) -> bool:
    """Install a pre-training metric as the best threshold when requested."""
    if not enabled or not checkpoint_manager.is_better(float(value)):
        return False
    checkpoint_manager.best_metric_value = float(value)
    return True


def _parse_auto_bool(value, *, name: str) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like or 'auto', got {value!r}")


def _dev_shm_total_gb() -> float | None:
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stat.f_frsize * stat.f_blocks / (1024**3)


def _should_enable_shm_bypass(cfg: dict, num_workers: int, logger: logging.Logger) -> bool:
    if num_workers <= 0:
        return False

    data_cfg = cfg.get("data", {})
    raw_mode = os.environ.get("HEATMAPVLN_SHM_BYPASS", data_cfg.get("shm_bypass", "auto"))
    mode = _parse_auto_bool(raw_mode, name="data.shm_bypass/HEATMAPVLN_SHM_BYPASS")

    if isinstance(mode, bool):
        logger.info("   🔀 ShmBypass %s by config/env", "enabled" if mode else "disabled")
        return mode

    raw_min_gb = os.environ.get(
        "HEATMAPVLN_SHM_BYPASS_MIN_GB",
        data_cfg.get("shm_bypass_min_gb", 8.0),
    )
    min_gb = float(raw_min_gb)
    shm_gb = _dev_shm_total_gb()
    if shm_gb is None:
        logger.warning(
            "   🔀 ShmBypass auto: cannot inspect /dev/shm; enabling conservative IPC bypass"
        )
        return True

    enabled = shm_gb < min_gb
    logger.info(
        "   🔀 ShmBypass auto: /dev/shm=%.1fGB threshold=%.1fGB -> %s",
        shm_gb,
        min_gb,
        "enabled" if enabled else "disabled",
    )
    return enabled


def _log_notification_result(
    logger: logging.Logger,
    sent: bool,
    event_name: str,
    error: str | None = None,
) -> None:
    if sent:
        logger.info("📢 飞书通知已发送: %s", event_name)
    elif error:
        logger.warning("飞书通知未发送: %s (%s)", event_name, error)
    else:
        logger.warning("飞书通知未发送: %s", event_name)


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
                        help='执行一批完整训练 preflight（含 backward/DDP/optimizer），不保存 checkpoint')
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
    parser.add_argument('--shm-bypass', type=str, default=None,
                        help="覆盖 data.shm_bypass: auto/on/off")

    args = parser.parse_args()

    if args.dry_run and (args.resume or args.auto_resume):
        raise ValueError('--dry-run cannot be combined with --resume/--auto-resume')

    # 加载配置
    cfg = load_config(args.config)
    heatmap_control_resume_guard = heatmap_control_enabled(cfg)
    if heatmap_control_resume_guard:
        # The formal launcher has already hashed the 14 released model files
        # once.  Ranks only validate its scalar env contract and persist the
        # resulting closure in cfg/checkpoints; they never re-hash the shards.
        inject_native_internnav_dependency_from_env(cfg)
    formal_no_train_eval_contract = (
        assert_formal_heatmap_control_no_training_eval(cfg)
    )
    if formal_no_train_eval_contract is not None:
        cfg.setdefault("runtime", {})[
            "formal_training_evaluation_contract"
        ] = formal_no_train_eval_contract
        if args.epochs not in (None, 3):
            raise FormalHeatmapControlContractError(
                "formal heatmap-control training forbids --epochs overrides "
                f"other than 3; got {args.epochs}"
            )
    reject_heatmap_control_load_weights(cfg, args.load_weights)
    if args.distributed:
        cfg.setdefault('gpu', {}).setdefault('multi_gpu', {})['enabled'] = True
    if args.num_workers is not None:
        cfg.setdefault('data', {})['num_workers'] = args.num_workers
    if args.prefetch_factor is not None:
        cfg.setdefault('data', {})['prefetch_factor'] = args.prefetch_factor
    if args.pin_memory is not None:
        cfg.setdefault('data', {})['pin_memory'] = args.pin_memory
    if args.shm_bypass is not None:
        cfg.setdefault('data', {})['shm_bypass'] = args.shm_bypass

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
        if args.dry_run:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = base_out_dir / f'preflight_{run_timestamp}'
        elif args.auto_resume and latest_link.exists():
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
        run_prefix = 'preflight' if args.dry_run else 'run'
        run_dir = base_out_dir / f'{run_prefix}_{run_timestamp}'

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
        if not args.dry_run:
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

        logger.info("🧾 Writing run manifest...")
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
                "record_type": "preflight_start" if args.dry_run else "run_start",
                "run_name": run_dir.name,
                "is_resuming": is_resuming,
                "output_dir": str(run_dir),
                "distributed": dist_context.enabled,
                "world_size": dist_context.world_size,
            },
        )
        logger.info("   ✓ Run manifest written")

    # ==================== TensorBoard ====================
    tb_writer = None
    if dist_context.is_main and cfg['log'].get('use_tensorboard', False):
        tb_base_cfg = cfg['log'].get('tensorboard_dir')
        tb_parent_dir = Path(tb_base_cfg) if tb_base_cfg else None
        live_tb_dir = (tb_parent_dir / run_dir.name) if tb_parent_dir else tb_run_dir
        if not is_resuming:
            if live_tb_dir.exists():
                _clear_directory(live_tb_dir)
            else:
                live_tb_dir.mkdir(parents=True, exist_ok=True)
        else:
            live_tb_dir.mkdir(parents=True, exist_ok=True)

        if tb_parent_dir:
            _safe_symlink(tb_run_dir, live_tb_dir)
        tb_writer = SummaryWriter(log_dir=str(live_tb_dir))
        logger.info(f"📊 TensorBoard: {tb_run_dir}")
        logger.info(f"   实时监控目录: {live_tb_dir}")
        logger.info(f"   autodl入口: tensorboard --logdir {tb_parent_dir or live_tb_dir}")
    if not dist_context.is_main:
        metrics_jsonl_path = None
    _dist_barrier()

    cfg['loss']

    logger.info("=" * 60)
    logger.info("VLN 训练 (shared Habitat/InternNav env)")
    logger.info("=" * 60)

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
    stage_train_future = bool(stage_cfg.get('train_future', False))
    stage_uses_heatmap_targets = bool(
        stage_cfg.get('train_history', True) or stage_cfg.get('train_future', False)
    )
    validation_enabled = bool(cfg.get('validation', {}).get('enabled', True))

    # 构建数据集
    logger.info("📂 Loading datasets...")
    dataset_type = cfg['data'].get('dataset_type', 'sliding_window')
    logger.info(f"  Dataset type: {dataset_type}")
    val_root_cfg = cfg['data'].get('val_root')
    if val_root_cfg:
        logger.info(f"  Validation from separate root: {val_root_cfg} (split={cfg['data'].get('val_split', 'val')})")
    dataset_overrides = {}
    if dataset_type in {'trajectory', 'expert_dagger_mixture'} and stage_train_future:
        dataset_overrides['load_future_trajectory_heatmap'] = True
        logger.info(
            "  Future supervision enabled from the exact expert System1 "
            "action target (four no-depth temporal bins)"
        )
    if dataset_type in {'trajectory', 'expert_dagger_mixture'} and not stage_uses_heatmap_targets:
        dataset_overrides['load_history_heatmap'] = False
        logger.info("  Trajectory dataset override: load_history_heatmap=False (stage has no heatmap supervision)")
    train_dataset = build_dataset(cfg, split='train', **dataset_overrides)

    if not validation_enabled:
        # Construction itself can require validation-only artifacts (for
        # example a strict AMB3R endpoint cache). A construction-only training
        # smoke with validation disabled must therefore never instantiate it.
        val_dataset = None
        logger.info("  Validation disabled before dataset construction "
                    "(validation.enabled=false)")
    elif dataset_type == 'trajectory_dagger':
        dagger_cfg = cfg['data'].get('trajectory_dagger', {})
        val_roots = (
            dagger_cfg.get('val_collection_roots')
            or dagger_cfg.get('val_collection_root')
        )
        if val_roots:
            val_dataset = build_dataset(
                cfg,
                split=cfg['data'].get('val_split', 'val'),
                collection_roots=val_roots,
            )
        else:
            val_dataset = None
            logger.info(
                "  No trajectory_dagger.val_collection_roots configured, "
                "skipping validation dataset"
            )
    elif dataset_type == 'expert_dagger_mixture':
        dagger_cfg = cfg['data'].get('trajectory_dagger', {})
        val_roots = (
            dagger_cfg.get('val_collection_roots')
            or dagger_cfg.get('val_collection_root')
        )
        val_root = cfg['data'].get('val_root')
        val_samples = cfg['data'].get('trajectory', {}).get(
            'val_samples_per_clip',
            2,
        )
        expert_val_overrides = {
            'root': val_root,
            'samples_per_clip': val_samples,
            'random_subsequence': False,
            'enable_trajectory_augmentation': False,
            **dataset_overrides,
        }
        if val_roots and val_root:
            val_dataset = build_dataset(
                cfg,
                split=cfg['data'].get('val_split', 'val'),
                expert_overrides=expert_val_overrides,
                dagger_overrides={'collection_roots': val_roots},
            )
        elif val_root:
            expert_val_dataset = build_trajectory_dataset(
                cfg,
                split=cfg['data'].get('val_split', 'val'),
                **expert_val_overrides,
            )
            val_dataset = SourceMixtureDataset({'expert': expert_val_dataset})
            logger.info(
                "  Mixture validation: expert-only held-out trajectories "
                "(no DAgger validation roots configured)"
            )
        else:
            val_dataset = None
            logger.info(
                "  Mixture validation requires data.val_root; skipping validation"
            )
    elif dataset_type == 'trajectory':
        val_root = cfg['data'].get('val_root')
        if val_root:
            val_samples = cfg['data'].get('trajectory', cfg['data'].get('sliding_window', {})).get('val_samples_per_clip', 2)
            val_dataset = build_dataset(
                cfg, split=cfg['data'].get('val_split', 'val'), root=val_root,
                samples_per_clip=val_samples,
                random_subsequence=False,
                enable_trajectory_augmentation=False,
                **dataset_overrides,
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
        if not cfg.get('validation', {}).get('enabled', True):
            logger.info("  Val: disabled (validation.enabled=false)")
        else:
            logger.info("  Val: disabled (no val_root for trajectory, or no val_loader)")

    # 构建模型
    logger.info("🏗️  Building model...")
    model = build_model(cfg, verbose=dist_context.is_main)

    nextdit_cfg = cfg.get('model', {}).get('action_head', {}).get('nextdit', {})
    default_require_complete_system1 = bool(
        stage_cfg.get('train_action', True)
        and nextdit_cfg.get('enabled', False)
        and (
            nextdit_cfg.get('internnav_system1_path')
            or nextdit_cfg.get('internnav_model_path')
        )
    )
    require_complete_system1 = stage_cfg.get(
        'require_complete_internnav_system1',
        default_require_complete_system1,
    )
    if require_complete_system1 is None:
        require_complete_system1 = default_require_complete_system1
    system1_required_tensors = 0
    system1_source = None
    if require_complete_system1:
        system1_required_tensors = assert_complete_internnav_system1_load(
            model,
            logger=logger,
        )
        system1_source = model._internnav_system1_load_audit.get('source')

    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        out_dir=str(ckpt_dir),
        max_ckpts=cfg['log'].get('max_ckpts', 3)
    )
    validation_cfg = cfg.get('validation', {})
    best_selection_enabled = bool(
        validation_cfg.get('best_selection_enabled', True)
    )
    if (
        not best_selection_enabled
        and validation_cfg.get('baseline_as_best_threshold', False)
    ):
        raise ValueError(
            "validation.baseline_as_best_threshold requires "
            "validation.best_selection_enabled=true"
        )
    save_best_metric = str(
        validation_cfg.get('save_best_metric', 'val_loss')
    )
    save_best_mode = str(
        validation_cfg.get('save_best_mode', 'min')
    ).lower()
    ckpt_manager.configure_best_metric(save_best_metric, save_best_mode)
    checkpoint_selector = BestCheckpointSelector(
        primary_metric=save_best_metric,
        primary_mode=save_best_mode,
        baseline_as_incumbent=bool(
            validation_cfg.get('baseline_as_best_threshold', False)
        ),
        overall_metric=str(
            validation_cfg.get(
                'baseline_overall_metric',
                'val_heatmap_joint_pck8',
            )
        ),
        overall_tolerance=float(
            validation_cfg.get('baseline_overall_tolerance', 0.02)
        ),
        back_metric=str(
            validation_cfg.get(
                'baseline_back_metric',
                'val_heatmap_back_pck8',
            )
        ),
        back_tolerance=float(
            validation_cfg.get('baseline_back_tolerance', 0.03)
        ),
        direction_metrics=validation_cfg.get(
            'baseline_direction_metrics'
        ),
        direction_tolerance=float(
            validation_cfg.get('baseline_direction_tolerance', 0.03)
        ),
        loss_metric=str(
            validation_cfg.get(
                'save_best_loss_tiebreak_metric',
                'val_loss',
            )
        ),
    )
    constrained_selection = bool(
        best_selection_enabled and checkpoint_selector.baseline_as_incumbent
    )
    if best_selection_enabled:
        logger.info(
            "  Best-checkpoint selection: metric=%s mode=%s constrained=%s",
            save_best_metric,
            save_best_mode,
            constrained_selection,
        )
    else:
        logger.info("  Best-checkpoint selection: disabled")

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

    if (
        resume_path
        and Path(resume_path).exists()
        and not args.load_weights
        and not heatmap_control_resume_guard
    ):
        inferred_base = _infer_base_checkpoint_from_resume(Path(resume_path), logger)
        if inferred_base:
            args.load_weights = inferred_base
            logger.info("🔗 Inferred base checkpoint from resume metadata: %s", inferred_base)

    resume_skip_batches: int | None = None
    resume_l2_sp_reference = None
    resume_selection_state = None
    resume_mixture_sampler_state = None
    resume_best_metric_matches = False

    if resume_path and Path(resume_path).exists():
        if heatmap_control_resume_guard:
            resume_guard_report = validate_heatmap_control_resume_checkpoint(
                str(resume_path), model, cfg
            )
            logger.info(
                "  ✓ Heatmap-control resume boundary validated before model "
                "load: tensors=%d online=%s EMA=%s",
                resume_guard_report['state_tensor_count'],
                resume_guard_report['validated_online_state'],
                resume_guard_report['validated_ema_shadow'],
            )
        resume_info = load_checkpoint_for_resume(
            str(resume_path),
            model,
            optimizer=None,
            scheduler=None,
            logger=logger,
            strict_state_restore=heatmap_control_resume_guard,
            # The lazy Heatmap Head does not exist yet at this first pass.
            # Pose-adaptation checkpoints are self-contained, so inspect only
            # epoch/selection metadata now and restore all 79+34 tensors in the
            # second pass after _ensure_heatmap_vln() and optimizer creation.
            metadata_only=bool(
                stage_cfg.get('heatmap_pose_adaptation_init', False)
            ),
        )
        resume_epoch = resume_info['epoch']
        if resume_info.get('batch') is not None:
            # Mid-epoch checkpoint: stay on the same epoch and skip batches
            # that were already processed before the save.
            resume_skip_batches = resume_info['batch']
        ckpt_manager.best_val_loss = resume_info['best_val_loss']
        resume_best_name = resume_info.get('best_metric_name', 'val_loss')
        resume_best_mode = resume_info.get('best_metric_mode', 'min')
        resume_best_metric_matches = (
            resume_best_name == save_best_metric
            and resume_best_mode == save_best_mode
        )
        if resume_best_metric_matches:
            ckpt_manager.best_metric_value = float(
                resume_info['best_metric_value']
            )
        else:
            logger.warning(
                "Resume checkpoint selected best by %s/%s, but current config "
                "uses %s/%s; resetting best comparison for the new metric",
                resume_best_name,
                resume_best_mode,
                save_best_metric,
                save_best_mode,
            )
        resume_l2_sp_reference = resume_info.get('l2_sp_reference_state')
        resume_selection_state = resume_info.get(
            'checkpoint_selection_state'
        )
        resume_mixture_sampler_state = resume_info.get(
            'mixture_sampler_state'
        )

        if best_selection_enabled and resume_selection_state is not None:
            try:
                checkpoint_selector.load_state_dict(
                    resume_selection_state
                )
                logger.info(
                    "  ✓ Restored checkpoint-selection state: "
                    "incumbent epoch=%s source=%s",
                    checkpoint_selector.incumbent_epoch,
                    checkpoint_selector.incumbent_source,
                )
            except ValueError:
                if constrained_selection:
                    raise
                logger.warning(
                    "Resume checkpoint selection policy differs from the "
                    "current unconstrained policy; resetting selector state",
                    exc_info=True,
                )
        elif best_selection_enabled and constrained_selection:
            raise RuntimeError(
                "Constrained checkpoint selection cannot resume from a "
                "checkpoint without checkpoint_selection_state. Resume from "
                "a checkpoint produced by this policy, or start a new run "
                "from weights so an exact step-0 baseline can be evaluated."
            )

        if (
            best_selection_enabled
            and checkpoint_selector.incumbent_metrics is None
            and not constrained_selection
            and resume_best_metric_matches
            and math.isfinite(ckpt_manager.best_metric_value)
        ):
            checkpoint_selector.set_incumbent(
                {save_best_metric: ckpt_manager.best_metric_value},
                epoch=resume_epoch,
                source="legacy_resume",
            )

    requires_base_checkpoint = bool(
        stage_cfg.get('requires_base_checkpoint', False)
        or stage_cfg.get('bridge_only', False)
    )
    if requires_base_checkpoint and not args.load_weights:
        raise ValueError(
            "This training stage requires the Stage1-S2 panoramic System2 SFT checkpoint. "
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

    if notifier and not args.dry_run:
        try:
            sent = notifier.send_training_start(
                config_name=Path(args.config).stem,
                stages=[stage_cfg],
                total_epochs=total_epochs,
            )
            _log_notification_result(
                logger, sent, "训练开始", getattr(notifier, "last_error", None)
            )
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
            or stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False))
            or stage_train_future,
        ),
    )
    stage_train_action = bool(stage_cfg.get('train_action', True))
    stage_train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
    stage_teacher_force_system2_answer = bool(stage_cfg.get('teacher_force_system2_answer', False))
    nextdit_enabled = bool(
        cfg.get('model', {})
        .get('action_head', {})
        .get('nextdit', {})
        .get('enabled', False)
    )
    heatmap_control_collator_enabled = bool(
        cfg.get('model', {})
        .get('action_head', {})
        .get('nextdit', {})
        .get('heatmap_control', {})
        .get('enabled', False)
    )
    past_plan_action_enabled = bool(
        cfg.get('model', {})
        .get('past_plan_action', {})
        .get('enabled', False)
    )
    use_heatmap_control_collator = (
        (heatmap_control_collator_enabled or past_plan_action_enabled)
        and (stage_train_action or stage_train_future)
        and use_worker_tokenized_collator
    )
    use_panoramic_tokenized_collator = (
        use_worker_tokenized_collator
        and (
            cfg['model'].get('heatmap', {}).get('enable', True)
            or stage_train_lm
            or (stage_train_action and nextdit_enabled)
        )
        and getattr(train_dataset, '_is_panoramic', False)
        and not getattr(train_dataset, 'single_view_rgb_input', False)
        and (val_dataset is None or getattr(val_dataset, '_is_panoramic', False))
    )
    heatmap_input_mode = str(
        cfg.get('model', {}).get('heatmap', {}).get('input_mode', 'panoramic')
    ).strip().lower()
    use_single_view_heatmap_collator = (
        heatmap_input_mode == 'internnav_single_view'
        and stage_uses_heatmap_targets
    )
    if use_heatmap_control_collator:
        if heatmap_input_mode != 'internnav_single_view':
            raise RuntimeError(
                "heatmap control requires model.heatmap.input_mode=internnav_single_view"
            )
        if not getattr(train_dataset, '_is_panoramic', False):
            raise RuntimeError("heatmap control requires panoramic source observations")
        if val_dataset is not None and not getattr(val_dataset, '_is_panoramic', False):
            raise RuntimeError("heatmap control validation source must be panoramic")
        llm_cfg = cfg['model'].get('llm', {})
        llm_model_path = llm_cfg.get('model_path', './models/internnav_backbone')
        ensure_transformers_runtime_compat(
            model_path=llm_model_path,
            requested_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
            requested_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
            logger=logger,
        )
        from transformers import AutoProcessor

        logger.info(
            "🔄 Loading released InternNav processor for native System2 + "
            "independent frozen heatmap inputs..."
        )
        control_processor = AutoProcessor.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        n_traj_query = int(
            cfg['model']['action_head']['nextdit'].get('n_query', 4)
        )
        require_amb3r_provider = bool(
            past_plan_action_enabled
            and (
                stage_train_future
                or str(stage_cfg.get('past_plan_action_stage', '')).startswith(
                    'stage2'
                )
            )
        )
        if require_amb3r_provider and not getattr(
            train_dataset, 'require_amb3r_pose_cache', False
        ):
            raise RuntimeError(
                "Past→Plan Stage1/2 expert training requires trajectory "
                "endpoint-v2 AMB3R cache; GT history-pose fallback is forbidden"
            )
        actual_collate_fn = InternNavHeatmapControlCollator(
            control_processor,
            n_traj_query=n_traj_query,
            max_seq_length=int(llm_cfg.get('max_seq_length', 8192)),
            teacher_force_system2_answer=stage_teacher_force_system2_answer,
            include_future_trajectory_targets=stage_train_future,
            required_history_pose_provider=(
                AMB3R_POSE_PROVIDER if require_amb3r_provider else None
            ),
        )
        logger.info(
            "   ✅ Native InternNav joint collator enabled: System2 front-history/"
            "current/lookdown + %d TRAJ tokens; heatmap images are namespaced",
            n_traj_query,
        )
    elif use_single_view_heatmap_collator:
        if stage_train_action or stage_train_lm:
            raise RuntimeError(
                "internnav_single_view worker collation is a heatmap-only "
                "stage; System1 and System2 must remain frozen"
            )
        if not getattr(train_dataset, '_is_panoramic', False):
            raise RuntimeError(
                "internnav_single_view requires a panoramic source for "
                "four-direction heatmap supervision"
            )
        datasets_to_check = [train_dataset]
        if val_dataset is not None:
            datasets_to_check.append(val_dataset)
        invalid_datasets = [
            type(dataset).__name__
            for dataset in datasets_to_check
            if not getattr(dataset, 'single_view_rgb_input', False)
        ]
        if invalid_datasets:
            raise RuntimeError(
                "internnav_single_view requires "
                "data.sliding_window.single_view_rgb_input=true for every "
                f"dataset, invalid={invalid_datasets}"
            )
        if any(getattr(dataset, 'defer_heatmap_to_gpu', False) for dataset in datasets_to_check):
            raise RuntimeError(
                "internnav_single_view requires defer_heatmap_to_gpu=false; "
                "the current GPU target path is not a four-camera target builder"
            )
        llm_cfg = cfg['model'].get('llm', {})
        llm_model_path = llm_cfg.get('model_path', './models/internnav_backbone')
        ensure_transformers_runtime_compat(
            model_path=llm_model_path,
            requested_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
            requested_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
            logger=logger,
        )
        from transformers import AutoProcessor

        logger.info(
            "🔄 Loading native InternNav image processor for front-only heatmap training..."
        )
        single_view_processor = AutoProcessor.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        actual_collate_fn = SingleViewHeatmapCollator(single_view_processor)
        logger.info(
            "   ✅ Single-view heatmap collator enabled: history front images + "
            "current front image; four-view RGB is forbidden"
        )
    elif use_panoramic_tokenized_collator:
        llm_cfg = cfg['model'].get('llm', {})
        llm_model_path = llm_cfg.get('model_path', './models/internnav_backbone')
        ensure_transformers_runtime_compat(
            model_path=llm_model_path,
            requested_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
            requested_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
            logger=logger,
        )
        from transformers import AutoProcessor

        logger.info("🔄 Loading Qwen processor for panoramic worker-side tokenization...")
        pano_processor = AutoProcessor.from_pretrained(llm_model_path, trust_remote_code=True)
        n_traj_query = cfg.get('model', {}).get('action_head', {}).get('nextdit', {}).get('n_query', 0)
        if not cfg.get('model', {}).get('action_head', {}).get('nextdit', {}).get('enabled', False):
            n_traj_query = 0
        if not stage_cfg.get('train_action', False):
            n_traj_query = 0
        train_lm = bool(stage_cfg.get('train_lm', stage_cfg.get('train_system2_sft', False)))
        sft_prompt_mode = train_lm or stage_teacher_force_system2_answer
        traj_cfg = cfg.get('data', {}).get('trajectory', cfg.get('data', {}).get('sliding_window', {}))
        sft_protocol = stage_cfg.get(
            'system2_sft_protocol',
            traj_cfg.get('system2_sft_protocol', 'direct'),
        )
        sft_protocol = str(sft_protocol).lower()
        max_seq_len = int(llm_cfg.get('max_seq_length', 8192))
        actual_collate_fn = PanoramicTokenizedCollator(
            pano_processor,
            n_traj_query=n_traj_query,
            sft_mode=sft_prompt_mode,
            sft_include_turns=stage_cfg.get('sft_include_turns', True),
            sft_include_forward=stage_cfg.get('sft_include_forward', False),
            sft_protocol=sft_protocol,
            build_sft_labels=train_lm,
            max_seq_length=max_seq_len,
            include_heatmap_targets=stage_uses_heatmap_targets,
            include_history_rel_poses=stage_cfg.get(
                'retain_history_rel_poses',
                stage_uses_heatmap_targets
                or dataset_type == 'trajectory_dagger',
            ),
            retain_raw_panoramic_views=stage_cfg.get(
                'retain_raw_panoramic_views',
                True,
            ),
            compute_pano_text_anchor_positions=stage_cfg.get(
                'compute_pano_text_anchor_positions',
                True,
            ),
            heatmap_layout=stage_uses_heatmap_targets,
        )
        logger.info(
            "   ✅ Panoramic tokenized collator enabled "
            "(n_traj_query=%d, sft_mode=%s, build_sft_labels=%s, return_lm_loss=%s, "
            "protocol=%s, max_seq_len=%d, heatmap_targets=%s, raw_pano=%s, anchors=%s, "
            "heatmap_layout=%s)",
            n_traj_query,
            sft_prompt_mode,
            train_lm,
            train_lm,
            sft_protocol,
            max_seq_len,
            stage_uses_heatmap_targets,
            stage_cfg.get('retain_raw_panoramic_views', True),
            stage_cfg.get('compute_pano_text_anchor_positions', True),
            stage_uses_heatmap_targets,
        )
    elif getattr(train_dataset, '_is_panoramic', False) and not stage_cfg.get('train_action', True):
        logger.info("   ✅ Heatmap-only stage: using standard panoramic collate path (skip AutoProcessor worker tokenization)")

    mp_context = 'fork' if num_workers > 0 else None

    # -- /dev/shm bypass: wrap datasets + collate only when needed ----
    # On small Docker /dev/shm mounts this avoids DataLoader IPC failures.
    # On MXC500-style nodes with huge /dev/shm it is slower, so keep it
    # configurable and auto-disable it when shared memory is sufficient.
    if _should_enable_shm_bypass(cfg, num_workers, logger):
        train_dataset = ShmBypassDataset(train_dataset)
        if val_dataset is not None:
            val_dataset = ShmBypassDataset(val_dataset)
        actual_collate_fn = ShmBypassCollate(actual_collate_fn)
        logger.info("   🔀 ShmBypass active: tensor↔numpy IPC")

    uses_dynamic_sampling = _dataset_uses_dynamic_sampling(train_dataset)

    # set_epoch() mutates the main-process dataset. Persistent worker copies
    # would keep the previous epoch's sample_index indefinitely.
    persistent_workers = num_workers > 0 and not uses_dynamic_sampling
    if dataset_type == 'expert_dagger_mixture':
        mixture_cfg = cfg['data'].get('mixture', {})
        global_batch = dist_context.world_size * int(cfg['optim']['batch_size'])
        configured_epoch_size = mixture_cfg.get('epoch_size')
        if heatmap_control_resume_guard and configured_epoch_size is None:
            raise RuntimeError(
                'Heatmap-control mixture training requires an explicit '
                'data.mixture.epoch_size'
            )
        requested_epoch_size = int(
            len(train_dataset)
            if configured_epoch_size is None
            else configured_epoch_size
        )
        epoch_size = (requested_epoch_size // global_batch) * global_batch
        if epoch_size <= 0:
            raise RuntimeError(
                'Mixture epoch_size must cover at least one full global batch'
            )
        if heatmap_control_resume_guard and epoch_size != requested_epoch_size:
            raise RuntimeError(
                'Heatmap-control mixture epoch_size must be divisible by the '
                f'global microbatch ({global_batch}); got {requested_epoch_size}'
            )
        sampler_kwargs = {
            'weights': mixture_cfg.get('weights'),
        } if mixture_cfg.get('weights') is not None else {
            'profile': mixture_cfg.get('profile', 'expert50_normal20_hard30'),
        }
        train_sampler = DeterministicMixtureSampler(
            train_dataset,
            epoch_size=epoch_size,
            seed=int(mixture_cfg.get('seed', cfg.get('seed', 42))),
            num_replicas=dist_context.world_size,
            rank=dist_context.rank,
            drop_last=True,
            **sampler_kwargs,
        )
        logger.info(
            '   ⚖️ Deterministic mixture sampler: profile=%s weights=%s '
            'epoch_size=%d (requested=%d)',
            getattr(train_sampler, 'profile', None),
            getattr(train_sampler, 'weights', None),
            epoch_size,
            requested_epoch_size,
        )
        if resume_path and Path(resume_path).exists():
            if resume_mixture_sampler_state is None:
                raise RuntimeError(
                    'Exact heatmap-control resume requires mixture_sampler_state'
                )
            train_sampler.load_state_dict(resume_mixture_sampler_state)
            logger.info(
                '   ✓ Restored deterministic mixture sampler contract: epoch=%d',
                train_sampler.epoch,
            )
    else:
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
    configured_in_order = cfg['data'].get('in_order')
    train_loader_in_order = (
        not num_workers > 0
        if configured_in_order is None
        else bool(configured_in_order)
    )
    if (
        dataset_type == 'expert_dagger_mixture'
        and heatmap_control_resume_guard
        and not train_loader_in_order
    ):
        raise RuntimeError(
            'Heatmap-control deterministic mixture requires data.in_order=true'
        )
    logger.info(
        '   📦 Training DataLoader completion order: in_order=%s',
        train_loader_in_order,
    )
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
        in_order=train_loader_in_order,
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
        logger.info(
            "   ✅ Dynamic sampling enabled; persistent_workers=False so each "
            "epoch receives the rebuilt sample index"
        )
    if cfg["log"].get("show_gpu_memory", False):
        logger.info(f"   🧠 Memory config: num_workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    if dist_context.enabled:
        logger.info(
            f"   🔀 DistributedSampler enabled: world_size={dist_context.world_size}, rank={dist_context.rank}"
        )

    # Materialize the backbone before constructing dependent heads and the
    # optimizer. This supports both legacy LoRA stages and the frozen-native
    # single-view visual extractor; the latter contains no LoRA parameters.
    raw_model = model
    vlm_backbone = getattr(raw_model, 'vlm_backbone', getattr(raw_model, 'qwen2_5_vl', None))
    if vlm_backbone is not None and hasattr(vlm_backbone, '_load_model'):
        if vlm_backbone.model is None:
            logger.info("🔄 Pre-loading VLM backbone before head/optimizer setup...")
            vlm_backbone._load_model()
        logger.info(
            "   🧠 Qwen attention implementation: %s",
            getattr(vlm_backbone.config, 'attn_implementation', 'unknown'),
        )
    if getattr(raw_model.config, 'enable_heatmap', False):
        logger.info("🔄 Constructing HeatmapVLN before optimizer setup...")
        raw_model._ensure_heatmap_vln()

    matched_lora_tensors = 0
    merged_lora_tensors = 0
    warmstart_contract = stage_cfg.get('heatmap_warmstart_contract') or {}
    warmstart_policy = warmstart_contract.get('policy')
    pose_adaptation_init = bool(stage_cfg.get('heatmap_pose_adaptation_init', False))
    single_view_artifact_loaded = False
    single_view_warmstart_report = None
    if (
        warmstart_policy == SINGLE_VIEW_WARMSTART_POLICY
        and not pose_adaptation_init
        and not args.load_weights
        and not (resume_path and Path(resume_path).exists())
    ):
        raise ValueError(
            "The single-view warm-start contract requires the derived "
            "heatmap-only artifact via --load-weights for a new run"
        )
    if (
        pose_adaptation_init
        and not args.load_weights
        and not (resume_path and Path(resume_path).exists())
    ):
        raise ValueError(
            "A new AMB3R pose-adaptation run requires the current GT-pose "
            "best.pth via --load-weights"
        )
    if args.load_weights and pose_adaptation_init:
        if resume_path and Path(resume_path).exists():
            raise ValueError(
                "Do not combine AMB3R pose-adaptation --load-weights with --resume"
            )
        weights_path = Path(args.load_weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pose-adaptation initializer not found: {weights_path}"
            )
        ppa_stage = stage_cfg.get('past_plan_action_stage')
        if ppa_stage is not None:
            ppa_reset_bridge = bool(
                stage_cfg.get('past_plan_action_reset_bridge', False)
            )
            single_view_warmstart_report = load_past_plan_action_initialization(
                raw_model,
                weights_path,
                stage=str(ppa_stage),
                load_trained_bridge=bool(
                    stage_cfg.get('past_plan_action_bridge_only', False)
                )
                and not ppa_reset_bridge,
            )
            if ppa_reset_bridge:
                logger.info(
                    "  ✓ PPA bridge retrains from its exact-zero fresh state "
                    "(past_plan_action_reset_bridge=true); the trained Stage-2 "
                    "bridge in the base checkpoint was intentionally not loaded"
                )
        else:
            single_view_warmstart_report = load_pose_adaptation_initialization(
                raw_model,
                weights_path,
            )
        single_view_artifact_loaded = True
        cfg.setdefault('runtime', {})['pose_adaptation_init_checkpoint'] = str(
            weights_path.resolve()
        )
        logger.info(
            "  ✓ Fresh-initialized complete Heatmap/PPA learned state from deployment/EMA "
            "state: tensors=%d (no hash lock; optimizer/scheduler fresh)",
            single_view_warmstart_report['loaded_tensor_count'],
        )
        if dist_context.is_main:
            _write_json(
                manifest_dir / 'amb3r_pose_adaptation_init.json',
                single_view_warmstart_report,
            )
        torch.cuda.empty_cache()
    elif args.load_weights and warmstart_policy == SINGLE_VIEW_WARMSTART_POLICY:
        if resume_path and Path(resume_path).exists():
            raise ValueError(
                "Do not combine a single-view initialization artifact with "
                "--resume. Resume already contains the complete trained head."
            )
        weights_path = Path(args.load_weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Single-view heatmap artifact not found: {weights_path}"
            )
        cfg.setdefault('runtime', {})['single_view_heatmap_init_artifact'] = str(
            weights_path.resolve()
        )
        single_view_warmstart_report = load_single_view_heatmap_artifact(
            raw_model,
            weights_path,
        )
        single_view_artifact_loaded = True
        logger.info(
            "  ✓ Loaded provenance-locked single-view heatmap initializer: "
            "tensors=%d source_sha256=%s migrated_state_sha256=%s",
            single_view_warmstart_report['loaded_tensor_count'],
            single_view_warmstart_report['source_checkpoint_sha256'],
            single_view_warmstart_report['selected_state_content_sha256'],
        )
        if dist_context.is_main:
            _write_json(
                manifest_dir / 'single_view_heatmap_warmstart.json',
                {
                    'artifact_path': str(weights_path.resolve()),
                    'artifact_file_sha256': single_view_artifact_sha256(
                        weights_path
                    ),
                    **single_view_warmstart_report,
                },
            )
        torch.cuda.empty_cache()

    if args.load_weights and not single_view_artifact_loaded:
        weights_path = Path(args.load_weights)
        if weights_path.exists():
            cfg.setdefault('runtime', {})['base_checkpoint'] = str(weights_path.resolve())
            ckpt = safe_torch_load(str(weights_path))
            state_dict = ckpt.get('trainable_state_dict', {})
            loaded_count = 0
            if state_dict:
                if requires_base_checkpoint:
                    matched_lora_tensors = assert_complete_lora_checkpoint_match(
                        raw_model,
                        state_dict,
                        checkpoint_path=str(weights_path),
                    )
                    logger.info(
                        "  ✓ Verified complete LoRA checkpoint match: %d tensors",
                        matched_lora_tensors,
                    )
                state_to_load = state_dict
                if stage_cfg.get('base_checkpoint_lora_only', False):
                    state_to_load = extract_lora_checkpoint_state(state_dict)
                    if not state_to_load:
                        raise RuntimeError(
                            f'Base checkpoint contains no LoRA tensors: {weights_path}'
                        )
                    logger.info(
                        '  🔒 Base checkpoint LoRA-only guard: loading %d/%d tensors; '
                        'InternNav System1 and adapter weights cannot be overwritten',
                        len(state_to_load),
                        len(state_dict),
                    )
                heatmap_warmstart_report = validate_heatmap_warmstart_contract(
                    raw_model,
                    state_to_load,
                    stage_cfg,
                    checkpoint_metadata=ckpt.get('metadata'),
                    checkpoint_path=str(weights_path),
                )
                missing, unexpected, loaded_count = _load_normalized_state_dict(
                    raw_model,
                    state_to_load,
                )
                verify_heatmap_warmstart_loaded(
                    raw_model,
                    heatmap_warmstart_report,
                    loaded_count=loaded_count,
                )
                if heatmap_warmstart_report is not None:
                    logger.info(
                        "  ✓ Heatmap warm-start contract passed: policy=%s "
                        "loaded=%d counts=%s",
                        heatmap_warmstart_report["policy"],
                        loaded_count,
                        heatmap_warmstart_report["counts"],
                    )
                logger.info(f"✓ Loaded {loaded_count} params from {weights_path.name} (weights only, fresh optimizer/scheduler)")
                if loaded_count < len(state_to_load):
                    logger.warning(f"  ⚠ Only {loaded_count}/{len(state_to_load)} checkpoint params matched!")
                if missing:
                    logger.info(f"  Missing keys (in model but not checkpoint): {len(missing)}")
                if unexpected:
                    logger.info(f"  Unexpected keys (in checkpoint but not model): {len(unexpected)}")
            else:
                logger.warning(f"⚠ No trainable_state_dict found in {weights_path}")
            if requires_base_checkpoint and loaded_count == 0:
                raise RuntimeError(
                    "Training stage did not load any parameters from the Stage1-S2 "
                    f"base checkpoint: {weights_path}"
                )
            del ckpt
            torch.cuda.empty_cache()
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if stage_cfg.get('name') == 'stage3':
        stage3_llm_cfg = cfg.get('model', {}).get('llm', {})
        logger.info(
            'Stage3 frozen-Qwen execution: merge_lora=%s '
            'inference_mode=%s last_hidden_state_only=%s',
            bool(stage_cfg.get('merge_frozen_lora', False)),
            bool(stage3_llm_cfg.get('frozen_traj_inference_mode', False)),
            bool(stage3_llm_cfg.get('traj_last_hidden_state_only', False)),
        )

    if stage_cfg.get('merge_frozen_lora', False):
        trainable_names = set(stage_cfg.get('trainable_modules', []))
        if trainable_names & {'lora', 'vlm_lora'}:
            raise RuntimeError(
                'merge_frozen_lora cannot be enabled when LoRA is trainable'
            )
        merge_lora = getattr(vlm_backbone, 'merge_lora_for_frozen_forward', None)
        if not callable(merge_lora):
            raise RuntimeError(
                'merge_frozen_lora requested but the VLM backbone does not '
                'provide merge_lora_for_frozen_forward'
            )
        merged_lora_tensors = merge_lora(safe_merge=True)
        logger.info(
            '  ✓ Stage3 frozen-Qwen optimization: merged %d LoRA tensors',
            merged_lora_tensors,
        )
        torch.cuda.empty_cache()

    heatmap_control_cfg = nextdit_cfg.get('heatmap_control') or {}
    frozen_heatmap_dependency = None
    if bool(heatmap_control_cfg.get('enabled', False)):
        dependency_path = heatmap_control_cfg.get('heatmap_checkpoint_path')
        dependency_sha256 = heatmap_control_cfg.get('heatmap_checkpoint_sha256')
        if not dependency_path or not dependency_sha256:
            raise ValueError(
                "heatmap control requires heatmap_checkpoint_path and "
                "heatmap_checkpoint_sha256"
            )
        logger.info(
            "🔒 Loading frozen single-view heatmap dependency with exact SHA/coverage..."
        )
        frozen_heatmap_dependency = load_frozen_heatmap_checkpoint(
            raw_model,
            dependency_path,
            expected_sha256=dependency_sha256,
        )
        cfg.setdefault('runtime', {})['frozen_heatmap_dependency'] = (
            frozen_heatmap_dependency
        )
        logger.info(
            "  ✓ Frozen heatmap dependency: tensors=%d sha256=%s",
            frozen_heatmap_dependency['tensor_count'],
            frozen_heatmap_dependency['checkpoint_sha256'],
        )
        if dist_context.is_main:
            _write_json(
                manifest_dir / 'frozen_heatmap_dependency.json',
                frozen_heatmap_dependency,
            )

    # 设置可训练模块
    logger.info("🔧 Setting trainable modules...")
    set_trainable_modules(raw_model, stage_cfg, logger)

    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    if resume_l2_sp_reference:
        l2_sp_reference = resume_l2_sp_reference
        l2_sp_numel = sum(t.numel() for t in l2_sp_reference.values() if torch.is_tensor(t))
        logger.info(
            "  L2-SP reference restored from checkpoint: %d tensors / %s params",
            len(l2_sp_reference),
            f"{l2_sp_numel:,}",
        )
    else:
        l2_sp_reference = build_l2_sp_reference(raw_model, cfg, logger=logger)
    stage3_l2_cfg = cfg.get('loss', {}).get('l2_sp', {})
    if (
        stage_name == 'stage3'
        and bool(stage3_l2_cfg.get('enabled', False))
        and float(stage3_l2_cfg.get('weight', 0.0) or 0.0) > 0.0
        and not l2_sp_reference
    ):
        raise RuntimeError(
            'Stage3 L2-SP is enabled but no trainable parameters matched its '
            'reference. Include pano_latent_adapter in loss.l2_sp.modules.'
        )

    # 构建优化器和调度器
    optimizer = build_optimizer(raw_model, cfg, stage_cfg)
    single_view_static_report = assert_single_view_training_contract(
        raw_model,
        optimizer,
        cfg,
        stage_cfg,
    )
    if single_view_static_report:
        logger.info(
            "  ✓ Single-view frozen-native safety contract: %s",
            single_view_static_report,
        )

    nextdit_warmup_steps = apply_nextdit_warmup_freeze(raw_model, cfg, logger)

    grad_accum_steps = cfg['optim'].get('grad_accum_steps', 1)
    batches_per_epoch = len(train_loader)
    if batches_per_epoch < 1:
        raise RuntimeError(
            'Training DataLoader has zero full batches. Increase the dataset size '
            'or reduce optim.batch_size/world_size.'
        )
    # train_one_epoch performs an optimizer step for the final partial
    # accumulation window, so scheduler sizing and resume offsets must use
    # ceil here as well.  Flooring drifts by one step per odd-sized epoch.
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
    total_steps = steps_per_epoch * total_epochs
    if total_steps < 1:
        raise RuntimeError(
            'Training schedule has zero optimizer steps. Reduce '
            'optim.grad_accum_steps or increase the number of batches.'
        )
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    amp_type = cfg['optim'].get('amp', 'bf16')
    scaler = make_grad_scaler(dist_context.device, amp_type)

    # EMA must exist before resume so its shadow and step counter are restored
    # together with the optimizer-matched online weights.
    ema_decay = cfg.get('optim', {}).get('ema_decay', 0.999)
    ema_warmup = cfg.get('optim', {}).get('ema_warmup_steps', 2000)
    ema = EMAModel(raw_model, decay=ema_decay, warmup_steps=ema_warmup)
    logger.info(f"📐 EMA enabled: decay={ema_decay}, warmup_steps={ema_warmup}")

    if resume_path and Path(resume_path).exists():
        if heatmap_control_resume_guard:
            validate_heatmap_control_resume_checkpoint(
                str(resume_path), raw_model, cfg
            )
        load_checkpoint_for_resume(
            str(resume_path), raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            logger=logger,
            strict_state_restore=heatmap_control_resume_guard,
        )
    converted_heatmap_states = ensure_heatmap_optimizer_state_fp32(
        optimizer,
        require_fp32_params=bool(stage_cfg.get('heatmap_fp32', True)),
    )
    if converted_heatmap_states:
        logger.info(
            "  ✓ Converted %d restored Heatmap AdamW state tensors to FP32",
            converted_heatmap_states,
        )

    best_metric_value = ckpt_manager.best_metric_value
    if resume_epoch > 0:
        if resume_skip_batches is not None:
            # Mid-epoch checkpoint: resume the same epoch starting from
            # resume_skip_batches.  The saved batch landed *after* the
            # gradient-accumulation step so the next batch to process is
            # resume_skip_batches + 1.  We record resume_skip_batches as
            # the skip count so train_one_epoch can fast-forward the
            # dataloader iterator.
            start_epoch = resume_epoch
            resume_skip_batches = resume_skip_batches  # pass through as-is
            logger.info(
                "📂 Mid-epoch resume: epoch=%d skip=%d batches",
                start_epoch,
                resume_skip_batches,
            )
        else:
            start_epoch = resume_epoch + 1
            resume_skip_batches = None
        global_epoch_counter = resume_epoch
    else:
        start_epoch = args.start_epoch
        global_epoch_counter = start_epoch - 1

    patience = cfg['validation'].get('patience', 5)
    configured_eval_every_epochs = int(
        cfg.get('validation', {}).get('eval_every_epochs', 1)
    )
    if validation_enabled and configured_eval_every_epochs < 1:
        raise ValueError(
            "validation.eval_every_epochs must be >=1 when validation is enabled"
        )
    eval_every_epochs = (
        configured_eval_every_epochs if validation_enabled else 0
    )
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

    log_cfg = cfg.get('log', {})
    page_cache_drop_enabled = bool(log_cfg.get('page_cache_drop_enabled', True))
    page_cache_drop_threshold = float(log_cfg.get('page_cache_drop_threshold', 0.80))
    initial_page_cache_drop = bool(log_cfg.get('initial_page_cache_drop', page_cache_drop_enabled))
    if page_cache_drop_enabled and initial_page_cache_drop:
        _drop_page_cache(force=True, threshold=page_cache_drop_threshold)
    if cfg['log'].get('show_gpu_memory', False):
        cg_init = _cgroup_mem_usage_gb()
        logger.info(f"  cgroup memory after initial page cache drop: {cg_init:.1f}/{_CG_LIMIT_GB:.0f}GB")

    if args.dry_run:
        logger.info("=" * 80)
        logger.info("🧪 REAL TRAINING PREFLIGHT: no checkpoint will be saved")
        logger.info("=" * 80)
        preflight_metrics = run_training_preflight(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            cfg,
            logger,
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            train_dataset=train_dataset,
            train_sampler=train_sampler,
            gpu_heatmap_computer=gpu_heatmap_computer,
            gpu_has_depth=gpu_has_depth,
            gpu_depth_normalized=gpu_depth_normalized,
            ema=ema,
            total_train_steps=total_steps,
            dist_context=dist_context,
            nextdit_warmup_steps=nextdit_warmup_steps,
            l2_sp_reference=l2_sp_reference,
        )
        _dist_barrier()
        if dist_context.is_main:
            checkpoint_files = sorted(
                str(path.relative_to(run_dir))
                for path in ckpt_dir.rglob('*')
                if path.is_file()
            )
            if checkpoint_files:
                raise RuntimeError(
                    'Dry-run preflight unexpectedly wrote checkpoint files: '
                    f'{checkpoint_files}'
                )
            preflight_record = {
                'record_type': 'preflight_pass',
                'status': 'passed',
                'stage': stage_name,
                'world_size': dist_context.world_size,
                'batch_size_per_rank': cfg['optim']['batch_size'],
                'global_batch_size': cfg['optim']['batch_size'] * dist_context.world_size,
                'num_workers_per_rank': num_workers,
                'prefetch_factor': prefetch_factor,
                'system1_source': system1_source,
                'system1_required_tensors': system1_required_tensors,
                'matched_lora_tensors': matched_lora_tensors,
                'merged_lora_tensors': merged_lora_tensors,
                'base_checkpoint_lora_only': bool(
                    stage_cfg.get('base_checkpoint_lora_only', False)
                ),
                'metrics': preflight_metrics,
                'peak_gpu_memory_gb_rank0': (
                    torch.cuda.max_memory_allocated(dist_context.device) / 1024**3
                    if dist_context.device.type == 'cuda'
                    else 0.0
                ),
                'checkpoint_files': checkpoint_files,
            }
            _write_json(manifest_dir / 'preflight.json', preflight_record)
            _append_jsonl(metrics_jsonl_path, preflight_record)
            logger.info(
                '✅ REAL TRAINING PREFLIGHT PASSED: System1=%d tensors, '
                'LoRA matched/merged=%d/%d tensors, '
                'optimizer_steps=1, checkpoint_files=0',
                system1_required_tensors,
                matched_lora_tensors,
                merged_lora_tensors,
            )
            logger.info('   Report: %s', manifest_dir / 'preflight.json')
        _dist_barrier()
        if tb_writer is not None:
            tb_writer.close()
        cleanup_distributed()
        return

    evaluate_before_training = bool(
        validation_cfg.get('evaluate_before_training', False)
    )
    if evaluate_before_training and not validation_enabled:
        raise ValueError(
            "validation.evaluate_before_training=true requires "
            "validation.enabled=true"
        )
    if (
        constrained_selection
        and checkpoint_selector.baseline_metrics is None
        and not evaluate_before_training
    ):
        raise ValueError(
            "validation.baseline_as_best_threshold=true requires "
            "validation.evaluate_before_training=true for a new run"
        )

    should_evaluate_baseline = validation_enabled and evaluate_before_training and (
        not constrained_selection
        or checkpoint_selector.baseline_metrics is None
    )
    if should_evaluate_baseline:
        if val_loader is None:
            raise ValueError(
                "validation.evaluate_before_training=true requires an enabled "
                "validation dataset"
            )
        logger.info("📊 Running pre-training validation baseline...")
        with ema.apply():
            baseline_val_metrics = validate(
                model,
                val_loader,
                cfg,
                logger,
                stage_cfg,
                tb_writer,
                epoch=0,
                vis_dir=vis_val_dir,
                max_batches=args.max_batches,
                gpu_heatmap_computer=gpu_heatmap_computer,
                gpu_has_depth=gpu_has_depth,
                gpu_depth_normalized=gpu_depth_normalized,
                dist_context=dist_context,
            )
        if save_best_metric not in baseline_val_metrics:
            raise KeyError(
                "validation.save_best_metric is absent from pre-training "
                f"baseline output: {save_best_metric!r}; "
                f"available={sorted(baseline_val_metrics)}"
            )
        baseline_metric_value = float(
            baseline_val_metrics[save_best_metric]
        )
        baseline_as_best_threshold = bool(
            validation_cfg.get('baseline_as_best_threshold', False)
        )
        baseline_selection_record = None
        if baseline_as_best_threshold:
            baseline_selection_record = checkpoint_selector.set_baseline(
                baseline_val_metrics,
                epoch=0,
            )
        baseline_threshold_installed = _install_baseline_best_threshold(
            ckpt_manager,
            baseline_metric_value,
            enabled=baseline_as_best_threshold,
        )
        if baseline_threshold_installed:
            best_metric_value = baseline_metric_value
        if baseline_as_best_threshold and not baseline_threshold_installed:
            raise RuntimeError(
                "Failed to install the exact step-0 validation result as "
                "the checkpoint-selection incumbent"
            )
        if dist_context.is_main:
            baseline_record = {
                "record_type": "pre_training_validation",
                "epoch": 0,
                "stage": stage_name,
                "save_best_metric": save_best_metric,
                "save_best_mode": save_best_mode,
                "baseline_as_best_threshold": baseline_as_best_threshold,
                "metrics": baseline_val_metrics,
                "checkpoint_selection": baseline_selection_record,
            }
            _write_json(manifest_dir / "pre_training_validation.json", baseline_record)
            _append_jsonl(metrics_jsonl_path, baseline_record)
            if baseline_selection_record is not None:
                _write_json(
                    manifest_dir / "checkpoint_selection.json",
                    {
                        "policy": checkpoint_selector.state_dict()["config"],
                        "baseline": baseline_selection_record,
                        "state": checkpoint_selector.state_dict(),
                        "latest_decision": None,
                    },
                )
            logger.info(
                "  Baseline %s=%.6f (%s)",
                save_best_metric,
                baseline_metric_value,
                (
                    "installed as best threshold"
                    if baseline_threshold_installed
                    else (
                        "existing best threshold retained"
                        if baseline_as_best_threshold
                        else "recorded only"
                    )
                ),
            )
            if baseline_as_best_threshold:
                baseline_extra_state = {
                    "checkpoint_kind": "pre_training_baseline",
                    "checkpoint_selection_state": (
                        checkpoint_selector.state_dict()
                    ),
                }
                if isinstance(train_sampler, DeterministicMixtureSampler):
                    baseline_extra_state["mixture_sampler_state"] = (
                        train_sampler.state_dict()
                    )
                if l2_sp_reference:
                    baseline_extra_state["l2_sp_reference_state"] = (
                        l2_sp_reference
                    )
                ckpt_manager.save(
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=0,
                    stage_idx=0,
                    stage_name=stage_name,
                    metrics=baseline_val_metrics,
                    cfg=cfg,
                    is_best=True,
                    scaler=scaler,
                    extra_state=baseline_extra_state,
                    ema=ema,
                    best_only=True,
                )
        _dist_barrier()
        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()

    timer = TrainingTimer(total_epochs=total_epochs)
    timer.start()

    for epoch in range(start_epoch, total_epochs + 1):
        timer.start_epoch()

        if uses_dynamic_sampling:
            train_dataset.set_epoch(epoch)
            logger.info(
                f"   🔄 Resampled {len(train_dataset)} samples for epoch {epoch}; "
                "workers will be rebuilt"
            )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        logger.info("=" * 80)
        logger.info(f"[{stage_name}] Epoch {epoch}/{total_epochs}")
        logger.info("=" * 80)

        epoch_offset = (epoch - 1) * steps_per_epoch
        skip_batches = resume_skip_batches if epoch == start_epoch else None
        if skip_batches is not None:
            logger.info(
                "⏭️  Skipping first %d batches (mid-epoch resume)",
                skip_batches,
            )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, logger, tb_writer, epoch_offset,
            stage_idx=0, stage_name=stage_name, stage_cfg=stage_cfg,
            max_batches=args.max_batches,
            skip_first_n_batches=skip_batches,
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
            l2_sp_reference=l2_sp_reference,
            checkpoint_selection_state=(
                checkpoint_selector.state_dict()
                if best_selection_enabled
                else None
            ),
            train_sampler=train_sampler,
        )

        timer.end_epoch()

        gc.collect()
        torch.cuda.empty_cache()
        _malloc_trim()
        if page_cache_drop_enabled:
            _drop_page_cache(threshold=page_cache_drop_threshold)

        do_eval = bool(
            validation_enabled
            and val_loader is not None
            and (
                (epoch % eval_every_epochs == 0)
                or (epoch == total_epochs)
            )
        )

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
            if page_cache_drop_enabled:
                _drop_page_cache(threshold=page_cache_drop_threshold)
        else:
            if val_loader is None:
                if not cfg.get('validation', {}).get('enabled', True):
                    logger.info("  ⏭️  跳过验证（validation.enabled=false）")
                else:
                    logger.info("  ⏭️  跳过验证（未配置验证集）")
            else:
                logger.info(
                    f"  ⏭️  跳过验证（eval_every_epochs={eval_every_epochs}，将在 epoch "
                    f"{epoch + eval_every_epochs - (epoch % eval_every_epochs)} 验证）"
                )

        if cfg['log'].get('show_gpu_memory', False):
            process = psutil.Process()
            mem_info = process.memory_info()
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"  🧠 Memory: CPU={mem_info.rss / (1024**3):.2f}GB, GPU={gpu_mem:.2f}GB (reserved={gpu_reserved:.2f}GB)")

        train_traj_str = f", traj: {train_metrics['trajectory_loss']:.4f}" if train_metrics.get('trajectory_loss', 0) > 0 else ""
        train_lm_str = f", lm: {train_metrics['lm_loss']:.4f}" if train_metrics.get('lm_loss', 0) > 0 else ""
        train_l2_sp_str = f", l2sp: {train_metrics['l2_sp_loss']:.6f}" if train_metrics.get('l2_sp_loss', 0) > 0 else ""
        logger.info(
            f"  Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f}{train_traj_str}{train_lm_str}{train_l2_sp_str})"
        )

        eta = timer.get_eta(epoch, total_epochs)
        logger.info(f"  ⏱️  Epoch time: {timer.get_epoch_time()} | ETA: {eta}")

        checkpoint_selection_decision = None
        if do_eval and val_metrics:
            val_hm_mse_str = f", infer_mse: {val_metrics['val_heatmap_mse']:.6f}" if val_metrics.get('val_heatmap_mse', 0) > 0 else ""
            val_traj_str = f", traj: {val_metrics['val_trajectory_loss']:.4f}" if val_metrics.get('val_trajectory_loss', 0) > 0 else ""
            val_lm_str = f", lm: {val_metrics['val_lm_loss']:.4f}" if val_metrics.get('val_lm_loss', 0) > 0 else ""
            logger.info(
                f"  Val Loss: {val_metrics['val_loss']:.4f} "
                f"(hm: {val_metrics['val_heatmap_loss']:.4f}{val_traj_str}{val_lm_str}{val_hm_mse_str})"
            )
            is_best = False
            if best_selection_enabled:
                if save_best_metric not in val_metrics:
                    raise KeyError(
                        "validation.save_best_metric is absent from validation "
                        f"output: {save_best_metric!r}; "
                        f"available={sorted(val_metrics)}"
                    )
                selected_metric_value = float(val_metrics[save_best_metric])
                ckpt_manager.best_val_loss = min(
                    ckpt_manager.best_val_loss,
                    float(val_metrics['val_loss']),
                )
                checkpoint_selection_decision = checkpoint_selector.consider(
                    val_metrics,
                    epoch=epoch,
                )
                is_best = bool(
                    checkpoint_selection_decision["accepted_as_best"]
                )
                if is_best:
                    best_metric_value = selected_metric_value
                    ckpt_manager.best_metric_value = selected_metric_value
                    no_improve_count = 0
                    logger.info(
                        "  ⭐ New best %s: %.6f (%s)",
                        save_best_metric,
                        best_metric_value,
                        save_best_mode,
                    )
                else:
                    no_improve_count += 1
                    logger.info(
                        "  Checkpoint candidate rejected: %s",
                        "; ".join(
                            checkpoint_selection_decision["reason_details"]
                        ),
                    )
        else:
            is_best = False

        global_epoch_counter += 1
        current_lr = scheduler.get_last_lr()[0] if scheduler else 0

        if dist_context.is_main:
            if checkpoint_selection_decision is not None:
                _append_jsonl(
                    metrics_jsonl_path,
                    checkpoint_selection_decision,
                )
                _write_json(
                    manifest_dir / "checkpoint_selection.json",
                    {
                        "policy": (
                            checkpoint_selector.state_dict()["config"]
                        ),
                        "baseline_metrics": (
                            checkpoint_selector.baseline_metrics
                        ),
                        "state": checkpoint_selector.state_dict(),
                        "latest_decision": (
                            checkpoint_selection_decision
                        ),
                    },
                )
            _append_jsonl(
                metrics_jsonl_path,
                {
                    "record_type": "epoch_summary",
                    "epoch": epoch,
                    "global_epoch": global_epoch_counter,
                    "stage": stage_name,
                    "is_best": is_best,
                    "checkpoint_selection": (
                        checkpoint_selection_decision
                    ),
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
            if train_metrics.get('trajectory_loss', 0) > 0:
                tb_writer.add_scalar('epoch/train_trajectory_loss', train_metrics['trajectory_loss'], global_epoch_counter)
            if train_metrics.get('l2_sp_loss', 0) > 0:
                tb_writer.add_scalar('epoch/train_l2_sp_loss', train_metrics['l2_sp_loss'], global_epoch_counter)

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

                for hm_key in (
                    'peak_loss',
                    'vis_loss',
                    'coord_loss',
                    'neg_loss',
                    'view_macro_loss',
                    'direction_macro_loss',
                    'panoramic_view_loss',
                ):
                    train_key = f'hm_{hm_key}'
                    val_key = f'val_hm_{hm_key}'
                    if train_key in train_metrics:
                        tb_writer.add_scalar(
                            f'epoch/train_hm_{hm_key}',
                            train_metrics[train_key],
                            global_epoch_counter,
                        )
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
                sent = notifier.send_epoch_report(
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
                    best_val_loss=best_metric_value,
                )
                _log_notification_result(
                    logger, sent, f"Epoch {epoch}", getattr(notifier, "last_error", None)
                )
            except Exception as e:
                logger.warning(f"飞书通知发送失败: {e}")

        if epoch % cfg['log']['save_every_epochs'] == 0 or is_best:
            if dist_context.is_main:
                checkpoint_extra_state = {}
                if best_selection_enabled:
                    checkpoint_extra_state["checkpoint_selection_state"] = (
                        checkpoint_selector.state_dict()
                    )
                if isinstance(train_sampler, DeterministicMixtureSampler):
                    checkpoint_extra_state["mixture_sampler_state"] = (
                        train_sampler.state_dict()
                    )
                if checkpoint_selection_decision is not None:
                    checkpoint_extra_state[
                        "checkpoint_selection_decision"
                    ] = checkpoint_selection_decision
                if l2_sp_reference:
                    checkpoint_extra_state["l2_sp_reference_state"] = (
                        l2_sp_reference
                    )
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
                    extra_state=checkpoint_extra_state,
                    ema=ema,
                )
            _dist_barrier()

        if best_selection_enabled and no_improve_count >= patience:
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
        "best_metric_name": save_best_metric,
        "best_metric_mode": save_best_mode,
        "best_metric_value_runtime": best_metric_value,
        "best_checkpoint_selection_enabled": best_selection_enabled,
        "checkpoint_selection_state": (
            checkpoint_selector.state_dict()
            if best_selection_enabled
            else None
        ),
        # Kept for consumers that still expect this historical field name.
        "best_val_loss_runtime": ckpt_manager.best_val_loss,
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
                "best_val_loss": ckpt_manager.best_val_loss,
                "best_metric_name": save_best_metric,
                "best_metric_mode": save_best_mode,
                "best_metric_value": best_metric_value,
            },
        )

    if notifier:
        try:
            sent = notifier.send_training_complete(
                total_time=timer.get_total_elapsed() if timer else "N/A",
                best_val_loss=best_metric_value,
                final_stage=stage_name,
            )
            _log_notification_result(
                logger, sent, "训练完成", getattr(notifier, "last_error", None)
            )
        except Exception as e:
            logger.warning(f"飞书通知发送失败: {e}")

    if tb_writer is not None:
        tb_writer.close()
    cleanup_distributed()


if __name__ == '__main__':
    main()
