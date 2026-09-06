"""
training — modular training utilities for HeatmapVLN.

Re-exports every public symbol so ``from scripts.training import X`` works
from the repository entrypoints.

Uses ``__getattr__`` for lazy loading so that importing the package does
not pull in heavy dependencies (tqdm, transformers, etc.) unless the
caller actually accesses a symbol that needs them.
"""

import importlib

# Lightweight modules — always imported eagerly (no heavy deps)
from .collate import collate_fn
from .manifest import (
    _append_jsonl,
    _capture_env_state,
    _capture_git_state,
    _clear_directory,
    _find_resume_checkpoint,
    _make_json_safe,
    _run_git_command,
    _safe_symlink,
    _write_json,
    _write_yaml,
)
from .source_fingerprint import (
    SOURCE_FINGERPRINT_ENV,
    check_pinned_source_fingerprint,
    compute_source_fingerprint,
)
from .memory import (
    _CG_LIMIT_GB,
    ShmBypassCollate,
    ShmBypassDataset,
    _cgroup_mem_limit_gb,
    _cgroup_mem_usage_gb,
    _drop_page_cache,
    _malloc_trim,
    _worker_init_fn,
)
from .utils import (
    _dist_backend,
    _dist_is_initialized,
    _format_decode_internal_timing,
    _format_qwen_internal_timing,
    _get_trainable_params,
    _load_normalized_state_dict,
    _mean_timing,
    _normalize_state_key,
    _normalized_model_state_dict,
    _normalized_trainable_param_names,
    _unwrap_model,
    assert_complete_lora_checkpoint_match,
    build_l2_sp_reference,
    build_heatmap_loss_fn,
    compute_l2_sp_loss,
    extract_lora_checkpoint_state,
    load_config,
    make_autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
    safe_torch_load,
    set_seed,
)

# Heavy modules — loaded lazily via __getattr__ to avoid importing
# tqdm, transformers, psutil, cv2, matplotlib at package import time.
_LAZY_MODULES = {
    # checkpoint.py
    "CheckpointManager": "checkpoint",
    "load_checkpoint_for_resume": "checkpoint",
    # distributed.py
    "DistributedContext": "distributed",
    "_dist_all_reduce_in_place": "distributed",
    "_dist_barrier": "distributed",
    "_dist_broadcast_in_place": "distributed",
    "_get_supported_trainable_sync_modules": "distributed",
    "cleanup_distributed": "distributed",
    "init_distributed_context": "distributed",
    "initialize_trainable_module_sync": "distributed",
    "synchronize_trainable_module_gradients": "distributed",
    # ema.py
    "EMAModel": "ema",
    "_EMAContext": "ema",
    # model_builder.py
    "apply_nextdit_warmup_freeze": "model_builder",
    "assert_complete_internnav_system1_load": "model_builder",
    "build_model": "model_builder",
    "end_nextdit_warmup": "model_builder",
    "ensure_trainable_heatmap_fp32": "model_builder",
    "freeze_module": "model_builder",
    "set_trainable_modules": "model_builder",
    # optimizer.py
    "build_optimizer": "optimizer",
    "build_scheduler": "optimizer",
    "ensure_heatmap_optimizer_state_fp32": "optimizer",
    "get_heatmap_temperature": "optimizer",
    # pose_adaptation.py
    "load_past_plan_action_initialization": "pose_adaptation",
    "load_pose_adaptation_initialization": "pose_adaptation",
    # heatmap_warmstart.py
    "validate_heatmap_warmstart_contract": "heatmap_warmstart",
    "verify_heatmap_warmstart_loaded": "heatmap_warmstart",
    # frozen_heatmap_checkpoint.py
    "FrozenHeatmapCheckpointError": "frozen_heatmap_checkpoint",
    "load_frozen_heatmap_checkpoint": "frozen_heatmap_checkpoint",
    # heatmap_control_resume.py
    "HeatmapControlResumeError": "heatmap_control_resume",
    "heatmap_control_enabled": "heatmap_control_resume",
    "reject_heatmap_control_load_weights": "heatmap_control_resume",
    "validate_heatmap_control_resume_checkpoint": "heatmap_control_resume",
    # preflight.py
    "run_training_preflight": "preflight",
    # plotter.py
    "TrainingPlotter": "plotter",
    # timer.py
    "TrainingTimer": "timer",
    # train_loop.py
    "train_one_epoch": "train_loop",
    # validate.py
    "validate": "validate",
    # visualization.py
    "_select_primary_heatmap_slice": "visualization",
    "_should_use_gpu_gt": "visualization",
    "visualize_heatmap_predictions": "visualization",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = importlib.import_module(f".{_LAZY_MODULES[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
