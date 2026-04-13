"""
training — modular training utilities for HeatmapVLN.

Re-exports every public symbol so ``from scripts.training import X`` works
from the repository entrypoints.
"""

from .checkpoint import (
    CheckpointManager,
    load_checkpoint_for_resume,
)
from .collate import collate_fn
from .distributed import (
    DistributedContext,
    _dist_all_reduce_in_place,
    _dist_barrier,
    _dist_broadcast_in_place,
    _get_supported_trainable_sync_modules,
    cleanup_distributed,
    init_distributed_context,
    initialize_trainable_module_sync,
    synchronize_trainable_module_gradients,
)
from .ema import EMAModel, _EMAContext
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
from .model_builder import (
    apply_nextdit_warmup_freeze,
    build_model,
    end_nextdit_warmup,
    freeze_module,
    set_trainable_modules,
)
from .optimizer import (
    build_optimizer,
    build_scheduler,
    get_heatmap_temperature,
)
from .plotter import TrainingPlotter
from .timer import TrainingTimer
from .train_loop import train_one_epoch
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
    build_heatmap_loss_fn,
    load_config,
    safe_torch_load,
    set_seed,
)
from .validate import validate
from .visualization import (
    _select_primary_heatmap_slice,
    _should_use_gpu_gt,
    visualize_heatmap_predictions,
)
