"""
training — modular training utilities for HeatmapVLN.

Re-exports every public symbol so ``from scripts.training import X`` works
from the repository entrypoints.
"""

from .utils import (
    _dist_is_initialized,
    _unwrap_model,
    _dist_backend,
    _normalize_state_key,
    _normalized_model_state_dict,
    _normalized_trainable_param_names,
    _load_normalized_state_dict,
    _get_trainable_params,
    _mean_timing,
    _format_qwen_internal_timing,
    _format_decode_internal_timing,
    load_config,
    set_seed,
)

from .distributed import (
    DistributedContext,
    _dist_barrier,
    _dist_broadcast_in_place,
    _dist_all_reduce_in_place,
    _get_supported_trainable_sync_modules,
    initialize_trainable_module_sync,
    synchronize_trainable_module_gradients,
    init_distributed_context,
    cleanup_distributed,
)

from .memory import (
    _malloc_trim,
    _cgroup_mem_usage_gb,
    _cgroup_mem_limit_gb,
    _CG_LIMIT_GB,
    _drop_page_cache,
    _worker_init_fn,
)

from .ema import EMAModel, _EMAContext

from .timer import TrainingTimer

from .plotter import TrainingPlotter

from .visualization import (
    visualize_heatmap_predictions,
    _should_use_gpu_gt,
    _select_primary_heatmap_slice,
)

from .collate import collate_fn

from .model_builder import (
    build_model,
    freeze_module,
    set_trainable_modules,
    apply_nextdit_warmup_freeze,
    end_nextdit_warmup,
)

from .optimizer import (
    build_optimizer,
    build_scheduler,
    get_heatmap_temperature,
)

from .train_loop import train_one_epoch

from .validate import validate

from .checkpoint import (
    CheckpointManager,
    load_checkpoint_for_resume,
)

from .manifest import (
    _make_json_safe,
    _write_json,
    _write_yaml,
    _append_jsonl,
    _safe_symlink,
    _clear_directory,
    _run_git_command,
    _capture_git_state,
    _capture_env_state,
    _find_resume_checkpoint,
)
