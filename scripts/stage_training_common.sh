#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${HEATMAPVLN_INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}}"
INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export INTERNNAV_MODEL_PATH
export INTERNNAV_BACKBONE

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
STAGE_DRY_RUN="${STAGE_DRY_RUN:-${PIPELINE_DRY_RUN:-0}}"
KEEP_TMP_CONFIGS="${KEEP_TMP_CONFIGS:-$STAGE_DRY_RUN}"

FEISHU_NOTIFY="${FEISHU_NOTIFY:-0}"
FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"
export FEISHU_WEBHOOK_URL

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

TMP_CONFIGS=()

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_tmp_configs() {
  case "${KEEP_TMP_CONFIGS:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      if ((${#TMP_CONFIGS[@]} > 0)); then
        printf '[%s] Keeping temporary configs: %s\n' "$(timestamp)" "${TMP_CONFIGS[*]}"
      fi
      return
      ;;
  esac
  if ((${#TMP_CONFIGS[@]} > 0)); then
    rm -f "${TMP_CONFIGS[@]}"
  fi
}
trap cleanup_tmp_configs EXIT

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "Missing required file: $path"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    log "Missing required directory: $path"
    exit 1
  fi
}

require_hf_model_dir() {
  local path="$1"
  require_dir "$path"
  require_file "$path/config.json"
}

count_csv_items() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
}

preflight_gpu() {
  local gpu_count
  gpu_count="$(count_csv_items "$GPU_DEVICES")"
  if [[ "$gpu_count" != "$NPROC_PER_NODE" ]]; then
    log "GPU_DEVICES has ${gpu_count} entries but NPROC_PER_NODE=${NPROC_PER_NODE}."
    log "Set both consistently, e.g. GPU_DEVICES=0,1,2,3 NPROC_PER_NODE=4."
    exit 1
  fi
}

preflight_notify() {
  local enabled="${1:-$FEISHU_NOTIFY}"
  if is_truthy "$enabled" && [[ -z "$FEISHU_WEBHOOK_URL" ]]; then
    log "FEISHU notification is enabled but FEISHU_WEBHOOK_URL is empty."
    exit 1
  fi
}

choose_checkpoint() {
  local ckpt_dir="$1"
  local preference="${2:-best}"
  local first
  local second

  if [[ "$preference" == "latest" ]]; then
    first="${ckpt_dir}/latest.pth"
    second="${ckpt_dir}/best.pth"
  else
    first="${ckpt_dir}/best.pth"
    second="${ckpt_dir}/latest.pth"
  fi

  if [[ -f "$first" ]]; then
    printf '%s\n' "$first"
    return 0
  fi
  if [[ -f "$second" ]]; then
    printf '%s\n' "$second"
    return 0
  fi

  log "No checkpoint found in $ckpt_dir (looked for best.pth/latest.pth)"
  exit 1
}

prepare_stage2_assets() {
  if [[ -n "$STAGE2_SYSTEM1_CKPT" ]]; then
    require_file "$STAGE2_SYSTEM1_CKPT"
  else
    require_hf_model_dir "$STAGE2_INTERNNAV_MODEL"
  fi

  STAGE2_EFFECTIVE_DAV2_CKPT=""
  if [[ -n "$STAGE2_DAV2_CKPT" && -f "$STAGE2_DAV2_CKPT" ]]; then
    STAGE2_EFFECTIVE_DAV2_CKPT="$STAGE2_DAV2_CKPT"
  elif is_truthy "$STAGE2_REQUIRE_DAV2_CKPT"; then
    require_file "$STAGE2_DAV2_CKPT"
  else
    log "DepthAnythingV2 checkpoint not configured; Stage2 will rely on InternNav full-model/System1 weights"
  fi
  export STAGE2_EFFECTIVE_DAV2_CKPT
}

make_stage_config() {
  local stage_prefix="$1"
  local base_config="$2"
  local output_config="$3"

  STAGE_PREFIX="$stage_prefix" python - "$base_config" "$output_config" <<'PY'
import os
import sys

import yaml

from src.config_schema import prepare_config_for_use

base_config, output_config = sys.argv[1], sys.argv[2]
prefix = os.environ["STAGE_PREFIX"]


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{prefix}_{name}", default)


def env_bool(name: str, default: str = "false") -> bool:
    return str(env(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def set_int(section: dict, key: str, value: str | None) -> None:
    if value is not None and str(value).strip() != "":
        section[key] = int(value)


def set_bool(section: dict, key: str, value: str | None) -> None:
    if value is not None and str(value).strip() != "":
        section[key] = str(value).strip().lower() in {"1", "true", "yes", "on"}


with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = prepare_config_for_use(cfg)

data = cfg.setdefault("data", {})
root = env("DATA_ROOT")
if root:
    data["root"] = root
if f"{prefix}_VAL_ROOT" in os.environ:
    val_root = env("VAL_ROOT")
    data["val_root"] = val_root if val_root else None
set_int(data, "num_workers", env("NUM_WORKERS"))
set_int(data, "prefetch_factor", env("PREFETCH_FACTOR"))
if env("PIN_MEMORY") is not None:
    data["pin_memory"] = env_bool("PIN_MEMORY")

trajectory = data.setdefault("trajectory", {})
set_int(trajectory, "num_history_sample", env("NUM_HISTORY_SAMPLE"))
set_bool(trajectory, "panoramic_vlm_input", env("PANORAMIC_VLM_INPUT"))
set_bool(trajectory, "random_subsequence", env("RANDOM_SUBSEQUENCE"))
set_bool(trajectory, "enable_trajectory_augmentation", env("ENABLE_TRAJECTORY_AUGMENTATION"))
set_int(trajectory, "max_clips", env("MAX_CLIPS"))

llm = cfg.setdefault("model", {}).setdefault("llm", {})
attn_impl = env("LLM_ATTN_IMPLEMENTATION")
if attn_impl:
    llm["attn_implementation"] = attn_impl
set_bool(llm, "gradient_checkpointing", env("LLM_GRADIENT_CHECKPOINTING"))

heatmap = cfg.setdefault("model", {}).setdefault("heatmap", {})
set_bool(heatmap, "enable", env("HEATMAP_ENABLE"))

optim = cfg.setdefault("optim", {})
set_int(optim, "batch_size", env("BATCH_SIZE"))
set_int(optim, "grad_accum_steps", env("GRAD_ACCUM_STEPS"))

gpu = cfg.setdefault("gpu", {})
visible_device_count = len([
    token for token in os.environ["GPU_DEVICES"].split(",")
    if token.strip()
])
gpu["devices"] = list(range(visible_device_count))
gpu.setdefault("multi_gpu", {})["enabled"] = visible_device_count > 1
gpu.setdefault("backend", "nccl")

log_cfg = cfg.setdefault("log", {})
out_dir = env("OUT_DIR")
if out_dir:
    log_cfg["out_dir"] = out_dir
tb_dir = env("TB_DIR")
if tb_dir:
    log_cfg["tensorboard_dir"] = tb_dir
notify = log_cfg.setdefault("notify", {})
notify["enabled"] = env_bool("FEISHU_NOTIFY")
notify["webhook_url"] = os.environ.get("FEISHU_WEBHOOK_URL", "")

validation_cfg = cfg.setdefault("validation", {})
if f"{prefix}_VALIDATION_ENABLED" in os.environ:
    validation_cfg["enabled"] = env_bool("VALIDATION_ENABLED", "true")

stages = cfg.setdefault("training", {}).setdefault("stages", [])
if not stages:
    raise RuntimeError("training.stages is empty in the base config")
epochs = env("EPOCHS")
if epochs is not None and str(epochs).strip() != "":
    stages[0]["epochs"] = int(epochs)

if prefix == "STAGE2":
    nextdit = (
        cfg.setdefault("model", {})
        .setdefault("action_head", {})
        .setdefault("nextdit", {})
    )
    internnav_model = os.environ.get("STAGE2_INTERNNAV_MODEL")
    if internnav_model:
        nextdit["internnav_model_path"] = internnav_model
        nextdit["internnav_system1_path"] = ""
    system1 = os.environ.get("STAGE2_SYSTEM1_CKPT")
    if system1:
        nextdit["internnav_system1_path"] = system1
        nextdit["internnav_model_path"] = ""
    dav2 = os.environ.get("STAGE2_EFFECTIVE_DAV2_CKPT")
    if dav2 is not None:
        nextdit["dav2_ckpt_path"] = dav2

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

log_config_summary() {
  local label="$1"
  local config_path="$2"

  python - "$label" "$config_path" "$NPROC_PER_NODE" <<'PY' | while IFS= read -r line; do
import os
import sys

import yaml

label, config_path, world_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

optim = cfg.get("optim", {}) or {}
data = cfg.get("data", {}) or {}
trajectory = data.get("trajectory", {}) or {}
model = cfg.get("model", {}) or {}
llm = (model.get("llm", {}) or {})
heatmap = (model.get("heatmap", {}) or {})
action_head = (model.get("action_head", {}) or {})
nextdit = (action_head.get("nextdit", {}) or {})
stages = ((cfg.get("training", {}) or {}).get("stages", []) or [{}])
stage = stages[0] if stages else {}

batch_size = int(optim.get("batch_size", 0) or 0)
grad_accum = int(optim.get("grad_accum_steps", 1) or 1)
global_batch = batch_size * world_size * grad_accum

print(f"{label} effective config:")
print(
    "  batch_size=%s, grad_accum_steps=%s, world_size=%s, effective_global_batch=%s"
    % (batch_size, grad_accum, world_size, global_batch)
)
print(
    "  epochs=%s, llm.gradient_checkpointing=%s, attn=%s, require_flash_attn=%s"
    % (
        stage.get("epochs"),
        llm.get("gradient_checkpointing"),
        llm.get("attn_implementation"),
        os.environ.get("HEATMAPVLN_REQUIRE_FLASH_ATTN", ""),
    )
)
print(
    "  num_workers=%s, prefetch_factor=%s, pin_memory=%s"
    % (
        data.get("num_workers"),
        data.get("prefetch_factor"),
        data.get("pin_memory"),
    )
)
print(
    "  trajectory.num_history_sample=%s, panoramic_vlm_input=%s, "
    "load_lookdown_for_system2=%s, max_clips=%s"
    % (
        trajectory.get("num_history_sample"),
        trajectory.get("panoramic_vlm_input"),
        trajectory.get("load_lookdown_for_system2"),
        trajectory.get("max_clips", 0),
    )
)
print(
    "  trajectory.random_subsequence=%s, enable_trajectory_augmentation=%s"
    % (
        trajectory.get("random_subsequence"),
        trajectory.get("enable_trajectory_augmentation"),
    )
)
print(
    "  heatmap.enable=%s, action_head.enable=%s, nextdit.enabled=%s"
    % (
        heatmap.get("enable"),
        action_head.get("enable"),
        nextdit.get("enabled"),
    )
)
PY
    log "$line"
  done
}

run_training_stage() {
  local name="$1"
  local master_port="$2"
  local config_path="$3"
  local load_weights="$4"
  local max_batches="$5"

  local train_args=(--config "$config_path" --load-weights "$load_weights")
  if [[ -n "$max_batches" ]]; then
    train_args+=(--max-batches "$max_batches")
  fi

  log "Starting ${name}"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
    --master_port="$master_port" \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train.py "${train_args[@]}"
  log "Finished ${name}"
}
