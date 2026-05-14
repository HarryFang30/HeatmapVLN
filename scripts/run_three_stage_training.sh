#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# -----------------------------
# Configurable runtime settings
# -----------------------------

INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-/workspace/InternNav_Model}"
export INTERNNAV_BACKBONE
# Cluster usage:
#   export INTERNNAV_BACKBONE=/path/to/full/InternNav_Model

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-120}"
PIPELINE_DRY_RUN="${PIPELINE_DRY_RUN:-0}"
KEEP_TMP_CONFIGS="${KEEP_TMP_CONFIGS:-$PIPELINE_DRY_RUN}"

STAGE1_HM_CONFIG="${STAGE1_HM_CONFIG:-configs/train_heatmap_config_lora.yaml}"
STAGE1_S2_CONFIG="${STAGE1_S2_CONFIG:-configs/train_system2_panoramic_sft_2gpu.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/train_config_internnav_4gpu.yaml}"

# Data roots.  PANORAMIC_DATA_ROOT also honors DATA_ROOT for compatibility
# with the existing single-stage cluster scripts.
PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-${DATA_ROOT:-/workspace/r2r_panoramic_data}}"
HEATMAP_DATA_ROOT="${HEATMAP_DATA_ROOT:-/workspace/heatmap_train_data}"
HEATMAP_VAL_ROOT="${HEATMAP_VAL_ROOT:-/workspace/val_unseen}"

STAGE1_HM_DATA_ROOT="${STAGE1_HM_DATA_ROOT:-$HEATMAP_DATA_ROOT}"
STAGE1_HM_VAL_ROOT="${STAGE1_HM_VAL_ROOT:-$HEATMAP_VAL_ROOT}"
STAGE1_S2_DATA_ROOT="${STAGE1_S2_DATA_ROOT:-$PANORAMIC_DATA_ROOT}"
STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-$PANORAMIC_DATA_ROOT}"

# Checkpoints and pretrained assets.
STAGE1_HM_INIT_CKPT="${STAGE1_HM_INIT_CKPT:-${HEATMAP_BASE_CKPT:-/workspace/heatmap_training_outputs/run_20260407_004635/checkpoints/best.pth}}"
STAGE2_INTERNNAV_MODEL="${STAGE2_INTERNNAV_MODEL:-$INTERNNAV_BACKBONE}"
STAGE2_SYSTEM1_CKPT="${STAGE2_SYSTEM1_CKPT:-}"
STAGE2_DAV2_CKPT="${STAGE2_DAV2_CKPT:-}"
STAGE2_REQUIRE_DAV2_CKPT="${STAGE2_REQUIRE_DAV2_CKPT:-0}"

# Output and TensorBoard directories.  Old variable names are still honored.
STAGE1_HM_OUT_DIR="${STAGE1_HM_OUT_DIR:-${HEATMAP_LORA_OUT_DIR:-/workspace/heatmap_lora_training_outputs}}"
STAGE1_HM_TB_DIR="${STAGE1_HM_TB_DIR:-/workspace/tf-logs-lora}"
STAGE1_S2_OUT_DIR="${STAGE1_S2_OUT_DIR:-${SYSTEM2_SFT_OUT_DIR:-/root/autodl-tmp/vln_system2_sft_outputs}}"
STAGE1_S2_TB_DIR="${STAGE1_S2_TB_DIR:-/root/tf-logs-system2-sft}"
STAGE2_OUT_DIR="${STAGE2_OUT_DIR:-/root/autodl-tmp/vln_training_outputs}"
STAGE2_TB_DIR="${STAGE2_TB_DIR:-/root/tf-logs-stage2}"

# Per-stage training knobs.  Leave *_EPOCHS empty to use the value from YAML.
STAGE1_HM_EPOCHS="${STAGE1_HM_EPOCHS:-}"
STAGE1_S2_EPOCHS="${STAGE1_S2_EPOCHS:-}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-}"

STAGE1_HM_BATCH_SIZE="${STAGE1_HM_BATCH_SIZE:-2}"
STAGE1_HM_GRAD_ACCUM_STEPS="${STAGE1_HM_GRAD_ACCUM_STEPS:-1}"
STAGE1_HM_NUM_WORKERS="${STAGE1_HM_NUM_WORKERS:-4}"
STAGE1_HM_PREFETCH_FACTOR="${STAGE1_HM_PREFETCH_FACTOR:-2}"
STAGE1_HM_PIN_MEMORY="${STAGE1_HM_PIN_MEMORY:-true}"

STAGE1_S2_BATCH_SIZE="${STAGE1_S2_BATCH_SIZE:-1}"
STAGE1_S2_GRAD_ACCUM_STEPS="${STAGE1_S2_GRAD_ACCUM_STEPS:-2}"
STAGE1_S2_NUM_WORKERS="${STAGE1_S2_NUM_WORKERS:-2}"
STAGE1_S2_PREFETCH_FACTOR="${STAGE1_S2_PREFETCH_FACTOR:-2}"
STAGE1_S2_PIN_MEMORY="${STAGE1_S2_PIN_MEMORY:-true}"

STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-1}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_PREFETCH_FACTOR="${STAGE2_PREFETCH_FACTOR:-2}"
STAGE2_PIN_MEMORY="${STAGE2_PIN_MEMORY:-true}"

STAGE1_HM_MAX_BATCHES="${STAGE1_HM_MAX_BATCHES:-}"
STAGE1_S2_MAX_BATCHES="${STAGE1_S2_MAX_BATCHES:-}"
STAGE2_MAX_BATCHES="${STAGE2_MAX_BATCHES:-}"

STAGE1_HM_CHECKPOINT_PREFERENCE="${STAGE1_HM_CHECKPOINT_PREFERENCE:-best}"
STAGE1_S2_CHECKPOINT_PREFERENCE="${STAGE1_S2_CHECKPOINT_PREFERENCE:-latest}"
STAGE2_CHECKPOINT_PREFERENCE="${STAGE2_CHECKPOINT_PREFERENCE:-latest}"

MASTER_PORT_STAGE1_HM="${MASTER_PORT_STAGE1_HM:-29616}"
MASTER_PORT_STAGE1_S2="${MASTER_PORT_STAGE1_S2:-29617}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29618}"

# Notifications are off by default so missing FEISHU_WEBHOOK_URL does not block
# dry runs or cluster smoke tests.  Enable globally with FEISHU_NOTIFY=1 or per
# stage with STAGE*_FEISHU_NOTIFY=1.
FEISHU_NOTIFY="${FEISHU_NOTIFY:-0}"
STAGE1_HM_FEISHU_NOTIFY="${STAGE1_HM_FEISHU_NOTIFY:-$FEISHU_NOTIFY}"
STAGE1_S2_FEISHU_NOTIFY="${STAGE1_S2_FEISHU_NOTIFY:-$FEISHU_NOTIFY}"
STAGE2_FEISHU_NOTIFY="${STAGE2_FEISHU_NOTIFY:-$FEISHU_NOTIFY}"
FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:-}"
export FEISHU_WEBHOOK_URL

# Evaluation is opt-in; Habitat/Xvfb failures should not invalidate a completed
# three-stage training run unless explicitly requested.
RUN_EVAL="${RUN_EVAL:-0}"
DISPLAY_NUM="${DISPLAY_NUM:-200}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
EVAL_SIM_GPU_ID="${EVAL_SIM_GPU_ID:-0}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${STAGE2_OUT_DIR}/latest/eval_r2r_val_unseen}"
EVAL_SCENES_DIR="${EVAL_SCENES_DIR:-}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
EVAL_MAX_EPISODES="${EVAL_MAX_EPISODES:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Single-node A800 runs do not need InfiniBand.  Keep this configurable because
# multi-node jobs should normally leave NCCL_IB_DISABLE=0 and set NCCL_SOCKET_IFNAME.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

TMP_CONFIGS=()

cleanup() {
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
trap cleanup EXIT

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

cooldown() {
  local seconds="$1"
  if (( seconds > 0 )); then
    log "Cooling down for ${seconds}s ..."
    sleep "$seconds"
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

optim = cfg.setdefault("optim", {})
set_int(optim, "batch_size", env("BATCH_SIZE"))
set_int(optim, "grad_accum_steps", env("GRAD_ACCUM_STEPS"))

gpu = cfg.setdefault("gpu", {})
visible_device_count = len([
    token for token in os.environ["GPU_DEVICES"].split(",")
    if token.strip()
])
gpu["devices"] = list(range(visible_device_count))
gpu.setdefault("multi_gpu", {})["enabled"] = True
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

start_xvfb() {
  if ! command -v Xvfb >/dev/null 2>&1; then
    log "Xvfb not found; evaluation cannot start."
    exit 1
  fi

  if pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    log "Xvfb :${DISPLAY_NUM} is already running"
  else
    log "Starting Xvfb :${DISPLAY_NUM}"
    Xvfb ":${DISPLAY_NUM}" -screen 0 1024x768x24 >"/tmp/xvfb_${DISPLAY_NUM}.log" 2>&1 &
    sleep 2
  fi

  export DISPLAY=":${DISPLAY_NUM}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
}

run_eval() {
  local stage1_s2_ckpt="$1"
  local stage2_ckpt="$2"
  start_xvfb

  local eval_args=(
    r2r
    --config "$STAGE2_TMP_CONFIG"
    --base_checkpoint "$stage1_s2_ckpt"
    --checkpoint "$stage2_ckpt"
    --gpu_id "$EVAL_GPU_ID"
    --sim_gpu_id "$EVAL_SIM_GPU_ID"
    --output_path "$EVAL_OUTPUT_PATH"
  )
  if [[ -n "$EVAL_SCENES_DIR" ]]; then
    eval_args+=(--scenes_dir "$EVAL_SCENES_DIR")
  fi
  if [[ -n "$EVAL_DATA_PATH" ]]; then
    eval_args+=(--data_path "$EVAL_DATA_PATH")
  fi
  if [[ -n "$EVAL_MAX_EPISODES" ]]; then
    eval_args+=(--max_episodes "$EVAL_MAX_EPISODES")
  fi

  log "Starting R2R val_unseen evaluation"
  CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python scripts/evaluate.py "${eval_args[@]}"
  log "Finished R2R val_unseen evaluation"
}

preflight() {
  local gpu_count
  gpu_count="$(count_csv_items "$GPU_DEVICES")"
  if [[ "$gpu_count" != "$NPROC_PER_NODE" ]]; then
    log "GPU_DEVICES has ${gpu_count} entries but NPROC_PER_NODE=${NPROC_PER_NODE}."
    log "Set both consistently, e.g. GPU_DEVICES=0,1,2,3 NPROC_PER_NODE=4."
    exit 1
  fi

  require_file "$STAGE1_HM_CONFIG"
  require_file "$STAGE1_S2_CONFIG"
  require_file "$STAGE2_CONFIG"
  require_file "$STAGE1_HM_INIT_CKPT"
  require_hf_model_dir "$INTERNNAV_BACKBONE"
  require_dir "$STAGE1_HM_DATA_ROOT"
  require_dir "$STAGE1_HM_VAL_ROOT"
  require_dir "$STAGE1_S2_DATA_ROOT"
  require_dir "$STAGE2_DATA_ROOT"
  require_file "$ROOT_DIR/data/fgr2r/subinstr_mapping.json.gz"

  if [[ -n "$STAGE2_SYSTEM1_CKPT" ]]; then
    require_file "$STAGE2_SYSTEM1_CKPT"
  else
    require_hf_model_dir "$STAGE2_INTERNNAV_MODEL"
  fi

  STAGE2_EFFECTIVE_DAV2_CKPT=""
  if [[ -n "$STAGE2_DAV2_CKPT" && -f "$STAGE2_DAV2_CKPT" ]]; then
    STAGE2_EFFECTIVE_DAV2_CKPT="$STAGE2_DAV2_CKPT"
    export STAGE2_EFFECTIVE_DAV2_CKPT
  elif is_truthy "$STAGE2_REQUIRE_DAV2_CKPT"; then
    require_file "$STAGE2_DAV2_CKPT"
  else
    log "DepthAnythingV2 checkpoint not configured; Stage2 will rely on InternNav full-model System1 weights"
    STAGE2_EFFECTIVE_DAV2_CKPT=""
    export STAGE2_EFFECTIVE_DAV2_CKPT
  fi

  if is_truthy "$STAGE1_HM_FEISHU_NOTIFY" || is_truthy "$STAGE1_S2_FEISHU_NOTIFY" || is_truthy "$STAGE2_FEISHU_NOTIFY"; then
    if [[ -z "$FEISHU_WEBHOOK_URL" ]]; then
      log "FEISHU notification is enabled but FEISHU_WEBHOOK_URL is empty."
      exit 1
    fi
  fi

  if is_truthy "$RUN_EVAL" && ! command -v Xvfb >/dev/null 2>&1; then
    log "RUN_EVAL=$RUN_EVAL but Xvfb is not installed."
    exit 1
  fi

  mkdir -p \
    "$STAGE1_HM_OUT_DIR" "$STAGE1_HM_TB_DIR" \
    "$STAGE1_S2_OUT_DIR" "$STAGE1_S2_TB_DIR" \
    "$STAGE2_OUT_DIR" "$STAGE2_TB_DIR"
}

preflight

STAGE1_HM_TMP_CONFIG="$(mktemp "/tmp/stage1_hm.XXXXXX")"
STAGE1_S2_TMP_CONFIG="$(mktemp "/tmp/stage1_s2.XXXXXX")"
STAGE2_TMP_CONFIG="$(mktemp "/tmp/stage2.XXXXXX")"
TMP_CONFIGS+=("$STAGE1_HM_TMP_CONFIG" "$STAGE1_S2_TMP_CONFIG" "$STAGE2_TMP_CONFIG")

export GPU_DEVICES

export STAGE1_HM_DATA_ROOT STAGE1_HM_VAL_ROOT STAGE1_HM_OUT_DIR STAGE1_HM_TB_DIR
export STAGE1_HM_EPOCHS STAGE1_HM_BATCH_SIZE STAGE1_HM_GRAD_ACCUM_STEPS
export STAGE1_HM_NUM_WORKERS STAGE1_HM_PREFETCH_FACTOR STAGE1_HM_PIN_MEMORY
export STAGE1_HM_FEISHU_NOTIFY

export STAGE1_S2_DATA_ROOT STAGE1_S2_OUT_DIR STAGE1_S2_TB_DIR
export STAGE1_S2_EPOCHS STAGE1_S2_BATCH_SIZE STAGE1_S2_GRAD_ACCUM_STEPS
export STAGE1_S2_NUM_WORKERS STAGE1_S2_PREFETCH_FACTOR STAGE1_S2_PIN_MEMORY
export STAGE1_S2_FEISHU_NOTIFY

export STAGE2_DATA_ROOT STAGE2_OUT_DIR STAGE2_TB_DIR
export STAGE2_EPOCHS STAGE2_BATCH_SIZE STAGE2_GRAD_ACCUM_STEPS
export STAGE2_NUM_WORKERS STAGE2_PREFETCH_FACTOR STAGE2_PIN_MEMORY
export STAGE2_FEISHU_NOTIFY STAGE2_INTERNNAV_MODEL STAGE2_SYSTEM1_CKPT

make_stage_config STAGE1_HM "$STAGE1_HM_CONFIG" "$STAGE1_HM_TMP_CONFIG"
make_stage_config STAGE1_S2 "$STAGE1_S2_CONFIG" "$STAGE1_S2_TMP_CONFIG"
make_stage_config STAGE2 "$STAGE2_CONFIG" "$STAGE2_TMP_CONFIG"

log "Pipeline root: $ROOT_DIR"
log "Training GPUs: $GPU_DEVICES (nproc_per_node=$NPROC_PER_NODE)"
log "InternNav backbone: $INTERNNAV_BACKBONE"
log "Stage1-HM config: $STAGE1_HM_TMP_CONFIG"
log "Stage1-S2 config: $STAGE1_S2_TMP_CONFIG"
log "Stage2 config: $STAGE2_TMP_CONFIG"
log "Cooldown between steps: ${COOLDOWN_SECONDS}s"

if is_truthy "$PIPELINE_DRY_RUN"; then
  log "PIPELINE_DRY_RUN=$PIPELINE_DRY_RUN; preflight and config generation completed, skipping training"
  exit 0
fi

run_training_stage "Stage1-HM heatmap LoRA" "$MASTER_PORT_STAGE1_HM" \
  "$STAGE1_HM_TMP_CONFIG" "$STAGE1_HM_INIT_CKPT" "$STAGE1_HM_MAX_BATCHES"

cooldown "$COOLDOWN_SECONDS"

STAGE1_HM_CKPT="$(choose_checkpoint "${STAGE1_HM_OUT_DIR}/latest/checkpoints" "$STAGE1_HM_CHECKPOINT_PREFERENCE")"
log "Stage1-HM checkpoint: $STAGE1_HM_CKPT"

run_training_stage "Stage1-S2 panoramic System2 SFT" "$MASTER_PORT_STAGE1_S2" \
  "$STAGE1_S2_TMP_CONFIG" "$STAGE1_HM_CKPT" "$STAGE1_S2_MAX_BATCHES"

cooldown "$COOLDOWN_SECONDS"

STAGE1_S2_CKPT="$(choose_checkpoint "${STAGE1_S2_OUT_DIR}/latest/checkpoints" "$STAGE1_S2_CHECKPOINT_PREFERENCE")"
log "Stage1-S2 checkpoint: $STAGE1_S2_CKPT"

run_training_stage "Stage2 bridge-only" "$MASTER_PORT_STAGE2" \
  "$STAGE2_TMP_CONFIG" "$STAGE1_S2_CKPT" "$STAGE2_MAX_BATCHES"

STAGE2_CKPT="$(choose_checkpoint "${STAGE2_OUT_DIR}/latest/checkpoints" "$STAGE2_CHECKPOINT_PREFERENCE")"
log "Stage2 checkpoint: $STAGE2_CKPT"

if is_truthy "$RUN_EVAL"; then
  cooldown "$COOLDOWN_SECONDS"
  run_eval "$STAGE1_S2_CKPT" "$STAGE2_CKPT"
else
  log "RUN_EVAL=$RUN_EVAL; skipping evaluation"
fi

log "Pipeline completed"
