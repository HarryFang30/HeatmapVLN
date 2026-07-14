#!/usr/bin/env bash
# HeatmapVLN Stage3 pano-System1 action fine-tuning（8×沐曦 C500）
#
# Usage:
#   bash scripts/run_stage3_pano_system1_h1024_8gpu_mxc500_launcher.sh
#
# This stage does not collect teacher sidecars.  It starts from:
#   1. Stage1-S2 panoramic Qwen LoRA checkpoint (--load-weights)
#   2. Stage2 h1024 pano latent adapter checkpoint
# then trains only the pano latent adapter with GT trajectory flow-matching
# loss, keeping InternNav System1 frozen by default.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# MXC500 / distributed runtime
# ---------------------------------------------------------------------------
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29620}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ "${WORLD_SIZE}" == "1" && "${RANK}" == "0" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK}"

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
QWEN25_ENV="/mnt/afs/lixiaoou/intern/fjl/envs/qwen25"

activate_qwen25_via_path() {
  if [[ ! -x "${QWEN25_ENV}/bin/python" ]]; then
    return 1
  fi
  export PATH="${QWEN25_ENV}/bin:${PATH}"
  export CONDA_PREFIX="${QWEN25_ENV}"
  export CONDA_DEFAULT_ENV="qwen25"
  hash -r
  echo "[launcher] Activated by PATH: ${QWEN25_ENV} (python=$(command -v python))"
  return 0
}

_CONDA_SH=""
if [[ -n "${CONDA_INIT_SH:-}" && -f "${CONDA_INIT_SH}" ]]; then
  _CONDA_SH="${CONDA_INIT_SH}"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [[ -f "/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if [[ -n "${_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${_CONDA_SH}"
  conda activate "${QWEN25_ENV}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${QWEN25_ENV}"
elif activate_qwen25_via_path; then
  :
else
  echo "Cannot activate qwen25 env at ${QWEN25_ENV}." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Cluster paths and tunables
# ---------------------------------------------------------------------------
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data}"

export STAGE3_CONFIG="${STAGE3_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_BASE_CKPT="${STAGE3_BASE_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_ADAPTER_CKPT="${STAGE3_ADAPTER_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter_full_11000_alllora_h1024/latest.pth}"
export STAGE3_OUT_DIR="${STAGE3_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords}"
export STAGE3_TB_DIR="${STAGE3_TB_DIR:-/mnt/afs/lixiaoou/intern/fjl/tensorlog/heatmapvln_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords}"
export STAGE_TMP_DIR="${STAGE_TMP_DIR:-/mnt/afs/lixiaoou/intern/fjl/tmp}"

export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT_STAGE3="${MASTER_PORT_STAGE3:-$MASTER_PORT}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-}"
export STAGE3_GRAD_ACCUM_STEPS="${STAGE3_GRAD_ACCUM_STEPS:-}"
export STAGE3_PANO_ADAPTER_LR="${STAGE3_PANO_ADAPTER_LR:-}"
export STAGE3_L2_SP_ENABLED="${STAGE3_L2_SP_ENABLED:-}"
export STAGE3_L2_SP_WEIGHT="${STAGE3_L2_SP_WEIGHT:-}"
export STAGE3_L2_SP_NORMALIZATION="${STAGE3_L2_SP_NORMALIZATION:-}"
export STAGE3_TRAJECTORY_SEQUENCE_MODE="${STAGE3_TRAJECTORY_SEQUENCE_MODE:-}"
export STAGE3_VIEW_WEIGHT_FRONT="${STAGE3_VIEW_WEIGHT_FRONT:-}"
export STAGE3_VIEW_WEIGHT_RIGHT="${STAGE3_VIEW_WEIGHT_RIGHT:-}"
export STAGE3_VIEW_WEIGHT_BACK="${STAGE3_VIEW_WEIGHT_BACK:-}"
export STAGE3_VIEW_WEIGHT_LEFT="${STAGE3_VIEW_WEIGHT_LEFT:-}"
export STAGE3_NUM_WORKERS="${STAGE3_NUM_WORKERS:-16}"
export STAGE3_PREFETCH_FACTOR="${STAGE3_PREFETCH_FACTOR:-4}"
export STAGE3_PIN_MEMORY="${STAGE3_PIN_MEMORY:-1}"
export STAGE3_SHM_BYPASS="${STAGE3_SHM_BYPASS:-auto}"
export STAGE3_SHM_BYPASS_MIN_GB="${STAGE3_SHM_BYPASS_MIN_GB:-8.0}"
export STAGE3_ENABLE_TIMING="${STAGE3_ENABLE_TIMING:-1}"
export STAGE3_SHOW_GPU_MEMORY="${STAGE3_SHOW_GPU_MEMORY:-0}"
export STAGE3_LOG_INTERVAL="${STAGE3_LOG_INTERVAL:-20}"
export STAGE3_TENSORBOARD_INTERVAL="${STAGE3_TENSORBOARD_INTERVAL:-20}"
export STAGE3_PAGE_CACHE_DROP_ENABLED="${STAGE3_PAGE_CACHE_DROP_ENABLED:-0}"
export STAGE3_SYSTEM2_SAMPLE_STEP="${STAGE3_SYSTEM2_SAMPLE_STEP:-1}"
export STAGE3_MAX_CLIPS="${STAGE3_MAX_CLIPS:-}"
export STAGE3_MAX_BATCHES="${STAGE3_MAX_BATCHES:-}"
export STAGE3_DRY_RUN="${STAGE3_DRY_RUN:-${STAGE_DRY_RUN:-0}}"
export STAGE3_MERGE_FROZEN_LORA="${STAGE3_MERGE_FROZEN_LORA:-0}"
export STAGE3_FROZEN_TRAJ_INFERENCE_MODE="${STAGE3_FROZEN_TRAJ_INFERENCE_MODE:-1}"
export STAGE3_TRAJ_LAST_HIDDEN_STATE_ONLY="${STAGE3_TRAJ_LAST_HIDDEN_STATE_ONLY:-0}"
export STAGE3_REQUIRE_FLASH_ATTN="${STAGE3_REQUIRE_FLASH_ATTN:-1}"
export HEATMAPVLN_REQUIRE_FLASH_ATTN="$STAGE3_REQUIRE_FLASH_ATTN"

mkdir -p "$REPO_ROOT/logs" "$STAGE3_OUT_DIR" "$STAGE3_TB_DIR" "$STAGE_TMP_DIR"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/stage3_pano_system1_h1024_8gpu_mxc500.log}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

count_csv_items() {
  python - "$1" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]))
PY
}

gpu_count="$(count_csv_items "$GPU_DEVICES")"
if [[ "$gpu_count" != "$NPROC_PER_NODE" ]]; then
  echo "GPU_DEVICES has ${gpu_count} entries but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

require_file "$STAGE3_CONFIG"
require_file "$STAGE3_BASE_CKPT"
require_file "$STAGE3_ADAPTER_CKPT"
require_dir "$PANORAMIC_DATA_ROOT"
require_dir "$INTERNNAV_MODEL_PATH"
TMP_CONFIG="$(mktemp "${STAGE_TMP_DIR%/}/stage3_pano_system1.XXXXXX.yaml")"
cleanup() {
  if [[ "${KEEP_TMP_CONFIGS:-0}" != "1" ]]; then
    rm -f "$TMP_CONFIG"
  else
    echo "[launcher] Keeping temp config: $TMP_CONFIG"
  fi
}
trap cleanup EXIT

python - "$STAGE3_CONFIG" "$TMP_CONFIG" <<'PY'
import os
import sys

import yaml

base_config, output_config = sys.argv[1], sys.argv[2]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def set_int(section, key, env_name):
    value = os.environ.get(env_name)
    if value is not None and str(value).strip() != "":
        section[key] = int(value)

def set_float(section, key, env_name):
    value = os.environ.get(env_name)
    if value is not None and str(value).strip() != "":
        section[key] = float(value)

def set_bool(section, key, env_name):
    value = os.environ.get(env_name)
    if value is None or str(value).strip() == "":
        return
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        section[key] = True
    elif normalized in {"0", "false", "no", "n", "off"}:
        section[key] = False
    else:
        raise RuntimeError(f"{env_name} must be boolean-like, got {value!r}")

def set_str(section, key, env_name):
    value = os.environ.get(env_name)
    if value is not None and str(value).strip() != "":
        section[key] = str(value).strip()

paths = cfg.setdefault("paths", {})
paths["dataset_root"] = os.environ["PANORAMIC_DATA_ROOT"]
paths["log_out_dir"] = os.environ["STAGE3_OUT_DIR"]
paths["tensorboard_dir"] = os.environ["STAGE3_TB_DIR"]
paths["internnav_model_path"] = os.environ["INTERNNAV_MODEL_PATH"]

data = cfg.setdefault("data", {})
data["root"] = os.environ["PANORAMIC_DATA_ROOT"]
set_int(data, "num_workers", "STAGE3_NUM_WORKERS")
set_int(data, "prefetch_factor", "STAGE3_PREFETCH_FACTOR")
set_bool(data, "pin_memory", "STAGE3_PIN_MEMORY")
if os.environ.get("STAGE3_SHM_BYPASS", "").strip():
    data["shm_bypass"] = os.environ["STAGE3_SHM_BYPASS"].strip()
set_float(data, "shm_bypass_min_gb", "STAGE3_SHM_BYPASS_MIN_GB")
trajectory = data.setdefault("trajectory", {})
set_int(trajectory, "system2_sample_step", "STAGE3_SYSTEM2_SAMPLE_STEP")
set_int(trajectory, "max_clips", "STAGE3_MAX_CLIPS")
if trajectory.get("trajectory_target_convention") != "internnav_habitat":
    raise RuntimeError(
        "Stage3 requires data.trajectory.trajectory_target_convention="
        "internnav_habitat so GT actions match frozen InternNav System1"
    )

model = cfg.setdefault("model", {})
llm = model.setdefault("llm", {})
llm["model_path"] = os.environ["INTERNNAV_MODEL_PATH"]
set_bool(llm, "frozen_traj_inference_mode", "STAGE3_FROZEN_TRAJ_INFERENCE_MODE")
set_bool(llm, "traj_last_hidden_state_only", "STAGE3_TRAJ_LAST_HIDDEN_STATE_ONLY")
nextdit = model.setdefault("action_head", {}).setdefault("nextdit", {})
nextdit["internnav_model_path"] = os.environ["INTERNNAV_MODEL_PATH"]
adapter = nextdit.setdefault("pano_latent_adapter", {})
adapter["pretrained_path"] = os.environ["STAGE3_ADAPTER_CKPT"]

optim = cfg.setdefault("optim", {})
set_int(optim, "batch_size", "STAGE3_BATCH_SIZE")
set_int(optim, "grad_accum_steps", "STAGE3_GRAD_ACCUM_STEPS")
set_float(optim, "pano_latent_adapter_lr", "STAGE3_PANO_ADAPTER_LR")

loss = cfg.setdefault("loss", {})
l2_sp = loss.setdefault("l2_sp", {})
set_bool(l2_sp, "enabled", "STAGE3_L2_SP_ENABLED")
set_float(l2_sp, "weight", "STAGE3_L2_SP_WEIGHT")
set_str(l2_sp, "normalization", "STAGE3_L2_SP_NORMALIZATION")
view_weights_cfg = loss.setdefault("trajectory_view_weights", {})
view_weights = view_weights_cfg.setdefault("weights", {})
set_float(view_weights, "front", "STAGE3_VIEW_WEIGHT_FRONT")
set_float(view_weights, "right", "STAGE3_VIEW_WEIGHT_RIGHT")
set_float(view_weights, "back", "STAGE3_VIEW_WEIGHT_BACK")
set_float(view_weights, "left", "STAGE3_VIEW_WEIGHT_LEFT")

stages = cfg.setdefault("training", {}).setdefault("stages", [])
if not stages:
    raise RuntimeError("training.stages is empty")
set_bool(stages[0], "merge_frozen_lora", "STAGE3_MERGE_FROZEN_LORA")
set_str(stages[0], "trajectory_sequence_mode", "STAGE3_TRAJECTORY_SEQUENCE_MODE")
epochs = os.environ.get("STAGE3_EPOCHS")
if epochs is not None and epochs.strip():
    stages[0]["epochs"] = int(epochs)

sequence_mode = str(stages[0].get("trajectory_sequence_mode", "all"))
if sequence_mode not in {"all", "first_only"}:
    raise RuntimeError(
        "Stage3 trajectory_sequence_mode must be all or first_only, got "
        f"{sequence_mode!r}"
    )
if bool(l2_sp.get("enabled", False)) and float(l2_sp.get("weight", 0.0) or 0.0) > 0.0:
    if "pano_latent_adapter" not in set(l2_sp.get("modules") or []):
        raise RuntimeError(
            "Stage3 L2-SP must include pano_latent_adapter; otherwise the "
            "adapter-only training has no regularized parameter"
        )
    if str(l2_sp.get("normalization", "mean_parameter_mse")) not in {
        "mean_parameter_mse",
        "relative_l2",
    }:
        raise RuntimeError("Unsupported Stage3 L2-SP normalization")

log = cfg.setdefault("log", {})
set_bool(log, "enable_timing", "STAGE3_ENABLE_TIMING")
set_bool(log, "show_gpu_memory", "STAGE3_SHOW_GPU_MEMORY")
set_bool(log, "page_cache_drop_enabled", "STAGE3_PAGE_CACHE_DROP_ENABLED")
set_int(log, "log_interval", "STAGE3_LOG_INTERVAL")
set_int(log, "tensorboard_interval", "STAGE3_TENSORBOARD_INTERVAL")

visible_device_count = len([x for x in os.environ["GPU_DEVICES"].split(",") if x.strip()])
gpu = cfg.setdefault("gpu", {})
gpu["devices"] = list(range(visible_device_count))
gpu.setdefault("multi_gpu", {})["enabled"] = visible_device_count > 1

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

echo "[launcher] Stage3 pano-System1 action fine-tuning"
echo "[launcher] repo=$REPO_ROOT"
echo "[launcher] config=$TMP_CONFIG"
echo "[launcher] data=$PANORAMIC_DATA_ROOT"
echo "[launcher] base=$STAGE3_BASE_CKPT"
echo "[launcher] adapter=$STAGE3_ADAPTER_CKPT"
echo "[launcher] out=$STAGE3_OUT_DIR"
echo "[launcher] trajectory_target_convention=internnav_habitat (x=forward, y=left, yaw=left-positive)"
effective_sequence_mode="$(python - "$TMP_CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["training"]["stages"][0].get("trajectory_sequence_mode", "all"))
PY
)"
echo "[launcher] trajectory_sequence_mode=$effective_sequence_mode"
echo "[launcher] gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE"

train_args=(--config "$TMP_CONFIG" --load-weights "$STAGE3_BASE_CKPT")
if [[ -n "$STAGE3_MAX_BATCHES" ]]; then
  train_args+=(--max-batches "$STAGE3_MAX_BATCHES")
fi
case "${STAGE3_DRY_RUN,,}" in
  1|true|yes|y|on)
    train_args+=(--dry-run)
    echo "[launcher] real dry-run preflight enabled: one full forward/backward/DDP/optimizer batch, no checkpoint"
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --master_port="$MASTER_PORT_STAGE3" \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py "${train_args[@]}" \
  2>&1 | tee "$LOG_FILE"

echo "[launcher] Stage3 checkpoints: ${STAGE3_OUT_DIR}/latest/checkpoints"
