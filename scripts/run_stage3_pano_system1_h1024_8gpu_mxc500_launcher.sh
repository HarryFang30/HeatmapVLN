#!/usr/bin/env bash
# HeatmapVLN Stage3 pano-System1 action fine-tuning（8×沐曦 C500）
#
# Usage:
#   bash scripts/run_stage3_pano_system1_h1024_8gpu_mxc500_launcher.sh
#
# This stage does not collect teacher sidecars.  It starts from:
#   1. Stage1-S2 panoramic Qwen LoRA checkpoint (--load-weights)
#   2. Stage2 h1024 pano latent adapter checkpoint
# then trains the adapter plus InternNav System1 action-side modules with GT
# trajectory flow-matching loss.

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
export STAGE3_BASE_CKPT="${STAGE3_BASE_CKPT:-/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN/checkpoints/stage1-s2_latest.pth}"
export STAGE3_ADAPTER_CKPT="${STAGE3_ADAPTER_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter_h1024/latest.pth}"
export STAGE3_OUT_DIR="${STAGE3_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage3_pano_system1_h1024}"
export STAGE3_TB_DIR="${STAGE3_TB_DIR:-/mnt/afs/tensorlog/heatmapvln_stage3_pano_system1_h1024}"

export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT_STAGE3="${MASTER_PORT_STAGE3:-$MASTER_PORT}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-}"
export STAGE3_GRAD_ACCUM_STEPS="${STAGE3_GRAD_ACCUM_STEPS:-}"
export STAGE3_NUM_WORKERS="${STAGE3_NUM_WORKERS:-8}"
export STAGE3_PREFETCH_FACTOR="${STAGE3_PREFETCH_FACTOR:-2}"
export STAGE3_SYSTEM2_SAMPLE_STEP="${STAGE3_SYSTEM2_SAMPLE_STEP:-1}"
export STAGE3_MAX_BATCHES="${STAGE3_MAX_BATCHES:-}"
export STAGE3_REQUIRE_FLASH_ATTN="${STAGE3_REQUIRE_FLASH_ATTN:-1}"
export HEATMAPVLN_REQUIRE_FLASH_ATTN="$STAGE3_REQUIRE_FLASH_ATTN"

mkdir -p "$REPO_ROOT/logs" "$STAGE3_OUT_DIR" "$STAGE3_TB_DIR"
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
TMP_CONFIG="$(mktemp "/tmp/stage3_pano_system1.XXXXXX.yaml")"
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

paths = cfg.setdefault("paths", {})
paths["dataset_root"] = os.environ["PANORAMIC_DATA_ROOT"]
paths["log_out_dir"] = os.environ["STAGE3_OUT_DIR"]
paths["tensorboard_dir"] = os.environ["STAGE3_TB_DIR"]
paths["internnav_model_path"] = os.environ["INTERNNAV_MODEL_PATH"]

data = cfg.setdefault("data", {})
data["root"] = os.environ["PANORAMIC_DATA_ROOT"]
set_int(data, "num_workers", "STAGE3_NUM_WORKERS")
set_int(data, "prefetch_factor", "STAGE3_PREFETCH_FACTOR")
trajectory = data.setdefault("trajectory", {})
set_int(trajectory, "system2_sample_step", "STAGE3_SYSTEM2_SAMPLE_STEP")

model = cfg.setdefault("model", {})
llm = model.setdefault("llm", {})
llm["model_path"] = os.environ["INTERNNAV_MODEL_PATH"]
nextdit = model.setdefault("action_head", {}).setdefault("nextdit", {})
nextdit["internnav_model_path"] = os.environ["INTERNNAV_MODEL_PATH"]
adapter = nextdit.setdefault("pano_latent_adapter", {})
adapter["pretrained_path"] = os.environ["STAGE3_ADAPTER_CKPT"]

optim = cfg.setdefault("optim", {})
set_int(optim, "batch_size", "STAGE3_BATCH_SIZE")
set_int(optim, "grad_accum_steps", "STAGE3_GRAD_ACCUM_STEPS")

stages = cfg.setdefault("training", {}).setdefault("stages", [])
if not stages:
    raise RuntimeError("training.stages is empty")
epochs = os.environ.get("STAGE3_EPOCHS")
if epochs is not None and epochs.strip():
    stages[0]["epochs"] = int(epochs)

visible_device_count = len([x for x in os.environ["GPU_DEVICES"].split(",") if x.strip()])
gpu = cfg.setdefault("gpu", {})
gpu["devices"] = list(range(visible_device_count))
gpu.setdefault("multi_gpu", {})["enabled"] = True

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
echo "[launcher] gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE"

train_args=(--config "$TMP_CONFIG" --load-weights "$STAGE3_BASE_CKPT")
if [[ -n "$STAGE3_MAX_BATCHES" ]]; then
  train_args+=(--max-batches "$STAGE3_MAX_BATCHES")
fi

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --master_port="$MASTER_PORT_STAGE3" \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py "${train_args[@]}" \
  2>&1 | tee "$LOG_FILE"

echo "[launcher] Stage3 checkpoints: ${STAGE3_OUT_DIR}/latest/checkpoints"
