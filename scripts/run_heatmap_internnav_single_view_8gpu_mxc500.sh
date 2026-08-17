#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
QWEN25_ENV="/mnt/afs/lixiaoou/intern/fjl/envs/qwen25"
cd "$REPO_ROOT"

# MXC500 runtime.  These defaults match the launchers that have already run
# successfully in this repository and remain overridable by the cluster job.
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# A path-based activation works even when a non-interactive cluster shell does
# not expose the conda shell function.
if [[ ! -x "$QWEN25_ENV/bin/python" || ! -x "$QWEN25_ENV/bin/torchrun" ]]; then
  echo "Missing qwen25 Python/torchrun under $QWEN25_ENV" >&2
  exit 1
fi
export PATH="$QWEN25_ENV/bin:$PATH"
export CONDA_PREFIX="$QWEN25_ENV"
export CONDA_DEFAULT_ENV=qwen25
hash -r

export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_BACKBONE="$INTERNNAV_MODEL_PATH"
export HEATMAP_DATA_ROOT="${HEATMAP_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/data/heatmap_randomwalk_train_v1}"

# All artifacts for this experiment live below one root: run checkpoints,
# TensorBoard files, launcher logs, and the audited initializer.
export SINGLE_VIEW_HM_EXPERIMENT_ROOT="${SINGLE_VIEW_HM_EXPERIMENT_ROOT:-/mnt/afs/lixiaoou/intern/fjl/model/output_heatmap_internnav_single_view_v1}"
export SINGLE_VIEW_HM_OUT_DIR="${SINGLE_VIEW_HM_OUT_DIR:-$SINGLE_VIEW_HM_EXPERIMENT_ROOT/runs}"
export SINGLE_VIEW_HM_TB_DIR="${SINGLE_VIEW_HM_TB_DIR:-$SINGLE_VIEW_HM_EXPERIMENT_ROOT/tensorboard}"
export SINGLE_VIEW_HM_INIT_CKPT="${SINGLE_VIEW_HM_INIT_CKPT:-$SINGLE_VIEW_HM_EXPERIMENT_ROOT/init/from_legacy_heatmap_53tensors_v2.pth}"
LOG_DIR="$SINGLE_VIEW_HM_EXPERIMENT_ROOT/launcher_logs"

CONFIG="${SINGLE_VIEW_HM_CONFIG:-configs/train_heatmap_internnav_single_view_8gpu.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29621}"

require_file() {
  [[ -s "$1" ]] || { echo "Missing required non-empty file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

require_file "$CONFIG"
require_file "$SINGLE_VIEW_HM_INIT_CKPT"
require_dir "$HEATMAP_DATA_ROOT"
require_dir "$INTERNNAV_MODEL_PATH"

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [[ "${#GPU_LIST[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "GPU_DEVICES has ${#GPU_LIST[@]} entries but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi

mkdir -p "$SINGLE_VIEW_HM_OUT_DIR" "$SINGLE_VIEW_HM_TB_DIR" "$LOG_DIR"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${RUN_STAMP}.log}"

# Validate environment expansion and every schema-backed field before the
# expensive distributed model construction starts.
"$QWEN25_ENV/bin/python" - "$CONFIG" <<'PY'
import os
import sys
import torch

from src.config_schema import load_and_validate_config
from scripts.training.single_view_heatmap_warmstart import (
    STATE_KEY,
    file_sha256,
    validate_artifact,
)

cfg = load_and_validate_config(sys.argv[1])
assert cfg["data"]["root"] == os.environ["HEATMAP_DATA_ROOT"]
assert cfg["model"]["llm"]["model_path"] == os.environ["INTERNNAV_MODEL_PATH"]
assert cfg["model"]["llm"]["use_lora"] is False
assert cfg["model"]["action_head"]["enable"] is False
assert cfg["model"]["heatmap"]["input_mode"] == "internnav_single_view"
assert cfg["training"]["stages"][0]["trainable_modules"] == ["heatmap_vln"]
print("Validated native single-view heatmap config:", sys.argv[1])

artifact_path = os.environ["SINGLE_VIEW_HM_INIT_CKPT"]
artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
report = validate_artifact(artifact, artifact[STATE_KEY])
print(
    "Validated audited initializer before torchrun:",
    artifact_path,
    "file_sha256=" + file_sha256(artifact_path),
    "tensors=" + str(report["loaded_tensor_count"]),
)
PY

echo "[launcher] repo=$REPO_ROOT"
echo "[launcher] config=$CONFIG"
echo "[launcher] data=$HEATMAP_DATA_ROOT"
echo "[launcher] native_internnav=$INTERNNAV_MODEL_PATH"
echo "[launcher] init=$SINGLE_VIEW_HM_INIT_CKPT"
echo "[launcher] output=$SINGLE_VIEW_HM_OUT_DIR"
echo "[launcher] tensorboard=$SINGLE_VIEW_HM_TB_DIR"
echo "[launcher] log=$LOG_FILE"
echo "[launcher] gpu_devices=$GPU_DEVICES nproc=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$QWEN25_ENV/bin/torchrun" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train.py \
  --config "$CONFIG" \
  --load-weights "$SINGLE_VIEW_HM_INIT_CKPT" \
  2>&1 | tee "$LOG_FILE"
