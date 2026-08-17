#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
QWEN25_ENV="${ALLOWED_ROOT}/envs/qwen25"
cd "$REPO_ROOT"

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

# Keep framework/model caches within the authorized fjl subtree.
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${ALLOWED_ROOT}/model/.runtime_cache}"
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${RUNTIME_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_CACHE_ROOT}/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

if [[ ! -x "$QWEN25_ENV/bin/python" || ! -x "$QWEN25_ENV/bin/torchrun" ]]; then
  echo "Missing qwen25 Python/torchrun under $QWEN25_ENV" >&2
  exit 1
fi
export PATH="$QWEN25_ENV/bin:$PATH"
export CONDA_PREFIX="$QWEN25_ENV"
export CONDA_DEFAULT_ENV=qwen25
hash -r

export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${ALLOWED_ROOT}/InternNav-Model}"
export INTERNNAV_BACKBONE="$INTERNNAV_MODEL_PATH"
export HEATMAP_DATA_ROOT="${HEATMAP_DATA_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}"
export HEATMAP_AMB3R_POSE_CACHE_ROOT="${HEATMAP_AMB3R_POSE_CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2}"

export POSE_ADAPT_EXPERIMENT_ROOT="${POSE_ADAPT_EXPERIMENT_ROOT:-${ALLOWED_ROOT}/model/output_heatmap_amb3r_pose_adapt_endpoint_v2}"
export SINGLE_VIEW_HM_OUT_DIR="${SINGLE_VIEW_HM_OUT_DIR:-$POSE_ADAPT_EXPERIMENT_ROOT/runs}"
export SINGLE_VIEW_HM_TB_DIR="${SINGLE_VIEW_HM_TB_DIR:-$POSE_ADAPT_EXPERIMENT_ROOT/tensorboard}"
export POSE_ADAPT_INIT_CKPT="${POSE_ADAPT_INIT_CKPT:-${ALLOWED_ROOT}/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth}"
LOG_DIR="$POSE_ADAPT_EXPERIMENT_ROOT/launcher_logs"

CONFIG="${POSE_ADAPT_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_8gpu.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
EXPECTED_NUM_GPUS="${EXPECTED_NUM_GPUS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$EXPECTED_NUM_GPUS}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29641}"
MAP_INIT_WINDOW="${MAP_INIT_WINDOW:-20}"
MAP_EVERY="${MAP_EVERY:-8}"
CACHE_READY="$HEATMAP_AMB3R_POSE_CACHE_ROOT/_control/cache.ready.json"
CACHE_PLAN="$HEATMAP_AMB3R_POSE_CACHE_ROOT/_control/plan.json"

require_file() {
  [[ -s "$1" ]] || { echo "Missing required non-empty file: $1" >&2; exit 1; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

require_file "$CONFIG"
require_file "$POSE_ADAPT_INIT_CKPT"
require_file "$CACHE_READY"
require_file "$CACHE_PLAN"
require_dir "$HEATMAP_DATA_ROOT"
require_dir "$HEATMAP_AMB3R_POSE_CACHE_ROOT"
require_dir "$INTERNNAV_MODEL_PATH"

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [[ "${#GPU_LIST[@]}" -ne "$NPROC_PER_NODE" || "$NPROC_PER_NODE" -ne "$EXPECTED_NUM_GPUS" ]]; then
  echo "Pose adaptation requires exactly $EXPECTED_NUM_GPUS GPU devices" >&2
  exit 1
fi

mkdir -p "$SINGLE_VIEW_HM_OUT_DIR" "$SINGLE_VIEW_HM_TB_DIR" "$LOG_DIR"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${RUN_STAMP}.log}"

# Validate all 6000 endpoint sidecars semantically before loading either large
# model. This checks identities and causal metadata, not a checkpoint digest.
"$QWEN25_ENV/bin/python" scripts/amb3r_vo/validate_training_cache.py \
  --plan "$CACHE_PLAN" --workers 24 --require-shard-ready

"$QWEN25_ENV/bin/python" - \
  "$CONFIG" "$POSE_ADAPT_INIT_CKPT" "$CACHE_READY" "$CACHE_PLAN" \
  "$MAP_INIT_WINDOW" "$MAP_EVERY" "$EXPECTED_NUM_GPUS" <<'PY'
import json
import os
import sys
import torch

from src.config_schema import load_and_validate_config
from scripts.training.pose_adaptation import (
    EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS,
    HEATMAP_HEAD_PREFIXES,
    POSE_ADAPTATION_PREFIXES,
)
from scripts.training.utils import _normalize_state_key

cfg = load_and_validate_config(sys.argv[1])
stage = cfg["training"]["stages"][0]
ready = json.loads(open(sys.argv[3], encoding="utf-8").read())
plan = json.loads(open(sys.argv[4], encoding="utf-8").read())
map_init_window = int(sys.argv[5])
map_every = int(sys.argv[6])
expected_num_gpus = int(sys.argv[7])
expected_row_policy = "official_map_update_endpoints_plus_final"
assert ready["schema"] == "heatmapvln-amb3r-endpoint-pose-cache-ready-v2"
assert plan["schema"] == "heatmapvln-amb3r-endpoint-pose-cache-plan-v2"
for payload in (ready, plan):
    assert payload["endpoint_only"] is True
    assert payload["row_policy"] == expected_row_policy
    assert payload["query_only_at_map_endpoints"] is True
    assert payload["query_every_frame"] is False
    assert payload["future_pose_revisions_used"] is False
    assert payload["map_init_window"] == map_init_window
    assert payload["map_every"] == map_every
assert ready["complete"] is True
assert ready["cache_root"] == os.environ["HEATMAP_AMB3R_POSE_CACHE_ROOT"]
assert ready["dataset_root"] == os.environ["HEATMAP_DATA_ROOT"]
assert plan["cache_root"] == os.environ["HEATMAP_AMB3R_POSE_CACHE_ROOT"]
assert plan["dataset_root"] == os.environ["HEATMAP_DATA_ROOT"]
assert cfg["data"]["root"] == os.environ["HEATMAP_DATA_ROOT"]
assert cfg["data"]["sliding_window"]["amb3r_pose_cache_root"] == os.environ["HEATMAP_AMB3R_POSE_CACHE_ROOT"]
assert cfg["data"]["sliding_window"]["require_amb3r_pose_cache"] is True
assert cfg["model"]["llm"]["model_path"] == os.environ["INTERNNAV_MODEL_PATH"]
assert cfg["model"]["llm"]["use_lora"] is False
assert cfg["model"]["action_head"]["enable"] is False
assert len(cfg["gpu"]["devices"]) == expected_num_gpus
assert stage["required_history_pose_provider"] == "amb3r_vo_cache"
assert set(stage["heatmap_trainable_parameter_prefixes"]) == set(POSE_ADAPTATION_PREFIXES)
assert cfg["optim"]["heatmap_proj_traj_lr"] == 1e-4
assert cfg["optim"]["heatmap_coarse_lr"] == 2e-5
assert cfg["validation"]["save_best_metric"] == "val_heatmap_joint_pck4"

payload = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
state = payload.get("trainable_state_dict")
if not isinstance(state, dict):
    raise RuntimeError("Initializer has no deployment trainable_state_dict")
head = {
    _normalize_state_key(name): value
    for name, value in state.items()
    if _normalize_state_key(name).startswith(HEATMAP_HEAD_PREFIXES)
}
if len(head) != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS:
    raise RuntimeError(
        "Initializer is not the exact complete 79-tensor Heatmap Head: "
        f"found={len(head)}"
    )
print(
    "Validated endpoint-v2 root-ready/plan, AMB3R pose-adaptation config, "
    "complete initializer, and no-GT provider contract"
)
PY

echo "[pose-adapt] config=$CONFIG"
echo "[pose-adapt] data=$HEATMAP_DATA_ROOT"
echo "[pose-adapt] cache=$HEATMAP_AMB3R_POSE_CACHE_ROOT"
echo "[pose-adapt] init=$POSE_ADAPT_INIT_CKPT (runtime path; no hash pin)"
echo "[pose-adapt] output=$SINGLE_VIEW_HM_OUT_DIR"
echo "[pose-adapt] log=$LOG_FILE"

TRAIN_ARGS=(scripts/train.py --config "$CONFIG")
if [[ -n "${POSE_ADAPT_RESUME:-}" ]]; then
  TRAIN_ARGS+=(--resume "$POSE_ADAPT_RESUME")
  echo "[pose-adapt] resume=$POSE_ADAPT_RESUME (self-contained checkpoint)"
else
  TRAIN_ARGS+=(--load-weights "$POSE_ADAPT_INIT_CKPT")
fi
if [[ "${POSE_ADAPT_DRY_RUN:-0}" == "1" ]]; then
  if [[ -n "${POSE_ADAPT_RESUME:-}" ]]; then
    echo "POSE_ADAPT_DRY_RUN cannot be combined with POSE_ADAPT_RESUME" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--dry-run --max-batches "${POSE_ADAPT_MAX_BATCHES:-1}")
elif [[ -n "${POSE_ADAPT_MAX_BATCHES:-}" ]]; then
  TRAIN_ARGS+=(--max-batches "$POSE_ADAPT_MAX_BATCHES")
fi

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$QWEN25_ENV/bin/torchrun" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
