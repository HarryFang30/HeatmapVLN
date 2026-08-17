#!/usr/bin/env bash
# Dedicated no-checkpoint, one-step-per-rank AMB3R pose-adaptation smoke.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
QWEN_PYTHON="${ALLOWED_ROOT}/envs/qwen25/bin/python"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}"
EXPECTED_NUM_GPUS="${EXPECTED_NUM_GPUS:-8}"
SMOKE_ROOT="${POSE_ADAPT_SMOKE_ROOT:-${ALLOWED_ROOT}/data/heatmap_amb3r_pose_adapt_8gpu_smoke_v2}"
SMOKE_DATA_ROOT="${SMOKE_ROOT}/data"
SMOKE_CACHE_ROOT="${SMOKE_ROOT}/cache"
SMOKE_CONFIG="${SMOKE_CONFIG:-${SMOKE_ROOT}/train_heatmap_amb3r_pose_adapt_${EXPECTED_NUM_GPUS}gpu_smoke.yaml}"
SMOKE_READY="${SMOKE_ROOT}/smoke.ready.json"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
SMOKE_TRAIN_CLIPS="${SMOKE_TRAIN_CLIPS:-2}"
POSE_ADAPT_BASE_CONFIG="${POSE_ADAPT_BASE_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_8gpu.yaml}"
MAP_INIT_WINDOW="${MAP_INIT_WINDOW:-20}"
MAP_EVERY="${MAP_EVERY:-8}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$REPO_ROOT"
[[ -x "$QWEN_PYTHON" ]] || { echo "Missing qwen25 Python: $QWEN_PYTHON" >&2; exit 2; }
IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
[[ "${#GPU_LIST[@]}" -eq "$EXPECTED_NUM_GPUS" ]] || {
  echo "GPU_DEVICES must contain exactly $EXPECTED_NUM_GPUS devices" >&2
  exit 2
}

mkdir -p "$SMOKE_ROOT"
rm -f "$SMOKE_READY"

echo "[pose-adapt-smoke] preparing ${SMOKE_TRAIN_CLIPS}-train-clip real-directory view"
"$QWEN_PYTHON" scripts/amb3r_vo/prepare_pose_adapt_smoke_view.py \
  --source-root "$SOURCE_DATA_ROOT" \
  --smoke-root "$SMOKE_ROOT" \
  --num-train-clips "$SMOKE_TRAIN_CLIPS" \
  --allowed-root "$ALLOWED_ROOT"

echo "[pose-adapt-8gpu-smoke] deriving runtime-only smoke config from production config"
"$QWEN_PYTHON" scripts/tools/build_pose_adapt_smoke_config.py \
  --base "$POSE_ADAPT_BASE_CONFIG" \
  --output "$SMOKE_CONFIG"

echo "[pose-adapt-smoke] caching train clips with ${EXPECTED_NUM_GPUS} shards"
ALLOWED_ROOT="$ALLOWED_ROOT" \
DATASET_ROOT="$SMOKE_DATA_ROOT" \
CACHE_ROOT="$SMOKE_CACHE_ROOT" \
PLAN_PATH="$SMOKE_CACHE_ROOT/_control/plan.json" \
SPLITS=train \
MAX_CLIPS_PER_SPLIT="$SMOKE_TRAIN_CLIPS" \
AMB3R_GPU_DEVICES="$GPU_DEVICES" \
EXPECTED_NUM_GPUS="$EXPECTED_NUM_GPUS" \
MAP_INIT_WINDOW="$MAP_INIT_WINDOW" \
MAP_EVERY="$MAP_EVERY" \
RUN_TAG="smoke_${RUN_TAG}" \
bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh

echo "[pose-adapt-smoke] running production train.py dry-run on ${EXPECTED_NUM_GPUS} ranks"
export HEATMAP_DATA_ROOT="$SMOKE_DATA_ROOT"
export HEATMAP_AMB3R_POSE_CACHE_ROOT="$SMOKE_CACHE_ROOT"
export POSE_ADAPT_EXPERIMENT_ROOT="$SMOKE_ROOT/training"
export SINGLE_VIEW_HM_OUT_DIR="$POSE_ADAPT_EXPERIMENT_ROOT/runs"
export SINGLE_VIEW_HM_TB_DIR="$POSE_ADAPT_EXPERIMENT_ROOT/tensorboard"
export POSE_ADAPT_CONFIG="$SMOKE_CONFIG"
export POSE_ADAPT_DRY_RUN=1
export POSE_ADAPT_MAX_BATCHES=1
export POSE_ADAPT_8GPU_SMOKE_AUDIT=1
export POSE_ADAPT_SMOKE_WORLD_SIZE="$EXPECTED_NUM_GPUS"
export MAP_INIT_WINDOW
export MAP_EVERY
export GPU_DEVICES
export NPROC_PER_NODE="$EXPECTED_NUM_GPUS"
export EXPECTED_NUM_GPUS
export MASTER_PORT="${MASTER_PORT:-29651}"
bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_mxc500.sh

REPORT=""
for candidate in "$SINGLE_VIEW_HM_OUT_DIR"/preflight_*/manifest/preflight.json; do
  [[ -f "$candidate" ]] && REPORT="$candidate"
done
[[ -n "$REPORT" ]] || {
  echo "No train.py preflight report found under $SINGLE_VIEW_HM_OUT_DIR" >&2
  exit 1
}

echo "[pose-adapt-smoke] validating $((EXPECTED_NUM_GPUS * 2)) unique identities, gradients, sync, EMA, and no checkpoint"
"$QWEN_PYTHON" scripts/tools/validate_pose_adapt_8gpu_smoke.py \
  --preflight-report "$REPORT" \
  --output "$SMOKE_READY" \
  --world-size "$EXPECTED_NUM_GPUS" \
  --allowed-root "$ALLOWED_ROOT"

echo "[pose-adapt-smoke] READY: $SMOKE_READY"
