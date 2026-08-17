#!/usr/bin/env bash
# One queued 8-GPU job: strict real eight-rank smoke, resumable causal endpoint
# cache, then 5-epoch Head-only AMB3R pose adaptation. No phase pins a
# checkpoint digest or uses locks.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
cd "$REPO_ROOT"

export AMB3R_GPU_DEVICES="${AMB3R_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export GPU_DEVICES="${GPU_DEVICES:-$AMB3R_GPU_DEVICES}"
export EXPECTED_NUM_GPUS="${EXPECTED_NUM_GPUS:-8}"
FORMAL_DATA_ROOT="${FORMAL_DATA_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}"
FORMAL_CACHE_ROOT="${FORMAL_CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2}"
FORMAL_CONFIG="${FORMAL_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_8gpu.yaml}"
FORMAL_EXPERIMENT_ROOT="${FORMAL_EXPERIMENT_ROOT:-${ALLOWED_ROOT}/model/output_heatmap_amb3r_pose_adapt_endpoint_v2}"
FORMAL_INIT_CKPT="${FORMAL_INIT_CKPT:-${ALLOWED_ROOT}/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth}"
POSE_ADAPT_SMOKE_ROOT="${POSE_ADAPT_SMOKE_ROOT:-${ALLOWED_ROOT}/data/heatmap_amb3r_pose_adapt_8gpu_smoke_v2}"
FORMAL_MAP_INIT_WINDOW="${FORMAL_MAP_INIT_WINDOW:-20}"
FORMAL_MAP_EVERY="${FORMAL_MAP_EVERY:-8}"
FORMAL_RUN_TAG="${FORMAL_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_MASTER_PORT="${SMOKE_MASTER_PORT:-29651}"
FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-29641}"

# Resolve symlinks, including a dangling final component. This prevents
# user/web-job overrides from aliasing the formal cache into the disposable
# smoke subtree.
canonical_path() {
  local raw="$1"
  realpath -m -- "$raw"
}
FORMAL_CACHE_CANON="$(canonical_path "$FORMAL_CACHE_ROOT")"
SMOKE_ROOT_CANON="$(canonical_path "$POSE_ADAPT_SMOKE_ROOT")"
FORMAL_DATA_CANON="$(canonical_path "$FORMAL_DATA_ROOT")"
FORMAL_EXPERIMENT_CANON="$(canonical_path "$FORMAL_EXPERIMENT_ROOT")"
FORMAL_INIT_CANON="$(canonical_path "$FORMAL_INIT_CKPT")"
for formal_path in \
  "$FORMAL_DATA_CANON" \
  "$FORMAL_CACHE_CANON" \
  "$FORMAL_EXPERIMENT_CANON" \
  "$FORMAL_INIT_CANON" \
  "$SMOKE_ROOT_CANON"; do
  case "$formal_path" in
    "$ALLOWED_ROOT"/*) ;;
    *)
      echo "Production path must stay below $ALLOWED_ROOT: $formal_path" >&2
      exit 2
      ;;
  esac
done
case "$FORMAL_CACHE_CANON" in
  "$SMOKE_ROOT_CANON"|"$SMOKE_ROOT_CANON"/*)
    echo "Formal cache must be disjoint from smoke root: $FORMAL_CACHE_CANON" >&2
    exit 2
    ;;
esac
case "$SMOKE_ROOT_CANON" in
  "$FORMAL_CACHE_CANON"|"$FORMAL_CACHE_CANON"/*)
    echo "Smoke root must be disjoint from formal cache: $SMOKE_ROOT_CANON" >&2
    exit 2
    ;;
esac

# Ignore stale variables inherited from a previous interactive/web job.  The
# FORMAL_* namespace above is the only supported override surface for this
# three-phase production workflow.
unset DATASET_ROOT CACHE_ROOT PLAN_PATH SPLITS MAX_CLIPS_PER_SPLIT LOG_ROOT
unset SOURCE_DATA_ROOT HEATMAP_DATA_ROOT HEATMAP_AMB3R_POSE_CACHE_ROOT
unset POSE_ADAPT_CONFIG POSE_ADAPT_DRY_RUN POSE_ADAPT_MAX_BATCHES
unset POSE_ADAPT_RESUME POSE_ADAPT_8GPU_SMOKE_AUDIT POSE_ADAPT_SMOKE_WORLD_SIZE POSE_ADAPT_EXPERIMENT_ROOT
unset SINGLE_VIEW_HM_OUT_DIR SINGLE_VIEW_HM_TB_DIR

echo "[pose-adapt-pipeline] phase 1/3: strict real eight-rank training smoke"
POSE_ADAPT_SMOKE_ROOT="$POSE_ADAPT_SMOKE_ROOT" \
SOURCE_DATA_ROOT="$FORMAL_DATA_ROOT" \
POSE_ADAPT_INIT_CKPT="$FORMAL_INIT_CKPT" \
GPU_DEVICES="$GPU_DEVICES" \
EXPECTED_NUM_GPUS="$EXPECTED_NUM_GPUS" \
POSE_ADAPT_BASE_CONFIG="$FORMAL_CONFIG" \
MASTER_PORT="$SMOKE_MASTER_PORT" \
RUN_TAG="$FORMAL_RUN_TAG" \
MAP_INIT_WINDOW="$FORMAL_MAP_INIT_WINDOW" \
MAP_EVERY="$FORMAL_MAP_EVERY" \
bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_smoke_mxc500.sh

SMOKE_READY="$POSE_ADAPT_SMOKE_ROOT/smoke.ready.json"
if [[ ! -s "$SMOKE_READY" ]]; then
  echo "Eight-rank pose-adaptation smoke did not publish: $SMOKE_READY" >&2
  exit 3
fi

echo "[pose-adapt-pipeline] phase 2/3: causal AMB3R endpoint cache (semantic resume)"
ALLOWED_ROOT="$ALLOWED_ROOT" \
DATASET_ROOT="$FORMAL_DATA_ROOT" \
CACHE_ROOT="$FORMAL_CACHE_ROOT" \
PLAN_PATH="$FORMAL_CACHE_ROOT/_control/plan.json" \
SPLITS="train,val" \
MAX_CLIPS_PER_SPLIT=0 \
NUM_HISTORY=8 \
MIN_HISTORY=5 \
MAP_INIT_WINDOW="$FORMAL_MAP_INIT_WINDOW" \
MAP_EVERY="$FORMAL_MAP_EVERY" \
CLIP_RETRIES=2 \
SHARD_MAX_ATTEMPTS=0 \
LOG_ROOT="$FORMAL_CACHE_ROOT/_control/logs/$FORMAL_RUN_TAG" \
AMB3R_GPU_DEVICES="$AMB3R_GPU_DEVICES" \
EXPECTED_NUM_GPUS="$EXPECTED_NUM_GPUS" \
bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh

echo "[pose-adapt-pipeline] phase 3/3: existing Head pose-domain adaptation"
HEATMAP_DATA_ROOT="$FORMAL_DATA_ROOT" \
HEATMAP_AMB3R_POSE_CACHE_ROOT="$FORMAL_CACHE_ROOT" \
POSE_ADAPT_CONFIG="$FORMAL_CONFIG" \
POSE_ADAPT_INIT_CKPT="$FORMAL_INIT_CKPT" \
POSE_ADAPT_EXPERIMENT_ROOT="$FORMAL_EXPERIMENT_ROOT" \
SINGLE_VIEW_HM_OUT_DIR="$FORMAL_EXPERIMENT_ROOT/runs" \
SINGLE_VIEW_HM_TB_DIR="$FORMAL_EXPERIMENT_ROOT/tensorboard" \
POSE_ADAPT_DRY_RUN=0 \
POSE_ADAPT_MAX_BATCHES= \
POSE_ADAPT_RESUME= \
POSE_ADAPT_8GPU_SMOKE_AUDIT=0 \
POSE_ADAPT_SMOKE_WORLD_SIZE="$EXPECTED_NUM_GPUS" \
MAP_INIT_WINDOW="$FORMAL_MAP_INIT_WINDOW" \
MAP_EVERY="$FORMAL_MAP_EVERY" \
GPU_DEVICES="$GPU_DEVICES" \
NPROC_PER_NODE="$EXPECTED_NUM_GPUS" \
EXPECTED_NUM_GPUS="$EXPECTED_NUM_GPUS" \
MASTER_PORT="$FORMAL_MASTER_PORT" \
bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_mxc500.sh
