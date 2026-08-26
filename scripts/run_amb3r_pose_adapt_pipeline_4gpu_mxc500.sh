#!/usr/bin/env bash
# One queued four-GPU job: strict smoke, resumable endpoint cache, then the
# existing Head's five-epoch AMB3R pose-domain adaptation. No hash pin or lock.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/liwenhao/agent/370910109/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/liwenhao/agent/370910109"
cd "$REPO_ROOT"

export EXPECTED_NUM_GPUS=4
export AMB3R_GPU_DEVICES="${AMB3R_GPU_DEVICES:-0,1,2,3}"
export GPU_DEVICES="${GPU_DEVICES:-$AMB3R_GPU_DEVICES}"
export SMOKE_TRAIN_CLIPS="${SMOKE_TRAIN_CLIPS:-4}"
export FORMAL_DATA_ROOT="${FORMAL_DATA_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}"
export FORMAL_CACHE_ROOT="${FORMAL_CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu}"
export FORMAL_CONFIG="${FORMAL_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_4gpu.yaml}"
export FORMAL_EXPERIMENT_ROOT="${FORMAL_EXPERIMENT_ROOT:-${ALLOWED_ROOT}/model/output_heatmap_amb3r_pose_adapt_endpoint_v2_4gpu}"
export POSE_ADAPT_SMOKE_ROOT="${POSE_ADAPT_SMOKE_ROOT:-${ALLOWED_ROOT}/data/heatmap_amb3r_pose_adapt_4gpu_smoke_v2}"
export SMOKE_MASTER_PORT="${SMOKE_MASTER_PORT:-29652}"
export FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-29642}"

exec bash scripts/run_amb3r_pose_adapt_pipeline_8gpu_mxc500.sh
