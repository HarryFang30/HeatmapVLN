#!/usr/bin/env bash
# Four persistent AMB3R workers using the shared endpoint-v2 cache contract.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
cd "$REPO_ROOT"

export EXPECTED_NUM_GPUS=4
export AMB3R_GPU_DEVICES="${AMB3R_GPU_DEVICES:-0,1,2,3}"
export CACHE_ROOT="${CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu}"
export PLAN_PATH="${PLAN_PATH:-${CACHE_ROOT}/_control/plan.json}"

exec bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh
