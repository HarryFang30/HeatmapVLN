#!/usr/bin/env bash
# Real four-rank cache + forward/backward/DDP/EMA smoke; writes no checkpoint.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
cd "$REPO_ROOT"

export EXPECTED_NUM_GPUS=4
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export SMOKE_TRAIN_CLIPS="${SMOKE_TRAIN_CLIPS:-4}"
export POSE_ADAPT_BASE_CONFIG="${POSE_ADAPT_BASE_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_4gpu.yaml}"
export POSE_ADAPT_SMOKE_ROOT="${POSE_ADAPT_SMOKE_ROOT:-${ALLOWED_ROOT}/data/heatmap_amb3r_pose_adapt_4gpu_smoke_v2}"
export MASTER_PORT="${MASTER_PORT:-29652}"

exec bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_smoke_mxc500.sh
