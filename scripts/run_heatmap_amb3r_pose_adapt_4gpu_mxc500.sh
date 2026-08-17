#!/usr/bin/env bash
# Four-rank Head-only AMB3R pose-domain adaptation.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
cd "$REPO_ROOT"

export EXPECTED_NUM_GPUS=4
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export MASTER_PORT="${MASTER_PORT:-29642}"
export POSE_ADAPT_CONFIG="${POSE_ADAPT_CONFIG:-configs/train_heatmap_amb3r_pose_adapt_4gpu.yaml}"
export HEATMAP_AMB3R_POSE_CACHE_ROOT="${HEATMAP_AMB3R_POSE_CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu}"
export POSE_ADAPT_EXPERIMENT_ROOT="${POSE_ADAPT_EXPERIMENT_ROOT:-${ALLOWED_ROOT}/model/output_heatmap_amb3r_pose_adapt_endpoint_v2_4gpu}"
export SINGLE_VIEW_HM_OUT_DIR="${SINGLE_VIEW_HM_OUT_DIR:-${POSE_ADAPT_EXPERIMENT_ROOT}/runs}"
export SINGLE_VIEW_HM_TB_DIR="${SINGLE_VIEW_HM_TB_DIR:-${POSE_ADAPT_EXPERIMENT_ROOT}/tensorboard}"

exec bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_mxc500.sh
