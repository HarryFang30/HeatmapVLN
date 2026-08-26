#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/liwenhao/agent/370910109/HeatmapVLN"

# Keep the same effective global batch as the validated eight-card run:
# 4 ranks * 2 samples/rank * 4 gradient-accumulation steps = 32.
export GPU_DEVICES="0,1,2,3"
export NPROC_PER_NODE="4"
export MASTER_PORT="${MASTER_PORT:-29623}"
export SINGLE_VIEW_HM_CONFIG="configs/train_heatmap_internnav_single_view_4gpu.yaml"
export HEATMAP_DATA_ROOT="/mnt/afs/liwenhao/agent/370910109/data/heatmap_randomwalk_train_v1"
export INTERNNAV_MODEL_PATH="/mnt/afs/liwenhao/agent/370910109/InternNav-Model"

# Isolate four-card checkpoints/logs while reusing the already audited,
# heatmap-only 53-tensor initializer (which contains no Qwen LoRA tensors).
export SINGLE_VIEW_HM_EXPERIMENT_ROOT="/mnt/afs/liwenhao/agent/370910109/model/output_heatmap_internnav_single_view_v1_4gpu"
export SINGLE_VIEW_HM_OUT_DIR="$SINGLE_VIEW_HM_EXPERIMENT_ROOT/runs"
export SINGLE_VIEW_HM_TB_DIR="$SINGLE_VIEW_HM_EXPERIMENT_ROOT/tensorboard"
export SINGLE_VIEW_HM_INIT_CKPT="/mnt/afs/liwenhao/agent/370910109/model/output_heatmap_internnav_single_view_v1/init/from_legacy_heatmap_53tensors_v2.pth"

# Do not inherit a generic LOG_FILE from a previous submission; the parent
# launcher will create a timestamped log below this experiment root.
unset LOG_FILE

exec bash "$REPO_ROOT/scripts/run_heatmap_internnav_single_view_8gpu_mxc500.sh"
