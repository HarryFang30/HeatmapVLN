#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN
PY=/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python
CLIP_DIR=${FUTURE_HEATMAP_CLIP_DIR:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data/train/S9hNv5qa7GM/clip_007409}
OUTPUT_ROOT=${FUTURE_HEATMAP_OUTPUT_ROOT:-/mnt/afs/lixiaoou/intern/fjl/model/future_heatmap_visualization_v1}

export MPLCONFIGDIR=/mnt/afs/lixiaoou/intern/fjl/model/.runtime_cache/matplotlib_future_heatmap
mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"

cd "$REPO"
"$PY" scripts/visualization/visualize_system2_future_heatmap.py \
  --clip-dir "$CLIP_DIR" \
  --output "$OUTPUT_ROOT/S9hNv5qa7GM_clip_007409_future_heatmap.png" \
  --semantics-output "$OUTPUT_ROOT/future_heatmap_semantics.png" \
  --frame-count "${FUTURE_HEATMAP_FRAME_COUNT:-24}" \
  --confidence "${FUTURE_HEATMAP_DISPLAY_CONFIDENCE:-0.85}" \
  --tile "${FUTURE_HEATMAP_TILE:-64}" \
  --gap "${FUTURE_HEATMAP_GAP:-4}"

echo "[future-heatmap] visualization ready: $OUTPUT_ROOT"

