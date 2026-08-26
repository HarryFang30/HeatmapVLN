#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/afs/liwenhao/agent/370910109/HeatmapVLN
PYTHON_BIN=/mnt/afs/liwenhao/agent/370910109/envs/qwen25/bin/python
AUDIT_ROOT=${PROBE_AUDIT_ROOT:-/mnt/afs/liwenhao/agent/370910109/data/candidate_support_audit_v2/train_balanced_512_native_seed42}
OUTPUT_DIR=${PROBE_OUTPUT_DIR:-/mnt/afs/liwenhao/agent/370910109/model/candidate_identifiability_probe_v2/train_balanced_512_native_seed42}
GPU_DEVICE=${PROBE_GPU_DEVICE:-0}

cd "$PROJECT_ROOT"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

if ! test -x "$PYTHON_BIN"; then
  echo "[probe] missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

for shard_id in 00 01 02 03 04 05 06 07; do
  shard_dir="$AUDIT_ROOT/shard_$shard_id"
  if ! test -s "$shard_dir/records.jsonl" || ! test -s "$shard_dir/manifest.json"; then
    echo "[probe] balanced audit is not sealed: $shard_dir" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${PROBE_OMP_NUM_THREADS:-8}
mkdir -p "$OUTPUT_DIR/runtime/triton_cache"
export TRITON_CACHE_DIR="$OUTPUT_DIR/runtime/triton_cache"

arguments=(
  --audit-root "$AUDIT_ROOT"
  --output-dir "$OUTPUT_DIR"
  --expected-shards 8
  --scene-split-seed "${PROBE_SCENE_SPLIT_SEED:-20260810}"
  --split-ratios "${PROBE_SPLIT_RATIOS:-0.7,0.15,0.15}"
  --model-seeds "${PROBE_MODEL_SEEDS:-17,42,73}"
  --variants "${PROBE_VARIANTS:-candidate_only,candidate_system2,candidate_system2_heatmap_metadata,candidate_system2_heatmap_tokens}"
  --metric-resolution-m "${PROBE_METRIC_RESOLUTION_M:-0.05}"
  --max-validation-destroy-state-rate "${PROBE_MAX_VALIDATION_DESTROY_STATE_RATE:-0.02}"
  --hidden-width "${PROBE_HIDDEN_WIDTH:-128}"
  --batch-size "${PROBE_BATCH_SIZE:-64}"
  --epochs "${PROBE_EPOCHS:-30}"
  --patience "${PROBE_PATIENCE:-5}"
  --learning-rate "${PROBE_LEARNING_RATE:-0.0003}"
  --weight-decay "${PROBE_WEIGHT_DECAY:-0.0001}"
  --device cuda
)

if test "${PROBE_PREFLIGHT_ONLY:-0}" = 1; then
  arguments+=(--preflight-only)
fi
if test "${PROBE_DEV_ALLOW_NONDISJOINT_SCENE_SPLIT:-0}" = 1; then
  arguments+=(--dev-allow-nondisjoint-scene-split)
fi

echo "[probe] audit_root=$AUDIT_ROOT"
echo "[probe] output_dir=$OUTPUT_DIR"
echo "[probe] gpu_device=$GPU_DEVICE"
exec "$PYTHON_BIN" scripts/evaluation/probe_candidate_identifiability.py "${arguments[@]}"
