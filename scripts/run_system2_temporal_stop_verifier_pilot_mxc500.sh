#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
STATIC_STOP_HEAD="${SYSTEM2_TEMPORAL_STATIC_STOP_HEAD:-${FJL_ROOT}/model/output_system2_stop_head_full_11000_alllora_h1024/latest/checkpoints/epoch_001.pth}"
OUTPUT_DIR="${SYSTEM2_TEMPORAL_STOP_OUT_DIR:-${FJL_ROOT}/model/output_system2_temporal_stop_verifier_train_rollouts_pilot}"
GPU_DEVICE="${SYSTEM2_TEMPORAL_STOP_GPU:-7}"

LABEL_FILES=(
  "${FJL_ROOT}/model/smoke_stop_dagger_collect_train3_fixed_20260716_170658/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/smoke_stop_dagger_collect_train_all61_seed46_a_20260716/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/smoke_stop_dagger_collect_train_all61_seed46_b_20260716/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/smoke_stop_dagger_collect_train_scene16_seed43_20260716/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/smoke_stop_dagger_collect_train_scene8_20260716_1727/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_00/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_01/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_02/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_03/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_04/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_05/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/system2_stop_dagger_long6_20260716/cohort_missing_gz6/system2_stop_rollout_labels.jsonl"
  "${FJL_ROOT}/model/smoke_system2_stop_force_continue_train3_20260717_1235/system2_stop_rollout_labels.jsonl"
)

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ ! -s "$STATIC_STOP_HEAD" ]]; then
  echo "Missing frozen static STOP prior: $STATIC_STOP_HEAD" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite temporal STOP output: $OUTPUT_DIR" >&2
  exit 1
fi

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

label_args=()
for labels_file in "${LABEL_FILES[@]}"; do
  label_args+=(--labels-jsonl "$labels_file")
done

echo "[temporal-stop] static_prior=$STATIC_STOP_HEAD"
echo "[temporal-stop] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_temporal_stop_verifier_from_rollout_cache.py \
  "${label_args[@]}" \
  --static-stop-head-checkpoint "$STATIC_STOP_HEAD" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 16 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --feature-load-workers 32 \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --val-fraction 0.2 \
  --seed 123 \
  --device cuda

PYTHONPATH="$REPO_ROOT" "$QWEN25_PYTHON" - "$OUTPUT_DIR/latest.pth" <<'PY'
import sys

from scripts.evaluation.preflight_stage3_rpc_eval import (
    validate_temporal_stop_verifier_checkpoint,
)

summary = validate_temporal_stop_verifier_checkpoint(sys.argv[1])
if summary["veto_only"] is not True:
    raise SystemExit("Temporal STOP postflight is not veto-only")
print(f"Temporal STOP postflight passed: {summary}")
PY

echo "[temporal-stop] complete: $OUTPUT_DIR/latest.pth"
