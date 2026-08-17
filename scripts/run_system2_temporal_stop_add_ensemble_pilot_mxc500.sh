#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
STATIC_STOP_HEAD="${SYSTEM2_TEMPORAL_ADD_STATIC_STOP_HEAD:-${FJL_ROOT}/model/output_system2_stop_head_full_11000_alllora_h1024/latest/checkpoints/epoch_001.pth}"
OUTPUT_DIR="${SYSTEM2_TEMPORAL_ADD_OUT_DIR:-${FJL_ROOT}/model/output_system2_temporal_stop_add_ensemble_train_rollouts_pilot}"
GPU_DEVICE="${SYSTEM2_TEMPORAL_ADD_GPU:-7}"

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
  echo "Refusing to overwrite temporal STOP-add output: $OUTPUT_DIR" >&2
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

echo "[temporal-stop-add] static_prior=$STATIC_STOP_HEAD"
echo "[temporal-stop-add] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_temporal_stop_ensemble_from_rollout_cache.py \
  "${label_args[@]}" \
  --static-stop-head-checkpoint "$STATIC_STOP_HEAD" \
  --output-dir "$OUTPUT_DIR" \
  --objective add \
  --candidate-scope original_nonterminal \
  --folds 5 \
  --epochs 80 \
  --batch-size 64 \
  --hidden-dim 16 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --feature-load-workers 32 \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --seed 123 \
  --device cuda

"$QWEN25_PYTHON" - "$OUTPUT_DIR/latest.pth" <<'PY'
import math
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint.get("config", {}).get("temporal_stop_verifier", {})
metrics = checkpoint.get("metrics", {}).get("oof", {})
state = checkpoint.get("trainable_state_dict")
errors = []
if checkpoint.get("stage_name") != "system2_temporal_stop_add_ensemble":
    errors.append("wrong stage_name")
if config.get("veto_only") is not False or config.get("add_only") is not True:
    errors.append("checkpoint is not add-only")
if config.get("candidate_scope") != "original_nonterminal":
    errors.append("wrong candidate scope")
if config.get("aggregation") != "unanimous" or int(config.get("ensemble_size", 0)) != 5:
    errors.append("wrong ensemble contract")
if float(metrics.get("false_positive_rate", 1.0)) != 0.0:
    errors.append("OOF contains false STOP additions")
if float(metrics.get("recall", 0.0)) <= 0.0:
    errors.append("OOF has zero STOP-add recall")
if not isinstance(state, dict) or not state:
    errors.append("missing ensemble state")
elif not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    errors.append("non-finite ensemble state")
thresholds = config.get("acceptance_thresholds") or []
if not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in thresholds):
    errors.append("invalid thresholds")
if errors:
    raise SystemExit("Temporal STOP-add postflight failed: " + "; ".join(errors))
print(
    "Temporal STOP-add postflight passed: "
    f"recall={float(metrics['recall']):.4f} fpr=0 "
    f"tp={int(metrics['tp'])} fn={int(metrics['fn'])}"
)
PY

echo "[temporal-stop-add] complete: $OUTPUT_DIR/latest.pth"
