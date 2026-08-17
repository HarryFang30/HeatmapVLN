#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
INIT_CHECKPOINT="${SYSTEM2_STOP_VETO_INIT_CHECKPOINT:-${FJL_ROOT}/model/output_system2_stop_head_full_11000_alllora_h1024/latest/checkpoints/epoch_001.pth}"
OUTPUT_DIR="${SYSTEM2_STOP_VETO_OUT_DIR:-${FJL_ROOT}/model/output_system2_stop_veto_intervention_train3_pilot}"
GPU_DEVICE="${SYSTEM2_STOP_VETO_GPU:-7}"

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
if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "Missing static STOP prior: $INIT_CHECKPOINT" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing STOP-veto pilot: $OUTPUT_DIR" >&2
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

echo "[stop-veto-pilot] init=$INIT_CHECKPOINT"
echo "[stop-veto-pilot] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_stop_head_from_rollout_cache.py \
  "${label_args[@]}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --training-scope original-terminal \
  --epochs 30 \
  --batch-size 64 \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --l2-sp-weight 1e-3 \
  --terminal-negative-weight 1 \
  --hard-negative-threshold 0.8 \
  --hard-negative-weight 2 \
  --feature-load-workers 32 \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --val-fraction 0.2 \
  --seed 123 \
  --device cuda

"$QWEN25_PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
checkpoint = torch.load(output_dir / "latest.pth", map_location="cpu", weights_only=False)

if summary.get("training_scope") != "original-terminal":
    raise SystemExit("STOP-veto pilot has the wrong training scope")
if int(summary.get("positive_records", 0)) < 20:
    raise SystemExit("STOP-veto pilot has too few positive terminal candidates")
if int(summary.get("negative_records", 0)) < 20:
    raise SystemExit("STOP-veto pilot has too few negative terminal candidates")
state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or len(state) != 10:
    raise SystemExit("STOP-veto pilot checkpoint is incomplete")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("STOP-veto pilot checkpoint contains non-finite tensors")
scope = checkpoint["config"]["rollout_stop_training"].get("training_scope")
if scope != "original-terminal":
    raise SystemExit(f"Checkpoint training scope mismatch: {scope!r}")
head_config = checkpoint["config"]["model"]["stop_head"]
veto = float(head_config["veto_stop_threshold"])
add = float(head_config["add_stop_threshold"])
if not (math.isfinite(veto) and math.isfinite(add) and 0.0 <= veto < add <= 1.0):
    raise SystemExit(f"Invalid STOP-veto thresholds: veto={veto} add={add}")
print(
    "STOP-veto pilot postflight passed: "
    f"records={summary['records']} positives={summary['positive_records']} "
    f"negatives={summary['negative_records']} best_epoch={summary['best_epoch']} "
    f"veto={veto:.3f}"
)
PY

echo "[stop-veto-pilot] complete: $OUTPUT_DIR/latest.pth"
