#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

QWEN25_ENV="${QWEN25_ENV:-/mnt/afs/lixiaoou/intern/fjl/envs/qwen25}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${QWEN25_ENV}/bin/python}"
STOP_PRIOR_ROOT="${SYSTEM2_STOP_PRIOR_ROOT:-/mnt/afs/lixiaoou/intern/fjl/model/output_system2_stop_head_full_11000_alllora_h1024}"
INIT_CHECKPOINT="${SYSTEM2_STOP_DAGGER_INIT_CHECKPOINT:-${STOP_PRIOR_ROOT}/latest/checkpoints/epoch_001.pth}"
OUTPUT_DIR="${SYSTEM2_STOP_DAGGER_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_system2_stop_head_fullprior_dagger_all61_boundary301_seed123}"
GPU_DEVICE="${SYSTEM2_STOP_DAGGER_GPU:-7}"
WAIT_INTERVAL_S="${SYSTEM2_STOP_DAGGER_WAIT_INTERVAL_S:-300}"
SETTLE_S="${SYSTEM2_STOP_DAGGER_CHECKPOINT_SETTLE_S:-30}"

EPOCHS="${SYSTEM2_STOP_DAGGER_EPOCHS:-20}"
BATCH_SIZE="${SYSTEM2_STOP_DAGGER_BATCH_SIZE:-256}"
LEARNING_RATE="${SYSTEM2_STOP_DAGGER_LR:-1e-5}"
FEATURE_LOAD_WORKERS="${SYSTEM2_STOP_DAGGER_FEATURE_LOAD_WORKERS:-32}"
VAL_FRACTION="0.2"
SEED="123"
RELABEL_RADIUS_M="3.01"

LABEL_FILES=(
  /mnt/afs/lixiaoou/intern/fjl/model/smoke_stop_dagger_collect_train3_fixed_20260716_170658/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/smoke_stop_dagger_collect_train_all61_seed46_a_20260716/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/smoke_stop_dagger_collect_train_all61_seed46_b_20260716/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/smoke_stop_dagger_collect_train_scene16_seed43_20260716/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/smoke_stop_dagger_collect_train_scene8_20260716_1727/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_00/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_01/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_02/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_03/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_04/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_05/system2_stop_rollout_labels.jsonl
  /mnt/afs/lixiaoou/intern/fjl/model/system2_stop_dagger_long6_20260716/cohort_missing_gz6/system2_stop_rollout_labels.jsonl
)

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing DAgger output: $OUTPUT_DIR" >&2
  exit 1
fi

while [[ ! -s "$INIT_CHECKPOINT" ]]; do
  printf '[%s] waiting for complete full-data STOP prior: %s\n' \
    "$(date '+%F %T')" "$INIT_CHECKPOINT"
  sleep "$WAIT_INTERVAL_S"
done
sleep "$SETTLE_S"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

label_args=()
for labels_file in "${LABEL_FILES[@]}"; do
  label_args+=(--labels-jsonl "$labels_file")
done

echo "[stop-dagger] init=$INIT_CHECKPOINT"
echo "[stop-dagger] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"
echo "[stop-dagger] epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE feature_load_workers=$FEATURE_LOAD_WORKERS val_fraction=$VAL_FRACTION seed=$SEED relabel_radius_m=$RELABEL_RADIUS_M"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_stop_head_from_rollout_cache.py \
  "${label_args[@]}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --weight-decay 0.01 \
  --l2-sp-weight 1e-4 \
  --terminal-negative-weight 4 \
  --hard-negative-threshold 0.8 \
  --hard-negative-weight 8 \
  --feature-load-workers "$FEATURE_LOAD_WORKERS" \
  --relabel-ambiguous-negative-radius-m "$RELABEL_RADIUS_M" \
  --val-fraction "$VAL_FRACTION" \
  --seed "$SEED" \
  --device cuda

"$QWEN25_PYTHON" - "$OUTPUT_DIR" "$INIT_CHECKPOINT" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
init_checkpoint = Path(sys.argv[2])
checkpoint_path = output_dir / "latest.pth"
summary_path = output_dir / "summary.json"
if not checkpoint_path.is_file() or not summary_path.is_file():
    raise SystemExit("DAgger output is missing latest.pth or summary.json")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected = {
    "records": 20857,
    "train_records": 16228,
    "val_records": 4629,
    "positive_records": 1119,
    "negative_records": 19738,
    "relabelled_negative_records": 415,
}
for key, value in expected.items():
    if int(summary.get(key, -1)) != value:
        raise SystemExit(
            f"Unexpected DAgger summary {key}={summary.get(key)!r}; expected {value}"
        )

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("stage_name") != "system2_stop_head":
    raise SystemExit("DAgger checkpoint has the wrong stage_name")
if Path(checkpoint.get("source_init_checkpoint", "")) != init_checkpoint:
    raise SystemExit("DAgger checkpoint records the wrong initialization checkpoint")
state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or not state:
    raise SystemExit("DAgger checkpoint has no trainable_state_dict")
if not all(str(name).startswith("stop_head.") for name in state):
    raise SystemExit("DAgger checkpoint contains non-STOP-head trainable tensors")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("DAgger checkpoint contains invalid STOP-head tensors")

head_config = checkpoint["config"]["model"]["stop_head"]
add_threshold = float(head_config["add_stop_threshold"])
veto_threshold = float(head_config["veto_stop_threshold"])
if not (
    math.isfinite(veto_threshold)
    and math.isfinite(add_threshold)
    and 0.0 <= veto_threshold < add_threshold <= 1.0
):
    raise SystemExit(
        f"Invalid STOP thresholds: veto={veto_threshold} add={add_threshold}"
    )
print(
    "Verified full-prior DAgger checkpoint: "
    f"records={summary['records']} best_epoch={summary['best_epoch']} "
    f"veto={veto_threshold:.3f} add={add_threshold:.3f}"
)
PY

echo "[stop-dagger] complete: $OUTPUT_DIR/latest.pth"
