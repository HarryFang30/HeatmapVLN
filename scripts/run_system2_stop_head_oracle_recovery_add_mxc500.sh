#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_ENV="${QWEN25_ENV:-${FJL_ROOT}/envs/qwen25}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${QWEN25_ENV}/bin/python}"
INIT_CHECKPOINT="${SYSTEM2_STOP_RECOVERY_INIT_CHECKPOINT:-${FJL_ROOT}/model/output_system2_stop_head_fullprior_dagger_all61_boundary301_seed123/latest.pth}"
RECOVERY_ROOT="${SYSTEM2_STOP_RECOVERY_ROOT:-${FJL_ROOT}/model/output_system2_stop_oracle_recovery_seeded_near17_20260717}"
OUTPUT_DIR="${SYSTEM2_STOP_RECOVERY_OUT_DIR:-${FJL_ROOT}/model/output_system2_stop_head_oracle_recovery_add_near17_seed123}"
GPU_DEVICE="${SYSTEM2_STOP_RECOVERY_GPU:-7}"

EPOCHS="${SYSTEM2_STOP_RECOVERY_EPOCHS:-15}"
BATCH_SIZE="${SYSTEM2_STOP_RECOVERY_BATCH_SIZE:-256}"
LEARNING_RATE="${SYSTEM2_STOP_RECOVERY_LR:-5e-6}"
FEATURE_LOAD_WORKERS="${SYSTEM2_STOP_RECOVERY_FEATURE_LOAD_WORKERS:-32}"
RECOVERY_POSITIVE_WEIGHT="${SYSTEM2_STOP_RECOVERY_POSITIVE_WEIGHT:-4}"

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
  "${FJL_ROOT}/model/smoke_system2_stop_bounded_oracle_recovery_train3_20260717_1954/system2_stop_rollout_labels.jsonl"
  "${RECOVERY_ROOT}/seed_42/system2_stop_rollout_labels.jsonl"
  "${RECOVERY_ROOT}/seed_118/system2_stop_rollout_labels.jsonl"
  "${RECOVERY_ROOT}/seed_119/system2_stop_rollout_labels.jsonl"
  "${RECOVERY_ROOT}/seed_121/system2_stop_rollout_labels.jsonl"
  "${RECOVERY_ROOT}/seed_122/system2_stop_rollout_labels.jsonl"
)

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "Missing initialized STOP head: $INIT_CHECKPOINT" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing recovery STOP-head output: $OUTPUT_DIR" >&2
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

echo "[stop-recovery-add] init=$INIT_CHECKPOINT"
echo "[stop-recovery-add] recovery_root=$RECOVERY_ROOT"
echo "[stop-recovery-add] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"
echo "[stop-recovery-add] epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE recovery_positive_weight=$RECOVERY_POSITIVE_WEIGHT"

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
  --oracle-recovery-positive-weight "$RECOVERY_POSITIVE_WEIGHT" \
  --feature-load-workers "$FEATURE_LOAD_WORKERS" \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --val-fraction 0.2 \
  --seed 123 \
  --training-scope all \
  --selection-objective add \
  --device cuda

"$QWEN25_PYTHON" - "$OUTPUT_DIR" "$INIT_CHECKPOINT" "$RECOVERY_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
init_checkpoint = Path(sys.argv[2]).resolve()
recovery_root = Path(sys.argv[3]).resolve()
checkpoint_path = output_dir / "latest.pth"
summary_path = output_dir / "summary.json"
if not checkpoint_path.is_file() or not summary_path.is_file():
    raise SystemExit("Recovery STOP-head output lacks latest.pth or summary.json")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary.get("records", 0)) <= 20857:
    raise SystemExit("Recovery labels did not add any deduplicated training records")
if int(summary.get("positive_records", 0)) <= 1119:
    raise SystemExit("Recovery collection did not add positive STOP records")
if summary.get("selection_objective") != "add":
    raise SystemExit("Recovery STOP head was not selected for the add objective")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("stage_name") != "system2_stop_head":
    raise SystemExit("Recovery checkpoint has the wrong stage_name")
if Path(checkpoint.get("source_init_checkpoint", "")).resolve() != init_checkpoint:
    raise SystemExit("Recovery checkpoint records the wrong initialization")
rollout_config = checkpoint["config"].get("rollout_stop_training", {})
if rollout_config.get("selection_objective") != "add":
    raise SystemExit("Checkpoint config lacks selection_objective=add")
label_paths = {Path(path).resolve() for path in rollout_config.get("labels_jsonl", [])}
expected_recovery_paths = {
    recovery_root / f"seed_{seed}" / "system2_stop_rollout_labels.jsonl"
    for seed in (42, 118, 119, 121, 122)
}
if not expected_recovery_paths.issubset(label_paths):
    raise SystemExit("Checkpoint provenance omits seeded oracle-recovery labels")
if int(rollout_config.get("oracle_recovery_positive_records", 0)) < 1:
    raise SystemExit("Training split contains no oracle-recovery positives")

state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or not state:
    raise SystemExit("Recovery checkpoint has no trainable state")
if not all(str(name).startswith("stop_head.") for name in state):
    raise SystemExit("Recovery checkpoint contains non-STOP-head tensors")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("Recovery checkpoint contains non-finite tensors")

metrics = checkpoint.get("metrics", {})
add_metrics = metrics.get("val_at_add_threshold", {})
add_threshold = float(metrics.get("add_stop_threshold", float("nan")))
if not math.isfinite(add_threshold) or not 0.9 <= add_threshold <= 1.0:
    raise SystemExit(f"Invalid add threshold: {add_threshold}")
if float(add_metrics.get("false_positive_rate", 1.0)) != 0.0:
    raise SystemExit("Recovery STOP-add threshold has validation false positives")
if float(add_metrics.get("recall", 0.0)) <= 0.0:
    raise SystemExit("Recovery STOP-add threshold has zero validation recall")

print(
    "Verified recovery STOP-add checkpoint: "
    f"records={summary['records']} positives={summary['positive_records']} "
    f"best_epoch={summary['best_epoch']} add={add_threshold:.3f} "
    f"recall={float(add_metrics['recall']):.4f} fpr=0"
)
PY

echo "[stop-recovery-add] complete: $OUTPUT_DIR/latest.pth"
