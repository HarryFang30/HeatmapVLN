#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
INIT_CHECKPOINT="${SYSTEM2_STOP_OOF_INIT_CHECKPOINT:-${FJL_ROOT}/model/output_system2_stop_head_full_11000_alllora_h1024/latest/checkpoints/epoch_001.pth}"
ORACLE_PATH_ROOT="${SYSTEM2_STOP_ORACLE_PATH_ROOT:-${FJL_ROOT}/model/output_system2_stop_oracle_path_q25_train_all60_20260718}"
OUTPUT_DIR="${SYSTEM2_STOP_OOF_OUT_DIR:-${FJL_ROOT}/model/output_system2_stop_head_oracle_path_add_scene_oof_seed123}"
GPU_DEVICE="${SYSTEM2_STOP_OOF_GPU:-7}"

EPOCHS="${SYSTEM2_STOP_OOF_EPOCHS:-15}"
BATCH_SIZE="${SYSTEM2_STOP_OOF_BATCH_SIZE:-256}"
LEARNING_RATE="${SYSTEM2_STOP_OOF_LR:-5e-6}"
FEATURE_LOAD_WORKERS="${SYSTEM2_STOP_OOF_FEATURE_LOAD_WORKERS:-32}"
FOLDS="${SYSTEM2_STOP_OOF_FOLDS:-5}"
CONFIRMATIONS="${SYSTEM2_STOP_OOF_CONFIRMATIONS:-2}"
TERMINAL_CONFIRMATIONS="${SYSTEM2_STOP_OOF_TERMINAL_CONFIRMATIONS:-4}"
BOUNDARY_NEGATIVE_MIN_DISTANCE_M="${SYSTEM2_STOP_OOF_BOUNDARY_NEGATIVE_MIN_DISTANCE_M:-}"
BOUNDARY_NEGATIVE_MAX_DISTANCE_M="${SYSTEM2_STOP_OOF_BOUNDARY_NEGATIVE_MAX_DISTANCE_M:-}"
BOUNDARY_NEGATIVE_WEIGHT="${SYSTEM2_STOP_OOF_BOUNDARY_NEGATIVE_WEIGHT:-1}"
OPTIMIZATION_POSITIVE_RADIUS_M="${SYSTEM2_STOP_OOF_OPTIMIZATION_POSITIVE_RADIUS_M:-}"
OPTIMIZATION_NEGATIVE_RADIUS_M="${SYSTEM2_STOP_OOF_OPTIMIZATION_NEGATIVE_RADIUS_M:-}"
PROBE_LABELS="${SYSTEM2_STOP_OOF_PROBE_LABELS:-}"
CROSSFIT_PROBE_LABELS="${SYSTEM2_STOP_OOF_CROSSFIT_PROBE_LABELS:-}"

RECOVERY_ROOT="${SYSTEM2_STOP_RECOVERY_ROOT:-${FJL_ROOT}/model/output_system2_stop_oracle_recovery_seeded_near17_20260717}"
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
  "${ORACLE_PATH_ROOT}/system2_stop_rollout_labels.jsonl"
)

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "Missing leakage-free initialized STOP head: $INIT_CHECKPOINT" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite OOF STOP-add output: $OUTPUT_DIR" >&2
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

PROBE_LABEL_FILES=()
if [[ -n "$PROBE_LABELS" ]]; then
  IFS=':' read -r -a PROBE_LABEL_FILES <<< "$PROBE_LABELS"
fi
probe_label_args=()
for labels_file in "${PROBE_LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP probe labels: $labels_file" >&2
    exit 1
  fi
  probe_label_args+=(--probe-labels-jsonl "$labels_file")
done

CROSSFIT_PROBE_LABEL_FILES=()
if [[ -n "$CROSSFIT_PROBE_LABELS" ]]; then
  IFS=':' read -r -a CROSSFIT_PROBE_LABEL_FILES <<< "$CROSSFIT_PROBE_LABELS"
fi
crossfit_probe_label_args=()
for labels_file in "${CROSSFIT_PROBE_LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP cross-fit probe labels: $labels_file" >&2
    exit 1
  fi
  crossfit_probe_label_args+=(--crossfit-probe-labels-jsonl "$labels_file")
done

boundary_negative_args=()
if [[ -n "$BOUNDARY_NEGATIVE_MIN_DISTANCE_M" || -n "$BOUNDARY_NEGATIVE_MAX_DISTANCE_M" ]]; then
  if [[ -z "$BOUNDARY_NEGATIVE_MIN_DISTANCE_M" || -z "$BOUNDARY_NEGATIVE_MAX_DISTANCE_M" ]]; then
    echo "Boundary-negative min/max distances must be set together" >&2
    exit 1
  fi
  boundary_negative_args+=(
    --boundary-negative-min-distance-m "$BOUNDARY_NEGATIVE_MIN_DISTANCE_M"
    --boundary-negative-max-distance-m "$BOUNDARY_NEGATIVE_MAX_DISTANCE_M"
    --boundary-negative-weight "$BOUNDARY_NEGATIVE_WEIGHT"
  )
fi

optimization_radius_args=()
if [[ -n "$OPTIMIZATION_POSITIVE_RADIUS_M" || -n "$OPTIMIZATION_NEGATIVE_RADIUS_M" ]]; then
  if [[ -z "$OPTIMIZATION_POSITIVE_RADIUS_M" || -z "$OPTIMIZATION_NEGATIVE_RADIUS_M" ]]; then
    echo "Optimization positive/negative radii must be set together" >&2
    exit 1
  fi
  optimization_radius_args+=(
    --optimization-positive-radius-m "$OPTIMIZATION_POSITIVE_RADIUS_M"
    --optimization-negative-radius-m "$OPTIMIZATION_NEGATIVE_RADIUS_M"
  )
fi

echo "[stop-add-oof] init=$INIT_CHECKPOINT"
echo "[stop-add-oof] oracle_path=$ORACLE_PATH_ROOT"
echo "[stop-add-oof] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"
echo "[stop-add-oof] external_probe_labels=${#PROBE_LABEL_FILES[@]}"
echo "[stop-add-oof] crossfit_probe_labels=${#CROSSFIT_PROBE_LABEL_FILES[@]}"
echo "[stop-add-oof] folds=$FOLDS epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE confirmations=$CONFIRMATIONS terminal_confirmations=$TERMINAL_CONFIRMATIONS"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_stop_head_add_scene_oof.py \
  "${label_args[@]}" \
  "${probe_label_args[@]}" \
  "${crossfit_probe_label_args[@]}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --folds "$FOLDS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --weight-decay 0.01 \
  --l2-sp-weight 1e-4 \
  --terminal-negative-weight 4 \
  --hard-negative-threshold 0.8 \
  --hard-negative-weight 8 \
  --oracle-recovery-positive-weight 4 \
  "${boundary_negative_args[@]}" \
  "${optimization_radius_args[@]}" \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --confirmations "$CONFIRMATIONS" \
  --minimum-add-threshold 0.9 \
  --terminal-confirmations "$TERMINAL_CONFIRMATIONS" \
  --minimum-terminal-confirm-threshold 0.0 \
  --seed 123 \
  --feature-load-workers "$FEATURE_LOAD_WORKERS" \
  --device cuda

"$QWEN25_PYTHON" - \
  "$OUTPUT_DIR" \
  "$INIT_CHECKPOINT" \
  "$ORACLE_PATH_ROOT" \
  "$FOLDS" \
  "$CONFIRMATIONS" \
  "$TERMINAL_CONFIRMATIONS" \
  "$PROBE_LABELS" \
  "${#PROBE_LABEL_FILES[@]}" \
  "$CROSSFIT_PROBE_LABELS" \
  "${#CROSSFIT_PROBE_LABEL_FILES[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
init_checkpoint = Path(sys.argv[2]).resolve()
oracle_path_root = Path(sys.argv[3]).resolve()
expected_folds = int(sys.argv[4])
expected_confirmations = int(sys.argv[5])
expected_terminal_confirmations = int(sys.argv[6])
expected_probe_labels = {
    Path(path).resolve() for path in sys.argv[7].split(":") if path
}
expected_probe_count = int(sys.argv[8])
expected_crossfit_probe_labels = {
    Path(path).resolve() for path in sys.argv[9].split(":") if path
}
expected_crossfit_probe_count = int(sys.argv[10])
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
checkpoint_path = output_dir / "latest.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if Path(summary["source_init_checkpoint"]).resolve() != init_checkpoint:
    raise SystemExit("OOF training used the wrong initialization checkpoint")
oof = summary["scene_oof"]
if int(oof["fold_count"]) != expected_folds or not oof["scene_disjoint"]:
    raise SystemExit("OOF result is not scene-disjoint with the requested fold count")
if int(oof["confirmations"]) != expected_confirmations:
    raise SystemExit("OOF event calibration used the wrong confirmation count")
terminal = oof["terminal_confirmation"]
if int(terminal["confirmations"]) != expected_terminal_confirmations:
    raise SystemExit("OOF terminal calibration used the wrong confirmation count")
# The RPC policy consumes add/veto thresholds; this terminal-only branch is diagnostic.
terminal_threshold = float(terminal["threshold"])
if not math.isfinite(terminal_threshold) or not 0.0 <= terminal_threshold <= 1.0:
    raise SystemExit(f"Invalid diagnostic terminal threshold: {terminal_threshold}")
if int(oof["oof_event_metrics"]["false_stop_episodes"]) != 0:
    raise SystemExit("OOF STOP-add calibration has false-stop episodes")
if int(oof["oof_event_metrics"]["true_stop_episodes"]) < 1:
    raise SystemExit("OOF STOP-add calibration has zero true-stop episodes")
if not bool(oof["deployable"]):
    raise SystemExit(f"OOF STOP-add quality gate failed: {oof['quality_gate']}")
threshold = float(summary["add_stop_threshold"])
if not math.isfinite(threshold) or not 0.9 <= threshold <= 1.0:
    raise SystemExit(f"Invalid OOF add threshold: {threshold}")
training = checkpoint["config"]["rollout_stop_training"]
label_paths = {Path(path).resolve() for path in training["labels_jsonl"]}
expected_new_labels = oracle_path_root / "system2_stop_rollout_labels.jsonl"
if expected_new_labels not in label_paths:
    raise SystemExit("OOF checkpoint provenance omits the oracle-path labels")
probe_label_paths = {
    Path(path).resolve() for path in training.get("probe_labels_jsonl", [])
}
if len(expected_probe_labels) != expected_probe_count:
    raise SystemExit("Duplicate or malformed probe label paths were requested")
if probe_label_paths != expected_probe_labels:
    raise SystemExit("OOF checkpoint provenance has the wrong external probe labels")
if bool(training.get("probe_rows_used_for_training", True)):
    raise SystemExit("External probe rows were not marked evaluation-only")
if expected_probe_count:
    probe_summary = oof.get("external_probe_sweeps")
    if not isinstance(probe_summary, dict):
        raise SystemExit("OOF summary omits external probe diagnostics")
    probe_scores = checkpoint.get("diagnostic_probe_oof_probabilities")
    if not torch.is_tensor(probe_scores) or not torch.isfinite(probe_scores).all():
        raise SystemExit("OOF checkpoint has invalid external probe scores")
    fold_heads = checkpoint.get("diagnostic_scene_fold_heads")
    if not isinstance(fold_heads, list) or len(fold_heads) != expected_folds:
        raise SystemExit("OOF checkpoint omits diagnostic scene-fold heads")
crossfit_probe_label_paths = {
    Path(path).resolve()
    for path in training.get("crossfit_probe_labels_jsonl", [])
}
if len(expected_crossfit_probe_labels) != expected_crossfit_probe_count:
    raise SystemExit("Duplicate or malformed cross-fit probe label paths were requested")
if crossfit_probe_label_paths != expected_crossfit_probe_labels:
    raise SystemExit("OOF checkpoint provenance has the wrong cross-fit probe labels")
if expected_crossfit_probe_count:
    if not bool(training.get("crossfit_probe_rows_used_for_training", False)):
        raise SystemExit("Cross-fit probe rows were not marked as training inputs")
    if not bool(training.get("crossfit_probe_evaluation_scene_disjoint", False)):
        raise SystemExit("Cross-fit probe diagnostics were not marked scene-disjoint")
    crossfit_summary = oof.get("crossfit_probe_sweeps")
    if not isinstance(crossfit_summary, dict):
        raise SystemExit("OOF summary omits cross-fit probe diagnostics")
    crossfit_scores = checkpoint.get("diagnostic_crossfit_probe_oof_probabilities")
    if not torch.is_tensor(crossfit_scores) or not torch.isfinite(crossfit_scores).all():
        raise SystemExit("OOF checkpoint has invalid cross-fit probe scores")
state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or not state:
    raise SystemExit("OOF STOP-add checkpoint has no state")
if not all(str(name).startswith("stop_head.") for name in state):
    raise SystemExit("OOF STOP-add checkpoint contains non-head tensors")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("OOF STOP-add checkpoint contains non-finite tensors")
print(
    "Verified scene-OOF STOP-add checkpoint: "
    f"records={summary['records']} candidates={oof['candidate_records']} "
    f"episodes={oof['candidate_episodes']} threshold={threshold:.6f} "
    f"true_stop={oof['oof_event_metrics']['true_stop_episodes']} "
    f"false_stop={oof['oof_event_metrics']['false_stop_episodes']}"
)
PY

echo "[stop-add-oof] complete: $OUTPUT_DIR/latest.pth"
