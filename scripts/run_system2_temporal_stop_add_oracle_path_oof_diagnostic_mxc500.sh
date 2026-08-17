#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
STATIC_STOP_HEAD="${SYSTEM2_TEMPORAL_ADD_STATIC_STOP_HEAD:-${FJL_ROOT}/model/output_system2_stop_head_full_11000_alllora_h1024/latest/checkpoints/epoch_001.pth}"
ORACLE_PATH_ROOT="${SYSTEM2_STOP_ORACLE_PATH_ROOT:-${FJL_ROOT}/model/output_system2_stop_oracle_path_q25_train_all60_20260718}"
OUTPUT_DIR="${SYSTEM2_TEMPORAL_ADD_OOF_OUT_DIR:-${FJL_ROOT}/model/diag_system2_temporal_stop_add_oracle_path_scene_oof_20260718}"
GPU_DEVICE="${SYSTEM2_TEMPORAL_ADD_OOF_GPU:-7}"

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
if [[ ! -s "$STATIC_STOP_HEAD" ]]; then
  echo "Missing leakage-free static STOP prior: $STATIC_STOP_HEAD" >&2
  exit 1
fi
for labels_file in "${LABEL_FILES[@]}"; do
  if [[ ! -s "$labels_file" ]]; then
    echo "Missing STOP rollout labels: $labels_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite temporal STOP-add OOF diagnostic: $OUTPUT_DIR" >&2
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

echo "[temporal-add-oof] static_prior=$STATIC_STOP_HEAD"
echo "[temporal-add-oof] oracle_path=$ORACLE_PATH_ROOT"
echo "[temporal-add-oof] output=$OUTPUT_DIR gpu=$GPU_DEVICE labels=${#LABEL_FILES[@]}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$QWEN25_PYTHON" \
  scripts/training/train_temporal_stop_add_seed_ensemble_diagnostic.py \
  "${label_args[@]}" \
  --static-stop-head-checkpoint "$STATIC_STOP_HEAD" \
  --output-dir "$OUTPUT_DIR" \
  --folds 5 \
  --members-per-fold 5 \
  --epochs 30 \
  --batch-size 256 \
  --hidden-dim 16 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --confirmations 2 \
  --feature-load-workers 32 \
  --relabel-ambiguous-negative-radius-m 3.01 \
  --seed 123 \
  --device cuda

"$QWEN25_PYTHON" - "$OUTPUT_DIR" "$ORACLE_PATH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
oracle_path_root = Path(sys.argv[2]).resolve()
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
checkpoint = torch.load(output_dir / "latest.pth", map_location="cpu", weights_only=False)
selected = summary["selected"]
event = selected["event"]
if int(event["false_stop_episodes"]) != 0:
    raise SystemExit("Temporal STOP-add OOF diagnostic has false-stop episodes")
if int(event["true_stop_episodes"]) < 1:
    raise SystemExit("Temporal STOP-add OOF diagnostic has no true-stop episodes")
if checkpoint["config"]["seed_ensemble"].get("deployable") is not False:
    raise SystemExit("Diagnostic checkpoint must remain explicitly non-deployable")
label_paths = {Path(path).resolve() for path in checkpoint["training"]["labels_jsonl"]}
if oracle_path_root / "system2_stop_rollout_labels.jsonl" not in label_paths:
    raise SystemExit("Temporal diagnostic provenance omits oracle-path labels")
state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or not state:
    raise SystemExit("Temporal diagnostic has no member state")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("Temporal diagnostic contains non-finite tensors")
print(
    "Verified temporal STOP-add OOF diagnostic: "
    f"strategy={summary['selected_strategy']} "
    f"true_stop={event['true_stop_episodes']} "
    f"false_stop={event['false_stop_episodes']} "
    f"recall={float(event['positive_episode_recall']):.4f}"
)
PY

echo "[temporal-add-oof] complete: $OUTPUT_DIR/latest.pth"
