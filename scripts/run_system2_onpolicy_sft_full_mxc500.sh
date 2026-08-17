#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
RUN_STAMP="${SYSTEM2_ONPOLICY_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

REPORT="${SYSTEM2_ONPOLICY_ROLLOUT_REPORT:?Set SYSTEM2_ONPOLICY_ROLLOUT_REPORT to the validated merged report}"
OUTPUT_DIR="${SYSTEM2_ONPOLICY_OUTPUT_DIR:-${FJL_ROOT}/model/output_system2_onpolicy_lora_full_${RUN_STAMP}}"
MIN_TRAIN_FALSE_STOPS="${SYSTEM2_ONPOLICY_MIN_TRAIN_FALSE_STOPS:-200}"
MIN_VALIDATION_FALSE_STOPS="${SYSTEM2_ONPOLICY_MIN_VALIDATION_FALSE_STOPS:-40}"
MIN_TRAIN_FALSE_SCENES="${SYSTEM2_ONPOLICY_MIN_TRAIN_FALSE_SCENES:-40}"

for required in "$QWEN25_PYTHON" "$REPORT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required full continuation input: $required" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite full continuation output: $OUTPUT_DIR" >&2
  exit 1
fi

"$QWEN25_PYTHON" - "$REPORT" "$MIN_TRAIN_FALSE_STOPS" "$MIN_VALIDATION_FALSE_STOPS" "$MIN_TRAIN_FALSE_SCENES" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
minimum_train_false = int(sys.argv[2])
minimum_validation_false = int(sys.argv[3])
minimum_train_scenes = int(sys.argv[4])
train_false = int(report.get("train_policy_counts", {}).get("false_stop_negative", 0))
validation_false = int(
    report.get("validation_policy_counts", {}).get("false_stop_negative", 0)
)
train_scenes = int(report.get("train_false_stop_scenes", 0))
if report.get("status") != "passed":
    raise SystemExit("Rollout report did not pass validation")
if train_false < minimum_train_false:
    raise SystemExit(f"train false-STOP rows {train_false} < {minimum_train_false}")
if validation_false < minimum_validation_false:
    raise SystemExit(
        f"validation false-STOP rows {validation_false} < {minimum_validation_false}"
    )
if train_scenes < minimum_train_scenes:
    raise SystemExit(f"train false-STOP scenes {train_scenes} < {minimum_train_scenes}")
print(
    "Full continuation data preflight passed: "
    f"train_false={train_false} validation_false={validation_false} "
    f"train_false_scenes={train_scenes}"
)
PY

export SYSTEM2_ONPOLICY_OUTPUT_DIR="$OUTPUT_DIR"
export SYSTEM2_ONPOLICY_ROLLOUT_REPORT="$REPORT"
export SYSTEM2_ONPOLICY_DRY_RUN=0
export SYSTEM2_ONPOLICY_MAX_CLIPS="${SYSTEM2_ONPOLICY_MAX_CLIPS:-0}"
export SYSTEM2_ONPOLICY_MAX_STEPS="${SYSTEM2_ONPOLICY_MAX_STEPS:-2000}"
export SYSTEM2_ONPOLICY_BATCH_SIZE="${SYSTEM2_ONPOLICY_BATCH_SIZE:-8}"
export SYSTEM2_ONPOLICY_GRAD_ACCUM_STEPS="${SYSTEM2_ONPOLICY_GRAD_ACCUM_STEPS:-1}"
export SYSTEM2_ONPOLICY_NUM_WORKERS="${SYSTEM2_ONPOLICY_NUM_WORKERS:-16}"
export SYSTEM2_ONPOLICY_PREFETCH_FACTOR="${SYSTEM2_ONPOLICY_PREFETCH_FACTOR:-4}"
export SYSTEM2_ONPOLICY_LEARNING_RATE="${SYSTEM2_ONPOLICY_LEARNING_RATE:-1e-5}"
export SYSTEM2_ONPOLICY_MIN_LEARNING_RATE="${SYSTEM2_ONPOLICY_MIN_LEARNING_RATE:-2e-6}"
export SYSTEM2_ONPOLICY_L2_SP_WEIGHT="${SYSTEM2_ONPOLICY_L2_SP_WEIGHT:-1.0}"
export SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_WEIGHT="${SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_WEIGHT:-0.25}"
export SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_GAP="${SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_GAP:-1.0}"
export SYSTEM2_ONPOLICY_NATIVE_SLOTS="${SYSTEM2_ONPOLICY_NATIVE_SLOTS:-14}"
export SYSTEM2_ONPOLICY_POSITIVE_SLOTS="${SYSTEM2_ONPOLICY_POSITIVE_SLOTS:-3}"
export SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_SLOTS="${SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_SLOTS:-1}"
export SYSTEM2_ONPOLICY_FALSE_STOP_NEGATIVE_SLOTS="${SYSTEM2_ONPOLICY_FALSE_STOP_NEGATIVE_SLOTS:-2}"
export SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS="${SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS:--10}"
export SYSTEM2_ONPOLICY_VALIDATION_INTERVAL="${SYSTEM2_ONPOLICY_VALIDATION_INTERVAL:-100}"
export SYSTEM2_ONPOLICY_MAX_VALIDATION_SAMPLES="${SYSTEM2_ONPOLICY_MAX_VALIDATION_SAMPLES:-384}"
export SYSTEM2_ONPOLICY_MAX_TRAIN_EVALUATION_SAMPLES="${SYSTEM2_ONPOLICY_MAX_TRAIN_EVALUATION_SAMPLES:-192}"
export SYSTEM2_ONPOLICY_SAVE_VALIDATION_CHECKPOINTS=1

bash scripts/run_system2_onpolicy_sft_smoke_mxc500.sh

"$QWEN25_PYTHON" scripts/training/select_system2_onpolicy_checkpoint.py \
  --output-dir "$OUTPUT_DIR"

echo "[system2-onpolicy-full] READY selected=$OUTPUT_DIR/selected.pth"
