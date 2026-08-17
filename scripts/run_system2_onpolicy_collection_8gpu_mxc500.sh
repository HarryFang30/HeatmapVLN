#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
RUN_STAMP="${SYSTEM2_ONPOLICY_COLLECTION_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
BASE_REPORT="${SYSTEM2_ONPOLICY_COLLECTION_BASE_REPORT:-${FJL_ROOT}/model/system2_stop_multimodal_onpolicy_wave1_validation_20260721.json}"
BASE_CHECKPOINT="${SYSTEM2_ONPOLICY_COLLECTION_BASE_CHECKPOINT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
STAGE3_CHECKPOINT="${SYSTEM2_ONPOLICY_COLLECTION_STAGE3_CHECKPOINT:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords_priorfix/latest/checkpoints/epoch_002.pth}"
TRAIN_DATASET="${SYSTEM2_ONPOLICY_COLLECTION_TRAIN_DATASET:-${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
COLLECTION_ROOT="${SYSTEM2_ONPOLICY_COLLECTION_OUT_DIR:-${FJL_ROOT}/model/output_system2_stop_onpolicy_wave2_8gpu_${RUN_STAMP}}"
REPORT_OUTPUT="${SYSTEM2_ONPOLICY_COLLECTION_REPORT:-${COLLECTION_ROOT}/merged_rollout_report.json}"

SELECTION="${SYSTEM2_ONPOLICY_COLLECTION_SELECTION:-q75}"
SEED="${SYSTEM2_ONPOLICY_COLLECTION_SEED:-20260722}"
RPC_PORT_BASE="${SYSTEM2_ONPOLICY_COLLECTION_RPC_PORT_BASE:-50120}"
DISPLAY_ADDR="${SYSTEM2_ONPOLICY_COLLECTION_DISPLAY:-localhost:200.0}"
DECODE_WORKERS="${SYSTEM2_ONPOLICY_COLLECTION_DECODE_WORKERS:-64}"
MIN_NEW_FALSE_STOPS="${SYSTEM2_ONPOLICY_COLLECTION_MIN_NEW_FALSE_STOPS:-100}"
MIN_TRAIN_FALSE_SCENES="${SYSTEM2_ONPOLICY_COLLECTION_MIN_TRAIN_FALSE_SCENES:-40}"
RESUME="${SYSTEM2_ONPOLICY_COLLECTION_RESUME:-0}"
PREFLIGHT_ONLY="${SYSTEM2_ONPOLICY_COLLECTION_PREFLIGHT_ONLY:-0}"

for required in "$QWEN25_PYTHON" "$BASE_REPORT" "$BASE_CHECKPOINT" "$STAGE3_CHECKPOINT" "$TRAIN_DATASET"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required on-policy collection input: $required" >&2
    exit 1
  fi
done
case "${RESUME,,}" in
  0|false|no|off)
    RESUME=0
    if [[ -e "$COLLECTION_ROOT" ]]; then
      echo "Refusing to overwrite collection output: $COLLECTION_ROOT" >&2
      exit 1
    fi
    if [[ -e "$REPORT_OUTPUT" ]]; then
      echo "Refusing to overwrite rollout report: $REPORT_OUTPUT" >&2
      exit 1
    fi
    ;;
  1|true|yes|on)
    RESUME=1
    if [[ ! -d "$COLLECTION_ROOT" ]]; then
      echo "Resume requested but collection output is missing: $COLLECTION_ROOT" >&2
      exit 1
    fi
    ;;
  *)
    echo "SYSTEM2_ONPOLICY_COLLECTION_RESUME must be boolean, got: $RESUME" >&2
    exit 1
    ;;
esac
case "${PREFLIGHT_ONLY,,}" in
  0|false|no|off) PREFLIGHT_ONLY=0 ;;
  1|true|yes|on) PREFLIGHT_ONLY=1 ;;
  *)
    echo "SYSTEM2_ONPOLICY_COLLECTION_PREFLIGHT_ONLY must be boolean, got: $PREFLIGHT_ONLY" >&2
    exit 1
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_DEVICES"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_DEVICES must contain at least one GPU" >&2
  exit 1
fi
GPU_COUNT="${#GPUS[@]}"
NUM_COHORTS="${SYSTEM2_ONPOLICY_COLLECTION_NUM_COHORTS:-$GPU_COUNT}"
if [[ ! "$NUM_COHORTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYSTEM2_ONPOLICY_COLLECTION_NUM_COHORTS must be a positive integer: $NUM_COHORTS" >&2
  exit 1
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]] || [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "GPU_DEVICES must contain unique physical ids in [0, 7]: $GPU_DEVICES" >&2
    exit 1
  fi
  SEEN_GPUS[$gpu]=1
done

COHORT_DIR="$COLLECTION_ROOT/cohorts"
LOG_DIR="$COLLECTION_ROOT/logs"
mkdir -p "$COHORT_DIR" "$LOG_DIR"

if [[ "$RESUME" -eq 0 ]]; then
  "$QWEN25_PYTHON" scripts/evaluation/build_stop_dagger_cohorts.py \
    --dataset "$TRAIN_DATASET" \
    --exclude-rollout-report "$BASE_REPORT" \
    --output-dir "$COHORT_DIR" \
    --prefix wave2 \
    --num-cohorts "$NUM_COHORTS" \
    --per-scene-per-cohort 1 \
    --seed "$SEED" \
    --selection "$SELECTION" \
    --allow-incomplete-scenes
else
  echo "[collection] resuming existing output: $COLLECTION_ROOT"
fi

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  "$QWEN25_PYTHON" - "$COHORT_DIR" "$NUM_COHORTS" <<'PY'
import glob
import json
import sys

paths = sorted(glob.glob(f"{sys.argv[1]}/wave2_*.json"))
expected = int(sys.argv[2])
if len(paths) != expected:
    raise SystemExit(f"Expected {expected} cohorts, found {len(paths)}")
rows = [json.load(open(path, encoding="utf-8"))["episodes"] for path in paths]
if any(not cohort for cohort in rows):
    raise SystemExit("Generated an empty collection cohort")
if any(len(cohort) != len({row["scene_id"] for row in cohort}) for cohort in rows):
    raise SystemExit("A collection cohort contains duplicate scenes")
keys = [
    (str(row["scene_id"]), int(row["episode_id"]))
    for cohort in rows
    for row in cohort
]
if len(keys) != len(set(keys)):
    raise SystemExit("Collection cohorts overlap")
print(
    "REAL collection preflight passed: "
    f"cohorts={len(paths)} episodes={len(keys)} "
    f"scenes_per_cohort={[len(cohort) for cohort in rows]}"
)
PY
  exit 0
fi

declare -a PIDS=()
declare -a PID_INDICES=()
declare -a OUTPUTS=()
declare -a COHORTS=()
declare -a WORKER_LOGS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

run_worker() {
    local cohort="$1"
    local output="$2"
    local gpu="$3"
    local port="$4"
    local protocol_seed="$5"
    local worker_resume="$6"
    export HEATMAPVLN_REPO_ROOT="$REPO_ROOT"
    export STAGE3_EVAL_BASE_CKPT="$BASE_CHECKPOINT"
    export STAGE3_EVAL_CHECKPOINT="$STAGE3_CHECKPOINT"
    export STAGE3_EVAL_EXPECTED_EPOCH=2
    export STAGE3_EVAL_EPISODE_LIST="$cohort"
    export STAGE3_EVAL_OUTPUT_PATH="$output"
    export STAGE3_FORCE_CONTINUE_GPU="$gpu"
    export STAGE3_FORCE_CONTINUE_RPC_PORT="$port"
    export STAGE3_FORCE_CONTINUE_DISPLAY="$DISPLAY_ADDR"
    export STAGE3_EVAL_RPC_PROTOCOL_SEED="$protocol_seed"
    export STAGE3_EVAL_RESUME="$worker_resume"
    exec bash scripts/run_system2_stop_force_continue_train3_smoke_mxc500.sh
}

for ((batch_start = 0; batch_start < NUM_COHORTS; batch_start += GPU_COUNT)); do
  batch_end=$((batch_start + GPU_COUNT))
  if ((batch_end > NUM_COHORTS)); then
    batch_end="$NUM_COHORTS"
  fi
  PIDS=()
  PID_INDICES=()
  echo "[collection] starting cohort batch [$batch_start, $batch_end) on $GPU_COUNT GPU(s)"
  for ((index = batch_start; index < batch_end; index++)); do
    gpu_slot=$((index - batch_start))
    gpu="${GPUS[$gpu_slot]}"
    printf -v cohort "%s/wave2_%02d.json" "$COHORT_DIR" "$index"
    printf -v output "%s/rollout_%02d_gpu%s" "$COLLECTION_ROOT" "$index" "$gpu"
    printf -v log "%s/rollout_%02d_gpu%s.log" "$LOG_DIR" "$index" "$gpu"
    port=$((RPC_PORT_BASE + gpu_slot))
    protocol_seed=$((SEED + index))
    if [[ ! -s "$cohort" ]]; then
      echo "Missing generated cohort: $cohort" >&2
      exit 1
    fi
    OUTPUTS[$index]="$output"
    COHORTS[$index]="$cohort"
    WORKER_LOGS[$index]="$log"
    if [[ -s "$output/result.json" ]]; then
      echo "[collection] worker $index already complete: $output"
      continue
    fi
    worker_resume=0
    if [[ -e "$output" ]]; then
      if [[ "$RESUME" -ne 1 || ! -d "$output" ]]; then
        echo "Refusing unexpected worker output: $output" >&2
        exit 1
      fi
      worker_resume=1
    fi
    echo "[collection] worker=$index gpu=$gpu cohort=$cohort port=$port output=$output"
    if [[ "$worker_resume" -eq 1 ]]; then
      run_worker "$cohort" "$output" "$gpu" "$port" "$protocol_seed" "$worker_resume" >>"$log" 2>&1 &
    else
      run_worker "$cohort" "$output" "$gpu" "$port" "$protocol_seed" "$worker_resume" >"$log" 2>&1 &
    fi
    PIDS+=("$!")
    PID_INDICES+=("$index")
  done

  failed=0
  for position in "${!PIDS[@]}"; do
    index="${PID_INDICES[$position]}"
    if wait "${PIDS[$position]}"; then
      echo "[collection] worker $index completed: ${OUTPUTS[$index]}"
    else
      echo "[collection] worker $index failed: ${WORKER_LOGS[$index]}" >&2
      failed=1
    fi
  done
  PIDS=()
  PID_INDICES=()
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one on-policy collection worker failed" >&2
    exit 1
  fi
done

validate_args=(
  scripts/training/validate_system2_onpolicy_rollouts.py
  --base-report "$BASE_REPORT"
  --output "$REPORT_OUTPUT"
  --split-seed 20260720
  --holdout-scene-fraction 0.2
  --decode-workers "$DECODE_WORKERS"
)
for ((index = 0; index < NUM_COHORTS; index++)); do
  validate_args+=(
    --rollout-root "${OUTPUTS[$index]}"
    --rollout-cohort "${COHORTS[$index]}"
  )
done
if [[ -e "$REPORT_OUTPUT" ]]; then
  if [[ "$RESUME" -ne 1 ]]; then
    echo "Refusing unexpected rollout report: $REPORT_OUTPUT" >&2
    exit 1
  fi
  validate_args+=(--overwrite)
fi
"$QWEN25_PYTHON" "${validate_args[@]}"

"$QWEN25_PYTHON" - "$BASE_REPORT" "$REPORT_OUTPUT" "$MIN_NEW_FALSE_STOPS" "$MIN_TRAIN_FALSE_SCENES" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
merged = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
minimum_new_false = int(sys.argv[3])
minimum_train_scenes = int(sys.argv[4])

def false_count(report):
    return sum(
        int(report.get(split, {}).get("false_stop_negative", 0))
        for split in ("train_policy_counts", "validation_policy_counts")
    )

new_false = false_count(merged) - false_count(base)
train_scenes = int(merged.get("train_false_stop_scenes", 0))
if merged.get("status") != "passed":
    raise SystemExit("Merged rollout report did not pass")
if new_false < minimum_new_false:
    raise SystemExit(
        f"Insufficient new false-STOP rows: {new_false} < {minimum_new_false}"
    )
if train_scenes < minimum_train_scenes:
    raise SystemExit(
        f"Insufficient train false-STOP scene coverage: {train_scenes} < {minimum_train_scenes}"
    )
print(
    "On-policy collection postflight passed: "
    f"new_false_stops={new_false} train_false_stop_scenes={train_scenes} "
    f"report={sys.argv[2]}"
)
PY

echo "[collection] READY report=$REPORT_OUTPUT"
