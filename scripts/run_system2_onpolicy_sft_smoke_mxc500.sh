#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
RUN_STAMP="${SYSTEM2_ONPOLICY_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

CONFIG="${SYSTEM2_ONPOLICY_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
BASE_CHECKPOINT="${SYSTEM2_ONPOLICY_BASE_CHECKPOINT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
ROLLOUT_REPORT="${SYSTEM2_ONPOLICY_ROLLOUT_REPORT:-${FJL_ROOT}/model/system2_stop_multimodal_onpolicy_expanded_validation_20260721.json}"

GPU_DEVICES="${SYSTEM2_ONPOLICY_GPU_DEVICES:-${SYSTEM2_ONPOLICY_GPU:-0}}"
IFS=',' read -r -a GPU_DEVICE_ARRAY <<< "$GPU_DEVICES"
NPROC_PER_NODE="${SYSTEM2_ONPOLICY_NPROC_PER_NODE:-${#GPU_DEVICE_ARRAY[@]}}"
MASTER_ADDR="${SYSTEM2_ONPOLICY_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${SYSTEM2_ONPOLICY_MASTER_PORT:-29624}"
GPU_TAG="${GPU_DEVICES//,/_}"
OUTPUT_DIR="${SYSTEM2_ONPOLICY_OUTPUT_DIR:-${FJL_ROOT}/model/smoke_system2_onpolicy_sft_gpu${GPU_TAG}_256clips_800step_${RUN_STAMP}}"
MAX_CLIPS="${SYSTEM2_ONPOLICY_MAX_CLIPS:-256}"
MAX_STEPS="${SYSTEM2_ONPOLICY_MAX_STEPS:-800}"
BATCH_SIZE="${SYSTEM2_ONPOLICY_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${SYSTEM2_ONPOLICY_GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${SYSTEM2_ONPOLICY_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${SYSTEM2_ONPOLICY_PREFETCH_FACTOR:-4}"
VALIDATION_INTERVAL="${SYSTEM2_ONPOLICY_VALIDATION_INTERVAL:-100}"
SAVE_VALIDATION_CHECKPOINTS="${SYSTEM2_ONPOLICY_SAVE_VALIDATION_CHECKPOINTS:-1}"
MAX_VALIDATION_SAMPLES="${SYSTEM2_ONPOLICY_MAX_VALIDATION_SAMPLES:-128}"
MAX_TRAIN_EVALUATION_SAMPLES="${SYSTEM2_ONPOLICY_MAX_TRAIN_EVALUATION_SAMPLES:-48}"
DRY_RUN="${SYSTEM2_ONPOLICY_DRY_RUN:-0}"
REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS="${SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS:-}"
LEARNING_RATE="${SYSTEM2_ONPOLICY_LEARNING_RATE:-5e-6}"
MIN_LEARNING_RATE="${SYSTEM2_ONPOLICY_MIN_LEARNING_RATE:-1e-6}"
PAIRWISE_STOP_MARGIN_WEIGHT="${SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_WEIGHT:-0}"
PAIRWISE_STOP_MARGIN_GAP="${SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_GAP:-1.0}"
L2_SP_WEIGHT="${SYSTEM2_ONPOLICY_L2_SP_WEIGHT:-1.0}"
NATIVE_SLOTS="${SYSTEM2_ONPOLICY_NATIVE_SLOTS:-14}"
POSITIVE_SLOTS="${SYSTEM2_ONPOLICY_POSITIVE_SLOTS:-3}"
REGULAR_NEGATIVE_SLOTS="${SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_SLOTS:-1}"
FALSE_STOP_NEGATIVE_SLOTS="${SYSTEM2_ONPOLICY_FALSE_STOP_NEGATIVE_SLOTS:-2}"

for required_file in "$QWEN25_PYTHON" "$CONFIG" "$BASE_CHECKPOINT" "$ROLLOUT_REPORT"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing required System2 continuation input: $required_file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite System2 continuation output: $OUTPUT_DIR" >&2
  exit 1
fi
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYSTEM2_ONPOLICY_NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi
if [[ "${#GPU_DEVICE_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "GPU count (${#GPU_DEVICE_ARRAY[@]}) must equal nproc_per_node ($NPROC_PER_NODE)" >&2
  exit 1
fi
SEEN_GPUS=","
for gpu in "${GPU_DEVICE_ARRAY[@]}"; do
  if ! [[ "$gpu" =~ ^[0-7]$ ]]; then
    echo "Every SYSTEM2_ONPOLICY_GPU_DEVICES entry must be in [0, 7]: $GPU_DEVICES" >&2
    exit 1
  fi
  if [[ "$SEEN_GPUS" == *",$gpu,"* ]]; then
    echo "SYSTEM2_ONPOLICY_GPU_DEVICES contains duplicate GPU $gpu" >&2
    exit 1
  fi
  SEEN_GPUS+="$gpu,"
done

"$QWEN25_PYTHON" - "$ROLLOUT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
roots = report.get("roots")
if report.get("status") != "passed":
    raise SystemExit(f"Rollout report did not pass: {path}")
if not isinstance(roots, list) or len(roots) != int(report.get("root_count", -1)):
    raise SystemExit(f"Invalid rollout report root contract: {path}")
print(
    "Validated rollout report: "
    f"roots={len(roots)} rows={report.get('rows')} "
    f"train_false={report.get('train_policy_counts', {}).get('false_stop_negative')} "
    f"validation_false={report.get('validation_policy_counts', {}).get('false_stop_negative')}"
)
PY

export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-${FJL_ROOT}/r2r_paronamic_data}"
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${FJL_ROOT}/InternNav-Model}"
export INTERNNAV_REPO="${INTERNNAV_REPO:-${FJL_ROOT}/InternNav}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export TOKENIZERS_PARALLELISM=false
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${MACA_HOME}/ompi/lib:${MACA_HOME}/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for required_dir in "$PANORAMIC_DATA_ROOT" "$INTERNNAV_MODEL_PATH" "$INTERNNAV_REPO"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Missing required System2 continuation directory: $required_dir" >&2
    exit 1
  fi
done

args=(
  scripts/training/train_system2_onpolicy_sft.py
  --config "$CONFIG"
  --base-checkpoint "$BASE_CHECKPOINT"
  --rollout-report "$ROLLOUT_REPORT"
  --dataset-root "$PANORAMIC_DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --device cuda
  --max-clips "$MAX_CLIPS"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --learning-rate "$LEARNING_RATE"
  --min-learning-rate "$MIN_LEARNING_RATE"
  --weight-decay 0
  --warmup-ratio 0.05
  --grad-clip 1.0
  --l2-sp-weight "$L2_SP_WEIGHT"
  --pairwise-stop-margin-weight "$PAIRWISE_STOP_MARGIN_WEIGHT"
  --pairwise-stop-margin-gap "$PAIRWISE_STOP_MARGIN_GAP"
  --native-slots "$NATIVE_SLOTS"
  --onpolicy-positive-slots "$POSITIVE_SLOTS"
  --onpolicy-regular-negative-slots "$REGULAR_NEGATIVE_SLOTS"
  --onpolicy-false-stop-negative-slots "$FALSE_STOP_NEGATIVE_SLOTS"
  --holdout-scene-fraction 0.2
  --max-validation-samples "$MAX_VALIDATION_SAMPLES"
  --max-train-evaluation-samples "$MAX_TRAIN_EVALUATION_SAMPLES"
  --validation-interval "$VALIDATION_INTERVAL"
  --seed 20260720
  --log-interval 10
)
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
elif [[ "$DRY_RUN" != "0" ]]; then
  echo "SYSTEM2_ONPOLICY_DRY_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ -n "$REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS" ]]; then
  args+=(
    --regular-negative-min-stop-log-odds
    "$REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS"
  )
fi
if [[ "$SAVE_VALIDATION_CHECKPOINTS" == "1" ]]; then
  args+=(--save-validation-checkpoints)
elif [[ "$SAVE_VALIDATION_CHECKPOINTS" != "0" ]]; then
  echo "SYSTEM2_ONPOLICY_SAVE_VALIDATION_CHECKPOINTS must be 0 or 1" >&2
  exit 1
fi

echo "[$(date '+%F %T')] System2 on-policy all-layer LoRA continuation"
echo "[system2-onpolicy] gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"
echo "[system2-onpolicy] max_clips=$MAX_CLIPS max_steps=$MAX_STEPS batch_per_rank=$BATCH_SIZE global_batch=$((BATCH_SIZE * NPROC_PER_NODE)) grad_accum=$GRAD_ACCUM_STEPS"
echo "[system2-onpolicy] lr=$LEARNING_RATE min_lr=$MIN_LEARNING_RATE"
echo "[system2-onpolicy] pairwise_stop_margin_weight=$PAIRWISE_STOP_MARGIN_WEIGHT gap=$PAIRWISE_STOP_MARGIN_GAP"
echo "[system2-onpolicy] l2_sp_weight=$L2_SP_WEIGHT slots=native:$NATIVE_SLOTS,positive:$POSITIVE_SLOTS,regular_negative:$REGULAR_NEGATIVE_SLOTS,false_stop_negative:$FALSE_STOP_NEGATIVE_SLOTS"
echo "[system2-onpolicy] base=$BASE_CHECKPOINT"
echo "[system2-onpolicy] report=$ROLLOUT_REPORT"
echo "[system2-onpolicy] output=$OUTPUT_DIR"
echo "[system2-onpolicy] regular_negative_min_stop_log_odds=${REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS:-disabled}"
echo "[system2-onpolicy] save_validation_checkpoints=$SAVE_VALIDATION_CHECKPOINTS"
common_env=(
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  PYTHONUNBUFFERED=1
  USE_TF=0
  TRANSFORMERS_NO_TF=1
  TF_CPP_MIN_LOG_LEVEL=3
  HEATMAPVLN_REQUIRE_FLASH_ATTN=1
)
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  exec env "${common_env[@]}" "$QWEN25_PYTHON" -u "${args[@]}"
fi

TORCHRUN="${SYSTEM2_ONPOLICY_TORCHRUN:-$(dirname "$QWEN25_PYTHON")/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  echo "Missing qwen25 torchrun executable: $TORCHRUN" >&2
  exit 1
fi
exec env "${common_env[@]}" "$TORCHRUN" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${args[@]}"
