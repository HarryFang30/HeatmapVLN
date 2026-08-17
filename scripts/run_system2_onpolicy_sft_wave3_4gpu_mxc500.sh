#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$SCRIPT_DIR/.." && pwd)"
FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
export FJL_ROOT

BASE="$FJL_ROOT/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth"
REPORT="$FJL_ROOT/model/output_system2_stop_onpolicy_wave3_topup_3gpu_177eps/merged_rollout_report.json"
OUT="${SYSTEM2_ONPOLICY_OUTPUT_DIR:-$FJL_ROOT/model/output_system2_onpolicy_lora_wave3_ddp4_globalb8_2000step}"
LOG_FILE="${LOG_FILE:-$CODE/logs/system2_onpolicy_lora_wave3_ddp4_globalb8_2000step.log}"

expect_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 1
  fi
}

for required_dir in \
  "$FJL_ROOT/r2r_paronamic_data" \
  "$FJL_ROOT/InternNav-Model" \
  "$FJL_ROOT/InternNav"; do
  [[ -d "$required_dir" ]] || {
    echo "Missing required directory: $required_dir" >&2
    exit 1
  }
done

expect_sha256 "$CODE/scripts/training/train_system2_onpolicy_sft.py" \
  11f09ccc8690a536f57abb24dc403c637be7832afae4cc2f55375ad9a45cf095
expect_sha256 "$CODE/scripts/run_system2_onpolicy_sft_smoke_mxc500.sh" \
  b73683b5c59254cad17a4b870b2d8a93c475c72ed85d6a2da2f5e9ef070059f3
expect_sha256 "$CODE/scripts/run_system2_onpolicy_sft_full_mxc500.sh" \
  4b520401ee306863be950c29a4487217a7c41a3e3b15ceff4411556d4846140d
expect_sha256 "$CODE/scripts/training/select_system2_onpolicy_checkpoint.py" \
  824f8e556ce204fb10b6a5f32f8e3f477ff5580070291f0021ad37b01735d547
expect_sha256 "$CODE/src/data/stop_rollout_dataset.py" \
  8ba84525a3e82e01bcb4230b1db08934b2fc29152fe02558ca5382e89cbf0651
expect_sha256 "$BASE" \
  a56558d61869c2143bce02ed7ef00bc980b66c9520d07b959ae4fc729949ed16
expect_sha256 "$REPORT" \
  8301dc6e6e30ac4b0ea9fccfbb660cfd7938a3c33cb280f5ffa2bb31c45d9bfe

if [[ -e "$OUT" ]]; then
  echo "Refusing to reuse formal System2 continuation output: $OUT" >&2
  exit 1
fi

cd "$CODE"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG_FILE")"

# Keep the machine/runtime contract aligned with the successful Task39 jobs.
export CONDA_INIT_SH="${CONDA_INIT_SH:-/opt/conda/etc/profile.d/conda.sh}"
export QWEN25_PYTHON="${QWEN25_PYTHON:-$FJL_ROOT/envs/qwen25/bin/python}"
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1

export PANORAMIC_DATA_ROOT="$FJL_ROOT/r2r_paronamic_data"
export INTERNNAV_MODEL_PATH="$FJL_ROOT/InternNav-Model"
export INTERNNAV_BACKBONE="$INTERNNAV_MODEL_PATH"
export INTERNNAV_REPO="$FJL_ROOT/InternNav"

export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export MASTER_PORT="${MASTER_PORT:-29624}"
IFS=',' read -r -a GPU_DEVICE_ARRAY <<< "$GPU_DEVICES"
if [[ "${#GPU_DEVICE_ARRAY[@]}" -ne 4 ]]; then
  echo "Formal wave3 launcher requires exactly four GPU IDs: $GPU_DEVICES" >&2
  exit 1
fi
export SYSTEM2_ONPOLICY_GPU_DEVICES="$GPU_DEVICES"
export SYSTEM2_ONPOLICY_NPROC_PER_NODE="$NPROC_PER_NODE"
export SYSTEM2_ONPOLICY_MASTER_ADDR="$MASTER_ADDR"
export SYSTEM2_ONPOLICY_MASTER_PORT="$MASTER_PORT"

export SYSTEM2_ONPOLICY_CONFIG=configs/train_stage3_pano_system1_h1024_8gpu.yaml
export SYSTEM2_ONPOLICY_BASE_CHECKPOINT="$BASE"
export SYSTEM2_ONPOLICY_ROLLOUT_REPORT="$REPORT"
export SYSTEM2_ONPOLICY_OUTPUT_DIR="$OUT"
export SYSTEM2_ONPOLICY_DRY_RUN=0
export SYSTEM2_ONPOLICY_MAX_CLIPS=0
export SYSTEM2_ONPOLICY_MAX_STEPS=2000

# Four ranks x two samples preserves the validated global batch of eight.
export SYSTEM2_ONPOLICY_BATCH_SIZE=2
export SYSTEM2_ONPOLICY_GRAD_ACCUM_STEPS=1
export SYSTEM2_ONPOLICY_NUM_WORKERS=16
export SYSTEM2_ONPOLICY_PREFETCH_FACTOR=4

export SYSTEM2_ONPOLICY_LEARNING_RATE=1e-5
export SYSTEM2_ONPOLICY_MIN_LEARNING_RATE=2e-6
export SYSTEM2_ONPOLICY_L2_SP_WEIGHT=1.0
export SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_WEIGHT=0.25
export SYSTEM2_ONPOLICY_PAIRWISE_STOP_MARGIN_GAP=1.0

export SYSTEM2_ONPOLICY_NATIVE_SLOTS=14
export SYSTEM2_ONPOLICY_POSITIVE_SLOTS=3
export SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_SLOTS=1
export SYSTEM2_ONPOLICY_FALSE_STOP_NEGATIVE_SLOTS=2
export SYSTEM2_ONPOLICY_REGULAR_NEGATIVE_MIN_STOP_LOG_ODDS=-10

export SYSTEM2_ONPOLICY_VALIDATION_INTERVAL=100
export SYSTEM2_ONPOLICY_MAX_VALIDATION_SAMPLES=384
export SYSTEM2_ONPOLICY_MAX_TRAIN_EVALUATION_SAMPLES=192
export SYSTEM2_ONPOLICY_SAVE_VALIDATION_CHECKPOINTS=1
export SYSTEM2_ONPOLICY_MIN_TRAIN_FALSE_STOPS=200
export SYSTEM2_ONPOLICY_MIN_VALIDATION_FALSE_STOPS=40
export SYSTEM2_ONPOLICY_MIN_TRAIN_FALSE_SCENES=40

unset STAGE_DRY_RUN
unset PIPELINE_DRY_RUN
unset SYSTEM2_ONPOLICY_GPU

echo "Verified formal System2 wave3 inputs and code"
echo "code=$CODE"
echo "gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE global_batch=8"
echo "base=$BASE"
echo "report=$REPORT"
echo "output=$OUT"
echo "log=$LOG_FILE"

bash scripts/run_system2_onpolicy_sft_full_mxc500.sh 2>&1 | tee "$LOG_FILE"
