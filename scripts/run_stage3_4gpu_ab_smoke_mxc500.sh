#!/usr/bin/env bash
# Four-GPU Stage3 baseline/optimized throughput smoke for MXC500.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export GPU_DEVICES="${STAGE3_AB_GPU_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${STAGE3_AB_NPROC_PER_NODE:-4}"

export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data}"
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_REPO="${INTERNNAV_REPO:-/mnt/afs/lixiaoou/intern/fjl/InternNav}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"

export STAGE3_CONFIG="${STAGE3_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_BASE_CKPT="${STAGE3_BASE_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
# Same h1024 adapter architecture as the formal Stage2 output. Override this
# with STAGE3_ADAPTER_CKPT if a newer smoke/formal adapter is preferred.
export STAGE3_ADAPTER_CKPT="${STAGE3_ADAPTER_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/smoke_stage2_prefetch_fix_detached_4gpu_256_20260709_095659/latest.pth}"

export STAGE3_EPOCHS=1
export STAGE3_BATCH_SIZE=8
export STAGE3_GRAD_ACCUM_STEPS=1
export STAGE3_PANO_ADAPTER_LR=5e-5
export STAGE3_NUM_WORKERS="${STAGE3_AB_NUM_WORKERS:-16}"
export STAGE3_PREFETCH_FACTOR="${STAGE3_AB_PREFETCH_FACTOR:-4}"
export STAGE3_PIN_MEMORY=1
export STAGE3_SHM_BYPASS=auto
export STAGE3_ENABLE_TIMING=1
export STAGE3_SHOW_GPU_MEMORY=1
export STAGE3_LOG_INTERVAL=20
export STAGE3_TENSORBOARD_INTERVAL=20
export STAGE3_PAGE_CACHE_DROP_ENABLED=0
export STAGE3_SYSTEM2_SAMPLE_STEP=1
export STAGE3_MAX_CLIPS="${STAGE3_AB_MAX_CLIPS:-64}"
export STAGE3_MAX_BATCHES="${STAGE3_AB_MAX_BATCHES:-20}"
export STAGE3_DRY_RUN=0
export STAGE_DRY_RUN=0
export STAGE3_REQUIRE_FLASH_ATTN=1
export KEEP_TMP_CONFIGS=0

BASE_PORT="${STAGE3_AB_MASTER_PORT:-29630}"
STAMP="${STAGE3_AB_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="${STAGE3_AB_MODEL_ROOT:-/mnt/afs/lixiaoou/intern/fjl/model}"
TB_ROOT="${STAGE3_AB_TB_ROOT:-/mnt/afs/lixiaoou/intern/fjl/tensorlog}"
LOG_ROOT="${STAGE3_AB_LOG_ROOT:-$REPO_ROOT/logs}"
RUN_BASELINE="${STAGE3_AB_RUN_BASELINE:-1}"

mkdir -p "$MODEL_ROOT" "$TB_ROOT" "$LOG_ROOT"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required non-empty file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file "$STAGE3_CONFIG"
require_file "$STAGE3_BASE_CKPT"
require_file "$STAGE3_ADAPTER_CKPT"
require_dir "$PANORAMIC_DATA_ROOT"
require_dir "$INTERNNAV_MODEL_PATH"

echo "[stage3-ab] commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[stage3-ab] gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE batch_per_rank=$STAGE3_BATCH_SIZE"
echo "[stage3-ab] max_clips=$STAGE3_MAX_CLIPS max_batches=$STAGE3_MAX_BATCHES"
echo "[stage3-ab] base=$STAGE3_BASE_CKPT"
echo "[stage3-ab] adapter=$STAGE3_ADAPTER_CKPT"

run_variant() {
  local name="$1"
  local merge_lora="$2"
  local inference_mode="$3"
  local last_hidden_only="$4"
  local port="$5"

  export MASTER_PORT="$port"
  export MASTER_PORT_STAGE3="$port"
  export STAGE3_MERGE_FROZEN_LORA="$merge_lora"
  export STAGE3_FROZEN_TRAJ_INFERENCE_MODE="$inference_mode"
  export STAGE3_TRAJ_LAST_HIDDEN_STATE_ONLY="$last_hidden_only"

  export STAGE3_OUT_DIR="$MODEL_ROOT/smoke_stage3_4gpu_ab_${STAMP}_${name}"
  export STAGE3_TB_DIR="$TB_ROOT/smoke_stage3_4gpu_ab_${STAMP}_${name}"
  export LOG_FILE="$LOG_ROOT/smoke_stage3_4gpu_ab_${STAMP}_${name}.log"
  local outer_log="$LOG_ROOT/smoke_stage3_4gpu_ab_${STAMP}_${name}.outer.log"
  local monitor_csv="$LOG_ROOT/smoke_stage3_4gpu_ab_${STAMP}_${name}.mx.csv"
  local monitor_pid=""

  rm -f "$monitor_csv"
  if command -v mx-smi >/dev/null 2>&1; then
    mx-smi -i "$GPU_DEVICES" --show-usage --show-memory -l 1000 \
      -o "$monitor_csv" >/tmp/heatmapvln_stage3_${name}_mx_smi.log 2>&1 &
    monitor_pid=$!
  fi

  echo "[$(date '+%F %T')] START $name port=$port out=$STAGE3_OUT_DIR"
  local rc=0
  if bash scripts/run_stage3_pano_system1_h1024_8gpu_mxc500_launcher.sh \
      >"$outer_log" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED $name rc=$rc outer=$outer_log" >&2
    tail -100 "$outer_log" >&2 || true
    return "$rc"
  fi

  local checkpoint="$STAGE3_OUT_DIR/latest/checkpoints/latest.pth"
  require_file "$checkpoint"
  if [[ "$name" == "optimized" ]]; then
    if ! grep -Fq \
      "Stage3 frozen-Qwen execution: merge_lora=False inference_mode=True last_hidden_state_only=True" \
      "$LOG_FILE"; then
      echo "Optimized run did not confirm the expected frozen-Qwen execution flags" >&2
      return 1
    fi
  fi

  echo "[$(date '+%F %T')] PASSED $name checkpoint=$checkpoint"
  echo "[stage3-ab] key $name lines:"
  grep -E "Verified complete frozen InternNav System1|Verified complete LoRA|frozen-Qwen execution|Trainable params|Batch (1|2|3|20)/|Train Loss" \
    "$LOG_FILE" | tail -30 || true
}

if is_truthy "$RUN_BASELINE"; then
  run_variant baseline 0 0 0 "$BASE_PORT"
fi
run_variant optimized 0 1 1 "$((BASE_PORT + 1))"

echo "[stage3-ab] COMPLETE tag=$STAMP"
echo "[stage3-ab] logs: $LOG_ROOT/smoke_stage3_4gpu_ab_${STAMP}_*.log"
echo "[stage3-ab] monitors: $LOG_ROOT/smoke_stage3_4gpu_ab_${STAMP}_*.mx.csv"
