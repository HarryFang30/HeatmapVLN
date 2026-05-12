#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-120}"

STAGE1_HM_CONFIG="${STAGE1_HM_CONFIG:-configs/train_heatmap_config_lora.yaml}"
STAGE1_S2_CONFIG="${STAGE1_S2_CONFIG:-configs/train_system2_panoramic_sft_2gpu.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/train_config_internnav_4gpu.yaml}"

HEATMAP_BASE_CKPT="${HEATMAP_BASE_CKPT:-/workspace/heatmap_training_outputs/run_20260407_004635/checkpoints/best.pth}"
HEATMAP_LORA_OUT_DIR="${HEATMAP_LORA_OUT_DIR:-/workspace/heatmap_lora_training_outputs}"
SYSTEM2_SFT_OUT_DIR="${SYSTEM2_SFT_OUT_DIR:-/root/autodl-tmp/vln_system2_sft_outputs}"
STAGE2_OUT_DIR="${STAGE2_OUT_DIR:-/root/autodl-tmp/vln_training_outputs}"

STAGE1_HM_EPOCHS="${STAGE1_HM_EPOCHS:-4}"
STAGE1_S2_EPOCHS="${STAGE1_S2_EPOCHS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"

MASTER_PORT_STAGE1_HM="${MASTER_PORT_STAGE1_HM:-29616}"
MASTER_PORT_STAGE1_S2="${MASTER_PORT_STAGE1_S2:-29617}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29618}"

RUN_EVAL="${RUN_EVAL:-1}"
DISPLAY_NUM="${DISPLAY_NUM:-200}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${STAGE2_OUT_DIR}/latest/eval_r2r_val_unseen}"

export FEISHU_WEBHOOK_URL="${FEISHU_WEBHOOK_URL:?FEISHU_WEBHOOK_URL is required. Export it before running this script.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "Missing required file: $path"
    exit 1
  fi
}

cooldown() {
  local seconds="$1"
  if (( seconds > 0 )); then
    log "Cooling down for ${seconds}s ..."
    sleep "$seconds"
  fi
}

choose_checkpoint() {
  local ckpt_dir="$1"
  local preference="${2:-best}"
  local first
  local second

  if [[ "$preference" == "latest" ]]; then
    first="${ckpt_dir}/latest.pth"
    second="${ckpt_dir}/best.pth"
  else
    first="${ckpt_dir}/best.pth"
    second="${ckpt_dir}/latest.pth"
  fi

  if [[ -f "$first" ]]; then
    printf '%s\n' "$first"
    return 0
  fi
  if [[ -f "$second" ]]; then
    printf '%s\n' "$second"
    return 0
  fi

  log "No checkpoint found in $ckpt_dir (looked for best.pth/latest.pth)"
  exit 1
}

run_training_stage() {
  local name="$1"
  local master_port="$2"
  shift 2

  log "Starting ${name}"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
    --master_port="$master_port" \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train.py "$@"
  log "Finished ${name}"
}

start_xvfb() {
  if ! command -v Xvfb >/dev/null 2>&1; then
    log "Xvfb not found; evaluation cannot start."
    exit 1
  fi

  if pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    log "Xvfb :${DISPLAY_NUM} is already running"
  else
    log "Starting Xvfb :${DISPLAY_NUM}"
    Xvfb ":${DISPLAY_NUM}" -screen 0 1024x768x24 >"/tmp/xvfb_${DISPLAY_NUM}.log" 2>&1 &
    sleep 2
  fi

  export DISPLAY=":${DISPLAY_NUM}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
}

run_eval() {
  local stage2_ckpt="$1"
  start_xvfb

  log "Starting R2R val_unseen evaluation"
  CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python scripts/evaluate.py r2r \
    --config "$STAGE2_CONFIG" \
    --checkpoint "$stage2_ckpt" \
    --gpu_id "$EVAL_GPU_ID" \
    --output_path "$EVAL_OUTPUT_PATH"
  log "Finished R2R val_unseen evaluation"
}

require_file "$HEATMAP_BASE_CKPT"

log "Pipeline root: $ROOT_DIR"
log "Training GPUs: $GPU_DEVICES (nproc_per_node=$NPROC_PER_NODE)"
log "Cooldown between steps: ${COOLDOWN_SECONDS}s"

run_training_stage "Stage1-HM heatmap LoRA" "$MASTER_PORT_STAGE1_HM" \
  --config "$STAGE1_HM_CONFIG" \
  --load-weights "$HEATMAP_BASE_CKPT" \
  --epochs "$STAGE1_HM_EPOCHS"

cooldown "$COOLDOWN_SECONDS"

STAGE1_HM_CKPT="$(choose_checkpoint "${HEATMAP_LORA_OUT_DIR}/latest/checkpoints" best)"
log "Stage1-HM checkpoint: $STAGE1_HM_CKPT"

run_training_stage "Stage1-S2 panoramic System2 SFT" "$MASTER_PORT_STAGE1_S2" \
  --config "$STAGE1_S2_CONFIG" \
  --load-weights "$STAGE1_HM_CKPT" \
  --epochs "$STAGE1_S2_EPOCHS"

cooldown "$COOLDOWN_SECONDS"

STAGE1_S2_CKPT="$(choose_checkpoint "${SYSTEM2_SFT_OUT_DIR}/latest/checkpoints" latest)"
log "Stage1-S2 checkpoint: $STAGE1_S2_CKPT"

run_training_stage "Stage2 bridge-only" "$MASTER_PORT_STAGE2" \
  --config "$STAGE2_CONFIG" \
  --load-weights "$STAGE1_S2_CKPT" \
  --epochs "$STAGE2_EPOCHS"

STAGE2_CKPT="$(choose_checkpoint "${STAGE2_OUT_DIR}/latest/checkpoints" latest)"
log "Stage2 checkpoint: $STAGE2_CKPT"

if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" || "$RUN_EVAL" == "yes" ]]; then
  cooldown "$COOLDOWN_SECONDS"
  run_eval "$STAGE2_CKPT"
else
  log "RUN_EVAL=$RUN_EVAL; skipping evaluation"
fi

log "Pipeline completed"
