#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage_training_common.sh"

STAGE1_HM_CONFIG="${STAGE1_HM_CONFIG:-configs/train_heatmap_config_lora.yaml}"
HEATMAP_DATA_ROOT="${HEATMAP_DATA_ROOT:-/workspace/heatmap_train_data}"
# 未设置时用默认；显式设为空字符串表示「无单独 val 根目录」（训练里 val 会回退到 train 的 root）
HEATMAP_VAL_ROOT="${HEATMAP_VAL_ROOT-/workspace/val_unseen}"
STAGE1_HM_DATA_ROOT="${STAGE1_HM_DATA_ROOT:-$HEATMAP_DATA_ROOT}"
STAGE1_HM_VAL_ROOT="${STAGE1_HM_VAL_ROOT-$HEATMAP_VAL_ROOT}"
STAGE1_HM_INIT_CKPT="${STAGE1_HM_INIT_CKPT:-${HEATMAP_BASE_CKPT:-/workspace/heatmap_training_outputs/run_20260407_004635/checkpoints/best.pth}}"
STAGE1_HM_OUT_DIR="${STAGE1_HM_OUT_DIR:-${HEATMAP_LORA_OUT_DIR:-/workspace/heatmap_lora_training_outputs}}"
STAGE1_HM_TB_DIR="${STAGE1_HM_TB_DIR:-/workspace/tf-logs-lora}"
STAGE1_HM_EPOCHS="${STAGE1_HM_EPOCHS:-}"
STAGE1_HM_BATCH_SIZE="${STAGE1_HM_BATCH_SIZE:-2}"
STAGE1_HM_GRAD_ACCUM_STEPS="${STAGE1_HM_GRAD_ACCUM_STEPS:-1}"
STAGE1_HM_NUM_WORKERS="${STAGE1_HM_NUM_WORKERS:-4}"
STAGE1_HM_PREFETCH_FACTOR="${STAGE1_HM_PREFETCH_FACTOR:-2}"
STAGE1_HM_PIN_MEMORY="${STAGE1_HM_PIN_MEMORY:-true}"
STAGE1_HM_MAX_BATCHES="${STAGE1_HM_MAX_BATCHES:-}"
STAGE1_HM_CHECKPOINT_PREFERENCE="${STAGE1_HM_CHECKPOINT_PREFERENCE:-best}"
STAGE1_HM_FEISHU_NOTIFY="${STAGE1_HM_FEISHU_NOTIFY:-$FEISHU_NOTIFY}"
# 设为 0/false 关闭验证循环（不写 val 集、不跑 validate）；默认开启
STAGE1_HM_VALIDATION_ENABLED="${STAGE1_HM_VALIDATION_ENABLED:-1}"
MASTER_PORT_STAGE1_HM="${MASTER_PORT_STAGE1_HM:-29616}"

preflight_gpu
preflight_notify "$STAGE1_HM_FEISHU_NOTIFY"
require_file "$STAGE1_HM_CONFIG"
require_file "$STAGE1_HM_INIT_CKPT"
require_hf_model_dir "$INTERNNAV_BACKBONE"
require_dir "$STAGE1_HM_DATA_ROOT"
if [[ -n "${STAGE1_HM_VAL_ROOT:-}" ]]; then
  require_dir "$STAGE1_HM_VAL_ROOT"
else
  log "STAGE1_HM_VAL_ROOT empty: no separate val root preflight; val_root cleared in generated config"
fi
mkdir -p "$STAGE1_HM_OUT_DIR" "$STAGE1_HM_TB_DIR"

STAGE1_HM_TMP_CONFIG="$(mktemp "/tmp/stage1_hm.XXXXXX")"
TMP_CONFIGS+=("$STAGE1_HM_TMP_CONFIG")

export GPU_DEVICES
export STAGE1_HM_DATA_ROOT STAGE1_HM_VAL_ROOT STAGE1_HM_OUT_DIR STAGE1_HM_TB_DIR
export STAGE1_HM_VALIDATION_ENABLED
export STAGE1_HM_EPOCHS STAGE1_HM_BATCH_SIZE STAGE1_HM_GRAD_ACCUM_STEPS
export STAGE1_HM_NUM_WORKERS STAGE1_HM_PREFETCH_FACTOR STAGE1_HM_PIN_MEMORY
export STAGE1_HM_FEISHU_NOTIFY

make_stage_config STAGE1_HM "$STAGE1_HM_CONFIG" "$STAGE1_HM_TMP_CONFIG"

log "Stage: Stage1-HM heatmap LoRA"
log "Repo root: $ROOT_DIR"
log "Training GPUs: $GPU_DEVICES (nproc_per_node=$NPROC_PER_NODE)"
log "InternNav backbone: $INTERNNAV_BACKBONE"
log "Config: $STAGE1_HM_TMP_CONFIG"
log "Load weights: $STAGE1_HM_INIT_CKPT"
log "Output dir: $STAGE1_HM_OUT_DIR"

if is_truthy "$STAGE_DRY_RUN"; then
  log "STAGE_DRY_RUN=$STAGE_DRY_RUN; preflight and config generation completed, skipping training"
  exit 0
fi

run_training_stage "Stage1-HM heatmap LoRA" "$MASTER_PORT_STAGE1_HM" \
  "$STAGE1_HM_TMP_CONFIG" "$STAGE1_HM_INIT_CKPT" "$STAGE1_HM_MAX_BATCHES"

FINAL_CKPT="$(choose_checkpoint "${STAGE1_HM_OUT_DIR}/latest/checkpoints" "$STAGE1_HM_CHECKPOINT_PREFERENCE")"
log "Stage1-HM checkpoint: $FINAL_CKPT"
