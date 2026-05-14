#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${HEATMAPVLN_INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}}"
INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export INTERNNAV_MODEL_PATH
export INTERNNAV_BACKBONE
# 与 paths.internnav_model_path / 环境变量覆盖逻辑一致，默认指向完整 InternNav HF 权重目录

BASE_CONFIG="${BASE_CONFIG:-configs/train_system2_panoramic_sft_2gpu.yaml}"

DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:?STAGE1_INIT_CKPT is required}"

RUN_NAME="${RUN_NAME:-stage1_s2_8gpu_cluster}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29617}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-true}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/${RUN_NAME}}"
TB_DIR="${TB_DIR:-$ROOT_DIR/tf-logs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/run-logs}"

RUN_SANITY="${RUN_SANITY:-1}"
SANITY_SPLIT="${SANITY_SPLIT:-train}"
SANITY_NUM_SAMPLES="${SANITY_NUM_SAMPLES:-128}"
SANITY_PRINT_EXAMPLES="${SANITY_PRINT_EXAMPLES:-20}"
SANITY_DEVICE="${SANITY_DEVICE:-cuda:0}"
SANITY_OUTPUT="${SANITY_OUTPUT:-$LOG_DIR/${RUN_NAME}_sanity.jsonl}"
CHECKPOINT_PREFERENCE="${CHECKPOINT_PREFERENCE:-latest}"

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

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    log "Missing required directory: $path"
    exit 1
  fi
}

choose_checkpoint() {
  local ckpt_dir="$1"
  local preference="${2:-latest}"
  local first
  local second

  if [[ "$preference" == "best" ]]; then
    first="${ckpt_dir}/best.pth"
    second="${ckpt_dir}/latest.pth"
  else
    first="${ckpt_dir}/latest.pth"
    second="${ckpt_dir}/best.pth"
  fi

  if [[ -f "$first" ]]; then
    printf '%s\n' "$first"
    return 0
  fi
  if [[ -f "$second" ]]; then
    printf '%s\n' "$second"
    return 0
  fi

  log "No checkpoint found in $ckpt_dir"
  exit 1
}

mkdir -p "$OUT_DIR" "$TB_DIR" "$LOG_DIR"

require_file "$BASE_CONFIG"
require_file "$STAGE1_INIT_CKPT"
require_dir "$DATA_ROOT"
require_dir "$INTERNNAV_BACKBONE"
require_file "$ROOT_DIR/data/fgr2r/subinstr_mapping.json.gz"

TMP_CONFIG="$(mktemp "/tmp/${RUN_NAME}.XXXXXX.yaml")"
trap 'rm -f "$TMP_CONFIG"' EXIT

export DATA_ROOT GPU_DEVICES OUT_DIR TB_DIR
export NUM_WORKERS PREFETCH_FACTOR PIN_MEMORY
export EPOCHS BATCH_SIZE GRAD_ACCUM_STEPS

python - "$BASE_CONFIG" "$TMP_CONFIG" <<'PY'
import os
import sys

import yaml

from src.config_schema import prepare_config_for_use

base_config, output_config = sys.argv[1], sys.argv[2]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = prepare_config_for_use(cfg)

cfg.setdefault("data", {})
cfg["data"]["root"] = os.environ["DATA_ROOT"]
cfg["data"]["num_workers"] = int(os.environ["NUM_WORKERS"])
cfg["data"]["prefetch_factor"] = int(os.environ["PREFETCH_FACTOR"])
cfg["data"]["pin_memory"] = os.environ["PIN_MEMORY"].strip().lower() in {
    "1", "true", "yes", "on",
}

cfg.setdefault("optim", {})
cfg["optim"]["batch_size"] = int(os.environ["BATCH_SIZE"])
cfg["optim"]["grad_accum_steps"] = int(os.environ["GRAD_ACCUM_STEPS"])

cfg.setdefault("gpu", {})
cfg["gpu"]["devices"] = [
    int(token.strip())
    for token in os.environ["GPU_DEVICES"].split(",")
    if token.strip()
]
cfg.setdefault("gpu", {}).setdefault("multi_gpu", {})["enabled"] = True
cfg["gpu"].setdefault("backend", "nccl")

cfg.setdefault("log", {})
cfg["log"]["out_dir"] = os.environ["OUT_DIR"]
cfg["log"]["tensorboard_dir"] = os.environ["TB_DIR"]

stages = cfg.setdefault("training", {}).setdefault("stages", [])
if not stages:
    raise RuntimeError("training.stages is empty in the base config")
stages[0]["epochs"] = int(os.environ["EPOCHS"])

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

log "Repo root: $ROOT_DIR"
log "Generated config: $TMP_CONFIG"
log "DATA_ROOT=$DATA_ROOT"
log "STAGE1_INIT_CKPT=$STAGE1_INIT_CKPT"
log "GPU_DEVICES=$GPU_DEVICES"
log "OUT_DIR=$OUT_DIR"
log "TB_DIR=$TB_DIR"
log "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py \
  --config "$TMP_CONFIG" \
  --load-weights "$STAGE1_INIT_CKPT" \
  --epochs "$EPOCHS"

FINAL_CKPT="$(choose_checkpoint "$OUT_DIR/latest/checkpoints" "$CHECKPOINT_PREFERENCE")"
log "Training finished, selected checkpoint: $FINAL_CKPT"

if [[ "$RUN_SANITY" == "1" || "$RUN_SANITY" == "true" || "$RUN_SANITY" == "yes" ]]; then
  log "Running Stage1-S2 sanity check"
  python scripts/evaluation/system2_sft_sanity_check.py \
    --config "$TMP_CONFIG" \
    --checkpoint "$FINAL_CKPT" \
    --root "$DATA_ROOT" \
    --split "$SANITY_SPLIT" \
    --num-samples "$SANITY_NUM_SAMPLES" \
    --print-examples "$SANITY_PRINT_EXAMPLES" \
    --device "$SANITY_DEVICE" \
    --output "$SANITY_OUTPUT"
  log "Sanity output: $SANITY_OUTPUT"
else
  log "RUN_SANITY=$RUN_SANITY, skip sanity check"
fi
