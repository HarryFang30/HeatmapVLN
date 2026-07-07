#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage_training_common.sh"

# ------------------------------------------------------------------
# Stage2: Pano Adapter Native-Teacher Distillation
# ------------------------------------------------------------------
# Trains PanoLatentSpaceAdapter to translate frozen Pano-System2
# traj_hidden_states into native InternNav's executable 3584-dim latent space.
#
# Student: Stage1-S2 checkpoint (frozen LoRA Qwen VLM)
# Frozen executor: InternNav cond_projector + NextDiT
# Train:   Adapter only (1.84M params with hidden_dim=256)

STAGE2_ADAPTER_STUDENT_CONFIG="${STAGE2_ADAPTER_STUDENT_CONFIG:-configs/train_pano_adapter_stage2_8gpu.yaml}"
STAGE2_ADAPTER_CONFIG="${STAGE2_ADAPTER_CONFIG:-$STAGE2_ADAPTER_STUDENT_CONFIG}"
PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-${DATA_ROOT:-/workspace/r2r_panoramic_data}}"
STAGE2_ADAPTER_DATA_ROOT="${STAGE2_ADAPTER_DATA_ROOT:-$PANORAMIC_DATA_ROOT}"
STAGE1_S2_OUT_DIR="${STAGE1_S2_OUT_DIR:-${SYSTEM2_SFT_OUT_DIR:-/root/autodl-tmp/vln_system2_sft_outputs}}"
STAGE1_S2_CHECKPOINT_PREFERENCE="${STAGE1_S2_CHECKPOINT_PREFERENCE:-latest}"

# Paths
STAGE2_ADAPTER_LOAD_WEIGHTS="${STAGE2_ADAPTER_LOAD_WEIGHTS:-${BASE_CHECKPOINT:-}}"
STAGE2_ADAPTER_INTERNNAV_MODEL="${STAGE2_ADAPTER_INTERNNAV_MODEL:-$INTERNNAV_MODEL_PATH}"
STAGE2_ADAPTER_INTERNNAV_REPO="${STAGE2_ADAPTER_INTERNNAV_REPO:-${INTERNNAV_REPO:-~/InternNav}}"
STAGE2_ADAPTER_OUT_DIR="${STAGE2_ADAPTER_OUT_DIR:-/root/autodl-tmp/vln_pano_adapter_outputs}"
# Adapter training iterates records directly (no DataLoader).  num_workers /
# prefetch_factor from the config YAML do not affect this script's path,
# and TensorBoard logging is not yet wired — only console logging is active.
# Native teacher JSONL collected with:
#   collect_internnav_teacher_sidecar.py --coord-source dataset --tensor-output-dir ...
# Records are aligned by (clip_idx,current_t), not integer dataset_index.
STAGE2_ADAPTER_TEACHER_JSONL="${STAGE2_ADAPTER_TEACHER_JSONL:-}"

# Training hyperparams (empty values defer to adapter config YAML)
STAGE2_ADAPTER_EPOCHS="${STAGE2_ADAPTER_EPOCHS:-}"
STAGE2_ADAPTER_BATCH_SIZE="${STAGE2_ADAPTER_BATCH_SIZE:-}"
STAGE2_ADAPTER_LR="${STAGE2_ADAPTER_LR:-}"
STAGE2_ADAPTER_WEIGHT_DECAY="${STAGE2_ADAPTER_WEIGHT_DECAY:-}"
STAGE2_ADAPTER_GRAD_CLIP="${STAGE2_ADAPTER_GRAD_CLIP:-}"
STAGE2_ADAPTER_MAX_SAMPLES="${STAGE2_ADAPTER_MAX_SAMPLES:-0}"
STAGE2_ADAPTER_PREFETCH_BATCHES="${STAGE2_ADAPTER_PREFETCH_BATCHES:-}"
STAGE2_ADAPTER_PREFETCH_WORKERS="${STAGE2_ADAPTER_PREFETCH_WORKERS:-}"
STAGE2_ADAPTER_TEACHER_CACHE_MODE="${STAGE2_ADAPTER_TEACHER_CACHE_MODE:-}"
STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS="${STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS:-}"
STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE="${STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE:-}"
STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS="${STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS:-}"
STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES="${STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES:-}"
STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS="${STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS:-}"

# Teacher targets / loss weights
STAGE2_ADAPTER_TEACHER_MODE="${STAGE2_ADAPTER_TEACHER_MODE:-native_sidecar}"
STAGE2_ADAPTER_COMPUTE_TEACHER_MSE="${STAGE2_ADAPTER_COMPUTE_TEACHER_MSE:-0}"
STAGE2_ADAPTER_TEACHER_DTYPE="${STAGE2_ADAPTER_TEACHER_DTYPE:-bfloat16}"
STAGE2_ADAPTER_TEACHER_ATTN="${STAGE2_ADAPTER_TEACHER_ATTN:-sdpa}"
STAGE2_ADAPTER_REQUIRE_FLASH_ATTN="${STAGE2_ADAPTER_REQUIRE_FLASH_ATTN:-1}"
STAGE2_ADAPTER_RAW_WEIGHT="${STAGE2_ADAPTER_RAW_WEIGHT:-0.1}"
STAGE2_ADAPTER_COND_WEIGHT="${STAGE2_ADAPTER_COND_WEIGHT:-1.0}"
STAGE2_ADAPTER_GT_WEIGHT="${STAGE2_ADAPTER_GT_WEIGHT:-0.2}"

# Dataset behavior is controlled by data.trajectory in the adapter config YAML.
# This direct record loop intentionally has no DataLoader worker settings.

# Misc
STAGE2_ADAPTER_SEED="${STAGE2_ADAPTER_SEED:-42}"
STAGE2_ADAPTER_LOG_INTERVAL="${STAGE2_ADAPTER_LOG_INTERVAL:-10}"
STAGE2_ADAPTER_RESUME="${STAGE2_ADAPTER_RESUME:-}"
STAGE2_ADAPTER_FEISHU_NOTIFY="${STAGE2_ADAPTER_FEISHU_NOTIFY:-$FEISHU_NOTIFY}"
MASTER_PORT_STAGE2_ADAPTER="${MASTER_PORT_STAGE2_ADAPTER:-29618}"

# Resolve student checkpoint if not explicitly set
if [[ -z "$STAGE2_ADAPTER_LOAD_WEIGHTS" ]]; then
  STAGE2_ADAPTER_LOAD_WEIGHTS="$(choose_checkpoint "${STAGE1_S2_OUT_DIR}/latest/checkpoints" "$STAGE1_S2_CHECKPOINT_PREFERENCE")"
fi

preflight_gpu
preflight_notify "$STAGE2_ADAPTER_FEISHU_NOTIFY"
require_file "$STAGE2_ADAPTER_STUDENT_CONFIG"
require_file "$STAGE2_ADAPTER_LOAD_WEIGHTS"
require_hf_model_dir "$INTERNNAV_BACKBONE"
require_dir "$STAGE2_ADAPTER_DATA_ROOT"
if [[ "$STAGE2_ADAPTER_TEACHER_MODE" == "native_sidecar" && -z "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
  echo "STAGE2_ADAPTER_TEACHER_MODE=native_sidecar requires STAGE2_ADAPTER_TEACHER_JSONL." >&2
  echo "Collect it with scripts/evaluation/collect_internnav_teacher_sidecar.py --coord-source dataset --tensor-output-dir ..." >&2
  exit 1
fi

if [[ -n "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
  require_file "$STAGE2_ADAPTER_TEACHER_JSONL"
fi

mkdir -p "$STAGE2_ADAPTER_OUT_DIR"

export GPU_DEVICES
export HEATMAPVLN_REQUIRE_FLASH_ATTN="$STAGE2_ADAPTER_REQUIRE_FLASH_ATTN"

# ------------------------------------------------------------------
# Build CLI arguments from env vars
# ------------------------------------------------------------------
build_adapter_args() {
  local args=(
    --student-config "$STAGE2_ADAPTER_STUDENT_CONFIG"
    --adapter-config "$STAGE2_ADAPTER_CONFIG"
    --root "$STAGE2_ADAPTER_DATA_ROOT"
    --base-checkpoint "$STAGE2_ADAPTER_LOAD_WEIGHTS"
    --internnav-model-path "$STAGE2_ADAPTER_INTERNNAV_MODEL"
    --internnav-repo "$STAGE2_ADAPTER_INTERNNAV_REPO"
    --output-dir "$STAGE2_ADAPTER_OUT_DIR"
    --seed "$STAGE2_ADAPTER_SEED"
    --log-interval "$STAGE2_ADAPTER_LOG_INTERVAL"
    --teacher-target-mode "$STAGE2_ADAPTER_TEACHER_MODE"
    --teacher-torch-dtype "$STAGE2_ADAPTER_TEACHER_DTYPE"
    --teacher-attn-implementation "$STAGE2_ADAPTER_TEACHER_ATTN"
    --raw-distill-weight "$STAGE2_ADAPTER_RAW_WEIGHT"
    --cond-distill-weight "$STAGE2_ADAPTER_COND_WEIGHT"
    --gt-weight "$STAGE2_ADAPTER_GT_WEIGHT"
  )

  # Optional overrides (only pass if non-empty)
  if [[ -n "${STAGE2_ADAPTER_EPOCHS:-}" ]]; then
    args+=(--epochs "$STAGE2_ADAPTER_EPOCHS")
  fi
  if [[ -n "${STAGE2_ADAPTER_BATCH_SIZE:-}" ]]; then
    args+=(--batch-size "$STAGE2_ADAPTER_BATCH_SIZE")
  fi
  if [[ -n "${STAGE2_ADAPTER_LR:-}" ]]; then
    args+=(--lr "$STAGE2_ADAPTER_LR")
  fi
  if [[ -n "${STAGE2_ADAPTER_WEIGHT_DECAY:-}" ]]; then
    args+=(--weight-decay "$STAGE2_ADAPTER_WEIGHT_DECAY")
  fi
  if [[ -n "${STAGE2_ADAPTER_GRAD_CLIP:-}" ]]; then
    args+=(--grad-clip "$STAGE2_ADAPTER_GRAD_CLIP")
  fi
  if [[ -n "${STAGE2_ADAPTER_MAX_SAMPLES:-}" && "${STAGE2_ADAPTER_MAX_SAMPLES}" -gt 0 ]]; then
    args+=(--max-samples "$STAGE2_ADAPTER_MAX_SAMPLES")
  fi
  if [[ -n "${STAGE2_ADAPTER_PREFETCH_BATCHES:-}" ]]; then
    args+=(--prefetch-batches "$STAGE2_ADAPTER_PREFETCH_BATCHES")
  fi
  if [[ -n "${STAGE2_ADAPTER_PREFETCH_WORKERS:-}" ]]; then
    args+=(--prefetch-workers "$STAGE2_ADAPTER_PREFETCH_WORKERS")
  fi
  if [[ -n "${STAGE2_ADAPTER_TEACHER_CACHE_MODE:-}" ]]; then
    args+=(--teacher-cache-mode "$STAGE2_ADAPTER_TEACHER_CACHE_MODE")
  fi
  if [[ -n "${STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS:-}" ]]; then
    args+=(--teacher-cache-max-items "$STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS")
  fi
  if [[ -n "${STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE:-}" ]]; then
    if is_truthy "$STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE"; then
      args+=(--teacher-preload-cache)
    else
      args+=(--no-teacher-preload-cache)
    fi
  fi
  if [[ -n "${STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS:-}" ]]; then
    args+=(--teacher-preload-workers "$STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS")
  fi
  if [[ -n "${STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES:-}" ]]; then
    if is_truthy "$STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES"; then
      args+=(--check-teacher-tensor-files)
    else
      args+=(--no-check-teacher-tensor-files)
    fi
  fi
  if [[ -n "${STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS:-}" ]]; then
    if is_truthy "$STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS"; then
      args+=(--trust-native-sidecar-pano-labels)
    else
      args+=(--no-trust-native-sidecar-pano-labels)
    fi
  fi

  # Teacher JSONL (required in native_sidecar mode)
  if [[ -n "${STAGE2_ADAPTER_TEACHER_JSONL:-}" ]]; then
    args+=(--teacher-jsonl "$STAGE2_ADAPTER_TEACHER_JSONL")
  fi
  if is_truthy "$STAGE2_ADAPTER_COMPUTE_TEACHER_MSE"; then
    args+=(--compute-teacher-mse)
  fi

  # Resume
  if [[ -n "${STAGE2_ADAPTER_RESUME:-}" ]]; then
    args+=(--resume-adapter "$STAGE2_ADAPTER_RESUME")
  fi

  printf '%s\n' "${args[@]}"
}

log "Stage: Stage2 Pano Adapter Native-Teacher Distillation"
log "Repo root: $ROOT_DIR"
log "Training GPUs: $GPU_DEVICES (nproc_per_node=$NPROC_PER_NODE)"
log "Student config: $STAGE2_ADAPTER_STUDENT_CONFIG"
log "Adapter config: $STAGE2_ADAPTER_CONFIG"
log "Load weights (Stage1-S2): $STAGE2_ADAPTER_LOAD_WEIGHTS"
log "InternNav model: $STAGE2_ADAPTER_INTERNNAV_MODEL"
log "Output dir: $STAGE2_ADAPTER_OUT_DIR"
log "Teacher target mode: $STAGE2_ADAPTER_TEACHER_MODE"
log "Loss weights: raw=$STAGE2_ADAPTER_RAW_WEIGHT cond=$STAGE2_ADAPTER_COND_WEIGHT gt=$STAGE2_ADAPTER_GT_WEIGHT"
log "Prefetch batches: ${STAGE2_ADAPTER_PREFETCH_BATCHES:-<config default>}"
log "Prefetch workers: ${STAGE2_ADAPTER_PREFETCH_WORKERS:-<config default>}"
log "Teacher cache: mode=${STAGE2_ADAPTER_TEACHER_CACHE_MODE:-<config default>} preload=${STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE:-<config default>} preload_workers=${STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS:-<config default>}"
log "Teacher startup filter: check_tensor_files=${STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES:-<config default>} trust_native_pano_labels=${STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS:-<config default>}"
log "Teacher MSE diagnostic: $STAGE2_ADAPTER_COMPUTE_TEACHER_MSE"
log "Record JSONL: ${STAGE2_ADAPTER_TEACHER_JSONL:-<none>}"

if is_truthy "$STAGE_DRY_RUN"; then
  log "STAGE_DRY_RUN=$STAGE_DRY_RUN; preflight and config validation completed, skipping training"
  log "Would run: torchrun --master_port=$MASTER_PORT_STAGE2_ADAPTER --nproc_per_node=$NPROC_PER_NODE"
  log "  scripts/training/train_pano_latent_adapter.py $(build_adapter_args | xargs echo)"
  exit 0
fi

log "Starting Stage2 Pano Adapter training"
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --master_port="$MASTER_PORT_STAGE2_ADAPTER" \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/training/train_pano_latent_adapter.py $(build_adapter_args)

FINAL_CKPT="$STAGE2_ADAPTER_OUT_DIR/latest.pth"
if [[ -f "$FINAL_CKPT" ]]; then
  log "Stage2 Pano Adapter checkpoint: $FINAL_CKPT"
else
  log "Stage2 Pano Adapter training completed (checkpoints in $STAGE2_ADAPTER_OUT_DIR)"
fi
