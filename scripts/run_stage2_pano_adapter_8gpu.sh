#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage_training_common.sh"

# ------------------------------------------------------------------
# Stage2: Pano Adapter Teacher-Student Training
# ------------------------------------------------------------------
# Trains GeometryAwarePanoToNextDiTAdapter to translate frozen
# Pano-System2 traj_hidden_states into InternNav System1 condition space.
#
# Student: Stage1-S2 checkpoint (frozen LoRA Qwen VLM)
# Teacher: InternNav System1 (frozen, on-the-fly or pre-collected)
# Train:   Adapter only (~3M params)

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
# Teacher JSONL (optional for aligned mode — auto-generated from dataset if empty)
STAGE2_ADAPTER_TEACHER_JSONL="${STAGE2_ADAPTER_TEACHER_JSONL:-}"

# Training hyperparams (empty values defer to adapter config YAML)
STAGE2_ADAPTER_EPOCHS="${STAGE2_ADAPTER_EPOCHS:-}"
STAGE2_ADAPTER_BATCH_SIZE="${STAGE2_ADAPTER_BATCH_SIZE:-}"
STAGE2_ADAPTER_LR="${STAGE2_ADAPTER_LR:-}"
STAGE2_ADAPTER_WEIGHT_DECAY="${STAGE2_ADAPTER_WEIGHT_DECAY:-}"
STAGE2_ADAPTER_GRAD_CLIP="${STAGE2_ADAPTER_GRAD_CLIP:-}"
STAGE2_ADAPTER_MAX_SAMPLES="${STAGE2_ADAPTER_MAX_SAMPLES:-0}"

# Teacher
STAGE2_ADAPTER_TEACHER_MODE="${STAGE2_ADAPTER_TEACHER_MODE:-aligned}"
STAGE2_ADAPTER_TEACHER_DTYPE="${STAGE2_ADAPTER_TEACHER_DTYPE:-bfloat16}"
STAGE2_ADAPTER_TEACHER_ATTN="${STAGE2_ADAPTER_TEACHER_ATTN:-sdpa}"
STAGE2_ADAPTER_REQUIRE_FLASH_ATTN="${STAGE2_ADAPTER_REQUIRE_FLASH_ATTN:-1}"

# Loss weights
STAGE2_ADAPTER_COSINE_WEIGHT="${STAGE2_ADAPTER_COSINE_WEIGHT:-}"
STAGE2_ADAPTER_MSE_WEIGHT="${STAGE2_ADAPTER_MSE_WEIGHT:-}"
STAGE2_ADAPTER_POLICY_WEIGHT="${STAGE2_ADAPTER_POLICY_WEIGHT:-}"
STAGE2_ADAPTER_GT_WEIGHT="${STAGE2_ADAPTER_GT_WEIGHT:-}"

# Dataset behavior is controlled by data.trajectory in the adapter config YAML.
# This direct record loop intentionally has no DataLoader worker settings.
STAGE2_ADAPTER_USE_TRAJ_IMAGES="${STAGE2_ADAPTER_USE_TRAJ_IMAGES:-true}"

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
require_file "$ROOT_DIR/data/fgr2r/subinstr_mapping.json.gz"

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
  if [[ -n "${STAGE2_ADAPTER_COSINE_WEIGHT:-}" ]]; then
    args+=(--cosine-weight "$STAGE2_ADAPTER_COSINE_WEIGHT")
  fi
  if [[ -n "${STAGE2_ADAPTER_MSE_WEIGHT:-}" ]]; then
    args+=(--mse-weight "$STAGE2_ADAPTER_MSE_WEIGHT")
  fi
  if [[ -n "${STAGE2_ADAPTER_POLICY_WEIGHT:-}" ]]; then
    args+=(--policy-weight "$STAGE2_ADAPTER_POLICY_WEIGHT")
  fi
  if [[ -n "${STAGE2_ADAPTER_GT_WEIGHT:-}" ]]; then
    args+=(--gt-weight "$STAGE2_ADAPTER_GT_WEIGHT")
  fi
  if [[ -n "${STAGE2_ADAPTER_MAX_SAMPLES:-}" && "${STAGE2_ADAPTER_MAX_SAMPLES}" -gt 0 ]]; then
    args+=(--max-samples "$STAGE2_ADAPTER_MAX_SAMPLES")
  fi

  # use_traj_images
  if is_truthy "${STAGE2_ADAPTER_USE_TRAJ_IMAGES:-true}"; then
    args+=(--use-traj-images)
  else
    args+=(--no-use-traj-images)
  fi

  # Teacher JSONL (optional in aligned mode)
  if [[ -n "${STAGE2_ADAPTER_TEACHER_JSONL:-}" ]]; then
    args+=(--teacher-jsonl "$STAGE2_ADAPTER_TEACHER_JSONL")
  fi

  # Resume
  if [[ -n "${STAGE2_ADAPTER_RESUME:-}" ]]; then
    args+=(--resume-adapter "$STAGE2_ADAPTER_RESUME")
  fi

  printf '%s\n' "${args[@]}"
}

log "Stage: Stage2 Pano Adapter Teacher-Student Training"
log "Repo root: $ROOT_DIR"
log "Training GPUs: $GPU_DEVICES (nproc_per_node=$NPROC_PER_NODE)"
log "Student config: $STAGE2_ADAPTER_STUDENT_CONFIG"
log "Adapter config: $STAGE2_ADAPTER_CONFIG"
log "Load weights (Stage1-S2): $STAGE2_ADAPTER_LOAD_WEIGHTS"
log "InternNav model: $STAGE2_ADAPTER_INTERNNAV_MODEL"
log "Output dir: $STAGE2_ADAPTER_OUT_DIR"
log "Teacher mode: $STAGE2_ADAPTER_TEACHER_MODE"
log "Teacher JSONL: ${STAGE2_ADAPTER_TEACHER_JSONL:-<none — aligned mode requires no sidecar>}"

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
