#!/usr/bin/env bash
# Wait for the final Stage2 h1024 adapter, validate it, then launch formal Stage3.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Never inherit smoke/debug limits into the formal run.
unset STAGE_DRY_RUN
unset STAGE3_DRY_RUN
unset STAGE3_MAX_CLIPS
unset STAGE3_MAX_BATCHES

export MASTER_PORT="${MASTER_PORT:-29620}"
export MASTER_PORT_STAGE3="${MASTER_PORT_STAGE3:-$MASTER_PORT}"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data}"
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_REPO="${INTERNNAV_REPO:-/mnt/afs/lixiaoou/intern/fjl/InternNav}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"

export STAGE3_CONFIG="${STAGE3_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_BASE_CKPT="${STAGE3_BASE_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"

export STAGE2_ADAPTER_OUT_DIR="${STAGE2_ADAPTER_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter_full_11000_alllora_h1024}"
export STAGE2_ADAPTER_FINAL_EPOCH="${STAGE2_ADAPTER_FINAL_EPOCH:-3}"
printf -v STAGE2_FINAL_CHECKPOINT_NAME 'epoch_%03d.pth' "$STAGE2_ADAPTER_FINAL_EPOCH"
export STAGE3_ADAPTER_CKPT="${STAGE3_ADAPTER_CKPT:-${STAGE2_ADAPTER_OUT_DIR}/${STAGE2_FINAL_CHECKPOINT_NAME}}"

export STAGE3_OUT_DIR="${STAGE3_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage3_pano_system1_full_11000_alllora_h1024}"
export STAGE3_TB_DIR="${STAGE3_TB_DIR:-/mnt/afs/lixiaoou/intern/fjl/tensorlog/heatmapvln_stage3_pano_system1_full_11000_alllora_h1024}"
export LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/stage3_pano_system1_full_11000_alllora_h1024_8gpu_mxc500.log}"

# Formal Stage3 settings validated by the four-GPU, 100-step smoke run.
export STAGE3_EPOCHS=2
export STAGE3_BATCH_SIZE=8
export STAGE3_GRAD_ACCUM_STEPS=1
export STAGE3_PANO_ADAPTER_LR=5e-5
export STAGE3_L2_SP_ENABLED=1
export STAGE3_L2_SP_WEIGHT=1e-4

export STAGE3_NUM_WORKERS=16
export STAGE3_PREFETCH_FACTOR=4
export STAGE3_PIN_MEMORY=1
export STAGE3_SHM_BYPASS=auto
export STAGE3_ENABLE_TIMING=1
export STAGE3_SHOW_GPU_MEMORY=0
export STAGE3_LOG_INTERVAL=20
export STAGE3_TENSORBOARD_INTERVAL=20
export STAGE3_PAGE_CACHE_DROP_ENABLED=0
export STAGE3_SYSTEM2_SAMPLE_STEP=1

# Keep the verified frozen-Qwen execution path.
export STAGE3_MERGE_FROZEN_LORA=0
export STAGE3_FROZEN_TRAJ_INFERENCE_MODE=1
export STAGE3_TRAJ_LAST_HIDDEN_STATE_ONLY=0
export STAGE3_REQUIRE_FLASH_ATTN=1
export STAGE3_DRY_RUN=0
export STAGE_DRY_RUN=0

export STAGE3_CHECKPOINT_WAIT_INTERVAL_S="${STAGE3_CHECKPOINT_WAIT_INTERVAL_S:-300}"
export STAGE3_CHECKPOINT_SETTLE_S="${STAGE3_CHECKPOINT_SETTLE_S:-30}"
QWEN_PYTHON="${QWEN_PYTHON:-/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python}"

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

validate_stage2_adapter() {
  "$QWEN_PYTHON" - \
    "$STAGE3_ADAPTER_CKPT" \
    "$STAGE2_ADAPTER_FINAL_EPOCH" \
    "$STAGE3_BASE_CKPT" <<'PY'
import os
import sys

import torch

checkpoint_path, expected_epoch_text, expected_base = sys.argv[1:]
expected_epoch = int(expected_epoch_text)

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"Stage2 checkpoint is not readable yet: {checkpoint_path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

errors = []
if checkpoint.get("adapter_type") != "pano_latent_space":
    errors.append(f"adapter_type={checkpoint.get('adapter_type')!r}")
if int(checkpoint.get("epoch", -1)) != expected_epoch:
    errors.append(f"epoch={checkpoint.get('epoch')!r}, expected={expected_epoch}")
if int(checkpoint.get("step", 0)) <= 0:
    errors.append(f"invalid step={checkpoint.get('step')!r}")

args = checkpoint.get("args") or {}
if int(args.get("epochs", -1)) != expected_epoch:
    errors.append(f"args.epochs={args.get('epochs')!r}, expected={expected_epoch}")
if int(args.get("adapter_hidden_dim", -1)) != 1024:
    errors.append(f"args.adapter_hidden_dim={args.get('adapter_hidden_dim')!r}, expected=1024")

actual_base = os.path.realpath(os.path.expanduser(str(args.get("base_checkpoint", ""))))
expected_base = os.path.realpath(os.path.expanduser(expected_base))
if actual_base != expected_base:
    errors.append(f"args.base_checkpoint={actual_base!r}, expected={expected_base!r}")

state = checkpoint.get("adapter_state_dict")
if not isinstance(state, dict):
    errors.append("adapter_state_dict is missing")
    state = {}

tensor_state = {key: value for key, value in state.items() if torch.is_tensor(value)}
numel = sum(value.numel() for value in tensor_state.values())
if len(tensor_state) != 4:
    errors.append(f"adapter tensor count={len(tensor_state)}, expected=4")
if numel != 7_344_640:
    errors.append(f"adapter numel={numel}, expected=7344640")

nonfinite = [
    key for key, value in tensor_state.items()
    if not bool(torch.isfinite(value.float()).all())
]
if nonfinite:
    errors.append(f"non-finite adapter tensors={nonfinite}")
if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
    errors.append("optimizer_state_dict is missing")

if errors:
    print("Stage2 final checkpoint validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(
    "Validated Stage2 final adapter: "
    f"epoch={expected_epoch} step={checkpoint['step']} "
    f"tensors={len(tensor_state)} params={numel} path={checkpoint_path}"
)
PY
}

require_file "$STAGE3_CONFIG"
require_file "$STAGE3_BASE_CKPT"
require_dir "$PANORAMIC_DATA_ROOT"
require_dir "$INTERNNAV_MODEL_PATH"
if [[ ! -x "$QWEN_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN_PYTHON" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Waiting for final Stage2 adapter: $STAGE3_ADAPTER_CKPT"
while true; do
  if [[ -s "$STAGE3_ADAPTER_CKPT" ]] && validate_stage2_adapter; then
    size_before="$(stat -c %s "$STAGE3_ADAPTER_CKPT")"
    echo "[$(date '+%F %T')] Candidate checkpoint is valid; waiting ${STAGE3_CHECKPOINT_SETTLE_S}s for size stability"
    sleep "$STAGE3_CHECKPOINT_SETTLE_S"
    size_after="$(stat -c %s "$STAGE3_ADAPTER_CKPT")"
    if [[ "$size_before" == "$size_after" ]] && validate_stage2_adapter; then
      break
    fi
    echo "[$(date '+%F %T')] Checkpoint changed during settle window; continuing to wait"
  fi
  echo "[$(date '+%F %T')] Stage2 final checkpoint not ready; retrying in ${STAGE3_CHECKPOINT_WAIT_INTERVAL_S}s"
  sleep "$STAGE3_CHECKPOINT_WAIT_INTERVAL_S"
done

mkdir -p "$STAGE3_OUT_DIR" "$STAGE3_TB_DIR" "$(dirname "$LOG_FILE")"

echo "[$(date '+%F %T')] Stage2 final checkpoint is ready"
echo "[stage3] base=$STAGE3_BASE_CKPT"
echo "[stage3] adapter=$STAGE3_ADAPTER_CKPT"
echo "[stage3] out=$STAGE3_OUT_DIR"
echo "[stage3] gpus=$GPU_DEVICES nproc=$NPROC_PER_NODE batch_per_rank=$STAGE3_BATCH_SIZE epochs=$STAGE3_EPOCHS"

bash scripts/run_stage3_pano_system1_h1024_8gpu_mxc500_launcher.sh
