#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
DAGGER_OUTPUT_DIR="${SYSTEM2_STOP_DAGGER_OUT_DIR:-${FJL_ROOT}/model/output_system2_stop_head_fullprior_dagger_all61_boundary301_seed123}"
DAGGER_CHECKPOINT="${SYSTEM2_STOP_DAGGER_CHECKPOINT:-${DAGGER_OUTPUT_DIR}/latest.pth}"
DAGGER_SUMMARY="${SYSTEM2_STOP_DAGGER_SUMMARY:-${DAGGER_OUTPUT_DIR}/summary.json}"
DAGGER_RUN_LOG="${SYSTEM2_STOP_DAGGER_RUN_LOG:-${REPO_ROOT}/logs/system2_stop_head_fullprior_dagger_all61_boundary301_seed123_detached_20260717.log}"
WAIT_INTERVAL_S="${STAGE3_FIXED3_WAIT_INTERVAL_S:-60}"
SETTLE_S="${STAGE3_FIXED3_CHECKPOINT_SETTLE_S:-30}"

export STAGE3_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_EVAL_BASE_CKPT="${STAGE3_EVAL_BASE_CKPT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords_priorfix/latest/checkpoints/epoch_002.pth}"
export STAGE3_EVAL_EXPECTED_EPOCH=2
export STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT="$DAGGER_CHECKPOINT"
export STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY=1e-4

export STAGE3_EVAL_MODEL_GPU="${STAGE3_FIXED3_GPU:-7}"
export STAGE3_EVAL_SIM_GPU="$STAGE3_EVAL_MODEL_GPU"
export STAGE3_EVAL_RPC_PORT="${STAGE3_FIXED3_RPC_PORT:-50067}"
export STAGE3_EVAL_DISPLAY="${STAGE3_FIXED3_DISPLAY:-localhost:200.0}"
export STAGE3_EVAL_OUTPUT_PATH="${STAGE3_FIXED3_OUTPUT_PATH:-${FJL_ROOT}/model/eval_system2_stop_head_fullprior_dagger_all61_boundary301_fixed3_gate1e4_20260717}"
export STAGE3_EVAL_EPISODE_LIST=configs/eval_cohorts/stage3_stop_head_fixed3.json
export STAGE3_EVAL_EXPECTED_EPISODES=1839

export STAGE3_EVAL_RPC_PROTOCOL_SEED=42
export STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1
export STAGE3_EVAL_MAX_STEPS=500
export STAGE3_EVAL_ACTION_CHUNK_SIZE=4
export STAGE3_EVAL_STOP_CONFIRMATIONS=2
export STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS=1
export STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP=0
unset STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD
export STAGE3_EVAL_CLOSED_LOOP_GUARD=1
export STAGE3_EVAL_RECOVERY_TURNS=6
export STAGE3_EVAL_RECOVERY_FORWARD_STEPS=2
export STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN=1

export STAGE3_EVAL_TRAJECTORY_SELECTION=mean
export STAGE3_EVAL_TRAJECTORY_X_SIGN=1
export STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT=none
export STAGE3_EVAL_SYSTEM1_COORD_ORDER=generated
export STAGE3_EVAL_AUTO_STOP_DISTANCE=0
export STAGE3_EVAL_ORACLE_SYSTEM2=0
export STAGE3_EVAL_ALLOW_PRIVILEGED=0
export STAGE3_EVAL_COLLECT_STOP_FEATURES=0
export STAGE3_EVAL_RESUME=0
export STAGE3_EVAL_OVERWRITE=0
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS=0

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ -e "$STAGE3_EVAL_OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite existing fixed3 output: $STAGE3_EVAL_OUTPUT_PATH" >&2
  exit 1
fi

completion_marker="[stop-dagger] complete: $DAGGER_CHECKPOINT"
while [[ ! -s "$DAGGER_CHECKPOINT" || ! -s "$DAGGER_SUMMARY" ]] \
  || [[ ! -s "$DAGGER_RUN_LOG" ]] \
  || ! grep -Fq "$completion_marker" "$DAGGER_RUN_LOG"; do
  printf '[%s] waiting for postflight-complete full-prior DAgger checkpoint: %s\n' \
    "$(date '+%F %T')" "$DAGGER_CHECKPOINT"
  sleep "$WAIT_INTERVAL_S"
done

size_before="$(stat -c %s "$DAGGER_CHECKPOINT")"
sleep "$SETTLE_S"
size_after="$(stat -c %s "$DAGGER_CHECKPOINT")"
if [[ "$size_before" != "$size_after" ]]; then
  echo "DAgger checkpoint changed during settle window: $DAGGER_CHECKPOINT" >&2
  exit 1
fi

"$QWEN25_PYTHON" - "$DAGGER_CHECKPOINT" "$DAGGER_SUMMARY" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
summary = json.loads(summary_path.read_text(encoding="utf-8"))

expected = {
    "records": 20857,
    "train_records": 16228,
    "val_records": 4629,
    "positive_records": 1119,
    "negative_records": 19738,
    "relabelled_negative_records": 415,
}
for key, value in expected.items():
    if int(summary.get(key, -1)) != value:
        raise SystemExit(f"Unexpected DAgger summary {key}={summary.get(key)!r}; expected {value}")

if checkpoint.get("stage_name") != "system2_stop_head":
    raise SystemExit("DAgger checkpoint has the wrong stage_name")
state = checkpoint.get("trainable_state_dict")
if not isinstance(state, dict) or len(state) != 10:
    raise SystemExit(f"Expected 10 STOP-head tensors, found {0 if not isinstance(state, dict) else len(state)}")
if not all(str(name).startswith("stop_head.") for name in state):
    raise SystemExit("DAgger checkpoint contains non-STOP-head trainable tensors")
if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
    raise SystemExit("DAgger checkpoint contains invalid STOP-head tensors")

head_config = checkpoint["config"]["model"]["stop_head"]
veto = float(head_config["veto_stop_threshold"])
add = float(head_config["add_stop_threshold"])
if not (math.isfinite(veto) and math.isfinite(add) and 0.0 <= veto < add <= 1.0):
    raise SystemExit(f"Invalid STOP thresholds: veto={veto} add={add}")
print(f"Fixed3 preflight passed: records={summary['records']} veto={veto:.3f} add={add:.3f}")
PY

echo "[$(date '+%F %T')] launching fixed3 closed-loop regression on GPU $STAGE3_EVAL_MODEL_GPU"
exec bash scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh
