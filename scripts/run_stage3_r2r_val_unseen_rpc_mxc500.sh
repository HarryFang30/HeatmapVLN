#!/usr/bin/env bash
# Wait for the final Stage3 adapter and run R2R val_unseen through the
# qwen25-model <-> vlnce-Habitat RPC bridge on one MXC500 node.

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export FJL_ROOT="${FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}"
export RPC_ROOT="${RPC_ROOT:-${FJL_ROOT}/rpc}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
VLNCE_PYTHON="${VLNCE_PYTHON:-${FJL_ROOT}/envs/vlnce/bin/python}"

export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${FJL_ROOT}/InternNav-Model}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export STAGE3_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_EVAL_BASE_CKPT="${STAGE3_EVAL_BASE_CKPT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_EVAL_TRAIN_OUT_DIR="${STAGE3_EVAL_TRAIN_OUT_DIR:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords}"
export STAGE3_EVAL_EXPECTED_EPOCH="${STAGE3_EVAL_EXPECTED_EPOCH:-2}"
printf -v STAGE3_EVAL_CHECKPOINT_NAME 'epoch_%03d.pth' "$STAGE3_EVAL_EXPECTED_EPOCH"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${STAGE3_EVAL_TRAIN_OUT_DIR}/latest/checkpoints/${STAGE3_EVAL_CHECKPOINT_NAME}}"

export STAGE3_EVAL_SCENES_DIR="${STAGE3_EVAL_SCENES_DIR:-${FJL_ROOT}/habitat/VLN-CE/data/scene_datasets}"
export STAGE3_EVAL_DATA_PATH="${STAGE3_EVAL_DATA_PATH:-${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz}"
export STAGE3_EVAL_DATASET_SPLIT="${STAGE3_EVAL_DATASET_SPLIT:-val_unseen}"
export STAGE3_EVAL_EXPECTED_EPISODES="${STAGE3_EVAL_EXPECTED_EPISODES:-1839}"
export STAGE3_EVAL_OUTPUT_PATH="${STAGE3_EVAL_OUTPUT_PATH:-${FJL_ROOT}/model/eval_stage3_r2r_val_unseen_full_11000_alllora_h1024_internnavcoords_epoch${STAGE3_EVAL_EXPECTED_EPOCH}_no_privileged_stop}"

export STAGE3_EVAL_MODEL_GPU="${STAGE3_EVAL_MODEL_GPU:-0}"
export STAGE3_EVAL_DISPLAY="${STAGE3_EVAL_DISPLAY:-localhost:200.0}"
export STAGE3_EVAL_RPC_HOST="${STAGE3_EVAL_RPC_HOST:-127.0.0.1}"
export STAGE3_EVAL_RPC_PORT="${STAGE3_EVAL_RPC_PORT:-50061}"
export STAGE3_EVAL_RPC_TIMEOUT_MS="${STAGE3_EVAL_RPC_TIMEOUT_MS:-600000}"
export STAGE3_EVAL_RPC_JPEG_QUALITY="${STAGE3_EVAL_RPC_JPEG_QUALITY:-90}"
export STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1="${STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1:-1}"
export STAGE3_EVAL_SERVER_START_TIMEOUT_S="${STAGE3_EVAL_SERVER_START_TIMEOUT_S:-1800}"

export STAGE3_EVAL_MAX_EPISODES="${STAGE3_EVAL_MAX_EPISODES:-}"
export STAGE3_EVAL_EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-}"
export STAGE3_EVAL_MAX_STEPS="${STAGE3_EVAL_MAX_STEPS:-500}"
export STAGE3_EVAL_MAX_SYSTEM2_CALLS="${STAGE3_EVAL_MAX_SYSTEM2_CALLS:-0}"
export STAGE3_EVAL_NUM_HISTORY="${STAGE3_EVAL_NUM_HISTORY:-8}"
export STAGE3_EVAL_TRAJECTORY_SELECTION="${STAGE3_EVAL_TRAJECTORY_SELECTION:-mean}"
# Corrected Stage3 checkpoints already emit native InternNav coordinates.
# Set -1 explicitly only when diagnosing a legacy pre-coordinate-fix checkpoint.
export STAGE3_EVAL_TRAJECTORY_X_SIGN="${STAGE3_EVAL_TRAJECTORY_X_SIGN:-1}"
export STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT="${STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT:-none}"
export STAGE3_EVAL_SYSTEM1_COORD_ORDER="${STAGE3_EVAL_SYSTEM1_COORD_ORDER:-generated}"
export STAGE3_EVAL_AUTO_STOP_DISTANCE="${STAGE3_EVAL_AUTO_STOP_DISTANCE:-0.0}"
export STAGE3_EVAL_ORACLE_SYSTEM2="${STAGE3_EVAL_ORACLE_SYSTEM2:-0}"
export STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY="${STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY:-farthest_visible}"
export STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M="${STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M:-2.0}"
export STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M="${STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M:-0.5}"
export STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M="${STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M:-6.0}"
export STAGE3_EVAL_ALLOW_PRIVILEGED="${STAGE3_EVAL_ALLOW_PRIVILEGED:-0}"
export STAGE3_EVAL_RESUME="${STAGE3_EVAL_RESUME:-1}"
export STAGE3_EVAL_OVERWRITE="${STAGE3_EVAL_OVERWRITE:-0}"
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS="${STAGE3_EVAL_SAVE_TRAJECTORY_STEPS:-0}"
export STAGE3_EVAL_PREFLIGHT_ONLY="${STAGE3_EVAL_PREFLIGHT_ONLY:-0}"
export STAGE3_EVAL_RPC_REQUIRE_DETERMINISTIC_SAMPLING="${STAGE3_EVAL_RPC_REQUIRE_DETERMINISTIC_SAMPLING:-1}"

export STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER="${STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER:-0}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT="${STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT:-${FJL_ROOT}/data/heatmap_system1_dagger_v1}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND="${STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND:-0}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_GB="${STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_GB:-300}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_NORMAL_QUOTA="${STAGE3_EVAL_TRAJECTORY_DAGGER_NORMAL_QUOTA:-1}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_QUOTA="${STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_QUOTA:-2}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_JPEG_QUALITY="${STAGE3_EVAL_TRAJECTORY_DAGGER_JPEG_QUALITY:-75}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_OFFPATH_M="${STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_OFFPATH_M:-0.75}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_ORACLE_ACTIONS="${STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_ORACLE_ACTIONS:-128}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_MIN_HISTORY="${STAGE3_EVAL_TRAJECTORY_DAGGER_MIN_HISTORY:-2}"
export STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT="${STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT:-}"

export STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S="${STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S:-300}"
export STAGE3_EVAL_CHECKPOINT_SETTLE_S="${STAGE3_EVAL_CHECKPOINT_SETTLE_S:-30}"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export HEATMAPVLN_REQUIRE_FLASH_ATTN=1

RPC_PYTHONPATH="${RPC_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RPC_SERVER_ADDR="${STAGE3_EVAL_RPC_HOST}:${STAGE3_EVAL_RPC_PORT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${STAGE3_EVAL_LOG_DIR:-${REPO_ROOT}/logs}"
SERVER_LOG="${STAGE3_EVAL_SERVER_LOG:-${LOG_DIR}/stage3_r2r_rpc_server_${RUN_STAMP}.log}"
CLIENT_LOG="${STAGE3_EVAL_CLIENT_LOG:-${LOG_DIR}/stage3_r2r_val_unseen_${RUN_STAMP}.log}"
SERVER_PID=""

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

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

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] Stopping RPC model server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$VLNCE_PYTHON" ]]; then
  echo "Missing vlnce Python: $VLNCE_PYTHON" >&2
  exit 1
fi

validate_eval_checkpoint() {
  PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" \
    scripts/evaluation/preflight_stage3_rpc_eval.py \
    --config "$STAGE3_EVAL_CONFIG" \
    --base-checkpoint "$STAGE3_EVAL_BASE_CKPT" \
    --stage3-checkpoint "$STAGE3_EVAL_CHECKPOINT" \
    --expected-epoch "$STAGE3_EVAL_EXPECTED_EPOCH" \
    --expected-adapter-hidden-dim 1024
}

if is_true "$STAGE3_EVAL_RESUME" && is_true "$STAGE3_EVAL_OVERWRITE"; then
  echo "STAGE3_EVAL_RESUME and STAGE3_EVAL_OVERWRITE cannot both be enabled" >&2
  exit 1
fi
case "$STAGE3_EVAL_TRAJECTORY_X_SIGN" in
  -1|-1.0|1|1.0) ;;
  *)
    echo "STAGE3_EVAL_TRAJECTORY_X_SIGN must be -1 or 1" >&2
    exit 1
    ;;
esac
case "$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT" in
  none|pano_pixel) ;;
  *)
    echo "STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT must be none or pano_pixel" >&2
    exit 1
    ;;
esac
case "$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY" in
  farthest_visible|lookahead) ;;
  *)
    echo "STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY must be farthest_visible or lookahead" >&2
    exit 1
    ;;
esac
"$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M" \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M" \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M" <<'PY'
import math
import sys

names = ("lookahead_m", "min_ahead_m", "max_side_dist_m")
for name, raw in zip(names, sys.argv[1:]):
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"Oracle System2 {name} must be finite and > 0, got {raw!r}")
PY

privileged_requested="$($QWEN25_PYTHON - "$STAGE3_EVAL_AUTO_STOP_DISTANCE" <<'PY'
import sys
print(int(float(sys.argv[1]) > 0.0))
PY
)"
if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
  privileged_requested=1
fi
if [[ "$privileged_requested" == "1" ]] && ! is_true "$STAGE3_EVAL_ALLOW_PRIVILEGED"; then
  echo "Privileged evaluation requested (oracle System2 or auto_stop_distance > 0)." >&2
  echo "Main val_unseen evaluation must keep both disabled. Set STAGE3_EVAL_ALLOW_PRIVILEGED=1 only for a labelled diagnostic run." >&2
  exit 1
fi

if is_true "$STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER"; then
  if [[ "$STAGE3_EVAL_DATASET_SPLIT" != "train" ]]; then
    echo "Trajectory DAgger collection requires STAGE3_EVAL_DATASET_SPLIT=train" >&2
    exit 1
  fi
  if [[ "$privileged_requested" == "1" ]]; then
    echo "Trajectory DAgger learner rollout forbids oracle System2 and privileged auto-stop" >&2
    exit 1
  fi
  if ! is_true "$STAGE3_EVAL_RPC_REQUIRE_DETERMINISTIC_SAMPLING"; then
    echo "Trajectory DAgger collection requires deterministic RPC sampling" >&2
    exit 1
  fi
  case "$STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT" in
    /mnt/afs/liwenhao/agent/370910109|/mnt/afs/liwenhao/agent/370910109/*) ;;
    *)
      echo "Trajectory DAgger root must stay under /mnt/afs/liwenhao/agent/370910109" >&2
      exit 1
      ;;
  esac
  case "$STAGE3_EVAL_OUTPUT_PATH" in
    "$STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT"|"$STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT"/*)
      echo "Evaluation output must not be inside the capacity-guarded DAgger root" >&2
      exit 1
      ;;
  esac
fi
require_file "$STAGE3_EVAL_CONFIG"
require_file "$STAGE3_EVAL_BASE_CKPT"
require_file "$STAGE3_EVAL_DATA_PATH"
require_dir "$STAGE3_EVAL_SCENES_DIR"
require_dir "$INTERNNAV_MODEL_PATH"
require_dir "$RPC_ROOT/src/vla_rpc"
if [[ -n "$STAGE3_EVAL_EPISODE_LIST" ]]; then
  require_file "$STAGE3_EVAL_EPISODE_LIST"
fi
gzip -t "$STAGE3_EVAL_DATA_PATH"

scene_count="$(find -L "$STAGE3_EVAL_SCENES_DIR" -type f -name '*.glb' | wc -l | tr -d ' ')"
if [[ "$scene_count" -lt 90 ]]; then
  echo "MP3D scene preflight failed: found ${scene_count}, expected at least 90 in $STAGE3_EVAL_SCENES_DIR" >&2
  exit 1
fi

"$VLNCE_PYTHON" - "$STAGE3_EVAL_DATA_PATH" "$STAGE3_EVAL_SCENES_DIR" \
  "$STAGE3_EVAL_EXPECTED_EPISODES" "$STAGE3_EVAL_DATASET_SPLIT" <<'PY'
import gzip
import json
import sys
from pathlib import Path

data_path, scenes_dir = map(Path, sys.argv[1:3])
expected_episodes, dataset_split = int(sys.argv[3]), sys.argv[4]
with gzip.open(data_path, "rt", encoding="utf-8") as handle:
    episodes = json.load(handle).get("episodes", [])
if len(episodes) != expected_episodes:
    raise SystemExit(f"Expected {expected_episodes} R2R {dataset_split} episodes, found {len(episodes)}")
scene_id = str(episodes[0].get("scene_id", ""))
scene_asset = scenes_dir / scene_id
if not scene_asset.is_file():
    raise SystemExit(
        "Scene root is incompatible with dataset scene ids: "
        f"root={scenes_dir} first_scene_id={scene_id!r} expected={scene_asset}"
    )
print(f"Dataset/scene preflight passed: episodes={len(episodes)} first_scene={scene_asset}")
PY

for python_bin in "$QWEN25_PYTHON" "$VLNCE_PYTHON"; do
  PYTHONPATH="$RPC_PYTHONPATH" "$python_bin" - <<'PY'
from vla_rpc.client import VLAClient
from vla_rpc.core.image import decode_jpeg_to_rgb, encode_rgb_to_jpeg
from vla_rpc.proto import vla_pb2, vla_pb2_grpc

assert hasattr(VLAClient, "infer_json")
print("RPC API import passed")
PY
done

if ! DISPLAY="$STAGE3_EVAL_DISPLAY" timeout 10 xdpyinfo >/dev/null 2>&1; then
  echo "Xvfb/GLX display is unavailable: DISPLAY=$STAGE3_EVAL_DISPLAY" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Waiting for final Stage3 checkpoint: $STAGE3_EVAL_CHECKPOINT"
while true; do
  if [[ -s "$STAGE3_EVAL_CHECKPOINT" ]] && validate_eval_checkpoint; then
    size_before="$(stat -c %s "$STAGE3_EVAL_CHECKPOINT")"
    echo "[$(date '+%F %T')] Checkpoint passed preflight; waiting ${STAGE3_EVAL_CHECKPOINT_SETTLE_S}s for size stability"
    sleep "$STAGE3_EVAL_CHECKPOINT_SETTLE_S"
    size_after="$(stat -c %s "$STAGE3_EVAL_CHECKPOINT")"
    if [[ "$size_before" == "$size_after" ]] && validate_eval_checkpoint; then
      break
    fi
    echo "[$(date '+%F %T')] Checkpoint changed during settle window; continuing to wait"
  fi
  echo "[$(date '+%F %T')] Final Stage3 checkpoint not ready; retrying in ${STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S}s"
  sleep "$STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S"
done

if is_true "$STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER" && [[ -z "$STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT" ]]; then
  base_sha256="$(sha256sum "$STAGE3_EVAL_BASE_CKPT" | cut -d' ' -f1)"
  stage3_sha256="$(sha256sum "$STAGE3_EVAL_CHECKPOINT" | cut -d' ' -f1)"
  export STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT="base:${base_sha256};stage3:${stage3_sha256}"
fi
PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_RPC_HOST" "$STAGE3_EVAL_RPC_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
finally:
    sock.close()
print(f"RPC port is available: {host}:{port}")
PY

echo "[stage3-eval] code_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[stage3-eval] base=$STAGE3_EVAL_BASE_CKPT"
echo "[stage3-eval] stage3=$STAGE3_EVAL_CHECKPOINT"
echo "[stage3-eval] scenes=$STAGE3_EVAL_SCENES_DIR (${scene_count})"
echo "[stage3-eval] data=$STAGE3_EVAL_DATA_PATH"
echo "[stage3-eval] rpc=$RPC_SERVER_ADDR rpc_root=$RPC_ROOT"
echo "[stage3-eval] model_gpu=$STAGE3_EVAL_MODEL_GPU display=$STAGE3_EVAL_DISPLAY"
echo "[stage3-eval] output=$STAGE3_EVAL_OUTPUT_PATH"
echo "[stage3-eval] auto_stop=$STAGE3_EVAL_AUTO_STOP_DISTANCE oracle_system2=$STAGE3_EVAL_ORACLE_SYSTEM2"
echo "[stage3-eval] oracle_strategy=$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY lookahead_m=$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M min_ahead_m=$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M max_side_dist_m=$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"
echo "[stage3-eval] trajectory_selection=$STAGE3_EVAL_TRAJECTORY_SELECTION trajectory_x_sign=$STAGE3_EVAL_TRAJECTORY_X_SIGN heading_alignment=$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"
echo "[stage3-eval] pano_recenter_before_system1=$STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1"

if is_true "$STAGE3_EVAL_PREFLIGHT_ONLY"; then
  echo "[$(date '+%F %T')] STAGE3_EVAL_PREFLIGHT_ONLY=1; all static preflights passed"
  exit 0
fi

mkdir -p "$LOG_DIR" "$STAGE3_EVAL_OUTPUT_PATH"

env \
  PYTHONPATH="$RPC_PYTHONPATH" \
  CUDA_VISIBLE_DEVICES="$STAGE3_EVAL_MODEL_GPU" \
  USE_TF=0 \
  TRANSFORMERS_NO_TF=1 \
  TF_CPP_MIN_LOG_LEVEL=3 \
  HEATMAPVLN_REQUIRE_FLASH_ATTN=1 \
  "$QWEN25_PYTHON" -u scripts/evaluation/rpc_model_server.py \
    --config "$STAGE3_EVAL_CONFIG" \
    --base_checkpoint "$STAGE3_EVAL_BASE_CKPT" \
    --pano_latent_adapter_checkpoint "$STAGE3_EVAL_CHECKPOINT" \
    --internnav_model_path "$INTERNNAV_MODEL_PATH" \
    --gpu_id 0 \
    --host "$STAGE3_EVAL_RPC_HOST" \
    --port "$STAGE3_EVAL_RPC_PORT" \
    --workers 1 \
    --log_level INFO \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
echo "[$(date '+%F %T')] RPC model server starting pid=$SERVER_PID log=$SERVER_LOG"

server_ready=0
start_time="$(date +%s)"
while [[ "$(($(date +%s) - start_time))" -lt "$STAGE3_EVAL_SERVER_START_TIMEOUT_S" ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "RPC model server exited during startup" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    wait "$SERVER_PID" || true
    exit 1
  fi
  if PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "$RPC_SERVER_ADDR" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient

client = VLAClient(server_addr=sys.argv[1], timeout_ms=5000)
try:
    client.connect()
    info = client.get_server_info()
    if not client.health_check() or info is None:
        raise SystemExit(1)
    if info.version != "heatmapvln-r2r-json-v3":
        raise SystemExit(2)
finally:
    client.close()
PY
  then
    server_ready=1
    break
  fi
  sleep 10
done

if [[ "$server_ready" != "1" ]]; then
  echo "RPC model server was not healthy within ${STAGE3_EVAL_SERVER_START_TIMEOUT_S}s" >&2
  tail -200 "$SERVER_LOG" >&2 || true
  exit 1
fi

for required_log in \
  "Verified complete frozen InternNav System1 for RPC evaluation: 608 tensors" \
  "Verified complete LoRA checkpoint match: 224 tensors" \
  "Base checkpoint LoRA-only guard: loading 224/224 tensors" \
  "Verified pano latent adapter: tensors=4 parameters=7344640 dim=3584 hidden_dim=1024"; do
  if ! grep -Fq "$required_log" "$SERVER_LOG"; then
    echo "RPC startup assertion missing from server log: $required_log" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done
echo "[$(date '+%F %T')] RPC model server is healthy and all model-load guards passed"

PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_OUTPUT_PATH/eval_manifest.json" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "code_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    ).stdout.strip() or "unknown",
    "config": os.environ["STAGE3_EVAL_CONFIG"],
    "base_checkpoint": os.environ["STAGE3_EVAL_BASE_CKPT"],
    "stage3_checkpoint": os.environ["STAGE3_EVAL_CHECKPOINT"],
    "expected_epoch": int(os.environ["STAGE3_EVAL_EXPECTED_EPOCH"]),
    "scenes_dir": os.environ["STAGE3_EVAL_SCENES_DIR"],
    "data_path": os.environ["STAGE3_EVAL_DATA_PATH"],
    "rpc_root": os.environ["RPC_ROOT"],
    "rpc_protocol": "heatmapvln-r2r-json-v3",
    "rpc_capability": "pano-two-phase-front-system1-v1",
    "auto_stop_distance": float(os.environ["STAGE3_EVAL_AUTO_STOP_DISTANCE"]),
    "oracle_system2": os.environ["STAGE3_EVAL_ORACLE_SYSTEM2"].lower() in {"1", "true", "yes", "on"},
    "oracle_system2_strategy": os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY"],
    "oracle_system2_lookahead_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M"]),
    "oracle_system2_min_ahead_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M"]),
    "oracle_system2_max_side_dist_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"]),
    "trajectory_selection": os.environ["STAGE3_EVAL_TRAJECTORY_SELECTION"],
    "trajectory_x_sign": float(os.environ["STAGE3_EVAL_TRAJECTORY_X_SIGN"]),
    "trajectory_heading_alignment": os.environ["STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"],
    "system1_coord_order": os.environ["STAGE3_EVAL_SYSTEM1_COORD_ORDER"],
    "pano_recenter_before_system1": os.environ[
        "STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1"
    ].lower() in {"1", "true", "yes", "on"},
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

client_args=(
  --config "$STAGE3_EVAL_CONFIG"
  --rpc_server "$RPC_SERVER_ADDR"
  --rpc_timeout_ms "$STAGE3_EVAL_RPC_TIMEOUT_MS"
  --rpc_jpeg_quality "$STAGE3_EVAL_RPC_JPEG_QUALITY"
  --scenes_dir "$STAGE3_EVAL_SCENES_DIR"
  --data_path "$STAGE3_EVAL_DATA_PATH"
  --dataset_split "$STAGE3_EVAL_DATASET_SPLIT"
  --output_path "$STAGE3_EVAL_OUTPUT_PATH"
  --sim_gpu_id 0
  --resize_w 256
  --resize_h 256
  --num_history "$STAGE3_EVAL_NUM_HISTORY"
  --max_steps_per_episode "$STAGE3_EVAL_MAX_STEPS"
  --auto_stop_distance "$STAGE3_EVAL_AUTO_STOP_DISTANCE"
  --max_system2_calls_per_episode "$STAGE3_EVAL_MAX_SYSTEM2_CALLS"
  --trajectory_selection "$STAGE3_EVAL_TRAJECTORY_SELECTION"
  --trajectory_x_sign "$STAGE3_EVAL_TRAJECTORY_X_SIGN"
  --trajectory_heading_alignment "$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"
  --system1_coord_order "$STAGE3_EVAL_SYSTEM1_COORD_ORDER"
  --no-debug_input_trace
  --debug_save_input_images 0
)
if is_true "$STAGE3_EVAL_RPC_REQUIRE_DETERMINISTIC_SAMPLING"; then
  client_args+=(--rpc_require_deterministic_sampling)
fi
if is_true "$STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER"; then
  client_args+=(
    --collect_trajectory_dagger
    --trajectory_dagger_root "$STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT"
    --trajectory_dagger_round "$STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND"
    --trajectory_dagger_max_gb "$STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_GB"
    --trajectory_dagger_normal_quota "$STAGE3_EVAL_TRAJECTORY_DAGGER_NORMAL_QUOTA"
    --trajectory_dagger_hard_quota "$STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_QUOTA"
    --trajectory_dagger_jpeg_quality "$STAGE3_EVAL_TRAJECTORY_DAGGER_JPEG_QUALITY"
    --trajectory_dagger_hard_offpath_m "$STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_OFFPATH_M"
    --trajectory_dagger_max_oracle_actions "$STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_ORACLE_ACTIONS"
    --trajectory_dagger_min_history "$STAGE3_EVAL_TRAJECTORY_DAGGER_MIN_HISTORY"
    --trajectory_dagger_policy_fingerprint "$STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT"
  )
fi
if [[ -n "$STAGE3_EVAL_MAX_EPISODES" ]]; then
  client_args+=(--max_episodes "$STAGE3_EVAL_MAX_EPISODES")
fi
if [[ -n "$STAGE3_EVAL_EPISODE_LIST" ]]; then
  client_args+=(--episode_list "$STAGE3_EVAL_EPISODE_LIST")
fi
if is_true "$STAGE3_EVAL_RESUME"; then
  client_args+=(--resume)
fi
if is_true "$STAGE3_EVAL_OVERWRITE"; then
  client_args+=(--overwrite_output)
fi
if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
  client_args+=(
    --oracle_system2
    --oracle_system2_strategy "$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY"
    --oracle_system2_lookahead_m "$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M"
    --oracle_system2_min_ahead_m "$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M"
    --oracle_system2_max_side_dist_m "$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"
  )
fi
if is_true "$STAGE3_EVAL_SAVE_TRAJECTORY_STEPS"; then
  client_args+=(--save_trajectory_steps)
fi
if is_true "$STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1"; then
  client_args+=(--pano_recenter_before_system1)
else
  client_args+=(--no-pano_recenter_before_system1)
fi

echo "[$(date '+%F %T')] Starting Habitat val_unseen client log=$CLIENT_LOG"
env \
  PYTHONPATH="$RPC_PYTHONPATH" \
  DISPLAY="$STAGE3_EVAL_DISPLAY" \
  CUDA_VISIBLE_DEVICES=0 \
  HABITAT_GL_GPU_ID=0 \
  HEATMAPVLN_PREINIT_GL=0 \
  HEATMAPVLN_PREINIT_EMPTY_GL=1 \
  USE_TF=0 \
  TRANSFORMERS_NO_TF=1 \
  TF_CPP_MIN_LOG_LEVEL=3 \
  "$VLNCE_PYTHON" -u scripts/evaluation/r2r_val_unseen.py \
    "${client_args[@]}" \
    2>&1 | tee "$CLIENT_LOG"

require_file "$STAGE3_EVAL_OUTPUT_PATH/result.json"
echo "[$(date '+%F %T')] Stage3 R2R val_unseen evaluation complete"
echo "[stage3-eval] result=$STAGE3_EVAL_OUTPUT_PATH/result.json"
echo "[stage3-eval] progress=$STAGE3_EVAL_OUTPUT_PATH/progress.json"
echo "[stage3-eval] server_log=$SERVER_LOG"
echo "[stage3-eval] client_log=$CLIENT_LOG"
