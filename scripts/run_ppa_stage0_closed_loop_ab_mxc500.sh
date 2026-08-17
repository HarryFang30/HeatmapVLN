#!/usr/bin/env bash
# Exact-zero Past->Plan bridge: native-baseline vs treatment closed-loop gate.

set -Eeuo pipefail

REPO_ROOT="${PPA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FJL_ROOT="${PPA_ALLOWED_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
QWEN_PYTHON="${PPA_QWEN_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
VLNCE_PYTHON="${PPA_VLNCE_PYTHON:-${FJL_ROOT}/envs/vlnce/bin/python}"
RPC_ROOT="${PPA_RPC_ROOT:-${FJL_ROOT}/rpc}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${FJL_ROOT}/InternNav-Model}"
CONFIG="${PPA_STAGE0_CONFIG:-${REPO_ROOT}/configs/ppa_stage1_map_pretrain_4gpu.yaml}"
CHECKPOINT="${PPA_STAGE0_CHECKPOINT:-}"
COHORT="${PPA_STAGE0_COHORT:-${REPO_ROOT}/configs/eval_cohorts/ppa_stage0_fixed4_seed42.json}"
SCENES_DIR="${PPA_STAGE0_SCENES_DIR:-${FJL_ROOT}/habitat/VLN-CE/data/scene_datasets}"
DATA_PATH="${PPA_STAGE0_DATA_PATH:-${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz}"
MODEL_GPU="${PPA_STAGE0_MODEL_GPU:-0}"
DISPLAY_VALUE="${PPA_STAGE0_DISPLAY:-localhost:200.0}"
RPC_HOST="${PPA_STAGE0_RPC_HOST:-127.0.0.1}"
RPC_PORT="${PPA_STAGE0_RPC_PORT:-50131}"
PROTOCOL_SEED="${PPA_STAGE0_PROTOCOL_SEED:-20260813}"
MAX_STEPS="${PPA_STAGE0_MAX_STEPS:-80}"
MAX_SYSTEM2_CALLS="${PPA_STAGE0_MAX_SYSTEM2_CALLS:-20}"
START_TIMEOUT="${PPA_STAGE0_SERVER_START_TIMEOUT_S:-1800}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${PPA_STAGE0_OUTPUT_ROOT:-${FJL_ROOT}/model/ppa_stage0_closed_loop_ab/${RUN_STAMP}}"
BASELINE_OUT="${OUTPUT_ROOT}/baseline"
TREATMENT_OUT="${OUTPUT_ROOT}/treatment"
REPORT="${OUTPUT_ROOT}/stage0_closed_loop_ab_report.json"
LOG_DIR="${OUTPUT_ROOT}/logs"
RPC_ADDRESS="${RPC_HOST}:${RPC_PORT}"
SERVER_PID=""

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export HEATMAPVLN_REQUIRE_FLASH_ATTN=1
export INTERNNAV_MODEL_PATH
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-${MACA_HOME}}"
export MACA_DIR="${MACA_DIR:-${MACA_PATH}}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
RPC_PYTHONPATH="${RPC_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

die() {
  echo "[ppa-stage0] $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || die "missing non-empty file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
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
  SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT TERM

[[ -n "$CHECKPOINT" ]] || die "set PPA_STAGE0_CHECKPOINT to one exact-zero PPA checkpoint"
[[ -x "$QWEN_PYTHON" ]] || die "missing qwen Python: $QWEN_PYTHON"
[[ -x "$VLNCE_PYTHON" ]] || die "missing vlnce Python: $VLNCE_PYTHON"
require_file "$CONFIG"
require_file "$CHECKPOINT"
require_file "$COHORT"
require_file "$DATA_PATH"
require_dir "$SCENES_DIR"
require_dir "$INTERNNAV_MODEL_PATH"
require_dir "$RPC_ROOT/src/vla_rpc"
require_file "$REPO_ROOT/scripts/evaluation/rpc_model_server.py"
require_file "$REPO_ROOT/scripts/evaluation/r2r_val_unseen.py"
require_file "$REPO_ROOT/scripts/evaluation/compare_ppa_stage0_closed_loop_ab.py"
require_file "$REPO_ROOT/src/models/action/treatment_spec.py"

case "$OUTPUT_ROOT" in
  "$FJL_ROOT"/*) ;;
  *) die "PPA_STAGE0_OUTPUT_ROOT must stay under $FJL_ROOT" ;;
esac
[[ ! -e "$OUTPUT_ROOT" ]] || die "refusing existing output root: $OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

if ! DISPLAY="$DISPLAY_VALUE" timeout 10 xdpyinfo >/dev/null 2>&1; then
  die "Xvfb/GLX display unavailable: DISPLAY=$DISPLAY_VALUE"
fi
gzip -t "$DATA_PATH"

COHORT_COUNT="$($QWEN_PYTHON - "$CONFIG" "$COHORT" "$DATA_PATH" <<'PY'
import gzip
import json
import sys
from pathlib import Path

import yaml

config_path, cohort_path, data_path = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text())
ppa = cfg.get("model", {}).get("past_plan_action", {})
if not ppa.get("enabled", False):
    raise SystemExit("config must enable model.past_plan_action")
heatmap_control = (
    cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("heatmap_control", {})
)
if (heatmap_control or {}).get("enabled", False):
    raise SystemExit("Stage-0 forbids legacy heatmap control")
if cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("pano_latent_adapter", {}).get("enabled", False):
    raise SystemExit("Stage-0 forbids pano latent adapter")
cohort = json.loads(cohort_path.read_text()).get("episodes", [])
if not cohort:
    raise SystemExit("empty Stage-0 cohort")
keys = [(str(row["scene_id"]), int(row["episode_id"])) for row in cohort]
if len(keys) != len(set(keys)):
    raise SystemExit("duplicate Stage-0 cohort episode")
with gzip.open(data_path, "rt", encoding="utf-8") as handle:
    dataset = json.load(handle).get("episodes", [])
available = {
    (str(row["scene_id"]).split("/")[-2], int(row["episode_id"]))
    for row in dataset
}
missing = sorted(set(keys) - available)
if missing:
    raise SystemExit(f"cohort episodes absent from dataset: {missing}")
print(len(keys))
PY
)"

CHECKPOINT_SHA_BEFORE="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
CONFIG_SHA="$(sha256sum "$CONFIG" | cut -d' ' -f1)"
COHORT_SHA="$(sha256sum "$COHORT" | cut -d' ' -f1)"

"$QWEN_PYTHON" - "$RPC_HOST" "$RPC_PORT" <<'PY'
import socket
import sys
sock = socket.socket()
try:
    sock.bind((sys.argv[1], int(sys.argv[2])))
finally:
    sock.close()
PY

run_arm() {
  local arm="$1"
  local output="$2"
  local server_log="${LOG_DIR}/${arm}_server.log"
  local client_log="${LOG_DIR}/${arm}_client.log"

  env \
    PYTHONPATH="$RPC_PYTHONPATH" \
    CUDA_VISIBLE_DEVICES="$MODEL_GPU" \
    "$QWEN_PYTHON" -u "$REPO_ROOT/scripts/evaluation/rpc_model_server.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --internnav_model_path "$INTERNNAV_MODEL_PATH" \
      --gpu_id 0 \
      --host "$RPC_HOST" \
      --port "$RPC_PORT" \
      --workers 1 \
      --require_deterministic_sampling \
      --ppa_stage0_action_arm "$arm" \
      --log_level INFO \
      >"$server_log" 2>&1 &
  SERVER_PID="$!"

  local start_time
  start_time="$(date +%s)"
  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -200 "$server_log" >&2 || true
      die "$arm server exited during startup"
    fi
    if PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "$RPC_ADDRESS" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient
client = VLAClient(server_addr=sys.argv[1], timeout_ms=5000)
try:
    client.connect()
    info = client.get_server_info()
    if not client.health_check() or info is None or info.version != "heatmapvln-r2r-json-v3":
        raise SystemExit(1)
finally:
    client.close()
PY
    then
      break
    fi
    if [[ "$(($(date +%s) - start_time))" -ge "$START_TIMEOUT" ]]; then
      tail -200 "$server_log" >&2 || true
      die "$arm server did not become healthy in ${START_TIMEOUT}s"
    fi
    sleep 10
  done
  grep -Fq "PPA Stage-0 action arm=${arm}; exact-zero bridge verified" "$server_log" || {
    tail -200 "$server_log" >&2 || true
    die "$arm server did not prove exact-zero bridge"
  }

  DISPLAY="$DISPLAY_VALUE" PYTHONPATH="$RPC_PYTHONPATH" \
    "$VLNCE_PYTHON" -u "$REPO_ROOT/scripts/evaluation/r2r_val_unseen.py" \
      --config "$CONFIG" \
      --rpc_server "$RPC_ADDRESS" \
      --rpc_timeout_ms 600000 \
      --rpc_jpeg_quality 90 \
      --rpc_protocol_seed "$PROTOCOL_SEED" \
      --rpc_require_deterministic_sampling \
      --scenes_dir "$SCENES_DIR" \
      --data_path "$DATA_PATH" \
      --dataset_split val_unseen \
      --output_path "$output" \
      --sim_gpu_id 0 \
      --resize_w 256 \
      --resize_h 256 \
      --num_history 8 \
      --max_steps_per_episode "$MAX_STEPS" \
      --max_system2_calls_per_episode "$MAX_SYSTEM2_CALLS" \
      --auto_stop_distance 0 \
      --trajectory_selection mean \
      --trajectory_x_sign 1 \
      --trajectory_heading_alignment none \
      --system1_coord_order generated \
      --no-debug_input_trace \
      --episode_list "$COHORT" \
      --max_episodes "$COHORT_COUNT" \
      --overwrite_output \
      2>&1 | tee "$client_log"

  stop_server
  require_file "$output/progress.json"
  require_file "$output/result.json"
}

echo "[ppa-stage0] checkpoint_sha256=$CHECKPOINT_SHA_BEFORE"
echo "[ppa-stage0] config_sha256=$CONFIG_SHA cohort_sha256=$COHORT_SHA episodes=$COHORT_COUNT"
run_arm baseline "$BASELINE_OUT"
run_arm treatment "$TREATMENT_OUT"

CHECKPOINT_SHA_AFTER="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
[[ "$CHECKPOINT_SHA_AFTER" == "$CHECKPOINT_SHA_BEFORE" ]] || die "checkpoint changed during A/B"

PYTHONPATH="$REPO_ROOT" "$QWEN_PYTHON" \
  "$REPO_ROOT/scripts/evaluation/compare_ppa_stage0_closed_loop_ab.py" \
    --cohort "$COHORT" \
    --baseline-progress "$BASELINE_OUT/progress.json" \
    --treatment-progress "$TREATMENT_OUT/progress.json" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --report "$REPORT"

echo "[ppa-stage0] PASSED report=$REPORT"
