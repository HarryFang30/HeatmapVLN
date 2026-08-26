#!/usr/bin/env bash
set -Eeuo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
REPO="$FJL_ROOT/HeatmapVLN"
INTERNNAV_REPO="$FJL_ROOT/InternNav"
PLAN="$FJL_ROOT/evaluation_plans/heatmap_control_r2r_val_unseen_8gpu_20260804"
LOCKED_NATIVE_PLAN="$FJL_ROOT/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
QWEN_PYTHON="$FJL_ROOT/envs/qwen25/bin/python"
VLNCE_PYTHON="$FJL_ROOT/envs/vlnce/bin/python"
RPC_ROOT="$FJL_ROOT/rpc"

CONFIG="$PLAN/configs/heatmap_control_eval.yaml"
NATIVE_MODEL="$FJL_ROOT/InternNav-Model"
CONTROL_SERVER="$REPO/scripts/evaluation/rpc_candidate_support_server.py"
CONTROL_CLIENT="${AUDIT_CONTROL_CLIENT:-$REPO/scripts/evaluation/r2r_candidate_support_audit_client.py}"
AUDIT_SUMMARIZER="${AUDIT_SUMMARIZER_SCRIPT:-$REPO/scripts/evaluation/summarize_candidate_support_audit.py}"
AUDIT_SUMMARY_BASENAME="${AUDIT_SUMMARY_BASENAME:-candidate_support_summary.json}"
BALANCED_COHORT_TOOL="$REPO/scripts/evaluation/build_candidate_audit_cohorts.py"
LOCKED_RPC_PROTOCOL="$PLAN/tools/locked_rpc_protocol.py"
EXPECTED_LOCKED_RPC_PROTOCOL_SHA256=7980e66bec2d26d2e496257facb138e53507a5df816311c78ad03095966eb029
NATIVE_MODEL_MANIFEST="$LOCKED_NATIVE_PLAN/manifests/internnav_model.sha256"
EXPECTED_NATIVE_MODEL_MANIFEST_SHA256=f37a6df2e0703e38c34ccdba89c861bb8490ad3a36201bc1ec24a7509bf56581
HEATMAP_CHECKPOINT="${AUDIT_HEATMAP_CHECKPOINT:-$FJL_ROOT/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth}"
CONTROL_CHECKPOINT="${AUDIT_CONTROL_CHECKPOINT:-$FJL_ROOT/model/output_heatmap_system1_control_v1/runs/run_20260807_112540/checkpoints/epoch_003.pth}"
CONTROL_MODE=on
DEPLOYMENT_ARM="${AUDIT_DEPLOYMENT_ARM:-native}"
DATASET_SPLIT="${AUDIT_DATASET_SPLIT:-train}"
SCENES_DIR="$FJL_ROOT/habitat/VLN-CE/data/scene_datasets"
COHORT_VERIFIER="$PLAN/tools/verify_locked_8gpu_cohorts.py"
case "$DATASET_SPLIT" in
  train)
    DATASET="$FJL_ROOT/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
    COHORTS_DIR="$FJL_ROOT/data/heatmap_system1_training_v1/cohorts/round_000/full_train_8way_seed17"
    EXPECTED_DATASET_SHA256=340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea
    EXPECTED_DATASET_EPISODES=10819
    ;;
  val_unseen)
    DATASET="$FJL_ROOT/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
    COHORTS_DIR="$LOCKED_NATIVE_PLAN/cohorts"
    EXPECTED_DATASET_SHA256=1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3
    EXPECTED_DATASET_EPISODES=1839
    ;;
  *)
    echo "AUDIT_DATASET_SPLIT must be train or val_unseen; got: $DATASET_SPLIT" >&2
    exit 1
    ;;
esac
CUSTOM_COHORTS=0
CUSTOM_COHORT_EPISODES_PER_SHARD="${AUDIT_COHORT_EPISODES_PER_SHARD:-64}"
if [[ -n "${AUDIT_COHORTS_DIR:-}" ]]; then
  if [[ "$DATASET_SPLIT" != train ]]; then
    echo "AUDIT_COHORTS_DIR is currently supported only for train" >&2
    exit 1
  fi
  COHORTS_DIR="$AUDIT_COHORTS_DIR"
  case "$COHORTS_DIR" in
    "$FJL_ROOT"/*) ;;
    *) echo "AUDIT_COHORTS_DIR must stay under $FJL_ROOT" >&2; exit 1 ;;
  esac
  CUSTOM_COHORTS=1
fi
X11_BUNDLE="$FJL_ROOT/tools/x11_headless_bundle_ubuntu22_20260801_v4"
X11_BUNDLE_MANIFEST="$X11_BUNDLE/manifest.sha256"
XVFB_BIN="$X11_BUNDLE/bin/Xvfb"
XDPYINFO_BIN="$X11_BUNDLE/bin/xdpyinfo"
GLXINFO_BIN="$X11_BUNDLE/bin/glxinfo"
X11_DRI_PATH="$X11_BUNDLE/dri"
X11_FONT_PATH="$X11_BUNDLE/share/fonts/misc"
X11_XKB_PATH="$X11_BUNDLE/share/X11/xkb"
EVAL_X11_MODE="${EVAL_X11_MODE:-bundle}"

AUDIT_ROOT="${AUDIT_ROOT:-$FJL_ROOT/data/candidate_support_audit_v1/${DATASET_SPLIT}_${DEPLOYMENT_ARM}_seed42}"
OUTPUT_ROOT="${AUDIT_OUTPUT_ROOT:-$FJL_ROOT/model/candidate_support_audit_v1/${DATASET_SPLIT}_${DEPLOYMENT_ARM}_seed42}"
WORKERS_DIR="$OUTPUT_ROOT/workers"
MERGED_DIR="$OUTPUT_ROOT/summary"
JOB_TOKEN="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
RUN_STAMP="${EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_job${JOB_TOKEN}}"
RUNTIME_DIR="$OUTPUT_ROOT/runtime/$RUN_STAMP"
SMOKE_AUDIT_ROOT="$RUNTIME_DIR/candidate_audit_smoke"
SMOKE_AUDIT_SUMMARY="$RUNTIME_DIR/candidate_audit_smoke_summary.json"

GPU_CSV="${EVAL_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_DEVICES:-0,1,2,3,4,5,6,7}}}"
NUM_SHARDS="${AUDIT_NUM_SHARDS:-8}"
RPC_PORT_BASE="${EVAL_RPC_PORT_BASE:-51400}"
DISPLAY_BASE="${EVAL_DISPLAY_BASE:-280}"
RPC_TIMEOUT_MS="${EVAL_RPC_TIMEOUT_MS:-600000}"
SERVER_START_TIMEOUT_S="${EVAL_SERVER_START_TIMEOUT_S:-2400}"
SERVER_STAGGER_S="${EVAL_SERVER_STAGGER_S:-2}"
PROTOCOL_SEED=42
LP_THREADS="${EVAL_LP_NUM_THREADS:-8}"
SERVER_CPU_THREADS="${EVAL_SERVER_CPU_THREADS:-4}"
CLIENT_CPU_THREADS="${EVAL_CLIENT_CPU_THREADS:-1}"
MAX_EPISODES_PER_SHARD="${AUDIT_MAX_EPISODES_PER_SHARD:-32}"
MAX_GB_TOTAL="${AUDIT_MAX_GB_TOTAL:-80}"
MAX_GB_PER_SHARD="${AUDIT_MAX_GB_PER_SHARD:-10}"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export INTERNNAV_MODEL_PATH="$NATIVE_MODEL"
export INTERNNAV_BACKBONE="$INTERNNAV_MODEL_PATH"
export INTERNNAV_REPO
export HEATMAPVLN_REPO="$REPO"
export HEATMAPVLN_FJL_ROOT="$FJL_ROOT"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export HEATMAPVLN_REQUIRE_FLASH_ATTN=1

X11_TOOL_LD_LIBRARY_PATH="$X11_BUNDLE/lib:$LD_LIBRARY_PATH"
X11_CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:$LD_LIBRARY_PATH"
SERVER_LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib"

RPC_PYTHONPATH="$PLAN/tools:$RPC_ROOT/src:$REPO:$INTERNNAV_REPO${PYTHONPATH:+:$PYTHONPATH}"

declare -a GPUS=()
declare -a PORTS=()
declare -a DISPLAYS=()
declare -a SERVER_PIDS=()
declare -a XVFB_PIDS=()
declare -a CLIENT_PIDS=()
declare -A CLIENT_LOG_BY_PID=()
DIST_ENV_UNSET_ARGS=(
  -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE
  -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u NODE_RANK
  -u MASTER_ADDR -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS
  -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE
  -u OMPI_COMM_WORLD_LOCAL_RANK -u OMPI_COMM_WORLD_LOCAL_SIZE
  -u PMI_RANK -u PMI_SIZE -u PMIX_RANK
  -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NTASKS -u SLURM_NPROCS
  -u SLURM_STEP_ID -u SLURM_STEP_NUM_TASKS
)
GL_ENV_UNSET_ARGS=(-u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES)
SERVER_GL_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY
  -u LIBGL_DRIVERS_PATH -u LIBGL_ALWAYS_SOFTWARE -u GALLIUM_DRIVER
  -u MESA_LOADER_DRIVER_OVERRIDE -u MESA_SHADER_CACHE_DIR
  -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES
  -u EGL_PLATFORM -u PYOPENGL_PLATFORM -u VK_ICD_FILENAMES -u LD_PRELOAD
)

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${EVAL_SMOKE_ONLY:-0}" && is_true "${EVAL_SKIP_SMOKE:-0}"; then
  echo "EVAL_SMOKE_ONLY and EVAL_SKIP_SMOKE are mutually exclusive" >&2
  exit 1
fi
if [[ "$DEPLOYMENT_ARM" != native && "$DEPLOYMENT_ARM" != heatmap_control ]]; then
  echo "AUDIT_DEPLOYMENT_ARM must be native or heatmap_control; got: $DEPLOYMENT_ARM" >&2
  exit 1
fi
case "$AUDIT_ROOT" in "$FJL_ROOT"/*) ;; *) echo "AUDIT_ROOT must stay under $FJL_ROOT" >&2; exit 1;; esac
case "$OUTPUT_ROOT" in "$FJL_ROOT"/*) ;; *) echo "AUDIT_OUTPUT_ROOT must stay under $FJL_ROOT" >&2; exit 1;; esac
if ! awk -v total="$MAX_GB_TOTAL" -v per="$MAX_GB_PER_SHARD" \
  'BEGIN { exit !(total > 0 && total <= 300 && per > 0 && per * 8 <= total) }'; then
  echo "Require 0 < 8*AUDIT_MAX_GB_PER_SHARD <= AUDIT_MAX_GB_TOTAL <= 300" >&2
  exit 1
fi
if [[ ! "$MAX_EPISODES_PER_SHARD" =~ ^[0-9]+$ ]]; then
  echo "AUDIT_MAX_EPISODES_PER_SHARD must be a non-negative integer" >&2
  exit 1
fi
if [[ "$CUSTOM_COHORTS" -eq 1 ]] && {
  [[ ! "$CUSTOM_COHORT_EPISODES_PER_SHARD" =~ ^[1-9][0-9]*$ ]] ||
  [[ "$CUSTOM_COHORT_EPISODES_PER_SHARD" -lt 1 ]]
}; then
  echo "AUDIT_COHORT_EPISODES_PER_SHARD must be a positive integer" >&2
  exit 1
fi

require_file() {
  [[ -s "$1" ]] || { echo "Missing required non-empty file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${CLIENT_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${SERVER_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$REPO"

if [[ "${BASH_VERSINFO[0]}" -lt 5 || ( "${BASH_VERSINFO[0]}" -eq 5 && "${BASH_VERSINFO[1]}" -lt 1 ) ]]; then
  echo "Bash >= 5.1 is required for fail-fast child monitoring" >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 8 ]]; then
  echo "AUDIT_NUM_SHARDS must be in [1,8], got: $NUM_SHARDS" >&2
  exit 1
fi
if [[ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "EVAL_GPU_DEVICES must contain exactly 8 devices, got: $GPU_CSV" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -ne "$NUM_SHARDS" ]]; then
  echo "EVAL_GPU_DEVICES contains duplicates: $GPU_CSV" >&2
  exit 1
fi

for path in "$QWEN_PYTHON" "$VLNCE_PYTHON"; do
  [[ -x "$path" ]] || { echo "Missing executable Python: $path" >&2; exit 1; }
done
for path in "$CONFIG" "$CONTROL_SERVER" "$CONTROL_CLIENT" "$AUDIT_SUMMARIZER" "$BALANCED_COHORT_TOOL" "$LOCKED_RPC_PROTOCOL" "$NATIVE_MODEL_MANIFEST" "$COHORT_VERIFIER" "$DATASET" "$HEATMAP_CHECKPOINT" "$CONTROL_CHECKPOINT"; do
  require_file "$path"
done
HEATMAP_SHA256="${AUDIT_HEATMAP_SHA256:-$(sha256sum "$HEATMAP_CHECKPOINT" | awk '{print $1}')}"
CONTROL_SHA256="${AUDIT_CONTROL_SHA256:-$(sha256sum "$CONTROL_CHECKPOINT" | awk '{print $1}')}"
for path in "$SCENES_DIR" "$NATIVE_MODEL" "$INTERNNAV_REPO" "$RPC_ROOT/src/vla_rpc" "$X11_BUNDLE" "$X11_DRI_PATH" "$X11_FONT_PATH" "$X11_XKB_PATH"; do
  require_dir "$path"
done
if [[ "$EVAL_X11_MODE" != bundle ]]; then
  echo "EVAL_X11_MODE must be bundle for this locked evaluation plan, got: $EVAL_X11_MODE" >&2
  exit 1
fi
for path in "$XVFB_BIN" "$XDPYINFO_BIN" "$GLXINFO_BIN"; do
  [[ -x "$path" ]] || { echo "Missing bundled X11 executable: $path" >&2; exit 1; }
done
for path in "$X11_BUNDLE_MANIFEST" "$X11_DRI_PATH/swrast_dri.so" "$X11_BUNDLE/manifest.json"; do
  require_file "$path"
done
for command_name in timeout sha256sum gzip flock; do
  command -v "$command_name" >/dev/null || { echo "Missing command: $command_name" >&2; exit 1; }
done
if [[ "$SERVER_LD_LIBRARY_PATH" == *"$X11_BUNDLE"* ]]; then
  echo "Model-server LD_LIBRARY_PATH must not contain the X11/Mesa bundle" >&2
  exit 1
fi

mkdir -p "$AUDIT_ROOT" "$WORKERS_DIR" "$MERGED_DIR" "$RUNTIME_DIR/logs" "$RUNTIME_DIR/smoke"
exec {EVAL_LOCK_FD}>"$OUTPUT_ROOT/.eval.lock"
if ! flock -n "$EVAL_LOCK_FD"; then
  echo "Another evaluation is already using OUTPUT_ROOT=$OUTPUT_ROOT" >&2
  exit 1
fi
COMMON_RUNTIME="$RUNTIME_DIR/common"
mkdir -p \
  "$COMMON_RUNTIME/tmp" \
  "$COMMON_RUNTIME/xdg_cache" \
  "$COMMON_RUNTIME/xdg_runtime" \
  "$COMMON_RUNTIME/hf_home" \
  "$COMMON_RUNTIME/torch_extensions" \
  "$COMMON_RUNTIME/triton_cache" \
  "$COMMON_RUNTIME/matplotlib"
chmod 700 "$COMMON_RUNTIME/xdg_runtime"
export TMPDIR="$COMMON_RUNTIME/tmp"
export XDG_CACHE_HOME="$COMMON_RUNTIME/xdg_cache"
export XDG_RUNTIME_DIR="$COMMON_RUNTIME/xdg_runtime"
export HF_HOME="$COMMON_RUNTIME/hf_home"
export TORCH_EXTENSIONS_DIR="$COMMON_RUNTIME/torch_extensions"
export TRITON_CACHE_DIR="$COMMON_RUNTIME/triton_cache"
export MPLCONFIGDIR="$COMMON_RUNTIME/matplotlib"

echo "[eval] static native-model/control/config/data preflight"
echo "[eval] verifying locked X11/Mesa bundle"
"$QWEN_PYTHON" - <<'PY'
import os
import platform
import re

machine = platform.machine()
if machine != "x86_64":
    raise SystemExit(f"X11 bundle requires x86_64, got {machine}")
glibc = os.confstr("CS_GNU_LIBC_VERSION") or ""
match = re.search(r"(\d+)\.(\d+)", glibc)
if match is None:
    raise SystemExit(f"Could not determine compute-node glibc version: {glibc!r}")
version = tuple(map(int, match.groups()))
if version < (2, 35):
    raise SystemExit(f"X11 bundle requires glibc >= 2.35, got {glibc}")
print(f"X11 host ABI compatible: architecture={machine}, {glibc}")
PY
sha256sum -c "$X11_BUNDLE_MANIFEST" >/dev/null
echo "[eval] X11/Mesa bundle hash verification passed"
gzip -t "$DATASET"
[[ "$(sha256sum "$LOCKED_RPC_PROTOCOL" | awk '{print $1}')" == "$EXPECTED_LOCKED_RPC_PROTOCOL_SHA256" ]] || {
  echo "Locked RPC protocol SHA256 mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$NATIVE_MODEL_MANIFEST" | awk '{print $1}')" == "$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256" ]] || {
  echo "Native model manifest SHA256 mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$HEATMAP_CHECKPOINT" | awk '{print $1}')" == "$HEATMAP_SHA256" ]] || {
  echo "Heatmap checkpoint SHA256 mismatch after resolution" >&2
  exit 1
}
[[ "$(sha256sum "$CONTROL_CHECKPOINT" | awk '{print $1}')" == "$CONTROL_SHA256" ]] || {
  echo "Control checkpoint SHA256 mismatch after resolution" >&2
  exit 1
}
"$QWEN_PYTHON" -m py_compile \
  "$REPO/src/models/action/nextdit_action_head.py" \
  "$REPO/scripts/evaluation/candidate_support_audit.py" \
  "$BALANCED_COHORT_TOOL" \
  "$CONTROL_SERVER" \
  "$CONTROL_CLIENT" \
  "$AUDIT_SUMMARIZER"
if is_true "${AUDIT_SKIP_COHORT_VERIFY:-0}"; then
  echo "[eval] cohort-plan verification skipped for targeted development smoke"
elif [[ "$CUSTOM_COHORTS" -eq 1 ]]; then
  "$QWEN_PYTHON" "$BALANCED_COHORT_TOOL" \
    --dataset "$DATASET" \
    --output-dir "$COHORTS_DIR" \
    --num-shards "$NUM_SHARDS" \
    --episodes-per-shard "$CUSTOM_COHORT_EPISODES_PER_SHARD" \
    --verify-only
elif [[ "$DATASET_SPLIT" == val_unseen ]]; then
  "$QWEN_PYTHON" "$COHORT_VERIFIER" \
    --dataset "$DATASET" \
    --cohorts-dir "$COHORTS_DIR" \
    --expected-dataset-sha256 "$EXPECTED_DATASET_SHA256"
else
  "$QWEN_PYTHON" - "$DATASET" "$COHORTS_DIR/plan.json" \
    "$EXPECTED_DATASET_SHA256" "$EXPECTED_DATASET_EPISODES" "$NUM_SHARDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset, plan_path = Path(sys.argv[1]), Path(sys.argv[2])
expected_dataset_sha, expected_episodes, expected_shards = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
if hashlib.sha256(dataset.read_bytes()).hexdigest() != expected_dataset_sha:
    raise SystemExit("train dataset SHA256 mismatch")
plan = json.loads(plan_path.read_text(encoding="utf-8"))
if plan.get("schema") != "r2r-dagger-shard-plan-v1":
    raise SystemExit("wrong train cohort schema")
if plan.get("selected_episode_count") != expected_episodes or plan.get("num_shards") != expected_shards:
    raise SystemExit("train cohort size/count mismatch")
if plan.get("route_grouped") is not True or plan.get("dataset", {}).get("sha256") != expected_dataset_sha:
    raise SystemExit("train cohort route-group/dataset contract mismatch")
entries = plan.get("shards")
if not isinstance(entries, list) or len(entries) != expected_shards:
    raise SystemExit("train cohort entries incomplete")
for index, entry in enumerate(entries):
    path = plan_path.parent / f"shard_{index:02d}.json"
    if entry.get("index") != index or entry.get("file") != path.name:
        raise SystemExit(f"train cohort entry {index} mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
        raise SystemExit(f"train cohort shard {index} SHA256 mismatch")
print(json.dumps({"status": "passed", "split": "train", "episodes": expected_episodes, "shards": expected_shards}))
PY
fi

env "${DIST_ENV_UNSET_ARGS[@]}" "${SERVER_GL_ENV_UNSET_ARGS[@]}" \
  PYTHONPATH="$RPC_PYTHONPATH" \
  LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" \
  HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=0 \
  "$QWEN_PYTHON" -c 'import locked_rpc_protocol as protocol; from scripts.evaluation import rpc_candidate_support_server as server; assert server.CANDIDATE_EXPORT_PROTO_VERSION == "paired-candidate-export-v1"; assert server.PROTO_VERSION == protocol.HEATMAPVLN_RPC_PROTOCOL_VERSION == "heatmapvln-r2r-json-v2"; assert protocol.HEATMAPVLN_RPC_SAMPLING_PROTOCOL == "heatmapvln-nextdit-sha256-v1"; sources = server.build_candidate_source_manifest(server.HEATMAP_REPO); assert len(sources) == len(server.CANDIDATE_SOURCE_RELATIVE_PATHS); print(f"Candidate-export RPC/source preflight passed: files={len(sources)}")'
env "${DIST_ENV_UNSET_ARGS[@]}" "${GL_ENV_UNSET_ARGS[@]}" \
  PYTHONPATH="$RPC_PYTHONPATH" \
  LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH" \
  AUDIT_EXPECTED_SPLIT="$DATASET_SPLIT" \
  AUDIT_EXPECTED_DATA_PATH="$DATASET" \
  AUDIT_EXPECTED_SCENES_DIR="$SCENES_DIR" \
  LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  LIBGL_ALWAYS_SOFTWARE=1 \
  GALLIUM_DRIVER=llvmpipe \
  MESA_LOADER_DRIVER_OVERRIDE=swrast \
  HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=0 \
  "$VLNCE_PYTHON" -c 'import os; from types import SimpleNamespace; import magnum, habitat_sim, numpy as np; from scripts.evaluation import r2r_candidate_support_audit_client as client; from vla_rpc.client import VLAClient; expected = os.environ["AUDIT_EXPECTED_SPLIT"]; cfg = client.build_habitat_config(SimpleNamespace(dataset_split=expected, scenes_dir=os.environ["AUDIT_EXPECTED_SCENES_DIR"], data_path=os.environ["AUDIT_EXPECTED_DATA_PATH"], sim_gpu_id=0)); assert cfg.DATASET.SPLIT == expected, (cfg.DATASET.SPLIT, expected); probe = {"views": {}, "c2w": None, "capture_step": 0, "system2_call_index": 0}; assert client._sample_history_records([probe], 8) == [probe]; trajectories = np.zeros((2, 32, 3), dtype=np.float32); candidates = client.build_candidate_set(trajectories, heatmap_trajectories=trajectories, trajectory_to_actions=client._single_trajectory_to_native_actions); assert candidates.native_sample_total == candidates.heatmap_sample_total == 2; print(f"Habitat import/config/history/candidate preflight passed: split={expected}, magnum + habitat_sim + candidate audit client")'

echo "[eval] checking RPC ports ${RPC_PORT_BASE}..$((RPC_PORT_BASE + NUM_SHARDS - 1))"
"$QWEN_PYTHON" - "$RPC_PORT_BASE" "$NUM_SHARDS" <<'PY'
import socket
import sys

base, count = map(int, sys.argv[1:])
for port in range(base, base + count):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    finally:
        sock.close()
print(f"RPC ports available: {base}..{base + count - 1}")
PY

echo "[eval] starting or reusing 8 isolated Xvfb displays"
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  display_num=$((DISPLAY_BASE + rank))
  display_addr="127.0.0.1:${display_num}.0"
  DISPLAYS[$rank]="$display_addr"
  if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
    DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    if is_true "${EVAL_REUSE_XVFB:-0}"; then
      XVFB_PIDS[$rank]=""
      echo "[eval] rank=$rank explicitly reusing DISPLAY=$display_addr"
    else
      echo "DISPLAY=$display_addr is already active. Choose a unique EVAL_DISPLAY_BASE, " \
           "or set EVAL_REUSE_XVFB=1 only when intentional." >&2
      exit 1
    fi
  else
    xvfb_log="$RUNTIME_DIR/logs/xvfb_${rank}.log"
    xvfb_runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/xvfb"
    mkdir -p "$xvfb_runtime/.xkb-cache"
    (
      cd "$xvfb_runtime"
      exec 9<"$xvfb_runtime/.xkb-cache"
      exec env "${GL_ENV_UNSET_ARGS[@]}" \
        PATH="$X11_BUNDLE/bin:$PATH" \
        LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
        LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
        LIBGL_ALWAYS_SOFTWARE=1 \
        GALLIUM_DRIVER=llvmpipe \
        MESA_LOADER_DRIVER_OVERRIDE=swrast \
        LP_NUM_THREADS="$LP_THREADS" \
        "$XVFB_BIN" ":$display_num" \
        -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
        -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
    ) >"$xvfb_log" 2>&1 &
    XVFB_PIDS[$rank]="$!"
    ready=0
    for _ in $(seq 1 60); do
      if ! kill -0 "${XVFB_PIDS[$rank]}" 2>/dev/null; then
        echo "Xvfb rank=$rank exited during startup" >&2
        tail -100 "$xvfb_log" >&2 || true
        exit 1
      fi
      if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
        DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 1
    done
    [[ "$ready" == 1 ]] || { echo "Xvfb rank=$rank did not become ready" >&2; exit 1; }
  fi
  renderer="$(env "${GL_ENV_UNSET_ARGS[@]}" \
    LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
    DISPLAY="$display_addr" \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    timeout 120 "$GLXINFO_BIN" -B 2>/dev/null | \
    grep -F 'OpenGL renderer string:' | head -1 || true)"
  if [[ "${renderer,,}" != *llvmpipe* ]]; then
    echo "DISPLAY=$display_addr is not using llvmpipe: ${renderer:-missing renderer}" >&2
    exit 1
  fi
  echo "[eval] rank=$rank DISPLAY=$display_addr $renderer"
done

if is_true "${EVAL_PREFLIGHT_ONLY:-0}"; then
  echo "[eval] EVAL_PREFLIGHT_ONLY=1: static, bundle, dual-env, ports and 8x GLX checks passed"
  exit 0
fi

echo "[audit] starting 8 independent paired candidate RPC servers deployment_arm=$DEPLOYMENT_ARM"
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$rank]}"
  port=$((RPC_PORT_BASE + rank))
  PORTS[$rank]="$port"
  server_log="$RUNTIME_DIR/logs/server_${rank}.log"
  server_runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/server"
  mkdir -p \
    "$server_runtime/tmp" \
    "$server_runtime/xdg_cache" \
    "$server_runtime/xdg_runtime" \
    "$server_runtime/hf_home" \
    "$server_runtime/torch_extensions" \
    "$server_runtime/triton_cache" \
    "$server_runtime/matplotlib"
  chmod 700 "$server_runtime/xdg_runtime"
  env "${DIST_ENV_UNSET_ARGS[@]}" "${SERVER_GL_ENV_UNSET_ARGS[@]}" \
    PYTHONPATH="$RPC_PYTHONPATH" \
    LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TMPDIR="$server_runtime/tmp" \
    XDG_CACHE_HOME="$server_runtime/xdg_cache" \
    XDG_RUNTIME_DIR="$server_runtime/xdg_runtime" \
    HF_HOME="$server_runtime/hf_home" \
    TORCH_EXTENSIONS_DIR="$server_runtime/torch_extensions" \
    TRITON_CACHE_DIR="$server_runtime/triton_cache" \
    MPLCONFIGDIR="$server_runtime/matplotlib" \
    OMP_NUM_THREADS="$SERVER_CPU_THREADS" \
    MKL_NUM_THREADS="$SERVER_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$SERVER_CPU_THREADS" \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    HEATMAPVLN_REQUIRE_FLASH_ATTN=1 \
    "$QWEN_PYTHON" -u "$CONTROL_SERVER" \
      --config "$CONFIG" \
      --model_path "$NATIVE_MODEL" \
      --frozen_heatmap_checkpoint "$HEATMAP_CHECKPOINT" \
      --frozen_heatmap_sha256 "$HEATMAP_SHA256" \
      --control_checkpoint "$CONTROL_CHECKPOINT" \
      --control_checkpoint_sha256 "$CONTROL_SHA256" \
      --native_model_manifest "$NATIVE_MODEL_MANIFEST" \
      --native_model_manifest_sha256 "$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256" \
      --control_mode "$CONTROL_MODE" \
      --deployment_arm "$DEPLOYMENT_ARM" \
      --gpu_id 0 \
      --host 127.0.0.1 \
      --port "$port" \
      --workers 1 \
      --require_deterministic_sampling \
      --log_level INFO \
      >"$server_log" 2>&1 &
  SERVER_PIDS[$rank]="$!"
  echo "[eval] server rank=$rank gpu=$gpu port=$port pid=${SERVER_PIDS[$rank]} log=$server_log"
  sleep "$SERVER_STAGGER_S"
done

start_time="$(date +%s)"
declare -a SERVER_READY=()
for rank in $(seq 0 $((NUM_SHARDS - 1))); do SERVER_READY[$rank]=0; done
while true; do
  ready_count=0
  for rank in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ "${SERVER_READY[$rank]}" == 1 ]]; then
      ready_count=$((ready_count + 1))
      continue
    fi
    if ! kill -0 "${SERVER_PIDS[$rank]}" 2>/dev/null; then
      echo "RPC server rank=$rank exited during startup" >&2
      tail -200 "$RUNTIME_DIR/logs/server_${rank}.log" >&2 || true
      exit 1
    fi
    if env "${DIST_ENV_UNSET_ARGS[@]}" "${GL_ENV_UNSET_ARGS[@]}" PYTHONPATH="$RPC_PYTHONPATH" \
      "$VLNCE_PYTHON" - "127.0.0.1:${PORTS[$rank]}" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient

client = VLAClient(server_addr=sys.argv[1], timeout_ms=5000)
try:
    client.connect()
    info = client.get_server_info()
    if not client.health_check() or info is None or info.version != "heatmapvln-r2r-json-v2":
        raise SystemExit(1)
finally:
    client.close()
PY
    then
      SERVER_READY[$rank]=1
      ready_count=$((ready_count + 1))
      echo "[eval] server rank=$rank healthy"
    fi
  done
  [[ "$ready_count" -eq "$NUM_SHARDS" ]] && break
  if [[ $(( $(date +%s) - start_time )) -ge "$SERVER_START_TIMEOUT_S" ]]; then
    echo "RPC servers were not all ready within ${SERVER_START_TIMEOUT_S}s" >&2
    for rank in $(seq 0 $((NUM_SHARDS - 1))); do
      echo "--- server rank=$rank ---" >&2
      tail -80 "$RUNTIME_DIR/logs/server_${rank}.log" >&2 || true
    done
    exit 1
  fi
  sleep 10
done

for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  server_log="$RUNTIME_DIR/logs/server_${rank}.log"
  for required in \
    "Candidate source provenance verified: files=10" \
    "Native InternNav checkpoint index verified: tensors=1338 shards=4 lora=0 adapter=0" \
    "Native InternNav System1 strict load verified:" \
    "Frozen heatmap strict load verified:" \
    "Control EMA strict load verified:" \
    "Candidate-support audit mode: native_front_only_system2=True native_frozen_system1=True control_mode=${CONTROL_MODE} deployment_arm=${DEPLOYMENT_ARM} adapters=12 vlm_image_size=384 lookdown_vlm_size=640x480 traj_image_size=224" \
    "require_deterministic_sampling=True" \
    "Candidate-support RPC server listening on 127.0.0.1:${PORTS[$rank]}"; do
    if ! grep -Fq "$required" "$server_log"; then
      echo "Server rank=$rank missing startup assertion: $required" >&2
      tail -240 "$server_log" >&2 || true
      exit 1
    fi
  done
done
echo "[audit] all 8 servers passed native/frozen/EMA strict-load guards"

launch_clients() {
  local mode="$1"
  local output_base="$2"
  local candidate_audit_root="$AUDIT_ROOT"
  if [[ "$mode" == smoke ]]; then
    # Smoke data validates the pipeline but must never become part of the
    # resumable main counterfactual dataset.
    candidate_audit_root="$SMOKE_AUDIT_ROOT"
  fi
  CLIENT_PIDS=()
  CLIENT_LOG_BY_PID=()
  echo "[audit] client mode=$mode candidate_audit_root=$candidate_audit_root"
  for rank in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS[$rank]}"
    shard_output="$output_base/shard_$(printf '%02d' "$rank")"
    cohort="$COHORTS_DIR/shard_$(printf '%02d' "$rank").json"
    if [[ -s "$COHORTS_DIR/dataset_shard_$(printf '%02d' "$rank").json.gz" ]]; then
      dataset_shard="$COHORTS_DIR/dataset_shard_$(printf '%02d' "$rank").json.gz"
    else
      dataset_shard="$DATASET"
    fi
    client_log="$RUNTIME_DIR/logs/client_${mode}_${rank}.log"
    require_file "$cohort"
    require_file "$dataset_shard"
    gzip -t "$dataset_shard"
    mkdir -p "$shard_output"
    client_args=(
      --config "$CONFIG"
      --rpc_server "127.0.0.1:${PORTS[$rank]}"
      --rpc_timeout_ms "$RPC_TIMEOUT_MS"
      --rpc_jpeg_quality 90
      --rpc_protocol_seed "$PROTOCOL_SEED"
      --rpc_require_deterministic_sampling
      --scenes_dir "$SCENES_DIR"
      --data_path "$dataset_shard"
      --dataset_split "$DATASET_SPLIT"
      --output_path "$shard_output"
      --episode_list "$cohort"
      --sim_gpu_id 0
      --resize_w 384
      --resize_h 384
      --num_history 8
      --max_steps_per_episode 500
      --auto_stop_distance 0.0
      --max_system2_calls_per_episode 0
      --trajectory_selection mean
      --trajectory_x_sign 1
      --trajectory_heading_alignment none
      --expected_control_mode "$CONTROL_MODE"
      --expected_heatmap_checkpoint_sha256 "$HEATMAP_SHA256"
      --expected_control_checkpoint_sha256 "$CONTROL_SHA256"
      --expected_native_model_manifest_sha256 "$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256"
      --candidate_audit_root "$candidate_audit_root"
      --candidate_audit_shard_id "$rank"
      --candidate_audit_max_gb_per_shard "$MAX_GB_PER_SHARD"
      --candidate_audit_deployment_arm "$DEPLOYMENT_ARM"
      --candidate_audit_success_radius_m 3.0
      --system1_coord_order generated
      --no-debug_input_trace
      --debug_save_input_images 0
    )
    if [[ "$mode" == smoke ]]; then
      client_args+=(--max_episodes 1 --overwrite_output)
    else
      client_args+=(--resume)
      if [[ "$MAX_EPISODES_PER_SHARD" -gt 0 ]]; then
        client_args+=(--max_episodes "$MAX_EPISODES_PER_SHARD")
      fi
    fi
    client_runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/client_${mode}"
    mkdir -p \
      "$client_runtime/tmp" \
      "$client_runtime/xdg_cache" \
      "$client_runtime/xdg_runtime" \
      "$client_runtime/hf_home" \
      "$client_runtime/matplotlib" \
      "$client_runtime/mesa_shader_cache" \
      "$client_runtime/numba_cache"
    chmod 700 "$client_runtime/xdg_runtime"
    env "${DIST_ENV_UNSET_ARGS[@]}" "${GL_ENV_UNSET_ARGS[@]}" \
      PYTHONPATH="$RPC_PYTHONPATH" \
      LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH" \
      LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
      DISPLAY="${DISPLAYS[$rank]}" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      TMPDIR="$client_runtime/tmp" \
      XDG_CACHE_HOME="$client_runtime/xdg_cache" \
      XDG_RUNTIME_DIR="$client_runtime/xdg_runtime" \
      HF_HOME="$client_runtime/hf_home" \
      MPLCONFIGDIR="$client_runtime/matplotlib" \
      MESA_SHADER_CACHE_DIR="$client_runtime/mesa_shader_cache" \
      NUMBA_CACHE_DIR="$client_runtime/numba_cache" \
      HABITAT_GL_GPU_ID=0 \
      HEATMAPVLN_PREINIT_GL=0 \
      HEATMAPVLN_PREINIT_EMPTY_GL=1 \
      HEATMAPVLN_ALLOW_NVIDIA_GLX=0 \
      LIBGL_ALWAYS_SOFTWARE=1 \
      GALLIUM_DRIVER=llvmpipe \
      MESA_LOADER_DRIVER_OVERRIDE=swrast \
      LP_NUM_THREADS="$LP_THREADS" \
      OMP_NUM_THREADS="$CLIENT_CPU_THREADS" \
      MKL_NUM_THREADS="$CLIENT_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$CLIENT_CPU_THREADS" \
      NUMBA_NUM_THREADS="$CLIENT_CPU_THREADS" \
      CANDIDATE_COLLECTION_MODE="$mode" \
      USE_TF=0 \
      TRANSFORMERS_NO_TF=1 \
      TF_CPP_MIN_LOG_LEVEL=3 \
      "$VLNCE_PYTHON" -u "$CONTROL_CLIENT" \
        "${client_args[@]}" \
        >"$client_log" 2>&1 &
    pid="$!"
    CLIENT_PIDS+=("$pid")
    CLIENT_LOG_BY_PID[$pid]="$client_log"
    echo "[eval] client mode=$mode rank=$rank gpu=$gpu display=${DISPLAYS[$rank]} pid=$pid log=$client_log"
  done
}

wait_for_clients() {
  local -a active=("${CLIENT_PIDS[@]}")
  local finished_pid=""
  local rc=0
  while [[ "${#active[@]}" -gt 0 ]]; do
    finished_pid=""
    if wait -n -p finished_pid "${active[@]}" "${SERVER_PIDS[@]}"; then
      rc=0
    else
      rc=$?
    fi
    if [[ -z "$finished_pid" ]]; then
      echo "wait -n did not report a child pid" >&2
      return 1
    fi
    for rank in $(seq 0 $((NUM_SHARDS - 1))); do
      if [[ "$finished_pid" == "${SERVER_PIDS[$rank]}" ]]; then
        echo "RPC server rank=$rank pid=$finished_pid exited while clients were running (rc=$rc)" >&2
        tail -240 "$RUNTIME_DIR/logs/server_${rank}.log" >&2 || true
        return 1
      fi
    done
    local -a remaining=()
    local matched_client=0
    for pid in "${active[@]}"; do
      if [[ "$pid" == "$finished_pid" ]]; then
        matched_client=1
      else
        remaining+=("$pid")
      fi
    done
    if [[ "$matched_client" -ne 1 ]]; then
      echo "wait -n reported unknown child pid=$finished_pid" >&2
      return 1
    fi
    active=("${remaining[@]}")
    CLIENT_PIDS=("${active[@]}")
    if [[ "$rc" -ne 0 ]]; then
      echo "Client pid=$finished_pid failed with rc=$rc" >&2
      tail -240 "${CLIENT_LOG_BY_PID[$finished_pid]}" >&2 || true
      return "$rc"
    fi
    echo "[eval] client pid=$finished_pid completed; remaining=${#active[@]}"
  done
  CLIENT_PIDS=()
  return 0
}

if ! is_true "${EVAL_SKIP_SMOKE:-0}"; then
  echo "[eval] running one deterministic episode per shard as an 8-way end-to-end smoke"
  launch_clients smoke "$RUNTIME_DIR/smoke"
  wait_for_clients
  "$QWEN_PYTHON" "$PLAN/tools/verify_smoke.py" \
    --root "$RUNTIME_DIR/smoke" \
    --num-shards "$NUM_SHARDS" \
    --protocol heatmapvln-r2r-json-v2 \
    --protocol-seed "$PROTOCOL_SEED"
  "$QWEN_PYTHON" "$AUDIT_SUMMARIZER" \
    --audit-root "$SMOKE_AUDIT_ROOT" \
    --output "$SMOKE_AUDIT_SUMMARY" \
    --expected-shards "$NUM_SHARDS" \
    >"$RUNTIME_DIR/logs/candidate_audit_smoke_summary.log"
  "$QWEN_PYTHON" - "$SMOKE_AUDIT_ROOT" "$SMOKE_AUDIT_SUMMARY" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
num_shards = int(sys.argv[3])
manifests = [
    json.loads((root / f"shard_{index:02d}" / "manifest.json").read_text(encoding="utf-8"))
    for index in range(num_shards)
]
record_counts = [int(manifest["record_count"]) for manifest in manifests]
if any(count <= 0 for count in record_counts):
    raise SystemExit(f"candidate smoke did not exercise every shard: {record_counts}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary["storage"]["shards"]) != num_shards:
    raise SystemExit("candidate smoke summary shard count mismatch")
if int(summary["storage"]["records"]) != sum(record_counts):
    raise SystemExit("candidate smoke summary record count mismatch")
print(
    json.dumps(
        {
            "status": "passed",
            "candidate_record_counts": record_counts,
            "candidate_records": sum(record_counts),
            "candidate_array_bytes": int(
                summary["storage"]["compressed_array_bytes"]
            ),
        },
        sort_keys=True,
    )
)
PY
  echo "[eval] 8-way navigation + candidate-dataset end-to-end smoke passed"
fi

if is_true "${EVAL_SMOKE_ONLY:-0}"; then
  echo "[eval] EVAL_SMOKE_ONLY=1: stopping after successful 8-way smoke"
  exit 0
fi

if [[ "$MAX_EPISODES_PER_SHARD" -gt 0 ]]; then
  echo "[audit] starting bounded main audit: up to $MAX_EPISODES_PER_SHARD episodes per shard"
elif [[ "$CUSTOM_COHORTS" -eq 1 ]]; then
  echo "[audit] starting complete balanced cohort: $CUSTOM_COHORT_EPISODES_PER_SHARD episodes per shard"
else
  echo "[audit] starting full ${EXPECTED_DATASET_EPISODES}-episode $DATASET_SPLIT candidate audit"
fi
launch_clients full "$WORKERS_DIR"
wait_for_clients

"$QWEN_PYTHON" "$AUDIT_SUMMARIZER" \
  --audit-root "$AUDIT_ROOT" \
  --output "$MERGED_DIR/$AUDIT_SUMMARY_BASENAME" \
  --expected-shards "$NUM_SHARDS"

echo "[audit] COMPLETE"
echo "[audit] compact_dataset=$AUDIT_ROOT"
echo "[audit] summary=$MERGED_DIR/$AUDIT_SUMMARY_BASENAME"
