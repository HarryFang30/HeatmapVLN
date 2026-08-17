#!/usr/bin/bash -p
# Direct, foreground entrypoint for a web-scheduled one-node, eight-GPU
# native InternNav trajectory-DAgger collection. The web platform allocates
# the node; this script only manages the eight independent local workers.
# Each GPU runs an independent original InternNav System2 + System1 policy on
# one exact, route-disjoint R2R-train cohort shard. No torchrun is used.
# Re-submitting this script resumes unfinished shards and verifies sealed ones.
# Request one node with 8 GPUs (recommended: 128 CPU, 256GB RAM, 36h) in the
# web form, then run this script directly in the foreground. Do not use nohup
# or a top-level '&'. Set DAGGER_8GPU_PREPARE_ONLY=1 only for a no-GPU preflight.

set -Eeuo pipefail
set +m
unset BASH_ENV ENV
umask 077

# Keep every shell and subprocess on a fixed, administrator-owned command path.
# Privileged Bash mode above also prevents BASH_ENV and exported Bash functions
# from being imported before this script can sanitize the worker environment.
readonly FIXED_SYSTEM_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/maca-3.3.0/bin:/opt/maca-3.3.0/ompi/bin:/opt/mxdriver/bin"
export PATH="$FIXED_SYSTEM_PATH"

readonly ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
readonly REPO_ROOT="${ALLOWED_ROOT}/HeatmapVLN"
readonly PYTHON="${ALLOWED_ROOT}/envs/qwen25/bin/python"
readonly SHARD_BUILDER="${REPO_ROOT}/scripts/tools/build_r2r_train_dagger_shards.py"
readonly SHARD_FINALIZER="${REPO_ROOT}/scripts/tools/finalize_trajectory_dagger_shards.py"
readonly WORKER_WRAPPER="${REPO_ROOT}/scripts/run_heatmap_system1_dagger_collection_mxc500.sh"
readonly TRAIN_DATA="${ALLOWED_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
readonly VLNCE_PYTHON="${ALLOWED_ROOT}/envs/vlnce/bin/python"
readonly X11_BUNDLE="${ALLOWED_ROOT}/tools/x11_headless_bundle_ubuntu22_20260801_v4"
readonly X11_MANIFEST="${X11_BUNDLE}/manifest.sha256"
readonly X11_DRI_PATH="${X11_BUNDLE}/dri"
readonly X11_XKB_PATH="${X11_BUNDLE}/share/X11/xkb"
readonly X11_FONT_PATH="${X11_BUNDLE}/share/fonts/misc"
readonly SETSID_BIN="/usr/bin/setsid"
readonly FLOCK_BIN="/usr/bin/flock"
readonly BASH_BIN="/usr/bin/bash"
readonly ENV_BIN="/usr/bin/env"

readonly NUM_SHARDS=8
readonly TOTAL_EPISODES=10819
readonly SEED=17
readonly ABSOLUTE_TOTAL_LIMIT_BYTES=300000000000
readonly WRAPPER_RESERVE_BYTES=5000000000

RUN_TAG="${DAGGER_8GPU_RUN_TAG:-full_train_8way_seed17}"
RUN_INSTANCE_ID="${DAGGER_8GPU_RUN_INSTANCE_ID:-}"
if [[ "${DAGGER_8GPU_GPU_DEVICES+x}" == x ]]; then
  GPU_DEVICES="$DAGGER_8GPU_GPU_DEVICES"
  GPU_DEVICE_SOURCE="DAGGER_8GPU_GPU_DEVICES"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_DEVICES="$CUDA_VISIBLE_DEVICES"
  GPU_DEVICE_SOURCE="inherited CUDA_VISIBLE_DEVICES"
else
  GPU_DEVICES="0,1,2,3,4,5,6,7"
  GPU_DEVICE_SOURCE="default logical devices 0..7"
fi
BASE_RPC_PORT="${DAGGER_8GPU_BASE_RPC_PORT:-53000}"
BASE_DISPLAY_NUM="${DAGGER_8GPU_BASE_DISPLAY_NUM:-400}"
STAGGER_SECONDS="${DAGGER_8GPU_STAGGER_SECONDS:-45}"
PER_SHARD_MAX_BYTES="${DAGGER_8GPU_MAX_BYTES_PER_SHARD:-30000000000}"
PER_SHARD_SOFT_BYTES="${DAGGER_8GPU_SOFT_BYTES_PER_SHARD:-25000000000}"
PREPARE_ONLY="${DAGGER_8GPU_PREPARE_ONLY:-0}"
LP_THREADS="${DAGGER_8GPU_LP_NUM_THREADS:-8}"

if [[ -z "$RUN_INSTANCE_ID" ]]; then
  RUN_INSTANCE_ID="web_$(date -u '+%Y%m%dT%H%M%SZ')_pid$$"
fi

COHORT_DIR="${ALLOWED_ROOT}/data/heatmap_system1_training_v1/cohorts/round_000/${RUN_TAG}"
PLAN_PATH="${COHORT_DIR}/plan.json"
COLLECTION_BASE="${ALLOWED_ROOT}/data/heatmap_system1_dagger_v1/round_000/${RUN_TAG}"
CONTROL_BASE="${ALLOWED_ROOT}/data/heatmap_system1_training_v1/rollout_control/round_000/${RUN_TAG}"
CLUSTER_LOG_DIR="${ALLOWED_ROOT}/data/heatmap_system1_training_v1/cluster_logs"
TRAINING_ROOTS_MANIFEST="${CONTROL_BASE}/training_roots.json"

die() {
  echo "[dagger-8gpu] ERROR: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_file() {
  [[ -f "$1" && -s "$1" && ! -L "$1" ]] || die "missing or unsafe file: $1"
}

require_dir() {
  [[ -d "$1" && ! -L "$1" ]] || die "missing or unsafe directory: $1"
}

require_executable() {
  local resolved
  resolved="$(readlink -f -- "$1")"
  canonical_under_root "$resolved" >/dev/null
  [[ -f "$resolved" && -x "$resolved" ]] || die "missing executable: $1"
}

canonical_under_root() {
  local candidate="$1"
  local resolved
  resolved="$(readlink -m -- "$candidate")"
  case "${resolved}/" in
    "${ALLOWED_ROOT}/"*) ;;
    *) die "path escapes allowed root: $candidate -> $resolved" ;;
  esac
  printf '%s\n' "$resolved"
}

[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || die "unsafe RUN_TAG: $RUN_TAG"
[[ "$RUN_INSTANCE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || die "unsafe RUN_INSTANCE_ID: $RUN_INSTANCE_ID"
is_uint "$BASE_RPC_PORT" || die "base RPC port must be an integer"
is_uint "$BASE_DISPLAY_NUM" || die "base display number must be an integer"
is_uint "$STAGGER_SECONDS" || die "stagger seconds must be an integer"
is_uint "$PER_SHARD_MAX_BYTES" || die "per-shard hard limit must be an integer"
is_uint "$PER_SHARD_SOFT_BYTES" || die "per-shard soft limit must be an integer"
is_uint "$LP_THREADS" || die "llvmpipe thread count must be an integer"

BASE_RPC_PORT=$((10#$BASE_RPC_PORT))
BASE_DISPLAY_NUM=$((10#$BASE_DISPLAY_NUM))
STAGGER_SECONDS=$((10#$STAGGER_SECONDS))
PER_SHARD_MAX_BYTES=$((10#$PER_SHARD_MAX_BYTES))
PER_SHARD_SOFT_BYTES=$((10#$PER_SHARD_SOFT_BYTES))
LP_THREADS=$((10#$LP_THREADS))

(( BASE_RPC_PORT >= 1024 && BASE_RPC_PORT + NUM_SHARDS - 1 <= 65535 )) || die "RPC port range is invalid"
(( BASE_DISPLAY_NUM + NUM_SHARDS - 1 <= 9999 )) || die "display range is invalid"
(( STAGGER_SECONDS <= 300 )) || die "stagger must be <= 300 seconds"
(( LP_THREADS >= 1 && LP_THREADS <= 16 )) || die "llvmpipe threads must be in [1,16]"
(( PER_SHARD_MAX_BYTES > WRAPPER_RESERVE_BYTES )) || die "per-shard hard limit must exceed 5GB wrapper reserve"
(( PER_SHARD_SOFT_BYTES > 0 )) || die "per-shard soft limit must be positive"
(( PER_SHARD_SOFT_BYTES <= PER_SHARD_MAX_BYTES - WRAPPER_RESERVE_BYTES )) || die "per-shard soft limit exceeds commit ceiling"
(( PER_SHARD_MAX_BYTES * NUM_SHARDS <= ABSOLUTE_TOTAL_LIMIT_BYTES )) || die "aggregate hard ceilings exceed 300GB"

case "${PREPARE_ONLY,,}" in
  0|false|no|n|off|1|true|yes|y|on) ;;
  *) die "DAGGER_8GPU_PREPARE_ONLY must be an explicit boolean" ;;
esac
[[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+){7}$ ]] || \
  die "GPU mapping must contain exactly 8 comma-separated decimal device ids"
IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
[[ "${#GPU_LIST[@]}" -eq "$NUM_SHARDS" ]] || die "exactly 8 comma-separated GPU ids are required"
declare -A GPU_SEEN=()
for gpu_index in "${!GPU_LIST[@]}"; do
  raw_gpu="${GPU_LIST[$gpu_index]}"
  is_uint "$raw_gpu" || die "invalid GPU id: $raw_gpu"
  gpu=$((10#$raw_gpu))
  GPU_LIST[$gpu_index]="$gpu"
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || die "duplicate GPU id after decimal normalization: $gpu"
  GPU_SEEN[$gpu]=1
done

for path in "$REPO_ROOT" "$COHORT_DIR" "$COLLECTION_BASE" "$CONTROL_BASE" "$CLUSTER_LOG_DIR"; do
  canonical_under_root "$path" >/dev/null
done
require_executable "$PYTHON"
require_file "$SHARD_BUILDER"
require_file "$SHARD_FINALIZER"
require_file "$WORKER_WRAPPER"
require_file "$TRAIN_DATA"
require_executable "$VLNCE_PYTHON"
require_executable "$X11_BUNDLE/bin/Xvfb"
require_executable "$X11_BUNDLE/bin/xdpyinfo"
require_executable "$X11_BUNDLE/bin/glxinfo"
require_executable "$X11_BUNDLE/bin/xkbcomp"
require_dir "$X11_BUNDLE"
require_dir "$X11_DRI_PATH"
require_dir "$X11_XKB_PATH"
require_dir "$X11_FONT_PATH"
require_file "$X11_MANIFEST"
require_file "$X11_DRI_PATH/swrast_dri.so"
[[ -x "$SETSID_BIN" ]] || die "missing setsid: $SETSID_BIN"
[[ -x "$FLOCK_BIN" ]] || die "missing flock: $FLOCK_BIN"
[[ -x "$BASH_BIN" ]] || die "missing fixed bash: $BASH_BIN"
[[ -x "$ENV_BIN" ]] || die "missing fixed env: $ENV_BIN"
[[ "$(uname -m)" == "x86_64" ]] || die "X11 bundle requires x86_64"

mkdir -p -- "$(dirname "$COHORT_DIR")" "$COLLECTION_BASE" "$CONTROL_BASE" "$CLUSTER_LOG_DIR"

RUN_LOCK_DIR="${CONTROL_BASE}/.outer_run.lock"
RUN_LOCK_OWNER="${RUN_LOCK_DIR}/owner.json"
RUN_LOCK_FLOCK="${RUN_LOCK_DIR}/advisory.flock"
RUN_LOCK_TOKEN=""
RUN_LOCK_OWNED=0
RUN_LOCK_FD=""

release_run_lock() {
  local actual_token=""
  (( RUN_LOCK_OWNED == 1 )) || return 0
  if [[ -f "$RUN_LOCK_OWNER" && ! -L "$RUN_LOCK_OWNER" ]]; then
    actual_token="$("$PYTHON" - "$RUN_LOCK_OWNER" <<'PY'
import json
import sys
from pathlib import Path
try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(2)
token = value.get("token")
if not isinstance(token, str):
    raise SystemExit(3)
print(token)
PY
)" || actual_token=""
  fi
  if [[ "$actual_token" != "$RUN_LOCK_TOKEN" ]]; then
    echo "[dagger-8gpu] refusing to remove run lock with mismatched owner: $RUN_LOCK_DIR" >&2
    return 0
  fi
  if [[ -n "$RUN_LOCK_FD" ]]; then
    exec {RUN_LOCK_FD}>&- || true
  fi
  rm -f -- "$RUN_LOCK_OWNER" "$RUN_LOCK_FLOCK"
  if ! rmdir -- "$RUN_LOCK_DIR"; then
    echo "[dagger-8gpu] run lock directory retained because it is not empty: $RUN_LOCK_DIR" >&2
    return 0
  fi
  RUN_LOCK_OWNED=0
  echo "[dagger-8gpu] released run lock: $RUN_LOCK_DIR"
}
trap release_run_lock EXIT

job_id_tag="$RUN_INSTANCE_ID"
if ! is_true "$PREPARE_ONLY"; then
  RUN_LOCK_TOKEN="$(date -u '+%Y%m%dT%H%M%SZ')_run${job_id_tag}_pid$$_r${RANDOM}"
  [[ "$RUN_LOCK_TOKEN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "invalid run lock token"
  if ! mkdir -- "$RUN_LOCK_DIR" 2>/dev/null; then
    owner_hint=""
    if [[ -f "$RUN_LOCK_OWNER" && ! -L "$RUN_LOCK_OWNER" ]]; then
      owner_hint="$(tr '\n' ' ' < "$RUN_LOCK_OWNER" | cut -c1-1000 || true)"
    fi
    die "run lock already exists; another job may own RUN_TAG=$RUN_TAG: $RUN_LOCK_DIR ${owner_hint:+owner=$owner_hint}"
  fi
  RUN_LOCK_OWNED=1
  "$PYTHON" - "$RUN_LOCK_OWNER" "$RUN_LOCK_TOKEN" "$job_id_tag" "$$" "${HOSTNAME:-unknown}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "heatmapvln-dagger-outer-run-lock-v2",
    "token": sys.argv[2],
    "run_instance_id": sys.argv[3],
    "launcher_mode": "web-direct",
    "outer_pid": int(sys.argv[4]),
    "host": sys.argv[5],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  exec {RUN_LOCK_FD}> "$RUN_LOCK_FLOCK"
  "$FLOCK_BIN" -n "$RUN_LOCK_FD" || die "cannot acquire advisory run lock: $RUN_LOCK_FLOCK"
  echo "[dagger-8gpu] acquired atomic run lock: $RUN_LOCK_DIR"
fi

sha256sum -c "$X11_MANIFEST" >/dev/null
PREFLIGHT_RUNTIME="$(canonical_under_root "${CONTROL_BASE}/outer_preflight/${job_id_tag}_pid$$")"
[[ ! -L "$PREFLIGHT_RUNTIME" ]] || die "refusing symlinked preflight runtime: $PREFLIGHT_RUNTIME"
mkdir -p \
  "$PREFLIGHT_RUNTIME/home" \
  "$PREFLIGHT_RUNTIME/tmp" \
  "$PREFLIGHT_RUNTIME/xdg_cache" \
  "$PREFLIGHT_RUNTIME/xdg_runtime" \
  "$PREFLIGHT_RUNTIME/numba_cache" \
  "$PREFLIGHT_RUNTIME/mesa_shader_cache" \
  "$PREFLIGHT_RUNTIME/pycache"
chmod 700 "$PREFLIGHT_RUNTIME/xdg_runtime"
readonly FIXED_MACA_LD_LIBRARY_PATH="/opt/maca-3.3.0/lib:/opt/maca-3.3.0/ompi/lib:/opt/maca-3.3.0/ucx/lib:/opt/mxdriver/lib"
X11_CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:$FIXED_MACA_LD_LIBRARY_PATH"
PREFLIGHT_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY
  -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES
  -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME
  -u BASH_ENV -u ENV
)
while IFS='=' read -r -d '' inherited_name _; do
  case "$inherited_name" in
    BASH_FUNC_*|__GLX_*|__EGL_*|EGL_*|GLX_*|LIBGL_*|GALLIUM_*|MESA_*|HABITAT_*|MAGNUM_*|GBM_*)
      PREFLIGHT_ENV_UNSET_ARGS+=(-u "$inherited_name")
      ;;
  esac
done < <("$ENV_BIN" -0)
"$ENV_BIN" "${PREFLIGHT_ENV_UNSET_ARGS[@]}" \
  PATH="$FIXED_SYSTEM_PATH" \
  HOME="$PREFLIGHT_RUNTIME/home" \
  TMPDIR="$PREFLIGHT_RUNTIME/tmp" \
  XDG_CACHE_HOME="$PREFLIGHT_RUNTIME/xdg_cache" \
  XDG_RUNTIME_DIR="$PREFLIGHT_RUNTIME/xdg_runtime" \
  NUMBA_CACHE_DIR="$PREFLIGHT_RUNTIME/numba_cache" \
  MESA_SHADER_CACHE_DIR="$PREFLIGHT_RUNTIME/mesa_shader_cache" \
  PYTHONPYCACHEPREFIX="$PREFLIGHT_RUNTIME/pycache" \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="" \
  LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH" \
  LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  LIBGL_ALWAYS_SOFTWARE=1 \
  GALLIUM_DRIVER=llvmpipe \
  MESA_LOADER_DRIVER_OVERRIDE=swrast \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  timeout 120 "$VLNCE_PYTHON" -c 'import magnum, habitat_sim; print("magnum/habitat_sim import OK")'

echo "[dagger-8gpu] building or verifying exact full-train shard plan"
"$ENV_BIN" -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  PATH="$FIXED_SYSTEM_PATH" LD_LIBRARY_PATH="$FIXED_MACA_LD_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" "$SHARD_BUILDER" \
    --dataset "$TRAIN_DATA" \
    --count "$TOTAL_EPISODES" \
    --num-shards "$NUM_SHARDS" \
    --seed "$SEED" \
    --output-dir "$COHORT_DIR"

"$ENV_BIN" -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  PATH="$FIXED_SYSTEM_PATH" LD_LIBRARY_PATH="$FIXED_MACA_LD_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" - "$PLAN_PATH" "$TOTAL_EPISODES" "$NUM_SHARDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_episodes = int(sys.argv[2])
expected_shards = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "r2r-dagger-shard-plan-v1":
    raise SystemExit("wrong shard plan schema")
if payload.get("selected_episode_count") != expected_episodes:
    raise SystemExit("shard plan episode count mismatch")
if payload.get("num_shards") != expected_shards:
    raise SystemExit("shard plan count mismatch")
if payload.get("route_grouped") is not True:
    raise SystemExit("shard plan is not canonical-route grouped")
entries = payload.get("shards")
if not isinstance(entries, list) or len(entries) != expected_shards:
    raise SystemExit("shard plan entries are incomplete")
if sum(item.get("episode_count", 0) for item in entries) != expected_episodes:
    raise SystemExit("shard episode counts do not sum to full train")
for index, item in enumerate(entries):
    expected_name = f"shard_{index:02d}.json"
    if item.get("index") != index or item.get("file") != expected_name:
        raise SystemExit(f"invalid shard plan entry {index}")
    shard_path = path.parent / expected_name
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    if digest != item.get("sha256"):
        raise SystemExit(f"shard {index} digest mismatch")
print(
    "verified shard plan:",
    "episodes=" + str(expected_episodes),
    "routes=" + str(payload.get("selected_route_count")),
    "loads=" + str([item["episode_count"] for item in entries]),
)
PY

echo "[dagger-8gpu] launcher_mode=web-direct run_instance_id=$RUN_INSTANCE_ID"
echo "[dagger-8gpu] gpu_devices=$GPU_DEVICES source=$GPU_DEVICE_SOURCE"
if is_true "$PREPARE_ONLY"; then
  echo "[dagger-8gpu] prepare-only complete; no GPU process was launched"
  exit 0
fi

job_token="$(date -u '+%Y%m%dT%H%M%SZ')_run${job_id_tag}_pid$$"
[[ "$job_token" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "invalid job token"
declare -A ACTIVE_WORKERS=()
declare -A WORKER_NAMES=()
declare -A WORKER_PGIDS=()
failure_status=0
failure_name=""
CLEANUP_ACTIVE=0
CLEANUP_RESIDUAL=0

process_group_alive() {
  local pgid="$1"
  kill -0 -- "-$pgid" 2>/dev/null
}

cleanup_workers() {
  local pid pgid deadline any_alive
  set +e
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    pgid="${WORKER_PGIDS[$pid]:-}"
    if [[ "$pgid" =~ ^[1-9][0-9]*$ ]] && process_group_alive "$pgid"; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${!ACTIVE_WORKERS[@]}"; do
      pgid="${WORKER_PGIDS[$pid]:-}"
      if [[ "$pgid" =~ ^[1-9][0-9]*$ ]] && process_group_alive "$pgid"; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    pgid="${WORKER_PGIDS[$pid]:-}"
    if [[ "$pgid" =~ ^[1-9][0-9]*$ ]] && process_group_alive "$pgid"; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${!ACTIVE_WORKERS[@]}"; do
      pgid="${WORKER_PGIDS[$pid]:-}"
      if [[ "$pgid" =~ ^[1-9][0-9]*$ ]] && process_group_alive "$pgid"; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    pgid="${WORKER_PGIDS[$pid]:-}"
    if [[ "$pgid" =~ ^[1-9][0-9]*$ ]] && process_group_alive "$pgid"; then
      echo "[dagger-8gpu] residual process group=$pgid; retaining run lock" >&2
      CLEANUP_RESIDUAL=1
      continue
    fi
    wait "$pid" 2>/dev/null || true
    unset "ACTIVE_WORKERS[$pid]" "WORKER_NAMES[$pid]" "WORKER_PGIDS[$pid]"
  done
}
top_level_cleanup() {
  local status=$?
  (( CLEANUP_ACTIVE == 0 )) || exit "$status"
  CLEANUP_ACTIVE=1
  trap - EXIT INT TERM HUP
  cleanup_workers
  if (( CLEANUP_RESIDUAL == 0 )); then
    release_run_lock
  else
    echo "[dagger-8gpu] run lock retained because worker processes survived SIGKILL" >&2
  fi
  exit "$status"
}
trap top_level_cleanup EXIT
trap 'echo "[dagger-8gpu] received termination signal" >&2; exit 130' INT TERM HUP

reap_finished_workers() {
  local pid status name pgid
  local -a finished=()
  for pid in "${!ACTIVE_WORKERS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      finished+=("$pid")
    fi
  done
  for pid in "${finished[@]}"; do
    name="${WORKER_NAMES[$pid]}"
    pgid="${WORKER_PGIDS[$pid]}"
    set +e
    wait "$pid"
    status=$?
    set -e
    if process_group_alive "$pgid"; then
      echo "[dagger-8gpu] worker=$name leader exited but process group remains alive" >&2
      (( status == 0 )) && status=1
    else
      unset "ACTIVE_WORKERS[$pid]" "WORKER_NAMES[$pid]" "WORKER_PGIDS[$pid]"
    fi
    if (( status != 0 )) && (( failure_status == 0 )); then
      failure_status="$status"
      failure_name="$name"
    fi
    echo "[dagger-8gpu] worker=$name exited status=$status"
  done
}

CHILD_ENV_UNSET_ARGS=(
  -u BASH_ENV
  -u ENV
  -u SHELLOPTS
  -u BASHOPTS
  -u CDPATH
  -u GLOBIGNORE
  -u PROMPT_COMMAND
  -u MACA_HOME
  -u MACA_PATH
  -u MACA_DIR
  -u CUDA_VISIBLE_DEVICES
  -u MUSA_VISIBLE_DEVICES
  -u HIP_VISIBLE_DEVICES
  -u STAGE3_EVAL_SCENES_DIR
  -u HEATMAP_SYSTEM1_MAX_EPISODES
  -u STAGE3_EVAL_MAX_EPISODES
  -u HEATMAP_SYSTEM1_POLICY_FINGERPRINT
  -u STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT
  -u INTERNNAV_NATIVE_POLICY_FINGERPRINT
  -u STAGE3_EVAL_BASE_CKPT
  -u STAGE3_EVAL_CHECKPOINT
  -u STAGE3_EVAL_PANO_LATENT_ADAPTER_CHECKPOINT
  -u HEATMAP_SYSTEM1_BASE_RPC_LAUNCHER
  -u HEATMAPVLN_REPO_ROOT
  -u QWEN25_PYTHON
  -u VLNCE_PYTHON
  -u PYTHONPATH
  -u PYTHONHOME
  -u LD_LIBRARY_PATH
  -u LD_PRELOAD
  -u DISPLAY
  -u WAYLAND_DISPLAY
  -u XAUTHORITY
  -u __GLX_VENDOR_LIBRARY_NAME
  -u __EGL_VENDOR_LIBRARY_FILENAMES
  -u LIBGL_DRIVERS_PATH
  -u LIBGL_ALWAYS_SOFTWARE
  -u LIBGL_ALWAYS_INDIRECT
  -u GALLIUM_DRIVER
  -u MESA_LOADER_DRIVER_OVERRIDE
  -u MESA_SHADER_CACHE_DIR
  -u LP_NUM_THREADS
  -u HABITAT_GL_GPU_ID
  -u HEATMAPVLN_ALLOW_NVIDIA_GLX
  -u HEATMAPVLN_PREINIT_GL
  -u HEATMAPVLN_PREINIT_EMPTY_GL
)
while IFS='=' read -r -d '' inherited_name _; do
  case "$inherited_name" in
    BASH_FUNC_*|DAGGER_8GPU_*|STAGE3_EVAL_*|HEATMAP_SYSTEM1_*|INTERNNAV_NATIVE_*|HEATMAPVLN_*|MACA_*|__GLX_*|__EGL_*|EGL_*|GLX_*|LIBGL_*|GALLIUM_*|MESA_*|HABITAT_*|MAGNUM_*|GBM_*)
      CHILD_ENV_UNSET_ARGS+=(-u "$inherited_name")
      ;;
  esac
done < <("$ENV_BIN" -0)

echo "[dagger-8gpu] launching 8 independent native InternNav workers"
echo "[dagger-8gpu] collection_base=$COLLECTION_BASE"
echo "[dagger-8gpu] control_base=$CONTROL_BASE"
echo "[dagger-8gpu] per_shard_hard=$PER_SHARD_MAX_BYTES per_shard_soft=$PER_SHARD_SOFT_BYTES"
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  printf -v shard_name 'shard_%02d' "$shard_index"
  cohort_path="$(canonical_under_root "${COHORT_DIR}/${shard_name}.json")"
  collection_root="$(canonical_under_root "${COLLECTION_BASE}/${shard_name}")"
  control_root="$(canonical_under_root "${CONTROL_BASE}/${shard_name}")"
  log_raw="${CLUSTER_LOG_DIR}/${RUN_TAG}_${job_token}_${shard_name}.log"
  [[ ! -e "$log_raw" && ! -L "$log_raw" ]] || die "worker log target already exists: $log_raw"
  log_path="$(canonical_under_root "$log_raw")"
  [[ "$log_path" == "$log_raw" ]] || die "worker log path changed during canonicalization: $log_raw -> $log_path"
  rpc_port=$((BASE_RPC_PORT + shard_index))
  display_num=$((BASE_DISPLAY_NUM + shard_index))
  gpu="${GPU_LIST[$shard_index]}"

  require_file "$cohort_path"
  mkdir -p "$control_root/outer_home" "$control_root/outer_pycache"
  set -o noclobber
  if ! exec {worker_log_fd}> "$log_path"; then
    set +o noclobber
    die "cannot exclusively create worker log: $log_path"
  fi
  set +o noclobber

  "$SETSID_BIN" --wait "$ENV_BIN" "${CHILD_ENV_UNSET_ARGS[@]}" \
    PATH="$FIXED_SYSTEM_PATH" \
    HOME="$control_root/outer_home" \
    PYTHONPYCACHEPREFIX="$control_root/outer_pycache" \
    PYTHONDONTWRITEBYTECODE=1 \
    HEATMAP_SYSTEM1_COLLECTION_ROOT="$collection_root" \
    HEATMAP_SYSTEM1_CONTROL_ROOT="$control_root" \
    HEATMAP_SYSTEM1_TRAIN_DATA_PATH="$TRAIN_DATA" \
    HEATMAP_SYSTEM1_EPISODE_LIST="$cohort_path" \
    HEATMAP_SYSTEM1_DAGGER_ROUND=0 \
    HEATMAP_SYSTEM1_MAX_BYTES="$PER_SHARD_MAX_BYTES" \
    HEATMAP_SYSTEM1_SOFT_STOP_BYTES="$PER_SHARD_SOFT_BYTES" \
    HEATMAP_SYSTEM1_SIZE_CHECK_INTERVAL_S=15 \
    HEATMAP_SYSTEM1_MAX_STEPS=120 \
    HEATMAP_SYSTEM1_MAX_SYSTEM2_CALLS=64 \
    INTERNNAV_NATIVE_GPU_ID="$gpu" \
    INTERNNAV_NATIVE_RPC_PORT="$rpc_port" \
    INTERNNAV_NATIVE_DISPLAY_NUM="$display_num" \
    INTERNNAV_NATIVE_RUN_STAMP="${job_token}_${shard_name}" \
    INTERNNAV_NATIVE_LP_NUM_THREADS="$LP_THREADS" \
    INTERNNAV_NATIVE_SERVER_CPU_THREADS=4 \
    INTERNNAV_NATIVE_CLIENT_CPU_THREADS=1 \
    STAGE3_EVAL_PREFLIGHT_ONLY=0 \
    "$BASH_BIN" -p "$WORKER_WRAPPER" \
    >&"$worker_log_fd" 2>&1 &
  pid=$!
  ACTIVE_WORKERS[$pid]=1
  WORKER_NAMES[$pid]="$shard_name"
  WORKER_PGIDS[$pid]="$pid"
  exec {worker_log_fd}>&-
  sleep 0.2
  if ! kill -0 "$pid" 2>/dev/null; then
    set +e
    wait "$pid"
    status=$?
    set -e
    die "worker $shard_name exited during process-group startup with status $status; log=$log_path"
  fi
  pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
  sid="$(ps -o sid= -p "$pid" | tr -d '[:space:]')"
  if [[ "$pgid" != "$pid" || "$sid" != "$pid" ]]; then
    die "worker $shard_name failed session isolation: pid=$pid pgid=${pgid:-missing} sid=${sid:-missing}"
  fi
  WORKER_PGIDS[$pid]="$pgid"
  echo "[dagger-8gpu] started $shard_name pid=$pid pgid=$pgid gpu=$gpu rpc=$rpc_port display=$display_num log=$log_path"

  if (( shard_index + 1 < NUM_SHARDS && STAGGER_SECONDS > 0 )); then
    sleep "$STAGGER_SECONDS"
    reap_finished_workers
    if (( failure_status != 0 )); then
      die "worker $failure_name failed during stagger with status $failure_status"
    fi
  fi
done

while (( ${#ACTIVE_WORKERS[@]} > 0 )); do
  sleep 15
  reap_finished_workers
  if (( failure_status != 0 )); then
    die "worker $failure_name failed with status $failure_status"
  fi
done

echo "[dagger-8gpu] all shards exited successfully; finalizing aggregate training roots"
FINALIZER_RUNTIME="$(canonical_under_root "${CONTROL_BASE}/finalizer_runtime/${job_token}")"
mkdir -p "$FINALIZER_RUNTIME/home" "$FINALIZER_RUNTIME/tmp" "$FINALIZER_RUNTIME/xdg_cache" "$FINALIZER_RUNTIME/pycache"
"$ENV_BIN" \
  -u PYTHONPATH -u PYTHONHOME -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
  -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES \
  -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PATH="$FIXED_SYSTEM_PATH" \
  LD_LIBRARY_PATH="$FIXED_MACA_LD_LIBRARY_PATH" \
  HOME="$FINALIZER_RUNTIME/home" \
  TMPDIR="$FINALIZER_RUNTIME/tmp" \
  XDG_CACHE_HOME="$FINALIZER_RUNTIME/xdg_cache" \
  PYTHONPYCACHEPREFIX="$FINALIZER_RUNTIME/pycache" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" "$SHARD_FINALIZER" \
    --plan "$PLAN_PATH" \
    --collection-base "$COLLECTION_BASE" \
    --control-base "$CONTROL_BASE" \
    --output "$TRAINING_ROOTS_MANIFEST" \
    --max-bytes "$ABSOLUTE_TOTAL_LIMIT_BYTES"

echo "[dagger-8gpu] READY: $TRAINING_ROOTS_MANIFEST"
