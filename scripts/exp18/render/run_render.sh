#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXP-18: render tier C (R2R val_unseen), D (HM3D / ScaleVLN) or E (designed
# routes) with the VLN-CE panoramic collector, in exactly the
# r2r_panoramic_data_v2 chunk format (256x256 HFOV 90, 5 views, depth for
# front + front_down).
#
# Same certified headless recipe as scripts/run_collect_panoramic_mxc500.sh:
# one Xvfb per worker from the /mnt/afs X11 bundle, Mesa llvmpipe software GL,
# the vlnce python called by absolute path.  Differences from that launcher:
#   * --config comes from select_episodes.py (EPISODES_ALLOWED etc.);
#   * logs and Xvfb runtime dirs go to <RENDER_ROOT>/<tier>/logs, never into
#     the shared VLN-CE checkout; PYTHONDONTWRITEBYTECODE=1 keeps .pyc out too;
#   * every worker renders its whole shard (the collector stops when its
#     episode iterator cycles), so no selected episode is left out;
#   * SHARD_BY=scene gives each worker its own scenes (default for D, where
#     every episode-sharded worker would otherwise load all 30 HM3D scenes);
#   * resumable: re-running skips episodes that already have meta.json, and
#     clip dirs left without meta.json by a killed run are moved to
#     <raw>/_incomplete/<stamp>/ first (they would reuse a clip id and keep
#     stale chunks).  A finalized tier (finalize_clip_lists.py ran) is refused
#     unless FORCE_RERENDER=1.
#   * rendering is CPU only (llvmpipe): CUDA_VISIBLE_DEVICES is pinned to 7.
#
# Exit status: 0 = every selected episode has a clip; 1 = setup error or a
# worker failed/timed out; 3 = workers finished but some selected episodes
# have no clip (collector "Failed:" lines in the worker logs).  On 1 or 3,
# re-run the same command: it resumes and retries only the missing episodes.
# Then run finalize_clip_lists.py, which refuses a tier with missing episodes.
#
# Website submission (blank container, parameters as env vars):
#   cd /mnt/afs/liwenhao/agent/370910109/<staged HeatmapVLN copy>
#   export TIER=C            # C | D | E
#   export NUM_WORKERS=8     # optional
#   bash scripts/exp18/render/run_render.sh
#
# Env: TIER (required), EXP18_RENDER_ROOT (default data/exp18_renders),
#   CONFIG (default <RENDER_ROOT>/configs/<TIER>.yaml), NUM_WORKERS (8),
#   BASE_DISPLAY (C 310 / D 330 / E 350), SHARD_BY (episode|scene),
#   COLLECT_MAX_STEPS (C/D 300 like v2, E 800), LOG_DIR, WORKER_TIMEOUT_S
#   (10800), COLLECT_LP_NUM_THREADS (8), COLLECT_IO_WORKERS (8), FORCE_RERENDER.
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
PROJECT_DIR="${ROOT}/habitat/VLN-CE"
VLNCE_PYTHON="${ROOT}/envs/vlnce/bin/python"
X11_BUNDLE="${ROOT}/tools/x11_headless_bundle_ubuntu22_20260801_v4"
XVFB_BIN="${X11_BUNDLE}/bin/Xvfb"
XDPYINFO_BIN="${X11_BUNDLE}/bin/xdpyinfo"
GLXINFO_BIN="${X11_BUNDLE}/bin/glxinfo"
X11_DRI_PATH="${X11_BUNDLE}/dri"
X11_FONT_PATH="${X11_BUNDLE}/share/fonts/misc"
X11_XKB_PATH="${X11_BUNDLE}/share/X11/xkb"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PYTHONDONTWRITEBYTECODE=1
TIER="${1:-${TIER:-}}"
case "$TIER" in
  C|D|E) ;;
  *) echo "[ERROR] TIER must be C, D or E (got '${TIER}')" >&2; exit 1 ;;
esac

# Paths come from scripts/exp18/common.py (honours EXP18_RENDER_ROOT).
LAYOUT_VARS="$(cd "$REPO_ROOT" && "$VLNCE_PYTHON" -m scripts.exp18.render.layout --shell "$TIER")"
eval "$LAYOUT_VARS"

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_DISPLAY="${BASE_DISPLAY:-$DEFAULT_BASE_DISPLAY}"
SHARD_BY="${SHARD_BY:-$DEFAULT_SHARD_BY}"
MAX_STEPS="${COLLECT_MAX_STEPS:-$DEFAULT_MAX_STEPS}"
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"
WORKER_TIMEOUT_S="${WORKER_TIMEOUT_S:-10800}"
LP_THREADS="${COLLECT_LP_NUM_THREADS:-8}"
IO_WORKERS="${COLLECT_IO_WORKERS:-8}"
CLIP_ID_BLOCK=100000

GL_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u EGL_PLATFORM
  -u __EGL_VENDOR_LIBRARY_FILENAMES -u __GLX_VENDOR_LIBRARY_NAME
  -u LIBGL_ALWAYS_INDIRECT -u MESA_LOADER_DRIVER_OVERRIDE -u LIBGL_DRIVERS_PATH
)

if [[ "$NUM_WORKERS" -le 0 || "$NUM_WORKERS" -gt 16 ]]; then
  echo "[ERROR] NUM_WORKERS must be in 1..16 (more llvmpipe workers oversubscribe the node)" >&2
  exit 1
fi
if [[ "$SHARD_BY" != episode && "$SHARD_BY" != scene ]]; then
  echo "[ERROR] SHARD_BY must be episode or scene" >&2
  exit 1
fi
for path in "$PROJECT_DIR/collect/panoramic/collector.py" "$CONFIG" \
    "$XVFB_BIN" "$XDPYINFO_BIN" "$GLXINFO_BIN" "$X11_DRI_PATH/swrast_dri.so" \
    "$X11_XKB_PATH" "$X11_FONT_PATH" "$VLNCE_PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done
if ! grep -q -- "--episode-modulo" "$PROJECT_DIR/collect/panoramic/collector.py"; then
  echo "[ERROR] collector.py lacks the sharding patch (--episode-modulo)" >&2
  exit 1
fi
if [[ -e "$FINALIZED_MARKER" && "${FORCE_RERENDER:-0}" != 1 ]]; then
  echo "[ERROR] tier $TIER is finalized ($FINALIZED_MARKER); re-rendering would redo the" \
       "excluded short clips. Set FORCE_RERENDER=1 only if that is intended." >&2
  exit 1
fi

mkdir -p "$OUTPUT/$SPLIT" "$LOG_DIR"
exec 9>"${OUTPUT}/.parallel_collection.lock"
if ! flock -n 9; then
  echo "[ERROR] Another collector is already using $OUTPUT" >&2
  exit 1
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LOG_DIR}/render_${TIER}_${NUM_WORKERS}w_${RUN_STAMP}.log"
XVFB_RUNTIME="${LOG_DIR}/xvfb_${RUN_STAMP}"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

EXPECTED="$(grep -m1 'EPISODES_ALLOWED:' "$CONFIG" | "$VLNCE_PYTHON" -c \
  'import json,sys; print(len(json.loads(sys.stdin.read().split(":",1)[1])))')"

echo "============================================================"
echo "EXP-18 tier $TIER render (VLN-CE panoramic collector, Xvfb + llvmpipe)"
echo "============================================================"
echo "Config:        $CONFIG ($EXPECTED episodes)"
echo "Output:        $OUTPUT/$SPLIT"
echo "Workers:       $NUM_WORKERS (shard by $SHARD_BY)"
echo "Displays:      :${BASE_DISPLAY}.."
echo "Max steps:     $MAX_STEPS"
echo "LP threads:    $LP_THREADS per worker"
echo "Launcher log:  $LAUNCH_LOG"
echo "============================================================"

# Clip dirs without meta.json are leftovers of a killed run: move them aside.
mapfile -t PARTIAL < <(find "$OUTPUT/$SPLIT" -mindepth 2 -maxdepth 2 -type d -name 'clip_*' \
  ! -exec test -e '{}/meta.json' ';' -print | sort)
for clip in "${PARTIAL[@]:-}"; do
  [[ -z "$clip" ]] && continue
  dest="${INCOMPLETE_DIR}/${RUN_STAMP}/$(basename "$(dirname "$clip")")"
  mkdir -p "$dest"
  mv "$clip" "$dest/"
  echo "[RESUME] moved incomplete $clip -> $dest/"
done
DONE_BEFORE="$(find "$OUTPUT/$SPLIT" -mindepth 3 -maxdepth 3 -name meta.json | wc -l)"
echo "[RESUME] clips with meta.json already present: $DONE_BEFORE"

# Per-worker configs + shard arguments.
declare -a WORKER_CONFIGS=() WORKER_SHARD_ARGS=()
if [[ "$SHARD_BY" == scene ]]; then
  mapfile -t WORKER_CONFIGS < <(cd "$REPO_ROOT" && "$VLNCE_PYTHON" -m scripts.exp18.render.layout \
    --worker-config "$CONFIG" "$NUM_WORKERS" "${LOG_DIR}/worker_configs_${RUN_STAMP}")
  NUM_WORKERS="${#WORKER_CONFIGS[@]}"
  if [[ "$NUM_WORKERS" -eq 0 ]]; then
    echo "[ERROR] no per-worker configs written from $CONFIG" >&2
    exit 1
  fi
  for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    WORKER_SHARD_ARGS+=("--episode-modulo 1 --episode-remainder 0")
  done
else
  for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    WORKER_CONFIGS+=("$CONFIG")
    WORKER_SHARD_ARGS+=("--episode-modulo $NUM_WORKERS --episode-remainder $worker")
  done
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
X11_TOOL_LD_LIBRARY_PATH="${X11_BUNDLE}/lib:${LD_LIBRARY_PATH}"

declare -a XVFB_PIDS=()
declare -a WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    # the worker subshell's child is `timeout`, which forwards TERM to python
    [[ -n "$pid" ]] && { pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; } || true
  done
  for pid in "${XVFB_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# ------------------------------------------------------------
# 1. One isolated Xvfb per worker, from the certified bundle.
# ------------------------------------------------------------
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  display_num=$((BASE_DISPLAY + worker))
  display_addr="localhost:${display_num}.0"
  if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    echo "[ERROR] DISPLAY $display_addr is already active; choose another BASE_DISPLAY" >&2
    exit 1
  fi
  xvfb_dir="${XVFB_RUNTIME}/display_${display_num}"
  mkdir -p "${xvfb_dir}/.xkb-cache"
  (
    cd "$xvfb_dir"
    exec 9<"${xvfb_dir}/.xkb-cache"
    exec env "${GL_ENV_UNSET_ARGS[@]}" \
      PATH="${X11_BUNDLE}/bin:$PATH" \
      LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
      LIBGL_ALWAYS_SOFTWARE=1 \
      GALLIUM_DRIVER=llvmpipe \
      MESA_LOADER_DRIVER_OVERRIDE=swrast \
      LP_NUM_THREADS="$LP_THREADS" \
      "$XVFB_BIN" ":${display_num}" \
      -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
      -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
  ) >"${xvfb_dir}/xvfb.log" 2>&1 &
  XVFB_PIDS+=("$!")

  ready=0
  for _ in $(seq 1 60); do
    if ! kill -0 "${XVFB_PIDS[-1]}" 2>/dev/null; then
      echo "[ERROR] Xvfb :${display_num} exited during startup" >&2
      tail -50 "${xvfb_dir}/xvfb.log" >&2 || true
      exit 1
    fi
    if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
        DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "[ERROR] Xvfb :${display_num} did not become ready" >&2
    exit 1
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
    echo "[ERROR] DISPLAY $display_addr is not llvmpipe: ${renderer:-no renderer}" >&2
    exit 1
  fi
  echo "[XVFB] :${display_num} ready (${renderer#*: })"
done

# ------------------------------------------------------------
# 2. Collection workers.  Each owns clip ids w*100000+1 .. w*100000+99999
#    and renders its entire shard; the collector exits when its episode
#    iterator cycles.
# ------------------------------------------------------------
echo "[INFO] Python: $VLNCE_PYTHON ($("$VLNCE_PYTHON" --version 2>&1))"

run_worker() {
  local worker="$1"
  local display_addr="$2"
  local config="$3"
  local shard_args="$4"
  local id_start=$((worker * CLIP_ID_BLOCK + 1))
  local id_end=$((worker * CLIP_ID_BLOCK + CLIP_ID_BLOCK - 1))
  local worker_log="${LOG_DIR}/render_${TIER}_w${worker}_${RUN_STAMP}.log"

  echo "[LAUNCH] worker=$worker config=$config shard='$shard_args' ids=${id_start}.. " \
       "display=$display_addr log=$worker_log"
  (
    cd "$PROJECT_DIR"
    export DISPLAY="$display_addr"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=7
    export LP_NUM_THREADS="$LP_THREADS"
    export OMP_NUM_THREADS="$LP_THREADS"
    ulimit -c 0
    # Worker-local bundle Mesa/GLVND client stack with forced llvmpipe (blank
    # containers ship no system GL; magnum dlopens libOpenGL.so.0).
    export LD_LIBRARY_PATH="${X11_BUNDLE}/mesa_lib:${LD_LIBRARY_PATH:-}"
    export LIBGL_DRIVERS_PATH="$X11_DRI_PATH"
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
    export MESA_LOADER_DRIVER_OVERRIDE=swrast
    unset WAYLAND_DISPLAY EGL_PLATFORM __EGL_VENDOR_LIBRARY_FILENAMES
    unset __GLX_VENDOR_LIBRARY_NAME LIBGL_ALWAYS_INDIRECT
    # shellcheck disable=SC2086
    exec timeout --kill-after=60 "$WORKER_TIMEOUT_S" "$VLNCE_PYTHON" -m collect panoramic \
      --config "$config" \
      --output "$OUTPUT" \
      --split "$SPLIT" \
      --num-clips "$id_end" \
      --max-steps "$MAX_STEPS" \
      --num-workers "$IO_WORKERS" \
      --gpu 0 \
      --depth-directions front front_down \
      $shard_args \
      --clip-id-start "$id_start" \
      --clip-id-end "$id_end"
  ) >"$worker_log" 2>&1
}

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  run_worker "$worker" "localhost:$((BASE_DISPLAY + worker)).0" \
    "${WORKER_CONFIGS[$worker]}" "${WORKER_SHARD_ARGS[$worker]}" &
  WORKER_PIDS+=("$!")
done

STATUS=0
for index in "${!WORKER_PIDS[@]}"; do
  if wait "${WORKER_PIDS[$index]}"; then
    echo "[DONE] worker=$index"
  else
    echo "[FAILED] worker=$index (see ${LOG_DIR}/render_${TIER}_w${index}_${RUN_STAMP}.log)" >&2
    STATUS=1
  fi
done

FAILED_EPISODES="$(cat "${LOG_DIR}"/render_"${TIER}"_w*_"${RUN_STAMP}".log | grep -c -E '^  Failed: [^ ]|^  Skip: (too few|missing)' || true)"
TOTAL_COLLECTED="$(find "$OUTPUT/$SPLIT" -mindepth 3 -maxdepth 3 -name meta.json | wc -l)"
echo "============================================================"
echo "[SUMMARY] clips with meta.json in $OUTPUT/$SPLIT: $TOTAL_COLLECTED / $EXPECTED selected"
echo "[SUMMARY] collector failures/skips this run: $FAILED_EPISODES"
echo "[SUMMARY] next: <python> -m scripts.exp18.render.finalize_clip_lists --tier $TIER"
echo "[SUMMARY] launcher log: $LAUNCH_LOG"
echo "============================================================"
# A plain assignment, so a crash of the check itself aborts (set -e) instead of reading as "none missing".
MISSING_IDS="$(cd "$REPO_ROOT" && "$VLNCE_PYTHON" -m scripts.exp18.render.layout --missing "$CONFIG" "$OUTPUT")"
if [[ -n "$MISSING_IDS" ]]; then
  echo "[ERROR] $(wc -l <<< "$MISSING_IDS") of $EXPECTED selected episodes have no clip:" \
       "$(head -20 <<< "$MISSING_IDS" | tr '\n' ' ')" >&2
  echo "[ERROR] see 'Failed:' in the worker logs; re-run this script to retry them (it resumes)" >&2
  [[ "$STATUS" -eq 0 ]] && STATUS=3
else
  echo "[SUMMARY] every selected episode has a clip"
fi
exit "$STATUS"
