#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run one envs/vlnce python command on a private Xvfb display with Mesa
# llvmpipe GLX, then stop that Xvfb (also on TERM/INT).  Same certified
# recipe as scripts/run_collect_panoramic_mxc500.sh: bundle Xvfb with the
# fd-9 xkb cache, renderer checked to be llvmpipe, client on the bundle's
# mesa_lib.  Works in a blank container that mounts only /mnt/afs.
#
#   DISPLAY_NUM=371 bash scripts/exp18/topdown/with_xvfb.sh \
#       -m scripts.exp18.topdown.render_topdown --scenes 2azQ1b91cZZ
#
# All arguments go to the vlnce python (run from the current directory).
# Env: DISPLAY_NUM (370), XVFB_RUNTIME_DIR (/tmp/exp18_xvfb/display_<N>),
#      LP_NUM_THREADS (8), VLNCE_PYTHON, X11_BUNDLE.
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
X11_BUNDLE="${X11_BUNDLE:-${ROOT}/tools/x11_headless_bundle_ubuntu22_20260801_v4}"
VLNCE_PYTHON="${VLNCE_PYTHON:-${ROOT}/envs/vlnce/bin/python}"
DISPLAY_NUM="${DISPLAY_NUM:-370}"
DISPLAY_ADDR="localhost:${DISPLAY_NUM}.0"
RUNTIME_DIR="${XVFB_RUNTIME_DIR:-/tmp/exp18_xvfb/display_${DISPLAY_NUM}}"
LP_THREADS="${LP_NUM_THREADS:-8}"

X11_DRI_PATH="${X11_BUNDLE}/dri"
TOOL_LD="${X11_BUNDLE}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CLIENT_LD="${X11_BUNDLE}/mesa_lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
GL_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u EGL_PLATFORM
  -u __EGL_VENDOR_LIBRARY_FILENAMES -u __GLX_VENDOR_LIBRARY_NAME
  -u LIBGL_ALWAYS_INDIRECT -u MESA_LOADER_DRIVER_OVERRIDE -u LIBGL_DRIVERS_PATH
)

for path in "${X11_BUNDLE}/bin/Xvfb" "${X11_BUNDLE}/bin/xdpyinfo" "${X11_BUNDLE}/bin/glxinfo" \
    "${X11_DRI_PATH}/swrast_dri.so" "$VLNCE_PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

xdpy() {
  env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$TOOL_LD" DISPLAY="$DISPLAY_ADDR" \
    timeout 5 "${X11_BUNDLE}/bin/xdpyinfo" >/dev/null 2>&1
}
if xdpy; then
  echo "[ERROR] DISPLAY $DISPLAY_ADDR is already active; choose another DISPLAY_NUM" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}/.xkb-cache"
(
  cd "$RUNTIME_DIR"
  exec 9<"${RUNTIME_DIR}/.xkb-cache"
  exec env "${GL_ENV_UNSET_ARGS[@]}" \
    PATH="${X11_BUNDLE}/bin:${PATH}" LD_LIBRARY_PATH="$TOOL_LD" LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
    LP_NUM_THREADS="$LP_THREADS" \
    "${X11_BUNDLE}/bin/Xvfb" ":${DISPLAY_NUM}" \
    -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
    -fp "${X11_BUNDLE}/share/fonts/misc" -xkbdir "${X11_BUNDLE}/share/X11/xkb"
) >"${RUNTIME_DIR}/xvfb.log" 2>&1 &
XVFB_PID=$!
CLIENT_PID=""

cleanup() {
  if [[ -n "$CLIENT_PID" ]]; then
    kill "$CLIENT_PID" 2>/dev/null || true
  fi
  kill "$XVFB_PID" 2>/dev/null || true
  wait "$XVFB_PID" 2>/dev/null || true
  echo "[XVFB] :${DISPLAY_NUM} stopped"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "[ERROR] Xvfb :${DISPLAY_NUM} exited during startup" >&2
    tail -50 "${RUNTIME_DIR}/xvfb.log" >&2 || true
    exit 1
  fi
  if xdpy; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "[ERROR] Xvfb :${DISPLAY_NUM} did not become ready" >&2
  exit 1
fi
renderer="$(env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$TOOL_LD" LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  DISPLAY="$DISPLAY_ADDR" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
  timeout 120 "${X11_BUNDLE}/bin/glxinfo" -B 2>/dev/null | grep -F 'OpenGL renderer string:' | head -1 || true)"
if [[ "${renderer,,}" != *llvmpipe* ]]; then
  echo "[ERROR] DISPLAY $DISPLAY_ADDR is not llvmpipe: ${renderer:-no renderer}" >&2
  exit 1
fi
echo "[XVFB] :${DISPLAY_NUM} ready (${renderer#*: })"

# The client runs in the background so TERM/INT reach the trap immediately.
ulimit -c 0
env "${GL_ENV_UNSET_ARGS[@]}" \
  DISPLAY="$DISPLAY_ADDR" LD_LIBRARY_PATH="$CLIENT_LD" LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
  LP_NUM_THREADS="$LP_THREADS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 GLOG_minloglevel="${GLOG_minloglevel:-2}" MAGNUM_LOG="${MAGNUM_LOG:-quiet}" \
  "$VLNCE_PYTHON" -u "$@" &
CLIENT_PID=$!
set +e
wait "$CLIENT_PID"
status=$?
set -e
CLIENT_PID=""
exit "$status"
