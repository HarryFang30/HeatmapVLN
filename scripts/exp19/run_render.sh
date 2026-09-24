#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXP-19 [E]: re-render a finished rerun's call-step states in the
# r2r_panoramic_data_v2 collector geometry (4 views F,R,B,L, 256x256 HFOV 90,
# front depth in metres, camera at 1.25 m) -> <EXP19_ROOT>/renders/<ep_key>.{npz,json}.
#
# CPU only: envs/vlnce + one private Xvfb with Mesa llvmpipe, through
# scripts/exp18/topdown/with_xvfb.sh (the certified headless recipe; works in
# a blank container that mounts only /mnt/afs).  Two steps, each on its own
# short-lived Xvfb:
#   1. the mandatory self-check: a stored r2r_panoramic_data_v2 clip rendered
#      at its own poses must reproduce its depth / RGB / pose convention
#      (renders/self_check.json); a failed self-check stops here;
#   2. every episode of runs/<EXP19_RUN> (needs runs/<EXP19_RUN>/DONE).
# Resumable: an episode whose renders/<ep_key>.json exists is skipped if it was
# rendered from this run's steps.jsonl (sha256) at the same steps; a render
# left by another run (renders/ is shared under EXP19_ROOT) is redone.
# EXP19_RENDER_OVERWRITE=1 re-renders everything.
#
# Website submission (blank container, parameters as env vars):
#   cd /mnt/afs/liwenhao/agent/370910109/model/exp19_behavior_viz/src_<sha>
#   export EXP19_RUN=main
#   bash scripts/exp19/run_render.sh
#
# Env: EXP19_RUN (required unless EXP19_RENDER_MODE=self-check),
#   EXP19_ROOT (default <workspace>/model/exp19_behavior_viz),
#   EXP19_RENDER_MODE (run | self-check; default run = self-check, then render),
#   EXP19_RENDER_OUT (default <EXP19_ROOT>/renders), EXP19_EPISODES (space
#   separated ep_keys; default all), EXP19_RENDER_OVERWRITE (0),
#   EXP19_RENDER_DISPLAY (390), LP_NUM_THREADS (8).
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP19_ROOT="${EXP19_ROOT:-${ROOT}/model/exp19_behavior_viz}"
MODE="${EXP19_RENDER_MODE:-run}"
OUT="${EXP19_RENDER_OUT:-${EXP19_ROOT}/renders}"
DISPLAY_NUM="${EXP19_RENDER_DISPLAY:-390}"
LOG_DIR="${OUT}/logs"
WITH_XVFB="${REPO_ROOT}/scripts/exp18/topdown/with_xvfb.sh"

export PYTHONDONTWRITEBYTECODE=1
export EXP19_ROOT

case "$MODE" in
  run|self-check) ;;
  *) echo "[ERROR] EXP19_RENDER_MODE must be run or self-check (got '${MODE}')" >&2; exit 1 ;;
esac
RUN_DIR=""
if [[ "$MODE" == run ]]; then
  if [[ -z "${EXP19_RUN:-}" ]]; then
    echo "[ERROR] EXP19_RUN is required (the rerun under ${EXP19_ROOT}/runs/)" >&2
    exit 1
  fi
  RUN_DIR="${EXP19_ROOT}/runs/${EXP19_RUN}"
  if [[ ! -f "${RUN_DIR}/DONE" ]]; then
    echo "[ERROR] ${RUN_DIR}/DONE missing: the rerun has not finished" >&2
    exit 1
  fi
fi
for path in "$WITH_XVFB" "${REPO_ROOT}/scripts/exp19/render_views.py"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LOG_DIR}/render_${MODE}_${EXP19_RUN:-none}_${RUN_STAMP}.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "============================================================"
echo "EXP-19 re-render (collector geometry, Xvfb + llvmpipe)"
echo "============================================================"
echo "Code:     $REPO_ROOT (git sha: $(cat "${REPO_ROOT}/.exp19_git_sha" 2>/dev/null || echo unknown))"
echo "Mode:     $MODE"
echo "Run:      ${RUN_DIR:-<none>}"
echo "Output:   $OUT"
echo "Display:  :${DISPLAY_NUM}"
echo "Log:      $LAUNCH_LOG"
echo "============================================================"

cd "$REPO_ROOT"
render() {
  DISPLAY_NUM="$DISPLAY_NUM" XVFB_RUNTIME_DIR="${LOG_DIR}/xvfb_${RUN_STAMP}/display_${DISPLAY_NUM}" \
    bash "$WITH_XVFB" scripts/exp19/render_views.py --out-dir "$OUT" "$@"
}

echo "[STEP] self-check against a stored r2r_panoramic_data_v2 clip"
if ! render --self-check; then
  echo "[ERROR] self-check failed (see ${OUT}/self_check.json): renders would not match the label geometry" >&2
  exit 1
fi
if [[ "$MODE" == self-check ]]; then
  exit 0
fi

echo "[STEP] render ${RUN_DIR}"
args=(--run-dir "$RUN_DIR")
if [[ -n "${EXP19_EPISODES:-}" ]]; then
  # shellcheck disable=SC2206
  args+=(--episodes ${EXP19_EPISODES})
fi
if [[ "${EXP19_RENDER_OVERWRITE:-0}" == 1 ]]; then
  args+=(--overwrite)
fi
render "${args[@]}"
echo "[DONE] renders in $OUT"
