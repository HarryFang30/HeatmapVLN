#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXP-18 orthographic top-down maps for a tier or a scene list.
#
# NUM_PROCS render processes, each on its own Xvfb + llvmpipe display
# (BASE_DISPLAY + i, via with_xvfb.sh); scenes are sharded i/NUM_PROCS over
# the sorted list.  CPU only (no GPU is touched).  Re-running resumes:
# scenes that already have topdown.json are skipped (OVERWRITE=1 redoes them).
#
# Website submission (blank container, /mnt/afs only):
#   cd /mnt/afs/liwenhao/agent/370910109/<exp18 code checkout>
#   export TIER=C                  # A|B|C|D|E; and/or SCENES="2azQ1b91cZZ kfPV7w3FaU5"
#   export NUM_PROCS=4
#   bash scripts/exp18/topdown/run_topdown.sh
#
# Env:
#   TIER / SCENES     at least one; SCENES is space/comma separated (HM3D ids
#                     with or without .basis)
#   MANIFEST          tier D HM3D selection (scene list, clip list or JSON);
#                     default: $EXP18_ROOT/clip_lists/D.txt
#   EXP18_ROOT        default /mnt/afs/liwenhao/agent/370910109/model/exp18_first_person_viz
#   OUT_DIR           default $EXP18_ROOT/topdown
#   LOG_DIR           default $EXP18_ROOT/logs/topdown_<stamp>  (worker logs + Xvfb runtime)
#   NUM_PROCS (4)  BASE_DISPLAY (370)  LP_NUM_THREADS (8)  OVERWRITE (0)
#   SRC_DIR           code checkout to import from (default: repo holding this script)
#   EXTRA_ARGS        extra render_topdown.py flags, e.g. "--fade-radius 0.8"
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export EXP18_ROOT="${EXP18_ROOT:-${ROOT}/model/exp18_first_person_viz}"
OUT_DIR="${OUT_DIR:-${EXP18_ROOT}/topdown}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${EXP18_ROOT}/logs/topdown_${RUN_STAMP}}"
NUM_PROCS="${NUM_PROCS:-4}"
BASE_DISPLAY="${BASE_DISPLAY:-370}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
TIER="${TIER:-}"
SCENES="${SCENES:-}"
MANIFEST="${MANIFEST:-}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$TIER" && -z "$SCENES" ]]; then
  echo "[ERROR] set TIER (A|B|C|D|E) and/or SCENES" >&2
  exit 1
fi
if [[ "$NUM_PROCS" -le 0 || "$NUM_PROCS" -gt 16 ]]; then
  echo "[ERROR] NUM_PROCS must be 1..16 (llvmpipe processes)" >&2
  exit 1
fi
for path in "${SRC_DIR}/scripts/exp18/topdown/render_topdown.py" "${SCRIPT_DIR}/with_xvfb.sh"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

ARGS=(-m scripts.exp18.topdown.render_topdown --out "$OUT_DIR")
if [[ -n "$TIER" ]]; then ARGS+=(--tier "$TIER"); fi
if [[ -n "$SCENES" ]]; then ARGS+=(--scenes ${SCENES//,/ }); fi
if [[ -n "$MANIFEST" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ "$OVERWRITE" == 1 ]]; then ARGS+=(--overwrite); fi
# shellcheck disable=SC2206
if [[ -n "$EXTRA_ARGS" ]]; then ARGS+=($EXTRA_ARGS); fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec 9>"${OUT_DIR%/}.lock"   # sibling of OUT_DIR, so OUT_DIR holds only scene dirs
if ! flock -n 9; then
  echo "[ERROR] another run_topdown.sh is writing $OUT_DIR" >&2
  exit 1
fi
exec > >(tee -a "${LOG_DIR}/launcher.log") 2>&1

echo "============================================================"
echo "EXP-18 top-down maps (Xvfb + llvmpipe, CPU)"
echo "Source:     $SRC_DIR ($(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo no-git))"
echo "Tier:       ${TIER:-none}   Scenes: ${SCENES:-none}   Manifest: ${MANIFEST:-default}"
echo "Out:        $OUT_DIR"
echo "Logs:       $LOG_DIR"
echo "Processes:  $NUM_PROCS on :${BASE_DISPLAY}..:$((BASE_DISPLAY + NUM_PROCS - 1)), LP threads $LP_NUM_THREADS"
echo "Args:       ${ARGS[*]}"
echo "============================================================"
cd "$SRC_DIR"

declare -a PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap 'cleanup; exit 143' INT TERM

for ((i = 0; i < NUM_PROCS; i++)); do
  display_num=$((BASE_DISPLAY + i))
  echo "[LAUNCH] shard $i/$NUM_PROCS display :$display_num log ${LOG_DIR}/worker_${i}.log"
  DISPLAY_NUM="$display_num" XVFB_RUNTIME_DIR="${LOG_DIR}/xvfb_${display_num}" \
    bash "${SCRIPT_DIR}/with_xvfb.sh" "${ARGS[@]}" --shard "${i}/${NUM_PROCS}" \
    >"${LOG_DIR}/worker_${i}.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[DONE] shard $i"
  else
    echo "[FAILED] shard $i (see ${LOG_DIR}/worker_${i}.log)" >&2
    STATUS=1
  fi
  grep -h "summary:" "${LOG_DIR}/worker_${i}.log" || true
done
echo "[SUMMARY] scenes with topdown.json in $OUT_DIR: $(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -name topdown.json | wc -l)"
exit "$STATUS"
