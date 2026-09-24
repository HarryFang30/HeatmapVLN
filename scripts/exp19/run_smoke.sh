#!/usr/bin/env bash
# EXP-19 smoke: scripts/exp19/run_rerun.sh on ONE GPU with ONE hand-picked episode list.
#
# Stages EXP19_SMOKE_LIST as gpu0.json in a private temp dir and runs the full launcher
# on it (same servers, same client flags, same checks, same runs/<run>/DONE), so a smoke
# run exercises exactly the code path of the real rerun. The launcher copies the list
# into runs/<run>/gpu0/episode_list.json, so the temp dir is removed afterwards.
#
#   cd <EXP19_SRC>
#   EXP19_SRC=$PWD EXP19_GPUS=0 EXP19_SMOKE_LIST=/path/to/smoke.json bash scripts/exp19/run_smoke.sh
# Env:
#   EXP19_SMOKE_LIST   client episode list {"cohort_name", "episodes": [{"scene_id", "episode_id"}]} (required)
#   EXP19_GPUS         exactly one physical GPU id (required)
#   EXP19_RUN          default smoke_<UTC timestamp>
#   EXP19_SRC, EXP19_ROOT, EXP19_TRACE_DIAGNOSTICS, EXP19_DRY_RUN, ...   passed through to run_rerun.sh
set -euo pipefail
trap '' PIPE  # like run_rerun.sh: a closed stdout must not kill this shell before its cleanup

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
LIST="${EXP19_SMOKE_LIST:?set EXP19_SMOKE_LIST to a client episode list JSON}"
GPU="${EXP19_GPUS:?set EXP19_GPUS to exactly one GPU id}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "[exp19-smoke] EXP19_GPUS must be exactly one GPU id, got '$GPU'" >&2; exit 2; }
[[ -s "$LIST" ]] || { echo "[exp19-smoke] missing episode list: $LIST" >&2; exit 2; }
LIST="$(cd "$(dirname "$LIST")" && pwd -P)/$(basename "$LIST")"
export EXP19_SMOKE_LIST="$LIST"  # recorded in DONE
export EXP19_RUN="${EXP19_RUN:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"

LISTS=$(mktemp -d "${TMPDIR:-/tmp}/exp19_smoke_lists.XXXXXX")
cp "$LIST" "$LISTS/gpu0.json"
child=""
cleanup() {
  local status=$?
  set +e
  trap - EXIT
  trap '' INT TERM HUP
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true  # the launcher's own trap stops its servers
    wait "$child" 2>/dev/null || true
  fi
  rm -rf "$LISTS"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' INT TERM
trap 'exit 129' HUP

echo "[exp19-smoke] run=$EXP19_RUN gpu=$GPU list=$LIST" || true
EXP19_LISTS="$LISTS" bash "$HERE/run_rerun.sh" &
child=$!
wait "$child"
