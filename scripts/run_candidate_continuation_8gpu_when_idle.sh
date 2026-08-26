#!/usr/bin/env bash
set -Eeuo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
REPO="$FJL_ROOT/HeatmapVLN"
MODEL_ROOT="$FJL_ROOT/model/candidate_continuation_v1"
LAUNCHER="$REPO/scripts/run_candidate_continuation_8gpu_mxc500.sh"

CURRENT_PID_FILE="${1:?usage: $0 CURRENT_PID_FILE CURRENT_LOG}"
CURRENT_LOG="${2:?usage: $0 CURRENT_PID_FILE CURRENT_LOG}"
POLL_SECONDS="${CONTINUATION_IDLE_POLL_SECONDS:-60}"
IDLE_THRESHOLD_KB="${CONTINUATION_IDLE_THRESHOLD_KB:-4194304}"
IDLE_CONFIRMATIONS="${CONTINUATION_IDLE_CONFIRMATIONS:-3}"
WAIT_LOCK="$MODEL_ROOT/.continuation_auto8_wait.lock"

for path in "$CURRENT_PID_FILE" "$CURRENT_LOG" "$LAUNCHER"; do
  case "$path" in
    "$FJL_ROOT"/*) ;;
    *) echo "Refusing path outside $FJL_ROOT: $path" >&2; exit 1 ;;
  esac
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

if ! mkdir "$WAIT_LOCK" 2>/dev/null; then
  echo "Another continuation auto-8 waiter already owns $WAIT_LOCK" >&2
  exit 1
fi
trap 'rmdir "$WAIT_LOCK" 2>/dev/null || true' EXIT

CURRENT_PID="$(tr -d '[:space:]' < "$CURRENT_PID_FILE")"
[[ "$CURRENT_PID" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid current launcher PID: $CURRENT_PID" >&2
  exit 1
}

echo "[auto8] waiting for single-GPU shard-0 launcher pid=$CURRENT_PID"
while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

if ! grep -Fq '[audit] COMPLETE' "$CURRENT_LOG"; then
  echo "[auto8] shard-0 launcher ended without [audit] COMPLETE; refusing automatic 8-GPU launch" >&2
  exit 1
fi

echo "[auto8] shard 0 sealed; waiting for all 8 GPUs to remain below ${IDLE_THRESHOLD_KB} KiB"
stable=0
while (( stable < IDLE_CONFIRMATIONS )); do
  mapfile -t used_kb < <(
    mx-smi --show-memory 2>/dev/null |
      awk '/vis_vram used/ {print $(NF-1)}'
  )
  all_idle=1
  if (( ${#used_kb[@]} != 8 )); then
    all_idle=0
  else
    for value in "${used_kb[@]}"; do
      if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value >= IDLE_THRESHOLD_KB )); then
        all_idle=0
        break
      fi
    done
  fi
  if (( all_idle )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  echo "[auto8] $(date -Iseconds) used_kb=${used_kb[*]:-unavailable} idle_confirmations=$stable/$IDLE_CONFIRMATIONS"
  if (( stable < IDLE_CONFIRMATIONS )); then
    sleep "$POLL_SECONDS"
  fi
done

STAMP="auto8_resume_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$MODEL_ROOT/formal_logs"
FULL_LOG="$LOG_DIR/${STAMP}.log"
mkdir -p "$LOG_DIR"

echo "[auto8] launching 8-GPU smoke + resumable formal collection: $FULL_LOG"
cd "$REPO"
unset EVAL_PREFLIGHT_ONLY EVAL_SMOKE_ONLY EVAL_SKIP_SMOKE AUDIT_SKIP_COHORT_VERIFY
export AUDIT_NUM_SHARDS=8
export EVAL_GPU_DEVICES=0,1,2,3,4,5,6,7
export EVAL_RPC_PORT_BASE=51700
export EVAL_DISPLAY_BASE=320
export EVAL_RUN_STAMP="$STAMP"

set +e
bash "$LAUNCHER" >"$FULL_LOG" 2>&1
rc=$?
set -e
echo "[auto8] 8-GPU launcher exited rc=$rc log=$FULL_LOG"
exit "$rc"
