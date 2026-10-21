#!/usr/bin/env bash
# EXP-33 scheduler, inside fjl-habitat: 4 camera viewpoints x 2 arms (A0, A1 = bridge off), seed 42, the
# fixed 500-episode subset.  Each job takes one free GPU (under 1000 MiB used; GPU 0 under 6000 MiB, the
# EXP-21 rule the user approved 2026-10-09) from ORCH_GPUS, checked right before its launch.  Order: A0 V0,
# V1, V3, V2, then A1 the same, so a cut-short schedule still has the main question and both extremes.
# While $HOLD exists, or ORCH_START_WHEN (a shell command) fails, no job starts (running ones go on): EXP-32's
# 4090 controller uses GPUs 5,4,0 by name and is not told about anyone else.  A job that ends (any exit code)
# is not restarted; state in $STATE; rerunning resumes.  Criteria and settings: docs/experiments/README.md, EXP-33.
set -o pipefail  # no -u: bash 5.0 calls an empty associative array unbound
SRC="${ORCH_SRC:?ORCH_SRC=/workspace/exp33/src_<commit>}"
SUBSET=/workspace/exp21/subset500
ROOT=/workspace/exp33
LOG="${ORCH_LOG:-$ROOT/logs}"
STATE="${ORCH_STATE:-$ROOT/state}"
HOLD="${ORCH_HOLD:-$ROOT/HOLD}"
GPUS=" ${ORCH_GPUS:-0 1 2 3 4 5} "
START_WHEN="${ORCH_START_WHEN:-true}"
DRY="${ORCH_DRY:-0}"
POLL="${ORCH_POLL:-300}"
GAP="${ORCH_LAUNCH_GAP:-180}"
mkdir -p "$LOG" "$STATE"
JOBS=(a0_v0 a0_v1 a0_v3 a0_v2 a1_v0 a1_v1 a1_v3 a1_v2)
declare -A PID=() GPU=()

gpu_table() {
  if [[ "$DRY" == 1 ]]; then cat "$STATE/fake_gpus"; else nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; fi
}
free_gpu() {
  local claimed=" ${GPU[*]:-} " g
  for g in $(gpu_table | awk -F', ' '($2 < 1000) || ($1 == 0 && $2 < 6000) {print $1}'); do
    [[ "$GPUS" == *" $g "* && "$claimed" != *" $g "* ]] && { echo "$g"; return; }
  done
}
switches_of() {  # the pre-registered viewpoints and arms
  case "${1#*_}" in
    v0) ;;
    v1) echo PPA_EVAL_CAMERA_HEIGHT_M=0.6 PPA_EVAL_CAMERA_PITCH_DEG=-15 PPA_EVAL_LOOKDOWN_MODE=same_frame ;;
    v2) echo PPA_EVAL_CAMERA_HEIGHT_M=0.7 PPA_EVAL_CAMERA_PITCH_DEG=-15 PPA_EVAL_LOOKDOWN_MODE=same_frame ;;
    v3) echo PPA_EVAL_CAMERA_HEIGHT_M=0.4 PPA_EVAL_CAMERA_PITCH_DEG=0 PPA_EVAL_LOOKDOWN_MODE=same_frame ;;
  esac
  [[ "${1%%_*}" == a1 ]] && echo PPA_EVAL_BRIDGE_OFF=1
}
launch() {
  local j="$1" g="$2" k="$3" out label
  local -a sw
  read -r -a sw <<< "$(switches_of "$j" | tr '\n' ' ')"
  out="/workspace/eval_runs/exp33_${j}_seed42_4090"; label="exp33_${j}_seed42"
  (
    cd "$SRC" || exit 2
    if [[ "$DRY" == 1 ]]; then
      echo "DRY $j gpu=$g ${sw[*]} out=$out ports=$((53100 + 10 * k))/$((53200 + 10 * k)) display=$((600 + 10 * k))"
      sleep "${ORCH_DRY_SECONDS:-3}"; rc=0
    else
      env "${sw[@]}" PPA_EVAL_EPISODE_LISTS_DIR="$SUBSET" PPA_EVAL_REPO="$SRC" PPA_EVAL_GPU_DEVICES="$g" \
        PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_ARM="$label" PPA_EVAL_OUTPUT_ROOT="$out" \
        PPA_EVAL_MODEL_PORT_BASE=$((53100 + 10 * k)) PPA_EVAL_VO_PORT_BASE=$((53200 + 10 * k)) \
        PPA_EVAL_DISPLAY_BASE=$((600 + 10 * k)) bash scripts/run_ppa_r2r_val_unseen_cuda.sh
      rc=$?
      chown -R 1015:1015 "$out" 2>/dev/null
    fi
    echo "$rc" > "$STATE/$j.exit"
  ) >"$LOG/$j.out" 2>&1 &
  PID[$j]=$!; GPU[$j]="$g"
  echo "$g $!" > "$STATE/$j.running"
}
echo "[orch33] $(date -u '+%FT%TZ') start dry=$DRY src=$SRC gpus=$GPUS hold=$HOLD start_when=$START_WHEN"
waiting=0
while true; do
  for j in "${!PID[@]}"; do
    if ! kill -0 "${PID[$j]}" 2>/dev/null; then
      wait "${PID[$j]}" 2>/dev/null
      echo "[orch33] $(date -u '+%FT%TZ') $j done exit=$(cat "$STATE/$j.exit" 2>/dev/null || echo '?') gpu=${GPU[$j]}"
      rm -f "$STATE/$j.running"; unset "PID[$j]" "GPU[$j]"
    fi
  done
  pending=0
  ready=1
  [[ -e "$HOLD" ]] && ready=0
  (( ready )) && { bash -c "$START_WHEN" >/dev/null 2>&1 || ready=0; }
  if (( ready != 1 - waiting )); then
    waiting=$((1 - ready)); echo "[orch33] $(date -u '+%FT%TZ') $( ((ready)) && echo 'starting jobs' || echo 'holding: HOLD or ORCH_START_WHEN')"
  fi
  for k in "${!JOBS[@]}"; do
    j="${JOBS[$k]}"
    [[ -n "${PID[$j]:-}" || -f "$STATE/$j.exit" || -f "$STATE/$j.running" ]] && continue
    pending=$((pending + 1))
    (( ready )) || continue
    g=$(free_gpu); [[ -z "$g" ]] && continue
    launch "$j" "$g" "$k"
    echo "[orch33] $(date -u '+%FT%TZ') $j started on GPU $g"
    sleep "$GAP"
  done
  (( pending == 0 && ${#PID[@]} == 0 )) && break
  sleep "$POLL"
done
echo "[orch33] $(date -u '+%FT%TZ') all jobs ended"
