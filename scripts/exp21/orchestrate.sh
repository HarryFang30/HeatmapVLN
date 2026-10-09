#!/usr/bin/env bash
# EXP-21 scheduler, inside fjl-habitat (user 2026-10-09: run the sensitivity points on the 4090).  One job per
# point, A0 weights, seed 42, the fixed 500-episode subset; each job takes one free GPU (under 1000 MiB used, i.e.
# at most an idle viewer; GPU 0 under 6000 MiB: the user approved sharing it with its foreign processes, 2026-10-09).  Order: the
# extremes first (S=1, M=2, K=5), so a cut-short schedule still has both ends of every curve.  A job that ends (any
# exit code) is not restarted; state in $STATE; rerunning resumes.  docs/experiments/exp21-sensitivity-runbook.md.
set -o pipefail  # no -u: bash 5.0 calls an empty associative array unbound
SRC="${ORCH_SRC:-/workspace/exp21/src_0a8c9b2}"
SUBSET=/workspace/exp21/subset500
LOG="${ORCH_LOG:-/workspace/exp21/logs}"
STATE="${ORCH_STATE:-/workspace/exp21/state}"
DRY="${ORCH_DRY:-0}"
POLL="${ORCH_POLL:-300}"
GAP="${ORCH_LAUNCH_GAP:-180}"
mkdir -p "$LOG" "$STATE"
JOBS=(s1 m2 k5 s8 m5 k6 s64 m20 k7)
declare -A PID=() GPU=()

gpu_table() {
  if [[ "$DRY" == 1 ]]; then cat "$STATE/fake_gpus"; else nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; fi
}
free_gpu() {
  local claimed=" ${GPU[*]:-} " g
  for g in $(gpu_table | awk -F', ' '($2 < 1000) || ($1 == 0 && $2 < 6000) {print $1}'); do
    [[ "$claimed" == *" $g "* ]] || { echo "$g"; return; }
  done
}
switch_of() {
  case "$1" in
    s*) echo "PPA_EVAL_NUM_SAMPLE_TRAJS=${1#s}" ;;
    m*) echo "PPA_EVAL_NUM_INFERENCE_STEPS=${1#m}" ;;
    k*) echo "PPA_EVAL_NUM_HISTORY=${1#k}" ;;
  esac
}
launch() {
  local j="$1" g="$2" k="$3" sw out label
  sw=$(switch_of "$j")
  out="/workspace/eval_runs/exp21_${j}_seed42_4090"; label="exp21_${j}_seed42"
  (
    cd "$SRC" || exit 2
    if [[ "$DRY" == 1 ]]; then
      echo "DRY $j gpu=$g $sw out=$out ports=$((52800 + 10 * k))/$((52900 + 10 * k)) display=$((500 + 10 * k))"
      sleep "${ORCH_DRY_SECONDS:-3}"; rc=0
    else
      env "$sw" PPA_EVAL_EPISODE_LISTS_DIR="$SUBSET" PPA_EVAL_REPO="$SRC" PPA_EVAL_GPU_DEVICES="$g" \
        PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_ARM="$label" PPA_EVAL_OUTPUT_ROOT="$out" \
        PPA_EVAL_MODEL_PORT_BASE=$((52800 + 10 * k)) PPA_EVAL_VO_PORT_BASE=$((52900 + 10 * k)) \
        PPA_EVAL_DISPLAY_BASE=$((500 + 10 * k)) bash scripts/run_ppa_r2r_val_unseen_cuda.sh
      rc=$?
      chown -R 1015:1015 "$out" 2>/dev/null
    fi
    echo "$rc" > "$STATE/$j.exit"
  ) >"$LOG/$j.out" 2>&1 &
  PID[$j]=$!; GPU[$j]="$g"
  echo "$g $!" > "$STATE/$j.running"
}
echo "[orch21] $(date '+%F %T') start dry=$DRY src=$SRC"
while true; do
  for j in "${!PID[@]}"; do
    if ! kill -0 "${PID[$j]}" 2>/dev/null; then
      wait "${PID[$j]}" 2>/dev/null
      echo "[orch21] $(date '+%F %T') $j done exit=$(cat "$STATE/$j.exit" 2>/dev/null || echo '?') gpu=${GPU[$j]}"
      rm -f "$STATE/$j.running"; unset "PID[$j]" "GPU[$j]"
    fi
  done
  pending=0
  for k in "${!JOBS[@]}"; do
    j="${JOBS[$k]}"
    [[ -n "${PID[$j]:-}" || -f "$STATE/$j.exit" || -f "$STATE/$j.running" ]] && continue
    pending=$((pending + 1))
    g=$(free_gpu); [[ -z "$g" ]] && continue
    launch "$j" "$g" "$k"
    echo "[orch21] $(date '+%F %T') $j started on GPU $g"
    sleep "$GAP"
  done
  (( pending == 0 && ${#PID[@]} == 0 )) && break
  sleep "$POLL"
done
echo "[orch21] $(date '+%F %T') all points ended"
