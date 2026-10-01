#!/usr/bin/env bash
# EXP-20 scheduler, inside fjl-habitat (user 2026-09-30: start an arm's full runs as soon as its canary passes).
# Jobs in priority order; each takes one free GPU (no other process: under 500 MiB used; GPU 0 also under
# 4000 MiB, its idle foreign process approved by the user).  An arm's full runs wait for its canary verdict and
# are skipped if the canary failed.  A job that ends (any exit code) is not restarted; state in $STATE.
# Rerunning this script resumes: finished, running-elsewhere and blocked jobs are skipped by their state files.
set -o pipefail  # no -u: bash 5.0 calls an empty associative array unbound
SRC=/workspace/exp20/src_2441392
W=/workspace/weights_exp20
LOG="${ORCH_LOG:-/workspace/exp20/logs}"
STATE="${ORCH_STATE:-/workspace/exp20/state}"
DRY="${ORCH_DRY:-0}"
POLL="${ORCH_POLL:-300}"
GAP="${ORCH_LAUNCH_GAP:-180}"   # let a job's servers take their memory before the next scan
PLAN=/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802
mkdir -p "$LOG" "$STATE"
JOBS=(canary_a1 canary_a2 canary_a3 a1_42 a2_42 a3_42 a0_1337 a1_1337 a2_1337 a3_1337)
declare -A PID=() GPU=()

gpu_table() {
  if [[ "$DRY" == 1 ]]; then cat "$STATE/fake_gpus"; else nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; fi
}
free_gpu() {
  local claimed=" ${GPU[*]:-} " g
  for g in $(gpu_table | awk -F', ' '($2 < 500) || ($1 == 0 && $2 < 4000) {print $1}'); do
    [[ "$claimed" == *" $g "* ]] || { echo "$g"; return; }
  done
}
arm_of() { local j="${1#canary_}"; echo "${j%%_*}"; }
seed_of() { if [[ "$1" == canary_* ]]; then echo 42; else echo "${1##*_}"; fi; }
verdict_of() { echo "$LOG/canary_$(arm_of "$1").verdict.json"; }
prereq() {  # 0 go, 1 wait, 2 blocked
  [[ "$1" == canary_* || "$1" == a0_* ]] && return 0
  local v; v=$(verdict_of "$1")
  [[ -s "$v" ]] || return 1
  /opt/conda/bin/python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] is True else 3)' "$v" \
    && return 0 || return 2
}
launch() {
  local j="$1" g="$2" k="$3" arm seed out label
  local -a extra=()
  arm=$(arm_of "$j"); seed=$(seed_of "$j")
  case "$arm" in
    a1) extra=(PPA_EVAL_BRIDGE_OFF=1) ;;
    a2) extra=(PPA_EVAL_CHECKPOINT="$W/exp05_v1_unconstrained_bridge_best_deployment_full.pth"
               PPA_EVAL_CONFIG="$SRC/configs/ppa_action_refine_8gpu.yaml") ;;
    a3) extra=(PPA_EVAL_CHECKPOINT="$W/exp09c_stage3_ablation_best.pth") ;;
  esac
  if [[ "$j" == canary_* ]]; then
    extra+=(PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2)
    out="/workspace/eval_runs/exp20_${arm}_canary"; label="exp20_${arm}_canary"
  else
    out="/workspace/eval_runs/exp20_${arm}_seed${seed}_4090"; label="exp20_${arm}_seed${seed}"
  fi
  (
    cd "$SRC" || exit 2
    if [[ "$DRY" == 1 ]]; then
      echo "DRY $j gpu=$g seed=$seed ${extra[*]} out=$out ports=$((52600 + 10 * k))/$((52700 + 10 * k)) display=$((390 + 10 * k))"
      sleep "${ORCH_DRY_SECONDS:-3}"; rc=0
    else
      env "${extra[@]}" PPA_EVAL_REPO="$SRC" PPA_EVAL_GPU_DEVICES="$g" PPA_EVAL_PROTOCOL_SEED="$seed" \
        PPA_EVAL_ARM="$label" PPA_EVAL_OUTPUT_ROOT="$out" PPA_EVAL_MODEL_PORT_BASE=$((52600 + 10 * k)) \
        PPA_EVAL_VO_PORT_BASE=$((52700 + 10 * k)) PPA_EVAL_DISPLAY_BASE=$((390 + 10 * k)) \
        bash scripts/run_ppa_r2r_val_unseen_cuda.sh
      rc=$?
    fi
    if [[ "$j" == canary_* ]]; then
      if [[ "$DRY" == 1 ]]; then
        p=true; [[ " ${ORCH_DRY_FAIL:-} " == *" $arm "* ]] && p=false
        echo "{\"arm\": \"$arm\", \"pass\": $p}" > "$(verdict_of "$j")"
      else
        PYTHONPATH="$PLAN/tools:/workspace/rpc/src:$SRC" /opt/conda/bin/python /workspace/exp20/canary_check.py \
          "$SRC" "$out" "$arm" "$(verdict_of "$j")"
      fi
    fi
    [[ "$DRY" == 1 ]] || chown -R 1015:1015 "$out" 2>/dev/null
    echo "$rc" > "$STATE/$j.exit"
  ) >"$LOG/$j.out" 2>&1 &
  PID[$j]=$!; GPU[$j]="$g"
  echo "$g $!" > "$STATE/$j.running"
}
echo "[orch] $(date '+%F %T') start dry=$DRY"
while true; do
  for j in "${!PID[@]}"; do
    if ! kill -0 "${PID[$j]}" 2>/dev/null; then
      wait "${PID[$j]}" 2>/dev/null
      echo "[orch] $(date '+%F %T') $j done exit=$(cat "$STATE/$j.exit" 2>/dev/null || echo '?') gpu=${GPU[$j]}"
      rm -f "$STATE/$j.running"; unset "PID[$j]" "GPU[$j]"
    fi
  done
  pending=0
  for k in "${!JOBS[@]}"; do
    j="${JOBS[$k]}"
    [[ -n "${PID[$j]:-}" || -f "$STATE/$j.exit" || -f "$STATE/$j.blocked" || -f "$STATE/$j.running" ]] && continue
    prereq "$j"; p=$?
    if [[ "$p" -eq 2 ]]; then
      touch "$STATE/$j.blocked"; echo "[orch] $(date '+%F %T') $j skipped: canary $(arm_of "$j") failed"; continue
    fi
    pending=$((pending + 1))
    [[ "$p" -eq 1 ]] && continue
    g=$(free_gpu); [[ -z "$g" ]] && continue
    launch "$j" "$g" "$k"
    echo "[orch] $(date '+%F %T') $j started on GPU $g"
    sleep "$GAP"
  done
  (( pending == 0 && ${#PID[@]} == 0 )) && break
  sleep "$POLL"
done
echo "[orch] $(date '+%F %T') all jobs ended or skipped"
