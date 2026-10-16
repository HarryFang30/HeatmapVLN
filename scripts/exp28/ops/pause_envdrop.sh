#!/usr/bin/env bash
# Stop the EnvDrop collection launchers (PPA_COLLECT_OUTPUT_ROOT=/workspace/exp23/envdrop1) by PID; EXP-28 takes GPUs 0/4/5.
# Same identification as exp26/ckpt_eval_null.sh; a launcher's trap stops its slots, committed records stay.
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a /workspace/exp23/envdrop1/logs/pause_exp28.log; }
for p in $(pgrep -f "^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
  ps -o args= -p "$(ps -o ppid= -p $p | tr -d ' ')" | grep -q run_s2_dagger_collect_cuda && continue
  tr '\0' '\n' < /proc/$p/environ | grep -q "^PPA_COLLECT_OUTPUT_ROOT=/workspace/exp23/envdrop1$" && kill -TERM $p && log "TERM EnvDrop launcher $p (EXP-28 takes the GPUs)"
done
for i in $(seq 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==0 && $2>6000) || (($1==4 || $1==5) && $2>2000)' | wc -l)
  [[ "$busy" -eq 0 ]] && { log "GPUs 0/4/5 free"; exit 0; }
  sleep 10
done
log "GPUs 0/4/5 still busy after 10 min"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a /workspace/exp23/envdrop1/logs/pause_exp28.log
