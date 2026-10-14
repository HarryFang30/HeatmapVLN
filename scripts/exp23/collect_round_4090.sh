#!/usr/bin/env bash
# EXP-23, 4090 half of round N's collection (N >= 2): the rest of round N-1's cohort,
# collected through the 910B's servers that serve round N-1's adapter
# (scripts/exp23/collect_servers_910b.sh).  Inside the fjl-habitat container, beside
# round N-1's closed-loop evaluation, which keeps the local GPUs: the clients here render
# on the CPU and only take CUDA_VISIBLE_DEVICES from PPA_EVAL_GPU_DEVICES, as in round 1.
#
#   EXP23_ROUND=2 EXP23_WT=/workspace/exp23/wt_<commit> \
#     nohup setsid bash scripts/exp23/collect_round_4090.sh > /dev/null 2>&1 &
#
# Phases:
#   1. wait for round N-1's adapter on this box (round_post_4090.sh pulls it only if the
#      910B's gates passed, so a failed gate leaves this script waiting, collecting nothing);
#   2. round N's cohort = round N-1's cohort minus its committed records
#      (scripts/tools/remaining_cohort.py);
#   3. pull export/npu_servers_r<N> (the servers' runtime evidence) from the export port,
#      always afresh: the 910B withdraws it whenever its server set ends, so a rerun after
#      a restart waits for the new set instead of starting against the old one;
#   4. run_s2_dagger_collect_cuda.sh against those servers with PPA_EVAL_SYSTEM2_LORA set,
#      so it refuses servers that do not serve this adapter.  COLLECT_STATUS carries the
#      record and shard counts every 5 min, and DEGRADED as soon as a shard fails (a
#      retired NPU slot or a dropped tunnel fails shards long before the launcher ends);
#      the launcher keeps going, and a rerun collects the failed shards (--resume).

set -uo pipefail

ROUND="${EXP23_ROUND:?set EXP23_ROUND to the round being collected, e.g. 2}"
PREV=$((ROUND - 1))
WT="${EXP23_WT:?set EXP23_WT to the pinned code worktree, e.g. /workspace/exp23/wt_<commit>}"
PREV_DIR="${EXP23_PREV_DIR:-/workspace/exp23/round$PREV}"
ROUND_DIR="${EXP23_ROUND_DIR:-/workspace/exp23/round$ROUND}"
PYTHON="${EXP23_PYTHON:-/opt/conda/bin/python}"
EXPORT_PORT="${EXP23_EXPORT_PORT:-52417}"
CLIENT_GPUS="${EXP23_CLIENT_GPUS:-5,4,0,5,4,0,5}"  # one entry per 910B slot
DISPLAY_BASE="${EXP23_DISPLAY_BASE:-601}"           # the evaluation uses 380 and 400+
export PYTHONPATH="$WT"

(( ROUND >= 2 )) || { echo "EXP23_ROUND must be >= 2" >&2; exit 2; }
mkdir -p "$ROUND_DIR/logs"
LOG="$ROUND_DIR/logs/collect.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$ROUND_DIR/COLLECT_STATUS"; log "STATUS $*"; }
die() { status "STOPPED: $*"; chown -R 1015:1015 "$ROUND_DIR" 2>/dev/null; exit 2; }
git -c safe.directory='*' -C "$WT" diff --quiet HEAD -- || die "$WT has uncommitted changes"
log "round=$ROUND wt=$WT commit=$(git -c safe.directory='*' -C "$WT" rev-parse HEAD)"

# 1 -------------------------------------------------------------------------------
ADAPTER="$PREV_DIR/adapter_pull/adapter_r$PREV"
status "waiting for adapter_r$PREV (pulled by round_post_4090.sh once the gates pass)"
until [[ -e "$ADAPTER/.pulled" ]]; do sleep 120; done
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || die "the pulled adapter has no weights"
log "adapter_r$PREV: $(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-16)"

# 2 -------------------------------------------------------------------------------
if [[ ! -s "$ROUND_DIR/cohorts/manifest.json" ]]; then
  status "building round $ROUND's cohort from what round $PREV left"
  "$PYTHON" "$WT/scripts/tools/remaining_cohort.py" --cohort "$PREV_DIR/cohorts" --records "$PREV_DIR/records" \
    --out "$ROUND_DIR/cohorts" >> "$LOG" 2>&1 || die "could not build the cohort"
fi

# 3 -------------------------------------------------------------------------------
SERVERS="$ROUND_DIR/npu_servers/npu_servers_r$ROUND"
if [[ -e "$SERVERS" ]]; then  # an earlier run's copy; keep it for the record
  mv "$SERVERS" "$SERVERS.stale_$(date +%Y%m%d_%H%M%S)"
fi
rm -rf "$ROUND_DIR/npu_servers/.npu_servers_r$ROUND.part"
status "waiting for the 910B's round $ROUND servers"
until [[ -e "$SERVERS/.pulled" ]]; do
  "$PYTHON" "$WT/scripts/exp23/pull_dirs.py" "$ROUND_DIR/npu_servers" "$EXPORT_PORT" "npu_servers_r$ROUND" \
    >> "$ROUND_DIR/logs/pull_servers.log" 2>&1
  [[ -e "$SERVERS/.pulled" ]] || sleep 120
done
[[ -s "$SERVERS/servers.json" ]] || die "the pulled server directory has no servers.json"
slots="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["slots"])' "$SERVERS/servers.json")"
log "servers: $("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["server_instance"])' "$SERVERS/servers.json")"
IFS=',' read -r -a gpus <<< "$CLIENT_GPUS"
[[ "${#gpus[@]}" -eq "$slots" ]] || die "EXP23_CLIENT_GPUS has ${#gpus[@]} entries, the 910B serves $slots slots"

# 4 -------------------------------------------------------------------------------
STAMP="r${ROUND}npu_$(date +%Y%m%d_%H%M%S)"
FAILED_DIR="$ROUND_DIR/runtime/$STAMP/failed"  # run_s2_dagger_collect_cuda.sh: one file per failed shard
status "collecting round $ROUND"
chown -R 1015:1015 "$ROUND_DIR"
( cd "$WT" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_COLLECT_COHORT_DIR="$ROUND_DIR/cohorts" \
    PPA_COLLECT_OUTPUT_ROOT="$ROUND_DIR" PPA_EVAL_EXTERNAL_SERVERS=1 PPA_EVAL_EXTERNAL_SERVER_DIR="$SERVERS" \
    PPA_EVAL_GPU_DEVICES="$CLIENT_GPUS" PPA_EVAL_MODEL_PORT_BASE=52410 PPA_EVAL_VO_PORT_BASE=52510 \
    PPA_EVAL_DISPLAY_BASE="$DISPLAY_BASE" PPA_EVAL_SHARD_RETRIES=5 PPA_EVAL_SYSTEM2_LORA="$ADAPTER" \
    PPA_EVAL_RUN_STAMP="$STAMP" \
    bash scripts/run_s2_dagger_collect_cuda.sh > "$ROUND_DIR/logs/npu_launcher.log" 2>&1 ) &
lpid=$!
last=""
while kill -0 "$lpid" 2>/dev/null; do
  records=$(find "$ROUND_DIR/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l)
  done_shards=$(find "$ROUND_DIR/workers" -mindepth 2 -maxdepth 2 -name DONE 2>/dev/null | wc -l)
  failed=$(ls "$FAILED_DIR" 2>/dev/null | sed 's/^shard_//' | paste -sd, -)
  msg="collecting round $ROUND: $records records, $done_shards shards done"
  [[ -n "$failed" ]] && msg="DEGRADED: $msg; failed shard(s) $failed (logs/npu_launcher.log; check the 910B's SERVERS_STATUS and the tunnel)"
  [[ "$msg" == "$last" ]] || { status "$msg"; last="$msg"; }
  sleep 300
done
wait "$lpid"
rc=$?
records=$(find "$ROUND_DIR/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l)
chown -R 1015:1015 "$ROUND_DIR"
[[ "$rc" -eq 0 ]] || die "the collection launcher exited with $rc after $records records (see logs/npu_launcher.log; if the 910B restarted its servers, rerun this script: it pulls the new set)"
status "DONE: $records records"
