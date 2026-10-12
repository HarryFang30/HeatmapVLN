#!/usr/bin/env bash
# EXP-23, 4090 half of one round after collection: label, pack and push the training
# stream and the held-out records, check the new code against A0, fetch the adapter
# once the 910B publishes it, and run the paired closed-loop evaluation (seed 42, then
# seed 1337) with the in-run stop rule.  Inside the fjl-habitat container.
#
# Runs unattended; every phase waits on evidence and logs to $ROUND_DIR/logs/post.log.
# The 910B half is scripts/exp23/round_train_watch_910b.sh; the adapter appears there only
# if the offline gates pass, so a failed gate leaves this script waiting at phase 5.
#
#   EXP23_ROUND=1 nohup setsid bash scripts/exp23/round_post_4090.sh > /dev/null 2>&1 &
#
# Phases:
#   1. wait for $ROUND_DIR/STOPPED_AT_N (scripts' stop_at_n) and for the replay pull;
#   2. relabel every record (idempotent; the rules digest decides) and pack the stream;
#   3. push the held-out records + labels (then HOLDOUT_DONE) and, in the background,
#      the stream, through the tunnels (the pushers wait for the 910B receivers);
#   4. identity check: the new code serving no adapter on the first episodes of shard 0,
#      call for call against A0 seed 42 (EXP-20) -- the pairing below assumes it;
#   5. fetch adapter_r$ROUND from the 910B's export port;
#   6. full val_unseen with the adapter, seed 42 then seed 1337, each paired against the
#      A0 run of its seed; paired_progress_monitor.py stops a run whose SR is >= 3 pt
#      below A0 after 600 paired episodes.

set -uo pipefail

ROUND="${EXP23_ROUND:-1}"
WT="${EXP23_WT:?set EXP23_WT to the pinned code worktree, e.g. /workspace/exp23/wt_<commit>}"
ROUND_DIR="${EXP23_ROUND_DIR:-/workspace/exp23/round$ROUND}"
PYTHON="${EXP23_PYTHON:-/opt/conda/bin/python}"
REPLAY_ROOT="${EXP23_REPLAY_ROOT:-/workspace/exp23/data/interndata_n1/r2r}"
EPISODES="${EXP23_EPISODES:-/workspace/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
TOTAL_SAMPLES="${EXP23_TOTAL_SAMPLES:-72192}"
STREAM_PORTS="${EXP23_STREAM_PORTS:-52410,52411,52412,52413}"
HOLDOUT_PORTS="${EXP23_HOLDOUT_PORTS:-52414,52415,52416}"
EXPORT_PORT="${EXP23_EXPORT_PORT:-52417}"
EVAL_GPUS="${EXP23_EVAL_GPUS:-5,4,0}"
CANARY_GPU="${EXP23_CANARY_GPU:-5}"
SEEDS="${EXP23_SEEDS:-42 1337}"
EVAL_ROOT="${EXP23_EVAL_ROOT:-/workspace/eval_runs}"
# 1: the closed-loop runs use EXP-24's --stop_at_step_budget (user, 2026-10-13).  The
# identity check below never does: it is about the code, against A0's protocol.  The
# paired monitor scores a switched-on run under the standard protocol too (exactly
# derivable), and the stop rule uses that one.
STOP_AT_STEP_BUDGET="${EXP23_STOP_AT_STEP_BUDGET:-0}"
[[ "$STOP_AT_STEP_BUDGET" =~ ^[01]$ ]] || { echo "EXP23_STOP_AT_STEP_BUDGET must be 0 or 1" >&2; exit 2; }
RUN_SUFFIX=""
[[ "$STOP_AT_STEP_BUDGET" -eq 1 ]] && RUN_SUFFIX="_stepbudget"
declare -A A0_RUN=([42]=/workspace/eval_runs/exp20_a0_seed42_4090 [1337]=/workspace/eval_runs/exp20_a0_seed1337_4090)
export PYTHONPATH="$WT:/workspace/rpc/src"

mkdir -p "$ROUND_DIR/logs"
LOG="$ROUND_DIR/logs/post.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$ROUND_DIR/POST_STATUS"; log "STATUS $*"; }
die() { status "STOPPED: $*"; chown -R 1015:1015 "$ROUND_DIR" 2>/dev/null; exit 2; }
git -c safe.directory='*' -C "$WT" diff --quiet HEAD -- || die "$WT has uncommitted changes"
log "round=$ROUND wt=$WT commit=$(git -c safe.directory='*' -C "$WT" rev-parse HEAD) stop_at_step_budget=$STOP_AT_STEP_BUDGET"

# 1 -------------------------------------------------------------------------------
status "waiting for collection to stop"
until [[ -e "$ROUND_DIR/STOPPED_AT_N" ]]; do sleep 120; done
while pgrep -f '[p]ull_dirs\.py' >/dev/null; do sleep 60; done
log "collection stopped; replay scenes: $(ls "$REPLAY_ROOT" | grep -vc '^\.')"

# 2 -------------------------------------------------------------------------------
status "labelling"
"$PYTHON" "$WT/scripts/tools/label_s2_records.py" --record_root "$ROUND_DIR/records" --label_root "$ROUND_DIR/labels" \
  --episodes "$EPISODES" --workers 32 --summary "$ROUND_DIR/labels/summary.json" >> "$LOG" 2>&1 \
  || die "labelling failed"
status "packing the stream"
rm -rf "$ROUND_DIR/stream"
"$PYTHON" "$WT/scripts/tools/build_s2_train_stream.py" --record-root "$ROUND_DIR/records" --label-root "$ROUND_DIR/labels" \
  --replay-root "$REPLAY_ROOT" --out-dir "$ROUND_DIR/stream" --shard-size 256 --global-batch 64 --world-size 8 \
  --total-samples "$TOTAL_SAMPLES" --workers 24 >> "$LOG" 2>&1 || die "stream packing failed"
chown -R 1015:1015 "$ROUND_DIR"

# 3 -------------------------------------------------------------------------------
status "pushing"
( cd "$WT" && "$PYTHON" scripts/tools/stream_transfer.py push --src "$ROUND_DIR/stream" --ports "$STREAM_PORTS" \
    --retries 1000000 --busy-sleep 15 > "$ROUND_DIR/logs/push_stream.log" 2>&1 ) &
"$PYTHON" - "$ROUND_DIR" > "$ROUND_DIR/holdout.list" <<'PY' || die "could not list the held-out records"
import sys
from pathlib import Path
from src.data.s2_records import iter_record_paths, label_path_for
from src.data.s2_sft_dataset import holdout_scenes
root = Path(sys.argv[1])
held = set(holdout_scenes())
for rec in iter_record_paths(root / "records"):
    if rec.parent.name not in held:
        continue
    lab = label_path_for(rec, root / "labels")
    if not lab.is_file():
        continue
    for p in (rec, rec.with_name(rec.name + ".sha256"), lab):
        print(p.relative_to(root))
PY
log "held-out files: $(wc -l < "$ROUND_DIR/holdout.list")"
( cd "$WT" && "$PYTHON" scripts/tools/stream_transfer.py push --src "$ROUND_DIR" --ports "$HOLDOUT_PORTS" \
    --files "$ROUND_DIR/holdout.list" --retries 1000000 > "$ROUND_DIR/logs/push_holdout.log" 2>&1 ) \
  || die "pushing the held-out records failed"
date -u +%FT%TZ > "$ROUND_DIR/HOLDOUT_DONE"
echo HOLDOUT_DONE > "$ROUND_DIR/holdout_done.list"
( cd "$WT" && "$PYTHON" scripts/tools/stream_transfer.py push --src "$ROUND_DIR" --ports "${HOLDOUT_PORTS%%,*}" \
    --files "$ROUND_DIR/holdout_done.list" --retries 1000000 >> "$ROUND_DIR/logs/push_holdout.log" 2>&1 ) \
  || die "pushing HOLDOUT_DONE failed"
log "held-out records pushed"

# 4 -------------------------------------------------------------------------------
status "identity check against A0 seed 42"
CANARY="$ROUND_DIR/canary_noadapter"
rm -rf "$CANARY"
( cd "$WT" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES="$CANARY_GPU" PPA_EVAL_SHARDS=0 \
    PPA_EVAL_MAX_EPISODES_PER_SHARD=3 PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_OUTPUT_ROOT="$CANARY" PPA_EVAL_DISPLAY_BASE=380 \
    bash scripts/run_ppa_r2r_val_unseen_cuda.sh > "$ROUND_DIR/logs/canary.log" 2>&1 )
"$PYTHON" "$WT/scripts/tools/compare_client_logs.py" --a "${A0_RUN[42]}/runtime/*/logs/client_shard_*.log" \
  --b "$CANARY/runtime/*/logs/client_shard_*.log" --out "$ROUND_DIR/canary_identity.json" --require-identical >> "$LOG" 2>&1 \
  || die "the new code without an adapter does not reproduce A0 call for call (see canary_identity.json); pairing invalid"
log "identity check passed"

# 5 -------------------------------------------------------------------------------
status "waiting for adapter_r$ROUND from the 910B"
ADAPTER="$ROUND_DIR/adapter_pull/adapter_r$ROUND"
until [[ -e "$ADAPTER/.pulled" ]]; do
  "$PYTHON" "$WT/scripts/exp23/pull_dirs.py" "$ROUND_DIR/adapter_pull" "$EXPORT_PORT" "adapter_r$ROUND" >> "$ROUND_DIR/logs/pull_adapter.log" 2>&1
  [[ -e "$ADAPTER/.pulled" ]] || sleep 300
done
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || die "the pulled adapter has no weights"
log "adapter pulled: $(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-16)"
chown -R 1015:1015 "$ROUND_DIR"

# 6 -------------------------------------------------------------------------------
for seed in $SEEDS; do
  OUT="$EVAL_ROOT/exp23_r${ROUND}_seed${seed}_4090${RUN_SUFFIX}"
  status "closed loop seed $seed -> $OUT"
  # setsid run directly (no subshell): $! is then the leader of the run's own process
  # group, so the stop rule can TERM the launcher, its servers and its clients at once.
  cd "$WT"
  PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES="$EVAL_GPUS" PPA_EVAL_PROTOCOL_SEED="$seed" \
    PPA_EVAL_SYSTEM2_LORA="$ADAPTER" PPA_EVAL_OUTPUT_ROOT="$OUT" PPA_EVAL_DISPLAY_BASE=400 PPA_EVAL_SHARD_RETRIES=3 \
    PPA_EVAL_STOP_AT_STEP_BUDGET="$STOP_AT_STEP_BUDGET" \
    setsid bash scripts/run_ppa_r2r_val_unseen_cuda.sh > "$ROUND_DIR/logs/eval_seed${seed}.log" 2>&1 &
  launcher=$!
  while kill -0 "$launcher" 2>/dev/null; do
    sleep 1200
    "$PYTHON" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$OUT/workers/shard_*/progress.json" \
      --reference "${A0_RUN[$seed]}/merged/progress.jsonl" --out "$ROUND_DIR/paired_seed${seed}.json" >> "$LOG" 2>&1
    if [[ $? -eq 3 ]]; then
      log "stop rule fired for seed $seed: SR >= 3 pt below A0 after 600 paired episodes"
      kill -TERM -- "-$launcher" 2>/dev/null || kill -TERM "$launcher" 2>/dev/null
      sleep 60
      for pid in $(pgrep -f "[r]2r_val_unseen\.py.*$OUT"); do kill -TERM "$pid"; done
      die "aborted seed $seed by the in-run stop rule"
    fi
  done
  "$PYTHON" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$OUT/workers/shard_*/progress.json" \
    --reference "${A0_RUN[$seed]}/merged/progress.jsonl" --out "$ROUND_DIR/paired_seed${seed}.json" >> "$LOG" 2>&1
  log "seed $seed finished: $(cat "$ROUND_DIR/paired_seed${seed}.json" | tr -d '\n' | cut -c1-300)"
  chown -R 1015:1015 "$ROUND_DIR" "$OUT"
done
status "DONE: both seeds evaluated"
