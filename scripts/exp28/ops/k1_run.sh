#!/usr/bin/env bash
# EXP-28 K1 on the 4090 (CPU, Xvfb llvmpipe): once >= 530 training-cohort records are committed, select the off-route
# states, roll line B out from them in 8 scene shards, and write k1_summary.json.  STATUS in $OUT/STATUS.
WT=/workspace/exp23/wt_exp28
REC=/workspace/exp28/collect_base/records
OUT=/workspace/exp28/k1
PY=/opt/conda/bin/python
mkdir -p "$OUT/logs"
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/k1.log" > "$OUT/STATUS"; }
train_records() { $PY /workspace/exp28/s0_check.py "$REC" | $PY -c 'import json,sys; print(json.load(sys.stdin)["by_part"]["train"])'; }
st "waiting for >= 530 training records"
until [[ "$(train_records)" -ge 530 ]]; do sleep 600; done
cd "$WT" || exit 2
export PYTHONPATH="$WT"
st "select"
$PY scripts/exp28/label_policy_rollout.py select --cohort-dir /workspace/exp28/episodes/cohorts --record-root "$REC" \
  --episodes /workspace/exp28/episodes/pool.json.gz --out-dir "$OUT" --force > "$OUT/logs/select.log" 2>&1 \
  || { st "STOPPED: select failed (logs/select.log)"; exit 2; }
st "rollouts (8 shards)"
pids=()
for i in 0 1 2 3 4 5 6 7; do
  $PY scripts/exp28/label_policy_rollout.py run --out-dir "$OUT" --shard "$i/8" --xvfb-display $((811 + i)) \
    > "$OUT/logs/run_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[[ "$rc" -eq 0 ]] || { st "STOPPED: a rollout shard failed (logs/run_*.log)"; exit 2; }
st "report"
$PY scripts/exp28/label_policy_rollout.py report --out-dir "$OUT" > "$OUT/logs/report.log" 2>&1
rc=$?
chown -R 1015:1015 "$OUT"
st "DONE (report exit $rc)"
