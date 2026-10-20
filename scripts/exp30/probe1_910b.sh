#!/usr/bin/env bash
# EXP-30 probe 1 on the 910B (writes only under ~/work/zhr/zhr_1): once the records and states.jsonl have arrived in
# $D/data (stream_transfer on 52413), decode with the base model on all 8 cards, one process per card, into the
# export directory (served on 52417 as probe1_decode); DONE marks the end.  Status in $D/STATUS.
R=~/work/zhr/zhr_1; D=$R/exp30; SRC=$R/exp23/src_x28; P=$R/envs/ppa/bin/python
OUT=$R/exp23/round1/export/probe1_decode
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/probe1.log"; echo "$(date -u +%FT%TZ) $*" > "$D/STATUS"; }
log "waiting for states.jsonl"
until [[ -s "$D/data/states.jsonl" ]]; do sleep 30; done
n=$(wc -l < "$D/data/states.jsonl"); have=$(ls "$D"/data/records/*/*.s2rec.tar 2>/dev/null | wc -l)
log "states $n, records $have"
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PYTHONPATH="$SRC"
mkdir -p "$OUT" "$R/tmp/probe1"
export TMPDIR="$R/tmp/probe1"
cd "$SRC" || exit 2
pids=()
for i in 0 1 2 3 4 5 6 7; do
  ASCEND_RT_VISIBLE_DEVICES=$i $P scripts/exp30/probe_decode.py --states "$D/data/states.jsonl" --record-parent "$D/data" \
    --model-path "$R/InternNav_Model" --out "$OUT" --shard "$i/8" --device npu > "$D/decode_$i.log" 2>&1 &
  pids+=($!)
done
log "decoding on 8 cards"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
rows=$(cat "$OUT"/decode.shard*.jsonl 2>/dev/null | wc -l)
if [[ "$rc" -eq 0 && "$rows" -eq "$n" ]]; then date -u +%FT%TZ > "$OUT/DONE"; log "DONE: $rows rows"
else log "STOPPED: exit $rc, $rows of $n rows (decode_*.log)"; fi
