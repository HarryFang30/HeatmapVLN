#!/usr/bin/env bash
# EXP-37 on the 4090 (ledger 运行记录 2): the whole offline gate on this box, on X37 = the step-400 checkpoint.
# G1 on the held-out pairs, then G2 (with the report-only O4 correction, O5, L) decoded as deployed, line B
# scored, gate_verdict.py attempt 1.  The tools and their thresholds are 910B's run_gate_910b.sh / gate_4090.sh;
# only the device changes.  8 shards, as the tools' rows and the score expect, worked through by the lanes in
# X37_GATE_LANES (a GPU id per lane; two on one GPU share it, a process holds about 20 GB).  Rows resume.
# G1 failing stops here: the one retry needs the 910B's cards and the training stream, neither of which is here.
set -uo pipefail
: "${X37_SHA:?the sha256 of the adapter}"
X=/workspace/exp37/run4090; SRC=/workspace/exp37/src_x37; PY=/opt/conda/bin/python
G=$X/gate; ADAPTER=/workspace/exp37/x37_s400/adapter; PAIRS=/workspace/exp37/heldout_pairs_g1
HM3D=/workspace/exp28/gates/hm3d_heldout; SCENES=/workspace/exp28/episodes/splits/heldout_scenes.txt
LANES=(${X37_GATE_LANES:-0 0})
mkdir -p "$G/g1" "$G/o4" "$X/logs"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/logs/gate.log"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$G/STATUS"; log "STATUS $*"; }
fail() { status "STOPPED: $*"; exit 2; }
old=$(cat "$G/.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "gate_4090.sh"; then
  log "refused: gate_4090.sh runs as $old"; exit 1
fi
echo $$ > "$G/.pid"

on_lanes() {  # $1 label, then the tool's arguments with @K@ for the shard
  local label=$1 lane; shift
  run_lane() {
    local lane=$1 gpu=$2 k try ok; shift 2
    for k in 0 1 2 3 4 5 6 7; do
      (( k % ${#LANES[@]} == lane )) || continue
      ok=0
      for try in 1 2 3; do
        ( cd "$SRC" && env CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$SRC:/workspace/rpc/src" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "${@//@K@/$k}" >> "$G/${label}_$k.log" 2>&1 < /dev/null ) && { ok=1; break; }
        sleep 60
      done
      (( ok )) || return 1
    done
  }
  local pids=() p failed=0
  for lane in "${!LANES[@]}"; do run_lane "$lane" "${LANES[$lane]}" "$@" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  (( failed == 0 )) || fail "$failed lanes failed $label 3 times (see $G/${label}_*.log)"
}
summary_state() {  # pass / fail / invalid / none, read from the summary JSON
  python3 - "$1" "$X37_SHA" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("none"); sys.exit()
print("other" if d.get("adapter_sha256") not in (None, sys.argv[2]) else "invalid" if not d.get("valid") else "pass" if (d.get("g1") or {}).get("pass") else "fail")
PY
}

[[ "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" == "$X37_SHA" ]] || fail "the adapter is not $X37_SHA"
[[ "$(sha256sum "$PAIRS/stream_manifest.json" | cut -c1-12)" == cf93aaffdc99 ]] || fail "the held-out pairs are not run record 1's"

g1=$(summary_state "$G/g1/heldout_summary.json")
if [[ $g1 != pass && $g1 != fail ]]; then
  status "G1: scoring the held-out pairs, lanes ${LANES[*]}"
  on_lanes g1_heldout scripts/exp37/eval_pairs.py score --pairs "$PAIRS" --adapter "$ADAPTER" \
    --model-path /workspace/InternNav_Model --shard @K@/8 --device cuda --output "$G/g1/heldout.rows.shard0@K@of08.jsonl"
  ( cd "$SRC" && PYTHONPATH="$SRC:/workspace/rpc/src" "$PY" scripts/exp37/eval_pairs.py summarize --rows "$G/g1/heldout.rows.shard*of08.jsonl" \
      --adapter "$ADAPTER" --pairs "$PAIRS" --output "$G/g1/heldout_summary.json" >> "$G/g1_summarize.log" 2>&1 )
  g1=$(summary_state "$G/g1/heldout_summary.json")
fi
[[ $g1 == pass || $g1 == fail ]] || fail "G1's summary is $g1 (see $G/g1_summarize.log)"
[[ $g1 == pass ]] || fail "G1 failed ($G/g1/heldout_summary.json); no val, and the one retry cannot run here"
log "G1 passes"

status "G2: decoding the HM3D held-out states (--on-mode base-stop), lanes ${LANES[*]}"
on_lanes o4 scripts/exp28/hm3d_route_gate.py decode --record-root "$HM3D/records" --label-root "$HM3D/labels" \
  --holdout-scenes-file "$SCENES" --adapter "$ADAPTER" --model-path /workspace/InternNav_Model \
  --output "$G/o4/hm3d.json" --shard @K@/8 --shuffle-instructions --on-mode base-stop --device cuda

status "G2: scoring line B"
: > "$G/verdict.json"
rm -f "$G/o4/hm3d.json" "$G/o4/hm3d.calls.jsonl"
( cd "$SRC" && PYTHONPATH="$SRC:/workspace/rpc/src" "$PY" scripts/exp28/hm3d_route_gate.py score \
    --record-root "$HM3D/records" --label-root "$HM3D/labels" --holdout-scenes-file "$SCENES" \
    --scenes-dir /workspace/exp28/scene_datasets/hm3d --output "$G/o4/hm3d.json" ) > "$G/score.log" 2>&1
log "score exit $?"
[[ -s $G/o4/hm3d.json ]] || fail "no hm3d.json (see $G/score.log)"
( cd "$SRC" && PYTHONPATH="$SRC" "$PY" scripts/exp37/gate_verdict.py --hrg "$G/o4/hm3d.json" --pairs "$G/g1/heldout_summary.json" \
    --adapter "$ADAPTER" --attempt 1 --output "$G/verdict.json" ) > "$G/verdict.log" 2>&1
log "gate_verdict exit $?"
cat "$G/verdict.log" >> "$X/logs/gate.log"
[[ -s $G/verdict.json ]] || fail "no verdict.json (see $G/verdict.log)"
status "gate done: $(python3 -c "import json;d=json.load(open('$G/verdict.json'));print({k:d.get(k) for k in ('all_pass','measured','next')})")"
