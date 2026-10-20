#!/usr/bin/env bash
# EXP-30 probe 1 on the 4090: pull probe1_decode from the 910B's export port once it carries DONE, then judge with
# line B (probe_judge.py) and write /workspace/exp30/probe1/probe1.json.  Gives up at 10-21 00:00 UTC.
WT=/workspace/exp23/wt_exp28; PY=/opt/conda/bin/python; X=/workspace/exp30/probe1
END=$(date -d "2026-10-21T00:00:00Z" +%s)
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/judge.log" > "$X/STATUS"; }
cd "$WT" || exit 2
export PYTHONPATH="$WT:/workspace/rpc/src"
st "waiting for probe1_decode/DONE on 52417"
until [[ -e "$X/pull/probe1_decode/DONE" ]]; do
  (( $(date +%s) < END )) || { st "STOPPED: no finished decode by 10-21 00:00 UTC"; exit 2; }
  rm -rf "$X/pull/probe1_decode"
  "$PY" scripts/exp23/pull_dirs.py "$X/pull" 52417 probe1_decode >> "$X/pull.log" 2>&1
  [[ -e "$X/pull/probe1_decode/DONE" ]] || sleep 600
done
st "judging $(cat "$X"/pull/probe1_decode/decode.shard*.jsonl | wc -l) decoded states with line B"
"$PY" scripts/exp30/probe_judge.py --states "$X/states.jsonl" --decoded "$X/pull/probe1_decode" --out "$X" \
  > "$X/judge_out.txt" 2>&1
st "DONE: $(head -c 600 "$X/judge_out.txt" | tr '\n' ' ')"
chown -R 1015:1015 "$X"
