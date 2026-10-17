#!/usr/bin/env bash
# EXP-28's unattended decision tree on the 4090 (fjl-habitat), from C1's first offline-gate verdict to the end
# (ledger EXP-28 D/E/F, run records 3-9).  Status in $A/STATUS, log $A/auto.log, the end state in $A/SUMMARY.txt.
#
#   gates attempt 1 (post_collect.sh writes gates/c1_a1/verdict.json), "next":
#     val              -> candidate = C1
#     scale_0.5        -> gates attempt 2 on C1 with lora_B x 0.5; val -> candidate = that scaled copy; else end
#     retrain_2_epochs -> two-pass stream, the 910B retrains (signal RETRAIN_E2), gates attempt 2 on it; val -> it
#     anything else    -> end (kill, invalid, no verdict)
#   candidate -> F's A0 per-call identity check (sdpa, no adapter, 3 episodes, call for call) -> val_unseen seed 42
#     (FA2, step-budget STOP on, 600-episode rule: paired standard-protocol dSR <= -3 pt at >= 600 episodes stops it)
#     and, in parallel, P: the held-out cohort on the 910B serving the candidate; P = kill with its rerun check
#     valid (>= 9 of the first 10 rerun episodes identical call for call) withdraws val;
#   seed 42 complete (1,839 episodes) -> standard-protocol paired dSR <= -0.5 -> end (futility); else seed 1337
#     (no in-run stop rule: the ledger sets it for seed 42 only), started only if it can end before 10-20 14:00 UTC;
#   seed 1337 complete -> paired bootstrap of both seeds against A0 (20,000 draws) -> the H-28 verdict.
# Signals reach the 910B through tunnel port 52517 (c1_910b.sh's receiver); streams through 52414-52416.
set -uo pipefail
WT=/workspace/exp23/wt_exp28; PY=/opt/conda/bin/python
X=/workspace/exp28/c1; R1=/workspace/exp28/round1; G=/workspace/exp28/gates; A=/workspace/exp28/auto
C=/workspace/exp28/collect_base; COH=/workspace/exp28/episodes/cohorts; H=/workspace/exp28/episodes/splits/heldout_scenes.txt
SIG=$A/signals_out
A0_DIR=/workspace/eval_runs/exp20_a0_seed42_4090
declare -A A0=([42]=/workspace/eval_runs/exp20_a0_seed42_4090/merged/progress.jsonl
               [1337]=/workspace/eval_runs/exp20_a0_seed1337_4090/merged/progress.jsonl)
GIVE_UP=$(date -d "2026-10-19T00:00:00Z" +%s)      # nothing started after this can finish before 10-20 14:00
LAST_1337=$(date -d "2026-10-19T20:00:00Z" +%s)    # a seed takes ~17.5 h
export PYTHONPATH="$WT:/workspace/rpc/src"
mkdir -p "$A" "$SIG" "$X/adapter_pull" "$X/logs"
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$A/auto.log" > "$A/STATUS"; }
note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$A/auto.log" >> "$A/SUMMARY.txt"; }
jget() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2" 2>/dev/null; }
signal() {  # push files (paths relative to $SIG) to the 910B's signal receiver; gives up after $SIGNAL_S seconds
  printf '%s\n' "$@" > "$A/sig_files.txt"
  timeout "${SIGNAL_S:-21600}" "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$SIG" --ports 52517 \
    --files "$A/sig_files.txt" --retries 1000000 >> "$A/signals.log" 2>&1
}
finish() {
  st "END: $*"; printf '%s END: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$A/SUMMARY.txt"
  date -u +%FT%TZ > "$SIG/STOP_SERVING"; SIGNAL_S=1800 signal STOP_SERVING  # free the 910B
  chown -R 1015:1015 "$A" "$X" 2>/dev/null; exit 0
}
pull() {  # $1 export name, $2 destination dir; gives up at GIVE_UP
  until [[ -e "$2/$1/.pulled" ]]; do
    (( $(date +%s) < GIVE_UP )) || return 1
    "$PY" "$WT/scripts/exp23/pull_dirs.py" "$2" 52417 "$1" >> "$A/pull.log" 2>&1
    [[ -e "$2/$1/.pulled" ]] || sleep 300
  done
}
gates() {  # $1 adapter dir, $2 attempt, $3 lora scale; prints the verdict's "next"
  local out="$G/c1_a$2"; [[ "$3" == 1 ]] || out="${out}_scale$3"
  EXP28_WT="$WT" EXP28_ADAPTER="$1" EXP28_NAME=c1 EXP28_ATTEMPT="$2" EXP28_LORA_SCALE="$3" EXP28_HOLDOUT_SCENES_FILE="$H" \
    EXP28_O4_RECORDS="$C/records" EXP28_O1_BASE=/workspace/exp23/b3/scalevln_b3.json \
    EXP28_O2_BASE=/workspace/exp23/exp26/base_r2r_holdout.json \
    bash "$WT/scripts/exp28/run_offline_gates.sh" > "$A/gates_a$2_scale$3.log" 2>&1
  jget "$out/verdict.json" next || echo "no_verdict"
}
val_launcher_pids() {  # $1 output root: the val launchers writing there
  local p
  for p in $(pgrep -f "^bash scripts/run_ppa_r2r_val_unseen_cuda[.]sh$"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^PPA_EVAL_OUTPUT_ROOT=$1$" && echo "$p"
  done
}
term_val() {  # $1 output root: stop that val run (launcher process group, then stray clients)
  local p
  for p in $(val_launcher_pids "$1"); do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  sleep 60
  for p in $(pgrep -f "[r]2r_val_unseen\.py.*$1"); do kill -TERM "$p" 2>/dev/null; done
}
merged_rows() { wc -l < "$1/merged/progress.jsonl" 2>/dev/null || echo 0; }
run_val() {  # $1 seed, $2 abort rule (1/0). Returns 0 complete, 1 incomplete, 2 aborted by the rule, 3 withdrawn by P
  local seed=$1 abort=$2 out="/workspace/eval_runs/exp28_${ROUND}_seed$1_4090_stepbudget" try rc launcher
  for try in 1 2 3; do  # the launcher resumes (--resume): a retry costs at most the episodes in flight
    st "val seed $seed ($ROUND), try $try -> $out"
    ( cd "$WT" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES=5,4,0 PPA_EVAL_PROTOCOL_SEED="$seed" \
        PPA_EVAL_SYSTEM2_LORA="$CAND" PPA_EVAL_OUTPUT_ROOT="$out" PPA_EVAL_DISPLAY_BASE=400 PPA_EVAL_SHARD_RETRIES=3 \
        PPA_EVAL_STOP_AT_STEP_BUDGET=1 exec setsid bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) \
      > "$X/logs/val_seed${seed}_try$try.log" 2>&1 &
    launcher=$!
    while kill -0 "$launcher" 2>/dev/null; do
      sleep 1200
      [[ -e "$A/P_KILL" ]] && { term_val "$out"; wait "$launcher"; return 3; }
      "$PY" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$out/workers/shard_*/progress.json" \
        --reference "${A0[$seed]}" --out "$X/paired_seed$seed.json" >> "$A/auto.log" 2>&1
      rc=$?
      if [[ "$abort" -eq 1 && "$rc" -eq 3 ]]; then
        note "seed $seed: the 600-episode rule fired ($(tr -d '\n' < "$X/paired_seed$seed.json" | cut -c1-300))"
        term_val "$out"; wait "$launcher"; return 2
      fi
    done
    wait "$launcher"; rc=$?
    [[ -e "$A/P_KILL" ]] && return 3
    note "seed $seed try $try: launcher exit $rc, merged rows $(merged_rows "$out")"
    [[ "$(merged_rows "$out")" -eq 1839 ]] && break
  done
  "$PY" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$out/workers/shard_*/progress.json" \
    --reference "${A0[$seed]}" --out "$X/paired_seed$seed.json" >> "$A/auto.log" 2>&1
  [[ "$(merged_rows "$out")" -eq 1839 ]] || return 1
  return 0
}

# 0 ------------------------------------------------------------------------------
st "waiting for C1's first gate verdict (post_collect.sh)"
while :; do
  ps_line=$(cat "$R1/POST_STATUS" 2>/dev/null)
  [[ "$ps_line" =~ ^[^\ ]+\ DONE:\ gates ]] && break
  [[ "$ps_line" =~ ^[^\ ]+\ STOPPED: ]] && finish "post_collect stopped before the gates: $ps_line"
  (( $(date +%s) < GIVE_UP )) || finish "no gate verdict by 10-19 00:00 UTC (post_collect: $ps_line)"
  sleep 300
done
[[ -s "$G/c1_a1/verdict.json" ]] || finish "the first gate pass wrote no verdict ($(cat "$R1/POST_STATUS"))"
next1=$(jget "$G/c1_a1/verdict.json" next)
note "gates attempt 1: next=$next1"
A1="$X/adapter_pull/adapter_rX28C1"

# 1 ------------------------------------------------------------------------------
case "$next1" in
  val)
    CAND="$A1"; ROUND=X28C1; TAG=c1 ;;
  scale_0.5)
    st "gates attempt 2: C1 with lora_B x 0.5"
    next2=$(gates "$A1" 2 0.5); note "gates attempt 2 (scale 0.5): next=$next2"
    [[ "$next2" == val ]] || finish "C1 stopped at gates attempt 2 (scale 0.5): $next2"
    ROUND=X28C1S05; CAND="$X/adapter_pull/adapter_r$ROUND"; TAG=c1s05
    rm -rf "$CAND"; cp -r "$G/c1_a2_scale0.5/adapter_scale0.5" "$CAND"
    # P serves the very bytes val serves: send them to the 910B, then the signal
    rm -rf "$SIG/adapter_s05"; cp -r "$CAND" "$SIG/adapter_s05"
    st "sending the scaled adapter to the 910B"
    signal $(cd "$SIG" && find adapter_s05 -type f | sort) || note "could not send the scaled adapter: P will not run"
    date -u +%FT%TZ > "$SIG/SCALE05"; signal SCALE05 || note "could not send SCALE05: P will not run" ;;
  retrain_2_epochs)
    st "retrain_2_epochs: two-pass stream, signal RETRAIN_E2, push"
    "$PY" /workspace/exp28/double_stream.py --src "$R1/stream" --dst "$R1/stream_e2" > "$A/double_stream.log" 2>&1 \
      || finish "could not build the two-pass stream ($A/double_stream.log)"
    date -u +%FT%TZ > "$SIG/RETRAIN_E2"; signal RETRAIN_E2 || finish "could not signal RETRAIN_E2 to the 910B"
    timeout 43200 "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$R1/stream_e2" --ports 52414,52415,52416 \
      --retries 1000000 --busy-sleep 15 > "$A/push_stream_e2.log" 2>&1 || finish "pushing the two-pass stream failed"
    st "waiting for adapter_rX28C1E2"
    pull adapter_rX28C1E2 "$X/adapter_pull" || finish "no adapter_rX28C1E2 by 10-19 00:00 UTC"
    next2=$(gates "$X/adapter_pull/adapter_rX28C1E2" 2 1); note "gates attempt 2 (2 epochs): next=$next2"
    [[ "$next2" == val ]] || finish "C1 stopped at gates attempt 2 (2 epochs): $next2"
    ROUND=X28C1E2; CAND="$X/adapter_pull/adapter_r$ROUND"; TAG=c1e2 ;;
  *)
    finish "C1 stopped at gates attempt 1: $next1" ;;
esac
note "candidate $ROUND ($(sha256sum "$CAND/adapter_model.safetensors" | cut -c1-16)) goes to val; P on the 910B tag $TAG"

# 2 ------------------------------------------------------------------------------
# F: the code without an adapter, sdpa, reproduces A0 seed 42 call for call on 3 episodes of shard 0 (one retry).
ok=0
for try in 1 2; do
  CAN="$X/canary_noadapter_t$try"; rm -rf "$CAN"
  st "A0 per-call identity check (sdpa, no adapter, 3 episodes), try $try"
  ( cd "$WT" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES=5 PPA_EVAL_SHARDS=0 \
      PPA_EVAL_MAX_EPISODES_PER_SHARD=3 PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_OUTPUT_ROOT="$CAN" PPA_EVAL_DISPLAY_BASE=380 \
      PPA_EVAL_ATTN_IMPL=sdpa bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) > "$X/logs/canary_t$try.log" 2>&1
  if "$PY" "$WT/scripts/tools/compare_client_logs.py" --a "$A0_DIR/runtime/*/logs/client_shard_*.log" \
       --b "$CAN/runtime/*/logs/client_shard_*.log" --out "$A/canary_identity_t$try.json" --require-identical \
       >> "$A/auto.log" 2>&1; then ok=1; break; fi
done
(( ok )) || finish "F's A0 per-call identity check failed ($A/canary_identity_t*.json): pairing with A0 invalid, no val"
note "A0 per-call identity check passed (try $try)"

# P in the background: collect the held-out cohort against the 910B's $TAG servers, judge, withdraw val on a kill
rm -f "$A/P_KILL" "$A/P_DONE"
PO="/workspace/exp28/collect_$TAG"
(
  if [[ "$TAG" == c1s05 && ! -e "$SIG/SCALE05" ]]; then note "P skipped (the scaled adapter never reached the 910B)"
  else
    timeout 43200 env EXP28_MODE=external EXP28_TAG="$TAG" EXP28_ADAPTER="$CAND" EXP28_WT="$WT" EXP28_COHORT_DIR="$COH" \
      EXP28_OUT_DIR="$PO" EXP28_SHARDS=0-19 EXP28_DISPLAY_BASE=761 bash "$WT/scripts/exp28/collect_4090.sh" > /dev/null 2>&1
    # the rerun check: exactly the first 10 committed rerun episodes
    RR="$A/p_repro_first10"; rm -rf "$RR"; mkdir -p "$RR"
    for sha in $(ls -tr /workspace/exp28/p_repro/records/*/*.s2rec.tar.sha256 2>/dev/null | head -10); do
      mkdir -p "$RR/$(basename "$(dirname "$sha")")"
      ln -s "${sha%.sha256}" "$RR/$(basename "$(dirname "$sha")")/"; ln -s "$sha" "$RR/$(basename "$(dirname "$sha")")/"
    done
    valid=$("$PY" "$WT/scripts/tools/canary_compare.py" --rerun-records "$RR" --reference-records "$C/records" \
              --min-identical-episodes 9 --output "$A/p_identity.json" 2>&1 | grep -m1 -E "^(PASS|FAIL) per-call")
    p=$("$PY" "$WT/scripts/tools/canary_compare.py" --canary-records "$PO/records" --base-records "$C/records" \
          --cohort "$COH" --cohort-part heldout --criteria exp28-p --output "$A/p_$TAG.json" 2>&1 \
          | grep -m1 -E "^(PASS|KILL|INCOMPLETE|INVALID) P ")
    note "P: ${p:-no verdict} | rerun check (first 10): ${valid:-none}"
    if [[ "$valid" == PASS* && "$p" == KILL* ]]; then note "P kill: withdrawing val"; echo kill > "$A/P_KILL"; fi
  fi
  date -u +%FT%TZ > "$SIG/STOP_SERVING"; SIGNAL_S=7200 signal STOP_SERVING
  date -u +%FT%TZ > "$A/P_DONE"
) &

run_val 42 1; r=$?
case $r in
  2) finish "seed 42 stopped by the 600-episode rule (standard-protocol dSR <= -3 pt)" ;;
  3) finish "val withdrawn by P" ;;
  1) finish "seed 42 incomplete after 3 tries ($(merged_rows "/workspace/eval_runs/exp28_${ROUND}_seed42_4090_stepbudget") of 1839)" ;;
esac
OUT42="/workspace/eval_runs/exp28_${ROUND}_seed42_4090_stepbudget"

# 3 ------------------------------------------------------------------------------
d42=$(jget "$X/paired_seed42.json" sr.delta_pt); n42=$(jget "$X/paired_seed42.json" sr.n)
"$PY" "$WT/scripts/tools/paired_closed_loop_bootstrap.py" --treatment "$OUT42/merged/progress.jsonl" --control "${A0[42]}" \
  --standard-protocol --bootstrap-draws 20000 --label "${ROUND}_seed42_vs_a0" --output-json "$A/seed42_bootstrap.json" \
  > "$A/seed42_bootstrap.txt" 2>&1
dos=$("$PY" -c 'import json,sys; print(round(100*json.load(open(sys.argv[1]))["strata"]["all"]["os"]["diff"],2))' "$A/seed42_bootstrap.json" 2>/dev/null)
note "seed 42 paired (standard protocol): dSR=$d42 pt over $n42 episodes; dOS=${dos:-?} pt"
[[ -n "$d42" ]] || finish "seed 42 has no paired dSR"
if "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= -0.5 else 1)' "$d42"; then
  finish "futility: seed 42 dSR $d42 <= -0.5, no seed 1337 (C2 is not run unattended; ledger G gives its conditions)"
fi

# 4 ------------------------------------------------------------------------------
st "waiting for P before seed 1337"
for _ in $(seq 72); do [[ -e "$A/P_DONE" ]] && break; sleep 600; done  # <= 12 h
[[ -e "$A/P_KILL" ]] && finish "val withdrawn by P (after seed 42: dSR $d42)"
(( $(date +%s) <= LAST_1337 )) || finish "no time for seed 1337 (seed 42 dSR $d42); it could not end before 10-20 14:00 UTC"
run_val 1337 0; r=$?
OUT1337="/workspace/eval_runs/exp28_${ROUND}_seed1337_4090_stepbudget"
[[ $r -eq 0 ]] || finish "seed 1337 incomplete after 3 tries ($(merged_rows "$OUT1337") of 1839; seed 42 dSR $d42)"
note "seed 1337 paired: $(tr -d '\n' < "$X/paired_seed1337.json" 2>/dev/null | cut -c1-300)"
"$PY" "$WT/scripts/tools/paired_closed_loop_bootstrap.py" --treatment "$OUT42/merged/progress.jsonl" \
  --treatment "$OUT1337/merged/progress.jsonl" --control "${A0[42]}" --control "${A0[1337]}" --standard-protocol \
  --bootstrap-draws 20000 --label "${ROUND}_vs_a0_two_seeds" --output-json "$A/final_bootstrap.json" > "$A/final_bootstrap.txt" 2>&1
v=$("$PY" - "$A/final_bootstrap.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["strata"]["all"]["success"]
d, lo, hi = 100 * s["diff"], 100 * s["ci95"][0], 100 * s["ci95"][1]
h = "supported" if d >= 1.0 and lo > 0 else ("refuted" if d <= 0 else "not detected")
print(f"H-28 {h}: two-seed dSR {d:+.2f} pt, 95% CI [{lo:+.2f}, {hi:+.2f}]")
PY
)
finish "${v:-the final bootstrap failed ($A/final_bootstrap.txt)}"
