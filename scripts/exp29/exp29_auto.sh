#!/usr/bin/env bash
# EXP-29's unattended tree on the 4090 (fjl-habitat), ledger EXP-29 (criteria fixed 2026-10-18 02:45 UTC, eced6f0).
#   pull adapter_rX29V1 (the 910B's exp29_910b.sh) -> one offline-gate pass (run_offline_gates.sh, O3 on EXP-25's 3,000
#   seeded calls) -> EXP-29 verdict = every G-A line but G0 (EXP-28's G0 stands), O3(f), O1, O2, HM3D far STOP;
#   -> the code without an adapter reproduces A0 call for call (sdpa, 3 episodes)
#   -> PASS: V1 with lora_B = 0 reproduces A0 too, then FA2 val seed 42 (600-episode rule), futility (dSR <= -0.5 ->
#      no seed 1337), seed 1337, two-seed bootstrap -> H-29 verdict;
#      FAIL: S1-TM two-seed FA2 val, calibration only (no stop rules, no H-29 claim).
# Status $A/STATUS, log $A/auto.log, end state $A/SUMMARY.txt.  Helpers are exp28_auto.sh's (reviewed 10-17).
set -uo pipefail
WT=/workspace/exp23/wt_exp28; PY=/opt/conda/bin/python
X=/workspace/exp29; R1=/workspace/exp28/round1; G=/workspace/exp28/gates; A=/workspace/exp29/auto
C=/workspace/exp28/collect_base; COH=/workspace/exp28/episodes/cohorts; H=/workspace/exp28/episodes/splits/heldout_scenes.txt
SIG=$A/signals_out
A0_DIR=/workspace/eval_runs/exp20_a0_seed42_4090
declare -A A0=([42]=/workspace/eval_runs/exp20_a0_seed42_4090/merged/progress.jsonl
               [1337]=/workspace/eval_runs/exp20_a0_seed1337_4090/merged/progress.jsonl)
GIVE_UP=$(date -d "2026-10-19T00:00:00Z" +%s)      # no adapter by then: two seeds cannot finish before 10-20 14:00
LAST_1337=$(date -d "2026-10-20T06:00:00Z" +%s)    # moved 10-18 22:45 UTC after 20 h were lost to the expired export port: seed 1337 may end after the deadline
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
  local out="$G/x29v1_a$2"; [[ "$3" == 1 ]] || out="${out}_scale$3"
  EXP28_O3_MAX_CALLS=3000 EXP28_WT="$WT" EXP28_ADAPTER="$1" EXP28_NAME=x29v1 EXP28_ATTEMPT="$2" EXP28_LORA_SCALE="$3" EXP28_HOLDOUT_SCENES_FILE="$H" \
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

canary() {  # $1 label, $2 adapter ('' = none): sdpa, 3 episodes of shard 0, seed 42, call for call against A0
  local can="$X/canary_$1" try
  for try in 1 2; do
    rm -rf "$can"; st "per-call check against A0: $1 (sdpa, 3 episodes), try $try"
    ( cd "$WT" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES=5 PPA_EVAL_SHARDS=0 \
        PPA_EVAL_MAX_EPISODES_PER_SHARD=3 PPA_EVAL_PROTOCOL_SEED=42 PPA_EVAL_OUTPUT_ROOT="$can" PPA_EVAL_DISPLAY_BASE=380 \
        PPA_EVAL_ATTN_IMPL=sdpa ${2:+PPA_EVAL_SYSTEM2_LORA="$2"} bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) \
      > "$X/logs/canary_$1_t$try.log" 2>&1
    "$PY" "$WT/scripts/tools/compare_client_logs.py" --a "$A0_DIR/runtime/*/logs/client_shard_*.log" \
      --b "$can/runtime/*/logs/client_shard_*.log" --out "$A/canary_$1_t$try.json" --require-identical \
      >> "$A/auto.log" 2>&1 && return 0
  done
  return 1
}
two_seeds() {  # $1 = 1: H-29 rules (600-episode stop, futility, verdict); 0: calibration (both seeds, no stop rules)
  local rules=$1 r d42 dos v
  run_val 42 "$rules"; r=$?
  [[ $r -eq 2 ]] && finish "seed 42 stopped by the 600-episode rule ($ROUND)"
  [[ $r -eq 0 ]] || finish "seed 42 incomplete after 3 tries ($ROUND)"
  OUT42="/workspace/eval_runs/exp29_${ROUND}_seed42_4090_stepbudget"
  d42=$(jget "$X/paired_seed42.json" sr.delta_pt)
  "$PY" "$WT/scripts/tools/paired_closed_loop_bootstrap.py" --treatment "$OUT42/merged/progress.jsonl" --control "${A0[42]}" \
    --standard-protocol --bootstrap-draws 20000 --label "${ROUND}_seed42_vs_a0" --output-json "$A/seed42_bootstrap.json" \
    > "$A/seed42_bootstrap.txt" 2>&1
  dos=$("$PY" -c 'import json,sys; print(round(100*json.load(open(sys.argv[1]))["strata"]["all"]["os"]["diff"],2))' "$A/seed42_bootstrap.json" 2>/dev/null)
  note "$ROUND seed 42 paired (standard protocol): dSR=$d42 pt over $(jget "$X/paired_seed42.json" sr.n) episodes; dOS=${dos:-?} pt"
  [[ -n "$d42" ]] || finish "seed 42 has no paired dSR ($ROUND)"
  if [[ "$rules" -eq 1 ]] && "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= -0.5 else 1)' "$d42"; then
    finish "futility: $ROUND seed 42 dSR $d42 <= -0.5, no seed 1337 (H-29 refuted on seed 42 alone is not claimed; see the ledger)"
  fi
  (( $(date +%s) <= LAST_1337 )) || finish "no time for seed 1337 ($ROUND seed 42 dSR $d42)"
  run_val 1337 0; r=$?
  OUT1337="/workspace/eval_runs/exp29_${ROUND}_seed1337_4090_stepbudget"
  [[ $r -eq 0 ]] || finish "seed 1337 incomplete after 3 tries ($ROUND; seed 42 dSR $d42)"
  note "$ROUND seed 1337 paired: $(tr -d '\n' < "$X/paired_seed1337.json" 2>/dev/null | cut -c1-300)"
  "$PY" "$WT/scripts/tools/paired_closed_loop_bootstrap.py" --treatment "$OUT42/merged/progress.jsonl" \
    --treatment "$OUT1337/merged/progress.jsonl" --control "${A0[42]}" --control "${A0[1337]}" --standard-protocol \
    --bootstrap-draws 20000 --label "${ROUND}_vs_a0_two_seeds" --output-json "$A/final_bootstrap.json" > "$A/final_bootstrap.txt" 2>&1
  v=$("$PY" - "$A/final_bootstrap.json" "$rules" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["strata"]["all"]["success"]
d, lo, hi = 100 * s["diff"], 100 * s["ci95"][0], 100 * s["ci95"][1]
if sys.argv[2] == "1":
    h = "supported" if d >= 1.0 and lo > 0 else ("refuted" if d <= 0 else "not detected")
    print(f"H-29 {h}: two-seed dSR {d:+.2f} pt, 95% CI [{lo:+.2f}, {hi:+.2f}]")
else:
    print(f"calibration only (S1-TM, not a test of H-29): two-seed dSR {d:+.2f} pt, 95% CI [{lo:+.2f}, {hi:+.2f}]")
PY
)
  finish "${v:-the final bootstrap failed ($A/final_bootstrap.txt)}"
}

# 0 ------------------------------------------------------------------------------
st "waiting for adapter_rX29V1 on 52417"
pull adapter_rX29V1 "$X/adapter_pull" || finish "no adapter_rX29V1 by 10-19 00:00 UTC"
V1="$X/adapter_pull/adapter_rX29V1"
note "V1 pulled ($(sha256sum "$V1/adapter_model.safetensors" | cut -c1-16))"

# 1 ------------------------------------------------------------------------------
st "offline gates on V1 (one pass)"
next=$(gates "$V1" 1 1)
GO="$G/x29v1_a1"
verdict=$("$PY" - "$GO/verdict.json" "$GO/o3/gates.stratified.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
ga = s["G_A"]["criteria"]; o3 = s["O3_exp28"]["criteria"]; g = v["groups"]
items = {k: bool(c["pass"]) for k, c in ga.items() if k != "e_G0"}          # G0: EXP-28 run record 10
items.update({k: bool(o3[k]["pass"]) for k in ("f_le1_stop_keep", "f_pixel_1_3_stop_keep")})
items.update({"O1": bool(g["O1"]), "O2": bool(g["O2"]), "O4_stop_hm3d_far": bool(g["O4_stop"])})
fails = [k for k, ok in items.items() if not ok]
vals = {k: ga[k]["value"] for k in ga if k != "e_G0"}
print(("PASS" if not fails else "FAIL") + " fails=" + ",".join(fails) + " " + json.dumps(vals))
PY
)
note "EXP-29 gate verdict: ${verdict:-could not be read (gates next=$next)}"
chown -R 1015:1015 "$G/x29v1_a1" 2>/dev/null

# 2 ------------------------------------------------------------------------------
canary noadapter "" || finish "the code without an adapter does not reproduce A0 call for call: pairing invalid, no val"
note "per-call check without an adapter passed"
if [[ "$verdict" == PASS* ]]; then
  ROUND=X29V1; CAND="$V1"
  Z="$X/adapter_v1_zero"; rm -rf "$Z"
  "$PY" - "$V1" "$Z" <<'PY' || finish "could not write the zero-adapter copy"
import shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(src, dst)
w = load_file(str(src / "adapter_model.safetensors"))
keys = [k for k in w if ".lora_B." in k]
assert keys, "no lora_B tensors"
for k in keys: w[k] = w[k] * 0
save_file(w, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
PY
  canary zeroadapter "$Z" || finish "V1 with lora_B = 0 does not reproduce A0 call for call: the adapter path differs, no val"
  note "per-call check with the zeroed adapter passed; V1 goes to val (FA2, H-29 rules)"
  two_seeds 1
else
  ROUND=X29S1TM; CAND=/workspace/exp23/roundS1TM/adapter_pull/adapter_rS1TMs500
  [[ -s "$CAND/adapter_model.safetensors" ]] || finish "V1 failed its gates and the S1-TM adapter is missing: nothing goes to val"
  note "V1 failed its gates: S1-TM two-seed val as calibration only (FA2, no stop rules)"
  two_seeds 0
fi
