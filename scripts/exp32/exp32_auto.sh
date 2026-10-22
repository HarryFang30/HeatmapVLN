#!/usr/bin/env bash
# EXP-32's unattended tree on the 4090 (fjl-habitat), ledger EXP-32 第二部分 (bd87831, written before any part-1
# reading), stages in the ledger's timeline order:
#   1 wait for pass 1 to end (T0); A1-A4 on pass 1 and R0-S (aud.py validity); the pass-2 cohort (cohorts.sh pass2);
#   2 pass 2 (B1): 3 local sampling slots, and the 910B's sampling servers (tag $P2_TAG, 7 slots) once the 910B says
#     which set serves the pass (exp32_sampling_p2), claim the cohort's shards, System1 seed 42; launchers stopped by
#     PID 7 h after the first STATUS line, as exp31/r0_4090.sh does;
#   3 B2 on A-valid sources (build_rft_stream.py --dry-run): N_rft < 4,000 -> PASS3 to the 910B, pass 3 (seed 2024,
#     7 h; its external arm after exp32_sampling_p3); N_rft < 2,000 after it -> no training, end; then SAMPLING_DONE;
#   4 B3's held-out sampled runs start (local, seeds 42 then 1337); RFT stream, merge (dry run first), AUD (D1); G0's
#     records (R0-G) and D3's (150 npu RFT trajectories drawn with seed 32) to the 910B on 52413, its parity outputs
#     back on 52417 (D2: exact >= 0.98; D3: aud.py dose); the stream to the 910B on 52414-52416 only when D1-D3 pass;
#   5 adapter_rX32R1 from 52417 -> P starts (6); once the held-out runs end (a card never holds two models): A1/A4 on
#     them, run_offline_gates.sh as is (EXP28_O3_MAX_CALLS=3000, EXP-29's O1/O2 base files), rft_gate.py R and M (one
#     decode shard a card; B3's seed-2024 pass when R asks for it), hm3d_route_gate.py verdict --rules exp32 (the
#     "next" that run_offline_gates.sh's own exp28 verdict prints is ignored) -> F's table: val; scale_0.5 (attempt 2
#     on the lora_B x 0.5 copy, R on that copy, SCALE05 with the copy to the 910B, P again); retrain_2_epochs
#     (double_stream.py, RETRAIN_E2, adapter_rX32R1E2, attempt 2, P again); kill; invalid (fix the inputs, no retry);
#   6 P (G): the held-out cohort against the 910B serving the candidate (collect_4090.sh external, seed 42, never
#     sampling), re-collected at most twice while incomplete; the p_repro first-10 rerun check (>= 9/10 call for call)
#     -- failed: REBASE_P, the base on x32 into $X/p_base, P judged against it; canary_compare.py --criteria exp28-p
#     --require-device npu: kill -> no val, no retry;
#   7 H, each check twice at most, against A0 seed 42's client logs (sdpa, GPU 5, 3 episodes of shard 0, 69/69):
#     (1) no adapter, (2) no adapter + vision mask (fails: val without the mask), (3) the candidate with lora_B = 0;
#   8 val_unseen (J) seeds 42 -> 1337 -> 2024 -> 3407 on GPUs 5,4,0, PPA_EVAL_ATTN_IMPL=sdpa set here (the launcher's
#     default is FA2), step-budget STOP on, System2 adapter on, sampling env unset; seed 42's 600-episode rule against
#     the per-episode mean of its four pool arms (pool_ref_seed42.jsonl); after each seed pool_estimator.py estimate.
# Every wait has a deadline.  A watchdog writes $A/ALARMS (and the log) when a claimed shard of a running collection
# commits no record for 45 min or a collection STATUS_* stays the same for 2 h.  Status $A/STATUS, log $A/auto.log,
# end state $A/SUMMARY.txt.  Helpers are exp28_auto.sh's / exp29_auto.sh's with their lessons (one val directory
# name everywhere, control files of this run only, sdpa explicit, deadlines, a stale export port raises an alarm).
#
# The 910B side (scripts/exp32/exp32_910b.sh) and this script meet only through the tunnels (tests/
# test_exp32_controllers.py checks that the two scripts' walkthroughs name the same files, exports and tags):
#   signals, files pushed into its signal directory on 52517: PASS3 (B2 runs pass 3: keep or restart the sampled set
#     so it lives >= 8.5 h more), SAMPLING_DONE (pass 2/3 over: stop the sampling servers), SCALE05 (after
#     adapter_s05/*: serve that copy as x32s05), RETRAIN_E2 (receive stream_e2 on 52414-52416, train, export
#     adapter_rX32R1E2, serve it as x32e2), REBASE_P (serve the base from src_x32 as x32base), STOP_SERVING (stop, end);
#   52413: records/<scene>/<file> (R0-G, G0's records), then G0_PUSHED; <tag>/<scene>/<file> (D3's records, tag
#     p1|r0s|p2|p3: aud.py and rft_gate.py map a D3 row to its trajectory by that directory) and picks.json, then
#     D3_PUSHED;
#   52414-52416: the merged stream (and stream_e2);
#   52417 exports: exp32_sampling_p2 / exp32_sampling_p3 (the sampled set serving that pass is up: servers.json and
#     ready.txt; a pass's external arm starts only after it), npu_servers_exp28_<tag> (tags $P2_TAG, x32r1, x32s05,
#     x32e2, x32base), exp32_g0_npu and exp32_d3_npu (each parity_g0.{json,calls.jsonl} of train_system2_lora.py
#     --parity-check and its log), adapter_rX32R1, adapter_rX32R1E2.
# A rerun resumes: passes, cohorts, streams and pulls already done are kept, a signal or push already delivered is not
# sent again ($A/<signal>.sent, $A/g0d3.pushed, $A/stream.pushed); once STOP_SERVING has gone out it refuses to start.
#
#   nohup setsid bash /workspace/exp23/wt_x32/scripts/exp32/exp32_auto.sh > /workspace/exp32/auto.out 2>&1 &
#   bash scripts/exp32/exp32_auto.sh canaries12     # H (1) and (2) only, before T0 and only with GPU 5 free (T10);
#                                                   # stage 7 reuses a measured result of the same commit
#   EXP32_DRY=1 bash scripts/exp32/exp32_auto.sh    # walkthrough: prints every command, runs none, writes ./exp32_dry;
#                                                   # EXP32_DRY_<KNOB> scripts each decision (dryval below)
#   EXP32_LIB=1 source scripts/exp32/exp32_auto.sh  # the functions only (tests feed the parsers real artefacts)
set -uo pipefail
DRY="${EXP32_DRY:-0}"; LIB="${EXP32_LIB:-0}"
WT="${EXP32_WT:-/workspace/exp23/wt_x32}"; PY="${EXP32_PYTHON:-/opt/conda/bin/python}"
if [[ "$DRY" == 1 || "$LIB" == 1 ]]; then  # a walkthrough or a test: everything it writes stays under one local dir
  SANDBOX="${EXP32_DRY_DIR:-./exp32_dry}"; mkdir -p "$SANDBOX"; SANDBOX="$(cd "$SANDBOX" && pwd)"
  X="${EXP32_X:-$SANDBOX/exp32}"; G="${EXP32_GATES:-$SANDBOX/gates}"; EVAL="${EXP32_EVAL:-$SANDBOX/eval_runs}"
else
  X="${EXP32_X:-/workspace/exp32}"; G="${EXP32_GATES:-/workspace/exp28/gates}"; EVAL="${EXP32_EVAL:-/workspace/eval_runs}"
fi
A="$X/auto"; SIG="$A/signals_out"
# read only
P1=/workspace/exp32/collect_a; R0S=/workspace/exp31/r0s; R0G=/workspace/exp31/r0g
C=/workspace/exp28/collect_base; COH=/workspace/exp28/episodes/cohorts; P_REPRO=/workspace/exp28/p_repro/records
H=/workspace/exp28/episodes/splits/heldout_scenes.txt; T=/workspace/exp28/episodes/splits/train_scenes.txt
R2R=/workspace/exp23/data/interndata_n1/r2r; MODEL=/workspace/InternNav_Model
RPC_SRC="${EXP32_RPC_SRC:-/workspace/rpc/src}"  # vla_rpc, for the record readers (run_offline_gates.sh has its own)
O1_BASE=/workspace/exp23/b3/scalevln_b3.json; O2_BASE=/workspace/exp23/exp26/base_r2r_holdout.json
A0_DIR=/workspace/eval_runs/exp20_a0_seed42_4090
declare -A A0=([42]=/workspace/eval_runs/exp20_a0_seed42_4090/merged/progress.jsonl
               [1337]=/workspace/eval_runs/exp20_a0_seed1337_4090/merged/progress.jsonl)
# the 910B
P2_TAG="${EXP32_P2_TAG:-rfts}"  # its sampling server set: pass 1's, kept (or restarted under the same tag) for 2 and 3
EXPORT_PORT=52417; SIG_PORT=52517; G0D3_PORT=52413; STREAM_PORTS=52414,52415,52416
PAR_G0_EXPORT=exp32_g0_npu; PAR_D3_EXPORT=exp32_d3_npu; GPUS=5,4,0
SEED_ORDER=(42 1337 2024 3407)
# deadlines (seconds)
P1_UNTIL=$(date -d "${EXP32_P1_DEADLINE:-2026-10-21T14:00:00Z}" +%s)  # pass 1 (T0) must have ended by then
CUT_S=25200       # B1/B2: a pass is cut 7 h after its first STATUS line
READY_S=14400     # the 910B's exp32_sampling_<pass> (a restart of its set takes <= 3.5 h); else local slots only
HELD_S=21600      # one held-out sampled seed (112 episodes, 3 slots, about 2.5 h) is cut at 6 h
PUSH_S=43200; PARITY_S=14400; ADAPTER_S=43200; E2_S=64800; GATES_S=28800; DECODE_S=10800; P_S=43200
SERVER_WAIT_S=14400  # a P try that still waits for its npu_servers_exp28_<tag> after 4 h is given up (not retried)
CANARY_S=7200; VAL_S=144000  # val: 40 h a seed over all its tries (20-25 h expected)
# EXP32_NO_PARITY=1 (the user's call, 2026-10-21, ledger EXP-32 运行记录 4): no G0/D3 -- D2 and D3 are not run, the
# stream goes to the 910B right after D1, and M has no states (R gates as usual; M is needed only after R alone fails)
NO_PARITY="${EXP32_NO_PARITY:-0}"
# EXP32_NO_P=1 (the user's call, 2026-10-21, ledger EXP-32 运行记录 6): P is not run -- the 910B's new instance cannot
# serve closed-loop episodes (CANN 8.5.2).  The offline gates alone decide; P is recorded as 没测出来, never a pass.
NO_P="${EXP32_NO_P:-0}"

mkdir -p "$A" "$SIG" "$X/logs" "$X/aud" "$X/adapter_pull"
exec 3>&2  # x's walkthrough lines bypass every redirection
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$A/auto.log" > "$A/STATUS"; }
note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$A/auto.log" >> "$A/SUMMARY.txt"; }
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$A/auto.log"; }
alarm() { printf '%s ALARM %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$A/auto.log" >> "$A/ALARMS"; }
x() {  # run a command; under EXP32_DRY=1 print it instead (status 0)
  if [[ "$DRY" == 1 ]]; then printf 'DRY $ %s\n' "$*" >&3; log "DRY \$ $*"; return 0; fi
  "$@"
}
nap() { [[ "$DRY" == 1 ]] || sleep "$1"; }
dryval() {  # $1 knob, $2 default: a decision under EXP32_DRY=1 (EXP32_DRY_<knob>, a list is consumed one per call)
  local var="EXP32_DRY_$1" n=0 vals
  read -r -a vals <<< "${!var:-$2}"
  [[ -s "$A/.dry_$1" ]] && n=$(cat "$A/.dry_$1")
  echo $((n + 1)) > "$A/.dry_$1"
  (( n < ${#vals[@]} )) || n=$(( ${#vals[@]} - 1 ))
  printf '%s\n' "${vals[$n]}"
}
jget() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[int(k)] if isinstance(d, list) else d[k]
print(d)' "$1" "$2" 2>/dev/null; }
sha16() { sha256sum "$1/adapter_model.safetensors" 2>/dev/null | cut -c1-16; }
signal() {  # push files (paths relative to $SIG) to the 910B's signal receiver; gives up after $SIGNAL_S seconds
  printf '%s\n' "$@" > "$A/sig_files.txt"
  x timeout "${SIGNAL_S:-21600}" "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$SIG" --ports "$SIG_PORT" \
    --files "$A/sig_files.txt" --retries 1000000 >> "$A/signals.log" 2>&1
}
raise() {  # $1 signal name; $A/$1.sent once the 910B's receiver took it
  date -u +%FT%TZ > "$SIG/$1"; log "signal $1"; signal "$1" && date -u +%FT%TZ > "$A/$1.sent"
}
sent() { [[ -e "$A/$1.sent" ]] && log "$1 was sent before ($(cat "$A/$1.sent")): not again"; }  # a rerun's guard
own() { chown -R 1015:1015 "$A" "$X/aud" "$X/logs" "$X/adapter_pull" "$X/stream_rft" "$X/stream" "$X/stream_e2" \
          "$X/push_g0d3" "$X/parity_pull" "$X/sampling_pull" "$X"/canary_* "$X"/adapter_zero_* 2>/dev/null; }
bg_track() { echo "$1" >> "$A/bg.pids"; }  # a background job of this run: finish stops it
finish() {  # the end: background jobs, then the collections they left running ($A/watch.list), STOP_SERVING
  local p r
  st "END: $*"; printf '%s END: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$A/SUMMARY.txt"
  if [[ "$DRY" != 1 ]]; then
    for p in $(cat "$A/bg.pids" 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
    for r in $(sort -u "$A/watch.list" 2>/dev/null); do collect_alive "$r" && stop_collect "$r"; done
  fi
  date -u +%FT%TZ > "$SIG/STOP_SERVING"; SIGNAL_S=1800 signal STOP_SERVING  # free the 910B
  own; exit 0
}

# --- collections ---------------------------------------------------------------------------------------------
collectors() {  # $1 output root: pids of the collect_4090.sh wrappers and run_s2_dagger_collect_cuda.sh launchers on it
  local p
  for p in $(pgrep -f "collect_4090[.]sh|^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q -E "^(EXP28_OUT_DIR|PPA_COLLECT_OUTPUT_ROOT)=$1$" && echo "$p"
  done
}
collect_alive() { [[ "$DRY" != 1 && -n "$(collectors "$1")" ]]; }
arm_alive() {  # $1 output root, $2 mode (external|local): a collect_4090.sh of that mode (or its launcher) runs on it
  local p
  [[ "$DRY" == 1 ]] && return 1
  for p in $(collectors "$1"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^EXP28_MODE=$2$" && return 0
  done
  return 1
}
term_collect() {  # $1 output root: TERM its launchers (not their children), as exp31/r0_4090.sh
  local p
  for p in $(pgrep -f "^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
    ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" | grep -q run_s2_dagger_collect_cuda && continue
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^PPA_COLLECT_OUTPUT_ROOT=$1$" && kill -TERM "$p" && log "TERM launcher $p ($1)"
  done
}
stop_collect() {  # $1 output root: its launchers, then (3 min later) a collect_4090.sh still waiting for servers
  local p
  [[ "$DRY" == 1 ]] && { x stop_collect "$1"; return 0; }
  term_collect "$1"; sleep 180
  for p in $(collectors "$1"); do kill -TERM "$p" 2>/dev/null && log "TERM collector $p ($1)"; done
  sleep 30
}
status_of() { cat "$1"/STATUS_* 2>/dev/null | cut -c1-160 | paste -sd'|' -; }
records_in() { find "$1/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l; }
pass_over() {  # $1 collection dir: nothing collects into it and every STATUS_* it has says DONE or STOPPED
  local f n=0
  collect_alive "$1" && return 1
  for f in "$1"/STATUS_*; do
    [[ -f "$f" ]] || continue
    n=$((n + 1)); grep -q -E '^[^ ]+ (DONE|STOPPED):' "$f" || return 1
  done
  (( n > 0 ))
}
first_status_epoch() {  # $1 collection dir: the time of the earliest first STATUS line in its collect_4090.sh logs
  local f t best=""
  for f in "$1"/logs/collect_*.log; do
    t=$(grep -m1 ' STATUS ' "$f" 2>/dev/null | cut -d' ' -f1)
    [[ -n "$t" ]] && t=$(date -d "$t" +%s 2>/dev/null) || continue
    [[ -z "$best" || "$t" -lt "$best" ]] && best=$t
  done
  echo "$best"
}
spawn() {  # detach "$@" in its own session (from $WT), like exp31/r0_4090.sh's arms
  if [[ "$DRY" == 1 ]]; then x "$@"; return 0; fi
  ( cd "$WT" && exec setsid "$@" ) > /dev/null 2>&1 < /dev/null &
}
collect() {  # $1 mode, $2 out, $3 cohort, $4 System1 seed, $5 display base, $6 sampled (1/0), $7 tag[, $8 adapter]
  local shards env_s
  shards=$( [[ "$DRY" == 1 ]] && echo 20 || jget "$3/manifest.json" num_shards )
  if [[ "$6" == 1 ]]; then env_s=(env PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE=0.7 PPA_EVAL_SYSTEM2_SAMPLE_TOP_P=0.95)
  else env_s=(env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P); fi
  echo "$2" >> "$A/watch.list"
  spawn "${env_s[@]}" PPA_EVAL_PROTOCOL_SEED="$4" EXP28_MODE="$1" EXP28_TAG="$7" EXP28_WT="$WT" \
    EXP28_COHORT_DIR="$3" EXP28_OUT_DIR="$2" EXP28_SHARDS="0-$((${shards:-1} - 1))" EXP28_LOCAL_GPUS="$GPUS" \
    EXP28_DISPLAY_BASE="$5" ${8:+EXP28_ADAPTER="$8"} bash "$WT/scripts/exp28/collect_4090.sh"
}
sampling_ready() {  # $1 pass (p2|p3), $2 deadline (epoch): the 910B's exp32_sampling_<pass> pulled -- the sampled set
                    # that serves this pass is up and will not be replaced under it; 1 at the deadline
  pull "exp32_sampling_$1" "$X/sampling_pull" "$2" || return 1
  note "$1: the 910B's sampled set is up: $(head -n 1 "$X/sampling_pull/exp32_sampling_$1/ready.txt" 2>/dev/null | cut -c1-200)"
}
instance_check() {  # $1 pass, $2 out: 0 unless the external arm pulled another server set than exp32_sampling_<pass> named
  local want got
  want=$(jget "$X/sampling_pull/exp32_sampling_$1/servers.json" server_instance)
  got=$(jget "$2/npu_servers/npu_servers_exp28_$P2_TAG/servers.json" server_instance)
  [[ -z "$want" || -z "$got" || "$want" == "$got" ]]
}
pass_collect() {  # $1 name (p2|p3), $2 System1 seed, $3 cohort: B1/B2's pass, cut 7 h after its first STATUS line. Both
                  # arms start once the 910B exports exp32_sampling_$1; 1 when it has not within READY_S (nothing ran:
                  # a pass on the local slots alone would feed B2 a reading the design never meant)
  local out="$X/$1" t0 now started said_stop=0 said_inst=0
  if [[ -e "$A/$1.over" ]]; then note "$1 already over: $(status_of "$out")"; return 0; fi
  started=$(date +%s)
  if ! collect_alive "$out"; then
    st "$1 (System1 seed $2): waiting for the 910B's exp32_sampling_$1 on $EXPORT_PORT (<= $((READY_S / 3600)) h)"
    sampling_ready "$1" $(( started + READY_S )) || return 1
    collect external "$out" "$3" "$2" 801 0 "$P2_TAG"; nap 20
    collect local "$out" "$3" "$2" 811 1 "$1"
    started=$(date +%s)
  fi
  while ! pass_over "$out"; do
    [[ "$DRY" == 1 ]] && break
    now=$(date +%s); t0=$(first_status_epoch "$out"); t0=${t0:-$started}
    if (( now >= t0 + CUT_S )); then
      st "$1: 7 h since its first STATUS line ($(date -u -d "@$t0" +%FT%TZ)): stopping its launchers"
      stop_collect "$out"; break
    fi
    if (( now - started > 600 )) && ! collect_alive "$out"; then
      note "$1: nothing collects any more ($(status_of "$out")): the pass ends here"; break
    fi
    grep -q "waiting for the 910B" "$out/STATUS_external" 2>/dev/null && (( now - started > 1200 )) \
      && alarm "$1: the external launcher still waits for npu_servers_exp28_$P2_TAG after $(( (now - started) / 60 )) min"
    if (( ! said_stop )) && grep -q -E '^[^ ]+ STOPPED:' "$out/STATUS_external" 2>/dev/null && arm_alive "$out" local; then
      said_stop=1; alarm "$1: the external arm stopped while the local slots collect: $(cut -c1-200 "$out/STATUS_external")"
    fi
    if (( ! said_inst )) && ! instance_check "$1" "$out"; then
      said_inst=1; alarm "$1: the external arm pulled another server set than exp32_sampling_$1 names"
    fi
    st "$1 (System1 seed $2): $(status_of "$out")"
    sleep 600
  done
  : > "$A/$1.over"
  note "$1 over: $(records_in "$out") committed records; $(status_of "$out")"
}
heldout_collect() {  # $1 System1 seed: B3's held-out sampled run on the 3 local slots into $X/heldout_s$1, cut at 6 h
  local out="$X/heldout_s$1" started
  [[ -e "$A/heldout_s$1.over" ]] && return 0
  collect_alive "$out" || collect local "$out" "$HELD_COH" "$1" 831 1 "h$1"
  started=$(date +%s)
  while ! pass_over "$out"; do
    [[ "$DRY" == 1 ]] && break
    if (( $(date +%s) >= started + HELD_S )); then log "heldout_s$1: 6 h, stopping it"; stop_collect "$out"; break; fi
    sleep 300
  done
  : > "$A/heldout_s$1.over"
  log "heldout_s$1 over: $(records_in "$out") committed records; $(status_of "$out")"
}
heldout_bg() { heldout_collect 42; heldout_collect 1337; date -u +%FT%TZ > "$A/HELDOUT_DONE"; }
cohort() {  # $1 mode (pass2|heldout|pass3), $2 out[, $3 non-empty: the collections read have no collector left, though
            # a STATUS_* may not say DONE/STOPPED]: scripts/exp32/cohorts.sh, or the cohort it already built
  [[ "$DRY" != 1 && -s "$2/cohort/manifest.json" ]] && return 0
  x env EXP32_WT="$WT" EXP32_PYTHON="$PY" EXP32_P1="$P1" EXP32_R0S="$R0S" EXP32_P2="$X/p2" EXP32_BASE="$C" \
    EXP32_COHORT="$COH" EXP32_HOLDOUT_SCENES="$H" EXP32_TRAIN_SCENES="$T" ${3:+EXP32_ALLOW_UNFINISHED=1} \
    bash "$WT/scripts/exp32/cohorts.sh" "$1" "$2" >> "$A/cohorts.log" 2>&1
}
watchdog() {  # background: alarms for $A/watch.list's collections (a claimed shard 45 min without a committed record;
              # a STATUS_* unchanged for 2 h), every 10 min while the main script lives; each alarm at most hourly
  local d f now age claim shard owner last
  declare -A said=()
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    now=$(date +%s)
    while read -r d; do
      [[ -n "$d" && -d "$d" && -n "$(collectors "$d")" ]] || continue  # only collections still running
      for f in "$d"/STATUS_*; do
        [[ -f "$f" ]] || continue
        grep -q -E '^[^ ]+ (DONE|STOPPED):' "$f" && continue
        age=$(( now - $(stat -c %Y "$f") ))
        if (( age >= 7200 && now - ${said[$f]:-0} >= 3600 )); then
          said[$f]=$now; alarm "$f unchanged for $((age / 60)) min: $(cut -c1-160 "$f")"
        fi
      done
      for claim in "$d"/workers/shard_*.claim; do
        [[ -d "$claim" ]] || continue
        shard="${claim%.claim}"; [[ -e "$shard/DONE" ]] && continue
        owner=$(cut -d' ' -f1 "$claim/owner" 2>/dev/null)
        [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null || continue
        last=$(stat -c %Y "$claim/owner" "$shard/progress.json" 2>/dev/null | sort -n | tail -1)
        age=$(( now - ${last:-$now} ))
        if (( age >= 2700 && now - ${said[$shard]:-0} >= 3600 )); then
          said[$shard]=$now; alarm "$(basename "$shard") of $d: no committed record for $((age / 60)) min"
        fi
      done
    done < <(sort -u "$A/watch.list" 2>/dev/null)
    sleep 600
  done
}

# --- the 910B's exports ---------------------------------------------------------------------------------------
pull() {  # $1 export name, $2 destination dir, $3 deadline (epoch); 1 at the deadline.  A port that refuses for an
          # hour raises an alarm (EXP-29 lost 20 h to an expired export port).
  local since; since=$(date +%s)
  until [[ -e "$2/$1/.pulled" ]]; do
    (( $(date +%s) < $3 )) || return 1
    x "$PY" "$WT/scripts/exp23/pull_dirs.py" "$2" "$EXPORT_PORT" "$1" >> "$A/pull.log" 2>&1
    [[ "$DRY" == 1 ]] && return 0
    [[ -e "$2/$1/.pulled" ]] && break
    if tail -n 3 "$A/pull.log" | grep -q -i -E "refused|reset|timed out"; then
      (( $(date +%s) - since >= 3600 )) && { alarm "export port $EXPORT_PORT has not answered for $1 for an hour"; since=$(date +%s); }
    else since=$(date +%s); fi
    sleep 300
  done
}

# --- stage 3/4 parsers ---------------------------------------------------------------------------------------
validity() {  # $1 report name, then TAG=DIR:SEED ...: aud.py validity (A1-A4) -> $X/aud/validity_$1.json; 0 every source valid
  local name=$1 rc s l args=(); shift
  for s in "$@"; do args+=(--source "$s"); done
  x "$PY" "$WT/scripts/exp32/aud.py" validity "${args[@]}" --anchor-dir "$C" --git-repo "$WT" \
    --out "$X/aud/validity_$name.json" > "$X/aud/validity_$name.txt" 2>&1
  rc=$?; [[ "$DRY" == 1 ]] && rc=$(dryval "VALIDITY_${name^^}" 0)
  note "A1-A4 ($name): $(grep -m1 -E '^VALIDITY ' "$X/aud/validity_$name.txt" 2>/dev/null || tail -n 1 "$X/aud/validity_$name.txt" 2>/dev/null)"
  grep -E '^  ' "$X/aud/validity_$name.txt" 2>/dev/null | head -n 8 | while read -r l; do note "  $l"; done  # void reasons, A3 notes
  return "$rc"
}
n_rft_of() {  # $1 build_rft_stream.py --dry-run stdout: B2's N_rft, the sum of drawn's category_rft_*
  "$PY" - "$1" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
first, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
print(sum(int(v) for k, v in first["drawn"].items() if k.startswith("category_rft_")))
PY
}
dose_rule() {  # $1 pass: N_rft of the A-valid sources so far (build_rft_stream.py --dry-run), printed
  local out="$X/aud/rft_dryrun_$1.txt"
  x "$PY" "$WT/scripts/exp32/build_rft_stream.py" "${RFT_SRC[@]}" "${BRS_ARGS[@]}" --dry-run --out-dir "$X/stream_rft" \
    > "$out" 2> "$X/aud/rft_dryrun_$1.err"
  if [[ "$DRY" == 1 ]]; then dryval NRFT 7500; else n_rft_of "$out" 2>/dev/null; fi
}
stage_g0d3() {  # G0's records (R0-G, all) and D3's (150 npu RFT trajectories, seed 32, from rft_records.txt) linked
                # under $1 as records/<scene>/<file> (the 910B's G0 root) and <tag>/<scene>/<file>; prints the files
                # to push (relative to $1), G0's first, then D3's and picks.json
  "$PY" - "$R0G/records" "$X/stream_rft/rft_records.txt" "$1" <<'PY'
import json, os, random, sys
from pathlib import Path
r0g, listing, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
files = []
def link(src, rel):
    for f in (src, src.with_name(src.name + ".sha256")):
        dst = out / rel.parent / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_symlink():
            os.symlink(f, dst)
        files.append(str(dst.relative_to(out)))
g0 = sorted(p for p in r0g.glob("*/*.s2rec.tar") if p.with_name(p.name + ".sha256").is_file())
for p in g0:
    link(p, Path("records") / p.parent.name / p.name)
rows = [line.split("\t") for line in listing.read_text().splitlines() if line.strip()]
npu = sorted((path, tag) for path, device, tag in rows if device == "npu")
pick = sorted(random.Random(32).sample(npu, min(150, len(npu))))
for path, tag in pick:
    link(Path(path), Path(tag) / Path(path).parent.name / Path(path).name)
(out / "picks.json").write_text(json.dumps({"g0_records": len(g0), "d3_npu_pool": len(npu), "d3_seed": 32,
                                            "d3": [{"path": p, "tag": t} for p, t in pick]}, indent=1) + "\n")
print("\n".join(files + ["picks.json"]))
if not g0 or len(pick) < min(150, len(npu)) or not pick:
    raise SystemExit(f"G0 records {len(g0)}, D3 picks {len(pick)} of {len(npu)} npu trajectories")
PY
}
d2_line() {  # $1 G0's parity_g0.json: D2 (exact >= 0.98, npu records replayed on npu, split all); 0 pass
  "$PY" - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("g0_pass")) and float(d["exact_rate"]) >= 0.98 and d.get("record_device") == "npu" \
    and d.get("device") == "npu" and d.get("split") == "all"
print(f"{'PASS' if ok else 'FAIL'} D2 G0: exact {d['exact_rate']} over {d.get('n')} calls "
      f"(records {d.get('record_device')}, device {d.get('device')}, split {d.get('split')})")
sys.exit(0 if ok else 1)
PY
}
push_files() {  # $1 list (paths relative to $X/push_g0d3), $2 seconds: those files to the 910B's parity receiver
  x timeout "$2" "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$X/push_g0d3" --ports "$G0D3_PORT" \
    --files "$1" --retries 1000000 >> "$A/push_g0d3.log" 2>&1
}
push_g0d3() {  # G0's records then G0_PUSHED, D3's records and picks.json then D3_PUSHED (exp32_910b.sh's stage 3 waits
               # for each marker before that check); $A/g0d3.pushed once all of it arrived
  push_files "$A/g0_files.txt" "$PUSH_S" && push_files "$A/G0_marker.txt" 3600 \
    && push_files "$A/d3_files.txt" "$PUSH_S" && push_files "$A/D3_marker.txt" 3600 \
    && date -u +%FT%TZ > "$A/g0d3.pushed"
}
adapter_pulled() { [[ "$DRY" != 1 && -e "$X/adapter_pull/adapter_rX32R1/.pulled" ]]; }  # the 910B consumed stage 4's pushes

# --- stage 5 parsers -----------------------------------------------------------------------------------------
gates_dir() { local d="$G/x32r1_a$1"; [[ "$2" == 1 ]] || d="${d}_scale$2"; echo "$d"; }  # $1 attempt, $2 scale
gates() {  # $1 adapter, $2 attempt, $3 lora scale: one gate pass (run_offline_gates.sh unchanged; a second run resumes
           # when the first left no scored O3/O4); 0 when O1-O4 were scored
  local out try; out=$(gates_dir "$2" "$3")
  for try in 1 2; do
    x timeout "$GATES_S" env EXP28_O3_MAX_CALLS=3000 EXP28_WT="$WT" EXP28_ADAPTER="$1" EXP28_NAME=x32r1 \
      EXP28_ATTEMPT="$2" EXP28_LORA_SCALE="$3" EXP28_GATE_ROOT="$G" EXP28_GPUS="$GPUS" EXP28_HOLDOUT_SCENES_FILE="$H" \
      EXP28_O4_RECORDS="$C/records" EXP28_O1_BASE="$O1_BASE" EXP28_O2_BASE="$O2_BASE" \
      bash "$WT/scripts/exp28/run_offline_gates.sh" > "$A/gates_a$2_scale$3_t$try.log" 2>&1
    [[ "$DRY" == 1 ]] && return 0
    [[ -s "$out/o3/gates.stratified.json" && -s "$out/o4/hm3d.json" && -s "$out/o4/hm3d.calls.jsonl" ]] && return 0
    note "gate pass a$2 scale $3 scored nothing on try $try ($(cat "$out/STATUS" 2>/dev/null))"
  done
  return 1
}
m_args() { [[ "$NO_PARITY" == 1 ]] || printf '%s\n' --m-states "$1/m_states.jsonl"; }  # $1 rgate dir (none: no D3)
rgate_decode() {  # $1 rgate dir, $2 adapter: rft_gate.py decode, one shard a card on $GPUS at once.  PYTHONPATH as
                  # exp30/probe1_4090.sh sets it: inference_messages decodes the recorded JPEGs through vla_rpc.
  local gl i rc=0 pids=() ma=()
  IFS=',' read -r -a gl <<< "$GPUS"; mapfile -t ma < <(m_args "$1")
  for i in "${!gl[@]}"; do
    if [[ "$DRY" == 1 ]]; then
      x env PYTHONPATH="$WT:$RPC_SRC" CUDA_VISIBLE_DEVICES="${gl[$i]}" "$PY" "$WT/scripts/exp32/rft_gate.py" decode \
        --r-states "$1/r_states.jsonl" \
        ${ma[@]+"${ma[@]}"} --adapter "$2" --model-path "$MODEL" --output "$1/rft_gate.json" \
        --shard "$i/${#gl[@]}" --device cuda
      continue
    fi
    PYTHONPATH="$WT:$RPC_SRC" CUDA_VISIBLE_DEVICES="${gl[$i]}" timeout "$DECODE_S" "$PY" "$WT/scripts/exp32/rft_gate.py" decode \
      --r-states "$1/r_states.jsonl" ${ma[@]+"${ma[@]}"} --adapter "$2" --model-path "$MODEL" \
      --output "$1/rft_gate.json" --shard "$i/${#gl[@]}" --device cuda > "$1/decode_$i.log" 2>&1 &
    pids+=($!)
  done
  for i in ${pids[@]+"${pids[@]}"}; do wait "$i" || rc=1; done
  return $rc
}
r_sources() {  # R_SRC: select-r's sources, B3's seeds 42 and 1337, and 2024 once its pass has run
  R_SRC=(--source "s42=$X/heldout_s42/records:42" --source "s1337=$X/heldout_s1337/records:1337")
  [[ -e "$A/heldout_s2024.over" ]] && R_SRC+=(--source "s2024=$X/heldout_s2024/records:2024")
  return 0
}
r_measure() {  # $1 gate pass dir, $2 the adapter it gated: R and M into $1/rgate; returns score's exit (0, 1, 2 invalid)
  local rg="$1/rgate" rc ma=()
  mkdir -p "$rg"; r_sources; mapfile -t ma < <(m_args "$rg")
  x "$PY" "$WT/scripts/exp32/rft_gate.py" select-r "${R_SRC[@]}" --holdout-scenes-file "$H" --failed \
    --output "$rg/r_states.jsonl" > "$rg/select_r.log" 2>&1 || return 2
  if [[ "$NO_PARITY" != 1 ]]; then
    x "$PY" "$WT/scripts/exp32/rft_gate.py" select-m --d3-calls "$PAR_D3/parity_g0.calls.jsonl" \
      --rft-stream "$X/stream_rft" --output "$rg/m_states.jsonl" > "$rg/select_m.log" 2>&1 || return 2
  fi
  rgate_decode "$rg" "$2" || { note "R/M decode failed ($rg/decode_*.log)"; return 2; }
  x "$PY" "$WT/scripts/exp32/rft_gate.py" score --r-states "$rg/r_states.jsonl" ${ma[@]+"${ma[@]}"} \
    --adapter "$2" --output "$rg/rft_gate.json" > "$rg/score.txt" 2>&1
  rc=$?; [[ "$DRY" == 1 ]] && rc=$(dryval RGATE 0)
  log "R/M: $(head -n 1 "$rg/score.txt" 2>/dev/null)"
  return "$rc"
}
rgate_needs() {  # $1 rft_gate.json: what the measurement still needs ("heldout_s2024" when R has < 150 divergent
                 # states without B3's seed-2024 pass), else empty
  if [[ "$DRY" == 1 ]]; then dryval RNEEDS heldout_s2024; return; fi
  jget "$1" gate.needs
}
r_gate() {  # $1 gate pass dir, $2 adapter: r_measure, with B3's seed-2024 pass (once) when R asks for it
  local rc
  r_measure "$1" "$2"; rc=$?
  if [[ $rc -eq 2 && "$(rgate_needs "$1/rgate/rft_gate.json")" == heldout_s2024 && ! -e "$A/heldout_s2024.over" ]]; then
    note "R: fewer than 150 divergent states without seed 2024 -> B3's seed-2024 held-out pass (local, 6 h at most)"
    st "B3: held-out sampled run, System1 seed 2024"
    heldout_collect 2024
    validity heldout_s2024 "heldout_s2024=$X/heldout_s2024:2024" || { note "heldout_s2024 is void (A1/A4)"; return 2; }
    r_measure "$1" "$2"; rc=$?  # select-r now reads seed 2024 too; decode resumes and decodes only the new states
  fi
  return "$rc"
}
verdict32() {  # $1 gate pass dir, $2 attempt: hm3d_route_gate.py verdict --rules exp32 -> $1/verdict_exp32.json
  mkdir -p "$1"
  x "$PY" "$WT/scripts/exp28/hm3d_route_gate.py" verdict --rules exp32 --o1 "$1/o1/scalevln.json" --o1-base "$O1_BASE" \
    --o2 "$1/o2/r2r.json" --o2-base "$O2_BASE" --o3 "$1/o3/gates.stratified.json" --o4 "$1/o4/hm3d.json" \
    --o4-calls "$1/o4/hm3d.calls.jsonl" --rgate "$1/rgate/rft_gate.json" --attempt "$2" \
    --output "$1/verdict_exp32.json" > "$1/verdict_exp32.txt" 2>&1
}
verdict_next() {  # $1 verdict_exp32.json: its "next" (val, scale_0.5, retrain_2_epochs, kill, invalid); no_verdict when
                  # the file is not EXP-32's verdict (run_offline_gates.sh's own verdict.json is EXP-28's: ignored)
  if [[ "$DRY" == 1 ]]; then dryval NEXT val; return; fi
  "$PY" - "$1" 2>/dev/null <<'PY' || echo no_verdict
import json, sys
v = json.load(open(sys.argv[1]))
if v.get("schema") != "heatmapvln-exp32-offline-gates-verdict-v1":
    raise SystemExit(1)
print(v["next"])
PY
}
branch_of() {  # $1 the verdict's "next", $2 attempt -> go | scale | retrain | kill | invalid (F's table: one retry)
  case "$1" in
    val) echo go ;;
    scale_0.5) if [[ "$2" == 1 ]]; then echo scale; else echo kill; fi ;;
    retrain_2_epochs) if [[ "$2" == 1 ]]; then echo retrain; else echo kill; fi ;;
    kill) echo kill ;;
    *) echo invalid ;;
  esac
}
gate_pass() {  # $1 adapter, $2 attempt, $3 scale[, $4 non-empty: the gates already ran]: gates, R/M, the EXP-32
               # verdict -> BRANCH (a global: this runs in the main shell, never inside $(...))
  local out; out=$(gates_dir "$2" "$3")
  BRANCH=invalid
  if [[ -z "${4:-}" ]]; then gates "$1" "$2" "$3" || { note "gate pass a$2 (scale $3) scored nothing twice"; return; }; fi
  [[ "$3" == 1 ]] || set -- "$out/adapter_scale$3" "$2" "$3"  # R reads the very copy O1-O4 gated
  st "R and M on attempt $2 (adapter $(sha16 "$1"))"
  r_gate "$out" "$1"
  verdict32 "$out" "$2"
  note "gates attempt $2 (scale $3): $(head -n 1 "$out/verdict_exp32.txt" 2>/dev/null) | R: $(head -n 1 "$out/rgate/score.txt" 2>/dev/null)"
  BRANCH=$(branch_of "$(verdict_next "$out/verdict_exp32.json")" "$2")
}

# --- stage 6 parsers -----------------------------------------------------------------------------------------
p_judge() {  # $1 candidate collection dir, $2 base records, $3 json: canary_compare.py's exp28-p line
  local line
  if [[ "$DRY" == 1 ]]; then x "$PY" canary_compare.py --canary-records "$1/records" --base-records "$2" --criteria exp28-p
    dryval P "PASS"; return; fi
  line=$("$PY" "$WT/scripts/tools/canary_compare.py" --canary-records "$1/records" --base-records "$2" --cohort "$COH" \
           --cohort-part heldout --criteria exp28-p --require-device npu --output "$3" 2>>"$A/auto.log" \
         | grep -m1 -E "^(PASS|KILL|INCOMPLETE|INVALID) P ")
  echo "${line:-INVALID P no verdict}"
}
sampled_logs() {  # $1 collection dir: model-server logs of a greedy collection that say System2 sampled (A1's text)
  grep -l -E "System2 SAMPLES" "$1"/npu_servers/*/logs/model_*.log "$1"/runtime/*/logs/model_*.log \
    "$1"/runtime/*/external_server_logs/model_*.log 2>/dev/null | wc -l
}
rerun_check() {  # G's validity: the first 10 committed p_repro records against Phase-0, call for call (>= 9/10); once
  local rr="$A/p_repro_first10" sha d line
  [[ -s "$A/p_identity.line" ]] && { cat "$A/p_identity.line"; return; }
  if [[ "$DRY" == 1 ]]; then x canary_compare.py --rerun-records "$rr" --reference-records "$C/records" \
      --min-identical-episodes 9; line="$(dryval RERUN PASS) per-call identity (walkthrough)"
  else
    rm -rf "$rr"; mkdir -p "$rr"
    for sha in $(ls -tr "$P_REPRO"/*/*.s2rec.tar.sha256 2>/dev/null | head -n 10); do
      d="$rr/$(basename "$(dirname "$sha")")"; mkdir -p "$d"; ln -sf "${sha%.sha256}" "$d/"; ln -sf "$sha" "$d/"
    done
    line=$("$PY" "$WT/scripts/tools/canary_compare.py" --rerun-records "$rr" --reference-records "$C/records" \
             --min-identical-episodes 9 --output "$A/p_identity.json" 2>>"$A/auto.log" | grep -m1 -E "^(PASS|FAIL) per-call")
  fi
  echo "${line:-FAIL per-call identity: no verdict}" | tee "$A/p_identity.line"
}
p_collect() {  # background: the held-out cohort against the 910B's $1 servers into $3 (adapter $2), re-collected at
               # most twice while incomplete (not a kill); the line against Phase-0 -> $A/P_$1.collected.  A try still
               # waiting for npu_servers_exp28_$1 after SERVER_WAIT_S ends P as not judged (no more tries)
  local tag=$1 out=$3 try line="" job t0 cmd
  for try in 1 2 3; do
    echo "$out" >> "$A/watch.list"
    cmd=(timeout "$P_S" env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P PPA_EVAL_PROTOCOL_SEED=42
         EXP28_MODE=external EXP28_TAG="$tag" ${2:+EXP28_ADAPTER="$2"} EXP28_WT="$WT" EXP28_COHORT_DIR="$COH"
         EXP28_OUT_DIR="$out" EXP28_SHARDS=0-19 EXP28_DISPLAY_BASE="${P_DISPLAY:-761}" bash "$WT/scripts/exp28/collect_4090.sh")
    if [[ "$DRY" == 1 ]]; then x "${cmd[@]}"
    else
      "${cmd[@]}" > /dev/null 2>&1 &
      job=$!; t0=$(date +%s)
      while kill -0 "$job" 2>/dev/null; do
        sleep 60
        grep -q "waiting for the 910B" "$out/STATUS_external" 2>/dev/null && (( $(date +%s) - t0 >= SERVER_WAIT_S )) || continue
        kill -TERM "$job" 2>/dev/null; wait "$job"
        line="INVALID P: no npu_servers_exp28_$tag on $EXPORT_PORT within $((SERVER_WAIT_S / 3600)) h (try $try)"
        alarm "P ($tag): $line"; break
      done
      wait "$job" 2>/dev/null
      pass_over "$out" || stop_collect "$out"
      [[ "$line" == INVALID* ]] && break
    fi
    line=$(p_judge "$out" "$C/records" "$A/p_${tag}_vs_phase0_t$try.json")
    [[ "$line" == INCOMPLETE* ]] || break
  done
  echo "$line" > "$A/P_$tag.collected"
}
p_start() {  # $1 tag, $2 adapter, $3 out: P's collection in the background
  [[ "$NO_P" == 1 ]] && { log "P ($1) not started (EXP32_NO_P=1)"; return 0; }
  rm -f "$A/P_$1.collected"
  log "P: collecting against the 910B's $1 ($(sha16 "$2")) into $3"
  if [[ "$DRY" == 1 ]]; then p_collect "$@"; return; fi
  p_collect "$@" &
  echo $! > "$A/P_$1.pid"; bg_track $!
}
p_stop() {  # $1 tag, $2 out: that P is moot (the gates chose a retry): its job and its launchers
  local pid; pid=$(cat "$A/P_$1.pid" 2>/dev/null)
  [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null
  [[ "$DRY" == 1 ]] || { collect_alive "$2" && stop_collect "$2"; }
  log "P ($1) stopped: the gates chose a retry"
}
p_rebase() {  # G: the base on x32 (REBASE_P, the 910B, npu) over the held-out cohort into $X/p_base, complete; once
  local line=""
  [[ -s "$A/P_x32base.collected" ]] && line=$(cat "$A/P_x32base.collected")
  if [[ -z "$line" ]]; then
    st "P: the rerun check failed -> REBASE_P, the base on x32 over the held-out cohort (about 5 h)"
    raise REBASE_P || return 1
    P_DISPLAY=771 p_collect x32base "" "$X/p_base"; line=$(cat "$A/P_x32base.collected")
  fi
  note "P's base rerun on x32 against Phase-0 (report only): $line; sampled server logs: $(sampled_logs "$X/p_base")"
  [[ "$line" == PASS* || "$line" == KILL* ]] && [[ "$(sampled_logs "$X/p_base")" -eq 0 ]]
}
p_final() {  # $1 tag, $2 out: P on the final candidate; 0 pass, 1 kill, 2 not judged
  local tag=$1 out=$2 base="$C/records" line rerun deadline
  st "P: waiting for the $tag collection"
  deadline=$(( $(date +%s) + 3 * P_S ))  # three collection passes at most
  until [[ -s "$A/P_$tag.collected" ]]; do
    (( $(date +%s) < deadline )) || { note "P ($tag): no collection result by the deadline"; return 2; }
    sleep 300
  done
  line=$(cat "$A/P_$tag.collected")
  [[ "$line" == INCOMPLETE* || "$line" == INVALID* ]] && { note "P ($tag) not judged: $line"; return 2; }
  [[ "$(sampled_logs "$out")" -eq 0 ]] || { note "P ($tag): its server logs say System2 sampled (A1's text): invalid"; return 2; }
  rerun=$(rerun_check)
  if [[ "$rerun" != PASS* ]]; then
    p_rebase || { note "P: the base rerun on x32 did not complete"; return 2; }
    base="$X/p_base/records"
  fi
  line=$(p_judge "$out" "$base" "$A/p_$tag.json")
  note "P ($tag, against $base): $line | rerun check (first 10): $rerun"
  case "$line" in PASS*) return 0 ;; KILL*) return 1 ;; *) return 2 ;; esac
}

# --- stage 7 -------------------------------------------------------------------------------------------------
canary_ok() {  # $1 compare_client_logs.py json: H's 69/69 (3 shared episodes, all identical, 69 calls compared)
  "$PY" - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ok = r["shared_episodes"] == 3 and r["identical"] == 3 and r["calls_compared"] == 69
print(f"{'PASS' if ok else 'FAIL'} {r['identical']}/{r['shared_episodes']} episodes identical, {r['calls_compared']} calls")
sys.exit(0 if ok else 1)
PY
}
eval_env() {  # $1 vision mask (1/0): EV = the env prefix of a val or canary run (sampling never; the mask only on 1;
              # env takes its -u options before the first assignment)
  if [[ "$1" == 1 ]]; then
    EV=(env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P HEATMAPVLN_QWEN_VISION_MASK_PATCH=1)
  else
    EV=(env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P -u HEATMAPVLN_QWEN_VISION_MASK_PATCH)
  fi
}
wt_head() { git -c safe.directory='*' -C "$WT" rev-parse --short=12 HEAD 2>/dev/null; }
canary() {  # $1 label, $2 adapter ('' = none), $3 mask: H's check, twice at most; the line.  Only a measurement (the
            # comparison of all 3 episodes) is kept, as "<line> @<commit>" in $A/H_$1.result, and reused only at that
            # commit: a run that crashed (out of memory on a busy GPU 5, say) measured nothing and is run again
  local can="$X/canary_$1" try line="FAIL not run" head cached measured=0
  head=$(wt_head)
  if [[ -s "$A/H_$1.result" ]]; then
    cached=$(cat "$A/H_$1.result")
    if [[ "$cached" == *" @$head" && -n "$head" ]]; then echo "${cached% @*}"; return; fi
    log "H ($1): the kept result '$cached' is not a measurement at $head: run again"; rm -f "$A/H_$1.result"
  fi
  eval_env "$3"
  for try in 1 2; do
    log "H: per-call check against A0 ($1), try $try"
    x rm -rf "$can"; rm -f "$A/canary_$1_t$try.json"  # an earlier run's comparison is never read as this one's
    ( { cd "$WT" || [[ "$DRY" == 1 ]]; } && x timeout "$CANARY_S" "${EV[@]}" PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" \
        PPA_EVAL_GPU_DEVICES=5 PPA_EVAL_SHARDS=0 PPA_EVAL_MAX_EPISODES_PER_SHARD=3 PPA_EVAL_PROTOCOL_SEED=42 \
        PPA_EVAL_OUTPUT_ROOT="$can" PPA_EVAL_DISPLAY_BASE=380 PPA_EVAL_ATTN_IMPL=sdpa ${2:+PPA_EVAL_SYSTEM2_LORA="$2"} \
        bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) > "$X/logs/canary_$1_t$try.log" 2>&1
    x "$PY" "$WT/scripts/tools/compare_client_logs.py" --a "$A0_DIR/runtime/*/logs/client_shard_*.log" \
      --b "$can/runtime/*/logs/client_shard_*.log" --out "$A/canary_$1_t$try.json" --require-identical >> "$A/auto.log" 2>&1
    if [[ "$DRY" == 1 ]]; then line="$(dryval CANARY PASS) $1 (walkthrough)"; measured=1
    else
      line=$(canary_ok "$A/canary_$1_t$try.json" 2>/dev/null); [[ -n "$line" ]] || line="FAIL no comparison ($1)"
      measured=0; [[ "$(jget "$A/canary_$1_t$try.json" shared_episodes)" == 3 ]] && measured=1
    fi
    [[ "$line" == PASS* ]] && break
  done
  echo "$line (try $try)"
  if (( measured )); then echo "$line (try $try) @$head" > "$A/H_$1.result"
  else log "H ($1): the last try compared no full 3 episodes: nothing kept"; fi
}
zero_copy() {  # $1 adapter, $2 destination: the copy with every lora_B tensor zeroed (H (3))
  x rm -rf "$2"
  x "$PY" - "$1" "$2" <<'PY'
import shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(src, dst)
w = load_file(str(src / "adapter_model.safetensors"))
keys = [k for k in w if ".lora_B." in k]
assert keys, "no lora_B tensors"
for k in keys:
    w[k] = w[k] * 0
save_file(w, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
PY
}
gpu5_busy() {  # why H's canaries cannot run on GPU 5 now ('' = they can): pass 1 collects, or GPU 5 holds memory
               # (nvidia-smi's -i is PCI order, as the launchers' CUDA_DEVICE_ORDER=PCI_BUS_ID)
  local used
  collect_alive "$P1" && { echo "pass 1 still collects into $P1"; return; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] || { echo "nvidia-smi cannot read GPU 5"; return; }
  (( used <= 2048 )) || echo "GPU 5 has $used MiB in use"
}
canaries12() {  # H (1) and (2); sets MASK
  local h1 h2
  h1=$(canary noadapter "" 0); note "H (1) no adapter: $h1"
  [[ "$h1" == PASS* ]] || return 1
  h2=$(canary mask "" 1); note "H (2) no adapter + vision mask: $h2"
  if [[ "$h2" == PASS* ]]; then MASK=1; else MASK=0; note "H (2) failed: val runs without the vision mask"; fi
}

# --- stage 8 -------------------------------------------------------------------------------------------------
val_out() { echo "$EVAL/exp32_${ROUND}_seed$1_4090_stepbudget"; }  # the one name run_val, the rule and the estimate read
merged_rows() { wc -l < "$1/merged/progress.jsonl" 2>/dev/null || echo 0; }
val_launcher_pids() {  # $1 output root: the val launchers writing there
  local p
  for p in $(pgrep -f "^bash scripts/run_ppa_r2r_val_unseen_cuda[.]sh$"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^PPA_EVAL_OUTPUT_ROOT=$1$" && echo "$p"
  done
}
term_val() {  # $1 output root: stop that val run (launcher process group, then stray clients)
  local p
  [[ "$DRY" == 1 ]] && { x term_val "$1"; return; }
  for p in $(val_launcher_pids "$1"); do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  sleep 60
  for p in $(pgrep -f "[r]2r_val_unseen\.py.*$1"); do kill -TERM "$p" 2>/dev/null; done
}
pool_ref() {  # $1 seed (42|1337): the per-episode mean of that seed's four pool arms (pool_estimator.py pool-ref)
  [[ "$DRY" != 1 && -s "$X/pool_ref_seed$1.jsonl" ]] && return 0
  x "$PY" "$WT/scripts/exp32/pool_estimator.py" pool-ref --seed "$1" --out "$X/pool_ref_seed$1.jsonl" >> "$A/auto.log" 2>&1
}
rule600() {  # $1 seed-42 run dir: the 600-episode rule against pool_ref_seed42.jsonl; 3 = it fired, 0 go on, 2 no pair
  x "$PY" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$1/workers/shard_*/progress.json" \
    --reference "$X/pool_ref_seed42.jsonl" --min-episodes 600 --abort-delta -3.0 --out "$A/rule600_seed42.json" \
    >> "$A/auto.log" 2>&1
}
monitor_a0() {  # $1 seed, $2 run dir: report only, paired with A0 of the same seed (42, 1337)
  [[ -n "${A0[$1]:-}" ]] || return 0
  x "$PY" "$WT/scripts/tools/paired_progress_monitor.py" --candidate "$2/workers/shard_*/progress.json" \
    --reference "${A0[$1]}" --out "$A/paired_a0_seed$1.json" > /dev/null 2>&1
}
run_val() {  # $1 seed, $2 the 600-episode rule (1: seed 42). 0 complete, 1 incomplete, 2 stopped by the rule, 4 sampled
  local seed=$1 rule=$2 out try rc launcher started
  out=$(val_out "$seed"); started=$(date +%s)
  eval_env "$MASK"
  for try in 1 2 3; do  # the run and at most two resumes (J)
    st "val seed $seed ($ROUND, mask $MASK), try $try -> $out"
    if [[ "$DRY" == 1 ]]; then
      ( cd "$WT" 2>/dev/null; x "${EV[@]}" PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES="$GPUS" \
          PPA_EVAL_PROTOCOL_SEED="$seed" PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_SYSTEM2_LORA="$CAND" PPA_EVAL_OUTPUT_ROOT="$out" \
          PPA_EVAL_DISPLAY_BASE=400 PPA_EVAL_SHARD_RETRIES=3 PPA_EVAL_STOP_AT_STEP_BUDGET=1 \
          setsid bash scripts/run_ppa_r2r_val_unseen_cuda.sh )
      if [[ "$rule" == 1 ]]; then rule600 "$out"; [[ "$(dryval RULE600 0)" == 3 ]] && return 2; fi
      [[ "$(dryval VALROWS 1839)" == 1839 ]] && return 0
      continue
    fi
    ( cd "$WT" && exec "${EV[@]}" PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_EVAL_GPU_DEVICES="$GPUS" \
        PPA_EVAL_PROTOCOL_SEED="$seed" PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_SYSTEM2_LORA="$CAND" PPA_EVAL_OUTPUT_ROOT="$out" \
        PPA_EVAL_DISPLAY_BASE=400 PPA_EVAL_SHARD_RETRIES=3 PPA_EVAL_STOP_AT_STEP_BUDGET=1 \
        setsid bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) > "$X/logs/val_${ROUND}_seed${seed}_try$try.log" 2>&1 &
    launcher=$!
    while kill -0 "$launcher" 2>/dev/null; do
      sleep 1200
      if (( $(date +%s) > started + VAL_S )); then
        note "seed $seed: past its $((VAL_S / 3600)) h deadline"; term_val "$out"; wait "$launcher"; return 1
      fi
      monitor_a0 "$seed" "$out"
      if [[ "$rule" == 1 ]]; then
        rule600 "$out"; rc=$?
        if [[ $rc -eq 3 ]]; then
          note "seed $seed: the 600-episode rule fired ($(tr -d '\n' < "$A/rule600_seed42.json" | cut -c1-300))"
          term_val "$out"; wait "$launcher"; return 2
        fi
      fi
    done
    wait "$launcher"; rc=$?
    note "seed $seed try $try: launcher exit $rc, merged rows $(merged_rows "$out")"
    [[ "$(merged_rows "$out")" -eq 1839 ]] && break
  done
  monitor_a0 "$seed" "$out"
  [[ "$(merged_rows "$out")" -eq 1839 ]] || return 1
  [[ "$(sampled_logs "$out")" -eq 0 ]] || { note "seed $seed: a model-server log says System2 sampled"; return 4; }
  return 0
}
estimate() {  # $1 output stem, then SEED=RUN ...: pool_estimator.py estimate; prints its decision line, returns its exit
              # (0 run the next seed, 3 decided, 1 refused)
  local stem=$1 rc args=(); shift
  for s in "$@"; do args+=(--candidate "$s"); done
  x "$PY" "$WT/scripts/exp32/pool_estimator.py" estimate "${args[@]}" --out "$stem.json" > "$stem.txt" 2>&1
  rc=$?
  if [[ "$DRY" == 1 ]]; then rc=$(dryval EST 0); echo "walkthrough estimate k=$# exit $rc"; return "$rc"; fi
  head -n 1 "$stem.txt"
  return "$rc"
}
next_seed_of() {  # $1 estimate json: the next seed its decision names (empty when the decision is terminal)
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))["decision"]
print("" if d["terminal"] else d["next_seeds"][0])' "$1" 2>/dev/null
}

[[ "$LIB" == 1 ]] && { return 0 2>/dev/null || exit 0; }
if [[ "${1:-}" == canaries12 ]]; then  # T10: H (1) and (2) before T0, on GPU 5 when it is free
  if [[ "$DRY" != 1 ]]; then
    why=$(gpu5_busy); [[ -z "$why" ]] || { echo "GPU 5 busy ($why): H (1) and (2) run in stage 7"; exit 1; }
  fi
  MASK=0; canaries12; st "canaries12: $(cat "$A"/H_noadapter.result "$A"/H_mask.result 2>/dev/null | paste -sd'|' -)"; exit 0
fi

if [[ "$DRY" != 1 && -e "$SIG/STOP_SERVING" ]]; then  # the 910B controller has ended: no stage resumes without it
  note "refused to start: an earlier run sent STOP_SERVING ($(cat "$SIG/STOP_SERVING")), so exp32_910b.sh has ended;" \
    "restart it and remove STOP_SERVING from $SIG and from its signal dir to go on (H and val after P: by hand)"
  st "END: refused: STOP_SERVING was sent before"; exit 1
fi
MAIN_PID=$$
if [[ "$DRY" != 1 ]]; then watchdog & WATCHDOG=$!; trap 'kill "$WATCHDOG" 2>/dev/null' EXIT; fi
rm -f "$A/HELDOUT_DONE" "$A"/P_*.collected "$A"/P_*.pid "$A/bg.pids" "$A"/.dry_*  # an earlier run's control files
[[ "$DRY" == 1 ]] && rm -f "$A"/*.sent "$A"/*.pushed "$A"/*.over  # a walkthrough repeats every step
RFT_SRC=(--rft-source "p1=$P1/records:1337" --rft-source "r0s=$R0S/records:1337")
BRS_ARGS=(--anchor-root "$C/records" --train-scenes-file "$T" --holdout-scenes-file "$H" --first-calls 48 --min-distance 3
          --max-anchors-per-episode 6 --anchor-ratio 1.0 --anchor-seed 32 --max-trajectories-per-episode 2 --no-pad)
VALID=("$X/aud/validity_p1r0s.json")
HELD_COH="$X/heldout_cohort/cohort"
PAR_G0="$X/parity_pull/$PAR_G0_EXPORT"; PAR_D3="$X/parity_pull/$PAR_D3_EXPORT"
note "start (wt $WT at $(git -c safe.directory='*' -C "$WT" rev-parse --short HEAD 2>/dev/null), dry=$DRY)"
for f in "$A"/H_*.result; do [[ -s "$f" ]] && note "kept H result $(basename "$f"): $(cat "$f")"; done

# 0 ------------------------------------------------------------------------------------------------------------
[[ "$DRY" == 1 ]] || git -c safe.directory='*' -C "$WT" diff --quiet HEAD -- || finish "$WT has uncommitted changes"
pool_ref 42 && pool_ref 1337 || finish "pool_estimator.py pool-ref refused: the reference pool is not the ledger's (停查)"
cohort heldout "$X/heldout_cohort" || finish "the held-out cohort (B3) could not be built ($A/cohorts.log)"

# 1 ------------------------------------------------------------------------------------------------------------
st "waiting for pass 1 ($P1) to end"
# over = it has a STATUS and nothing collects into it (a wrapper killed by hand may leave its STATUS mid-word)
until [[ "$DRY" == 1 ]] || { ls "$P1"/STATUS_* > /dev/null 2>&1 && ! collect_alive "$P1"; }; do
  (( $(date +%s) < P1_UNTIL )) || finish "pass 1 has not ended by $(date -u -d "@$P1_UNTIL" +%FT%TZ): $(status_of "$P1")"
  sleep 600
done
note "pass 1 over (T0): $(records_in "$P1") committed records; $(status_of "$P1")"
st "A1-A4 on pass 1 and R0-S"
validity p1r0s "p1=$P1:1337" "r0s=$R0S:1337" \
  || finish "A1-A4: a source is void ($X/aud/validity_p1r0s.txt); pass 1 is resampled only by hand"
# a source left mid-word by a wrapper killed by hand has no collector left: the cohort reads what it committed
allow=""; [[ "$DRY" == 1 ]] || { pass_over "$P1" && pass_over "$R0S"; } || allow=1
cohort pass2 "$X/p2_cohort" "$allow" || finish "the pass-2 cohort could not be built ($A/cohorts.log)"
note "pass-2 cohort: $(jget "$X/p2_cohort/cohort/manifest.json" total_episodes) episodes"

# 2 ------------------------------------------------------------------------------------------------------------
st "pass 2: System1 seed 42, the 910B's $P2_TAG servers and 3 local sampling slots"
pass_collect p2 42 "$X/p2_cohort/cohort" \
  || finish "the 910B never said its sampled set serves pass 2 (no exp32_sampling_p2 within $((READY_S / 3600)) h): pass 2 did not start; rerun once the 910B serves (not a B2 reading)"
validity p2 "p2=$X/p2:42" || finish "A1-A4: pass 2 is void ($X/aud/validity_p2.txt)"
VALID+=("$X/aud/validity_p2.json"); RFT_SRC+=(--rft-source "p2=$X/p2/records:42")

# 3 ------------------------------------------------------------------------------------------------------------
st "B2: N_rft after pass 2"
n2=$(dose_rule p2); [[ "$n2" =~ ^[0-9]+$ ]] || finish "build_rft_stream.py --dry-run gave no N_rft ($X/aud/rft_dryrun_p2.err)"
note "B2: N_rft after pass 2 = $n2"
if (( n2 < 4000 )); then
  note "B2: N_rft $n2 < 4,000 -> pass 3 (System1 seed 2024)"
  # the 910B keeps (or restarts) its sampled set for pass 3 only on PASS3; sent first, so it works while the cohort builds
  [[ -e "$A/p3.over" ]] || sent PASS3 || SIGNAL_S=3600 raise PASS3 \
    || finish "could not send PASS3: the 910B would end its sampled set, and pass 3 would run on the local slots only"
  # a launcher stopped at the cutoff may leave a STATUS mid-word; so may pass 1 or R0-S (as above)
  allow=""; [[ "$DRY" == 1 ]] || { pass_over "$P1" && pass_over "$R0S" && pass_over "$X/p2"; } || allow=1
  cohort pass3 "$X/p3_cohort" "$allow" || finish "the pass-3 cohort could not be built ($A/cohorts.log)"
  pass_collect p3 2024 "$X/p3_cohort/cohort" \
    || finish "the 910B never said its sampled set serves pass 3 (no exp32_sampling_p3 within $((READY_S / 3600)) h): pass 3 did not start; rerun once the 910B serves (not a B2 reading)"
  validity p3 "p3=$X/p3:2024" || finish "A1-A4: pass 3 is void ($X/aud/validity_p3.txt)"
  VALID+=("$X/aud/validity_p3.json"); RFT_SRC+=(--rft-source "p3=$X/p3/records:2024")
  n3=$(dose_rule p3); [[ "$n3" =~ ^[0-9]+$ ]] || finish "build_rft_stream.py --dry-run gave no N_rft ($X/aud/rft_dryrun_p3.err)"
  note "B2: N_rft after pass 3 = $n3"
  (( n3 >= 2000 )) || finish "B2: N_rft $n3 < 2,000 after pass 3: no training, the data are reported only (EXP-32 stops)"
fi
sent SAMPLING_DONE || raise SAMPLING_DONE \
  || note "could not send SAMPLING_DONE: the 910B ends its sampled set only at its own deadline"

# 4 ------------------------------------------------------------------------------------------------------------
st "B3: held-out sampled runs start (local); building the RFT stream"
if [[ "$DRY" == 1 ]]; then heldout_bg; else heldout_bg & bg_track $!; fi
HELD_UNTIL=$(( $(date +%s) + 2 * HELD_S + 3600 ))
if [[ "$DRY" == 1 || ! -s "$X/stream_rft/stream_manifest.json" ]]; then
  x "$PY" "$WT/scripts/exp32/build_rft_stream.py" "${RFT_SRC[@]}" "${BRS_ARGS[@]}" --out-dir "$X/stream_rft" \
    > "$X/logs/stream_rft.log" 2>&1 || finish "the RFT stream failed ($X/logs/stream_rft.log)"
fi
note "RFT stream: $(tail -n 1 "$X/logs/stream_rft.log" 2>/dev/null)"
G0D3_PID=""
if [[ "$NO_PARITY" == 1 ]]; then
  note "G0/D3 not run (EXP32_NO_PARITY=1, the user's call 2026-10-21): D2 and D3 skipped, M unmeasured"
elif [[ -e "$A/g0d3.pushed" ]] || adapter_pulled; then
  log "G0's and D3's records went to the 910B before ($(cat "$A/g0d3.pushed" 2>/dev/null)): not again"
else
  st "G0's and D3's records to the 910B ($G0D3_PORT)"
  mkdir -p "$X/push_g0d3"
  if [[ "$DRY" == 1 ]]; then x stage_g0d3 "$X/push_g0d3"
  else
    stage_g0d3 "$X/push_g0d3" > "$A/g0d3_files.txt" 2>> "$A/auto.log" || finish "could not stage G0's and D3's records"
    grep '^records/' "$A/g0d3_files.txt" > "$A/g0_files.txt"; grep -v '^records/' "$A/g0d3_files.txt" > "$A/d3_files.txt"
  fi
  note "G0/D3 staged: $("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))
print(d["g0_records"], "R0-G records (G0); D3", len(d["d3"]), "of", d["d3_npu_pool"], "npu RFT trajectories, seed 32")' \
    "$X/push_g0d3/picks.json" 2>/dev/null)"
  for m in G0_PUSHED D3_PUSHED; do date -u +%FT%TZ > "$X/push_g0d3/$m"; echo "$m" > "$A/${m%_PUSHED}_marker.txt"; done
  if [[ "$DRY" == 1 ]]; then push_g0d3; else push_g0d3 & G0D3_PID=$!; bg_track $!; fi
fi
if [[ "$DRY" == 1 || ! -s "$X/stream_replay_r2r/stream_manifest.json" ]]; then
  x "$PY" "$WT/scripts/tools/build_replay_stream.py" --root "r2r=$R2R" --samples r2r=8000 --exclude-holdout \
    --stop-upsample 5 --out-dir "$X/stream_replay_r2r" > "$X/logs/stream_replay_r2r.log" 2>&1 \
    || finish "the r2r replay stream failed ($X/logs/stream_replay_r2r.log)"
fi
MERGE=(--dagger-stream "$X/stream_rft" --replay-stream "$X/stream_replay_r2r" --holdout-scenes-file "$H"
       --replay-fraction 0.2 --replay-shares equal --allow-other-rules --out-dir "$X/stream")
if [[ "$DRY" == 1 || ! -s "$X/stream/stream_manifest.json" ]]; then
  x "$PY" "$WT/scripts/exp28/merge_streams.py" "${MERGE[@]}" --dry-run > "$A/merge_dryrun.txt" 2>&1 \
    || finish "merge --dry-run refused ($A/merge_dryrun.txt)"
  note "merge (dry run, for the ledger): $(tr -s ' \n' ' ' < "$A/merge_dryrun.txt" 2>/dev/null | cut -c1-600)"
  x "$PY" "$WT/scripts/exp28/merge_streams.py" "${MERGE[@]}" > "$X/logs/merge.log" 2>&1 || finish "merge refused ($X/logs/merge.log)"
fi
st "D1: AUD"
AUDV=(); for v in "${VALID[@]}"; do AUDV+=(--validity "$v"); done
x "$PY" "$WT/scripts/exp32/aud.py" stream --stream "$X/stream" "${AUDV[@]}" --anchor-root "$C/records" \
  --train-scenes-file "$T" --holdout-scenes-file "$H" --git-repo "$WT" --code-commit HEAD --out "$X/aud/stream.json" \
  > "$X/aud/stream.txt" 2>&1
rc=$?; [[ "$DRY" == 1 ]] && rc=$(dryval AUD 0)
note "D1: $(grep -m1 '^AUD ' "$X/aud/stream.txt" 2>/dev/null | cut -c1-400)"
[[ $rc -eq 0 ]] || finish "D1 (AUD) did not pass: no training; fix the builder, the lines stay ($X/aud/stream.txt)"
if [[ "$NO_PARITY" != 1 ]]; then
st "waiting for the G0/D3 push and G0's parity output ($PAR_G0_EXPORT on $EXPORT_PORT)"
[[ -n "$G0D3_PID" ]] && { wait "$G0D3_PID" || finish "pushing G0's and D3's records failed ($A/push_g0d3.log)"; }
par_until=$(( $(date +%s) + PARITY_S ))  # G0's and D3's exports share it
pull "$PAR_G0_EXPORT" "$X/parity_pull" "$par_until" \
  || finish "no $PAR_G0_EXPORT on $EXPORT_PORT within $((PARITY_S / 3600)) h: D2 unread, no training"
if [[ "$DRY" == 1 ]]; then d2="$(dryval D2 PASS) D2 (walkthrough)"; else d2=$(d2_line "$PAR_G0/parity_g0.json" 2>&1); fi
note "$d2"
[[ "$d2" == PASS* ]] || finish "D2 (G0) did not pass: no training"
st "waiting for D3's parity output ($PAR_D3_EXPORT on $EXPORT_PORT)"
pull "$PAR_D3_EXPORT" "$X/parity_pull" "$par_until" \
  || finish "no $PAR_D3_EXPORT on $EXPORT_PORT within $((PARITY_S / 3600)) h of the G0/D3 push: D3 unread, no training"
x "$PY" "$WT/scripts/exp32/aud.py" dose --parity-dir "$PAR_D3" --rft-stream "$X/stream_rft" \
  --m-states "$X/aud/m_states.jsonl" --out "$X/aud/dose.json" > "$X/aud/dose.txt" 2>&1
rc=$?; [[ "$DRY" == 1 ]] && rc=$(dryval D3 0)
note "D3: $(grep -m1 '^D3 ' "$X/aud/dose.txt" 2>/dev/null | cut -c1-400)"
[[ $rc -eq 0 ]] || finish "D3 did not pass (1 - exact < 0.10, or unreadable): no training ($X/aud/dose.txt)"
fi
if [[ -e "$A/stream.pushed" ]] || adapter_pulled; then
  log "the stream went to the 910B before ($(cat "$A/stream.pushed" 2>/dev/null)): not again"
else
  st "pushing the stream ($STREAM_PORTS)"
  x timeout "$PUSH_S" "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$X/stream" --ports "$STREAM_PORTS" \
    --retries 1000000 --busy-sleep 15 > "$A/push_stream.log" 2>&1 || finish "pushing the stream failed ($A/push_stream.log)"
  date -u +%FT%TZ > "$A/stream.pushed"
  note "stream pushed: $(tail -n 1 "$A/push_stream.log" 2>/dev/null | cut -c1-160)"
fi

# 5 ------------------------------------------------------------------------------------------------------------
st "waiting for adapter_rX32R1 on $EXPORT_PORT"
pull adapter_rX32R1 "$X/adapter_pull" $(( $(date +%s) + ADAPTER_S )) || finish "no adapter_rX32R1 within $((ADAPTER_S / 3600)) h"
A1="$X/adapter_pull/adapter_rX32R1"
[[ "$DRY" == 1 || -s "$A1/adapter_model.safetensors" ]] || finish "the pulled adapter_rX32R1 has no weights"
note "adapter_rX32R1 pulled ($(sha16 "$A1")); export files: $(ls "$A1" 2>/dev/null | paste -sd, -)"
p_start x32r1 "$A1" "$X/p_cand"  # 6 runs beside 5 (G: 与 F 并行)
st "waiting for the held-out sampled runs (they hold GPUs $GPUS)"
until [[ -e "$A/HELDOUT_DONE" || "$DRY" == 1 ]]; do
  (( $(date +%s) < HELD_UNTIL )) || finish "the held-out sampled runs have not ended by their deadline"
  sleep 300
done
validity heldout "heldout_s42=$X/heldout_s42:42" "heldout_s1337=$X/heldout_s1337:1337" \
  || finish "A1/A4: a held-out sampled run is void: R is unmeasurable (invalid: fix and rerun, not a retry)"
st "offline gates, attempt 1"
gate_pass "$A1" 1 1
case "$BRANCH" in
  go) ROUND=X32R1; CAND="$A1"; PTAG=x32r1; POUT="$X/p_cand" ;;
  scale)
    p_stop x32r1 "$X/p_cand"
    GO2=$(gates_dir 2 0.5); S05="$GO2/adapter_scale0.5"
    st "gates attempt 2: lora_B x 0.5 (P on that copy beside it)"
    ga=""; if [[ "$DRY" == 1 ]]; then gates "$A1" 2 0.5; else gates "$A1" 2 0.5 & ga=$!; fi
    until [[ -z "$ga" || -s "$S05/adapter_model.safetensors" ]]; do  # run_offline_gates.sh scales the copy first
      kill -0 "$ga" 2>/dev/null || break; sleep 30
    done
    if [[ "$DRY" == 1 || -s "$S05/adapter_model.safetensors" ]]; then
      x rm -rf "$SIG/adapter_s05"; x cp -r "$S05" "$SIG/adapter_s05"  # P serves the very bytes the gates and val read
      files=(adapter_s05/adapter_model.safetensors adapter_s05/adapter_config.json)
      [[ "$DRY" == 1 ]] || mapfile -t files < <(cd "$SIG" && find adapter_s05 -type f | sort)
      if signal "${files[@]}" && raise SCALE05; then p_start x32s05 "$S05" "$X/p_cand_s05"
      else note "could not send the scaled adapter or SCALE05: P on it cannot run"; fi
    fi
    [[ -n "$ga" ]] && wait "$ga"
    [[ "$DRY" == 1 || -s "$GO2/o4/hm3d.json" ]] || finish "gates attempt 2 (scale 0.5) scored nothing"
    gate_pass "$A1" 2 0.5 ran
    [[ "$BRANCH" == go ]] || finish "stopped at gates attempt 2 (scale 0.5): $BRANCH, no val"
    ROUND=X32R1S05; CAND="$S05"; PTAG=x32s05; POUT="$X/p_cand_s05" ;;
  retrain)
    p_stop x32r1 "$X/p_cand"
    st "retrain_2_epochs: the two-pass stream, RETRAIN_E2, push"
    x "$PY" "$WT/scripts/exp28/ops/double_stream.py" --src "$X/stream" --dst "$X/stream_e2" > "$A/double_stream.log" 2>&1 \
      || finish "could not build the two-pass stream ($A/double_stream.log)"
    raise RETRAIN_E2 || finish "could not signal RETRAIN_E2 to the 910B"
    x timeout "$PUSH_S" "$PY" "$WT/scripts/tools/stream_transfer.py" push --src "$X/stream_e2" --ports "$STREAM_PORTS" \
      --retries 1000000 --busy-sleep 15 > "$A/push_stream_e2.log" 2>&1 || finish "pushing the two-pass stream failed"
    st "waiting for adapter_rX32R1E2 on $EXPORT_PORT"
    pull adapter_rX32R1E2 "$X/adapter_pull" $(( $(date +%s) + E2_S )) || finish "no adapter_rX32R1E2 within $((E2_S / 3600)) h"
    E2="$X/adapter_pull/adapter_rX32R1E2"
    [[ "$DRY" == 1 || -s "$E2/adapter_model.safetensors" ]] || finish "the pulled adapter_rX32R1E2 has no weights"
    note "adapter_rX32R1E2 pulled ($(sha16 "$E2"))"
    p_start x32e2 "$E2" "$X/p_cand_e2"
    gate_pass "$E2" 2 1
    [[ "$BRANCH" == go ]] || finish "stopped at gates attempt 2 (two epochs): $BRANCH, no val"
    ROUND=X32R1E2; CAND="$E2"; PTAG=x32e2; POUT="$X/p_cand_e2" ;;
  kill) finish "stopped at gates attempt 1: kill (F's table), no val" ;;
  *) finish "gates attempt 1 could not be read ($BRANCH): fix the inputs and rerun the pass (not a retry)" ;;
esac
note "candidate $ROUND ($(sha16 "$CAND")) passed its gates; P on the 910B's $PTAG"

# 6 ------------------------------------------------------------------------------------------------------------
if [[ "$NO_P" == 1 ]]; then
  note "P not run (EXP32_NO_P=1, the user's call 2026-10-21: the 910B's new instance cannot serve closed-loop" \
    "episodes under CANN 8.5.2): 没测出来, not a pass; the offline gates alone sent $ROUND to H and val"
else
  p_final "$PTAG" "$POUT"; r=$?
  case $r in
    0) note "P passed: $ROUND goes on to H and val" ;;
    1) finish "P kill ($ROUND): no val, no retry" ;;
    *) finish "P could not be judged ($ROUND): re-collect the missing episodes or fix the inputs; not a kill" ;;
  esac
fi
date -u +%FT%TZ > "$SIG/STOP_SERVING"; SIGNAL_S=7200 signal STOP_SERVING || note "could not send STOP_SERVING"

# 7 ------------------------------------------------------------------------------------------------------------
MASK=0
canaries12 || finish "H (1): the code without an adapter does not reproduce A0 call for call: no val"
zero_copy "$CAND" "$X/adapter_zero_$ROUND" || finish "could not write the zeroed copy of $ROUND"
h3=$(canary "zero_$ROUND" "$X/adapter_zero_$ROUND" "$MASK"); note "H (3) $ROUND with lora_B = 0 (mask $MASK): $h3"
[[ "$h3" == PASS* ]] || finish "H (3): $ROUND with lora_B = 0 does not reproduce A0 call for call: no val"

# 8 ------------------------------------------------------------------------------------------------------------
DONE_SEEDS=(); seed=42
while [[ -n "$seed" ]]; do
  rule=0; [[ "$seed" == 42 ]] && rule=1
  run_val "$seed" "$rule"; r=$?
  case $r in
    2) finish "seed 42 stopped by the 600-episode rule (paired with the per-episode mean of its four pool arms, <= -3.0 pt)" ;;
    1) finish "seed $seed incomplete after 3 tries ($(merged_rows "$(val_out "$seed")") of 1839)" ;;
    4) finish "seed $seed is invalid: a model server sampled" ;;
  esac
  DONE_SEEDS+=("$seed")
  args=(); for s in "${DONE_SEEDS[@]}"; do args+=("$s=$(val_out "$s")"); done
  line=$(estimate "$A/estimate_k${#DONE_SEEDS[@]}" "${args[@]}"); r=$?
  note "J after seed $seed: $line | paired with A0 seed $seed (report only): $(jget "$A/paired_a0_seed$seed.json" sr.delta_pt) pt"
  case $r in
    3) finish "$line" ;;
    0) if [[ "$DRY" == 1 ]]; then seed=${SEED_ORDER[${#DONE_SEEDS[@]}]:-}
       else seed=$(next_seed_of "$A/estimate_k${#DONE_SEEDS[@]}.json"); fi
       [[ -n "$seed" ]] || finish "J named no next seed after ${DONE_SEEDS[*]} ($line)" ;;
    *) finish "pool_estimator.py refused after seeds ${DONE_SEEDS[*]}: $line" ;;
  esac
done
