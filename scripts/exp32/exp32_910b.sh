#!/usr/bin/env bash
# EXP-32 on the 910B (ledger EXP-32 part 2, B-G; writes only under ~/work/zhr/zhr_1, its own files under exp32/).
# Start it at T0, once pass 1's two 4090 launchers are DONE (the 4090 starts pass 2 only after stage 1 says so, so the
# order of the two controllers' starts does not matter):
#   nohup setsid bash scripts/exp32/exp32_910b.sh > ~/work/zhr/zhr_1/exp32/exp32_910b.out 2>&1 < /dev/null &
# Status in $D/STATUS (STOPPED: / DONE: end it), log $D/exp32.log.  The names it shares with exp32_auto.sh (signals,
# markers, record roots, exports, server tags) are checked against that script by tests/test_exp32_controllers.py.
# Stages:
#   0  refuse unless the export port 52417 (serve_dirs) has >= 7 days to live and $SRC has the top_k fix; the signal
#      receiver on 52517 (tunnel from the 4090) writes into $SIG.
#   1  pass 2's sampled servers (T 0.7, top_p 0.95, top_k 0; cards 0-6, ports 52410-16 / 52510-16): keep pass 1's set
#      (tag rfts) if it serves the base, sampled, from $SRC's own commit (the 4090's x32 clients refuse another one)
#      with >= 10 h to live; otherwise restart it from $SRC under the same tag for 10 h (a set with clients on it is
#      replaced only once they leave, 8 h at most).  Then export exp32_sampling_p2 (its servers.json and ready.txt) on
#      52417: the 4090 starts pass 2 only once it has pulled that.
#   2  pass 2's set serves until SAMPLING_DONE, or PASS3 (B2 triggers pass 3: keep the set, or restart it, so it lives
#      >= 8.5 h more, then export exp32_sampling_p3), or 12 h after stage 1 (the 4090 may take any time between the
#      passes, so no idle rule yet).  Pass 3's set serves until SAMPLING_DONE, 30 minutes without a client (counted
#      once a client was seen 1 h or more after PASS3), or 8.5 h after PASS3.
#   3  receivers: parity records on 52413 into $IN, the stream on 52414-52416 into $D/stream.  G0 = R0-G's records
#      pushed as records/<scene>/<file> from /workspace/exp31/r0g, then a file G0_PUSHED; --parity-check 2000
#      --parity-split all, adapters off, exact >= 0.98.  D3 = the 150 npu RFT records pushed as <tag>/<scene>/<file>
#      (tag p1, r0s, p2 or p3, so aud.py dose and rft_gate.py select-m can tell copies of one episode apart), then
#      D3_PUSHED; the same plus --parity-device npu, 1 - exact >= 0.10.  Each is exported (parity_g0.json,
#      parity_g0.calls.jsonl, the log) as exp32_g0_npu / exp32_d3_npu on 52417, before it is judged; either failing
#      stops here.
#   4  train from the base on the stream (ledger E, the command of c1_910b.sh); export adapter_rX32R1 only if
#      train_log.jsonl has its done event, manifest.json says loss_normalization token, and the run took
#      total_samples / 64 steps (the done event, manifest.json and the last step row agree).
#   5  serve it, greedy (no sampling env), tag x32r1, for P's candidate arm.
#   6  act on the 4090's signals, oldest first; each replaces the set that serves, so the 4090 sends one only once it
#      is done with the current set: SCALE05 (serve $SIG/adapter_s05, pushed before it, as x32s05), RETRAIN_E2
#      (stop serving, receive the two-pass stream on 52414-52416 into $D/stream_e2, train, export adapter_rX32R1E2,
#      serve it as x32e2), REBASE_P (serve the base, greedy, from $SRC as x32base for P's base rerun), STOP_SERVING
#      (stop and end; it also ends every earlier wait).  Sets started here live 24 h (P needs >= 12 h).  A signal
#      counts as handled ($D/handled.<signal>) only once its set serves.
# A restart resumes: once adapter_rX32R1 is exported, stages 3-5 are skipped and the set named in $D/SERVING is
# adopted while it serves (served again otherwise).
# Lessons kept from c1_910b.sh / exp31 r0_910b.sh / exp30 probe1_910b.sh: idle = no ESTABLISHED connection on the
# model/VO ports; settled() before stopping a set; busy cards through npu_hbm_used_mib.awk (no number = busy);
# PYTHONPATH with $R/rpc/src; one TMPDIR per process under $R/tmp.
#   EXP32_DRY=1 bash scripts/exp32/exp32_910b.sh          print the commands, walk every branch, write nothing
#   bash scripts/exp32/exp32_910b.sh life [PORT]          seconds the listener on PORT (52417) has left
#   bash scripts/exp32/exp32_910b.sh smoke-train STREAM   2 training steps on NPU ${EXP32_SMOKE_NPUS:-7}, then the checks
set -uo pipefail
R="${EXP32_ROOT:-$HOME/work/zhr/zhr_1}"; E=$R/exp23; D=$R/exp32; SIG=$D/signals; IN=$D/parity_in
X=$E/round1/export; SRC="${EXP32_SRC:-$E/src_x32}"; P="${EXP32_PYTHON:-$R/envs/ppa/bin/python}"
STAG=rfts; SRUN="$R/exp28/serve_$STAG"         # pass 1's sampled set (serve_910b.sh's default run dir)
SAMPLE_S=36000; PASS3_S=30600; P_SERVE_S=86400  # sampled sets live 10 h; pass 3 needs 8.5 h more; P's sets 24 h
P2_HOLD_S=43200                                 # pass 2's set ends 12 h after stage 1 unless a signal came first
DRY="${EXP32_DRY:-0}"; LOGF=$D/exp32.log; STATF=$D/STATUS; MAIN=1  # MAIN=0: smoke-train leaves receivers and $SIG alone
export PYTHONPATH="$SRC:$R/rpc/src" TMPDIR="$R/tmp/x32/main"
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
log() { if (( DRY )); then say "$*"; else say "$*" >> "$LOGF"; say "$*" > "$STATF"; fi; }
q() {  # the arguments as a command line, quoting only those that need it
  local a o=""
  for a in "$@"; do if [[ -n "$a" && "$a" =~ ^[A-Za-z0-9_./=:,@%+-]+$ ]]; then o+="$a "; else o+="$(printf '%q' "$a") "; fi; done
  printf '%s' "${o% }"
}
run() { if (( DRY )); then echo "[dry] $(q "$@")"; else "$@"; fi; }
runlog() {  # $1 log file, then a command run in $SRC
  local out=$1; shift
  if (( DRY )); then echo "[dry] (cd $SRC && $(q "$@")) > $out 2>&1"; else ( cd "$SRC" && "$@" ) > "$out" 2>&1; fi
}
bg() {  # $1 log file, then a command started detached in $SRC
  local out=$1; shift
  if (( DRY )); then echo "[dry] (cd $SRC && nohup setsid $(q "$@") > $out 2>&1 < /dev/null &)"; return 0; fi
  ( cd "$SRC" && nohup setsid "$@" > "$out" 2>&1 < /dev/null & )
}
stop_recv() {  # $1 a destination ('' = every receiver of ours under $D, the signal receiver included)
  local p
  (( MAIN )) || return 0
  for p in $(pgrep -u "$(id -u)" -f "[s]tream_transfer[.]py recv --dest ${1:-$D/}${1:+ --ports}"); do run kill -TERM "$p"; done
}
finish() { log "DONE: $*"; stop_recv ""; exit 0; }
die() { log "STOPPED: $*"; (( DRY )) && return 0; stop_recv ""; exit 2; }   # dry: report and walk on
await() {  # $1 seconds, $2 what, then a test polled every 30 s; STOP_SERVING ends the run, the deadline stops it
  local end=$(( $(date +%s) + $1 )) limit=$1 what=$2; shift 2
  if (( DRY )); then echo "[dry] wait (<= $(( limit / 60 )) min) for $what"; return 0; fi
  log "waiting for $what"
  until "$@"; do
    (( MAIN )) && [[ -e "$SIG/STOP_SERVING" ]] && finish "STOP_SERVING while waiting for $what"
    (( $(date +%s) < end )) || die "gave up after $(( limit / 60 )) min waiting for $what"
    sleep 30
  done
}
listening() { local p; for p in "$@"; do ss -Hltn "sport = :$p" | grep -q . || return 1; done; }
ports_free() { local p; for p in "$@"; do ss -Hltn "sport = :$p" | grep -q . && return 1; done; return 0; }
connected() { ss -Htn state established '( sport >= :52410 and sport <= :52416 ) or ( sport >= :52510 and sport <= :52516 )' | grep -q .; }
no_client() { ! connected; }
busy_cards() {  # $@ card ids: how many hold over 4096 MiB HBM, or 99 when the table cannot be read
  local t c=0 id u; t="$(npu-smi info 2>/dev/null)"
  for id in "$@"; do
    u="$(printf '%s\n' "$t" | awk -v id="$id" -f "$SRC/scripts/ascend/npu_hbm_used_mib.awk")"
    [[ "$u" =~ ^[0-9]+$ ]] || { echo 99; return; }
    (( u > 4096 )) && c=$((c + 1))
  done
  echo "$c"
}
cards_idle() { [[ "$(busy_cards ${1//,/ })" -eq 0 ]]; }  # $1 card ids, comma-separated
port_pid() {  # $1 port: the pid of our listener on it ('' = none, or not ours)
  local p; p=$(ss -Hltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [[ -n "$p" ]] || p=$(pgrep -u "$(id -u)" -f "[s]erve_dirs[.]py .*$1" | head -1)
  echo "$p"
}
left_s() {  # $1 pid: seconds left under its nearest `timeout` (itself or an ancestor); inf under none; '' unknown
  local pid=$1 a i dur s el
  while [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )); do
    if [[ "$(cat "/proc/$pid/comm" 2>/dev/null)" == timeout ]]; then
      mapfile -t a < <(tr '\0' '\n' < "/proc/$pid/cmdline"); dur=""
      for ((i = 1; i < ${#a[@]}; i++)); do
        case "${a[i]}" in -k|-s|--kill-after|--signal) i=$((i + 1)) ;; -*) ;; *) dur=${a[i]}; break ;; esac
      done
      s=$(awk -v d="$dur" 'BEGIN { if (d !~ /^[0-9]+(\.[0-9]+)?[smhd]?$/) exit 1; u = substr(d, length(d))
        m = (u == "m") ? 60 : (u == "h") ? 3600 : (u == "d") ? 86400 : 1; printf "%d", d * m }') || { echo ""; return; }
      el=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
      [[ "$el" =~ ^[0-9]+$ ]] || { echo ""; return; }
      (( s == 0 )) && echo inf || echo $(( s - el )); return
    fi
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  done
  [[ "$pid" == 1 || "$pid" == 0 ]] && echo inf || echo ""
}
hours() { [[ "$1" == inf ]] && echo "no time limit" || echo "$(( $1 / 3600 )) h $(( $1 % 3600 / 60 )) min"; }
rundir() { if [[ "$1" == "$STAG" ]]; then echo "$SRUN"; else echo "$D/serve_$1"; fi; }
settled() {  # $1 tag: wait (<= 2 h) until that set is serving or has ended, so a stop finds its launcher
  local rd; rd=$(rundir "$1"); (( DRY )) && return 0
  for _ in $(seq 240); do grep -q -E "serving|STOPPED|ENDED" "$rd/STATUS" 2>/dev/null && return 0; sleep 30; done
  log "the $1 servers never settled ($rd/STATUS: $(cat "$rd/STATUS" 2>/dev/null))"
}
stop_servers() {  # $1 tag: TERM its launcher, then wait (<= 30 min) for its ports, cards 0-6 and its serve_910b.sh
  local p busy rd; rd=$(rundir "$1")
  if (( DRY )); then echo "[dry] TERM run_ppa_servers_npu.sh exp28$1_*; wait for 52410-6/52510-6, cards 0-6, $rd/STATUS ENDED"; return 0; fi
  for p in $(pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh exp28$1_"); do kill -TERM "$p" && log "TERM server launcher $p ($1)"; done
  for _ in $(seq 90); do  # the old serve_910b.sh withdraws its publication within 5 min; a new set of this tag waits for that
    busy=0; ports_free 5241{0..6} 5251{0..6} || busy=1
    grep -q -E "serving|starting" "$rd/STATUS" 2>/dev/null && busy=1
    [[ "$busy" -eq 0 && "$(busy_cards 0 1 2 3 4 5 6)" -eq 0 ]] && return 0; sleep 20
  done
  die "the $1 servers did not release their ports, cards or STATUS in 30 min"
}
serve() {  # $1 tag, $2 adapter dir ('' = the base), $3 seconds, $4 'sample' = System2 sampling (T 0.7, top_p 0.95)
  local rd v=(); rd=$(rundir "$1")
  v=(EXP23_ROOT="$R" EXP28_TAG="$1" EXP28_SRC="$SRC" EXP28_RUN_DIR="$rd" EXP28_ADAPTER="$2" EXP28_SERVE_S="$3" TMPDIR="$R/tmp/x32/serve_$1")
  [[ "${4:-}" == sample ]] && v+=(PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE=0.7 PPA_EVAL_SYSTEM2_SAMPLE_TOP_P=0.95)
  run rm -f "$rd/STATUS"; run mkdir -p "$R/tmp/x32/serve_$1"
  bg "$D/serve_$1.out" env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P "${v[@]}" bash scripts/exp28/serve_910b.sh
  log "starting the $1 servers (${2:-the base}, ${4:-greedy}, $(hours "$3"))"
}
set_up() {  # $1 tag: wait (<= 3 h) until that set serves; stop the run if it stopped instead
  local st="" rd; rd=$(rundir "$1")
  (( DRY )) && { echo "[dry] wait (<= 180 min) for $rd/STATUS: serving"; return 0; }
  for _ in $(seq 360); do
    st=$(cat "$rd/STATUS" 2>/dev/null)
    [[ "$st" == *"serving:"* ]] && { log "the $1 servers serve (${st#* serving: })"; return 0; }
    [[ "$st" == *STOPPED* || "$st" == *ENDED* ]] && die "the $1 servers did not come up: $st"
    sleep 30
  done
  die "the $1 servers are not serving after 3 h ($rd/STATUS: $st)"
}
sampling_ok() {  # $1 seconds the sampled set must still live: why it cannot be kept ('' = keep it)
  local pid left commit pub="$X/npu_servers_exp28_$STAG/servers.json"
  grep -q "serving:" "$SRUN/STATUS" 2>/dev/null || { echo "STATUS is not serving: $(cat "$SRUN/STATUS" 2>/dev/null)"; return; }
  pid=$(pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh exp28${STAG}_" | head -1)
  [[ -n "$pid" ]] || { echo "no launcher runs"; return; }
  left=$(left_s "$pid")
  [[ -n "$left" ]] || { echo "the launcher's time limit cannot be read"; return; }
  [[ "$left" == inf ]] || (( left >= $1 )) || { echo "$(hours "$left") left, $(hours "$1") needed"; return; }
  commit=$("$P" -c 'import json,sys; d=json.load(open(sys.argv[1]))
print("" if d.get("repo_dirty") or d.get("system2_lora_sha256") else d["repo_commit"])' "$pub" 2>/dev/null)
  [[ -n "$commit" ]] || { echo "$pub is missing, from a dirty checkout or names an adapter"; return; }
  git -C "$SRC" show "$commit:scripts/evaluation/rpc_model_server.py" 2>/dev/null | grep -q "top_k=0" \
    || { echo "its commit ${commit:0:7} lacks the top_k fix (or is not in $SRC)"; return; }
  # the 4090's pass-2/3 clients run wt_x32 and refuse servers of another commit (run_s2_dagger_collect_cuda.sh)
  [[ "$commit" == "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" ]] \
    || { echo "its commit ${commit:0:7} is not $SRC's HEAD, which the 4090's x32 clients run"; return; }
  awk '/ tag=/ {b = ""} {b = b $0 "\n"} END {printf "%s", b}' "$SRUN/serve.log" 2>/dev/null \
    | grep -q "System2 SAMPLES (T=0.7 top_p=0.95)" || { echo "its serve.log does not say System2 SAMPLES (T=0.7 top_p=0.95)"; return; }
  log "keeping the sampled set $STAG (commit ${commit:0:7}, $(hours "$left") left)" >&2
}
ensure_sampling() {  # $1 seconds needed: keep the sampled set, or restart it under its tag (no client may be on it)
  local why; why=$(sampling_ok "$1")
  [[ -z "$why" ]] && (( ! DRY )) && return 0
  log "the sampled set $STAG cannot be kept: ${why:-(dry run: shown anyway)}"
  (( ! DRY )) && connected && log "clients are on the $STAG servers (a collection still runs): replaced once they leave"
  await 28800 "the $STAG servers to have no client before they are replaced" no_client  # a 7 h pass and a margin
  run rm -rf "$X/exp32_sampling_p2" "$X/exp32_sampling_p3"  # no export names a set while it is replaced
  stop_servers "$STAG"
  serve "$STAG" "" "$SAMPLE_S" sample
  set_up "$STAG"
}
ready_export() {  # $1 p2|p3: export exp32_sampling_$1 (the set's servers.json and ready.txt) on 52417 -- the 4090
                  # starts that pass's external arm only once it has pulled it, so it never meets a set being replaced
  local pub="$X/npu_servers_exp28_$STAG" out="$X/exp32_sampling_$1"
  if (( DRY )); then echo "[dry] export $out: $pub/servers.json + ready.txt (tag $STAG, its STATUS)"; return 0; fi
  rm -rf "$out.part"; mkdir -p "$out.part"
  cp "$pub/servers.json" "$out.part/" || die "no $pub/servers.json to export as exp32_sampling_$1"
  say "$STAG $(cat "$SRUN/STATUS" 2>/dev/null)" > "$out.part/ready.txt"
  rm -rf "$out"; mv "$out.part" "$out"
  log "exported exp32_sampling_$1: the $STAG set serves pass ${1#p}"
}
serving() {  # $1 tag: that set serves (its STATUS says so and its launcher runs)
  grep -q "serving:" "$(rundir "$1")/STATUS" 2>/dev/null && pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh exp28$1_" > /dev/null
}
recv() {  # $1 destination, $2 ports, $3 cap GB, $4 name: a receiver of ours with its own TMPDIR (kept if it runs)
  pgrep -u "$(id -u)" -f "[s]tream_transfer[.]py recv --dest $1 --ports" > /dev/null && return 0
  ports_free ${2//,/ } || die "a port of $2 is held: $(ss -Hltnp "sport = :${2%%,*}" 2>/dev/null)"
  run mkdir -p "$1" "$R/tmp/x32/$4"
  bg "$D/$4.log" env TMPDIR="$R/tmp/x32/$4" "$P" scripts/tools/stream_transfer.py recv --dest "$1" --ports "$2" --cap-gb "$3"
  await 60 "the $4 receiver on $2" listening ${2//,/ }
}
judge() {  # $1 g0|d3, $2 parity output dir: "PASS ..." or "FAIL ..." (G0: the trainer's exact >= 0.98; D3: 1 - exact >= 0.10)
  "$P" - "$1" "$2" <<'PY' 2>/dev/null || echo "FAIL: no readable parity report in $2"
import json, sys
from fractions import Fraction
kind, d = sys.argv[1], sys.argv[2]
rep = json.load(open(f"{d}/parity_g0.json"))
rows = [json.loads(line) for line in open(f"{d}/parity_g0.calls.jsonl") if line.strip()]
n, exact = len(rows), sum(bool(r.get("exact")) for r in rows)
bad = [f"split {rep.get('split')}"] if rep.get("split") != "all" else []
if rep.get("record_device") != "npu" or rep.get("device") != "npu":
    bad.append(f"records {rep.get('record_device')} on {rep.get('device')}")
if rep.get("adapter"):
    bad.append(f"adapter {rep['adapter']}")
if not n or n != rep.get("n"):
    bad.append(f"{n} rows, the report says {rep.get('n')}")
if kind == "g0":
    ok, what = bool(rep.get("g0_pass")), f"G0-npu exact {exact}/{n} = {rep.get('exact_rate')} (line {rep.get('g0_threshold')})"
else:
    g = Fraction(n - exact, n) if n else Fraction(0)
    ok, what = g >= Fraction(1, 10), f"D3 G-samp 1 - {exact}/{n} = {float(g):.4f} (line 0.10)"
print(("PASS " if ok and not bad else "FAIL ") + what + "".join(f"; {b}" for b in bad))
PY
}
parity() {  # $1 g0|d3, $2 record roots (colon-separated), $3 extra trainer args: replay 2,000 calls on 8 cards
  local out="$D/$1_npu" verdict
  await 1800 "cards 0-7 under 4096 MiB HBM" cards_idle 0,1,2,3,4,5,6,7
  runlog "$out.log" env S2_LORA_ROOT="$R" S2_LORA_MODE=parity S2_LORA_RECORD_ROOTS="$2" S2_LORA_OUTPUT_DIR="$out" \
    S2_LORA_PARITY_N=2000 S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 S2_LORA_STREAM_DIR= S2_LORA_LABEL_ROOTS= S2_LORA_ADAPTER= \
    S2_LORA_EXTRA_ARGS="$3" bash scripts/ascend/run_s2_lora_train_npu.sh
  run rm -rf "$X/exp32_$1_npu.part"; run mkdir -p "$X/exp32_$1_npu.part"
  run cp "$out/parity_g0.json" "$out/parity_g0.calls.jsonl" "$out.log" "$X/exp32_$1_npu.part/"
  run rm -rf "$X/exp32_$1_npu"; run mv "$X/exp32_$1_npu.part" "$X/exp32_$1_npu"
  (( DRY )) && { echo "[dry] judge $1 from $out/parity_g0.json + .calls.jsonl; anything but PASS stops here"; return 0; }
  verdict=$(judge "$1" "$out"); log "$verdict (exported as exp32_$1_npu)"
  [[ "$verdict" == PASS* ]] || die "$1 parity did not pass: $verdict ($out.log)"
}
train() {  # $1 stream, $2 output dir[, $3 NPUs, $4 extra args]: one pass from the base, ledger E (c1_910b.sh's command)
  local hold
  hold=$("$P" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["holdout"]["scenes"]))' "$1/stream_manifest.json" 2>/dev/null)
  (( DRY )) && hold=${hold:-HOLDOUT_FROM_MANIFEST}
  [[ -n "$hold" ]] || { die "$1 names no held-out scenes"; return 1; }
  await 1800 "cards ${3:-0,1,2,3,4,5,6,7} under 4096 MiB HBM" cards_idle "${3:-0,1,2,3,4,5,6,7}"
  log "training $(basename "$2") from $(basename "$1") (token mean, from the base; holdout $(tr ',' '\n' <<< "$hold" | wc -l) scenes)"
  runlog "$2.train.log" env S2_LORA_ROOT="$R" S2_LORA_MODE=train S2_LORA_STREAM_DIR="$1" S2_LORA_OUTPUT_DIR="$2" \
    S2_LORA_NPU_DEVICES="${3:-0,1,2,3,4,5,6,7}" S2_LORA_RECORD_ROOTS= S2_LORA_LABEL_ROOTS= S2_LORA_ADAPTER= \
    S2_LORA_EXTRA_ARGS="--global-batch 64 --save-every 100 --num-workers 4 --log-every 10 --loss-normalization token --holdout-scenes $hold${4:+ $4}" \
    bash scripts/ascend/run_s2_lora_train_npu.sh
}
checks() {  # $1 output dir, $2 stream[, $3 steps asked for]: 'ok', or why the adapter may not be exported (ledger E)
  "$P" - "$@" <<'PY' 2>/dev/null || echo "no readable train_log.jsonl / manifest.json / stream_manifest.json"
import json, sys
out, stream = sys.argv[1], sys.argv[2]
events = [json.loads(line) for line in open(f"{out}/train_log.jsonl") if line.strip()]
man = json.load(open(f"{out}/manifest.json"))
total = int(json.load(open(f"{stream}/stream_manifest.json"))["total_samples"])
done = [e for e in events if e.get("event") == "done"]
bad = [] if done else ["no done event in train_log.jsonl"]
norm = man.get("hyperparameters", {}).get("loss_normalization", "sample (the default)")
if norm != "token":
    bad.append(f"manifest.json: loss_normalization {norm}")
want = total // 64 if len(sys.argv) < 4 else int(sys.argv[3])
if total % 64:
    bad.append(f"total_samples {total} is not a multiple of 64")
rows = [int(e["step"]) for e in events if e.get("event") == "step"]
got = {"done": done[-1].get("steps") if done else None, "manifest": man.get("training", {}).get("steps"),
       "last step row": max(rows) if rows else None}
if any(v != want for v in got.values()):
    bad.append(f"steps {got}, wanted {want} (total_samples {total} / 64)")
print("; ".join(bad) or "ok")
PY
}
train_export() {  # $1 stream, $2 output dir, $3 export name, $4 seconds to wait for the stream's manifest
  if [[ -d "$X/$3" ]]; then log "$3 is already exported; not training again"; return 0; fi
  await "$4" "the stream ($1/stream_manifest.json)" test -e "$1/stream_manifest.json"
  train "$1" "$2" || die "training $3 failed ($2.train.log)"
  local why; why=$(checks "$2" "$1"); (( DRY )) && why=ok
  [[ "$why" == ok ]] || die "$3 not exported: $why"
  run rm -rf "$X/$3.part"; run mkdir -p "$X/$3.part"; run cp "$2/adapter/"* "$X/$3.part/"; run mv "$X/$3.part" "$X/$3"
  (( DRY )) || log "exported $X/$3 ($(sha256sum "$X/$3/adapter_model.safetensors" | cut -c1-16))"
}
serve_p() {  # $1 tag, $2 adapter dir ('' = the base): replace the set that serves by this one, greedy, for P (a set of
             # this tag that already serves is kept: a restart that redoes a signal)
  if [[ "${TAG:-}" == "$1" ]] && (( ! DRY )) && serving "$1"; then log "the $1 servers serve already: kept"; return 0; fi
  [[ -n "${TAG:-}" ]] && { settled "$TAG"; stop_servers "$TAG"; }
  serve "$1" "$2" "$P_SERVE_S"; TAG=$1; (( DRY )) || echo "$1 $2" > "$D/SERVING"
  set_up "$1"
}
on_signal() {  # $1 SCALE05 | RETRAIN_E2 | REBASE_P; handled.$1 only once its set serves (a restart redoes it otherwise)
  log "signal $1"
  case "$1" in
    SCALE05)
      [[ -s "$SIG/adapter_s05/adapter_model.safetensors" && -s "$SIG/adapter_s05/adapter_config.json" ]] || (( DRY )) \
        || die "SCALE05 without the scaled adapter in $SIG/adapter_s05"
      serve_p x32s05 "$SIG/adapter_s05" ;;
    RETRAIN_E2)
      if [[ "${TAG:-}" != x32e2 ]] || (( DRY )) || ! serving x32e2; then  # (a restart may find x32e2 serving already)
        [[ -n "${TAG:-}" ]] && { settled "$TAG"; stop_servers "$TAG"; }; TAG=""
        recv "$D/stream_e2" 52414,52415,52416 60 recv_stream_e2
        train_export "$D/stream_e2" "$D/lora_e2" adapter_rX32R1E2 43200  # (not again once exported)
        stop_recv "$D/stream_e2"; await 120 "ports 52414-52416 to close" ports_free 52414 52415 52416
      fi
      serve_p x32e2 "$X/adapter_rX32R1E2" ;;
    REBASE_P)
      serve_p x32base "" ;;
  esac
  run touch "$D/handled.$1"
}
next_signal() {  # the oldest signal in $SIG not handled yet
  local f
  for f in $(ls -tr "$SIG" 2>/dev/null); do
    case "$f" in SCALE05|RETRAIN_E2|REBASE_P) [[ -e "$D/handled.$f" ]] || { echo "$f"; return; } ;; esac
  done
}

case "${1:-run}" in
  life) pid=$(port_pid "${2:-52417}"); echo "port ${2:-52417}: pid ${pid:-none}, left $([[ -n "$pid" ]] && left_s "$pid")"; exit 0 ;;
  smoke-train)
    [[ -s "${2:-}/stream_manifest.json" ]] || { echo "usage: $0 smoke-train STREAM_DIR (with a stream_manifest.json)"; exit 64; }
    out="$D/smoke_train_$(date -u +%Y%m%d_%H%M%S)"; LOGF=$out.log; STATF=/dev/null; MAIN=0
    (( DRY )) || mkdir -p "$D" "$R/tmp/x32/main"
    train "$2" "$out" "${EXP32_SMOKE_NPUS:-7}" "--max-steps 2 --save-every 0 --log-every 1"; rc=$?
    echo "smoke-train: exit $rc, checks: $(checks "$out" "$2" 2) ($out)"; exit 0 ;;
  run) ;;
  *) echo "usage: $0 [run | life [PORT] | smoke-train STREAM_DIR]" >&2; exit 64 ;;
esac

# 0 --------------------------------------------------------------------------------------------------------------
(( DRY )) || mkdir -p "$D" "$SIG" "$R/tmp/x32/main"
log "EXP-32 910B controller (src $SRC $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null))"
[[ -x "$P" ]] || die "no python at $P"
grep -q "top_k=0" "$SRC/scripts/evaluation/rpc_model_server.py" 2>/dev/null || die "$SRC lacks the top_k fix"
pid=$(port_pid 52417); left=$([[ -n "$pid" ]] && left_s "$pid")
if [[ -z "$pid" ]]; then die "nothing of ours serves the export port 52417"
elif [[ -z "$left" ]]; then die "cannot tell how long the export port 52417 lives (pid $pid)"
elif [[ "$left" != inf ]] && (( left < 604800 )); then
  die "the export port 52417 has $(hours "$left") left, under 7 days: restart serve_dirs with a longer timeout first"
else log "export port 52417: $(hours "$left") left"; fi
# The signal channel takes 52517 (c1_910b.sh's): our TensorBoard and earlier rounds' signal receivers give it up.
if ! pgrep -u "$(id -u)" -f "[s]tream_transfer[.]py recv --dest $SIG --ports 52517" > /dev/null; then
  for p in $(pgrep -u "$(id -u)" -f "[t]b_tool/tb_serve[.]py|[s]tream_transfer[.]py recv .*--ports 52517"); do
    log "TERM $p on 52517 ($(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-120))"; run kill -TERM "$p"
  done
  (( DRY )) || for _ in $(seq 30); do ports_free 52517 && break; sleep 2; done
  recv "$SIG" 52517 4 recv_signals
fi

# 1-2 ------------------------------------------------------------------------------------------------------------
if [[ -e "$D/sampling.done" ]]; then log "the sampled collection is over ($(cat "$D/sampling.done"))"
else
  pass=p2; need=$SAMPLE_S; [[ -e "$D/handled.PASS3" ]] && { pass=p3; need=$PASS3_S; }
  if [[ -d "$X/exp32_sampling_$pass" ]] && (( ! DRY )); then  # a restart: the pass may be under way on that set
    log "exp32_sampling_$pass was exported before: any $STAG set that still serves is kept"; need=0
  fi
  ensure_sampling "$need"; ready_export "$pass"
  start=$(date +%s); end=$(( start + P2_HOLD_S )); [[ "$pass" == p3 ]] && end=$(( start + PASS3_S ))
  arm=$(( start + 3600 )); seen=0; quiet=0; why=""
  log "the $STAG servers serve pass ${pass#p}; waiting for SAMPLING_DONE$([[ "$pass" == p2 ]] && echo ", PASS3" || echo ", 30 idle min") or $(date -u -d "@$end" +%FT%TZ 2>/dev/null)"
  if (( DRY )); then
    echo "[dry] pass 2: poll every 60 s for STOP_SERVING / SAMPLING_DONE / PASS3 until $(( P2_HOLD_S / 3600 )) h after stage 1 (no idle rule)"
    echo "[dry] --- on PASS3"; ensure_sampling "$PASS3_S"; ready_export p3; run touch "$D/handled.PASS3"
    echo "[dry] pass 3: STOP_SERVING / SAMPLING_DONE / 30 idle min (armed 60 min after PASS3) / $(( PASS3_S / 60 )) min after PASS3"
    why="SAMPLING_DONE"
  fi
  until [[ -n "$why" ]]; do
    now=$(date +%s)
    if [[ -e "$SIG/STOP_SERVING" ]]; then stop_servers "$STAG"; finish "STOP_SERVING during the sampled collection"
    elif [[ -e "$SIG/SAMPLING_DONE" ]]; then why=SAMPLING_DONE  # (sent after pass 3 if there was one: it wins over PASS3)
    elif [[ "$pass" == p2 && -e "$SIG/PASS3" ]]; then
      log "PASS3: pass 3 follows"
      ensure_sampling "$PASS3_S"; ready_export p3; touch "$D/handled.PASS3"
      pass=p3; start=$(date +%s); end=$(( start + PASS3_S )); arm=$(( start + 3600 )); seen=0; quiet=0
      log "the $STAG servers serve pass 3; waiting for SAMPLING_DONE, 30 idle min or $(date -u -d "@$end" +%FT%TZ)"
    elif (( now >= end )); then why="the deadline (pass ${pass#p})"
    else  # the idle rule only in pass 3: between pass 2 and PASS3 the 4090 judges validity and the dose rule
      if [[ "$pass" == p3 ]] && (( now >= arm )); then
        if connected; then seen=1; quiet=0; elif (( seen )); then quiet=$((quiet + 1)); fi
      fi
      if (( quiet >= 30 )); then why="no client for 30 min (pass 3)"; else sleep 60; fi
    fi
  done
  log "stopping the sampled servers ($why)"
  settled "$STAG"; stop_servers "$STAG"
  (( DRY )) || say "$why" > "$D/sampling.done"
fi

# 3-4 ------------------------------------------------------------------------------------------------------------
if [[ -d "$X/adapter_rX32R1" ]] && (( ! DRY )); then  # a restart after training (a P set may hold 52410-52416 now)
  log "adapter_rX32R1 is exported: the parity checks and the training are not rerun"
  stop_recv "$IN"; stop_recv "$D/stream"  # (a run that ended between the export and serving left them)
else
  recv "$IN" 52413 24 recv_parity
  recv "$D/stream" 52414,52415,52416 60 recv_stream
  if [[ "$(judge g0 "$D/g0_npu")" == PASS* && -d "$X/exp32_g0_npu" ]]; then log "G0-npu passed before; not rerun"
  else
    await 28800 "R0-G's G0 records ($IN/G0_PUSHED)" test -e "$IN/G0_PUSHED"
    parity g0 "$IN/records" "--parity-split all"
  fi
  if [[ "$(judge d3 "$D/d3_npu")" == PASS* && -d "$X/exp32_d3_npu" ]]; then log "D3 passed before; not rerun"
  else
    await 28800 "the D3 records ($IN/D3_PUSHED)" test -e "$IN/D3_PUSHED"
    roots=""; for t in p1 r0s p2 p3; do [[ -d "$IN/$t" ]] && roots="${roots:+$roots:}$IN/$t"; done
    (( DRY )) && roots=${roots:-$IN/p1:$IN/r0s:$IN/p2}
    [[ -n "$roots" ]] || die "D3_PUSHED arrived without records under $IN/{p1,r0s,p2,p3}"
    parity d3 "$roots" "--parity-split all --parity-device npu"
  fi
  train_export "$D/stream" "$D/lora" adapter_rX32R1 28800
  stop_recv "$IN"; stop_recv "$D/stream"
fi

# 5 --------------------------------------------------------------------------------------------------------------
[[ -e "$SIG/STOP_SERVING" ]] && finish "STOP_SERVING before the candidate was served"
TAG=""; last=""; (( DRY )) || last=$(cat "$D/SERVING" 2>/dev/null)  # "<tag> <adapter>" of the last P set started
if [[ -n "$last" ]] && serving "${last%% *}"; then
  TAG=${last%% *}; log "the $TAG servers still serve (a restart): adopted"
else
  await 120 "ports 52413-52416 to close" ports_free 52413 52414 52415 52416
  if [[ -n "$last" ]]; then log "serving $last again (a restart; it no longer serves)"; serve_p "${last%% *}" "${last#* }"
  else serve_p x32r1 "$X/adapter_rX32R1"; fi
fi

# 6 --------------------------------------------------------------------------------------------------------------
log "$TAG serves P; listening for the 4090's signals in $SIG"
if (( DRY )); then
  for s in SCALE05 RETRAIN_E2 REBASE_P; do echo "[dry] --- on $s"; on_signal "$s"; done
  echo "[dry] --- on STOP_SERVING"; stop_servers "$TAG"; finish "STOP_SERVING ($TAG stopped)"
fi
listen_end=$(( $(date +%s) + 259200 ))
while :; do
  settled "$TAG"
  [[ -e "$SIG/STOP_SERVING" ]] && { stop_servers "$TAG"; finish "STOP_SERVING ($TAG stopped)"; }
  s=$(next_signal); [[ -n "$s" ]] && on_signal "$s"
  (( $(date +%s) < listen_end )) || { stop_servers "$TAG"; finish "no STOP_SERVING within 3 days ($TAG stopped)"; }
  sleep 60
done
