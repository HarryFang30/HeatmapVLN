#!/usr/bin/env bash
# EXP-37, 4090 container: X37's val (ledger EXP-37 设置·val, 判据·主判据 and 中途无效性).  EXP-35's
# round2_val9_4090.sh with the EXP-37 changes; the client is unchanged (/workspace/exp23/wt_x32), since the
# base-decides-STOP mode lives in the servers.
# X37 seed 1337 runs in 9 parts (every shard's episodes cut in cohort order into ninths).  A worker whose
# X37 set is published (tags x37 / x37w1 / x37w2) takes the next part nobody started, one part at a time; a
# part is resumed on the worker that holds it, and handed to a free serving worker when its own cannot take
# it for HANDOFF s (tunnel down, set unpublished).  Every start pulls the worker's publication again (a
# re-served set has a new server instance).  No part starts before the zero-adapter equivalence check reads
# identical ($EQUIV, equiv_check.py compare: B0 seed 1337, its episodes 1/25/41, every one identical, servers
# in base-stop mode of one commit).  The adapter is an allowlist, not the first set published: the one
# candidate gate_verdict.py's verdict ($GATE, copied or written on this box) names with all_pass and next
# "val" (ledger 离线门: 其他任何不过都不上 val; a retried attempt 1's verdict reads retry_lr_5e-5, so only the
# attempt that passed can name it); X37_SHA, if given, must agree, and stands alone only with
# X37_GATE_OVERRIDE="<reason>" (a logged deviation).  It is pinned: every set must name it, every set's
# servers.json must say "system2_base_stop": true with a model_version ending in +base-stop-v1, and every
# set must be built from the commit the equivalence check passed on.  The equivalence check's zero adapter,
# X34M and X35 are never pinned.
# Interim futility, one look: the first time the completed parts (not yet all nine) pair >= 900 episodes with
# B0 seed 1337, the as-run dSR on those episodes is read; at <= -2.0 pt the remaining parts are stopped and the
# verdict is 否定（中途无效性）, otherwise the rule is spent (later growths only check the rows' mode, their
# outcomes are not looked at until all nine are in).  When all 9 parts are complete: one workers dir, the
# locked merge tool over all 1,839 episodes, and the paired bootstrap against B0 seed 1337, as run and
# standard protocol, with the primary criterion's verdict; the mechanism counts are report only.  Every row
# the parts recorded must carry the base-stop model_version (the server's own name, which the client copies
# into rpc_model_version), or the val stops.  Markers in $V make a rerun pick up where it stopped; a part's
# running launcher is adopted from its pid file.
#   mkdir -p /workspace/exp37/auto && cp <repo>/scripts/exp37/val_4090.sh /workspace/exp37/auto/
#   setsid nohup bash /workspace/exp37/auto/val_4090.sh >> /workspace/exp37/auto/val_4090.out 2>&1 < /dev/null &
# Client slots: X37_GPU_DEVICES (default 0,1,2,3,4,5,6,7) is PPA_EVAL_GPU_DEVICES, one slot per entry, slot s
# talking to the worker's model/VO server s.  With external servers the client never initialises CUDA (the
# RPC path of r2r_val_unseen.py has no torch.cuda / .to(device), rendering is llvmpipe), so the ids are only
# CUDA_VISIBLE_DEVICES labels: EXP-35's p6-p9 ran slots 6 and 7 with CUDA ids 6 and 7 on this box after it
# came to show 6 GPUs (nvidia-smi -L: 0-5, 2026-10-28; GPUs 1-3 belong to other users' jobs) and completed.
# Eight labels keep all 8 servers of a worker busy and B0's slot = shard layout; fewer entries leave servers
# idle and run two shards one after the other on a slot.
WT=/workspace/exp23/wt_x32
A=/workspace/exp37/auto
V=$A/val9
NS=/workspace/exp37/npu_servers
PLAN=/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802
B0=/workspace/eval_runs/exp34_B0_seed1337_910b_stepbudget
RUN=/workspace/eval_runs/exp37_X37_seed1337_910b_stepbudget   # _p1 .. _p9 parts, _merged
EQUIV=${X37_EQUIV_JSON:-/workspace/exp37/equiv/equiv.json}   # equiv_check.py command's compare line writes here
EQUIV_EPISODES="1 25 41"   # B0's first three of shard 0 (equiv_check.py EPISODES)
GATE=${X37_GATE_VERDICT:-/workspace/exp37/gate/verdict.json}  # gate_verdict.py --output (it runs on this box)
GATE_SCHEMA=heatmapvln-exp37-offline-gate-verdict-v1
VS_PY=/workspace/exp35/auto/val_split3.py   # EXP-35's split / count / combine / verdict, used as it ran there
VS_SHA=c119326418d24513ea628871f696b41dda3a8966d66e142903e44af039d4f46a
MODE_SUFFIX=+base-stop-v1
# Adapters that are not X37 and are never pinned: X34M, X35, the equivalence check's zero adapter (read from
# $EQUIV in step 1), X37_ZERO_SHA and every sha256 in $A/not_x37.sha
NOT_X37=(9f09d1d04af8936684655095a3841dc57c702abfa841a9ff476e6285c255a970
         1d21d6f0aac807356dee796b351bf4650eb73b5519ca3eea98a35101090d5549 ${X37_ZERO_SHA:-})
GPUS=${X37_GPU_DEVICES:-0,1,2,3,4,5,6,7}
FUTILITY_N=900      # interim futility: at least this many paired episodes in the completed parts ...
FUTILITY_PT=-2.0    # ... and their as-run dSR at or below this (pt), the coded test is 50*S <= -n, S = sum of dSR in episodes
RETRIES=1000
TICK=300            # seconds between passes
LATE=21600          # ALARM when a worker has not served X37 this long after the adapter was adopted
HANG=1800           # a completed part's launcher still alive this long is stopped
STALL=3600          # ALARM when a part has made no progress this long
HANDOFF=7200        # a part whose worker cannot take it this long goes to a free serving worker
WORKERS=(w0 w1 w2)  # the order a free part is offered in
PARTS=(p1 p2 p3 p4 p5 p6 p7 p8 p9)
declare -A TAG=([w0]=x37 [w1]=x37w1 [w2]=x37w2) MB=([w0]=52410 [w1]=52420 [w2]=52430) \
           VB=([w0]=52510 [w1]=52520 [w2]=52530) DISP=([w0]=420 [w1]=380 [w2]=400) \
           PID=() RESUMES=() LATE_SAID=() DONE_AT=()
mkdir -p "$V" "$NS"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$V/val_4090.log"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$V/STATUS"; log "STATUS $*"; }
summary() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$A/SUMMARY.txt"; log "SUMMARY $*"; }
# One controller at a time, by its pid (a lock fd would be inherited by its children, even a sleep,
# and outlive a killed controller); a rerun after the controller died adopts the running parts.
old=$(cat "$V/controller.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "val_4090.sh"; then
  log "refused: val_4090.sh $old runs"; exit 1
fi
echo $$ > "$V/controller.pid"
if [[ -e $V/futility.done ]]; then status "X37 val stopped by the interim futility rule (see $V/verdict_vs_b0.txt); nothing to do"; exit 0; fi
if [[ -e $V/paired.done ]]; then status "X37 val done: paired results in $V (verdict_vs_b0.txt)"; exit 0; fi
[[ "$(sha256sum "$VS_PY" 2>/dev/null | cut -d' ' -f1)" == "$VS_SHA" ]] \
  || { summary "ALARM: $VS_PY is not the val_split3.py EXP-35 ran (sha256 $VS_SHA)"; status "STOPPED: val_split3.py changed"; exit 2; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if ! [[ $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] || (( ${#GPU_LIST[@]} > 8 )) \
   || (( $(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l) != ${#GPU_LIST[@]} )); then
  summary "ALARM: X37_GPU_DEVICES=$GPUS is not 1-8 distinct ids"; status "STOPPED: bad X37_GPU_DEVICES"; exit 2
fi
VS() { /opt/conda/bin/python "$VS_PY" "$@"; }
rows() { [[ -s "$1/merged/progress.jsonl" ]] && wc -l < "$1/merged/progress.jsonl" || echo 0; }
part_done() {  # $1 part
  local got listed; read -r got listed < <(VS count "${RUN}_$1" "$V/lists_$1")
  [[ "$got" =~ ^[0-9]+$ && "$listed" =~ ^[0-9]+$ ]] && (( listed > 0 && got >= listed ))
}
not_x37() {  # $1 sha256: one of the adapters that is not X37
  local s
  for s in "${NOT_X37[@]}"; do [[ $1 == "$s" ]] && return 0; done
  grep -qxF -- "$1" "$A/not_x37.sha" 2>/dev/null
}
pub_info() {  # $1 publication name $2 dir: pull it into $2 and print its adapter sha256 ('-' for none), its
              # mode (base-stop: system2_base_stop true and model_version ending in $MODE_SUFFIX; else plain)
              # and its repo_commit
  /opt/conda/bin/python "$WT/scripts/exp23/pull_dirs.py" "$2" 52418 "$1" >/dev/null 2>&1 \
    && [[ -s "$2/$1/servers.json" ]] \
    && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
mode='base-stop' if d.get('system2_base_stop') is True and str(d.get('model_version','')).endswith(sys.argv[2]) else 'plain'
print(d.get('system2_lora_sha256') or '-', mode, d.get('repo_commit') or '-')" "$2/$1/servers.json" "$MODE_SUFFIX"
}
refresh_pub() {  # $1 worker: 0 when its publication names $SHA in base-stop mode at $EQUIV_COMMIT (the local
                 # copy replaced if it changed), 1 when nothing is published, 2 when it names another adapter,
                 # mode or commit.  Only while the worker has no live launcher (a launcher copies servers.json
                 # when it starts).
  local name=npu_servers_exp28_${TAG[$1]} tmp sha mode commit why=""
  tmp=$(mktemp -d "$NS/.probe.XXXXXX")
  if ! read -r sha mode commit < <(pub_info "$name" "$tmp"); then rm -rf "$tmp"; return 1; fi
  if [[ $sha != "$SHA" ]]; then
    log "$name is published with adapter $sha, not $SHA; not starting on $1"; rm -rf "$tmp"; return 2
  fi
  [[ $mode == base-stop ]] || why="is not in base-stop mode (servers.json: system2_base_stop true, model_version ...$MODE_SUFFIX)"
  [[ -z $why && -n $EQUIV_COMMIT && $commit != "$EQUIV_COMMIT" ]] && why="is built from $commit, not $EQUIV_COMMIT (the equivalence check's)"
  if [[ -n $why ]]; then
    [[ "$(cat "$V/badpub_$1" 2>/dev/null)" == "$why" ]] || { summary "ALARM: $name $why; not starting on $1"; echo "$why" > "$V/badpub_$1"; }
    rm -rf "$tmp"; return 2
  fi
  rm -f "$V/badpub_$1"
  if ! cmp -s "$tmp/$name/servers.json" "$NS/$name/servers.json" 2>/dev/null; then
    rm -rf "${NS:?}/$name"; mv "$tmp/$name" "$NS/"
    summary "$name published (adapter $sha, base-stop, commit $commit, server instance $(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('server_instance') or '-')" "$NS/$name/servers.json"))"
  fi
  rm -rf "$tmp"; return 0
}
COMMON=(env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P
        PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO=$WT PPA_EVAL_EXTERNAL_SERVERS=1
        PPA_EVAL_GPU_DEVICES=$GPUS PPA_EVAL_ALLOW_COMMIT_MISMATCH=1)
start_part() {  # $1 part $2 worker: EXP-35's part launch, on this part's lists.
  # The subshell execs setsid, which (the subshell leads no process group) execs env and then the
  # launcher in place, so $! is the launcher itself, in a session of its own.
  local p=$1 w=$2
  stat -c %s "$V/val_$p.out" 2>/dev/null > "$V/logoff_$p" || echo 0 > "$V/logoff_$p"  # server_trouble reads past it
  ( cd "$WT" && exec setsid "${COMMON[@]}" PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_EXTERNAL_SERVER_DIR=$NS/npu_servers_exp28_${TAG[$w]} \
      PPA_EVAL_OUTPUT_ROOT=${RUN}_$p PPA_EVAL_MODEL_PORT_BASE=${MB[$w]} PPA_EVAL_VO_PORT_BASE=${VB[$w]} PPA_EVAL_PROTOCOL_SEED=1337 \
      PPA_EVAL_STOP_AT_STEP_BUDGET=1 PPA_EVAL_DISPLAY_BASE=${DISP[$w]} PPA_EVAL_SHARD_RETRIES=$RETRIES \
      PPA_EVAL_SYSTEM2_LORA="$SHA" PPA_EVAL_EPISODE_LISTS_DIR=$V/lists_$p \
      bash scripts/run_ppa_r2r_val_unseen_cuda.sh >> "$V/val_$p.out" 2>&1 < /dev/null ) &
  PID[$p]=$!; echo "${PID[$p]}" > "$V/pid_$p"
  log "part $p started on $w (pid ${PID[$p]}, servers ${TAG[$w]}, $(VS count "${RUN}_$p" "$V/lists_$p") recorded/listed)"
}
is_launcher() {  # $1 pid $2 part: a val launcher (or one of its slot subshells) writing this part's run
  [[ -r /proc/$1/cmdline ]] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q "run_ppa_r2r_val_unseen_cuda.sh" \
    && grep -qzx -- "PPA_EVAL_OUTPUT_ROOT=${RUN}_$2" "/proc/$1/environ" 2>/dev/null
}
alive() {  # $1 part: the launcher this or an earlier controller started, or any launcher of this part
  local pid=${PID[$1]:-} q
  [[ -z $pid && -s $V/pid_$1 ]] && pid=$(cat "$V/pid_$1")
  [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && is_launcher "$pid" "$1" && return 0
  for q in $(pgrep -f "[r]un_ppa_r2r_val_unseen_cuda.sh"); do is_launcher "$q" "$1" && return 0; done
  return 1
}
stop_part() {  # $1 part: its launcher's session (setsid made the launcher the session's leader)
  local pid; pid=$(cat "$V/pid_$1" 2>/dev/null)
  [[ -n $pid ]] && is_launcher "$pid" "$1" && kill -TERM -- "-$pid" 2>/dev/null
}
stop_all() {  # every part's launcher; waits up to 120 s for them to go
  local p t
  for p in "${PARTS[@]}"; do alive "$p" && { log "stopping part $p"; stop_part "$p"; }; done
  for t in $(seq 1 24); do
    for p in "${PARTS[@]}"; do alive "$p" && { sleep 5; continue 2; }; done
    return 0
  done
  summary "ALARM: a part's launcher still runs 120 s after it was stopped ($(for p in "${PARTS[@]}"; do alive "$p" && printf '%s ' "$p"; done)); stop it by hand"
}
ports_up() {  # $1 worker: every model and VO forward to that worker accepts a connection
  local s
  for s in 0 1 2 3 4 5 6 7; do
    (exec 3<>"/dev/tcp/127.0.0.1/$(( MB[$1] + s ))") 2>/dev/null || return 1
    (exec 3<>"/dev/tcp/127.0.0.1/$(( VB[$1] + s ))") 2>/dev/null || return 1
  done
}
owner() { cat "$V/owner_$1" 2>/dev/null; }
current() {  # $1 worker: the part it holds that is not complete (at most one), or nothing
  local p
  for p in "${PARTS[@]}"; do [[ "$(owner "$p")" == "$1" ]] && ! part_done "$p" && { echo "$p"; return 0; }; done
  return 1
}
busy() {  # $1 worker: one of its parts still has a live launcher (a complete part's may be finishing)
  local p
  for p in "${PARTS[@]}"; do [[ "$(owner "$p")" == "$1" ]] && alive "$p" && return 0; done
  return 1
}
instance_of() {  # $1 worker: the server instance its local copy names
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('server_instance') or '-')" "$NS/npu_servers_exp28_${TAG[$1]}/servers.json" 2>/dev/null
}
suspect() {  # $1 worker: marked as published but not answering, for the instance its copy names now
  local s; s=$(cat "$V/suspect_$1" 2>/dev/null); [[ -n $s ]] || return 1
  [[ $s == "$(instance_of "$1")" ]] && return 0
  rm -f "$V/suspect_$1"; summary "${TAG[$1]} is served again (instance $(instance_of "$1")); $1 takes parts again"; return 1
}
server_trouble() {  # $1 part: its launcher's output since the last start shows the servers failing, not the part
  local off; off=$(cat "$V/logoff_$1" 2>/dev/null); [[ $off =~ ^[0-9]+$ ]] || off=0
  tail -c "+$((off + 1))" "$V/val_$1.out" 2>/dev/null \
    | grep -qE "RPC startup timeout at slot|no healthy RPC within|stopped listening|are not the ones in|not the recorded set"
}
free_worker() {  # $1 worker: serving X37 on a live tunnel, not suspect, holding no part nor a live launcher
  current "$1" >/dev/null && return 1
  busy "$1" && return 1
  ports_up "$1" || return 1
  refresh_pub "$1" || return 1
  ! suspect "$1"
}
row_check() {  # $1 B0 progress.jsonl, then run dirs: one line "<reading> <n paired> <dSR> <SE> <lower> <dSPL> <rows
               # not base-stop>" for the parts' rows paired with B0, as run (reading: short / continue / futility /
               # badmode).  Rows are read as val_split3.py reads them (a row being written is skipped, the
               # first of a repeat within a run kept, one episode in two runs refused).
  python3 - "$1" "$SHA" "$MODE_SUFFIX" "$FUTILITY_N" "${@:2}" <<'PY'
import glob, json, math, sys
b0, sha, suffix, need, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
key = lambda r: (str(r["scene_id"]), str(r["episode_id"]))
t, owner, bad = {}, {}, 0
for run in runs:
    for path in sorted(glob.glob(f"{run}/workers/shard_0*/progress.json")):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = key(r)
            if k in owner and owner[k] != run:
                sys.exit(f"episode {k} is recorded by both {owner[k]} and {run}")
            if k in t:
                continue
            owner[k], t[k] = run, r
            v = str(r.get("rpc_model_version") or "")
            bad += not (v.endswith(suffix) and sha in v)
c = {}
for line in open(b0):
    if line.strip():
        r = json.loads(line); c[key(r)] = r
ids = sorted(set(t) & set(c))
n = len(ids)
if n < 2:
    print("badmode" if bad else "short", n, "- - - -", bad); sys.exit(0)
s = sum(int(round(float(t[i]["success"]) - float(c[i]["success"]))) for i in ids)
d = [100.0 * (float(t[i]["success"]) - float(c[i]["success"])) for i in ids]
m = sum(d) / n
se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
dspl = 100.0 * sum(float(t[i]["spl"]) - float(c[i]["spl"]) for i in ids) / n
# -2.0 pt exactly: 100 * s / n <= -2  <=>  50 * s <= -n (integers, no rounding)
reading = "badmode" if bad else "short" if n < need else "futility" if 50 * s <= -n else "continue"
print(reading, n, f"{m:+.2f}", f"{se:.2f}", f"{m - 1.96 * se:+.2f}", f"{dspl:+.2f}", bad)
PY
}
interim() {  # the ledger's interim futility: one look, at the first set of completed parts (not 9 of 9) that
             # pairs >= FUTILITY_N episodes with B0 ($V/interim.final marks it spent); every growth of the set
             # before and after checks the rows' mode, only that look reads their outcomes
  local done_parts=() p reading n m se lower dspl bad
  for p in "${PARTS[@]}"; do part_done "$p" && done_parts+=("$p"); done
  (( ${#done_parts[@]} > 0 && ${#done_parts[@]} < ${#PARTS[@]} )) || return 0
  [[ "${done_parts[*]}" == "$(cat "$V/interim_seen" 2>/dev/null)" ]] && return 0
  if ! read -r reading n m se lower dspl bad < <(row_check "$B0/merged/progress.jsonl" \
       $(for p in "${done_parts[@]}"; do printf '%s ' "${RUN}_$p"; done) 2>> "$V/val_4090.log"); then
    log "interim: the row check failed on ${done_parts[*]}; trying again next pass"; return 0
  fi
  if [[ -e $V/interim.final && $reading != badmode ]]; then  # the one look is spent: the mode check only
    log "interim: parts ${done_parts[*]} in base-stop mode with adapter $SHA (the futility look was taken at $(cat "$V/interim.final"))"
    echo "${done_parts[*]}" > "$V/interim_seen"; return 0
  fi
  case $reading in
    badmode)
      summary "ALARM: $bad rows of the completed parts (${done_parts[*]}) do not name adapter $SHA in base-stop mode (rpc_model_version ...$MODE_SUFFIX); stopping the val"
      stop_all; status "STOPPED: rows not in base-stop mode"; exit 2 ;;
    short) log "interim: parts ${done_parts[*]} pair $n episodes with B0 (< $FUTILITY_N); no reading yet" ;;
    continue)
      summary "interim, parts ${done_parts[*]}: n $n, dSR $m (SE $se, lower $lower), dSPL $dspl (as run) > $FUTILITY_PT; the val goes on (the one futility look is spent)"
      echo "parts ${done_parts[*]}, n $n" > "$V/interim.final" ;;
    futility)
      summary "interim, parts ${done_parts[*]}: n $n, dSR $m (SE $se, lower $lower), dSPL $dspl (as run) <= $FUTILITY_PT: 否定（中途无效性）; stopping the remaining parts"
      stop_all
      echo "X37 vs B0 seed 1337 (as run, interim on parts ${done_parts[*]}): n $n, dSR $m (SE $se, lower $lower), dSPL $dspl -> 否定（中途无效性） (Δ̂ <= $FUTILITY_PT at >= $FUTILITY_N paired episodes)" > "$V/verdict_vs_b0.txt"
      summary "$(cat "$V/verdict_vs_b0.txt")"
      touch "$V/futility.done"; status "X37 val stopped by the interim futility rule (see $V/verdict_vs_b0.txt)"; exit 0 ;;
    *) log "interim: unreadable row check '$reading' on ${done_parts[*]}; trying again next pass"; return 0 ;;
  esac
  echo "${done_parts[*]}" > "$V/interim_seen"
}

(( $(rows "$B0") >= 1839 )) || { summary "ALARM: B0 seed 1337 has $(rows "$B0") merged rows, not 1,839"; status "STOPPED: B0 incomplete"; exit 2; }

# --- 1. the equivalence check: the zero adapter in base-stop mode identical to B0 call by call (equiv_check.py
#        compare); its servers' commit is the one every X37 set must be built from
#        (status identical is not enough: it must be against B0 seed 1337 on its episodes 1/25/41, every one
#        compared and identical, and every candidate set in base-stop mode with one commit and one adapter, the
#        zero adapter, which is never pinned as X37)
EQUIV_COMMIT=""; ZERO_SHA=""; estatus=""
t0=$(date +%s); waited=""; alarmed=""
until read -r estatus EQUIV_COMMIT ZERO_SHA < <(python3 -c "
import json,os,sys
d=json.load(open(sys.argv[1]))
servers=d.get('candidate_servers') or []
commits={s.get('repo_commit') for s in servers}
shas={s.get('system2_lora_sha256') for s in servers}
asked=d.get('episodes_asked') or []
why=[]
if os.path.normpath(str(d.get('reference')))!=sys.argv[2]: why.append('reference %r is not B0 seed 1337'%d.get('reference'))
if sorted(asked)!=sorted(int(x) for x in sys.argv[3].split()): why.append('episodes %r, not %s'%(asked,sys.argv[3]))
if d.get('identical')!=len(asked) or len(d.get('episodes') or [])!=len(asked): why.append('%r of %d episodes compared identical'%(d.get('identical'),len(asked)))
if not servers or any(s.get('system2_base_stop') is not True for s in servers): why.append('candidate servers not all in base-stop mode')
if len(commits)!=1 or len(shas)!=1: why.append('candidate servers of %d commits and %d adapters'%(len(commits),len(shas)))
st=d.get('status') or 'none'
print(st if st!='identical' or not why else 'identical-but:'+'|'.join(why).replace(' ','_'),
      commits.pop() if len(commits)==1 else '-', (shas.pop() or '-') if len(shas)==1 else '-')" "$EQUIV" "$B0" "$EQUIV_EPISODES" 2>/dev/null) \
      && [[ $estatus == identical && $EQUIV_COMMIT =~ ^[0-9a-f]{40}$ && $ZERO_SHA =~ ^[0-9a-f]{64}$ ]]; do
  if [[ -n ${X37_EQUIV_OVERRIDE:-} ]]; then  # only on the user's word, with the reason
    summary "deviation: the val starts without a passed equivalence check ($EQUIV reads '${estatus:-nothing}'): $X37_EQUIV_OVERRIDE"
    EQUIV_COMMIT=""; ZERO_SHA=""; break
  fi
  [[ -n $waited ]] || { status "waiting for a passed equivalence check in $EQUIV (equiv_check.py compare), now '${estatus:-none}'"; waited=1; }
  if [[ -z $alarmed ]] && (( $(date +%s) - t0 > LATE )); then
    summary "ALARM: no passed equivalence check in $EQUIV after $((LATE / 3600)) h; the val waits for it"; alarmed=1
  fi
  sleep "$TICK"
done
if [[ -n $EQUIV_COMMIT ]]; then
  NOT_X37+=("$ZERO_SHA")
  log "equivalence check passed (zero adapter $ZERO_SHA, never pinned); X37 sets must be built from $EQUIV_COMMIT"
fi

# --- 2. the adapter: the one candidate a passed offline gate names ($GATE: all_pass, next val), X37_SHA agreeing
#        with it; X37_SHA alone only with X37_GATE_OVERRIDE (on the user's word, with the reason).  A rerun
#        checks the pinned sha against the gate again.  A verdict whose next step is no_val stops the val.
gate_sha() {  # the verdict's adapter if it lets the candidate go to val, else "<next> -"
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
named={v for v in (d.get('adapter_sha256') or {}).values() if v}
ok=d.get('schema')==sys.argv[2] and d.get('all_pass') is True and d.get('measured') is True and d.get('next')=='val'
print(d.get('next') or '-', named.pop() if ok and len(named)==1 else '-')" "$GATE" "$GATE_SCHEMA" 2>/dev/null
}
SHA=""; GATE_SHA=""; gnext=""
t0=$(date +%s); waited=""; alarmed=""
until read -r gnext GATE_SHA < <(gate_sha) && [[ $GATE_SHA =~ ^[0-9a-f]{64}$ ]]; do
  if [[ -n ${X37_GATE_OVERRIDE:-} ]]; then
    [[ ${X37_SHA:-} =~ ^[0-9a-f]{64}$ ]] \
      || { summary "ALARM: X37_GATE_OVERRIDE needs X37_SHA (the adapter's sha256)"; status "STOPPED: no adapter"; exit 2; }
    summary "deviation: the val pins X37_SHA $X37_SHA without a passed offline gate ($GATE reads next '${gnext:-nothing}'): $X37_GATE_OVERRIDE"
    GATE_SHA=$X37_SHA; break
  fi
  if [[ $gnext == no_val ]]; then
    summary "ALARM: the offline gate ($GATE) reads next no_val: the candidate does not go to val (ledger 离线门)"
    status "STOPPED: the offline gate failed"; exit 2
  fi
  [[ -n $waited ]] || { status "waiting for a passed offline gate in $GATE (gate_verdict.py: all_pass, next val), now '${gnext:-none}'"; waited=1; }
  if [[ -z $alarmed ]] && (( $(date +%s) - t0 > LATE )); then
    summary "ALARM: no passed offline gate in $GATE after $((LATE / 3600)) h; the val waits for it"; alarmed=1
  fi
  sleep "$TICK"
done
if [[ -n ${X37_SHA:-} && $X37_SHA != "$GATE_SHA" ]]; then
  summary "ALARM: X37_SHA $X37_SHA is not the adapter the offline gate passed ($GATE_SHA)"; status "STOPPED: X37_SHA disagrees with the gate"; exit 2
fi
if [[ -s $V/x37.sha ]]; then
  SHA=$(cat "$V/x37.sha")
  [[ $SHA == "$GATE_SHA" ]] \
    || { summary "ALARM: the pinned adapter $SHA ($V/x37.sha) is not the one the gate names now ($GATE_SHA)"; status "STOPPED: pinned adapter changed"; exit 2; }
else
  SHA=$GATE_SHA
  summary "X37 val: adapter $SHA ($([[ -n ${X37_GATE_OVERRIDE:-} ]] && echo "X37_SHA, gate overridden" || echo "passed the offline gate, $GATE"))"
  echo "$SHA" > "$V/x37.sha"; [[ -n ${X37_GATE_OVERRIDE:-} ]] && echo "override" > "$V/adopted_from" || echo "$GATE" > "$V/adopted_from"
fi
[[ $SHA =~ ^[0-9a-f]{64}$ ]] || { summary "ALARM: $V/x37.sha holds no sha256"; status "STOPPED: no adapter"; exit 2; }
not_x37 "$SHA" && { summary "ALARM: the pinned adapter $SHA is X34M's, X35's, the zero adapter's or in $A/not_x37.sha"; status "STOPPED: wrong adapter"; exit 2; }
[[ -s $V/adopted.t ]] || date +%s > "$V/adopted.t"

# --- 3. nine parts, each a ninth of every shard in cohort order (EXP-35's split)
if [[ ! -e $V/split.done ]]; then
  rm -rf "$V"/lists_p*
  VS split "$PLAN/cohorts" $(for p in "${PARTS[@]}"; do printf '%s ' "$V/lists_$p"; done) 2> "$V/split.txt" \
    || { summary "ALARM: the nine-way split failed (see $V/split.txt)"; status "STOPPED: split failed"; exit 2; }
  touch "$V/split.done"; summary "X37 val split in nine: $(tr '\n' ';' < "$V/split.txt")"
fi

# --- 4. every pass: resume or hand over the started parts whose launcher is gone (a resume waits out a
#        tunnel outage or an unpublished set uncounted, the count restarts whenever the part made progress,
#        and 20 resumes without progress raise an ALARM and slow the resumes to one per 30 min); the
#        interim futility reading; then a free worker takes the next unstarted part
status "X37 val: 9 parts on the workers serving X37 (adapter $SHA, base-stop)"
while true; do
  for p in "${PARTS[@]}"; do
    w=$(owner "$p"); [[ -n $w ]] || continue
    if part_done "$p"; then
      if alive "$p"; then
        DONE_AT[$p]=${DONE_AT[$p]:-$(date +%s)}
        if (( $(date +%s) - DONE_AT[$p] > HANG )); then
          log "part $p is complete but its launcher still runs after $HANG s; stopping it"; stop_part "$p"
        fi
      fi
      continue
    fi
    read -r got _ < <(VS count "${RUN}_$p" "$V/lists_$p")
    [[ $got =~ ^[0-9]+$ ]] || { log "part $p: its count failed; skipped this pass"; continue; }
    if [[ $got != "$(cat "$V/got_$p" 2>/dev/null)" ]]; then  # progress (or the first look): the clocks restart
      date +%s > "$V/prog_t_$p"; echo "$got" > "$V/got_$p"; rm -f "$V/stall_alarmed_$p"; RESUMES[$p]=0
    fi
    pt=$(cat "$V/prog_t_$p" 2>/dev/null); [[ $pt =~ ^[0-9]+$ ]] || { pt=$(date +%s); echo "$pt" > "$V/prog_t_$p"; }
    stalled=$(( $(date +%s) - pt ))
    if (( stalled > STALL )) && [[ ! -e $V/stall_alarmed_$p ]]; then
      summary "ALARM: part $p on $w: no progress for $((stalled / 60)) min at $got recorded (launcher $(alive "$p" && echo alive || echo gone), tunnel $(ports_up "$w" && echo up || echo down); see $V/val_$p.out)"
      touch "$V/stall_alarmed_$p"
    fi
    alive "$p" && continue
    why=""
    if ! ports_up "$w"; then
      why="the tunnel to ${TAG[$w]} is down"
    else
      refresh_pub "$w"; case $? in
        1) why="npu_servers_exp28_${TAG[$w]} is not published" ;;
        2) why="npu_servers_exp28_${TAG[$w]} names another adapter, mode or commit" ;;
      esac
    fi
    # published and reachable, but its servers do not answer (a retired NPU slot, say): after HANDOFF
    # without progress, two resumes, and the launcher's own words for it, the worker is suspect for
    # the instance it serves until it is served again (a new instance)
    if [[ -z $why ]] && suspect "$w"; then why="${TAG[$w]} (instance $(instance_of "$w")) is published but did not answer"; fi
    if [[ -z $why ]] && (( stalled > HANDOFF && ${RESUMES[$p]:-0} >= 2 )) && server_trouble "$p"; then
      instance_of "$w" > "$V/suspect_$w"
      summary "ALARM: ${TAG[$w]} (instance $(instance_of "$w")) is published but its servers do not answer (see $V/val_$p.out); it takes no part until served again (swap_servers.sh on worker-${w#w})"
      why="${TAG[$w]} is published but did not answer"
    fi
    if [[ -n $why ]]; then
      if (( stalled > HANDOFF )); then
        for x in "${WORKERS[@]}"; do
          [[ $x == "$w" ]] && continue
          free_worker "$x" || continue
          summary "part $p: $why and no progress for $((stalled / 60)) min; handed from $w to $x"
          echo "$x" > "$V/owner_$p"; date +%s > "$V/prog_t_$p"; RESUMES[$p]=0; start_part "$p" "$x"; continue 2
        done
      fi
      log "part $p stopped; $why, waiting (not counted)"; continue
    fi
    RESUMES[$p]=$(( ${RESUMES[$p]:-0} + 1 ))
    (( RESUMES[$p] == 21 )) && summary "ALARM: part $p: 20 resumes without progress at $got recorded; resuming every 30 min from now (see $V/val_$p.out)"
    if (( RESUMES[$p] <= 20 || RESUMES[$p] % 6 == 0 )); then
      log "part $p stopped before completing; resuming on $w (try ${RESUMES[$p]} since its last progress, $got recorded)"; start_part "$p" "$w"
    fi
  done
  interim
  for w in "${WORKERS[@]}"; do  # new parts for free workers
    current "$w" >/dev/null && continue
    busy "$w" && continue
    next=""; for p in "${PARTS[@]}"; do [[ -z "$(owner "$p")" ]] && { next=$p; break; }; done
    [[ -n $next ]] || break
    if ! refresh_pub "$w"; then
      at=$(cat "$V/adopted.t" 2>/dev/null); [[ $at =~ ^[0-9]+$ ]] || { at=$(date +%s); echo "$at" > "$V/adopted.t"; }
      if [[ -z ${LATE_SAID[$w]:-} ]] && (( $(date +%s) - at > LATE )); then
        summary "ALARM: npu_servers_exp28_${TAG[$w]} not published with adapter $SHA in base-stop mode $((LATE / 60)) min after the adapter was adopted"
        LATE_SAID[$w]=1
      fi
      continue
    fi
    suspect "$w" && continue
    ports_up "$w" || { log "$w serves X37 but its tunnel is down; $next waits"; continue; }
    echo "$w" > "$V/owner_$next"; start_part "$next" "$w"
  done
  all=1; for p in "${PARTS[@]}"; do part_done "$p" || all=0; done
  (( all )) && break
  sleep "$TICK"
done
for p in "${PARTS[@]}"; do  # an aborted loop (a bash expansion error drops the whole command) must not merge
  part_done "$p" || { summary "ALARM: the main loop ended with part $p incomplete; not merging (rerun the controller)"; status "STOPPED: main loop ended early"; exit 2; }
done
summary "X37 val: all nine parts complete ($(for p in "${PARTS[@]}"; do printf '%s on %s %s; ' "$p" "$(owner "$p")" "$(VS count "${RUN}_$p" "$V/lists_$p")"; done))"

# --- 5. merge with the locked tool, check every row's mode, pair (paired.done only when the B0 pairing and its
#        verdict worked)
CM=${RUN}_merged
rm -rf "$CM"; mkdir -p "$CM/merged"
if VS combine "$CM" $(for p in "${PARTS[@]}"; do printf '%s ' "${RUN}_$p"; done) > "$V/merge.txt" 2>&1 \
   && ( cd "$WT" && PYTHONPATH=$PLAN/tools:/workspace/rpc/src:$WT /opt/conda/bin/python \
     "$PLAN/tools/merge_shards.py" --dataset /workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
     --cohorts-dir "$PLAN/cohorts" --workers-dir "$CM/workers" --output-dir "$CM/merged" --num-shards 8 \
     --expected-episodes 1839 --protocol heatmapvln-r2r-json-v3 --sampling-protocol heatmapvln-nextdit-sha256-v1 \
     --protocol-seed 1337 --evaluation-arm ppa_refine_v2_online_amb3r_cuda >> "$V/merge.txt" 2>&1 ); then
  summary "X37 merged: $(rows "$CM") rows in $CM/merged"
else
  summary "ALARM: merging X37's nine parts failed (see $V/merge.txt)"; status "STOPPED: merge failed"; exit 2
fi
read -r reading n _ _ _ _ bad < <(row_check "$B0/merged/progress.jsonl" "$CM" 2>> "$V/val_4090.log")
if [[ $reading == badmode || $bad != 0 || $n != 1839 ]]; then
  summary "ALARM: X37's merged rows: ${bad:-?} not in base-stop mode with adapter $SHA, ${n:-?} paired with B0 (not 1,839); not pairing"
  status "STOPPED: merged rows fail the mode check"; exit 2
fi
bad=""
for mode in asrun standard; do
  extra=(); [[ $mode == standard ]] && extra=(--standard-protocol)
  out=$V/paired_vs_b0_seed1337_$mode.json; rm -f "$out"
  ( cd "$WT" && /opt/conda/bin/python scripts/tools/paired_closed_loop_bootstrap.py --treatment "$CM/merged/progress.jsonl" \
      --control "$B0/merged/progress.jsonl" --bootstrap-draws 20000 --label "x37_vs_b0_seed1337_$mode" \
      "${extra[@]}" --output-json "$out" > "${out%.json}.txt" 2>&1 )
  rc=$?
  if (( rc != 0 )) || [[ ! -s $out ]]; then
    summary "ALARM: pairing X37 vs B0 ($mode) failed (exit $rc, see ${out%.json}.txt)"; bad+=" $mode"
  fi
done
# the primary criterion (support: dSR - 1.96 SE > 0 and dSPL >= -1.0; refuted: dSR <= 0; else not measured), as run
# being the one that decides and the standard protocol reported beside it
PYTHONPATH=$WT/scripts/tools /opt/conda/bin/python "$VS_PY" verdict "$CM/merged/progress.jsonl" \
  "$B0/merged/progress.jsonl" "X37 vs B0" main > "$V/verdict_vs_b0.txt" 2>&1
rc=$?
(( rc == 0 )) || { summary "ALARM: the verdict X37 vs B0 failed (exit $rc, see $V/verdict_vs_b0.txt)"; bad+=" verdict"; }
summary "$(tr '\n' ';' < "$V/verdict_vs_b0.txt") (primary criterion: the as-run line)"
# the mechanism (report only): not within 3 m, by how it ended (its own STOP / the step budget), and within 3 m
# but failed
python3 - "$CM/merged/progress.jsonl" "$B0/merged/progress.jsonl" > "$V/mechanism.txt" 2>&1 <<'PY'
import json, sys
def counts(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    far = [r for r in rows if not float(r["os"]) > 0]
    budget = sum(1 for r in far if r.get("step_budget_stop"))
    rbf = sum(1 for r in rows if float(r["os"]) > 0 and not float(r["success"]) > 0)
    return len(far) - budget, budget, rbf
(xs, xb, xr), (bs, bb, br) = counts(sys.argv[1]), counts(sys.argv[2])
print(f"mechanism (report only), X37 vs B0: never within 3 m {xs + xb} vs {bs + bb} (own STOP {xs} vs {bs}, "
      f"step budget {xb} vs {bb}) -> {'fewer' if xs + xb < bs + bb else 'not fewer'}; within 3 m but failed {xr} vs {br} "
      f"-> {'not more' if xr <= br else 'more'}")
PY
summary "$(cat "$V/mechanism.txt")"
[[ -z $bad ]] || { status "STOPPED: the B0 pairing failed ($bad); a rerun redoes step 5"; exit 2; }
touch "$V/paired.done"
status "X37 val done: paired results in $V (verdict_vs_b0.txt, paired_vs_b0_seed1337_{asrun,standard}.json, mechanism.txt)"
