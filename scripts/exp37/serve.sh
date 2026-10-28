#!/usr/bin/env bash
# EXP-37, any worker: replace this worker's server set with one served from $R/exp37/src_x37, published as
# npu_servers_exp28_<tag> on 52418 by scripts/exp28/serve_910b.sh (as EXP-34's swap_servers.sh does, on
# this worker's ports: w0 52410/52510, w1 52420/52520, w2 52430/52530).  Three modes:
#   serve.sh TAG [ADAPTER_DIR]           X37 for the val: the adapter (default $R/exp37/train/X37/adapter,
#                                        which must be the one its training's manifest.json lists) with
#                                        PPA_EVAL_SYSTEM2_BASE_STOP=1 (server mode base-stop-v1), only once the
#                                        offline gate passed it: gate_verdict.py's verdict.json, copied from the
#                                        4090 to $X37_GATE_VERDICT (default $R/exp37/gate/verdict.json), must read
#                                        all_pass, measured, next val and name this adapter's sha256 alone (as
#                                        val_4090.sh reads it).  X37_SERVE_WITHOUT_GATE="<reason>" serves before
#                                        that, logged as a deviation (val_4090.sh still pins only a passed sha).
#   serve.sh --zero-check TAG ADAPTER    any adapter with the same flag: the zero adapter of
#                                        scripts/exp37/make_zero_adapter.py for the equivalence check against B0
#   serve.sh --base TAG                  the released System2, no adapter, no flag (a B0-like reference set)
# The val's tags x37 / x37w1 / x37w2 are for the first mode only.
# Before serving it waits for any training, gate or its controller on this worker to end (their cards), then stops
# the set this worker holds (SIGTERM to run_ppa_servers_npu.sh; its trap stops every server) and waits for
# serve_910b.sh to exit.  Once the new set is published it re-reads the published servers.json and
# withdraws the set (unpublish + SIGTERM) unless it carries the asked adapter sha256 (none for --base),
# system2_base_stop true (false for --base), the base-stop-v1 model_version suffix, and this checkout's
# commit, clean.  Then it stays with the set and logs its end.  Writes only under $R.
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/serve.sh x37 >> $R/exp37/logs/serve_x37.out 2>&1 < /dev/null &        # worker-0
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/serve.sh x37w1 >> $R/exp37/logs/serve_x37w1.out 2>&1 < /dev/null &    # worker-1
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/serve.sh --zero-check x37z $R/exp37/zero_adapter >> ... &
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/serve.sh --base x37b0 >> ... &
set -uo pipefail
R=/mnt/sfs_turbo/zhr/zhr_1
SRC=${X37_SRC:-$R/exp37/src_x37}
OUT=${X37_OUT:-$R/exp37/train/X37}
EXPORT=$R/exp23/round1/export
LOGS=$R/exp37/logs
MODE=x37
case ${1:-} in
  --zero-check) MODE=zero; shift ;;
  --base) MODE=base; shift ;;
  -*) echo "usage: $0 [--zero-check|--base] TAG [ADAPTER_DIR]" >&2; exit 2 ;;
esac
TAG=${1:-}
ADAPTER=${2:-}
[[ $TAG =~ ^[a-z0-9]+$ ]] || { echo "usage: $0 [--zero-check|--base] TAG [ADAPTER_DIR] (TAG: lower-case letters and digits)" >&2; exit 2; }
case $MODE in
  x37) ADAPTER=${ADAPTER:-$OUT/adapter}; BASE_STOP=1 ;;
  zero) [[ -n $ADAPTER ]] || { echo "--zero-check needs the adapter directory" >&2; exit 2; }; BASE_STOP=1 ;;
  base) [[ -z $ADAPTER ]] || { echo "--base takes no adapter" >&2; exit 2; }; BASE_STOP=0 ;;
esac
# val_4090.sh adopts a base-stop set on these tags as X37 unless its sha is listed as not X37: keep them for X37.
case $MODE:$TAG in
  zero:x37|zero:x37w1|zero:x37w2|base:x37|base:x37w1|base:x37w2)
    echo "the val tags x37 / x37w1 / x37w2 are for X37 only; give --${MODE/zero/zero-check} another tag" >&2; exit 2 ;;
esac
W=$(hostname | grep -o 'worker-[0-9]*$' | cut -d- -f2)
case $W in
  0) MB=52410 VB=52510 ;;
  1) MB=52420 VB=52520 ;;
  2) MB=52430 VB=52530 ;;
  *) echo "$(hostname) is not worker-0/1/2 of this job" >&2; exit 2 ;;
esac
RUN_DIR=$R/exp37/serve/$TAG
PUB=$EXPORT/npu_servers_exp28_$TAG
mkdir -p "$LOGS" "$R/exp37/serve"
log() { printf '%s w%s %s %s\n' "$(date -u +%FT%TZ)" "$W" "$TAG" "$*" | tee -a "$LOGS/serve.log"; }
old=$(cat "$R/exp37/serve/.serve_w$W.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "exp37/serve.sh"; then
  log "refused: serve.sh runs on this worker as $old"; exit 1
fi
echo $$ > "$R/exp37/serve/.serve_w$W.pid"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
# Greedy, as B0 was served: no sampling knob leaks in from the caller's environment.
unset PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE PPA_EVAL_SYSTEM2_SAMPLE_TOP_P ASCEND_RT_VISIBLE_DEVICES

git_src() { git -c safe.directory="$SRC" -C "$SRC" "$@"; }
[[ -d $SRC/.git ]] || { log "refused: no code checkout at $SRC"; exit 2; }
git_src diff --quiet HEAD -- && [[ -z "$(git_src status --porcelain)" ]] || { log "refused: $SRC is not a clean checkout"; exit 2; }
COMMIT=$(git_src rev-parse HEAD)
for f in scripts/ascend/run_ppa_servers_npu.sh scripts/exp28/serve_910b.sh; do
  grep -q 'PPA_EVAL_SYSTEM2_BASE_STOP' "$SRC/$f" || { log "refused: $SRC ($COMMIT) $f has no PPA_EVAL_SYSTEM2_BASE_STOP"; exit 2; }
done
SHA=""
if [[ -n $ADAPTER ]]; then
  given=$ADAPTER
  ADAPTER=$(realpath -e -- "$given") || { log "refused: no adapter directory $given"; exit 2; }
  [[ -s $ADAPTER/adapter_model.safetensors && -s $ADAPTER/adapter_config.json ]] || { log "refused: no adapter at $ADAPTER"; exit 2; }
  SHA=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
fi
if [[ $MODE == x37 ]]; then
  # Only an adapter its training finished and listed: the done event and the manifest's sha256 of this file.
  T=$(dirname "$ADAPTER")
  grep -q '"event": "done"' "$T/train_log.jsonl" 2>/dev/null || { log "refused: $T has no finished training (train_log.jsonl done event)"; exit 2; }
  listed=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['adapter']['files']['adapter_model.safetensors'])" "$T/manifest.json" 2>/dev/null)
  [[ $listed == "$SHA" ]] || { log "refused: $ADAPTER/adapter_model.safetensors is $SHA, $T/manifest.json lists '${listed:-nothing}'"; exit 2; }
  # ... and that the offline gate passed (ledger 离线门: nothing that fails it goes to val).
  VERDICT=${X37_GATE_VERDICT:-$R/exp37/gate/verdict.json}
  gate=$(python3 - "$VERDICT" "$SHA" 2>&1 <<'PY'
import json, sys
path, sha = sys.argv[1:3]
try:
    d = json.load(open(path))
except (OSError, ValueError) as exc:
    sys.exit(f"no readable verdict ({exc})")
named = {v for v in (d.get("adapter_sha256") or {}).values() if v}
if d.get("schema") != "heatmapvln-exp37-offline-gate-verdict-v1":
    sys.exit(f"schema {d.get('schema')!r}")
if not (d.get("all_pass") is True and d.get("measured") is True and d.get("next") == "val"):
    sys.exit(f"all_pass {d.get('all_pass')!r}, measured {d.get('measured')!r}, next {d.get('next')!r}")
if named != {sha}:
    sys.exit(f"it names {sorted(named)}, not {sha} alone")
print("passed")
PY
)
  if [[ $gate != passed ]]; then
    [[ -n ${X37_SERVE_WITHOUT_GATE:-} ]] || { log "refused: no passed offline gate for $SHA in $VERDICT: $gate"; exit 2; }
    log "deviation: serving $SHA without a passed offline gate ($VERDICT: $gate): $X37_SERVE_WITHOUT_GATE"
  else
    log "offline gate passed for $SHA ($VERDICT)"
  fi
fi
log "mode $MODE: src $SRC at $COMMIT, adapter ${ADAPTER:-none} ${SHA:+($SHA)}, base_stop $BASE_STOP, ports $MB/$VB"

# The cards: whatever trains or decodes a gate here finishes first (serve_910b.sh gives up after 30 min of busy
# cards), controllers included: they hold the cards between their workers (summarize, retry sleeps, handshakes).
busy_here() { pgrep -u "$(id -u)" -f "[r]un_gate_910b.sh|[t]rain_node.sh|[r]un_s2_lora_train_npu.sh|[t]rain_system2_lora.py|[e]val_pairs.py score|[h]m3d_route_gate.py decode|[s]top_rate_gate.py decode" >/dev/null; }
if busy_here; then log "waiting for this worker's training / gate (and their controllers) to end"; fi
while busy_here; do sleep 30; done
for p in $(pgrep -u "$(id -u)" -f "^bash scripts/ascend/run_ppa_servers_npu.sh"); do
  log "stopping server set: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -TERM "$p"
done
while pgrep -u "$(id -u)" -f "[s]cripts/exp28/serve_910b.sh" >/dev/null; do sleep 10; done
log "old set gone; serving"

RR=$(cd "$R" && pwd -P)  # serve_910b.sh's server TMPDIRs go under $EXP23_ROOT/t: a real path (E40021)
mkdir -p "$RUN_DIR"; rm -f "$RUN_DIR/STATUS"
( cd "$SRC" && exec env EXP23_ROOT="$RR" EXP28_TAG="$TAG" EXP28_ADAPTER="$ADAPTER" EXP28_SRC="$SRC" \
    EXP28_NPUS=0,1,2,3,4,5,6,7 EXP28_MODEL_PORT_BASE=$MB EXP28_VO_PORT_BASE=$VB EXP28_EXPORT_PORT=52418 \
    EXP28_EXPORT_DIR="$EXPORT" EXP28_RUN_DIR="$RUN_DIR" EXP28_SERVE_S=1209600 PPA_EVAL_SYSTEM2_BASE_STOP=$BASE_STOP \
    bash "$SRC/scripts/exp28/serve_910b.sh" > "$R/exp37/serve/$TAG.out" 2>&1 < /dev/null ) &
spid=$!
until grep -q 'serving:\|STOPPED\|ENDED' "$RUN_DIR/STATUS" 2>/dev/null; do
  kill -0 "$spid" 2>/dev/null || break
  sleep 20
done
if ! grep -q 'serving:' "$RUN_DIR/STATUS" 2>/dev/null; then
  wait "$spid"
  log "serve_910b.sh did not serve: $(cat "$RUN_DIR/STATUS" 2>/dev/null || echo 'no STATUS')"; exit 2
fi
# What the 4090 will pull, read back: the asked adapter, the mode, this commit, clean.
# Into a file, not a command substitution: a crash of this check must refuse the set, not read as "no problems".
python3 - "$PUB/servers.json" "$SHA" "$BASE_STOP" "$COMMIT" > "$RUN_DIR/servers_check.txt" 2>&1 <<'PY'
import json, sys
path, sha, base_stop, commit = sys.argv[1:5]
try:
    s = json.load(open(path))
except Exception as exc:
    print(f"cannot read {path}: {exc}"); sys.exit(0)
want = base_stop == "1"
problems = []
if s.get("system2_lora_sha256", "") != sha:
    problems.append(f"system2_lora_sha256 {s.get('system2_lora_sha256')!r}, want {sha!r}")
if s.get("system2_base_stop", False) is not want:
    problems.append(f"system2_base_stop {s.get('system2_base_stop')!r}, want {want}")
if ("+base-stop-v1" in str(s.get("model_version", ""))) is not want:
    problems.append(f"model_version {s.get('model_version')!r} {'lacks' if want else 'carries'} +base-stop-v1")
if s.get("repo_commit") != commit or int(s.get("repo_dirty", 1)) != 0:
    problems.append(f"repo_commit {s.get('repo_commit')!r} dirty {s.get('repo_dirty')!r}, want {commit} clean")
print("; ".join(problems))
PY
rc=$?
why=$(cat "$RUN_DIR/servers_check.txt" 2>/dev/null)
(( rc == 0 )) || why="the servers.json check exited $rc: ${why:-no output}"
if [[ -n $why ]]; then
  rm -rf "$PUB" "$PUB.part"
  for p in $(pgrep -u "$(id -u)" -f "^bash scripts/ascend/run_ppa_servers_npu.sh"); do kill -TERM "$p"; done
  log "REFUSED and withdrawn: $why"
  wait "$spid"; exit 2
fi
log "serving $(sed 's/^[^ ]* //' "$RUN_DIR/STATUS"); servers.json checked (adapter ${SHA:-none}, base_stop $BASE_STOP)"
wait "$spid"
log "server set $TAG ended: $(cat "$RUN_DIR/STATUS" 2>/dev/null)"
