#!/usr/bin/env bash
# EXP-37, one node of X37's training (ledger EXP-37 设置 配方): per-state DPO from the base on the
# pair stream, 16 cards = worker-0 (node 0) + worker-1 (node 1), global batch 64 pairs, accumulation 4.
# Modelled on EXP-35's round2_w0.sh / round2_w1.sh (the handshake is files in $MN on the shared SFS):
#   node 0 (worker-0): checks the checkout (clean, records its commit), the pair stream and the trainer's
#     arguments (a parse on the CPU) before touching anything, then stops this worker's server set (SIGTERM
#     to run_ppa_servers_npu.sh, as swap_servers.sh does) and waits for its 8 cards to be free; then, per
#     attempt K = <run>.<n>: paths.K / args.K / go.K -> worker-1 checks the same commit and paths, stops its
#     set, answers ready.K -> start.K, both nodes launch run_s2_lora_train_npu.sh; either node's failure
#     writes abort.K and the other kills its launcher.  Up to 3 attempts, resuming.  No answer from
#     worker-1 within 30 min, or its refusal, goes to 8 cards at once: the 16-card output is moved aside
#     (the run key holds world_size and grad_accum, so a resume across card counts is refused) and X37 is
#     trained from the start on this worker's 8 cards, accumulation 8 (the same 64 pairs per step: each
#     shard's rows are split by rank and its 4 reader workers are fixed), up to 3 attempts, resuming.
#   node 1 (worker-1): waits for worker-0's go.<run>.<n>, refuses (abort.K) another commit or other paths,
#     else frees its cards and runs node 1; worker-0's finished "<run> trained|fallback" ends it.
# Writes only under $R (the user's rule for this box): logs in $R/exp37/logs, handshake in $A/mn.
#
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/train_node.sh 1 >> $R/exp37/logs/train_node1.out 2>&1 < /dev/null &   # worker-1
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/train_node.sh 0 >> $R/exp37/logs/train_node0.out 2>&1 < /dev/null &   # worker-0
# Start node 1 first, or within 30 min of node 0 (node 0 waits that long for ready.K).  Node 1 exits at
# once if $A/mn/finished already names the current run (a rerun after a finished 16-card phase).
#
# The one retry the ledger allows (G1 fails and the training pairs' reward accuracy < 0.70: lr 5e-5, from
# the base again) is the same two commands with, on both workers,
#   X37_LR=5e-5 X37_OUT=$R/exp37/train/X37_lr5e-5 X37_A=$R/exp37/auto_lr5e-5
# X37_SRC / X37_STREAM / X37_A / X37_OUT must be the same on both nodes (node 1 refuses otherwise).
set -uo pipefail
R=/mnt/sfs_turbo/zhr/zhr_1
NODE=${1:-}
[[ $NODE == 0 || $NODE == 1 ]] || { echo "usage: $0 0|1   (0 on worker-0, 1 on worker-1)" >&2; exit 2; }
SRC=${X37_SRC:-$R/exp37/src_x37}
A=${X37_A:-$R/exp37/auto}
STREAM=${X37_STREAM:-$R/exp37/stream/main}
OUT=${X37_OUT:-$R/exp37/train/X37}
LR=${X37_LR:-1e-5}
LOGS=$R/exp37/logs
MN=$A/mn
PY=$R/envs/ppa/bin/python
# EXP-35's stream: its held-out scenes are X37's (ledger EXP-37: the pair stream holds out the same scenes).
EXP35_MANIFEST=$R/exp35/stream/main/stream_manifest.json
PAIR_SCHEMA=heatmapvln-s2-pair-stream-v1
mkdir -p "$LOGS" "$MN" "$R/exp37/train" "$R/exp37/tmp"
RR=$(cd "$R" && pwd -P)  # a real path: CANN 8.5.2 refuses every JIT compile under a TMPDIR with a link (E40021)
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGS/train_node$NODE.log"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$A/STATUS_node$NODE"; log "STATUS $*"; }
fail() { status "STOPPED: $*"; exit 2; }
mn_put() { printf '%s\n' "$2" > "$MN/.$1.tmp.n$NODE" && mv "$MN/.$1.tmp.n$NODE" "$MN/$1"; }  # atomic for the peer's poll
git_src() { git -c safe.directory="$SRC" -C "$SRC" "$@"; }

[[ "$(hostname)" == *-worker-$NODE ]] || { log "this is $(hostname), not worker-$NODE; refusing"; exit 2; }
old=$(cat "$A/.train_node$NODE.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "train_node.sh $NODE"; then
  log "refused: train_node.sh $NODE runs as $old"; exit 1
fi
echo $$ > "$A/.train_node$NODE.pid"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
unset ASCEND_RT_VISIBLE_DEVICES

# --- the checkout: clean (tracked and untracked), its commit recorded, and the DPO trainer in it
[[ -d $SRC/.git ]] || fail "no code checkout at $SRC"
git_src diff --quiet HEAD -- && [[ -z "$(git_src status --porcelain)" ]] || fail "$SRC is not a clean checkout: $(git_src status --porcelain | head -5 | tr '\n' ' ')"
COMMIT=$(git_src rev-parse HEAD) || fail "no commit at $SRC"
grep -q 'S2_LORA_NNODES' "$SRC/scripts/ascend/run_s2_lora_train_npu.sh" || fail "$SRC ($COMMIT) has no multi-node launcher"
grep -q -- '--objective' "$SRC/scripts/train_system2_lora.py" || fail "$SRC ($COMMIT) has no --objective in the trainer (no DPO path)"
printf '%s node=%s src=%s commit=%s stream=%s out=%s lr=%s\n' "$(date -u +%FT%TZ)" "$NODE" "$SRC" "$COMMIT" "$STREAM" "$OUT" "$LR" >> "$LOGS/X37.commits"
log "node $NODE: src $SRC at $COMMIT, stream $STREAM, out $OUT, lr $LR"

training_here() { pgrep -u "$(id -u)" -f "[r]un_s2_lora_train_npu.sh|[t]rain_system2_lora.py" >/dev/null; }
trained() { grep -q '"event": "done"' "$OUT/train_log.jsonl" 2>/dev/null && [[ -s $OUT/adapter/adapter_model.safetensors ]]; }
stop_servers() {  # this worker's set, as swap_servers.sh stops one, then its 8 cards free
  local p waited=0
  for p in $(pgrep -u "$(id -u)" -f "^bash scripts/ascend/run_ppa_servers_npu.sh"); do
    log "stopping server set: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -TERM "$p"
  done
  while pgrep -u "$(id -u)" -f "[s]cripts/exp28/serve_910b.sh" >/dev/null; do sleep 10; done
  until cards_free; do
    (( waited > 1800 )) && return 1
    sleep 20; waited=$((waited + 20))
  done
}
cards_free() {  # all 8 cards at <= 4096 MiB HBM, read with the launcher's own parser (it prints nothing on a layout it does not know)
  local table k used
  table=$(source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; npu-smi info 2>&1) || return 1
  for k in 0 1 2 3 4 5 6 7; do
    used=$(printf '%s\n' "$table" | awk -v id="$k" -f "$SRC/scripts/ascend/npu_hbm_used_mib.awk")
    [[ -n $used ]] && (( used <= 4096 )) || return 1
  done
}
LAUNCH_ENV=(S2_LORA_ROOT=$RR S2_LORA_REPO=$SRC S2_LORA_MODE=train S2_LORA_STREAM_DIR=$STREAM S2_LORA_OUTPUT_DIR=$OUT
            S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 S2_LORA_RECORD_ROOTS= S2_LORA_LABEL_ROOTS= S2_LORA_ADAPTER=)

if [[ $NODE == 1 ]]; then
  # ------------------------------------------------------------------ node 1 (worker-1)
  next_attempt() {  # $1 run: the lowest n with go.<run>.<n> and neither exit.<run>.<n>.w1 nor abort.<run>.<n>
    local n
    for n in 1 2 3 4 5 6 7 8 9; do
      [[ -e $MN/go.$1.$n && ! -e $MN/exit.$1.$n.w1 && ! -e $MN/abort.$1.$n ]] && { echo "$n"; return 0; }
    done
    return 1
  }
  refuse() { log "attempt $1: refusing ($2)"; mn_put "abort.$1" "w1: $2"; mn_put "exit.$1.w1" "refused"; }
  status "waiting for worker-0's 16-card attempts (go.<run>.<n> in $MN)"
  while true; do
    run=$(cat "$MN/run" 2>/dev/null)
    if [[ -n $run && -e $MN/finished && "$(cut -d' ' -f1 "$MN/finished")" == "$run" ]]; then
      status "worker-0 finished the 16-card phase ($(cat "$MN/finished")); nothing left here"; exit 0
    fi
    [[ -n $run ]] && n=$(next_attempt "$run") || { sleep 15; continue; }
    K=$run.$n
    read -r master port extra < "$MN/go.$K"
    read -r p_src p_stream p_out p_commit < "$MN/paths.$K" 2>/dev/null
    [[ "$(realpath -m "$p_src") $(realpath -m "$p_stream") $(realpath -m "$p_out")" == "$(realpath -m "$SRC") $(realpath -m "$STREAM") $(realpath -m "$OUT")" ]] \
      || { refuse "$K" "worker-0 uses '$p_src $p_stream $p_out', this node '$SRC $STREAM $OUT'"; continue; }
    [[ $p_commit == "$COMMIT" ]] || { refuse "$K" "worker-0 runs commit $p_commit, this node $COMMIT"; continue; }
    status "attempt $K: freeing the cards (rendezvous $master:$port${extra:+, $extra})"
    while training_here; do sleep 15; done
    stop_servers || { refuse "$K" "this worker's cards are not free 30 min after its servers stopped"; continue; }
    mn_put "ready.$K" ready
    waited=0
    # A node 0 that died after go.K and was started again writes a new run: leave this attempt at once
    # (a new node 0 also aborts it), else node 0 times out on ready and trains on 8 cards alone.
    until [[ -e $MN/start.$K || -e $MN/abort.$K || "$(cat "$MN/run" 2>/dev/null)" != "$run" ]] || (( waited > 3600 )); do
      sleep 2; waited=$((waited + 2))
    done
    if [[ ! -e $MN/start.$K && ! -e $MN/abort.$K && "$(cat "$MN/run" 2>/dev/null)" != "$run" ]]; then
      log "attempt $K: superseded by worker-0's run $(cat "$MN/run" 2>/dev/null)"; mn_put "exit.$K.w1" "superseded"; continue
    fi
    if [[ ! -e $MN/start.$K ]]; then
      log "attempt $K: no start ($( [[ -e $MN/abort.$K ]] && cat "$MN/abort.$K" || echo "1 h"))"; mn_put "exit.$K.w1" "not started"; continue
    fi
    status "training X37 as node 1 (attempt $K)"
    ( cd "$SRC" && exec setsid env "${LAUNCH_ENV[@]}" S2_LORA_NNODES=2 S2_LORA_NODE_RANK=1 \
        S2_LORA_MASTER_ADDR="$master" S2_LORA_MASTER_PORT="$port" S2_LORA_EXTRA_ARGS="$(sed -n 1p "$MN/args.$K")" \
        bash scripts/ascend/run_s2_lora_train_npu.sh >> "$LOGS/X37.launch_node1.log" 2>&1 < /dev/null ) &
    lpid=$!
    while kill -0 "$lpid" 2>/dev/null; do
      why=""
      [[ -e $MN/abort.$K ]] && why="abort.$K ($(cat "$MN/abort.$K"))"
      if [[ -z $why && -e $MN/finished && "$(cut -d' ' -f1 "$MN/finished")" == "$run" ]] \
         && (( $(date +%s) - $(stat -c %Y "$MN/finished") > 900 )); then
        why="worker-0 finished ($(cat "$MN/finished")) 15 min ago"
      fi
      if [[ -n $why ]]; then
        log "$why: stopping node 1"; kill -TERM -- "-$lpid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$lpid" 2>/dev/null || break; sleep 5; done
        kill -KILL -- "-$lpid" 2>/dev/null
      fi
      sleep 30
    done
    wait "$lpid"; rc=$?
    (( rc == 0 )) || [[ -e $MN/abort.$K ]] || mn_put "abort.$K" "w1: node 1 exited $rc"
    mn_put "exit.$K.w1" "$rc"
    log "attempt $K: node 1 exited $rc"
  done
fi

# -------------------------------------------------------------------- node 0 (worker-0)
MASTER=$(tr '\0' '\n' < /proc/1/environ | grep '^VC_WORKER_HOSTS=' | cut -d= -f2 | cut -d, -f1)
# The pair stream: built for this recipe, complete on disk, and holding out EXP-35's scenes.  A plain
# command, not an assignment: a redirection on an assignment does not catch its substitution's stderr.
if ! python3 - "$STREAM/stream_manifest.json" "$EXP35_MANIFEST" "$PAIR_SCHEMA" \
     > "$A/stream_check.scenes" 2>> "$LOGS/train_node0.log" <<'PY'
import json, os, sys
path, exp35, schema = sys.argv[1:4]
m = json.load(open(path))
problems = []
if m.get("pairs") is not True:
    problems.append("manifest lacks pairs: true")
if m.get("schema") != schema:
    problems.append(f"schema {m.get('schema')!r}, want {schema!r}")
if int(m.get("global_batch", 0)) != 64 or int(m.get("shard_size", 0)) % 64:
    problems.append(f"global_batch {m.get('global_batch')} / shard_size {m.get('shard_size')} is not a 64-pair layout")
held = list(m.get("holdout", {}).get("scenes", []))
want = list(json.load(open(exp35))["holdout"]["scenes"])
if sorted(held) != sorted(want):
    problems.append(f"holds out {len(held)} scenes, EXP-35 held out {len(want)} others")
shards = m.get("shards", [])
if len(shards) != int(m.get("num_shards", -1)):
    problems.append(f"{len(shards)} shards listed, num_shards {m.get('num_shards')}")
d = os.path.dirname(path)
bad = [s["name"] for s in shards if not os.path.isfile(os.path.join(d, s["name"])) or os.path.getsize(os.path.join(d, s["name"])) != int(s["bytes"])]
if bad:
    problems.append(f"{len(bad)} shards missing or of another size, e.g. {bad[:3]}")
if problems:
    sys.exit("pair stream refused: " + "; ".join(problems))
n = int(m["total_samples"])
print(",".join(held))
print(f"pair stream: {n} pairs in {len(shards)} shards of {m['shard_size']} ({len(shards) % 4} mod 4), {n // 64} steps", file=sys.stderr)
PY
then
  fail "the pair stream at $STREAM is not usable (see $LOGS/train_node0.log)"
fi
H=$(sed -n 1p "$A/stream_check.scenes")
[[ -n $H ]] || fail "no held-out scenes read from $STREAM/stream_manifest.json"
# No --loss-normalization (EXP-35 had token): the ledger's loss is a mean over the 64 pairs of a step, and
# the trainer's DPO path refuses anything but its default, sample.  No --kl-beta: the reference term is the DPO's.
ARGS="--objective dpo --dpo-beta 0.5 --dpo-nll-weight 0.1 --global-batch 64 --micro-batch 1 --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 --lr $LR --warmup-ratio 0.03 --keep-stream --save-every 50 --keep-checkpoints 2 --log-every 1 --num-workers 4 --holdout-scenes $H"
# The trainer's own argument checks, on the CPU, before any server stops (an unknown flag or a combination
# the DPO path refuses ends here).  Its train()-time checks (stream vs objective, divisibility) run at launch.
PRE_TMP=$(mkdir -p "$RR/exp37/tmp/preflight" && cd "$RR/exp37/tmp/preflight" && pwd -P)
# shellcheck disable=SC2086
( cd "$SRC" && env TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PRE_TMP" PYTHONPATH="$SRC:$R/rpc/src" \
    "$PY" - --stream-dir "$STREAM" --output-dir "$OUT" --model-path "$R/InternNav_Model" --device npu $ARGS <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("train_system2_lora", "scripts/train_system2_lora.py")
mod = importlib.util.module_from_spec(spec)
# In sys.modules before it runs: @dataclass resolves KW_ONLY through sys.modules[cls.__module__].
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
a = mod.parse_args(sys.argv[1:])
keys = ("objective", "dpo_beta", "dpo_nll_weight", "lr", "warmup_ratio", "lora_r", "lora_alpha", "lora_dropout",
        "global_batch", "micro_batch", "loss_normalization", "kl_beta", "save_every", "keep_stream", "seed")
print("trainer arguments: " + " ".join(f"{k}={getattr(a, k, None)}" for k in keys))
if getattr(a, "objective", None) != "dpo" or getattr(a, "kl_beta", 0) != 0:
    sys.exit("the parsed arguments are not a DPO run without KL")
PY
) >> "$LOGS/train_node0.log" 2>&1 || fail "the trainer refused the arguments (see $LOGS/train_node0.log): $ARGS"

# 16 cards.  The handshake files carry this run's id, so a write left over from an earlier run of either
# node never lands in this one.
if ! trained && [[ ! -e $MN/fallback8 ]]; then
  [[ -n $MASTER ]] || fail "no VC_WORKER_HOSTS in the job's environment (worker-0's address for the rendezvous)"
  RUNID=$(date -u +%Y%m%dT%H%M%SZ)-$$
  # An earlier node 0's attempts that worker-1 may still be waiting on (go without its exit or an abort):
  # aborted, so it leaves them now and takes this run's go instead of holding its cards for an hour.
  for f in "$MN"/go.*; do
    [[ -e $f ]] || continue
    k=${f#"$MN"/go.}
    [[ -e $MN/exit.$k.w1 || -e $MN/abort.$k ]] || { mn_put "abort.$k" "w0: superseded by run $RUNID"; log "aborted stale attempt $k"; }
  done
  mn_put run "$RUNID"
  log "16-card phase: handshake run $RUNID, rendezvous at $MASTER"
  for attempt in 1 2 3; do
    trained && break
    while training_here; do sleep 30; done
    trained && break
    status "16-card attempt $attempt: freeing this worker's cards"
    stop_servers || { log "this worker's cards are not free 30 min after its servers stopped"; break; }
    extra=""; [[ -s $OUT/train_log.jsonl ]] && extra=" --resume"
    [[ $attempt -gt 1 ]] && sleep 120
    K=$RUNID.$attempt
    mn_put "paths.$K" "$SRC $STREAM $OUT $COMMIT"
    mn_put "args.$K" "$ARGS$extra"
    mn_put "go.$K" "$MASTER $((29710 + attempt)) $extra"
    status "16-card attempt $attempt$extra: waiting for worker-1 (ready.$K)"
    waited=0
    until [[ -e $MN/ready.$K || -e $MN/abort.$K ]] || (( waited > 1800 )); do sleep 5; waited=$((waited + 5)); done
    if [[ ! -e $MN/ready.$K ]]; then
      why=$( [[ -e $MN/abort.$K ]] && cat "$MN/abort.$K" || echo "no answer within 30 min")
      [[ -e $MN/abort.$K ]] || mn_put "abort.$K" "w0: no ready"
      log "worker-1 did not take attempt $attempt ($why); going to 8 cards"
      break
    fi
    mn_put "start.$K" go
    status "training X37 on 16 cards (attempt $attempt$extra; node 0 here, node 1 on worker-1)"
    ( cd "$SRC" && exec setsid env "${LAUNCH_ENV[@]}" S2_LORA_NNODES=2 S2_LORA_NODE_RANK=0 \
        S2_LORA_MASTER_ADDR="$MASTER" S2_LORA_MASTER_PORT=$((29710 + attempt)) S2_LORA_EXTRA_ARGS="$ARGS$extra" \
        bash scripts/ascend/run_s2_lora_train_npu.sh >> "$LOGS/X37.launch_node0.log" 2>&1 < /dev/null ) &
    lpid=$!
    while kill -0 "$lpid" 2>/dev/null; do
      if [[ -e $MN/abort.$K ]]; then
        log "abort.$K ($(cat "$MN/abort.$K")): stopping node 0"; kill -TERM -- "-$lpid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$lpid" 2>/dev/null || break; sleep 5; done
        kill -KILL -- "-$lpid" 2>/dev/null
      fi
      sleep 30
    done
    wait "$lpid"; rc=$?
    mn_put "exit.$K.w0" "$rc"
    (( rc == 0 )) || [[ -e $MN/abort.$K ]] || mn_put "abort.$K" "w0: node 0 exited $rc"
    trained && break
    [[ -e $MN/abort.$K ]] || mn_put "abort.$K" "w0: node 0 exited $rc without the done event"
    log "16-card attempt $attempt ended without the done event (node 0 exit $rc, worker-1: $(cat "$MN/exit.$K.w1" 2>/dev/null || echo running))"
    for _ in $(seq 180); do [[ -e $MN/exit.$K.w1 ]] && break; sleep 10; done
  done
  mn_put finished "$RUNID $(trained && echo trained || echo fallback)"
  if trained; then
    log "16-card training done: $(python3 -c "import json,sys;h=json.load(open(sys.argv[1]))['hyperparameters'];print('world_size', h.get('world_size'), 'grad_accum', h.get('grad_accum'))" "$OUT/manifest.json" 2>&1)"
  else
    touch "$MN/fallback8"
    [[ -d $OUT ]] && mv "$OUT" "$OUT.16cards.$(date -u +%Y%m%dT%H%M%SZ)"
    log "16 cards did not finish; training on 8 cards from the start (a new run key: world_size 8, grad_accum 8)"
  fi
fi
# The fallback: this worker's 8 cards, resuming at most twice.
for attempt in 1 2 3; do
  trained && break
  while training_here; do sleep 30; done
  stop_servers || fail "this worker's cards are not free 30 min after its servers stopped"
  extra=""; [[ -s $OUT/train_log.jsonl ]] && extra=" --resume"
  [[ $attempt -gt 1 ]] && sleep 120
  status "training X37 on 8 cards (attempt $attempt$extra)"
  ( cd "$SRC" && env "${LAUNCH_ENV[@]}" S2_LORA_EXTRA_ARGS="$ARGS$extra" \
      bash scripts/ascend/run_s2_lora_train_npu.sh >> "$LOGS/X37.launch_node0.log" 2>&1 < /dev/null )
done
trained || fail "X37 did not finish training after 3 attempts (see $LOGS/X37.launch_node0.log and $OUT/logs)"
sha=$(sha256sum "$OUT/adapter/adapter_model.safetensors" | cut -d' ' -f1)
printf '%s %s/adapter commit=%s\n' "$sha" "$OUT" "$COMMIT" > "$OUT.adapter.sha256"
status "trained: $(grep '"event": "done"' "$OUT/train_log.jsonl" | tail -1) adapter $sha"
