#!/usr/bin/env bash
# EXP-37, worker-0 after X37's training: the offline gate's 910B half (ledger EXP-37 判据 离线门), on this
# worker's 8 cards, one process per card, each retried twice (both tools resume their rows):
#   1. G1: scripts/exp37/eval_pairs.py score on the held-out pairs ($R/exp37/heldout_pairs, a pair stream),
#      then summarize -> $G/g1/heldout_summary.json (valid, g1.pass);
#   2. only if G1 fails (a valid summary, g1.pass false): the same score on the first $X37_TRAIN_PAIRS_MAX training pairs (the pair
#      stream, sha256 order) and summarize -> $G/g1/train_summary.json, the training reward accuracy that the
#      one retry's rule reads (G1 fails alone and the training accuracy < 0.70 -> lr 5e-5);
#   3. G2 (and the report-only O4 correction, O5, L): scripts/exp28/hm3d_route_gate.py decode --on-mode
#      base-stop --shuffle-instructions on the HM3D-20 held-out records copied to $G/hm3d_heldout (the command
#      documented in hm3d_route_gate.py's module doc) -> $G/o4/hm3d.decode.shardKKof08.jsonl;
#   4. publish the decode rows, the G1 summaries and the adapter as exp37_gate on 52418 (any stop publishes
#      exp37_gate_failed with this script's STATUS instead).  hm3d_route_gate.py score needs habitat_sim and
#      the navmeshes, so it and scripts/exp37/gate_verdict.py run on the 4090.
# Waits for X37's training to finish (its done event and adapter) and for no training process on this
# worker; refuses while a server set holds the cards unless X37_GATE_STOP_SERVERS=1 (then it stops it as
# swap_servers.sh does).  Re-runnable: a step is skipped only when its summary is valid and of this adapter
# (rows resumed otherwise); G1's outcome is read from the summary JSON, never from summarize's exit code (a
# Python crash also exits 1).  Refuses a gate directory holding another adapter's summaries.
# Writes only under $R.
#   nohup setsid bash $R/exp37/src_x37/scripts/exp37/run_gate_910b.sh >> $R/exp37/logs/gate.out 2>&1 < /dev/null &
# X37_OUT / X37_A / X37_GATE_DIR for the retry (attempt 2): e.g. X37_OUT=$R/exp37/train/X37_lr5e-5
# X37_A=$R/exp37/auto_lr5e-5 X37_GATE_DIR=$R/exp37/gate_lr5e-5 (the held-out inputs stay in $R/exp37/gate).
set -uo pipefail
R=/mnt/sfs_turbo/zhr/zhr_1
SRC=${X37_SRC:-$R/exp37/src_x37}
A=${X37_A:-$R/exp37/auto}
OUT=${X37_OUT:-$R/exp37/train/X37}
STREAM=${X37_STREAM:-$R/exp37/stream/main}
PAIRS=${X37_HELDOUT_PAIRS:-$R/exp37/heldout_pairs}
HM3D=${X37_HM3D_DIR:-$R/exp37/gate/hm3d_heldout}  # records/, labels/, heldout_scenes.txt, copied from the 4090
G=${X37_GATE_DIR:-$R/exp37/gate}
TRAIN_MAX=${X37_TRAIN_PAIRS_MAX:-2000}
EXPORT=$R/exp23/round1/export
LOGS=$R/exp37/logs
PY=$R/envs/ppa/bin/python
ADAPTER=$OUT/adapter
mkdir -p "$LOGS" "$G/g1" "$G/o4" "$R/exp37/tmp"
RR=$(cd "$R" && pwd -P)  # TMPDIRs on a real path: CANN 8.5.2 refuses every JIT compile under a link (E40021)
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGS/gate.log"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$G/STATUS"; log "STATUS $*"; }
publish() {  # $1 name $2.. files or dirs: a directory on the export port, swapped in whole
  local name=$1; shift
  rm -rf "$EXPORT/$name" "$EXPORT/$name.part"; mkdir -p "$EXPORT/$name.part"
  cp -r "$@" "$EXPORT/$name.part/" && mv "$EXPORT/$name.part" "$EXPORT/$name"
}
fail() { status "STOPPED: $*"; publish exp37_gate_failed "$G/STATUS"; exit 2; }
git_src() { git -c safe.directory="$SRC" -C "$SRC" "$@"; }

[[ "$(hostname)" == *-worker-0 ]] || { log "this is $(hostname), not worker-0; refusing (nothing published)"; exit 2; }
old=$(cat "$G/.run_gate.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "run_gate_910b.sh"; then
  log "refused: run_gate_910b.sh runs as $old"; exit 1
fi
echo $$ > "$G/.run_gate.pid"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
unset ASCEND_RT_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE
[[ -d $SRC/.git ]] || fail "no code checkout at $SRC"
git_src diff --quiet HEAD -- && [[ -z "$(git_src status --porcelain)" ]] || fail "$SRC is not a clean checkout"
COMMIT=$(git_src rev-parse HEAD)
[[ -s $SRC/scripts/exp37/eval_pairs.py ]] || fail "$SRC ($COMMIT) has no scripts/exp37/eval_pairs.py"
grep -q -- '--on-mode' "$SRC/scripts/exp28/hm3d_route_gate.py" || fail "$SRC ($COMMIT) hm3d_route_gate.py has no --on-mode"
[[ -s $PAIRS/stream_manifest.json ]] || fail "no held-out pair stream at $PAIRS"
for f in records labels heldout_scenes.txt; do [[ -e $HM3D/$f ]] || fail "no $HM3D/$f (copy the HM3D held-out records and labels from the 4090 first)"; done
log "gate: src $SRC at $COMMIT, adapter $ADAPTER, held-out pairs $PAIRS, HM3D $HM3D, out $G"

trained() { grep -q '"event": "done"' "$OUT/train_log.jsonl" 2>/dev/null && [[ -s $ADAPTER/adapter_model.safetensors ]]; }
busy_here() { pgrep -u "$(id -u)" -f "[r]un_s2_lora_train_npu.sh|[t]rain_system2_lora.py" >/dev/null; }
status "waiting for X37's training ($OUT)"
until trained; do
  grep -q STOPPED "$A/STATUS_node0" 2>/dev/null && fail "training stopped: $(cat "$A/STATUS_node0")"
  sleep 120
done
while busy_here; do sleep 30; done
SHA=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
listed=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['adapter']['files']['adapter_model.safetensors'])" "$OUT/manifest.json" 2>/dev/null)
[[ $listed == "$SHA" ]] || fail "$ADAPTER/adapter_model.safetensors is $SHA, $OUT/manifest.json lists '${listed:-nothing}'"
log "adapter $SHA"
summary_state() {  # $1 a summarize --output: pass | fail (valid, this adapter) | invalid | other (another adapter's) | none
  python3 - "$1" "$SHA" 2>/dev/null <<'PY' || echo none
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("none"); sys.exit(0)
sha = d.get("adapter_sha256")
if sha and sha != sys.argv[2]:
    print("other")
elif d.get("valid") is not True or sha != sys.argv[2] or not isinstance(d.get("g1"), dict):
    print("invalid")
else:
    print("pass" if d["g1"].get("pass") is True else "fail")
PY
}
for f in "$G/g1/heldout_summary.json" "$G/g1/train_summary.json"; do
  [[ $(summary_state "$f") == other ]] && fail "$f is another adapter's summary, not $SHA's: give this attempt its own X37_GATE_DIR"
done
for f in "$G"/g1/*.rows.shard*of08.jsonl; do  # eval_pairs.py score refuses these too, but only after 3 tries per card
  [[ -s $f ]] || continue
  head -1 "$f" | python3 -c "import json,sys;sys.exit(json.loads(sys.stdin.readline()).get('adapter_sha256')!=sys.argv[1])" "$SHA" 2>/dev/null || fail "$f holds rows of another adapter than $SHA: give this attempt its own X37_GATE_DIR"
done

# The cards: no server set here (unless asked to stop it), all 8 at <= 4096 MiB.
servers=$(pgrep -u "$(id -u)" -f "^bash scripts/ascend/run_ppa_servers_npu.sh")
if [[ -n $servers ]]; then
  [[ ${X37_GATE_STOP_SERVERS:-} == 1 ]] || fail "a server set holds this worker's cards (launcher $servers); X37_GATE_STOP_SERVERS=1 stops it"
  for p in $servers; do log "stopping server set: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -TERM "$p"; done
  while pgrep -u "$(id -u)" -f "[s]cripts/exp28/serve_910b.sh" >/dev/null; do sleep 10; done
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
cards_free() {
  local table k used
  table=$(npu-smi info 2>&1) || return 1
  for k in 0 1 2 3 4 5 6 7; do
    used=$(printf '%s\n' "$table" | awk -v id="$k" -f "$SRC/scripts/ascend/npu_hbm_used_mib.awk")
    [[ -n $used ]] && (( used <= 4096 )) || return 1
  done
}
waited=0
until cards_free; do
  (( waited > 1800 )) && fail "this worker's cards are not free after 30 min"
  sleep 20; waited=$((waited + 20))
done

on_cards() {  # $1 label, then the tool's arguments with @K@ for the card: one process per card, 3 tries each
  local label=$1; shift
  run_card() {
    local k=$1 tmp try; shift
    tmp=$(mkdir -p "$RR/exp37/tmp/gate.$k" && cd "$RR/exp37/tmp/gate.$k" && pwd -P)
    for try in 1 2 3; do
      ( cd "$SRC" && env ASCEND_RT_VISIBLE_DEVICES="$k" TMPDIR="$tmp" PYTHONPATH="$SRC:$R/rpc/src" \
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "${@//@K@/$k}" >> "$G/${label}_$k.log" 2>&1 < /dev/null ) && return 0
      sleep 60
    done
    return 1
  }
  local pids=() k p failed=0
  for k in 0 1 2 3 4 5 6 7; do run_card "$k" "$@" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  (( failed == 0 )) || fail "$failed of 8 cards failed $label 3 times (see $G/${label}_*.log)"
}

# --- 1. G1 on the held-out pairs (scored again, rows resumed, unless a valid summary of this adapter exists)
g1=$(summary_state "$G/g1/heldout_summary.json")
if [[ $g1 != pass && $g1 != fail ]]; then
  status "G1: scoring the held-out pairs on 8 cards"
  on_cards g1_heldout scripts/exp37/eval_pairs.py score --pairs "$PAIRS" --adapter "$ADAPTER" \
    --model-path "$R/InternNav_Model" --shard @K@/8 --device npu --output "$G/g1/heldout.rows.shard0@K@of08.jsonl"
  rm -f "$G/g1/heldout_summary.json"
  ( cd "$SRC" && PYTHONPATH="$SRC:$R/rpc/src" "$PY" scripts/exp37/eval_pairs.py summarize --rows "$G/g1/heldout.rows.shard*of08.jsonl" \
      --adapter "$ADAPTER" --pairs "$PAIRS" --output "$G/g1/heldout_summary.json" >> "$G/g1_summarize.log" 2>&1 )
  log "G1 summarize exited $?"
  g1=$(summary_state "$G/g1/heldout_summary.json")
fi
[[ $g1 == pass || $g1 == fail ]] || fail "G1's summary is $g1 (see $G/g1_summarize.log and $G/g1/heldout_summary.json)"
log "G1 $( [[ $g1 == pass ]] && echo passes || echo fails ) ($G/g1/heldout_summary.json)"

# --- 2. the training pairs' reward accuracy, only for the retry rule
if [[ $g1 == fail ]]; then
  g1t=$(summary_state "$G/g1/train_summary.json")
  if [[ $g1t != pass && $g1t != fail ]]; then
    status "G1 failed: scoring the first $TRAIN_MAX training pairs for the retry rule"
    on_cards g1_train scripts/exp37/eval_pairs.py score --pairs "$STREAM" --adapter "$ADAPTER" \
      --model-path "$R/InternNav_Model" --shard @K@/8 --device npu --max-pairs "$TRAIN_MAX" \
      --output "$G/g1/train.rows.shard0@K@of08.jsonl"
    rm -f "$G/g1/train_summary.json"
    ( cd "$SRC" && PYTHONPATH="$SRC:$R/rpc/src" "$PY" scripts/exp37/eval_pairs.py summarize --rows "$G/g1/train.rows.shard*of08.jsonl" \
        --adapter "$ADAPTER" --pairs "$STREAM" --max-pairs "$TRAIN_MAX" --output "$G/g1/train_summary.json" \
        >> "$G/g1_summarize.log" 2>&1 )
    log "training pairs' summarize exited $?"
    g1t=$(summary_state "$G/g1/train_summary.json")
  fi
  [[ $g1t == pass || $g1t == fail ]] || fail "the training pairs' summary is $g1t (see $G/g1_summarize.log and $G/g1/train_summary.json)"
fi

# --- 3. G2: the HM3D held-out states decoded as X37 is deployed
status "G2: decoding the HM3D held-out states (--on-mode base-stop) on 8 cards"
on_cards o4 scripts/exp28/hm3d_route_gate.py decode --record-root "$HM3D/records" --label-root "$HM3D/labels" \
  --holdout-scenes-file "$HM3D/heldout_scenes.txt" --adapter "$ADAPTER" --model-path "$R/InternNav_Model" \
  --output "$G/o4/hm3d.json" --shard @K@/8 --shuffle-instructions --on-mode base-stop --device npu

# --- 4. publish for the 4090's score and verdict
files=("$G"/o4/hm3d.decode.shard*of08.jsonl "$G/g1/heldout_summary.json" "$ADAPTER")
[[ $g1 == fail ]] && files+=("$G/g1/train_summary.json")  # valid and of this adapter (step 2)
rm -rf "$EXPORT/exp37_gate_failed"
publish exp37_gate "${files[@]}" || fail "could not publish exp37_gate"
status "gate decoded (adapter $SHA, commit $COMMIT); rows, G1 summaries and adapter published as exp37_gate on 52418"
