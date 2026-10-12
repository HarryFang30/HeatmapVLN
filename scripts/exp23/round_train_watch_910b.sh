#!/usr/bin/env bash
# EXP-23, 910B half of one round after collection: stop the collection servers, train
# from the 4090's stream, gate, and publish the adapter only if the gates pass.
#
# Runs unattended (the operator's laptop may sleep), so every phase waits on evidence
# rather than on time, writes $ROUND_DIR/STATUS, and stops for a human on anything
# unexpected.  The 4090 half is scripts/exp23/round_post_4090.sh.
#
#   EXP23_ROUND=1 nohup setsid bash scripts/exp23/round_train_watch_910b.sh > log 2>&1 &
#
# Phases:
#   1. collection over  = no established connection on the model ports for IDLE_S;
#   2. stop the server launcher (TERM; it stops its servers), wait for free cards;
#   3. receivers: the stream on STREAM_PORTS (cap STREAM_CAP_GB), the held-out records on
#      HOLDOUT_PORTS -- both loopback only, inside $ROUND_DIR;
#   4. train from the stream on all cards (scripts/ascend/run_s2_lora_train_npu.sh);
#   5. gates G0-G4 on the held-out records once HOLDOUT_DONE arrives;
#   6. all_pass -> copy the adapter to export/ and serve it on EXPORT_PORT for the 4090
#      (scripts/tools/... serve_dirs-style tar streamer); otherwise GATES_FAILED and stop.
# Writes only under $ROOT (the user's rule for this box).

set -uo pipefail

ROOT="${EXP23_ROOT:-$HOME/work/zhr/zhr_1}"
ROUND="${EXP23_ROUND:-1}"
SRC="${EXP23_SRC:-$ROOT/exp23/src_train}"
ROUND_DIR="${EXP23_ROUND_DIR:-$ROOT/exp23/round$ROUND}"
PYTHON="${EXP23_PYTHON:-$ROOT/envs/ppa/bin/python}"
MODEL_PORTS_RE="${EXP23_MODEL_PORTS_RE:-sport >= :52410 and sport <= :52416}"
IDLE_S="${EXP23_IDLE_S:-900}"
STREAM_PORTS="${EXP23_STREAM_PORTS:-52410,52411,52412,52413}"
HOLDOUT_PORTS="${EXP23_HOLDOUT_PORTS:-52414,52415,52416}"
EXPORT_PORT="${EXP23_EXPORT_PORT:-52417}"
STREAM_CAP_GB="${EXP23_STREAM_CAP_GB:-10}"
HOLDOUT_CAP_GB="${EXP23_HOLDOUT_CAP_GB:-8}"
NPUS="${EXP23_NPUS:-0,1,2,3,4,5,6,7}"
TRAIN_EXTRA="${EXP23_TRAIN_EXTRA:---global-batch 64 --save-every 100 --num-workers 4 --log-every 10}"
GATE_MAX_CALLS="${EXP23_GATE_MAX_CALLS:-3000}"

mkdir -p "$ROUND_DIR" "$ROOT/logs"
LOG="$ROUND_DIR/watch.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$ROUND_DIR/STATUS"; log "STATUS $*"; }
die() { status "STOPPED: $*"; exit 2; }

[[ -d "$SRC/.git" ]] || die "no code checkout at $SRC"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
git -C "$SRC" diff --quiet HEAD -- || die "$SRC has uncommitted changes"
log "round=$ROUND src=$SRC commit=$COMMIT"

# 1. collection over -------------------------------------------------------------
status "waiting for collection to end (no client connection for ${IDLE_S}s)"
idle_since=""
while true; do
  n=$(ss -Htn state established "( $MODEL_PORTS_RE )" 2>/dev/null | wc -l)
  if [[ "$n" -eq 0 ]]; then
    idle_since="${idle_since:-$(date +%s)}"
    (( $(date +%s) - idle_since >= IDLE_S )) && break
  else
    idle_since=""
  fi
  sleep 60
done
log "no client for ${IDLE_S}s: collection is over"

# 2. stop the servers -----------------------------------------------------------
status "stopping the collection servers"
for pid in $(pgrep -u "$(id -u)" -f '[r]un_ppa_servers_npu\.sh'); do kill -TERM "$pid" 2>/dev/null && log "TERM launcher $pid"; done
for _ in $(seq 60); do
  pgrep -u "$(id -u)" -f '[r]pc_model_server\.py|[r]pc_amb3r_vo_server\.py' >/dev/null || break
  sleep 5
done
for pid in $(pgrep -u "$(id -u)" -f '[r]pc_model_server\.py|[r]pc_amb3r_vo_server\.py'); do
  kill -KILL "$pid" 2>/dev/null && log "KILL leftover server $pid"
done
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
for _ in $(seq 60); do
  busy=$(npu-smi info 2>/dev/null | awk '/\/ 65536/ {split($0, a, "/"); n = a[length(a)-1]; gsub(/[^0-9]/, "", n); if (n + 0 > 4096) c++} END {print c + 0}')
  [[ "$busy" -eq 0 ]] && break
  sleep 10
done
[[ "${busy:-1}" -eq 0 ]] || die "cards still busy after stopping the servers"
log "all cards free"

# 3. receivers --------------------------------------------------------------------
status "receivers up; waiting for the stream"
mkdir -p "$ROUND_DIR/stream" "$ROUND_DIR/holdout"
( cd "$SRC" && nohup setsid timeout 259200 "$PYTHON" scripts/tools/stream_transfer.py recv --dest "$ROUND_DIR/stream" \
    --ports "$STREAM_PORTS" --cap-gb "$STREAM_CAP_GB" > "$ROUND_DIR/recv_stream.log" 2>&1 < /dev/null & )
( cd "$SRC" && nohup setsid timeout 259200 "$PYTHON" scripts/tools/stream_transfer.py recv --dest "$ROUND_DIR/holdout" \
    --ports "$HOLDOUT_PORTS" --cap-gb "$HOLDOUT_CAP_GB" > "$ROUND_DIR/recv_holdout.log" 2>&1 < /dev/null & )
until [[ -s "$ROUND_DIR/stream/stream_manifest.json" ]]; do sleep 30; done
log "stream manifest arrived"

# 4. train ------------------------------------------------------------------------
status "training"
( cd "$SRC" && S2_LORA_MODE=train S2_LORA_STREAM_DIR="$ROUND_DIR/stream" S2_LORA_OUTPUT_DIR="$ROUND_DIR/lora" \
    S2_LORA_NPU_DEVICES="$NPUS" S2_LORA_EXTRA_ARGS="$TRAIN_EXTRA" \
    bash scripts/ascend/run_s2_lora_train_npu.sh > "$ROUND_DIR/train.log" 2>&1 )
[[ -s "$ROUND_DIR/lora/manifest.json" && -s "$ROUND_DIR/lora/adapter/adapter_model.safetensors" ]] \
  || die "training did not produce an adapter (see $ROUND_DIR/train.log)"
log "adapter written"

# 5. gates ------------------------------------------------------------------------
status "waiting for the held-out records"
until [[ -e "$ROUND_DIR/holdout/HOLDOUT_DONE" ]]; do sleep 30; done
status "gates"
( cd "$SRC" && S2_LORA_MODE=eval S2_LORA_RECORD_ROOTS="$ROUND_DIR/holdout/records" \
    S2_LORA_LABEL_ROOTS="$ROUND_DIR/holdout/labels" S2_LORA_OUTPUT_DIR="$ROUND_DIR/lora" \
    S2_LORA_ADAPTER="$ROUND_DIR/lora/adapter" S2_LORA_NPU_DEVICES="$NPUS" \
    S2_LORA_EXTRA_ARGS="--max-calls $GATE_MAX_CALLS" \
    bash scripts/ascend/run_s2_lora_train_npu.sh > "$ROUND_DIR/gates.log" 2>&1 )
GATES="$(ls -t "$ROUND_DIR"/lora/gates_*.json 2>/dev/null | head -1)"
[[ -n "$GATES" ]] || die "no gate report (see $ROUND_DIR/gates.log)"
verdict="$("$PYTHON" -c 'import json, sys; g = json.load(open(sys.argv[1]))["gates"]; print("PASS" if g["all_pass"] else "FAIL", json.dumps({k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in g.items()}, sort_keys=True))' "$GATES")"
log "gates: $verdict"
if [[ "$verdict" != PASS* ]]; then
  cp "$GATES" "$ROUND_DIR/GATES_FAILED"
  die "gates failed ($GATES); the adapter is not published"
fi

# 6. publish --------------------------------------------------------------------
status "publishing the adapter on port $EXPORT_PORT"
EXPORT="$ROUND_DIR/export/adapter_r$ROUND"
rm -rf "$EXPORT" && mkdir -p "$EXPORT"
cp "$ROUND_DIR/lora/adapter/"* "$EXPORT/"
cp "$ROUND_DIR/lora/manifest.json" "$EXPORT/train_manifest.json"
cp "$GATES" "$EXPORT/gates.json"
for pid in $(pgrep -u "$(id -u)" -f '[s]erve_dirs\.py'); do kill -TERM "$pid" 2>/dev/null && log "stopped serve_dirs $pid"; done
sleep 2
( cd "$ROUND_DIR/export" && nohup setsid timeout 259200 "$PYTHON" "$SRC/scripts/exp23/serve_dirs.py" "$ROUND_DIR/export" \
    "$EXPORT_PORT" > "$ROUND_DIR/serve_export.log" 2>&1 < /dev/null & )
sleep 3
status "DONE: adapter_r$ROUND published on $EXPORT_PORT ($(sha256sum "$EXPORT/adapter_model.safetensors" | cut -c1-16))"
