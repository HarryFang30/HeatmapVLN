#!/usr/bin/env bash
# EXP-28 C1 on the 910B (writes only under ~/work/zhr/zhr_1).  Waits for $D/GO (touched by the operator once the 4090
# reports both collection launchers done) or the collection cutoff 10-17 14:00 UTC, then:
#   stop the base servers -> receivers (G0 records on 52413, the stream on 52414-52416) -> G0-npu parity (2,000 npu
#   calls, adapters off, >= 0.98) -> train C1 from the base (token mean, recipe defaults) -> export adapter_rX28C1
#   -> serve C1 (tag c1) for P -> act on the 4090's signals (exp28_auto.sh), which arrive through tunnel port 52517
#   in $SIG: RETRAIN_E2 (stop serving, receive the two-pass stream on 52414-52416, train, export adapter_rX28C1E2,
#   serve it as c1e2), SCALE05 (stop serving, serve $SIG/adapter_s05 as c1s05), STOP_SERVING (stop, end).
R=~/work/zhr/zhr_1; E=$R/exp23; D=$R/exp28/c1; SRC=$E/src_x28; P=$R/envs/ppa/bin/python; SIG=$R/exp28/signals
mkdir -p "$D" "$SIG"
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/c1.log"; echo "$(date -u +%FT%TZ) $*" > "$D/STATUS"; }
# The signal channel takes 52517 from our own TensorBoard (exp23/tb_tool/tb_serve.py, EXP-23's).
for p in $(pgrep -u "$(id -u)" -f "[t]b_tool/tb_serve[.]py"); do kill -TERM "$p" && log "TERM our TensorBoard $p (52517 -> signals)"; done
for _ in $(seq 30); do ss -Hltn "sport = :52517" | grep -q . || break; sleep 2; done
( cd "$SRC" && nohup setsid $P scripts/tools/stream_transfer.py recv --dest "$SIG" --ports 52517 --cap-gb 4 \
    > "$R/exp28/recv_signals.log" 2>&1 < /dev/null & )
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
busy_cards() {  # cards over 4096 MiB HBM, or 99 when the table cannot be read (the repo's parser prints nothing then)
  local t c=0 id u; t="$(npu-smi info 2>/dev/null)"
  for id in 0 1 2 3 4 5 6 7; do
    u="$(printf '%s\n' "$t" | awk -v id="$id" -f "$SRC/scripts/ascend/npu_hbm_used_mib.awk")"
    [[ "$u" =~ ^[0-9]+$ ]] || { echo 99; return; }
    (( u > 4096 )) && c=$((c + 1))
  done
  echo "$c"
}
stop_servers() {  # $1 = tag: stop that server set, then wait for its ports and its cards
  local p port busy
  for p in $(pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh exp28$1_"); do kill -TERM "$p" && log "TERM server launcher $p ($1)"; done
  for _ in $(seq 90); do
    busy=0
    for port in 52410 52411 52412 52413 52414 52415 52416; do ss -Hltn "sport = :$port" | grep -q . && busy=1; done
    [[ "$busy" -eq 0 && "$(busy_cards)" -eq 0 ]] && return 0; sleep 20
  done
  log "STOPPED: the $1 servers did not release their ports or cards in 30 min"; exit 2
}
train_stream() {  # $1 stream dir, $2 output dir: one pass over the stream, recipe defaults, token mean
  local hold
  until [[ -e "$1/stream_manifest.json" ]]; do [[ -e "$SIG/STOP_SERVING" ]] && { log "DONE: STOP_SERVING before training"; exit 0; }; sleep 30; done
  hold=$($P -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["holdout"]["scenes"]))' "$1/stream_manifest.json")
  [[ -n "$hold" ]] || { log "STOPPED: $1 names no held-out scenes"; exit 2; }
  log "training $(basename "$2") from $(basename "$1") (token mean, from the base; holdout $(tr ',' '\n' <<< "$hold" | wc -l) scenes)"
  S2_LORA_MODE=train S2_LORA_STREAM_DIR="$1" S2_LORA_OUTPUT_DIR="$2" S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 \
    S2_LORA_EXTRA_ARGS="--global-batch 64 --save-every 100 --num-workers 4 --log-every 10 --loss-normalization token --holdout-scenes $hold" \
    bash scripts/ascend/run_s2_lora_train_npu.sh > "$2.train.log" 2>&1
  grep -q '"event": "done"' "$2/train_log.jsonl" 2>/dev/null
}
export_adapter() {  # $1 trained output dir, $2 export name
  local x=$E/round1/export/$2
  rm -rf "$x.part" && mkdir -p "$x.part" && cp "$1/adapter/"* "$x.part/" && mv "$x.part" "$x"
  log "exported $x ($(sha256sum "$x/adapter_model.safetensors" | cut -c1-16))"
}
settled() {  # $1 tag: wait (<= 2 h) until that server set is serving or has ended, so a stop finds its launcher
  for _ in $(seq 240); do grep -q -E "serving|STOPPED|ENDED" "$R/exp28/serve_$1/STATUS" 2>/dev/null && return 0; sleep 30; done
  log "the $1 servers never settled (serve_$1/STATUS: $(cat "$R/exp28/serve_$1/STATUS" 2>/dev/null))"
}
serve() {  # $1 tag, $2 adapter dir
  rm -f "$R/exp28/serve_$1/STATUS"
  ( cd "$SRC" && EXP28_TAG="$1" EXP28_ADAPTER="$2" EXP28_SRC="$SRC" nohup setsid bash scripts/exp28/serve_910b.sh \
      > "$D/serve_$1.out" 2>&1 < /dev/null & )
  log "serving $1 ($(sha256sum "$2/adapter_model.safetensors" | cut -c1-16))"
}
CUTOFF=$(date -d "2026-10-17T14:00:00Z" +%s)
IDLE_FROM=$(date -d "2026-10-17T02:00:00Z" +%s)  # the cohort cannot be done before this
# Without the operator: from IDLE_FROM on, 30 consecutive minutes with no client connected to the model or VO ports
# (every 4090 client holds a tunnelled TCP connection to its slot while it runs) = the collection is over.
connected() { ss -Htn state established '( sport >= :52410 and sport <= :52416 ) or ( sport >= :52510 and sport <= :52516 )' | grep -q .; }
log "waiting for GO, 30 idle minutes after 10-17 02:00 UTC, or the cutoff"
why=""; quiet=0
until [[ -n "$why" ]]; do
  if [[ -e "$D/GO" ]]; then why=GO
  elif (( $(date +%s) >= CUTOFF )); then why=cutoff
  else
    if (( $(date +%s) >= IDLE_FROM )) && ! connected; then quiet=$((quiet + 1)); else quiet=0; fi
    if (( quiet >= 30 )); then why="no client for 30 min"; else sleep 60; fi
  fi
done
log "stopping the base servers ($why)"
for p in $(pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh exp28base_"); do kill -TERM "$p" && log "TERM server launcher $p"; done
for _ in $(seq 60); do
  busy=0
  for port in 52413 52414 52415 52416; do ss -Hltn "sport = :$port" | grep -q . && busy=1; done
  [[ "$busy" -eq 0 ]] && break; sleep 10
done
[[ "$busy" -eq 0 ]] || { log "STOPPED: ports 52413-52416 still held after 10 min"; exit 2; }
cd "$SRC" || exit 2
mkdir -p "$D/g0_records" "$D/stream"
nohup setsid $P scripts/tools/stream_transfer.py recv --dest "$D/g0_records" --ports 52413 --cap-gb 8 > "$D/recv_g0.log" 2>&1 < /dev/null &
nohup setsid $P scripts/tools/stream_transfer.py recv --dest "$D/stream" --ports 52414,52415,52416 --cap-gb 60 > "$D/recv_stream.log" 2>&1 < /dev/null &
log "receivers up; waiting for the G0 records"
until [[ -e "$D/g0_records/G0_PUSHED" ]]; do [[ -e "$SIG/STOP_SERVING" ]] && { log "DONE: STOP_SERVING before G0"; exit 0; }; sleep 30; done
log "G0-npu parity"
for _ in $(seq 90); do [[ "$(busy_cards)" -eq 0 ]] && break; sleep 20; done
[[ "$(busy_cards)" -eq 0 ]] || { log "STOPPED: cards still busy 30 min after the servers stopped ($(busy_cards))"; exit 2; }
S2_LORA_MODE=parity S2_LORA_RECORD_ROOTS="$D/g0_records/records" S2_LORA_OUTPUT_DIR="$D/g0_npu" S2_LORA_PARITY_N=2000 \
  S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 S2_LORA_EXTRA_ARGS="--parity-split all" \
  bash scripts/ascend/run_s2_lora_train_npu.sh > "$D/g0_npu.log" 2>&1
g0=$($P -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["exact_rate"], d["g0_pass"])' "$D/g0_npu/parity_g0.json" 2>/dev/null)
log "G0-npu: ${g0:-no report}"
[[ "$g0" == *True ]] || { log "STOPPED: G0-npu did not pass (${g0:-no report}; see g0_npu.log)"; exit 2; }
if ! train_stream "$D/stream" "$D/lora"; then
  log "STOPPED: training failed (see lora.train.log)"
  for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/"); do kill -TERM "$p"; done
  exit 2
fi
export_adapter "$D/lora" adapter_rX28C1
for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/"); do kill -TERM "$p"; done
sleep 5
# P's candidate arm: serve C1 on cards 0-6 for the 4090's held-out clients (they start only if the offline gates say val).
[[ -e "$SIG/STOP_SERVING" ]] && { log "DONE: STOP_SERVING before C1 was served"; exit 0; }
serve c1 "$E/round1/export/adapter_rX28C1"; TAG=c1

log "C1 served; listening for the 4090's signals in $SIG"
while :; do
  settled "$TAG"
  if [[ -e "$SIG/STOP_SERVING" ]]; then
    stop_servers "$TAG"; log "DONE: STOP_SERVING ($TAG stopped)"; exit 0
  fi
  if [[ -e "$SIG/RETRAIN_E2" && ! -e "$D/e2.started" ]]; then
    touch "$D/e2.started"; log "RETRAIN_E2"
    stop_servers "$TAG"
    mkdir -p "$D/stream_e2"
    ( cd "$SRC" && nohup setsid $P scripts/tools/stream_transfer.py recv --dest "$D/stream_e2" --ports 52414,52415,52416 \
        --cap-gb 60 > "$D/recv_stream_e2.log" 2>&1 < /dev/null & )
    if ! train_stream "$D/stream_e2" "$D/lora_e2"; then log "STOPPED: the 2-epoch training failed (lora_e2.train.log)"; exit 2; fi
    export_adapter "$D/lora_e2" adapter_rX28C1E2
    for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/"); do kill -TERM "$p"; done
    sleep 5
    serve c1e2 "$E/round1/export/adapter_rX28C1E2"; TAG=c1e2
  fi
  if [[ -e "$SIG/SCALE05" && ! -e "$D/s05.started" ]]; then
    touch "$D/s05.started"; log "SCALE05"
    [[ -s "$SIG/adapter_s05/adapter_model.safetensors" && -s "$SIG/adapter_s05/adapter_config.json" ]] \
      || { log "STOPPED: SCALE05 without the scaled adapter in $SIG/adapter_s05"; exit 2; }
    stop_servers "$TAG"
    serve c1s05 "$SIG/adapter_s05"; TAG=c1s05
  fi
  sleep 60
done
