#!/usr/bin/env bash
# EXP-28 C1 on the 910B (writes only under ~/work/zhr/zhr_1).  Waits for $D/GO (touched by the operator once the 4090
# reports both collection launchers done) or the collection cutoff 10-17 14:00 UTC, then:
#   stop the base servers -> receivers (G0 records on 52413, the stream on 52414-52416) -> G0-npu parity (2,000 npu
#   calls, adapters off, >= 0.98) -> train C1 from the base (token mean, recipe defaults) -> export adapter_rX28C1
#   -> serve C1 (tag c1) for P.
R=~/work/zhr/zhr_1; E=$R/exp23; D=$R/exp28/c1; SRC=$E/src_x28; P=$R/envs/ppa/bin/python
mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/c1.log"; echo "$(date -u +%FT%TZ) $*" > "$D/STATUS"; }
CUTOFF=$(date -d "2026-10-17T14:00:00Z" +%s)
IDLE_FROM=$(date -d "2026-10-17T02:00:00Z" +%s)  # the cohort cannot be done before this
# Without the operator: from IDLE_FROM on, no VO server has logged for 30 min = the 4090's clients are gone.
idle() { (( $(date +%s) >= IDLE_FROM )) && [[ -z "$(find $R/servers/exp28base_*/logs -name 'vo_*.log' -mmin -30 2>/dev/null)" ]]; }
log "waiting for GO, idle servers after 10-17 02:00 UTC, or the cutoff"
why=""
until [[ -n "$why" ]]; do
  if [[ -e "$D/GO" ]]; then why=GO; elif (( $(date +%s) >= CUTOFF )); then why=cutoff; elif idle; then why="idle 30 min"; else sleep 60; fi
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
until [[ -e "$D/g0_records/G0_PUSHED" ]]; do sleep 30; done
log "G0-npu parity"
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
for _ in $(seq 90); do [[ "$(busy_cards)" -eq 0 ]] && break; sleep 20; done
[[ "$(busy_cards)" -eq 0 ]] || { log "STOPPED: cards still busy 30 min after the servers stopped ($(busy_cards))"; exit 2; }
S2_LORA_MODE=parity S2_LORA_RECORD_ROOTS="$D/g0_records/records" S2_LORA_OUTPUT_DIR="$D/g0_npu" S2_LORA_PARITY_N=2000 \
  S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 S2_LORA_EXTRA_ARGS="--parity-split all" \
  bash scripts/ascend/run_s2_lora_train_npu.sh > "$D/g0_npu.log" 2>&1
g0=$($P -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["exact_rate"], d["g0_pass"])' "$D/g0_npu/parity_g0.json" 2>/dev/null)
log "G0-npu: ${g0:-no report}"
[[ "$g0" == *True ]] || { log "STOPPED: G0-npu did not pass (${g0:-no report}; see g0_npu.log)"; exit 2; }
log "waiting for the stream manifest"
until [[ -e "$D/stream/stream_manifest.json" ]]; do sleep 30; done
HOLD=$($P -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["holdout"]["scenes"]))' "$D/stream/stream_manifest.json")
[[ -n "$HOLD" ]] || { log "STOPPED: the stream manifest names no held-out scenes"; exit 2; }
log "training C1 (token mean, from the base; holdout $(tr ',' '\n' <<< "$HOLD" | wc -l) scenes)"
S2_LORA_MODE=train S2_LORA_STREAM_DIR="$D/stream" S2_LORA_OUTPUT_DIR="$D/lora" S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 \
  S2_LORA_EXTRA_ARGS="--global-batch 64 --save-every 100 --num-workers 4 --log-every 10 --loss-normalization token --holdout-scenes $HOLD" \
  bash scripts/ascend/run_s2_lora_train_npu.sh > "$D/train.log" 2>&1
if grep -q '"event": "done"' "$D/lora/train_log.jsonl" 2>/dev/null; then
  X=$E/round1/export/adapter_rX28C1; rm -rf "$X.part" && mkdir -p "$X.part" && cp "$D/lora/adapter/"* "$X.part/" && mv "$X.part" "$X"
  log "exported $X ($(sha256sum "$X/adapter_model.safetensors" | cut -c1-16))"
else
  log "STOPPED: training failed (see train.log)"
  for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/"); do kill -TERM "$p"; done
  exit 2
fi
for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/"); do kill -TERM "$p"; done
sleep 5
# P's candidate arm: serve C1 on cards 0-6 for the 4090's held-out clients (they start only if the offline gates say val).
EXP28_TAG=c1 EXP28_ADAPTER="$X" EXP28_SRC="$SRC" nohup setsid bash scripts/exp28/serve_910b.sh > "$D/serve_c1.out" 2>&1 < /dev/null &
log "DONE: exported $X ($(sha256sum "$X/adapter_model.safetensors" | cut -c1-16)); serving C1 for P"
