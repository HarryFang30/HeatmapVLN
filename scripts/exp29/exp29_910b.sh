#!/usr/bin/env bash
# EXP-29 V1 on the 910B (writes only under ~/work/zhr/zhr_1): train EXP-25 A's own stream (pushed into $D/stream on
# 52414-52416) from the base with --loss-normalization token, check the run really used token mean, export
# adapter_rX29V1 on the export port (52417) for the 4090's gates.  Status in $D/STATUS.
R=~/work/zhr/zhr_1; E=$R/exp23; D=$R/exp29; SRC=$E/src_x28; P=$R/envs/ppa/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/exp29.log"; echo "$(date -u +%FT%TZ) $*" > "$D/STATUS"; }
cd "$SRC" || exit 2
log "waiting for the stream manifest"
until [[ -e "$D/stream/stream_manifest.json" ]]; do sleep 30; done
HOLD=$($P -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["holdout"]["scenes"]))' "$D/stream/stream_manifest.json")
[[ -n "$HOLD" ]] || { log "STOPPED: the manifest names no held-out scenes"; exit 2; }
log "training V1 (A's stream, token mean, from the base; holdout $HOLD)"
S2_LORA_MODE=train S2_LORA_STREAM_DIR="$D/stream" S2_LORA_OUTPUT_DIR="$D/lora" S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 \
  S2_LORA_EXTRA_ARGS="--global-batch 64 --save-every 100 --num-workers 4 --log-every 10 --loss-normalization token --holdout-scenes $HOLD" \
  bash scripts/ascend/run_s2_lora_train_npu.sh > "$D/train.log" 2>&1
grep -q '"event": "done"' "$D/lora/train_log.jsonl" 2>/dev/null || { log "STOPPED: training failed (train.log)"; exit 2; }
# the trainer's default is per-sample: the run must say token
grep -q '"loss_normalization": "token"' "$D/lora/manifest.json" "$D/lora/train_log.jsonl" 2>/dev/null \
  || { log "STOPPED: the run does not record loss_normalization=token (manifest.json / train_log.jsonl)"; exit 2; }
X=$E/round1/export/adapter_rX29V1
rm -rf "$X.part" && mkdir -p "$X.part" && cp "$D/lora/adapter/"* "$X.part/" && mv "$X.part" "$X"
for p in $(pgrep -u "$(id -u)" -f "stream_transfer[.]py recv --dest $D/stream"); do kill -TERM "$p"; done
log "DONE: exported $X ($(sha256sum "$X/adapter_model.safetensors" | cut -c1-16)), $(grep -c '"event": "step"' "$D/lora/train_log.jsonl") step rows"
