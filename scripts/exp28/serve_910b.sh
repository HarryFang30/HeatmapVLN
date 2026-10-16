#!/usr/bin/env bash
# EXP-28, 910B half of an HM3D closed-loop collection: serve the base model (or an adapter) on the
# cards for the 4090's clients (scripts/exp28/collect_4090.sh), and publish the servers' runtime
# evidence as export/npu_servers_exp28_<tag> on the export port, where the 4090 pulls it.
#
# Same server stack as EXP-23's rounds (scripts/exp23/collect_servers_910b.sh, whose round-N
# hand-off this script does not need): run_ppa_servers_npu.sh, greedy decoding unless the caller
# exports PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE (the server logs a warning and tags every reply).
#
#   EXP28_TAG=base nohup setsid bash scripts/exp28/serve_910b.sh > log 2>&1 &
#   EXP28_TAG=c1 EXP28_ADAPTER=$HOME/work/zhr/zhr_1/exp23/round1/export/adapter_exp28_c1 \
#     nohup setsid bash scripts/exp28/serve_910b.sh > log 2>&1 &
#
# The publication is withdrawn whenever this server set ends, so the 4090 never starts against a
# dead set.  Writes only under $ROOT (the user's rule for this box).

set -uo pipefail

ROOT="${EXP23_ROOT:-$HOME/work/zhr/zhr_1}"
TAG="${EXP28_TAG:?set EXP28_TAG, e.g. base or c1}"
ADAPTER="${EXP28_ADAPTER:-}"
SRC="${EXP28_SRC:-$ROOT/exp23/src_tm}"
EXPORT_DIR="${EXP28_EXPORT_DIR:-$ROOT/exp23/round1/export}"
RUN_DIR="${EXP28_RUN_DIR:-$ROOT/exp28/serve_$TAG}"
PYTHON="${EXP23_PYTHON:-$ROOT/envs/ppa/bin/python}"
NPUS="${EXP28_NPUS:-0,1,2,3,4,5,6}"
MODEL_PORT_BASE="${EXP28_MODEL_PORT_BASE:-52410}"
VO_PORT_BASE="${EXP28_VO_PORT_BASE:-52510}"
EXPORT_PORT="${EXP28_EXPORT_PORT:-52417}"
SERVE_S="${EXP28_SERVE_S:-259200}"
READY_S="${EXP28_READY_S:-5400}"

mkdir -p "$RUN_DIR" "$ROOT/logs" "$ROOT/t"
LOG="$RUN_DIR/serve.log"
PUB="$EXPORT_DIR/npu_servers_exp28_$TAG"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$RUN_DIR/STATUS"; log "STATUS $*"; }
unpublish() { rm -rf "$PUB" "$PUB.part"; }
die() { unpublish; status "STOPPED: $*"; exit 2; }

[[ -d "$SRC/.git" ]] || die "no code checkout at $SRC"
git -C "$SRC" diff --quiet HEAD -- || die "$SRC has uncommitted changes"
log "tag=$TAG src=$SRC commit=$(git -C "$SRC" rev-parse HEAD) npus=$NPUS adapter=${ADAPTER:-none}"
if [[ -n "${PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE:-}" ]]; then
  log "System2 SAMPLES (T=$PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE top_p=${PPA_EVAL_SYSTEM2_SAMPLE_TOP_P:-1.0})"
fi
ADAPTER_SHA=""
if [[ -n "$ADAPTER" ]]; then
  [[ -s "$ADAPTER/adapter_model.safetensors" && -s "$ADAPTER/adapter_config.json" ]] || die "no adapter at $ADAPTER"
  ADAPTER_SHA="$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)"
  log "adapter ${ADAPTER_SHA:0:16}"
fi

unpublish
NSLOTS=$(awk -F, '{print NF}' <<< "$NPUS")
holder() { ss -Hltnp "sport = :$1" 2>/dev/null; }
for ((i = 0; i < NSLOTS; i++)); do
  for p in $((MODEL_PORT_BASE + i)) $((VO_PORT_BASE + i)); do
    h="$(holder "$p")"; [[ -z "$h" ]] || die "port $p is taken: $h"
  done
done
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
hbm_used() {  # "<card> <HBM MB used>" per card, from npu-smi info's two-line blocks
  npu-smi info 2>/dev/null | awk '/^\| [0-9]+ +910B/ {card = $2} /\/ 65536/ && card != "" {split($0, a, "|"); n = split(a[4], b, "/"); m = b[n - 1]; sub(/.*[ ]/, "", m); print card, m + 0; card = ""}'
}
for _ in $(seq 90); do
  busy=$(hbm_used | awk -v want=",$NPUS," 'index(want, "," $1 ",") && $2 + 0 > 4096 {c++} END {print c + 0}')
  [[ "$busy" -eq 0 ]] && break
  sleep 20
done
[[ "$busy" -eq 0 ]] || die "cards $NPUS still busy after 30 min"

STAMP="exp28${TAG}_$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$ROOT/logs/${STAMP}_servers.log"
RUNTIME="$ROOT/servers/$STAMP"
status "starting servers $STAMP"
lora_env=()
[[ -n "$ADAPTER" ]] && lora_env=(PPA_EVAL_SYSTEM2_LORA="$ADAPTER")
# The trailing $STAMP is ignored by the launcher; it tags the command line for launcher_alive.
( cd "$SRC" && env PPA_EVAL_REPO="$SRC" PPA_NPU_DEVICES="$NPUS" PPA_EVAL_MODEL_PORT_BASE="$MODEL_PORT_BASE" \
    PPA_EVAL_VO_PORT_BASE="$VO_PORT_BASE" PPA_NPU_TMP_ROOT="$ROOT/t" PPA_EVAL_RUN_STAMP="$STAMP" \
    PPA_NPU_RUNTIME_DIR="$RUNTIME" "${lora_env[@]}" \
    nohup setsid timeout "$SERVE_S" bash scripts/ascend/run_ppa_servers_npu.sh "$STAMP" > "$SERVER_LOG" 2>&1 < /dev/null & )
launcher_alive() { pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh $STAMP\$" >/dev/null && [[ ! -e "$RUNTIME/STOPPED" ]]; }
sleep 30
deadline=$(( $(date +%s) + READY_S ))
until grep -q "all servers ready" "$SERVER_LOG" 2>/dev/null; do
  launcher_alive || die "the server launcher exited before its servers were ready (see $SERVER_LOG)"
  (( $(date +%s) < deadline )) || die "servers not ready after ${READY_S}s (see $SERVER_LOG)"
  sleep 30
done
[[ -s "$RUNTIME/servers.json" ]] || die "no servers.json in $RUNTIME"
if [[ -n "$ADAPTER_SHA" ]]; then
  grep -q "\"system2_lora_sha256\": \"$ADAPTER_SHA\"" "$RUNTIME/servers.json" || die "servers.json does not carry the adapter's sha256"
else
  grep -q '"system2_lora_sha256": "[0-9a-f]' "$RUNTIME/servers.json" && die "servers.json names an adapter, the base was asked for"
fi

mkdir -p "$PUB.part"
cp "$RUNTIME/servers.json" "$PUB.part/" && cp -r "$RUNTIME/logs" "$PUB.part/logs"
mv "$PUB.part" "$PUB"
probe="$(mktemp -d "$ROOT/t/pubcheck.XXXXXX")"
"$PYTHON" "$SRC/scripts/exp23/pull_dirs.py" "$probe" "$EXPORT_PORT" "npu_servers_exp28_$TAG" >> "$LOG" 2>&1
served=0
[[ -e "$probe/npu_servers_exp28_$TAG/.pulled" ]] && cmp -s "$probe/npu_servers_exp28_$TAG/servers.json" "$PUB/servers.json" && served=1
rm -rf "$probe"
[[ "$served" -eq 1 ]] || die "port $EXPORT_PORT does not serve npu_servers_exp28_$TAG"
status "serving: $STAMP (published as npu_servers_exp28_$TAG on $EXPORT_PORT)"
while launcher_alive; do sleep 300; done
unpublish
status "ENDED: servers $STAMP stopped (see $SERVER_LOG)"
