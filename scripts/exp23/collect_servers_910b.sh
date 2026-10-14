#!/usr/bin/env bash
# EXP-23, 910B half of round N's collection (N >= 2): once round N-1 has published its
# adapter (round_train_watch_910b.sh: all gates passed), serve the model with that
# adapter for the 4090's collection clients (scripts/exp23/collect_round_4090.sh).
#
# User, 2026-10-14: round 2 starts as soon as round 1's gates pass, on the 910B, while
# the 4090 evaluates round 1.  Runs unattended and starts nothing if round N-1 stopped
# (a failed gate leaves no adapter to collect with).  It reports in
# $ROUND_DIR/SERVERS_STATUS; $ROUND_DIR/STATUS belongs to round N's
# round_train_watch_910b.sh, which reads this file before it takes over.
#
#   EXP23_ROUND=2 nohup setsid bash scripts/exp23/collect_servers_910b.sh > log 2>&1 &
#
# Phases:
#   0. at launch: withdraw any earlier npu_servers_r<N> publication, and fail now (not
#      at the hand-off) if a port this round needs is held by anything but round N-1's
#      receivers;
#   1. wait for round N-1's STATUS "DONE: adapter_r<N-1> published";
#   2. stop round N-1's receivers (they hold the model ports), wait for the ports and cards;
#   3. run_ppa_servers_npu.sh on the cards with PPA_EVAL_SYSTEM2_LORA = the exported
#      adapter (servers.json records its sha256; the 4090 launcher checks it);
#   4. publish servers.json + logs as export/npu_servers_r<N> on the export port (start
#      serve_dirs.py there if nothing listens), check the port really serves it, and watch
#      the launcher; the publication is withdrawn whenever this server set ends, so the
#      4090 never starts against a dead set.
# Writes only under $ROOT (the user's rule for this box).

set -uo pipefail

ROOT="${EXP23_ROOT:-$HOME/work/zhr/zhr_1}"
ROUND="${EXP23_ROUND:?set EXP23_ROUND to the round being collected, e.g. 2}"
PREV=$((ROUND - 1))
SRC="${EXP23_SRC:-$ROOT/exp23/src}"
PREV_DIR="${EXP23_PREV_DIR:-$ROOT/exp23/round$PREV}"
ROUND_DIR="${EXP23_ROUND_DIR:-$ROOT/exp23/round$ROUND}"
PYTHON="${EXP23_PYTHON:-$ROOT/envs/ppa/bin/python}"
NPUS="${EXP23_COLLECT_NPUS:-0,1,2,3,4,5,6}"
MODEL_PORT_BASE="${EXP23_MODEL_PORT_BASE:-52410}"
VO_PORT_BASE="${EXP23_VO_PORT_BASE:-52510}"
EXPORT_PORT="${EXP23_EXPORT_PORT:-52417}"
SERVE_S="${EXP23_SERVE_S:-345600}"  # the servers stop on their own after 4 days
READY_S="${EXP23_READY_S:-5400}"     # the launcher's own start timeout is 3600 s, plus preflight

(( ROUND >= 2 )) || { echo "EXP23_ROUND must be >= 2" >&2; exit 2; }
mkdir -p "$ROUND_DIR" "$ROOT/logs" "$ROOT/t"
LOG="$ROUND_DIR/servers_watch.log"
PUB="$PREV_DIR/export/npu_servers_r$ROUND"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$ROUND_DIR/SERVERS_STATUS"; log "STATUS $*"; }
unpublish() { rm -rf "$PUB" "$PUB.part"; }
die() { unpublish; status "STOPPED: $*"; exit 2; }

[[ -d "$SRC/.git" ]] || die "no code checkout at $SRC"
git -C "$SRC" diff --quiet HEAD -- || die "$SRC has uncommitted changes"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
log "round=$ROUND src=$SRC commit=$COMMIT npus=$NPUS"

# 0 -------------------------------------------------------------------------------
NSLOTS=$(awk -F, '{print NF}' <<< "$NPUS")
receiver_pids() { pgrep -u "$(id -u)" -f "[s]tream_transfer\.py recv --dest $PREV_DIR/"; }
holder() { ss -Hltnp "sport = :$1" 2>/dev/null; }  # empty = free
check_ports() {  # $1 = "allow-receivers" (round N-1 may still hold the model ports) or "free"
  local rx i p h pid
  rx=" $(receiver_pids | tr '\n' ' ') "
  for ((i = 0; i < NSLOTS; i++)); do
    p=$((VO_PORT_BASE + i)); h="$(holder "$p")"
    [[ -z "$h" ]] || die "VO port $p is taken: $h"
    p=$((MODEL_PORT_BASE + i)); h="$(holder "$p")"
    [[ -z "$h" ]] && continue
    pid="$(grep -o 'pid=[0-9]*' <<< "$h" | head -1 | cut -d= -f2)"
    [[ "$1" == allow-receivers && -n "$pid" && "$rx" == *" $pid "* ]] \
      || die "model port $p is taken by something other than round $PREV's receivers: $h"
  done
}
unpublish
check_ports allow-receivers

# 1 -------------------------------------------------------------------------------
status "waiting for round $PREV to publish adapter_r$PREV"
while true; do
  prev_status="$(cut -d' ' -f2- "$PREV_DIR/STATUS" 2>/dev/null || true)"
  [[ "$prev_status" == "DONE: adapter_r$PREV published"* ]] && break
  if [[ "$prev_status" == STOPPED* ]]; then
    status "not collecting: round $PREV stopped ($prev_status)"
    exit 2
  fi
  sleep 60
done
ADAPTER="$PREV_DIR/export/adapter_r$PREV"
[[ -s "$ADAPTER/adapter_model.safetensors" && -s "$ADAPTER/adapter_config.json" ]] || die "no adapter at $ADAPTER"
ADAPTER_SHA="$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)"
log "adapter_r$PREV: ${ADAPTER_SHA:0:16}"

# 2 -------------------------------------------------------------------------------
status "stopping round $PREV's receivers"
check_ports allow-receivers
for pid in $(receiver_pids); do kill -TERM "$pid" 2>/dev/null && log "TERM receiver $pid"; done
for _ in $(seq 30); do receiver_pids >/dev/null || break; sleep 2; done
check_ports free
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
for _ in $(seq 90); do
  busy=$(npu-smi info 2>/dev/null | awk '/\/ 65536/ {split($0, a, "/"); n = a[length(a)-1]; gsub(/[^0-9]/, "", n); if (n + 0 > 4096) c++} END {print c + 0}')
  [[ "$busy" -eq 0 ]] && break
  sleep 20
done
[[ "${busy:-1}" -eq 0 ]] || die "cards still busy 30 min after round $PREV finished"

# 3 -------------------------------------------------------------------------------
STAMP="exp23r${ROUND}_$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$ROOT/logs/${STAMP}_servers.log"
RUNTIME="$ROOT/servers/$STAMP"
status "starting servers $STAMP with adapter_r$PREV"
# The trailing $STAMP is ignored by the launcher; it tags the command line so the checks
# below follow this launcher and not another one run under the same uid.
( cd "$SRC" && PPA_EVAL_REPO="$SRC" PPA_NPU_DEVICES="$NPUS" PPA_EVAL_MODEL_PORT_BASE="$MODEL_PORT_BASE" \
    PPA_EVAL_VO_PORT_BASE="$VO_PORT_BASE" PPA_NPU_TMP_ROOT="$ROOT/t" PPA_EVAL_RUN_STAMP="$STAMP" \
    PPA_NPU_RUNTIME_DIR="$RUNTIME" PPA_EVAL_SYSTEM2_LORA="$ADAPTER" \
    nohup setsid timeout "$SERVE_S" bash scripts/ascend/run_ppa_servers_npu.sh "$STAMP" > "$SERVER_LOG" 2>&1 < /dev/null & )
launcher_alive() { pgrep -u "$(id -u)" -f "[r]un_ppa_servers_npu\.sh $STAMP\$" >/dev/null && [[ ! -e "$RUNTIME/STOPPED" ]]; }
sleep 30
deadline=$(( $(date +%s) + READY_S ))
until grep -q "all servers ready" "$SERVER_LOG" 2>/dev/null; do
  launcher_alive || die "the server launcher exited before its servers were ready (see $SERVER_LOG)"
  (( $(date +%s) < deadline )) || die "servers not ready after ${READY_S}s (see $SERVER_LOG); the launcher is still running"
  sleep 30
done
[[ -s "$RUNTIME/servers.json" ]] || die "no servers.json in $RUNTIME"
grep -q "\"system2_lora_sha256\": \"$ADAPTER_SHA\"" "$RUNTIME/servers.json" \
  || die "servers.json does not carry adapter_r$PREV's sha256"

# 4 -------------------------------------------------------------------------------
mkdir -p "$PUB.part"
cp "$RUNTIME/servers.json" "$PUB.part/" && cp -r "$RUNTIME/logs" "$PUB.part/logs"
mv "$PUB.part" "$PUB"
if [[ -z "$(holder "$EXPORT_PORT")" ]]; then
  log "nothing listens on $EXPORT_PORT; starting serve_dirs on $PREV_DIR/export"
  ( cd "$PREV_DIR/export" && nohup setsid timeout "$SERVE_S" "$PYTHON" "$SRC/scripts/exp23/serve_dirs.py" \
      "$PREV_DIR/export" "$EXPORT_PORT" > "$ROUND_DIR/serve_export.log" 2>&1 < /dev/null & )
  sleep 5
fi
probe="$(mktemp -d "$ROOT/t/pubcheck.XXXXXX")"  # a fresh dir: pull_dirs.py skips names already pulled
"$PYTHON" "$SRC/scripts/exp23/pull_dirs.py" "$probe" "$EXPORT_PORT" "npu_servers_r$ROUND" >> "$LOG" 2>&1
served=0
[[ -e "$probe/npu_servers_r$ROUND/.pulled" ]] && cmp -s "$probe/npu_servers_r$ROUND/servers.json" "$PUB/servers.json" && served=1
rm -rf "$probe"
[[ "$served" -eq 1 ]] || die "port $EXPORT_PORT does not serve npu_servers_r$ROUND"
status "serving round $ROUND collection: $STAMP (published as npu_servers_r$ROUND on $EXPORT_PORT)"
while launcher_alive; do sleep 300; done
unpublish
rc="$(sed -n 's/^\[ppa-npu\] stopped (status=\([0-9]*\)).*/\1/p' "$SERVER_LOG" | tail -1)"
if [[ "$rc" == 130 ]]; then  # the launcher's trap turns INT/TERM into 130
  status "ENDED: servers stopped by a signal (round_train_watch, an operator or the ${SERVE_S}s timeout); see $SERVER_LOG"
  exit 0
fi
die "the server launcher exited with status ${rc:-unknown} (see $SERVER_LOG)"
