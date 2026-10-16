#!/usr/bin/env bash
# EXP-28, 4090 half of an HM3D closed-loop collection, inside the fjl-habitat container.
# Two modes over one cohort (scripts/exp28/build_hm3d_episodes.py), each pinned to its own shards
# so a shard is always collected on the same platform:
#   external: clients here, servers on the 910B (scripts/exp28/serve_910b.sh), reached through the
#             SSH tunnel; servers.json is pulled afresh from the export port as npu_servers_exp28_<tag>;
#   local:    servers and clients on the local GPUs.
# run_s2_dagger_collect_cuda.sh does the work, with the scenes directory pointed at the EXP-28 scene
# tree (mp3d plus hm3d/<ID>/<ID>.glb and the rebaked v1 navmeshes, scripts/exp28/prepare_hm3d_scenes.py).
#
#   EXP28_MODE=external EXP28_TAG=base EXP28_WT=/workspace/exp23/wt_<commit> \
#     EXP28_COHORT_DIR=/workspace/exp28/cohorts EXP28_OUT_DIR=/workspace/exp28/collect_base \
#     EXP28_SHARDS=0-19 nohup setsid bash scripts/exp28/collect_4090.sh > /dev/null 2>&1 &
#
# STATUS carries record and shard counts every 5 min, and DEGRADED as soon as a shard fails.

set -uo pipefail

MODE="${EXP28_MODE:?set EXP28_MODE to external or local}"
WT="${EXP28_WT:?set EXP28_WT to the pinned code worktree}"
COHORT_DIR="${EXP28_COHORT_DIR:?set EXP28_COHORT_DIR}"
OUT_DIR="${EXP28_OUT_DIR:?set EXP28_OUT_DIR}"
SHARDS="${EXP28_SHARDS:?set EXP28_SHARDS (indices and ranges, e.g. 0-19)}"
TAG="${EXP28_TAG:-base}"
ADAPTER="${EXP28_ADAPTER:-}"  # a local copy of the adapter the servers must serve (C1, C2); empty = base
SCENES_DIR="${EXP28_SCENES_DIR:-/workspace/exp28/scene_datasets}"
PYTHON="${EXP28_PYTHON:-/opt/conda/bin/python}"
EXPORT_PORT="${EXP28_EXPORT_PORT:-52417}"
CLIENT_GPUS="${EXP28_CLIENT_GPUS:-5,4,0,5,4,0,5}"  # external: one entry per 910B slot
LOCAL_GPUS="${EXP28_LOCAL_GPUS:-5,4,0}"
DISPLAY_BASE="${EXP28_DISPLAY_BASE:-701}"           # evaluation uses 380 and 400+, EnvDrop 640+
MODEL_PORT_BASE="${EXP28_MODEL_PORT_BASE:-}"
VO_PORT_BASE="${EXP28_VO_PORT_BASE:-}"
export PYTHONPATH="$WT"

mkdir -p "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/collect_$MODE.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$OUT_DIR/STATUS_$MODE"; log "STATUS $*"; }
die() { status "STOPPED: $*"; chown -R 1015:1015 "$OUT_DIR" 2>/dev/null; exit 2; }
git -c safe.directory='*' -C "$WT" diff --quiet HEAD -- || die "$WT has uncommitted changes"
[[ -s "$COHORT_DIR/manifest.json" ]] || die "no cohort at $COHORT_DIR"
[[ -d "$SCENES_DIR/mp3d" && -d "$SCENES_DIR/hm3d" ]] || die "$SCENES_DIR lacks mp3d/ or hm3d/"
log "mode=$MODE tag=$TAG wt=$WT commit=$(git -c safe.directory='*' -C "$WT" rev-parse HEAD) shards=$SHARDS adapter=${ADAPTER:-none}"

declare -a env_mode
if [[ "$MODE" == external ]]; then
  SERVERS="$OUT_DIR/npu_servers/npu_servers_exp28_$TAG"
  [[ -e "$SERVERS" ]] && mv "$SERVERS" "$SERVERS.stale_$(date +%Y%m%d_%H%M%S)"
  rm -rf "$OUT_DIR/npu_servers/.npu_servers_exp28_$TAG.part"
  status "waiting for the 910B's servers (npu_servers_exp28_$TAG)"
  until [[ -e "$SERVERS/.pulled" ]]; do
    "$PYTHON" "$WT/scripts/exp23/pull_dirs.py" "$OUT_DIR/npu_servers" "$EXPORT_PORT" "npu_servers_exp28_$TAG" \
      >> "$OUT_DIR/logs/pull_servers.log" 2>&1
    [[ -e "$SERVERS/.pulled" ]] || sleep 120
  done
  [[ -s "$SERVERS/servers.json" ]] || die "the pulled server directory has no servers.json"
  slots="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["slots"])' "$SERVERS/servers.json")"
  IFS=',' read -r -a gpus <<< "$CLIENT_GPUS"
  [[ "${#gpus[@]}" -eq "$slots" ]] || die "EXP28_CLIENT_GPUS has ${#gpus[@]} entries, the 910B serves $slots slots"
  env_mode=(PPA_EVAL_EXTERNAL_SERVERS=1 PPA_EVAL_EXTERNAL_SERVER_DIR="$SERVERS" PPA_EVAL_GPU_DEVICES="$CLIENT_GPUS"
            PPA_EVAL_MODEL_PORT_BASE="${MODEL_PORT_BASE:-52410}" PPA_EVAL_VO_PORT_BASE="${VO_PORT_BASE:-52510}")
elif [[ "$MODE" == local ]]; then
  env_mode=(PPA_EVAL_GPU_DEVICES="$LOCAL_GPUS" PPA_EVAL_MODEL_PORT_BASE="${MODEL_PORT_BASE:-52470}"
            PPA_EVAL_VO_PORT_BASE="${VO_PORT_BASE:-52570}")
else
  die "EXP28_MODE must be external or local, got $MODE"
fi
[[ -n "$ADAPTER" ]] && env_mode+=(PPA_EVAL_SYSTEM2_LORA="$ADAPTER")

STAMP="x28${MODE:0:3}${TAG}_$(date +%Y%m%d_%H%M%S)"
FAILED_DIR="$OUT_DIR/runtime/$STAMP/failed"  # run_s2_dagger_collect_cuda.sh: one file per failed shard
status "collecting ($STAMP)"
chown -R 1015:1015 "$OUT_DIR"
( cd "$WT" && env PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$WT" PPA_COLLECT_COHORT_DIR="$COHORT_DIR" \
    PPA_COLLECT_OUTPUT_ROOT="$OUT_DIR" PPA_COLLECT_SHARDS="$SHARDS" PPA_EVAL_SCENES_DIR="$SCENES_DIR" \
    PPA_EVAL_DISPLAY_BASE="$DISPLAY_BASE" PPA_EVAL_SHARD_RETRIES=5 PPA_EVAL_RUN_STAMP="$STAMP" "${env_mode[@]}" \
    bash scripts/run_s2_dagger_collect_cuda.sh > "$OUT_DIR/logs/launcher_$MODE.log" 2>&1 ) &
lpid=$!
last=""
while kill -0 "$lpid" 2>/dev/null; do
  records=$(find "$OUT_DIR/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l)
  done_shards=$(find "$OUT_DIR/workers" -mindepth 2 -maxdepth 2 -name DONE 2>/dev/null | wc -l)
  failed=$(ls "$FAILED_DIR" 2>/dev/null | sed 's/^shard_//' | paste -sd, -)
  msg="collecting: $records records, $done_shards shards done"
  [[ -n "$failed" ]] && msg="DEGRADED: $msg; failed shard(s) $failed (logs/launcher_$MODE.log)"
  [[ "$msg" == "$last" ]] || { status "$msg"; last="$msg"; }
  sleep 300
done
wait "$lpid"
rc=$?
records=$(find "$OUT_DIR/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l)
chown -R 1015:1015 "$OUT_DIR"
[[ "$rc" -eq 0 ]] || die "the collection launcher exited with $rc after $records records (see logs/launcher_$MODE.log)"
status "DONE: $records records"
