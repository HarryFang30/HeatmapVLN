#!/usr/bin/env bash
# EXP-23 System2 DAgger collection on the R2R-CE train split
# (docs/experiments/exp23-s2-dagger-design.md section 2 A).
#
# The deployed main model runs closed loop on train episodes exactly as
# scripts/run_ppa_r2r_val_unseen_cuda.sh evaluates it -- the same client flags,
# deterministic sampling, online AMB3R-VO, llvmpipe rendering and the 500-step cap --
# and the client writes every System2 call to --s2_record_dir: one tar per episode,
# failed ones included (src/data/s2_records.py).  What differs from the evaluation:
#   - the cohort: route-disjoint train shards from scripts/tools/build_r2r_train_cohorts.py
#     (shard_NNN.json + dataset_shard_NNN.json.gz + manifest.json), any number of them;
#   - shards are claimed, not dealt: every slot takes the next shard that is neither done
#     nor claimed, so a launcher on external Ascend servers and one on this box's own GPUs
#     can work through the same cohort side by side (claims assume one pid namespace,
#     i.e. both launchers run in the same container);
#   - no merge: a shard is DONE when its progress.json holds exactly its list and every
#     row points at a committed record; the run ends with a summary of the requested shards.
#
# Servers: local CUDA (one model + one VO server per GPU in PPA_EVAL_GPU_DEVICES, as in
# the evaluation) or external (PPA_EVAL_EXTERNAL_SERVERS=1: servers on the Ascend box
# reached through the SSH tunnel on 127.0.0.1:<port>, servers.json checked exactly as the
# evaluation launcher checks it; PPA_EVAL_GPU_DEVICES then only sets the clients'
# CUDA_VISIBLE_DEVICES, they render on the CPU, so IDs may repeat).
#
# Meant to run inside the fjl-habitat container (paths below are container paths).
# Example, local servers on GPU 5, two shards:
#   PPA_COLLECT_COHORT_DIR=/workspace/exp23/round1/cohorts PPA_COLLECT_OUTPUT_ROOT=/workspace/exp23/round1 \
#     PPA_EVAL_GPU_DEVICES=5 PPA_COLLECT_SHARDS=0,1 bash scripts/run_s2_dagger_collect_cuda.sh

set -Eeuo pipefail

ROOT="${PPA_EVAL_ROOT:-/workspace}"
REPO="${PPA_EVAL_REPO:-$ROOT/HeatmapVLN}"
RPC_ROOT="${PPA_EVAL_RPC_ROOT:-$ROOT/rpc}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$ROOT/InternNav_Model}"
AMB3R_ROOT="${PPA_EVAL_AMB3R_ROOT:-$ROOT/amb3r}"
DA3_CHECKPOINT="${PPA_EVAL_DA3_CHECKPOINT:-$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE}"
PYTHON="${PPA_EVAL_PYTHON:-/opt/conda/bin/python}"
CLIENT_PYTHON="${PPA_EVAL_CLIENT_PYTHON:-$PYTHON}"
XVFB_BIN="${PPA_EVAL_XVFB:-$(command -v Xvfb || true)}"

PPA_CHECKPOINT="${PPA_EVAL_CHECKPOINT:-$ROOT/weights/ppa_refine_v2_best.pth}"
PPA_CONFIG="${PPA_EVAL_CONFIG:-$REPO/configs/ppa_action_refine_v2_8gpu.yaml}"
# Only for PYTHONPATH parity with the certified evaluation; nothing here reads its cohorts.
LOCKED_PLAN="${PPA_EVAL_LOCKED_PLAN:-$ROOT/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802}"
# Must contain mp3d/<scene>/<scene>.glb (episode scene ids are relative to it).
SCENES_DIR="${PPA_EVAL_SCENES_DIR:-/dataset}"
# Round 2 collects with round 1's adapter (rpc_model_server.py --system2_lora_adapter);
# local servers only, external servers are started with it on their own side.
SYSTEM2_LORA="${PPA_EVAL_SYSTEM2_LORA:-}"

COHORT_DIR="${PPA_COLLECT_COHORT_DIR:-}"
COLLECT_ROOT="${PPA_COLLECT_OUTPUT_ROOT:-}"
SHARD_SPEC="${PPA_COLLECT_SHARDS:-all}"
# The 500-step cap and K=8 are the evaluation's; a different cap cuts off exactly the
# lingering states the collection is for (design section 2 A), so neither is a knob.
MAX_STEPS=500
NUM_HISTORY=8

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
CLIENT="$REPO/scripts/evaluation/r2r_val_unseen.py"

PROTOCOL_SEED="${PPA_EVAL_PROTOCOL_SEED:-42}"
GPU_CSV="${PPA_EVAL_GPU_DEVICES:-0}"
VO_GPU_CSV="${PPA_EVAL_VO_GPU_DEVICES:-$GPU_CSV}"
EXTERNAL_SERVERS="${PPA_EVAL_EXTERNAL_SERVERS:-0}"
EXTERNAL_SERVER_DIR="${PPA_EVAL_EXTERNAL_SERVER_DIR:-}"
# Restarts of a client that dies mid-shard; --resume makes one cost at most the episode
# in flight (whose record was never committed).  Collection runs unattended.
SHARD_RETRIES="${PPA_EVAL_SHARD_RETRIES:-3}"
RETRY_PAUSE_S="${PPA_COLLECT_RETRY_PAUSE_S:-15}"
ALLOW_COMMIT_MISMATCH="${PPA_EVAL_ALLOW_COMMIT_MISMATCH:-0}"
RUN_STAMP="${PPA_EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"

MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"
DISPLAY_BASE="${PPA_EVAL_DISPLAY_BASE:-360}"
SERVER_START_TIMEOUT_S="${PPA_EVAL_SERVER_START_TIMEOUT_S:-1800}"
SERVER_STAGGER_S="${PPA_EVAL_SERVER_STAGGER_S:-15}"
RPC_TIMEOUT_MS="${PPA_EVAL_RPC_TIMEOUT_MS:-600000}"

die() { printf '[s2-collect] ERROR: %s\n' "$*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || die "missing file: $1"; }
# The same digest the model server reports for a System2 LoRA (hashlib over the file).
file_sha256() {
  "$CLIENT_PYTHON" -c 'import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
print(digest.hexdigest())' "$1"
}
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
tcp_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

[[ -n "$COHORT_DIR" ]] || die "PPA_COLLECT_COHORT_DIR is required (scripts/tools/build_r2r_train_cohorts.py output)"
[[ -n "$COLLECT_ROOT" ]] || die "PPA_COLLECT_OUTPUT_ROOT is required (e.g. /workspace/exp23/round1)"
WORKERS_DIR="$COLLECT_ROOT/workers"
RECORD_DIR="${PPA_COLLECT_RECORD_DIR:-$COLLECT_ROOT/records}"
RUNTIME_DIR="$COLLECT_ROOT/runtime/$RUN_STAMP"

# Placeholders the train config expands on load; the client never reads them.
PLACEHOLDER_DIR="$COLLECT_ROOT/config_placeholders"
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$ROOT" HEATMAPVLN_MP3D_ROOT="$SCENES_DIR/mp3d"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
# Collection is never timed: the actions must be the evaluation's.
export HEATMAPVLN_TIMING=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

RPC_PYTHONPATH="$LOCKED_PLAN/tools:$RPC_ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

declare -a GPUS VO_GPUS SHARDS MODEL_PIDS VO_PIDS SLOT_PIDS XVFB_PIDS

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

# A shard is claimed by creating <workers>/shard_NNN.claim (mkdir is atomic) holding this
# launcher's pid.  A claim whose launcher is gone is stale: it is renamed away first, so
# two launchers cannot both take over the same one.  A claim without its owner file yet
# is being made, unless it is older than a minute.
claim_alive() {
  local claim="$1" owner
  owner="$(cut -d' ' -f1 "$claim/owner" 2>/dev/null || true)"
  if [[ -z "$owner" ]]; then
    [[ -n "$(find "$claim" -maxdepth 0 -mmin -1 2>/dev/null)" ]]
    return
  fi
  kill -0 "$owner" 2>/dev/null || return 1
  [[ -r "/proc/$owner/cmdline" ]] || return 0
  grep -qaF "run_s2_dagger_collect_cuda.sh" "/proc/$owner/cmdline"
}
claim_shard() {
  local claim="$WORKERS_DIR/shard_$1.claim" stale
  if ! mkdir "$claim" 2>/dev/null; then
    claim_alive "$claim" && return 1
    stale="$claim.stale.$$.$RANDOM"
    mv "$claim" "$stale" 2>/dev/null || return 1
    rm -rf "$stale"
    mkdir "$claim" 2>/dev/null || return 1
  fi
  echo "$$ $RUN_STAMP" >"$claim/owner"
}
release_shard() {
  local claim="$WORKERS_DIR/shard_$1.claim"
  [[ "$(cut -d' ' -f1 "$claim/owner" 2>/dev/null || true)" == "$$" ]] && rm -rf "$claim"
  return 0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${SLOT_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  for shard in "${SHARDS[@]:-}"; do
    [[ -n "$shard" ]] && release_shard "$(printf '%03d' "$shard")"
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

require_file "$COHORT_DIR/manifest.json"
for file in "$PPA_CONFIG" "$CLIENT"; do
  require_file "$file"
done
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$SCENES_DIR/mp3d"; do
  require_dir "$directory"
done
if [[ "$EXTERNAL_SERVERS" -eq 0 ]]; then
  for file in "$PPA_CHECKPOINT" "$MODEL_SERVER" "$VO_SERVER" \
    "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
    require_file "$file"
  done
  require_dir "$INTERNNAV_MODEL_PATH"
  [[ -z "$SYSTEM2_LORA" ]] || require_file "$SYSTEM2_LORA/adapter_config.json"
fi
for executable in "$PYTHON" "$CLIENT_PYTHON" "$XVFB_BIN"; do
  [[ -n "$executable" && -x "$executable" ]] || die "missing executable: ${executable:-Xvfb}"
done

NUM_SHARDS="$("$CLIENT_PYTHON" - "$COHORT_DIR/manifest.json" <<'MANIFEST'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("schema") != "heatmapvln-r2r-train-cohorts-v1" or manifest.get("split") != "train":
    raise SystemExit(f"not a train cohort manifest: schema={manifest.get('schema')!r} split={manifest.get('split')!r}")
print(int(manifest["num_shards"]))
MANIFEST
)" || die "$COHORT_DIR/manifest.json is not a build_r2r_train_cohorts.py train manifest"

if [[ "$SHARD_SPEC" == all ]]; then
  mapfile -t SHARDS < <(seq 0 $((NUM_SHARDS - 1)))
else
  IFS=',' read -r -a parts <<< "$SHARD_SPEC"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      for shard in $(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"); do SHARDS+=("$shard"); done
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      SHARDS+=("$part")
    else
      die "PPA_COLLECT_SHARDS takes 'all' or indices and ranges like 0,3,5-9; got '$part'"
    fi
  done
fi
(( ${#SHARDS[@]} >= 1 )) || die "no shard selected"
[[ "$(printf '%s\n' "${SHARDS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "${#SHARDS[@]}" ]] || die "shard IDs must be unique"
for shard in "${SHARDS[@]}"; do
  (( 10#$shard < NUM_SHARDS )) || die "shard $shard is outside the cohort's $NUM_SHARDS shards"
  tag="$(printf '%03d' "$shard")"
  require_file "$COHORT_DIR/shard_$tag.json"
  require_file "$COHORT_DIR/dataset_shard_$tag.json.gz"
done

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a VO_GPUS <<< "$VO_GPU_CSV"
NUM_SLOTS="${#GPUS[@]}"
(( NUM_SLOTS >= 1 && NUM_SLOTS <= 16 )) || die "PPA_EVAL_GPU_DEVICES must hold 1-16 IDs"
[[ "${#VO_GPUS[@]}" -eq "$NUM_SLOTS" ]] || die "PPA_EVAL_VO_GPU_DEVICES must have one ID per slot"
for gpu in "${GPUS[@]}" "${VO_GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
done
[[ "$PROTOCOL_SEED" =~ ^[0-9]+$ ]] || die "PPA_EVAL_PROTOCOL_SEED must be a non-negative integer"
[[ "$EXTERNAL_SERVERS" =~ ^[01]$ ]] || die "PPA_EVAL_EXTERNAL_SERVERS must be 0 or 1"
[[ "$SHARD_RETRIES" =~ ^[0-9]+$ ]] || die "PPA_EVAL_SHARD_RETRIES must be a non-negative integer"
[[ "$RETRY_PAUSE_S" =~ ^[0-9]+$ ]] || die "PPA_COLLECT_RETRY_PAUSE_S must be a non-negative integer"
[[ "$ALLOW_COMMIT_MISMATCH" =~ ^[01]$ ]] || die "PPA_EVAL_ALLOW_COMMIT_MISMATCH must be 0 or 1"
if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
  [[ -n "$EXTERNAL_SERVER_DIR" ]] \
    || die "PPA_EVAL_EXTERNAL_SERVERS=1 requires PPA_EVAL_EXTERNAL_SERVER_DIR (the servers' runtime directory, for the preflight evidence)"
  require_file "$EXTERNAL_SERVER_DIR/servers.json"
  [[ "$VO_GPU_CSV" == "$GPU_CSV" ]] \
    || die "PPA_EVAL_VO_GPU_DEVICES means nothing in external mode; the servers choose their own devices"
  # External servers are started with their adapter on their own side; here
  # PPA_EVAL_SYSTEM2_LORA names the adapter they must be serving (a local copy of the
  # directory, or its 64-hex sha256), and empty means the released System2.  Round 2
  # collecting with the wrong policy would otherwise go unnoticed.
  if [[ -n "$SYSTEM2_LORA" && ! "$SYSTEM2_LORA" =~ ^[0-9a-f]{64}$ ]]; then
    require_file "$SYSTEM2_LORA/adapter_model.safetensors"
  fi
else
  # Each local slot holds one model + one VO server on its own card.
  [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "$NUM_SLOTS" ]] || die "GPU IDs must be unique"
fi

mkdir -p "$WORKERS_DIR" "$RECORD_DIR" "$RUNTIME_DIR/logs" "$RUNTIME_DIR/failed" "$PLACEHOLDER_DIR"
cp -a "$COHORT_DIR/manifest.json" "$RUNTIME_DIR/cohort_manifest.json"
echo "[s2-collect] slots=$NUM_SLOTS gpus=$GPU_CSV vo_gpus=$VO_GPU_CSV shards=${#SHARDS[@]}/$NUM_SHARDS seed=$PROTOCOL_SEED retries=$SHARD_RETRIES"
echo "[s2-collect] checkpoint=$PPA_CHECKPOINT config=$PPA_CONFIG system2_lora=${SYSTEM2_LORA:-none}"
echo "[s2-collect] cohort=$COHORT_DIR records=$RECORD_DIR output=$COLLECT_ROOT external_servers=$EXTERNAL_SERVERS"

for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  display_num=$((DISPLAY_BASE + slot))
  tcp_open $((6000 + display_num)) && die "display :$display_num is already active; choose another PPA_EVAL_DISPLAY_BASE"
  env LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
    "$XVFB_BIN" ":$display_num" -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
    >"$RUNTIME_DIR/logs/xvfb_${slot}.log" 2>&1 &
  XVFB_PIDS[$slot]="$!"
  ready=0
  for _ in $(seq 1 60); do
    if tcp_open $((6000 + display_num)); then ready=1; break; fi
    kill -0 "${XVFB_PIDS[$slot]}" 2>/dev/null || break
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || die "Xvfb slot $slot failed; see $RUNTIME_DIR/logs/xvfb_${slot}.log"
done

if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
  # The same evidence checks as scripts/run_ppa_r2r_val_unseen_cuda.sh: something must
  # listen on each tunnelled port, and the archived startup evidence must be theirs.
  cp -a "$EXTERNAL_SERVER_DIR/servers.json" "$RUNTIME_DIR/external_servers.json"
  [[ -d "$EXTERNAL_SERVER_DIR/logs" ]] && cp -a "$EXTERNAL_SERVER_DIR/logs" "$RUNTIME_DIR/external_server_logs"
  for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    tcp_open $((MODEL_PORT_BASE + slot)) \
      || die "nothing is listening on 127.0.0.1:$((MODEL_PORT_BASE + slot)); is the SSH tunnel up?"
    tcp_open $((VO_PORT_BASE + slot)) \
      || die "nothing is listening on 127.0.0.1:$((VO_PORT_BASE + slot)); is the SSH tunnel up?"
  done
  [[ -e "$EXTERNAL_SERVER_DIR/STOPPED" ]] \
    && die "$EXTERNAL_SERVER_DIR/STOPPED exists: that server set has already been stopped; start the servers again and copy the new runtime directory"
  if [[ -e "$EXTERNAL_SERVER_DIR/RETIRED" ]]; then
    cat "$EXTERNAL_SERVER_DIR/RETIRED" >&2
    die "$EXTERNAL_SERVER_DIR/RETIRED exists: at least one slot was retired after its NPU became unusable (see above)"
  fi
  if [[ -d "$RUNTIME_DIR/external_server_logs" ]] \
    && grep -rlF "NPU device unusable after a failed request" "$RUNTIME_DIR/external_server_logs" 2>/dev/null; then
    die "an external server log reports its NPU unusable (listed above); restart the servers before running"
  fi
  CLIENT_COMMIT="$(git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  EXPECTED_LORA_SHA256=""
  if [[ "$SYSTEM2_LORA" =~ ^[0-9a-f]{64}$ ]]; then
    EXPECTED_LORA_SHA256="$SYSTEM2_LORA"
  elif [[ -n "$SYSTEM2_LORA" ]]; then
    EXPECTED_LORA_SHA256="$(file_sha256 "$SYSTEM2_LORA/adapter_model.safetensors")" || die "could not hash $SYSTEM2_LORA"
  fi
  # Collection runs the deployed arm: bridge on, untimed, the config's S and M.
  "$CLIENT_PYTHON" - "$RUNTIME_DIR/external_servers.json" "$NUM_SLOTS" "$MODEL_PORT_BASE" "$VO_PORT_BASE" \
    0 0 "" "" "$CLIENT_COMMIT" "$ALLOW_COMMIT_MISMATCH" "$EXPECTED_LORA_SHA256" <<'EXT' || die "external servers.json does not match this run"
import json
import re
import sys

(path, slots, model_base, vo_base, bridge_off, timing, num_sample_trajs,
 num_inference_steps, client_commit, allow_commit_mismatch, expected_lora) = sys.argv[1:12]
record = json.load(open(path, encoding="utf-8"))
problems = []
if str(record.get("system2_lora_sha256") or "") != expected_lora:
    problems.append(
        f"servers system2_lora_sha256={record.get('system2_lora_sha256')!r}, this run expects "
        f"{expected_lora or 'none'!r} (PPA_EVAL_SYSTEM2_LORA)"
    )
if record.get("schema") != "heatmapvln-npu-servers-v2":
    problems.append(f"unexpected schema {record.get('schema')!r}")
if int(record.get("slots", 0)) < int(slots):
    problems.append(f"servers expose {record.get('slots')} slot(s), this run wants {slots}")
want_model = [int(model_base) + i for i in range(int(slots))]
want_vo = [int(vo_base) + i for i in range(int(slots))]
if record.get("model_ports", [])[: int(slots)] != want_model:
    problems.append(f"model ports {record.get('model_ports')} != {want_model}")
if record.get("vo_ports", [])[: int(slots)] != want_vo:
    problems.append(f"VO ports {record.get('vo_ports')} != {want_vo}")
if int(record.get("bridge_off", 0)) != int(bridge_off):
    problems.append(f"servers bridge_off={record.get('bridge_off')}, this run sets {bridge_off}")
if int(record.get("timing", 0)) != int(timing):
    problems.append(f"servers timing={record.get('timing')}, this run sets {timing}")
for key, want in (("num_sample_trajs", num_sample_trajs), ("num_inference_steps", num_inference_steps)):
    if str(record.get(key, "")) != str(want):
        problems.append(f"servers {key}={record.get(key)!r}, this run sets {want!r}")
if not record.get("server_instance"):
    problems.append("servers.json has no server_instance; start them with the launcher, not by hand")
if record.get("device") != "npu":
    problems.append(f"servers.json says device={record.get('device')!r}, expected 'npu'")
if not re.fullmatch(r"[0-9a-f]{40}", str(record.get("repo_commit", ""))):
    problems.append(f"servers.json repo_commit={record.get('repo_commit')!r} is not a commit")
if int(record.get("repo_dirty", 0)) != 0:
    problems.append("the servers ran from a dirty working tree (repo_dirty=1)")
if (
    str(record.get("repo_commit")) != client_commit
    and int(allow_commit_mismatch) == 0
):
    problems.append(
        f"servers ran {str(record.get('repo_commit'))[:12]}, this client is {client_commit[:12]}; "
        "set PPA_EVAL_ALLOW_COMMIT_MISMATCH=1 if that is intended"
    )
if problems:
    print("external server mismatch: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
print(json.dumps({"external_servers": record}, sort_keys=True))
EXT
  echo "[s2-collect] using external servers from $EXTERNAL_SERVER_DIR"
else
declare -a MODEL_EXTRA=()
[[ -n "$SYSTEM2_LORA" ]] && MODEL_EXTRA=(--system2_lora_adapter "$SYSTEM2_LORA")
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  runtime="$RUNTIME_DIR/slot_${slot}"
  mkdir -p "$runtime"/model/{tmp,xdg,hf,torch_extensions,triton,matplotlib} "$runtime"/vo/{tmp,xdg,hf,triton}
  tcp_open $((MODEL_PORT_BASE + slot)) && die "port $((MODEL_PORT_BASE + slot)) is already in use"
  tcp_open $((VO_PORT_BASE + slot)) && die "port $((VO_PORT_BASE + slot)) is already in use"
  env PYTHONPATH="$RPC_PYTHONPATH" CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" \
    TMPDIR="$runtime/model/tmp" XDG_CACHE_HOME="$runtime/model/xdg" HF_HOME="$runtime/model/hf" \
    TORCH_EXTENSIONS_DIR="$runtime/model/torch_extensions" TRITON_CACHE_DIR="$runtime/model/triton" \
    MPLCONFIGDIR="$runtime/model/matplotlib" HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0 \
    "$PYTHON" -u "$MODEL_SERVER" \
      --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT" \
      --internnav_model_path "$INTERNNAV_MODEL_PATH" \
      --gpu_id 0 --host 127.0.0.1 --port $((MODEL_PORT_BASE + slot)) --workers 1 \
      --require_deterministic_sampling --require_ppa_online_amb3r ${MODEL_EXTRA[@]+"${MODEL_EXTRA[@]}"} \
      --log_level INFO >"$RUNTIME_DIR/logs/model_${slot}.log" 2>&1 &
  MODEL_PIDS[$slot]="$!"
  env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH" CUDA_VISIBLE_DEVICES="${VO_GPUS[$slot]}" \
    TMPDIR="$runtime/vo/tmp" XDG_CACHE_HOME="$runtime/vo/xdg" HF_HOME="$runtime/vo/hf" \
    TRITON_CACHE_DIR="$runtime/vo/triton" \
    "$PYTHON" -u "$VO_SERVER" \
      --repo "$REPO" --amb3r-root "$AMB3R_ROOT" \
      --da3-checkpoint "$DA3_CHECKPOINT" --device cuda:0 \
      --host 127.0.0.1 --port $((VO_PORT_BASE + slot)) \
      --map-init-window 20 --map-every 8 --max-history 8 \
      --resolution 518 392 --translation-scale 1.0 \
      --max-frames-limit 4096 --max-message-mb 32 \
      --log-level INFO >"$RUNTIME_DIR/logs/vo_${slot}.log" 2>&1 &
  VO_PIDS[$slot]="$!"
  sleep "$SERVER_STAGGER_S"
done
fi

# Both servers of a slot answer, are healthy and speak their protocol; in external mode
# they must also be the recorded server set (exit 3 when they answer but are not).
rpc_ready() {
  PYTHONPATH="$RPC_PYTHONPATH" "$CLIENT_PYTHON" - "$1" "$2" "${3:-}" "${4:-}" "${5:-}" "${6:-}" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient

model_address, vo_address, token, timing, rng_seed, model_version = (sys.argv[1:7] + [""] * 6)[:6]
wanted = {model_address: ["ppa-online-amb3r-v1"], vo_address: ["json+jpeg"]}
if token:
    for address, expected in wanted.items():
        expected += [f"heatmapvln-instance:{token}", "heatmapvln-device:npu", f"heatmapvln-timing:{int(timing)}"]
    wanted[vo_address].append(f"heatmapvln-vo-rng-seed:{rng_seed}")
for address, expected in wanted.items():
    client = VLAClient(server_addr=address, timeout_ms=5000)
    try:
        client.connect()
        info = client.get_server_info()
        if not client.health_check() or info is None:
            raise SystemExit(1)
        formats = set(info.supported_formats)
        protocol = expected[0]
        if protocol not in formats:
            raise SystemExit(1 if not token else 3)
        if [item for item in expected[1:] if item not in formats]:
            raise SystemExit(3)
        if model_version and address == model_address and info.model_version != model_version:
            raise SystemExit(3)
    finally:
        client.close()
PY
}

echo "[s2-collect] waiting for $((2 * NUM_SLOTS)) RPC servers"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
declare -a IDENTITY=()
if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
  external_instance="$("$CLIENT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["server_instance"])' "$RUNTIME_DIR/external_servers.json")"
  external_seed="$("$CLIENT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vo_rng_seed"])' "$RUNTIME_DIR/external_servers.json")"
  external_model_version="$("$CLIENT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_version"])' "$RUNTIME_DIR/external_servers.json")"
fi
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  model_addr="127.0.0.1:$((MODEL_PORT_BASE + slot))"
  vo_addr="127.0.0.1:$((VO_PORT_BASE + slot))"
  if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
    IDENTITY=("$external_instance/slot$slot" 0 "$external_seed" "$external_model_version")
  fi
  while true; do
    if [[ "$EXTERNAL_SERVERS" -eq 0 ]]; then
      kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot exited"; }
      kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot exited"; }
    else
      tcp_open $((MODEL_PORT_BASE + slot)) || die "127.0.0.1:$((MODEL_PORT_BASE + slot)) stopped listening (remote server or tunnel down)"
      tcp_open $((VO_PORT_BASE + slot)) || die "127.0.0.1:$((VO_PORT_BASE + slot)) stopped listening (remote server or tunnel down)"
    fi
    rc=0
    rpc_ready "$model_addr" "$vo_addr" ${IDENTITY[@]+"${IDENTITY[@]}"} || rc=$?
    if (( rc == 0 )); then
      break
    fi
    (( rc != 3 )) \
      || die "slot $slot: the servers on $model_addr / $vo_addr are not the ones in $EXTERNAL_SERVER_DIR/servers.json (wrong instance, device, timing or VO seed)"
    (( $(date +%s) < deadline )) || die "RPC startup timeout at slot $slot"
    sleep 10
  done
  if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
    model_log="$RUNTIME_DIR/external_server_logs/model_${slot}.log"
  else
    model_log="$RUNTIME_DIR/logs/model_${slot}.log"
  fi
  require_file "$model_log"
  grep -F "Formal PPA online AMB3R runtime enabled" "$model_log" >/dev/null \
    || die "model slot $slot lacks PPA preflight evidence"
  echo "[s2-collect] slot=$slot gpu=${GPUS[$slot]} vo_gpu=${VO_GPUS[$slot]} model=$model_addr vo=$vo_addr ready"
done

run_shard_once() {
  local slot="$1" tag="$2"
  local output="$WORKERS_DIR/shard_$tag"
  mkdir -p "$output"
  env PYTHONPATH="$RPC_PYTHONPATH" DISPLAY="127.0.0.1:$((DISPLAY_BASE + slot)).0" \
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" HABITAT_GL_GPU_ID=0 \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
    HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1 \
    "$CLIENT_PYTHON" -u "$CLIENT" \
      --config "$PPA_CONFIG" \
      --rpc_server "127.0.0.1:$((MODEL_PORT_BASE + slot))" \
      --history_pose_source amb3r_vo_da3 \
      --amb3r_vo_rpc_server "127.0.0.1:$((VO_PORT_BASE + slot))" \
      --amb3r_vo_rpc_timeout_ms "$RPC_TIMEOUT_MS" \
      --amb3r_vo_rpc_jpeg_quality 95 \
      --rpc_timeout_ms "$RPC_TIMEOUT_MS" --rpc_jpeg_quality 90 \
      --rpc_protocol_seed "$PROTOCOL_SEED" --rpc_require_deterministic_sampling \
      --rpc_policy_mode heatmapvln \
      --scenes_dir "$SCENES_DIR" \
      --data_path "$COHORT_DIR/dataset_shard_$tag.json.gz" \
      --dataset_split train --episode_list "$COHORT_DIR/shard_$tag.json" \
      --output_path "$output" --s2_record_dir "$RECORD_DIR" --sim_gpu_id 0 \
      --resize_w 384 --resize_h 384 --num_history "$NUM_HISTORY" \
      --max_steps_per_episode "$MAX_STEPS" --max_system2_calls_per_episode 0 \
      --auto_stop_distance 0 --trajectory_selection mean \
      --trajectory_x_sign 1 --trajectory_heading_alignment none \
      --system1_coord_order generated --no-pano_recenter_before_system1 \
      --no-debug_input_trace --debug_save_input_images 0 --resume \
      >>"$RUNTIME_DIR/logs/client_shard_$tag.log" 2>&1
}

# Exit 0 and write DONE when progress.json holds exactly the shard's list, once each, and
# every row names a committed record (tar + .sha256 marker) of that episode.
shard_complete() {
  "$CLIENT_PYTHON" - "$WORKERS_DIR/shard_$1" "$COHORT_DIR/shard_$1.json" <<'PY'
import json
import sys
from pathlib import Path

output, listing = Path(sys.argv[1]), Path(sys.argv[2])
want = [(str(item["scene_id"]), int(item["episode_id"])) for item in json.loads(listing.read_text())["episodes"]]
progress = output / "progress.json"
rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()] if progress.exists() else []
got = [(str(row.get("scene_id")), int(row.get("episode_id", -1))) for row in rows]
problems = []
if sorted(got) != sorted(want):
    problems.append(f"{len(set(want) - set(got))} listed episode(s) missing, {len(set(got) - set(want))} unlisted, "
                    f"{len(got) - len(set(got))} duplicate row(s)")
for row in rows:
    record = (row.get("s2_record") or {}).get("path")
    expected = f"{row.get('scene_id')}_{int(row.get('episode_id', -1)):06d}.s2rec.tar"
    if not record or Path(record).name != expected:
        problems.append(f"{row.get('scene_id')}/{row.get('episode_id')}: no s2_record in its progress row")
    elif not Path(record + ".sha256").is_file():
        problems.append(f"{record}: not committed (no .sha256 marker)")
if problems:
    print("shard incomplete: " + "; ".join(problems[:5]), file=sys.stderr)
    raise SystemExit(1)
(output / "DONE").write_text(json.dumps({"episodes": len(rows)}) + "\n")
PY
}

run_shard() {
  local slot="$1" tag="$2" attempt=0
  while true; do
    if run_shard_once "$slot" "$tag" && shard_complete "$tag" 2>>"$RUNTIME_DIR/logs/client_shard_$tag.log"; then
      return 0
    fi
    (( attempt++ ))
    if (( attempt > SHARD_RETRIES )); then
      echo "[s2-collect] slot=$slot shard=$tag failed after $attempt attempt(s)" >&2
      return 1
    fi
    echo "[s2-collect] slot=$slot shard=$tag died; restarting (attempt $((attempt + 1))/$((SHARD_RETRIES + 1)))" >&2
    if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
      # An open tunnel port proves nothing; wait for the recorded server to answer.
      local identity=() recover_deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S )) probe=0
      identity=("$external_instance/slot$slot" 0 "$external_seed" "$external_model_version")
      until rpc_ready "127.0.0.1:$((MODEL_PORT_BASE + slot))" "127.0.0.1:$((VO_PORT_BASE + slot))" "${identity[@]}"; do
        probe=$?
        if (( probe == 3 )); then
          echo "[s2-collect] slot=$slot shard=$tag: the servers on these ports are not the recorded set; giving up the shard" >&2
          return 1
        fi
        if (( $(date +%s) >= recover_deadline )); then
          echo "[s2-collect] slot=$slot shard=$tag: no healthy RPC within ${SERVER_START_TIMEOUT_S}s; giving up the shard" >&2
          return 1
        fi
        sleep 10
      done
    fi
    sleep "$RETRY_PAUSE_S"
  done
}

# One pass over the requested shards per slot: skip DONE, claimed and (this run) failed
# shards, claim the next one, run it, release it.
run_slot() {
  local slot="$1" shard tag
  for shard in "${SHARDS[@]}"; do
    tag="$(printf '%03d' "$shard")"
    [[ -e "$WORKERS_DIR/shard_$tag/DONE" || -e "$RUNTIME_DIR/failed/shard_$tag" ]] && continue
    claim_shard "$tag" || continue
    if [[ -e "$WORKERS_DIR/shard_$tag/DONE" ]]; then
      release_shard "$tag"
      continue
    fi
    echo "[s2-collect] slot=$slot shard=$tag start"
    if run_shard "$slot" "$tag"; then
      echo "[s2-collect] slot=$slot shard=$tag done"
    else
      touch "$RUNTIME_DIR/failed/shard_$tag"
      echo "[s2-collect] slot=$slot shard=$tag FAILED" >&2
      tail -40 "$RUNTIME_DIR/logs/client_shard_$tag.log" >&2 2>/dev/null || true
    fi
    release_shard "$tag"
  done
}

echo "[s2-collect] running ${#SHARDS[@]} shard(s) over $NUM_SLOTS slot(s)"
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  ( set +e; run_slot "$slot" ) &
  SLOT_PIDS[$slot]="$!"
  # Stagger the first claims so slot 0 does not take every shard's first look.
  sleep 1
done
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  wait "${SLOT_PIDS[$slot]}" || true
done

# Summary of every requested shard, whoever ran it.  Fails unless all of them are DONE.
"$CLIENT_PYTHON" - "$WORKERS_DIR" "$RUNTIME_DIR/failed" "${SHARDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

workers, failed_dir, shards = Path(sys.argv[1]), Path(sys.argv[2]), [f"{int(s):03d}" for s in sys.argv[3:]]
rows, done, failed, pending = [], [], [], []
for tag in shards:
    output = workers / f"shard_{tag}"
    progress = output / "progress.json"
    if progress.exists():
        rows += [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    if (output / "DONE").exists():
        done.append(tag)
    elif (failed_dir / f"shard_{tag}").exists():
        failed.append(tag)
    else:
        pending.append(tag)
n = len(rows)
mean = lambda key: round(sum(float(row[key]) for row in rows) / n, 4) if n else None
records = [row.get("s2_record") or {} for row in rows]
summary = {
    "status": "complete" if len(done) == len(shards) else "incomplete",
    "shards": len(shards),
    "shards_done": len(done),
    "shards_failed": failed,
    "shards_not_done_elsewhere_or_unstarted": pending,
    "episodes": n,
    "sr": mean("success"),
    "spl": mean("spl"),
    "os": mean("os"),
    "ne": mean("ne"),
    "ppa_applied_calls": sum(int(row.get("ppa_applied_calls", 0)) for row in rows),
    "system2_calls": sum(int(record.get("num_calls", 0)) for record in records),
    "records": sum(1 for record in records if record.get("path")),
    "front_mismatch_calls": sum(int(record.get("front_mismatch_calls", 0)) for record in records),
}
print(json.dumps(summary, sort_keys=True))
if n and summary["ppa_applied_calls"] <= 0:
    raise SystemExit("trained PPA was never applied after AMB3R warmup")
if summary["status"] != "complete":
    raise SystemExit(1)
PY

echo "[s2-collect] COMPLETE output=$COLLECT_ROOT records=$RECORD_DIR"
echo "[s2-collect] runtime logs=$RUNTIME_DIR/logs"
