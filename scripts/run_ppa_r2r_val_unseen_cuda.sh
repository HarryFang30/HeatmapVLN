#!/usr/bin/env bash
# R2R val-unseen closed-loop evaluation of the deployed PPA model on NVIDIA CUDA GPUs.
#
# Same protocol as scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh: the locked
# 8-shard cohort, deterministic sampling, the online AMB3R-VO server, llvmpipe
# rendering under Xvfb, and identical client flags.  Only the platform parts differ:
#   - no MACA environment; servers and client may share one Python;
#   - the system Xvfb, probed over TCP;
#   - any number of GPU slots.  Each slot runs one model server, one VO server and one
#     Xvfb, and works through its share of the 8 shards one after another;
#   - an optional per-shard episode cap for canaries.  Capped or partial runs print a
#     summary instead of merging;
#   - optional latency timing (PPA_EVAL_TIMING=1, docs/ops/deploy_rtx4090.md section 5).
#
# Meant to run inside the fjl-habitat container (paths below are container paths).
# Example canary on GPU 4, 2 episodes from each of shards 0 and 1:
#   PPA_EVAL_GPU_DEVICES=4 PPA_EVAL_SHARDS=0,1 PPA_EVAL_MAX_EPISODES_PER_SHARD=2 \
#     bash scripts/run_ppa_r2r_val_unseen_cuda.sh

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
LOCKED_PLAN="${PPA_EVAL_LOCKED_PLAN:-$ROOT/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802}"
COHORTS_DIR="$LOCKED_PLAN/cohorts"
MERGE_TOOL="$LOCKED_PLAN/tools/merge_shards.py"
DATASET="${PPA_EVAL_DATASET:-$ROOT/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz}"
# Must contain mp3d/<scene>/<scene>.glb (episode scene ids are relative to it).
SCENES_DIR="${PPA_EVAL_SCENES_DIR:-/dataset}"
EXPECTED_EPISODES=1839
NUM_SHARDS=8

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
CLIENT="$REPO/scripts/evaluation/r2r_val_unseen.py"

PROTOCOL_SEED="${PPA_EVAL_PROTOCOL_SEED:-42}"
EVAL_ARM="${PPA_EVAL_ARM:-ppa_refine_v2_online_amb3r_cuda}"
GPU_CSV="${PPA_EVAL_GPU_DEVICES:-0}"
VO_GPU_CSV="${PPA_EVAL_VO_GPU_DEVICES:-$GPU_CSV}"
SHARD_CSV="${PPA_EVAL_SHARDS:-0,1,2,3,4,5,6,7}"
MAX_EPISODES="${PPA_EVAL_MAX_EPISODES_PER_SHARD:-}"
# 1: servers and clients time every stage (src/utils/latency.py); each client writes
# <shard output>/timing/*.jsonl and the run ends with a latency summary.  Stages
# synchronise CUDA, so a timed run is slower; the actions are the same.
TIMING="${PPA_EVAL_TIMING:-0}"
# 1: EXP-20 A1, System1 samples from Z instead of the injected Z~ (rpc_model_server.py --ppa_bridge_off).
BRIDGE_OFF="${PPA_EVAL_BRIDGE_OFF:-0}"
# EXP-21 sensitivity: history frames K (client), System1 samples S and denoising steps M (server; empty = config).
NUM_HISTORY="${PPA_EVAL_NUM_HISTORY:-8}"
NUM_SAMPLE_TRAJS="${PPA_EVAL_NUM_SAMPLE_TRAJS:-}"
NUM_INFERENCE_STEPS="${PPA_EVAL_NUM_INFERENCE_STEPS:-}"
# Per-shard episode lists (shard_0N.json); another directory than the locked cohorts means a subset run: no merge.
EPISODE_LISTS_DIR="${PPA_EVAL_EPISODE_LISTS_DIR:-$COHORTS_DIR}"
# 1: the model and VO servers already run elsewhere (e.g. on Ascend NPUs, reached
# through an SSH tunnel on 127.0.0.1:<port>).  This script then starts no servers and
# only runs the clients; the client flags, protocol and merge stay exactly the same.
# docs/ops/deploy_ascend_910b.md describes the split deployment.
EXTERNAL_SERVERS="${PPA_EVAL_EXTERNAL_SERVERS:-0}"
# How many times to restart a client that dies mid-shard.  The client has no RPC
# retry of its own: one failed call raises and takes the whole shard with it, which
# over a 1839-episode run makes a single transient server or network hiccup expensive.
# --resume makes a restart cost at most the episode that was in flight.  0 keeps the
# original behaviour, which is what the certified CUDA runs used.
SHARD_RETRIES="${PPA_EVAL_SHARD_RETRIES:-0}"
# The remote servers' runtime directory, copied or mounted here: servers.json plus
# the startup logs, which is where the preflight evidence lives in external mode.
EXTERNAL_SERVER_DIR="${PPA_EVAL_EXTERNAL_SERVER_DIR:-}"
# 1: let a capped run resume or retry.  Off, because --max_episodes counts only
# episodes that are not already done (r2r_val_unseen.py:_eval_limit), and the cap is
# re-passed unchanged on every restart and every relaunch: a "4-episode canary" that
# dies once quietly becomes 5, and with retries=2 up to 12, with nothing in
# progress.json saying which rows were the extra ones.  Pin the episodes with
# PPA_EVAL_EPISODE_LISTS_DIR instead (scripts/tools/make_episode_lists_from_run.py),
# and then no cap is needed at all.
ALLOW_CAPPED_RESUME="${PPA_EVAL_ALLOW_CAPPED_RESUME:-0}"
# 1: accept servers built from a different commit than this client.
ALLOW_COMMIT_MISMATCH="${PPA_EVAL_ALLOW_COMMIT_MISMATCH:-0}"
if [[ -n "$MAX_EPISODES" ]]; then
  default_output="$ROOT/eval_runs/canary_seed${PROTOCOL_SEED}"
else
  default_output="$ROOT/eval_runs/ppa_refine_v2_seed${PROTOCOL_SEED}"
fi
# One output root per seed: --resume would otherwise skip episodes run under another seed.
OUTPUT_ROOT="${PPA_EVAL_OUTPUT_ROOT:-$default_output}"
WORKERS_DIR="$OUTPUT_ROOT/workers"
MERGED_DIR="$OUTPUT_ROOT/merged"
RUN_STAMP="${PPA_EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_DIR="$OUTPUT_ROOT/runtime/$RUN_STAMP"

MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"
DISPLAY_BASE="${PPA_EVAL_DISPLAY_BASE:-360}"
SERVER_START_TIMEOUT_S="${PPA_EVAL_SERVER_START_TIMEOUT_S:-1800}"
SERVER_STAGGER_S="${PPA_EVAL_SERVER_STAGGER_S:-15}"
RPC_TIMEOUT_MS="${PPA_EVAL_RPC_TIMEOUT_MS:-600000}"

# Placeholders the train config expands on load; the eval never reads them.
PLACEHOLDER_DIR="$OUTPUT_ROOT/config_placeholders"
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$ROOT" HEATMAPVLN_MP3D_ROOT="$SCENES_DIR/mp3d"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
# Keep the reference DA3 code path (non-xformers SwiGLU, chunked SDPA) used for the
# certified numbers; both are optional on CUDA.
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
# A container shell may carry a local proxy that does not exist here; the eval is offline.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

RPC_PYTHONPATH="$LOCKED_PLAN/tools:$RPC_ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

declare -a GPUS VO_GPUS SHARDS MODEL_PIDS VO_PIDS SLOT_PIDS XVFB_PIDS

die() { printf '[ppa-eval] ERROR: %s\n' "$*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
tcp_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

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

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${SLOT_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for file in "$PPA_CONFIG" "$CLIENT" "$MERGE_TOOL" "$DATASET"; do
  require_file "$file"
done
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$SCENES_DIR/mp3d" "$COHORTS_DIR"; do
  require_dir "$directory"
done
if [[ "$EXTERNAL_SERVERS" -eq 0 ]]; then
  # Weights, the AMB3R tree and the server scripts are server-side inputs; in
  # external mode they live on the machine that runs the servers.
  for file in "$PPA_CHECKPOINT" "$MODEL_SERVER" "$VO_SERVER" \
    "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
    require_file "$file"
  done
  require_dir "$INTERNNAV_MODEL_PATH"
fi
for executable in "$PYTHON" "$CLIENT_PYTHON" "$XVFB_BIN"; do
  [[ -n "$executable" && -x "$executable" ]] || die "missing executable: ${executable:-Xvfb}"
done

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a VO_GPUS <<< "$VO_GPU_CSV"
IFS=',' read -r -a SHARDS <<< "$SHARD_CSV"
NUM_SLOTS="${#GPUS[@]}"
(( NUM_SLOTS >= 1 && NUM_SLOTS <= NUM_SHARDS )) || die "PPA_EVAL_GPU_DEVICES must hold 1-$NUM_SHARDS IDs"
[[ "${#VO_GPUS[@]}" -eq "$NUM_SLOTS" ]] || die "PPA_EVAL_VO_GPU_DEVICES must have one ID per slot"
[[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "$NUM_SLOTS" ]] || die "GPU IDs must be unique"
[[ "$PROTOCOL_SEED" =~ ^[0-9]+$ ]] || die "PPA_EVAL_PROTOCOL_SEED must be a non-negative integer"
[[ -z "$MAX_EPISODES" || "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_MAX_EPISODES_PER_SHARD must be a positive integer"
[[ "$TIMING" =~ ^[01]$ ]] || die "PPA_EVAL_TIMING must be 0 or 1"
[[ "$BRIDGE_OFF" =~ ^[01]$ ]] || die "PPA_EVAL_BRIDGE_OFF must be 0 or 1"
declare -a MODEL_EXTRA=()
[[ "$BRIDGE_OFF" -eq 1 ]] && MODEL_EXTRA=(--ppa_bridge_off)
[[ "$NUM_HISTORY" =~ ^[1-8]$ ]] || die "PPA_EVAL_NUM_HISTORY must be 1-8"
[[ -z "$NUM_SAMPLE_TRAJS" || "$NUM_SAMPLE_TRAJS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_SAMPLE_TRAJS must be a positive integer"
[[ -z "$NUM_INFERENCE_STEPS" || "$NUM_INFERENCE_STEPS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_INFERENCE_STEPS must be a positive integer"
[[ -n "$NUM_SAMPLE_TRAJS" ]] && MODEL_EXTRA+=(--nextdit_num_sample_trajs "$NUM_SAMPLE_TRAJS")
[[ -n "$NUM_INFERENCE_STEPS" ]] && MODEL_EXTRA+=(--nextdit_num_inference_steps "$NUM_INFERENCE_STEPS")
# Always set, so a value left in the calling shell cannot switch timing on or off.
export HEATMAPVLN_TIMING="$TIMING"
for gpu in "${GPUS[@]}" "${VO_GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
done
[[ "$(printf '%s\n' "${SHARDS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "${#SHARDS[@]}" ]] || die "shard IDs must be unique"
[[ "$EXTERNAL_SERVERS" =~ ^[01]$ ]] || die "PPA_EVAL_EXTERNAL_SERVERS must be 0 or 1"
[[ "$SHARD_RETRIES" =~ ^[0-9]+$ ]] || die "PPA_EVAL_SHARD_RETRIES must be a non-negative integer"
if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
  # Without an explicit root the default would be this box's own CUDA output
  # directory, and --resume would then skip episodes that CUDA servers produced,
  # silently mixing two platforms into one result.
  [[ -n "${PPA_EVAL_OUTPUT_ROOT:-}" ]] \
    || die "PPA_EVAL_EXTERNAL_SERVERS=1 requires an explicit PPA_EVAL_OUTPUT_ROOT (never reuse a local-server run's directory)"
  [[ -n "$EXTERNAL_SERVER_DIR" ]] \
    || die "PPA_EVAL_EXTERNAL_SERVERS=1 requires PPA_EVAL_EXTERNAL_SERVER_DIR (the servers' runtime directory, for the preflight evidence)"
  require_file "$EXTERNAL_SERVER_DIR/servers.json"
  [[ "$VO_GPU_CSV" == "$GPU_CSV" ]] \
    || die "PPA_EVAL_VO_GPU_DEVICES means nothing in external mode; the servers choose their own devices"
fi
[[ "$ALLOW_CAPPED_RESUME" =~ ^[01]$ ]] || die "PPA_EVAL_ALLOW_CAPPED_RESUME must be 0 or 1"
[[ "$ALLOW_COMMIT_MISMATCH" =~ ^[01]$ ]] || die "PPA_EVAL_ALLOW_COMMIT_MISMATCH must be 0 or 1"
# A capped run is a fixed, pre-registered sample, and both of these silently change
# which episodes are in it.  Refused here rather than noticed afterwards, because
# nothing downstream can tell a capped run's rows apart from a restart's.
if [[ -n "$MAX_EPISODES" && "$ALLOW_CAPPED_RESUME" -eq 0 ]]; then
  (( SHARD_RETRIES == 0 )) \
    || die "PPA_EVAL_MAX_EPISODES_PER_SHARD with PPA_EVAL_SHARD_RETRIES=$SHARD_RETRIES would run $MAX_EPISODES more new episodes per restart; pin the episodes with PPA_EVAL_EPISODE_LISTS_DIR, or set PPA_EVAL_ALLOW_CAPPED_RESUME=1"
  for shard in "${SHARDS[@]}"; do
    progress="$WORKERS_DIR/shard_0${shard}/progress.json"
    [[ -s "$progress" ]] \
      && die "capped run into $progress, which already has $(wc -l < "$progress" | tr -d ' ') episode(s): --resume skips them and the cap would add $MAX_EPISODES more; use a fresh PPA_EVAL_OUTPUT_ROOT, or set PPA_EVAL_ALLOW_CAPPED_RESUME=1"
  done
fi
for shard in "${SHARDS[@]}"; do
  [[ "$shard" =~ ^[0-7]$ ]] || die "invalid shard: $shard"
  require_file "$COHORTS_DIR/shard_0${shard}.json"
  require_file "$EPISODE_LISTS_DIR/shard_0${shard}.json"
  require_file "$COHORTS_DIR/dataset_shard_0${shard}.json.gz"
done

mkdir -p "$WORKERS_DIR" "$MERGED_DIR" "$RUNTIME_DIR/logs" "$PLACEHOLDER_DIR"
# Timing logs newer than this marker belong to this run (--resume keeps older ones).
[[ "$TIMING" -eq 0 ]] || touch "$RUNTIME_DIR/timing_start"
echo "[ppa-eval] slots=$NUM_SLOTS gpus=$GPU_CSV vo_gpus=$VO_GPU_CSV shards=$SHARD_CSV seed=$PROTOCOL_SEED max_episodes_per_shard=${MAX_EPISODES:-all} timing=$TIMING bridge_off=$BRIDGE_OFF"
echo "[ppa-eval] checkpoint=$PPA_CHECKPOINT config=$PPA_CONFIG"
echo "[ppa-eval] num_history=$NUM_HISTORY num_sample_trajs=${NUM_SAMPLE_TRAJS:-config} num_inference_steps=${NUM_INFERENCE_STEPS:-config} episode_lists=$EPISODE_LISTS_DIR"
echo "[ppa-eval] output=$OUTPUT_ROOT external_servers=$EXTERNAL_SERVERS"

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
  # The servers run elsewhere; here there must already be something listening on
  # each tunnelled port, and the archived startup evidence must come from them.
  cp -a "$EXTERNAL_SERVER_DIR/servers.json" "$RUNTIME_DIR/external_servers.json"
  [[ -d "$EXTERNAL_SERVER_DIR/logs" ]] && cp -a "$EXTERNAL_SERVER_DIR/logs" "$RUNTIME_DIR/external_server_logs"
  for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    tcp_open $((MODEL_PORT_BASE + slot)) \
      || die "nothing is listening on 127.0.0.1:$((MODEL_PORT_BASE + slot)); is the SSH tunnel up?"
    tcp_open $((VO_PORT_BASE + slot)) \
      || die "nothing is listening on 127.0.0.1:$((VO_PORT_BASE + slot)); is the SSH tunnel up?"
  done
  # A server set that stopped, or a slot whose device died, must not be resumed
  # against: the servers log the marker and report NOT_SERVING, but their ports can
  # still be open through the tunnel.
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
  "$CLIENT_PYTHON" - "$RUNTIME_DIR/external_servers.json" "$NUM_SLOTS" "$MODEL_PORT_BASE" "$VO_PORT_BASE" \
    "$BRIDGE_OFF" "$TIMING" "$NUM_SAMPLE_TRAJS" "$NUM_INFERENCE_STEPS" "$CLIENT_COMMIT" \
    "$ALLOW_COMMIT_MISMATCH" <<'EXT' || die "external servers.json does not match this run"
import json
import re
import sys

(path, slots, model_base, vo_base, bridge_off, timing, num_sample_trajs,
 num_inference_steps, client_commit, allow_commit_mismatch) = sys.argv[1:11]
record = json.load(open(path, encoding="utf-8"))
problems = []
if record.get("schema") != "heatmapvln-npu-servers-v2":
    problems.append(f"unexpected schema {record.get('schema')!r}")
if int(record.get("slots", 0)) < int(slots):
    problems.append(f"servers expose {record.get('slots')} slot(s), this run wants {slots}")
# The tunnel maps each remote port to the same local port, so a mismatch here means
# the clients would reach a different set of servers than the evidence describes.
want_model = [int(model_base) + i for i in range(int(slots))]
want_vo = [int(vo_base) + i for i in range(int(slots))]
if record.get("model_ports", [])[: int(slots)] != want_model:
    problems.append(f"model ports {record.get('model_ports')} != {want_model}")
if record.get("vo_ports", [])[: int(slots)] != want_vo:
    problems.append(f"VO ports {record.get('vo_ports')} != {want_vo}")
if int(record.get("bridge_off", 0)) != int(bridge_off):
    problems.append(f"servers bridge_off={record.get('bridge_off')}, this run sets {bridge_off}")
# Timing is per process: with the client timed and the servers not, every check used
# to pass and the latency summary silently covered client stages only.
if int(record.get("timing", 0)) != int(timing):
    problems.append(f"servers timing={record.get('timing')}, this run sets {timing}")
# The arm is the servers' sampling configuration as much as the client's flags.
for key, want in (("num_sample_trajs", num_sample_trajs), ("num_inference_steps", num_inference_steps)):
    if str(record.get(key, "")) != str(want):
        problems.append(f"servers {key}={record.get(key)!r}, this run sets {want!r}")
# Written but never checked until now: a token-less or hand-started server set, or
# one whose build cannot be identified, cannot be the subject of a recorded result.
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
  echo "[ppa-eval] using external servers from $EXTERNAL_SERVER_DIR"
else
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  runtime="$RUNTIME_DIR/slot_${slot}"
  mkdir -p "$runtime"/model/{tmp,xdg,hf,torch_extensions,triton,matplotlib} "$runtime"/vo/{tmp,xdg,hf,triton}
  # A live port here means a stale server, or a tunnel left from an external-mode run.
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
      --require_deterministic_sampling --require_ppa_online_amb3r "${MODEL_EXTRA[@]}" \
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

# Both servers of a slot answer, are healthy and speak their protocol.  In external
# mode it also checks WHICH servers answered, against the recorded servers.json: a
# TCP connect proves only that something holds the port, and the something it reaches
# is the local ssh forwarder, so neither a dead remote server nor a local CUDA pair on
# the same default ports can be told apart without asking the server who it is.
# Exit 1: no usable answer (down, unhealthy, or still starting).  Exit 3: answered,
# but it is not the recorded server set -- retrying cannot fix that.
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
        # The build the servers.json evidence describes, as the server names it now.
        if model_version and address == model_address and info.model_version != model_version:
            raise SystemExit(3)
    finally:
        client.close()
PY
}

echo "[ppa-eval] waiting for $((2 * NUM_SLOTS)) RPC servers"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
# In external mode every readiness probe also demands this identity; empty locally,
# which leaves the certified CUDA path checking exactly what it checked before.
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
    IDENTITY=("$external_instance/slot$slot" "$TIMING" "$external_seed" "$external_model_version")
  fi
  while true; do
    if [[ "$EXTERNAL_SERVERS" -eq 0 ]]; then
      kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot exited"; }
      kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot exited"; }
    else
      # No local pid to watch: a vanished port means the remote server or the tunnel died.
      tcp_open $((MODEL_PORT_BASE + slot)) || die "127.0.0.1:$((MODEL_PORT_BASE + slot)) stopped listening (remote server or tunnel down)"
      tcp_open $((VO_PORT_BASE + slot)) || die "127.0.0.1:$((VO_PORT_BASE + slot)) stopped listening (remote server or tunnel down)"
    fi
    rc=0
    rpc_ready "$model_addr" "$vo_addr" "${IDENTITY[@]:-}" || rc=$?
    if (( rc == 0 )); then
      break
    fi
    (( rc != 3 )) \
      || die "slot $slot: the servers on $model_addr / $vo_addr are not the ones in $EXTERNAL_SERVER_DIR/servers.json (wrong instance, device, timing or VO seed); is a local CUDA server holding these ports, or the tunnel pointing elsewhere?"
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
  if [[ "$BRIDGE_OFF" -eq 1 ]]; then
    grep -F "PPA bridge off (EXP-20 A1)" "$model_log" >/dev/null \
      || die "model slot $slot lacks bridge-off evidence"
  fi
  for key in num_sample_trajs num_inference_steps; do
    if { [[ "$key" == num_sample_trajs && -n "$NUM_SAMPLE_TRAJS" ]] || [[ "$key" == num_inference_steps && -n "$NUM_INFERENCE_STEPS" ]]; }; then
      grep -F "Sensitivity override (EXP-21): nextdit.$key" "$model_log" >/dev/null \
        || die "model slot $slot lacks the $key override evidence"
    fi
  done
  echo "[ppa-eval] slot=$slot gpu=${GPUS[$slot]} vo_gpu=${VO_GPUS[$slot]} model=$model_addr vo=$vo_addr ready"
done

run_shard_once() {
  local slot="$1" shard="$2"
  local output="$WORKERS_DIR/shard_0${shard}"
  local -a cap=()
  [[ -n "$MAX_EPISODES" ]] && cap=(--max_episodes "$MAX_EPISODES")
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
      --data_path "$COHORTS_DIR/dataset_shard_0${shard}.json.gz" \
      --dataset_split val_unseen --episode_list "$EPISODE_LISTS_DIR/shard_0${shard}.json" \
      --output_path "$output" --sim_gpu_id 0 \
      --resize_w 384 --resize_h 384 --num_history "$NUM_HISTORY" \
      --max_steps_per_episode 500 --max_system2_calls_per_episode 0 \
      --auto_stop_distance 0 --trajectory_selection mean \
      --trajectory_x_sign 1 --trajectory_heading_alignment none \
      --system1_coord_order generated --no-pano_recenter_before_system1 \
      --no-debug_input_trace --debug_save_input_images 0 --resume "${cap[@]}" \
      >>"$RUNTIME_DIR/logs/client_shard_0${shard}.log" 2>&1
}

run_shard() {
  local slot="$1" shard="$2" attempt=0
  while true; do
    if run_shard_once "$slot" "$shard"; then
      return 0
    fi
    (( attempt++ ))
    if (( attempt > SHARD_RETRIES )); then
      echo "[ppa-eval] slot=$slot shard=$shard failed after $attempt attempt(s)" >&2
      return 1
    fi
    # The episodes already recorded are kept; --resume skips them on the next attempt.
    echo "[ppa-eval] slot=$slot shard=$shard died; restarting (attempt $((attempt + 1))/$((SHARD_RETRIES + 1)))" >&2
    if [[ "$EXTERNAL_SERVERS" -eq 1 ]]; then
      # Wait for a server that answers a real RPC, not merely for an open port.  An
      # Ascend device error leaves the process alive and LISTENing, and in external
      # mode the thing that accepts the connection is the local ssh forwarder anyway,
      # so the old tcp_open wait would return at once and the restarted client would
      # burn every remaining retry on a server that cannot compute.
      local identity=() recover_deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S )) probe=0
      [[ -n "${external_instance:-}" ]] && identity=("$external_instance/slot$slot" "$TIMING" "$external_seed" "$external_model_version")
      until rpc_ready "127.0.0.1:$((MODEL_PORT_BASE + slot))" "127.0.0.1:$((VO_PORT_BASE + slot))" "${identity[@]:-}"; do
        probe=$?
        if (( probe == 3 )); then
          echo "[ppa-eval] slot=$slot shard=$shard: the servers on these ports are not the recorded set; giving up the shard" >&2
          return 1
        fi
        if (( $(date +%s) >= recover_deadline )); then
          echo "[ppa-eval] slot=$slot shard=$shard: no healthy RPC within ${SERVER_START_TIMEOUT_S}s; giving up the shard" >&2
          return 1
        fi
        sleep 10
      done
    fi
    sleep 15
  done
}

echo "[ppa-eval] running shards $SHARD_CSV over $NUM_SLOTS slot(s) retries=$SHARD_RETRIES"
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  (
    set +e
    for idx in "${!SHARDS[@]}"; do
      (( idx % NUM_SLOTS == slot )) || continue
      shard="${SHARDS[$idx]}"
      echo "[ppa-eval] slot=$slot shard=$shard start"
      run_shard "$slot" "$shard" || { echo "[ppa-eval] slot=$slot shard=$shard FAILED" >&2; exit 1; }
      echo "[ppa-eval] slot=$slot shard=$shard done"
    done
  ) &
  SLOT_PIDS[$slot]="$!"
done

# bash 5.0 (Ubuntu 20.04 containers) has no `wait -n -p`; wait for the slots in turn.
failed=0
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  wait "${SLOT_PIDS[$slot]}" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  for shard in "${SHARDS[@]}"; do
    tail -60 "$RUNTIME_DIR/logs/client_shard_0${shard}.log" >&2 2>/dev/null || true
  done
  die "an evaluation slot failed"
fi

if [[ "$TIMING" -eq 1 ]]; then
  mapfile -t timing_files < <(find "$WORKERS_DIR" -path '*/timing/*.jsonl' -newer "$RUNTIME_DIR/timing_start" | sort)
  if (( ${#timing_files[@]} > 0 )); then
    summary_rc=0
    "$PYTHON" "$REPO/scripts/tools/summarize_latency.py" "${timing_files[@]}" --output-dir "$RUNTIME_DIR/latency" \
      >"$RUNTIME_DIR/logs/latency_summary.log" 2>&1 || summary_rc=$?
    case "$summary_rc" in
      0) echo "[ppa-eval] latency summary=$RUNTIME_DIR/latency/latency_summary.md" ;;
      # The summary exists but covers client stages only: timing is per process, so
      # this is what a timed client in front of untimed servers produces.
      3) echo "[ppa-eval] latency summary=$RUNTIME_DIR/latency/latency_summary.md"
         echo "[ppa-eval] WARNING: the servers were not running with HEATMAPVLN_TIMING=1; the summary is client-side only and no end-to-end latency may be quoted from it" >&2 ;;
      *) echo "[ppa-eval] WARNING: latency summary failed; see $RUNTIME_DIR/logs/latency_summary.log" >&2 ;;
    esac
  else
    echo "[ppa-eval] WARNING: timing was on but this run wrote no timing log" >&2
  fi
fi

if [[ -z "$MAX_EPISODES" && "${#SHARDS[@]}" -eq "$NUM_SHARDS" && "$EPISODE_LISTS_DIR" == "$COHORTS_DIR" ]]; then
  PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" "$MERGE_TOOL" \
    --dataset "$DATASET" --cohorts-dir "$COHORTS_DIR" \
    --workers-dir "$WORKERS_DIR" --output-dir "$MERGED_DIR" \
    --num-shards "$NUM_SHARDS" --expected-episodes "$EXPECTED_EPISODES" \
    --protocol heatmapvln-r2r-json-v3 \
    --sampling-protocol heatmapvln-nextdit-sha256-v1 \
    --protocol-seed "$PROTOCOL_SEED" --evaluation-arm "$EVAL_ARM"
  SUMMARY_ROOT="$MERGED_DIR"
  SUMMARY_EXPECTED="$EXPECTED_EPISODES"
else
  SUMMARY_ROOT="$WORKERS_DIR"
  SUMMARY_EXPECTED=0
fi

"$PYTHON" - "$SUMMARY_ROOT" "$SUMMARY_EXPECTED" "$SHARD_CSV" <<'PY'
import json
import sys
from pathlib import Path

root, expected, shards = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
if expected:  # the merge tool writes progress.jsonl
    files = [path for path in (root / "progress.jsonl", root / "progress.json") if path.exists()][:1]
else:
    files = [root / f"shard_0{s}" / "progress.json" for s in shards]
rows = []
for path in files:
    if path.exists():
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("no episodes were recorded")
if expected and len(rows) != expected:
    raise SystemExit(f"merged episode count mismatch: {len(rows)} != {expected}")
if any(row.get("history_pose_source") != "amb3r_vo_da3" for row in rows):
    raise SystemExit("result contains a non-AMB3R pose provider")
applied = sum(int(row.get("ppa_applied_calls", 0)) for row in rows)
if applied <= 0:
    raise SystemExit("trained PPA was never applied after AMB3R warmup")
n = len(rows)
mean = lambda key: sum(float(row[key]) for row in rows) / n
summary = {
    "status": "passed",
    "episodes": n,
    "ppa_applied_calls": applied,
    "sr": round(100 * mean("success"), 2),
    "spl": round(100 * mean("spl"), 2),
    "os": round(100 * mean("os"), 2),
    "ne": round(mean("ne"), 3),
}
if expected:
    summary["result"] = json.loads((root / "result.json").read_text())
print(json.dumps(summary, sort_keys=True))
PY

# A pinned-list run must have produced exactly the pinned episodes.  The merge tool
# enforces this for a full run, and a capped run has no fixed set to check against,
# so without this an uncapped subset run was the one shape that could come out short
# or long and still print "passed".  Printed after the summary, so a mismatch still
# leaves the numbers visible, and then fails.
set_rc=0
if [[ -z "$MAX_EPISODES" ]]; then
  "$PYTHON" - "$WORKERS_DIR" "$EPISODE_LISTS_DIR" "$SHARD_CSV" <<'PY' || set_rc=$?
import json
import sys
from pathlib import Path

workers, lists, shards = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
problems = []
for shard in shards:
    listed = json.loads((lists / f"shard_0{shard}.json").read_text(encoding="utf-8"))
    listed = listed.get("episodes", listed) if isinstance(listed, dict) else listed
    want = {(str(item["scene_id"]), int(item["episode_id"])) for item in listed}
    rows = [
        json.loads(line)
        for line in (workers / f"shard_0{shard}" / "progress.json").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row.get("scene_id") is None or row.get("episode_id") is None for row in rows):
        problems.append(f"shard {shard}: a progress row has no scene_id/episode_id, so the set cannot be checked")
        continue
    got = [(str(row["scene_id"]), int(row["episode_id"])) for row in rows]
    if len(got) != len(set(got)):
        problems.append(f"shard {shard}: {len(got) - len(set(got))} duplicate episode row(s)")
    if set(got) != want:
        missing, extra = sorted(want - set(got)), sorted(set(got) - want)
        problems.append(f"shard {shard}: {len(missing)} listed episode(s) missing, {len(extra)} unlisted episode(s) present")
if problems:
    print("episode set mismatch: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
PY
fi
(( set_rc == 0 )) || die "the recorded episodes are not the listed ones (see above)"

echo "[ppa-eval] COMPLETE output=$OUTPUT_ROOT"
echo "[ppa-eval] runtime logs=$RUNTIME_DIR/logs"
