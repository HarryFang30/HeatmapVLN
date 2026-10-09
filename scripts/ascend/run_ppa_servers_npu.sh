#!/usr/bin/env bash
# Start the PPA model and AMB3R-VO RPC servers on Ascend NPUs and keep them running.
#
# This is the server half of the split deployment: the Habitat clients, Xvfb and the
# locked evaluation plan stay on the x86 CUDA box and reach these servers over an SSH
# tunnel, so nothing in the client path changes (docs/ops/deploy_ascend_910b.md).
# There is no Habitat, no dataset and no client here.
#
# Both servers bind 127.0.0.1 only.  The gRPC channels have no authentication at
# either end, so they must never be reachable from outside this host; the tunnel is
# what carries them to the client box.
#
# One slot = one (model server, VO server) pair.  They share a card by default: the
# certified CUDA pair peaked at about 41 GB together and a 910B3 has 64 GB.
#
#   PPA_NPU_DEVICES=0,1 bash scripts/ascend/run_ppa_servers_npu.sh
#
# Stop it with SIGINT/SIGTERM; the trap stops every server it started.

set -Eeuo pipefail

ROOT="${PPA_EVAL_ROOT:-$HOME/work/zhr/zhr_1}"
REPO="${PPA_EVAL_REPO:-$ROOT/HeatmapVLN}"
RPC_ROOT="${PPA_EVAL_RPC_ROOT:-$ROOT/rpc}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$ROOT/InternNav_Model}"
AMB3R_ROOT="${PPA_EVAL_AMB3R_ROOT:-$ROOT/amb3r}"
DA3_CHECKPOINT="${PPA_EVAL_DA3_CHECKPOINT:-$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE}"
PYTHON="${PPA_EVAL_PYTHON:-$ROOT/envs/ppa/bin/python}"

PPA_CHECKPOINT="${PPA_EVAL_CHECKPOINT:-$ROOT/weights/ppa_refine_v2_best.pth}"
PPA_CONFIG="${PPA_EVAL_CONFIG:-$REPO/configs/ppa_action_refine_v2_8gpu.yaml}"

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"

# Logical NPU ids, one per slot.  Each server process sees only its own card, as
# device npu:0, so a wrong id cannot silently co-locate two slots on one card.
NPU_CSV="${PPA_NPU_DEVICES:-0}"
VO_NPU_CSV="${PPA_NPU_VO_DEVICES:-$NPU_CSV}"
# A card counts as free below this many MiB of HBM in use.  The cards are shared.
MAX_USED_MIB="${PPA_NPU_MAX_USED_MIB:-4096}"
THREADS_PER_SERVER="${PPA_NPU_THREADS_PER_SERVER:-8}"
ASCEND_ENV="${PPA_NPU_ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

# Same names as the CUDA launcher, so one set of exports drives both halves.
TIMING="${PPA_EVAL_TIMING:-0}"
BRIDGE_OFF="${PPA_EVAL_BRIDGE_OFF:-0}"
NUM_SAMPLE_TRAJS="${PPA_EVAL_NUM_SAMPLE_TRAJS:-}"
NUM_INFERENCE_STEPS="${PPA_EVAL_NUM_INFERENCE_STEPS:-}"
# Re-seed the VO device RNG at every episode.  Required for run-to-run reproducible
# poses on Ascend, where the default device generator seed is not a constant.
VO_RNG_SEED="${PPA_NPU_VO_RNG_SEED:-0}"

MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"
SERVER_START_TIMEOUT_S="${PPA_EVAL_SERVER_START_TIMEOUT_S:-3600}"
SERVER_STAGGER_S="${PPA_EVAL_SERVER_STAGGER_S:-20}"

RUN_STAMP="${PPA_EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_DIR="${PPA_NPU_RUNTIME_DIR:-$ROOT/servers/$RUN_STAMP}"

declare -a NPUS VO_NPUS MODEL_PIDS VO_PIDS

die() { printf '[ppa-npu] ERROR: %s\n' "$*" >&2; exit 2; }
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
  # Orphaned servers would hold their cards and their ports, so always stop them.
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  printf '[ppa-npu] stopped (status=%s)\n' "$status"
  [[ -d "$RUNTIME_DIR" ]] && date +%Y-%m-%dT%H:%M:%S%z > "$RUNTIME_DIR/STOPPED"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# A non-interactive shell does not necessarily load CANN; without it torch_npu
# cannot see a device at all (the MACA equivalent is in CLAUDE.md section 3.1).
[[ -r "$ASCEND_ENV" ]] || die "CANN env script not readable: $ASCEND_ENV (set PPA_NPU_ASCEND_ENV)"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
[[ -n "${ASCEND_TOOLKIT_HOME:-}${ASCEND_HOME_PATH:-}" ]] || die "sourcing $ASCEND_ENV left ASCEND_TOOLKIT_HOME empty"

# An inherited ASCEND_RT_VISIBLE_DEVICES would remap the per-slot ids below.
[[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] || die "ASCEND_RT_VISIBLE_DEVICES is already set; unset it"
unset CUDA_VISIBLE_DEVICES
# The eval is offline; a proxy left in the shell only breaks local connections.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$VO_SERVER" \
  "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
  require_file "$file"
done
# The locked plan is a client-side input: it holds the cohorts, the dataset shards
# and the merge tool, none of which either server imports.  It stays on the client box.
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_MODEL_PATH"; do
  require_dir "$directory"
done
[[ -x "$PYTHON" ]] || die "missing executable: $PYTHON"
command -v npu-smi >/dev/null || die "npu-smi not found"

IFS=',' read -r -a NPUS <<< "$NPU_CSV"
IFS=',' read -r -a VO_NPUS <<< "$VO_NPU_CSV"
NUM_SLOTS="${#NPUS[@]}"
(( NUM_SLOTS >= 1 && NUM_SLOTS <= 8 )) || die "PPA_NPU_DEVICES must hold 1-8 ids"
[[ "${#VO_NPUS[@]}" -eq "$NUM_SLOTS" ]] || die "PPA_NPU_VO_DEVICES must have one id per slot"
[[ "$(printf '%s\n' "${NPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "$NUM_SLOTS" ]] || die "NPU ids must be unique"
for npu in "${NPUS[@]}" "${VO_NPUS[@]}"; do
  [[ "$npu" =~ ^[0-9]+$ ]] || die "invalid NPU id: $npu"
done
[[ "$TIMING" =~ ^[01]$ ]] || die "PPA_EVAL_TIMING must be 0 or 1"
[[ "$BRIDGE_OFF" =~ ^[01]$ ]] || die "PPA_EVAL_BRIDGE_OFF must be 0 or 1"
[[ -z "$NUM_SAMPLE_TRAJS" || "$NUM_SAMPLE_TRAJS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_SAMPLE_TRAJS must be a positive integer"
[[ -z "$NUM_INFERENCE_STEPS" || "$NUM_INFERENCE_STEPS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_INFERENCE_STEPS must be a positive integer"
[[ "$VO_RNG_SEED" =~ ^[0-9]+$ ]] || die "PPA_NPU_VO_RNG_SEED must be a non-negative integer"

declare -a MODEL_EXTRA=()
[[ "$BRIDGE_OFF" -eq 1 ]] && MODEL_EXTRA=(--ppa_bridge_off)
[[ -n "$NUM_SAMPLE_TRAJS" ]] && MODEL_EXTRA+=(--nextdit_num_sample_trajs "$NUM_SAMPLE_TRAJS")
[[ -n "$NUM_INFERENCE_STEPS" ]] && MODEL_EXTRA+=(--nextdit_num_inference_steps "$NUM_INFERENCE_STEPS")

# Placeholders the train config expands on load; the eval never reads them.
PLACEHOLDER_DIR="$RUNTIME_DIR/config_placeholders"
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$ROOT"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep the DA3 code path the certified numbers came from: the reference SwiGLU and
# the query-chunked SDPA.  Both must be set here too, or the forward silently takes
# a different path (and unchunked attention materialises a ~20 GiB score matrix).
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
export HEATMAPVLN_TIMING="$TIMING"

RPC_PYTHONPATH="$RPC_ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUNTIME_DIR/logs" "$PLACEHOLDER_DIR"

npu_used_mib() {
  # "HBM-Usage(MB)" for one chip, from npu-smi's machine-readable per-device query.
  npu-smi info -t usages -i "$1" 2>/dev/null \
    | awk -F: '/HBM Usage Rate|HBM Capacity|Memory Usage Rate/ {next} /HBM Usage/ {gsub(/ /,"",$2); print $2; exit}'
}

npu_free_mib() {
  local id="$1" used
  used="$(npu-smi info 2>/dev/null | awk -v id="$id" '
    $2 == id && $3 ~ /^910/ { want = 1; next }
    want { n = split($0, f, "/"); gsub(/[^0-9]/, "", f[1]); print f[1] + 0; exit }
  ')"
  printf '%s' "${used:-unknown}"
}

echo "[ppa-npu] slots=$NUM_SLOTS npus=$NPU_CSV vo_npus=$VO_NPU_CSV timing=$TIMING bridge_off=$BRIDGE_OFF"
echo "[ppa-npu] num_sample_trajs=${NUM_SAMPLE_TRAJS:-config} num_inference_steps=${NUM_INFERENCE_STEPS:-config} vo_rng_seed=$VO_RNG_SEED"
echo "[ppa-npu] checkpoint=$PPA_CHECKPOINT config=$PPA_CONFIG"
echo "[ppa-npu] runtime=$RUNTIME_DIR"

# The cards are shared with other users, like the 4090 box: only take free ones.
for npu in "${NPUS[@]}" "${VO_NPUS[@]}"; do
  used="$(npu_free_mib "$npu")"
  if [[ "$used" == "unknown" ]]; then
    die "could not read HBM use of NPU $npu from npu-smi"
  fi
  (( used <= MAX_USED_MIB )) || die "NPU $npu already has ${used} MiB in use (limit $MAX_USED_MIB); pick another card"
  echo "[ppa-npu] npu=$npu used=${used}MiB free enough"
done

# Platform preflight: everything the servers silently depend on, checked once, loudly.
PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - <<'PY' || die "platform preflight failed"
import json
import sys

import torch
import torch_npu  # noqa: F401
import transformers

problems = []
if transformers.__version__ != "4.51.0":
    problems.append(f"transformers {transformers.__version__} != 4.51.0 (runtime_compat gate)")
if not torch.npu.is_available():
    problems.append("torch.npu.is_available() is False")
elif not torch.npu.is_bf16_supported():
    problems.append("torch.npu.is_bf16_supported() is False; the reference dtype is bf16")
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "npu_count": torch.npu.device_count() if torch.npu.is_available() else 0,
}, sort_keys=True))
if problems:
    print("PREFLIGHT: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
PY

for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  model_port=$((MODEL_PORT_BASE + slot))
  vo_port=$((VO_PORT_BASE + slot))
  # A live port here means a stale server or another run; starting anyway would
  # leave the client talking to whichever process won the bind.
  tcp_open "$model_port" && die "port $model_port is already in use"
  tcp_open "$vo_port" && die "port $vo_port is already in use"

  runtime="$RUNTIME_DIR/slot_${slot}"
  mkdir -p "$runtime"/model/{tmp,xdg,hf,matplotlib} "$runtime"/vo/{tmp,xdg,hf}
  # Each server sees exactly one card, as npu:0.
  env PYTHONPATH="$RPC_PYTHONPATH" ASCEND_RT_VISIBLE_DEVICES="${NPUS[$slot]}" \
    OMP_NUM_THREADS="$THREADS_PER_SERVER" MKL_NUM_THREADS="$THREADS_PER_SERVER" \
    TMPDIR="$runtime/model/tmp" XDG_CACHE_HOME="$runtime/model/xdg" HF_HOME="$runtime/model/hf" \
    MPLCONFIGDIR="$runtime/model/matplotlib" HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0 \
    "$PYTHON" -u "$MODEL_SERVER" \
      --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT" \
      --internnav_model_path "$INTERNNAV_MODEL_PATH" \
      --device npu --gpu_id 0 --host 127.0.0.1 --port "$model_port" --workers 1 \
      --require_deterministic_sampling --require_ppa_online_amb3r "${MODEL_EXTRA[@]}" \
      --log_level INFO >"$RUNTIME_DIR/logs/model_${slot}.log" 2>&1 &
  MODEL_PIDS[$slot]="$!"
  env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH" \
    ASCEND_RT_VISIBLE_DEVICES="${VO_NPUS[$slot]}" \
    OMP_NUM_THREADS="$THREADS_PER_SERVER" MKL_NUM_THREADS="$THREADS_PER_SERVER" \
    TMPDIR="$runtime/vo/tmp" XDG_CACHE_HOME="$runtime/vo/xdg" HF_HOME="$runtime/vo/hf" \
    "$PYTHON" -u "$VO_SERVER" \
      --repo "$REPO" --amb3r-root "$AMB3R_ROOT" \
      --da3-checkpoint "$DA3_CHECKPOINT" --device npu:0 \
      --host 127.0.0.1 --port "$vo_port" \
      --map-init-window 20 --map-every 8 --max-history 8 \
      --resolution 518 392 --translation-scale 1.0 \
      --max-frames-limit 4096 --max-message-mb 32 \
      --rng-seed "$VO_RNG_SEED" \
      --log-level INFO >"$RUNTIME_DIR/logs/vo_${slot}.log" 2>&1 &
  VO_PIDS[$slot]="$!"
  sleep "$SERVER_STAGGER_S"
done

echo "[ppa-npu] waiting for $((2 * NUM_SLOTS)) RPC servers"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  model_addr="127.0.0.1:$((MODEL_PORT_BASE + slot))"
  vo_addr="127.0.0.1:$((VO_PORT_BASE + slot))"
  while true; do
    kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot exited"; }
    kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot exited"; }
    if PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - "$model_addr" "$vo_addr" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient

for address, expected in ((sys.argv[1], "ppa-online-amb3r-v1"), (sys.argv[2], "json+jpeg")):
    client = VLAClient(server_addr=address, timeout_ms=5000)
    try:
        client.connect()
        info = client.get_server_info()
        if not client.health_check() or info is None:
            raise SystemExit(1)
        if expected not in set(info.supported_formats):
            raise SystemExit(2)
    finally:
        client.close()
PY
    then
      break
    fi
    (( $(date +%s) < deadline )) || die "RPC startup timeout at slot $slot"
    sleep 10
  done
  # Same preflight evidence the CUDA launcher greps for, plus the device lines that
  # only this platform can get wrong.
  grep -F "Formal PPA online AMB3R runtime enabled" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
    || die "model slot $slot lacks PPA preflight evidence"
  grep -F "Model server device: npu:0" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
    || die "model slot $slot did not report an NPU device"
  grep -F "VO server device: npu:0" "$RUNTIME_DIR/logs/vo_${slot}.log" >/dev/null \
    || die "VO slot $slot did not report an NPU device"
  grep -F "DA3_SDPA_QUERY_CHUNK_SIZE=256" "$RUNTIME_DIR/logs/vo_${slot}.log" >/dev/null \
    || die "VO slot $slot did not take the chunked-SDPA path"
  if [[ "$BRIDGE_OFF" -eq 1 ]]; then
    grep -F "PPA bridge off (EXP-20 A1)" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
      || die "model slot $slot lacks bridge-off evidence"
  fi
  for key in num_sample_trajs num_inference_steps; do
    if { [[ "$key" == num_sample_trajs && -n "$NUM_SAMPLE_TRAJS" ]] || [[ "$key" == num_inference_steps && -n "$NUM_INFERENCE_STEPS" ]]; }; then
      grep -F "Sensitivity override (EXP-21): nextdit.$key" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
        || die "model slot $slot lacks the $key override evidence"
    fi
  done
  echo "[ppa-npu] slot=$slot npu=${NPUS[$slot]} vo_npu=${VO_NPUS[$slot]} model=$model_addr vo=$vo_addr ready"
done

# What the client box needs in order to point its tunnel at these servers and to
# record which build served its episodes.
"$PYTHON" - "$RUNTIME_DIR/servers.json" "$NUM_SLOTS" "$MODEL_PORT_BASE" "$VO_PORT_BASE" \
  "$NPU_CSV" "$VO_NPU_CSV" "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)" \
  "$VO_RNG_SEED" "$TIMING" "$BRIDGE_OFF" <<'PY'
import json
import sys

path, slots, model_base, vo_base, npus, vo_npus, commit, rng_seed, timing, bridge_off = sys.argv[1:11]
record = {
    "schema": "heatmapvln-npu-servers-v1",
    "slots": int(slots),
    "model_ports": [int(model_base) + i for i in range(int(slots))],
    "vo_ports": [int(vo_base) + i for i in range(int(slots))],
    "npus": npus,
    "vo_npus": vo_npus,
    "repo_commit": commit,
    "vo_rng_seed": int(rng_seed),
    "timing": int(timing),
    "bridge_off": int(bridge_off),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
print(json.dumps(record, sort_keys=True))
PY

echo "[ppa-npu] all servers ready; servers.json=$RUNTIME_DIR/servers.json"
echo "[ppa-npu] logs=$RUNTIME_DIR/logs"
echo "[ppa-npu] holding; SIGINT/SIGTERM stops every server"

# Hold the servers and die loudly if any of them does, so the client box does not
# keep sending requests into a dead port.
while true; do
  for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -60 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot died"; }
    kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -60 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot died"; }
  done
  sleep 30
done
