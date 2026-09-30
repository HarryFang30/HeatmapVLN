#!/usr/bin/env bash
# Start the two servers a robot's NavAgent (src/deploy/nav_agent.py) talks to, on one CUDA GPU:
# the model server (System 2 + System 1 + PPA heads) and the online AMB3R VO server.
#
# The server command lines are the ones of scripts/run_ppa_r2r_val_unseen_cuda.sh (same
# checkpoint, config, flags, environment); no Habitat, no Xvfb, no evaluation client.  Stays in
# the foreground until Ctrl-C or until either server exits, and then stops both.
#
# Meant for the fjl-habitat container (paths below are container paths).  Example, GPU 4:
#   NAV_GPU=4 bash scripts/deploy/start_nav_servers_cuda.sh
# Env (defaults in brackets):
#   NAV_GPU [0], NAV_VO_GPU [NAV_GPU]      model and VO GPUs; together about 37 GB on one 48 GB card
#   NAV_HOST [127.0.0.1]                    bind address of both servers.  The gRPC channel is
#                                           insecure (no TLS, no auth).  fjl-habitat is on the docker
#                                           bridge with no published ports, so 127.0.0.1 is reachable
#                                           only inside the container; for a robot bind 0.0.0.0 and
#                                           SSH-tunnel to the container's bridge IP
#                                           (docs/deploy/navigation_interface.md section 2).
#   NAV_MODEL_PORT [52400], NAV_VO_PORT [52500]   the evaluation launcher's slot k binds 52400+k /
#                                           52500+k in the same container: pick free ports while it
#                                           runs (a busy port is refused below).
#   NAV_ROOT [/workspace], NAV_REPO, NAV_RPC_ROOT, INTERNNAV_MODEL_PATH, NAV_AMB3R_ROOT,
#   NAV_DA3_CHECKPOINT, NAV_CHECKPOINT, NAV_CONFIG, NAV_PYTHON   as in the launcher (PPA_EVAL_*)
#   NAV_RUNTIME_DIR [$NAV_ROOT/nav_servers/<stamp>]              logs and caches
#   NAV_SERVER_START_TIMEOUT_S [1800]
#   HEATMAPVLN_TIMING                       passed through: servers that support it add timing_ms

set -Eeuo pipefail

ROOT="${NAV_ROOT:-/workspace}"
REPO="${NAV_REPO:-$ROOT/HeatmapVLN}"
RPC_ROOT="${NAV_RPC_ROOT:-$ROOT/rpc}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$ROOT/InternNav_Model}"
AMB3R_ROOT="${NAV_AMB3R_ROOT:-$ROOT/amb3r}"
DA3_CHECKPOINT="${NAV_DA3_CHECKPOINT:-$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE}"
PYTHON="${NAV_PYTHON:-/opt/conda/bin/python}"
PPA_CHECKPOINT="${NAV_CHECKPOINT:-$ROOT/weights/ppa_refine_v2_best.pth}"
PPA_CONFIG="${NAV_CONFIG:-$REPO/configs/ppa_action_refine_v2_8gpu.yaml}"
GPU="${NAV_GPU:-0}"
VO_GPU="${NAV_VO_GPU:-$GPU}"
HOST="${NAV_HOST:-127.0.0.1}"
MODEL_PORT="${NAV_MODEL_PORT:-52400}"
VO_PORT="${NAV_VO_PORT:-52500}"
RUNTIME_DIR="${NAV_RUNTIME_DIR:-$ROOT/nav_servers/$(date +%Y%m%d_%H%M%S)_$$}"
START_TIMEOUT_S="${NAV_SERVER_START_TIMEOUT_S:-1800}"

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
RPC_PYTHONPATH="$RPC_ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

die() { printf '[nav-servers] ERROR: %s\n' "$*" >&2; exit 2; }

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$VO_SERVER" "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
  [[ -s "$file" ]] || die "missing file: $file"
done
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_MODEL_PATH"; do
  [[ -d "$directory" ]] || die "missing directory: $directory"
done
[[ -x "$PYTHON" ]] || die "missing executable: $PYTHON"
for value in "$GPU" "$VO_GPU" "$MODEL_PORT" "$VO_PORT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "not a non-negative integer: $value"
done

# Placeholders the train config expands on load; the servers never read them.
PLACEHOLDER_DIR="$RUNTIME_DIR/config_placeholders"
mkdir -p "$PLACEHOLDER_DIR" "$RUNTIME_DIR"/model/{tmp,xdg,hf,torch_extensions,triton,matplotlib} "$RUNTIME_DIR"/vo/{tmp,xdg,hf,triton}
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
export DA3_DISABLE_XFORMERS=1 DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

probe_host="$HOST"
[[ "$probe_host" == "0.0.0.0" ]] && probe_host=127.0.0.1
# gRPC binds with SO_REUSEPORT, so a port another server already listens on (an evaluation slot)
# would be shared silently, splitting its connections: refuse a port that answers.
for port in "$MODEL_PORT" "$VO_PORT"; do
  if "$PYTHON" - "$probe_host" "$port" <<'PY'
import socket
import sys

try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2).close()
except OSError:
    raise SystemExit(1)
PY
  then
    die "port $port on $probe_host already has a listener (an evaluation slot?); set NAV_MODEL_PORT / NAV_VO_PORT"
  fi
done

MODEL_PID="" VO_PID=""
stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_pid "$MODEL_PID"
  stop_pid "$VO_PID"
  echo "[nav-servers] stopped; logs in $RUNTIME_DIR"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

env PYTHONPATH="$RPC_PYTHONPATH" CUDA_VISIBLE_DEVICES="$GPU" \
  TMPDIR="$RUNTIME_DIR/model/tmp" XDG_CACHE_HOME="$RUNTIME_DIR/model/xdg" HF_HOME="$RUNTIME_DIR/model/hf" \
  TORCH_EXTENSIONS_DIR="$RUNTIME_DIR/model/torch_extensions" TRITON_CACHE_DIR="$RUNTIME_DIR/model/triton" \
  MPLCONFIGDIR="$RUNTIME_DIR/model/matplotlib" HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0 \
  "$PYTHON" -u "$MODEL_SERVER" \
    --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT" \
    --internnav_model_path "$INTERNNAV_MODEL_PATH" \
    --gpu_id 0 --host "$HOST" --port "$MODEL_PORT" --workers 1 \
    --require_deterministic_sampling --require_ppa_online_amb3r \
    --log_level INFO >"$RUNTIME_DIR/model.log" 2>&1 &
MODEL_PID=$!
env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH" CUDA_VISIBLE_DEVICES="$VO_GPU" \
  TMPDIR="$RUNTIME_DIR/vo/tmp" XDG_CACHE_HOME="$RUNTIME_DIR/vo/xdg" HF_HOME="$RUNTIME_DIR/vo/hf" \
  TRITON_CACHE_DIR="$RUNTIME_DIR/vo/triton" \
  "$PYTHON" -u "$VO_SERVER" \
    --repo "$REPO" --amb3r-root "$AMB3R_ROOT" \
    --da3-checkpoint "$DA3_CHECKPOINT" --device cuda:0 \
    --host "$HOST" --port "$VO_PORT" \
    --map-init-window 20 --map-every 8 --max-history 8 \
    --resolution 518 392 --translation-scale 1.0 \
    --max-frames-limit 4096 --max-message-mb 32 \
    --log-level INFO >"$RUNTIME_DIR/vo.log" 2>&1 &
VO_PID=$!
echo "[nav-servers] model pid=$MODEL_PID gpu=$GPU, vo pid=$VO_PID gpu=$VO_GPU; logs in $RUNTIME_DIR"

deadline=$(( $(date +%s) + START_TIMEOUT_S ))
while true; do
  kill -0 "$MODEL_PID" 2>/dev/null || { tail -80 "$RUNTIME_DIR/model.log" >&2; die "model server exited"; }
  kill -0 "$VO_PID" 2>/dev/null || { tail -80 "$RUNTIME_DIR/vo.log" >&2; die "VO server exited"; }
  if PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - "$probe_host:$MODEL_PORT" "$probe_host:$VO_PORT" <<'PY' >/dev/null 2>&1
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
  (( $(date +%s) < deadline )) || die "servers not ready after ${START_TIMEOUT_S}s"
  sleep 10
done
grep -F "Formal PPA online AMB3R runtime enabled" "$RUNTIME_DIR/model.log" >/dev/null \
  || die "model server lacks the PPA preflight line"
echo "[nav-servers] READY model=$HOST:$MODEL_PORT vo=$HOST:$VO_PORT (Ctrl-C stops both)"

# bash 5.0 in the container has no `wait -n -p`: poll both.
while kill -0 "$MODEL_PID" 2>/dev/null && kill -0 "$VO_PID" 2>/dev/null; do
  sleep 5
done
die "a server exited; see $RUNTIME_DIR/model.log and $RUNTIME_DIR/vo.log"
