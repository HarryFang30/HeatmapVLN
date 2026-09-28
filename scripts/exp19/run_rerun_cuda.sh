#!/usr/bin/env bash
# EXP-19 on the RTX 4090 box: closed-loop rerun of listed cases with the traced model server.
#
# The C500 launcher (scripts/exp19/run_rerun.sh, which made runs/main) ported to the
# CUDA deployment of scripts/run_ppa_r2r_val_unseen_cuda.sh (EXP-18 branch, validated
# on the box on 2026-09-28; docs/ops/deploy_rtx4090.md there).  Everything that reaches
# a result is the C500 rerun's: the same deployed checkpoint (sha256 0b5a0644...6d69),
# config, locked-plan tools, model / AMB3R-VO server flags, health wait, client flags
# (seed 42, trajectory_selection mean, auto_stop_distance 0, ...), the traced server
# (scripts/exp19/rpc_model_server_trace.py) and the read-only step trace
# (--step_state_trace_dir).  Only the platform parts are the CUDA launcher's:
#   * no MACA environment and no bundled X11: one interpreter (/opt/conda/bin/python)
#     for servers and client, the system Xvfb with llvmpipe, probed over TCP;
#   * config placeholders (PPA_DATA_ROOT, ...) point at an empty dir: the eval never
#     reads them (the C500 pointed them at training data that is not on this box);
#   * the GPU-free check reads nvidia-smi (the box is shared: only free GPUs);
#   * bash 5.0 in the container has no `wait -n -p`: clients are polled instead.
# The run layout, the per-episode validation and runs/<run>/DONE are run_rerun.sh's
# (DONE adds "platform").  Runs are never resumed or overwritten.
#
# Meant to run inside the fjl-habitat container (container paths):
#   cd <EXP19_ROOT>/src_<sha>
#   EXP19_SRC=$PWD EXP19_ROOT=/workspace/exp19_behavior_viz_4090 EXP19_RUN=main4090 EXP19_GPUS=4,5 \
#     bash scripts/exp19/run_rerun_cuda.sh
# A dry run (EXP19_DRY_RUN=1) runs every check and prints every command.
# Env:
#   EXP19_SRC       archived source dir (git archive <sha> + .exp19_git_sha); required, and
#                   this launcher must be the copy inside it
#   EXP19_RUN       run name; runs/<run> must not exist yet; required
#   EXP19_GPUS      comma list of 1..6 GPU ids (nvidia-smi / PCI bus order), one rank each; required.
#                   Every listed GPU must be free: no compute process, <= 500 MiB used (nvidia-smi inside
#                   fjl-habitat lists the processes of every container on the host)
#   EXP19_ALLOW_BUSY_GPU=1   launch on busy GPUs anyway (the check then only warns)
#   EXP19_ROOT      artifacts root; required (there is no shared workspace default on this box)
#   EXP19_LISTS     dir holding exactly gpu0.json .. gpu<n-1>.json (default EXP19_ROOT/cases/episode_lists)
#   EXP19_WORKSPACE deployment root (default /workspace): rpc/, InternNav_Model/, amb3r/,
#                   weights/ppa_refine_v2_best.pth, evaluation_plans/, R2R_VLNCE_v1-3_preprocessed/
#   EXP19_SCENES_DIR   parent of mp3d/<scene>/<scene>.glb (default /dataset)
#   EXP19_PYTHON    interpreter for servers, client and checks (default /opt/conda/bin/python)
#   EXP19_TRACE_DIAGNOSTICS   1 (default) | 0, passed to the trace server
#   EXP19_SERVER_START_TIMEOUT_S   1800
#   EXP19_RUNTIME_ROOT   absolute dir for the servers' scratch caches (default: inside the run dir)
#   EXP19_DRY_RUN=1 run every check and print every command; start nothing, write nothing
#   All path variables must be absolute.
set -euo pipefail
# A closed stdout/stderr (dropped ssh, `| head`) must fail a write, not kill the launcher
# before its cleanup has stopped the servers; log()/die() writes are best effort.
trap '' PIPE

W="${EXP19_WORKSPACE:-/workspace}"
SRC="${EXP19_SRC:?set EXP19_SRC to the archived source dir (git archive <sha> + .exp19_git_sha)}"
RUN="${EXP19_RUN:?set EXP19_RUN to a new run name}"
GPU_CSV="${EXP19_GPUS:?set EXP19_GPUS to 1-6 comma-separated GPU ids}"
EXP_ROOT="${EXP19_ROOT:?set EXP19_ROOT to the artifacts root}"
LISTS_DIR="${EXP19_LISTS:-$EXP_ROOT/cases/episode_lists}"
SCENES_DIR="${EXP19_SCENES_DIR:-/dataset}"
PYTHON="${EXP19_PYTHON:-/opt/conda/bin/python}"
TRACE_DIAGNOSTICS="${EXP19_TRACE_DIAGNOSTICS:-1}"
SERVER_START_TIMEOUT_S="${EXP19_SERVER_START_TIMEOUT_S:-1800}"
DRY_RUN="${EXP19_DRY_RUN:-0}"
ALLOW_BUSY_GPU="${EXP19_ALLOW_BUSY_GPU:-0}"
RUNTIME_ROOT="${EXP19_RUNTIME_ROOT:-}"
for var in EXP19_WORKSPACE EXP19_SRC EXP19_ROOT EXP19_LISTS EXP19_SCENES_DIR EXP19_PYTHON EXP19_RUNTIME_ROOT; do
  if [[ -n "${!var:-}" && "${!var}" != /* ]]; then
    printf '[exp19-rerun-cuda] ERROR: %s must be an absolute path, got %s\n' "$var" "${!var}" >&2 || true
    exit 2
  fi
done
if [[ -d "$SRC" ]]; then SRC=$(cd "$SRC" && pwd -P); fi  # a missing SRC is reported by the preflight
RUN_DIR="$EXP_ROOT/runs/$RUN"
LAUNCHER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

# ---- the deployed stack (the CUDA launcher's layout on the box) ----
RPC_ROOT="$W/rpc"
INTERNNAV_MODEL_PATH="$W/InternNav_Model"
AMB3R_ROOT="$W/amb3r"
DA3_CHECKPOINT="$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE"
PPA_CHECKPOINT="$W/weights/ppa_refine_v2_best.pth"
PPA_CHECKPOINT_SHA256=0b5a06444736ae0bbda4765d2c871d8d875989af7149c4a73470bf25288c6d69  # = the C500 best.pth
PPA_CONFIG="$SRC/configs/ppa_action_refine_v2_8gpu.yaml"
LOCKED_PLAN="$W/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
DATASET="$W/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
PLACEHOLDER_DIR="$EXP_ROOT/config_placeholders"
XVFB_BIN="${EXP19_XVFB:-$(command -v Xvfb || true)}"

MODEL_SERVER="$SRC/scripts/exp19/rpc_model_server_trace.py"
DEPLOYED_SERVER="$SRC/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$SRC/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
CLIENT="$SRC/scripts/evaluation/r2r_val_unseen.py"
STEP_TRACE="$SRC/scripts/exp19/step_trace.py"
FINGERPRINT_TOOL="$SRC/scripts/tools/source_fingerprint.py"

PROTOCOL_SEED=42
MODEL_PORT_BASE=52640
VO_PORT_BASE=52740
DISPLAY_BASE=380
MAX_RANKS=6
SERVER_STAGGER_S=15
RPC_TIMEOUT_MS=600000
GPU_BUSY_MIB=500  # an idle 4090 shows ~4 MiB; nvidia-smi in the container lists every host process
PPA_EVIDENCE="Formal PPA online AMB3R runtime enabled"
TRACE_EVIDENCE="[exp19-trace] tracing every plan_panoramic call to "  # printed by the trace entry point

# Placeholders the train config expands on load; the eval never reads them.
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$W" HEATMAPVLN_MP3D_ROOT="$SCENES_DIR/mp3d"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
# The container's .bashrc sets a proxy that does not exist; the rerun is offline.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

RPC_PYTHONPATH="$LOCKED_PLAN/tools:$RPC_ROOT/src:$SRC"

read -r -d '' HEALTH_PY <<'PY' || true
import sys
from vla_rpc.client import VLAClient

for address, expected in (
    (sys.argv[1], "ppa-online-amb3r-v1"),
    (sys.argv[2], "json+jpeg"),
):
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

declare -a GPUS=() MODEL_PIDS=() VO_PIDS=() CLIENT_PIDS=() XVFB_PIDS=() CLIENT_RC=() PROBLEMS=() CMD=()
OWN_RUN_DIR=0

log() {
  local line
  line="[exp19-rerun-cuda] $(date -u +%FT%TZ) $*"
  printf '%s\n' "$line" || true
  if [[ "$OWN_RUN_DIR" == 1 ]]; then printf '%s\n' "$line" >> "$RUN_DIR/launcher.log" || true; fi
}
die() {
  local line="[exp19-rerun-cuda] ERROR: $*"
  printf '%s\n' "$line" >&2 || true
  if [[ "$OWN_RUN_DIR" == 1 ]]; then printf '%s\n' "$line" >> "$RUN_DIR/launcher.log" || true; fi
  exit 2
}
problem() { PROBLEMS+=("$*"); }
need_file() { [[ -s "$1" ]] || problem "missing file: $1"; }
need_dir() { [[ -d "$1" ]] || problem "missing directory: $1"; }
need_exec() { [[ -n "$1" && -x "$1" ]] || problem "missing executable: ${1:-Xvfb}"; }

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
  set +e
  trap - EXIT
  trap '' INT TERM HUP
  if [[ "$OWN_RUN_DIR" == 1 && ! -e "$RUN_DIR/DONE" ]]; then
    log "stopping every process (exit $status) before DONE was written"
  fi
  for pid in "${CLIENT_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP

rank_dir() { printf '%s/gpu%d' "$RUN_DIR" "$1"; }
runtime_dir() {
  if [[ -n "$RUNTIME_ROOT" ]]; then printf '%s/%s/gpu%d' "$RUNTIME_ROOT" "$RUN" "$1"
  else printf '%s/runtime' "$(rank_dir "$1")"; fi
}
display_of() { printf '127.0.0.1:%d.0' "$((DISPLAY_BASE + $1))"; }
qcmd() { printf '%q ' "${CMD[@]}"; }

build_xvfb_cmd() {  # $1 = rank; the CUDA launcher's Xvfb line
  CMD=(env LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast
    "$XVFB_BIN" ":$((DISPLAY_BASE + $1))" -screen 0 1024x768x24
    -nolock -nolisten unix -listen tcp +iglx -ac)
}

build_model_cmd() {  # $1 = rank
  local dir runtime
  dir=$(rank_dir "$1")
  runtime="$(runtime_dir "$1")/model"
  CMD=(env PYTHONPATH="$RPC_PYTHONPATH" CUDA_VISIBLE_DEVICES="${GPUS[$1]}"
    TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg" HF_HOME="$runtime/hf"
    TORCH_EXTENSIONS_DIR="$runtime/torch_extensions"
    TRITON_CACHE_DIR="$runtime/triton" MPLCONFIGDIR="$runtime/matplotlib"
    HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0
    EXP19_TRACE_DIR="$dir/trace" EXP19_TRACE_DIAGNOSTICS="$TRACE_DIAGNOSTICS"
    "$PYTHON" -u "$MODEL_SERVER"
    --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT"
    --internnav_model_path "$INTERNNAV_MODEL_PATH"
    --gpu_id 0 --host 127.0.0.1 --port "$((MODEL_PORT_BASE + $1))" --workers 1
    --require_deterministic_sampling --require_ppa_online_amb3r
    --log_level INFO)
}

build_vo_cmd() {  # $1 = rank
  local runtime
  runtime="$(runtime_dir "$1")/vo"
  CMD=(env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH"
    CUDA_VISIBLE_DEVICES="${GPUS[$1]}"
    TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg"
    HF_HOME="$runtime/hf" TRITON_CACHE_DIR="$runtime/triton"
    "$PYTHON" -u "$VO_SERVER"
    --repo "$SRC" --amb3r-root "$AMB3R_ROOT"
    --da3-checkpoint "$DA3_CHECKPOINT" --device cuda:0
    --host 127.0.0.1 --port "$((VO_PORT_BASE + $1))"
    --map-init-window 20 --map-every 8 --max-history 8
    --resolution 518 392 --translation-scale 1.0
    --max-frames-limit 4096 --max-message-mb 32
    --log-level INFO)
}

build_client_cmd() {  # $1 = rank; flags identical to the eval's except data/list/output/steps
  local dir
  dir=$(rank_dir "$1")
  CMD=(env PYTHONPATH="$RPC_PYTHONPATH"
    DISPLAY="$(display_of "$1")" CUDA_VISIBLE_DEVICES="${GPUS[$1]}" HABITAT_GL_GPU_ID=0
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast
    HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1
    "$PYTHON" -u "$CLIENT"
    --config "$PPA_CONFIG"
    --rpc_server "127.0.0.1:$((MODEL_PORT_BASE + $1))"
    --history_pose_source amb3r_vo_da3
    --amb3r_vo_rpc_server "127.0.0.1:$((VO_PORT_BASE + $1))"
    --amb3r_vo_rpc_timeout_ms "$RPC_TIMEOUT_MS"
    --amb3r_vo_rpc_jpeg_quality 95
    --rpc_timeout_ms "$RPC_TIMEOUT_MS" --rpc_jpeg_quality 90
    --rpc_protocol_seed "$PROTOCOL_SEED" --rpc_require_deterministic_sampling
    --rpc_policy_mode heatmapvln
    --scenes_dir "$SCENES_DIR"
    --data_path "$DATASET"
    --dataset_split val_unseen --episode_list "$dir/episode_list.json"
    --output_path "$dir/client_out" --sim_gpu_id 0
    --resize_w 384 --resize_h 384 --num_history 8
    --max_steps_per_episode 500 --max_system2_calls_per_episode 0
    --auto_stop_distance 0 --trajectory_selection mean
    --trajectory_x_sign 1 --trajectory_heading_alignment none
    --system1_coord_order generated --no-pano_recenter_before_system1
    --no-debug_input_trace --debug_save_input_images 0 --resume
    --step_state_trace_dir "$dir/steps")
}

emit_commands() {
  local j dir
  echo "# EXP-19 rerun (CUDA) '$RUN': src=$SRC git_sha=$GIT_SHA fingerprint=$SOURCE_FINGERPRINT"
  echo "# global env: PPA_DATA_ROOT=$PPA_DATA_ROOT INTERNNAV_MODEL_PATH=$INTERNNAV_MODEL_PATH" \
    "HEATMAPVLN_FJL_ROOT=$HEATMAPVLN_FJL_ROOT HEATMAPVLN_MP3D_ROOT=$HEATMAPVLN_MP3D_ROOT" \
    "HEATMAPVLN_SOURCE_FINGERPRINT=$SOURCE_FINGERPRINT CUDA_DEVICE_ORDER=PCI_BUS_ID" \
    "USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false DA3_DISABLE_XFORMERS=1" \
    "DA3_SDPA_QUERY_CHUNK_SIZE=256 PYTHONDONTWRITEBYTECODE=1 (http(s)_proxy unset)"
  echo "cd $(printf '%q' "$SRC")"
  for j in "${!GPUS[@]}"; do
    dir=$(rank_dir "$j")
    echo "# ---- rank $j: GPU ${GPUS[$j]}, display $(display_of "$j"), model :$((MODEL_PORT_BASE + j)), VO :$((VO_PORT_BASE + j))"
    echo "cp $(printf '%q' "$LISTS_DIR/gpu$j.json") $(printf '%q' "$dir/episode_list.json")"
    build_xvfb_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/xvfb.log") 2>&1 &"
    build_model_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/model.log") 2>&1 &"
    build_vo_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/vo.log") 2>&1 &"
    echo "# wait: PYTHONPATH=$RPC_PYTHONPATH $PYTHON - 127.0.0.1:$((MODEL_PORT_BASE + j)) 127.0.0.1:$((VO_PORT_BASE + j)) <<HEALTH_PY" \
      "(every 10 s, <= ${SERVER_START_TIMEOUT_S} s); then grep -F '$PPA_EVIDENCE' and '$TRACE_EVIDENCE' in $dir/logs/model.log"
    build_client_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/client.log") 2>&1 &"
  done
  echo "# after all clients: fingerprint re-check, per-episode validation -> $RUN_DIR/DONE"
}

port_in_use() {
  timeout 2 bash -c 'exec 3<>"/dev/tcp/127.0.0.1/$1"' _ "$1" >/dev/null 2>&1
}
display_in_use() {  # $1 = display number (lock file, socket or TCP listener; no X client binary needed)
  [[ -e "/tmp/.X$1-lock" || -e "/tmp/.X11-unix/X$1" ]] && return 0
  port_in_use "$((6000 + $1))"
}
rank_conflicts() {
  port_in_use "$((MODEL_PORT_BASE + $1))" && echo "model port 127.0.0.1:$((MODEL_PORT_BASE + $1)) is in use"
  port_in_use "$((VO_PORT_BASE + $1))" && echo "VO port 127.0.0.1:$((VO_PORT_BASE + $1)) is in use"
  display_in_use "$((DISPLAY_BASE + $1))" && echo "display :$((DISPLAY_BASE + $1)) is in use"
  return 0
}

# One line per listed GPU that is not free: any compute process on it, or > $1 MiB used.
# nvidia-smi indices follow the PCI bus order, as CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA's.
gpu_busy_problems() {  # $1 = MiB limit, $2.. = GPU ids
  local limit="$1" gpus apps gpu row used uuid procs
  shift
  gpus=$(timeout 60 nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>&1) \
    || { echo "nvidia-smi failed; cannot check that GPU(s) $* are free: $gpus"; return 0; }
  apps=$(timeout 60 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>&1) \
    || { echo "nvidia-smi failed to list compute processes: $apps"; return 0; }
  for gpu in "$@"; do
    row=$(awk -F', *' -v g="$gpu" '$1 == g' <<< "$gpus")
    if [[ -z "$row" ]]; then
      echo "GPU $gpu is not listed by nvidia-smi"
      continue
    fi
    uuid=$(awk -F', *' '{print $2}' <<< "$row")
    used=$(awk -F', *' '{print $3}' <<< "$row")
    procs=$(grep -cF "$uuid" <<< "$apps" || true)
    if [[ ! "$used" =~ ^[0-9]+$ ]]; then
      echo "GPU $gpu: no memory use found in the nvidia-smi output ($row)"
    elif (( procs > 0 || used > limit )); then
      echo "GPU $gpu is busy: $procs compute process(es), $used MiB used (free = no process and <= $limit MiB)"
    fi
  done
  return 0
}

# ------------------------------------------------------------------ preflight
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "EXP19_DRY_RUN must be 0 or 1, got '$DRY_RUN'"
[[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || problem "EXP19_RUN must be a plain name ([A-Za-z0-9._-]), got '$RUN'"
[[ "$TRACE_DIAGNOSTICS" == 0 || "$TRACE_DIAGNOSTICS" == 1 ]] || problem "EXP19_TRACE_DIAGNOSTICS must be 0 or 1"
[[ "$ALLOW_BUSY_GPU" == 0 || "$ALLOW_BUSY_GPU" == 1 ]] || problem "EXP19_ALLOW_BUSY_GPU must be 0 or 1"
[[ "$SERVER_START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || problem "EXP19_SERVER_START_TIMEOUT_S must be a positive integer"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if (( ${#GPUS[@]} < 1 || ${#GPUS[@]} > MAX_RANKS )); then
  problem "EXP19_GPUS must list 1..$MAX_RANKS GPU ids, got '$GPU_CSV'"
fi
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || problem "invalid GPU id '$gpu' in EXP19_GPUS"
done
[[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "${#GPUS[@]}" ]] || problem "GPU ids must be unique: $GPU_CSV"

GIT_SHA=unknown
SOURCE_FINGERPRINT=unavailable
if [[ ! -d "$SRC" ]]; then
  problem "EXP19_SRC is not a directory: $SRC"
else
  if [[ -d "$W/HeatmapVLN" && "$SRC" == "$(cd "$W/HeatmapVLN" && pwd -P)" ]]; then
    problem "EXP19_SRC is the deployment checkout; git archive the commit to $EXP_ROOT/src_<sha> (ledger §5 lesson 27)"
  fi
  [[ "$LAUNCHER_ROOT" == "$SRC" ]] || problem "run the launcher from the archived source: bash $SRC/scripts/exp19/run_rerun_cuda.sh (this one is under $LAUNCHER_ROOT)"
  need_file "$SRC/.exp19_git_sha"
  [[ -s "$SRC/.exp19_git_sha" ]] && GIT_SHA=$(tr -d '[:space:]' < "$SRC/.exp19_git_sha")
fi

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$DEPLOYED_SERVER" "$VO_SERVER" "$CLIENT" \
  "$STEP_TRACE" "$FINGERPRINT_TOOL" "$DATASET" "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
  need_file "$file"
done
for directory in "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_MODEL_PATH" "$SCENES_DIR/mp3d" "$LOCKED_PLAN/tools" "$LISTS_DIR"; do
  need_dir "$directory"
done
for executable in "$PYTHON" "$XVFB_BIN"; do
  need_exec "$executable"
done
command -v nvidia-smi >/dev/null 2>&1 || problem "nvidia-smi not found: cannot check that the GPUs are free"
[[ -s "$INTERNNAV_MODEL_PATH/config.json" ]] || problem "InternNav model has no config.json: $INTERNNAV_MODEL_PATH"
if [[ -s "$CLIENT" ]] && ! grep -qF -- '--step_state_trace_dir' "$CLIENT"; then
  problem "client lacks --step_state_trace_dir (source copy predates the EXP-19 step trace): $CLIENT"
fi
if [[ -s "$MODEL_SERVER" ]] && ! grep -qF 'EXP19_TRACE_DIR' "$MODEL_SERVER"; then
  problem "trace server does not read EXP19_TRACE_DIR: $MODEL_SERVER"
fi
if [[ -e "$RUN_DIR" ]]; then
  problem "run dir already exists (choose a new EXP19_RUN; runs are never resumed): $RUN_DIR"
fi
if [[ -s "$PPA_CHECKPOINT" ]]; then
  got=$(sha256sum "$PPA_CHECKPOINT" | awk '{print $1}')
  [[ "$got" == "$PPA_CHECKPOINT_SHA256" ]] \
    || problem "checkpoint sha256 $got is not the deployed v2 checkpoint's $PPA_CHECKPOINT_SHA256: $PPA_CHECKPOINT"
fi

for j in "${!GPUS[@]}"; do
  while IFS= read -r conflict; do
    [[ -z "$conflict" ]] || problem "$conflict"
  done <<< "$(rank_conflicts "$j")"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  busy=()
  while IFS= read -r line; do
    [[ -z "$line" ]] || busy+=("$line")
  done <<< "$(gpu_busy_problems "$GPU_BUSY_MIB" "${GPUS[@]}")"
  for line in "${busy[@]+"${busy[@]}"}"; do
    if [[ "$ALLOW_BUSY_GPU" == 1 ]]; then
      log "WARNING (EXP19_ALLOW_BUSY_GPU=1): $line"
    else
      problem "$line (EXP19_ALLOW_BUSY_GPU=1 launches anyway)"
    fi
  done
  ((${#busy[@]})) || log "GPU(s) $GPU_CSV free (nvidia-smi: no compute process, <= $GPU_BUSY_MIB MiB used)"
fi

if [[ -x "$PYTHON" && -d "$LISTS_DIR" && -s "$DATASET" ]]; then
  list_rc=0
  list_report=$("$PYTHON" - "$LISTS_DIR" "${#GPUS[@]}" "$DATASET" "$SCENES_DIR" <<'PY' 2>&1
import gzip, json, re, sys
from pathlib import Path

lists_dir, n, dataset, scenes = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
problems = []
present = sorted(p.name for p in lists_dir.glob("gpu*.json"))
wanted = [f"gpu{j}.json" for j in range(n)]
extra = [name for name in present if name not in wanted]
if extra:
    problems.append(f"{lists_dir} holds lists for more ranks than EXP19_GPUS: {extra} (their episodes would never run)")
with gzip.open(dataset, "rt") as f:
    known = {(Path(e["scene_id"]).stem, int(e["episode_id"])) for e in json.load(f)["episodes"]}
seen = {}
for j, name in enumerate(wanted):
    path = lists_dir / name
    if not path.is_file():
        problems.append(f"missing episode list {path}")
        continue
    try:
        data = json.loads(path.read_text())
        episodes = data["episodes"]
        keys = [(str(e["scene_id"]), int(e["episode_id"])) for e in episodes]
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(f"{path}: not a client episode list ({exc!r})")
        continue
    if not keys:
        problems.append(f"{path}: empty 'episodes'")
    for scene, ep in keys:
        key = f"{scene}_{ep:04d}"
        if not re.fullmatch(r"[A-Za-z0-9]+", scene):
            problems.append(f"{path}: scene_id must be the scene stem, got {scene!r}")
        elif (scene, ep) not in known:
            problems.append(f"{path}: {key} is not in {dataset}")
        elif not (scenes / "mp3d" / scene / f"{scene}.glb").is_file():
            problems.append(f"{path}: scene mesh missing for {key}")
        if key in seen:
            problems.append(f"{key} listed twice ({seen[key]} and {name})")
        seen[key] = name
    print(f"gpu{j}: {len(keys)} episodes, cohort={data.get('cohort_name')!r}: " + " ".join(f"{s}_{e:04d}" for s, e in keys))
for p in problems:
    print(f"PROBLEM: {p}")
sys.exit(1 if problems else 0)
PY
  ) || list_rc=$?
  while IFS= read -r line; do
    case "$line" in
      PROBLEM:\ *) problem "${line#PROBLEM: }" ;;
      gpu*) log "  $line" ;;
      *) [[ -z "$line" ]] || problem "episode list check: $line" ;;
    esac
  done <<< "$list_report"
  if [[ "$list_rc" -eq 0 ]]; then
    log "episode lists OK ($LISTS_DIR)"
  else
    problem "episode list check failed (exit $list_rc)"
  fi
fi

if [[ -x "$PYTHON" && -d "$RPC_ROOT/src/vla_rpc" ]]; then
  PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" -c 'from vla_rpc.client import VLAClient' >/dev/null 2>&1 \
    || problem "vla_rpc.client does not import with $PYTHON and RPC_PYTHONPATH"
fi
if [[ -x "$PYTHON" && -s "$FINGERPRINT_TOOL" ]]; then
  SOURCE_FINGERPRINT=$("$PYTHON" "$FINGERPRINT_TOOL" "$SRC") || { SOURCE_FINGERPRINT=unavailable; problem "cannot fingerprint $SRC"; }
fi
export HEATMAPVLN_SOURCE_FINGERPRINT="$SOURCE_FINGERPRINT"

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY RUN: nothing is started or written; the launcher would run:"
  emit_commands
fi
if ((${#PROBLEMS[@]})); then
  for p in "${PROBLEMS[@]}"; do printf '[exp19-rerun-cuda] PROBLEM: %s\n' "$p" >&2 || true; done
  die "${#PROBLEMS[@]} preflight problem(s); nothing was started"
fi
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY RUN OK: ${#GPUS[@]} rank(s) on GPU(s) $GPU_CSV, git_sha=$GIT_SHA, fingerprint=$SOURCE_FINGERPRINT"
  exit 0
fi

# ------------------------------------------------------------------ launch
cd "$SRC"
mkdir -p "$EXP_ROOT/runs" "$PLACEHOLDER_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR (does it exist already?)"
OWN_RUN_DIR=1
STARTED_UTC=$(date -u +%FT%TZ)
printf '%s\n' "$SOURCE_FINGERPRINT" > "$RUN_DIR/source_fingerprint.txt"
for j in "${!GPUS[@]}"; do
  dir=$(rank_dir "$j")
  mkdir -p "$dir/logs" "$dir/trace" "$dir/steps" "$dir/client_out" \
    "$(runtime_dir "$j")/model"/{tmp,xdg,hf,torch_extensions,triton,matplotlib} "$(runtime_dir "$j")/vo"/{tmp,xdg,hf,triton}
  cp "$LISTS_DIR/gpu$j.json" "$dir/episode_list.json"
done
emit_commands > "$RUN_DIR/commands.txt"
log "run=$RUN dir=$RUN_DIR src=$SRC git_sha=$GIT_SHA fingerprint=$SOURCE_FINGERPRINT gpus=$GPU_CSV diagnostics=$TRACE_DIAGNOSTICS allow_busy_gpu=$ALLOW_BUSY_GPU"

log "starting ${#GPUS[@]} Xvfb display(s)"
for j in "${!GPUS[@]}"; do
  conflicts=$(rank_conflicts "$j")
  [[ -z "$conflicts" ]] || die "rank $j: ${conflicts//$'\n'/; } (taken since the preflight)"
done
for j in "${!GPUS[@]}"; do
  dir=$(rank_dir "$j")
  build_xvfb_cmd "$j"
  "${CMD[@]}" >"$dir/logs/xvfb.log" 2>&1 &
  XVFB_PIDS[j]="$!"
  ready=0
  for _ in $(seq 1 120); do
    if timeout 5 bash -c "exec 3<>/dev/tcp/127.0.0.1/$((6000 + DISPLAY_BASE + j))" 2>/dev/null; then
      ready=1
      break
    fi
    kill -0 "${XVFB_PIDS[j]}" 2>/dev/null || break
    sleep 1
  done
  if [[ "$ready" -ne 1 ]] || ! kill -0 "${XVFB_PIDS[j]}" 2>/dev/null; then
    die "Xvfb rank $j failed; see $dir/logs/xvfb.log"
  fi
done

log "starting traced model RPC server(s)"
for j in "${!GPUS[@]}"; do
  build_model_cmd "$j"
  "${CMD[@]}" >"$(rank_dir "$j")/logs/model.log" 2>&1 &
  MODEL_PIDS[j]="$!"
  sleep "$SERVER_STAGGER_S"
done

log "starting online AMB3R RPC server(s)"
for j in "${!GPUS[@]}"; do
  build_vo_cmd "$j"
  "${CMD[@]}" >"$(rank_dir "$j")/logs/vo.log" 2>&1 &
  VO_PIDS[j]="$!"
  sleep "$SERVER_STAGGER_S"
done

log "waiting for $((2 * ${#GPUS[@]})) RPC server(s)"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
for j in "${!GPUS[@]}"; do
  dir=$(rank_dir "$j")
  model_addr="127.0.0.1:$((MODEL_PORT_BASE + j))"
  vo_addr="127.0.0.1:$((VO_PORT_BASE + j))"
  while true; do
    kill -0 "${MODEL_PIDS[j]}" 2>/dev/null || {
      tail -120 "$dir/logs/model.log" >&2 || true
      die "model server rank $j exited"
    }
    kill -0 "${VO_PIDS[j]}" 2>/dev/null || {
      tail -120 "$dir/logs/vo.log" >&2 || true
      die "VO server rank $j exited"
    }
    if PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - "$model_addr" "$vo_addr" <<< "$HEALTH_PY" >/dev/null 2>&1; then
      break
    fi
    (( $(date +%s) < deadline )) || die "RPC startup timeout at rank $j"
    sleep 10
  done
  grep -F "$PPA_EVIDENCE" "$dir/logs/model.log" >/dev/null \
    || die "model rank $j lacks PPA preflight evidence"
  grep -F "$TRACE_EVIDENCE" "$dir/logs/model.log" >/dev/null \
    || die "model rank $j lacks EXP-19 trace evidence (is $MODEL_SERVER the traced entry point?)"
  log "rank=$j gpu=${GPUS[j]} model=$model_addr vo=$vo_addr ready"
done

log "starting ${#GPUS[@]} client(s) (protocol seed $PROTOCOL_SEED)"
for j in "${!GPUS[@]}"; do
  build_client_cmd "$j"
  "${CMD[@]}" >"$(rank_dir "$j")/logs/client.log" 2>&1 &
  CLIENT_PIDS[j]="$!"
done

# Every rank runs to its end; its servers stop as soon as its client exits.  bash 5.0
# has no `wait -n -p`, so the clients are polled and reaped with a plain `wait <pid>`.
declare -A DONE_RANK=()
while (( ${#DONE_RANK[@]} < ${#GPUS[@]} )); do
  for j in "${!GPUS[@]}"; do
    [[ -z "${DONE_RANK[$j]:-}" ]] || continue
    kill -0 "${CLIENT_PIDS[j]}" 2>/dev/null && continue
    rc=0
    wait "${CLIENT_PIDS[j]}" || rc=$?
    CLIENT_RC[j]="$rc"
    DONE_RANK[$j]=1
    if [[ "$rc" -ne 0 ]]; then
      tail -80 "$(rank_dir "$j")/logs/client.log" >&2 || true
    fi
    stop_pid "${MODEL_PIDS[j]}"
    stop_pid "${VO_PIDS[j]}"
    stop_pid "${XVFB_PIDS[j]}"
    log "rank=$j client exited rc=$rc; remaining=$(( ${#GPUS[@]} - ${#DONE_RANK[@]} ))"
  done
  if (( ${#DONE_RANK[@]} < ${#GPUS[@]} )); then sleep 5; fi
done

fingerprint_end=$("$PYTHON" "$FINGERPRINT_TOOL" "$SRC") || fingerprint_end=unavailable
[[ "$fingerprint_end" == "$SOURCE_FINGERPRINT" ]] \
  || log "WARNING: the source tree changed during the run ($SOURCE_FINGERPRINT -> $fingerprint_end)"

rank_specs=()
for j in "${!GPUS[@]}"; do rank_specs+=("$j:${GPUS[j]}:${CLIENT_RC[j]}"); done
PLATFORM_JSON=$("$PYTHON" - <<'PY' 2>/dev/null || echo '{}'
import json, subprocess
info = {"accelerator": "cuda"}
try:
    import torch
    info.update(torch=torch.__version__, cuda=torch.version.cuda)
except Exception as exc:  # recorded, not fatal
    info["torch_error"] = repr(exc)
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout
    info["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
except Exception as exc:
    info["nvidia_smi_error"] = repr(exc)
print(json.dumps(info))
PY
)
set +e
EXP19_PLATFORM_JSON="$PLATFORM_JSON" "$PYTHON" - "$RUN_DIR" "$RUN" "$GIT_SHA" "$SRC" "$LISTS_DIR" "$SOURCE_FINGERPRINT" "$fingerprint_end" \
  "$STARTED_UTC" "$TRACE_DIAGNOSTICS" "$PROTOCOL_SEED" "${rank_specs[@]}" <<'PY' 2>&1 | tee -a "$RUN_DIR/launcher.log"
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, run, git_sha, src, lists_dir, fp_start, fp_end, started, diagnostics, seed = sys.argv[1:11]
run_dir, src = Path(run_dir), Path(src)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def jsonl(path):
    rows = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"type": "_unparseable"})
    return rows


ranks, complete, error_lines_total = [], fp_start == fp_end, 0
for spec in sys.argv[11:]:
    j, gpu, rc = (int(x) for x in spec.split(":"))
    d = run_dir / f"gpu{j}"
    listed = json.loads((d / "episode_list.json").read_text())["episodes"]
    progress = {f"{Path(str(r['scene_id'])).stem}_{int(r['episode_id']):04d}": r
                for r in jsonl(d / "client_out" / "progress.json") if "episode_id" in r}
    rank_errors = len(jsonl(d / "trace" / "trace_errors.jsonl"))
    episodes = []
    for item in listed:
        key = f"{item['scene_id']}_{int(item['episode_id']):04d}"
        row = progress.get(key)
        vlm_calls = int(row["vlm_calls"]) if row else None
        trace = d / "trace" / key
        call_json = sorted(p.stem for p in trace.glob("call_*.json"))
        call_npz = sorted(p.stem for p in trace.glob("call_*.npz"))
        expected = [f"call_{i:03d}" for i in range(vlm_calls)] if vlm_calls is not None else None
        errors = len(jsonl(trace / "trace_errors.jsonl"))
        rank_errors += errors
        end = any(r.get("type") == "episode_end" for r in jsonl(d / "steps" / key / "steps.jsonl"))
        ok = row is not None and end and call_json == expected and call_npz == expected
        complete &= ok
        episodes.append({"ep_key": key, "progress_row": row is not None, "steps_episode_end": end,
                         "vlm_calls": vlm_calls, "trace_calls_json": len(call_json),
                         "trace_calls_npz": len(call_npz), "trace_error_lines": errors, "complete": ok})
    complete &= rc == 0
    error_lines_total += rank_errors
    ranks.append({"rank": j, "gpu": gpu, "client_exit_code": rc,
                  "episode_list_source": str(Path(lists_dir) / f"gpu{j}.json"),
                  "episode_list_sha256": sha256(d / "episode_list.json"),
                  "n_episodes": len(episodes), "n_complete": sum(e["complete"] for e in episodes),
                  "trace_error_lines": rank_errors, "episodes": episodes})
    print(f"[exp19-rerun-cuda] rank={j} gpu={gpu} rc={rc} complete={ranks[-1]['n_complete']}/{len(episodes)} "
          f"trace_error_lines={rank_errors}")
code = {rel: sha256(src / rel) for rel in (
    "scripts/exp19/run_rerun_cuda.sh", "scripts/exp19/rpc_model_server_trace.py", "scripts/exp19/step_trace.py",
    "scripts/evaluation/rpc_model_server.py", "scripts/evaluation/r2r_val_unseen.py",
    "configs/ppa_action_refine_v2_8gpu.yaml")}
try:
    platform = json.loads(os.environ.get("EXP19_PLATFORM_JSON") or "{}")
except json.JSONDecodeError:
    platform = {"raw": os.environ.get("EXP19_PLATFORM_JSON")}
done = {
    "schema": "exp19-run-done-v1", "run": run, "git_sha": git_sha, "src": str(src),
    "platform": platform,
    "source_fingerprint": {"start": fp_start, "end": fp_end, "unchanged": fp_start == fp_end},
    "code_sha256": code, "protocol_seed": int(seed), "trace_diagnostics": diagnostics == "1",
    "started_utc": started, "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "complete" if complete else "incomplete", "trace_error_lines": error_lines_total,
    "smoke_list": os.environ.get("EXP19_SMOKE_LIST"),
    "ranks": ranks,
}
(run_dir / "DONE").write_text(json.dumps(done, indent=2) + "\n")
print(f"[exp19-rerun-cuda] status={done['status']} trace_error_lines={error_lines_total} -> {run_dir / 'DONE'}")
sys.exit(0 if complete else 1)
PY
status=${PIPESTATUS[0]}
set -e
[[ "$status" -eq 0 ]] || die "run '$RUN' is incomplete (status $status); see $RUN_DIR/DONE and the per-rank logs"
log "COMPLETE: $RUN_DIR/DONE"
