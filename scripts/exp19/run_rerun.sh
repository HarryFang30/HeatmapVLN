#!/usr/bin/env bash
# EXP-19: closed-loop rerun of the pre-registered cases with the traced model server.
#
# Mirrors scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh, the launcher of the
# main-table eval (model/eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu): same
# checkpoint + config, same global env and MACA libraries, same bundled X11 with the
# Xvfb readiness check, same model / AMB3R-VO server flags, same health wait and the
# same client flags (seed 42, trajectory_selection mean, auto_stop_distance 0, ...).
# Per rank j (one per GPU in EXP19_GPUS) it runs one Xvfb, one model server, one VO
# server and one client on that GPU. Differences, all from the ledger's EXP-19
# pre-registration ("模型与栈", "设置"):
#   * at most 3 ranks (dev-machine rule); rank j runs $EXP19_LISTS/gpu<j>.json against
#     the FULL val_unseen.json.gz (the eval used its locked shard files);
#   * model server = scripts/exp19/rpc_model_server_trace.py (wraps the deployed
#     server at class level, CLI unchanged) with EXP19_TRACE_DIR=runs/<run>/gpu<j>/trace;
#   * client adds --step_state_trace_dir runs/<run>/gpu<j>/steps (read-only); never add
#     --save_trajectory_steps: its extra panorama captures change later history frames;
#   * ports 52640+j (model) / 52740+j (VO), displays :380+j; refuses if any is taken;
#   * code runs from an archived copy (EXP19_SRC, ledger §5 lesson 27), the launcher
#     itself included, and the launching shell's PYTHONPATH is not inherited;
#   * no shard merge: each rank is checked episode by episode (client progress row,
#     steps.jsonl episode_end, one call_<i>.json/.npz per System2 call) and the result
#     goes to runs/<run>/DONE with every client's exit code.
# The source fingerprint (scripts/tools/source_fingerprint.py) is exported as
# HEATMAPVLN_SOURCE_FINGERPRINT like run_exp13_system2_memory_8gpu_mxc500.sh does (the
# eval launcher does not, and no server / client reads it: here it is provenance),
# recorded, and checked again after the clients exit.
#
# Website (blank container; all paths absolute, python by absolute path):
#   cd /mnt/afs/liwenhao/agent/370910109/model/exp19_behavior_viz/src_<sha>
#   export EXP19_SRC=$PWD EXP19_RUN=main EXP19_GPUS=0,1,2
#   bash scripts/exp19/run_rerun.sh
# Dev machine (setsid-friendly: per-process output goes to log files):
#   ssh -n -f finn_cci_c500 'bash -lc "cd $EXP && mkdir -p logs && EXP19_SRC=$SRC EXP19_RUN=main \
#     EXP19_GPUS=0,1,2 setsid nohup bash $SRC/scripts/exp19/run_rerun.sh > logs/rerun_main.out 2>&1 < /dev/null &"'
# A dry run (EXP19_DRY_RUN=1) first is cheap: every check below, every command printed.
# Env:
#   EXP19_SRC       archived source dir (git archive <sha> + .exp19_git_sha); required, and
#                   this launcher must be the copy inside it
#   EXP19_RUN       run name; runs/<run> must not exist yet (MACA forwards are not
#                   repeatable, so a run is never resumed or overwritten); required
#   EXP19_GPUS      comma list of 1..3 physical GPU ids, one rank each; required. When mx-smi
#                   is available, every listed GPU must be free: no process, <= 2000 MiB used
#   EXP19_ALLOW_BUSY_GPU=1   launch on busy GPUs anyway (the check then only warns)
#   EXP19_ROOT      artifacts root (default <workspace>/model/exp19_behavior_viz)
#   EXP19_LISTS     dir holding exactly gpu0.json .. gpu<n-1>.json (default EXP19_ROOT/cases/episode_lists)
#   EXP19_SRC, EXP19_ROOT, EXP19_LISTS must be absolute paths
#   EXP19_TRACE_DIAGNOSTICS   1 (default) | 0, passed to the trace server
#   EXP19_SERVER_START_TIMEOUT_S   3600
#   EXP19_DRY_RUN=1 run every check and print every command; start nothing, write nothing
set -euo pipefail
# A closed stdout/stderr (dropped ssh, `| head`) must fail a write, not kill the launcher
# before its cleanup has stopped the servers; log()/die() writes are best effort.
trap '' PIPE

W=/mnt/afs/liwenhao/agent/370910109
SRC="${EXP19_SRC:?set EXP19_SRC to the archived source dir (git archive <sha> + .exp19_git_sha)}"
RUN="${EXP19_RUN:?set EXP19_RUN to a new run name}"
GPU_CSV="${EXP19_GPUS:?set EXP19_GPUS to 1-3 comma-separated GPU ids}"
EXP_ROOT="${EXP19_ROOT:-$W/model/exp19_behavior_viz}"
LISTS_DIR="${EXP19_LISTS:-$EXP_ROOT/cases/episode_lists}"
TRACE_DIAGNOSTICS="${EXP19_TRACE_DIAGNOSTICS:-1}"
SERVER_START_TIMEOUT_S="${EXP19_SERVER_START_TIMEOUT_S:-3600}"
DRY_RUN="${EXP19_DRY_RUN:-0}"
ALLOW_BUSY_GPU="${EXP19_ALLOW_BUSY_GPU:-0}"
# A relative path would resolve against whatever directory the website / setsid shell starts in.
for var in EXP19_SRC EXP19_ROOT EXP19_LISTS; do
  if [[ -n "${!var:-}" && "${!var}" != /* ]]; then
    printf '[exp19-rerun] ERROR: %s must be an absolute path, got %s\n' "$var" "${!var}" >&2 || true
    exit 2
  fi
done
if [[ -d "$SRC" ]]; then SRC=$(cd "$SRC" && pwd -P); fi  # a missing SRC is reported by the preflight
RUN_DIR="$EXP_ROOT/runs/$RUN"
LAUNCHER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

# ---- the deployed stack, identical to the main-table eval ----
RPC_ROOT="$W/rpc"
INTERNNAV_REPO="$W/InternNav"
INTERNNAV_MODEL_PATH="$W/InternNav-Model"
AMB3R_ROOT="$W/amb3r"
DA3_CHECKPOINT="$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE"
QWEN_PYTHON="$W/envs/qwen25/bin/python"
VLNCE_PYTHON="$W/envs/vlnce/bin/python"
PPA_CHECKPOINT="$W/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth"
PPA_CONFIG="$SRC/configs/ppa_action_refine_v2_8gpu.yaml"
# Only used to expand config placeholders, exactly as the eval launcher exports them.
PPA_TRAIN_DATA="$W/r2r_panoramic_data_v2/train"
PPA_TRAIN_CACHE="$W/data/amb3r_endpoint_v3_full_r2r"
PPA_STAGE2_OUTPUT_ROOT="$W/model/output_past_plan_action_v1_8gpu_stage2_retry1/stage2_joint"
PPA_TENSORBOARD_ROOT="$W/model/output_past_plan_action_v1_8gpu_stage2_retry1/tensorboard"
LOCKED_PLAN="$W/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
DATASET="$W/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
SCENES_DIR="$W/habitat/VLN-CE/data/scene_datasets"

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
MAX_RANKS=3
SERVER_STAGGER_S=15
RPC_TIMEOUT_MS=600000
GPU_BUSY_MIB=2000  # an idle C500 shows ~860 MiB used and no process
PPA_EVIDENCE="Formal PPA online AMB3R runtime enabled"
TRACE_EVIDENCE="[exp19-trace] tracing every plan_panoramic call to "  # printed by the trace entry point

X11_BUNDLE="$W/tools/x11_headless_bundle_ubuntu22_20260801_v4"
XVFB_BIN="$X11_BUNDLE/bin/Xvfb"
XDPYINFO_BIN="$X11_BUNDLE/bin/xdpyinfo"
XKBCOMP_BIN="$X11_BUNDLE/bin/xkbcomp"
X11_DRI_PATH="$X11_BUNDLE/dri"
X11_FONT_PATH="$X11_BUNDLE/share/fonts/misc"
X11_XKB_PATH="$X11_BUNDLE/share/X11/xkb"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export INTERNNAV_MODEL_PATH INTERNNAV_REPO
export PPA_DATA_ROOT="$PPA_TRAIN_DATA"
export PPA_AMB3R_CACHE_ROOT="$PPA_TRAIN_CACHE"
export PPA_STAGE2_OUTPUT_ROOT PPA_TENSORBOARD_ROOT
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1

RPC_PYTHONPATH="$LOCKED_PLAN/tools:$RPC_ROOT/src:$SRC:$INTERNNAV_REPO"
SERVER_LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib"
CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:$LD_LIBRARY_PATH"
TOOL_LD_LIBRARY_PATH="$X11_BUNDLE/lib:$LD_LIBRARY_PATH"

# Same readiness probe as the eval launcher: health + the capability each server advertises.
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
declare -A PID_RANK=()
OWN_RUN_DIR=0  # 1 once this launcher created RUN_DIR; only then does it write there

log() {  # best effort: a failed write must never abort the launcher (or its cleanup)
  local line
  line="[exp19-rerun] $(date -u +%FT%TZ) $*"
  printf '%s\n' "$line" || true
  if [[ "$OWN_RUN_DIR" == 1 ]]; then printf '%s\n' "$line" >> "$RUN_DIR/launcher.log" || true; fi
}
die() {
  local line="[exp19-rerun] ERROR: $*"
  printf '%s\n' "$line" >&2 || true
  if [[ "$OWN_RUN_DIR" == 1 ]]; then printf '%s\n' "$line" >> "$RUN_DIR/launcher.log" || true; fi
  exit 2
}
problem() { PROBLEMS+=("$*"); }
need_file() { [[ -s "$1" ]] || problem "missing file: $1"; }
need_dir() { [[ -d "$1" ]] || problem "missing directory: $1"; }
need_exec() { [[ -x "$1" ]] || problem "missing executable: $1"; }

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
  # Nothing may cut the stop below short: not errexit, not a second signal.
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
display_of() { printf '127.0.0.1:%d.0' "$((DISPLAY_BASE + $1))"; }
qcmd() { printf '%q ' "${CMD[@]}"; }

build_xvfb_cmd() {  # $1 = rank; run from runtime/xvfb with fd 9 on its .xkb-cache
  CMD=(env PATH="$X11_BUNDLE/bin:$PATH" LD_LIBRARY_PATH="$TOOL_LD_LIBRARY_PATH"
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
    MESA_LOADER_DRIVER_OVERRIDE=swrast
    "$XVFB_BIN" ":$((DISPLAY_BASE + $1))" -screen 0 1024x768x24
    -nolock -nolisten unix -listen tcp +iglx -ac
    -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH")
}

build_model_cmd() {  # $1 = rank
  local dir runtime
  dir=$(rank_dir "$1")
  runtime="$dir/runtime/model"
  CMD=(env PYTHONPATH="$RPC_PYTHONPATH" LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES="${GPUS[$1]}" TMPDIR="$runtime/tmp"
    XDG_CACHE_HOME="$runtime/xdg" HF_HOME="$runtime/hf"
    TORCH_EXTENSIONS_DIR="$runtime/torch_extensions"
    TRITON_CACHE_DIR="$runtime/triton" MPLCONFIGDIR="$runtime/matplotlib"
    HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0
    EXP19_TRACE_DIR="$dir/trace" EXP19_TRACE_DIAGNOSTICS="$TRACE_DIAGNOSTICS"
    "$QWEN_PYTHON" -u "$MODEL_SERVER"
    --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT"
    --internnav_model_path "$INTERNNAV_MODEL_PATH"
    --gpu_id 0 --host 127.0.0.1 --port "$((MODEL_PORT_BASE + $1))" --workers 1
    --require_deterministic_sampling --require_ppa_online_amb3r
    --log_level INFO)
}

build_vo_cmd() {  # $1 = rank
  local runtime
  runtime="$(rank_dir "$1")/runtime/vo"
  CMD=(env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH"
    LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES="${GPUS[$1]}"
    TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg"
    HF_HOME="$runtime/hf" TRITON_CACHE_DIR="$runtime/triton"
    "$QWEN_PYTHON" -u "$VO_SERVER"
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
  CMD=(env PYTHONPATH="$RPC_PYTHONPATH" LD_LIBRARY_PATH="$CLIENT_LD_LIBRARY_PATH"
    DISPLAY="$(display_of "$1")" CUDA_VISIBLE_DEVICES="${GPUS[$1]}" HABITAT_GL_GPU_ID=0
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" LIBGL_ALWAYS_SOFTWARE=1
    GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast
    HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1
    "$VLNCE_PYTHON" -u "$CLIENT"
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

emit_commands() {  # everything the launcher runs, in order, as copy-pasteable shell
  local j dir
  echo "# EXP-19 rerun '$RUN': src=$SRC git_sha=$GIT_SHA fingerprint=$SOURCE_FINGERPRINT"
  echo "# global env: MACA_HOME=$MACA_HOME PPA_DATA_ROOT=$PPA_DATA_ROOT PPA_AMB3R_CACHE_ROOT=$PPA_AMB3R_CACHE_ROOT" \
    "INTERNNAV_MODEL_PATH=$INTERNNAV_MODEL_PATH HEATMAPVLN_SOURCE_FINGERPRINT=$SOURCE_FINGERPRINT" \
    "USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false DA3_DISABLE_XFORMERS=1" \
    "DA3_SDPA_QUERY_CHUNK_SIZE=256 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "cd $(printf '%q' "$SRC")"
  for j in "${!GPUS[@]}"; do
    dir=$(rank_dir "$j")
    echo "# ---- rank $j: GPU ${GPUS[$j]}, display $(display_of "$j"), model :$((MODEL_PORT_BASE + j)), VO :$((VO_PORT_BASE + j))"
    echo "cp $(printf '%q' "$LISTS_DIR/gpu$j.json") $(printf '%q' "$dir/episode_list.json")"
    build_xvfb_cmd "$j"
    echo "(cd $(printf '%q' "$dir/runtime/xvfb") && exec 9<$(printf '%q' "$dir/runtime/xvfb/.xkb-cache") && exec $(qcmd)) > $(printf '%q' "$dir/logs/xvfb.log") 2>&1 &"
    build_model_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/model.log") 2>&1 &"
    build_vo_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/vo.log") 2>&1 &"
    echo "# wait: PYTHONPATH=$RPC_PYTHONPATH $VLNCE_PYTHON - 127.0.0.1:$((MODEL_PORT_BASE + j)) 127.0.0.1:$((VO_PORT_BASE + j)) <<HEALTH_PY" \
      "(every 10 s, <= ${SERVER_START_TIMEOUT_S} s); then grep -F '$PPA_EVIDENCE' and '$TRACE_EVIDENCE' in $dir/logs/model.log"
    build_client_cmd "$j"
    echo "$(qcmd)> $(printf '%q' "$dir/logs/client.log") 2>&1 &"
  done
  echo "# after all clients: fingerprint re-check, per-episode validation -> $RUN_DIR/DONE"
}

port_in_use() {  # something accepts connections on 127.0.0.1:$1
  timeout 2 bash -c 'exec 3<>"/dev/tcp/127.0.0.1/$1"' _ "$1" >/dev/null 2>&1
}
display_in_use() {  # $1 = display number
  [[ -e "/tmp/.X$1-lock" || -e "/tmp/.X11-unix/X$1" ]] && return 0
  port_in_use "$((6000 + $1))" && return 0
  env LD_LIBRARY_PATH="$TOOL_LD_LIBRARY_PATH" DISPLAY="127.0.0.1:$1.0" \
    timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1
}
rank_conflicts() {  # $1 = rank; one line per port / display of that rank already taken
  port_in_use "$((MODEL_PORT_BASE + $1))" && echo "model port 127.0.0.1:$((MODEL_PORT_BASE + $1)) is in use"
  port_in_use "$((VO_PORT_BASE + $1))" && echo "VO port 127.0.0.1:$((VO_PORT_BASE + $1)) is in use"
  display_in_use "$((DISPLAY_BASE + $1))" && echo "display :$((DISPLAY_BASE + $1)) is in use"
  return 0
}

# stdin: plain `mx-smi` output. One line per GPU it lists: "<gpu> <used MiB|NA> <processes>".
# GPU table: '| <gpu>  MetaX C500  Off | <bus-id> | ...' then
# '| 37C  58W / 350W  P0 | 858/65536 MiB | Available |'; after 'Process:', one row per
# process '|  <gpu>  <pid>  <name>  <MiB>  |' (or '|  no process found  |').
mx_smi_usage() {
  awk '
    function first_field(line) { sub(/^\|[[:space:]]*/, "", line); split(line, f, /[[:space:]]+/); return f[1] }
    /^\|[[:space:]]*Process:/ { in_proc = 1; next }
    !in_proc && /^\|[[:space:]]*[0-9]+[[:space:]]+[A-Za-z]/ {
      gpu = first_field($0); seen[gpu] = 1; if (!(gpu in used)) used[gpu] = "NA"; next
    }
    !in_proc && gpu != "" && match($0, /[0-9]+\/[0-9]+[[:space:]]*MiB/) {
      used[gpu] = substr($0, RSTART, RLENGTH); sub(/\/.*/, "", used[gpu]); gpu = ""; next
    }
    in_proc && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]/ { g = first_field($0); seen[g] = 1; procs[g]++ }
    END { for (g in seen) printf "%s %s %d\n", g, (used[g] == "" ? "NA" : used[g]), procs[g] + 0 }
  '
}
gpu_busy_problems() {  # stdin: plain `mx-smi` output; $1 = MiB limit, $2.. = GPU ids; one line per GPU not free
  local limit="$1" usage gpu row _id used procs
  shift
  usage=$(mx_smi_usage)
  for gpu in "$@"; do
    row=$(awk -v g="$gpu" '$1 == g' <<< "$usage")
    if [[ -z "$row" ]]; then
      echo "GPU $gpu is not listed by mx-smi"
      continue
    fi
    read -r _id used procs <<< "$row"
    if [[ ! "$used" =~ ^[0-9]+$ ]]; then
      echo "GPU $gpu: no memory use found in the mx-smi output"
    elif (( procs > 0 || used > limit )); then
      echo "GPU $gpu is busy: $procs process(es), $used MiB used (free = no process and <= $limit MiB)"
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
  problem "EXP19_GPUS must list 1..$MAX_RANKS GPU ids (dev-machine rule), got '$GPU_CSV'"
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
    problem "EXP19_SRC is the shared checkout; git archive the commit to $EXP_ROOT/src_<sha> (ledger §5 lesson 27)"
  fi
  [[ "$LAUNCHER_ROOT" == "$SRC" ]] || problem "run the launcher from the archived source: bash $SRC/scripts/exp19/run_rerun.sh (this one is under $LAUNCHER_ROOT)"
  need_file "$SRC/.exp19_git_sha"
  [[ -s "$SRC/.exp19_git_sha" ]] && GIT_SHA=$(tr -d '[:space:]' < "$SRC/.exp19_git_sha")
fi

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$DEPLOYED_SERVER" "$VO_SERVER" "$CLIENT" \
  "$STEP_TRACE" "$FINGERPRINT_TOOL" "$DATASET" "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
  need_file "$file"
done
for directory in "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_REPO" "$INTERNNAV_MODEL_PATH" "$SCENES_DIR" "$LOCKED_PLAN/tools" \
  "$PPA_TRAIN_DATA" "$PPA_TRAIN_CACHE" "$X11_BUNDLE" "$X11_DRI_PATH" "$X11_FONT_PATH" "$X11_XKB_PATH" "$LISTS_DIR"; do
  need_dir "$directory"
done
for executable in "$QWEN_PYTHON" "$VLNCE_PYTHON" "$XVFB_BIN" "$XDPYINFO_BIN" "$XKBCOMP_BIN"; do
  need_exec "$executable"
done
[[ -s "$INTERNNAV_MODEL_PATH/config.json" ]] || problem "InternNav model has no config.json: $INTERNNAV_MODEL_PATH"
# The two EXP-19 hooks must be in this source copy, or the run records nothing.
if [[ -s "$CLIENT" ]] && ! grep -qF -- '--step_state_trace_dir' "$CLIENT"; then
  problem "client lacks --step_state_trace_dir (source copy predates the EXP-19 step trace): $CLIENT"
fi
if [[ -s "$MODEL_SERVER" ]] && ! grep -qF 'EXP19_TRACE_DIR' "$MODEL_SERVER"; then
  problem "trace server does not read EXP19_TRACE_DIR: $MODEL_SERVER"
fi
if [[ -e "$RUN_DIR" ]]; then
  problem "run dir already exists (choose a new EXP19_RUN; runs are never resumed): $RUN_DIR"
fi

if [[ -x "$XDPYINFO_BIN" ]]; then
  for j in "${!GPUS[@]}"; do
    while IFS= read -r conflict; do
      [[ -z "$conflict" ]] || problem "$conflict"
    done <<< "$(rank_conflicts "$j")"
  done
fi

# Another job on a rank's GPU slows it or runs it out of memory, and a run is never resumed.
if command -v mx-smi >/dev/null 2>&1; then
  busy=()
  smi_rc=0
  smi_out=$(timeout 60 mx-smi 2>&1) || smi_rc=$?
  if [[ "$smi_rc" -ne 0 ]]; then
    busy+=("mx-smi failed (exit $smi_rc); cannot check that GPU(s) $GPU_CSV are free")
  else
    while IFS= read -r line; do
      [[ -z "$line" ]] || busy+=("$line")
    done <<< "$(gpu_busy_problems "$GPU_BUSY_MIB" "${GPUS[@]}" <<< "$smi_out")"
  fi
  for line in "${busy[@]+"${busy[@]}"}"; do
    if [[ "$ALLOW_BUSY_GPU" == 1 ]]; then
      log "WARNING (EXP19_ALLOW_BUSY_GPU=1): $line"
    else
      problem "$line (EXP19_ALLOW_BUSY_GPU=1 launches anyway)"
    fi
  done
  ((${#busy[@]})) || log "GPU(s) $GPU_CSV free (mx-smi: no process, <= $GPU_BUSY_MIB MiB used)"
else
  log "mx-smi not found: GPU-free check skipped"
fi

if [[ -x "$QWEN_PYTHON" && -d "$LISTS_DIR" && -s "$DATASET" ]]; then
  # Episode lists: exactly gpu0..gpu<n-1>.json, client format, every key in the
  # full val_unseen dataset with its scene on disk, no key twice across lists.
  list_rc=0
  list_report=$("$QWEN_PYTHON" - "$LISTS_DIR" "${#GPUS[@]}" "$DATASET" "$SCENES_DIR" <<'PY' 2>&1
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

if [[ -x "$VLNCE_PYTHON" && -d "$RPC_ROOT/src/vla_rpc" ]]; then
  PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" -c 'from vla_rpc.client import VLAClient' >/dev/null 2>&1 \
    || problem "vla_rpc.client does not import with the client python and RPC_PYTHONPATH"
fi
if [[ -x "$QWEN_PYTHON" && -s "$FINGERPRINT_TOOL" ]]; then
  SOURCE_FINGERPRINT=$("$QWEN_PYTHON" "$FINGERPRINT_TOOL" "$SRC") || { SOURCE_FINGERPRINT=unavailable; problem "cannot fingerprint $SRC"; }
fi
export HEATMAPVLN_SOURCE_FINGERPRINT="$SOURCE_FINGERPRINT"

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY RUN: nothing is started or written; the launcher would run:"
  emit_commands
fi
if ((${#PROBLEMS[@]})); then
  for p in "${PROBLEMS[@]}"; do printf '[exp19-rerun] PROBLEM: %s\n' "$p" >&2 || true; done
  die "${#PROBLEMS[@]} preflight problem(s); nothing was started"
fi
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY RUN OK: ${#GPUS[@]} rank(s) on GPU(s) $GPU_CSV, git_sha=$GIT_SHA, fingerprint=$SOURCE_FINGERPRINT"
  exit 0
fi

# ------------------------------------------------------------------ launch
cd "$SRC"
mkdir -p "$EXP_ROOT/runs"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR (does it exist already?)"
OWN_RUN_DIR=1
STARTED_UTC=$(date -u +%FT%TZ)
printf '%s\n' "$SOURCE_FINGERPRINT" > "$RUN_DIR/source_fingerprint.txt"
for j in "${!GPUS[@]}"; do
  dir=$(rank_dir "$j")
  mkdir -p "$dir/logs" "$dir/trace" "$dir/steps" "$dir/client_out" "$dir/runtime/xvfb/.xkb-cache" \
    "$dir/runtime/model"/{tmp,xdg,hf,torch_extensions,triton,matplotlib} "$dir/runtime/vo"/{tmp,xdg,hf,triton}
  cp "$LISTS_DIR/gpu$j.json" "$dir/episode_list.json"
done
emit_commands > "$RUN_DIR/commands.txt"
log "run=$RUN dir=$RUN_DIR src=$SRC git_sha=$GIT_SHA fingerprint=$SOURCE_FINGERPRINT gpus=$GPU_CSV diagnostics=$TRACE_DIAGNOSTICS allow_busy_gpu=$ALLOW_BUSY_GPU"

log "starting ${#GPUS[@]} Xvfb display(s)"
# Checked again right before the start, as the eval launcher does: a display or port
# taken since the preflight would let the readiness probes pass on someone else's server.
for j in "${!GPUS[@]}"; do
  conflicts=$(rank_conflicts "$j")
  [[ -z "$conflicts" ]] || die "rank $j: ${conflicts//$'\n'/; } (taken since the preflight)"
done
for j in "${!GPUS[@]}"; do
  dir=$(rank_dir "$j")
  runtime="$dir/runtime/xvfb"
  build_xvfb_cmd "$j"
  (
    cd "$runtime"
    exec 9<"$runtime/.xkb-cache"
    exec "${CMD[@]}"
  ) >"$dir/logs/xvfb.log" 2>&1 &
  XVFB_PIDS[j]="$!"
  # Readiness = the display's TCP port accepts a connection (bash /dev/tcp, no binary
  # load). The eval launcher probed with xdpyinfo under `timeout 5`, but on a cold AFS
  # the FUSE read of the xdpyinfo binary alone can take longer than that, so every
  # probe was killed while Xvfb was already listening (smoke1, 2026-09-24).
  ready=0
  for _ in $(seq 1 300); do
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
    if PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "$model_addr" "$vo_addr" <<< "$HEALTH_PY" >/dev/null 2>&1; then
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
  PID_RANK[$!]="$j"
done

# Unlike the eval, a failed client does not abort the others: every rank runs to
# its end, and its servers are stopped as soon as its client exits.
remaining=("${CLIENT_PIDS[@]}")
while ((${#remaining[@]})); do
  finished=""
  rc=0
  wait -n -p finished "${remaining[@]}" || rc=$?
  j="${PID_RANK[$finished]}"
  CLIENT_RC[j]="$rc"
  if [[ "$rc" -ne 0 ]]; then
    tail -80 "$(rank_dir "$j")/logs/client.log" >&2 || true
  fi
  stop_pid "${MODEL_PIDS[j]}"
  stop_pid "${VO_PIDS[j]}"
  stop_pid "${XVFB_PIDS[j]}"
  next=()
  for pid in "${remaining[@]}"; do
    [[ "$pid" == "$finished" ]] || next+=("$pid")
  done
  remaining=("${next[@]+"${next[@]}"}")
  log "rank=$j client exited rc=$rc; remaining=${#remaining[@]}"
done

fingerprint_end=$("$QWEN_PYTHON" "$FINGERPRINT_TOOL" "$SRC") || fingerprint_end=unavailable
[[ "$fingerprint_end" == "$SOURCE_FINGERPRINT" ]] \
  || log "WARNING: the source tree changed during the run ($SOURCE_FINGERPRINT -> $fingerprint_end)"

rank_specs=()
for j in "${!GPUS[@]}"; do rank_specs+=("$j:${GPUS[j]}:${CLIENT_RC[j]}"); done
set +e
"$QWEN_PYTHON" - "$RUN_DIR" "$RUN" "$GIT_SHA" "$SRC" "$LISTS_DIR" "$SOURCE_FINGERPRINT" "$fingerprint_end" \
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
    print(f"[exp19-rerun] rank={j} gpu={gpu} rc={rc} complete={ranks[-1]['n_complete']}/{len(episodes)} "
          f"trace_error_lines={rank_errors}")
code = {rel: sha256(src / rel) for rel in (
    "scripts/exp19/run_rerun.sh", "scripts/exp19/rpc_model_server_trace.py", "scripts/exp19/step_trace.py",
    "scripts/evaluation/rpc_model_server.py", "scripts/evaluation/r2r_val_unseen.py",
    "configs/ppa_action_refine_v2_8gpu.yaml")}
done = {
    "schema": "exp19-run-done-v1", "run": run, "git_sha": git_sha, "src": str(src),
    "source_fingerprint": {"start": fp_start, "end": fp_end, "unchanged": fp_start == fp_end},
    "code_sha256": code, "protocol_seed": int(seed), "trace_diagnostics": diagnostics == "1",
    "started_utc": started, "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "complete" if complete else "incomplete", "trace_error_lines": error_lines_total,
    "smoke_list": os.environ.get("EXP19_SMOKE_LIST"),  # set by run_smoke.sh; its gpu0.json copy is temporary
    "ranks": ranks,
}
(run_dir / "DONE").write_text(json.dumps(done, indent=2) + "\n")
print(f"[exp19-rerun] status={done['status']} trace_error_lines={error_lines_total} -> {run_dir / 'DONE'}")
sys.exit(0 if complete else 1)
PY
status=${PIPESTATUS[0]}
set -e
[[ "$status" -eq 0 ]] || die "run '$RUN' is incomplete (status $status); see $RUN_DIR/DONE and the per-rank logs"
log "COMPLETE: $RUN_DIR/DONE"
