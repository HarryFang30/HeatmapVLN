#!/usr/bin/env bash
# EXP-17: 8-GPU R2R val-unseen evaluation of a System2 cognition arm.
# Derived from run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh: the same Xvfb,
# causal AMB3R-VO RPC, sharding and merge; only the model server mode, the
# readiness evidence, the optional episode subset (canary) and the post-merge
# check differ.  The fine-tuned System2 decides (pose tokens, optional
# cognition prefix); Z0 for System1 comes from the native weights with the
# adapters disabled, so the Plan bridge is never applied.

set -Eeuo pipefail

FJL_ROOT="${COG_EVAL_FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}"
REPO="${COG_EVAL_REPO:-$FJL_ROOT/HeatmapVLN}"
RPC_ROOT="${COG_EVAL_RPC_ROOT:-$FJL_ROOT/rpc}"
INTERNNAV_REPO="${COG_EVAL_INTERNNAV_REPO:-$FJL_ROOT/InternNav}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$FJL_ROOT/InternNav-Model}"
AMB3R_ROOT="${COG_EVAL_AMB3R_ROOT:-$FJL_ROOT/amb3r}"
DA3_CHECKPOINT="${COG_EVAL_DA3_CHECKPOINT:-$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE}"
QWEN_PYTHON="${COG_EVAL_QWEN_PYTHON:-$FJL_ROOT/envs/qwen25/bin/python}"
VLNCE_PYTHON="${COG_EVAL_VLNCE_PYTHON:-$FJL_ROOT/envs/vlnce/bin/python}"

PPA_CHECKPOINT="${COG_EVAL_CHECKPOINT:?set COG_EVAL_CHECKPOINT to the trained best.pth of the arm}"
# The *deployment* config: the training arm config keeps action_head.enable=false
# and would start a server with no System1 to execute the pixel goals.
PPA_CONFIG="${COG_EVAL_CONFIG:-$REPO/configs/exp17b_system2_cognition_eval_8gpu.yaml}"
# Only used to expand $PPA_DATA_ROOT/$PPA_AMB3R_CACHE_ROOT placeholders when
# the train config is loaded; the eval itself never reads training data.  The
# original v1 dataset and cache were deleted — default to their v2 successors.
PPA_TRAIN_DATA="${COG_EVAL_TRAIN_DATA:-$FJL_ROOT/r2r_panoramic_data_v2/train}"
PPA_TRAIN_CACHE="${COG_EVAL_TRAIN_CACHE:-$FJL_ROOT/data/amb3r_endpoint_v3_full_r2r}"
PPA_STAGE2_OUTPUT_ROOT="$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/stage2_joint"
PPA_TENSORBOARD_ROOT="$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/tensorboard"
# The arm configs carry the EXP-13 training placeholders; resolve them so the
# server can load the config.  None of these paths is read during evaluation.
DAGGER_ROOT="${COG_EVAL_DAGGER_ROOT:-$FJL_ROOT/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17}"
export EXP13_OUTPUT_ROOT="${COG_EVAL_TRAIN_OUTPUT_ROOT:-$FJL_ROOT/model/exp17_cognition_prefix/_config_placeholder}"
export EXP13_TENSORBOARD_ROOT="$EXP13_OUTPUT_ROOT/tensorboard"
export EXP13_ORACLE_VIEWS="${EXP13_ORACLE_VIEWS:-$FJL_ROOT/model/exp12_recovery_gate/d1_per_state.jsonl}"
export R2R_TRAIN_JSON="${R2R_TRAIN_JSON:-$FJL_ROOT/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
export DAGGER_ROOT_00="$DAGGER_ROOT/shard_00" DAGGER_ROOT_01="$DAGGER_ROOT/shard_01"
export DAGGER_ROOT_02="$DAGGER_ROOT/shard_02" DAGGER_ROOT_03="$DAGGER_ROOT/shard_03"
export DAGGER_POLICY_FINGERPRINT="${DAGGER_POLICY_FINGERPRINT:-$("$QWEN_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['contract']['policy_fingerprint'])" "$DAGGER_ROOT/shard_00/collection_manifest.json")}"

LOCKED_PLAN="$FJL_ROOT/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
COHORTS_DIR="$LOCKED_PLAN/cohorts"
MERGE_TOOL="$LOCKED_PLAN/tools/merge_shards.py"
DATASET="$FJL_ROOT/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
SCENES_DIR="$FJL_ROOT/habitat/VLN-CE/data/scene_datasets"
# Canary: COG_EVAL_MAX_EPISODES_PER_SHARD=75 reads the first 75 episodes of each
# of the 8 locked shards (600 total, iterator order, deterministic); 0 = all 1839.
MAX_EPISODES_PER_SHARD="${COG_EVAL_MAX_EPISODES_PER_SHARD:-0}"
if [[ "$MAX_EPISODES_PER_SHARD" -gt 0 ]]; then
  EXPECTED_EPISODES=$((MAX_EPISODES_PER_SHARD * 8))
  MAX_EPISODES_ARGS=(--max_episodes "$MAX_EPISODES_PER_SHARD")
else
  EXPECTED_EPISODES=1839
  MAX_EPISODES_ARGS=()
fi
# Optional: also decode the native first turn on every ready call (2x System2 cost).
AUDIT_ARGS=()
[[ "${COG_EVAL_AUDIT_NATIVE:-0}" == "1" ]] && AUDIT_ARGS=(--cognition_audit_native)

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
CLIENT="$REPO/scripts/evaluation/r2r_val_unseen.py"
OUTPUT_ROOT="${COG_EVAL_OUTPUT_ROOT:-$FJL_ROOT/model/eval_system2_cognition_r2r_val_unseen_8gpu}"
WORKERS_DIR="$OUTPUT_ROOT/workers"
MERGED_DIR="$OUTPUT_ROOT/merged"
RUN_STAMP="${COG_EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_job${JOB_ID:-$$}}"
RUNTIME_DIR="$OUTPUT_ROOT/runtime/$RUN_STAMP"

GPU_CSV="${COG_EVAL_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_SHARDS=8
MODEL_PORT_BASE="${COG_EVAL_MODEL_PORT_BASE:-53400}"
VO_PORT_BASE="${COG_EVAL_VO_PORT_BASE:-53500}"
DISPLAY_BASE="${COG_EVAL_DISPLAY_BASE:-380}"
SERVER_START_TIMEOUT_S="${COG_EVAL_SERVER_START_TIMEOUT_S:-3600}"
SERVER_STAGGER_S="${COG_EVAL_SERVER_STAGGER_S:-15}"
RPC_TIMEOUT_MS="${COG_EVAL_RPC_TIMEOUT_MS:-600000}"
# Deterministic-sampling protocol seed and the merged-result arm label.  Seed
# 42 is the certified main-table run; a second seed must land in its own
# COG_EVAL_OUTPUT_ROOT so --resume never mixes seeds.
PROTOCOL_SEED="${COG_EVAL_PROTOCOL_SEED:-42}"
EVAL_ARM="${COG_EVAL_ARM:-system2_cognition_exp17b}"

X11_BUNDLE="$FJL_ROOT/tools/x11_headless_bundle_ubuntu22_20260801_v4"
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

RPC_PYTHONPATH="$LOCKED_PLAN/tools:$RPC_ROOT/src:$REPO:$INTERNNAV_REPO${PYTHONPATH:+:$PYTHONPATH}"
SERVER_LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib"
CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:$LD_LIBRARY_PATH"
TOOL_LD_LIBRARY_PATH="$X11_BUNDLE/lib:$LD_LIBRARY_PATH"

declare -a GPUS MODEL_PIDS VO_PIDS CLIENT_PIDS XVFB_PIDS

die() { printf '[cog-eval] ERROR: %s\n' "$*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

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
  for pid in "${CLIENT_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${XVFB_PIDS[@]:-}"; do stop_pid "$pid"; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$VO_SERVER" "$CLIENT" "$MERGE_TOOL" "$DATASET" "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml"; do
  require_file "$file"
done
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_MODEL_PATH" "$SCENES_DIR" "$COHORTS_DIR" "$X11_BUNDLE" "$X11_DRI_PATH" "$X11_FONT_PATH" "$X11_XKB_PATH"; do
  require_dir "$directory"
done
for executable in "$QWEN_PYTHON" "$VLNCE_PYTHON" "$XVFB_BIN" "$XDPYINFO_BIN" "$XKBCOMP_BIN"; do
  [[ -x "$executable" ]] || die "missing executable: $executable"
done

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ "${#GPUS[@]}" -eq 8 ]] || die "COG_EVAL_GPU_DEVICES must contain 8 IDs"
[[ "$PROTOCOL_SEED" =~ ^[0-9]+$ ]] || die "COG_EVAL_PROTOCOL_SEED must be a non-negative integer"
[[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || die "GPU IDs must be unique"
for rank in $(seq 0 7); do
  [[ "${GPUS[$rank]}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${GPUS[$rank]}"
  require_file "$COHORTS_DIR/shard_$(printf '%02d' "$rank").json"
  require_file "$COHORTS_DIR/dataset_shard_$(printf '%02d' "$rank").json.gz"
done

mkdir -p "$WORKERS_DIR" "$MERGED_DIR" "$RUNTIME_DIR/logs" "$RUNTIME_DIR/ranks"
export PYTHONDONTWRITEBYTECODE=1

echo "[cog-eval] starting 8 isolated Xvfb displays"
for rank in $(seq 0 7); do
  display_num=$((DISPLAY_BASE + rank))
  display="127.0.0.1:${display_num}.0"
  log="$RUNTIME_DIR/logs/xvfb_${rank}.log"
  runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/xvfb"
  mkdir -p "$runtime/.xkb-cache"
  if env LD_LIBRARY_PATH="$TOOL_LD_LIBRARY_PATH" DISPLAY="$display" \
    timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    die "DISPLAY=$display is already active; choose another COG_EVAL_DISPLAY_BASE"
  fi
  (
    cd "$runtime"
    exec 9<"$runtime/.xkb-cache"
    exec env \
      PATH="$X11_BUNDLE/bin:$PATH" \
      LD_LIBRARY_PATH="$TOOL_LD_LIBRARY_PATH" \
      LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
      LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
      MESA_LOADER_DRIVER_OVERRIDE=swrast \
      "$XVFB_BIN" ":$display_num" -screen 0 1024x768x24 \
      -nolock -nolisten unix -listen tcp +iglx -ac \
      -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
  ) >"$log" 2>&1 &
  XVFB_PIDS[$rank]="$!"
  ready=0
  for _ in $(seq 1 60); do
    if env LD_LIBRARY_PATH="$TOOL_LD_LIBRARY_PATH" DISPLAY="$display" \
      timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
      ready=1
      break
    fi
    kill -0 "${XVFB_PIDS[$rank]}" 2>/dev/null || break
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || die "Xvfb rank $rank failed; see $log"
done

echo "[cog-eval] starting model RPC servers"
for rank in $(seq 0 7); do
  gpu="${GPUS[$rank]}"
  port=$((MODEL_PORT_BASE + rank))
  runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/model"
  log="$RUNTIME_DIR/logs/model_${rank}.log"
  mkdir -p "$runtime"/{tmp,xdg,hf,torch_extensions,triton,matplotlib}
  env \
    PYTHONPATH="$RPC_PYTHONPATH" LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" TMPDIR="$runtime/tmp" \
    XDG_CACHE_HOME="$runtime/xdg" HF_HOME="$runtime/hf" \
    TORCH_EXTENSIONS_DIR="$runtime/torch_extensions" \
    TRITON_CACHE_DIR="$runtime/triton" MPLCONFIGDIR="$runtime/matplotlib" \
    HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0 \
    "$QWEN_PYTHON" -u "$MODEL_SERVER" \
      --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT" \
      --internnav_model_path "$INTERNNAV_MODEL_PATH" \
      --gpu_id 0 --host 127.0.0.1 --port "$port" --workers 1 \
      --require_deterministic_sampling --system2_cognition_arm "${AUDIT_ARGS[@]}" \
      --log_level INFO >"$log" 2>&1 &
  MODEL_PIDS[$rank]="$!"
  sleep "$SERVER_STAGGER_S"
done

echo "[cog-eval] starting online AMB3R RPC servers"
for rank in $(seq 0 7); do
  gpu="${GPUS[$rank]}"
  port=$((VO_PORT_BASE + rank))
  runtime="$RUNTIME_DIR/ranks/rank_$(printf '%02d' "$rank")/vo"
  log="$RUNTIME_DIR/logs/vo_${rank}.log"
  mkdir -p "$runtime"/{tmp,xdg,hf,triton}
  env \
    PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH" \
    LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES="$gpu" \
    TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg" \
    HF_HOME="$runtime/hf" TRITON_CACHE_DIR="$runtime/triton" \
    "$QWEN_PYTHON" -u "$VO_SERVER" \
      --repo "$REPO" --amb3r-root "$AMB3R_ROOT" \
      --da3-checkpoint "$DA3_CHECKPOINT" --device cuda:0 \
      --host 127.0.0.1 --port "$port" \
      --map-init-window 20 --map-every 8 --max-history 8 \
      --resolution 518 392 --translation-scale 1.0 \
      --max-frames-limit 4096 --max-message-mb 32 \
      --log-level INFO >"$log" 2>&1 &
  VO_PIDS[$rank]="$!"
  sleep "$SERVER_STAGGER_S"
done

echo "[cog-eval] waiting for 16 RPC servers"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
for rank in $(seq 0 7); do
  model_addr="127.0.0.1:$((MODEL_PORT_BASE + rank))"
  vo_addr="127.0.0.1:$((VO_PORT_BASE + rank))"
  while true; do
    kill -0 "${MODEL_PIDS[$rank]}" 2>/dev/null || {
      tail -120 "$RUNTIME_DIR/logs/model_${rank}.log" >&2
      die "model server rank $rank exited"
    }
    kill -0 "${VO_PIDS[$rank]}" 2>/dev/null || {
      tail -120 "$RUNTIME_DIR/logs/vo_${rank}.log" >&2
      die "VO server rank $rank exited"
    }
    if PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "$model_addr" "$vo_addr" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient

for address, expected in (
    (sys.argv[1], "system2-cognition-arm-v1"),
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
    then
      break
    fi
    (( $(date +%s) < deadline )) || die "RPC startup timeout at rank $rank"
    sleep 10
  done
  grep -F "System2 cognition arm runtime enabled" \
    "$RUNTIME_DIR/logs/model_${rank}.log" >/dev/null \
    || die "model rank $rank lacks cognition-arm preflight evidence"
  echo "[cog-eval] rank=$rank model=$model_addr vo=$vo_addr ready"
done

# The evaluation reads this shared checkout live for many hours; an edit landing
# mid-run would change the servers restarted afterwards.  Record the tree now and
# check it again before the merge (ledger §5, lesson 27).
SOURCE_FINGERPRINT="$("$QWEN_PYTHON" "$REPO/scripts/tools/source_fingerprint.py" "$REPO")" \
  || die "cannot fingerprint the source tree"
printf '%s\n' "$SOURCE_FINGERPRINT" > "$RUNTIME_DIR/source_fingerprint.txt"
echo "[cog-eval] source fingerprint: $SOURCE_FINGERPRINT"
if git_status="$(git -c safe.directory="$REPO" -C "$REPO" status --short --untracked-files=no 2>/dev/null)"; then
  [[ -z "$git_status" ]] || die "refusing to evaluate an unversioned tree: $REPO has uncommitted changes"
fi

echo "[cog-eval] starting $EXPECTED_EPISODES val-unseen episodes across 8 shards (protocol seed $PROTOCOL_SEED, arm $EVAL_ARM)"
for rank in $(seq 0 7); do
  gpu="${GPUS[$rank]}"
  shard="$(printf '%02d' "$rank")"
  display="127.0.0.1:$((DISPLAY_BASE + rank)).0"
  output="$WORKERS_DIR/shard_${shard}"
  log="$RUNTIME_DIR/logs/client_${rank}.log"
  mkdir -p "$output"
  env \
    PYTHONPATH="$RPC_PYTHONPATH" LD_LIBRARY_PATH="$CLIENT_LD_LIBRARY_PATH" \
    DISPLAY="$display" CUDA_VISIBLE_DEVICES="$gpu" HABITAT_GL_GPU_ID=0 \
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast \
    HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1 \
    "$VLNCE_PYTHON" -u "$CLIENT" \
      --config "$PPA_CONFIG" \
      --rpc_server "127.0.0.1:$((MODEL_PORT_BASE + rank))" \
      --history_pose_source amb3r_vo_da3 \
      --amb3r_vo_rpc_server "127.0.0.1:$((VO_PORT_BASE + rank))" \
      --amb3r_vo_rpc_timeout_ms "$RPC_TIMEOUT_MS" \
      --amb3r_vo_rpc_jpeg_quality 95 \
      --rpc_timeout_ms "$RPC_TIMEOUT_MS" --rpc_jpeg_quality 90 \
      --rpc_protocol_seed "$PROTOCOL_SEED" --rpc_require_deterministic_sampling \
      --rpc_policy_mode heatmapvln \
      --scenes_dir "$SCENES_DIR" \
      --data_path "$COHORTS_DIR/dataset_shard_${shard}.json.gz" \
      --dataset_split val_unseen --episode_list "$COHORTS_DIR/shard_${shard}.json" \
      --output_path "$output" --sim_gpu_id 0 \
      --resize_w 384 --resize_h 384 --num_history 8 \
      --max_steps_per_episode 500 --max_system2_calls_per_episode 0 \
      --auto_stop_distance 0 --trajectory_selection mean \
      --trajectory_x_sign 1 --trajectory_heading_alignment none \
      --system1_coord_order generated --no-pano_recenter_before_system1 \
      --no-debug_input_trace --debug_save_input_images 0 --resume "${MAX_EPISODES_ARGS[@]}" \
      >"$log" 2>&1 &
  CLIENT_PIDS[$rank]="$!"
done

remaining=("${CLIENT_PIDS[@]}")
while ((${#remaining[@]})); do
  finished=""
  if ! wait -n -p finished "${remaining[@]}"; then
    for rank in $(seq 0 7); do
      tail -80 "$RUNTIME_DIR/logs/client_${rank}.log" >&2 || true
    done
    die "a Habitat evaluation shard failed"
  fi
  next=()
  for pid in "${remaining[@]}"; do
    [[ "$pid" == "$finished" ]] || next+=("$pid")
  done
  remaining=("${next[@]}")
  echo "[cog-eval] client pid=$finished complete; remaining=${#remaining[@]}"
done

fingerprint_now="$("$QWEN_PYTHON" "$REPO/scripts/tools/source_fingerprint.py" "$REPO")" || fingerprint_now="unavailable"
[[ "$fingerprint_now" == "$SOURCE_FINGERPRINT" ]] \
  || die "the source tree changed during the run ($SOURCE_FINGERPRINT -> $fingerprint_now); the shards are not one arm"

"$QWEN_PYTHON" "$MERGE_TOOL" \
  --dataset "$DATASET" --cohorts-dir "$COHORTS_DIR" \
  --workers-dir "$WORKERS_DIR" --output-dir "$MERGED_DIR" \
  --num-shards 8 --expected-episodes "$EXPECTED_EPISODES" \
  --protocol heatmapvln-r2r-json-v3 \
  --sampling-protocol heatmapvln-nextdit-sha256-v1 \
  --protocol-seed "$PROTOCOL_SEED" --evaluation-arm "$EVAL_ARM"

"$QWEN_PYTHON" - "$MERGED_DIR" "$EXPECTED_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

root, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for line in (root / "progress.json").read_text().splitlines() if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"merged episode count mismatch: {len(rows)} != {expected}")
if any(row.get("history_pose_source") != "amb3r_vo_da3" for row in rows):
    raise SystemExit("merged result contains a non-AMB3R pose provider")
applied = sum(int(row.get("cognition_applied_calls", 0)) for row in rows)
if applied <= 0:
    raise SystemExit("the cognition arm was never applied after AMB3R warmup")
result = json.loads((root / "result.json").read_text())
if int(result.get("total_episodes", -1)) != expected:
    raise SystemExit("merged result total_episodes mismatch")
print(json.dumps({"status": "passed", "episodes": expected, "cognition_applied_calls": applied, "result": result}, sort_keys=True))
PY

echo "[cog-eval] COMPLETE result=$MERGED_DIR/result.json"
echo "[cog-eval] merged progress=$MERGED_DIR/progress.json"
echo "[cog-eval] runtime logs=$RUNTIME_DIR/logs"
