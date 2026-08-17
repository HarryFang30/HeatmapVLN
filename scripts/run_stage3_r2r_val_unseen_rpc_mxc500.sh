#!/usr/bin/env bash
# Wait for the final Stage3 adapter and run R2R val_unseen through the
# qwen25-model <-> vlnce-Habitat RPC bridge on one MXC500 node.

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
export RPC_ROOT="${RPC_ROOT:-${FJL_ROOT}/rpc}"
QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
VLNCE_PYTHON="${VLNCE_PYTHON:-${FJL_ROOT}/envs/vlnce/bin/python}"

export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-${FJL_ROOT}/InternNav-Model}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export STAGE3_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_EVAL_BASE_CKPT="${STAGE3_EVAL_BASE_CKPT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_EVAL_TRAIN_OUT_DIR="${STAGE3_EVAL_TRAIN_OUT_DIR:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords}"
export STAGE3_EVAL_EXPECTED_EPOCH="${STAGE3_EVAL_EXPECTED_EPOCH:-2}"
printf -v STAGE3_EVAL_CHECKPOINT_NAME 'epoch_%03d.pth' "$STAGE3_EVAL_EXPECTED_EPOCH"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${STAGE3_EVAL_TRAIN_OUT_DIR}/latest/checkpoints/${STAGE3_EVAL_CHECKPOINT_NAME}}"
# Optional isolated binary STOP head. The original all-layer LoRA remains the
# only Qwen adapter for both waypoint generation and System1 latent extraction.
export STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT="${STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT:-}"
export STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT="${STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT:-}"
export STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT="${STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT:-}"
export STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD="${STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD:-}"
export STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY="${STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY:-0}"

export STAGE3_EVAL_SCENES_DIR="${STAGE3_EVAL_SCENES_DIR:-${FJL_ROOT}/habitat/VLN-CE/data/scene_datasets}"
export STAGE3_EVAL_DATA_PATH="${STAGE3_EVAL_DATA_PATH:-${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz}"
export STAGE3_EVAL_EXPECTED_EPISODES="${STAGE3_EVAL_EXPECTED_EPISODES:-1839}"
export STAGE3_EVAL_OUTPUT_PATH="${STAGE3_EVAL_OUTPUT_PATH:-${FJL_ROOT}/model/eval_stage3_r2r_val_unseen_full_11000_alllora_h1024_internnavcoords_epoch${STAGE3_EVAL_EXPECTED_EPOCH}_no_privileged_stop}"

export STAGE3_EVAL_MODEL_GPU="${STAGE3_EVAL_MODEL_GPU:-0}"
export STAGE3_EVAL_SIM_GPU="${STAGE3_EVAL_SIM_GPU:-$STAGE3_EVAL_MODEL_GPU}"
export STAGE3_EVAL_DISPLAY="${STAGE3_EVAL_DISPLAY:-localhost:200.0}"
export STAGE3_EVAL_RPC_HOST="${STAGE3_EVAL_RPC_HOST:-127.0.0.1}"
export STAGE3_EVAL_RPC_PORT="${STAGE3_EVAL_RPC_PORT:-50061}"
export STAGE3_EVAL_RPC_TIMEOUT_MS="${STAGE3_EVAL_RPC_TIMEOUT_MS:-600000}"
export STAGE3_EVAL_RPC_JPEG_QUALITY="${STAGE3_EVAL_RPC_JPEG_QUALITY:-90}"
export STAGE3_EVAL_RPC_PROTOCOL_SEED="${STAGE3_EVAL_RPC_PROTOCOL_SEED:-42}"
export STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING="${STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING:-0}"
export STAGE3_EVAL_SERVER_START_TIMEOUT_S="${STAGE3_EVAL_SERVER_START_TIMEOUT_S:-1800}"

export STAGE3_EVAL_MAX_EPISODES="${STAGE3_EVAL_MAX_EPISODES:-}"
export STAGE3_EVAL_EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-}"
export STAGE3_EVAL_MAX_STEPS="${STAGE3_EVAL_MAX_STEPS:-500}"
export STAGE3_EVAL_MAX_SYSTEM2_CALLS="${STAGE3_EVAL_MAX_SYSTEM2_CALLS:-0}"
export STAGE3_EVAL_ACTION_CHUNK_SIZE="${STAGE3_EVAL_ACTION_CHUNK_SIZE:-4}"
export STAGE3_EVAL_STOP_CONFIRMATIONS="${STAGE3_EVAL_STOP_CONFIRMATIONS:-1}"
export STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS="${STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS:-0}"
export STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP="${STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP:-0}"
export STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED="${STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED:-0}"
export STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN="${STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN:-0.005}"
export STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD="${STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD:-}"
export STAGE3_EVAL_STOP_PROBE_TURN="${STAGE3_EVAL_STOP_PROBE_TURN:-left}"
export STAGE3_EVAL_CLOSED_LOOP_GUARD="${STAGE3_EVAL_CLOSED_LOOP_GUARD:-0}"
export STAGE3_EVAL_COLLISION_EPSILON_M="${STAGE3_EVAL_COLLISION_EPSILON_M:-0.03}"
export STAGE3_EVAL_COLLISION_FORWARD_LIMIT="${STAGE3_EVAL_COLLISION_FORWARD_LIMIT:-3}"
export STAGE3_EVAL_MOTION_WINDOW_STEPS="${STAGE3_EVAL_MOTION_WINDOW_STEPS:-32}"
export STAGE3_EVAL_MOTION_MIN_PATH_M="${STAGE3_EVAL_MOTION_MIN_PATH_M:-2.0}"
export STAGE3_EVAL_MOTION_MAX_NET_M="${STAGE3_EVAL_MOTION_MAX_NET_M:-0.75}"
export STAGE3_EVAL_PLAN_WINDOW_CALLS="${STAGE3_EVAL_PLAN_WINDOW_CALLS:-20}"
export STAGE3_EVAL_PLAN_VIEW_DOMINANCE="${STAGE3_EVAL_PLAN_VIEW_DOMINANCE:-0.9}"
export STAGE3_EVAL_PLAN_MIN_PATH_M="${STAGE3_EVAL_PLAN_MIN_PATH_M:-3.0}"
export STAGE3_EVAL_PLAN_MAX_NET_M="${STAGE3_EVAL_PLAN_MAX_NET_M:-1.5}"
export STAGE3_EVAL_RECOVERY_TURNS="${STAGE3_EVAL_RECOVERY_TURNS:-3}"
export STAGE3_EVAL_RECOVERY_FORWARD_STEPS="${STAGE3_EVAL_RECOVERY_FORWARD_STEPS:-0}"
export STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN="${STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN:-0}"
export STAGE3_EVAL_RECOVERY_COOLDOWN_STEPS="${STAGE3_EVAL_RECOVERY_COOLDOWN_STEPS:-12}"
export STAGE3_EVAL_RECOVERY_HISTORY_KEEP="${STAGE3_EVAL_RECOVERY_HISTORY_KEEP:-2}"
export STAGE3_EVAL_NUM_HISTORY="${STAGE3_EVAL_NUM_HISTORY:-8}"
export STAGE3_EVAL_TRAJECTORY_SELECTION="${STAGE3_EVAL_TRAJECTORY_SELECTION:-mean}"
# Corrected Stage3 checkpoints already emit native InternNav coordinates.
# Set -1 explicitly only when diagnosing a legacy pre-coordinate-fix checkpoint.
export STAGE3_EVAL_TRAJECTORY_X_SIGN="${STAGE3_EVAL_TRAJECTORY_X_SIGN:-1}"
export STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT="${STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT:-none}"
export STAGE3_EVAL_SYSTEM1_COORD_ORDER="${STAGE3_EVAL_SYSTEM1_COORD_ORDER:-generated}"
export STAGE3_EVAL_AUTO_STOP_DISTANCE="${STAGE3_EVAL_AUTO_STOP_DISTANCE:-0.0}"
export STAGE3_EVAL_ORACLE_SYSTEM2="${STAGE3_EVAL_ORACLE_SYSTEM2:-0}"
export STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY="${STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY:-farthest_visible}"
export STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M="${STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M:-2.0}"
export STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M="${STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M:-0.5}"
export STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M="${STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M:-6.0}"
export STAGE3_EVAL_ALLOW_PRIVILEGED="${STAGE3_EVAL_ALLOW_PRIVILEGED:-0}"
export STAGE3_EVAL_RESUME="${STAGE3_EVAL_RESUME:-1}"
export STAGE3_EVAL_OVERWRITE="${STAGE3_EVAL_OVERWRITE:-0}"
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS="${STAGE3_EVAL_SAVE_TRAJECTORY_STEPS:-0}"
export STAGE3_EVAL_COLLECT_STOP_FEATURES="${STAGE3_EVAL_COLLECT_STOP_FEATURES:-0}"
export STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES="${STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES:-0}"
export STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS="${STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS:-}"
export STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES="${STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES:-0}"
export STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE="${STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE:-0}"
export STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START="${STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START:-0}"
export STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP="${STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP:-0}"
export STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M="${STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M:-3.01}"
export STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M="${STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M:-6.0}"
export STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS="${STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS:-8}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS="${STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS:-0}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES="${STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES:-8}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL="${STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL:-1}"
export STAGE3_EVAL_STOP_POSITIVE_RADIUS_M="${STAGE3_EVAL_STOP_POSITIVE_RADIUS_M:-3.0}"
export STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M="${STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M:-4.0}"
export STAGE3_EVAL_PREFLIGHT_ONLY="${STAGE3_EVAL_PREFLIGHT_ONLY:-0}"

export STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S="${STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S:-300}"
export STAGE3_EVAL_CHECKPOINT_SETTLE_S="${STAGE3_EVAL_CHECKPOINT_SETTLE_S:-30}"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export HEATMAPVLN_REQUIRE_FLASH_ATTN=1

RPC_PYTHONPATH="${RPC_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RPC_SERVER_ADDR="${STAGE3_EVAL_RPC_HOST}:${STAGE3_EVAL_RPC_PORT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_STAMP}_p${STAGE3_EVAL_RPC_PORT}_pid$$"
LOG_DIR="${STAGE3_EVAL_LOG_DIR:-${REPO_ROOT}/logs}"
SERVER_LOG="${STAGE3_EVAL_SERVER_LOG:-${LOG_DIR}/stage3_r2r_rpc_server_${RUN_TAG}.log}"
CLIENT_LOG="${STAGE3_EVAL_CLIENT_LOG:-${LOG_DIR}/stage3_r2r_val_unseen_${RUN_TAG}.log}"
SERVER_PID=""

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required non-empty file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] Stopping RPC model server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -x "$QWEN25_PYTHON" ]]; then
  echo "Missing qwen25 Python: $QWEN25_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$VLNCE_PYTHON" ]]; then
  echo "Missing vlnce Python: $VLNCE_PYTHON" >&2
  exit 1
fi

validate_eval_checkpoint() {
  local system2_args=()
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" ]]; then
    system2_args+=(--system2-stop-head-checkpoint "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT")
  fi
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" ]]; then
    system2_args+=(
      --system2-stop-decision-adapter-checkpoint
      "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT"
    )
  fi
  if [[ -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
    system2_args+=(
      --system2-temporal-stop-verifier-checkpoint
      "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"
    )
  fi
  PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" \
    scripts/evaluation/preflight_stage3_rpc_eval.py \
    --config "$STAGE3_EVAL_CONFIG" \
    --base-checkpoint "$STAGE3_EVAL_BASE_CKPT" \
    --stage3-checkpoint "$STAGE3_EVAL_CHECKPOINT" \
    --expected-epoch "$STAGE3_EVAL_EXPECTED_EPOCH" \
    --expected-adapter-hidden-dim 1024 \
    "${system2_args[@]}"
}

if is_true "$STAGE3_EVAL_RESUME" && is_true "$STAGE3_EVAL_OVERWRITE"; then
  echo "STAGE3_EVAL_RESUME and STAGE3_EVAL_OVERWRITE cannot both be enabled" >&2
  exit 1
fi
for gpu_var in STAGE3_EVAL_MODEL_GPU STAGE3_EVAL_SIM_GPU; do
  gpu_id="${!gpu_var}"
  if [[ ! "$gpu_id" =~ ^[0-7]$ ]]; then
    echo "$gpu_var must be a physical GPU id in [0, 7], got: $gpu_id" >&2
    exit 1
  fi
done
case "$STAGE3_EVAL_TRAJECTORY_X_SIGN" in
  -1|-1.0|1|1.0) ;;
  *)
    echo "STAGE3_EVAL_TRAJECTORY_X_SIGN must be -1 or 1" >&2
    exit 1
    ;;
esac
case "$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT" in
  none|pano_pixel) ;;
  *)
    echo "STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT must be none or pano_pixel" >&2
    exit 1
    ;;
esac
case "$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY" in
  farthest_visible|lookahead) ;;
  *)
    echo "STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY must be farthest_visible or lookahead" >&2
    exit 1
    ;;
esac
"$QWEN25_PYTHON" - "$STAGE3_EVAL_RPC_PROTOCOL_SEED" <<'PY'
import sys

seed = int(sys.argv[1])
if not 0 <= seed <= (1 << 63) - 1:
    raise SystemExit("STAGE3_EVAL_RPC_PROTOCOL_SEED must be in [0, 2**63 - 1]")
PY
"$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M" \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M" \
  "$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M" <<'PY'
import math
import sys

names = ("lookahead_m", "min_ahead_m", "max_side_dist_m")
for name, raw in zip(names, sys.argv[1:]):
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"Oracle System2 {name} must be finite and > 0, got {raw!r}")
PY
"$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_ACTION_CHUNK_SIZE" \
  "$STAGE3_EVAL_STOP_CONFIRMATIONS" \
  "$STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS" \
  "$STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP" \
  "$STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED" \
  "$STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN" \
  "$STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD" \
  "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" \
  "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" \
  "$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD" \
  "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY" \
  "$STAGE3_EVAL_COLLISION_EPSILON_M" \
  "$STAGE3_EVAL_COLLISION_FORWARD_LIMIT" \
  "$STAGE3_EVAL_MOTION_WINDOW_STEPS" \
  "$STAGE3_EVAL_MOTION_MIN_PATH_M" \
  "$STAGE3_EVAL_MOTION_MAX_NET_M" \
  "$STAGE3_EVAL_PLAN_WINDOW_CALLS" \
  "$STAGE3_EVAL_PLAN_VIEW_DOMINANCE" \
  "$STAGE3_EVAL_PLAN_MIN_PATH_M" \
  "$STAGE3_EVAL_PLAN_MAX_NET_M" \
  "$STAGE3_EVAL_RECOVERY_TURNS" \
  "$STAGE3_EVAL_RECOVERY_FORWARD_STEPS" \
  "$STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN" \
  "$STAGE3_EVAL_RECOVERY_COOLDOWN_STEPS" \
  "$STAGE3_EVAL_RECOVERY_HISTORY_KEEP" <<'PY'
import math
import sys

(
    action_chunk,
    stop_confirmations,
    stop_confirmation_max_gap_calls,
    stop_confirmation_view_sweep,
    stop_accept_temporal_confirmed,
    stop_temporal_trust_min_margin,
    stop_high_confidence_threshold,
    stop_head_checkpoint,
    stop_decision_adapter_checkpoint,
    stop_add_threshold,
    stop_add_min_qwen_stop_probability,
    collision_epsilon,
    collision_limit,
    motion_window,
    motion_min_path,
    motion_max_net,
    plan_window,
    plan_dominance,
    plan_min_path,
    plan_max_net,
    recovery_turns,
    recovery_forward_steps,
    recovery_follow_last_turn,
    recovery_cooldown,
    recovery_history_keep,
) = sys.argv[1:]
action_chunk = int(action_chunk)
if not 1 <= action_chunk <= 4:
    raise SystemExit(f"STAGE3_EVAL_ACTION_CHUNK_SIZE must be in [1, 4], got {action_chunk}")
stop_confirmations = int(stop_confirmations)
if stop_confirmations < 1:
    raise SystemExit("STAGE3_EVAL_STOP_CONFIRMATIONS must be >= 1")
if int(stop_confirmation_max_gap_calls) < 0:
    raise SystemExit(
        "STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS must be >= 0"
    )
if stop_confirmation_view_sweep.lower() not in {
    "0", "1", "false", "true", "no", "yes", "off", "on"
}:
    raise SystemExit("STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP must be boolean")
if stop_accept_temporal_confirmed.lower() not in {
    "0", "1", "false", "true", "no", "yes", "off", "on"
}:
    raise SystemExit("STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED must be boolean")
stop_temporal_trust_min_margin = float(stop_temporal_trust_min_margin)
if not math.isfinite(stop_temporal_trust_min_margin) or not 0.0 <= stop_temporal_trust_min_margin <= 1.0:
    raise SystemExit(
        "STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN must be finite and in [0, 1]"
    )
if (
    stop_confirmation_view_sweep.lower() in {"1", "true", "yes", "on"}
    and int(stop_confirmation_max_gap_calls) < 1
):
    raise SystemExit(
        "STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP requires "
        "STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS >= 1"
    )
if stop_high_confidence_threshold:
    threshold = float(stop_high_confidence_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit(
            "STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD must be in [0, 1]"
        )
    if stop_confirmations < 2:
        raise SystemExit(
            "STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD requires "
            "STAGE3_EVAL_STOP_CONFIRMATIONS >= 2"
        )
add_min_qwen = float(stop_add_min_qwen_stop_probability)
if not math.isfinite(add_min_qwen) or not 0.0 <= add_min_qwen <= 1.0:
    raise SystemExit(
        "STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY must be in [0, 1]"
    )
if add_min_qwen > 0.0 and not stop_head_checkpoint:
    raise SystemExit(
        "STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY requires a STOP head"
    )
if stop_add_threshold:
    threshold = float(stop_add_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit(
            "STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD must be in [0, 1]"
        )
    if not (stop_head_checkpoint or stop_decision_adapter_checkpoint):
        raise SystemExit(
            "STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD requires a STOP policy"
        )
if int(collision_limit) < 1 or int(motion_window) < 2 or int(plan_window) < 2:
    raise SystemExit("Closed-loop collision/window counts are invalid")
if (
    int(recovery_turns) < 1
    or int(recovery_forward_steps) < 0
    or int(recovery_cooldown) < 0
    or int(recovery_history_keep) < 0
):
    raise SystemExit("Closed-loop recovery counts are invalid")
if int(recovery_turns) + int(recovery_forward_steps) > 8:
    raise SystemExit("Closed-loop recovery actions must be <= 8")
if recovery_follow_last_turn.lower() not in {
    "0", "1", "false", "true", "no", "yes", "off", "on"
}:
    raise SystemExit("STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN must be boolean")
for name, raw in (
    ("collision_epsilon", collision_epsilon),
    ("motion_min_path", motion_min_path),
    ("motion_max_net", motion_max_net),
    ("plan_min_path", plan_min_path),
    ("plan_max_net", plan_max_net),
):
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"{name} must be finite and >= 0, got {raw!r}")
dominance = float(plan_dominance)
if not math.isfinite(dominance) or not 0.5 < dominance <= 1.0:
    raise SystemExit("STAGE3_EVAL_PLAN_VIEW_DOMINANCE must be in (0.5, 1]")
PY
case "$STAGE3_EVAL_STOP_PROBE_TURN" in
  left|right) ;;
  *)
    echo "STAGE3_EVAL_STOP_PROBE_TURN must be left or right" >&2
    exit 1
    ;;
esac

privileged_requested="$($QWEN25_PYTHON - "$STAGE3_EVAL_AUTO_STOP_DISTANCE" <<'PY'
import sys
print(int(float(sys.argv[1]) > 0.0))
PY
)"
case "${STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES must be boolean" >&2
    exit 1
    ;;
esac
case "${STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES must be boolean" >&2
    exit 1
    ;;
esac
if is_true "$STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES" \
  && ! is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  echo "Multimodal STOP collection requires STAGE3_EVAL_COLLECT_STOP_FEATURES=1" >&2
  exit 1
fi
if [[ -n "$STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS" ]]; then
  "$QWEN25_PYTHON" - "$STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value):
    raise SystemExit("Multimodal regular-negative STOP log-odds threshold must be finite")
PY
  if ! is_true "$STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES"; then
    echo "Multimodal regular-negative filtering requires multimodal collection" >&2
    exit 1
  fi
fi
case "${STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE must be boolean" >&2
    exit 1
    ;;
esac
case "${STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START must be boolean" >&2
    exit 1
    ;;
esac
case "${STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP must be boolean" >&2
    exit 1
    ;;
esac
case "${STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS,,}" in
  0|1|false|true|no|yes|off|on) ;;
  *)
    echo "STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS must be boolean" >&2
    exit 1
    ;;
esac
if is_true "$STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES" \
  && ! is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  echo "Forced STOP-negative continuation requires STAGE3_EVAL_COLLECT_STOP_FEATURES=1" >&2
  exit 1
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE" \
  && ! is_true "$STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES"; then
  echo "Oracle STOP recovery requires forced STOP-negative continuation" >&2
  exit 1
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START"; then
  if ! is_true "$STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES"; then
    echo "Oracle path-from-start collection requires forced STOP-negative continuation" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE"; then
    echo "Oracle path-from-start and recovery-after-negative are mutually exclusive" >&2
    exit 1
  fi
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP" \
  && ! is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START"; then
  echo "Boundary probe sweep requires oracle path-from-start collection" >&2
  exit 1
fi
if is_true "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS"; then
  if ! is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE"; then
    echo "Cohort-triggered recovery requires oracle recovery collection" >&2
    exit 1
  fi
  if [[ -z "$STAGE3_EVAL_EPISODE_LIST" ]]; then
    echo "Cohort-triggered recovery requires STAGE3_EVAL_EPISODE_LIST" >&2
    exit 1
  fi
fi
if ! [[ "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES must be an integer >= 1" >&2
  exit 1
fi
if ! [[ "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL must be an integer >= 1" >&2
  exit 1
fi
if ! [[ "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS" =~ ^[0-9]+$ ]] \
  || (( STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS < 2 )); then
  echo "STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS must be an integer >= 2" >&2
  exit 1
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" ]]; then
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" \
    || -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
    echo "STOP-decision adapter is mutually exclusive with static/temporal STOP policies" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
    echo "STOP-decision adapter cannot run during privileged feature collection" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
    echo "STOP-decision adapter requires the real System2 policy" >&2
    exit 1
  fi
  if [[ "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY" != "0" \
    && "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY" != "0.0" ]]; then
    echo "STOP-decision adapter does not use the original Qwen STOP-probability gate" >&2
    exit 1
  fi
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
  if ! is_true "$STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"; then
    echo "Temporal STOP verifier requires STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
    echo "Temporal STOP verifier cannot run during privileged feature collection" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
    echo "Temporal STOP verifier requires the real System2 policy" >&2
    exit 1
  fi
  if [[ "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY" != "0" \
    && "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY" != "0.0" ]]; then
    echo "Temporal STOP verifier does not support static STOP-add controls" >&2
    exit 1
  fi
fi
if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
  privileged_requested=1
fi
if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  privileged_requested=1
  if ! is_true "$STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"; then
    echo "STOP feature collection requires STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1" >&2
    exit 1
  fi
  if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
    echo "STOP feature collection requires the real System2 policy, not oracle System2" >&2
    exit 1
  fi
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" \
    || -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" \
    || -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
    echo "STOP feature collection must use the unmodified original System2 policy" >&2
    exit 1
  fi
  "$QWEN25_PYTHON" - \
    "$STAGE3_EVAL_STOP_POSITIVE_RADIUS_M" \
    "$STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M" \
    "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M" \
    "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M" \
    "$STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP" <<'PY'
import math
import sys

positive, negative, probe_min, probe_max = map(float, sys.argv[1:5])
probe_enabled = sys.argv[5].lower() in {"1", "true", "yes", "on"}
if not all(math.isfinite(value) for value in (positive, negative, probe_min, probe_max)):
    raise SystemExit("STOP collection radii must be finite")
if not 0.0 < positive < negative:
    raise SystemExit("STOP collection requires 0 < positive radius < negative radius")
if probe_enabled and not positive < probe_min < probe_max:
    raise SystemExit("Boundary probe sweep requires positive radius < min < max")
PY
fi
if [[ "$privileged_requested" == "1" ]] && ! is_true "$STAGE3_EVAL_ALLOW_PRIVILEGED"; then
  echo "Privileged evaluation requested (oracle System2 or auto_stop_distance > 0)." >&2
  echo "Main val_unseen evaluation must keep both disabled. Set STAGE3_EVAL_ALLOW_PRIVILEGED=1 only for a labelled diagnostic run." >&2
  exit 1
fi

require_file "$STAGE3_EVAL_CONFIG"
require_file "$STAGE3_EVAL_BASE_CKPT"
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" ]]; then
  require_file "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT"
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" ]]; then
  require_file "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT"
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
  require_file "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"
fi
require_file "$STAGE3_EVAL_DATA_PATH"
require_dir "$STAGE3_EVAL_SCENES_DIR"
require_dir "$INTERNNAV_MODEL_PATH"
require_dir "$RPC_ROOT/src/vla_rpc"
if [[ -n "$STAGE3_EVAL_EPISODE_LIST" ]]; then
  require_file "$STAGE3_EVAL_EPISODE_LIST"
fi
if is_true "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS"; then
  PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" - \
    "$STAGE3_EVAL_EPISODE_LIST" \
    "$STAGE3_EVAL_RPC_PROTOCOL_SEED" \
    "$STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M" <<'PY'
import sys

from scripts.evaluation.episode_cohort import load_episode_cohort
from scripts.evaluation.stop_dagger import (
    parse_historical_false_stop_trigger,
    validate_historical_false_stop_source,
)

cohort_path, raw_seed, raw_negative_radius = sys.argv[1:]
keys, metadata = load_episode_cohort(cohort_path)
for scene_id, episode_id in keys:
    trigger = parse_historical_false_stop_trigger(
        metadata[(scene_id, episode_id)],
        expected_protocol_seed=int(raw_seed),
        negative_radius_m=float(raw_negative_radius),
    )
    validate_historical_false_stop_source(
        trigger,
        scene_id=scene_id,
        episode_id=episode_id,
    )
print(f"Historical false-STOP cohort preflight passed: episodes={len(keys)}")
PY
fi
gzip -t "$STAGE3_EVAL_DATA_PATH"

scene_count="$(find -L "$STAGE3_EVAL_SCENES_DIR" -type f -name '*.glb' | wc -l | tr -d ' ')"
if [[ "$scene_count" -lt 90 ]]; then
  echo "MP3D scene preflight failed: found ${scene_count}, expected at least 90 in $STAGE3_EVAL_SCENES_DIR" >&2
  exit 1
fi

"$VLNCE_PYTHON" - \
  "$STAGE3_EVAL_DATA_PATH" \
  "$STAGE3_EVAL_SCENES_DIR" \
  "$STAGE3_EVAL_EXPECTED_EPISODES" <<'PY'
import gzip
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
scenes_dir = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])
with gzip.open(data_path, "rt", encoding="utf-8") as handle:
    episodes = json.load(handle).get("episodes", [])
if len(episodes) != expected_episodes:
    raise SystemExit(
        f"Expected {expected_episodes} R2R episodes, found {len(episodes)}"
    )
scene_id = str(episodes[0].get("scene_id", ""))
scene_asset = scenes_dir / scene_id
if not scene_asset.is_file():
    raise SystemExit(
        "Scene root is incompatible with dataset scene ids: "
        f"root={scenes_dir} first_scene_id={scene_id!r} expected={scene_asset}"
    )
print(f"Dataset/scene preflight passed: episodes={len(episodes)} first_scene={scene_asset}")
PY

for python_bin in "$QWEN25_PYTHON" "$VLNCE_PYTHON"; do
  PYTHONPATH="$RPC_PYTHONPATH" "$python_bin" - <<'PY'
from vla_rpc.client import VLAClient
from vla_rpc.core.image import decode_jpeg_to_rgb, encode_rgb_to_jpeg
from vla_rpc.proto import vla_pb2, vla_pb2_grpc

assert hasattr(VLAClient, "infer_json")
print("RPC API import passed")
PY
done

if ! DISPLAY="$STAGE3_EVAL_DISPLAY" timeout 10 xdpyinfo >/dev/null 2>&1; then
  echo "Xvfb/GLX display is unavailable: DISPLAY=$STAGE3_EVAL_DISPLAY" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Waiting for final Stage3 checkpoint: $STAGE3_EVAL_CHECKPOINT"
while true; do
  if [[ -s "$STAGE3_EVAL_CHECKPOINT" ]] && validate_eval_checkpoint; then
    size_before="$(stat -c %s "$STAGE3_EVAL_CHECKPOINT")"
    echo "[$(date '+%F %T')] Checkpoint passed preflight; waiting ${STAGE3_EVAL_CHECKPOINT_SETTLE_S}s for size stability"
    sleep "$STAGE3_EVAL_CHECKPOINT_SETTLE_S"
    size_after="$(stat -c %s "$STAGE3_EVAL_CHECKPOINT")"
    if [[ "$size_before" == "$size_after" ]] && validate_eval_checkpoint; then
      break
    fi
    echo "[$(date '+%F %T')] Checkpoint changed during settle window; continuing to wait"
  fi
  echo "[$(date '+%F %T')] Final Stage3 checkpoint not ready; retrying in ${STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S}s"
  sleep "$STAGE3_EVAL_CHECKPOINT_WAIT_INTERVAL_S"
done

PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_RPC_HOST" "$STAGE3_EVAL_RPC_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
finally:
    sock.close()
print(f"RPC port is available: {host}:{port}")
PY

echo "[stage3-eval] code_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[stage3-eval] base=$STAGE3_EVAL_BASE_CKPT"
echo "[stage3-eval] system2_stop_head=${STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT:-disabled} add_threshold_override=${STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD:-checkpoint} add_min_qwen_stop_probability=$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY"
echo "[stage3-eval] system2_stop_decision_adapter=${STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT:-disabled} waypoint_and_latent_adapter=default_only"
echo "[stage3-eval] system2_temporal_stop_verifier=${STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT:-disabled} veto_only=1"
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" \
  && -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
  echo "[stage3-eval] system2_stop_policy=hybrid_static_add_temporal_veto"
fi
echo "[stage3-eval] stage3=$STAGE3_EVAL_CHECKPOINT"
echo "[stage3-eval] scenes=$STAGE3_EVAL_SCENES_DIR (${scene_count})"
echo "[stage3-eval] data=$STAGE3_EVAL_DATA_PATH expected_episodes=$STAGE3_EVAL_EXPECTED_EPISODES"
echo "[stage3-eval] rpc=$RPC_SERVER_ADDR rpc_root=$RPC_ROOT"
echo "[stage3-eval] rpc_protocol_seed=$STAGE3_EVAL_RPC_PROTOCOL_SEED require_deterministic_sampling=$STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"
echo "[stage3-eval] model_gpu=$STAGE3_EVAL_MODEL_GPU sim_gpu=$STAGE3_EVAL_SIM_GPU display=$STAGE3_EVAL_DISPLAY"
echo "[stage3-eval] output=$STAGE3_EVAL_OUTPUT_PATH"
echo "[stage3-eval] auto_stop=$STAGE3_EVAL_AUTO_STOP_DISTANCE oracle_system2=$STAGE3_EVAL_ORACLE_SYSTEM2"
echo "[stage3-eval] oracle_strategy=$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY lookahead_m=$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M min_ahead_m=$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M max_side_dist_m=$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"
echo "[stage3-eval] trajectory_selection=$STAGE3_EVAL_TRAJECTORY_SELECTION trajectory_x_sign=$STAGE3_EVAL_TRAJECTORY_X_SIGN heading_alignment=$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"
echo "[stage3-eval] closed_loop action_chunk=$STAGE3_EVAL_ACTION_CHUNK_SIZE stop_confirmations=$STAGE3_EVAL_STOP_CONFIRMATIONS stop_confirmation_max_gap_calls=$STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS stop_confirmation_view_sweep=$STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP accept_temporal_confirmed=$STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED temporal_trust_min_margin=$STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN stop_high_confidence_threshold=${STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD:-disabled} stop_probe_turn=$STAGE3_EVAL_STOP_PROBE_TURN loop_guard=$STAGE3_EVAL_CLOSED_LOOP_GUARD recovery=turn:$STAGE3_EVAL_RECOVERY_TURNS,forward:$STAGE3_EVAL_RECOVERY_FORWARD_STEPS,follow_last_turn:$STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN"
echo "[stage3-eval] stop_feature_collection=$STAGE3_EVAL_COLLECT_STOP_FEATURES multimodal_examples=$STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES multimodal_regular_min_stop_log_odds=${STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS:-disabled} force_continue_negatives=$STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES oracle_recovery_after_negative=$STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE oracle_path_from_start=$STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START boundary_probe=$STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP:[$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M,$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M]x$STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS cohort_triggers=$STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS recovery_goal_probes=$STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES recovery_actions_per_call=$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL radii=[$STAGE3_EVAL_STOP_POSITIVE_RADIUS_M,$STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M]"

if is_true "$STAGE3_EVAL_PREFLIGHT_ONLY"; then
  echo "[$(date '+%F %T')] STAGE3_EVAL_PREFLIGHT_ONLY=1; all static preflights passed"
  exit 0
fi

mkdir -p "$LOG_DIR" "$STAGE3_EVAL_OUTPUT_PATH"

server_deterministic_args=()
if is_true "$STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"; then
  server_deterministic_args+=(--require_deterministic_sampling)
fi
server_system2_stop_head_args=()
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" ]]; then
  server_system2_stop_head_args+=(
    --system2_stop_head_checkpoint "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT"
    --system2_stop_add_min_qwen_stop_probability "$STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY"
  )
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD" ]]; then
    server_system2_stop_head_args+=(
      --system2_stop_add_threshold "$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD"
    )
  fi
fi
server_system2_stop_decision_adapter_args=()
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" ]]; then
  server_system2_stop_decision_adapter_args+=(
    --system2_stop_decision_adapter_checkpoint
    "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT"
  )
  if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD" ]]; then
    server_system2_stop_decision_adapter_args+=(
      --system2_stop_add_threshold "$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD"
    )
  fi
fi
server_system2_temporal_stop_args=()
if [[ -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
  server_system2_temporal_stop_args+=(
    --system2_temporal_stop_verifier_checkpoint
    "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"
  )
fi
server_system2_stop_feature_args=()
if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  server_system2_stop_feature_args+=(
    --system2_stop_feature_dump_dir
    "$STAGE3_EVAL_OUTPUT_PATH/system2_stop_features"
  )
fi

env \
  PYTHONPATH="$RPC_PYTHONPATH" \
  CUDA_VISIBLE_DEVICES="$STAGE3_EVAL_MODEL_GPU" \
  USE_TF=0 \
  TRANSFORMERS_NO_TF=1 \
  TF_CPP_MIN_LOG_LEVEL=3 \
  HEATMAPVLN_REQUIRE_FLASH_ATTN=1 \
  "$QWEN25_PYTHON" -u scripts/evaluation/rpc_model_server.py \
    --config "$STAGE3_EVAL_CONFIG" \
    --base_checkpoint "$STAGE3_EVAL_BASE_CKPT" \
    --pano_latent_adapter_checkpoint "$STAGE3_EVAL_CHECKPOINT" \
    --internnav_model_path "$INTERNNAV_MODEL_PATH" \
    --gpu_id 0 \
    --host "$STAGE3_EVAL_RPC_HOST" \
    --port "$STAGE3_EVAL_RPC_PORT" \
    --workers 1 \
    --log_level INFO \
    "${server_system2_stop_head_args[@]}" \
    "${server_system2_stop_decision_adapter_args[@]}" \
    "${server_system2_temporal_stop_args[@]}" \
    "${server_system2_stop_feature_args[@]}" \
    "${server_deterministic_args[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
echo "[$(date '+%F %T')] RPC model server starting pid=$SERVER_PID log=$SERVER_LOG"

server_ready=0
start_time="$(date +%s)"
while [[ "$(($(date +%s) - start_time))" -lt "$STAGE3_EVAL_SERVER_START_TIMEOUT_S" ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "RPC model server exited during startup" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    wait "$SERVER_PID" || true
    exit 1
  fi
  if PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "$RPC_SERVER_ADDR" <<'PY' >/dev/null 2>&1
import sys
from scripts.evaluation.rpc_protocol import HEATMAPVLN_RPC_PROTOCOL_VERSION
from vla_rpc.client import VLAClient

client = VLAClient(server_addr=sys.argv[1], timeout_ms=5000)
try:
    client.connect()
    info = client.get_server_info()
    if not client.health_check() or info is None:
        raise SystemExit(1)
    if info.version != HEATMAPVLN_RPC_PROTOCOL_VERSION:
        raise SystemExit(2)
finally:
    client.close()
PY
  then
    server_ready=1
    break
  fi
  sleep 10
done

if [[ "$server_ready" != "1" ]]; then
  echo "RPC model server was not healthy within ${STAGE3_EVAL_SERVER_START_TIMEOUT_S}s" >&2
  tail -200 "$SERVER_LOG" >&2 || true
  exit 1
fi

for required_log in \
  "Verified complete frozen InternNav System1 for RPC evaluation: 608 tensors" \
  "Verified complete LoRA checkpoint match: 224 tensors" \
  "Base checkpoint LoRA-only guard: loading 224/224 tensors" \
  "Verified pano latent adapter: tensors=4 parameters=7344640 dim=3584 hidden_dim=1024" \
  "hidden_dim=1024 dtype=torch.bfloat16"; do
  if ! grep -Fq "$required_log" "$SERVER_LOG"; then
    echo "RPC startup assertion missing from server log: $required_log" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT" ]]; then
  for required_system2_log in \
    "Verified isolated System2 STOP head: tensors=10 add_threshold=" \
    "veto_threshold=" \
    "original Stage1-S2 LoRA remains the only Qwen adapter"; do
    if ! grep -Fq "$required_system2_log" "$SERVER_LOG"; then
      echo "RPC startup assertion missing from server log: $required_system2_log" >&2
      tail -200 "$SERVER_LOG" >&2 || true
      exit 1
    fi
  done
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT" ]]; then
  for required_stop_decision_log in \
    "Verified isolated System2 STOP-decision LoRA: tensors=64" \
    "navigation generation and System1 latent extraction remain default-LoRA-only"; do
    if ! grep -Fq "$required_stop_decision_log" "$SERVER_LOG"; then
      echo "RPC startup assertion missing from STOP-decision LoRA: $required_stop_decision_log" >&2
      tail -200 "$SERVER_LOG" >&2 || true
      exit 1
    fi
  done
fi
if [[ -n "$STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT" ]]; then
  for required_temporal_log in \
    "Verified veto-only System2 temporal STOP verifier:" \
    "original non-STOP outputs can never be changed"; do
    if ! grep -Fq "$required_temporal_log" "$SERVER_LOG"; then
      echo "RPC startup assertion missing from temporal STOP verifier: $required_temporal_log" >&2
      tail -200 "$SERVER_LOG" >&2 || true
      exit 1
    fi
  done
fi
if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  if ! grep -Fq "System2 STOP DAgger feature collection is ACTIVE" "$SERVER_LOG"; then
    echo "RPC startup assertion missing for STOP DAgger feature collection" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
fi
echo "[$(date '+%F %T')] RPC model server is healthy and all model-load guards passed"

PYTHONPATH="$RPC_PYTHONPATH" "$QWEN25_PYTHON" - \
  "$STAGE3_EVAL_OUTPUT_PATH/eval_manifest.json" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
)

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "code_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    ).stdout.strip() or "unknown",
    "config": os.environ["STAGE3_EVAL_CONFIG"],
    "base_checkpoint": os.environ["STAGE3_EVAL_BASE_CKPT"],
    "system2_stop_head_checkpoint": os.environ["STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT"],
    "system2_stop_decision_adapter_checkpoint": os.environ[
        "STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT"
    ],
    "system2_temporal_stop_verifier_checkpoint": os.environ[
        "STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"
    ],
    "system2_stop_policy_mode": (
        "isolated_stop_decision_lora"
        if os.environ["STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT"]
        else "hybrid_static_add_temporal_veto"
        if os.environ["STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT"]
        and os.environ["STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"]
        else "static_add_and_veto"
        if os.environ["STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT"]
        else "temporal_veto_only"
        if os.environ["STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT"]
        else "original_system2"
    ),
    "system2_stop_add_min_qwen_stop_probability": float(
        os.environ["STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY"]
    ),
    "system2_stop_add_threshold_override": (
        float(os.environ["STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD"])
        if os.environ["STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD"]
        else None
    ),
    "stage3_checkpoint": os.environ["STAGE3_EVAL_CHECKPOINT"],
    "expected_epoch": int(os.environ["STAGE3_EVAL_EXPECTED_EPOCH"]),
    "scenes_dir": os.environ["STAGE3_EVAL_SCENES_DIR"],
    "data_path": os.environ["STAGE3_EVAL_DATA_PATH"],
    "expected_episodes": int(os.environ["STAGE3_EVAL_EXPECTED_EPISODES"]),
    "rpc_root": os.environ["RPC_ROOT"],
    "rpc_protocol": HEATMAPVLN_RPC_PROTOCOL_VERSION,
    "rpc_sampling_protocol": HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
    "rpc_deterministic_sampling_enabled": True,
    "rpc_protocol_seed": int(os.environ["STAGE3_EVAL_RPC_PROTOCOL_SEED"]),
    "rpc_require_deterministic_sampling": os.environ[
        "STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"
    ].lower() in {"1", "true", "yes", "on"},
    "auto_stop_distance": float(os.environ["STAGE3_EVAL_AUTO_STOP_DISTANCE"]),
    "oracle_system2": os.environ["STAGE3_EVAL_ORACLE_SYSTEM2"].lower() in {"1", "true", "yes", "on"},
    "oracle_system2_strategy": os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY"],
    "oracle_system2_lookahead_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M"]),
    "oracle_system2_min_ahead_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M"]),
    "oracle_system2_max_side_dist_m": float(os.environ["STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"]),
    "trajectory_selection": os.environ["STAGE3_EVAL_TRAJECTORY_SELECTION"],
    "trajectory_x_sign": float(os.environ["STAGE3_EVAL_TRAJECTORY_X_SIGN"]),
    "trajectory_heading_alignment": os.environ["STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"],
    "system1_coord_order": os.environ["STAGE3_EVAL_SYSTEM1_COORD_ORDER"],
    "rpc_action_chunk_size": int(os.environ["STAGE3_EVAL_ACTION_CHUNK_SIZE"]),
    "system2_stop_confirmations": int(os.environ["STAGE3_EVAL_STOP_CONFIRMATIONS"]),
    "system2_stop_confirmation_max_gap_calls": int(
        os.environ["STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS"]
    ),
    "system2_stop_confirmation_view_sweep": os.environ[
        "STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_accept_temporal_confirmed": os.environ[
        "STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_temporal_trust_min_margin": float(
        os.environ["STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN"]
    ),
    "system2_stop_high_confidence_threshold": (
        float(os.environ["STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD"])
        if os.environ["STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD"]
        else None
    ),
    "system2_stop_feature_collection": os.environ[
        "STAGE3_EVAL_COLLECT_STOP_FEATURES"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_multimodal_example_collection": os.environ[
        "STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_multimodal_regular_min_stop_log_odds": (
        float(os.environ["STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS"])
        if os.environ["STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS"]
        else None
    ),
    "system2_stop_collect_force_continue_negatives": os.environ[
        "STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_collect_oracle_recovery_after_negative": os.environ[
        "STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_collect_oracle_path_from_start": os.environ[
        "STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_collect_boundary_probe_sweep": os.environ[
        "STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_boundary_probe_min_distance_m": float(
        os.environ["STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M"]
    ),
    "system2_stop_boundary_probe_max_distance_m": float(
        os.environ["STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M"]
    ),
    "system2_stop_boundary_probe_views": int(
        os.environ["STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS"]
    ),
    "system2_stop_oracle_recovery_from_cohort_triggers": os.environ[
        "STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS"
    ].lower() in {"1", "true", "yes", "on"},
    "system2_stop_oracle_recovery_goal_probes": int(
        os.environ["STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES"]
    ),
    "system2_stop_oracle_recovery_actions_per_call": int(
        os.environ["STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL"]
    ),
    "system2_stop_positive_radius_m": float(
        os.environ["STAGE3_EVAL_STOP_POSITIVE_RADIUS_M"]
    ),
    "system2_stop_negative_radius_m": float(
        os.environ["STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M"]
    ),
    "system2_stop_probe_turn": os.environ["STAGE3_EVAL_STOP_PROBE_TURN"],
    "closed_loop_guard": os.environ["STAGE3_EVAL_CLOSED_LOOP_GUARD"].lower()
    in {"1", "true", "yes", "on"},
    "closed_loop_collision_epsilon_m": float(os.environ["STAGE3_EVAL_COLLISION_EPSILON_M"]),
    "closed_loop_collision_forward_limit": int(os.environ["STAGE3_EVAL_COLLISION_FORWARD_LIMIT"]),
    "closed_loop_motion_window_steps": int(os.environ["STAGE3_EVAL_MOTION_WINDOW_STEPS"]),
    "closed_loop_motion_min_path_m": float(os.environ["STAGE3_EVAL_MOTION_MIN_PATH_M"]),
    "closed_loop_motion_max_net_m": float(os.environ["STAGE3_EVAL_MOTION_MAX_NET_M"]),
    "closed_loop_plan_window_calls": int(os.environ["STAGE3_EVAL_PLAN_WINDOW_CALLS"]),
    "closed_loop_plan_view_dominance": float(os.environ["STAGE3_EVAL_PLAN_VIEW_DOMINANCE"]),
    "closed_loop_plan_min_path_m": float(os.environ["STAGE3_EVAL_PLAN_MIN_PATH_M"]),
    "closed_loop_plan_max_net_m": float(os.environ["STAGE3_EVAL_PLAN_MAX_NET_M"]),
    "closed_loop_recovery_turns": int(os.environ["STAGE3_EVAL_RECOVERY_TURNS"]),
    "closed_loop_recovery_forward_steps": int(
        os.environ["STAGE3_EVAL_RECOVERY_FORWARD_STEPS"]
    ),
    "closed_loop_recovery_follow_last_turn": os.environ[
        "STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN"
    ].lower() in {"1", "true", "yes", "on"},
    "closed_loop_recovery_cooldown_steps": int(os.environ["STAGE3_EVAL_RECOVERY_COOLDOWN_STEPS"]),
    "closed_loop_recovery_history_keep": int(os.environ["STAGE3_EVAL_RECOVERY_HISTORY_KEEP"]),
}
path = Path(sys.argv[1])
resume = os.environ["STAGE3_EVAL_RESUME"].lower() in {"1", "true", "yes", "on"}
overwrite = os.environ["STAGE3_EVAL_OVERWRITE"].lower() in {"1", "true", "yes", "on"}
if resume and overwrite:
    raise SystemExit("STAGE3_EVAL_RESUME and STAGE3_EVAL_OVERWRITE cannot both be true")
progress_exists = (path.parent / "progress.json").exists()
if path.exists():
    existing = json.loads(path.read_text())
    comparable_existing = {key: existing.get(key) for key in manifest if key != "created_at"}
    comparable_new = {key: value for key, value in manifest.items() if key != "created_at"}
    if resume:
        if comparable_existing != comparable_new:
            raise SystemExit(
                "Existing eval_manifest.json does not match this resume contract"
            )
    elif not overwrite:
        raise SystemExit(
            "Existing eval_manifest.json requires STAGE3_EVAL_RESUME=1 or "
            "STAGE3_EVAL_OVERWRITE=1"
        )
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
elif progress_exists:
    raise SystemExit("progress.json exists without eval_manifest.json; resume refused")
else:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

client_args=(
  --config "$STAGE3_EVAL_CONFIG"
  --rpc_server "$RPC_SERVER_ADDR"
  --rpc_timeout_ms "$STAGE3_EVAL_RPC_TIMEOUT_MS"
  --rpc_jpeg_quality "$STAGE3_EVAL_RPC_JPEG_QUALITY"
  --rpc_protocol_seed "$STAGE3_EVAL_RPC_PROTOCOL_SEED"
  --scenes_dir "$STAGE3_EVAL_SCENES_DIR"
  --data_path "$STAGE3_EVAL_DATA_PATH"
  --output_path "$STAGE3_EVAL_OUTPUT_PATH"
  --sim_gpu_id 0
  --resize_w 256
  --resize_h 256
  --num_history "$STAGE3_EVAL_NUM_HISTORY"
  --max_steps_per_episode "$STAGE3_EVAL_MAX_STEPS"
  --auto_stop_distance "$STAGE3_EVAL_AUTO_STOP_DISTANCE"
  --max_system2_calls_per_episode "$STAGE3_EVAL_MAX_SYSTEM2_CALLS"
  --rpc_action_chunk_size "$STAGE3_EVAL_ACTION_CHUNK_SIZE"
  --system2_stop_confirmations "$STAGE3_EVAL_STOP_CONFIRMATIONS"
  --system2_stop_confirmation_max_gap_calls "$STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS"
  --system2_stop_probe_turn "$STAGE3_EVAL_STOP_PROBE_TURN"
  --closed_loop_collision_epsilon_m "$STAGE3_EVAL_COLLISION_EPSILON_M"
  --closed_loop_collision_forward_limit "$STAGE3_EVAL_COLLISION_FORWARD_LIMIT"
  --closed_loop_motion_window_steps "$STAGE3_EVAL_MOTION_WINDOW_STEPS"
  --closed_loop_motion_min_path_m "$STAGE3_EVAL_MOTION_MIN_PATH_M"
  --closed_loop_motion_max_net_m "$STAGE3_EVAL_MOTION_MAX_NET_M"
  --closed_loop_plan_window_calls "$STAGE3_EVAL_PLAN_WINDOW_CALLS"
  --closed_loop_plan_view_dominance "$STAGE3_EVAL_PLAN_VIEW_DOMINANCE"
  --closed_loop_plan_min_path_m "$STAGE3_EVAL_PLAN_MIN_PATH_M"
  --closed_loop_plan_max_net_m "$STAGE3_EVAL_PLAN_MAX_NET_M"
  --closed_loop_recovery_turns "$STAGE3_EVAL_RECOVERY_TURNS"
  --closed_loop_recovery_forward_steps "$STAGE3_EVAL_RECOVERY_FORWARD_STEPS"
  --closed_loop_recovery_cooldown_steps "$STAGE3_EVAL_RECOVERY_COOLDOWN_STEPS"
  --closed_loop_recovery_history_keep "$STAGE3_EVAL_RECOVERY_HISTORY_KEEP"
  --trajectory_selection "$STAGE3_EVAL_TRAJECTORY_SELECTION"
  --trajectory_x_sign "$STAGE3_EVAL_TRAJECTORY_X_SIGN"
  --trajectory_heading_alignment "$STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT"
  --system1_coord_order "$STAGE3_EVAL_SYSTEM1_COORD_ORDER"
  --system2_stop_temporal_trust_min_margin "$STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN"
  --no-debug_input_trace
  --debug_save_input_images 0
)
if is_true "$STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING"; then
  client_args+=(--rpc_require_deterministic_sampling)
fi
if is_true "$STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP"; then
  client_args+=(--system2_stop_confirmation_view_sweep)
else
  client_args+=(--no-system2_stop_confirmation_view_sweep)
fi
if is_true "$STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED"; then
  client_args+=(--system2_stop_accept_temporal_confirmed)
else
  client_args+=(--no-system2_stop_accept_temporal_confirmed)
fi
if [[ -n "$STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD" ]]; then
  client_args+=(
    --system2_stop_high_confidence_threshold
    "$STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD"
  )
fi
if is_true "$STAGE3_EVAL_COLLECT_STOP_FEATURES"; then
  client_args+=(
    --collect_system2_stop_features
    --system2_stop_positive_radius_m "$STAGE3_EVAL_STOP_POSITIVE_RADIUS_M"
    --system2_stop_negative_radius_m "$STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M"
  )
fi
if is_true "$STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES"; then
  client_args+=(--collect_system2_stop_multimodal_examples)
  if [[ -n "$STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS" ]]; then
    client_args+=(
      --system2_stop_multimodal_regular_min_stop_log_odds
      "$STAGE3_EVAL_STOP_MULTIMODAL_REGULAR_MIN_STOP_LOG_ODDS"
    )
  fi
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES"; then
  client_args+=(--system2_stop_collect_force_continue_negatives)
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE"; then
  client_args+=(
    --system2_stop_collect_oracle_recovery_after_negative
    --system2_stop_oracle_recovery_goal_probes
    "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES"
    --system2_stop_oracle_recovery_actions_per_call
    "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL"
  )
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START"; then
  client_args+=(
    --system2_stop_collect_oracle_path_from_start
    --system2_stop_oracle_recovery_goal_probes
    "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES"
    --system2_stop_oracle_recovery_actions_per_call
    "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL"
  )
fi
if is_true "$STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP"; then
  client_args+=(
    --system2_stop_collect_boundary_probe_sweep
    --system2_stop_boundary_probe_min_distance_m
    "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MIN_DISTANCE_M"
    --system2_stop_boundary_probe_max_distance_m
    "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_MAX_DISTANCE_M"
    --system2_stop_boundary_probe_views
    "$STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS"
  )
fi
if is_true "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS"; then
  client_args+=(--system2_stop_oracle_recovery_from_cohort_triggers)
fi
if is_true "$STAGE3_EVAL_CLOSED_LOOP_GUARD"; then
  client_args+=(--closed_loop_guard)
else
  client_args+=(--no-closed_loop_guard)
fi
if is_true "$STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN"; then
  client_args+=(--closed_loop_recovery_follow_last_turn)
else
  client_args+=(--no-closed_loop_recovery_follow_last_turn)
fi
if [[ -n "$STAGE3_EVAL_MAX_EPISODES" ]]; then
  client_args+=(--max_episodes "$STAGE3_EVAL_MAX_EPISODES")
fi
if [[ -n "$STAGE3_EVAL_EPISODE_LIST" ]]; then
  client_args+=(--episode_list "$STAGE3_EVAL_EPISODE_LIST")
fi
if is_true "$STAGE3_EVAL_RESUME"; then
  client_args+=(--resume)
fi
if is_true "$STAGE3_EVAL_OVERWRITE"; then
  client_args+=(--overwrite_output)
fi
if is_true "$STAGE3_EVAL_ORACLE_SYSTEM2"; then
  client_args+=(
    --oracle_system2
    --oracle_system2_strategy "$STAGE3_EVAL_ORACLE_SYSTEM2_STRATEGY"
    --oracle_system2_lookahead_m "$STAGE3_EVAL_ORACLE_SYSTEM2_LOOKAHEAD_M"
    --oracle_system2_min_ahead_m "$STAGE3_EVAL_ORACLE_SYSTEM2_MIN_AHEAD_M"
    --oracle_system2_max_side_dist_m "$STAGE3_EVAL_ORACLE_SYSTEM2_MAX_SIDE_DIST_M"
  )
fi
if is_true "$STAGE3_EVAL_SAVE_TRAJECTORY_STEPS"; then
  client_args+=(--save_trajectory_steps)
fi

echo "[$(date '+%F %T')] Starting Habitat val_unseen client log=$CLIENT_LOG"
env \
  PYTHONPATH="$RPC_PYTHONPATH" \
  DISPLAY="$STAGE3_EVAL_DISPLAY" \
  CUDA_VISIBLE_DEVICES="$STAGE3_EVAL_SIM_GPU" \
  HABITAT_GL_GPU_ID=0 \
  HEATMAPVLN_PREINIT_GL=0 \
  HEATMAPVLN_PREINIT_EMPTY_GL=1 \
  USE_TF=0 \
  TRANSFORMERS_NO_TF=1 \
  TF_CPP_MIN_LOG_LEVEL=3 \
  "$VLNCE_PYTHON" -u scripts/evaluation/r2r_val_unseen.py \
    "${client_args[@]}" \
    2>&1 | tee "$CLIENT_LOG"

require_file "$STAGE3_EVAL_OUTPUT_PATH/result.json"
echo "[$(date '+%F %T')] Stage3 R2R val_unseen evaluation complete"
echo "[stage3-eval] result=$STAGE3_EVAL_OUTPUT_PATH/result.json"
echo "[stage3-eval] progress=$STAGE3_EVAL_OUTPUT_PATH/progress.json"
echo "[stage3-eval] server_log=$SERVER_LOG"
echo "[stage3-eval] client_log=$CLIENT_LOG"
