#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
STATIC_CHECKPOINT="${SYSTEM2_STATIC_OOF_CHECKPOINT:-${FJL_ROOT}/model/output_system2_stop_head_oracle_path_add_scene_oof_seed123/latest.pth}"
TEMPORAL_CHECKPOINT="${SYSTEM2_TEMPORAL_VETO_CHECKPOINT:-${FJL_ROOT}/model/output_system2_temporal_stop_verifier_ensemble_train_rollouts_pilot/latest.pth}"

export STAGE3_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_EVAL_BASE_CKPT="${STAGE3_EVAL_BASE_CKPT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords_priorfix/latest/checkpoints/epoch_002.pth}"
export STAGE3_EVAL_EXPECTED_EPOCH=2
export STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT="$STATIC_CHECKPOINT"
export STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT="$TEMPORAL_CHECKPOINT"
export STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD="${SYSTEM2_STATIC_OOF_ADD_THRESHOLD:-0.98}"
export STAGE3_EVAL_STOP_ADD_MIN_QWEN_STOP_PROBABILITY=0

export STAGE3_EVAL_MODEL_GPU="${STAGE3_STATIC_OOF_FIXED3_GPU:-7}"
export STAGE3_EVAL_SIM_GPU="$STAGE3_EVAL_MODEL_GPU"
export STAGE3_EVAL_RPC_PORT="${STAGE3_STATIC_OOF_FIXED3_RPC_PORT:-50077}"
export STAGE3_EVAL_DISPLAY="${STAGE3_STATIC_OOF_FIXED3_DISPLAY:-localhost:200.0}"
export STAGE3_EVAL_OUTPUT_PATH="${STAGE3_STATIC_OOF_FIXED3_OUTPUT_PATH:-${FJL_ROOT}/model/eval_system2_static_oof_threshold098_fixed3_20260718}"
export STAGE3_EVAL_EPISODE_LIST=configs/eval_cohorts/stage3_stop_head_fixed3.json
export STAGE3_EVAL_EXPECTED_EPISODES=1839

export STAGE3_EVAL_RPC_PROTOCOL_SEED=42
export STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1
export STAGE3_EVAL_MAX_STEPS=500
export STAGE3_EVAL_ACTION_CHUNK_SIZE=4
# This exactly matches the scene-OOF event calibration: two strict consecutive
# static-add decisions, with one left-view probe between the two RPC calls.
export STAGE3_EVAL_STOP_CONFIRMATIONS=2
export STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS=0
export STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP=0
export STAGE3_EVAL_STOP_ACCEPT_TEMPORAL_CONFIRMED=1
export STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN=0.005
unset STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD
export STAGE3_EVAL_CLOSED_LOOP_GUARD=1
export STAGE3_EVAL_RECOVERY_TURNS=6
export STAGE3_EVAL_RECOVERY_FORWARD_STEPS=2
export STAGE3_EVAL_RECOVERY_FOLLOW_LAST_TURN=1

export STAGE3_EVAL_TRAJECTORY_SELECTION=mean
export STAGE3_EVAL_TRAJECTORY_X_SIGN=1
export STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT=none
export STAGE3_EVAL_SYSTEM1_COORD_ORDER=generated
export STAGE3_EVAL_AUTO_STOP_DISTANCE=0
export STAGE3_EVAL_ORACLE_SYSTEM2=0
export STAGE3_EVAL_ALLOW_PRIVILEGED=0
export STAGE3_EVAL_COLLECT_STOP_FEATURES=0
export STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES=0
export STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE=0
export STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START=0
export STAGE3_EVAL_RESUME=0
export STAGE3_EVAL_OVERWRITE=0
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS=1

for checkpoint in "$STATIC_CHECKPOINT" "$TEMPORAL_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing STOP-policy checkpoint: $checkpoint" >&2
    exit 1
  fi
done
if [[ -e "$STAGE3_EVAL_OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite static-OOF fixed3 output: $STAGE3_EVAL_OUTPUT_PATH" >&2
  exit 1
fi

echo "[$(date '+%F %T')] launching static-OOF threshold A/B on GPU $STAGE3_EVAL_MODEL_GPU"
echo "[static-oof-fixed3] static_add=$STATIC_CHECKPOINT threshold=$STAGE3_EVAL_SYSTEM2_STOP_ADD_THRESHOLD"
echo "[static-oof-fixed3] temporal_veto=$TEMPORAL_CHECKPOINT"
exec bash scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh
