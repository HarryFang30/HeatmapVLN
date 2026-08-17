#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
RUN_STAMP="${STAGE3_FORCE_CONTINUE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export STAGE3_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-configs/train_stage3_pano_system1_h1024_8gpu.yaml}"
export STAGE3_EVAL_BASE_CKPT="${STAGE3_EVAL_BASE_CKPT:-${FJL_ROOT}/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords_priorfix/latest/checkpoints/epoch_002.pth}"
export STAGE3_EVAL_EXPECTED_EPOCH=2

export STAGE3_EVAL_DATA_PATH="${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
export STAGE3_EVAL_EXPECTED_EPISODES=10819
export STAGE3_EVAL_EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-configs/eval_cohorts/system2_stop_force_continue_train3.json}"
export STAGE3_EVAL_OUTPUT_PATH="${STAGE3_EVAL_OUTPUT_PATH:-${FJL_ROOT}/model/smoke_system2_stop_force_continue_train3_${RUN_STAMP}}"

export STAGE3_EVAL_MODEL_GPU="${STAGE3_FORCE_CONTINUE_GPU:-7}"
export STAGE3_EVAL_SIM_GPU="$STAGE3_EVAL_MODEL_GPU"
export STAGE3_EVAL_RPC_PORT="${STAGE3_FORCE_CONTINUE_RPC_PORT:-50073}"
export STAGE3_EVAL_DISPLAY="${STAGE3_FORCE_CONTINUE_DISPLAY:-localhost:200.0}"

export STAGE3_EVAL_RPC_PROTOCOL_SEED="${STAGE3_EVAL_RPC_PROTOCOL_SEED:-117}"
export STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1
export STAGE3_EVAL_MAX_STEPS="${STAGE3_EVAL_MAX_STEPS:-500}"
export STAGE3_EVAL_ACTION_CHUNK_SIZE=4
export STAGE3_EVAL_STOP_CONFIRMATIONS=1
export STAGE3_EVAL_STOP_CONFIRMATION_MAX_GAP_CALLS=0
export STAGE3_EVAL_STOP_CONFIRMATION_VIEW_SWEEP=0
unset STAGE3_EVAL_STOP_HIGH_CONFIDENCE_THRESHOLD
export STAGE3_EVAL_CLOSED_LOOP_GUARD=0

export STAGE3_EVAL_TRAJECTORY_SELECTION=mean
export STAGE3_EVAL_TRAJECTORY_X_SIGN=1
export STAGE3_EVAL_TRAJECTORY_HEADING_ALIGNMENT=none
export STAGE3_EVAL_SYSTEM1_COORD_ORDER=generated
export STAGE3_EVAL_AUTO_STOP_DISTANCE=0
export STAGE3_EVAL_ORACLE_SYSTEM2=0
export STAGE3_EVAL_SYSTEM2_STOP_HEAD_CKPT=""
export STAGE3_EVAL_SYSTEM2_STOP_DECISION_ADAPTER_CKPT=""
export STAGE3_EVAL_SYSTEM2_TEMPORAL_STOP_VERIFIER_CKPT=""

export STAGE3_EVAL_COLLECT_STOP_FEATURES=1
export STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES="${STAGE3_EVAL_COLLECT_STOP_MULTIMODAL_EXAMPLES:-1}"
export STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES=1
export STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE="${STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE:-1}"
export STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START="${STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START:-0}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES="${STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES:-8}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL="${STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL:-4}"
export STAGE3_EVAL_STOP_POSITIVE_RADIUS_M=3.0
export STAGE3_EVAL_STOP_NEGATIVE_RADIUS_M=3.01
export STAGE3_EVAL_ALLOW_PRIVILEGED=1
export STAGE3_EVAL_RESUME="${STAGE3_EVAL_RESUME:-0}"
export STAGE3_EVAL_OVERWRITE=0
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS=0

if [[ -e "$STAGE3_EVAL_OUTPUT_PATH" ]]; then
  case "${STAGE3_EVAL_RESUME,,}" in
    1|true|yes|y|on)
      if [[ ! -d "$STAGE3_EVAL_OUTPUT_PATH" ]]; then
        echo "Resume output is not a directory: $STAGE3_EVAL_OUTPUT_PATH" >&2
        exit 1
      fi
      if [[ -s "$STAGE3_EVAL_OUTPUT_PATH/result.json" ]]; then
        echo "Refusing to resume an already completed output: $STAGE3_EVAL_OUTPUT_PATH" >&2
        exit 1
      fi
      echo "Resuming interrupted force-continue smoke: $STAGE3_EVAL_OUTPUT_PATH"
      ;;
    0|false|no|n|off)
      echo "Refusing to overwrite existing force-continue smoke output: $STAGE3_EVAL_OUTPUT_PATH" >&2
      exit 1
      ;;
    *)
      echo "STAGE3_EVAL_RESUME must be boolean, got: $STAGE3_EVAL_RESUME" >&2
      exit 1
      ;;
  esac
else
  case "${STAGE3_EVAL_RESUME,,}" in
    1|true|yes|y|on)
      echo "Resume requested but output does not exist: $STAGE3_EVAL_OUTPUT_PATH" >&2
      exit 1
      ;;
    0|false|no|n|off) ;;
    *)
      echo "STAGE3_EVAL_RESUME must be boolean, got: $STAGE3_EVAL_RESUME" >&2
      exit 1
      ;;
  esac
fi

echo "[$(date '+%F %T')] launching train3 false-STOP intervention smoke"
echo "[force-continue-smoke] gpu=$STAGE3_EVAL_MODEL_GPU seed=$STAGE3_EVAL_RPC_PROTOCOL_SEED"
echo "[force-continue-smoke] output=$STAGE3_EVAL_OUTPUT_PATH"
exec bash scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh
