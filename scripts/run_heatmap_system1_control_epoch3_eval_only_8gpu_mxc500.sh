#!/usr/bin/env bash
# Evaluate the sealed epoch-3 heatmap-control deployment without re-entering
# training. The locked evaluator runs an 8-way smoke before all 1839 episodes.

set -Eeuo pipefail
umask 027

readonly FJL_ROOT="/mnt/afs/lixiaoou/intern/fjl"
readonly CONTROL_CHECKPOINT="${FJL_ROOT}/model/output_heatmap_system1_control_v1/runs/run_20260807_112540/checkpoints/epoch_003.pth"
readonly EXPECTED_CONTROL_SHA256="a556329887be4e6d33f129e1bc670c6515d6a3634b2f3a210ff40b8d21dc9635"
readonly HEATMAP_CHECKPOINT="${FJL_ROOT}/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth"
readonly EXPECTED_HEATMAP_SHA256="08d3dc9b8673c4c8d626a0c9719ad61a99d98654a36853285b5c4ff1bc463b0a"
readonly EVAL_PLAN="${FJL_ROOT}/evaluation_plans/heatmap_control_r2r_val_unseen_8gpu_20260804"
readonly EVAL_SERVER="${EVAL_PLAN}/tools/rpc_heatmap_control_server.py"
readonly EVAL_LAUNCHER="${EVAL_PLAN}/scripts/run_8gpu_heatmap_control_rpc_eval.sh"

die() {
  echo "[eval-only] ERROR: $*" >&2
  exit 1
}

require_regular_file() {
  [[ -f "$1" && -s "$1" && ! -L "$1" ]] || die "missing, empty, or symlinked file: $1"
}

file_sha256() {
  local line hash ignored
  line="$(sha256sum -- "$1")" || return 1
  read -r hash ignored <<< "$line"
  [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s\n' "$hash"
}

for path in "$CONTROL_CHECKPOINT" "$HEATMAP_CHECKPOINT" "$EVAL_SERVER" "$EVAL_LAUNCHER"; do
  require_regular_file "$path"
done

CONTROL_SHA256="$(file_sha256 "$CONTROL_CHECKPOINT")" || die "could not hash epoch-3 checkpoint"
HEATMAP_SHA256="$(file_sha256 "$HEATMAP_CHECKPOINT")" || die "could not hash heatmap checkpoint"
EVAL_SERVER_SHA256="$(file_sha256 "$EVAL_SERVER")" || die "could not hash eval server"
[[ "$CONTROL_SHA256" == "$EXPECTED_CONTROL_SHA256" ]] || die "epoch-3 checkpoint SHA-256 mismatch"
[[ "$HEATMAP_SHA256" == "$EXPECTED_HEATMAP_SHA256" ]] || die "heatmap checkpoint SHA-256 mismatch"

export EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export EVAL_RPC_PORT_BASE="${EVAL_RPC_PORT_BASE:-51400}"
export EVAL_DISPLAY_BASE="${EVAL_DISPLAY_BASE:-280}"
export EVAL_CONTROL_MODE=on
export EVAL_X11_MODE=bundle
export EVAL_HEATMAP_CHECKPOINT="$HEATMAP_CHECKPOINT"
export EVAL_HEATMAP_SHA256="$HEATMAP_SHA256"
export EVAL_CONTROL_CHECKPOINT="$CONTROL_CHECKPOINT"
export EVAL_CONTROL_SHA256="$CONTROL_SHA256"

default_output="${FJL_ROOT}/model/output_heatmap_system1_control_v1/evaluation/r2r_val_unseen_epoch003_${CONTROL_SHA256:0:12}_plan${EVAL_SERVER_SHA256:0:12}"
EVAL_OUTPUT_ROOT="$(readlink -m -- "${EVAL_OUTPUT_ROOT:-$default_output}")"
[[ "$EVAL_OUTPUT_ROOT" != "$FJL_ROOT" ]] || die "broad FJL root is not a valid output"
case "${EVAL_OUTPUT_ROOT}/" in
  "${FJL_ROOT}/"*) ;;
  *) die "EVAL_OUTPUT_ROOT escapes ${FJL_ROOT}: ${EVAL_OUTPUT_ROOT}" ;;
esac
export EVAL_OUTPUT_ROOT

unset EVAL_PREFLIGHT_ONLY EVAL_SMOKE_ONLY EVAL_SKIP_SMOKE EVAL_REUSE_XVFB

echo "[eval-only] control_checkpoint=$CONTROL_CHECKPOINT"
echo "[eval-only] control_sha256=$CONTROL_SHA256"
echo "[eval-only] heatmap_sha256=$HEATMAP_SHA256"
echo "[eval-only] eval_server_sha256=$EVAL_SERVER_SHA256"
echo "[eval-only] output=$EVAL_OUTPUT_ROOT"
echo "[eval-only] starting 8-way smoke, then full 1839-episode val_unseen evaluation"

exec /usr/bin/bash "$EVAL_LAUNCHER"
