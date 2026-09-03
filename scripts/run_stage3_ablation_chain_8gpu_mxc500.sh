#!/usr/bin/env bash
# Chain the Stage-3 (bridge-only action-refine) ablation arms of EXP-08/EXP-09
# on one 8-GPU blank container, one arm after another.  Every arm reuses the
# certified v2 launch (configs/ppa_action_refine_v2_8gpu.yaml semantics, fresh
# optimizer, exact-zero bridge, 3 epochs) and differs from it in exactly one
# thing — see configs/ablation/*.yaml and docs/experiments/README.md.
#
# Parameters are environment variables (website submission form):
#   ABLATION_ARMS   space-separated subset/order of: exp09a exp09b exp09c exp08
#   PPA_ABLATION_ROOT  output root; each arm writes <root>/<arm>/run_<stamp>/
#   PPA_DATA_ROOT / PPA_AMB3R_CACHE_ROOT / INTERNNAV_MODEL_PATH  as for training
# A failed arm is reported and the chain continues with the next one.

set -uo pipefail

FJL_ROOT="${PPA_EVAL_FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}"
REPO="${PPA_ABLATION_REPO:-$FJL_ROOT/HeatmapVLN}"
PYTHON="${PPA_ABLATION_PYTHON:-$FJL_ROOT/envs/qwen25/bin/python}"
ROOT="${PPA_ABLATION_ROOT:-$FJL_ROOT/model/ablation_stage3}"
ARMS="${ABLATION_ARMS:-exp09a exp09b exp09c exp08}"
EPOCHS="${PPA_ABLATION_EPOCHS:-3}"
MASTER_PORT_BASE="${PPA_ABLATION_MASTER_PORT_BASE:-29701}"

export PPA_DATA_ROOT="${PPA_DATA_ROOT:-$FJL_ROOT/r2r_panoramic_data_v2/train}"
export PPA_AMB3R_CACHE_ROOT="${PPA_AMB3R_CACHE_ROOT:-$FJL_ROOT/data/amb3r_endpoint_v3_full_r2r}"
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$FJL_ROOT/InternNav-Model}"

# Parents: the recipe arms start from the Stage-2 heads exactly like v2; EXP-08
# starts from the Stage-1 heads (never saw the action loss).  Bridges are reset.
STAGE2_BEST="${PPA_ABLATION_STAGE2_BEST:-$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/stage2_joint/run_20260818_104438/checkpoints/best.pth}"
STAGE1_BEST="${PPA_ABLATION_STAGE1_BEST:-$FJL_ROOT/model/output_past_plan_action_v1_8gpu/stage1_map_pretrain/run_20260817_205027/checkpoints/best.pth}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

# Warm HF/torch/triton caches from the Stage-2 run (not keyed to data or arm).
RUNTIME_ROOT="${PPA_ABLATION_RUNTIME_ROOT:-$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache}"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"

die() { printf '[ablation] ERROR: %s\n' "$*" >&2; exit 2; }
arm_config() {
  case "$1" in
    exp09a) echo "$REPO/configs/ablation/exp09a_stage3_no_trust_region_8gpu.yaml" ;;
    exp09b) echo "$REPO/configs/ablation/exp09b_stage3_no_advantage_8gpu.yaml" ;;
    exp09c) echo "$REPO/configs/ablation/exp09c_stage3_v1_penalties_8gpu.yaml" ;;
    exp08)  echo "$REPO/configs/ablation/exp08_stage3_from_stage1_heads_8gpu.yaml" ;;
    *) return 1 ;;
  esac
}
arm_parent() {
  case "$1" in
    exp08) echo "$STAGE1_BEST" ;;
    exp09a|exp09b|exp09c) echo "$STAGE2_BEST" ;;
    *) return 1 ;;
  esac
}

[[ -x "$PYTHON" ]] || die "missing python: $PYTHON"
[[ -f "$STAGE2_BEST" && -f "$STAGE1_BEST" ]] || die "missing parent checkpoint(s)"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#gpus[@]}" -eq 8 ]] || die "exactly 8 GPUs are required (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for arm in $ARMS; do
  cfg="$(arm_config "$arm")" || die "unknown arm: $arm"
  [[ -f "$cfg" ]] || die "missing config for $arm: $cfg"
done

cd "$REPO" || die "cannot cd to $REPO"
mkdir -p "$ROOT"
declare -a summary
index=0
for arm in $ARMS; do
  cfg="$(arm_config "$arm")"; parent="$(arm_parent "$arm")"
  port=$((MASTER_PORT_BASE + index)); index=$((index + 1))
  export PPA_ACTION_REFINE_OUTPUT_ROOT="$ROOT/$arm"
  export PPA_TENSORBOARD_ROOT="$ROOT/$arm/tensorboard"
  mkdir -p "$PPA_ACTION_REFINE_OUTPUT_ROOT"
  echo "[ablation] ===== $arm  config=$cfg  parent=$parent  out=$PPA_ACTION_REFINE_OUTPUT_ROOT  $(date -u +%FT%TZ)"
  if "$PYTHON" -m torch.distributed.run \
      --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="$port" \
      scripts/train.py --config "$cfg" --load-weights "$parent" \
      --distributed --epochs "$EPOCHS" --num-workers 2 --pin-memory --prefetch-factor 2; then
    summary+=("$arm: OK")
  else
    summary+=("$arm: FAILED (exit $?)")
  fi
  echo "[ablation] ===== $arm finished $(date -u +%FT%TZ)"
done
printf '[ablation] summary:\n'; printf '  %s\n' "${summary[@]}"
for line in "${summary[@]}"; do [[ "$line" == *": OK" ]] || exit 1; done
