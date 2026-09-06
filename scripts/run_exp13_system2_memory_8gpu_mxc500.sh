#!/usr/bin/env bash
# EXP-13 B: fine-tune System2 with the history memory in its prompt.
#
# Chains the two arms on one 8-GPU blank container, treatment first:
#
#   exp13a  model.system2_memory.mode=memory     (M_t reaches the prompt)
#   exp13b  model.system2_memory.mode=constant   (same tokens, no memory)
#
# The two configs differ in that one line and the stage name; everything else,
# including the relabelled data and the LoRA budget, is identical.  Running
# both is the point: a DAgger fine-tune can improve the policy on its own, and
# without the control arm no gain could be attributed to the memory.
#
# EXP-14 reuses this launcher with EXP13_ARMS="exp14a exp14b": the same two
# arms, trained on data that also carries the stop relabelling
# (data.dagger_system2_sft.stop_supervision=true).  Ledger: EXP-14.
#
# GPU count follows CUDA_VISIBLE_DEVICES and must be 4 or 8; the matching
# configs/ablation/<arm>_..._{4,8}gpu.yaml is selected.  The 4-GPU configs
# double grad_accum_steps so every optimizer step still consumes the same
# 16-sample window (2026-09-06).  Only the EXP-14 arms have 4-GPU configs.
#
# A failed arm is reported and the chain continues, so one bad arm does not
# cost the other one's GPU hours.
#
# Website submission form (CLAUDE.md 1.1).  Ledger: docs/experiments/README.md.

set -uo pipefail

FJL_ROOT="${EXP13_FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}"
REPO="${EXP13_REPO:-$FJL_ROOT/HeatmapVLN}"
PYTHON="${EXP13_PYTHON:-$FJL_ROOT/envs/qwen25/bin/python}"
ROOT="${EXP13_TRAIN_ROOT:-$FJL_ROOT/model/exp13_system2_memory}"
ARMS="${EXP13_ARMS:-exp13a exp13b}"
EPOCHS="${EXP13_EPOCHS:-2}"
MASTER_PORT_BASE="${EXP13_MASTER_PORT_BASE:-29781}"
DAGGER_ROOT="${EXP13_DAGGER_ROOT:-$FJL_ROOT/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17}"
PARENT="${EXP13_PARENT_CHECKPOINT:-$FJL_ROOT/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth}"

export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$FJL_ROOT/InternNav-Model}"
export EXP13_ORACLE_VIEWS="${EXP13_ORACLE_VIEWS:-$FJL_ROOT/model/exp12_recovery_gate/d1_per_state.jsonl}"
# EXP-17 (C3): reference-path lengths are the progress denominator of the cognition prefix.
export R2R_TRAIN_JSON="${R2R_TRAIN_JSON:-$FJL_ROOT/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

RUNTIME_ROOT="${EXP13_RUNTIME_ROOT:-$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache}"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"

die() { printf '[exp13-train] ERROR: %s\n' "$*" >&2; exit 2; }

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${#gpus[@]}"
[[ "$NPROC" -eq 8 || "$NPROC" -eq 4 ]] || die "4 or 8 GPUs are required (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

arm_config() {
  case "$1" in
    exp13a) echo "$REPO/configs/ablation/exp13a_system2_memory_lora_${NPROC}gpu.yaml" ;;
    exp13b) echo "$REPO/configs/ablation/exp13b_system2_constant_lora_${NPROC}gpu.yaml" ;;
    exp14a) echo "$REPO/configs/ablation/exp14a_system2_memory_stop_lora_${NPROC}gpu.yaml" ;;
    exp14b) echo "$REPO/configs/ablation/exp14b_system2_constant_stop_lora_${NPROC}gpu.yaml" ;;
    exp17a) echo "$REPO/configs/ablation/exp17a_c1_geometry_stop_lora_${NPROC}gpu.yaml" ;;
    exp17b) echo "$REPO/configs/ablation/exp17b_c3_geometry_prefix_stop_lora_${NPROC}gpu.yaml" ;;
    *) return 1 ;;
  esac
}

[[ -x "$PYTHON" ]] || die "missing python: $PYTHON"
[[ -d "$DAGGER_ROOT" ]] || die "missing DAgger collection: $DAGGER_ROOT"
[[ -f "$PARENT" ]] || die "missing parent checkpoint: $PARENT"
[[ -f "$EXP13_ORACLE_VIEWS" ]] || die "missing oracle rows: $EXP13_ORACLE_VIEWS"
[[ -f "$R2R_TRAIN_JSON" ]] || die "missing R2R train annotation: $R2R_TRAIN_JSON"
[[ -e "$INTERNNAV_MODEL_PATH" ]] || die "missing model path: $INTERNNAV_MODEL_PATH"
for arm in $ARMS; do
  cfg="$(arm_config "$arm")" || die "unknown arm: $arm"
  [[ -f "$cfg" ]] || die "missing config for $arm: $cfg"
done

# The sealed collection names its own shards and its own collecting policy.
# Reading them here keeps the fingerprint in the config from drifting away
# from the data it is supposed to gate.
shards=("$DAGGER_ROOT"/shard_*)
[[ "${#shards[@]}" -eq 4 ]] || die "expected 4 sealed shards, found ${#shards[@]}"
export DAGGER_ROOT_00="${shards[0]}"
export DAGGER_ROOT_01="${shards[1]}"
export DAGGER_ROOT_02="${shards[2]}"
export DAGGER_ROOT_03="${shards[3]}"
fingerprints="$("$PYTHON" - "$DAGGER_ROOT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
values = {
    json.loads((shard / "collection_manifest.json").read_text())["contract"]["policy_fingerprint"]
    for shard in sorted(root.glob("shard_*"))
}
print("\n".join(sorted(values)))
PY
)" || die "cannot read collection manifests"
[[ "$(wc -l <<< "$fingerprints")" -eq 1 ]] || die "shards disagree on the collecting policy"
export DAGGER_POLICY_FINGERPRINT="$fingerprints"
echo "[exp13-train] policy fingerprint: $DAGGER_POLICY_FINGERPRINT"

echo "[exp13-train] world size: $NPROC (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for arm in $ARMS; do echo "[exp13-train] $arm -> $(arm_config "$arm")"; done
if [[ "${EXP13_DRY_RUN:-0}" == "1" ]]; then
  echo "[exp13-train] EXP13_DRY_RUN=1: preflight passed, not launching"
  exit 0
fi

cd "$REPO" || die "cannot cd to $REPO"
mkdir -p "$ROOT"
declare -a summary
index=0
for arm in $ARMS; do
  cfg="$(arm_config "$arm")"
  port=$((MASTER_PORT_BASE + index)); index=$((index + 1))
  export EXP13_OUTPUT_ROOT="$ROOT/$arm"
  export EXP13_TENSORBOARD_ROOT="$ROOT/$arm/tensorboard"
  mkdir -p "$EXP13_OUTPUT_ROOT"
  echo "[exp13-train] ===== $arm config=$cfg out=$EXP13_OUTPUT_ROOT $(date -u +%FT%TZ)"
  if "$PYTHON" -m torch.distributed.run \
      --nproc_per_node="$NPROC" --master_addr=127.0.0.1 --master_port="$port" \
      scripts/train.py --config "$cfg" --load-weights "$PARENT" \
      --distributed --epochs "$EPOCHS" --num-workers 2 --pin-memory --prefetch-factor 2; then
    summary+=("$arm: OK")
  else
    summary+=("$arm: FAILED (exit $?)")
  fi
  echo "[exp13-train] ===== $arm finished $(date -u +%FT%TZ)"
done
printf '[exp13-train] summary:\n'; printf '  %s\n' "${summary[@]}"
for line in "${summary[@]}"; do [[ "$line" == *": OK" ]] || exit 1; done
