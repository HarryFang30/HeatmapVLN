#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/afs/liwenhao/agent/370910109
REPO=$ROOT/HeatmapVLN
AMB3R_REPO=$ROOT/amb3r
PY=$ROOT/envs/qwen25/bin/python
AMB3R_CKPT=${AMB3R_CKPT:-$AMB3R_REPO/checkpoints/DA3NESTED-GIANT-LARGE}
OUTPUT_ROOT=${FUTURE_HEATMAP_OUTPUT_ROOT:-$ROOT/model/future_heatmap_amb3r_rgb_only_v2}
GPU_DEVICE=${FUTURE_HEATMAP_GPU_DEVICE:-0}

export MACA_HOME=/opt/maca-3.3.0
export MACA_PATH=/opt/maca-3.3.0
export MACA_DIR=/opt/maca-3.3.0
export LD_LIBRARY_PATH=/opt/maca-3.3.0/lib:/opt/maca-3.3.0/ompi/lib:/opt/maca-3.3.0/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export HF_HOME=$AMB3R_REPO/checkpoints/runtime_cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$AMB3R_REPO/checkpoints/runtime_cache/torch
export XDG_CACHE_HOME=$AMB3R_REPO/checkpoints/runtime_cache/xdg
export MPLCONFIGDIR=$ROOT/model/.runtime_cache/matplotlib_future_amb3r
export TRITON_CACHE_DIR=$ROOT/model/.runtime_cache/triton_future_amb3r
export PYTHONPATH=$REPO:$AMB3R_REPO:$AMB3R_REPO/thirdparty:${PYTHONPATH:-}

mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

EPISODE_ROOT=$ROOT/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17/shard_00/episodes
EPISODE_A=${FUTURE_HEATMAP_EPISODE_A:-$EPISODE_ROOT/round00_17DRP5sb8fy_000575/episode.tar}
EPISODE_B=${FUTURE_HEATMAP_EPISODE_B:-$EPISODE_ROOT/round00_1LXtFkjw3qL_000097/episode.tar}
EPISODE_C=${FUTURE_HEATMAP_EPISODE_C:-$EPISODE_ROOT/round00_1LXtFkjw3qL_002134/episode.tar}

for required in "$PY" "$AMB3R_CKPT/config.json" "$EPISODE_A" "$EPISODE_B" "$EPISODE_C"; do
  if [[ ! -e "$required" ]]; then
    echo "[future-amb3r] missing required path: $required" >&2
    exit 1
  fi
done

cd "$REPO"
"$PY" scripts/visualization/visualize_system2_future_heatmap_amb3r.py \
  --episode-tar "$EPISODE_A" \
  --episode-tar "$EPISODE_B" \
  --episode-tar "$EPISODE_C" \
  --output-root "$OUTPUT_ROOT" \
  --amb3r-repo "$AMB3R_REPO" \
  --checkpoint "$AMB3R_CKPT" \
  --device cuda:0 \
  --max-records "${FUTURE_HEATMAP_MAX_RECORDS:-8}" \
  --confidence "${FUTURE_HEATMAP_DISPLAY_CONFIDENCE:-0.85}" \
  --process-res "${FUTURE_HEATMAP_PROCESS_RES:-504}" \
  --tile "${FUTURE_HEATMAP_TILE:-80}" \
  --gap "${FUTURE_HEATMAP_GAP:-4}"

echo "[future-amb3r] RGB-only visualizations ready: $OUTPUT_ROOT"
