#!/usr/bin/env bash
# 20 fixed val_unseen episodes in tmux (HeatmapVLN Stage2 + eval patches).
set -euo pipefail

SESSION="${1:-eval_20ep_v4}"
ROOT="/workspace/HeatmapVLN"
OUT="${2:-${ROOT}/logs/compare_rerun_20ep_v4/heatmapvln}"
DEBUG_SAVE_IMAGES="${DEBUG_SAVE_INPUT_IMAGES:-8}"
EP="${ROOT}/logs/compare_rerun_20ep/episode_list.json"

export DISPLAY="${DISPLAY:-:200}"
export USE_TF=0 TRANSFORMERS_NO_TF=1
export HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1
export HABITAT_GL_GPU_ID=0
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}"
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

pgrep -a Xvfb >/dev/null || Xvfb :200 -screen 0 1024x768x24 >/tmp/xvfb_200.log 2>&1 &

mkdir -p "$OUT"
rm -f "$OUT/progress.json" "$OUT/result.json"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$ROOT" bash -lc "
set -e
exec > >(tee -a '$OUT/run.log') 2>&1
echo '=== eval start: \$(date -Is) ==='
echo 'session=$SESSION gpu=\$CUDA_VISIBLE_DEVICES'
python -u scripts/evaluate.py r2r \\
  --config configs/train_config_internnav_8gpu.yaml \\
  --base_checkpoint checkpoints/stage1-s2_latest.pth \\
  --checkpoint checkpoints/stage2_latest.pth \\
  --gpu_id 0 --sim_gpu_id 0 \\
  --scenes_dir /dataset/mp3d \\
  --data_path /workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz \\
  --episode_list '$EP' \\
  --auto_stop_distance 3.0 \\
  --max_system2_calls_per_episode 160 \\
  --debug_save_input_images $DEBUG_SAVE_IMAGES \\
  --overwrite_output \\
  --output_path '$OUT'
echo '=== eval end: \$(date -Is) ==='
"

echo "tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: $OUT/run.log"
