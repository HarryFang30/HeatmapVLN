#!/usr/bin/env bash
# Restart Phase1 1B: adapter 100ep on CUDA:3, max_steps=300, resume progress.
set -euo pipefail

ROOT="/workspace/HeatmapVLN"
OUT="${ROOT}/logs/phase1_baseline"
EP100="${OUT}/episode_list_100.json"
SESSION="${1:-phase1_adapter_100}"

export DISPLAY="${DISPLAY:-:200}"
pgrep -a Xvfb >/dev/null || Xvfb :200 -screen 0 1024x768x24 >/tmp/xvfb_200.log 2>&1 &

tmux kill-session -t "${SESSION}" 2>/dev/null || true
pkill -f "evaluate.py r2r.*phase1_baseline/adapter_100ep" 2>/dev/null || true
sleep 2

mkdir -p "${OUT}/adapter_100ep"

tmux new-session -d -s "${SESSION}" -c "${ROOT}" bash -lc "
set -e
exec > >(tee -a '${OUT}/adapter_100ep/run.log') 2>&1
echo '=== Phase1 1B restart GPU3 max_steps=300: \$(date -Is) ==='
export DISPLAY=:200
export CUDA_VISIBLE_DEVICES=3
export HABITAT_GL_GPU_ID=0
export HEATMAPVLN_PREINIT_GL=0
export HEATMAPVLN_PREINIT_EMPTY_GL=1
export USE_TF=0 TRANSFORMERS_NO_TF=1
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0
export INTERNNAV_MODEL_PATH=\${INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}
python -u scripts/evaluate.py r2r \\
  --config configs/train_config_internnav_8gpu_stage2_wider.yaml \\
  --base_checkpoint checkpoints/stage1-s2_latest.pth \\
  --pano_latent_adapter_checkpoint checkpoints/pano_latent_adapter_goldcoord_2000.pth \\
  --gpu_id 0 --sim_gpu_id 0 \\
  --scenes_dir /dataset/mp3d \\
  --data_path '/workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz' \\
  --episode_list '${EP100}' \\
  --max_steps_per_episode 300 \\
  --auto_stop_distance 3.0 \\
  --max_system2_calls_per_episode 160 \\
  --no-debug_input_trace \\
  --debug_save_input_images 0 \\
  --resume \\
  --output_path '${OUT}/adapter_100ep'
echo '=== Phase1 1B done: \$(date -Is) ==='
"

echo "tmux: ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "log: ${OUT}/adapter_100ep/run.log"
echo "progress: ${OUT}/adapter_100ep/progress.json ($(wc -l < "${OUT}/adapter_100ep/progress.json" 2>/dev/null || echo 0) episodes done)"
