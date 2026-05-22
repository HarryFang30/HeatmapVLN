#!/usr/bin/env bash
# Compare HeatmapVLN vs InternNav on the same 20 val_unseen episodes.
set -euo pipefail

ROOT="/workspace/HeatmapVLN"
INTERN="/workspace/InternNav"
export DISPLAY="${DISPLAY:-:200}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}"
export HEATMAPVLN_PREINIT_GL=0
export HEATMAPVLN_PREINIT_EMPTY_GL=1
export HABITAT_GL_GPU_ID=0
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0

SCENES_DIR="/dataset/mp3d"
DATA_PATH="/workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"
OUT_ROOT="${ROOT}/logs/compare_rerun_20ep"
EP_LIST="${OUT_ROOT}/episode_list.json"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}"
pgrep -a Xvfb >/dev/null || Xvfb :200 -screen 0 1024x768x24 >/tmp/xvfb_200.log 2>&1 &

echo "=== Export fixed 20-episode list ==="
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/evaluation/export_episode_list.py \
  --scenes_dir "${SCENES_DIR}" \
  --data_path "${DATA_PATH}" \
  --count 20 \
  --output "${EP_LIST}"

rm -f "${OUT_ROOT}/internnav/progress.json" "${OUT_ROOT}/internnav/result.json"
rm -f "${OUT_ROOT}/heatmapvln/progress.json" "${OUT_ROOT}/heatmapvln/result.json"
mkdir -p "${OUT_ROOT}/internnav" "${OUT_ROOT}/heatmapvln"

echo "=== InternNav (20 fixed episodes) ==="
cd "${INTERN}"
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval/eval_r2r_val_unseen.py \
  --model_path "${INTERNNAV_MODEL_PATH}" \
  --scenes_dir "${SCENES_DIR}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUT_ROOT}/internnav" \
  --gpu_id 0 \
  --episode_list "${EP_LIST}" \
  --max_steps_per_episode 500 \
  2>&1 | tee "${OUT_ROOT}/internnav/run.log"

echo "=== HeatmapVLN Stage2 (same 20 episodes) ==="
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/evaluate.py r2r \
  --config configs/train_config_internnav_8gpu.yaml \
  --base_checkpoint checkpoints/stage1-s2_latest.pth \
  --checkpoint checkpoints/stage2_latest.pth \
  --gpu_id 0 --sim_gpu_id 0 \
  --scenes_dir "${SCENES_DIR}" \
  --data_path "${DATA_PATH}" \
  --episode_list "${EP_LIST}" \
  --overwrite_output \
  --output_path "${OUT_ROOT}/heatmapvln" \
  2>&1 | tee "${OUT_ROOT}/heatmapvln/run.log"

echo "=== Summary ==="
python3 "${ROOT}/scripts/evaluation/summarize_compare.py" \
  --episode_list "${EP_LIST}" \
  --internnav "${OUT_ROOT}/internnav/progress.json" \
  --heatmapvln "${OUT_ROOT}/heatmapvln/progress.json"
