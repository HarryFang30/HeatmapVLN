#!/usr/bin/env bash
# Phase 1 fair baseline: InternNav native (20 ep, GPU0) + adapter distillation (100 ep, GPU3).
set -euo pipefail

ROOT="/workspace/HeatmapVLN"
INTERN="/workspace/InternNav"
OUT="${ROOT}/logs/phase1_baseline"
EP20="${ROOT}/logs/compare_rerun_20ep/episode_list.json"
EP100="${OUT}/episode_list_100.json"
SCENES="/dataset/mp3d"
DATA="/workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"

export DISPLAY="${DISPLAY:-:200}"
export USE_TF=0 TRANSFORMERS_NO_TF=1
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/workspace/InternNav_Model}"
export HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0

mkdir -p "${OUT}/internnav_20ep" "${OUT}/adapter_100ep"
pgrep -a Xvfb >/dev/null || Xvfb :200 -screen 0 1024x768x24 >/tmp/xvfb_200.log 2>&1 &

if [[ ! -f "${EP20}" ]]; then
  echo "Missing ${EP20}; export 20 episodes first."
  exit 1
fi

echo "=== Export first 100 val_unseen episodes (same iterator as EP20) ==="
cd "${ROOT}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/export_episode_list.py \
  --scenes_dir "${SCENES}" \
  --data_path "${DATA}" \
  --count 100 \
  --output "${EP100}"

# Sanity: first 20 of EP100 must match EP20
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/HeatmapVLN")
k20 = [(e["scene_id"], int(e["episode_id"])) for e in json.loads((root/"logs/compare_rerun_20ep/episode_list.json").read_text())["episodes"]]
k100 = [(e["scene_id"], int(e["episode_id"])) for e in json.loads((root/"logs/phase1_baseline/episode_list_100.json").read_text())["episodes"]]
if k100[:20] != k20:
    raise SystemExit(f"EP100 prefix mismatch with EP20\n  ep20[0]={k20[0]}\n  ep100[0]={k100[0]}")
print("OK: episode_list_100[:20] == episode_list_20")
PY

rm -f "${OUT}/internnav_20ep/progress.json" "${OUT}/internnav_20ep/result.json"
rm -f "${OUT}/adapter_100ep/progress.json" "${OUT}/adapter_100ep/result.json"

SESSION_INTERN="${1:-phase1_internnav_20}"
SESSION_ADAPTER="${2:-phase1_adapter_100}"

tmux kill-session -t "${SESSION_INTERN}" 2>/dev/null || true
tmux kill-session -t "${SESSION_ADAPTER}" 2>/dev/null || true

# 1A: InternNav native, same 20 episodes, CUDA:0
tmux new-session -d -s "${SESSION_INTERN}" -c "${INTERN}" bash -lc "
set -e
exec > >(tee -a '${OUT}/internnav_20ep/run.log') 2>&1
echo '=== Phase1 1A InternNav native start: \$(date -Is) ==='
export CUDA_VISIBLE_DEVICES=0
export HABITAT_GL_GPU_ID=0
export USE_TF=0 TRANSFORMERS_NO_TF=1 HEATMAPVLN_REQUIRE_FLASH_ATTN=0
python -u scripts/eval/eval_r2r_val_unseen.py \\
  --model_path '${INTERNNAV_MODEL_PATH}' \\
  --scenes_dir '${SCENES}' \\
  --data_path '${DATA}' \\
  --output_path '${OUT}/internnav_20ep' \\
  --gpu_id 0 \\
  --episode_list '${EP20}' \\
  --max_steps_per_episode 500
echo '=== Phase1 1A done: \$(date -Is) ==='
"

# 1B: distillation route (pano VLM + adapter + frozen NextDiT), 100 episodes, CUDA:3
tmux new-session -d -s "${SESSION_ADAPTER}" -c "${ROOT}" bash -lc "
set -e
exec > >(tee -a '${OUT}/adapter_100ep/run.log') 2>&1
echo '=== Phase1 1B adapter 100ep start: \$(date -Is) ==='
export CUDA_VISIBLE_DEVICES=3
export HABITAT_GL_GPU_ID=0
export USE_TF=0 TRANSFORMERS_NO_TF=1
export HEATMAPVLN_PREINIT_GL=0 HEATMAPVLN_PREINIT_EMPTY_GL=1
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0
export INTERNNAV_MODEL_PATH='${INTERNNAV_MODEL_PATH}'
python -u scripts/evaluate.py r2r \\
  --config configs/train_config_internnav_8gpu_stage2_wider.yaml \\
  --base_checkpoint checkpoints/stage1-s2_latest.pth \\
  --pano_latent_adapter_checkpoint checkpoints/pano_latent_adapter_goldcoord_2000.pth \\
  --gpu_id 0 --sim_gpu_id 0 \\
  --scenes_dir '${SCENES}' \\
  --data_path '${DATA}' \\
  --episode_list '${EP100}' \\
  --auto_stop_distance 3.0 \\
  --max_system2_calls_per_episode 160 \\
  --no-debug_input_trace \\
  --debug_save_input_images 0 \\
  --overwrite_output \\
  --output_path '${OUT}/adapter_100ep'
echo '=== Phase1 1B done: \$(date -Is) ==='
"

echo ""
echo "Phase 1 tmux sessions:"
echo "  1A InternNav 20ep (GPU0): tmux attach -t ${SESSION_INTERN}"
echo "  1B Adapter 100ep (GPU3): tmux attach -t ${SESSION_ADAPTER}"
echo "Logs:"
echo "  ${OUT}/internnav_20ep/run.log"
echo "  ${OUT}/adapter_100ep/run.log"
