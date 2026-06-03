#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# 3-way ablation: InternNav native vs HeatmapVLN (no adapter) vs HeatmapVLN (with adapter)
# ═══════════════════════════════════════════════════════════════════════
# Runs all 3 experiments IN PARALLEL on 3 GPUs (0, 1, 2).
# All experiments evaluate R2R val_unseen in the SAME order, so results
# are directly comparable.
# ═══════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

# ── Common environment ─────────────────────────────────────────────────
export DISPLAY=:400
if ! pgrep -f "Xvfb :400" >/dev/null 2>&1; then
  Xvfb :400 -screen 0 1024x768x24 >/tmp/xvfb_400.log 2>&1 &
  sleep 1
  echo "Started Xvfb :400"
fi

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export HEATMAPVLN_PREINIT_GL=0
export HEATMAPVLN_PREINIT_EMPTY_GL=1
export HABITAT_GL_GPU_ID=0
export HEATMAPVLN_REQUIRE_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export INTERNNAV_MODEL_PATH=/workspace/InternNav_Model
export INTERNNAV_BACKBONE=/workspace/InternNav_Model

# ── Paths ──────────────────────────────────────────────────────────────
REPO=/workspace/HeatmapVLN
INTERNNAV_REPO=/workspace/InternNav
BASE_CKPT="$REPO/checkpoints/stage1-s2_latest.pth"
ADAPTER_CKPT="$REPO/checkpoints/stage2_latest.pth"
OUT_BASE="$REPO/logs/eval_ablation_$(date +%Y%m%d_%H%M)"
N_EPISODES=100

# R2R data (shared across all experiments — MUST be identical for fair comparison)
SCENES_DIR=data/scene_data/mp3d_ce
DATA_PATH="data/vln_ce/raw_data/r2r/{split}/{split}.json.gz"

echo "========== Ablation output: $OUT_BASE =========="
mkdir -p "$OUT_BASE"

# ═══════════════════════════════════════════════════════════════════════
# Exp 1: InternNav NATIVE baseline (single front-camera, no panorama)
#   Model: InternVLAN1ForCausalLM (original InternNav Qwen2-VL)
#   Input: front RGB 640×480 → resize 384×384 + lookdown 224×224
#   Prompt: single-turn + legacy conjunction + ↓ lookdown protocol
# ═══════════════════════════════════════════════════════════════════════
echo "[1/3] Launching InternNav native baseline on GPU 0 ..."
(
  cd "$INTERNNAV_REPO"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval/eval_r2r_val_unseen.py \
    --model_path "$INTERNNAV_MODEL_PATH" \
    --scenes_dir "$SCENES_DIR" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT_BASE/internnav_native" \
    --gpu_id 0 \
    --resize_w 384 --resize_h 384 \
    --num_history 8 \
    --max_steps_per_episode 500 \
    2>&1 | tee "$OUT_BASE/internnav_native.log"
  echo "[1/3] InternNav native DONE"
) &

# ═══════════════════════════════════════════════════════════════════════
# Exp 2: HeatmapVLN panoramic WITHOUT adapter
#   Model: Qwen2.5-VL Stage1-S2 LoRA + cond_projector (frozen)
#   Input: 4-view panorama 256×256 (front/right/back/left)
#   Prompt: structured pano (view + pixel format)
# ═══════════════════════════════════════════════════════════════════════
echo "[2/3] Launching HeatmapVLN no-adapter on GPU 1 ..."
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=1 python -u scripts/evaluation/r2r_val_unseen.py \
    --config configs/train_system2_panoramic_sft_8gpu.yaml \
    --base_checkpoint "$BASE_CKPT" \
    --output_path "$OUT_BASE/no_adapter" \
    --gpu_id 0 \
    --scenes_dir "$SCENES_DIR" \
    --data_path "$DATA_PATH" \
    --max_episodes "$N_EPISODES" \
    --trajectory_selection mean \
    --save_trajectory_steps \
    2>&1 | tee "$OUT_BASE/no_adapter.log"
  echo "[2/3] HeatmapVLN no-adapter DONE"
) &

# ═══════════════════════════════════════════════════════════════════════
# Exp 3: HeatmapVLN panoramic WITH adapter
#   Model: Qwen2.5-VL Stage1-S2 LoRA + GeometryAwarePanoToNextDiTAdapter + NextDiT
#   Input: 4-view panorama 256×256 (same as Exp 2)
#   Key difference: cond_projector replaced by trained adapter
# ═══════════════════════════════════════════════════════════════════════
echo "[3/3] Launching HeatmapVLN with-adapter on GPU 2 ..."
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=2 python -u scripts/evaluation/r2r_val_unseen.py \
    --config configs/train_system2_panoramic_sft_8gpu.yaml \
    --base_checkpoint "$BASE_CKPT" \
    --pano_latent_adapter_checkpoint "$ADAPTER_CKPT" \
    --output_path "$OUT_BASE/with_adapter" \
    --gpu_id 0 \
    --scenes_dir "$SCENES_DIR" \
    --data_path "$DATA_PATH" \
    --max_episodes "$N_EPISODES" \
    --trajectory_selection mean \
    --save_trajectory_steps \
    2>&1 | tee "$OUT_BASE/with_adapter.log"
  echo "[3/3] HeatmapVLN with-adapter DONE"
) &

echo ""
echo "All 3 experiments launched. Waiting for completion ..."
echo "  GPU 0: InternNav native"
echo "  GPU 1: HeatmapVLN no-adapter"
echo "  GPU 2: HeatmapVLN with-adapter"
echo ""

wait

echo ""
echo "========== All 3 experiments DONE =========="
echo "Results in: $OUT_BASE"
echo ""
echo "Compare:"
echo "  cat $OUT_BASE/internnav_native/result.json"
echo "  cat $OUT_BASE/no_adapter/result.json"
echo "  cat $OUT_BASE/with_adapter/result.json"
echo ""
echo "Visualize (download to local):"
echo "  python $REPO/scripts/visualization/generate_trajectory_html.py --input-dir $OUT_BASE/no_adapter --output-dir $OUT_BASE/no_adapter/html"
echo "  python $REPO/scripts/visualization/generate_trajectory_html.py --input-dir $OUT_BASE/with_adapter --output-dir $OUT_BASE/with_adapter/html"
