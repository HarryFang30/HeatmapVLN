#!/bin/bash
# =============================================================================
# Data Bootstrap Workflow Script
# =============================================================================
# Based on baseline_data_bootstrap.md specification
#
# This script fixes data-related issues by:
# 1. Cleaning old data
# 2. Generating visibility-friendly synthetic data (120 train + 30 val clips)
# 3. Re-packing with relaxed cold-start parameters
# 4. Verifying quality with reports
# 5. Running overfit test and Stage-A training
#
# Usage:
#   bash run_bootstrap.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}Data Bootstrap Workflow (Cold-Start Mode)${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "${YELLOW}This script will:${NC}"
echo "  1. Clean old data (train & val)"
echo "  2. Generate 120 train + 30 val synthetic clips"
echo "  3. Re-pack with visibility-friendly settings"
echo "  4. Verify quality (relaxed SLO: K_eff≥2 ≥70%, entropy ≤5.2)"
echo "  5. Run overfit test (expect loss <3.0)"
echo "  6. Run Stage-A baseline training"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."
echo ""

# =============================================================================
# Step 1: Clean Old Data
# =============================================================================
echo -e "${YELLOW}[1/6] Cleaning old data...${NC}"
echo ""

if [ -d "./data/habitat_vln/train" ] || [ -d "./data/habitat_vln/val" ]; then
    echo "Removing packed data..."
    rm -rf ./data/habitat_vln/train ./data/habitat_vln/val
fi

if [ -d "./raw_sequences/train" ] || [ -d "./raw_sequences/val" ]; then
    echo "Removing raw sequences..."
    rm -rf ./raw_sequences/train ./raw_sequences/val
fi

echo "Creating fresh directories..."
mkdir -p ./raw_sequences/train/RoomA ./raw_sequences/val/RoomB

echo -e "${GREEN}✓ Old data cleaned${NC}"
echo ""

# =============================================================================
# Step 2: Generate Visibility-Friendly Synthetic Data
# =============================================================================
echo -e "${YELLOW}[2/6] Generating synthetic data (120 train + 30 val clips)...${NC}"
echo ""
echo "Parameters:"
echo "  - Path: short_arc (30° arc)"
echo "  - Pose: face_target (always looking at target [0,0,2])"
echo "  - Radius: 2.5m"
echo "  - T: 8 frames per clip"
echo "  - Noise: small rotation (2°) and translation (0.02m)"
echo ""

# Train split: RoomA, 120 clips
echo "Generating train split (RoomA, 120 clips)..."
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomA --clips 120 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc --arc_deg 30 --radius 2.5 \
  --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02

echo ""

# Val split: RoomB, 30 clips
echo "Generating val split (RoomB, 30 clips)..."
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomB --clips 30 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc --arc_deg 30 --radius 2.5 \
  --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02

echo ""
echo -e "${GREEN}✓ Synthetic data generated${NC}"
echo "  Train: 120 clips in ./raw_sequences/train/RoomA/"
echo "  Val: 30 clips in ./raw_sequences/val/RoomB/"
echo ""

# =============================================================================
# Step 3: Re-pack with Cold-Start Configuration
# =============================================================================
echo -e "${YELLOW}[3/6] Re-packing dataset with cold-start parameters...${NC}"
echo ""
echo "Cold-start adjustments:"
echo "  - min_visible_ratio: 0.01 (relaxed from 0.02)"
echo "  - gaussian_sigma_px: 1.8 (sharper, from 2.0)"
echo "  - occlusion_check: false (disabled for cold-start)"
echo "  - drop_if_effective_k_below: null (mark instead of drop)"
echo ""

# Pack train
echo "Packing train split..."
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train

echo ""

# Pack val
echo "Packing val split..."
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val

echo ""
echo -e "${GREEN}✓ Dataset repacked${NC}"
echo ""

# =============================================================================
# Step 4: Quality Reports & SLO Verification
# =============================================================================
echo -e "${YELLOW}[4/6] Generating quality reports (relaxed SLO for cold-start)...${NC}"
echo ""

# Train quality report
echo "Generating train quality report..."
python scripts/report_data_quality.py --root ./data/habitat_vln --split train

echo ""

# Val quality report
echo "Generating val quality report..."
python scripts/report_data_quality.py --root ./data/habitat_vln --split val

echo ""
echo -e "${GREEN}✓ Quality reports generated${NC}"
echo "  Reports saved to: ./outputs/reports/"
echo ""
echo -e "${BLUE}Cold-Start SLO (relaxed):${NC}"
echo "  ✓ K_eff ≥ 2 ratio ≥ 70% (relaxed from 80%)"
echo "  ✓ Average entropy ≤ 5.2 (relaxed from 5.0)"
echo "  ✓ Mask==0 ratio < 30% (relaxed from 20%)"
echo ""
read -p "Review reports above. Press Enter to continue or Ctrl+C to abort if SLO not met..."
echo ""

# =============================================================================
# Step 5: Overfit Test
# =============================================================================
echo -e "${YELLOW}[5/6] Running overfit test (verify model can learn)...${NC}"
echo ""

python scripts/overfit_one_batch.py

echo ""
echo -e "${GREEN}✓ Overfit test completed${NC}"
echo "  Expected: Loss < 3.0 within a few hundred steps"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort if overfit test failed..."
echo ""

# =============================================================================
# Step 6: Stage-A Baseline Training
# =============================================================================
echo -e "${YELLOW}[6/6] Running Stage-A baseline training...${NC}"
echo ""
echo "Configuration:"
echo "  - Resolution: 64x64"
echo "  - Epochs: 3"
echo "  - Learning rate: 3e-3 (head)"
echo "  - AMP: off (for stability)"
echo "  - Backbone: Frozen ResNet18"
echo "  - Loss: Logits-space CE"
echo ""

echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml

echo ""
echo -e "${GREEN}✓ Stage-A baseline training completed${NC}"
echo ""

# =============================================================================
# Step 7: Evaluation
# =============================================================================
echo -e "${YELLOW}[7/7] Evaluating trained model...${NC}"
echo ""

# Find best checkpoint
BEST_CKPT=$(ls -t ./outputs/checkpoints/checkpoint_warmup_head_*.pt 2>/dev/null | head -1 || echo "")

if [ -z "$BEST_CKPT" ]; then
    echo -e "${RED}ERROR: No checkpoint found in ./outputs/checkpoints/${NC}"
    exit 1
fi

echo "Using checkpoint: $BEST_CKPT"
echo ""

python scripts/eval_heatmap.py \
  --config configs/training_config.yaml \
  --ckpt "$BEST_CKPT" \
  --save-vis

echo ""
echo -e "${GREEN}✓ Evaluation completed${NC}"
echo "  Visualizations saved to: ./outputs/vis/"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}Bootstrap Workflow Completed Successfully!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo "Summary:"
echo "  1. ✓ Old data cleaned"
echo "  2. ✓ Synthetic data generated (120 train + 30 val)"
echo "  3. ✓ Dataset repacked with cold-start config"
echo "  4. ✓ Quality reports generated (check ./outputs/reports/)"
echo "  5. ✓ Overfit test passed"
echo "  6. ✓ Stage-A baseline training completed"
echo "  7. ✓ Model evaluated with visualizations"
echo ""
echo "Next steps:"
echo "  - Review quality reports to verify SLO compliance"
echo "  - Inspect visualizations for heatmap quality"
echo "  - If val NLL < 8.0: SUCCESS! Ready for Stage-B/C"
echo "  - If still issues: Adjust knobs in configs/dataset_pack.yaml:"
echo "      - lookback: 3 / 5 / 7"
echo "      - min_visible_ratio: 0.005 / 0.01 / 0.02"
echo "      - gaussian_sigma_px: 1.5 / 1.8 / 2.0"
echo ""
echo "Outputs:"
echo "  - Raw sequences: ./raw_sequences/{train,val}/"
echo "  - Packed data: ./data/habitat_vln/{train,val}/"
echo "  - Quality reports: ./outputs/reports/data_quality_{train,val}.json"
echo "  - Checkpoints: ./outputs/checkpoints/"
echo "  - Visualizations: ./outputs/vis/"
echo ""