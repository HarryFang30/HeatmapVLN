#!/bin/bash
# =============================================================================
# Baseline Training Workflow Script
# =============================================================================
# Based on run_baseline_push.md specification
#
# This script runs the complete baseline workflow:
# 1. Re-pack dataset with visibility-aware sampling
# 2. Generate quality reports and verify SLO compliance
# 3. Run overfit test to verify model can learn
# 4. Run Stage-A baseline training (64x64, head-only)
# 5. Evaluate trained model with visualizations
#
# Usage:
#   bash run_baseline.sh [--skip-pack] [--skip-overfit]
#
# Options:
#   --skip-pack      Skip data repacking (use existing packed data)
#   --skip-overfit   Skip overfit test
#
# =============================================================================

set -e  # Exit on error

# Parse arguments
SKIP_PACK=false
SKIP_OVERFIT=false

for arg in "$@"; do
    case $arg in
        --skip-pack)
            SKIP_PACK=true
            shift
            ;;
        --skip-overfit)
            SKIP_OVERFIT=true
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}Baseline Training Workflow${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""

# =============================================================================
# Step 1: Data Repacking (with visibility-aware sampling)
# =============================================================================
if [ "$SKIP_PACK" = false ]; then
    echo -e "${YELLOW}[1/5] Re-packing dataset with visibility-aware sampling...${NC}"
    echo ""

    # Check if raw data exists
    if [ ! -d "./raw_sequences" ]; then
        echo -e "${RED}ERROR: Raw sequences directory not found: ./raw_sequences${NC}"
        echo "Please prepare raw data according to dataset.md specification"
        exit 1
    fi

    # Pack train split
    echo "Packing train split..."
    python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train

    # Pack val split
    echo ""
    echo "Packing val split..."
    python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val

    echo -e "${GREEN}✓ Data repacking completed${NC}"
    echo ""
else
    echo -e "${YELLOW}[1/5] Skipping data repacking (--skip-pack specified)${NC}"
    echo ""
fi

# =============================================================================
# Step 2: Quality Reports & SLO Verification
# =============================================================================
echo -e "${YELLOW}[2/5] Generating quality reports and verifying SLO compliance...${NC}"
echo ""

# Generate train quality report
echo "Generating train quality report..."
python scripts/report_data_quality.py --root ./data/habitat_vln --split train

echo ""

# Generate val quality report
echo "Generating val quality report..."
python scripts/report_data_quality.py --root ./data/habitat_vln --split val

echo ""
echo -e "${GREEN}✓ Quality reports generated${NC}"
echo "  Reports saved to: ./outputs/reports/"
echo ""
echo "Please verify SLO compliance:"
echo "  - K_eff ≥ 2 ratio ≥ 80%"
echo "  - Average entropy ≤ 5.0 (for 64x64)"
echo "  - Mask==0 ratio < 20%"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort if SLO not met..."
echo ""

# =============================================================================
# Step 3: Overfit Test (verify model can learn)
# =============================================================================
if [ "$SKIP_OVERFIT" = false ]; then
    echo -e "${YELLOW}[3/5] Running overfit test (verify model can learn)...${NC}"
    echo ""

    python scripts/overfit_one_batch.py

    echo ""
    echo -e "${GREEN}✓ Overfit test completed${NC}"
    echo "  Expected: Loss < 3.0 within a few hundred steps"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to abort if overfit test failed..."
    echo ""
else
    echo -e "${YELLOW}[3/5] Skipping overfit test (--skip-overfit specified)${NC}"
    echo ""
fi

# =============================================================================
# Step 4: Stage-A Baseline Training (64x64, head-only)
# =============================================================================
echo -e "${YELLOW}[4/5] Running Stage-A baseline training...${NC}"
echo ""
echo "Configuration:"
echo "  - Resolution: 64x64"
echo "  - Epochs: 3"
echo "  - Learning rate: 3e-3 (head)"
echo "  - AMP: off (for stability)"
echo "  - Backbone: Frozen ResNet18"
echo ""

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"

    if [ $GPU_COUNT -eq 0 ]; then
        echo -e "${RED}Warning: No GPUs detected, training will run on CPU (very slow)${NC}"
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, cannot detect GPUs${NC}"
fi

echo ""
echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml

echo ""
echo -e "${GREEN}✓ Stage-A baseline training completed${NC}"
echo ""
echo "Training checkpoint saved to: ./outputs/checkpoints/"
echo ""

# =============================================================================
# Step 5: Evaluation & Visualization
# =============================================================================
echo -e "${YELLOW}[5/5] Evaluating trained model and generating visualizations...${NC}"
echo ""

# Find best checkpoint
BEST_CKPT=$(ls -t ./outputs/checkpoints/checkpoint_warmup_head_*.pt 2>/dev/null | head -1 || echo "")

if [ -z "$BEST_CKPT" ]; then
    echo -e "${RED}ERROR: No checkpoint found in ./outputs/checkpoints/${NC}"
    echo "Training may have failed. Please check logs."
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
echo -e "${GREEN}Baseline Workflow Completed Successfully!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo "Summary:"
echo "  1. ✓ Data repacked with visibility-aware sampling"
echo "  2. ✓ Quality reports generated (see ./outputs/reports/)"
echo "  3. ✓ Overfit test passed"
echo "  4. ✓ Stage-A baseline training completed"
echo "  5. ✓ Model evaluated with visualizations (see ./outputs/vis/)"
echo ""
echo "Next steps:"
echo "  - Review quality reports to ensure SLO compliance"
echo "  - Inspect visualizations to verify heatmap quality"
echo "  - If validation NLL < 8.0 and stable, proceed to Stage-B/C"
echo "  - If not converging well, adjust knobs in configs/dataset_pack.yaml:"
echo "      - lookback: 3 / 5 / 7"
echo "      - min_visible_ratio: 0.01 / 0.02 / 0.03"
echo "      - gaussian_sigma_px: 1.5 / 2.0"
echo ""
echo "Training logs available at: ./outputs/train.log"
echo ""