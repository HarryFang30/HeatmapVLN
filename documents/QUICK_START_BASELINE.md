# Quick Start: Baseline Training

**Status**: ✅ Ready to run
**Time**: ~1-2 hours (depending on data size)

## TL;DR

```bash
cd /home/VLN/Project
conda activate spatial-mllm
bash run_baseline.sh
```

## Prerequisites

### 1. Raw Data Prepared

Your raw sequences should be in this structure:
```
./raw_sequences/
  train/
    RoomA/
      clip_000001/
        rgb/           # PNG images
        depth/         # NPY depth maps
        poses.json     # 4x4 camera-to-world transforms
        intrinsics.json  # Camera intrinsics
      clip_000002/
        ...
  val/
    RoomB/
      clip_000001/
        ...
```

**Don't have data?** See `/Project/dataset.md` for data collection guidelines.

### 2. Environment Ready

```bash
conda activate spatial-mllm
cd /home/VLN/Project
```

## Automated Workflow (Recommended)

```bash
bash run_baseline.sh
```

This will automatically:
1. ✅ Re-pack dataset with visibility-aware sampling
2. ✅ Generate quality reports and verify SLO
3. ✅ Run overfit test (verify model can learn)
4. ✅ Train Stage-A baseline (64×64, 3 epochs)
5. ✅ Evaluate model with visualizations

**Interactive checkpoints**: The script will pause after quality reports and overfit test for you to verify results.

## Manual Workflow (For Fine Control)

### Step 1: Pack Data (5-10 min)
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
```

### Step 2: Quality Check (1 min)
```bash
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```

**Verify SLO**:
- ✅ K_eff ≥ 2 ratio ≥ 80%
- ✅ Average entropy ≤ 5.0
- ✅ Overall: PASS

### Step 3: Overfit Test (2-5 min)
```bash
python scripts/overfit_one_batch.py
```

**Expected**: Loss drops from ~8.3 → <3.0 within 200-300 steps

### Step 4: Train Baseline (30-60 min)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml
```

**Expected**: Val NLL < 8.0 after 3 epochs

### Step 5: Evaluate (1-2 min)
```bash
python scripts/eval_heatmap.py \
  --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt \
  --save-vis
```

## What to Expect

### Quality Reports
```
🎯 SLO Compliance:
   K_eff ≥ 2 ratio: ✅ PASS (95.0% > 80%)
   Average Entropy: ✅ PASS (4.523 < 5.0)
   Overall: ✅ PASS
```

### Overfit Test
```
Step 200 | Loss: 2.456 | ✅ < 3.0
✅ SUCCESS: Model can learn
```

### Training
```
Epoch 3 - Train Loss: 5.123, Val Loss: 5.012  ✅ < 8.0
✅ SUCCESS: Baseline converged
```

## Troubleshooting

### SLO Not Met?
Adjust in `configs/dataset_pack.yaml`:
```yaml
pack:
  lookback: 3              # Try: 3 or 7
  min_visible_ratio: 0.01  # Try: 0.01 or 0.03

heatmap:
  gaussian_sigma_px: 1.5   # Try: 1.5 (sharper)
```
Then re-run Step 1.

### Overfit Test Fails?
Check data quality → go back to Step 1.

### Training Not Converging?
Reduce learning rate:
```yaml
optim:
  head_lr: 1.5e-3  # Instead of 3e-3
```

## Output Files

After completion, you'll find:

```
./outputs/
  reports/
    data_quality_train.json      # Quality metrics
    data_quality_val.json
  checkpoints/
    checkpoint_warmup_head_epoch_1.pt
    checkpoint_warmup_head_epoch_2.pt
    checkpoint_warmup_head_epoch_3.pt
  vis/
    sample_000_comparison.png    # GT vs Pred heatmaps
    sample_001_comparison.png
    ...
```

## Next Steps

Once baseline succeeds (val NLL < 8.0):

1. **Review**: Check quality reports and visualizations
2. **Optional**: Fine-tune baseline (2-3 more epochs if needed)
3. **Future**: Proceed to Stage-B/C (higher resolution, LoRA fine-tuning)

## Need Help?

- **Full documentation**: See `BASELINE_IMPLEMENTATION_SUMMARY.md`
- **Data format**: See `dataset.md`
- **Training details**: See `run_training.md`
- **Quality improvements**: See `data_quality_push.md`

## Quick Command Reference

```bash
# Full automated workflow
bash run_baseline.sh

# Skip repacking (if already done)
bash run_baseline.sh --skip-pack

# Skip overfit test (if already verified)
bash run_baseline.sh --skip-overfit

# Manual steps
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/overfit_one_batch.py
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml
python scripts/eval_heatmap.py --config configs/training_config.yaml --ckpt <path> --save-vis
```

## Time Estimates

| Step | Time | GPU Required |
|------|------|--------------|
| Data packing | 5-10 min | No |
| Quality reports | 1 min | No |
| Overfit test | 2-5 min | Yes |
| Training (3 epochs) | 30-60 min | Yes |
| Evaluation | 1-2 min | Yes |
| **Total** | **~45-80 min** | |

GPU recommended: Any CUDA GPU with 8GB+ VRAM (e.g., RTX 2080, V100, A100)

---

**Ready?** Run `bash run_baseline.sh` to start!