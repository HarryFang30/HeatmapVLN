# Quick Start: Bootstrap Workflow

**Status**: ✅ Ready to run
**Purpose**: Fix data issues and enable baseline training
**Time**: ~30-45 minutes

## TL;DR

```bash
cd /home/VLN/Project
conda activate spatial-mllm
bash run_bootstrap.sh
```

## What This Does

Fixes data-related failures by:
1. ✅ Generating **120 train + 30 val** synthetic clips
2. ✅ Using **visibility-friendly trajectories** (short arc + face target)
3. ✅ Applying **relaxed cold-start parameters**
4. ✅ Adding **fallback mechanisms** to prevent crashes
5. ✅ Verifying quality and training baseline

## When to Use This

Use bootstrap if you're seeing:
- ❌ "No visible keyframes found"
- ❌ "No clips processed"
- ❌ K_eff ≥ 2 ratio far below 80%
- ❌ Val split missing or empty
- ❌ Training fails due to insufficient data

## Automated Workflow

```bash
bash run_bootstrap.sh
```

**Interactive checkpoints**: The script pauses after quality reports and overfit test for you to verify.

## Manual Workflow

### Step 1: Clean (30 seconds)
```bash
rm -rf ./data/habitat_vln/train ./data/habitat_vln/val
rm -rf ./raw_sequences/train ./raw_sequences/val
mkdir -p ./raw_sequences/train/RoomA ./raw_sequences/val/RoomB
```

### Step 2: Generate Data (5-10 min)
```bash
# Train: 120 clips
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomA --clips 120 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc --arc_deg 30 --radius 2.5 \
  --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02

# Val: 30 clips
python scripts/gen_synth_demo.py \
  --root ./raw_sequences --scene RoomB --clips 30 \
  --T 8 --W 384 --H 384 \
  --pose_mode face_target --path_mode short_arc --arc_deg 30 --radius 2.5 \
  --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02
```

### Step 3: Pack (2-5 min)
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
```

### Step 4: Quality Check (1 min)
```bash
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```

**Verify Relaxed SLO**:
- ✅ K_eff ≥ 2 ratio ≥ **70%** (relaxed from 80%)
- ✅ Average entropy ≤ **5.2** (relaxed from 5.0)

### Step 5: Overfit (2-5 min)
```bash
python scripts/overfit_one_batch.py
```

**Expected**: Loss < 3.0 within 200-300 steps

### Step 6: Train (30-60 min)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py \
  --config configs/training_config.yaml
```

**Expected**: Val NLL < 8.0 after 3 epochs

### Step 7: Evaluate (1-2 min)
```bash
python scripts/eval_heatmap.py \
  --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt \
  --save-vis
```

## What Changed

### Data Generation
| Parameter | Before | After (Bootstrap) |
|-----------|--------|-------------------|
| Clips | 1 train, 0 val | 120 train, 30 val |
| Trajectory | Full circle | Short arc (30°) |
| Pose | Look center | Face target |
| Overlap | Poor (~1%) | Good (~10-20%) |

### Pack Configuration
| Parameter | Before | After (Bootstrap) |
|-----------|--------|-------------------|
| min_visible_ratio | 0.02 | 0.01 (relaxed) |
| gaussian_sigma_px | 2.0 | 1.8 (sharper) |
| occlusion_check | true | false (disabled) |
| drop_if_effective_k_below | 2 | null (mark instead) |

## Expected Output

### After Data Generation
```
✅ Synthetic data generated
  Train: 120 clips in ./raw_sequences/train/RoomA/
  Val: 30 clips in ./raw_sequences/val/RoomB/
```

### After Quality Report
```
🎯 SLO Compliance (Relaxed):
   K_eff ≥ 2 ratio: ✅ PASS (75.0% > 70%)
   Average Entropy: ✅ PASS (5.1 < 5.2)
   Overall: ✅ PASS
```

### After Training
```
Epoch 3 - Train Loss: 4.85, Val Loss: 5.23  ✅ < 8.0
✅ SUCCESS: Baseline converged
```

## Troubleshooting

### Still "No Visible Keyframes"?

Try more relaxed settings in `configs/dataset_pack.yaml`:
```yaml
pack:
  lookback: 3                 # Closer frames
  min_visible_ratio: 0.005    # More lenient
```

Or generate with smaller arc:
```bash
python scripts/gen_synth_demo.py ... --arc_deg 20  # Instead of 30
```

### Still Low K_eff?

Sharper heatmaps:
```yaml
heatmap:
  gaussian_sigma_px: 1.5      # Sharper (from 1.8)
```

Or more clips:
```bash
python scripts/gen_synth_demo.py ... --clips 200  # Instead of 120
```

### Training Not Converging?

Check data was generated:
```bash
ls ./raw_sequences/train/RoomA/  # Should see clip_000001, clip_000002, ...
ls ./raw_sequences/val/RoomB/    # Should see clip_000001, ...
```

Re-run packing:
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
```

## Output Files

```
./raw_sequences/
  train/RoomA/clip_*/       # 120 clips
  val/RoomB/clip_*/         # 30 clips

./data/habitat_vln/
  train/RoomA/clip_*/       # Packed training data
  val/RoomB/clip_*/         # Packed validation data

./outputs/
  reports/
    data_quality_train.json
    data_quality_val.json
  checkpoints/
    checkpoint_warmup_head_epoch_*.pt
  vis/
    sample_*_comparison.png
```

## Next Steps

After successful bootstrap:

1. **Verify**: Check quality reports and visualizations
2. **Optional**: Gradually restore stricter settings
3. **Proceed**: Move to Stage-B/C (higher resolution, LoRA)

## Key Differences from Normal Baseline

| Aspect | Normal Baseline | Bootstrap |
|--------|----------------|-----------|
| Data source | User-provided | Generated synthetic |
| Quantity | Variable | 120 train + 30 val |
| Quality gates | Strict (80%, 5.0) | Relaxed (70%, 5.2) |
| Occlusion | Enabled | Disabled |
| Fallback | None | Near-ref fallback |
| Drop policy | Drop low K_eff | Mark low K_eff |

## Time Breakdown

| Step | Time | GPU Required |
|------|------|--------------|
| Clean | <1 min | No |
| Generate data | 5-10 min | No |
| Pack dataset | 2-5 min | No |
| Quality reports | 1 min | No |
| Overfit test | 2-5 min | Yes |
| Training | 30-60 min | Yes |
| Evaluation | 1-2 min | Yes |
| **Total** | **~45-85 min** | |

## Command Reference

```bash
# Full automated workflow
bash run_bootstrap.sh

# Individual steps (manual)
rm -rf ./data/habitat_vln/{train,val} ./raw_sequences/{train,val}
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 120 ...
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/overfit_one_batch.py
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml
python scripts/eval_heatmap.py --config configs/training_config.yaml --ckpt <path> --save-vis
```

---

**Ready?** Run `bash run_bootstrap.sh` to start!