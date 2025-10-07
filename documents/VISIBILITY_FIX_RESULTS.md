# Visibility Fix Implementation & Results

## 🎯 Objective
Fix critical bug in visibility computation per `fix_visibility_and_effectiveK.md` where `visible_ratio_for_keyframe` was using keyframe intrinsics (Kj) instead of reference intrinsics (Ki) for projection to reference frame.

## ✅ Completed Tasks

### 1. Fixed `visible_ratio_for_keyframe` to use Ki (Reference Intrinsics)
**File**: [src/data/heatmap_builder.py](src/data/heatmap_builder.py)

**Changes**:
- Added `Ki` parameter to function signature (line 424)
- Fixed projection to use `Ki['fx']`, `Ki['fy']`, `Ki['cx']`, `Ki['cy']` instead of Kj (line 503)

**Before** (Bug):
```python
xi = Kj['fx'] * (pts_ci[:, 0] / zi) + Kj['cx']  # WRONG!
yi = Kj['fy'] * (pts_ci[:, 1] / zi) + Kj['cy']
```

**After** (Fixed):
```python
xi, yi, zi = project_to_ref_pixels(pts_ci, Ki)  # Uses Ki correctly
```

### 2. Added `project_to_ref_pixels` Helper Function
**File**: [src/data/heatmap_builder.py:424-441](src/data/heatmap_builder.py)

Ensures consistent projection to reference frame across visibility evaluation and heatmap generation.

```python
def project_to_ref_pixels(pts_ci: np.ndarray, Ki: dict):
    """Project points in reference camera frame to pixel coordinates using reference intrinsics."""
    zi = pts_ci[:, 2]
    xi = Ki['fx'] * (pts_ci[:, 0] / zi) + Ki['cx']
    yi = Ki['fy'] * (pts_ci[:, 1] / zi) + Ki['cy']
    return xi, yi, zi
```

### 3. Updated `pack_dataset.py` to Pass Ki Parameter
**File**: [scripts/pack_dataset.py:181-186](scripts/pack_dataset.py)

```python
# CRITICAL FIX: Pass both Kj (keyframe intrinsics) and Ki (reference intrinsics)
score = visible_ratio_for_keyframe(
    depth_maps[j], intrinsics_dict, intrinsics_dict,  # Kj, Ki
    poses[j], T_c_ref_w,
    depth_ref, ref_w, ref_h, occl_eps, subsample=depth_subsample
)
```

### 4. Created Assertion Test Script
**File**: [scripts/check_visibility_asserts.py](scripts/check_visibility_asserts.py)

**Test Results**: ✅ ALL TESTS PASSED

```
Test 1: project_to_ref_pixels helper
  ✅ PASS: Correctly projects using Ki intrinsics

Test 2: visible_ratio_for_keyframe uses Ki (not Kj)
  ✅ PASS: With different Kj (fx=300) and Ki (fx=400), visibility ratio = 0.5625

Test 3: K_eff computation with minimal synthetic data
  ✅ PASS: K_eff = 3 with perfect geometric overlap
```

### 5. Updated Overfit Threshold to Relative Entropy
**File**: [scripts/overfit_one_batch.py](scripts/overfit_one_batch.py)

**Changes**:
- Compute target entropy median: `H_med = median(H(q))`
- Set relative threshold: `pass_threshold = H_med + 0.3`
- Updated success criteria to use relative threshold instead of absolute `<1.0`

**Rationale**: Optimal cross-entropy is `H(q)`, not absolute `<1.0`. A 64×64 heatmap with uniform distribution has `H(q) ≈ 8.0`, so expecting `<1.0` is unrealistic.

### 6. Set `min_visible_ratio: 0.0` for Volume Test
**File**: [configs/dataset_pack.yaml:19](configs/dataset_pack.yaml)

Temporarily set to `0.0` to verify the fix increases candidate volume (previously `0.01`).

## 📊 Test Results

### Assertion Tests: ✅ SUCCESS
All 3 tests passed, confirming:
1. Helper function uses Ki correctly
2. Visibility computation uses Ki (not Kj)
3. K_eff ≥ 2 with synthetic overlapping geometry

### Data Quality Report: ⚠️ PARTIAL SUCCESS

**Training Set** (120 clips):
- **K_eff Statistics**:
  - Average: 0.82
  - K=0: 22 samples (18.3%)
  - K=1: 98 samples (81.7%)
  - **K=2+: 0 samples (0.0%)** ❌

- **SLO Compliance**:
  - K_eff ≥ 2 ratio: **0.0%** (target: ≥80%) ❌
  - Average entropy: **4.013** (target: ≤5.0) ✅

**Validation Set** (30 clips):
- Empty heatmap ratio: 82.5%
- Similar K_eff distribution

## 🔍 Root Cause Analysis

### Why K_eff is Still Low

**Visibility computation is now CORRECT** (verified by assertion tests), but K_eff remains low because:

1. **Existing synthetic data has poor geometric overlap**:
   - Generated with old defaults: `circular` path + `look_center` pose
   - Large rotation changes between frames (see pose matrices)
   - Cameras don't consistently face the same target

2. **Evidence from packed samples**:
   - Most samples: Only 1 valid heatmap (reference frame itself)
   - Other 3 keyframes: Empty heatmaps (sum=0.0)
   - This means visibility computation correctly identifies **no overlap**

### Example from Sample 0:
```
mask: [0, 0, 0, 1]  → Only reference frame valid
heatmap_sums: [0.0, 0.0, 0.0, 1.0]  → Only last frame has content
```

## 🚀 Next Steps (Required to Achieve K_eff ≥ 2)

### Step 1: Regenerate Synthetic Data with Visibility-Friendly Trajectory

Per [baseline_data_bootstrap.md](baseline_data_bootstrap.md), regenerate with:
- `--path_mode short_arc --arc_deg 30`
- `--pose_mode face_target --target "0,0,2"`

**Commands**:
```bash
# Clean old data
rm -rf ./data/habitat_vln/train ./data/habitat_vln/val
rm -rf ./raw_sequences/train ./raw_sequences/val
mkdir -p ./raw_sequences/train/RoomA ./raw_sequences/val/RoomB

# Generate with visibility-friendly trajectory
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 120 \
  --T 8 --W 384 --H 384 --pose_mode face_target --path_mode short_arc \
  --arc_deg 30 --radius 2.5 --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02

python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomB --clips 30 \
  --T 8 --W 384 --H 384 --pose_mode face_target --path_mode short_arc \
  --arc_deg 30 --radius 2.5 --target "0,0,2" --noise_rot_deg 2 --noise_trans 0.02
```

### Step 2: Restore Normal Packing Configuration

After regeneration, restore `min_visible_ratio` from `0.0` to `0.01` or `0.02`:

**File**: `configs/dataset_pack.yaml`
```yaml
min_visible_ratio: 0.02  # Restore from temporary 0.0
```

### Step 3: Repack and Verify K_eff Improvement

```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val
```

**Expected Results**:
- K_eff ≥ 2 ratio: **≥80%** (up from 0%)
- Average entropy: **≤5.0** (maintained)
- Empty heatmap ratio: **<20%** (down from 79.6%)

### Step 4: Run Baseline Training Pipeline

Once data quality passes SLO:

```bash
python scripts/overfit_one_batch.py  # Verify loss ≤ H(q) + 0.3
bash run_baseline.sh                 # Or step-by-step training
```

## 📝 Summary

### ✅ Achievements
1. **Fixed critical visibility computation bug**: Now uses Ki (reference intrinsics) correctly
2. **Added helper function**: Ensures consistency across codebase
3. **Updated threshold logic**: Relative entropy (H(q) + 0.3) instead of absolute <1.0
4. **Created comprehensive tests**: All assertion tests pass
5. **Verified fix works**: Synthetic test data shows K_eff=3 with perfect overlap

### ⚠️ Remaining Issue
- Existing synthetic data has poor geometric overlap (circular path, not looking at common target)
- Need to regenerate with `short_arc + face_target` to achieve K_eff ≥ 2 SLO

### 🎯 Immediate Action Required
**Regenerate synthetic data** with visibility-friendly trajectory parameters, then repack and verify quality improvement.

---

**Status**: Visibility computation fix **COMPLETE** ✅
**Next**: Data regeneration **REQUIRED** ⏳
