"""
check_mesh_visibility.py
========================

Validation script to verify mesh rendering produces sufficient depth coverage
and visibility for keyframe selection.

Usage:
    python scripts/check_mesh_visibility.py
"""

import os
import json
import numpy as np
from glob import glob
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def depth_coverage(depth):
    """Compute percentage of non-zero depth pixels."""
    H, W = depth.shape
    valid_pixels = (depth > 0).sum()
    total_pixels = H * W
    return float(valid_pixels) / float(total_pixels)


def check_clip_visibility(clip_path):
    """Check visibility metrics for a single clip."""
    # Check for poses.json (gen_synth_demo.py format)
    poses_path = Path(clip_path) / 'poses.json'
    if not poses_path.exists():
        print(f"⚠️  Warning: poses.json not found in {clip_path}")
        return None

    with open(poses_path, 'r') as f:
        poses = json.load(f)

    T = len(poses)
    ref_idx = T - 1  # Last frame is reference by default

    # Read depth maps
    depth_dir = Path(clip_path) / 'depth'
    if not depth_dir.exists():
        print(f"⚠️  Warning: depth directory not found in {clip_path}")
        return None

    coverages = []
    for t in range(T):
        depth_path = depth_dir / f"{t:06d}.npy"
        if depth_path.exists():
            depth = np.load(depth_path)
            cov = depth_coverage(depth)
            coverages.append(cov)
        else:
            print(f"⚠️  Warning: depth file not found: {depth_path}")
            coverages.append(0.0)

    return {
        'clip_path': str(clip_path),
        'T': T,
        'ref_idx': ref_idx,
        'coverages': coverages,
        'avg_coverage': np.mean(coverages),
        'min_coverage': np.min(coverages),
        'max_coverage': np.max(coverages),
    }


def main():
    """Run visibility checks on generated clips."""
    raw_root = Path('./raw_sequences')

    print("🔍 Checking mesh visibility and depth coverage\n")

    # Check train clips
    train_clips = sorted(glob(str(raw_root / 'train' / '*' / 'clip_*')))
    if not train_clips:
        print("❌ No training clips found. Please run gen_synth_demo.py first.")
        return 1

    print(f"Found {len(train_clips)} training clips")

    all_results = []
    for clip_path in train_clips[:3]:  # Check first 3 clips
        print(f"\nChecking: {clip_path}")
        result = check_clip_visibility(clip_path)
        if result:
            all_results.append(result)
            print(f"  T={result['T']}, ref_idx={result['ref_idx']}")
            print(f"  Coverage: avg={result['avg_coverage']:.3f}, "
                  f"min={result['min_coverage']:.3f}, "
                  f"max={result['max_coverage']:.3f}")

            # Check specific frames
            if result['ref_idx'] > 0:
                cov_ref = result['coverages'][result['ref_idx']]
                cov_prev = result['coverages'][result['ref_idx'] - 1]
                print(f"  Ref frame coverage: {cov_ref:.3f}")
                print(f"  Prev frame coverage: {cov_prev:.3f}")

    # Aggregate statistics
    if all_results:
        print("\n" + "="*60)
        print("📊 Summary Statistics")
        print("="*60)

        avg_coverages = [r['avg_coverage'] for r in all_results]
        min_coverages = [r['min_coverage'] for r in all_results]

        print(f"Overall avg coverage: {np.mean(avg_coverages):.3f}")
        print(f"Overall min coverage: {np.mean(min_coverages):.3f}")

        # Validation checks
        print("\n✅ Validation Checks:")

        target_coverage = 0.7
        passed = True

        if np.mean(avg_coverages) > target_coverage:
            print(f"✅ Average coverage > {target_coverage:.1%}: PASS")
        else:
            print(f"❌ Average coverage < {target_coverage:.1%}: FAIL")
            print("   → Increase --mesh_grid or check rendering")
            passed = False

        if np.mean(min_coverages) > 0.5:
            print(f"✅ Minimum coverage > 50%: PASS")
        else:
            print(f"⚠️  Minimum coverage < 50%: WARNING")
            print("   → Some frames may have low visibility")

        print("\n" + "="*60)

        if not passed:
            print("\n❌ Coverage validation FAILED")
            print("   Please adjust mesh_grid or rendering parameters")
            return 1
        else:
            print("\n✅ Coverage validation PASSED")
            print("   Depth coverage is sufficient for K_eff >= 2")
            return 0

    else:
        print("\n❌ No valid results to analyze")
        return 1


if __name__ == '__main__':
    exit(main())
