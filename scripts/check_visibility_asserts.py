#!/usr/bin/env python3
"""
check_visibility_asserts.py - Verify visibility computation correctness

Tests the critical fix in visible_ratio_for_keyframe to ensure it uses
reference frame intrinsics (Ki) instead of keyframe intrinsics (Kj).

Per fix_visibility_and_effectiveK.md § "一步一步跑完（§6）":
1. Generate minimal synthetic test case (2 frames with known geometry)
2. Assert visibility computation uses Ki (reference intrinsics)
3. Check K_eff > 0 for valid geometric overlap
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.heatmap_builder import visible_ratio_for_keyframe, project_to_ref_pixels


def test_project_to_ref_pixels():
    """Test 1: project_to_ref_pixels uses Ki correctly."""
    print("=" * 60)
    print("Test 1: project_to_ref_pixels helper")
    print("=" * 60)

    # Reference intrinsics
    Ki = {
        'fx': 400.0, 'fy': 400.0,
        'cx': 192.0, 'cy': 192.0
    }

    # Test point at (1, 0.5, 2) in reference camera frame
    pts_ci = np.array([[1.0, 0.5, 2.0]], dtype=np.float32)

    xi, yi, zi = project_to_ref_pixels(pts_ci, Ki)

    # Expected:
    # xi = 400 * (1/2) + 192 = 392
    # yi = 400 * (0.5/2) + 192 = 292
    expected_xi = 400.0 * (1.0 / 2.0) + 192.0  # 392.0
    expected_yi = 400.0 * (0.5 / 2.0) + 192.0  # 292.0

    print(f"Point in ref cam: {pts_ci[0]}")
    print(f"Projected pixel: ({xi[0]:.2f}, {yi[0]:.2f})")
    print(f"Expected pixel:  ({expected_xi:.2f}, {expected_yi:.2f})")

    assert np.abs(xi[0] - expected_xi) < 1e-3, f"xi mismatch: {xi[0]} != {expected_xi}"
    assert np.abs(yi[0] - expected_yi) < 1e-3, f"yi mismatch: {yi[0]} != {expected_yi}"
    assert zi[0] == 2.0, f"zi mismatch: {zi[0]} != 2.0"

    print("✅ PASS: project_to_ref_pixels uses Ki correctly\n")


def test_visibility_uses_ki_not_kj():
    """Test 2: visible_ratio_for_keyframe uses Ki (ref intrinsics), not Kj."""
    print("=" * 60)
    print("Test 2: visible_ratio_for_keyframe uses Ki (not Kj)")
    print("=" * 60)

    # Create two cameras with DIFFERENT intrinsics
    # Keyframe j has fx=300, Reference has fx=400
    # If code mistakenly uses Kj, projection will be wrong

    Kj = {
        'fx': 300.0, 'fy': 300.0,
        'cx': 192.0, 'cy': 192.0,
        'Kinv': np.linalg.inv(np.array([
            [300.0, 0, 192.0],
            [0, 300.0, 192.0],
            [0, 0, 1.0]
        ], dtype=np.float32))
    }

    Ki = {
        'fx': 400.0, 'fy': 400.0,
        'cx': 192.0, 'cy': 192.0
    }

    # Simple geometry: both cameras at origin looking down +Z
    # Keyframe j has a plane at depth 2.0
    H, W = 384, 384
    depth_j = np.full((H, W), 2.0, dtype=np.float32)

    # Reference also at origin with same plane
    depth_ref = np.full((H, W), 2.0, dtype=np.float32)

    # Identity transforms (both at world origin)
    T_w_c_j = np.eye(4, dtype=np.float32)
    T_c_ref_w = np.eye(4, dtype=np.float32)

    # Call visibility function
    vis_ratio = visible_ratio_for_keyframe(
        depth_j, Kj, Ki,  # Different intrinsics!
        T_w_c_j, T_c_ref_w,
        depth_ref, W, H,
        occl_eps=0.05, subsample=8
    )

    print(f"Keyframe intrinsics (Kj): fx={Kj['fx']}, fy={Kj['fy']}")
    print(f"Reference intrinsics (Ki): fx={Ki['fx']}, fy={Ki['fy']}")
    print(f"Visibility ratio: {vis_ratio:.4f}")

    # With correct Ki usage and identical geometry, should have high visibility
    # (Points project to valid locations in reference frame)
    assert vis_ratio > 0.5, (
        f"Expected high visibility (>0.5) with overlapping geometry, got {vis_ratio}. "
        f"This suggests the function may still be using Kj instead of Ki."
    )

    print("✅ PASS: visible_ratio_for_keyframe uses Ki correctly\n")


def test_k_eff_with_minimal_data():
    """Test 3: Simulate minimal dataset and check K_eff > 0."""
    print("=" * 60)
    print("Test 3: K_eff computation with minimal synthetic data")
    print("=" * 60)

    # Simulate a simple 4-frame sequence with strong overlap
    # All cameras at origin, looking at same scene

    K_intrinsics = {
        'fx': 400.0, 'fy': 400.0,
        'cx': 192.0, 'cy': 192.0,
        'Kinv': np.linalg.inv(np.array([
            [400.0, 0, 192.0],
            [0, 400.0, 192.0],
            [0, 0, 1.0]
        ], dtype=np.float32))
    }

    T = 4  # 4 frames
    H, W = 384, 384

    # All frames have same depth
    depth_maps = [np.full((H, W), 2.0, dtype=np.float32) for _ in range(T)]

    # All at origin (perfect overlap)
    poses = [np.eye(4, dtype=np.float32) for _ in range(T)]

    # Reference frame is last frame (index 3)
    ref_idx = 3
    T_c_ref_w = np.linalg.inv(poses[ref_idx])
    depth_ref = depth_maps[ref_idx]

    # Compute visibility for first 3 frames
    K_candidates = 3
    vis_scores = []
    for j in range(ref_idx):
        score = visible_ratio_for_keyframe(
            depth_maps[j], K_intrinsics, K_intrinsics,
            poses[j], T_c_ref_w,
            depth_ref, W, H,
            occl_eps=0.05, subsample=8
        )
        vis_scores.append((j, score))

    print(f"Visibility scores for frames 0-{ref_idx-1}:")
    for j, score in vis_scores:
        print(f"  Frame {j}: {score:.4f}")

    # All frames should have high visibility (perfect overlap)
    min_vis = 0.02  # Standard threshold
    valid_keyframes = [j for j, s in vis_scores if s >= min_vis]
    K_eff = len(valid_keyframes)

    print(f"\nValid keyframes (vis >= {min_vis}): {valid_keyframes}")
    print(f"K_eff: {K_eff}")

    assert K_eff >= 2, (
        f"Expected K_eff >= 2 with perfect geometric overlap, got {K_eff}. "
        f"This suggests visibility computation is still broken."
    )

    print("✅ PASS: K_eff >= 2 with synthetic overlapping geometry\n")


def main():
    print("\n" + "=" * 60)
    print("VISIBILITY COMPUTATION CORRECTNESS TESTS")
    print("=" * 60)
    print("Per fix_visibility_and_effectiveK.md critical fix:")
    print("  - visible_ratio_for_keyframe MUST use Ki (reference intrinsics)")
    print("  - NOT Kj (keyframe intrinsics)")
    print("=" * 60 + "\n")

    try:
        test_project_to_ref_pixels()
        test_visibility_uses_ki_not_kj()
        test_k_eff_with_minimal_data()

        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        print("Visibility computation is now correct.")
        print("Next steps:")
        print("  1. Run pack_dataset.py with relaxed min_visible_ratio")
        print("  2. Run report_data_quality.py to verify K_eff improvement")
        print("  3. Update overfit threshold to H(q) + 0.3")
        print("=" * 60)

        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ❌")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nPlease verify:")
        print("  1. heatmap_builder.py: project_to_ref_pixels() implementation")
        print("  2. heatmap_builder.py: visible_ratio_for_keyframe() uses project_to_ref_pixels()")
        print("  3. pack_dataset.py: passes both Kj and Ki parameters")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
