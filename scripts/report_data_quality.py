"""
Data Quality Reporting Script (report_data_quality.py)
======================================================

Analyze and report data quality metrics for packed VLN heatmap dataset.

Metrics reported:
- Number of samples
- Effective K statistics (avg, histogram)
- Heatmap entropy statistics (avg, percentiles)
- Quality threshold compliance

Usage:
    python scripts/report_data_quality.py --root ./data/habitat_vln --split train
    python scripts/report_data_quality.py --root ./data/habitat_vln --split val
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.data.quality_metrics import heatmap_entropy, effective_k

logger = logging.getLogger(__name__)


def analyze_dataset_quality(root: str, split: str, frames_per_clip: int = 8,
                            heatmap_per_clip: int = 4, hm_size: tuple = (64, 64)) -> Dict:
    """
    Analyze data quality metrics for the dataset.

    Args:
        root: Dataset root directory
        split: Split name ('train', 'val', 'test')
        frames_per_clip: Number of frames per clip (T)
        heatmap_per_clip: Number of heatmaps per clip (K)
        hm_size: Heatmap size (Hm, Wm)

    Returns:
        dict: Quality metrics report
    """
    logger.info(f"Loading dataset: {root}/{split}")

    # Create dataset
    try:
        ds = VLNHeatmapDataset(
            root=root,
            split=split,
            frames_per_clip=frames_per_clip,
            heatmap_per_clip=heatmap_per_clip,
            image_size=(384, 384),
            hm_size=hm_size
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

    if len(ds) == 0:
        logger.warning(f"Empty dataset: {root}/{split}")
        return {
            'num_samples': 0,
            'error': 'Empty dataset'
        }

    logger.info(f"Analyzing {len(ds)} samples...")

    # Collect metrics
    entropies = []
    k_eff_list = []
    samples_with_low_k = 0
    min_effective_k_threshold = 2

    for i in range(len(ds)):
        try:
            item = ds[i]
            hm = item['gt_heatmaps']  # [K, Hm, Wm]
            mask = item['mask']       # [K]

            # Compute effective K
            k_eff = effective_k(mask).item()
            k_eff_list.append(k_eff)

            # Count low-quality samples
            if k_eff < min_effective_k_threshold:
                samples_with_low_k += 1

            # Compute entropy for valid heatmaps
            H = heatmap_entropy(hm)  # [K]
            m = mask.bool()

            if m.any():
                # Only include valid heatmaps
                valid_entropies = H[m].tolist()
                entropies.extend(valid_entropies)

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(ds)} samples analyzed")

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue

    # Compute statistics
    report = {
        'dataset': {
            'root': root,
            'split': split,
            'num_samples': len(ds),
            'frames_per_clip': frames_per_clip,
            'heatmap_per_clip': heatmap_per_clip,
            'hm_size': list(hm_size)
        },
        'effective_k': {
            'avg': float(np.mean(k_eff_list)) if k_eff_list else None,
            'min': float(np.min(k_eff_list)) if k_eff_list else None,
            'max': float(np.max(k_eff_list)) if k_eff_list else None,
            'std': float(np.std(k_eff_list)) if k_eff_list else None,
            'histogram': {}
        },
        'entropy': {
            'avg': float(np.mean(entropies)) if entropies else None,
            'min': float(np.min(entropies)) if entropies else None,
            'max': float(np.max(entropies)) if entropies else None,
            'std': float(np.std(entropies)) if entropies else None,
            'p50': float(np.percentile(entropies, 50)) if entropies else None,
            'p90': float(np.percentile(entropies, 90)) if entropies else None,
            'p95': float(np.percentile(entropies, 95)) if entropies else None
        },
        'quality': {
            'samples_with_k_eff_below_2': samples_with_low_k,
            'ratio_k_eff_below_2': float(samples_with_low_k) / len(ds) if len(ds) > 0 else None,
            'samples_with_k_eff_gte_2': len(ds) - samples_with_low_k,
            'ratio_k_eff_gte_2': float(len(ds) - samples_with_low_k) / len(ds) if len(ds) > 0 else None
        }
    }

    # Build effective K histogram
    for k in range(heatmap_per_clip + 1):
        count = sum(1 for x in k_eff_list if x == k)
        report['effective_k']['histogram'][str(k)] = count

    # SLO compliance check
    target_k_eff_gte_2_ratio = 0.80  # 80% target
    target_avg_entropy_max = 5.0     # max 5.0 for 64x64

    report['slo_compliance'] = {
        'k_eff_gte_2_ratio_target': target_k_eff_gte_2_ratio,
        'k_eff_gte_2_ratio_actual': report['quality']['ratio_k_eff_gte_2'],
        'k_eff_gte_2_pass': report['quality']['ratio_k_eff_gte_2'] >= target_k_eff_gte_2_ratio if report['quality']['ratio_k_eff_gte_2'] is not None else False,
        'avg_entropy_target': target_avg_entropy_max,
        'avg_entropy_actual': report['entropy']['avg'],
        'avg_entropy_pass': report['entropy']['avg'] <= target_avg_entropy_max if report['entropy']['avg'] is not None else False
    }

    return report


def print_report(report: Dict):
    """Print quality report in human-readable format."""
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    print(f"\n📁 Dataset: {report['dataset']['root']}/{report['dataset']['split']}")
    print(f"   Samples: {report['dataset']['num_samples']}")
    print(f"   Frames per clip: {report['dataset']['frames_per_clip']}")
    print(f"   Heatmaps per clip: {report['dataset']['heatmap_per_clip']}")
    print(f"   Heatmap size: {report['dataset']['hm_size']}")

    print("\n📊 Effective K Statistics:")
    if report['effective_k']['avg'] is not None:
        print(f"   Average: {report['effective_k']['avg']:.2f}")
        print(f"   Range: [{report['effective_k']['min']:.0f}, {report['effective_k']['max']:.0f}]")
        print(f"   Std Dev: {report['effective_k']['std']:.2f}")
        print(f"   Histogram:")
        for k, count in sorted(report['effective_k']['histogram'].items()):
            ratio = count / report['dataset']['num_samples'] if report['dataset']['num_samples'] > 0 else 0
            print(f"     K={k}: {count} samples ({ratio:.1%})")
    else:
        print("   No data")

    print("\n🔥 Heatmap Entropy Statistics:")
    if report['entropy']['avg'] is not None:
        print(f"   Average: {report['entropy']['avg']:.3f}")
        print(f"   Range: [{report['entropy']['min']:.3f}, {report['entropy']['max']:.3f}]")
        print(f"   Std Dev: {report['entropy']['std']:.3f}")
        print(f"   Percentiles:")
        print(f"     P50 (median): {report['entropy']['p50']:.3f}")
        print(f"     P90: {report['entropy']['p90']:.3f}")
        print(f"     P95: {report['entropy']['p95']:.3f}")
    else:
        print("   No data")

    print("\n✅ Quality Metrics:")
    print(f"   Samples with K_eff ≥ 2: {report['quality']['samples_with_k_eff_gte_2']} "
          f"({report['quality']['ratio_k_eff_gte_2']:.1%})")
    print(f"   Samples with K_eff < 2: {report['quality']['samples_with_k_eff_below_2']} "
          f"({report['quality']['ratio_k_eff_below_2']:.1%})")

    print("\n🎯 SLO Compliance:")
    slo = report['slo_compliance']
    k_eff_status = "✅ PASS" if slo['k_eff_gte_2_pass'] else "❌ FAIL"
    entropy_status = "✅ PASS" if slo['avg_entropy_pass'] else "❌ FAIL"

    print(f"   K_eff ≥ 2 ratio: {k_eff_status}")
    print(f"     Target: ≥ {slo['k_eff_gte_2_ratio_target']:.1%}")
    print(f"     Actual: {slo['k_eff_gte_2_ratio_actual']:.1%}")

    print(f"   Average Entropy: {entropy_status}")
    print(f"     Target: ≤ {slo['avg_entropy_target']:.1f}")
    print(f"     Actual: {slo['avg_entropy_actual']:.3f}")

    overall_pass = slo['k_eff_gte_2_pass'] and slo['avg_entropy_pass']
    overall_status = "✅ PASS" if overall_pass else "❌ FAIL"
    print(f"\n   Overall: {overall_status}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Report data quality metrics for VLN heatmap dataset")
    parser.add_argument('--root', type=str, default='./data/habitat_vln',
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split to analyze')
    parser.add_argument('--frames', type=int, default=8,
                        help='Frames per clip (T)')
    parser.add_argument('--heatmaps', type=int, default=4,
                        help='Heatmaps per clip (K)')
    parser.add_argument('--hm-size', type=int, nargs=2, default=[64, 64],
                        help='Heatmap size (Hm Wm)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (optional)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Analyze dataset
    report = analyze_dataset_quality(
        root=args.root,
        split=args.split,
        frames_per_clip=args.frames,
        heatmap_per_clip=args.heatmaps,
        hm_size=tuple(args.hm_size)
    )

    if report is None:
        logger.error("Failed to generate report")
        sys.exit(1)

    # Print report
    print_report(report)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")
    else:
        # Default save location
        default_output = Path('./outputs/reports')
        default_output.mkdir(parents=True, exist_ok=True)
        output_file = default_output / f'data_quality_{args.split}.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_file}")


if __name__ == '__main__':
    main()