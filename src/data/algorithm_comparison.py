"""
Algorithm Comparison and Performance Analysis
============================================

Comprehensive comparison between the original greedy coverage algorithm
and the enhanced multi-objective submodular optimization approach.

This module provides:
1. Side-by-side performance comparison
2. Quality metrics analysis
3. Computational efficiency evaluation
4. Selection quality visualization
5. Integration testing
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import logging
from dataclasses import dataclass

from .frame_sampler import SpaceAwareFrameSampler, SamplingConfig
from .enhanced_frame_sampler import EnhancedFrameSampler, EnhancedSamplingConfig

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for algorithm comparison."""
    target_frames: int = 16
    candidate_frames: int = 128

    # Test scenarios
    test_scenarios: List[str] = None
    num_trials: int = 5

    # Quality metrics
    evaluate_coverage: bool = True
    evaluate_diversity: bool = True
    evaluate_temporal: bool = True
    evaluate_efficiency: bool = True

    # Visualization
    save_visualizations: bool = False
    output_dir: str = "./comparison_results"

    # System
    device: str = "cuda"

    def __post_init__(self):
        if self.test_scenarios is None:
            self.test_scenarios = [
                "high_coverage",      # Scenarios with high spatial coverage needs
                "high_diversity",     # Scenarios requiring semantic diversity
                "temporal_coherence", # Scenarios needing good temporal distribution
                "mixed_objectives"    # Balanced multi-objective scenarios
            ]


class AlgorithmComparator:
    """
    Comprehensive comparison between original and enhanced algorithms.

    Provides detailed analysis of performance, quality, and efficiency
    across different scenarios and use cases.
    """

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize both algorithms
        self._setup_algorithms()

        # Results storage
        self.comparison_results = {
            'original_algorithm': [],
            'enhanced_algorithm': [],
            'performance_metrics': [],
            'quality_analysis': [],
            'efficiency_metrics': []
        }

    def _setup_algorithms(self):
        """Initialize both original and enhanced algorithms."""

        # Original greedy coverage algorithm
        original_config = SamplingConfig(
            target_frames=self.config.target_frames,
            candidate_frames=self.config.candidate_frames,
            device=self.config.device
        )
        self.original_sampler = SpaceAwareFrameSampler(original_config)

        # Enhanced multi-objective algorithm
        enhanced_config = EnhancedSamplingConfig(
            target_frames=self.config.target_frames,
            candidate_frames=self.config.candidate_frames,
            optimization_method="submodular",
            device=self.config.device
        )
        self.enhanced_sampler = EnhancedFrameSampler(enhanced_config)

        logger.info("Both algorithms initialized successfully")

    def compare_algorithms(
        self,
        vggt_predictions: Dict[str, torch.Tensor],
        visual_features: Optional[torch.Tensor] = None,
        scenario: str = "mixed_objectives"
    ) -> Dict:
        """
        Compare both algorithms on the same input data.

        Args:
            vggt_predictions: VGGT model outputs
            visual_features: Optional visual features for semantic analysis
            scenario: Test scenario type

        Returns:
            Detailed comparison results
        """

        logger.info(f"Comparing algorithms for scenario: {scenario}")

        # Run original algorithm
        original_start = time.time()
        original_result = self.original_sampler(vggt_predictions)
        original_time = time.time() - original_start

        # Run enhanced algorithm
        enhanced_start = time.time()
        enhanced_result = self.enhanced_sampler(vggt_predictions, visual_features)
        enhanced_time = time.time() - enhanced_start

        # Analyze results
        comparison = self._analyze_comparison(
            original_result, enhanced_result,
            original_time, enhanced_time,
            vggt_predictions, scenario
        )

        return comparison

    def _analyze_comparison(
        self,
        original_result: Dict,
        enhanced_result: Dict,
        original_time: float,
        enhanced_time: float,
        vggt_predictions: Dict[str, torch.Tensor],
        scenario: str
    ) -> Dict:
        """Detailed analysis of comparison results."""

        analysis = {
            'scenario': scenario,
            'timing': {
                'original_time': original_time,
                'enhanced_time': enhanced_time,
                'speedup_ratio': original_time / enhanced_time if enhanced_time > 0 else float('inf')
            },
            'selection_analysis': {},
            'quality_metrics': {},
            'recommendations': []
        }

        # Extract selections
        original_indices = original_result['selected_indices'].cpu().numpy().flatten()
        enhanced_indices = enhanced_result['selected_indices'].cpu().numpy().flatten()

        # Selection overlap analysis
        original_set = set(original_indices)
        enhanced_set = set(enhanced_indices)
        overlap = len(original_set.intersection(enhanced_set))

        analysis['selection_analysis'] = {
            'original_selection': original_indices.tolist(),
            'enhanced_selection': enhanced_indices.tolist(),
            'selection_overlap': overlap,
            'overlap_ratio': overlap / len(original_set) if len(original_set) > 0 else 0,
            'unique_to_original': list(original_set - enhanced_set),
            'unique_to_enhanced': list(enhanced_set - original_set)
        }

        # Quality metrics comparison
        if self.config.evaluate_coverage:
            analysis['quality_metrics']['coverage'] = self._compare_coverage_quality(
                original_result, enhanced_result, vggt_predictions
            )

        if self.config.evaluate_diversity:
            analysis['quality_metrics']['diversity'] = self._compare_diversity_quality(
                original_indices, enhanced_indices
            )

        if self.config.evaluate_temporal:
            analysis['quality_metrics']['temporal'] = self._compare_temporal_quality(
                original_indices, enhanced_indices
            )

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _compare_coverage_quality(
        self,
        original_result: Dict,
        enhanced_result: Dict,
        vggt_predictions: Dict[str, torch.Tensor]
    ) -> Dict:
        """Compare spatial coverage quality."""

        coverage_comparison = {
            'original_coverage': 0,
            'enhanced_coverage': 0,
            'coverage_improvement': 0.0
        }

        # Extract coverage information
        if 'total_coverage' in original_result:
            coverage_comparison['original_coverage'] = original_result['total_coverage']

        if 'optimization_result' in enhanced_result:
            opt_result = enhanced_result['optimization_result']
            if 'objective_history' in opt_result and opt_result['objective_history']:
                # Use final objective value as coverage proxy
                coverage_comparison['enhanced_coverage'] = opt_result['objective_history'][-1]

        # Calculate improvement
        if coverage_comparison['original_coverage'] > 0:
            improvement = (
                (coverage_comparison['enhanced_coverage'] - coverage_comparison['original_coverage']) /
                coverage_comparison['original_coverage']
            )
            coverage_comparison['coverage_improvement'] = improvement

        return coverage_comparison

    def _compare_diversity_quality(self, original_indices: np.ndarray, enhanced_indices: np.ndarray) -> Dict:
        """Compare semantic/temporal diversity quality."""

        # Simple diversity metrics based on temporal distribution
        original_diversity = self._calculate_temporal_diversity(original_indices)
        enhanced_diversity = self._calculate_temporal_diversity(enhanced_indices)

        return {
            'original_diversity': original_diversity,
            'enhanced_diversity': enhanced_diversity,
            'diversity_improvement': enhanced_diversity - original_diversity
        }

    def _calculate_temporal_diversity(self, indices: np.ndarray) -> float:
        """Calculate temporal diversity score."""

        if len(indices) <= 1:
            return 0.0

        # Sort indices
        sorted_indices = np.sort(indices)

        # Calculate gaps between consecutive frames
        gaps = np.diff(sorted_indices)

        # Diversity based on gap consistency and coverage
        gap_std = np.std(gaps) if len(gaps) > 0 else 0
        gap_mean = np.mean(gaps) if len(gaps) > 0 else 0

        # Lower std (more consistent gaps) and higher mean gaps = better diversity
        if gap_mean > 0:
            consistency_score = 1.0 / (1.0 + gap_std / gap_mean)
            coverage_score = (sorted_indices[-1] - sorted_indices[0]) / max(sorted_indices[-1], 1)
            diversity = 0.6 * consistency_score + 0.4 * coverage_score
        else:
            diversity = 0.0

        return diversity

    def _compare_temporal_quality(self, original_indices: np.ndarray, enhanced_indices: np.ndarray) -> Dict:
        """Compare temporal coherence quality."""

        original_temporal = self._calculate_temporal_coherence(original_indices)
        enhanced_temporal = self._calculate_temporal_coherence(enhanced_indices)

        return {
            'original_temporal_coherence': original_temporal,
            'enhanced_temporal_coherence': enhanced_temporal,
            'temporal_improvement': enhanced_temporal - original_temporal
        }

    def _calculate_temporal_coherence(self, indices: np.ndarray) -> float:
        """Calculate temporal coherence score."""

        if len(indices) <= 1:
            return 1.0

        sorted_indices = np.sort(indices)

        # Coherence based on temporal smoothness
        gaps = np.diff(sorted_indices)

        # Prefer consistent, moderate gaps
        ideal_gap = max(sorted_indices) / len(indices) if len(indices) > 0 else 1
        gap_errors = np.abs(gaps - ideal_gap)
        coherence = 1.0 / (1.0 + np.mean(gap_errors))

        return coherence

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on comparison analysis."""

        recommendations = []

        # Timing recommendations
        timing = analysis['timing']
        if timing['enhanced_time'] > timing['original_time'] * 2:
            recommendations.append(
                "Enhanced algorithm is significantly slower - consider optimization method tuning"
            )
        elif timing['enhanced_time'] < timing['original_time']:
            recommendations.append(
                "Enhanced algorithm shows better computational efficiency"
            )

        # Quality recommendations
        quality = analysis.get('quality_metrics', {})

        if 'coverage' in quality:
            coverage_improvement = quality['coverage']['coverage_improvement']
            if coverage_improvement > 0.1:
                recommendations.append(
                    f"Enhanced algorithm provides {coverage_improvement:.1%} better coverage"
                )
            elif coverage_improvement < -0.1:
                recommendations.append(
                    f"Original algorithm provides better coverage by {-coverage_improvement:.1%}"
                )

        if 'diversity' in quality:
            diversity_improvement = quality['diversity']['diversity_improvement']
            if diversity_improvement > 0.1:
                recommendations.append(
                    "Enhanced algorithm provides significantly better diversity"
                )

        if 'temporal' in quality:
            temporal_improvement = quality['temporal']['temporal_improvement']
            if temporal_improvement > 0.1:
                recommendations.append(
                    "Enhanced algorithm provides better temporal coherence"
                )

        # Selection analysis recommendations
        selection = analysis['selection_analysis']
        if selection['overlap_ratio'] < 0.5:
            recommendations.append(
                f"Low selection overlap ({selection['overlap_ratio']:.1%}) - algorithms have different strategies"
            )
        elif selection['overlap_ratio'] > 0.9:
            recommendations.append(
                "High selection overlap - both algorithms produce similar results"
            )

        return recommendations

    def run_comprehensive_evaluation(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        visual_features_list: Optional[List[torch.Tensor]] = None
    ) -> Dict:
        """
        Run comprehensive evaluation across multiple scenarios and datasets.

        Args:
            test_data: List of VGGT prediction dictionaries
            visual_features_list: Optional list of visual features

        Returns:
            Comprehensive evaluation results
        """

        logger.info(f"Running comprehensive evaluation on {len(test_data)} datasets")

        all_results = []

        for i, vggt_predictions in enumerate(test_data):
            visual_features = visual_features_list[i] if visual_features_list else None

            for scenario in self.config.test_scenarios:
                for trial in range(self.config.num_trials):
                    try:
                        result = self.compare_algorithms(
                            vggt_predictions, visual_features, scenario
                        )
                        result['dataset_idx'] = i
                        result['trial'] = trial
                        all_results.append(result)

                    except Exception as e:
                        logger.warning(f"Failed comparison for dataset {i}, scenario {scenario}, trial {trial}: {e}")

        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)

        return {
            'individual_results': all_results,
            'aggregated_metrics': aggregated_results,
            'summary_recommendations': self._generate_summary_recommendations(aggregated_results)
        }

    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all trials and scenarios."""

        aggregated = {
            'timing_statistics': {},
            'quality_statistics': {},
            'scenario_analysis': {}
        }

        # Timing statistics
        original_times = [r['timing']['original_time'] for r in all_results]
        enhanced_times = [r['timing']['enhanced_time'] for r in all_results]

        aggregated['timing_statistics'] = {
            'original_mean_time': np.mean(original_times),
            'original_std_time': np.std(original_times),
            'enhanced_mean_time': np.mean(enhanced_times),
            'enhanced_std_time': np.std(enhanced_times),
            'average_speedup': np.mean([r['timing']['speedup_ratio'] for r in all_results])
        }

        # Quality statistics by scenario
        for scenario in self.config.test_scenarios:
            scenario_results = [r for r in all_results if r['scenario'] == scenario]

            if scenario_results:
                scenario_analysis = {
                    'num_trials': len(scenario_results),
                    'avg_overlap_ratio': np.mean([
                        r['selection_analysis']['overlap_ratio'] for r in scenario_results
                    ])
                }

                # Quality metrics if available
                for metric_type in ['coverage', 'diversity', 'temporal']:
                    metric_improvements = []
                    for r in scenario_results:
                        if metric_type in r.get('quality_metrics', {}):
                            if f'{metric_type}_improvement' in r['quality_metrics'][metric_type]:
                                metric_improvements.append(
                                    r['quality_metrics'][metric_type][f'{metric_type}_improvement']
                                )

                    if metric_improvements:
                        scenario_analysis[f'{metric_type}_improvement_mean'] = np.mean(metric_improvements)
                        scenario_analysis[f'{metric_type}_improvement_std'] = np.std(metric_improvements)

                aggregated['scenario_analysis'][scenario] = scenario_analysis

        return aggregated

    def _generate_summary_recommendations(self, aggregated_results: Dict) -> List[str]:
        """Generate high-level recommendations based on aggregated results."""

        recommendations = []

        # Timing recommendations
        timing = aggregated_results['timing_statistics']
        if timing['enhanced_mean_time'] > timing['original_mean_time'] * 1.5:
            recommendations.append(
                "Enhanced algorithm is consistently slower - consider performance optimizations"
            )
        elif timing['enhanced_mean_time'] < timing['original_mean_time'] * 0.8:
            recommendations.append(
                "Enhanced algorithm shows consistent performance improvements"
            )

        # Scenario-specific recommendations
        scenario_analysis = aggregated_results['scenario_analysis']

        best_scenarios = []
        for scenario, analysis in scenario_analysis.items():
            total_improvement = 0
            improvement_count = 0

            for metric in ['coverage', 'diversity', 'temporal']:
                if f'{metric}_improvement_mean' in analysis:
                    total_improvement += analysis[f'{metric}_improvement_mean']
                    improvement_count += 1

            if improvement_count > 0:
                avg_improvement = total_improvement / improvement_count
                if avg_improvement > 0.1:
                    best_scenarios.append((scenario, avg_improvement))

        if best_scenarios:
            best_scenarios.sort(key=lambda x: x[1], reverse=True)
            recommendations.append(
                f"Enhanced algorithm performs best in: {', '.join([s[0] for s in best_scenarios[:2]])}"
            )

        return recommendations


def create_algorithm_comparator(
    target_frames: int = 16,
    candidate_frames: int = 128,
    test_scenarios: Optional[List[str]] = None,
    device: str = "cuda"
) -> AlgorithmComparator:
    """
    Factory function to create an algorithm comparator.

    Returns:
        Configured AlgorithmComparator instance
    """
    config = ComparisonConfig(
        target_frames=target_frames,
        candidate_frames=candidate_frames,
        test_scenarios=test_scenarios,
        device=device
    )

    return AlgorithmComparator(config)


# Example usage and testing
if __name__ == "__main__":
    comparator = create_algorithm_comparator(
        target_frames=16,
        candidate_frames=32,  # Reduced for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Algorithm Comparator created successfully!")
    print(f"Test scenarios: {comparator.config.test_scenarios}")