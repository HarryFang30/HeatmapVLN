"""
Flexible Testing Framework for Frame Sampling Algorithms
========================================================

This module provides a comprehensive testing framework that supports:
- Multiple algorithm testing with automatic comparison
- Configurable test scenarios and datasets
- Performance benchmarking and profiling
- Quality metrics evaluation
- Test result visualization and reporting
- Automated test discovery and execution

Features:
- Test scenario definitions
- Synthetic data generation
- Real data integration
- Performance monitoring
- Result comparison and analysis
- Extensible test case framework
"""

import torch
import numpy as np
import time
import logging
import json
import yaml
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd

from ..data.algorithm_registry import AlgorithmRegistry, AlgorithmConfig, AlgorithmType, get_registry
from ..data.algorithm_factory import AlgorithmFactory, get_factory

logger = logging.getLogger(__name__)


class TestScenario(Enum):
    """Predefined test scenarios."""
    QUICK_SMOKE = "quick_smoke"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    QUALITY_ANALYSIS = "quality_analysis"
    SCALABILITY_TEST = "scalability_test"
    ROBUSTNESS_TEST = "robustness_test"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CUSTOM = "custom"


@dataclass
class TestDataSpec:
    """Specification for test data generation."""
    batch_size: int = 1
    num_frames: int = 32
    height: int = 64
    width: int = 64
    feature_dim: int = 256
    device: str = "cpu"

    # Data characteristics
    scene_complexity: float = 0.5  # 0.0 = simple, 1.0 = complex
    temporal_coherence: float = 0.7  # 0.0 = random, 1.0 = smooth
    noise_level: float = 0.1  # 0.0 = clean, 1.0 = noisy

    # Spatial patterns
    depth_range: Tuple[float, float] = (1.0, 20.0)
    confidence_range: Tuple[float, float] = (0.2, 0.95)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestMetrics:
    """Metrics collected during testing."""
    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0

    # Quality metrics
    selection_diversity: float = 0.0
    temporal_distribution: float = 0.0
    spatial_coverage: float = 0.0

    # Algorithm-specific metrics
    algorithm_specific: Dict[str, float] = field(default_factory=dict)

    # Metadata
    algorithm_name: str = ""
    test_scenario: str = ""
    data_spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseTestCase(ABC):
    """Abstract base class for test cases."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_test_data(self, data_spec: TestDataSpec) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Generate test data for this test case."""
        pass

    @abstractmethod
    def evaluate_result(self, result: Dict[str, torch.Tensor], data_spec: TestDataSpec) -> Dict[str, float]:
        """Evaluate algorithm result and return quality metrics."""
        pass

    def setup(self):
        """Setup before test execution."""
        pass

    def teardown(self):
        """Cleanup after test execution."""
        pass


class SyntheticTestCase(BaseTestCase):
    """Test case using synthetic data generation."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)

    def generate_test_data(self, data_spec: TestDataSpec) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Generate synthetic VGGT predictions and visual features."""

        device = torch.device(data_spec.device)

        # Generate synthetic VGGT predictions
        vggt_predictions = self._generate_vggt_predictions(data_spec, device)

        # Generate synthetic visual features
        visual_features = self._generate_visual_features(data_spec, device)

        return vggt_predictions, visual_features

    def _generate_vggt_predictions(self, data_spec: TestDataSpec, device: torch.device) -> Dict[str, torch.Tensor]:
        """Generate synthetic VGGT predictions."""

        B, S, H, W = data_spec.batch_size, data_spec.num_frames, data_spec.height, data_spec.width

        # Depth maps with spatial structure
        depth_maps = torch.zeros(B, S, H, W, 1, device=device)
        depth_conf = torch.zeros(B, S, H, W, device=device)
        world_points = torch.zeros(B, S, H, W, 3, device=device)
        world_points_conf = torch.zeros(B, S, H, W, device=device)
        pose_enc = torch.zeros(B, S, 9, device=device)

        for b in range(B):
            for s in range(S):
                # Create spatial depth pattern
                y_grid, x_grid = torch.meshgrid(
                    torch.linspace(0, 1, H, device=device),
                    torch.linspace(0, 1, W, device=device),
                    indexing='ij'
                )

                # Base depth with temporal variation
                temporal_factor = torch.sin(torch.tensor(s * np.pi / 8, device=device)) * data_spec.temporal_coherence
                depth_base = data_spec.depth_range[0] + (data_spec.depth_range[1] - data_spec.depth_range[0]) * 0.5
                depth_pattern = torch.full_like(x_grid, depth_base) * (1 + 0.3 * temporal_factor)

                # Add spatial complexity
                if data_spec.scene_complexity > 0.3:
                    complexity_pattern = (
                        torch.sin(x_grid * 10 * data_spec.scene_complexity) *
                        torch.cos(y_grid * 8 * data_spec.scene_complexity)
                    )
                    depth_pattern += 2.0 * complexity_pattern

                # Add noise
                if data_spec.noise_level > 0:
                    noise = torch.randn_like(depth_pattern) * data_spec.noise_level
                    depth_pattern += noise

                depth_maps[b, s, :, :, 0] = depth_pattern.clamp(*data_spec.depth_range)

                # Confidence patterns
                conf_base = (data_spec.confidence_range[0] + data_spec.confidence_range[1]) / 2
                conf_variation = (data_spec.confidence_range[1] - data_spec.confidence_range[0]) / 2
                confidence = conf_base + conf_variation * (0.5 - data_spec.noise_level + torch.rand_like(depth_pattern) * 0.4)
                depth_conf[b, s] = confidence.clamp(*data_spec.confidence_range)

                # Convert to world points (simplified)
                fx = fy = 200.0
                cx, cy = W // 2, H // 2

                for y in range(H):
                    for x in range(W):
                        z = depth_maps[b, s, y, x, 0]
                        world_x = (x - cx) * z / fx
                        world_y = (y - cy) * z / fy
                        world_points[b, s, y, x, 0] = world_x
                        world_points[b, s, y, x, 1] = world_y
                        world_points[b, s, y, x, 2] = z
                        world_points_conf[b, s, y, x] = depth_conf[b, s, y, x] * 0.9

                # Camera poses
                pose_enc[b, s, 0] = s * 0.1  # tx
                pose_enc[b, s, 1] = torch.sin(torch.tensor(s * 0.2, device=device))  # ty
                pose_enc[b, s, 2] = s * 0.05  # tz
                pose_enc[b, s, 6] = 1.0  # qw (identity rotation)
                pose_enc[b, s, 7] = fx  # fx
                pose_enc[b, s, 8] = fy  # fy

        return {
            'depth': depth_maps,
            'depth_conf': depth_conf,
            'world_points': world_points,
            'world_points_conf': world_points_conf,
            'pose_enc': pose_enc
        }

    def _generate_visual_features(self, data_spec: TestDataSpec, device: torch.device) -> torch.Tensor:
        """Generate synthetic visual features."""

        B, S, D = data_spec.batch_size, data_spec.num_frames, data_spec.feature_dim

        # Create structured features with temporal coherence
        features = torch.randn(B, S, D, device=device)

        # Add temporal structure
        if data_spec.temporal_coherence > 0.5:
            # Create feature clusters for temporal coherence
            for b in range(B):
                num_clusters = max(2, S // 4)
                cluster_size = S // num_clusters

                for cluster_id in range(num_clusters):
                    start_frame = cluster_id * cluster_size
                    end_frame = min((cluster_id + 1) * cluster_size, S)

                    # Cluster center
                    cluster_center = torch.randn(D, device=device)

                    for s in range(start_frame, end_frame):
                        variation = torch.randn(D, device=device) * (1 - data_spec.temporal_coherence)
                        features[b, s] = cluster_center + variation

        return features

    def evaluate_result(self, result: Dict[str, torch.Tensor], data_spec: TestDataSpec) -> Dict[str, float]:
        """Evaluate synthetic test result."""

        selected_indices = result['selected_indices'].cpu().numpy().flatten()

        # Temporal distribution quality
        temporal_dist = self._calculate_temporal_distribution(selected_indices, data_spec.num_frames)

        # Selection diversity (based on temporal spacing)
        diversity = self._calculate_selection_diversity(selected_indices)

        # Coverage (proxy based on frame span)
        coverage = (selected_indices.max() - selected_indices.min()) / max(data_spec.num_frames - 1, 1)

        return {
            'temporal_distribution': temporal_dist,
            'selection_diversity': diversity,
            'spatial_coverage': coverage
        }

    def _calculate_temporal_distribution(self, indices: np.ndarray, total_frames: int) -> float:
        """Calculate temporal distribution quality."""
        if len(indices) <= 1:
            return 1.0

        sorted_indices = np.sort(indices)
        gaps = np.diff(sorted_indices)

        # Ideal gap
        ideal_gap = total_frames / len(indices)
        gap_errors = np.abs(gaps - ideal_gap)

        # Quality based on how close gaps are to ideal
        quality = 1.0 / (1.0 + np.mean(gap_errors) / ideal_gap)
        return float(quality)

    def _calculate_selection_diversity(self, indices: np.ndarray) -> float:
        """Calculate selection diversity score."""
        if len(indices) <= 1:
            return 0.0

        sorted_indices = np.sort(indices)
        gaps = np.diff(sorted_indices)

        # Diversity based on gap variance (lower variance = higher diversity)
        gap_std = np.std(gaps)
        gap_mean = np.mean(gaps)

        if gap_mean > 0:
            diversity = 1.0 / (1.0 + gap_std / gap_mean)
        else:
            diversity = 0.0

        return float(diversity)


class FlexibleTestFramework:
    """
    Comprehensive testing framework for frame sampling algorithms.

    Provides flexible testing capabilities with configurable scenarios,
    automated benchmarking, and comprehensive result analysis.
    """

    def __init__(self, factory: Optional[AlgorithmFactory] = None):
        self.factory = factory or get_factory()
        self.test_cases: Dict[str, BaseTestCase] = {}
        self.test_results: List[TestMetrics] = []

        # Register built-in test cases
        self._register_builtin_test_cases()

        logger.info("FlexibleTestFramework initialized")

    def register_test_case(self, test_case: BaseTestCase):
        """Register a custom test case."""
        self.test_cases[test_case.name] = test_case
        logger.info(f"Registered test case: {test_case.name}")

    def run_test_scenario(
        self,
        scenario: Union[TestScenario, str],
        algorithms: Optional[List[str]] = None,
        data_specs: Optional[List[TestDataSpec]] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, List[TestMetrics]]:
        """Run a predefined test scenario."""

        if isinstance(scenario, str):
            scenario = TestScenario(scenario)

        logger.info(f"Running test scenario: {scenario.value}")

        # Get scenario configuration
        config = self._get_scenario_config(scenario)

        # Use provided algorithms or default
        if algorithms is None:
            algorithms = config.get('algorithms', ['greedy_coverage', 'enhanced_submodular'])

        # Use provided data specs or default
        if data_specs is None:
            data_specs = config.get('data_specs', [TestDataSpec()])

        # Run tests
        results = self.run_algorithm_comparison(
            algorithms=algorithms,
            data_specs=data_specs,
            test_cases=config.get('test_cases', ['synthetic']),
            save_results=save_results,
            output_dir=output_dir
        )

        logger.info(f"Completed test scenario: {scenario.value}")
        return results

    def run_algorithm_comparison(
        self,
        algorithms: List[str],
        data_specs: List[TestDataSpec],
        test_cases: Optional[List[str]] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, List[TestMetrics]]:
        """Run comprehensive algorithm comparison."""

        if test_cases is None:
            test_cases = list(self.test_cases.keys())

        results = {}

        for algorithm_name in algorithms:
            logger.info(f"Testing algorithm: {algorithm_name}")
            algorithm_results = []

            # Create algorithm instance
            try:
                if algorithm_name in self.factory.registry.get_available_configs():
                    algorithm = self.factory.create_from_preset(algorithm_name)
                else:
                    algorithm = self.factory.create_auto_configured("balanced")
            except Exception as e:
                logger.error(f"Failed to create algorithm {algorithm_name}: {e}")
                continue

            # Test on each data specification
            for data_spec in data_specs:
                # Test with each test case
                for test_case_name in test_cases:
                    if test_case_name not in self.test_cases:
                        logger.warning(f"Unknown test case: {test_case_name}")
                        continue

                    test_case = self.test_cases[test_case_name]

                    try:
                        metrics = self._run_single_test(algorithm, test_case, data_spec)
                        metrics.algorithm_name = algorithm_name
                        metrics.test_scenario = test_case_name
                        metrics.data_spec = data_spec.to_dict()

                        algorithm_results.append(metrics)
                        self.test_results.append(metrics)

                    except Exception as e:
                        logger.error(f"Test failed for {algorithm_name}/{test_case_name}: {e}")

            results[algorithm_name] = algorithm_results

        # Save results if requested
        if save_results and output_dir:
            self._save_test_results(results, output_dir)

        return results

    def run_performance_benchmark(
        self,
        algorithms: List[str],
        frame_counts: List[int] = None,
        iterations: int = 3
    ) -> pd.DataFrame:
        """Run performance benchmark across different frame counts."""

        if frame_counts is None:
            frame_counts = [16, 32, 64, 128]

        benchmark_results = []

        for algorithm_name in algorithms:
            logger.info(f"Benchmarking algorithm: {algorithm_name}")

            try:
                algorithm = self.factory.create_from_preset(algorithm_name)
            except:
                algorithm = self.factory.create_auto_configured("balanced")

            for frame_count in frame_counts:
                data_spec = TestDataSpec(num_frames=frame_count, device="cpu")
                test_case = self.test_cases['synthetic']

                times = []
                for iteration in range(iterations):
                    metrics = self._run_single_test(algorithm, test_case, data_spec)
                    times.append(metrics.execution_time)

                benchmark_results.append({
                    'algorithm': algorithm_name,
                    'frame_count': frame_count,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                })

        return pd.DataFrame(benchmark_results)

    def _run_single_test(
        self,
        algorithm: Any,
        test_case: BaseTestCase,
        data_spec: TestDataSpec
    ) -> TestMetrics:
        """Run a single test and collect metrics."""

        # Setup
        test_case.setup()

        try:
            # Generate test data
            vggt_predictions, visual_features = test_case.generate_test_data(data_spec)

            # Monitor performance
            start_time = time.time()
            start_memory = self._get_memory_usage()

            # Run algorithm
            result = algorithm(vggt_predictions, visual_features)

            # Collect performance metrics
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_usage = end_memory - start_memory

            # Evaluate quality
            quality_metrics = test_case.evaluate_result(result, data_spec)

            # Create test metrics
            metrics = TestMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                selection_diversity=quality_metrics.get('selection_diversity', 0.0),
                temporal_distribution=quality_metrics.get('temporal_distribution', 0.0),
                spatial_coverage=quality_metrics.get('spatial_coverage', 0.0),
                algorithm_specific=quality_metrics
            )

            return metrics

        finally:
            test_case.teardown()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _register_builtin_test_cases(self):
        """Register built-in test cases."""

        # Synthetic test case
        synthetic_test = SyntheticTestCase(
            "synthetic",
            "Synthetic data generation test case"
        )
        self.register_test_case(synthetic_test)

    def _get_scenario_config(self, scenario: TestScenario) -> Dict[str, Any]:
        """Get configuration for a test scenario."""

        if scenario == TestScenario.QUICK_SMOKE:
            return {
                'algorithms': ['greedy_coverage'],
                'data_specs': [TestDataSpec(num_frames=16, height=32, width=32)],
                'test_cases': ['synthetic']
            }

        elif scenario == TestScenario.PERFORMANCE_BENCHMARK:
            return {
                'algorithms': ['greedy_coverage', 'enhanced_submodular'],
                'data_specs': [
                    TestDataSpec(num_frames=16),
                    TestDataSpec(num_frames=32),
                    TestDataSpec(num_frames=64)
                ],
                'test_cases': ['synthetic']
            }

        elif scenario == TestScenario.QUALITY_ANALYSIS:
            return {
                'algorithms': ['greedy_coverage', 'enhanced_submodular'],
                'data_specs': [
                    TestDataSpec(scene_complexity=0.2, temporal_coherence=0.8),
                    TestDataSpec(scene_complexity=0.8, temporal_coherence=0.3),
                    TestDataSpec(scene_complexity=0.5, temporal_coherence=0.5)
                ],
                'test_cases': ['synthetic']
            }

        else:  # Default
            return {
                'algorithms': ['greedy_coverage', 'enhanced_submodular'],
                'data_specs': [TestDataSpec()],
                'test_cases': ['synthetic']
            }

    def _save_test_results(self, results: Dict[str, List[TestMetrics]], output_dir: Path):
        """Save test results to files."""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_results = {}
        for algorithm, metrics_list in results.items():
            json_results[algorithm] = [metrics.to_dict() for metrics in metrics_list]

        with open(output_dir / "test_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save as CSV for easy analysis
        all_metrics = []
        for algorithm, metrics_list in results.items():
            all_metrics.extend(metrics_list)

        df_data = [metrics.to_dict() for metrics in all_metrics]
        df = pd.DataFrame(df_data)
        df.to_csv(output_dir / "test_results.csv", index=False)

        logger.info(f"Test results saved to: {output_dir}")

    def generate_report(self, results: Dict[str, List[TestMetrics]], output_path: Path):
        """Generate comprehensive test report."""

        # Create summary statistics
        summary = {}
        for algorithm, metrics_list in results.items():
            if not metrics_list:
                continue

            times = [m.execution_time for m in metrics_list]
            diversity = [m.selection_diversity for m in metrics_list]

            summary[algorithm] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'mean_diversity': np.mean(diversity),
                'num_tests': len(metrics_list)
            }

        # Generate plots if matplotlib available
        try:
            self._create_performance_plots(results, output_path.parent)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")

        # Save summary
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"Test report saved to: {output_path}")

    def _create_performance_plots(self, results: Dict[str, List[TestMetrics]], output_dir: Path):
        """Create performance visualization plots."""

        # Performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        algorithms = list(results.keys())
        times = [np.mean([m.execution_time for m in results[alg]]) for alg in algorithms]
        diversity = [np.mean([m.selection_diversity for m in results[alg]]) for alg in algorithms]

        ax1.bar(algorithms, times)
        ax1.set_title('Average Execution Time')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(algorithms, diversity)
        ax2.set_title('Average Selection Diversity')
        ax2.set_ylabel('Diversity Score')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    # Test flexible framework
    framework = FlexibleTestFramework()

    # Run quick smoke test
    results = framework.run_test_scenario(
        TestScenario.QUICK_SMOKE,
        algorithms=['greedy_coverage'],
        save_results=False
    )

    print("Test Results:")
    for algorithm, metrics_list in results.items():
        for metrics in metrics_list:
            print(f"{algorithm}: {metrics.execution_time:.3f}s, diversity: {metrics.selection_diversity:.3f}")

    print("Flexible test framework test completed successfully!")