import math

from scripts.evaluation.navigation_metrics import aggregate_navigation_metrics


def test_aggregate_navigation_metrics_matches_expected_means():
    result = aggregate_navigation_metrics(
        successes=[1.0, 0.0],
        spls=[0.5, 0.0],
        oracle_successes=[1.0, 1.0],
        navigation_errors=[2.0, 6.0],
    )

    assert result == {
        "SR": 0.5,
        "SPL": 0.25,
        "OS": 1.0,
        "NE": 4.0,
        "total_episodes": 2,
    }


def test_aggregate_navigation_metrics_sanitizes_spl_and_nonfinite_ne():
    result = aggregate_navigation_metrics(
        successes=[0.0, 1.0, 1.0],
        spls=[float("nan"), float("inf"), 0.75],
        oracle_successes=[0.0, 1.0, 1.0],
        navigation_errors=[float("nan"), float("inf"), 3.0],
    )

    assert math.isclose(result["SPL"], 0.25)
    assert result["NE"] == 3.0


def test_aggregate_navigation_metrics_handles_empty_input():
    assert aggregate_navigation_metrics([], [], [], []) == {
        "SR": 0.0,
        "SPL": 0.0,
        "OS": 0.0,
        "NE": 0.0,
        "total_episodes": 0,
    }
