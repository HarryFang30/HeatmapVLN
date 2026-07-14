from types import SimpleNamespace

import pytest
import torch
from scripts.tools.diagnose_pose_free_query_separation import (
    aggregate_samples,
    analyze_sample,
    assert_blank_pair_chains_identical,
    capture_matcher_inputs,
    output_peak_diversity,
    pairwise_metrics,
    regroup_matcher_captures,
    validate_args,
)

from src.models.heatmap.pose_free_matching import PoseFreeHistoryMatcher


def _blank_transformed():
    return {
        "history_panoramas": torch.zeros(4, 4, 3, 2, 2),
        "history_frames": torch.zeros(4, 3, 2, 2),
        "current_views": torch.zeros(4, 3, 2, 2),
        "current_frame": torch.zeros(3, 2, 2),
    }


def test_pairwise_metrics_reports_known_query_separation():
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    metrics = pairwise_metrics(vectors)

    assert metrics["vector_shape"] == [4, 2]
    assert metrics["pairwise_cosine"][0] == pytest.approx([1.0, 0.0, -1.0, 0.0])
    assert metrics["off_diagonal"]["cosine_mean"] == pytest.approx(-1 / 3)
    assert metrics["off_diagonal"]["euclidean_min"] == pytest.approx(2**0.5)
    assert metrics["off_diagonal"]["euclidean_max"] == pytest.approx(2.0)


def test_output_peak_diversity_uses_circular_panorama_coordinates():
    visibility = torch.full((4, 4), -10.0)
    heatmaps = torch.zeros(4, 4, 8, 8)
    peaks = [(0, 1, 2), (1, 3, 4), (2, 5, 6), (3, 7, 1)]
    for history, (view, x, y) in enumerate(peaks):
        visibility[history, view] = 10.0
        heatmaps[history, view, y, x] = 1.0

    diversity = output_peak_diversity(visibility, heatmaps)

    assert diversity["unique_peak_count"] == 4
    assert diversity["all_histories_have_distinct_peaks"] is True
    assert diversity["unique_selected_view_count"] == 4
    assert [item["panorama_x"] for item in diversity["peaks"]] == [1, 11, 21, 31]
    # Circular x distance between panorama_x=1 and 31 is only two pixels.
    assert diversity["minimum_pairwise_peak_distance"] == pytest.approx(5**0.5)


def test_matcher_hook_captures_real_raw_and_projected_isolated_queries():
    torch.manual_seed(9)
    matcher = PoseFreeHistoryMatcher(
        current_dim=6,
        query_dim=5,
        match_dim=3,
        heatmap_size=(8, 8),
        visibility_hidden_dim=4,
    ).eval()
    current = torch.randn(4, 4, 2, 2, 6)
    # The repeated current contract should be exactly observable in capture.
    current[:] = current[0]
    queries = torch.randn(4, 1, 5)

    with capture_matcher_inputs(matcher) as captures:
        outputs = [
            matcher(
                current_patches=current[index : index + 1],
                history_queries=queries[index : index + 1],
            )
            for index in range(4)
        ]

    assert all(output["heatmaps"].shape == (1, 1, 4, 8, 8) for output in outputs)
    assert len(captures) == 4
    capture = regroup_matcher_captures(captures)
    assert capture["current_patches_shape"] == [4, 4, 2, 2, 6]
    assert capture["history_queries_shape"] == [4, 1, 5]
    assert capture["per_call_current_patches_shapes"] == [[1, 4, 2, 2, 6]] * 4
    torch.testing.assert_close(capture["raw_queries"], queries[:, 0])
    expected_projection = matcher.query_projection(matcher.query_norm(queries[:, 0]))
    torch.testing.assert_close(capture["projected_queries"], expected_projection)
    assert capture["current_chain_max_abs_difference_from_chain0"] == [0.0] * 4


def test_blank_intervention_requires_all_four_transformed_chains_to_be_exactly_identical():
    transformed = _blank_transformed()

    contract = assert_blank_pair_chains_identical(transformed)

    assert set(contract) == {
        "video_frames",
        "current_observation",
        "current_views",
        "history_panoramas",
    }
    assert all(item["four_chains_bitwise_identical"] for item in contract.values())

    transformed["history_panoramas"][2, 1, 0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="history_panoramas"):
        assert_blank_pair_chains_identical(transformed)


def _synthetic_capture_and_prediction():
    raw = torch.eye(4)
    projected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    capture = {
        "current_patches_shape": [4, 4, 2, 2, 6],
        "history_queries_shape": [4, 1, 4],
        "raw_queries": raw,
        "projected_queries": projected,
        "pooled_current_patches": torch.ones(4, 6),
        "current_chain_max_abs_difference_from_chain0": [0.0] * 4,
        "current_chain_mean_abs_difference_from_chain0": [0.0] * 4,
    }
    visibility = torch.full((1, 4, 4), -10.0)
    heatmaps = torch.zeros(1, 4, 4, 8, 8)
    for history in range(4):
        visibility[0, history, history] = 10.0
        heatmaps[0, history, history, history + 1, history + 2] = 1.0
    return capture, {"visibility": visibility, "heatmaps": heatmaps}


def test_sample_and_aggregate_reports_query_and_output_diversity():
    capture, prediction = _synthetic_capture_and_prediction()
    sample = analyze_sample(capture, prediction, sample_id="sample-1")
    aggregate = aggregate_samples([sample, sample])

    assert sample["captured_input_contract"]["histories_per_chain"] == 1
    assert sample["captured_input_contract"]["pose_slot_frame_model_inputs"] is False
    assert sample["output_peak_diversity"]["unique_peak_count"] == 4
    assert aggregate["samples"] == 2
    assert aggregate["fraction_with_four_distinct_peaks"] == 1.0
    assert aggregate["maximum_current_patch_cross_chain_abs_difference"] == 0.0
    assert aggregate["projected_query_off_diagonal_cosine_mean"] == pytest.approx(-1 / 3)


def test_validation_rejects_empty_diagnostic():
    validate_args(SimpleNamespace(max_samples=1))
    with pytest.raises(ValueError, match="positive"):
        validate_args(SimpleNamespace(max_samples=0))
    with pytest.raises(ValueError, match="empty"):
        aggregate_samples([])
