import numpy as np

from src.data.trajectory_dataset import VLNTrajectoryDataset


def _dataset_stub() -> VLNTrajectoryDataset:
    dataset = object.__new__(VLNTrajectoryDataset)
    dataset.system2_min_pixel_goal_len = 3
    dataset.sft_include_turns = True
    dataset.sft_include_forward = False
    dataset.system2_stop_path_radius_m = 3.0
    dataset.system2_near_stop_hard_negative_oversample = 2
    dataset.system2_near_stop_hard_negative_min_path_m = 0.0
    dataset.system2_near_stop_hard_negative_max_path_m = 0.0
    dataset.system2_near_stop_hard_negative_min_goal_distance_m = 4.0
    dataset.system2_near_stop_hard_negative_max_goal_distance_m = 18.0
    dataset._system2_stop_hard_negative_distance = "goal_euclidean"
    return dataset


def test_result_has_system2_sft_target_accepts_side_pano_goal_without_legacy_goal():
    dataset = _dataset_stub()
    result = {
        "discrete_action": 1,
        "is_stop": 0.0,
        "pano_sample_kind": "pixel",
        "pano_view_id": "right",
        "pano_pixel_goal": [211, 128],
        "pano_pixel_goal_relative_len": 5,
    }

    assert dataset._result_has_system2_sft_target(result)


def test_result_has_system2_sft_target_rejects_too_short_pano_goal():
    dataset = _dataset_stub()
    result = {
        "discrete_action": 1,
        "is_stop": 0.0,
        "pano_sample_kind": "pixel",
        "pano_view_id": "left",
        "pano_pixel_goal": [32, 128],
        "pano_pixel_goal_relative_len": 2,
    }

    assert not dataset._result_has_system2_sft_target(result)


def test_result_has_system2_sft_target_handles_structured_stop_and_turn():
    dataset = _dataset_stub()
    assert dataset._result_has_system2_sft_target({
        "pano_sample_kind": "stop",
        "pano_view_id": "view_stop",
        "discrete_action": 0,
        "is_stop": 1.0,
    })
    assert dataset._result_has_system2_sft_target({
        "pano_sample_kind": "turn",
        "pano_view_id": "view_turn",
        "discrete_action": 2,
        "is_stop": 0.0,
    })


def test_system1_goal_length_prefers_structured_pano_goal():
    result = {
        "pano_sample_kind": "pixel",
        "pano_pixel_goal": [211, 128],
        "pano_pixel_goal_relative_len": 7,
        "pixel_goal": [120, 96],
        "pixel_goal_relative_len": 19,
    }

    assert VLNTrajectoryDataset._system1_goal_relative_len(result) == 7


def test_near_stop_hard_negative_repeats_only_nonterminal_pixel_samples():
    dataset = _dataset_stub()

    assert dataset._near_stop_hard_negative_repeat(
        kind="pixel", remaining_path_m=7.0, endpoint_distance_m=4.0,
    ) == 2
    assert dataset._near_stop_hard_negative_repeat(
        kind="pixel", remaining_path_m=7.0, endpoint_distance_m=18.1,
    ) == 0


def test_stop_metric_margin_is_neither_positive_nor_negative():
    dataset = _dataset_stub()
    actions = np.ones(10, dtype=np.int32)

    assert dataset._internnav_sft_frame_kind(
        0, None, 5, 10, actions, remaining_path_m=2.9, endpoint_distance_m=2.0,
    ) == "stop"
    assert dataset._internnav_sft_frame_kind(
        0, None, 5, 10, actions, remaining_path_m=3.5, endpoint_distance_m=3.5,
    ) is None
    assert dataset._near_stop_hard_negative_repeat(
        kind="stop", remaining_path_m=4.0, endpoint_distance_m=4.0,
    ) == 0


def test_remaining_path_distance_ignores_in_place_turns():
    poses = []
    for xyz in ((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 2)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = xyz
        poses.append(pose)

    remaining = VLNTrajectoryDataset._remaining_path_distances(poses, 4)

    np.testing.assert_allclose(remaining, [3.0, 3.0, 2.0, 0.0])


def test_endpoint_distance_makes_only_geometrically_safe_negatives():
    poses = []
    for xyz in ((0, 0, 0), (5, 0, 0), (0.5, 0, 0)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = xyz
        poses.append(pose)

    remaining = VLNTrajectoryDataset._remaining_path_distances(poses, 3)
    endpoint = VLNTrajectoryDataset._endpoint_euclidean_distances(poses, 3)

    assert remaining[0] > 9.0
    assert endpoint[0] == 0.5


def test_metric_stop_override_replaces_pixel_target_only():
    dataset = _dataset_stub()
    dataset._system2_sft_kind_override = {4: "stop"}
    result = {
        "pano_view_id": "front",
        "pano_sample_kind": "pixel",
        "pano_pixel_goal": [10, 20],
        "pano_pixel_goal_relative_len": 7,
        "discrete_action": 1,
        "is_stop": 0.0,
    }

    dataset._apply_system2_sft_label_override(result, 4)

    assert result["pano_view_id"] == "view_stop"
    assert result["pano_sample_kind"] == "stop"
    assert result["pano_pixel_goal"] is None
    assert "pano_pixel_goal_relative_len" not in result
    assert result["discrete_action"] == 1
    assert result["is_stop"] == 0.0


def test_clip_subset_remaps_sample_metadata_without_crossing_clips():
    dataset = _dataset_stub()
    dataset.sample_index = [(0, 4), (1, 8), (0, 12), (2, 16)]
    dataset._sample_subsequence_range = {
        0: (0, 20),
        1: (0, 20),
        2: (0, 20),
        3: (0, 20),
    }
    dataset._system2_sft_kind_override = {2: "stop", 3: "stop"}
    dataset._epoch = 0

    subset = dataset.subset_by_clip_indices({0, 2})

    assert subset.sample_index == [(0, 4), (0, 12), (2, 16)]
    assert subset._sample_subsequence_range == {
        0: (0, 20),
        1: (0, 20),
        2: (0, 20),
    }
    assert subset._system2_sft_kind_override == {1: "stop", 2: "stop"}
    assert dataset.sample_index == [(0, 4), (1, 8), (0, 12), (2, 16)]
