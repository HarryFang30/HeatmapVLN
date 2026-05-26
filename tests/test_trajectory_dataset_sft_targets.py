from src.data.trajectory_dataset import VLNTrajectoryDataset


def _dataset_stub() -> VLNTrajectoryDataset:
    dataset = object.__new__(VLNTrajectoryDataset)
    dataset.system2_min_pixel_goal_len = 3
    dataset.sft_include_turns = True
    dataset.sft_include_forward = False
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

