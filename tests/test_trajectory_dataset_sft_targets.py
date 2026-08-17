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


def test_system1_goal_length_prefers_structured_pano_goal():
    result = {
        "pano_sample_kind": "pixel",
        "pano_pixel_goal": [211, 128],
        "pano_pixel_goal_relative_len": 7,
        "pixel_goal": [120, 96],
        "pixel_goal_relative_len": 19,
    }

    assert VLNTrajectoryDataset._system1_goal_relative_len(result) == 7


def test_exact_native_projection_keeps_the_pano_goal_frame():
    dataset = _dataset_stub()
    poses = [object() for _ in range(10)]
    calls = []
    dataset._load_poses_for_direction = lambda _clip_idx, direction: (
        poses if direction == "front_down" else None
    )
    dataset._load_depth = lambda *_args, **_kwargs: "depth"

    def project(current_pose, goal_pose, **kwargs):
        calls.append((current_pose, goal_pose, kwargs))
        return [151, 202]

    dataset._compute_pixel_goal = project
    result = dataset._project_exact_goal_to_native_view(
        clip_idx=0,
        clip_dir=None,
        current_t=4,
        num_frames=10,
        goal_relative_len=3,
        img_size=(256, 256),
    )

    assert result == [151, 202]
    assert calls[0][0] is poses[4]
    assert calls[0][1] is poses[7]


def test_exact_native_projection_does_not_search_for_a_different_waypoint():
    dataset = _dataset_stub()
    poses = [object() for _ in range(10)]
    dataset._load_poses_for_direction = lambda *_args: poses
    dataset._load_depth = lambda *_args, **_kwargs: "depth"
    dataset._compute_pixel_goal = lambda *_args, **_kwargs: None

    assert dataset._project_exact_goal_to_native_view(
        clip_idx=0,
        clip_dir=None,
        current_t=4,
        num_frames=10,
        goal_relative_len=3,
        img_size=(256, 256),
    ) is None
