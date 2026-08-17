"""Contract tests for the front-RGB-only heatmap collator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.single_view_heatmap_collator import SingleViewHeatmapCollator

_DIRECTION_ORDER = ("front", "right", "back", "left")
_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)


class _RecordingImageProcessor:
    def __init__(self) -> None:
        self.image_means: list[int] = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.image_means = [
            int(np.asarray(image, dtype=np.uint8).mean()) for image in images
        ]
        count = len(images)
        return {
            "pixel_values": torch.arange(
                count * 7,
                dtype=torch.float32,
            ).reshape(count, 7),
            "image_grid_thw": torch.tensor(
                [[1, 2, 3]] * count,
                dtype=torch.long,
            ),
        }


class _Processor:
    def __init__(self) -> None:
        self.image_processor = _RecordingImageProcessor()


def _rgb(value: float) -> torch.Tensor:
    return torch.full((3, 8, 8), value, dtype=torch.float32)


def _sample(
    history_values: list[float],
    current_value: float,
    *,
    sample_identity: str | None = None,
) -> dict[str, object]:
    length = len(history_values)
    result = {
        "history_frames": torch.stack([_rgb(value) for value in history_values]),
        "current_frame": _rgb(current_value),
        "history_rel_poses": torch.arange(
            length * 4,
            dtype=torch.float32,
        ).reshape(length, 4),
        "heatmap": torch.arange(
            length * 4 * 6 * 6,
            dtype=torch.float32,
        ).reshape(length, 4, 6, 6),
        "gt_visibility": torch.ones(length, 4),
        "heatmap_direction_order": _DIRECTION_ORDER,
        "history_pose_convention": _POSE_CONVENTION,
        "history_pose_provider": "amb3r_vo_cache",
        "action": torch.zeros(2),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "text": "go",
    }
    if sample_identity is not None:
        result["sample_identity"] = sample_identity
    return result


def test_flattens_front_rgb_then_drops_raw_images_and_pads_histories() -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)

    output = collator(
        [
            _sample([0.0, 0.1], 0.2),
            _sample([0.3], 0.4),
        ]
    )

    # Sample-major order: history..., current; history..., current.
    assert processor.image_processor.image_means == [0, 26, 51, 76, 102]
    assert output["pixel_values"].shape == (5, 7)
    assert output["image_grid_thw"].shape == (5, 3)
    assert output["num_histories"].tolist() == [2, 1]
    assert output["image_offsets"].tolist() == [0, 3, 5]
    assert output["history_mask"].tolist() == [
        [True, True],
        [True, False],
    ]

    assert output["heatmap"].shape == (2, 2, 4, 6, 6)
    assert output["history_rel_poses"].shape == (2, 2, 4)
    assert output["gt_visibility"].shape == (2, 2, 4)
    assert torch.count_nonzero(output["heatmap"][1, 1]) == 0
    assert torch.count_nonzero(output["history_rel_poses"][1, 1]) == 0
    assert torch.count_nonzero(output["gt_visibility"][1, 1]) == 0

    assert output["heatmap_direction_order"] == _DIRECTION_ORDER
    assert output["history_pose_convention"] == _POSE_CONVENTION
    assert output["history_pose_provider"] == "amb3r_vo_cache"
    for forbidden in (
        "history_frames",
        "current_frame",
        "current_views",
        "history_panoramas",
    ):
        assert forbidden not in output


@pytest.mark.parametrize("forbidden_key", ["current_views", "history_panoramas"])
def test_rejects_panoramic_rgb_before_image_processing(forbidden_key: str) -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)
    sample = _sample([0.1], 0.2)
    sample[forbidden_key] = torch.zeros(1)

    with pytest.raises(RuntimeError, match="forbidden panoramic RGB keys"):
        collator([sample])
    assert processor.image_processor.image_means == []


@pytest.mark.parametrize(
    ("metadata_key", "invalid_value", "message"),
    [
        ("heatmap_direction_order", None, "heatmap direction order"),
        (
            "heatmap_direction_order",
            ("front", "left", "back", "right"),
            "heatmap direction order",
        ),
        ("history_pose_convention", None, "history pose convention"),
        ("history_pose_convention", "legacy_plus_z", "history pose convention"),
    ],
)
def test_metadata_contract_fails_closed(
    metadata_key: str,
    invalid_value: object,
    message: str,
) -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)
    sample = _sample([0.1], 0.2)
    if invalid_value is None:
        sample.pop(metadata_key)
    else:
        sample[metadata_key] = invalid_value

    with pytest.raises(ValueError, match=message):
        collator([sample])
    assert processor.image_processor.image_means == []


def test_rejects_mixed_history_pose_providers() -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)
    first = _sample([0.1], 0.2)
    second = _sample([0.1], 0.2)
    second["history_pose_provider"] = "habitat_gt"

    with pytest.raises(ValueError, match="Mixed history_pose_provider"):
        collator([first, second])
    assert processor.image_processor.image_means == []


def test_preserves_sample_identity_as_metadata_only() -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)

    output = collator(
        [
            _sample([0.1], 0.2, sample_identity="scene/clip_000001@000007"),
            _sample([0.3], 0.4, sample_identity="scene/clip_000002@000009"),
        ]
    )

    assert output["sample_identity"] == [
        "scene/clip_000001@000007",
        "scene/clip_000002@000009",
    ]


def test_rejects_partially_missing_sample_identity() -> None:
    processor = _Processor()
    collator = SingleViewHeatmapCollator(processor)
    first = _sample([0.1], 0.2, sample_identity="scene/clip_000001@000007")
    second = _sample([0.3], 0.4)

    with pytest.raises(ValueError, match="sample_identity must be present"):
        collator([first, second])
