"""Contract tests for the joint native InternNav + heatmap collator."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT / "src/data"
_MODELS_ROOT = _REPO_ROOT / "src/models"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load only the leaf modules used by this unit test. Importing src.models
# eagerly pulls optional accelerator/model stacks into otherwise CPU-only
# collator tests.
_INPUT_MODULE = "src.models.heatmap.input_constructor"
_INTEGRATION_MODULE = "src.models.qwen2_5_vl.integration"
_SENTINEL = object()
_SAVED_MODULES = {
    name: sys.modules.get(name, _SENTINEL)
    for name in (_INPUT_MODULE, _INTEGRATION_MODULE)
}
_PACKAGE_NAME = "_internnav_heatmap_control_collator_testpkg"
try:
    _load_module(
        _INPUT_MODULE,
        _MODELS_ROOT / "heatmap/input_constructor.py",
    )
    _integration = types.ModuleType(_INTEGRATION_MODULE)
    _integration.TRAJ_TOKEN_INDEX = 151667
    sys.modules[_INTEGRATION_MODULE] = _integration

    _package = types.ModuleType(_PACKAGE_NAME)
    _package.__path__ = [str(_DATA_ROOT)]
    _package.__package__ = _PACKAGE_NAME
    sys.modules[_PACKAGE_NAME] = _package
    _load_module(
        f"{_PACKAGE_NAME}._constants",
        _DATA_ROOT / "_constants.py",
    )
    _load_module(
        f"{_PACKAGE_NAME}.panoramic_tokenized_collator",
        _DATA_ROOT / "panoramic_tokenized_collator.py",
    )
    _load_module(
        f"{_PACKAGE_NAME}.single_view_heatmap_collator",
        _DATA_ROOT / "single_view_heatmap_collator.py",
    )
    _joint_module = _load_module(
        f"{_PACKAGE_NAME}.internnav_heatmap_control_collator",
        _DATA_ROOT / "internnav_heatmap_control_collator.py",
    )
finally:
    for _name, _previous in _SAVED_MODULES.items():
        if _previous is _SENTINEL:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

InternNavHeatmapControlCollator = (
    _joint_module.InternNavHeatmapControlCollator
)
TRAJ_TOKEN_INDEX = 151667
_DIRECTION_ORDER = ("front", "right", "back", "left")
_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"
    truncation_side = "right"

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [ord(character) + 3 for character in text]


class _RecordingImageProcessor:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []
        self.image_means: list[int] = []

    def __call__(
        self,
        *,
        images: list[Any],
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.calls.append(list(images))
        self.image_means = [
            int(np.asarray(image, dtype=np.uint8).mean())
            for image in images
        ]
        count = len(images)
        return {
            "pixel_values": torch.arange(
                count * 5,
                dtype=torch.float32,
            ).reshape(count, 5),
            "image_grid_thw": torch.tensor(
                [[1, 2, 3]] * count,
                dtype=torch.long,
            ),
        }


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _RecordingImageProcessor()
        self.last_messages_batch: list[list[dict[str, Any]]] | None = None
        self.last_rendered_text: list[str] | None = None
        self.native_images: list[Any] = []
        self.native_image_means: list[int] = []
        self.chat_calls = 0

    def apply_chat_template(
        self,
        messages_batch: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self.chat_calls += 1
        self.last_messages_batch = messages_batch
        rows: list[list[int]] = []
        for messages in messages_batch:
            row: list[int] = []
            for message in messages:
                row.extend(
                    self.tokenizer.encode(f"<{message['role']}>")
                )
                for item in message["content"]:
                    if item["type"] == "text":
                        text = item["text"]
                    else:
                        text = f"<{item['type']}>"
                    row.extend(self.tokenizer.encode(text))
                if message["role"] == "assistant":
                    row.append(self.tokenizer.eos_token_id)
            if kwargs.get("add_generation_prompt", False):
                row.extend(self.tokenizer.encode("<assistant>"))
            rows.append(row)

        max_length = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            padding = max_length - len(row)
            input_ids.append(
                [self.tokenizer.pad_token_id] * padding + row
            )
            attention_mask.append(
                [0] * padding + [1] * len(row)
            )
        if kwargs.get("tokenize", True) is False:
            self.last_rendered_text = [
                f"native-rendered-prompt-{index}" for index in range(len(rows))
            ]
            return self.last_rendered_text
        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Any],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is True
        self.native_images = list(images)
        self.native_image_means = [
            int(np.asarray(image, dtype=np.uint8).mean())
            for image in images
        ]
        rows = [self.tokenizer.encode(value) for value in text]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor([
            [self.tokenizer.pad_token_id] * (width - len(row)) + row
            for row in rows
        ], dtype=torch.long)
        attention_mask = torch.tensor([
            [0] * (width - len(row)) + [1] * len(row)
            for row in rows
        ], dtype=torch.long)
        count = len(images)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.arange(count * 5).reshape(count, 5).float(),
            "image_grid_thw": torch.tensor([[1, 2, 3]] * count),
        }


def _chw(value: float, size: int = 4) -> torch.Tensor:
    return torch.full(
        (3, size, size),
        value,
        dtype=torch.float32,
    )


def _hwc(value: float, size: int = 3) -> torch.Tensor:
    return torch.full(
        (size, size, 3),
        value,
        dtype=torch.float32,
    )


def _sample(
    *,
    source_type: str,
    history_values: list[float],
    history_valid_mask: list[bool],
    expert_layout: bool,
) -> dict[str, Any]:
    history_count = len(history_values)
    history_frames = torch.stack(
        [_chw(value) for value in history_values],
        dim=0,
    )
    current_frame = _chw(0.70 + 0.01 * history_count)
    current_views = torch.stack(
        [
            current_frame,
            _chw(0.76),
            _chw(0.77),
            _chw(0.78),
        ],
        dim=0,
    )
    history_panoramas = torch.stack(
        [
            torch.stack(
                [
                    history_frames[index],
                    _chw(0.30 + 0.01 * index),
                    _chw(0.40 + 0.01 * index),
                    _chw(0.50 + 0.01 * index),
                ],
                dim=0,
            )
            for index in range(history_count)
        ],
        dim=0,
    )

    horizon = 5
    canonical_trajectory = torch.arange(
        horizon * 3,
        dtype=torch.float32,
    ).reshape(horizon, 3)
    if expert_layout:
        trajectory = torch.stack(
            (canonical_trajectory, canonical_trajectory + 100.0),
            dim=0,
        )
        traj_images = torch.stack(
            (_hwc(0.11), _hwc(0.22), _hwc(0.33)),
            dim=0,
        )
        trajectory_valid = torch.tensor([1.0, 0.0])
    else:
        trajectory = canonical_trajectory + 200.0
        traj_images = torch.stack(
            (_hwc(0.44), _hwc(0.55)),
            dim=0,
        )
        trajectory_valid = torch.tensor(0.75)

    return {
        "sample_key": f"{source_type}-sample",
        "source_type": source_type,
        "text": f"navigate from {source_type}",
        "history_frames": history_frames,
        "current_frame": current_frame,
        "current_views": current_views,
        "history_panoramas": history_panoramas,
        "lookdown_frame": _chw(0.90),
        "history_rel_poses": torch.arange(
            history_count * 4,
            dtype=torch.float32,
        ).reshape(history_count, 4),
        "history_frame_ids": torch.arange(
            10,
            10 + history_count,
            dtype=torch.long,
        ),
        "history_age_steps": torch.arange(
            history_count,
            0,
            -1,
            dtype=torch.long,
        ),
        "history_valid_mask": torch.tensor(
            history_valid_mask,
            dtype=torch.bool,
        ),
        "history_mask": torch.tensor(
            history_valid_mask,
            dtype=torch.float32,
        ),
        "heatmap_direction_order": _DIRECTION_ORDER,
        "history_pose_convention": _POSE_CONVENTION,
        "trajectory": trajectory,
        "trajectory_valid": trajectory_valid,
        "traj_images": traj_images,
        "pixel_goal": [12, 23],
        "action": torch.zeros(2, dtype=torch.float32),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "progress": 0.25,
    }


def test_panorama_samples_still_use_native_stage2_prompt_and_separate_images() -> None:
    processor = _FakeProcessor()
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
    )
    sample = _sample(
        source_type="expert",
        history_values=[0.10, 0.20],
        history_valid_mask=[True, True],
        expert_layout=True,
    )
    assert "current_views" in sample
    assert "history_panoramas" in sample

    output = collator([sample])

    assert processor.chat_calls == 1
    assert processor.last_messages_batch is not None
    messages = processor.last_messages_batch[0]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    first_user_types = [
        item["type"] for item in messages[0]["content"]
    ]
    assert "video" not in first_user_types
    assert first_user_types.count("image") == 3
    assert [
        item["type"] for item in messages[2]["content"]
    ] == ["text", "image", "text"]
    assert messages[-1]["content"][0]["text"] == "23 12"
    assert output["native_system2_num_histories"] == [2]
    assert output["pano_inputs"]["image_grid_thw"].shape == (4, 3)
    assert len(processor.native_images) == 4
    assert "video_grid_thw" not in output["pano_inputs"]
    assert output["pano_num_histories"] == [0]
    assert output["pano_text_anchor_positions"] is None
    assert "current_views" not in output

    # The heatmap path sees K independent history stills plus one current
    # still; that namespace excludes the native lookdown image.
    assert len(processor.image_processor.calls) == 1
    assert len(processor.image_processor.calls[0]) == 3
    assert output["heatmap_single_view_num_histories"] == [2]
    assert output[
        "heatmap_single_view_inputs"
    ]["image_grid_thw"].shape == (3, 3)
    assert torch.all(
        output["pano_inputs"]["input_ids"][0, -4:]
        == TRAJ_TOKEN_INDEX
    )


def test_mixed_sources_canonicalize_and_preserve_explicit_history_mask() -> None:
    processor = _FakeProcessor()
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
    )
    expert = _sample(
        source_type="expert",
        history_values=[0.10, 0.20, 0.30],
        history_valid_mask=[True, False, True],
        expert_layout=True,
    )
    dagger = _sample(
        source_type="dagger",
        history_values=[0.40],
        history_valid_mask=[True],
        expert_layout=False,
    )
    expected_expert_trajectory = expert["trajectory"][0].clone()
    expected_dagger_trajectory = dagger["trajectory"].clone()
    expected_expert_image = expert["traj_images"][0].clone()
    expected_dagger_images = dagger["traj_images"].clone()

    output = collator([expert, dagger])

    assert output["native_system2_num_histories"] == [2, 1]
    assert output["pano_inputs"]["image_grid_thw"].shape == (7, 3)
    assert 51 not in processor.native_image_means
    assert [
        sum(item["type"] == "image" for item in messages[0]["content"])
        for messages in processor.last_messages_batch
    ] == [3, 2]

    assert output["trajectory"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["trajectory"][0],
        expected_expert_trajectory,
    )
    torch.testing.assert_close(
        output["trajectory"][1],
        expected_dagger_trajectory,
    )
    assert output["traj_images"].shape == (2, 2, 3, 3, 3)
    torch.testing.assert_close(
        output["traj_images"][0, 0],
        expected_expert_image,
    )
    torch.testing.assert_close(
        output["traj_images"][0, 1],
        expected_expert_image,
    )
    torch.testing.assert_close(
        output["traj_images"][1],
        expected_dagger_images,
    )
    torch.testing.assert_close(
        output["trajectory_valid"],
        torch.tensor([1.0, 0.75]),
    )

    expected_mask = torch.tensor(
        [[True, False, True], [True, False, False]]
    )
    assert torch.equal(
        output["history_valid_mask"],
        expected_mask,
    )
    assert torch.equal(
        output["heatmap_control_history_mask"],
        expected_mask,
    )
    assert torch.equal(
        output["history_mask"],
        expected_mask.float(),
    )
    assert output["history_frames"].shape[:2] == (2, 3)
    assert output["history_rel_poses"].shape == (2, 3, 4)
    assert output["history_age_steps"].tolist() == [
        [3, 2, 1],
        [1, 0, 0],
    ]
    assert output["history_frame_ids"].tolist() == [
        [10, 11, 12],
        [10, 0, 0],
    ]

    # K values are 3 and 1, so the frozen heatmap branch receives
    # (3 + 1) + (1 + 1) = 6 independent still-image groups.
    assert output["heatmap_single_view_num_histories"] == [3, 1]
    assert output[
        "heatmap_single_view_inputs"
    ]["image_grid_thw"].shape == (6, 3)
    assert len(processor.image_processor.calls[0]) == 6

    # Heatmaps are generated online by the frozen branch; no target is
    # required or emitted by this joint collator.
    assert "heatmap" not in output
    assert "gt_visibility" not in output


def test_all_invalid_history_is_excluded_only_from_native_system2() -> None:
    processor = _FakeProcessor()
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
    )
    all_invalid = _sample(
        source_type="dagger",
        history_values=[0.20, 0.30],
        history_valid_mask=[False, False],
        expert_layout=False,
    )
    valid = _sample(
        source_type="dagger",
        history_values=[0.40],
        history_valid_mask=[True],
        expert_layout=False,
    )

    output = collator([all_invalid, valid])

    expected_mask = torch.tensor([
        [False, False],
        [True, False],
    ])
    assert torch.equal(output["heatmap_control_history_mask"], expected_mask)
    assert output["native_system2_num_histories"] == [0, 1]
    assert output["pano_inputs"]["image_grid_thw"].shape == (5, 3)
    assert len(processor.native_images) == 5
    assert 51 not in processor.native_image_means
    assert 76 not in processor.native_image_means
    assert output["heatmap_single_view_num_histories"] == [2, 1]
    assert output[
        "heatmap_single_view_inputs"
    ]["image_grid_thw"].shape == (5, 3)
