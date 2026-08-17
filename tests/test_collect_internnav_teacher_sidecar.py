import json
from types import SimpleNamespace

import torch

import scripts.evaluation.collect_internnav_teacher_sidecar as collector


def test_collect_one_pano_structured_coord_adds_teacher_metadata(monkeypatch):
    monkeypatch.setattr(collector, "_build_first_turn", lambda *args: ([], []))
    monkeypatch.setattr(
        collector,
        "condition_on_pano_coord",
        lambda *args: (
            "view: right\npixel: 211 128",
            torch.tensor([[1, 2]]),
            SimpleNamespace(),
            2,
            [211, 128],
            [128, 211],
            "right",
        ),
    )
    monkeypatch.setattr(
        collector,
        "_run_system1",
        lambda *args, **kwargs: {
            "actions8": [1, 0, 0, 0, 0, 0, 0, 0],
            "local4": [1, 0, 0, 0],
            "forward_count8": 1,
            "first_action": 1,
        },
    )
    monkeypatch.setattr(collector, "_sample_metadata", lambda *args: {})
    monkeypatch.setattr(collector, "sidecar_alignment_metadata", lambda *args: {})

    rec = collector._collect_one(
        0,
        {
            "pano_sample_kind": "pixel",
            "pano_view_id": "right",
            "pano_pixel_goal": [211, 128],
        },
        dataset=object(),
        model=object(),
        processor=object(),
        traj_to_actions_fn=lambda _trajectory: [1],
        device=torch.device("cpu"),
        dtype=torch.float32,
        action_scale=4.0,
        args=SimpleNamespace(
            seed=42,
            shard_index=0,
            coord_source="pano",
            two_turn_lookdown=True,
            skip_system1_errors=False,
            include_text=False,
        ),
        rng=object(),
    )

    assert rec["teacher"]["pano_view_id"] == "right"
    assert rec["teacher"]["structured_assistant_text"] == "view: right\npixel: 211 128"


def test_native_coordinate_conditioning_writes_yx_text(monkeypatch):
    monkeypatch.setattr(
        collector,
        "_build_second_turn",
        lambda first_messages, first_images, *_args: (first_messages, first_images),
    )

    class Inputs:
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2, 3]])

        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            return " | ".join(
                item["text"]
                for message in messages
                for item in message.get("content", [])
                if item.get("type") == "text"
            )

        def __call__(self, **_kwargs):
            return Inputs()

    output, _ids, _inputs, _prompt_len, coord_uv, goal_yx = (
        collector._condition_on_dataset_coord(
            Processor(),
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            [],
            {"aligned_native_pixel_goal_uv": [151, 202]},
            SimpleNamespace(),
            object(),
            torch.device("cpu"),
            pixel_goal_key="aligned_native_pixel_goal_uv",
        )
    )

    assert output == "202 151"
    assert coord_uv == [151, 202]
    assert goal_yx == [202, 151]


def test_incremental_resume_does_not_skip_shifted_dataset_index(tmp_path):
    old_clip = tmp_path / "old_scene" / "clip_000001"
    new_clip = tmp_path / "new_scene" / "clip_000001"
    old_clip.mkdir(parents=True)
    new_clip.mkdir(parents=True)
    shard = tmp_path / "shard_00.jsonl"
    shard.write_text(
        json.dumps(
            {
                "status": "ok",
                "dataset_index": 0,
                "clip_dir": str(old_clip),
                "current_t": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    done_indices, done_keys = collector._load_done_markers(shard)

    dataset = SimpleNamespace(sample_index=[(0, 5)], clips=[new_clip])
    stable_key = collector._stable_sample_key_from_dataset(dataset, 0)
    should_skip = (stable_key and stable_key in done_keys) or (
        not stable_key and 0 in done_indices
    )

    assert 0 in done_indices
    assert stable_key not in done_keys
    assert not should_skip


def test_stable_tensor_path_avoids_dataset_index_collision(tmp_path):
    old_clip = tmp_path / "old_scene" / "clip_000001"
    new_clip = tmp_path / "new_scene" / "clip_000001"
    old_clip.mkdir(parents=True)
    new_clip.mkdir(parents=True)
    args = SimpleNamespace(
        tensor_output_dir=str(tmp_path / "tensors"),
        tensor_path_mode="stable_key",
        tensor_shard_size=1000,
    )

    old_path = collector._tensor_sidecar_path(
        args,
        0,
        {"clip_dir": str(old_clip), "current_t": 5},
    )
    new_path = collector._tensor_sidecar_path(
        args,
        0,
        {"clip_dir": str(new_clip), "current_t": 5},
    )

    assert old_path != new_path
    assert old_path.parent.name != "shard_00000"
