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
