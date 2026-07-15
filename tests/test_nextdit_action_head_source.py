from pathlib import Path


def test_nextdit_action_head_has_no_stale_scheduler_references():
    source = (Path(__file__).resolve().parents[1] / "src" / "models" / "action" / "nextdit_action_head.py").read_text(
        encoding="utf-8"
    )

    assert "noise_self" not in source
    assert "hasattr(scheduler" not in source


def test_nextdit_inference_noise_accepts_an_explicit_local_generator():
    source = (Path(__file__).resolve().parents[1] / "src" / "models" / "action" / "nextdit_action_head.py").read_text(
        encoding="utf-8"
    )

    assert "generator=None" not in source
    assert "generator=generator" in source
