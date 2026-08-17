from pathlib import Path

import pytest
import torch

from scripts.training.select_system2_onpolicy_checkpoint import select_checkpoint


def _state() -> dict[str, torch.Tensor]:
    return {f"layer_{index}.lora_A": torch.tensor([float(index)]) for index in range(224)}


def _metrics(*, recall_gain: float, false_gain: float, gap: float, passed: bool):
    return {
        "quality_passed": passed,
        "stop_recall_improvement": recall_gain,
        "false_stop_fpr_improvement": false_gain,
        "positive_false_stop_margin_gap": gap,
    }


def _save_interval(path: Path, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": _state(),
            "training": {"optimizer_steps": step},
            "validation": {"at_checkpoint": metrics},
        },
        path,
    )


def _save_final(path: Path, step: int, metrics: dict) -> None:
    torch.save(
        {
            "trainable_state_dict": _state(),
            "training": {"optimizer_steps": step},
            "validation": {"final": metrics},
        },
        path,
    )


def test_selects_balanced_passing_checkpoint(tmp_path):
    first = tmp_path / "validation_checkpoints" / "step_000100.pth"
    second = tmp_path / "validation_checkpoints" / "step_000200.pth"
    _save_interval(
        first,
        100,
        _metrics(recall_gain=0.1, false_gain=0.2, gap=-5.0, passed=True),
    )
    _save_interval(
        second,
        200,
        _metrics(recall_gain=0.15, false_gain=0.1, gap=-3.0, passed=True),
    )
    _save_final(
        tmp_path / "latest.pth",
        300,
        _metrics(recall_gain=0.3, false_gain=0.0, gap=1.0, passed=False),
    )

    result = select_checkpoint(tmp_path)

    assert result["status"] == "passed"
    assert result["selected_step"] == 100
    assert (tmp_path / "selected.pth").resolve() == first.resolve()


def test_fails_closed_when_no_checkpoint_passes(tmp_path):
    _save_final(
        tmp_path / "latest.pth",
        100,
        _metrics(recall_gain=0.0, false_gain=0.2, gap=-5.0, passed=False),
    )

    with pytest.raises(RuntimeError, match="No System2 continuation checkpoint"):
        select_checkpoint(tmp_path)

    assert not (tmp_path / "selected.pth").exists()
    assert '"status": "failed"' in (tmp_path / "selection.json").read_text()


def test_rejects_incomplete_lora_checkpoint(tmp_path):
    torch.save(
        {
            "trainable_state_dict": {"layer_0.lora_A": torch.ones(1)},
            "training": {"optimizer_steps": 100},
            "validation": {
                "final": _metrics(
                    recall_gain=0.1,
                    false_gain=0.1,
                    gap=-1.0,
                    passed=True,
                )
            },
        },
        tmp_path / "latest.pth",
    )

    with pytest.raises(RuntimeError, match="complete 224-LoRA"):
        select_checkpoint(tmp_path)
