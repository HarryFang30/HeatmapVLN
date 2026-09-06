"""Contract tests for the EXP-17 generation evaluator's pure helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "scripts/tools/eval_system2_cognition_prefix.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cognition_eval", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_decisions_are_classified_from_the_native_answer_only() -> None:
    assert tool.decision_of("↓") == ("lookdown", None)
    assert tool.decision_of("←←") == ("turn_left", "left")
    assert tool.decision_of("→") == ("turn_right", "right")
    assert tool.decision_of("STOP") == ("stop", None)
    assert tool.decision_of("460 197") == ("pixel", None)
    assert tool.decision_of("") == ("other", None)
    keep = {"kind": "keep_pixel", "target_texts": ["↓", "460 197"]}
    turn = {"kind": "correct_turn", "target_texts": ["→→"]}
    stop = {"kind": "correct_stop", "target_texts": ["STOP"]}
    assert tool.target_of(keep) == ("lookdown", None)
    assert tool.target_of(turn) == ("turn_right", "right")
    assert tool.target_of(stop) == ("stop", None)
    assert tool.decision_correct(tool.decision_of("↓"), tool.target_of(keep))
    assert not tool.decision_correct(tool.decision_of("←"), tool.target_of(turn))
    assert tool.decision_correct(tool.decision_of("→→→"), tool.target_of(turn))
    assert tool.decision_correct(tool.decision_of("STOP"), tool.target_of(stop))


def test_pose_noise_is_deterministic_and_keeps_the_yaw_unit_norm() -> None:
    poses = np.zeros((1, 3, 4), dtype=np.float32)
    poses[:, :, 2] = 1.0
    ages = np.array([[0, 4, 9]], dtype=np.float32)
    a = tool.perturb_rel_poses_np(poses, translation_m=0.2, rotation_deg=10.0, ages=ages, drift=True, seed=7)
    b = tool.perturb_rel_poses_np(poses, translation_m=0.2, rotation_deg=10.0, ages=ages, drift=True, seed=7)
    assert np.array_equal(a, b)
    assert not np.allclose(a, poses)
    norms = np.sqrt(a[:, :, 2] ** 2 + a[:, :, 3] ** 2)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.array_equal(tool.perturb_rel_poses_np(poses, translation_m=0.0, rotation_deg=0.0, ages=ages, drift=True, seed=7), poses)
    with pytest.raises(ValueError):
        tool.perturb_rel_poses_np(poses, translation_m=0.2, rotation_deg=0.0, ages=None, drift=True, seed=7)
    assert tool.state_noise_seed(42, "k1") == tool.state_noise_seed(42, "k1")
    assert tool.state_noise_seed(42, "k1") != tool.state_noise_seed(43, "k1")


def test_prefix_scoring_grades_only_valid_slots() -> None:
    truth = {"slots": [("前", "近"), "空", ("后", "远")], "progress": "三"}
    predicted = {"slots": [("前", "中"), "空", ("后", "远")], "progress": "三"}
    score = tool.score_prefix(predicted, truth)
    assert score["prefix_wellformed"] and score["progress_correct"]
    assert score["slot_view_hits"] == [True, True] and score["slot_dist_hits"] == [False, True]
    assert score["slot_view_acc"] == 1.0
    assert tool.score_prefix(None, truth)["prefix_wellformed"] is False


def test_summarise_reports_passes_drops_and_associations_without_crashing() -> None:
    def state(key, kind, source, natural_ok, placeholder_ok, progress_ok):
        pred = "stop" if kind == "correct_stop" else "lookdown"
        wrong = "lookdown" if kind == "correct_stop" else "stop"
        return {
            "sample_key": key,
            "scene_id": "s",
            "episode_key": key.split(":")[0],
            "source_type": source,
            "relabel_kind": kind,
            "failure_tags": [],
            "target_texts": ["STOP"] if kind == "correct_stop" else ["↓", "1 2"],
            "target_decision": "stop" if kind == "correct_stop" else "lookdown",
            "prefix_truth": "x",
            "passes": {
                "natural": {
                    "decision": pred if natural_ok else wrong,
                    "decision_correct": natural_ok,
                    "predicted_is_stop": (pred if natural_ok else wrong) == "stop",
                    "predicted_nonpixel": (pred if natural_ok else wrong) == "stop",
                    "prefix_truth_present": True,
                    "prefix_wellformed": True,
                    "progress_correct": progress_ok,
                    "slot_view_hits": [True, False],
                    "slot_view_truth": ["前", "后"],
                    "slot_dist_hits": [True, True],
                    "slot_view_acc": 0.5,
                },
                "placeholder": {
                    "decision": pred if placeholder_ok else wrong,
                    "decision_correct": placeholder_ok,
                    "predicted_is_stop": (pred if placeholder_ok else wrong) == "stop",
                    "predicted_nonpixel": (pred if placeholder_ok else wrong) == "stop",
                    "prefix_truth_present": False,
                },
            },
        }

    states = [
        state("e1:c1", "correct_stop", "dagger_hard", True, False, True),
        state("e1:c2", "correct_stop", "dagger_hard", False, False, False),
        state("e2:c1", "keep_pixel", "dagger_normal", True, True, True),
        state("e3:c1", "keep_pixel", "dagger_normal", True, True, False),
        state("e4:c1", "correct_stop", "dagger_hard", True, True, True),
    ]
    report = tool.summarise(states, ["natural", "placeholder"])
    natural = report["passes"]["natural"]
    assert natural["stop_recall"] == pytest.approx(2 / 3)
    assert natural["stop_false_alarm_normal"] == 0.0
    assert natural["preservation_generated"] == 1.0
    assert natural["prefix"]["progress_acc"] == pytest.approx(3 / 5)
    assert report["placeholder_drop_pt"]["stop_recall"] == pytest.approx(round(100 * (2 / 3 - 1 / 3), 2))
    assert "stop_vs_progress (relevant)" in report["natural_association"]
