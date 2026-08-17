from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/evaluation/compare_ppa_stage0_closed_loop_ab.py"
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _row(arm: str) -> dict:
    return {
        "scene_id": "scene",
        "episode_id": 7,
        "success": 1,
        "spl": 0.8,
        "os": 1,
        "ne": 1.0,
        "steps": 4,
        "vlm_calls": 1,
        "trajectory_calls": 1,
        "recenter_calls": 0,
        "recenter_actions_executed": 0,
        "ppa_stage0_action_arm": arm,
        "ppa_stage0_action_trace": [
            {
                "schema": "heatmapvln-ppa-stage0-call-trace-v1",
                "arm": arm,
                "bridge_memory_source": (
                    "native_bypass" if arm == "baseline" else "finite_zero_probe"
                ),
                "phase": "joint",
                "system2_call_index": 0,
                "sampling": {"per_call_seed": 123},
                "kind": "trajectory",
                "llm_output": "pixel: 12 34",
                "pixel_goal": [12, 34],
                "pano_goal_view": "front",
                "actions": [1, 0, 0, 0],
                "anti_deadlock": False,
                "treatment_spec": {
                    "response_actions": [1, 0, 0, 0],
                    "habitat_actions": [1],
                    "end_reason": "local_stop_replan",
                },
            }
        ],
    }


def _run(tmp_path: Path, *, mutate_treatment: bool = False) -> subprocess.CompletedProcess[str]:
    cohort = tmp_path / "cohort.json"
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    checkpoint = tmp_path / "checkpoint.pth"
    config = tmp_path / "config.yaml"
    report = tmp_path / "report.json"
    _write(cohort, {"episodes": [{"scene_id": "scene", "episode_id": 7}]})
    _write(baseline, _row("baseline"))
    treatment_row = _row("treatment")
    if mutate_treatment:
        treatment_row["ppa_stage0_action_trace"][0]["actions"] = [2]
    _write(treatment, treatment_row)
    checkpoint.write_bytes(b"same checkpoint")
    config.write_text("model: test\n", encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cohort",
            str(cohort),
            "--baseline-progress",
            str(baseline),
            "--treatment-progress",
            str(treatment),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--report",
            str(report),
        ],
        text=True,
        capture_output=True,
    )


def test_comparator_accepts_exact_cross_arm_trace(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["exact_treatment_spec_equal"] is True


def test_comparator_fails_on_one_action_difference(tmp_path: Path) -> None:
    result = _run(tmp_path, mutate_treatment=True)
    assert result.returncode != 0
    assert "closed-loop trace mismatch" in result.stderr
