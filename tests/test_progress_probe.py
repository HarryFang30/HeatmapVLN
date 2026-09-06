"""Contract tests for the EXP-16 progress probe and the scene-leakage preflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("probe_progress_readout", "scripts/tools/probe_progress_readout.py")
leak = _load("check_scene_split_leakage", "scripts/tools/check_scene_split_leakage.py")
readout = _load("fit_recovery_readout", "scripts/tools/fit_recovery_readout.py")


def test_split_matches_fit_recovery_readout_rule():
    scenes = np.asarray(["1LXtFkjw3qL", "82sE5b5pLXE", "JmbYfDe2QKZ", "zsNo4HB9uLZ", "17DRP5sb8fy"])
    expected = readout.scene_split(scenes, 15, 25)
    assert [probe.split_code(s) for s in scenes.tolist()] == expected.tolist()
    assert leak.bucket_name("82sE5b5pLXE", 25, 15) == "val"


def test_progress_bin_edges_and_invalid():
    assert probe.progress_bin(0.0, 10.0) == 0
    assert probe.progress_bin(2.4, 10.0) == 0
    assert probe.progress_bin(2.5, 10.0) == 1
    assert probe.progress_bin(9.99, 10.0) == 3
    assert probe.progress_bin(12.0, 10.0) == 3  # clipped, never out of range
    assert probe.progress_bin(1.0, 0.0) is None
    assert probe.progress_bin(float("nan"), 10.0) is None


def test_arrived_requires_terminal_and_horizon():
    assert probe.arrived_flag({"terminal": True, "travelled_m": 1.9}, 2.0)
    assert not probe.arrived_flag({"terminal": True, "travelled_m": 2.1}, 2.0)
    assert not probe.arrived_flag({"terminal": False, "travelled_m": 0.1}, 2.0)
    assert not probe.arrived_flag({"terminal": True, "travelled_m": None}, 2.0)
    assert not probe.arrived_flag(None, 2.0)


def test_episode_false_stop_rate_and_net_benefit():
    calls = np.asarray([10, 20])
    rate = probe.episode_false_stop_rate(0.005, calls)
    assert rate == pytest.approx(np.mean([1 - 0.995**10, 1 - 0.995**20]))
    assert probe.episode_false_stop_rate(0.0, calls) == 0.0
    # refined cost is the success the episode would otherwise have had
    assert probe.net_benefit(0.5, 0.08, refined=True) == pytest.approx(probe.BONUS * 0.5 - 0.08 * probe.SR_NATIVE)
    assert probe.net_benefit(0.5, 0.08, refined=False) == pytest.approx(probe.BONUS * 0.5 - 0.08)
    assert probe.net_benefit(0.5, 0.08, refined=True) > probe.net_benefit(0.5, 0.08, refined=False)


def test_polyline_length():
    assert probe.polyline_length([[0, 0, 0], [3, 4, 0]]) == pytest.approx(5.0)
    assert probe.polyline_length([[0, 0, 0]]) == 0.0


def test_leakage_tool_flags_training_source_in_val(tmp_path):
    source = tmp_path / "src"
    (source / "82sE5b5pLXE").mkdir(parents=True)  # val bucket
    (source / "1LXtFkjw3qL").mkdir()
    jsonl = tmp_path / "states.jsonl"
    jsonl.write_text(json.dumps({"scene_id": "zsNo4HB9uLZ"}) + "\n", encoding="utf-8")
    assert leak.load_scenes(source) == {"82sE5b5pLXE", "1LXtFkjw3qL"}
    assert leak.load_scenes(jsonl) == {"zsNo4HB9uLZ"}
    import subprocess, sys

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/tools/check_scene_split_leakage.py"), "--source", f"r2r=@{source}".replace("@", "")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "LEAKAGE" in proc.stderr
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/tools/check_scene_split_leakage.py"), "--source", f"r2r={source}", "--evaluation-only", "r2r"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
