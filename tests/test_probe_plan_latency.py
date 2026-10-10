"""The latency probe has to be a trustworthy instrument, so test it like one.

scripts/tools/probe_plan_latency.py measures per-plan-call latency on the deployment host
and diffs what two runs decided.  Its ``compare`` half is what licenses a speed change:
"the server answered the same thing with the knob flipped" is the claim, and a diff that
quietly compares nothing, or that counts a timing difference as a real one, would make
every such claim worthless.  An earlier ad hoc version of this diff did exactly the first
thing -- it printed "identical" having compared zero plan calls, because it read the wrong
attribute off the step record.  These tests exist for that failure.

So: the frames must be reproducible from the seed alone (both arms must see the same
pixels), timing must never count as a difference, a decision difference must be caught in
every compared field including the whole response payload, and an empty comparison must be
refused rather than reported as agreement.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "probe_plan_latency.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_plan_latency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO_ROOT
    )


def _call(**over):
    call = {
        "call_index": 0,
        "step": 0,
        "model_rpc_ms": 4500.0,
        "vo_query_ms": 30.0,
        "server_timing_ms": {"total": 4400.0},
        "llm_output_chars": 5,
        "kind": "two_turn",
        "llm_output": "hello",
        "actions": [1, 1, 3, 0],
        "pixel_goal": [10, 20],
        "terminal": False,
        "pose_ready": True,
        "ppa_applied": True,
        "vo_frame_id": 1,
        "vo_history_frame_ids": [0],
        "vo_provider_phase": "ready",
        "vo_trajectory_revision": 1,
        "history_capture_steps": [0],
        "response": {"kind": "two_turn", "actions": [1, 1, 3, 0]},
    }
    call.update(over)
    return call


def _run_json(label="arm", calls=None, steps=None, env=None):
    return {
        "label": label,
        "client_env": env or {"HEATMAPVLN_TIMING": "0"},
        "server_info": {"model": "ppa-refine-v2", "vo": "amb3r-vo-1"},
        "wall_s": 12.3,
        "steps": steps
        if steps is not None
        else [
            {
                "episode_id": 0,
                "step": 0,
                "action": 1,
                "action_name": "FORWARD",
                "source": "plan",
                "wall_ms": 4600.0,
                "had_call": True,
            }
        ],
        "calls": [_call()] if calls is None else calls,
    }


def _write(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# --------------------------------------------------------------- the compared field set
def test_timing_is_never_part_of_the_comparison():
    """Compare timing and every A/B is DIFFERENT, which would make the tool useless."""
    assert "model_rpc_ms" not in probe.DECISION_FIELDS
    assert "step_wall_ms" not in probe.DECISION_FIELDS
    for key in probe.TIMING_MODE_KEYS:
        assert key not in probe.DECISION_FIELDS


def test_the_whole_response_payload_is_compared():
    """The decided fields are a convenience; the response is the server's actual answer."""
    assert "response" in probe.DECISION_FIELDS
    for name in ("llm_output", "actions", "pixel_goal", "ppa_applied"):
        assert name in probe.DECISION_FIELDS


# ------------------------------------------------------------------------------- frames
def test_frames_come_from_the_seed_alone():
    np = pytest.importorskip("numpy")
    a = probe.make_frames(7, 64, 48)
    b = probe.make_frames(7, 64, 48)
    first = a(3, "front")
    assert first.shape == (48, 64, 3) and first.dtype == np.uint8
    # Same seed, same step, same kind -> the same pixels, in any order of asking.
    assert np.array_equal(first, b(3, "front"))
    assert np.array_equal(a(3, "front"), first)
    # Different step, kind or seed -> different pixels, or the VO has nothing to track.
    assert not np.array_equal(first, a(4, "front"))
    assert not np.array_equal(first, a(3, "lookdown"))
    assert not np.array_equal(first, probe.make_frames(8, 64, 48)(3, "front"))


# ------------------------------------------------------------------------------ compare
def test_two_runs_that_decided_the_same_thing_are_identical(tmp_path):
    a = _write(tmp_path, "a.json", _run_json("queue on"))
    b = _write(tmp_path, "b.json", _run_json("queue off"))
    out = _run("compare", a, b)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "VERDICT: IDENTICAL" in out.stdout
    # It has to say what it compared, or "identical" means nothing.
    assert "compared 1 plan calls" in out.stdout
    # And which arms it compared, which only the labels record.
    assert "queue on" in out.stdout and "queue off" in out.stdout


def test_comparing_two_runs_with_the_same_label_says_so(tmp_path):
    """The arm lives in --label; identical labels mean the output cannot name it."""
    a = _write(tmp_path, "a.json", _run_json("same name"))
    b = _write(tmp_path, "b.json", _run_json("same name"))
    out = _run("compare", a, b)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "same --label" in out.stdout


def test_a_difference_in_timing_alone_is_not_a_difference(tmp_path):
    """Including what the server adds to the response only because timing is on.

    On the 910B that is timing_ms plus cuda_memory_mib (a memory reading taken beside
    the stage times).  Counting either as a decision difference would make every
    timing-on-vs-off comparison DIFFERENT, which is the comparison that H2 needs.
    """
    a = _write(tmp_path, "a.json", _run_json(calls=[_call(model_rpc_ms=4500.0)]))
    slow = _call(model_rpc_ms=9100.0, server_timing_ms={"total": 9000.0})
    slow["response"] = {
        **slow["response"],
        "timing_ms": {"total": 9000.0},
        "cuda_memory_mib": {"peak_reserved": 33652.0},
    }
    b = _write(tmp_path, "b.json", _run_json(calls=[slow]))
    out = _run("compare", a, b)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "VERDICT: IDENTICAL" in out.stdout


@pytest.mark.parametrize(
    "field, value",
    [
        ("llm_output", "goodbye"),
        ("actions", [1, 1, 3, 1]),
        ("pixel_goal", [10, 21]),
        ("ppa_applied", False),
        ("pose_ready", False),
        ("vo_trajectory_revision", 2),
        ("response", {"kind": "two_turn", "actions": [2, 2, 2, 0]}),
    ],
)
def test_any_decided_field_differing_is_caught(tmp_path, field, value):
    a = _write(tmp_path, "a.json", _run_json())
    b = _write(tmp_path, "b.json", _run_json(calls=[_call(**{field: value})]))
    out = _run("compare", a, b)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "VERDICT: DIFFERENT" in out.stdout
    assert field in out.stdout


def test_a_different_action_sequence_is_caught_even_with_no_plan_call_difference(tmp_path):
    a = _write(tmp_path, "a.json", _run_json())
    steps = [dict(_run_json()["steps"][0], action=3, action_name="TURN_RIGHT")]
    b = _write(tmp_path, "b.json", _run_json(steps=steps))
    out = _run("compare", a, b)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "action" in out.stdout


@pytest.mark.parametrize("which", ["a", "b", "both"])
def test_comparing_nothing_is_refused_not_called_identical(tmp_path, which):
    """The failure this guard exists for: zero compared calls reading as agreement."""
    empty = _run_json(calls=[], steps=[])
    a = _write(tmp_path, "a.json", empty if which in ("a", "both") else _run_json())
    b = _write(tmp_path, "b.json", empty if which in ("b", "both") else _run_json())
    out = _run("compare", a, b)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "VACUOUS" in out.stderr
    assert "IDENTICAL" not in out.stdout


def test_an_unreadable_run_is_refused(tmp_path):
    a = _write(tmp_path, "a.json", _run_json())
    (tmp_path / "bad.json").write_text("{not json")
    out = _run("compare", a, str(tmp_path / "bad.json"))
    assert out.returncode == 2
    assert "cannot read" in out.stderr
    out = _run("compare", a, str(tmp_path / "missing.json"))
    assert out.returncode == 2


def test_a_differing_call_count_is_a_difference(tmp_path):
    a = _write(tmp_path, "a.json", _run_json(calls=[_call(), _call(call_index=1, step=4)]))
    b = _write(tmp_path, "b.json", _run_json())
    out = _run("compare", a, b)
    assert out.returncode == 1
    assert "plan-call count differs" in out.stdout


# ---------------------------------------------------------------------------------- run
def test_run_needs_two_servers_or_the_fakes():
    out = _run("run", "--out", "/dev/null")
    assert out.returncode == 2
    assert "--fake" in out.stderr


def test_driving_the_fake_servers_twice_decides_the_same_thing(tmp_path):
    """End to end over the real NavAgent, no accelerator: the driver itself is reproducible."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    paths = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.json"
        out = _run("run", "--fake", "--plan-calls", "3", "--out", str(path), "--label", name)
        assert out.returncode == 0, out.stdout + out.stderr
        recorded = json.loads(path.read_text())
        assert len(recorded["calls"]) == 3
        assert all(c["model_rpc_ms"] > 0 for c in recorded["calls"])
        paths.append(str(path))
    out = _run("compare", *paths)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "VERDICT: IDENTICAL" in out.stdout
