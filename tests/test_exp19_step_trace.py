"""EXP-19 client step-state trace: the writer, and how the client wires it in.

The writer (scripts/exp19/step_trace.py) is driven with fake agent states in
the order the RPC client calls it.  The client (scripts/evaluation/r2r_val_unseen.py)
imports habitat at module level, so its wiring is checked on the source AST:
the flag defaults to None, every tracer use sits under
``if step_tracer is not None``, and each action site passes the right phase
and System2 call index.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.exp19 import step_trace as st

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "scripts" / "evaluation" / "r2r_val_unseen.py"

RGB_SENSOR = SimpleNamespace(HEIGHT=480, WIDTH=640, HFOV=79, POSITION=[0, 1.25, 0])


# ── fakes ───────────────────────────────────────────────────────────────


class FakeQuat:
    """Only the named components, like numpy-quaternion; no coefficient order."""

    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z


def yaw_quat(yaw_deg: float) -> FakeQuat:
    half = np.deg2rad(yaw_deg) / 2.0
    return FakeQuat(np.cos(half), 0.0, np.sin(half), 0.0)


def fake_state(position, yaw_deg):
    position = np.asarray(position, dtype=np.float32)
    rotation = yaw_quat(yaw_deg)
    sensor = SimpleNamespace(position=position + np.array([0.0, 1.25, 0.0], np.float32), rotation=rotation)
    return SimpleNamespace(position=position, rotation=rotation, sensor_states={"rgb": sensor, "depth": sensor})


def fake_obs(seed: int) -> dict:
    # Smooth content so the JPEG round trip stays close.
    yy, xx = np.mgrid[0:480, 0:640]
    rgb = np.stack([(xx + 7 * seed) % 256, (yy + 3 * seed) % 256, np.full_like(xx, 40 * seed % 256)], axis=-1)
    return {"rgb": rgb.astype(np.uint8), "depth": np.zeros((480, 640, 1), np.float32)}


def fake_episode():
    return SimpleNamespace(
        instruction=SimpleNamespace(instruction_text="Walk past the sofa and stop at the door."),
        goals=[SimpleNamespace(position=[3.0, 0.1, -2.0], radius=3.0)],
        reference_path=[[0.0, 0.1, 0.0], [1.5, 0.1, -1.0], [3.0, 0.1, -2.0]],
        info={"geodesic_distance": 3.7},
        start_position=[0.0, 0.1, 0.0],
        start_rotation=[0.0, 0.3826834, 0.0, 0.9238795],  # dataset order x, y, z, w
    )


def run_fake_episode(tracer, *, scene="zsNo4HB9uLZ", episode_id=7):
    """Replays the client's hook order for one short episode.

    call 0 at step 0 -> actions [1, 2, 1, 0]: rpc_first, local x2, then the
    queued STOP flushes and replans at step 3 (no action record, no new state);
    call 1 at step 3 -> terminal STOP.
    """
    positions = {0: [0.0, 0.1, 0.0], 1: [0.25, 0.1, 0.0], 2: [0.25, 0.1, 0.0], 3: [0.5, 0.1, 0.0], 4: [0.5, 0.1, 0.0]}
    yaws = {0: 45.0, 1: 45.0, 2: 60.0, 3: 60.0, 4: 60.0}
    state = lambda s: fake_state(positions[s], yaws[s])  # noqa: E731
    tracer.begin_episode(scene, episode_id, fake_episode(), "Walk past the sofa and stop at the door", state(0))
    queue = []
    system2_calls = 0
    loop = [0, 1, 2, 3, 3]  # step ids seen at the loop top (3 twice: replan)
    for visit, step in enumerate(loop):
        tracer.record_state(step, state(step), fake_obs(step), vo_frame_id=step, queue_len=len(queue))
        if visit == 0:
            system2_calls += 1
            queue = [2, 1, 0]
            tracer.record_action(step, 1, "rpc_first", system2_calls - 1)
        elif visit in (1, 2):
            tracer.record_action(step, queue.pop(0), "local_action", system2_calls - 1)
        elif visit == 3:
            assert queue.pop(0) == 0  # STOP -> replan at the same step
        else:
            system2_calls += 1
            tracer.record_action(step, 0, "terminal", system2_calls - 1, response_kind="stop")
    metrics = {"distance_to_goal": np.float64(2.5), "success": 1.0, "spl": np.float32(0.8), "oracle_success": 1.0}
    tracer.end_episode(4, state(4), fake_obs(4), metrics, done=True, max_steps=500, system2_calls=system2_calls)
    return tracer.root / st.episode_key(scene, episode_id)


# ── writer ──────────────────────────────────────────────────────────────


def test_episode_records_follow_the_client_loop(tmp_path):
    ep_dir = run_fake_episode(st.StepStateTracer(tmp_path / "steps", RGB_SENSOR))
    assert ep_dir.name == "zsNo4HB9uLZ_0007"
    records = st.read_steps(ep_dir)
    assert [r["type"] for r in records] == [
        "episode_start",
        "state", "action",  # step 0, call 0 first action
        "state", "action",  # step 1, queued
        "state", "action",  # step 2, queued
        "state", "action",  # step 3 (seen twice: one record), call 1 terminal
        "state",  # final state after STOP
        "episode_end",
    ]

    start = records[0]
    assert start["schema"] == st.SCHEMA == "exp19-step-trace-v1"
    assert start["ep_key"] == "zsNo4HB9uLZ_0007"
    assert start["instruction"] == "Walk past the sofa and stop at the door"
    assert start["instruction_raw"] == "Walk past the sofa and stop at the door."
    assert start["sensor"] == {"rgb_hw": [480, 640], "hfov_deg": 79.0, "sensor_height_m": 1.25}
    assert start["goal_radius"] == 3.0 and start["geodesic_distance"] == 3.7
    assert start["reference_path"][-1] == pytest.approx([3.0, 0.1, -2.0])
    # State rotation is w-first; the dataset's start rotation is kept in its own x, y, z, w order.
    w, x, y, z = start["start_rotation_wxyz"]
    assert (w, y) == pytest.approx((np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5))))
    assert x == z == 0.0
    assert start["dataset_start_rotation_xyzw"] == pytest.approx([0.0, 0.3826834, 0.0, 0.9238795])

    states = [r for r in records if r["type"] == "state"]
    assert [s["step"] for s in states] == [0, 1, 2, 3, 4]
    assert [s["queue_len"] for s in states] == [0, 3, 2, 1, 0]
    assert [s["vo_frame_id"] for s in states] == [0, 1, 2, 3, None]
    assert [s["final"] for s in states] == [False, False, False, False, True]
    assert states[2]["rotation_wxyz"][2] == pytest.approx(np.sin(np.deg2rad(30.0)))
    assert list(np.subtract(states[1]["rgb_sensor_position"], states[1]["position"])) == pytest.approx([0, 1.25, 0])

    actions = [r for r in records if r["type"] == "action"]
    assert [(a["step_before"], a["action"], a["phase"], a["system2_call_index"]) for a in actions] == [
        (0, 1, "rpc_first", 0),
        (1, 2, "local_action", 0),
        (2, 1, "local_action", 0),
        (3, 0, "terminal", 1),
    ]
    # Only the terminal / empty-actions sites carry the response kind.
    assert [a.get("response_kind", "absent") for a in actions] == ["absent", "absent", "absent", "stop"]

    end = records[-1]
    assert end["ended_by"] == "stop" and end["stop_phase"] == "terminal"
    assert end["terminal_response_kind"] == "stop"
    assert end["steps"] == 4 and end["vlm_calls"] == 2
    assert end["num_states"] == 5 and end["num_actions"] == 4
    assert end["metrics"] == {"distance_to_goal": 2.5, "success": 1.0, "spl": pytest.approx(0.8), "oracle_success": 1.0}
    json.dumps(end)  # numpy scalars were converted


def test_front_frames_are_native_resolution_and_one_per_step(tmp_path):
    ep_dir = run_fake_episode(st.StepStateTracer(tmp_path, RGB_SENSOR))
    fronts = sorted(p.name for p in ep_dir.glob("front_*.jpg"))
    assert fronts == [st.front_name(s) for s in range(5)] == ["front_0000.jpg", "front_0001.jpg",
                                                                "front_0002.jpg", "front_0003.jpg", "front_0004.jpg"]
    for step in range(5):
        decoded = np.asarray(Image.open(ep_dir / st.front_name(step)).convert("RGB"), dtype=np.float32)
        assert decoded.shape == (480, 640, 3)
        assert np.abs(decoded - fake_obs(step)["rgb"]).mean() < 3.0


def test_step_cap_and_other_endings(tmp_path):
    tracer = st.StepStateTracer(tmp_path, RGB_SENSOR)
    tracer.begin_episode("s", 1, fake_episode(), "go", fake_state([0, 0, 0], 0))
    tracer.record_state(0, fake_state([0, 0, 0], 0), fake_obs(0), vo_frame_id=None, queue_len=0)
    tracer.record_action(0, 1, "rpc_first", 0)
    tracer.end_episode(1, fake_state([0.25, 0, 0], 0), fake_obs(1), {}, done=False, max_steps=1, system2_calls=1)
    end = st.read_steps(tmp_path / "s_0001")[-1]
    assert (end["ended_by"], end["stop_phase"], end["terminal_response_kind"]) == ("step_cap", None, None)

    tracer.begin_episode("s", 2, fake_episode(), "go", fake_state([0, 0, 0], 0))
    tracer.record_state(0, fake_state([0, 0, 0], 0), fake_obs(0), vo_frame_id=None, queue_len=0)
    tracer.record_action(0, 0, "auto_stop", None)
    tracer.end_episode(1, fake_state([0, 0, 0], 0), fake_obs(1), {}, done=True, max_steps=500, system2_calls=0)
    records = st.read_steps(tmp_path / "s_0002")
    assert records[2]["system2_call_index"] is None and "response_kind" not in records[2]
    assert (records[-1]["ended_by"], records[-1]["stop_phase"]) == ("stop", "auto_stop")
    assert records[-1]["terminal_response_kind"] is None

    tracer.begin_episode("s", 5, fake_episode(), "go", fake_state([0, 0, 0], 0))
    tracer.record_state(0, fake_state([0, 0, 0], 0), fake_obs(0), vo_frame_id=None, queue_len=0)
    tracer.record_action(0, 0, "rpc_empty_actions", 0, response_kind="pano_goal")
    tracer.end_episode(1, fake_state([0, 0, 0], 0), fake_obs(1), {}, done=True, max_steps=500, system2_calls=1)
    end = st.read_steps(tmp_path / "s_0005")[-1]
    assert (end["ended_by"], end["stop_phase"], end["terminal_response_kind"]) == (
        "stop", "rpc_empty_actions", "pano_goal"
    )


def test_rerun_of_an_episode_replaces_the_earlier_attempt(tmp_path):
    tracer = st.StepStateTracer(tmp_path, RGB_SENSOR)
    tracer.begin_episode("s", 3, fake_episode(), "go", fake_state([0, 0, 0], 0))
    for step in range(6):
        tracer.record_state(step, fake_state([0, 0, 0], 0), fake_obs(step), vo_frame_id=None, queue_len=0)
    # crash: no end_episode; a resumed run starts the episode again
    tracer.begin_episode("s", 3, fake_episode(), "go", fake_state([0, 0, 0], 0))
    tracer.record_state(0, fake_state([0, 0, 0], 0), fake_obs(0), vo_frame_id=None, queue_len=0)
    tracer.end_episode(1, fake_state([0, 0, 0], 0), fake_obs(1), {}, done=True, max_steps=500, system2_calls=0)
    ep_dir = tmp_path / "s_0003"
    assert sorted(p.name for p in ep_dir.glob("front_*.jpg")) == ["front_0000.jpg", "front_0001.jpg"]
    assert [r["type"] for r in st.read_steps(ep_dir)] == ["episode_start", "state", "state", "episode_end"]


def test_writer_rejects_misuse(tmp_path):
    tracer = st.StepStateTracer(tmp_path, RGB_SENSOR)
    with pytest.raises(RuntimeError, match="begin_episode"):
        tracer.record_action(0, 1, "rpc_first", 0)
    tracer.begin_episode("s", 4, fake_episode(), "go", fake_state([0, 0, 0], 0))
    with pytest.raises(ValueError, match="unknown action phase"):
        tracer.record_action(0, 1, "rpc_first_action", 0)
    with pytest.raises(ValueError, match="response_kind is recorded only"):
        tracer.record_action(0, 1, "rpc_first", 0, response_kind="trajectory")
    with pytest.raises(ValueError, match="sensor spec"):
        tracer.record_state(0, fake_state([0, 0, 0], 0), {"rgb": np.zeros((384, 384, 3), np.uint8)},
                            vo_frame_id=None, queue_len=0)
    with pytest.raises(TypeError, match="w/x/y/z"):
        st.quat_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))


def test_rotation_matches_numpy_quaternion_wxyz():
    quaternion = pytest.importorskip("quaternion")
    q = quaternion.quaternion(0.1, 0.2, 0.3, 0.4)
    assert st.quat_wxyz(q) == pytest.approx(list(quaternion.as_float_array(q)))
    assert st.quat_wxyz(q) == pytest.approx([0.1, 0.2, 0.3, 0.4])


# ── client wiring (static) ──────────────────────────────────────────────


def _client_tree() -> ast.Module:
    return ast.parse(CLIENT.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _is_tracer_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "step_tracer"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _parents(root: ast.AST) -> dict:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def test_parser_default_is_off():
    calls = [
        n for n in ast.walk(_function(_client_tree(), "main"))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--step_state_trace_dir"
    ]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(keywords["default"], ast.Constant) and keywords["default"].value is None
    assert "action" not in keywords  # a plain path option, not a store_true switch


def test_every_tracer_use_is_behind_the_none_guard():
    tree = _client_tree()
    # The tracer module is imported only inside run_eval_rpc_panoramic, only when the flag is set.
    imports = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("scripts.exp19")
    ]
    fn = _function(tree, "run_eval_rpc_panoramic")
    parents = _parents(fn)
    assert len(imports) == 1 and imports[0] in parents
    flag_check = parents[imports[0]]
    assert isinstance(flag_check, ast.If) and any(
        isinstance(n, ast.Constant) and n.value == "step_state_trace_dir" for n in ast.walk(flag_check.test)
    )

    def guarded(node) -> bool:
        child, parent = node, parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.If) and child in parent.body and _is_tracer_guard(parent.test):
                return True
            child, parent = parent, parents.get(parent)
        return False

    uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "step_tracer"]
    assigned = [n for n in uses if isinstance(n.ctx, ast.Store)]
    guards = [n for n in uses if isinstance(parents[n], ast.Compare) and _is_tracer_guard(parents[n])]
    method_calls = [n for n in uses if n not in assigned and n not in guards]
    assert len(assigned) == 2  # "= None", then the tracer under the flag check
    assert method_calls and all(guarded(n) for n in method_calls)
    # Anything outside run_eval_rpc_panoramic never touches the tracer.
    outside = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "step_tracer" and n not in parents]
    assert outside == []


def test_client_records_every_action_site_with_its_call_index():
    fn = _function(_client_tree(), "run_eval_rpc_panoramic")
    calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "record_action"
    ]
    by_phase = {}
    response_kind = ast.dump(ast.parse('response.get("kind")', mode="eval").body)
    for call in calls:
        assert len(call.args) == 4
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "step_id"  # recorded before step_id += 1
        phase = call.args[2].value
        by_phase[phase] = ast.dump(call.args[3])
        # Only the terminal / empty-actions sites pass the response kind, straight from the response.
        keywords = {k.arg: ast.dump(k.value) for k in call.keywords}
        assert keywords == ({"response_kind": response_kind} if phase in st.RESPONSE_KIND_PHASES else {}), phase
    assert sorted(by_phase) == sorted(st.ACTION_PHASES)
    call_index = ast.dump(ast.parse("system2_calls - 1", mode="eval").body)
    none = ast.dump(ast.parse("None", mode="eval").body)
    for phase, arg in by_phase.items():
        expected = none if phase in ("auto_stop", "max_system2_stop") else call_index
        assert arg == expected, phase
    # env.step executions in the RPC loop == traced action sites (lookdown capture is a helper, not here).
    applies = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_apply_habitat_action"
    ]
    auto_stop = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_maybe_stop_at_success"
    ]
    assert len(applies) + len(auto_stop) == len(calls)
