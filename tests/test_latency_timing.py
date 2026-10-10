"""Opt-in latency timing: src/utils/latency.py and every place that uses it.

Off (the default) must be a no-op everywhere: no clock read, no CUDA sync, no
new response field, no file.  On, the stages are timed and the responses carry
``timing_ms``.

Run anywhere: StageTimer, the VO server dispatcher (with the real online AMB3R
session when its imports resolve), the VO bridge, the client's
PlanCallTimingLog (its source is executed out of the client file, which imports
habitat at module level) and the client wiring (AST), and the summariser.
Need ``vla_rpc`` (and habitat for the client import), skipped otherwise: the
model server end to end on CPU fakes, and _rpc_plan_panoramic's payload.  Run
those in the RTX 4090 container with ``CUDA_VISIBLE_DEVICES=``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from scripts.amb3r_vo.rpc_amb3r_vo_server import AMB3RVORPCApplication
from scripts.tools import summarize_latency as summary_tool

from src.utils import latency

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "scripts" / "evaluation" / "r2r_val_unseen.py"


def _boom(*_args, **_kwargs):
    raise AssertionError("touched a clock or CUDA while timing is off")


@pytest.fixture()
def no_clock(monkeypatch):
    """Any clock read or CUDA sync through src/utils/latency.py fails the test."""
    monkeypatch.setattr(latency, "time", SimpleNamespace(perf_counter=_boom))
    monkeypatch.setattr(latency, "_accel_sync_fn", _boom)


class FakeClock:
    """perf_counter stand-in, in seconds, advanced by the test."""

    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000.0


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------


def test_timing_enabled_reads_the_env_var(monkeypatch):
    monkeypatch.delenv(latency.TIMING_ENV, raising=False)
    assert not latency.timing_enabled()
    for value in ("", "0", "false", "No", " off "):
        monkeypatch.setenv(latency.TIMING_ENV, value)
        assert not latency.timing_enabled(), value
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(latency.TIMING_ENV, value)
        assert latency.timing_enabled(), value


def test_stage_timer_off_is_a_no_op(monkeypatch, no_clock):
    monkeypatch.delenv(latency.TIMING_ENV, raising=False)
    for timer in (latency.StageTimer(), latency.StageTimer(enabled=False)):
        assert not timer.enabled
        with timer.stage("a"):
            pass
        with pytest.raises(ValueError, match="propagates"), timer.stage("b"):
            raise ValueError("propagates")
        timer.add("c", 5.0)
        timer.rename("a", "d")
        assert timer.as_dict() == {} and timer.counts == {}


def test_stage_timer_on_accumulates_counts_and_syncs(monkeypatch):
    clock = FakeClock()
    syncs = []
    monkeypatch.setattr(latency, "time", clock)
    monkeypatch.setattr(latency, "_accel_sync_fn", lambda _device: (lambda: syncs.append(clock.now)))
    timer = latency.StageTimer(enabled=True)
    for ms in (250.0, 500.0):
        with timer.stage("generate"):
            clock.advance(ms)
    with pytest.raises(RuntimeError), timer.stage("nextdit"):
        clock.advance(1.0)
        raise RuntimeError("a failing stage is still recorded")
    timer.add("nextdit", 2.0)
    assert timer.as_dict() == {"generate": 750.0, "nextdit": 3.0}
    assert timer.counts == {"generate": 2, "nextdit": 2}
    assert len(syncs) == 6  # before and after each of the three stages

    timer.rename("nextdit", "generate")
    assert timer.as_dict() == {"generate": 753.0} and timer.counts == {"generate": 4}
    timer.rename("missing", "x")
    assert timer.as_dict() == {"generate": 753.0}


def test_stage_timer_without_cuda_sync_never_asks_for_it(monkeypatch):
    monkeypatch.setattr(latency, "time", FakeClock())
    monkeypatch.setattr(latency, "_accel_sync_fn", _boom)
    timer = latency.StageTimer(enabled=True, cuda_sync=False)
    with timer.stage("a"):
        pass
    assert timer.counts == {"a": 1}


class _FakeCuda:
    """torch.cuda stand-in on a machine with CUDA: records what was synchronised."""

    def __init__(self) -> None:
        self.synced = []
        self.reset = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: self.synced.append(device))
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self.reset.append)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 3 * 2**20)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 5 * 2**20)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 * 2**20, 48 * 2**20))

    def module(self) -> types.ModuleType:
        """The same surface as a standalone module, for standing in as ``torch.npu``."""
        module = types.ModuleType("fake_accelerator")
        module.is_available = lambda: True
        module.synchronize = lambda device=None: self.synced.append(device)
        module.reset_peak_memory_stats = self.reset.append
        module.max_memory_allocated = lambda device: 3 * 2**20
        module.max_memory_reserved = lambda device: 5 * 2**20
        module.mem_get_info = lambda device: (40 * 2**20, 48 * 2**20)
        return module


def test_stage_timer_syncs_the_device_it_is_given(monkeypatch):
    cuda = _FakeCuda()
    cuda.install(monkeypatch)
    for device, expected in (("cuda:1", ["cuda:1"] * 2), (None, [None] * 2), (torch.device("cpu"), [])):
        cuda.synced.clear()
        with latency.StageTimer(enabled=True, device=device).stage("a"):
            pass
        assert cuda.synced == expected, device


def test_accel_memory_helpers(monkeypatch):
    assert latency.accel_memory_mib(torch.device("cpu")) is None
    latency.reset_accel_peak(torch.device("cpu"))  # no-op on CPU
    cuda = _FakeCuda()
    cuda.install(monkeypatch)
    latency.reset_accel_peak("cuda:1")
    assert cuda.reset == ["cuda:1"]
    assert latency.accel_memory_mib("cuda:1") == {"peak_allocated": 3.0, "peak_reserved": 5.0, "device_used": 8.0}
    assert latency.accel_memory_mib(torch.device("cpu")) is None


def test_naming_an_unusable_accelerator_raises_instead_of_timing_nothing(monkeypatch):
    """A skipped synchronisation would report launch time as compute time."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for call in (
        lambda: latency.StageTimer(enabled=True, device="cuda:0"),
        lambda: latency.reset_accel_peak("cuda:0"),
        lambda: latency.accel_memory_mib("cuda:0"),
    ):
        with pytest.raises(RuntimeError, match="not usable here"):
            call()
    # An unnamed device still means "whichever accelerator is available", or none.
    assert latency.StageTimer(enabled=True, device=None) is not None


def test_npu_is_an_accelerator_like_cuda(monkeypatch):
    """The same helpers drive torch.npu, so Ascend runs are timed, not silently skipped."""
    npu = _FakeCuda()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "npu", npu.module(), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    # "npu" is a device type only once the real torch_npu has registered it, so on
    # the 4090 and the C500 torch.device("npu:1") itself raises before latency ever
    # reaches the fake backend.  latency reads nothing but .type from the result.
    real = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda spec: types.SimpleNamespace(type="npu") if str(spec).startswith("npu") else real(spec),
    )
    latency.reset_accel_peak("npu:1")
    assert npu.reset == ["npu:1"]
    assert latency.accel_memory_mib("npu:1") == {"peak_allocated": 3.0, "peak_reserved": 5.0, "device_used": 8.0}
    with latency.StageTimer(enabled=True, device="npu:1").stage("a"):
        pass
    assert npu.synced == ["npu:1"] * 2


# ---------------------------------------------------------------------------
# VO server dispatcher
# ---------------------------------------------------------------------------


class _RevisionSession:
    """Online-session fake whose map revision moves as scripted per call."""

    def __init__(self, bumps: dict[str, list[int]]) -> None:
        self.trajectory_revision = 0
        self.bumps = {method: list(values) for method, values in bumps.items()}

    def reset(self, session_id, *, max_frames):
        self.trajectory_revision = 0
        return {"schema": "fake-reset", "session_id": session_id, "max_frames": max_frames}

    def ingest(self, session_id, *, frame_id, frame_rgb, capture_step):
        self.trajectory_revision += self.bumps["ingest"].pop(0)
        return {
            "schema": "fake-ingest",
            "session_id": session_id,
            "frame_id": frame_id,
            "capture_step": capture_step,
            "trajectory_revision": self.trajectory_revision,
        }

    def query(self, session_id, *, current_frame_id, history_frame_ids, translation_scale):
        self.trajectory_revision += self.bumps["query"].pop(0)
        payload = {
            "session_id": session_id,
            "current_frame_id": current_frame_id,
            "history_frame_ids": list(history_frame_ids),
            "history_rel_poses": [[0.0, 0.0, 1.0, 0.0] for _ in history_frame_ids],
            "ready": True,
            "provider_phase": "fake",
            "trajectory_revision": self.trajectory_revision,
        }
        return SimpleNamespace(to_payload=lambda: dict(payload))


def _blob():
    return {"name": "rgb_front", "data": b"jpeg", "mime_type": "image/jpeg", "height": 2, "width": 3}


def _vo_sequence(application) -> list[dict]:
    session = "scene/0001"
    calls = [("reset_episode", {"session_id": session, "max_frames": 10}, [])]
    for frame in range(3):
        calls.append(("ingest_frame", {"session_id": session, "frame_id": frame, "capture_step": frame}, [_blob()]))
    calls.append(("query_relative_poses", {"session_id": session, "current_frame_id": 2, "history_frame_ids": [0]}, []))
    calls.append(("query_relative_poses", {"session_id": session, "current_frame_id": 2, "history_frame_ids": [1]}, []))
    return [application.dispatch(method, payload, blobs) for method, payload, blobs in calls]


def _vo_application(timing: bool, timing_device=None) -> AMB3RVORPCApplication:
    session = _RevisionSession({"ingest": [0, 1, 1], "query": [0, 1]})
    return AMB3RVORPCApplication(
        session,
        jpeg_decoder=lambda _data: np.zeros((2, 3, 3), dtype=np.uint8),
        timing=timing,
        timing_device=timing_device,
    )


def test_vo_server_off_adds_nothing(monkeypatch, no_clock):
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, name, _boom)
    application = _vo_application(timing=False, timing_device="cuda:0")
    assert application._latency is None
    responses = _vo_sequence(application)
    assert all("timing_ms" not in response and "cuda_memory_mib" not in response for response in responses)


def test_vo_server_on_reports_the_cuda_memory_of_its_device(monkeypatch):
    cuda = _FakeCuda()
    cuda.install(monkeypatch)
    responses = _vo_sequence(_vo_application(timing=True, timing_device="cuda:1"))
    assert all(
        r["cuda_memory_mib"] == {"peak_allocated": 3.0, "peak_reserved": 5.0, "device_used": 8.0} for r in responses
    )
    assert cuda.reset == ["cuda:1"] * len(responses)  # a peak window per request
    assert set(cuda.synced) == {"cuda:1"}
    # No device given (the dependency-free tests): timing only, no memory field.
    assert all("cuda_memory_mib" not in r for r in _vo_sequence(_vo_application(timing=True)))


def test_vo_server_on_names_stages_by_map_event(monkeypatch):
    monkeypatch.setattr(latency, "_accel_sync_fn", lambda _device: (lambda: None))
    off = _vo_sequence(_vo_application(timing=False))
    on = _vo_sequence(_vo_application(timing=True))
    assert [{k: v for k, v in r.items() if k != "timing_ms"} for r in on] == off
    assert [list(r["timing_ms"]) for r in on] == [
        ["total"],
        ["jpeg_decode", "ingest", "total"],
        ["jpeg_decode", "ingest_map_init", "total"],
        ["jpeg_decode", "ingest_map_update", "total"],
        ["query", "total"],
        ["query_map_update", "total"],
    ]
    assert all(value >= 0.0 for r in on for value in r["timing_ms"].values())


def test_vo_server_on_with_the_real_online_session(monkeypatch):
    online = pytest.importorskip("src.vo.online_amb3r")
    monkeypatch.setattr(latency, "_accel_sync_fn", lambda _device: (lambda: None))

    class Backend:
        def reset(self, *, max_frames):
            self.count = 0

        def initialize(self, frames_rgb):
            return np.tile(np.eye(4, dtype=np.float32), (len(frames_rgb), 1, 1))

        def map_increment(self, frames_rgb, *, start_index, end_index):
            return np.tile(np.eye(4, dtype=np.float32), (end_index + 1, 1, 1))

        def poses(self, *, frame_count):
            return np.tile(np.eye(4, dtype=np.float32), (frame_count, 1, 1))

    session = online.OnlineAMB3RSession(
        Backend(), map_init_window=2, map_every=2, max_history=8, frame_processor=lambda frame: frame
    )
    application = AMB3RVORPCApplication(
        session, jpeg_decoder=lambda _data: np.zeros((2, 3, 3), dtype=np.uint8), timing=True
    )
    sid = "scene/0002"

    def ingest(frame):
        return application.dispatch(
            "ingest_frame", {"session_id": sid, "frame_id": frame, "capture_step": frame}, [_blob()]
        )

    def query(current, history):
        return application.dispatch(
            "query_relative_poses",
            {"session_id": sid, "current_frame_id": current, "history_frame_ids": history},
            [],
        )

    application.dispatch("reset_episode", {"session_id": sid, "max_frames": 10}, [])
    events = [
        ingest(0),  # stored
        ingest(1),  # map_init_window reached
        ingest(2),  # stored, one unmapped frame
        query(2, [0, 1]),  # maps the pending tail first
        ingest(3),
        ingest(4),  # map_every reached
        query(4, [2, 3]),  # already mapped
    ]
    names = [sorted(set(r["timing_ms"]) - {"jpeg_decode", "total"}) for r in events]
    assert names == [
        ["ingest"],
        ["ingest_map_init"],
        ["ingest"],
        ["query_map_update"],
        ["ingest"],
        ["ingest_map_update"],
        ["query"],
    ]
    assert session.trajectory_revision == 3


# ---------------------------------------------------------------------------
# VO bridge (client side)
# ---------------------------------------------------------------------------


def _load_vo_client():
    """Load src/vo/{rpc_protocol,rpc_client}.py without src/vo/__init__.py, like the client."""
    package = "_latency_test_vo"
    if f"{package}.rpc_client" not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / "src" / "vo")]
        sys.modules[package] = module
        for name in ("rpc_protocol", "rpc_client"):
            spec = importlib.util.spec_from_file_location(f"{package}.{name}", ROOT / "src" / "vo" / f"{name}.py")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
    return sys.modules[f"{package}.rpc_client"], sys.modules[f"{package}.rpc_protocol"]


class _FakeVOClient:
    def __init__(self, protocol, server_ms) -> None:
        self.protocol = protocol
        self.server_ms = server_ms
        self.calls = []

    def infer_json(self, method, payload, blobs):
        self.calls.append((method, json.dumps(payload, sort_keys=True), [dict(b) for b in blobs]))
        response = {"ok": True, "proto_v": self.protocol.AMB3R_VO_RPC_PROTOCOL_VERSION, **payload}
        if method == self.protocol.AMB3R_VO_QUERY_METHOD:
            response.update(
                {
                    "ready": False,
                    "provider_phase": "map_warmup",
                    "trajectory_revision": 0,
                    "pose_provider": self.protocol.AMB3R_VO_POSE_PROVIDER,
                    "history_rel_poses": [],
                }
            )
        if self.server_ms is not None:
            response["timing_ms"] = dict(self.server_ms)
            response["cuda_memory_mib"] = {"device_used": 8.0}
        return response, []


def _bridge_run(bridge):
    bridge.reset_episode("scene/1", max_frames=8)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    bridge.ingest_rgb(rgb, capture_step=0)
    bridge.ingest_rgb(rgb, capture_step=0)  # repeated capture step: no RPC
    bridge.ingest_rgb(rgb, capture_step=1)
    return bridge.query_model_pose_fields(current_frame_id=1, history_frame_ids=[0])


def test_vo_bridge_timing_off_and_on(monkeypatch):
    rpc_client, protocol = _load_vo_client()
    encoder = lambda _rgb, quality: f"jpeg-{quality}".encode()  # noqa: E731

    monkeypatch.setattr(rpc_client, "time", SimpleNamespace(perf_counter=_boom))
    off_client = _FakeVOClient(protocol, server_ms=None)
    off = rpc_client.OnlineVORPCBridge(off_client, jpeg_encoder=encoder)
    off_fields = _bridge_run(off)
    assert off.timing_entries == [] and off.drain_timing() == []

    monkeypatch.setattr(rpc_client, "time", FakeClock())
    on_client = _FakeVOClient(protocol, server_ms={"total": 1.5})
    on = rpc_client.OnlineVORPCBridge(on_client, jpeg_encoder=encoder, timing=True)
    assert _bridge_run(on) == off_fields
    assert on_client.calls == off_client.calls  # identical requests
    entries = on.drain_timing()
    assert [entry["method"] for entry in entries] == [
        "reset_episode",
        "ingest_frame",
        "ingest_frame",
        "query_relative_poses",
    ]
    assert all(
        entry["server_ms"] == {"total": 1.5}
        and entry["server_cuda_mib"] == {"device_used": 8.0}
        and entry["rpc_ms"] == 0.0
        for entry in entries
    )
    assert on.drain_timing() == []


# ---------------------------------------------------------------------------
# Client: PlanCallTimingLog and its wiring
# ---------------------------------------------------------------------------


def _client_timing(clock) -> dict[str, Any]:
    """PlanCallTimingLog and _timed, executed from the client source with a fake clock."""
    source = CLIENT.read_text(encoding="utf-8")
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("PlanCallTimingLog", "_timed")
    ]
    assert [node.name for node in nodes] == ["PlanCallTimingLog", "_timed"]
    code = "from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(source, n) for n in nodes)
    namespace = {"contextlib": contextlib, "json": json, "time": clock, "Path": Path, "Any": Any}
    exec(compile(code, str(CLIENT), "exec"), namespace)
    return namespace


class _EntryBridge:
    def __init__(self) -> None:
        self.entries = []

    def drain_timing(self):
        entries, self.entries = self.entries, []
        return entries


def test_client_timed_is_inert_without_a_log():
    namespace = _client_timing(SimpleNamespace(perf_counter=_boom))
    with namespace["_timed"](None, "env_step"):
        pass


def test_plan_call_log_follows_the_client_loop(tmp_path):
    clock = FakeClock()
    namespace = _client_timing(clock)
    timed = namespace["_timed"]
    bridge = _EntryBridge()
    bridge.entries.append({"method": "reset_episode", "rpc_ms": 1.0, "server_ms": None})
    log = namespace["PlanCallTimingLog"](tmp_path / "timing" / "client.jsonl", vo_bridge=bridge)

    def run(name, ms, vo=None):
        with timed(log, name):
            clock.advance(ms)
            if vo is not None:
                bridge.entries.append(vo)

    ingest = {"method": "ingest_frame", "rpc_ms": 4.0, "server_ms": {"total": 3.0}}
    query = {"method": "query_relative_poses", "rpc_ms": 2.0, "server_ms": {"total": 1.0}}
    log.begin_episode("zsNo4HB9uLZ", 1)  # drops the reset entry
    run("vo_ingest", 5.0, ingest)  # frame 0
    log.begin_call(0, 0)
    run("pano_capture", 40.0)
    run("vo_query", 2.5, query)
    run("lookdown_capture", 30.0)
    run("model_encode", 10.0)
    run("model_rpc", 300.0)
    log.end_plan({"kind": "trajectory", "actions": [1, 1, 5], "ppa_applied": False, "timing_ms": {"handler_total": 280.0}}, False)
    run("env_step", 20.0)  # first action: step 0 -> 1
    run("vo_ingest", 6.0, ingest)  # frame 1
    run("pano_capture", 41.0)
    run("env_step", 21.0)  # queued: step 1 -> 2
    run("vo_ingest", 7.0, ingest)  # frame 2
    run("pano_capture", 42.0)  # the popped action was STOP: replan, no step
    run("vo_ingest", 0.5)  # same capture step again: no VO call, not recorded
    log.begin_call(1, 2)
    run("model_rpc", 100.0)
    log.end_plan({"kind": "stop", "actions": [0], "terminal": True}, True)
    run("env_step", 3.0)
    log.end_episode(3)

    lines = [json.loads(line) for line in (tmp_path / "timing" / "client.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    first, second = lines
    assert first == {
        "schema": "heatmapvln-latency-v1",
        "episode": "zsNo4HB9uLZ/0001",
        "scene_id": "zsNo4HB9uLZ",
        "episode_id": 1,
        "call_index": 0,
        "step": 0,
        "plan_ms": {
            "pano_capture": 40.0,
            "vo_query": 2.5,
            "lookdown_capture": 30.0,
            "model_encode": 10.0,
            "model_rpc": 300.0,
        },
        "kind": "trajectory",
        "pose_ready": False,
        "ppa_applied": False,
        "actions_returned": 3,
        "model_server_ms": {"handler_total": 280.0},
        "model_cuda_mib": None,
        "actions_executed": 2,
        "cycle_wall_ms": 525.0,
        "step_ms": {
            "vo_ingest": [5.0, 6.0, 7.0],
            "env_step": [20.0, 21.0],
            "pano_capture": [41.0, 42.0],
        },
        "vo_rpc": [ingest, query, ingest, ingest],
    }
    assert second["call_index"] == 1 and second["pose_ready"] is True
    assert second["plan_ms"] == {"model_rpc": 100.0}
    assert second["actions_executed"] == 1 and second["cycle_wall_ms"] == 103.0
    assert second["step_ms"] == {"env_step": [3.0]} and second["vo_rpc"] == []


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _parents(root: ast.AST) -> dict:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def _is_log_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "timing_log"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def test_client_touches_the_log_only_behind_its_guard():
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
    for fn_name in ("run_eval_rpc_panoramic", "_rpc_plan_panoramic"):
        fn = _function(tree, fn_name)
        parents = _parents(fn)

        def guarded(node, parents=parents) -> bool:
            child, parent = node, parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.If) and child in parent.body and (
                    _is_log_guard(parent.test) or ast.unparse(parent.test) == "timing_enabled()"
                ):
                    return True
                child, parent = parent, parents.get(parent)
            return False

        uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "timing_log"]
        for node in uses:
            parent = parents[node]
            if isinstance(node.ctx, ast.Store) or (isinstance(parent, ast.Compare) and _is_log_guard(parent)):
                continue
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "_timed":
                assert parent.args[0] is node
                continue
            if isinstance(parent, ast.keyword) and parent.arg == "timing_log":
                continue
            assert guarded(node), f"{fn_name}:{node.lineno} uses timing_log outside `if timing_log is not None`"

    fn = _function(tree, "run_eval_rpc_panoramic")
    parents = _parents(fn)
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "timing_log" and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 2  # "= None", then the log under `if timing_enabled():`
    created = parents[parents[stores[1]]]
    assert isinstance(created, ast.If) and ast.unparse(created.test) == "timing_enabled()"
    stages = {
        n.args[1].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_timed"
    }
    assert stages == {
        "vo_ingest", "pano_capture", "env_step", "vo_query", "lookdown_capture", "model_encode", "model_rpc",
    }
    bridge = next(
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "OnlineVORPCBridge"
    )
    assert {k.arg: ast.unparse(k.value) for k in bridge.keywords}["timing"] == "timing_enabled()"


# ---------------------------------------------------------------------------
# Summariser
# ---------------------------------------------------------------------------


def _record(call_index, *, pose_ready, actions, wall, plan, server, steps, vo_rpc, kind="trajectory", cuda=None):
    return {
        "schema": "heatmapvln-latency-v1",
        "episode": "zsNo4HB9uLZ/0001",
        "scene_id": "zsNo4HB9uLZ",
        "episode_id": 1,
        "call_index": call_index,
        "step": 0,
        "plan_ms": plan,
        "kind": kind,
        "pose_ready": pose_ready,
        # The server applies PPA only on a ready trajectory call (S: _plan_panoramic_native).
        "ppa_applied": bool(pose_ready) and kind == "trajectory",
        "actions_returned": 4,
        "model_server_ms": server,
        "model_cuda_mib": cuda,
        "actions_executed": actions,
        "cycle_wall_ms": wall,
        "step_ms": steps,
        "vo_rpc": vo_rpc,
    }


PLAN = {"pano_capture": 40.0, "vo_query": 5.0, "lookdown_capture": 30.0, "model_encode": 10.0, "model_rpc": 200.0}
WARMUP = _record(
    0,
    pose_ready=False,
    actions=2,
    wall=500.0,
    plan=PLAN,
    server={"request_decode": 5.0, "system2_turn1_generate": 100.0, "system1_nextdit_sampling": 40.0, "handler_total": 190.0},
    steps={"vo_ingest": [6.0, 8.0], "env_step": [20.0, 30.0], "pano_capture": [40.0]},
    vo_rpc=[
        {
            "method": "ingest_frame",
            "rpc_ms": 5.0,
            "server_ms": {"jpeg_decode": 1.0, "ingest": 2.0, "total": 4.0},
            "server_cuda_mib": {"peak_allocated": 300.0, "peak_reserved": 400.0, "device_used": 30000.0},
        },
        {"method": "query_relative_poses", "rpc_ms": 4.0, "server_ms": {"query": 1.0, "total": 2.0}},
    ],
    cuda={"peak_allocated": 1000.0, "peak_reserved": 1500.0, "device_used": 30000.0},
)
READY = _record(
    1,
    pose_ready=True,
    actions=5,
    wall=900.0,
    plan={**PLAN, "vo_query": 15.0, "model_rpc": 300.0},
    server={"ppa_history_memory": 30.0, "ppa_bridge": 2.0, "handler_total": 280.0},
    steps={"vo_ingest": [6.0, 6.0, 6.0, 6.0, 51.0], "env_step": [20.0] * 5, "pano_capture": [40.0] * 4},
    vo_rpc=[{"method": "ingest_frame", "rpc_ms": 9.0, "server_ms": {"jpeg_decode": 1.0, "ingest_map_update": 45.0, "total": 47.0}}],
    cuda={"peak_allocated": 1200.0, "peak_reserved": 1600.0, "device_used": 30500.0},
)


def test_summary_of_known_records(tmp_path):
    timing = tmp_path / "workers" / "shard_00" / "timing"
    timing.mkdir(parents=True)
    stale = dict(READY, cycle_wall_ms=99999.0)  # the same call from an earlier file: replaced
    (timing / "a.jsonl").write_text(json.dumps(stale) + "\n" + json.dumps({"schema": "other"}) + "\n")
    (timing / "b.jsonl").write_text(json.dumps(WARMUP) + "\n\n" + json.dumps(READY) + "\n")

    assert summary_tool.main([str(tmp_path / "workers"), "--output-dir", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "latency_summary.json").read_text())
    markdown = (tmp_path / "out" / "latency_summary.md").read_text()

    assert summary["calls"] == 2 and summary["actions_executed"] == 7
    assert {key: group["calls"] for key, group in summary["groups"].items()} == {"ppa": 1, "native_system1": 1, "all": 2}
    # Per call:  warm-up / ready
    #   model      10+200=210 / 10+300=310
    #   vo         5+6+8=19 / 15+6*4+51=90
    #   simulator  40+30+50+40=160 / 40+30+100+160=330
    #   timed      389 / 730;  wall 500 / 900;  untimed 111 / 170
    #   plan       285 / 395;  server 190 / 280;  overhead 10 / 20
    # p90 of two values a<b is a + 0.9 (b - a).
    all_calls = markdown.split("## All calls: 2 calls, 7 actions")[1]
    assert (
        "### Per plan call\n\n"
        "| stage | n | mean | median | p90 |\n"
        "|---|---:|---:|---:|---:|\n"
        "| model | 2 | 260.0 | 260.0 | 300.0 |\n"
        "| vo | 2 | 54.5 | 54.5 | 82.9 |\n"
        "| simulator | 2 | 245.0 | 245.0 | 313.0 |\n"
        "| timed_total | 2 | 559.5 | 559.5 | 695.9 |\n"
        "| cycle_wall | 2 | 700.0 | 700.0 | 860.0 |\n"
        "| untimed | 2 | 140.5 | 140.5 | 164.1 |\n"
        "| plan_latency | 2 | 340.0 | 340.0 | 384.0 |\n"
        "| model_server | 2 | 235.0 | 235.0 | 271.0 |\n"
        "| model_rpc_overhead | 2 | 15.0 | 15.0 | 19.0 |\n"
    ) in all_calls
    # Per action: model 210/2=105 and 310/5=62; pooled 520/7.
    assert "| model | 2 | 83.5 | 83.5 | 100.7 | 74.3 |" in all_calls
    per_action = summary["groups"]["all"]["per_action"]
    # cycle wall per action: 500/2=250 and 900/5=180; pooled 1400/7.
    assert per_action["cycle_wall"] == {"n": 2, "mean": 215.0, "median": 215.0, "p90": 243.0, "pooled": 200.0}

    ready = summary["groups"]["ppa"]
    assert list(ready["model_server"]) == ["ppa_history_memory", "ppa_bridge", "handler_total"]
    assert ready["per_call"]["simulator"] == {"n": 1, "mean": 330.0, "median": 330.0, "p90": 330.0}
    warmup = summary["groups"]["native_system1"]
    assert list(warmup["model_server"]) == [
        "request_decode", "system2_turn1_generate", "system1_nextdit_sampling", "handler_total",
    ]
    assert summary["groups"]["all"]["cuda_memory_mib"] == {
        "model.peak_allocated": {"n": 2, "median": 1100.0, "max": 1200.0},
        "model.peak_reserved": {"n": 2, "median": 1550.0, "max": 1600.0},
        "model.device_used": {"n": 2, "median": 30250.0, "max": 30500.0},
        "vo.peak_allocated": {"n": 1, "median": 300.0, "max": 300.0},
        "vo.peak_reserved": {"n": 1, "median": 400.0, "max": 400.0},
        "vo.device_used": {"n": 1, "median": 30000.0, "max": 30000.0},
    }
    assert "| model.peak_reserved | 2 | 1550.0 | 1600.0 |" in all_calls
    vo = summary["groups"]["all"]["vo_server"]
    assert list(vo) == [
        "ingest_frame.jpeg_decode",
        "ingest_frame.ingest",
        "ingest_frame.ingest_map_update",
        "ingest_frame.total",
        "ingest_frame.round_trip",
        "query_relative_poses.query",
        "query_relative_poses.total",
        "query_relative_poses.round_trip",
    ]
    assert vo["ingest_frame.round_trip"] == {"n": 2, "mean": 7.0, "median": 7.0, "p90": 8.6}
    client = summary["groups"]["all"]["client"]
    assert client["step.env_step"]["n"] == 7 and client["step.pano_capture"]["mean"] == 40.0
    assert client["plan.vo_query"] == {"n": 2, "mean": 10.0, "median": 10.0, "p90": 14.0}


def test_summary_groups_calls_by_the_path_they_took(tmp_path):
    """pose_ready alone does not tell the path: arrows and stops never reach System 1."""
    arrows = _record(
        2,
        pose_ready=True,
        kind="native_actions",
        actions=3,
        wall=400.0,
        plan=PLAN,
        server={"system2_turn1_generate": 90.0, "handler_total": 120.0},
        steps={"env_step": [20.0] * 3},
        vo_rpc=[],
    )
    stop = _record(
        3,
        pose_ready=False,
        kind="stop",
        actions=1,
        wall=300.0,
        plan=PLAN,
        server={"system2_turn1_generate": 80.0, "handler_total": 110.0},
        steps={"env_step": [20.0]},
        vo_rpc=[],
    )
    no_vo = dict(WARMUP, call_index=4, pose_ready=None, ppa_applied=None)  # a run without the VO server
    records = [WARMUP, READY, arrows, stop, no_vo]
    assert [summary_tool._group(r) for r in records] == [
        "native_system1", "ppa", "system2_only", "system2_only", "native_system1",
    ]
    summary = summary_tool.summarise(records, ["x.jsonl"])
    groups = summary["groups"]
    assert {key: group["calls"] for key, group in groups.items()} == {
        "ppa": 1, "native_system1": 2, "system2_only": 2, "all": 5,
    }
    assert groups["ppa"]["model_server"]["handler_total"]["n"] == 1
    assert list(groups["system2_only"]["model_server"]) == ["system2_turn1_generate", "handler_total"]
    assert groups["system2_only"]["actions_executed"] == 4
    assert groups["all"]["actions_executed"] == 2 + 5 + 3 + 1 + 2
    markdown = summary_tool.to_markdown(summary)
    assert "## System 2 only (native_actions / stop / fallback_stop: no System 1): 2 calls, 4 actions" in markdown


def test_summary_refuses_empty_input(tmp_path):
    (tmp_path / "x.jsonl").write_text(json.dumps({"schema": "other"}) + "\n")
    assert summary_tool.main([str(tmp_path)]) == 1


# ---------------------------------------------------------------------------
# Model server end to end on CPU fakes (needs vla_rpc)
# ---------------------------------------------------------------------------

PLAN_DIM, HIDDEN_DIM, MEMORY_DIM = 8, 6, 4


def _jpeg(size, color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _Tokenizer:
    eos_token_id = 99

    def __init__(self, answers):
        self.answers = list(answers)

    def encode(self, _text, add_special_tokens=False):
        return [41, 42]

    def decode(self, _ids, skip_special_tokens=True):
        return self.answers.pop(0)


class _Processor:
    def __init__(self, answers):
        self.tokenizer = _Tokenizer(answers)

    def apply_chat_template(self, _messages, **_kwargs):
        ids = torch.tensor([[1, 2, 3]])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(4, 3),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }

    def image_processor(self, images, return_tensors):
        return {"pixel_values": torch.zeros(len(images), 3), "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long)}


def _trajectory(*_args, **_kwargs):
    delta = torch.zeros(32, 32, 3)
    delta[..., 0] = 0.8  # straight ahead: 0.2 m per step after action_scale 4
    return delta


def _model_runtime(server, answers):
    runtime = object.__new__(server.HeatmapVLNRuntime)
    runtime.require_deterministic_sampling = True
    runtime.device = torch.device("cpu")
    runtime.processor = _Processor(answers)
    runtime.train_cfg = {
        "data": {
            "image_size": [384, 384],
            "trajectory": {
                "system2_sft_protocol": "internnav",
                "structured_pano_output": False,
                "traj_image_size": [224, 224],
            },
        }
    }

    def history_head(*, inputs, num_histories, history_rel_poses, explicit_history_mask, return_memory_tokens):
        return {
            "history_memory": torch.ones(1, 8, MEMORY_DIM),
            "history_memory_mask": explicit_history_mask,
            "panoramic_vit_features": torch.zeros(1),
        }

    def form_plan(traj_hs, *, frozen_cond_projector, history_memory, history_memory_mask, return_diagnostics):
        plan = torch.ones(1, 4, PLAN_DIM)
        return plan, plan + 0.01, {"delta_z": torch.full((1, 4, PLAN_DIM), 0.01)}

    def decode_future(plan_z, *, past_output, past_head, time_mask=None):
        return {
            "future_visibility_probability": torch.full((1, 4, 4), 0.5),
            "future_heatmaps_gated": torch.ones(1, 4, 4, 8, 8),
        }

    qwen = SimpleNamespace(
        generate=lambda **kw: SimpleNamespace(sequences=torch.cat([kw["input_ids"], torch.tensor([[7, 8]])], dim=1)),
        generate_latents=lambda **_kw: torch.zeros(1, 4, HIDDEN_DIM),
    )
    qwen.model = qwen
    runtime.model = SimpleNamespace(
        qwen2_5_vl=qwen,
        latent_queries=torch.zeros(1, 4, HIDDEN_DIM),
        config=SimpleNamespace(dtype=torch.float32),
        nextdit_action_head=SimpleNamespace(
            config=SimpleNamespace(latent_emb_size=PLAN_DIM),
            cond_projector=None,
            get_trajectory_from_projected=_trajectory,
            get_trajectory=_trajectory,
        ),
        past_plan_action=SimpleNamespace(form_plan=form_plan, decode_future=decode_future),
        heatmap_vln=object(),
        _forward_frozen_single_view_heatmap=history_head,
    )
    runtime.pano_latent_adapter = None
    runtime.has_nextdit = True
    runtime.num_sample_trajs = 32
    runtime.action_scale = 4.0
    runtime.ppa_stage0_action_arm = "disabled"
    runtime.ppa_online_amb3r = True
    runtime.system2_cognition_arm = False
    runtime.model_version = "ppa-stage2-online-amb3r:test"
    return runtime


def _model_request(vla_pb2, call_index, *, pose_ready, num_history=3):
    from scripts.evaluation.rpc_protocol import build_rpc_sampling_metadata

    views = ("front", "right", "back", "left")
    blobs = [vla_pb2.BinaryBlob(name=f"current/{v}", data=_jpeg((384, 384), (40 * i, 9, 9))) for i, v in enumerate(views)]
    for slot in range(num_history):
        blobs += [vla_pb2.BinaryBlob(name=f"history/{slot}/{v}", data=_jpeg((384, 384), (9, 30 * slot, 9))) for v in views]
    blobs.append(vla_pb2.BinaryBlob(name="lookdown", data=_jpeg((640, 480), (90, 90, 90))))
    steps = list(range(num_history))
    current = num_history + 2
    payload = {
        "instruction": "walk past the sofa and stop at the door",
        "num_history": num_history,
        "vlm_image_size": [384, 384],
        "traj_image_size": [224, 224],
        "system1_coord_order": "generated",
        "trajectory_selection": "mean",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "require_deterministic_sampling": True,
        "phase": "joint",
        "deterministic_sampling": build_rpc_sampling_metadata(
            protocol_seed=42, scene_id="zsNo4HB9uLZ", episode_id=1, system2_call_index=call_index
        ),
        "pose_provider": "amb3r_vo_da3",
        "pose_ready": pose_ready,
        "vo_current_frame_id": current,
        "vo_history_frame_ids": steps,
        "vo_provider_phase": "stateful_backend" if pose_ready else "map_warmup",
        "vo_trajectory_revision": 2 if pose_ready else 0,
        "current_capture_step": current,
        "history_capture_steps": steps,
        "history_age_steps": [current - step for step in steps],
    }
    if pose_ready:
        payload["history_rel_poses"] = [[-0.25 * (num_history - s), 0.0, 1.0, 0.0] for s in steps]
    return vla_pb2.JSONRequest(ts=7, method="plan_panoramic", json_payload=json.dumps(payload), blobs=blobs)


# (request kwargs, System2 answers, stages the server must report)
MODEL_CASES = {
    "ready_two_turns": (
        {"pose_ready": True},
        ["↓", "180 176"],
        [
            "request_decode",
            "system2_turn1_prep",
            "system2_turn1_generate",
            "system2_turn2_prep",
            "system2_turn2_generate",
            "ppa_history_memory",
            "system1_condition_latents",
            "ppa_bridge",
            "system1_nextdit_sampling",
            "future_heatmap_diagnostics",
            "trajectory_to_actions",
            "handler_total",
        ],
    ),
    "warmup_one_turn": (
        {"pose_ready": False},
        ["180 176"],
        [
            "request_decode",
            "system2_turn1_prep",
            "system2_turn1_generate",
            "system1_condition_latents",
            "system1_nextdit_sampling",
            "trajectory_to_actions",
            "handler_total",
        ],
    ),
    "native_arrows": (
        {"pose_ready": True},
        ["←←"],
        ["request_decode", "system2_turn1_prep", "system2_turn1_generate", "handler_total"],
    ),
}

# sha256 of InferJSON's json_payload for each case on the commit before any timing
# code (63580b0), in the RTX 4090 container (torch there): its servicer and runtime
# on these fakes, HEATMAPVLN_TIMING unset.  With timing off the timed server must
# return these same bytes, so a later edit inside a timed block that changes the
# response fails here.  If the response changes on purpose, recompute them from
# that change's parent the same way.
HEAD_PAYLOAD_SHA256 = {
    "native_arrows": "ad04732bc5f087b9986a02437904592f8cb8c55b4b84ec1e5545c28f5da6fc2e",
    "ready_two_turns": "f0abdc775de84f3a4e2f9f2efd6cd007619ae5bca227df7621832af8edf40db8",
    "warmup_one_turn": "740e566fc07cc7cc3e72b4e80fd7540d818110ed670bc20cf890d909632a0363",
}


def _infer(server, vla_pb2, monkeypatch, case, timing):
    kwargs, answers, _stages = MODEL_CASES[case]
    if timing:
        monkeypatch.setenv(latency.TIMING_ENV, "1")
    else:
        monkeypatch.delenv(latency.TIMING_ENV, raising=False)
    servicer = server.HeatmapVLNRPCServicer(_model_runtime(server, answers))
    errors = []
    context = SimpleNamespace(set_details=errors.append, set_code=errors.append)
    response = servicer.InferJSON(_model_request(vla_pb2, 0, **kwargs), context)
    assert errors == []
    assert server._request_timer() is server._TIMING_OFF  # reset after the request
    return response


@pytest.fixture()
def model_server(monkeypatch):
    pytest.importorskip("vla_rpc")
    from scripts.evaluation import rpc_model_server as server
    from vla_rpc.proto import vla_pb2

    monkeypatch.setattr(latency, "_accel_sync_fn", lambda _device: (lambda: None))
    return server, vla_pb2


@pytest.mark.parametrize("case", sorted(MODEL_CASES))
def test_model_server_timing_off_changes_nothing_and_on_only_adds_timing(model_server, monkeypatch, case):
    server, vla_pb2 = model_server
    # Off: no clock, no sync, no CUDA memory call; the payload is what the runtime returned.
    with monkeypatch.context() as patch:
        patch.setattr(latency, "time", SimpleNamespace(perf_counter=_boom))
        patch.setattr(latency, "_accel_sync_fn", _boom)
        for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "mem_get_info"):
            patch.setattr(torch.cuda, name, _boom)
        off = _infer(server, vla_pb2, monkeypatch, case, timing=False)
    kwargs, answers, stages = MODEL_CASES[case]
    direct = _model_runtime(server, answers).plan_panoramic(
        json.loads(_model_request(vla_pb2, 0, **kwargs).json_payload),
        _model_request(vla_pb2, 0, **kwargs).blobs,
    )
    assert off.json_payload == json.dumps(direct, ensure_ascii=False)
    assert hashlib.sha256(off.json_payload.encode("utf-8")).hexdigest() == HEAD_PAYLOAD_SHA256[case]
    assert off.ts == 7 and off.model_v == "ppa-stage2-online-amb3r:test"

    on = _infer(server, vla_pb2, monkeypatch, case, timing=True)
    on_payload = json.loads(on.json_payload)
    timing = on_payload.pop("timing_ms")
    # Counts, not milliseconds, so they ride beside timing_ms rather than in it --
    # summarize_latency.py and the assertions below read timing_ms as stage -> ms.
    # This runtime is a stub with no vision tower, so the count is zero here; what
    # the key pins is that timing on adds exactly these two and nothing else.
    assert on_payload.pop("vision_tower") == {"vision_tower_calls": 0, "vision_tower_shapes": []}
    assert "cuda_memory_mib" not in on_payload  # CPU runtime
    assert on_payload == json.loads(off.json_payload)
    assert sorted(timing) == sorted(stages)
    assert all(value >= 0.0 for value in timing.values())
    assert timing["handler_total"] >= max(v for k, v in timing.items() if k != "handler_total")


# ---------------------------------------------------------------------------
# Client _rpc_plan_panoramic (needs habitat and vla_rpc: run in the container)
# ---------------------------------------------------------------------------


def test_rpc_plan_panoramic_request_is_identical_with_timing(tmp_path):
    pytest.importorskip("habitat")
    pytest.importorskip("vla_rpc")
    from scripts.evaluation import r2r_val_unseen as client_module

    class Client:
        def __init__(self):
            self.calls = []

        def infer_json(self, method, payload, blobs):
            self.calls.append((method, json.dumps(payload, sort_keys=True), [dict(b) for b in blobs]))
            response = {
                "ok": True,
                "proto_v": client_module.HEATMAPVLN_RPC_PROTOCOL_VERSION,
                "phase": "joint",
                "kind": "trajectory",
                "actions": [1, 1],
                "deterministic_sampling": payload["deterministic_sampling"],
                "timing_ms": {"handler_total": 1.0},
            }
            return response, []

    views = {v: Image.new("RGB", (384, 384), (i * 50, 0, 0)) for i, v in enumerate(("front", "right", "back", "left"))}
    kwargs = dict(
        instruction="go to the door",
        current_views=views,
        history_panoramas=[views],
        lookdown_img=Image.new("RGB", (640, 480), (5, 5, 5)),
        vlm_image_size=(384, 384),
        traj_image_size=(224, 224),
        system1_coord_order="generated",
        trajectory_selection="mean",
        trajectory_x_sign=1.0,
        trajectory_heading_alignment="none",
        jpeg_quality=90,
        scene_id="zsNo4HB9uLZ",
        episode_id=1,
        system2_call_index=0,
        protocol_seed=42,
        require_deterministic_sampling=True,
    )
    off_client, on_client = Client(), Client()
    off = client_module._rpc_plan_panoramic(off_client, **kwargs)
    log = client_module.PlanCallTimingLog(tmp_path / "timing" / "c.jsonl")
    log.begin_episode("zsNo4HB9uLZ", 1)
    log.begin_call(0, 0)
    on = client_module._rpc_plan_panoramic(on_client, **kwargs, timing_log=log)
    assert on == off
    assert on_client.calls == off_client.calls
    assert set(log._call["plan_ms"]) == {"model_encode", "model_rpc"}
