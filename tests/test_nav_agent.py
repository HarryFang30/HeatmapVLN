"""NavAgent (src/deploy/nav_agent.py) against scripted stand-ins of the two servers.

No GPU, Habitat or vla_rpc: JPEGs come from PIL here.  The request-level comparison with the
Habitat client's own code is tests/test_nav_agent_habitat.py (4090 container).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from src.deploy.fake_servers import FakeModelServer, FakeVOServer, decode_jpeg, default_script, pil_jpeg_encoder
from src.deploy.nav_agent import Action, CameraSpec, NavAgent

# The deployed launcher's RTX 4090 canary (eval_runs/canary_cuda_seed42, client logs of shards
# 0 and 1, 2026-09-28): per plan call "step kind chunk history revision", kind N=native_actions,
# T=trajectory, S=stop; history = the logged VO history frame ids; plus the episode's steps.
CANARY = {
    ("zsNo4HB9uLZ", 1, 49): "0 N 2220 - 0|3 T 1113 0,1,2 0|7 T 1112 0,1,2,3,4,5,6 0|11 T 1121 0,1,2,4,5,7,8,10 0|"
    "15 N 2200 0,2,4,6,8,10,12,14 0|17 T 1211 0,2,4,6,9,11,13,16 0|21 T 2111 0,2,5,8,11,14,17,20 2|"
    "25 T 1112 0,3,6,10,13,17,20,24 3|29 T 1311 0,4,8,12,16,20,24,28 4|33 T 1121 0,4,9,13,18,22,27,32 5|"
    "37 T 1312 0,5,10,15,20,25,30,36 6|41 T 1131 0,5,11,17,22,28,34,40 7|45 T 1110 0,6,12,18,25,31,37,44 8|"
    "48 S 0 0,6,13,20,26,33,40,47 9",
    ("zsNo4HB9uLZ", 25, 105): "0 N 3333 - 0|4 N 3000 0,1,2,3 0|5 N 3000 0,1,2,3,4 0|6 N 3330 0,1,2,3,4,5 0|"
    "9 T 1312 0,1,2,3,4,5,6,8 0|13 T 1311 0,1,3,5,6,8,10,12 0|17 T 1113 0,2,4,6,9,11,13,16 0|"
    "21 N 3300 0,2,5,8,11,14,17,20 2|23 T 3131 0,3,6,9,12,15,18,22 3|27 T 1131 0,3,7,11,14,18,22,26 4|"
    "31 T 1111 0,4,8,12,17,21,25,30 5|35 T 1112 0,4,9,14,19,24,29,34 6|39 N 2222 0,5,10,16,21,27,32,38 7|"
    "43 N 2222 0,6,12,18,24,30,36,42 8|47 T 2213 0,6,13,19,26,32,39,46 9|51 N 3333 0,7,14,21,28,35,42,50 10|"
    "55 N 3333 0,7,15,23,30,38,46,54 11|59 T 1121 0,8,16,24,33,41,49,58 12|63 T 2111 0,8,17,26,35,44,53,62 13|"
    "67 T 2111 0,9,18,28,37,47,56,66 14|71 T 2111 0,10,20,30,40,50,60,70 15|75 T 1111 0,10,21,31,42,52,63,74 16|"
    "79 T 1110 0,11,22,33,44,55,66,78 17|82 T 1111 0,11,23,34,46,57,69,81 18|86 T 1111 0,12,24,36,48,60,72,85 19|"
    "90 T 1111 0,12,25,38,50,63,76,89 20|94 T 1111 0,13,26,39,53,66,79,93 21|98 T 1111 0,13,27,41,55,69,83,97 22|"
    "102 T 1100 0,14,28,43,57,72,86,101 23|104 S 0 0,14,29,44,58,73,88,103 24",
    ("zsNo4HB9uLZ", 2, 89): "0 N 2220 - 0|3 T 1111 0,1,2 0|7 T 3111 0,1,2,3,4,5,6 0|11 N 2000 0,1,2,4,5,7,8,10 0|"
    "12 T 1131 0,1,3,4,6,7,9,11 0|16 N 3000 0,2,4,6,8,10,12,15 0|17 N 3000 0,2,4,6,9,11,13,16 0|"
    "18 T 1110 0,2,4,7,9,12,14,17 0|21 N 3000 0,2,5,8,11,14,17,20 2|22 N 3333 0,3,6,9,12,15,18,21 3|"
    "26 N 3333 0,3,7,10,14,17,21,25 4|30 T 3312 0,4,8,12,16,20,24,29 5|34 N 2222 0,4,9,14,18,23,28,33 6|"
    "38 N 2222 0,5,10,15,21,26,31,37 7|42 N 2222 0,5,11,17,23,29,35,41 8|46 N 2222 0,6,12,19,25,32,38,45 9|"
    "50 T 2212 0,7,14,21,28,35,42,49 10|54 T 1113 0,7,15,22,30,37,45,53 11|58 T 1121 0,8,16,24,32,40,48,57 12|"
    "62 T 1131 0,8,17,26,34,43,52,61 13|66 T 1111 0,9,18,27,37,46,55,65 14|70 T 1121 0,9,19,29,39,49,59,69 15|"
    "74 T 1110 0,10,20,31,41,52,62,73 16|77 T 1100 0,10,21,32,43,54,65,76 17|79 T 1110 0,11,22,33,44,55,66,78 18|"
    "82 T 1210 0,11,23,34,46,57,69,81 19|85 T 1210 0,12,24,36,48,60,72,84 20|88 S 0 0,12,24,37,49,62,74,87 21",
    ("zsNo4HB9uLZ", 26, 94): "0 N 3333 - 0|4 N 3333 0,1,2,3 0|8 N 3333 0,1,2,3,4,5,6,7 0|12 T 2212 0,1,3,4,6,7,9,11 0|"
    "16 T 1113 0,2,4,6,8,10,12,15 0|20 T 1131 0,2,5,8,10,13,16,19 2|24 N 3000 0,3,6,9,13,16,19,23 3|"
    "25 N 3000 0,3,6,10,13,17,20,24 4|26 T 1313 0,3,7,10,14,17,21,25 5|30 T 1111 0,4,8,12,16,20,24,29 6|"
    "34 T 1311 0,4,9,14,18,23,28,33 7|38 N 2222 0,5,10,15,21,26,31,37 8|42 N 2222 0,5,11,17,23,29,35,41 9|"
    "46 T 2131 0,6,12,19,25,32,38,45 10|50 T 3113 0,7,14,21,28,35,42,49 11|54 T 1111 0,7,15,22,30,37,45,53 12|"
    "58 T 1111 0,8,16,24,32,40,48,57 13|62 T 1111 0,8,17,26,34,43,52,61 14|66 T 1111 0,9,18,27,37,46,55,65 15|"
    "70 T 1111 0,9,19,29,39,49,59,69 16|74 T 1111 0,10,20,31,41,52,62,73 17|78 T 1111 0,11,22,33,44,55,66,77 18|"
    "82 T 1311 0,11,23,34,46,57,69,81 19|86 T 2131 0,12,24,36,48,60,72,85 20|90 N 2000 0,12,25,38,50,63,76,89 21|"
    "91 T 1100 0,12,25,38,51,64,77,90 22|93 S 0 0,13,26,39,52,65,78,92 23",
}
KINDS = {"N": "native_actions", "T": "trajectory", "S": "stop"}
# Key order of the Habitat client's plan payload (_rpc_plan_panoramic, then the VO fields).
PAYLOAD_KEYS = [
    "instruction", "num_history", "vlm_image_size", "traj_image_size", "system1_coord_order",
    "trajectory_selection", "trajectory_x_sign", "trajectory_heading_alignment",
    "require_deterministic_sampling", "phase", "deterministic_sampling", "pose_provider", "pose_ready",
    "vo_current_frame_id", "vo_history_frame_ids", "vo_provider_phase", "vo_trajectory_revision",
]
STEP_KEYS = ["current_capture_step", "history_capture_steps", "history_age_steps"]


def _frame(step: int, salt: int = 0) -> np.ndarray:
    return np.random.default_rng(1000 * salt + step).integers(0, 256, (480, 640, 3), dtype=np.uint8)


class _Robot:
    """Hands NavAgent a distinct frame per step, and a distinct look-down and level re-capture."""

    def __init__(self, salt: int = 0) -> None:
        self.salt = salt
        self.lookdown_steps: list[int] = []
        self.level_steps: list[int] = []
        self.agent: NavAgent | None = None

    def lookdown(self) -> np.ndarray:
        self.lookdown_steps.append(self.agent.steps)
        return _frame(self.agent.steps, salt=self.salt + 500)

    def level(self) -> np.ndarray:
        self.level_steps.append(self.agent.steps)
        return _frame(self.agent.steps, salt=self.salt + 700)

    def act(self, agent: NavAgent) -> Action:
        self.agent = agent
        return agent.act(_frame(agent.steps, self.salt), self.lookdown, self.level)

    def run(self, agent: NavAgent, max_acts: int = 1000) -> list[Action]:
        actions = []
        while not agent.done and len(actions) < max_acts:
            actions.append(self.act(agent))
        return actions


def _agent(script=default_script, *, model_timing=False, vo_timing=False, **kwargs):
    records: list[dict] = []
    model = FakeModelServer(records.append, script=script, timing=model_timing)
    vo = FakeVOServer(records.append, timing=vo_timing)
    agent = NavAgent(model, vo, jpeg_encoder=pil_jpeg_encoder, **kwargs)
    return agent, records


def _plans(records):
    return [json.loads(r["payload"]) for r in records if r["target"] == "model"]


def _chunk_script(chunks):
    def script(call_index, _payload):
        kind, actions = chunks[call_index]
        entry = {"kind": kind, "llm_output": "STOP" if kind == "stop" else "x", "actions": actions}
        if kind == "trajectory":
            entry["pixel_goal"] = [100, 200]
        return entry

    return script


def _jpeg(array: np.ndarray, size, quality: int) -> bytes:
    image = Image.fromarray(array).convert("RGB").resize(size)
    return pil_jpeg_encoder(np.asarray(image), quality=quality)


@pytest.mark.parametrize("key", list(CANARY), ids=lambda k: f"{k[0]}_{k[1]}")
def test_replays_canary_bookkeeping(key) -> None:
    """Fed the canary's action chunks, NavAgent plans at the same steps with the same history."""
    scene_id, episode_id, steps = key
    calls = [row.split() for row in CANARY[key].split("|")]
    chunks = [(KINDS[kind], [int(a) for a in chunk]) for _step, kind, chunk, _history, _rev in calls]
    agent, records = _agent(_chunk_script(chunks))
    agent.reset("Exit the bedroom. ", scene_id=scene_id, episode_id=episode_id)
    actions = _Robot().run(agent)
    plans = _plans(records)
    assert len(plans) == len(calls)
    for payload, (step, _kind, _chunk, history, revision) in zip(plans, calls):
        expected_history = [] if history == "-" else [int(h) for h in history.split(",")]
        assert payload["current_capture_step"] == int(step)
        assert payload["vo_current_frame_id"] == int(step)  # one VO frame per executed step
        assert payload["vo_history_frame_ids"] == expected_history
        assert payload["history_capture_steps"] == expected_history
        assert payload["vo_trajectory_revision"] == int(revision)
        assert payload["pose_ready"] is (int(revision) > 0)
    assert agent.steps == steps and actions[-1] is Action.STOP and agent.done
    assert agent.calls == len(calls)


def test_warmup_then_ready_payloads() -> None:
    chunks = [("trajectory", [1, 1, 1, 1])] * 7 + [("stop", [0])]
    agent, records = _agent(_chunk_script(chunks))
    agent.reset("Go.", scene_id="lab", episode_id=3)
    _Robot().run(agent)
    plans = _plans(records)
    assert [p["current_capture_step"] for p in plans] == [0, 4, 8, 12, 16, 20, 24, 28]
    first = plans[0]
    assert list(first) == PAYLOAD_KEYS + STEP_KEYS  # warm-up: no history_rel_poses at all
    assert first["num_history"] == 0 and first["pose_ready"] is False
    assert first["vo_provider_phase"] == "insufficient_history"
    assert plans[1]["vo_provider_phase"] == "map_warmup" and plans[4]["pose_ready"] is False
    ready = plans[5]  # step 20: 21 frames ingested
    assert list(ready) == PAYLOAD_KEYS + ["history_rel_poses"] + STEP_KEYS
    assert ready["pose_ready"] is True and np.asarray(ready["history_rel_poses"]).shape == (8, 4)
    assert agent.ppa_warmup_calls == 5 and agent.ppa_applied_calls == 2


def test_arrow_chunk_with_stop_replans_at_the_same_step() -> None:
    lines: list[str] = []
    agent, records = _agent(on_log=lines.append)
    agent.reset("Turn around.", scene_id="lab", episode_id=1)
    robot = _Robot()
    got = [robot.act(agent) for _ in range(4)]
    # call 0 [2,2,2,0]: three turns, then the queued STOP means "replan here": call 1 [1,1,1,3]
    assert got == [Action.TURN_LEFT] * 3 + [Action.FORWARD]
    plans = _plans(records)
    assert [p["current_capture_step"] for p in plans] == [0, 3]
    assert plans[1]["vo_history_frame_ids"] == [0, 1, 2]
    assert robot.lookdown_steps == [0, 3]  # the look-down is taken only for plan calls
    assert "  [debug] local trajectory STOP -> replan" in lines
    assert lines[0] == "  [amb3r-vo] frame=0 history=[] ready=False phase=insufficient_history revision=0"
    assert lines[1] == "  step_id: 0, RPC kind=native_actions, VLM output: ←←←"
    assert lines[2] == "  [debug] actions=[2, 2, 2, 0]"


def test_trajectory_chunk_runs_four_actions_then_replans() -> None:
    agent, records = _agent(_chunk_script([("trajectory", [1, 3, 1, 2]), ("stop", [0])]))
    agent.reset("Go.", scene_id="lab", episode_id=1)
    actions = _Robot().run(agent)
    assert actions == [Action.FORWARD, Action.TURN_RIGHT, Action.FORWARD, Action.TURN_LEFT, Action.STOP]
    assert [p["current_capture_step"] for p in _plans(records)] == [0, 4]
    assert agent.last_call.kind == "stop" and agent.last_step.source == "terminal"
    assert agent.trajectory_calls == 1


def test_look_down_tilts_one_frame() -> None:
    agent, records = _agent(_chunk_script([("native_actions", [5, 1, 0, 0]), ("stop", [0])]))
    agent.reset("Look at the floor.", scene_id="lab", episode_id=1)
    robot = _Robot()
    actions = robot.run(agent)
    assert actions == [Action.LOOK_DOWN, Action.FORWARD, Action.STOP]
    assert Action.LOOK_DOWN.camera_pitch_deg == -30.0 and Action.FORWARD.camera_pitch_deg == 0.0
    assert robot.lookdown_steps == [0, 2] and robot.level_steps == []
    # The frame after LOOK_DOWN (step 1, taken tilted) is what the VO gets for that step.
    ingests = [r for r in records if r["target"] == "vo" and r["method"] == "ingest_frame"]
    assert [json.loads(r["payload"])["capture_step"] for r in ingests] == [0, 1, 2]
    assert ingests[1]["blobs"][0]["sha256"] == hashlib.sha256(pil_jpeg_encoder(_frame(1), quality=95)).hexdigest()
    lookdown_blob = [r for r in records if r["target"] == "model"][1]["blobs"][-1]
    expected = _jpeg(_frame(2, salt=500), (640, 480), 90)
    assert lookdown_blob["name"] == "lookdown" and lookdown_blob["sha256"] == hashlib.sha256(expected).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_look_down_then_queued_stop_replans_from_a_level_frame() -> None:
    """The client's second panorama capture in a step sees the sensor its first capture levelled."""
    chunks = [("native_actions", [3, 5, 0, 0]), ("native_actions", [5, 0, 0, 0]), ("stop", [0])]
    agent, records = _agent(_chunk_script(chunks))
    agent.reset("Look down.", scene_id="lab", episode_id=1)
    robot = _Robot()
    actions = robot.run(agent)
    assert actions == [Action.TURN_RIGHT, Action.LOOK_DOWN, Action.LOOK_DOWN, Action.STOP]
    assert robot.level_steps == [2, 3] and robot.lookdown_steps == [0, 2, 3]
    plans = [r for r in records if r["target"] == "model"]
    blobs = [{b["name"]: b["sha256"] for b in plan["blobs"]} for plan in plans]
    front = lambda step, salt=0: _sha(_jpeg(_frame(step, salt), (384, 384), 90))  # noqa: E731
    # Step 2 (after LOOK_DOWN): the current view is the level re-capture ...
    assert blobs[1]["current/front"] == front(2, salt=700)
    assert json.loads(plans[1]["payload"])["current_capture_step"] == 2
    # ... while the history keeps the step's first, tilted capture, as sample_unique_past_indices does.
    assert json.loads(plans[2]["payload"])["history_capture_steps"] == [0, 1, 2]
    assert blobs[2]["history/2/front"] == front(2) and blobs[2]["current/front"] == front(3, salt=700)
    # The VO still gets the tilted native frame of each step.
    ingests = [r for r in records if r["method"] == "ingest_frame"]
    assert [r["blobs"][0]["sha256"] for r in ingests][2:] == [_sha(pil_jpeg_encoder(_frame(s), quality=95)) for s in (2, 3)]


def test_blobs_match_the_client_encoding() -> None:
    captured = []

    class _Capture(FakeModelServer):
        def _answer(self, method, payload, blobs):
            captured.append(blobs)
            return super()._answer(method, payload, blobs)

    records: list[dict] = []
    agent = NavAgent(
        _Capture(records.append, script=_chunk_script([("native_actions", [3, 3, 0, 0]), ("stop", [0])])),
        FakeVOServer(records.append),
        jpeg_encoder=pil_jpeg_encoder,
    )
    agent.reset("Walk to the rug. ", scene_id="lab", episode_id=7)
    _Robot().run(agent)
    payload = _plans(records)[1]
    assert payload["instruction"] == "Walk to the rug. "  # only a final . ! ? is dropped
    blobs = captured[1]
    names = [b["name"] for b in blobs]
    assert names[:4] == ["current/front", "current/right", "current/back", "current/left"]
    assert names[4:-1] == [f"history/{i}/{v}" for i in range(2) for v in ("front", "right", "back", "left")]
    assert names[-1] == "lookdown"
    assert blobs[0]["data"] == _jpeg(_frame(2), (384, 384), 90)
    assert blobs[4]["data"] == _jpeg(_frame(0), (384, 384), 90)
    assert blobs[8]["data"] == _jpeg(_frame(1), (384, 384), 90)
    black = decode_jpeg(blobs[1]["data"])
    assert black.shape == (384, 384, 3) and int(black.max()) <= 2
    assert all(b["mime_type"] == "image/jpeg" for b in blobs)
    assert (blobs[-1]["width"], blobs[-1]["height"]) == (640, 480)


def test_reset_starts_a_fresh_episode() -> None:
    agent, records = _agent(_chunk_script([("native_actions", [2, 0, 0, 0]), ("stop", [0])]))
    for episode_id in (1, 2):
        agent.reset("Go.", scene_id="lab", episode_id=episode_id)
        _Robot().run(agent)
    resets = [json.loads(r["payload"]) for r in records if r["method"] == "reset_episode"]
    assert resets == [{"session_id": "lab/0001", "max_frames": 501}, {"session_id": "lab/0002", "max_frames": 501}]
    plans = _plans(records)
    assert [p["deterministic_sampling"]["system2_call_index"] for p in plans] == [0, 1, 0, 1]
    assert [p["deterministic_sampling"]["episode_id"] for p in plans] == [1, 1, 2, 2]
    assert plans[2]["num_history"] == 0 and plans[2]["current_capture_step"] == 0


def test_stop_ends_the_episode() -> None:
    agent, _ = _agent(_chunk_script([("stop", [0])]))
    agent.reset("Stay.", scene_id="lab", episode_id=1)
    robot = _Robot()
    assert robot.act(agent) is Action.STOP
    assert agent.done and agent.steps == 1
    with pytest.raises(RuntimeError, match="episode is over"):
        robot.act(agent)


def test_step_cap_returns_stop_without_requests() -> None:
    agent, records = _agent(_chunk_script([("native_actions", [1, 1, 1, 1])] * 3), max_steps=6)
    agent.reset("Go.", scene_id="lab", episode_id=1)
    actions = _Robot().run(agent)
    assert actions == [Action.FORWARD] * 6 + [Action.STOP]
    assert agent.last_step.source == "step_cap" and agent.steps == 6
    resets = [json.loads(r["payload"]) for r in records if r["method"] == "reset_episode"]
    assert resets[0]["max_frames"] == 7
    assert len([r for r in records if r["method"] == "ingest_frame"]) == 6


def test_long_chunk_is_cut_after_eight_actions() -> None:
    agent, records = _agent(_chunk_script([("native_actions", [1] * 10), ("stop", [0])]))
    agent.reset("Go.", scene_id="lab", episode_id=1)
    actions = _Robot().run(agent)
    assert actions == [Action.FORWARD] * 8 + [Action.STOP]
    assert [p["current_capture_step"] for p in _plans(records)] == [0, 8]


def test_camera_is_checked_not_resized() -> None:
    agent, _ = _agent()
    agent.reset("Go.", scene_id="lab", episode_id=1)
    robot = _Robot()
    robot.agent = agent
    with pytest.raises(ValueError, match=r"\(480, 640, 3\)"):
        agent.act(np.zeros((720, 1280, 3), np.uint8), robot.lookdown, robot.level)
    with pytest.raises(RuntimeError, match="call reset"):
        robot.act(agent)
    agent.reset("Go.", scene_id="lab", episode_id=2)
    with pytest.raises(ValueError):
        agent.act(_frame(0).astype(np.float32), robot.lookdown, robot.level)
    agent.reset("Go.", scene_id="lab", episode_id=3)
    with pytest.raises(ValueError, match="lookdown_fn"):
        agent.act(_frame(0), lambda: np.zeros((480, 640, 4), np.uint8), robot.level)
    level_agent, _ = _agent(_chunk_script([("native_actions", [5, 0, 0, 0])]))
    level_agent.reset("Go.", scene_id="lab", episode_id=1)
    robot.act(level_agent)
    with pytest.raises(ValueError, match="level_fn"):
        level_agent.act(_frame(1), robot.lookdown, lambda: np.zeros((240, 320, 3), np.uint8))
    with pytest.raises(ValueError, match="640x480"):
        NavAgent(FakeModelServer(), FakeVOServer(), camera=CameraSpec(width=1280, height=720))
    with pytest.raises(ValueError, match="scene_id"):
        agent.reset("Go.", scene_id="a/b", episode_id=1)


class _Tampered(FakeModelServer):
    def __init__(self, change, **kwargs) -> None:
        super().__init__(**kwargs)
        self.change = change

    def infer_json(self, method, payload, blobs=None):
        result = super().infer_json(method, payload, blobs)
        return self.change(result)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: None, "no response"),
        (lambda r: ({**r[0], "native_front_only": False}, []), "front-only"),
        (lambda r: ({**r[0], "actions": [4, 1]}, []), "cannot execute"),
        (lambda r: ({**r[0], "pose_ready": True}, []), "pose_ready"),
        (lambda r: ({**r[0], "ok": False}, []), "server error"),
        (lambda r: ({**r[0], "deterministic_sampling": None}, []), "sampling"),
    ],
)
def test_server_contract_violations_fail_closed(change, message) -> None:
    agent = NavAgent(_Tampered(change), FakeVOServer(), jpeg_encoder=pil_jpeg_encoder)
    agent.reset("Go.", scene_id="lab", episode_id=1)
    with pytest.raises(RuntimeError, match=message):
        _Robot().act(agent)
    with pytest.raises(RuntimeError, match="call reset"):
        _Robot().act(agent)


def test_timing_changes_no_request() -> None:
    runs = {}
    for timing in (False, True):
        agent, records = _agent(timing=timing, model_timing=timing, vo_timing=timing)
        agent.reset("Go.", scene_id="lab", episode_id=1)
        _Robot().run(agent)
        runs[timing] = (records, agent)
    assert runs[False][0] == runs[True][0]
    off, on = runs[False][1], runs[True][1]
    assert off.last_step.timing_ms == {} and off.last_call.server_timing_ms is None
    assert {"act_total", "vo_ingest", "vo_query", "lookdown_capture", "model_rpc"} <= set(on.last_call.client_timing_ms)
    assert on.last_call.server_timing_ms == {"total": 0.0}
    assert on.last_step.vo_server_timing_ms == {"ingest_frame": {"total": 0.0}, "query_relative_poses": {"total": 0.0}}


def test_act_needs_reset() -> None:
    agent, _ = _agent()
    with pytest.raises(RuntimeError, match="reset"):
        agent.act(_frame(0), lambda: _frame(0), lambda: _frame(0))
