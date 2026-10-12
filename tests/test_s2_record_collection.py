"""EXP-23 collector: --s2_record_dir in the RPC client (design §2 A).

The client (scripts/evaluation/r2r_val_unseen.py) runs end to end here, through its own
``main()``, on fakes: a Habitat env that renders RGB and depth from the agent pose, a
model server whose answers depend on the request bytes, and an AMB3R-VO bridge.  Habitat
and gym are stubbed where they are not installed (the 910B) and used as they are where
they are (the 4090 container); quaternion, torch, yaml and tqdm are real.

The same episodes run once without and once with recording, on both client paths (the
PPA/AMB3R main model, history chosen by VO frame id, and plain native InternNav), and must
give the same env.step sequence, the same requests and the same progress rows.  The
records must then hold exactly that run: the bytes each request carried, the lookdown's
depth in metres, every primitive step's pose, every call's answer.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CLIENT_PATH = REPO / "scripts" / "evaluation" / "r2r_val_unseen.py"

MODEL_ADDR = "127.0.0.1:59001"
VO_ADDR = "127.0.0.1:59002"
FINGERPRINT = "fake-native-fingerprint"
INSTRUCTION = "Walk past the sofa and stop at the door."
CONFIG_YAML = """\
data:
  image_size: [384, 384]
  trajectory:
    traj_image_size: [224, 224]
    panoramic_vlm_input: false
    system2_sft_protocol: internnav
    structured_pano_output: false
model:
  past_plan_action:
    enabled: true
"""


# ── importing the client without the simulator ─────────────────────────────


def _stub(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _ConfigNode:
    """Attribute tree standing in for habitat's yacs Config when habitat is absent."""

    def __init__(self):
        object.__setattr__(self, "_values", {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._values.setdefault(name, _ConfigNode())

    def __setattr__(self, name, value):
        self._values[name] = value

    def defrost(self):
        pass

    def freeze(self):
        pass


def _jpeg_encode(rgb, quality=85):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    for name in ("quaternion", "torch", "yaml", "tqdm"):
        pytest.importorskip(name)
    monkeypatch.setenv("HEATMAPVLN_PREINIT_GL", "0")
    monkeypatch.delenv("HEATMAPVLN_PREINIT_EMPTY_GL", raising=False)
    monkeypatch.delenv("HEATMAPVLN_TIMING", raising=False)
    if importlib.util.find_spec("habitat") is None:
        registry = SimpleNamespace(get_measure=lambda name: object(), register_measure=lambda cls: cls)
        _stub(monkeypatch, "habitat", Env=None, make_dataset=None)
        _stub(monkeypatch, "habitat.config")
        _stub(monkeypatch, "habitat.config.default", Config=_ConfigNode, get_config=_ConfigNode)
        _stub(monkeypatch, "habitat.core")
        _stub(monkeypatch, "habitat.core.embodied_task", EmbodiedTask=object, Measure=object)
        _stub(monkeypatch, "habitat.core.registry", registry=registry)
        _stub(monkeypatch, "habitat.tasks")
        _stub(monkeypatch, "habitat.tasks.nav")
        _stub(monkeypatch, "habitat.tasks.nav.nav", DistanceToGoal=SimpleNamespace(cls_uuid="distance_to_goal"))
    if importlib.util.find_spec("gym") is None:

        class Discrete:
            def __init__(self, n, *args, **kwargs):
                self.n = n

        _stub(monkeypatch, "gym", spaces=_stub(monkeypatch, "gym.spaces", Discrete=Discrete))
    # The request encoder and the RPC client are always fakes: hermetic and byte-stable.
    _stub(monkeypatch, "vla_rpc")
    _stub(monkeypatch, "vla_rpc.core")
    _stub(monkeypatch, "vla_rpc.core.image", encode_rgb_to_jpeg=_jpeg_encode)
    _stub(monkeypatch, "vla_rpc.client", VLAClient=FakeVLAClient)
    # The client installs flash_attn stubs process-wide; put back what was there.
    saved = {key: module for key, module in sys.modules.items() if key.startswith("flash_attn")}
    name = "_s2_collection_test_client"
    spec = importlib.util.spec_from_file_location(name, CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in [key for key in sys.modules if key.startswith("flash_attn")]:
            del sys.modules[key]
        sys.modules.update(saved)


# ── fakes ──────────────────────────────────────────────────────────────────


def _episode(scene: str, episode_id: int, start, goal):
    return SimpleNamespace(
        scene_id=f"mp3d/{scene}/{scene}.glb",
        episode_id=episode_id,
        trajectory_id=100 + episode_id,
        instruction=SimpleNamespace(instruction_text=INSTRUCTION),
        goals=[SimpleNamespace(position=list(goal), radius=3.0)],
        reference_path=[list(start), [start[0], start[1], start[2] - 1.0], list(goal)],
        start_position=list(start),
        start_rotation=[0.0, 0.0, 0.0, 1.0],
        info={"geodesic_distance": 4.0},
    )


EPISODES = (
    ("sceneA", 1, (0.0, 0.1, 0.0), (0.0, 0.1, -4.0)),
    ("sceneB", 7, (2.0, 0.2, 1.0), (5.0, 0.2, 1.0)),
)


class FakeEnv:
    """Habitat as the client uses it; renders RGB and normalised depth from the pose."""

    def __init__(self, quaternion):
        self.q = quaternion
        self.episodes = [_episode(*spec) for spec in EPISODES]
        self._next = 0
        self.current_episode = None
        self.trace: list[tuple] = []  # every env.step
        self.primitive: list[tuple] = []  # every _apply_habitat_action, with the pose after it
        self.lookdown_depths: list[np.ndarray] = []  # the capture lookdown's observation
        self.in_primitive = False
        self._agent = SimpleNamespace(get_state=self._state, set_state=self._set_state)
        self._sim = SimpleNamespace(get_agent=lambda index: self._agent, get_sensor_observations=self._observe)

    @property
    def key(self):
        return (self.current_episode.scene_id.split("/")[-2], int(self.current_episode.episode_id))

    def reset(self):
        self.current_episode = self.episodes[self._next % len(self.episodes)]
        self._next += 1
        self.position = np.asarray(self.current_episode.start_position, dtype=np.float64)
        x, y, z, w = self.current_episode.start_rotation
        self.rotation = self.q.quaternion(w, x, y, z)
        self.tilt = 0
        self.over = False
        self.min_distance = self._distance()
        return self._observe()

    @property
    def episode_over(self):
        return self.over

    def _yaw(self):
        return 2.0 * math.atan2(self.rotation.y, self.rotation.w)

    def _state(self):
        return SimpleNamespace(position=self.position.copy(), rotation=self.rotation, sensor_states={})

    def _set_state(self, state, reset_sensors=True):
        self.position = np.asarray(state.position, dtype=np.float64)
        self.rotation = state.rotation
        if reset_sensors:
            self.tilt = 0

    def _observe(self):
        base = int(round(self.position[0] * 40 + self.position[2] * 70 + math.degrees(self._yaw()) * 3 + self.tilt * 5))
        rows = np.arange(480)[:, None] + np.zeros((1, 640), dtype=np.int64)
        cols = np.arange(640)[None, :] + np.zeros((480, 1), dtype=np.int64)
        rgb = np.stack([(rows + base) % 256, (2 * cols + base) % 256, (rows + cols + 3 * base) % 256], axis=-1)
        depth = (((rows + cols + base) % 97) / 100.0).astype(np.float32)[..., None]
        return {"rgb": rgb.astype(np.uint8), "depth": depth}

    def step(self, action):
        action = int(action)
        self.trace.append((self.key, action))
        if action == 0:
            self.over = True
        elif action == 1:
            yaw = self._yaw()
            self.position = self.position + 0.25 * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
        elif action in (2, 3):
            sign = 1.0 if action == 2 else -1.0
            self.rotation = self.rotation * self.q.from_rotation_vector([0.0, sign * math.radians(15.0), 0.0])
        elif action == 4:
            self.tilt += 15
        elif action == 5:
            self.tilt -= 15
        self.min_distance = min(self.min_distance, self._distance())
        observation = self._observe()
        if action == 5 and self.tilt == -30 and not self.in_primitive:
            self.lookdown_depths.append(observation["depth"].copy())
        return observation

    def _distance(self):
        return float(np.linalg.norm(self.position - np.asarray(self.current_episode.goals[0].position)))

    def get_metrics(self):
        distance = self._distance()
        success = float(self.over and distance <= 3.0)
        return {
            "distance_to_goal": distance,
            "success": success,
            "spl": 0.5 * success,
            "oracle_success": float(self.min_distance <= 3.0),
        }

    def close(self):
        pass


class FakeModelServer:
    """Answers that depend on the request's image bytes, so changed bytes change actions."""

    def __init__(self, client, mode: str, *, mismatch: bool):
        self.client, self.mode, self.mismatch = client, mode, mismatch
        self.requests: list[dict] = []
        self.responses: list[dict] = []

    def info(self):
        if self.mode == "native":
            return SimpleNamespace(
                version=self.client.HEATMAPVLN_RPC_PROTOCOL_VERSION,
                model_version=f"internnav-native-r2r:{FINGERPRINT}",
                supported_formats=[self.client.NATIVE_INTERNNAV_CAPABILITY],
            )
        return SimpleNamespace(
            version=self.client.HEATMAPVLN_RPC_PROTOCOL_VERSION,
            model_version="fake-ppa-refine-v2",
            supported_formats=["ppa-online-amb3r-v1"],
        )

    def answer(self, method, payload, blobs):
        c = self.client
        sent = {str(blob["name"]): bytes(blob["data"]) for blob in blobs}
        self.requests.append({"method": method, "payload": json.loads(json.dumps(payload)), "blobs": sent})
        sampling = payload[c.HEATMAPVLN_RPC_SAMPLING_FIELD]
        call = int(sampling["system2_call_index"])
        digest = hashlib.sha256(b"".join(sent[name] for name in sorted(sent))).digest()
        choice = (digest[0] + int(sampling["per_call_seed"])) % 3
        if call >= 5:
            kind, actions = "stop", [0]
        elif call == 0:
            kind, actions = "trajectory", [1, 1, 0, 0]  # a queued STOP: re-plan at the same step
        elif call == 1 and self.mismatch:
            kind, actions = "native_actions", [5, 0]  # look down, then re-plan at that step
        elif choice == 0:
            kind, actions = "trajectory", [1, 3, 1, 0]
        elif choice == 1:
            kind, actions = "native_actions", [2, 1]
        else:
            kind, actions = "trajectory", [3, 1, 0, 0]
        text = {"stop": "STOP", "native_actions": "←↑"}.get(kind, f"{100 + call} {200 + choice}")
        response = {
            "ok": True,
            "proto_v": c.HEATMAPVLN_RPC_PROTOCOL_VERSION,
            "phase": payload["phase"],
            "kind": kind,
            "actions": actions,
            "terminal": kind == "stop",
            "llm_output": text,
            "native_first_output": "↓" if kind == "trajectory" else text,
            "native_lookdown_turns": int(kind == "trajectory"),
            "pixel_goal": [100 + call, 200 + choice] if kind == "trajectory" else None,
            c.HEATMAPVLN_RPC_SAMPLING_FIELD: dict(sampling),
        }
        if kind == "trajectory":
            response["trajectory_summary"] = f"fake {call}"
        if self.mode == "native":
            response.update(
                {
                    "policy_backend": c.RPC_POLICY_INTERNNAV_NATIVE,
                    "policy_fingerprint": FINGERPRINT,
                    "native_protocol": c.NATIVE_INTERNNAV_CAPABILITY,
                    "native_front_only": True,
                    "native_checkpoint_only": True,
                    "system2_source": "internnav_native",
                    "system1_source": "internnav_native_nextdit_async",
                    "oracle_system2": None,
                    "pano_goal_view": "front",
                }
            )
            if kind == "trajectory":
                response.update({"trajectory_x_sign": 1.0, "trajectory_heading_alignment": "none"})
        else:
            response.update(
                {
                    "ppa_runtime": "ppa-online-amb3r-v1",
                    "pose_provider": c.AMB3R_VO_POSE_PROVIDER,
                    "pose_ready": payload["pose_ready"],
                    "ppa_applied": bool(kind == "trajectory" and payload["pose_ready"]),
                }
            )
        self.responses.append(response)
        return response, []


class FakeVLAClient:
    server: FakeModelServer | None = None
    vo_info = None

    def __init__(self, server_addr, timeout_ms=0, jpeg_quality=0):
        self.address = server_addr

    def connect(self):
        pass

    def health_check(self):
        return True

    def get_server_info(self):
        return FakeVLAClient.vo_info if self.address == VO_ADDR else FakeVLAClient.server.info()

    def infer_json(self, method, payload, blobs):
        assert self.address == MODEL_ADDR
        return FakeVLAClient.server.answer(method, payload, blobs)

    def close(self):
        pass


class FakeVOBridge:
    """OnlineVORPCBridge's interface: one frame id per capture step, poses ready from frame 3."""

    def __init__(self, client, *, jpeg_quality=95, timing=False):
        self.client = client
        self.frames: dict[int, int] = {}

    def reset_episode(self, session_id, *, max_frames):
        self.frames = {}

    def ingest_rgb(self, frame_rgb, *, capture_step):
        return self.frames.setdefault(int(capture_step), len(self.frames))

    def query_model_pose_fields(self, *, current_frame_id, history_frame_ids):
        return {
            "vo_current_frame_id": int(current_frame_id),
            "vo_history_frame_ids": [int(frame) for frame in history_frame_ids],
            "pose_ready": bool(current_frame_id >= 3),
            "vo_provider_phase": "fake",
            "vo_trajectory_revision": 0,
        }

    def drain_timing(self):
        return []


def _run(client, monkeypatch, tmp_path: Path, *, mode: str, record: bool, tag: str, mismatch: bool = False):
    quaternion = pytest.importorskip("quaternion")
    env = FakeEnv(quaternion)
    server = FakeModelServer(client, mode, mismatch=mismatch)
    FakeVLAClient.server = server
    FakeVLAClient.vo_info = SimpleNamespace(
        version=client.AMB3R_VO_RPC_PROTOCOL_VERSION, model_version="amb3r-vo-da3-online", supported_formats=["json+jpeg"]
    )
    original_apply = client.__dict__.setdefault("_unwrapped_apply_habitat_action", client._apply_habitat_action)

    def apply_and_log(env_, action):
        env_.in_primitive = True
        try:
            result = original_apply(env_, action)
        finally:
            env_.in_primitive = False
        env_.primitive.append(
            (env_.key, int(action), env_.position.copy(), quaternion.as_float_array(env_.rotation))
        )
        return result

    monkeypatch.setattr(client, "_apply_habitat_action", apply_and_log)
    monkeypatch.setattr(client, "_create_habitat_env", lambda hab_cfg, args: env)
    monkeypatch.setattr(client, "_resolve_eval_paths", lambda args, split="val_unseen": None)
    monkeypatch.setattr(client, "OnlineVORPCBridge", FakeVOBridge)
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG_YAML)
    output = tmp_path / f"out_{tag}"
    records = tmp_path / f"records_{tag}"
    argv = [
        "r2r_val_unseen.py",
        "--config", str(config),
        "--rpc_server", MODEL_ADDR,
        "--output_path", str(output),
        "--dataset_split", "train",
        "--data_path", str(tmp_path / "unused.json.gz"),
        "--scenes_dir", str(tmp_path),
        "--rpc_protocol_seed", "42", "--rpc_require_deterministic_sampling",
        "--rpc_jpeg_quality", "90", "--num_history", "8",
        "--max_steps_per_episode", "80", "--max_system2_calls_per_episode", "0",
        "--auto_stop_distance", "0", "--trajectory_selection", "mean",
        "--trajectory_x_sign", "1", "--trajectory_heading_alignment", "none",
        "--system1_coord_order", "generated", "--no-pano_recenter_before_system1",
        "--no-debug_input_trace",
    ]
    if mode == "native":
        argv += ["--rpc_policy_mode", "internnav_native", "--rpc_policy_fingerprint", FINGERPRINT]
    else:
        argv += ["--rpc_policy_mode", "heatmapvln", "--history_pose_source", "amb3r_vo_da3",
                 "--amb3r_vo_rpc_server", VO_ADDR]
    if record:
        argv += ["--s2_record_dir", str(records)]
    monkeypatch.setattr(sys, "argv", argv)
    client.main()
    progress = [json.loads(line) for line in (output / "progress.json").read_text().splitlines() if line.strip()]
    return SimpleNamespace(env=env, server=server, progress=progress, records=records)


def _without_record(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "s2_record"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── end to end ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["vo", "native"])
@pytest.mark.parametrize("mismatch", [False, True], ids=["plain", "relooked_step"])
def test_recording_changes_no_action_and_records_exactly_the_run(client, monkeypatch, tmp_path, mode, mismatch):
    plain = _run(client, monkeypatch, tmp_path, mode=mode, record=False, tag="plain", mismatch=mismatch)
    recorded = _run(client, monkeypatch, tmp_path, mode=mode, record=True, tag="recorded", mismatch=mismatch)

    # Same seed, same behaviour: every env.step, every request byte, every outcome.
    assert recorded.env.trace == plain.env.trace
    assert [r["method"] for r in recorded.server.requests] == [r["method"] for r in plain.server.requests]
    assert [r["payload"] for r in recorded.server.requests] == [r["payload"] for r in plain.server.requests]
    assert [r["blobs"] for r in recorded.server.requests] == [r["blobs"] for r in plain.server.requests]
    assert [_without_record(row) for row in recorded.progress] == plain.progress
    assert not plain.records.exists()
    assert len(recorded.progress) == len(EPISODES)

    records = client._load_s2_records_module()
    assert len(list(records.iter_record_paths(recorded.records))) == len(EPISODES)
    requests = list(zip(recorded.server.requests, recorded.server.responses))
    depths = list(recorded.env.lookdown_depths)
    assert len(depths) == len(requests)
    mismatched_calls = 0
    for row in recorded.progress:
        key = (row["scene_id"], int(row["episode_id"]))
        record = records.EpisodeRecord(row["s2_record"]["path"], verify=True)
        assert Path(row["s2_record"]["path"]) == records.record_path(recorded.records, *key)
        mine = [
            (request, response)
            for request, response in requests
            if (request["payload"]["deterministic_sampling"]["scene_id"],
                request["payload"]["deterministic_sampling"]["episode_id"]) == key
        ]
        my_depths = [
            depth
            for depth, (request, _) in zip(depths, requests)
            if (request["payload"]["deterministic_sampling"]["scene_id"],
                request["payload"]["deterministic_sampling"]["episode_id"]) == key
        ]
        assert len(record.calls) == len(mine) == row["vlm_calls"] == row["s2_record"]["num_calls"]
        for call, (request, response), depth in zip(record.calls, mine, my_depths):
            blobs = request["blobs"]
            names = [f"history/{i}/front" for i in range(len(call["history_step_ids"]))] + ["current/front"]
            steps = call["history_step_ids"] + [call["step_id"]]
            assert sum(name.startswith("history/") and name.endswith("/front") for name in blobs) == len(names) - 1
            assert call["front_sha256"] == [_sha(blobs[name]) for name in names]
            # The stored front is what the call sent, except where the call says it is not.
            differing = sorted({step for step, name in zip(steps, names) if record.front_jpeg(step) != blobs[name]})
            assert differing == sorted(call["front_mismatch_steps"])
            assert len(call["front_mismatch_mean_abs_diff"]) == len(call["front_mismatch_max_abs_diff"]) == len(differing)
            assert all(value > 0 for value in call["front_mismatch_max_abs_diff"])
            mismatched_calls += bool(call["front_mismatch_steps"])
            assert record.lookdown_jpeg(call["call_index"]) == blobs["lookdown"]
            assert call["lookdown_sha256"] == _sha(blobs["lookdown"])
            assert call["lookdown_size"] == [640, 480]
            np.testing.assert_allclose(record.lookdown_depth(call["call_index"]), depth[..., 0] * 10.0, atol=6e-4)
            for field in ("llm_output", "native_first_output", "native_lookdown_turns", "kind", "pixel_goal",
                          "actions", "terminal"):
                assert call[field] == response[field]
            assert call["sampling"] == response["deterministic_sampling"]
            if mode == "vo":
                assert call["step_id"] == request["payload"]["current_capture_step"]
                assert call["history_step_ids"] == request["payload"]["history_capture_steps"]
                assert call["vo"]["pose_ready"] is request["payload"]["pose_ready"]
                assert call["server_fields"]["ppa_runtime"] == "ppa-online-amb3r-v1"
            else:
                assert call["vo"] is None
                assert call["server_fields"]["policy_fingerprint"] == FINGERPRINT
            assert call["position"] == pytest.approx(record.step(call["step_id"])["position"])
            assert call["rotation_wxyz"] == pytest.approx(record.step(call["step_id"])["rotation_wxyz"])

        # One steps.jsonl row per primitive step, with the pose Habitat had after it.
        primitive = [item for item in recorded.env.primitive if item[0] == key]
        assert [s["step_id"] for s in record.steps] == list(range(row["steps"] + 1))
        assert [s["action"] for s in record.steps] == [None] + [action for _, action, _, _ in primitive]
        for step_row, (_, _, position, rotation) in zip(record.steps[1:], primitive):
            assert step_row["position"] == pytest.approx(position.tolist())
            assert step_row["rotation_wxyz"] == pytest.approx(rotation.tolist())
        episode_spec = next(spec for spec in EPISODES if spec[:2] == key)
        assert record.steps[0]["position"] == pytest.approx(list(episode_spec[2]))
        assert record.steps[0]["rotation_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])

        meta = record.episode
        assert (meta["scene_id"], meta["episode_id"], meta["trajectory_id"]) == (*key, 100 + key[1])
        assert meta["split"] == "train"
        assert meta["instruction"] == INSTRUCTION[:-1] and meta["instruction_raw"] == INSTRUCTION
        assert meta["goals"] == [{"position": pytest.approx(list(episode_spec[3])), "radius": 3.0}]
        assert len(meta["reference_path"]) == 3
        assert meta["start_position"] == pytest.approx(list(episode_spec[2]))
        assert meta["protocol_seed"] == 42 and meta["num_history"] == 8 and meta["rpc_jpeg_quality"] == 90
        assert meta["vlm_image_size"] == [384, 384] and meta["traj_image_size"] == [224, 224]
        assert meta["lookdown_image_size"] == [640, 480]
        assert meta["server"]["model_version"] == FakeVLAClient.server.info().model_version
        assert meta["camera"] == records.CAMERA
        outcome = meta["outcome"]
        assert outcome["success"] == row["success"] and outcome["spl"] == row["spl"]
        assert outcome["distance_to_goal"] == row["ne"] and outcome["oracle_success"] == row["os"]
        assert outcome["steps"] == row["steps"] and outcome["system2_calls"] == row["vlm_calls"]
        assert outcome["trajectory_calls"] == row["trajectory_calls"]
        assert outcome["episode_over"] is True
        positions = np.asarray([s["position"] for s in record.steps])
        assert outcome["path_length"] == pytest.approx(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
        assert outcome["front_mismatch_calls"] == row["s2_record"]["front_mismatch_calls"]

    # A step looked down at and then re-captured level is sent twice with different
    # bytes: the record keeps the first and names the calls it cannot reproduce.
    assert (mismatched_calls > 0) is mismatch


def test_each_step_keeps_the_bytes_most_calls_sent_and_flags_the_rest(client, tmp_path):
    # The real-stack pattern: a step goes out once as one call's current view and then,
    # captured differently, as history of the later calls.
    hab_cfg = client.build_habitat_config(SimpleNamespace(dataset_split="train", scenes_dir="s", data_path="d", sim_gpu_id=0))
    collector = client.S2RecordCollector(tmp_path, hab_cfg=hab_cfg, run_meta={"split": "train"})
    env = FakeEnv(pytest.importorskip("quaternion"))
    env.reset()
    collector.begin_episode(env, env.current_episode, scene_id="sceneA", episode_id=1, instruction="go")
    jpeg = lambda value: _jpeg_encode(np.full((48, 64, 3), value, dtype=np.uint8), quality=90)
    clean, first, step1, step2, tie_a, tie_b = jpeg(100), jpeg(104), jpeg(30), jpeg(60), jpeg(200), jpeg(210)
    depth = np.full((480, 640, 1), 0.25, dtype=np.float32)
    response = {"llm_output": "←", "kind": "native_actions", "actions": [2], "terminal": False}

    def call(index, step, current, history):
        blobs = [{"name": "current/front", "data": current}, {"name": "lookdown", "data": b"L", "width": 640, "height": 480}]
        blobs += [{"name": f"history/{i}/front", "data": data} for i, (_, data) in enumerate(history)]
        collector.add_call(env, call_index=index, step_id=step, history_step_ids=[s for s, _ in history],
                           sent_blobs=blobs, lookdown_depth=depth, response=response, pose_fields=None)

    call(0, 0, clean, [])
    call(1, 1, step1, [(0, first)])
    call(2, 2, step2, [(0, first), (1, step1)])
    call(3, 3, tie_a, [(0, first)])
    call(4, 4, step2, [(3, tie_b)])  # one call each way: the first sent wins
    summary = collector.end_episode(metrics=env.get_metrics(), steps=0, system2_calls=5, trajectory_calls=0,
                                    episode_over=False)
    record = client._load_s2_records_module().EpisodeRecord(summary["path"], verify=True)
    assert record.front_jpeg(0) == first and record.front_jpeg(3) == tie_a
    assert [c["front_mismatch_steps"] for c in record.calls] == [[0], [], [], [], [3]]
    assert summary["front_mismatch_calls"] == record.episode["outcome"]["front_mismatch_calls"] == 2
    assert 3.0 <= record.calls[0]["front_mismatch_mean_abs_diff"][0] <= 5.0
    assert record.calls[4]["front_mismatch_max_abs_diff"][0] >= 9
    assert record.calls[2]["front_sha256"] == [_sha(first), _sha(step1), _sha(step2)]
    assert [c["call_index"] for c in record.calls] == list(range(5))
    np.testing.assert_allclose(record.lookdown_depth(0), 2.5)
    assert [s["step_id"] for s in record.steps] == [0] and record.steps[0]["phase"] == "start"


def test_a_call_whose_blobs_disagree_with_its_history_or_lacks_depth_fails(client, tmp_path):
    hab_cfg = client.build_habitat_config(SimpleNamespace(dataset_split="train", scenes_dir="s", data_path="d", sim_gpu_id=0))
    collector = client.S2RecordCollector(tmp_path, hab_cfg=hab_cfg, run_meta={})
    env = FakeEnv(pytest.importorskip("quaternion"))
    env.reset()
    collector.begin_episode(env, env.current_episode, scene_id="sceneA", episode_id=1, instruction="go")
    blobs = [{"name": "current/front", "data": b"A"}, {"name": "lookdown", "data": b"L", "width": 640, "height": 480}]
    common = dict(call_index=0, step_id=0, response={"kind": "stop", "llm_output": "STOP"}, pose_fields=None)
    with pytest.raises(RuntimeError, match="do not match the prompt history"):
        collector.add_call(env, history_step_ids=[0], sent_blobs=blobs, lookdown_depth=np.zeros((480, 640)), **common)
    with pytest.raises(RuntimeError, match="no 'depth'"):
        collector.add_call(env, history_step_ids=[], sent_blobs=blobs, lookdown_depth=None, **common)


def test_the_record_camera_is_the_client_habitat_camera(client):
    hab_cfg = client.build_habitat_config(SimpleNamespace(dataset_split="train", scenes_dir="s", data_path="d", sim_gpu_id=0))
    records = client._load_s2_records_module()
    assert client.S2RecordCollector._check_camera(hab_cfg, records.CAMERA) == 10.0
    sim = hab_cfg.SIMULATOR
    sensor = lambda s, **kw: SimpleNamespace(**{**{k: getattr(s, k) for k in (
        "WIDTH", "HEIGHT", "HFOV", "POSITION")}, **kw})
    depth = SimpleNamespace(**vars(sensor(sim.DEPTH_SENSOR)), MIN_DEPTH=sim.DEPTH_SENSOR.MIN_DEPTH,
                            MAX_DEPTH=sim.DEPTH_SENSOR.MAX_DEPTH, NORMALIZE_DEPTH=sim.DEPTH_SENSOR.NORMALIZE_DEPTH)
    changed = SimpleNamespace(SIMULATOR=SimpleNamespace(
        RGB_SENSOR=sensor(sim.RGB_SENSOR, HFOV=90), DEPTH_SENSOR=depth, TILT_ANGLE=sim.TILT_ANGLE))
    with pytest.raises(ValueError, match="hfov_deg"):
        client.S2RecordCollector._check_camera(changed, records.CAMERA)


# ── preflight ──────────────────────────────────────────────────────────────


def _preflight_args(**overrides):
    args = dict(
        s2_record_dir="records",
        rpc_server="127.0.0.1:1",
        oracle_system2=False,
        auto_stop_distance=0.0,
        pano_recenter_before_system1=False,
        collect_trajectory_dagger=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    "overrides, phrase",
    [
        ({"rpc_server": ""}, "requires --rpc_server"),
        ({"oracle_system2": True}, "--oracle_system2 replaces them"),
        ({"auto_stop_distance": 3.0}, "--auto_stop_distance 0"),
        ({"pano_recenter_before_system1": True}, "two-phase recenter"),
        ({"collect_trajectory_dagger": True}, "separate collections"),
    ],
)
def test_preflight_refuses_runs_that_are_not_the_policy(client, overrides, phrase):
    with pytest.raises(ValueError, match=phrase):
        client._preflight_s2_record_args(_preflight_args(**overrides))


def test_preflight_is_silent_when_off_and_resolves_the_directory(client, tmp_path, monkeypatch):
    off = _preflight_args(s2_record_dir=None, rpc_server="", auto_stop_distance=3.0)
    client._preflight_s2_record_args(off)
    assert off.s2_record_dir is None
    monkeypatch.chdir(tmp_path)
    on = _preflight_args()
    client._preflight_s2_record_args(on)
    assert on.s2_record_dir == str((tmp_path / "records").resolve())


# ── the wiring, on the source ──────────────────────────────────────────────


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _is_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "s2_recorder"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def test_every_recorder_use_in_the_loop_is_behind_its_guard():
    tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
    fn = _function(tree, "run_eval_rpc_panoramic")
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    uses = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "s2_recorder"
    ]
    assert uses
    for use in uses:
        node = use
        while node in parents and not (isinstance(node, ast.If) and _is_guard(node.test)):
            node = parents[node]
        assert isinstance(node, ast.If) and _is_guard(node.test), f"unguarded s2_recorder use at line {use.lineno}"
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
    # Every primitive-step site that feeds the step recorder feeds the s2 record too.
    post_steps = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "_record_post_action_step"]
    add_steps = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "add_step"
                 and isinstance(c.func.value, ast.Name) and c.func.value.id == "s2_recorder"]
    assert len(add_steps) == len(post_steps) == 7
    # Off, the two hooks into the request path are None: the calls are the pre-EXP-23 ones.
    for callee, keyword, name in (
        ("_rpc_plan_panoramic", "sent_blobs", "s2_sent_blobs"),
        ("capture_lookdown_view", "depth_out", "s2_lookdown_depth"),
    ):
        hooked = [
            c for c in calls
            if isinstance(c.func, ast.Name) and c.func.id == callee
            and any(k.arg == keyword for k in c.keywords)
        ]
        assert len(hooked) == 1
        assert ast.unparse(next(k.value for k in hooked[0].keywords if k.arg == keyword)) == name
        assignment = next(
            node for node in ast.walk(fn)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name
        )
        assert ast.unparse(assignment.value) == "[] if s2_recorder is not None else None"
