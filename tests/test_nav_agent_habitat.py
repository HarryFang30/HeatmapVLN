"""NavAgent vs the deployed Habitat client's own code (runs in the fjl-habitat container).

1. Plan-call payload: for the same frames, NavAgent's request equals what the client's
   _rpc_plan_panoramic builds (JSON text and every JPEG byte), with the client's image
   conversion, JPEG encoder, image sizes from the deployed config and the launcher's flags.
   The client has no side views to give here, so both get NavAgent's black placeholders.
2. VO ingest: the client's OnlineVORPCBridge sends the same bytes for the same frames.
3. Closed loop (needs an X display and NAV_AGENT_HABITAT_DATA / NAV_AGENT_HABITAT_EPISODES):
   scripts/deploy/nav_agent_habitat_check.py runs the client's own loop and then NavAgent in
   Habitat against the same scripted servers; every request must match except the rendered
   side views, which the client sends and NavAgent replaces (layout still checked).

Skipped where Habitat or vla_rpc is missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("habitat")
pytest.importorskip("vla_rpc.core.image")

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.deploy.fake_servers import FakeModelServer, FakeVOServer, default_script  # noqa: E402
from src.deploy.nav_agent import NavAgent  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "ppa_action_refine_v2_8gpu.yaml"
POSE_KEYS = {
    "pose_provider", "pose_ready", "vo_current_frame_id", "vo_history_frame_ids",
    "vo_provider_phase", "vo_trajectory_revision", "history_rel_poses",
}


@pytest.fixture(scope="module")
def client():
    # Imported here, not at collection: the client patches flash_attn, gym and numpy on import.
    import scripts.evaluation.r2r_val_unseen as module

    return module


def _frame(step: int, salt: int = 0) -> np.ndarray:
    return np.random.default_rng(1000 * salt + step).integers(0, 256, (480, 640, 3), dtype=np.uint8)


class _Wire:
    """Keeps each request as sent: method, JSON text, blobs with their bytes."""

    def __init__(self, server) -> None:
        self.server = server
        self.requests = []

    def __getattr__(self, name):
        return getattr(self.server, name)

    def infer_json(self, method, payload, blobs=None):
        self.requests.append((method, json.dumps(payload, ensure_ascii=False), list(blobs or [])))
        return self.server.infer_json(method, payload, blobs)


def _run_nav_agent(instruction: str, episode_id: int):
    model, vo = _Wire(FakeModelServer(script=default_script)), _Wire(FakeVOServer())
    agent = NavAgent(model, vo)  # vla_rpc's encoder, as the client uses
    agent.reset(instruction, scene_id="zsNo4HB9uLZ", episode_id=episode_id)
    levels: dict[int, np.ndarray] = {}  # step -> level re-capture (replan right after LOOK_DOWN)

    def level() -> np.ndarray:
        return levels.setdefault(agent.steps, _frame(agent.steps, salt=700))

    while not agent.done:
        agent.act(_frame(agent.steps), lambda: _frame(agent.steps, salt=500), level)
    return model.requests, vo.requests, levels


def test_plan_payload_matches_the_client_builder(client) -> None:
    instruction = "Exit the bedroom and turn left. Walk straight and stop near the rug."
    model_requests, _, levels = _run_nav_agent(instruction, episode_id=1)
    with CONFIG.open() as handle:
        vlm_size, traj_size = client._eval_image_sizes(yaml.safe_load(handle))
    black = Image.new("RGB", vlm_size, color=(0, 0, 0))

    def views(step, front=None):
        front = _frame(step) if front is None else front
        return {"front": client._rgb_array_to_pil(front, vlm_size), "right": black, "back": black, "left": black}

    assert len(model_requests) == 13 and sorted(levels) == [11, 20]
    for method, text, blobs in model_requests:
        payload = json.loads(text)
        step = payload["current_capture_step"]
        rebuilt = []

        class _Recorder:
            def infer_json(self, m, p, b):
                rebuilt.append((m, json.dumps(p, ensure_ascii=False), b))
                return FakeModelServer(script=default_script).infer_json(m, p, b)

        client._rpc_plan_panoramic(
            _Recorder(),
            instruction=client._normalize_instruction(instruction),
            # After LOOK_DOWN a replan in the same step re-captures level; history keeps the first capture.
            current_views=views(step, levels.get(step)),
            history_panoramas=[views(s) for s in payload["history_capture_steps"]],
            # Two-turn InternNav protocol: the 640x480 conversational look-down (the client's plan branch)
            lookdown_img=client._rgb_array_to_pil(_frame(step, salt=500), client.NATIVE_INTERNNAV_LOOKDOWN_SIZE),
            vlm_image_size=vlm_size,
            traj_image_size=traj_size,
            # The deployed launcher's client flags (scripts/run_ppa_r2r_val_unseen_cuda.sh run_shard).
            system1_coord_order="generated",
            trajectory_selection="mean",
            trajectory_x_sign=1.0,
            trajectory_heading_alignment="none",
            jpeg_quality=90,
            scene_id="zsNo4HB9uLZ",
            episode_id=1,
            system2_call_index=payload["deterministic_sampling"]["system2_call_index"],
            protocol_seed=42,
            require_deterministic_sampling=True,
            rpc_policy_mode=client.RPC_POLICY_HEATMAPVLN,
            phase="joint",
            model_pose_fields={k: v for k, v in payload.items() if k in POSE_KEYS},
            current_capture_step=step,
            history_capture_steps=payload["history_capture_steps"],
        )
        (m, rebuilt_text, rebuilt_blobs), = rebuilt
        assert (m, rebuilt_text) == (method, text)
        assert len(rebuilt_blobs) == len(blobs)
        for ours, theirs in zip(blobs, rebuilt_blobs):
            assert ours == theirs, ours["name"]


def test_vo_ingest_matches_the_client_bridge(client) -> None:
    _, vo_requests, _ = _run_nav_agent("Go.", episode_id=2)
    wire = _Wire(FakeVOServer())
    bridge = client.OnlineVORPCBridge(wire, jpeg_quality=95)
    bridge.reset_episode("zsNo4HB9uLZ/0002", max_frames=501)
    ingests = [r for r in vo_requests if r[0] == "ingest_frame"]
    for method, text, blobs in ingests:
        bridge.ingest_rgb(_frame(json.loads(text)["capture_step"]), capture_step=json.loads(text)["capture_step"])
    theirs = [r for r in wire.requests if r[0] == "ingest_frame"]
    assert [(m, t) for m, t, _ in theirs] == [(m, t) for m, t, _ in ingests]
    assert [b for _, _, b in theirs] == [b for _, _, b in ingests]


@pytest.mark.skipif(
    not (os.environ.get("DISPLAY") and os.environ.get("NAV_AGENT_HABITAT_DATA") and os.environ.get("NAV_AGENT_HABITAT_EPISODES")),
    reason="closed loop needs DISPLAY, NAV_AGENT_HABITAT_DATA and NAV_AGENT_HABITAT_EPISODES",
)
def test_closed_loop_requests_match_the_client(tmp_path) -> None:
    script = REPO / "scripts" / "deploy" / "nav_agent_habitat_check.py"
    common = [
        sys.executable, str(script), "--fake-servers",
        "--data_path", os.environ["NAV_AGENT_HABITAT_DATA"],
        "--episode_list", os.environ["NAV_AGENT_HABITAT_EPISODES"],
        "--scenes_dir", os.environ.get("NAV_AGENT_HABITAT_SCENES", "/dataset"),
        "--max_episodes", os.environ.get("NAV_AGENT_HABITAT_MAX_EPISODES", "1"),
    ]
    ref = subprocess.run(
        common + ["--reference-client", "--output_path", str(tmp_path / "reference"),
                  "--record-requests", str(tmp_path / "reference.jsonl")],
        capture_output=True, text=True, timeout=3600,
    )
    assert ref.returncode == 0, ref.stdout[-3000:] + ref.stderr[-3000:]
    ours = subprocess.run(
        common + ["--output_path", str(tmp_path / "nav_agent"), "--record-requests", str(tmp_path / "nav_agent.jsonl"),
                  "--compare-requests", str(tmp_path / "reference.jsonl"),
                  "--compare-log", str(tmp_path / "reference" / "client.log")],
        capture_output=True, text=True, timeout=3600,
    )
    assert ours.returncode == 0, ours.stdout[-3000:] + ours.stderr[-3000:]
    result = json.loads((tmp_path / "nav_agent" / "compare_requests.json").read_text())
    assert result["identical"] and result["requests"]["ours"] > 50
    logs = json.loads((tmp_path / "nav_agent" / "compare_log.json").read_text())
    assert logs["identical"] and logs["episodes"]
