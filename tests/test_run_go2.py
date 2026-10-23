"""scripts/deploy/run_go2.py: the Go2 runner on its front camera, against a stand-in of the lab's base
(ROSPlatformClient's methods as docs/deploy/navigation_interface.md §6 uses them) and the fake servers.

No GPU, robot, ROS or vla_rpc: runs with ``--noconftest`` on the Orin's Python 3.8.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.deploy.fake_servers import FakeModelServer, FakeVOServer, pil_jpeg_encoder
from src.deploy.nav_agent import NavAgent

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy" / "run_go2.py"
spec = importlib.util.spec_from_file_location("run_go2", SCRIPT)
run_go2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_go2)


def _agent(script=None):
    model = FakeModelServer() if script is None else FakeModelServer(script=script)
    return NavAgent(model, FakeVOServer(), camera=run_go2.FRONT_CAMERA_MOUNT, jpeg_encoder=pil_jpeg_encoder)


def _records(out: Path):
    return [json.loads(line) for line in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_crop_sees_79_degrees_across_at_4_by_3():
    left, top, right, bottom = run_go2.crop_box(1920, 1080, 100.0)
    width = 1920 * math.tan(math.radians(39.5)) / math.tan(math.radians(50.0))
    assert right - left == round(width) and bottom - top == round(width * 3 / 4)
    assert left == round((1920 - width) / 2) and top == round((1080 - width * 3 / 4) / 2)  # centred
    assert run_go2.crop_box(640, 480, 79.0) == (0, 0, 640, 480)  # the simulated camera: nothing to crop
    with pytest.raises(ValueError, match="at least 79"):
        run_go2.crop_box(1920, 1080, 70.0)
    with pytest.raises(ValueError, match="too short"):
        run_go2.crop_box(1920, 1080, 90.0)  # 16:9 needs about 95.5 deg across for a 4:3 crop


def test_a_front_camera_jpeg_becomes_a_640x480_rgb_frame():
    robot = run_go2.OfflineRobot()
    frame = run_go2.to_model_frame(robot.jpeg, 100.0)
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8 and frame.flags["C_CONTIGUOUS"]
    # Red runs left to right across the 1920 columns: the crop keeps the middle 1328 of them.
    left, _, right, _ = run_go2.crop_box(1920, 1080, 100.0)
    assert abs(int(frame[240, 0, 0]) - left % 256) <= 6 and abs(int(frame[240, -1, 0]) - (right - 1) % 256) <= 6


def test_an_episode_moves_the_dog_once_per_motion_and_hands_control_back(tmp_path):
    robot = run_go2.OfflineRobot()
    log = run_go2.RunLog(tmp_path)
    summary = run_go2.run_episode(robot, _agent(), instruction="Walk to the door.", run_id=7, log=log, say=lambda _: None)
    log.close()
    records = _records(tmp_path)
    actions = [r["action"] for r in records]
    # fake_servers.default_script: 35 act() calls (29 motions, 5 LOOK_DOWNs, the STOP), STOP at call 12.
    assert summary["outcome"] == "model_stop" and summary["steps"] == 35 and len(records) == 35
    assert actions.count("LOOK_DOWN") == 5 and actions[-1] == "STOP"
    expected = []
    for name in actions:
        if name == "FORWARD":
            expected.append((0.25, 0.0))
        elif name in ("TURN_LEFT", "TURN_RIGHT"):
            expected.append((0.0, (1 if name == "TURN_LEFT" else -1) * math.radians(15.0)))
    assert robot.motions == expected and len(expected) == 29
    assert all((r["motion"] is None) == (r["action"] in ("LOOK_DOWN", "STOP")) for r in records)
    assert robot.acquired == robot.released == 1
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [f"step_{i:04d}.jpg" for i in range(35)]
    assert (tmp_path / "frames" / "step_0000.jpg").read_bytes() == robot.jpeg  # as received, not re-encoded
    assert json.loads((tmp_path / "summary.json").read_text())["outcome"] == "model_stop"


def test_a_dry_run_takes_frames_and_plans_but_never_takes_control(tmp_path):
    robot = run_go2.OfflineRobot()
    summary = run_go2.run_episode(robot, _agent(), instruction="Walk to the door.", run_id=8, move=False,
                                  log=run_go2.RunLog(tmp_path), say=lambda _: None)
    assert summary["outcome"] == "model_stop" and summary["steps"] == 35
    assert robot.motions == [] and robot.acquired == robot.released == 0
    assert all(r["motion"] is None for r in _records(tmp_path))


def test_a_failed_motion_ends_the_episode_and_still_releases(tmp_path):
    class Stumbling(run_go2.OfflineRobot):
        def move_relative(self, distance=0.0, angle=0.0):
            reply = super().move_relative(distance, angle)
            return dict(reply, state="failed", reason="lease_expired") if len(self.motions) == 3 else reply

    robot = Stumbling()
    with pytest.raises(RuntimeError, match="relative motion failed"):
        run_go2.run_episode(robot, _agent(), instruction="Walk.", run_id=9, log=run_go2.RunLog(tmp_path),
                            say=lambda _: None)
    assert robot.released == 1 and len(robot.motions) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["outcome"] == "failed"


def test_look_down_over_and_over_is_a_stuck_episode(tmp_path):
    robot = run_go2.OfflineRobot()
    always_down = lambda index, payload: {"kind": "native_actions", "llm_output": "↓", "actions": [5]}  # noqa: E731
    with pytest.raises(RuntimeError, match="LOOK_DOWN 3 times in a row"):
        run_go2.run_episode(robot, _agent(always_down), instruction="Walk.", run_id=10, say=lambda _: None)
    assert robot.motions == [] and robot.released == 1


def test_the_offline_check_passes(tmp_path):
    assert run_go2.main(["--offline", "--out", str(tmp_path / "run")]) == 0
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["outcome"] == "model_stop"


def test_frames_the_base_marks_unavailable_never_reach_the_model(tmp_path):
    class WarmingUp(run_go2.OfflineRobot):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_frame(self, camera, timeout, wait_new):
            self.calls += 1
            frame = super().get_frame(camera, timeout, wait_new)
            if self.calls <= 3:
                return {"jpeg": b"not a jpeg", "metadata": {"available": False, "warming_up": True}}
            return frame

    robot = WarmingUp()
    assert run_go2.next_jpeg(robot, "front_camera", 10.0) == robot.jpeg and robot.calls == 4

    class Broken(run_go2.OfflineRobot):
        def get_frame(self, camera, timeout, wait_new):
            return {"jpeg": b"", "metadata": {"available": False, "error": "videohub down"}}

    with pytest.raises(RuntimeError, match="stays unavailable"):
        run_go2.next_jpeg(Broken(), "front_camera", 0.05)


def test_a_frame_is_taken_twice_after_each_motion_and_the_second_is_used(tmp_path):
    class Counting(run_go2.OfflineRobot):
        def __init__(self):
            super().__init__()
            self.frames = 0
            buffer = io.BytesIO()
            Image.new("RGB", (1920, 1080), (200, 10, 10)).save(buffer, format="JPEG")
            self.second = buffer.getvalue()

        def get_frame(self, camera, timeout, wait_new):
            self.frames += 1
            assert camera == "front_camera" and wait_new and timeout >= 8
            return {"jpeg": self.jpeg if self.frames % 2 else self.second, "metadata": {}}

    robot = Counting()
    run_go2.run_episode(robot, _agent(), instruction="Walk.", run_id=11, log=run_go2.RunLog(tmp_path),
                        say=lambda _: None)
    assert robot.frames == 2 * 35
    assert all(p.read_bytes() == robot.second for p in (tmp_path / "frames").iterdir())
