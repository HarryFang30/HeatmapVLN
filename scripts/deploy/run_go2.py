#!/usr/bin/env python3
"""Run one navigation episode on the Go2 through the lab's ROS 2 base (robot_platform_ws).

The dog's own front camera ('front_camera') is the only camera.  Every step takes a fresh frame
after the last motion has finished, crops it to the model's 79 deg, 4:3 view, resizes it to
640x480 and gives it to NavAgent as the front, look-down and level image alike (one fixed
camera, as InternNav runs on its Go2).  FORWARD and TURN_LEFT / TURN_RIGHT go to the base's
move_relative; LOOK_DOWN does nothing, since the camera cannot tilt; STOP ends the episode.
Control is released however the episode ends, Ctrl-C included.

Run on the dog's Orin inside `./platform shell`, with the venv of
docs/deploy/navigation_interface.md §5 active, from the repository root, and the ssh tunnel to
the 4090 up (§4):

    python scripts/deploy/run_go2.py --offline                 # no dog, no 4090: checks the install
    python scripts/deploy/run_go2.py --dry-run -i "..."        # real camera and model, the dog stays put
    python scripts/deploy/run_go2.py -i "Walk to the door and stop."   # the dog moves

Each run writes <out>/frames/step_NNNN.jpg (the camera's JPEG as received), <out>/steps.jsonl
(one line per step) and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.deploy import FORWARD_STEP_M, TURN_STEP_DEG, Action, CameraSpec, NavAgent  # noqa: E402

MODEL_HFOV_DEG = 79.0  # the simulated camera the model was trained and evaluated with
MODEL_SIZE = (640, 480)
# Unitree's figure for the Go2 front camera; the base serves it at 1920x1080 (16:9).
FRONT_CAMERA_HFOV_DEG = 100.0
# What the mount is, for the log: NavAgent warns that it differs from the simulated camera.
FRONT_CAMERA_MOUNT = CameraSpec(height_m=0.4, pitch_deg=0.0, lookdown_pitch_deg=0.0)


def crop_box(width: int, height: int, hfov_deg: float) -> tuple:
    """(left, top, right, bottom) of the centred 4:3 crop that sees MODEL_HFOV_DEG across."""
    if not MODEL_HFOV_DEG <= hfov_deg < 180.0:
        raise ValueError(f"camera hfov {hfov_deg} deg must be at least {MODEL_HFOV_DEG} deg")
    crop_w = width * math.tan(math.radians(MODEL_HFOV_DEG / 2)) / math.tan(math.radians(hfov_deg / 2))
    crop_h = crop_w * 3 / 4
    if crop_h > height:
        raise ValueError(f"a {width}x{height} image with {hfov_deg} deg across is too short for a 4:3 crop")
    left, top = round((width - crop_w) / 2), round((height - crop_h) / 2)
    return left, top, left + round(crop_w), top + round(crop_h)


def to_model_frame(jpeg: bytes, hfov_deg: float) -> np.ndarray:
    """The base's JPEG -> 640x480 RGB uint8 seeing 79 deg across, as the model expects."""
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    image = image.crop(crop_box(image.width, image.height, hfov_deg)).resize(MODEL_SIZE, Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def next_jpeg(robot: Any, camera: str, timeout_s: float) -> bytes:
    """The next new frame the base marks available; a warming-up or failing camera's frames are skipped.

    The base's client already refuses frames older than 1 s; its README asks projects to wait for
    the metadata's availability too, so warm-up frames never reach the model.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        frame = robot.get_frame(camera, timeout=timeout_s, wait_new=True)
        if frame.get("metadata", {}).get("available", True) is not False:
            return frame["jpeg"]
        if time.monotonic() > deadline:
            raise RuntimeError(f"{camera} stays unavailable: {frame.get('metadata')}")


def stop_reason(agent: NavAgent) -> str:
    """Why the episode ended (docs/deploy/navigation_interface.md §2.3)."""
    step = agent.last_step
    if step is None:
        return "unknown"
    if step.source == "terminal":
        return "model_stop" if agent.last_call is not None and agent.last_call.kind == "stop" else "fallback_stop"
    return {"empty": "empty_chunk", "step_cap": "step_cap"}.get(step.source, step.source)


class RunLog:
    """frames/step_NNNN.jpg, steps.jsonl and summary.json under one directory."""

    def __init__(self, out: Path) -> None:
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self._steps = open(out / "steps.jsonl", "a", encoding="utf-8")

    def frame(self, step: int, jpeg: bytes) -> None:
        (self.out / "frames" / f"step_{step:04d}.jpg").write_bytes(jpeg)

    def step(self, record: Dict[str, Any]) -> None:
        self._steps.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._steps.flush()

    def summary(self, record: Dict[str, Any]) -> None:
        (self.out / "summary.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def close(self) -> None:
        self._steps.close()


def run_episode(
    robot: Any,
    agent: NavAgent,
    *,
    instruction: str,
    run_id: int,
    scene_id: str = "go2lab",
    camera: str = "front_camera",
    hfov_deg: float = FRONT_CAMERA_HFOV_DEG,
    move: bool = True,
    frames_per_step: int = 2,
    frame_timeout_s: float = 10.0,
    max_look_downs: int = 3,
    log: Optional[RunLog] = None,
    say: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """One episode; returns the summary.  Raises on any failure, after releasing control."""
    acquired = False
    summary: Dict[str, Any] = {"instruction": instruction, "run_id": run_id, "scene_id": scene_id, "camera": camera,
                               "hfov_deg": hfov_deg, "move": move, "outcome": None, "steps": 0, "error": None}
    started = time.monotonic()
    try:
        if move:
            robot.wait_for_motion_ready(timeout=40, on_wait=say)   # the base's preflight; no control yet
            robot.acquire(owner="heatmapvln", mode="relative")      # the client heartbeats every 0.25 s from here
            acquired = True
        agent.reset(instruction, scene_id=scene_id, episode_id=run_id)
        look_downs = 0
        while not agent.done:
            t0 = time.monotonic()
            # The first new frame may have been taken before the motion ended; the next one hardly ever is.
            for _ in range(frames_per_step):
                jpeg = next_jpeg(robot, camera, frame_timeout_s)
            frame = to_model_frame(jpeg, hfov_deg)
            step = agent.steps
            if log is not None:
                log.frame(step, jpeg)
            t1 = time.monotonic()
            same = lambda: frame  # noqa: E731  one fixed camera: the look-down and level images are this frame
            action = agent.act(frame, same, same)
            t2 = time.monotonic()
            look_downs = look_downs + 1 if action is Action.LOOK_DOWN else 0
            motion = None
            if action in (Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT) and move:
                if action is Action.FORWARD:
                    task = robot.move_relative(distance=FORWARD_STEP_M)
                else:
                    sign = 1.0 if action is Action.TURN_LEFT else -1.0  # the base turns left for a positive angle
                    task = robot.move_relative(angle=sign * math.radians(TURN_STEP_DEG))
                motion = {"state": task.get("state"), "reason": task.get("reason")}
            t3 = time.monotonic()
            call = agent.last_step.call if agent.last_step is not None else None
            record = {
                "step": step,
                "action": action.name,
                "source": agent.last_step.source if agent.last_step is not None else None,
                "plan": None if call is None else {
                    "kind": call.kind, "llm_output": call.llm_output, "actions": call.actions,
                    "pixel_goal": call.pixel_goal, "pose_ready": call.pose_ready, "ppa_applied": call.ppa_applied,
                },
                "motion": motion,
                "seconds": {"frame": round(t1 - t0, 3), "model": round(t2 - t1, 3), "motion": round(t3 - t2, 3)},
            }
            if log is not None:
                log.step(record)
            plan = "" if call is None else f"  plan {call.kind} {call.llm_output!r}"
            done = "" if motion is None else f"  move {motion['state']} {t3 - t2:.1f}s"
            say(f"step {step:3d}  {action.name:<10}{plan}  model {t2 - t1:.1f}s{done}")
            if motion is not None and motion["state"] != "succeeded":
                raise RuntimeError(f"relative motion failed: {task}")
            if look_downs >= max_look_downs:
                # The camera cannot tilt, so LOOK_DOWN changes nothing; the model is stuck on this view.
                raise RuntimeError(f"LOOK_DOWN {look_downs} times in a row")
        summary["outcome"] = stop_reason(agent)
    except BaseException as error:  # Ctrl-C too: the episode is void, control still goes back
        summary["outcome"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["steps"] = agent.steps
        summary["plan_calls"] = getattr(agent, "calls", 0)
        summary["seconds"] = round(time.monotonic() - started, 1)
        if acquired:
            try:
                robot.release()  # the base zeroes its output
            except Exception as error:  # control already lost (lease expired, e-stop): release reports it
                say(f"release: {error}")
        if log is not None:
            log.summary(summary)
        say(f"end: {summary['outcome']} after {summary['steps']} steps, {summary['seconds']} s")
    return summary


class OfflineRobot:
    """Stands in for ROSPlatformClient with --offline: a 1920x1080 test picture, every motion succeeds."""

    def __init__(self) -> None:
        image = Image.new("RGB", (1920, 1080))
        pixels = np.asarray(image).copy()
        pixels[..., 0] = np.arange(1920)[None, :] % 256
        pixels[..., 1] = np.arange(1080)[:, None] % 256
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
        self.jpeg = buffer.getvalue()
        self.motions = []
        self.acquired = self.released = 0

    def wait_for_motion_ready(self, timeout: float, on_wait: Callable[[str], None]) -> None:
        pass

    def acquire(self, owner: str, mode: str) -> None:
        self.acquired += 1

    def release(self) -> None:
        self.released += 1

    def get_frame(self, camera: str, timeout: float, wait_new: bool) -> Dict[str, Any]:
        return {"jpeg": self.jpeg, "metadata": {"camera": camera}}

    def move_relative(self, distance: float = 0.0, angle: float = 0.0) -> Dict[str, Any]:
        self.motions.append((distance, angle))
        return {"task_id": len(self.motions), "state": "succeeded", "reason": ""}

    def __enter__(self) -> "OfflineRobot":
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one navigation episode on the Go2 with its front camera.")
    parser.add_argument("-i", "--instruction", help="English instruction (required unless --offline)")
    parser.add_argument("--run-id", type=int, default=None, help="a new non-negative integer per run (default: the time)")
    parser.add_argument("--dry-run", action="store_true", help="take frames and plan, but print actions instead of moving")
    parser.add_argument("--offline", action="store_true", help="no dog and no 4090: a test picture and fake servers")
    parser.add_argument("--fake-servers", action="store_true", help="real dog or ./platform --backend sim, fake model servers")
    parser.add_argument("--model", default="127.0.0.1:52400", help="model server through the ssh tunnel")
    parser.add_argument("--vo", default="127.0.0.1:52500", help="odometry server through the ssh tunnel")
    parser.add_argument("--camera", default="front_camera", help="the base's camera name")
    parser.add_argument("--hfov", type=float, default=FRONT_CAMERA_HFOV_DEG,
                        help="horizontal field of view of that camera in degrees, for the crop to 79 deg")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None, help="log directory (default runs/go2/<time>_<run id>)")
    args = parser.parse_args(argv)

    if not args.instruction:
        if not args.offline:
            parser.error("--instruction is required")
        args.instruction = "Walk forward to the door and stop."
    run_id = args.run_id if args.run_id is not None else int(time.time())
    out = args.out or REPO / "runs" / "go2" / f"{time.strftime('%Y%m%d_%H%M%S')}_{run_id}"
    log = RunLog(out)
    print(f"logging to {out}")

    if args.offline or args.fake_servers:
        from src.deploy.fake_servers import FakeModelServer, FakeVOServer, pil_jpeg_encoder

        try:
            import vla_rpc  # noqa: F401
            encoder = None
        except ImportError:
            encoder = pil_jpeg_encoder  # only with fake servers: its bytes differ from the deployed encoder's
        agent = NavAgent(FakeModelServer(), FakeVOServer(), camera=FRONT_CAMERA_MOUNT, max_steps=args.max_steps,
                         jpeg_encoder=encoder)
    else:
        agent = NavAgent(args.model, args.vo, camera=FRONT_CAMERA_MOUNT, max_steps=args.max_steps, on_log=print)

    if args.offline:
        robot_context = OfflineRobot()
    else:
        from robot_platform.ros_client import ROSPlatformClient  # only inside ./platform shell

        robot_context = ROSPlatformClient(timeout=3.0)
    try:
        with robot_context as robot, agent:
            summary = run_episode(robot, agent, instruction=args.instruction, run_id=run_id, camera=args.camera,
                                  hfov_deg=args.hfov, move=not args.dry_run, log=log)
    finally:
        log.close()
    if args.offline:
        print("offline check OK" if summary["outcome"] == "model_stop" else "offline check FAILED")
        return 0 if summary["outcome"] == "model_stop" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
