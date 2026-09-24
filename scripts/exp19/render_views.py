"""EXP-19 [E]: re-render closed-loop states in the training collector's geometry (habitat-sim 0.1.7, ``envs/vlnce``).

The deployed camera is 640x480 HFOV 79, but the history head predicts on the
256x256 HFOV-90 label grid of ``r2r_panoramic_data_v2``.  Ground truth
(front depth for the occlusion test) and the 360-degree display backdrop must
therefore come from the geometry of the collector that wrote that data set
(and the EXP-18 renders): ``<VLN-CE>/collect/panoramic/collector.py`` with
``habitat_extensions/config/vlnce_collect.yaml``.  This tool builds the very
same habitat-lab ``Sim-v0`` from that config (plus the collector's two
overrides) and renders exactly as ``collect/common/multiview.py``
``capture_multiview`` does:

  * one RGB + one DEPTH sensor, 256x256, HFOV 90, at (0, 1.25, 0) on the agent
    body, no sensor rotation;
  * the four views are produced by ROTATING THE AGENT: body rotation
    ``q * q_yaw`` with yaw F 0, R -pi/2, B pi, L +pi/2 about +y
    (R = turned right), then ``sim.set_agent_state(position, rotation)``
    (sensors follow the body) and ``sim.get_sensor_observations()``;
  * depth is the raw habitat-sim planar (z) depth in metres, float16, NOT
    clipped to MAX_DEPTH (the collector bypasses habitat-lab's depth sensor
    post-processing); 0 = no geometry;
  * camera-to-world (Habitat: y up, camera looks along -z) =
    ``[R(q) | position + R(q) @ (0, 1.25, 0)]`` (collector ``compute_camera_pose``).

Inputs (one of):

  --run-dir runs/<run>        every ``gpu*/steps/<ep_key>/steps.jsonl`` [C] with
                              its call traces ``gpu*/trace/<ep_key>/call_*.json`` [B];
                              renders at every call's ``current_capture_step``
                              (calls without a trace: first ``step_before`` of their
                              [C] action records) plus the final state (the state
                              [C] flags ``final``, else the last ``state`` record)
  --steps F --trace-dir D     the same for one episode
  --poses F --scene S --ep-key K
                              explicit JSON list of {step, position, rotation_wxyz}
  --self-check                render a stored r2r_panoramic_data_v2 clip at its own
                              poses and compare (depth, RGB, pose convention)

Output per episode (``--out-dir``, default ``$EXP19_ROOT/renders``)::

    <ep_key>.npz   steps int32[N], rgb uint8[N,4,256,256,3] (RGB, views F,R,B,L),
                   depth_front float16[N,256,256] (metres), c2w_front float64[N,4,4],
                   position float64[N,3], rotation_wxyz float64[N,4] (the input body state)
    <ep_key>.json  schema exp19-renders-v1: which call(s) each step serves, sensor
                   spec, collector provenance (sha256), habitat versions, QA

QA per rendered step (any failure fails the episode, which then writes nothing):

  * habitat's own sensor pose after ``set_agent_state`` vs our ``c2w_front``
    (<= 1e-4);
  * trace mode: the deployed RGB sensor pose [C] recorded at that state
    (``rgb_sensor_position`` / ``rgb_sensor_rotation_wxyz``) vs ``c2w_front``
    (<= 1e-4): c2w_front is the deployed front camera's own pose;
  * trace mode: normalised cross-correlation of our front render, resampled
    into the deployed 640x480 HFOV-79 camera, with the client's own frame
    ``front_<step>.jpg``.  On real client frames (17 states of 3 episodes)
    correct poses gave 0.976-0.999; the same states with the body yaw off by
    15 / 30 / 90 deg gave medians 0.79 / 0.62 / -0.03.  Steps below 0.9 are
    listed as warnings; a MEDIAN below 0.9 is a systematic pose-convention
    (or state/frame alignment) error and fails;
  * body rotations must be yaw-only (the eval agent never tilts its body; a
    quaternion read in the wrong component order is not yaw-only);
  * trace mode: at least one of the two pose-convention checks above (sensor
    pose, front NCC) must run on some step; if neither can (no ``rgb_sensor_*``
    fields, no readable client frame) a WARNING is printed and QA fails.

An existing ``<ep_key>.json`` is re-rendered when its inputs, its steps or the
sha256 of this file differ; a ``<ep_key>.npz`` without its json is removed
first.  An episode whose job cannot be built fails alone (nonzero exit).

Needs an X display with llvmpipe GLX: run through ``run_render.sh`` (which uses
``scripts/exp18/topdown/with_xvfb.sh``), e.g.::

    DISPLAY_NUM=390 bash scripts/exp18/topdown/with_xvfb.sh \\
        scripts/exp19/render_views.py --self-check --out-dir /tmp/exp19_render_check
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp18 import common  # noqa: E402

SCHEMA = "exp19-renders-v1"
SELF_CHECK_SCHEMA = "exp19-render-self-check-v1"
EXP_ROOT = Path(os.environ.get("EXP19_ROOT", str(common.WORKSPACE / "model" / "exp19_behavior_viz")))

# The collector that wrote r2r_panoramic_data_v2 and the EXP-18 renders.
COLLECTOR_CONFIG = common.VLNCE_PROJECT / "habitat_extensions" / "config" / "vlnce_collect.yaml"
COLLECTOR_SOURCES = (
    common.VLNCE_PROJECT / "collect" / "panoramic" / "collector.py",
    common.VLNCE_PROJECT / "collect" / "common" / "multiview.py",
    common.VLNCE_PROJECT / "collect" / "common" / "geometry.py",
    COLLECTOR_CONFIG,
)
DEFAULT_SELF_CHECK_CLIP = common.R2R_V2_ROOT / "1LXtFkjw3qL" / "clip_000452"

VIEWS = ("front", "right", "back", "left")
# collect/common/multiview.py DIRECTION_YAW_OFFSETS, same expressions (radians, about +y).
VIEW_YAW_RAD = (0.0, -np.pi / 2, np.pi, np.pi / 2)
VIEW_YAW_DEG = (0.0, -90.0, 180.0, 90.0)
SENSOR_HEIGHT_M = 1.25
IMAGE_SIZE = 256
HFOV_DEG = 90.0
K_LABEL = [[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]]

# Deployed front camera (scripts/evaluation/r2r_val_unseen.py): only for the QA
# comparison with the client's own front frames, never for anything rendered.
EVAL_HW = (480, 640)
EVAL_HFOV_DEG = 79.0
EVAL_QA_HW = (240, 320)  # compared at half resolution (our 256 px render is coarser)
EVAL_NCC_MIN = 0.9  # per step: warning; episode median below it: failure

POSE_TOL = 1e-4  # m / rotation-matrix entries

CONVENTIONS = (
    "Habitat world, y up. c2w = camera-to-world, camera looks along -z, x right, y up. "
    "c2w_front = [R(q) | p + R(q) @ (0, 1.25, 0)], p = agent body position (floor), q = body rotation (wxyz). "
    "View v camera = body rotation q * q_yaw(v), yaw F 0, R -90, B 180, L +90 deg about +y (R = turned right); "
    "c2w_view = c2w_front @ R_y(yaw), R_y = [[c,0,s],[0,1,0],[-s,0,c]]. "
    "Pinhole K = [[128,0,128],[0,128,128],[0,0,1]] (pixel index p has centre p + 0.5). "
    "depth_front: raw habitat-sim planar z-depth in metres, float16, 0 = no geometry, not clipped."
)


# ---------------------------------------------------------------- geometry ---

def quat_wxyz_to_matrix(q) -> np.ndarray:
    """3x3 rotation of a (w, x, y, z) quaternion; collector ``quaternion_to_rotation_matrix`` in float64."""
    w, x, y, z = [float(v) for v in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat_wxyz(R) -> np.ndarray:
    """Unit (w, x, y, z) quaternion of a rotation matrix, w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quat_mul_wxyz(a, b) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions (numpy-quaternion's ``*``)."""
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Rotation by ``yaw_rad`` about +y (multiview.py ``q_yaw``)."""
    return np.array([np.cos(yaw_rad / 2), 0.0, np.sin(yaw_rad / 2), 0.0], dtype=np.float64)


def view_rotation_wxyz(rotation_wxyz, view: int) -> np.ndarray:
    """Body rotation that renders view ``view`` (multiview.py: ``orig_rot`` for front, else ``orig_rot * q_yaw``)."""
    if abs(VIEW_YAW_RAD[view]) < 1e-6:
        return np.asarray(rotation_wxyz, dtype=np.float64)
    return quat_mul_wxyz(rotation_wxyz, yaw_quat_wxyz(VIEW_YAW_RAD[view]))


def camera_c2w(position, rotation_wxyz) -> np.ndarray:
    """Collector ``compute_camera_pose``: T_world_agent @ T_agent_cam, sensor at (0, 1.25, 0), no sensor rotation."""
    R = quat_wxyz_to_matrix(rotation_wxyz)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = np.asarray(position, dtype=np.float64) + R @ np.array([0.0, SENSOR_HEIGHT_M, 0.0])
    return c2w


def view_c2w(position, rotation_wxyz, view: int) -> np.ndarray:
    return camera_c2w(position, view_rotation_wxyz(rotation_wxyz, view))


def yaw_only_error(rotation_wxyz) -> float:
    """max |R[:, 1] - (0, 1, 0)|: 0 for an upright (yaw-only) body rotation."""
    return float(np.abs(quat_wxyz_to_matrix(rotation_wxyz)[:, 1] - np.array([0.0, 1.0, 0.0])).max())


# ------------------------------------------------------------------- inputs ---

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Recorded in every <ep_key>.json: a render made by another version of this file is stale.
TOOL_SHA256 = sha256_file(Path(__file__).resolve())


def scene_glb(scene: str) -> Path:
    if scene.endswith(".glb"):
        path = Path(scene)
    else:
        scene = scene.split("/")[-1]
        path = common.MP3D_SCENES / scene / f"{scene}.glb"
    if not path.is_file():
        raise FileNotFoundError(f"scene mesh not found: {path}")
    return path


def read_steps_jsonl(path: Path) -> dict:
    """[C] step trace -> {"start", "end" (or None), "states": {step: record}, "actions": [records]}."""
    start, end, states, actions = None, None, {}, []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "episode_start":
                if start is not None:
                    raise ValueError(f"{path}:{line_no}: second episode_start")
                start = record
            elif kind == "state":
                step = int(record["step"])
                if step in states:
                    raise ValueError(f"{path}:{line_no}: duplicate state record for step {step}")
                states[step] = record
            elif kind == "action":
                actions.append(record)
            elif kind == "episode_end":
                end = record
    if start is None:
        raise ValueError(f"{path}: no episode_start record")
    if not states:
        raise ValueError(f"{path}: no state records")
    return {"start": start, "end": end, "states": states, "actions": actions}


def final_state_step(states: dict) -> int:
    """The state [C] flags ``final`` (after the last action); the last state record if none is flagged."""
    flagged = [step for step, record in states.items() if record.get("final")]
    if len(flagged) > 1 or (flagged and flagged[0] != max(states)):
        raise ValueError(f"inconsistent final state records: flagged {flagged}, last step {max(states)}")
    return flagged[0] if flagged else max(states)


def call_steps_from_actions(actions: list) -> dict:
    """{system2_call_index: step of its first executed action} from [C] action records.

    A call's response is acted on at the step it was made (``rpc_first`` / ``terminal`` /
    ``rpc_empty_actions`` have ``step_before`` = the call's step), so this recovers the
    capture step of every call that executed an action, even if its [B] trace is missing.
    """
    steps = {}
    for record in actions:
        index = record.get("system2_call_index")
        if index is not None:
            step = int(record["step_before"])
            steps[int(index)] = min(step, steps.get(int(index), step))
    return steps


def read_call_steps(trace_dir: Path) -> tuple:
    """[B] call traces -> ([(system2_call_index, current_capture_step)], [call files without a capture step])."""
    calls, missing = [], []
    for path in sorted(trace_dir.glob("call_*.json")):
        record = json.loads(path.read_text())
        index = int(record["system2_call_index"])
        step = record.get("current_capture_step")
        if step is None:
            missing.append(path.name)
        else:
            calls.append((index, int(step)))
    calls.sort()
    return calls, missing


def trace_job(steps_path: Path, trace_dir: Path) -> dict:
    trace = read_steps_jsonl(steps_path)
    start = trace["start"]
    scene_id = str(start["scene_id"]).split("/")[-1].replace(".glb", "")
    episode_id = int(start["episode_id"])
    ep_key = start.get("ep_key") or f"{scene_id}_{episode_id:04d}"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"{ep_key}: call trace dir missing: {trace_dir}")
    calls, calls_without_step = read_call_steps(trace_dir)
    if not calls:
        raise ValueError(f"{ep_key}: no call traces with current_capture_step in {trace_dir}")
    traced = dict(calls)
    from_actions = call_steps_from_actions(trace["actions"])
    step_mismatch = [[i, traced[i], s] for i, s in sorted(from_actions.items()) if i in traced and traced[i] != s]
    untraced = sorted(set(from_actions) - set(traced))
    if step_mismatch:
        print(f"[render] WARNING {ep_key}: call step in [B] trace != first [C] action step "
              f"(index, trace, actions): {step_mismatch}", flush=True)
    if untraced:
        print(f"[render] WARNING {ep_key}: calls with actions but no [B] trace (step taken from the "
              f"actions): {untraced}", flush=True)
    final_step = final_state_step(trace["states"])
    by_step = {}
    for index, step in sorted({**from_actions, **traced}.items()):  # the trace wins where both exist
        by_step.setdefault(step, []).append(index)
    by_step.setdefault(final_step, [])
    missing = sorted(s for s in by_step if s not in trace["states"])
    if missing:
        raise ValueError(f"{ep_key}: call steps without a state record in {steps_path}: {missing}")
    render_steps = []
    for step in sorted(by_step):
        state = trace["states"][step]
        frame = state.get("front_jpg")
        render_steps.append({
            "step": step,
            "call_indices": by_step[step],
            "final": step == final_step,
            "position": [float(v) for v in state["position"]],
            "rotation_wxyz": [float(v) for v in state["rotation_wxyz"]],
            "eval_front_jpg": str(steps_path.parent / frame) if frame else None,
            "eval_sensor_position": state.get("rgb_sensor_position"),
            "eval_sensor_rotation_wxyz": state.get("rgb_sensor_rotation_wxyz"),
        })
    return {
        "mode": "trace",
        "ep_key": ep_key,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "render_steps": render_steps,
        "inputs": {
            "steps_jsonl": str(steps_path),
            "steps_jsonl_sha256": sha256_file(steps_path),
            "trace_dir": str(trace_dir),
            "num_calls": len(calls),
            "calls_without_capture_step": calls_without_step,
            "call_step_mismatch_trace_vs_actions": step_mismatch,
            "calls_from_actions_without_trace": untraced,
            "num_state_records": len(trace["states"]),
            "final_state_step": final_step,
            "episode_end": trace["end"],
        },
    }


def stale_reason(record_path: Path, job: dict) -> str | None:
    """Why an existing ``<ep_key>.json`` does not belong to ``job`` (None = it does, skip it).

    ``renders/`` is shared by every run under ``EXP19_ROOT`` (smoke, main, ...),
    so an existing render counts only if it came from the same input file and
    covers the same steps.
    """
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as exc:
        return f"unreadable: {exc}"
    if record.get("schema") != SCHEMA or record.get("mode") != job["mode"]:
        return f"schema/mode {record.get('schema')}/{record.get('mode')}"
    if record.get("tool_sha256") != TOOL_SHA256:
        return "rendered by another version of render_views.py (tool_sha256 differs)"
    key = "steps_jsonl_sha256" if job["mode"] == "trace" else "poses_json_sha256"
    if (record.get("inputs") or {}).get(key) != job["inputs"][key]:
        return f"{key} differs"
    if [row["step"] for row in record.get("steps", [])] != [row["step"] for row in job["render_steps"]]:
        return "rendered steps differ"
    if not (record_path.parent / record.get("npz", "")).is_file():
        return "npz missing"
    return None


def run_jobs(run_dir: Path, episodes=None) -> tuple:
    """Every episode of a rerun: ``gpu*/steps/<ep_key>/steps.jsonl`` + ``gpu*/trace/<ep_key>/``.

    Returns ``(jobs, failed ep_keys)``: an episode whose job cannot be built fails alone.
    """
    found = {}
    for steps_path in sorted(run_dir.glob("gpu*/steps/*/steps.jsonl")):
        ep_key = steps_path.parent.name
        if ep_key in found:
            raise ValueError(f"episode {ep_key} appears twice in {run_dir}: {found[ep_key]} and {steps_path}")
        found[ep_key] = steps_path
    if episodes:
        unknown = sorted(set(episodes) - set(found))
        if unknown:
            raise ValueError(f"episodes not in {run_dir}: {unknown}")
        found = {k: found[k] for k in episodes}
    if not found:
        raise ValueError(f"no gpu*/steps/*/steps.jsonl under {run_dir}")
    jobs, failed = [], []
    for ep_key, steps_path in sorted(found.items()):
        gpu_dir = steps_path.parents[2]
        try:
            jobs.append(trace_job(steps_path, gpu_dir / "trace" / ep_key))
        except Exception:
            traceback.print_exc()
            failed.append(ep_key)
            print(f"[render] {ep_key}: FAILED to build its render job", flush=True)
    return jobs, failed


def poses_job(poses_path: Path, scene: str, ep_key: str) -> dict:
    poses = json.loads(poses_path.read_text())
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{poses_path}: expected a non-empty JSON list of {{step, position, rotation_wxyz}}")
    render_steps, seen = [], set()
    for pose in poses:
        step = int(pose["step"])
        if step in seen:
            raise ValueError(f"{poses_path}: duplicate step {step}")
        seen.add(step)
        render_steps.append({
            "step": step, "call_indices": [], "final": False,
            "position": [float(v) for v in pose["position"]],
            "rotation_wxyz": [float(v) for v in pose["rotation_wxyz"]],
            "eval_front_jpg": None, "eval_sensor_position": None, "eval_sensor_rotation_wxyz": None,
        })
    render_steps.sort(key=lambda row: row["step"])
    scene_id = Path(scene).stem if scene.endswith(".glb") else scene.split("/")[-1]
    episode_id = None
    if ep_key.startswith(scene_id + "_") and ep_key[len(scene_id) + 1:].isdigit():
        episode_id = int(ep_key[len(scene_id) + 1:])
    return {
        "mode": "poses",
        "ep_key": ep_key,
        "scene_id": scene_id,
        "scene_arg": scene,
        "episode_id": episode_id,
        "render_steps": render_steps,
        "inputs": {"poses_json": str(poses_path), "poses_json_sha256": sha256_file(poses_path)},
    }


# ----------------------------------------------------------------- renderer ---

class CollectorRenderer:
    """habitat-lab's ``Sim-v0`` built from the collector's own config; one per process, reconfigured per scene."""

    def __init__(self, gpu: int = 0):
        self.gpu = gpu
        self.sim = None
        self.scene = None
        self.config = None

    def _config(self, glb: Path):
        vlnce = str(common.VLNCE_PROJECT)
        if vlnce not in sys.path:
            sys.path.append(vlnce)  # appended: VLN-CE has its own ``scripts`` directory
        from habitat_extensions.config.default import get_extended_config
        config = get_extended_config(str(COLLECTOR_CONFIG))
        config.defrost()
        # collect/panoramic/collector.py main()
        config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.gpu
        config.SIMULATOR.SCENE = str(glb)
        config.freeze()
        return config

    def load(self, glb: Path) -> float:
        if self.scene == str(glb):
            return 0.0
        import habitat
        t0 = time.time()
        config = self._config(glb)
        if self.sim is None:
            self.sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)
        else:
            self.sim.reconfigure(config.SIMULATOR)
        self.scene, self.config = str(glb), config
        return time.time() - t0

    def sensor_spec(self) -> dict:
        sim_cfg = self.config.SIMULATOR
        spec = {}
        for name in ("RGB_SENSOR", "DEPTH_SENSOR"):
            cfg = sim_cfg[name]
            spec[name] = {key: (list(value) if isinstance(value, (list, tuple)) else value)
                          for key, value in cfg.items()}
        spec["AGENT_0"] = {"HEIGHT": sim_cfg.AGENT_0.HEIGHT, "RADIUS": sim_cfg.AGENT_0.RADIUS,
                           "SENSORS": list(sim_cfg.AGENT_0.SENSORS)}
        spec["HABITAT_SIM_V0"] = {key: value for key, value in sim_cfg.HABITAT_SIM_V0.items()}
        return spec

    def render(self, position, rotation_wxyz) -> dict:
        """F, R, B, L exactly like ``capture_multiview``; restores nothing (every call sets the full state)."""
        import quaternion  # noqa: F401  (registers np.quaternion)
        position = np.asarray(position, dtype=np.float32)
        body = np.quaternion(*[float(v) for v in rotation_wxyz])
        rgb = np.zeros((len(VIEWS), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        depth_front = None
        c2w = np.zeros((len(VIEWS), 4, 4), dtype=np.float64)
        pose_err = 0.0
        for v in range(len(VIEWS)):
            yaw = VIEW_YAW_RAD[v]
            rot = body if abs(yaw) < 1e-6 else body * np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            self.sim.set_agent_state(position, rot)
            obs = self.sim.get_sensor_observations()
            rgb[v] = obs["rgb"][..., :3]
            if v == 0:
                depth_front = np.asarray(obs["depth"], dtype=np.float32).copy()
            c2w[v] = view_c2w(position, rotation_wxyz, v)
            # The camera habitat actually rendered from must be the c2w we report.
            sensor = self.sim.get_agent(0).get_state().sensor_states["rgb"]
            sensor_R = quat_wxyz_to_matrix([sensor.rotation.w, sensor.rotation.x,
                                            sensor.rotation.y, sensor.rotation.z])
            pose_err = max(pose_err,
                           float(np.abs(np.asarray(sensor.position, dtype=np.float64) - c2w[v, :3, 3]).max()),
                           float(np.abs(sensor_R - c2w[v, :3, :3]).max()))
        if depth_front.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"unexpected depth shape {depth_front.shape}")
        return {"rgb": rgb, "depth_front": depth_front, "c2w": c2w, "sensor_pose_err": pose_err}

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.scene = None


def versions() -> dict:
    import habitat
    import habitat_sim
    import cv2
    return {
        "habitat_sim": getattr(habitat_sim, "__version__", "unknown"),
        "habitat": getattr(habitat, "__version__", "unknown"),
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "python": sys.version.split()[0],
    }


def collector_provenance() -> dict:
    return {
        "collector": "habitat/VLN-CE collect/panoramic/collector.py (capture_multiview, compute_camera_pose)",
        "config": str(COLLECTOR_CONFIG),
        "config_overrides": {"SIMULATOR.AGENT_0.SENSORS": ["RGB_SENSOR", "DEPTH_SENSOR"],
                             "SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID": "--gpu",
                             "SIMULATOR.SCENE": "scene mesh"},
        "sources_sha256": {str(p): (sha256_file(p) if p.is_file() else None) for p in COLLECTOR_SOURCES},
    }


def git_sha() -> str | None:
    marker = REPO_ROOT / ".exp19_git_sha"
    return marker.read_text().strip() if marker.is_file() else None


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, record: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(record), indent=1))
    os.replace(str(tmp), str(path))


# ----------------------------------------------------------------------- QA ---

def eval_sensor_error(row: dict, c2w_front: np.ndarray) -> float | None:
    """max |deployed RGB sensor pose ([C] ``rgb_sensor_*``) - our c2w_front| (position m, matrix entries)."""
    if row.get("eval_sensor_position") is None or row.get("eval_sensor_rotation_wxyz") is None:
        return None
    return max(float(np.abs(np.asarray(row["eval_sensor_position"], dtype=np.float64) - c2w_front[:3, 3]).max()),
               float(np.abs(quat_wxyz_to_matrix(row["eval_sensor_rotation_wxyz"]) - c2w_front[:3, :3]).max()))


def eval_front_ncc(front_rgb: np.ndarray, eval_jpg: str) -> float | None:
    """NCC (grey) of our HFOV-90 front render resampled into the deployed HFOV-79 camera vs the client's frame.

    Same centre and orientation, only the intrinsics differ, so this is a pure
    remap; the HFOV-79 frustum (+-39.5 x +-31.7 deg) lies inside the HFOV-90 image.
    """
    import cv2
    frame = cv2.imread(eval_jpg, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    if frame.shape != EVAL_HW:
        raise ValueError(f"{eval_jpg}: expected {EVAL_HW}, got {frame.shape}")
    h, w = EVAL_QA_HW
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    f_eval = (w / 2.0) / math.tan(math.radians(EVAL_HFOV_DEG) / 2.0)
    f_ours = (IMAGE_SIZE / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    map_x = ((u - w / 2.0) / f_eval * f_ours + IMAGE_SIZE / 2.0 - 0.5).astype(np.float32)
    map_y = ((v - h / 2.0) / f_eval * f_ours + IMAGE_SIZE / 2.0 - 0.5).astype(np.float32)
    grey = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2GRAY)
    ours = cv2.remap(grey, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    a, b = ours - ours.mean(), frame - frame.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else None


# ------------------------------------------------------------------ episode ---

def render_episode(renderer: CollectorRenderer, job: dict, out_dir: Path) -> dict:
    t0 = time.time()
    glb = scene_glb(job.get("scene_arg") or job["scene_id"])
    load_s = renderer.load(glb)
    rows = job["render_steps"]
    n = len(rows)
    steps = np.array([row["step"] for row in rows], dtype=np.int32)
    position = np.array([row["position"] for row in rows], dtype=np.float64)
    rotation = np.array([row["rotation_wxyz"] for row in rows], dtype=np.float64)
    for row, q in zip(rows, rotation):
        if abs(np.linalg.norm(q) - 1.0) > 1e-4:
            raise ValueError(f"{job['ep_key']} step {row['step']}: rotation_wxyz not unit: {q.tolist()}")
        err = yaw_only_error(q)
        if err > POSE_TOL:
            raise ValueError(f"{job['ep_key']} step {row['step']}: body rotation is not yaw-only "
                             f"(max |R[:,1]-e_y| = {err:.3g}); quaternion component order must be w,x,y,z")
    rgb = np.zeros((n, len(VIEWS), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    depth = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float16)
    c2w = np.zeros((n, 4, 4), dtype=np.float64)
    sensor_err, eval_err, ncc = [], [], []
    t1 = time.time()
    for i, row in enumerate(rows):
        out = renderer.render(position[i], rotation[i])
        rgb[i] = out["rgb"]
        depth[i] = out["depth_front"].astype(np.float16)
        c2w[i] = out["c2w"][0]
        sensor_err.append(out["sensor_pose_err"])
        ncc.append(eval_front_ncc(out["rgb"][0], row["eval_front_jpg"]) if row["eval_front_jpg"] else None)
        eval_err.append(eval_sensor_error(row, c2w[i]))
    render_s = time.time() - t1
    if max(sensor_err) > POSE_TOL:
        raise RuntimeError(f"{job['ep_key']}: habitat sensor pose differs from c2w by {max(sensor_err):.3g}")
    bad = [int(steps[i]) for i, e in enumerate(eval_err) if e is not None and e > POSE_TOL]
    if bad:
        # c2w_front must be the deployed front camera itself (same centre, same orientation, level).
        raise RuntimeError(f"{job['ep_key']}: deployed RGB sensor pose differs from c2w_front by "
                           f"{max(e for e in eval_err if e is not None):.3g} at steps {bad}")
    valid_ncc = [v for v in ncc if v is not None]
    checked_sensor = [e for e in eval_err if e is not None]
    if job["mode"] == "trace" and not valid_ncc and not checked_sensor:
        # Nothing ties c2w_front to the deployed camera: an unverified render must not reach [F].
        print(f"[render] WARNING {job['ep_key']}: no pose-convention check could run (no front NCC, "
              "no rgb_sensor_* fields in the step trace)", flush=True)
        raise RuntimeError(f"{job['ep_key']}: QA failed: no pose-convention check could run")
    low_ncc = [int(steps[i]) for i, v in enumerate(ncc) if v is not None and v < EVAL_NCC_MIN]
    if valid_ncc and float(np.median(valid_ncc)) < EVAL_NCC_MIN:
        # Systematic, not a single odd view: the state trace's pose convention does not match the client camera.
        raise RuntimeError(f"{job['ep_key']}: median NCC of front render vs client frames "
                           f"{float(np.median(valid_ncc)):.3f} < {EVAL_NCC_MIN}; wrong pose convention?")

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{job['ep_key']}.npz"
    tmp = out_dir / f"{job['ep_key']}.tmp.npz"
    np.savez_compressed(str(tmp), steps=steps, rgb=rgb, depth_front=depth, c2w_front=c2w,
                        position=position, rotation_wxyz=rotation)
    os.replace(str(tmp), str(npz_path))

    record = {
        "schema": SCHEMA,
        "ep_key": job["ep_key"],
        "scene_id": job["scene_id"],
        "episode_id": job["episode_id"],
        "scene_glb": str(glb),
        "mode": job["mode"],
        "inputs": job["inputs"],
        "npz": npz_path.name,
        "arrays": {
            "steps": "int32[N] rendered step ids (ascending)",
            "rgb": "uint8[N,4,256,256,3] RGB (not BGR), views F,R,B,L, raw render (no JPEG)",
            "depth_front": "float16[N,256,256] front planar depth, metres, 0 = no geometry",
            "c2w_front": "float64[N,4,4] front camera-to-world, Habitat (y up, camera looks along -z)",
            "position": "float64[N,3] input agent body position (floor)",
            "rotation_wxyz": "float64[N,4] input agent body rotation (w,x,y,z)",
        },
        "steps": [{"index": i, "step": row["step"], "call_indices": row["call_indices"], "final": row["final"],
                   "eval_front_jpg": row["eval_front_jpg"], "eval_front_ncc": ncc[i],
                   "sensor_pose_err": sensor_err[i], "eval_sensor_pose_err": eval_err[i]}
                  for i, row in enumerate(rows)],
        "sensor": {
            "views": list(VIEWS), "view_yaw_deg": list(VIEW_YAW_DEG), "resolution_hw": [IMAGE_SIZE, IMAGE_SIZE],
            "hfov_deg": HFOV_DEG, "sensor_height_m": SENSOR_HEIGHT_M, "K": K_LABEL,
            "four_views_by": "agent rotation (sim.set_agent_state), one RGB + one DEPTH sensor",
            "habitat_lab_config": renderer.sensor_spec(),
        },
        "conventions": CONVENTIONS,
        "collector": collector_provenance(),
        "versions": versions(),
        "qa": {
            "max_sensor_pose_err": float(max(sensor_err)),
            "max_eval_sensor_pose_err": max(checked_sensor, default=None),
            "eval_sensor_pose_checked_steps": len(checked_sensor),
            "eval_front_ncc": {"n": len(valid_ncc),
                               "min": float(min(valid_ncc)) if valid_ncc else None,
                               "median": float(np.median(valid_ncc)) if valid_ncc else None,
                               "warn_below": EVAL_NCC_MIN, "fail_if_median_below": EVAL_NCC_MIN, "steps_below": low_ncc,
                               "camera": {"hw": list(EVAL_HW), "hfov_deg": EVAL_HFOV_DEG,
                                          "compared_hw": list(EVAL_QA_HW)}},
        },
        "timing_s": {"scene_load": round(load_s, 3), "render": round(render_s, 3),
                     "per_step": round(render_s / max(n, 1), 4), "total": round(time.time() - t0, 3)},
        "git_sha": git_sha(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "tool": "scripts/exp19/render_views.py",
        "tool_sha256": TOOL_SHA256,
    }
    write_json(out_dir / f"{job['ep_key']}.json", record)
    if low_ncc:
        print(f"[render] WARNING {job['ep_key']}: front render vs client frame NCC < {EVAL_NCC_MIN} "
              f"at steps {low_ncc}", flush=True)
    return record


# --------------------------------------------------------------- self-check ---

def load_clip(clip_dir: Path) -> dict:
    """A collector clip: frame-ordered poses of all views, front depth and JPEG bytes."""
    meta = json.loads((clip_dir / "meta.json").read_text())
    trajectory = np.load(str(clip_dir / "trajectory_3d.npy"))
    parts = {key: [] for key in ["frame_ids", "depth_front"] + [f"pose_{v}" for v in VIEWS]
             + [f"rgb_{v}" for v in VIEWS]}
    for path in sorted(glob.glob(str(clip_dir / "chunks" / "chunk_*.npz"))):
        with np.load(path, allow_pickle=True) as chunk:  # our own data: JPEG bytes in object arrays
            for key in parts:
                parts[key].append(chunk[key])
    clip = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(clip["frame_ids"])
    clip = {key: value[order] for key, value in clip.items()}
    if not np.array_equal(clip["frame_ids"], np.arange(len(trajectory))):
        raise ValueError(f"{clip_dir}: chunk frame ids do not cover trajectory_3d frames")
    clip["trajectory_3d"] = trajectory
    clip["meta"] = meta
    return clip


def decode_bgr_jpeg(buf) -> np.ndarray:
    """Collector JPEGs are cv2-encoded BGR: decode with cv2 and return RGB."""
    import cv2
    return cv2.cvtColor(cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def encode_like_collector(rgb: np.ndarray, quality: int = 90) -> bytes:
    """collect/common/io_utils.py save_chunk_npz: cv2.imencode of the BGR image, JPEG quality 90."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError("JPEG encode failed")
    return buf.tobytes()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def float32_quaternion_candidates(R32: np.ndarray) -> list:
    """float32 (w,x,y,z) quaternions within 1 ulp per component of the recovered one whose collector
    matrix, cast to float32, equals the stored ``R32`` exactly.

    The collector stored ``float32(R(q))`` of the agent's float32 quaternion, which does not pin
    ``q`` down to the last bit (usually 1-3 candidates); habitat renders from ``q`` itself.
    """
    base = matrix_to_quat_wxyz(R32).astype(np.float32)
    steps = (np.float32(-np.inf), None, np.float32(np.inf))
    found = []
    for moves in np.ndindex(3, 3, 3, 3):
        q = np.array([base[i] if steps[m] is None else np.nextafter(base[i], steps[m])
                      for i, m in enumerate(moves)], dtype=np.float32)
        if np.array_equal(quat_wxyz_to_matrix(q.astype(np.float64)).astype(np.float32), R32):
            found.append(q.astype(np.float64))
    return found


def compare_frame(clip: dict, t: int, out: dict) -> dict:
    """Our render of frame ``t`` vs the stored one: front depth (float16, valid = stored > 0) and 4 JPEGs."""
    stored_depth = clip["depth_front"][t].astype(np.float32)
    ours_depth = out["depth_front"].astype(np.float16).astype(np.float32)
    valid = np.isfinite(stored_depth) & (stored_depth > 0)
    ours_valid = np.isfinite(ours_depth) & (ours_depth > 0)
    row = {"depth_abs": np.abs(ours_depth[valid] - stored_depth[valid]),
           "validity_mismatch": int((valid != ours_valid).sum()),
           "depth_identical": bool(np.array_equal(clip["depth_front"][t], out["depth_front"].astype(np.float16))),
           "psnr": [], "psnr_reencoded": [], "jpeg_identical": []}
    for v, name in enumerate(VIEWS):
        stored_bytes = np.asarray(clip[f"rgb_{name}"][t], dtype=np.uint8)
        stored_rgb = decode_bgr_jpeg(stored_bytes)
        ours_jpeg = encode_like_collector(out["rgb"][v])
        row["psnr"].append(psnr(stored_rgb, out["rgb"][v]))
        row["jpeg_identical"].append(ours_jpeg == stored_bytes.tobytes())
        row["psnr_reencoded"].append(psnr(stored_rgb, decode_bgr_jpeg(np.frombuffer(ours_jpeg, np.uint8))))
    row["bit_exact"] = row["depth_identical"] and all(row["jpeg_identical"])
    return row


def self_check(renderer: CollectorRenderer, clip_dir: Path, out_dir: Path, max_frames: int) -> int:
    """Render a stored collector clip at its own poses; compare depth, RGB and the pose convention.

    Body state per frame = (trajectory_3d[t], rotation of pose_front[t]), which is
    exactly what the closed-loop trace provides (body position + rotation).
    Pass criteria (fixed before the first run):
      pose   every |c2w difference| <= 1e-4 (float32 storage), incl. habitat's own sensor pose;
      depth  on valid stored pixels (> 0, finite) at most 0.1% differ by > 1 cm, and at most
             0.1% of all pixels disagree on validity; the max-abs error is reported;
      rgb    PSNR(stored JPEG, our raw render) >= 30 dB for every frame and view.
    Supporting evidence (not a criterion): a frame is "bit-exact" when its front depth is
    identical in float16 on every pixel and all four views re-encode (collector JPEG path) to
    the stored bytes.  Frames that are not bit-exact with the recovered rotation are re-rendered
    with every float32 quaternion that reproduces the stored float32 matrix exactly
    (``float32_quaternion_candidates``); the closed-loop trace carries the exact quaternion.
    """
    t0 = time.time()
    clip = load_clip(clip_dir)
    meta = clip["meta"]
    num = len(clip["trajectory_3d"])
    frames = list(range(num))
    if 0 < max_frames < num:
        frames = sorted(set(np.linspace(0, num - 1, max_frames).round().astype(int).tolist()))
    glb = scene_glb(meta["scene_id"])
    load_s = renderer.load(glb)

    offset = np.array([0.0, SENSOR_HEIGHT_M, 0.0])
    pose = {"stored_front_translation_minus_body_plus_1p25": 0.0, "stored_front_yaw_only": 0.0,
            "stored_view_vs_front_at_Ry": 0.0, "c2w_from_body_state_vs_stored": [0.0] * len(VIEWS),
            "habitat_sensor_vs_c2w": 0.0}
    per_frame = []
    bit_exact = {"recovered_rotation": 0, "best_float32_candidate": 0, "frames_needing_candidates": 0,
                 "candidate_renders": 0}
    montage_frame = frames[len(frames) // 2]
    montage = None
    t1 = time.time()
    for t in frames:
        body_pos = clip["trajectory_3d"][t].astype(np.float64)
        stored_front = clip["pose_front"][t].astype(np.float64)
        rotation = matrix_to_quat_wxyz(stored_front[:3, :3])
        pose["stored_front_translation_minus_body_plus_1p25"] = max(
            pose["stored_front_translation_minus_body_plus_1p25"],
            float(np.abs(stored_front[:3, 3] - body_pos - offset).max()))
        pose["stored_front_yaw_only"] = max(pose["stored_front_yaw_only"], yaw_only_error(rotation))
        out = renderer.render(clip["trajectory_3d"][t], rotation)
        pose["habitat_sensor_vs_c2w"] = max(pose["habitat_sensor_vs_c2w"], out["sensor_pose_err"])
        for v, name in enumerate(VIEWS):
            stored_view = clip[f"pose_{name}"][t].astype(np.float64)
            R_y = np.eye(4)
            R_y[:3, :3] = quat_wxyz_to_matrix(yaw_quat_wxyz(VIEW_YAW_RAD[v]))
            pose["stored_view_vs_front_at_Ry"] = max(pose["stored_view_vs_front_at_Ry"],
                                                     float(np.abs(stored_front @ R_y - stored_view).max()))
            pose["c2w_from_body_state_vs_stored"][v] = max(pose["c2w_from_body_state_vs_stored"][v],
                                                           float(np.abs(out["c2w"][v] - stored_view).max()))
        row = compare_frame(clip, t, out)
        per_frame.append(row)
        exact = row["bit_exact"]
        bit_exact["recovered_rotation"] += int(exact)
        if not exact:
            bit_exact["frames_needing_candidates"] += 1
            for q in float32_quaternion_candidates(clip["pose_front"][t][:3, :3]):
                bit_exact["candidate_renders"] += 1
                if compare_frame(clip, t, renderer.render(clip["trajectory_3d"][t], q))["bit_exact"]:
                    exact = True
                    break
        bit_exact["best_float32_candidate"] += int(exact)
        if t == montage_frame:
            montage = (t, [decode_bgr_jpeg(clip[f"rgb_{n}"][t]) for n in VIEWS], out["rgb"],
                       clip["depth_front"][t].astype(np.float32),
                       out["depth_front"].astype(np.float16).astype(np.float32))
    render_s = time.time() - t1
    bit_exact["frames"] = len(frames)

    errors = np.concatenate([row["depth_abs"] for row in per_frame])
    mismatch = sum(row["validity_mismatch"] for row in per_frame)
    depth = {
        "valid_pixels": int(errors.size),
        "max_abs_m": float(errors.max()) if errors.size else None,
        "mean_abs_m": float(errors.mean()) if errors.size else None,
        "p99_abs_m": float(np.percentile(errors, 99)) if errors.size else None,
        "p999_abs_m": float(np.percentile(errors, 99.9)) if errors.size else None,
        "frac_exact": float((errors == 0).mean()) if errors.size else None,
        "frac_over_1mm": float((errors > 1e-3).mean()) if errors.size else None,
        "frac_over_1cm": float((errors > 1e-2).mean()) if errors.size else None,
        "validity_mismatch_pixels": mismatch,
        "validity_mismatch_frac": mismatch / float(len(per_frame) * IMAGE_SIZE * IMAGE_SIZE),
    }
    rgb = {}
    for v, name in enumerate(VIEWS):
        raw = [row["psnr"][v] for row in per_frame]
        reencoded = [row["psnr_reencoded"][v] for row in per_frame if not row["jpeg_identical"][v]]
        rgb[name] = {"psnr_db_mean": float(np.mean(raw)), "psnr_db_min": float(np.min(raw)),
                     "jpeg_bytes_identical": sum(row["jpeg_identical"][v] for row in per_frame),
                     "frames": len(per_frame),
                     "psnr_db_reencoded_min_non_identical": float(np.min(reencoded)) if reencoded else None}
    pose_max = max([pose["stored_front_translation_minus_body_plus_1p25"], pose["stored_front_yaw_only"],
                    pose["stored_view_vs_front_at_Ry"], pose["habitat_sensor_vs_c2w"]]
                   + pose["c2w_from_body_state_vs_stored"])
    passed = {
        "pose": pose_max <= POSE_TOL,
        "depth": bool(errors.size) and depth["frac_over_1cm"] <= 1e-3 and depth["validity_mismatch_frac"] <= 1e-3,
        "rgb": min(rgb[name]["psnr_db_min"] for name in VIEWS) >= 30.0,
    }
    passed["all"] = all(passed.values())

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"self_check_{meta['scene_id']}_{clip_dir.name}"
    montage_path = out_dir / f"{stem}_frame{montage[0]:03d}.png" if montage else None
    if montage:
        write_montage(montage_path, *montage[1:])
    record = {
        "schema": SELF_CHECK_SCHEMA,
        "clip": str(clip_dir),
        "scene_id": meta["scene_id"],
        "episode_id": meta.get("episode_id"),
        "frames": frames,
        "num_clip_frames": num,
        "body_state": "position = trajectory_3d[t]; rotation = quaternion of pose_front[t][:3,:3]",
        "pass_criteria": {
            "pose": f"every c2w / sensor-pose difference <= {POSE_TOL}",
            "depth": "valid stored pixels (> 0, finite): frac |err| > 1 cm <= 1e-3; validity mismatch <= 1e-3",
            "rgb": "min PSNR(stored JPEG, raw render) >= 30 dB over frames and views",
        },
        "passed": passed,
        "pose": pose,
        "depth_front": depth,
        "rgb": rgb,
        "bit_exact_frames": bit_exact,
        "bit_exact_definition": ("front depth identical in float16 on every pixel and all 4 views re-encode "
                                 "(cv2 BGR JPEG q90, collector io_utils) to the stored bytes; "
                                 "best_float32_candidate also tries every float32 quaternion whose collector "
                                 "matrix equals the stored float32 pose exactly "
                                 "(supporting evidence, not a criterion)"),
        "montage_png": str(montage_path) if montage_path else None,
        "scene_glb": str(glb),
        "sensor": {"habitat_lab_config": renderer.sensor_spec()},
        "collector": collector_provenance(),
        "versions": versions(),
        "timing_s": {"scene_load": round(load_s, 3), "render": round(render_s, 3),
                     "per_frame": round(render_s / max(len(frames), 1), 4), "total": round(time.time() - t0, 3)},
        "git_sha": git_sha(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "tool": "scripts/exp19/render_views.py --self-check",
    }
    report_path = out_dir / f"{stem}.json"
    write_json(report_path, record)
    write_json(out_dir / "self_check.json", record)
    print(json.dumps(to_jsonable({"passed": passed, "pose": pose, "depth_front": depth, "rgb": rgb,
                                  "bit_exact_frames": bit_exact}), indent=1))
    print(f"[self-check] {'PASS' if passed['all'] else 'FAIL'} -> {report_path}", flush=True)
    return 0 if passed["all"] else 1


def write_montage(path: Path, stored_rgb, ours_rgb, stored_depth, ours_depth):
    """Rows F,R,B,L: stored | rendered | 4x|diff|; last row: stored depth | rendered depth | 50x|diff| (grey)."""
    import cv2
    rows = []
    for v in range(len(VIEWS)):
        diff = np.clip(np.abs(stored_rgb[v].astype(np.int16) - ours_rgb[v].astype(np.int16)) * 4, 0, 255)
        rows.append(np.concatenate([stored_rgb[v], ours_rgb[v], diff.astype(np.uint8)], axis=1))
    scale = lambda d: np.repeat((np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)  # noqa: E731
    diff = np.repeat((np.clip(np.abs(stored_depth - ours_depth) * 50, 0, 255)).astype(np.uint8)[..., None], 3, axis=2)
    rows.append(np.concatenate([scale(stored_depth), scale(ours_depth), diff], axis=1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------- main ---

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", default=None, help="rerun dir runs/<run> (gpu*/steps, gpu*/trace)")
    src.add_argument("--steps", default=None, help="one episode's steps.jsonl (needs --trace-dir)")
    src.add_argument("--poses", default=None,
                     help="JSON list of {step, position, rotation_wxyz} (needs --scene, --ep-key)")
    src.add_argument("--self-check", action="store_true", help="render a stored collector clip and compare")
    p.add_argument("--trace-dir", default=None, help="with --steps: that episode's call trace dir")
    p.add_argument("--scene", default=None, help="with --poses: MP3D scene id or .glb path")
    p.add_argument("--ep-key", default=None, help="with --poses: output name <ep_key>.npz")
    p.add_argument("--episodes", nargs="*", default=None, help="with --run-dir: only these ep_keys")
    p.add_argument("--clip", default=str(DEFAULT_SELF_CHECK_CLIP), help="with --self-check: collector clip dir")
    p.add_argument("--max-frames", type=int, default=0,
                   help="with --self-check: evenly spaced subset of the clip's frames (0 = all)")
    p.add_argument("--out-dir", default=str(EXP_ROOT / "renders"))
    p.add_argument("--overwrite", action="store_true",
                   help="re-render episodes whose <ep_key>.json exists even if it matches the inputs")
    p.add_argument("--gpu", type=int, default=0, help="HABITAT_SIM_V0.GPU_DEVICE_ID (the collector's --gpu)")
    args = p.parse_args(argv)
    if args.steps and not args.trace_dir:
        p.error("--steps needs --trace-dir")
    if args.poses and not (args.scene and args.ep_key):
        p.error("--poses needs --scene and --ep-key")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    failed = []
    if args.self_check:
        jobs = []
    elif args.run_dir:
        jobs, failed = run_jobs(Path(args.run_dir), args.episodes)
    elif args.steps:
        jobs = [trace_job(Path(args.steps), Path(args.trace_dir))]
    else:
        jobs = [poses_job(Path(args.poses), args.scene, args.ep_key)]
    if not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is not set: run through run_render.sh or with_xvfb.sh (Xvfb + llvmpipe)")

    renderer = CollectorRenderer(gpu=args.gpu)
    try:
        if args.self_check:
            return self_check(renderer, Path(args.clip), out_dir, args.max_frames)
        done, skipped = 0, 0
        jobs.sort(key=lambda job: (job["scene_id"], job["ep_key"]))  # one scene load per scene
        for k, job in enumerate(jobs):
            existing = out_dir / f"{job['ep_key']}.json"
            npz_path = out_dir / f"{job['ep_key']}.npz"
            if not existing.exists() and npz_path.exists():
                # An npz without its json is a crashed write; [F] reads only the npz.
                print(f"[render] {k + 1}/{len(jobs)} {job['ep_key']}: removing a stray {npz_path.name} "
                      "without its json", flush=True)
                npz_path.unlink()
            if existing.exists():
                reason = "--overwrite" if args.overwrite else stale_reason(existing, job)
                if reason is None:
                    skipped += 1
                    print(f"[render] {k + 1}/{len(jobs)} {job['ep_key']}: exists, skipped", flush=True)
                    continue
                print(f"[render] {k + 1}/{len(jobs)} {job['ep_key']}: replacing the existing render ({reason})",
                      flush=True)
                # A failed re-render must not leave the old render behind for [F] to read.
                existing.unlink()
                npz_path.unlink(missing_ok=True)
            try:
                record = render_episode(renderer, job, out_dir)
                done += 1
                print(f"[render] {k + 1}/{len(jobs)} {job['ep_key']}: {len(record['steps'])} steps in "
                      f"{record['timing_s']['total']:.1f} s (eval-front NCC min "
                      f"{record['qa']['eval_front_ncc']['min']})", flush=True)
            except Exception:
                traceback.print_exc()
                failed.append(job["ep_key"])
                if not existing.exists():  # the json is written last: an npz alone is not a render
                    npz_path.unlink(missing_ok=True)
                print(f"[render] {k + 1}/{len(jobs)} {job['ep_key']}: FAILED", flush=True)
        print(f"[render] done {done}, skipped {skipped}, failed {len(failed)}: {failed}", flush=True)
        return 1 if failed else 0
    finally:
        renderer.close()


if __name__ == "__main__":
    sys.exit(main())
