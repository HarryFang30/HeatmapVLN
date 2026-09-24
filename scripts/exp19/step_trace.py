"""EXP-19: read-only per-step state trace for the closed-loop RPC client.

``scripts/evaluation/r2r_val_unseen.py`` imports this module lazily, and only
when ``--step_state_trace_dir`` is set.  With the flag unset the client never
imports it and its only extra work is an ``if step_tracer is not None`` check
at each hook.

For each episode the tracer writes ``<dir>/<scene>_<episode:04d>/steps.jsonl``
plus ``front_<step:04d>.jpg``.  The JPEG is the native front RGB from the
``observations`` dict the client already holds (640x480, JPEG q90).  It is
written once per new step id.  ``steps.jsonl`` holds one JSON object per line,
of these types:

* ``episode_start``: once, right after ``env.reset()``.  It gives the
  instruction as sent to the model and as stored in the dataset, the start
  state, goal, reference path, geodesic distance and RGB sensor spec.
* ``state``: written at the top of the client's per-step loop, the first time
  each step id is seen.  At that state frame ``step`` is captured, ingested
  by VO and appended to history.  The client can revisit a step id: a queued
  STOP flushes the queue and replans at the same step.  A revisit writes no
  new record.  One more state, flagged ``final``, is written by
  ``end_episode``.  It is the state after the last action, which the loop
  never revisits.
* ``action``: every primitive action that went through ``env.step``,
  written before the client increments ``step_id``.  ``step_before`` is
  the state it was taken from.  ``system2_call_index`` is the zero-based
  index of the call whose response produced the action, i.e. the client's
  ``system2_calls - 1``.  It is ``None`` for stops no call produced
  (auto-stop, max-System2 stop).  Records of the two sites that act on a
  terminal or action-less response (``terminal``, ``rpc_empty_actions``)
  also carry ``response_kind``, that response's ``kind`` (e.g. ``stop`` vs
  ``fallback_stop``).  Two things are not action records.  A
  STOP that only flushes the queue never reaches ``env.step``.  The lookdown
  capture's LOOK_DOWN/LOOK_UP pairs only tilt the sensor.
* ``episode_end``: step count, ``env.get_metrics()``, and how the episode
  ended.  ``stop`` means the episode is over and the last executed action
  was STOP.  ``step_cap`` means the step limit was reached without a stop.
  Anything else is ``other``.  ``terminal_response_kind`` is the stopping
  action's ``response_kind`` (None unless ``stop`` came from one of those
  two sites).

Poses are ground-truth simulator state, meant for offline geometry.  Figures
must not draw them.  Rotations are stored explicitly as ``[w, x, y, z]``,
read off the quaternion's named components.  The dataset's own start
rotation is kept alongside for cross-checking, in Habitat's ``[x, y, z, w]``
coefficient order.

The tracer only reads ``agent.get_state()`` and the observations the client
passes in.  It never calls ``set_state``, ``env.step`` or a sensor render,
and it uses no RNG.  It must import under ``envs/vlnce`` (Python 3.8, no
torch needed), so it uses only the standard library, numpy and PIL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCHEMA = "exp19-step-trace-v1"
STEPS_FILE = "steps.jsonl"
FRONT_JPEG_QUALITY = 90
STOP = 0  # ActionCode.STOP in the client

# Every client site that executes an action through env.step.
ACTION_PHASES = (
    "local_action",  # queued action from the previous response
    "rpc_first",  # first action of a response, executed right away
    "terminal",  # terminal response (STOP)
    "rpc_empty_actions",  # response without actions -> STOP
    "pano_recenter",  # recenter turn (pano_recenter_before_system1 only)
    "max_system2_stop",  # --max_system2_calls_per_episode reached -> STOP
    "auto_stop",  # --auto_stop_distance privileged stop
)
# The sites that act on a terminal / action-less response; they record its kind.
RESPONSE_KIND_PHASES = ("terminal", "rpc_empty_actions")


def episode_key(scene_id: str, episode_id: int) -> str:
    return f"{scene_id}_{int(episode_id):04d}"


def front_name(step: int) -> str:
    return f"front_{int(step):04d}.jpg"


def quat_wxyz(rotation: Any) -> list[float]:
    """[w, x, y, z] from a numpy-quaternion (habitat-sim AgentState.rotation).

    Reads the named components, so no coefficient-order convention is involved.
    """
    if not all(hasattr(rotation, name) for name in ("w", "x", "y", "z")):
        raise TypeError(f"expected a quaternion with w/x/y/z components, got {type(rotation)!r}")
    return [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sensor_spec(rgb_sensor_config: Any) -> dict[str, Any]:
    """RGB sensor geometry from the Habitat ``SIMULATOR.RGB_SENSOR`` config node."""
    position = list(rgb_sensor_config.POSITION)
    return {
        "rgb_hw": [int(rgb_sensor_config.HEIGHT), int(rgb_sensor_config.WIDTH)],
        "hfov_deg": float(rgb_sensor_config.HFOV),
        "sensor_height_m": float(position[1]),
    }


class StepStateTracer:
    """Writes one ``steps.jsonl`` (+ front JPEGs) per episode; see the module docstring."""

    def __init__(self, root: str | Path, rgb_sensor_config: Any) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sensor = sensor_spec(rgb_sensor_config)
        self.ep_key: str | None = None
        self._dir: Path | None = None
        self._fh = None
        self._seen_steps: set[int] = set()
        self._last_action: dict[str, Any] | None = None
        self._num_actions = 0
        print(f"[exp19-step-trace] writing per-step state traces to {self.root}", flush=True)

    # ── records ─────────────────────────────────────────────────────────

    def begin_episode(
        self,
        scene_id: str,
        episode_id: int,
        episode: Any,
        instruction: str,
        agent_state: Any,
    ) -> None:
        self._close()
        self.ep_key = episode_key(scene_id, episode_id)
        self._dir = self.root / self.ep_key
        # A resumed run restarts an unfinished episode from step 0; drop the
        # partial attempt so the directory holds exactly one attempt.
        stale = sorted(self._dir.glob("front_*.jpg")) if self._dir.exists() else []
        if stale or (self._dir / STEPS_FILE).exists():
            print(
                f"[exp19-step-trace] replacing an earlier attempt of {self.ep_key} ({len(stale)} frames)",
                flush=True,
            )
        for path in stale:
            path.unlink()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._dir / STEPS_FILE, "w", encoding="utf-8")
        self._seen_steps = set()
        self._last_action = None
        self._num_actions = 0

        goal = episode.goals[0]
        goal_radius = getattr(goal, "radius", None)
        info = getattr(episode, "info", None)
        geodesic = info.get("geodesic_distance") if isinstance(info, dict) else None
        reference_path = getattr(episode, "reference_path", None)
        self._write(
            {
                "type": "episode_start",
                "schema": SCHEMA,
                "scene_id": str(scene_id),
                "episode_id": int(episode_id),
                "ep_key": self.ep_key,
                "instruction": str(instruction),
                "instruction_raw": str(episode.instruction.instruction_text),
                "start_position": _floats(agent_state.position),
                "start_rotation_wxyz": quat_wxyz(agent_state.rotation),
                "dataset_start_position": _floats(episode.start_position),
                "dataset_start_rotation_xyzw": _floats(episode.start_rotation),
                "goal_position": _floats(goal.position),
                "goal_radius": float(goal_radius) if goal_radius is not None else None,
                "reference_path": (
                    [_floats(p) for p in reference_path] if reference_path is not None else None
                ),
                "geodesic_distance": float(geodesic) if geodesic is not None else None,
                "sensor": dict(self.sensor),
            }
        )

    def record_state(
        self,
        step: int,
        agent_state: Any,
        observations: dict,
        *,
        vo_frame_id: int | None,
        queue_len: int,
        final: bool = False,
    ) -> None:
        """State at step ``step``; a step id already recorded is skipped."""
        self._require_episode()
        step = int(step)
        if step in self._seen_steps:
            return
        self._seen_steps.add(step)
        name = front_name(step)
        self._save_front(observations, self._dir / name)
        sensor_pose = (getattr(agent_state, "sensor_states", None) or {}).get("rgb")
        self._write(
            {
                "type": "state",
                "step": step,
                "position": _floats(agent_state.position),
                "rotation_wxyz": quat_wxyz(agent_state.rotation),
                "rgb_sensor_position": _floats(sensor_pose.position) if sensor_pose is not None else None,
                "rgb_sensor_rotation_wxyz": quat_wxyz(sensor_pose.rotation) if sensor_pose is not None else None,
                "vo_frame_id": int(vo_frame_id) if vo_frame_id is not None else None,
                "queue_len": int(queue_len),
                "front_jpg": name,
                "final": bool(final),
            }
        )

    def record_action(
        self,
        step_before: int,
        action: int,
        phase: str,
        system2_call_index: int | None,
        *,
        response_kind: str | None = None,
    ) -> None:
        self._require_episode()
        if phase not in ACTION_PHASES:
            raise ValueError(f"unknown action phase {phase!r}; expected one of {ACTION_PHASES}")
        if response_kind is not None and phase not in RESPONSE_KIND_PHASES:
            raise ValueError(f"response_kind is recorded only for phases {RESPONSE_KIND_PHASES}, not {phase!r}")
        record = {
            "type": "action",
            "step_before": int(step_before),
            "action": int(action),
            "phase": phase,
            "system2_call_index": int(system2_call_index) if system2_call_index is not None else None,
        }
        if phase in RESPONSE_KIND_PHASES:
            record["response_kind"] = str(response_kind) if response_kind is not None else None
        self._write(record)
        self._last_action = record
        self._num_actions += 1

    def end_episode(
        self,
        steps: int,
        agent_state: Any,
        observations: dict,
        metrics: dict,
        *,
        done: bool,
        max_steps: int,
        system2_calls: int,
    ) -> None:
        self._require_episode()
        steps = int(steps)
        self.record_state(steps, agent_state, observations, vo_frame_id=None, queue_len=0, final=True)
        last = self._last_action
        if done and last is not None and last["action"] == STOP:
            ended_by = "stop"
        elif not done and steps >= int(max_steps):
            ended_by = "step_cap"
        else:
            ended_by = "other"
        self._write(
            {
                "type": "episode_end",
                "steps": steps,
                "metrics": _jsonable(dict(metrics)),
                "ended_by": ended_by,
                "stop_phase": last["phase"] if ended_by == "stop" else None,
                "terminal_response_kind": last.get("response_kind") if ended_by == "stop" else None,
                "done": bool(done),
                "max_steps": int(max_steps),
                "vlm_calls": int(system2_calls),
                "num_states": len(self._seen_steps),
                "num_actions": self._num_actions,
            }
        )
        self._close()
        self.ep_key = None
        self._dir = None

    # ── helpers ─────────────────────────────────────────────────────────

    def _save_front(self, observations: dict, path: Path) -> None:
        rgb = np.asarray(observations["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"front RGB must be HxWx3/4, got shape {rgb.shape}")
        if list(rgb.shape[:2]) != self.sensor["rgb_hw"]:
            raise ValueError(f"front RGB is {rgb.shape[:2]}, sensor spec says {self.sensor['rgb_hw']}")
        # astype copies, so the client's observation buffer is never shared with PIL.
        Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(path, "JPEG", quality=FRONT_JPEG_QUALITY)

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _require_episode(self) -> None:
        if self._fh is None:
            raise RuntimeError("step trace record outside an episode (begin_episode not called)")

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_steps(path: str | Path) -> list[dict]:
    """All records of one ``steps.jsonl`` (a file or its episode directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / STEPS_FILE
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
