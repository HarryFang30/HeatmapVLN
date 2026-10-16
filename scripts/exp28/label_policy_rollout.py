#!/usr/bin/env python3
"""EXP-28 K1: line B as System2 in closed loop -- does it bring an off-route agent into the 3 m circle?

K1 (ledger EXP-28 C, written 2026-10-15 19:35 UTC before any HM3D closed-loop result) tests the labeller
before its labels train anything.  Off-route states of the base model's own training-cohort records
(``src/data/s2_records.py``) are restored in Habitat and line B answers every System2 call from there on
(``s2_route_labels.route_oracle`` with ``RouteLabelConfig()``; K1 may only fix bugs, never rule parameters,
so no rule parameter is a flag here):

* pixel -> its floor point (``s2_route_labels.pixel_floor_target``: the lookdown ray's floor point, cut short
           by the depth), snapped when it is not navigable, goes to habitat-lab's ``ShortestPathFollower``
           (goal_radius 0.25).  As InternNav's ``habitat_vln_evaluator._run_eval_system2`` does: at most
           ``--spf-max-actions`` (8, its MAX_STEPS) follower actions per goal, then System2 is asked again;
           a follower that stops at once gets one TURN_LEFT (its "random action to avoid deadlock");
* turn  -> its arrows are executed (``k`` x TURN_LEFT / TURN_RIGHT);
* skip  -> one MOVE_FORWARD;

until the geodesic distance to the goal is <= 3 m (checked after every primitive action) or ``--max-steps``
(500, the collection's episode cap) primitive actions.  At every call the lookdown is rendered the way the
client captures it (two 15 deg LOOK_DOWN of the client's own DEPTH_SENSOR, depth clipped to [0, 10] m and
quantised to the record's millimetre PNG; the sensor pose is checked against ``LookdownCamera`` every time),
and line B's route is rebuilt as ``label_route_corrections.label_record_rows`` builds it: the last reference
waypoint reached by any position visited so far (the record's up to the restored step, then the rollout's),
then the rejoin slack.  The "previous executed turn" of the first call is the record's answer before the
restored pose; after that it is line B's own (a pixel or a skip executes no turn).  The SPF expert starts
from the same restored pose with a fresh ``ShortestPathFollower`` (goal_radius 0.25) to the episode goal and
stops at the same 3 m test.

=====  ============================================================================================  =======
reach  share of states whose line-B rollout enters the 3 m circle                                     >= 0.90
flip   pairs of consecutive line-B calls of one rollout that both turn: share turning opposite ways   <= 0.01
ratio  median over states of line-B steps / expert steps; a line-B failure counts as infinite,        <= 1.5
       a state whose expert fails is left out (and counted)
=====  ============================================================================================  =======

Thresholds are compared in whole counts (a value exactly on its line passes).  Steps are primitive movement
actions (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT); the lookdown's LOOK_DOWN / LOOK_UP are not counted, for either.

Validity (no pass or fail is read otherwise; these are checks of the measurement, not K1 lines): every
selected state has exactly one row of this selection, line-B config and rollout settings; a state rolled out
twice gave the same result; at the restored pose of a state taken at a call, the lookdown rendered here is the
record's own lookdown depth of that call (>= 90% of the pixels within 5 cm, at all but 1% of those states),
so K1 runs in the world the records were collected in; the SPF expert, the ratio's reference, entered the
circle from at least 90% of the states.

States.  The training cohort of ``build_hm3d_episodes.py`` (parts ``train_*``, shards in index order,
episodes in shard order) is read until at least 100 episodes with a committed record have been scanned and
at least 200 states taken.  Per episode at most ``--max-states-per-episode`` (3) states, in this order, each
more than 3 m (geodesic) from the goal:

* ``bad_first_pixel``: the episode's first pixel goal is bad by EXP-25's B (``probe_metrics``: the first
  ``trajectory`` call answering ``u v`` whose floor point -- the lookdown ray on the agent's floor plane, no
  depth -- snaps and reaches the goal; bad when (g - g_T) / max(t, 0.25) <= -0.3).  The state is the pose
  right after it ran: that of the next System2 call, or the episode's last step when no call followed;
* ``off_route``: a System2 call more than 2 m (3-D) from the reference route (the navmesh paths between
  consecutive reference waypoints, the goal appended as the labeller does) whose previous call was not:
  the first call of each excursion.

Only the cohort's ``train*`` parts are read, and a scene of ``splits/heldout_scenes.txt`` among them is an
error.  Every record must name the navmesh it was collected on (``navmesh_sha256``, EXP-28's client) and that
must be the file at ``<scenes-dir>/<scene>/<scene>.navmesh``.

Subcommands::

    select  CPU, habitat_sim PathFinder on <scenes-dir>/<scene>/<scene>.navmesh (no renderer).  Writes
            k1_states.jsonl and k1_selection.json to --out-dir; refuses to replace them without --force.
            Exit 0 with enough states, 2 otherwise.
    run     Renders (an X display with llvmpipe GLX, as the client; --xvfb-display N starts a private Xvfb).
            Reads the scenes and episode files select read.  Rolls out the states of --shard i/n (dealt by
            scene) that no rollouts file holds yet and appends one row per state to rollouts.shardIIofNN.jsonl
            (resumable; a smoke run's rows count and are not redone).
    report  No simulator.  k1_summary.json and the verdict.  Exit 0 pass, 1 fail, 2 not measurable (see
            Validity; also too few episodes or states).

4090 container (fjl-habitat), CPU rendering::

    cd /workspace/exp23/wt_<commit> && export PYTHONPATH=$PWD
    /opt/conda/bin/python scripts/exp28/label_policy_rollout.py select \\
        --cohort-dir /workspace/exp28/episodes/cohorts \\
        --record-root /workspace/exp28/collect_local/records --record-root /workspace/exp28/collect_base/records \\
        --scenes-dir /workspace/exp28/scene_datasets/hm3d --out-dir /workspace/exp28/k1
    /opt/conda/bin/python scripts/exp28/label_policy_rollout.py run --out-dir /workspace/exp28/k1 \\
        --shard 0/1 --limit 2 --xvfb-display 760                                  # smoke
    for i in 0 1 2 3; do /opt/conda/bin/python scripts/exp28/label_policy_rollout.py run \\
        --out-dir /workspace/exp28/k1 --shard $i/4 --xvfb-display $((761 + i)) \\
        > /workspace/exp28/k1/run_$i.log 2>&1 & done; wait
    /opt/conda/bin/python scripts/exp28/label_policy_rollout.py report --out-dir /workspace/exp28/k1
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import hashlib
import json
import math
import os
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp25 import probe_metrics as pm  # noqa: E402
from scripts.exp28 import verify_hm3d_render as v0  # noqa: E402  (client config, Xvfb, versions)
from scripts.tools import label_route_corrections as lr  # noqa: E402  (numpy-only loader of src/data)
from scripts.tools.build_r2r_train_cohorts import SCHEMA as COHORT_SCHEMA  # noqa: E402

REC, ORL, RL = lr.REC, lr.ORL, lr.RL
base = lr.base

SCHEMA = "heatmapvln-exp28-k1-v1"
STATE_SCHEMA = "heatmapvln-exp28-k1-state-v1"
ROLLOUT_SCHEMA = "heatmapvln-exp28-k1-rollout-v1"
STATES_NAME = "k1_states.jsonl"
SELECTION_NAME = "k1_selection.json"
SUMMARY_NAME = "k1_summary.json"

# Ledger EXP-28 C / K1, written 2026-10-15 19:35 UTC before any HM3D result.  Not options.
MIN_EPISODES = 100
MIN_STATES = 200
OFF_ROUTE_M = 2.0
SUCCESS_RADIUS_M = 3.0
SPF_GOAL_RADIUS_M = 0.25
REACH_MIN = Fraction(9, 10)
FLIP_MAX = Fraction(1, 100)
RATIO_MEDIAN_MAX = Fraction(3, 2)
BAD_PIXEL_RATIO = pm.BAD  # -0.3: EXP-25's B ("first pixel goal bad")
PIXEL_MIN_TDIST_M = 0.25  # probe_metrics.pixel_ratio's max(tdist, 0.25)

# Implementation choices (not in the ledger); every rollout row records them and the report refuses a mix.
MAX_STEPS = 500  # the collection's episode cap (run_s2_dagger_collect_cuda.sh MAX_STEPS)
SPF_MAX_ACTIONS = 8  # InternNav habitat_vln_evaluator.MAX_STEPS: follower actions per pixel goal
MAX_STATES_PER_EPISODE = 3
EPISODE_STEP_CAP = 500  # report-only: entered the circle within what the episode still had

# Validity of the measurement (not K1 lines): no pass or fail is read when these fail.
# The lookdown rendered at a restored call pose must be the one the record holds for that call (same scene,
# pose and camera); per state, the share of pixels within DEPTH_MATCH_TOL_M must be >= DEPTH_MATCH_MIN.
DEPTH_MATCH_TOL_M = 0.05
DEPTH_MATCH_MIN = 0.9
DEPTH_MISMATCH_MAX = Fraction(1, 100)  # share of checked states allowed below DEPTH_MATCH_MIN
EXPERT_FAILED_MAX = Fraction(1, 10)  # the ratio's reference may fail from at most this share of the states

# The client's ActionCode, which is habitat-lab's HabitatSimActions under ACTION_SPACE_CONFIG "v1".
STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5
ACTION_NAMES = {FORWARD: "forward", LEFT: "left", RIGHT: "right"}
HABITAT_ACTIONS = {"STOP": STOP, "MOVE_FORWARD": FORWARD, "TURN_LEFT": LEFT, "TURN_RIGHT": RIGHT,
                   "LOOK_UP": LOOK_UP, "LOOK_DOWN": LOOK_DOWN}
POSE_TOL = 1e-4  # restored pose vs requested (m / rotation-matrix entries)
LOOKDOWN_POSE_TOL = 1e-3  # rendered depth sensor vs LookdownCamera


class K1Invalid(RuntimeError):
    """The measurement itself is broken (inputs, navmesh, camera); no pass or fail can be read from it."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rate(k: int, n: int) -> float | None:
    return round(k / n, 6) if n else None


def _num(value: Fraction | float | None) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return round(float(value), 6)


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rows(path: str | os.PathLike) -> tuple[list[dict[str, Any]], int]:
    """Complete rows of an append-only jsonl, and the byte length of an unterminated tail (0: none).

    A run killed mid-write, or another shard still writing, can leave a last line without its newline.
    """
    data = Path(path).read_bytes()
    cut = data.rfind(b"\n") + 1
    rows = [json.loads(line) for line in data[:cut].decode("utf-8").splitlines() if line.strip()]
    return rows, len(data) - cut


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """One row, flushed and fsynced: a killed run loses at most the state it was on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def scene_key(name: str) -> str:
    """The scene as records key it: ``00481-5jLhtVmWd5F`` (the HM3D directory) and ``5jLhtVmWd5F`` alike."""
    name = name.strip()
    head, sep, tail = name.partition("-")
    return tail if sep and head.isdigit() and tail else name


def read_scene_list(path: str | os.PathLike) -> list[str]:
    """One scene per line (``#`` comments and blank lines skipped)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def parse_shard(text: str) -> tuple[int, int]:
    index, _, count = str(text).partition("/")
    i, n = int(index), int(count or 1)
    if not 0 <= i < n:
        raise ValueError(f"--shard {text!r}: need i/n with 0 <= i < n")
    return i, n


def shard_of(scene: str, count: int) -> int:
    """States are dealt by scene, so a process loads each of its scenes once."""
    return int(hashlib.sha256(str(scene).encode("utf-8")).hexdigest()[:12], 16) % count


def _turns(answer: Any) -> list[str] | None:
    if answer is None:
        return None
    return [str(answer.turn1)] + ([str(answer.turn2)] if answer.turn2 is not None else [])


def yaw_deg(rotation_wxyz: Sequence[float]) -> float:
    return math.degrees(v0.yaw_of_wxyz(rotation_wxyz))


def _rot_err(a_wxyz: Sequence[float], b_wxyz: Sequence[float]) -> float:
    return float(np.abs(ORL.quat_wxyz_to_rot(a_wxyz) - ORL.quat_wxyz_to_rot(b_wxyz)).max())


# --------------------------------------------------------------------------- #
# Cohort and records
# --------------------------------------------------------------------------- #
def cohort_episodes(
    cohort_dir: str | os.PathLike, parts: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The cohort's episodes of ``parts`` (default: every ``train*`` part), shards in index order."""
    cohort_dir = Path(cohort_dir)
    manifest_path = cohort_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != COHORT_SCHEMA:
        raise K1Invalid(f"{manifest_path}: schema {manifest.get('schema')!r} is not {COHORT_SCHEMA}")
    known = list(manifest.get("parts") or {})
    wanted = list(parts) if parts else [p for p in known if p.startswith("train")]
    unknown = [p for p in wanted if p not in known]
    if not wanted or unknown:
        raise K1Invalid(f"{manifest_path}: parts {wanted} (unknown {unknown}); the manifest has {known}")
    episodes: list[dict[str, Any]] = []
    for shard in sorted(manifest["shards"], key=lambda s: int(s["index"])):
        if shard.get("part") not in wanted:
            continue
        shard_path = cohort_dir / shard["path"]
        if shard.get("sha256") and sha256_file(shard_path) != shard["sha256"]:
            raise K1Invalid(f"{shard_path} is not the shard the manifest lists (sha256)")
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        for entry in data["episodes"]:
            episodes.append({"scene_id": str(entry["scene_id"]), "episode_id": int(entry["episode_id"]),
                             "shard": int(shard["index"]), "part": str(shard["part"])})
    keys = [(e["scene_id"], e["episode_id"]) for e in episodes]
    if len(keys) != len(set(keys)):
        raise K1Invalid(f"{manifest_path}: an episode is in two shards")
    info = {"cohort_dir": str(cohort_dir), "manifest_sha256": sha256_file(manifest_path), "parts": wanted,
            "episodes": len(episodes)}
    return episodes, info


def find_record(record_roots: Sequence[str | os.PathLike], scene_id: str, episode_id: int) -> Path | None:
    """The committed record of an episode under any root; the same episode under two roots is an error."""
    found = []
    for root in record_roots:
        path = REC.record_path(root, scene_id, episode_id)
        if path.is_file() and path.with_name(path.name + ".sha256").is_file():
            found.append(path)
    if len(found) > 1:
        raise K1Invalid(f"{scene_id}/{episode_id} is recorded twice: {[str(p) for p in found]}")
    return found[0] if found else None


# --------------------------------------------------------------------------- #
# State selection (numpy; geometry injected)
# --------------------------------------------------------------------------- #
def reference_polyline(waypoints: np.ndarray, find_path: Callable) -> tuple[np.ndarray, int]:
    """The reference route as walked on the navmesh, and how many segments had to stay straight lines."""
    parts, straight = [], 0
    for k in range(len(waypoints) - 1):
        segment = find_path(waypoints[k], waypoints[k + 1])
        if segment is None or not len(segment):
            segment = np.stack([waypoints[k], waypoints[k + 1]])
            straight += 1
        parts.append(np.asarray(segment, dtype=np.float64).reshape(-1, 3))
    if not parts:
        return np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)[:1], 0
    return ORL.concat_paths(parts), straight


def floor_point(pixel: Sequence[float], position: Sequence[float], rotation_wxyz: Sequence[float],
                camera: Any) -> np.ndarray | None:
    """The lookdown ray of ``pixel`` on the agent's floor plane: ``probe_metrics.pixel_floor`` for a pose.

    No depth and no image bounds, as B's definition; None when the ray does not go down.
    """
    u, v = float(pixel[0]), float(pixel[1])
    c2w = camera.c2w(position, rotation_wxyz)
    cx, cy = camera.centre
    direction = c2w[:3, :3] @ np.array([(u - cx) / camera.focal, -(v - cy) / camera.focal, -1.0])
    if direction[1] >= -1e-6:
        return None
    origin = c2w[:3, 3]
    point = origin + ((float(position[1]) - origin[1]) / direction[1]) * direction
    point[1] = float(position[1])
    return point


def first_pixel_goal(items: Sequence[Mapping[str, Any]], geometry: Any, goal: np.ndarray, camera: Any) -> dict | None:
    """EXP-25's first pixel goal of an episode (``probe_metrics.first_pixel``) and its class.

    ``items`` are the episode's calls in order with ``index``, ``position``, ``rotation``, ``d``, the recorded
    RPC ``kind`` and the parsed ``model`` answer.  The first ``trajectory`` call (B reads the client log's
    ``RPC kind=trajectory``) answering a pixel whose floor point snaps and has a finite geodesic to the goal;
    ratio (g - g_T) / max(t, 0.25): bad <= -0.3, good >= 0.5, mid between.
    """
    for i, item in enumerate(items):
        model = item["model"]
        if item.get("kind") != "trajectory" or model.kind != "pixel" or model.pixel is None:
            continue
        target = floor_point(model.pixel, item["position"], item["rotation"], camera)
        if target is None:
            continue
        snapped = geometry.snap(target)
        if snapped is None or not np.all(np.isfinite(np.asarray(snapped, dtype=np.float64))):
            continue
        g_target = float(geometry.geodesic(np.asarray(snapped, dtype=np.float64), goal))
        if not math.isfinite(g_target):
            continue
        p = np.asarray(item["position"], dtype=np.float64)
        tdist = float(math.hypot(target[0] - p[0], target[2] - p[2]))
        ratio = (float(item["d"]) - g_target) / max(tdist, PIXEL_MIN_TDIST_M)
        cls = "bad" if ratio <= BAD_PIXEL_RATIO else ("good" if ratio >= pm.GOOD else "mid")
        return {"item": i, "call_index": int(item["index"]), "pixel": list(model.pixel), "g": ORL.rounded(item["d"]),
                "g_target": ORL.rounded(g_target), "tdist_m": ORL.rounded(tdist), "ratio": ORL.rounded(ratio),
                "class": cls}
    return None


def record_states(
    record: Any,
    geometry: Any,
    *,
    dataset_entry: Mapping[str, Any] | None,
    config: Any,
    camera: Any = None,
    max_states: int = MAX_STATES_PER_EPISODE,
    off_route_m: float = OFF_ROUTE_M,
    success_radius_m: float = SUCCESS_RADIUS_M,
) -> tuple[list[dict[str, Any]], collections.Counter]:
    """K1's states of one record (see the module doc) and what was passed over, by reason.

    ``geometry`` has ``geodesic``, ``find_path`` and ``snap`` (the labeller's navmesh geometry).
    """
    camera = camera if camera is not None else ORL.LookdownCamera()
    stats: collections.Counter = collections.Counter()
    record_key = REC.episode_key(record.scene_id, record.episode_id)
    ref, goal, start = base.resolve_episode_geometry(record.episode, dataset_entry)
    waypoints = ORL.route_waypoints(ref, goal)
    polyline, straight = reference_polyline(waypoints, geometry.find_path)
    stats["reference_segments_straight"] += straight

    steps = sorted(record.steps, key=lambda s: int(s["step_id"]))
    step_ids = [int(s["step_id"]) for s in steps]
    positions = [s["position"] for s in steps]
    if start is not None:
        step_ids, positions = [-1] + step_ids, [start] + positions
    reached = ORL.last_reached_by_step(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), waypoints, geometry.geodesic, config.reached_radius_m
    )

    items: list[dict[str, Any]] = []
    for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
        position, rotation = base.call_pose(record, call)
        p = np.asarray(position, dtype=np.float64)
        items.append({
            "index": int(call["call_index"]), "step_id": int(call["step_id"]), "position": position,
            "rotation": rotation, "d": float(geometry.geodesic(p, goal)), "kind": call.get("kind"),
            "off": float(ORL.nearest_on_polyline(polyline, p)[0]), "model": ORL.parse_model_answer(call),
        })

    def restore(tag: str, step_id: int, position: Sequence[float], rotation: Sequence[float],
                call_index: int | None, previous: Any, d: float | None = None) -> dict[str, Any]:
        p = np.asarray(position, dtype=np.float64)
        upto = int(np.searchsorted(step_ids, int(step_id), side="right")) - 1
        last = max(int(reached[upto]) if upto >= 0 else -1,
                   int(ORL.last_reached_by_step(p[None], waypoints, geometry.geodesic, config.reached_radius_m)[0]))
        return {
            "tag": tag, "call_index": call_index, "step_id": int(step_id),
            "position": [float(v) for v in position], "rotation_wxyz": [float(v) for v in rotation],
            "d": float(geometry.geodesic(p, goal)) if d is None else float(d),
            "off": float(ORL.nearest_on_polyline(polyline, p)[0]), "last_reached": last,
            "previous_turns": _turns(previous), "previous_turn_direction": RL.executed_turn_direction(previous),
        }

    def at_call(j: int) -> dict[str, Any]:
        item = items[j]
        return restore(f"c{item['index']:04d}", item["step_id"], item["position"], item["rotation"], item["index"],
                       items[j - 1]["model"] if j > 0 else None, item["d"])

    candidates: list[tuple[str, dict[str, Any]]] = []
    first = first_pixel_goal(items, geometry, goal, camera)
    if first is not None:
        stats[f"first_pixel_{first['class']}"] += 1
    else:
        stats["first_pixel_none"] += 1
    if first is not None and first["class"] == "bad":
        i = first["item"]
        if i + 1 < len(items):
            candidates.append(("bad_first_pixel", at_call(i + 1)))
        elif steps and int(steps[-1]["step_id"]) > items[i]["step_id"]:
            last = steps[-1]
            candidates.append(("bad_first_pixel", restore("end", int(last["step_id"]), last["position"],
                                                          last["rotation_wxyz"], None, items[i]["model"])))
        else:
            stats["bad_first_pixel_without_later_pose"] += 1
    previous_off = False
    for j, item in enumerate(items):
        off = item["off"] > off_route_m
        if off and not previous_off:
            candidates.append(("off_route", at_call(j)))
        previous_off = off

    states: list[dict[str, Any]] = []
    by_tag: dict[str, dict[str, Any]] = {}
    sidecar = base._sidecar_digest(Path(record.path))
    for source, r in candidates:
        if r["tag"] in by_tag:
            by_tag[r["tag"]]["sources"].append(source)
            continue
        if not math.isfinite(r["d"]):
            stats[f"{source}_goal_unreachable"] += 1
            continue
        if r["d"] <= success_radius_m:
            stats[f"{source}_within_radius"] += 1
            continue
        if len(states) >= max_states:
            stats[f"{source}_over_episode_cap"] += 1
            continue
        state = {
            "schema": STATE_SCHEMA,
            "key": f"{record_key}#{r['tag']}",
            "record_key": record_key,
            "record_path": str(record.path),
            "record_sha256": sidecar,
            "scene_id": record.scene_id,
            "episode_id": int(record.episode_id),
            "navmesh_sha256": record.episode.get("navmesh_sha256"),
            "sources": [source],
            "restore": {k: r[k] for k in ("call_index", "step_id", "position", "rotation_wxyz")},
            "distance_to_goal": float(r["d"]),
            "off_route_m": float(r["off"]),
            "last_reached": int(r["last_reached"]),
            "num_waypoints": int(len(waypoints)),
            "previous_turns": r["previous_turns"],
            "previous_turn_direction": int(r["previous_turn_direction"]),
            "first_pixel": {k: v for k, v in first.items() if k != "item"} if source == "bad_first_pixel" else None,
        }
        by_tag[r["tag"]] = state
        states.append(state)
        stats[f"states_{source}"] += 1
    return states, stats


def select_states(
    episodes: Sequence[Mapping[str, Any]],
    *,
    record_roots: Sequence[str | os.PathLike],
    geometry_for_scene: Callable[[str], tuple[Any, str | None]],
    config: Any,
    dataset_index: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    camera: Any = None,
    min_episodes: int = MIN_EPISODES,
    min_states: int = MIN_STATES,
    max_states_per_episode: int = MAX_STATES_PER_EPISODE,
    max_episodes: int = 0,
    holdout: Collection[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Episodes in cohort order until >= ``min_episodes`` scanned and >= ``min_states`` taken.

    ``geometry_for_scene(scene)`` gives ``(geometry, navmesh sha256 or None)``; with a sha256 (always, outside
    the tests) the record must name exactly that navmesh (``navmesh_sha256``, which the EXP-28 client writes
    for every HM3D episode: a record without it is not one of them).  A held-out scene among the episodes is
    refused too.
    """
    holdout = {scene_key(s) for s in holdout}
    stats: collections.Counter = collections.Counter()
    states: list[dict[str, Any]] = []
    scanned, contributing, missing = 0, 0, []
    navmeshes: dict[str, str | None] = {}
    policies: collections.Counter = collections.Counter()
    stopped = "cohort_exhausted"
    for position, item in enumerate(episodes):
        scene, episode_id = str(item["scene_id"]), int(item["episode_id"])
        if scene_key(scene) in holdout:
            raise K1Invalid(f"episode {scene}/{episode_id} of the cohort parts is in a held-out scene")
        path = find_record(record_roots, scene, episode_id)
        if path is None:
            stats["episodes_without_record"] += 1
            missing.append(REC.episode_key(scene, episode_id))
            continue
        record = REC.EpisodeRecord(path)
        if record.scene_id != scene or record.episode_id != episode_id:
            raise K1Invalid(f"{path}: holds {record.scene_id}/{record.episode_id}, not {scene}/{episode_id}")
        geometry, digest = geometry_for_scene(scene)
        navmeshes[scene] = digest
        expected = record.episode.get("navmesh_sha256")
        if digest is not None and expected != digest:
            raise K1Invalid(f"{path}: collected on navmesh {expected} (None: not an EXP-28 HM3D record), "
                            f"the scene's navmesh is {digest}")
        policies[f"{record.episode.get('rpc_policy_mode')}|{record.episode.get('rpc_policy_fingerprint')}"] += 1
        entry = None
        if dataset_index is not None:
            entry = dataset_index.get((scene, episode_id))
            if entry is None:
                raise K1Invalid(f"{scene}/{episode_id} is not in --episodes")
        found, why = record_states(record, geometry, dataset_entry=entry, config=config, camera=camera,
                                   max_states=max_states_per_episode)
        stats.update(why)
        scanned += 1
        contributing += bool(found)
        states.extend(found)
        if scanned >= min_episodes and len(states) >= min_states:
            stopped = f"enough_after_cohort_position_{position}"
            break
        if max_episodes and scanned >= max_episodes:
            stopped = "max_episodes"
            break
    meta = {
        "episodes_scanned": scanned,
        "episodes_with_states": contributing,
        "states": len(states),
        "stopped": stopped,
        "enough": scanned >= MIN_EPISODES and len(states) >= MIN_STATES,
        "states_by_source": dict(collections.Counter(s for st in states for s in st["sources"])),
        "scenes": sorted({st["scene_id"] for st in states}),
        "navmesh_sha256": dict(sorted(navmeshes.items())),
        # Report-only: the server policies of the scanned records (K1 reads the base's greedy collection).
        "record_policies": dict(sorted(policies.items())),
        "missing_records": {"n": len(missing), "first": missing[:20]},
        "stats": dict(sorted(stats.items())),
        "settings": {"min_episodes": min_episodes, "min_states": min_states,
                     "max_states_per_episode": max_states_per_episode, "max_episodes": max_episodes,
                     "off_route_m": OFF_ROUTE_M, "success_radius_m": SUCCESS_RADIUS_M,
                     "bad_pixel_ratio": BAD_PIXEL_RATIO, "config_digest": config.digest(),
                     "rules_version": RL.ROUTE_RULES_VERSION},
    }
    return states, meta


# --------------------------------------------------------------------------- #
# Rollouts (any world: HabitatWorld here, a flat kinematic world in the tests)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class RolloutSettings:
    success_radius_m: float = SUCCESS_RADIUS_M
    max_steps: int = MAX_STEPS
    spf_goal_radius_m: float = SPF_GOAL_RADIUS_M
    spf_max_actions: int = SPF_MAX_ACTIONS

    def __post_init__(self) -> None:
        # Every System2 call must execute at least one action, or a rollout could stand still forever.
        if self.max_steps < 1 or self.spf_max_actions < 1:
            raise ValueError(f"max_steps and spf_max_actions must be >= 1: {self}")


class _Run:
    """One rollout's primitive actions, with the 3 m test after each and the step cap.

    ``world`` has ``pose()``, ``act(action)``; ``geometry.geodesic`` is the labeller's.
    """

    def __init__(self, world: Any, geometry: Any, goal: np.ndarray, settings: RolloutSettings):
        self.world, self.geometry, self.goal, self.settings = world, geometry, goal, settings
        self.steps = 0
        self.actions: list[int] = []
        self.moved: list[np.ndarray] = []  # positions after each MOVE_FORWARD
        self.outcome: str | None = None
        self.d = self._distance()
        self.d0 = self.d
        self.min_d = self.d
        self._check()

    def _distance(self) -> float:
        return float(self.geometry.geodesic(np.asarray(self.world.pose()[0], dtype=np.float64), self.goal))

    def _check(self) -> bool:
        if self.d <= self.settings.success_radius_m:
            self.outcome = "reached"
        elif not math.isfinite(self.d):
            self.outcome = "goal_unreachable"
        elif self.steps >= self.settings.max_steps:
            self.outcome = "step_cap"
        return self.outcome is not None

    def act(self, action: int) -> bool:
        """One primitive action; True once the rollout is over."""
        if self.outcome is not None:
            return True
        action = int(action)
        if action not in ACTION_NAMES:
            raise ValueError(f"not a movement action: {action}")
        self.world.act(action)
        self.steps += 1
        self.actions.append(action)
        if action == FORWARD:
            self.moved.append(np.asarray(self.world.pose()[0], dtype=np.float64))
        self.d = self._distance()
        self.min_d = min(self.min_d, self.d)
        return self._check()

    def summary(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reached": self.outcome == "reached",
            "steps": self.steps,
            "d0": ORL.rounded(self.d0),
            "final_d": ORL.rounded(self.d),
            "min_d": ORL.rounded(self.min_d),
            "action_counts": dict(collections.Counter(ACTION_NAMES[a] for a in self.actions)),
        }


def follow(follower: Callable[[np.ndarray], Any], goal: np.ndarray) -> tuple[int | None, str | None]:
    """The follower's next action, or (None, None) when it stops, or (None, error) when it fails."""
    try:
        action = follower(np.asarray(goal, dtype=np.float64))
    except Exception as err:  # noqa: BLE001 -- habitat_sim.errors.GreedyFollowerError and friends
        return None, f"{type(err).__name__}: {err}"
    if action is None or int(action) == STOP:
        return None, None
    if int(action) not in ACTION_NAMES:
        return None, f"unexpected_action_{int(action)}"
    return int(action), None


def _pixel_goal(run: _Run, world: Any, follower: Callable, pixel: Sequence[int], position: Sequence[float],
                rotation: Sequence[float], depth: np.ndarray, camera: Any, settings: RolloutSettings) -> dict:
    """A pixel answer, executed as InternNav's S2 + ShortestPathFollower loop executes one."""
    target, how = RL.pixel_floor_target(tuple(int(c) for c in pixel), position=position, rotation_wxyz=rotation,
                                        depth_m=depth, camera=camera)
    info: dict[str, Any] = {"how": how, "target": ORL.rounded(target) if target is not None else None}
    if target is None:
        info["fallback"] = "no_floor_point"
        run.act(FORWARD)
        return info
    goal_point = np.asarray(target, dtype=np.float64)
    if not world.is_navigable(goal_point):
        snapped = world.snap(goal_point)
        if snapped is None:
            info["fallback"] = "snap_failed"
            run.act(FORWARD)
            return info
        snapped = np.asarray(snapped, dtype=np.float64)
        info["snap_m"] = ORL.rounded(float(np.linalg.norm(snapped - goal_point)))
        goal_point = snapped
    info["goal"] = ORL.rounded(goal_point)
    info["goal_dy_m"] = ORL.rounded(float(goal_point[1] - float(position[1])))
    action, error = follow(follower, goal_point)
    if action is None:
        # InternNav: "if action == STOP: ... action = LEFT  # random action to avoid deadlock"
        info.update(anti_deadlock=True, error=error, follower_actions=0, end="anti_deadlock")
        run.act(LEFT)
        return info
    executed, end = 0, None
    while True:
        if action is None:
            end = "follower_error" if error else "follower_stop"
            break
        if executed >= settings.spf_max_actions:
            end = "max_actions"
            break
        executed += 1
        if run.act(action):
            end = "rollout_over"
            break
        action, error = follow(follower, goal_point)
    info.update(anti_deadlock=False, error=error, follower_actions=executed, end=end)
    return info


def line_b_rollout(
    world: Any,
    geometry: Any,
    *,
    goal: Sequence[float],
    waypoints: np.ndarray,
    last_reached: int,
    previous_turn_direction: int,
    config: Any,
    camera: Any,
    settings: RolloutSettings,
) -> dict[str, Any]:
    """Line B answers every System2 call from the world's current pose (see the module doc)."""
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    run = _Run(world, geometry, goal, settings)
    remaining = ORL.remaining_reference_m(waypoints, geometry.geodesic)
    segments = [geometry.find_path(waypoints[k], waypoints[k + 1]) for k in range(len(waypoints) - 1)]
    follower = world.follower(settings.spf_goal_radius_m)
    best, seen = int(last_reached), 0
    prev_dir = int(np.sign(previous_turn_direction))
    calls: list[dict[str, Any]] = []
    while run.outcome is None:
        for moved in run.moved[seen:]:
            best = max(best, int(ORL.last_reached_by_step(moved[None], waypoints, geometry.geodesic,
                                                          config.reached_radius_m)[0]))
        seen = len(run.moved)
        position, rotation = world.pose()
        p = np.asarray(position, dtype=np.float64)
        here = int(ORL.last_reached_by_step(p[None], waypoints, geometry.geodesic, config.reached_radius_m)[0])
        last = max(best, here)
        k_star, _ = ORL.choose_waypoint(p, waypoints, last, remaining, geometry.geodesic, slack_m=config.rejoin_slack_m)
        route = ORL.route_points(p, waypoints, k_star, geometry.find_path, segments) if k_star is not None else None
        depth = world.lookdown_depth()
        oracle = RL.route_oracle(position=position, rotation_wxyz=rotation, route=route, depth_m=depth,
                                 previous_turn_direction=prev_dir, config=config, camera=camera)
        row: dict[str, Any] = {
            "call": len(calls), "step": run.steps, "d": ORL.rounded(run.d), "position": ORL.rounded(p),
            "yaw_deg": ORL.rounded(yaw_deg(rotation), 2), "last_reached": last, "k_star": k_star,
            "route_length_m": oracle.debug.get("route_length_m"), "oracle": oracle.kind, "reason": oracle.reason,
            "rule": oracle.debug.get("rule"), "text": [oracle.turn1, oracle.turn2],
            "pixel": list(oracle.pixel) if oracle.pixel is not None else None,
            "prev_direction": prev_dir, "direction": 0,
        }
        start = run.steps
        if oracle.kind == "pixel":
            row["spf"] = _pixel_goal(run, world, follower, oracle.pixel, position, rotation, depth, camera, settings)
            prev_dir = 0
        elif oracle.kind == "turn":
            direction, count = int(oracle.debug["turn_direction"]), int(oracle.debug["turn_count"])
            if RL.answer_direction(oracle.turn1, None) != direction or len(oracle.turn1 or "") != count:
                raise RuntimeError(f"line B's turn text {oracle.turn1!r} is not {count} x direction {direction}")
            row.update(direction=direction, count=count)
            for _ in range(count):
                if run.act(LEFT if direction > 0 else RIGHT):
                    break
            prev_dir = direction
        elif oracle.kind == "skip":
            run.act(FORWARD)
            prev_dir = 0
        else:
            raise RuntimeError(f"line B answered {oracle.kind!r}; it never answers STOP")
        row["actions"] = run.actions[start:]
        calls.append(row)
    pairs, flips = turn_flips(calls)
    pairs_any, flips_any = turn_flips_across(calls)
    first_vs_recorded = None
    if calls and calls[0]["oracle"] == "turn" and calls[0]["prev_direction"] != 0:
        first_vs_recorded = int(calls[0]["direction"] != calls[0]["prev_direction"])
    spf = [c["spf"] for c in calls if "spf" in c]
    return {
        **run.summary(),
        "num_calls": len(calls),
        "oracle_kinds": dict(collections.Counter(c["oracle"] for c in calls)),
        "turn_pairs": pairs,
        "turn_flips": flips,
        "turn_pairs_across_other_calls": pairs_any,
        "turn_flips_across_other_calls": flips_any,
        "first_call_flips_recorded_turn": first_vs_recorded,
        "pixel_goals": len(spf),
        "anti_deadlock": sum(1 for s in spf if s.get("anti_deadlock")),
        "follower_errors": sum(1 for s in spf if s.get("error")),
        "pixel_fallbacks": sum(1 for s in spf if s.get("fallback")),
        "calls": calls,
    }


def expert_rollout(world: Any, geometry: Any, *, goal: Sequence[float], settings: RolloutSettings) -> dict[str, Any]:
    """A fresh ShortestPathFollower to the episode goal from the world's current pose, to the same 3 m test."""
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    run = _Run(world, geometry, goal, settings)
    follower = world.follower(settings.spf_goal_radius_m)
    error = None
    while run.outcome is None:
        action, error = follow(follower, goal)
        if action is None:
            run.outcome = "follower_error" if error else "follower_stopped_outside_radius"
            break
        run.act(action)
    return {**run.summary(), "error": error}


def turn_flips(calls: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """(pairs of consecutive calls that both turn, how many of them turn opposite ways)."""
    pairs = flips = 0
    for a, b in zip(calls, calls[1:]):
        if a["oracle"] == "turn" and b["oracle"] == "turn":
            pairs += 1
            flips += int(a["direction"] != b["direction"])
    return pairs, flips


def turn_flips_across(calls: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Report-only: every turn call against the rollout's previous turn call, other calls in between or not."""
    pairs = flips = 0
    previous = 0
    for call in calls:
        if call["oracle"] != "turn":
            continue
        if previous:
            pairs += 1
            flips += int(call["direction"] != previous)
        previous = call["direction"]
    return pairs, flips


def depth_agreement(rendered: np.ndarray, recorded: np.ndarray, tol_m: float = DEPTH_MATCH_TOL_M) -> dict[str, Any]:
    """How far the lookdown rendered here is from the one the record holds (metres, 0 = no return in both)."""
    a = np.asarray(rendered, dtype=np.float64)
    b = np.asarray(recorded, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if b.ndim == 3:
        b = b[..., 0]
    if a.shape != b.shape:
        return {"match": 0.0, "shape": [list(a.shape), list(b.shape)]}
    diff = np.abs(a - b)
    return {"match": round(float((diff <= tol_m).mean()), 6), "mean_abs_m": round(float(diff.mean()), 4),
            "p99_abs_m": round(float(np.percentile(diff, 99)), 4)}


def rollout_state(
    world: Any,
    geometry: Any,
    state: Mapping[str, Any],
    record: Any,
    *,
    dataset_entry: Mapping[str, Any] | None,
    config: Any,
    camera: Any,
    settings: RolloutSettings,
) -> dict[str, Any]:
    """Restore the state, roll out line B, restore again, roll out the SPF expert; one row.

    A state restored at a call pose first renders the lookdown there and compares it with the record's own
    lookdown depth of that call (``restored_depth``): the report refuses a verdict when the world K1 rolls out
    in is not the one the records were collected in (another mesh, a pose or camera convention off).
    """
    ref, goal, _ = base.resolve_episode_geometry(record.episode, dataset_entry)
    waypoints = ORL.route_waypoints(ref, goal)
    if len(waypoints) != int(state["num_waypoints"]):
        raise K1Invalid(f"{state['key']}: {len(waypoints)} waypoints, the selection saw {state['num_waypoints']}")
    restore = state["restore"]
    started = time.time()
    world.set_pose(restore["position"], restore["rotation_wxyz"])
    d = float(geometry.geodesic(np.asarray(world.pose()[0], dtype=np.float64), goal))
    if not abs(d - float(state["distance_to_goal"])) <= 1e-3:
        raise K1Invalid(f"{state['key']}: geodesic {d} at the restored pose, "
                        f"the selection had {state['distance_to_goal']}")
    call_index = restore.get("call_index")
    restored_depth = None
    if call_index is not None and record.has_lookdown_depth(int(call_index)):
        restored_depth = depth_agreement(world.lookdown_depth(), record.lookdown_depth(int(call_index)))
    line_b = line_b_rollout(world, geometry, goal=goal, waypoints=waypoints, last_reached=int(state["last_reached"]),
                            previous_turn_direction=int(state["previous_turn_direction"]), config=config,
                            camera=camera, settings=settings)
    world.set_pose(restore["position"], restore["rotation_wxyz"])
    expert = expert_rollout(world, geometry, goal=goal, settings=settings)
    ratio = None
    if line_b["reached"] and expert["reached"] and expert["steps"] > 0:
        ratio = round(line_b["steps"] / expert["steps"], 6)
    budget = EPISODE_STEP_CAP - int(restore["step_id"])
    return {
        "schema": ROLLOUT_SCHEMA,
        "key": state["key"],
        "scene_id": state["scene_id"],
        "episode_id": int(state["episode_id"]),
        "sources": list(state["sources"]),
        "restore_step_id": int(restore["step_id"]),
        "restore_call_index": None if call_index is None else int(call_index),
        "restored_depth": restored_depth,
        "distance_to_goal": ORL.rounded(state["distance_to_goal"]),
        "off_route_m": ORL.rounded(state["off_route_m"]),
        "line_b": line_b,
        "expert": expert,
        "ratio": ratio,
        "within_episode_budget": bool(line_b["reached"] and line_b["steps"] <= budget),
        "seconds": round(time.time() - started, 2),
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def median(values: Sequence[Fraction | float]) -> Fraction | float | None:
    """Exact median of Fractions (``math.inf`` allowed); None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    a, b = ordered[n // 2 - 1], ordered[n // 2]
    if math.isinf(float(a)) or math.isinf(float(b)):
        return math.inf
    return (Fraction(a) + Fraction(b)) / 2


def _criterion(value: Any, threshold: Fraction, direction: str, ok: bool, **extra: Any) -> dict[str, Any]:
    return {"value": _num(value), "threshold": _num(threshold), "direction": direction, "pass": bool(ok), **extra}


def _outcome_signature(row: Mapping[str, Any]) -> str:
    """What a rollout measured (not when or how long it took): two rows of one state must agree on it."""
    return json.dumps({k: row.get(k) for k in ("line_b", "expert", "ratio", "restored_depth")}, sort_keys=True)


def _ratio_values(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Fraction | float], int]:
    values, expert_failed = [], 0
    for row in rows:
        expert, line_b = row["expert"], row["line_b"]
        if not expert["reached"] or int(expert["steps"]) <= 0:
            expert_failed += 1
            continue
        values.append(Fraction(int(line_b["steps"]), int(expert["steps"])) if line_b["reached"] else math.inf)
    return values, expert_failed


def _block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reach, flips and the ratio of a set of rows (report-only breakdowns use it too)."""
    n = len(rows)
    reached = sum(1 for r in rows if r["line_b"]["reached"])
    pairs = sum(int(r["line_b"]["turn_pairs"]) for r in rows)
    flips = sum(int(r["line_b"]["turn_flips"]) for r in rows)
    values, expert_failed = _ratio_values(rows)
    med = median(values)
    return {"n": n, "reached": reached, "reach_rate": _rate(reached, n), "turn_pairs": pairs, "turn_flips": flips,
            "flip_rate": _rate(flips, pairs), "ratio_n": len(values), "expert_failed": expert_failed,
            "ratio_median": _num(med)}


def summarize(states: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
              *, states_sha256: str | None = None, config_digest: str | None = None) -> dict[str, Any]:
    """K1's three criteria over the selected states, validity, and the report-only breakdowns."""
    problems: list[str] = []
    notes: list[str] = []
    if int(selection.get("episodes_scanned", 0)) < MIN_EPISODES:
        problems.append(f"{selection.get('episodes_scanned')} episodes scanned, K1 needs >= {MIN_EPISODES}")
    if len(states) < MIN_STATES:
        problems.append(f"{len(states)} states selected, K1 needs >= {MIN_STATES}")
    by_key: dict[str, Mapping[str, Any]] = {}
    duplicates, conflicting = 0, []
    for row in rows:
        if row["key"] in by_key:
            duplicates += 1
            if _outcome_signature(row) != _outcome_signature(by_key[row["key"]]):
                conflicting.append(row["key"])
            continue
        by_key[row["key"]] = row
    if duplicates:
        notes.append(f"{duplicates} duplicate rows (a state rolled out twice) ignored; the first file's row counts")
    if conflicting:
        problems.append(f"{len(conflicting)} states were rolled out twice with different results, so the rollout "
                        f"is not deterministic or the inputs changed (first: {conflicting[:5]})")
    keys = [s["key"] for s in states]
    missing = [k for k in keys if k not in by_key]
    extra = sorted(set(by_key) - set(keys))
    if missing:
        problems.append(f"{len(missing)} selected states have no rollout row (first: {missing[:5]})")
    if extra:
        problems.append(f"{len(extra)} rows are not of this selection (first: {extra[:5]})")
    measured = [by_key[k] for k in keys if k in by_key]
    if states_sha256 is not None:
        foreign = sorted({str(r.get("selection_sha256")) for r in measured} - {states_sha256})
        if foreign:
            problems.append(f"rows of another states file: {foreign}")
    if config_digest is not None:
        foreign = sorted({str(r.get("config_digest")) for r in measured} - {config_digest})
        if foreign:
            problems.append(f"rows of another line-B config: {foreign} (RouteLabelConfig() is {config_digest})")
        selected_with = (selection.get("config") or {}).get("digest")
        if selected_with is not None and selected_with != config_digest:
            problems.append(f"the states were selected under line-B config {selected_with}, the code's is "
                            f"{config_digest}")
    settings = sorted({json.dumps([r.get("settings"), r.get("step_mode")], sort_keys=True) for r in measured})
    if len(settings) > 1:
        problems.append(f"rows of different rollout settings / step modes: {settings}")

    # The world K1 rolled out in must be the one the records were collected in.
    depth_rows = [r for r in measured if r.get("restore_call_index") is not None]
    depth_checked = [r for r in depth_rows if r.get("restored_depth")]
    depth_bad = [r["key"] for r in depth_checked if float(r["restored_depth"]["match"]) < DEPTH_MATCH_MIN]
    if len(depth_checked) < len(depth_rows):
        problems.append(f"{len(depth_rows) - len(depth_checked)} states restored at a call have no rendered-vs-"
                        f"recorded lookdown check (records without lookdown depth?)")
    if depth_checked and Fraction(len(depth_bad), len(depth_checked)) > DEPTH_MISMATCH_MAX:
        problems.append(f"at {len(depth_bad)} of {len(depth_checked)} restored call poses the rendered lookdown "
                        f"depth is not the record's (< {DEPTH_MATCH_MIN} of the pixels within "
                        f"{DEPTH_MATCH_TOL_M} m; first: {depth_bad[:5]}): K1 is not rolling out in the world the "
                        f"records were collected in")

    block = _block(measured)
    n, reached = block["n"], block["reached"]
    pairs, flips = block["turn_pairs"], block["turn_flips"]
    values, expert_failed = _ratio_values(measured)
    med = median(values)
    if pairs == 0:
        notes.append("no two consecutive line-B calls both turned: the flip criterion holds vacuously")
    if not values and measured:
        problems.append("the SPF expert entered the circle from no state: the step ratio cannot be measured")
    elif measured and Fraction(expert_failed, len(measured)) > EXPERT_FAILED_MAX:
        problems.append(f"the SPF expert, the ratio's reference, failed from {expert_failed} of {len(measured)} "
                        f"states (> {float(EXPERT_FAILED_MAX):g}): check the follower before reading K1")
    criteria = {
        "reach": _criterion(Fraction(reached, n) if n else None, REACH_MIN, ">=",
                            bool(n) and Fraction(reached, n) >= REACH_MIN, k=reached, n=n),
        "flip": _criterion(Fraction(flips, pairs) if pairs else Fraction(0), FLIP_MAX, "<=",
                           pairs == 0 or Fraction(flips, pairs) <= FLIP_MAX, k=flips, n=pairs),
        "ratio_median": _criterion(med, RATIO_MEDIAN_MAX, "<=", med is not None and med <= RATIO_MEDIAN_MAX,
                                   n=len(values), expert_failed=expert_failed,
                                   line_b_failed=sum(1 for v in values if math.isinf(float(v)))),
    }
    valid = not problems
    all_pass = all(c["pass"] for c in criteria.values())
    verdict = "invalid" if not valid else ("pass" if all_pass else "fail")

    def q(vals: Iterable[float | None]) -> Any:
        return lr.quantiles([v for v in vals if v is not None], (5, 25, 50, 75, 95))

    both = [r for r in measured if r["ratio"] is not None]
    by_source = {}
    for source in sorted({s for r in measured for s in r["sources"]}):
        by_source[source] = _block([r for r in measured if source in r["sources"]])
    calls = [c for r in measured for c in r["line_b"]["calls"]]
    spf = [c["spf"] for c in calls if "spf" in c]
    first_flips = [r["line_b"]["first_call_flips_recorded_turn"] for r in measured
                   if r["line_b"]["first_call_flips_recorded_turn"] is not None]
    worst = sorted(measured, key=lambda r: (r["line_b"]["reached"], -(r["ratio"] or 0.0)))[:10]
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "valid": valid,
        "problems": problems,
        "notes": notes,
        "criteria": criteria,
        "all_pass": all_pass,
        "rule": "ledger EXP-28 C / K1, written 2026-10-15 19:35 UTC before any HM3D result; on failure fix bugs "
                "only (never rule parameters) and test once more; a second failure: C1 is not trained",
        "selection": {k: selection.get(k) for k in ("episodes_scanned", "episodes_with_states", "states", "stopped",
                                                    "states_by_source", "missing_records", "record_policies",
                                                    "settings")},
        "report_only": {
            "by_source": by_source,
            "line_b_outcomes": dict(collections.Counter(r["line_b"]["outcome"] for r in measured)),
            "expert_outcomes": dict(collections.Counter(r["expert"]["outcome"] for r in measured)),
            "line_b_steps": q(r["line_b"]["steps"] for r in measured if r["line_b"]["reached"]),
            "expert_steps": q(r["expert"]["steps"] for r in measured if r["expert"]["reached"]),
            "ratio_both_reached": q(r["ratio"] for r in both),
            "calls_per_rollout": q(r["line_b"]["num_calls"] for r in measured),
            "oracle_kinds": dict(collections.Counter(c["oracle"] for c in calls)),
            "oracle_rules": dict(collections.Counter(str(c.get("rule")) for c in calls)),
            # no_route / no_depth skips walk forward blind: a share of them means the route was not rebuilt
            "oracle_reasons": dict(collections.Counter(str(c.get("reason")) for c in calls)),
            "restored_depth": {"checked": len(depth_checked), "below_min": len(depth_bad),
                               "match": q(r["restored_depth"]["match"] for r in depth_checked),
                               "mean_abs_m": q(r["restored_depth"].get("mean_abs_m") for r in depth_checked)},
            "turn_flips_across_other_calls": {
                "k": sum(int(r["line_b"]["turn_flips_across_other_calls"]) for r in measured),
                "n": sum(int(r["line_b"]["turn_pairs_across_other_calls"]) for r in measured)},
            "first_call_vs_recorded_turn": {"flips": sum(first_flips), "n": len(first_flips)},
            "pixel_goals": {
                "n": len(spf),
                "anti_deadlock": sum(1 for s in spf if s.get("anti_deadlock")),
                "follower_errors": sum(1 for s in spf if s.get("error")),
                "fallbacks": dict(collections.Counter(s["fallback"] for s in spf if s.get("fallback"))),
                "end": dict(collections.Counter(str(s.get("end")) for s in spf if not s.get("fallback"))),
                "snap_m": q(s.get("snap_m") for s in spf),
                "goal_dy_gt_0.3m": sum(1 for s in spf if s.get("goal_dy_m") is not None and abs(s["goal_dy_m"]) > 0.3),
            },
            "within_episode_budget": {"k": sum(1 for r in measured if r["within_episode_budget"]), "n": n,
                                      "cap": EPISODE_STEP_CAP},
            "worst": [{"key": r["key"], "outcome": r["line_b"]["outcome"], "steps": r["line_b"]["steps"],
                       "expert_steps": r["expert"]["steps"], "ratio": r["ratio"], "final_d": r["line_b"]["final_d"],
                       "sources": r["sources"]} for r in worst],
        },
    }


# --------------------------------------------------------------------------- #
# Habitat
# --------------------------------------------------------------------------- #
def check_client_camera(cfg: Any) -> dict[str, Any]:
    """The client's DEPTH_SENSOR and actions must be the recorded deployment camera (s2_records.CAMERA)."""
    sim = cfg.SIMULATOR
    depth = sim.DEPTH_SENSOR
    spec = {
        "width": int(depth.WIDTH), "height": int(depth.HEIGHT), "hfov_deg": float(depth.HFOV),
        "position": [float(v) for v in depth.POSITION], "min_depth": float(depth.MIN_DEPTH),
        "max_depth": float(depth.MAX_DEPTH), "normalize_depth": bool(depth.NORMALIZE_DEPTH),
        "tilt_deg": float(sim.TILT_ANGLE), "turn_deg": float(sim.TURN_ANGLE),
        "forward_m": float(sim.FORWARD_STEP_SIZE), "action_space": str(sim.ACTION_SPACE_CONFIG),
    }
    camera = REC.CAMERA
    expected = {
        "width": int(camera["width"]), "height": int(camera["height"]), "hfov_deg": float(camera["hfov_deg"]),
        "position": [0.0, float(camera["height_m"]), 0.0], "min_depth": 0.0, "max_depth": float(camera["depth_max_m"]),
        "normalize_depth": True, "tilt_deg": float(camera["lookdown_pitch_deg"]) / 2.0, "turn_deg": 15.0,
        "forward_m": 0.25, "action_space": "v1",
    }
    problems = {k: (spec[k], expected[k]) for k in expected if spec[k] != expected[k]}
    if problems:
        raise K1Invalid(f"the client's Habitat config is not the deployment camera: {problems}")
    return spec


class HabitatWorld:
    """habitat-lab ``Sim-v0`` from the client's own config, DEPTH_SENSOR only; one per process.

    Movement uses the agent's own action specs (``agent.act``: habitat-sim's controls with the navmesh step
    filter, as ``Simulator.step`` applies them, without rendering); ``step_mode="sim_step"`` uses
    ``sim.step`` instead (renders every action, same motion).
    """

    def __init__(self, client: Any, *, sim_gpu_id: int = 0, step_mode: str = "act"):
        if step_mode not in ("act", "sim_step"):
            raise ValueError(f"step_mode {step_mode!r}")
        self.client = client
        self.sim_gpu_id = int(sim_gpu_id)
        self.step_mode = step_mode
        self.camera = ORL.LookdownCamera()
        self.sim = None
        self.cfg = None
        self.max_lookdown_pose_err = 0.0
        self.renders = 0

    def load_scene(self, glb: Path, *, expected_navmesh_sha256: str | None = None) -> dict[str, Any]:
        import habitat
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        started = time.time()
        actions = {name: int(getattr(HabitatSimActions, name)) for name in HABITAT_ACTIONS}
        if actions != HABITAT_ACTIONS:
            raise K1Invalid(f"HabitatSimActions {actions} are not the client's ActionCode {HABITAT_ACTIONS}")
        cfg = v0.client_habitat_config(self.client, glb, self.sim_gpu_id)
        cfg.defrost()
        cfg.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
        cfg.freeze()
        spec = check_client_camera(cfg)
        if self.sim is None:
            self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        else:
            self.sim.reconfigure(cfg.SIMULATOR)
        self.cfg = cfg
        navmesh = Path(os.path.splitext(str(glb))[0] + ".navmesh")
        # Loaded explicitly and its return value checked, as verify_hm3d_render.ClientSimRenderer.load: 0.1.7
        # does not check it, and a failed load can leave the previous scene's navmesh in place.  No in-process
        # rebake either: the records were collected on this file's navmesh.
        returned = self.sim.pathfinder.load_nav_mesh(str(navmesh)) if navmesh.is_file() else False
        loaded = (returned if isinstance(returned, bool) else navmesh.is_file()) and self.sim.pathfinder.is_loaded
        if not loaded:
            raise K1Invalid(f"{navmesh} did not load for {glb} (load_nav_mesh -> {returned}; prepare_hm3d_scenes.py)")
        digest = sha256_file(navmesh)
        if expected_navmesh_sha256 is not None and digest != expected_navmesh_sha256:
            raise K1Invalid(f"{navmesh} is {digest}; the records were collected on {expected_navmesh_sha256}")
        agent = cfg.SIMULATOR.AGENT_0
        return {"glb": str(glb), "glb_realpath": os.path.realpath(glb), "navmesh": str(navmesh),
                "navmesh_sha256": digest, "camera": spec, "agent_radius": float(agent.RADIUS),
                "agent_height": float(agent.HEIGHT), "load_s": round(time.time() - started, 2)}

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        import quaternion

        state = self.sim.get_agent_state()
        return (np.asarray(state.position, dtype=np.float64),
                np.asarray(quaternion.as_float_array(state.rotation), dtype=np.float64))

    def set_pose(self, position: Sequence[float], rotation_wxyz: Sequence[float]) -> None:
        """The body at ``position`` / ``rotation_wxyz``, sensors level (``reset_sensors=True``)."""
        import quaternion  # noqa: F401  (registers np.quaternion)

        q = np.quaternion(*[float(v) for v in rotation_wxyz])
        self.sim.set_agent_state(np.asarray(position, dtype=np.float32), q, reset_sensors=True)
        got_p, got_q = self.pose()
        err = max(float(np.abs(got_p - np.asarray(position, dtype=np.float64)).max()), _rot_err(got_q, rotation_wxyz))
        if err > POSE_TOL:
            raise K1Invalid(f"the agent did not take the requested pose (max error {err:g})")

    def act(self, action: int) -> None:
        if self.step_mode == "sim_step":
            self.sim.step(int(action))
            self.renders += 1
        else:
            self.sim.get_agent(0).act(int(action))

    def lookdown_depth(self) -> np.ndarray:
        """The client's lookdown depth at the current pose: 2 x LOOK_DOWN, metres in [0, 10], the record's mm."""
        position, rotation = self.pose()
        self.set_pose(position, rotation)
        agent = self.sim.get_agent(0)
        agent.act(LOOK_DOWN)
        agent.act(LOOK_DOWN)
        sensor = agent.get_state().sensor_states["depth"]
        expected = self.camera.c2w(position, rotation)
        actual_q = [sensor.rotation.w, sensor.rotation.x, sensor.rotation.y, sensor.rotation.z]
        err = max(float(np.abs(np.asarray(sensor.position, dtype=np.float64) - expected[:3, 3]).max()),
                  float(np.abs(ORL.quat_wxyz_to_rot(actual_q) - expected[:3, :3]).max()))
        self.max_lookdown_pose_err = max(self.max_lookdown_pose_err, err)
        if err > LOOKDOWN_POSE_TOL:
            raise K1Invalid(f"the lookdown depth sensor is {err:g} off LookdownCamera's pose")
        raw = np.asarray(self.sim.get_sensor_observations()["depth"], dtype=np.float64)
        self.renders += 1
        if raw.ndim == 3:
            raw = raw[..., 0]
        sensor_cfg = self.cfg.SIMULATOR.DEPTH_SENSOR
        depth = np.clip(raw, float(sensor_cfg.MIN_DEPTH), float(sensor_cfg.MAX_DEPTH))  # habitat-lab's depth sensor
        self.set_pose(position, rotation)
        return REC.depth_from_png_bytes(REC.depth_to_png_bytes(depth))

    def follower(self, goal_radius: float) -> Callable[[np.ndarray], int | None]:
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        try:
            spf = ShortestPathFollower(self.sim, goal_radius=float(goal_radius), return_one_hot=False,
                                       stop_on_error=False)
        except TypeError:  # a habitat-lab without stop_on_error
            spf = ShortestPathFollower(self.sim, float(goal_radius), False)

        def next_action(goal: np.ndarray) -> int | None:
            action = spf.get_next_action(np.asarray(goal, dtype=np.float32))
            if action is None:
                return None
            return int(np.asarray(action).reshape(-1)[0])

        return next_action

    def is_navigable(self, point: Sequence[float]) -> bool:
        return bool(self.sim.pathfinder.is_navigable(np.asarray(point, dtype=np.float32)))

    def snap(self, point: Sequence[float]) -> np.ndarray | None:
        snapped = np.asarray(self.sim.pathfinder.snap_point(np.asarray(point, dtype=np.float32)), dtype=np.float64)
        return snapped if snapped.shape == (3,) and np.all(np.isfinite(snapped)) else None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def navmesh_geometry(path: str | os.PathLike) -> Any:
    """The labeller's geometry (a bare habitat_sim.PathFinder on the navmesh file)."""
    return lr.RouteNavmeshGeometry(path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_selection(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    states_path, selection_path = out_dir / STATES_NAME, out_dir / SELECTION_NAME
    if not states_path.is_file() or not selection_path.is_file():
        raise K1Invalid(f"{out_dir} has no {STATES_NAME} / {SELECTION_NAME}: run select first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    digest = sha256_file(states_path)
    if selection.get("states_sha256") != digest:
        raise K1Invalid(f"{states_path} changed since select ({digest} != {selection.get('states_sha256')})")
    return read_jsonl(states_path), selection, digest


def rows_path_for(out_dir: Path, index: int, count: int) -> Path:
    return out_dir / f"rollouts.shard{index:02d}of{count:02d}.jsonl"


def rollout_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("rollouts.shard*of*.jsonl"))


def done_keys(out_dir: Path, rows_path: Path, digest: str) -> set[str]:
    """States with a row in any rollouts file (a smoke run's ``--shard 0/1`` too): not rolled out twice.

    This shard's own file loses an unterminated last row (a kill mid-write), so that state is rolled out
    again; another shard's unterminated row is that shard still writing and is just not counted.
    """
    done: set[str] = set()
    for path in rollout_files(out_dir):
        rows, tail = read_rows(path)
        if tail and path == rows_path:
            with open(path, "r+b") as handle:
                handle.truncate(path.stat().st_size - tail)
            print(f"[k1] {path}: dropped an unterminated last row ({tail} bytes)", flush=True)
        for row in rows:
            if row.get("selection_sha256") != digest:
                raise K1Invalid(f"{path} holds rows of another selection; move it away")
            done.add(row["key"])
    return done


def run_inputs(args: argparse.Namespace, selection: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``run`` reads the scenes and episode files ``select`` read; a different one given is refused."""
    inputs = selection.get("inputs") or {}
    selected_scenes, selected_episodes = inputs.get("scenes_dir"), list(inputs.get("episodes") or [])
    scenes_dir = str(Path(args.scenes_dir).resolve()) if args.scenes_dir else selected_scenes
    if not scenes_dir:
        raise K1Invalid("the selection names no scenes dir: pass --scenes-dir")
    if selected_scenes and scenes_dir != selected_scenes:
        raise K1Invalid(f"--scenes-dir {scenes_dir} is not the selection's {selected_scenes}")
    episodes = [str(Path(p).resolve()) for p in args.episodes] if args.episodes else selected_episodes
    if episodes != selected_episodes:
        raise K1Invalid(f"--episodes {episodes} are not the selection's {selected_episodes}")
    return scenes_dir, episodes


def run_states(
    todo: Sequence[Mapping[str, Any]],
    *,
    world: Any,
    scenes_dir: str | os.PathLike,
    geometry_for_navmesh: Callable[[Path], Any],
    dataset_index: Mapping[tuple[str, int], Mapping[str, Any]] | None,
    config: Any,
    camera: Any,
    settings: RolloutSettings,
    rows_path: Path,
    errors_path: Path,
    stamp: Mapping[str, Any],
    log_every: int = 10,
) -> collections.Counter:
    """Roll out ``todo`` scene by scene, appending a row per state; a state that raises is logged and skipped."""
    counts: collections.Counter = collections.Counter()
    by_scene: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for state in todo:
        by_scene[str(state["scene_id"])].append(state)
    started = time.time()
    for scene in sorted(by_scene):
        expected = {s.get("navmesh_sha256") for s in by_scene[scene]}
        if len(expected) != 1 or None in expected:
            raise K1Invalid(f"{scene}: the states name navmeshes {sorted(map(str, expected))}; each scene's records "
                            "must name the one navmesh they were collected on")
        glb = Path(scenes_dir) / scene / f"{scene}.glb"
        info = world.load_scene(glb, expected_navmesh_sha256=next(iter(expected)))
        geometry = geometry_for_navmesh(Path(scenes_dir) / scene / f"{scene}.navmesh")
        print(f"[k1] scene {scene}: {len(by_scene[scene])} states, loaded in {info.get('load_s')} s", flush=True)
        for state in by_scene[scene]:
            try:
                record = REC.EpisodeRecord(state["record_path"])
                if base._sidecar_digest(Path(state["record_path"])) != state["record_sha256"]:
                    raise K1Invalid(f"{state['record_path']} changed since select")
                entry = None
                if dataset_index is not None:
                    entry = dataset_index.get((record.scene_id, record.episode_id))
                    if entry is None:
                        raise K1Invalid(f"{record.scene_id}/{record.episode_id} is not in --episodes")
                row = rollout_state(world, geometry, state, record, dataset_entry=entry, config=config,
                                    camera=camera, settings=settings)
            except K1Invalid:
                raise
            except Exception as err:  # noqa: BLE001 -- one bad state must not stop the shard; report sees it missing
                counts["errors"] += 1
                append_jsonl(errors_path, {"key": state["key"], "error": f"{type(err).__name__}: {err}",
                                           "traceback": traceback.format_exc(limit=6)})
                print(f"[k1] ERROR {state['key']}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                continue
            row.update(stamp)
            row["lookdown_pose_err_max"] = ORL.rounded(getattr(world, "max_lookdown_pose_err", 0.0), 8)
            append_jsonl(rows_path, row)
            counts["done"] += 1
            counts["reached"] += int(row["line_b"]["reached"])
            depth_match = (row.get("restored_depth") or {}).get("match")
            print(f"[k1] {state['key']}: line B {row['line_b']['outcome']} in {row['line_b']['steps']}, expert "
                  f"{row['expert']['outcome']} in {row['expert']['steps']}, restored lookdown match {depth_match}",
                  flush=True)
            if log_every and counts["done"] % log_every == 0:
                print(f"[k1] {counts['done']}/{len(todo)} states, line B reached {counts['reached']}, "
                      f"{time.time() - started:.0f} s", flush=True)
    return counts


def cmd_select(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if (out_dir / STATES_NAME).exists() and not args.force:
        print(f"{out_dir / STATES_NAME} exists; the states must stay fixed across shards (--force to redo)", flush=True)
        return 2
    started = time.time()
    config, camera = RL.RouteLabelConfig(), ORL.LookdownCamera()
    geometries: dict[str, tuple[Any, str]] = {}

    def geometry_for_scene(scene: str) -> tuple[Any, str]:
        if scene not in geometries:
            path = Path(args.scenes_dir) / scene / f"{scene}.navmesh"
            geometries[scene] = (navmesh_geometry(path), sha256_file(path))
        return geometries[scene]

    record_roots = [str(Path(root).resolve()) for root in args.record_root]  # run reads the records by path
    episode_files = [str(Path(p).resolve()) for p in args.episodes or []]
    # build_hm3d_episodes.py writes <out>/cohorts and <out>/splits/heldout_scenes.txt side by side.
    holdout_file = Path(args.holdout_scenes_file or Path(args.cohort_dir).resolve().parent / "splits"
                        / "heldout_scenes.txt")
    try:
        if not holdout_file.is_file():
            raise K1Invalid(f"no held-out scene list at {holdout_file}: pass --holdout-scenes-file")
        episodes, cohort = cohort_episodes(args.cohort_dir, args.parts or None)
        holdout = read_scene_list(holdout_file)
        if not holdout:
            raise K1Invalid(f"{holdout_file} lists no scene")
        dataset_index = base.load_episode_index(episode_files) if episode_files else None
        states, meta = select_states(
            episodes, record_roots=record_roots, geometry_for_scene=geometry_for_scene, config=config,
            dataset_index=dataset_index, camera=camera, min_episodes=args.min_episodes, min_states=args.min_states,
            max_states_per_episode=args.max_states_per_episode, max_episodes=args.max_episodes, holdout=holdout,
        )
    except (K1Invalid, RuntimeError, ValueError, KeyError, OSError) as err:
        print(f"INVALID K1 selection: {type(err).__name__}: {err}", flush=True)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    states_path = out_dir / STATES_NAME
    tmp = states_path.with_name(f".{states_path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(s, sort_keys=True, ensure_ascii=False) + "\n" for s in states), encoding="utf-8")
    os.replace(tmp, states_path)
    selection = {
        "schema": SCHEMA,
        **meta,
        "states_sha256": sha256_file(states_path),
        "cohort": cohort,
        "inputs": {"record_roots": record_roots, "scenes_dir": str(Path(args.scenes_dir).resolve()),
                   "episodes": episode_files, "holdout_scenes_file": str(holdout_file.resolve()),
                   "holdout_scenes": len(holdout)},
        "config": {**dataclasses.asdict(config), "digest": config.digest(), "rules_version": RL.ROUTE_RULES_VERSION},
        "git_head": v0.git_head(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
    }
    _write_json(out_dir / SELECTION_NAME, selection)
    print(json.dumps({"episodes_scanned": meta["episodes_scanned"], "states": meta["states"],
                      "by_source": meta["states_by_source"], "scenes": len(meta["scenes"]),
                      "enough": meta["enough"], "stopped": meta["stopped"],
                      # reference segments without a navmesh path (off-route distances there are to a chord)
                      "reference_segments_straight": meta["stats"].get("reference_segments_straight", 0),
                      "record_policies": meta["record_policies"]}), flush=True)
    return 0 if meta["enough"] else 2


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    try:
        states, selection, digest = load_selection(out_dir)
        scenes_dir, episode_files = run_inputs(args, selection)
    except K1Invalid as err:
        print(f"INVALID K1 run: {err}", flush=True)
        return 2
    index, count = parse_shard(args.shard)
    rows_path = rows_path_for(out_dir, index, count)
    errors_path = out_dir / f"errors.shard{index:02d}of{count:02d}.jsonl"
    try:
        done = done_keys(out_dir, rows_path, digest)
    except K1Invalid as err:
        print(f"INVALID K1 run: {err}", flush=True)
        return 2
    mine = [s for s in states if shard_of(s["scene_id"], count) == index]
    todo = [s for s in mine if s["key"] not in done]
    before = len(mine) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[k1] shard {index}/{count}: {len(mine)} states, {before} done before, {len(todo)} to roll out", flush=True)
    if not todo:
        return 0
    settings = RolloutSettings(max_steps=args.max_steps, spf_max_actions=args.spf_max_actions)
    config, camera = RL.RouteLabelConfig(), ORL.LookdownCamera()
    if args.xvfb_display is not None:
        v0.start_xvfb(args.xvfb_display, args.xvfb_bin, out_dir / "logs" / f"xvfb_{args.xvfb_display}.log")
    elif not os.environ.get("DISPLAY"):
        print("INVALID K1 run: no DISPLAY; pass --xvfb-display N", flush=True)
        return 2
    # The client pre-initialises GL on an MP3D scene of the tree (<scenes-dir>/../mp3d), as the collection does.
    world = HabitatWorld(v0.import_client(Path(scenes_dir).parent), sim_gpu_id=args.sim_gpu_id,
                         step_mode=args.step_mode)
    stamp = {"selection_sha256": digest, "config_digest": config.digest(), "settings": dataclasses.asdict(settings),
             "step_mode": args.step_mode}
    try:
        dataset_index = base.load_episode_index(episode_files) if episode_files else None
        counts = run_states(todo, world=world, scenes_dir=scenes_dir, geometry_for_navmesh=navmesh_geometry,
                            dataset_index=dataset_index, config=config, camera=camera, settings=settings,
                            rows_path=rows_path, errors_path=errors_path, stamp=stamp, log_every=args.log_every)
    except K1Invalid as err:
        print(f"INVALID K1 run: {err}", flush=True)
        return 2
    finally:
        world.close()
    print(json.dumps({"shard": f"{index}/{count}", **dict(counts), "renders": world.renders,
                      "lookdown_pose_err_max": world.max_lookdown_pose_err, "versions": v0.versions(),
                      "rows": str(rows_path)}), flush=True)
    return 1 if counts["errors"] else 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    output = Path(args.output) if args.output else out_dir / SUMMARY_NAME
    try:
        states, selection, digest = load_selection(out_dir)
    except K1Invalid as err:
        print(f"INVALID K1: {err}", flush=True)
        return 2
    paths = rollout_files(out_dir)
    rows, partial = [], []
    for path in paths:
        got, tail = read_rows(path)
        rows.extend(got)
        if tail:
            partial.append(str(path))
    errors = [row for path in sorted(out_dir.glob("errors.shard*of*.jsonl")) for row in read_jsonl(path)]
    summary = summarize(states, rows, selection, states_sha256=digest, config_digest=RL.RouteLabelConfig().digest())
    if partial:  # a shard still running or killed mid-row
        summary["problems"].append(f"rollout files end in an unterminated row: {partial}")
        summary.update(valid=False, verdict="invalid")
    summary["inputs"] = {"out_dir": str(out_dir), "rollout_files": [str(p) for p in paths],
                         "logged_errors": len(errors), "states_sha256": digest}
    _write_json(output, summary)
    c = summary["criteria"]
    parts = [f"{name}={crit['value']}{'' if crit['pass'] else '(FAIL)'}" for name, crit in c.items()]
    head = {"pass": "PASS", "fail": "FAIL", "invalid": "INVALID"}[summary["verdict"]]
    print(f"{head} K1 " + " ".join(parts) + f" -> {output}", flush=True)
    for problem in summary["problems"]:
        print(f"[k1] {problem}", file=sys.stderr)
    return {"pass": 0, "fail": 1, "invalid": 2}[summary["verdict"]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="pick the off-route states (CPU, habitat_sim PathFinder)")
    sel.add_argument("--cohort-dir", required=True, help="build_hm3d_episodes.py's cohorts/ (manifest.json)")
    sel.add_argument("--record-root", action="append", required=True, help="committed base records (repeatable)")
    sel.add_argument("--scenes-dir", default="/workspace/exp28/scene_datasets/hm3d", help="<scene>/<scene>.navmesh")
    sel.add_argument("--episodes", nargs="*", default=[], help="R2R-CE-format pool json.gz (authoritative path/goal)")
    sel.add_argument("--out-dir", required=True)
    sel.add_argument("--parts", nargs="*", default=[], help="cohort parts (default: every train* part)")
    sel.add_argument("--holdout-scenes-file", default="",
                     help="held-out scene dirs; any of them in the parts is an error "
                          "(default: <cohort-dir>/../splits/heldout_scenes.txt, which must exist)")
    sel.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    sel.add_argument("--min-states", type=int, default=MIN_STATES)
    sel.add_argument("--max-states-per-episode", type=int, default=MAX_STATES_PER_EPISODE)
    sel.add_argument("--max-episodes", type=int, default=0, help="stop after this many records (0 = no cap)")
    sel.add_argument("--force", action="store_true", help="replace an existing selection")

    run = sub.add_parser("run", help="roll out line B and the SPF expert (renders)")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--scenes-dir", default="",
                     help="<scene>/<scene>.glb and .navmesh (default and only allowed value: select's)")
    run.add_argument("--episodes", nargs="*", default=[], help="default and only allowed value: select's")
    run.add_argument("--shard", default="0/1", help="i/n: the states of the scenes that hash to i")
    run.add_argument("--max-steps", type=int, default=MAX_STEPS)
    run.add_argument("--spf-max-actions", type=int, default=SPF_MAX_ACTIONS)
    run.add_argument("--step-mode", choices=("act", "sim_step"), default="act")
    run.add_argument("--xvfb-display", type=int, default=None, help="start a private Xvfb on :N (llvmpipe)")
    run.add_argument("--xvfb-bin", default=None)
    run.add_argument("--sim-gpu-id", type=int, default=0)
    run.add_argument("--limit", type=int, default=0, help="roll out at most this many states (smoke runs)")
    run.add_argument("--log-every", type=int, default=10)

    rep = sub.add_parser("report", help="K1's verdict over every shard's rows")
    rep.add_argument("--out-dir", required=True)
    rep.add_argument("--output", default="", help=f"default <out-dir>/{SUMMARY_NAME}")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return {"select": cmd_select, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
