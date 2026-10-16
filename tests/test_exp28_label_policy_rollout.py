"""EXP-28 K1 (scripts/exp28/label_policy_rollout.py): state selection, the line-B / SPF rollouts and the verdict.

Everything but the last test runs on the flat synthetic world of ``test_s2_oracle_labels``: geodesics are
straight lines, snapping is the identity, the lookdown depth is the analytic depth of the floor.  ``FlatSim``
stands in for ``HabitatWorld`` (0.25 m steps, 15 deg turns, a greedy stand-in for ShortestPathFollower), so
the rollouts can be worked out by hand.  The Habitat test needs habitat-lab, an X display and
``EXP28_K1_SCENE`` (a .glb with its v1 navmesh next to it).
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.exp25 import probe_metrics as pm
from scripts.exp28 import label_policy_rollout as k1
from src.data.s2_oracle_labels import heading_delta_deg, yaw_to_quat_wxyz
from src.data.s2_records import EpisodeRecord, EpisodeRecordWriter
from src.data.s2_route_labels import RouteLabelConfig
from tests.test_s2_oracle_labels import CAM, FACE_NEG_Z, _jpeg, euclid, floor_depth, straight

CONFIG = RouteLabelConfig()
REFERENCE = [[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 0.0, -8.0], [0.0, 0.0, -12.0]]
GOAL = REFERENCE[-1]
FACE_POS_Z = yaw_to_quat_wxyz(math.pi)
FACE_POS_X = yaw_to_quat_wxyz(-math.pi / 2)
JPEG = _jpeg((64, 48))

PIXEL = {"llm_output": "320 300", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory"}
SKY_PIXEL = {"llm_output": "320 10", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory"}
LEFT1 = {"llm_output": "←", "native_first_output": "←", "native_lookdown_turns": 0, "kind": "native_actions"}
RIGHT2 = {"llm_output": "→→", "native_first_output": "→→", "native_lookdown_turns": 0, "kind": "native_actions"}


class FlatWorld:
    """The labeller's geometry on the flat world."""

    def geodesic(self, a, b):
        return euclid(a, b)

    def find_path(self, a, b):
        return straight(a, b)

    def snap(self, p):
        return np.asarray(p, dtype=np.float64)


class Unreachable(FlatWorld):
    def geodesic(self, a, b):
        return math.inf


class FlatSim:
    """``HabitatWorld``'s interface on an endless floor y = 0."""

    def __init__(self, depth=None, follower=None):
        self.p = np.zeros(3)
        self.yaw = 0.0
        self.depth_fn = depth or (lambda p, q: floor_depth(p, q))
        self.follower_fn = follower
        self.acted = []
        self.loaded = []
        self.max_lookdown_pose_err = 0.0

    def load_scene(self, glb, *, expected_navmesh_sha256=None):
        self.loaded.append((str(glb), expected_navmesh_sha256))
        return {"load_s": 0.0}

    def set_pose(self, position, rotation_wxyz):
        self.p = np.asarray(position, dtype=np.float64).copy()
        self.yaw = k1.v0.yaw_of_wxyz(rotation_wxyz)

    def pose(self):
        return self.p.copy(), np.asarray(yaw_to_quat_wxyz(self.yaw), dtype=np.float64)

    def act(self, action):
        self.acted.append(action)
        if action == k1.FORWARD:
            self.p = self.p + 0.25 * np.array([-math.sin(self.yaw), 0.0, -math.cos(self.yaw)])
        elif action == k1.LEFT:
            self.yaw += math.radians(15)
        elif action == k1.RIGHT:
            self.yaw -= math.radians(15)
        else:
            raise AssertionError(f"not a movement action: {action}")

    def lookdown_depth(self):
        return self.depth_fn(*self.pose())

    def follower(self, goal_radius):
        if self.follower_fn is not None:
            return self.follower_fn

        def greedy(goal):
            p, q = self.pose()
            if euclid(p, goal) <= goal_radius:
                return None
            bearing = heading_delta_deg(q, p, goal)
            if abs(bearing) > 7.5 + 1e-9:
                return k1.LEFT if bearing > 0 else k1.RIGHT
            return k1.FORWARD

        return greedy

    def is_navigable(self, point):
        return True

    def snap(self, point):
        return np.asarray(point, dtype=np.float64)


def write_record(root, scene, episode_id, poses, answers, *, extra=None, depth=False):
    """Call i at step i (the record's pose), one step per pose; ``answers`` may be shorter than ``poses``.

    ``depth``: each call keeps the flat world's lookdown depth at its pose, as the client records it.
    """
    episode = {"reference_path": REFERENCE, "goals": [{"position": GOAL, "radius": 3.0}],
               "start_position": REFERENCE[0], "instruction": "walk ahead", **(extra or {})}
    writer = EpisodeRecordWriter(root=Path(root), scene_id=scene, episode_id=episode_id, episode=episode)
    for step, (position, rotation) in enumerate(poses):
        writer.add_step(step, position, rotation)
        writer.add_front(step, JPEG)
    for index, answer in enumerate(answers):
        writer.add_call({"call_index": index, "step_id": index, "history_step_ids": [], **answer},
                        lookdown_jpeg=JPEG, lookdown_depth_m=floor_depth(*poses[index]) if depth else None)
    return writer.commit()


# Off-route at calls 2-3 (3 m, 3.5 m) and 5 (2.5 m); call 7 is off-route but within 3 m of the goal.
EXCURSIONS = [
    ([0.0, 0.0, 0.0], FACE_NEG_Z), ([0.0, 0.0, -2.0], FACE_NEG_Z), ([3.0, 0.0, -4.0], FACE_NEG_Z),
    ([3.5, 0.0, -5.0], FACE_NEG_Z), ([0.5, 0.0, -6.0], FACE_NEG_Z), ([2.5, 0.0, -8.5], FACE_NEG_Z),
    ([0.0, 0.0, -9.0], FACE_NEG_Z), ([2.1, 0.0, -10.5], FACE_NEG_Z),
]


def excursion_record(root, scene="flatA", episode_id=1, **kwargs):
    return write_record(root, scene, episode_id, EXCURSIONS, [LEFT1, LEFT1, RIGHT2, LEFT1, LEFT1, LEFT1, LEFT1, LEFT1],
                        **kwargs)


def bad_pixel_record(root, scene="flatA", episode_id=2, *, with_next_call=True, **kwargs):
    poses = [([0.0, 0.0, 0.0], FACE_POS_Z), ([0.0, 0.0, 1.5], FACE_POS_Z)]
    answers = [PIXEL, LEFT1] if with_next_call else [PIXEL]
    return write_record(root, scene, episode_id, poses, answers, **kwargs)


# --------------------------------------------------------------------------- #
# B's first pixel goal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.1, math.pi])
@pytest.mark.parametrize("pixel", [(320, 300), (100, 455), (601, 57), (5, 250)])
def test_floor_point_is_probe_metrics_pixel_floor(yaw, pixel):
    position = [1.0, 0.3, -2.0]
    ours = k1.floor_point(pixel, position, yaw_to_quat_wxyz(yaw), CAM)
    theirs, _ = pm.pixel_floor(position, yaw, *pixel)
    if theirs is None:
        assert ours is None
    else:
        np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_floor_point_is_the_labellers_floor_target_without_depth():
    target, how = k1.RL.pixel_floor_target((320, 300), position=[0.0, 0.0, 0.0], rotation_wxyz=FACE_NEG_Z,
                                           depth_m=None, camera=CAM)
    assert how == "floor"
    np.testing.assert_allclose(k1.floor_point((320, 300), [0.0, 0.0, 0.0], FACE_NEG_Z, CAM), target, atol=1e-12)
    assert k1.floor_point((320, 10), [0.0, 0.0, 0.0], FACE_NEG_Z, CAM) is None  # above the horizon


def test_a_bad_first_pixel_gives_the_pose_after_it(tmp_path):
    record = EpisodeRecord(bad_pixel_record(tmp_path))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert stats["first_pixel_bad"] == 1
    assert [s["key"] for s in states] == ["flatA_000002#c0001"]
    state = states[0]
    assert state["sources"] == ["bad_first_pixel"]
    assert state["restore"]["call_index"] == 1 and state["restore"]["position"] == [0.0, 0.0, 1.5]
    assert state["distance_to_goal"] == pytest.approx(13.5)
    first = state["first_pixel"]
    assert first["call_index"] == 0 and first["class"] == "bad"
    assert first["ratio"] == pytest.approx(-1.0, abs=1e-3)  # walked 1.55 m straight away from the goal
    # The executed pixel is not a turn: line B's first call sees no previous turn.
    assert state["previous_turn_direction"] == 0 and state["previous_turns"] == ["↓", "320 300"]


def test_a_bad_first_pixel_without_a_later_call_restores_the_last_step(tmp_path):
    record = EpisodeRecord(bad_pixel_record(tmp_path, with_next_call=False))
    states, _ = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert [s["key"] for s in states] == ["flatA_000002#end"]
    assert states[0]["restore"]["call_index"] is None and states[0]["restore"]["step_id"] == 1


def test_the_first_pixel_is_the_first_one_that_projects(tmp_path):
    # A pixel above the horizon has no floor point (probe_metrics skips it); the next one is the first.
    poses = [([0.0, 0.0, 0.0], FACE_NEG_Z), ([0.0, 0.0, -1.0], FACE_NEG_Z), ([0.0, 0.0, -2.5], FACE_NEG_Z)]
    record = EpisodeRecord(write_record(tmp_path, "flatA", 3, poses, [SKY_PIXEL, PIXEL, LEFT1]))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert stats["first_pixel_good"] == 1 and states == []


def test_the_first_pixel_is_a_trajectory_call_as_b_reads_the_log(tmp_path):
    # B reads "RPC kind=trajectory" lines: digits in another kind of reply are not B's first pixel goal.
    not_trajectory = {**PIXEL, "kind": "native_actions"}
    record = EpisodeRecord(bad_pixel_record(tmp_path))
    assert k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)[1]["first_pixel_bad"] == 1
    poses = [([0.0, 0.0, 0.0], FACE_POS_Z), ([0.0, 0.0, 1.5], FACE_POS_Z)]
    record = EpisodeRecord(write_record(tmp_path / "other", "flatA", 2, poses, [not_trajectory, LEFT1]))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert stats["first_pixel_none"] == 1 and states == []


# --------------------------------------------------------------------------- #
# Off-route excursions
# --------------------------------------------------------------------------- #
def test_off_route_states_are_the_first_call_of_each_excursion_beyond_3m(tmp_path):
    record = EpisodeRecord(excursion_record(tmp_path))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert [s["key"] for s in states] == ["flatA_000001#c0002", "flatA_000001#c0005"]
    assert [s["off_route_m"] for s in states] == [pytest.approx(3.0), pytest.approx(2.5)]
    assert stats["first_pixel_none"] == 1
    assert stats["off_route_within_radius"] == 1  # call 7: 2.1 m off the route, 2.58 m from the goal
    # Previous executed answers: call 1 turned left; call 4 turned left.
    assert [s["previous_turn_direction"] for s in states] == [1, 1]
    assert all(s["last_reached"] == 0 for s in states)
    assert all(s["sources"] == ["off_route"] for s in states)


def test_the_episode_cap_keeps_the_earliest_states(tmp_path):
    record = EpisodeRecord(excursion_record(tmp_path))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG, max_states=1)
    assert [s["key"] for s in states] == ["flatA_000001#c0002"]
    assert stats["off_route_over_episode_cap"] == 1


def test_a_state_that_is_both_kinds_is_one_state(tmp_path):
    # The bad pixel (straight away from the goal) leaves the agent 2.69 m off the route: the next call is both.
    poses = [([0.0, 0.0, -1.0], FACE_POS_Z), ([2.5, 0.0, 1.0], FACE_POS_Z)]
    record = EpisodeRecord(write_record(tmp_path, "flatA", 4, poses, [PIXEL, LEFT1]))
    states, stats = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    assert stats["first_pixel_bad"] == 1
    assert len(states) == 1 and states[0]["sources"] == ["bad_first_pixel", "off_route"]
    assert states[0]["off_route_m"] == pytest.approx(math.hypot(2.5, 1.0))


# --------------------------------------------------------------------------- #
# Cohort order and the stopping rule
# --------------------------------------------------------------------------- #
def write_cohort(root, parts):
    """``parts``: [(part, [[(scene, episode_id), ...] per shard])] in shard order."""
    root.mkdir(parents=True)
    shards, index, info = [], 0, {}
    for part, part_shards in parts:
        for entries in part_shards:
            name = f"shard_{index:03d}.json"
            (root / name).write_text(json.dumps({"part": part, "episodes": [
                {"scene_id": s, "episode_id": e, "trajectory_id": e} for s, e in entries]}))
            shards.append({"index": index, "path": name, "part": part})
            info.setdefault(part, {"shards": ""})
            index += 1
    (root / "manifest.json").write_text(json.dumps({"schema": k1.COHORT_SCHEMA, "shards": shards, "parts": info}))
    return root


def test_cohort_episodes_reads_the_train_parts_in_shard_order(tmp_path):
    cohort = write_cohort(tmp_path / "cohort", [
        ("heldout", [[("flatH", 9)]]),
        ("train_pass1", [[("flatA", 1), ("flatA", 2)], [("flatB", 3)]]),
        ("train_pass2", [[("flatA", 4)]]),
    ])
    episodes, info = k1.cohort_episodes(cohort)
    order = [(e["scene_id"], e["episode_id"]) for e in episodes]
    assert order == [("flatA", 1), ("flatA", 2), ("flatB", 3), ("flatA", 4)]
    assert info["parts"] == ["train_pass1", "train_pass2"]
    only, _ = k1.cohort_episodes(cohort, ["train_pass2"])
    assert [(e["scene_id"], e["episode_id"]) for e in only] == [("flatA", 4)]
    with pytest.raises(k1.K1Invalid, match="unknown"):
        k1.cohort_episodes(cohort, ["train_pass3"])
    manifest = json.loads((cohort / "manifest.json").read_text())
    (cohort / "manifest.json").write_text(json.dumps({**manifest, "schema": "other"}))
    with pytest.raises(k1.K1Invalid, match="schema"):
        k1.cohort_episodes(cohort)


def _selection_inputs(tmp_path, *, navmesh_sha256=None, depth=False):
    records = tmp_path / "records"
    kwargs = {"extra": {"navmesh_sha256": navmesh_sha256} if navmesh_sha256 else None, "depth": depth}
    excursion_record(records, "flatA", 1, **kwargs)  # 2 states
    write_record(records, "flatA", 2, [([0.0, 0.0, 0.0], FACE_NEG_Z)], [LEFT1], **kwargs)  # none
    bad_pixel_record(records, "flatA", 4, **kwargs)  # 1 state; flatB/3 has no record
    episodes = [{"scene_id": s, "episode_id": e} for s, e in (("flatA", 1), ("flatA", 2), ("flatB", 3), ("flatA", 4))]
    return records, episodes


def test_selection_stops_once_both_minimums_hold(tmp_path):
    records, episodes = _selection_inputs(tmp_path)
    flat = lambda scene: (FlatWorld(), None)  # noqa: E731
    states, meta = k1.select_states(episodes, record_roots=[records], geometry_for_scene=flat, config=CONFIG,
                                    min_episodes=2, min_states=2)
    assert meta["episodes_scanned"] == 2 and meta["states"] == 2 and meta["stopped"].startswith("enough")
    assert meta["enough"] is False  # the ledger's 100 / 200, whatever the flags
    states, meta = k1.select_states(episodes, record_roots=[records], geometry_for_scene=flat, config=CONFIG,
                                    min_episodes=2, min_states=3)
    assert meta["episodes_scanned"] == 3 and meta["states"] == 3
    assert meta["missing_records"] == {"n": 1, "first": ["flatB_000003"]}
    assert meta["states_by_source"] == {"off_route": 2, "bad_first_pixel": 1}
    assert meta["episodes_with_states"] == 2
    states, meta = k1.select_states(episodes, record_roots=[records], geometry_for_scene=flat, config=CONFIG,
                                    min_episodes=10, min_states=10)
    assert meta["stopped"] == "cohort_exhausted" and meta["episodes_scanned"] == 3


def test_selection_refuses_held_out_scenes_and_foreign_navmeshes(tmp_path):
    records, episodes = _selection_inputs(tmp_path)
    flat = lambda scene: (FlatWorld(), None)  # noqa: E731
    with pytest.raises(k1.K1Invalid, match="held-out"):
        k1.select_states(episodes, record_roots=[records], geometry_for_scene=flat, config=CONFIG,
                         holdout=["00123-flatA"])
    write_record(tmp_path / "other", "flatC", 5, [([0.0, 0.0, 0.0], FACE_NEG_Z)], [LEFT1],
                 extra={"navmesh_sha256": "a" * 64})
    with pytest.raises(k1.K1Invalid, match="navmesh"):
        k1.select_states([{"scene_id": "flatC", "episode_id": 5}], record_roots=[tmp_path / "other"],
                         geometry_for_scene=lambda scene: (FlatWorld(), "b" * 64), config=CONFIG)
    # A record that names no navmesh is not an EXP-28 HM3D record: refused once the scene's navmesh is known.
    with pytest.raises(k1.K1Invalid, match="not an EXP-28 HM3D record"):
        k1.select_states(episodes, record_roots=[records], geometry_for_scene=lambda scene: (FlatWorld(), "b" * 64),
                         config=CONFIG)
    excursion_record(tmp_path / "twice", "flatA", 1)
    with pytest.raises(k1.K1Invalid, match="twice"):
        k1.select_states(episodes, record_roots=[records, tmp_path / "twice"], geometry_for_scene=flat, config=CONFIG)


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def rollout(world, *, position, rotation, previous=0, last_reached=0, geometry=None, **settings):
    world.set_pose(position, rotation)
    return k1.line_b_rollout(world, geometry or FlatWorld(), goal=GOAL,
                             waypoints=np.asarray(REFERENCE, dtype=np.float64), last_reached=last_reached,
                             previous_turn_direction=previous, config=CONFIG, camera=CAM,
                             settings=k1.RolloutSettings(**settings))


def test_a_skip_is_one_step_forward():
    # No depth return anywhere: the route straight ahead is never seen, so every call is a skip.
    world = FlatSim(depth=lambda p, q: np.zeros((480, 640), dtype=np.float32))
    result = rollout(world, position=[0.0, 0.0, 0.0], rotation=FACE_NEG_Z)
    assert result["outcome"] == "reached" and result["steps"] == 36  # 12 m -> 3 m in 0.25 m steps
    assert result["num_calls"] == 36 and result["oracle_kinds"] == {"skip": 36}
    assert all(call["actions"] == [k1.FORWARD] and call["rule"] == "skip_ahead_hidden" for call in result["calls"])
    world.set_pose([0.0, 0.0, 0.0], FACE_NEG_Z)
    expert = k1.expert_rollout(world, FlatWorld(), goal=GOAL, settings=k1.RolloutSettings())
    assert expert["reached"] and expert["steps"] == 36


def test_line_b_brings_an_off_route_agent_into_the_circle():
    world = FlatSim()
    result = rollout(world, position=[3.0, 0.0, -2.0], rotation=FACE_POS_X)
    assert result["outcome"] == "reached" and result["final_d"] <= 3.0
    assert result["turn_flips"] == 0
    kinds = result["oracle_kinds"]
    assert kinds.get("turn", 0) >= 1 and kinds.get("pixel", 0) >= 1  # turns back to the route, then follows pixels
    for call in result["calls"]:
        if call["oracle"] == "turn":
            assert call["actions"] == [k1.LEFT if call["direction"] > 0 else k1.RIGHT] * call["count"] or \
                call is result["calls"][-1]
        if call["oracle"] == "pixel":
            assert 1 <= len(call["actions"]) <= 8 and not call["spf"]["anti_deadlock"]
            u, v = call["pixel"]
            assert 100 <= u <= 540 and 32 <= v <= 455
    assert sum(len(c["actions"]) for c in result["calls"]) == result["steps"] == len(world.acted)
    world.set_pose([3.0, 0.0, -2.0], FACE_POS_X)
    expert = k1.expert_rollout(world, FlatWorld(), goal=GOAL, settings=k1.RolloutSettings())
    assert expert["reached"] and 0 < expert["steps"] <= result["steps"] < 2 * expert["steps"]


def test_the_first_call_keeps_turning_the_recorded_way_when_the_route_is_behind():
    for previous in (1, -1):
        world = FlatSim()
        result = rollout(world, position=[0.0, 0.0, -2.0], rotation=FACE_POS_Z, previous=previous, max_steps=4)
        first = result["calls"][0]
        assert (first["oracle"], first["rule"], first["direction"], first["prev_direction"]) == \
            ("turn", "hysteresis", previous, previous)
        assert result["outcome"] == "step_cap" and world.acted == [k1.LEFT if previous > 0 else k1.RIGHT] * 4
        assert result["first_call_flips_recorded_turn"] == 0


def test_a_follower_that_stops_at_once_gets_one_left_turn():
    world = FlatSim(follower=lambda goal: None)
    result = rollout(world, position=[0.0, 0.0, 0.0], rotation=FACE_NEG_Z, max_steps=1)
    first = result["calls"][0]
    assert first["oracle"] == "pixel" and first["spf"]["anti_deadlock"] and first["actions"] == [k1.LEFT]
    assert result["outcome"] == "step_cap" and result["anti_deadlock"] == 1


def test_a_failing_follower_is_counted_and_the_pixel_still_moves_the_agent():
    def broken(goal):
        raise RuntimeError("GreedyFollowerError")

    world = FlatSim(follower=broken)
    result = rollout(world, position=[0.0, 0.0, 0.0], rotation=FACE_NEG_Z, max_steps=1)
    spf = result["calls"][0]["spf"]
    assert spf["anti_deadlock"] and "GreedyFollowerError" in spf["error"] and result["follower_errors"] == 1


def test_the_follower_runs_at_most_spf_max_actions_per_pixel():
    world = FlatSim(follower=lambda goal: k1.FORWARD)  # never says it arrived
    result = rollout(world, position=[0.0, 0.0, 0.0], rotation=FACE_NEG_Z, spf_max_actions=3, max_steps=7)
    pixel_calls = [c for c in result["calls"] if c["oracle"] == "pixel"]
    assert [len(c["actions"]) for c in pixel_calls][:2] == [3, 3]
    assert pixel_calls[0]["spf"]["end"] == "max_actions"


def test_rollouts_end_on_an_unreachable_goal():
    world = FlatSim()
    result = rollout(world, position=[0.0, 0.0, 0.0], rotation=FACE_NEG_Z, geometry=Unreachable())
    assert result["outcome"] == "goal_unreachable" and result["steps"] == 0 and result["num_calls"] == 0
    world = FlatSim(follower=lambda goal: None)
    expert = k1.expert_rollout(world, FlatWorld(), goal=GOAL, settings=k1.RolloutSettings())
    assert expert["outcome"] == "follower_stopped_outside_radius" and not expert["reached"]


def test_settings_that_could_stand_still_are_refused():
    with pytest.raises(ValueError):
        k1.RolloutSettings(spf_max_actions=0)
    with pytest.raises(ValueError):
        k1.RolloutSettings(max_steps=0)


def test_turn_flips_count_consecutive_turn_calls_only():
    calls = [{"oracle": "turn", "direction": 1}, {"oracle": "turn", "direction": 1},
             {"oracle": "turn", "direction": -1}, {"oracle": "pixel", "direction": 0},
             {"oracle": "turn", "direction": 1}, {"oracle": "skip", "direction": 0},
             {"oracle": "turn", "direction": -1}]
    assert k1.turn_flips(calls) == (2, 1)
    assert k1.turn_flips_across(calls) == (4, 3)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
STAMP = {"selection_sha256": "s" * 64, "config_digest": CONFIG.digest(),
         "settings": {"max_steps": 500, "spf_goal_radius_m": 0.25, "spf_max_actions": 8, "success_radius_m": 3.0}}


def fake_row(key, *, reached=True, steps=15, expert_steps=10, pairs=1, flips=0, expert_reached=True):
    return {
        "key": key, "sources": ["off_route"], "ratio": steps / expert_steps if reached and expert_reached else None,
        "within_episode_budget": reached,
        "line_b": {"reached": reached, "outcome": "reached" if reached else "step_cap",
                   "steps": steps if reached else 500,
                   "final_d": 2.9, "turn_pairs": pairs, "turn_flips": flips, "turn_pairs_across_other_calls": pairs,
                   "turn_flips_across_other_calls": flips, "first_call_flips_recorded_turn": None, "num_calls": 5,
                   "calls": []},
        "expert": {"reached": expert_reached, "outcome": "reached" if expert_reached else "follower_error",
                   "steps": expert_steps},
        **STAMP,
    }


def verdict(n_states=200, scanned=100, *, failed=20, flips=2, slow=0, drop=0):
    states = [{"key": f"s{i:03d}"} for i in range(n_states)]
    rows = [fake_row(s["key"], reached=i >= failed, flips=int(i < flips), steps=16 if i >= n_states - slow else 15)
            for i, s in enumerate(states)][drop:]
    return k1.summarize(states, rows, {"episodes_scanned": scanned}, states_sha256=STAMP["selection_sha256"],
                        config_digest=CONFIG.digest())


def test_k1_passes_exactly_on_its_lines():
    summary = verdict()
    c = summary["criteria"]
    assert summary["verdict"] == "pass", summary
    assert (c["reach"]["k"], c["reach"]["n"], c["reach"]["value"]) == (180, 200, 0.9)
    assert (c["flip"]["k"], c["flip"]["n"], c["flip"]["value"]) == (2, 200, 0.01)
    # 20 failures count as infinite ratios; the median of 200 is still 15 / 10.
    assert c["ratio_median"]["value"] == 1.5 and c["ratio_median"]["line_b_failed"] == 20


def test_k1_fails_one_past_any_line():
    assert verdict(failed=21)["criteria"]["reach"]["pass"] is False
    assert verdict(flips=3)["criteria"]["flip"]["pass"] is False
    # 101 of the 200 ratios at 1.6: the median moves past 1.5.
    slow = verdict(failed=0, slow=101)
    assert slow["criteria"]["ratio_median"]["value"] == 1.6 and slow["verdict"] == "fail"
    assert verdict(failed=0, slow=100)["criteria"]["ratio_median"]["value"] == pytest.approx(1.55)


def test_the_median_counts_line_b_failures_as_infinite():
    assert k1.median([Fraction(1), math.inf, math.inf]) == math.inf
    assert k1.median([Fraction(1), Fraction(2), math.inf, math.inf]) == math.inf
    assert k1.median([Fraction(1), Fraction(2), Fraction(4), math.inf]) == Fraction(3)
    assert k1.median([]) is None


def test_k1_is_not_measurable_without_enough_or_matching_rows():
    assert verdict(n_states=199)["verdict"] == "invalid"
    assert verdict(scanned=99)["verdict"] == "invalid"
    assert verdict(drop=1)["verdict"] == "invalid"
    states = [{"key": f"s{i:03d}"} for i in range(200)]
    rows = [fake_row(s["key"]) for s in states]
    rows[0] = {**rows[0], "config_digest": "other"}
    summary = k1.summarize(states, rows, {"episodes_scanned": 100}, states_sha256=STAMP["selection_sha256"],
                           config_digest=CONFIG.digest())
    assert summary["verdict"] == "invalid" and any("config" in p for p in summary["problems"])
    rows = [fake_row(s["key"]) for s in states]
    rows[1] = {**rows[1], "settings": {**STAMP["settings"], "max_steps": 400}}
    assert k1.summarize(states, rows, {"episodes_scanned": 100})["verdict"] == "invalid"


def test_states_whose_expert_fails_are_left_out_of_the_ratio():
    states = [{"key": f"s{i:03d}"} for i in range(200)]
    rows = [fake_row(s["key"], expert_reached=i >= 10) for i, s in enumerate(states)]
    c = k1.summarize(states, rows, {"episodes_scanned": 100})["criteria"]["ratio_median"]
    assert c["n"] == 190 and c["expert_failed"] == 10 and c["pass"]


# --------------------------------------------------------------------------- #
# select -> rollouts -> report, with the flat world in place of Habitat
# --------------------------------------------------------------------------- #
def test_select_run_report_round_trip(tmp_path, monkeypatch, capsys):
    navmesh_bytes = b"navmesh flatA"
    records, _ = _selection_inputs(tmp_path, navmesh_sha256=k1.hashlib.sha256(navmesh_bytes).hexdigest(), depth=True)
    cohort = write_cohort(tmp_path / "cohort", [
        ("heldout", [[("flatH", 9)]]),
        ("train_pass1", [[("flatA", 1), ("flatA", 2)], [("flatB", 3)], [("flatA", 4)]]),
    ])
    scenes = tmp_path / "scenes"
    for scene in ("flatA", "flatB"):
        (scenes / scene).mkdir(parents=True)
        (scenes / scene / f"{scene}.navmesh").write_bytes(b"navmesh " + scene.encode())
    monkeypatch.setattr(k1, "navmesh_geometry", lambda path: FlatWorld())
    out = tmp_path / "k1"
    argv = ["select", "--cohort-dir", str(cohort), "--record-root", str(records), "--scenes-dir", str(scenes),
            "--out-dir", str(out)]
    # The held-out list next to the cohort (build_hm3d_episodes.py's splits/) is required.
    assert k1.main(argv) == 2 and "heldout_scenes.txt" in capsys.readouterr().out
    assert not (out / k1.STATES_NAME).exists()
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "heldout_scenes.txt").write_text("# held-out\n00123-flatH\n")
    assert k1.main(argv) == 2  # 3 states from 3 episodes: not enough for K1, but written
    assert k1.main(argv) == 2  # and kept: the states stay fixed across shards
    states, selection, digest = k1.load_selection(out)
    assert [s["key"] for s in states] == ["flatA_000001#c0002", "flatA_000001#c0005", "flatA_000004#c0001"]
    assert selection["episodes_scanned"] == 3 and selection["config"]["digest"] == CONFIG.digest()
    assert selection["inputs"]["holdout_scenes"] == 1 and selection["record_policies"] == {"None|None": 3}
    assert {s["navmesh_sha256"] for s in states} == {k1.hashlib.sha256(navmesh_bytes).hexdigest()}

    world = FlatSim()
    settings = k1.RolloutSettings()
    stamp = {"selection_sha256": digest, "config_digest": CONFIG.digest(),
             "settings": __import__("dataclasses").asdict(settings), "step_mode": "act"}
    rows_path = k1.rows_path_for(out, 0, 1)
    counts = k1.run_states(states, world=world, scenes_dir=scenes, geometry_for_navmesh=lambda path: FlatWorld(),
                           dataset_index=None, config=CONFIG, camera=CAM, settings=settings, rows_path=rows_path,
                           errors_path=out / "errors.shard00of01.jsonl", stamp=stamp)
    assert counts["done"] == 3 and counts["errors"] == 0
    assert world.loaded == [(str(scenes / "flatA" / "flatA.glb"), states[0]["navmesh_sha256"])]
    rows = k1.read_jsonl(rows_path)
    assert [r["key"] for r in rows] == [s["key"] for s in states]
    for row in rows:
        assert row["line_b"]["reached"] and row["expert"]["reached"] and row["ratio"] is not None
        assert row["selection_sha256"] == digest
        assert row["restored_depth"]["match"] == 1.0  # the world renders the record's own lookdown
    assert rows[2]["sources"] == ["bad_first_pixel"] and rows[2]["restore_step_id"] == 1
    assert [r["restore_call_index"] for r in rows] == [2, 5, 1]

    assert k1.main(["report", "--out-dir", str(out)]) == 2  # measured, but K1 needs 100 episodes / 200 states
    summary = json.loads((out / k1.SUMMARY_NAME).read_text())
    assert summary["verdict"] == "invalid" and summary["criteria"]["reach"]["k"] == 3
    assert summary["report_only"]["by_source"]["off_route"]["n"] == 2
    assert summary["report_only"]["restored_depth"]["checked"] == 3
    assert not any("lookdown" in p or "unterminated" in p for p in summary["problems"])

    # Another shard's runner skips what the smoke shard did; a kill mid-row is repaired in the own file only.
    assert k1.done_keys(out, k1.rows_path_for(out, 1, 4), digest) == {s["key"] for s in states}
    with open(rows_path, "a", encoding="utf-8") as handle:
        handle.write('{"key": "half a ro')
    assert k1.main(["report", "--out-dir", str(out)]) == 2
    assert any("unterminated" in p for p in json.loads((out / k1.SUMMARY_NAME).read_text())["problems"])
    assert k1.done_keys(out, k1.rows_path_for(out, 1, 4), digest) == {s["key"] for s in states}
    assert rows_path.read_text().endswith("ro")  # not this runner's file: left alone
    assert k1.done_keys(out, rows_path, digest) == {s["key"] for s in states}
    assert rows_path.read_text().endswith("}\n") and len(k1.read_jsonl(rows_path)) == 3

    (out / k1.STATES_NAME).write_text((out / k1.STATES_NAME).read_text() + "\n")
    with pytest.raises(k1.K1Invalid, match="changed since select"):
        k1.load_selection(out)


def test_run_reads_the_scenes_and_episodes_select_read(tmp_path):
    selection = {"inputs": {"scenes_dir": str(tmp_path / "scenes"), "episodes": [str(tmp_path / "pool.json.gz")]}}

    def args(scenes="", episodes=()):
        return SimpleNamespace(scenes_dir=scenes, episodes=list(episodes))

    assert k1.run_inputs(args(), selection) == (str(tmp_path / "scenes"), [str(tmp_path / "pool.json.gz")])
    assert k1.run_inputs(args(str(tmp_path / "scenes"), [tmp_path / "pool.json.gz"]), selection)[0] == \
        str(tmp_path / "scenes")
    with pytest.raises(k1.K1Invalid, match="scenes-dir"):
        k1.run_inputs(args(str(tmp_path / "other")), selection)
    with pytest.raises(k1.K1Invalid, match="episodes"):
        k1.run_inputs(args(episodes=[tmp_path / "other.json.gz"]), selection)
    with pytest.raises(k1.K1Invalid, match="episodes"):  # select read the records' own episode.json
        k1.run_inputs(args(episodes=[tmp_path / "pool.json.gz"]),
                      {"inputs": {"scenes_dir": str(tmp_path / "scenes"), "episodes": []}})


def test_a_world_that_does_not_render_the_records_lookdown_is_caught(tmp_path):
    record = EpisodeRecord(excursion_record(tmp_path, depth=True))
    state = k1.record_states(record, FlatWorld(), dataset_entry=None, config=CONFIG)[0][0]
    settings = k1.RolloutSettings(max_steps=5)
    same = k1.rollout_state(FlatSim(), FlatWorld(), state, record, dataset_entry=None, config=CONFIG, camera=CAM,
                            settings=settings)
    assert same["restore_call_index"] == 2 and same["restored_depth"]["match"] == 1.0
    # A wall 1 m ahead that the record never saw: most of the lookdown is off.
    walled = FlatSim(depth=lambda p, q: floor_depth(p, q, wall_z=float(p[2]) - 1.0))
    other = k1.rollout_state(walled, FlatWorld(), state, record, dataset_entry=None, config=CONFIG, camera=CAM,
                             settings=settings)
    assert other["restored_depth"]["match"] < k1.DEPTH_MATCH_MIN
    # Without depth in the record there is nothing to compare.
    bare = EpisodeRecord(excursion_record(tmp_path / "bare"))
    assert k1.rollout_state(FlatSim(), FlatWorld(), state, bare, dataset_entry=None, config=CONFIG, camera=CAM,
                            settings=settings)["restored_depth"] is None


def _rows_with_depth(n, bad=0, unchecked=0):
    rows = []
    for i in range(n):
        row = fake_row(f"s{i:03d}")
        row["restore_call_index"] = 0
        row["restored_depth"] = None if i < unchecked else {"match": 0.2 if i >= n - bad else 1.0}
        rows.append(row)
    return rows


def test_k1_is_not_read_when_the_rendered_lookdown_is_not_the_records():
    states = [{"key": f"s{i:03d}"} for i in range(200)]

    def verdict_of(rows):
        return k1.summarize(states, rows, {"episodes_scanned": 100}, states_sha256=STAMP["selection_sha256"],
                            config_digest=CONFIG.digest())

    assert verdict_of(_rows_with_depth(200, bad=2))["verdict"] == "pass"  # 1% of the states: allowed
    summary = verdict_of(_rows_with_depth(200, bad=3))
    assert summary["verdict"] == "invalid" and any("lookdown" in p for p in summary["problems"])
    summary = verdict_of(_rows_with_depth(200, unchecked=1))
    assert summary["verdict"] == "invalid" and any("no rendered-vs-recorded" in p for p in summary["problems"])


def test_duplicates_must_agree_and_the_expert_must_mostly_reach():
    states = [{"key": f"s{i:03d}"} for i in range(200)]
    rows = [fake_row(s["key"]) for s in states]
    same = k1.summarize(states, rows + [dict(rows[0])], {"episodes_scanned": 100})
    assert same["verdict"] == "pass" and any("duplicate" in n for n in same["notes"])
    other = k1.summarize(states, rows + [fake_row("s000", steps=40)], {"episodes_scanned": 100})
    assert other["verdict"] == "invalid" and any("different results" in p for p in other["problems"])
    for failed, verdict_ in ((20, "pass"), (21, "invalid")):
        rows = [fake_row(s["key"], expert_reached=i >= failed) for i, s in enumerate(states)]
        assert k1.summarize(states, rows, {"episodes_scanned": 100})["verdict"] == verdict_
    moved = k1.summarize(states, [fake_row(s["key"]) for s in states],
                         {"episodes_scanned": 100, "config": {"digest": "older"}}, config_digest=CONFIG.digest())
    assert moved["verdict"] == "invalid" and any("selected under" in p for p in moved["problems"])


def test_the_client_camera_check():
    def cfg(**over):
        depth = dict(WIDTH=640, HEIGHT=480, HFOV=79, POSITION=[0, 1.25, 0], MIN_DEPTH=0.0, MAX_DEPTH=10.0,
                     NORMALIZE_DEPTH=True)
        sim = dict(TILT_ANGLE=15, TURN_ANGLE=15, FORWARD_STEP_SIZE=0.25, ACTION_SPACE_CONFIG="v1")
        depth.update({k: v for k, v in over.items() if k in depth})
        sim.update({k: v for k, v in over.items() if k in sim})
        return SimpleNamespace(SIMULATOR=SimpleNamespace(DEPTH_SENSOR=SimpleNamespace(**depth), **sim))

    assert k1.check_client_camera(cfg())["hfov_deg"] == 79.0
    bad_specs = ({"HFOV": 90}, {"TILT_ANGLE": 30}, {"MAX_DEPTH": 5.0}, {"POSITION": [0, 1.5, 0]},
                 {"NORMALIZE_DEPTH": False})
    for bad in bad_specs:
        with pytest.raises(k1.K1Invalid):
            k1.check_client_camera(cfg(**bad))


def test_shards_deal_whole_scenes():
    assert k1.parse_shard("2/4") == (2, 4)
    with pytest.raises(ValueError):
        k1.parse_shard("4/4")
    assert {k1.shard_of("flatA", 3) for _ in range(3)} == {k1.shard_of("flatA", 3)}


# --------------------------------------------------------------------------- #
# Habitat
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("habitat")
    pytest.importorskip("quaternion")
    # Imported here, not at collection: the client patches flash_attn, gym and numpy on import.
    import scripts.evaluation.r2r_val_unseen as module

    return module


def test_the_client_config_with_depth_only_is_the_deployment_camera(client):
    cfg = k1.v0.client_habitat_config(client, "/x/hm3d/AbCdEfGhIjK/AbCdEfGhIjK.glb")
    spec = k1.check_client_camera(cfg)
    assert spec["position"] == [0.0, 1.25, 0.0] and spec["tilt_deg"] == 15.0


@pytest.mark.skipif(not (os.environ.get("EXP28_K1_SCENE") and os.environ.get("DISPLAY")),
                    reason="needs EXP28_K1_SCENE (a .glb with its v1 navmesh next to it) and an X display")
def test_habitat_world_renders_the_lookdown_and_moves_like_habitat(client):
    world = k1.HabitatWorld(client)
    try:
        world.load_scene(Path(os.environ["EXP28_K1_SCENE"]))
        point = np.asarray(world.sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        q = yaw_to_quat_wxyz(0.6)
        world.set_pose(point, q)
        depth = world.lookdown_depth()
        assert depth.shape == (480, 640) and depth.dtype == np.float32
        assert 0.0 <= float(depth.min()) and float(depth.max()) <= 10.0
        assert world.max_lookdown_pose_err < 1e-3
        # Deterministic: the duplicate-row and restored-lookdown checks rely on it.
        again = k1.depth_agreement(world.lookdown_depth(), depth)
        assert again["match"] == 1.0 and again["mean_abs_m"] == 0.0
        p, r = world.pose()  # the lookdown leaves the body where it was, sensors level
        np.testing.assert_allclose(p, point, atol=1e-4)
        world.act(k1.FORWARD)
        moved = float(np.linalg.norm(world.pose()[0] - p))
        assert moved <= 0.25 + 1e-4
        world.set_pose(point, q)
        world.act(k1.LEFT)
        assert k1.yaw_deg(world.pose()[1]) == pytest.approx(math.degrees(0.6) + 15.0, abs=1e-3)
        # sim.step moves the agent the same way agent.act does.
        world.set_pose(point, q)
        world.act(k1.FORWARD)
        via_act = world.pose()[0]
        world.step_mode = "sim_step"
        world.set_pose(point, q)
        world.act(k1.FORWARD)
        np.testing.assert_allclose(world.pose()[0], via_act, atol=1e-5)
        goal = np.asarray(world.sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        action = world.follower(0.25)(goal)
        assert action is None or action in (k1.STOP, k1.FORWARD, k1.LEFT, k1.RIGHT)
    finally:
        world.close()
