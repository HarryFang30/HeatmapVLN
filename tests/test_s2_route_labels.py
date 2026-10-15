"""EXP-25 line B: route-correction labels on a flat synthetic world.

Geodesics are straight lines, snapping is the identity and the lookdown depth is the
analytic depth of the floor (optionally cut by a wall), as in ``test_s2_oracle_labels``,
so every expected answer can be worked out by hand.
"""

import json
import math

import numpy as np
import pytest

from src.data.s2_oracle_labels import parse_model_answer, yaw_to_quat_wxyz
from src.data.s2_records import EpisodeRecordWriter, label_path_for, read_labels
from src.data.s2_route_labels import (
    RouteLabelConfig,
    angle_in_direction,
    label_route_call,
    pixel_floor_target,
    route_oracle,
    turn_count,
)
from src.data.s2_sft_dataset import index_calls, normalize_label
from tests.test_s2_oracle_labels import CAM, FACE_NEG_Z, _floor_pixel, _jpeg, euclid, floor_depth, straight

GOAL = np.array([0.0, 0.0, -8.0])
ROUTE = straight([0, 0, 0], GOAL)


def _bearing_point(deg, dist=6.0):
    """A floor point ``dist`` away at bearing ``deg`` (left positive) from an agent facing -z."""
    a = math.radians(deg)
    return [-dist * math.sin(a), 0.0, -dist * math.cos(a)]


def snap(p):
    return np.asarray(p, dtype=np.float64)


def _call(answer):
    texts = {
        "stop": {"llm_output": "STOP", "native_first_output": "STOP"},
        "left": {"llm_output": "←←"},
        "right": {"llm_output": "→"},
        "cancel": {"llm_output": "←→"},
        "up": {"llm_output": "↑↑"},
    }
    if isinstance(answer, tuple):
        u, v = answer
        return {"llm_output": f"{u} {v}", "native_first_output": "↓", "native_lookdown_turns": 1}
    return texts[answer]


def _label(answer, *, previous=None, d=None, route=ROUTE, goal=GOAL, wall_z=None, config=RouteLabelConfig(),
           depth="floor"):
    goal = np.asarray(goal, float)
    d = euclid([0, 0, 0], goal) if d is None else d
    depth_m = floor_depth([0, 0, 0], FACE_NEG_Z, wall_z=wall_z) if depth == "floor" else depth
    return label_route_call(
        model=parse_model_answer(_call(answer)),
        previous_model=None if previous is None else parse_model_answer(_call(previous)),
        position=[0, 0, 0],
        rotation_wxyz=FACE_NEG_Z,
        distance_to_goal=d,
        goal=goal,
        route=None if route is None else np.asarray(route, float),
        depth_m=depth_m,
        geodesic=euclid,
        snap=snap,
        config=config,
    )


# --------------------------------------------------------------------------- #
# No STOP, nothing within 3 m
# --------------------------------------------------------------------------- #
def test_states_within_three_metres_are_left_to_line_a():
    row = _label("left", d=2.9)
    assert row["skip"] is True and row["kind"] == "none" and row["reason"] == "within_line_a_radius"
    assert row["target_turn1"] is None and row["oracle_kind"] == "skip"
    assert normalize_label({"call_index": 0, **row}).skip


def test_a_stop_beyond_three_metres_is_corrected_to_walk_on_never_to_stop():
    row = _label("stop")
    assert row["agree"] is False and row["kind"] == "correction" and row["reason"] == "model_stop_beyond_radius"
    assert row["target_turn1"] == "↓" and row["target_turn2"] == row["oracle_turn2"]
    for answer in ("left", "right", "up", (320, 300)):
        assert _label(answer)["target_turn1"] != "STOP"


# --------------------------------------------------------------------------- #
# Pixel goals: InternData's convention
# --------------------------------------------------------------------------- #
def test_pixel_goal_is_the_farthest_visible_route_point_within_three_metres():
    answer = route_oracle(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, route=ROUTE,
                          depth_m=floor_depth([0, 0, 0], FACE_NEG_Z))
    assert answer.kind == "pixel" and answer.turn1 == "↓"
    assert answer.debug["pixel_ahead_arc_m"] == pytest.approx(3.0)
    assert answer.debug["pixel_ahead_euclid_m"] == pytest.approx(3.0)
    assert answer.pixel == _floor_pixel(0.0, -3.0)  # not the 6-8 m round 1 took
    assert answer.turn2 == "%d %d" % answer.pixel


def test_pixel_goal_on_a_bend_stays_inside_u_100_540():
    # 1 m ahead the route turns hard left and leaves the frame through its left edge.
    route = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-6.0, 0.0, -1.0]])
    answer = route_oracle(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, route=route,
                          depth_m=floor_depth([0, 0, 0], FACE_NEG_Z))
    assert answer.kind == "pixel"
    u, v = answer.pixel
    assert 100 <= u <= 540 and 32 <= v <= 455
    assert answer.debug["pixel_ahead_arc_m"] <= 3.0 and answer.debug["pixel_ahead_euclid_m"] <= 3.0


def test_a_route_seen_only_beyond_three_metres_is_no_pixel_goal():
    # The route starts 3.5 m ahead (the agent is off it): visible, but outside InternData's
    # 3 m, so no pixel; it is straight ahead, so no turn either.
    route = straight([0, 0, -3.5], [0, 0, -8.0])
    answer = route_oracle(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, route=route,
                          depth_m=floor_depth([0, 0, 0], FACE_NEG_Z))
    assert answer.kind == "skip" and answer.reason == "route_ahead_but_not_visible"
    assert answer.debug["n_visible"] > 0 and answer.debug["n_usable"] == 0


def test_a_route_hidden_straight_ahead_is_a_skip():
    answer = route_oracle(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, route=ROUTE,
                          depth_m=floor_depth([0, 0, 0], FACE_NEG_Z, wall_z=-0.45))
    assert answer.kind == "skip" and answer.debug["n_visible"] == 0


# --------------------------------------------------------------------------- #
# Turns: hysteresis, side band, tie-break
# --------------------------------------------------------------------------- #
def _turn(target_deg, *, previous=None, wall_z=None, config=RouteLabelConfig()):
    previous_direction = {None: 0, "left": 1, "right": -1}[previous]
    return route_oracle(
        position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, route=straight([0, 0, 0], _bearing_point(target_deg)),
        depth_m=floor_depth([0, 0, 0], FACE_NEG_Z, wall_z=wall_z), previous_turn_direction=previous_direction,
        config=config,
    )


def test_turn_count_and_angles():
    cfg = RouteLabelConfig()
    assert [turn_count(a, cfg) for a in (5, 20, 35, 52, 53, 300)] == [1, 1, 2, 3, 4, 4]
    assert angle_in_direction(170.0, 1) == pytest.approx(170.0)
    assert angle_in_direction(170.0, -1) == pytest.approx(190.0)
    assert angle_in_direction(-100.0, 1) == pytest.approx(260.0)


def test_hidden_route_keeps_the_previous_turn_direction():
    behind_left = _turn(170.0, previous="right")  # the shorter way is left
    assert (behind_left.kind, behind_left.turn1) == ("turn", "→→→→")
    assert behind_left.debug["rule"] == "hysteresis" and behind_left.debug["shorter_direction"] == 1
    right_side = _turn(-100.0, previous="left")
    assert right_side.turn1 == "←←←←" and right_side.debug["rule"] == "hysteresis"
    # hidden in front, 35 deg right, behind a wall: still no flip unseen ...
    occluded = _turn(-35.0, previous="left", wall_z=-0.45)
    assert occluded.turn1 == "←←←←" and occluded.debug["rule"] == "hysteresis"
    # ... unless the knob lets a small flip through
    flip = _turn(-35.0, previous="left", wall_z=-0.45, config=RouteLabelConfig(hysteresis_max_flip_deg=40.0))
    assert flip.turn1 == "→→" and flip.debug["rule"] == "shorter_rotation"


def test_without_a_previous_turn_the_shorter_rotation_and_left_when_behind():
    assert _turn(-100.0).turn1 == "→→→→"
    assert _turn(-35.0, wall_z=-0.45).turn1 == "→→"  # round(35 / 15) = 2
    behind = _turn(-170.0)
    assert behind.turn1 == "←←←←" and behind.debug["rule"] == "behind_tiebreak"
    assert _turn(-170.0, config=RouteLabelConfig(behind_direction=0)).turn1 == "→→→→"
    assert _turn(-170.0, previous="right").turn1 == "→→→→"  # hysteresis before the tie-break


def test_a_route_seen_in_the_side_band_turns_toward_it_even_against_the_previous_turn():
    # The route 30 deg left is in the image but outside a pixel window this narrow, so its
    # side is visible: turn towards it, whatever the agent was doing.
    config = RouteLabelConfig(pixel_u_min=300, pixel_u_max=340)
    answer = _turn(30.0, previous="right", config=config)
    assert answer.kind == "turn" and answer.turn1 == "←←" and answer.debug["rule"] == "seen_outside_window"
    assert answer.reason == "route_seen_outside_window"


# --------------------------------------------------------------------------- #
# Agreement: is the model's own answer making progress
# --------------------------------------------------------------------------- #
def test_a_pixel_that_makes_progress_agrees_and_keeps_its_own_text():
    u, v = _floor_pixel(0.0, -2.0)
    row = _label((u, v))
    assert row["agree"] is True and row["kind"] == "anchor" and row["reason"] == "model_pixel_progress"
    assert (row["target_turn1"], row["target_turn2"]) == ("↓", f"{u} {v}")
    assert row["debug"]["model_progress"]["ratio"] == pytest.approx(1.0, abs=0.05)
    # a sideways pixel that still gains: (8 - |(1.5, -6.5)|) / |(1.5, -1.5)| = 0.62
    assert _label(_floor_pixel(1.5, -1.5))["agree"] is True


def test_a_pixel_without_enough_progress_is_corrected_to_the_oracle_answer():
    # The route goes 2 m ahead, then left to the goal at (-6, 0, -2): d = 6.32.
    route = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [-6.0, 0.0, -2.0]])
    goal = [-6.0, 0.0, -2.0]
    row = _label(_floor_pixel(1.5, -2.5), route=route, goal=goal)  # right-forward: away from the goal
    assert row["agree"] is False and row["kind"] == "correction" and row["reason"] == "model_pixel_no_progress"
    assert row["oracle_kind"] == "pixel" and row["target_turn1"] == "↓"
    assert row["target_turn2"] == row["oracle_turn2"] == "%d %d" % _floor_pixel(-1.0, -2.0)
    assert row["debug"]["model_progress"]["ratio"] < 0
    # straight ahead 3 m when the goal lies 60 deg left, out of view: (8 - 7.0) / 3 = 0.33
    goal60 = _bearing_point(60.0, 8.0)
    turn = _label(_floor_pixel(0.0, -3.0), route=straight([0, 0, 0], goal60), goal=goal60)
    assert turn["reason"] == "model_pixel_no_progress" and turn["target_turn1"] == "←←←←"
    assert turn["debug"]["model_progress"]["ratio"] == pytest.approx(0.33, abs=0.03)


def test_a_pixel_with_no_floor_and_no_depth_has_no_target():
    depth = floor_depth([0, 0, 0], FACE_NEG_Z)
    depth[:20] = 10.0  # the sensor's maximum: no return
    row = _label((320, 5), depth=depth)
    assert row["agree"] is False and row["reason"].startswith("model_pixel_no_target")


def test_the_pixel_target_stops_at_what_the_depth_sees_first():
    u, v = _floor_pixel(0.0, -5.0)
    point, how = pixel_floor_target((u, v), position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z,
                                    depth_m=floor_depth([0, 0, 0], FACE_NEG_Z, wall_z=-2.0))
    assert how == "depth" and point[2] == pytest.approx(-2.0, abs=0.02) and point[1] == 0.0
    point, how = pixel_floor_target((u, v), position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, depth_m=None)
    assert how == "floor" and point[2] == pytest.approx(-5.0, abs=0.1)


def test_turns_agree_when_they_turn_toward_the_route_or_follow_the_rule():
    ahead = _label("left")  # the route straight ahead and visible: turning away from it
    assert ahead["agree"] is False and ahead["reason"] == "turned_away_from_visible_route"
    assert ahead["target_turn1"] == "↓"
    left30 = straight([0, 0, 0], _bearing_point(30.0, 8.0))
    goal30 = _bearing_point(30.0, 8.0)
    assert _label("left", route=left30, goal=goal30)["reason"] == "turn_toward_visible_route"
    assert _label("right", route=left30, goal=goal30)["agree"] is False

    hidden = straight([0, 0, 0], _bearing_point(-60.0, 8.0))  # out of view on the right
    goal = _bearing_point(-60.0, 8.0)
    rule = _label("left", previous="left", route=hidden, goal=goal)  # hysteresis says left
    assert rule["agree"] is True and rule["reason"] == "turn_matches_rule"
    shorter = _label("right", previous="left", route=hidden, goal=goal)
    assert shorter["agree"] is True and shorter["reason"] == "turn_toward_route"
    wrong = _label("left", route=hidden, goal=goal)  # no previous turn: the rule is right
    assert wrong["agree"] is False and wrong["reason"] == "turn_wrong_direction" and wrong["target_turn1"] == "→→→→"
    behind = straight([0, 0, 0], _bearing_point(-165.0, 8.0))
    assert _label("right", route=behind, goal=_bearing_point(-165.0, 8.0))["reason"] in (
        "turn_toward_route", "turn_route_behind_either_way")
    assert _label("cancel", route=hidden, goal=goal)["agree"] is False


def test_hysteresis_labels_never_flip_the_executed_turn_when_the_route_is_hidden():
    hidden = straight([0, 0, 0], _bearing_point(-60.0, 8.0))
    goal = _bearing_point(-60.0, 8.0)
    for previous in ("left", "right"):
        for answer in ("up", "stop", (320, 470)):
            row = _label(answer, previous=previous, route=hidden, goal=goal)
            if row["kind"] == "correction" and row["oracle_kind"] == "turn":
                assert row["target_turn1"][0] == {"left": "←", "right": "→"}[previous]


def test_an_oracle_without_an_answer_leaves_a_disagreement_untargeted():
    row = _label("up", wall_z=-0.45)  # route straight ahead, hidden: oracle skip
    assert row["oracle_kind"] == "skip" and row["agree"] is None and row["kind"] == "none"
    assert row["target_turn1"] is None and row["reason"].startswith("oracle_skip:")


def test_rows_are_json_and_read_by_the_trainer():
    for row in (_label(_floor_pixel(0.0, -2.0)), _label("stop"), _label("left", d=2.0), _label("up", wall_z=-0.45)):
        json.dumps(row)
        label = normalize_label({"call_index": 0, **row})
        assert label.skip == (row["kind"] == "none")


def test_config_digest_tracks_every_field():
    assert RouteLabelConfig().digest() == RouteLabelConfig().digest()
    assert RouteLabelConfig(pixel_max_ahead_m=2.5).digest() != RouteLabelConfig().digest()
    assert RouteLabelConfig(hysteresis=False).digest() != RouteLabelConfig().digest()


# --------------------------------------------------------------------------- #
# The labelling script, with a straight-line "navmesh"
# --------------------------------------------------------------------------- #
class FlatWorld:
    def geodesic(self, a, b):
        return euclid(a, b)

    def find_path(self, a, b):
        return straight(a, b)

    def snap(self, p):
        return np.asarray(p, dtype=np.float64)


def _record(root, episode=3, success=1.0):
    reference = [[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 0.0, -8.0]]
    writer = EpisodeRecordWriter(
        root=root, scene_id="flatScene", episode_id=episode,
        episode={"reference_path": reference, "goals": [{"position": reference[-1], "radius": 3.0}],
                 "instruction": "walk ahead", "outcome": {"success": success}},
    )
    poses = [([0.0, 0.0, 0.0], FACE_NEG_Z), ([0.0, 0.0, 0.0], yaw_to_quat_wxyz(math.radians(-100))),
             ([0.0, 0.0, 0.0], yaw_to_quat_wxyz(math.radians(-160))), ([0.0, 0.0, -6.5], FACE_NEG_Z)]
    for step, (position, rotation) in enumerate(poses):
        writer.add_step(step, position, rotation)
        writer.add_front(step, _jpeg((64, 48)))
    calls = [
        {"llm_output": "→→→→", "native_first_output": "→→→→", "native_lookdown_turns": 0, "kind": "native_actions"},
        {"llm_output": "→→→→", "native_first_output": "→→→→", "native_lookdown_turns": 0, "kind": "native_actions"},
        {"llm_output": "320 300", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory"},
        {"llm_output": "←", "native_first_output": "←", "native_lookdown_turns": 0, "kind": "native_actions"},
    ]
    for index, call in enumerate(calls):
        position, rotation = poses[index]
        writer.add_call({"call_index": index, "step_id": index, "history_step_ids": [], **call},
                        lookdown_jpeg=_jpeg(), lookdown_depth_m=floor_depth(position, rotation))
    return writer.commit()


def test_label_script_writes_idempotent_rows_the_trainer_indexes(tmp_path):
    from scripts.tools import label_route_corrections as lr

    path = _record(tmp_path / "records")
    kwargs = dict(label_root=tmp_path / "labels", geometry_for_scene=lambda scene: FlatWorld(),
                  dataset_index=None, config=RouteLabelConfig())
    first = lr.label_record(path, **kwargs)
    assert first["status"] == "labelled" and first["calls"] == 4 and first["success"] is True
    rows = read_labels(label_path_for(path, tmp_path / "labels"))
    assert [r["call_index"] for r in rows] == [0, 1, 2, 3]
    # call 0: facing the route, turning away from it
    assert rows[0]["kind"] == "correction" and rows[0]["reason"] == "turned_away_from_visible_route"
    # call 1: route 100 deg left and hidden, agent was turning right: keep turning right
    assert rows[1]["previous_turn_direction"] == -1 and rows[1]["oracle_turn1"] == "→→→→"
    assert rows[1]["agree"] is True and rows[1]["reason"] == "turn_matches_rule"
    # call 2: route behind (160 deg left), still hysteresis right; walking away is a correction
    assert rows[2]["oracle_turn1"] == "→→→→" and rows[2]["kind"] == "correction"
    assert rows[2]["reason"] == "model_pixel_no_progress"
    assert rows[2]["target_turn1"] == "→→→→" and rows[2]["previous_turn_direction"] == -1
    # call 3: 1.5 m from the goal: line A's
    assert rows[3]["skip"] is True and rows[3]["distance_to_goal"] == pytest.approx(1.5)
    assert all(r["labeler"] == RouteLabelConfig().digest() for r in rows)

    assert lr.label_record(path, **kwargs)["status"] == "exists"
    assert lr.label_record(path, **kwargs, force=True)["status"] == "labelled"
    assert lr.label_record(path, **{**kwargs, "config": RouteLabelConfig(pixel_max_ahead_m=2.5)})["status"] == "labelled"

    refs, stats = index_calls([tmp_path / "records"], [tmp_path / "labels"], split="all", holdout=())
    trainable = {r.call_index: r.label for r in refs if not r.label.skip}
    assert set(trainable) == {0, 1, 2}
    assert trainable[0].agree is False and trainable[1].agree is True


def test_label_script_main_writes_statistics(tmp_path, monkeypatch):
    from scripts.tools import label_route_corrections as lr

    _record(tmp_path / "records", episode=3, success=1.0)
    _record(tmp_path / "records", episode=4, success=0.0)
    monkeypatch.setattr(lr, "RouteNavmeshGeometry", lambda path: FlatWorld())
    argv = ["--record_root", str(tmp_path / "records"), "--label_root", str(tmp_path / "labels"), "--workers", "1"]
    assert lr.main(argv) == 0
    summary = json.loads((tmp_path / "labels" / "label_summary.json").read_text())
    stats = summary["stats"]
    assert summary["status"] == {"labelled": 2} and stats["calls"] == 8 and stats["labelled_calls"] == 6
    assert stats["corrections"]["k"] == 4 and stats["anchors"]["k"] == 2
    assert stats["corrections"]["by_target"] == {"pixel": 2, "turn": 2}
    assert stats["by_distance"]["1-2"] == {"calls": 2, "none": 2}
    flips = stats["turn_labels"]["flip_vs_previous_executed_turn"]
    assert flips["oracle_turn_labels"] == {"k": 0, "n": 4, "rate": 0.0}
    assert flips["model_turns"]["n"] == 2  # call 1 after call 0 (both right), per episode
    assert stats["pixel_labels"]["oracle"]["n"] == 2
    assert stats["pixel_labels"]["oracle"]["ahead_arc_m"]["p50"] == pytest.approx(3.0)
    assert summary["config"]["hysteresis"] is True
    # the same records against themselves: identical statistics
    argv2 = argv + ["--compare_label_root", str(tmp_path / "labels"), "--summary", str(tmp_path / "s2.json")]
    assert lr.main(argv2) == 0
    s2 = json.loads((tmp_path / "s2.json").read_text())
    assert s2["status"] == {"exists": 2}
    assert s2["compare"]["all"]["corrections"] == s2["stats"]["corrections"]


def test_round_one_rows_are_summarised_in_the_same_terms():
    from scripts.tools.label_route_corrections import route_label_stats

    rows = [  # round-1 shaped: no kind, no skip; STOP labels within 2 m
        {"call_index": 0, "distance_to_goal": 6.0, "model_kind": "turn", "model_turn1": "←←", "model_turn2": None,
         "oracle_kind": "turn", "oracle_turn1": "→→", "oracle_turn2": None, "agree": False,
         "target_turn1": "→→", "target_turn2": None, "reason": "turn_wrong_direction", "debug": {"delta_yaw_deg": -30}},
        {"call_index": 1, "distance_to_goal": 5.0, "model_kind": "turn", "model_turn1": "←", "model_turn2": None,
         "oracle_kind": "turn", "oracle_turn1": "→", "oracle_turn2": None, "agree": False,
         "target_turn1": "→", "target_turn2": None, "reason": "turn_wrong_direction", "debug": {}},
        {"call_index": 2, "distance_to_goal": 1.5, "model_kind": "pixel", "model_turn1": "↓", "model_turn2": "300 300",
         "oracle_kind": "stop", "oracle_turn1": "STOP", "oracle_turn2": None, "agree": False,
         "target_turn1": "STOP", "target_turn2": None, "reason": "model_did_not_stop", "debug": {}},
    ]
    stats = route_label_stats([(rows, True)])
    assert stats["corrections"]["by_target"] == {"turn": 2, "stop": 1}
    # call 1's oracle turn is right after the model's left turn: a flip
    assert stats["turn_labels"]["flip_vs_previous_executed_turn"]["oracle_turn_labels"] == {"k": 1, "n": 1, "rate": 1.0}
    beyond = route_label_stats([(rows, True)], min_distance_m=3.0)
    assert beyond["calls"] == 2 and "stop" not in beyond["corrections"]["by_target"]


def test_interndata_reference_reads_the_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from scripts.tools.label_route_corrections import interndata_reference

    scene = tmp_path / "sceneX"
    (scene / "meta").mkdir(parents=True)
    (scene / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["go"], "length": 13}) + "\n")
    (scene / "data" / "chunk-000").mkdir(parents=True)
    # frame t's action is parquet action t + 1: frames 0-3 turn left, then forward 0.25 m per frame
    actions = [-1, 2, 2, 2, 2] + [1] * 8
    poses = []
    for t in range(13):
        m = np.eye(4)
        m[0, 3] = 0.25 * max(0, t - 4)
        poses.append(m.tolist())
    goals = [[-1, -1]] * 4 + [[320, 200]] * 9
    rel = [0] * 4 + [8, 7, 6, 5, 4, 3, 2, 1, 0]
    pq.write_table(pa.table({"action": actions, "pose.125cm_0deg": poses, "goal.125cm_30deg": goals,
                             "relative_goal_frame_id.125cm_30deg": rel}),
                   scene / "data" / "chunk-000" / "episode_000000.parquet")
    ref = interndata_reference(tmp_path, workers=1)
    assert ref["turn"]["n"] == 1 and ref["turn"]["arrows"]["4"] == 1.0
    assert ref["pixel"]["n"] == 2  # frames 4 and 8 (rel 8 and 4); frame 12 is the last
    assert ref["pixel"]["ahead_euclid_m"]["p50"] == pytest.approx(1.5)
    assert ref["pixel"]["u_outside_100_540"]["k"] == 0
