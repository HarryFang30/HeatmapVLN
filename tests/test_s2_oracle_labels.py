"""EXP-23 part B: oracle answers and agreement on a flat synthetic world.

Geodesics are straight lines and the lookdown depth is the analytic depth of
the floor plane (optionally cut by a wall), so every expected answer can be
worked out by hand.
"""

import io
import json
import math

import numpy as np
import pytest
from PIL import Image

from src.data.s2_oracle_labels import (
    LookdownCamera,
    ModelAnswer,
    OracleConfig,
    choose_waypoint,
    heading_delta_deg,
    label_call,
    last_reached_by_step,
    nearest_on_polyline,
    oracle_answer,
    parse_model_answer,
    parse_native_tokens,
    pixel_text,
    remaining_reference_m,
    resample_polyline,
    route_points,
    route_waypoints,
    yaw_to_quat_wxyz,
)
from src.data.s2_records import CAMERA, EpisodeRecordWriter, LABEL_SCHEMA, label_path_for, read_labels

CAM = LookdownCamera()
FACE_NEG_Z = yaw_to_quat_wxyz(0.0)


def euclid(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def straight(a, b):
    return np.stack([np.asarray(a, float), np.asarray(b, float)])


def floor_depth(position, rotation, *, floor_y=0.0, wall_z=None):
    """Analytic z-depth of the lookdown on the floor y = floor_y (+ a wall z = wall_z)."""
    c2w = CAM.c2w(position, rotation)
    v, u = np.mgrid[0 : CAM.height, 0 : CAM.width].astype(np.float64)
    cx, cy = CAM.centre
    rays = np.stack([(u - cx) / CAM.focal, -(v - cy) / CAM.focal, -np.ones_like(u)], -1) @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rays[..., 1] < 0, (floor_y - origin[1]) / rays[..., 1], np.inf)
        if wall_z is not None:
            tw = np.where(rays[..., 2] < 0, (wall_z - origin[2]) / rays[..., 2], np.inf)
            t = np.minimum(t, tw)
    # rays have unit camera-z, so the ray parameter is the z-depth
    return np.clip(np.where(np.isfinite(t), t, CAMERA["depth_max_m"]), 0.0, CAMERA["depth_max_m"]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def test_camera_matches_the_deployed_sensor():
    assert (CAM.width, CAM.height, CAM.hfov_deg, CAM.height_m, CAM.pitch_deg) == (640, 480, 79.0, 1.25, 30.0)
    assert CAM.focal == pytest.approx(320.0 / math.tan(math.radians(39.5)))
    # Same intrinsics as scripts/exp19/gt.py::eval_intrinsics, in pixel index coordinates.
    assert CAM.centre == (319.5, 239.5)


def test_floor_point_on_the_optical_axis_projects_to_the_image_centre():
    # Camera 1.25 m up, pitched 30 deg down: the axis meets the floor 1.25 / tan 30 ahead.
    ahead = 1.25 / math.tan(math.radians(30.0))
    uv, z = CAM.project(np.array([[0.0, 0.0, -ahead]]), CAM.c2w([0.0, 0.0, 0.0], FACE_NEG_Z))
    np.testing.assert_allclose(uv[0], CAM.centre, atol=1e-6)
    assert z[0] == pytest.approx(math.hypot(ahead, 1.25))


def test_left_is_left_and_yaw_turns_the_view():
    c2w = CAM.c2w([0.0, 0.0, 0.0], FACE_NEG_Z)
    uv, _ = CAM.project(np.array([[-1.0, 0.0, -3.0], [1.0, 0.0, -3.0]]), c2w)
    assert uv[0, 0] < CAM.centre[0] < uv[1, 0]  # -x is left when facing -z
    # Turned 90 deg left the agent faces -x, so a point at -x is straight ahead.
    turned = CAM.c2w([0.0, 0.0, 0.0], yaw_to_quat_wxyz(math.pi / 2))
    uv, _ = CAM.project(np.array([[-2.165, 0.0, 0.0]]), turned)
    assert uv[0, 0] == pytest.approx(CAM.centre[0], abs=1e-6)


def test_unproject_inverts_project_with_z_depth():
    c2w = CAM.c2w([1.0, 0.2, -3.0], yaw_to_quat_wxyz(0.7))
    pts = np.array([[0.0, 0.2, -6.0], [2.5, 0.2, -5.0], [1.3, 1.1, -5.5]])
    uv, z = CAM.project(pts, c2w)
    for p, (u, v), zz in zip(pts, uv, z):
        np.testing.assert_allclose(CAM.unproject(u, v, zz, c2w), p, atol=1e-9)


def test_heading_delta_is_left_positive():
    assert heading_delta_deg(FACE_NEG_Z, [0, 0, 0], [-1, 0, 0]) == pytest.approx(90.0)
    assert heading_delta_deg(FACE_NEG_Z, [0, 0, 0], [1, 0, 0]) == pytest.approx(-90.0)
    assert heading_delta_deg(yaw_to_quat_wxyz(math.pi / 2), [0, 0, 0], [-1, 0, -1]) == pytest.approx(-45.0)


# --------------------------------------------------------------------------- #
# Text formats
# --------------------------------------------------------------------------- #
def test_pixel_text_is_column_then_row_like_the_released_model():
    # The released model answers "u v": its first number reaches 601 on a
    # 640x480 lookdown, and InternData-N1 prints goal [495, 345] as "495 345".
    assert pixel_text(495, 345) == "495 345"
    answer = parse_model_answer(
        {"llm_output": "601 57", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory"}
    )
    assert answer.kind == "pixel" and answer.pixel == (601, 57)
    assert (answer.turn1, answer.turn2) == ("↓", "601 57")


def test_the_swapped_response_pixel_goal_is_ignored():
    # The server's pixel_goal is [second, first] of the text, i.e. [v, u].
    answer = parse_model_answer(
        {"llm_output": "601 57", "native_first_output": "↓", "native_lookdown_turns": 1, "pixel_goal": [57, 601]}
    )
    assert answer.pixel == (601, 57)


def test_projection_reproduces_an_interndata_n1_goal_label():
    # InternData-N1 r2r gZ6f7yhEvPG episode 0: frame 0's 30 deg lookdown labels the
    # floor under frame 12's camera as goal [495, 345].  Their world is z-up,
    # x forward, y left, origin at the start camera's floor; Habitat is y-up,
    # -z forward, x right: (x, y, z)_hab = (-y, z, -x).
    camera = LookdownCamera()
    floor = np.array([1.32, -0.82, 1.16 - 1.25])  # frame 12 camera minus its 1.25 m height
    world = np.array([[-floor[1], floor[2], -floor[0]]])
    uv, z = camera.project(world, camera.c2w([0.0, 0.0, 0.0], FACE_NEG_Z))
    assert z[0] > 0
    assert abs(uv[0, 0] - 495) < 3 and abs(uv[0, 1] - 345) < 3


def test_native_token_parse_matches_the_deployed_parser():
    from src.models.heatmap.native_internnav_exact import NATIVE_ACTIONS, parse_native_actions

    code = {token: codes[0] for token, codes in NATIVE_ACTIONS.items()}
    for text in ("STOP", "←←←←", "→", "←→→→", "↓", "↑↑←", "hello", "", "→STOP←"):
        assert [code[t] for t in parse_native_tokens(text)] == parse_native_actions(text)


@pytest.mark.parametrize(
    "call, kind, direction",
    [
        ({"llm_output": "STOP", "native_first_output": "STOP"}, "stop", 0),
        ({"llm_output": "←←←←"}, "turn", 1),
        ({"llm_output": "→"}, "turn", -1),
        ({"llm_output": "←→→→"}, "turn", -1),  # executes L R R R: net right
        ({"llm_output": "←←→→"}, "turn", 0),
        ({"llm_output": "←←←←→→→→→"}, "turn", 1),  # only four actions run
        ({"llm_output": "↑↑"}, "other", 0),
        ({"llm_output": "hmm"}, "other", 0),  # the server's fallback_stop
        ({"llm_output": "←", "native_first_output": "↓", "native_lookdown_turns": 1}, "turn", 1),
    ],
)
def test_model_answer_kinds(call, kind, direction):
    answer = parse_model_answer(call)
    assert answer.kind == kind
    assert answer.turn_direction == direction


# --------------------------------------------------------------------------- #
# Oracle route
# --------------------------------------------------------------------------- #
WPS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -3.0], [3.0, 0.0, -3.0], [3.0, 0.0, -6.0]])


def test_reached_is_cumulative_and_geodesic_within_radius():
    visited = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -1.5], [0.4, 0.0, -2.5], [1.5, 0.0, -2.0]])
    np.testing.assert_array_equal(last_reached_by_step(visited, WPS, euclid, 1.0), [0, 0, 1, 1])


def test_literal_argmin_is_the_goal_and_slack_follows_the_reference():
    # A reference that detours: going straight to the goal is always cheaper.
    remaining = remaining_reference_m(WPS, euclid)
    np.testing.assert_allclose(remaining, [9.0, 6.0, 3.0, 0.0])
    p = np.array([0.0, 0.0, -1.0])
    k_literal, costs = choose_waypoint(p, WPS, 0, remaining, euclid, slack_m=0.0)
    assert k_literal == 3  # geo(p, goal) <= geo(p, w_k) + remaining(w_k) by the triangle inequality
    assert costs[1] == pytest.approx(8.0) and costs[3] == pytest.approx(math.hypot(3.0, 5.0))
    k, _ = choose_waypoint(p, WPS, 0, remaining, euclid, slack_m=2.5)
    assert k == 1


def test_missed_waypoint_is_abandoned_when_going_back_costs_more_than_the_slack():
    remaining = remaining_reference_m(WPS, euclid)
    p = np.array([3.0, 0.0, -4.5])  # walked past w1 and w2 without reaching them
    k, costs = choose_waypoint(p, WPS, 0, remaining, euclid, slack_m=2.0)
    assert k == 3 and costs[1] - costs[3] > 2.0


def test_all_reached_leaves_the_last_waypoint():
    k, costs = choose_waypoint(np.array([3.0, 0.0, -5.5]), WPS, 3, remaining_reference_m(WPS, euclid), euclid)
    assert k == 3 and list(costs) == [3]


def test_route_points_joins_find_path_and_reference_segments():
    route = route_points(np.array([0.0, 0.0, -1.0]), WPS, 1, straight)
    np.testing.assert_allclose(route, [[0, 0, -1], [0, 0, -3], [3, 0, -3], [3, 0, -6]])
    assert route_points(np.zeros(3), WPS, 1, lambda a, b: None) is None


def test_goal_is_appended_only_when_it_is_not_the_last_waypoint():
    assert len(route_waypoints(WPS, WPS[-1])) == 4
    assert len(route_waypoints(WPS, [9.0, 0.0, -9.0])) == 5


def test_resample_and_nearest_point():
    samples, arc = resample_polyline(np.array([[0, 0, 0], [0, 0, -1.0], [0, 0, -1.0], [1, 0, -1.0]]), 0.3)
    np.testing.assert_allclose(arc, [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    np.testing.assert_allclose(samples[-1], [1, 0, -1])
    dist, along = nearest_on_polyline(np.array([[0, 0, 0], [0, 0, -2.0], [2, 0, -2.0]]), [1.0, 0.0, -1.5])
    assert dist == pytest.approx(0.5) and along == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Oracle answers
# --------------------------------------------------------------------------- #
def _oracle(route, position=(0.0, 0.0, 0.0), rotation=FACE_NEG_Z, d=5.0, **depth_kwargs):
    depth = floor_depth(position, rotation, **depth_kwargs)
    return oracle_answer(
        position=position, rotation_wxyz=rotation, distance_to_goal=d, route=np.asarray(route, float), depth_m=depth
    )


def test_stop_inside_the_stop_radius():
    answer = oracle_answer(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, distance_to_goal=1.9, route=None, depth_m=None)
    assert (answer.kind, answer.turn1, answer.turn2) == ("stop", "STOP", None)


def test_straight_route_gives_the_farthest_visible_floor_point():
    answer = _oracle(straight([0, 0, 0], [0, 0, -6.0]))
    assert answer.kind == "pixel" and answer.turn1 == "↓"
    u, v = answer.pixel
    assert answer.turn2 == f"{u} {v}"
    assert abs(u - CAM.centre[0]) <= 1  # straight ahead: the centre column
    assert answer.debug["visible_arc_m"] == pytest.approx(6.0)
    expected_uv, _ = CAM.project(np.array([[0.0, 0.0, -6.0]]), CAM.c2w([0, 0, 0], FACE_NEG_Z))
    assert (u, v) == tuple(int(math.floor(x + 0.5)) for x in expected_uv[0])


def test_wall_occludes_the_route_behind_it():
    answer = _oracle(straight([0, 0, 0], [0, 0, -6.0]), wall_z=-4.0)
    assert answer.kind == "pixel"
    # Samples up to the wall foot (+ the 0.3 m tolerance) stay visible.
    assert 3.9 <= answer.debug["visible_arc_m"] <= 4.3


def test_route_straight_ahead_but_hidden_is_a_skip():
    # The lookdown sees the floor from ~0.67 m on; a wall at 0.45 m hides all of it.
    answer = _oracle(straight([0, 0, 0], [0, 0, -6.0]), wall_z=-0.45)
    assert answer.kind == "skip" and answer.reason == "route_ahead_but_occluded"


@pytest.mark.parametrize(
    "target, turn1",
    [
        ([-6.0, 0.0, 0.0], "←←←←"),  # 90 deg left -> 6 steps, capped at 4
        ([6.0, 0.0, 0.0], "→→→→"),
        ([-1.0, 0.0, 6.0], "←←←←"),  # behind, slightly left
    ],
)
def test_route_out_of_view_turns_towards_it(target, turn1):
    answer = _oracle(straight([0, 0, 0], target))
    assert (answer.kind, answer.turn1, answer.turn2) == ("turn", turn1, None)


def test_turn_count_rounds_the_angle_to_15_degree_steps():
    # Route 35 deg to the right, hidden behind a wall right in front of the agent.
    a = math.radians(-35.0)
    target = [-6.0 * math.sin(a), 0.0, -6.0 * math.cos(a)]
    answer = _oracle(straight([0, 0, 0], target), wall_z=-0.45)
    assert answer.kind == "turn" and answer.turn1 == "→→"  # round(35 / 15) = 2
    assert answer.debug["delta_yaw_deg"] == pytest.approx(-35.0, abs=0.01)


def test_missing_depth_or_route_is_a_skip():
    common = dict(position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, distance_to_goal=5.0)
    assert oracle_answer(route=None, depth_m=None, **common).reason == "no_route"
    assert oracle_answer(route=straight([0, 0, 0], [0, 0, -5]), depth_m=None, **common).reason == "no_depth"
    assert oracle_answer(route=None, depth_m=None, **{**common, "distance_to_goal": math.inf}).reason == "goal_unreachable"


# --------------------------------------------------------------------------- #
# Agreement and targets
# --------------------------------------------------------------------------- #
def _label(model_call, *, d=5.0, route=None, wall_z=None):
    route = straight([0, 0, 0], [0, 0, -6.0]) if route is None else np.asarray(route, float)
    return label_call(
        model=parse_model_answer(model_call),
        position=[0, 0, 0],
        rotation_wxyz=FACE_NEG_Z,
        distance_to_goal=d,
        route=route,
        depth_m=floor_depth([0, 0, 0], FACE_NEG_Z, wall_z=wall_z),
    )


def _pixel_call(u, v):
    return {"llm_output": f"{u} {v}", "native_first_output": "↓", "native_lookdown_turns": 1, "pixel_goal": [v, u]}


def _floor_pixel(x, z):
    uv, _ = CAM.project(np.array([[x, 0.0, z]]), CAM.c2w([0, 0, 0], FACE_NEG_Z))
    return tuple(int(math.floor(c + 0.5)) for c in uv[0])


def test_stop_agrees_inside_the_success_radius_and_keeps_its_own_text():
    row = _label({"llm_output": "STOP", "native_first_output": "STOP"}, d=2.6)
    assert row["agree"] is True and (row["target_turn1"], row["target_turn2"]) == ("STOP", None)
    assert row["oracle_kind"] == "pixel"  # outside the 2 m stop radius the oracle walks on


def test_stop_too_far_is_replaced_by_the_oracle():
    row = _label({"llm_output": "STOP"}, d=3.5)
    assert row["agree"] is False and row["reason"] == "model_stop_too_far"
    assert row["target_turn1"] == "↓" and row["target_turn2"] == row["oracle_turn2"]


def test_walking_on_inside_the_stop_radius_becomes_stop():
    u, v = _floor_pixel(0.0, -3.0)
    row = _label(_pixel_call(u, v), d=1.5)
    assert row["agree"] is False and row["reason"] == "model_did_not_stop"
    assert (row["target_turn1"], row["target_turn2"]) == ("STOP", None)


def test_pixel_on_the_route_agrees_and_keeps_the_models_text():
    u, v = _floor_pixel(0.6, -3.0)  # 0.6 m beside the route, 3 m along it
    row = _label(_pixel_call(u, v))
    assert row["agree"] is True and row["reason"] == "model_pixel_on_route"
    assert (row["target_turn1"], row["target_turn2"]) == ("↓", f"{u} {v}")
    assert row["debug"]["model_q_route_dist_m"] == pytest.approx(0.6, abs=0.02)


def test_pixel_off_the_route_or_without_progress_disagrees():
    off = _label(_pixel_call(*_floor_pixel(1.5, -3.0)))
    assert off["agree"] is False and off["reason"] == "model_pixel_off_route"
    assert off["target_turn2"] == off["oracle_turn2"]
    # Progress is counted along the route from its start: the floor 0.8 m ahead is
    # 1.3 m along a route that starts 0.5 m behind the agent, and 0 m along one that
    # leads away behind it (slightly left, so the oracle's turn is unambiguous).
    near = _label(_pixel_call(*_floor_pixel(0.0, -0.8)), route=straight([0, 0, 0.5], [0, 0, -6.0]))
    assert near["agree"] is True
    behind = _label(_pixel_call(*_floor_pixel(0.0, -0.8)), route=straight([0, 0, 0], [-1.0, 0, 6.0]))
    assert behind["agree"] is False and behind["reason"] == "model_pixel_no_progress"
    assert behind["oracle_turn1"] == "←←←←" and behind["target_turn1"] == "←←←←"


def test_pixel_above_the_horizon_disagrees():
    # Row 5 looks above the horizon: the depth saturates at 10 m, far off the route.
    row = _label(_pixel_call(320, 5))
    assert row["agree"] is False and row["reason"] == "model_pixel_off_route"


def test_pixel_without_a_depth_return_disagrees():
    left_route = straight([0, 0, 0], [-6.0, 0.0, 0.0])
    row = label_call(
        model=parse_model_answer(_pixel_call(320, 400)),
        position=[0, 0, 0],
        rotation_wxyz=FACE_NEG_Z,
        distance_to_goal=5.0,
        route=left_route,
        depth_m=np.zeros((480, 640), np.float32),
    )
    assert row["agree"] is False and row["reason"] == "model_pixel_no_depth"
    assert row["target_turn1"] == "←←←←"


def test_turn_agreement_needs_the_same_direction():
    left_route = straight([0, 0, 0], [-6.0, 0.0, 0.0])
    same = _label({"llm_output": "←←"}, route=left_route)
    assert same["oracle_turn1"] == "←←←←" and same["agree"] is True and same["target_turn1"] == "←←"
    wrong = _label({"llm_output": "→→→→"}, route=left_route)
    assert wrong["agree"] is False and wrong["target_turn1"] == "←←←←"
    cancelling = _label({"llm_output": "←←→→"}, route=left_route)
    assert cancelling["agree"] is False


def test_turning_with_the_route_in_view_disagrees():
    row = _label({"llm_output": "←←←←"})
    assert row["oracle_kind"] == "pixel" and row["agree"] is False and row["reason"] == "turned_with_route_in_view"
    assert row["target_turn1"] == "↓"


def test_oracle_skip_keeps_only_answers_judged_without_it():
    route = straight([0, 0, 0], [0, 0, -6.0])
    turn = _label({"llm_output": "←"}, route=route, wall_z=-0.45)
    assert turn["oracle_kind"] == "skip" and turn["agree"] is None and turn["target_turn1"] is None
    stop = _label({"llm_output": "STOP"}, d=2.5, route=route, wall_z=-0.45)
    assert stop["agree"] is True and stop["target_turn1"] == "STOP"
    far_stop = _label({"llm_output": "STOP"}, d=4.0, route=route, wall_z=-0.45)
    assert far_stop["agree"] is None and far_stop["target_turn1"] is None


def test_other_answers_never_agree():
    row = _label({"llm_output": "↑↑↑"})
    assert row["model_kind"] == "other" and row["agree"] is False and row["target_turn1"] == "↓"


def test_label_rows_are_json_serialisable():
    row = _label(_pixel_call(*_floor_pixel(0.0, -3.0)))
    json.dumps(row)


def test_config_digest_tracks_every_field():
    assert OracleConfig().digest() == OracleConfig().digest()
    assert OracleConfig(stop_radius_m=1.5).digest() != OracleConfig().digest()
    assert OracleConfig(rejoin_slack_m=0.0).digest() != OracleConfig().digest()


# --------------------------------------------------------------------------- #
# The labelling script, with a straight-line "navmesh"
# --------------------------------------------------------------------------- #
class FlatWorld:
    def geodesic(self, a, b):
        return euclid(a, b)

    def find_path(self, a, b):
        return straight(a, b)


def _jpeg(size=(640, 480)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def _record(tmp_path):
    reference = [[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 0.0, -8.0]]
    writer = EpisodeRecordWriter(
        root=tmp_path / "records",
        scene_id="flatScene",
        episode_id=3,
        episode={"reference_path": reference, "goals": [{"position": reference[-1], "radius": 3.0}]},
    )
    poses = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -6.5]]
    for step, position in enumerate(poses):
        writer.add_step(step, position, FACE_NEG_Z)
        writer.add_front(step, _jpeg((64, 48)))
    u, v = _floor_pixel(0.0, -3.0)
    calls = [
        ({"llm_output": f"{u} {v}", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory",
          "pixel_goal": [v, u], "distance_to_goal": 8.0}, 0),
        ({"llm_output": "→→", "native_first_output": "→→", "native_lookdown_turns": 0, "kind": "native_actions"}, 1),
        ({"llm_output": "←", "native_first_output": "←", "native_lookdown_turns": 0, "kind": "native_actions"}, 2),
    ]
    for index, (call, step) in enumerate(calls):
        position = poses[step]
        writer.add_call(
            {"call_index": index, "step_id": step, "history_step_ids": [], **call},
            lookdown_jpeg=_jpeg(),
            lookdown_depth_m=floor_depth(position, FACE_NEG_Z),
        )
    return writer.commit()


def test_label_script_writes_idempotent_schema_rows(tmp_path):
    from scripts.tools import label_s2_records as ls

    path = _record(tmp_path)
    config = OracleConfig()
    kwargs = dict(
        label_root=tmp_path / "labels", geometry_for_scene=lambda scene: FlatWorld(), dataset_index=None, config=config
    )
    first = ls.label_record(path, **kwargs, viz_dir=tmp_path / "viz", viz_every=1)
    assert first["status"] == "labelled" and first["calls"] == 3
    out = label_path_for(path, tmp_path / "labels")
    rows = read_labels(out)
    assert [r["schema"] for r in rows] == [LABEL_SCHEMA] * 3
    assert [r["call_index"] for r in rows] == [0, 1, 2]
    assert rows[0]["agree"] is True and rows[0]["oracle_kind"] == "pixel" and rows[0]["recorded_distance_to_goal"] == 8.0
    assert rows[0]["debug"]["k_star"] == 1 and rows[0]["debug"]["last_reached"] == 0
    assert rows[1]["reason"] == "turned_with_route_in_view" and rows[1]["target_turn1"] == "↓"
    assert rows[2]["distance_to_goal"] == pytest.approx(1.5) and rows[2]["target_turn1"] == "STOP"
    assert rows[2]["debug"]["last_reached"] == 0  # w1 (z=-4) was never within 1 m
    assert len(list((tmp_path / "viz").glob("*.png"))) == 3

    assert ls.label_record(path, **kwargs)["status"] == "exists"
    assert ls.label_record(path, **kwargs, force=True)["status"] == "labelled"
    changed = ls.label_record(path, **{**kwargs, "config": OracleConfig(stop_radius_m=1.0)})
    assert changed["status"] == "labelled"

    summary = ls.summarize_labels([out])
    assert summary["num_calls"] == 3 and summary["by_model_kind"] == {"pixel": 1, "turn": 2}


def test_label_script_main_writes_a_summary_and_reports_failures(tmp_path, monkeypatch):
    from scripts.tools import label_s2_records as ls

    _record(tmp_path)
    broken = EpisodeRecordWriter(root=tmp_path / "records", scene_id="flatScene", episode_id=4, episode={})
    broken.add_front(0, _jpeg((64, 48)))
    broken.add_step(0, [0.0, 0.0, 0.0], FACE_NEG_Z)
    broken.add_call(
        {"call_index": 0, "step_id": 0, "history_step_ids": [], "llm_output": "STOP", "kind": "stop"},
        lookdown_jpeg=_jpeg(),
        lookdown_depth_m=None,
    )
    broken.commit()  # no reference path anywhere
    monkeypatch.setattr(ls, "NavmeshGeometry", lambda path: FlatWorld())
    argv = ["--record_root", str(tmp_path / "records"), "--label_root", str(tmp_path / "labels"), "--workers", "1"]
    assert ls.main(argv) == 1
    summary = json.loads((tmp_path / "labels" / "label_summary.json").read_text())
    assert summary["status"] == {"labelled": 1, "failed": 1}
    assert "no reference path" in summary["failures"][0]["error"]
    assert summary["num_calls"] == 3 and summary["config"]["rejoin_slack_m"] == 2.0
    ls.main(argv)
    summary = json.loads((tmp_path / "labels" / "label_summary.json").read_text())
    assert summary["status"] == {"exists": 1, "failed": 1}


def test_episode_geometry_prefers_the_split_file_and_checks_agreement():
    from scripts.tools.label_s2_records import resolve_episode_geometry

    entry = {"reference_path": [[0, 0, 0], [0, 0, -4]], "goal": [0, 0, -4], "start_position": [0, 0, 0]}
    ref, goal, start = resolve_episode_geometry({}, entry)
    np.testing.assert_allclose(goal, [0, 0, -4])
    np.testing.assert_allclose(start, [0, 0, 0])
    ref, goal, _ = resolve_episode_geometry({"reference_path": [[0, 0, 0], [0, 0, -4]], "goal_position": [0, 0, -4]}, None)
    assert ref.shape == (2, 3)
    with pytest.raises(ValueError, match="disagrees"):
        resolve_episode_geometry({"goals": [{"position": [9, 0, 9]}]}, entry)
    with pytest.raises(ValueError, match="no reference path"):
        resolve_episode_geometry({}, None)


def test_label_rows_carry_the_oracle_route_for_gate_g4():
    u, v = _floor_pixel(0.6, -3.0)
    row = _label(_pixel_call(u, v))
    path = np.asarray(row["oracle_path"])
    assert path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2
    assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) <= 0.25 + 1e-6)


# --------------------------------------------------------------------------- #
# Pixel targets stay inside the released model's training window
# --------------------------------------------------------------------------- #
def test_a_route_leaving_sideways_gets_a_pixel_inside_the_window_not_on_the_border():
    # 1 m ahead the route turns hard left and leaves the frame through its left edge.
    route = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-6.0, 0.0, -1.0]])
    answer = _oracle(route)
    assert answer.kind == "pixel"
    u, v = answer.pixel
    cfg = OracleConfig()
    assert cfg.pixel_margin_u <= u <= CAM.width - 1 - cfg.pixel_margin_u
    assert cfg.pixel_min_row <= v <= cfg.pixel_max_row
    assert answer.debug["n_visible_in_window"] < answer.debug["n_visible"]  # the border band was there


def test_a_route_visible_only_outside_the_window_turns_like_the_expert_data():
    # Nothing passes a window this tight, so the visible route cannot become a pixel:
    # the oracle turns towards it instead (left, the way the route goes).
    route = np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, -3.0]])
    depth = floor_depth([0, 0, 0], FACE_NEG_Z)
    answer = oracle_answer(
        position=[0, 0, 0], rotation_wxyz=FACE_NEG_Z, distance_to_goal=5.0, route=route, depth_m=depth,
        config=OracleConfig(pixel_max_row=0),
    )
    assert answer.kind == "turn" and set(answer.turn1) == {"←"}


def test_the_config_digest_moves_with_the_rules_version(monkeypatch):
    import src.data.s2_oracle_labels as labels

    before = OracleConfig().digest()
    monkeypatch.setattr(labels, "LABEL_RULES_VERSION", labels.LABEL_RULES_VERSION + 1)
    assert OracleConfig().digest() != before
