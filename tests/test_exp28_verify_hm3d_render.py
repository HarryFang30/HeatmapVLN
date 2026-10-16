"""EXP-28 V0 (scripts/exp28/verify_hm3d_render.py): pose mapping, NCC, matching, verdict.

The mapping is pinned to the r2r goal label of tests/test_s2_oracle_labels.py (the InternData-N1 convention
the rest of the repo uses).  The end-to-end tests run the whole check on a synthetic world: every scene is a
panorama, a render is the window the body yaw looks at, InternData frames are cut from the same panorama at
the true headings and their poses are written in InternData's frame -- so a correct pipeline scores NCC ~1
and a json start heading off by 90 deg fails.  The client-config test needs habitat-lab; the render test
also needs an X display and ``EXP28_V0_RENDER_SCENE`` (a .glb with its navmesh).
"""

import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.exp28 import verify_hm3d_render as v0

ORL = v0.ORL


def xyzw(yaw_rad):
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


def rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# Names and instructions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        "hm3d/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb",
        "data/scene_datasets/hm3d/train/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb",
        "hm3d/5jLhtVmWd5F/5jLhtVmWd5F.glb",
        "00481-5jLhtVmWd5F",
        "/workspace/exp23/data/interndata_n1/scalevln/00481-5jLhtVmWd5F/",
        "5jLhtVmWd5F.basis.navmesh",
        "5jLhtVmWd5F",
    ],
)
def test_hm3d_id(name):
    assert v0.hm3d_id(name) == "5jLhtVmWd5F"


def test_match_key_is_the_client_text_up_to_case_and_spacing():
    # R2R-style ". " endings: the client keeps them, InternData may not; the key does not care.
    assert v0.match_key("Walk past the sofa. ") == v0.match_key("walk  past the sofa") == "walk past the sofa"
    assert v0.match_key("Go up the stairs!") == "go up the stairs"
    assert v0.match_key("Turn left.") != v0.match_key("Turn right.")


# --------------------------------------------------------------------------- #
# Pose mapping
# --------------------------------------------------------------------------- #
def test_mapping_reproduces_the_interndata_goal_label():
    # tests/test_s2_oracle_labels.py: InternData-N1 r2r gZ6f7yhEvPG episode 0, frame 0's 30 deg lookdown labels
    # the floor under frame 12's camera, (1.32, -0.82, 1.16) in InternData's frame, as goal [495, 345].
    floor = v0.interndata_to_habitat([[1.32, -0.82, 1.16]], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    assert floor[0] == pytest.approx([0.82, 1.16 - 1.25, -1.32])
    camera = ORL.LookdownCamera()
    uv, z = camera.project(floor, camera.c2w([0.0, 0.0, 0.0], ORL.yaw_to_quat_wxyz(0.0)))
    assert z[0] > 0 and abs(uv[0, 0] - 495) < 3 and abs(uv[0, 1] - 345) < 3


@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 90.0, -120.0, 180.0])
def test_the_label_survives_any_start_pose(yaw_deg):
    start, yaw = [3.0, 0.4, -7.5], math.radians(yaw_deg)
    floor = v0.interndata_to_habitat([[1.32, -0.82, 1.16]], start, xyzw(yaw))
    camera = ORL.LookdownCamera()
    uv, _ = camera.project(floor, camera.c2w(start, ORL.yaw_to_quat_wxyz(yaw)))
    assert abs(uv[0, 0] - 495) < 3 and abs(uv[0, 1] - 345) < 3


def test_mapping_axes_after_a_left_turn():
    # Facing -x (turned 90 deg left from -z): InternData forward goes to -x, left to +z, up to +y.
    out = v0.interndata_to_habitat([[1.0, 0.0, 1.25], [0.0, 1.0, 1.25], [0.0, 0.0, 2.25]], [0.0, 0.0, 0.0],
                                   xyzw(math.pi / 2))
    assert out == pytest.approx(np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), abs=1e-9)


def test_inverse_mapping_round_trips():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(20, 3))
    start, rot = [1.0, 0.2, -3.0], xyzw(0.7)
    back = v0.interndata_to_habitat(v0.habitat_to_interndata(points, start, rot), start, rot)
    assert back == pytest.approx(points, abs=1e-9)
    assert v0.habitat_to_interndata([start], start, rot)[0] == pytest.approx([0.0, 0.0, 1.25])


def test_mapping_variants_include_the_design_once():
    t = np.array([[1.0, 2.0, 1.25]])
    design = v0.interndata_to_habitat(t, [0, 0, 0], [0, 0, 0, 1])
    others = [v0.interndata_to_habitat(t, [0, 0, 0], [0, 0, 0, 1], variant=name)
              for name in v0.MAPPING_VARIANTS if not name.startswith("design")]
    assert len(others) == 7
    assert all(not np.allclose(o, design) for o in others)
    assert len({tuple(np.round(o[0], 6)) for o in others}) == 7


def test_relative_yaws_do_not_depend_on_the_camera_axes():
    # Any fixed camera convention C: R_t = Rz(theta_t) C; the yaw relative to frame 0 is theta_t - theta_0.
    c = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # OpenCV-like camera in a z-up world
    thetas = np.radians([10.0, 25.0, -50.0, 170.0])
    yaw, tilt = v0.relative_yaws(np.stack([rz(t) @ c for t in thetas]))
    assert yaw == pytest.approx(v0.wrap_rad(thetas - thetas[0]), abs=1e-9)
    assert tilt.max() < 1e-5
    pitched = np.stack([np.eye(3), np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])])
    assert v0.relative_yaws(pitched)[1][1] == pytest.approx(math.degrees(math.acos(0.8)))


def test_action_yaws_left_positive():
    # parquet action t led to frame t; 2 = left, 3 = right, 1 forward, 5 look down.
    assert np.degrees(v0.action_yaws([-1, 2, 2, 1, 3, 5, 0])) == pytest.approx([0, 15, 30, 30, 15, 15, 15])


def test_heading_rotation_composes_with_the_start():
    a, b = math.radians(40.0), math.radians(-75.0)
    q = v0.heading_rotation_wxyz(xyzw(a), b)
    assert ORL.quat_wxyz_to_rot(q) == pytest.approx(ORL.quat_wxyz_to_rot(ORL.yaw_to_quat_wxyz(a + b)), abs=1e-9)
    assert v0.yaw_of_wxyz(q) == pytest.approx(a + b)
    assert v0.yaw_of_wxyz(ORL.yaw_to_quat_wxyz(2.5)) == pytest.approx(2.5)


def test_camera_c2w_raises_the_sensor_along_the_body_up():
    c2w = v0.camera_c2w([1.0, 2.0, 3.0], ORL.yaw_to_quat_wxyz(0.3))
    assert c2w[:3, 3] == pytest.approx([1.0, 3.25, 3.0])


def test_point_to_polyline_and_mid_frames():
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [4.0, 0.0, -4.0]])
    d = v0.point_to_polyline([[1.0, 0.0, -2.0], [2.0, 0.0, -5.0], [0.0, 0.0, 1.0]], line)
    assert d == pytest.approx([1.0, 1.0, 1.0])
    assert v0.mid_frame_ids(41, 3) == [10, 20, 30]
    assert v0.mid_frame_ids(3, 3) == [1]
    assert v0.mid_frame_ids(2, 3) == []


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def test_ncc_properties():
    rng = np.random.default_rng(0)
    a = rng.random((120, 160))
    assert v0.ncc(a, a) == pytest.approx(1.0)
    assert v0.ncc(a, 3.0 * a + 7.0) == pytest.approx(1.0)
    assert v0.ncc(a, -a) == pytest.approx(-1.0)
    assert v0.ncc(a, np.ones_like(a)) is None
    assert abs(v0.ncc(a, rng.random((120, 160)))) < 0.1
    with pytest.raises(ValueError):
        v0.ncc(a, a[:, :10])


def test_grey_small_resizes_any_frame_to_160x120():
    rgb = np.zeros((480, 640, 4), dtype=np.uint8)  # RGBA, as Habitat returns it
    rgb[:, 320:, :3] = 255
    grey = v0.grey_small(rgb)
    assert grey.shape == (120, 160)
    assert grey[:, :80].max() == 0 and grey[:, 81:].min() == 255
    assert v0.image_ncc(rgb, Image.fromarray(rgb[..., :3]).resize((320, 240))) == pytest.approx(1.0, abs=1e-2)


def test_side_by_side_png(tmp_path):
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    v0.write_side_by_side(tmp_path / "x.png", [[("a", img), ("b", img), ("c", None)], [("d", img), ("e", img)]])
    with Image.open(tmp_path / "x.png") as out:
        assert out.size == (3 * v0.TILE[0], 2 * v0.TILE[1])


# --------------------------------------------------------------------------- #
# Held-out scenes, matching, scene files
# --------------------------------------------------------------------------- #
def _scene_dir_name(n):
    return f"{n:05d}-Sc{n:09d}"


def test_heldout_rule_is_b3_plus_15_hashed():
    names = [_scene_dir_name(n) for n in range(40)] + ["00481-5jLhtVmWd5F"]
    out = v0.heldout_by_rule(names)
    assert len(out) == 20 and set(v0.B3_SCENE_DIRS) <= set(out)
    hashed = [n for n in out if n not in v0.B3_SCENE_DIRS]
    import hashlib

    ranked = sorted((n for n in names if n != "00481-5jLhtVmWd5F"),
                    key=lambda d: hashlib.sha256(("exp28-hm3d-heldout:" + d).encode()).hexdigest())
    assert sorted(hashed) == sorted(ranked[:15])
    assert v0.heldout_by_rule(reversed(names)) == out


def test_read_holdout_file(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(["00481-5jLhtVmWd5F", "abc"]))
    (tmp_path / "b.json").write_text(json.dumps({"heldout_scenes": ["x"], "train_scenes": ["y"]}))
    (tmp_path / "c.txt").write_text("# held out\n00481-5jLhtVmWd5F\n\nxyz\n")
    assert v0.read_holdout_file(tmp_path / "a.json") == ["00481-5jLhtVmWd5F", "abc"]
    assert v0.read_holdout_file(tmp_path / "b.json") == ["x"]
    assert v0.read_holdout_file(tmp_path / "c.txt") == ["00481-5jLhtVmWd5F", "xyz"]


def _json_ep(episode_id, text, route):
    """A json episode of route ``route`` (routes differ in start, start rotation and goal)."""
    return {"episode_id": episode_id, "instruction": {"instruction_text": text},
            "start_position": [float(route), 0.0, 0.0], "start_rotation": xyzw(0.1 * route),
            "goals": [{"position": [float(route), 0.0, -5.0], "radius": 3.0}]}


def test_match_scene_drops_ambiguous_reused_and_route_conflicts():
    d = Path("/nowhere/00001-Sc000000001")
    intern = [
        v0.InternEpisode(d, 0, 10, ("Go to the kitchen.", "Walk into the kitchen")),
        v0.InternEpisode(d, 1, 12, ("Exit the room.", "Leave the bedroom.")),
        v0.InternEpisode(d, 2, 12, ("Exit the room", "Go downstairs.")),
        v0.InternEpisode(d, 3, 12, ("Climb the stairs.", "Go up to the attic.")),
    ]
    json_eps = [
        _json_ep(1, "Go to the kitchen. ", 0),
        _json_ep(2, "walk into the kitchen", 0),  # same route, same InternData episode: paired once
        _json_ep(3, "Exit the room.", 1),  # InternData episodes 1 and 2: ambiguous
        _json_ep(4, "Go downstairs.", 2),
        _json_ep(5, "Fly away.", 3),
        _json_ep(6, "Climb the stairs", 4),  # InternData episode 3 through its first instruction ...
        _json_ep(7, "go up to the attic", 5),  # ... and its second, from another route: which one it walked is unknown
    ]
    pairs, stats = v0.match_scene(json_eps, intern, seed="s", scene="Sc000000001")
    assert sorted(p.intern.episode_index for p in pairs) == [0, 2]
    assert stats["matched"] == 5 and stats["ambiguous"] == 1 and stats["unmatched"] == 1
    assert stats["route_conflict"] == 2 and stats["interndata_episode_reused"] == 1 and stats["pairs"] == 2
    again, _ = v0.match_scene(list(reversed(json_eps)), intern, seed="s", scene="Sc000000001")
    assert [p.episode["episode_id"] for p in again] == [p.episode["episode_id"] for p in pairs]


def test_one_json_text_on_two_routes_is_a_conflict():
    d = Path("/nowhere/00001-Sc000000001")
    intern = [v0.InternEpisode(d, 0, 10, ("Exit the room.",))]
    pairs, stats = v0.match_scene([_json_ep(1, "Exit the room.", 0), _json_ep(2, "Exit the room.", 1)], intern,
                                  seed="s", scene="Sc000000001")
    assert pairs == [] and stats["route_conflict"] == 2


def test_resolve_scene(tmp_path):
    scenes, hm3d = tmp_path / "scenes", tmp_path / "hm3d"
    raw = hm3d / "train" / "00007-AbCdEfGhIjK" / "AbCdEfGhIjK.basis.glb"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")
    sid = "hm3d/00007-AbCdEfGhIjK/AbCdEfGhIjK.basis.glb"
    kw = dict(scenes_dir=scenes, hm3d_root=hm3d)
    assert v0.resolve_scene(sid, "AbCdEfGhIjK", glb_source="auto", **kw) == (raw, None, "raw")
    with pytest.raises(FileNotFoundError):
        v0.resolve_scene(sid, "AbCdEfGhIjK", glb_source="tree", **kw)
    tree = scenes / "hm3d" / "AbCdEfGhIjK" / "AbCdEfGhIjK.glb"
    tree.parent.mkdir(parents=True)
    tree.write_bytes(b"")
    (tree.parent / "AbCdEfGhIjK.navmesh").write_bytes(b"")
    assert v0.resolve_scene(sid, "AbCdEfGhIjK", glb_source="auto", **kw) == (
        tree, tree.parent / "AbCdEfGhIjK.navmesh", "tree")
    assert v0.resolve_scene(sid, "AbCdEfGhIjK", glb_source="tree", **kw)[2] == "tree"
    assert v0.resolve_scene(sid, "AbCdEfGhIjK", glb_source="raw", **kw) == (
        raw, tree.parent / "AbCdEfGhIjK.navmesh", "raw")
    with pytest.raises(FileNotFoundError):
        v0.resolve_scene("hm3d/00008-ZZZZZZZZZZZ/ZZZZZZZZZZZ.basis.glb", "ZZZZZZZZZZZ", glb_source="auto", **kw)
    # A build_hm3d_episodes pool names the tree's glb; "raw" still means the original mesh.
    pool_id = "hm3d/AbCdEfGhIjK/AbCdEfGhIjK.glb"
    assert v0.resolve_scene(pool_id, "AbCdEfGhIjK", glb_source="raw", **kw)[0] == raw
    raw.unlink()
    with pytest.raises(FileNotFoundError):
        v0.resolve_scene(pool_id, "AbCdEfGhIjK", glb_source="raw", **kw)


def test_check_sensor_spec_rejects_another_camera():
    spec = {"RGB_SENSOR": {"WIDTH": 640, "HEIGHT": 480, "HFOV": 79.0, "POSITION": [0.0, 1.25, 0.0]},
            "TURN_ANGLE": 15.0}
    v0.check_sensor_spec(spec)
    spec["RGB_SENSOR"]["HFOV"] = 90.0
    with pytest.raises(RuntimeError):
        v0.check_sensor_spec(spec)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _good_rows(n_scenes=10, per_scene=4):
    rows = []
    for s in range(n_scenes):
        for e in range(per_scene):
            rows.append({"scene": f"S{s}", "episode_id": e, "start_ncc": 0.9, "yaw90_ncc": 0.2, "start_margin": 0.7,
                         "snap_m": [0.0] * 20, "end_to_goal_m": 0.5,
                         "mid": [{"ncc": 0.8}, {"ncc": 0.75}, {"ncc": 0.9}]})
    return rows


def test_evaluate_v0_passes_good_rows():
    out = v0.evaluate_v0(_good_rows(), rendered=True)
    assert out["verdict"] == "pass", out
    assert all(c["pass"] for c in out["criteria"].values())


@pytest.mark.parametrize(
    "mutate, failing",
    [
        (lambda rows: [r.update(start_ncc=0.79) for r in rows], "a1_start_ncc_median"),
        (lambda rows: [r.update(start_ncc=0.5) for r in rows[:5]], "a2_start_ncc_ge_0.60_share"),
        (lambda rows: [r.update(start_ncc=None) for r in rows[:5]], "a2_start_ncc_ge_0.60_share"),
        (lambda rows: [r.update(start_margin=0.29) for r in rows[:5]], "a3_yaw90_margin_ge_0.30_share"),
        (lambda rows: [r.update(snap_m=[0.31] * 2 + [0.0] * 18) for r in rows], "b1_snap_le_0.3m_frame_share"),
        (lambda rows: [r.update(snap_m=[None] * 2 + [0.0] * 18) for r in rows], "b1_snap_le_0.3m_frame_share"),
        (lambda rows: [r.update(end_to_goal_m=1.01) for r in rows], "b2_end_to_goal_median_m"),
        (lambda rows: [r.update(mid=[{"ncc": 0.69}]) for r in rows], "b3_mid_ncc_median"),
        (lambda rows: rows[0].update(error="boom"), "n_episodes"),
        (lambda rows: [r.update(scene="S0") for r in rows], "n_scenes"),
    ],
)
def test_evaluate_v0_fails_each_criterion(mutate, failing):
    rows = _good_rows()
    mutate(rows)
    out = v0.evaluate_v0(rows, rendered=True)
    assert out["verdict"] == "fail"
    assert [name for name, c in out["criteria"].items() if not c["pass"]] == [failing]


def test_evaluate_v0_boundaries_pass():
    rows = _good_rows()
    for r in rows:
        r.update(start_ncc=0.80, end_to_goal_m=1.0, mid=[{"ncc": 0.70}])
    for r in rows[:4]:
        r.update(start_ncc=0.59, start_margin=0.1)  # 4 of 40 = 10% under: 90% remain
    snaps = [0.3] * 19 + [0.31]  # 95% at <= 0.3 m
    for r in rows:
        r["snap_m"] = snaps
    assert v0.evaluate_v0(rows, rendered=True)["verdict"] == "pass"


def test_evaluate_v0_dry_run_skips_render_criteria():
    rows = [{k: v for k, v in r.items() if k in ("scene", "episode_id", "end_to_goal_m")} for r in _good_rows()]
    out = v0.evaluate_v0(rows, rendered=False)
    assert out["verdict"] == "dry_run"
    assert out["criteria"]["a1_start_ncc_median"]["pass"] is None
    assert out["criteria"]["b2_end_to_goal_median_m"]["pass"] is True


# --------------------------------------------------------------------------- #
# End to end on a synthetic world
# --------------------------------------------------------------------------- #
PANO_PX_PER_DEG = 4


def _panorama(seed):
    base = (np.random.default_rng(seed).random((12, 144)) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(base).resize((360 * PANO_PX_PER_DEG, 120), Image.BICUBIC), dtype=np.uint8)


def _view(pano, yaw_rad):
    c0 = int(round(-math.degrees(yaw_rad) * PANO_PX_PER_DEG)) % pano.shape[1]
    window = pano[:, (c0 + np.arange(320)) % pano.shape[1]]
    return np.repeat(window[..., None], 3, axis=-1)


class PanoramaRenderer:
    """A scene is a panorama; the render is the window the body yaw looks at; the floor is y = 0."""

    def __init__(self, panos):
        self.panos = panos
        self.pano = None
        self.loads = []
        self.last_pose_err = 0.0

    def load(self, glb, navmesh=None, *, require_file=False):
        self.loads.append(str(glb))
        self.pano = self.panos[v0.hm3d_id(Path(glb).name)]
        return {"navmesh_source": "fake"}

    def render(self, position, rotation_wxyz):
        return _view(self.pano, v0.yaw_of_wxyz(rotation_wxyz))

    def snap(self, point):
        p = np.array(point, dtype=np.float64)
        p[1] = 0.0
        return p

    def geodesic(self, a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _build_world(tmp_path, n_scenes, per_scene, *, start_yaw_error_deg=0.0, extra_heldout=True):
    """InternData scene dirs, a json.gz, a scene tree and the panoramas; returns (args list, panoramas)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(7)
    intern_root, scenes = tmp_path / "interndata", tmp_path / "scenes"
    episodes, panos = [], {}
    total = n_scenes + (1 if extra_heldout else 0)
    for n in range(total):
        dir_name = _scene_dir_name(n + 1)
        scene = v0.hm3d_id(dir_name)
        panos[scene] = _panorama(n)
        sdir = intern_root / dir_name
        (sdir / "meta").mkdir(parents=True)
        (sdir / "data" / "chunk-000").mkdir(parents=True)
        frames_dir = sdir / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg"
        frames_dir.mkdir(parents=True)
        tree = scenes / "hm3d" / scene / f"{scene}.glb"
        tree.parent.mkdir(parents=True)
        tree.write_bytes(b"")
        meta = []
        for i in range(per_scene + 1):  # one InternData episode more than needed: no json for it
            actions = [int(a) for a in rng.choice([1, 1, 1, 2, 3], size=16)]
            start = np.array([rng.uniform(-5, 5), 0.0, rng.uniform(-5, 5)])
            yaw0 = rng.uniform(-math.pi, math.pi)
            pos, yaw, positions, rel = start.copy(), yaw0, [start.copy()], [0.0]
            for a in actions:
                if a == 1:
                    pos = pos + 0.25 * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
                else:
                    yaw += math.radians(15.0) * (1 if a == 2 else -1)
                positions.append(pos.copy())
                rel.append(yaw - yaw0)
            positions = np.asarray(positions)
            trans = v0.habitat_to_interndata(positions, start, xyzw(yaw0))
            poses = []
            for t, theta in zip(trans, rel):
                m = np.eye(4)
                m[:3, :3] = rz(theta)
                m[:3, 3] = t
                poses.append(m.tolist())
            pq.write_table(pa.table({"action": [-1] + actions, "pose.125cm_0deg": poses}),
                           sdir / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
            for t, theta in enumerate(rel):
                Image.fromarray(_view(panos[scene], yaw0 + theta)).save(
                    frames_dir / f"episode_{i:06d}_{t}.jpg", quality=95)
            text = f"Scene {n} route {i}: walk ahead and turn."
            meta.append({"episode_index": i, "tasks": [text + v0.IR.INSTRUCTION_SEP + f"Another way {n} {i}."],
                         "length": len(positions)})
            if i < per_scene:
                episodes.append({
                    "episode_id": 1000 * (n + 1) + i,
                    "trajectory_id": 1000 * (n + 1) + i,
                    "scene_id": f"hm3d/{dir_name}/{scene}.basis.glb",
                    "start_position": start.tolist(),
                    "start_rotation": xyzw(yaw0 + math.radians(start_yaw_error_deg)),
                    "goals": [{"position": positions[-1].tolist(), "radius": 3.0}],
                    "reference_path": [start.tolist(), positions[len(positions) // 2].tolist(), positions[-1].tolist()],
                    "instruction": {"instruction_text": text + " "},
                    "info": {"geodesic_distance": 4.0},
                })
        (sdir / "meta" / "episodes.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    data = tmp_path / "episodes.json.gz"
    with gzip.open(data, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes, "instruction_vocab": {"word_list": []}}, handle)
    holdout = tmp_path / "holdout.json"
    named = [_scene_dir_name(total)] if extra_heldout else []
    fillers = [_scene_dir_name(900 + i) for i in range(v0.HELDOUT_COUNT - len(v0.B3_SCENE_DIRS) - len(named))]
    holdout.write_text(json.dumps([*v0.B3_SCENE_DIRS, *named, *fillers]))
    args = ["--episodes", str(data), "--interndata-root", str(intern_root), "--scenes-dir", str(scenes),
            "--hm3d-root", str(tmp_path / "no_hm3d"), "--holdout-file", str(holdout),
            "--num-scenes", str(n_scenes), "--episodes-per-scene", str(per_scene)]
    return args, panos


def test_end_to_end_passes_on_a_consistent_world(tmp_path):
    argv, panos = _build_world(tmp_path, 10, 4)
    out = tmp_path / "out"
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(out)]), PanoramaRenderer(panos))
    assert summary["verdict"] == "pass", json.dumps(summary["criteria"], indent=1)
    heldout = v0.hm3d_id(_scene_dir_name(11))
    chosen = [c["scene"] for c in summary["selection"]["chosen"]]
    assert len(chosen) == 10 and heldout not in chosen
    assert summary["selection"]["match"]["train"]["pairs"] == 40
    assert summary["selection"]["match"]["heldout"]["scenes"] == 1
    rows = [json.loads(line) for line in (out / v0.ROWS_NAME).read_text().splitlines()]
    assert len(rows) == 40 and not any(r.get("error") for r in rows)
    for r in rows:
        assert r["start_ncc"] > 0.95 and r["start_margin"] > 0.5
        assert all(m["ncc"] > 0.95 for m in r["mid"]) and len(r["mid"]) == 3
        assert r["end_to_goal_m"] < 1e-6 and r["snap_le_0.3m_share"] == 1.0
        assert r["frame0_translation_dev_m"] < 1e-9 and r["frame0_rotation_identity_err"] < 1e-9
        assert r["heading_pose_vs_actions_deg"]["max"] < 1e-6
        assert (out / r["png"]).is_file()
    report = summary["report"]
    assert report["mapping_variants_end_to_goal_p50_m"]["design(-y,-x)"] < 1e-6
    assert report["exact_client_text_share"] == 0.0  # the json keeps its ". " ending, InternData does not
    with pytest.raises(FileExistsError):
        v0.run_v0(v0.parse_args(argv + ["--out-dir", str(out)]), PanoramaRenderer(panos))
    assert summary["mesh_sources"] == {"tree": 10} and summary["pose_columns"] == {"pose.125cm_0deg": 40}
    assert summary["heldout"]["matches_rule"] is False  # the fillers are not the rule's 15


def test_end_to_end_fails_when_the_start_heading_is_off(tmp_path):
    argv, panos = _build_world(tmp_path, 2, 2, start_yaw_error_deg=90.0, extra_heldout=False)
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "out")]), PanoramaRenderer(panos))
    failing = {name for name, c in summary["criteria"].items() if not c["pass"]}
    assert {"a1_start_ncc_median", "a3_yaw90_margin_ge_0.30_share", "b2_end_to_goal_median_m",
            "b3_mid_ncc_median"} <= failing
    assert summary["verdict"] == "fail"


def test_dry_run_needs_no_renderer(tmp_path):
    argv, _ = _build_world(tmp_path, 3, 2, extra_heldout=False)
    out = tmp_path / "out"
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(out), "--dry-run", "--num-scenes", "3"]), None)
    assert summary["verdict"] == "dry_run"
    rows = [json.loads(line) for line in (out / v0.ROWS_NAME).read_text().splitlines()]
    assert len(rows) == 6 and all(r["end_to_goal_m"] < 1e-6 and "start_ncc" not in r for r in rows)
    assert not (out / "png").exists()


def test_heldout_rule_only_draws_scenes_the_json_annotates(tmp_path):
    # As build_hm3d_episodes.split_scenes: an InternData scene without json episodes is not held out.
    argv, _ = _build_world(tmp_path, 3, 2, extra_heldout=False)
    data = tmp_path / "episodes.json.gz"
    with gzip.open(data, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    dropped = v0.hm3d_id(_scene_dir_name(2))
    payload["episodes"] = [e for e in payload["episodes"] if v0.hm3d_id(e["scene_id"]) != dropped]
    with gzip.open(data, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    i = argv.index("--holdout-file")
    argv = argv[:i] + argv[i + 2:]
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "out"), "--dry-run"]), None)
    assert sorted(summary["heldout"]["scenes"]) == sorted([*v0.B3_SCENE_DIRS, _scene_dir_name(1), _scene_dir_name(3)])
    assert summary["selection"]["chosen"] == []


def test_main_refuses_an_existing_summary_before_rendering(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / v0.SUMMARY_NAME).write_text("{}")
    assert v0.main(["--episodes", str(tmp_path / "none.json.gz"), "--out-dir", str(out)]) == 2


def test_holdout_file_must_name_the_20_held_out_scenes(tmp_path):
    good = [*v0.B3_SCENE_DIRS, *(_scene_dir_name(i) for i in range(15))]
    v0.check_heldout(good, "x")
    with pytest.raises(ValueError):
        v0.check_heldout(good[:-1], "x")  # 19
    with pytest.raises(ValueError):
        v0.check_heldout(good[1:] + [_scene_dir_name(99)], "x")  # B3 incomplete, e.g. train_scenes.txt
    with pytest.raises(ValueError):
        v0.check_heldout(good[:-1] + [good[-2]], "x")  # a repeat
    argv, _ = _build_world(tmp_path, 2, 2, extra_heldout=False)
    i = argv.index("--holdout-file")
    (tmp_path / "train.txt").write_text("".join(f"{_scene_dir_name(n)}\n" for n in range(1, 3)))
    argv[i + 1] = str(tmp_path / "train.txt")
    with pytest.raises(ValueError):
        v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "out"), "--dry-run"]), None)
    assert v0.main(argv + ["--out-dir", str(tmp_path / "out2"), "--dry-run"]) == 2


def test_a_pair_without_interndata_frame0_is_not_drawn(tmp_path):
    argv, _ = _build_world(tmp_path, 3, 2, extra_heldout=False)
    lost = tmp_path / "interndata" / _scene_dir_name(2) / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg"
    (lost / "episode_000000_0.jpg").unlink()
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "out"), "--dry-run", "--num-scenes", "3"]),
                        None)
    lost_id = v0.hm3d_id(_scene_dir_name(2))
    assert lost_id not in [c["scene"] for c in summary["selection"]["chosen"]]
    assert "frame 0" in summary["selection"]["skipped_scenes"][lost_id]
    assert summary["selection"]["match"]["train"]["pairs_without_frame0"] == 1


def test_dry_run_selects_what_the_rendered_run_renders(tmp_path):
    argv, panos = _build_world(tmp_path, 3, 2, extra_heldout=False)
    missing = v0.hm3d_id(_scene_dir_name(1))
    (tmp_path / "scenes" / "hm3d" / missing / f"{missing}.glb").unlink()  # not in the tree: not drawn
    dry = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "dry"), "--dry-run"]), None)
    real = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "real")]), PanoramaRenderer(panos))
    assert dry["selection"]["chosen"] == real["selection"]["chosen"]
    assert missing not in [c["scene"] for c in real["selection"]["chosen"]]
    assert "scene tree mesh not found" in real["selection"]["skipped_scenes"][missing]


def test_a_render_off_the_requested_pose_is_an_error_not_a_score(tmp_path):
    argv, panos = _build_world(tmp_path, 2, 2, extra_heldout=False)
    renderer = PanoramaRenderer(panos)
    renderer.last_pose_err = 0.01
    summary = v0.run_v0(v0.parse_args(argv + ["--out-dir", str(tmp_path / "out"), "--num-scenes", "2"]), renderer)
    assert summary["errors"] == 4 and summary["verdict"] == "fail"
    rows = [json.loads(line) for line in (tmp_path / "out" / v0.ROWS_NAME).read_text().splitlines()]
    assert all("off the requested pose" in r["error"] for r in rows)


def test_force_replaces_the_pictures_of_the_previous_run(tmp_path):
    argv, panos = _build_world(tmp_path, 2, 2, extra_heldout=False)
    out = tmp_path / "out"
    v0.run_v0(v0.parse_args(argv + ["--out-dir", str(out), "--num-scenes", "2"]), PanoramaRenderer(panos))
    stale = out / "png" / "stale.png"
    stale.write_bytes(b"")
    v0.run_v0(v0.parse_args(argv + ["--out-dir", str(out), "--num-scenes", "2", "--force"]), PanoramaRenderer(panos))
    assert not stale.exists() and len(list((out / "png").glob("*.png"))) == 4


def test_no_display_is_exit_2_not_a_failed_verdict(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    argv, _ = _build_world(tmp_path, 2, 2, extra_heldout=False)
    assert v0.main(argv + ["--out-dir", str(tmp_path / "out")]) == 2


# --------------------------------------------------------------------------- #
# Navmesh loading (fake habitat: the load logic without a renderer)
# --------------------------------------------------------------------------- #
class _FakePathFinder:
    """``load_nav_mesh`` returns False on a version-2 file and keeps what it had, as Recast's loader does."""

    def __init__(self):
        self.loaded = None

    @property
    def is_loaded(self):
        return self.loaded is not None

    def load_nav_mesh(self, path):
        if Path(path).read_bytes() != b"v1":
            return False
        self.loaded = str(path)
        return True


class _FakeSim:
    def __init__(self):
        self.pathfinder = _FakePathFinder()  # kept across scenes: the worst case
        self.closed = False

    def reconfigure(self, cfg):
        pass

    def recompute_navmesh(self, pathfinder, settings):
        pathfinder.loaded = "baked"
        return True

    def close(self):
        self.closed = True


class _FakeCfg:
    def __init__(self):
        from types import SimpleNamespace

        self.SIMULATOR = SimpleNamespace(SCENE="", AGENT_0=SimpleNamespace(RADIUS=0.1, HEIGHT=1.5),
                                         RGB_SENSOR=SimpleNamespace(POSITION=[0, 1.25, 0]))

    def defrost(self):
        pass

    def freeze(self):
        pass


@pytest.fixture()
def fake_habitat(monkeypatch):
    from types import ModuleType, SimpleNamespace

    habitat = ModuleType("habitat")
    habitat.sims = SimpleNamespace(make_sim=lambda name, config: _FakeSim())
    habitat_sim = ModuleType("habitat_sim")

    class NavMeshSettings:
        def set_defaults(self):
            pass

    habitat_sim.NavMeshSettings = NavMeshSettings
    monkeypatch.setitem(__import__("sys").modules, "habitat", habitat)
    monkeypatch.setitem(__import__("sys").modules, "habitat_sim", habitat_sim)
    return SimpleNamespace(build_habitat_config=lambda args: _FakeCfg())


def _scene(tmp_path, scene, navmesh_bytes, meta=True):
    d = tmp_path / "hm3d" / scene
    d.mkdir(parents=True)
    (d / f"{scene}.glb").write_bytes(b"")
    (d / f"{scene}.navmesh").write_bytes(navmesh_bytes)
    if meta:
        digest = v0.sha256_file(d / f"{scene}.navmesh")
        (d / f"{scene}.navmesh.meta.json").write_text(json.dumps({"navmesh_sha256": digest}))
    return d / f"{scene}.glb", d / f"{scene}.navmesh"


def test_tree_navmesh_must_load_and_a_failed_load_is_not_hidden_by_the_previous_scene(tmp_path, fake_habitat):
    renderer = v0.ClientSimRenderer(fake_habitat)
    glb, navmesh = _scene(tmp_path, "AAAAAAAAAAA", b"v1")
    info = renderer.load(glb, navmesh, require_file=True)
    assert info["navmesh_source"] == "file" and info["navmesh_load_returned"] is True
    assert info["navmesh_matches_meta"] is True and info["navmesh_sha256"] == v0.sha256_file(navmesh)
    glb2, navmesh2 = _scene(tmp_path, "BBBBBBBBBBB", b"v2")
    assert renderer.sim.pathfinder.is_loaded  # still the first scene's navmesh
    with pytest.raises(RuntimeError, match="did not load"):
        renderer.load(glb2, navmesh2, require_file=True)


def test_raw_scene_rebakes_a_version2_navmesh_unless_files_are_required(tmp_path, fake_habitat):
    glb, navmesh = _scene(tmp_path, "CCCCCCCCCCC", b"v2", meta=False)
    renderer = v0.ClientSimRenderer(fake_habitat)
    info = renderer.load(glb, None)  # the mesh's own <stem>.navmesh
    assert info["navmesh_source"] == "recomputed_in_process" and renderer.sim.pathfinder.loaded == "baked"
    strict = v0.ClientSimRenderer(fake_habitat, require_navmesh_file=True)
    with pytest.raises(RuntimeError):
        strict.load(glb, None)


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


def test_client_config_is_the_deployment_camera(client):
    cfg = v0.client_habitat_config(client, "/x/hm3d/AbCdEfGhIjK/AbCdEfGhIjK.glb")
    assert cfg.SIMULATOR.SCENE == "/x/hm3d/AbCdEfGhIjK/AbCdEfGhIjK.glb"
    spec = v0.sensor_spec(cfg)
    v0.check_sensor_spec(spec)
    assert spec["RGB_SENSOR"] == {"WIDTH": 640, "HEIGHT": 480, "HFOV": 79.0, "POSITION": [0.0, 1.25, 0.0]}


@pytest.mark.skipif(not (os.environ.get("EXP28_V0_RENDER_SCENE") and os.environ.get("DISPLAY")),
                    reason="needs EXP28_V0_RENDER_SCENE (a .glb with its navmesh) and an X display")
def test_client_sim_renders_at_the_requested_pose(client):
    renderer = v0.ClientSimRenderer(client)
    try:
        info = renderer.load(Path(os.environ["EXP28_V0_RENDER_SCENE"]))
        assert renderer.sim.pathfinder.is_loaded and info["navmesh_source"] in ("file", "recomputed_in_process")
        point = np.asarray(renderer.sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        q = ORL.yaw_to_quat_wxyz(0.6)
        a = renderer.render(point, q)
        assert a.shape == (480, 640, 3) and a.dtype == np.uint8
        assert renderer.last_pose_err < 1e-4
        assert v0.image_ncc(a, renderer.render(point, q)) == pytest.approx(1.0)
        assert renderer.snap(point) == pytest.approx(point, abs=1e-3)
        assert renderer.geodesic(point, point) == pytest.approx(0.0, abs=1e-3)
    finally:
        renderer.close()
