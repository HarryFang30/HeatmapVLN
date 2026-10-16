"""EXP-28 HM3D episodes: scripts/exp28/build_hm3d_episodes.py on a synthetic world.

A ScaleVLN-style annotation, an R2R train file, an MP3D tree and InternData-N1 meta files are written to tmp_path;
the navmesh is a fake geometry (a floor at y = 0 for |x| <= 40), so nothing here needs habitat_sim.  The last test
touches the real PathFinder wrapper and runs only where habitat_sim imports.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.exp28 import build_hm3d_episodes as bhe
from scripts.tools import remaining_cohort
from scripts.tools.build_r2r_train_cohorts import CohortError

WORDS = ["<pad>", "<unk>", "walk", "to", "the", "kitchen", "and", "stop", "turn", "left", "."]
TOKEN_LENGTH = 12
MP3D = ("M0000000001", "M0000000002")
B3 = ("00901-B0000000901", "00902-B0000000902")
TRAIN_DIRS = tuple(f"{i:05d}-T{i:010d}" for i in range(10, 18))  # InternData scenes the annotation has
EMPTY_DIR = "00099-T0000000099"  # InternData has it, the annotation does not
EXTRA_DIR = "00950-X0000000950"  # the annotation has it, InternData does not
ANNOTATED = (*B3, *TRAIN_DIRS, EXTRA_DIR)
ROUTES = 10


class FloorGeometry:
    """A floor at y = 0 for |x| <= 40: points snap onto it; geodesics are straight lines along it."""

    sha256 = "ab" * 32

    def snap(self, point):
        x, _, z = point
        return [max(-40.0, min(40.0, x)), 0.0, z]

    def geodesic(self, a, b):
        if abs(a[0]) > 40.5 or abs(b[0]) > 40.5:
            return math.inf
        return math.dist((a[0], a[2]), (b[0], b[2]))


class NowhereGeometry(FloorGeometry):
    def snap(self, point):
        return None


def _floor(_hm3d_id):
    return FloorGeometry()


def _text(scene_index: int, route: int, k: int) -> str:
    return f"walk to the kitchen {scene_index} {route} {k}."


def _source_episodes() -> list[dict]:
    """Ten routes of two instructions per scene.  Dropped by the filter: route 9 everywhere (start 0.8 m above
    the floor), route 8 where scene index % 4 == 0 (goal off the floor), route 7 where % 4 == 1 (info geodesic 3 m
    off)."""
    episodes, episode_id = [], 1000
    for si, scene_dir in enumerate(ANNOTATED):
        hm3d_id = scene_dir.split("-")[1]
        for r in range(ROUTES):
            start, goal = [0.0, 0.0, float(r)], [4.0 + r, 0.0, float(r)]
            if r == 9:
                start = [0.0, 0.8, float(r)]
            if r == 8 and si % 4 == 0:
                goal = [45.0, 0.0, float(r)]
            info = goal[0] - start[0] + (3.0 if r == 7 and si % 4 == 1 else 0.0)
            path = [start, [(start[0] + goal[0]) / 2, 0.0, float(r)], goal]
            for k in range(2):
                episode_id += 1
                episodes.append({
                    "episode_id": episode_id,
                    "trajectory_id": r,
                    "scene_id": f"hm3d/{scene_dir}/{hm3d_id}.basis.glb",
                    "start_position": list(start),
                    "start_rotation": [0.0, 0.0, 0.0, 1.0],
                    "info": {"geodesic_distance": info, "something_else": 1},
                    "goals": [{"position": list(goal), "radius": 3.0}],
                    "instruction": {"instruction_text": _text(si, r, k), "instruction_tokens": [9, 9, 9]},
                    "reference_path": [list(p) for p in path],
                })
    return episodes


def _write_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _world(tmp_path: Path, *, overlap_routes: int = 9, edit=None, r2r_extra_key=False, mp3d=MP3D,
           loose_only_route: int | None = None) -> Path:
    world = tmp_path / "world"
    episodes = _source_episodes()
    if edit is not None:
        edit(episodes)
    _write_gz(world / "source.json.gz", {"episodes": episodes, "instruction_vocab": {"word_list": ["other"]}})

    tokenizer = bhe.VocabTokenizer(WORDS, TOKEN_LENGTH)
    r2r = []
    for i, scene in enumerate(MP3D):
        text = "Walk to the kitchen and stop."
        r2r.append({
            "episode_id": i + 1, "trajectory_id": i + 1, "scene_id": f"mp3d/{scene}/{scene}.glb",
            "start_position": [0.0, 0.0, 0.0], "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "info": {"geodesic_distance": 3.0}, "goals": [{"position": [3.0, 0.0, 0.0], "radius": 3.0}],
            "instruction": {"instruction_text": text, "instruction_tokens": tokenizer(text)},
            "reference_path": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            **({"extra": 1} if r2r_extra_key else {}),
        })
    _write_gz(world / "r2r.json.gz", {"episodes": r2r, "instruction_vocab": {"word_list": WORDS, "num_vocab": 11}})

    for scene in mp3d:
        (world / "mp3d" / scene).mkdir(parents=True)
        (world / "mp3d" / scene / f"{scene}.glb").write_bytes(b"glb")

    for root, dirs in (("b3", B3), ("interndata", (*TRAIN_DIRS, EMPTY_DIR))):
        for scene_dir in dirs:
            meta = world / root / scene_dir / "meta"
            meta.mkdir(parents=True)
            rows = []
            if scene_dir in ANNOTATED:
                si = ANNOTATED.index(scene_dir)
                for r in range(overlap_routes):
                    second = _text(si, r, 1)
                    if r == 0:
                        second = second.replace("kitchen", "kitchen,")  # only the loose key matches
                    if r == 1:
                        second = second.replace(" to ", "  to ") + " "  # the strict key alone misses it
                    rows.append({"episode_index": r, "tasks": [_text(si, r, 0) + "<INSTRUCTION_SEP>" + second]})
                if loose_only_route is not None:  # InternData has this route's first instruction up to a comma
                    text = _text(si, loose_only_route, 0).replace("kitchen", "kitchen,")
                    rows.append({"episode_index": len(rows), "tasks": [text]})
            else:
                rows.append({"episode_index": 0, "tasks": ["go somewhere else."]})
            (meta / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (world / "interndata" / "not_a_scene").mkdir()
    return world


def _argv(world: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--source", str(world / "source.json.gz"), "--source-sha256", "",
        "--r2r-train", str(world / "r2r.json.gz"),
        "--mp3d-dir", str(world / "mp3d"), "--expected-mp3d-scenes", "2",
        "--scenes-dir", str(world / "scenes"),
        "--interndata-root", str(world / "interndata"), "--expected-interndata-scenes", "9",
        "--b3-root", str(world / "b3"), "--b3-scenes", ",".join(B3),
        "--heldout-from-hash", "2", "--heldout-per-scene", "3", "--routes-per-pass", "2", "--train-passes", "2",
        "--workers", "1", "--out-dir", str(out), *extra,
    ]


def _load_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _ranked_train_dirs() -> list[str]:
    return sorted(TRAIN_DIRS, key=lambda d: hashlib.sha256(f"exp28-hm3d-heldout:{d}".encode()).hexdigest())


def test_builds_r2r_schema_episodes_scene_split_and_one_cohort(tmp_path, capsys):
    world = _world(tmp_path)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=_floor) == 0
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == "heatmapvln-exp28-hm3d-episodes-v1"

    # Scene split: B3 plus the two smallest hashes of the usable InternData scenes; the rest train.
    ranked = _ranked_train_dirs()
    splits = json.loads((out / "splits" / "splits.json").read_text())
    assert [s["dir"] for s in splits["heldout"]] == [*B3, *ranked[:2]]
    assert [s["source"] for s in splits["heldout"]] == ["b3", "b3", "hash", "hash"]
    assert [s["dir"] for s in splits["train"]] == sorted(ranked[2:])
    assert splits["interndata_scenes_without_episodes"] == [EMPTY_DIR]
    plain = sorted((*TRAIN_DIRS, EMPTY_DIR), key=bhe.heldout_rank)
    assert splits["heldout_hash_skipped"] == [d for d in plain[:2] if d == EMPTY_DIR]
    assert manifest["summary"]["used_routes_sharing_a_path"] == 0
    held_lines = [x for x in (out / "splits" / "heldout_scenes.txt").read_text().splitlines() if not x.startswith("#")]
    assert held_lines == [*B3, *ranked[:2]]
    heldout_ids = {d.split("-")[1] for d in held_lines}
    train_ids = {d.split("-")[1] for d in ranked[2:]}

    # D0: 10 used scenes x 20 episodes; route 9 everywhere, route 8 in scenes 0/4/8, route 7 in scenes 1/5/9.
    d0 = manifest["summary"]["d0"]
    assert (d0["dropped_episodes"], d0["episodes"], d0["dropped_routes"], d0["pass"]) == (32, 200, 16, True)
    assert d0["scenes_all_dropped"] == [] and manifest["summary"]["kept_routes_merged_into_a_same_path_route"] == 0
    assert manifest["summary"]["used_goal_radius"] == {"3.0": 200}
    report = json.loads((out / "filter_report.json").read_text())
    assert report["reasons"]["start_snap"] == 20 and report["reasons"]["goal_snap"] == 6
    assert report["reasons"]["geodesic_not_finite"] == 6 and report["reasons"]["geodesic_mismatch"] == 6

    # The pool: R2R train's schema and vocabulary, scene ids in the EXP-28 tree, one instruction per kept route.
    r2r = _load_gz(world / "r2r.json.gz")
    pool = _load_gz(out / "pool.json.gz")
    assert set(pool) == {"episodes", "instruction_vocab"} and pool["instruction_vocab"] == r2r["instruction_vocab"]
    assert len(pool["episodes"]) == 200 // 2 - 16 == summary["pool_routes"]
    for e in pool["episodes"]:
        assert set(e) == set(r2r["episodes"][0])
        assert set(e["instruction"]) == {"instruction_text", "instruction_tokens"}
        assert set(e["goals"][0]) == {"position", "radius"}
        # info: R2R's geodesic_distance plus the navmesh the episode was filtered on (the client checks it).
        assert e["info"] == {"geodesic_distance": 4.0 + e["trajectory_id"], "navmesh_sha256": FloorGeometry.sha256}
        hm3d_id = Path(e["scene_id"]).stem
        assert e["scene_id"] == f"hm3d/{hm3d_id}/{hm3d_id}.glb" and hm3d_id in heldout_ids | train_ids
        assert len(e["instruction"]["instruction_tokens"]) == TOKEN_LENGTH
        assert e["instruction"]["instruction_tokens"][:4] == [2, 3, 4, 5]  # walk to the kitchen, in R2R's vocab
        assert e["trajectory_id"] != 9
    routes = [(Path(e["scene_id"]).stem, e["trajectory_id"]) for e in pool["episodes"]]
    assert len(routes) == len(set(routes))
    source_ids = {e["episode_id"] for e in _source_episodes()}
    assert {e["episode_id"] for e in pool["episodes"]} <= source_ids

    # The cohort: held-out shards first, then pass 1, then pass 2; one scene per shard.
    cohort = json.loads((out / "cohorts" / "manifest.json").read_text())
    assert cohort["schema"] == "heatmapvln-r2r-train-cohorts-v1" and cohort["split"] == "train"
    assert {p: cohort["parts"][p]["shards"] for p in cohort["parts"]} == {
        "heldout": "0-3", "train_pass1": "4-9", "train_pass2": "10-15"}
    assert cohort["num_shards"] == 16 and cohort["total_episodes"] == 4 * 3 + 6 * 2 * 2 == cohort["total_routes"]
    assert summary["cohort"]["heldout"]["shards"] == "0-3" and summary["shortfall"] == {}
    by_episode = {e["episode_id"]: e for e in pool["episodes"]}
    seen_routes = set()
    for record in cohort["shards"]:
        listing = json.loads((out / "cohorts" / record["path"]).read_text())
        data = _load_gz(out / "cohorts" / record["dataset_path"])
        assert listing["shard_index"] == record["index"] and listing["part"] == record["part"]
        assert listing["dataset_sha256"] == cohort["dataset_sha256"] == hashlib.sha256(
            (out / "pool.json.gz").read_bytes()).hexdigest()
        assert len(record["scenes"]) == 1
        assert record["scenes"][0] in (heldout_ids if record["part"] == "heldout" else train_ids)
        assert len(listing["episodes"]) == (3 if record["part"] == "heldout" else 2)
        assert data["instruction_vocab"] == r2r["instruction_vocab"]
        assert [e["episode_id"] for e in data["episodes"]] == [x["episode_id"] for x in listing["episodes"]]
        for item, e in zip(listing["episodes"], data["episodes"]):
            assert e == by_episode[item["episode_id"]] and item["scene_id"] == Path(e["scene_id"]).stem
            route = (item["scene_id"], item["trajectory_id"])
            assert route not in seen_routes
            seen_routes.add(route)

    # Each scene's routes in draw order: held-out takes the first 3, training 2 + 2, the rest stay for C2.
    table = json.loads((out / "routes.json").read_text())
    for hm3d_id, entry in table.items():
        parts = [r["part"] for r in entry["routes"]]
        if hm3d_id in heldout_ids:
            assert parts[:3] == ["heldout"] * 3 and set(parts[3:]) == {None}
        else:
            assert parts[:4] == ["train_pass1"] * 2 + ["train_pass2"] * 2 and set(parts[4:]) == {None}

    # Overlap: InternData has routes 0-8 of every scene; one instruction of route 0 differs by a comma (only the
    # loose key matches), one of route 1 by whitespace (only the strict key misses).
    overlap = manifest["overlap"]["source_episodes_used_scenes"]
    assert (overlap["strict"]["rate"], overlap["normalized"]["rate"], overlap["loose"]["rate"]) == (0.8, 0.85, 0.9)
    assert manifest["overlap"]["heldout_nonoverlap_rule"] is False
    assert summary["overlap_pool"] == manifest["overlap"]["pool"]["normalized"]
    keys = json.loads((out / "overlap_keys.json").read_text())
    comma = [e for e in pool["episodes"]
             if e["trajectory_id"] == 0 and e["instruction"]["instruction_text"].endswith(" 0 1.")]
    assert keys == sorted([Path(e["scene_id"]).stem, e["episode_id"]] for e in pool["episodes"] if e not in comma)
    assert manifest["tokens"]["r2r_train_reproduction"] == 1.0
    assert manifest["navmesh_sha256"] == {i: FloorGeometry.sha256 for i in heldout_ids | train_ids}
    assert set(manifest["files"]) >= {"pool.json.gz", "cohorts/manifest.json", "routes.json", "splits/splits.json"}


def test_the_same_inputs_rebuild_the_same_bytes_and_remaining_cohort_reads_the_cohort(tmp_path):
    world = _world(tmp_path)
    for name, workers in (("a", "1"), ("b", "3")):  # b: the navmesh filter in a process pool
        assert bhe.main([*_argv(world, tmp_path / name), "--workers", workers], geometry_factory=_floor) == 0
    files = sorted(str(p.relative_to(tmp_path / "a")) for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert files == sorted(str(p.relative_to(tmp_path / "b")) for p in (tmp_path / "b").rglob("*") if p.is_file())
    for name in files:
        if not name.endswith("manifest.json"):  # these name the output directory
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name

    rest = remaining_cohort.remaining(tmp_path / "a" / "cohorts", tmp_path / "no_records", tmp_path / "rest")
    assert rest["total_episodes"] == 36 and rest["num_shards"] == 16


def test_low_overlap_draws_heldout_routes_only_from_routes_interndata_lacks(tmp_path):
    world = _world(tmp_path, overlap_routes=4)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=_floor) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["overlap"]["source_episodes_used_scenes"]["normalized"]["rate"] == 0.35
    assert manifest["overlap"]["heldout_nonoverlap_rule"] is True
    table = json.loads((out / "routes.json").read_text())
    held = [r for entry in table.values() for r in entry["routes"] if r["part"] == "heldout"]
    assert len(held) == 12 and all(not r["route_in_interndata"] and r["trajectory_id"] >= 4 for r in held)
    # Training is not filtered by overlap.
    train = [r for entry in table.values() for r in entry["routes"] if r["part"] == "train_pass1"]
    assert any(r["route_in_interndata"] for r in train)


def test_d0_failure_writes_only_the_report(tmp_path, capsys):
    world = _world(tmp_path)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=lambda _: NowhereGeometry()) == 3
    assert "D0" in capsys.readouterr().err
    assert not out.exists() and not list(tmp_path.glob(".out.*"))
    report = json.loads((tmp_path / "out.d0_failed.json").read_text())
    assert report["d0"]["pass"] is False and report["d0"]["dropped_episodes"] == 200


def _no_info(episodes):
    for e in episodes:
        del e["info"]


def test_episodes_without_info_geodesic_are_dropped_unless_filled(tmp_path, capsys):
    world = _world(tmp_path, edit=_no_info)
    assert bhe.main(_argv(world, tmp_path / "drop"), geometry_factory=_floor) == 3
    assert "--missing-info-geodesic fill" in capsys.readouterr().err
    report = json.loads((tmp_path / "drop.d0_failed.json").read_text())
    assert report["reasons"]["no_info_geodesic"] == 194  # all but the off-floor goals (no finite geodesic)
    assert report["summary"]["used_episodes_without_info_geodesic"] == 200

    out = tmp_path / "fill"
    assert bhe.main([*_argv(world, out), "--missing-info-geodesic", "fill"], geometry_factory=_floor) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rules"]["filter"]["missing_info_geodesic"] == "fill"
    # Route 9 (start) and route 8 in scenes 0/4/8 (goal) still drop; route 7 no longer has an info to miss.
    assert manifest["summary"]["d0"]["dropped_episodes"] == 26
    assert manifest["summary"]["kept_episodes_info_geodesic_filled"] == 174
    pool = _load_gz(out / "pool.json.gz")
    assert len(pool["episodes"]) == 100 - 13
    for e in pool["episodes"]:
        assert e["info"] == {"geodesic_distance": 4.0 + e["trajectory_id"], "navmesh_sha256": FloorGeometry.sha256}


def test_a_scene_whose_navmesh_does_not_load_stops_the_build(tmp_path, capsys):
    def factory(hm3d_id):
        if hm3d_id == "T0000000012":
            raise FileNotFoundError("navmesh not found: somewhere")
        return FloorGeometry()

    world = _world(tmp_path)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=factory) == 2
    assert "scene T0000000012: navmesh not found" in capsys.readouterr().err
    assert not out.exists() and not list(tmp_path.glob(".out.*"))


def test_overlap_key_is_v0s_pairing_key():
    v0 = pytest.importorskip("scripts.exp28.verify_hm3d_render")
    for text in ("Walk to the kitchen. ", "walk  to the\tkitchen!", "Go up the stairs?", "turn left", "Stop.."):
        assert bhe.instruction_key(text) == v0.match_key(text), text
    assert bhe.strict_key("Walk to the door. ") == "Walk to the door. "
    assert bhe.instruction_key("Walk to the door. ") == "walk to the door"
    assert bhe.loose_key("Walk, to the door.") == "walk to the door"


def test_routes_sharing_a_path_are_counted():
    def route(start):
        return [{"start_position": start, "goals": [{"position": [1.0, 0.0, 0.0]}],
                 "reference_path": [start, [1.0, 0.0, 0.0]]}]

    routes = {("A", 1): route([0.0, 0.0, 0.0]), ("A", 2): route([0.0, 0.0, 0.001]), ("A", 3): route([0.0, 0.0, 1.0]),
              ("B", 4): route([0.0, 0.0, 0.0])}
    assert bhe.routes_sharing_a_path(routes) == 2


def test_dry_run_prints_the_split_and_writes_nothing(tmp_path, capsys):
    world = _world(tmp_path)
    out = tmp_path / "out"
    assert bhe.main([*_argv(world, out), "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["split"]["heldout"] == [*B3, *_ranked_train_dirs()[:2]] and len(summary["split"]["train"]) == 6
    assert summary["used_episodes"] == 200 and not out.exists()


def _duplicate_id(episodes):
    episodes[5]["episode_id"] = episodes[4]["episode_id"]


def _two_paths_one_route(episodes):
    episodes[1]["start_position"] = [0.0, 0.0, 5.0]


def _bad_scene_id(episodes):
    episodes[0]["scene_id"] = "hm3d/train/00901-B0000000901/B0000000901.glb"


@pytest.mark.parametrize(
    "kwargs, extra, message",
    [
        ({"edit": _duplicate_id}, (), "not globally unique"),
        ({"edit": _two_paths_one_route}, (), "trajectory_id does not identify a route"),
        ({"edit": _bad_scene_id}, (), "is not hm3d/00xxx-<ID>/<ID>.basis.glb"),
        ({"mp3d": (*MP3D, "B0000000901")}, ("--expected-mp3d-scenes", "0"), "HM3D IDs equal to MP3D IDs"),
        ({"r2r_extra_key": True}, (), "would not match R2R"),
        ({}, ("--expected-interndata-scenes", "300"), "expected 300"),
        ({}, ("--source-sha256", "0" * 64), "not the expected"),
    ],
)
def test_input_checks_stop_before_anything_is_written(tmp_path, capsys, kwargs, extra, message):
    world = _world(tmp_path, **kwargs)
    out = tmp_path / "out"
    assert bhe.main([*_argv(world, out), *extra], geometry_factory=_floor) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_filter_thresholds_are_the_preregistered_ones():
    floor = FloorGeometry()
    keep = bhe.check_geometry([0, 0.5, 0], [10, 0, 0], [[0, 0, 0], [10, 0.5, 0]], 10.0, floor)
    assert keep[0] == []
    assert bhe.check_geometry([0, 0.51, 0], [10, 0, 0], [[0, 0, 0]], 10.0, floor)[0] == ["start_snap"]
    assert bhe.check_geometry([0, 0, 0], [10, 0, 0], [[0, 0.6, 0]], 10.0, floor)[0] == ["path_snap"]
    # max(1 m, 15% of info.geodesic_distance): 1.5 m at 10 m, 1 m at 4 m.
    for goal_x, info, reasons in ((11.4, 10.0, []), (8.6, 10.0, []), (11.6, 10.0, ["geodesic_mismatch"]),
                                  (8.4, 10.0, ["geodesic_mismatch"]), (4.95, 4.0, []),
                                  (5.05, 4.0, ["geodesic_mismatch"])):
        assert bhe.check_geometry([0, 0, 0], [goal_x, 0, 0], [[0, 0, 0]], info, floor)[0] == reasons, (goal_x, info)
    assert bhe.check_geometry([0, 0, 0], [4, 0, 0], [[0, 0, 0]], None, floor)[0] == ["no_info_geodesic"]
    reasons, stats = bhe.check_geometry([0, 0, 0], [50, 0, 0], [[0, 0, 0]], 50.0, floor)
    assert reasons == ["goal_snap", "geodesic_not_finite"] and stats["goal_snap_m"] == 10.0


def test_tokens_follow_habitats_vocabdict_rule():
    tokenizer = bhe.VocabTokenizer(["<pad>", "<unk>", "go", "to", "the", "kitchen", "door", "."], 10)
    assert bhe.VocabTokenizer.tokenize("Go to the kitchen's door, now?") == [
        "go", "to", "the", "kitchen", "'", "s", "door", "now"]
    assert tokenizer("Go to the kitchen's door, now?") == [2, 3, 4, 5, 1, 1, 6, 1, 0, 0]
    assert tokenizer("go " * 12) == [2] * 10  # truncated to the fixed length
    assert bhe.VocabTokenizer(["go"], None)("go away") == [1, 0]  # <unk> prepended as VocabDict does


def test_scene_ids_and_heldout_rank():
    assert bhe.parse_source_scene("hm3d/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb") == (
        "00481-5jLhtVmWd5F", "5jLhtVmWd5F")
    assert bhe.parse_source_scene("hm3d/train/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb")[1] == "5jLhtVmWd5F"
    with pytest.raises(CohortError):
        bhe.parse_source_scene("hm3d/00481-5jLhtVmWd5F/OTHERxxxxxx.basis.glb")
    assert bhe.hm3d_scene_id("5jLhtVmWd5F") == "hm3d/5jLhtVmWd5F/5jLhtVmWd5F.glb"
    assert bhe.heldout_rank("00481-5jLhtVmWd5F") == hashlib.sha256(b"exp28-hm3d-heldout:00481-5jLhtVmWd5F").hexdigest()
    assert bhe._ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_scene_tree_needs_the_glb_and_the_rebaked_navmesh(tmp_path):
    pytest.importorskip("habitat_sim")
    tree = bhe.SceneTree(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_hm3d_scenes"):
        tree("B0000000901")
    (tmp_path / "hm3d" / "B0000000901").mkdir(parents=True)
    (tmp_path / "hm3d" / "B0000000901" / "B0000000901.glb").write_bytes(b"glb")
    with pytest.raises(FileNotFoundError, match="navmesh not found"):
        tree("B0000000901")


def test_a_scene_that_keeps_no_episode_fails_d0_under_the_global_line(tmp_path, capsys):
    """habitat-sim 0.1.7 may call a navmesh loaded that answers nothing; the scene must not just leave the cohort."""
    def factory(hm3d_id):
        return NowhereGeometry() if hm3d_id == "T0000000012" else FloorGeometry()

    world = _world(tmp_path)
    out = tmp_path / "out"
    assert bhe.main([*_argv(world, out), "--max-drop-rate", "0.5"], geometry_factory=factory) == 3
    err = capsys.readouterr().err
    assert "keep no episode at all" in err and "T0000000012" in err
    report = json.loads((tmp_path / "out.d0_failed.json").read_text())
    assert report["d0"]["scenes_all_dropped"] == ["T0000000012"] and report["d0"]["drop_rate"] < 0.5
    assert not out.exists()


def _w_first(episodes):
    for e in episodes:  # a 90-degree yaw written [w, x, y, z]
        e["start_rotation"] = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]


def test_start_rotation_in_another_component_order_stops_the_build(tmp_path, capsys):
    world = _world(tmp_path, edit=_w_first)
    assert bhe.main([*_argv(world, tmp_path / "out"), "--dry-run"]) == 2
    assert "not a unit yaw quaternion [x, y, z, w]" in capsys.readouterr().err


def test_yaw_quaternion_rule():
    half = math.radians(37.0) / 2
    assert bhe.is_yaw_quaternion_xyzw([0.0, math.sin(half), 0.0, math.cos(half)])
    assert bhe.is_yaw_quaternion_xyzw([0.0, -1.0, 0.0, 0.0])  # 180 degrees
    assert not bhe.is_yaw_quaternion_xyzw([math.cos(half), 0.0, math.sin(half), 0.0])  # [w, x, y, z]
    assert not bhe.is_yaw_quaternion_xyzw([0.0, 0.5, 0.0, 0.5])  # not unit


def _route3_on_route2s_path(episodes):
    path_of = {e["scene_id"]: e for e in episodes if e["trajectory_id"] == 2}
    for e in episodes:
        if e["trajectory_id"] == 3:
            for key in ("start_position", "goals", "reference_path", "info"):
                e[key] = json.loads(json.dumps(path_of[e["scene_id"]][key]))


def test_trajectory_ids_over_one_path_are_one_route(tmp_path):
    world = _world(tmp_path, edit=_route3_on_route2s_path)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=_floor) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary"]["used_routes_sharing_a_path"] == 2 * 10
    assert manifest["summary"]["kept_routes_merged_into_a_same_path_route"] == 10
    pool = _load_gz(out / "pool.json.gz")
    assert len(pool["episodes"]) == 200 // 2 - 16 - 10
    table = json.loads((out / "routes.json").read_text())
    cohort = json.loads((out / "cohorts" / "manifest.json").read_text())
    in_cohort = {(e["scene_id"], e["trajectory_id"]) for r in cohort["shards"]
                 for e in json.loads((out / "cohorts" / r["path"]).read_text())["episodes"]}
    for hm3d_id, entry in table.items():
        rows = {r["trajectory_id"]: r for r in entry["routes"]}
        assert len({2, 3} & set(rows)) == 1, hm3d_id  # one of the two, the first drawn
        (kept,) = {2, 3} & set(rows)
        assert rows[kept]["same_path_trajectory_ids"] == [5 - kept] and rows[kept]["instructions"] == 4
        assert (hm3d_id, 5 - kept) not in in_cohort


def test_heldout_exclusion_uses_the_widest_key(tmp_path):
    """Under the < 80% branch a route InternData has only up to punctuation is not held out as unseen."""
    world = _world(tmp_path, overlap_routes=4, loose_only_route=4)
    out = tmp_path / "out"
    assert bhe.main(_argv(world, out), geometry_factory=_floor) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    rates = {k: v["rate"] for k, v in manifest["overlap"]["source_episodes_used_scenes"].items()}
    assert rates == {"strict": 0.3, "normalized": 0.35, "loose": 0.45}
    assert manifest["summary"]["overlap_keys_disagree_on_rule"] is False
    table = json.loads((out / "routes.json").read_text())
    held = [r for entry in table.values() for r in entry["routes"] if r["part"] == "heldout"]
    assert len(held) == 12 and all(r["trajectory_id"] >= 5 and not r["route_in_interndata_loose"] for r in held)
    route4 = [r for entry in table.values() for r in entry["routes"] if r["trajectory_id"] == 4]
    assert all(r["route_in_interndata_loose"] and not r["route_in_interndata"] for r in route4)


def test_split_reports_scenes_the_plain_hash_rule_would_have_held_out():
    plain = sorted(TRAIN_DIRS, key=bhe.heldout_rank)
    missing, last = plain[1], plain[-1]  # the annotation lacks the second-ranked scene and the last one
    annotated = {*TRAIN_DIRS, *B3} - {missing, last}
    split = bhe.split_scenes(TRAIN_DIRS, B3, annotated, from_hash=2)
    assert split["heldout_hash_skipped"] == [missing]  # the last one would not have been held out anyway
    assert split["heldout_from_hash"] == [plain[0], plain[2]]
    assert split["interndata_scenes_without_episodes"] == sorted([missing, last])
    assert split["train"] == sorted(plain[3:-1])
    assert bhe.split_scenes(TRAIN_DIRS, B3, {*TRAIN_DIRS, *B3}, from_hash=2)["heldout_hash_skipped"] == []
