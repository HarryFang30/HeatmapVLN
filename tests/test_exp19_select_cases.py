"""EXP-19 [A] case selection: the pure parts of scripts/exp19/select_cases.py (no server data).

Covers the pre-registered ordering (|steps_42 - category median| ascending,
ties -> smaller sha1("<scene>:<episode_id>")), the distinct-scene rule, the
pool rule (both seeds, ppa_applied >= 5 in both, seed-42 extra), the GPU
sharding, the client-log parser (call indices, [amb3r-vo] attribution, tqdm
``\\r`` fragments, physical line numbers) and the reference-path features.
"""
from __future__ import annotations

import hashlib

import pytest

from scripts.exp19 import select_cases as sc


def _entry(scene, eid, steps):
    return {"scene_id": scene, "episode_id": eid, "steps_42": steps}


def test_order_key_is_sha1_of_scene_colon_plain_episode_id():
    assert sc.order_key("zsNo4HB9uLZ", 7) == hashlib.sha1(b"zsNo4HB9uLZ:7").hexdigest()
    assert sc.order_key("zsNo4HB9uLZ", 7) != hashlib.sha1(b"zsNo4HB9uLZ:0007").hexdigest()
    assert sc.ep_key("zsNo4HB9uLZ", 7) == "zsNo4HB9uLZ_0007"


def test_order_pool_sorts_by_distance_to_median():
    pool = [_entry("A", 1, 100), _entry("B", 2, 60), _entry("C", 3, 70), _entry("D", 4, 72), _entry("E", 5, 40)]
    median, ordered = sc.order_pool(pool)
    assert median == 70.0
    assert [e["episode_id"] for e in ordered] == [3, 4, 2, 1, 5]  # |d| = 0, 2, 10, 30, 30 -> tie below
    assert [e["sort_key"][0] for e in ordered] == [0.0, 2.0, 10.0, 30.0, 30.0]
    assert all(e["sort_key"][1] == sc.order_key(e["scene_id"], e["episode_id"]) for e in ordered)
    # the |d| = 30 tie goes to the smaller sha1 key
    tie = sorted([(sc.order_key("A", 1), 1), (sc.order_key("E", 5), 5)])
    assert [e["episode_id"] for e in ordered[3:]] == [t[1] for t in tie]


def test_order_pool_even_count_uses_midpoint_median_and_breaks_ties_by_sha1():
    pool = [_entry("S", i, steps) for i, steps in enumerate([50, 60, 70, 80])]
    median, ordered = sc.order_pool(pool)
    assert median == 65.0
    assert {e["sort_key"][0] for e in ordered[:2]} == {5.0}
    first_two = sorted(ordered[:2], key=lambda e: e["sort_key"][1])
    assert ordered[:2] == first_two
    assert [e["steps_42"] for e in ordered[:2]] in ([60, 70], [70, 60])


def test_order_pool_does_not_depend_on_input_order():
    pool = [_entry(s, i, 60 + (i % 5)) for i, s in enumerate("ABCDEFGHIJ")]
    _, a = sc.order_pool(pool)
    _, b = sc.order_pool(list(reversed(pool)))
    assert [e["episode_id"] for e in a] == [e["episode_id"] for e in b]
    assert sc.order_pool([]) == (None, [])


def test_pick_distinct_scenes_skips_repeated_scene_and_stops_at_top():
    ordered = [_entry("A", 1, 0), _entry("A", 2, 0), _entry("B", 3, 0), _entry("A", 4, 0),
               _entry("C", 5, 0), _entry("D", 6, 0), _entry("B", 7, 0)]
    picked, skipped = sc.pick_distinct_scenes(ordered, top=3)
    assert [e["episode_id"] for e in picked] == [1, 3, 5]
    assert [e["episode_id"] for e in skipped] == [2, 4]  # only those passed over before the third pick


def test_pick_distinct_scenes_with_too_few_scenes_returns_fewer():
    picked, skipped = sc.pick_distinct_scenes([_entry("A", 1, 0), _entry("A", 2, 0), _entry("B", 3, 0)], top=3)
    assert [e["episode_id"] for e in picked] == [1, 3]
    assert [e["episode_id"] for e in skipped] == [2]


def _cand(cat, rank, steps):
    return {"category": cat, "rank": rank, "steps_42": steps, "ep_key": f"{cat}{rank}"}


def test_shard_lpt_balances_and_keeps_category_order():
    cands = [_cand(c, r, s) for c, steps in (("T1", (64, 64, 65)), ("T2", (57, 57, 57)), ("T3", (85, 85, 85)),
                                             ("F1", (78, 78, 78)), ("F2", (500, 500, 500)))
             for r, s in enumerate(steps)]
    lists = sc.shard_lpt(cands, 3)
    assert sorted(c["ep_key"] for lst in lists for c in lst) == sorted(c["ep_key"] for c in cands)
    loads = [sum(c["steps_42"] for c in lst) for lst in lists]
    assert max(loads) - min(loads) <= 65
    # one 500-step F2 per GPU, handed out in rank order to gpu0, gpu1, gpu2
    assert [[c["ep_key"] for c in lst if c["category"] == "F2"] for lst in lists] == [["F20"], ["F21"], ["F22"]]
    order = {c: i for i, c in enumerate(sc.CATEGORY_ORDER)}
    for lst in lists:
        keys = [(order[c["category"]], c["rank"]) for c in lst]
        assert keys == sorted(keys)


def test_shard_lpt_greedy_ties_go_to_lower_index():
    lists = sc.shard_lpt([_cand("T1", 0, 10), _cand("T1", 1, 10), _cand("T2", 0, 10), _cand("T3", 0, 5)], 2)
    assert [[c["ep_key"] for c in lst] for lst in lists] == [["T10", "T20"], ["T11", "T30"]]
    one = sc.shard_lpt([_cand("F2", 0, 500), _cand("T1", 0, 60)], 1)
    assert [c["ep_key"] for c in one[0]] == ["T10", "F20"]


def test_num_gpus_is_capped_at_three():
    assert sc.parse_args([]).num_gpus == 3
    with pytest.raises(SystemExit):
        sc.parse_args(["--num-gpus", "4"])
    with pytest.raises(SystemExit):
        sc.parse_args(["--num-gpus", "0"])


def _data(rows):
    """Minimal ``load_inputs``-shaped dict: rows = (eid, scene, static, run42, run1337)."""
    base_static = {"geodesic_m": 9.0, "dy_m": 0.0, "ref_turns45": 2, "n_room_runs": 3}
    base_run = {"success": 1, "os": 1, "steps": 100, "ppa_applied": 10, "ended_by": "stop"}
    data = {"eids": [], "static": {}, "runs": {"42": {}, "1337": {}}}
    for eid, scene, st, r42, r1337 in rows:
        data["eids"].append(eid)
        data["static"][eid] = dict(base_static, scene_id=scene, **st)
        data["runs"]["42"][eid] = dict(base_run, **r42)
        data["runs"]["1337"][eid] = dict(base_run, **r1337)
    return data


def test_category_pool_needs_both_seeds_ppa_and_seed42_extra():
    data = _data([
        (1, "A", {}, {}, {}),                            # in T1
        (2, "A", {}, {}, {"success": 0, "os": 1}),       # fails in seed 1337
        (3, "B", {}, {"ppa_applied": 4}, {}),            # ppa < 5 in seed 42
        (4, "B", {}, {}, {"ppa_applied": 4}),            # ppa < 5 in seed 1337
        (5, "C", {}, {"steps": 151}, {}),                # seed-42 step limit
        (6, "C", {}, {}, {"steps": 400}),                # seed-1337 steps do not matter
        (7, "D", {"dy_m": 1.2}, {}, {}),                 # T2, not T1
        (8, "D", {"n_room_runs": 2}, {}, {}),            # too few rooms
    ])
    assert [e["episode_id"] for e in sc.category_pool(data, "T1")] == [1, 6]
    assert [e["episode_id"] for e in sc.category_pool(data, "T2")] == [7]
    assert sc.category_pool(data, "T1")[0] == {"scene_id": "A", "episode_id": 1, "steps_42": 100}


def test_failure_categories_use_ended_by():
    fail_stop = {"success": 0, "os": 1, "ended_by": "stop", "steps": 80}
    fail_cap = {"success": 0, "os": 1, "ended_by": "step_cap", "steps": 500}
    data = _data([
        (1, "A", {}, fail_stop, fail_stop),
        (2, "B", {}, fail_cap, fail_cap),
        (3, "C", {}, fail_stop, fail_cap),               # differs across seeds -> neither
        (4, "D", {}, dict(fail_stop, os=0), fail_stop),  # os = 0 in seed 42 -> not F1
    ])
    assert [e["episode_id"] for e in sc.category_pool(data, "F1")] == [1]
    assert [e["episode_id"] for e in sc.category_pool(data, "F2")] == [2]


def test_ended_by():
    calls = lambda *kinds: {"calls": [{"kind": k} for k in kinds]}  # noqa: E731
    assert sc.ended_by(calls("trajectory", "stop"), 64) == "stop"
    assert sc.ended_by(calls("trajectory", "native_actions"), 500) == "step_cap"
    assert sc.ended_by(calls("trajectory"), 120) == "other"
    assert sc.ended_by(calls("stop"), 500) == "stop"


LOG = [
    "Fixed episode list (2): /x/shard.json\n",
    "\rEvaluating:   0%|   | 0/2 [00:00<?, ?it/s]\rEvaluating:   0%|  | 0/2 [00:00<?, ?it/s, SPL=?, SR=?]\n",
    "[1/2] Episode zsNo4HB9uLZ_0001: Exit the bedroom...\n",
    "  [amb3r-vo] frame=0 history=[] ready=False phase=insufficient_history revision=0\n",
    "  step_id: 0, RPC kind=native_actions, VLM output: ←←←\n",
    "  [debug] actions=[2, 2, 2, 0]\n",
    "  [debug] local trajectory STOP -> replan\n",
    "  [amb3r-vo] frame=21 history=[0, 2, 5, 8, 11, 14, 17, 20] ready=True phase=stateful_backend revision=2\n",
    "  step_id: 21, RPC kind=trajectory, VLM output: 180 176\n",
    "  [debug] trajectory traj_goal=(2.77,-1.07), direct=2.97, path_len=2.98, actions=[2, 1, 1, 1]\n",
    "  step_id: 25, RPC kind=stop, VLM output: STOP\n",
    "  [debug] actions=[0]\n",
    "  => success: 1.0, spl: 1.0000, os: 1.0, ne: 0.5122, vlm_calls: 3, trajectory_calls: 1\n",
    "\rEvaluating:  50%|##  | 1/2 [01:54<01:54, SPL=1.000, SR=1.000]\n",
    "[2/2] Episode x8F5xyUWy9e_0009: interrupted episode...\n",
    "  step_id: 0, RPC kind=native_actions, VLM output: →\n",
    "[1/1] Episode zsNo4HB9uLZ_0001: resumed rerun of the first episode...\n",
    "  step_id: 0, RPC kind=stop, VLM output: \n",
    "  => success: 0.0, spl: 0.0000, os: 0.0, ne: 7.0000, vlm_calls: 1, trajectory_calls: 0\n",
]


def test_parse_client_log_calls_vo_and_final():
    blocks, dup, incomplete = sc.parse_client_log("client_0.log", lines=LOG[:13])
    assert (dup, incomplete) == (0, 0)
    b = blocks[1]
    assert (b["scene_id"], b["line_start"], b["line_end"]) == ("zsNo4HB9uLZ", 3, 13)
    assert [c["call_index"] for c in b["calls"]] == [0, 1, 2]
    c0, c1, c2 = b["calls"]
    assert (c0["step"], c0["kind"], c0["vlm_output"], c0["actions"], c0["traj_goal"]) == \
        (0, "native_actions", "←←←", [2, 2, 2, 0], None)
    assert (c0["vo_frame"], c0["vo_history"], c0["vo_ready"], c0["vo_phase"]) == (0, [], False, "insufficient_history")
    assert (c1["kind"], c1["vlm_output"], c1["actions"], c1["traj_goal"]) == ("trajectory", "180 176", [2, 1, 1, 1],
                                                                              [2.77, -1.07])
    assert (c1["vo_frame"], c1["vo_history"][-1], c1["vo_ready"], c1["vo_revision"]) == (21, 20, True, 2)
    # no [amb3r-vo] line before the third call: nothing is carried over from the previous one
    assert (c2["kind"], c2["actions"], c2["vo_frame"], c2["vo_ready"]) == ("stop", [0], None, None)
    assert b["final"] == {"success": 1.0, "spl": 1.0, "os": 1.0, "ne": 0.5122, "vlm_calls": 3, "trajectory_calls": 1}


def test_parse_client_log_counts_physical_lines_and_handles_reruns():
    lines = LOG[:2] + ["\rEvaluating: 0%| | 0/2 [00:00<?, ?it/s]\rEvaluating: 0%| | 0/2 [SPL=?]\n"] + LOG[2:]
    blocks, dup, incomplete = sc.parse_client_log("client_0.log", lines=lines)
    # \r fragments do not shift line numbers: the header is physical line 4 (grep -n numbering)
    assert (dup, incomplete) == (1, 1)
    assert set(blocks) == {1}  # the interrupted episode 9 is dropped
    b = blocks[1]
    assert b["line_start"] == len(lines) - 2  # the later complete block wins
    assert [(c["kind"], c["vlm_output"]) for c in b["calls"]] == [("stop", "")]
    first, _, _ = sc.parse_client_log("client_0.log", lines=lines[:14])
    assert first[1]["line_start"] == 4


def test_parse_client_log_reads_payload_after_a_tqdm_carriage_return():
    lines = ["\rEvaluating: 0%| | 0/1\r[1/1] Episode A_0003: go\n",
             "  step_id: 0, RPC kind=native_actions, VLM output: →→\n",
             "\rEvaluating: 0%| | 0/1\r  [debug] actions=[3, 3, 0, 0]\n",
             "  => success: 0.0, spl: 0.0000, os: 0.0, ne: 5.0000, vlm_calls: 1, trajectory_calls: 0\n"]
    blocks, _, _ = sc.parse_client_log("client_0.log", lines=lines)
    assert (blocks[3]["line_start"], blocks[3]["line_end"]) == (1, 4)
    assert blocks[3]["calls"][0]["actions"] == [3, 3, 0, 0]


def test_reference_turns_and_dy():
    square = [[0, 0, 0], [0, 0, 2], [2, 0, 2], [2, 0.5, 0]]  # +z, +x, -z: two 90 deg turns
    assert sc.reference_turns(square) == 2
    assert sc.reference_dy(square) == 0.5
    # a segment of <= 0.3 m (xz) is dropped before headings are compared
    jog = [[0, 0, 0], [0, 0, 2], [0.2, 0, 2.1], [0, 0, 4]]
    assert sc.reference_turns(jog) == 0
    gentle = [[0, 0, 0], [0, 0, 2], [0.7, 0, 3.9]]  # ~20 deg
    assert sc.reference_turns(gentle) == 0
    assert sc.reference_turns([[0, 0, 0], [0, 0, 2]]) == 0
    assert sc.wrap_deg(190) == -170 and sc.wrap_deg(-190) == 170


def test_reference_rooms_and_runs():
    # house frame: (x, y, z) with z up; habitat (x, y, z) -> house (x, -z, y)
    panos = [(0.0, 0.0, 1.4, 5), (0.0, -3.0, 1.4, 6), (0.0, -3.0, 4.4, 9), (10.0, 0.0, 1.4, -1)]
    path = [[0, 0, 0], [0, 0, 1], [0, 0, 3], [0.2, 0, 3], [0, 3, 3], [10, 0, 0], [0, 0, 0]]
    rooms = sc.reference_rooms(panos, path)
    # [0,0,1] is 1 m from its nearest panorama -> unmatched; [0,3,3] only sees the upper-floor panorama
    assert rooms == [5, None, 6, 6, 9, -1, 5]
    assert sc.room_runs(rooms) == [5, 6, 9, 5]
    assert sc.room_runs([None, 3, None, 3, 4, 4, -1, 4]) == [3, 4]


def test_parse_house_reads_panoramas_and_regions(tmp_path):
    house = tmp_path / "x.house"
    house.write_text(
        "ASCII 1.1\n"
        "R 0 1 0 0 b 1.0 2.0 0.1  0 0 0  2 2 2  2.5 0 0 0 0\n"
        "P " + "a" * 32 + " 7 0 0  1.5 -2.0 1.45  0 0 0 0 0\n"
        "P short 8 0 0  9 9 9\n"
    )
    panos, regions = sc.parse_house(house)
    assert panos == [(1.5, -2.0, 1.45, 0)]
    assert regions == {0: (1, "b")}
