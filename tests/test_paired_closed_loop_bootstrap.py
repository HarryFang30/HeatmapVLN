"""paired_closed_loop_bootstrap: (scene, episode) keys, a scene-cluster bootstrap, the standard protocol.

The A0-style pairing (MP3D val_unseen, unique episode ids, two pooled seeds, a geodesic
stratum) must give exactly the numbers it gave before EXP-28's changes: the golden below
is what the script wrote at 1427292 on the same inputs.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest

from scripts.tools import paired_closed_loop_bootstrap as pb

MP3D_VAL_UNSEEN = ("2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP", "X7HyMhZNoso",
                   "Z6MFQCViBuw", "oLBMNvg9in8", "pLe4wQe7qrG", "x8F5xyUWy9e", "zsNo4HB9uLZ")


def _mp3d_like_inputs(root):
    """Two protocol seeds of an A0-style pairing: unique episode ids, rows out of order, one unpaired each."""
    rng = random.Random(20261015)
    ids = rng.sample(range(1, 4000), 90)
    scene_of = {e: MP3D_VAL_UNSEEN[rng.randrange(len(MP3D_VAL_UNSEEN))] for e in ids}
    geodesic = {e: round(rng.uniform(4.0, 18.0), 3) for e in ids}
    paths = {}
    for seed in (42, 1337):
        for arm, p_success in (("treatment", 0.66), ("control", 0.6)):
            rows = []
            for e in ids:
                success = float(rng.random() < p_success)
                rows.append({"scene_id": scene_of[e], "episode_id": e, "success": success,
                             "spl": round(success * rng.uniform(0.5, 1.0), 6), "os": float(success or rng.random() < 0.2),
                             "ne": round(rng.uniform(0.2, 2.9) if success else rng.uniform(3.0, 12.0), 4)})
            rng.shuffle(rows)
            if (seed, arm) == (42, "control"):
                rows = rows[1:]
            if (seed, arm) == (1337, "treatment"):
                rows = rows[:-1]
            path = Path(root) / f"{arm}_{seed}" / "progress.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths[(arm, seed)] = path
    dataset = Path(root) / "val_unseen.json.gz"
    with gzip.open(dataset, "wt") as handle:
        json.dump({"episodes": [{"episode_id": e, "scene_id": f"mp3d/{scene_of[e]}/{scene_of[e]}.glb",
                                 "info": {"geodesic_distance": geodesic[e]}} for e in sorted(ids)]}, handle)
    return paths, dataset


def _argv(paths, dataset, out, *extra):
    return ["--treatment", str(paths[("treatment", 42)]), "--control", str(paths[("control", 42)]),
            "--treatment", str(paths[("treatment", 1337)]), "--control", str(paths[("control", 1337)]),
            "--dataset", str(dataset), "--geodesic-min", "10", "--bootstrap-draws", "300",
            "--bootstrap-seed", "0", "--label", "a0_style", "--output-json", str(out), *extra]


# The json at 1427292 (minus the two input-path lists), same inputs and argv.
A0_GOLDEN = {
    "bootstrap_draws": 300, "bootstrap_seed": 0, "label": "a0_style", "paired_episodes": 178, "pairs_pooled": 2,
    "strata": {
        "all": {
            "n": 178,
            "ne": {"ci95": [-1.1293764044943828, 0.15993820224719113], "control": 3.63204213483146,
                   "diff": -0.5107382022471911, "treatment": 3.1213039325842726},
            "os": {"ci95": [-0.0898876404494382, 0.08426966292134831], "control": 0.7640449438202247,
                   "diff": -0.0056179775280898875, "treatment": 0.7584269662921348},
            "spl": {"ci95": [-0.0390187191011236, 0.10937092134831461], "control": 0.47783217415730317,
                    "diff": 0.03611515168539326, "treatment": 0.5139473258426964},
            "success": {"ci95": [-0.07303370786516854, 0.1404494382022472], "control": 0.6573033707865169,
                        "diff": 0.03932584269662921, "treatment": 0.6966292134831461},
        },
        "geodesic_ge_10m": {
            "n": 94,
            "ne": {"ci95": [-1.2444840425531916, 0.5698106382978723], "control": 3.4135255319148943,
                   "diff": -0.4125787234042553, "treatment": 3.000946808510638},
            "os": {"ci95": [-0.14893617021276595, 0.10638297872340426], "control": 0.776595744680851,
                   "diff": -0.02127659574468085, "treatment": 0.7553191489361702},
            "spl": {"ci95": [-0.09365556382978724, 0.11715262765957443], "control": 0.49689169148936163,
                    "diff": 0.018264404255319153, "treatment": 0.5151560957446809},
            "success": {"ci95": [-0.14893617021276595, 0.1276595744680851], "control": 0.7021276595744681,
                        "diff": 0.0, "treatment": 0.7021276595744681},
        },
    },
    "success_overlap": {"both": 80, "control_only": 37, "treatment_only": 44},
    "unpaired_episodes": 2,
}
A0_TABLE = """\
# a0_style: 178 paired episodes from 2 pair(s) (2 unpaired dropped)
| stratum | n | SR | SPL | OS | NE |
|---|---|---|---|---|---|
| all | 178 | 69.66 vs 65.73 → +3.93 [-7.30, +14.04] | 51.39 vs 47.78 → +3.61 [-3.90, +10.94] | 75.84 vs 76.40 → -0.56 [-8.99, +8.43] | 3.12 m vs 3.63 m → -0.51 [-1.13, +0.16] |
| geodesic_ge_10m | 94 | 70.21 vs 70.21 → +0.00 [-14.89, +12.77] | 51.52 vs 49.69 → +1.83 [-9.37, +11.72] | 75.53 vs 77.66 → -2.13 [-14.89, +10.64] | 3.00 m vs 3.41 m → -0.41 [-1.24, +0.57] |
success overlap: both=80 treatment_only=44 control_only=37
"""


def _report(out):
    report = json.loads(Path(out).read_text())
    report.pop("treatment")
    report.pop("control")
    return report


def test_a0_style_pairing_reads_exactly_as_before(tmp_path, capsys):
    paths, dataset = _mp3d_like_inputs(tmp_path)
    out = tmp_path / "report.json"
    pb.main(_argv(paths, dataset, out))
    assert _report(out) == A0_GOLDEN
    printed = capsys.readouterr().out
    assert printed == A0_TABLE + f"wrote {out}\n"

    # the same rows without scene_id (older runs) pair by episode id alone: same numbers
    for path in list(paths.values()):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "scene_id"}) + "\n" for r in rows))
    pb.main(_argv(paths, dataset, out))
    assert _report(out) == A0_GOLDEN


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _row(scene, episode, success, **extra):
    return {"scene_id": scene, "episode_id": episode, "success": float(success), "spl": 0.8 * success,
            "os": 1.0, "ne": 1.0 if success else 5.0, **extra}


def test_episode_ids_that_repeat_across_scenes_pair_within_their_scene(tmp_path, capsys):
    # HM3D-style: episode 1 exists in two scenes, with opposite outcomes
    t = _write(tmp_path / "t.jsonl", [_row("00aaa", 1, 1), _row("00bbb", 1, 0), _row("00bbb", 2, 1)])
    c = _write(tmp_path / "c.jsonl", [_row("00bbb", 1, 1), _row("00aaa", 1, 0), _row("00ccc", 3, 1)])
    t_rows, c_rows, pair = pb.pair_rows(pb.load_progress(t), pb.load_progress(c), t, c)
    assert pair == [("00aaa", "1"), ("00bbb", "1")]
    out = tmp_path / "r.json"
    pb.main(["--treatment", str(t), "--control", str(c), "--bootstrap-draws", "50", "--output-json", str(out)])
    report = json.loads(out.read_text())
    assert report["paired_episodes"] == 2 and report["unpaired_episodes"] == 2
    assert report["strata"]["all"]["success"]["diff"] == 0.0
    assert report["success_overlap"] == {"both": 0, "treatment_only": 1, "control_only": 1}
    capsys.readouterr()


def test_a_key_twice_or_two_datasets_is_an_error(tmp_path):
    twice = _write(tmp_path / "twice.jsonl", [_row("s", 1, 1), _row("s", 2, 0), _row("s", 1, 0)])
    with pytest.raises(SystemExit, match="appears twice"):
        pb.load_progress(twice)
    # without scene_id an episode id names the episode, so it must be unique
    a = _write(tmp_path / "a.jsonl", [{k: v for k, v in _row(s, 1, 1).items() if k != "scene_id"} for s in "xy"])
    with pytest.raises(SystemExit, match="appears twice"):
        pb.load_progress(a)
    mixed = _write(tmp_path / "mixed.jsonl", [_row("x", 1, 1), {k: v for k, v in _row("y", 1, 1).items()
                                                                 if k != "scene_id"}])
    ok = _write(tmp_path / "ok.jsonl", [_row("x", 1, 1)])
    with pytest.raises(SystemExit, match="appears in two scenes"):
        pb.pair_rows(pb.load_progress(mixed), pb.load_progress(ok), mixed, ok)
    # unique episode ids that sit in other scenes on the other arm: two different datasets
    t = _write(tmp_path / "t.jsonl", [_row("x", 1, 1), _row("x", 2, 1)])
    c = _write(tmp_path / "c.jsonl", [_row("y", 1, 1), _row("x", 2, 0)])
    with pytest.raises(SystemExit, match="ran different datasets"):
        pb.pair_rows(pb.load_progress(t), pb.load_progress(c), t, c)


def _clustered(tmp_path, seeds=(42,)):
    """Six scenes of ten episodes; the treatment's whole gain is in one scene."""
    paths = []
    for seed in seeds:
        t_rows, c_rows = [], []
        for s in range(6):
            for e in range(10):
                eid = 1000 * s + e
                t_rows.append(_row(f"scene{s}", eid, 1 if s == 0 else e % 2))
                c_rows.append(_row(f"scene{s}", eid, e % 2))
        paths.append((_write(tmp_path / f"t{seed}.jsonl", t_rows), _write(tmp_path / f"c{seed}.jsonl", c_rows)))
    return paths


def _run(paths, out, *extra):
    argv = []
    for t, c in paths:
        argv += ["--treatment", str(t), "--control", str(c)]
    pb.main(argv + ["--bootstrap-draws", "2000", "--output-json", str(out), *extra])
    return json.loads(Path(out).read_text())


def test_scene_cluster_bootstrap_resamples_whole_scenes(tmp_path, capsys):
    paths = _clustered(tmp_path)
    episode = _run(paths, tmp_path / "e.json")
    scene = _run(paths, tmp_path / "s.json", "--bootstrap-unit", "scene")
    assert "bootstrap_unit" not in episode and scene["bootstrap_unit"] == "scene"
    e_sr, s_sr = episode["strata"]["all"]["success"], scene["strata"]["all"]["success"]
    assert e_sr["diff"] == s_sr["diff"] == pytest.approx(5 / 60)
    assert scene["strata"]["all"]["clusters"] == 6 and "clusters" not in episode["strata"]["all"]
    # the gain lives in one scene of six: a draw holds it 0..6 times, so the interval reaches 0
    assert s_sr["ci95"][0] == 0.0 and e_sr["ci95"][0] > 0.0
    assert s_sr["ci95"][1] - s_sr["ci95"][0] > e_sr["ci95"][1] - e_sr["ci95"][0]
    assert "bootstrap over whole scenes (6 clusters)" in capsys.readouterr().out
    assert _run(paths, tmp_path / "s2.json", "--bootstrap-unit", "scene")["strata"] == scene["strata"]

    # two pooled seeds: a scene is one cluster across both
    pooled = _run(_clustered(tmp_path, seeds=(42, 1337)), tmp_path / "p.json", "--bootstrap-unit", "scene")
    assert pooled["paired_episodes"] == 120 and pooled["strata"]["all"]["clusters"] == 6

    # one scene: every draw is that scene, the interval is the point
    one = [(_write(tmp_path / "t1.jsonl", [_row("s", 1, 1), _row("s", 2, 0)]),
            _write(tmp_path / "c1.jsonl", [_row("s", 1, 0), _row("s", 2, 0)]))]
    single = _run(one, tmp_path / "one.json", "--bootstrap-unit", "scene")["strata"]["all"]["success"]
    assert single["ci95"] == [0.5, 0.5]

    # rows without scene_id cannot be clustered by scene
    bare = [(_write(tmp_path / "tb.jsonl", [{"episode_id": 1, "success": 1, "spl": 1, "os": 1, "ne": 1}]),
             _write(tmp_path / "cb.jsonl", [{"episode_id": 1, "success": 0, "spl": 0, "os": 1, "ne": 4}]))]
    with pytest.raises(SystemExit, match="needs scene_id"):
        _run(bare, tmp_path / "b.json", "--bootstrap-unit", "scene")
    capsys.readouterr()


def test_standard_protocol_scores_a_step_budget_stop_as_no_stop(tmp_path, capsys):
    budget = {"stop_at_step_budget": True}
    t = _write(tmp_path / "t.jsonl", [_row("s", 1, 1, step_budget_stop=True, **budget), _row("s", 2, 1, **budget),
                                      _row("s", 3, 0, step_budget_stop=False, **budget)])
    c = _write(tmp_path / "c.jsonl", [_row("s", 1, 1), _row("s", 2, 0), _row("s", 3, 0)])
    switched_on = _run([(t, c)], tmp_path / "on.json")
    assert switched_on["strata"]["all"]["success"]["treatment"] == 2 / 3
    # a switched-on run says so in the table and the json, not only on stderr
    assert switched_on["protocol"].startswith("switched on") and switched_on["step_budget_rows"] == {
        "treatment": 3, "control": 0}
    captured = capsys.readouterr()
    assert "--standard-protocol" in captured.err and "SWITCHED-ON protocol" in captured.out
    standard = _run([(t, c)], tmp_path / "std.json", "--standard-protocol")
    assert standard["strata"]["all"]["success"]["treatment"] == 1 / 3
    assert standard["strata"]["all"]["spl"]["treatment"] == pytest.approx(0.8 / 3)
    assert standard["step_budget_stops_rescored"] == {"treatment": 1, "control": 0}
    assert standard["step_budget_rows"] == {"treatment": 3, "control": 0}
    assert standard["success_overlap"] == {"both": 0, "treatment_only": 1, "control_only": 1}
    assert capsys.readouterr().err == ""

    # rows that lost the switch's fields: nothing is rescored, and the run says so
    lost = _write(tmp_path / "lost.jsonl", [_row("s", 1, 1), _row("s", 2, 1), _row("s", 3, 0)])
    nothing = _run([(lost, c)], tmp_path / "lost.json", "--standard-protocol")
    assert nothing["step_budget_rows"] == {"treatment": 0, "control": 0}
    assert nothing["strata"]["all"]["success"]["treatment"] == 2 / 3
    assert "nothing is rescored" in capsys.readouterr().err


def test_other_protocol_seeds_are_reported(tmp_path, capsys):
    t = _write(tmp_path / "t.jsonl", [_row("s", e, 1, rpc_protocol_seed=42) for e in (1, 2)])
    same = _write(tmp_path / "same.jsonl", [_row("s", e, 0, rpc_protocol_seed=42) for e in (1, 2)])
    other = _write(tmp_path / "other.jsonl", [_row("s", e, 0, rpc_protocol_seed=1337) for e in (1, 2)])
    assert "protocol_seed_mismatch" not in _run([(t, same)], tmp_path / "a.json")
    assert capsys.readouterr().err == ""
    report = _run([(t, same), (t, other)], tmp_path / "b.json")
    assert report["protocol_seed_mismatch"] == {"pair 1: treatment seed 42 vs control seed 1337": 2}
    captured = capsys.readouterr()
    assert "OTHER PROTOCOL SEEDS" in captured.out and "other protocol seeds" in captured.err


def test_geodesic_strata_follow_the_scene(tmp_path, capsys):
    t = _write(tmp_path / "t.jsonl", [_row("00aaa", 1, 1), _row("00bbb", 1, 1)])
    c = _write(tmp_path / "c.jsonl", [_row("00aaa", 1, 0), _row("00bbb", 1, 1)])
    dataset = tmp_path / "train.json.gz"
    with gzip.open(dataset, "wt") as handle:
        json.dump({"episodes": [
            {"episode_id": 1, "scene_id": "hm3d/00aaa/00aaa.glb", "info": {"geodesic_distance": 12.0}},
            {"episode_id": 1, "scene_id": "hm3d/00bbb/00bbb.glb", "info": {"geodesic_distance": 4.0}},
        ]}, handle)
    report = _run([(t, c)], tmp_path / "g.json", "--dataset", str(dataset), "--geodesic-min", "10")
    far = report["strata"]["geodesic_ge_10m"]
    assert far["n"] == 1 and far["success"]["diff"] == 1.0
    capsys.readouterr()


def test_scene_name_matches_the_cohort_builder():
    from scripts.tools.build_r2r_train_cohorts import scene_name

    for raw in ("mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb", "hm3d/5jLhtVmWd5F/5jLhtVmWd5F.glb",
                "/dataset/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"):
        assert pb.scene_name(raw) == scene_name(raw)
