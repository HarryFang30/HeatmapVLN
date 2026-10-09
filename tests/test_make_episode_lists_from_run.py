"""scripts/tools/make_episode_lists_from_run.py: pinning a canary to a reference run's exact episodes.

The written lists are read back with the client's own ``_load_episode_list``, executed out of
scripts/evaluation/r2r_val_unseen.py (importing the client would pull in habitat).  Torch-free:
  python3 -m pytest tests/test_make_episode_lists_from_run.py -q --noconftest -p no:cacheprovider
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools import make_episode_lists_from_run as tool

REPO = Path(__file__).resolve().parents[1]
CLIENT = REPO / "scripts" / "evaluation" / "r2r_val_unseen.py"
STAMP = "20260928_212453_1550"
# Shard 0's cohort order is ep 10, 11, 12, 13; the reference recorded 12 then 10, so cohort order differs.
COHORT = {
    0: [("sceneA", 10), ("sceneA", 11), ("sceneB", 12), ("sceneB", 13)],
    1: [("sceneC", 20), ("sceneC", 21), ("sceneD", 22)],
}
RECORDED = {0: [("sceneB", 12), ("sceneA", 10)], 1: [("sceneC", 21), ("sceneD", 22)]}


def _client_load_episode_list():
    source = CLIENT.read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_load_episode_list"]
    assert len(nodes) == 1
    code = "from __future__ import annotations\n" + ast.get_source_segment(source, nodes[0])
    namespace = {"json": json, "Path": Path}
    exec(compile(code, str(CLIENT), "exec"), namespace)
    return namespace["_load_episode_list"]


def _cohorts(root: Path) -> Path:
    root.mkdir(parents=True)
    for shard, keys in COHORT.items():
        eps = [{"scene_id": s, "episode_id": e, "trajectory_id": 1000 + e} for s, e in keys]
        (root / f"shard_0{shard}.json").write_text(json.dumps(
            {"cohort_name": f"rr_{shard:02d}_of_08", "dataset_sha256": "d" * 64, "num_shards": 8,
             "shard_index": shard, "episodes": eps}))
    return root


def _row(scene: str, episode: int) -> str:
    return json.dumps({"scene_id": scene, "episode_id": episode, "success": 1.0, "spl": 0.9, "os": 1.0,
                       "ne": 0.5, "history_pose_source": "amb3r_vo_da3", "ppa_applied_calls": 3})


def _client_log(starts) -> str:
    lines = []
    for done in starts:
        lines += ["Fixed episode list (4): /x/shard_00.json",
                  f"Episodes already done: {done}, remaining: {4 - done}, this run: 2", "  => success: 1"]
    return "\n".join(lines) + "\n"


def _reference(root: Path, recorded=RECORDED, starts=(0,)) -> Path:
    logs = root / "runtime" / STAMP / "logs"
    logs.mkdir(parents=True)
    for shard, keys in recorded.items():
        worker = root / "workers" / f"shard_0{shard}"
        worker.mkdir(parents=True)
        (worker / "progress.json").write_text("".join(_row(*k) + "\n" for k in keys))
        (logs / f"client_shard_0{shard}.log").write_text(_client_log(starts))
    (logs / "model_0.log").write_text("Formal PPA online AMB3R runtime enabled\n")
    return root


def _run(tmp_path: Path, reference: Path, out: Path | None = None, shards: str = "0,1", expect: int = 2):
    out = out or tmp_path / "lists"
    return tool.main(["--reference", str(reference), "--cohorts", str(tmp_path / "cohorts"), "--shards", shards,
                      "--expect-per-shard", str(expect), "--out", str(out)]), out


@pytest.fixture()
def cohorts(tmp_path):
    return _cohorts(tmp_path / "cohorts")


def test_lists_are_what_the_client_loads_in_cohort_order_copied_from_the_cohort(tmp_path, cohorts):
    reference = _reference(tmp_path / "canary_cuda_seed42")
    code, out = _run(tmp_path, reference)
    assert code == 0
    load = _client_load_episode_list()
    for shard, want in ((0, [("sceneA", 10), ("sceneB", 12)]), (1, [("sceneC", 21), ("sceneD", 22)])):
        keys, key_set = load(str(out / f"shard_0{shard}.json"))
        assert keys == want and key_set == set(RECORDED[shard])
        pinned = json.loads((out / f"shard_0{shard}.json").read_text())
        cohort = json.loads((cohorts / f"shard_0{shard}.json").read_text())
        assert pinned["episodes"] == [e for e in cohort["episodes"] if (e["scene_id"], e["episode_id"]) in want]
        for key in ("dataset_sha256", "num_shards", "shard_index"):
            assert pinned[key] == cohort[key]
        assert pinned["parent_cohort"] == cohort["cohort_name"] != pinned["cohort_name"]
        assert pinned["pinned_from_reference"] == str(reference.resolve())

    manifest = json.loads((out / "manifest.json").read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert manifest["reference"] == str(reference.resolve()) and manifest["runtime_stamp"] == STAMP
    assert manifest["expect_per_shard"] == 2 and [s["shard"] for s in manifest["shards"]] == [0, 1]
    for entry in manifest["shards"]:
        shard = entry["shard"]
        assert entry["progress"]["path"] == str(reference.resolve() / "workers" / f"shard_0{shard}" / "progress.json")
        assert entry["client_log"]["path"].endswith(f"runtime/{STAMP}/logs/client_shard_0{shard}.log")
        for item in ("progress", "client_log", "cohort", "output"):
            assert entry[item]["sha256"] == sha(entry[item]["path"])
        assert entry["cohort"]["path"] == str(cohorts.resolve() / f"shard_0{shard}.json")
        assert [(e["scene_id"], e["episode_id"]) for e in entry["episodes"]] == load(entry["output"]["path"])[0]
    assert manifest["tool"]["sha256"] == sha(REPO / "scripts" / "tools" / "make_episode_lists_from_run.py")
    assert "commit" in manifest["tool"]


def test_a_bare_list_cohort_is_accepted(tmp_path, cohorts):
    (cohorts / "shard_01.json").write_text(json.dumps([{"scene_id": s, "episode_id": e} for s, e in COHORT[1]]))
    code, out = _run(tmp_path, _reference(tmp_path / "ref"))
    assert code == 0
    assert _client_load_episode_list()(str(out / "shard_01.json"))[0] == [("sceneC", 21), ("sceneD", 22)]


def _refused(tmp_path, capsys, reference, match, **kwargs):
    code, out = _run(tmp_path, reference, **kwargs)
    assert code == 2
    assert match in capsys.readouterr().err
    return out


@pytest.mark.parametrize("recorded, match", [
    ({0: RECORDED[0] + [("sceneA", 11)], 1: RECORDED[1]}, "holds 3 rows / 3 unique episodes, expected exactly 2"),
    ({0: RECORDED[0][:1], 1: RECORDED[1]}, "holds 1 rows / 1 unique episodes, expected exactly 2"),
    ({0: RECORDED[0] + RECORDED[0][:1], 1: RECORDED[1]}, "holds 3 rows / 2 unique episodes, expected exactly 2"),
])
def test_refuses_a_shard_without_exactly_the_expected_episodes(tmp_path, cohorts, capsys, recorded, match):
    out = _refused(tmp_path, capsys, _reference(tmp_path / "ref", recorded=recorded), match)
    assert not out.exists()


def test_refuses_a_torn_progress_row(tmp_path, cohorts, capsys):
    reference = _reference(tmp_path / "ref")
    with open(reference / "workers" / "shard_01" / "progress.json", "a") as fh:
        fh.write('{"scene_id": "sceneD", "episo')
    _refused(tmp_path, capsys, reference, "progress row 3 is not valid JSON")


def test_refuses_more_than_one_launch_stamp(tmp_path, cohorts, capsys):
    reference = _reference(tmp_path / "ref")
    (reference / "runtime" / "20260929_010101_77" / "logs").mkdir(parents=True)
    _refused(tmp_path, capsys, reference, "must hold exactly one launch stamp, found 2")


def test_refuses_a_run_without_a_launch_stamp(tmp_path, cohorts, capsys):
    reference = _reference(tmp_path / "ref")
    for log in (reference / "runtime" / STAMP / "logs").iterdir():
        log.unlink()
    (reference / "runtime" / STAMP / "logs").rmdir()
    (reference / "runtime" / STAMP).rmdir()
    _refused(tmp_path, capsys, reference, "found 0 (none)")


@pytest.mark.parametrize("starts, match", [
    ((1,), "resumed start ('Episodes already done: 1', expected 0)"),
    ((0, 1), "shows 2 client starts"),
    ((), "never reports 'Episodes already done'"),
])
def test_refuses_a_client_log_that_did_not_start_once_from_nothing(tmp_path, cohorts, capsys, starts, match):
    _refused(tmp_path, capsys, _reference(tmp_path / "ref", starts=starts), match)


def test_refuses_a_missing_client_log(tmp_path, cohorts, capsys):
    reference = _reference(tmp_path / "ref")
    (reference / "runtime" / STAMP / "logs" / "client_shard_01.log").unlink()
    _refused(tmp_path, capsys, reference, "reference client log is missing")


def test_refuses_a_reference_episode_outside_the_cohort(tmp_path, cohorts, capsys):
    recorded = {0: RECORDED[0], 1: [("sceneC", 21), ("sceneA", 10)]}  # ep 10 belongs to shard 0's cohort
    out = _refused(tmp_path, capsys, _reference(tmp_path / "ref", recorded=recorded),
                   "shard 1: reference episodes not in the cohort file")
    assert not out.exists()


def test_never_overwrites_a_pinned_list(tmp_path, cohorts, capsys):
    reference = _reference(tmp_path / "ref")
    code, out = _run(tmp_path, reference)
    assert code == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    _refused(tmp_path, capsys, reference, "refusing to overwrite a pinned list")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before

    # Even one existing target is enough, and nothing else gets written next to it.
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "manifest.json").write_text("{}")
    _refused(tmp_path, capsys, reference, "manifest.json", out=lone)
    assert sorted(p.name for p in lone.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize("shards, expect, match", [
    ("0,0", 2, "distinct shards"), ("8", 2, "distinct shards"), ("a", 2, "comma-separated"),
    ("0,1", 0, "--expect-per-shard must be positive"),
])
def test_refuses_bad_arguments(tmp_path, cohorts, capsys, shards, expect, match):
    _refused(tmp_path, capsys, _reference(tmp_path / "ref"), match, shards=shards, expect=expect)
