"""EXP-23 round 2's cohort: round 1's cohort minus the episodes round 1 committed."""

from __future__ import annotations

import gzip
import json

import pytest

from scripts.tools import remaining_cohort
from scripts.tools.build_r2r_train_cohorts import CohortError, scene_name
from src.data.s2_records import record_path
from tests.test_s2_record_collection_launcher import _build, _shards, _train_json


def _commit(records, scene, episode_id, *, marker=True):
    path = record_path(records, scene, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"tar")
    if marker:
        path.with_name(path.name + ".sha256").write_text("x\n")


def test_the_rest_of_a_cohort_is_a_valid_cohort_of_exactly_the_uncommitted_episodes(tmp_path):
    dataset = tmp_path / "train" / "train.json.gz"
    _train_json(dataset)
    source = tmp_path / "round1" / "cohorts"
    assert _build("--dataset", str(dataset), "--output-dir", str(source), "--num-shards", "3").returncode == 0
    _, lists, _ = _shards(source)
    records = tmp_path / "round1" / "records"
    for e in lists[0]["episodes"]:  # shard 0 done
        _commit(records, e["scene_id"], e["episode_id"])
    half = lists[1]["episodes"][: len(lists[1]["episodes"]) // 2]
    for e in half:  # shard 1 half done
        _commit(records, e["scene_id"], e["episode_id"])
    in_flight = lists[2]["episodes"][0]  # shard 2: one tar without its marker, i.e. not committed
    _commit(records, in_flight["scene_id"], in_flight["episode_id"], marker=False)

    out = tmp_path / "round2" / "cohorts"
    assert remaining_cohort.main(["--cohort", str(source), "--records", str(records), "--out", str(out)]) == 0
    manifest, new_lists, new_datasets = _shards(out)

    assert manifest["schema"] == "heatmapvln-r2r-train-cohorts-v1" and manifest["split"] == "train"
    assert manifest["num_shards"] == 2 == len(new_lists)
    assert [r["source_shard"] for r in manifest["shards"]] == [1, 2]
    assert new_lists[0]["episodes"] == lists[1]["episodes"][len(half):]
    assert new_lists[1]["episodes"] == lists[2]["episodes"]  # the in-flight episode is still to do
    done = len(lists[0]["episodes"]) + len(half)
    total = sum(len(x["episodes"]) for x in lists)
    assert manifest["total_episodes"] == total - done
    assert manifest["derived_from"]["committed_episodes_dropped"] == done
    for index, (listing, data) in enumerate(zip(new_lists, new_datasets)):
        assert listing["shard_index"] == index and listing["num_shards"] == 2
        assert [(scene_name(e["scene_id"]), e["episode_id"]) for e in data["episodes"]] == \
            [(e["scene_id"], e["episode_id"]) for e in listing["episodes"]]
        assert data["instruction_vocab"] == {"word_list": ["go"], "num_vocab": 1}
        assert manifest["shards"][index]["episodes"] == len(listing["episodes"])

    with pytest.raises(CohortError, match="non-empty"):
        remaining_cohort.remaining(source, records, out)


def test_nothing_left_writes_nothing(tmp_path):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    source = tmp_path / "cohorts"
    assert _build("--dataset", str(dataset), "--output-dir", str(source), "--num-shards", "2").returncode == 0
    with gzip.open(dataset, "rt") as handle:
        for e in json.load(handle)["episodes"]:
            _commit(tmp_path / "records", scene_name(e["scene_id"]), e["episode_id"])
    out = tmp_path / "out"
    assert remaining_cohort.main(["--cohort", str(source), "--records", str(tmp_path / "records"), "--out", str(out)]) == 2
    assert not out.exists() and not list(tmp_path.glob(".out.*"))
