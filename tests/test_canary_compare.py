"""EXP-25's canary: its cohort (round 1's committed held-out episodes) and C-A against the base model."""

from __future__ import annotations

import json

import numpy as np

from scripts.tools import canary_compare as cc
from scripts.tools import remaining_cohort
from scripts.tools.build_r2r_train_cohorts import scene_name
from src.data.s2_records import EpisodeRecordWriter
from tests.test_remaining_cohort import _commit
from tests.test_s2_record_collection_launcher import _build, _shards, _train_json
from tests.test_s2_sft_dataset import INSTRUCTION, _jpeg
from tests.test_spinbreak_stream import _answer_calls


def _record(root, episode, plan, *, success, distance, steps):
    writer = EpisodeRecordWriter(root=root, scene_id="sceneA", episode_id=episode,
                                 episode={"instruction": INSTRUCTION, "outcome": {
                                     "success": float(success), "distance_to_goal": distance, "steps": steps}})
    calls = _answer_calls([a for a, _ in plan])
    for step in sorted({s for c in calls for s in c["history_step_ids"] + [c["step_id"]]}):
        writer.add_step(step, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(step, _jpeg(step))
    for call, (_, d) in zip(calls, plan):
        writer.add_call({"kind": "x", "distance_to_goal": d, **call}, lookdown_jpeg=_jpeg(77),
                        lookdown_depth_m=np.zeros((480, 640), dtype=np.float32))
    return writer.commit()


def test_c_a_reads_endings_reversals_and_sr_from_the_records(tmp_path):
    base, canary = tmp_path / "base", tmp_path / "canary"
    # episode 1: both stop at the goal
    for root in (base, canary):
        _record(root, 1, [("pixel", 4.0), ("stop", 0.5)], success=1, distance=0.5, steps=40)
    # episode 2: the base spins out of steps at the goal; the canary stops in the spin
    _record(base, 2, [("left", 0.9), ("left", 0.8), ("left", 0.7)], success=0, distance=0.7, steps=500)
    _record(canary, 2, [("left", 0.9), ("left", 0.8), ("stop", 0.7)], success=1, distance=0.7, steps=60)
    # episode 3: the canary stops 4 m short
    _record(base, 3, [("pixel", 6.0), ("stop", 1.0)], success=1, distance=1.0, steps=50)
    _record(canary, 3, [("pixel", 6.0), ("stop", 4.0)], success=0, distance=4.0, steps=30)
    # episode 4: the base stops far; the canary jitters left/right out of steps (a step-budget
    # STOP the client issued within 3 m: Habitat says success, the standard protocol does not)
    _record(base, 4, [("pixel", 5.0), ("stop", 5.0)], success=0, distance=5.0, steps=30)
    _record(canary, 4, [("left", 2.0), ("right", 2.0), ("left", 2.0)], success=1, distance=2.0, steps=500)

    cohort = {("sceneA", e) for e in (1, 2, 3, 4)}
    result = cc.compare(cc.load(canary), cc.load(base), cohort)
    c, b = result["canary"], result["base"]
    assert result["matched_episodes"] == 4 and result["complete"]
    assert (c["stop_far"], b["stop_far"]) == (1, 1)
    assert (c["out_of_steps"], b["out_of_steps"]) == (1, 1)
    assert (c["sr_pct"], b["sr_pct"]) == (50.0, 50.0) and c["sr_recorded_pct"] == 75.0
    assert (c["turn_calls"], c["reversals"]) == (5, 2) and b["reversals"] == 0
    assert (c["spin_states"], c["spin_stop_rate"]) == (2, 0.5) and b["spin_stop_rate"] == 0.0
    crit = result["criteria"]
    assert crit["stop_far_episodes"]["pass"] and crit["out_of_steps_episodes"]["pass"] and crit["sr_pct"]["pass"]
    assert not crit["turn_reversal_share"]["pass"] and not result["all_pass"]
    assert result["report_only"]["paired_flips"] == {"base_success_canary_fail": 1, "base_fail_canary_success": 1}

    incomplete = cc.compare(cc.load(canary), cc.load(base), cohort | {("sceneA", 9)})
    assert not incomplete["complete"] and incomplete["missing_in_canary"] == ["sceneA_000009"]
    assert cc.main(["--canary-records", str(canary), "--base-records", str(base)]) == 1


def test_canary_cohort_is_the_committed_episodes_of_the_chosen_scenes(tmp_path):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    source = tmp_path / "round1" / "cohorts"
    assert _build("--dataset", str(dataset), "--output-dir", str(source), "--num-shards", "3").returncode == 0
    _, lists, _ = _shards(source)
    records = tmp_path / "round1" / "records"
    committed = set()
    for listing in lists:
        for e in listing["episodes"][::2]:
            _commit(records, e["scene_id"], e["episode_id"])
            committed.add((e["scene_id"], e["episode_id"]))
    want = {k for k in committed if k[0] in ("sceneA", "sceneC")}

    out = tmp_path / "canary"
    argv = ["--cohort", str(source), "--records", str(records), "--out", str(out), "--keep", "committed",
            "--scenes", "sceneA,sceneC", "--num-shards", "2"]
    assert remaining_cohort.main(argv) == 0
    manifest, new_lists, new_datasets = _shards(out)
    got = {(e["scene_id"], e["episode_id"]) for listing in new_lists for e in listing["episodes"]}
    assert got == want and manifest["total_episodes"] == len(want) and manifest["num_shards"] == 2
    assert manifest["derived_from"]["keep"] == "committed" and manifest["derived_from"]["scenes"] == ["sceneA", "sceneC"]
    owner = {}
    for index, (listing, data) in enumerate(zip(new_lists, new_datasets)):
        assert listing["shard_index"] == index and listing["num_shards"] == 2 and listing["source_shards"]
        assert [(scene_name(e["scene_id"]), e["episode_id"]) for e in data["episodes"]] == \
            [(e["scene_id"], e["episode_id"]) for e in listing["episodes"]]
        assert data["instruction_vocab"] == {"word_list": ["go"], "num_vocab": 1}
        for e in listing["episodes"]:  # routes stay whole
            assert owner.setdefault((e["scene_id"], e["trajectory_id"]), index) == index
    # without regrouping the source shards are kept, renumbered
    kept = tmp_path / "canary_src"
    assert remaining_cohort.main(["--cohort", str(source), "--records", str(records), "--out", str(kept),
                                  "--keep", "committed", "--scenes", "sceneA,sceneC"]) == 0
    manifest2, lists2, _ = _shards(kept)
    assert {(e["scene_id"], e["episode_id"]) for x in lists2 for e in x["episodes"]} == want
    assert all("source_shard" in r for r in manifest2["shards"])
    assert json.loads((kept / "manifest.json").read_text())["derived_from"]["regrouped_num_shards"] is None
