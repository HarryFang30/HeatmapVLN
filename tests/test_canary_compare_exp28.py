"""canary_compare for EXP-28: (scene, episode) keys, the proxy P's lines, the scene bootstrap, per-call identity.

EXP-25's C-A must read exactly as it did before these were added (the golden below is
what the tool printed at 1427292 on the same records).  Nothing here needs torch or
habitat; only the pin against ``s2_sft_dataset.record_device`` skips without torch.
"""

from __future__ import annotations

import json

import pytest

from scripts.tools import canary_compare as cc
from src.data.s2_records import EpisodeRecord, EpisodeRecordWriter

CALL = {
    "pixel": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1,
              "kind": "native_pixel", "pixel_goal": [245, 320], "actions": [1, 1, 1], "terminal": False},
    "left": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0,
             "kind": "native_actions", "pixel_goal": None, "actions": [2, 2], "terminal": False},
    "right": {"llm_output": "→", "native_first_output": "→", "native_lookdown_turns": 0,
              "kind": "native_actions", "pixel_goal": None, "actions": [3], "terminal": False},
    "stop": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0,
             "kind": "stop", "pixel_goal": None, "actions": [0], "terminal": True},
}


# How the collector records the last primitive step of an episode: (action, phase).
ENDINGS = {
    "model_stop": (0, "rpc_terminal"),       # a STOP answer, executed
    "steps": (1, "local_action"),            # Habitat's step limit after a move
    "budget": (0, "step_budget_stop"),       # the client's STOP for the step budget
    "empty": (0, "rpc_empty_actions"),       # the client's STOP after an answer with no actions
}


def _rec(root, scene, episode, plan, *, distance, steps=40, success=None, oracle=None, device=None,
         client="client-a", tweak=None, ending=None, step_actions=True):
    """plan: [(answer, distance_to_goal)]; ``tweak(row)`` may change a call row before it is written.

    Steps carry the collector's action and phase, the last one ``ENDINGS[ending]`` (default: a
    STOP answer ends by the model's STOP, else the step limit); ``step_actions=False`` writes
    steps as records without them would have.
    """
    stopped = bool(plan) and plan[-1][0] == "stop"
    ending = ending or ("model_stop" if stopped else "steps")
    outcome = {"success": float(ENDINGS[ending][0] == 0 and distance < 3.0) if success is None else float(success),
               "distance_to_goal": distance, "steps": steps, "spl": 0.5, "system2_calls": len(plan),
               "trajectory_calls": 0, "episode_over": True}
    if oracle is not None:
        outcome["oracle_success"] = float(oracle)
    formats = ["json+jpeg"] + ([f"heatmapvln-device:{device}"] if device else [])
    writer = EpisodeRecordWriter(root=root, scene_id=scene, episode_id=episode, episode={
        "instruction": "go to the door", "client_code_sha256": client, "protocol_seed": 42,
        "server": {"model_version": "heatmapvln-r2r", "supported_formats": formats}, "outcome": outcome})

    def step(s, action, phase):
        extra = {"action": action, "phase": phase} if step_actions else {}
        writer.add_step(s, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], **extra)

    for index, (answer, d) in enumerate(plan):
        step_id = 2 * index + 2
        history = list(range(0, step_id, 2))[-3:]
        for s in history + [step_id]:
            if s not in writer._front:
                step(s, None if s == 0 else 1, "start" if s == 0 else "local_action")
                writer.add_front(s, b"front-%d" % s)
        row = {"call_index": index, "step_id": step_id, "history_step_ids": history, "distance_to_goal": d,
               "front_sha256": [f"f{s}" for s in history + [step_id]], "lookdown_sha256": f"l{index}",
               **CALL[answer]}
        if tweak is not None:
            tweak(row)
        writer.add_call(row, lookdown_jpeg=b"lookdown", lookdown_depth_m=None)
    step(2 * len(plan) + 1, *ENDINGS[ending])
    return writer.commit()


def _ca_records(tmp_path):
    """EXP-25-style canary and base roots; returns (canary, base, cohort keys)."""
    base, canary = tmp_path / "base", tmp_path / "canary"
    for root in (base, canary):  # both stop at the goal
        _rec(root, "sceneA", 1, [("pixel", 4.0), ("stop", 0.5)], distance=0.5)
    _rec(base, "sceneA", 2, [("left", 0.9), ("left", 0.8), ("left", 0.7)], distance=0.7, steps=500)
    _rec(canary, "sceneA", 2, [("left", 0.9), ("left", 0.8), ("stop", 0.7)], distance=0.7, steps=60)
    _rec(base, "sceneA", 3, [("pixel", 6.0), ("stop", 1.0)], distance=1.0, steps=50)
    _rec(canary, "sceneA", 3, [("pixel", 6.0), ("stop", 4.0)], distance=4.0, steps=30)
    _rec(base, "sceneA", 4, [("pixel", 5.0), ("stop", 5.0)], distance=5.0, steps=30)
    _rec(canary, "sceneA", 4, [("left", 2.0), ("right", 2.0), ("left", 2.0)], success=1, distance=2.0, steps=500,
         ending="budget")
    # the same episode id in another scene is another episode
    _rec(base, "sceneB", 1, [("pixel", 7.0), ("right", 6.0), ("stop", 6.5)], distance=6.5, steps=80)
    _rec(canary, "sceneB", 1, [("pixel", 7.0), ("pixel", 2.0), ("stop", 1.5)], distance=1.5, steps=70)
    cohort = {("sceneA", e) for e in (1, 2, 3, 4)} | {("sceneB", 1)}
    return canary, base, cohort


# cc.compare(cc.load(canary), cc.load(base), cohort) on _ca_records at 1427292, before EXP-28's changes.
CA_GOLDEN = {
    "schema": "heatmapvln-exp25-canary-compare-v1",
    "matched_episodes": 5,
    "expected_episodes": 5,
    "complete": True,
    "missing_in_canary": [],
    "missing_in_base": [],
    "criteria": {
        "stop_far_episodes": {"value": 1, "threshold": 6, "direction": "<=", "pass": True, "base": 2},
        "out_of_steps_episodes": {"value": 1, "threshold": 5, "direction": "<=", "pass": True, "base": 1},
        "turn_reversal_share": {"value": 0.4, "threshold": 0.02, "direction": "<=", "pass": False, "base": 0.0},
        "sr_pct": {"value": 60.0, "threshold": 39.0, "direction": ">=", "pass": True, "base": 40.0},
    },
    "all_pass": False,
    "canary": {"episodes": 5, "sr_pct": 60.0, "sr_recorded_pct": 80.0, "stop_far": 1, "out_of_steps": 1,
               "out_of_steps_min_steps": 500, "turn_calls": 5, "reversals": 2, "reversal_share": 0.4,
               "spin_states": 2, "spin_stop_rate": 0.5, "mean_calls": 2.6},
    "base": {"episodes": 5, "sr_pct": 40.0, "sr_recorded_pct": 40.0, "stop_far": 2, "out_of_steps": 1,
             "out_of_steps_min_steps": 500, "turn_calls": 4, "reversals": 0, "reversal_share": 0.0,
             "spin_states": 2, "spin_stop_rate": 0.0, "mean_calls": 2.4},
    "report_only": {"paired_flips": {"base_success_canary_fail": 1, "base_fail_canary_success": 2}},
}


def test_c_a_reads_exactly_as_before(tmp_path, capsys):
    canary, base, cohort = _ca_records(tmp_path)
    assert cc.compare(cc.load(canary), cc.load(base), cohort) == CA_GOLDEN
    assert cc.compare(cc.load(canary), cc.load(base), None) == {**CA_GOLDEN, "expected_episodes": None}
    # the CLI's verdict line and json are the same too: no new keys without the new options
    out = tmp_path / "ca.json"
    assert cc.main(["--canary-records", str(canary), "--base-records", str(base), "--output", str(out)]) == 1
    line = capsys.readouterr().out.strip()
    assert line.startswith("FAIL C-A matched=5 complete=True stop_far_episodes=1(base 2, line <=6)")
    assert "ΔSR" not in line
    written = json.loads(out.read_text())
    assert written.pop("inputs") == {"canary_records": str(canary), "base_records": str(base), "cohort": None}
    assert written == {**CA_GOLDEN, "expected_episodes": None}


def test_keys_are_scene_and_episode_and_must_be_unique(tmp_path):
    canary, base, cohort = _ca_records(tmp_path)
    assert {k for k in cc.load(canary)} == cohort  # episode 1 twice, in two scenes
    # a record filed under another scene's directory duplicates that episode
    stray = tmp_path / "stray"
    path = _rec(stray, "sceneA", 1, [("stop", 0.5)], distance=0.5)
    moved = path.parent.parent / "sceneZ" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    path.with_name(path.name + ".sha256").rename(moved.with_name(moved.name + ".sha256"))
    _rec(stray, "sceneA", 1, [("stop", 0.5)], distance=0.5)
    with pytest.raises(ValueError, match="are both scene sceneA episode 1"):
        cc.load(stray)
    # a cohort listing one episode twice
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    for index, episodes in enumerate(([("sceneA", 1), ("sceneB", 1)], [("sceneA", 1)])):
        (cohort_dir / f"shard_{index}.json").write_text(json.dumps(
            {"episodes": [{"scene_id": s, "episode_id": e} for s, e in episodes]}))
    (cohort_dir / "manifest.json").write_text(json.dumps(
        {"shards": [{"path": "shard_0.json"}, {"path": "shard_1.json"}]}))
    with pytest.raises(ValueError, match="listed twice"):
        cc.cohort_episodes(cohort_dir)
    (cohort_dir / "manifest.json").write_text(json.dumps({"shards": [{"path": "shard_0.json"}]}))
    assert cc.cohort_episodes(cohort_dir) == {("sceneA", 1), ("sceneB", 1)}


def _p_records(tmp_path, *, canary_device="npu", base_device="npu"):
    """20 base episodes over 4 scenes, all on the 910B; the canary loses 2 and stops short of the circle in 1."""
    base, canary = tmp_path / "base", tmp_path / "canary"
    keys = []
    for scene_index, scene in enumerate(("00aaa", "00bbb", "00ccc", "00ddd")):
        for episode in range(5):
            eid = 100 * scene_index + episode
            keys.append((scene, eid))
            good = [("pixel", 6.0), ("pixel", 2.5), ("stop", 1.0)]
            _rec(base, scene, eid, good, distance=1.0, oracle=1, device=base_device)
            if (scene, episode) == ("00aaa", 0):  # stops 5 m out, never inside 3 m
                _rec(canary, scene, eid, [("pixel", 6.0), ("stop", 5.0)], distance=5.0, oracle=0, device=canary_device)
            elif (scene, episode) == ("00bbb", 1):  # out of steps, a reversal on the way
                _rec(canary, scene, eid, [("left", 4.0), ("right", 4.0), ("pixel", 3.5)], distance=3.5,
                     steps=500, oracle=0, device=canary_device)
            else:
                _rec(canary, scene, eid, good, distance=1.0, oracle=1, device=canary_device)
    return canary, base, set(keys)


def test_p_lines_bootstrap_and_platform(tmp_path, capsys):
    canary, base, cohort = _p_records(tmp_path)
    boot = {"unit": "scene", "draws": 400, "seed": 0}
    result = cc.compare(cc.load(canary), cc.load(base), cohort, criteria="exp28-p", require_device="npu",
                        bootstrap=boot)
    crit = result["criteria"]
    assert result["schema"] == cc.P_SCHEMA and result["criteria_set"] == "exp28-p"
    assert crit["sr_delta_pt"]["value"] == -10.0 and not crit["sr_delta_pt"]["pass"]  # line: > -3.0
    assert (crit["stop_never_within_3m_delta_pt"]["value"], crit["stop_never_within_3m_delta_pt"]["pass"]) == (5.0, False)
    assert (crit["out_of_steps_delta_pt"]["value"], crit["out_of_steps_delta_pt"]["pass"]) == (5.0, False)
    assert crit["turn_reversal_share"]["value"] == 0.5 and not crit["turn_reversal_share"]["pass"]
    assert crit["platform"]["value"] == ["npu"] and crit["platform"]["pass"]
    assert crit["platform"]["canary"] == {"npu": 20}
    assert not result["all_pass"] and result["verdict"] == "kill"
    assert result["kill_lines_crossed"] == ["sr_delta_pt", "stop_never_within_3m_delta_pt", "out_of_steps_delta_pt",
                                            "turn_reversal_share"]
    report = result["report_only"]
    assert report["os_pct"] == {"canary": 90.0, "base": 100.0} and report["os_delta_pt"] == -10.0
    assert report["client_code_sha256"] == {"canary": ["client-a"], "base": ["client-a"]} and report["same_client_code"]
    interval = report["sr_bootstrap"]
    assert interval["metric"] == "success_habitat_standard"
    assert (interval["unit"], interval["clusters"], interval["episodes"], interval["diff_pt"]) == ("scene", 4, 20, -10.0)
    lo, hi = interval["ci95_pt"]
    # the two losses sit in two of four scenes: a draw of four scenes holds 0..4 of them, ΔSR 0..-20 pt
    assert -20.0 <= lo <= -10.0 <= hi <= 0.0 and lo < hi
    again = cc.compare(cc.load(canary), cc.load(base), cohort, criteria="exp28-p", bootstrap=boot)
    assert again["report_only"]["sr_bootstrap"] == interval  # deterministic in the seed
    episode_level = cc.compare(cc.load(canary), cc.load(base), cohort, criteria="exp28-p",
                               bootstrap={**boot, "unit": "episode"})["report_only"]["sr_bootstrap"]
    assert episode_level["unit"] == "episode" and episode_level["diff_pt"] == -10.0

    # the CLI defaults P to the 910B and 20,000 scene draws; a kill exits 1
    assert cc.main(["--canary-records", str(canary), "--base-records", str(base), "--criteria", "exp28-p",
                    "--bootstrap-draws", "200", "--output", str(tmp_path / "p.json")]) == 1
    line = capsys.readouterr().out
    assert line.startswith("KILL P matched=20") and "platform=['npu']" in line and "scene bootstrap, 4 scenes, 200 draws" in line
    written = json.loads((tmp_path / "p.json").read_text())
    assert written["criteria"]["platform"]["pass"] and written["report_only"]["sr_bootstrap"]["draws"] == 200


def test_p_passes_within_its_lines_and_fails_on_a_mixed_platform(tmp_path, capsys):
    base, canary = tmp_path / "base", tmp_path / "canary"
    keys = set()
    for scene in ("00aaa", "00bbb"):
        for eid in range(50):
            keys.add((scene, eid))
            plan = [("pixel", 6.0), ("stop", 1.0)]
            _rec(base, scene, eid, plan, distance=1.0, oracle=1, device="npu")
            if (scene, eid) == ("00aaa", 0):  # one loss in 100: -1 pt
                _rec(canary, scene, eid, [("pixel", 6.0), ("stop", 4.0)], distance=4.0, oracle=0, device="npu")
            else:
                _rec(canary, scene, eid, plan, distance=1.0, oracle=1, device="cuda" if eid == 7 else "npu")
    result = cc.compare(cc.load(canary), cc.load(base), keys, criteria="exp28-p")
    assert result["criteria"]["sr_delta_pt"]["value"] == -1.0 and result["all_pass"] and result["verdict"] == "pass"
    assert result["criteria"]["turn_reversal_share"]["value"] == 0.0  # no turn calls: no reversals
    mixed = cc.compare(cc.load(canary), cc.load(base), keys, criteria="exp28-p", require_device="npu")
    assert mixed["criteria"]["platform"]["value"] == ["cuda", "npu"] and not mixed["all_pass"]
    assert mixed["criteria"]["platform"]["canary"] == {"cuda": 2, "npu": 98}
    # no line is crossed, but the arms are no pair
    assert mixed["kill_lines_crossed"] == [] and mixed["verdict"] == "invalid"
    assert cc.main(["--canary-records", str(canary), "--base-records", str(base), "--criteria", "exp28-p",
                    "--bootstrap-draws", "0", "--output", str(tmp_path / "m.json")]) == 3
    assert capsys.readouterr().out.startswith("INVALID P matched=100")
    # one episode the candidate never finished: not a kill
    short = cc.compare(cc.load(canary), cc.load(base), keys | {("00bbb", 99)}, criteria="exp28-p")
    assert short["verdict"] == "incomplete" and not short["all_pass"] and short["kill_lines_crossed"] == []
    # nothing paired at all (a wrong base root): no verdict on the candidate, not a kill
    empty = cc.compare(cc.load(canary), {}, keys, criteria="exp28-p", require_device="npu")
    assert empty["matched_episodes"] == 0 and empty["verdict"] == "incomplete" and empty["kill_lines_crossed"] == []


def test_p_reads_the_executed_ending_not_the_last_answer(tmp_path):
    """SR is Habitat's (the record's success, a STOP called within 3 m); a step-budget STOP is no STOP."""
    base, canary = tmp_path / "base", tmp_path / "canary"
    keys = {("00aaa", e) for e in range(4)}
    plan = [("pixel", 6.0), ("stop", 1.0)]
    for _, eid in keys:
        _rec(base, "00aaa", eid, plan, distance=1.0, oracle=1, device="npu")
    _rec(canary, "00aaa", 0, plan, distance=1.0, oracle=1, device="npu")
    # the server answered a pixel with no actions; the client stopped at 1 m: Habitat's success
    _rec(canary, "00aaa", 1, [("pixel", 6.0), ("pixel", 1.0)], distance=1.0, oracle=1, device="npu", ending="empty")
    # the client's step-budget STOP at 2 m: Habitat says success, the standard protocol out of steps
    _rec(canary, "00aaa", 2, [("left", 4.0), ("pixel", 2.0)], distance=2.0, oracle=1, steps=500, device="npu",
         ending="budget")
    # the client's STOP after an empty answer 5 m out, never inside 3 m: stopped outside the circle
    _rec(canary, "00aaa", 3, [("pixel", 6.0), ("pixel", 5.0)], distance=5.0, oracle=0, device="npu", ending="empty")
    summary = cc.load(canary)
    assert summary[("00aaa", 1)]["success_recorded"] and summary[("00aaa", 2)]["success_recorded"]
    result = cc.compare(summary, cc.load(base), keys, criteria="exp28-p")
    c = result["canary"]
    assert c["sr_standard_pct"] == 50.0 and c["sr_pct"] == 25.0 and c["sr_recorded_pct"] == 75.0
    assert (c["out_of_steps_pct"], c["out_of_steps"]) == (25.0, 3)  # C-A's answer-based count says 3
    assert c["stopped_never_within_3m_pct"] == 25.0
    assert (c["step_budget_stops"], c["client_stops"], c["ending_disagrees_with_answer"]) == (1, 2, 2)
    assert c["success_disagrees_with_answer"] == 1 and c["os_from_distances"] == 0
    crit = result["criteria"]
    assert crit["sr_delta_pt"]["value"] == -50.0
    assert crit["out_of_steps_delta_pt"]["value"] == 25.0 and crit["stop_never_within_3m_delta_pt"]["value"] == 25.0
    assert result["report_only"]["paired_flips"] == {"base_success_canary_fail": 2, "base_fail_canary_success": 0}

    # records whose steps carry no action cannot say how they ended: P refuses them
    bare = tmp_path / "bare"
    for _, eid in keys:
        _rec(bare, "00aaa", eid, plan, distance=1.0, oracle=1, device="npu", step_actions=False)
    with pytest.raises(ValueError, match="steps without actions"):
        cc.compare(cc.load(bare), cc.load(base), keys, criteria="exp28-p")
    # C-A never read the steps' actions and still reads such records
    assert cc.compare(cc.load(bare), cc.load(base), keys)["matched_episodes"] == 4


def _parted_cohort(root, parts):
    """An EXP-28-style cohort: one manifest, shards tagged with their part."""
    root.mkdir()
    shards = []
    for index, (part, episodes) in enumerate(parts):
        name = f"shard_{index:03d}.json"
        (root / name).write_text(json.dumps({"part": part, "episodes": [
            {"scene_id": s, "episode_id": e, "trajectory_id": e} for s, e in episodes]}))
        shards.append({"index": index, "path": name, "part": part})
    (root / "manifest.json").write_text(json.dumps({"shards": shards, "parts": {}}))
    return root


def test_a_cohort_with_parts_needs_one_named(tmp_path, capsys):
    canary, base, _ = _p_records(tmp_path)
    heldout = sorted(cc.load(canary))
    cohort = _parted_cohort(tmp_path / "cohort", [("heldout", heldout[:10]), ("heldout", heldout[10:]),
                                                  ("train_pass1", [("00zzz", 1), ("00zzz", 2)])])
    assert cc.cohort_parts(cohort) == ["heldout", "train_pass1"]
    assert cc.cohort_episodes(cohort, "heldout") == set(heldout)
    assert len(cc.cohort_episodes(cohort)) == 22
    with pytest.raises(ValueError, match="no shard of part 'train_pass2'"):
        cc.cohort_episodes(cohort, "train_pass2")
    argv = ["--canary-records", str(canary), "--base-records", str(base), "--criteria", "exp28-p",
            "--bootstrap-draws", "0", "--cohort", str(cohort)]
    with pytest.raises(SystemExit):  # the whole cohort would leave P incomplete on the training routes
        cc.main(argv)
    assert "--cohort-part" in capsys.readouterr().err
    assert cc.main(argv + ["--cohort-part", "heldout", "--output", str(tmp_path / "p.json")]) == 1
    written = json.loads((tmp_path / "p.json").read_text())
    assert written["expected_episodes"] == 20 and written["complete"] and written["verdict"] == "kill"
    assert written["inputs"]["cohort_part"] == "heldout"
    capsys.readouterr()


def test_os_falls_back_to_the_recorded_distances(tmp_path):
    root = tmp_path / "r"
    inside = _rec(root, "s", 1, [("pixel", 4.0), ("pixel", 2.9), ("stop", 3.5)], distance=3.5)
    outside = _rec(root, "s", 2, [("pixel", 4.0), ("stop", 3.2)], distance=3.2)
    assert cc.episode_summary(EpisodeRecord(inside))["oracle_success"]
    assert not cc.episode_summary(EpisodeRecord(outside))["oracle_success"]


def test_record_device_matches_the_dataset_module(tmp_path):
    pytest.importorskip("torch")
    from src.data.s2_sft_dataset import record_device

    for device in (None, "npu", "cuda"):
        path = _rec(tmp_path / str(device), "s", 1, [("stop", 1.0)], distance=1.0, device=device)
        record = EpisodeRecord(path)
        assert cc.record_device(record) == record_device(record)


# ------------------------------------------------------------------------------ per-call identity
def _identity_records(tmp_path, device="npu"):
    rerun, reference = tmp_path / "rerun", tmp_path / "reference"
    plan = [("pixel", 6.0), ("left", 4.0), ("pixel", 2.5), ("stop", 1.0)]
    for eid in range(10):
        _rec(reference, "00aaa", eid, plan, distance=1.0, oracle=1, device="npu")
    for eid in range(10):
        if eid == 3:  # call 2 answers another pixel; the episode ends the same
            _rec(rerun, "00aaa", eid, plan, distance=1.0, oracle=1, device=device,
                 tweak=lambda row: row.update(llm_output="250 320", pixel_goal=[250, 320]) if row["call_index"] == 2 else None)
        elif eid == 5:  # same calls, other prompt bytes for call 1 (reported, not judged)
            _rec(rerun, "00aaa", eid, plan, distance=1.0, oracle=1, device=device,
                 tweak=lambda row: row.update(lookdown_sha256="other") if row["call_index"] == 1 else None)
        else:
            _rec(rerun, "00aaa", eid, plan, distance=1.0, oracle=1, device=device)
    return rerun, reference


def test_per_call_identity_counts_episodes_reproduced_call_by_call(tmp_path, capsys):
    rerun, reference = _identity_records(tmp_path)
    result = cc.identity_compare(rerun, reference, None, 9)
    assert (result["scope_episodes"], result["identical_episodes"], result["pass"]) == (10, 9, True)
    assert (result["calls"], result["identical_calls"], result["identical_input_calls"]) == (40, 39, 39)
    assert result["devices"] == ["npu"] and result["same_platform"] and result["complete"]
    bad = next(e for e in result["episodes"] if not e["identical"])
    assert bad["episode"] == "00aaa_000003" and bad["first_divergent_call"] == 2
    assert bad["first_divergent_fields"] == ["llm_output", "pixel_goal"] and bad["outcome_differs"] == []
    assert next(e for e in result["episodes"] if e["episode"] == "00aaa_000005")["identical"]
    # all ten required by default
    assert not cc.identity_compare(rerun, reference, None)["pass"]
    # an episode the rerun never finished is not reproduced
    scope = {("00aaa", e) for e in range(11)}
    short = cc.identity_compare(rerun, reference, scope, 9)
    assert not short["complete"] and short["missing_in_rerun"] == ["00aaa_000010"] and short["pass"]
    assert not cc.identity_compare(rerun, reference, scope, 10)["pass"]

    assert cc.main(["--rerun-records", str(rerun), "--reference-records", str(reference),
                    "--min-identical-episodes", "9", "--output", str(tmp_path / "id.json")]) == 0
    assert capsys.readouterr().out.startswith("PASS per-call identity 9/10 episodes (need 9) calls 39/40")
    assert json.loads((tmp_path / "id.json").read_text())["schema"] == cc.IDENTITY_SCHEMA


def test_per_call_identity_sees_outcomes_extra_calls_and_platforms(tmp_path):
    rerun, reference = tmp_path / "rerun", tmp_path / "reference"
    plan = [("pixel", 6.0), ("stop", 1.0)]
    _rec(reference, "s", 1, plan, distance=1.0, device="npu")
    _rec(rerun, "s", 1, plan, distance=1.25, device="npu")  # same calls, another ending
    _rec(reference, "s", 2, plan, distance=1.0, device="npu")
    _rec(rerun, "s", 2, [("pixel", 6.0), ("left", 2.0), ("stop", 1.0)], distance=1.0, device="npu")
    result = cc.identity_compare(rerun, reference, None, 0)
    one, two = result["episodes"]
    assert not one["identical"] and one["outcome_differs"] == ["distance_to_goal"] and one["first_divergent_call"] is None
    assert not two["identical"] and two["first_divergent_call"] == 1 and (two["calls"], two["reference_calls"]) == (3, 2)
    assert two["first_divergent_fields"] == ["kind", "native_first_output", "llm_output", "actions", "terminal"]
    assert result["pass"]  # min 0: only the platform and the scope are judged

    other = tmp_path / "other"
    _rec(other, "s", 1, plan, distance=1.0)  # the 4090's own server
    cross = cc.identity_compare(other, reference, None, 0)
    assert cross["devices"] == ["cuda", "npu"] and not cross["same_platform"] and not cross["pass"]


def test_cli_modes_do_not_mix(tmp_path):
    with pytest.raises(SystemExit):
        cc.main(["--rerun-records", str(tmp_path)])
    with pytest.raises(SystemExit):
        cc.main(["--rerun-records", str(tmp_path), "--reference-records", str(tmp_path), "--canary-records", "x"])
    with pytest.raises(SystemExit):
        cc.main(["--canary-records", str(tmp_path)])
    with pytest.raises(SystemExit):
        cc.main(["--canary-records", str(tmp_path), "--base-records", str(tmp_path), "--min-identical-episodes", "9"])
    with pytest.raises(SystemExit):
        cc.main(["--rerun-records", str(tmp_path), "--reference-records", str(tmp_path), "--min-identical-episodes", "0"])
    with pytest.raises(SystemExit):
        cc.main(["--canary-records", str(tmp_path), "--base-records", str(tmp_path), "--cohort-part", "heldout"])
