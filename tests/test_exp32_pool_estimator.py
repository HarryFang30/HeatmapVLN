"""EXP-32 J: the val_unseen pool reference and the estimator's pre-registered decisions (scripts/exp32/pool_estimator.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.exp32 import pool_estimator as pe
from scripts.tools import paired_closed_loop_bootstrap as pb
from scripts.tools import paired_progress_monitor

N = pe.EPISODES
# EXP-20's real success counts out of 1,839 (62.70 / 61.45 / 60.96 / 62.97; 60.74 / 61.34 / 61.50 / 61.23).
COUNTS = {("a0", 42): 1153, ("a1", 42): 1130, ("a2", 42): 1121, ("a3", 42): 1158,
          ("a0", 1337): 1117, ("a1", 1337): 1128, ("a2", 1337): 1131, ("a3", 1337): 1126}


def _rows(successes, seed, *, offset=0, n=N, budget_stops=0):
    """``successes`` episodes succeed (a window from ``offset``, so the arms differ per episode); with
    ``budget_stops`` that many failures end with the step budget's STOP inside 3 m (switched-on successes)."""
    rows, stops = [], 0
    for i in range(n):
        ok = (i - offset) % N < successes
        row = {"scene_id": f"scene{i % 11:02d}", "episode_id": i + 1, "success": float(ok), "spl": 0.8 * ok,
               "os": float(ok or i % 3 == 0), "ne": 1.5 if ok else 6.0, "steps": 120, "rpc_protocol_seed": seed}
        if budget_stops:
            row["stop_at_step_budget"] = True
            if not ok and stops < budget_stops:
                row.update(success=1.0, spl=0.5, ne=2.5, steps=500, step_budget_stop=True)
                stops += 1
        rows.append(row)
    return rows


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def pool(tmp_path):
    template = str(tmp_path / "eval_runs" / "exp20_{arm}_seed{seed}_4090" / "merged" / "progress.jsonl")
    for (arm, seed), count in COUNTS.items():
        _write(Path(template.format(arm=arm, seed=seed)), _rows(count, seed, offset=37 * pe.ARMS.index(arm) + seed % 5))
    return template


def _candidate_args(tmp_path, counts, budget_stops=0, n=N):
    args = []
    for seed, count in zip(pe.SEED_ORDER, counts):
        run = tmp_path / f"cand_seed{seed}"
        _write(run / "merged" / "progress.jsonl", _rows(count, seed, offset=11, budget_stops=budget_stops,
                                                       n=n if seed == 42 else N))
        args += ["--candidate", f"{seed}={run}"]
    return args


def _estimate(tmp_path, pool, counts, *extra, budget_stops=0):
    out = tmp_path / "estimate.json"
    code = pe.main(["estimate", "--ref-template", pool, *_candidate_args(tmp_path, counts, budget_stops),
                    "--bootstrap-draws", "50", "--out", str(out), *extra])
    return code, json.loads(out.read_text())


# ---- the reference pool -------------------------------------------------------------------------------------------

def test_pool_reproduces_the_ledger(pool):
    report = pe.check_pool(pe.load_pool(pool))
    assert report["seed_means_pt"] == {"42": round(100 * 4562 / (4 * N), 4), "1337": round(100 * 4502 / (4 * N), 4)}
    assert report["mean_pt"] == pytest.approx(61.61, abs=0.01) and report["sd_pt"] == pytest.approx(0.80, abs=0.01)
    assert report["runs_pt"]["a3_seed42"] == pytest.approx(62.97, abs=0.01)


def test_pool_off_the_ledger_is_refused(pool):
    # one more success in one arm moves its SR by 0.054 pt: more than the 0.01 the ledger allows
    _write(Path(pool.format(arm="a1", seed=42)), _rows(COUNTS[("a1", 42)] + 1, 42))
    with pytest.raises(SystemExit, match="停查.*run a1 seed 42"):
        pe.main(["pool-ref", "--seed", "42", "--ref-template", pool, "--out", "/dev/null"])


def test_pool_run_of_1838_rows_is_refused(pool, tmp_path):
    _write(Path(pool.format(arm="a2", seed=1337)), _rows(COUNTS[("a2", 1337)], 1337, n=N - 1))
    with pytest.raises(SystemExit, match=f"{N - 1} episodes"):
        pe.main(["pool-ref", "--seed", "42", "--ref-template", pool, "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit, match=f"{N - 1} episodes"):
        _estimate(tmp_path, pool, [1158])


def test_pool_run_of_another_seed_is_refused(pool, tmp_path):
    _write(Path(pool.format(arm="a3", seed=1337)), _rows(COUNTS[("a3", 1337)], 42))
    with pytest.raises(SystemExit, match="protocol seed"):
        pe.main(["pool-ref", "--seed", "1337", "--ref-template", pool, "--out", str(tmp_path / "x.jsonl")])


def test_pool_ref_is_the_four_arm_mean_and_the_monitor_reads_it(pool, tmp_path):
    out = tmp_path / "pool_ref_seed42.jsonl"
    assert pe.main(["pool-ref", "--seed", "42", "--ref-template", pool, "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == N and {r["rpc_protocol_seed"] for r in rows} == {42}
    assert {r["success"] for r in rows} == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert 100 * sum(r["success"] for r in rows) / N == pytest.approx(100 * 4562 / (4 * N))
    # the unchanged monitor: 600 episodes, all failed -> the -3.0 rule fires; A0's own first 600 do not
    _write(tmp_path / "w" / "p0.jsonl", _rows(0, 42, n=600))
    report_path = tmp_path / "m.json"
    argv = ["--candidate", str(tmp_path / "w" / "*.jsonl"), "--reference", str(out), "--min-episodes", "600",
            "--abort-delta", "-3.0", "--out", str(report_path)]
    assert paired_progress_monitor.main(argv) == 3
    report = json.loads(report_path.read_text())
    assert report["sr"]["n"] == 600 and report["abort"]
    a0_first = [json.loads(line) for line in Path(pool.format(arm="a0", seed=42)).read_text().splitlines()][:600]
    _write(tmp_path / "w" / "p0.jsonl", a0_first)
    assert paired_progress_monitor.main(argv) == 0


# ---- the decisions at the ledger's lines ----------------------------------------------------------------------------

def test_standard_error_floor():
    assert pe.standard_error(2, None) == pytest.approx(0.6986, abs=1e-4)      # ledger: SE_2 >= 0.699
    assert pe.standard_error(4, 0.3) == pytest.approx(0.5728, abs=1e-4)       # ledger: SE_4 >= 0.573
    assert pe.standard_error(2, 1.0) == pytest.approx((1.0 / 2 + 0.41 ** 2) ** 0.5)


@pytest.mark.parametrize("k, e_hat, s_c, paired, verdict", [
    (1, -0.50, None, None, "not_supported_futility"),
    (1, -0.49, None, None, "continue"),
    (2, 0.29, 0.5, 1.0, "not_detected"),
    (2, 0.30, 0.5, 1.0, "continue"),
    (2, 0.0, 0.5, 1.0, "not_supported"),
    (2, -1.0, 0.5, -1.0, "not_supported"),
    (2, 2.10, 0.5, 0.1, "supported_early"),      # 3 x SE_2 = 2.096 at the floor
    (2, 2.09, 0.5, 0.1, "continue"),
    (2, 2.10, 0.5, 0.0, "continue"),             # the pooled paired dSR vs A0 must be > 0
    (2, 2.10, 1.0, 0.1, "continue"),             # s_c > 0.80: 3 x SE_2 = 2.452
    (2, 2.46, 1.0, 0.1, "supported_early"),
    (3, 5.0, 0.5, 1.0, "continue"),
    (4, 1.13, 0.5, 0.1, "supported"),            # 1.96 x SE_4 = 1.1227 at the floor
    (4, 1.12, 0.5, 0.1, "not_detected"),
    (4, 1.13, 0.5, -0.1, "not_detected"),
    (4, 1.13, 0.9, 0.1, "not_detected"),         # s_c > 0.80: 1.96 x SE_4 = 1.193
    (4, 1.20, 0.9, 0.1, "supported"),
    (4, 0.0, 0.5, 0.1, "refuted"),
    (4, 0.01, 0.5, 0.1, "not_detected"),
])
def test_decide_at_the_lines(k, e_hat, s_c, paired, verdict):
    stage = pe.decide(k, e_hat, s_c, paired)
    assert stage["verdict"] == verdict
    assert stage["terminal"] == (verdict != "continue")


def test_decide_refuses_a_fifth_seed():
    with pytest.raises(ValueError):
        pe.decide(5, 1.0, 0.5, 0.1)


# ---- the decisions from 1,839-row files ------------------------------------------------------------------------------

@pytest.mark.parametrize("counts, verdict, code, e_hat", [
    ([1154, 1154, 1153, 1153], "not_detected", 3, 1.1143),   # Ê_4 just under 1.96 x SE_4 = 1.1227
    ([1154, 1154, 1154, 1153], "supported", 3, 1.1279),
    ([1139, 1138], "not_detected", 3, 0.2986),               # Ê_2 just under 0.3
    ([1139, 1139], "continue", 0, 0.3258),
    ([1172, 1171], "continue", 0, 2.0931),                   # just under the early line 2.096
    ([1172, 1172], "supported_early", 3, 2.1203),
    ([1200, 1144], "continue", 0, 2.1203),                   # same Ê_2, s_c = 2.15 > 0.80 widens the line
    ([1131], "not_supported_futility", 3, -0.5192),
    ([1132], "continue", 0, -0.4648),
    ([1150, 1130, 1080, 1080], "refuted", 3, -1.2511),
])
def test_estimate_on_files(pool, tmp_path, counts, verdict, code, e_hat):
    got, report = _estimate(tmp_path, pool, counts)
    assert got == code
    decision = report["decision"]
    assert decision["verdict"] == verdict and decision["k"] == len(counts)
    assert decision["e_hat"] == pytest.approx(e_hat, abs=1e-4)
    assert [st["k"] for st in report["stages"]] == list(range(1, len(counts) + 1))


def test_a_stop_at_seed_42_is_the_decision_whatever_follows(pool, tmp_path):
    code, report = _estimate(tmp_path, pool, [1131, 1300])
    assert code == 3
    assert report["decision"]["verdict"] == "not_supported_futility" and report["decision"]["decided_at_k"] == 1
    assert report["decision"]["seeds_past_decision"] == [1337]
    assert report["stages"][1]["verdict"] == "continue"     # reported, not judged


def test_a3_as_a_stand_in_candidate_continues(pool, tmp_path):
    code, report = _estimate(tmp_path, pool, [COUNTS[("a3", 42)], COUNTS[("a3", 1337)]])
    assert code == 0
    d = report["decision"]
    assert d["verdict"] == "continue" and d["next_seeds"] == [2024, 3407]
    assert round(d["e_hat"], 2) == 0.49 and d["s_c"] == pytest.approx(1.2304, abs=1e-3)
    assert d["se"] == pytest.approx(0.9616, abs=1e-3)


def test_paired_comparison_is_the_bootstrap_cli_and_budget_stops_are_failures(pool, tmp_path):
    code, report = _estimate(tmp_path, pool, [1160, 1120], budget_stops=5)
    assert code == 0
    cand = report["candidate"]
    assert cand["42"]["sr_pt"] == round(100 * 1160 / N, 4)
    assert cand["42"]["sr_switched_on_pt"] == round(100 * 1165 / N, 4) and cand["42"]["step_budget_stops"] == 5
    cli = tmp_path / "cli.json"
    argv = []
    for seed in pe.POOL_SEEDS:
        argv += ["--treatment", str(tmp_path / f"cand_seed{seed}" / "merged" / "progress.jsonl"),
                 "--control", pool.format(arm="a0", seed=seed)]
    pb.main(argv + ["--standard-protocol", "--bootstrap-draws", "50", "--output-json", str(cli)])
    expected = json.loads(cli.read_text())["strata"]["all"]
    mine = report["report_only"]["paired_vs_a0"]["seeds42+1337"]
    for metric in pb.METRICS:
        assert mine[metric] == pe._points(expected[metric], metric)
    assert mine["n"] == 2 * N
    assert report["decision"]["paired_delta_sr"] == pytest.approx(100 * (1160 + 1120 - 1153 - 1117) / (2 * N), abs=1e-4)
    assert report["report_only"]["vs_a0_two_seed_mean_pt"] == pytest.approx(report["decision"]["paired_delta_sr"], abs=1e-3)
    boot = report["report_only"]["pool_episode_bootstrap"]
    assert boot["delta_pt"] == pytest.approx(100 * (1160 + 1120) / (2 * N) - 100 * (4562 + 4502) / (8 * N), abs=1e-3)


def test_candidate_refusals(pool, tmp_path):
    with pytest.raises(SystemExit, match=f"{N - 1} episodes"):
        pe.main(["estimate", "--ref-template", pool, *_candidate_args(tmp_path, [1158, 1126], n=N - 1)])
    with pytest.raises(SystemExit, match="order"):
        pe.main(["estimate", "--ref-template", pool, "--candidate", f"42={tmp_path / 'cand_seed42'}",
                 "--candidate", f"2024={tmp_path / 'cand_seed1337'}"])
    _write(tmp_path / "other" / "merged" / "progress.jsonl", _rows(1150, 1337))
    with pytest.raises(SystemExit, match="protocol seed"):
        pe.main(["estimate", "--ref-template", pool, "--candidate", f"42={tmp_path / 'other'}"])
    shifted = [dict(r, episode_id=r["episode_id"] + 5000) for r in _rows(1150, 42)]
    _write(tmp_path / "shifted" / "merged" / "progress.jsonl", shifted)
    with pytest.raises(SystemExit, match="other episodes"):
        pe.main(["estimate", "--ref-template", pool, "--candidate", f"42={tmp_path / 'shifted'}"])
