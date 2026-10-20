"""EXP-32's verdict over the offline gates: ``hm3d_route_gate.py verdict --rules exp32`` (ledger EXP-32 第二部分 F).

Gated: O1, O2, O3, O4_anchor, O4_stop, L (on O5's states, the share whose answer under the donor instruction is not
``same_answer`` as under the episode's own: candidate >= base - 5.0 pt) and R (``scripts/exp32/rft_gate.py``'s
gate: R, or M when R is report-only).  O4's correction item and O5 are report-only.  The retry table is checked
over every combination of group results.  ``--rules exp28`` (the default) must write what it wrote before
EXP-32: ``EXP28_GOLDEN`` holds digests of its output on the verdict fixtures of ``test_hm3d_route_gate``, taken
with the code before the patch (``hm3d_route_gate.py`` unchanged since 1ded1cb).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path

import pytest

from scripts.exp28 import hm3d_route_gate as g
from scripts.exp32 import rft_gate as rg
from tests.test_hm3d_route_gate import _agreement, _o3, _o4, _rows, _teacher_forced

A, B = "a" * 64, "b" * 64
HARM = ("O1", "O2", "O3", "O4_anchor", "O4_stop")
PIXEL = ["↓", "300 300"]


# --------------------------------------------------------------------------- #
# --rules exp28: byte for byte what it was
# --------------------------------------------------------------------------- #
EXP28_CASES = ("pass", "o2_drop", "o2_drop_attempt2", "correction_only", "o5_degenerate_thin", "o4_invalid",
               "smoke_subset", "o3_unjoined", "o3_foreign", "missing_rows")


def _exp28_argv(root: Path, case: str) -> list[str]:
    """One verdict input set per case, written under ``root``: every next step, the notes, invalid inputs."""
    hits = {"o1": 86, "o1-base": 86, "o2": 98, "o2-base": 98}
    o3, o4, attempt = _o3(), _o4(), "1"
    if case.startswith("o2_drop"):
        hits["o2"] = 96
        attempt = "2" if case.endswith("attempt2") else "1"
    elif case == "correction_only":
        o4 = _o4(correction_agree_gain=True)
    elif case == "o5_degenerate_thin":
        o4 = {**_o4(correction_agree_gain=True, o5=True, degenerate=True),
              "selection": {"settings": {"max_episodes": 0}, "scenes_without_records": ["X", "Y"]}}
    elif case == "o4_invalid":
        o4 = {"valid": False, "invalid": "GateInvalid: 3 of 9 states were not decoded"}
    elif case == "smoke_subset":
        o4 = {**_o4(), "selection": {"settings": {"max_episodes": 5}}}
    elif case == "o3_unjoined":
        o3 = _o3(rows=2000, joined=1990)
    elif case == "o3_foreign":
        o3 = {"G_A": {"criteria": {}, "all_pass": True}}
    case_dir = root / case
    paths = {
        "o1": _teacher_forced(case_dir / "o1" / "scalevln.json", hits["o1"]),
        "o1-base": _teacher_forced(case_dir / "base" / "o1_scalevln.json", hits["o1-base"]),
        "o2": _teacher_forced(case_dir / "o2" / "r2r.json", hits["o2"]),
        "o2-base": _teacher_forced(case_dir / "base" / "o2_r2r.json", hits["o2-base"]),
    }
    if case == "missing_rows":
        paths["o2-base"].with_suffix(".calls.jsonl").unlink()
    (case_dir / "o3.json").write_text(json.dumps(o3))
    (case_dir / "o4.json").write_text(json.dumps(o4))
    argv = ["verdict", "--o3", str(case_dir / "o3.json"), "--o4", str(case_dir / "o4.json"),
            "--output", str(case_dir / "verdict.json"), "--attempt", attempt]
    for name, path in paths.items():
        argv += [f"--{name}", str(path)]
    return argv


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = g.main(argv)
    return code, out.getvalue(), err.getvalue()


def exp28_digests(root: Path, rules: str | None = None) -> dict[str, str]:
    """Per case, sha256 of (exit code, stdout, stderr, verdict.json) with ``root`` written as ``<root>``."""
    digests = {}
    base = root / (rules or "default")
    for case in EXP28_CASES:
        argv = _exp28_argv(base, case)
        if rules is not None:
            argv += ["--rules", rules]
        code, out, err = _run(argv)
        written = Path(argv[argv.index("--output") + 1])
        text = written.read_text(encoding="utf-8") if written.is_file() else "<none>"
        blob = "\x00".join([str(code), out, err, text])
        for path in (str(base.resolve()), str(base)):
            blob = blob.replace(path, "<root>")
        digests[case] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return digests


# exp28_digests() with the verdict code before this patch (4090 container, 2026-10-20); the cases' printed lines were
# PASS val / FAIL scale_0.5 / FAIL kill / FAIL retrain_2_epochs / FAIL kill / INVALID x4 / INVALID (no json) x2.
EXP28_GOLDEN = {
    "pass": "853919fbe09cb871a4fe121f26b717556e747e3c5fe5ccc6e4e5bb414c52169e",
    "o2_drop": "3cc8844f72bbf569bb8c6021488e4b93f900247c8522981f8ea76776343dad32",
    "o2_drop_attempt2": "12917504d6333338d7b2e3b9ff6033dc787768f9c9cdee321c4cc3c175f1173d",
    "correction_only": "71411467822e465775f8801f498579f676ddb36e84b723a82944a6f13ad8f49d",
    "o5_degenerate_thin": "3c5699adb2ad5df83f37c6eb9da6ff567e3e6b7fca550805beca7ba7c4c3429c",
    "o4_invalid": "9f35a767446082a274a2d9cd8149c3199bdc6036a4ce567d831499c5fdef1b49",
    "smoke_subset": "209df5f8c28141dbc96fb638d0efc421d964ee31c28caf609e814acd33b07135",
    "o3_unjoined": "f774209dbe842a234cb9517f889d6d7eedbe5a9c00808f73deb673d97c9db599",
    "o3_foreign": "aeea5a03e8b80e906a3b79463f2bd47cca4d124a389442ae6f949fd603f668cb",
    "missing_rows": "042d7a4d317475401a804fa030805dbaec2006234df921c061d0a5c61e326036",
}


def test_rules_exp28_writes_what_the_verdict_wrote_before_exp32(tmp_path):
    default = exp28_digests(tmp_path)
    assert default == EXP28_GOLDEN
    assert exp28_digests(tmp_path, "exp28") == default


def test_the_exp32_flags_belong_to_rules_exp32(tmp_path):
    argv = _exp28_argv(tmp_path, "pass")
    with pytest.raises(SystemExit):
        g.parse_args(argv + ["--rgate", "x.json"])
    with pytest.raises(SystemExit):
        g.parse_args(argv + ["--o4-calls", "x.jsonl"])
    with pytest.raises(SystemExit):
        g.parse_args(argv + ["--rules", "exp32"])
    args = g.parse_args(argv + ["--rules", "exp32", "--rgate", "r.json"])
    assert args.o4_calls == str(tmp_path / "pass" / "o4.calls.jsonl")
    assert g.parse_args(argv + ["--rules", "exp32", "--rgate", "r.json", "--o4-calls", "c.jsonl"]).o4_calls == "c.jsonl"


# --------------------------------------------------------------------------- #
# The retry table
# --------------------------------------------------------------------------- #
def _ledger_table(failed: set, attempt: int, learnt) -> str:
    """Ledger EXP-32 F's table, row by row."""
    if not failed:
        return "val"                       # 全过 -> 进 P
    if attempt == 2:
        return "kill"                      # 第二次仍有任一条不过
    if "L" in failed:
        return "kill"                      # L 不过
    if failed & set(HARM):
        return "kill" if "R" in failed else "scale_0.5"
    assert failed == {"R"}                 # 只有 R 不过
    if learnt is None:
        return "invalid"                   # the retry type cannot be read
    return "kill" if learnt else "retrain_2_epochs"


def test_next_step_follows_the_ledger_table_on_every_combination():
    seen = set()
    for passes in itertools.product((True, False), repeat=len(g.GROUPS_EXP32)):
        groups = dict(zip(g.GROUPS_EXP32, passes))
        failed = {name for name, ok in groups.items() if not ok}
        for attempt in (1, 2):
            for learnt in (True, False, None):
                step = g.next_step_exp32(groups, attempt, m_learnt=learnt)
                assert step == _ledger_table(failed, attempt, learnt), (failed, attempt, learnt)
                seen.add(step)
    assert seen == set(g.RETRY_RULES_EXP32)
    assert set(g.GROUPS_EXP32) == set(HARM) | {"L", "R"} and not set(g.REPORT_ONLY_EXP32) & set(g.GROUPS_EXP32)


@pytest.mark.parametrize("failed, attempt, learnt, step", [
    ((), 1, None, "val"),
    ((), 2, True, "val"),
    (("L",), 1, False, "kill"),
    (("L", "O1"), 1, False, "kill"),
    (("O1",), 1, None, "scale_0.5"),
    (("O4_stop", "O3"), 1, True, "scale_0.5"),
    (("O2", "R"), 1, False, "kill"),
    (("R",), 1, False, "retrain_2_epochs"),
    (("R",), 1, True, "kill"),
    (("R",), 1, None, "invalid"),
    (("R",), 2, False, "kill"),
    (("O4_anchor",), 2, None, "kill"),
])
def test_next_step_rows(failed, attempt, learnt, step):
    groups = {name: name not in failed for name in g.GROUPS_EXP32}
    assert g.next_step_exp32(groups, attempt, m_learnt=learnt) == step


# --------------------------------------------------------------------------- #
# L: instruction sensitivity on O5's states
# --------------------------------------------------------------------------- #
def _answer(row, arm, turns, shuffled=True):
    """Set an arm's answer (own or under the donor's words) without touching line B's ``agree``."""
    target = row["shuffled"] if shuffled else row
    target[arm] = {**target[arm], "turns": list(turns), "class": g.decision_class(turns[-1])}


def _sensitive(rows, arm, count, turns=("→",)):
    """The first ``count`` O5 states of ``arm`` answer ``turns`` under the donor's words (own answer: a pixel)."""
    for row in [r for r in rows if r.get("shuffled")][:count]:
        _answer(row, arm, turns)
    return rows


def _gate_rows(*, on_diff=0, off_diff=0, n=20, on_agree=11, off_agree=1, shuffled=(5, 2), **kw):
    """``score``-shaped rows that pass O4 and O5 (``test_hm3d_route_gate._rows``), with L's answers set."""
    rows = _rows(corrections=n, on_agree=on_agree, off_agree=off_agree, anchors=3, kept=3, shuffled=shuffled, **kw)
    return _sensitive(_sensitive(rows, "on", on_diff), "off", off_diff)


def test_l_line_is_exact_and_inclusive():
    on_line = g.language_gate(_gate_rows(on_diff=9, off_diff=10))
    assert (on_line["value"], on_line["threshold"], on_line["pass"]) == (0.45, 0.45, True)
    assert on_line["n"] == 20 and on_line["on_different"] == 9 and on_line["base_different"] == 10
    assert on_line["delta"] == -0.05 and on_line["base_rate"] == 0.5
    below = g.language_gate(_gate_rows(on_diff=8, off_diff=10))
    assert below["value"] == 0.4 and not below["pass"]
    # a candidate more sensitive than the base, or as insensitive as a base that ignores the words, passes
    assert g.language_gate(_gate_rows(on_diff=15, off_diff=3))["pass"]
    assert g.language_gate(_gate_rows())["pass"] and g.language_gate(_gate_rows())["value"] == 0.0
    # whole counts on a line floats would round: 60 states, base 33 -> line 30/60
    sixty = dict(n=60, on_agree=31, off_agree=1, shuffled=(10, 2), off_diff=33)
    assert g.language_gate(_gate_rows(on_diff=30, **sixty))["pass"]
    assert not g.language_gate(_gate_rows(on_diff=29, **sixty))["pass"]


def test_l_reads_same_answer_on_o5s_states_only():
    rows = _gate_rows(n=24, no_donor=4)                # 4 corrections without a donor stay out
    donors = [r for r in rows if r.get("shuffled")]
    _answer(donors[0], "on", ["↓", "350 250"])         # 50 px on both axes: the same answer
    _answer(donors[1], "on", ["↓", "351 300"])         # 51 px: different
    _answer(donors[2], "on", ["→"], shuffled=False)    # own: right, donor: left -- a turn the other way
    _answer(donors[2], "on", ["←←"])
    _answer(donors[3], "on", ["←"], shuffled=False)    # own: left, donor: further left -- the same answer
    _answer(donors[3], "on", ["←←←"])
    _answer(donors[4], "on", ["STOP"])                 # another class
    anchor = next(r for r in rows if r["label_kind"] == "anchor")
    anchor["shuffled"] = {"donor": "s_x", "on": {"turns": ["STOP"]}, "off": {"turns": ["STOP"]}}  # not O5's
    on = g.instruction_sensitivity(rows, "on")
    assert on["n"] == 20 and on["different"] == 3
    assert on["different_by"] == {"class": 2, "pixel_over_50px": 1}
    lang = g.language_gate(rows)
    assert lang["corrections_without_donor"] == 4 and lang["n"] == 20 and lang["on_different"] == 3
    # an O5 state without its shuffled answers: L is not measured (the verdict reads it as invalid)
    donors[5]["shuffled"] = None
    partial = g.language_gate(rows)
    assert not partial["pass"] and partial["corrections_not_decoded_shuffled"] == 1
    assert not partial["measured_every_correction_with_donor"]
    assert not g.language_gate([])["pass"] and g.language_gate([])["n"] == 0


# --------------------------------------------------------------------------- #
# verdict_exp32
# --------------------------------------------------------------------------- #
def _o4_report(rows, adapter=A):
    """``score``'s report on ``rows`` (valid when O4 and O5 can be read)."""
    summary = g.summarize(rows, shuffle_requested=True)
    problems = g.measurement_problems(summary)
    report = {"schema": g.SCHEMA, "valid": not problems, **summary,
              "join": {"adapter_sha256": adapter, "stats": {"rows": len(rows)}},
              "selection": {"settings": {"max_episodes": 0}, "scenes_without_records": []}}
    if problems:
        report["invalid"] = "GateInvalid: " + "; ".join(problems)
    return json.loads(json.dumps(report))


def _r_row(group, episode, on_agree, off_agree, source="s42"):
    return {"group": group, "source": source, "record_key": f"S_{episode:06d}", "decision": "pixel",
            "on": {"turns": ["x"], "class": "pixel"}, "off": {"turns": ["y"], "class": "pixel"},
            "on_agree": on_agree, "off_agree": off_agree}


def _rgate(*, r=True, uses="R", learnt=False, m=True, adapter=A):
    """``rft_gate.py score``'s report, built by its own ``summarize``.  ``uses``: "R" (150 divergent states,
    D_R 40/150 or 5/150), "M" (10 divergent states with seed 2024: report-only) or None (seed 2024 missing)."""
    if uses == "R":
        rows = [_r_row("R", e, e < (40 if r else 5) and i == 0, False) for e in range(75) for i in range(2)]
    else:
        rows = [_r_row("R", e, e == 0, False) for e in range(10)]
    rows += [_r_row("R_fail", 100 + e, e == 0, False) for e in range(10)]  # report-only
    if m:
        rows += [_r_row("M", i, i < (5 if learnt else 4), i < 2, source="p1") for i in range(20)]
    summary = rg.summarize(rows, r_seeds={42, 1337, 2024} if uses == "M" else {42, 1337})
    problems = rg.measurement_problems(summary)
    report = {"schema": rg.SCHEMA, "valid": not problems, "adapter_sha256": adapter, **summary}
    if problems:
        report["invalid"] = "GateInvalid: " + "; ".join(problems)
    return json.loads(json.dumps(report))


OK1 = g.compare_agreement(_agreement(3500, 3007), _agreement(3500, 3007), reference=g.O1_REFERENCE)
OK2 = g.compare_agreement(_agreement(600, 585), _agreement(600, 585), reference=g.O2_REFERENCE)
DROP1 = g.compare_agreement(_agreement(3500, 2970), _agreement(3500, 3007), reference=g.O1_REFERENCE)


def _verdict(*, rows=None, o4=None, rgate=None, o1=OK1, o3=None, attempt=1):
    rows = _gate_rows() if rows is None else rows
    return g.verdict_exp32(o1=o1, o2=OK2, o3=o3 or _o3(rows=3000), o4=o4 or _o4_report(rows), o4_rows=rows,
                           rgate=rgate or _rgate(), attempt=attempt)


def test_verdict_exp32_gates_l_and_r_and_reports_o4_correction_and_o5():
    result = _verdict()
    assert result["schema"] == g.VERDICT_SCHEMA_EXP32 and list(result["groups"]) == list(g.GROUPS_EXP32)
    assert result["all_pass"] and result["measured"] and result["next"] == "val" and result["notes"] == []
    assert result["R"]["gate"]["uses"] == "R" and result["adapter_sha256"] == {"O4": A, "R": A}
    # O4's correction item and O5 fail: report-only, the candidate still goes on
    weak = _gate_rows(on_agree=3, off_agree=1, shuffled=(6, 2))
    result = _verdict(rows=weak)
    assert result["report_only_groups"] == {"O4_correction": False, "O5": False}
    assert result["next"] == "val" and result["all_pass"]
    # L fails: kill, whatever else passes
    result = _verdict(rows=_gate_rows(on_diff=2, off_diff=10))
    assert not result["groups"]["L"] and result["next"] == "kill" and result["measured"]
    # a harm gate fails: scale 0.5 if R passed, kill if not; the retry that fails again: kill
    assert _verdict(o1=DROP1)["next"] == "scale_0.5"
    assert _verdict(o1=DROP1, rgate=_rgate(r=False))["next"] == "kill"
    assert _verdict(o1=DROP1, attempt=2)["next"] == "kill"
    assert _verdict(o3=_o3(False, rows=3000))["next"] == "scale_0.5"


def test_r_alone_failing_reads_m_for_the_retry():
    assert _verdict(rgate=_rgate(r=False, learnt=False))["next"] == "retrain_2_epochs"
    assert _verdict(rgate=_rgate(r=False, learnt=True))["next"] == "kill"
    unread = _verdict(rgate=_rgate(r=False, m=False))
    assert unread["next"] == "invalid" and not unread["measured"] and not unread["all_pass"]
    assert any("M was not measured" in n for n in unread["notes"])
    # M unmeasured does not matter when R passes
    assert _verdict(rgate=_rgate(m=False))["next"] == "val"
    # R report-only: M stands in for it
    via_m = _verdict(rgate=_rgate(uses="M", learnt=True))
    assert via_m["groups"]["R"] and via_m["next"] == "val" and any("report-only" in n for n in via_m["notes"])
    assert _verdict(rgate=_rgate(uses="M", learnt=False))["next"] == "retrain_2_epochs"


def test_what_cannot_be_measured_is_invalid_not_a_fail():
    waiting = _verdict(rgate=_rgate(uses=None))  # under 150 divergent states without B3's seed-2024 pass
    assert waiting["next"] == "invalid" and not waiting["measured"]
    assert any(n.startswith("R/M measurement invalid (rft_gate.json): GateInvalid: R has 10 divergent states") and
               "seed-2024" in n for n in waiting["notes"])
    lone = _rgate(uses="M", m=False)              # report-only R and no M
    assert not lone["valid"] and _verdict(rgate=lone)["next"] == "invalid"
    other = _verdict(rgate=_rgate(adapter=B))     # R decoded on another adapter (e.g. not the scaled copy)
    assert other["next"] == "invalid" and any("decoded with adapter" in n for n in other["notes"])
    # O4 invalid (here: one O5 state without its shuffled answers) leaves L unreadable too
    rows = _gate_rows()
    next(r for r in rows if r.get("shuffled"))["shuffled"] = None
    broken = _verdict(rows=rows)
    assert broken["next"] == "invalid" and any("O4/O5 measurement invalid" in n for n in broken["notes"])
    # the rows file is not the one O4 was scored from
    rows, foreign = _gate_rows(), _gate_rows(shuffled=(7, 2))
    stale = g.verdict_exp32(o1=OK1, o2=OK2, o3=_o3(rows=3000), o4=_o4_report(rows), o4_rows=foreign, rgate=_rgate(),
                            attempt=1)
    assert stale["next"] == "invalid" and any("not the one --o4 was scored from" in n for n in stale["notes"])
    assert _verdict(o4={**_o4_report(_gate_rows()), "selection": {"settings": {"max_episodes": 5}}})["next"] == \
        "invalid"
    with pytest.raises(ValueError, match="rft_gate.py"):
        _verdict(rgate={"schema": "something else"})
    with pytest.raises(ValueError, match="O3_exp28"):
        _verdict(o3={"G_A": {}})


def test_ledger_sizes_are_noted():
    small = g.compare_agreement(_agreement(100, 86), _agreement(100, 86), reference=g.O1_REFERENCE)
    result = _verdict(o1=small, o3=_o3(rows=2000))
    assert result["next"] == "val"
    assert "O1: 100 samples, the ledger's 3500" in result["notes"] and "O3: 2000 calls, the ledger's 3000" in \
        result["notes"]


def _write_exp32_inputs(root, rows, rgate, *, o1_hits=3007):
    (root / "o4").mkdir(parents=True, exist_ok=True)
    (root / "o4" / "hm3d.json").write_text(json.dumps(_o4_report(rows)))
    (root / "o4" / "hm3d.calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "o3.json").write_text(json.dumps(_o3(rows=3000)))
    (root / "rft_gate.json").write_text(json.dumps(rgate))
    argv = ["verdict", "--rules", "exp32", "--o3", str(root / "o3.json"), "--o4", str(root / "o4" / "hm3d.json"),
            "--rgate", str(root / "rft_gate.json"), "--output", str(root / "verdict_exp32.json")]
    for name, hits, n in (("o1", o1_hits, 3500), ("o1-base", 3007, 3500), ("o2", 585, 600), ("o2-base", 585, 600)):
        argv += [f"--{name}", str(_teacher_forced(root / name / "x.json", hits, n=n))]
    return argv


def test_verdict_cli_under_exp32_rules(tmp_path):
    argv = _write_exp32_inputs(tmp_path, _gate_rows(), _rgate())
    code, out, err = _run(argv)
    assert code == 0 and out.startswith("PASS EXP-32 offline gates attempt 1: O1=pass O2=pass O3=pass O4_anchor=pass")
    assert out.rstrip().endswith("L=pass R=pass -> next: val") and err == ""
    result = json.loads((tmp_path / "verdict_exp32.json").read_text())
    assert result["inputs"]["o4_calls"] == str((tmp_path / "o4" / "hm3d.calls.jsonl").resolve())
    assert result["inputs"]["rgate"] == str((tmp_path / "rft_gate.json").resolve())

    argv = _write_exp32_inputs(tmp_path, _gate_rows(), _rgate(r=False))
    code, out, _ = _run(argv + ["--attempt", "2"])
    assert code == 1 and out.startswith("FAIL EXP-32") and out.rstrip().endswith("R=FAIL -> next: kill")

    argv = _write_exp32_inputs(tmp_path, _gate_rows(), _rgate(uses="M", learnt=True))
    code, out, err = _run(argv)
    assert code == 0 and "(R read from M: R is report-only)" in out and "report-only" in err

    argv = _write_exp32_inputs(tmp_path, _gate_rows(), _rgate(uses=None))
    code, out, _ = _run(argv)
    assert code == 2 and out.startswith("INVALID EXP-32") and out.rstrip().endswith("-> next: invalid")

    (tmp_path / "rft_gate.json").unlink()
    code, out, _ = _run(argv)
    assert code == 2 and out.startswith("INVALID EXP-32 offline gates attempt 1: FileNotFoundError")
