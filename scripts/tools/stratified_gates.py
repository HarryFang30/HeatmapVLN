#!/usr/bin/env python3
"""EXP-25's stratified offline gates (G-A) and round 1b's S0, from a gate run's per-call rows.

EXP-23's gates looked at overall rates, and round 1 passed G0 while learning two things
they never measured: STOP on the approach after a pixel goal (1-3 m: base 1.9%, round 1
54.3%) and left/right jitter after a turn.  The ledger's EXP-25 judges an adapter by the
state it is in instead: the kind of the *previous* call in the episode (the one the base
model executed while collecting) crossed with the distance to the goal.

    stratified_gates.py --calls lora/gates_<stamp>.calls.jsonl \\
        [--report lora/gates_<stamp>.json] [--records <held-out record root>] [--output out.json]

Every row of ``eval_system2_lora_offline.py``'s ``<output>.calls.jsonl`` carries the
greedy decision of the adapter (``on``) and of the base model (``off``) on one held-out
call; its ``uid`` is ``<record path>#<call index>``.  The record gives the previous call
(``--records`` re-roots the path when the run read the records elsewhere: the record is
looked up as ``<records>/<scene>/<file>``).

Bins: previous call {pixel, turn, none (first call), other} x distance {<=1, 1-3, 3-6,
>6 m} (left-open: 1-3 is 1 < d <= 3).  Criteria (ledger EXP-25, fixed before any result):

====  ===============================================================  =============
G-A   quantity (adapter on, same calls as the base model)               line
====  ===============================================================  =============
(a)   spin state (previous call a turn, d <= 1 m): STOP rate            >= 0.30
(b)   STOP-rate increase over base: after a pixel 3-6 m                 <= +0.0025
      after a turn 3-6 m                                                <= +0.004
      beyond 6 m (any previous call)                                    <= +0.003
(c)   after a pixel 1-3 m: STOP rate                                    <= 0.05
(d)   after a turn: answer turns the opposite way to the executed turn  <= 0.01
      (share of all calls whose previous call was a turn); turn
      direction differs from base (calls where both turned)             <= 0.05
(e)   d >= 3 m: decision class equals base's                            >= 0.95
      G0 (adapter-off replay, from the run's gate report)               >= 0.99
====  ===============================================================  =============

S0 (round 1b, EXP-23 run record 12): after a pixel 1-3 m STOP rate >= 0.10, or reversal
>= 0.10, or turn direction differs from base >= 0.10 -> 1b does not run closed loop; a
reversal <= 0.02 means the jitter diagnosis was wrong (recheck before training A).

Only the standard library and ``src.data.s2_records`` are imported, so it runs where the
gates ran without touching an accelerator.  Prints PASS/FAIL first; exit 0 on PASS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.s2_records import EpisodeRecord  # noqa: E402

SCHEMA = "heatmapvln-s2-stratified-gates-v1"
SPIN_M = 1.0
DISTANCE_BINS = (("le1", None, 1.0), ("1-3", 1.0, 3.0), ("3-6", 3.0, 6.0), ("gt6", 6.0, None))
PREV_KINDS = ("pixel", "turn", "none", "other")

# EXP-25 G-A, written 2026-10-15 before any result.
A_SPIN_STOP_MIN = 0.30
B_PIXEL_3_6_MAX = 0.0025
B_TURN_3_6_MAX = 0.004
B_GT6_MAX = 0.003
C_PIXEL_1_3_MAX = 0.05
D_REVERSAL_MAX = 0.01
D_DIRECTION_MAX = 0.05
E_CLASS_AGREE_MIN = 0.95
E_G0_MIN = 0.99
# S0 for round 1b.
S0_PIXEL_1_3 = 0.10
S0_REVERSAL = 0.10
S0_DIRECTION = 0.10
S0_REVERSAL_RECHECK = 0.02

_DIGIT = re.compile(r"\d")
_INTEGER = re.compile(r"\d+")
_TOKEN = re.compile("STOP|↑|←|→|↓")
_TOKEN_CLASS = {"STOP": "stop", "↑": "forward", "←": "turn_left", "→": "turn_right", "↓": "lookdown"}


def decision_class(text: str) -> str:
    """``src.data.s2_sft_dataset.decision_class`` without its torch imports (tests pin them equal)."""
    text = text or ""
    if _DIGIT.search(text):
        return "pixel" if len(_INTEGER.findall(text)) >= 2 else "invalid"
    tokens = _TOKEN.findall(text)
    return _TOKEN_CLASS[tokens[0]] if tokens else "stop"


def recorded_class(call: Mapping[str, Any]) -> str:
    """Decision class of what the server executed for a recorded call (its final text)."""
    return decision_class(str(call.get("llm_output") or ""))


def turn_direction(cls: str | None) -> str | None:
    return {"turn_left": "L", "turn_right": "R"}.get(cls or "")


def prev_kind(cls: str | None) -> str:
    if cls is None:
        return "none"
    if cls == "pixel":
        return "pixel"
    if turn_direction(cls):
        return "turn"
    return "other"


def distance_bin(d: float) -> str:
    for name, low, high in DISTANCE_BINS:
        if (low is None or d > low) and (high is None or d <= high):
            return name
    raise AssertionError(d)


def episode_context(calls: Sequence[Mapping[str, Any]], distances: Mapping[int, float | None]) -> dict[int, dict]:
    """Per call index: previous call's class and how many earlier calls were within SPIN_M."""
    ordered = sorted(calls, key=lambda c: int(c["call_index"]))
    out: dict[int, dict] = {}
    within = 0
    previous: str | None = None
    for call in ordered:
        index = int(call["call_index"])
        out[index] = {"prev_class": previous, "earlier_within_1m": within}
        d = distances.get(index)
        if d is not None and d <= SPIN_M:
            within += 1
        previous = recorded_class(call)
    return out


def _record_path(uid: str, records: Path | None) -> tuple[Path, int]:
    path, _, index = uid.rpartition("#")
    path = Path(path)
    if records is not None:
        path = records / path.parent.name / path.name
    return path, int(index)


def attach_context(rows: Iterable[Mapping[str, Any]], records: Path | None) -> tuple[list[dict], Counter]:
    """Rows joined with their previous call (from the record); unmatched rows are counted."""
    stats: Counter = Counter()
    cache: dict[Path, dict[int, dict] | None] = {}
    joined = []
    for row in rows:
        path, index = _record_path(str(row["uid"]), records)
        if path not in cache:
            try:
                record = EpisodeRecord(path)
            except (FileNotFoundError, OSError, ValueError) as exc:
                stats["records_unreadable"] += 1
                print(f"[stratified] cannot read {path}: {exc}", file=sys.stderr)
                cache[path] = None
            else:
                distances = {int(c["call_index"]): c.get("distance_to_goal") for c in record.calls}
                cache[path] = episode_context(record.calls, distances)
        context = (cache[path] or {}).get(index)
        if context is None:
            stats["rows_without_record"] += 1
            continue
        joined.append(dict(row, **context))
    stats["rows"] = len(joined)
    stats["records"] = sum(1 for v in cache.values() if v is not None)
    return joined, stats


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _criterion(value: float | None, threshold: float, direction: str, **extra: Any) -> dict[str, Any]:
    ok = value is not None and (value <= threshold if direction == "<=" else value >= threshold)
    return {"value": value, "threshold": threshold, "direction": direction, "pass": bool(ok), **extra}


def _stop(row: Mapping[str, Any], mode: str) -> bool:
    return row[mode].get("class") == "stop"


def stop_table(rows: Sequence[Mapping[str, Any]], select) -> dict[str, Any]:
    chosen = [r for r in rows if select(r)]
    on = sum(_stop(r, "on") for r in chosen)
    off = sum(_stop(r, "off") for r in chosen)
    n = len(chosen)
    return {"n": n, "on_stops": on, "off_stops": off, "on_rate": _rate(on, n), "off_rate": _rate(off, n),
            "delta": round((on - off) / n, 6) if n else None}


def reversal(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    """Share of calls after an executed turn whose answer turns the other way."""
    after_turn = [r for r in rows if prev_kind(r["prev_class"]) == "turn"]
    turned = [r for r in after_turn if turn_direction(r[mode].get("class"))]
    reversed_ = [r for r in turned if turn_direction(r[mode]["class"]) != turn_direction(r["prev_class"])]
    return {"n_after_turn": len(after_turn), "answered_turn": len(turned), "reversed": len(reversed_),
            "rate": _rate(len(reversed_), len(after_turn)),
            "rate_of_turn_answers": _rate(len(reversed_), len(turned))}


def direction_differs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both = [r for r in rows if turn_direction(r["on"].get("class")) and turn_direction(r["off"].get("class"))]
    differ = sum(r["on"]["class"] != r["off"]["class"] for r in both)
    return {"n_both_turn": len(both), "differ": differ, "rate": _rate(differ, len(both))}


def stratify(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any] | None) -> dict[str, Any]:
    rows = [r for r in rows if r.get("distance_to_goal") is not None]

    def kind(r):
        return prev_kind(r["prev_class"])

    def dist(r):
        return float(r["distance_to_goal"])

    bins = {}
    for k in PREV_KINDS:
        for name, _, _ in DISTANCE_BINS:
            bins[f"{k}/{name}"] = stop_table(rows, lambda r, k=k, name=name: kind(r) == k and distance_bin(dist(r)) == name)
    spin = stop_table(rows, lambda r: kind(r) == "turn" and dist(r) <= SPIN_M)
    spin_strict = stop_table(rows, lambda r: kind(r) == "turn" and dist(r) <= SPIN_M and r["earlier_within_1m"] >= 1)
    pixel_1_3 = bins["pixel/1-3"]
    pixel_3_6 = bins["pixel/3-6"]
    turn_3_6 = bins["turn/3-6"]
    gt6 = stop_table(rows, lambda r: dist(r) > 6.0)
    rev_on, rev_off = reversal(rows, "on"), reversal(rows, "off")
    differs = direction_differs(rows)
    far = [r for r in rows if dist(r) >= 3.0]
    agree = sum(r["on"].get("class") == r["off"].get("class") for r in far)
    g0 = None
    if report is not None:
        g0 = (report.get("gates") or {}).get("G0_adapter_off_replay", {}).get("value")

    ga = {
        "a_spin_stop_rate": _criterion(spin["on_rate"], A_SPIN_STOP_MIN, ">=", n=spin["n"], base=spin["off_rate"],
                                       state="previous call a turn, d <= 1 m"),
        "b_pixel_3_6_stop_increase": _criterion(pixel_3_6["delta"], B_PIXEL_3_6_MAX, "<=", n=pixel_3_6["n"],
                                                on=pixel_3_6["on_rate"], base=pixel_3_6["off_rate"]),
        "b_turn_3_6_stop_increase": _criterion(turn_3_6["delta"], B_TURN_3_6_MAX, "<=", n=turn_3_6["n"],
                                               on=turn_3_6["on_rate"], base=turn_3_6["off_rate"]),
        "b_gt6_stop_increase": _criterion(gt6["delta"], B_GT6_MAX, "<=", n=gt6["n"], on=gt6["on_rate"], base=gt6["off_rate"]),
        "c_pixel_1_3_stop_rate": _criterion(pixel_1_3["on_rate"], C_PIXEL_1_3_MAX, "<=", n=pixel_1_3["n"],
                                            base=pixel_1_3["off_rate"]),
        "d_post_turn_reversal": _criterion(rev_on["rate"], D_REVERSAL_MAX, "<=", n=rev_on["n_after_turn"],
                                           reversed=rev_on["reversed"], base=rev_off["rate"]),
        "d_turn_direction_differs_from_base": _criterion(differs["rate"], D_DIRECTION_MAX, "<=", n=differs["n_both_turn"],
                                                         differ=differs["differ"]),
        "e_class_agree_ge3m": _criterion(_rate(agree, len(far)), E_CLASS_AGREE_MIN, ">=", n=len(far), agree=agree),
        "e_G0": _criterion(g0, E_G0_MIN, ">=", source="gate report G0_adapter_off_replay"),
    }
    ga_pass = all(c["pass"] for c in ga.values())
    s0_triggers = {
        "pixel_1_3_stop_rate": _criterion(pixel_1_3["on_rate"], S0_PIXEL_1_3, ">="),
        "post_turn_reversal": _criterion(rev_on["rate"], S0_REVERSAL, ">="),
        "turn_direction_differs": _criterion(differs["rate"], S0_DIRECTION, ">="),
    }
    s0_blocked = any(c["pass"] for c in s0_triggers.values())
    recheck = rev_on["rate"] is not None and rev_on["rate"] <= S0_REVERSAL_RECHECK
    return {
        "schema": SCHEMA,
        "rows_with_distance": len(rows),
        "G_A": {"criteria": ga, "all_pass": ga_pass},
        "S0_round1b": {
            "triggers": s0_triggers,
            "closed_loop": "no" if s0_blocked else "not blocked by S0",
            "reversal_le_0.02_recheck_diagnosis": recheck,
        },
        "bins": bins,
        "report_only": {
            "spin_strict_ge1_earlier_within_1m": spin_strict,
            "gt6_any_previous": gt6,
            "reversal_base": rev_off,
            "reversal_adapter": rev_on,
            "turn_direction": differs,
            "stop_rate_all": stop_table(rows, lambda r: True),
        },
    }


def summary_line(result: Mapping[str, Any]) -> str:
    ga = result["G_A"]["criteria"]
    parts = [f"{k}={v['value']}{'' if v['pass'] else '(FAIL)'}" for k, v in ga.items()]
    s0 = result["S0_round1b"]
    return (("PASS" if result["G_A"]["all_pass"] else "FAIL") + " G-A " + " ".join(parts)
            + f" | S0 closed_loop={s0['closed_loop']} recheck_diagnosis={s0['reversal_le_0.02_recheck_diagnosis']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", required=True, help="gates_<stamp>.calls.jsonl")
    parser.add_argument("--report", default="", help="gates_<stamp>.json (G0); default: beside --calls")
    parser.add_argument("--records", default="", help="re-root the rows' record paths here (<records>/<scene>/<file>)")
    parser.add_argument("--output", default="", help="default: <calls stem>.stratified.json beside --calls")
    args = parser.parse_args(argv)
    calls = Path(args.calls)
    report_path = Path(args.report) if args.report else calls.with_name(calls.name.replace(".calls.jsonl", ".json"))
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if report is None:
        print(f"[stratified] no gate report at {report_path}: G0 counts as missing", file=sys.stderr)
    rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines() if line.strip()]
    joined, stats = attach_context(rows, Path(args.records) if args.records else None)
    result = stratify(joined, report)
    result["inputs"] = {"calls": str(calls.resolve()), "report": str(report_path) if report else None,
                        "records": args.records or None, "join": dict(stats), "rows_in_file": len(rows)}
    output = Path(args.output) if args.output else calls.with_name(calls.name.replace(".calls.jsonl", "") + ".stratified.json")
    output.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_line(result))
    print(f"[stratified] {dict(stats)} -> {output}", file=sys.stderr)
    return 0 if result["G_A"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
