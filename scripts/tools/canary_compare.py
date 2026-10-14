#!/usr/bin/env python3
"""EXP-25's closed-loop canary (C-A): an adapter's episodes against the base model's round-1 outcomes.

The canary runs the held-out scenes' 392 round-1 episodes (``remaining_cohort.py --keep
committed --holdout``) with the adapter served, recording every call as round 1 did; the
base model's records of the same episodes are round 1's.  Both sides are read from the
records (``episode.json`` outcome, ``calls.jsonl``), episode by episode, on the episodes
both have.

    canary_compare.py --canary-records roundA/canary/records --base-records round1/records \\
        [--cohort roundA/canary_cohort] [--output canary_compare.json]

How an episode ended is read from its last call: a STOP answer ends it by the model's
STOP; otherwise it ran out of steps (500, or a step-budget STOP the client issued under
``--stop_at_step_budget``, which the model did not say).  Success under the standard
protocol is a model STOP within 3 m, so a step-budget STOP never counts (EXP-23 run record
8); the record's own ``success`` is reported beside it.

C-A (ledger EXP-25, fixed before any result):

* episodes ended by a STOP at >= 3 m:  canary <= base + 4
* episodes that ran out of steps:      canary <= base + 4
* turn reversals (a turn answer right after a turn the other way), share of turn calls:
  canary <= 0.02
* SR (standard protocol):              canary >= base - 1.0 pt

With ``--cohort`` every cohort episode must be on both sides, or the canary is incomplete
and does not pass.  Prints PASS/FAIL first; exit 0 on PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.stratified_gates import decision_class, turn_direction  # noqa: E402
from src.data.s2_records import EpisodeRecord, iter_record_paths, record_path  # noqa: E402

SCHEMA = "heatmapvln-exp25-canary-compare-v1"
FAR_STOP_M = 3.0
SPIN_M = 1.0
# C-A
STOP_FAR_MARGIN = 4
BUDGET_MARGIN = 4
REVERSAL_MAX = 0.02
SR_DROP_MAX_PT = 1.0


def episode_summary(record: EpisodeRecord) -> dict[str, Any]:
    calls = sorted(record.calls, key=lambda c: int(c["call_index"]))
    classes = [decision_class(str(c.get("llm_output") or "")) for c in calls]
    outcome = record.episode.get("outcome") or {}
    distance = float(outcome["distance_to_goal"])
    stopped = bool(classes) and classes[-1] == "stop"
    turns = sum(1 for c in classes if turn_direction(c))
    reversals = sum(1 for a, b in zip(classes, classes[1:])
                    if turn_direction(a) and turn_direction(b) and turn_direction(a) != turn_direction(b))
    spin, spin_stop = 0, 0
    for previous, call, cls in zip(classes, calls[1:], classes[1:]):
        d = call.get("distance_to_goal")
        if turn_direction(previous) and d is not None and float(d) <= SPIN_M:
            spin += 1
            spin_stop += cls == "stop"
    return {
        "ended_by": "stop" if stopped else "steps",
        "distance_to_goal": distance,
        "success_standard": bool(stopped and distance <= FAR_STOP_M),
        "success_recorded": float(outcome.get("success", 0.0)) > 0.5,
        "steps": int(outcome.get("steps", 0)),
        "calls": len(classes),
        "turn_calls": turns,
        "reversals": reversals,
        "spin_states": spin,
        "spin_stops": spin_stop,
    }


def load(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for path in iter_record_paths(root):
        record = EpisodeRecord(path)
        out[(record.scene_id, record.episode_id)] = episode_summary(record)
    return out


def side_totals(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    turns = sum(e["turn_calls"] for e in episodes)
    reversals = sum(e["reversals"] for e in episodes)
    spin = sum(e["spin_states"] for e in episodes)
    spin_stop = sum(e["spin_stops"] for e in episodes)
    return {
        "episodes": n,
        "sr_pct": round(100.0 * sum(e["success_standard"] for e in episodes) / n, 3) if n else None,
        "sr_recorded_pct": round(100.0 * sum(e["success_recorded"] for e in episodes) / n, 3) if n else None,
        "stop_far": sum(e["ended_by"] == "stop" and e["distance_to_goal"] >= FAR_STOP_M for e in episodes),
        "out_of_steps": sum(e["ended_by"] == "steps" for e in episodes),
        "out_of_steps_min_steps": min((e["steps"] for e in episodes if e["ended_by"] == "steps"), default=None),
        "turn_calls": turns,
        "reversals": reversals,
        "reversal_share": round(reversals / turns, 6) if turns else None,
        "spin_states": spin,
        "spin_stop_rate": round(spin_stop / spin, 6) if spin else None,
        "mean_calls": round(sum(e["calls"] for e in episodes) / n, 2) if n else None,
    }


def compare(canary: Mapping, base: Mapping, expected: set | None) -> dict[str, Any]:
    """C-A on the episodes in scope (the cohort's, else the canary's) that both sides have."""
    scope = set(expected) if expected is not None else set(canary)
    missing_canary = sorted(scope - set(canary))
    missing_base = sorted(scope - set(base))
    keys = sorted(scope & set(canary) & set(base))
    c = side_totals([canary[k] for k in keys])
    b = side_totals([base[k] for k in keys])

    def crit(value, threshold, direction, **extra):
        ok = value is not None and threshold is not None and (value <= threshold if direction == "<=" else value >= threshold)
        return {"value": value, "threshold": threshold, "direction": direction, "pass": bool(ok), **extra}

    criteria = {
        "stop_far_episodes": crit(c["stop_far"], b["stop_far"] + STOP_FAR_MARGIN, "<=", base=b["stop_far"]),
        "out_of_steps_episodes": crit(c["out_of_steps"], b["out_of_steps"] + BUDGET_MARGIN, "<=", base=b["out_of_steps"]),
        "turn_reversal_share": crit(c["reversal_share"], REVERSAL_MAX, "<=", base=b["reversal_share"]),
        "sr_pct": crit(c["sr_pct"], None if b["sr_pct"] is None else round(b["sr_pct"] - SR_DROP_MAX_PT, 3), ">=",
                       base=b["sr_pct"]),
    }
    complete = not missing_canary and not missing_base
    flips = {
        "base_success_canary_fail": sum(base[k]["success_standard"] and not canary[k]["success_standard"] for k in keys),
        "base_fail_canary_success": sum(canary[k]["success_standard"] and not base[k]["success_standard"] for k in keys),
    }
    return {
        "schema": SCHEMA,
        "matched_episodes": len(keys),
        "expected_episodes": len(expected) if expected is not None else None,
        "complete": complete,
        "missing_in_canary": [f"{s}_{e:06d}" for s, e in missing_canary][:50],
        "missing_in_base": [f"{s}_{e:06d}" for s, e in missing_base][:50],
        "criteria": criteria,
        "all_pass": complete and all(v["pass"] for v in criteria.values()),
        "canary": c,
        "base": b,
        "report_only": {"paired_flips": flips},
    }


def cohort_episodes(cohort: Path) -> set[tuple[str, int]]:
    manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
    keys = set()
    for shard in manifest["shards"]:
        document = json.loads((cohort / shard["path"]).read_text(encoding="utf-8"))
        keys.update((str(e["scene_id"]), int(e["episode_id"])) for e in document["episodes"])
    return keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canary-records", required=True)
    parser.add_argument("--base-records", required=True)
    parser.add_argument("--cohort", default="", help="the canary cohort: every episode must be on both sides")
    parser.add_argument("--output", default="", help="json (default: print only)")
    args = parser.parse_args(argv)
    expected = cohort_episodes(Path(args.cohort)) if args.cohort else None
    canary = load(Path(args.canary_records))
    base = {}
    for scene, episode in sorted(expected if expected is not None else canary):
        path = record_path(args.base_records, scene, episode)
        if path.with_name(path.name + ".sha256").is_file():
            base[(scene, episode)] = episode_summary(EpisodeRecord(path))
    result = compare(canary, base, expected)
    result["inputs"] = {"canary_records": args.canary_records, "base_records": args.base_records,
                        "cohort": args.cohort or None}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    line = ("PASS" if result["all_pass"] else "FAIL") + f" C-A matched={result['matched_episodes']}"
    line += f" complete={result['complete']} " + " ".join(
        f"{k}={v['value']}(base {v['base']}, line {v['direction']}{v['threshold']}){'' if v['pass'] else ' FAIL'}"
        for k, v in result["criteria"].items())
    print(line)
    if not args.output:
        print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
