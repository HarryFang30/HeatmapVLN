#!/usr/bin/env python3
"""EXP-23's revised offline gates, recomputed from a gate run's per-call records.

Revised on 2026-10-14 after round 1's gate results and before any result of round 1b
(ledger, EXP-23 run record 13):

* G0 unchanged: the base model replays its own same-device calls (>= 0.98), read from
  the original report;
* G1: on the states where the base model agreed with the oracle, the adapter keeps the
  base model's decision class (pixel / turn left / turn right / stop) on >= 0.90 of them.
  Exact text equality was dominated by pixel jitter (round 1: median 2 px); the pixel
  offsets are reported, not gated;
* G2: the adapter's STOP rate beyond the false-stop distance is at most 1.5x the base
  model's on the same calls.  The absolute 0.002 was below the base model's own 0.0049 on
  these DAgger-collected states.

    regate_s2_lora.py --report lora/gates_<stamp>.json --calls lora/gates_<stamp>.calls.jsonl

Writes ``<report stem>.revised.json`` beside the report and prints PASS or FAIL first.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

G1_MIN = 0.90
G2_MAX_RATIO = 1.5


def _pixel(turns):
    if turns and len(turns) == 2 and turns[0] == "↓":
        try:
            u, v = (float(x) for x in turns[1].split())
            return u, v
        except ValueError:
            return None
    return None


def _quantiles(values, probs=(0.5, 0.75, 0.9)):
    values = sorted(values)
    return {f"q{int(p * 100)}": round(values[int(p * (len(values) - 1))], 1) for p in probs} if values else {}


def regate(report: dict, rows: list[dict]) -> dict:
    rows = [r for r in rows if not r.get("skip")]
    far_m = float(report.get("settings", {}).get("false_stop_distance_m", 3.0))
    g0 = report["gates"]["G0_adapter_off_replay"]

    agree = [r for r in rows if r.get("agree")]
    kept = sum(r["on"]["class"] == r["off"]["class"] for r in agree)
    g1 = kept / len(agree) if agree else float("nan")
    offsets = []
    for r in agree:
        a, b = _pixel(r["off"].get("turns")), _pixel(r["on"].get("turns"))
        if a and b:
            offsets.append(math.dist(a, b))

    far = [r for r in rows if float(r["distance_to_goal"]) > far_m]
    on_stops = sum(r["on"]["class"] == "stop" for r in far)
    off_stops = sum(r["off"]["class"] == "stop" for r in far)
    on_rate = on_stops / len(far) if far else float("nan")
    off_rate = off_stops / len(far) if far else float("nan")
    limit = G2_MAX_RATIO * off_rate

    gates = {
        "G0_adapter_off_replay": {"value": g0["value"], "threshold": g0["threshold"], "direction": ">=", "pass": bool(g0["pass"])},
        "G1_class_keep": {"value": round(g1, 6), "threshold": G1_MIN, "direction": ">=", "pass": g1 >= G1_MIN,
                          "kept": kept, "n": len(agree)},
        "G2_false_stop_vs_base": {"value": round(on_rate, 6), "threshold": round(limit, 6), "direction": "<=",
                                  "pass": on_rate <= limit, "base_rate": round(off_rate, 6), "ratio_max": G2_MAX_RATIO,
                                  "stops": on_stops, "base_stops": off_stops, "n": len(far), "distance_gt_m": far_m},
    }
    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict))
    return {
        "schema": "heatmapvln-s2-lora-revised-gates-v1",
        "revised": "2026-10-14, after round 1's gate results, before any round 1b result (EXP-23 run record 13)",
        "gates": gates,
        "report_only": {
            "pixel_offset_on_agree_px": dict(_quantiles(offsets), n=len(offsets),
                                             identical=sum(d == 0 for d in offsets),
                                             within_20px=round(sum(d <= 20 for d in offsets) / len(offsets), 4) if offsets else None),
            "original_gates": report["gates"],
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True)
    parser.add_argument("--calls", required=True)
    args = parser.parse_args(argv)
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.calls).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = regate(report, rows)
    out = report_path.with_name(report_path.stem + ".revised.json")
    out.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    g = result["gates"]
    print(("PASS" if g["all_pass"] else "FAIL") + " " + json.dumps(
        {k: v["value"] for k, v in g.items() if isinstance(v, dict)} | {"G2_limit": g["G2_false_stop_vs_base"]["threshold"]}))
    return 0 if g["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
