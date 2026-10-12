#!/usr/bin/env python3
"""Compare two evaluations' client logs call by call, for the episodes both ran.

EXP-23 pairs every candidate run against the EXP-20 A0 runs of the same seed on the
same 4090.  That pairing is only clean if the new code, serving no adapter, makes the
same calls A0 made.  This reads both runs' ``client_shard_*.log`` with the EXP-19
parser (``scripts/exp19/select_cases.parse_client_log``) and, per shared episode,
reports whether every System2 call (step, kind, model text, executed actions) and
the episode end are identical, or where they first diverge.

    python scripts/tools/compare_client_logs.py \\
        --a '/workspace/eval_runs/exp20_a0_seed42_4090/runtime/*/logs/client_shard_*.log' \\
        --b '/workspace/exp23/round1/canary_noadapter/runtime/*/logs/client_shard_*.log' \\
        --out canary_identity.json --require-identical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp19.select_cases import parse_client_logs  # noqa: E402

CALL_FIELDS = ("step", "kind", "vlm_output", "actions")


def compare_blocks(a: dict, b: dict) -> dict:
    calls_a, calls_b = a["calls"], b["calls"]
    first = None
    for index, (ca, cb) in enumerate(zip(calls_a, calls_b)):
        if any(ca.get(f) != cb.get(f) for f in CALL_FIELDS):
            first = {"call_index": index, "a": {f: ca.get(f) for f in CALL_FIELDS},
                     "b": {f: cb.get(f) for f in CALL_FIELDS}}
            break
    if first is None and len(calls_a) != len(calls_b):
        first = {"call_index": min(len(calls_a), len(calls_b)), "a_calls": len(calls_a), "b_calls": len(calls_b)}
    return {
        "scene_id": a["scene_id"],
        "episode_id": a["episode_id"],
        "calls_a": len(calls_a),
        "calls_b": len(calls_b),
        "identical": first is None and a.get("final") == b.get("final"),
        "final_a": a.get("final"),
        "final_b": b.get("final"),
        "first_divergence": first,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="glob of the reference run's client logs")
    parser.add_argument("--b", required=True, help="glob of the compared run's client logs")
    parser.add_argument("--out", default="")
    parser.add_argument("--require-identical", action="store_true", help="exit 1 unless every shared episode is identical")
    args = parser.parse_args(argv)
    blocks_a, *_ = parse_client_logs(args.a)
    blocks_b, *_ = parse_client_logs(args.b)
    shared = sorted(set(blocks_a) & set(blocks_b))
    rows = [compare_blocks(blocks_a[e], blocks_b[e]) for e in shared]
    report = {
        "a": args.a,
        "b": args.b,
        "shared_episodes": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "calls_compared": sum(min(r["calls_a"], r["calls_b"]) for r in rows),
        "episodes": rows,
    }
    for r in rows:
        print(f"{r['scene_id']}/{r['episode_id']}: {'identical' if r['identical'] else 'DIVERGES'} "
              f"({r['calls_a']} vs {r['calls_b']} calls)"
              + ("" if r["identical"] else f" first at {r['first_divergence']}"))
    print(json.dumps({k: report[k] for k in ("shared_episodes", "identical", "calls_compared")}))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.require_identical and (not rows or report["identical"] != len(rows)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
