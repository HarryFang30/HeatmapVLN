#!/usr/bin/env python3
"""EXP-19: read two or more finished reruns as one run (the pre-registered main-case fallback).

The main case of a category is the first candidate, in rank order, whose rerun satisfies
the category predicate (ledger EXP-19 "主图"; run record 5 on the RTX 4090 box).  When a
rank-0 case fails there, its next candidate is rerun as a separate run, and the later
stages must see both runs together.  This tool makes ``runs/<out>/`` whose ``gpu<k>``
entries are relative symlinks to the source runs' rank dirs (renumbered in the order
given), and writes ``runs/<out>/DONE`` only if every source run is complete and all were
made by the same code: same git sha, same source fingerprint (start = end), same code
sha256s, same protocol seed, same platform, and no episode in two runs.  Nothing is
copied or changed in the source runs.

Usage::

  python -m scripts.exp19.merge_runs --exp-root <EXP> --out <merged run> <run> <run> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DONE = "exp19-run-done-v1"
SAME = ("git_sha", "code_sha256", "protocol_seed", "trace_diagnostics", "platform")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_done(run_dir: Path) -> dict:
    done = json.loads((run_dir / "DONE").read_text())
    if done.get("schema") != SCHEMA_DONE:
        raise ValueError(f"{run_dir / 'DONE'}: schema {done.get('schema')!r}, expected {SCHEMA_DONE}")
    return done


def check(dones: dict) -> list:
    problems = []
    for name, d in dones.items():
        if d.get("status") != "complete":
            problems.append(f"{name}: status {d.get('status')!r}, not 'complete'")
        fp = d.get("source_fingerprint") or {}
        if not fp.get("unchanged"):
            problems.append(f"{name}: the source tree changed during the run")
    first, ref = next(iter(dones.items()))
    for name, d in dones.items():
        for key in SAME:
            if d.get(key) != ref.get(key):
                problems.append(f"{name}: {key} differs from {first}'s")
        if (d.get("source_fingerprint") or {}).get("start") != (ref.get("source_fingerprint") or {}).get("start"):
            problems.append(f"{name}: source fingerprint differs from {first}'s")
    seen = {}
    for name, d in dones.items():
        for rank in d.get("ranks") or []:
            for ep in rank.get("episodes") or []:
                if ep["ep_key"] in seen:
                    problems.append(f"{ep['ep_key']} is in both {seen[ep['ep_key']]} and {name}")
                seen[ep["ep_key"]] = name
    return problems


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp-root", type=Path, required=True)
    p.add_argument("--out", required=True, help="name of the merged run (runs/<out> must not exist)")
    p.add_argument("runs", nargs="+", help="source run names under <exp-root>/runs, in rank order")
    args = p.parse_args(argv)
    if len(args.runs) < 2 or len(set(args.runs)) != len(args.runs):
        p.error("give two or more distinct source runs")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    runs_dir = args.exp_root / "runs"
    out = runs_dir / args.out
    if out.exists():
        print(f"{out} exists; merged runs are never overwritten", file=sys.stderr)
        return 1
    dones = {}
    for name in args.runs:
        if not (runs_dir / name / "DONE").is_file():
            print(f"{runs_dir / name / 'DONE'} missing", file=sys.stderr)
            return 1
        dones[name] = load_done(runs_dir / name)
    problems = check(dones)
    if problems:
        for msg in problems:
            print(f"PROBLEM: {msg}", file=sys.stderr)
        print("refusing to merge", file=sys.stderr)
        return 1

    out.mkdir(parents=True)
    ranks, k = [], 0
    for name in args.runs:
        for rank in sorted(dones[name]["ranks"], key=lambda r: int(r["rank"])):
            src = runs_dir / name / f"gpu{int(rank['rank'])}"
            if not src.is_dir():
                raise FileNotFoundError(src)
            os.symlink(os.path.join("..", name, src.name), out / f"gpu{k}")
            ranks.append(dict(rank, rank=k, source_run=name, source_rank=int(rank["rank"])))
            k += 1
    ref = dones[args.runs[0]]
    done = {key: ref.get(key) for key in ("schema", "git_sha", "src", "code_sha256", "protocol_seed",
                                          "trace_diagnostics", "platform", "source_fingerprint")}
    done.update(run=args.out, status="complete", ranks=ranks,
                trace_error_lines=sum(int(d.get("trace_error_lines") or 0) for d in dones.values()),
                started_utc=min(d["started_utc"] for d in dones.values()),
                finished_utc=max(d["finished_utc"] for d in dones.values()),
                merged_from=[{"run": name, "done_sha256": sha256(runs_dir / name / "DONE")} for name in args.runs],
                merged_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                merged_by="scripts/exp19/merge_runs.py")
    (out / "DONE").write_text(json.dumps(done, indent=2) + "\n")
    print(f"{out}: " + ", ".join(f"gpu{r['rank']} -> {r['source_run']}/gpu{r['source_rank']}" for r in ranks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
