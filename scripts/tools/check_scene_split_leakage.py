#!/usr/bin/env python3
"""Preflight: no training source may contain a scene that the shared split holds out.

Every EXP-13+ readout and fine-tune holds scenes out with one rule,
``md5(scene_id)[:8] % 100``: <25 val, 25..39 dev, else train.  Any dataset used for
training must be checked against that rule *before* a job is submitted, because
the sources overlap: the 26 R2R v2 expert scenes all sit inside the 61 DAgger
scenes, 6 of them in the val bucket and 4 in dev (measured 2026-09-06).

The tool takes any number of ``--source NAME=PATH`` entries.  A path is either a
directory whose immediate children are scene ids (R2R v2 layout) or a JSONL file
whose rows carry ``scene_id`` (the EXP-12 per-state file).  It prints the
intersections and exits non-zero when a source touches val or dev, unless that
source is explicitly declared ``--evaluation-only NAME``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def scene_bucket(scene: str) -> int:
    return int(hashlib.md5(str(scene).encode("utf-8")).hexdigest()[:8], 16) % 100


def bucket_name(scene: str, val_pct: int, dev_pct: int) -> str:
    bucket = scene_bucket(scene)
    if bucket < val_pct:
        return "val"
    if bucket < val_pct + dev_pct:
        return "dev"
    return "train"


def load_scenes(path: Path) -> set[str]:
    if path.is_dir():
        return {child.name for child in path.iterdir() if child.is_dir()}
    scenes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scenes.add(str(json.loads(line)["scene_id"]))
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[], help="NAME=PATH; repeatable")
    parser.add_argument("--evaluation-only", action="append", default=[], help="source NAME that is never trained on")
    parser.add_argument("--val-pct", type=int, default=25)
    parser.add_argument("--dev-pct", type=int, default=15)
    args = parser.parse_args()
    if not args.source:
        parser.error("at least one --source NAME=PATH is required")

    report: dict[str, dict[str, list[str]]] = {}
    failed = False
    for entry in args.source:
        name, _, raw = entry.partition("=")
        if not raw:
            parser.error(f"--source expects NAME=PATH, got {entry!r}")
        scenes = load_scenes(Path(raw))
        by_bucket = {"train": [], "dev": [], "val": []}
        for scene in sorted(scenes):
            by_bucket[bucket_name(scene, args.val_pct, args.dev_pct)].append(scene)
        report[name] = by_bucket
        touches = by_bucket["val"] or by_bucket["dev"]
        if touches and name not in args.evaluation_only:
            failed = True
    print(json.dumps({"rule": f"md5[:8]%100 <{args.val_pct} val, <{args.val_pct + args.dev_pct} dev", "sources": report}, indent=1))
    if failed:
        print("LEAKAGE: a training source contains held-out (val/dev) scenes; drop them or declare --evaluation-only", file=sys.stderr)
        return 1
    print("OK: no training source touches val or dev scenes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
