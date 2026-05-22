#!/usr/bin/env python3
"""Summarize paired InternNav vs HeatmapVLN progress JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_progress(path: str) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        res = json.loads(line)
        key = (res["scene_id"], int(res["episode_id"]))
        rows[key] = res
    return rows


def metrics(rows: dict[tuple[str, int], dict]) -> dict:
    if not rows:
        return {"SR": 0.0, "SPL": 0.0, "OS": 0.0, "NE": 0.0, "n": 0}
    n = len(rows)
    return {
        "SR": sum(r["success"] for r in rows.values()) / n,
        "SPL": sum(r["spl"] for r in rows.values()) / n,
        "OS": sum(r["os"] for r in rows.values()) / n,
        "NE": sum(r["ne"] for r in rows.values()) / n,
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_list", required=True)
    parser.add_argument("--internnav", required=True)
    parser.add_argument("--heatmapvln", required=True)
    args = parser.parse_args()

    ep_data = json.loads(Path(args.episode_list).read_text(encoding="utf-8"))
    keys = [(e["scene_id"], int(e["episode_id"])) for e in ep_data["episodes"]]

    intern = load_progress(args.internnav)
    heat = load_progress(args.heatmapvln)

    print(f"Episode list: {len(keys)} episodes")
    print(f"InternNav results: {len(intern)}")
    print(f"HeatmapVLN results: {len(heat)}")

    missing_i = [k for k in keys if k not in intern]
    missing_h = [k for k in keys if k not in heat]
    if missing_i:
        print(f"WARNING: InternNav missing {len(missing_i)} episodes")
    if missing_h:
        print(f"WARNING: HeatmapVLN missing {len(missing_h)} episodes")

    intern_ordered = {k: intern[k] for k in keys if k in intern}
    heat_ordered = {k: heat[k] for k in keys if k in heat}

    mi = metrics(intern_ordered)
    mh = metrics(heat_ordered)
    print("\n| Model | SR | SPL | OS | NE | n |")
    print("|-------|-----|------|-----|------|---|")
    print(
        f"| InternNav | {mi['SR']*100:.1f}% | {mi['SPL']*100:.1f}% | "
        f"{mi['OS']*100:.1f}% | {mi['NE']:.2f} | {mi['n']} |"
    )
    print(
        f"| HeatmapVLN | {mh['SR']*100:.1f}% | {mh['SPL']*100:.1f}% | "
        f"{mh['OS']*100:.1f}% | {mh['NE']:.2f} | {mh['n']} |"
    )

    print("\nPer-episode (scene_ep | InternNav SR | HeatmapVLN SR | steps I/H):")
    for k in keys:
        if k not in intern_ordered or k not in heat_ordered:
            continue
        ri, rh = intern_ordered[k], heat_ordered[k]
        print(
            f"  {k[0]}_{k[1]:04d} | "
            f"{int(ri['success'])} | {int(rh['success'])} | "
            f"{ri.get('steps', '?')}/{rh.get('steps', '?')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
