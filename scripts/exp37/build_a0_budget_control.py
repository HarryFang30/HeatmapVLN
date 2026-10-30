#!/usr/bin/env python3
"""EXP-37 on the 4090 (ledger 运行记录 2): the same-platform control, A0 seed 1337 + step-budget STOP.

A0's merged rows (EXP-20, 4090, sdpa) with the 37 pinned episodes' rows replaced by EXP-24's rerun with the
switch on, the row replacement ``/workspace/exp24/combine_budget.py`` did; written out as one progress.jsonl
because X37 pairs with it episode by episode.  Refuses unless it holds 1,839 unique episodes and reads
SR 62.59 / SPL 54.94, the numbers EXP-24 recorded.

    python scripts/exp37/build_a0_budget_control.py --output /workspace/eval_runs/exp37_A0budget_seed1337_4090
"""
import argparse
import json
from pathlib import Path

A0 = "/workspace/eval_runs/exp20_a0_seed1337_4090/merged/progress.jsonl"
RERUN = "/workspace/exp24/run_1020/seed1337"
SELECTION = "/workspace/exp24/lists/seed1337/selection.json"
EPISODES = 1839
SR, SPL = 62.59, 54.94


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return row["scene_id"], int(row["episode_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = {key(r): r for r in rows(A0)}
    if len(base) != EPISODES:
        raise SystemExit(f"{A0}: {len(base)} unique episodes, not {EPISODES}")
    pinned = {tuple(k.rsplit("_", 1)) for k in json.loads(Path(SELECTION).read_text())["a0"]}
    pinned = {(scene, int(episode)) for scene, episode in pinned}
    rerun = {key(r): r for p in sorted(Path(RERUN, "workers").glob("shard_0*/progress.json")) for r in rows(p)}
    missing = pinned - set(rerun)
    if missing:
        raise SystemExit(f"{len(missing)} pinned episodes have no rerun row: {sorted(missing)[:5]}")
    if not all(rerun[k].get("stop_at_step_budget") is True for k in pinned):
        raise SystemExit("a pinned episode's rerun row did not run with the switch on")

    control = {**base, **{k: rerun[k] for k in pinned}}
    n = len(control)
    sr, spl = (round(100 * sum(r[f] for r in control.values()) / n, 2) for f in ("success", "spl"))
    if (sr, spl) != (SR, SPL):
        raise SystemExit(f"control reads SR {sr} / SPL {spl}, not EXP-24's {SR} / {SPL}")

    out = args.output / "merged"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "progress.jsonl", "w") as handle:
        for k in sorted(control):
            handle.write(json.dumps(control[k]) + "\n")
    print(f"{n} episodes, {len(pinned)} replaced: SR {sr} SPL {spl} -> {out / 'progress.jsonl'}")


if __name__ == "__main__":
    main()
