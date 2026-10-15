#!/usr/bin/env python3
"""EXP-27: did the base System2 train on a dataset?  Compare on-grid and off-grid teacher-forced agreement.

InternNav's System2 SFT takes only frames ``4n`` of every episode as samples
(``internvla_n1_lerobot_dataset.py``, ``--sample_step 4``).  A model that trained on a dataset should do
better on those exact frames than on frames ``4n+2`` of the same episodes, labelled by the same rule
(``scripts/tools/teacher_forced_agreement.py --frame-phase 2``).  A dataset the model certainly trained
on (r2r) is the positive control; the verdict on another dataset (ScaleVLN) is read against it.

Each ``--group NAME=ON_CALLS,OFF_CALLS`` names the two ``*.calls.jsonl`` files of one dataset.  Rows whose
expert answer is STOP, and the first four frames of every episode (frame 0 has no history), are left out,
so both grids hold the same kinds of decisions.  Two metrics, on-grid minus off-grid, with a 95% interval
from resampling whole episodes:

- ``px10``: share of expert pixel targets answered with a pixel within 10 px (primary);
- ``agree``: decision-class agreement (secondary).

    python scripts/tools/memory_fingerprint.py \\
        --group r2r=r2r_p0.calls.jsonl,r2r_p2.calls.jsonl \\
        --group scalevln=b3_p0.calls.jsonl,b3_p2.calls.jsonl --control r2r --output fingerprint.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PX_RADIUS = 10.0
MIN_FRAME = 4
SHARE_OF_CONTROL = 0.20
RESAMPLES = 10000


def load(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("expert_class") != "stop" and int(r["frame_id"]) >= MIN_FRAME]


def _cells(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """(agree, px10 hit, px10 eligible) of one row."""
    pixel = row.get("expert_class") == "pixel"
    near = pixel and row.get("pixel_distance_px") is not None and float(row["pixel_distance_px"]) <= PX_RADIUS
    return float(bool(row["class_match"])), float(near), float(pixel)


def _per_episode(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(4))
    for row in rows:
        agree, hit, eligible = _cells(row)
        out[(row["scene_id"], int(row["episode_index"]))] += (agree, 1.0, hit, eligible)
    return out


def advantage(on: Sequence[Mapping[str, Any]], off: Sequence[Mapping[str, Any]], *, seed: int = 0,
              resamples: int = RESAMPLES) -> dict[str, Any]:
    """On-grid minus off-grid agree and px10, with episode-resampled 95% intervals."""
    a, b = _per_episode(on), _per_episode(off)
    episodes = sorted(set(a) | set(b))
    zero = np.zeros(4)
    A = np.stack([a.get(e, zero) for e in episodes])
    B = np.stack([b.get(e, zero) for e in episodes])

    def rates(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return m[..., 0] / m[..., 1], m[..., 2] / m[..., 3]

    point = [x - y for x, y in zip(rates(A.sum(0)), rates(B.sum(0)))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episodes), size=(resamples, len(episodes)))
    da, dp = [x - y for x, y in zip(rates(A[draws].sum(1)), rates(B[draws].sum(1)))]
    out = {"episodes": len(episodes), "n_on": len(on), "n_off": len(off)}
    for name, p, d, (on_rate, off_rate) in (
        ("agree", point[0], da, (rates(A.sum(0))[0], rates(B.sum(0))[0])),
        ("px10", point[1], dp, (rates(A.sum(0))[1], rates(B.sum(0))[1])),
    ):
        out[name] = {"on": round(float(on_rate), 5), "off": round(float(off_rate), 5), "delta": round(float(p), 5),
                     "ci95": [round(float(np.quantile(d, 0.025)), 5), round(float(np.quantile(d, 0.975)), 5)]}
    return out


def verdict(control: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    """EXP-27's pre-registered reading: the metric is px10 if the control shows it, else agree, else none."""
    metric = next((m for m in ("px10", "agree") if control[m]["ci95"][0] > 0), None)
    if metric is None:
        return {"metric": None, "reading": "undetermined: the positive control shows no on-grid advantage"}
    c, t = control[metric]["delta"], target[metric]
    floor = SHARE_OF_CONTROL * c
    if t["ci95"][0] > 0 and t["delta"] >= floor:
        reading = "seen"
    elif t["ci95"][1] < floor:
        reading = "unseen-compatible"
    else:
        reading = "undetermined"
    return {"metric": metric, "control_delta": c, "floor": round(floor, 5), "reading": reading}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", action="append", required=True, help="NAME=ON_CALLS,OFF_CALLS")
    parser.add_argument("--control", required=True, help="the group the model certainly trained on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    groups = {}
    for spec in args.group:
        name, _, files = spec.partition("=")
        on, _, off = files.partition(",")
        if not (name and on and off):
            raise SystemExit(f"--group takes NAME=ON_CALLS,OFF_CALLS, got {spec!r}")
        groups[name] = advantage(load(on), load(off), seed=args.seed)
    if args.control not in groups:
        raise SystemExit(f"--control {args.control!r} is not a group")
    report = {"px_radius": PX_RADIUS, "min_frame": MIN_FRAME, "share_of_control": SHARE_OF_CONTROL,
              "groups": groups,
              "verdicts": {k: verdict(groups[args.control], v) for k, v in groups.items() if k != args.control}}
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["verdicts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
