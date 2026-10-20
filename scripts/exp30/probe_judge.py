#!/usr/bin/env python3
"""EXP-30 probe 1, step 3 (4090, CPU): judge every decoded first-call answer with line B and read the probe.

Line B is O4's ``judge`` on ``call_contexts`` for call 0 (``RouteLabelConfig()`` defaults, the EXP-28 navmeshes, the
episode pool).  Before anything is read, line B must reproduce the label file's kind and agreement on the recorded
answer for every state.  Readings (ledger EXP-30): pass@16, majority-correct (>= 9 of 16), and the share of states whose
greedy answer is the same under the donor instruction (same class; pixels within 50 px on both axes; turns the same
way).  Prints the verdict line first and writes ``<out>/probe1.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _pixel(text):
    import re

    m = re.findall(r"-?\d+", str(text))
    return (int(m[0]), int(m[1])) if len(m) >= 2 else None


def same_answer(a, b) -> bool:
    from scripts.tools.stratified_gates import decision_class

    if not a or not b:
        return False
    ca, cb = decision_class(a[-1]), decision_class(b[-1])
    if ca != cb:
        return False
    if ca == "pixel":
        pa, pb = _pixel(a[-1]), _pixel(b[-1])
        return pa is not None and pb is not None and abs(pa[0] - pb[0]) <= 50 and abs(pa[1] - pb[1]) <= 50
    if ca in ("turn_left", "turn_right"):
        return True  # same class = same direction
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--decoded", required=True, help="directory of decode.shard*.jsonl")
    ap.add_argument("--record-parent", default="/workspace/exp28/collect_base")
    ap.add_argument("--labels", default="/workspace/exp28/labels_base")
    ap.add_argument("--scenes-dir", default="/workspace/exp28/scene_datasets/hm3d")
    ap.add_argument("--episodes", default="/workspace/exp28/episodes/pool.json.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import scripts.exp28.hm3d_route_gate as G
    from src.data import s2_records as REC

    states = {}
    for line in Path(args.states).read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            states[s["key"]] = s
    decoded = {}
    for path in sorted(glob.glob(os.path.join(args.decoded, "decode.shard*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                decoded[row["key"]] = row
    missing = sorted(set(states) - set(decoded))
    config = G.RL.RouteLabelConfig()
    dataset_index = G.lr.base.load_episode_index([args.episodes])
    contexts_for = G.navmesh_contexts(args.scenes_dir, dataset_index, config)

    per_group = defaultdict(list)
    mismatches = []
    for key, state in sorted(states.items()):
        row = decoded.get(key)
        if row is None:
            continue
        record_path = str(Path(args.record_parent) / state["record"])
        record = REC.EpisodeRecord(record_path)
        context = contexts_for(record, [0])[0]
        label = next(r for r in REC.read_labels(REC.label_path_for(Path(record_path), Path(args.labels)))
                     if int(r["call_index"]) == 0)
        recorded = G._turns(label.get("model_turn1"), label.get("model_turn2"))
        check = G.judge(context, recorded)
        if (check["kind"], check["agree"]) != (label["kind"], label["agree"]):
            mismatches.append(f"{key}: label {label['kind']}/{label['agree']} rejudged {check['kind']}/{check['agree']}")
        greedy = G.judge(context, row["greedy_own"])
        agrees = [bool(G.judge(context, s)["agree"]) for s in row["samples"]]
        per_group[state["group"]].append({
            "key": key, "greedy_agree": bool(greedy["agree"]), "sample_agree": sum(agrees), "samples": len(agrees),
            "same_under_shuffle": same_answer(row["greedy_own"], row.get("greedy_shuf")) if row.get("greedy_shuf") else None})

    def read(rows):
        n = len(rows)
        if not n:
            return {"n": 0}
        shuf = [r["same_under_shuffle"] for r in rows if r["same_under_shuffle"] is not None]
        return {"n": n,
                "greedy_agree": sum(r["greedy_agree"] for r in rows) / n,
                "pass_at_16": sum(r["sample_agree"] >= 1 for r in rows) / n,
                "majority_correct": sum(r["sample_agree"] >= 9 for r in rows) / n,
                "mean_sample_agree_rate": sum(r["sample_agree"] / max(r["samples"], 1) for r in rows) / n,
                "same_under_shuffle": (sum(shuf) / len(shuf)) if shuf else None, "shuffle_n": len(shuf)}

    result = {"schema": "heatmapvln-exp30-probe1-v1", "missing_states": len(missing), "label_mismatches": len(mismatches),
              "mismatch_examples": mismatches[:5], "target": read(per_group["target"]), "control": read(per_group["control"])}
    t = result["target"]
    if mismatches or missing or not t.get("n"):
        verdict = "INVALID (line B does not reproduce the labels, or states are missing)"
    elif t["pass_at_16"] >= 0.50 and t["majority_correct"] >= 0.25:
        verdict = "IN THE MODEL: decoding / calibration problem"
    elif t["pass_at_16"] <= 0.15:
        verdict = "KNOWLEDGE CEILING at T 0.7"
    else:
        verdict = "IN BETWEEN: neither ruled out"
    language = ""
    if t.get("same_under_shuffle") is not None:
        language = " | first-call errors ignore the instruction" if t["same_under_shuffle"] >= 0.80 else ""
    result["verdict"] = verdict
    os.makedirs(args.out, exist_ok=True)
    Path(args.out, "probe1.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"{verdict}{language} " + json.dumps({"target": t, "control": result["control"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
