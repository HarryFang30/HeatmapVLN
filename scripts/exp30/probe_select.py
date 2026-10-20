#!/usr/bin/env python3
"""EXP-30 probe 1, step 1 (4090, CPU): pick the states and their donor instructions.

Targets: EXP-28 training-cohort episodes the base model failed under greedy decoding whose first System2 call
(call 0) line B judged a correction.  Controls (report only): a seeded sample of successful episodes whose call 0
line B judged an anchor.  Each state gets a donor instruction: another episode of the same scene (seeded).
Writes ``states.jsonl`` (one row per state) and ``push_files.txt`` (record tars and their .sha256, relative to the
record root's parent) for ``stream_transfer.py push --files``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import tarfile
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", default="/workspace/exp28/collect_base/records")
    ap.add_argument("--labels", default="/workspace/exp28/labels_base")
    ap.add_argument("--first-train-shard", type=int, default=20, help="shards below are the held-out part")
    ap.add_argument("--controls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    call0 = {}
    for path in glob.glob(os.path.join(args.labels, "*", "*.s2labels.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if int(row["call_index"]) == 0:
                    call0[(row["scene_id"], int(row["episode_id"]))] = row
    episodes = []
    for sha in sorted(glob.glob(os.path.join(args.records, "*", "*.s2rec.tar.sha256"))):
        tar = sha[: -len(".sha256")]
        with tarfile.open(tar) as t:
            ep = json.load(t.extractfile("episode.json"))
        shard = int(re.search(r"shard_(\d+)\.json", ep["episode_list"]).group(1))
        if shard < args.first_train_shard:
            continue
        key = (ep["scene_id"], int(ep["episode_id"]))
        label = call0.get(key)
        episodes.append({"key": key, "tar": tar, "instruction": ep.get("instruction") or ep.get("instruction_raw"),
                         "success": bool(ep["outcome"].get("success")), "kind0": label["kind"] if label else None})
    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["key"][0]].append(ep)
    rng = random.Random(args.seed)
    targets = [e for e in episodes if not e["success"] and e["kind0"] == "correction"]
    pool = [e for e in episodes if e["success"] and e["kind0"] == "anchor"]
    controls = rng.sample(pool, min(args.controls, len(pool)))
    os.makedirs(args.out, exist_ok=True)
    root_parent = os.path.dirname(os.path.abspath(args.records))
    files = []
    with open(os.path.join(args.out, "states.jsonl"), "w", encoding="utf-8") as handle:
        for group, items in (("target", targets), ("control", controls)):
            for e in sorted(items, key=lambda x: x["key"]):
                donors = [d for d in by_scene[e["key"][0]] if d["key"] != e["key"] and d["instruction"] != e["instruction"]]
                donor = rng.choice(donors) if donors else None
                rel = os.path.relpath(e["tar"], root_parent)
                handle.write(json.dumps({
                    "key": f"{e['key'][0]}_{e['key'][1]}", "group": group, "record": rel, "call_index": 0,
                    "donor_key": f"{donor['key'][0]}_{donor['key'][1]}" if donor else None,
                    "donor_instruction": donor["instruction"] if donor else None}, ensure_ascii=False) + "\n")
                files += [rel, rel + ".sha256"]
    with open(os.path.join(args.out, "push_files.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(set(files))) + "\n")
    print(json.dumps({"targets": len(targets), "controls": len(controls), "control_pool": len(pool),
                      "files": len(set(files)), "no_donor": sum(1 for e in targets + controls if not any(
                          d["key"] != e["key"] and d["instruction"] != e["instruction"] for d in by_scene[e["key"][0]]))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
