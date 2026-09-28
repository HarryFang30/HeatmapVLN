#!/usr/bin/env python3
"""EXP-19: rebuild cases/candidates.json and client episode lists from the ledger's fixed candidate table.

``select_cases.py`` picked the 15 candidates from the two main-table evaluations on
the C500 (their progress.jsonl and client logs).  Those inputs are gone with the
C500, but the selection result is fixed in the ledger (docs/experiments/README.md,
EXP-19 run record (1)): category, rank and the seed-42 / seed-1337 step counts of
every candidate.  This tool writes that table back into the files the later
stages read, so a rerun elsewhere (the RTX 4090 box) follows the same
pre-registered rules:

* ``candidates.json`` (schema ``exp19-candidates-v1``): the 15 candidates by
  category and rank, each checked against ``--dataset`` (the episode exists and its
  scene is the ep_key's) and given its scene, instruction and geodesic distance from
  it.  ``build_records.py`` reads ``ep_key`` and ``rank`` only; the main case of a
  category is still the first candidate, in rank order, whose rerun satisfies the
  category predicate.  ``reconstructed`` says where the table came from.
* ``<lists-dir>/gpu<j>.json``: client ``--episode_list`` files (the cohort format of
  select_cases.py) for ``--episodes`` (default: every category's rank 0, i.e. the
  main cases of the C500 run), sharded like select_cases.py (longest-processing-time
  greedy on the seed-42 steps).  Lists of other ranks for a fallback run go to
  another ``--lists-dir`` with ``--lists-only`` (candidates.json then keeps
  describing the main run's lists).

No eval-log references are written: the C500 client logs they came from are gone,
so build_records.py has no main-table log to compare a rerun with (its
code-equivalence gate then fails and the batch's verdicts are void; the verdicts of
record stay those of the C500 run ``runs/main``).

Usage (stdlib only)::

  python -m scripts.exp19.rebuild_cases --dataset <val_unseen.json.gz> --out-dir <EXP>/cases \\
      [--episodes KEY ...] [--num-gpus N] [--lists-dir DIR] [--scenes-dir <parent of mp3d/>]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp19 import select_cases as sc  # noqa: E402

LEDGER = "docs/experiments/README.md, EXP-19 运行记录（1）"
# (category, rank, ep_key, seed-42 steps, seed-1337 steps), verbatim from the ledger's candidate table.
CANDIDATE_TABLE = (
    ("T1", 0, "X7HyMhZNoso_0601", 64, 59), ("T1", 1, "zsNo4HB9uLZ_1754", 64, 80), ("T1", 2, "TbHJrupSAjP_0472", 65, 64),
    ("T2", 0, "EU6Fwq7SyZv_0346", 57, 57), ("T2", 1, "TbHJrupSAjP_1447", 57, 58), ("T2", 2, "2azQ1b91cZZ_0541", 57, 70),
    ("T3", 0, "zsNo4HB9uLZ_0713", 85, 86), ("T3", 1, "2azQ1b91cZZ_1008", 85, 83), ("T3", 2, "QUCTc6BB5sX_1420", 85, 114),
    ("F1", 0, "zsNo4HB9uLZ_0163", 78, 106), ("F1", 1, "2azQ1b91cZZ_0800", 78, 73), ("F1", 2, "x8F5xyUWy9e_0031", 78, 58),
    ("F2", 0, "x8F5xyUWy9e_0950", 500, 500), ("F2", 1, "pLe4wQe7qrG_1777", 500, 500),
    ("F2", 2, "2azQ1b91cZZ_1432", 500, 500),
)
KEY_RE = re.compile(r"([A-Za-z0-9]+)_(\d{4})")


def split_key(ep_key: str) -> tuple:
    m = KEY_RE.fullmatch(ep_key)
    if not m:
        raise ValueError(f"not an ep_key (<scene>_<4-digit episode id>): {ep_key!r}")
    return m.group(1), int(m.group(2))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: Path) -> dict:
    """(scene stem, episode id) -> dataset episode."""
    with gzip.open(path, "rt") as fh:
        episodes = json.load(fh)["episodes"]
    return {(Path(str(e["scene_id"])).stem, int(e["episode_id"])): e for e in episodes}


def build_candidates(dataset: dict, scenes_dir: Path | None = None) -> tuple:
    """(categories, problems): the ledger table checked against the dataset (and the scene meshes)."""
    categories, problems = {}, []
    for cat in sc.CATEGORY_ORDER:
        categories[cat] = {"name": sc.CATEGORIES[cat]["name"], "candidates": []}
    seen = {}
    for cat, rank, key, steps_42, steps_1337 in CANDIDATE_TABLE:
        scene, episode_id = split_key(key)
        if key in seen:
            problems.append(f"{key} is a candidate of both {seen[key]} and {cat}")
        seen[key] = cat
        ep = dataset.get((scene, episode_id))
        if ep is None:
            problems.append(f"{key}: no episode {episode_id} of scene {scene} in the dataset")
            continue
        if scenes_dir is not None and not (scenes_dir / "mp3d" / scene / f"{scene}.glb").is_file():
            problems.append(f"{key}: scene mesh {scenes_dir / 'mp3d' / scene / (scene + '.glb')} missing")
        categories[cat]["candidates"].append({
            "rank": rank, "category": cat, "ep_key": key, "scene_id": scene, "episode_id": episode_id,
            "steps_42": steps_42, "steps_1337": steps_1337,
            "geodesic_m": float((ep.get("info") or {}).get("geodesic_distance", float("nan"))),
            "instruction": str((ep.get("instruction") or {}).get("instruction_text", "")),
        })
    for cat in sc.CATEGORY_ORDER:
        ranks = [c["rank"] for c in categories[cat]["candidates"]]
        if ranks != list(range(sc.TOP_PER_CATEGORY)):
            problems.append(f"{cat}: ranks {ranks}, expected {list(range(sc.TOP_PER_CATEGORY))}")
    return categories, problems


def episode_lists(cands: list, num_gpus: int, dataset_sha: str, tag: str) -> list:
    shards = sc.shard_lpt(cands, num_gpus)
    return [{"cohort_name": f"exp19_behavior_viz_{tag}_gpu{j}_of_{num_gpus}", "dataset_sha256": dataset_sha,
             "num_shards": num_gpus, "shard_index": j,
             "episodes": [{"scene_id": c["scene_id"], "episode_id": c["episode_id"]} for c in shard]}
            for j, shard in enumerate(shards)]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, required=True, help="R2R val_unseen.json.gz")
    p.add_argument("--out-dir", type=Path, required=True, help="<EXP>/cases")
    p.add_argument("--lists-dir", type=Path, default=None, help="default <out-dir>/episode_lists")
    p.add_argument("--episodes", nargs="*", default=None, help="ep_keys to list (default: every category's rank 0)")
    p.add_argument("--num-gpus", type=int, default=2)
    p.add_argument("--tag", default="rerun", help="cohort-name tag of the episode lists")
    p.add_argument("--scenes-dir", type=Path, default=None, help="parent of mp3d/<scene>/<scene>.glb (checked if given)")
    p.add_argument("--lists-only", action="store_true",
                   help="write only the episode lists (a fallback run's), leave candidates.json as it is")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.num_gpus < 1:
        print("--num-gpus must be >= 1", file=sys.stderr)
        return 2
    dataset_sha = sha256_file(args.dataset)
    categories, problems = build_candidates(load_dataset(args.dataset), args.scenes_dir)
    by_key = {c["ep_key"]: c for cat in sc.CATEGORY_ORDER for c in categories[cat]["candidates"]}
    wanted = args.episodes if args.episodes else [categories[cat]["candidates"][0]["ep_key"]
                                                  for cat in sc.CATEGORY_ORDER if categories[cat]["candidates"]]
    unknown = [k for k in wanted if k not in by_key]
    if unknown:
        problems.append(f"not candidates: {unknown}")
    if len(set(wanted)) != len(wanted):
        problems.append(f"an episode is listed twice: {wanted}")
    if len(wanted) < args.num_gpus:
        problems.append(f"{len(wanted)} episode(s) for {args.num_gpus} GPU lists: every list must be non-empty")
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        print("refusing to write", file=sys.stderr)
        return 1

    lists_dir = args.lists_dir or args.out_dir / "episode_lists"
    lists = episode_lists([by_key[k] for k in wanted], args.num_gpus, dataset_sha, args.tag)
    for stale in sorted(lists_dir.glob("gpu*.json")) if lists_dir.is_dir() else []:
        m = re.fullmatch(r"gpu(\d+)\.json", stale.name)
        if m and int(m.group(1)) >= args.num_gpus:
            stale.unlink()
            print(f"removed stale {stale}")
    for j, lst in enumerate(lists):
        write_json(lists_dir / f"gpu{j}.json", lst)
    if args.lists_only:
        for j, lst in enumerate(lists):
            keys = ["%s_%04d" % (e["scene_id"], int(e["episode_id"])) for e in lst["episodes"]]
            print(f"gpu{j}: {keys}")
        print(f"wrote {len(lists)} episode list(s) in {lists_dir} (candidates.json untouched)")
        return 0

    result = {
        "schema": sc.CANDIDATES_SCHEMA,
        "reconstructed": {
            "tool": "scripts/exp19/rebuild_cases.py",
            "source": LEDGER,
            "why": "the main-table logs select_cases.py read were on the retired C500; the selection it made is "
                   "fixed in the ledger and is copied here, checked against the dataset",
            "not_reconstructed": ["pool sizes and medians (in the ledger)", "eval_log_reference/ (C500 client logs)",
                                  "per-candidate main-table outcomes and reference-path features"],
        },
        "inputs": {str(args.dataset): dataset_sha},
        "constants": {"top_per_category": sc.TOP_PER_CATEGORY, "step_cap": sc.STEP_CAP,
                      "category_order": list(sc.CATEGORY_ORDER)},
        "categories": categories,
        "episode_lists": {f"gpu{j}": [f"{e['scene_id']}_{int(e['episode_id']):04d}" for e in lst["episodes"]]
                          for j, lst in enumerate(lists)},
        "episode_lists_dir": str(lists_dir),
    }
    write_json(args.out_dir / "candidates.json", result)
    for cat in sc.CATEGORY_ORDER:
        print(f"{cat} ({categories[cat]['name']}): " + ", ".join(f"#{c['rank']} {c['ep_key']}"
                                                                 for c in categories[cat]["candidates"]))
    for j, lst in enumerate(lists):
        keys = [f"{e['scene_id']}_{int(e['episode_id']):04d}" for e in lst["episodes"]]
        print(f"gpu{j}: {keys} (steps_42 {sum(by_key[k]['steps_42'] for k in keys)})")
    print(f"wrote {args.out_dir / 'candidates.json'} and {len(lists)} episode list(s) in {lists_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
