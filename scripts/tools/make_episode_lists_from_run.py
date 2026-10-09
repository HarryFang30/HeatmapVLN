#!/usr/bin/env python3
"""Pin a canary to the exact episodes a finished reference run recorded, as per-shard ``--episode_list`` files.

Why: a capped canary (``PPA_EVAL_MAX_EPISODES_PER_SHARD``) is not a fixed sample.  The client's cap counts only
episodes not yet in ``progress.json`` (``_eval_limit`` in scripts/evaluation/r2r_val_unseen.py) and the launcher
passes the same cap again on every restart, so a client that dies once runs that many *more* new episodes in the
shard and the canary silently stops being the pre-registered episodes
(docs/ops/ascend_910b_open_problems.md section 2.2).  With an explicit list and no cap, "pending" is bounded by
the list itself and a restart can only finish what the list names.

Reads the run layout the CUDA launcher writes (scripts/run_ppa_r2r_val_unseen_cuda.sh, also what
scripts/exp20/canary_check.py reads):
  <reference>/workers/shard_0N/progress.json            one JSON row per finished episode
  <reference>/runtime/<stamp>/logs/client_shard_0N.log  the client's stdout, appended across restarts

and writes ``<out>/shard_0N.json`` in the cohort file's own schema (the client reads only ``episodes``), with the
episodes in cohort order and every entry copied from the cohort, plus ``<out>/manifest.json`` with the sha256 of
every input it read.  It refuses, rather than writing a quietly wrong list, when the reference was itself resumed,
restarted or relaunched, holds a different number of episodes than expected, names an episode outside the cohort,
or when any output file already exists.

  python3 scripts/tools/make_episode_lists_from_run.py \\
    --reference /workspace/eval_runs/canary_cuda_seed42 \\
    --cohorts <LOCKED_PLAN>/cohorts --shards 0,1 --expect-per-shard 2 \\
    --out /workspace/eval_runs/ascend_canary_lists
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MANIFEST_SCHEMA = "pinned-episode-lists-from-run-v1"
NUM_SHARDS = 8
# The client prints this once per start, after reading progress.json (r2r_val_unseen.py, "Episodes already done").
# The launcher appends every restart to the same log, so a restarted shard shows the line more than once.
_DONE_LINE = re.compile(r"Episodes already done: (\d+)\b")

Key = Tuple[str, int]


class Refusal(Exception):
    """The inputs cannot be turned into a list that is certainly the reference's sample."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shards(text: str) -> List[int]:
    try:
        shards = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise Refusal(f"--shards must be comma-separated shard indices, got {text!r}") from None
    if not shards or len(set(shards)) != len(shards) or any(not 0 <= s < NUM_SHARDS for s in shards):
        raise Refusal(f"--shards must name distinct shards in 0..{NUM_SHARDS - 1}, got {text!r}")
    return sorted(shards)


def runtime_dir(reference: Path) -> Path:
    """The run's single launch stamp directory.

    Each launch of the launcher makes a new runtime/<stamp>/ but reuses workers/, and --resume makes the second
    launch skip what the first recorded.  So two stamps mean the progress rows may come from two launches with
    different caps, and no single client log can vouch for all of them.
    """
    root = reference / "runtime"
    if not root.is_dir():
        raise Refusal(f"reference has no runtime/ directory: {root}")
    stamps = sorted(p for p in root.iterdir() if p.is_dir())
    if len(stamps) != 1:
        names = ", ".join(p.name for p in stamps) or "none"
        raise Refusal(f"reference runtime/ must hold exactly one launch stamp, found {len(stamps)} ({names}): {root}")
    return stamps[0]


def check_client_log(log: Path) -> None:
    """The shard's client started once, from an empty progress file.

    "Episodes already done: 0" on the only start is what rules out a reference that was itself resumed into,
    which is the failure this tool exists to keep out of the pinned list.
    """
    if not log.is_file():
        raise Refusal(f"reference client log is missing: {log}")
    starts = [int(m.group(1)) for m in _DONE_LINE.finditer(log.read_text(encoding="utf-8", errors="replace"))]
    if not starts:
        raise Refusal(f"client log never reports 'Episodes already done', so the client never started: {log}")
    if len(starts) > 1:
        raise Refusal(f"client log shows {len(starts)} client starts (restarted within the launch, "
                      f"'Episodes already done' = {starts}): {log}")
    if starts[0] != 0:
        raise Refusal(f"client log shows a resumed start ('Episodes already done: {starts[0]}', expected 0): {log}")


def read_progress(progress: Path, expect: int) -> List[Key]:
    """(scene_id, episode_id) of each recorded row, in the order recorded.

    Counting raw rows as well as unique keys: the client's own loader keeps the last row per key, so a duplicate
    would vanish from its view while still meaning two runs of one episode.
    """
    if not progress.is_file():
        raise Refusal(f"reference progress file is missing: {progress}")
    keys: List[Key] = []
    for number, line in enumerate(progress.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            # A client killed mid-write leaves a torn last row: the episode it describes did not finish cleanly.
            raise Refusal(f"progress row {number} is not valid JSON: {progress}") from None
        if row.get("scene_id") in (None, "") or row.get("episode_id") is None:
            raise Refusal(f"progress row {number} has no scene_id/episode_id: {progress}")
        keys.append((str(row["scene_id"]), int(row["episode_id"])))
    unique = set(keys)
    if len(keys) != expect or len(unique) != expect:
        raise Refusal(f"reference progress holds {len(keys)} rows / {len(unique)} unique episodes, "
                      f"expected exactly {expect}: {progress}")
    return keys


def load_cohort(path: Path) -> Tuple[dict, list]:
    """(metadata, episode entries) of a cohort shard file; the client accepts a bare list or {"episodes": [...]}."""
    if not path.is_file():
        raise Refusal(f"cohort file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        metadata, entries = {}, data
    elif isinstance(data, dict) and isinstance(data.get("episodes"), list):
        metadata, entries = {k: v for k, v in data.items() if k != "episodes"}, data["episodes"]
    else:
        raise Refusal(f"cohort file is neither a list nor a dict with an 'episodes' list: {path}")
    keys = [(str(e["scene_id"]), int(e["episode_id"])) for e in entries]
    if len(set(keys)) != len(keys):
        raise Refusal(f"cohort file names an episode more than once, so its entry is ambiguous: {path}")
    return metadata, entries


def git_commit(path: Path) -> Dict[str, object]:
    """The commit this tool ran from, if it runs from a git checkout (the 4090 runs from src_<sha> copies)."""
    here = path.resolve().parent
    # The repo root is two levels up from scripts/tools, but the tool is also run as a
    # copy from somewhere shallow (/tmp): there, parents[1] does not exist, and raising
    # IndexError here would abort a pin that has already written its lists.
    safe = here.parents[1] if len(here.parents) > 1 else here
    git = ["git", "-c", f"safe.directory={safe}", "-C", str(here)]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(git + ["status", "--porcelain", "--", path.name], capture_output=True, text=True,
                               check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tool_modified": None}
    return {"commit": commit.strip(), "tool_modified": bool(dirty.strip())}


def pin(reference: Path, cohorts: Path, shards: List[int], expect: int, out: Path) -> dict:
    if expect < 1:
        raise Refusal(f"--expect-per-shard must be positive, got {expect}")
    reference = reference.resolve()
    cohorts = cohorts.resolve()
    targets = [out / f"shard_0{s}.json" for s in shards] + [out / "manifest.json"]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise Refusal("refusing to overwrite a pinned list: " + ", ".join(existing))

    runtime = runtime_dir(reference)
    payloads: Dict[int, dict] = {}
    record: List[dict] = []
    for shard in shards:
        progress = reference / "workers" / f"shard_0{shard}" / "progress.json"
        log = runtime / "logs" / f"client_shard_0{shard}.log"
        cohort_path = cohorts / f"shard_0{shard}.json"
        check_client_log(log)
        recorded = set(read_progress(progress, expect))
        metadata, entries = load_cohort(cohort_path)
        kept = [e for e in entries if (str(e["scene_id"]), int(e["episode_id"])) in recorded]
        outside = sorted(recorded - {(str(e["scene_id"]), int(e["episode_id"])) for e in kept})
        if outside:
            raise Refusal(f"shard {shard}: reference episodes not in the cohort file {cohort_path}: {outside}")
        parent = metadata.get("cohort_name")
        payloads[shard] = {
            **metadata,
            "cohort_name": f"{parent or f'shard_0{shard}'}_pinned_{reference.name}",
            "parent_cohort": parent,
            "pinned_from_reference": str(reference),
            "episodes": kept,
        }
        record.append({
            "shard": shard,
            "progress": {"path": str(progress), "sha256": sha256_file(progress)},
            "client_log": {"path": str(log), "sha256": sha256_file(log)},
            "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
            "episodes": [{"scene_id": str(e["scene_id"]), "episode_id": int(e["episode_id"])} for e in kept],
        })

    # Everything that could still fail happens before the first file is written, so a
    # failure leaves no half-pinned directory for someone to run with.
    tool = Path(__file__)
    tool_record = {"path": str(tool.resolve()), "sha256": sha256_file(tool), **git_commit(tool)}

    out.mkdir(parents=True, exist_ok=True)
    for entry in record:
        path = out / f"shard_0{entry['shard']}.json"
        # "x": a file that appeared since the check above is still never overwritten.
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(payloads[entry["shard"]], indent=2) + "\n")
        entry["output"] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "reference": str(reference),
        "runtime_stamp": runtime.name,
        "cohorts": str(cohorts),
        "expect_per_shard": expect,
        "shards": record,
        "tool": tool_record,
    }
    with open(out / "manifest.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, type=Path, help="finished run root (holds workers/ and runtime/)")
    ap.add_argument("--cohorts", required=True, type=Path, help="locked plan's cohorts/ directory")
    ap.add_argument("--shards", required=True, help="comma-separated shard indices, e.g. 0,1")
    ap.add_argument("--expect-per-shard", required=True, type=int,
                    help="exact number of episodes each shard of the reference must hold")
    ap.add_argument("--out", required=True, type=Path, help="directory for shard_0N.json and manifest.json")
    args = ap.parse_args(argv)
    try:
        manifest = pin(args.reference, args.cohorts, parse_shards(args.shards), args.expect_per_shard, args.out)
    except Refusal as exc:
        print(f"[make-episode-lists] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "runtime_stamp": manifest["runtime_stamp"],
                      "episodes": {f"shard_0{s['shard']}": len(s["episodes"]) for s in manifest["shards"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
