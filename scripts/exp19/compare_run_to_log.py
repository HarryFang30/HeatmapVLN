#!/usr/bin/env python3
"""EXP-19: compare a traced rerun with a deployed run's client log, call by call.

On the C500 the rerun was checked against the main-table evaluation's client log
(build_records.py: the code-equivalence gate = call 0's System2 text and action chunk
verbatim, fidelity = the first divergent call and whether the outcome is the same).
This tool runs the same comparison against ANY client log of the deployed, untraced
launcher (e.g. the RTX 4090 box's canary, scripts/run_ppa_r2r_val_unseen_cuda.sh) on
the same platform, for every episode the run and the log share:

* rerun side: ``runs/<run>/gpu*/trace/<ep>/call_*.json`` (System2 text, response action
  chunk, call step) and ``steps/<ep>/steps.jsonl`` ``episode_end`` (success, oracle
  success, steps, how it ended);
* log side: ``select_cases.parse_client_log`` (the selection scout's parser) and, when
  given, the log run's ``progress.json`` rows (steps; how it ended is read from the log
  as select_cases.py does);
* ``build_records.compare_calls`` / ``compare_outcome``, unchanged.

It also reports the rerun's own trace neutrality (build_records' gate: every trajectory
call's offline-recomputed action chunk equals the response chunk, ``actions_match``).
The run must be finished (``runs/<run>/DONE`` with status ``complete``).

``--require-call0`` makes the exit code 1 unless every compared episode's call 0 is
identical (the gate); ``--require-neutral`` unless every trajectory call of the run has
``actions_match`` true; ``--require-identical`` unless every call and the outcome are
identical (needs ``--progress``).  Writes ``--out`` (JSON) and prints one line per episode.

Usage (stdlib + numpy, no torch)::

  python -m scripts.exp19.compare_run_to_log --run-dir <EXP>/runs/<run> --client-log <log> \\
      [--progress <progress.json>] [--out <json>] [--require-call0] [--require-neutral] [--require-identical]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp19 import build_records as br  # noqa: E402
from scripts.exp19 import select_cases as sc  # noqa: E402

SCHEMA = "exp19-run-vs-log-v1"


def rerun_calls(trace_dir: Path) -> list:
    out = []
    for c in br.load_trace(trace_dir)["calls"]:
        resp = c.get("response") or {}
        out.append({"call_index": int(c["system2_call_index"]), "step": int(c["current_capture_step"]),
                    "kind": resp.get("kind"), "vlm_output": resp.get("llm_output"),
                    "actions": [int(a) for a in (resp.get("actions") or [])],
                    "actions_match": c.get("actions_match")})
    return out


def neutrality(calls: list) -> dict:
    traj = [c for c in calls if c["kind"] == "trajectory"]
    bad = [c["call_index"] for c in traj if c["actions_match"] is not True]
    return {"n_trajectory_calls": len(traj), "n_match": len(traj) - len(bad), "not_matching_calls": bad,
            "pass": bool(traj) and not bad}


def rerun_outcome(steps_dir: Path) -> dict | None:
    end = br.load_steps(steps_dir)["end"]
    if end is None:
        return None
    m = end.get("metrics") or {}
    return {"success": float(m.get("success", float("nan"))), "oracle_success": float(m.get("oracle_success", float("nan"))),
            "ne_m": float(m.get("distance_to_goal", float("nan"))), "steps": int(end["steps"]),
            "ended_by": str(end.get("ended_by") or "unknown")}


def load_progress(path: Path | None) -> dict:
    rows = {}
    if path is None:
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "episode_id" in r:
                rows[(Path(str(r["scene_id"])).stem, int(r["episode_id"]))] = r
    return rows


def compare_episode(ep_key: str, paths: dict, block: dict, progress_row: dict | None) -> dict:
    calls = rerun_calls(Path(paths["trace_dir"]))
    ref_calls = block["calls"]
    neutral = neutrality(calls)
    fidelity = br.compare_calls(calls, ref_calls)
    a = next((c for c in calls if c["call_index"] == 0), None)
    b = next((c for c in ref_calls if int(c["call_index"]) == 0), None)
    call0 = (a is not None and b is not None
             and str(a["vlm_output"] or "").strip() == str(b.get("vlm_output") or "").strip()
             and a["actions"] == [int(x) for x in (b.get("actions") or [])])
    final = dict(block["final"] or {})
    if progress_row is not None:
        final["steps"] = int(progress_row["steps"])
        final["ended_by"] = sc.ended_by(block, int(progress_row["steps"]))
    outcome = rerun_outcome(Path(paths["steps_dir"]))
    outcome_cmp = br.compare_outcome(outcome, final) if final else None
    first = fidelity["first_divergent_call"]
    return {
        "ep_key": ep_key, "call0_identical": bool(call0), "trace_neutrality": neutral,
        "calls": {k: fidelity[k] for k in ("first_divergent_call", "identical_calls", "identical_prefix_calls",
                                             "total_calls", "reference_calls", "all_identical")},
        "first_divergence": None if first is None else {
            "rerun": next((c for c in calls if c["call_index"] == first), None),
            "log": next((c for c in ref_calls if int(c["call_index"]) == first), None)},
        "outcome": outcome, "log_final": final or None, "outcome_comparison": outcome_cmp,
        "log": {"client_log": block["client_log"], "line_start": block["line_start"], "line_end": block["line_end"]},
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--client-log", type=Path, required=True, help="client log of the deployed (untraced) run")
    p.add_argument("--progress", type=Path, default=None, help="that run's progress.json (JSONL)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--require-call0", action="store_true")
    p.add_argument("--require-neutral", action="store_true")
    p.add_argument("--require-identical", action="store_true")
    args = p.parse_args(argv)
    if args.require_identical and args.progress is None:
        p.error("--require-identical needs --progress (the log's step count comes from progress.json)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        status = json.loads((args.run_dir / "DONE").read_text()).get("status")
    except (OSError, ValueError):
        status = None
    if status != "complete":
        print(f"{args.run_dir}/DONE is missing or not 'complete' (status {status!r}): the run is not finished",
              file=sys.stderr)
        return 1
    found = br.discover_run(args.run_dir)
    blocks, dup, incomplete = sc.parse_client_log(args.client_log)
    progress = load_progress(args.progress)
    episodes = []
    for ep_key, paths in sorted(found.items()):
        start = br.load_steps(Path(paths["steps_dir"]))["start"]
        scene, episode_id = str(start["scene_id"]).split("/")[-1].replace(".glb", ""), int(start["episode_id"])
        block = blocks.get(episode_id)
        if block is None or block["scene_id"] != scene:
            print(f"{ep_key}: not in {args.client_log}")
            continue
        res = compare_episode(ep_key, paths, block, progress.get((scene, episode_id)))
        episodes.append(res)
        oc = res["outcome_comparison"] or {}
        tn = res["trace_neutrality"]
        print(f"{ep_key}: call0 {'identical' if res['call0_identical'] else 'DIFFERS'}; trace neutrality "
              f"{tn['n_match']}/{tn['n_trajectory_calls']}; "
              f"{res['calls']['identical_calls']}/{res['calls']['total_calls']} calls identical "
              f"(log {res['calls']['reference_calls']}), first divergent call {res['calls']['first_divergent_call']}; "
              f"outcome {'same' if oc.get('same') else 'DIFFERENT'} (steps {oc.get('rerun', {}).get('steps')} vs "
              f"{oc.get('eval_log', {}).get('steps')})")
    record = {"schema": SCHEMA, "run_dir": str(args.run_dir), "client_log": str(args.client_log),
              "progress": str(args.progress) if args.progress else None,
              "log_blocks": {"complete": len(blocks), "duplicate": dup, "incomplete": incomplete},
              "n_compared": len(episodes),
              "all_call0_identical": bool(episodes) and all(e["call0_identical"] for e in episodes),
              "all_trace_neutral": bool(episodes) and all(e["trace_neutrality"]["pass"] for e in episodes),
              "all_identical": bool(episodes) and all(e["calls"]["all_identical"]
                                                      and (e["outcome_comparison"] or {}).get("same")
                                                      and (e["outcome_comparison"] or {}).get("steps_equal")
                                                      for e in episodes),
              "episodes": episodes}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not episodes:
        print("no episode of the run is in the log", file=sys.stderr)
        return 1
    if args.require_call0 and not record["all_call0_identical"]:
        return 1
    if args.require_neutral and not record["all_trace_neutral"]:
        return 1
    if args.require_identical and not record["all_identical"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
