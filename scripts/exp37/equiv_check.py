#!/usr/bin/env python3
"""EXP-37's zero-adapter equivalence check (ledger EXP-37 设置·等价检查): before the val, the new server mode
``base-stop-v1`` serving a zero adapter (every ``lora_B`` tensor zeroed) must give, call by call on 3 episodes,
the answers B0's servers gave.

With ``lora_B = 0`` the adapter's decode is the base's, and the base decides STOP from its own first token,
so the mode can differ from B0 only through what it adds: the extra base pass before the adapter's, the
STOP suppression at the adapter's first position (a no-op whenever the base did not say STOP, since the
first token is then not STOP either), and the bookkeeping.  Any difference is a bug or a numerics effect to
explain before the val (the ledger: 不同就先查清原因，再上 val).

The reference is B0 seed 1337 itself (``/workspace/eval_runs/exp34_B0_seed1337_910b_stepbudget``, servers
``exp28x34b0w2``, code e941a3e, slot s serving shard s).  Its per-call record is the client log
``runtime/<stamp>/logs/client_shard_0N.log``: one ``step_id: S, RPC kind=K, VLM output: O`` line per System2
call (O is the pixel goal ``x y``, arrows or ``STOP``), then the executed actions (and System1's
``traj_goal`` on a trajectory call), the ``[amb3r-vo]`` line before it, and the episode's ``=> success`` line;
``scripts/exp19/select_cases.parse_client_logs`` reads them (the parser EXP-32's check H used).  The progress
rows (``workers/shard_0N/progress.json``) give the outcome.

The 3 episodes are B0's first three of shard 0 (episodes 1, 25 and 41 of zsNo4HB9uLZ; EXP-32 H's three):
they ran first in B0's first client process on slot 0, so a 3-episode run on slot 0 of a fresh set
replays B0's order exactly.  All three hold pixel, arrow and STOP calls (16 + 18 + 46 = 80 calls), and
episode 25 is a base STOP 9.1 m from the goal, the case the mode exists for.  ``pick`` prints them.

Subcommands::

    pick     the first --count complete episodes of --shard in the reference's first client process, with
             their calls by kind; exit 1 unless every one has a STOP and a pixel (trajectory) call.
    lists    <out>/shard_0N.json, the cohort file with only --episodes (the launcher's episode lists).
    command  the exact lines that pull a server set's publication and run the 3 episodes against it; its
             compare line writes --equiv-json (default /workspace/exp37/equiv/equiv.json, the file
             scripts/exp37/val_4090.sh waits on; keep the two in sync).
    compare  per episode and call: step, kind, VLM output, actions, System1's traj_goal and the VO line
             before the call, then the outcome (the ``=> success`` line and the progress row).  Exit 0: all
             --episodes identical; 1: a difference (the first one is printed); 2: not measured (an episode
             missing from either run, a run's progress rows unreadable or repeated, the candidate rows /
             servers.json not in base-stop mode, or the reference rows not B0's base model).

    python scripts/exp37/equiv_check.py pick --reference /workspace/eval_runs/exp34_B0_seed1337_910b_stepbudget
    python scripts/exp37/equiv_check.py lists --episodes 1,25,41 --out /workspace/exp37/equiv/lists
    python scripts/exp37/equiv_check.py command --tag x37z --model-port-base 52410 --vo-port-base 52510 \\
        --sha <zero adapter sha256> --lists /workspace/exp37/equiv/lists --output /workspace/exp37/equiv/run_x37z
    python scripts/exp37/equiv_check.py compare --reference /workspace/eval_runs/exp34_B0_seed1337_910b_stepbudget \\
        --candidate /workspace/exp37/equiv/run_x37z --episodes 1,25,41 --output /workspace/exp37/equiv/equiv.json

stdlib only.  Run from a checkout of this repo, or anywhere with EXP37_PARSER_REPO naming a checkout that has
``scripts/exp19/select_cases.py`` (default /workspace/exp23/wt_x32, whose copy is byte-identical).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _root in (str(REPO_ROOT), os.environ.get("EXP37_PARSER_REPO", "/workspace/exp23/wt_x32")):
    if _root not in sys.path:
        sys.path.append(_root)

from scripts.exp19.select_cases import parse_client_log, parse_client_logs  # noqa: E402

B0_RUN = "/workspace/eval_runs/exp34_B0_seed1337_910b_stepbudget"
PLAN = "/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802"
WT = "/workspace/exp23/wt_x32"
EPISODES = "1,25,41"
EQUIV_JSON = "/workspace/exp37/equiv/equiv.json"  # scripts/exp37/val_4090.sh's default X37_EQUIV_JSON
MODE_SUFFIX = "+base-stop-v1"
B0_MODEL_VERSION = "ppa-stage2-online-amb3r:zhr_1"
CALL_FIELDS = ("step", "kind", "vlm_output", "actions", "traj_goal",
               "vo_frame", "vo_history", "vo_ready", "vo_phase", "vo_revision")
ROW_FIELDS = ("success", "spl", "os", "ne", "steps", "vlm_calls", "trajectory_calls", "step_budget_stop")


def log_glob(run: str) -> str:
    return f"{run}/runtime/*/logs/client_shard_*.log"


def episode_ids(text: str) -> list[int]:
    ids = [int(x) for x in text.split(",") if x.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit(f"--episodes {text!r}: distinct episode ids, comma-separated")
    return ids


class RowsUnreadable(ValueError):
    """A run's progress rows from which no outcome may be read (compare: not measured, never a difference)."""


def progress_rows(run: str) -> dict[int, dict]:
    """Every shard's progress row by episode id (R2R val_unseen ids are unique); a repeated id or a line that is
    no row is refused (``RowsUnreadable``)."""
    rows: dict[int, dict] = {}
    for path in sorted(glob.glob(f"{run}/workers/shard_0*/progress.json")):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = int(row["episode_id"])
            except (ValueError, KeyError, TypeError) as err:
                raise RowsUnreadable(f"{path}: a line that is no progress row ({type(err).__name__}: {err})") from err
            if key in rows:
                raise RowsUnreadable(f"{path}: episode {key} has two progress rows")
            rows[key] = row
    return rows


def first_process_order(path: str) -> list[int]:
    """Episode ids in the order the log's first client process started them (up to its next start line)."""
    order: list[int] = []
    starts = 0
    with open(path, errors="replace", newline="\n") as handle:
        for physical in handle:
            for line in physical.rstrip("\n").split("\r"):
                if line.startswith("Episodes already done:"):
                    starts += 1
                    if starts > 1:
                        return order
                    continue
                if line.startswith("[") and "] Episode " in line:
                    key = line.split("] Episode ", 1)[1].split(":", 1)[0]
                    order.append(int(key.rsplit("_", 1)[1]))
    return order


def cmd_pick(args) -> int:
    logs = sorted(glob.glob(f"{args.reference}/runtime/*/logs/client_shard_0{args.shard}.log"))
    if not logs:
        raise SystemExit(f"no client_shard_0{args.shard}.log under {args.reference}/runtime")
    blocks, _, _ = parse_client_log(logs[0])
    order = first_process_order(logs[0])[: args.count]
    ok = len(order) == args.count
    print(f"{logs[0]}: the first {args.count} episodes of its first client process")
    for episode in order:
        block = blocks.get(episode)
        if block is None:
            print(f"  episode {episode}: no complete block"); ok = False; continue
        kinds = collections.Counter(c["kind"] for c in block["calls"])
        good = kinds["stop"] > 0 and kinds["trajectory"] > 0
        ok &= good
        print(f"  {block['scene_id']}_{episode:04d}: {len(block['calls'])} calls {dict(kinds)} final {block['final']}"
              + ("" if good else "  (no STOP or no pixel call)"))
    print(f"--episodes {','.join(map(str, order))}" if ok else "FAIL: not every episode has a STOP and a pixel call")
    return 0 if ok else 1


def cmd_lists(args) -> int:
    ids = episode_ids(args.episodes)
    doc = json.loads((Path(args.cohorts) / f"shard_0{args.shard}.json").read_text())
    keep = [e for e in doc["episodes"] if int(e["episode_id"]) in ids]
    missing = sorted(set(ids) - {int(e["episode_id"]) for e in keep})
    if missing:
        raise SystemExit(f"episodes {missing} are not in cohort shard {args.shard}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard_0{args.shard}.json").write_text(json.dumps({**doc, "episodes": keep}, indent=1))
    print(f"{out}/shard_0{args.shard}.json: {len(keep)} episodes; run with PPA_EVAL_SHARDS={args.shard}")
    return 0


def cmd_command(args) -> int:
    if Path(args.output).exists():  # the launcher would --resume into it, and compare would read the old run too
        raise SystemExit(f"{args.output} exists; give the run a new --output")
    episode_ids(args.episodes)
    servers = f"{args.servers_root}/npu_servers_exp28_{args.tag}"
    q = shlex.quote
    print(f"mkdir -p {q(args.servers_root)} {q(str(Path(args.output).parent))}")
    print(f"/opt/conda/bin/python {WT}/scripts/exp23/pull_dirs.py {q(args.servers_root)} 52418 npu_servers_exp28_{args.tag}")
    print(f"python3 -c \"import json,sys;d=json.load(open(sys.argv[1]));"
          f"assert d.get('system2_base_stop') is True and d['model_version'].endswith('{MODE_SUFFIX}') "
          f"and d['system2_lora_sha256']==sys.argv[2], d\" {q(servers)}/servers.json {q(args.sha)}")
    # B0's launch (exp34/auto/auto4090.sh start_b0_resume), on one slot and the 3 episodes; no retries, so a
    # crash is seen rather than resumed mid-check
    print(f"( cd {WT} && env -u PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE -u PPA_EVAL_SYSTEM2_SAMPLE_TOP_P "
          f"PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO={WT} PPA_EVAL_EXTERNAL_SERVERS=1 "
          f"PPA_EVAL_GPU_DEVICES={args.gpu} PPA_EVAL_SHARDS={args.shard} PPA_EVAL_ALLOW_COMMIT_MISMATCH=1 "
          f"PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_EXTERNAL_SERVER_DIR={q(servers)} PPA_EVAL_OUTPUT_ROOT={q(args.output)} "
          f"PPA_EVAL_MODEL_PORT_BASE={args.model_port_base} PPA_EVAL_VO_PORT_BASE={args.vo_port_base} "
          f"PPA_EVAL_PROTOCOL_SEED=1337 PPA_EVAL_STOP_AT_STEP_BUDGET=1 PPA_EVAL_DISPLAY_BASE={args.display_base} "
          f"PPA_EVAL_SHARD_RETRIES=0 PPA_EVAL_SYSTEM2_LORA={q(args.sha)} PPA_EVAL_EPISODE_LISTS_DIR={q(args.lists)} "
          f"bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) > {q(args.output)}.out 2>&1 < /dev/null")
    # the file val_4090.sh waits on (X37_EQUIV_JSON); a later check overwrites it, so the last one run counts
    print(f"/opt/conda/bin/python {q(str(Path(__file__).resolve()))} compare --reference {B0_RUN} "
          f"--candidate {q(args.output)} --episodes {args.episodes} --output {q(args.equiv_json)}")
    return 0


def mode_problems(candidate: str, reference_rows: dict, candidate_rows: dict, ids: list[int]) -> list[str]:
    problems = []
    for path in sorted(glob.glob(f"{candidate}/runtime/*/external_servers.json")):
        d = json.loads(Path(path).read_text())
        if d.get("system2_base_stop") is not True or not str(d.get("model_version", "")).endswith(MODE_SUFFIX):
            problems.append(f"{path}: system2_base_stop {d.get('system2_base_stop')!r}, "
                            f"model_version {d.get('model_version')!r} (not base-stop-v1)")
    for e in ids:
        version = (candidate_rows.get(e) or {}).get("rpc_model_version")
        if version is not None and not str(version).endswith(MODE_SUFFIX):
            problems.append(f"candidate episode {e}: rpc_model_version {version!r} lacks {MODE_SUFFIX}")
        version = (reference_rows.get(e) or {}).get("rpc_model_version")
        if version is not None and version != B0_MODEL_VERSION:
            problems.append(f"reference episode {e}: rpc_model_version {version!r}, not B0's {B0_MODEL_VERSION!r}")
    return problems


def compare_episode(e: int, ref: dict, cand: dict, ref_row: dict, cand_row: dict) -> dict:
    first = None
    for i, (a, b) in enumerate(zip(ref["calls"], cand["calls"])):
        fields = [f for f in CALL_FIELDS if a.get(f) != b.get(f)]
        if fields:
            first = {"call_index": i, "step": a.get("step"), "fields": fields,
                     "reference": {f: a.get(f) for f in fields}, "candidate": {f: b.get(f) for f in fields}}
            break
    if first is None and len(ref["calls"]) != len(cand["calls"]):
        i = min(len(ref["calls"]), len(cand["calls"]))
        first = {"call_index": i, "fields": ["call count"],
                 "reference": {"calls": len(ref["calls"])}, "candidate": {"calls": len(cand["calls"])}}
    outcome = [f for f in ROW_FIELDS if ref_row.get(f) != cand_row.get(f)]
    if ref.get("final") != cand.get("final"):
        outcome.append("final line")
    return {
        "episode": f"{ref['scene_id']}_{e:04d}",
        "calls_reference": len(ref["calls"]),
        "calls_candidate": len(cand["calls"]),
        "calls_by_kind": dict(collections.Counter(c["kind"] for c in ref["calls"])),
        "identical": first is None and not outcome,
        "first_difference": first,
        "outcome_differs": outcome,
        "outcome_reference": {f: ref_row.get(f) for f in ROW_FIELDS},
        "outcome_candidate": {f: cand_row.get(f) for f in ROW_FIELDS},
    }


def cmd_compare(args) -> int:
    ids = episode_ids(args.episodes)
    ref_blocks, _, _, _ = parse_client_logs(log_glob(args.reference))
    try:
        cand_blocks, _, _, _ = parse_client_logs(log_glob(args.candidate))
    except FileNotFoundError:
        cand_blocks = {}
    unreadable = []
    ref_rows, cand_rows = {}, {}
    for name, run, into in (("reference", args.reference, ref_rows), ("candidate", args.candidate, cand_rows)):
        try:
            into.update(progress_rows(run))
        except RowsUnreadable as err:  # not measured (exit 2), not a difference
            unreadable.append(f"{name} rows: {err}")
    # the server sets the candidate ran against, as the launcher copied them (val_4090.sh pins the X37 sets'
    # repo_commit to the one checked here)
    servers = [{k: json.loads(Path(p).read_text()).get(k) for k in
                ("server_instance", "repo_commit", "repo_dirty", "model_version", "system2_lora_sha256",
                 "system2_base_stop")} for p in sorted(glob.glob(f"{args.candidate}/runtime/*/external_servers.json"))]
    report = {"reference": args.reference, "candidate": args.candidate, "episodes_asked": ids,
              "candidate_servers": servers,
              "problems": unreadable + mode_problems(args.candidate, ref_rows, cand_rows, ids), "episodes": []}
    if len({s["repo_commit"] for s in servers}) > 1:
        report["problems"].append(f"the candidate ran against servers of {len({s['repo_commit'] for s in servers})} commits")
    for e in (ids if not unreadable else ()):
        missing = [name for name, have in (("reference log", ref_blocks), ("candidate log", cand_blocks),
                                           ("reference row", ref_rows), ("candidate row", cand_rows)) if e not in have]
        if missing:
            report["problems"].append(f"episode {e}: no {', no '.join(missing)}")
            continue
        report["episodes"].append(compare_episode(e, ref_blocks[e], cand_blocks[e], ref_rows[e], cand_rows[e]))
    rows = report["episodes"]
    report["identical"] = sum(r["identical"] for r in rows)
    report["calls_compared"] = sum(min(r["calls_reference"], r["calls_candidate"]) for r in rows)
    differ = [r for r in rows if not r["identical"]]
    if report["problems"]:
        report["status"], code = "not measured", 2
        line = "NOT MEASURED: " + "; ".join(report["problems"])
    elif differ:
        report["status"], code = "differ", 1
        r = differ[0]
        d = r["first_difference"]
        where = (f"call {d['call_index']} (step {d.get('step')}): {', '.join(d['fields'])}: "
                 f"B0 {d['reference']} vs candidate {d['candidate']}") if d else f"outcome {r['outcome_differs']}"
        line = f"FAIL {report['identical']}/{len(ids)} episodes identical; first difference in {r['episode']} at {where}"
    else:
        report["status"], code = "identical", 0
        line = f"PASS {len(ids)}/{len(ids)} episodes identical, {report['calls_compared']} calls"
    for r in rows:
        print(f"{r['episode']}: {'identical' if r['identical'] else 'DIFFERS'} "
              f"({r['calls_reference']} vs {r['calls_candidate']} calls {r['calls_by_kind']})")
    print(line)
    report["line"] = line
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--reference", default=B0_RUN)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--count", type=int, default=3)
    p = sub.add_parser("lists")
    p.add_argument("--cohorts", default=f"{PLAN}/cohorts")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--episodes", default=EPISODES)
    p.add_argument("--out", required=True)
    p = sub.add_parser("command")
    p.add_argument("--tag", required=True, help="the publication npu_servers_exp28_<tag> of the zero-adapter set")
    p.add_argument("--model-port-base", type=int, required=True)
    p.add_argument("--vo-port-base", type=int, required=True)
    p.add_argument("--sha", required=True, help="sha256 of the zero adapter's adapter_model.safetensors")
    p.add_argument("--lists", required=True, help="the lists subcommand's --out")
    p.add_argument("--episodes", default=EPISODES, help="the episodes --lists holds, for the compare line")
    p.add_argument("--output", required=True, help="a new run directory")
    p.add_argument("--equiv-json", default=EQUIV_JSON, help="the compare line's --output (val_4090.sh's X37_EQUIV_JSON)")
    p.add_argument("--servers-root", default="/workspace/exp37/npu_servers")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--gpu", default="0", help="one slot label; the RPC client never touches CUDA (val_4090.sh)")
    p.add_argument("--display-base", type=int, default=460)
    p = sub.add_parser("compare")
    p.add_argument("--reference", default=B0_RUN)
    p.add_argument("--candidate", required=True)
    p.add_argument("--episodes", default=EPISODES)
    p.add_argument("--output", default="")
    args = parser.parse_args(argv)
    return {"pick": cmd_pick, "lists": cmd_lists, "command": cmd_command, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
