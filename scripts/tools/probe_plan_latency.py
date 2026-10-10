#!/usr/bin/env python
"""Drive the deployed servers from their own host and record what each plan call cost.

Why this exists.  Wall clock of a real episode cannot answer "how fast is the server",
because the Habitat client sits on another machine behind a tunnel and competes for its
own CPU -- the same unchanged 910B configuration measured 470 s and 824 s for one
episode purely from the client host's load (docs/ops/ascend_910b_open_problems.md).
This driver runs ``src.deploy.nav_agent.NavAgent`` against the real servers **on the
server host**, with frames that are generated from a seed rather than simulated, so

  * ``run`` gives per-plan-call latency that no other machine can perturb, and
  * ``compare`` checks two runs decided the same thing, field by field.

``compare`` is the half that keeps a speed change honest.  Every lever left on this
platform (fused attention, graph mode, static KV cache) can change numerics, and the
cheap ones -- reading a bounds vector once instead of per slice -- must be *shown* not
to.  It refuses to print a verdict when it compared no plan calls: an earlier ad hoc
version of this diff reported "identical" off zero compared calls because it read the
wrong attribute, which is worse than no evidence at all.

The frames are deterministic noise over a fixed gradient, not a scene.  That is fine
for a comparison (both arms see the same pixels) and for the shapes that dominate cost
(12 views, 9408 patches, fixed per call), but the decode length depends on what the
model says, so ``run`` reports the generated-token count beside the latency: compare
latencies only between runs whose token counts agree.

    PYTHONPATH=$ROOT/rpc/src:$ROOT/HeatmapVLN \
    probe_plan_latency.py run  --model 127.0.0.1:52400 --vo 127.0.0.1:52500 \
        --plan-calls 14 --warmup-calls 6 --out /tmp/arm_on.json --label "mask patch on"
    probe_plan_latency.py compare /tmp/arm_on.json /tmp/arm_off.json

``run`` against real servers needs ``vla_rpc`` on PYTHONPATH -- the same
``$ROOT/rpc/src`` the launcher puts there (scripts/ascend/run_ppa_servers_npu.sh).
``run --fake`` does not: it brings its own encoder, so it works in a bare checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src" / "deploy" / "nav_agent.py").is_file():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
# Else: the file was copied out of the tree (the deployment host keeps the repo
# read-only and pull-only), and PYTHONPATH has to name the repo instead.

# Everything the server decided.  Timing is deliberately absent: it is what differs.
DECISION_FIELDS = (
    "kind",
    "llm_output",
    "actions",
    "pixel_goal",
    "terminal",
    "pose_ready",
    "ppa_applied",
    "vo_frame_id",
    "vo_history_frame_ids",
    "vo_provider_phase",
    "vo_trajectory_revision",
    "history_capture_steps",
    "response",
)
# Keys a response carries as diagnostics rather than as an answer: the stage times, the
# accelerator memory reading taken beside them, and how the server reached the answer
# (how many vision-tower passes it ran, and which were served from an earlier one).
# None is a decision.  Comparing them would make every timing-on-vs-off run DIFFERENT,
# and would make an A/B of a reuse that is *meant* to change the pass count report a
# difference for the one thing it is allowed to change.
DIAGNOSTIC_KEYS = (
    "timing_ms",
    "server_timing_ms",
    "client_timing_ms",
    "cuda_memory_mib",
    "vision_tower",
)
# Env this DRIVER ran with.  Deliberately named for what it is: the server is a
# separate process, usually started from a different shell, so these say nothing about
# how it was configured.  The first version of this tool recorded them as "env" and a
# knob-flipping A/B then reported "env differences: none" while the two servers really
# did differ -- the arm has to come from --label, which the server cannot lie about
# either, so compare also refuses to tell two runs apart when their labels match.
RECORDED_CLIENT_ENV = (
    "HEATMAPVLN_TIMING",
    "ASCEND_LAUNCH_BLOCKING",
    "PYTHONPATH",
)


# --------------------------------------------------------------------------- frames
def make_frames(seed: int, width: int, height: int):
    """Deterministic RGB frames: one per (step, kind), from ``seed`` alone.

    A vertical gradient keeps the image from being pure noise (a uniformly random image
    makes the vision tower's attention degenerate), and the per-step noise keeps every
    step's pixels distinct so the VO has something to track.
    """
    import numpy as np

    base = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
    base = np.repeat(np.repeat(base, width, axis=1), 3, axis=2)

    def frame(step: int, kind: str) -> "np.ndarray":
        # A fresh generator per frame: the frame for (step, kind) is the same whatever
        # order the caller asks for them in, so a re-run or a second arm matches.
        rng = np.random.default_rng([seed, step, sum(kind.encode())])
        noise = rng.integers(-40, 41, size=(height, width, 3), dtype=np.int16)
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    return frame


# ------------------------------------------------------------------------------ run
def _call_record(call: Any) -> dict[str, Any]:
    record = {
        "call_index": call.call_index,
        "step": call.step,
        "model_rpc_ms": round(float(call.client_timing_ms.get("model_rpc", 0.0)), 2),
        "vo_query_ms": round(float(call.client_timing_ms.get("vo_query", 0.0)), 2),
        "server_timing_ms": call.server_timing_ms,
        "llm_output_chars": len(call.llm_output or ""),
    }
    for name in DECISION_FIELDS:
        value = getattr(call, name)
        if name == "response" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in DIAGNOSTIC_KEYS}
        record[name] = value
    return record


def _clients(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """The two servers to drive, and the kwargs NavAgent needs for them."""
    if not args.fake:
        return args.model, args.vo, {}
    # The scripted stand-ins: they check every request the real servers check and answer
    # from a script, so this exercises the whole driver without an accelerator.  The
    # latency they report is the driver's own overhead, which is worth knowing too.
    from src.deploy.fake_servers import FakeModelServer, FakeVOServer, pil_jpeg_encoder

    return FakeModelServer(timing=True), FakeVOServer(timing=True), {"jpeg_encoder": pil_jpeg_encoder}


def cmd_run(args: argparse.Namespace) -> int:
    from src.deploy.nav_agent import Action, NavAgent

    model, vo, extra = _clients(args)
    frame = make_frames(args.seed, args.width, args.height)
    steps: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    episode = args.episode_id
    started = time.time()

    with NavAgent(model, vo, timing=True, max_steps=args.max_steps, **extra) as agent:
        server_info = {
            "model": getattr(getattr(agent, "model_server_info", None), "model_version", None),
            "vo": getattr(getattr(agent, "vo_server_info", None), "model_version", None),
        }
        agent.reset(args.instruction, scene_id=args.scene_id, episode_id=episode)
        step_budget = args.max_total_steps
        while len(calls) < args.plan_calls and step_budget > 0:
            if agent.done:
                episode += 1
                agent.reset(args.instruction, scene_id=args.scene_id, episode_id=episode)
            step = agent.steps
            t0 = time.perf_counter()
            action = agent.act(
                frame(step, "front"),
                lambda s=step: frame(s, "lookdown"),
                lambda s=step: frame(s, "level"),
            )
            wall_ms = (time.perf_counter() - t0) * 1e3
            step_budget -= 1
            info = agent.last_step
            steps.append(
                {
                    "episode_id": episode,
                    "step": step,
                    "action": int(action),
                    "action_name": Action(action).name,
                    "source": info.source,
                    "wall_ms": round(wall_ms, 2),
                    "had_call": info.call is not None,
                    # Every client stage, so a slow step can be attributed without a
                    # second run: vo_ingest fires on every step, vo_query and model_rpc
                    # only on a planning one.  Not compared -- see compare's field list.
                    "client_timing_ms": {k: round(float(v), 2) for k, v in (info.timing_ms or {}).items()},
                    "vo_server_timing_ms": info.vo_server_timing_ms,
                }
            )
            if info.call is not None:
                record = _call_record(info.call)
                record["episode_id"] = episode
                record["step_wall_ms"] = round(wall_ms, 2)
                calls.append(record)
                print(
                    f"  call {len(calls):>2} step {step:>3} "
                    f"model_rpc {record['model_rpc_ms']:>8.1f} ms  "
                    f"vo {record['vo_query_ms']:>7.1f} ms  "
                    f"out {record['llm_output_chars']:>4} chars  "
                    f"actions={record['actions']}",
                    flush=True,
                )

    if len(calls) < args.plan_calls:
        print(
            f"WARNING: wanted {args.plan_calls} plan calls, got {len(calls)} in "
            f"{args.max_total_steps} steps",
            file=sys.stderr,
        )

    out = {
        "label": args.label,
        "argv": sys.argv[1:],
        "seed": args.seed,
        "server_info": server_info,
        "client_env": {name: os.environ.get(name) for name in RECORDED_CLIENT_ENV},
        "wall_s": round(time.time() - started, 2),
        "steps": steps,
        "calls": calls,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    _print_latency(out, args.warmup_calls)
    print(f"wrote {len(calls)} plan calls / {len(steps)} steps to {args.out}")
    return 0


def _call_class(call: dict[str, Any]) -> str:
    """Plan calls are not interchangeable, so never pool them into one median.

    A call that only turns the robot runs System 2 and stops; one that drives forward
    also runs the trajectory head and the PPA refine.  They differ by several seconds,
    so a single median over a mixed run says more about the mix than about the server.
    """
    return f"{call.get('kind')}{'+ppa' if call.get('ppa_applied') else ''}"


def _print_latency(run: dict[str, Any], warmup: int) -> None:
    calls = run["calls"]
    if not calls:
        print("no plan calls -- nothing to summarise")
        return
    print(f"--- {run.get('label') or 'run'}: {len(calls)} plan calls, {run['wall_s']} s wall")
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, call in enumerate(calls):
        groups.setdefault(_call_class(call), []).append({**call, "_index": index})
    for name in sorted(groups):
        rows = groups[name]
        ms = [r["model_rpc_ms"] for r in rows]
        steady = [r["model_rpc_ms"] for r in rows if r["_index"] >= warmup]
        chars = sorted({r["llm_output_chars"] for r in rows})
        print(
            f"    {name:<18} n={len(rows):<3} min {min(ms):8.1f}  "
            f"median {statistics.median(ms):8.1f}  max {max(ms):8.1f} ms"
        )
        if steady and len(steady) != len(rows):
            print(
                f"      {'after call ' + str(warmup):<16} n={len(steady):<3} min {min(steady):8.1f}  "
                f"median {statistics.median(steady):8.1f}  max {max(steady):8.1f} ms"
            )
        print(f"      llm_output chars {chars}")
    ms = [c["model_rpc_ms"] for c in calls]
    print(f"    {'ALL (mixed)':<18} n={len(ms):<3} median {statistics.median(ms):8.1f} ms"
          "   <- mix-dependent, do not compare across runs with different mixes")


# -------------------------------------------------------------------------- compare
def _decision(call: dict[str, Any], name: str) -> Any:
    """One recorded field, with any timing inside it removed.

    ``run`` already strips timing from the response it records, but strip it here too:
    a run recorded by an older version of this tool, or by hand, must not be reported
    as a difference because the server happened to be slower that day.
    """
    value = call.get(name)
    if name == "response" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in DIAGNOSTIC_KEYS}
    return value


def _diff_calls(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    diffs = []
    for name in DECISION_FIELDS:
        if _decision(a, name) != _decision(b, name):
            diffs.append(f"call {a['call_index']} step {a['step']}: {name} differs\n"
                         f"      A: {_decision(a, name)!r}\n      B: {_decision(b, name)!r}")
    return diffs


def cmd_compare(args: argparse.Namespace) -> int:
    runs = []
    for path in (args.a, args.b):
        try:
            runs.append(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as exc:
            print(f"cannot read {path}: {exc}", file=sys.stderr)
            return 2
    a_run, b_run = runs
    label_a, label_b = a_run.get("label") or "", b_run.get("label") or ""
    print(f"A: {label_a or args.a}")
    print(f"B: {label_b or args.b}")
    print(f"A server: {a_run.get('server_info')}")
    print(f"B server: {b_run.get('server_info')}")
    if label_a == label_b:
        # Not fatal -- a repeatability check compares two runs of the same arm on
        # purpose -- but the labels are the only record of what was different.
        print("NOTE: both runs carry the same --label, so this output does not say "
              "what was varied between them")

    a_calls, b_calls = a_run.get("calls", []), b_run.get("calls", [])
    # The failure this guard exists for: an empty comparison reads as agreement.
    if not a_calls or not b_calls:
        print(f"VACUOUS: compared no plan calls (A has {len(a_calls)}, B has {len(b_calls)})",
              file=sys.stderr)
        return 2

    diffs: list[str] = []
    a_steps, b_steps = a_run.get("steps", []), b_run.get("steps", [])
    if len(a_steps) != len(b_steps):
        diffs.append(f"step count differs: A {len(a_steps)} B {len(b_steps)}")
    for sa, sb in zip(a_steps, b_steps):
        for name in ("step", "action", "source", "had_call", "episode_id"):
            if sa.get(name) != sb.get(name):
                diffs.append(f"step {sa.get('step')}: {name} A={sa.get(name)!r} B={sb.get(name)!r}")
    if len(a_calls) != len(b_calls):
        diffs.append(f"plan-call count differs: A {len(a_calls)} B {len(b_calls)}")
    compared = 0
    for ca, cb in zip(a_calls, b_calls):
        diffs.extend(_diff_calls(ca, cb))
        compared += 1

    fields = len(DECISION_FIELDS)
    print(f"compared {compared} plan calls x {fields} decision fields, "
          f"{len(a_steps)} steps x 5 fields")
    for line in diffs[: args.max_diffs]:
        print(f"  {line}")
    if len(diffs) > args.max_diffs:
        print(f"  ... and {len(diffs) - args.max_diffs} more")

    for run in runs:
        _print_latency(run, args.warmup_calls)

    if diffs:
        print(f"VERDICT: DIFFERENT ({len(diffs)} differences)")
        return 1
    print("VERDICT: IDENTICAL")
    return 0


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive the servers and record every plan call")
    run.add_argument("--model", help="model server host:port")
    run.add_argument("--vo", help="AMB3R-VO server host:port")
    run.add_argument(
        "--fake",
        action="store_true",
        help="drive src/deploy/fake_servers.py instead: exercises this driver with no accelerator",
    )
    run.add_argument("--out", required=True, help="where to write the run JSON")
    run.add_argument("--label", default="", help="free-text name for this arm")
    run.add_argument("--plan-calls", type=int, default=8, help="stop after this many plan calls")
    run.add_argument("--max-total-steps", type=int, default=200, help="give up after this many act() calls")
    run.add_argument("--warmup-calls", type=int, default=2, help="calls excluded from the steady-state summary")
    run.add_argument("--seed", type=int, default=20261010, help="seeds the frames (not the server's sampling)")
    run.add_argument("--scene-id", default="probe")
    run.add_argument("--episode-id", type=int, default=0)
    run.add_argument("--instruction", default="Walk forward down the hallway and stop at the door on your left.")
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--max-steps", type=int, default=500, help="NavAgent's per-episode step cap")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="field-by-field diff of what two runs decided")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--warmup-calls", type=int, default=2)
    cmp_.add_argument("--max-diffs", type=int, default=20)
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run" and not args.fake and not (args.model and args.vo):
        parser.error("run needs --model and --vo (or --fake)")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
