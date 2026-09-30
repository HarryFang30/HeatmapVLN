#!/usr/bin/env python3
"""Drive NavAgent (src/deploy/nav_agent.py) inside Habitat, logged the way the deployed client logs.

NavAgent sees Habitat only as a robot would: the front RGB after every step, a look-down image
taken through the evaluation client's own capture_lookdown_view when it asks for one, and a
level re-render of the front view when it asks for one (a replan right after LOOK_DOWN).
Everything else comes from scripts/evaluation/r2r_val_unseen.py ("the client") unchanged: the
Habitat config, the fixed episode cohort, action stepping (LOOKDOWN twice), and one side effect
the robot contract has to reproduce: the client's per-pass panorama capture puts the camera back
to level (agent.set_state(..., reset_sensors=True)) before any further capture or action.

Writes to --output_path: client.log (the client's stdout lines: episode header, [amb3r-vo],
step_id / actions, local STOP replans, => success), progress.json (one row per episode) and
calls.jsonl (one row per plan call, with timing when --timing is on).

  # against running servers (scripts/deploy/start_nav_servers_cuda.sh), then compare with the
  # 4090 canary call by call:
  python scripts/deploy/nav_agent_habitat_check.py --model-addr 127.0.0.1:52400 --vo-addr 127.0.0.1:52500 \\
      --data_path <plan>/cohorts/dataset_shard_00.json.gz --episode_list <plan>/cohorts/shard_00.json \\
      --max_episodes 2 --output_path <dir> --compare-log <canary>/runtime/<stamp>/logs/client_shard_00.log

  # GPU-free: scripted servers (src/deploy/fake_servers.py).  --reference-client runs the
  # client's own loop instead of NavAgent on the same servers; --record-requests keeps every
  # request, --compare-requests checks this run's requests against such a recording.
  python scripts/deploy/nav_agent_habitat_check.py --fake-servers --reference-client --record-requests ref.jsonl ...
  python scripts/deploy/nav_agent_habitat_check.py --fake-servers --record-requests nav.jsonl --compare-requests ref.jsonl ...

Needs the Habitat container (habitat-lab 0.1.7, vla_rpc on PYTHONPATH) and an X display
(Xvfb + llvmpipe, as in scripts/run_ppa_r2r_val_unseen_cuda.sh).  Exit code 1 when a
comparison finds a difference.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = REPO / "configs" / "ppa_action_refine_v2_8gpu.yaml"
FAKE_MODEL_ADDR = "fake-model"
FAKE_VO_ADDR = "fake-vo"


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _level_camera(env) -> None:
    """The client's capture_panoramic_views ends with agent.set_state(state, reset_sensors=True)."""
    agent = env._sim.get_agent(0)
    agent.set_state(agent.get_state(), reset_sensors=True)


def run_nav_agent(args, client, model, vo, log: Callable[[str], None], calls_out) -> None:
    from src.deploy.nav_agent import LOOKDOWN_IMAGE_SIZE, NavAgent

    client.ensure_vln_measures_registered()
    env = client._create_habitat_env(client.build_habitat_config(args), args)
    target_list, target_set = client._episode_list_from_args(args)
    eval_limit = client._eval_limit(args, len(list(env.episodes)), target_list, set())
    agent = NavAgent(
        model,
        vo,
        protocol_seed=args.protocol_seed,
        timing=True if args.timing else None,
        max_steps=args.max_steps,
        rpc_timeout_ms=args.rpc_timeout_ms,
        on_log=log,
    )

    def lookdown():
        # The client's own look-down: 2 x LOOKDOWN, capture, 2 x LOOKUP (env steps).
        return np.asarray(client.capture_lookdown_view(env, image_size=LOOKDOWN_IMAGE_SIZE))

    def level():
        # The client's second panorama capture in one step: the yaw-0 view of the sensor its
        # first capture (emulated by _level_camera before act()) left level.
        _level_camera(env)
        rgb = client._extract_rgb_array(env._sim.get_sensor_observations())
        return np.ascontiguousarray(rgb[:, :, :3])

    progress = Path(args.output_path) / "progress.json"
    seen, count = set(), 0
    while count < eval_limit:
        observations = env.reset()
        episode = env.current_episode
        scene_id, episode_id = episode.scene_id.split("/")[-2], int(episode.episode_id)
        if (scene_id, episode_id) in seen:
            break
        seen.add((scene_id, episode_id))
        if target_set is not None and (scene_id, episode_id) not in target_set:
            continue
        count += 1
        agent.reset(episode.instruction.instruction_text, scene_id=scene_id, episode_id=episode_id)
        log(f"\n[{count}/{eval_limit}] Episode {scene_id}_{episode_id:04d}: {agent.instruction[:80]}...")
        done = False
        # The client's loop bound: at the cap it stops stepping; it never executes a STOP there.
        while not done and agent.steps < args.max_steps:
            front = client._extract_rgb_array(observations)
            _level_camera(env)
            action = agent.act(front, lookdown, level)
            step = agent.last_step
            if step.call is not None:
                call = step.call
                calls_out.write(json.dumps({
                    "scene_id": scene_id, "episode_id": episode_id, "call_index": call.call_index,
                    "step": call.step, "kind": call.kind, "llm_output": call.llm_output, "actions": call.actions,
                    "pixel_goal": call.pixel_goal, "pose_ready": call.pose_ready, "ppa_applied": call.ppa_applied,
                    "vo_frame_id": call.vo_frame_id, "vo_history_frame_ids": call.vo_history_frame_ids,
                    "client_timing_ms": call.client_timing_ms, "server_timing_ms": call.server_timing_ms,
                    "vo_server_timing_ms": step.vo_server_timing_ms,
                }, ensure_ascii=False) + "\n")
                calls_out.flush()
            observations, done = client._apply_habitat_action(env, int(action))
        metrics = env.get_metrics()
        log(
            f"  => success: {metrics['success']}, spl: {metrics['spl']:.4f}, "
            f"os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']:.4f}, "
            f"vlm_calls: {agent.calls}, trajectory_calls: {agent.trajectory_calls}"
        )
        row = {
            "scene_id": scene_id, "episode_id": episode_id, "success": metrics["success"], "spl": metrics["spl"],
            "os": metrics["oracle_success"], "ne": metrics["distance_to_goal"], "steps": agent.steps,
            "episode_instruction": agent.instruction, "vlm_calls": agent.calls,
            "trajectory_calls": agent.trajectory_calls, "ppa_applied_calls": agent.ppa_applied_calls,
            "ppa_warmup_calls": agent.ppa_warmup_calls, "rpc_protocol_seed": args.protocol_seed,
            "history_pose_source": "amb3r_vo_da3", "client": "nav_agent",
        }
        with progress.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    env.close()
    agent.close()


def run_reference_client(args, client, model, vo) -> None:
    """The client's own run_eval_rpc_panoramic, with the deployed launcher's flags, on fakes."""
    import vla_rpc.client

    servers = {FAKE_MODEL_ADDR: model, FAKE_VO_ADDR: vo}
    vla_rpc.client.VLAClient = lambda server_addr, **_kwargs: servers[server_addr]
    argv = [
        "--config", str(args.config), "--rpc_server", FAKE_MODEL_ADDR,
        "--history_pose_source", "amb3r_vo_da3", "--amb3r_vo_rpc_server", FAKE_VO_ADDR,
        "--amb3r_vo_rpc_timeout_ms", "600000", "--amb3r_vo_rpc_jpeg_quality", "95",
        "--rpc_timeout_ms", "600000", "--rpc_jpeg_quality", "90",
        "--rpc_protocol_seed", str(args.protocol_seed), "--rpc_require_deterministic_sampling",
        "--rpc_policy_mode", "heatmapvln", "--scenes_dir", args.scenes_dir, "--data_path", args.data_path,
        "--dataset_split", args.dataset_split, "--output_path", str(Path(args.output_path) / "reference"),
        "--sim_gpu_id", str(args.sim_gpu_id), "--resize_w", "384", "--resize_h", "384", "--num_history", "8",
        "--max_steps_per_episode", str(args.max_steps), "--max_system2_calls_per_episode", "0",
        "--auto_stop_distance", "0", "--trajectory_selection", "mean", "--trajectory_x_sign", "1",
        "--trajectory_heading_alignment", "none", "--system1_coord_order", "generated",
        "--no-pano_recenter_before_system1", "--no-debug_input_trace", "--debug_save_input_images", "0",
    ]
    if args.episode_list:
        argv += ["--episode_list", args.episode_list]
    if args.max_episodes is not None:
        argv += ["--max_episodes", str(args.max_episodes)]
    saved = sys.argv
    sys.argv = ["r2r_val_unseen.py", *argv]
    try:
        client.main()
    finally:
        sys.argv = saved


# ------------------------------------------------------------------------------ comparisons
def compare_requests(ours: Path, reference: Path) -> dict[str, Any]:
    """Request-by-request equality, except the side views NavAgent replaces with a placeholder.

    Payload JSON text and every blob's name, type, size and bytes must match; for
    current/{right,back,left} and history/<i>/{right,back,left} only name, type and size.
    """
    a = [json.loads(line) for line in ours.read_text().splitlines() if line.strip()]
    b = [json.loads(line) for line in reference.read_text().splitlines() if line.strip()]
    side = lambda name: name.split("/")[-1] in ("right", "back", "left")  # noqa: E731
    diffs = []
    for index, (x, y) in enumerate(zip(a, b)):
        why = []
        for key in ("target", "method", "payload"):
            if x[key] != y[key]:
                why.append(key)
        if [(o["name"], o["mime_type"], o["height"], o["width"]) for o in x["blobs"]] != [
            (o["name"], o["mime_type"], o["height"], o["width"]) for o in y["blobs"]
        ]:
            why.append("blob layout")
        else:
            why += [f"blob {o['name']}" for o, r in zip(x["blobs"], y["blobs"]) if not side(o["name"]) and o["sha256"] != r["sha256"]]
        if why:
            diffs.append({"index": index, "target": x["target"], "method": x["method"], "differs": why})
    counts = {"ours": len(a), "reference": len(b)}
    side_blobs = sum(side(o["name"]) for x in a for o in x["blobs"])
    compared = sum(not side(o["name"]) for x in a for o in x["blobs"])
    return {"requests": counts, "identical": not diffs and len(a) == len(b), "first_differences": diffs[:20],
            "blobs_compared_bytewise": compared, "side_view_blobs_layout_only": side_blobs}


def _call_key(call: dict) -> tuple:
    # scripts/exp19/build_records.py:_call_key (not imported: that module pulls pandas and exp18).
    return (int(call["step"]), str(call["kind"]), str(call.get("vlm_output") or "").strip(),
            [int(a) for a in (call.get("actions") or [])])


def compare_logs(ours: Path, reference: Path) -> dict[str, Any]:
    """Call-by-call and outcome equality of two client logs (scripts/exp19/select_cases.py parser)."""
    from scripts.exp19.select_cases import parse_client_log

    mine, _, _ = parse_client_log(ours)
    theirs, _, _ = parse_client_log(reference)
    episodes = []
    for episode_id, block in sorted(mine.items()):
        ref = theirs.get(episode_id)
        if ref is None or ref["scene_id"] != block["scene_id"]:
            continue
        a = {int(c["call_index"]): _call_key(c) for c in block["calls"]}
        b = {int(c["call_index"]): _call_key(c) for c in ref["calls"]}
        per_call = [i in a and a.get(i) == b.get(i) for i in range(max(list(a) + list(b), default=-1) + 1)]
        first = next((i for i, same in enumerate(per_call) if not same), None)
        episodes.append({
            "episode": f"{block['scene_id']}_{episode_id:04d}", "calls": len(a), "reference_calls": len(b),
            "identical_calls": int(sum(per_call)), "first_divergent_call": first,
            "final": block["final"], "reference_final": ref["final"],
            "identical": first is None and block["final"] == ref["final"],
        })
    return {"episodes": episodes, "identical": bool(episodes) and all(e["identical"] for e in episodes)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    servers = parser.add_mutually_exclusive_group(required=True)
    servers.add_argument("--model-addr", help="host:port of rpc_model_server.py")
    servers.add_argument("--fake-servers", action="store_true", help="scripted servers, no GPU")
    parser.add_argument("--vo-addr", help="host:port of rpc_amb3r_vo_server.py (with --model-addr)")
    parser.add_argument("--reference-client", action="store_true",
                        help="run the client's own loop instead of NavAgent (needs --fake-servers)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="read by --reference-client only")
    parser.add_argument("--scenes_dir", default="/dataset")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_split", default="val_unseen")
    parser.add_argument("--episode_list", default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sim_gpu_id", type=int, default=0)
    parser.add_argument("--protocol-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rpc-timeout-ms", type=int, default=600_000)
    parser.add_argument("--timing", action="store_true", help="NavAgent client timing (HEATMAPVLN_TIMING also works)")
    parser.add_argument("--record-requests", type=Path, default=None, help="JSONL of every request (--fake-servers)")
    parser.add_argument("--compare-requests", type=Path, default=None, help="reference JSONL from --record-requests")
    parser.add_argument("--compare-log", type=Path, default=None, help="client log of a deployed run, e.g. the canary")
    args = parser.parse_args(argv)
    if args.model_addr and not args.vo_addr:
        parser.error("--model-addr needs --vo-addr")
    if args.reference_client and not args.fake_servers:
        parser.error("--reference-client runs only against --fake-servers")
    if (args.record_requests or args.compare_requests) and not args.fake_servers:
        parser.error("--record-requests / --compare-requests need --fake-servers")
    if args.compare_requests and not args.record_requests:
        parser.error("--compare-requests needs --record-requests for this run")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("client.log", "progress.json", "calls.jsonl"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; choose a new --output_path")
    # The client module brings Habitat's runtime patches; import it before anything else heavy.
    import scripts.evaluation.r2r_val_unseen as client

    client._resolve_eval_paths(args, split=args.dataset_split)
    if args.fake_servers:
        from src.deploy.fake_servers import FakeModelServer, FakeVOServer

        record_file = args.record_requests.open("w") if args.record_requests else None
        recorder = (lambda item: record_file.write(json.dumps(item, ensure_ascii=False) + "\n")) if record_file else None
        model, vo = FakeModelServer(recorder), FakeVOServer(recorder)
    else:
        model, vo, record_file = args.model_addr, args.vo_addr, None

    status = 0
    with (out / "client.log").open("w") as log_file, (out / "calls.jsonl").open("w") as calls_out:
        if args.reference_client:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
                run_reference_client(args, client, model, vo)
        else:
            def log(line: str) -> None:
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

            run_nav_agent(args, client, model, vo, log, calls_out)
    if record_file is not None:
        record_file.close()

    if args.compare_requests:
        result = compare_requests(args.record_requests, args.compare_requests)
        (out / "compare_requests.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[nav-agent-check] requests {result['requests']} identical={result['identical']} "
              f"(bytewise blobs {result['blobs_compared_bytewise']}, side views layout-only "
              f"{result['side_view_blobs_layout_only']})")
        status |= 0 if result["identical"] else 1
    if args.compare_log:
        result = compare_logs(out / "client.log", args.compare_log)
        (out / "compare_log.json").write_text(json.dumps(result, indent=2) + "\n")
        for episode in result["episodes"]:
            print(f"[nav-agent-check] {episode['episode']}: {episode['identical_calls']}/{episode['calls']} calls "
                  f"identical (reference {episode['reference_calls']}), first divergent "
                  f"{episode['first_divergent_call']}, final {'same' if episode['final'] == episode['reference_final'] else 'DIFFERENT'}")
        status |= 0 if result["identical"] else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
