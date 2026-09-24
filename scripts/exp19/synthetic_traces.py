#!/usr/bin/env python3
"""EXP-19: a synthetic artifact tree for developing and smoke-testing build_records.py.

No simulator and no model: two fake episodes on an open flat floor are driven
through the same control flow as the RPC client (a call whenever the action
queue is empty; a queued STOP flushes the queue and replans at the same step),
and every file build_records.py reads is written in its contract schema:

  <root>/cases/candidates.json                          exp19-candidates-v1
  <root>/cases/eval_log_reference/<ep_key>.json         exp19-eval-log-ref-v1
  <root>/runs/<run>/gpu0/trace/<ep_key>/call_XXX.{json,npz}   exp19-call-trace-v1
  <root>/runs/<run>/gpu0/steps/<ep_key>/steps.jsonl + front_XXXX.jpg   exp19-step-trace-v1
  <root>/renders/<ep_key>.{npz,json}

Episode A (category T1) ends with STOP near the goal (success).  Episode B
(category F1) passes 1.5 m from the goal and stops more than 3 m away
(oracle success, failure, STOP-ended).  Calls before step 20 are AMB3R warm-up
(no history head); afterwards trajectory calls are "ready".  Pixel goals are the
System1 path endpoint projected into the decision image and written with the
``field_vu`` convention (text "u v", response pixel_goal [v, u]) unless
``--pixel-goal-convention field_uv``.  History-head predictions are the real GT
labels (scripts/exp19/gt.py) perturbed with a fixed RNG: most slots right, some
off by 10 px, some in the wrong view.  Episode A matches its eval-log reference
call for call; episode B diverges at one call.  Depth is 20 m everywhere (no occlusion).

Usage: python -m scripts.exp19.synthetic_traces --root /tmp/exp19_dev_F/synth [--run synth]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp19 import gt  # noqa: E402

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
FORWARD_M, TURN_DEG = 0.25, 15.0
READY_FROM_STEP = 20
STEP_CAP = 500
DEPTH_M = 20.0

EPISODES = {
    "A": {"scene_id": "SynthSceneA", "episode_id": 7, "category": "T1",
          "instruction": "Walk down the hall, turn left at the table and stop by the sofa",
          "native": ("←←", [LEFT, LEFT, STOP, STOP]),
          "ready": [[2, 2, 1, 1], [1, 1, 1, 1], [3, 3, 3, 1], [1, 1, 0, 0], [2, 2, 2, 1], [1, 1, 1, 1],
                    [3, 3, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 0]],
          "diverge_at": None},
    "B": {"scene_id": "SynthSceneB", "episode_id": 42, "category": "F1",
          "instruction": "Go past the kitchen and wait at the bedroom door",
          "native": ("→→→", [RIGHT, RIGHT, RIGHT, STOP]),
          "ready": [[1, 1, 1, 1], [2, 2, 2, 1], [1, 1, 1, 1], [3, 3, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1],
                    [1, 1, 0, 0], [3, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1],
                    [3, 1, 1, 1], [1, 1, 1, 1]],
          "diverge_at": 9},
}


def ep_key(scene_id: str, episode_id: int) -> str:
    return f"{scene_id}_{int(episode_id):04d}"


class Agent:
    def __init__(self):
        self.pos = np.zeros(3)
        self.yaw = 0.0  # radians, left-positive about +y; 0 faces world -z

    def act(self, a: int) -> None:
        if a == FORWARD:
            self.pos = self.pos + FORWARD_M * np.array([-math.sin(self.yaw), 0.0, -math.cos(self.yaw)])
        elif a == LEFT:
            self.yaw += math.radians(TURN_DEG)
        elif a == RIGHT:
            self.yaw -= math.radians(TURN_DEG)

    def quat_wxyz(self) -> list:
        return [math.cos(self.yaw / 2), 0.0, math.sin(self.yaw / 2), 0.0]


def chunk_path(actions: list, length_m: float = 3.2) -> np.ndarray:
    """Robot-frame [33, 2] path (x forward, y left) that the chunk starts, continued straight, resampled by arc length."""
    pts, xy, phi = [np.zeros(2)], np.zeros(2), 0.0
    for a in actions:
        if a == STOP:
            break
        if a == FORWARD:
            xy = xy + FORWARD_M * np.array([math.cos(phi), math.sin(phi)])
            pts.append(xy)
        else:
            phi += math.radians(TURN_DEG if a == LEFT else -TURN_DEG)
    run = FORWARD_M * (len(pts) - 1)
    pts.append(xy + max(length_m - run, 0.3) * np.array([math.cos(phi), math.sin(phi)]))
    p = np.asarray(pts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], 33)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=-1)


def jpeg(img: np.ndarray, quality: int = 90) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def pattern(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    b = np.full((h, w), (37 * seed) % 256, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def simulate(spec: dict) -> dict:
    """Client control flow on the fake agent: states, actions and call records."""
    agent = Agent()
    chunks = [("native_actions", spec["native"][0], list(spec["native"][1]))]
    chunks += [("trajectory", None, [1, 1, 1, 1])] * 5  # AMB3R warm-up: steps 2..21
    chunks += [("trajectory", None, list(c)) for c in spec["ready"]]
    chunks += [("stop", "STOP", [STOP])]
    states, actions, calls = {}, [], []
    step, queue, done, n_call = 0, [], False, 0
    while not done and step < STEP_CAP:
        if step not in states:
            states[step] = {"position": agent.pos.tolist(), "rotation_wxyz": agent.quat_wxyz()}
        if queue:
            a = queue.pop(0)
            if a == STOP:
                queue = []
                continue
            actions.append({"step_before": step, "action": a, "phase": "local_action", "system2_call_index": n_call - 1})
            agent.act(a)
            step += 1
            continue
        kind, text, acts = chunks[n_call]
        hist = list(range(step))
        if len(hist) > 8:
            hist = [hist[int(i)] for i in np.linspace(0, len(hist) - 1, 8, dtype=np.int64)]
        calls.append({"call_index": n_call, "step": step, "kind": kind, "text": text, "actions": acts,
                      "history_steps": hist, "ready": kind == "trajectory" and step >= READY_FROM_STEP})
        n_call += 1
        if kind == "stop":
            actions.append({"step_before": step, "action": STOP, "phase": "terminal", "system2_call_index": n_call - 1})
            step += 1
            done = True
            continue
        actions.append({"step_before": step, "action": acts[0], "phase": "rpc_first", "system2_call_index": n_call - 1})
        agent.act(acts[0])
        step += 1
        queue = list(acts[1:])
    final = {"position": agent.pos.tolist(), "rotation_wxyz": agent.quat_wxyz()}
    return {"states": states, "actions": actions, "calls": calls, "steps": step, "final": final}


def fake_history_head(gmap, gvis, mask, rng):
    """Predictions near the GT labels: (logits [8,4], sigmoid maps [8,4,64,64], gated, none [8], peaks [8,4,2])."""
    cls, row, col = gt.gt_view_class_and_peak(gmap, gvis)
    logits = np.full((8, 4), -3.0, dtype=np.float32)
    maps = np.full((8, 4, 64, 64), 0.01, dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    for k in np.nonzero(mask)[0]:
        if cls[k] == 0:
            continue
        v, r, c = int(cls[k] - 1), int(row[k]), int(col[k])
        u = rng.random()
        if u < 0.15:
            r, c = min(r + 10, 63), c  # right view, 10 px off
        elif u < 0.25:
            v = (v + 2) % 4  # wrong view
        else:
            r, c = int(np.clip(r + rng.integers(-3, 4), 0, 63)), int(np.clip(c + rng.integers(-3, 4), 0, 63))
        logits[k, v] = 3.0
        maps[k, v] = np.maximum(maps[k, v], np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 3.0 ** 2)) * 0.95)
    p = np.exp(np.concatenate([np.zeros((8, 1)), logits], axis=1))
    p = p / p.sum(1, keepdims=True)
    soft = maps / maps.reshape(8, 4, -1).sum(-1)[..., None, None]
    gated = (soft * p[:, 1:, None, None]).astype(np.float32)
    none = p[:, 0].astype(np.float32)
    none[~mask] = 1.0
    ry, rx = np.unravel_index(maps.reshape(8, 4, -1).argmax(-1), (64, 64))
    return logits, maps, gated, none, np.stack([ry, rx], -1).astype(np.int16)


def write_episode(root: Path, run: str, spec: dict, sim: dict, conv: str, rng: np.random.Generator) -> dict:
    key = ep_key(spec["scene_id"], spec["episode_id"])
    trace_dir = root / "runs" / run / "gpu0" / "trace" / key
    steps_dir = root / "runs" / run / "gpu0" / "steps" / key
    trace_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    states = sim["states"]
    cam = {s: gt.camera_c2w(st["position"], st["rotation_wxyz"]) for s, st in states.items()}
    positions = np.asarray([states[s]["position"] for s in sorted(states)])

    # goal: A -> 1.2 m from the final position; B -> 1.5 m left of a mid-route state, far from the end
    if spec["category"] == "T1":
        goal = np.asarray(sim["final"]["position"]) + np.array([1.0, 0.0, 0.6])
    else:
        mid = sorted(states)[len(states) // 2]
        yaw = 2 * math.atan2(states[mid]["rotation_wxyz"][2], states[mid]["rotation_wxyz"][0])
        goal = np.asarray(states[mid]["position"]) + 1.5 * np.array([-math.cos(yaw), 0.0, math.sin(yaw)])
    dists = np.linalg.norm(positions - goal, axis=1)
    final_d = float(np.linalg.norm(np.asarray(sim["final"]["position"]) - goal))
    success, os_ = float(final_d <= 3.0), float(min(dists.min(), final_d) <= 3.0)
    if spec["category"] == "F1":
        assert os_ == 1.0 and success == 0.0, (dists.min(), final_d)

    ref_calls = []
    for c in sim["calls"]:
        idx, step = c["call_index"], c["step"]
        path = chunk_path(c["actions"]) if c["kind"] == "trajectory" else None
        arrays = {}
        for i, v in enumerate(("front", "right", "back", "left")):
            arrays[f"jpeg__current__{v}"] = jpeg(pattern(384, 384, step * 4 + i))
        for i, s in enumerate(c["history_steps"]):
            arrays[f"jpeg__history__{i}__front"] = jpeg(pattern(384, 384, s * 4))
        arrays["jpeg__lookdown"] = jpeg(pattern(480, 640, step + 100))
        lookdown_turns, first, text, pixel_goal = 0, c["text"], c["text"], None
        if path is not None:
            lookdown_turns = int(idx % 2 == 0)
            wh = gt.DECISION_IMAGES["lookdown" if lookdown_turns else "front"]["wh"]
            pitch = gt.DECISION_IMAGES["lookdown" if lookdown_turns else "front"]["pitch_deg"]
            u, v = np.nan_to_num(gt.project_to_decision_image(gt.path_camera_points(path[-1:], pitch), wh)[0],
                                 nan=wh[0] / 2)
            u, v = int(np.clip(np.rint(u), 0, wh[0] - 1)), int(np.clip(np.rint(v), 0, wh[1] - 1))
            text = f"{u} {v}" if conv == "field_vu" else f"{v} {u}"
            first = "↓" if lookdown_turns else text
            nums = [int(x) for x in text.split()]
            pixel_goal = [nums[1], nums[0]]  # rpc_model_server._parse_internnav_pixel_goal
            arrays["trajectory_raw"] = np.repeat(gt.path_xy_to_action_deltas(path)[None], 32, axis=0)
            arrays["selected_path_xy"] = path
        diag = {"diagnostic_only": True, "enabled": True, "skipped_reason": None, "bridge_attention_available": False,
                "bridge_attention_check": None, "counterfactual_no_memory": None, "replay_same_plan": None, "errors": []}
        if c["ready"]:
            hsteps = c["history_steps"]
            gmap, gvis = gt.history_labels([cam[s] for s in hsteps], cam[step], np.full((256, 256), DEPTH_M, np.float16))
            gmap, gvis, mask = gt.pad_slots(gmap, gvis)
            logits, maps, gated, none, peaks = fake_history_head(gmap, gvis, mask, rng)
            fut = gt.future_reference_from_path(path)
            fvis = np.full((4, 4), 0.1, dtype=np.float32)
            for b in range(4):
                if fut.view5[b] > 0:
                    fvis[b, (fut.view5[b] - 1) if rng.random() < 0.8 else 0] = 0.85
            arrays.update(hist_heatmaps_gated=gated.astype(np.float16), hist_heatmaps=maps.astype(np.float16),
                          hist_view_peak_yx=peaks, hist_visibility_logits=logits, hist_none_probability=none,
                          hist_mask=mask, plan_z0=np.zeros((4, 768), np.float32), plan_z=np.zeros((4, 768), np.float32),
                          delta_token_ratio=np.full(4, 0.02, np.float32),
                          fut_heatmaps_gated=(fut.heatmap * fvis[..., None, None]).astype(np.float16),
                          fut_heatmaps=fut.heatmap.astype(np.float16), fut_visibility_probability=fvis)
            changed = idx % 4 == 0
            cf_actions = ([3 if a == 2 else 2 if a == 3 else a for a in c["actions"]] if changed and
                          any(a in (2, 3) for a in c["actions"]) else [1, 1, 1, 1] if changed else list(c["actions"]))
            cf_path = chunk_path(cf_actions)
            arrays["cf_selected_path_xy"] = cf_path
            diag["counterfactual_no_memory"] = {
                "plan": "plan_z0", "generator_seed": 1234 + idx, "actions": cf_actions, "anti_deadlock": False,
                "actions_changed": cf_actions != c["actions"],
                "endpoint_shift_m": float(np.linalg.norm(cf_path[-1] - path[-1])), "selected_path_xy": cf_path.tolist()}
            diag["replay_same_plan"] = {"plan": "plan_z", "generator_seed": 1234 + idx, "bitwise_equal": True,
                                        "max_abs_diff_raw": 0.0, "actions": list(c["actions"]), "actions_equal": True,
                                        "endpoint_shift_m": 0.0}
        else:
            diag["skipped_reason"] = "the bridge did not run on this call"
        response = {"ok": True, "llm_output": text, "native_first_output": first, "native_lookdown_turns": lookdown_turns,
                    "kind": c["kind"], "actions": list(c["actions"]), "terminal": c["kind"] == "stop",
                    "pose_ready": step >= READY_FROM_STEP, "ppa_applied": bool(c["ready"]) if c["kind"] == "trajectory" else None}
        if pixel_goal is not None:
            response.update(pixel_goal=pixel_goal, pano_goal_view="front",
                            trajectory_summary=f"traj_goal=({path[-1, 0]:.2f},{path[-1, 1]:.2f})")
        request = {"instruction": spec["instruction"], "num_history": len(c["history_steps"]),
                   "current_capture_step": step, "history_capture_steps": c["history_steps"],
                   "history_age_steps": [step - s for s in c["history_steps"]], "pose_ready": step >= READY_FROM_STEP,
                   "trajectory_selection": "mean", "trajectory_x_sign": 1.0, "phase": "joint",
                   "deterministic_sampling": {"protocol_seed": 42, "scene_id": spec["scene_id"],
                                              "episode_id": spec["episode_id"], "system2_call_index": idx,
                                              "per_call_seed": 1234 + idx}}
        record = {
            "schema": "exp19-call-trace-v1", "synthetic": True, "scene_id": spec["scene_id"], "episode_id": spec["episode_id"],
            "ep_key": key, "system2_call_index": idx, "per_call_seed": 1234 + idx, "protocol_seed": 42,
            "current_capture_step": step, "history_capture_steps": c["history_steps"],
            "history_age_steps": request["history_age_steps"], "vo_current_frame_id": step,
            "vo_history_frame_ids": c["history_steps"], "pose_ready": step >= READY_FROM_STEP,
            "vo_provider_phase": "stateful_backend" if step >= READY_FROM_STEP else "map_warmup",
            "instruction": spec["instruction"], "request": request, "blob_names": sorted(k for k in arrays if k.startswith("jpeg__")),
            "response": response,
            "trajectory_path": ("ppa" if c["ready"] else "warmup") if c["kind"] == "trajectory" else None,
            "has_past_output": bool(c["ready"]), "has_future_output": bool(c["ready"]),
            "selected_path_xy": path.tolist() if path is not None else None,
            "recomputed_actions": list(c["actions"]) if path is not None else None,
            "recomputed_anti_deadlock": False if path is not None else None,
            "actions_match": True if path is not None else None, "checks": {}, "trace_warnings": [],
            "diagnostic": diag, "shapes": {k: list(v.shape) for k, v in arrays.items()}, "timing_s": {},
        }
        np.savez_compressed(trace_dir / f"call_{idx:03d}.npz", **arrays)
        (trace_dir / f"call_{idx:03d}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        ref_text = text if spec["diverge_at"] != idx else "123 45"
        ref_calls.append({"call_index": idx, "step": step, "kind": c["kind"], "vlm_output": ref_text,
                          "actions": list(c["actions"]), "traj_goal": path[-1].tolist() if path is not None else None,
                          "vo_frame": step, "vo_history": c["history_steps"], "vo_ready": step >= READY_FROM_STEP})

    # steps.jsonl + native fronts
    lines = [{"type": "episode_start", "schema": "exp19-step-trace-v1", "scene_id": spec["scene_id"],
              "episode_id": spec["episode_id"], "ep_key": key, "instruction": spec["instruction"],
              "instruction_raw": spec["instruction"] + ".", "start_position": states[0]["position"],
              "start_rotation_wxyz": states[0]["rotation_wxyz"], "goal_position": goal.tolist(), "goal_radius": 3.0,
              "reference_path": [states[0]["position"], positions[len(positions) // 2].tolist(), goal.tolist()],
              "geodesic_distance": float(np.linalg.norm(goal - positions[0])),
              "sensor": {"rgb_hw": [480, 640], "hfov_deg": 79.0, "sensor_height_m": 1.25}}]
    by_step = {}
    for a in sim["actions"]:
        by_step.setdefault(a["step_before"], []).append(a)
    all_states = dict(states)
    all_states[sim["steps"]] = sim["final"]
    for s in sorted(all_states):
        name = f"front_{s:04d}.jpg"
        Image.fromarray(pattern(480, 640, s)).save(steps_dir / name, "JPEG", quality=90)
        sensor = gt.camera_c2w(all_states[s]["position"], all_states[s]["rotation_wxyz"])  # level, 1.25 m up
        lines.append({"type": "state", "step": s, "position": all_states[s]["position"],
                      "rotation_wxyz": all_states[s]["rotation_wxyz"], "rgb_sensor_position": sensor[:3, 3].tolist(),
                      "rgb_sensor_rotation_wxyz": all_states[s]["rotation_wxyz"],
                      "vo_frame_id": s if s in states else None, "queue_len": 0, "front_jpg": name,
                      "final": s not in states})
        lines += [dict(type="action", **a) for a in by_step.get(s, [])]
    metrics = {"distance_to_goal": final_d, "success": success, "spl": success * 0.8, "oracle_success": os_,
               "oracle_navigation_error": float(dists.min())}
    lines.append({"type": "episode_end", "steps": sim["steps"], "metrics": metrics, "ended_by": "stop",
                  "stop_phase": "terminal", "done": True, "max_steps": STEP_CAP, "vlm_calls": len(sim["calls"]),
                  "num_states": len(all_states), "num_actions": len(sim["actions"])})
    with open(steps_dir / "steps.jsonl", "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # renders at every call step + the final state
    rsteps = sorted({c["step"] for c in sim["calls"]} | {sim["steps"]})
    rcam = {**cam, sim["steps"]: gt.camera_c2w(sim["final"]["position"], sim["final"]["rotation_wxyz"])}
    (root / "renders").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "renders" / f"{key}.npz", steps=np.asarray(rsteps, np.int32),
                        rgb=np.stack([np.stack([pattern(256, 256, s * 4 + v) for v in range(4)]) for s in rsteps]),
                        depth_front=np.full((len(rsteps), 256, 256), DEPTH_M, np.float16),
                        c2w_front=np.stack([rcam[s] for s in rsteps]))
    (root / "renders" / f"{key}.json").write_text(json.dumps({"schema": "exp19-renders-v1", "synthetic": True}) + "\n")

    ref_dir = root / "cases" / "eval_log_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    final = {"success": success, "spl": success * 0.8, "os": os_, "ne": final_d, "vlm_calls": len(sim["calls"]),
             "trajectory_calls": sum(c["kind"] == "trajectory" for c in sim["calls"]), "steps": sim["steps"],
             "ended_by": "stop"}
    (ref_dir / f"{key}.json").write_text(json.dumps({
        "schema": "exp19-eval-log-ref-v1", "ep_key": key, "scene_id": spec["scene_id"], "episode_id": spec["episode_id"],
        "client_log": "synthetic", "calls": ref_calls, "final": final}, indent=1) + "\n")
    return {"rank": 0, "category": spec["category"], "ep_key": key, "scene_id": spec["scene_id"],
            "episode_id": spec["episode_id"], "steps_42": sim["steps"], "steps_1337": sim["steps"],
            "success_42": success, "os_42": os_, "ne_42": final_d, "ended_by_42": "stop",
            "ppa_applied_42": sum(c["ready"] for c in sim["calls"]), "ppa_applied_1337": sum(c["ready"] for c in sim["calls"]),
            "geodesic_m": float(np.linalg.norm(goal - positions[0])), "dy_m": 0.0, "ref_turns45": 2, "n_room_runs": 3,
            "sort_key": [0, "0" * 40], "eval_log": {"client_log": "synthetic", "line_start": 0}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", default="synth")
    p.add_argument("--pixel-goal-convention", choices=("field_vu", "field_uv"), default="field_vu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cands = {cat: [] for cat in ("T1", "T2", "T3", "F1", "F2")}
    for name, spec in EPISODES.items():
        sim = simulate(spec)
        cands[spec["category"]].append(write_episode(args.root, args.run, spec, sim, args.pixel_goal_convention, rng))
        print(f"episode {name}: {sim['steps']} steps, {len(sim['calls'])} calls, "
              f"{sum(c['ready'] for c in sim['calls'])} ready", flush=True)
    keys = [c["ep_key"] for cat in cands.values() for c in cat]
    (args.root / "cases").mkdir(parents=True, exist_ok=True)
    (args.root / "cases" / "candidates.json").write_text(json.dumps({
        "schema": "exp19-candidates-v1", "git_sha": None, "synthetic": True, "inputs": {}, "definitions": {},
        "categories": {cat: {"pool_size": len(c), "median_steps_42": None, "ordered_pool_head": [], "candidates": c}
                       for cat, c in cands.items()},
        "episode_lists": {"gpu0": keys}}, indent=1) + "\n")
    (args.root / "runs" / args.run / "DONE").write_text("synthetic\n")
    print(f"wrote {args.root} (run {args.run}, pixel goals {args.pixel_goal_convention})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
