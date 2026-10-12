#!/usr/bin/env python3
"""Write oracle labels for recorded System2 calls (EXP-23 part B).

For every committed ``*.s2rec.tar`` under ``--record_root`` this writes
``<label_root>/<scene>/<scene>_<episode:06d>.s2labels.jsonl``: one row per
System2 call (schema ``heatmapvln-s2-labels-v1``) with the recomputed geodesic
distance to the goal, the oracle's answer, the model's answer, whether they
agree, and the SFT target (the model's own text when it agrees, else the
oracle's).  The rules live in ``src/data/s2_oracle_labels.py``; this script
only supplies Habitat geometry:

* geodesics and paths come from ``habitat_sim.PathFinder`` on the scene's
  ``<scenes_dir>/<scene>/<scene>.navmesh`` -- the navmesh the simulator loads
  next to the ``.glb`` -- so nothing is rendered and no GPU is needed;
* occlusion uses the lookdown depth stored in the record.

The reference path and goal come from ``--episodes`` (the R2R-CE split file the
collector ran) when given, else from the record's ``episode.json``; with both,
they must agree.

Labelling is idempotent: a label file whose rows carry the record's sha256 and
this labeller's config digest is left alone unless ``--force``.  Each file is
written to a temp name and renamed.  A summary of all label files of the
processed records goes to ``--summary`` (default ``<label_root>/label_summary.json``).

``--viz_dir`` additionally draws, for every ``--viz_every``-th call, the
oracle route samples (green visible / red occluded), the oracle pixel (cyan
cross) and the model's pixel (yellow circle) on the recorded lookdown.

Example (4090 container)::

    /opt/conda/bin/python scripts/tools/label_s2_records.py \\
        --record_root /workspace/exp23/records --label_root /workspace/exp23/labels \\
        --episodes /workspace/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \\
        --scenes_dir /dataset/mp3d --workers 16
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SUMMARY_SCHEMA = "heatmapvln-s2-label-summary-v1"


def _s2_modules():
    """(s2_records, s2_oracle_labels) without running ``src/data/__init__.py``.

    That ``__init__`` imports torch and the training datasets; the labeller
    needs numpy and PIL only.  Reuse the real package when it is already
    imported (tests), so classes are shared.
    """
    if "src.data" in sys.modules:
        return (
            importlib.import_module("src.data.s2_records"),
            importlib.import_module("src.data.s2_oracle_labels"),
        )
    name = "_s2_label_code"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(SOURCE_ROOT / "src" / "data")]
        pkg.__package__ = name
        sys.modules[name] = pkg
    return (
        importlib.import_module(f"{name}.s2_records"),
        importlib.import_module(f"{name}.s2_oracle_labels"),
    )


REC, ORL = _s2_modules()


# --------------------------------------------------------------------------- #
# Episode geometry
# --------------------------------------------------------------------------- #
def load_episode_index(paths: Sequence[str | os.PathLike]) -> dict[tuple[str, int], dict[str, Any]]:
    """(scene, episode_id) -> reference path / goal / start from R2R-CE split files."""
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        for ep in data["episodes"]:
            scene = Path(str(ep["scene_id"])).stem
            index[(scene, int(ep["episode_id"]))] = {
                "reference_path": ep["reference_path"],
                "goal": ep["goals"][0]["position"],
                "start_position": ep.get("start_position"),
            }
    return index


def _episode_goal(episode: Mapping[str, Any]) -> Any:
    for key in ("goal_position", "goal"):
        value = episode.get(key)
        if isinstance(value, Mapping):
            value = value.get("position")
        if value is not None:
            return value
    goals = episode.get("goals")
    if goals:
        first = goals[0]
        return first.get("position") if isinstance(first, Mapping) else first
    return None


def resolve_episode_geometry(
    episode: Mapping[str, Any],
    dataset_entry: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(reference path [N, 3], goal [3], start position [3] or None)."""
    ref = episode.get("reference_path")
    goal = _episode_goal(episode)
    start = episode.get("start_position")
    if dataset_entry is not None:
        for name, mine, theirs in (
            ("reference_path", ref, dataset_entry["reference_path"]),
            ("goal", goal, dataset_entry["goal"]),
        ):
            if mine is not None and not np.allclose(np.asarray(mine, float), np.asarray(theirs, float), atol=1e-3):
                raise ValueError(f"episode.json {name} disagrees with the split file")
        ref, goal = dataset_entry["reference_path"], dataset_entry["goal"]
        start = start if start is not None else dataset_entry.get("start_position")
    if ref is None or goal is None:
        raise ValueError("no reference path / goal: pass --episodes or record them in episode.json")
    start_arr = None if start is None else np.asarray(start, dtype=np.float64).reshape(3)
    return np.asarray(ref, dtype=np.float64).reshape(-1, 3), np.asarray(goal, dtype=np.float64).reshape(3), start_arr


def call_pose(record: Any, call: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Agent pose at the call: the call row's own, else that of its current-frame step."""
    if call.get("position") is not None and call.get("rotation_wxyz") is not None:
        return list(call["position"]), list(call["rotation_wxyz"])
    step = record.step(int(call["step_id"]))
    return list(step["position"]), list(step["rotation_wxyz"])


# --------------------------------------------------------------------------- #
# Navmesh
# --------------------------------------------------------------------------- #
class NavmeshGeometry:
    """Geodesics and paths on one scene's navmesh (habitat_sim.PathFinder, no renderer)."""

    def __init__(self, navmesh_path: str | os.PathLike):
        import habitat_sim  # deferred: only the labelling process needs it

        self._habitat_sim = habitat_sim
        self.path = Path(navmesh_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"navmesh not found: {self.path}")
        self.pathfinder = habitat_sim.PathFinder()
        self.pathfinder.load_nav_mesh(str(self.path))
        if not self.pathfinder.is_loaded:
            raise RuntimeError(f"could not load navmesh {self.path}")

    def _query(self, a: Sequence[float], b: Sequence[float]):
        sp = self._habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, dtype=np.float32)
        sp.requested_end = np.asarray(b, dtype=np.float32)
        found = self.pathfinder.find_path(sp)
        return found, sp

    def geodesic(self, a: Sequence[float], b: Sequence[float]) -> float:
        found, sp = self._query(a, b)
        dist = float(sp.geodesic_distance)
        return dist if found and math.isfinite(dist) else math.inf

    def find_path(self, a: Sequence[float], b: Sequence[float]) -> np.ndarray | None:
        found, sp = self._query(a, b)
        if not found or not math.isfinite(float(sp.geodesic_distance)):
            return None
        points = np.asarray([np.asarray(p, dtype=np.float64) for p in sp.points]).reshape(-1, 3)
        return points if len(points) else None


# --------------------------------------------------------------------------- #
# One record
# --------------------------------------------------------------------------- #
def _sidecar_digest(record_path: Path) -> str:
    sidecar = record_path.with_name(record_path.name + ".sha256")
    return sidecar.read_text(encoding="utf-8").split()[0]


def _existing_is_current(label_path: Path, digest: str, config_digest: str, num_calls: int) -> bool:
    if not label_path.is_file():
        return False
    try:
        rows = REC.read_labels(label_path)
    except (ValueError, json.JSONDecodeError):
        return False
    return len(rows) == num_calls and all(
        r.get("record_sha256") == digest and r.get("labeler") == config_digest for r in rows
    )


def label_record_rows(
    record: Any,
    geometry: Any,
    *,
    dataset_entry: Mapping[str, Any] | None,
    config: Any,
    camera: Any = None,
    on_call: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    """Label rows of every call of ``record``.  ``geometry`` has ``geodesic`` and ``find_path``."""
    camera = camera if camera is not None else ORL.LookdownCamera()
    ref, goal, start = resolve_episode_geometry(record.episode, dataset_entry)
    waypoints = ORL.route_waypoints(ref, goal)
    remaining = ORL.remaining_reference_m(waypoints, geometry.geodesic)
    segments = [geometry.find_path(waypoints[k], waypoints[k + 1]) for k in range(len(waypoints) - 1)]

    steps = sorted(record.steps, key=lambda s: int(s["step_id"]))
    step_ids = [int(s["step_id"]) for s in steps]
    positions = [s["position"] for s in steps]
    if start is not None:
        step_ids, positions = [-1] + step_ids, [start] + positions
    reached = ORL.last_reached_by_step(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), waypoints, geometry.geodesic, config.reached_radius_m
    )

    rows = []
    for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
        index = int(call["call_index"])
        position, rotation = call_pose(record, call)
        p = np.asarray(position, dtype=np.float64)
        upto = int(np.searchsorted(step_ids, int(call["step_id"]), side="right")) - 1
        last_reached = max(
            int(reached[upto]) if upto >= 0 else -1,
            int(ORL.last_reached_by_step(p[None], waypoints, geometry.geodesic, config.reached_radius_m)[0]),
        )
        d = geometry.geodesic(p, goal)
        route = None
        k_star, costs = (None, {})
        if math.isfinite(d):
            k_star, costs = ORL.choose_waypoint(
                p, waypoints, last_reached, remaining, geometry.geodesic, slack_m=config.rejoin_slack_m
            )
            if k_star is not None:
                route = ORL.route_points(p, waypoints, k_star, geometry.find_path, segments)
        depth = record.lookdown_depth(index) if record.has_lookdown_depth(index) else None
        model = ORL.parse_model_answer(call)
        label = ORL.label_call(
            model=model,
            position=position,
            rotation_wxyz=rotation,
            distance_to_goal=d,
            route=route,
            depth_m=depth,
            config=config,
            camera=camera,
        )
        label["debug"].update(
            {
                "position": ORL.rounded(position),
                "last_reached": last_reached,
                "k_star": k_star,
                "num_waypoints": int(len(waypoints)),
                "waypoint_costs_m": {str(k): ORL.rounded(c) for k, c in costs.items()},
            }
        )
        recorded = call.get("distance_to_goal")
        row = {
            "schema": REC.LABEL_SCHEMA,
            "scene_id": record.scene_id,
            "episode_id": record.episode_id,
            "call_index": index,
            "step_id": int(call["step_id"]),
            "recorded_kind": call.get("kind"),
            "recorded_distance_to_goal": None if recorded is None else ORL.rounded(recorded),
            **label,
        }
        rows.append(row)
        if on_call is not None:
            on_call(call=call, row=row, route=route, depth=depth, position=position, rotation=rotation)
    return rows


def _draw_call(record: Any, out_dir: Path, *, call, row, route, depth, position, rotation, config, camera) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(record.lookdown_jpeg(int(call["call_index"])))).convert("RGB")
    draw = ImageDraw.Draw(image)
    if route is not None and depth is not None:
        vis = ORL.visible_route_samples(route, camera.c2w(position, rotation), depth, camera, config)
        for c, r, ok in zip(vis["col"], vis["row"], vis["visible"]):
            if 0 <= c < camera.width and 0 <= r < camera.height:
                draw.ellipse([c - 2, r - 2, c + 2, r + 2], fill=(0, 220, 0) if ok else (230, 30, 30))
    if row["oracle_pixel"] is not None:
        u, v = row["oracle_pixel"]
        draw.line([u - 10, v, u + 10, v], fill=(0, 255, 255), width=3)
        draw.line([u, v - 10, u, v + 10], fill=(0, 255, 255), width=3)
    if row["model_pixel"] is not None:
        u, v = row["model_pixel"]
        draw.ellipse([u - 8, v - 8, u + 8, v + 8], outline=(255, 230, 0), width=3)
    text = (
        f"call {row['call_index']} d={row['distance_to_goal']} oracle={row['oracle_kind']} "
        f"{row['oracle_turn1']}{' ' + row['oracle_turn2'] if row['oracle_turn2'] else ''} | "
        f"model={row['model_kind']} agree={row['agree']} ({row['reason']})"
    )
    draw.rectangle([0, 0, camera.width, 16], fill=(0, 0, 0))
    # PIL's default font has no arrow glyphs.
    draw.text((4, 2), text.translate(str.maketrans({"←": "<", "→": ">", "↓": "v", "↑": "^"})), fill=(255, 255, 255))
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{record.scene_id}_{record.episode_id:06d}_call{int(call['call_index']):04d}.png")


def label_record(
    record_path: str | os.PathLike,
    *,
    label_root: str | os.PathLike,
    geometry_for_scene: Callable[[str], Any],
    dataset_index: Mapping[tuple[str, int], Mapping[str, Any]] | None,
    config: Any,
    force: bool = False,
    viz_dir: str | os.PathLike | None = None,
    viz_every: int = 0,
) -> dict[str, Any]:
    """Label one record; returns ``{"record", "status", ...}`` (status labelled / exists)."""
    record_path = Path(record_path)
    record = REC.EpisodeRecord(record_path)
    digest = _sidecar_digest(record_path)
    out = REC.label_path_for(record_path, label_root)
    config_digest = config.digest()
    if not force and _existing_is_current(out, digest, config_digest, len(record.calls)):
        return {"record": str(record_path), "labels": str(out), "status": "exists"}
    entry = None
    if dataset_index is not None:
        entry = dataset_index.get((record.scene_id, record.episode_id))
        if entry is None:
            raise KeyError(f"{record.scene_id}/{record.episode_id} is not in --episodes")
    camera = ORL.LookdownCamera()
    on_call = None
    if viz_dir is not None and viz_every > 0:

        def on_call(**kwargs):
            if int(kwargs["call"]["call_index"]) % viz_every == 0:
                _draw_call(record, Path(viz_dir), config=config, camera=camera, **kwargs)

    rows = label_record_rows(
        record,
        geometry_for_scene(record.scene_id),
        dataset_entry=entry,
        config=config,
        camera=camera,
        on_call=on_call,
    )
    for row in rows:
        row["record_sha256"] = digest
        row["labeler"] = config_digest
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(tmp, out)
    return {"record": str(record_path), "labels": str(out), "status": "labelled", "calls": len(rows)}


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _target_kind(row: Mapping[str, Any]) -> str:
    t1, t2 = row.get("target_turn1"), row.get("target_turn2")
    if t1 is None:
        return "none"
    if t1.strip() == ORL.STOP_TEXT:
        return "stop"
    if t2 is not None and any(ch.isdigit() for ch in t2):
        return "pixel"
    if t2 is None and any(ch.isdigit() for ch in t1):
        return "pixel"
    return "turn" if set(ORL.parse_native_tokens(t2 if t2 is not None else t1)) <= {"←", "→"} else "other"


def summarize_labels(label_paths: Sequence[Path]) -> dict[str, Any]:
    by_oracle = collections.Counter()
    by_model = collections.Counter()
    by_target = collections.Counter()
    reasons = collections.Counter()
    agree_by_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    pair = collections.Counter()
    near_goal = collections.Counter()
    n = 0
    for path in label_paths:
        for row in REC.read_labels(path):
            n += 1
            by_oracle[row["oracle_kind"]] += 1
            by_model[row["model_kind"]] += 1
            by_target[_target_kind(row)] += 1
            reasons[row["reason"].split("|")[0]] += 1
            agree_by_model[row["model_kind"]][str(row["agree"]).lower()] += 1
            pair[f"{row['model_kind']}->{row['oracle_kind']}"] += 1
            d = row.get("distance_to_goal")
            if d is not None:
                near_goal["d<=2" if d <= 2 else "2<d<=3" if d <= 3 else "d>3"] += 1
    agreed = sum(c["true"] for c in agree_by_model.values())
    judged = sum(c["true"] + c["false"] for c in agree_by_model.values())
    return {
        "num_label_files": len(label_paths),
        "num_calls": n,
        "agree_rate_judged": round(agreed / judged, 4) if judged else None,
        "by_oracle_kind": dict(by_oracle),
        "by_model_kind": dict(by_model),
        "by_target_kind": dict(by_target),
        "agree_by_model_kind": {k: dict(v) for k, v in agree_by_model.items()},
        "model_to_oracle": dict(pair),
        "distance_bins": dict(near_goal),
        "reasons": dict(reasons.most_common()),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _worker_init(state: dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER.update(state)
    _WORKER["navmeshes"] = {}


def _geometry_for_scene(scene_id: str) -> NavmeshGeometry:
    cache = _WORKER["navmeshes"]
    if scene_id not in cache:
        cache[scene_id] = NavmeshGeometry(Path(_WORKER["scenes_dir"]) / scene_id / f"{scene_id}.navmesh")
    return cache[scene_id]


def _worker_label(record_path: str) -> dict[str, Any]:
    try:
        return label_record(
            record_path,
            label_root=_WORKER["label_root"],
            geometry_for_scene=_geometry_for_scene,
            dataset_index=_WORKER["dataset_index"],
            config=_WORKER["config"],
            force=_WORKER["force"],
            viz_dir=_WORKER["viz_dir"],
            viz_every=_WORKER["viz_every"],
        )
    except Exception as err:  # noqa: BLE001 -- one bad record must not stop the run
        return {
            "record": record_path,
            "status": "failed",
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(limit=4),
        }


def build_config(args: argparse.Namespace) -> Any:
    return ORL.OracleConfig(
        stop_radius_m=args.stop_radius_m,
        success_radius_m=args.success_radius_m,
        reached_radius_m=args.reached_radius_m,
        rejoin_slack_m=args.rejoin_slack_m,
        sample_step_m=args.sample_step_m,
        occlusion_tol_m=args.occlusion_tol_m,
        min_pixel_ahead_m=args.min_pixel_ahead_m,
        turn_target_ahead_m=args.turn_target_ahead_m,
        agree_path_dist_m=args.agree_path_dist_m,
        agree_min_progress_m=args.agree_min_progress_m,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = ORL.OracleConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--scenes_dir", default="/dataset/mp3d", help="holds <scene>/<scene>.navmesh")
    parser.add_argument("--episodes", nargs="*", default=[], help="R2R-CE split json.gz (authoritative path/goal)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="relabel even when current labels exist")
    parser.add_argument("--limit", type=int, default=0, help="label at most this many records (0 = all)")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--viz_dir", default=None)
    parser.add_argument("--viz_every", type=int, default=0)
    parser.add_argument("--stop_radius_m", type=float, default=defaults.stop_radius_m)
    parser.add_argument("--success_radius_m", type=float, default=defaults.success_radius_m)
    parser.add_argument("--reached_radius_m", type=float, default=defaults.reached_radius_m)
    parser.add_argument("--rejoin_slack_m", type=float, default=defaults.rejoin_slack_m)
    parser.add_argument("--sample_step_m", type=float, default=defaults.sample_step_m)
    parser.add_argument("--occlusion_tol_m", type=float, default=defaults.occlusion_tol_m)
    parser.add_argument("--min_pixel_ahead_m", type=float, default=defaults.min_pixel_ahead_m)
    parser.add_argument("--turn_target_ahead_m", type=float, default=defaults.turn_target_ahead_m)
    parser.add_argument("--agree_path_dist_m", type=float, default=defaults.agree_path_dist_m)
    parser.add_argument("--agree_min_progress_m", type=float, default=defaults.agree_min_progress_m)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    records = [str(p) for p in REC.iter_record_paths(args.record_root)]
    if args.limit > 0:
        records = records[: args.limit]
    dataset_index = load_episode_index(args.episodes) if args.episodes else None
    state = {
        "label_root": args.label_root,
        "scenes_dir": args.scenes_dir,
        "dataset_index": dataset_index,
        "config": config,
        "force": bool(args.force),
        "viz_dir": args.viz_dir,
        "viz_every": int(args.viz_every),
    }
    started = time.time()
    results: list[dict[str, Any]] = []
    # Records are sorted by scene, so each worker mostly reuses its loaded navmesh.
    if args.workers <= 1 or len(records) <= 1:
        _worker_init(state)
        results = [_worker_label(r) for r in records]
    else:
        with mp.get_context("fork").Pool(args.workers, initializer=_worker_init, initargs=(state,)) as pool:
            for result in pool.imap_unordered(_worker_label, records, chunksize=4):
                results.append(result)
                if len(results) % 200 == 0:
                    print(f"[label-s2] {len(results)}/{len(records)} records", flush=True)
    status = collections.Counter(r["status"] for r in results)
    failures = [r for r in results if r["status"] == "failed"]
    for failure in failures[:10]:
        print(f"[label-s2] FAILED {failure['record']}: {failure['error']}", file=sys.stderr, flush=True)
    label_paths = [Path(r["labels"]) for r in results if r["status"] != "failed"]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "label_schema": REC.LABEL_SCHEMA,
        "record_root": str(args.record_root),
        "label_root": str(args.label_root),
        "episodes": list(args.episodes),
        "config": {**dataclasses.asdict(config), "digest": config.digest()},
        "num_records": len(records),
        "status": dict(status),
        "failures": [{k: f[k] for k in ("record", "error")} for f in failures],
        "elapsed_s": round(time.time() - started, 1),
        **summarize_labels(sorted(label_paths)),
    }
    summary_path = Path(args.summary) if args.summary else Path(args.label_root) / "label_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("num_records", "status", "num_calls", "agree_rate_judged")}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
