#!/usr/bin/env python3
"""Write EXP-25 line-B route-correction labels for recorded System2 calls.

For every committed ``*.s2rec.tar`` under ``--record_root`` this writes
``<label_root>/<scene>/<scene>_<episode:06d>.s2labels.jsonl``: one row per System2 call
(schema ``heatmapvln-s2-labels-v1``, so ``s2_sft_dataset.index_calls`` reads it) with the
geodesic distance to the goal, the row ``kind`` (``correction`` / ``anchor`` / ``none``),
the oracle's answer, the model's answer, whether they agree, why, and the SFT target.
States within 3 m of the goal are rows with ``skip`` (line A owns them); no row ever
targets STOP.  With ``--terminal_stop_radius_m`` > 0 (line C, EXP-35) those states are
labelled too: STOP inside that radius, line B's route to the goal between it and 3 m.
The rules live in ``src/data/s2_route_labels.py``; this script supplies
Habitat geometry (``habitat_sim.PathFinder`` on ``<scenes_dir>/<scene>/<scene>.navmesh``:
geodesics, paths, snapping -- no renderer, no GPU) and the recorded lookdown depth.

The reference path and goal come from the record's ``episode.json`` (EnvDrop records carry
them), or from ``--episodes`` (an R2R-CE-format split file) when given; with both, they
must agree.

Idempotent: a label file whose rows carry the record's sha256 and this labeller's config
digest is left alone unless ``--force``; files are written to a temp name and renamed.

The summary (``--summary``, default ``<label_root>/label_summary.json``) holds the label
statistics EXP-25 compares against InternData-N1 and round 1: pixel look-ahead and (u, v),
turn arrows and the flip rate against the previously executed turn, corrections by kind
and distance, agreement.  ``--compare_label_root`` adds the same statistics for another
labelling of the same records (round 1's ``label_s2_records.py`` rows are read too);
``--interndata_root`` adds InternData-N1's own distributions, measured from its parquet.

Example (4090 container, CPU only)::

    /opt/conda/bin/python scripts/tools/label_route_corrections.py \\
        --record_root /workspace/exp23/envdrop1/records \\
        --label_root /workspace/exp23/envdrop1/labels_routeB --workers 12 \\
        --interndata_root /workspace/exp23/data/interndata_n1/r2r
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.tools import label_s2_records as base  # noqa: E402  (numpy-only loader of src/data)

REC, ORL = base.REC, base.ORL
RL = importlib.import_module(ORL.__name__.rsplit(".", 1)[0] + ".s2_route_labels")

SUMMARY_SCHEMA = "heatmapvln-s2-route-label-summary-v1"
DISTANCE_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, math.inf))
QUANTILES = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------- #
# Navmesh
# --------------------------------------------------------------------------- #
class RouteNavmeshGeometry(base.NavmeshGeometry):
    """Round 1's navmesh geometry plus snapping a point onto the navmesh."""

    def snap(self, point: Sequence[float]) -> np.ndarray | None:
        snapped = np.asarray(self.pathfinder.snap_point(np.asarray(point, dtype=np.float32)), dtype=np.float64)
        return snapped if snapped.shape == (3,) and np.all(np.isfinite(snapped)) else None


# --------------------------------------------------------------------------- #
# One record
# --------------------------------------------------------------------------- #
def record_outcome(record: Any) -> bool | None:
    outcome = record.episode.get("outcome")
    if isinstance(outcome, Mapping) and outcome.get("success") is not None:
        return float(outcome["success"]) > 0.5
    return None


def label_record_rows(
    record: Any,
    geometry: Any,
    *,
    dataset_entry: Mapping[str, Any] | None,
    config: Any,
    camera: Any = None,
) -> list[dict[str, Any]]:
    """Line-B (or C) rows of every call of ``record``.  ``geometry`` has ``geodesic``, ``find_path``, ``snap``."""
    camera = camera if camera is not None else ORL.LookdownCamera()
    ref, goal, start = base.resolve_episode_geometry(record.episode, dataset_entry)
    waypoints = ORL.route_waypoints(ref, goal)
    remaining = segments = None  # only needed once a call gets the route oracle

    steps = sorted(record.steps, key=lambda s: int(s["step_id"]))
    step_ids = [int(s["step_id"]) for s in steps]
    positions = [s["position"] for s in steps]
    if start is not None:
        step_ids, positions = [-1] + step_ids, [start] + positions
    reached = ORL.last_reached_by_step(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), waypoints, geometry.geodesic, config.reached_radius_m
    )

    rows: list[dict[str, Any]] = []
    previous = None
    for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
        index = int(call["call_index"])
        position, rotation = base.call_pose(record, call)
        p = np.asarray(position, dtype=np.float64)
        d = geometry.geodesic(p, goal)
        model = ORL.parse_model_answer(call)
        route, depth, extra = None, None, {}
        if math.isfinite(d) and d > config.route_min_distance_m:
            if remaining is None:
                remaining = ORL.remaining_reference_m(waypoints, geometry.geodesic)
                segments = [geometry.find_path(waypoints[k], waypoints[k + 1]) for k in range(len(waypoints) - 1)]
            upto = int(np.searchsorted(step_ids, int(call["step_id"]), side="right")) - 1
            last_reached = max(
                int(reached[upto]) if upto >= 0 else -1,
                int(ORL.last_reached_by_step(p[None], waypoints, geometry.geodesic, config.reached_radius_m)[0]),
            )
            last_reached = RL.rejoin_after(last_reached, len(waypoints), d, config)
            k_star, _ = ORL.choose_waypoint(
                p, waypoints, last_reached, remaining, geometry.geodesic, slack_m=config.rejoin_slack_m
            )
            if k_star is not None:
                route = ORL.route_points(p, waypoints, k_star, geometry.find_path, segments)
            depth = record.lookdown_depth(index) if record.has_lookdown_depth(index) else None
            extra = {"position": ORL.rounded(position), "last_reached": last_reached, "k_star": k_star,
                     "num_waypoints": int(len(waypoints))}
        label = RL.label_route_call(
            model=model,
            previous_model=previous,
            position=position,
            rotation_wxyz=rotation,
            distance_to_goal=d,
            goal=goal,
            route=route,
            depth_m=depth,
            geodesic=geometry.geodesic,
            snap=geometry.snap,
            config=config,
            camera=camera,
        )
        label["debug"].update(extra)
        recorded = call.get("distance_to_goal")
        rows.append(
            {
                "schema": REC.LABEL_SCHEMA,
                "scene_id": record.scene_id,
                "episode_id": record.episode_id,
                "call_index": index,
                "step_id": int(call["step_id"]),
                "recorded_kind": call.get("kind"),
                "recorded_distance_to_goal": None if recorded is None else ORL.rounded(recorded),
                **label,
            }
        )
        previous = model
    return rows


def label_record(
    record_path: str | os.PathLike,
    *,
    label_root: str | os.PathLike,
    geometry_for_scene: Callable[[str], Any],
    dataset_index: Mapping[tuple[str, int], Mapping[str, Any]] | None,
    config: Any,
    force: bool = False,
) -> dict[str, Any]:
    """Label one record; returns ``{"record", "labels", "status", "success", ...}``."""
    record_path = Path(record_path)
    record = REC.EpisodeRecord(record_path)
    digest = base._sidecar_digest(record_path)
    out = REC.label_path_for(record_path, label_root)
    config_digest = config.digest()
    result = {"record": str(record_path), "labels": str(out), "success": record_outcome(record)}
    if not force and base._existing_is_current(out, digest, config_digest, len(record.calls)):
        return {**result, "status": "exists"}
    entry = None
    if dataset_index is not None:
        entry = dataset_index.get((record.scene_id, record.episode_id))
        if entry is None:
            raise KeyError(f"{record.scene_id}/{record.episode_id} is not in --episodes")
    rows = label_record_rows(record, geometry_for_scene(record.scene_id), dataset_entry=entry, config=config)
    for row in rows:
        row["record_sha256"] = digest
        row["labeler"] = config_digest
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, out)
    return {**result, "status": "labelled", "calls": len(rows)}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def quantiles(values: Iterable[float], qs: Sequence[int] = QUANTILES) -> dict[str, float] | None:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if not arr.size:
        return None
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs} | {"mean": round(float(arr.mean()), 3)}


def _share(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None}


def _bin(d: float | None) -> str:
    if d is None:
        return "unknown"
    for lo, hi in DISTANCE_BINS:
        if lo < d <= hi or (lo == 0.0 and d <= hi):
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}"
    return "unknown"


def pixel_stats(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    us = [v["oracle_pixel"][0] for v in views]
    vs = [v["oracle_pixel"][1] for v in views]
    euclid = [v["ahead_euclid"] for v in views if v["ahead_euclid"] is not None]
    return {
        "n": len(views),
        "ahead_arc_m": quantiles(v["ahead_arc"] for v in views if v["ahead_arc"] is not None),
        "ahead_euclid_m": quantiles(euclid),
        "ahead_euclid_gt_3m": _share(sum(1 for e in euclid if e > 3.0), len(euclid)),
        "u": quantiles(us),
        "v": quantiles(vs),
        "u_outside_100_540": _share(sum(1 for u in us if u < 100 or u > 540), len(us)),
        "v_lt_177": _share(sum(1 for v in vs if v < 177), len(vs)),
    }


def _arrows(views: Sequence[Mapping[str, Any]], key: str = "target1") -> dict[str, Any]:
    counts = collections.Counter(min(len([c for c in (v[key] or "") if c in "←→"]), 4) for v in views)
    n = sum(counts.values())
    return {"n": n, **{str(k): round(counts[k] / n, 4) if n else None for k in (1, 2, 3, 4)}}


def route_label_stats(
    episodes: Iterable[tuple[Sequence[Mapping[str, Any]], bool | None]], *, min_distance_m: float | None = None
) -> dict[str, Any]:
    """Statistics over episodes' label rows (line B or round 1), optionally only ``d > min_distance_m``.

    The previous executed turn of a call is the model's answer at the call before it in the
    episode (whether or not that call is in the distance filter).
    """
    views: list[dict[str, Any]] = []
    per_episode_corrections: list[int] = []
    outcome_calls: collections.Counter = collections.Counter()
    n_files = 0
    for rows, success in episodes:
        n_files += 1
        previous_model, corrections = None, 0
        for row in sorted(rows, key=lambda r: int(r["call_index"])):
            view = RL.summary_view(row)
            view["prev_direction"] = (
                previous_model["model_direction"] if previous_model and previous_model["model_kind"] == "turn" else 0
            )
            previous_model = view
            d = view["d"]
            if min_distance_m is not None and (d is None or d <= min_distance_m):
                continue
            view["success"] = success
            views.append(view)
            corrections += view["kind"] == "correction"
            outcome_label = "success" if success else ("failed" if success is False else "unknown")
            outcome_calls[f"{outcome_label}_calls"] += 1
            outcome_calls[f"{outcome_label}_{view['kind']}"] += 1
        per_episode_corrections.append(corrections)

    labelled = [v for v in views if not v["skip"]]
    corrections = [v for v in views if v["kind"] == "correction"]
    anchors = [v for v in views if v["kind"] == "anchor"]
    judged = [v for v in labelled if v["agree"] is not None]

    def target_kind(v: Mapping[str, Any]) -> str:
        if v["target1"] is None:
            return "none"
        if v["target_is_stop"]:
            return "stop"
        if v["target_is_pixel"]:
            return "pixel"
        return "turn" if v["target_direction"] != 0 else "other"

    by_distance: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in views:
        c = by_distance[_bin(v["d"])]
        c["calls"] += 1
        c[v["kind"]] += 1
        if v["kind"] == "correction":
            c[f"correction_{target_kind(v)}"] += 1

    agree_by_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in judged:
        agree_by_model[v["model_kind"]][str(v["agree"]).lower()] += 1

    def flips(selected: Sequence[Mapping[str, Any]], direction_key: str) -> dict[str, Any]:
        pairs = [v for v in selected if v["prev_direction"] != 0 and v[direction_key] != 0]
        return _share(sum(1 for v in pairs if v[direction_key] != v["prev_direction"]), len(pairs))

    oracle_pixels = [v for v in labelled if v["oracle_kind"] == "pixel" and v["oracle_pixel"] is not None]
    corrected_pixels = [v for v in corrections if v["target_is_pixel"] and v["oracle_pixel"] is not None]
    oracle_turns = [v for v in labelled if v["oracle_kind"] == "turn"]
    model_turns = [v for v in views if v["model_kind"] == "turn"]
    turn_targets = [v for v in views if v["kind"] in ("correction", "anchor") and target_kind(v) == "turn"]
    corrected_turns = [v for v in corrections if target_kind(v) == "turn"]
    model_progress = [v["model_progress_ratio"] for v in labelled if v["model_progress_ratio"] is not None]
    hysteresis = [v for v in oracle_turns if v["oracle_rule"] == "hysteresis"]
    n_calls = len(views)
    return {
        "episodes": n_files,
        "calls": n_calls,
        "labelled_calls": len(labelled),
        "by_distance": {k: dict(v) for k, v in sorted(by_distance.items())},
        "oracle_kind": dict(collections.Counter(v["oracle_kind"] for v in labelled)),
        "oracle_rule": dict(collections.Counter(v["oracle_rule"] for v in oracle_turns)),
        "agreement": {
            **_share(sum(1 for v in judged if v["agree"]), len(judged)),
            "by_model_kind": {k: dict(c) for k, c in sorted(agree_by_model.items())},
            "unjudged_labelled": len(labelled) - len(judged),
        },
        "corrections": {
            **_share(len(corrections), n_calls),
            "share_of_labelled": round(len(corrections) / len(labelled), 4) if labelled else None,
            "by_target": dict(collections.Counter(target_kind(v) for v in corrections)),
            "by_reason": dict(collections.Counter(v["reason"] for v in corrections).most_common()),
            "by_model_kind": dict(collections.Counter(v["model_kind"] for v in corrections)),
            "per_episode": quantiles(per_episode_corrections, (25, 50, 75, 90, 95)),
            "episodes_with_any": _share(sum(1 for c in per_episode_corrections if c), len(per_episode_corrections)),
        },
        "anchors": {**_share(len(anchors), n_calls), "by_target": dict(collections.Counter(target_kind(v) for v in anchors))},
        "by_outcome": dict(outcome_calls),
        "pixel_labels": {"oracle": pixel_stats(oracle_pixels), "corrections": pixel_stats(corrected_pixels)},
        "turn_labels": {
            "oracle_arrows": _arrows(oracle_turns, "oracle_turn1"),
            "correction_arrows": _arrows(corrected_turns),
            "flip_vs_previous_executed_turn": {
                # round 1's 34%: oracle turn labels after a call that executed a turn
                "oracle_turn_labels": flips(oracle_turns, "oracle_direction"),
                "training_turn_targets": flips(turn_targets, "target_direction"),
                "corrected_turn_targets": flips(corrected_turns, "target_direction"),
                # the base model's own 0.81%
                "model_turns": flips(model_turns, "model_direction"),
            },
            "hysteresis_with_route_within_40deg": _share(
                sum(1 for v in hysteresis if v["oracle_dyaw"] is not None and abs(v["oracle_dyaw"]) < 40.0),
                len(hysteresis),
            ),
        },
        "model_pixel_progress": {
            "n": len(model_progress),
            "good_ge_0.5": _share(sum(1 for r in model_progress if r >= 0.5), len(model_progress)),
            "bad_le_-0.3": _share(sum(1 for r in model_progress if r <= -0.3), len(model_progress)),
            "ratio": quantiles(model_progress),
        },
        "oracle_pixel_progress": quantiles(v["oracle_progress_ratio"] for v in oracle_pixels),
    }


def stats_for(
    label_paths: Sequence[Path], outcomes: Mapping[str, bool | None], *, min_distance_m: float | None = None
) -> dict[str, Any]:
    episodes = ((REC.read_labels(path), outcomes.get(str(path))) for path in label_paths)
    return route_label_stats(episodes, min_distance_m=min_distance_m)


# --------------------------------------------------------------------------- #
# InternData-N1 reference
# --------------------------------------------------------------------------- #
def _interndata_scene(scene_dir: str) -> dict[str, list]:
    import pyarrow.parquet as pq

    scene_dir = Path(scene_dir)
    out: dict[str, list] = collections.defaultdict(list)
    with open(scene_dir / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    for ep in episodes:
        index = int(ep["episode_index"])
        table = pq.read_table(
            scene_dir / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet",
            columns=["action", "pose.125cm_0deg", "goal.125cm_30deg", "relative_goal_frame_id.125cm_30deg"],
        ).to_pydict()
        actions = [int(a) for a in table["action"][1:]] + [0]
        length = len(actions)
        if length < 4:
            continue
        loc = np.asarray([np.asarray(m, dtype=np.float64) for m in table["pose.125cm_0deg"]])[:, :3, 3]
        goals = table["goal.125cm_30deg"]
        rel = [int(r) for r in table["relative_goal_frame_id.125cm_30deg"]]
        previous_turn = 0
        for n in range(length // 4 + 1):
            t = n * 4
            if t in (length, length - 1):
                continue
            goal = [int(g) for g in goals[t]]
            if goal[0] == -1:
                if actions[t] == 1:
                    previous_turn = 0
                    continue
                turns = []
                for i in range(t, min(length, t + 4)):
                    if actions[i] == 1:
                        break
                    turns.append(actions[i])
                direction = int(np.sign(turns.count(2) - turns.count(3)))
                out["arrows"].append(len(turns))
                if previous_turn and direction:
                    out["flip"].append(int(direction != previous_turn))
                previous_turn = direction
                continue
            previous_turn = 0
            if rel[t] < 3:
                continue
            tg = min(t + rel[t], length - 1)
            out["u"].append(goal[0])
            out["v"].append(goal[1])
            out["ahead_euclid"].append(float(np.linalg.norm(loc[tg, :2] - loc[t, :2])))
            out["ahead_fwd"].append(0.25 * sum(1 for a in actions[t:tg] if a == 1))
    return dict(out)


def interndata_reference(root: str | os.PathLike, workers: int = 8) -> dict[str, Any]:
    """InternData-N1's own pixel-goal and turn distributions (``interndata_replay`` sampling)."""
    scenes = sorted(str(p) for p in Path(root).iterdir() if (p / "meta" / "episodes.jsonl").is_file())
    merged: dict[str, list] = collections.defaultdict(list)
    with mp.get_context("fork").Pool(max(1, min(workers, len(scenes)))) as pool:
        for part in pool.imap_unordered(_interndata_scene, scenes):
            for key, values in part.items():
                merged[key].extend(values)
    us, vs, arrows = merged["u"], merged["v"], collections.Counter(merged["arrows"])
    n_turns = sum(arrows.values())
    return {
        "root": str(root),
        "scenes": len(scenes),
        "pixel": {
            "n": len(us),
            "ahead_euclid_m": quantiles(merged["ahead_euclid"]),
            "ahead_fwd_steps_m": quantiles(merged["ahead_fwd"]),
            "ahead_euclid_gt_3m": _share(sum(1 for e in merged["ahead_euclid"] if e > 3.0), len(us)),
            "u": quantiles(us),
            "v": quantiles(vs),
            "u_outside_100_540": _share(sum(1 for u in us if u < 100 or u > 540), len(us)),
            "v_lt_177": _share(sum(1 for v in vs if v < 177), len(vs)),
            "u_min_max": [min(us), max(us)] if us else None,
            "v_min_max": [min(vs), max(vs)] if vs else None,
        },
        "turn": {
            "n": n_turns,
            "arrows": {str(k): round(arrows[k] / n_turns, 4) if n_turns else None for k in (1, 2, 3, 4)},
            "flip_vs_previous_turn_sample": _share(sum(merged["flip"]), len(merged["flip"])),
        },
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _worker_init(state: dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER.update(state)
    _WORKER["navmeshes"] = {}


def _geometry_for_scene(scene_id: str) -> RouteNavmeshGeometry:
    cache = _WORKER["navmeshes"]
    if scene_id not in cache:
        cache[scene_id] = RouteNavmeshGeometry(Path(_WORKER["scenes_dir"]) / scene_id / f"{scene_id}.navmesh")
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
        )
    except Exception as err:  # noqa: BLE001 -- one bad record must not stop the run
        return {"record": record_path, "status": "failed", "error": f"{type(err).__name__}: {err}",
                "traceback": traceback.format_exc(limit=4)}


CONFIG_FLAGS = {f.name: f for f in dataclasses.fields(RL.RouteLabelConfig)}


def build_config(args: argparse.Namespace) -> Any:
    return RL.RouteLabelConfig(**{name: getattr(args, name) for name in CONFIG_FLAGS})


def select_records(records: Sequence[str], *, sample: int, seed: int, limit: int) -> list[str]:
    """A seeded subset of ``sample`` records (kept in path order), then the first ``limit``."""
    records = list(records)
    if sample and sample < len(records):
        pick = np.random.default_rng(seed).choice(len(records), size=sample, replace=False)
        records = [records[i] for i in sorted(int(i) for i in pick)]
    return records[:limit] if limit > 0 else records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--scenes_dir", default="/dataset/mp3d", help="holds <scene>/<scene>.navmesh")
    parser.add_argument("--episodes", nargs="*", default=[], help="R2R-CE-format split json.gz (authoritative path/goal)")
    parser.add_argument("--workers", type=int, default=8, help="CPU processes (EXP-25 on the 4090: at most 12)")
    parser.add_argument("--force", action="store_true", help="relabel even when current labels exist")
    parser.add_argument("--sample", type=int, default=0, help="label a seeded subset of this many records (0 = all)")
    parser.add_argument("--sample_seed", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0, help="then at most this many records (0 = all)")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--compare_label_root", default=None,
                        help="another labelling of the same records (e.g. round 1's) for side-by-side statistics")
    parser.add_argument("--interndata_root", default=None, help="InternData-N1 r2r scene dirs: add its distributions")
    defaults = RL.RouteLabelConfig()
    for name, f in CONFIG_FLAGS.items():
        value = getattr(defaults, name)
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    records = select_records(
        [str(p) for p in REC.iter_record_paths(args.record_root)], sample=args.sample, seed=args.sample_seed, limit=args.limit
    )
    dataset_index = base.load_episode_index(args.episodes) if args.episodes else None
    state = {"label_root": args.label_root, "scenes_dir": args.scenes_dir, "dataset_index": dataset_index,
             "config": config, "force": bool(args.force)}
    started = time.time()
    results: list[dict[str, Any]] = []
    if args.workers <= 1 or len(records) <= 1:
        _worker_init(state)
        results = [_worker_label(r) for r in records]
    else:
        with mp.get_context("fork").Pool(args.workers, initializer=_worker_init, initargs=(state,)) as pool:
            for result in pool.imap_unordered(_worker_label, records, chunksize=4):
                results.append(result)
                if len(results) % 200 == 0:
                    print(f"[label-route] {len(results)}/{len(records)} records", flush=True)
    status = collections.Counter(r["status"] for r in results)
    failures = [r for r in results if r["status"] == "failed"]
    for failure in failures[:10]:
        print(f"[label-route] FAILED {failure['record']}: {failure['error']}", file=sys.stderr, flush=True)
    ok = sorted((r for r in results if r["status"] != "failed"), key=lambda r: r["labels"])
    label_paths = [Path(r["labels"]) for r in ok]
    outcomes = {r["labels"]: r["success"] for r in ok}
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "label_schema": REC.LABEL_SCHEMA,
        "record_root": str(args.record_root),
        "label_root": str(args.label_root),
        "episodes": list(args.episodes),
        "sample": {"n": args.sample, "seed": args.sample_seed, "limit": args.limit},
        "config": {**dataclasses.asdict(config), "digest": config.digest(), "rules_version": RL.ROUTE_RULES_VERSION},
        "num_records": len(records),
        "status": dict(status),
        "failures": [{k: f[k] for k in ("record", "error")} for f in failures],
        "label_s": round(time.time() - started, 1),
        "stats": stats_for(label_paths, outcomes),
    }
    if config.line == RL.TERMINAL_LABEL_LINE:  # the part a line-B labelling of the same records must equal
        summary["stats_beyond_line_a_radius"] = stats_for(label_paths, outcomes, min_distance_m=config.label_min_distance_m)
    if args.compare_label_root:
        compare_paths, compare_outcomes, missing = [], {}, 0
        for r in ok:
            other = REC.label_path_for(r["record"], args.compare_label_root)
            if other.is_file():
                compare_paths.append(other)
                compare_outcomes[str(other)] = r["success"]
            else:
                missing += 1
        summary["compare"] = {
            "label_root": str(args.compare_label_root),
            "missing": missing,
            "all": stats_for(compare_paths, compare_outcomes),
            "beyond_line_a_radius": stats_for(compare_paths, compare_outcomes, min_distance_m=config.label_min_distance_m),
        }
    if args.interndata_root:
        summary["interndata_reference"] = interndata_reference(args.interndata_root, workers=max(1, args.workers))
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary_path = Path(args.summary) if args.summary else Path(args.label_root) / "label_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = summary["stats"]
    print(json.dumps({"num_records": len(records), "status": dict(status), "calls": stats["calls"],
                      "corrections": stats["corrections"]["k"], "agreement": stats["agreement"]["rate"]}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
