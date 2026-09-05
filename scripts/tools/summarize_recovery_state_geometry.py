#!/usr/bin/env python3
"""EXP-12 D1 + D3a: where the oracle actually goes in DAgger recovery states.

D1 asks whether a recovery decision layer has a target at all: in the states the
DAgger collector labelled ``dagger_hard``, does the oracle require leaving the
front view, and does the native System2 proposal point somewhere else?  D3a asks
how often such states occur naturally and whether episodes containing them fail
more often.  Both read the same episode tars, so they share one pass.

Geometry notes that are easy to get wrong:

* ``native_future_poses`` and ``arrays/oracle_future_poses.npy`` are **agent**
  poses (floor height); the production future labels live in the **camera**
  frame (see ``relative_future_centers_from_world``, which takes camera c2w on
  both sides).  Every point is therefore lifted by this sample's own camera
  height offset before projection, or almost everything would fall below the
  image and be scored "invisible in all four views".
* The native rollout is four primitive actions (about one metre); the oracle
  rollout is about 3.25 m.  The pre-registered angle compares the two
  **endpoints**; a distance-matched variant is reported alongside it as a
  descriptive number only, and is not part of any criterion.
* A turn-only native rollout has no displacement direction.  Below
  ``--turn-only-travel-m`` its final heading is used instead, mirroring how the
  collector computed ``native_oracle_heading_disagreement_deg``.

Output is JSON; the ledger cites the path.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.future_trajectory_heatmap import _project_unique_view  # noqa: E402

VIEWS = ("front", "right", "back", "left")
REVISIT_TAGS = ("loop", "avoidable_revisit", "oscillation")
IMAGE_SIZE = (384, 384)


def build_intrinsics(image_size: tuple[int, int]) -> np.ndarray:
    """HFOV 90 degrees, matching src/data/heatmap_geometry.py."""
    width, height = image_size
    focal = width / (2.0 * math.tan(math.radians(90.0) / 2.0))
    return np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )


def to_camera_frame(
    agent_positions: np.ndarray,
    current_camera_c2w: np.ndarray,
    camera_height_offset: float,
) -> np.ndarray:
    """Lift agent-height points to camera height, then express them in the current camera."""
    lifted = np.asarray(agent_positions, dtype=np.float32).copy()
    lifted[:, 1] += float(camera_height_offset)
    homogeneous = np.concatenate(
        (lifted, np.ones((len(lifted), 1), dtype=np.float32)), axis=1
    )
    return (np.linalg.inv(np.asarray(current_camera_c2w, dtype=np.float32)) @ homogeneous.T).T[:, :3]


def first_point_beyond(points_cam: np.ndarray, radius_m: float) -> np.ndarray | None:
    """First point whose horizontal distance from the origin exceeds radius_m."""
    if len(points_cam) == 0:
        return None
    horizontal = np.linalg.norm(points_cam[:, [0, 2]], axis=1)
    beyond = np.nonzero(horizontal > float(radius_m))[0]
    if len(beyond) == 0:
        return None
    return points_cam[int(beyond[0])]


def direction_deg(point_cam: np.ndarray) -> float:
    """Bearing in the camera xz-plane; 0 is straight ahead, positive to the right."""
    x, _, z = (float(v) for v in point_cam)
    return math.degrees(math.atan2(x, -z))


def angle_between_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def heading_direction_deg(relative_rotation: np.ndarray) -> float:
    """Bearing of a pose's own forward axis (-z) expressed in the current camera."""
    forward = -np.asarray(relative_rotation, dtype=np.float32)[:3, 2]
    return math.degrees(math.atan2(float(forward[0]), -float(forward[2])))


def view_of(point_cam: np.ndarray, intrinsics: np.ndarray) -> int | None:
    projection = _project_unique_view(
        np.asarray(point_cam, dtype=np.float32), intrinsics, IMAGE_SIZE
    )
    return None if projection is None else int(projection[0])


def analyse_episode(tar_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    intrinsics = build_intrinsics(IMAGE_SIZE)
    samples_out: list[dict[str, Any]] = []
    with tarfile.open(tar_path) as archive:
        episode = json.loads(archive.extractfile("episode.json").read().decode("utf-8"))
        rows = [
            json.loads(line)
            for line in archive.extractfile("samples.jsonl").read().decode("utf-8").splitlines()
            if line
        ]
        frames = [
            json.loads(line)
            for line in archive.extractfile("frames.jsonl").read().decode("utf-8").splitlines()
            if line
        ]
        oracle_poses = np.load(
            io.BytesIO(archive.extractfile("arrays/oracle_future_poses.npy").read())
        )

    goal = np.asarray(episode["goal_position"], dtype=np.float64)
    last_pose = np.asarray(frames[-1]["pose"], dtype=np.float64)
    final_distance_m = float(
        np.linalg.norm(last_pose[[0, 2], 3] - goal[[0, 2]])
    )

    for row in rows:
        current_camera = np.asarray(row["current_camera_pose"], dtype=np.float32)
        current_agent = np.asarray(row["current_agent_pose"], dtype=np.float32)
        height_offset = float(current_camera[1, 3] - current_agent[1, 3])

        oracle_slice = oracle_poses[int(row["future_pose_start"]) : int(row["future_pose_end"])]
        native_slice = np.asarray(row["native_future_poses"], dtype=np.float32)
        if len(oracle_slice) < 2 or len(native_slice) < 2:
            continue

        oracle_cam = to_camera_frame(oracle_slice[:, :3, 3], current_camera, height_offset)
        native_cam = to_camera_frame(native_slice[:, :3, 3], current_camera, height_offset)

        oracle_first = first_point_beyond(oracle_cam, args.first_point_radius_m)
        oracle_view = None if oracle_first is None else view_of(oracle_first, intrinsics)

        oracle_endpoint_dir = direction_deg(oracle_cam[-1])
        native_travel_m = float(
            np.linalg.norm(np.diff(native_cam[:, [0, 2]], axis=0), axis=1).sum()
        )
        if native_travel_m < args.turn_only_travel_m:
            relative_rotation = (
                np.linalg.inv(current_camera[:3, :3]) @ native_slice[-1][:3, :3]
            )
            native_endpoint_dir = heading_direction_deg(relative_rotation)
            native_kind = "turn_only"
        else:
            native_endpoint_dir = direction_deg(native_cam[-1])
            native_kind = "translating"

        endpoint_angle = angle_between_deg(native_endpoint_dir, oracle_endpoint_dir)

        native_first = first_point_beyond(native_cam, args.first_point_radius_m)
        matched_angle = (
            None
            if native_first is None or oracle_first is None
            else angle_between_deg(direction_deg(native_first), direction_deg(oracle_first))
        )

        native_view = None if native_first is None else view_of(native_first, intrinsics)

        tags = list(row.get("failure_tags") or [])
        signals = row.get("candidate_signals") or {}
        samples_out.append(
            {
                "sample_key": row.get("key"),
                "native_view": native_view,
                "source_type": row.get("source_type"),
                "tags": tags,
                "oracle_view": oracle_view,
                "oracle_has_first_point": oracle_first is not None,
                "endpoint_angle_deg": endpoint_angle,
                "matched_angle_deg": matched_angle,
                "native_kind": native_kind,
                "native_travel_m": native_travel_m,
                "revisit_state": any(tag in REVISIT_TAGS for tag in tags)
                or bool(signals.get("loop_detected"))
                or bool(signals.get("oscillation_detected")),
            }
        )

    return {
        "episode_key": episode["episode_key"],
        "scene_id": episode["scene_id"],
        "final_distance_m": final_distance_m,
        "samples": samples_out,
    }


def _worker(payload: tuple[str, argparse.Namespace]) -> dict[str, Any] | None:
    path, args = payload
    try:
        return analyse_episode(Path(path), args)
    except Exception as exc:  # noqa: BLE001 - one bad tar must not kill the sweep
        return {"error": f"{path}: {type(exc).__name__}: {exc}"}


def summarise_bucket(samples: list[dict[str, Any]], angle_threshold: float) -> dict[str, Any]:
    total = len(samples)
    if total == 0:
        return {"states": 0}
    view_counts = Counter(
        VIEWS[s["oracle_view"]] if s["oracle_view"] is not None else "invisible"
        for s in samples
    )
    outside_front = sum(1 for s in samples if s["oracle_view"] != 0)
    endpoint_angles = np.asarray([s["endpoint_angle_deg"] for s in samples], dtype=np.float64)
    matched = np.asarray(
        [s["matched_angle_deg"] for s in samples if s["matched_angle_deg"] is not None],
        dtype=np.float64,
    )
    return {
        "states": total,
        "oracle_view_counts": dict(view_counts),
        "oracle_view_fractions": {k: v / total for k, v in view_counts.items()},
        "oracle_outside_front_frac": outside_front / total,
        "endpoint_angle_median_deg": float(np.median(endpoint_angles)),
        "endpoint_angle_mean_deg": float(endpoint_angles.mean()),
        f"endpoint_angle_gt_{angle_threshold:g}_frac": float(
            (endpoint_angles > angle_threshold).mean()
        ),
        "matched_angle_states": int(len(matched)),
        "matched_angle_median_deg": float(np.median(matched)) if len(matched) else None,
        f"matched_angle_gt_{angle_threshold:g}_frac": (
            float((matched > angle_threshold).mean()) if len(matched) else None
        ),
        "turn_only_frac": sum(1 for s in samples if s["native_kind"] == "turn_only") / total,
        "tag_counts": dict(Counter(tag for s in samples for tag in s["tags"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--per-state-jsonl",
        type=Path,
        default=None,
        help="also emit one row per state so D2 can reuse this exact oracle-view definition",
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means every episode")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--first-point-radius-m", type=float, default=0.5)
    parser.add_argument("--turn-only-travel-m", type=float, default=0.05)
    parser.add_argument("--angle-threshold-deg", type=float, default=45.0)
    parser.add_argument("--failure-distance-m", type=float, default=3.0)
    args = parser.parse_args()

    tars = sorted(args.collection_root.glob("shard_*/episodes/*/episode.tar"))
    if not tars:
        raise SystemExit(f"no episode tars under {args.collection_root}")
    if args.max_episodes > 0:
        tars = tars[: args.max_episodes]
    print(f"episodes: {len(tars)}", flush=True)

    payloads = [(str(path), args) for path in tars]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with mp.Pool(processes=max(1, args.workers)) as pool:
        for index, result in enumerate(pool.imap_unordered(_worker, payloads, chunksize=8), 1):
            if result is None:
                continue
            if "error" in result:
                errors.append(result["error"])
            else:
                results.append(result)
            if index % 1000 == 0:
                print(f"  {index}/{len(tars)}", flush=True)

    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for episode in results:
        for sample in episode["samples"]:
            by_bucket.setdefault(str(sample["source_type"]), []).append(sample)

    all_states = [s for samples in by_bucket.values() for s in samples]
    revisit_states = sum(1 for s in all_states if s["revisit_state"])

    with_revisit = [e for e in results if any(s["revisit_state"] for s in e["samples"])]
    without_revisit = [e for e in results if not any(s["revisit_state"] for s in e["samples"])]

    def failure_rate(episodes: list[dict[str, Any]]) -> float | None:
        if not episodes:
            return None
        failed = sum(
            1 for e in episodes if e["final_distance_m"] > args.failure_distance_m
        )
        return failed / len(episodes)

    rate_with = failure_rate(with_revisit)
    rate_without = failure_rate(without_revisit)

    report = {
        "schema": "heatmapvln-exp12-recovery-geometry-v1",
        "inputs": {
            "collection_root": str(args.collection_root),
            "episodes_analysed": len(results),
            "episodes_failed_to_read": len(errors),
            "first_point_radius_m": args.first_point_radius_m,
            "turn_only_travel_m": args.turn_only_travel_m,
            "angle_threshold_deg": args.angle_threshold_deg,
            "failure_distance_m": args.failure_distance_m,
            "image_size": list(IMAGE_SIZE),
            "hfov_deg": 90.0,
            "view_order": list(VIEWS),
        },
        "d1_by_bucket": {
            bucket: summarise_bucket(samples, args.angle_threshold_deg)
            for bucket, samples in sorted(by_bucket.items())
        },
        "d3a": {
            "total_states": len(all_states),
            "revisit_states": revisit_states,
            "revisit_state_frac": revisit_states / len(all_states) if all_states else None,
            "episodes_with_revisit_state": len(with_revisit),
            "episodes_without_revisit_state": len(without_revisit),
            "failure_rate_with_revisit": rate_with,
            "failure_rate_without_revisit": rate_without,
            "episode_gap_pt": (
                (rate_with - rate_without) * 100.0
                if rate_with is not None and rate_without is not None
                else None
            ),
        },
        "errors": errors[:20],
    }

    if args.per_state_jsonl is not None:
        args.per_state_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.per_state_jsonl.open("w", encoding="utf-8") as handle:
            for episode in results:
                for sample in episode["samples"]:
                    handle.write(
                        json.dumps(
                            {
                                "episode_key": episode["episode_key"],
                                "scene_id": episode["scene_id"],
                                **sample,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        print(f"wrote {args.per_state_jsonl}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["d1_by_bucket"], ensure_ascii=False, indent=2))
    print(json.dumps(report["d3a"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
