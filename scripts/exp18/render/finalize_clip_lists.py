"""Apply the pre-registered keep rule to rendered EXP-18 tier C/D/E clips.

Maps every rendered clip to (scene, episode_id) through its meta.json, then
walks the selection manifest (select_episodes.py) in sha1 order per scene:

  C  keep the first 25 per scene with num_frames >= 20
  D  keep the first 4 per scene with num_frames >= 20
  E  keep every designed route with num_frames >= 20

Clips with < 20 frames are MOVED out of the tier data root into the sibling
``excluded_short/<scene>/<clip>`` (the AMB3R cache builder scans the whole data
root and aborts the plan on a short clip).  ``>= 20``-frame clips beyond the
quota ("surplus") stay in place unless ``--move-surplus``; a second clip of
the same episode (never produced by the collector's resume logic) goes to
``excluded_duplicate/``.

Every clip is also validated (meta.json, chunk keys identical to
r2r_panoramic_data_v2, frame ids 0..T-1 across chunks, trajectory length) and
its route following is measured from trajectory_3d.npy (end-to-start,
end-to-final-reference, in-order reference-point passage).  A clip follows
its route when every reference point is passed in order within
``--route-tolerance`` and it ends within that distance of the last one.

The run FAILS (exit 2: no clip list, no marker, only quarantine moves) when
  * a clip is invalid (e.g. a worker killed while chunk writes were pending),
    unreadable (orphan) or not in the selection (unexpected);
  * a selected episode has no clip while its scene's quota walk is still open
    (C/D: at a rank before the quota filled; E: any).  The collector only
    skips an episode itself for < 10 frames, which no selected path allows,
    so a missing episode is a render failure (reset/step exception, timeout);
    letting the next rank fill in would silently break the keep rule;
  * a clip the walk would keep did not follow its route (collector hit
    --max-steps, or its follower gave up on an unreachable point).
Remedy: ``--quarantine-invalid`` (moves invalid/orphan clips to
``<raw>/_incomplete/``), re-run run_render.sh (it resumes: renders only
episodes without a clip in the data root), finalize again.  Short clips are
only moved on success, so such a resume never re-renders them.  Only if an
episode cannot be rendered, override with ``--allow-missing REASON`` (the next
rank fills in) or ``--allow-off-route REASON`` (the clip is not kept, the next
rank fills in); the reason goes into the list header, report and marker.

Writes the clip list (common.write_clip_list; header documents every drop),
a JSON report (also on failure), and ``<raw>/.exp18_finalized.json``
(run_render.sh refuses to render into a finalized tier).  Standard library +
numpy, Python 3.8 compatible (runs under envs/vlnce or envs/qwen25)::

    cd <repo> && <python> -m scripts.exp18.render.finalize_clip_lists --tier C
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.exp18 import common  # noqa: E402
from scripts.exp18.render import layout  # noqa: E402

SCHEMA = "exp18-render-finalize-v2"  # v2: missing / off-route episodes block
DIRECTIONS = ("front", "right", "back", "left", "front_down")
EXPECTED_CHUNK_KEYS = frozenset(
    ["frame_ids", "depth_front", "depth_front_down"]
    + [f"rgb_{d}" for d in DIRECTIONS]
    + [f"pose_{d}" for d in DIRECTIONS]
)
KEEP_RULES = {
    "C": "per scene, episodes in sha1('<scene>:<episode_id>') order; keep the first 25 with num_frames >= 20",
    "D": "per scene, episodes in sha1('<scene>:<episode_id>') order; keep the first 4 with num_frames >= 20",
    "E": "keep every designed route with num_frames >= 20",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _route_stats(traj: np.ndarray, ref: np.ndarray, tolerance: float) -> dict:
    """In-order match of reference points against the trajectory.

    Each point is matched to its first passage: the first run of frames within
    ``tolerance`` (the miss is that run's closest approach).  The next point is
    searched from the start of that run, so the order is enforced (an
    out-and-back must really come back) while closely spaced points at a
    turnaround can share one run.
    """
    t0, reached, worst = 0, 0, 0.0
    for point in ref:
        dist = np.linalg.norm(traj[t0:] - point, axis=1)
        hits = np.nonzero(dist <= tolerance)[0]
        if hits.size:
            end = int(hits[0])
            while end + 1 < dist.size and dist[end + 1] <= tolerance:
                end += 1
            miss = float(dist[hits[0]:end + 1].min())
            t0 += int(hits[0])
            reached += 1
        else:
            miss = float(dist.min())
        worst = max(worst, miss)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    end_to_final = float(np.linalg.norm(traj[-1] - ref[-1]))
    return {
        "route_ok": bool(reached == len(ref) and end_to_final <= tolerance),
        "start_xyz": [round(float(v), 4) for v in traj[0]],
        "end_xyz": [round(float(v), 4) for v in traj[-1]],
        "path_length_m": round(float(steps.sum()), 3),
        "end_to_start_m": round(float(np.linalg.norm(traj[-1] - traj[0])), 3),
        "end_to_final_ref_m": round(end_to_final, 3),
        "ref_points": int(len(ref)),
        "ref_reached_in_order": reached,
        "ref_max_miss_m": round(worst, 3),
    }


def inspect_clip(clip_dir: Path, tolerance: float) -> dict:
    row = {"valid": False, "reason": None}
    meta_path = clip_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        row["reason"] = f"meta.json unreadable: {exc.__class__.__name__}"
        return row
    row.update(scene_id=str(meta.get("scene_id")), episode_id=str(meta.get("episode_id")),
               trajectory_id=str(meta.get("trajectory_id")), num_frames=int(meta.get("num_frames", -1)))
    chunks = sorted((clip_dir / "chunks").glob("chunk_*.npz"))
    if not chunks:
        row["reason"] = "no chunks"
        return row
    frame_ids = []
    try:
        for chunk in chunks:
            with np.load(str(chunk)) as npz:
                keys = frozenset(npz.files)
                if keys != EXPECTED_CHUNK_KEYS:
                    row["reason"] = f"{chunk.name} keys differ from v2: {sorted(keys ^ EXPECTED_CHUNK_KEYS)}"
                    return row
                frame_ids.append(np.asarray(npz["frame_ids"]))
        traj = np.load(str(clip_dir / "trajectory_3d.npy")).astype(np.float64)
    except Exception as exc:  # truncated npz / npy from an interrupted write
        row["reason"] = f"unreadable array: {exc.__class__.__name__}: {exc}"
        return row
    ids = np.concatenate(frame_ids)
    if not np.array_equal(ids, np.arange(row["num_frames"])):
        row["reason"] = f"frame_ids are not 0..{row['num_frames'] - 1} (got {ids.size} ids)"
        return row
    if traj.shape != (row["num_frames"], 3):
        row["reason"] = f"trajectory_3d shape {traj.shape} != ({row['num_frames']}, 3)"
        return row
    ref = np.asarray(meta.get("reference_path") or [], dtype=np.float64).reshape(-1, 3)
    if ref.size == 0:
        row["reason"] = "meta.json has no reference_path"
        return row
    row.update(_route_stats(traj, ref, tolerance))
    row["valid"] = True
    return row


def _scan(root: Path, location: str) -> list:
    rows = []
    if not root.is_dir():
        return rows
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in scene_dir.glob("clip_*") if p.is_dir()):
            rows.append({"scene_dir": scene_dir.name, "clip": clip_dir.name,
                         "path": clip_dir, "location": location})
    return rows


def _move(path: Path, target_root: Path) -> Path:
    target = target_root / path.parent.name / path.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def _scene_glb(tier: str, scene: str, entry: dict) -> str:
    if tier == "D":
        return entry["glb"]
    return str(common.MP3D_SCENES / scene / f"{scene}.glb")


def _describe(row: dict) -> str:
    text = f"{row.get('clip_key', '-')} ep={row['episode_id']} rank={row['rank']} frames={row.get('num_frames')}"
    if row["status"] == "missing" and not row["quota_open"]:
        text += " (beyond quota, harmless)"
    if row["status"] == "off_route":
        text += (f" ref_in_order={row['ref_reached_in_order']}/{row['ref_points']}"
                 f" end_to_final_ref={row['end_to_final_ref_m']}m")
    return text + (f" {row['reason']}" if row.get("reason") else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default=os.environ.get("TIER"), choices=layout.RENDER_TIERS)
    parser.add_argument("--manifest", type=Path, default=layout.manifest_path())
    parser.add_argument("--data-root", type=Path, default=None, help="default: common.TIERS[tier]['data_root']")
    parser.add_argument("--clip-list", type=Path, default=None, help="default: common.clip_list_path(tier)")
    parser.add_argument("--report", type=Path, default=None, help="default: next to the clip list")
    parser.add_argument("--route-tolerance", type=float, default=0.5,
                        help="a reference point counts as passed within this distance (m)")
    parser.add_argument("--move-surplus", action="store_true",
                        help="also move >= 20-frame clips beyond the quota out of the data root")
    parser.add_argument("--quarantine-invalid", action="store_true",
                        help="move invalid clips to <raw>/_incomplete/<stamp>/ (re-render them afterwards)")
    parser.add_argument("--allow-missing", metavar="REASON", default=None,
                        help="finalize although selected episodes inside the quota walk have no clip "
                             "(the next rank fills in); REASON goes into header, report and marker")
    parser.add_argument("--allow-off-route", metavar="REASON", default=None,
                        help="finalize although clips the walk would keep did not follow their route "
                             "(they are not kept, the next rank fills in); REASON is recorded likewise")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; move and write nothing (exit code as a real run would give)")
    args = parser.parse_args()
    if not args.tier:
        parser.error("--tier (or env TIER) is required")
    overrides = OrderedDict()
    for flag in ("allow_missing", "allow_off_route"):
        value = getattr(args, flag)
        if value is not None:
            if not value.strip():
                parser.error(f"--{flag.replace('_', '-')} needs a non-empty REASON")
            overrides[flag] = value.strip()
    tier = args.tier
    data_root = args.data_root or layout.data_root(tier)
    raw = data_root.parent
    excluded_short = raw / "excluded_short"
    excluded_surplus = raw / "excluded_surplus"
    quarantine = raw / "_incomplete" / time.strftime("%Y%m%d_%H%M%S")
    marker = raw / ".exp18_finalized.json"
    clip_list = args.clip_list or common.clip_list_path(tier)
    report_path = args.report or clip_list.with_name(f"{tier}_finalize_report.json")

    manifest = json.loads(args.manifest.read_text())
    tier_manifest = manifest["tiers"][tier]
    quota = tier_manifest.get("keep_per_scene")
    min_frames = int(manifest["rules"]["min_frames"])

    # ---- index rendered clips by (scene_dir, episode_id) ----------------------
    selected = set()
    for scene, entry in tier_manifest["scenes"].items():
        for sel in entry["episodes"]:
            selected.add((entry.get("render_scene_dir", scene), sel["episode_id"]))
    rendered, unexpected, orphans = {}, [], []
    for clip in (_scan(data_root, "data_root") + _scan(excluded_short, "excluded_short")
                 + _scan(excluded_surplus, "excluded_surplus")):
        clip.update(inspect_clip(clip["path"], args.route_tolerance))
        key = (clip["scene_dir"], clip.get("episode_id"))
        if "episode_id" not in clip:  # no readable meta.json: cannot be mapped to an episode
            orphans.append(clip)
        elif key not in selected:
            unexpected.append(clip)
        else:
            rendered.setdefault(key, []).append(clip)

    # ---- apply the keep rule in manifest order ---------------------------------
    # Quarantine moves (invalid/orphan) happen even when the run fails, so a
    # resume re-renders those episodes; every other move only on success.
    rows, keys = [], []
    quarantine_moves = [(clip, quarantine) for clip in orphans] if args.quarantine_invalid else []
    final_moves = []
    scene_summary = OrderedDict()
    for scene, entry in tier_manifest["scenes"].items():
        scene_dir = entry.get("render_scene_dir", scene)
        kept = 0
        for sel in entry["episodes"]:
            quota_open = quota is None or kept < quota  # this rank could still be kept
            # the collector never renders an episode twice; if it happened, keep the first valid clip
            clips = sorted(rendered.get((scene_dir, sel["episode_id"]), []),
                           key=lambda c: (not c["valid"], c["clip"]))
            base = {"scene": scene, "scene_dir": scene_dir, "episode_id": sel["episode_id"],
                    "sha1": sel.get("sha1", sel.get("source_sha1")), "rank": sel.get("rank", sel.get("index")),
                    "scene_glb": _scene_glb(tier, scene, entry), "quota_open": quota_open}
            if "pattern" in sel:
                base["pattern"] = sel["pattern"]
            if not clips:
                rows.append(dict(base, status="missing"))
                continue
            primary, duplicates = clips[0], clips[1:]
            for dup in duplicates:
                rows.append(dict(base, clip_key=f"{scene_dir}/{dup['clip']}", status="duplicate",
                                 location=dup["location"], num_frames=dup.get("num_frames"),
                                 reason=dup["reason"]))
                if dup["location"] == "data_root":
                    final_moves.append((dup, raw / "excluded_duplicate"))
            row = dict(base, **{k: v for k, v in primary.items() if k not in ("path", "scene_dir", "clip")})
            row["clip_key"] = f"{scene_dir}/{primary['clip']}"
            if not primary["valid"]:
                row["status"] = "invalid"
                if args.quarantine_invalid:
                    quarantine_moves.append((primary, quarantine))
            elif primary["num_frames"] < min_frames:
                row["status"] = "short"
                if primary["location"] != "excluded_short":
                    final_moves.append((primary, excluded_short))
            elif not quota_open:
                row["status"] = "surplus"
                if args.move_surplus and primary["location"] == "data_root":
                    final_moves.append((primary, excluded_surplus))
            elif not primary["route_ok"]:
                row["status"] = "off_route"  # never kept; blocks unless --allow-off-route
            else:
                row["status"] = "kept"
                kept += 1
                keys.append(row["clip_key"])
                if primary["location"] != "data_root":
                    raise RuntimeError(f"{row['clip_key']} must be kept but sits in {primary['location']}")
            rows.append(row)
        scene_summary[scene] = {"selected": len(entry["episodes"]), "kept": kept, "quota": quota}

    counts = OrderedDict((status, sum(r["status"] == status for r in rows))
                         for status in ("kept", "short", "surplus", "off_route", "missing", "invalid", "duplicate"))
    counts["missing_quota_open"] = sum(r["status"] == "missing" and r["quota_open"] for r in rows)
    counts["selected"] = sum(len(e["episodes"]) for e in tier_manifest["scenes"].values())
    counts["unexpected"] = len(unexpected)
    counts["orphan"] = len(orphans)
    below = {s: v for s, v in scene_summary.items() if quota is not None and v["kept"] < quota}
    routes = OrderedDict()
    for pattern in sorted({r.get("pattern") for r in rows if r["status"] == "kept"}, key=str):
        chosen = [r for r in rows if r["status"] == "kept" and r.get("pattern") == pattern]
        routes[pattern or "all"] = {
            "n": len(chosen),
            "num_frames_median": float(np.median([r["num_frames"] for r in chosen])),
            "end_to_start_m_max": max(r["end_to_start_m"] for r in chosen),
            "end_to_final_ref_m_max": max(r["end_to_final_ref_m"] for r in chosen),
            "all_ref_points_reached": sum(r["ref_reached_in_order"] == r["ref_points"] for r in chosen),
            "ref_max_miss_m_max": max(r["ref_max_miss_m"] for r in chosen),
        }

    problems, hints = [], []
    if counts["invalid"] or orphans or unexpected:
        problems.append(f"{counts['invalid']} invalid / {len(orphans)} orphan / {len(unexpected)} unexpected clips")
        hints.append("invalid/orphan clips were quarantined to <raw>/_incomplete: re-run run_render.sh"
                     if args.quarantine_invalid else "re-run with --quarantine-invalid, then run_render.sh")
    if counts["missing_quota_open"] and not args.allow_missing:
        problems.append(f"{counts['missing_quota_open']} selected episodes have no clip while their scene's "
                        "quota walk is open (render failures: see the worker logs)")
        hints.append("re-run run_render.sh (it resumes and renders only episodes without a clip); if an episode "
                     "cannot be rendered, finalize with --allow-missing REASON (the next rank fills in)")
    if counts["off_route"] and not args.allow_off_route:
        problems.append(f"{counts['off_route']} clips the walk would keep did not follow their route")
        hints.append("check num_frames against COLLECT_MAX_STEPS + 1 and the follower in the worker log; to "
                     "re-render, move the clip out of the data root and re-run run_render.sh (larger "
                     "COLLECT_MAX_STEPS if it hit the cap); else --allow-off-route REASON (clip not kept)")

    report = OrderedDict([
        ("schema", SCHEMA), ("tier", tier), ("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("finalized", not problems and not args.dry_run), ("problems", problems), ("overrides", overrides),
        ("dry_run", bool(args.dry_run)), ("smoke", bool(manifest.get("smoke"))),
        ("rule", KEEP_RULES[tier]), ("min_frames", min_frames),
        ("data_root", str(data_root)), ("clip_list", str(clip_list)),
        ("manifest", {"path": str(args.manifest), "sha256": _sha256(args.manifest)}),
        ("counts", counts), ("scenes", scene_summary), ("scenes_below_quota", below),
        ("routes_by_pattern", routes), ("route_tolerance_m", args.route_tolerance),
        ("unexpected_clips", [{"clip": f"{c['scene_dir']}/{c['clip']}", "episode_id": c.get("episode_id"),
                               "location": c["location"]} for c in unexpected]),
        ("orphan_clips", [{"clip": f"{c['scene_dir']}/{c['clip']}", "reason": c["reason"],
                           "location": c["location"]} for c in orphans]),
        ("clips", rows),
    ])
    print(f"[{tier}] " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for pattern, stats in routes.items():
        print(f"[{tier}] pattern={pattern}: {stats}")
    if below:
        print(f"[{tier}] below quota: " + ", ".join(f"{s} {v['kept']}/{quota}" for s, v in below.items()))
    for r in rows:
        if r["status"] in ("short", "invalid", "duplicate", "missing", "off_route"):
            print(f"[{tier}] {r['status']}: {_describe(r)}")
    for name, reason in overrides.items():
        print(f"[{tier}] override --{name.replace('_', '-')}: {reason}")

    if args.dry_run:
        print(json.dumps({k: report[k] for k in ("counts", "scenes_below_quota", "problems")}, indent=2))
        return 2 if problems else 0

    moved = []
    for clip, target_root in quarantine_moves + ([] if problems else final_moves):
        target = _move(clip["path"], target_root)
        moved.append({"clip": f"{clip['scene_dir']}/{clip['clip']}", "to": str(target)})
        print(f"[{tier}] moved {clip['scene_dir']}/{clip['clip']} -> {target}")
    report["moved"] = moved
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[{tier}] report: {report_path}")

    if problems:
        for problem in problems:
            print(f"[{tier}] ERROR: {problem}", file=sys.stderr)
        for hint in hints:
            print(f"[{tier}] next: {hint}", file=sys.stderr)
        print(f"[{tier}] clip list and marker NOT written; short/duplicate/surplus clips NOT moved", file=sys.stderr)
        if marker.exists():
            print(f"[{tier}] WARNING: {marker} from an earlier finalize exists; its clip list is stale",
                  file=sys.stderr)
        return 2

    def _listing(status, fmt):
        return ", ".join(fmt(r) for r in rows if r["status"] == status) or "none"

    header = [
        f"EXP-18 tier {tier} ({common.TIERS[tier]['label']}) clip list; "
        f"scripts/exp18/render/finalize_clip_lists.py {report['created_utc']}",
        f"data_root: {data_root} (keys are <scene>/<clip> relative to it)",
        f"rule: {KEEP_RULES[tier]}",
        f"manifest: {args.manifest} sha256={report['manifest']['sha256']} smoke={report['smoke']}",
        "counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        f"dropped short (< {min_frames} frames, moved to {excluded_short}): "
        + _listing("short", lambda r: f"{r['clip_key']}(ep={r['episode_id']},T={r['num_frames']})"),
        "surplus (>= %d frames beyond quota): %d, %s" % (
            min_frames, counts["surplus"],
            f"moved to {excluded_surplus}" if args.move_surplus else "left in the data root, not listed"),
        "missing (selected but no clip): " + _listing("missing", lambda r: "%s:%s(rank=%s%s)" % (
            r["scene"], r["episode_id"], r["rank"], "" if r["quota_open"] else ", beyond quota")),
        "off-route (did not follow its route, not kept): " + _listing("off_route", lambda r: "%s(ep=%s,%d/%d refs)" % (
            r["clip_key"], r["episode_id"], r["ref_reached_in_order"], r["ref_points"])),
        "overrides: " + ("; ".join(f"--{k.replace('_', '-')}: {v}" for k, v in overrides.items()) or "none"),
        "scenes below quota: " + (", ".join(f"{s} {v['kept']}/{quota}" for s, v in below.items()) or "none"),
        f"report: {report_path}",
    ]
    common.write_clip_list(tier, keys, header="\n".join(header), path=clip_list)
    marker.write_text(json.dumps({"tier": tier, "created_utc": report["created_utc"], "clip_list": str(clip_list),
                                  "report": str(report_path), "counts": counts, "overrides": overrides},
                                 indent=2) + "\n")
    print(f"[{tier}] clip list: {clip_list} ({len(keys)} clips); marker: {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
