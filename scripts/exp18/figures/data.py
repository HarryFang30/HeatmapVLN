"""Data access for the EXP-18 figures: dump rows, surround RGB and the top-down level.

A History Head dump (``dump_history_predictions.py``, schema v2) holds, for one
clip, a few query rows (cache endpoints t = 19, 27, ..., final frame).  Each row
asks the head where each of K = 8 past camera centres lies in the current
first-person surround view.  This module turns one row into a :class:`CaseRow`
with everything a figure needs, computed exactly as ``compute_metrics.py`` does:

* GT bearing = atan2(left, forward) of the GT rel pose (left-positive degrees);
* predicted bearing = joint argmax of ``heatmaps_gated`` (the dump's exact f32
  ``*_gated_argmax``) converted by ``scripts/exp18/geometry.py``;
* joint PCK@8 = 5-way view class correct and argmax within 8 px in the GT view
  (``validate.py``), over GT-visible slots;
* the constant "always behind" floor: class back, peak (32, 32), never none.

The miss rule every figure uses (orchestrator decision D1): a slot is
**missed** iff it is GT-visible and fails joint PCK@8 -- 5-way view class wrong
(``visibility_logits`` with class 0 = none, vs ``gt_view_class``), or the
per-view argmax (``pred_*_view_peak_yx``) more than 8 px from
``gt_view_peak_yx`` in the GT view.  These are exactly the fields and the rule
of ``compute_metrics.py``, so the misses of a row number ``n_visible - hits``
of its header.  ``CaseRow.misses`` / ``CaseRow.misses_with_peak`` /
``CaseRow.misses_predicted_none`` split them by what a figure can draw (an x at
the joint argmax of ``heatmaps_gated``, ``pred_*_gated_argmax``, or a note
when the prediction says "not visible").  ``row_notes`` groups the slots that
need a note (no view shows them, or visible but predicted not visible) into
runs of consecutive slots, and ``check_accounting`` asserts that every valid
slot of a row ends up with a badge or a note and every miss is numbered.

Designed routes (tier E, ``fig_routes``): ``route_split`` (frame where the
route turns back), ``route_key_rows`` (the ``select_cases`` pattern rule on one
dump) and ``front_tally`` (per scored row: past positions in the front view and
the prediction's joint PCK@8 hits among them).

Nothing here draws; ``common_draw.py`` and the figure modules do.  numpy + PIL.
"""
from __future__ import annotations

import glob
import io
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from scripts.exp18 import geometry as geo
from scripts.exp18.topdown.topdown_io import Level, load_topdown

ARMS = ("vo", "gt")
K = 8
PCK_RADIUS_PX = 8
DEFAULT_CAMERA_HEIGHT_M = 1.25  # camera above the navmesh in every EXP-18 render
SAME_SPOT_M = 0.05  # history slots closer than this share one marker ("1–6")
AT_ROBOT_M = 0.1  # a past position closer than this to the robot: "at the robot" (the previous frame, typically)
NONE_THRESHOLD = 0.5  # P(not visible) above this: the prediction says "not visible" (no x is drawn)

TIER_NAMES = {
    "A": "training scene",
    "B": "held-out scene",
    "C": "unseen scene",
    "D": "HM3D scene (cross-dataset)",
    "E": "designed route",
    "F": "real-world clip",
}
TIER_NAMES_ZH = {
    "A": "训练场景",
    "B": "留出场景",
    "C": "未见场景",
    "D": "HM3D 场景（跨数据集）",
    "E": "设计路线",
    "F": "真实场景",
}


# --------------------------------------------------------------------------- #
# Dump
# --------------------------------------------------------------------------- #
@dataclass
class Dump:
    """One dump npz, loaded eagerly (a clip's arrays are a few MB)."""

    path: Path
    arrays: Dict[str, np.ndarray]
    meta: dict
    tier: str
    scene: str
    clip: str
    clip_dir: str
    episode_id: str
    frame_count: int
    arms: List[str]
    positions: np.ndarray  # [T, 3] camera centres of every frame (world, y up)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    @property
    def n_rows(self) -> int:
        return int(len(self.arrays["current_frame_ids"]))

    def reference_path(self) -> Optional[np.ndarray]:
        ref = self.meta.get("reference_path")
        if isinstance(ref, str):
            try:
                ref = json.loads(ref)
            except ValueError:
                return None
        return None if not ref else np.asarray(ref, dtype=np.float64).reshape(-1, 3)

    def camera_height(self) -> float:
        """Camera height above the navmesh: frame 0 vs the route start, else the render default."""
        ref = self.reference_path()
        if ref is not None and len(self.positions):
            h = float(self.positions[0, 1] - ref[0, 1])
            if 0.3 < h < 2.5:
                return h
        return DEFAULT_CAMERA_HEIGHT_M

    def query_rows(self, arm: Optional[str]) -> List[int]:
        """Scored rows (cache endpoints) on which ``arm`` has a prediction (``None``: every endpoint row)."""
        t = self.arrays["current_frame_ids"].astype(np.int64)
        if "cache_endpoint_frame_ids" in self.arrays and self.arrays["cache_endpoint_frame_ids"].size:
            ok = np.isin(t, self.arrays["cache_endpoint_frame_ids"].astype(np.int64))
        else:
            ok = np.ones(len(t), bool)
        if arm == "vo" and "vo_available" in self.arrays:
            ok &= self.arrays["vo_available"].astype(bool)
        return [int(i) for i in np.nonzero(ok)[0]]

    def tier_name(self, lang: str = "en") -> str:
        names = TIER_NAMES_ZH if lang == "zh" else TIER_NAMES
        return names.get(self.tier, f"tier {self.tier}")


def load_dump(path) -> Dump:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    def s(key: str, default: str = "") -> str:
        return str(arrays[key]) if key in arrays else default

    meta = json.loads(s("meta_json", "{}") or "{}")
    if "arms" in arrays:
        arms = [str(a) for a in arrays["arms"]]
    else:
        arms = [a for a in ARMS if f"pred_{a}_heatmaps_gated" in arrays]
    return Dump(
        path=path,
        arrays=arrays,
        meta=meta,
        tier=s("tier", "?"),
        scene=s("scene") or s("scene_id") or str(meta.get("scene_id", "")),
        clip=s("clip"),
        clip_dir=s("clip_dir"),
        episode_id=s("episode_id") or str(meta.get("episode_id", "")),
        frame_count=int(arrays["frame_count"]) if "frame_count" in arrays else int(len(arrays["clip_c2w"])),
        arms=arms,
        positions=geo.c2w_position(arrays["clip_c2w"].astype(np.float64)),
    )


# --------------------------------------------------------------------------- #
# One query row
# --------------------------------------------------------------------------- #
@dataclass
class ArmPrediction:
    """One pose arm's prediction for the 8 slots of a row."""

    arm: str
    gated: np.ndarray  # [8, 4, 64, 64] heatmaps_gated (each slot sums to 1 - none_p)
    none_p: np.ndarray  # [8]
    peak_view: np.ndarray  # [8] joint argmax view (-1: none)
    peak_bearing: np.ndarray  # [8] degrees, left-positive
    peak_elev: np.ndarray  # [8] degrees, up-positive
    err: np.ndarray  # [8] |predicted - GT bearing|, NaN where GT is not visible
    joint8: np.ndarray  # [8] bool, joint PCK@8 hit
    pose_bearing: np.ndarray  # [8] bearing of the rel pose this arm was given
    pose_err: np.ndarray  # [8] |pose bearing - GT bearing| (0 for the GT arm)


@dataclass
class CaseRow:
    index: int
    frame: int
    is_final: bool
    cur_c2w: np.ndarray  # [4, 4]
    hist_c2w: np.ndarray  # [8, 4, 4]
    hist_frames: np.ndarray  # [8]
    valid: np.ndarray  # [8] history_mask
    visible: np.ndarray  # [8] GT-visible (view class > 0) and valid
    gt_class: np.ndarray  # [8] 0 none, 1..4 F R B L
    gt_bearing: np.ndarray  # [8]
    gt_dist: np.ndarray  # [8] metres on the floor plane
    gt_maps: np.ndarray  # [8, 4, 64, 64]
    arms: Dict[str, ArmPrediction]
    floor_err: np.ndarray  # [8] always-behind bearing error
    floor_joint8: np.ndarray  # [8] always-behind joint PCK@8 hit
    groups: List[List[int]] = field(default_factory=list)  # visible slots sharing a camera centre
    gt_peak_elev: Optional[np.ndarray] = None  # [8] elevation (deg, up-positive) of the GT peak, NaN if not visible

    @property
    def hist_pos(self) -> np.ndarray:
        return geo.c2w_position(self.hist_c2w)

    @property
    def cur_pos(self) -> np.ndarray:
        return geo.c2w_position(self.cur_c2w)

    @property
    def n_visible(self) -> int:
        return int(self.visible.sum())

    def summary(self, arm: str) -> dict:
        """median / max bearing error and joint PCK@8 hits over GT-visible slots."""
        v = self.visible
        if arm == "floor":
            err, hits = self.floor_err, self.floor_joint8
        else:
            err, hits = self.arms[arm].err, self.arms[arm].joint8
        e = err[v & np.isfinite(err)]
        return {
            "median": float(np.median(e)) if e.size else float("nan"),
            "max": float(np.max(e)) if e.size else float("nan"),
            "hits": int((hits & v).sum()),
            "n": int(v.sum()),
        }

    def invisible_slots(self) -> List[int]:
        return [int(k) for k in np.nonzero(self.valid & ~self.visible)[0]]

    # ---- the D1 miss rule (see the module doc) --------------------------------
    def misses(self, arm: str) -> List[int]:
        """GT-visible slots failing joint PCK@8 (the header's ``n - hits``)."""
        return [int(k) for k in np.nonzero(self.visible & ~self.arms[arm].joint8)[0]]

    def peak_slots(self, arm: str) -> List[int]:
        """GT-visible slots whose prediction is drawn as an x: P(not visible) <= 0.5 and a joint argmax."""
        p = self.arms[arm]
        return [int(k) for k in np.nonzero(self.visible & (p.none_p <= NONE_THRESHOLD) & (p.peak_view >= 0))[0]]

    def misses_with_peak(self, arm: str) -> List[int]:
        """Missed slots with an x (numbered on the prediction row)."""
        drawn = set(self.peak_slots(arm))
        return [k for k in self.misses(arm) if k in drawn]

    def misses_predicted_none(self, arm: str) -> List[int]:
        """Missed slots without an x (the model calls them not visible): numbered in a note."""
        drawn = set(self.peak_slots(arm))
        return [k for k in self.misses(arm) if k not in drawn]

    def false_positive_slots(self, arm: str) -> List[int]:
        """Slots no view shows that the prediction still calls visible (P(not visible) <= 0.5).

        Not scored (joint PCK@8 counts GT-visible slots only); their map shows in
        the prediction row, weighted by the predicted visibility, and their note
        gives P(not visible)."""
        p = self.arms[arm]
        return [k for k in self.invisible_slots() if p.none_p[k] <= NONE_THRESHOLD]


def group_label(group: Sequence[int]) -> str:
    """Slot numbers are 1-based in every figure: [0] -> "1", [0..5] -> "1–6"."""
    return str(group[0] + 1) if len(group) == 1 else f"{group[0] + 1}–{group[-1] + 1}"


def slot_runs(slots: Sequence[int]) -> List[List[int]]:
    """Runs of consecutive slot indices: [0, 1, 2, 5] -> [[0, 1, 2], [5]]."""
    runs: List[List[int]] = []
    for k in sorted(int(s) for s in slots):
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


NOTE_KINDS = ("previous", "at_robot", "not_visible", "predicted_none")


def row_notes(row: "CaseRow", arm: str) -> List[dict]:
    """The notes a figure prints for one row, as data: ``{"kind", "slots", "p"}`` per run of consecutive slots.

    kinds (in this order):

    * ``previous``: slot 8 (the previous frame) at the robot (closer than
      ``AT_ROBOT_M``) -- no view can show it;
    * ``at_robot``: other past positions at the robot (the robot stood still);
    * ``not_visible``: no view shows the past position (out of sight);
    * ``predicted_none``: GT-visible, but the prediction says not visible
      (P(not visible) > 0.5) -- a miss (D1) without an x, numbered in its note.

    ``p`` = P(not visible) of each slot of the run.  A run of ``at_robot`` that
    reaches slot 8 absorbs it (one note "6–8 at the robot").
    """
    p = row.arms[arm].none_p
    inv = row.invisible_slots()
    at = [k for k in inv if row.gt_dist[k] < AT_ROBOT_M]
    unseen = [k for k in inv if row.gt_dist[k] >= AT_ROBOT_M]
    out: List[dict] = []
    for run in slot_runs(at):
        kind = "previous" if run == [K - 1] else "at_robot"
        out.append({"kind": kind, "slots": run, "p": [float(p[k]) for k in run]})
    for run in slot_runs(unseen):
        out.append({"kind": "not_visible", "slots": run, "p": [float(p[k]) for k in run]})
    for run in slot_runs(row.misses_predicted_none(arm)):
        out.append({"kind": "predicted_none", "slots": run, "p": [float(p[k]) for k in run]})
    return out


def check_accounting(row: "CaseRow", arm: str, badge_slots: Sequence[int], note_slots: Sequence[int],
                     numbered_on_row: Sequence[int], numbered_in_notes: Sequence[int]) -> None:
    """Raise ``RuntimeError`` unless every valid slot of ``row`` has a badge or a note, and the numbered misses
    (on the prediction row + in notes) are exactly ``row.misses(arm)`` (= header n - hits), each once."""
    valid = {int(k) for k in np.nonzero(row.valid)[0]}
    shown = set(int(k) for k in badge_slots) | set(int(k) for k in note_slots)
    missing = sorted(valid - shown)
    numbered = [int(k) for k in numbered_on_row] + [int(k) for k in numbered_in_notes]
    s = row.summary(arm)
    problems = []
    if missing:
        problems.append(f"slots {[k + 1 for k in missing]} have neither a badge nor a note")
    if sorted(numbered) != sorted(row.misses(arm)) or len(numbered) != len(set(numbered)):
        problems.append(f"numbered misses {sorted(k + 1 for k in numbered)} != joint PCK@8 misses "
                        f"{[k + 1 for k in row.misses(arm)]}")
    if len(numbered) != s["n"] - s["hits"]:
        problems.append(f"{len(numbered)} numbered misses, header says {s['n'] - s['hits']} (n {s['n']} - hits "
                        f"{s['hits']})")
    if problems:
        raise RuntimeError(f"frame {row.frame} (row {row.index}): " + "; ".join(problems))


def _group_same_spot(pos: np.ndarray, slots: Sequence[int], tol: float = SAME_SPOT_M) -> List[List[int]]:
    groups: List[List[int]] = []
    for k in slots:
        for g in groups:
            if k == g[-1] + 1 and np.linalg.norm(pos[k] - pos[g[0]]) < tol:
                g.append(int(k))
                break
        else:
            groups.append([int(k)])
    return groups


def _take_view(a: np.ndarray, view: np.ndarray) -> np.ndarray:
    return a[np.arange(len(view)), view]


def case_row(dump: Dump, i: int) -> CaseRow:
    z = dump.arrays
    valid = z["history_mask"][i].astype(bool)
    gt_class = z["gt_view_class"][i].astype(np.int64)
    visible = valid & (gt_class > 0)
    gt_rel = z["gt_rel_poses"][i].astype(np.float64)
    gt_bearing = np.asarray(geo.bearing_from_rel_pose(gt_rel[:, 0], gt_rel[:, 1]), dtype=np.float64)
    tv = np.clip(gt_class - 1, 0, 3)
    gt_yx = _take_view(z["gt_view_peak_yx"][i].astype(np.int64), tv)  # [8, 2]

    arms: Dict[str, ArmPrediction] = {}
    for arm in dump.arms:
        if f"pred_{arm}_heatmaps_gated" not in z:
            continue
        ga = z[f"pred_{arm}_gated_argmax"][i].astype(np.int64)  # exact f32 argmax (view, row, col)
        pb, pel = geo.pixel_to_bearing_elev(np.maximum(ga[:, 0], 0), ga[:, 2], ga[:, 1])
        pb = np.where(ga[:, 0] >= 0, np.asarray(pb, dtype=np.float64), np.nan)
        logits = z[f"pred_{arm}_visibility_logits"][i].astype(np.float64)
        pred_cls = np.concatenate([np.zeros((K, 1)), logits], axis=1).argmax(1)
        p_yx = _take_view(z[f"pred_{arm}_view_peak_yx"][i].astype(np.int64), tv)
        joint8 = visible & (pred_cls == gt_class) & (((p_yx - gt_yx) ** 2).sum(1) <= PCK_RADIUS_PX ** 2)
        rel = z[f"{arm}_rel_poses"][i].astype(np.float64) if f"{arm}_rel_poses" in z else gt_rel
        pose_b = np.asarray(geo.bearing_from_rel_pose(rel[:, 0], rel[:, 1]), dtype=np.float64)
        arms[arm] = ArmPrediction(
            arm=arm,
            gated=z[f"pred_{arm}_heatmaps_gated"][i].astype(np.float32),
            none_p=z[f"pred_{arm}_none_probability"][i].astype(np.float64),
            peak_view=ga[:, 0],
            peak_bearing=pb,
            peak_elev=np.asarray(pel, dtype=np.float64),
            err=np.where(visible, np.asarray(geo.circular_abs_diff(pb, gt_bearing)), np.nan),
            joint8=joint8,
            pose_bearing=pose_b,
            pose_err=np.asarray(geo.circular_abs_diff(pose_b, gt_bearing)),
        )

    floor_sq = ((gt_yx - np.asarray(geo.FLOOR_PEAK_YX)) ** 2).sum(1)
    cur = z["current_c2w"][i].astype(np.float64)
    hist = z["history_c2w"][i].astype(np.float64)
    fwd, left, _ = geo.world_to_rel(cur, geo.c2w_position(hist))
    row = CaseRow(
        index=int(i),
        frame=int(z["current_frame_ids"][i]),
        is_final=int(z["current_frame_ids"][i]) == dump.frame_count - 1,
        cur_c2w=cur,
        hist_c2w=hist,
        hist_frames=z["history_frame_ids"][i].astype(np.int64),
        valid=valid,
        visible=visible,
        gt_class=gt_class,
        gt_bearing=gt_bearing,
        gt_dist=np.hypot(gt_rel[:, 0], gt_rel[:, 1]),
        gt_maps=z["gt_heatmap"][i].astype(np.float32),
        arms=arms,
        floor_err=np.where(visible, np.asarray(geo.circular_abs_diff(180.0, gt_bearing)), np.nan),
        floor_joint8=visible & (gt_class == 1 + geo.BACK) & (floor_sq <= PCK_RADIUS_PX ** 2),
    )
    # the GT rel pose and the stored c2w must agree (catches a convention slip early)
    b_c2w = np.asarray(geo.bearing_from_rel_pose(fwd, left))
    check = visible & (row.gt_dist > 0.05)
    if float(np.max(np.asarray(geo.circular_abs_diff(b_c2w, gt_bearing))[check], initial=0.0)) > 1.0:
        raise ValueError(f"{dump.path}: row {i}: gt_rel_poses and history_c2w disagree")
    row.groups = _group_same_spot(row.hist_pos, [int(k) for k in np.nonzero(visible)[0]])
    _, gt_el = geo.pixel_to_bearing_elev(tv, gt_yx[:, 1], gt_yx[:, 0])
    row.gt_peak_elev = np.where(visible, np.asarray(gt_el, dtype=np.float64), np.nan)
    return row


def key_rows(dump: Dump, arm: str = "vo") -> List[int]:
    """Pre-registered key positions: first scored row, widest GT-visible bearing span, final row.

    The widest span breaks ties by the earliest row; when it coincides with the
    first or final row the middle scored row is used instead.
    """
    rows = dump.query_rows(arm)
    if not rows:
        raise ValueError(f"{dump.path}: no scored rows with arm {arm!r}")
    spans = []
    for i in rows:
        r = case_row(dump, i)
        spans.append(geo.circular_range_deg(r.gt_bearing[r.visible]))
    first, last = rows[0], rows[-1]
    widest = rows[int(np.argmax(spans))]
    if widest in (first, last) and len(rows) >= 3:
        widest = rows[len(rows) // 2]
    out: List[int] = []
    for i in (first, widest, last):
        if i not in out:
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Designed routes (tier E): where the route turns back, key rows, front-view tally
# --------------------------------------------------------------------------- #
ROUTE_PATTERNS = ("out_and_back", "loop")
ROUTE_ROLES = ("before_turnaround", "turnaround", "after_turnaround", "return_to_start")
BACK_AT_START_M = 1.0  # a final row closer than this to frame 0 counts as "back at the start"


def route_split(dump: Dump, pattern: str) -> dict:
    """Frame at which the route turns back: {"frame", "rule"}.

    out_and_back with a palindromic ``reference_path``: the route's own
    turnaround (``geo.out_and_back_turnaround``, the frame compute_metrics
    stores as ``turnaround_frame``).  Otherwise (loops, and any clip without
    such a path): the frame farthest (3D euclidean) from frame 0, first max.
    """
    if pattern == "out_and_back":
        turn = geo.out_and_back_turnaround(dump.positions, dump.reference_path())
        if turn["frame"] >= 0:
            return {"frame": int(turn["frame"]), "rule": "route turnaround (reference_path midpoint)"}
    d = np.linalg.norm(dump.positions - dump.positions[0], axis=1)
    return {"frame": int(np.argmax(d)), "rule": "frame farthest from the start"}


def route_key_rows(dump: Dump, pattern: str, arm: str = "vo", turnaround_frame: Optional[int] = None):
    """Key rows of the route-pattern figure, the rule of ``select_cases.pattern_figure`` on one dump.

    Returns ``(rows, roles, info)``.  Turnaround row: out_and_back with a
    route turnaround frame -> the scored row nearest to it (ties: earlier);
    otherwise the scored row farthest (3D) from frame 0 (ties: earliest).  Key
    rows: the scored rows just before and after it, the row itself, and the
    final row.  ``cases.json`` stays authoritative for tier E; this is the
    fallback for a dump without a pick (and for development stand-ins).
    """
    rows = dump.query_rows(arm)
    if not rows:
        raise ValueError(f"{dump.path}: no scored rows with arm {arm!r}")
    t = dump.arrays["current_frame_ids"][rows].astype(np.int64)
    cur = geo.c2w_position(dump.arrays["current_c2w"][rows].astype(np.float64))
    dist = np.linalg.norm(cur - dump.positions[0], axis=-1)
    frame = turnaround_frame
    if frame is None and pattern == "out_and_back":
        turn = geo.out_and_back_turnaround(dump.positions, dump.reference_path())
        frame = turn["frame"] if turn["frame"] >= 0 else None
    if pattern == "out_and_back" and frame is not None and frame >= 0:
        j = int(np.abs(t - int(frame)).argmin())
        rule = "route turnaround frame -> nearest scored row"
    else:
        j = int(dist.argmax())
        rule = "scored row farthest (3D euclidean) from the first frame"
    picked, roles = [], []
    for jj, role in ((j - 1, "before_turnaround"), (j, "turnaround"), (j + 1, "after_turnaround")):
        if 0 <= jj < len(rows):
            picked.append(rows[jj])
            roles.append(role)
    final = [i for i in rows if int(dump.arrays["current_frame_ids"][i]) == dump.frame_count - 1]
    last = final[-1] if final else rows[-1]
    if last not in picked:
        picked.append(last)
        roles.append("return_to_start")
    info = {"turnaround_rule": rule, "turnaround_frame": None if frame is None else int(frame),
            "turnaround_row_t": int(t[j])}
    return picked, roles, info


def front_tally(dump: Dump, arm: str = "vo") -> List[dict]:
    """Per scored row: GT-visible past positions, those in the front view, and the front ones predicted right.

    "Predicted right" is the joint PCK@8 hit of ``case_row`` (validate.py rule).
    """
    out = []
    for i in dump.query_rows(arm):
        r = case_row(dump, i)
        front = r.visible & (r.gt_class == 1 + geo.FRONT)
        out.append({"row": int(i), "t": r.frame, "n_visible": r.n_visible, "n_front": int(front.sum()),
                    "hits_front": int((r.arms[arm].joint8 & front).sum()),
                    "dist_from_start_m": float(np.linalg.norm(r.cur_pos - dump.positions[0]))})
    return out


# --------------------------------------------------------------------------- #
# Heat composites (what the heat rows show)
# --------------------------------------------------------------------------- #
def gt_composite(row: CaseRow) -> np.ndarray:
    """[4, 64, 64]: max over GT-visible slots of each slot's map divided by its own peak."""
    m = row.gt_maps
    peak = m.reshape(K, -1).max(1)
    norm = m / np.maximum(peak, 1e-12)[:, None, None, None]
    norm = np.where(row.visible[:, None, None, None], norm, 0.0)
    return norm.max(0)


def pred_composite(row: CaseRow, arm: str) -> np.ndarray:
    """[4, 64, 64]: max over valid slots of (slot map / its peak) x (1 - P(none))."""
    p = row.arms[arm]
    peak = p.gated.reshape(K, -1).max(1)
    norm = p.gated / np.maximum(peak, 1e-12)[:, None, None, None] * (1.0 - p.none_p)[:, None, None, None]
    norm = np.where(row.valid[:, None, None, None], norm, 0.0)
    return norm.max(0)


# --------------------------------------------------------------------------- #
# Surround RGB
# --------------------------------------------------------------------------- #
def resolve_clip_dir(dump: Dump, clip_root_override=None) -> Path:
    """The clip directory recorded in the dump, or its copy under ``clip_root_override``.

    Under an override root the clip is looked up as ``<root>/<scene>/<clip>``,
    then ``<root>/<clip>``, then ``<root>`` itself.
    """
    recorded = Path(dump.clip_dir) if dump.clip_dir else None
    candidates = []
    if clip_root_override:
        root = Path(clip_root_override)
        tail = recorded.parts[-2:] if recorded is not None else (dump.scene, dump.clip)
        candidates += [root.joinpath(*tail), root / tail[-1], root]
    if recorded is not None:
        candidates.append(recorded)
    for c in candidates:
        if (c / "chunks").is_dir():
            return c
    raise FileNotFoundError(f"no clip directory with chunks/ among {[str(c) for c in candidates]}")


@lru_cache(maxsize=8)
def _frame_index(clip_dir: str) -> Dict[int, tuple]:
    index = {}
    for f in sorted(glob.glob(str(Path(clip_dir) / "chunks" / "chunk_*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            for j, fid in enumerate(z["frame_ids"]):
                index[int(fid)] = (f, j)
    return index


def surround_views(clip_dir, frame: int) -> np.ndarray:
    """[4, H, W, 3] uint8 RGB of views front, right, back, left at ``frame``.

    Chunks store JPEG bytes in object arrays, hence ``allow_pickle=True`` on
    the clip's own data files.
    """
    index = _frame_index(str(clip_dir))
    if frame not in index:
        raise KeyError(f"frame {frame} not in {clip_dir}")
    path, j = index[frame]
    with np.load(path, allow_pickle=True) as z:
        return np.stack([np.asarray(Image.open(io.BytesIO(z[f"rgb_{v}"][j])).convert("RGB")) for v in geo.VIEW_NAMES])


# --------------------------------------------------------------------------- #
# Top-down
# --------------------------------------------------------------------------- #
def topdown_level(dump: Dump, camera_y: float, root=None) -> Level:
    """The floor level under a camera at height ``camera_y`` (world y)."""
    scene = load_topdown(dump.scene, root=root)
    return scene.pick_level(float(camera_y) - dump.camera_height())
