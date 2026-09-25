"""EXP-19 figures v2: the ONLINE timeline, i.e. how the affordance maps evolve over a whole rerun.

Input: ``records_v2/<ep_key>_timeline.{npz,json}`` (schema ``exp19-timeline-v1``,
written by ``scripts/exp19/build_timeline.py``; presentation only).  Per ready
call r (a trajectory call where the history head, bridge and future head ran):

* ``step[r]`` .. ``next_step[r]``: the steps until the next call of any kind;
* ``hist_ring[r, 360]``: predicted history affordance map (per slot divided by
  its own peak x (1 - P(none)), max over slots) on the equiangular ring, max
  over elevation; bin i is centred on bearing +179.5 - i (left-positive);
* ``hist_pred_peak_bearing[r, 8]`` / ``hist_pred_conf[r, 8]`` (1 - P(none));
* ``hist_gt_bearing[r, 8]``: true direction of each past frame (NaN when padded
  or not visible in any view);
* ``fut_ring[r, 4, 360]``: predicted future affordance map per time bin;
* ``s1_path_bearing`` / ``s1_path_dist [r, 33]``: the System1 mean path;
* episode level: every call's step / kind / ready flag, the first ready step,
  the episode length and the key moments (label, call index, step).

The panel (``draw_timeline``): x = step, y = bearing, UP = LEFT on every panel
(the future sub-panel, the history panel and the turn track).  The HISTORY panel
runs, top to bottom, from ahead through left, behind (in the middle) and right
back to ahead (``hist_y``: y = -bearing mod 360, so y = 0 and 360 are both
ahead), so directions behind the robot sit in the middle of the panel in one
piece, and a left turn (which swings what is behind towards the robot's left)
moves them up, like the turn track's bar.  The panel runs ``WRAP_PAD`` deg past
both ends with the field wrapped (periodic axis); every mark is drawn once, at
its own bearing (no wrapped copy), unclipped so a mark right ahead is whole.
The future sub-panel is centred on ahead (+180 top, 0 middle, -180 bottom).
Each ready call paints a column from its step to the next call with its history
ring (faint orange); the column is the call's own action chunk (the next call
comes when the chunk is done).  Its marks are spread across the column by slot,
slot 1 (the oldest past frame) at the left: in slot k's position, a small
dark-orange dot = the predicted peak of slot k when the head calls it visible
(1 - P(none) >= 0.5) and a small hollow blue circle = the true direction of past
frame k (only when that past frame is visible from here), so every dot sits in
the same x position as its own circle.  ``mark_plan`` decides which slots a
column marks so that neighbouring circles keep ``GAP_PT`` of air: all 8 where
the column is wide enough, else past frames 1, 4 and 8 (``OVERVIEW_SLOTS``; the
overview always), and on long reruns whose columns are narrower than that the 8
marks of a call are stacked at the column's centre for every n-th call only
(``marker_stride``; the key moments always; the legend's timeline entry says
so, nothing is written over the data), the fields for every call.
Steps without a map are hatched (warm-up) or tinted (System 2 answered with
turns / STOP).  K1-K4 are ink hairlines at their calls' steps (the left edge of
the call's column) with a badge on top; badges of key moments closer than a
badge width are raised one level.  Below, a thin sub-panel shows the predicted
future affordance map (time bins painted early -> late, darker = later) and the
endpoint bearing of the System 1 path (one ink dot at the column's centre), and
a track of the executed turns per step (up = left, down = right;
``step_action``).

Long spans without an affordance map can be compressed into fixed narrow blocks
(``compress_axis``): the warm-up (steps 0 .. W, ending at the first call whose
history input is ready, ``pose_ready`` in ``records/<ep_key>.json``; without the
record, at the first ready call), and -- only when the ready calls cover less
than ``NOMAP_COVER_MAX`` of the rerun -- every span of at least
``NOMAP_MIN_STEPS`` steps where System 2 answered with turns / STOP.  The step
axis is piecewise linear (``Timeline.x_of``), with a break mark where the scale
changes and the steps written at each block's ends and at the axis' end; the
unit ("step") sits left of the axis, so no tick is ever dropped for it.
Both fields are drawn display-smoothed along the bearing (``display_hist_rings``
/ ``display_fut_rings``: ``panels_v2.smooth_rings``, a circular Gaussian of
``TL_SMOOTH_DEG`` deg, each ring rescaled to keep its own maximum).  A ring is
painted unchanged across its whole column, so every ripple of a few degrees
(the maps' 4-pixel decoder grid, the cusps between neighbouring slots' peaks)
would show as thin horizontal pinstripes; 3.5 deg is about 0.5 pt on the
printed panel (the strips' 2 deg blur is about 0.6 pt), well under the marks'
size.  The stored rings are unchanged; the dots mark the raw peaks; the captions
say so.  The first ready column carries tiny "1" and "8" at its ends (which
past frame sits where) when the two fit with 2 pt of air between them
(``slot_end_labels``: measured widths, the same rule in every language).
Nothing here draws a pose, a heading or an odometry value: bearings are the
axis of the affordance maps, as on the strips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18 import geometry as geo
from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import panels as pn
from scripts.exp19.figures import panels_v2 as p2

import matplotlib  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

SCHEMA = "exp19-timeline-v1"
N_BEARING = 360
NUM_SLOTS = bd.NUM_SLOTS
NUM_BINS = bd.NUM_BINS
PATH_POINTS = bd.PATH_POINTS
FUT_BAND = 180.0  # the future sub-panel spans all bearings, like the history panel (nothing is cropped)
MIN_PATH_DIST = 0.05  # m: a path endpoint closer than this has no direction

# name -> (kind, shape); "R" ready calls, "C" all calls, "K" key moments
ARRAYS = {
    "call_index": ("int", ("R",)),
    "step": ("int", ("R",)),
    "next_step": ("int", ("R",)),
    "bearing_deg": ("float", (N_BEARING,)),
    "hist_ring": ("float", ("R", N_BEARING)),
    "hist_pred_peak_bearing": ("float", ("R", NUM_SLOTS)),
    "hist_pred_conf": ("float", ("R", NUM_SLOTS)),
    "hist_gt_bearing": ("float", ("R", NUM_SLOTS)),
    "hist_gt_visible": ("bool", ("R", NUM_SLOTS)),
    "fut_ring": ("float", ("R", NUM_BINS, N_BEARING)),
    "s1_path_bearing": ("float", ("R", PATH_POINTS)),
    "s1_path_dist": ("float", ("R", PATH_POINTS)),
    "calls_step": ("int", ("C",)),
    "calls_kind": ("str", ("C",)),
    "calls_ready": ("bool", ("C",)),
    "first_ready_step": ("int", ()),
    "episode_steps": ("int", ()),
    "key_labels": ("str", ("K",)),
    "key_call_index": ("int", ("K",)),
    "key_step": ("int", ("K",)),
}
NAN_OK = ("hist_pred_peak_bearing", "hist_pred_conf", "hist_gt_bearing", "s1_path_bearing", "s1_path_dist")


def call_positions(a: Dict[str, np.ndarray]) -> Dict[int, int]:
    """call index -> position in the calls_* arrays (``calls_index`` when present, else the position itself)."""
    n = len(np.asarray(a["calls_step"]))
    idx = np.asarray(a["calls_index"]) if "calls_index" in a else np.arange(n)
    return {int(c): i for i, c in enumerate(idx)}


def bin_bearings(n: int = N_BEARING) -> np.ndarray:
    """Bearing of each ring bin centre: +179.5 ... -179.5 (bin i <-> +179.5 - i for n = 360)."""
    return geo.ring_column_azimuths(n)


# --------------------------------------------------------------------------- #
# Validation and loading
# --------------------------------------------------------------------------- #
def validate_timeline(meta: dict, a: Dict[str, np.ndarray]) -> List[str]:
    """Every schema problem of a timeline (empty = valid)."""
    errors: List[str] = []
    if meta.get("schema", SCHEMA) != SCHEMA:
        errors.append(f"schema {meta.get('schema')!r}, expected {SCHEMA!r}")
    dims: Dict[str, int] = {}
    for name, (kind, shape) in ARRAYS.items():
        if name not in a:
            errors.append(f"missing array {name!r}")
            continue
        x = np.asarray(a[name])
        if kind == "int" and not np.issubdtype(x.dtype, np.integer):
            errors.append(f"{name}: dtype {x.dtype}, expected int")
        elif kind == "float" and not np.issubdtype(x.dtype, np.floating):
            errors.append(f"{name}: dtype {x.dtype}, expected float")
        elif kind == "bool" and x.dtype != np.bool_:
            errors.append(f"{name}: dtype {x.dtype}, expected bool")
        elif kind == "str" and x.dtype.kind not in "US":
            errors.append(f"{name}: dtype {x.dtype}, expected str")
        if x.ndim != len(shape):
            errors.append(f"{name}: shape {x.shape}, expected {shape}")
            continue
        for d, s in zip(x.shape, shape):
            if isinstance(s, str):
                if dims.setdefault(s, d) != d:
                    errors.append(f"{name}: {s} = {d}, other arrays have {dims[s]}")
            elif s != d:
                errors.append(f"{name}: shape {x.shape}, expected {shape}")
        if kind == "float" and name not in NAN_OK and x.size and not np.isfinite(x).all():
            errors.append(f"{name}: non-finite values")
    if errors:
        return errors
    b = np.asarray(a["bearing_deg"], dtype=np.float64)
    if not np.allclose(b, bin_bearings(), atol=1e-3):
        errors.append("bearing_deg is not +179.5 - i (left edge +180 ... right edge -180)")
    steps, nxt = np.asarray(a["step"]), np.asarray(a["next_step"])
    if len(steps) and (np.any(nxt <= steps) or np.any(np.diff(steps) <= 0)):
        errors.append("ready calls must be in step order with next_step > step")
    n_steps = int(a["episode_steps"])
    if len(steps) and int(nxt.max()) > n_steps:
        errors.append(f"next_step {int(nxt.max())} beyond episode_steps {n_steps}")
    ci, calls_step = np.asarray(a["call_index"]), np.asarray(a["calls_step"])
    pos = call_positions(a)
    if any(int(c) not in pos for c in ci):
        errors.append("call_index has calls that are not in the call list")
    elif len(ci) and np.any(calls_step[[pos[int(c)] for c in ci]] != steps):
        errors.append("call_index does not point at calls_step entries with the same step")
    for lab, kci, kst in zip(a["key_labels"], a["key_call_index"], a["key_step"]):
        rows = np.nonzero(ci == kci)[0]
        if not rows.size:
            errors.append(f"key {lab}: call {int(kci)} is not a ready call")
        elif int(steps[rows[0]]) != int(kst):
            errors.append(f"key {lab}: step {int(kst)} != its call's step {int(steps[rows[0]])}")
    return errors


@dataclass
class Timeline:
    """A loaded timeline; ``pose_ready`` (per call, optional) comes from the episode record."""

    meta: dict
    a: Dict[str, np.ndarray]
    path: Optional[Path] = None
    pose_ready: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    x0: float = 0.0  # left end of the x axis (data units); > 0 when the warm-up is compressed
    wblock: float = 0.0  # width of the compressed warm-up block in step units (0 = the axis is linear)
    blocks: List[Tuple[float, float, float]] = field(default_factory=list)  # compressed no-map spans (s0, s1, width)

    @property
    def R(self) -> int:
        return int(len(self.a["step"]))

    @property
    def steps(self) -> int:
        return int(self.a["episode_steps"])

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    def x_center(self) -> np.ndarray:
        """[R] x of each ready call's column centre."""
        return (self.x_of(np.asarray(self.a["step"], dtype=np.float64))
                + self.x_of(np.asarray(self.a["next_step"], dtype=np.float64))) / 2

    @property
    def compressed(self) -> bool:
        """The warm-up is compressed into a block."""
        return self.wblock > 0.0

    def knots(self) -> Tuple[np.ndarray, np.ndarray]:
        """(steps, x) of the piecewise-linear step axis: slope 1 outside the compressed blocks (x = step from the
        end of a compressed warm-up on, until the first compressed no-map block)."""
        s_k, x_k = [0.0], [float(self.x0)]
        if self.compressed:
            w = float(self.warmup_end())
            s_k.append(w)
            x_k.append(w)
        for a, b, width in sorted(self.blocks):
            xa = x_k[-1] + (float(a) - s_k[-1])
            s_k += [float(a), float(b)]
            x_k += [xa, xa + float(width)]
        n = float(self.steps)
        if n > s_k[-1]:
            x_k.append(x_k[-1] + (n - s_k[-1]))
            s_k.append(n)
        return np.asarray(s_k), np.asarray(x_k)

    def x_of(self, s):
        """x of step s: identity until the first compressed span; a compressed warm-up maps linearly onto
        [x0, W], a compressed no-map span onto its block, and the rest keeps slope 1."""
        if not self.compressed and not self.blocks:
            return s if np.ndim(s) else float(s)
        s_k, x_k = self.knots()
        out = np.interp(np.asarray(s, dtype=np.float64), s_k, x_k)
        return out if np.ndim(s) else float(out)

    def xlim(self) -> Tuple[float, float]:
        return (float(self.x0), float(self.x_of(float(self.steps))))

    def span_steps(self) -> float:
        """Length of the x axis in data units."""
        lo, hi = self.xlim()
        return max(hi - lo, 1e-9)

    def breaks(self) -> List[float]:
        """Steps where the axis scale changes (a break mark each): the end of a compressed warm-up and both ends of
        every compressed no-map block (not the episode's end)."""
        out = [float(self.warmup_end())] if self.compressed else []
        for a, b, _ in sorted(self.blocks):
            out += [float(a), float(b)]
        return sorted({s for s in out if 0.0 < s < float(self.steps)})

    def key_rows(self) -> Dict[str, int]:
        ci = np.asarray(self.a["call_index"])
        out = {}
        for lab, kci in zip(self.a["key_labels"], self.a["key_call_index"]):
            rows = np.nonzero(ci == int(kci))[0]
            if rows.size:
                out[str(lab)] = int(rows[0])
        return out

    def warmup_end(self) -> int:
        """End of the warm-up period (no affordance map yet): the first pose-ready call, else the first ready call."""
        calls_step = np.asarray(self.a["calls_step"])
        if self.pose_ready is not None and len(self.pose_ready) == len(calls_step) and np.any(self.pose_ready):
            return int(calls_step[int(np.argmax(self.pose_ready))])
        first = int(self.a["first_ready_step"])
        return first if 0 <= first <= self.steps else self.steps

    def spans(self) -> List[Tuple[str, float, float]]:
        """(kind, step0, step1) of the steps without an affordance map: "warmup" then "nomap" per non-ready call."""
        calls_step = np.asarray(self.a["calls_step"], dtype=np.int64)
        pos = call_positions(self.a)
        ready_pos = set(pos[int(c)] for c in self.a["call_index"])
        end = self.warmup_end()
        out = [("warmup", 0.0, float(end))] if end > 0 else []
        bounds = list(calls_step) + [self.steps]
        for c in range(len(calls_step)):
            s0, s1 = int(bounds[c]), int(bounds[c + 1])
            if c in ready_pos or s1 <= end:
                continue
            s0 = max(s0, end)
            if s1 > s0:
                if out and out[-1][0] == "nomap" and out[-1][2] == s0:
                    out[-1] = ("nomap", out[-1][1], float(s1))
                else:
                    out.append(("nomap", float(s0), float(s1)))
        return out

    def path_end_bearing(self) -> np.ndarray:
        """[R] bearing of the last System1 waypoint at least MIN_PATH_DIST away (NaN if none)."""
        b = np.asarray(self.a["s1_path_bearing"], dtype=np.float64)
        d = np.asarray(self.a["s1_path_dist"], dtype=np.float64)
        out = np.full(len(b), np.nan)
        for r in range(len(b)):
            ok = np.nonzero(np.isfinite(b[r]) & np.isfinite(d[r]) & (d[r] >= MIN_PATH_DIST))[0]
            if ok.size:
                out[r] = b[r, ok[-1]]
        return out


def timeline_paths(path) -> Tuple[Path, Path]:
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".json", ".npz") else p
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def timeline_path_for(records_v2, ep_key: str) -> Path:
    return Path(records_v2) / f"{ep_key}_timeline.json"


def pose_ready_from_record(record_path) -> Optional[Dict[int, bool]]:
    """call index -> ``pose_ready`` from ``records/<ep_key>.json`` (None when the record or the field is missing)."""
    p = Path(record_path)
    if not p.is_file():
        return None
    calls = json.loads(p.read_text(encoding="utf-8")).get("calls") or []
    if not calls or any("pose_ready" not in c or "call_index" not in c for c in calls):
        return None
    return {int(c["call_index"]): bool(c["pose_ready"]) for c in calls}


def load_timeline(path, record_path=None, validate: bool = True) -> Timeline:
    json_path, npz_path = timeline_paths(path)
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else {}
    with np.load(npz_path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if validate:
        errors = validate_timeline(meta, a)
        if errors:
            raise ValueError(f"{npz_path}: " + "; ".join(errors))
    tl = Timeline(meta, a, json_path)
    if "calls_pose_ready" in a:
        tl.pose_ready = np.asarray(a["calls_pose_ready"], dtype=bool)
    elif record_path is not None:
        by_call = pose_ready_from_record(record_path)
        if by_call is not None:
            pos = call_positions(a)
            if set(by_call) != set(pos):
                tl.warnings.append(f"record has calls {sorted(set(by_call) ^ set(pos))} the timeline has not (or vice "
                                   "versa): pose_ready ignored")
            else:
                ready = np.zeros(len(pos), dtype=bool)
                for c, i in pos.items():
                    ready[i] = by_call[c]
                tl.pose_ready = ready
    if tl.pose_ready is None:
        tl.warnings.append("no per-call pose_ready: warm-up drawn up to the first ready call")
    return tl


def check_against_bundle(tl: Timeline, b: bd.Bundle) -> List[str]:
    """Key moments of the timeline vs the bundle (label -> call index and step must agree)."""
    out = []
    tk = {str(lab): (int(ci), int(st)) for lab, ci, st in zip(tl.a["key_labels"], tl.a["key_call_index"],
                                                             tl.a["key_step"])}
    for ks in b.keys:
        got = tk.get(ks.label)
        if got is None:
            out.append(f"{ks.label}: not in the timeline")
        elif got != (int(ks.call_index), int(ks.step)):
            out.append(f"{ks.label}: timeline (call, step) {got} != bundle ({ks.call_index}, {ks.step})")
    if int(tl.a["episode_steps"]) != int((b.outcome or {}).get("steps", tl.steps)):
        out.append(f"episode_steps {tl.steps} != outcome steps {b.outcome['steps']}")
    return out


# --------------------------------------------------------------------------- #
# SYNTHETIC timeline (layout work and tests only; stamped on every page)
# --------------------------------------------------------------------------- #
def ring_of(maps: np.ndarray, elev: float = 45.0) -> np.ndarray:
    """[360] max over elevation of four 64x64 label maps stitched to the equiangular ring (+-elev)."""
    ring, _ = geo.stitch_ring(maps, width=N_BEARING, height=int(round(N_BEARING * 2 * elev / 360.0)),
                              elev_top=elev, elev_bottom=-elev, fill=0.0)
    return np.clip(np.nan_to_num(ring).max(0), 0.0, 1.0)


def _roll(ring: np.ndarray, shift_deg: float) -> np.ndarray:
    """A ring seen after the robot turned: bearing b moves to b + shift (bin i -> i - shift)."""
    return np.roll(ring, -int(round(shift_deg)), axis=-1)


def synthetic_timeline(b: bd.Bundle, calls: Sequence[dict]) -> Tuple[dict, Dict[str, np.ndarray]]:
    """A PLAUSIBLE timeline from a bundle's key moments and the episode's call list (layout only, not a result).

    Every ready call takes the maps of the nearest key moment, turned by the net
    executed turn between the two calls.  ``calls``: records/<ep>.json "calls"
    (call_index, step, kind, ready, ppa_applied, executed_actions).
    """
    calls = sorted(calls, key=lambda c: int(c["call_index"]))
    steps_total = int((b.outcome or {}).get("steps") or (max(int(c["step"]) for c in calls) + 1))
    ready = [c for c in calls if c.get("kind") == "trajectory" and c.get("ppa_applied")]
    turn_at = np.zeros(steps_total + 1)
    step_action = np.full(steps_total, -1, np.int8)
    for c in calls:
        for j, act in enumerate(c.get("executed_actions") or []):
            s = int(c["step"]) + j
            if s < steps_total:
                step_action[s] = int(act)
                turn_at[s + 1] = 15.0 if act == bd.LEFT else -15.0 if act == bd.RIGHT else 0.0
    cum = np.cumsum(turn_at)  # net turn before step s (synthetic only: shifts the stand-in rings)
    keys = list(b.keys)
    R = len(ready)
    a = {k: None for k in ARRAYS}
    hist_ring = np.zeros((R, N_BEARING), np.float32)
    fut_ring = np.zeros((R, NUM_BINS, N_BEARING), np.float32)
    pb = np.full((R, NUM_SLOTS), np.nan, np.float32)
    pc = np.full((R, NUM_SLOTS), np.nan, np.float32)
    gb = np.full((R, NUM_SLOTS), np.nan, np.float32)
    gv = np.zeros((R, NUM_SLOTS), bool)
    sb = np.full((R, PATH_POINTS), np.nan, np.float32)
    sd = np.full((R, PATH_POINTS), np.nan, np.float32)
    cache = {}
    for r, c in enumerate(ready):
        s = int(c["step"])
        if not keys:
            continue
        ks = min(keys, key=lambda k: abs(int(k.step) - s))
        if ks.label not in cache:
            comp = bd.history_pred_composite(ks.hist_pred, ks.hist_none, ks.hist_mask)
            fut = bd.future_bin_maps(ks.fut_pred)
            gt, pred = p2.history_strip_marks(ks)
            conf = 1.0 - np.asarray(ks.hist_none, dtype=np.float64)
            pbs = np.full(NUM_SLOTS, np.nan)
            for k in np.nonzero(ks.hist_mask)[0]:
                v, rr, cc = np.unravel_index(int(np.argmax(ks.hist_pred[k])), ks.hist_pred[k].shape)
                pbs[k] = pn.pixel_to_ring(int(v), float(rr), float(cc))[0]
            gbs = np.full(NUM_SLOTS, np.nan)
            for k, bb, _ in gt:
                gbs[k] = bb
            pcs = np.where(ks.hist_mask, conf, np.nan)
            bearing, _, idx = bd.path_directions(ks.path_cam)
            p = np.asarray(ks.path_cam, dtype=np.float64)
            dist = np.hypot(p[:, 0], p[:, 2])
            path_b = np.full(PATH_POINTS, np.nan)
            path_b[idx] = bearing
            cache[ks.label] = (ring_of(comp), np.stack([ring_of(fut[i]) for i in range(NUM_BINS)]), pbs, pcs, gbs,
                               path_b, dist)
        ring, fr, pbs, pcs, gbs, path_b, dist = cache[ks.label]
        shift = float(cum[int(ks.step)] - cum[s]) if s <= steps_total else 0.0
        hist_ring[r] = _roll(ring, shift)
        fut_ring[r] = fr
        pb[r] = geo.wrap_deg(pbs + shift)
        pc[r] = pcs
        gb[r] = geo.wrap_deg(gbs + shift)
        gv[r] = np.isfinite(gbs)
        sb[r] = path_b
        sd[r] = dist
    call_steps = [int(c["step"]) for c in calls]
    nxt = []
    for c in ready:
        later = [st for st in call_steps if st > int(c["step"])]
        nxt.append(min(later) if later else steps_total)
    key_rows = [(k.label, int(k.call_index), int(k.step)) for k in keys]
    a.update({
        "call_index": np.asarray([int(c["call_index"]) for c in ready], np.int64),
        "step": np.asarray([int(c["step"]) for c in ready], np.int64),
        "next_step": np.asarray(nxt, np.int64),
        "bearing_deg": bin_bearings().astype(np.float32),
        "hist_ring": hist_ring, "hist_pred_peak_bearing": pb, "hist_pred_conf": pc,
        "hist_gt_bearing": gb, "hist_gt_visible": gv, "fut_ring": fut_ring,
        "s1_path_bearing": sb, "s1_path_dist": sd,
        "calls_step": np.asarray(call_steps, np.int64),
        "calls_kind": np.asarray([str(c.get("kind")) for c in calls]),
        "calls_ready": np.asarray([bool(c.get("kind") == "trajectory" and c.get("ppa_applied")) for c in calls]),
        "calls_pose_ready": np.asarray([bool(c.get("pose_ready", False)) for c in calls]),
        "step_action": step_action,
        "first_ready_step": np.asarray(int(ready[0]["step"]) if ready else -1, np.int64),
        "episode_steps": np.asarray(steps_total, np.int64),
        "key_labels": np.asarray([k[0] for k in key_rows], dtype="<U4"),
        "key_call_index": np.asarray([k[1] for k in key_rows], np.int64),
        "key_step": np.asarray([k[2] for k in key_rows], np.int64),
    })
    meta = {"schema": SCHEMA, "ep_key": b.ep_key, "category": b.category, "rank": b.category_rank,
            "is_main": b.is_main, "synthetic": True,
            "note": "SYNTHETIC: key-moment maps copied to the other ready calls and shifted by the executed turns"}
    return meta, a


def write_timeline(stem, meta: dict, a: Dict[str, np.ndarray]) -> Path:
    json_path, npz_path = timeline_paths(stem)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **a)
    json_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
UPSAMPLE = 4  # raster columns per step (keeps column edges crisp in PDF viewers)
TL_MARK = 0.85  # timeline markers are a little smaller than on the strips
SPREAD_MIN_SCALE = 0.6  # smallest marker scale of spread marks
GAP_PT = 0.5  # spread marks: neighbouring marked slots' circles keep at least this much air
MARK_SPACING_PT = 5.0  # neighbouring stacked marker columns at least this far apart (else every n-th call only)
KEY_LW = 0.45  # key-moment hairline
WRAP_PAD = 12  # deg the bearing axes run past their ends (field wrapped), so a mark at an end is whole inside
WARM_BLOCK_PT = {"page": 34.0, "overview": 30.0, "anim": 40.0}  # width of a compressed warm-up / no-map block
NOMAP_COVER_MAX = 0.30  # no-map spans are compressed only when the ready calls cover less than this of the rerun
NOMAP_MIN_STEPS = 20  # ... and only spans at least this long
SLOT_LABEL_INSET_PT = 0.9  # "1" / "8" sit this far inside the first ready column's left / right end
SLOT_LABEL_AIR_PT = 2.0  # ... and are drawn only when this much air is left between them
ALL_SLOTS = tuple(range(NUM_SLOTS))
OVERVIEW_SLOTS = (0, 3, 7)  # past frames 1, 4, 8: the overview, and every column too narrow for all 8
HIST_TICKS = (0.0, 90.0, 180.0, 270.0, 360.0)  # history panel y, bottom to top: ahead, right, behind, left, ahead
FUT_TICKS = (90.0, 0.0, -90.0)  # future panel: left, ahead, right (+-180 at the edges)
BADGE_LEVEL_PT = 9.4  # a raised K badge sits this far above the first row
BADGE_Y0_FRAC = 0.62  # centre of a first-row K badge: this fraction of the first row's height above the panel
BADGE_HALF_H_PT = 4.3


def compress_warmup(tl: Timeline, width_pt: float, block_pt: float) -> Timeline:
    """Compress the warm-up (steps 0 .. W, no affordance map) into a block ``block_pt`` wide at the left end.

    Only when no ready call falls inside the warm-up and the block is clearly
    narrower than the warm-up drawn to scale; otherwise the axis stays linear.
    Everything from W on keeps x = step; the axis gets a break mark at W.
    """
    tl.x0, tl.wblock, tl.blocks = 0.0, 0.0, []
    if not _warmup_compressible(tl, width_pt, block_pt):
        return tl
    w, n = int(tl.warmup_end()), int(tl.steps)
    b = block_pt * (n - w) / (width_pt - block_pt)
    tl.x0, tl.wblock = float(w - b), float(b)
    return tl


def _warmup_compressible(tl: Timeline, width_pt: float, block_pt: float) -> bool:
    w, n = int(tl.warmup_end()), int(tl.steps)
    steps = np.asarray(tl.a["step"])
    if w <= 0 or n <= w or width_pt <= 2 * block_pt or (len(steps) and int(steps.min()) < w):
        return False
    return w * width_pt / n > 1.25 * block_pt


def ready_cover(tl: Timeline) -> float:
    """Fraction of the rerun's steps inside a ready call's column."""
    if not tl.R or tl.steps <= 0:
        return 0.0
    return float(np.sum(np.asarray(tl.a["next_step"]) - np.asarray(tl.a["step"]))) / float(tl.steps)


def compress_axis(tl: Timeline, width_pt: float, block_pt: float, max_cover: float = NOMAP_COVER_MAX,
                  min_steps: int = NOMAP_MIN_STEPS) -> Timeline:
    """The warm-up (``compress_warmup``'s rule) and, when the ready calls cover less than ``max_cover`` of the
    rerun, every no-map span of at least ``min_steps`` steps, each compressed into a block ``block_pt`` wide.

    One rule for every timeline; the rest of the axis keeps one scale.  A no-map
    block is compressed only when it is clearly narrower than the span drawn to
    that scale.  Block widths are whole raster columns (1 / UPSAMPLE step), so
    the ready columns stay on the raster's column edges.
    """
    compress_warmup(tl, width_pt, block_pt)
    if not tl.R or ready_cover(tl) >= max_cover:
        return tl
    warm = tl.compressed
    w, n = float(tl.warmup_end()), float(tl.steps)
    cand = [(s0, s1) for kind, s0, s1 in tl.spans() if kind == "nomap" and s1 - s0 >= min_steps]
    if not cand:
        return tl
    for _ in range(3):  # drop spans that would not shrink much, then size the blocks for the scale left
        linear = n - (w if warm else 0.0) - sum(s1 - s0 for s0, s1 in cand)
        nb = len(cand) + (1 if warm else 0)
        room = width_pt - nb * block_pt
        if linear <= 0 or room <= 2 * block_pt:
            return tl
        per = room / linear
        keep = [(s0, s1) for s0, s1 in cand if (s1 - s0) * per > 1.25 * block_pt]
        if keep == cand:
            break
        cand = keep
        if not cand:
            return tl
    width = max(round(block_pt / per * UPSAMPLE), 1) / UPSAMPLE
    tl.blocks = [(float(s0), float(s1), float(width)) for s0, s1 in cand]
    if warm:
        b = block_pt / per
        tl.x0, tl.wblock = float(w - b), float(b)
    return tl


def raster_xlim(tl: Timeline) -> Tuple[float, float]:
    """x range of the column raster: from the whole step at or left of the axis start to the axis end."""
    lo, hi = tl.xlim()
    return float(math.floor(lo)), float(hi)


def column_raster(tl: Timeline, rgba_rows: np.ndarray, pad: int = 0, unwrap: bool = False) -> np.ndarray:
    """[360 + 2 pad, columns, 4]: each ready call's RGBA ring painted over its column (x of its step .. x of the
    next call, or of ``column_end`` when the timeline has it: the animation's columns grow with the playhead),
    transparent elsewhere; drawn with ``extent=(*raster_xlim(tl), ...)``.

    Columns are ``UPSAMPLE`` per x unit from ``raster_xlim(tl)[0]``; ready columns never fall inside a
    compressed span, so their edges are whole raster columns.  Rows: ``hist_rows`` (``unwrap``, the history
    panel's axis: ahead at both ends, behind in the middle) or ``wrap_rows`` (+-180), each with ``pad`` wrapped
    rows past both ends.
    """
    x_lo, x_hi = raster_xlim(tl)
    ncol = max(int(math.ceil((x_hi - x_lo) * UPSAMPLE - 1e-6)), 1)
    img = np.zeros((rgba_rows.shape[1], ncol, 4))
    xs = tl.x_of(np.asarray(tl.a["step"], dtype=np.float64))
    xn = tl.x_of(np.asarray(tl.a.get("column_end", tl.a["next_step"]), dtype=np.float64))
    for r in range(tl.R):
        c0 = int(round((float(xs[r]) - x_lo) * UPSAMPLE))
        c1 = int(round((float(xn[r]) - x_lo) * UPSAMPLE))
        img[:, max(c0, 0):max(c1, 0)] = rgba_rows[r][:, None, :]
    return hist_rows(img, pad) if unwrap else wrap_rows(img, pad)


def wrap_rows(img: np.ndarray, pad: int) -> np.ndarray:
    """Ring rows (+179.5 at row 0 ... -179.5) extended by ``pad`` wrapped rows at both ends."""
    return np.concatenate([img[-pad:], img, img[:pad]], axis=0) if pad > 0 else img


def hist_rows(img: np.ndarray, pad: int) -> np.ndarray:
    """Ring rows (+179.5 at row 0 ... -179.5) reordered for the history panel's axis (``hist_y``): top row y =
    359.5 = bearing +0.5 (ahead, a little left), then left (+90 at y 270), behind (y 180, the middle), right (-90 at
    y 90) down to y = 0.5 = bearing -0.5 at the bottom; ``pad`` wrapped rows past both ends."""
    n = img.shape[0]
    return wrap_rows(np.roll(img[::-1], -(n // 2), axis=0), pad)


def bearing_ylim(pad: float = WRAP_PAD) -> Tuple[float, float]:
    """The future panel's bearing axis: -180 - pad .. +180 + pad (ahead in the middle)."""
    return (-180.0 - pad, 180.0 + pad)


def hist_ylim(pad: float = WRAP_PAD) -> Tuple[float, float]:
    """The history panel's unwrapped bearing axis: -pad .. 360 + pad (behind in the middle)."""
    return (-pad, 360.0 + pad)


def hist_y(bearing: float) -> float:
    """y of a bearing (left-positive) on the history panel: y = -bearing mod 360, so up = left as on every panel.
    Top to bottom: ahead (y 360, bearing +0), left (270), behind (180, the middle), right (90), ahead (y 0,
    bearing -0).  One y per bearing: a mark right ahead is drawn once, at the end its sign puts it."""
    return float((-float(bearing)) % 360.0)


def step_ticks(n: int, lo: int = 0) -> np.ndarray:
    """Round step ticks over [lo, n] (at most about 7)."""
    for every in (10, 20, 25, 50, 100, 200):
        if (n - lo) / every <= 7:
            break
    else:
        every = 500
    return np.arange(int(math.ceil(lo / every)) * every, n + 1, every)


def per_step_pt(tl: Timeline, ax) -> float:
    """Points per step of x (outside a compressed span); ``ax``: the axes, or the panel's width in points."""
    width = float(ax) if isinstance(ax, (int, float)) else ax.get_position().width * ax.figure.get_figwidth() * 72.0
    return width / tl.span_steps()


def per_step_pt_axes(ax) -> float:
    """Points per x unit of any axes (from its x limits)."""
    lo, hi = ax.get_xlim()
    return ax.get_position().width * ax.figure.get_figwidth() * 72.0 / max(abs(hi - lo), 1e-9)


def column_pt(tl: Timeline, ax) -> float:
    """Median width of a ready call's column, in points."""
    if not tl.R:
        return float("inf")
    return float(np.median(tl.a["next_step"] - tl.a["step"])) * per_step_pt(tl, ax)


def columns_pt(tl: Timeline, ax) -> np.ndarray:
    """[R] width of each ready call's column (its step .. the next call) in points."""
    xs = np.asarray(tl.x_of(np.asarray(tl.a["step"], dtype=np.float64)), dtype=np.float64)
    xn = np.asarray(tl.x_of(np.asarray(tl.a["next_step"], dtype=np.float64)), dtype=np.float64)
    return (xn - xs) * per_step_pt(tl, ax)


def slot_gap(slots: Optional[Sequence[int]] = None) -> int:
    """Smallest distance (in slots) between two marked slots (1 when every slot is marked)."""
    s = sorted(set(range(NUM_SLOTS) if slots is None else slots))
    return int(min(np.diff(s))) if len(s) > 1 else NUM_SLOTS


def ring_outer_pt(scale: float) -> float:
    """Outer diameter (pt) of a true-direction circle drawn at ``scale`` (``p2.gt_ring``)."""
    return p2.GT_MS * scale + p2.GT_MEW * max(min(scale, 1.0), 0.75)


def fits(col_pt: float, slots: Sequence[int], scale: float = SPREAD_MIN_SCALE) -> bool:
    """The marked ``slots`` spread over a column ``col_pt`` wide keep GAP_PT of air between neighbouring circles."""
    return col_pt * slot_gap(slots) / NUM_SLOTS >= ring_outer_pt(scale) + GAP_PT - 1e-9


def spread_scale(pitch_pt: float) -> float:
    """The largest marker scale (SPREAD_MIN_SCALE .. TL_MARK) whose circles keep GAP_PT at a pitch of ``pitch_pt``."""
    s = TL_MARK
    while s > SPREAD_MIN_SCALE + 1e-9 and ring_outer_pt(s) + GAP_PT > pitch_pt:
        s -= 0.005
    return float(round(max(s, SPREAD_MIN_SCALE), 3))


@dataclass
class MarkPlan:
    """How a timeline's history marks are laid out (``mark_plan``).

    ``slots[r]``: the slots ready row r marks, spread over its column (None = its 8 marks stacked at the column's
    centre); ``rows``: the rows marked at all; ``mode``: "all" (8 per column), "some" (8, and 1, 4, 8 in the
    narrow columns), "148" (1, 4, 8) or "stacked" (every ``stride``-th call and every key moment).
    """

    mode: str
    scale: float
    stride: int
    slots: Dict[int, Optional[Tuple[int, ...]]]
    rows: List[int]

    @property
    def spread(self) -> bool:
        return self.mode != "stacked"


def mark_plan(tl: Timeline, ax, slots: Optional[Sequence[int]] = None) -> MarkPlan:
    """The marks of timeline ``tl`` drawn on axes ``ax`` (``slots``: the slots a column may mark; default all 8,
    the overview ``OVERVIEW_SLOTS``).

    A column marks every allowed slot when neighbouring circles keep ``GAP_PT`` of air at the smallest scale
    (``fits``), else past frames 1, 4 and 8.  When even that does not fit the median column (long reruns), every
    call's marks are stacked at its column's centre, for every ``marker_stride``-th call and every key moment.
    Spread marks get the largest scale (up to TL_MARK) that keeps the gap in the median column.
    """
    base = tuple(ALL_SLOTS if slots is None else sorted(set(slots)))
    if not tl.R:
        return MarkPlan("all" if base == ALL_SLOTS else "148", TL_MARK, 1, {}, [])
    widths = columns_pt(tl, ax)
    med = float(np.median(widths))
    main = base if fits(med, base) else (OVERVIEW_SLOTS if fits(med, OVERVIEW_SLOTS) else None)
    if main is None:
        stride = marker_stride(tl, ax)
        return MarkPlan("stacked", float(np.clip(med / 6.0, 0.62, 1.0)) * TL_MARK, stride,
                        {r: None for r in range(tl.R)}, marked_rows(tl, stride))
    per_row: Dict[int, Optional[Tuple[int, ...]]] = {}
    for r, w in enumerate(widths):
        per_row[r] = main if fits(w, main) else (OVERVIEW_SLOTS if fits(w, OVERVIEW_SLOTS) else None)
    if main != ALL_SLOTS:
        mode = "148"
    else:
        mode = "all" if all(v == ALL_SLOTS for v in per_row.values()) else "some"
    return MarkPlan(mode, spread_scale(med * slot_gap(main) / NUM_SLOTS), 1, per_row, list(range(tl.R)))


def narrow_columns(plan: MarkPlan) -> List[int]:
    """Ready rows whose column marks fewer slots than the plan's main set (frames 1, 4, 8 or stacked)."""
    if not plan.spread:
        return []
    main = ALL_SLOTS if plan.mode in ("all", "some") else OVERVIEW_SLOTS
    return sorted(r for r, s in plan.slots.items() if s != main)


def slot_xs(tl: Timeline, r: int, spread_marks: bool) -> np.ndarray:
    """[8] x of each slot's marks of ready call r: slot k at (k + 0.5) / 8 of the column (1 = oldest at the left),
    or all at the column's centre when the marks are stacked.  The column is the call's own action chunk (its step
    .. the next call), so the positions are known when the call is made."""
    s0, s1 = float(tl.x_of(float(tl.a["step"][r]))), float(tl.x_of(float(tl.a["next_step"][r])))
    if spread_marks:
        return s0 + (np.arange(NUM_SLOTS) + 0.5) / NUM_SLOTS * (s1 - s0)
    return np.full(NUM_SLOTS, (s0 + s1) / 2.0)


def marker_stride(tl: Timeline, ax) -> int:
    """Markers on every n-th ready call so that marker columns are >= MARK_SPACING_PT apart (1 = every call)."""
    col = column_pt(tl, ax)
    return 1 if not np.isfinite(col) or col <= 0 else max(1, int(math.ceil(MARK_SPACING_PT / col - 1e-9)))


def marked_rows(tl: Timeline, stride: int) -> List[int]:
    """Ready calls that get markers: every ``stride``-th one, counted from the first, plus every key moment."""
    keys = set(tl.key_rows().values())
    return [r for r in range(tl.R) if r % stride == 0 or r in keys]


def history_marks(tl: Timeline, rows: Sequence[int], spread_marks: bool = True, xs_of=None,
                  slots: Optional[Sequence[int]] = None, plan: Optional[MarkPlan] = None) -> Tuple[list, list]:
    """((x, y) of the true-direction circles, (x, y) of the predicted-peak dots) of the given ready calls, on the
    history panel's axis (``hist_y``: up = left, behind in the middle), one mark per slot (no wrapped copy).

    Slot k's circle and dot share x (``slot_xs``).  ``plan`` (``mark_plan``) gives each row's slots (None =
    all 8 stacked at the centre); without it, ``slots`` (default all) and ``spread_marks`` apply to every row.
    ``xs_of(r)`` overrides the x positions.
    """
    gt, pred = [], []
    for r in rows:
        if plan is not None:
            row_slots = plan.slots.get(r)
            sp = row_slots is not None
            marked = row_slots if sp else ALL_SLOTS
        else:
            sp = spread_marks
            marked = ALL_SLOTS if slots is None else slots
        keep = np.zeros(NUM_SLOTS, dtype=bool)
        keep[list(marked)] = True
        xs = slot_xs(tl, r, sp) if xs_of is None else xs_of(r)
        gb = np.asarray(tl.a["hist_gt_bearing"][r], dtype=np.float64)
        ok = np.isfinite(gb) & np.asarray(tl.a["hist_gt_visible"][r], dtype=bool) & keep
        for k in np.nonzero(ok)[0]:
            gt.append((float(xs[k]), hist_y(float(gb[k]))))
        pb = np.asarray(tl.a["hist_pred_peak_bearing"][r], dtype=np.float64)
        pc = np.asarray(tl.a["hist_pred_conf"][r], dtype=np.float64)
        for k in np.nonzero(np.isfinite(pb) & np.isfinite(pc) & (pc >= 0.5) & keep)[0]:
            pred.append((float(xs[k]), hist_y(float(pb[k]))))
    return gt, pred


def _setup_axes(ax, tl: Timeline, ylim: Tuple[float, float]) -> None:
    ax.set_xlim(*tl.xlim())
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)
    ax.set_facecolor(style.SURFACE)
    for s in ax.spines.values():
        s.set_edgecolor(style.AXIS)
        s.set_linewidth(p2.HAIR)
    ax.tick_params(axis="both", which="both", length=1.8, width=p2.HAIR, color=style.AXIS, pad=1.5,
                   labelsize=p2.MIN_FS, labelcolor=style.INK_2)


def _span_patches(ax, tl: Timeline, ylim: Tuple[float, float]) -> None:
    with matplotlib.rc_context({"hatch.linewidth": p2.HATCH_LW}):
        for kind, s0, s1 in tl.spans():
            x0, x1 = tl.x_of(s0), tl.x_of(s1)
            if kind == "warmup":
                ax.add_patch(Rectangle((x0, ylim[0]), x1 - x0, ylim[1] - ylim[0], fc=style.SURFACE,
                                       ec=p2.HATCH_COLOR, lw=0.0, hatch=p2.HATCH, zorder=0.5))
            else:
                ax.add_patch(Rectangle((x0, ylim[0]), x1 - x0, ylim[1] - ylim[0], fc=p2.NOMAP_FILL, ec="none",
                                       lw=0.0, zorder=0.5))


def _hairlines(ax, tl: Timeline, ys: Sequence[float]) -> None:
    for y in ys:
        ax.plot(list(tl.xlim()), [y, y], color=style.GRID, lw=p2.HAIR, zorder=0.8, solid_capstyle="butt")


# "//" across an axis line (custom marker: drawn in points, so it keeps its size at every dpi)
_BREAK = MPath([(-0.75, -1.0), (-0.1, 1.0), (0.1, -1.0), (0.75, 1.0)],
               [MPath.MOVETO, MPath.LINETO, MPath.MOVETO, MPath.LINETO])
_BREAK_GAP = MPath([(-0.42, -1.0), (0.42, -1.0), (0.42, 1.0), (-0.42, 1.0), (-0.42, -1.0)],
                   [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY])


def break_mark(ax, tl: Timeline, y_axes: float = 0.0, size: float = 4.2) -> None:
    """Axis-break marks where the step scale changes (``tl.breaks()``: the end of a compressed warm-up and both
    ends of each compressed no-map block), on the axis line at ``y_axes`` (axes fraction)."""
    xs = [float(tl.x_of(s)) for s in tl.breaks()]
    if not xs:
        return
    tr = ax.get_xaxis_transform()
    ax.plot(xs, [y_axes] * len(xs), ls="none", marker=_BREAK_GAP, ms=size, mfc=style.SURFACE, mec="none",
            transform=tr, clip_on=False, zorder=9)
    ax.plot(xs, [y_axes] * len(xs), ls="none", marker=_BREAK, ms=size, mfc="none", mec=style.INK_2, mew=0.45,
            transform=tr, clip_on=False, zorder=9.5)


def key_steps(tl: Timeline) -> Dict[str, float]:
    """Key label -> x of its call's step (where its hairline and badge go)."""
    return {lab: float(tl.x_of(float(tl.a["step"][r]))) for lab, r in tl.key_rows().items()}


def _key_lines(ax, tl: Timeline, ylim: Tuple[float, float]) -> None:
    """K1-K4: an ink hairline at the step of each key moment's call (its column starts there)."""
    for lab, s in key_steps(tl).items():
        ax.plot([s, s], list(ylim), color=style.INK, lw=KEY_LW, zorder=6, solid_capstyle="butt", clip_on=False)


def warmup_label_lines(fig, tl: Timeline, L_warm: str, L_range: Optional[str], width_pt: float,
                       fs: float = p2.MIN_FS) -> List[str]:
    """Lines of the warm-up label that fit ``width_pt``: the name, then its step range when it fits."""
    lines: List[str] = []
    if cd.text_width_pt(fig, L_warm, fs) + 3.0 <= width_pt:
        lines.append(L_warm)
        end = int(tl.warmup_end()) - 1
        if L_range and end >= 0:
            for fmt in (L_range, "{a}–{b}"):
                text = fmt.format(a=0, b=end)
                if cd.text_width_pt(fig, text, fs) + 3.0 <= width_pt:
                    lines.append(text)
                    break
    return lines


def draw_warmup_label(ax, tl: Timeline, label: str, range_label: Optional[str] = None,
                      y: Optional[float] = None) -> None:
    """The warm-up's name (and step range) inside its hatched span, when it fits (centred on ``y``, default the
    middle of the panel)."""
    y = float(np.mean(ax.get_ylim())) if y is None else y
    for kind, s0, s1 in tl.spans():
        if kind != "warmup":
            continue
        x0, x1 = tl.x_of(s0), tl.x_of(s1)
        w_pt = (x1 - x0) * per_step_pt(tl, ax)
        lines = warmup_label_lines(ax.figure, tl, label, range_label if tl.compressed else None, w_pt - 2.0)
        if lines:
            ax.text((x0 + x1) / 2, y, "\n".join(lines), ha="center", va="center", fontsize=p2.MIN_FS,
                    color=style.INK_2, zorder=3, linespacing=1.25, multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.3", fc=style.SURFACE, ec="none"))


TL_SMOOTH_DEG = 3.5  # display blur of the timeline's rings (sigma, deg of bearing): about 0.5 pt on the page


def display_hist_rings(tl: Timeline, sigma_deg: float = TL_SMOOTH_DEG) -> np.ndarray:
    """[R, 360] the history rings as drawn: display-smoothed along the bearing, each keeping its maximum."""
    return p2.smooth_rings(np.asarray(tl.a["hist_ring"], dtype=np.float64), sigma_deg)


def display_fut_rings(tl: Timeline, sigma_deg: float = TL_SMOOTH_DEG) -> np.ndarray:
    """[R, 4, 360] the future rings (per time bin) as drawn: display-smoothed along the bearing, each keeping its
    maximum."""
    return p2.smooth_rings(np.asarray(tl.a["fut_ring"], dtype=np.float64), sigma_deg)


def hist_field(ax, tl: Timeline, rgba_rows: np.ndarray) -> None:
    """The history rings of the ready calls on the history panel's axis (faint orange columns)."""
    ylim = hist_ylim()
    ax.imshow(column_raster(tl, rgba_rows, WRAP_PAD, unwrap=True), extent=(*raster_xlim(tl), *ylim),
              origin="upper", aspect="auto", interpolation="nearest", zorder=1)


def fut_field(ax, tl: Timeline, rgba_rows: np.ndarray) -> None:
    """The future rings of the ready calls on the +-180 axis (faint teal columns, later bins on top)."""
    ylim = bearing_ylim()
    ax.imshow(column_raster(tl, rgba_rows, WRAP_PAD), extent=(*raster_xlim(tl), *ylim), origin="upper",
              aspect="auto", interpolation="nearest", zorder=1)


def hist_ticks(ax, labels: Sequence[str]) -> None:
    """History panel ticks at y 0, 90, 180, 270, 360 = ahead, right, behind, left, ahead (bottom to top; ``labels``
    in that order), so up = left as on the future panel and the turn track."""
    ax.set_yticks(list(HIST_TICKS))
    ax.set_yticklabels(list(labels))


def fut_ticks(ax, labels: Sequence[str]) -> None:
    """Future panel ticks: left +90, ahead 0, right -90 (``labels`` in that order).  The +90 label sits just
    above its line and the -90 label just below, so the three never touch on a thin panel."""
    ax.set_yticks(list(FUT_TICKS))
    ax.set_yticklabels(list(labels))
    tl_ = ax.get_yticklabels()
    if len(tl_) == 3:
        tl_[0].set_va("bottom")
        tl_[2].set_va("top")


def draw_marks(ax, gt: Sequence[Tuple[float, float]], pred: Sequence[Tuple[float, float]], scale: float) -> None:
    """True-direction circles and predicted-peak dots on the history panel, unclipped (a mark right ahead, at the
    panel's top or bottom end, stays whole inside the WRAP_PAD margin)."""
    if gt:
        p2.gt_ring(ax, [g[0] for g in gt], [g[1] for g in gt], scale=scale, clip_on=False)
    if pred:
        p2.pred_dot(ax, [q[0] for q in pred], [q[1] for q in pred], scale=scale, clip_on=False)


def draw_history_panel(ax, tl: Timeline, labels: Sequence[str], warmup_label: Optional[str] = None,
                       warmup_range: Optional[str] = None, slots: Optional[Sequence[int]] = None) -> dict:
    """The history part of the timeline (``labels``: ahead, right, behind, left, ahead from the bottom);
    ``slots``: the slots a column may mark (default all).  Nothing is written over the data: a stacked plan's
    stride goes into the legend's timeline entry (``fig_v2.slot_legend_text``).  Returns {"scale", "n_pred",
    "n_gt", "marker_stride", "spread", "mode", ...}."""
    ylim = hist_ylim()
    _setup_axes(ax, tl, ylim)
    _span_patches(ax, tl, ylim)
    _hairlines(ax, tl, HIST_TICKS)
    if tl.R:
        hist_field(ax, tl, p2.heat_rgba(display_hist_rings(tl), p2.HIST_COLOR, p2.TL_HEAT_ALPHA))
    plan = mark_plan(tl, ax, slots)
    gt, pred = history_marks(tl, plan.rows, plan=plan)
    draw_marks(ax, gt, pred, plan.scale)
    _key_lines(ax, tl, ylim)  # (the axis break is marked on the future panel and the step axis below)
    hist_ticks(ax, labels)
    ax.set_xticks([])
    if warmup_label:
        draw_warmup_label(ax, tl, warmup_label, warmup_range, y=180.0)
    return {"scale": plan.scale, "n_pred": len(pred), "n_gt": len(gt), "marker_stride": plan.stride,
            "spread": plan.spread, "mode": plan.mode,
            "narrow_columns": narrow_columns(plan),
            "compressed_warmup": tl.compressed, "nomap_blocks": len(tl.blocks),
            "slots": list(range(NUM_SLOTS) if slots is None else slots)}


def slot_end_labels(ax, tl: Timeline, plan: MarkPlan, fs: float = p2.MIN_FS) -> List[str]:
    """Tiny "1" and "8" inside the first ready column, at its left and right ends, near the panel's bottom edge
    (ahead) -- or its top edge when a mark of that column is there: where past frames 1 and 8 sit in every
    column.  Nothing when the first column's marks are stacked, when the two do not fit with
    ``SLOT_LABEL_AIR_PT`` between them (``slot_labels_fit``: measured widths, one rule for every language), or
    when both edges are taken.  Returns the texts."""
    if not tl.R or plan.slots.get(0) is None:
        return []
    x0, x1 = float(tl.x_of(float(tl.a["step"][0]))), float(tl.x_of(float(tl.a["next_step"][0])))
    if not slot_labels_fit(ax.figure, (x1 - x0) * per_step_pt(tl, ax), fs):
        return []
    gt, pred = history_marks(tl, [0], plan=plan)
    ys = [y for _, y in gt + pred]
    h_pt = ax.get_position().height * ax.figure.get_figheight() * 72.0
    lo, hi = ax.get_ylim()
    zone = (fs + 3.0) / h_pt * (hi - lo)  # the label's height (plus air) in bearing units
    for va, edge, dy in (("bottom", 0.0, 1.0), ("top", 1.0, -1.0)):
        y_edge = lo if edge == 0.0 else hi
        if any(abs(y - y_edge) < zone for y in ys):
            continue
        for x, text, ha, dx in ((x0, "1", "left", SLOT_LABEL_INSET_PT),
                                (x1, str(NUM_SLOTS), "right", -SLOT_LABEL_INSET_PT)):
            ax.annotate(text, (x, edge), xycoords=("data", "axes fraction"), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, fontsize=fs, color=style.INK_2,
                        annotation_clip=False, zorder=9)
        return ["1", str(NUM_SLOTS)]
    return []


def slot_labels_fit(fig, col_pt: float, fs: float = p2.MIN_FS) -> bool:
    """"1" and "8" fit a column ``col_pt`` wide: both measured at ``fs``, ``SLOT_LABEL_INSET_PT`` inside its ends,
    with at least ``SLOT_LABEL_AIR_PT`` of air between them."""
    need = (2 * SLOT_LABEL_INSET_PT + cd.text_width_pt(fig, "1", fs) + cd.text_width_pt(fig, str(NUM_SLOTS), fs)
            + SLOT_LABEL_AIR_PT)
    return col_pt >= need - 1e-9


def edge_tick_labels(ax) -> None:
    """Top / bottom y tick labels inside the panel's height, so stacked panels' labels never touch."""
    labels = ax.get_yticklabels()
    if labels:
        labels[0].set_va("top")
        labels[-1].set_va("bottom")


def draw_future_panel(ax, tl: Timeline, labels: Sequence[str], plan: Optional[MarkPlan] = None) -> dict:
    """The future sub-panel: time bins painted early -> late (all bearings, ahead in the middle) and the System1
    path endpoint bearing (one ink dot at the column's centre, on the rows the history ``plan`` marks).  Ticks:
    left / ahead / right (``labels``); the hairlines at +-180 are the edges' "behind"."""
    ylim = bearing_ylim()
    _setup_axes(ax, tl, ylim)
    _span_patches(ax, tl, ylim)
    _hairlines(ax, tl, (180.0, 90.0, 0.0, -90.0, -180.0))
    if tl.R:
        fut = display_fut_rings(tl)
        fut_field(ax, tl, np.stack([p2.bins_rgba(fut[r]) for r in range(tl.R)]))  # [R, 360, 4]
    plan = mark_plan(tl, ax) if plan is None else plan
    xs, ys = future_marks(tl, plan.rows)
    if xs:
        p2.path_marks(ax, xs, ys, ms=1.9 * max(plan.scale, 0.75), rim=True)
    _key_lines(ax, tl, ylim)
    break_mark(ax, tl)
    fut_ticks(ax, labels)
    ax.set_xticks([])
    return {"path_ends": len(xs)}


def future_marks(tl: Timeline, rows: Sequence[int], xc: Optional[np.ndarray] = None) -> Tuple[list, list]:
    """(xs, ys) of the System 1 path endpoint dots (column centres; +-180 copies within WRAP_PAD)."""
    ends = tl.path_end_bearing()
    xc = tl.x_center() if xc is None else xc
    xs, ys = [], []
    for r in rows:
        if np.isfinite(ends[r]):
            for y in p2.both_ends(float(ends[r]), margin=WRAP_PAD):
                xs.append(float(xc[r]))
                ys.append(y)
    return xs, ys


def axis_ticks(fig, tl: Timeline, per: float, fs: float = p2.MIN_FS) -> List[Tuple[float, str]]:
    """(x, label) of the step ticks: on a compressed axis the ends of every compressed span (0 at a compressed
    warm-up's left end, W at its break, both ends of a no-map block) and the axis' end (the rerun's last step)
    first, then round steps inside the linear parts that keep clear of them (3 pt).  ``per``: points per x unit."""
    must: List[Tuple[float, str]] = []
    if tl.compressed:
        must += [(float(tl.x0), "0"), (float(tl.x_of(tl.warmup_end())), str(int(tl.warmup_end())))]
    for a, b, _ in sorted(tl.blocks):
        must += [(float(tl.x_of(a)), str(int(a))), (float(tl.x_of(b)), str(int(b)))]
    if must:
        must.append((float(tl.xlim()[1]), str(int(tl.steps))))
    seen, uniq = set(), []
    for x, lab in must:
        if lab not in seen:
            seen.add(lab)
            uniq.append((x, lab))
    must = uniq
    if not must:
        return [(float(t), str(int(t))) for t in step_ticks(tl.steps)]
    # linear parts: between the compressed spans
    s_k, x_k = tl.knots()
    linear = [(float(s_k[i]), float(s_k[i + 1])) for i in range(len(s_k) - 1)
              if abs((x_k[i + 1] - x_k[i]) - (s_k[i + 1] - s_k[i])) < 1e-6 and s_k[i + 1] > s_k[i]]
    out = list(must)

    def clear(x, lab):
        for x2, lab2 in out:
            gap = abs(x - x2) * per - (cd.text_width_pt(fig, lab, fs) + cd.text_width_pt(fig, lab2, fs)) / 2
            if gap < 3.0:
                return False
        return True
    total = sum(b - a for a, b in linear) or float(tl.steps)
    every = 10
    for every in (10, 20, 25, 50, 100, 200, 500):  # round ticks every 10 / 20 / ... steps of the linear parts
        if total / every <= 7:
            break
    for a, b in linear:
        for t in np.arange(int(math.ceil(a / every)) * every, b + 1e-9, every):
            x = float(tl.x_of(float(t)))
            if clear(x, str(int(t))):
                out.append((x, str(int(t))))
    return sorted(out)


TICK_LEN, TICK_PAD = 1.8, 1.5  # step axis: tick length and label pad (pt)


def step_axis(ax, tl: Timeline, unit: str) -> List[Tuple[float, str]]:
    """Step ticks on the bottom axes of the timeline, and the unit left of the axis on the tick labels' line (so
    no tick is dropped for it).  A tick label at the axis' right end is right-aligned to stay on the page.

    With compressed spans: the steps at both ends of each compressed span and at the axis' end (``axis_ticks``),
    then round ticks that keep clear of them, and a break mark wherever the scale changes.  Returns the ticks."""
    fig = ax.figure
    per = per_step_pt(tl, ax)
    ticks = axis_ticks(fig, tl, per)
    x_lo, x_end = tl.xlim()
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks])
    ax.tick_params(axis="x", which="both", length=TICK_LEN, width=p2.HAIR, color=style.AXIS, pad=TICK_PAD,
                   labelsize=p2.MIN_FS, labelcolor=style.INK_2)
    for (x, lab), text in zip(ticks, ax.get_xticklabels()):
        if (x_end - x) * per < cd.text_width_pt(fig, lab, p2.MIN_FS) / 2 + 0.5:
            text.set_ha("right")
    first = [lab for x, lab in ticks if abs(x - x_lo) * per < 0.5]
    dx = (cd.text_width_pt(fig, first[0], p2.MIN_FS) / 2 if first else 0.0) + 3.0
    ax.annotate(unit, (0.0, 0.0), xycoords="axes fraction", xytext=(-dx, -(TICK_LEN + TICK_PAD)),
                textcoords="offset points", ha="right", va="top", fontsize=p2.MIN_FS, color=style.INK_2,
                annotation_clip=False)
    break_mark(ax, tl)
    return ticks


def draw_turn_track(ax, tl: Timeline, label: str) -> int:
    """Executed turns per step (from ``step_action``): up = turn left 15 deg, down = turn right 15 deg.

    Up = left, as on the bearing axes above.  A step inside a compressed span
    gets a proportionally narrower bar.  Returns the number of turns drawn (-1
    when the timeline has no per-step actions).
    """
    ax.set_xlim(*tl.xlim())
    ax.set_ylim(-1.0, 1.0)
    ax.set_autoscale_on(False)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(style.AXIS)
        ax.spines[side].set_linewidth(p2.HAIR)
    ax.set_xticks([])
    ax.set_yticks([0.0])
    ax.set_yticklabels([label])
    ax.tick_params(axis="y", which="both", length=0, pad=2.0, labelsize=p2.MIN_FS, labelcolor=style.INK_2)
    ax.plot(list(tl.xlim()), [0, 0], color=style.GRID, lw=p2.HAIR, zorder=1, solid_capstyle="butt")
    if "step_action" not in tl.a:
        return -1
    acts = np.asarray(tl.a["step_action"])
    n = 0
    width = 0.62
    for s_, act in enumerate(acts[: tl.steps]):
        if act in (bd.LEFT, bd.RIGHT):
            h = 0.92 if act == bd.LEFT else -0.92
            k = float(tl.x_of(s_ + 1.0)) - float(tl.x_of(float(s_)))
            xc = tl.x_of(s_ + 0.5)
            ax.add_patch(Rectangle((xc - width * k / 2, 0.0), width * k, h, fc=p2.TURN_COLOR, ec="none", lw=0.0,
                                   zorder=2))
            n += 1
    _key_lines(ax, tl, (-1.0, 1.0))
    return n


def badge_layout(fig, tl: Timeline, width_pt: float, fs: float = p2.MIN_FS) -> List[dict]:
    """Where the K badges go above the timeline, in points from the panel's left edge.

    Badges sit on their hairlines (``target``).  A badge that would touch the
    previous one (in time) is raised one level, its leader running straight down
    beside the lower badge; a lower badge that its neighbour's leader would cross
    moves aside, on a short slanted leader.  Returns dicts: label, target, x, level, w.
    """
    xs = key_steps(tl)
    if not xs:
        return []
    per = width_pt / tl.span_steps()
    items = sorted(({"label": lab, "target": (x - tl.x0) * per,
                     "w": cd.text_width_pt(fig, lab, fs, fontweight="bold") + 2 * 0.22 * fs + 1.0}
                    for lab, x in xs.items()), key=lambda d: d["target"])
    rows: Dict[int, List[dict]] = {0: [], 1: []}
    for it in items:
        it["x"] = it["target"]
        it["level"] = 0
        for level in (0, 1):
            prev = rows[level][-1] if rows[level] else None
            if prev is None or it["x"] - prev["x"] >= (it["w"] + prev["w"]) / 2 + 1.5:
                it["level"] = level
                break
        else:  # both rows taken: stay on the first row, pushed right of the previous badge
            prev = rows[0][-1]
            it["x"] = prev["x"] + (it["w"] + prev["w"]) / 2 + 1.5
        rows[it["level"]].append(it)
    for hi in rows[1]:  # a raised badge's leader must not cross a first-row badge
        for lo in rows[0]:
            if abs(lo["x"] - hi["target"]) < lo["w"] / 2 + 1.2:
                if lo["target"] <= hi["target"]:
                    lo["x"] = hi["target"] - lo["w"] / 2 - 1.2
                else:
                    lo["x"] = hi["target"] + lo["w"] / 2 + 1.2
    for it in items:
        it["x"] = float(np.clip(it["x"], it["w"] / 2, width_pt - it["w"] / 2))
    return items


def badge_levels(fig, tl: Timeline, width_pt: float, fs: float = p2.MIN_FS) -> int:
    """Rows of K badges the timeline needs (1 or 2)."""
    return 1 + max([it["level"] for it in badge_layout(fig, tl, width_pt, fs)] or [0])


def badges_height_in(levels: int, base: float = 0.12) -> float:
    return base + (levels - 1) * BADGE_LEVEL_PT / 72.0


def key_badges(badge_ax, data_ax, tl: Timeline, fs: float = p2.MIN_FS) -> List[dict]:
    """K badges above the history panel on their hairlines (the key calls' steps), raised one row when two would
    touch, each joined to its hairline.  ``badge_ax``: point-unit axes spanning the panel's width (x = 0 at the
    panel's left edge, y = 0 at the panel's top edge)."""
    width_pt = data_ax.get_position().width * data_ax.figure.get_figwidth() * 72.0
    items = badge_layout(badge_ax.figure, tl, width_pt, fs)
    levels = 1 + max([it["level"] for it in items] or [0])
    y0 = BADGE_Y0_FRAC * (badge_ax.get_ylim()[1] - (levels - 1) * BADGE_LEVEL_PT)
    for it in items:
        y = y0 + it["level"] * BADGE_LEVEL_PT
        cd.key_badge(badge_ax, it["x"], y, it["label"], fs=fs)
        badge_ax.plot([it["x"], it["target"]], [y - BADGE_HALF_H_PT, 0.0], color=style.INK, lw=KEY_LW, zorder=5,
                      solid_capstyle="butt", gid=f"leader:{it['label']}")
    return items
