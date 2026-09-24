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

    def query_rows(self, arm: str) -> List[int]:
        """Scored rows (cache endpoints) on which ``arm`` has a prediction."""
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


def group_label(group: Sequence[int]) -> str:
    """Slot numbers are 1-based in every figure: [0] -> "1", [0..5] -> "1–6"."""
    return str(group[0] + 1) if len(group) == 1 else f"{group[0] + 1}–{group[-1] + 1}"


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
