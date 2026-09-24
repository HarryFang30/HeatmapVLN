#!/usr/bin/env python3
"""EXP-18 gallery figure: the predicted affordance map across tiers, at pre-registered (not hand-picked) episodes.

Layout (``style.WIDTH_DOUBLE`` wide; one band + one row of tiles per tier, one column per percentile):

              10th percentile          50th percentile          90th percentile
  A  Training scenes  R2R · 22 scenes · 220 episodes · scenes used to train the model
       +----------------------+ +----------------------+ +----------------------+
       | (disc)  scene id     | | (disc)  ...          | | (disc)  ...          |
       |  episode 6768  median PCK@8                   | |                      |
       |  all 9 frames    1.3°  69/71                  | |                      |
       |  frame 35 of 77  1.4°   8/8   (the one shown) | |                      |
       |  always-behind guess 41°  1/8                 | |                      |
       | lane: numbered badges at the true bearings    | |                      |
       | RGB   F | R | B | L                           | |                      |
 truth | ground-truth affordance map (blue)            | |                      |
 pred. | predicted affordance map (orange) + x marks   | |                      |
       | ticks at the true bearings; miss badges  (7)  | |                      |
       +----------------------+ +----------------------+ +----------------------+
  B  Held-out scenes ...
            shared bearing axis under the last row; legend (two lines)

Selection (docs/experiments/README.md EXP-18 "案例挑选规则", implemented by
``select_cases.py`` into cases.json ``gallery``): per tier, the episodes whose
per-episode median bearing error is nearest the 10th / 50th / 90th percentile
of that tier.  The 90th is the high-error end (the pre-registered failure
case) and is always drawn.  Each tile shows ONE scored frame of its episode,
the most typical one (``typical_row``): the frame whose median bearing error
over its visible past positions is nearest the episode's (ties: the wider
bearing span of those positions, then the earlier frame).  The tile prints the
episode's median and joint PCK@8 over all scored frames next to the frame's,
so the frame can be checked against its episode.

Encodings are those of the main case figure (``fig_case.py``; primitives in
``common_draw.py``): blue = ground truth, orange = prediction, numbered history
badges (1 = oldest of the K = 8 queried), a heading-up local map disc whose rim
is the bearing ring the strip unrolls clockwise from the front view's left edge,
x = predicted peak (within ``fc.MERGE_DEG`` merged, touching marks staggered).
The disc is ``fig_case.draw_inset`` (letters "slide"; ``settle_sector_letters``
then moves any sector letter that still touches a badge along the badge ring,
or inside the rim, so R/L never leave the disc's square) and the legend's first
line is ``fig_case.draw_legend``.  Gallery-specific rules:

* the affordance map rows show the case figure's elevation window
  (``fc.EL_HEAT``, the same in every tile), stretched vertically to at least
  ``MIN_HEAT_H`` when that window is thin at gallery width; a peak beyond it
  (stairs, other levels) gets the case figure's caret at the row's edge (blue
  on the ground-truth row; black on the prediction row, its x drawn just
  inside);
* the rows are named once per tier, left of the first column ("truth" /
  "pred."), never inside a view;
* a peak more than ``fc.MISS_DEG`` from its true bearing (or placed although
  the past position is not visible) gets its slot badge in a lane under the
  prediction row, dodged sideways, with a leader to its x (``miss_lane``).

Figure policy (user decision, 2026-09-24): the figure shows the affordance map
only.  It never mentions poses, odometry or the pose-source ablation; the
prediction drawn is always the deployed model's (the dump's ``vo`` arm).  The
pose-source split stays in the EXP-18 ledger/report.  Honesty that remains:
only the current front image is marked as given to the model (framed, in
colour; legend and axis say so), the other three are marked display only;
ground truth blue vs prediction orange, never swapped; misses are shown, not
hidden; the constant "always behind" guess is printed in every tile for
reference; each tier band says what the model saw of those scenes in training.

Page budget: one tier row is ``BAND_H + TILE_H`` (about 1.8 in).  Planned use:
tiers A-C in the main text (``MAIN_TIERS``, about 6.4 in tall) and D-E in the
supplement (``SUPP_TIERS``); ``tiers=None`` draws every tier present.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_gallery --cases <cases.json> [--tiers A,B,C] [--dumps-root DIR]
      [--topdown-root DIR] [--clip-root DIR] [--metrics DIR|metrics.json] [--lang en|zh] --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from scripts.exp18 import common
from scripts.exp18 import geometry as geo
from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import fig_case as fc
from scripts.exp18.figures import style

from matplotlib.patches import Rectangle  # noqa: E402  (matplotlib is configured in cd.setup)

TIER_ORDER = ("A", "B", "C", "D", "E")
MAIN_TIERS = ("A", "B", "C")  # main-text gallery (about 6.4 in tall)
SUPP_TIERS = ("D", "E")  # supplement gallery
DEFAULT_PERCENTILES = (10, 50, 90)
# scene counts fixed by the pre-registered tier table (README EXP-18), used when metrics.json is not at hand
PREREG_SCENES = {"A": 22, "B": 4, "C": 11, "E": 11}

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); every string on the figure comes from here
# (zh strings avoid full-width parentheses: Droid Sans Fallback sets them with wide side bearings)
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        # tier -> (name, dataset, what the model saw of these scenes in training)
        "tiers": {
            "A": ("Training scenes", "R2R", "scenes used to train the model"),
            "B": ("Held-out scenes", "R2R", "never used to train the head that predicts the map; "
                                            "the image backbone was trained on them"),
            "C": ("Unseen scenes", "R2R val-unseen", "seen by neither the head nor the image backbone"),
            "D": ("Cross-dataset", "HM3D", "head trained on MP3D scenes only; whether the backbone saw HM3D is "
                                           "unknown"),
            "E": ("Designed routes", "out-and-back and loop routes", "in the unseen scenes of C"),
        },
        "n_scenes": "{s} scenes",
        "n_episodes": "{n} episodes",
        "sep": " · ",
        "ranking": "Episodes nearest the 10th / 50th / 90th percentile of their tier's per-episode median bearing "
                   "error (pre-registered rule, not hand-picked)",
        "better": "lower error",
        "worse": "higher error",
        "percentile": {10: "10th percentile", 50: "50th percentile (median)", 90: "90th percentile"},
        "percentile_other": "{p}th percentile",
        "episode": "episode {ep}",
        "col_med": "median",
        "col_pck": "PCK@8",
        "row_ep": "all {n} frames",
        "row_ep_one": "its only frame",
        "row_frame": "frame {t} of {T}",
        "row_floor": ("always-behind guess", "always behind"),
        "frame_na": "no visible past position",
        "pattern": {"out_and_back": "out-and-back", "loop": "loop"},
        "same_as": "same episode as the {p}th",
        "current_prev": "= previous frame, same spot: no view",
        "current": "= at the robot's spot: no view",
        "current_fp": "= at the robot's spot, yet predicted (×)",
        "not_visible": "not visible in any view",
        "false_pos": "not visible, yet predicted (×)",
        "pred_none": "predicted not visible",
        "missing_tile": "episode not available\n{why}",
        "no_rgb": "images not available",
        "gt_tag": "truth",
        "pred_tag": "pred.",
        "legend_input": "front view (framed): current image given to the model",
        "legend_display": "right, back, left views: display only, not given to the model",
    },
    "zh": {
        "tiers": {
            "A": ("训练场景", "R2R", "训练模型用过的场景"),
            "B": ("留出场景", "R2R", "从未用于训练预测 affordance map 的头，图像骨干训练时见过"),
            "C": ("未见场景", "R2R val-unseen", "预测头与图像骨干都没见过"),
            "D": ("跨数据集", "HM3D", "预测头只在 MP3D 场景上训练过，骨干是否见过 HM3D 未知"),
            "E": ("设计路线", "去而复返与绕圈", "位于 C 层的未见场景"),
        },
        "n_scenes": "{s} 个场景",
        "n_episodes": "{n} 集",
        "sep": " · ",
        "ranking": "每层按逐集方位误差中位数取最接近 10 / 50 / 90 分位的一集：预注册规则，非人工挑选",
        "better": "误差低",
        "worse": "误差高",
        "percentile": {10: "10 分位", 50: "50 分位 · 中位数", 90: "90 分位"},
        "percentile_other": "{p} 分位",
        "episode": "第 {ep} 集",
        "col_med": "中位误差",
        "col_pck": "PCK@8",
        "row_ep": "全部 {n} 帧",
        "row_ep_one": "仅此 1 帧",
        "row_frame": "第 {t} / {T} 帧",
        "row_floor": ("恒答正后方基线", "恒答正后方"),
        "frame_na": "无可见历史位置",
        "pattern": {"out_and_back": "去而复返", "loop": "绕圈"},
        "same_as": "与 {p} 分位为同一集",
        "current_prev": "= 上一帧，同一位置：无视角",
        "current": "= 在机器人所在位置：无视角",
        "current_fp": "= 在机器人所在位置，却被预测 ×",
        "not_visible": "任何视角都不可见",
        "false_pos": "不可见，却被预测 ×",
        "pred_none": "被预测为不可见",
        "missing_tile": "该集不可用\n{why}",
        "no_rgb": "图像不可用",
        "gt_tag": "真值",
        "pred_tag": "预测",
        "legend_input": "加框的前视：给模型的当前图像",
        "legend_display": "右、后、左视：仅作展示，模型看不到",
    },
}

CAPTION = {
    "en": (
        "Predicted affordance maps across {tiers_text}, at episodes chosen by a pre-registered rule rather than by "
        "hand. Each tier is one row; its band gives the dataset, the number of scenes and episodes, and what the model "
        "saw of these scenes in training. The columns are the episodes whose median bearing error over all scored "
        "frames lies nearest the 10th, 50th and 90th percentile of that tier; the 90th percentile is the high-error "
        "end (the pre-registered failure case) and is always shown. Each tile shows its episode's most typical frame: "
        "the scored frame whose median bearing error is nearest the episode's (ties: the wider bearing span of the "
        "visible past positions, then the earlier frame). Left: map around the robot, heading up; its rim is the "
        "bearing ring that the strip below unrolls clockwise from the front view's left edge (arrow in the first "
        "tile); dashed radii are the view seams. Blue lines run from the robot through each past position (dot; "
        "1 = oldest of the 8 queried) to its number on the rim. Right: scene and episode, then the median bearing "
        "error of the predicted peaks and the joint PCK@8 over the visible past positions, for all scored frames of "
        "the episode (its median is the ranking statistic), for the frame shown, and for the constant 'always behind' "
        "guess on that frame as a reference. Past positions not visible in any view (the previous frame is taken at "
        "the robot's own spot) and visible ones the model calls not visible are listed by number. Strip (bearing "
        "axis shared by all tiles, labelled at the bottom): the surround images at this frame, of which only the "
        "framed front view is given to the model (the right, back and left images are shown for reference only); the "
        "ground-truth affordance map (blue, 'truth'); the predicted affordance map (orange, 'pred.'). Each slot's map "
        "is divided by its own peak (the prediction also multiplied by its predicted visibility), the maximum over "
        "slots is shown, and colour is linear in that value in both rows. The map rows span ±{el:g}° of elevation around "
        "the horizon, as in the main case figure{stretch}; a caret at a row's edge marks a peak beyond that range "
        "(e.g. on stairs; blue: ground truth; black: prediction, whose x is then drawn just inside the edge). "
        "x: predicted peak of each slot (peaks within {merge:g}° merged; touching marks staggered vertically). A peak "
        "more than {miss:g}° from its true bearing, or placed although the past position is not visible, is numbered "
        "by its badge under the row, joined to the x by a thin line; blue ticks under the prediction repeat the true "
        "bearings.{heldout}{missing}"
    ),
    "zh": (
        "{tiers_text}上的预测 affordance map；各集由预注册规则选出，而非人工挑选。每层占一行，行上方的标题栏注明数据集、"
        "场景数与集数，以及模型训练时见过这些场景的程度。三列分别是全部评分帧上方位误差中位数最接近该层 10、50、90 分位的"
        "一集；90 分位即误差高的一端（预注册的失败案例），必须出现。每格画该集最典型的一帧：方位误差中位数最接近该集中位数"
        "的评分帧（并列时取可见历史位置方位跨度更大者，再取更早者）。左：机器人周围的局部地图，朝向朝上；圆周就是下方条带"
        "从前视左缘顺时针展开的方位环（第一格的箭头），虚线半径为视角分界。蓝线从机器人穿过每个历史位置（圆点；8 个查询中"
        " 1 = 最早）连到圆周上的编号。右：场景与集号，随后是可见历史位置上预测峰值的方位误差中位数与 joint PCK@8，依次"
        "针对该集全部评分帧（其中位数即排序所用的统计量）、所画这一帧，以及作参照的恒答正后方基线在这一帧上的结果。任何"
        "视角都不可见的历史位置（上一帧就在机器人所在位置）和被模型判为不可见的可见位置按编号列出。条带（所有格共用方位轴，"
        "标在最下方）：这一帧的环视图像，只有加框的前视图给模型，右/后/左三张仅作展示；真值 affordance map（蓝，“真值”）；"
        "预测 affordance map（橙，“预测”）。每个槽位的图除以自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，"
        "两行都按该值线性着色。affordance map 行只显示地平线上下 ±{el:g}° 的仰角范围，与主案例图相同{stretch}；行边的小三角"
        "表示峰值落在该范围之外（如楼梯；蓝：真值；黑：预测，其 × 画在行边内侧）。×：各槽位的预测峰值（{merge:g}° 内合并，"
        "相互挨着的上下错开）。偏离真值方位 {miss:g}° 以上、或历史位置不可见却被预测的峰值，在该行下方用其编号标出，细线"
        "连到 ×；预测行下方的蓝色短线重复真值方位。{heldout}{missing}"
    ),
}
STRETCH_TEXT = {
    "en": ", drawn {a:.1f}x taller than the bearing scale so that the thin rows stay legible",
    "zh": "，为便于辨认，纵向按方位刻度的 {a:.1f} 倍绘制",
}
HELDOUT_TEXT = {
    "en": " Held-out scenes (B): the head that predicts the affordance map never trained on them, but the image "
          "backbone did, so they test the head only.",
    "zh": "留出场景（B）：预测 affordance map 的头从未在这些场景上训练，但图像骨干训练时见过它们，因此只检验头本身。",
}
TIERS_TEXT = {
    "en": {"A": "training scenes (A)", "B": "held-out scenes (B)", "C": "unseen scenes (C)",
           "D": "a second dataset, HM3D (D)", "E": "designed out-and-back and loop routes (E)"},
    "zh": {"A": "训练场景（A）", "B": "留出场景（B）", "C": "未见场景（C）", "D": "跨数据集 HM3D（D）",
           "E": "设计路线（去而复返、绕圈，E）"},
}
MISSING_TEXT = {
    "en": " Tier{s} {tiers} not shown: no scored episodes yet.",
    "zh": "{tiers} 层暂无评分数据，未画出。",
}

# The prediction drawn is always the deployed model's output (dump arm "vo").
ARM = fc.ARM

# --------------------------------------------------------------------------- #
# Geometry of the page (inches)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # fc.Page places axes on this width
X_GRID = 0.27  # left edge of the first tile column; the "truth" / "pred." tags sit left of it
COL_GAP = 0.13
R_MARGIN = 0.03
W_TILE = (FIG_W - R_MARGIN - X_GRID - 2 * COL_GAP) / 3.0
DISC = 0.80  # side of the disc axes (the disc itself is 0.74 of it)
DISC_DX = -0.02  # disc axes offset from the tile's left edge
TEXT_DX = 0.87  # text block starts here, relative to the tile's left edge (clear of the widest rim pill)
EL_RGB = 12.0  # RGB row: elevation +-12 deg (square degrees)
EL_HEAT = float(fc.EL_HEAT)  # affordance map rows: the case figure's window, so one case looks the same in both
MIN_HEAT_H = 10.0 / 72.0  # ... drawn at least this tall (a thinner row cannot hold a staggered x)
CARET_PT = float(getattr(fc, "CARET_PT", 4.4))  # off-row caret (as the case figure's)
RGB_H = W_TILE * 2 * EL_RGB / 360.0
HEAT_SQUARE_H = W_TILE * 2 * EL_HEAT / 360.0
HEAT_H = max(HEAT_SQUARE_H, MIN_HEAT_H)
HEAT_STRETCH = HEAT_H / HEAT_SQUARE_H  # vertical exaggeration of the map rows (1.0 = square degrees)
LANE_H = 0.14
ROW_GAP = 0.024
TICK_PT = (1.0, 4.0)  # true-bearing ticks under the prediction row (points below its edge)
MISS_BADGE_PT = 8.3  # centre of a miss badge below the prediction row's edge (points)
MISS_H = (MISS_BADGE_PT + 3.7 + 0.8) / 72.0  # lane under the prediction row: ticks + miss badges
BODY_Y = DISC + 0.015  # strip starts here, relative to the tile's top
TILE_H = BODY_Y + LANE_H + RGB_H + 2 * ROW_GAP + 2 * HEAT_H + MISS_H
BAND_H = 0.17  # tier band above each row of tiles
TIER_GAP = 0.07
TOP_H = 0.33
AXIS_H = 0.25
LEGEND2_H = 0.17
RGB_RING_W = 856  # ring columns per 360 deg (multiple of 8: exact roll), ~400 dpi at W_TILE
HEAT_RING_W = 720
FS = {"col": 6.6, "rank": 6.0, "tier": 8.5, "tier_name": 6.8, "tier_sub": 6.0, "scene": 6.3, "line": 6.0,
      "muted": 5.8, "hdr": 5.7, "note": 5.7, "tag": 5.9, "axis": 5.8, "legend": 6.1, "missing": 6.0}
LINE_PT = 7.4  # baseline-to-baseline distance in the text block ...
MIN_LINE_PT = 6.5  # ... tightened down to this when a tile has many notes
COL_SEP_PT = 2.8  # between the text block's table columns
GT_INK = "#1c5cab"  # tag colours: the dark tones of the two ramps (as fig_anim's row labels)
PRED_INK = "#b53f12"


def fig_height(n_tiers: int) -> float:
    n = max(n_tiers, 1)
    return TOP_H + n * (BAND_H + TILE_H) + (n - 1) * TIER_GAP + AXIS_H + fc.LEGEND_H + LEGEND2_H


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeStats:
    """The prediction over all scored frames of an episode (compute_metrics' per-episode numbers)."""

    median: float  # median bearing error over every visible (frame, slot)
    hits: int  # joint PCK@8 hits
    n: int  # visible (frame, slot) pairs
    n_frames: int  # scored frames


@dataclass
class Tile:
    tier: str
    percentile: int
    pick: dict
    dump: Optional[dd.Dump] = None
    row: Optional[dd.CaseRow] = None
    episode: Optional[EpisodeStats] = None
    span_deg: float = float("nan")
    views: Optional[np.ndarray] = None
    level: object = None
    same_as: Optional[int] = None  # percentile of an earlier tile showing the same episode
    problems: List[str] = field(default_factory=list)
    error: Optional[str] = None  # the tile cannot be drawn at all


def load_cases(cases_json) -> dict:
    if isinstance(cases_json, dict):
        return cases_json
    return json.loads(Path(cases_json).read_text(encoding="utf-8"))


def scene_counts(cases: dict, metrics=None) -> Dict[str, int]:
    """Scenes per tier: metrics.json ``tiers.<t>.n_scenes`` (``metrics`` = dict, file or dir; default the
    cases' ``metrics_dir``), else the pre-registered counts."""
    out = dict(PREREG_SCENES)
    m = metrics
    if m is None and cases.get("metrics_dir"):
        m = Path(cases["metrics_dir"])
    if m is not None and not isinstance(m, dict):
        p = Path(m)
        p = p / "metrics.json" if p.is_dir() else p
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = None
    for t, blk in ((m or {}).get("tiers") or {}).items():
        if isinstance(blk, dict) and blk.get("n_scenes"):
            out[t] = int(blk["n_scenes"])
    return out


def dump_path_for(pick: dict, dumps_root=None) -> Path:
    """``<dumps_root>/<tier>/<scene>/<clip>.npz``; without a root, the pick's recorded ``npz_path``."""
    tail = Path(pick["tier"]) / pick["scene"] / (pick["clip"] + ".npz")
    if dumps_root is not None:
        return Path(dumps_root) / tail
    if pick.get("npz_path"):
        return Path(pick["npz_path"])
    return common.EXP_ROOT / "dumps" / tail


def typical_row(dump: dd.Dump, arm: str = ARM) -> Tuple[dd.CaseRow, float, EpisodeStats]:
    """The episode's most typical scored frame, its GT bearing span and the episode's statistics.

    Typical = the frame whose median bearing error over its GT-visible slots is
    nearest the episode's median over all (frame, slot) pairs; ties -> the wider
    GT bearing span (circular range), then the earlier frame.  Frames without a
    visible slot are never picked (unless no frame has one).
    """
    idx = dump.query_rows(arm)
    if not idx:
        raise ValueError(f"no scored rows with a prediction in {dump.path}")
    recs = [dd.case_row(dump, i) for i in idx]
    errs, hits, n = [], 0, 0
    for r in recs:
        e = r.arms[arm].err
        errs.append(e[r.visible & np.isfinite(e)])
        s = r.summary(arm)
        hits, n = hits + s["hits"], n + s["n"]
    errs = np.concatenate(errs) if errs else np.zeros(0)
    ep = EpisodeStats(median=float(np.median(errs)) if errs.size else float("nan"), hits=hits, n=n,
                      n_frames=len(recs))
    best, best_key, best_span = recs[0], None, 0.0
    for r in recs:
        s = r.summary(arm)
        span = float(geo.circular_range_deg(r.gt_bearing[r.visible])) if r.n_visible else 0.0
        if not s["n"] or not np.isfinite(s["median"]):
            continue
        key = (round(abs(s["median"] - ep.median), 9), -round(span, 9), r.index)
        if best_key is None or key < best_key:
            best, best_key, best_span = r, key, span
    return best, best_span, ep


def prepare_tile(tier: str, pick: dict, dumps_root, topdown_root, clip_root_override) -> Tile:
    tile = Tile(tier=tier, percentile=int(pick.get("percentile", -1)), pick=pick)
    path = dump_path_for(pick, dumps_root)
    try:
        dump = dd.load_dump(path)
    except (OSError, ValueError, KeyError) as exc:
        tile.error = f"dump {path.name}: {type(exc).__name__}"
        print(f"[fig_gallery] {tier} P{tile.percentile}: cannot load {path}: {exc}")
        return tile
    if ARM not in dump.arms:
        tile.error = f"no '{ARM}' prediction in {path.name}"
        return tile
    tile.dump = dump
    try:
        tile.row, tile.span_deg, tile.episode = typical_row(dump)
    except ValueError as exc:
        tile.error = str(exc)
        return tile
    # the dump must reproduce the per-episode numbers cases.json ranked on
    ep = tile.episode
    rec_med = pick.get("vo_bearing_err_median")
    if rec_med is not None and abs(float(rec_med) - ep.median) > 0.05:
        tile.problems.append(f"episode median {ep.median:.3f} deg differs from cases.json {float(rec_med):.3f} deg")
    if pick.get("vo_pck8") is not None and pick.get("n_visible_slots"):
        rec_hits = float(pick["vo_pck8"]) * int(pick["n_visible_slots"])
        if int(pick["n_visible_slots"]) != ep.n or abs(rec_hits - ep.hits) > 0.01:
            tile.problems.append(f"episode PCK@8 {ep.hits}/{ep.n} differs from cases.json "
                                 f"{rec_hits:.1f}/{pick['n_visible_slots']}")
    if pick.get("n_scored_rows") is not None and int(pick["n_scored_rows"]) != ep.n_frames:
        tile.problems.append(f"{ep.n_frames} scored frames, cases.json says {pick['n_scored_rows']}")
    try:
        tile.views = surround_views(dump, tile.row.frame, clip_root_override)
    except (OSError, KeyError, FileNotFoundError, ValueError) as exc:
        tile.problems.append(f"no surround images: {exc}")
    try:
        tile.level = dd.topdown_level(dump, float(tile.row.cur_pos[1]), root=topdown_root)
    except (OSError, KeyError, ValueError) as exc:
        tile.problems.append(f"no top-down map: {exc}")
    for msg in tile.problems:
        print(f"[fig_gallery] {tier} P{tile.percentile} {pick.get('clip_key')}: {msg}")
    return tile


def gallery_rows(cases: dict, tiers: Optional[Sequence[str]] = None
                 ) -> Tuple[List[Tuple[str, dict]], List[str], Tuple[int, ...]]:
    """(tier, gallery entry) for requested tiers with picks, requested tiers without, and the percentiles."""
    gal = cases.get("gallery") or {}
    wanted = [t for t in TIER_ORDER if tiers is None or t in tiers]
    present, missing = [], []
    percentiles: Optional[Tuple[int, ...]] = None
    for tier in wanted:
        g = gal.get(tier) or {}
        if g.get("status") == "ok" and g.get("picks"):
            present.append((tier, g))
            if percentiles is None:
                percentiles = tuple(int(p["percentile"]) for p in g["picks"])
        else:
            missing.append(tier)
    return present, missing, percentiles or DEFAULT_PERCENTILES


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def fit_lines(fig, text: str, fs: float, width_pt: float, **kw) -> List[str]:
    """Greedy word wrap of ``text`` to ``width_pt`` (CJK text wraps between characters)."""
    if cd.text_width_pt(fig, text, fs, **kw) <= width_pt:
        return [text]
    spaced = " " in text.strip()
    words, joiner = (text.split(" "), " ") if spaced else (list(text), "")
    lines, cur = [], ""
    for w in words:
        cand = (cur + joiner + w) if cur else w
        if cur and cd.text_width_pt(fig, cand, fs, **kw) > width_pt:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def draw_missing_tile(page: fc.Page, x: float, y: float, text: str) -> None:
    ax = page.ax(x, y, W_TILE, TILE_H - MISS_H)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=cd.MAP_PLATE, ec=style.AXIS, lw=0.5))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=FS["missing"], color=style.INK_2, linespacing=1.3)


def tile_notes(r: dd.CaseRow, L: dict) -> List[Tuple[List[List[int]], str]]:
    """(runs of slots, text) for past positions the strip cannot show on its own, one line per kind.

    Slots without a ground-truth view (at the robot's own spot: the previous
    frame, or earlier frames when the robot turned on the spot; or hidden in
    every view) and GT-visible slots the model calls not visible (P(not
    visible) > 0.5).  A slot without a view that the model still places in view
    is also numbered under the prediction row.  Slots of one kind share a line;
    consecutive ones share a badge ("3–5").
    """
    p = r.arms[ARM]
    kinds: Dict[str, List[int]] = {}
    for k in r.invisible_slots():
        placed = p.none_p[k] <= 0.5 and p.peak_view[k] >= 0
        if r.gt_dist[k] < 0.1:
            if placed:
                key = "current_fp"
            elif int(r.hist_frames[k]) == r.frame - 1:
                key = "current_prev"
            else:
                key = "current"
            kinds.setdefault(key, []).append(k)
        else:
            kinds.setdefault("false_pos" if placed else "not_visible", []).append(k)
    for k in range(dd.K):
        if r.visible[k] and p.none_p[k] > 0.5:
            kinds.setdefault("pred_none", []).append(k)
    notes = []
    for key in ("current_prev", "current", "current_fp", "not_visible", "false_pos", "pred_none"):
        runs: List[List[int]] = []
        for k in sorted(kinds.get(key, [])):
            if runs and runs[-1][-1] == k - 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        if runs:
            notes.append((runs, L[key]))
    return notes


def gt_peak_elev(r: dd.CaseRow) -> np.ndarray:
    """[8] elevation (deg, up-positive) of each visible slot's ground-truth peak, NaN elsewhere."""
    el = getattr(r, "gt_peak_elev", None)
    if el is not None:
        return np.asarray(el, dtype=np.float64)
    out = np.full(dd.K, np.nan)
    for k in np.nonzero(r.visible)[0]:
        v = int(r.gt_class[k]) - 1
        row, col = np.unravel_index(int(np.argmax(r.gt_maps[k, v])), r.gt_maps[k, v].shape)
        out[k] = float(geo.pixel_to_bearing_elev(v, col, row)[1])
    return out


def surround_views(dump: dd.Dump, frame: int, clip_root_override=None) -> np.ndarray:
    """``dd.surround_views``, also for a clip whose last chunk holds one frame.

    ``np.savez`` of a one-element object array of JPEG byte arrays collapses it
    to a 2-D object array of ints, which ``dd.surround_views`` cannot open
    (b8cTxDM8gDG/clip_700155 frame 64); the bytes are rebuilt here.  Remove
    once data.py handles it.
    """
    clip_dir = dd.resolve_clip_dir(dump, clip_root_override)
    try:
        return dd.surround_views(clip_dir, frame)
    except OSError:
        path, j = dd._frame_index(str(clip_dir))[frame]
        with np.load(path, allow_pickle=True) as z:
            return np.stack([np.asarray(Image.open(io.BytesIO(np.asarray(z[f"rgb_{v}"][j], dtype=np.uint8).tobytes()))
                                        .convert("RGB")) for v in geo.VIEW_NAMES])


def _boxes_overlap(a, b, pad: float) -> bool:
    return not (a.x1 + pad <= b.x0 or b.x1 + pad <= a.x0 or a.y1 + pad <= b.y0 or b.y1 + pad <= a.y0)


def settle_sector_letters(ax, names: Sequence[str], half_m: float, rays: Sequence[float] = (),
                          ring_offset_pt: float = 5.2, letter_pt: float = 4.6, max_slide_deg: float = 38.0) -> List[str]:
    """Move any sector letter of ``fc.draw_inset`` that touches a rim badge (or was pushed outward).

    ``cd.disc_sector_letters`` tests badges by angle, which misses the radial
    extent of a pill badge ("1–3") beside the disc, and pushes a displaced R/L
    letter outward, into the column gap or the text block.  Here every letter
    on or outside the badge ring is tested against the drawn extents of the
    badges, the scale-bar label and bar, and the rim leaders / arrow; a letter
    that touches one, or an R/L letter off the badge ring, slides along the
    badge ring to the free spot nearest its sector centre within
    +-``max_slide_deg`` (it stays in its own 90 deg sector), else goes just
    inside the rim at the angle within that range nearest the centre that keeps
    clear of the blue ``rays`` (plot angles, degrees).  A letter the disc code
    already put inside the rim stays.  Returns the letters moved.
    """
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    ring = half_m + ring_offset_pt * per_pt
    inner_pt = half_m / per_pt - letter_pt / 2 - 2.4
    obstacles = []
    letters = []
    for t in ax.texts:
        if t.get_bbox_patch() is not None:
            t.update_bbox_position_size(rend)
            obstacles.append(t.get_bbox_patch().get_window_extent(rend))
        elif t.get_text() in names:
            letters.append(t)
        elif t.get_text().strip():
            obstacles.append(t.get_window_extent(rend))
    for ln in ax.lines:  # rim leaders and the direction arrow (clip-free), the scale bar (haloed)
        if (not ln.get_clip_on() or ln.get_path_effects()) and len(ln.get_xdata()) > 1:
            obstacles.append(ln.get_window_extent(rend))
    for pa in ax.patches:
        if not pa.get_clip_on():
            obstacles.append(pa.get_window_extent(rend))
    steps = sorted(np.arange(-max_slide_deg, max_slide_deg + 1e-6, 1.0), key=lambda d: (abs(d), -d))
    need = math.degrees((letter_pt / 2 + 1.4) / max(inner_pt, 1.0))
    moved = []
    for t in letters:
        v = list(names).index(t.get_text())
        centre = 90.0 + geo.VIEW_YAWS_DEG[v]
        x0, y0 = t.get_position()
        radius = math.hypot(x0, y0)
        if radius < half_m:
            continue
        off_ring = radius > ring + 0.5 * per_pt

        def clear() -> bool:
            bb = t.get_window_extent(rend)
            return not any(_boxes_overlap(bb, o, 0.6 * px) for o in obstacles)

        if clear() and not off_ring:
            continue
        for d in steps:
            th = math.radians(centre + d)
            t.set_position((ring * math.cos(th), ring * math.sin(th)))
            if clear():
                break
        else:
            dist = {d: min([abs((centre + d - r + 180) % 360 - 180) for r in rays] + [360.0]) for d in steps}
            ok = [d for d in steps if dist[d] >= need]
            th = math.radians(centre + (ok[0] if ok else max(steps, key=lambda d: dist[d])))
            t.set_position((inner_pt * per_pt * math.cos(th), inner_pt * per_pt * math.sin(th)))
        moved.append(t.get_text())
    return moved


def draw_text_block(page: fc.Page, x: float, y: float, tile: Tile, L: dict) -> float:
    """Scene and episode, a small table (episode, frame shown, always-behind guess), notes; right of the disc.

    Table columns: median bearing error and joint PCK@8 (headers on the episode
    line).  Lines are laid out first, then drawn at LINE_PT pitch, tightened
    (down to MIN_LINE_PT) when they would run past the disc into the strip
    lane.  Returns the block's height in points (the caller warns if it still
    does).
    """
    fig = page.fig
    r, pick, ep = tile.row, tile.pick, tile.episode
    width_pt = (W_TILE - TEXT_DX) * 72.0
    items: List[dict] = []

    def add(text, fs, color=style.INK, weight="normal", badges=(), gap=0.0):
        indent = sum(cd.badge_width_pt(b[0]) + 1.0 for b in badges) + (1.2 if badges else 0.0)
        for j, line in enumerate(fit_lines(fig, text, fs, width_pt - indent, fontweight=weight)):
            items.append(dict(kind="text", text=line, fs=fs, color=color, weight=weight, indent=indent,
                              badges=list(badges) if j == 0 else [], gap=gap if j == 0 else 0.0))

    pattern = str(pick.get("pattern") or "")
    head = str(pick.get("scene") or tile.dump.scene)
    if pattern:
        head += " · " + L["pattern"].get(pattern, pattern)
    add(head, FS["scene"], weight="bold")

    # ---- the table: (label(s), median text, PCK text, colour, weight)
    s, sf = r.summary(ARM), r.summary("floor")

    def med(v, nd):
        return "–" if not np.isfinite(v) else f"{v:.{nd}f}°"

    ep_label = L["row_ep"].format(n=ep.n_frames) if ep.n_frames > 1 else L["row_ep_one"]
    rows = [((ep_label,), med(ep.median, 1), f"{ep.hits}/{ep.n}" if ep.n else "–", style.INK_2, "normal"),
            ((L["row_frame"].format(t=r.frame, T=tile.dump.frame_count),), med(s["median"], 1),
             f"{s['hits']}/{s['n']}" if s["n"] else "–", style.INK, "bold"),
            (tuple(L["row_floor"]), med(sf["median"], 0), f"{sf['hits']}/{sf['n']}" if sf["n"] else "–",
             style.MUTED, "normal")]
    fs_row = FS["line"]
    w_pck = max([cd.text_width_pt(fig, L["col_pck"], FS["hdr"])]
                + [cd.text_width_pt(fig, p_, fs_row, fontweight=w) for _, _, p_, _, w in rows])
    med_right = width_pt - w_pck - COL_SEP_PT
    table = []
    for labels, m, p_, color, weight in rows:
        room = med_right - cd.text_width_pt(fig, m, fs_row, fontweight=weight) - COL_SEP_PT
        choice = None
        for lab in labels:
            for fs in (fs_row, fs_row - 0.3):
                if cd.text_width_pt(fig, lab, fs, fontweight=weight) <= room:
                    choice = (lab, fs)
                    break
            if choice:
                break
        if choice is None:
            choice = (labels[-1], fs_row - 0.3)
            print(f"[fig_gallery] {tile.tier} P{tile.percentile}: table label {labels[-1]!r} wider than its column")
        table.append(dict(kind="row", label=choice[0], label_fs=choice[1], med=m, pck=p_, fs=fs_row, color=color,
                          weight=weight, gap=0.0))
    ep_text = L["episode"].format(ep=pick.get("episode_id") or tile.dump.episode_id)
    w_med_hdr = cd.text_width_pt(fig, L["col_med"], FS["hdr"])
    hdr = dict(kind="hdr", text=ep_text, gap=0.0)
    if cd.text_width_pt(fig, ep_text, FS["muted"]) + COL_SEP_PT > med_right - w_med_hdr:
        items.append(dict(kind="text", text=ep_text, fs=FS["muted"], color=style.INK_2, weight="normal", indent=0.0,
                          badges=[], gap=0.0))
        hdr["text"] = ""
    items.append(hdr)
    items += table
    if tile.same_as is not None:
        add(L["same_as"].format(p=tile.same_as), FS["muted"], style.INK_2)
    if not s["n"]:
        add(L["frame_na"], FS["note"], style.INK_2)
    for j, (runs, text) in enumerate(tile_notes(r, L)):
        add(text, FS["note"], style.INK_2, badges=[(dd.group_label(run), run[0]) for run in runs],
            gap=1.2 if j == 0 else 0.0)

    room = BODY_Y * 72.0
    gaps = sum(it["gap"] for it in items) + 1.5  # + the step under the scene line
    pitch = float(np.clip((room - gaps) / max(len(items), 1), MIN_LINE_PT, LINE_PT))
    ax = page.ax(x + TEXT_DX, y, W_TILE - TEXT_DX, BODY_Y)  # points, y down
    ax.set_xlim(0, width_pt)
    ax.set_ylim(room, 0)
    ax.axis("off")
    y_pt = 0.0
    for n, it in enumerate(items):
        y_pt += it["gap"] + (1.5 if n == 1 else 0.0)
        yc = y_pt + 0.5 * pitch
        if it["kind"] == "text":
            xb = 0.4
            for label, slot in it["badges"]:
                w = cd.badge_width_pt(label)
                cd.history_badge(ax, xb + w / 2, yc, label, slot)
                xb += w + 1.0
            ax.text(it["indent"], yc, it["text"], ha="left", va="center", fontsize=it["fs"], color=it["color"],
                    fontweight=it["weight"], clip_on=False)
        elif it["kind"] == "hdr":
            if it["text"]:
                ax.text(0, yc, it["text"], ha="left", va="center", fontsize=FS["muted"], color=style.INK_2)
            ax.text(med_right, yc, L["col_med"], ha="right", va="center", fontsize=FS["hdr"], color=style.MUTED)
            ax.text(width_pt, yc, L["col_pck"], ha="right", va="center", fontsize=FS["hdr"], color=style.MUTED)
        else:
            kw = dict(va="center", color=it["color"], fontweight=it["weight"], clip_on=False)
            ax.text(0, yc, it["label"], ha="left", fontsize=it["label_fs"], **kw)
            ax.text(med_right, yc, it["med"], ha="right", fontsize=it["fs"], **kw)
            ax.text(width_pt, yc, it["pck"], ha="right", fontsize=it["fs"], **kw)
        y_pt += pitch
    return y_pt


def miss_lane(ax_pr, ax_ms, drawn, pt_per_deg: float, tick: float, el_h: float) -> None:
    """Slot badge of each numbered x in the lane under the prediction row, joined to its x by a leader.

    ``drawn``: (x deg, y row units, slot or None) of every drawn x.  Badges sit
    in one row MISS_BADGE_PT under the prediction row, dodged sideways
    (``cd.dodge_1d``, kept inside the tile) so none overlap; a thin leader runs
    from just under the x to the row's edge (under the marks, so crossing x's
    keep their halo) and on to the badge.
    """
    miss = sorted([d for d in drawn if d[2] is not None], key=lambda d: d[0])
    if not miss:
        return
    widths = [cd.badge_width_pt(str(k + 1)) / pt_per_deg for _, _, k in miss]
    xs = cd.dodge_1d([m[0] for m in miss], widths, 0.0, 360.0, 1.2 / pt_per_deg)
    top = -(MISS_BADGE_PT - 3.7)
    for (xm, yd, k), xb in zip(miss, xs):
        y0 = yd - (fc.MARK_PT / 2 + 0.9) * tick
        if y0 > -el_h:
            ax_pr.plot([xm, xm], [y0, -el_h], color=style.INK_2, lw=0.5, zorder=6.5, clip_on=False,
                       solid_capstyle="butt", path_effects=cd.HALO_THIN)
        ax_ms.plot([xm, xb], [0.0, top], color=style.INK_2, lw=0.5, zorder=5, clip_on=False, solid_capstyle="butt")
        cd.history_badge(ax_ms, xb, -MISS_BADGE_PT, str(k + 1), k, zorder=8)


def axis_labels_for(fcL: dict) -> Tuple[str, ...]:
    """The case figure's view names over its bearing labels ("Front · model input" / "0° (heading)")."""
    return tuple(f"{v}\n{a}" for v, a in zip(fcL["views"], fcL["axis"]))


def draw_bearing_axis(ax, fcL: dict) -> None:
    cd.azimuth_axis(ax, axis_labels_for(fcL), fs=FS["axis"])
    ax.tick_params(axis="x", which="major", pad=1.2)
    labels = ax.get_xticklabels()
    for t in labels:
        t.set_linespacing(1.25)
    if labels:
        labels[0].set_color(style.INK)


def draw_strip(page: fc.Page, x: float, y: float, tile: Tile, L: dict, fcL: dict, axis_labels: bool) -> None:
    """Lane of GT badges, RGB row, GT and predicted map rows, x marks, and the tick / miss-badge lane."""
    r = tile.row
    el_h = EL_HEAT  # the rows' y axis is elevation, spanning +-EL_HEAT (drawn HEAT_STRETCH x taller than square)
    y_lane = y + BODY_Y
    y_rgb = y_lane + LANE_H
    y_gt = y_rgb + RGB_H + ROW_GAP
    y_pr = y_gt + HEAT_H + ROW_GAP
    ax_lane = page.ax(x, y_lane, W_TILE, LANE_H)
    ax_rgb = page.ax(x, y_rgb, W_TILE, RGB_H)
    ax_gt = page.ax(x, y_gt, W_TILE, HEAT_H)
    ax_pr = page.ax(x, y_pr, W_TILE, HEAT_H)
    ax_ms = page.ax(x, y_pr + HEAT_H, W_TILE, MISS_H)

    if tile.views is not None:
        cd.draw_rgb_row(ax_rgb, cd.rgb_strip(tile.views, RGB_RING_W, EL_RGB), EL_RGB)
    else:
        cd.setup_strip_axes(ax_rgb, EL_RGB)
        ax_rgb.add_patch(Rectangle((0, -EL_RGB), 360, 2 * EL_RGB, fc=cd.MAP_PLATE, ec=style.AXIS, lw=0.5))
        ax_rgb.text(180, 0, L["no_rgb"], ha="center", va="center", fontsize=FS["note"], color=style.MUTED)
    cd.draw_heat_row(ax_gt, cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, el_h), el_h, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, cd.heat_strip(dd.pred_composite(r, ARM), HEAT_RING_W, el_h), el_h, cd.PRED_CMAP)
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")
    ms_pt = MISS_H * 72.0
    ax_ms.set_xlim(0, 360)
    ax_ms.set_ylim(-ms_pt, 0)  # points below the prediction row's edge
    cd.clean_axes(ax_ms)
    ax_ms.patch.set_visible(False)

    # ground truth: numbered badges in the lane, guide through the RGB row, ticks under the prediction
    pt_per_deg = W_TILE * 72.0 / 360.0
    groups = r.groups
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in groups])
    labels = [dd.group_label(g) for g in groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / pt_per_deg for s in labels], 0.0, 360.0, 0.8 / pt_per_deg)
    y_badge = 0.58
    for g, t, xb, lab in zip(groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([xb, xb, t, t], [y_badge, 0.34, 0.12, 0.0], color=col, lw=0.5, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, xb, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.5, zorder=3)
        ax_ms.plot([t, t], [-TICK_PT[0], -TICK_PT[1]], color=col, lw=0.8, zorder=3, clip_on=False,
                   solid_capstyle="butt")

    # predicted peaks (fig_case's rule): one x per cluster; misses > fc.MISS_DEG alone and numbered; marks
    # closer than their own width staggered vertically (+-STAGGER_PT)
    p = r.arms[ARM]
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]  # row units per point
    shown = [k for k in range(dd.K) if r.valid[k] and p.none_p[k] <= 0.5 and p.peak_view[k] >= 0]
    alone = [k for k in shown if (not r.visible[k]) or p.err[k] > fc.MISS_DEG]
    merged = [k for k in shown if k not in alone]
    px = {k: float(cd.strip_x(p.peak_bearing[k])) for k in shown}
    marks = []  # (x, elevation, slot needing a badge or None)
    for cl in cd.cluster_1d([px[k] for k in merged], fc.MERGE_DEG):
        ks = [merged[i] for i in cl]
        marks.append((float(np.mean([px[k] for k in ks])), float(np.mean([p.peak_elev[k] for k in ks])), None))
    marks += [(px[k], float(p.peak_elev[k]), k) for k in alone]
    marks.sort(key=lambda m: m[0])
    offsets, sign = [0.0] * len(marks), 1.0
    for j in range(1, len(marks)):
        if (marks[j][0] - marks[j - 1][0]) * pt_per_deg < fc.MARK_PT + 3.4:  # x plus its halo
            if offsets[j - 1] == 0.0:
                offsets[j - 1] = sign * fc.STAGGER_PT
            offsets[j] = -np.sign(offsets[j - 1]) * fc.STAGGER_PT
            sign = -sign
    lim = el_h - (fc.MARK_PT / 2 + 0.3) * tick  # the whole x stays inside its row
    drawn = []  # (x deg, y in row units, slot to number or None)
    for (xm, el, k), off in zip(marks, offsets):
        if abs(el) > el_h:  # beyond the row: caret tip at the edge, the x inside it, clear of the caret
            sgn = 1.0 if el > 0 else -1.0
            yd = sgn * (el_h - (CARET_PT + 3.0 + abs(off)) * tick)
            ax_pr.plot([xm], [sgn * (el_h - (CARET_PT / 2 + 0.4) * tick)], ls="none", marker="^" if sgn > 0 else "v",
                       ms=CARET_PT, color=style.INK, mec="white", mew=0.5, zorder=7.5, clip_on=False)
        else:
            yd = el + off * tick
        yd = float(np.clip(yd, -lim, lim))
        cd.peak_mark(ax_pr, xm, yd, size=fc.MARK_PT)
        drawn.append((xm, yd, k))
    # ground-truth peaks beyond the row: a blue caret at the ground-truth row's edge (one per spot and side)
    gt_el = np.nan_to_num(gt_peak_elev(r), nan=0.0)
    tick_gt = cd.pts_to_data(ax_gt, 0.0, 1.0)[1]
    done = []
    for k in np.nonzero(r.visible & (np.abs(gt_el) > el_h - 1.0))[0]:
        xk, sgn = float(cd.strip_x(r.gt_bearing[k])), (1.0 if gt_el[k] > 0 else -1.0)
        if any(abs(xk - x0) < 1.5 and sgn == s0 for x0, s0 in done):
            continue
        done.append((xk, sgn))
        ax_gt.plot([xk], [sgn * (el_h - (CARET_PT / 2 + 0.4) * tick_gt)], ls="none", marker="^" if sgn > 0 else "v",
                   ms=CARET_PT, mfc=GT_INK, mec="white", mew=0.5, zorder=6, clip_on=False)
    miss_lane(ax_pr, ax_ms, drawn, pt_per_deg, tick, el_h)
    if axis_labels:
        draw_bearing_axis(ax_ms, fcL)


def draw_tile(page: fc.Page, x: float, y: float, tile: Tile, L: dict, fcL: dict, first: bool,
              axis_labels: bool) -> None:
    if tile.error is not None:
        draw_missing_tile(page, x, y, L["missing_tile"].format(why=tile.error))
        if axis_labels:  # keep the shared bearing axis under every column
            ax = page.ax(x, y + TILE_H - MISS_H, W_TILE, MISS_H)
            ax.set_xlim(0, 360)
            cd.clean_axes(ax)
            ax.patch.set_visible(False)
            draw_bearing_axis(ax, fcL)
        return
    draw_strip(page, x, y, tile, L, fcL, axis_labels)
    used = draw_text_block(page, x, y, tile, L)
    if used > BODY_Y * 72.0 + 1.0:
        print(f"[fig_gallery] {tile.tier} P{tile.percentile}: text block runs {used - BODY_Y * 72.0:.1f} pt into the "
              f"strip lane")
    ax_in = page.ax(x + DISC_DX, y, DISC, DISC)
    if tile.level is not None:
        r = tile.row
        fc.draw_inset(ax_in, tile.level, tile.dump, r, show_arrow=first, L=fcL, letters="slide")
        if hasattr(fc, "inset_half"):
            half = fc.inset_half(r)
        else:  # draw_inset's disc radius (metres)
            half = max(1.0, 1.12 * float(np.max(r.gt_dist[r.visible])) if r.visible.any() else 1.12)
        moved = settle_sector_letters(ax_in, fcL["sectors"], half, rays=[90.0 + float(r.gt_bearing[g[0]])
                                                                          for g in r.groups])
        if moved:
            print(f"[fig_gallery] {tile.tier} P{tile.percentile}: sector letters moved clear of badges: {moved}")
    else:
        ax_in.axis("off")
        ax_in.text(0.5, 0.5, "—", ha="center", va="center", transform=ax_in.transAxes, color=style.MUTED)


def draw_tier_band(page: fc.Page, y: float, tier: str, g: dict, n_scenes: Optional[int], L: dict) -> None:
    """One line above the tier's tiles: letter, name, dataset · scenes · episodes · what training saw."""
    fig = page.fig
    name, dataset, note = L["tiers"].get(tier, (f"tier {tier}", "", ""))
    parts = [dataset] if dataset else []
    if n_scenes:
        parts.append(L["n_scenes"].format(s=n_scenes))
    if g.get("n_episodes") is not None:
        parts.append(L["n_episodes"].format(n=int(g["n_episodes"])))
    if note:
        parts.append(note)
    yc = y + BAND_H * 0.42
    x = 0.02
    page.text(x, yc, tier, ha="left", va="center", fontsize=FS["tier"], fontweight="bold", color=style.INK)
    x += cd.text_width_pt(fig, tier, FS["tier"], fontweight="bold") / 72.0 + 0.09
    page.text(x, yc, name, ha="left", va="center", fontsize=FS["tier_name"], fontweight="bold", color=style.INK)
    x += cd.text_width_pt(fig, name, FS["tier_name"], fontweight="bold") / 72.0 + 0.10
    page.text(x, yc, L["sep"].join(parts), ha="left", va="center", fontsize=FS["tier_sub"], color=style.INK_2)


def draw_row_tags(page: fc.Page, y_tile: float, L: dict) -> None:
    """'truth' / 'pred.' once per tier, left of the first column, level with the two map rows."""
    y_gt = y_tile + BODY_Y + LANE_H + RGB_H + ROW_GAP
    y_pr = y_gt + HEAT_H + ROW_GAP
    for yt, text, color in ((y_gt, L["gt_tag"], GT_INK), (y_pr, L["pred_tag"], PRED_INK)):
        page.text(X_GRID - 0.035, yt + HEAT_H / 2, text, ha="right", va="center", fontsize=FS["tag"],
                  fontweight="bold", color=color)


def draw_column_headers(page: fc.Page, percentiles: Sequence[int], L: dict) -> None:
    """Ranking rule on one line; below it the column titles sit on an arrow from lower to higher error."""
    fig = page.fig
    page.text(X_GRID, 0.075, L["ranking"], ha="left", va="center", fontsize=FS["rank"], color=style.INK_2)
    y_col = 0.215
    total = (FIG_W - R_MARGIN - X_GRID) * 72.0
    ax = page.ax(X_GRID, y_col - 0.05, FIG_W - R_MARGIN - X_GRID, 0.10)
    ax.set_xlim(0, total)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    w_b = cd.text_width_pt(fig, L["better"], FS["rank"])
    w_w = cd.text_width_pt(fig, L["worse"], FS["rank"])
    ax.text(0, 0, L["better"], ha="left", va="center", fontsize=FS["rank"], color=style.MUTED)
    ax.text(total, 0, L["worse"], ha="right", va="center", fontsize=FS["rank"], color=style.MUTED)
    ax.annotate("", xy=(total - w_w - 3.0, 0), xytext=(w_b + 3.0, 0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=style.AXIS, lw=0.7, mutation_scale=6.0, shrinkA=0,
                                shrinkB=0))
    for c, pct in enumerate(percentiles):
        xc = (c * (W_TILE + COL_GAP) + W_TILE / 2) * 72.0
        name = L["percentile"].get(pct) or L["percentile_other"].format(p=pct)
        ax.text(xc, 0, name, ha="center", va="center", fontsize=FS["col"], fontweight="bold", color=style.INK,
                zorder=3, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="none"))


def _crop_band(img: np.ndarray, aspect_wh: float) -> np.ndarray:
    """Central horizontal band of ``img`` with width / height = ``aspect_wh``."""
    h, w = img.shape[:2]
    bh = max(2, min(h, int(round(w / aspect_wh))))
    y0 = (h - bh) // 2
    return img[y0:y0 + bh]


def draw_legend2(page: fc.Page, y_top: float, L: dict, views: Optional[np.ndarray]) -> None:
    """Second legend line: which surround image the model is given (swatches cut from a real frame)."""
    fig = page.fig
    ax = page.ax(0.0, y_top, FIG_W, LEGEND2_H)
    w_pt, h_pt = FIG_W * 72.0, LEGEND2_H * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis("off")
    y = h_pt * 0.55
    sw, sh = 16.0, 6.4
    x = 1.0
    if views is not None:
        front = _crop_band(views[geo.FRONT], sw / sh)
        back = cd.mute(_crop_band(views[geo.BACK], sw / sh), sat=0.12, white=0.5)
        ax.imshow(front, extent=(x, x + sw, y - sh / 2, y + sh / 2), aspect="auto", interpolation="bilinear",
                  zorder=1)
    else:
        ax.add_patch(Rectangle((x, y - sh / 2), sw, sh, fc="#d6d2c8", ec="none", zorder=1))
        back = None
    ax.add_patch(Rectangle((x, y - sh / 2), sw, sh, fill=False, ec=style.INK, lw=1.0, zorder=2))
    ax.text(x + sw + 3.0, y, L["legend_input"], ha="left", va="center", fontsize=FS["legend"], color=style.INK)
    x += sw + 3.0 + cd.text_width_pt(fig, L["legend_input"], FS["legend"]) + 18.0
    if back is not None:
        ax.imshow(back, extent=(x, x + sw, y - sh / 2, y + sh / 2), aspect="auto", interpolation="bilinear",
                  zorder=1)
    else:
        ax.add_patch(Rectangle((x, y - sh / 2), sw, sh, fc="#e8e6e0", ec="none", zorder=1))
    ax.text(x + sw + 3.0, y, L["legend_display"], ha="left", va="center", fontsize=FS["legend"], color=style.INK)
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def make_gallery_figure(cases_json, dumps_root=None, topdown_root=None, clip_root_override=None,
                        out_stem="gallery", lang: str = "en", tiers: Optional[Sequence[str]] = None,
                        metrics=None) -> dict:
    """Render the gallery from a cases.json (path or dict); returns {"files": [...], "tiles": [...], ...}.

    ``tiers``: the tiers to draw (e.g. ``MAIN_TIERS``); None draws every tier
    with picks.  ``metrics``: metrics.json (dict, file or its directory) for the
    scene counts; default the cases' ``metrics_dir``, else the pre-registered
    counts.
    """
    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    L = LABELS[lang]
    # the case figure's labels (legend line 1, view names, sector letters); its revised wording when it has one
    fcL = fc.labels_for(lang, "revised") if hasattr(fc, "labels_for") else fc.LABELS[lang]
    cases = load_cases(cases_json)
    present, missing, percentiles = gallery_rows(cases, tiers)
    if not present:
        raise ValueError(f"cases.json has no gallery picks for tiers {tiers or TIER_ORDER}")
    n_scenes = scene_counts(cases, metrics)

    grid: List[Tuple[str, dict, List[Tile]]] = []
    for tier, g in present:
        tiles, seen = [], {}
        by_pct = {int(pk["percentile"]): pk for pk in g["picks"]}
        for pct in percentiles:
            pk = by_pct.get(pct)
            if pk is None:
                t = Tile(tier=tier, percentile=pct, pick={}, error="no pick for this percentile")
            else:
                t = prepare_tile(tier, pk, dumps_root, topdown_root, clip_root_override)
                key = pk.get("clip_key")
                if key in seen:
                    t.same_as = seen[key]
                else:
                    seen[key] = pct
            tiles.append(t)
        grid.append((tier, g, tiles))

    height = fig_height(len(grid))
    fig = plt.figure(figsize=(FIG_W, height))
    page = fc.Page(fig, height)
    draw_column_headers(page, percentiles, L)
    for n, (tier, g, tiles) in enumerate(grid):
        y_band = TOP_H + n * (BAND_H + TILE_H + TIER_GAP)
        if n > 0:  # hairline between tiers
            rule = page.ax(0.0, y_band - TIER_GAP / 2, FIG_W, 0.001)
            rule.axis("off")
            rule.axhline(0.5, color=style.GRID, lw=0.6)
        draw_tier_band(page, y_band, tier, g, n_scenes.get(tier), L)
        y = y_band + BAND_H
        draw_row_tags(page, y, L)
        for c, t in enumerate(tiles):
            x = X_GRID + c * (W_TILE + COL_GAP)
            draw_tile(page, x, y, t, L, fcL, first=(n == 0 and c == 0), axis_labels=(n == len(grid) - 1))
    y_leg = TOP_H + len(grid) * (BAND_H + TILE_H) + (len(grid) - 1) * TIER_GAP + AXIS_H
    fc.draw_legend(page, y_leg, ARM, fcL)
    sample = next((t.views for _, _, ts in grid for t in ts if t.views is not None), None)
    draw_legend2(page, y_leg + fc.LEGEND_H, L, sample)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)

    shown = [t for t, _, _ in grid]
    tiers_text = TIERS_TEXT[lang]
    sep = ", " if lang == "en" else "、"
    listed = [tiers_text[t] for t in shown]
    if lang == "en" and len(listed) > 1:
        joined = ", ".join(listed[:-1]) + (", and " if len(listed) > 2 else " and ") + listed[-1]
    else:
        joined = sep.join(listed)
    miss = [t for t in missing if t in TIER_ORDER]
    missing_text = ""
    if miss:
        missing_text = MISSING_TEXT[lang].format(s="s" if len(miss) > 1 else "", tiers=sep.join(miss))
    stretch = STRETCH_TEXT[lang].format(a=HEAT_STRETCH) if HEAT_STRETCH > 1.05 else ""
    caption = CAPTION[lang].format(tiers_text=joined, el=EL_HEAT, stretch=stretch, merge=fc.MERGE_DEG,
                                   miss=fc.MISS_DEG, heldout=HELDOUT_TEXT[lang] if "B" in shown else "",
                                   missing=missing_text)
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption + "\n", encoding="utf-8")
    files.append(cap_path)

    stats = []
    for tier, _, tiles in grid:
        for t in tiles:
            entry = {"tier": tier, "percentile": t.percentile, "clip_key": t.pick.get("clip_key"),
                     "episode_median_err": t.pick.get("vo_bearing_err_median"), "error": t.error,
                     "problems": t.problems}
            if t.row is not None:
                entry.update(row=t.row.index, frame=t.row.frame, frame_rule="median error nearest the episode's",
                             span_deg=round(t.span_deg, 2), prediction=t.row.summary(ARM),
                             floor=t.row.summary("floor"),
                             episode={"median": round(t.episode.median, 3), "hits": t.episode.hits,
                                      "n": t.episode.n, "n_frames": t.episode.n_frames})
            stats.append(entry)
    return {"files": [str(f) for f in files], "tiers": shown, "missing_tiers": miss, "tiles": stats,
            "size_in": (FIG_W, height)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True, help="cases.json written by select_cases.py")
    ap.add_argument("--tiers", default=None, help="comma-separated tiers to draw, e.g. A,B,C (default: all present)")
    ap.add_argument("--dumps-root", default=None, help="dump root <root>/<tier>/<scene>/<clip>.npz "
                                                       "(default: the npz_path recorded in cases.json)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--metrics", default=None, help="metrics.json or its directory (scene counts; default: the "
                                                    "cases' metrics_dir)")
    ap.add_argument("--out", default="gallery", help="output stem (writes .pdf, .png, _caption.txt)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    args = ap.parse_args(argv)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()] if args.tiers else None
    res = make_gallery_figure(args.cases, dumps_root=args.dumps_root, topdown_root=args.topdown_root,
                              clip_root_override=args.clip_root, out_stem=args.out, lang=args.lang, tiers=tiers,
                              metrics=args.metrics)
    for f in res["files"]:
        print(f)
    for s in res["tiles"]:
        print(s)
    print("tiers", res["tiers"], "missing", res["missing_tiers"])
    print("size_in", tuple(round(v, 3) for v in res["size_in"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
