#!/usr/bin/env python3
"""EXP-18 main case figure: the predicted affordance map vs ground truth at key positions of one episode.

Layout (7.0 in wide, double column; one block per key position K1..Kn):

  a  Route           b  Local map, heading up    c  Front (model input) | Right | Back | Left
  +---------+        K1 frame 19                        metrics of the row (right-aligned)
  | top-down|        ( disc )   lane: numbered badges at the true bearings
  |  map    |                   RGB row: the four views unrolled clockwise
  |  route, |                   ground-truth heat row (blue)
  |  K1..Kn |                   predicted heat row (orange) with an x per predicted peak
  +---------+        K2 ...

Encodings (one meaning each; see ``common_draw`` for the shared vocabulary):

* a  Route: muted top-down map (a light plate where there is no floor), the
  whole route (grey line, open circle = start), each key position as a black
  dot + heading arrow + ``K`` badge; tier, scene and episode underneath.
* b  Local map: the map around the robot, rotated heading-up and clipped to a
  disc whose rim IS the bearing ring that c unrolls.  Dashed radii are the
  view seams (+-45 deg, +-135 deg), so the four sectors are the four panels
  of c; letters F/R/B/L mark the sector centres outside the rim (moved
  outward past the badges when one sits there).  The grey line is the route
  so far.  Every past position that is visible in the ground truth gets a
  blue line from the robot through its true position (dot) to the rim and
  its number just outside the rim at that bearing (slots at one spot share a
  badge, "1–6").  Badges only slide along the rim (thin leader); dots never
  move, so distances and order on the map are true.  The curved arrow (first
  block) shows where c starts and that it runs clockwise.  Scale bar per row.
* c  Surround strip, horizontal axis = bearing, square degrees, starting at
  the front view's left edge (+45 deg) and running clockwise F, R, B, L, so
  the front view is one whole panel and reading the rim of b clockwise gives
  the order of c.
  - lane: the same numbered badges as b, dodged sideways, leaders to the true
    bearing; the blue guide line continues through the RGB row.  Slots with
    no ground-truth view (e.g. the current position) get a one-line note.
  - RGB row: only the front view is the model's image input: framed, in
    colour, headed "model input".  Right/back/left are washed out and
    bracketed "images not given to the model"; they are display only.
  - ground-truth affordance map row (blue) and predicted affordance map row
    (orange; the deployed model's output): each slot's map divided by its own
    peak (the prediction also x (1 - P(none))), max over slots, colour linear
    in that value, the same ramp position for both rows.
  - x = predicted peak (joint argmax of heatmaps_gated) at its bearing and
    elevation; peaks within 4 deg of each other share one x, and marks that
    would touch are staggered vertically by 2.6 pt.  A peak that misses its
    true bearing by more than 5 deg is drawn alone with its slot badge.
  - short blue ticks under the prediction row repeat the true bearings.
  - header: bearing error of the predicted peaks (median, max over GT-visible
    slots) and joint PCK@8 (validate.py rule), then the constant "always
    behind" guess (back view, centre pixel) in grey for reference.

Figure policy (user decision, 2026-09-24): the figure shows the affordance map
only.  It never mentions poses, odometry or the pose-source ablation; the
prediction drawn is always the deployed model's (the dump's ``vo`` arm).  The
pose-source split stays in the EXP-18 ledger/report.  Honesty that remains on
the figure: only the front image is marked as model input, the other three
are marked display-only; ground truth blue vs prediction orange, never
swapped; misses are shown, not hidden.

Extensions (``CaseOptions``, used by ``fig_routes``): title line, per-block role
text, a replacement route panel and a panel under it, route legs in the insets,
sector letters kept out of the headers, clamped peaks, merged notes.  With no
options the output is unchanged, pixel for pixel.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_case --dump <clip.npz> [--rows 0,2,8] [--topdown-root DIR]
      [--clip-root DIR] [--lang en|zh] --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import style
from scripts.exp18.topdown.topdown_io import EgoCrop, forward_from_c2w

from matplotlib.patches import Rectangle  # noqa: E402  (matplotlib is configured in cd.setup)

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); every string on the figure comes from here
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "a": "Route",
        "b": "Local map, heading up",
        "views": ("Front · model input", "Right", "Back", "Left"),
        "sectors": ("F", "R", "B", "L"),
        "not_given": "images not given to the model (display only)",
        "frame": "frame {t}",
        "frame_last": "frame {t} (last)",
        "gt_row": "ground truth",
        "pred_row": "prediction",
        "metric_main": "bearing error median {med:.1f}°, max {mx:.1f}°  ·  PCK@8 {hits}/{n}",
        "metric_floor": "always-behind guess {med:.0f}°, {hits}/{n}",
        "sep": "   ·   ",
        "current": "= current position: not visible · predicted P(not visible) = {p:.2f}",
        "not_visible": "not visible in any view · predicted P(not visible) = {p:.2f}",
        "pred_none": "predicted not visible (P = {p:.2f})",
        "not_visible_n": "not visible in any view · predicted P(not visible) = {ps}",
        "pred_none_n": "predicted not visible (P = {ps})",
        "false_pos": "not visible, predicted here",
        "axis": ("0° (heading)", "−90°", "180°", "+90°"),
        "legend_hist": "past position k (1 = oldest)",
        "legend_gt": "ground-truth affordance map",
        "legend_pred": "predicted affordance map",
        "legend_peak": "predicted peak (numbered: miss > 5°)",
        "legend_robot": "robot",
        "scene": "{tier} {scene}\nepisode {ep} · {T} frames",
        "start": "start",
    },
    "zh": {
        "a": "路线",
        "b": "局部地图（朝向朝上）",
        "views": ("前 · 模型输入", "右", "后", "左"),
        "sectors": ("前", "右", "后", "左"),
        "not_given": "模型看不到这三张图（仅作展示）",
        "frame": "第 {t} 帧",
        "frame_last": "第 {t} 帧（末帧）",
        "gt_row": "真值",
        "pred_row": "预测",
        "metric_main": "方位误差 中位 {med:.1f}°，最大 {mx:.1f}°  ·  PCK@8 {hits}/{n}",
        "metric_floor": "恒答正后方 {med:.0f}°，{hits}/{n}",
        "sep": "  ·  ",
        "current": "= 当前位置：不可见 · 预测不可见概率 {p:.2f}",
        "not_visible": "任何视角都不可见 · 预测不可见概率 {p:.2f}",
        "pred_none": "预测为不可见（{p:.2f}）",
        "not_visible_n": "任何视角都不可见 · 预测不可见概率 {ps}",
        "pred_none_n": "预测为不可见（{ps}）",
        "false_pos": "不可见，却预测在此",
        "axis": ("0°（朝向）", "−90°", "180°", "+90°"),
        "legend_hist": "历史位置 k（1 = 最早）",
        "legend_gt": "真值 affordance map",
        "legend_pred": "预测 affordance map",
        "legend_peak": "预测峰值（编号 = 偏差 > 5°）",
        "legend_robot": "机器人",
        "scene": "{tier} {scene}\n第 {ep} 集 · {T} 帧",
        "start": "起点",
    },
}

CAPTION = {
    "en": (
        "Predicted affordance maps at {n} key positions of one episode ({tier} {scene}, episode {ep}; key positions = "
        "first scored frame, frame with the widest ground-truth bearing spread, last frame). (a) Route on the top-down "
        "map. (b) Map around each key position, heading up; its rim is the bearing ring that (c) unrolls clockwise from "
        "the front view's left edge (arrow); dashed radii are the view seams. Blue lines run from the robot through "
        "each past position (dot; 1 = oldest of the 8 queried) to its number on the rim. (c) The surround view on the "
        "same bearings: only the framed front image is given to the model; the right, back and left images are shown "
        "for reference only. Below it, the ground-truth affordance map (blue) and the predicted affordance map "
        "(orange), both over ±8° of elevation around the horizon (the images span ±15°); each slot's map is divided by its own peak (the prediction also multiplied by its predicted "
        "visibility), the maximum over slots is shown, and colour is linear in that value in both rows. Maps stop at "
        "view seams because every label and prediction lives in one 90° view. x: predicted peak of each slot (peaks "
        "within 4° merged; touching marks staggered vertically); a peak more than 5° from its true bearing is "
        "numbered. Blue ticks under the prediction repeat the true bearings. Headers: bearing error of the predicted "
        "peaks over visible slots (median, max) and joint PCK@8, and for reference the constant 'always behind' guess."
    ),
    "zh": (
        "同一集（{tier} {scene}，第 {ep} 集）{n} 个关键位置上的预测 affordance map（关键位置：第一个评分帧、真值方位跨度"
        "最大的帧、末帧）。(a) 俯视图上的路线。(b) 各关键位置的局部地图，朝向朝上；圆周就是 (c) 从前视左缘顺时针展开的"
        "方位环（箭头），虚线半径为视角分界。蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号。"
        "(c) 同一方位轴上的环视：只有加框的前视图是模型输入，右/后/左三张仅作展示。下方为真值 affordance map（蓝）与"
        "预测 affordance map（橙），两行都只显示地平线上下 ±8° 的仰角范围（环视图为 ±15°）；每个槽位的图除以自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值"
        "线性着色。图在视角分界处截断，因为每个标签和预测都只落在一个 90° 视角里。×：各槽位的预测峰值（4° 内合并，相互挨着"
        "的上下错开）；偏离真值方位 5° 以上的单独编号。预测行下方的蓝色短线重复真值方位。行首：可见槽位上预测峰值的方位"
        "误差（中位、最大）与 joint PCK@8，以及作参照的恒答正后方基线。"
    ),
}

# The prediction drawn is always the deployed model's output (dump arm "vo").
ARM = "vo"

# Wording of the revised layout (``CaseOptions.wording="revised"``), laid over LABELS: the reference
# guess's numbers are named, one notation for P(not visible), the current-position note names the past
# frame, and zh uses half-width parentheses (the CJK font's full-width ones print with wide gaps).
LABELS_REVISED: Dict[str, Dict[str, object]] = {
    "en": {
        "metric_floor": "always-behind guess: median {med:.0f}°, PCK@8 {hits}/{n}",
        "current": "= frame {f}, same spot as now: not visible · predicted P(not visible) = {p:.2f}",
        "pred_none": "visible, but predicted P(not visible) = {p:.2f}",
        "pred_none_n": "visible, but predicted P(not visible) = {ps}",
        "legend_peak": "predicted peak (misses > 5° numbered below the row)",
        "misses": "misses",
    },
    "zh": {
        "b": "局部地图(前方朝上)",
        "frame_last": "第 {t} 帧(末帧)",
        "metric_floor": "恒答正后方：中位 {med:.0f}°，PCK@8 {hits}/{n}",
        "current": "= 第 {f} 帧，与当前位置重合：不可见 · 预测不可见概率 {p:.2f}",
        "pred_none": "可见，但预测不可见概率 {p:.2f}",
        "pred_none_n": "可见，但预测不可见概率 {ps}",
        "pred_none_n_sep": "、",
        "axis": ("0°(朝向)", "−90°", "180°", "+90°"),
        "legend_peak": "预测峰值(偏差 > 5° 的在行下方编号)",
        "legend_hist": "历史位置 k(1 = 最早)",
        "not_given": "模型看不到这三张图(仅作展示)",
    },
}


def labels_for(lang: str, wording: str = "approved") -> dict:
    """``LABELS[lang]``, with ``LABELS_REVISED[lang]`` laid over it for ``wording="revised"``."""
    L = dict(LABELS[lang])
    if wording == "revised":
        L.update(LABELS_REVISED[lang])
    return L


# Caption sentences of the revised layout (``case_caption``); the approved layout keeps CAPTION.
CAPTION_PARTS = {
    "en": {
        "head": ("Predicted affordance maps at {n} key positions of one episode ({tier} {scene}, episode {ep}; key "
                 "positions = first scored frame, frame with the widest ground-truth bearing spread, last frame). "),
        "a": "(a) Route on the top-down map. ",
        "b": ("(b) Map around each key position, heading up; its rim is the bearing ring that (c) unrolls clockwise "
              "from the front view's left edge (arrow); dashed radii are the view seams. Blue lines run from the robot "
              "through each past position (dot; 1 = oldest of the 8 queried) to its number on the rim. "),
        "c": ("(c) The surround view on the same bearings: only the framed front image is given to the model; the "
              "right, back and left images are shown for reference only. Below it, the ground-truth affordance map "
              "(blue) and the predicted affordance map (orange), both over ±8° of elevation around the horizon (the "
              "images span ±15°); each slot's map is divided by its own peak (the prediction also multiplied by its "
              "predicted visibility), the maximum over slots is shown, and colour is linear in that value in both "
              "rows. Maps stop at view seams because every label and prediction lives in one 90° view. "),
        "x": "x: predicted peak of each slot (peaks within 4° merged; touching marks staggered vertically). ",
        "x_lane": ("A peak more than 5° from its true bearing is a miss: its slot number sits under the row, joined "
                   "to its x by a line. "),
        "x_inline": "A peak more than 5° from its true bearing is numbered beside its x. ",
        "carets": ("A caret at a row's edge marks a peak beyond the row's elevation range (blue: ground truth; "
                   "black: prediction, whose x is then drawn at the edge). "),
        "carets_pred": ("A predicted peak beyond the row's elevation range is drawn at the row's edge with a "
                        "caret. "),
        "carets_gt": "A blue caret at the ground-truth row's edge marks a ground-truth peak beyond its range. ",
        "ticks": "Blue ticks under the prediction repeat the true bearings. ",
        "notes": ("Notes above the images list past positions that no view shows (the current spot, or out of "
                  "sight) and visible ones the model calls not visible, each with the predicted probability of "
                  "'not visible'. "),
        "headers": ("Headers: bearing error of the predicted peaks over visible slots (median, max) and joint "
                    "PCK@8, and for reference the constant 'always behind' guess (back view, centre pixel)."),
    },
    "zh": {
        "head": ("同一集（{tier} {scene}，第 {ep} 集）{n} 个关键位置上的预测 affordance map（关键位置：第一个评分帧、"
                 "真值方位跨度最大的帧、末帧）。"),
        "a": "(a) 俯视图上的路线。",
        "b": ("(b) 各关键位置的局部地图，前方朝上；圆周就是 (c) 从前视左缘顺时针展开的方位环（箭头），虚线半径为视角分界。"
              "蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号。"),
        "c": ("(c) 同一方位轴上的环视：只有加框的前视图是模型输入，右/后/左三张仅作展示。下方为真值 affordance map（蓝）"
              "与预测 affordance map（橙），两行都只显示地平线上下 ±8° 的仰角范围（环视图为 ±15°）；每个槽位的图除以"
              "自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值线性着色。图在视角分界处截断，因为"
              "每个标签和预测都只落在一个 90° 视角里。"),
        "x": "×：各槽位的预测峰值（4° 内合并，相互挨着的上下错开）。",
        "x_lane": "偏离真值方位 5° 以上的算作偏差：其槽位编号放在该行下方，并用细线连到对应的 ×。",
        "x_inline": "偏离真值方位 5° 以上的在 × 旁编号。",
        "carets": "行边的小三角表示峰值落在该行仰角范围之外（蓝：真值；黑：预测，其 × 画在行边）。",
        "carets_pred": "落在该行仰角范围之外的预测峰值画在行边并加小三角。",
        "carets_gt": "真值行边的蓝色小三角表示真值峰值落在该行范围之外。",
        "ticks": "预测行下方的蓝色短线重复真值方位。",
        "notes": "图像上方的注释列出任何视角都看不到的历史位置（当前所在处或视线之外），以及可见却被模型判为不可见的位置，并给出预测的不可见概率。",
        "headers": "行首：可见槽位上预测峰值的方位误差（中位、最大）与 joint PCK@8，以及作参照的恒答正后方基线（后视中心像素）。",
    },
}


@dataclass
class CaseOptions:
    """Optional extensions of the case layout (used by ``fig_routes``); ``None`` fields change nothing.

    * ``title`` / ``title_note``: a title line (bold) and a muted note after it,
      in a band of ``TITLE_H`` added above the panel names.
    * ``roles``: per key position, text appended to the block header after the
      frame ("frame 83 · after the turnaround").
    * ``route_panel``: ``f(ax, level, dump, recs, L)`` drawn instead of
      ``draw_route_panel`` into panel a.
    * ``route_foot_h`` / ``route_foot``: height (in) reserved at the bottom of
      column a, under the map and its scene note, and ``f(page, x, y_top, w, h)``
      drawing into it.
    * ``split_frame``: frame where the route turned back; insets past it draw
      the route so far as outbound (solid) and return (dashed) legs.
    * ``letters``: where a sector letter displaced by badges goes:
      "outward" (default, past the badges), "slide" (along the badge ring,
      within its own sector; just inside the rim when the arc is full) or
      "inside" (just inside the rim).  The last two keep letters out of the
      block headers when return-leg badges crowd the front.
    * ``clamp_peaks``: a predicted peak above or below the heat row's +-8 deg
      is drawn at the row's edge with a caret pointing to its side (instead of
      far outside the row, over the lane or the header).
    * ``merge_notes``: slots with the same note ("predicted not visible")
      share one note with all their badges, so no note is dropped for lack of
      room.
    * ``scene_text``: the two lines under the map, instead of ``L["scene"]``.
    * ``caption``: caption text written instead of ``CAPTION[lang]``.

    Revised layout (review 2026-09-24; ``CaseOptions.revised()`` turns all of
    these on, each can also be used alone):

    * ``row_labels="fixed"``: the "ground truth" / "prediction" names sit at
      one strip panel for the whole figure (the front view's left end unless
      heat lies there in some block; ``cd.fixed_label_panel``) on a white
      pill, instead of each block's quietest panel.
    * ``miss_lane``: a numbered miss is not labelled inside the prediction
      row; its slot badge goes to a lane under the row (dodged sideways) with
      a thin line to its own x.  Adds ``MISS_LANE_H`` to every block when any
      block has a miss.
    * ``wrap_notes``: notes ("not visible ...") never get dropped: a note
      goes left of the lane badges only with a clear gap to them, else onto
      note lines between the block header and the lane (``NOTE_LINE_H``
      each); nothing is appended to the right of the badge row.
    * ``gt_carets``: a blue caret at the ground-truth row's edge where a
      visible slot's ground-truth peak lies beyond the row's +-8 deg.
    * ``route_fit``: panel a cropped to the route (+ ``ROUTE_PAD_M``), turned
      by a quarter turn when that shows it larger, and only as tall as that
      needs (top-aligned; the rest of column a goes to ``route_foot``); the
      scale bar sits in a band inside the frame.
    * ``scale_corner="fixed"``: the insets' scale bars share one corner (the
      one clear of badges in every block).
    * ``wording="revised"``: ``LABELS_REVISED`` over ``LABELS``.
    * ``banner`` (+ ``banner_italic``): one muted line above the panel names
      (a development stand-in notice), without a title.
    """

    title: Optional[str] = None
    title_note: Optional[str] = None
    roles: Optional[Sequence[Optional[str]]] = None
    route_panel: Optional[Callable] = None
    route_foot_h: float = 0.0
    route_foot: Optional[Callable] = None
    split_frame: Optional[int] = None
    letters: str = "outward"
    clamp_peaks: bool = False
    merge_notes: bool = False
    scene_text: Optional[str] = None
    caption: Optional[str] = None
    row_labels: str = "quietest"
    miss_lane: bool = False
    wrap_notes: bool = False
    gt_carets: bool = False
    route_fit: bool = False
    scale_corner: str = "auto"
    wording: str = "approved"
    banner: Optional[str] = None
    banner_italic: bool = True

    @classmethod
    def revised(cls, **kw) -> "CaseOptions":
        """Every fix of the 2026-09-24 review on (fixed row labels, miss lane, wrapped and merged notes,
        ground-truth and clamped-peak carets, fitted route panel, one inset scale-bar corner, revised wording);
        ``kw`` overrides any field."""
        base = dict(row_labels="fixed", miss_lane=True, wrap_notes=True, merge_notes=True, gt_carets=True,
                    clamp_peaks=True, route_fit=True, scale_corner="fixed", wording="revised")
        base.update(kw)
        return cls(**base)

    @property
    def is_approved_layout(self) -> bool:
        return (self.row_labels == "quietest" and not self.miss_lane and not self.wrap_notes and not self.gt_carets
                and not self.route_fit and self.scale_corner == "auto" and self.wording == "approved")

# --------------------------------------------------------------------------- #
# Geometry of the page (inches)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # 7.0
W_STRIP = 4.54
X_STRIP = FIG_W - 0.02 - W_STRIP
X_ROUTE, W_ROUTE = 0.0, 1.20
X_INSET = 1.28
W_INSET = X_STRIP - 0.13 - X_INSET
EL_RGB = 15.0  # RGB row: elevation +-15 deg
EL_HEAT = 8.0  # heat rows: +-8 deg (history camera centres sit near the horizon)
RGB_H = W_STRIP * 2 * EL_RGB / 360.0  # square degrees
HEAT_H = W_STRIP * 2 * EL_HEAT / 360.0
TOP_H = 0.31
HDR_H = 0.15
LANE_H = 0.14
ROW_GAP = 0.028
BLOCK_GAP = 0.10
AXIS_H = 0.17
LEGEND_H = 0.20
BODY_H = LANE_H + RGB_H + 2 * ROW_GAP + 2 * HEAT_H
BLOCK_H = HDR_H + BODY_H
RGB_RING_W = 1816  # ring columns per 360 deg (multiple of 8: exact roll), ~400 dpi at W_STRIP
HEAT_RING_W = 1440
MISS_DEG = 5.0  # a predicted peak further than this from its true bearing is drawn alone and numbered
MERGE_DEG = 4.0  # other peaks closer than this share one x (about the width of the mark)
MARK_PT = 4.4  # size of the predicted-peak x
STAGGER_PT = 2.6  # vertical offset of an x that would touch its neighbour
FS = {"title": 7.0, "name": 6.6, "header": 6.3, "small": 5.8, "note": 5.7, "legend": 6.1, "axis": 6.0}
# revised layout (CaseOptions)
NOTE_LINE_H = 0.13  # one note line between a block header and its lane (wrap_notes)
MISS_LANE_H = 0.17  # lane under the prediction row for the numbered misses (miss_lane)
MISS_BADGE_PT = 8.3  # badge centre below the row's bottom edge (the true-bearing ticks take the first 4 pt)
NOTE_GAP_PT = 14.0  # a note left of the lane badges keeps at least this far from them (wrap_notes)
CARET_PT = 4.4  # size of an off-row caret
GT_INK = "#1c5cab"  # dark tone of the ground-truth ramp (carets)
ROUTE_PAD_M = 0.45  # route_fit: margin around the route (metres)
BANNER_H = 0.20  # height of the optional banner line


def fig_height(n_blocks: int) -> float:
    return TOP_H + n_blocks * BLOCK_H + (n_blocks - 1) * BLOCK_GAP + AXIS_H + LEGEND_H


TITLE_H = 0.25  # optional title band (CaseOptions.title)
_DROPPED: List[str] = []  # notes the current figure left out (make_case_figure returns them as "notes_dropped")


class Page:
    """Axes placement in inches from the top-left corner (``y0``: extra band above everything)."""

    def __init__(self, fig, height: float, y0: float = 0.0):
        self.fig = fig
        self.h = height
        self.y0 = y0

    def ax(self, x: float, y_top: float, w: float, h: float, **kw):
        y_top = y_top + self.y0
        return self.fig.add_axes([x / FIG_W, 1 - (y_top + h) / self.h, w / FIG_W, h / self.h], **kw)

    def text(self, x: float, y: float, s: str, **kw):
        y = y + self.y0
        return self.fig.text(x / FIG_W, 1 - y / self.h, s, **kw)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_route_panel(ax, level, dump: dd.Dump, rows: Sequence[dd.CaseRow], L: dict) -> None:
    xz = dump.positions[:, [0, 2]]
    limits = cd.fit_limits(xz, pad=0.7, aspect_hw=cd.axes_aspect_hw(ax))
    cd.draw_topdown(ax, level, limits, sat=0.2, white=0.56)
    cd.clean_axes(ax, spines=True)
    cd.draw_route(ax, xz, start_label=L["start"])
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    for n, r in enumerate(rows):
        p = r.cur_pos[[0, 2]]
        f = forward_from_c2w(r.cur_c2w)
        cd.heading_arrow(ax, p, f, 15.0 * per_pt, zorder=6)
        ax.plot(*p, marker="o", ms=3.0, mfc=style.INK, mec="white", mew=0.5, zorder=7)
        # badge beside the route, on the side (left/right of the heading, or behind) with most room
        right = np.array([-f[1], f[0]])
        best, best_score = None, -np.inf
        for side in (right, -right, -f, (right - f) / np.sqrt(2), (-right - f) / np.sqrt(2)):
            q = p + side * 11.0 * per_pt
            inside = (limits[0] + 7 * per_pt < q[0] < limits[1] - 7 * per_pt
                      and limits[2] + 6 * per_pt < q[1] < limits[3] - 6 * per_pt)
            score = np.min(np.linalg.norm(xz - q, axis=1)) - (0 if inside else 1e3)
            if score > best_score:
                best, best_score = q, score
        cd.key_badge(ax, best[0], best[1], f"K{n + 1}", fs=6.0)
    bar = cd.nice_length(0.4 * (limits[1] - limits[0]))
    cd.scale_bar(ax, limits[0] + 5 * per_pt, limits[2] + 6.5 * per_pt, bar, f"{bar:g} m")


def _leader(ax, anchor, centre, w_pt: float, h_pt: float, color=style.INK_2, lw: float = 0.5, zorder: float = 7.5):
    """Thin line from ``anchor`` to the edge of a (w, h)-point box centred at ``centre`` (data coords)."""
    T = ax.transData
    a, c = T.transform(anchor), T.transform(centre)
    d = c - a
    px = ax.figure.dpi / 72.0
    t = min(w_pt * px / 2 / (abs(d[0]) + 1e-9), h_pt * px / 2 / (abs(d[1]) + 1e-9), 1.0)
    end = T.inverted().transform(c - d * t)
    ax.plot([anchor[0], end[0]], [anchor[1], end[1]], color=color, lw=lw, zorder=zorder, solid_capstyle="butt")


class FittedRoutePanel:
    """Panel a cropped to the route: ``CaseOptions.route_fit``, and ``fig_routes`` with the route's two legs.

    The map is a heading-up crop (``cd.route_frame``: a quarter turn when that
    shows the route larger) in local metres, x right and y up.  ``height``
    says how tall the panel needs to be at the column's width (the route +
    ``pad`` metres, plus a band on top for the scale bar and one at the bottom
    for the legend); ``__call__`` draws it (``CaseOptions.route_panel``).

    * ``split``: frame where the route turns back: outbound leg solid, return
      dashed, each on its own right-hand side, chevrons, a diamond at the turn
      (``cd.draw_route_legs``); ``None``: one grey line (the case figure).
    * ``legend``: [(kind, text)] with kind "out", "back", "turn", "start",
      drawn in the bottom band, two columns when they fit.
    * ``start_text``: label the start circle on the map (placed clear of
      everything), for panels without a legend entry for it.
    * K badges sit clear of the route, arrows, markers, the bands and each
      other, preferring the side away from the route's centre; their leaders
      never cross.
    """

    TOP_BAND_PT = 15.0
    LEG_ROW_PT = 7.2
    K_FS = 6.0

    def __init__(self, split: Optional[int] = None, legend: Optional[Sequence[tuple]] = None,
                 start_text: Optional[str] = None, pad: float = ROUTE_PAD_M, min_h: float = 0.9):
        self.split = split
        self.legend = list(legend or [])
        self.start_text = start_text
        self.pad = pad
        self.min_h = min_h
        self._frame = None

    # ---- geometry
    def _bottom_band_pt(self, fig) -> float:
        if not self.legend:
            return 4.0
        return self._legend_rows(fig) * self.LEG_ROW_PT + 5.0

    def _legend_cols(self, fig, w_pt: float):
        widths = [self._item_w(fig, t) for _, t in self.legend]
        if len(self.legend) > 1:
            c1, c2 = widths[0::2], widths[1::2]
            if max(c1) + 6.0 + max(c2) <= w_pt - 7.0:
                return 2, max(c1)
        return 1, max(widths)

    def _legend_rows(self, fig) -> int:
        w_pt = W_ROUTE * 72.0
        ncol, _ = self._legend_cols(fig, w_pt)
        return int(math.ceil(len(self.legend) / ncol))

    @staticmethod
    def _item_w(fig, text: str) -> float:
        return 11.0 + 2.5 + cd.text_width_pt(fig, text, 5.8)

    def height(self, fig, dump: dd.Dump, w_in: float, h_max_in: float) -> float:
        """Panel height (in) that shows the route at the column's width, within [min_h, h_max_in]."""
        world = dump.positions[:, [0, 2]]
        w_pt = w_in * 72.0
        bands = self.TOP_BAND_PT + self._bottom_band_pt(fig)
        avail = max(h_max_in * 72.0 - bands, 10.0)
        centre, fwd = cd.route_frame(world, self.split, avail / w_pt, self.pad)
        self._frame = (centre, fwd)
        crop = EgoCrop(centre, fwd, 1.0, 2)
        a, b = crop.world_to_local(world[:, 0], world[:, 1])
        pw, ph = np.ptp(a) + 2 * self.pad, np.ptp(b) + 2 * self.pad
        scale = min(w_pt / pw, avail / ph)
        need = (ph * scale + bands) / 72.0
        return float(np.clip(need, min(self.min_h, h_max_in), h_max_in))

    # ---- drawing
    def __call__(self, ax, level, dump: dd.Dump, recs: Sequence[dd.CaseRow], L: dict) -> None:
        fig = ax.figure
        world = dump.positions[:, [0, 2]]
        box = ax.get_position()
        w_pt = box.width * fig.get_size_inches()[0] * 72.0
        h_pt = box.height * fig.get_size_inches()[1] * 72.0
        top_pt, bot_pt = self.TOP_BAND_PT, self._bottom_band_pt(fig)
        aspect = (h_pt - top_pt - bot_pt) / w_pt
        if self._frame is None:
            self._frame = cd.route_frame(world, self.split, aspect, self.pad)
        centre, fwd = self._frame
        frame = EgoCrop(centre, fwd, 1.0, 2)

        def local(x, z):
            a, b = frame.world_to_local(x, z)
            return np.stack([np.asarray(a, dtype=float), np.asarray(b, dtype=float)], axis=-1)

        xy = local(world[:, 0], world[:, 1])
        x0, x1, y0, y1 = cd.fit_limits(xy, pad=self.pad, aspect_hw=aspect)
        m_per_pt = (x1 - x0) / w_pt
        limits = (x0, x1, y0 - bot_pt * m_per_pt, y1 + top_pt * m_per_pt)
        half = 1.01 * max(abs(v) for v in limits)
        img = (cd.mute_map(level.rgb(), sat=0.2, white=0.56) * 255).astype(np.uint8)
        plate = tuple(int(round(255 * c)) for c in cd.MAP_PLATE)
        n_px = int(np.clip(2 * half / level.mpp, 400, 2600))
        crop = level.heading_up_crop(img, centre, forward_xz=fwd, half_size_m=half, out_px=n_px, fill=plate)
        ax.imshow(crop.image, extent=crop.extent, interpolation="bilinear", zorder=0)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_autoscale_on(False)
        ax.set_facecolor(cd.MAP_PLATE)
        cd.clean_axes(ax, spines=True)
        per_pt = cd.pts_to_data(ax, 1.0)[0]
        if self.split is not None:
            legs = cd.draw_route_legs(ax, xy, self.split, lw=0.95, offset_pt=1.6)
            lines = [v for v in legs.values() if v is not None]
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=style.INK_2, lw=1.0, solid_capstyle="round", solid_joinstyle="round",
                    zorder=3)
            lines = []
        ax.plot(*xy[0], marker="o", ms=4.2, mfc="white", mec=style.INK_2, mew=0.9, zorder=6)
        if self.split is not None:  # under the K dots: a K position at the turn shows as a dot inside the diamond
            cd.turn_marker(ax, *xy[self.split], size=5.4, zorder=7.2)
        tips, anchors = [], []
        for r in recs:
            p = local(r.cur_pos[0], r.cur_pos[2])
            fw = forward_from_c2w(r.cur_c2w)
            f = local(centre[0] + fw[0], centre[1] + fw[1])
            cd.heading_arrow(ax, p, f, 13.0 * per_pt, zorder=6)
            ax.plot(*p, marker="o", ms=3.0, mfc=style.INK, mec="white", mew=0.5, zorder=7.4)
            fn = f / (np.linalg.norm(f) + 1e-12)
            side = np.array([-fn[1], fn[0]])
            tips.append(np.concatenate([p + fn * per_pt * np.linspace(3.0, 15.0, 7)[:, None],
                                        (p + fn * 11.0 * per_pt + side * 2.2 * per_pt)[None],
                                        (p + fn * 11.0 * per_pt - side * 2.2 * per_pt)[None]]))
            anchors.append(p)
        # the scale bar takes the top band, the legend the bottom band; the labels keep clear of both
        obstacles = np.concatenate([xy] + lines + tips + ([xy[self.split][None]] if self.split is not None else []),
                                   axis=0)
        bar = cd.nice_length(0.4 * (limits[1] - limits[0]))
        bar_box = (bar / per_pt + 2.0, 11.0)
        bx = limits[0] + (4.0 + bar_box[0] / 2) * per_pt
        by = limits[3] - (2.5 + bar_box[1] / 2) * per_pt
        fixed = [(bx, by, *bar_box)]
        if self.legend:
            lw_pt = w_pt - 6.0
            fixed.append(((limits[0] + limits[1]) / 2, limits[2] + (bot_pt / 2) * per_pt, lw_pt, bot_pt - 1.0))
        k_boxes = [(cd.text_width_pt(fig, f"K{n + 1}", self.K_FS, fontweight="bold") + 3.2, 8.6)
                   for n in range(len(recs))]
        boxes = list(k_boxes)
        if self.start_text:
            anchors.append(xy[0])
            boxes.append((cd.text_width_pt(fig, self.start_text, 5.8) + 1.5, 6.6))
        centroid = xy.mean(0)
        order_a = list(range(len(anchors)))
        order_b = order_a[len(recs):] + order_a[:len(recs)]
        tries = [cd.place_near(ax, anchors, obstacles, limits, boxes, fixed_boxes=fixed, order=o, clear_pt=6.0,
                               radii_pt=(2.5, 4.5, 7.0, 10.0, 14.0, 19.0, 25.0, 32.0), n_dir=24,
                               away_from=centroid, away_weight=1.2)
                 for o in (order_a, order_b)]
        placed = max(tries, key=lambda pl: (min(q[3] for q in pl), sum(q[3] for q in pl)))  # worst label first
        nk = len(recs)
        perm = cd.uncross([anchors[i] for i in range(nk)], [placed[i][:2] for i in range(nk)])
        for n in range(nk):
            x, y = placed[perm[n]][:2]
            gap_pt = float(np.hypot(*((np.asarray([x, y]) - anchors[n]) / per_pt)))
            if gap_pt > 0.5 * max(k_boxes[n]) + 3.0:
                _leader(ax, anchors[n], (x, y), *k_boxes[n])
            cd.key_badge(ax, x, y, f"K{n + 1}", fs=self.K_FS)
        if self.start_text:
            x, y, gap, _ = placed[nk]
            if gap > 3.0:
                _leader(ax, xy[0], (x, y), *boxes[nk], color=style.MUTED, lw=0.45, zorder=5.5)
            ax.text(x, y, self.start_text, ha="center", va="center", fontsize=5.8, color=style.INK_2,
                    path_effects=cd.HALO, zorder=8)
        cd.scale_bar(ax, bx - (bar_box[0] / 2 - 1.0) * per_pt, by - 3.0 * per_pt, bar, f"{bar:g} m")
        if self.legend:
            self._draw_legend(ax, limits, per_pt, w_pt, bot_pt)

    def _draw_legend(self, ax, limits, per_pt: float, w_pt: float, bot_pt: float) -> None:
        fig = ax.figure
        ncol, c1 = self._legend_cols(fig, w_pt)
        x_left = limits[0] + 4.0 * per_pt
        y_top = limits[2] + (bot_pt - 2.5 - self.LEG_ROW_PT / 2) * per_pt
        for i, (kind, text) in enumerate(self.legend):
            row, col = (i // ncol, i % ncol)
            x = x_left + (col * (c1 + 6.0)) * per_pt
            y = y_top - row * self.LEG_ROW_PT * per_pt
            xs = np.array([x, x + 11.0 * per_pt])
            if kind in ("out", "back"):
                ls = cd.LEG_OUT_LS if kind == "out" else cd.LEG_BACK_LS
                ax.plot(xs, [y, y], color=cd.LEG_COLOR, lw=0.95, ls=ls, zorder=8, solid_capstyle="butt",
                        dash_capstyle="butt", path_effects=cd.HALO_THIN)
                cd.chevrons(ax, np.stack([xs, [y, y]], axis=1), spacing_pt=40.0, end_pt=0.0, zorder=8.2)
            elif kind == "turn":
                cd.turn_marker(ax, xs.mean(), y, size=5.0, zorder=8.2)
            elif kind == "start":
                ax.plot(xs.mean(), y, marker="o", ms=4.2, mfc="white", mec=style.INK_2, mew=0.9, zorder=8.2)
            ax.text(xs[1] + 2.5 * per_pt, y, text, ha="left", va="center", fontsize=5.8, color=style.INK_2,
                    path_effects=cd.HALO, zorder=8)


INSET_CORNERS = {135.0: (-1, 1), 45.0: (1, 1), 225.0: (-1, -1), 315.0: (1, -1)}  # plot angle -> signs
ARROW_MID = 90.0 + cd.STRIP_START_DEG - 17.0  # plot angle of the middle of the first block's direction arrow


def inset_half(r: dd.CaseRow) -> float:
    """Disc radius (metres) of panel b: 1.12 x the farthest visible past position, at least 1 m."""
    far = float(np.max(r.gt_dist[r.visible])) if r.visible.any() else 1.0
    return max(1.0, 1.12 * far)


def inset_corner_clearance(ax, r: dd.CaseRow, show_arrow: bool) -> Dict[float, float]:
    """Per free corner of the inset square (plot angle 45/135/225/315): angular clearance to the rim badges.

    The same badge layout ``draw_inset`` will draw (nothing is drawn here); the
    top-left corner is not free when the direction arrow is drawn there.
    """
    half = inset_half(r)
    lim = half / 0.74
    ax.set_xlim(-lim, lim)  # draw_local_disc's limits, so the point scale is the drawn one
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    bearings = [float(r.gt_bearing[g[0]]) for g in r.groups]
    occupied = []
    if r.groups:
        _, _, placed, widths = cd.disc_rim_layout(ax, half, r.groups, bearings)
        occupied = [(float(p % 360.0), float(w / 2)) for p, w in zip(placed, widths)]
    corners = dict(INSET_CORNERS)
    if show_arrow and all(abs((t - ARROW_MID + 180) % 360 - 180) > 17.0 + w + 4.0 for t, w in occupied):
        corners.pop(135.0)
    return {c: min([abs((c - t + 180) % 360 - 180) for t, _ in occupied] + [360.0]) for c in corners}


def draw_inset(ax, level, dump: dd.Dump, r: dd.CaseRow, show_arrow: bool, L: dict,
               split_frame: Optional[int] = None, letters: str = "outward",
               scale_corner: Optional[float] = None) -> None:
    """Panel b for one key position (``split_frame``/``letters_inside``: see :class:`CaseOptions`).

    ``scale_corner``: plot angle (45, 135, 225, 315) of the corner for the
    scale bar; default the free corner farthest from every badge.
    """
    fwd = forward_from_c2w(r.cur_c2w)
    half = inset_half(r)
    past = dump.positions[: r.frame + 1][:, [0, 2]]
    if split_frame is not None:
        crop = cd.draw_local_disc(ax, level, r.cur_pos[[0, 2]], fwd, half, past_xz=past,
                                  past_split=split_frame if split_frame < r.frame else len(past) - 1)
    else:
        crop = cd.draw_local_disc(ax, level, r.cur_pos[[0, 2]], fwd, half, past_xz=past)
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    bearings = []
    for g in r.groups:
        k = g[0]
        a, b = crop.world_to_local(r.hist_pos[k, 0], r.hist_pos[k, 2])
        ang = math.atan2(b, a)
        ax.plot([0, half * math.cos(ang)], [0, half * math.sin(ang)], color=cd.history_line_color(k), lw=0.55,
                zorder=3.5, solid_capstyle="butt")
        ax.plot([a], [b], marker="o", ms=3.3, mfc=style.history_color(k), mec="white", mew=0.5, zorder=4.5)
        bearings.append(float(r.gt_bearing[k]))
    cd.robot_glyph(ax, 0.0, 0.0, (0.0, 1.0), size_pt=6.6, zorder=6)
    occupied = cd.disc_rim_labels(ax, half, r.groups, bearings)
    if letters != "outward":  # the arrow's arc (if drawn) is taken too, so a sliding letter avoids it
        mid = 90.0 + cd.STRIP_START_DEG - 17.0
        arrow = show_arrow and all(abs((t - mid + 180) % 360 - 180) > 17.0 + w + 4.0 for t, w in occupied)
        cd.disc_sector_letters(ax, half, list(occupied) + ([(mid, 19.0)] if arrow else []), names=L["sectors"],
                               displaced=letters, rays=[90.0 + b for b in bearings])
    else:
        cd.disc_sector_letters(ax, half, occupied, names=L["sectors"])
    corners = {135.0: (-1, 1), 45.0: (1, 1), 225.0: (-1, -1), 315.0: (1, -1)}  # plot angle -> signs
    if show_arrow:  # the arrow sits outside the rim from the strip start (+45 deg bearing) clockwise
        mid = 90.0 + cd.STRIP_START_DEG - 17.0
        if all(abs((t - mid + 180) % 360 - 180) > 17.0 + w + 4.0 for t, w in occupied):
            cd.disc_direction_arrow(ax, half)
            corners.pop(135.0)
    # scale bar in the square's free corner (outside the disc) farthest from every badge
    lim = ax.get_xlim()[1]

    def clearance(theta):
        return min([abs((theta - t + 180) % 360 - 180) for t, _ in occupied] + [360.0])

    if scale_corner is not None and scale_corner in corners:
        sx, sy = corners[scale_corner]
    else:
        sx, sy = corners[max(corners, key=clearance)]
    bar = cd.nice_length(0.6 * half)
    x = sx * (lim - 1.0 * per_pt)
    y = sy * (lim - (8.5 if sy > 0 else 2.5) * per_pt)
    cd.scale_bar(ax, x, y, bar, f"{bar:g} m", fs=5.6, ha="left" if sx < 0 else "right")


def _metrics_text(r: dd.CaseRow, arm: str, L: dict):
    """(main, rest): the prediction's numbers, then the always-behind guess for reference."""
    s = r.summary(arm)
    main = L["metric_main"].format(med=s["median"], mx=s["max"], hits=s["hits"], n=s["n"])
    sf = r.summary("floor")
    return main, L["sep"] + L["metric_floor"].format(med=sf["median"], hits=sf["hits"], n=sf["n"])


def _place_peak_labels(ax, fig, strip: np.ndarray, marks, pt_per_deg: float, texts: Dict[int, str]) -> None:
    """Slot badge (+ text) beside each numbered x, clear of every other x and label.

    ``marks``: (x, y, slot or None) of all drawn x marks.  A label goes on the
    side with more free room (less heat on a tie); when that side is taken it
    slides further out, joined to its x by a thin leader.
    """
    half_mark = MARK_PT / 2 + 0.8
    taken = [(x * pt_per_deg - half_mark, x * pt_per_deg + half_mark) for x, _, _ in marks]
    q = strip.shape[1] / 360.0

    def heat(a_pt, b_pt):
        a, b = a_pt / pt_per_deg, b_pt / pt_per_deg
        if a < 0 or b > 360:
            return np.inf
        return float(strip[:, int(a * q):max(int(b * q), int(a * q) + 1)].sum())

    def overlap(a, b):
        return sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in taken)

    for x, y, k in sorted([m for m in marks if m[2] is not None], key=lambda m: m[0]):
        text = texts.get(k, "")
        bw = cd.badge_width_pt(str(k + 1))
        tw = cd.text_width_pt(fig, text, FS["small"]) + 1.6 if text else 0.0
        width = bw + tw
        xp = x * pt_per_deg
        best = None
        for shift in np.arange(0.0, 60.0, 1.5):
            for sgn in (1.0, -1.0):
                a = xp + sgn * (half_mark + 0.6 + shift)
                lo, hi = (a, a + width) if sgn > 0 else (a - width, a)
                if lo < 0 or hi > 360 * pt_per_deg:
                    continue
                cost = (overlap(lo, hi), heat(lo, hi))
                if best is None or cost < best[0]:
                    best = (cost, sgn, lo, hi, shift)
            if best is not None and best[0][0] == 0.0:
                break
        (_, sgn, lo, hi, shift) = best
        cx = (lo + bw / 2) if sgn > 0 else (hi - bw / 2)
        if shift > 0:
            edge = xp + sgn * half_mark
            ax.plot([edge / pt_per_deg, (cx - sgn * bw / 2) / pt_per_deg], [y, y], color=style.INK_2, lw=0.5,
                    zorder=6.5)
        cd.history_badge(ax, cx / pt_per_deg, y, str(k + 1), k, zorder=8)
        if text:
            tx = cx + sgn * (bw / 2 + 1.6)
            ax.text(tx / pt_per_deg, y, text, ha="left" if sgn > 0 else "right", va="center",
                    fontsize=FS["small"], color=style.INK, zorder=8, path_effects=cd.HALO)
        taken.append((lo, hi))


@dataclass
class BlockExt:
    """Revised-layout settings of one block (``make_case_figure`` fills it from :class:`CaseOptions`)."""

    note_lines: List[list]  # wrap_notes: note lines between the header and the lane ([(x deg, slots, text)])
    lane_notes: List[tuple]  # wrap_notes: notes left of the lane badges ([(x deg, slots, text)])
    wrap_notes: bool = False
    miss_lane: bool = False  # reserve MISS_LANE_H under the prediction row and number misses there
    label_panel: Optional[int] = None  # row_labels="fixed": strip panel of the row names
    gt_carets: bool = False

    @property
    def extra_h(self) -> float:
        return len(self.note_lines) * NOTE_LINE_H + (MISS_LANE_H if self.miss_lane else 0.0)


def lane_layout(r: dd.CaseRow):
    """(targets, centres, labels, spans) of the ground-truth badges in a block's lane (strip degrees)."""
    ppd = W_STRIP * 72.0 / 360.0
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in r.groups])
    labels = [dd.group_label(g) for g in r.groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / ppd for s in labels], 0.0, 360.0, 1.0 / ppd)
    spans = [(x - cd.badge_width_pt(s) / ppd / 2, x + cd.badge_width_pt(s) / ppd / 2) for x, s in zip(xs, labels)]
    return targets, xs, labels, spans


def note_items(r: dd.CaseRow, p, L: dict, merge: bool) -> List[tuple]:
    """Notes of one block as (slots, text): no ground-truth view (the current spot, or out of sight), and
    visible slots the prediction calls not visible (P(not visible) > 0.5); ``merge``: one note per kind."""
    current = [k for k in r.invisible_slots() if r.gt_dist[k] < 0.1]
    unseen = [k for k in r.invisible_slots() if r.gt_dist[k] >= 0.1]
    pred_none = [k for k in range(dd.K) if r.visible[k] and p.none_p[k] > 0.5]
    out = [([k], L["current"].format(p=p.none_p[k], f=int(r.hist_frames[k]))) for k in current]
    sep = L.get("pred_none_n_sep", ", ")
    for ks, key in ((unseen, "not_visible"), (pred_none, "pred_none")):
        if merge and len(ks) > 1:
            out.append((ks, L[key + "_n"].format(ps=sep.join(f"{p.none_p[k]:.2f}" for k in ks))))
        else:
            out += [([k], L[key].format(p=p.none_p[k])) for k in ks]
    return out


def _note_width_pt(fig, ks, text: str) -> float:
    return sum(cd.badge_width_pt(str(k + 1)) for k in ks) + 1.0 * (len(ks) - 1) + 1.2 + cd.text_width_pt(
        fig, text, FS["note"])


def layout_notes(fig, items: Sequence[tuple], lane_spans: Sequence[tuple]):
    """``wrap_notes``: (lane_notes, note_lines) for a block's notes; every note is placed.

    Notes go left of the lane badges while they fit with ``NOTE_GAP_PT`` to
    spare (never to the right of the badge row, where their badges would read
    as more past positions); the rest flow onto full-width note lines.
    """
    ppd = W_STRIP * 72.0 / 360.0
    left_hi = min([a for a, _ in lane_spans], default=360.0 + NOTE_GAP_PT / ppd) - NOTE_GAP_PT / ppd
    lane, lines, cur, x, cx = [], [], [], 1.0, 1.0
    rest = False
    for ks, text in items:
        w = _note_width_pt(fig, ks, text) / ppd
        if not rest and x + w <= left_hi:
            lane.append((x, ks, text))
            x += w + 8.0 / ppd
            continue
        rest = True
        if cur and cx + w > 359.0:
            lines.append(cur)
            cur, cx = [], 1.0
        cur.append((cx, ks, text))
        cx += w + 10.0 / ppd
    if cur:
        lines.append(cur)
    return lane, lines


def _draw_note(ax, fig, x: float, ks, text: str, y: float) -> None:
    ppd = W_STRIP * 72.0 / 360.0
    for k in ks:
        bw = cd.badge_width_pt(str(k + 1)) / ppd
        cd.history_badge(ax, x + bw / 2, y, str(k + 1), k)
        x += bw + 1.0 / ppd
    ax.text(x - 1.0 / ppd + 1.2 / ppd, y, text, ha="left", va="center", fontsize=FS["note"], color=style.INK_2)


def _draw_miss_lane(ax_pr, fig, drawn, texts: Dict[int, str], ppd: float) -> None:
    """``miss_lane``: each numbered miss's badge (+ text) in the lane under the row, a line to its own x."""
    misses = sorted([m for m in drawn if m[2] is not None], key=lambda m: m[0])
    if not misses:
        return
    bw = [cd.badge_width_pt(str(k + 1)) for _, _, k in misses]
    tw = [(cd.text_width_pt(fig, texts[k], FS["small"]) + 1.6) if texts.get(k) else 0.0 for _, _, k in misses]
    widths = [(b + t) / ppd for b, t in zip(bw, tw)]
    targets = [x - b / 2 / ppd + w / 2 for (x, _, _), b, w in zip(misses, bw, widths)]
    centres = cd.dodge_1d(np.asarray(targets), widths, 0.0, 360.0, 1.6 / ppd)
    y_b = -EL_HEAT - MISS_BADGE_PT / ppd
    for (x, y, k), b, w, c in zip(misses, bw, widths, centres):
        bx = c - w / 2 + b / 2 / ppd
        ax_pr.plot([bx, x], [y_b + 3.6 / ppd, y], color=style.INK_2, lw=0.55, zorder=6.6, clip_on=False,
                   solid_capstyle="butt", path_effects=cd.HALO_THIN)
        cd.history_badge(ax_pr, bx, y_b, str(k + 1), k, zorder=8).set_clip_on(False)
        if texts.get(k):
            t = ax_pr.text(bx + (b / 2 + 1.6) / ppd, y_b, texts[k], ha="left", va="center", fontsize=FS["small"],
                           color=style.INK, zorder=8, path_effects=cd.HALO)
            t.set_clip_on(False)


def draw_block(page: Page, y_top: float, n: int, r: dd.CaseRow, views: np.ndarray, arm: str, L: dict,
               last: bool, role: Optional[str] = None, clamp_peaks: bool = False, merge_notes: bool = False,
               ext: Optional[BlockExt] = None):
    """One key position: header, lane, RGB row, both heat rows (options: see :class:`CaseOptions`).

    ``ext`` (:class:`BlockExt`): the revised layout's per-block settings; ``None`` = the approved block.
    """
    fig = page.fig
    # ---- header: K badge + frame (inset column), metrics (right-aligned over the strip)
    y_mid = y_top + HDR_H * 0.45
    page.text(X_INSET, y_mid, f"K{n + 1}", ha="left", va="center", fontsize=FS["header"], fontweight="bold",
              color="white", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.3", fc=style.INK, ec="none"))
    frame_txt = (L["frame_last"] if r.is_final else L["frame"]).format(t=r.frame)
    t_frame = page.text(X_INSET + 0.25, y_mid, frame_txt, ha="left", va="center", fontsize=FS["header"],
                        color=style.INK)
    if role:  # "· after the turnaround", bold, right after the frame
        w_frame = t_frame.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        page.text(X_INSET + 0.25 + w_frame, y_mid, L.get("role_sep", " · ") + role, ha="left", va="center",
                  fontsize=FS["header"], color=style.INK, fontweight="bold")
    main, rest = _metrics_text(r, arm, L)
    t_rest = page.text(X_STRIP + W_STRIP, y_mid, rest, ha="right", va="center", fontsize=FS["header"],
                       color=style.MUTED)
    w_rest = t_rest.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    page.text(X_STRIP + W_STRIP - w_rest, y_mid, main, ha="right", va="center", fontsize=FS["header"],
              color=style.INK)

    # ---- axes
    y_lane = y_top + HDR_H + (len(ext.note_lines) * NOTE_LINE_H if ext is not None else 0.0)
    y_rgb = y_lane + LANE_H
    y_gt = y_rgb + RGB_H + ROW_GAP
    y_pr = y_gt + HEAT_H + ROW_GAP
    ax_lane = page.ax(X_STRIP, y_lane, W_STRIP, LANE_H)
    ax_rgb = page.ax(X_STRIP, y_rgb, W_STRIP, RGB_H)
    ax_gt = page.ax(X_STRIP, y_gt, W_STRIP, HEAT_H)
    ax_pr = page.ax(X_STRIP, y_pr, W_STRIP, HEAT_H)
    side = min(W_INSET, BODY_H)
    ax_in = page.ax(X_INSET + (W_INSET - side) / 2, y_lane + (BODY_H - side) / 2, side, side)

    cd.draw_rgb_row(ax_rgb, cd.rgb_strip(views, RGB_RING_W, EL_RGB), EL_RGB)
    gt_strip = cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, EL_HEAT)
    pr_strip = cd.heat_strip(dd.pred_composite(r, arm), HEAT_RING_W, EL_HEAT)
    cd.draw_heat_row(ax_gt, gt_strip, EL_HEAT, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, pr_strip, EL_HEAT, cd.PRED_CMAP)
    if ext is not None and ext.label_panel is not None:  # one place for the whole figure, on a pill
        for ax, label in ((ax_gt, L["gt_row"]), (ax_pr, L["pred_row"])):
            cd.row_label(ax, ext.label_panel * 90 + 2.0, label, FS["small"])
    else:
        v_label = cd.quietest_panel(gt_strip, pr_strip)
        for ax, label in ((ax_gt, L["gt_row"]), (ax_pr, L["pred_row"])):
            ax.text(v_label * 90 + 2.0, 0, label, ha="left", va="center", fontsize=FS["small"], color=style.INK_2,
                    zorder=6, path_effects=cd.HALO)
    if ext is not None and ext.gt_carets and r.gt_peak_elev is not None:
        _draw_gt_carets(ax_gt, r)
    if ext is not None:  # note lines between the header and the lane
        for i, line in enumerate(ext.note_lines):
            ax_note = page.ax(X_STRIP, y_top + HDR_H + i * NOTE_LINE_H, W_STRIP, NOTE_LINE_H)
            ax_note.set_xlim(0, 360)
            ax_note.set_ylim(0, 1)
            ax_note.axis("off")
            for x, ks, text in line:
                _draw_note(ax_note, fig, x, ks, text, 0.5)
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")

    # ---- ground truth: numbered badges in the lane, guide through the RGB row, ticks under the prediction
    pt_per_deg = W_STRIP * 72.0 / 360.0
    groups = r.groups
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in groups])
    labels = [dd.group_label(g) for g in groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / pt_per_deg for s in labels], 0.0, 360.0, 1.0 / pt_per_deg)
    y_badge = 0.56
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    for g, t, x, lab in zip(groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([x, x, t, t], [y_badge, 0.34, 0.12, 0.0], color=col, lw=0.55, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, x, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.55, zorder=3)
        ax_pr.plot([t, t], [-EL_HEAT - 1.0 * tick, -EL_HEAT - 4.0 * tick], color=col, lw=0.8, zorder=3,
                   clip_on=False, solid_capstyle="butt")
    lane_spans = [(x - cd.badge_width_pt(s) / pt_per_deg / 2, x + cd.badge_width_pt(s) / pt_per_deg / 2)
                  for x, s in zip(xs, labels)]

    # ---- predicted peaks: one x per cluster; misses > MISS_DEG alone, with the slot's badge.  Marks
    #      closer than their own width are staggered vertically (+-STAGGER_PT) so none fuse into a blob.
    p = r.arms[arm]
    shown = [k for k in range(dd.K) if r.valid[k] and p.none_p[k] <= 0.5 and p.peak_view[k] >= 0]
    alone = [k for k in shown if (not r.visible[k]) or p.err[k] > MISS_DEG]
    merged = [k for k in shown if k not in alone]
    px = {k: float(cd.strip_x(p.peak_bearing[k])) for k in shown}
    marks = []  # (x, elevation, slot needing a label or None)
    for cl in cd.cluster_1d([px[k] for k in merged], MERGE_DEG):
        ks = [merged[i] for i in cl]
        marks.append((float(np.mean([px[k] for k in ks])), float(np.mean([p.peak_elev[k] for k in ks])), None))
    marks += [(px[k], float(p.peak_elev[k]), k) for k in alone]
    marks.sort(key=lambda m: m[0])
    per_pt_y = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    offsets, sign = [0.0] * len(marks), 1.0
    for j in range(1, len(marks)):
        if (marks[j][0] - marks[j - 1][0]) * pt_per_deg < MARK_PT + 3.4:  # x plus its halo
            if offsets[j - 1] == 0.0:
                offsets[j - 1] = sign * STAGGER_PT
            offsets[j] = -np.sign(offsets[j - 1]) * STAGGER_PT
            sign = -sign
    drawn = []
    marks_y = []
    for (x, el, k), off in zip(marks, offsets):
        if clamp_peaks:  # keep the x inside the row; a caret at the edge says the peak lies beyond it
            if abs(el) > EL_HEAT:  # caret tip at the edge, the x just inside it (clear of the caret)
                s = 1.0 if el > 0 else -1.0
                y = s * (EL_HEAT - (CARET_PT + 3.0 + abs(off)) * per_pt_y)  # a stagger only moves it inward
                ax_pr.plot([x], [s * (EL_HEAT - (CARET_PT / 2 + 0.4) * per_pt_y)], marker="^" if s > 0 else "v",
                           ms=CARET_PT, color=style.INK, mec="white", mew=0.5, zorder=7.5)
            else:
                edge = EL_HEAT - (MARK_PT / 2 + 1.0) * per_pt_y
                y = float(np.clip(el + off * per_pt_y, -edge, edge))
            cd.peak_mark(ax_pr, x, y, size=MARK_PT)
        else:
            y = el + off * per_pt_y
            cd.peak_mark(ax_pr, x, y, size=MARK_PT)
        drawn.append((x, 0.0, k))
        marks_y.append((x, float(y), k))
    texts = {}
    for k in alone:
        if not r.visible[k]:
            texts[k] = L["false_pos"]
    if ext is not None and ext.miss_lane:
        _draw_miss_lane(ax_pr, fig, marks_y, texts, pt_per_deg)
    else:
        _place_peak_labels(ax_pr, fig, pr_strip, drawn, pt_per_deg, texts)

    # ---- notes (slot badge + text) for slots without a GT view or predicted "not visible"
    notes = []
    for k in r.invisible_slots():
        notes.append((k, L["current" if r.gt_dist[k] < 0.1 else "not_visible"].format(p=p.none_p[k],
                                                                                     f=int(r.hist_frames[k]))))
    for k in range(dd.K):
        if r.visible[k] and p.none_p[k] > 0.5:
            notes.append((k, L["pred_none"].format(p=p.none_p[k])))
    # notes go left of the lane badges, else right of them; a note that fits nowhere is reported, not drawn
    free = [(1.0, min([a for a, _ in lane_spans], default=360.0) - 2.0),
            (max([b for _, b in lane_spans], default=0.0) + 2.0, 359.0)]
    if ext is not None and ext.wrap_notes:  # laid out beforehand (layout_notes): nothing is dropped
        for x, ks, text in ext.lane_notes:
            _draw_note(ax_lane, fig, x, ks, text, y_badge)
        notes = []
    elif merge_notes:
        _draw_merged_notes(ax_lane, fig, r, p, L, free, y_badge, pt_per_deg)
        notes = []
    for k, text in notes:
        bw = cd.badge_width_pt(str(k + 1)) / pt_per_deg
        width = bw + (1.2 + cd.text_width_pt(fig, text, FS["note"])) / pt_per_deg
        for j, (lo, hi) in enumerate(free):
            if hi - lo >= width:
                cd.history_badge(ax_lane, lo + bw / 2, y_badge, str(k + 1), k)
                ax_lane.text(lo + bw + 1.2 / pt_per_deg, y_badge, text, ha="left", va="center",
                             fontsize=FS["note"], color=style.INK_2)
                free[j] = (lo + width + 8.0 / pt_per_deg, hi)
                break
        else:
            print(f"[fig_case] frame {r.frame}: no room for the note on slot {k + 1}: {text}")
            _DROPPED.append(f"frame {r.frame}, slot {k + 1}: {text}")

    if last:
        cd.azimuth_axis(ax_pr, L["axis"], fs=FS["axis"])
        lane_pt = MISS_LANE_H * 72.0 if (ext is not None and ext.miss_lane) else 0.0
        ax_pr.tick_params(axis="x", which="major", pad=5.5 + lane_pt)
        if lane_pt:
            ax_pr.tick_params(axis="x", which="minor", length=2.4 + lane_pt)
    return ax_in


def _draw_gt_carets(ax_gt, r: dd.CaseRow) -> None:
    """``gt_carets``: blue caret at the ground-truth row's edge where a visible slot's peak lies beyond it."""
    per_pt_y = cd.pts_to_data(ax_gt, 0.0, 1.0)[1]
    el = np.nan_to_num(r.gt_peak_elev, nan=0.0)
    done = []
    for k in np.nonzero(r.visible & (np.abs(el) > EL_HEAT - 1.0))[0]:
        x = float(cd.strip_x(r.gt_bearing[k]))
        s = 1.0 if el[k] > 0 else -1.0
        if any(abs(x - x0) < 1.5 and s == s0 for x0, s0 in done):
            continue
        done.append((x, s))
        ax_gt.plot([x], [s * (EL_HEAT - (CARET_PT / 2 + 0.4) * per_pt_y)], ls="none", marker="^" if s > 0 else "v",
                   ms=CARET_PT, mfc=GT_INK, mec="white", mew=0.5, zorder=6, clip_on=False)


def _draw_merged_notes(ax_lane, fig, r: dd.CaseRow, p, L: dict, free: list, y_badge: float,
                       pt_per_deg: float) -> None:
    """``CaseOptions.merge_notes``: one note per kind, carrying every slot's badge and P(not visible)."""
    current = [k for k in r.invisible_slots() if r.gt_dist[k] < 0.1]
    unseen = [k for k in r.invisible_slots() if r.gt_dist[k] >= 0.1]
    pred_none = [k for k in range(dd.K) if r.visible[k] and p.none_p[k] > 0.5]
    notes = [([k], L["current"].format(p=p.none_p[k])) for k in current]
    for ks, key in ((unseen, "not_visible"), (pred_none, "pred_none")):
        if len(ks) == 1:
            notes.append((ks, L[key].format(p=p.none_p[ks[0]])))
        elif ks:
            notes.append((ks, L[key + "_n"].format(ps=", ".join(f"{p.none_p[k]:.2f}" for k in ks))))
    gap = 1.0 / pt_per_deg
    for ks, text in notes:
        bws = [cd.badge_width_pt(str(k + 1)) / pt_per_deg for k in ks]
        badges = sum(bws) + gap * (len(ks) - 1)
        width = badges + (1.2 + cd.text_width_pt(fig, text, FS["note"])) / pt_per_deg
        for j, (lo, hi) in enumerate(free):
            if hi - lo >= width:
                x = lo
                for k, bw in zip(ks, bws):
                    cd.history_badge(ax_lane, x + bw / 2, y_badge, str(k + 1), k)
                    x += bw + gap
                ax_lane.text(lo + badges + 1.2 / pt_per_deg, y_badge, text, ha="left", va="center",
                             fontsize=FS["note"], color=style.INK_2)
                free[j] = (lo + width + 8.0 / pt_per_deg, hi)
                break
        else:
            print(f"[fig_case] frame {r.frame}: no room for the note on slots {[k + 1 for k in ks]}: {text}")
            _DROPPED.append(f"frame {r.frame}, slots {[k + 1 for k in ks]}: {text}")


def draw_top_band(page: Page, L: dict) -> None:
    y_names = TOP_H - 0.085
    page.text(X_ROUTE, y_names, "a", ha="left", va="center", fontsize=FS["title"] + 0.5, fontweight="bold",
              color=style.INK)
    page.text(X_ROUTE + 0.13, y_names, L["a"], ha="left", va="center", fontsize=FS["name"], color=style.INK)
    page.text(X_INSET, y_names, "b", ha="left", va="center", fontsize=FS["title"] + 0.5, fontweight="bold",
              color=style.INK)
    page.text(X_INSET + 0.13, y_names, L["b"], ha="left", va="center", fontsize=FS["name"], color=style.INK)
    page.text(X_STRIP, y_names, "c", ha="left", va="center", fontsize=FS["title"] + 0.5, fontweight="bold",
              color=style.INK)
    q = W_STRIP / 4
    for v, name in enumerate(L["views"]):
        page.text(X_STRIP + (v + 0.5) * q, y_names, name, ha="center", va="center", fontsize=FS["name"],
                  color=style.INK if v == 0 else style.INK_2, fontweight="bold" if v == 0 else "normal")
    # bracket over the three views the model never sees
    y_br = y_names - 0.085
    ax = page.ax(X_STRIP + q + 0.02, y_br - 0.03, 3 * q - 0.04, 0.06)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, 0, 1, 1], [0.1, 0.55, 0.55, 0.1], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    page.text(X_STRIP + 2.5 * q, y_br - 0.075, L["not_given"], ha="center", va="center", fontsize=FS["small"],
              color=style.INK_2, fontstyle="italic")


def draw_legend(page: Page, y_top: float, arm: str, L: dict) -> None:
    """One line, five entries, spread over the full width."""
    fig = page.fig
    ax = page.ax(0.0, y_top, FIG_W, LEGEND_H)
    w_pt, h_pt = FIG_W * 72.0, LEGEND_H * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis("off")
    y = h_pt * 0.42

    def hist(x):
        for j, k in enumerate((0, 4, 7)):
            cd.history_badge(ax, x + 3.7 + j * 8.6, y, str(k + 1), k)

    def swatch(cmap):
        def draw(x):
            n = 24
            for i in range(n):
                ax.add_patch(Rectangle((x + i * 18.0 / n, y - 3.2), 18.0 / n + 0.05, 6.4, lw=0, ec="none",
                                       fc=cmap(cd.HEAT_TOP * (i + 0.5) / n)))
        return draw

    entries = [
        (hist, 3 * 8.6 - 1.2, L["legend_hist"]),
        (swatch(cd.GT_CMAP), 18.0, L["legend_gt"]),
        (swatch(cd.PRED_CMAP), 18.0, L["legend_pred"]),
        (lambda x: cd.peak_mark(ax, x + 3.0, y), 6.0, L["legend_peak"]),
        (lambda x: cd.robot_glyph(ax, x + 3.0, y, (0.0, 1.0), size_pt=6.4), 6.0, L["legend_robot"]),
    ]
    inner = 3.0
    widths = [cd.text_width_pt(fig, text, FS["legend"]) for _, _, text in entries]
    total = sum(gw + inner + tw for (_, gw, _), tw in zip(entries, widths))
    gap = min(18.0, (w_pt - 2.0 - total) / (len(entries) - 1))
    x = 1.0
    for (draw, gw, text), tw in zip(entries, widths):
        draw(x)
        ax.text(x + gw + inner, y, text, ha="left", va="center", fontsize=FS["legend"], color=style.INK)
        x += gw + inner + tw + gap


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def numbered_misses(r: dd.CaseRow, arm: str = ARM) -> List[int]:
    """Slots whose predicted peak is drawn alone and numbered: shown, and not visible or > MISS_DEG off."""
    p = r.arms[arm]
    shown = [k for k in range(dd.K) if r.valid[k] and p.none_p[k] <= 0.5 and p.peak_view[k] >= 0]
    return [k for k in shown if (not r.visible[k]) or p.err[k] > MISS_DEG]


def case_caption(lang: str, opt: CaseOptions, **fmt) -> str:
    """Caption of the case figure: ``CAPTION[lang]`` for the approved layout, else built from CAPTION_PARTS."""
    if opt.is_approved_layout and not opt.clamp_peaks:
        return CAPTION[lang].format(**fmt)
    P = CAPTION_PARTS[lang]
    return (P["head"] + P["a"] + P["b"] + P["c"]).format(**fmt) + caption_marks(lang, opt)


def caption_marks(lang: str, opt: CaseOptions) -> str:
    """The x / caret / tick / note / header sentences of the caption, matching the options drawn."""
    P = CAPTION_PARTS[lang]
    text = P["x"] + (P["x_lane"] if opt.miss_lane else P["x_inline"])
    if opt.gt_carets and opt.clamp_peaks:
        text += P["carets"]
    elif opt.clamp_peaks:
        text += P["carets_pred"]
    elif opt.gt_carets:
        text += P["carets_gt"]
    text += P["ticks"]
    if opt.wrap_notes:
        text += P["notes"]
    return text + P["headers"]


def _choose_scale_corner(insets, recs) -> Optional[float]:
    """``scale_corner="fixed"``: the inset corner with the most clearance from the badges in every block."""
    clear = [inset_corner_clearance(ax, r, show_arrow=(n == 0)) for n, (ax, r) in enumerate(zip(insets, recs))]
    common = [c for c in (45.0, 135.0, 315.0, 225.0) if all(c in cl for cl in clear)]
    if not common:
        return None
    best = max(common, key=lambda c: (round(min(cl[c] for cl in clear)), -common.index(c)))
    worst = min(cl[best] for cl in clear)
    if worst < 12.0:
        print(f"[fig_case] inset scale bars: corner {best:g} deg is only {worst:.0f} deg from a badge in some block")
    return best


def make_case_figure(dump_npz_path, rows: Optional[List[int]] = None, topdown_root=None, clip_root_override=None,
                     out_stem="case", lang: str = "en", options: Optional[CaseOptions] = None) -> dict:
    """Render the case figure for one dump; returns {"files": [...], "rows": [...], "stats": [...]}.

    ``options`` (:class:`CaseOptions`) adds the route-figure extensions and the
    revised layout; without it the output is the case figure exactly.  The
    result also has "notes_dropped" (notes that found no room and were left
    out; always empty with ``wrap_notes``) and "layout" (what the revised
    layout decided: label panel, scale-bar corner, note lines, miss lane).
    """
    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    arm = ARM
    opt = options or CaseOptions()
    L = labels_for(lang, opt.wording) if opt.wording != "approved" else LABELS[lang]
    if opt.wording != "approved" and not opt.miss_lane:
        L["legend_peak"] = LABELS[lang]["legend_peak"]
    dump = dd.load_dump(dump_npz_path)
    if arm not in dump.arms:
        raise ValueError(f"arm {arm!r} not in dump arms {dump.arms}")
    rows = list(rows) if rows else dd.key_rows(dump, arm)
    bad = [i for i in rows if not 0 <= i < dump.n_rows]
    if bad or len(set(rows)) != len(rows):
        raise ValueError(f"rows {rows}: must be distinct indices in 0..{dump.n_rows - 1}")
    recs = [dd.case_row(dump, i) for i in rows]
    clip_dir = dd.resolve_clip_dir(dump, clip_root_override)
    level = dd.topdown_level(dump, float(np.median(dump.positions[:, 1])), root=topdown_root)

    if opt.roles is not None and len(opt.roles) != len(recs):
        raise ValueError(f"roles {opt.roles}: need one per row ({len(recs)})")
    del _DROPPED[:]

    # ---- revised layout: decided before the page exists (block heights depend on it)
    exts: Optional[List[BlockExt]] = None
    layout: dict = {}
    if not opt.is_approved_layout:
        fig_m = plt.figure(figsize=(FIG_W, 2.0))  # only for measuring text
        ppd = W_STRIP * 72.0 / 360.0
        label_panel = None
        if opt.row_labels == "fixed":
            strips = []
            for r in recs:
                strips += [cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, EL_HEAT),
                           cd.heat_strip(dd.pred_composite(r, arm), HEAT_RING_W, EL_HEAT)]
            w_deg = (max(cd.text_width_pt(fig_m, L[k], FS["small"]) for k in ("gt_row", "pred_row")) + 5.0) / ppd
            label_panel = cd.fixed_label_panel(strips, w_deg)
        lane = bool(opt.miss_lane and any(numbered_misses(r, arm) for r in recs))
        exts = []
        for r in recs:
            lane_notes, note_lines = [], []
            if opt.wrap_notes:
                items = note_items(r, r.arms[arm], L, opt.merge_notes)
                lane_notes, note_lines = layout_notes(fig_m, items, lane_layout(r)[3])
            exts.append(BlockExt(note_lines=note_lines, lane_notes=lane_notes, wrap_notes=opt.wrap_notes,
                                 miss_lane=lane, label_panel=label_panel, gt_carets=opt.gt_carets))
        plt.close(fig_m)
        layout = {"label_panel": label_panel, "miss_lane": lane, "note_lines": [len(e.note_lines) for e in exts]}
    block_h = [BLOCK_H + (e.extra_h if exts else 0.0) for e in (exts or [None] * len(recs))]

    title_h = TITLE_H if opt.title else (BANNER_H if opt.banner else 0.0)
    height = fig_height(len(recs)) + title_h
    if exts:
        height += sum(block_h) - len(recs) * BLOCK_H
    fig = plt.figure(figsize=(FIG_W, height))
    page = Page(fig, height, y0=title_h)
    if opt.title:
        t = page.text(X_ROUTE, -title_h + 0.11, opt.title, ha="left", va="center", fontsize=FS["title"] + 0.8,
                      fontweight="bold", color=style.INK)
        if opt.title_note:
            w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            page.text(X_ROUTE + w + 0.12, -title_h + 0.11, opt.title_note, ha="left", va="center",
                      fontsize=FS["name"], color=style.INK_2, fontstyle="italic")
    elif opt.banner:
        page.text(X_ROUTE, -title_h + 0.09, opt.banner, ha="left", va="center", fontsize=FS["name"],
                  color=style.INK_2, fontstyle="italic" if (opt.banner_italic and lang != "zh") else "normal")
    draw_top_band(page, L)
    insets = []
    for n, r in enumerate(recs):
        y_top = TOP_H + n * (BLOCK_H + BLOCK_GAP) + (sum(block_h[:n]) - n * BLOCK_H if exts else 0.0)
        views = dd.surround_views(clip_dir, r.frame)
        role = opt.roles[n] if opt.roles is not None else None
        insets.append(draw_block(page, y_top, n, r, views, arm, L, last=(n == len(recs) - 1), role=role,
                                 clamp_peaks=opt.clamp_peaks, merge_notes=opt.merge_notes,
                                 ext=exts[n] if exts else None))
    corner = _choose_scale_corner(insets, recs) if opt.scale_corner == "fixed" else None
    layout["scale_corner"] = corner
    for n, (ax_in, r) in enumerate(zip(insets, recs)):
        lvl = dd.topdown_level(dump, float(r.cur_pos[1]), root=topdown_root)
        draw_inset(ax_in, lvl, dump, r, show_arrow=(n == 0), L=L, split_frame=opt.split_frame,
                   letters=opt.letters, scale_corner=corner)

    y_route = TOP_H + HDR_H
    y_axis = TOP_H + len(recs) * BLOCK_H + (len(recs) - 1) * BLOCK_GAP
    if exts:
        y_axis += sum(block_h) - len(recs) * BLOCK_H
    scene_h = 0.20  # two lines under the map: tier + scene, episode + length
    scene_text = opt.scene_text if opt.scene_text is not None else L["scene"].format(
        tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id, T=dump.frame_count)
    panel = opt.route_panel
    if opt.route_fit and panel is None:
        panel = FittedRoutePanel(start_text=L["start"])
    if opt.route_fit and hasattr(panel, "height"):
        # the route panel only as tall as the route needs; column a's rest goes to the foot (or stays blank)
        avail = y_axis + AXIS_H - y_route - scene_h
        foot = opt.route_foot if opt.route_foot is not None and opt.route_foot_h > 0 else None
        foot_min = opt.route_foot_h if foot is not None else 0.0
        h_route = panel.height(fig, dump, W_ROUTE, avail - foot_min)
        ax_route = page.ax(X_ROUTE, y_route, W_ROUTE, h_route)
        panel(ax_route, level, dump, recs, L)
        page.text(X_ROUTE, y_route + h_route + 0.035, scene_text, ha="left", va="top", fontsize=FS["note"],
                  color=style.MUTED, linespacing=1.15)
        if foot is not None:
            foot_h = avail - h_route
            foot_max = getattr(foot, "max_h", None)
            if foot_max is not None:
                foot_h = min(foot_h, foot_max)
            foot(page, X_ROUTE, y_route + h_route + scene_h, W_ROUTE, foot_h)
            layout["foot_h"] = round(foot_h, 3)
        layout["route_h"] = round(h_route, 3)
    else:
        y_bottom = y_axis + AXIS_H - opt.route_foot_h
        ax_route = page.ax(X_ROUTE, y_route, W_ROUTE, y_bottom - scene_h - y_route)
        (panel or draw_route_panel)(ax_route, level, dump, recs, L)
        page.text(X_ROUTE, y_bottom - scene_h + 0.035, scene_text, ha="left", va="top", fontsize=FS["note"],
                  color=style.MUTED, linespacing=1.15)
        if opt.route_foot is not None and opt.route_foot_h > 0:
            opt.route_foot(page, X_ROUTE, y_bottom, W_ROUTE, opt.route_foot_h)
    draw_legend(page, y_axis + AXIS_H, arm, L)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)
    caption = opt.caption if opt.caption is not None else case_caption(
        lang, opt, n=len(recs), tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id)
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption + "\n", encoding="utf-8")
    files.append(cap_path)
    stats = []
    for n, r in enumerate(recs):
        entry = {"key": f"K{n + 1}", "row": r.index, "frame": r.frame}
        for a in list(r.arms) + ["floor"]:
            entry[a] = r.summary(a)
        stats.append(entry)
    return {"files": [str(f) for f in files], "rows": rows, "stats": stats, "size_in": (FIG_W, height),
            "notes_dropped": list(_DROPPED), "layout": layout}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True, help="History Head dump npz of one clip")
    ap.add_argument("--rows", default=None, help="comma-separated query rows (default: pre-registered key rows)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--out", default="case", help="output stem (writes .pdf, .png, _caption.txt)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    ap.add_argument("--revised", action="store_true",
                    help="the revised layout (CaseOptions.revised(): review fixes of 2026-09-24)")
    args = ap.parse_args(argv)
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    res = make_case_figure(args.dump, rows=rows, topdown_root=args.topdown_root, clip_root_override=args.clip_root,
                           out_stem=args.out, lang=args.lang,
                           options=CaseOptions.revised() if args.revised else None)
    for f in res["files"]:
        print(f)
    for s in res["stats"]:
        print(s)
    print("size_in", tuple(round(v, 3) for v in res["size_in"]))
    if args.revised:
        print("layout", res["layout"], "notes_dropped", res["notes_dropped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
