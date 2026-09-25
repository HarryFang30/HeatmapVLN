#!/usr/bin/env python3
"""EXP-18 main case figure: the predicted affordance map vs ground truth at key positions of one episode.

Layout (7.0 in wide, double column; one block per key position K1..Kn):

  a  Route          b  Local map (robot facing up)    c  Front · model input | Right | Back | Left
  +---------+       K1 frame 20 of 79                       metrics of the row (right-aligned)
  | top-down|                        notes (slots with no view / predicted not visible), wrapped
  |  map,   |       ( disc )         lane: numbered blue badges at the true bearings
  | cropped |                        RGB row: the four current views unrolled clockwise (+-15 deg)
  | to the  |       ground truth ▬   ground-truth affordance map row (blue)
  |  route  |       prediction   ▬   predicted affordance map row (orange), x per predicted peak
  +---------+                        (lane under the row for numbered misses, only when needed)

Encodings (one meaning each; ``common_draw`` holds the shared vocabulary):

* **Blue = ground truth / past positions, orange = prediction, never swapped
  and never used for anything else** (``style``).
* a  Route: muted top-down map cropped to the route's bounding box plus
  ``ROUTE_PAD_M`` (``FittedRoutePanel``; a quarter turn when that shows the
  route larger), the whole route (grey line, open circle = start), each key
  position as a black dot + facing arrow + ``K`` badge fanned out clear of the
  route (leaders when moved, never crossing); scale bar in a band >= 2 pt
  inside the map frame; tier, scene and episode underneath.
* b  Local map: the map around the robot, rotated so the robot faces up and
  clipped to a disc whose rim IS the bearing ring that c unrolls.  Dashed
  radii are the view seams (+-45 deg, +-135 deg), so the four sectors are the
  four panels of c.  The grey line is the route so far.  Every past position
  visible in the ground truth gets a blue line from the robot through its true
  position (dot) to the rim and its number just outside the rim (slots at one
  spot share a badge, "1–6"); badges are dodged along the rim in bearing
  order (thin leader when moved; dots never move).  Sector letters F/R/B/L
  sit at the sector centres on the badge ring and are tested against the
  drawn badges, leaders and scale bar: blocked, they slide within their own
  sector, else move outward past the badges (into the free room of the block
  above / below / beside the disc), else just inside the rim -- never under a
  badge.  Each disc is as large as its rim badges allow: a pill badge ("1–3")
  that would run into the row names or column a first nudges the disc
  sideways (<= 5 pt), then shrinks it (``fit_inset``).  The curved arrow
  (first block) shows where c starts and that it runs clockwise.  Scale bar
  per block, one shared corner.
* c  Surround strip, horizontal axis = bearing in square degrees, starting at
  the front view's left edge (+45 deg) and running clockwise F, R, B, L.
  - header: "K1  frame 20 of 79" (``cd.frame_label``: 1-based count, D6), then
    right-aligned the bearing error of the predicted peaks (median, max over
    GT-visible slots) and joint PCK@8 hits/visible, and in grey the constant
    "always-behind guess: median .., PCK@8 ../.." for reference.  A header
    that would collide with a long frame/role text wraps the reference onto a
    second line.
  - notes (D5): slots that need words -- "8 = previous frame (at the robot):
    not visible · predicted P(not visible) = 1.00", runs of consecutive slots
    no view shows ("2–4 not visible in any view · predicted P(not visible)
    0.67–0.98"), and GT-visible slots the model calls not visible (a miss
    without an x; its number in the miss badge).  Notes go left of the lane
    badges when they fit, else onto note lines (the block grows); a note is
    never dropped.  ``data.check_accounting`` raises unless every valid slot
    has a lane badge or a note and every miss is numbered exactly once.
  - lane: the numbered blue badges, dodged sideways, leaders to the true
    bearing; the blue guide line continues through the RGB row.
  - RGB row: of the four current views only the framed front view is given
    to the model (the model also receives the past frames' front images);
    right/back/left are washed out and bracketed "shown for reference only".
  - ground-truth affordance map row (blue) and predicted affordance map row
    (orange; always the deployed model's output): each slot's map divided by
    its own peak (the prediction also x (1 - P(not visible))), max over
    slots, colour linear in that value, the same ramp position in both rows.
    Row names "ground truth" / "prediction" sit in a fixed gutter left of the
    strip with a tiny colour key (D3), in every block.
  - elevation window (D4, ``cd.elevation_window``): +-10 deg, widened per
    block just enough to include every GT-visible and predicted peak (at most
    +-45 deg); the caption states each block's window.
  - x = predicted peak (joint argmax of heatmaps_gated) of each GT-visible
    slot with P(not visible) <= 0.5; hits within 4 deg share one x, touching
    marks are staggered vertically.  A slot is **missed** iff it fails joint
    PCK@8 (D1, ``CaseRow.misses``: 5-way view class wrong or per-view argmax
    > 8 px from the GT peak in the GT view -- the fields and rule of
    ``compute_metrics``), so a block's numbered misses = its header's
    n - hits.  A missed x carries its number in ink in a white disc ringed
    orange (D2, ``cd.miss_badge``), beside it or, when crowded, moved along a
    lane inside the row or to a lane under the row, joined by a short ink
    leader (white halo); misses whose peaks would print as one blot share one
    x and one badge ("4–5"); a thin dotted grey line joins the x to the slot's
    true bearing (blue tick under the row) when they are < 45 deg apart
    (``cd.CONNECTOR_MAX_DEG``).  Slots no view shows get no x
    (their map shows in the row, their note gives P(not visible)).
  - short blue ticks under the prediction row repeat the true bearings.

Figure policy (user decision, 2026-09-24): the figure shows the affordance map
only.  It never mentions poses, their sources or the pose-source ablation; the
prediction drawn is always the deployed model's (the dump's ``vo`` arm).
Honesty that remains on the figure: of the current views only the front view
is marked as model input, the other three display-only; ground truth blue vs
prediction orange, never swapped; misses shown (numbered), never hidden; the
always-behind guess as a reference; no claim of localization from vision.

Options (``CaseOptions``, used by ``fig_routes`` and ``make_all``): title line
(+ note), a banner, per-block role text, a replacement route panel and a panel
under it, route legs in the insets, the sector-letter fallback, scene text and
caption overrides.  The D1-D7 conventions are always on; the older switches
(``row_labels``, ``miss_lane``, ``wrap_notes``, ``merge_notes``,
``gt_carets``, ``clamp_peaks``, ``wording``) are accepted for compatibility and
no longer change anything.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_case --dump <clip.npz> [--rows 0,2,8] [--topdown-root DIR]
      [--clip-root DIR] [--lang en|zh] [--title T] [--title-note N] --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        "b": "Local map (robot facing up)",
        "views": ("Front · model input", "Right", "Back", "Left"),
        "sectors": ("F", "R", "B", "L"),
        "not_given": "shown for reference only (not given to the model)",
        "frame": "frame {n} of {T}",
        "frame_last": "frame {n} of {T}",
        "gt_row": "ground truth",
        "pred_row": "prediction",
        "metric_main": "bearing error median {med:.1f}°, max {mx:.1f}°  ·  PCK@8 {hits}/{n}",
        "metric_floor": "always-behind guess: median {med:.0f}°, PCK@8 {hits}/{n}",
        "sep": "   ·   ",
        "role_sep": " · ",
        # notes (D5, D6); {ps} = "= 0.77" for one slot, "0.67–0.98" for a run
        "note_previous": "= previous frame (at the robot): not visible · predicted P(not visible) {ps}",
        "note_at_robot": "at the robot: not visible · predicted P(not visible) {ps}",
        "note_at_robot_prev": "at the robot (8 = previous frame): not visible · predicted P(not visible) {ps}",
        "note_not_visible": "not visible in any view · predicted P(not visible) {ps}",
        "note_predicted_none": "visible, but predicted P(not visible) {ps}",
        "ps_one": "= {p:.2f}",
        "ps_range": "{a:.2f}–{b:.2f}",
        # older keys (fig_anim reads some); same wording
        "current": "= previous frame (at the robot): not visible · predicted P(not visible) = {p:.2f}",
        "not_visible": "not visible in any view · predicted P(not visible) = {p:.2f}",
        "pred_none": "visible, but predicted P(not visible) = {p:.2f}",
        "not_visible_n": "not visible in any view · predicted P(not visible) {ps}",
        "pred_none_n": "visible, but predicted P(not visible) {ps}",
        "false_pos": "not visible in any view, predicted here",
        "axis": ("0° (straight ahead)", "−90°", "180°", "+90°"),
        "legend_hist": "past position k (1 = oldest)",
        "legend_gt": "ground-truth affordance map",
        "legend_pred": "predicted affordance map",
        "legend_peak": "predicted peak",
        "legend_miss": "miss (fails joint PCK@8)",
        "legend_robot": "robot",
        "scene": "{tier} {scene}\nepisode {ep} · {T} frames",
        "start": "start",
    },
    "zh": {
        "a": "路线",
        "b": "局部地图（机器人朝上）",
        "views": ("前 · 模型输入", "右", "后", "左"),
        "sectors": ("前", "右", "后", "左"),
        "not_given": "仅作展示（不输入模型）",
        "frame": "第 {n} 帧（共 {T} 帧）",
        "frame_last": "第 {n} 帧（共 {T} 帧）",
        "gt_row": "真值",
        "pred_row": "预测",
        "metric_main": "方位误差 中位 {med:.1f}°，最大 {mx:.1f}°  ·  PCK@8 {hits}/{n}",
        "metric_floor": "恒答正后方：中位 {med:.0f}°，PCK@8 {hits}/{n}",
        "sep": "  ·  ",
        "role_sep": " · ",
        "note_previous": "= 上一帧（与机器人重合）：不可见 · 预测不可见概率 {ps}",
        "note_at_robot": "与机器人重合：不可见 · 预测不可见概率 {ps}",
        "note_at_robot_prev": "与机器人重合（8 = 上一帧）：不可见 · 预测不可见概率 {ps}",
        "note_not_visible": "任何视角都不可见 · 预测不可见概率 {ps}",
        "note_predicted_none": "可见，但预测不可见概率 {ps}",
        "ps_one": "{p:.2f}",
        "ps_range": "{a:.2f}–{b:.2f}",
        "current": "= 上一帧（与机器人重合）：不可见 · 预测不可见概率 {p:.2f}",
        "not_visible": "任何视角都不可见 · 预测不可见概率 {p:.2f}",
        "pred_none": "可见，但预测不可见概率 {p:.2f}",
        "not_visible_n": "任何视角都不可见 · 预测不可见概率 {ps}",
        "pred_none_n": "可见，但预测不可见概率 {ps}",
        "false_pos": "任何视角都不可见，却预测在此",
        "axis": ("0°（正前方）", "−90°", "180°", "+90°"),
        "legend_hist": "历史位置 k（1 = 最早）",
        "legend_gt": "真值 affordance map",
        "legend_pred": "预测 affordance map",
        "legend_peak": "预测峰值",
        "legend_miss": "未命中（joint PCK@8 不通过）",
        "legend_robot": "机器人",
        "scene": "{tier} {scene}\n第 {ep} 集 · {T} 帧",
        "start": "起点",
    },
}

# Kept for callers that lay it over LABELS (``labels_for(lang, "revised")``): the D1-D7 wording is LABELS now.
LABELS_REVISED: Dict[str, Dict[str, object]] = {"en": {}, "zh": {}}

# The prediction drawn is always the deployed model's output (dump arm "vo").
ARM = "vo"


def labels_for(lang: str, wording: str = "approved") -> dict:
    """``LABELS[lang]`` (a copy); ``wording`` is accepted for compatibility (one wording since D6)."""
    L = dict(LABELS[lang])
    L.update(LABELS_REVISED.get(lang, {}))
    return L


# Caption parts.  "{elev_window}" is replaced by make_case_figure with the blocks' elevation windows (D4), also in
# a caption a caller passes through ``CaseOptions.caption`` (fig_routes builds one from these parts).
KEY_RULE = {
    "en": ("key positions: first scored frame, frame with the widest ground-truth bearing spread, last frame (the "
           "middle scored frame when the widest spread is at the first or last frame)"),
    "zh": "关键位置：第一个评分帧、真值方位跨度最大的帧、末帧（跨度最大的帧恰为首帧或末帧时改取中间的评分帧）",
}
CAPTION_PARTS = {
    "en": {
        "head": "Predicted affordance maps at {n} key positions of one episode ({tier} {scene}, episode {ep}; {rule}). ",
        "a": "(a) Route on the top-down map. ",
        "b": ("(b) Map around each key position, robot facing up; its rim is the bearing ring that (c) unrolls "
              "clockwise from the front view's left edge (arrow); dashed radii are the view seams. Blue lines run from "
              "the robot through each past position (dot; 1 = oldest of the 8 queried) to its number on the rim. "),
        "c": ("(c) The surround view on the same bearings: of the four current views, only the framed front view is "
              "given to the model; right, back and left are shown for reference (the model also receives the past "
              "frames' front images). Below it, the ground-truth affordance map (blue) and the predicted affordance "
              "map (orange; the deployed model's output); {elev_window} (the images span ±15°). Each slot's map is "
              "divided by its own peak (the prediction also multiplied by its predicted visibility), the maximum over "
              "slots is shown, and colour is linear in that value in both rows. Maps stop at view seams because every "
              "label and prediction lives in one 90° view. "),
        "x": ("×: predicted peak of each slot visible in the ground truth (hits within 4° share one ×; missed slots "
              "whose peaks coincide share one × and one number, e.g. 4–5; touching marks are staggered vertically). "),
        "misses": ("A slot is missed when it fails joint PCK@8 (predicted view wrong, or peak more than 8 px of 64 from "
                   "the true peak in that view); its number, in a white disc ringed in orange, sits at its own × (joined "
                   "to it by a short dark leader where it had to move), and a dotted line joins the × to the slot's "
                   "true bearing when they are less than 45° apart. A missed slot the model calls not visible (P(not visible) > 0.5) has no ×; its "
                   "orange-ringed number is in the notes. Each block's numbered misses equal its visible slots minus "
                   "its PCK@8 hits. "),
        "ticks": "Blue ticks under the prediction repeat the true bearings. ",
        "notes": ("Notes above the images list past positions that no view shows (the previous frame at the robot, "
                  "or out of sight), with the predicted probability P(not visible); they are not scored and get no ×, "
                  "but one predicted visible still shows in the orange row. "),
        "headers": ("Headers: frame (counted from 1), bearing error of the predicted peaks over visible slots (median, "
                    "max) and joint PCK@8, and for reference the constant always-behind guess (back view, centre "
                    "pixel)."),
        # older keys (fig_routes' caption_marks callers); folded into the parts above
        "x_lane": "", "x_inline": "", "carets": "", "carets_pred": "", "carets_gt": "",
    },
    "zh": {
        "head": "同一集（{tier} {scene}，第 {ep} 集）{n} 个关键位置上的预测 affordance map（{rule}）。",
        "a": "(a) 俯视图上的路线。",
        "b": ("(b) 各关键位置的局部地图，机器人朝上；圆周就是 (c) 从前视左缘顺时针展开的方位环（箭头），虚线半径为视角分界。"
              "蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号。"),
        "c": ("(c) 同一方位轴上的环视：当前四个视角中只有加框的前视图输入模型，右/后/左仅作展示（模型另外还接收历史帧的前视图）。"
              "下方为真值 affordance map（蓝）与预测 affordance map（橙，即部署模型的输出）；{elev_window}（环视图为 ±15°）。"
              "每个槽位的图除以自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值线性着色。图在视角分界处截断，"
              "因为每个标签和预测都只落在一个 90° 视角里。"),
        "x": "×：真值可见的各槽位的预测峰值（命中的槽位 4° 内共用一个 ×；峰值重合的未命中槽位共用一个 × 和一个编号，如 4–5；相互挨着的上下错开）。",
        "misses": ("joint PCK@8 不通过（预测视角错误，或峰值在该视角中距真值峰值超过 8 px（共 64 px））即为未命中：其编号写在橙色描边"
                   "的白色圆内，放在对应的 × 旁（需要挪开时用深色短线相连），两者相距 45° 以内时再用虚线把 × 连到该槽位的真值方位。模型判为不可见"
                   "（预测不可见概率 > 0.5）的未命中槽位没有 ×，其橙色描边编号列在注释中。每个关键位置的编号未命中数等于可见槽位数"
                   "减去 PCK@8 命中数。"),
        "ticks": "预测行下方的蓝色短线重复真值方位。",
        "notes": "图像上方的注释列出任何视角都看不到的历史位置（与机器人重合的上一帧，或视线之外），并给出预测不可见概率；这些槽位不计分、没有 ×，但被预测为可见的仍会出现在橙色行中。",
        "headers": "行首：帧号（从 1 数起）、可见槽位上预测峰值的方位误差（中位、最大）与 joint PCK@8，以及作参照的恒答正后方基线（后视中心像素）。",
        "x_lane": "", "x_inline": "", "carets": "", "carets_pred": "", "carets_gt": "",
    },
}
# The whole default caption with the fig_case key rule (kept as a name for callers).
CAPTION = {lang: "".join(P[k] for k in ("head", "a", "b", "c", "x", "misses", "ticks", "notes", "headers"))
           for lang, P in CAPTION_PARTS.items()}


@dataclass
class CaseOptions:
    """Optional extensions of the case layout (``fig_routes``, ``make_all``); ``None`` fields change nothing.

    * ``title`` / ``title_note``: a title line (bold) and a muted note after it
      (italic in en only), in a band of ``TITLE_H`` added above the panel names.
    * ``banner`` (+ ``banner_italic``): one muted line above the panel names
      (a development stand-in notice), without a title.
    * ``roles``: per key position, text appended to the block header after the
      frame ("frame 84 of 120 · after the turnaround").
    * ``route_panel``: ``f(ax, level, dump, recs, L)`` drawn instead of the
      fitted route panel into panel a (an object with ``height(fig, dump, w,
      h_max)`` is sized like ``FittedRoutePanel``).
    * ``route_foot_h`` / ``route_foot``: height (in) reserved at the bottom of
      column a, under the map and its scene note, and ``f(page, x, y_top, w, h)``
      drawing into it.
    * ``split_frame``: frame where the route turned back; insets past it draw
      the route so far as outbound (solid) and return (dashed) legs.
    * ``letters``: fallback of a blocked sector letter after sliding within
      its sector: "outward" (default; R/L outward, then inside the rim),
      "slide" (outward for all four, then inside) or "inside".
    * ``route_fit`` (default True): panel a cropped to the route plus
      ``ROUTE_PAD_M`` (``FittedRoutePanel``); False: the older whole-map panel.
    * ``scale_corner``: "fixed" (default; the insets' scale bars share the
      corner clear of badges in every block) or "auto" (per block).
    * ``scene_text``: the two lines under the map, instead of ``L["scene"]``.
    * ``caption``: caption text written instead of the built one ("{elev_window}"
      is still filled in).
    * ``caption_extra``: a sentence appended to the caption (e.g. "Candidates 2
      and 4 are two instructions of one R2R path.").
    * ``key_rule``: the words for how the key positions were chosen (default
      ``KEY_RULE``, the rule of ``data.key_rows``).

    Accepted for compatibility, no effect since the D1-D7 revision (the
    conventions are always on): ``row_labels``, ``miss_lane``, ``wrap_notes``,
    ``merge_notes``, ``gt_carets``, ``clamp_peaks``, ``wording``.
    """

    title: Optional[str] = None
    title_note: Optional[str] = None
    roles: Optional[Sequence[Optional[str]]] = None
    route_panel: Optional[Callable] = None
    route_foot_h: float = 0.0
    route_foot: Optional[Callable] = None
    split_frame: Optional[int] = None
    letters: str = "outward"
    clamp_peaks: bool = True
    merge_notes: bool = True
    scene_text: Optional[str] = None
    caption: Optional[str] = None
    row_labels: str = "gutter"
    miss_lane: bool = True
    wrap_notes: bool = True
    gt_carets: bool = True
    route_fit: bool = True
    scale_corner: str = "fixed"
    wording: str = "revised"
    banner: Optional[str] = None
    banner_italic: bool = True
    caption_extra: Optional[str] = None
    key_rule: Optional[str] = None

    @classmethod
    def revised(cls, **kw) -> "CaseOptions":
        """The layout of the 2026-09-24 review (now the default); ``kw`` overrides any field."""
        return cls(**kw)

    @property
    def is_approved_layout(self) -> bool:
        """Always False: the approved (pre-review) layout was retired by the D1-D7 decisions."""
        return False


# --------------------------------------------------------------------------- #
# Geometry of the page (inches)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # 7.0
X_ROUTE, W_ROUTE = 0.0, 1.02
X_INSET = 1.10
W_INSET = 0.96
GUTTER_W = 0.58  # row names (D3) between the inset column and the strip
X_STRIP = X_INSET + W_INSET + GUTTER_W
W_STRIP = FIG_W - 0.02 - X_STRIP
PPD = W_STRIP * 72.0 / 360.0  # points per degree on the strip (both axes: square degrees)
EL_RGB = 15.0  # RGB row: elevation +-15 deg
EL_HEAT = cd.EL_DEFAULT  # default half window of the affordance map rows (D4: widened per block, cd.elevation_window)
RGB_H = W_STRIP * 2 * EL_RGB / 360.0  # square degrees
HEAT_H = W_STRIP * 2 * EL_HEAT / 360.0  # at the default window
TOP_H = 0.31
HDR_H = 0.15
HDR2_H = 0.12  # second header line (reference numbers wrapped)
LANE_H = 0.14
ROW_GAP = 0.028
BLOCK_GAP = 0.10
AXIS_H = 0.17
LEGEND_H = 0.20
BODY_H = LANE_H + RGB_H + 2 * ROW_GAP + 2 * HEAT_H
BLOCK_H = HDR_H + BODY_H
RGB_RING_W = 1816  # ring columns per 360 deg (multiple of 8: exact roll), ~400 dpi at W_STRIP
HEAT_RING_W = 1440
MISS_DEG = 5.0  # unused since D1 (misses = joint PCK@8 failures, data.CaseRow.misses); kept for importers
MERGE_DEG = cd.MERGE_DEG
MARK_PT = cd.MARK_PT
STAGGER_PT = cd.STAGGER_PT
FS = {"title": 7.0, "name": 6.6, "header": 6.3, "small": 5.8, "note": 5.7, "legend": 6.1, "axis": 6.0}
NOTE_LINE_H = 0.13  # one note line between a block header and its lane
MISS_LANE_H = cd.BELOW_LANE_H_PT / 72.0  # lane under the prediction row for miss badges that do not fit inside it
MISS_BADGE_PT = cd.BELOW_LANE_PT
NOTE_GAP_PT = 14.0  # a note left of the lane badges keeps at least this far from them
CARET_PT = 4.4
GT_INK = style.GT_INK
ROUTE_PAD_M = 0.45  # route_fit: margin around the route (metres)
TITLE_H = 0.25  # optional title band (CaseOptions.title)
BANNER_H = 0.20  # height of the optional banner line
_DROPPED: List[str] = []  # kept for importers; notes are never dropped any more


def fig_height(n_blocks: int) -> float:
    """Height of a figure of ``n_blocks`` default blocks (no notes, +-10 deg rows, no miss lane)."""
    return TOP_H + n_blocks * BLOCK_H + (n_blocks - 1) * BLOCK_GAP + AXIS_H + LEGEND_H


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

    def bbox(self, x: float, y_top: float, w: float, h: float):
        """Display-space box of a page rectangle (inches from the top-left, like ``ax``)."""
        from matplotlib.transforms import Bbox

        y_top = y_top + self.y0
        dpi = self.fig.dpi
        return Bbox.from_bounds(x * dpi, (self.h - y_top - h) * dpi, w * dpi, h * dpi)


# --------------------------------------------------------------------------- #
# Panel a
# --------------------------------------------------------------------------- #
def draw_route_panel(ax, level, dump: dd.Dump, rows: Sequence[dd.CaseRow], L: dict) -> None:
    """The older whole-map panel a (``CaseOptions(route_fit=False)``)."""
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
    # bar 12 pt below the frame's top: its label (above the bar) keeps >= 2 pt inside the map
    cd.scale_bar(ax, limits[0] + 5 * per_pt, limits[2] + 12.0 * per_pt, bar, f"{bar:g} m")


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
        self.scale_bar_inset_pt = None  # set when drawn: the scale bar's (and its label's) distance to the frame

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
        # the route + pad, grown until every K position's facing arrow (13 pt) also fits with 3 pt to spare
        pts = xy
        for _ in range(3):
            x0, x1, y0, y1 = cd.fit_limits(pts, pad=self.pad, aspect_hw=aspect)
            m_per_pt = (x1 - x0) / w_pt
            tips = []
            for r in recs:
                p = local(r.cur_pos[0], r.cur_pos[2])
                fw = forward_from_c2w(r.cur_c2w)
                f = local(centre[0] + fw[0], centre[1] + fw[1])
                f = f / (np.linalg.norm(f) + 1e-12)
                tips.append(p + f * 16.0 * m_per_pt)
            tips = np.asarray(tips).reshape(-1, 2)
            inside = (np.all(tips[:, 0] >= x0 + 3 * m_per_pt) and np.all(tips[:, 0] <= x1 - 3 * m_per_pt)
                      and np.all(tips[:, 1] >= y0 + 3 * m_per_pt) and np.all(tips[:, 1] <= y1 - 3 * m_per_pt))
            if inside:
                break
            pts = np.concatenate([xy, tips], axis=0)  # fit_limits pads them like the route
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
        sb = cd.scale_bar(ax, bx - (bar_box[0] / 2 - 1.0) * per_pt, by - 3.0 * per_pt, bar, f"{bar:g} m")
        self.scale_bar_inset_pt = round(cd.inset_from_frame_pt(ax, sb), 2)  # judge rule: >= 2 pt inside the map
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




# --------------------------------------------------------------------------- #
# Panel b
# --------------------------------------------------------------------------- #
INSET_CORNERS = {135.0: (-1, 1), 45.0: (1, 1), 225.0: (-1, -1), 315.0: (1, -1)}  # plot angle -> signs
ARROW_MID = 90.0 + cd.STRIP_START_DEG - 17.0  # plot angle of the middle of the first block's direction arrow


def inset_half(r: dd.CaseRow) -> float:
    """Disc radius (metres) of panel b: 1.12 x the farthest visible past position, at least 1 m."""
    far = float(np.max(r.gt_dist[r.visible])) if r.visible.any() else 1.0
    return max(1.0, 1.12 * far)


def _arrow_fits(occupied, show_arrow: bool) -> bool:
    return show_arrow and all(abs((t - ARROW_MID + 180) % 360 - 180) > 17.0 + w + 4.0 for t, w in occupied)


INSET_RADIUS_FRAC = 0.74  # disc radius / half the inset square (cd.draw_local_disc's default)
INSET_RADIUS_FRACS = (0.74, 0.71, 0.68, 0.65, 0.62)  # tried in turn when a rim badge would run into the gutter
INSET_SHIFTS_PT = (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0)  # sideways nudges of the inset (pt)


def inset_corner_clearance(ax, r: dd.CaseRow, show_arrow: bool,
                           radius_frac: float = INSET_RADIUS_FRAC) -> Dict[float, float]:
    """Per free corner of the inset square (plot angle 45/135/225/315): angular clearance to the rim badges.

    The same badge layout ``draw_inset`` will draw (nothing is drawn here); the
    top-left corner is not free when the direction arrow is drawn there.
    """
    half = inset_half(r)
    lim = half / radius_frac
    ax.set_xlim(-lim, lim)  # draw_local_disc's limits, so the point scale is the drawn one
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    bearings = [float(r.gt_bearing[g[0]]) for g in r.groups]
    occupied = []
    if r.groups:
        _, _, placed, widths = cd.disc_rim_layout(ax, half, r.groups, bearings)
        occupied = [(float(p % 360.0), float(w / 2)) for p, w in zip(placed, widths)]
    corners = dict(INSET_CORNERS)
    if _arrow_fits(occupied, show_arrow):
        corners.pop(135.0)
    return {c: min([abs((c - t + 180) % 360 - 180) for t, _ in occupied] + [360.0]) for c in corners}


def rim_badge_boxes(ax, r: dd.CaseRow, radius_frac: float = INSET_RADIUS_FRAC) -> list:
    """Display-space boxes of the rim badges ``draw_inset`` would draw on ``ax`` at ``radius_frac`` (sets the
    axes' limits as ``draw_local_disc`` will; draws nothing)."""
    from matplotlib.transforms import Bbox

    half = inset_half(r)
    lim = half / radius_frac
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    if not r.groups:
        return []
    labels, _, placed, _ = cd.disc_rim_layout(ax, half, r.groups, [float(r.gt_bearing[g[0]]) for g in r.groups])
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    r_pt = half / per_pt
    px = ax.figure.dpi / 72.0
    out = []
    for lab, p in zip(labels, placed):
        u = np.array([math.cos(math.radians(p)), math.sin(math.radians(p))])
        c = ax.transData.transform(u * (r_pt + 1.5 + cd._badge_support(lab, u)) * per_pt)
        w, h = cd.badge_width_pt(lab) * px, 7.4 * px
        out.append(Bbox.from_bounds(c[0] - w / 2, c[1] - h / 2, w, h))
    return out


def fit_inset(ax, r: dd.CaseRow, keepout: Sequence, margin_pt: float = 1.5) -> Tuple[float, float, bool]:
    """Largest disc (``INSET_RADIUS_FRACS``), then smallest sideways nudge (``INSET_SHIFTS_PT``), whose rim badges
    keep ``margin_pt`` clear of every display box in ``keepout`` (row names and their keys in the gutter, the
    route panel); moves ``ax`` by the nudge.  Returns ``(radius_frac, shift_pt, ok)``."""
    fig = ax.figure
    px = fig.dpi / 72.0
    pos0 = ax.get_position()
    m = margin_pt * px
    W = fig.get_size_inches()[0] * fig.dpi

    def clear(boxes):
        return not any(b.x0 - m < k.x1 and k.x0 < b.x1 + m and b.y0 - m < k.y1 and k.y0 < b.y1 + m
                       for b in boxes for k in keepout)

    for rf in INSET_RADIUS_FRACS:
        for dx in INSET_SHIFTS_PT:
            ax.set_position([pos0.x0 + dx * px / W, pos0.y0, pos0.width, pos0.height])
            if clear(rim_badge_boxes(ax, r, rf)):
                return rf, dx, True
    ax.set_position(pos0)
    return INSET_RADIUS_FRACS[-1], 0.0, False


def draw_inset(ax, level, dump: dd.Dump, r: dd.CaseRow, show_arrow: bool, L: dict,
               split_frame: Optional[int] = None, letters: str = "outward",
               scale_corner: Optional[float] = None, radius_frac: float = INSET_RADIUS_FRAC,
               letter_bounds=None) -> None:
    """Panel b for one key position (``split_frame`` / ``letters``: see :class:`CaseOptions`).

    Draw order: map disc, blue rays and dots, robot, rim badges (bearing
    order), direction arrow, scale bar, then the sector letters, which are
    placed against everything already drawn (``cd.disc_sector_letters``).
    ``scale_corner``: plot angle (45, 135, 225, 315) of the corner for the
    scale bar; default the free corner farthest from every badge.
    ``radius_frac``: disc radius / half the inset square (``fit_inset``).
    ``letter_bounds``: display box a displaced sector letter may use (default
    the inset square), e.g. the free room above and below the disc in a tall
    block, so a blocked F or B can sit just outside the badges.
    """
    fwd = forward_from_c2w(r.cur_c2w)
    half = inset_half(r)
    past = dump.positions[: r.frame + 1][:, [0, 2]]
    if split_frame is not None:
        crop = cd.draw_local_disc(ax, level, r.cur_pos[[0, 2]], fwd, half, past_xz=past, radius_frac=radius_frac,
                                  past_split=split_frame if split_frame < r.frame else len(past) - 1)
    else:
        crop = cd.draw_local_disc(ax, level, r.cur_pos[[0, 2]], fwd, half, past_xz=past, radius_frac=radius_frac)
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
    corners = dict(INSET_CORNERS)
    if _arrow_fits(occupied, show_arrow):
        cd.disc_direction_arrow(ax, half)
        corners.pop(135.0)
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
    cd.disc_sector_letters(ax, half, occupied, names=L["sectors"], displaced=letters,
                           rays=[90.0 + b for b in bearings], bounds=letter_bounds)


# --------------------------------------------------------------------------- #
# Notes (D5) and the block plan
# --------------------------------------------------------------------------- #
def _ps(p: Sequence[float], L: dict) -> str:
    if len(p) == 1 or max(p) - min(p) < 0.005:
        return L["ps_one"].format(p=p[0])
    return L["ps_range"].format(a=min(p), b=max(p))


def note_items(r: dd.CaseRow, L: dict, arm: str = ARM) -> List[dict]:
    """The notes of one block (``data.row_notes`` + wording): [{"kind", "slots", "label", "style", "text"}].

    ``style`` "hist" (blue past-position badge: no view shows the slot) or
    "miss" (orange-ringed: GT-visible, predicted not visible -- a numbered miss).
    """
    out = []
    for note in dd.row_notes(r, arm):
        kind, ks = note["kind"], note["slots"]
        key = {"previous": "note_previous", "at_robot": "note_at_robot", "not_visible": "note_not_visible",
               "predicted_none": "note_predicted_none"}[kind]
        if kind == "at_robot" and dd.K - 1 in ks:
            key = "note_at_robot_prev"
        out.append({"kind": kind, "slots": list(ks), "label": dd.group_label(ks),
                    "style": "miss" if kind == "predicted_none" else "hist",
                    "text": L[key].format(ps=_ps(note["p"], L))})
    return out


def _note_width_pt(fig, item: dict) -> float:
    return cd.note_width_pt(fig, item, FS["note"])


def lane_layout(r: dd.CaseRow):
    """(targets, centres, labels, spans) of the ground-truth badges in a block's lane (strip degrees)."""
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in r.groups])
    labels = [dd.group_label(g) for g in r.groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / PPD for s in labels], 0.0, 360.0, 1.0 / PPD)
    spans = [(x - cd.badge_width_pt(s) / PPD / 2, x + cd.badge_width_pt(s) / PPD / 2) for x, s in zip(xs, labels)]
    return targets, xs, labels, spans


def _split_note(fig, item: dict, max_pt: float) -> List[dict]:
    """A note wider than a whole line, cut so each part fits (``cd.split_note``; continuations get no badge)."""
    return cd.split_note(fig, item, max_pt, FS["note"])


def layout_notes(fig, items: Sequence[dict], lane_spans: Sequence[tuple]):
    """(lane_notes, note_lines) for a block's notes, in strip degrees; every note is placed (D5, ``cd.wrap_notes``).

    Notes go left of the lane badges while they fit with ``NOTE_GAP_PT`` to
    spare (never to the right of the badge row, where their badges would read
    as more past positions); the rest flow onto full-width note lines, as
    many as they need.
    """
    line_pt = 358.0 * PPD
    left_hi = min([a for a, _ in lane_spans], default=360.0 + NOTE_GAP_PT / PPD) - NOTE_GAP_PT / PPD
    first, lines = cd.wrap_notes(fig, items, line_pt, first_pt=max((left_hi - 1.0) * PPD, 0.0), fs=FS["note"],
                                 sep_pt=8.0)
    lane = [(1.0 + x / PPD, it) for x, it in first]
    return lane, [[(1.0 + x / PPD, it) for x, it in line] for line in lines]


def _draw_note(ax, x: float, item: dict, y: float) -> None:
    cd.draw_note(ax, x, y, item, per_pt=1.0 / PPD, fs=FS["note"])


@dataclass
class BlockPlan:
    """Everything one block needs, decided before the page exists (its height depends on it)."""

    r: dd.CaseRow
    win: Tuple[float, float]
    gt_strip: np.ndarray
    pr_strip: np.ndarray
    marks: List[dict]
    placed: dict
    below: bool
    items: List[dict]
    lane_notes: list
    note_lines: list
    hdr_lines: int = 1
    warnings: List[str] = field(default_factory=list)

    @property
    def heat_h(self) -> float:
        return cd.strip_height_in(W_STRIP, self.win)

    @property
    def body_h(self) -> float:
        return LANE_H + RGB_H + 2 * ROW_GAP + 2 * self.heat_h

    @property
    def top_h(self) -> float:
        """Header line(s) and note lines, above the lane."""
        return HDR_H + (self.hdr_lines - 1) * HDR2_H + len(self.note_lines) * NOTE_LINE_H

    @property
    def height(self) -> float:
        return self.top_h + self.body_h + (MISS_LANE_H if self.below else 0.0)


def _metrics_text(r: dd.CaseRow, arm: str, L: dict):
    """(main, reference): the prediction's numbers, then the always-behind guess."""
    s = r.summary(arm)
    main = L["metric_main"].format(med=s["median"], mx=s["max"], hits=s["hits"], n=s["n"])
    sf = r.summary("floor")
    return main, L["metric_floor"].format(med=sf["median"], hits=sf["hits"], n=sf["n"])


def _frame_text(r: dd.CaseRow, frame_count: int, L: dict) -> str:
    return L["frame"].format(n=r.frame + 1, T=frame_count)


def plan_block(fig_m, n: int, r: dd.CaseRow, L: dict, frame_count: int, role: Optional[str] = None,
               arm: str = ARM) -> BlockPlan:
    """Window, marks, miss badges, notes and header lines of block ``n`` (``fig_m``: a figure for measuring)."""
    warnings = []
    win = cd.elevation_window([r], arm)
    if win != (-cd.EL_DEFAULT, cd.EL_DEFAULT):
        warnings.append(f"K{n + 1}: elevation window widened to {cd.fmt_deg(win[0])}..{cd.fmt_deg(win[1])}")
    gt_strip = cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, win)
    pr_strip = cd.heat_strip(dd.pred_composite(r, arm), HEAT_RING_W, win)
    marks = cd.peak_marks(r, arm, win, PPD)
    heat = cd.heat_lookup(pr_strip, win)
    conn = cd.miss_connectors(r, marks, win, PPD)
    placed = cd.place_miss_labels(marks, PPD, win, heat=heat, below=False, lines=conn)
    below = False
    if not placed["clean"]:
        placed_b = cd.place_miss_labels(marks, PPD, win, heat=heat, below=True, lines=conn)
        below = any(v["below"] for v in placed_b["labels"].values())
        placed = placed_b
        if below:
            warnings.append(f"K{n + 1}: miss badges need the lane under the prediction row")
        if not placed["clean"]:
            warnings.append(f"K{n + 1}: a miss badge overlaps or its leader crosses another mark")
    items = note_items(r, L, arm)
    lane_notes, note_lines = layout_notes(fig_m, items, lane_layout(r)[3])
    if len(note_lines) > 1:
        warnings.append(f"K{n + 1}: notes wrap onto {len(note_lines)} lines")
    # header: one line unless the frame (+ role) text would run into the numbers
    main, rest = _metrics_text(r, arm, L)
    w_left = (cd.text_width_pt(fig_m, _frame_text(r, frame_count, L) + ((L["role_sep"] + role) if role else ""),
                               FS["header"], fontweight="bold" if role else "normal") / 72.0 + 0.25)
    w_right = cd.text_width_pt(fig_m, main + L["sep"] + rest, FS["header"]) / 72.0
    hdr_lines = 1 if X_INSET + w_left + 0.12 <= X_STRIP + W_STRIP - w_right else 2
    return BlockPlan(r=r, win=win, gt_strip=gt_strip, pr_strip=pr_strip, marks=marks, placed=placed, below=below,
                     items=items, lane_notes=lane_notes, note_lines=note_lines, hdr_lines=hdr_lines,
                     warnings=warnings)


def draw_block(page: Page, y_top: float, n: int, plan: BlockPlan, views: np.ndarray, arm: str, L: dict,
               last: bool, role: Optional[str] = None, frame_count: int = 0, keepout: Optional[list] = None):
    """One key position: header, notes, lane, RGB row, both affordance map rows.

    Returns ``(ax_inset, numbered_on_row)`` (the slots whose miss badge sits on the prediction row); the row
    names and their keys are appended to ``keepout`` (artists the inset's rim badges must not touch).
    """
    fig = page.fig
    r, win = plan.r, plan.win
    lo, hi = win
    # ---- header: K badge + frame (inset column), numbers (right-aligned over the strip)
    y_mid = y_top + HDR_H * 0.45
    page.text(X_INSET, y_mid, f"K{n + 1}", ha="left", va="center", fontsize=FS["header"], fontweight="bold",
              color="white", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.3", fc=style.INK, ec="none"))
    t_frame = page.text(X_INSET + 0.25, y_mid, _frame_text(r, frame_count, L), ha="left", va="center",
                        fontsize=FS["header"], color=style.INK)
    if role:
        w_frame = t_frame.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        page.text(X_INSET + 0.25 + w_frame, y_mid, L["role_sep"] + role, ha="left", va="center",
                  fontsize=FS["header"], color=style.INK, fontweight="bold",
                  path_effects=cd.bold_effects(role, style.INK))
    main, rest = _metrics_text(r, arm, L)
    x_end = X_STRIP + W_STRIP
    if plan.hdr_lines == 1:
        t_rest = page.text(x_end, y_mid, L["sep"] + rest, ha="right", va="center", fontsize=FS["header"],
                           color=style.MUTED)
        w_rest = t_rest.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        page.text(x_end - w_rest, y_mid, main, ha="right", va="center", fontsize=FS["header"], color=style.INK)
    else:
        page.text(x_end, y_mid, main, ha="right", va="center", fontsize=FS["header"], color=style.INK)
        page.text(x_end, y_mid + HDR2_H, rest, ha="right", va="center", fontsize=FS["header"], color=style.MUTED)

    # ---- axes
    y_notes = y_top + HDR_H + (plan.hdr_lines - 1) * HDR2_H
    y_lane = y_notes + len(plan.note_lines) * NOTE_LINE_H
    y_rgb = y_lane + LANE_H
    y_gt = y_rgb + RGB_H + ROW_GAP
    y_pr = y_gt + plan.heat_h + ROW_GAP
    ax_lane = page.ax(X_STRIP, y_lane, W_STRIP, LANE_H)
    ax_rgb = page.ax(X_STRIP, y_rgb, W_STRIP, RGB_H)
    ax_gt = page.ax(X_STRIP, y_gt, W_STRIP, plan.heat_h)
    ax_pr = page.ax(X_STRIP, y_pr, W_STRIP, plan.heat_h)
    side = min(W_INSET, plan.body_h)
    ax_in = page.ax(X_INSET + (W_INSET - side) / 2, y_lane + (plan.body_h - side) / 2, side, side)

    cd.draw_rgb_row(ax_rgb, cd.rgb_strip(views, RGB_RING_W, EL_RGB), EL_RGB)
    cd.draw_heat_row(ax_gt, plan.gt_strip, win, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, plan.pr_strip, win, cd.PRED_CMAP)
    for ax_row, text, kind in ((ax_gt, L["gt_row"], "gt"), (ax_pr, L["pred_row"], "pred")):
        artists = cd.gutter_row_label(ax_row, text, kind)
        if keepout is not None:
            keepout.extend(artists)

    for i, line in enumerate(plan.note_lines):  # note lines between the header and the lane
        ax_note = page.ax(X_STRIP, y_notes + i * NOTE_LINE_H, W_STRIP, NOTE_LINE_H)
        ax_note.set_xlim(0, 360)
        ax_note.set_ylim(0, 1)
        ax_note.axis("off")
        for x, it in line:
            _draw_note(ax_note, x, it, 0.5)
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")

    # ---- ground truth: numbered badges in the lane, guide through the RGB row, ticks under the prediction
    targets, xs, labels, _ = lane_layout(r)
    y_badge = 0.56
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    for g, t, x, lab in zip(r.groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([x, x, t, t], [y_badge, 0.34, 0.12, 0.0], color=col, lw=0.55, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, x, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.55, zorder=3)
        ax_pr.plot([t, t], [lo - 1.0 * tick, lo - 4.0 * tick], color=col, lw=0.8, zorder=3, clip_on=False,
                   solid_capstyle="butt")
    for x, it in plan.lane_notes:
        _draw_note(ax_lane, x, it, y_badge)

    # ---- predicted peaks, misses (D1/D2)
    cd.draw_miss_connectors(ax_pr, r, plan.marks, win, PPD)
    cd.draw_peak_marks(ax_pr, plan.marks)
    numbered = cd.draw_miss_labels(ax_pr, plan.placed)

    if last:
        if plan.below:  # degree labels under the miss lane, on an axis of their own (no ticks through the lane)
            ax_ax = page.ax(X_STRIP, y_pr + plan.heat_h + MISS_LANE_H, W_STRIP, 0.001)
            ax_ax.set_xlim(0, 360)
            cd.clean_axes(ax_ax)
            ax_ax.patch.set_visible(False)
            cd.azimuth_axis(ax_ax, L["axis"], fs=FS["axis"])
            ax_ax.tick_params(axis="x", which="major", pad=3.0)
        else:
            cd.azimuth_axis(ax_pr, L["axis"], fs=FS["axis"])
            ax_pr.tick_params(axis="x", which="major", pad=5.5)
    return ax_in, numbered


def draw_top_band(page: Page, L: dict) -> None:
    y_names = TOP_H - 0.085
    for x, letter, name in ((X_ROUTE, "a", L["a"]), (X_INSET, "b", L["b"])):
        page.text(x, y_names, letter, ha="left", va="center", fontsize=FS["title"] + 0.5, fontweight="bold",
                  color=style.INK)
        page.text(x + 0.13, y_names, name, ha="left", va="center", fontsize=FS["name"], color=style.INK)
    page.text(X_STRIP - 0.10, y_names, "c", ha="left", va="center", fontsize=FS["title"] + 0.5,
              fontweight="bold", color=style.INK)
    q = W_STRIP / 4
    for v, name in enumerate(L["views"]):
        page.text(X_STRIP + (v + 0.5) * q, y_names, name, ha="center", va="center", fontsize=FS["name"],
                  color=style.INK if v == 0 else style.INK_2, fontweight="bold" if v == 0 else "normal",
                  path_effects=cd.bold_effects(name, style.INK) if v == 0 else None)
    # bracket over the three views not given to the model, its note set into the top line
    fig = page.fig
    y_br = y_names - 0.085
    ax = page.ax(X_STRIP + q + 0.02, y_br - 0.03, 3 * q - 0.04, 0.06)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w_note = cd.text_width_pt(fig, L["not_given"], FS["small"], fontstyle="italic") / 72.0 + 0.14
    frac = w_note / (3 * q - 0.04)
    ax.plot([0, 0, 0.5 - frac / 2], [0.1, 0.55, 0.55], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.plot([0.5 + frac / 2, 1, 1], [0.55, 0.55, 0.1], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.text(0.5, 0.55, L["not_given"], ha="center", va="center", fontsize=FS["small"], color=style.INK_2,
            fontstyle="italic" if not any("一" <= ch <= "鿿" for ch in L["not_given"]) else "normal")


def draw_legend(page: Page, y_top: float, arm: str, L: dict) -> List[str]:
    """One line of six entries over the full width (the font shrinks to 5.8 pt before entries crowd).

    Returns warnings (entries closer than 6 pt).
    """
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
        (lambda x: cd.miss_badge(ax, x + 3.7, y, "3"), 7.4, L.get("legend_miss", "")),
        (lambda x: cd.robot_glyph(ax, x + 3.0, y, (0.0, 1.0), size_pt=6.4), 6.0, L["legend_robot"]),
    ]
    entries = [e for e in entries if e[2]]
    inner = 3.0
    warnings = []
    for fs in (FS["legend"], 5.8):
        widths = [cd.text_width_pt(fig, text, fs) for _, _, text in entries]
        total = sum(gw + inner + tw for (_, gw, _), tw in zip(entries, widths))
        gap = min(18.0, (w_pt - 2.0 - total) / (len(entries) - 1))
        if gap >= 8.0:
            break
    if gap < 6.0:
        warnings.append(f"legend entries only {gap:.1f} pt apart")
    x = 1.0
    for (draw, gw, text), tw in zip(entries, widths):
        draw(x)
        ax.text(x + gw + inner, y, text, ha="left", va="center", fontsize=fs, color=style.INK)
        x += gw + inner + tw + gap
    return warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def numbered_misses(r: dd.CaseRow, arm: str = ARM) -> List[int]:
    """Slots numbered on the prediction row: missed (D1: fail joint PCK@8) and drawn as an x."""
    return r.misses_with_peak(arm)


def caption_marks(lang: str, opt: Optional[CaseOptions] = None) -> str:
    """The x / miss / tick / note / header sentences of the caption (``opt`` accepted for compatibility)."""
    P = CAPTION_PARTS[lang]
    return P["x"] + P["misses"] + P["ticks"] + P["notes"] + P["headers"]


def case_caption(lang: str, opt: Optional[CaseOptions] = None, **fmt) -> str:
    """Caption of the case figure from ``CAPTION_PARTS`` ("{elev_window}" left for ``make_case_figure``)."""
    opt = opt or CaseOptions()
    P = CAPTION_PARTS[lang]
    fmt.setdefault("rule", opt.key_rule or KEY_RULE[lang])
    head = P["head"].format(**fmt)
    return head + P["a"] + P["b"] + P["c"] + caption_marks(lang, opt)


def window_caption(wins: Sequence[Tuple[float, float]], lang: str) -> str:
    """"rows show elevation −10° to +10°", with the blocks that differ named: "(K2: −26° to +10°)"."""
    wins = [tuple(w) for w in wins]
    base = max(set(wins), key=lambda w: (wins.count(w), w == (-cd.EL_DEFAULT, cd.EL_DEFAULT)))
    text = cd.elevation_text(base, lang)
    other = [(i, w) for i, w in enumerate(wins) if w != base]
    if not other:
        return text
    if lang == "zh":
        return text + "（" + "；".join(f"K{i + 1}：{cd.fmt_deg(w[0])} 至 {cd.fmt_deg(w[1])}" for i, w in other) + "）"
    return text + " (" + "; ".join(f"K{i + 1}: {cd.fmt_deg(w[0])} to {cd.fmt_deg(w[1])}" for i, w in other) + ")"


def wrap_text(fig, text: str, width_in: float, fs: float) -> str:
    """Word-wrap each line of ``text`` to ``width_in`` at ``fs`` (a word longer than the width stays whole)."""
    out = []
    for line in text.split("\n"):
        cur = ""
        for word in line.split(" "):
            trial = word if not cur else cur + " " + word
            if cur and cd.text_width_pt(fig, trial, fs) / 72.0 > width_in:
                out.append(cur)
                cur = word
            else:
                cur = trial
        out.append(cur)
    return "\n".join(out)


def _choose_scale_corner(insets, recs) -> Tuple[Optional[float], List[str]]:
    """``scale_corner="fixed"``: the inset corner with the most clearance from the badges in every block."""
    return _choose_scale_corner_from([inset_corner_clearance(ax, r, show_arrow=(n == 0))
                                      for n, (ax, r) in enumerate(zip(insets, recs))])


def _choose_scale_corner_from(clear: Sequence[Dict[float, float]]) -> Tuple[Optional[float], List[str]]:
    """The corner (plot angle) free in every block with the most clearance from the badges (``inset_corner_clearance``
    per block), and a warning when that clearance is under 12 deg."""
    common = [c for c in (45.0, 135.0, 315.0, 225.0) if all(c in cl for cl in clear)]
    if not common:
        return None, []
    best = max(common, key=lambda c: (round(min(cl[c] for cl in clear)), -common.index(c)))
    worst = min(cl[best] for cl in clear)
    warn = [] if worst >= 12.0 else [f"inset scale bars: corner {best:g} deg is {worst:.0f} deg from a badge"]
    return best, warn


def make_case_figure(dump_npz_path, rows: Optional[List[int]] = None, topdown_root=None, clip_root_override=None,
                     out_stem="case", lang: str = "en", options: Optional[CaseOptions] = None,
                     title: Optional[str] = None, title_note: Optional[str] = None,
                     caption_extra: Optional[str] = None, key_rule: Optional[str] = None) -> dict:
    """Render the case figure for one dump.

    ``title`` / ``title_note`` / ``caption_extra`` / ``key_rule`` override the
    same :class:`CaseOptions` fields.  Returns {"files", "rows", "stats",
    "size_in", "windows", "warnings", "layout", "notes_dropped"}: ``warnings``
    lists what a reviewer should know (widened windows, wrapped notes, miss
    badges moved under a row, a crowded legend); ``notes_dropped`` is always
    empty (a slot without a badge or a note raises instead, D5).
    """
    cd.setup(lang)
    import dataclasses

    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    arm = ARM
    opt = dataclasses.replace(options) if options is not None else CaseOptions()
    for name, val in (("title", title), ("title_note", title_note), ("caption_extra", caption_extra),
                      ("key_rule", key_rule)):
        if val is not None:
            setattr(opt, name, val)
    L = labels_for(lang)
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
    roles = list(opt.roles) if opt.roles is not None else [None] * len(recs)
    del _DROPPED[:]

    # ---- every block planned before the page exists (heights depend on windows, notes, miss lanes)
    fig_m = plt.figure(figsize=(FIG_W, 2.0))
    plans = [plan_block(fig_m, n, r, L, dump.frame_count, roles[n], arm) for n, r in enumerate(recs)]
    plt.close(fig_m)
    warnings: List[str] = [w for p in plans for w in p.warnings]

    title_h = TITLE_H if opt.title else (BANNER_H if opt.banner else 0.0)
    blocks_h = sum(p.height for p in plans) + (len(plans) - 1) * BLOCK_GAP
    height = TOP_H + blocks_h + AXIS_H + LEGEND_H + title_h
    fig = plt.figure(figsize=(FIG_W, height))
    page = Page(fig, height, y0=title_h)
    if opt.title:
        t = page.text(X_ROUTE, -title_h + 0.11, opt.title, ha="left", va="center", fontsize=FS["title"] + 0.8,
                      fontweight="bold", color=style.INK, path_effects=cd.bold_effects(opt.title, style.INK))
        if opt.title_note:
            w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            page.text(X_ROUTE + w + 0.12, -title_h + 0.11, opt.title_note, ha="left", va="center",
                      fontsize=FS["name"], color=style.INK_2, fontstyle="italic" if lang != "zh" else "normal")
    elif opt.banner:
        page.text(X_ROUTE, -title_h + 0.09, opt.banner, ha="left", va="center", fontsize=FS["name"],
                  color=style.INK_2, fontstyle="italic" if (opt.banner_italic and lang != "zh") else "normal")
    draw_top_band(page, L)
    insets = []
    keepout_artists: list = []
    y = TOP_H
    for n, (plan, r) in enumerate(zip(plans, recs)):
        views = dd.surround_views(clip_dir, r.frame)
        ax_in, numbered = draw_block(page, y, n, plan, views, arm, L, last=(n == len(recs) - 1), role=roles[n],
                                     frame_count=dump.frame_count, keepout=keepout_artists)
        y_lane = y + plan.top_h
        insets.append((ax_in, page.bbox(X_INSET, y_lane, W_INSET, plan.body_h)))
        badge_slots = [k for g in r.groups for k in g]
        note_slots = [k for it in plan.items for k in it["slots"]]
        in_notes = [k for it in plan.items if it["kind"] == "predicted_none" for k in it["slots"]]
        dd.check_accounting(r, arm, badge_slots, note_slots, numbered, in_notes)
        y += plan.height + BLOCK_GAP
    y_axis = y - BLOCK_GAP

    # ---- column a: route panel (fitted to the route), scene note, optional foot
    y_route = TOP_H + HDR_H
    scene_h = 0.20
    ep = dump.episode_id
    if ep.startswith(f"exp18E_{dump.scene}_"):  # designed routes: "exp18E_<scene>_loop1" -> "loop1"
        ep = ep[len(f"exp18E_{dump.scene}_"):]
    scene_text = opt.scene_text if opt.scene_text is not None else L["scene"].format(
        tier=dump.tier_name(lang), scene=dump.scene, ep=ep, T=dump.frame_count)
    scene_text = wrap_text(fig, scene_text, W_ROUTE, FS["note"])
    scene_h = max(scene_h, 0.095 * (scene_text.count("\n") + 1) + 0.01)
    panel = opt.route_panel
    if opt.route_fit and panel is None:
        panel = FittedRoutePanel(start_text=L["start"])
    layout: dict = {"windows": [list(p.win) for p in plans], "note_lines": [len(p.note_lines) for p in plans],
                    "header_lines": [p.hdr_lines for p in plans], "miss_lane": [p.below for p in plans],
                    "label_panel": "gutter"}
    col_a = []  # display boxes of what column a draws (the insets' rim badges keep clear of them)
    if opt.route_fit and hasattr(panel, "height"):
        avail = y_axis + AXIS_H - y_route - scene_h
        foot = opt.route_foot if opt.route_foot is not None and opt.route_foot_h > 0 else None
        foot_min = opt.route_foot_h if foot is not None else 0.0
        h_route = panel.height(fig, dump, W_ROUTE, avail - foot_min)
        ax_route = page.ax(X_ROUTE, y_route, W_ROUTE, h_route)
        panel(ax_route, level, dump, recs, L)
        t_scene = page.text(X_ROUTE, y_route + h_route + 0.035, scene_text, ha="left", va="top", fontsize=FS["note"],
                            color=style.MUTED, linespacing=1.15)
        col_a += [page.bbox(X_ROUTE, y_route, W_ROUTE, h_route), t_scene]
        if foot is not None:
            foot_h = avail - h_route
            foot_max = getattr(foot, "max_h", None)
            if foot_max is not None:
                foot_h = min(foot_h, foot_max)
            foot(page, X_ROUTE, y_route + h_route + scene_h, W_ROUTE, foot_h)
            col_a.append(page.bbox(X_ROUTE, y_route + h_route + scene_h, W_ROUTE, foot_h))
            layout["foot_h"] = round(foot_h, 3)
        layout["route_h"] = round(h_route, 3)
        layout["route_scale_bar_inset_pt"] = getattr(panel, "scale_bar_inset_pt", None)
        if layout["route_scale_bar_inset_pt"] is not None and layout["route_scale_bar_inset_pt"] < 2.0:
            warnings.append(f"route scale bar only {layout['route_scale_bar_inset_pt']} pt inside the map")
    else:
        y_bottom = y_axis + AXIS_H - opt.route_foot_h
        ax_route = page.ax(X_ROUTE, y_route, W_ROUTE, y_bottom - scene_h - y_route)
        (panel or draw_route_panel)(ax_route, level, dump, recs, L)
        t_scene = page.text(X_ROUTE, y_bottom - scene_h + 0.035, scene_text, ha="left", va="top", fontsize=FS["note"],
                            color=style.MUTED, linespacing=1.15)
        col_a += [page.bbox(X_ROUTE, y_route, W_ROUTE, y_bottom - y_route)]
        if opt.route_foot is not None and opt.route_foot_h > 0:
            opt.route_foot(page, X_ROUTE, y_bottom, W_ROUTE, opt.route_foot_h)
            col_a.append(page.bbox(X_ROUTE, y_bottom, W_ROUTE, opt.route_foot_h))

    # ---- panel b: each disc as large as its rim badges allow beside the row names and column a
    rend = fig.canvas.get_renderer()
    keepout = [a if hasattr(a, "x0") else a.get_window_extent(rend) for a in keepout_artists + col_a]
    fits = []
    for n, ((ax_in, _), r) in enumerate(zip(insets, recs)):
        rf, dx, ok = fit_inset(ax_in, r, keepout)
        fits.append((rf, dx))
        if not ok:
            warnings.append(f"K{n + 1}: a rim badge of the local map touches a row name or column a")
    corner = None
    if opt.scale_corner == "fixed":
        clear = [inset_corner_clearance(ax, r, show_arrow=(n == 0), radius_frac=rf)
                 for n, ((ax, _), r, (rf, _)) in enumerate(zip(insets, recs, fits))]
        corner, w = _choose_scale_corner_from(clear)
        warnings += w
    for n, ((ax_in, bounds), r, (rf, _)) in enumerate(zip(insets, recs, fits)):
        lvl = dd.topdown_level(dump, float(r.cur_pos[1]), root=topdown_root)
        draw_inset(ax_in, lvl, dump, r, show_arrow=(n == 0), L=L, split_frame=opt.split_frame,
                   letters=opt.letters, scale_corner=corner, radius_frac=rf, letter_bounds=bounds)
    layout["scale_corner"] = corner
    layout["inset_radius_frac"] = [rf for rf, _ in fits]
    layout["inset_shift_pt"] = [dx for _, dx in fits]
    warnings += draw_legend(page, y_axis + AXIS_H, arm, L)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)
    caption = opt.caption if opt.caption is not None else case_caption(
        lang, opt, n=len(recs), tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id)
    caption = caption.replace("{elev_window}", window_caption([p.win for p in plans], lang))
    if opt.caption_extra:
        caption = caption.rstrip() + " " + opt.caption_extra
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption + "\n", encoding="utf-8")
    files.append(cap_path)
    stats = []
    for n, (r, plan) in enumerate(zip(recs, plans)):
        entry = {"key": f"K{n + 1}", "row": r.index, "frame": r.frame, "window": list(plan.win),
                 "misses": [k + 1 for k in r.misses(arm)]}
        for a in list(r.arms) + ["floor"]:
            entry[a] = r.summary(a)
        stats.append(entry)
    return {"files": [str(f) for f in files], "rows": rows, "stats": stats, "size_in": (FIG_W, height),
            "windows": [list(p.win) for p in plans], "warnings": warnings, "layout": layout, "notes_dropped": []}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True, help="History Head dump npz of one clip")
    ap.add_argument("--rows", default=None, help="comma-separated query rows (default: pre-registered key rows)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--out", default="case", help="output stem (writes .pdf, .png, _caption.txt)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    ap.add_argument("--title", default=None)
    ap.add_argument("--title-note", default=None)
    ap.add_argument("--caption-extra", default=None)
    ap.add_argument("--revised", action="store_true", help="accepted for compatibility (the layout is always revised)")
    args = ap.parse_args(argv)
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    res = make_case_figure(args.dump, rows=rows, topdown_root=args.topdown_root, clip_root_override=args.clip_root,
                           out_stem=args.out, lang=args.lang, title=args.title, title_note=args.title_note,
                           caption_extra=args.caption_extra)
    for f in res["files"]:
        print(f)
    for s in res["stats"]:
        print(s)
    print("size_in", tuple(round(v, 3) for v in res["size_in"]))
    print("layout", res["layout"])
    for w in res["warnings"]:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
