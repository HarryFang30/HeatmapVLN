#!/usr/bin/env python3
"""EXP-18 supplementary animation: the predicted affordance map vs ground truth along one episode.

The video starts on the first query row.  One video frame per scored query row
(cache endpoints: every 8 frames and the last frame) is held for ``hold_s``
seconds; between two query rows the robot glides along the route (one step per
clip frame, ``substeps`` video frames per step, linear in position and
shortest-arc in facing direction).  While it moves, the route and the local map
follow it live (the map turns with the robot and zooms smoothly, log-linear with
ease-in-out, from one query's scale to the next one's), the last query's strip
and numbers stay on screen faded, the info column says "showing query i of n ·
frame t of T (faded)", and the next query is named under the timeline.  The
affordance maps are never interpolated: every map shown is one query row's.

Frame layout (16:9, drawn on an 8.0 x 4.5 in page; 1920 x 1080 = 240 dpi):

  Predicted affordance map vs ground truth                  tier scene · episode · frames
  Route                    Local map (robot facing up)      frame t of T
  +-------------------+    ( disc, rim = bearing )          query i of n
  | top-down map      |                                     |--|--|--v--|--|  timeline, next query under it
  | route: dark = done|                                     metrics of the row, reference, running totals
  | robot, past pos.  |                                     legend (2 columns)
  +-------------------+
                 [ bracket: shown for reference only (not given to the model) ]
                 Front · model input | Right | Back | Left
                 note lines (only as many as the episode's rows need)
                 lane: numbered blue badges at the true bearings (+ notes left of them)
  query i of n   RGB row (front in colour and framed; the others washed out)
  frame t of T
  ground truth ▬ ground-truth affordance map row (blue)        } the query's own elevation window
  prediction   ▬ predicted affordance map row (orange), x = predicted peaks, numbered misses
                 (lane under the row for miss badges, only when a row needs it)
  elevation ..   0° (straight ahead)   −90°   180°   +90°

The visual language is ``fig_case``'s (see its module doc and ``common_draw``),
through the shared functions, so the video and fig1 follow one set of rules:

* **D1 misses**: a slot is missed iff it fails joint PCK@8
  (``data.CaseRow.misses``, the fields and rule of ``compute_metrics``); the
  numbered misses of a row (on the prediction row + in notes) equal the
  header's n - hits, checked by ``data.check_accounting`` (raises).
* **D2**: a miss number is ink in a white disc ringed in orange
  (``cd.miss_badge``), beside its own x or moved and joined to it by a short
  ink leader (``cd.place_miss_labels`` / ``cd.draw_miss_labels``); misses whose
  peaks coincide share one x and badge; a dotted connector joins the x to the
  slot's true-bearing tick when < 45 deg apart (``cd.draw_miss_connectors``).
* **D3**: "ground truth" / "prediction" in the fixed left gutter with a colour
  key (``cd.gutter_row_label``).
* **D4**: each held query shows one row, as one block of fig1 does, so its
  rows use that row's own window (``cd.elevation_window([row])``: +-10 deg,
  widened just enough for every GT-visible and predicted peak, at most +-45
  deg): the same window, and the same blob shapes (square degrees), as fig1's
  block of that row.  The page is laid out once (``_geometry``): the strip at
  full width, room below the images for the tallest query's rows, the top band
  never under ``H_TOP_MIN``; a query whose rows still do not fit is drawn
  squeezed vertically (``_RowPlan.v``, laid out in drawn units by
  ``_scaled_row``; a warning).  Each query's window is printed left of the
  degree labels (with the squeeze factor, if any), and the caption states the
  windows (``_elev_caption``).
* **D5**: notes via ``fig_case.note_items`` (runs of slots merged into ranges,
  P(not visible) wording), left of the lane badges when they fit, else on
  note lines above the lane (``cd.wrap_notes``); the layout reserves the most
  note lines any row needs.  Nothing is dropped; ``make_animation`` returns a
  ``warnings`` list.
* **D6**: frame labels ``cd.frame_label`` ("frame 20 of 73", 1-based) in the
  info column, the strip's gutter and the timeline; fig_case's wording for the
  notes, baseline, input bracket, panel b and the axis.

Rim badges on the local map: dodged along the rim in bearing order with a
thin leader to the true bearing (``cd.disc_rim_labels``; ``_draw_inset`` draws
``fig_case.draw_inset``'s panel).  Past positions whose bearings (nearly)
coincide keep slot order in the lane and on the rim alike (``_ordered_x``).
The dodge is capped: when it would move a badge more than ``RIM_CAP_DEG`` from
its bearing, past positions within ``RIM_MERGE_DEG`` of one direction share one
badge ("3–8", ``_merge_rim_groups``; typical when the robot drove straight),
and if the dodge is still over the cap the badges alternate between two
staggered rings, which halves the spread.  The local map's scale bar stays in
one corner of the inset square, outside the disc, in every frame of the video
(held and moving; only its length and label follow the zoom): bottom left,
unless some query's rim badges, leaders or letters would come within
``BAR_CLEAR_PT`` of it there, then the first clear one of bottom right and top
right (``_Context._pin_bar_corner``; top left holds the direction arrow).

The caption (``CAPTION_PARTS``) explains only what the animation shows: merged
rim numbers, two rings, shared miss numbers, dotted lines and notes are
described when some query has them (``_caption_flags``).

Miss badges: ``cd.place_miss_labels``' candidates and cost, placed by a small
search (``_place_search``) that also counts leaders crossing other leaders or
the dotted connectors, and treats a leader grazing another x as crossing it
(``LEADER_CLEAR_PT``).

Figure policy (user decision): the animation shows the affordance map only.
It never mentions poses, their sources or the pose-source ablation; the
prediction drawn is always the deployed model's output (the dump's ``vo`` arm,
as in ``fig_case.ARM``).  Honesty kept on every frame: of the four current
views only the framed front view is marked as model input, the other three are
bracketed "shown for reference only"; ground truth blue vs prediction orange,
never swapped; misses drawn and numbered; the constant always-behind guess is
printed as a reference.

Encoding: H.264 MP4 (yuv420p, BT.709 tagged, +faststart) through PyAV
(``av``, bundled with its own FFmpeg + libx264, so it works in a blank
container), else an ``ffmpeg`` executable (imageio-ffmpeg's, ``$PATH`` or
``/opt/conda/bin/ffmpeg``).  A looping GIF (``gif_width`` px wide, default
1280, i.e. 1280 x 720, so badge digits and small notes stay legible; one frame
per row and per clip step, one palette per segment, no dithering) is written
with Pillow.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_anim --dump <clip.npz> --out <dir/stem> [--fps 15] [--lang en|zh]
      [--size 1920x1080] [--hold 2.0] [--substeps 2] [--rows 0,1,2] [--gif-width 1280] [--no-gif]
      [--stills] [--stills-only] [--topdown-root DIR] [--clip-root DIR]
Writes <stem>.mp4, <stem>.gif (unless --no-gif), <stem>_caption.txt and with
--stills one PNG per query row (<stem>_q<ii>_f<frame>.png, full resolution);
--stills-only writes the query-row PNGs and one mid-move PNG per gap, no video.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import fig_case as fc
from scripts.exp18.figures import style
from scripts.exp18.topdown.topdown_io import forward_from_c2w, load_topdown

from matplotlib.lines import Line2D  # noqa: E402  (matplotlib is configured in cd.setup)
from matplotlib.patches import Rectangle  # noqa: E402

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); shared wording comes from fig_case.LABELS
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Predicted affordance map vs ground truth",
        "sub": "{tier} {scene}  ·  episode {ep}  ·  {T} frames",
        "query": "query {i} of {n}",
        "query_sched": "query {i} of {n}  (every 8 frames and the last frame)",
        "showing": "showing query {i} of {n} · {frame} (faded)",
        "next": "next query: {frame}",
        "strip_query": "query {i} of {n}",
        "err": "bearing error  median {med:.1f}°, max {mx:.1f}°",
        "pck": "PCK@8  {hits}/{n} past positions",
        "none_visible": "no past position visible in any view",
        "floor": "always-behind guess: median {med:.0f}°, PCK@8 {hits}/{n}",
        "so_far": "episode so far: median {med:.1f}°, PCK@8 {hits}/{n}",
        "legend_query": "query position",
        "elev": ("elevation {lo} to {hi}", "elev. {lo} to {hi}", "{lo} to {hi}"),  # the first that fits the gutter
        "squeezed": "squeezed ×{v:.2f}",
    },
    "zh": {
        "title": "预测 affordance map 与真值对比",
        "sub": "{tier} {scene}  ·  第 {ep} 集  ·  {T} 帧",
        "query": "第 {i} 次查询（共 {n} 次）",
        "query_sched": "第 {i} 次查询（共 {n} 次，每 8 帧及末帧）",
        "showing": "淡化显示：第 {i} 次查询 · {frame}",
        "next": "下一次查询：{frame}",
        "strip_query": "第 {i} 次查询",
        "err": "方位误差  中位 {med:.1f}°，最大 {mx:.1f}°",
        "pck": "PCK@8  {hits}/{n} 个历史位置",
        "none_visible": "任何视角都看不到历史位置",
        "floor": "恒答正后方：中位 {med:.0f}°，PCK@8 {hits}/{n}",
        "so_far": "本集累计：中位 {med:.1f}°，PCK@8 {hits}/{n}",
        "legend_query": "查询位置",
        "elev": ("仰角 {lo} 至 {hi}", "仰角 {lo}–{hi}", "{lo}–{hi}"),
        "squeezed": "纵向压缩 ×{v:.2f}",
    },
}

CAPTION_SCHEDULE = {  # which query frames the animation holds: every endpoint / those with an output / a pick
    "en": {"full": ": every 8 frames and the last frame",
           "gaps": ": every 8 frames and the last frame, where the model has an output",
           "subset": ", a subset of the query frames chosen for this animation"},
    "zh": {"full": "：每 8 帧及末帧", "gaps": "：每 8 帧及末帧中模型有输出的帧", "subset": "，本动画只取了部分查询帧"},
}
# Caption parts; ``_caption`` joins them and leaves out the ones describing something the animation does not show
# (a caption that explains merged rim numbers, two rings, notes or dotted lines the viewer never sees only
# confuses).  Conditions: see ``_caption_flags``.
CAPTION_PARTS = {
    "en": {
        "head": ("Supplementary animation: predicted affordance map vs ground truth along one episode ({tier} {scene}, "
                 "episode {ep}; {n} query frames{sched}; frames are counted from 1). The video starts at the first "
                 "query, {first}, and holds each query. Between queries the robot moves along the route and the local "
                 "map follows it (turning with the robot and zooming smoothly to the next query's scale), while the "
                 "last query's strip and numbers stay on screen, faded and labelled with their frame, because "
                 "predictions exist only at query frames; the timeline names the next query. "),
        "left": ("Left: route on the top-down map (dark = driven so far, open circles = query positions, blue dots = "
                 "the past positions of the query on screen, light = older). "),
        "middle": ("Middle: map around the robot, robot facing up; its rim is the bearing ring that the strip below "
                   "unrolls clockwise from the front view's left edge (arrow); dashed radii are the view seams. Blue "
                   "lines run from the robot through each past position (dot; 1 = oldest of the 8 queried) to its "
                   "number on the rim, joined to its bearing by a thin leader where it had to move"),
        "middle_merge": ("; where many past positions lie in one direction, those within 6° of one another share one "
                         "number on the rim (e.g. 3–8 = past positions 3 to 8)"),
        "middle_rings": ", and where the numbers still crowd they alternate between two rings",
        "middle_end": ". ",
        "bottom": ("Bottom: the surround view on the same bearings: of the four current views, only the framed front "
                   "view is given to the model; right, back and left are shown for reference (the model also receives "
                   "the past frames' front images). Below it, the ground-truth affordance map (blue) and the predicted "
                   "affordance map (orange; the deployed model's output); {elev}; the images span ±15°. Each slot's "
                   "map is divided by its own peak (the prediction also multiplied by its predicted visibility), the "
                   "maximum over slots is shown, and colour is linear in that value in both rows. Maps stop at view "
                   "seams because every label and prediction lives in one 90° view. "),
        "groups": "A range such as 1–3 names past positions at one spot (e.g. where the robot turned on the spot). ",
        "x": "×: predicted peak of each slot visible in the ground truth (hits within 4° share one ×",
        "x_miss_share": "; missed slots whose peaks coincide share one × and one number, e.g. 4–5",
        "x_end": "). ",
        "misses": ("A slot is missed when it fails joint PCK@8 (predicted view wrong, or peak more than 8 px of 64 "
                   "from the true peak in that view); its number, in a white disc ringed in orange, sits at its own × "
                   "(joined to it by a short dark leader where it had to move)"),
        "miss_conn": ", and a dotted line joins the × to the slot's true bearing when they are less than 45° apart",
        "miss_conn_dropped": (" (left out in a query whose misses crowd together, where the lines would push the "
                              "numbers away from their × or cross their leaders)"),
        "misses_end": ". ",
        "miss_none": ("A missed slot the model calls not visible (P(not visible) > 0.5) has no ×; its orange-ringed "
                      "number is in the notes. "),
        "miss_count": "Each query's numbered misses equal its visible past positions minus its PCK@8 hits. ",
        "ticks": "Blue ticks under the prediction repeat the true bearings. ",
        "notes": ("Notes above the images list past positions that no view shows (the previous frame at the robot, "
                  "or out of sight), with the predicted P(not visible). "),
        "right": ("Right: bearing error of the predicted peaks over visible past positions (median, max), joint "
                  "PCK@8, the constant always-behind guess (back view, centre pixel) for reference, and the running "
                  "totals of the episode."),
    },
    "zh": {
        "head": ("补充动画：同一集（{tier} {scene}，第 {ep} 集；{n} 个查询帧{sched}；帧号从 1 数起）上预测 affordance map 与"
                 "真值的对比。视频从第一次查询、即{first}开始，每次查询停留片刻。两次查询之间机器人沿路线移动，局部地图随之"
                 "转动，并平滑缩放到下一次查询的比例；上一次查询的条带与数值留在画面上、变淡并标出其帧号，因为只有查询帧才有"
                 "预测；时间轴下方标出下一次查询。"),
        "left": "左：俯视图上的路线（深色 = 已走过，空心圆 = 查询位置，蓝点 = 当前画面那次查询的历史位置，浅 = 更早）。",
        "middle": ("中：机器人周围的局部地图，机器人朝上；圆周就是下方条带从前视左缘顺时针展开的方位环（箭头），虚线半径为视角"
                   "分界。蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号，挪开的编号用细线连到其方位"),
        "middle_merge": "；许多历史位置方向相同时，方位相差 6° 以内的在圆周上合用一个编号（如 3–8 即历史位置 3 至 8）",
        "middle_rings": "，仍然拥挤时编号交替排在内外两圈",
        "middle_end": "。",
        "bottom": ("下：同一方位轴上的环视：当前四个视角中只有加框的前视图输入模型，右/后/左仅作展示（模型另外还接收历史帧的"
                   "前视图）。其下为真值 affordance map（蓝）与预测 affordance map（橙，即部署模型的输出）；{elev}；环视图为 "
                   "±15°。每个槽位的图除以自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值线性着色。图在"
                   "视角分界处截断，因为每个标签和预测都只落在一个 90° 视角里。"),
        "groups": "1–3 这样的范围表示位于同一处的历史位置（如机器人原地转向时）。",
        "x": "×：真值可见的各槽位的预测峰值（命中的槽位 4° 内共用一个 ×",
        "x_miss_share": "；峰值重合的未命中槽位共用一个 × 和一个编号，如 4–5",
        "x_end": "）。",
        "misses": ("joint PCK@8 不通过（预测视角错误，或峰值在该视角中距真值峰值超过 8 px，视图宽 64 px）即为未命中：其编号"
                   "写在橙色描边的白色圆内，放在对应的 × 旁（需要挪开时用深色短线相连）"),
        "miss_conn": "，两者相距 45° 以内时再用虚线把 × 连到该槽位的真值方位",
        "miss_conn_dropped": "（某次查询的未命中挤在一起、虚线会把编号挤离其 × 或与引线交叉时，该次查询不画虚线）",
        "misses_end": "。",
        "miss_none": "模型判为不可见（预测不可见概率 > 0.5）的未命中槽位没有 ×，其橙色描边编号列在注释中。",
        "miss_count": "每次查询的编号未命中数等于可见历史位置数减去 PCK@8 命中数。",
        "ticks": "预测行下方的蓝色短线重复真值方位。",
        "notes": "图像上方的注释列出任何视角都看不到的历史位置（与机器人重合的上一帧，或视线之外），并给出预测不可见概率。",
        "right": ("右：可见历史位置上预测峰值的方位误差（中位、最大）、joint PCK@8、作参照的恒答正后方基线（后视中心像素），"
                  "以及本集累计。"),
    },
}
ELEV_CAPTION = {  # the rows' elevation windows (D4, per query), stated in the caption as fig1 states its blocks'
    "en": {"all": "rows show elevation {lo} to {hi} (printed left of the degree labels)",
           "mixed": ("rows show elevation {lo} to {hi}, widened just enough for a query whose peaks lie higher or "
                     "lower ({items}); each query's window is printed left of the degree labels"),
           "item": "query {i}: {lo} to {hi}",
           "item_sq": "query {i}: {lo} to {hi}, squeezed vertically ×{v:.2f} to fit the page", "sep": "; "},
    "zh": {"all": "各行显示仰角 {lo} 至 {hi}（标在度数标注左侧）",
           "mixed": "各行显示仰角 {lo} 至 {hi}，峰值更高或更低的查询恰好放宽到包含其峰值（{items}）；每次查询的窗口标在度数标注左侧",
           "item": "第 {i} 次查询：{lo} 至 {hi}",
           "item_sq": "第 {i} 次查询：{lo} 至 {hi}，纵向压缩为 {v:.2f} 倍以放入画面", "sep": "；"},
}
CAPTION_ORDER = (("head", None), ("left", None), ("middle", None), ("middle_merge", "rim_merged"),
                 ("middle_rings", "rim_two_rings"), ("middle_end", None), ("bottom", None), ("groups", "groups"),
                 ("x", None), ("x_miss_share", "shared_miss"), ("x_end", None), ("misses", None),
                 ("miss_conn", "dotted"), ("miss_conn_dropped", "dotted_dropped"), ("misses_end", None),
                 ("miss_none", "miss_in_notes"), ("miss_count", None), ("ticks", None), ("notes", "notes"),
                 ("right", None))


ARM = fc.ARM  # the deployed model's output; the only prediction ever drawn

# --------------------------------------------------------------------------- #
# Page geometry (inches, origin top-left) on an 8.0 x 4.5 in page
# --------------------------------------------------------------------------- #
FIG_W, FIG_H = 8.0, 4.5
MARGIN = 0.10
Y_TITLE = 0.20
Y_NAMES = 0.46
Y_TOP = 0.55  # top of the route / disc / info band
X_STRIP_MIN = 1.02  # left of the strip: the gutter (row names, the strip's query and frame); the same in en
# and zh, so both languages lay the strip (and its miss badges) out identically
EL_RGB = fc.EL_RGB  # the images: +-15 deg, as in fig1
EL_HEAT = cd.EL_DEFAULT  # default half window of the affordance map rows (D4: widened by cd.elevation_window)
ROW_GAP = 0.035
H_LANE = 0.17
NOTE_LINE_H = 0.15
AXIS_H = 0.24  # degree labels under the prediction row (or under its miss lane)
MISS_LANE_H = cd.BELOW_LANE_H_PT / 72.0
H_TOP_MIN = 1.45  # the route / disc / info band never gets smaller (a query's rows are squeezed instead)
V_MIN = 0.25  # least vertical scale of a squeezed query's rows (a warning below it)
W_ROUTE = 2.80
HEAT_RING_W = 1440
MARK_PT = cd.MARK_PT  # predicted-peak x, as in fig1
RIM_CAP_DEG = 24.0  # a rim badge dodged farther than this from its bearing: merge by direction, then two rings
RIM_MERGE_DEG = 6.0  # (only past the cap) consecutive slots whose bearings span at most this share one rim badge
BAR_CORNERS = {225.0: "bottom-left", 315.0: "bottom-right", 45.0: "top-right"}  # the local map's scale bar:
# one corner of the inset square for the whole video, the first of these (in this order) that stays BAR_CLEAR_PT
# clear of every query's rim badges, leaders and letters, else the clearest (top-left holds the direction arrow)
BAR_CLEAR_PT = 2.0
BAR_EDGE_PT = {-1: 1.0, 1: 6.0}  # the bar's outer end this far inside the square's left / right edge: a right-hand
# bar keeps clear of the info column's legend and text, so it reads as the map's (it would otherwise sit nearer them)
TIE_DEG, TIE_STEP = 0.25, 0.01  # badge groups whose true bearings lie within TIE_DEG: slot order (_ordered_x)
PROX_LEADER_PT, PROX_MARGIN_PT = 6.0, 1.5  # _unclear_badges: a leader this long names its x; else the own x
# must be clearly nearer than any other
LEADER_CLEAR_PT = 0.6  # a miss badge's leader keeps this far from every other x / badge box (None: the shared
# rule only, which lets a leader graze another x); its first GRAZE_SKIP_PT (at its own x) are not tested
GRAZE_SKIP_PT = 1.2
SEARCH_BUDGET = 150  # badge placements each search of _place_search may try per row layout
TOPK, TOPK_SEP_PT = 3, 3.0  # _place_search also tries each badge's 3 cheapest spots at least 3 pt apart
FS = {"title": 11.0, "sub": 7.0, "name": 7.4, "frame": 9.6, "info": 7.0, "muted": 6.5, "legend": 6.6,
      "rowlab": 7.0, "bracket": 6.4, "note": 6.2, "axis": 6.6, "views": 7.2, "next": 6.6,
      "gutter": 7.0, "gutter_sub": 6.2, "elev": 6.0}
ROUTE_AHEAD = "#b9b7ae"  # the part of the route still ahead
GT_INK = style.GT_INK  # dark tones of the ground-truth / prediction ramps (text, keys)
PRED_INK = style.PRED_INK
FADE_ALPHA = 0.66  # white veil over the last query's strip and numbers while the robot moves
ROBOT_PT = 8.6
DISC_ROBOT_PT = 6.6  # as fig_case.draw_inset
RADIUS_FRAC = fc.INSET_RADIUS_FRAC  # disc radius / axes half
# info column, from the frame line (Y_NAMES) down: query line, timeline, metrics; the legend sits at the band's foot
INFO_QUERY_DY = 0.18
TL_DY, TL_H = 0.24, 0.30
METRICS_GAP, METRICS_STEP = 0.10, 0.14
LEG_ROW_H = 0.125


def _ax(fig, x: float, y_top: float, w: float, h: float, **kw):
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H], **kw)


def _text(fig, x: float, y: float, s: str, bold: bool = False, **kw):
    """Figure text at page inches (top-left origin); ``bold``: Latin bold, CJK regular weight (``cd.bold_effects``)."""
    if bold:
        kw.setdefault("fontweight", cd.bold_weight(s))
        eff = cd.bold_effects(s, kw.get("color", style.INK))
        if eff:
            kw.setdefault("path_effects", eff)
    return fig.text(x / FIG_W, 1 - y / FIG_H, s, **kw)


def _rect(fig, x0: float, y0: float, x1: float, y1: float, **kw) -> Rectangle:
    """Figure-level rectangle given in page inches (top-left origin)."""
    r = Rectangle((x0 / FIG_W, 1 - y1 / FIG_H), (x1 - x0) / FIG_W, (y1 - y0) / FIG_H, transform=fig.transFigure,
                  **kw)
    fig.add_artist(r)
    return r


# --------------------------------------------------------------------------- #
# Layout: one page geometry for the whole animation (the most note lines any query needs, room for its tallest
# rows); each query's affordance map rows use that query's own D4 window (``_RowPlan.win``)
# --------------------------------------------------------------------------- #
@dataclass
class _Layout:
    note_lines: int
    x_strip: float
    w_strip: float
    h_rgb: float = 0.0
    y_gt: float = 0.0  # top of the ground-truth row (the rows below it are per query)
    y_rgb: float = 0.0
    y_lane: float = 0.0
    y_notes: float = 0.0
    y_views: float = 0.0
    y_bracket: float = 0.0
    h_top: float = 0.0
    x_disc: float = 0.0
    w_disc: float = 0.0
    x_info: float = 0.0
    w_info: float = 0.0

    @property
    def ppd(self) -> float:
        """Points per degree along the strip (and up the rows of a query drawn at vertical scale 1)."""
        return self.w_strip * 72.0 / 360.0

    @property
    def rows_room(self) -> float:
        """Height (in) from the top of the ground-truth row to the foot of the page."""
        return FIG_H - self.y_gt


def _rows_need(h_heat: float, below: bool) -> float:
    """Height (in) a query's rows take below ``_Layout.y_gt``: both rows, their gap, the miss lane, the axis."""
    return 2 * h_heat + ROW_GAP + (MISS_LANE_H if below else 0.0) + AXIS_H


def _geometry(note_lines: int, rows_need: float, gutter_right: float) -> _Layout:
    """The page for ``note_lines`` note lines and rows needing ``rows_need`` in: the strip at full width, the top
    band (route, disc, info) as tall as the rest leaves, but never under ``H_TOP_MIN`` (a query whose rows then do
    not fit is drawn squeezed vertically, ``_Context``)."""
    x_strip = max(X_STRIP_MIN, gutter_right)
    w = FIG_W - MARGIN - x_strip
    lay = _Layout(note_lines=note_lines, x_strip=x_strip, w_strip=w)
    lay.h_rgb = w * 2 * EL_RGB / 360.0
    above = 0.07 + 0.155 + 0.075 + note_lines * NOTE_LINE_H + H_LANE + lay.h_rgb + ROW_GAP  # band foot -> y_gt
    lay.h_top = max(H_TOP_MIN, FIG_H - rows_need - above - Y_TOP)
    lay.y_bracket = Y_TOP + lay.h_top + 0.07
    lay.y_views = lay.y_bracket + 0.155
    lay.y_notes = lay.y_views + 0.075
    lay.y_lane = lay.y_notes + note_lines * NOTE_LINE_H
    lay.y_rgb = lay.y_lane + H_LANE
    lay.y_gt = lay.y_rgb + lay.h_rgb + ROW_GAP
    lay.x_disc = MARGIN + W_ROUTE + 0.12
    lay.w_disc = lay.h_top
    lay.x_info = lay.x_disc + lay.w_disc + 0.13
    lay.w_info = FIG_W - MARGIN - lay.x_info
    return lay


# --------------------------------------------------------------------------- #
# Per-row plan (decided before any frame is drawn: the layout depends on it)
# --------------------------------------------------------------------------- #
@dataclass
class _RowPlan:
    r: dd.CaseRow
    gt_strip: np.ndarray
    pr_strip: np.ndarray
    marks: List[dict]
    placed: dict
    below: bool
    items: List[dict]
    lane: tuple  # (targets, centres, labels, spans) of the ground-truth badges, strip degrees
    lane_notes: list
    note_lines: list
    connectors: bool = True  # the dotted tick-to-x lines of the misses are drawn (False: crowded row, see _place)
    n_conn: int = 0  # dotted connectors the row has (drawn only when ``connectors``)
    leader_pt: float = 0.0  # longest miss-badge leader (pt)
    warnings: List[str] = field(default_factory=list)
    win: Tuple[float, float] = (-cd.EL_DEFAULT, cd.EL_DEFAULT)  # the query's elevation window (D4, degrees)
    v: float = 1.0  # vertical scale of the rows (1: square degrees; < 1: squeezed to fit the page)
    rv: Optional[dd.CaseRow] = None  # the row with its elevations times ``v`` (what the marks were placed on)

    @property
    def win_v(self) -> Tuple[float, float]:
        """The window in drawn units: degrees times ``v`` (the rows' y axis)."""
        return self.win[0] * self.v, self.win[1] * self.v

    def h_heat(self, ppd: float) -> float:
        """Height (in) of each affordance map row at ``ppd`` pt per degree."""
        return (self.win[1] - self.win[0]) * self.v * ppd / 72.0


def _scaled_row(r: dd.CaseRow, v: float) -> dd.CaseRow:
    """``r`` with the predicted and ground-truth peak elevations times ``v``: a row drawn squeezed vertically is
    laid out (marks, badges, connectors) in the drawn units, where the shared functions' square-degree geometry
    holds again.  Bearings, maps, hits and misses are unchanged."""
    if v == 1.0:
        return r
    p = r.arms[ARM]
    arms = dict(r.arms)
    arms[ARM] = dataclasses.replace(p, peak_elev=np.asarray(p.peak_elev, dtype=float) * v)
    gpe = None if r.gt_peak_elev is None else np.asarray(r.gt_peak_elev, dtype=float) * v
    return dataclasses.replace(r, arms=arms, gt_peak_elev=gpe)


def _ordered_x(r: dd.CaseRow) -> np.ndarray:
    """Strip position (deg) of each badge group of ``r``, with near-ties put in slot order.

    Groups whose true bearings lie within ``TIE_DEG`` of one another (a chain) get positions ``TIE_STEP``
    apart around their mean, in slot order along the strip (clockwise).  The lane (``cd.dodge_1d``: sorted
    by position) and the disc rim (``cd.disc_rim_layout``: sorted by plot angle) each break exact ties by
    slot but in opposite directions, and a 0.01 deg difference decides the order on its own, so two past
    positions straight behind read "7 8" in the lane but "8 7" on the rim, and "5 4" where they are 0.01 deg
    apart.  Only the badges' order uses these positions; rays, dots, guides and ticks stay at the true
    bearings (the nudge is under 0.1 deg for up to 8 slots).
    """
    x = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in r.groups])
    out = x.copy()
    if len(x) < 2:
        return out
    order = [int(i) for i in np.argsort(x, kind="stable")]

    def flush(chain):
        if len(chain) > 1:
            m = float(np.mean(x[chain]))
            for j, i in enumerate(sorted(chain, key=lambda i: r.groups[i][0])):
                out[i] = m + (j - (len(chain) - 1) / 2.0) * TIE_STEP

    chain = [order[0]]
    for i in order[1:]:
        if x[i] - x[chain[-1]] <= TIE_DEG:
            chain.append(i)
        else:
            flush(chain)
            chain = [i]
    flush(chain)
    return out


def _ordered_bearings(r: dd.CaseRow) -> List[float]:
    """``_ordered_x`` as bearings (deg, (-180, 180]) for the disc rim."""
    b = np.mod(cd.STRIP_START_DEG - _ordered_x(r) + 180.0, 360.0) - 180.0
    return [float(v) for v in b]


def _lane_layout(r: dd.CaseRow, ppd: float):
    """``fig_case.lane_layout`` at the animation's scale, near-ties in slot order (``_ordered_x``)."""
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in r.groups])
    labels = [dd.group_label(g) for g in r.groups]
    xs = cd.dodge_1d(_ordered_x(r), [cd.badge_width_pt(s) / ppd for s in labels], 0.0, 360.0, 1.0 / ppd)
    spans = [(x - cd.badge_width_pt(s) / ppd / 2, x + cd.badge_width_pt(s) / ppd / 2) for x, s in zip(xs, labels)]
    return targets, xs, labels, spans


def _segments_cross(p1, p2, q1, q2) -> bool:
    """Proper crossing of two segments (touching end points do not count)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p1, p2, q1), orient(p1, p2, q2)
    d3, d4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _leader_quality(placed: dict, conn: Sequence, ppd: float) -> Tuple[int, float]:
    """(crossings, longest leader in pt) of a ``cd.place_miss_labels`` layout: a crossing is a badge's
    leader crossing another leader or a dotted connector (``conn``), which ``place_miss_labels`` does
    not test."""
    leaders = [lab["leader"] for lab in placed["labels"].values() if lab["leader"] is not None]
    longest = max([math.hypot(b[0] - a[0], b[1] - a[1]) * ppd for a, b in leaders], default=0.0)
    n = 0
    for i, (a, b) in enumerate(leaders):
        n += sum(_segments_cross(a, b, ln[0], ln[-1]) for ln in conn)
        n += sum(_segments_cross(a, b, c, d) for c, d in leaders[i + 1:])
    return n, longest


def _miss_order(marks) -> List[int]:
    """Mark indices of the misses, left to right (``cd.place_miss_labels``'s order)."""
    return [j for j, m in sorted([(j, m) for j, m in enumerate(marks) if m.get("miss", m.get("slot") is not None)],
                                 key=lambda jm: jm[1]["x"])]


class _Placer:
    """A port of an earlier ``cd.place_miss_labels``, one badge at a time: its candidates and cost.

    ``place(j, placed)`` returns the badge of miss mark ``j`` given the boxes of the badges already placed
    (in placing order), as ``(label entry, cost tuple, box)``.  One difference: a leader also counts as
    crossing when it grazes another x or badge (passes within ``LEADER_CLEAR_PT`` of its box beyond its
    first ``GRAZE_SKIP_PT``), since a long leader running along the edge of other x marks reads as
    pointing at them.  When ported (``LEADER_CLEAR_PT = None``, misses left to right) it reproduced the shared
    function exactly on 480 rows of all tiers; the shared function has since added its own checks (leaders
    against the dotted connectors and earlier leaders, exact leader-through-x tests, ``cd.plan_miss_badges``),
    so the two no longer match line for line.  ``_place_search`` tries other orders.  Candidates that cannot
    beat the best so far on the cost's leading flags are skipped before their remaining terms are computed
    (same result, faster).
    """

    def __init__(self, marks, ppd: float, elev, heat=None, below: bool = False, mark_pt: float = MARK_PT,
                 lines=()):
        self.marks, self.ppd, self.heat, self.below = marks, ppd, heat, below
        lo, hi = cd.elev_window(elev)
        self.Ylo, self.Yhi, self.W = lo * ppd, hi * ppd, 360.0 * ppd
        self.half = mark_pt / 2 + 0.8
        self.bh = cd.MISS_BH
        half = self.half
        self.boxes = [(m["x"] * ppd - half, m["x"] * ppd + half, m["y"] * ppd - half, m["y"] * ppd + half)
                      for m in marks]
        self.in_lo, self.in_hi = self.Ylo + self.bh + 0.9, self.Yhi - self.bh - 0.9
        self.grid = (list(np.arange(self.in_lo, self.in_hi + 1e-6, 2.0)) if self.in_hi >= self.in_lo
                     else [(self.Ylo + self.Yhi) / 2])
        self.below_y = self.Ylo - cd.BELOW_LANE_PT
        line_pts = []
        for ln in lines:
            q = np.asarray(ln, dtype=float) * ppd
            for a_, b_ in zip(q[:-1], q[1:]):
                n_ = max(2, int(np.hypot(*(b_ - a_)) / 1.0))
                line_pts.append(np.linspace(a_, b_, n_))
        self.line_pts = np.concatenate(line_pts) if line_pts else np.zeros((0, 2))

    def _on_line(self, lo_, hi_, ylo, yhi) -> bool:
        lp = self.line_pts
        if not len(lp):
            return False
        return bool(np.any((lp[:, 0] > lo_ - 0.4) & (lp[:, 0] < hi_ + 0.4) & (lp[:, 1] > ylo - 0.4)
                           & (lp[:, 1] < yhi + 0.4)))

    def place(self, j_own: int, placed: Sequence[tuple], k: int = 1):
        """The best badge of miss mark ``j_own`` (``k`` = 1, the shared rule), or with ``k`` > 1 a list of the
        ``k`` best with centres at least ``TOPK_SEP_PT`` apart (for ``_place_search``)."""
        ppd, half, bh, W = self.ppd, self.half, self.bh, self.W
        boxes = self.boxes + list(placed)
        m = self.marks[j_own]
        label = m.get("label") or cd.slots_label(m["slots"])
        bw = cd.badge_width_pt(label)
        xp, yp = m["x"] * ppd, m["y"] * ppd
        others = [boxes[j] for j in range(len(boxes)) if j != j_own]
        in_lo, in_hi = self.in_lo, self.in_hi

        def gap(b, lo_, hi_, ylo, yhi):
            b0, b1, c0, c1 = b
            return math.hypot(max(0.0, b0 - hi_, lo_ - b1), max(0.0, c0 - yhi, ylo - c1))

        def overlap(b, lo_, hi_, ylo, yhi):
            b0, b1, c0, c1 = b
            return max(0.0, min(hi_, b1) - max(lo_, b0)) * max(0.0, min(yhi, c1) - max(ylo, c0))

        ts = np.linspace(0.0, 1.0, 20)
        clear = LEADER_CLEAR_PT

        def crosses(p0, p1):
            qx, qy = p0[0] + ts * (p1[0] - p0[0]), p0[1] + ts * (p1[1] - p0[1])
            if any(np.any((qx > b0 + 0.3) & (qx < b1 - 0.3) & (qy > c0 + 0.3) & (qy < c1 - 0.3))
                   for (b0, b1, c0, c1) in others):
                return True
            if clear is None:
                return False
            # grazing: a leader running along the edge of another x reads as pointing at it
            far = np.hypot(qx - p0[0], qy - p0[1]) > GRAZE_SKIP_PT
            return any(np.any(far & (qx > b0 - clear) & (qx < b1 + clear) & (qy > c0 - clear) & (qy < c1 + clear))
                       for (b0, b1, c0, c1) in others)

        lanes = [(float(np.clip(yp, in_lo, in_hi)) if in_hi >= in_lo else self.grid[0], False)]
        lanes += [(float(y), False) for y in self.grid]
        if self.below:
            lanes.append((self.below_y, True))
        best, pool = None, []
        for ly, is_below in lanes:
            cands = []
            for shift in np.arange(0.0, 96.0, 1.5):
                for sgn in (1.0, -1.0):
                    a = xp + sgn * (half + 0.6 + shift)
                    lo_, hi_ = (a, a + bw) if sgn > 0 else (a - bw, a)
                    cands.append((lo_, hi_, sgn, shift))
            if abs(ly - yp) >= half + bh + 0.4:  # straight above / below its x
                cands.append((xp - bw / 2, xp + bw / 2, 0.0, 0.0))
            for lo_, hi_, sgn, shift in cands:
                if lo_ < 0.5 or hi_ > W - 0.5:
                    continue
                ylo, yhi = ly - bh, ly + bh
                hard = sum(overlap(b, lo_, hi_, ylo, yhi) for b in others)
                if sgn == 0.0:
                    hard += overlap(boxes[j_own], lo_, hi_, ylo, yhi)
                if k == 1 and best is not None and (hard > 0.01) > best[0][0]:
                    continue
                cx = (lo_ + hi_) / 2
                d = np.array([cx - xp, ly - yp])
                dist = float(np.hypot(*d))
                u = d / max(dist, 1e-9)
                p0 = (xp + u[0] * half * 0.85, yp + u[1] * half * 0.85)
                rb = bh if bw <= 7.5 else min(bw / 2 / (abs(u[0]) + 1e-9), bh / (abs(u[1]) + 1e-9))
                p1 = (cx - u[0] * (rb + 0.2), ly - u[1] * (rb + 0.2))
                leader = (shift > 0 or abs(ly - yp) > 1.5 or is_below or sgn == 0.0) and math.hypot(
                    p1[0] - p0[0], p1[1] - p0[1]) >= 1.2
                cross = leader and crosses(p0, p1)
                if k == 1 and best is not None and (hard > 0.01, cross) > best[0][:2]:
                    continue
                near = min([gap(b, lo_, hi_, ylo, yhi) for b in others], default=99.0)
                ambiguous = near < (cd.LABEL_CLEAR_PT if not leader else 2.0)  # nearer another x than its own
                soft = (shift + 0.8 * abs(ly - yp) + (2.0 if leader else 0.0)
                        + 2.0 * max(0.0, cd.LABEL_CLEAR_PT - near) + (8.0 if is_below else 0.0))
                h = self.heat(lo_ / ppd, hi_ / ppd, ylo / ppd, yhi / ppd) if self.heat is not None else 0.0
                cost = (hard > 0.01, cross, ambiguous, self._on_line(lo_, hi_, ylo, yhi), hard, soft, h)
                if k > 1:
                    pool.append((cost, cx, ly, p0, p1, leader, is_below))
                if best is None or cost < best[0]:
                    best = (cost, cx, ly, p0, p1, leader, is_below)

        def entry(c):
            cost, cx, ly, p0, p1, leader, is_below = c
            lab = {"cx": cx / ppd, "cy": ly / ppd, "below": bool(is_below), "label": label,
                   "slots": list(m["slots"]),
                   "leader": ((p0[0] / ppd, p0[1] / ppd), (p1[0] / ppd, p1[1] / ppd)) if leader else None}
            return lab, cost, (cx - bw / 2, cx + bw / 2, ly - bh, ly + bh)

        if k == 1:
            return entry(best)
        keep = []
        for c in sorted(pool, key=lambda c: c[0]):
            if all(math.hypot(c[1] - q[1], c[2] - q[2]) >= TOPK_SEP_PT for q in keep):
                keep.append(c)
                if len(keep) == k:
                    break
        return [entry(c) for c in keep]


def _as_layout(seq: Sequence[tuple]) -> dict:
    """``cd.place_miss_labels``-style result from ``[(j, label entry, cost, box), ...]`` (placing order)."""
    return {"labels": {j: lab for j, lab, _, _ in seq}, "costs": {j: c for j, _, c, _ in seq},
            "clean": not any(c[0] or c[1] for _, _, c, _ in seq)}


def _place_ordered(marks, ppd: float, elev, heat=None, below: bool = False, mark_pt: float = MARK_PT,
                   lines=(), order: Optional[Sequence[int]] = None) -> dict:
    """``cd.place_miss_labels`` with the badges placed in ``order`` (mark indices of the misses; default
    left to right, which reproduces the shared result exactly).  Also returns each badge's cost tuple under
    ``"costs"``."""
    P = _Placer(marks, ppd, elev, heat=heat, below=below, mark_pt=mark_pt, lines=lines)
    seq = []
    for j in (_miss_order(marks) if order is None else order):
        lab, cost, box = P.place(j, [b for _, _, _, b in seq])
        seq.append((j, lab, cost, box))
    return _as_layout(seq)


def _box_gap(a, b) -> float:
    """Gap (pt) between two boxes (x0, x1, y0, y1); 0 when they touch or overlap."""
    return math.hypot(max(0.0, a[0] - b[1], b[0] - a[1]), max(0.0, a[2] - b[3], b[2] - a[3]))


def _unclear_badges(placed: dict, marks, ppd: float) -> int:
    """Badges that read as labelling another x: no leader or a short one (under ``PROX_LEADER_PT``, hard
    to see) while another x is about as close as their own (within ``PROX_MARGIN_PT``).  A long leader
    names its x on its own."""
    half = MARK_PT / 2 + 0.8
    xbox = [(m["x"] * ppd - half, m["x"] * ppd + half, m["y"] * ppd - half, m["y"] * ppd + half) for m in marks]
    n = 0
    for j, lab in placed["labels"].items():
        bw = cd.badge_width_pt(lab["label"])
        cx, cy = lab["cx"] * ppd, lab["cy"] * ppd
        box = (cx - bw / 2, cx + bw / 2, cy - cd.MISS_BH, cy + cd.MISS_BH)
        lead = 0.0
        if lab["leader"] is not None:
            (x0, y0), (x1, y1) = lab["leader"]
            lead = math.hypot(x1 - x0, y1 - y0) * ppd
        if lead >= PROX_LEADER_PT:
            continue
        own = _box_gap(box, xbox[j])
        other = min([_box_gap(box, b) for i, b in enumerate(xbox) if i != j], default=99.0)
        n += other < own + PROX_MARGIN_PT
    return n


def _layout_key(placed: dict, conn, ppd: float, marks) -> tuple:
    """Badness of a miss-badge layout, compared lexicographically: not clean, leader crossings (with
    leaders or connectors; ``cd.place_miss_labels`` does not test them), badges nearer another x than their
    own (the shared rule), badges that read as labelling another x (``_unclear_badges``), badges on a
    connector, then the summed soft cost of ``cd.place_miss_labels``.  Every term only grows as badges are
    added, so the key of a partial layout bounds all its completions."""
    crossings, _ = _leader_quality(placed, conn, ppd)
    costs = list(placed["costs"].values())
    return (not placed["clean"], crossings, sum(bool(c[2]) for c in costs), _unclear_badges(placed, marks, ppd),
            sum(bool(c[3]) for c in costs), round(sum(c[5] for c in costs), 3))


def _place_search(marks, ppd: float, win, heat, conn, below: bool) -> dict:
    """The best miss-badge layout found by a small search (``_Placer``), by ``_layout_key``.

    ``cd.place_miss_labels`` places greedily left to right, each badge at its own cheapest spot, so in a
    crowded row an early badge can take the one spot a later badge needed, and it does not test a
    leader crossing another leader or a dotted connector.  When the left-to-right layout has a problem
    the key counts (anything before the soft cost), two depth-first searches with branch and bound on the
    key look for a better one: over placing orders (each badge at its cheapest spot), and left to right
    with each badge also tried at its 2nd and 3rd cheapest distinct spot; ``SEARCH_BUDGET`` badge
    placements each.  Ties keep the left-to-right layout (the shared result).
    """
    P = _Placer(marks, ppd, win, heat=heat, below=below, mark_pt=MARK_PT, lines=conn)
    asc = _miss_order(marks)
    seq = []
    for j in asc:
        lab, cost, box = P.place(j, [b for _, _, _, b in seq])
        seq.append((j, lab, cost, box))
    best = [_layout_key(_as_layout(seq), conn, ppd, marks), _as_layout(seq)]
    if not any(best[0][:5]):
        return best[1]

    def done(prefix):
        lay = _as_layout(prefix)
        key = _layout_key(lay, conn, ppd, marks)
        if key < best[0]:
            best[0], best[1] = key, lay

    def promising(prefix):
        return _layout_key(_as_layout(prefix), conn, ppd, marks) < best[0]  # else every completion is as bad

    def by_order(prefix, remaining, budget):
        if not remaining:
            return done(prefix)
        for j in remaining:
            if budget[0] <= 0:
                return
            budget[0] -= 1
            nxt = prefix + [(j,) + P.place(j, [b for _, _, _, b in prefix])]
            if promising(nxt):
                by_order(nxt, [i for i in remaining if i != j], budget)

    def by_spot(prefix, remaining, budget):
        if not remaining:
            return done(prefix)
        if budget[0] <= 0:
            return
        budget[0] -= 1
        j = remaining[0]
        for choice in P.place(j, [b for _, _, _, b in prefix], k=TOPK):
            nxt = prefix + [(j,) + choice]
            if promising(nxt):
                by_spot(nxt, remaining[1:], budget)

    by_order([], asc, [SEARCH_BUDGET])
    by_spot([], asc, [SEARCH_BUDGET])
    return best[1]


def _place(marks, ppd: float, win, heat, conn, tag: str) -> Tuple[dict, bool, List[str]]:
    """Miss badges inside the row, else with the lane under it (``fig_case.plan_block``'s rule), each
    time through ``_place_search``."""
    warnings = []
    placed = _place_search(marks, ppd, win, heat, conn, below=False)
    below = False
    if not placed["clean"]:
        placed = _place_search(marks, ppd, win, heat, conn, below=True)
        below = any(v["below"] for v in placed["labels"].values())
        if below:
            warnings.append(f"{tag}: miss badges need the lane under the prediction row")
        if not placed["clean"]:
            warnings.append(f"{tag}: a miss badge overlaps or its leader crosses another mark")
    return placed, below, warnings


def _plan_row(fig_m, r: dd.CaseRow, win, ppd: float, CL: dict, v: float = 1.0) -> _RowPlan:
    """Marks, miss badges and notes of one query's row over its elevation window ``win`` (``fig_case.plan_block``),
    drawn at vertical scale ``v`` (1 = square degrees, as fig1).

    Miss badges: ``_place`` (the shared placement, searched further when its layout has a problem
    ``_layout_key`` counts).  Crowded rows: the dotted connectors (D2: allowed, not required) take room
    the badges need and can cross their leaders, so a row whose layout still has such a problem is also
    laid out without them; that layout is used, and the row draws no connectors, only when it is
    strictly better by ``_layout_key`` before the soft cost (ties keep the connectors).  A squeezed row
    (``v`` < 1) is laid out in drawn units (``_scaled_row``); its maps are sampled over the true window.
    """
    tag = f"frame {r.frame + 1}"  # 1-based, as every frame label (D6)
    gt_strip = cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, win)
    pr_strip = cd.heat_strip(dd.pred_composite(r, ARM), HEAT_RING_W, win)
    rv = _scaled_row(r, v)
    win_v = (win[0] * v, win[1] * v)
    marks = cd.peak_marks(rv, ARM, win_v, ppd, mark_pt=MARK_PT)
    heat = cd.heat_lookup(pr_strip, win_v)
    conn = cd.miss_connectors(rv, marks, win_v, ppd, mark_pt=MARK_PT)
    win_true, win = win, win_v  # the placement below works in drawn units
    placed, below, warnings = _place(marks, ppd, win, heat, conn, tag)
    connectors = True
    key = _layout_key(placed, conn, ppd, marks)
    if conn and any(key[:5]):
        alt, alt_below, alt_warn = _place(marks, ppd, win, heat, [], tag)
        alt_key = _layout_key(alt, [], ppd, marks)
        if alt_below <= below and alt_key[:5] < key[:5]:
            placed, below, warnings, key, connectors = alt, alt_below, alt_warn, alt_key, False
    crossings, longest = _leader_quality(placed, conn if connectors else [], ppd)
    if crossings:
        warnings.append(f"{tag}: a miss badge's leader crosses {crossings} connector(s) or leader(s)")
    items = fc.note_items(r, CL, ARM)
    lane = _lane_layout(r, ppd)
    gap = fc.NOTE_GAP_PT / ppd
    left_hi = min([a for a, _ in lane[3]], default=360.0 + gap) - gap
    first, lines = cd.wrap_notes(fig_m, items, 358.0 * ppd, first_pt=max((left_hi - 1.0) * ppd, 0.0),
                                 fs=FS["note"], sep_pt=8.0)
    lane_notes = [(1.0 + x / ppd, it) for x, it in first]
    note_lines = [[(1.0 + x / ppd, it) for x, it in line] for line in lines]
    return _RowPlan(r=r, gt_strip=gt_strip, pr_strip=pr_strip, marks=marks, placed=placed, below=below,
                    items=items, lane=lane, lane_notes=lane_notes, note_lines=note_lines, connectors=connectors,
                    n_conn=len(conn), leader_pt=longest, warnings=warnings, win=tuple(win_true), v=float(v), rv=rv)


# --------------------------------------------------------------------------- #
# Robot along the route (clip frames; fractional t interpolates)
# --------------------------------------------------------------------------- #
class _Track:
    def __init__(self, dump: dd.Dump):
        self.xz = dump.positions[:, [0, 2]].astype(np.float64)
        self.y = dump.positions[:, 1].astype(np.float64)
        f = forward_from_c2w(dump["clip_c2w"].astype(np.float64))
        self.ang = np.unwrap(np.arctan2(f[:, 1], f[:, 0]))  # continuous facing angle, map (x, z) plane
        self.last = len(self.xz) - 1

    def at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        t = float(np.clip(t, 0.0, self.last))
        i = min(int(math.floor(t)), max(self.last - 1, 0))
        a = t - i if self.last > 0 else 0.0
        j = min(i + 1, self.last)
        p = (1 - a) * self.xz[i] + a * self.xz[j]
        th = (1 - a) * self.ang[i] + a * self.ang[j]
        return p, np.array([math.cos(th), math.sin(th)])

    def trail(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.last))
        p, _ = self.at(t)
        return np.vstack([self.xz[: int(math.floor(t)) + 1], p[None]])

    def height(self, t: float) -> float:
        return float(self.y[int(round(float(np.clip(t, 0.0, self.last))))])


def _ease(a: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _route_panel(ax, level, dump: dd.Dump, CL: dict):
    xz = dump.positions[:, [0, 2]]
    limits = cd.fit_limits(xz, pad=0.8, aspect_hw=cd.axes_aspect_hw(ax))
    cd.draw_topdown(ax, level, limits, sat=0.2, white=0.56)
    cd.clean_axes(ax, spines=True)
    cd.draw_route(ax, xz, color=ROUTE_AHEAD, lw=1.1, start_label=None, zorder=3)
    # label on the side away from where the route goes (draw_route's rule, measured at the first point
    # 0.3 m out: routes often start by turning on the spot)
    away = np.nonzero(np.linalg.norm(xz - xz[0], axis=1) > 0.3)[0]
    d = xz[away[0] if len(away) else -1] - xz[0]
    d = -d / (np.linalg.norm(d) + 1e-9)
    ha = "right" if d[0] < -0.35 else ("left" if d[0] > 0.35 else "center")
    va = "top" if d[1] > 0.35 else ("bottom" if d[1] < -0.35 else "center")  # +z is down on the map
    ax.annotate(CL["start"], xz[0], xytext=(d[0] * 6.0, -d[1] * 6.0), textcoords="offset points", ha=ha, va=va,
                fontsize=6.2, color=style.INK_2, path_effects=cd.HALO, zorder=4.6)
    # scale bar in the top-left corner, placed in display space (the map's y axis may run either way): the bar
    # 12.5 pt under the frame's top edge, so its label (1.6 pt above it, ~6 pt tall) keeps >= 3 pt to the frame
    bar = cd.nice_length(0.3 * (limits[1] - limits[0]))
    px = ax.figure.dpi / 72.0
    bb = ax.get_window_extent()
    x_d, y_d = ax.transData.inverted().transform((bb.x0 + 6.0 * px, bb.y1 - 12.5 * px))
    line, text = cd.scale_bar(ax, float(x_d), float(y_d), bar, f"{bar:g} m", fs=6.0)
    return limits, (line, text)


def _strip_header(fig, lay: _Layout, CL: dict) -> None:
    q = lay.w_strip / 4
    for v, name in enumerate(CL["views"]):
        _text(fig, lay.x_strip + (v + 0.5) * q, lay.y_views, name, bold=(v == 0), ha="center", va="center",
              fontsize=FS["views"], color=style.INK if v == 0 else style.INK_2)
    # bracket over the three views not given to the model, the note set into its top line
    ax = _ax(fig, lay.x_strip + q + 0.03, lay.y_bracket, 3 * q - 0.06, 0.07)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    italic = not cd.has_cjk(CL["not_given"])
    w_note = cd.text_width_pt(fig, CL["not_given"], FS["bracket"], fontstyle="italic" if italic else "normal") / 72.0
    frac = (w_note + 0.16) / (3 * q - 0.06)
    ax.plot([0, 0, 0.5 - frac / 2], [0.0, 0.5, 0.5], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.plot([0.5 + frac / 2, 1, 1], [0.5, 0.5, 0.0], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.text(0.5, 0.5, CL["not_given"], ha="center", va="center", fontsize=FS["bracket"], color=style.INK_2,
            fontstyle="italic" if italic else "normal")


def _gutter_texts(L: dict, lang: str, idx: int, n: int, frame: int, T: int) -> Tuple[str, str]:
    return L["strip_query"].format(i=idx + 1, n=n), cd.frame_label(frame, T, lang)


def _strip_label(fig, lay: _Layout, L: dict, lang: str, r: dd.CaseRow, idx: int, n: int, T: int) -> None:
    """Gutter label of the RGB row: the query the whole strip belongs to (stays sharp while faded)."""
    x, yc = lay.x_strip - 0.07, lay.y_rgb + lay.h_rgb / 2
    a, b = _gutter_texts(L, lang, idx, n, r.frame, T)
    _text(fig, x, yc - 0.06, a, bold=True, ha="right", va="center", fontsize=FS["gutter"], color=style.INK)
    _text(fig, x, yc + 0.075, b, ha="right", va="center", fontsize=FS["gutter_sub"], color=style.INK_2)


def _elev_label(fig, lay: _Layout, plan: _RowPlan, L: dict, y_axis: float) -> List[str]:
    """The query's elevation window in the gutter, level with the degree labels (right-aligned, muted); a
    squeezed query (``plan.v`` < 1) also gets its factor on a line above, beside the foot of the prediction row.
    Returns the lines drawn."""
    x = lay.x_strip - 0.07
    room = (x - MARGIN) * 72.0
    lo, hi = cd.fmt_deg(plan.win[0]), cd.fmt_deg(plan.win[1])
    forms = [f.format(lo=lo, hi=hi) for f in L["elev"]]
    text = next((t for t in forms if cd.text_width_pt(fig, t, FS["elev"]) <= room), forms[-1])
    _text(fig, x, y_axis, text, ha="right", va="center", fontsize=FS["elev"], color=style.MUTED)
    out = [text]
    if plan.v < 1.0 - 1e-6:
        sq = L["squeezed"].format(v=plan.v)
        _text(fig, x, y_axis - 0.125, sq, ha="right", va="center", fontsize=FS["elev"], color=style.MUTED)
        out.insert(0, sq)
    return out


def _strip(fig, lay: _Layout, plan: _RowPlan, views: np.ndarray, CL: dict, rgb_ring_w: int, L: dict) -> dict:
    """The surround strip of one query row: notes, lane, RGB, ground truth, prediction (``fig_case.draw_block``).

    Returns ``{"numbered": [...], "in_notes": [...], "notes": [...]}`` (1-based slots) for the checks; raises
    (``data.check_accounting``) unless every valid slot has a badge or a note and the numbered misses are exactly
    the row's joint PCK@8 misses.
    """
    r, win, ppd = plan.r, plan.win_v, lay.ppd  # the rows' y axis is in drawn units (degrees times plan.v)
    lo, _ = win
    h_heat = plan.h_heat(ppd)
    y_pr = lay.y_gt + h_heat + ROW_GAP
    ax_lane = _ax(fig, lay.x_strip, lay.y_lane, lay.w_strip, H_LANE)
    ax_rgb = _ax(fig, lay.x_strip, lay.y_rgb, lay.w_strip, lay.h_rgb)
    ax_gt = _ax(fig, lay.x_strip, lay.y_gt, lay.w_strip, h_heat)
    ax_pr = _ax(fig, lay.x_strip, y_pr, lay.w_strip, h_heat)
    cd.draw_rgb_row(ax_rgb, cd.rgb_strip(views, rgb_ring_w, EL_RGB), EL_RGB)
    cd.draw_heat_row(ax_gt, plan.gt_strip, win, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, plan.pr_strip, win, cd.PRED_CMAP)
    for ax_row, text, kind in ((ax_gt, CL["gt_row"], "gt"), (ax_pr, CL["pred_row"], "pred")):
        cd.gutter_row_label(ax_row, text, kind, fs=FS["rowlab"], gap_pt=5.0, key_w_pt=26.0, key_h_pt=2.8)

    # note lines just above the lane (a row with fewer lines than the layout reserves leaves the top ones empty)
    n_lines = len(plan.note_lines)
    for i, line in enumerate(plan.note_lines):
        y_top = lay.y_lane - (n_lines - i) * NOTE_LINE_H
        ax_note = _ax(fig, lay.x_strip, y_top, lay.w_strip, NOTE_LINE_H)
        ax_note.set_xlim(0, 360)
        ax_note.set_ylim(0, 1)
        ax_note.axis("off")
        for x, it in line:
            cd.draw_note(ax_note, x, 0.5, it, per_pt=1.0 / ppd, fs=FS["note"])
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")

    # ground truth: badges in the lane, guide through the RGB row, ticks under the prediction
    targets, xs, labels, _ = plan.lane
    y_badge = 0.52
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    for g, t, x, lab in zip(r.groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([x, x, t, t], [y_badge, 0.30, 0.10, 0.0], color=col, lw=0.6, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, x, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.6, zorder=3)
        ax_pr.plot([t, t], [lo - 1.0 * tick, lo - 4.2 * tick], color=col, lw=0.9, zorder=3, clip_on=False,
                   solid_capstyle="butt")
    for x, it in plan.lane_notes:
        cd.draw_note(ax_lane, x, y_badge, it, per_pt=1.0 / ppd, fs=FS["note"])

    # predicted peaks and misses (D1, D2); the dotted connectors unless the row is crowded (``_plan_row``)
    if plan.connectors:
        cd.draw_miss_connectors(ax_pr, plan.rv, plan.marks, win, ppd, mark_pt=MARK_PT)
    cd.draw_peak_marks(ax_pr, plan.marks, size=MARK_PT)
    numbered = cd.draw_miss_labels(ax_pr, plan.placed)

    badge_slots = [k for g in r.groups for k in g]
    note_slots = [k for it in plan.items for k in it["slots"]]
    in_notes = [k for it in plan.items if it["kind"] == "predicted_none" for k in it["slots"]]
    dd.check_accounting(r, ARM, badge_slots, note_slots, numbered, in_notes)

    if plan.below:  # degree labels under the miss lane, on an axis of their own (no ticks through the lane)
        ax_ax = _ax(fig, lay.x_strip, y_pr + h_heat + MISS_LANE_H, lay.w_strip, 0.001)
        ax_ax.set_xlim(0, 360)
        cd.clean_axes(ax_ax)
        ax_ax.patch.set_visible(False)
        cd.azimuth_axis(ax_ax, CL["axis"], fs=FS["axis"])
        ax_ax.tick_params(axis="x", which="major", pad=3.0)
        axis_labels = ax_ax.get_xticklabels()
    else:
        cd.azimuth_axis(ax_pr, CL["axis"], fs=FS["axis"])
        ax_pr.tick_params(axis="x", which="major", pad=6.0)
        axis_labels = ax_pr.get_xticklabels()
    # the elevation window, level with the degree labels
    fig.canvas.draw()
    bb = axis_labels[0].get_window_extent()
    y_axis = FIG_H - (bb.y0 + bb.y1) / 2.0 / fig.dpi
    elev = _elev_label(fig, lay, plan, L, y_axis)
    return {"numbered": sorted(int(k) + 1 for k in numbered), "in_notes": sorted(int(k) + 1 for k in in_notes),
            "notes": [dict(kind=it["kind"], slots=[k + 1 for k in it["slots"]], text=it["text"]) for it in plan.items],
            "elevation_label": elev, "rows_bottom_in": round(y_pr + h_heat + (MISS_LANE_H if plan.below else 0.0)
                                                                 + AXIS_H, 3)}


# --------------------------------------------------------------------------- #
# Local map with rim badges on one or two rings (displacement capped)
# --------------------------------------------------------------------------- #
def _rim_spread(ax, half: float, groups, bearings) -> float:
    """Largest angle (deg) between a rim badge and its bearing after ``cd.disc_rim_layout``'s dodge."""
    if not groups:
        return 0.0
    _, theta, placed, _ = cd.disc_rim_layout(ax, half, groups, bearings)
    return float(max(abs((p - t + 180.0) % 360.0 - 180.0) for p, t in zip(placed, theta)))


def _two_ring_labels(ax, half: float, groups, bearings, ring_gap_pt: float = 8.6) -> List[Tuple[float, float]]:
    """Rim badges alternating between an inner and an outer ring, each ring dodged on its own (bearing order).

    Used when one ring would push a badge more than ``RIM_CAP_DEG`` from its bearing: in bearing order, even
    badges go on the inner ring and odd ones ``ring_gap_pt`` farther out, so neighbours overlap in angle but
    not in space and the spread halves.  Each badge gets a leader from its true bearing on the rim; an
    outer badge's leader passes between two inner ones.  Returns ``(theta, half_width)`` per badge, as
    ``cd.disc_rim_labels``.
    """
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    r_pt = half / per_pt
    order = sorted(range(len(groups)), key=lambda i: ((90.0 + bearings[i]) % 360.0, groups[i][0]))
    rings = {0: [i for j, i in enumerate(order) if j % 2 == 0], 1: [i for j, i in enumerate(order) if j % 2 == 1]}
    out, spread = [], 0.0
    for ring, idx in rings.items():
        if not idx:
            continue
        off = 5.2 + ring * ring_gap_pt
        sub_g = [groups[i] for i in idx]
        sub_b = [bearings[i] for i in idx]
        labels, theta, placed, widths = cd.disc_rim_layout(ax, half, sub_g, sub_b, offset_pt=off)
        for g, lab, t, p, w in zip(sub_g, labels, theta, placed, widths):
            k = g[0]
            u_t = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
            u_p = np.array([math.cos(math.radians(p)), math.sin(math.radians(p))])
            ring_pt = r_pt + (off - 3.7) + cd._badge_support(lab, u_p)
            c = u_p * ring_pt * per_pt
            if ring == 1 or abs(((p - t + 180) % 360) - 180) > 0.8:
                rim, mid = u_t * half, u_t * (r_pt + 1.8) * per_pt
                ax.plot([rim[0], mid[0], c[0]], [rim[1], mid[1], c[1]], color=cd.history_line_color(k), lw=0.5,
                        zorder=5 - 0.1 * ring, solid_joinstyle="round", clip_on=False)
            cd.history_badge(ax, c[0], c[1], lab, k, zorder=6)
            out.append((float(p % 360.0), float(w / 2)))
            spread = max(spread, abs((p - t + 180.0) % 360.0 - 180.0))
    return out, float(spread)


def _merge_rim_groups(groups, bearings, max_deg: float = RIM_MERGE_DEG):
    """Rim badges merged by direction: ``(groups, bearings, merged labels)``.

    Past positions whose bearings lie within ``max_deg`` of one another (a cluster, in rim order; a robot
    that drove straight leaves its past positions in one line behind it) are named by one badge when their
    slots run without a gap ("3–8" = slots 3 to 8, at their mean bearing); a cluster whose slots have gaps
    is split into its gap-free runs.  The disc cannot resolve those bearings anyway: a badge is ~10 deg of
    rim wide.
    """
    if len(groups) < 2:
        return [list(g) for g in groups], list(bearings), []
    theta = np.mod(90.0 + np.asarray(bearings, dtype=float), 360.0)
    order = list(np.argsort(theta, kind="stable"))
    ts = theta[order]
    gaps = np.diff(np.r_[ts, ts[0] + 360.0])
    start = (int(np.argmax(gaps)) + 1) % len(order)  # cut the circle at its widest gap
    order = order[start:] + order[:start]
    unwrapped = np.unwrap(np.radians(theta[order]))
    clusters, cur = [], [0]
    for i in range(1, len(order)):
        if math.degrees(unwrapped[i] - unwrapped[cur[0]]) <= max_deg:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    out_g, out_b, merged = [], [], []
    for cl in clusters:
        slot_bearing = {}
        for i in cl:
            for k in groups[order[i]]:
                slot_bearing[int(k)] = float(bearings[order[i]])
        slots = sorted(slot_bearing)
        runs, run = [], [slots[0]]
        for k in slots[1:]:
            if k == run[-1] + 1:
                run.append(k)
            else:
                runs.append(run)
                run = [k]
        runs.append(run)
        for run in runs:
            b = [slot_bearing[k] for k in run]
            ang = math.degrees(math.atan2(np.mean(np.sin(np.radians(b))), np.mean(np.cos(np.radians(b)))))
            out_g.append(run)
            out_b.append(ang)
            if not any(list(g) == run for g in groups):
                merged.append(dd.group_label(run))
    return out_g, out_b, merged


def _inset_bar(ax, half: float, corner: Tuple[float, float]):
    """The local map's scale bar in corner ``corner`` = (sx, sy) of the inset square, outside the disc: a nice
    length near 0.6 x the radius, label above (``cd.scale_bar``; returns its ``(line, label)``)."""
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    lim = ax.get_xlim()[1]
    bar = cd.nice_length(0.6 * half)
    sx, sy = corner
    return cd.scale_bar(ax, sx * (lim - BAR_EDGE_PT[int(sx)] * per_pt), sy * (lim - (8.5 if sy > 0 else 2.5) * per_pt),
                        bar, f"{bar:g} m", fs=5.6, ha="left" if sx < 0 else "right")


def _bar_gap_pt(ax, bar, obstacles=None) -> float:
    """Gap (pt) between the scale bar ``bar`` (line + label, 0.5 pt halo) and everything else drawn outside the
    disc on ``ax`` (``cd._rendered_obstacles``: badges, leaders, letters, arrow); ``obstacles``: those boxes
    measured beforehand (without the bar).  Negative never: 0 when they touch."""
    line, text = bar
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    if obstacles is None:
        line.remove()  # not its own obstacle
        try:
            obstacles = cd._rendered_obstacles(ax, skip=(text,))
        finally:
            ax.add_line(line)
    pad = 0.5 * fig.dpi / 72.0
    gaps = [float("inf")]
    for b in (line.get_window_extent(rend), text.get_window_extent(rend)):
        for o in obstacles:
            dx = max(0.0, o.x0 - (b.x1 + pad), (b.x0 - pad) - o.x1)
            dy = max(0.0, o.y0 - (b.y1 + pad), (b.y0 - pad) - o.y1)
            gaps.append(math.hypot(dx, dy) * 72.0 / fig.dpi)
    return min(gaps)


def _draw_inset(ax, level, dump: dd.Dump, r: dd.CaseRow, CL: dict,
                bar_corner: Optional[Tuple[float, float]]) -> dict:
    """Panel b of one query: ``fig_case.draw_inset``'s drawing (disc, rays and dots, robot, rim badges,
    direction arrow, scale bar, sector letters), with the rim badges in the lane's order
    (``_ordered_bearings``: near-ties in slot order); when the rim dodge would move a badge more than
    ``RIM_CAP_DEG`` from its bearing, the badges are merged by direction (``_merge_rim_groups``), and if that
    is still over the cap, also put on two staggered rings.  The scale bar sits in ``bar_corner``, the one
    corner of the whole video (``_Context._pin_bar_corner``); ``None`` draws no bar (that pre-pass).

    Returns ``{"half_m", "rim_spread_deg", "rings", "rim_merged", "rim_spread_drawn_deg", "bar"}``
    (``bar``: the scale bar's ``(line, label)`` or None).
    """
    half = fc.inset_half(r)
    lim = half / RADIUS_FRAC
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    bearings = [float(r.gt_bearing[g[0]]) for g in r.groups]  # true: rays and the sector letters' test
    rim_b = _ordered_bearings(r)
    rim_g = [list(g) for g in r.groups]
    spread = _rim_spread(ax, half, rim_g, rim_b)
    merged: List[str] = []
    spread_m = spread
    if spread > RIM_CAP_DEG:
        rim_g, rim_b, merged = _merge_rim_groups(rim_g, rim_b)
        spread_m = _rim_spread(ax, half, rim_g, rim_b)
    fwd = forward_from_c2w(r.cur_c2w)
    past = dump.positions[: r.frame + 1][:, [0, 2]]
    crop = cd.draw_local_disc(ax, level, r.cur_pos[[0, 2]], fwd, half, past_xz=past, radius_frac=RADIUS_FRAC)
    for g in r.groups:
        k = g[0]
        a, b = crop.world_to_local(r.hist_pos[k, 0], r.hist_pos[k, 2])
        ang = math.atan2(b, a)
        ax.plot([0, half * math.cos(ang)], [0, half * math.sin(ang)], color=cd.history_line_color(k), lw=0.55,
                zorder=3.5, solid_capstyle="butt")
        ax.plot([a], [b], marker="o", ms=3.3, mfc=style.history_color(k), mec="white", mew=0.5, zorder=4.5)
    cd.robot_glyph(ax, 0.0, 0.0, (0.0, 1.0), size_pt=DISC_ROBOT_PT, zorder=6)
    if spread_m <= RIM_CAP_DEG:
        occupied, rings, drawn = cd.disc_rim_labels(ax, half, rim_g, rim_b), 1, spread_m
    else:
        (occupied, drawn), rings = _two_ring_labels(ax, half, rim_g, rim_b), 2
    if fc._arrow_fits(occupied, True):
        cd.disc_direction_arrow(ax, half)
    bar = _inset_bar(ax, half, bar_corner) if bar_corner is not None else None
    cd.disc_sector_letters(ax, half, occupied, names=CL["sectors"], displaced="outward",
                           rays=[90.0 + b for b in bearings])
    return {"half_m": half, "rim_spread_deg": spread, "rings": rings, "rim_merged": merged,
            "rim_spread_drawn_deg": drawn, "bar": bar}


def _legend(fig, lay: _Layout, y_top: float, L: dict, CL: dict) -> float:
    """Five entries in two columns in the info column (the map ramps are keyed beside their rows).

    Returns the legend's height (in)."""
    row_h = LEG_ROW_H
    h = 3 * row_h
    ax = _ax(fig, lay.x_info, y_top, lay.w_info, h)
    w_pt, h_pt = lay.w_info * 72.0, h * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis("off")
    ys = [h_pt - row_h * 72.0 * (j + 0.5) for j in range(3)]
    glyph_w = 3 * 8.6 + 1.0
    x_text = glyph_w + 2.0
    col1 = [(ys[0], CL["legend_hist"]), (ys[1], CL["legend_peak"]), (ys[2], CL["legend_miss"])]
    for j, k in enumerate((0, 4, 7)):
        cd.history_badge(ax, 3.7 + j * 8.6, ys[0], str(k + 1), k)
    cd.peak_mark(ax, 3.7, ys[1], size=MARK_PT)
    cd.miss_badge(ax, 3.7, ys[2], "3")
    w1 = max(cd.text_width_pt(fig, t, FS["legend"]) for _, t in col1)
    for y, t in col1:
        ax.text(x_text, y, t, ha="left", va="center", fontsize=FS["legend"], color=style.INK)
    x2 = x_text + w1 + 12.0
    ax.plot([x2 + 3.7], [ys[1]], ls="none", marker="o", ms=3.4, mfc="white", mec=style.INK_2, mew=0.7)
    cd.robot_glyph(ax, x2 + 3.7, ys[0], (0.0, 1.0), size_pt=7.0)
    for y, t in ((ys[0], CL["legend_robot"]), (ys[1], L["legend_query"])):
        ax.text(x2 + 10.0, y, t, ha="left", va="center", fontsize=FS["legend"], color=style.INK)
    return h


def _fmt_summary(s: dict, L: dict) -> Tuple[str, str]:
    if s["n"] == 0:
        return L["none_visible"], ""
    return L["err"].format(med=s["median"], mx=s["max"]), L["pck"].format(hits=s["hits"], n=s["n"])


# --------------------------------------------------------------------------- #
# One query row = one static scene + animated artists
# --------------------------------------------------------------------------- #
class _Scene:
    """Figure of one query row; ``render(t, mode)`` blits the moving parts over the static background.

    mode: ``hold`` (the row as is) or ``move`` (robot between this row and the next: live route and local
    map, the row's strip and numbers faded, the next query named under the timeline).
    """

    def __init__(self, plt, ctx: "_Context", idx: int, views: np.ndarray, level):
        self.ctx, self.idx = ctx, idx
        plan = ctx.plans[idx]
        r = plan.r
        self.r, self.plan = r, plan
        L, CL, dump, lay = ctx.L, ctx.CL, ctx.dump, ctx.lay
        n = len(ctx.rows)
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=ctx.dpi)
        fig.patch.set_facecolor("white")
        self.fig = fig
        self.warnings: List[str] = list(plan.warnings)

        # title band
        _text(fig, MARGIN, Y_TITLE, L["title"], bold=True, ha="left", va="center", fontsize=FS["title"],
              color=style.INK)
        _text(fig, FIG_W - MARGIN, Y_TITLE, L["sub"].format(tier=dump.tier_name(ctx.lang), scene=dump.scene,
                                                             ep=dump.episode_id, T=dump.frame_count),
              ha="right", va="center", fontsize=FS["sub"], color=style.MUTED)
        _text(fig, MARGIN, Y_NAMES, CL["a"], ha="left", va="center", fontsize=FS["name"], color=style.INK)
        _text(fig, lay.x_disc, Y_NAMES, CL["b"], ha="left", va="center", fontsize=FS["name"], color=style.INK)

        # route (static part) + animated trail / past positions
        self.ax_route = _ax(fig, MARGIN, Y_TOP, W_ROUTE, lay.h_top)
        _, bar = _route_panel(self.ax_route, ctx.route_level, dump, CL)
        self.trail = Line2D([], [], color=style.INK_2, lw=1.5, solid_capstyle="round", solid_joinstyle="round",
                            zorder=4, animated=True)
        self.ax_route.add_line(self.trail)
        xz = dump.positions[:, [0, 2]]
        q = xz[list(ctx.query_frames)]
        self.marks = [  # drawn over the trail: query positions (open circles) and the start
            Line2D(q[:, 0], q[:, 1], ls="none", marker="o", ms=3.4, mfc="white", mec=style.INK_2, mew=0.7,
                   zorder=4.5, animated=True),
            Line2D([xz[0, 0]], [xz[0, 1]], ls="none", marker="o", ms=4.4, mfc="white", mec=style.INK_2, mew=0.9,
                   zorder=4.6, animated=True),
        ]
        for m in self.marks:
            self.ax_route.add_line(m)
        hp = r.hist_pos[r.valid][:, [0, 2]]
        cols = [style.history_color(int(k)) for k in np.nonzero(r.valid)[0]]
        self.hist = self.ax_route.scatter(hp[:, 0], hp[:, 1], s=13.0, c=cols, edgecolors="white", linewidths=0.6,
                                          zorder=6, animated=True)

        # local disc and strip
        self.ax_in = _ax(fig, lay.x_disc, Y_TOP, lay.w_disc, lay.w_disc)
        _strip_header(fig, lay, CL)
        _strip_label(fig, lay, L, ctx.lang, r, idx, n, dump.frame_count)
        self.drawn = _strip(fig, lay, plan, views, CL, ctx.rgb_ring_w, L)
        h_heat = plan.h_heat(lay.ppd)
        self.heat_boxes_px = [((lay.x_strip + 0.01) * ctx.dpi, (y + 0.01) * ctx.dpi,
                               (lay.x_strip + lay.w_strip - 0.01) * ctx.dpi, (y + h_heat - 0.01) * ctx.dpi)
                              for y in (lay.y_gt, lay.y_gt + h_heat + ROW_GAP)]  # the GIF's ordered dither
        self.inset = _draw_inset(self.ax_in, level, dump, r, CL, ctx.bar_corner)
        self.half = self.inset["half_m"]
        # live disc while moving: an invisible twin of the inset axes whose artists are drawn one by one
        self.ax_live = _ax(fig, lay.x_disc, Y_TOP, lay.w_disc, lay.w_disc)
        self.ax_live.set_visible(False)

        # info column
        x = lay.x_info
        self.t_frame = _text(fig, x, Y_NAMES, "", ha="left", va="center", fontsize=FS["frame"],
                             fontweight=cd.bold_weight(cd.frame_label(0, 1, ctx.lang)), color=style.INK, animated=True)
        self.t_query = _text(fig, x, Y_NAMES + INFO_QUERY_DY, "", ha="left", va="center", fontsize=FS["muted"],
                             color=style.INK_2, animated=True)
        self._timeline(fig, Y_NAMES + TL_DY)
        y = Y_NAMES + TL_DY + TL_H + METRICS_GAP
        s = r.summary(ARM)
        a, b = _fmt_summary(s, L)
        lines = [(a, FS["info"], style.INK), (b, FS["info"], style.INK)]
        sf = r.summary("floor")
        if sf["n"]:
            lines.append((L["floor"].format(med=sf["median"], hits=sf["hits"], n=sf["n"]), FS["muted"], style.MUTED))
        so = ctx.so_far[idx]
        if so["n"]:
            lines.append((L["so_far"].format(med=so["median"], hits=so["hits"], n=so["n"]), FS["muted"], style.MUTED))
        step = METRICS_STEP
        for j, (text, fs, col) in enumerate(lines):
            if text:
                _text(fig, x, y + j * step, text, ha="left", va="center", fontsize=fs, color=col)
        metrics_bottom = y + (len(lines) - 1) * step + 0.07
        metrics_box = (x - 0.03, y - 0.09, FIG_W - 0.02, metrics_bottom + 0.02)
        h_leg = 3 * LEG_ROW_H
        y_leg = max(Y_TOP + lay.h_top - h_leg, metrics_bottom + 0.06)
        if y_leg + h_leg > lay.y_bracket - 0.02:
            self.warnings.append("legend runs into the strip header")
        _legend(fig, lay, y_leg, L, CL)

        # checks the drawn page can answer
        self.check = {"route_bar_inset_pt": round(cd.inset_from_frame_pt(self.ax_route, bar), 2),
                      "disc_bar_gap_pt": round(_bar_gap_pt(self.ax_in, self.inset["bar"]), 2)}
        if self.check["route_bar_inset_pt"] < 2.0:
            self.warnings.append(f"route scale bar only {self.check['route_bar_inset_pt']} pt inside the map")
        if self.check["disc_bar_gap_pt"] < BAR_CLEAR_PT:
            self.warnings.append(f"frame {r.frame + 1}: the local map's scale bar is only "
                                 f"{self.check['disc_bar_gap_pt']} pt from a rim badge, leader or letter")

        # veils (drawn only while moving): the strip (not its gutter labels) and the row's numbers; the disc is
        # covered by an opaque plate and redrawn live
        veil = dict(fc="white", ec="none", zorder=50, animated=True, alpha=FADE_ALPHA)
        self.veils = [
            _rect(fig, lay.x_strip - 0.03, lay.y_notes - 0.01, FIG_W - 0.02, self.drawn["rows_bottom_in"] - 0.02,
                  **veil),
            _rect(fig, *metrics_box, **veil),
        ]
        self.disc_plate = _rect(fig, lay.x_disc - 0.08, Y_TOP - 0.03, lay.x_disc + lay.w_disc + 0.06,
                                Y_TOP + lay.h_top + 0.02, fc="white", ec="none", zorder=50, animated=True)

        fig.canvas.draw()
        self.bg = fig.canvas.copy_from_bbox(fig.bbox)

    def _timeline(self, fig, y_top: float) -> None:
        ctx = self.ctx
        lay = ctx.lay
        h = TL_H
        ax = _ax(fig, lay.x_info, y_top, lay.w_info - 0.08, h)
        last = max(ctx.track.last, 1)
        ax.set_xlim(-0.01 * last, 1.01 * last)
        ax.set_ylim(0, 1)
        ax.axis("off")
        y_line = 0.58
        ax.plot([0, last], [y_line, y_line], color=style.AXIS, lw=1.2, solid_capstyle="round", zorder=1)
        qf = np.asarray(ctx.query_frames, dtype=float)
        ax.plot(qf, np.full_like(qf, y_line), ls="none", marker="|", ms=5.0, mew=0.8, color=style.MUTED, zorder=2)
        ax.plot([self.r.frame], [y_line], ls="none", marker="|", ms=8.0, mew=1.4, color=style.INK, zorder=3)
        self.ax_tl, self.tl_y = ax, y_line
        self.tl_marker = Line2D([], [], ls="none", marker="v", ms=4.6, mfc=style.INK, mec="white", mew=0.4,
                                zorder=5, animated=True)
        ax.add_line(self.tl_marker)
        # the next query: its tick picked out, named under the line
        self.tl_next = Line2D([], [], ls="none", marker="|", ms=8.0, mew=1.4, color=style.INK_2, zorder=4,
                              animated=True)
        ax.add_line(self.tl_next)
        self.tl_next_text = ax.text(0, 0.10, "", ha="center", va="center", fontsize=FS["next"], color=style.INK,
                                    animated=True)

    def _next_label(self, frame: int) -> None:
        ax, txt = self.ax_tl, self.tl_next_text
        ctx = self.ctx
        last = max(ctx.track.last, 1)
        txt.set_text(ctx.L["next"].format(frame=cd.frame_label(frame, ctx.dump.frame_count, ctx.lang)))
        renderer = self.fig.canvas.get_renderer()
        txt.set_ha("center")
        txt.set_x(frame)
        bb = txt.get_window_extent(renderer)
        lo, hi = ax.transData.transform([(0.0, 0.0), (float(last), 0.0)])[:, 0]
        if bb.x0 < lo:
            txt.set_ha("left")
            txt.set_x(0.0)
        elif bb.x1 > hi:
            txt.set_ha("right")
            txt.set_x(float(last))
        self.tl_next.set_data([frame], [self.tl_y])
        ax.draw_artist(self.tl_next)
        ax.draw_artist(txt)

    def _live_disc(self, t: float, half: float) -> None:
        """The local map at clip time ``t`` (no query here, so no rays or badges), radius ``half`` m; its scale
        bar in the video's one corner (``ctx.bar_corner``)."""
        ctx, ax = self.ctx, self.ax_live
        before = set(ax.get_children())
        p, f = ctx.track.at(t)
        level = ctx.level_at(t)
        cd.draw_local_disc(ax, level, p, f, half, past_xz=ctx.track.trail(t), radius_frac=RADIUS_FRAC)
        cd.robot_glyph(ax, 0.0, 0.0, (0.0, 1.0), size_pt=DISC_ROBOT_PT, zorder=6)
        cd.disc_direction_arrow(ax, half)
        _inset_bar(ax, half, ctx.bar_corner)
        cd.disc_sector_letters(ax, half, [], names=ctx.CL["sectors"])
        new = [a for a in ax.get_children() if a not in before]
        self.fig.draw_artist(self.disc_plate)
        for a in sorted(new, key=lambda a: a.get_zorder()):
            ax.draw_artist(a)
        for a in new:
            a.remove()

    def render(self, t: float, mode: str, nxt: Optional["_Scene"] = None) -> np.ndarray:
        ctx, fig, ax = self.ctx, self.fig, self.ax_route
        canvas = fig.canvas
        canvas.restore_region(self.bg)
        # route: trail, past positions (hold only), robot
        tr = ctx.track.trail(t)
        self.trail.set_data(tr[:, 0], tr[:, 1])
        ax.draw_artist(self.trail)
        for m in self.marks:
            ax.draw_artist(m)
        if mode == "hold":
            ax.draw_artist(self.hist)
        p, f = ctx.track.at(t)
        robot = cd.robot_glyph(ax, float(p[0]), float(p[1]), f, size_pt=ROBOT_PT, zorder=8)
        robot.set_animated(True)
        ax.draw_artist(robot)
        robot.remove()
        if mode == "move":
            for v in self.veils:
                fig.draw_artist(v)
            t0, t1 = float(self.r.frame), float(nxt.r.frame)
            a = _ease((t - t0) / max(t1 - t0, 1e-9))
            half = math.exp((1 - a) * math.log(self.half) + a * math.log(nxt.half))
            self._live_disc(t, half)
            self._next_label(nxt.r.frame)
        # info
        ti = int(round(t))
        T = ctx.dump.frame_count
        self.t_frame.set_text(cd.frame_label(ti, T, ctx.lang))
        fig.draw_artist(self.t_frame)
        n = len(ctx.query_frames)
        if mode == "move":
            self.t_query.set_text(ctx.L["showing"].format(i=self.idx + 1, n=n,
                                                          frame=cd.frame_label(self.r.frame, T, ctx.lang)))
        else:
            key = "query_sched" if ctx.schedule == "full" else "query"
            self.t_query.set_text(ctx.L[key].format(i=self.idx + 1, n=n))
        fig.draw_artist(self.t_query)
        self.tl_marker.set_data([t], [0.93])
        self.ax_tl.draw_artist(self.tl_marker)
        buf = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(buf[..., :3])

    def close(self, plt) -> None:
        plt.close(self.fig)


class _Context:
    """Everything shared by the frames of one animation: rows, their plans, the one layout, running totals."""

    def __init__(self, plt, dump: dd.Dump, rows: List[dd.CaseRow], lang: str, dpi: float, schedule: str, scene_td,
                 cam_h: float):
        self.dump, self.rows, self.lang, self.dpi = dump, rows, lang, dpi
        self.schedule = schedule  # "full" (every endpoint row), "gaps" (those with an output) or "subset"
        self.L = LABELS[lang]
        self.CL = fc.labels_for(lang)
        self.track = _Track(dump)
        self.scene_td, self.cam_h = scene_td, cam_h
        self.route_level = scene_td.pick_level(float(np.median(dump.positions[:, 1])) - cam_h)
        self.query_frames = [r.frame for r in rows]
        self.warnings: List[str] = []

        # gutter width: the strip's query / frame label and the row names
        fig_m = plt.figure(figsize=(FIG_W, FIG_H), dpi=72)
        n, T = len(rows), dump.frame_count
        widths = [cd.gutter_width_pt(fig_m, [self.CL["gt_row"], self.CL["pred_row"]], fs=FS["rowlab"], gap_pt=5.0)]
        for i, r in enumerate(rows):
            a, b = _gutter_texts(self.L, lang, i, n, r.frame, T)
            widths += [cd.text_width_pt(fig_m, a, FS["gutter"], fontweight="bold") + 0.07 * 72,
                       cd.text_width_pt(fig_m, b, FS["gutter_sub"]) + 0.07 * 72]
        gutter_right = MARGIN + max(widths) / 72.0 + 0.02

        # D4 per query: each held query shows one row, like one block of fig1, so its rows use that row's own
        # window (the same window, and the same blob shapes, as fig1's block of that row).  Plans at full width
        # and square degrees; the page holds the tallest query's rows, down to H_TOP_MIN for the top band; a
        # query whose rows still do not fit is drawn squeezed vertically (and says so in the gutter).
        lay0 = _geometry(0, 0.0, gutter_right)
        ppd = lay0.ppd
        wins = [cd.elevation_window([r], ARM) for r in rows]
        plans = [_plan_row(fig_m, r, win, ppd, self.CL) for r, win in zip(rows, wins)]
        note_lines = max(len(p.note_lines) for p in plans)
        lay = _geometry(note_lines, max(_rows_need(p.h_heat(ppd), p.below) for p in plans), gutter_right)
        room = lay.rows_room
        for i, (r, win) in enumerate(zip(rows, wins)):
            p = plans[i]
            if _rows_need(p.h_heat(ppd), p.below) <= room + 1e-9:
                continue
            h_sq = (win[1] - win[0]) * ppd / 72.0
            below = p.below
            for _ in range(3):  # the miss lane under the row may come or go with the scale
                v = (room - ROW_GAP - (MISS_LANE_H if below else 0.0) - AXIS_H) / (2.0 * h_sq)
                v = max(v, V_MIN)
                p = _plan_row(fig_m, r, win, ppd, self.CL, v=v)
                if p.below == below:
                    break
                below = p.below
            plans[i] = p
            if _rows_need(p.h_heat(ppd), p.below) > room + 1e-6:
                self.warnings.append(f"frame {r.frame + 1}: rows run past the foot of the page even at vertical "
                                     f"scale {p.v:.2f}")
        plt.close(fig_m)
        self.plans, self.lay = plans, lay
        if gutter_right > X_STRIP_MIN + 0.005:
            self.warnings.append(f"gutter labels need {gutter_right:.2f} in (> {X_STRIP_MIN:.2f}): the strip starts "
                                 f"later than in the other language")
        for p in plans:
            if p.win != (-cd.EL_DEFAULT, cd.EL_DEFAULT):
                self.warnings.append(f"frame {p.r.frame + 1}: elevation window {cd.fmt_deg(p.win[0])}.."
                                     f"{cd.fmt_deg(p.win[1])}" + (f", squeezed to vertical scale {p.v:.2f} to fit the "
                                                                  f"page" if p.v < 1.0 - 1e-6 else ""))
        if lay.note_lines:
            self.warnings.append(f"{lay.note_lines} note line(s) reserved above the lane")
        for p in plans:
            self.warnings += p.warnings
        w = int(round(lay.w_strip * dpi / 8.0)) * 8  # ring columns: ~1 per device pixel, multiple of 8 (exact roll)
        self.rgb_ring_w = max(w, 360)
        self.bar_corner, self.bar_corner_name, self.bar_gaps = self._pin_bar_corner(plt)
        # running totals over the rows shown so far (visible slots only)
        self.so_far = []
        errs, hits, n_vis = [], 0, 0
        for r in rows:
            v = r.visible
            e = r.arms[ARM].err[v]
            errs.extend([float(x) for x in e if np.isfinite(x)])
            hits += int((r.arms[ARM].joint8 & v).sum())
            n_vis += int(v.sum())
            self.so_far.append({"median": float(np.median(errs)) if errs else float("nan"), "hits": hits, "n": n_vis})

    def level_at(self, t: float):
        return self.scene_td.pick_level(self.track.height(t) - self.cam_h)

    def _pin_bar_corner(self, plt) -> Tuple[Tuple[float, float], str, Dict[str, float]]:
        """The one corner of the inset square where the local map's scale bar sits in every frame of the video
        (a bar that jumps corners between queries reads as a change of the map): the first of ``BAR_CORNERS``
        whose bar stays ``BAR_CLEAR_PT`` clear of every query's rim badges, leaders, direction arrow and
        sector letters (each query's panel b drawn without a bar, the bar tried in each corner at that query's
        scale), else the clearest.  The moving frames' disc has only the arrow (top left) and the four
        letters, so any of these corners is free there.  Returns ``((sx, sy), name, {name: worst gap pt})``.
        """
        lay = self.lay
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=self.dpi)
        worst = {c: float("inf") for c in BAR_CORNERS}
        for r in self.rows:
            ax = _ax(fig, lay.x_disc, Y_TOP, lay.w_disc, lay.w_disc)
            level = self.scene_td.pick_level(float(r.cur_pos[1]) - self.cam_h)
            info = _draw_inset(ax, level, self.dump, r, self.CL, None)
            obst = cd._rendered_obstacles(ax)
            for c in BAR_CORNERS:
                bar = _inset_bar(ax, info["half_m"], fc.INSET_CORNERS[c])
                worst[c] = min(worst[c], _bar_gap_pt(ax, bar, obst))
                for a in bar:
                    a.remove()
            ax.remove()
        plt.close(fig)
        ok = [c for c in BAR_CORNERS if worst[c] >= BAR_CLEAR_PT]
        pick = ok[0] if ok else max(BAR_CORNERS, key=lambda c: worst[c])
        gaps = {BAR_CORNERS[c]: round(worst[c], 2) for c in BAR_CORNERS}
        if not ok:
            self.warnings.append(f"no corner keeps the local map's scale bar {BAR_CLEAR_PT} pt clear in every "
                                 f"query (worst gaps {gaps} pt); {BAR_CORNERS[pick]} used")
        return tuple(float(v) for v in fc.INSET_CORNERS[pick]), BAR_CORNERS[pick], gaps

    @property
    def lay_info(self) -> dict:
        lay = self.lay
        wins = [list(p.win) for p in self.plans]
        return {"elevation_windows": wins, "elevation_window_default": [-cd.EL_DEFAULT, cd.EL_DEFAULT],
                "vertical_scale": [round(p.v, 3) for p in self.plans], "note_lines": lay.note_lines,
                "miss_lane": [bool(p.below) for p in self.plans], "strip_w_in": round(lay.w_strip, 3),
                "top_band_h_in": round(lay.h_top, 3), "disc_scale_bar_corner": self.bar_corner_name,
                "disc_scale_bar_worst_gap_pt": dict(self.bar_gaps)}


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for c in (shutil.which("ffmpeg"), "/opt/conda/bin/ffmpeg"):
        if c and os.access(c, os.X_OK):
            return c
    return None


class _Mp4:
    """H.264 yuv420p writer: PyAV if importable, else an ffmpeg executable fed raw RGB on stdin.

    ``encoder``: ``auto`` (PyAV, else ffmpeg), ``pyav`` or ``ffmpeg`` (force one backend).
    """

    X264_TAGS = "colorprim=bt709:transfer=bt709:colormatrix=bt709"

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 18, encoder: str = "auto"):
        if encoder not in ("auto", "pyav", "ffmpeg"):
            raise ValueError(f"encoder {encoder!r}: auto, pyav or ffmpeg")
        self.path, self.w, self.h = Path(path), width, height
        self.frames = 0
        self.backend = None
        tmp = self.path.with_name(self.path.stem + ".part.mp4")
        self.tmp = tmp
        try:
            if encoder == "ffmpeg":
                raise ImportError("ffmpeg forced")
            import av  # noqa: F401

            self._open_av(tmp, fps, crf)
        except ImportError:
            if encoder == "pyav":
                raise
            exe = _ffmpeg_exe()
            if exe is None:
                raise RuntimeError("no H.264 encoder: install PyAV (av) or imageio-ffmpeg, or put ffmpeg on PATH")
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                   "-r", str(fps), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
                   "-x264-params", self.X264_TAGS, "-color_primaries", "bt709", "-color_trc", "bt709",
                   "-colorspace", "bt709", "-movflags", "+faststart", str(tmp)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.backend = f"ffmpeg ({exe})"

    def _open_av(self, tmp: Path, fps: int, crf: int) -> None:
        import av
        from fractions import Fraction

        self.av = av
        self.time_base = Fraction(1, fps)
        self.container = av.open(str(tmp), mode="w", container_options={"movflags": "+faststart"})
        st = self.container.add_stream("libx264", rate=fps)
        st.codec_context.time_base = self.time_base
        st.width, st.height = self.w, self.h
        st.pix_fmt = "yuv420p"
        st.options = {"crf": str(crf), "preset": "medium", "x264-params": self.X264_TAGS}
        self.stream = st
        self.backend = f"pyav {av.__version__} libx264"

    def write(self, frame: np.ndarray, n: int = 1) -> None:
        if n <= 0:
            return
        if frame.shape[:2] != (self.h, self.w):
            raise ValueError(f"frame {frame.shape[:2]} != {(self.h, self.w)}")
        if hasattr(self, "stream"):
            vf = self.av.VideoFrame.from_ndarray(frame, format="rgb24")
            try:
                vf = vf.reformat(format="yuv420p", dst_colorspace="ITU709")
            except Exception:  # pragma: no cover - older PyAV: let the encoder convert (BT.601 matrix)
                pass
            for j in range(n):  # a held frame is re-sent with increasing timestamps
                vf.pts = self.frames + j
                vf.time_base = self.time_base
                for pkt in self.stream.encode(vf):
                    self.container.mux(pkt)
        else:
            data = frame.tobytes()
            for _ in range(n):
                self.proc.stdin.write(data)
        self.frames += n

    def close(self) -> Path:
        if hasattr(self, "stream"):
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
        else:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        os.replace(self.tmp, self.path)
        return self.path


def _reserved_colors() -> List[Tuple[int, int, int]]:
    """UI colours every GIF palette keeps exactly (badges, map ramps, ink): pixel counts cannot merge them away."""
    from matplotlib.colors import to_rgb

    cols = ["white", style.INK, style.INK_2, style.MUTED, style.AXIS, style.GRID, style.SURFACE, ROUTE_AHEAD,
            GT_INK, PRED_INK, cd.MISS_RING, cd.MAP_PLATE]
    cols += [style.history_color(k) for k in range(dd.K)] + [cd.history_line_color(k) for k in range(dd.K)]
    cols += [cd.GT_CMAP(cd.HEAT_TOP * i / 15) for i in range(16)] + [cd.PRED_CMAP(cd.HEAT_TOP * i / 15)
                                                                     for i in range(16)]
    out = []
    for c in cols:
        rgb = tuple(int(round(255 * v)) for v in to_rgb(c))
        if rgb not in out:
            out.append(rgb)
    return out


GIF_DITHER_AMP = 12.0  # ordered-dither amplitude (8-bit levels) in the affordance map rows of the GIF


def _bayer(n: int = 8) -> np.ndarray:
    """``n`` x ``n`` Bayer threshold matrix in [-0.5, 0.5)."""
    m = np.zeros((1, 1))
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size - 0.5


class _Gif:
    """Looping GIF: frames downscaled to ``width``; one palette per segment (reserved UI colours +
    adaptive colours of the segment's first frame), no error-diffusion dithering, so unchanged pixels
    stay unchanged.  The affordance map rows (``dither`` boxes) get an ordered (Bayer) dither anchored to
    the page, which breaks the palette's steps in the smooth ramps into a fine texture instead of rings
    and is the same in every frame for the same pixels.

    Frames are added with their start time on the video clock; each lasts until the next one.  255
    colours leave Pillow one free index, which ``optimize`` uses to make unchanged pixels of a delta
    frame transparent (runs of one index: the moving-robot frames cost little each).
    """

    def __init__(self, path: Path, width: int, colors: int = 255):
        self.path, self.width, self.colors = Path(path), int(width), min(int(colors), 255)
        self.frames: List = []
        self.starts: List[float] = []
        self._palette = None
        self._reserved = _reserved_colors()
        self.size: Optional[Tuple[int, int]] = None

    def _make_palette(self, im):
        from PIL import Image

        k = max(self.colors - len(self._reserved), 16)
        q = im.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        adaptive = q.getpalette()[: 3 * k]
        flat = [v for c in self._reserved for v in c] + adaptive
        flat = flat[: 3 * self.colors]
        pal = Image.new("P", (1, 1))
        pal.putpalette(flat)
        return pal

    def add(self, frame: np.ndarray, start_s: float, new_palette: bool,
            dither: Sequence[Tuple[float, float, float, float]] = ()) -> None:
        """``dither``: boxes (x0, y0, x1, y1) in ``frame`` pixels that get the ordered dither."""
        from PIL import Image

        h = int(round(frame.shape[0] * self.width / frame.shape[1] / 2)) * 2
        self.size = (self.width, h)
        im = Image.fromarray(frame).resize((self.width, h), Image.Resampling.LANCZOS)
        if new_palette or self._palette is None:
            self._palette = self._make_palette(im)
        if dither and GIF_DITHER_AMP > 0:
            a = np.asarray(im).astype(np.float32)
            sx, sy = self.width / frame.shape[1], h / frame.shape[0]
            B = _bayer(8) * GIF_DITHER_AMP
            for x0, y0, x1, y1 in dither:
                c0, c1 = max(int(round(x0 * sx)), 0), min(int(round(x1 * sx)), self.width)
                r0, r1 = max(int(round(y0 * sy)), 0), min(int(round(y1 * sy)), h)
                if c1 <= c0 or r1 <= r0:
                    continue
                block = a[r0:r1, c0:c1]
                vals, counts = np.unique(block.reshape(-1, 3), axis=0, return_counts=True)
                bg = vals[int(np.argmax(counts))]  # the row's empty surface stays one flat colour
                mask = np.any(np.abs(block - bg) > 2.0, axis=2)
                yy, xx = np.mgrid[r0:r1, c0:c1]
                block += (B[yy % 8, xx % 8] * mask)[..., None]
            im = Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))
        q = im.quantize(palette=self._palette, dither=Image.Dither.NONE)
        self.frames.append(q)
        self.starts.append(float(start_s))

    def close(self, end_s: float) -> Path:
        # GIF delays are in 1/100 s: round the clock, not each delay, so the total stays in step with the video
        cs = [int(round(100 * t)) for t in self.starts + [float(end_s)]]
        durations = [max(2, b - a) * 10 for a, b in zip(cs[:-1], cs[1:])]
        tmp = self.path.with_name(self.path.stem + ".part.gif")
        self.frames[0].save(tmp, save_all=True, append_images=self.frames[1:], duration=durations, loop=0,
                            optimize=True, disposal=1)
        os.replace(tmp, self.path)
        return self.path


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _prepare(dump_npz, topdown_root, clip_root_override, lang: str, size: Tuple[int, int],
             rows: Optional[Sequence[int]]):
    w_px, h_px = int(size[0]), int(size[1])
    if w_px * 9 != h_px * 16 or w_px % 2 or h_px % 2:
        raise ValueError(f"size {size}: need 16:9 with even sides (e.g. 1920x1080, 1280x720)")
    if lang not in LABELS:
        raise ValueError(f"lang {lang!r}: one of {sorted(LABELS)}")
    dpi = w_px / FIG_W
    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    dump = dd.load_dump(dump_npz)
    if ARM not in dump.arms:
        raise ValueError(f"{dump.path}: no {ARM!r} prediction (arms {dump.arms})")
    scored = dump.query_rows(ARM)
    if not scored:
        raise ValueError(f"{dump.path}: no scored query rows")
    if rows:
        bad = [i for i in rows if i not in scored]
        if bad:
            raise ValueError(f"rows {bad} are not scored rows {scored}")
        scored = sorted(set(int(i) for i in rows))
    recs = [dd.case_row(dump, i) for i in scored]
    clip_dir = dd.resolve_clip_dir(dump, clip_root_override)
    scene_td = load_topdown(dump.scene, root=topdown_root)
    cam_h = dump.camera_height()
    shown = [r.index for r in recs]
    schedule = ("subset" if shown != dump.query_rows(ARM) else
                "full" if shown == dump.query_rows(None) else "gaps")
    ctx = _Context(plt, dump, recs, lang, dpi, schedule, scene_td, cam_h)

    def build(i: int) -> _Scene:
        r = recs[i]
        views = dd.surround_views(clip_dir, r.frame)
        level = scene_td.pick_level(float(r.cur_pos[1]) - cam_h)
        return _Scene(plt, ctx, i, views, level)

    return plt, dump, recs, ctx, build, (w_px, h_px)


def _caption_flags(ctx: "_Context", stats: Sequence[dict]) -> Dict[str, bool]:
    """What the animation shows that the caption has to explain (``CAPTION_ORDER``'s conditions)."""
    plans = ctx.plans
    dotted = any(p.connectors and p.n_conn for p in plans)
    return {
        "rim_merged": any(s["rim_merged"] for s in stats),
        "rim_two_rings": any(s["rim_rings"] == 2 for s in stats),
        "groups": any(len(g) > 1 for p in plans for g in p.r.groups),
        "shared_miss": any(len(lab["slots"]) > 1 for p in plans for lab in p.placed["labels"].values()),
        "dotted": dotted,
        "dotted_dropped": dotted and any(not p.connectors for p in plans),  # a remark on the dotted lines
        "miss_in_notes": any(s["in_notes"] for s in stats),
        "notes": any(p.items for p in plans),
    }


def _elev_caption(ctx: "_Context", lang: str) -> str:
    """The rows' elevation windows for the caption: the most common one, and each query that differs (D4)."""
    E = ELEV_CAPTION[lang]
    wins = [tuple(p.win) for p in ctx.plans]
    base = max(dict.fromkeys(wins), key=wins.count)  # the most common window (ties: the first shown)
    items = []
    for i, p in enumerate(ctx.plans):
        lo, hi = cd.fmt_deg(p.win[0]), cd.fmt_deg(p.win[1])
        if p.v < 1.0 - 1e-6:
            items.append(E["item_sq"].format(i=i + 1, lo=lo, hi=hi, v=p.v))
        elif tuple(p.win) != base:
            items.append(E["item"].format(i=i + 1, lo=lo, hi=hi))
    lo, hi = cd.fmt_deg(base[0]), cd.fmt_deg(base[1])
    if not items:
        return E["all"].format(lo=lo, hi=hi)
    return E["mixed"].format(lo=lo, hi=hi, items=E["sep"].join(items))


def _caption(ctx: "_Context", lang: str, stats: Sequence[dict]) -> str:
    """The caption: ``CAPTION_PARTS`` in ``CAPTION_ORDER``, each conditional part only when the animation shows
    what it explains (``_caption_flags``)."""
    dump, recs = ctx.dump, ctx.rows
    flags = _caption_flags(ctx, stats)
    P = CAPTION_PARTS[lang]
    text = "".join(P[k] for k, cond in CAPTION_ORDER if cond is None or flags[cond])
    return text.format(tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id, n=len(recs),
                       sched=CAPTION_SCHEDULE[lang][ctx.schedule], elev=_elev_caption(ctx, lang),
                       first=cd.frame_label(recs[0].frame, dump.frame_count, lang)).strip()


def _row_stats(scene: _Scene) -> dict:
    r = scene.r
    return {"row": r.index, "frame": r.frame, "frame_label": cd.frame_label(r.frame, scene.ctx.dump.frame_count),
            ARM: r.summary(ARM), "floor": r.summary("floor"), "misses": [k + 1 for k in r.misses(ARM)],
            "half_m": round(scene.half, 3), "rim_spread_deg": round(scene.inset["rim_spread_deg"], 1),
            "rim_spread_drawn_deg": round(scene.inset["rim_spread_drawn_deg"], 1),
            "rim_rings": scene.inset["rings"], "rim_merged": scene.inset["rim_merged"],
            "connectors": scene.plan.connectors, "elevation_window": list(scene.plan.win),
            "vertical_scale": round(scene.plan.v, 3),
            "longest_leader_pt": round(scene.plan.leader_pt, 1), **scene.check, **scene.drawn,
            "row_warnings": list(scene.warnings)}  # (not "warnings": make_all collects those at any depth,
    # and the top-level list already carries them)


def make_animation(dump_npz, out_stem="anim", topdown_root=None, clip_root_override=None, fps: int = 15,
                   lang: str = "en", size: Tuple[int, int] = (1920, 1080), hold_s: float = 2.0,
                   end_hold_s: float = 3.0, substeps: int = 2, max_move_s: float = 3.0,
                   rows: Optional[Sequence[int]] = None, gif: bool = True, gif_width: int = 1280,
                   gif_step: int = 1, stills: bool = False, crf: int = 18, encoder: str = "auto") -> dict:
    """Render the supplementary animation of one dump.

    Returns ``{"files": [...], "rows": [...], "frames": n, "duration_s": s, "size": (w, h), "fps": fps,
    "encoder": str, "render_s": s, "layout": {...}, "warnings": [...], "stats": [...]}``; each stats entry
    has the row's numbers (``vo`` / ``floor`` summaries), its misses, the disc radius ``half_m``, the rim
    badges' largest dodge, the slots numbered on the prediction row / in notes, the notes and the row's
    warnings.  ``rows`` restricts the query rows (default: every scored row of the deployed model's
    output); ``size`` must be 16:9 with even sides.  Raises (``data.check_accounting``) if a slot of a row
    would end up with neither a badge nor a note, or the numbered misses differ from joint PCK@8's.
    """
    t_start = time.time()
    fps, substeps = int(fps), max(1, int(substeps))
    plt, dump, recs, ctx, build, (w_px, h_px) = _prepare(dump_npz, topdown_root, clip_root_override, lang, size,
                                                         rows)
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = _Mp4(out.parent / (out.name + ".mp4"), w_px, h_px, fps, crf=crf, encoder=encoder)
    gw = _Gif(out.parent / (out.name + ".gif"), gif_width) if gif else None
    files: List[str] = []
    stats: List[dict] = []

    def move(scene: _Scene, nxt: _Scene) -> None:
        t0, t1 = float(scene.r.frame), float(nxt.r.frame)
        n_clip = max(int(round(t1 - t0)), 1)
        n_vid = min(n_clip * substeps, max(int(max_move_s * fps), 1))
        last_gif_t = None
        for j in range(1, n_vid):
            t = t0 + (t1 - t0) * j / n_vid
            img = scene.render(t, "move", nxt)
            # GIF: the first moving frame (the veil appears), then one frame per ``gif_step`` clip frames
            if gw is not None and (last_gif_t is None or t - last_gif_t >= gif_step - 1e-6):
                gw.add(img, mp4.frames / fps, new_palette=last_gif_t is None, dither=scene.heat_boxes_px)
                last_gif_t = t
            mp4.write(img)

    scene = build(0)
    for i, r in enumerate(recs):
        last = i == len(recs) - 1
        img = scene.render(float(r.frame), "hold")
        if gw is not None:
            gw.add(img, mp4.frames / fps, new_palette=True, dither=scene.heat_boxes_px)
        mp4.write(img, int(round((end_hold_s if last else hold_s) * fps)))
        if stills:
            from PIL import Image

            p = out.parent / f"{out.name}_q{i:02d}_f{r.frame + 1:03d}.png"
            Image.fromarray(img).save(p)
            files.append(str(p))
        stats.append(_row_stats(scene))
        nxt = None
        if not last:
            nxt = build(i + 1)
            move(scene, nxt)
        scene.close(plt)
        scene = nxt

    files.insert(0, str(mp4.close()))
    if gw is not None:
        files.insert(1, str(gw.close(mp4.frames / fps)))
    cap = out.parent / (out.name + "_caption.txt")
    caption = _caption(ctx, lang, stats)
    cap.write_text(caption + "\n", encoding="utf-8")
    files.insert(2 if gw is not None else 1, str(cap))
    warnings = list(ctx.warnings) + [w for s in stats for w in s["row_warnings"] if w not in ctx.warnings]
    return {"files": files, "rows": [r.index for r in recs], "frames": mp4.frames,
            "duration_s": mp4.frames / fps, "size": (w_px, h_px), "fps": fps, "encoder": mp4.backend,
            "gif_size": gw.size if gw is not None else None, "render_s": time.time() - t_start,
            "layout": ctx.lay_info, "caption": caption, "caption_flags": _caption_flags(ctx, stats),
            "warnings": warnings, "notes_dropped": [], "stats": stats}


def render_stills(dump_npz, out_dir, stem: str = "anim", topdown_root=None, clip_root_override=None,
                  lang: str = "en", size: Tuple[int, int] = (1920, 1080), rows: Optional[Sequence[int]] = None,
                  moves: Optional[Sequence[int]] = None, move_frac: float = 0.5) -> dict:
    """The animation's query-row frames as PNGs (no video), plus the frame ``move_frac`` of the way to the next
    query after each query index in ``moves`` (default: every gap).  For review; same drawing as the video."""
    t_start = time.time()
    plt, dump, recs, ctx, build, _ = _prepare(dump_npz, topdown_root, clip_root_override, lang, size, rows)
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    moves = set(range(len(recs) - 1) if moves is None else moves)
    files, stats = [], []
    scene = build(0)
    for i, r in enumerate(recs):
        p = out / f"{stem}_q{i:02d}_f{r.frame + 1:03d}.png"
        Image.fromarray(scene.render(float(r.frame), "hold")).save(p)
        files.append(str(p))
        stats.append(_row_stats(scene))
        nxt = None
        if i < len(recs) - 1:
            nxt = build(i + 1)
            if i in moves:
                t = r.frame + move_frac * (nxt.r.frame - r.frame)
                p = out / f"{stem}_q{i:02d}_move_f{int(round(t)) + 1:03d}.png"
                Image.fromarray(scene.render(t, "move", nxt)).save(p)
                files.append(str(p))
        scene.close(plt)
        scene = nxt
    cap = out / f"{stem}_caption.txt"
    caption = _caption(ctx, lang, stats)
    cap.write_text(caption + "\n", encoding="utf-8")
    files.append(str(cap))
    warnings = list(ctx.warnings) + [w for s in stats for w in s["row_warnings"] if w not in ctx.warnings]
    return {"files": files, "rows": [r.index for r in recs], "layout": ctx.lay_info, "caption": caption,
            "caption_flags": _caption_flags(ctx, stats), "warnings": warnings, "notes_dropped": [], "stats": stats,
            "render_s": time.time() - t_start}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True, help="History Head dump npz of one clip")
    ap.add_argument("--out", default="anim", help="output stem (writes .mp4, .gif, _caption.txt)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    ap.add_argument("--size", default="1920x1080", help="WxH, 16:9 (e.g. 1920x1080, 1280x720)")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds each query row is held")
    ap.add_argument("--end-hold", type=float, default=3.0, help="seconds the last row is held")
    ap.add_argument("--substeps", type=int, default=2, help="video frames per clip frame while moving")
    ap.add_argument("--rows", default=None, help="comma-separated query rows (default: every scored row)")
    ap.add_argument("--gif-width", type=int, default=1280)
    ap.add_argument("--gif-step", type=int, default=1, help="clip frames per GIF frame while moving")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--stills", action="store_true", help="also save each query row's frame as PNG")
    ap.add_argument("--stills-only", action="store_true", help="only the query-row and mid-move PNGs, no video")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, larger)")
    ap.add_argument("--encoder", default="auto", choices=("auto", "pyav", "ffmpeg"))
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    if args.stills_only:
        out = Path(args.out)
        res = render_stills(args.dump, out.parent, stem=out.name, topdown_root=args.topdown_root,
                            clip_root_override=args.clip_root, lang=args.lang, size=(w, h), rows=rows)
    else:
        res = make_animation(args.dump, out_stem=args.out, topdown_root=args.topdown_root,
                             clip_root_override=args.clip_root, fps=args.fps, lang=args.lang, size=(w, h),
                             hold_s=args.hold, end_hold_s=args.end_hold, substeps=args.substeps, rows=rows,
                             gif=not args.no_gif, gif_width=args.gif_width, gif_step=args.gif_step,
                             stills=args.stills, crf=args.crf, encoder=args.encoder)
    for f in res["files"]:
        print(f)
    for s in res["stats"]:
        print(s)
    for wmsg in res["warnings"]:
        print(f"warning: {wmsg}")
    if "frames" in res:
        print(f"frames {res['frames']}  duration {res['duration_s']:.1f} s  {res['size'][0]}x{res['size'][1]} "
              f"@ {res['fps']} fps  encoder {res['encoder']}  render {res['render_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
