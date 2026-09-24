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

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_case --dump <clip.npz> [--rows 0,2,8] [--topdown-root DIR]
      [--clip-root DIR] [--lang en|zh] --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import style
from scripts.exp18.topdown.topdown_io import forward_from_c2w

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
        "(orange); each slot's map is divided by its own peak (the prediction also multiplied by its predicted "
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
        "预测 affordance map（橙）；每个槽位的图除以自身峰值（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值"
        "线性着色。图在视角分界处截断，因为每个标签和预测都只落在一个 90° 视角里。×：各槽位的预测峰值（4° 内合并，相互挨着"
        "的上下错开）；偏离真值方位 5° 以上的单独编号。预测行下方的蓝色短线重复真值方位。行首：可见槽位上预测峰值的方位"
        "误差（中位、最大）与 joint PCK@8，以及作参照的恒答正后方基线。"
    ),
}

# The prediction drawn is always the deployed model's output (dump arm "vo").
ARM = "vo"

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


def fig_height(n_blocks: int) -> float:
    return TOP_H + n_blocks * BLOCK_H + (n_blocks - 1) * BLOCK_GAP + AXIS_H + LEGEND_H


class Page:
    """Axes placement in inches from the top-left corner."""

    def __init__(self, fig, height: float):
        self.fig = fig
        self.h = height

    def ax(self, x: float, y_top: float, w: float, h: float, **kw):
        return self.fig.add_axes([x / FIG_W, 1 - (y_top + h) / self.h, w / FIG_W, h / self.h], **kw)

    def text(self, x: float, y: float, s: str, **kw):
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


def draw_inset(ax, level, dump: dd.Dump, r: dd.CaseRow, show_arrow: bool, L: dict) -> None:
    fwd = forward_from_c2w(r.cur_c2w)
    far = float(np.max(r.gt_dist[r.visible])) if r.visible.any() else 1.0
    half = max(1.0, 1.12 * far)
    past = dump.positions[: r.frame + 1][:, [0, 2]]
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


def draw_block(page: Page, y_top: float, n: int, r: dd.CaseRow, views: np.ndarray, arm: str, L: dict,
               last: bool):
    fig = page.fig
    # ---- header: K badge + frame (inset column), metrics (right-aligned over the strip)
    y_mid = y_top + HDR_H * 0.45
    page.text(X_INSET, y_mid, f"K{n + 1}", ha="left", va="center", fontsize=FS["header"], fontweight="bold",
              color="white", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.3", fc=style.INK, ec="none"))
    frame_txt = (L["frame_last"] if r.is_final else L["frame"]).format(t=r.frame)
    page.text(X_INSET + 0.25, y_mid, frame_txt, ha="left", va="center", fontsize=FS["header"], color=style.INK)
    main, rest = _metrics_text(r, arm, L)
    t_rest = page.text(X_STRIP + W_STRIP, y_mid, rest, ha="right", va="center", fontsize=FS["header"],
                       color=style.MUTED)
    w_rest = t_rest.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    page.text(X_STRIP + W_STRIP - w_rest, y_mid, main, ha="right", va="center", fontsize=FS["header"],
              color=style.INK)

    # ---- axes
    y_lane = y_top + HDR_H
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
    v_label = cd.quietest_panel(gt_strip, pr_strip)
    for ax, label in ((ax_gt, L["gt_row"]), (ax_pr, L["pred_row"])):
        ax.text(v_label * 90 + 2.0, 0, label, ha="left", va="center", fontsize=FS["small"], color=style.INK_2,
                zorder=6, path_effects=cd.HALO)
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
    for (x, el, k), off in zip(marks, offsets):
        cd.peak_mark(ax_pr, x, el + off * per_pt_y, size=MARK_PT)
        drawn.append((x, 0.0, k))
    texts = {}
    for k in alone:
        if not r.visible[k]:
            texts[k] = L["false_pos"]
    _place_peak_labels(ax_pr, fig, pr_strip, drawn, pt_per_deg, texts)

    # ---- notes (slot badge + text) for slots without a GT view or predicted "not visible"
    notes = []
    for k in r.invisible_slots():
        notes.append((k, L["current" if r.gt_dist[k] < 0.1 else "not_visible"].format(p=p.none_p[k])))
    for k in range(dd.K):
        if r.visible[k] and p.none_p[k] > 0.5:
            notes.append((k, L["pred_none"].format(p=p.none_p[k])))
    # notes go left of the lane badges, else right of them; a note that fits nowhere is reported, not drawn
    free = [(1.0, min([a for a, _ in lane_spans], default=360.0) - 2.0),
            (max([b for _, b in lane_spans], default=0.0) + 2.0, 359.0)]
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

    if last:
        cd.azimuth_axis(ax_pr, L["axis"], fs=FS["axis"])
        ax_pr.tick_params(axis="x", which="major", pad=5.5)
    return ax_in


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
def make_case_figure(dump_npz_path, rows: Optional[List[int]] = None, topdown_root=None, clip_root_override=None,
                     out_stem="case", lang: str = "en") -> dict:
    """Render the case figure for one dump; returns {"files": [...], "rows": [...], "stats": [...]}."""
    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    L = LABELS[lang]
    arm = ARM
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

    height = fig_height(len(recs))
    fig = plt.figure(figsize=(FIG_W, height))
    page = Page(fig, height)
    draw_top_band(page, L)
    insets = []
    for n, r in enumerate(recs):
        y_top = TOP_H + n * (BLOCK_H + BLOCK_GAP)
        views = dd.surround_views(clip_dir, r.frame)
        insets.append(draw_block(page, y_top, n, r, views, arm, L, last=(n == len(recs) - 1)))
    for n, (ax_in, r) in enumerate(zip(insets, recs)):
        lvl = dd.topdown_level(dump, float(r.cur_pos[1]), root=topdown_root)
        draw_inset(ax_in, lvl, dump, r, show_arrow=(n == 0), L=L)

    y_route = TOP_H + HDR_H
    y_axis = TOP_H + len(recs) * BLOCK_H + (len(recs) - 1) * BLOCK_GAP
    scene_h = 0.20  # two lines under the map: tier + scene, episode + length
    ax_route = page.ax(X_ROUTE, y_route, W_ROUTE, y_axis + AXIS_H - scene_h - y_route)
    draw_route_panel(ax_route, level, dump, recs, L)
    page.text(X_ROUTE, y_axis + AXIS_H - scene_h + 0.035, L["scene"].format(
        tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id, T=dump.frame_count),
        ha="left", va="top", fontsize=FS["note"], color=style.MUTED, linespacing=1.15)
    draw_legend(page, y_axis + AXIS_H, arm, L)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)
    caption = CAPTION[lang].format(n=len(recs), tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id)
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption + "\n", encoding="utf-8")
    files.append(cap_path)
    stats = []
    for n, r in enumerate(recs):
        entry = {"key": f"K{n + 1}", "row": r.index, "frame": r.frame}
        for a in list(r.arms) + ["floor"]:
            entry[a] = r.summary(a)
        stats.append(entry)
    return {"files": [str(f) for f in files], "rows": rows, "stats": stats, "size_in": (FIG_W, height)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True, help="History Head dump npz of one clip")
    ap.add_argument("--rows", default=None, help="comma-separated query rows (default: pre-registered key rows)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--out", default="case", help="output stem (writes .pdf, .png, _caption.txt)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    args = ap.parse_args(argv)
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    res = make_case_figure(args.dump, rows=rows, topdown_root=args.topdown_root, clip_root_override=args.clip_root,
                           out_stem=args.out, lang=args.lang)
    for f in res["files"]:
        print(f)
    for s in res["stats"]:
        print(s)
    print("size_in", tuple(round(v, 3) for v in res["size_in"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
