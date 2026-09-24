#!/usr/bin/env python3
"""EXP-18 supplementary animation: the predicted affordance map vs ground truth along one episode.

The video starts on the first query row.  One video frame per scored query row
(cache endpoints: every 8 frames and the last frame) is held for ``hold_s``
seconds; between two query rows the robot glides along the route (one step per
clip frame, ``substeps`` video frames per step, linear in position and
shortest-arc in heading).  While it moves, the route and the local map follow
it live (the map turns with the heading and zooms smoothly, log-linear with
ease-in-out, from one query's scale to the next one's), the previous query's
strip and numbers are faded and labelled with their frame in the left gutter,
and the timeline names the next query frame.  Heat is never interpolated:
every affordance map shown is one query row's.

Frame layout (16:9, drawn on an 8.0 x 4.5 in page; 1920 x 1080 = 240 dpi):

  Predicted affordance map vs ground truth                  tier scene · episode · frames
  Route                    Local map, heading up           frame t of T
  +-------------------+    ( disc, rim = bearing )         query i of n
  | top-down map      |                                    |--|--|--v--|--|  timeline (+ next query)
  | route: dark = done|                                    metrics of the row, reference
  | robot, past pos.  |                                    legend
  +-------------------+
                [ bracket: images not given to the model (display only) ]
                Front · model input | Right | Back | Left
                lane: numbered badges at the true bearings (+ notes)
  frame t       RGB row (front in colour and framed; the others washed out)
  ground truth  ground-truth affordance map row (blue)
  prediction    predicted affordance map row (orange), x = predicted peaks
                0° (heading)   −90°   180°   +90°

The visual language is ``fig_case``'s (see its module doc and
``common_draw``): blue = ground truth, orange = prediction, numbered history
badges (1 = oldest of the K = 8 queried past positions), black = the robot.
The local disc is ``fig_case.draw_inset`` at query rows and
``common_draw.draw_local_disc`` while moving; the strip repeats
``fig_case.draw_block`` on a full-width strip with the paper figure's
elevation extents (``fig_case.EL_RGB`` / ``EL_HEAT``), so the blobs have the
same shapes as in fig1.  Numbered x (the miss rule, consistent with the numbers
printed beside the strip): a predicted peak is drawn alone and labelled with
its slot number when that past position is a joint PCK@8 miss (or is not
visible at all); the label is a white badge ringed in the prediction's orange,
never a blue ground-truth badge.  So on every row, numbered x plus "predicted
not visible" notes = n - hits of the printed PCK@8.  Hits within
``fig_case.MERGE_DEG`` share one x; marks that would touch are staggered
vertically; numbered labels stay inside the prediction row (three lanes,
nearest free spot, a thin leader when moved).  On the route map, the grey line
is the whole route, the dark line the part driven so far, open circles the
query positions, and blue dots (light = older) the K past positions of the
query row on screen.

Figure policy (user decision): the animation shows the affordance map only.
It never mentions poses, odometry or the pose-source ablation; the prediction
drawn is always the deployed model's output (the dump's ``vo`` arm, as in
``fig_case.ARM``).  Honesty kept on every frame: of the four current views
only the front image is marked as model input, the other three are marked
display-only; ground truth blue vs prediction orange, never swapped; misses
are drawn and numbered; the constant "always behind" guess is printed as a
reference.

Frame numbers are the clip's own (0-based): "frame 27 of 79" is frame index 27
of a 79-frame episode, as in the gallery.

Encoding: H.264 MP4 (yuv420p, BT.709 tagged, +faststart) through PyAV
(``av``, bundled with its own FFmpeg + libx264, so it works in a blank
container), else an ``ffmpeg`` executable (imageio-ffmpeg's, ``$PATH`` or
``/opt/conda/bin/ffmpeg``).  A looping GIF (``gif_width`` px wide, default
1280 so badge digits and small notes stay legible; one frame per row and per
clip step, one palette per segment, no dithering) is written with Pillow.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_anim --dump <clip.npz> --out <dir/stem> [--fps 15] [--lang en|zh]
      [--size 1920x1080] [--hold 2.0] [--substeps 2] [--rows 0,1,2] [--gif-width 1280] [--no-gif]
      [--stills] [--topdown-root DIR] [--clip-root DIR]
Writes <stem>.mp4, <stem>.gif (unless --no-gif), <stem>_caption.txt and with
--stills one PNG per query row (<stem>_q<ii>_f<frame>.png, full resolution).
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18 import geometry as geo
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
        "route": "Route",
        "disc": "Local map, heading up",
        "frame": "frame {t} of {T}",
        "frame_last": "frame {t} of {T} (last)",
        "query": "query {i} of {n}",
        "schedule": "  (every 8 frames and the last frame)",
        "faded": "faded panels: query {i} at frame {t}",
        "next": "next query: frame {t}",
        "strip_frame": "frame {t}",
        "strip_query": "query {i} of {n}",
        "err": "bearing error  median {med:.1f}°, max {mx:.1f}°",
        "pck": "PCK@8  {hits}/{n} past positions",
        "none_visible": "no past position visible in any view",
        "floor": "always-behind guess: median {med:.0f}°, PCK@8 {hits}/{n}",
        "so_far": "episode so far: median {med:.1f}°, PCK@8 {hits}/{n}",
        "gt_row": "ground-truth\naffordance map",
        "pred_row": "predicted\naffordance map",
        "legend_peak": "predicted peak; numbered = PCK@8 miss",
        "legend_query": "query position (route)",
    },
    "zh": {
        "title": "预测 affordance map 与真值对比",
        "sub": "{tier} {scene}  ·  第 {ep} 集  ·  {T} 帧",
        "route": "路线",
        "disc": "局部地图（机器人朝上）",
        "frame": "第 {t} 帧（共 {T} 帧）",
        "frame_last": "第 {t} 帧（共 {T} 帧，末帧）",
        "query": "第 {i} 次查询，共 {n} 次",
        "schedule": "（每 8 帧及末帧）",
        "faded": "淡化画面：第 {i} 次查询（第 {t} 帧）",
        "next": "下一次查询：第 {t} 帧",
        "strip_frame": "第 {t} 帧",
        "strip_query": "第 {i}/{n} 次查询",
        "err": "方位误差  中位 {med:.1f}°，最大 {mx:.1f}°",
        "pck": "PCK@8  {hits}/{n} 个历史位置",
        "none_visible": "任何视角都看不到历史位置",
        "floor": "恒答正后方：中位 {med:.0f}°，PCK@8 {hits}/{n}",
        "so_far": "本集累计：中位 {med:.1f}°，PCK@8 {hits}/{n}",
        "gt_row": "真值\naffordance map",
        "pred_row": "预测\naffordance map",
        "legend_peak": "预测峰值；编号 = PCK@8 未命中",
        "legend_query": "查询位置（路线上）",
    },
}

CAPTION_SCHEDULE = {  # which query frames the animation holds: every endpoint / those with an output / a pick
    "en": {"full": " (every 8 frames and the last frame)",
           "gaps": " (every 8 frames and the last frame, where the model has an output)",
           "subset": " (a subset of the query frames, chosen for this animation)"},
    "zh": {"full": "（每 8 帧及末帧）", "gaps": "（每 8 帧及末帧中模型有输出的帧）", "subset": "（本动画只取了部分查询帧）"},
}
CAPTION = {
    "en": (
        "Supplementary animation: predicted affordance map vs ground truth along one episode ({tier} {scene}, "
        "episode {ep}, {n} query frames; frames are numbered from 0). Each held frame is one query{sched}. Between "
        "queries the robot moves along the route and the local map follows it (turning with the heading and zooming "
        "smoothly to the next query's scale), while the previous query's strip and numbers are faded and labelled "
        "with their frame at the left, because predictions exist only at query frames; the timeline names the next "
        "query. Left: route on the top-down map (dark = driven so far, open circles = query positions, blue dots = "
        "the 8 past positions of the query on screen, light = older). Middle: map around the robot, heading up; its "
        "rim is the bearing ring the strip below unrolls clockwise from the front view's left edge; blue lines and "
        "numbers (1 = oldest) mark the past positions. Bottom: the surround view on the same bearings (of the four "
        "current views, only the framed front view is given to the model; right, back and left are display only), "
        "the ground-truth affordance map (blue) and the predicted affordance map (orange; each slot's map divided by "
        "its own peak, the prediction also multiplied by its predicted visibility, maximum over slots). x = predicted "
        "peak; a past position that joint PCK@8 counts as a miss has its x drawn alone and numbered in an "
        "orange-ringed badge, so the numbered x plus the 'predicted not visible' notes are exactly the misses in the "
        "printed PCK@8. Blue ticks under the prediction repeat the true bearings. The rows span ±{el:g}° of elevation "
        "as in the paper figure; a blue caret at a row edge marks a ground-truth peak beyond it (other levels, "
        "stairs), and an x beyond it sits at the edge with a small black caret. Right: bearing error of the "
        "predicted peaks over visible past positions (median, max), joint PCK@8, the constant 'always behind' guess "
        "for reference, and the running totals of the episode."
    ),
    "zh": (
        "补充动画：同一集（{tier} {scene}，第 {ep} 集，{n} 个查询帧；帧号从 0 起）上预测 affordance map 与真值的对比。"
        "每个停留画面是一次查询{sched}。两次查询之间机器人沿路线移动，局部地图随之转动并平滑缩放到下一次查询的比例；"
        "上一次查询的条带与数值变淡，并在左侧标出其帧号，因为只有查询帧才有预测；时间轴上标出下一次查询。左：俯视图上的"
        "路线（深色 = 已走过，空心圆 = 查询位置，蓝点 = 当前画面那次查询的 8 个历史位置，浅 = 更早）。中：机器人周围的"
        "局部地图，机器人朝上；圆周就是下方条带从前视左缘顺时针展开的方位环，蓝线与编号（1 = 最早）标出历史位置。下：同一"
        "方位轴上的环视（当前四个视角中只有加框的前视图是模型输入，右/后/左仅作展示）、真值 affordance map（蓝）与预测 "
        "affordance map（橙；每个槽位的图除以自身峰值，预测再乘以其预测可见概率，取各槽位最大值）。× = 预测峰值；joint "
        "PCK@8 判为未命中的历史位置，其 × 单独画出并用橙色圈的编号标注，因此编号的 × 加上“预测为不可见”的注记恰好就是"
        "所示 PCK@8 中的未命中数。预测行下方的蓝色短线重复真值方位。两行与论文图一样覆盖 ±{el:g}° 俯仰；行边缘的蓝色"
        "尖角表示超出该范围的真值峰值（其他楼层、楼梯），超出范围的 × 画在边缘并带黑色小尖角。右：可见历史位置上预测峰值"
        "的方位误差（中位、最大）、joint PCK@8、作参照的恒答正后方基线，以及本集累计。"
    ),
}

ARM = fc.ARM  # the deployed model's output; the only prediction ever drawn

# --------------------------------------------------------------------------- #
# Page geometry (inches, origin top-left) on an 8.0 x 4.5 in page
# --------------------------------------------------------------------------- #
FIG_W, FIG_H = 8.0, 4.5
MARGIN = 0.10
Y_TITLE = 0.20
Y_NAMES = 0.46
X_STRIP = 0.84  # left of it: the row labels
W_STRIP = FIG_W - MARGIN - X_STRIP
# Rows in square degrees with the paper figure's elevation extents, so the video and fig1 show the same blob
# shapes; ground-truth peaks beyond the heat rows' band get a caret at the row edge.
EL_RGB = fc.EL_RGB
EL_HEAT = fc.EL_HEAT
H_RGB = W_STRIP * 2 * EL_RGB / 360.0
H_HEAT = W_STRIP * 2 * EL_HEAT / 360.0
ROW_GAP = 0.035
H_LANE = 0.17
Y_PR = FIG_H - 0.24 - H_HEAT  # bottom-up: axis labels under the prediction row
Y_GT = Y_PR - ROW_GAP - H_HEAT
Y_RGB = Y_GT - ROW_GAP - H_RGB
Y_LANE = Y_RGB - H_LANE
Y_VIEWS = Y_LANE - 0.075
Y_BRACKET = Y_VIEWS - 0.155
Y_TOP = 0.55
H_TOP = Y_BRACKET - 0.07 - Y_TOP  # route / disc / info band
X_ROUTE, W_ROUTE = MARGIN, 2.80
X_DISC, W_DISC = X_ROUTE + W_ROUTE + 0.12, H_TOP
X_INFO = X_DISC + W_DISC + 0.13
W_INFO = FIG_W - MARGIN - X_INFO
HEAT_RING_W = 1440
FS = {"title": 11.0, "sub": 7.0, "name": 7.4, "frame": 9.6, "info": 7.0, "muted": 6.5, "legend": 6.8,
      "rowlab": 6.6, "bracket": 6.4, "small": 6.0, "note": 6.2, "axis": 6.6, "views": 7.2, "next": 6.6,
      "gutter": 7.2, "gutter_sub": 6.2}
ROUTE_AHEAD = "#b9b7ae"  # the part of the route still ahead
GT_INK = "#1c5cab"  # label text in the ground-truth / prediction ramps' dark tones
PRED_INK = "#b53f12"
FADE_ALPHA = 0.66  # white veil over the previous query's strip and numbers while the robot moves
ROBOT_PT = 8.6
DISC_ROBOT_PT = 6.6  # as fig_case.draw_inset
RADIUS_FRAC = inspect.signature(cd.draw_local_disc).parameters["radius_frac"].default  # disc radius / axes half
PEAK_EDGE_PT = fc.MARK_PT / 2 + 0.6  # an x is kept this far (its half width) inside the prediction row
LABEL_CLEAR_PT = 3.0  # a numbered label with no leader keeps this gap to every x other than its own


def _ax(fig, x: float, y_top: float, w: float, h: float, **kw):
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H], **kw)


def _text(fig, x: float, y: float, s: str, **kw):
    return fig.text(x / FIG_W, 1 - y / FIG_H, s, **kw)


def _rect(fig, x0: float, y0: float, x1: float, y1: float, **kw) -> Rectangle:
    """Figure-level rectangle given in page inches (top-left origin)."""
    r = Rectangle((x0 / FIG_W, 1 - y1 / FIG_H), (x1 - x0) / FIG_W, (y1 - y0) / FIG_H, transform=fig.transFigure,
                  **kw)
    fig.add_artist(r)
    return r


def pred_badge(ax, x: float, y: float, label: str, zorder: float = 8, fs: float = cd.BADGE_FS):
    """Slot number on the prediction row: white badge ringed in the prediction's orange (not a blue GT badge).

    Same box as ``cd.history_badge`` (so ``cd.badge_width_pt`` holds).
    """
    box = "circle,pad=0.22" if len(label) == 1 else "round,pad=0.24,rounding_size=0.62"
    return ax.text(x, y, label, ha="center", va="center", fontsize=fs, fontweight="bold", color=PRED_INK,
                   zorder=zorder, bbox=dict(boxstyle=box, fc="white", ec=PRED_INK, lw=0.8))


def numbered_slots(r: dd.CaseRow, arm: str = ARM) -> List[int]:
    """Slots whose predicted peak is drawn alone and numbered: an x is shown (P(not visible) <= 0.5 and a peak)
    and the slot is a joint PCK@8 miss, or is not visible in any view (a false positive)."""
    p = r.arms[arm]
    return [k for k in range(dd.K) if r.valid[k] and p.none_p[k] <= 0.5 and p.peak_view[k] >= 0
            and ((not r.visible[k]) or not bool(p.joint8[k]))]


# --------------------------------------------------------------------------- #
# Robot along the route (clip frames; fractional t interpolates)
# --------------------------------------------------------------------------- #
class _Track:
    def __init__(self, dump: dd.Dump):
        self.xz = dump.positions[:, [0, 2]].astype(np.float64)
        self.y = dump.positions[:, 1].astype(np.float64)
        f = forward_from_c2w(dump["clip_c2w"].astype(np.float64))
        self.ang = np.unwrap(np.arctan2(f[:, 1], f[:, 0]))  # continuous heading, map (x, z) plane
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
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    bar = cd.nice_length(0.3 * (limits[1] - limits[0]))
    cd.scale_bar(ax, limits[0] + 6 * per_pt, limits[2] + 7.5 * per_pt, bar, f"{bar:g} m", fs=6.0)
    return limits


def _row_labels(fig, L: dict) -> None:
    x = X_STRIP - 0.07
    for y_top, text, color, cmap in ((Y_GT, L["gt_row"], GT_INK, cd.GT_CMAP), (Y_PR, L["pred_row"], PRED_INK,
                                                                                cd.PRED_CMAP)):
        yc = y_top + H_HEAT / 2
        _text(fig, x, yc - 0.03, text, ha="right", va="center", fontsize=FS["rowlab"], color=color,
              fontweight="bold", linespacing=1.05)
        sw = _ax(fig, x - 0.56, yc + 0.095, 0.56, 0.04)  # colour ramp under the label
        sw.imshow(cmap(cd.HEAT_TOP * np.linspace(0, 1, 64))[None], aspect="auto", extent=(0, 1, 0, 1))
        sw.axis("off")


def _strip_label(fig, L: dict, r: dd.CaseRow, idx: int, n: int) -> None:
    """Gutter label of the RGB row: the query frame the whole strip belongs to (stays sharp while faded)."""
    x, yc = X_STRIP - 0.07, Y_RGB + H_RGB / 2
    _text(fig, x, yc - 0.055, L["strip_frame"].format(t=r.frame), ha="right", va="center", fontsize=FS["gutter"],
          fontweight="bold", color=style.INK)
    _text(fig, x, yc + 0.075, L["strip_query"].format(i=idx + 1, n=n), ha="right", va="center",
          fontsize=FS["gutter_sub"], color=style.INK_2)


def _strip_header(fig, CL: dict) -> None:
    q = W_STRIP / 4
    for v, name in enumerate(CL["views"]):
        _text(fig, X_STRIP + (v + 0.5) * q, Y_VIEWS, name, ha="center", va="center", fontsize=FS["views"],
              color=style.INK if v == 0 else style.INK_2, fontweight="bold" if v == 0 else "normal")
    # bracket over the three views the model never sees, the note set into its top line
    ax = _ax(fig, X_STRIP + q + 0.03, Y_BRACKET, 3 * q - 0.06, 0.07)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w_note = cd.text_width_pt(fig, CL["not_given"], FS["bracket"], fontstyle="italic") / 72.0 + 0.16
    frac = w_note / (3 * q - 0.06)
    ax.plot([0, 0, 0.5 - frac / 2], [0.0, 0.5, 0.5], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.plot([0.5 + frac / 2, 1, 1], [0.5, 0.5, 0.0], color=style.MUTED, lw=0.6, solid_joinstyle="miter")
    ax.text(0.5, 0.5, CL["not_given"], ha="center", va="center", fontsize=FS["bracket"], color=style.INK_2,
            fontstyle="italic")


def _gt_elev(dump: dd.Dump, r: dd.CaseRow) -> np.ndarray:
    """[8] elevation (deg, up-positive) of each ground-truth peak (NaN where not visible)."""
    tv = np.clip(r.gt_class - 1, 0, 3)
    yx = dump["gt_view_peak_yx"][r.index].astype(np.int64)[np.arange(dd.K), tv]
    _, el = geo.pixel_to_bearing_elev(tv, yx[:, 1], yx[:, 0])
    return np.where(r.visible, np.asarray(el, dtype=np.float64), np.nan)


def _strip(fig, r: dd.CaseRow, gt_el: np.ndarray, views: np.ndarray, CL: dict, rgb_ring_w: int) -> dict:
    """The surround strip of one query row: lane, RGB, ground truth, prediction (``fig_case.draw_block``).

    Peaks outside the rows' +-EL_HEAT band: a ground-truth one gets a blue caret at the row edge (its
    blob may be cut off); a predicted x is drawn at the edge with a small black caret.  Returns
    ``{"numbered": [...], "pred_none": [...]}`` (0-based slots) for checks.
    """
    ax_lane = _ax(fig, X_STRIP, Y_LANE, W_STRIP, H_LANE)
    ax_rgb = _ax(fig, X_STRIP, Y_RGB, W_STRIP, H_RGB)
    ax_gt = _ax(fig, X_STRIP, Y_GT, W_STRIP, H_HEAT)
    ax_pr = _ax(fig, X_STRIP, Y_PR, W_STRIP, H_HEAT)
    cd.draw_rgb_row(ax_rgb, cd.rgb_strip(views, rgb_ring_w, EL_RGB), EL_RGB)
    gt_strip = cd.heat_strip(dd.gt_composite(r), HEAT_RING_W, EL_HEAT)
    pr_strip = cd.heat_strip(dd.pred_composite(r, ARM), HEAT_RING_W, EL_HEAT)
    cd.draw_heat_row(ax_gt, gt_strip, EL_HEAT, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, pr_strip, EL_HEAT, cd.PRED_CMAP)
    caret = cd.pts_to_data(ax_gt, 0.0, 3.0)[1]
    for k in np.nonzero(r.visible & (np.abs(np.nan_to_num(gt_el)) > EL_HEAT - 1.0))[0]:
        up = gt_el[k] > 0
        ax_gt.plot([float(cd.strip_x(r.gt_bearing[k]))], [(EL_HEAT - caret) * (1 if up else -1)], ls="none",
                   marker="^" if up else "v", ms=4.6, mfc=GT_INK, mec="white", mew=0.5, zorder=6, clip_on=False)
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")

    # ground truth: badges in the lane, guide through the RGB row, ticks under the prediction
    pt_per_deg = W_STRIP * 72.0 / 360.0
    groups = r.groups
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in groups])
    labels = [dd.group_label(g) for g in groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / pt_per_deg for s in labels], 0.0, 360.0, 1.0 / pt_per_deg)
    y_badge = 0.52
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    for g, t, x, lab in zip(groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([x, x, t, t], [y_badge, 0.30, 0.10, 0.0], color=col, lw=0.6, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, x, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.6, zorder=3)
        ax_pr.plot([t, t], [-EL_HEAT - 1.0 * tick, -EL_HEAT - 4.2 * tick], color=col, lw=0.9, zorder=3,
                   clip_on=False, solid_capstyle="butt")
    lane_spans = [(x - cd.badge_width_pt(s) / pt_per_deg / 2, x + cd.badge_width_pt(s) / pt_per_deg / 2)
                  for x, s in zip(xs, labels)]

    # predicted peaks: PCK@8 misses (and false positives) alone and numbered, hits within MERGE_DEG share
    # one x, touching marks staggered vertically (fig_case's rule otherwise)
    p = r.arms[ARM]
    shown = [k for k in range(dd.K) if r.valid[k] and p.none_p[k] <= 0.5 and p.peak_view[k] >= 0]
    alone = numbered_slots(r, ARM)
    merged = [k for k in shown if k not in alone]
    px = {k: float(cd.strip_x(p.peak_bearing[k])) for k in shown}
    marks = []
    for cl in cd.cluster_1d([px[k] for k in merged], fc.MERGE_DEG):
        ks = [merged[i] for i in cl]
        marks.append((float(np.mean([px[k] for k in ks])), float(np.mean([p.peak_elev[k] for k in ks])), None))
    marks += [(px[k], float(p.peak_elev[k]), k) for k in alone]
    marks.sort(key=lambda m: m[0])
    per_pt_y = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    offsets, sign = [0.0] * len(marks), 1.0
    for j in range(1, len(marks)):
        if (marks[j][0] - marks[j - 1][0]) * pt_per_deg < fc.MARK_PT + 3.4:
            if offsets[j - 1] == 0.0:
                offsets[j - 1] = sign * fc.STAGGER_PT
            offsets[j] = -np.sign(offsets[j - 1]) * fc.STAGGER_PT
            sign = -sign
    edge = EL_HEAT - PEAK_EDGE_PT * per_pt_y  # an x never crosses the row's frame
    drawn = []
    for (x, el, k), off in zip(marks, offsets):
        y = float(np.clip(el + off * per_pt_y, -edge, edge))
        cd.peak_mark(ax_pr, x, y, size=fc.MARK_PT)
        if abs(el) > EL_HEAT:  # the peak lies beyond the row: a caret just inside the edge points to its side
            s = 1.0 if el > 0 else -1.0
            ax_pr.plot([x], [s * (EL_HEAT - 1.3 * per_pt_y)], marker="^" if s > 0 else "v", ms=2.8,
                       color=style.INK, mec="white", mew=0.4, zorder=7.5)
        drawn.append((x, y, k))
    texts = {k: CL["false_pos"] for k in alone if not r.visible[k]}
    _label_misses(ax_pr, fig, pr_strip, drawn, pt_per_deg, texts)

    # notes (badge + text) for slots without a ground-truth view or predicted "not visible"
    notes = []
    for k in r.invisible_slots():
        notes.append((k, CL["current" if r.gt_dist[k] < 0.1 else "not_visible"].format(p=p.none_p[k])))
    pred_none = [k for k in range(dd.K) if r.visible[k] and p.none_p[k] > 0.5]
    for k in pred_none:
        notes.append((k, CL["pred_none"].format(p=p.none_p[k])))
    free = [(1.0, min([a for a, _ in lane_spans], default=360.0) - 2.0),
            (max([b for _, b in lane_spans], default=0.0) + 2.0, 359.0)]
    for k, text in notes:
        bw = cd.badge_width_pt(str(k + 1)) / pt_per_deg
        width = bw + (1.4 + cd.text_width_pt(fig, text, FS["note"])) / pt_per_deg
        for j, (lo, hi) in enumerate(free):
            if hi - lo >= width:
                cd.history_badge(ax_lane, lo + bw / 2, y_badge, str(k + 1), k)
                ax_lane.text(lo + bw + 1.4 / pt_per_deg, y_badge, text, ha="left", va="center",
                             fontsize=FS["note"], color=style.INK_2)
                free[j] = (lo + width + 8.0 / pt_per_deg, hi)
                break
        else:
            print(f"[fig_anim] frame {r.frame}: no room for the note on slot {k + 1}: {text}")
    cd.azimuth_axis(ax_pr, CL["axis"], fs=FS["axis"])
    ax_pr.tick_params(axis="x", which="major", pad=6.0)
    return {"numbered": [int(k) for k in alone], "pred_none": [int(k) for k in pred_none]}


def _label_misses(ax, fig, strip: np.ndarray, marks, pt_per_deg: float, texts: Dict[int, str]) -> None:
    """Orange-ringed slot badge (+ text) for each numbered x: ``fig_case._place_peak_labels`` in two dimensions.

    ``marks``: (x deg, y deg, slot or None) of every drawn x.  Three label lanes (centre, upper, lower),
    all inside the row.  A label beside its x, or slid along a lane, never overlaps another x or label,
    and its leader (drawn when the label is not level with its x or slid sideways) never runs through
    another x or label.  A label without a visible leader must be clearly nearer its own x than any other
    x (``LABEL_CLEAR_PT``), so a crowded x gets a short leader instead of an ambiguous neighbour.  Among
    the rest: nearest to its x (lane distance counting 0.8 per point, a leader 2 pt, closeness to other
    x up to ``LABEL_CLEAR_PT`` 2 per point), then less heat underneath.
    """
    ppd = pt_per_deg  # the heat rows are in square degrees: the same points per degree on both axes
    half = fc.MARK_PT / 2 + 0.8
    bh = 3.9  # half height of a badge
    boxes = [(x * ppd - half, x * ppd + half, y * ppd - half, y * ppd + half) for x, y, _ in marks]
    own = {k: j for j, (_, _, k) in enumerate(marks) if k is not None}
    lane = EL_HEAT * ppd - bh - 0.9  # the badge's edge 0.9 pt inside the row's frame
    lanes = (0.0, lane, -lane)
    q = strip.shape[1] / 360.0
    n_marks = len(marks)

    def heat(a_pt, b_pt):
        a, b = a_pt / ppd, b_pt / ppd
        return float(strip[:, int(a * q):max(int(b * q), int(a * q) + 1)].sum())

    def gap(box, lo, hi, ylo, yhi):  # distance between two boxes (0 when they touch or overlap)
        b0, b1, c0, c1 = box
        return math.hypot(max(0.0, b0 - hi, lo - b1), max(0.0, c0 - yhi, ylo - c1))

    def overlap(box, lo, hi, ylo, yhi):
        b0, b1, c0, c1 = box
        return max(0.0, min(hi, b1) - max(lo, b0)) * max(0.0, min(yhi, c1) - max(ylo, c0))

    def crosses(p0, p1, skip):  # the leader p0 -> p1 runs through a box other than ``skip``
        ts = np.linspace(0.0, 1.0, 16)
        px, py = p0[0] + ts * (p1[0] - p0[0]), p0[1] + ts * (p1[1] - p0[1])
        return any(np.any((px > b0 + 0.3) & (px < b1 - 0.3) & (py > c0 + 0.3) & (py < c1 - 0.3))
                   for j, (b0, b1, c0, c1) in enumerate(boxes) if j != skip)

    for x, y, k in sorted([mk for mk in marks if mk[2] is not None], key=lambda mk: mk[0]):
        text = texts.get(k, "")
        bw = cd.badge_width_pt(str(k + 1))
        tw = cd.text_width_pt(fig, text, FS["small"]) + 1.6 if text else 0.0
        width = bw + tw
        xp, yp = x * ppd, y * ppd
        j_own = own[k]
        others = [j for j in range(n_marks) if j != j_own]
        best = None
        for ly in lanes:
            for shift in np.arange(0.0, 90.0, 1.5):
                for sgn in (1.0, -1.0):
                    a = xp + sgn * (half + 0.6 + shift)
                    lo, hi = (a, a + width) if sgn > 0 else (a - width, a)
                    if lo < 0 or hi > 360 * ppd:
                        continue
                    ylo, yhi = ly - bh, ly + bh
                    hard = sum(overlap(b, lo, hi, ylo, yhi) for j, b in enumerate(boxes) if j != j_own)
                    cx = (lo + bw / 2) if sgn > 0 else (hi - bw / 2)
                    p0, p1 = (xp + sgn * half * 0.7, yp), (cx - sgn * bw / 2, ly)
                    leader = (shift > 0 or abs(ly - yp) > 2.5) and math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 3.0
                    cross = leader and crosses(p0, p1, j_own)
                    near = min([gap(boxes[j], lo, hi, ylo, yhi) for j in others], default=99.0)
                    ambiguous = (not leader) and near < LABEL_CLEAR_PT
                    soft = shift + 0.8 * abs(ly - yp) + (2.0 if leader else 0.0) + 2.0 * max(0.0, LABEL_CLEAR_PT - near)
                    cost = (hard > 0.01, cross, ambiguous, hard, soft, heat(lo, hi))
                    if best is None or cost < best[0]:
                        best = (cost, sgn, lo, hi, cx, p0, p1, leader, ly)
        (_, sgn, lo, hi, cx, p0, p1, leader, ly) = best
        if leader:
            ax.plot([p0[0] / ppd, p1[0] / ppd], [p0[1] / ppd, p1[1] / ppd], color=PRED_INK, lw=0.6, zorder=6.5,
                    solid_capstyle="butt")
        pred_badge(ax, cx / ppd, ly / ppd, str(k + 1), zorder=8)
        if text:
            tx = cx + sgn * (bw / 2 + 1.6)
            ax.text(tx / ppd, ly / ppd, text, ha="left" if sgn > 0 else "right", va="center",
                    fontsize=FS["small"], color=style.INK, zorder=8, path_effects=cd.HALO)
        boxes.append((lo, hi, ly - bh, ly + bh))


def _legend(fig, y_top: float, h: float, L: dict, CL: dict) -> None:
    """Four entries stacked in the info column (the heat ramps are labelled beside their rows)."""
    ax = _ax(fig, X_INFO, y_top, W_INFO, h)
    w_pt, h_pt = W_INFO * 72.0, h * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis("off")
    n = 4
    step = h_pt / n
    ys = [h_pt - step * (j + 0.5) for j in range(n)]
    for j, k in enumerate((0, 4, 7)):
        cd.history_badge(ax, 3.7 + j * 8.6, ys[0], str(k + 1), k)
    cd.peak_mark(ax, 3.7, ys[1])
    pred_badge(ax, 3.7 + 9.4, ys[1], "3", zorder=8)
    ax.plot([3.7], [ys[2]], ls="none", marker="o", ms=3.4, mfc="white", mec=style.INK_2, mew=0.7)
    cd.robot_glyph(ax, 3.7, ys[3], (0.0, 1.0), size_pt=7.0)
    x_text = 3 * 8.6 + 2.5
    for y, text in zip(ys, (CL["legend_hist"], L["legend_peak"], L["legend_query"], CL["legend_robot"])):
        ax.text(x_text, y, text, ha="left", va="center", fontsize=FS["legend"], color=style.INK)


def _fmt_summary(s: dict, L: dict) -> Tuple[str, str]:
    if s["n"] == 0:
        return L["none_visible"], ""
    return L["err"].format(med=s["median"], mx=s["max"]), L["pck"].format(hits=s["hits"], n=s["n"])


def _bar_corner(ax) -> Tuple[float, float]:
    """Corner (sx, sy) where ``fig_case.draw_inset`` put the disc's scale bar (its label ends in ' m')."""
    for t in ax.texts:
        if t.get_text().endswith(" m") and hasattr(t, "xy"):
            x, y = t.xy
            return (1.0 if x > 0 else -1.0), (1.0 if y > 0 else -1.0)
    return 1.0, -1.0


# --------------------------------------------------------------------------- #
# One query row = one static scene + animated artists
# --------------------------------------------------------------------------- #
class _Scene:
    """Figure of one query row; ``render(t, mode)`` blits the moving parts over the static background.

    mode: ``hold`` (the row as is) or ``move`` (robot between this row and the next: live route and local
    map, the row's strip and numbers faded, the next query named on the timeline).
    """

    def __init__(self, plt, ctx: "_Context", idx: int, r: dd.CaseRow, views: np.ndarray, level):
        self.ctx, self.idx, self.r = ctx, idx, r
        L, CL, dump = ctx.L, ctx.CL, ctx.dump
        n = len(ctx.rows)
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=ctx.dpi)
        fig.patch.set_facecolor("white")
        self.fig = fig

        # title band
        _text(fig, MARGIN, Y_TITLE, L["title"], ha="left", va="center", fontsize=FS["title"], fontweight="bold",
              color=style.INK)
        _text(fig, FIG_W - MARGIN, Y_TITLE, L["sub"].format(tier=dump.tier_name(ctx.lang), scene=dump.scene,
                                                             ep=dump.episode_id, T=dump.frame_count),
              ha="right", va="center", fontsize=FS["sub"], color=style.MUTED)
        _text(fig, X_ROUTE, Y_NAMES, L["route"], ha="left", va="center", fontsize=FS["name"], color=style.INK)
        _text(fig, X_DISC, Y_NAMES, L["disc"], ha="left", va="center", fontsize=FS["name"], color=style.INK)

        # route (static part) + animated trail / past positions
        self.ax_route = _ax(fig, X_ROUTE, Y_TOP, W_ROUTE, H_TOP)
        _route_panel(self.ax_route, ctx.route_level, dump, CL)
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

        # local disc (fig_case.draw_inset) and strip
        self.ax_in = _ax(fig, X_DISC, Y_TOP, W_DISC, W_DISC)
        _strip_header(fig, CL)
        _row_labels(fig, L)
        _strip_label(fig, L, r, idx, n)
        self.drawn = _strip(fig, r, _gt_elev(dump, r), views, CL, ctx.rgb_ring_w)
        fc.draw_inset(self.ax_in, level, dump, r, show_arrow=True, L=CL)
        self.half = float(self.ax_in.get_xlim()[1]) * RADIUS_FRAC  # the disc radius draw_inset chose (m)
        self.bar_corner = _bar_corner(self.ax_in)
        # live disc while moving: an invisible twin of the inset axes whose artists are drawn one by one
        self.ax_live = _ax(fig, X_DISC, Y_TOP, W_DISC, W_DISC)
        self.ax_live.set_visible(False)

        # info column
        self.t_frame = _text(fig, X_INFO, Y_NAMES, "", ha="left", va="center", fontsize=FS["frame"],
                             fontweight="bold", color=style.INK, animated=True)
        self.t_query = _text(fig, X_INFO, Y_NAMES + 0.19, "", ha="left", va="center", fontsize=FS["muted"],
                             color=style.INK_2, animated=True)
        self._timeline(fig, Y_NAMES + 0.27)
        y = Y_NAMES + 0.75
        s = r.summary(ARM)
        a, b = _fmt_summary(s, L)
        _text(fig, X_INFO, y, a, ha="left", va="center", fontsize=FS["info"], color=style.INK)
        _text(fig, X_INFO, y + 0.15, b, ha="left", va="center", fontsize=FS["info"], color=style.INK)
        sf = r.summary("floor")
        if sf["n"]:
            _text(fig, X_INFO, y + 0.31, L["floor"].format(med=sf["median"], hits=sf["hits"], n=sf["n"]),
                  ha="left", va="center", fontsize=FS["muted"], color=style.MUTED)
        so = ctx.so_far[idx]
        if so["n"]:
            _text(fig, X_INFO, y + 0.45, L["so_far"].format(med=so["median"], hits=so["hits"], n=so["n"]),
                  ha="left", va="center", fontsize=FS["muted"], color=style.MUTED)
        metrics_box = (X_INFO - 0.03, y - 0.09, FIG_W - 0.02, y + 0.53)
        h_leg = 0.56
        _legend(fig, Y_TOP + H_TOP - h_leg, h_leg, L, CL)

        # veils (drawn only while moving): the strip (not its gutter labels) and the row's numbers; the disc is
        # covered by an opaque plate and redrawn live
        veil = dict(fc="white", ec="none", zorder=50, animated=True, alpha=FADE_ALPHA)
        self.veils = [
            _rect(fig, X_STRIP - 0.03, Y_LANE - 0.01, FIG_W - 0.02, Y_PR + H_HEAT + 0.07, **veil),
            _rect(fig, *metrics_box, **veil),
        ]
        self.disc_plate = _rect(fig, X_DISC - 0.08, Y_TOP - 0.03, X_DISC + W_DISC + 0.06, Y_TOP + H_TOP + 0.02,
                                fc="white", ec="none", zorder=50, animated=True)

        fig.canvas.draw()
        self.bg = fig.canvas.copy_from_bbox(fig.bbox)

    def _timeline(self, fig, y_top: float) -> None:
        ctx = self.ctx
        h = 0.34
        ax = _ax(fig, X_INFO, y_top, W_INFO - 0.08, h)
        last = max(ctx.track.last, 1)
        ax.set_xlim(-0.01 * last, 1.01 * last)
        ax.set_ylim(0, 1)
        ax.axis("off")
        y_line = 0.56
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
        self.tl_next_text = ax.text(0, 0.12, "", ha="center", va="center", fontsize=FS["next"], color=style.INK,
                                    animated=True)

    def _next_label(self, frame: int) -> None:
        ax, txt = self.ax_tl, self.tl_next_text
        last = max(self.ctx.track.last, 1)
        txt.set_text(self.ctx.L["next"].format(t=frame))
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

    def _live_disc(self, t: float, half: float, corner: Tuple[float, float]) -> None:
        """The local map at clip time ``t`` (no query here, so no rays or badges), radius ``half`` m."""
        ctx, ax = self.ctx, self.ax_live
        before = set(ax.get_children())
        p, f = ctx.track.at(t)
        level = ctx.level_at(t)
        cd.draw_local_disc(ax, level, p, f, half, past_xz=ctx.track.trail(t))
        cd.robot_glyph(ax, 0.0, 0.0, (0.0, 1.0), size_pt=DISC_ROBOT_PT, zorder=6)
        cd.disc_sector_letters(ax, half, [], names=ctx.CL["sectors"])
        cd.disc_direction_arrow(ax, half)
        per_pt = cd.pts_to_data(ax, 1.0)[0]
        lim = ax.get_xlim()[1]
        bar = cd.nice_length(0.6 * half)
        sx, sy = corner
        cd.scale_bar(ax, sx * (lim - 1.0 * per_pt), sy * (lim - (8.5 if sy > 0 else 2.5) * per_pt), bar,
                     f"{bar:g} m", fs=5.6, ha="left" if sx < 0 else "right")
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
            self._live_disc(t, half, nxt.bar_corner)
            self._next_label(nxt.r.frame)
        # info
        ti = int(round(t))
        key = "frame_last" if ti == ctx.track.last else "frame"
        self.t_frame.set_text(ctx.L[key].format(t=ti, T=ctx.dump.frame_count))
        fig.draw_artist(self.t_frame)
        if mode == "move":
            self.t_query.set_text(ctx.L["faded"].format(i=self.idx + 1, t=self.r.frame))
        else:
            self.t_query.set_text(ctx.L["query"].format(i=self.idx + 1, n=len(ctx.query_frames))
                                  + (ctx.L["schedule"] if ctx.schedule == "full" else ""))
        fig.draw_artist(self.t_query)
        self.tl_marker.set_data([t], [0.92])
        self.ax_tl.draw_artist(self.tl_marker)
        buf = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(buf[..., :3])

    def close(self, plt) -> None:
        plt.close(self.fig)


class _Context:
    def __init__(self, dump: dd.Dump, rows: List[dd.CaseRow], lang: str, dpi: float, schedule: str, scene_td,
                 cam_h: float):
        self.dump, self.rows, self.lang, self.dpi = dump, rows, lang, dpi
        self.schedule = schedule  # "full" (every endpoint row), "gaps" (those with an output) or "subset"
        self.L = LABELS[lang]
        self.CL = fc.LABELS[lang]
        self.track = _Track(dump)
        self.scene_td, self.cam_h = scene_td, cam_h
        self.route_level = scene_td.pick_level(float(np.median(dump.positions[:, 1])) - cam_h)
        self.query_frames = [r.frame for r in rows]
        w = int(round(W_STRIP * dpi / 8.0)) * 8  # ring columns: ~1 per device pixel, multiple of 8 (exact roll)
        self.rgb_ring_w = max(w, 360)
        # running totals over the rows shown so far (visible slots only)
        self.so_far = []
        errs, hits, n = [], 0, 0
        for r in rows:
            v = r.visible
            e = r.arms[ARM].err[v]
            errs.extend([float(x) for x in e if np.isfinite(x)])
            hits += int((r.arms[ARM].joint8 & v).sum())
            n += int(v.sum())
            self.so_far.append({"median": float(np.median(errs)) if errs else float("nan"), "hits": hits, "n": n})

    def level_at(self, t: float):
        return self.scene_td.pick_level(self.track.height(t) - self.cam_h)


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
    """UI colours every GIF palette keeps exactly (badges, heat ramps, ink): pixel counts cannot merge them away."""
    from matplotlib.colors import to_rgb

    cols = ["white", style.INK, style.INK_2, style.MUTED, style.AXIS, style.GRID, style.SURFACE, ROUTE_AHEAD,
            GT_INK, PRED_INK, cd.MAP_PLATE]
    cols += [style.history_color(k) for k in range(dd.K)] + [cd.history_line_color(k) for k in range(dd.K)]
    cols += [cd.GT_CMAP(cd.HEAT_TOP * i / 15) for i in range(16)] + [cd.PRED_CMAP(cd.HEAT_TOP * i / 15)
                                                                     for i in range(16)]
    out = []
    for c in cols:
        rgb = tuple(int(round(255 * v)) for v in to_rgb(c))
        if rgb not in out:
            out.append(rgb)
    return out


class _Gif:
    """Looping GIF: frames downscaled to ``width``; one palette per segment (reserved UI colours +
    adaptive colours of the segment's first frame), no dithering so unchanged pixels stay unchanged.

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

    def add(self, frame: np.ndarray, start_s: float, new_palette: bool) -> None:
        from PIL import Image

        h = int(round(frame.shape[0] * self.width / frame.shape[1] / 2)) * 2
        im = Image.fromarray(frame).resize((self.width, h), Image.Resampling.LANCZOS)
        if new_palette or self._palette is None:
            self._palette = self._make_palette(im)
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
# Entry point
# --------------------------------------------------------------------------- #
def make_animation(dump_npz, out_stem="anim", topdown_root=None, clip_root_override=None, fps: int = 15,
                   lang: str = "en", size: Tuple[int, int] = (1920, 1080), hold_s: float = 2.0,
                   end_hold_s: float = 3.0, substeps: int = 2, max_move_s: float = 3.0,
                   rows: Optional[Sequence[int]] = None, gif: bool = True, gif_width: int = 1280,
                   gif_step: int = 1, stills: bool = False, crf: int = 18, encoder: str = "auto") -> dict:
    """Render the supplementary animation of one dump.

    Returns ``{"files": [...], "rows": [...], "frames": n, "duration_s": s, "size": (w, h), "fps": fps,
    "encoder": str, "render_s": s, "stats": [...]}``; each stats entry has the row's numbers
    (``vo`` / ``floor`` summaries), the disc radius ``half_m`` and the slots drawn as ``numbered`` x
    and ``pred_none`` notes.  ``rows`` restricts the query rows (default: every scored row of the
    deployed model's output); ``size`` must be 16:9 with even sides.
    """
    t_start = time.time()
    w_px, h_px = int(size[0]), int(size[1])
    if w_px * 9 != h_px * 16 or w_px % 2 or h_px % 2:
        raise ValueError(f"size {size}: need 16:9 with even sides (e.g. 1920x1080, 1280x720)")
    if lang not in LABELS:
        raise ValueError(f"lang {lang!r}: one of {sorted(LABELS)}")
    fps, substeps = int(fps), max(1, int(substeps))
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
    ctx = _Context(dump, recs, lang, dpi, schedule, scene_td, cam_h)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = _Mp4(out.parent / (out.name + ".mp4"), w_px, h_px, fps, crf=crf, encoder=encoder)
    gw = _Gif(out.parent / (out.name + ".gif"), gif_width) if gif else None
    files: List[str] = []
    stats: List[dict] = []

    def build(i: int) -> _Scene:
        r = recs[i]
        views = dd.surround_views(clip_dir, r.frame)
        level = scene_td.pick_level(float(r.cur_pos[1]) - cam_h)
        return _Scene(plt, ctx, i, r, views, level)

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
                gw.add(img, mp4.frames / fps, new_palette=last_gif_t is None)
                last_gif_t = t
            mp4.write(img)

    scene = build(0)
    for i, r in enumerate(recs):
        last = i == len(recs) - 1
        img = scene.render(float(r.frame), "hold")
        if gw is not None:
            gw.add(img, mp4.frames / fps, new_palette=True)
        mp4.write(img, int(round((end_hold_s if last else hold_s) * fps)))
        if stills:
            from PIL import Image

            p = out.parent / f"{out.name}_q{i:02d}_f{r.frame:03d}.png"
            Image.fromarray(img).save(p)
            files.append(str(p))
        stats.append({"row": r.index, "frame": r.frame, ARM: r.summary(ARM), "floor": r.summary("floor"),
                      "half_m": round(scene.half, 3), **scene.drawn})
        nxt = None
        if not last:
            nxt = build(i + 1)
            move(scene, nxt)
        scene.close(plt)
        scene = nxt

    files.insert(0, str(mp4.close()))
    if gw is not None:
        files.insert(1, str(gw.close(mp4.frames / fps)))
    caption = CAPTION[lang].format(tier=dump.tier_name(lang), scene=dump.scene, ep=dump.episode_id, n=len(recs),
                                   sched=CAPTION_SCHEDULE[lang][ctx.schedule], el=EL_HEAT)
    cap = out.parent / (out.name + "_caption.txt")
    cap.write_text(caption + "\n", encoding="utf-8")
    files.insert(2 if gw is not None else 1, str(cap))
    return {"files": files, "rows": [r.index for r in recs], "frames": mp4.frames,
            "duration_s": mp4.frames / fps, "size": (w_px, h_px), "fps": fps, "encoder": mp4.backend,
            "render_s": time.time() - t_start, "stats": stats}


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
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, larger)")
    ap.add_argument("--encoder", default="auto", choices=("auto", "pyav", "ffmpeg"))
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    res = make_animation(args.dump, out_stem=args.out, topdown_root=args.topdown_root,
                         clip_root_override=args.clip_root, fps=args.fps, lang=args.lang, size=(w, h),
                         hold_s=args.hold, end_hold_s=args.end_hold, substeps=args.substeps, rows=rows,
                         gif=not args.no_gif, gif_width=args.gif_width, gif_step=args.gif_step, stills=args.stills,
                         crf=args.crf, encoder=args.encoder)
    for f in res["files"]:
        print(f)
    for s in res["stats"]:
        print(s)
    print(f"frames {res['frames']}  duration {res['duration_s']:.1f} s  {res['size'][0]}x{res['size'][1]} "
          f"@ {res['fps']} fps  encoder {res['encoder']}  render {res['render_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
