#!/usr/bin/env python3
"""EXP-19 paper figures: IEEE double column (7.16 in wide), height 0.80 x width, from the same records as ``fig_v2``.

Figure A -- key moments.  One row per case (default T1, T2, T3)::

  (a) Multi-room, multi-turn  Success, 0.2 m
  route map | 1 | 2 | 3 | 4

  all key moments of the episode in time order, numbered 1-4 with black circles (the same circles on the route
  map).  Each moment, top to bottom: the image System 2 decided on (pixel goal ring, System 1 path dots, its number
  in the top-left corner, the executed action chunk as chips in a corner clear of the pixel goal, ``chips_corner``),
  the 360° history affordance map strip (past frames 1, 4, 8 marked; a grey line joins a frame's predicted peak to
  its true direction) and the 360° future affordance map strip.  The strip rows are named once (rotated, left of
  the first row), the bearings once (under the first column of the last row).  The column width is solved so the
  page is ``ASPECT`` x its width tall.

Figure B -- the affordance maps online.  One row per case (default T1, T3, F1)::

  (a) Multi-room, multi-turn  Success, 0.2 m  “instruction ...”
  route map | online timeline: history panel, future panel, executed turns

  the step axis starts at the first call with an affordance map (``timeline_panel.crop_warmup``): the steps before
  it are left out, not compressed; each column marks past frames 1, 4, 8 as one group (``Timeline.compact148``);
  the key moments carry Figure A's circled numbers on their hairlines.  The panel heights are solved for ``ASPECT``.

Key moments are numbered in time order (``time_numbers``), not by the rule that chose them (the bundle's K1-K4), so
an episode in both figures carries the same numbers at the same steps; the captions name the rule behind each
number from the branch each key moment took (``moment_rules``).

Outputs per figure and language: PDF (TrueType text), SVG (live ``<text>``) and PNG (400 dpi), plus a caption
``.txt``; every text is real text (the route maps' white halos become small white boxes, ``editable_text``), so the
PDF / SVG can be fine-tuned in Illustrator or Inkscape once the fonts are installed (Nimbus Sans; zh: Droid Sans
Fallback).  Images and affordance maps are embedded rasters.  The layout numbers (inches) are the ``FIG_A`` /
``FIG_B`` constants below.

Policy as ``fig_v2``: no poses, VO or odometry; both heatmaps are "affordance map"; nothing is drawn from the future
map to the actions; the captions describe what is drawn and make no claim (the RTX 4090 batch is descriptive only,
see the ledger's EXP-19 run record 5).

Usage (repo root on PYTHONPATH)::

  python -m scripts.exp19.figures.fig_paper --records <EXP>/records --timelines <EXP>/records_v2 \\
      --topdown-root <EXP>/topdown --out-dir <EXP>/figures_paper [--lang en zh] \\
      [--fig-a T1 T2 T3] [--fig-b T1 T3 F1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import fig_behavior as fb
from scripts.exp19.figures import fig_v2 as f2
from scripts.exp19.figures import panels_v2 as p2
from scripts.exp19.figures import timeline_panel as tp

import matplotlib  # noqa: E402  (configured in setup())
from matplotlib.patches import BoxStyle, Circle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

MANIFEST_SCHEMA = "exp19-figures-paper-manifest-v3"
PAPER_W = 7.16  # IEEE double column
EDGE = 0.01  # the right-most frame lines stay this far inside the page
ASPECT = 0.80  # height / width the layouts solve for ...
ASPECT_RANGE = (0.77, 0.83)  # ... and outside this range is a warning
FS = p2.FS
FS_TITLE, FS_BODY = 7.0, 6.3  # a row's title; its outcome
LINE = 0.105  # in per 6-6.3 pt text line (wrapped heads)
LEGEND_LINE = 0.13  # in per legend line: its goal glyph and circled number are taller than a text line
LEADER_CLEAR_PT = 3.5  # a route map's leader keeps 2 pt off the rim of another key-moment dot (``draw_route``)
DEFAULT_A = ("T1", "T2", "T3")
DEFAULT_B = ("T1", "T3", "F1")
LETTERS = "abcdefghij"

FIG_A = {"head_h": 0.16, "row_gap": 0.08, "route_gap": 0.05, "gutter": 0.11, "k_gap": 0.05, "dec_gap": 0.025,
         "strip_gap": 0.02, "fut_elev": 36.0, "ticks_h": 0.12, "legend_gap": 0.09, "k_w_max": 1.45,
         "route_w_min": 1.3, "chip_pt": 6.4, "chip_inset_pt": 2.4}
FIG_B = {"top_pad": 0.02, "route_w": 1.53, "name_w": 0.20, "ylab": 0.28, "head_gap": 0.03, "badges": 0.13,
         "gap": 0.055, "turn_gap": f2.TURN_GAP, "turn": f2.TURN_H, "ticks": 0.125, "row_gap": 0.11,
         "legend_gap": 0.08, "hist_frac": 0.59, "panels_min": 0.5, "panels_max": 1.6}

LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "letter": "({l})",
        "success": "Success, {ne} m",
        "failure": "Failure, {ne} m",
        "outcome_missing": "outcome not recorded",
        "names": {"F1": "reached the goal area, stopped elsewhere"},  # else fb.CATEGORY_NAMES
        "rows": ("History", "Future"),
        "legend": {
            "route": "route", "ref": "reference path", "offlevel": "other floor", "start": "start",
            "goal": "goal ({r:g} m)", "stop": "stop", "num": "key moment", "pgoal": "pixel goal",
            "path": "System 1 path", "actions": "executed actions", "hist": "history affordance map",
            "past": "past-frame direction, true / predicted", "past_b": "past-frame direction, true / predicted",
            "fut": "future affordance map", "path_end": "System 1 path end", "turns": "executed turns (up = left)",
            "nomap": "no affordance map",
        },
    },
    "zh": {
        "letter": "({l})",
        "success": "成功，{ne} m",
        "failure": "失败，{ne} m",
        "outcome_missing": "结局未记录",
        "names": {},
        "rows": ("历史", "未来"),
        "legend": {
            "route": "路线", "ref": "参考路径", "offlevel": "另一楼层", "start": "起点", "goal": "目标（{r:g} m）",
            "stop": "停止处", "num": "关键时刻", "pgoal": "像素目标", "path": "快系统路径", "actions": "执行的动作",
            "hist": "历史 affordance map", "past": "历史帧方向：真实 / 预测", "past_b": "历史帧方向：真实 / 预测",
            "fut": "未来 affordance map", "path_end": "快系统路径终点", "turns": "执行的转向（上 = 左）",
            "nomap": "无 affordance map",
        },
    },
}

# How each numbered key moment was chosen, by the branch it took (``bd.KEY_BRANCHES``; fb.KEY_RULES in full).  The
# head names the pool (System 2 calls with an affordance map = keysteps' ready calls), so "one" is always one of them:
# the K2 fallback's "no later one" does not count the grey no-map calls, whose turns keysteps never looks at.
RULES = {
    "en": {"K1_first": "the first",
           "K2_turn": "the later one with the largest executed turn",
           "K2_fallback": "the one a third of the way through (no later one's executed turn reaches 30°)",
           "K3_two_thirds": "the one nearest two thirds of the episode",
           "K3_f1_closest": "the last one up to the closest approach to the goal",
           "K3_f1_fallback_after": "the first one after the closest approach to the goal",
           "K4_last": "the last",
           "K4_shifted": "the latest one not chosen before",
           "head": "Key moments {r} are chosen among the System 2 calls with an affordance map: {rules}",
           "all": "The numbered key moments are all the System 2 calls with an affordance map",
           "sep": ", ", "last": " and ", "case": "; in {l}, {parts}", "part": "{n} is {rule}",
           "part_next": "{n} {rule}", "lt4": "; {l} has only {n}, all shown", "lt4_one": "; {l} has only one",
           "end": "."},
    "zh": {"K1_first": "第一次",
           "K2_turn": "其后执行转角最大的一次",
           "K2_fallback": "位于三分之一处的一次（其后没有一次执行转角达到 30°）",
           "K3_two_thirds": "最接近本集 2/3 处的一次",
           "K3_f1_closest": "最接近目标那一步及之前的最后一次",
           "K3_f1_fallback_after": "最接近目标那一步之后的第一次",
           "K4_last": "最后一次",
           "K4_shifted": "此前未选中的最晚一次",
           "head": "关键时刻 {r} 从给出 affordance map 的慢系统调用中选取：{rules}",
           "all": "编号的关键时刻即全部给出 affordance map 的慢系统调用",
           "sep": "、", "last": "和", "case": "；{l} 中{parts}", "part": " {n} 为{rule}", "part_next": "，{n} 为{rule}",
           "lt4": "；{l} 只有 {n} 次，全部画出", "lt4_one": "；{l} 只有 1 次", "end": "。"},
}

# Five sentences each.  A: the rows; the decision image; the two strips (with the display smoothing); the past-frame
# marks; how the moments were chosen (``moment_rules``).  B: the rows; the axes; a column (with the smoothing); the
# past-frame marks; the moments.
CAPTION_A = {
    "en": ("Key moments of closed-loop R2R val-unseen episodes; headings give the outcome and the final distance to "
           "the goal. Each moment shows the image on which System 2 placed its pixel goal (ring), with the System 1 "
           "path (dots, also on the future map) and the executed actions (chips). Below it are the 360° history "
           "(orange; camera's 79° view boxed) and future (green, darker = later) affordance maps, smoothed for display "
           "(σ = {s:g}°). Blue circles and orange dots are the true and predicted directions of past frames 1 (oldest), "
           "4 and 8 of the eight given to System 2{edge}. {rules}"),
    "zh": ("闭环复跑（R2R val-unseen）的关键时刻；标题为结局与停止处到目标的距离。每个关键时刻上方为慢系统据以给出像素目标"
           "（圈）的图像，叠有快系统路径（点，未来图上同）与执行的动作（方块）。下方为 360° 历史（橙，黑框为相机 79° 视野）"
           "与未来（绿，越深越晚）affordance map，为显示做了平滑（σ = {s:g}°）。蓝圈与橙点为送入慢系统的 8 个历史帧中"
           "第 1（最早）、4、8 帧的真实与预测方向{edge}。{rules}"),
}
CAPTION_B = {
    "en": ("Affordance maps predicted online in closed-loop R2R val-unseen episodes; headings give the outcome, the "
           "final distance to the goal and the instruction. The x-axis is the step, starting at the first System 2 "
           "call with an affordance map, and the y-axis the bearing around the robot (up = left){grey}. Each column is "
           "one call, with its history (orange) and future (green, darker = later) affordance maps, smoothed for "
           "display (σ = {s:g}°); black dots mark the end of its System 1 path. Blue circles and orange dots are the "
           "true and predicted directions of past frames 1 (oldest), 4 and 8 of the eight given to System 2. {rules}"
           "{stacked}"),
    "zh": ("闭环复跑（R2R val-unseen）中在线预测的 affordance map；标题为结局、停止处到目标的距离与指令。横轴为步数，从第一次"
           "给出 affordance map 的慢系统调用开始；纵轴为机器人周围的方位，向上 = 向左{grey}。每列为一次调用的历史（橙）与未来"
           "（绿，越深越晚）affordance map，为显示做了平滑（σ = {s:g}°）；黑点为快系统路径终点。蓝圈与橙点为送入慢系统的 "
           "8 个历史帧中第 1（最早）、4、8 帧的真实与预测方向。{rules}{stacked}"),
}
CAPTION_GREY = {"en": "; grey columns are calls without an affordance map", "zh": "；灰列为没有 affordance map 的调用"}
CAPTION_EDGE = {"en": "; a triangle on a strip's edge marks a direction beyond its elevation range",
                "zh": "；条带边缘的小三角表示超出其俯仰范围的方向"}



# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
def setup(lang: str) -> None:
    """``fig_v2.setup`` plus real text in every vector output (TrueType in the PDF, live text in the SVG)."""
    f2.setup(lang)
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def paper_labels(lang: str) -> dict:
    return {**f2.LABELS[lang], **LABELS[lang]}


def heading_weight(lang: str) -> str:
    """Bold headings in English; the only CJK face (Droid Sans Fallback) has one weight, so a zh heading in bold
    would mix a bold "(a)" with regular CJK: zh headings are regular, a little larger (``title_style``)."""
    return "bold" if lang == "en" else "normal"


def title_style(lang: str) -> dict:
    return {"fontsize": FS_TITLE if lang == "en" else FS_TITLE + 0.5, "fontweight": heading_weight(lang),
            "color": style.INK}


def category_of(b: bd.Bundle) -> str:
    return b.membership(main=True)["category"]


def case_title(b: bd.Bundle, letter: str, L: dict, lang: str) -> str:
    """"(a) Multi-room, multi-turn": the letter and the category's name (``LABELS[...]["names"]`` first)."""
    cat = category_of(b)
    name = L["names"].get(cat, fb.CATEGORY_NAMES[lang][cat])
    return f"{L['letter'].format(l=letter)} {name[:1].upper() + name[1:] if lang == 'en' else name}"


def outcome_short(b: bd.Bundle, L: dict) -> str:
    """"Success, 0.2 m": the outcome and the final distance to the goal."""
    o = b.outcome
    if o is None:
        return L["outcome_missing"]
    ne = f"{o['ne_m']:.1f}" if fb._finite(o["ne_m"]) else "?"
    return L["success" if o["success"] else "failure"].format(ne=ne)


def time_order(keys: Sequence[bd.KeyStep]) -> List[bd.KeyStep]:
    return sorted(keys, key=lambda k: (int(k.step), int(k.index)))


def time_numbers(b: bd.Bundle) -> Dict[str, str]:
    """Key label (K1-K4) -> its number in time order ("1"-"4"); the same in both figures."""
    return {k.label: str(i + 1) for i, k in enumerate(time_order(b.keys))}


BADGE_PAD = 0.14  # circle padding of a key-moment number (x font size)


def circle_badges(ax, labels) -> int:
    """The black rounded badges the shared helpers draw (``cd.key_badge``: route map, decision image, timeline)
    turned into circles, for the texts in ``labels``.  Returns how many were changed."""
    n = 0
    for t in ax.texts:
        if t.get_text() in labels and t.get_bbox_patch() is not None:
            t.get_bbox_patch().set_boxstyle("circle", pad=BADGE_PAD)
            n += 1
    return n


def num_badge(ax, x: float, y: float, text: str, fs: float = p2.MIN_FS):
    """A circled key-moment number, as ``circle_badges`` makes them (for the legend)."""
    t = cd.key_badge(ax, x, y, text, fs=fs, zorder=11)
    t.get_bbox_patch().set_boxstyle("circle", pad=BADGE_PAD)
    return t


def row_name(page: fb.Page, x: float, y: float, text: str, lang: str) -> None:
    """The name of a strip row or timeline panel, centred on (x, y) in inches: rotated in English, upright characters
    stacked in Chinese."""
    kw = dict(ha="center", va="center", fontsize=p2.MIN_FS, color=style.INK_2)
    if lang == "zh":
        page.text(x, y, "\n".join(text), linespacing=1.05, **kw)
    else:
        page.text(x, y, text, rotation=90, **kw)


def off_level_visible(b: bd.Bundle, topdown) -> bool:
    """The route map shows dotted steps on another floor (at least ``fb.FLOOR_NOTE_MIN_M`` of them; shorter ones
    hide under the start circle): the legend's "other floor" entry."""
    off = fb.off_level_steps(b, topdown)
    xz = b.xz("route_xz")
    if off is None or not np.any(off) or len(xz) < 2:
        return False
    seg = np.asarray(off, dtype=bool)
    seg = seg[1:] | seg[:-1]
    return bool(np.linalg.norm(np.diff(xz, axis=0), axis=1)[seg].sum() >= fb.FLOOR_NOTE_MIN_M)


def draw_route(page: fb.Page, x: float, y: float, w: float, h: float, b: bd.Bundle, topdown,
               nums: Dict[str, str], L: dict) -> dict:
    """Route map (``p2.draw_route_map``) with the key moments as circled numbers, no start / radius text (both are in
    the legend); a leader keeps ``LEADER_CLEAR_PT`` off the other key-moment dots.  Returns its info ({"stop_drawn",
    "start_label", "badges"})."""
    ax = page.ax(x, y, w, h)
    keys = time_order(b.keys)
    labels = [nums[k.label] for k in keys]
    info = p2.draw_route_map(ax, topdown[1] if topdown is not None else None, b.xz("route_xz"),
                             b.xz("reference_path_xz"), b.start_xz, b.goal_xz, float(b.goal_radius_m),
                             [k.position_xz for k in keys], labels, "", "", off_level=fb.off_level_steps(b, topdown),
                             leader_clear_pt=LEADER_CLEAR_PT)
    info["badges"] = circle_badges(ax, set(labels))
    if topdown is None:
        ax.text(0.5, 0.03, L["no_map"], transform=ax.transAxes, ha="center", va="bottom", fontsize=p2.MIN_FS,
                color=style.INK_2, fontstyle="italic")
    return info


# --------------------------------------------------------------------------- #
# Legend: centred lines of glyph + short text, groups kept together
# --------------------------------------------------------------------------- #
GLYPH_W = {"route": 12.0, "ref": 12.0, "offlevel": 12.0, "start": 5.0, "goal": 9.6, "stop": 4.0, "num": 8.6,
           "pgoal": 7.0, "path": 11.0, "actions": 13.8, "hist": 12.0, "past": 15.0, "past_b": 10.0, "fut": 12.0,
           "path_end": 3.2, "turns": 14.0, "nomap": 9.0}
ITEM_GAP, GROUP_GAP, GLYPH_GAP = 7.0, 15.0, 2.6  # legend spacing (pt)


def glyph(ax, key: str, x: float, y: float) -> None:
    """A legend glyph from x (its left edge), centred on y, point units (``GLYPH_W[key]`` wide)."""
    if key == "route":
        ax.plot([x, x + 12.0], [y, y], color=style.INK, lw=0.95, solid_capstyle="round")
    elif key == "ref":
        ax.plot([x, x + 12.0], [y, y], color=style.MUTED, lw=0.8, ls=(0, (3.0, 1.8)))
    elif key == "offlevel":
        ax.plot([x, x + 12.0], [y, y], color=style.MUTED, lw=0.9, ls=(0, (1.0, 1.4)), dash_capstyle="round")
    elif key == "start":
        ax.plot([x + 2.5], [y], marker="o", ms=4.0, mfc="white", mec=style.INK, mew=0.9)
    elif key == "goal":
        ax.add_patch(Circle((x + 4.8, y), 4.3, fc=cd.mix(style.INK, "white", 0.94), ec=style.INK_2, lw=0.5,
                            ls=(0, (2.2, 1.6))))
        ax.plot([x + 4.8], [y], marker="*", ms=6.0, mfc=style.INK, mec="white", mew=0.4)
    elif key == "stop":
        ax.plot([x + 2.0], [y], marker="s", ms=p2.STOP_MS, mfc=style.INK, mec="white", mew=0.45)
    elif key == "num":
        num_badge(ax, x + 4.3, y, "1")
    elif key == "pgoal":
        p2.goal_ring(ax, x + 3.5, y)
    elif key == "path":
        p2.path_marks(ax, [x + 1.0 + 3.2 * i for i in range(4)], [y] * 4, clip_on=False)
    elif key == "actions":
        p2.action_chip(ax, x, y, bd.FORWARD, size=6.4)
        p2.action_chip(ax, x + 7.4, y, bd.LEFT, size=6.4)
    elif key == "past":  # the strips: true direction, grey line, predicted peak
        ax.plot([x + 2.2, x + 12.8], [y, y], color=p2.PAIR_COLOR, lw=p2.PAIR_LW, solid_capstyle="butt")
        p2.gt_ring(ax, x + 2.2, y, clip_on=False)
        p2.pred_dot(ax, x + 12.8, y, clip_on=False)
    elif key == "past_b":  # the timeline: true direction, predicted peak
        p2.gt_ring(ax, x + 2.2, y, clip_on=False)
        p2.pred_dot(ax, x + 8.0, y, clip_on=False)
    elif key == "path_end":
        p2.path_marks(ax, [x + 1.6], [y], ms=1.9 * tp.TL_MARK, rim=True, clip_on=False)
    else:  # hist, fut, turns, nomap
        p2.legend_glyph(ax, key, x, y, w=GLYPH_W[key])


def line_width(line: Sequence[Sequence[Tuple[str, str, float]]]) -> float:
    return (sum(w for grp in line for _, _, w in grp) + sum(ITEM_GAP * (len(grp) - 1) for grp in line)
            + GROUP_GAP * (len(line) - 1))


def legend_lines(fig, groups: Sequence[Sequence[Tuple[str, str]]], width_pt: float,
                 fs: float = p2.MIN_FS) -> List[List[List[Tuple[str, str, float]]]]:
    """``groups`` of (key, text) flowed into lines ``width_pt`` wide: a group joins the current line when it fits,
    else starts the next one; a group wider than a line is split between its items.  Returns lines of groups of
    (key, text, width in pt)."""
    lines: List[list] = [[]]
    for grp in groups:
        items = [(k, t, GLYPH_W[k] + GLYPH_GAP + cd.text_width_pt(fig, t, fs)) for k, t in grp]
        if lines[-1] and line_width(lines[-1] + [items]) > width_pt:
            lines.append([])
        part: list = []
        for it in items:
            if part and line_width(lines[-1] + [part + [it]]) > width_pt:
                lines[-1].append(part)
                lines.append([])
                part = []
            part.append(it)
        lines[-1].append(part)
    return [ln for ln in lines if ln]


def legend_height(lines) -> float:
    return LEGEND_LINE * len(lines) + 0.02


def draw_legend(page: fb.Page, y: float, lines, fs: float = p2.MIN_FS) -> float:
    """The legend lines, each centred on the page; returns the height (in)."""
    h = legend_height(lines)
    width_pt = (PAPER_W - EDGE) * 72.0
    ax = page.pt_axes(0.0, y, PAPER_W - EDGE, h, zorder=3)
    for i, line in enumerate(lines):
        x = (width_pt - line_width(line)) / 2.0
        yy = h * 72.0 - (i + 0.5) * LEGEND_LINE * 72.0 - 0.5
        for g, grp in enumerate(line):
            if g:
                x += GROUP_GAP
            for j, (key, text, w) in enumerate(grp):
                x += ITEM_GAP if j else 0.0
                glyph(ax, key, x, yy)
                ax.text(x + GLYPH_W[key] + GLYPH_GAP, yy, text, ha="left", va="center", fontsize=fs, color=style.INK)
                x += w
    return h


def route_group(L: dict, stop: bool, offlevel: bool, radius: float) -> List[Tuple[str, str]]:
    T = L["legend"]
    out = [("route", T["route"]), ("ref", T["ref"])] + ([("offlevel", T["offlevel"])] if offlevel else [])
    out += [("start", T["start"]), ("goal", T["goal"].format(r=radius))] + ([("stop", T["stop"])] if stop else [])
    return out + [("num", T["num"])]


def legend_groups_a(L: dict, stop: bool, offlevel: bool, radius: float) -> List[List[Tuple[str, str]]]:
    T = L["legend"]
    return [route_group(L, stop, offlevel, radius), [(k, T[k]) for k in ("pgoal", "path", "actions")],
            [(k, T[k]) for k in ("hist", "past", "fut")]]


def legend_groups_b(L: dict, stop: bool, offlevel: bool, radius: float, nomap: bool) -> List[List[Tuple[str, str]]]:
    T = L["legend"]
    return [route_group(L, stop, offlevel, radius), [(k, T[k]) for k in (("turns", "nomap") if nomap else ("turns",))],
            [(k, T[k]) for k in ("hist", "past_b", "fut", "path_end")]]


# --------------------------------------------------------------------------- #
# Checks, captions, output
# --------------------------------------------------------------------------- #
def editable_text(fig) -> int:
    """Texts drawn with a stroke halo (route-map labels, scale bar) come out as outlines in the PDF / SVG; give them a
    small white box instead, so every text on the page stays text.  Returns how many were changed."""
    n = 0
    for t in fig.findobj(Text):
        if t.get_path_effects() and t.get_text().strip():
            t.set_path_effects([])
            if t.get_bbox_patch() is None:
                t.set_bbox(dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8))
            n += 1
    return n


def badge_circles(fig) -> List[Tuple[str, float, float, float]]:
    """(text, x, y, radius) in pixels of every circled number on the page."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(Text):
        patch = t.get_bbox_patch()
        if t.get_visible() and patch is not None and isinstance(patch.get_boxstyle(), BoxStyle.Circle):
            bb = patch.get_window_extent(renderer)
            out.append((t.get_text(), (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, min(bb.width, bb.height) / 2))
    return out


def badge_overlaps(fig, tol_px: float = 0.5) -> List[str]:
    """Pairs of circled numbers whose circles touch (a text-box test would miss two round badges)."""
    c = badge_circles(fig)
    return [f"{a[0]!r} x {b[0]!r}" for i, a in enumerate(c) for b in c[i + 1:]
            if np.hypot(a[1] - b[1], a[2] - b[2]) < a[3] + b[3] - tol_px]


def audit(fig, height: float, name: str) -> dict:
    """Layout checks: texts outside the page, overlapping texts or circled numbers, leaders crossing texts, smallest
    font, aspect."""
    outside, overlaps, crossings = f2.texts_outside(fig), f2.text_overlaps(fig), f2.leader_crossings(fig)
    overlaps += badge_overlaps(fig)
    min_fs = f2.min_font_size(fig)
    aspect = height / PAPER_W
    warnings = []
    if outside:
        warnings.append(f"{name}: texts outside the page: {outside}")
    if overlaps:
        warnings.append(f"{name}: overlapping texts: {overlaps}")
    if crossings:
        warnings.append(f"{name}: leaders crossing texts: {crossings}")
    if min_fs < p2.MIN_FS - 1e-6:
        warnings.append(f"{name}: smallest text {min_fs:.2f} pt < {p2.MIN_FS} pt")
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        warnings.append(f"{name}: height / width {aspect:.3f} outside {ASPECT_RANGE}")
    return {"outside": len(outside), "overlaps": len(overlaps), "leader_crossings": len(crossings),
            "min_font_pt": round(min_fs, 2), "aspect": round(aspect, 3), "warnings": warnings}


def moment_rules(cases: Sequence[Tuple[str, Sequence]], lang: str) -> str:
    """One sentence on how the numbered key moments were chosen, true for every row: per number, the rule (branch)
    most rows took, then each row that took another rule at some number (named with its own rules); a row with
    fewer than four calls with an affordance map (branch "all_lt4": all of them are drawn) is named as such, and when
    every row is one, the sentence says just that.  ``cases``: (panel letter, key moments in time order) per row.
    "" when nothing is drawn."""
    R = RULES[lang]
    full = [(f"({letter})", [k.branch for k in keys]) for letter, keys in cases if keys]
    if not full:
        return ""
    regular = [br for _, br in full if "all_lt4" not in br]
    if not regular:
        return R["all"] + R["end"]
    n = max(len(br) for br in regular)

    def rule(branch: str) -> str:
        return R.get(branch, fb.KEY_RULES[lang].get(branch, branch))

    def listed(items: List[str]) -> str:  # "A, B and C" (zh: "A、B和C")
        return items[0] if len(items) == 1 else R["sep"].join(items[:-1]) + R["last"] + items[-1]
    majority = []
    for i in range(n):
        seen = [br[i] for br in regular if len(br) > i]
        majority.append(max(dict.fromkeys(seen), key=seen.count))  # most rows; the earliest row's on a tie
    out = R["head"].format(r=f"1–{n}", rules=listed([rule(b_) for b_ in majority]))
    for letter, br in full:
        if "all_lt4" in br:
            out += R["lt4_one" if len(br) == 1 else "lt4"].format(l=letter, n=len(br))
            continue
        diff = [(i, b_) for i, b_ in enumerate(br) if b_ != majority[i]]
        if diff:  # en "2 is A and 3 B"; zh " 2 为A，3 为B"
            bits = [(R["part"] if j == 0 else R["part_next"]).format(n=i + 1, rule=rule(b_))
                    for j, (i, b_) in enumerate(diff)]
            out += R["case"].format(l=letter, parts=listed(bits) if lang == "en" else "".join(bits))
    return out + R["end"]


def join_caption(parts: Sequence[str], lang: str) -> str:
    """Caption sentences joined (a space between English sentences, none between Chinese ones)."""
    text = (" " if lang == "en" else "").join(p.strip() for p in parts if p and p.strip())
    return " ".join(text.split()) if lang == "en" else text


def save(fig, stem: Path, lang: str, caption: str) -> List[str]:
    """PDF (TrueType text), SVG (live text), PNG (400 dpi) and the caption."""
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    files = [stem.parent / f"{stem.name}_{lang}.{ext}" for ext in ("pdf", "svg", "png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=300, bbox_inches=None)
    fig.savefig(files[2], dpi=400, bbox_inches=None)
    plt.close(fig)
    cap = stem.parent / f"{stem.name}_caption_{lang}.txt"
    cap.write_text(caption + "\n", encoding="utf-8")
    return [str(f) for f in files + [cap]]


# --------------------------------------------------------------------------- #
# Figure A: route + key moments
# --------------------------------------------------------------------------- #
def moment_rows(k_w: float, g: dict = FIG_A) -> Dict[str, float]:
    """Top offset (in) of every part of a key-moment column ``k_w`` wide: decision image, history strip, future
    strip; "end" = its height."""
    y, out = 0.0, {}
    for name, h in (("dec", 0.75 * k_w), ("gap1", g["dec_gap"]), ("hist", p2.strip_height(k_w, p2.HIST_ELEV)),
                    ("gap2", g["strip_gap"]), ("fut", p2.strip_height(k_w, g["fut_elev"]))):
        out[name] = y
        y += h
    out["end"] = y
    return out


def solve_a(n_rows: int, n_cols: int, legend_h: float, g: dict = FIG_A) -> Tuple[float, float, float]:
    """(column width, route-map width, page height): the column width that makes the page ``ASPECT`` x its width
    tall, capped at ``k_w_max`` and where the route map would be narrower than ``route_w_min``."""
    fixed = n_rows * g["head_h"] + (n_rows - 1) * g["row_gap"] + g["ticks_h"] + g["legend_gap"] + legend_h
    c0 = moment_rows(0.0, g)["end"]
    c1 = moment_rows(1.0, g)["end"] - c0  # the column height is linear in its width
    room = PAPER_W - EDGE - g["route_gap"] - g["gutter"] - (n_cols - 1) * g["k_gap"]
    k_w = ((ASPECT * PAPER_W - fixed) / n_rows - c0) / c1
    k_w = max(min(k_w, g["k_w_max"], (room - g["route_w_min"]) / n_cols), 0.3)
    return k_w, room - n_cols * k_w, fixed + n_rows * moment_rows(k_w, g)["end"]


def drawn_image_marks(ks: bd.KeyStep) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """((u, v) of the System 1 waypoints ``p2.draw_decision_image`` draws inside the image, the pixel goal's (u, v)
    when it is drawn, else None)."""
    img = ks.decision_rgb
    h, w = img.shape[:2]
    uv = np.asarray(ks.path_uv, dtype=np.float64).reshape(-1, 2)
    if len(uv):
        uv = uv[p2.subsample_path(len(uv))]
        uv = uv[(uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)]
    goal = ks.pixel_goal_uv
    goal = np.asarray(goal, dtype=np.float64).reshape(2) if goal is not None and bd.goal_inside(goal, img.shape) \
        else None
    return uv, goal


CHIP_CORNERS = ("right", "left", "top")  # bottom right, bottom left, top right (top left: the number), preferred first
CHIP_AIR_PT = 3.0  # a System 1 waypoint this close to the chips counts as under them
GOAL_R_PT = p2.GOAL_MS / 2 + 0.85  # the pixel goal ring's outer radius with its white halo (``p2.goal_ring``)


def chips_box(ks: bd.KeyStep, w: float, width_pt: float, corner: str, air_pt: float = 0.0,
              g: dict = FIG_A) -> Tuple[float, float, float, float]:
    """(u0, v0, u1, v1) in image pixels of the chips drawn in ``corner`` of a decision image ``w`` in wide (their
    inset included), grown by ``air_pt``."""
    h_img, w_img = ks.decision_rgb.shape[:2]
    per = w_img / (w * 72.0)  # image pixels per point
    bw, bh = (width_pt + g["chip_inset_pt"] + air_pt) * per, (g["chip_pt"] + g["chip_inset_pt"] + air_pt) * per
    u0, u1 = (0.0, bw) if corner == "left" else (w_img - bw, float(w_img))
    v0, v1 = (0.0, bh) if corner == "top" else (h_img - bh, float(h_img))
    return u0 - 0.5, v0 - 0.5, u1 - 0.5, v1 - 0.5


def _inside(pts: np.ndarray, box) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])


def chips_corner(ks: bd.KeyStep, w: float, width_pt: float, g: dict = FIG_A) -> str:
    """The corner of the decision image for the executed-action chips: never one whose chips would touch the pixel
    goal's ring (``GOAL_R_PT`` + 1 pt of air); among the others the one with the fewest System 1 waypoints under the
    chips (``CHIP_AIR_PT`` of air), bottom right, bottom left, top right in that order on a tie, so top right only
    when both bottom corners would cover something.  (Chips narrower than half the image: a ring touches one corner's
    chips at most, so a corner free of it always exists.)"""
    path, goal = drawn_image_marks(ks)

    def cost(corner):
        ring = chips_box(ks, w, width_pt, corner, GOAL_R_PT + 1.0, g)
        covers_goal = goal is not None and bool(_inside(goal, ring)[0])
        return covers_goal, int(np.sum(_inside(path, chips_box(ks, w, width_pt, corner, CHIP_AIR_PT, g)))), \
            CHIP_CORNERS.index(corner)
    return min(CHIP_CORNERS, key=cost)


def chips_width(ks: bd.KeyStep, g: dict = FIG_A) -> float:
    acts = [int(a) for a in ks.executed_actions]
    return sum(p2.chip_width(a, g["chip_pt"]) for a in acts) + 1.5 * max(len(acts) - 1, 0)


def draw_chips(page: fb.Page, x: float, y: float, w: float, ks: bd.KeyStep, g: dict = FIG_A) -> str:
    """The executed action chunk as chips in a corner of the decision image (``chips_corner``); returns it."""
    size, inset, width = g["chip_pt"], g["chip_inset_pt"], chips_width(ks, g)
    corner = chips_corner(ks, w, width, g)
    kax = page.pt_axes(x, y, w, w * 0.75, zorder=4)
    x0 = inset if corner == "left" else w * 72.0 - inset - width
    yc = w * 0.75 * 72.0 - inset - size / 2 if corner == "top" else inset + size / 2
    p2.action_chips(kax, x0, yc, [int(a) for a in ks.executed_actions], size=size)
    return corner


def draw_moment(page: fb.Page, x: float, y: float, k_w: float, ks: bd.KeyStep, num: str, L: dict,
                ticks: bool, g: dict = FIG_A) -> dict:
    """One key moment: decision image (its number top left, executed actions in a free corner), 360° history strip
    (past frames 1, 4, 8 marked), 360° future strip; ``ticks``: the bearing labels under the future strip."""
    rows = moment_rows(k_w, g)
    dax = page.ax(x, y + rows["dec"], k_w, 0.75 * k_w)
    p2.draw_decision_image(dax, ks, badge=num)
    circle_badges(dax, {num})
    corner = draw_chips(page, x, y + rows["dec"], k_w, ks, g)
    hax = page.ax(x, y + rows["hist"], k_w, p2.strip_height(k_w, p2.HIST_ELEV))
    info = p2.draw_history_strip(hax, ks, f2._ring_px(k_w, 340), f2._ring_px(k_w, 272), slots=tp.OVERVIEW_SLOTS)
    fax = page.ax(x, y + rows["fut"], k_w, p2.strip_height(k_w, g["fut_elev"]))
    p2.draw_future_strip(fax, ks, f2._ring_px(k_w, 272), elev=g["fut_elev"])
    if ticks:
        p2.strip_ticks(fax, L["axis"])
    _, goal = drawn_image_marks(ks)
    box = chips_box(ks, k_w, chips_width(ks, g), corner, GOAL_R_PT, g)  # the ring touching the chips counts
    return {**info, "chips_corner": corner, "chips_cover_goal": goal is not None and bool(_inside(goal, box)[0])}


def draw_head_a(page: fb.Page, y: float, b: bd.Bundle, letter: str, lang: str, L: dict, g: dict = FIG_A) -> None:
    """"(a) Multi-room, multi-turn  Success, 0.2 m" on one line."""
    ax = page.pt_axes(0.0, y, PAPER_W - EDGE, g["head_h"])
    yy = g["head_h"] * 72.0 * 0.45
    title, st = case_title(b, letter, L, lang), title_style(lang)
    ax.text(0.0, yy, title, ha="left", va="center", **st)
    xo = cd.text_width_pt(page.fig, title, st["fontsize"], fontweight=st["fontweight"]) + 6.0
    ax.text(xo, yy, outcome_short(b, L), ha="left", va="center", fontsize=FS_BODY, color=style.INK_2)


def make_fig_a(bundles: Sequence[bd.Bundle], out_dir: Path, lang: str, topdown_root,
               name: str = "fig_a_key_moments") -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = paper_labels(lang)
    g = FIG_A
    n = len(bundles)
    fig = plt.figure(figsize=(PAPER_W, 6.0))
    moments = [time_order(b.keys) for b in bundles]
    nums = [time_numbers(b) for b in bundles]
    n_cols = max([len(m) for m in moments] + [1])
    topdowns = [fb.resolve_level(b, topdown_root) for b in bundles]
    radius = float(bundles[0].goal_radius_m) if bundles else 3.0
    offlevel = any(off_level_visible(b, td) for b, td in zip(bundles, topdowns))
    legend_h = legend_height([0, 0])  # the legend's height sets the maps' size, whose stop squares set the legend
    for _ in range(2):
        k_w, route_w, height = solve_a(n, n_cols, legend_h)
        route_h = moment_rows(k_w)["end"]
        stops = [p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                               route_w, route_h) for b in bundles]
        lines = legend_lines(fig, legend_groups_a(L, any(stops), offlevel, radius), (PAPER_W - EDGE) * 72.0)
        legend_h = legend_height(lines)
    k_w, route_w, height = solve_a(n, n_cols, legend_h)
    route_h = moment_rows(k_w)["end"]
    fig.set_size_inches(PAPER_W, height)
    page = fb.Page(fig, PAPER_W, height)
    x_cols = route_w + g["route_gap"] + g["gutter"]
    rows = moment_rows(k_w)
    y = 0.0
    checks, edge, pairs, corners, cover, stop_drawn = [], 0, 0, [], 0, []
    for i, (b, ms, nm, td) in enumerate(zip(bundles, moments, nums, topdowns)):
        draw_head_a(page, y, b, LETTERS[i], lang, L)
        yb = y + g["head_h"]
        rinfo = draw_route(page, 0.0, yb, route_w, route_h, b, td, nm, L)
        stop_drawn.append(rinfo["stop_drawn"])
        for c, ks in enumerate(ms):
            info = draw_moment(page, x_cols + c * (k_w + g["k_gap"]), yb, k_w, ks, nm[ks.label], L,
                               ticks=(i == n - 1 and c == 0))
            edge += info["edge"]
            pairs += info["pairs"]
            corners.append(info["chips_corner"])
            cover += int(info["chips_cover_goal"])
        if i == 0:  # the strip rows named once, in the gutter left of the first column
            gx = route_w + g["route_gap"] + g["gutter"] * 0.42
            for key, text in zip(("hist", "fut"), L["rows"]):
                h_strip = p2.strip_height(k_w, p2.HIST_ELEV if key == "hist" else g["fut_elev"])
                row_name(page, gx, yb + rows[key] + h_strip / 2, text, lang)
        checks.append({"ep_key": b.ep_key, "category": category_of(b),
                       "moments": [(k.label, nm[k.label], int(k.step), k.branch) for k in ms],
                       "stop_drawn": rinfo["stop_drawn"], "route_badges": rinfo["badges"]})
        y = yb + route_h + g["row_gap"]
    y += g["ticks_h"] - g["row_gap"] + g["legend_gap"]
    draw_legend(page, y, lines)
    n_halo = editable_text(fig)
    rules = moment_rules([(LETTERS[i], ms) for i, ms in enumerate(moments)], lang)
    caption = join_caption([CAPTION_A[lang].format(rules=rules, edge=CAPTION_EDGE[lang] if edge else "",
                                                   s=p2.SMOOTH_DEG)], lang)
    res = audit(fig, height, f"{name} [{lang}]")
    if stop_drawn != stops:
        res["warnings"].append(f"{name} [{lang}]: end-of-rerun squares drawn {stop_drawn} but planned {stops}")
    if cover:
        res["warnings"].append(f"{name} [{lang}]: executed-action chips cover {cover} pixel goals")
    files = save(fig, Path(out_dir) / name, lang, caption)
    return {"files": files, "size_in": (PAPER_W, round(height, 3)), **res, "checks": checks,
            "k_w_in": round(k_w, 3), "route_in": (round(route_w, 3), round(route_h, 3)), "edge_marks": edge,
            "strip_pair_lines": pairs, "chips_corners": corners, "chips_cover_goal": cover,
            "legend_lines": len(lines), "halo_texts_boxed": n_halo, "caption_chars": len(caption)}


# --------------------------------------------------------------------------- #
# Figure B: route + online timeline (the steps before the first affordance map left out)
# --------------------------------------------------------------------------- #
RUN_GAP_PT = 4.0  # extra air between two style runs of a heading (title, outcome, instruction)


def head_parts(b: bd.Bundle, letter: str, lang: str, L: dict) -> List[Tuple[str, dict]]:
    """A heading's style runs: title, outcome, instruction in quotes (italic)."""
    return [(case_title(b, letter, L, lang), title_style(lang)),
            (outcome_short(b, L), {"fontsize": FS_BODY, "color": style.INK_2}),
            (L["instruction"].format(text=" ".join(b.instruction.split())),
             {"fontsize": p2.MIN_FS, "fontstyle": "italic", "color": style.INK_2})]


def _font_kw(st: dict) -> dict:
    return {k: v for k, v in st.items() if k in ("fontweight", "fontstyle")}


def head_lines(fig, parts: Sequence[Tuple[str, dict]], width_pt: float) -> List[List[Tuple[int, str]]]:
    """A run-in heading greedy-wrapped on measured word widths (an italic word's ink overhang counted): lines of
    (style run, text) pieces."""
    space = {i: cd.text_width_pt(fig, "a a", st["fontsize"], **_font_kw(st))
             - cd.text_width_pt(fig, "aa", st["fontsize"], **_font_kw(st)) for i, (_, st) in enumerate(parts)}
    lines: List[List[Tuple[int, str]]] = [[]]
    cur = 0.0
    for i, (text, st) in enumerate(parts):
        for word in text.split():
            ww = cd.text_width_pt(fig, word, st["fontsize"], **_font_kw(st))
            if st.get("fontstyle") == "italic":
                ww = max(ww, f2.ink_extent_pt(word, st["fontsize"], fontstyle="italic")[1])
            sep = (space[i] + (RUN_GAP_PT if lines[-1][-1][0] != i else 0.0)) if lines[-1] else 0.0
            if lines[-1] and cur + sep + ww > width_pt - f2.INK_MARGIN_PT:
                lines.append([])
                cur, sep = 0.0, 0.0
            if lines[-1] and lines[-1][-1][0] == i:
                lines[-1][-1] = (i, lines[-1][-1][1] + " " + word)
            else:
                lines[-1].append((i, word))
            cur += sep + ww
    return lines


def draw_head_b(page: fb.Page, y: float, parts: Sequence[Tuple[str, dict]], lines) -> None:
    fig = page.fig
    for j, line in enumerate(lines):
        x = 0.0
        for r, (i, text) in enumerate(line):
            st = parts[i][1]
            if r:
                x += cd.text_width_pt(fig, "a a", st["fontsize"], **_font_kw(st)) - \
                    cd.text_width_pt(fig, "aa", st["fontsize"], **_font_kw(st)) + RUN_GAP_PT
            page.text(x / 72.0, y + LINE * (j + 0.5), text, ha="left", va="center", **st)
            x += cd.text_width_pt(fig, text, st["fontsize"], **_font_kw(st))


def number_keys(tl: tp.Timeline, nums: Dict[str, str]) -> List[Tuple[str, str, int]]:
    """Rename the timeline's key moments to their numbers in time order (``time_numbers``; in memory only), so the
    shared badge row draws them.  Returns (label, number, step) per key moment on the timeline, in time order."""
    rows = tl.key_rows()
    out = sorted(((str(lab), nums.get(str(lab), str(lab)), int(tl.a["step"][r])) for lab, r in rows.items()),
                 key=lambda t: t[2])
    tl.a = {**tl.a, "key_labels": np.asarray([nums.get(str(lab), str(lab)) for lab in tl.a["key_labels"]])}
    return out


def visible_nomap(tl: tp.Timeline) -> bool:
    """A no-map span (System 2 answered with turns or STOP) inside the drawn axis."""
    lo, hi = tl.xlim()
    return any(kind == "nomap" and s1 > lo and s0 < hi for kind, s0, s1 in tl.spans())


def cropped_nomap_steps(tl: tp.Timeline) -> int:
    """Steps of no-map calls (after the warm-up) left out by the crop."""
    w, x0 = float(tl.warmup_end()), float(tl.x0)
    return int(round(sum(max(0.0, min(s1, x0) - max(s0, w)) for kind, s0, s1 in tl.spans() if kind == "nomap")))


def draw_timeline(page: fb.Page, x: float, y: float, w: float, hh: float, fh: float, badges_h: float,
                  tl: tp.Timeline, L: dict, g: dict = FIG_B) -> dict:
    """Numbered badges on the key hairlines, history panel, future panel, executed-turn track with the step axis."""
    bax = page.pt_axes(x, y, w, badges_h, zorder=6)
    hax = page.ax(x, y + badges_h, w, hh)
    fax = page.ax(x, y + badges_h + hh + g["gap"], w, fh)
    tax = page.ax(x, y + badges_h + hh + g["gap"] + fh + g["turn_gap"], w, g["turn"])
    info = tp.draw_history_panel(hax, tl, L["y_hist"], slots=tp.OVERVIEW_SLOTS)
    plan = tp.mark_plan(tl, hax, tp.OVERVIEW_SLOTS)
    info.update(tp.draw_future_panel(fax, tl, L["y_fut"], plan=plan))
    info["turns_drawn"] = tp.draw_turn_track(tax, tl, L["turns"])
    tp.step_axis(tax, tl, L["x_steps"])
    items = tp.key_badges(bax, hax, tl)
    circle_badges(bax, {it["label"] for it in items})
    info["badges"] = [(it["label"], round(it["x"], 1), it["level"]) for it in items]
    return info


def stacked_sentence(checks: Sequence[dict], lang: str) -> str:
    """A long rerun's timeline stacks all eight past frames' marks mid-column, for one call in k (``tp.mark_plan``),
    so "past frames 1, 4 and 8" is not what its row shows: the caption says which rows (``fig_v2``'s sentence);
    "" when every row marks frames 1, 4, 8 on every call."""
    rows = [i for i, c in enumerate(checks) if c.get("mode") == "stacked"]
    if not rows:
        return ""
    k = max(int(checks[i].get("marker_stride") or 1) for i in rows)
    return (" " if lang == "en" else "") + f2.stride_sentence(lang, k, cases=[f"({LETTERS[i]})" for i in rows])


def panel_heights(fixed: float, n: int, g: dict = FIG_B) -> Tuple[float, float]:
    """(history, future) panel heights (in) that make the page ``ASPECT`` x its width tall, within
    [``panels_min``, ``panels_max``] per row."""
    both = min(max((ASPECT * PAPER_W - fixed) / max(n, 1), g["panels_min"]), g["panels_max"])
    return both * g["hist_frac"], both * (1.0 - g["hist_frac"])


def make_fig_b(bundles: Sequence[bd.Bundle], timelines: Sequence[tp.Timeline], out_dir: Path, lang: str,
               topdown_root, name: str = "fig_b_online_timeline") -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = paper_labels(lang)
    g = FIG_B
    n = len(bundles)
    fig = plt.figure(figsize=(PAPER_W, 6.0))
    tl_x = g["route_w"] + g["name_w"] + g["ylab"]
    tl_w = PAPER_W - tl_x - EDGE
    nums = [time_numbers(b) for b in bundles]
    moments = []
    for tl, nm in zip(timelines, nums):
        tp.crop_warmup(tl)
        tl.compact148 = True
        moments.append(number_keys(tl, nm))
    badges_h = max([tp.badges_height_in(tp.badge_levels(fig, tl, tl_w * 72.0), g["badges"]) for tl in timelines]
                   or [g["badges"]])
    parts = [head_parts(b, LETTERS[i], lang, L) for i, b in enumerate(bundles)]
    heads = [head_lines(fig, p, (PAPER_W - EDGE) * 72.0) for p in parts]
    head_h = [LINE * len(h) + g["head_gap"] for h in heads]
    topdowns = [fb.resolve_level(b, topdown_root) for b in bundles]
    radius = float(bundles[0].goal_radius_m) if bundles else 3.0
    offlevel = any(off_level_visible(b, td) for b, td in zip(bundles, topdowns))
    grey = any(visible_nomap(tl) for tl in timelines)
    fixed_row = badges_h + g["gap"] + g["turn_gap"] + g["turn"] + g["ticks"]
    fixed0 = g["top_pad"] + sum(head_h) + n * fixed_row + (n - 1) * g["row_gap"] + g["legend_gap"]
    legend_h = legend_height([0, 0])  # two passes, as figure A
    for _ in range(2):
        hh, fh = panel_heights(fixed0 + legend_h, n)
        route_h = fixed_row - g["ticks"] + hh + fh
        stops = [p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                               g["route_w"], route_h) for b in bundles]
        lines = legend_lines(fig, legend_groups_b(L, any(stops), offlevel, radius, grey), (PAPER_W - EDGE) * 72.0)
        legend_h = legend_height(lines)
    fixed = fixed0 + legend_h
    hh, fh = panel_heights(fixed, n)
    route_h = fixed_row - g["ticks"] + hh + fh
    height = fixed + n * (hh + fh)
    fig.set_size_inches(PAPER_W, height)
    page = fb.Page(fig, PAPER_W, height)
    y = g["top_pad"]
    checks, stop_drawn = [], []
    for i, (b, tl, nm, td) in enumerate(zip(bundles, timelines, nums, topdowns)):
        draw_head_b(page, y, parts[i], heads[i])
        yb = y + head_h[i]
        info = draw_timeline(page, tl_x, yb, tl_w, hh, fh, badges_h, tl, L)
        rinfo = draw_route(page, 0.0, yb, g["route_w"], route_h, b, td, nm, L)
        stop_drawn.append(rinfo["stop_drawn"])
        if i == 0:  # the panels named once, 3 pt left of the widest tick label, clear of the route map
            ticks_pt = tp.TICK_LEN + tp.TICK_PAD + max(cd.text_width_pt(fig, t, p2.MIN_FS) for t in
                                                       list(L["y_hist"]) + list(L["y_fut"]) + [L["turns"]])
            name_pt = max(cd.text_width_pt(fig, c, p2.MIN_FS) for c in "".join(L["rows"])) if lang == "zh" \
                else p2.MIN_FS  # the stacked characters' width / the rotated line's height
            gx = max(tl_x - (ticks_pt + 3.0 + name_pt / 2) / 72.0, g["route_w"] + g["name_w"] * 0.46)
            for yc, text in zip((yb + badges_h + hh / 2, yb + badges_h + hh + g["gap"] + fh / 2), L["rows"]):
                row_name(page, gx, yc, text, lang)
        checks.append({"ep_key": b.ep_key, "category": category_of(b), "x0_step": float(tl.x0),
                       "first_ready_step": int(np.min(tl.a["step"])) if tl.R else None,
                       "warmup_end": int(tl.warmup_end()), "cropped_nomap_steps": cropped_nomap_steps(tl),
                       "moments": [(lab, num, st, next((k.branch for k in b.keys if k.label == lab), None))
                                   for lab, num, st in moments[i]],
                       "stop_drawn": rinfo["stop_drawn"], "route_badges": rinfo["badges"],
                       **{k: v for k, v in info.items() if k in ("mode", "n_pred", "n_gt", "path_ends", "turns_drawn",
                                                                  "badges", "compressed_warmup", "marker_stride")}})
        y = yb + route_h + g["ticks"] + g["row_gap"]
    y += g["legend_gap"] - g["row_gap"]
    draw_legend(page, y, lines)
    n_halo = editable_text(fig)
    rules = moment_rules([(LETTERS[i], time_order(b.keys)) for i, b in enumerate(bundles)], lang)
    caption = join_caption([CAPTION_B[lang].format(grey=CAPTION_GREY[lang] if grey else "", rules=rules,
                                                   s=tp.TL_SMOOTH_DEG, stacked=stacked_sentence(checks, lang))], lang)
    res = audit(fig, height, f"{name} [{lang}]")
    if stop_drawn != stops:
        res["warnings"].append(f"{name} [{lang}]: end-of-rerun squares drawn {stop_drawn} but planned {stops}")
    for b, ms in zip(bundles, moments):  # the timeline's numbered key moments are the bundle's, in time order
        want = [(k.label, time_numbers(b)[k.label], int(k.step)) for k in time_order(b.keys)]
        if list(ms) != want:
            res["warnings"].append(f"{name} [{lang}]: {b.ep_key} timeline key moments {ms} != bundle {want}")
    files = save(fig, Path(out_dir) / name, lang, caption)
    return {"files": files, "size_in": (PAPER_W, round(height, 3)), **res, "checks": checks,
            "panels_in": (round(hh, 3), round(fh, 3)), "timeline_w_in": round(tl_w, 3), "legend_lines": len(lines),
            "halo_texts_boxed": n_halo, "caption_chars": len(caption)}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
IGNORED_BUNDLE_WARNINGS = ("fidelity sentence omitted",)  # about fig_v2's page captions, not these figures


def bundle_warnings(b: bd.Bundle) -> List[str]:
    return [w for w in b.warnings if not any(s in w for s in IGNORED_BUNDLE_WARNINGS)]


def pick(bundles: Sequence[bd.Bundle], wanted: Sequence[str]) -> Tuple[List[bd.Bundle], List[str]]:
    """Bundles in the order asked: a category code (T1 ... F2) means that category's one main case, else an ep_key.
    Returns (bundles, warnings: an ep_key whose rerun no longer meets its category)."""
    out, warnings = [], []
    for w in wanted:
        hit = [b for b in bundles if b.ep_key == w]
        if not hit:
            hit = [b for b in bundles if b.membership(main=True).get("is_main")
                   and b.membership(main=True)["category"] == w]
            if len(hit) != 1:
                raise SystemExit(f"{w!r}: {len(hit)} main cases (want exactly one), and no ep_key of that name")
        b = hit[0]
        if b.membership(main=True).get("predicate_holds_on_rerun") is False:
            warnings.append(f"{b.ep_key}: the rerun no longer meets the {category_of(b)} definition")
        out.append(b)
    return out, warnings


def code_hashes() -> Dict[str, Optional[str]]:
    here = Path(__file__).parent
    exp18 = here.parents[1] / "exp18" / "figures"
    files = {n: here / n for n in ("fig_paper.py", "fig_v2.py", "panels_v2.py", "panels.py", "timeline_panel.py",
                                   "bundle.py", "fig_behavior.py")}
    files.update({f"exp18/{n}": exp18 / n for n in ("common_draw.py", "style.py")})
    return {n: f2._sha256(p) for n, p in files.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True, help="records dir: <ep_key>_bundle.{json,npz} and <ep_key>.json")
    ap.add_argument("--timelines", required=True, help="records_v2 dir: <ep_key>_timeline.{json,npz}")
    ap.add_argument("--topdown-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=sorted(LABELS))
    ap.add_argument("--fig-a", nargs="*", default=list(DEFAULT_A), help="rows of figure A (categories or ep_keys)")
    ap.add_argument("--fig-b", nargs="*", default=list(DEFAULT_B), help="rows of figure B (categories or ep_keys)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.resolve().name in ("figures", "figures_v2"):
        print("refusing to write into the v1 / v2 figures dir", file=sys.stderr)
        return 2
    bundles = [bd.load_bundle(p) for p in bd.find_bundles(args.records)]
    jobs = []
    if args.fig_a:
        jobs.append(("fig_a", *pick(bundles, args.fig_a)))
    if args.fig_b:
        jobs.append(("fig_b", *pick(bundles, args.fig_b)))
    manifest_path = out_dir / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    figures = {f["figure"]: f for f in old.get("figures", []) if old.get("schema") == MANIFEST_SCHEMA}
    for kind, members, pick_warnings in jobs:
        sources = []
        for b in members:
            tl_path = tp.timeline_path_for(args.timelines, b.ep_key)
            rec = Path(args.records) / f"{b.ep_key}.json"
            sources.append({"ep_key": b.ep_key, "category": category_of(b),
                            "sha256": {"bundle_json": f2._sha256(b.path),
                                       "bundle_npz": f2._sha256(bd.bundle_paths(b.path)[1]),
                                       "timeline_json": f2._sha256(tp.timeline_paths(tl_path)[0]),
                                       "timeline_npz": f2._sha256(tp.timeline_paths(tl_path)[1]),
                                       "record": f2._sha256(rec)}})
        entry = figures.get(kind) if figures.get(kind, {}).get("cases") == sources else None
        entry = entry or {"figure": kind, "cases": sources, "files": {}, "size_in": {}, "checks": {}, "warnings": {}}
        entry["args"] = {"rows": [s["ep_key"] for s in sources]}
        for lang in args.lang:
            warnings = list(pick_warnings)
            if kind == "fig_a":
                res = make_fig_a(members, out_dir, lang, args.topdown_root)
                warnings += [f"{b.ep_key}: {w}" for b in members for w in bundle_warnings(b)]
            else:
                tls = [tp.load_timeline(tp.timeline_path_for(args.timelines, b.ep_key),
                                        record_path=Path(args.records) / f"{b.ep_key}.json") for b in members]
                warnings += [f"{b.ep_key}: {w}" for b, tl in zip(members, tls)
                             for w in bundle_warnings(b) + tl.warnings + tp.check_against_bundle(tl, b)]
                res = make_fig_b(members, tls, out_dir, lang, args.topdown_root)
            entry["files"][lang] = {Path(f).name: f2._sha256(f) for f in res["files"]}
            entry["size_in"][lang] = res["size_in"]
            entry["checks"][lang] = {k: v for k, v in res.items() if k not in ("files", "size_in", "warnings")}
            entry["warnings"][lang] = warnings + res["warnings"]
            print(f"{kind} [{lang}] {res['size_in']} aspect {res['aspect']} min {res['min_font_pt']} pt "
                  f"-> {res['files'][2]}")
            for w in entry["warnings"][lang]:
                print(f"  WARNING {w}", file=sys.stderr)
        figures[kind] = entry
    manifest = {"schema": MANIFEST_SCHEMA, "paper_width_in": PAPER_W, "aspect": ASPECT, "aspect_range": ASPECT_RANGE,
                "code_sha256": code_hashes(), "layout": {"fig_a": FIG_A, "fig_b": FIG_B},
                "figures": [figures[k] for k in ("fig_a", "fig_b") if k in figures]}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=f2._json_default) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
