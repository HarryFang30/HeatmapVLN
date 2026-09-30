#!/usr/bin/env python3
"""EXP-19 paper figures: IEEE double column (7.16 in wide), about 1:1, from the same records as ``fig_v2``.

Figure A -- what the robot saw and predicted at key moments.  One row per case (default T1, T2, T3)::

  (a) category · outcome
      "instruction"
  route map | K_a | K_b | K_c        (default K1, K2, K4; up to four)

  each key moment, top to bottom: the look-down image System 2 decided on (its pixel goal as a ring, the System 1
  path as dots; K badge + step in its top corner, the executed action chunk in a bottom corner), the 360° history
  affordance map strip (marks of past frames 1, 4, 8) and the 360° future affordance map strip.  Unlike the v2
  overview: no past-frame thumbnails, no System 2 text line (the pixel goal is the ring), no timeline.

Figure B -- the affordance maps online.  One row per case (default T1, T2, T3, F1)::

  (a) category · outcome
      "instruction"
  route map | online timeline: history panel, future panel, executed turns

  the step axis starts at the first call with an affordance map (``timeline_panel.crop_warmup``): the steps before
  it are left out, not compressed; each column marks past frames 1, 4, 8 as one group (``Timeline.compact148``).

Outputs per figure and language: PDF (TrueType text), SVG (live ``<text>``) and PNG (400 dpi), plus a caption
``.txt``; every text is real text (the route maps' white halos become small white boxes, ``editable_text``), so the
PDF / SVG can be fine-tuned in Illustrator or Inkscape once the fonts are installed (Nimbus Sans; zh: Droid Sans
Fallback).  Images and affordance maps are embedded rasters.  The layout numbers (inches) are the ``FIG_A`` /
``FIG_B`` constants below.

Policy as ``fig_v2``: no poses, VO or odometry; both heatmaps are "affordance map"; nothing is drawn from the future
map to the actions (the captions say it does not feed them); the captions make no claim (the RTX 4090 batch is
descriptive only, see the ledger's EXP-19 run record 5).

Usage (repo root on PYTHONPATH)::

  python -m scripts.exp19.figures.fig_paper --records <EXP>/records --timelines <EXP>/records_v2 \\
      --topdown-root <EXP>/topdown --out-dir <EXP>/figures_paper [--lang en zh] \\
      [--fig-a T1 T2 T3] [--fig-b T1 T2 T3 F1] [--keys K1 K2 K4]
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
from matplotlib.text import Text  # noqa: E402

MANIFEST_SCHEMA = "exp19-figures-paper-manifest-v2"
PAPER_W = 7.16  # IEEE double column
EDGE = 0.01  # the right-most frame lines stay this far inside the page
ASPECT_RANGE = (0.9, 1.15)  # height / width the figures aim for ("about 1:1"); outside it is a warning
FS = p2.FS
LINE = f2.LINE
KEY_LABELS = ("K1", "K2", "K3", "K4")

FIG_A = {"k_w_max": 1.52, "route_w_min": 1.5, "k_gap": 0.08, "route_gap": 0.12, "title_h": 0.15,
         "instr_gap": 0.035, "dec_gap": 0.03, "cap_h": 0.115, "strip_gap": 0.025, "chip_pt": 7.6,
         "chip_inset_pt": 3.0, "ticks_h": 0.13, "row_gap": 0.1, "legend_gap": 0.07}
FIG_B = {"route_w": 1.5, "ylab_min": 0.27, "title_h": 0.15, "instr_gap": 0.035, "badges": 0.12, "hist": 0.46,
         "gap_first": 0.13, "gap": 0.08, "fut": 0.42, "hist_title": f2.HIST_TITLE_H, "ticks": 0.13, "row_gap": 0.1,
         "legend_gap": 0.07}
DEFAULT_A = ("T1", "T2", "T3")
DEFAULT_B = ("T1", "T2", "T3", "F1")
DEFAULT_KEYS = ("K1", "K2", "K4")
LETTERS = "abcdefghij"

LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "letter": "({l})",
        "outcome_success": "success: stopped {ne} m from the goal at step {steps}",
        "outcome_stop": "failure: stopped {ne} m from the goal at step {steps}{os}",
        "outcome_cap": "failure: {steps}-step limit, {ne} m from the goal{os}",
        "outcome_other": "failure: ended at step {steps}, {ne} m from the goal{os}",
        "outcome_os": "; it had been within {r:g} m earlier",
        "cap_hist": "360° history affordance map",
        "cap_fut": "360° future affordance map",
        "panel_b_hist": "History affordance map",
        "panel_b_fut": "Future affordance map",
        "groups": ("Route", "History affordance map", "Future affordance map", "Decisions"),
        "legend": {
            "route": "route / reference path",
            "start_goal": "start / goal (3 m radius)",
            "stop": "where the rerun ended",
            "key": "key moment",
            "hist": "predicted field",
            "pred": "predicted peak of a past frame",
            "gt": "true direction of that frame",
            "pair": "peak joined to its true direction",
            "frame": "framed: the model's 79° front view",
            "slots_ov": "a column = one call; past frames 1, 4, 8",
            "fut": "predicted field (darker = later)",
            "path": "System 1 path",
            "path_end": "System 1 path endpoint",
            "goal": "pixel goal (System 2)",
            "actions": "executed actions",
            "turns": "executed turns (up = left)",
            "nomap": "System 2 gave turns or STOP",
        },
    },
    "zh": {
        "letter": "({l})",
        "outcome_success": "成功：第 {steps} 步停在距目标 {ne} m 处",
        "outcome_stop": "失败：第 {steps} 步停在距目标 {ne} m 处{os}",
        "outcome_cap": "失败：撞上 {steps} 步上限，距目标 {ne} m{os}",
        "outcome_other": "失败：第 {steps} 步结束，距目标 {ne} m{os}",
        "outcome_os": "；此前曾到过目标 {r:g} m 以内",
        "cap_hist": "360° 历史 affordance map",
        "cap_fut": "360° 未来 affordance map",
        "panel_b_hist": "历史 affordance map",
        "panel_b_fut": "未来 affordance map",
        "groups": ("路线", "历史 affordance map", "未来 affordance map", "决策"),
        "legend": {
            "route": "执行路线 / 参考路径",
            "start_goal": "起点 / 目标（3 m 半径）",
            "stop": "复跑结束处",
            "key": "关键时刻",
            "hist": "预测场",
            "pred": "历史帧的预测峰值",
            "gt": "该帧的真实方向",
            "pair": "峰值与真实方向连线",
            "frame": "黑框 = 输入模型的 79° 前视",
            "slots_ov": "一列 = 一次调用；画历史帧 1、4、8",
            "fut": "预测场（越深越晚）",
            "path": "快系统路径",
            "path_end": "快系统路径终点",
            "goal": "像素目标（慢系统）",
            "actions": "执行的动作",
            "turns": "执行的转向（上 = 左转）",
            "nomap": "慢系统直接给出转向或停止",
        },
    },
}
GROUPS_A = (("route", "start_goal", "stop", "key"), ("hist", "pred", "gt", "pair", "frame"), ("fut",),
            ("path", "goal", "actions"))
GROUPS_B = (("route", "start_goal", "stop", "key"), ("hist", "pred", "gt", "slots_ov"), ("fut",),
            ("path_end", "turns", "nomap"))

MARKS = {
    "en": ("On the history {where}, orange dots are the predicted peaks and blue circles the true directions of past "
           "frames 1 (the episode's first frame), 4 and 8 (the latest) of the eight given to System 2; the orange field "
           "combines all eight. A circle without a dot: that frame was predicted not visible; a dot without a circle: "
           "that frame is not visible from there."),
    "zh": ("历史{where}上，橙点为预测峰值、蓝圈为真实方向，只画送入慢系统的 8 个历史帧中的第 1（本集第一帧）、4、8（最近一帧）帧，"
           "橙色场综合全部 8 帧。有圈无点：模型认为该帧不可见；有点无圈：该帧从那里不可见。"),
}
MARKS_WHERE = {"en": {"a": "strips", "b": "panel"}, "zh": {"a": "条带", "b": "图"}}
CAPTION_A = {
    "en": ("Closed-loop reruns of {n} R2R val_unseen episodes: {rows}. Each row: the executed route on the top-down map "
           "with the key moments {keys}; then, at each key moment, the look-down image System 2 decided on (its pixel "
           "goal as a ring, the System 1 path as black dots, the executed action chunk as chips: ↑ forward 0.25 m, ←/→ "
           "turn 15°) and the 360° surroundings re-rendered at that position with the predicted history affordance map "
           "(orange) and the predicted future affordance map (green, darker = later; the System 1 path overlaid as "
           "black dots). {marks} A grey line joins a frame's peak to its true direction when the two are apart. Only "
           "the framed front 79° of the surroundings was given to the model; the strips repeat 8° past ±180°. The "
           "future affordance map does not feed the actions."),
    "zh": ("{n} 集 R2R val_unseen 的闭环复跑：{rows}。每行依次为：俯视图上的执行路线与关键时刻 {keys}；各关键时刻慢系统据以决策的"
           "下视帧（圈 = 像素目标，黑点 = 快系统路径，图像下角的方块 = 执行的动作块：↑ 前进 0.25 m，←/→ 转 15°），以及在该位置重"
           "渲染的 360° 环视及其上的预测历史 affordance map（橙）与预测未来 affordance map（绿，越深越晚；黑点为叠加的快系统路径）。"
           "{marks}同一帧的峰值与真实方向相距较远时以灰线相连。环视中只有黑框内的前视 79° 输入了模型；条带两端各重复 8°。未来 "
           "affordance map 不回流到动作。"),
}
CAPTION_B = {
    "en": ("The affordance maps online in {n} R2R val_unseen reruns: {rows}. Each row: the route on the top-down map with "
           "the key moments {keys}, and the timeline. x = step, from the first call with an affordance map (the steps "
           "before it are not shown); y = bearing around the robot, up = left: the history panel runs ahead, left, "
           "back, right, ahead from top to bottom; the future panel is centred on ahead, with back at its edges. Each "
           "call that returned a pixel goal is a column spanning its executed action chunk, with that call's predicted "
           "history affordance map (orange), the marks of its past frames, its predicted future affordance map (green, "
           "darker = later) and the endpoint of its System 1 path (black dot). {marks} {grey}Executed turns are marked "
           "below the panels (up = left). The future affordance map does not feed the actions."),
    "zh": ("{n} 集 R2R val_unseen 复跑中在线运行的 affordance map：{rows}。每行为俯视图上带关键时刻 {keys} 的路线与时间线。横轴为"
           "步数，从第一次给出 affordance map 的调用开始（此前的步数不画）；纵轴为机器人周围的方位，向上 = 向左：历史图自上而下为"
           "前、左、后、右、前，未来图以正前方居中、上下两端为后。每次给出像素目标的调用占一列，跨它执行的动作块，列内为该次调用"
           "的预测历史 affordance map（橙）、历史帧的标记、预测未来 affordance map（绿，越深越晚）及快系统路径终点（黑点）。{marks}"
           "{grey}图下方为执行的转向（上 = 左转）。未来 affordance map 不回流到动作。"),
}
CAPTION_GREY = {"en": "Grey columns: System 2 answered with turns or STOP (no affordance map). ",
                "zh": "灰列：慢系统直接给出转向或停止（没有 affordance map）。"}
CAPTION_K3_ELSEWHERE = {"en": "(K3 is marked on the timelines of the companion figure.)",
                        "zh": "（K3 标在另一张图的时间线上。）"}
CAPTION_TAIL = {
    "en": "Reruns on an RTX 4090, shown as examples; no claim is made from them.",
    "zh": "复跑于 RTX 4090，仅作示例，不据此下结论。",
}
CAPTION_SMOOTH = {
    "a": {"en": "The affordance maps are blurred for display (Gaussian, σ = {s:g}°); orange dots mark the unblurred peaks.",
          "zh": "affordance map 为显示做了高斯平滑（σ = {s:g}°）；橙点为未平滑的峰值。"},
    "b": {"en": ("The affordance maps are blurred in bearing for display (Gaussian, σ = {s:g}°); orange dots mark the "
                 "unblurred peaks."),
          "zh": "affordance map 在方位向为显示做了高斯平滑（σ = {s:g}°）；橙点为未平滑的峰值。"},
}
SHORT_RULES = {  # the key-moment rules by the branch each key moment took (``bd.KeyStep.branch``)
    "en": {"K1_first": "the first call with an affordance map",
           "K2_turn": ("the call with an affordance map, other than K1, whose executed chunk has the largest net turn "
                       "(at least 30°; the earliest if tied)"),
           "K2_fallback": ("the call a third of the way through the calls with an affordance map, as no other such "
                           "call's chunk has a net turn of 30° or more"),
           "K3_two_thirds": ("the call with an affordance map, other than K1 and K2, nearest two thirds of the "
                             "episode's steps"),
           "K3_f1_closest": ("the last call with an affordance map, other than K1 and K2, at or before the step "
                             "closest to the goal"),
           "K3_f1_fallback_after": ("the first call with an affordance map, other than K1 and K2, after the step "
                                    "closest to the goal (none at or before it; not in the pre-registration)"),
           "K4_last": "the last call with an affordance map",
           "K4_shifted": "the latest call with an affordance map not already chosen",
           "rule": "{label} = {rule}", "where": "; in {cases}, {label} = {rule}", "join": "; ", "cases": " and ",
           "end": ".", "lt4": " In {case}, {text}", "order": " Key moments are numbered by these rules, not in time order."},
    "zh": {"K1_first": "第一次给出 affordance map 的调用",
           "K2_turn": "除 K1 外、给出 affordance map 的调用中执行动作块净转角最大者（≥ 30°，并列取最早）",
           "K2_fallback": "这些调用中位于三分之一处者，因为其余调用的动作块净转角都不到 30°",
           "K3_two_thirds": "除 K1、K2 外、步号最接近全集 2/3 的给出 affordance map 的调用",
           "K3_f1_closest": "除 K1、K2 外，距目标最近那一步及其之前的最后一次给出 affordance map 的调用",
           "K3_f1_fallback_after": "除 K1、K2 外，距目标最近那一步之后的第一次给出 affordance map 的调用（该步及之前没有；此兜底不在预注册中）",
           "K4_last": "最后一次给出 affordance map 的调用",
           "K4_shifted": "尚未被选的最晚一次给出 affordance map 的调用",
           "rule": "{label} 为{rule}", "where": "；{cases} 中 {label} 为{rule}", "join": "；", "cases": "、",
           "end": "。", "lt4": "{case} 中，{text}", "order": "关键时刻按上述规则编号，不按时间先后。"},
}


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
    would mix a bold "(a)" with regular CJK: zh headings are regular, one step larger."""
    return "bold" if lang == "en" else "normal"


def cap_first(text: str, lang: str) -> str:
    return text[:1].upper() + text[1:] if lang == "en" else text


def outcome_short(b: bd.Bundle, L: dict) -> str:
    o = b.outcome
    if o is None:
        return L["outcome_missing"]
    ne = f"{o['ne_m']:.1f}" if fb._finite(o["ne_m"]) else "?"
    if o["success"]:
        return L["outcome_success"].format(steps=o["steps"], ne=ne)
    os_note = L["outcome_os"].format(r=float(b.goal_radius_m)) if o["oracle_success"] else ""
    key = {"stop": "outcome_stop", "step_cap": "outcome_cap"}.get(o["ended_by"], "outcome_other")
    return L[key].format(steps=o["steps"], ne=ne, os=os_note)


def category_of(b: bd.Bundle) -> str:
    return b.membership(main=True)["category"]


def row_head(fig, b: bd.Bundle, letter: str, lang: str, L: dict, width: float) -> dict:
    """The row's title (letter + category name), outcome, and the instruction wrapped to ``width`` (in)."""
    cat = category_of(b)
    name = cap_first(fb.CATEGORY_NAMES[lang][cat], lang)
    instr = f2.wrap_ink(fig, f2.LABELS[lang]["instruction"].format(text=" ".join(b.instruction.split())), FS["small"],
                        width * 72.0, fontstyle="italic")
    return {"title": f"{L['letter'].format(l=letter)} {name}", "outcome": outcome_short(b, L), "instr": instr,
            "cat": cat, "letter": letter, "name": fb.CATEGORY_NAMES[lang][cat]}


def head_height(head: dict, g: dict) -> float:
    return g["title_h"] + LINE * len(head["instr"]) + g["instr_gap"]


def draw_head(page: fb.Page, y: float, head: dict, lang: str) -> None:
    fig = page.fig
    ax = page.pt_axes(0.0, y, PAPER_W, 0.15)
    fs, w = (FS["head"] + 0.6 if lang == "en" else FS["head"] + 1.2), heading_weight(lang)
    ax.text(0.0, 5.4, head["title"], ha="left", va="center", fontsize=fs, fontweight=w, color=style.INK)
    xo = cd.text_width_pt(fig, head["title"], fs, fontweight=w) + 7.0
    ax.text(xo, 5.4, head["outcome"], ha="left", va="center", fontsize=FS["small"], color=style.INK_2)
    for j, line in enumerate(head["instr"]):
        page.text(0.0, y + 0.15 + LINE * (j + 0.5), line, ha="left", va="center", fontsize=FS["small"],
                  color=style.INK_2, fontstyle="italic")


def rule(page: fb.Page, y: float) -> None:
    """A full-width hairline between rows (drawn unclipped: ``fb.Page.rule`` clips it to a 0.001-in axes)."""
    ax = page.ax(0.0, y, PAPER_W, 0.001)
    ax.axis("off")
    ax.axhline(0.5, color=style.AXIS, lw=0.6, clip_on=False)


GLYPH_W, GLYPH_TEXT_PT, LEGEND_GAP_PT = 14.0, f2.GLYPH_TEXT_PT, 10.0


def glyph(ax, key: str, x: float, y: float) -> None:
    if key == "path_end":  # the timeline's one dot per column
        p2.path_marks(ax, [x + GLYPH_W / 2], [y], ms=1.9 * tp.TL_MARK, rim=True, clip_on=False)
    else:
        p2.legend_glyph(ax, key, x, y, w=GLYPH_W)


def legend_layout(fig, groups: Sequence[Sequence[str]], names: Sequence[str], texts: Dict[str, str],
                  width_pt: float, fs: float, head_weight: str) -> dict:
    """One column per group (its name on top, entries below), in the figure's left-to-right order; while the
    columns do not fit ``width_pt`` the widest entry is wrapped onto two lines.  Returns {"cols", "x", "rows"}."""
    cols = [(name, [[k, [texts[k]]] for k in grp]) for name, grp in zip(names, groups) if grp]

    def col_w(head, items):
        w = cd.text_width_pt(fig, head, fs, fontweight=head_weight)
        return max([w] + [GLYPH_TEXT_PT + max(cd.text_width_pt(fig, ln, fs) for ln in lines) for _, lines in items])

    for _ in range(12):
        widths = [col_w(h, it) for h, it in cols]
        if sum(widths) + LEGEND_GAP_PT * (len(cols) - 1) <= width_pt:
            break
        c = int(np.argmax(widths))
        single = [it for it in cols[c][1] if len(it[1]) == 1]
        if not single:
            break
        it = max(single, key=lambda e: cd.text_width_pt(fig, e[1][0], fs))
        text = it[1][0]
        it[1] = f2.split_at_semicolon(text) or fb.wrap(fig, text, fs, cd.text_width_pt(fig, text, fs) * 0.6 + 6.0)[:2]
    widths = [col_w(h, it) for h, it in cols]
    spare = max(0.0, width_pt - sum(widths) - LEGEND_GAP_PT * (len(cols) - 1))
    xs, x = [], 0.0
    for w in widths:
        xs.append(x)
        x += w + LEGEND_GAP_PT + (spare / (len(cols) - 1) if len(cols) > 1 else 0.0)
    rows = max(sum(len(lines) for _, lines in items) for _, items in cols) if cols else 0
    return {"cols": cols, "x": xs, "rows": rows}


def draw_legend(page: fb.Page, y: float, lay: dict, lang: str) -> float:
    fs = FS["small"]
    h = f2.LEGEND_HEAD_H + LINE * lay["rows"]
    ax = page.pt_axes(0.0, y, PAPER_W, h)
    top = h * 72.0
    for (head, items), cx in zip(lay["cols"], lay["x"]):
        ax.text(cx, top - f2.LEGEND_HEAD_H * 72.0 * 0.45, head, ha="left", va="center", fontsize=fs,
                fontweight=heading_weight(lang), color=style.INK)
        i = 0
        for key, lines in items:
            yy = top - f2.LEGEND_HEAD_H * 72.0 - (i + 0.5) * LINE * 72.0
            glyph(ax, key, cx, yy - (len(lines) - 1) * LINE * 36.0)
            for j, line in enumerate(lines):
                ax.text(cx + GLYPH_TEXT_PT, yy - j * LINE * 72.0, line, ha="left", va="center", fontsize=fs,
                        color=style.INK)
            i += len(lines)
    return h


def legend_groups(groups, drop: Sequence[str]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(k for k in grp if k not in drop) for grp in groups)


def legend_for(fig, groups, L: dict, lang: str) -> dict:
    names = [n for n, grp in zip(L["groups"], groups) if grp]
    return legend_layout(fig, [g for g in groups if g], names, L["legend"], PAPER_W * 72.0, FS["small"],
                         heading_weight(lang))


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


def key_rules(cases: Sequence[Tuple[str, Sequence]], lang: str) -> str:
    """How the drawn key moments were chosen, from the branch each one took: per label, the branch most rows took,
    then the rows that took another (named with theirs); a row with fewer than four calls with an affordance map
    (branch "all_lt4") gets its own sentence; the numbering-vs-time note only when some row runs against time.
    ``cases``: (panel letter, key moments drawn) per row.  "" when nothing is drawn."""
    R, full = SHORT_RULES[lang], fb.KEY_RULES[lang]
    by_label: Dict[str, Dict[str, List[str]]] = {}
    lt4 = []
    for letter, keys in cases:
        if any(k.branch == "all_lt4" for k in keys):
            text = (full["all_lt4_one"] if len(keys) == 1 else full["all_lt4"]).format(
                n=len(keys), labels=", ".join(k.label for k in keys))
            lt4.append(R["lt4"].format(case=f"({letter})", text=text[:1].lower() + text[1:] if lang == "en" else text))
            continue
        for k in keys:
            by_label.setdefault(k.label, {}).setdefault(k.branch, []).append(f"({letter})")
    parts = []
    for label in sorted(by_label):
        items = sorted(by_label[label].items(), key=lambda kv: -len(kv[1]))
        text = R["rule"].format(label=label, rule=R.get(items[0][0], full.get(items[0][0], items[0][0])))
        for branch, where in items[1:]:
            text += R["where"].format(cases=R["cases"].join(where), label=label,
                                      rule=R.get(branch, full.get(branch, branch)))
        parts.append(text)
    out = (R["join"].join(parts) + R["end"]) if parts else ""
    out += "".join(lt4)
    against_time = any(int(b.step) < int(a.step) for _, keys in cases for a in keys for b in keys
                       if str(b.label) > str(a.label))
    if parts and against_time:
        out += R["order"]
    return out.strip()


def rows_text(heads: Sequence[dict], lang: str) -> str:
    return ("; " if lang == "en" else "；").join(f"({h['letter']}) {h['name']}" for h in heads)


def join_caption(parts: Sequence[str], lang: str) -> str:
    """Caption sentences joined (a space between English sentences, none between Chinese ones)."""
    text = (" " if lang == "en" else "").join(p.strip() for p in parts if p and p.strip())
    return " ".join(text.split()) if lang == "en" else text


def audit(fig, height: float, name: str) -> dict:
    """Layout checks: texts outside the page, overlapping texts, leaders crossing texts, smallest font, aspect."""
    outside, overlaps, crossings = f2.texts_outside(fig), f2.text_overlaps(fig), f2.leader_crossings(fig)
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
def key_width(n_keys: int, g: dict = FIG_A) -> float:
    """Width (in) of a key-moment column: ``k_w_max``, narrower when more columns would leave the route map less
    than ``route_w_min``."""
    room = PAPER_W - EDGE - g["route_w_min"] - g["route_gap"] - (n_keys - 1) * g["k_gap"]
    return min(g["k_w_max"], room / max(n_keys, 1))


def moment_rows(w: float, captions: bool, ticks: bool, g: dict = FIG_A) -> Dict[str, float]:
    """Top offset (in) of every part of a key-moment column; ``captions``: the strip caption rows (first row only),
    ``ticks``: the bearing labels under the future strip (last row only)."""
    y, out = 0.0, {}
    cap = g["cap_h"] if captions else 0.0
    for name, h in (("dec", w * 0.75), ("gap1", g["dec_gap"]), ("cap_hist", cap),
                    ("hist", p2.strip_height(w, p2.HIST_ELEV)), ("gap2", 0.0 if captions else g["strip_gap"]),
                    ("cap_fut", cap), ("fut", p2.strip_height(w, p2.FUT_ELEV)), ("ticks", g["ticks_h"] if ticks else 0.0)):
        out[name] = y
        y += h
    out["end"] = y
    return out


def drawn_image_marks(ks: bd.KeyStep) -> np.ndarray:
    """(u, v) of the marks ``p2.draw_decision_image`` draws: every waypoint it keeps inside the image, and the pixel
    goal when inside."""
    img = ks.decision_rgb
    h, w = img.shape[:2]
    uv = np.asarray(ks.path_uv, dtype=np.float64).reshape(-1, 2)
    out = []
    if len(uv):
        uv = uv[p2.subsample_path(len(uv))]
        inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)
        out += list(uv[inside])
    goal = ks.pixel_goal_uv
    if goal is not None and bd.goal_inside(goal, img.shape):
        out.append(np.asarray(goal, dtype=np.float64).reshape(2))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def chips_corner(ks: bd.KeyStep, w: float, width_pt: float, g: dict = FIG_A) -> str:
    """The bottom corner of the decision image for the executed-action chips: the right one, unless a drawn mark
    (pixel goal or System 1 waypoint, ``drawn_image_marks``) falls under the chips there and none under the left."""
    h_img, w_img = ks.decision_rgb.shape[:2]
    per = w_img / (w * 72.0)  # image pixels per point
    box_w, box_h = (width_pt + 2 * g["chip_inset_pt"]) * per, (g["chip_pt"] + 2 * g["chip_inset_pt"]) * per
    pts = drawn_image_marks(ks)
    if not len(pts):
        return "right"
    low = pts[:, 1] >= h_img - box_h - 6 * per
    right = int(np.sum(low & (pts[:, 0] >= w_img - box_w - 6 * per)))
    left = int(np.sum(low & (pts[:, 0] <= box_w + 6 * per)))
    return "left" if right and not left else "right"


def draw_chips(page: fb.Page, x: float, y: float, w: float, ks: bd.KeyStep, g: dict = FIG_A) -> str:
    """The executed action chunk as chips in a bottom corner of the decision image (``chips_corner``)."""
    size, inset = g["chip_pt"], g["chip_inset_pt"]
    acts = [int(a) for a in ks.executed_actions]
    width = sum(p2.chip_width(a, size) for a in acts) + 1.5 * max(len(acts) - 1, 0)
    corner = chips_corner(ks, w, width)
    kax = page.pt_axes(x, y, w, w * 0.75, zorder=4)
    x0 = w * 72.0 - inset - width if corner == "right" else inset
    p2.action_chips(kax, x0, inset + size / 2, acts, size=size)
    return corner


def draw_moment(page: fb.Page, x: float, y: float, w: float, ks: bd.KeyStep, L: dict, captions: bool,
                ticks: bool, first_col: bool = False) -> dict:
    """One key moment: decision image (badge + step in its top corner, executed actions in a bottom corner),
    360° history strip (past frames 1, 4, 8 marked), 360° future strip.  ``captions``: the strips' caption rows
    (first row), with their text in the first column only."""
    rows = moment_rows(w, captions, ticks)
    dax = page.ax(x, y + rows["dec"], w, w * 0.75)
    p2.draw_decision_image(dax, ks, tag=L["step"].format(s=ks.step), badge=ks.label)
    corner = draw_chips(page, x, y + rows["dec"], w, ks)
    if captions and first_col:
        for key in ("cap_hist", "cap_fut"):
            page.text(x, y + rows[key] + FIG_A["cap_h"] * 0.5, L[key], ha="left", va="center", fontsize=p2.MIN_FS,
                      color=style.INK_2)
    bax = page.ax(x, y + rows["hist"], w, p2.strip_height(w, p2.HIST_ELEV))
    info = p2.draw_history_strip(bax, ks, f2._ring_px(w, 340), f2._ring_px(w, 272), slots=tp.OVERVIEW_SLOTS)
    cax = page.ax(x, y + rows["fut"], w, p2.strip_height(w, p2.FUT_ELEV))
    p2.draw_future_strip(cax, ks, f2._ring_px(w, 272))
    if ticks:
        p2.strip_ticks(cax, L["axis"])
    return {**info, "chips_corner": corner}


def make_fig_a(bundles: Sequence[bd.Bundle], out_dir: Path, lang: str, topdown_root,
               keys: Sequence[str] = DEFAULT_KEYS, name: str = "fig_a_key_moments") -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = paper_labels(lang)
    g = FIG_A
    n_k = len(keys)
    k_w = key_width(n_k)
    route_w = PAPER_W - EDGE - n_k * k_w - (n_k - 1) * g["k_gap"] - g["route_gap"]
    x_k = route_w + g["route_gap"]
    fig = plt.figure(figsize=(PAPER_W, 10.0))
    heads = [row_head(fig, b, LETTERS[i], lang, L, PAPER_W) for i, b in enumerate(bundles)]
    chosen = [fb.main_keys_of(b, keys) for b in bundles]
    last = len(bundles) - 1
    body = [moment_rows(k_w, captions=(i == 0), ticks=(i == last))["end"] for i in range(len(bundles))]
    route_h = [bh - (g["ticks_h"] if i == last else 0.0) for i, bh in enumerate(body)]  # flush with the strips
    row_h = [head_height(h, g) + bh for h, bh in zip(heads, body)]
    stops = [p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m), route_w,
                           rh) for b, rh in zip(bundles, route_h)]
    groups = legend_groups(GROUPS_A, [] if any(stops) else ["stop"])
    lay = legend_for(fig, groups, L, lang)
    lh = f2.LEGEND_HEAD_H + LINE * lay["rows"]
    height = sum(row_h) + g["row_gap"] * last + g["legend_gap"] + lh + 0.02
    fig.set_size_inches(PAPER_W, height)
    page = fb.Page(fig, PAPER_W, height)
    y = 0.0
    checks, edge, pairs, stop_drawn, corners = [], 0, 0, [], []
    for i, (b, head, ch) in enumerate(zip(bundles, heads, chosen)):
        draw_head(page, y, head, lang)
        yb = y + head_height(head, g)
        rinfo = f2.draw_route(page, 0.0, yb, route_w, route_h[i], b, fb.resolve_level(b, topdown_root), L,
                              labels=[k.label for k in ch])
        stop_drawn.append(rinfo["stop_drawn"])
        for c, ks in enumerate(ch):
            info = draw_moment(page, x_k + c * (k_w + g["k_gap"]), yb, k_w, ks, L, captions=(i == 0),
                               ticks=(i == last), first_col=(c == 0))
            edge += info["edge"]
            pairs += info.get("pairs", 0)
            corners.append(info["chips_corner"])
        checks.append({"ep_key": b.ep_key, "category": head["cat"], "keys": [(k.label, int(k.step)) for k in ch],
                       "stop_drawn": rinfo["stop_drawn"], "start_label": rinfo["start_label"]})
        y += row_h[i] + g["row_gap"]
        if i < last:
            rule(page, y - g["row_gap"] / 2)
    y += g["legend_gap"] - g["row_gap"]
    rule(page, y - g["legend_gap"] / 2)
    draw_legend(page, y, lay, lang)
    n_halo = editable_text(fig)
    labels = sorted({k.label for ch in chosen for k in ch})
    rules = key_rules([(LETTERS[i], ch) for i, ch in enumerate(chosen)], lang)
    k3_elsewhere = CAPTION_K3_ELSEWHERE[lang] if "K3" not in labels and any(
        any(k.label == "K3" for k in b.keys) for b in bundles) else ""
    caption = join_caption([CAPTION_A[lang].format(n=len(bundles), rows=rows_text(heads, lang),
                                                   keys=("、" if lang == "zh" else ", ").join(labels),
                                                   marks=MARKS[lang].format(where=MARKS_WHERE[lang]["a"])),
                            rules, k3_elsewhere, f2.CAPTION_EDGE[lang] if edge else "", CAPTION_TAIL[lang],
                            CAPTION_SMOOTH["a"][lang].format(s=p2.SMOOTH_DEG)], lang)
    res = audit(fig, height, f"{name} [{lang}]")
    if stop_drawn != stops:
        res["warnings"].append(f"{name} [{lang}]: end-of-rerun squares drawn {stop_drawn} but planned {stops}")
    files = save(fig, Path(out_dir) / name, lang, caption)
    return {"files": files, "size_in": (PAPER_W, round(height, 3)), **res, "checks": checks, "edge_marks": edge,
            "strip_pair_lines": pairs, "chips_corners": corners, "halo_texts_boxed": n_halo,
            "caption_chars": len(caption)}


# --------------------------------------------------------------------------- #
# Figure B: route + online timeline (the steps before the first affordance map left out)
# --------------------------------------------------------------------------- #
def timeline_block_h(badges_h: float, first: bool, g: dict = FIG_B) -> float:
    """Height of a row's timeline block below the head: (history title +) badges + history + gap + future + turns +
    step ticks."""
    title = g["hist_title"] if first else 0.0
    gap = g["gap_first"] if first else g["gap"]
    return f2.timeline_height(g["hist"], gap, g["fut"], badges_h, title) + g["ticks"]


def visible_nomap(tl: tp.Timeline) -> bool:
    """A no-map span (System 2 answered with turns or STOP) inside the drawn axis."""
    lo, hi = tl.xlim()
    return any(kind == "nomap" and s1 > lo and s0 < hi for kind, s0, s1 in tl.spans())


def cropped_nomap_steps(tl: tp.Timeline) -> int:
    """Steps of no-map calls (after the warm-up) left out by the crop."""
    w, x0 = float(tl.warmup_end()), float(tl.x0)
    return int(round(sum(max(0.0, min(s1, x0) - max(s0, w)) for kind, s0, s1 in tl.spans() if kind == "nomap")))


def make_fig_b(bundles: Sequence[bd.Bundle], timelines: Sequence[tp.Timeline], out_dir: Path, lang: str,
               topdown_root, name: str = "fig_b_online_timeline") -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = paper_labels(lang)
    g = FIG_B
    fig = plt.figure(figsize=(PAPER_W, 10.0))
    heads = [row_head(fig, b, LETTERS[i], lang, L, PAPER_W) for i, b in enumerate(bundles)]
    ylab_w = max(cd.text_width_pt(fig, t, p2.MIN_FS) for t in list(L["y_hist"]) + list(L["y_fut"]) + [L["turns"]])
    tl_x = g["route_w"] + max(g["ylab_min"], (ylab_w + 9.0) / 72.0)
    tl_w = PAPER_W - tl_x - EDGE
    badges = []
    for tl in timelines:
        tp.crop_warmup(tl)
        tl.compact148 = True
        badges.append(tp.badges_height_in(tp.badge_levels(fig, tl, tl_w * 72.0), g["badges"]))
    badges_h = max(badges)
    blocks = [timeline_block_h(badges_h, first=(i == 0)) for i in range(len(bundles))]
    row_h = [head_height(h, g) + bh for h, bh in zip(heads, blocks)]
    route_hs = [bh - g["ticks"] for bh in blocks]
    stops = [p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                           g["route_w"], rh) for b, rh in zip(bundles, route_hs)]
    grey = any(visible_nomap(tl) for tl in timelines)
    groups = legend_groups(GROUPS_B, ([] if any(stops) else ["stop"]) + ([] if grey else ["nomap"]))
    lay = legend_for(fig, groups, L, lang)
    lh = f2.LEGEND_HEAD_H + LINE * lay["rows"]
    last = len(bundles) - 1
    height = sum(row_h) + g["row_gap"] * last + g["legend_gap"] + lh + 0.02
    fig.set_size_inches(PAPER_W, height)
    page = fb.Page(fig, PAPER_W, height)
    y = 0.0
    checks, stop_drawn = [], []
    for i, (b, tl, head, route_h) in enumerate(zip(bundles, timelines, heads, route_hs)):
        first = i == 0
        draw_head(page, y, head, lang)
        yb = y + head_height(head, g)
        info = f2.draw_timeline(page, tl_x, yb, tl_w, g["hist"], g["gap_first"] if first else g["gap"], g["fut"],
                                tl, L, badges_h=badges_h, fut_title=first, warmup_label=False,
                                title_h=g["hist_title"] if first else 0.0, slots=tp.OVERVIEW_SLOTS)
        info.pop("badges", None)
        rinfo = f2.draw_route(page, 0.0, yb, g["route_w"], route_h, b, fb.resolve_level(b, topdown_root), L)
        stop_drawn.append(rinfo["stop_drawn"])
        checks.append({"ep_key": b.ep_key, "category": head["cat"], "x0_step": float(tl.x0),
                       "first_ready_step": int(np.min(tl.a["step"])) if tl.R else None,
                       "warmup_end": int(tl.warmup_end()), "cropped_nomap_steps": cropped_nomap_steps(tl),
                       "stop_drawn": rinfo["stop_drawn"], "start_label": rinfo["start_label"], **info})
        y += row_h[i] + g["row_gap"]
        if i < last:
            rule(page, y - g["row_gap"] / 2)
    y += g["legend_gap"] - g["row_gap"]
    rule(page, y - g["legend_gap"] / 2)
    draw_legend(page, y, lay, lang)
    n_halo = editable_text(fig)
    drawn_keys = [[k for k in b.keys if k.label in tl.key_rows()] for b, tl in zip(bundles, timelines)]
    labels = sorted({k.label for ks in drawn_keys for k in ks})
    rules = key_rules([(LETTERS[i], ks) for i, ks in enumerate(drawn_keys)], lang)
    keys_text = (f"{labels[0]}–{labels[-1]}" if len(labels) > 2 else ("、" if lang == "zh" else " and ").join(labels))
    caption = join_caption([CAPTION_B[lang].format(n=len(bundles), rows=rows_text(heads, lang), keys=keys_text,
                                                   marks=MARKS[lang].format(where=MARKS_WHERE[lang]["b"]),
                                                   grey=CAPTION_GREY[lang] if grey else ""),
                            rules, CAPTION_TAIL[lang], CAPTION_SMOOTH["b"][lang].format(s=tp.TL_SMOOTH_DEG)], lang)
    res = audit(fig, height, f"{name} [{lang}]")
    if stop_drawn != stops:
        res["warnings"].append(f"{name} [{lang}]: end-of-rerun squares drawn {stop_drawn} but planned {stops}")
    files = save(fig, Path(out_dir) / name, lang, caption)
    return {"files": files, "size_in": (PAPER_W, round(height, 3)), **res, "checks": checks,
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


def check_keys(keys: Sequence[str]) -> List[str]:
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys) or any(k not in KEY_LABELS for k in keys):
        raise SystemExit(f"--keys {keys}: want 1-4 distinct labels from {KEY_LABELS}")
    return sorted(keys)


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
    ap.add_argument("--keys", nargs="+", default=list(DEFAULT_KEYS), help="key moments of figure A (1-4 of K1-K4)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.resolve().name in ("figures", "figures_v2"):
        print("refusing to write into the v1 / v2 figures dir", file=sys.stderr)
        return 2
    keys = check_keys(args.keys)
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
        entry["args"] = {"rows": [s["ep_key"] for s in sources], **({"keys": keys} if kind == "fig_a" else {})}
        for lang in args.lang:
            warnings = list(pick_warnings)
            if kind == "fig_a":
                res = make_fig_a(members, out_dir, lang, args.topdown_root, keys=keys)
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
    manifest = {"schema": MANIFEST_SCHEMA, "paper_width_in": PAPER_W, "aspect_range": ASPECT_RANGE,
                "code_sha256": code_hashes(), "layout": {"fig_a": FIG_A, "fig_b": FIG_B},
                "figures": [figures[k] for k in ("fig_a", "fig_b") if k in figures]}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=f2._json_default) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
