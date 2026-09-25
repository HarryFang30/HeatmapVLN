#!/usr/bin/env python3
"""EXP-18 route-pattern figures (tier E): what the predicted affordance map does when the past lies ahead.

One figure per designed route pattern, ``out_and_back`` and ``loop``, from the
pre-registered picks in ``cases.json`` (``select_cases.py``: the episode whose
per-episode PCK@8 of the deployed model is the pattern's median; key rows = the
scored rows just before, at and just after the turnaround, and the final row).
On R2R routes the past lies behind the robot; on the way back along a designed
route it moves into the FRONT view, which is rare in training.  The figure
shows whatever the prediction does there, hit or miss, and the caption states
the result (panel d's return-leg totals and the tier-E front- vs back-view
PCK@8 from ``metrics.json``), not only the setup.

Only tier-E dumps are drawn: ``make_route_figure`` and ``make_route_figures``
raise on anything else (the development stand-ins of the pre-E builds are
retired, with their banner and caption preamble).

Layout = the case figure (``fig_case.make_case_figure``: D1 misses = joint
PCK@8 failures, D2 ringed miss numbers with leaders, D3 row names in a fixed
gutter, D4 one elevation-window rule stated in the caption, D5 notes wrapped,
never dropped, D6 wording) plus:

* no title: the claim and the result are the caption's first sentences;
* a  Route (``fig_case.FittedRoutePanel``): cropped to the route plus
  ``fig_case.ROUTE_PAD_M``, turned by a quarter turn when that shows it
  larger; the route in two legs split where it turns back, outbound solid
  and return dashed, each on its own right-hand side, chevrons for the
  direction of travel, a diamond at the turn (a K position there shows as a
  dot inside it), an open circle at the start; a legend in the panel names
  the four; K badges fan out from the route with leaders that never cross;
  one scale bar, always top left;
* b  insets: the route so far in the same two leg styles; the scale bar in
  one corner for every block;
* block headers name the key position's role ("after the turnaround");
* d  (``FrontPanel``, under the map, the rest of column a, >= 1.2 in of rows):
  legend first, then one row per scored frame, time running down like the
  blocks (1-based frame number on the left, K badges on the right); per row
  two bars side by side with their counts printed: blue = past positions whose
  true direction lies in the front view, orange = how many of them the
  predicted affordance map gets right (joint PCK@8); the return leg is shaded
  (diamond = the robot reaching the turn) and its totals are printed under the
  chart, the blue count in ground-truth ink and the orange one in prediction
  ink;
* caption: when key positions share one spot (the robot turns on the spot at
  the turn, e.g. K2 and K3), it says so.

Figure policy (user decision, 2026-09-24; as ``fig_case``): the figure shows
the affordance map only and never mentions poses or their sources; the
prediction is always the deployed model's (the dump's ``vo`` arm).  Ground
truth blue, prediction orange; misses shown; only the front image is marked as
model input.

Where the route "turns back" (the leg split, the diamond and d's shading):
out_and_back -> the route's own turnaround frame (``turnaround_frame`` in
cases.json, else ``geometry.out_and_back_turnaround`` on the dump's
reference_path); loop, and any clip without a palindromic reference_path ->
the frame farthest (3D) from the first frame (``data.route_split``).

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_routes --cases <cases.json> [--dumps-root DIR] [--topdown-root DIR]
      [--clip-root DIR] [--metrics-json FILE] [--lang en,zh] --out-dir DIR
  # one tier-E dump by hand (rows by the same rule, or explicit)
  python -m scripts.exp18.figures.fig_routes --dump <tier-E clip.npz> --pattern out_and_back [--rows 4,5,6,14]
      [--lang en] --out-dir DIR
Writes route_<pattern>[_zh].{pdf,png} and _caption.txt per pattern.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import fig_case as fc
from scripts.exp18.figures import style

from matplotlib.patches import Rectangle  # noqa: E402

ARM = fc.ARM  # the deployed model's prediction
FRONT_SHARE_TRAINING = "0.27%"  # EXP-18 ledger (EXP-11): share of training labels in the front view

# --------------------------------------------------------------------------- #
# Labels (every string on the figure and in the caption)
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "claim": {"out_and_back": "Out-and-back route: on the way back, past positions come to lie ahead of the "
                                  "robot, in the front view, and the predicted affordance map {verdict}.",
                  "loop": "Loop route: on the way back to the start, past positions come to lie ahead of the robot, "
                          "in the front view, and the predicted affordance map {verdict}."},
        "claim_none": {"out_and_back": "Out-and-back route: predicted affordance maps on the way out and back.",
                       "loop": "Loop route: predicted affordance maps around the loop."},
        "verdict_wrong": "gets most of them wrong",
        "verdict_right": "gets most of them right",
        "verdict_half": "gets half of them right",
        "roles": {
            "out_and_back": {"before_turnaround": "before the turnaround", "turnaround": "at the turnaround",
                             "after_turnaround": "after the turnaround"},
            "loop": {"before_turnaround": "before the farthest point", "turnaround": "at the farthest point",
                     "after_turnaround": "after the farthest point"},
        },
        "back_at_start": "back at the start",
        "from_start": "{d:.1f} m from the start",
        "turn": {"out_and_back": "turnaround", "loop": "farthest point"},
        "start": "start",
        "leg_out": "outbound",
        "leg_back": "return",
        "d_title": "Past positions in the front view",
        "d_frame": "frame",
        "d_xlabel": "past positions (of 8)",
        "d_leg_front": "in the front view\n(ground truth)",
        "d_leg_hits": "predicted correctly",
        "d_leg_back": {"out_and_back": "return leg", "loop": "return leg"},
        "d_total_head": "Return leg in total:",
        "d_total_n": "{n} in the front view",
        "d_total_h": "{h} predicted correctly",
        "d_total_0": "Return leg: none in the front view",
        "same_spot": "At the {turn} the robot turns on the spot, so {keys} share one position. ",
        "scene_E": "unseen scene {scene}\nroute {short} · {T} frames",
        "where": "unseen scene {scene}, route {short}",
    },
    "zh": {
        "claim": {"out_and_back": "去而复返：回程时部分历史位置来到机器人前方、落在前视里，预测 affordance map {verdict}。",
                  "loop": "绕圈：返回起点途中部分历史位置来到机器人前方、落在前视里，预测 affordance map {verdict}。"},
        "claim_none": {"out_and_back": "去而复返：去程与回程上的预测 affordance map。",
                       "loop": "绕圈：一圈中的预测 affordance map。"},
        "verdict_wrong": "对其中大多数判错",
        "verdict_right": "对其中大多数判对",
        "verdict_half": "对其中一半判对",
        "roles": {
            "out_and_back": {"before_turnaround": "折返前", "turnaround": "折返点", "after_turnaround": "折返后"},
            "loop": {"before_turnaround": "最远点前", "turnaround": "最远点", "after_turnaround": "最远点后"},
        },
        "back_at_start": "回到起点",
        "from_start": "距起点 {d:.1f} m",
        "turn": {"out_and_back": "折返点", "loop": "最远点"},
        "start": "起点",
        "leg_out": "去程",
        "leg_back": "回程",
        "d_title": "前视中的历史位置",
        "d_frame": "帧",
        "d_xlabel": "历史位置个数（共 8 个）",
        "d_leg_front": "落在前视中（真值）",
        "d_leg_hits": "其中预测正确的",
        "d_leg_back": {"out_and_back": "回程", "loop": "回程"},
        "d_total_head": "回程合计：",
        "d_total_n": "前视中 {n} 个",
        "d_total_h": "其中预测正确 {h} 个",
        "d_total_0": "回程合计：前视中没有",
        "same_spot": "机器人在{turn}原地转身，{keys} 位于同一位置。",
        "scene_E": "未见场景 {scene}\n路线 {short} · {T} 帧",
        "where": "未见场景 {scene}，路线 {short}",
    },
}

_RESULT = {
    "en": {"main": ("Over the return leg's {rows} scored frames, {n} past positions lie in the front view and {h} of "
                    "them are predicted correctly (joint PCK@8){out}. "),
           "out_0": "; on the outbound leg none does",
           "out_n": "; on the outbound leg: {n} in the front view, {h} predicted correctly",
           "none": ("On this route no past position lies in the front view on the return leg ({rows} scored frames)"
                    "{out}. "),
           "tier": ("Over all {clips} designed routes, {front:.1f}% of past positions in the front view are predicted "
                    "correctly (n = {n_front:,}), against {back:.1f}% in the back view (n = {n_back:,}). ")},
    "zh": {"main": "回程 {rows} 个评分帧中共有 {n} 个历史位置落在前视，预测正确 {h} 个（joint PCK@8）{out}。",
           "out_0": "；去程中没有落在前视的",
           "out_n": "；去程中落在前视的有 {n} 个，预测正确 {h} 个",
           "none": "这条路线的回程（{rows} 个评分帧）中没有落在前视的历史位置{out}。",
           "tier": ("全部 {clips} 条设计路线上，落在前视的历史位置预测正确 {front:.1f}%（n = {n_front}），"
                    "落在后视的为 {back:.1f}%（n = {n_back}）。")},
}
_SETUP = {
    "en": {
        "out_and_back": ("Predicted affordance maps on a designed out-and-back route ({where}): the robot follows an "
                         "R2R reference path to its end and comes back along it. Past positions in the front view are "
                         "rare in training ({share} of the past positions in the training labels). Key positions "
                         "(pre-registered): the scored frames just before, nearest to and just after the turnaround, "
                         "and the last frame. "
                         "(a) Route on the top-down map: outbound leg solid, return leg dashed, each drawn on its own "
                         "right-hand side so the two stay apart where the return retraces the outbound path; chevrons "
                         "give the direction of travel; diamond = turnaround (the end of the reference path), circle "
                         "= start; black dots with arrows = the key positions K1–K4 and the way the robot faces "
                         "there. "),
        "loop": ("Predicted affordance maps on a designed loop ({where}): the robot visits three waypoints on one "
                 "floor and returns to its start. Past positions in the front view are rare in training ({share} of "
                 "the past positions in the training labels). Key positions (pre-registered): the scored frames just "
                 "before, at and just after the point farthest from the start, and the last frame. (a) Route on the "
                 "top-down map: the leg up to the point farthest from the start (diamond) solid, the rest dashed, "
                 "each drawn on its own right-hand side; chevrons give the direction of travel; circle = start; "
                 "black dots with arrows = the key positions K1–K4 and the way the robot faces there. "),
    },
    "zh": {
        "out_and_back": ("设计的去而复返路线上的预测 affordance map（{where}）：机器人沿一条 R2R 参考路径走到头，再原路返回。"
                         "落在前视的历史位置在训练中很少见（只占训练标签中历史位置的 {share}）。关键位置（预注册）：折返前、离折返最近、"
                         "折返后的评分帧，以及末帧。(a) 俯视图上的路线：去程实线、回程虚线，各自画在行进方向的右侧，使原路"
                         "返回的两段不重叠；路线上的 V 形标记表示行进方向；菱形 = 折返点（参考路径的终点），圆圈 = 起点；带箭头的黑点 = "
                         "关键位置 K1–K4 及机器人在该处面对的方向。"),
        "loop": ("设计的绕圈路线上的预测 affordance map（{where}）：机器人在同一层依次经过三个路点后回到起点。落在前视的"
                 "历史位置在训练中很少见（只占训练标签中历史位置的 {share}）。关键位置（预注册）：离起点最远处之前、当时、之后的"
                 "评分帧，以及末帧。(a) 俯视图上的路线：到最远点（菱形）为止实线，其后虚线，各自画在行进方向的右侧；"
                 "路线上的 V 形标记表示行进方向；圆圈 = 起点；带箭头的黑点 = 关键位置 K1–K4 及机器人在该处面对的方向。"),
    },
}
_B_LEGS = {"en": "The route so far is drawn in the same two line styles as in (a). ",
           "zh": "已走过的路线用与 (a) 相同的两种线型。"}
_D = {
    "en": ("(d) One row per scored frame of the route, time running down (frame number on the left, counted from 1; "
           "K1–K4 on the right): the blue bar and its number = past positions whose true direction lies in the front "
           "view; the orange bar and its number under it = how many of them the predicted affordance map gets right "
           "(joint PCK@8). Rows after the robot reaches the {turn} (diamond) are shaded as the return leg; its totals "
           "are printed under the chart in the bars' colours."),
    "zh": ("(d) 路线上每个评分帧一行，时间自上而下（左侧为帧号，从 1 数起；右侧 K1–K4 为关键位置）：蓝条及其数字 = 真值方向"
           "落在前视的历史位置数；其下的橙条及其数字 = 其中预测 affordance map 判对的个数（joint PCK@8）。机器人到达{turn}"
           "（菱形）之后的各行加阴影表示回程，图下用条形的颜色写有回程合计。"),
}

# --------------------------------------------------------------------------- #
# Panel d geometry (points; the panel is W_ROUTE wide, under the map in column a)
# --------------------------------------------------------------------------- #
D_ROW_MIN_PT, D_ROW_MAX_PT = 10.0, 13.0  # pitch of one scored frame's row (two bars + their counts)
D_FS = 5.8  # counts, frame numbers, ticks, legend
D_TITLE_FS = 6.2
D_TOTAL_FS = 6.0
D_TOTAL_LH = 7.3  # line height of the totals
D_LH = 7.0  # line height of D_FS text
D_LEG_X = 9.5  # legend text after its swatch
D_LEG_GAP = 1.6  # between two legend entries
D_TITLE_LH = 7.4
D_TITLE_X = 9.4  # the title text starts after the bold "d"
D_TOP_GAP = 3.6
D_BAR = 0.34  # bar thickness / row pitch
D_BAR_GAP = 0.10  # gap between the two bars / row pitch
D_LABEL_PAD = 1.3  # count label after its bar end
D_K_FS = 5.6
FS = fc.FS
GT_BAR_COLOR = style.GT_COLOR  # past positions in the front view (ground truth)
PRED_BAR_COLOR = style.PRED_COLOR  # of these, predicted correctly (the prediction)
SHADE = style.GRID  # the return leg
SAME_SPOT_M = 0.05  # key positions this close share one position (the robot turning on the spot)

_CJK = "　-鿿＀-￯"
_TOKEN = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+|\s+")
_NO_BREAK_BEFORE = set("，。、：；）」』！？,.;:)")


def wrap_mixed(fig, text: str, width_pt: float, fs: float, **kw) -> List[str]:
    """Greedy word wrap for Latin and CJK text (breaks at spaces and between CJK characters, never before a
    closing punctuation mark); a single token wider than the line stays whole."""
    lines: List[str] = []
    for para in str(text).split("\n"):
        cur = ""
        for tok in _TOKEN.findall(para):
            trial = cur + tok
            fits = cd.text_width_pt(fig, trial.rstrip(), fs, **kw) <= width_pt
            if fits or not cur.strip() or tok.isspace() or tok in _NO_BREAK_BEFORE:
                cur = trial
            else:
                lines.append(cur.rstrip())
                cur = tok
        lines.append(cur.rstrip())
    return lines


def route_frame(xz: np.ndarray, split: int, aspect_hw: float, pad: float):
    """``common_draw.route_frame`` (kept here for callers of the first version)."""
    return cd.route_frame(xz, split, aspect_hw, pad)


def return_totals(tally: Sequence[dict], split: int) -> dict:
    back = [t for t in tally if t["t"] > split]
    out = [t for t in tally if t["t"] <= split]
    return {"return_rows": len(back), "return_front": sum(t["n_front"] for t in back),
            "return_hits": sum(t["hits_front"] for t in back),
            "outbound_rows": len(out), "outbound_front": sum(t["n_front"] for t in out),
            "outbound_hits": sum(t["hits_front"] for t in out)}


# --------------------------------------------------------------------------- #
# Panel d: past positions in the front view, per scored frame
# --------------------------------------------------------------------------- #
class FrontPanel:
    """Panel d, drawn by ``fig_case`` into the foot of column a (``CaseOptions.route_foot``).

    Everything is laid out in points inside one axes (x right, y down from the
    panel's top), so text, bars and badges keep their sizes whatever height
    column a leaves, top to bottom: title (wrapped), the three-entry legend
    (blue bar, orange bar, return-leg shading with its diamond), "frame"
    header, one row per scored frame (pitch ``D_ROW_MIN_PT``..``D_ROW_MAX_PT``;
    two bars side by side, each with its count printed after it), the 0/4/8
    axis, and the return-leg totals (a bold head line, then the blue count in
    ground-truth ink and the orange count in prediction ink).  ``min_h`` /
    ``max_h`` (inches) tell ``fig_case`` the heights it can use; ``warnings``
    collects what a reviewer should know (rows squeezed, K badges moved).
    """

    def __init__(self, fig_m, tally: Sequence[dict], split: int, key_frames: Sequence[int], LR: dict,
                 pattern: str, width_in: float = fc.W_ROUTE):
        self.tally = list(tally)
        self.split = int(split)
        self.key_frames = [int(f) for f in key_frames]
        self.LR = LR
        self.pattern = pattern
        self.W = width_in * 72.0
        self.warnings: List[str] = []
        tot = return_totals(self.tally, self.split)
        self.totals = tot
        if tot["return_front"]:
            parts = [(LR["d_total_head"], style.INK),
                     (LR["d_total_n"].format(n=tot["return_front"]), style.GT_INK),
                     (LR["d_total_h"].format(h=tot["return_hits"]), style.PRED_INK)]
        else:
            parts = [(LR["d_total_0"], style.INK)]
        self.total_lines = [(line, color) for text, color in parts
                            for line in wrap_mixed(fig_m, text, self.W, D_TOTAL_FS, fontweight="bold")]
        self.title_lines = wrap_mixed(fig_m, LR["d_title"], self.W - D_TITLE_X, D_TITLE_FS)
        self.legend = []
        for kind, text in (("front", LR["d_leg_front"]), ("hits", LR["d_leg_hits"]),
                           ("back", LR["d_leg_back"][pattern])):
            self.legend.append((kind, wrap_mixed(fig_m, text, self.W - D_LEG_X, D_FS)))
        frame_w = max(cd.text_width_pt(fig_m, str(t["t"] + 1), D_FS) for t in self.tally) if self.tally else 8.0
        self.x0 = frame_w + 3.2  # chart's left edge (frame numbers right-aligned before it)
        self.lane = cd.text_width_pt(fig_m, f"K{max(len(self.key_frames), 1)}", D_K_FS, fontweight="bold") + 4.2
        self.n_rows = max(len(self.tally), 1)
        self.min_h = (self._fixed_pt() + self.n_rows * D_ROW_MIN_PT) / 72.0
        self.max_h = (self._fixed_pt() + self.n_rows * D_ROW_MAX_PT) / 72.0
        self.layout: dict = {}

    # ---- geometry
    def _fixed_pt(self) -> float:
        legend_lines = sum(len(lines) for _, lines in self.legend)
        return (D_TOP_GAP + len(self.title_lines) * D_TITLE_LH + 2.4  # title
                + legend_lines * D_LH + (len(self.legend) - 1) * D_LEG_GAP + 3.4  # legend
                + D_LH + 1.0  # "frame" header
                + 2.4 + D_LH + D_LH + 0.6  # ticks, tick labels, axis label
                + 3.4 + len(self.total_lines) * D_TOTAL_LH + 1.0)  # totals

    # ---- drawing
    def __call__(self, page: fc.Page, x: float, y_top: float, w: float, h: float) -> None:
        fig = page.fig
        LR = self.LR
        W, H = w * 72.0, h * 72.0
        ax = page.ax(x, y_top, w, h)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # y runs down, like the page
        ax.axis("off")
        ax.patch.set_visible(False)
        pitch = (H - self._fixed_pt()) / self.n_rows
        if pitch < D_ROW_MIN_PT - 0.05:
            self.warnings.append(f"panel d rows squeezed to {pitch:.1f} pt (min {D_ROW_MIN_PT:g})")
        pitch = float(np.clip(pitch, 1.0, D_ROW_MAX_PT))

        # title
        y = D_TOP_GAP
        ax.text(0.0, y, "d", ha="left", va="top", fontsize=FS["title"] + 0.5, fontweight="bold", color=style.INK)
        for i, line in enumerate(self.title_lines):
            ax.text(D_TITLE_X, y + i * D_TITLE_LH, line, ha="left", va="top", fontsize=D_TITLE_FS, color=style.INK)
        y += len(self.title_lines) * D_TITLE_LH + 2.4
        # legend (before the chart: the reader meets the encoding first): blue, orange, the shading with the diamond
        for n_leg, (kind, lines) in enumerate(self.legend):
            ym = y + D_LH / 2
            if kind == "back":
                ax.add_patch(Rectangle((0.3, ym - 2.2), 7.0, 6.2, fc=SHADE, ec="none", alpha=0.9))
                cd.turn_marker(ax, 3.8, ym - 2.2, size=3.6, zorder=6)  # on the shading's top edge, as in the chart
            else:
                swatch = GT_BAR_COLOR if kind == "front" else PRED_BAR_COLOR
                ax.add_patch(Rectangle((0.6, ym - 1.6), 6.4, 3.2, fc=swatch, ec="none"))
            for i, line in enumerate(lines):
                ax.text(D_LEG_X, y + i * D_LH, line, ha="left", va="top", fontsize=D_FS, color=style.INK)
            y += len(lines) * D_LH + (D_LEG_GAP if n_leg < len(self.legend) - 1 else 0.0)
        y += 3.4
        # "frame" over the frame numbers
        ax.text(0.0, y, LR["d_frame"], ha="left", va="top", fontsize=D_FS, color=style.INK_2)
        y += D_LH + 1.0

        x0, lane = self.x0, self.lane
        cw = W - x0 - lane  # chart width
        unit = cw / 9.3  # 0..8 past positions plus room for the count after an 8
        rows_top = y
        rows_bot = rows_top + self.n_rows * pitch
        centres = [rows_top + (i + 0.5) * pitch for i in range(len(self.tally))]
        # return leg: shading from the boundary between the last outbound row and the first return row
        first_back = next((i for i, t in enumerate(self.tally) if t["t"] > self.split), None)
        if first_back is not None:
            yb = rows_top + first_back * pitch
            ax.add_patch(Rectangle((x0, yb), cw, rows_bot - yb, fc=SHADE, ec="none", alpha=0.75, zorder=0))
            cd.turn_marker(ax, x0 + cw - 3.0, yb, size=4.6, zorder=6)  # right end: clear of the counts
            self.layout["return_from_row"] = first_back
        bt, bg = D_BAR * pitch, D_BAR_GAP * pitch
        for t, yc in zip(self.tally, centres):
            ax.text(x0 - 2.2, yc, str(t["t"] + 1), ha="right", va="center", fontsize=D_FS, color=style.INK_2)
            n, hts = int(t["n_front"]), int(t["hits_front"])
            if n == 0:  # nothing in the front view: one muted 0, no bars
                ax.text(x0 + D_LABEL_PAD, yc, "0", ha="left", va="center", fontsize=D_FS, color=style.MUTED)
                continue
            y_blue = yc - bg / 2 - bt / 2
            y_or = yc + bg / 2 + bt / 2
            ax.add_patch(Rectangle((x0, y_blue - bt / 2), n * unit, bt, fc=GT_BAR_COLOR, ec="none", zorder=2))
            self._count(ax, fig, x0, n, unit, cw, y_blue, style.GT_INK)
            if hts:
                ax.add_patch(Rectangle((x0, y_or - bt / 2), hts * unit, bt, fc=PRED_BAR_COLOR, ec="none", zorder=2))
            self._count(ax, fig, x0, hts, unit, cw, y_or, style.PRED_INK)
        # axis: left spine over the rows, 0/4/8 at the bottom
        ax.plot([x0, x0], [rows_top, rows_bot], color=style.AXIS, lw=0.5, zorder=3, solid_capstyle="butt")
        ax.plot([x0, x0 + 8 * unit], [rows_bot, rows_bot], color=style.AXIS, lw=0.5, zorder=3,
                solid_capstyle="butt")
        for v in (0, 4, 8):
            xv = x0 + v * unit
            ax.plot([xv, xv], [rows_bot, rows_bot + 2.0], color=style.AXIS, lw=0.5, zorder=3)
            ax.text(xv, rows_bot + 2.4, str(v), ha="center", va="top", fontsize=D_FS, color=style.INK_2)
        y = rows_bot + 2.4 + D_LH
        ax.text(x0 + 4 * unit, y, LR["d_xlabel"], ha="center", va="top", fontsize=D_FS, color=style.INK_2)
        y += D_LH + 0.6 + 3.4
        # return-leg totals: head in ink, the blue count in ground-truth ink, the orange count in prediction ink
        for i, (line, color) in enumerate(self.total_lines):
            ax.text(0.0, y + i * D_TOTAL_LH, line, ha="left", va="top", fontsize=D_TOTAL_FS, color=color,
                    fontweight="bold", path_effects=cd.bold_effects(line, color, stroke_pt=0.3))
        y += len(self.total_lines) * D_TOTAL_LH + 1.0
        # K badges in the lane right of the chart, at their rows (moved apart with a leader only when needed)
        key_rows = []
        for n_k, fr in enumerate(self.key_frames):
            j = next((i for i, t in enumerate(self.tally) if t["t"] == fr), None)
            if j is None:
                self.warnings.append(f"panel d: K{n_k + 1} (frame {fr + 1}) is not a scored row")
                continue
            key_rows.append((n_k, j))
        if key_rows:
            bh = 8.4
            targets = np.array([centres[j] for _, j in key_rows])
            ys = cd.dodge_1d(targets, [bh] * len(key_rows), rows_top, rows_bot, 0.6)
            xb = x0 + cw + lane / 2 + 0.6
            for (n_k, j), yk, yt in zip(key_rows, ys, targets):
                if abs(yk - yt) > 0.5:
                    ax.plot([x0 + cw - 0.5, x0 + cw + 1.5, xb - lane / 2 + 1.0], [yt, yt, yk], color=style.INK_2,
                            lw=0.5, zorder=3, solid_joinstyle="round")
                    self.warnings.append(f"panel d: K{n_k + 1} badge moved {abs(yk - yt):.1f} pt off its row")
                cd.key_badge(ax, xb, yk, f"K{n_k + 1}", fs=D_K_FS)
        self.layout.update({"row_pitch_pt": round(pitch, 2), "rows": self.n_rows, "chart_w_pt": round(cw, 1),
                            "unit_pt": round(unit, 2), "rows_h_in": round(self.n_rows * pitch / 72.0, 3),
                            "panel_h_in": round(h, 3), "used_h_in": round(y / 72.0, 3)})
        if y > H + 0.5:
            self.warnings.append(f"panel d content {y:.1f} pt > panel {H:.1f} pt")

    @staticmethod
    def _count(ax, fig, x0: float, v: int, unit: float, cw: float, yc: float, ink) -> None:
        """The count of a bar, after its end; inside it (white) when the end is too close to the chart's edge."""
        s = str(v)
        tw = cd.text_width_pt(fig, s, D_FS)
        xe = x0 + v * unit
        if xe + D_LABEL_PAD + tw <= x0 + cw:
            ax.text(xe + D_LABEL_PAD, yc, s, ha="left", va="center", fontsize=D_FS, color=ink, zorder=4)
        else:
            ax.text(xe - D_LABEL_PAD, yc, s, ha="right", va="center", fontsize=D_FS, color="white", zorder=4)


# --------------------------------------------------------------------------- #
# Caption
# --------------------------------------------------------------------------- #
def _role_text(role: str, pattern: str, r: dd.CaseRow, start: np.ndarray, LR: dict) -> str:
    if role == "return_to_start":
        d = float(np.linalg.norm(r.cur_pos - start))
        return LR["back_at_start"] if d <= dd.BACK_AT_START_M else LR["from_start"].format(d=d)
    return LR["roles"][pattern].get(role, role.replace("_", " "))


def tier_front_back(metrics_json) -> Optional[dict]:
    """Tier E's joint PCK@8 of the deployed model for past positions in the front and in the back view
    (``metrics.json`` tiers.E.strata.gt_view), or None when the file or the fields are missing."""
    if not metrics_json or not Path(metrics_json).is_file():
        return None
    try:
        m = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
        E = m["tiers"]["E"]
        gv = E["strata"]["gt_view"]
        return {"clips": int(E["n_clips"]), "front": 100.0 * float(gv["front"][ARM]["joint_pck8"]),
                "n_front": int(gv["front"]["n"]), "back": 100.0 * float(gv["back"][ARM]["joint_pck8"]),
                "n_back": int(gv["back"]["n"])}
    except (KeyError, TypeError, ValueError):
        return None


def _key_list(keys: Sequence[str], lang: str) -> str:
    if lang == "zh":
        return "、".join(keys[:-1]) + " 与 " + keys[-1] if len(keys) > 1 else keys[0]
    return ", ".join(keys[:-1]) + " and " + keys[-1] if len(keys) > 1 else keys[0]


def same_spot_groups(recs: Sequence[dd.CaseRow], tol_m: float = SAME_SPOT_M) -> List[List[int]]:
    """Runs of consecutive key positions (indices) that lie within ``tol_m`` of each other (the robot turning on
    the spot), each run of length >= 2."""
    groups: List[List[int]] = []
    for i in range(1, len(recs)):
        if float(np.linalg.norm(recs[i].cur_pos - recs[i - 1].cur_pos)) <= tol_m:
            if groups and groups[-1][-1] == i - 1:
                groups[-1].append(i)
            else:
                groups.append([i - 1, i])
    return groups


def route_caption(lang: str, pattern: str, where: str, tot: dict, tier: Optional[dict] = None,
                  same_spot: Optional[Sequence[Sequence[str]]] = None) -> str:
    """Caption: claim + verdict, the return-leg result (panel d's totals), the tier-E front vs back numbers,
    the setup (with "K2 and K3 share one position" for key positions where the robot turns on the spot), then
    panels a-d.  "{elev_window}" is left for ``fig_case`` (D4)."""
    LR = LABELS[lang]
    R = _RESULT[lang]
    out = (R["out_0"] if tot["outbound_front"] == 0
           else R["out_n"].format(n=tot["outbound_front"], h=tot["outbound_hits"]))
    if tot["return_front"]:
        share = tot["return_hits"] / tot["return_front"]
        verdict = LR["verdict_wrong"] if share < 0.5 else (LR["verdict_right"] if share > 0.5 else LR["verdict_half"])
        claim = LR["claim"][pattern].format(verdict=verdict)
        result = R["main"].format(rows=tot["return_rows"], n=tot["return_front"], h=tot["return_hits"], out=out)
    else:
        claim = LR["claim_none"][pattern]
        result = R["none"].format(rows=tot["return_rows"], out=out)
    sep = " " if lang == "en" else ""
    text = claim + sep + result
    if tier is not None:
        text += R["tier"].format(**tier)
    text += _SETUP[lang][pattern].format(where=where, share=FRONT_SHARE_TRAINING)
    for keys in same_spot or []:
        text += LR["same_spot"].format(turn=LR["turn"][pattern], keys=_key_list(list(keys), lang))
    P = fc.CAPTION_PARTS[lang]
    text += P["b"] + _B_LEGS[lang] + P["c"] + fc.caption_marks(lang) + sep
    text += _D[lang].format(turn=LR["turn"][pattern])
    return text


# --------------------------------------------------------------------------- #
# One figure
# --------------------------------------------------------------------------- #
def make_route_figure(dump_npz_path, pattern: str, rows: Optional[Sequence[int]] = None,
                      roles: Optional[Sequence[str]] = None, split_frame: Optional[int] = None, topdown_root=None,
                      clip_root_override=None, out_stem="route", lang: str = "en", metrics_json=None) -> dict:
    """Render the route-pattern figure for one tier-E dump (anything else raises ``ValueError``).

    ``rows`` (dump row indices) with ``roles`` (select_cases role names) come
    from the cases.json pick; without them the same rule runs on the dump
    (``data.route_key_rows``).  ``split_frame``: where the route turns back
    (default ``data.route_split``).  ``metrics_json``: for the caption's
    tier-E front- vs back-view sentence (left out, with a warning, without it).
    Returns {"files", "pattern", "dump", "rows", "roles", "split_frame",
    "split_rule", "tally", "tally_totals", "tier_front_back", "same_spot",
    "stats", "size_in", "windows", "warnings", "notes_dropped", "layout"}.
    """
    if pattern not in dd.ROUTE_PATTERNS:
        raise ValueError(f"pattern {pattern!r} not in {dd.ROUTE_PATTERNS}")
    LR = LABELS[lang]
    dump = dd.load_dump(dump_npz_path)
    if dump.tier != "E":
        raise ValueError(f"{dump_npz_path}: tier {dump.tier!r}; the route figure is drawn only from a designed route "
                         f"(tier E)")
    if ARM not in dump.arms:
        raise ValueError(f"arm {ARM!r} not in dump arms {dump.arms}")
    split_info = dd.route_split(dump, pattern)
    if split_frame is None:
        split_frame = split_info["frame"]
    else:
        split_info = {"frame": int(split_frame), "rule": "given (cases.json turnaround_frame)"}
    split_frame = int(np.clip(split_frame, 0, dump.frame_count - 1))
    if rows is None:
        rows, roles, _ = dd.route_key_rows(dump, pattern, arm=ARM,
                                           turnaround_frame=split_frame if pattern == "out_and_back" else None)
    rows = [int(i) for i in rows]
    if roles is None:
        roles = [""] * len(rows)
    if len(roles) != len(rows):
        raise ValueError(f"roles {roles} do not match rows {rows}")
    recs = [dd.case_row(dump, i) for i in rows]
    start = dump.positions[0]
    role_txt = [_role_text(role, pattern, r, start, LR) if role else None for role, r in zip(roles, recs)]
    tally = dd.front_tally(dump, ARM)
    tot = return_totals(tally, split_frame)
    tier = tier_front_back(metrics_json)
    turn_keys = [n for n, role in enumerate(roles) if role == "turnaround"]
    same_spot = [[f"K{i + 1}" for i in g] for g in same_spot_groups(recs) if set(g) & set(turn_keys)]
    warnings: List[str] = []
    if tier is None:
        warnings.append(f"no tier-E front/back numbers (metrics.json {metrics_json}): caption sentence left out")

    short = dump.episode_id.replace(f"exp18E_{dump.scene}_", "")
    where = LR["where"].format(scene=dump.scene, short=short)
    scene_text = LR["scene_E"].format(scene=dump.scene, short=short, T=dump.frame_count)

    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    fig_m = plt.figure(figsize=(fc.FIG_W, 2.0))
    front = FrontPanel(fig_m, tally, split_frame, [r.frame for r in recs], LR, pattern)
    plt.close(fig_m)
    legend = [("out", LR["leg_out"]), ("turn", LR["turn"][pattern]), ("back", LR["leg_back"]),
              ("start", LR["start"])]
    opts = fc.CaseOptions(
        roles=role_txt, route_panel=fc.FittedRoutePanel(split=split_frame, legend=legend),
        route_foot_h=front.min_h, route_foot=front, split_frame=split_frame, letters="slide", scene_text=scene_text,
        caption=route_caption(lang, pattern, where, tot, tier, same_spot))
    res = fc.make_case_figure(dump_npz_path, rows=rows, topdown_root=topdown_root,
                              clip_root_override=clip_root_override, out_stem=out_stem, lang=lang, options=opts)
    if res["notes_dropped"]:  # cannot happen (fig_case raises on an unaccounted slot); never hide a note
        raise RuntimeError(f"notes left out of the route figure: {res['notes_dropped']}")
    if "foot_h" not in res["layout"]:
        raise RuntimeError("panel d was not drawn (fig_case left no room under the route panel)")
    if res["layout"]["foot_h"] < 1.2:
        warnings.append(f"panel d only {res['layout']['foot_h']:.2f} in tall")
    warnings += list(res.get("warnings", [])) + front.warnings
    stats = []
    for n, (r, role) in enumerate(zip(recs, roles)):
        fr = r.visible & (r.gt_class == 1)
        stats.append({"key": f"K{n + 1}", "role": role, "row": r.index, "frame": r.frame,
                      "prediction": r.summary(ARM), "always_behind": r.summary("floor"),
                      "misses": [k + 1 for k in r.misses(ARM)],
                      "front": {"n": int(fr.sum()), "hits": int((r.arms[ARM].joint8 & fr).sum())}})
    layout = dict(res["layout"])
    layout["panel_d"] = front.layout
    return {"files": res["files"], "pattern": pattern, "dump": str(dump.path), "rows": rows, "roles": list(roles),
            "split_frame": split_frame, "split_rule": split_info["rule"], "tally": tally, "tally_totals": tot,
            "tier_front_back": tier, "same_spot": same_spot, "stats": stats, "size_in": res["size_in"],
            "windows": res.get("windows"),
            "warnings": warnings, "notes_dropped": res["notes_dropped"], "layout": layout}


# --------------------------------------------------------------------------- #
# From cases.json
# --------------------------------------------------------------------------- #
def resolve_dump(pick: dict, dumps_root=None) -> Path:
    """The pick's dump: ``<dumps_root>/<tier>/<scene>/<clip>.npz`` when given and present, else ``npz_path``."""
    if dumps_root:
        cand = Path(dumps_root) / str(pick["tier"]) / str(pick["scene"]) / f"{pick['clip']}.npz"
        if cand.is_file():
            return cand
    path = Path(str(pick.get("npz_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"dump of {pick.get('clip_key')} not found (npz_path {path}, dumps_root {dumps_root})")
    return path


def _metrics_json_for(cases: dict, cases_json) -> Optional[Path]:
    """metrics.json of the run that wrote ``cases.json``: its ``metrics_dir``, else next to the file."""
    for d in (cases.get("metrics_dir"), Path(cases_json).parent):
        if d and (Path(d) / "metrics.json").is_file():
            return Path(d) / "metrics.json"
    return None


def make_route_figures(cases_json, dumps_root=None, topdown_root=None, clip_root_override=None, out_dir="routes",
                       lang: str = "en", metrics_json=None) -> dict:
    """One figure per route pattern from the pre-registered picks in ``cases.json`` (``pattern_figure``).

    Patterns without a pick (status not "ok", or no key rows) are skipped
    with the reason in "skipped" (nothing printed).  A pick that is not a
    tier-E dump raises ``ValueError``: the figure is drawn from designed routes
    only.  ``metrics_json`` defaults to the cases' ``metrics_dir``.  Returns
    {"files": [...], "figures": {pattern: result}, "skipped": {pattern:
    reason}, "warnings": [...]} (each result carries its own "warnings").
    """
    cases = json.loads(Path(cases_json).read_text(encoding="utf-8"))
    pf = cases.get("pattern_figure") or {}
    metrics_json = metrics_json or _metrics_json_for(cases, cases_json)
    out_dir = Path(out_dir)
    files: List[str] = []
    figures, skipped = {}, {}
    for pattern in dd.ROUTE_PATTERNS:
        pick = pf.get(pattern)
        if not isinstance(pick, dict) or pick.get("status") != "ok":
            status = pick.get("status") if isinstance(pick, dict) else None
            skipped[pattern] = (f"no pick in {cases_json} (status {status}; "
                                f"tiers present {cases.get('tiers_present')})")
            continue
        key_rows = pick.get("key_rows") or []
        if not key_rows:
            skipped[pattern] = "pick has no key_rows"
            continue
        if str(pick.get("tier")) != "E":
            raise ValueError(f"{pattern} pick {pick.get('clip_key')} is tier {pick.get('tier')!r}, not a designed "
                             f"route (tier E); the route figure is not drawn from other tiers")
        npz = resolve_dump(pick, dumps_root)
        dump = dd.load_dump(npz)
        if dump.tier != "E":
            raise ValueError(f"{npz}: the {pattern} pick's dump is tier {dump.tier!r}, not tier E")
        t_dump = dump.arrays["current_frame_ids"]
        for k in key_rows:  # the pick's rows must be the dump's rows
            if not 0 <= int(k["row"]) < dump.n_rows or int(t_dump[int(k["row"])]) != int(k["t"]):
                raise ValueError(f"{npz}: key row {k} does not match the dump (frame ids {t_dump.tolist()})")
        split = pick.get("turnaround_frame") if pattern == "out_and_back" else None
        stem = out_dir / (f"route_{pattern}" + ("" if lang == "en" else f"_{lang}"))
        res = make_route_figure(npz, pattern, rows=[int(k["row"]) for k in key_rows],
                                roles=[str(k["role"]) for k in key_rows],
                                split_frame=None if split is None else int(split), topdown_root=topdown_root,
                                clip_root_override=clip_root_override, out_stem=stem, lang=lang,
                                metrics_json=metrics_json)
        res["pick"] = {k: pick.get(k) for k in ("clip_key", "episode_id", "vo_pck8", "pattern_median_vo_pck8",
                                                "n_episodes", "turnaround_rule", "turnaround_frame",
                                                "turnaround_row_t")}
        figures[pattern] = res
        files += res["files"]
    return {"files": files, "figures": figures, "skipped": skipped,
            "warnings": [f"{p}: skipped, {why}" for p, why in skipped.items()]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", help="cases.json from select_cases.py (pattern_figure picks)")
    src.add_argument("--dump", help="one tier-E dump npz (a manual pick)")
    ap.add_argument("--pattern", choices=dd.ROUTE_PATTERNS, help="with --dump: the route pattern to lay out")
    ap.add_argument("--rows", default=None, help="with --dump: comma-separated rows (default: the pattern rule)")
    ap.add_argument("--roles", default=None, help="with --dump and --rows: comma-separated select_cases roles")
    ap.add_argument("--split-frame", type=int, default=None, help="with --dump: frame where the route turns back")
    ap.add_argument("--dumps-root", default=None, help="with --cases: <root>/<tier>/<scene>/<clip>.npz")
    ap.add_argument("--metrics-json", default=None, help="metrics.json (default: the cases' metrics_dir)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--out-dir", default="routes")
    ap.add_argument("--lang", default="en", help="en, zh or en,zh")
    args = ap.parse_args(argv)
    langs = [s for s in args.lang.split(",") if s]
    for lang in langs:
        if lang not in LABELS:
            ap.error(f"--lang {lang}: choose from {sorted(LABELS)}")
        if args.cases:
            res = make_route_figures(args.cases, dumps_root=args.dumps_root, topdown_root=args.topdown_root,
                                     clip_root_override=args.clip_root, out_dir=args.out_dir, lang=lang,
                                     metrics_json=args.metrics_json)
            for p, why in res["skipped"].items():
                print(f"[fig_routes] {p}: skipped, {why}")
            results = list(res["figures"].values())
        else:
            if not args.pattern:
                ap.error("--dump needs --pattern")
            rows = [int(x) for x in args.rows.split(",")] if args.rows else None
            roles = args.roles.split(",") if args.roles else None
            if rows is not None and roles is None:
                roles = dd.ROUTE_ROLES[:len(rows)] if len(rows) == 4 else None
            stem = Path(args.out_dir) / (f"route_{args.pattern}" + ("" if lang == "en" else f"_{lang}"))
            results = [make_route_figure(args.dump, args.pattern, rows=rows, roles=roles,
                                         split_frame=args.split_frame, topdown_root=args.topdown_root,
                                         clip_root_override=args.clip_root, out_stem=stem, lang=lang,
                                         metrics_json=args.metrics_json)]
        for r in results:
            for f in r["files"]:
                print(f)
            print(json.dumps({k: r[k] for k in ("pattern", "rows", "roles", "split_frame", "split_rule",
                                                "tally_totals", "size_in", "layout")}, default=str))
            for s in r["stats"]:
                print(json.dumps(s, default=float))
            for w in r["warnings"]:
                print("warning:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
