#!/usr/bin/env python3
"""EXP-18 route-pattern figures (tier E): what the predicted affordance map does when the past lies ahead.

One figure per designed route pattern, ``out_and_back`` and ``loop``, from the
pre-registered picks in ``cases.json`` (``select_cases.py``: the episode whose
per-episode PCK@8 of the deployed model is the pattern's median; key rows = the
scored rows just before, at and just after the turnaround, and the final row).
On R2R routes the past lies behind the robot; on the way back along a designed
route it moves into the FRONT view, which is rare in training.  The figure
shows whatever the prediction does there, hit or miss, and the caption states
the result (panel d's return-leg totals), not only the setup.

Layout = the case figure (``fig_case.make_case_figure``) in its revised layout
(``fig_case.CaseOptions.revised``: one place for the row names, numbered
misses in a lane under the prediction row with a line to their x, notes that
wrap instead of being dropped, carets for peaks beyond a row's +-8 deg, one
inset scale-bar corner, labelled reference numbers) plus:

* no title: the claim ("on the way back, the past lies ahead") is the
  caption's first sentence; a development stand-in (a dump that is not tier E)
  gets a one-line banner and a caption preamble, and ``make_route_figures``
  refuses to draw one unless ``allow_stand_in`` is set;
* a  Route (``fig_case.FittedRoutePanel``): cropped to the route, turned by a
  quarter turn when that shows it larger, only as tall as it needs; the
  route in two legs split where it turns back, outbound solid and return
  dashed, each on its own right-hand side, chevrons for the direction of
  travel, a diamond at the turn (a K position there shows as a dot inside
  it), an open circle at the start; a legend in the panel names the four;
  K badges fan out from the route with leaders that never cross;
* b  insets: the route so far in the same two leg styles; an F/B sector letter
  that badges displace slides along the rim within its sector or sits just
  inside it (return-leg rows crowd the front); R/L go outward;
* block headers name the key position's role ("after the turnaround");
* d  (under the map, taking the rest of column a): one row per scored frame,
  time running down like the blocks; bar length = past positions whose true
  direction lies in the front view (blue), its orange part = how many of them
  the predicted affordance map gets right (joint PCK@8); the return leg is
  shaded and carries its totals ("1 of 13 predicted correctly").

Figure policy (user decision, 2026-09-24; as ``fig_case``): the figure shows
the affordance map only and never mentions poses, odometry or the pose-source
split; the prediction is always the deployed model's (the dump's ``vo`` arm).
Ground truth blue, prediction orange; misses shown; only the front image is
marked as model input.

Where the route "turns back" (the leg split, the diamond and d's shading):
out_and_back -> the route's own turnaround frame (``turnaround_frame`` in
cases.json, else ``geometry.out_and_back_turnaround`` on the dump's
reference_path); loop, and any clip without a palindromic reference_path ->
the frame farthest (3D) from the first frame (``data.route_split``).

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_routes --cases <cases.json> [--dumps-root DIR] [--topdown-root DIR]
      [--clip-root DIR] [--lang en,zh] [--allow-stand-in] --out-dir DIR
  # development stand-in (no tier-E dump yet): any dump, rows by the same rule or explicit
  python -m scripts.exp18.figures.fig_routes --dump <clip.npz> --pattern out_and_back [--rows 10,11,12,15]
      [--lang en] --out-dir DIR
Writes route_<pattern>[_zh].{pdf,png} and _caption.txt per pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import fig_case as fc
from scripts.exp18.figures import style

from matplotlib.patches import Rectangle  # noqa: E402

ARM = fc.ARM  # the deployed model's prediction
FRONT_SHARE_TRAINING = "0.27 %"  # EXP-18 ledger: share of training labels in the front view

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "claim": {"out_and_back": "Out-and-back route: on the way back, the past lies ahead.",
                  "loop": "Loop route: heading back to the start, the past lies ahead."},
        "stand_in": "development stand-in: {tier} episode {ep}, not a designed route",
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
        "d_title": "Past positions in the front\nview, per scored frame",
        "d_frame": "frame",
        "d_xlabel": "past positions (of 8)",
        "d_leg_front": "in front view (ground truth)",
        "d_leg_hits": "of these, predicted correctly",
        "d_band": "return leg:",
        "d_band_n": "{h} of {n} predicted correctly",
        "d_band_0": "none in the front view",
        "scene_E": "unseen scene {scene}\nroute {short} · {T} frames",
    },
    "zh": {
        "claim": {"out_and_back": "去而复返：回程时，走过的位置到了前方。",
                  "loop": "绕圈：返回起点时，走过的位置到了前方。"},
        "stand_in": "开发替身：{tier}第 {ep} 集，并非设计路线",
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
        "d_title": "前视中的历史位置\n(逐评分帧)",
        "d_frame": "帧",
        "d_xlabel": "历史位置个数(共 8 个)",
        "d_leg_front": "落在前视中(真值)",
        "d_leg_hits": "其中预测正确的",
        "d_band": "回程：",
        "d_band_n": "{n} 个中预测对 {h} 个",
        "d_band_0": "前视中没有",
        "scene_E": "未见场景 {scene}\n路线 {short} · {T} 帧",
    },
}

_B_EN = (
    "(b) Map around each key position, heading up; its rim is the bearing ring that (c) unrolls clockwise from the "
    "front view's left edge (arrow); dashed radii are the view seams; the route so far is drawn in the same two leg "
    "styles. Blue lines run from the robot through each past position (dot; 1 = oldest of the 8 queried) to its "
    "number on the rim; where numbers take a sector letter's place, the letter moves outward (R, L) or along or just "
    "inside the rim within its own sector (F, B). ")
_B_ZH = (
    "(b) 各关键位置的局部地图，前方朝上；圆周就是 (c) 从前视左缘顺时针展开的方位环（箭头），虚线半径为视角分界；已走路线用同样的"
    "两种线型。蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号；扇区字母被编号占位时，左右两个外移，"
    "前后两个在本扇区内沿圆周挪开或移到圆周内侧。")
_D = {
    "en": ("(d) One row per scored frame of the route, time running down: bar length = past positions whose true "
           "direction lies in the front view (blue), orange part = how many of them the predicted affordance map gets "
           "right (joint PCK@8: right view and peak within 8 px); shading = return leg, with its totals."),
    "zh": ("(d) 路线上每个评分帧一行，时间自上而下：条长 = 真值方向落在前视的历史位置数（蓝），其中橙色部分 = 预测 affordance map "
           "判对的个数（joint PCK@8：视角正确且峰值偏差 ≤ 8 像素）；阴影为回程，并写有其合计。"),
}
_RESULT = {
    "en": ("Result: summed over the return leg's {rows} scored frames, {n} past positions lie in the front view and "
           "the predicted affordance map gets {h} of them right (joint PCK@8); on the outbound leg, {n_out} lie in the "
           "front view and {h_out} are right. "),
    "zh": ("结果：回程 {rows} 个评分帧合计有 {n} 个历史位置落在前视中，预测 affordance map 判对其中 {h} 个（joint PCK@8）；"
           "去程中落在前视的有 {n_out} 个，判对 {h_out} 个。"),
}
CAPTION = {
    "en": {
        "out_and_back": (
            "{claim} Predicted affordance maps on a designed out-and-back route ({where}): the robot follows an R2R "
            "reference path to its end and comes back along it. On the way back the past positions lie ahead of the "
            "robot, in the front view, which is rare in training ({share} of training labels). {result}Key positions "
            "(pre-registered): the scored frames just before, nearest to and just after the turnaround, and the last "
            "frame. (a) Route on the top-down map: outbound leg solid, return leg dashed, each drawn on its own "
            "right-hand side so the two stay apart where the return retraces the outbound path; chevrons give the "
            "direction of travel; diamond = turnaround (the end of the reference path), circle = start. "),
        "loop": (
            "{claim} Predicted affordance maps on a designed loop ({where}): the robot visits three waypoints on one "
            "floor and returns to its start. Heading back to the start, past positions come to lie ahead of the "
            "robot, in the front view, which is rare in training ({share} of training labels). {result}Key positions "
            "(pre-registered): the scored frames just before, at and just after the point farthest from the start, "
            "and the last frame. (a) Route on the top-down map: the leg up to the point farthest from the start "
            "(diamond) solid, the rest dashed, each drawn on its own right-hand side; chevrons give the direction of "
            "travel; circle = start. "),
    },
    "zh": {
        "out_and_back": (
            "{claim}设计的去而复返路线上的预测 affordance map（{where}）：机器人沿一条 R2R 参考路径走到头，再原路返回。回程时"
            "历史位置位于机器人前方、落在前视里，而这在训练中很少见（训练标签中占 {share}）。{result}关键位置（预注册）："
            "折返前、离折返最近、折返后的评分帧，以及末帧。(a) 俯视图上的路线：去程实线、回程虚线，各自画在行进方向的右侧，"
            "使原路返回的两段不重叠；箭头标出行进方向；菱形 = 折返点（参考路径的终点），圆圈 = 起点。"),
        "loop": (
            "{claim}设计的绕圈路线上的预测 affordance map（{where}）：机器人在同一层依次经过三个路点后回到起点。返回起点"
            "途中，历史位置来到机器人前方、落在前视里，而这在训练中很少见（训练标签中占 {share}）。{result}关键位置（预注册）："
            "离起点最远处之前、当时、之后的评分帧，以及末帧。(a) 俯视图上的路线：到最远点（菱形）为止实线，其后虚线，各自画在"
            "行进方向的右侧；箭头标出行进方向；圆圈 = 起点。"),
    },
}
STAND_IN_CAPTION = {
    "en": ("DEVELOPMENT STAND-IN, NOT FOR THE PAPER: {tier} {scene} episode {ep} is an R2R episode that turns back, "
           "used to build this layout before the tier-E dumps exist; the caption below describes the tier-E figure. "),
    "zh": "开发替身，不可用于论文：{tier} {scene} 第 {ep} 集是一条会往回走的 R2R 轨迹，只用于在 E 层导出之前搭建版式；下文描述的是 E 层图。",
}

# --------------------------------------------------------------------------- #
# Panel d geometry (inches): under the map in column a, taking the rest of it
# --------------------------------------------------------------------------- #
D_GAP = 0.05  # between the scene note under the map and the title of d
D_TITLE_H = 0.25
D_AXIS_H = 0.22
D_LEG_H = 0.25
D_LEFT = 0.17  # frame tick labels
D_RIGHT = 0.25  # K badges
D_ROW_MIN_PT, D_ROW_MAX_PT = 5.5, 12.0  # height of one scored frame's row
D_BAND_TEXT_PT = 16.0  # the return leg's totals, at the bottom of its shading
FS = fc.FS
BLUE = cd.GT_CMAP(cd.HEAT_TOP)
ORANGE = cd.PRED_CMAP(cd.HEAT_TOP)


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
def make_front_panel(tally: Sequence[dict], split: int, key_frames: Sequence[int], n_frames: int, LR: dict):
    """Panel d drawer for ``CaseOptions.route_foot`` (``min_h`` / ``max_h``: the heights it can use)."""
    n_rows = max(len(tally), 1)
    fixed_h = D_GAP + D_TITLE_H + D_AXIS_H + D_LEG_H

    def draw(page: fc.Page, x: float, y_top: float, w: float, h: float) -> None:
        fig = page.fig
        y_top = y_top + D_GAP
        page.text(x, y_top, "d", ha="left", va="top", fontsize=FS["title"] + 0.5, fontweight="bold",
                  color=style.INK)
        page.text(x + 0.13, y_top, LR["d_title"], ha="left", va="top", fontsize=FS["name"] - 0.2,
                  color=style.INK, linespacing=1.1)
        y_chart = y_top + D_TITLE_H
        chart_h = h - fixed_h
        cx, cw = x + D_LEFT, w - D_LEFT - D_RIGHT
        ax = page.ax(cx, y_chart, cw, chart_h)
        step = float(np.median(np.diff([t["t"] for t in tally]))) if len(tally) > 1 else 8.0
        top = -0.6 * step
        last = float(max([t["t"] for t in tally] + [n_frames - 1]))
        # frames per point, with D_BAND_TEXT_PT under the last row for the return leg's totals
        f_per_pt = (last + 0.6 * step - top) / max(chart_h * 72.0 - D_BAND_TEXT_PT, 10.0)
        bottom = last + 0.6 * step + D_BAND_TEXT_PT * f_per_pt
        ax.set_xlim(0, dd.K)
        ax.set_ylim(bottom, top)  # time runs down
        ax.set_autoscale_on(False)
        ax.set_facecolor("white")
        ax.axhspan(split, bottom, fc=style.GRID, ec="none", alpha=0.75, zorder=0)
        th = 0.78 * step
        for t in tally:
            y = t["t"]
            ax.plot([0, 0.14], [y, y], color=style.MUTED, lw=0.6, zorder=1, solid_capstyle="butt")  # a scored frame
            if t["n_front"]:
                ax.add_patch(Rectangle((0, y - th / 2), t["n_front"], th, fc=BLUE, ec="none", zorder=2))
            if t["hits_front"]:
                ax.add_patch(Rectangle((0, y - th / 2), t["hits_front"], th, fc=ORANGE, ec="none", zorder=3))
                ax.plot([t["hits_front"]] * 2, [y - th / 2, y + th / 2], color="white", lw=0.6, zorder=3.5)
        tot = return_totals(tally, split)
        per_pt_y = 1.0 / (chart_h * 72.0 / (bottom - top))
        y_txt = bottom - (D_BAND_TEXT_PT - 3.0) * per_pt_y
        band = LR["d_band_n"].format(h=tot["return_hits"], n=tot["return_front"]) if tot["return_front"] else \
            LR["d_band_0"]
        ax.text(0.15, y_txt - 6.2 * per_pt_y, LR["d_band"], ha="left", va="center", fontsize=FS["small"] - 0.2,
                color=style.INK_2, zorder=5)
        ax.text(0.15, y_txt, band, ha="left", va="center", fontsize=FS["small"] - 0.1, color=style.INK,
                fontweight="bold", zorder=5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(style.AXIS)
            ax.spines[s].set_linewidth(0.5)
        ticks = [v for v in range(0, int(last) + 1, 50)]
        ax.set_yticks(ticks)
        ax.tick_params(axis="y", labelsize=FS["small"] - 0.4, length=2.0, width=0.5, pad=1.2, colors=style.INK_2)
        ax.set_xticks([0, 4, 8])
        ax.tick_params(axis="x", labelsize=FS["small"] - 0.4, length=2.0, width=0.5, pad=1.2, colors=style.INK_2)
        ax.set_xlabel(LR["d_xlabel"], fontsize=FS["small"] - 0.4, color=style.INK_2, labelpad=1.0)
        page.text(cx - 0.02, y_chart - 0.015, LR["d_frame"], ha="right", va="bottom", fontsize=FS["small"] - 0.4,
                  color=style.INK_2)
        # K badges right of the chart, dodged down the rows, a leader to their row
        lane = page.ax(cx + cw, y_chart, D_RIGHT, chart_h)
        lane.set_xlim(0, D_RIGHT * 72.0)
        lane.set_ylim(bottom, top)
        lane.axis("off")
        pt_per_frame = chart_h * 72.0 / (bottom - top)
        labels = [f"K{n + 1}" for n in range(len(key_frames))]
        hs = [9.6 / pt_per_frame] * len(labels)
        ys = cd.dodge_1d(np.asarray(key_frames, dtype=float), hs, top, bottom, 0.8 / pt_per_frame)
        for s, yk, t in zip(labels, ys, key_frames):
            bw = cd.text_width_pt(fig, s, 5.6, fontweight="bold") + 2.6
            xb = 5.0 + bw / 2
            lane.plot([0.0, 2.0, xb - bw / 2], [t, t, yk], color=style.INK_2, lw=0.5, zorder=3, clip_on=False,
                      solid_joinstyle="round")
            cd.key_badge(lane, xb, yk, s, fs=5.6)
        # legend: the two colours
        y_leg = y_chart + chart_h + D_AXIS_H
        for j, (col, text) in enumerate(((BLUE, LR["d_leg_front"]), (ORANGE, LR["d_leg_hits"]))):
            yy = y_leg + 0.105 * j + 0.05
            sw = page.ax(x + 0.01, yy - 0.032, 0.064, 0.064)
            sw.set_xlim(0, 1)
            sw.set_ylim(0, 1)
            sw.axis("off")
            sw.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, fc=col, ec="none"))
            page.text(x + 0.1, yy, text, ha="left", va="center", fontsize=FS["small"] - 0.2, color=style.INK)

    draw.min_h = fixed_h + (n_rows * D_ROW_MIN_PT + D_BAND_TEXT_PT) / 72.0
    draw.max_h = fixed_h + (n_rows * D_ROW_MAX_PT + D_BAND_TEXT_PT) / 72.0
    return draw


# --------------------------------------------------------------------------- #
# One figure
# --------------------------------------------------------------------------- #
def _role_text(role: str, pattern: str, r: dd.CaseRow, start: np.ndarray, LR: dict) -> str:
    if role == "return_to_start":
        d = float(np.linalg.norm(r.cur_pos - start))
        return LR["back_at_start"] if d <= dd.BACK_AT_START_M else LR["from_start"].format(d=d)
    return LR["roles"][pattern].get(role, role.replace("_", " "))


def route_caption(lang: str, pattern: str, opts: fc.CaseOptions, where: str, tot: dict) -> str:
    """Caption: claim, setup, the return-leg result (panel d's totals), then panels a-d."""
    LR = LABELS[lang]
    result = _RESULT[lang].format(rows=tot["return_rows"], n=tot["return_front"], h=tot["return_hits"],
                                  n_out=tot["outbound_front"], h_out=tot["outbound_hits"])
    head = CAPTION[lang][pattern].format(claim=LR["claim"][pattern], where=where, share=FRONT_SHARE_TRAINING,
                                         result=result)
    c = fc.CAPTION_PARTS[lang]["c"]
    return head + (_B_EN if lang == "en" else _B_ZH) + c + fc.caption_marks(lang, opts) + " " + _D[lang]


def make_route_figure(dump_npz_path, pattern: str, rows: Optional[Sequence[int]] = None,
                      roles: Optional[Sequence[str]] = None, split_frame: Optional[int] = None, topdown_root=None,
                      clip_root_override=None, out_stem="route", lang: str = "en") -> dict:
    """Render the route-pattern figure for one dump.

    ``rows`` (dump row indices) with ``roles`` (select_cases role names) come
    from the cases.json pick; without them the same rule runs on the dump
    (``data.route_key_rows``).  ``split_frame``: where the route turns back
    (default ``data.route_split``).  A dump that is not tier E is drawn as a
    development stand-in (banner + caption preamble).  Returns {"files",
    "rows", "roles", "split_frame", "stand_in", "tally_totals", "stats",
    "size_in", "notes_dropped", "layout"}.
    """
    if pattern not in dd.ROUTE_PATTERNS:
        raise ValueError(f"pattern {pattern!r} not in {dd.ROUTE_PATTERNS}")
    LR = LABELS[lang]
    dump = dd.load_dump(dump_npz_path)
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
    stand_in = dump.tier != "E"

    tier_name = dump.tier_name(lang)
    if stand_in:
        where = (f"{tier_name} {dump.scene}, episode {dump.episode_id}" if lang == "en"
                 else f"{tier_name} {dump.scene}，第 {dump.episode_id} 集")
        scene_text = None  # the case figure's default: tier, scene, episode
        banner = LR["stand_in"].format(tier=tier_name, ep=dump.episode_id)
    else:
        short = dump.episode_id.replace(f"exp18E_{dump.scene}_", "")
        where = (f"unseen scene {dump.scene}, route {short}" if lang == "en" else f"未见场景 {dump.scene}，路线 {short}")
        scene_text = LR["scene_E"].format(scene=dump.scene, short=short, T=dump.frame_count)
        banner = None

    legend = [("out", LR["leg_out"]), ("turn", LR["turn"][pattern]), ("back", LR["leg_back"]),
              ("start", LR["start"])]
    front = make_front_panel(tally, split_frame, [r.frame for r in recs], dump.frame_count, LR)
    opts = fc.CaseOptions.revised(
        roles=role_txt, banner=banner, route_panel=fc.FittedRoutePanel(split=split_frame, legend=legend),
        route_foot_h=front.min_h, route_foot=front, split_frame=split_frame, letters="slide", scene_text=scene_text)
    caption = route_caption(lang, pattern, opts, where, tot)
    if stand_in:
        caption = STAND_IN_CAPTION[lang].format(tier=tier_name, scene=dump.scene, ep=dump.episode_id) + caption
    opts.caption = caption
    res = fc.make_case_figure(dump_npz_path, rows=rows, topdown_root=topdown_root,
                              clip_root_override=clip_root_override, out_stem=out_stem, lang=lang, options=opts)
    if res["notes_dropped"]:  # cannot happen with wrap_notes; a figure must not hide a note silently
        raise RuntimeError(f"notes left out of the route figure: {res['notes_dropped']}")
    stats = []
    for n, (r, role) in enumerate(zip(recs, roles)):
        fr = r.visible & (r.gt_class == 1)
        stats.append({"key": f"K{n + 1}", "role": role, "row": r.index, "frame": r.frame,
                      "prediction": r.summary(ARM), "always_behind": r.summary("floor"),
                      "front": {"n": int(fr.sum()), "hits": int((r.arms[ARM].joint8 & fr).sum())}})
    return {"files": res["files"], "pattern": pattern, "dump": str(dump.path), "rows": rows, "roles": list(roles),
            "split_frame": split_frame, "split_rule": split_info["rule"], "stand_in": stand_in,
            "tally_totals": tot, "stats": stats, "size_in": res["size_in"], "notes_dropped": res["notes_dropped"],
            "layout": res["layout"]}


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


def make_route_figures(cases_json, dumps_root=None, topdown_root=None, clip_root_override=None, out_dir="routes",
                       lang: str = "en", allow_stand_in: bool = False) -> dict:
    """One figure per route pattern from the pre-registered picks in ``cases.json`` (``pattern_figure``).

    Patterns without a pick (tier E not dumped yet, or no scored episode) are
    skipped with the reason, not an error.  A pick whose dump is not tier E
    (a development stand-in) is skipped too unless ``allow_stand_in``: the
    paper figure must come from a designed route.  Returns {"files": [...],
    "figures": {pattern: result}, "skipped": {pattern: reason}}.
    """
    cases = json.loads(Path(cases_json).read_text(encoding="utf-8"))
    pf = cases.get("pattern_figure") or {}
    out_dir = Path(out_dir)
    files: List[str] = []
    figures, skipped = {}, {}
    for pattern in dd.ROUTE_PATTERNS:
        pick = pf.get(pattern)
        if not isinstance(pick, dict) or pick.get("status") != "ok":
            status = pick.get("status") if isinstance(pick, dict) else None
            skipped[pattern] = (f"no pick in {cases_json} (status {status}; "
                                f"tiers present {cases.get('tiers_present')})")
            print(f"[fig_routes] {pattern}: skipped, {skipped[pattern]}")
            continue
        key_rows = pick.get("key_rows") or []
        if not key_rows:
            skipped[pattern] = "pick has no key_rows"
            continue
        npz = resolve_dump(pick, dumps_root)
        dump = dd.load_dump(npz)
        if dump.tier != "E" and not allow_stand_in:
            skipped[pattern] = (f"pick {pick.get('clip_key')} is a tier-{dump.tier} dump, not a designed route "
                                f"(tier E); a development stand-in is drawn only with allow_stand_in")
            print(f"[fig_routes] {pattern}: skipped, {skipped[pattern]}")
            continue
        t_dump = dump.arrays["current_frame_ids"]
        for k in key_rows:  # the pick's rows must be the dump's rows
            if not 0 <= int(k["row"]) < dump.n_rows or int(t_dump[int(k["row"])]) != int(k["t"]):
                raise ValueError(f"{npz}: key row {k} does not match the dump (frame ids {t_dump.tolist()})")
        split = pick.get("turnaround_frame") if pattern == "out_and_back" else None
        stem = out_dir / (f"route_{pattern}" + ("" if lang == "en" else f"_{lang}"))
        res = make_route_figure(npz, pattern, rows=[int(k["row"]) for k in key_rows],
                                roles=[str(k["role"]) for k in key_rows],
                                split_frame=None if split is None else int(split), topdown_root=topdown_root,
                                clip_root_override=clip_root_override, out_stem=stem, lang=lang)
        res["pick"] = {k: pick.get(k) for k in ("clip_key", "episode_id", "vo_pck8", "pattern_median_vo_pck8",
                                                "n_episodes", "turnaround_rule", "turnaround_frame",
                                                "turnaround_row_t")}
        figures[pattern] = res
        files += res["files"]
    return {"files": files, "figures": figures, "skipped": skipped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", help="cases.json from select_cases.py (pattern_figure picks)")
    src.add_argument("--dump", help="one dump npz (development stand-in or a manual pick)")
    ap.add_argument("--pattern", choices=dd.ROUTE_PATTERNS, help="with --dump: the route pattern to lay out")
    ap.add_argument("--rows", default=None, help="with --dump: comma-separated rows (default: the pattern rule)")
    ap.add_argument("--roles", default=None, help="with --dump and --rows: comma-separated select_cases roles")
    ap.add_argument("--split-frame", type=int, default=None, help="with --dump: frame where the route turns back")
    ap.add_argument("--dumps-root", default=None, help="with --cases: <root>/<tier>/<scene>/<clip>.npz")
    ap.add_argument("--allow-stand-in", action="store_true",
                    help="with --cases: also draw picks whose dump is not tier E (development only)")
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
                                     allow_stand_in=args.allow_stand_in)
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
                                         clip_root_override=args.clip_root, out_stem=stem, lang=lang)]
        for r in results:
            for f in r["files"]:
                print(f)
            print(json.dumps({k: r[k] for k in ("pattern", "rows", "roles", "split_frame", "split_rule", "stand_in",
                                                "tally_totals", "size_in", "layout")}, default=str))
            for s in r["stats"]:
                print(json.dumps(s, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
