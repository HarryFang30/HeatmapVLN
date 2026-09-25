#!/usr/bin/env python3
"""EXP-19 behaviour figures v2 (paper-grade): episode pages and the two overview figures.

v1 (``fig_behavior.py``) stays as it is; v2 draws the same cases and key
moments with a cleaner layout, small crisp markers and one new element, the
ONLINE timeline: how the affordance maps evolve over the whole rerun
(``timeline_panel.py``, data from ``records_v2/<ep_key>_timeline.*``).

Episode page (7.0 in wide, <= 6.8 in tall)::

  header: category · instruction (italic, wrapped) · rerun outcome (no bookkeeping such as the candidate rank:
          that is in the manifest)
  a  route map (where the rerun ended = small square, unless the goal star covers it; a route that would hide
     under the start ring gets "route stays within d m of the start" under the map)
  b  online timeline: x = step (the warm-up, and on sparse reruns the long no-map spans, compressed into blocks
     with axis breaks; a compressed axis names its last step), y = bearing with up = left on every panel: the
     history panel runs ahead / left / behind (middle) / right / ahead from top to bottom, the future sub-panel is
     centred on ahead; K1-K4 = hairlines at their calls' steps, badges raised one row where two would touch;
     history affordance map per ready call with each slot's peak and true direction spread across the column
     (slot 1 = oldest at the left; frames 1, 4, 8 only where a column is too narrow for eight; stacked at the
     centre on long reruns, the legend's timeline entry naming the stride -- nothing is written over the data);
     System 1 path endpoint · executed turns
  c  K1 | K2 | K3 | K4:  step · decision image (pixel goal, System1 path) · System 2 output ("↓" glossed as
                         "look down") · history frames 2x4 with their steps · 360° history strip (every mark at its
                         own bearing and elevation, a hairline joining a frame's peak to its true direction) ·
                         360° future strip (both 8 deg wrapped past +-180) · executed actions
  legend: a fixed grid of three grouped columns (history | future and decisions | route and timeline; the entry
  saying what a timeline column marks moves to the third column when it needs two lines and that saves a row) ·
  one-line data-flow note (the bearing conventions are in the caption)

Heat fields are drawn display-smoothed (strips ``panels_v2.SMOOTH_DEG``, timeline
``timeline_panel.TL_SMOOTH_DEG``; peaks kept; the sigmas are in the manifest and
every caption says so, ``smooth_sentence``); wrapped header lines keep their
rendered ink (italic overhang included) 2 pt inside the page (``wrap_ink``).

Overview figures (7.0 in wide, <= 8.0 in): one block per is_main case
(successes T1-T3 in ``main_T``, failures F1-F2 in ``main_F``), each block
route + online timeline (titled; past frames 1, 4, 8 marked per column; a long
rerun's marks stacked, with its own legend row) + K2 + K3 with the same
geometry, so blocks align; the key columns carry their badge and step in the
image's corner, and the first block names the frames / history strip / future
strip rows.

Policy (unchanged from v1 and the ledger): no poses, VO, odometry, relative
poses or heading arrows; both heatmaps are "affordance map"; only the framed
front +-39.5 deg of a 360° strip was given to the model (the rest is muted,
display only); the future affordance map and the System1 path are both decoded
from Z̃ and nothing is drawn from the future map to the actions; caption claims
come only from ``fig_behavior.claim_sentences`` (metrics.json verdicts).

Usage (repo root on PYTHONPATH)::

  python -m scripts.exp19.figures.fig_v2 --records <EXP>/records --timelines <EXP>/records_v2 \\
      --metrics <EXP>/metrics/metrics.json --out-dir <EXP>/figures_v2 [--lang en zh] [--pages] [--overview]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import fig_behavior as fb
from scripts.exp19.figures import panels as pn
from scripts.exp19.figures import panels_v2 as p2
from scripts.exp19.figures import timeline_panel as tp

import matplotlib  # noqa: E402  (configured in fb.setup())
from matplotlib.text import Text  # noqa: E402

MANIFEST_SCHEMA = "exp19-figures-v2-manifest-v1"
FIG_W = style.WIDTH_DOUBLE  # 7.0
PAGE_MAX_H_IN = 6.8
OVERVIEW_MAX_H_IN = 8.0
FS = p2.FS
LINE = 0.105  # inches per 6-6.3 pt text line

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
ZT = r"$\tilde{\mathrm{Z}}$"  # Z~ (mathtext in the body font, see setup())
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "title": "{cat} · {name}",
        "ids": "rerun · scene {scene} · episode {ep}",
        "instruction": "“{text}”",
        "outcome_success": "Rerun: success, stopped at step {steps}, {ne} m from the goal",
        "outcome_stop": "Rerun: failure, stopped at step {steps}, {ne} m from the goal{os}",
        "outcome_cap": "Rerun: failure, hit the {steps}-step limit {ne} m from the goal{os}",
        "outcome_other": "Rerun: failure, ended without STOP at step {steps}, {ne} m from the goal{os}",
        "outcome_os": " (it had been within the success radius earlier)",
        "outcome_missing": "Rerun outcome not recorded",
        "ne_missing": "?",
        "panel_a": "Route of the rerun",
        "panel_b": "Online: the history affordance map predicted at each System 2 call",
        "panel_b_hist": "Predicted history affordance map (orange) and true directions (blue)",
        "panel_b_fut": "Predicted future affordance map and System 1 path endpoint",
        "panel_c": "Key moments",
        "x_steps": "step",
        "y_hist": ("ahead", "right", "back", "left", "ahead"),  # bottom to top (y 0, 90, 180, 270, 360): up = left
        "y_fut": ("left", "ahead", "right"),
        "warmup": "warm-up",
        "warmup_range": "steps {a}–{b}",
        "step": "step {s}",
        "s2": "System 2: {texts}",
        "s2_look": " (look down)",
        "goal_suffix": " (pixel goal)",
        "then": " → ",
        "executed": "Executed:",
        "lookdown": "look-down image",
        "front": "front image",
        "lookdown_short": "look-down",
        "front_short": "front",
        "cap_frames": "Past frames to System 2 (1 = oldest; corner = step)",
        "cap_hist": "360°: predicted history affordance map",
        "cap_fut": "360°: predicted future affordance map",
        "axis": ("back", "left", "ahead", "right", "back"),
        "start": "start",
        "radius": "3 m",
        "no_map": "no top-down map for this scene",
        "floor_note": "Route changes floor; the map shows the floor of most steps, dotted = on another floor.",
        "floor_note_nomap": "Route changes floor (height range over 1 m); all steps are drawn on one plate.",
        "no_keys": ("This rerun has no ready System 2 call, so there is no key moment and no affordance map to "
                    "show."),
        "synthetic": "SYNTHETIC: layout test, not a result",
        "synthetic_tl": "SYNTHETIC TIMELINE: layout test, not a result",
        "void": "VOID BATCH, not evidence (a pre-registered validity condition failed): {reasons}",
        "predicate_fails": "The rerun no longer meets the {cat} definition; shown anyway, as pre-registered.",
        "legend": {
            "hist": "predicted history affordance map",
            "pred": "predicted peak (predicted visible, p ≥ 0.5)",
            "gt": "true direction of a past frame visible from here",
            "pair": "strip: a frame's peak joined to its true direction",
            "fut": "predicted future affordance map (darker = later)",
            "path": "System 1 path",
            "goal": "high-level decision (pixel goal)",
            "actions": "executed actions",
            "turns": "executed turns (up = left, down = right)",
            "route": "route / reference path",
            "start_goal": "start / goal (3 m radius)",
            "stop": "where the rerun ended",
            "warmup": "warm-up (no affordance map yet)",
            "nomap": "System 2 gave turns or STOP (no affordance map)",
            "frame": "framed: the model's 79° view; muted: not given to the model",
            "slots": "timeline: a call's past frames 1→8 left→right (not time)",
            "slots_some": "timeline: a call's past frames 1→8 left→right (not time); narrow columns: frames 1, 4, 8 only",
            "slots_ov": "timeline: a call's past frames 1, 4, 8 left→right (not time)",
            "slots_stacked": "timeline: a call's 8 past frames stacked mid-column",
            "slots_stacked_ov": "{cases} timeline: a call's 8 past frames stacked mid-column",
            "frames": "past frames given to System 2 (1 = oldest)",
            "key": "K1–K4: pre-registered rule, not time order",
        },
        "legend_groups": ("History affordance map", "Future affordance map and decisions", "Route and timeline"),
        "stride_tail": "; one call in {k} (and every key moment)",
        "flow_note": ["Data flow: the future affordance map and the System 1 path are both decoded from ", ZT,
                      "; the future map does not feed the actions."],
        "turns": "turns",
        "col_route": "Route",
        "col_timeline": "Online timeline",
        "col_key": "Key moment {k}",
        "row_labels": ("past frames", "360° history", "360° future"),
        "route_note": "route stays within {d:.1f} m of the start",
    },
    "zh": {
        "title": "{cat} · {name}",
        "ids": "复跑 · 场景 {scene} · 第 {ep} 集",
        "instruction": "“{text}”",
        "outcome_success": "复跑：成功，第 {steps} 步停下，距目标 {ne} m",
        "outcome_stop": "复跑：失败，第 {steps} 步停下，距目标 {ne} m{os}",
        "outcome_cap": "复跑：失败，撞上 {steps} 步上限，距目标 {ne} m{os}",
        "outcome_other": "复跑：失败，第 {steps} 步未经 STOP 结束，距目标 {ne} m{os}",
        "outcome_os": "（此前曾进入成功半径）",
        "outcome_missing": "复跑结局未记录",
        "ne_missing": "?",
        "panel_a": "复跑路线",
        "panel_b": "在线运行：每次慢系统调用时预测的历史 affordance map",
        "panel_b_hist": "预测历史 affordance map（橙）与真实来路方向（蓝圈）",
        "panel_b_fut": "预测未来 affordance map 与快系统路径终点",
        "panel_c": "关键时刻",
        "x_steps": "步",
        "y_hist": ("前", "右", "后", "左", "前"),  # 自下而上（y 0、90、180、270、360）：向上 = 向左
        "y_fut": ("左", "前", "右"),
        "warmup": "预热期",
        "warmup_range": "第 {a}–{b} 步",
        "step": "第 {s} 步",
        "s2": "慢系统：{texts}",
        "s2_look": "（先俯视）",
        "goal_suffix": "（像素目标）",
        "then": " → ",
        "executed": "执行：",
        "lookdown": "俯视帧",
        "front": "前视帧",
        "lookdown_short": "俯视帧",
        "front_short": "前视帧",
        "cap_frames": "送入慢系统的历史帧（1 = 最早；角标 = 该帧的步号）",
        "cap_hist": "360° 预测历史 affordance map",
        "cap_fut": "360° 预测未来 affordance map",
        "axis": ("后", "左", "前", "右", "后"),
        "start": "起点",
        "radius": "3 m",
        "no_map": "该场景没有俯视图",
        "floor_note": "路线跨楼层；俯视图为多数步所在楼层，点线 = 在另一层。",
        "floor_note_nomap": "路线跨楼层（高差超过 1 m）；所有步画在同一底板上。",
        "no_keys": "本次复跑没有就绪调用，因此没有关键时刻，也没有 affordance map 可画。",
        "synthetic": "合成数据：仅用于排版测试，不是结果",
        "synthetic_tl": "合成时间线：仅用于排版测试，不是结果",
        "void": "整批作废，不是证据（预注册的有效性条件未满足）：{reasons}",
        "predicate_fails": "复跑已不满足 {cat} 的定义；按预注册照常出图。",
        "legend": {
            "hist": "预测历史 affordance map",
            "pred": "预测峰值（模型认为该帧可见，p ≥ 0.5）",
            "gt": "真实来路方向（仅画此处可见的历史帧）",
            "pair": "条带上连线：同一历史帧的预测峰值与真实方向",
            "fut": "预测未来 affordance map（越深越晚）",
            "path": "快系统路径",
            "goal": "高层决策（像素目标）",
            "actions": "执行的动作",
            "turns": "执行的转向（上 = 左转，下 = 右转）",
            "route": "执行路线 / 参考路径",
            "start_goal": "起点 / 目标（3 m 成功半径）",
            "stop": "复跑结束处",
            "warmup": "预热期（尚无 affordance map）",
            "nomap": "慢系统直接给出转向或停止（无 affordance map）",
            "frame": "黑框 = 输入模型的 79° 视野；压灰部分未输入模型",
            "slots": "时间线：同一次调用的历史帧 1→8 从左到右（非时间）",
            "slots_some": "时间线：同一次调用的历史帧 1→8 从左到右（非时间）；窄列只画 1、4、8",
            "slots_ov": "时间线：同一次调用的历史帧 1、4、8 从左到右（非时间）",
            "slots_stacked": "时间线：一次调用的 8 个历史帧叠在列中央",
            "slots_stacked_ov": "{cases} 的时间线：一次调用的 8 个历史帧叠在列中央",
            "frames": "送入慢系统的历史帧（1 = 最早）",
            "key": "关键时刻 K1–K4 按预注册规则编号，不按时间先后",
        },
        "legend_groups": ("历史 affordance map", "未来 affordance map 与决策", "路线与时间线"),
        "stride_tail": "；每 {k} 次调用画一次（关键时刻都画）",
        "flow_note": ["数据流：预测未来 affordance map 与快系统路径都由", ZT, "解码；未来图不回流到动作。"],
        "turns": "转向",
        "col_route": "复跑路线",
        "col_timeline": "在线运行",
        "col_key": "关键时刻 {k}",
        "row_labels": ("历史帧", "360° 历史", "360° 未来"),
        "route_note": "路线始终在起点 {d:.1f} m 以内",
    },
}
SLOT_KEYS = ("slots", "slots_some", "slots_ov", "slots_stacked")  # one of them per figure (``slot_key``)
LEGEND_GROUPS = (("hist", "pred", "gt", "pair", "slots", "frame"),  # the history affordance map and its marks
                 ("fut", "path", "goal", "actions", "turns", "key"),  # the future map and the decisions
                 ("warmup", "nomap", "route", "start_goal", "stop", "frames"))  # timeline and route-map context
LEGEND_KEYS = tuple(k for group in LEGEND_GROUPS for k in group) + SLOT_KEYS[1:]
SLOT_KEY = {"all": "slots", "some": "slots_some", "148": "slots_ov", "stacked": "slots_stacked"}  # MarkPlan.mode
SLOT_EXTRA = ("slots_stacked_ov",)  # the overview's second slot entry: its stacked rows (after the slot entry)


def legend_labels(L: dict, texts: Optional[Dict[str, str]] = None) -> dict:
    """``L`` with some legend texts replaced (``texts``: key -> text), for one figure."""
    return dict(L, legend={**L["legend"], **(texts or {})}) if texts else L


def slot_legend_text(L: dict, mode: str, stride: int) -> str:
    """The legend's timeline entry for a mark plan (``tp.MarkPlan`` mode / stride): what a column marks and, on a
    stacked timeline thinned to one call in k, that stride (in the legend, not over the data)."""
    text = L["legend"][SLOT_KEY[mode]]
    return text + L["stride_tail"].format(k=stride) if mode == "stacked" and stride > 1 else text


def stacked_overview_text(L: dict, lang: str, stacked: Sequence[Tuple[str, int]]) -> str:
    """The overview's extra legend row for its stacked timelines (``stacked``: (category, stride) per stacked
    row): which rows, what they mark and the stride."""
    join = " and " if lang == "en" else "、"
    text = L["legend"]["slots_stacked_ov"].format(cases=join.join(c for c, _ in stacked))
    ks = [k for _, k in stacked]
    if any(k > 1 for k in ks):
        k = ks[0] if len(set(ks)) == 1 else (", " if lang == "en" else "、").join(
            f"{k_} ({c})" if lang == "en" else f"{k_}（{c}）" for c, k_ in stacked)
        text += L["stride_tail"].format(k=k)
    return text


SMOOTH_NOTE = {
    "en": ("Heat fields are drawn with a Gaussian blur (σ = {s:g}° on the 360° strips, {t:g}° in bearing on the "
           "timelines) for display; dots mark the raw peaks."),
    "zh": "热力场为显示做了高斯平滑（环视条带 σ = {s:g}°，时间线方位向 σ = {t:g}°）；圆点为未平滑的峰值。",
}


def smooth_sentence(lang: str) -> str:
    """The captions' (and the animation note's) disclosure of the display-only blur of the heat fields."""
    return SMOOTH_NOTE[lang].format(s=p2.SMOOTH_DEG, t=tp.TL_SMOOTH_DEG)


CAPTION_PAGE = {
    "en": (
        "Closed-loop rerun of R2R val_unseen episode {ep} in scene {scene}, category {cat}: {cat_name}. Marks as in the "
        "legend. "
        "(a) Route, with the key moments K1–K{nk}.{route} (b) The affordance maps online: x = step; y = bearing around "
        "the robot (0° = ahead, left positive, ±180° = behind), up = left on every panel. The history panel runs from "
        "ahead (top) through left, behind (middle) and right to ahead again (bottom), so directions behind the robot "
        "sit in the middle of the panel; the future sub-panel is centred on ahead. Each ready System 2 call (a call "
        "after the warm-up that returned a pixel goal) is a column from its step to the next call, i.e. its executed "
        "action chunk, with {slots} and the System 1 path endpoint at the column's centre.{warm}{stride} (c) Each key "
        "moment: the decision image (the look-down image after System 2's “↓”), the past frames given to System 2, the "
        "360° surroundings re-rendered at the recorded position with the predicted history (top) and future (bottom) "
        "affordance maps, and the executed action chunk (↑ forward 0.25 m, ←/→ turn 15°). On the history strip every "
        "mark sits at its own bearing and elevation, and a grey line joins a past frame's predicted peak to its true "
        "direction when the two are apart; the strips repeat 8° past ±180°, so a mark right behind can appear at both "
        "ends. An orange dot without a blue circle: predicted visible, but the past frame is not visible from here. "
        "{key_rules}{order} The System 1 path and the future affordance map are both decoded from "
        "Z̃ = Z + bridge(Z, M), System 2's latent Z plus a correction the bridge computes from the history memory M; "
        "the future map does not feed the actions."
    ),
    "zh": (
        "R2R val_unseen 第 {ep} 集、场景 {scene} 的闭环复跑，类别 {cat} {cat_name}；标记见图例。(a) 复跑路线与关键时刻 "
        "K1–K{nk}。{route}(b) 在线运行中的 affordance map：横轴为步数；纵轴为机器人周围的方位（0° = 正前方，左为正，"
        "±180° = 正后方），各子图都是向上 = 向左。历史图自上而下从正前方经左、正后方（居中）、右回到正前方，身后的方向"
        "位于纵轴中部；未来子图以正前方居中。每次就绪调用（预热期之后、慢系统给出像素目标的调用）占一列，从该调用的步延续到"
        "下一次调用，即它执行的动作块；{slots}，快系统路径终点画在列中央。{warm}{stride}(c) 各关键时刻：慢系统据以决策的图"
        "（答“↓”后为俯视帧），送入慢系统的历史帧，在记录位置重渲染的 360° 环视及其上的预测历史（上）与预测未来（下）"
        "affordance map，以及该次调用后执行的动作块（↑ 前进 0.25 m，←/→ 转 15°）。历史条带上每个标记都画在它自己的方位与"
        "仰角处，同一历史帧的预测峰值与真实方向相距较远时以灰线相连；条带两端各重复 8°，正后方的标记可能在两端各出现一次。"
        "有橙点而无蓝圈：模型认为该历史帧可见，但它从此处并不可见。{key_rules}{order}快系统路径与预测未来 affordance map "
        "都由 Z̃ 解码，Z̃ 为桥接把历史认知头的概括向量 M 注入慢系统隐变量 Z 所得；未来图不回流到动作。"
    ),
}
CAPTION_SLOTS = {  # what a column marks (tp.MarkPlan.mode)
    "en": {"all": "its past frames' marks spread from 1 (oldest, left) to 8",
           "some": ("its past frames' marks spread from 1 (oldest, left) to 8 (in columns too narrow for eight, frames "
                    "1, 4 and 8 only)"),
           "148": "the marks of its past frames 1 (oldest, left), 4 and 8",
           "stacked": "its past frames' marks stacked at the column's centre (all eight at one x)"},
    "zh": {"all": "列内各历史帧的标记从 1（最早，左）到 8 排开",
           "some": "列内各历史帧的标记从 1（最早，左）到 8 排开（放不下 8 个的窄列只画 1、4、8）",
           "148": "列内画历史帧 1（最早，左）、4、8 的标记",
           "stacked": "各历史帧的标记叠在列中央（8 个在同一横坐标）"},
}

CAPTION_NO_KEYS = {
    "en": ("Closed-loop rerun of R2R val_unseen episode {ep} (scene {scene}; category {cat}, {cat_name}). The rerun "
           "has no ready System 2 call, so no affordance map and no key moment is drawn. (a) The executed route "
           "(black), the reference path (grey, dashed), start (open circle) and goal (star) with its 3 m success "
           "radius. (b) The calls over the rerun: hatched = warm-up, grey = System 2 answered with turns or STOP."),
    "zh": ("R2R val_unseen 第 {ep} 集（场景 {scene}；类别 {cat}，{cat_name}）的闭环复跑。本次复跑没有就绪调用，因此不画 "
           "affordance map 与关键时刻。(a) 执行路线（黑）、参考路径（灰虚线）、起点（空心圆）、目标（星）及 3 m 成功半径。"
           "(b) 复跑中的调用：斜线 = 预热期，灰底 = 慢系统直接给出转向或停止。"),
}

CAPTION_OVERVIEW = {
    "en": (
        "Closed-loop reruns of one typical {group} episode per category ({cats}), chosen by the pre-registered rule. "
        "Each row: the route; the online timeline (x = step; y = bearing, up = left, the history panel running "
        "ahead, left, behind, right, ahead from top to bottom; one column per ready System 2 call, i.e. after the "
        "warm-up and returning a pixel goal, marking past frames 1, 4 and 8{warm}); key moments {labels}, chosen by "
        "the pre-registered rules stated in each episode's caption (K1, K4 on the episode pages), with the decision "
        "image, the past frames given to System 2 and the 360° strips re-rendered at the recorded position (8° "
        "repeated past ±180°). An orange dot without a blue circle: predicted visible, but the past frame is not "
        "visible from here. {exceptions}{order}"
    ),
    "zh": (
        "闭环复跑，每类一集典型{group}案例（{cats}），按预注册规则选取。每行依次为：复跑路线；在线时间线（横轴 = 步数；纵轴 = "
        "方位，向上 = 向左，历史图自上而下为前、左、后、右、前；每次就绪调用即预热期之后、给出像素目标的调用占一列，列内画历史帧"
        " 1、4、8{warm}）；关键时刻 {labels}（按各集图注所述的预注册规则选取；K1、K4 见逐集页面），含决策图、送入慢系统的历史帧"
        "以及在记录位置重渲染的 360° 条带（两端各重复 8°）。有橙点而无蓝圈：模型认为该历史帧可见，但它从此处并不可见。"
        "{exceptions}{order}"
    ),
}
CAPTION_WARM = {
    "page": {"en": " The warm-up (steps 0–{w1}, no affordance map yet) is compressed into the hatched block at the "
                   "left; the axis breaks at step {w}.",
             "zh": "预热期（第 0–{w1} 步，尚无 affordance map）压缩为左端的斜线块，坐标轴在第 {w} 步处断开。"},
    "overview": {"en": "; warm-up compressed into the hatched block, with an axis break",
                 "zh": "；预热期压缩为斜线块，坐标轴断开"},
    "overview_some": {"en": "; warm-up compressed with an axis break in {comp} only",
                      "zh": "；预热期仅在 {comp} 中压缩并断轴"},
}
CAPTION_NOMAP = {
    "page": {"en": (" Ready calls cover only {pct}% of this rerun, so the long spans where System 2 gave turns or STOP "
                    "(steps {spans}) are compressed into grey blocks, with axis breaks."),
             "zh": "本次复跑中就绪调用只覆盖 {pct}% 的步数，因此慢系统直接给出转向或停止的长段（第 {spans} 步）也压缩为灰块并断轴。"},
    "overview": {"en": "; in {cases}, long spans without a map are compressed into grey blocks, with axis breaks",
                 "zh": "；{cases} 中无 affordance map 的长段压缩为灰块并断轴"},
}
CAPTION_ORDER = {
    "en": {"one": "{a} (step {sa}) comes before {b} (step {sb}) in time", "join": "; ", "page": " Here {pairs}.",
           "case": " In {cat}, {pairs}."},
    "zh": {"one": "{a}（第 {sa} 步）在时间上早于 {b}（第 {sb} 步）", "join": "；", "page": "此处 {pairs}。",
           "case": "{cat} 中 {pairs}。"},
}
CAPTION_STRIDE = {
    "en": " On {which}, the dots and circles are drawn for one call in {k} (and at every key moment).",
    "zh": "{which}点和圈每 {k} 次调用画一次（关键时刻都画）。",
}
STRIDE_WHICH = {"en": {"page": "this long rerun"}, "zh": {"page": "该复跑较长，"}}


CAPTION_STACKED_OV = {
    "en": "On the long rerun{s} of {cases}, all eight past frames' marks are stacked at the column's centre{every}.",
    "zh": "{cases} 较长，8 个历史帧的标记叠在列中央{every}。",
}
CAPTION_EVERY_OV = {"en": " and drawn for one call in {k} (and at every key moment)",
                    "zh": "，且每 {k} 次调用画一次（关键时刻都画）"}


def stride_sentence(lang: str, k: int, cases: Optional[Sequence[str]] = None) -> str:
    """Caption sentence for timelines whose markers are thinned to every k-th call ('' when k <= 1 on a page).
    With ``cases`` (the overview, whose timelines otherwise mark past frames 1, 4, 8): the stacked cases, said
    to be stacked even when k = 1."""
    if cases is not None:
        every = CAPTION_EVERY_OV[lang].format(k=k) if k > 1 else ""
        return CAPTION_STACKED_OV[lang].format(s="s" if len(cases) > 1 else "",
                                               cases=(" and " if lang == "en" else "、").join(cases), every=every)
    if k <= 1:
        return ""
    return CAPTION_STRIDE[lang].format(which=STRIDE_WHICH[lang]["page"], k=k)


def order_pairs(keys: Sequence[bd.KeyStep], lang: str) -> str:
    """"K3 (step 31) comes before K2 (step 35) in time" for every pair whose numbers run against time ('' if none)."""
    O = CAPTION_ORDER[lang]
    ks = sorted(keys, key=lambda k: k.label)
    pairs = [O["one"].format(a=b.label, sa=b.step, b=a.label, sb=a.step)
             for i, a in enumerate(ks) for b in ks[i + 1:] if int(b.step) < int(a.step)]
    return O["join"].join(pairs)


NONREG_BRANCHES = ("K3_f1_fallback_after",)  # key-moment branches that are not in the pre-registration


def merged_exceptions(bundles: Sequence[bd.Bundle], chosen: Sequence[Sequence[bd.KeyStep]], cats: Sequence[str],
                      lang: str, branches: object = True) -> str:
    """``fb.main_exceptions`` with the same fallback of several cases said once ("In F1 and F2, K2 is ...").
    ``branches``: True = every branch other than the rule's main clause; "nonreg" = only the branches that are not
    in the pre-registration (``NONREG_BRANCHES``: the overview names the rules only by reference, so these must be
    stated); False = none.  Cases without a key moment or with fewer than four are always stated."""
    R, M = fb.KEY_RULES[lang], fb.MAIN_TEXT[lang]
    out, by_rule = [], {}
    for b, ch, cat in zip(bundles, chosen, cats):
        if not b.keys or any(k.branch == "all_lt4" for k in b.keys):
            out.append(fb.main_exceptions(b, ch, cat, lang))
            continue
        for k in ch:
            if branches == "nonreg":
                if k.branch in NONREG_BRANCHES:
                    by_rule.setdefault((k.label, k.branch), []).append(cat)
            elif branches and k.branch != fb.standard_branch(k.label, cat):
                by_rule.setdefault((k.label, k.branch), []).append(cat)
    for (label, branch), cs in by_rule.items():
        who = (" and " if lang == "en" else " 与 ").join(cs)
        out.append(M["branch"].format(cat=who, label=label, rule=R[branch]))
    return "".join(out)


CAPTION_SYNTH_TL = {
    "en": "SYNTHETIC TIMELINE (layout test): the timeline is not the rerun's data.",
    "zh": "合成时间线（排版测试）：时间线不是复跑数据。",
}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def outcome_line(b: bd.Bundle, L: dict) -> str:
    o = b.outcome
    if o is None:
        return L["outcome_missing"]
    ne = f"{o['ne_m']:.1f}" if fb._finite(o["ne_m"]) else L["ne_missing"]
    if o["success"]:
        return L["outcome_success"].format(steps=o["steps"], ne=ne)
    os_note = L["outcome_os"] if o["oracle_success"] else ""
    key = {"stop": "outcome_stop", "step_cap": "outcome_cap"}.get(o["ended_by"], "outcome_other")
    return L[key].format(steps=o["steps"], ne=ne, os=os_note)


PIXEL_GOAL_TEXT = re.compile(r"^\s*\d+\s+\d+\s*$")


LOOK_DOWN_TEXT = "↓"  # System 2's "look down first" answer (not the executed action "move back")


def s2_parts(ks: bd.KeyStep, L: dict, look: bool, goal: bool) -> List[str]:
    """System 2's outputs, each quoted, with the glosses asked for: "(look down)" after "↓", "(pixel goal)" after
    a pixel goal "u v"."""
    texts = [str(t) for t in ks.system2_texts]
    out = []
    for i, t in enumerate(texts):
        part = f"“{t}”"
        if look and t.strip() == LOOK_DOWN_TEXT:
            part += L["s2_look"]
        if goal and i == len(texts) - 1 and PIXEL_GOAL_TEXT.match(t):
            part += L["goal_suffix"]
        out.append(part)
    return out


def s2_text(ks: bd.KeyStep, L: dict) -> str:
    return L["s2"].format(texts=L["then"].join(s2_parts(ks, L, False, False)))


def s2_lines(fig, ks: bd.KeyStep, L: dict, width_pt: float, fs: float, max_lines: int = 1) -> List[str]:
    """System 2's output in at most ``max_lines`` lines of ``width_pt``, with as many glosses as fit.

    Tried in order: both glosses on one line, both on two lines (broken before
    the arrow), the look-down gloss only (one line, then two), no gloss.  The
    look-down gloss matters most: the executed-action chips use the same
    arrows with another meaning (↑ = forward).
    """
    def fits(lines):
        return all(cd.text_width_pt(fig, ln, fs) <= width_pt for ln in lines)
    for look, goal in ((True, True), (True, False), (False, False)):
        parts = s2_parts(ks, L, look, goal)
        one = [L["s2"].format(texts=L["then"].join(parts))]
        if fits(one):
            return one
        if max_lines >= 2 and len(parts) >= 2:
            two = [L["s2"].format(texts=L["then"].join(parts[:-1])), L["then"].lstrip() + parts[-1]]
            if fits(two):
                return two
    return [s2_text(ks, L)]


def s2_line(fig, ks: bd.KeyStep, L: dict, width_pt: float, fs: float) -> str:
    """System 2's output on one line (``s2_lines`` with max_lines = 1)."""
    return s2_lines(fig, ks, L, width_pt, fs, 1)[0]


def setup(lang: str) -> None:
    """``fb.setup`` (exp18 style, fonts) and mathtext in the body font, so the Z~ of the notes matches the text."""
    fb.setup(lang)
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
                                "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold"})


def min_font_size(fig) -> float:
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return float(min(sizes)) if sizes else float("nan")


def texts_outside(fig, tol_px: float = 1.0) -> List[str]:
    """Visible texts that stick out of the figure (clipped on the page)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        bb = t.get_window_extent(renderer)
        if bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px:
            out.append(t.get_text()[:40])
    return out


INK_DPI = 400  # the PNG's resolution: the ink extent is measured as the page is rasterised
INK_MARGIN_PT = 2.0  # a wrapped line's ink stays this far inside its width
_INK_CACHE: Dict[tuple, Tuple[float, float]] = {}


def ink_extent_pt(text: str, fs: float, **kw) -> Tuple[float, float]:
    """(left, right) of the ink of ``text`` drawn left-aligned at x = 0 with the current fonts, in points: rendered
    at ``INK_DPI``, so an italic glyph's overhang past its advance width (the last "p" of an italic line) counts,
    unlike ``cd.text_width_pt`` (the layout box).  Left < 0 when a glyph starts left of its origin."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rc = matplotlib.rcParams
    key = (text, float(fs), tuple(sorted(kw.items())), tuple(rc["font.family"]), tuple(rc["font.sans-serif"]))
    if key in _INK_CACHE:
        return _INK_CACHE[key]
    lead = 0.25  # in: room for a left overhang
    w_in = lead * 2 + len(text) * fs * 1.2 / 72.0 + 0.5
    fig = Figure(figsize=(w_in, fs * 4.0 / 72.0), dpi=INK_DPI)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    fig.text(lead / w_in, 0.5, text, fontsize=fs, ha="left", va="center", color="black", **kw)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    cols = np.nonzero((rgb < 250).any(axis=-1).any(axis=0))[0]
    px = 72.0 / INK_DPI
    out = (0.0, 0.0) if not cols.size else ((cols[0] - lead * INK_DPI) * px, (cols[-1] + 1 - lead * INK_DPI) * px)
    _INK_CACHE[key] = out
    return out


def wrap_ink(fig, text: str, fs: float, width_pt: float, margin_pt: float = INK_MARGIN_PT, **kw) -> List[str]:
    """``fb.wrap`` whose lines' rendered ink (``ink_extent_pt``, italic overhang included) ends at least
    ``margin_pt`` inside ``width_pt``: the wrap width shrinks until every line fits."""
    limit = width_pt - margin_pt
    w = limit
    lines = fb.wrap(fig, text, fs, w, **kw)
    for _ in range(12):
        over = max([ink_extent_pt(ln, fs, **kw)[1] - limit for ln in lines] or [0.0])
        if over <= 0.0:
            break
        w -= over + 0.25
        lines = fb.wrap(fig, text, fs, w, **kw)
    return lines


def text_overlaps(fig, min_px2: float = 2.0) -> List[str]:
    """Pairs of visible texts whose rendered boxes overlap (a collision on the page).

    Mathtext segments ("$...$", the Z~ of the data-flow note) are skipped: their
    box includes the full math ascent, taller than the glyphs drawn.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and not t.get_text().startswith("$"):
            bb = t.get_window_extent(renderer)
            boxes.append((t.get_text()[:30], bb.x0, bb.y0, bb.x1, bb.y1))
    out = []
    for i in range(len(boxes)):
        a = boxes[i]
        for b in boxes[i + 1:]:
            w = min(a[3], b[3]) - max(a[1], b[1])
            h = min(a[4], b[4]) - max(a[2], b[2])
            if w > 0 and h > 0 and w * h > min_px2:
                out.append(f"{a[0]!r} x {b[0]!r}")
    return out


MATH_GAP_PT = 0.9  # gap kept between a mathtext run and an adjacent non-space glyph (CJK text has no spaces)


def rich_line(ax, x: float, y: float, parts: Sequence[str], fs: float, color=style.INK_2, **kw):
    """Text segments placed one after the other on the line y (point-unit axes); returns the last text artist.

    Each segment after the first is anchored to the right edge of the one
    before as drawn (an annotation, resolved at draw time), so the spacing is
    exact at every output dpi: widths measured at the screen dpi drift by points
    over a long line, and a 6 pt mathtext Z~ measures 5.1 pt wide at 100 dpi but
    draws 3.6 pt wide at 400 dpi.  Next to a CJK glyph (no space) a mathtext
    segment keeps ``MATH_GAP_PT``.
    """
    parts = list(parts)
    prev = None
    for i, part in enumerate(parts):
        math, prev_math = part.startswith("$"), i > 0 and parts[i - 1].startswith("$")
        dx = 0.0
        if math and i > 0 and not parts[i - 1][-1:].isspace():
            dx += MATH_GAP_PT
        if prev_math and not part[:1].isspace() and part[:1] not in ";,.:)；，。：）":
            dx += MATH_GAP_PT
        if prev is None:
            prev = ax.text(x + dx, y, part, ha="left", va="center", fontsize=fs, color=color, **kw)
        else:
            prev = ax.annotate(part, xy=(1.0, y), xycoords=(prev, "data"), xytext=(dx, 0.0),
                               textcoords="offset points", ha="left", va="center", fontsize=fs, color=color, **kw)
    return prev


def rich_wrap(fig, parts: Sequence[str], fs: float, width_pt: float) -> List[List[str]]:
    """Greedy wrap of text segments, where a "$...$" segment (mathtext) is one unbreakable token.

    Returns lines, each a list of runs (plain text or one mathtext segment) for ``rich_line``.
    """
    tokens: List[str] = []
    for part in parts:
        tokens += [part] if part.startswith("$") else fb._TOKEN.findall(part)
    tokens = [t for t in tokens if t]
    lines: List[List[str]] = [[]]
    used = 0.0
    for tok in tokens:
        if not lines[-1] and tok.isspace():
            continue
        w = cd.text_width_pt(fig, tok.replace(" ", "\u00a0"), fs) + (2 * MATH_GAP_PT if tok.startswith("$") else 0.0)
        if lines[-1] and not tok.isspace() and tok not in fb._NO_LINE_START and used + w > width_pt - 2.0:
            lines.append([])
            used = 0.0
        lines[-1].append(tok)
        used += w
    out = []
    for line in lines:
        runs, buf = [], ""
        for tok in line:
            if tok.startswith("$"):
                if buf:
                    runs.append(buf)
                buf = ""
                runs.append(tok)
            else:
                buf += tok
        if buf.rstrip():
            runs.append(buf.rstrip())
        out.append(runs)
    return out


# --------------------------------------------------------------------------- #
# Shared blocks
# --------------------------------------------------------------------------- #
class Geom:
    """Key-moment column geometry (inches)."""

    def __init__(self, w: float, captions: bool, ticks: bool, badge_h: float = 0.14, s2_h: float = 0.125,
                 cap_h: float = 0.1, gap: float = 0.04, chips_h: float = 0.14, thumb_gap: float = 0.022,
                 tick_h: float = 0.135, s2_lines: int = 1, row_caps: bool = False):
        """``captions``: the page's column (row captions in the first column, steps on the thumbnails);
        ``row_caps``: rows for short row labels (the overview's first block) without the page's extras."""
        self.w, self.captions, self.ticks = w, captions, ticks
        self.row_caps = bool(row_caps)
        self.s2_lines = max(1, int(s2_lines))
        s2_h = s2_h + (self.s2_lines - 1) * LINE
        self.badge_h, self.s2_h, self.chips_h, self.thumb_gap, self.tick_h = badge_h, s2_h, chips_h, thumb_gap, tick_h
        self.cap_h = cap_h if (captions or row_caps) else gap
        self.dec_h = w * 0.75
        self.thumb_w = (w - 3 * thumb_gap) / 4
        self.thumb_h = self.thumb_w * 0.75
        self.grid_h = 2 * self.thumb_h + thumb_gap
        self.hist_h = p2.strip_height(w, p2.HIST_ELEV)
        self.fut_h = p2.strip_height(w, p2.FUT_ELEV)

    def rows(self) -> Dict[str, float]:
        """Top offset of every row from the column top."""
        y, out = 0.0, {}
        for name, h in (("badge", self.badge_h), ("dec", self.dec_h), ("s2", self.s2_h), ("cap1", self.cap_h),
                        ("grid", self.grid_h), ("cap2", self.cap_h), ("hist", self.hist_h), ("cap3", self.cap_h),
                        ("fut", self.fut_h), ("ticks", self.tick_h if self.ticks else 0.02),
                        ("chips", self.chips_h)):
            out[name] = y
            y += h
        out["end"] = y
        return out

    @property
    def height(self) -> float:
        return self.rows()["end"]


def overview_tag(fig, ks: bd.KeyStep, L: dict, wpt: float, fs: float = FS["small"]) -> str:
    """The overview's image-corner tag: "step N · look-down" when it fits beside the K badge (the room the tag
    box actually has: badge at 3 pt, tag 3 pt + 1 pt after it, the box's padding, 1.5 pt air), else "step N"."""
    tag = L["step"].format(s=ks.step)
    longer = tag + " · " + L[ks.decision_image + "_short"]
    bw = cd.text_width_pt(fig, ks.label, p2.MIN_FS, fontweight="bold") + 2 * 0.22 * p2.MIN_FS
    room = wpt - (p2.TAG_X0_PT + bw + p2.TAG_GAP_PT + 1.0) - 2 * 0.18 * fs - 3.0
    return longer if cd.text_width_pt(fig, longer, fs) <= room else tag


def draw_key_column(page: fb.Page, x: float, y: float, g: Geom, ks: bd.KeyStep, L: dict, first: bool,
                    letter: Optional[str] = None, row_labels: Optional[Sequence[str]] = None) -> dict:
    """One key moment, top to bottom: badge + step, decision image, System2 output, 2x4 history frames,
    360° history strip, 360° future strip, executed actions.  ``row_labels``: short labels of the frames / history
    strip / future strip rows (the overview's first block).  Returns {"edge": edge marks}."""
    fig = page.fig
    rows = g.rows()
    wpt = g.w * 72.0
    dax = page.ax(x, y + rows["dec"], g.w, g.dec_h)
    if g.badge_h > 0:  # badge + step above the image
        hax = page.pt_axes(x, y + rows["badge"], g.w, g.badge_h)
        mid = g.badge_h * 72.0 * 0.5
        x0 = 0.0
        if letter:
            hax.text(0.0, mid, letter, ha="left", va="center", fontsize=FS["panel"], fontweight="bold",
                     color=style.INK)
            x0 = cd.text_width_pt(fig, letter, FS["panel"], fontweight="bold") + 4.0
        bw = cd.text_width_pt(fig, ks.label, p2.MIN_FS, fontweight="bold")
        cd.key_badge(hax, x0 + bw / 2 + 2.0, mid, ks.label, fs=p2.MIN_FS)
        hax.text(x0 + bw + 7.0, mid, L["step"].format(s=ks.step), ha="left", va="center", fontsize=FS["head"],
                 fontweight="bold", color=style.INK)
        p2.draw_decision_image(dax, ks, tag=L[ks.decision_image])
    else:  # overview: badge + step (+ image kind when it fits) in the image's corner
        p2.draw_decision_image(dax, ks, tag=overview_tag(fig, ks, L, wpt), badge=ks.label)
    # System2 output (with the "↓" = look down gloss; two lines where the column is narrow)
    sax = page.pt_axes(x, y + rows["s2"], g.w, g.s2_h)
    lines = s2_lines(fig, ks, L, wpt, FS["small"], g.s2_lines)
    top = g.s2_h * 72.0 - (g.s2_h - (g.s2_lines - 1) * LINE) * 72.0 * 0.55
    for j, line in enumerate(lines):
        sax.text(0.0, top - j * LINE * 72.0, line, ha="left", va="center", fontsize=FS["small"], color=style.INK)
    # captions (first column only)
    if first and g.captions:
        for key, row in (("cap_frames", "cap1"), ("cap_hist", "cap2"), ("cap_fut", "cap3")):
            page.text(x, y + rows[row] + g.cap_h * 0.52, L[key], ha="left", va="center", fontsize=FS["small"],
                      color=style.INK_2)
    elif row_labels and g.row_caps:
        for text, row in zip(row_labels, ("cap1", "cap2", "cap3")):
            page.text(x, y + rows[row] + g.cap_h * 0.5, text, ha="left", va="center", fontsize=p2.MIN_FS,
                      color=style.INK_2)
    # 2x4 history frames, slot 1 = oldest
    lane = page.pt_axes(x, y + rows["grid"], g.w, g.grid_h, zorder=5)
    K = int(ks.history_count)
    for j in range(bd.NUM_SLOTS):
        cx = x + (j % 4) * (g.thumb_w + g.thumb_gap)
        cy = y + rows["grid"] + (j // 4) * (g.thumb_h + g.thumb_gap)
        fax = page.ax(cx, cy, g.thumb_w, g.thumb_h)
        if j < K:
            pn.draw_image(fax, ks.history_rgb[j], frame_color=style.AXIS, frame_lw=p2.HAIR)
            pad = p2.SLOT_PAD if g.captions else p2.SLOT_PAD_SMALL
            off = 4.6 if g.captions else 4.1
            bx = (cx - x) * 72.0 + off
            by = (g.grid_h - (cy - y - rows["grid"])) * 72.0 - off
            p2.slot_badge(lane, bx, by, j, pad=pad)
            steps = list(ks.history_steps or [])
            if g.captions and j < len(steps):  # pages: the step of each past frame, bottom-right corner
                p2.step_tag(lane, (cx - x + g.thumb_w) * 72.0, (g.grid_h - (cy - y - rows["grid"]) - g.thumb_h) * 72.0,
                            str(int(steps[j])))
        else:
            fax.set_facecolor(style.SURFACE)
            cd.clean_axes(fax, spines=True, color=style.GRID, lw=p2.HAIR)
    # 360° strips
    bax = page.ax(x, y + rows["hist"], g.w, g.hist_h)
    info = p2.draw_history_strip(bax, ks, _ring_px(g.w, 340), _ring_px(g.w, 272))
    cax = page.ax(x, y + rows["fut"], g.w, g.fut_h)
    p2.draw_future_strip(cax, ks, _ring_px(g.w, 272))
    if g.ticks:
        p2.strip_ticks(cax, L["axis"])
    # executed actions
    kax = page.pt_axes(x, y + rows["chips"], g.w, g.chips_h)
    cy = g.chips_h * 72.0 * 0.5
    kax.text(0.0, cy, L["executed"], ha="left", va="center", fontsize=FS["small"], color=style.INK_2)
    xw = cd.text_width_pt(fig, L["executed"], FS["small"]) + 3.0
    p2.action_chips(kax, xw, cy, ks.executed_actions, size=7.6)
    return info


def _ring_px(w_in: float, ppi: float) -> int:
    return max(16, int(round(w_in * ppi / 8.0)) * 8)


LEGEND_HEAD_H = 0.095  # the legend's group-header row (in)
LEGEND_GAP_PT = 10.0  # air between two legend columns
GLYPH_TEXT_PT = 17.0  # a legend entry's text starts this far right of its glyph's left edge


SLOT_ALT_GROUP, SLOT_ALT_AFTER = 2, "nomap"  # the slot entry's other place: with the timeline's context marks


def split_at_semicolon(text: str) -> Optional[List[str]]:
    """A legend text broken after its first semicolon ("…; …" or "…；…"), or None when it has none."""
    for sep, keep in (("；", "；"), ("; ", ";")):
        i = text.find(sep)
        if 0 < i < len(text) - len(sep):
            return [text[:i] + keep, text[i + len(sep):].strip()]
    return None


def legend_columns(fig, L: dict, width_pt: float, keys: Sequence[str] = LEGEND_KEYS,
                   groups: Sequence[Sequence[str]] = LEGEND_GROUPS, gap: float = LEGEND_GAP_PT) -> dict:
    """The legend as a fixed grid, one column per group (history | future and decisions | route and timeline),
    each headed by its group name, entries one under the other.  ``keys``: the entries this figure draws (the
    group's "slots" place takes whichever slot entry is in ``keys``).  Column widths follow their widest entry;
    while the columns do not fit ``width_pt``, the widest single-line entry is wrapped onto two lines (after its
    semicolon when it has one).  The slot entry (what a timeline column marks) sits in the history column, or,
    when it needs two lines and that makes the legend shorter, with the timeline's context marks (after "System 2
    gave turns or STOP").
    Returns {"cols": [(header, [(key, lines)])], "x": [column x (pt)], "rows": entry rows, "slot_group"}."""
    fs = FS["small"]
    slots = [k for k in SLOT_KEYS + SLOT_EXTRA if k in keys]  # the slot entry (+ the overview's stacked rows)
    slot = slots[0] if slots else None

    def col_w(head, items):
        w = cd.text_width_pt(fig, head, fs, fontweight="bold")
        for _, lines in items:
            w = max(w, GLYPH_TEXT_PT + max(cd.text_width_pt(fig, ln, fs) for ln in lines))
        return w

    def layout(slot_group: int) -> dict:
        cols = []
        for gi, (group, head) in enumerate(zip(groups, L["legend_groups"])):
            items = []
            for k in group:
                if k == "slots":
                    if slot is not None and gi == slot_group:
                        items += [[k_, [L["legend"][k_]]] for k_ in slots]
                    continue
                if k in keys:
                    items.append([k, [L["legend"][k]]])
            if slot is not None and gi == slot_group and "slots" not in group:
                at = next((i + 1 for i, it in enumerate(items) if it[0] == SLOT_ALT_AFTER), 0)
                items[at:at] = [[k_, [L["legend"][k_]]] for k_ in slots]
            cols.append((head, items))
        for _ in range(12):
            widths = [col_w(h, it) for h, it in cols]
            if sum(widths) + gap * (len(cols) - 1) <= width_pt:
                break
            c = int(np.argmax(widths))
            single = [it for it in cols[c][1] if len(it[1]) == 1]
            if not single:
                break
            it = max(single, key=lambda e: cd.text_width_pt(fig, e[1][0], fs))
            text = it[1][0]
            it[1] = split_at_semicolon(text) or fb.wrap(fig, text, fs, cd.text_width_pt(fig, text, fs) * 0.6 + 6.0)[:2] \
                or [text]
        widths = [col_w(h, it) for h, it in cols]
        spare = max(0.0, width_pt - sum(widths) - gap * (len(cols) - 1))
        xs, x = [], 0.0
        for w in widths:  # spare room spread between the columns, so the grid spans the page
            xs.append(x)
            x += w + gap + (spare / (len(cols) - 1) if len(cols) > 1 else 0.0)
        rows = max(sum(len(lines) for _, lines in items) for _, items in cols) if cols else 0
        return {"cols": cols, "x": xs, "rows": rows, "slot_group": slot_group}

    home = next((gi for gi, g in enumerate(groups) if "slots" in g), 0)
    lay = layout(home)
    wrapped = any(k in slots and len(lines) > 1 for _, items in lay["cols"] for k, lines in items)
    if wrapped and SLOT_ALT_GROUP < len(groups) and SLOT_ALT_GROUP != home:
        alt = layout(SLOT_ALT_GROUP)
        if alt["rows"] < lay["rows"]:
            lay = alt
    return lay


def legend_height(fig, L: dict, width_pt: float, keys: Sequence[str]) -> float:
    return LEGEND_HEAD_H + LINE * legend_columns(fig, L, width_pt, keys)["rows"]


def draw_legend(page: fb.Page, x: float, y: float, w: float, L: dict, keys: Sequence[str] = LEGEND_KEYS) -> float:
    """Markers explained once, in a fixed grid of grouped columns (``legend_columns``); returns the height (in)."""
    lay = legend_columns(page.fig, L, w * 72.0, keys)
    h = LEGEND_HEAD_H + LINE * lay["rows"]
    ax = page.pt_axes(x, y, w, h)
    top = h * 72.0
    for (head, items), cx in zip(lay["cols"], lay["x"]):
        ax.text(cx, top - LEGEND_HEAD_H * 72.0 * 0.45, head, ha="left", va="center", fontsize=FS["small"],
                fontweight="bold", color=style.INK)
        i = 0
        for key, lines in items:
            yy = top - LEGEND_HEAD_H * 72.0 - (i + 0.5) * LINE * 72.0
            p2.legend_glyph(ax, key, cx, yy - (len(lines) - 1) * LINE * 36.0)
            for j, line in enumerate(lines):
                ax.text(cx + GLYPH_TEXT_PT, yy - j * LINE * 72.0, line, ha="left", va="center", fontsize=FS["small"],
                        color=style.INK)
            i += len(lines)
    return h


def note_lines(fig, w: float, L: dict) -> List[List[str]]:
    """The one-line data-flow note, wrapped to ``w`` inches (runs per line; Z~ in mathtext)."""
    return rich_wrap(fig, list(L["flow_note"]), FS["small"], w * 72.0)


def draw_notes(page: fb.Page, x: float, y: float, w: float, L: dict) -> float:
    """The data-flow note (the bearing conventions are in the caption); returns the height used (in)."""
    lines = note_lines(page.fig, w, L)
    h = LINE * len(lines)
    ax = page.pt_axes(x, y, w, h)
    for i, runs in enumerate(lines):
        rich_line(ax, 0.0, h * 72.0 - (i + 0.5) * LINE * 72.0, runs, FS["small"])
    return h


def notes_height(fig, w: float, L: dict) -> float:
    return LINE * len(note_lines(fig, w, L))


def draw_route(page: fb.Page, x: float, y: float, w: float, h: float, b: bd.Bundle, topdown, L: dict,
               labels: Optional[Sequence[str]] = None) -> dict:
    """Route map (``p2.draw_route_map``); returns its info ({"stop_drawn", "start_label"})."""
    ax = page.ax(x, y, w, h)
    keys = [k for k in b.keys if labels is None or k.label in labels]
    info = p2.draw_route_map(ax, topdown[1] if topdown is not None else None, b.xz("route_xz"),
                             b.xz("reference_path_xz"), b.start_xz, b.goal_xz, float(b.goal_radius_m),
                             [k.position_xz for k in keys], [k.label for k in keys], L["start"], L["radius"],
                             off_level=fb.off_level_steps(b, topdown))
    if topdown is None:
        ax.text(0.5, 0.03, L["no_map"], transform=ax.transAxes, ha="center", va="bottom", fontsize=p2.MIN_FS,
                color=style.INK_2, fontstyle="italic")
    return info


def leader_crossings(fig, tol_px: float = 0.5) -> List[str]:
    """Leader lines (gid "leader:<K>") that run through a text box other than their own badge's (route maps and
    the timeline's badge row)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leaders = [ln for ln in ax.lines if str(ln.get_gid() or "").startswith("leader:")]
        if not leaders:
            continue
        texts = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        boxes = []
        for t in texts:
            bb = t.get_window_extent(renderer)
            patch = t.get_bbox_patch()
            if patch is not None:
                bb = patch.get_window_extent(renderer)
            boxes.append((t.get_text(), bb))
        for ln in leaders:
            own = ln.get_gid().split(":", 1)[1]
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            (x0, y0), (x1, y1) = xy[0], xy[-1]
            t = np.linspace(0.0, 1.0, 64)
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for text, bb in boxes:
                if text == own:
                    continue
                hit = (px > bb.x0 + tol_px) & (px < bb.x1 - tol_px) & (py > bb.y0 + tol_px) & (py < bb.y1 - tol_px)
                if np.any(hit):
                    out.append(f"leader of {own} crosses {text[:20]!r}")
    return out


TURN_H = 0.13  # executed-turn track under the future panel
TURN_GAP = 0.03


HIST_TITLE_H = 0.11  # overview: the title row above the history panel


def timeline_height(h_hist: float, gap: float, h_fut: float, badges_h: float, title_h: float = 0.0) -> float:
    """(Title +) badges + history panel + gap + future panel + turn track (without the step tick labels)."""
    return title_h + badges_h + h_hist + gap + h_fut + TURN_GAP + TURN_H


def prepare_timeline(fig, tl: tp.Timeline, w: float, block: str, base_badges_h: float) -> float:
    """Compress the warm-up for a timeline ``w`` inches wide (``tp.WARM_BLOCK_PT[block]``); returns the height (in)
    of the badge row it needs (one more row when two K badges would touch)."""
    tp.compress_axis(tl, w * 72.0, tp.WARM_BLOCK_PT[block])
    return tp.badges_height_in(tp.badge_levels(fig, tl, w * 72.0), base_badges_h)


def draw_timeline(page: fb.Page, x: float, y: float, w: float, h_hist: float, gap: float, h_fut: float,
                  tl: tp.Timeline, L: dict, badges_h: float = 0.13, fut_title: bool = True,
                  warmup_label: bool = True, title_h: float = 0.0, slots: Optional[Sequence[int]] = None) -> dict:
    """(Title row +) badges row + history panel + future sub-panel + executed-turn track (with the step axis).
    ``slots``: the past frames marked in each column (default all 8; the overview ``tp.OVERVIEW_SLOTS``).

    Call ``prepare_timeline`` first (compressed spans and the badge rows)."""
    if title_h > 0:
        page.text(x, y + title_h * 0.45, L["panel_b_hist"], ha="left", va="center", fontsize=FS["small"],
                  color=style.INK_2)
        y += title_h
    bax = page.pt_axes(x, y, w, badges_h, zorder=6)
    hax = page.ax(x, y + badges_h, w, h_hist)
    fax = page.ax(x, y + badges_h + h_hist + gap, w, h_fut)
    tax = page.ax(x, y + badges_h + h_hist + gap + h_fut + TURN_GAP, w, TURN_H)
    info = tp.draw_history_panel(hax, tl, L["y_hist"], warmup_label=L["warmup"] if warmup_label else None,
                                 warmup_range=L["warmup_range"], slots=slots)
    plan = tp.mark_plan(tl, hax, slots)
    info["slot_labels"] = tp.slot_end_labels(hax, tl, plan)
    info.update(tp.draw_future_panel(fax, tl, L["y_fut"], plan=plan))
    info["turns_drawn"] = tp.draw_turn_track(tax, tl, L["turns"])
    tp.step_axis(tax, tl, L["x_steps"])
    info["badges"] = tp.key_badges(bax, hax, tl)
    if fut_title:
        page.text(x, y + badges_h + h_hist + gap * 0.5, L["panel_b_fut"], ha="left", va="center",
                  fontsize=FS["small"], color=style.INK_2)
    if tl.synthetic:
        hax.text(0.5, 0.97, L["synthetic_tl"], transform=hax.transAxes, ha="center", va="top", fontsize=p2.MIN_FS,
                 fontweight="bold", color="#d03b3b", zorder=20)
    return info


def panel_title(page: fb.Page, x: float, y: float, letter: Optional[str], text: str) -> None:
    ax = page.pt_axes(x, y, 5.0, 0.12)
    xx = 0.0
    if letter:
        ax.text(0.0, 4.3, letter, ha="left", va="center", fontsize=FS["panel"], fontweight="bold", color=style.INK)
        xx = cd.text_width_pt(page.fig, letter, FS["panel"], fontweight="bold") + 4.5
    ax.text(xx, 4.3, text, ha="left", va="center", fontsize=FS["panel"], fontweight="bold", color=style.INK)


# --------------------------------------------------------------------------- #
# Episode page
# --------------------------------------------------------------------------- #
ROUTE_W = 1.95
TL_X = 2.30
HDR_TITLE_H = 0.15


def header_lines(fig, b: bd.Bundle, L: dict, void) -> dict:
    """The header's wrapped lines: every line's rendered ink (italic overhang included) ends ``INK_MARGIN_PT``
    inside the page (``wrap_ink``)."""
    instr = wrap_ink(fig, L["instruction"].format(text=" ".join(b.instruction.split())), FS["body"], FIG_W * 72.0,
                     fontstyle="italic")
    extra = wrap_ink(fig, L["void"].format(reasons="; ".join(void)), FS["body"], FIG_W * 72.0) if void else []
    if b.predicate_holds_on_rerun is False:
        extra.append(L["predicate_fails"].format(cat=b.category))
    return {"instr": instr, "extra": extra}


def header_height(lines: dict) -> float:
    return HDR_TITLE_H + LINE * len(lines["instr"]) + 0.125 + LINE * len(lines["extra"]) + 0.07


def draw_header(page: fb.Page, b: bd.Bundle, lines: dict, lang: str, L: dict, synthetic_tl: bool) -> float:
    y = 0.075
    page.text(0.0, y, L["title"].format(cat=b.category, name=fb.CATEGORY_NAMES[lang][b.category]), ha="left",
              va="center", fontsize=FS["title"], fontweight="bold", color=style.INK)
    page.text(FIG_W, y, L["ids"].format(scene=b.scene_id, ep=b.episode_id), ha="right", va="center",
              fontsize=FS["small"], color=style.MUTED)
    if b.synthetic or synthetic_tl:
        page.text(FIG_W / 2 + 0.2, y, L["synthetic"] if b.synthetic else L["synthetic_tl"], ha="center",
                  va="center", fontsize=FS["small"], fontweight="bold", color="#d03b3b")
    y = HDR_TITLE_H
    for line in lines["instr"]:
        page.text(0.0, y + LINE / 2, line, ha="left", va="center", fontsize=FS["body"], color=style.INK,
                  fontstyle="italic")
        y += LINE
    page.text(0.0, y + 0.0625, outcome_line(b, L), ha="left", va="center", fontsize=FS["body"], fontweight="bold",
              color=style.INK)
    y += 0.125
    for line in lines["extra"]:
        page.text(0.0, y + LINE / 2, line, ha="left", va="center", fontsize=FS["body"], color="#d03b3b")
        y += LINE
    y += 0.035
    page.rule(y)
    return y + 0.035


def page_geometry() -> dict:
    col_gap = 0.11
    w = (FIG_W - 3 * col_gap) / 4
    return {"col_w": w, "col_gap": col_gap, "a_title": 0.14, "badges": 0.12, "hist": 0.80, "gap": 0.11,
            "fut": 0.40, "ticks": 0.12, "band_gap": 0.05, "cap": 0.09, "chips": 0.13, "tick_h": 0.125}


def legend_keys_for(stop: bool, overview: bool = False, mode: str = "all") -> Tuple[str, ...]:
    """Legend keys of a page / an overview: "where the rerun ended" only when drawn; the slot entry that says what
    the timeline marks (``SLOT_KEY[mode]``, ``tp.MarkPlan.mode``; the overview: past frames 1, 4, 8 -- a stacked
    overview row says so in its panel); the frames entry on the overview only (a page names that row in words)."""
    slot = "slots_ov" if overview else SLOT_KEY[mode]
    drop = {k for k in SLOT_KEYS if k != slot}
    if not overview:
        drop.add("frames")
    if not stop:
        drop.add("stop")
    return tuple(k for k in LEGEND_KEYS if k not in drop)


def make_episode_page(bundle_path, timeline_path, out_dir, lang: str = "en", verdicts: Optional[dict] = None,
                      topdown_root=None, void: Optional[Sequence[str]] = None, record_path=None) -> dict:
    """One episode page; returns {"files", "size_in", "min_font_pt", "claims", "warnings", "checks"}."""
    setup(lang)
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    b = bd.load_bundle(bundle_path)
    tl = tp.load_timeline(timeline_path, record_path=record_path)
    warnings = list(b.warnings) + list(tl.warnings) + tp.check_against_bundle(tl, b)
    claims = fb.claim_sentences(verdicts, lang)
    topdown = fb.resolve_level(b, topdown_root)
    G = page_geometry()
    g = Geom(G["col_w"], captions=True, ticks=True, cap_h=G["cap"], chips_h=G["chips"], tick_h=G["tick_h"])

    fig = plt.figure(figsize=(FIG_W, 10.0))
    lines = header_lines(fig, b, L, void)
    floor = fb.floor_note(b, topdown, L)
    floor_lines = fb.wrap(fig, floor, FS["small"], ROUTE_W * 72.0) if floor else []
    tl_w = FIG_W - TL_X - 0.01
    badges_h = prepare_timeline(fig, tl, tl_w, "page", G["badges"])
    h_hist = G["hist"] - (badges_h - G["badges"])  # a second badge row comes out of the history panel
    tl_h = timeline_height(h_hist, G["gap"], G["fut"], badges_h)
    band_a = G["a_title"] + tl_h + G["ticks"]
    route_d = p2.route_note_m(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                              ROUTE_W, tl_h - LINE * (len(floor_lines) + 1))
    if route_d is not None:  # a route that would hide under the start ring: say how far it went
        floor_lines += fb.wrap(fig, L["route_note"].format(d=route_d), FS["small"], ROUTE_W * 72.0)
    route_h = tl_h - LINE * len(floor_lines)
    stop = p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m), ROUTE_W,
                         route_h)
    plan = tp.mark_plan(tl, tl_w * 72.0)  # what the timeline will mark: the legend says it (and a stride)
    mode = plan.mode
    keys = legend_keys_for(stop, overview=False, mode=mode)
    Lg = legend_labels(L, {SLOT_KEY[mode]: slot_legend_text(L, mode, plan.stride)})
    n = len(b.keys)
    band_b = g.height if n else 0.3
    legend_h = legend_height(fig, Lg, FIG_W * 72.0, keys)
    height = (header_height(lines) + 0.035 + band_a + G["band_gap"] + band_b + G["band_gap"] + legend_h + 0.03
              + notes_height(fig, FIG_W, L) + 0.02)
    fig.set_size_inches(FIG_W, height)
    page = fb.Page(fig, FIG_W, height)

    y = draw_header(page, b, lines, lang, L, tl.synthetic)
    # band A: route | online timeline
    panel_title(page, 0.0, y, "a", L["panel_a"])
    panel_title(page, TL_X - 0.28, y, "b", L["panel_b"])
    y_top = y + G["a_title"]
    route_info = draw_route(page, 0.0, y_top, ROUTE_W, route_h, b, topdown, L)
    for j, line in enumerate(floor_lines):
        page.text(0.0, y_top + route_h + LINE * (j + 0.5), line, ha="left", va="center", fontsize=FS["small"],
                  color=style.INK_2, fontstyle="italic")
    checks = draw_timeline(page, TL_X, y_top, tl_w, h_hist, G["gap"], G["fut"], tl, L, badges_h=badges_h)
    y = y_top + band_a - G["a_title"] + G["band_gap"]
    # band B: key moments
    edge, pairs = 0, 0
    if n:
        for i, ks in enumerate(b.keys):
            info = draw_key_column(page, i * (G["col_w"] + G["col_gap"]), y, g, ks, L, first=(i == 0),
                                   letter="c" if i == 0 else None)
            edge += info["edge"]
            pairs += info.get("pairs", 0)
    else:
        page.text(0.0, y + 0.1, L["no_keys"], ha="left", va="center", fontsize=FS["body"], color=style.INK_2,
                  fontstyle="italic")
    y += band_b + G["band_gap"]
    page.rule(y - G["band_gap"] / 2)
    y += draw_legend(page, 0.0, y, FIG_W, Lg, keys) + 0.03
    draw_notes(page, 0.0, y, FIG_W, L)

    caption = page_caption(b, lang, claims, void, edge_marks=edge > 0, synthetic_tl=tl.synthetic,
                           stride=int(checks.get("marker_stride", 1)), tl=tl, route_d=route_d,
                           mode=checks.get("mode", mode))
    if checks.get("mode", mode) != mode or int(checks.get("marker_stride", plan.stride)) != plan.stride:
        warnings.append(f"timeline drawn with marks '{checks.get('mode')}' / stride {checks.get('marker_stride')} "
                        f"but the legend planned '{mode}' / {plan.stride}")
    stem = Path(out_dir) / b.category / f"{b.category_rank}_{b.ep_key}"
    min_fs = min_font_size(fig)
    outside = texts_outside(fig)
    overlaps = text_overlaps(fig)
    crossings = leader_crossings(fig)
    files = fb._save(fig, stem, lang, caption)
    warnings += fb.over_budget(height, PAGE_MAX_H_IN, f"page [{lang}]")
    if overlaps:
        warnings.append(f"overlapping texts [{lang}]: {overlaps}")
    if outside:
        warnings.append(f"texts outside the page: {outside}")
    if crossings:
        warnings.append(f"leaders crossing texts [{lang}]: {crossings}")
    if min_fs < p2.MIN_FS - 1e-6:
        warnings.append(f"smallest text {min_fs:.2f} pt < {p2.MIN_FS} pt")
    if route_info["stop_drawn"] != stop:
        warnings.append(f"end-of-rerun square drawn {route_info['stop_drawn']} but legend planned {stop}")
    checks.pop("badges", None)
    lay = legend_columns(fig, Lg, FIG_W * 72.0, keys)
    checks["legend_rows"], checks["legend_slot_group"] = lay["rows"], lay["slot_group"]
    checks["legend_slot_text"] = Lg["legend"][SLOT_KEY[mode]]
    checks["strip_pair_lines"] = pairs
    checks.update(edge_marks=edge, stop_drawn=route_info["stop_drawn"], start_label=route_info["start_label"],
                  leader_crossings=len(crossings), overlaps=len(overlaps), route_note_m=route_d,
                  nomap_blocks=[[a_, b_] for a_, b_, _ in tl.blocks])
    return {"files": files, "size_in": (FIG_W, round(height, 3)), "min_font_pt": round(min_fs, 2),
            "claims": claims, "warnings": warnings, "checks": checks, "synthetic_timeline": tl.synthetic,
            "caption_chars": len(caption)}


def warm_sentence(tl: Optional[tp.Timeline], lang: str, kind: str = "page") -> str:
    """The page caption's sentences on the compressed spans of the step axis ('' when the axis is linear)."""
    if tl is None:
        return ""
    out = ""
    if tl.compressed:
        w = int(tl.warmup_end())
        out += CAPTION_WARM[kind][lang].format(w=w, w1=w - 1)
    if tl.blocks:
        spans = ("、" if lang == "zh" else ", ").join(f"{int(a)}–{int(b)}" for a, b, _ in sorted(tl.blocks))
        out += CAPTION_NOMAP["page"][lang].format(pct=int(round(100 * tp.ready_cover(tl))), spans=spans)
    return out


def warm_overview(timelines: Sequence[tp.Timeline], cats: Sequence[str], lang: str) -> str:
    """The overview caption's clause on the compressed spans, per case: all warm-ups compressed, some (naming
    which are drawn to scale), none; and the cases with compressed no-map spans."""
    join = " and " if lang == "en" else "、"
    comp = [c for tl, c in zip(timelines, cats) if tl.compressed]
    flat = [c for tl, c in zip(timelines, cats) if not tl.compressed]
    out = ""
    if comp and not flat:
        out = CAPTION_WARM["overview"][lang]
    elif comp:
        out = CAPTION_WARM["overview_some"][lang].format(comp=join.join(comp), flat=join.join(flat))
    blocks = [c for tl, c in zip(timelines, cats) if tl.blocks]
    if blocks:
        out += CAPTION_NOMAP["overview"][lang].format(cases=join.join(blocks))
    return out


def route_sentence(b: bd.Bundle, lang: str, d: Optional[float]) -> str:
    """The route-note sentence of the page caption ('' when the route is drawn large enough to see)."""
    if d is None:
        return ""
    text = LABELS[lang]["route_note"].format(d=d)
    return f" The {text} (noted under the map)." if lang == "en" else f"{text}（见图下注）。"


def page_caption(b: bd.Bundle, lang: str, claims: Sequence[str], void, edge_marks: bool,
                 synthetic_tl: bool, stride: int = 1, tl: Optional[tp.Timeline] = None,
                 route_d: Optional[float] = None, mode: str = "all") -> str:
    """``mode``: the timeline's mark plan (``tp.MarkPlan.mode``), which the caption describes."""
    template = CAPTION_PAGE[lang] if b.keys else CAPTION_NO_KEYS[lang]
    pairs = order_pairs(b.keys, lang)
    order = CAPTION_ORDER[lang]["page"].format(pairs=pairs) if pairs else ""
    parts = [CAPTION_SYNTH_TL[lang] if synthetic_tl else "", fb.void_sentence(void, lang),
             template.format(ep=b.episode_id, scene=b.scene_id, cat=b.category,
                             cat_name=fb.CATEGORY_NAMES[lang][b.category], nk=len(b.keys),
                             key_rules=fb.key_rules_sentence(b.keys, lang) if b.keys else "",
                             stride=stride_sentence(lang, stride), slots=CAPTION_SLOTS[lang][mode],
                             warm=warm_sentence(tl, lang), order=order, route=route_sentence(b, lang, route_d)),
             smooth_sentence(lang) if b.keys or (tl is not None and tl.R) else ""]  # else no heat field is drawn
    if edge_marks:
        parts.append(CAPTION_EDGE[lang])
    parts += list(claims)
    parts.append(fb.fidelity_sentence(b, lang))
    if b.predicate_holds_on_rerun is False:
        parts.append(LABELS[lang]["predicate_fails"].format(cat=b.category) + ("." if lang == "en" else "。"))
    sep = " " if lang == "en" else ""
    return sep.join(p for p in parts if p)


CAPTION_EDGE = {
    "en": ("Directions more than 45° above or below the horizon are drawn as small triangles on the strip's edge: "
           "open blue = a true past direction, dark orange = a predicted peak."),
    "zh": "偏离水平方向超过 45° 的方向画成条带边缘的小三角：空心蓝 = 历史帧的真实方向，深橙 = 预测峰值。",
}


# --------------------------------------------------------------------------- #
# Overview figures
# --------------------------------------------------------------------------- #
OV = {"k_w": 1.08, "k_gap": 0.1, "left_gap": 0.13, "route_w": 1.33, "tl_ylab": 0.27, "badges": 0.12,
      "fut": 0.40, "gap": 0.1, "ticks": 0.13, "block_gap": 0.08, "head_row": 0.15, "min_hist": 0.72,
      "row_cap": 0.085}


def make_overview(bundles: Sequence[bd.Bundle], timelines: Sequence[tp.Timeline], out_dir: Path, group: str,
                  lang: str, verdicts: Optional[dict], topdown_root, keys: Sequence[str] = fb.MAIN_KEYS,
                  void: Optional[Sequence[str]] = None) -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    k_w, k_gap = OV["k_w"], OV["k_gap"]
    x_k = FIG_W - len(keys) * k_w - (len(keys) - 1) * k_gap
    left_w = x_k - OV["left_gap"]
    fig = plt.figure(figsize=(FIG_W, 10.0))
    chosen = [fb.main_keys_of(b, keys) for b in bundles]
    heads = []  # per block: (outcome on the title line?, outcome lines below it, instruction lines)
    for b in bundles:
        cat = b.membership(main=True)["category"]
        title = L["title"].format(cat=cat, name=fb.CATEGORY_NAMES[lang][cat])
        tw = cd.text_width_pt(fig, title, FS["head"] + 0.4, fontweight="bold") + 8.0
        out = outcome_line(b, L)
        inline = tw + cd.text_width_pt(fig, out, FS["small"]) <= left_w * 72.0
        heads.append((inline, [] if inline else wrap_ink(fig, out, FS["small"], left_w * 72.0),
                      wrap_ink(fig, L["instruction"].format(text=" ".join(b.instruction.split())), FS["small"],
                               left_w * 72.0, fontstyle="italic")))
    n_instr = max(len(h[1]) + len(h[2]) for h in heads)
    s2_n = max([len(s2_lines(fig, ks, L, k_w * 72.0, FS["small"], 2)) for ch in chosen for ks in ch] or [1])
    geoms = [Geom(k_w, captions=False, ticks=(i == len(bundles) - 1), badge_h=0.0, s2_h=0.12, gap=0.033,
                  chips_h=0.13, thumb_gap=0.018, s2_lines=s2_n, row_caps=(i == 0), cap_h=OV["row_cap"])
             for i in range(len(bundles))]
    ylab_w = max(cd.text_width_pt(fig, t, p2.MIN_FS) for t in list(L["y_hist"]) + list(L["y_fut"]) + [L["turns"]])
    tl_x = OV["route_w"] + max(OV["tl_ylab"], (ylab_w + 9.0) / 72.0)
    tl_w = left_w - tl_x
    badges_h = max(prepare_timeline(fig, tl, tl_w, "overview", OV["badges"]) for tl in timelines)
    left_top = 0.15 + LINE * n_instr + 0.03  # title line + instruction lines
    left_h = left_top + timeline_height(OV["min_hist"], OV["gap"], OV["fut"], badges_h, HIST_TITLE_H) + OV["ticks"]
    block_h = [max(g.height, left_h) for g in geoms]
    h_hists = [OV["min_hist"]] * len(geoms)  # the same timeline in every block (a taller key column leaves air below)
    route_hs = [timeline_height(hh, OV["gap"], OV["fut"], badges_h, HIST_TITLE_H) for hh in h_hists]
    stops = [p2.stop_shown(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                           OV["route_w"], rh) for b, rh in zip(bundles, route_hs)]
    stacked = []  # (category, stride) of the rows whose marks are stacked: their own legend row
    for b, tl in zip(bundles, timelines):
        plan = tp.mark_plan(tl, tl_w * 72.0, tp.OVERVIEW_SLOTS)
        if plan.mode == "stacked":
            stacked.append((b.membership(main=True)["category"], plan.stride))
    lkeys = legend_keys_for(any(stops), overview=True) + (SLOT_EXTRA if stacked else ())
    Lg = legend_labels(L, {"slots_stacked_ov": stacked_overview_text(L, lang, stacked)} if stacked else None)
    legend_h = legend_height(fig, Lg, FIG_W * 72.0, lkeys)
    banner = wrap_ink(fig, L["void"].format(reasons="; ".join(void)), FS["small"], FIG_W * 72.0) if void else []
    y0 = LINE * len(banner) + OV["head_row"]
    height = (y0 + sum(block_h) + OV["block_gap"] * (len(bundles) - 1) + 0.07 + legend_h + 0.03
              + notes_height(fig, FIG_W, L) + 0.02)
    fig.set_size_inches(FIG_W, height)
    page = fb.Page(fig, FIG_W, height)
    for j, line in enumerate(banner):
        page.text(0.0, LINE * (j + 0.5), line, ha="left", va="center", fontsize=FS["small"], fontweight="bold",
                  color="#d03b3b")
    # column heads
    yh = LINE * len(banner)
    for xh, text in ((0.0, L["col_route"]), (tl_x, L["col_timeline"])):
        page.text(xh, yh + 0.06, text, ha="left", va="center", fontsize=FS["head"], fontweight="bold",
                  color=style.INK)
    for c, k in enumerate(keys):
        page.text(x_k + c * (k_w + k_gap), yh + 0.06, L["col_key"].format(k=k), ha="left", va="center",
                  fontsize=FS["head"], fontweight="bold", color=style.INK)
    y = y0
    edge = 0
    checks = []
    synth = False
    stop_drawn = []
    for i, (b, tl, instr, h, g, ch, h_hist, route_h) in enumerate(zip(bundles, timelines, heads, block_h, geoms,
                                                                      chosen, h_hists, route_hs)):
        cat = b.membership(main=True)["category"]
        synth = synth or tl.synthetic or b.synthetic
        title = L["title"].format(cat=cat, name=fb.CATEGORY_NAMES[lang][cat])
        tax = page.pt_axes(0.0, y, left_w, 0.15)
        tax.text(0.0, 5.4, title, ha="left", va="center", fontsize=FS["head"] + 0.4, fontweight="bold",
                 color=style.INK)
        inline, out_lines, instr_lines = instr
        if inline:
            xo = cd.text_width_pt(fig, title, FS["head"] + 0.4, fontweight="bold") + 8.0
            tax.text(xo, 5.4, outcome_line(b, L), ha="left", va="center", fontsize=FS["small"], color=style.INK)
        for j, line in enumerate(out_lines):
            page.text(0.0, y + 0.15 + LINE * (j + 0.5), line, ha="left", va="center", fontsize=FS["small"],
                      color=style.INK)
        for j, line in enumerate(instr_lines, start=len(out_lines)):
            page.text(0.0, y + 0.15 + LINE * (j + 0.5), line, ha="left", va="center", fontsize=FS["small"],
                      color=style.INK_2, fontstyle="italic")
        yt = y + left_top
        info = draw_timeline(page, tl_x, yt, tl_w, h_hist, OV["gap"], OV["fut"], tl, L, badges_h=badges_h,
                             fut_title=True, warmup_label=True, title_h=HIST_TITLE_H, slots=tp.OVERVIEW_SLOTS)
        info.pop("badges", None)
        rinfo = draw_route(page, 0.0, yt, OV["route_w"], route_h, b, fb.resolve_level(b, topdown_root), L,
                           labels=[k.label for k in ch])
        stop_drawn.append(rinfo["stop_drawn"])
        rd = p2.route_note_m(b.xz("route_xz"), b.xz("reference_path_xz"), b.goal_xz, float(b.goal_radius_m),
                             OV["route_w"], route_h)
        if rd is not None:  # under the map, in the step-axis row of the timeline beside it
            page.text(0.0, yt + route_h + OV["ticks"] * 0.5, L["route_note"].format(d=rd), ha="left",
                      va="center", fontsize=FS["small"], color=style.INK_2, fontstyle="italic")
        info["route_note_m"] = rd
        for c, ks in enumerate(ch):
            edge += draw_key_column(page, x_k + c * (k_w + k_gap), y, g, ks, L, first=False,
                                    row_labels=L["row_labels"] if (i == 0 and c == 0) else None)["edge"]
        if not ch:
            page.text(x_k, y + 0.2, L["no_keys"], ha="left", va="center", fontsize=FS["small"], color=style.INK_2,
                      fontstyle="italic")
        checks.append({"ep_key": b.ep_key, **info, "stop_drawn": rinfo["stop_drawn"],
                       "start_label": rinfo["start_label"]})
        y += h + OV["block_gap"]
        if i < len(bundles) - 1:
            page.rule(y - OV["block_gap"] / 2)
    y += 0.07 - OV["block_gap"]
    page.rule(y - 0.035)
    y += draw_legend(page, 0.0, y, FIG_W, Lg, lkeys) + 0.03
    draw_notes(page, 0.0, y, FIG_W, L)

    cats = [b.membership(main=True)["category"] for b in bundles]
    M = fb.MAIN_TEXT[lang]
    sep = " " if lang == "en" else ""
    O = CAPTION_ORDER[lang]
    order = "".join(O["case"].format(cat=cat, pairs=order_pairs(b.keys, lang))
                    for b, cat in zip(bundles, cats) if order_pairs(b.keys, lang))
    warm = warm_overview(timelines, cats, lang)
    caption = CAPTION_OVERVIEW[lang].format(
        group=M["group"][group], cats=(", " if lang == "en" else "、").join(cats),  # names: in the block titles
        labels=(" and " if lang == "en" else "、").join(keys), warm=warm,
        exceptions=merged_exceptions(bundles, chosen, cats, lang, branches="nonreg").strip(), order=order)
    strides: Dict[int, List[str]] = {}  # stacked rows, by their stride
    for b, c in zip(bundles, checks):
        if c.get("mode") == "stacked":
            strides.setdefault(int(c.get("marker_stride", 1)), []).append(b.membership(main=True)["category"])
    drawn = [(b.membership(main=True)["category"], int(c.get("marker_stride", 1))) for b, c in zip(bundles, checks)
             if c.get("mode") == "stacked"]
    warnings_ov = [f"stacked rows drawn {drawn} but the legend planned {stacked}"] if drawn != stacked else []
    parts = [CAPTION_SYNTH_TL[lang] if synth else "", fb.void_sentence(void, lang), caption,
             smooth_sentence(lang) if any(tl.R for tl in timelines) else ""] + \
        ([CAPTION_EDGE[lang]] if edge else []) + [stride_sentence(lang, k, c) for k, c in sorted(strides.items())]
    caption = re.sub(r" {2,}", " ", sep.join([p.strip() if lang == "en" else p for p in parts if p]
                                             + fb.claim_sentences(verdicts, lang)))
    min_fs = min_font_size(fig)
    outside = texts_outside(fig)
    overlaps = text_overlaps(fig)
    crossings = leader_crossings(fig)
    files = fb._save(fig, Path(out_dir) / f"main_{group}", lang, caption)
    warnings = fb.over_budget(height, OVERVIEW_MAX_H_IN, f"overview {group} [{lang}]") + warnings_ov
    if overlaps:
        warnings.append(f"overlapping texts [{lang}]: {overlaps}")
    if outside:
        warnings.append(f"texts outside the page: {outside}")
    if crossings:
        warnings.append(f"leaders crossing texts [{lang}]: {crossings}")
    if stop_drawn != stops:
        warnings.append(f"end-of-rerun squares drawn {stop_drawn} but legend planned {stops}")
    return {"files": files, "size_in": (FIG_W, round(height, 3)), "min_font_pt": round(min_fs, 2),
            "warnings": warnings, "checks": checks, "synthetic_timeline": synth, "caption_chars": len(caption),
            "legend_stacked_rows": stacked}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _sha256(path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True, help="records dir: <ep_key>_bundle.{json,npz} and <ep_key>.json")
    ap.add_argument("--timelines", required=True, help="records_v2 dir: <ep_key>_timeline.{json,npz}")
    ver = ap.add_mutually_exclusive_group(required=True)
    ver.add_argument("--metrics", help="metrics/metrics.json; its 'verdicts' decide the caption claims")
    ver.add_argument("--no-verdicts", action="store_true", help="no claims in captions (layout work)")
    ap.add_argument("--out-dir", required=True, help="figures_v2 dir (never the v1 figures dir)")
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=sorted(LABELS))
    ap.add_argument("--topdown-root", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="ep_keys to draw (default: every bundle)")
    ap.add_argument("--pages", action="store_true", help="episode pages")
    ap.add_argument("--overview", action="store_true", help="overview figures main_T / main_F")
    ap.add_argument("--allow-void", action="store_true")
    args = ap.parse_args(argv)
    if not (args.pages or args.overview):
        args.pages = args.overview = True
    out_dir = Path(args.out_dir)
    if out_dir.resolve().name == "figures":
        print("refusing to write into a v1 'figures' dir", file=sys.stderr)
        return 2

    paths = bd.find_bundles(args.records)
    if args.only:
        paths = [p for p in paths if p.name[:-len("_bundle.json")] in set(args.only)]
    if not paths:
        print("no bundles found", file=sys.stderr)
        return 2
    verdicts, void = None, []
    if args.metrics:
        verdicts, void = fb.load_metrics(args.metrics)
        if void and not args.allow_void:
            print(f"{args.metrics}: the batch is void ({'; '.join(void)}); no figures drawn", file=sys.stderr)
            return 3
    source = {"path": str(args.metrics), "sha256": _sha256(args.metrics), "verdicts": verdicts} if args.metrics \
        else None
    code = {name: _sha256(Path(__file__).with_name(name)) for name in ("fig_v2.py", "panels_v2.py", "timeline_panel.py",
                                                                     "panels.py", "bundle.py", "fig_behavior.py")}
    display = {"strip_smoothing_sigma_deg": p2.SMOOTH_DEG, "timeline_smoothing_sigma_deg": tp.TL_SMOOTH_DEG,
               "note": ("heat fields are drawn after a Gaussian blur of these sigmas (360-degree strips: bearing and "
                        "elevation; timeline rings: bearing; circular in bearing), each ring / strip rescaled to keep "
                        "its maximum; about 0.5-0.6 pt at print size.  Without it the decoder's 4-pixel grid pattern "
                        "and the cusps between neighbouring slots' peaks show as pinstripes / a checker texture.  "
                        "Display only: the dots mark the raw argmax; records_v2 and the bundles are unchanged.  "
                        "Every caption discloses it (smooth_sentence)."),
               "caption_sentence": {lang: smooth_sentence(lang) for lang in args.lang}}
    manifest = {"schema": MANIFEST_SCHEMA, "code_sha256": code, "verdicts_source": source, "display": display,
                "claims": {lang: fb.claim_sentences(verdicts, lang) for lang in args.lang},
                "void_batch": void or None, "pages": [], "overview": []}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.pages:
            manifest["pages"] = old.get("pages", [])
        if not args.overview:
            manifest["overview"] = old.get("overview", [])

    def sources(json_path: Path) -> dict:
        ep = json_path.name[:-len("_bundle.json")]
        tl = tp.timeline_path_for(args.timelines, ep)
        rec = Path(args.records) / f"{ep}.json"
        return {"ep_key": ep, "bundle": str(json_path), "timeline": str(tl), "record": str(rec),
                "sha256": {"bundle_json": _sha256(json_path), "bundle_npz": _sha256(bd.bundle_paths(json_path)[1]),
                           "timeline_json": _sha256(tl), "timeline_npz": _sha256(tp.timeline_paths(tl)[1]),
                           "record": _sha256(rec)}}

    status = 0
    if args.pages:
        for json_path in paths:
            src = sources(json_path)
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            entry = {**src, "category": meta.get("category"), "category_rank": meta.get("category_rank"),
                     "is_main": meta.get("is_main"), "files": [], "size_in": {}, "min_font_pt": {}, "warnings": [],
                     "checks": {}}
            for lang in args.lang:
                res = make_episode_page(json_path, src["timeline"], out_dir, lang=lang, verdicts=verdicts,
                                        topdown_root=args.topdown_root, void=void or None, record_path=src["record"])
                entry["files"] += res["files"]
                entry["size_in"][lang] = res["size_in"]
                entry["min_font_pt"][lang] = res["min_font_pt"]
                entry["warnings"] += [w for w in res["warnings"] if w not in entry["warnings"]]
                entry["checks"] = res["checks"]
                entry["synthetic_timeline"] = res["synthetic_timeline"]
                entry.setdefault("caption_chars", {})[lang] = res["caption_chars"]
                print(f"{src['ep_key']} [{lang}] {res['size_in']} min {res['min_font_pt']} pt "
                      f"caption {res['caption_chars']} chars -> {res['files'][1]}")
            for w in entry["warnings"]:
                print(f"  WARNING {src['ep_key']}: {w}", file=sys.stderr)
            manifest["pages"].append(entry)
    if args.overview:
        all_paths = bd.find_bundles(args.records)
        bundles = [bd.load_bundle(p) for p in all_paths if bd.is_main_case(json.loads(p.read_text("utf-8")))]
        for group, cats in fb.MAIN_GROUPS:
            members = sorted((b for b in bundles if b.membership(main=True)["category"] in cats),
                             key=lambda b: fb.CATEGORY_ORDER[b.membership(main=True)["category"]])
            if not members:
                continue
            tls = [tp.load_timeline(tp.timeline_path_for(args.timelines, b.ep_key),
                                    record_path=Path(args.records) / f"{b.ep_key}.json") for b in members]
            entry = {"group": group, "cases": [sources(b.path) for b in members], "files": [], "size_in": {},
                     "min_font_pt": {}, "warnings": []}
            for b, tl in zip(members, tls):
                entry["warnings"] += [f"{b.ep_key}: {w}" for w in tl.warnings + tp.check_against_bundle(tl, b)]
            for lang in args.lang:
                res = make_overview(members, tls, out_dir / "main", group, lang, verdicts, args.topdown_root,
                                    void=void or None)
                entry["files"] += res["files"]
                entry["size_in"][lang] = res["size_in"]
                entry["min_font_pt"][lang] = res["min_font_pt"]
                entry["warnings"] += [w for w in res["warnings"] if w not in entry["warnings"]]
                entry["checks"] = res["checks"]
                entry["synthetic_timeline"] = res["synthetic_timeline"]
                entry["legend_stacked_rows"] = res["legend_stacked_rows"]
                entry.setdefault("caption_chars", {})[lang] = res["caption_chars"]
                print(f"main_{group} [{lang}] {res['size_in']} min {res['min_font_pt']} pt "
                      f"caption {res['caption_chars']} chars -> {res['files'][1]}")
            for w in entry["warnings"]:
                print(f"  WARNING main_{group}: {w}", file=sys.stderr)
            manifest["overview"].append(entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=_json_default) + "\n",
                             encoding="utf-8")
    print(manifest_path)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
