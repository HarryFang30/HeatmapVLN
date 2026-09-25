#!/usr/bin/env python3
"""EXP-18 gallery figure: the predicted affordance map across tiers, at pre-registered (not hand-picked) episodes.

Two sets are drawn from one cases.json (orchestrator decision D8):

* the main-text gallery, tiers C, D, E (``MAIN_TIERS``: generalization), at
  most ``MAX_HEIGHT_IN["main"]`` (5.5 in) tall -- what ``make_gallery_figure``
  draws when given neither a variant nor tiers;
* the supplementary gallery, all five tiers A-E (``SUPP_TIERS``), at most
  ``MAX_HEIGHT_IN["supp"]`` (8.5 in) tall.

Layout (7.0 in wide; one row per tier, one column per percentile):

            10th percentile         50th percentile (median)      90th percentile
            lower error  ------------------------------------------------>  higher error
  C         scene · episode 993 · frame 44 of 73     (one header line per tile)
  unseen    +------+ episode median 1.3°
  (11       | disc | this frame 1.4°, PCK@8 7/7          (bold)
  scenes)   |      | always-behind guess:
  R2R ...   +------+ median 60°, PCK@8 2/7              (grey)
  elevation          (8) = previous frame (at the robot):   (notes: wrapped at clause breaks, a " · "
  −10° to +15°       not visible                            separator dropped at the break, never dropped)
                     predicted P(not visible) = 1.00
            lane: numbered blue badges at the true bearings
            RGB row: the four current views unrolled clockwise (+-10 deg), front framed
  ground truth ▬  ground-truth affordance map row (blue)
  prediction   ▬  predicted affordance map row (orange), x per predicted peak (3.8 pt, staggered up and
                  down when two would touch), numbered misses, orange numbers of note slots predicted visible
                  blue ticks at the true bearings (+ a lane for miss badges, only when needed)
            ...
            Front (input)    Right    Back    Left      (view names and bearings, last row only; zh: fig1's
            0°               −90°     180°    +90°       "前 · 模型输入" / "0°（正前方）", which fit a view)
  legend line 1 (``fig_case.draw_legend``); legend line 2: the framed front view is the model's current image input

The left column holds each tier's label (letter, name with its scene count --
tier B reads "held-out (4 scenes)" --, dataset, episodes, and in grey the
elevation window of the tier's map rows, D4) and, level with the
two map rows of the first column, the fixed row names "ground truth" /
"prediction" with a tiny colour key (``cd.gutter_row_label``, D3); the rows of
the three tiles of a tier are level, so the names serve the whole row.

Selection (docs/experiments/README.md EXP-18 "案例挑选规则", implemented by
``select_cases.py`` into cases.json ``gallery``): per tier, the episodes whose
per-episode median bearing error is nearest the 10th / 50th / 90th percentile
of that tier; the 90th is the pre-registered failure case.  Frame shown (D8, a
presentation choice): the scored frame whose median bearing error over its
visible past positions is closest to the episode's median (ties: the wider
ground-truth bearing span, then the earlier frame, ``typical_row``); the tile
prints "episode median X° · this frame Y°, PCK@8 a/b".

Shared conventions (``common_draw`` / ``fig_case`` / ``data``; the same as the
main case figure):

* blue = ground truth / past positions, orange = prediction, never swapped;
* D1: a slot is missed iff it fails joint PCK@8 (``CaseRow.misses``, the
  fields and rule of ``compute_metrics``); every tile's numbered misses equal
  its header's n - hits (``data.check_accounting`` raises otherwise);
* D2: a missed x carries its number in ink in a white disc ringed orange
  (``cd.miss_badge``), beside it or moved (inside the row, else to a lane under
  it) with a short ink leader; a dotted line joins the x to its true-bearing
  tick when they are < 45 deg apart and the line runs through no other x
  (``clear_connectors``);
* D4: one elevation window per tier row (``cd.elevation_window`` over the
  three rows shown): +-10 deg, widened just enough for every ground-truth and
  predicted peak of the row (at most +-45 deg), in square degrees (the rows
  use the bearing axis's degree scale); the caption states each tier's window;
* D5: notes (``fig_case.note_items``) wrap onto as many lines as they need
  and merge consecutive slots; a slot without a badge or a note raises; a
  note slot the model still predicts visible (P(not visible) <= 0.5) gets its
  number in orange at its predicted peak where no x is near (``plan_fp_tags``:
  no disc, no ring, so it never reads as a numbered miss);
* D6: wording (frame labels ``cd.frame_label``, "8 = previous frame (at the
  robot) ...", "always-behind guess: median .., PCK@8 ../..", P(not visible)).

The caption explains only the marks the figure draws (``caption_flags`` /
``marks_caption``): e.g. the sentence on misses the model calls not visible
appears only when a tile has one.

Figure policy (user decision, 2026-09-24): the figure shows the affordance map
only.  It never mentions poses, their sources or the pose-source ablation; the
prediction drawn is always the deployed model's (the dump's ``vo`` arm).
Honesty that remains: of the four current views only the framed front view is
marked as given to the model (the model also takes the past frames' front
images; the caption says so), the other three are display only; misses shown,
never hidden; the constant always-behind guess printed in every tile; the
caption says what the model saw of each tier's scenes in training (tier B: the
map head never trained on its 4 scenes, the image backbone did).

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_gallery --cases <cases.json> [--variant main|supp|both] [--tiers C,D,E]
      [--dumps-root DIR] [--topdown-root DIR] [--clip-root DIR] [--metrics DIR|metrics.json] [--lang en|zh]
      --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt (``--variant both``: <stem>_main* and <stem>_supp*).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from scripts.exp18 import common
from scripts.exp18 import geometry as geo
from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import data as dd
from scripts.exp18.figures import fig_case as fc
from scripts.exp18.figures import style

from matplotlib.patches import Rectangle  # noqa: E402  (matplotlib is configured in cd.setup)

TIER_ORDER = ("A", "B", "C", "D", "E")
MAIN_TIERS = ("C", "D", "E")  # main-text gallery: generalization (D8)
SUPP_TIERS = ("A", "B", "C", "D", "E")  # supplementary gallery: every tier (D8)
VARIANT_TIERS = {"main": MAIN_TIERS, "supp": SUPP_TIERS}
MAX_HEIGHT_IN = {"main": 5.5, "supp": 8.5}
DEFAULT_PERCENTILES = (10, 50, 90)
# scene counts fixed by the pre-registered tier table (README EXP-18), used when metrics.json is not at hand
PREREG_SCENES = {"A": 22, "B": 4, "C": 11, "D": 30, "E": 11}

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); every string on the figure comes from here
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        # tier -> (name with its scene count, dataset); episodes are appended from cases.json
        "tiers": {
            "A": ("training ({s} scenes)", "R2R"),
            "B": ("held-out ({s} scenes)", "R2R"),
            "C": ("unseen ({s} scenes)", "R2R val-unseen"),
            "D": ("cross-dataset ({s} scenes)", "HM3D"),
            "E": ("designed routes ({s} scenes)", "out-and-back, loop"),
        },
        "n_episodes": "{n} episodes",
        "n_routes": "{n} routes",
        "ranking": ("Episodes nearest the {pcts} percentile of their tier's per-episode median bearing "
                    "error (pre-registered rule, not hand-picked)"),
        "better": "lower error",
        "worse": "higher error",
        "percentile": {10: "10th percentile", 50: "50th percentile (median)", 90: "90th percentile"},
        "percentile_other": "{p} percentile",
        "head": "{scene} · {episode} · {frame}",
        "episode": "episode {ep}",
        "route": {"oab": "out-and-back {i}", "loop": "loop {i}"},
        "ep_median": "episode median {m:.1f}°",  # D8: "episode median X° · this frame Y°, PCK@8 a/b"
        "this_frame": "this frame {m:.1f}°, PCK@8 {h}/{n}",
        "d8_sep": " · ",
        "floor": "always-behind guess: median {m:.0f}°, PCK@8 {h}/{n}",  # D6
        "frame_na": "no visible past position",
        "same_as": "same episode as the {p}th percentile",
        "missing_tile": "episode not available\n{why}",
        "no_rgb": "images not available",
        "legend_input": ("of the four current views, only the framed front view is given to the model; right, back "
                         "and left are shown for reference"),  # D6
        # the bearing axis under the last row: a view is 37 pt wide here, and fig1's "Front · model input" /
        # "0° (straight ahead)" (48 pt each) overhung the strip and ran into "Right"; the caption says 0° is
        # straight ahead
        "axis_views": ("Front (input)", "Right", "Back", "Left"),
        "axis_bearings": ("0°", "−90°", "180°", "+90°"),
        "elev": "elevation {lo} to {hi}",  # D4, under each tier's label
        "elev_split": ("elevation", "{lo} to {hi}"),
    },
    "zh": {
        "tiers": {
            "A": ("训练（{s} 个场景）", "R2R"),
            "B": ("留出（{s} 个场景）", "R2R"),
            "C": ("未见（{s} 个场景）", "R2R val-unseen"),
            "D": ("跨数据集（{s} 个场景）", "HM3D"),
            "E": ("设计路线（{s} 个场景）", "去而复返、绕圈"),
        },
        "n_episodes": "{n} 集",
        "n_routes": "{n} 条路线",
        "ranking": "每层按逐集方位误差中位数取最接近 {pcts} 分位的一集（预注册规则，非人工挑选）",
        "better": "误差低",
        "worse": "误差高",
        "percentile": {10: "10 分位", 50: "50 分位（中位数）", 90: "90 分位"},
        "percentile_other": "{p} 分位",
        "head": "{scene} · {episode} · {frame}",
        "episode": "第 {ep} 集",
        "route": {"oab": "去而复返 {i}", "loop": "绕圈 {i}"},
        "ep_median": "该集误差中位 {m:.1f}°",
        "this_frame": "本帧 {m:.1f}°，PCK@8 {h}/{n}",
        "d8_sep": " · ",
        "floor": "恒答正后方：中位 {m:.0f}°，PCK@8 {h}/{n}",
        "frame_na": "无可见历史位置",
        "same_as": "与 {p} 分位为同一集",
        "missing_tile": "该集不可用\n{why}",
        "no_rgb": "图像不可用",
        "legend_input": "当前四个视角中只有加框的前视图输入模型，右/后/左仅作展示",
        # fig1's labels fit a 37 pt view in Chinese ("前 · 模型输入" 34 pt, "0°（正前方）" 34 pt): kept (D6)
        "axis_views": None,
        "axis_bearings": None,
        "elev": "仰角 {lo} 至 {hi}",
        "elev_split": ("仰角", "{lo} 至 {hi}"),
    },
}

# What the model saw of each tier's scenes in training (README EXP-18 tier table); caption only.
TRAINING_TEXT = {
    "en": {
        "A": "A: the 22 R2R scenes that both the map head and the image backbone were trained on.",
        "B": ("B: 4 held-out R2R scenes; the head that predicts the affordance map never trained on them, but the "
              "image backbone did, so they test the head only (with 4 scenes, statistics over B are coarse)."),
        "C": "C: R2R val-unseen scenes, seen in training by neither the map head nor the image backbone.",
        "D": ("D: HM3D scenes; the head was trained on MP3D scenes only, and whether the image backbone saw HM3D is "
              "unknown."),
        "E": "E: out-and-back and loop routes designed in the unseen scenes of C.",
    },
    "zh": {
        "A": "A：预测 affordance map 的头与图像骨干训练时都用过的 22 个 R2R 场景。",
        "B": "B：4 个留出的 R2R 场景，预测 affordance map 的头从未在其上训练，但图像骨干训练时见过，因此只检验头本身（只有 4 个场景，B 层的统计较粗）。",
        "C": "C：R2R val-unseen 场景，头与图像骨干训练时都没见过。",
        "D": "D：HM3D 场景，头只在 MP3D 场景上训练过，图像骨干是否见过 HM3D 未知。",
        "E": "E：在 C 层未见场景中设计的去而复返与绕圈路线。",
    },
}
TIERS_TEXT = {
    "en": {"A": "training scenes (A)", "B": "held-out scenes (B)", "C": "unseen scenes (C)",
           "D": "a second dataset, HM3D (D)", "E": "designed out-and-back and loop routes (E)"},
    "zh": {"A": "训练场景（A）", "B": "留出场景（B）", "C": "未见场景（C）", "D": "跨数据集 HM3D（D）",
           "E": "设计路线（去而复返、绕圈；E）"},
}
CAPTION = {
    "en": (
        "Predicted affordance maps across {tiers_text}, at episodes chosen by a pre-registered rule, not by hand. "
        "Rows are tiers; the left column gives the tier, its number of scenes, the dataset, the number of "
        "episodes{routes_note} and the elevation its map rows show. "
        "Columns are the episodes whose median bearing error over all scored frames lies nearest the 10th, 50th and "
        "90th percentile of their tier; the 90th percentile is the high-error end (the pre-registered failure case). "
        "{totals}Each tile shows the scored frame whose median bearing error is closest to its episode's median (ties: the "
        "wider bearing spread of the visible past positions) and prints both, with the frame's joint PCK@8 "
        "(frames counted from 1). Left in each tile: local map (robot facing up); its rim is the bearing "
        "ring that the strip below unrolls clockwise from the front view's left edge (arrow in the first tile); "
        "dashed radii are the view seams{letters}. Blue lines run from the robot through each past position (dot; "
        "1 = oldest of the 8 queried) to its number on the rim. Strip: of the four current views, only the framed front view is "
        "given to the model; right, back and left are shown for reference (the model also receives the past frames' "
        "front images); under the last row each view is named over its bearing, 0° being straight ahead. Above the strip each past position's number sits at its true bearing, with a thin line "
        "down through the images. Below it, the ground-truth affordance map (blue) and the predicted affordance map "
        "(orange; "
        "the deployed model's output), on the degree scale of the bearing axis; {elev_window}; the images span "
        "±{el_rgb:g}°. Each slot's map is divided by its own peak (the prediction also multiplied by its predicted "
        "visibility), the maximum over slots is shown, and colour is linear in that value in both rows. {marks}"
        "For reference each tile gives the constant always-behind guess (back view, centre pixel) on the frame "
        "shown. {training}{missing}"
    ),
    "zh": (
        "{tiers_text}上的预测 affordance map；各集由预注册规则选出，而非人工挑选。每行一层，左栏注明层级、场景数、数据集、集数{routes_note}"
        "以及该层 affordance map 行所显示的仰角范围。"
        "三列分别是全部评分帧上方位误差中位数最接近该层 10、50、90 分位的一集；90 分位即误差高的一端（预注册的失败案例）。"
        "{totals}每格画方位误差中位数最接近该集中位数的评分帧（并列时取可见历史位置方位跨度更大者），并给出两者以及该帧的 joint PCK@8"
        "（帧号从 1 数起）。每格左侧：局部地图（机器人朝上）；圆周就是下方条带从前视左缘顺时针展开的方位环（第一格的箭头），"
        "虚线半径为视角分界{letters}。蓝线从机器人穿过每个历史位置（圆点；8 个查询中 1 = 最早）连到圆周上的编号。"
        "条带：当前四个视角中只有加框的"
        "前视图输入模型，右/后/左仅作展示（模型另外还接收历史帧的前视图）。条带上方每个历史位置的编号标在其真值方位上，细线向下穿过"
        "环视图。条带下方为真值 affordance map（蓝）与预测 affordance map"
        "（橙，即部署模型的输出），纵横同用方位轴的角度刻度；{elev_window}；环视图为 ±{el_rgb:g}°。每个槽位的图除以自身峰值"
        "（预测再乘以其预测可见概率），显示各槽位的最大值，两行都按该值线性着色。{marks}每格另给出作参照的恒答"
        "正后方基线（后视中心像素）在所画这一帧上的结果。{training}{missing}"
    ),
}
# The disc's sector letters (``cd.disc_sector_letters``): named in the caption; the clause on letters left out
# appears only when some tile left one out (``caption_flags["letters_out"]``).
LETTERS_TEXT = {"en": ("; F, R, B, L name the sectors", " (R, B or L is left out where the rim numbers fill its sector)"),
                "zh": ("；前、右、后、左标出各扇区", "（右、后、左所在扇区被编号占满时省略该字）")}
# The caption's account of the marks, sentence by sentence; a clause is included only when the figure draws what
# it describes (``caption_flags``), so the caption never explains a mark the reader cannot find.
MARKS_TEXT = {
    "en": {
        "x": "×: predicted peak of each slot visible in the ground truth{share}{stagger}. ",
        "stagger": "; × marks that would touch are staggered up and down, never along the bearing axis",
        "share_open": " (", "share_sep": "; ", "share_close": ")",
        "share_hits": "hits within {merge:g}° share one ×",
        "share_miss": "missed slots whose peaks coincide share one × and one number",
        "miss": ("A slot is missed when it fails joint PCK@8 (predicted view wrong, or peak more than 8 px of 64 from "
                 "the true peak in that view); its number, in a white disc ringed in orange, sits beside its own "
                 "×{moved}{conn}. "),
        "moved": " (joined to it by a short dark leader where it had to move{below})",
        "below": ", into a lane under the row where the row had no room",
        "conn": (", and a dotted line joins the × to the slot's true bearing when they are less than 45° apart and "
                 "the line would cross no other ×"),
        "pred_none": ("A missed slot the model calls not visible (P(not visible) > 0.5) has no ×; its orange-ringed "
                      "number is in the notes. "),
        "count": "Each tile's numbered misses equal its visible slots minus its PCK@8 hits. ",
        "no_miss": ("No slot shown fails joint PCK@8 (predicted view wrong, or peak more than 8 px of 64 from the true "
                    "peak in that view). "),
        "ticks": "Blue ticks under the prediction repeat the true bearings. ",
        "notes": ("Notes list past positions that no view shows ({kinds}) with the predicted P(not visible); they are "
                  "not scored and get no ×{fp}. "),
        "fp": ", but those the model predicts visible (P(not visible) ≤ 0.5) still show in the orange row{tagged}",
        "fp_tagged": "; where no × is near, their number is printed in orange at the predicted peak",
        "k_prev": "the previous frame at the robot",
        "k_at": "at the robot's current spot",
        "k_at_prev": "at the robot's current spot, like the previous frame",
        "k_out": "out of sight",
        "k_join": (", or ", "; or "),  # plain, and after a part that has a comma of its own
    },
    "zh": {
        "x": "×：真值可见的各槽位的预测峰值{share}{stagger}。",
        "stagger": "；会相互重叠的 × 上下错开，方位不动",
        "share_open": "（", "share_sep": "；", "share_close": "）",
        "share_hits": "命中的槽位 {merge:g}° 内共用一个 ×",
        "share_miss": "峰值重合的未命中槽位共用一个 × 和一个编号",
        "miss": ("joint PCK@8 不通过（预测视角错误，或峰值在该视角中距真值峰值超过 8 px（共 64 px））即为未命中：其编号写在"
                 "橙色描边的白色圆内，放在对应的 × 旁{moved}{conn}。"),
        "moved": "（需要挪开时用深色短线相连{below}）",
        "below": "，行内放不下时移到行下方",
        "conn": "；两者相距 45° 以内且连线不经过其他 × 时，再用虚线把 × 连到该槽位的真值方位",
        "pred_none": "模型判为不可见（预测不可见概率 > 0.5）的未命中槽位没有 ×，其橙色描边编号列在注释中。",
        "count": "每格的编号未命中数等于可见槽位数减去 PCK@8 命中数。",
        "no_miss": "所画各槽位的 joint PCK@8 均通过（预测视角正确，且峰值在该视角中距真值峰值不超过 8 px（共 64 px））。",
        "ticks": "预测行下方的蓝色短线重复真值方位。",
        "notes": "注释列出任何视角都看不到的历史位置（{kinds}）及其预测不可见概率；这些槽位不计分、没有 ×{fp}。",
        "fp": "，但被预测为可见（预测不可见概率 ≤ 0.5）的仍会出现在橙色行中{tagged}",
        "fp_tagged": "；附近没有 × 时，其编号以橙色标在预测峰值处",
        "k_prev": "与机器人重合的上一帧",
        "k_at": "与机器人当前位置重合",
        "k_at_prev": "与机器人当前位置重合，如上一帧",
        "k_out": "视线之外",
        "k_join": ("，或", "；或"),
    },
}
WINDOW_TEXT = {
    "en": {"one": "rows show elevation {lo} to {hi}", "tier": "{lo} to {hi} for {t}",
           "many": "rows show elevation {list} (±10°, widened per tier just enough to include every peak shown)",
           "join": ", "},
    "zh": {"one": "各行显示仰角 {lo} 至 {hi}", "tier": "{t} 层 {lo} 至 {hi}",
           "many": "各行显示的仰角范围为 {list}（默认 ±10°，每层只放宽到恰好容纳所画的全部峰值）",
           "join": "，"},
}
# Each tier's joint PCK@8 over all its scored past positions (metrics.json), so the three tiles of a tier can
# be read against the tier as a whole; caption only.
TOTALS_TEXT = {
    "en": {"sentence": ("Over all scored past positions, joint PCK@8 is {pred}, against {base} for the "
                        "always-behind guess. "),
           "item": "{v:.1f}% ({t})", "base_item": "{v:.1f}%"},
    "zh": {"sentence": "全部评分历史位置上的 joint PCK@8 为 {pred}，恒答正后方依次为 {base}。",
           "item": "{t} 层 {v:.1f}%", "base_item": "{v:.1f}%"},
}
MISSING_TEXT = {
    "en": " Tier{s} {tiers} not shown: no gallery picks.",
    "zh": "{tiers} 层没有画廊入选集，未画出。",
}

# The prediction drawn is always the deployed model's output (dump arm "vo").
ARM = fc.ARM

# --------------------------------------------------------------------------- #
# Geometry of the page (inches)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # 7.0; fc.Page places axes on this width
X_GRID = 0.64  # left edge of the first tile column; the tier labels and row names (D3) sit left of it
COL_GAP = 0.11
R_MARGIN = 0.02
W_TILE = (FIG_W - R_MARGIN - X_GRID - 2 * COL_GAP) / 3.0
PPD = W_TILE * 72.0 / 360.0  # points per degree on a tile's strip (both axes: square degrees)
HEAD_H = 0.135  # the tile's header line (scene · episode · frame); the disc's "F" sits just below it
HEAD_Y = 0.045  # the header's baseline-centre, below the tile's top
DISC = 0.65  # side of the disc axes (the disc itself is at most 0.74 of it); also the body's least height
DISC_X = 0.035  # the disc axes' left edge, relative to the tile's left edge
TEXT_DX = 0.79  # the text column starts here, relative to the tile's left edge (room for rim badges and "R")
TEXT_W_PT = (W_TILE - TEXT_DX) * 72.0
BODY_PAD = 0.02  # between the body (disc and text) and the lane
LANE_H = 0.112
EL_HEAT = cd.EL_DEFAULT  # default half window of the affordance map rows (D4: widened per tier row)
EL_RGB = 10.0  # RGB row: elevation +-10 deg (square degrees)
RGB_H = cd.strip_height_in(W_TILE, EL_RGB)
ROW_GAP = 0.016
TICK_LANE_H = 5.2 / 72.0  # under the prediction row: the true-bearing ticks (1..4 pt)
BELOW_LANE_H = cd.BELOW_LANE_H_PT / 72.0  # ... or the lane for miss badges that do not fit inside the row
TIER_GAP = 0.04
ELEV_GAP_PT = 2.0  # above the elevation window in a tier's label
TOP_H = 0.25
AXIS_H = 0.215
LEGEND_UP = 0.025  # fig1's legend line is drawn this much higher: its axes leave that much free above it
LEGEND_H = fc.LEGEND_H - LEGEND_UP
LEGEND2_H = 0.14
RING_W = 816  # ring columns per 360 deg (multiple of 8: exact roll); ~400 dpi at W_TILE
LINE_PT = 6.45  # baseline-to-baseline distance in the text column
NOTE_GAP_PT = 1.4  # extra space above the first note
FS = {"col": 6.5, "rank": 6.0, "letter": 9.0, "tier_name": 6.0, "tier_sub": 5.8, "head": 6.0, "line": 5.8,
      "note": cd.NOTE_FS, "axis": 5.8, "legend": 6.1, "missing": 6.0}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeStats:
    """The prediction over all scored frames of an episode (compute_metrics' per-episode numbers)."""

    median: float  # median bearing error over every visible (frame, slot)
    hits: int  # joint PCK@8 hits
    n: int  # visible (frame, slot) pairs
    n_frames: int  # scored frames


@dataclass
class Tile:
    tier: str
    percentile: int
    pick: dict
    dump: Optional[dd.Dump] = None
    row: Optional[dd.CaseRow] = None
    episode: Optional[EpisodeStats] = None
    span_deg: float = float("nan")
    views: Optional[np.ndarray] = None
    level: object = None
    same_as: Optional[int] = None  # percentile of an earlier tile showing the same episode
    problems: List[str] = field(default_factory=list)
    error: Optional[str] = None  # the tile cannot be drawn at all
    # the plan (``plan_tier``)
    head: str = ""
    lines: List[dict] = field(default_factory=list)
    text_h_pt: float = 0.0
    items: List[dict] = field(default_factory=list)
    marks: List[dict] = field(default_factory=list)
    placed: dict = field(default_factory=dict)
    connectors: List[list] = field(default_factory=list)  # dotted true-bearing connectors drawn (D2)
    connectors_dropped: int = 0  # ... left out because they would run through another x
    below: bool = False
    gt_strip: Optional[np.ndarray] = None
    pr_strip: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    numbered: List[int] = field(default_factory=list)
    disc_gap_pt: float = float("inf")  # clearance of the disc's labels from the text around it
    disc_gap_what: str = ""
    marks_touching: int = 0  # pairs of x marks that still touch after ``separate_marks``
    fp_tags: List[dict] = field(default_factory=list)  # orange numbers of note slots predicted visible
    fp_untagged: List[int] = field(default_factory=list)  # ... and those left untagged (an x is too close)
    letters: List[dict] = field(default_factory=list)  # the disc's sector letters (``cd.disc_sector_letters``)


@dataclass
class TierRow:
    tier: str
    g: dict
    tiles: List[Tile]
    win: Tuple[float, float] = (-cd.EL_DEFAULT, cd.EL_DEFAULT)
    body_h: float = DISC
    below: bool = False

    @property
    def heat_h(self) -> float:
        return cd.strip_height_in(W_TILE, self.win)

    @property
    def under_h(self) -> float:
        return BELOW_LANE_H if self.below else TICK_LANE_H

    @property
    def height(self) -> float:
        return HEAD_H + self.body_h + BODY_PAD + LANE_H + RGB_H + 2 * ROW_GAP + 2 * self.heat_h + self.under_h


def load_cases(cases_json) -> dict:
    if isinstance(cases_json, dict):
        return cases_json
    return json.loads(Path(cases_json).read_text(encoding="utf-8"))


def load_metrics(cases: dict, metrics=None) -> Optional[dict]:
    """metrics.json as a dict: ``metrics`` (dict, file or its directory), default the cases' ``metrics_dir``;
    None when neither can be read."""
    m = metrics
    if m is None and cases.get("metrics_dir"):
        m = Path(cases["metrics_dir"])
    if m is not None and not isinstance(m, dict):
        p = Path(m)
        p = p / "metrics.json" if p.is_dir() else p
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = None
    return m if isinstance(m, dict) else None


def scene_counts(cases: dict, metrics=None) -> Dict[str, int]:
    """Scenes per tier: metrics.json ``tiers.<t>.n_scenes`` (``metrics``: see ``load_metrics``), else the
    pre-registered counts."""
    out = dict(PREREG_SCENES)
    m = metrics if isinstance(metrics, dict) else load_metrics(cases, metrics)
    for t, blk in ((m or {}).get("tiers") or {}).items():
        if isinstance(blk, dict) and blk.get("n_scenes"):
            out[t] = int(blk["n_scenes"])
    return out


def tier_pck8(m: Optional[dict], tiers: Sequence[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    """(prediction, always-behind guess) joint PCK@8 of each tier over all its scored past positions, from
    metrics.json ``tiers.<t>.arms.{vo,floor}.joint_pck8.value`` (the numbers of the quantitative figure);
    None unless every tier has both."""
    out = {}
    for t in tiers:
        arms = ((((m or {}).get("tiers") or {}).get(t) or {}).get("arms")) or {}
        try:
            out[t] = (float(arms[ARM]["joint_pck8"]["value"]), float(arms["floor"]["joint_pck8"]["value"]))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in out[t]):
            return None
    return out


def totals_caption(pck: Optional[Dict[str, Tuple[float, float]]], lang: str) -> str:
    """The caption's sentence on each tier's joint PCK@8 over all its scored past positions ("" without it)."""
    if not pck:
        return ""
    T = TOTALS_TEXT[lang]
    tiers = list(pck)
    pred = [T["item"].format(v=100.0 * pck[t][0], t=t) for t in tiers]
    base = [T["base_item"].format(v=100.0 * pck[t][1], t=t) for t in tiers]
    return T["sentence"].format(pred=_join_list(pred, lang), base=_join_list(base, lang))


def _join_list(items: Sequence[str], lang: str) -> str:
    """"a, b and c" (en) / "a、b、c" (zh)."""
    if lang != "en":
        return "、".join(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def dump_path_for(pick: dict, dumps_root=None) -> Path:
    """``<dumps_root>/<tier>/<scene>/<clip>.npz``; without a root, the pick's recorded ``npz_path``."""
    tail = Path(pick["tier"]) / pick["scene"] / (pick["clip"] + ".npz")
    if dumps_root is not None:
        return Path(dumps_root) / tail
    if pick.get("npz_path"):
        return Path(pick["npz_path"])
    return common.EXP_ROOT / "dumps" / tail


def typical_row(dump: dd.Dump, arm: str = ARM) -> Tuple[dd.CaseRow, float, EpisodeStats]:
    """The frame a tile shows (D8), its ground-truth bearing span and the episode's statistics.

    The scored frame whose median bearing error over its GT-visible slots is
    closest to the episode's median over all (frame, slot) pairs; ties -> the
    wider GT bearing span (circular range), then the earlier frame.  Frames
    without a visible slot are never picked (unless no frame has one).
    """
    idx = dump.query_rows(arm)
    if not idx:
        raise ValueError(f"no scored rows with a prediction in {dump.path}")
    recs = [dd.case_row(dump, i) for i in idx]
    errs, hits, n = [], 0, 0
    for r in recs:
        e = r.arms[arm].err
        errs.append(e[r.visible & np.isfinite(e)])
        s = r.summary(arm)
        hits, n = hits + s["hits"], n + s["n"]
    errs = np.concatenate(errs) if errs else np.zeros(0)
    ep = EpisodeStats(median=float(np.median(errs)) if errs.size else float("nan"), hits=hits, n=n,
                      n_frames=len(recs))
    best, best_key, best_span = recs[0], None, 0.0
    for r in recs:
        s = r.summary(arm)
        span = float(geo.circular_range_deg(r.gt_bearing[r.visible])) if r.n_visible else 0.0
        if not s["n"] or not np.isfinite(s["median"]):
            continue
        key = (round(abs(s["median"] - ep.median), 9), -round(span, 9), r.index)
        if best_key is None or key < best_key:
            best, best_key, best_span = r, key, span
    return best, best_span, ep


def surround_views(dump: dd.Dump, frame: int, clip_root_override=None) -> np.ndarray:
    """``dd.surround_views``, also for a clip whose last chunk holds one frame.

    ``np.savez`` of a one-element object array of JPEG byte arrays collapses it
    to a 2-D object array of ints, which ``dd.surround_views`` cannot open
    (b8cTxDM8gDG/clip_700155 frame 64); the bytes are rebuilt here.
    """
    clip_dir = dd.resolve_clip_dir(dump, clip_root_override)
    try:
        return dd.surround_views(clip_dir, frame)
    except (OSError, TypeError, ValueError):
        path, j = dd._frame_index(str(clip_dir))[frame]
        with np.load(path, allow_pickle=True) as z:
            return np.stack([np.asarray(Image.open(io.BytesIO(np.asarray(z[f"rgb_{v}"][j], dtype=np.uint8).tobytes()))
                                        .convert("RGB")) for v in geo.VIEW_NAMES])


def prepare_tile(tier: str, pick: dict, dumps_root, topdown_root, clip_root_override) -> Tile:
    tile = Tile(tier=tier, percentile=int(pick.get("percentile", -1)), pick=pick)
    path = dump_path_for(pick, dumps_root)
    try:
        dump = dd.load_dump(path)
    except (OSError, ValueError, KeyError) as exc:
        tile.error = f"dump {path.name}: {type(exc).__name__}"
        tile.problems.append(f"cannot load {path}: {exc}")
        return tile
    if ARM not in dump.arms:
        tile.error = f"no prediction in {path.name}"
        return tile
    tile.dump = dump
    try:
        tile.row, tile.span_deg, tile.episode = typical_row(dump)
    except ValueError as exc:
        tile.error = str(exc)
        return tile
    # the dump must reproduce the per-episode numbers cases.json ranked on
    ep = tile.episode
    rec_med = pick.get("vo_bearing_err_median")
    if rec_med is not None and abs(float(rec_med) - ep.median) > 0.05:
        tile.problems.append(f"episode median {ep.median:.3f} deg differs from cases.json {float(rec_med):.3f} deg")
    if pick.get("vo_pck8") is not None and pick.get("n_visible_slots"):
        rec_hits = float(pick["vo_pck8"]) * int(pick["n_visible_slots"])
        if int(pick["n_visible_slots"]) != ep.n or abs(rec_hits - ep.hits) > 0.01:
            tile.problems.append(f"episode PCK@8 {ep.hits}/{ep.n} differs from cases.json "
                                 f"{rec_hits:.1f}/{pick['n_visible_slots']}")
    if pick.get("n_scored_rows") is not None and int(pick["n_scored_rows"]) != ep.n_frames:
        tile.problems.append(f"{ep.n_frames} scored frames, cases.json says {pick['n_scored_rows']}")
    try:
        tile.views = surround_views(dump, tile.row.frame, clip_root_override)
    except (OSError, KeyError, ValueError) as exc:
        tile.problems.append(f"no surround images: {exc}")
    try:
        tile.level = dd.topdown_level(dump, float(tile.row.cur_pos[1]), root=topdown_root)
    except (OSError, KeyError, ValueError) as exc:
        tile.problems.append(f"no top-down map: {exc}")
    return tile


def gallery_rows(cases: dict, tiers: Optional[Sequence[str]] = None
                 ) -> Tuple[List[Tuple[str, dict]], List[str], Tuple[int, ...]]:
    """(tier, gallery entry) for requested tiers with picks, requested tiers without, and the percentiles."""
    gal = cases.get("gallery") or {}
    wanted = [t for t in TIER_ORDER if tiers is None or t in tiers]
    present, missing = [], []
    percentiles: Optional[Tuple[int, ...]] = None
    for tier in wanted:
        g = gal.get(tier) or {}
        if g.get("status") == "ok" and g.get("picks"):
            present.append((tier, g))
            if percentiles is None:
                percentiles = tuple(int(p["percentile"]) for p in g["picks"])
        else:
            missing.append(tier)
    return present, missing, percentiles or DEFAULT_PERCENTILES


# --------------------------------------------------------------------------- #
# Text layout
# --------------------------------------------------------------------------- #
# Where a line may break, coarsest first: the level indexes BREAK_PENALTY.
_CLOSERS = "）：，、。；"  # full-width marks that never start a line
BREAK_PENALTY = (0.0, 1.0, 2.0, 3.0, 8.0)  # " · ", colon, semicolon, comma / 、, space or between CJK characters


def _is_cjk_char(ch: str) -> bool:
    return "　" <= ch <= "鿿" or "＀" <= ch <= "￯"


def break_points(text: str, dot_prefix: bool = False) -> List[Tuple[int, int, str, int]]:
    """Candidate line breaks of ``text`` as ``(end, start, prefix, level)``: a line may end at ``text[:end]`` and
    the next begin with ``prefix + text[start:]``.  Levels: 0 " · " (the separator is dropped: the break itself
    separates the clauses, and no line starts with "·"; ``dot_prefix`` keeps the older "· " line start), 1 after
    ": " / "：", 2 after "; ", 3 after ", " / "，" / "、" (the mark stays at the line end), 4 at a space or
    between two CJK characters.  Never inside a parenthesis ("(...)" or "（...）")."""
    out: List[Tuple[int, int, str, int]] = []
    depth, i, n = 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth = max(depth - 1, 0)
        if depth == 0:
            if text.startswith(" · ", i):
                out.append((i, i + 3, "· " if dot_prefix else "", 0))
                i += 3
                continue
            pair = text[i:i + 2]
            if pair in (": ", "; ", ", "):
                out.append((i + 1, i + 2, "", {": ": 1, "; ": 2, ", ": 3}[pair]))
                i += 2
                continue
            if ch in "：，、":
                j = i + 1
                while j < n and text[j] == " ":
                    j += 1
                if j < n:
                    out.append((i + 1, j, "", 1 if ch == "：" else 3))
                i = j
                continue
            if ch == " ":
                out.append((i, i + 1, "", 4))
            elif (_is_cjk_char(ch) and ch not in "（" and i + 1 < n and _is_cjk_char(text[i + 1])
                  and text[i + 1] not in _CLOSERS):
                out.append((i + 1, i + 1, "", 4))
        i += 1
    return out


def wrap_clauses(fig, text: str, width_pt: float, fs: float, weight: str = "normal", dot_prefix: bool = False,
                 first_pt: Optional[float] = None) -> List[str]:
    """Lines of ``text`` no wider than ``width_pt`` (the first line: ``first_pt``, e.g. beside a badge).

    Optimal, not greedy: the fewest lines, then the coarsest breaks
    (``BREAK_PENALTY``: " · " before a colon before a comma before a space).
    A break at " · " drops the separator (no line starts with "·"), so "8 =
    previous frame (at the robot): not visible · predicted P(not visible) =
    1.00" in the gallery's 90 pt column reads "8 = previous frame (at the
    robot):" / "not visible" / "predicted P(not visible) = 1.00", and "3 at
    the robot: not visible" stays on one line when it fits.  Never inside a
    parenthesis; nothing is dropped (a piece wider than the line with no break
    inside stays whole, as few such lines as possible).
    """
    first_pt = width_pt if first_pt is None else first_pt
    cache: Dict[str, float] = {}

    def w(s: str) -> float:
        if s not in cache:
            cache[s] = cd.text_width_pt(fig, s, fs, fontweight=weight)
        return cache[s]

    if w(text) <= first_pt:
        return [text]
    bps = break_points(text, dot_prefix)
    starts = [(0, "")] + [(b[1], b[2]) for b in bps]  # state a: a line begins here
    ends = [(b[0], b[3]) for b in bps] + [(len(text), -1)]  # line may end at break c (last: the text's end)
    memo: Dict[Tuple[int, bool], Tuple[tuple, List[str]]] = {}

    def best(a: int, first: bool) -> Tuple[tuple, List[str]]:
        key = (a, first)
        if key in memo:
            return memo[key]
        lim = first_pt if first else width_pt
        s, pre = starts[a]
        result = None
        for c in range(max(a, 0), len(ends)):
            e, lvl = ends[c]
            if e <= s:
                continue
            line = pre + text[s:e].strip()
            over = w(line) - lim
            if over > 0 and result is not None and c > a:
                break  # longer lines only get wider
            here = (1 if over > 0 else 0, max(over, 0.0), 1, BREAK_PENALTY[lvl] if lvl >= 0 else 0.0)
            if lvl < 0:
                cost, lines = here, [line]
            else:
                rest_cost, rest = best(c + 1, False)
                cost, lines = tuple(p + q for p, q in zip(here, rest_cost)), [line] + rest
            if result is None or cost < result[0]:
                result = (cost, lines)
        if result is None:  # nothing left after this break (cannot happen for a well-formed text)
            result = ((0, 0.0, 0, 0.0), [])
        memo[key] = result
        return result

    return [ln for ln in best(0, True)[1] if ln] or [text]


def note_lines(fig, item: dict, width_pt: float, fs: float = FS["note"]) -> List[dict]:
    """One note (``fc.note_items``) as text-column lines: its badge and first line, then the continuations
    from the column's left edge, as ``cd.wrap_notes`` lays out the case figure's note lines.  Nothing is
    dropped (D5)."""
    bw = cd.badge_width_pt(item["label"]) + cd.NOTE_BADGE_GAP_PT if item.get("label") else 0.0
    texts = wrap_clauses(fig, item["text"], width_pt, fs, first_pt=width_pt - bw)
    out = [dict(kind="note", item=dict(item, text=texts[0]), x=0.0, fs=fs, gap=0.0)]
    out += [dict(kind="text", text=t, x=0.0, fs=fs, color=style.INK_2, weight="normal", gap=0.0) for t in texts[1:]]
    return out


GLYPH_SLACK_PT = 0.6  # a text line's glyphs leave about this much of its pitch free above and below


def badge_pad_pt(line: dict) -> float:
    """Room a text-column line needs above and below it for its note badge beyond the line pitch (0: none).

    Badge heights at ``cd.BADGE_FS``, measured on the renders: a one-digit
    disc 7.4 pt, a pill ("1–3") 8.3 pt; a miss badge (``cd.miss_badge``) also
    has a white outline 1.2 pt wide on each side, drawn over the text.
    """
    item = line.get("item") if line.get("kind") == "note" else None
    if not item or not item.get("label"):
        return 0.0
    h = cd.badge_width_pt(item["label"]) if len(item["label"]) == 1 else 8.3
    if item.get("style") == "miss":
        h += 2.4
    return max(0.0, (h - LINE_PT) / 2.0 - GLYPH_SLACK_PT)


def episode_text(tile: Tile, L: dict) -> str:
    """"episode 993", or for a designed route "loop 2" / "out-and-back 1": the scene's routes counted from 1,
    like the frames (episode id ``exp18E_<scene>_loop1`` -> "loop 2"; the id itself is in the result's tiles)."""
    pick, dump = tile.pick, tile.dump
    ep = str(pick.get("episode_id") or dump.episode_id)
    m = re.match(rf"exp18E_{re.escape(dump.scene)}_(oab|loop)(\d+)$", ep)
    if m:
        return L["route"][m.group(1)].format(i=int(m.group(2)) + 1)
    return L["episode"].format(ep=ep)


def scene_text(scene: str) -> str:
    """HM3D scene ids carry the ".basis" file suffix; the figure shows the scene name."""
    return scene[:-len(".basis")] if scene.endswith(".basis") else scene


def plan_text(fig, tile: Tile, L: dict, fcL: dict, lang: str) -> None:
    """The tile's header line and text column (D8 numbers, always-behind guess, notes)."""
    r, ep = tile.row, tile.episode
    tile.head = L["head"].format(scene=scene_text(tile.dump.scene), episode=episode_text(tile, L),
                                 frame=cd.frame_label(r.frame, tile.dump.frame_count, lang))
    s, sf = r.summary(ARM), r.summary("floor")
    lines: List[dict] = []

    def add(text, fs, color, weight="normal", gap=0.0):
        for j, t in enumerate(wrap_clauses(fig, text, TEXT_W_PT, fs, weight)):
            lines.append(dict(kind="text", text=t, x=0.0, fs=fs, color=color, weight=weight,
                              gap=gap if j == 0 else 0.0))

    # D8: "episode median X° · this frame Y°, PCK@8 a/b", broken at its separator: the frame's part in bold
    add(L["ep_median"].format(m=ep.median), FS["line"], style.INK_2)
    if s["n"]:
        add(L["this_frame"].format(m=s["median"], h=s["hits"], n=s["n"]), FS["line"], style.INK, "bold")
    else:
        add(L["frame_na"], FS["line"], style.INK)
    if sf["n"]:
        add(L["floor"].format(m=sf["median"], h=sf["hits"], n=sf["n"]), FS["line"], style.MUTED)
    if tile.same_as is not None:
        add(L["same_as"].format(p=tile.same_as), FS["line"], style.INK_2)
    tile.items = fc.note_items(r, fcL, ARM)
    for j, it in enumerate(tile.items):
        nl = note_lines(fig, it, TEXT_W_PT)
        nl[0]["gap"] = NOTE_GAP_PT if j == 0 else 0.0
        lines += nl
    # a note's badge is taller than the line pitch (a pill, or a miss badge with its white outline): open the
    # gaps on both sides of its line so it never covers the text above or below it (the badges are drawn on top)
    for i in range(1, len(lines)):
        need = badge_pad_pt(lines[i - 1]) + badge_pad_pt(lines[i])
        lines[i]["gap"] = max(lines[i]["gap"], need)
    tile.lines = lines
    tile.text_h_pt = sum(ln["gap"] for ln in lines) + len(lines) * LINE_PT


# The gallery's x: 3.8 pt, not fig1's 4.4 pt.  Its rows are 0.41 pt per degree (about 10 pt tall), so a full-size
# x staggered clear of its neighbour reached the row's frame; the smaller x keeps ink and frame apart (its arms,
# with projecting caps, reach MARK_G / 2 + 0.67 pt from the centre).
MARK_G = 3.8
CONNECTOR_CLEAR_PT = MARK_G / 2 + 0.3  # a dotted connector passing closer than this to another x is dropped


def clear_connectors(row: dd.CaseRow, marks: Sequence[dict], win) -> Tuple[List[list], int]:
    """``cd.miss_connectors`` without the ones that would run through another x (D2 allows, not requires, them).

    Staggered neighbours put a hit's x right on the dotted line from a miss's
    true-bearing tick to the miss's own x (tier C P90: slot 4's line ended on
    slot 5's x and read as joining that hit to the tick).  The miss keeps its
    orange-ringed number at its own x and its blue lane badge at the true
    bearing.  Returns (connectors kept, number dropped).
    """
    kept, dropped = [], 0
    for own, mark in enumerate(marks):
        # one mark at a time, so each line is known to belong to ``mark`` (its end can lie nearer another x)
        for ln in cd.miss_connectors(row, [mark], win, PPD, mark_pt=MARK_G):
            (x0, y0), (x1, y1) = ln
            ts = np.linspace(0.0, 1.0, 60)
            qx, qy = (x0 + ts * (x1 - x0)) * PPD, (y0 + ts * (y1 - y0)) * PPD
            near = min((float(np.min(np.hypot(qx - m["x"] * PPD, qy - m["y"] * PPD)))
                        for j, m in enumerate(marks) if j != own), default=float("inf"))
            if near < CONNECTOR_CLEAR_PT:
                dropped += 1
            else:
                kept.append(ln)
    return kept, dropped


def draw_connectors(ax, lines: Sequence[list]) -> None:
    """The kept dotted connectors, styled as ``cd.draw_miss_connectors``."""
    for (x0, y0), (x1, y1) in lines:
        ax.plot([x0, x1], [y0, y1], color=style.INK_2, lw=0.55, ls=(0, (0.9, 1.1)), zorder=7.2,
                clip_on=False, solid_capstyle="round", dash_capstyle="round")


SEP_TOUCH_PT = MARK_G + 0.3  # two x marks closer than this in both directions print as one double x ...
SEP_TARGET_PT = MARK_G + 0.8  # ... so they are staggered this far apart where the row is tall enough,
SEP_EDGE_PT = MARK_G / 2 + 0.67 + 0.45  # ... keeping the ink 0.45 pt inside the row's frame
STAGGER_SAY_PT = 1.0  # the caption mentions staggering once an x sits this far from its predicted elevation
LETTER_CLEAR_PT = 1.0  # a disc's sector letter keeps this far from the text and lane badges around the disc
LETTER_LANE_FRAC = 0.5  # ... and may reach this far down into the lane under the body


def separate_marks(marks: List[dict], win, ppd: float = PPD) -> int:
    """Stagger touching x marks by about a full mark instead of ``cd.peak_marks``' 2.6 pt; returns the pairs
    whose arms still overlap (a row too low to hold them one above the other).

    At the gallery's scale (0.41 pt per degree) two x marks a few degrees
    apart overlapped into one double x under the shared stagger (tier E P90:
    the hits "5–6" and "7" near 180 deg).  Marks are taken in bearing order;
    a run of marks each closer than ``SEP_TOUCH_PT`` along the strip to the
    next gets levels like an interval colouring (each mark the first level
    whose last mark is far enough along the strip: two levels for a chain,
    three where three marks crowd together).  The levels are
    ``SEP_TARGET_PT`` apart (less where the row is lower), centred on the
    run's mean predicted elevation and kept ``SEP_EDGE_PT`` inside the row;
    which level goes on top is the order that keeps the marks nearest their
    predicted elevations.  Only the drawn height of an x moves: its bearing
    (x) never does, and ``y_peak`` keeps the predicted elevation; a mark
    that touches no other keeps ``cd.peak_marks``' place.
    """
    import itertools

    lo, hi = cd.elev_window(win)
    if len(marks) < 2:
        return 0
    y_min, y_max = lo + SEP_EDGE_PT / ppd, hi - SEP_EDGE_PT / ppd
    if y_max < y_min:
        y_min = y_max = (lo + hi) / 2.0
    order = sorted(range(len(marks)), key=lambda j: marks[j]["x"])
    runs, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if (marks[b]["x"] - marks[a]["x"]) * ppd < SEP_TOUCH_PT:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    for run in runs:
        if len(run) < 2:
            continue
        last_x: List[float] = []  # per level, the bearing of its last mark
        colour = []
        for j in run:
            x = marks[j]["x"]
            c = next((i for i, lx in enumerate(last_x) if (x - lx) * ppd >= SEP_TOUCH_PT), None)
            if c is None:
                c = len(last_x)
                last_x.append(x)
            last_x[c] = x
            colour.append(c)
        k = len(last_x)
        peaks = [marks[j].get("y_peak", marks[j]["y"]) for j in run]
        sep = min(SEP_TARGET_PT / ppd, (y_max - y_min) / (k - 1))
        span = sep * (k - 1)
        mid = float(np.clip(np.mean(peaks), y_min + span / 2, y_max - span / 2))
        heights = [mid + span / 2 - i * sep for i in range(k)]  # top first
        best = min(itertools.permutations(range(k)),
                   key=lambda perm: sum(abs(heights[perm[c]] - y) for c, y in zip(colour, peaks)))
        for j, c in zip(run, colour):
            marks[j]["y"] = heights[best[c]]
    touching = 0  # the arms of two x marks less than MARK_G - 0.4 apart in both directions overlap
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            if (abs(marks[i]["x"] - marks[j]["x"]) * ppd < MARK_G - 0.4
                    and abs(marks[i]["y"] - marks[j]["y"]) * ppd < MARK_G - 0.4):
                touching += 1
    return touching


def plan_marks(tile: Tile, win) -> None:
    """Heat strips, x marks and miss badges of the tile's prediction row (D1, D2; ``fig_case.plan_block``)."""
    r = tile.row
    tile.gt_strip = cd.heat_strip(dd.gt_composite(r), RING_W, win)
    tile.pr_strip = cd.heat_strip(dd.pred_composite(r, ARM), RING_W, win)
    tile.marks = cd.peak_marks(r, ARM, win, PPD, mark_pt=MARK_G)
    tile.marks_touching = separate_marks(tile.marks, win)
    if tile.marks_touching:
        tile.warnings.append(f"{tile.marks_touching} pair(s) of x marks still touch (row too low to stagger them)")
    heat = cd.heat_lookup(tile.pr_strip, win)
    conn, tile.connectors_dropped = clear_connectors(r, tile.marks, win)
    tile.connectors = conn
    placed = cd.place_miss_labels(tile.marks, PPD, win, heat=heat, below=False, lines=conn, mark_pt=MARK_G)
    tile.below = False
    if not placed["clean"]:
        placed = cd.place_miss_labels(tile.marks, PPD, win, heat=heat, below=True, lines=conn, mark_pt=MARK_G)
        tile.below = any(v["below"] for v in placed["labels"].values())
        if not placed["clean"]:
            tile.warnings.append("a miss badge overlaps or its leader crosses another mark")
    tile.placed = placed


FP_TAG_FS = cd.BADGE_FS  # a note slot's number at its predicted peak (orange, no disc, no x)
FP_TAG_COLOR = cd.MISS_RING  # the darkest prediction orange: reads on the orange map inside its white halo
FP_TAG_H_PT = 4.6  # drawn height of the digits with their halo
FP_TAG_CLEAR_PT = 1.2  # a tag keeps this far from every x and miss badge (else it is left out)
FP_TAG_SHIFT_PT = 3.0  # ... and may sit at most this far from its peak along the strip


def plan_fp_tags(fig, tile: Tile, win) -> None:
    """Orange numbers for the note slots the model still predicts visible (P(not visible) <= 0.5).

    Such a slot has a note (no view shows it) and no x (it is not scored), but
    its map shows in the prediction row weighted by 1 - P(not visible); without
    a label a reader cannot tell whose orange it is (tier B P90: slots 1-2,
    P(not visible) 0.01-0.03, a wide band at the right end of the Left view).
    Each such slot gets its number in orange (no disc, no ring: not a miss,
    D1/D2; orange: the prediction, D7) at its predicted peak, moved at most
    ``FP_TAG_SHIFT_PT`` along the strip and anywhere within the row's height;
    slots whose tags would touch share one ("1–2").  A tag that finds no spot
    ``FP_TAG_CLEAR_PT`` clear of every x and miss badge is left out
    (``fp_untagged``; its orange then sits in a blob an x already marks, e.g.
    tier E P90 slot 3 in the Back view).  The caption's sentence on such slots
    covers both.
    """
    r = tile.row
    p = r.arms[ARM]
    lo, hi = cd.elev_window(win)
    tile.fp_tags, tile.fp_untagged = [], []
    note_slots = sorted(k for it in tile.items if it["kind"] != "predicted_none" for k in it["slots"])
    fp = [k for k in note_slots if p.none_p[k] <= dd.NONE_THRESHOLD and p.peak_view[k] >= 0
          and np.isfinite(p.peak_bearing[k])]
    if not fp:
        return
    pts = sorted((float(cd.strip_x(p.peak_bearing[k])), float(p.peak_elev[k]), k) for k in fp)
    groups: List[List[tuple]] = []
    for q in pts:  # slots whose single-digit tags would touch share one tag
        if groups and (q[0] - groups[-1][-1][0]) * PPD < 2 * cd.text_width_pt(fig, "8", FP_TAG_FS, fontweight="bold"):
            groups[-1].append(q)
        else:
            groups.append([q])
    obst = []  # (x0, x1, y0, y1) in pt
    for m in tile.marks:
        h = MARK_G / 2 + FP_TAG_CLEAR_PT
        obst.append((m["x"] * PPD - h, m["x"] * PPD + h, m["y"] * PPD - h, m["y"] * PPD + h))
    for lab in (tile.placed.get("labels") or {}).values():
        hw = cd.badge_width_pt(lab["label"]) / 2 + 1.2 + FP_TAG_CLEAR_PT
        hh = cd.MISS_BH + 1.2 + FP_TAG_CLEAR_PT
        obst.append((lab["cx"] * PPD - hw, lab["cx"] * PPD + hw, lab["cy"] * PPD - hh, lab["cy"] * PPD + hh))
    for grp in groups:
        slots = [q[2] for q in grp]
        label = cd.slots_label(slots)
        w = cd.text_width_pt(fig, label, FP_TAG_FS, fontweight="bold") + 1.2
        x_pk = float(np.mean([q[0] for q in grp]))
        y_pk = float(np.mean([q[1] for q in grp]))
        x_lo, x_hi = w / 2 / PPD, 360.0 - w / 2 / PPD
        y_lo, y_hi = lo + FP_TAG_H_PT / 2 / PPD, hi - FP_TAG_H_PT / 2 / PPD
        spots = []
        for dx in np.arange(0.0, FP_TAG_SHIFT_PT + 0.01, 0.5):
            for sx in ((0.0,) if dx == 0 else (dx, -dx)):
                for dy in np.arange(0.0, (hi - lo) * PPD + 0.01, 0.5):
                    for sy in ((0.0,) if dy == 0 else (dy, -dy)):
                        spots.append((abs(sx) * 2.0 + abs(sy), sx, sy))  # prefer moving up/down to moving along
        spots.sort(key=lambda t: t[0])
        found = None
        for _, sx, sy in spots:
            cx = x_pk + sx / PPD
            cy = y_pk + sy / PPD
            if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
                continue
            box = (cx * PPD - w / 2, cx * PPD + w / 2, cy * PPD - FP_TAG_H_PT / 2, cy * PPD + FP_TAG_H_PT / 2)
            if not any(box[0] < o[1] and o[0] < box[1] and box[2] < o[3] and o[2] < box[3] for o in obst):
                found = (cx, cy, box)
                break
        if found is None:
            tile.fp_untagged += slots
            continue
        cx, cy, box = found
        obst.append(box)
        tile.fp_tags.append({"x": cx, "y": cy, "label": label, "slots": slots, "x_peak": x_pk, "y_peak": y_pk})


def draw_fp_tags(ax, tags: Sequence[dict]) -> None:
    """``plan_fp_tags``' orange numbers on the prediction row."""
    import matplotlib.patheffects as pe

    for t in tags:
        ax.text(t["x"], t["y"], t["label"], ha="center", va="center", fontsize=FP_TAG_FS, fontweight="bold",
                color=FP_TAG_COLOR, zorder=7.8, clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def plan_tier(fig, tier: str, g: dict, tiles: List[Tile], L: dict, fcL: dict, lang: str) -> TierRow:
    """One elevation window for the tier's three rows (D4), then each tile's text and marks."""
    rows = [t.row for t in tiles if t.error is None and t.row is not None]
    tr = TierRow(tier=tier, g=g, tiles=tiles)
    tr.win = cd.elevation_window(rows, ARM) if rows else (-cd.EL_DEFAULT, cd.EL_DEFAULT)
    body_pt = DISC * 72.0
    for t in tiles:
        if t.error is not None or t.row is None:
            continue
        plan_text(fig, t, L, fcL, lang)
        plan_marks(t, tr.win)
        plan_fp_tags(fig, t, tr.win)
        body_pt = max(body_pt, t.text_h_pt)
    tr.body_h = body_pt / 72.0
    tr.below = any(t.below for t in tiles)
    return tr


def elev_lines(fig, win, L: dict, width_pt: float, fs: float) -> List[str]:
    """The tier row's elevation window (D4) for its label: "elevation −33° to +10°" on one line when it fits,
    else the word over the range (never broken inside the range)."""
    lo, hi = cd.fmt_deg(win[0]), cd.fmt_deg(win[1])
    one = L["elev"].format(lo=lo, hi=hi)
    if cd.text_width_pt(fig, one, fs) <= width_pt:
        return [one]
    return [part.format(lo=lo, hi=hi) for part in L["elev_split"]]


def tier_label_lines(fig, tier: str, g: dict, n_scenes: Optional[int], L: dict, width_pt: float,
                     win=None) -> List[dict]:
    """The tier's label in the left column: letter; name with scene count (bold); the dataset; the number of
    episodes (routes for tier E) on its own line; with ``win``, the elevation window of the tier's map rows
    (D4; grey), so a taller row explains itself (tier B: −33° to +10°)."""
    name, dataset = L["tiers"].get(tier, (f"tier {tier}", ""))
    name = name.format(s=n_scenes if n_scenes else "?")
    sub = [dataset] if dataset else []
    if g.get("n_episodes") is not None:
        sub.append(L["n_routes" if tier == "E" else "n_episodes"].format(n=int(g["n_episodes"])))
    out = [dict(text=tier, fs=FS["letter"], color=style.INK, weight="bold", pitch=10.5)]
    for t in wrap_clauses(fig, name, width_pt, FS["tier_name"], "bold"):
        out.append(dict(text=t, fs=FS["tier_name"], color=style.INK, weight="bold", pitch=7.0))
    for part in sub:
        for t in wrap_clauses(fig, part, width_pt, FS["tier_sub"]):
            out.append(dict(text=t, fs=FS["tier_sub"], color=style.INK_2, weight="normal", pitch=6.8))
    if win is not None:
        for j, t in enumerate(elev_lines(fig, win, L, width_pt, FS["tier_sub"])):
            out.append(dict(text=t, fs=FS["tier_sub"], color=style.MUTED, weight="normal", pitch=6.8,
                            gap=ELEV_GAP_PT if j == 0 else 0.0))
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
BOLD_STROKE_PER_PT = 0.065  # fake-bold stroke (pt) per pt of font size, only if ``cd.CJK_FAKE_BOLD`` (off)


def _text_kw(text: str, color, weight: str = "normal", fs: float = 6.0) -> dict:
    """Bold CJK text is set in the regular weight (the CJK font has no bold face; ``cd.bold_effects`` adds no
    fake-bold stroke unless ``cd.CJK_FAKE_BOLD``).

    A bold line that mixes CJK and Latin ("本帧 1.3°，PCK@8 7/7") is set in the
    regular face throughout, so its Latin part is not heavier than the CJK
    beside it.
    """
    kw = dict(color=color, fontweight=weight)
    if weight == "bold" and cd.has_cjk(text):
        kw["fontweight"] = "normal"
        kw["path_effects"] = cd.bold_effects(text, color, stroke_pt=BOLD_STROKE_PER_PT * fs)
    return kw


def row_name(ax_row, text: str, kind: str, fs: float = cd.ROW_LABEL_FS, gap_pt: float = 4.0, key_w_pt: float = 19.0,
             key_h_pt: float = 2.3) -> list:
    """Row name ("ground truth" / "prediction") in the fixed left gutter with its tiny colour key (D3).

    ``cd.gutter_row_label``'s design (bold name right-aligned ``gap_pt`` left
    of the row, a ramp key under its end), with the pair centred on the row:
    the gallery's rows are about 8 pt tall at +-10 deg, and the shared
    placement (name above the row's middle) would put the name level with the
    row above.  Returns the artists (text, key axes).
    """
    fig = ax_row.figure
    box = ax_row.get_position()
    W, H = fig.get_size_inches() * 72.0
    x_right = box.x0 - gap_pt / W
    yc = (box.y0 + box.y1) / 2.0
    color, cmap = (cd.GT_INK, cd.GT_CMAP) if kind == "gt" else (cd.PRED_INK, cd.PRED_CMAP)
    t = fig.text(x_right, yc + 0.35 / H, text, ha="right", va="baseline", fontsize=fs, color=color,
                 fontweight="bold", path_effects=cd.bold_effects(text, color, stroke_pt=0.065 * fs))
    key = fig.add_axes([x_right - key_w_pt / W, yc - (1.45 + key_h_pt) / H, key_w_pt / W, key_h_pt / H])
    key.imshow(cmap(cd.HEAT_TOP * np.linspace(0.0, 1.0, 64))[None], aspect="auto", extent=(0, 1, 0, 1),
               interpolation="bilinear")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")
    return [t, key]


def text_box(a, rend):
    """Display box of a text artist: its bbox patch (badges) or its glyph extent."""
    if a.get_bbox_patch() is not None:
        a.update_bbox_position_size(rend)
        return a.get_bbox_patch().get_window_extent(rend)
    return a.get_window_extent(rend)


def min_gap_pt(texts, boxes, rend, dpi: float) -> Tuple[float, str, int]:
    """Smallest distance (pt; negative = overlap) between any non-empty text of ``texts`` and any display box:
    (distance, that text, the box's index)."""
    px = dpi / 72.0
    best = (float("inf"), "", -1)
    for a in texts:
        if not a.get_text().strip():
            continue
        b = text_box(a, rend)
        for i, k in enumerate(boxes):
            dx = max(k.x0 - b.x1, b.x0 - k.x1)
            dy = max(k.y0 - b.y1, b.y0 - k.y1)
            if max(dx, dy) / px < best[0]:
                best = (max(dx, dy) / px, a.get_text(), i)
    return best


def draw_missing_tile(page: fc.Page, x: float, y: float, h: float, text: str) -> None:
    ax = page.ax(x, y, W_TILE, h)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=cd.MAP_PLATE, ec=style.AXIS, lw=0.5))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=FS["missing"], color=style.INK_2, linespacing=1.3)


def draw_tier_label(page: fc.Page, y: float, tr: TierRow, n_scenes: Optional[int], L: dict) -> list:
    """The tier's label at the top of the left column; returns its text artists (the discs keep clear)."""
    fig = page.fig
    w_pt = (X_GRID - 0.07) * 72.0  # ends >= 2.9 pt left of where a displaced "L" of a first-column disc may go
    h = HEAD_H + tr.body_h
    ax = page.ax(0.02, y, X_GRID - 0.06, h)
    ax.set_xlim(0, w_pt)
    ax.set_ylim(h * 72.0, 0)
    ax.axis("off")
    arts = []
    y_pt = 0.0
    for j, ln in enumerate(tier_label_lines(fig, tr.tier, tr.g, n_scenes, L, w_pt, win=tr.win)):
        y_pt += ln.get("gap", 0.0)
        yc = y_pt + ln["pitch"] / 2 + (0.4 if j == 0 else 0.0)
        arts.append(ax.text(0, yc, ln["text"], ha="left", va="center", fontsize=ln["fs"], clip_on=False,
                            **_text_kw(ln["text"], ln["color"], ln["weight"], ln["fs"])))
        y_pt += ln["pitch"] + (0.8 if j == 0 else 0.0)
    return arts


def draw_text_column(page: fc.Page, x: float, y_body: float, tile: Tile, body_h: float) -> list:
    """The tile's text column right of the disc; returns its artists."""
    ax = page.ax(x + TEXT_DX, y_body, W_TILE - TEXT_DX, body_h)
    ax.set_xlim(0, TEXT_W_PT)
    ax.set_ylim(body_h * 72.0, 0)
    ax.axis("off")
    n0 = len(ax.texts)
    y_pt = 0.0
    for ln in tile.lines:
        y_pt += ln["gap"]
        yc = y_pt + LINE_PT / 2
        if ln["kind"] == "note":
            cd.draw_note(ax, ln["x"], yc, ln["item"], per_pt=1.0, fs=ln["fs"])
        else:
            ax.text(ln["x"], yc, ln["text"], ha="left", va="center", fontsize=ln["fs"], clip_on=False,
                    **_text_kw(ln["text"], ln["color"], ln["weight"], ln["fs"]))
        y_pt += LINE_PT
    return list(ax.texts[n0:])


def draw_strip(page: fc.Page, x: float, y_lane: float, tile: Tile, tr: TierRow, L: dict, first_col: bool,
               fcL: dict) -> Tuple[list, dict]:
    """Lane of ground-truth badges, RGB row, both affordance map rows, x marks, misses, ticks.

    Returns (lane badge artists, the axes by name)."""
    r, win = tile.row, tr.win
    lo, _ = win
    y_rgb = y_lane + LANE_H
    y_gt = y_rgb + RGB_H + ROW_GAP
    y_pr = y_gt + tr.heat_h + ROW_GAP
    ax_lane = page.ax(x, y_lane, W_TILE, LANE_H)
    ax_rgb = page.ax(x, y_rgb, W_TILE, RGB_H)
    ax_gt = page.ax(x, y_gt, W_TILE, tr.heat_h)
    ax_pr = page.ax(x, y_pr, W_TILE, tr.heat_h)

    if tile.views is not None:
        cd.draw_rgb_row(ax_rgb, cd.rgb_strip(tile.views, RING_W, EL_RGB), EL_RGB)
    else:
        cd.setup_strip_axes(ax_rgb, EL_RGB)
        ax_rgb.add_patch(Rectangle((0, -EL_RGB), 360, 2 * EL_RGB, fc=cd.MAP_PLATE, ec=style.AXIS, lw=0.5))
        ax_rgb.text(180, 0, L["no_rgb"], ha="center", va="center", fontsize=FS["note"], color=style.MUTED)
    cd.draw_heat_row(ax_gt, tile.gt_strip, win, cd.GT_CMAP)
    cd.draw_heat_row(ax_pr, tile.pr_strip, win, cd.PRED_CMAP)
    row_names = []
    if first_col:  # D3: row names in the fixed left gutter, once per tier row (the three tiles' rows are level)
        row_names += row_name(ax_gt, fcL["gt_row"], "gt")[:1]
        row_names += row_name(ax_pr, fcL["pred_row"], "pred")[:1]
    ax_lane.set_xlim(0, 360)
    ax_lane.set_ylim(0, 1)
    ax_lane.axis("off")

    # ground truth: numbered badges in the lane, guide through the RGB row, ticks under the prediction
    targets = np.array([float(cd.strip_x(r.gt_bearing[g[0]])) for g in r.groups])
    labels = [dd.group_label(g) for g in r.groups]
    xs = cd.dodge_1d(targets, [cd.badge_width_pt(s) / PPD for s in labels], 0.0, 360.0, 0.8 / PPD)
    y_badge = 0.58
    tick = cd.pts_to_data(ax_pr, 0.0, 1.0)[1]
    n0 = len(ax_lane.texts)
    for g, t, xb, lab in zip(r.groups, targets, xs, labels):
        k = g[0]
        col = cd.history_line_color(k)
        ax_lane.plot([xb, xb, t, t], [y_badge, 0.34, 0.12, 0.0], color=col, lw=0.5, zorder=3, clip_on=False,
                     solid_joinstyle="round")
        cd.history_badge(ax_lane, xb, y_badge, lab, k)
        ax_rgb.plot([t, t], [-EL_RGB, EL_RGB], color=col, lw=0.5, zorder=3)
        ax_pr.plot([t, t], [lo - 1.0 * tick, lo - 4.0 * tick], color=col, lw=0.8, zorder=3, clip_on=False,
                   solid_capstyle="butt")
    badges = list(ax_lane.texts[n0:])

    # predicted peaks and numbered misses (D1, D2)
    draw_connectors(ax_pr, tile.connectors)
    cd.draw_peak_marks(ax_pr, tile.marks, size=MARK_G)
    # the row's frame again, over the x marks' white halos (under badges, leaders and connectors)
    lo_, hi_ = cd.elev_window(win)
    ax_pr.add_patch(Rectangle((0, lo_), 360, hi_ - lo_, fill=False, ec=style.AXIS, lw=0.5, zorder=7.05,
                              clip_on=False))
    tile.numbered = cd.draw_miss_labels(ax_pr, tile.placed)
    draw_fp_tags(ax_pr, tile.fp_tags)
    return badges, {"lane": ax_lane, "rgb": ax_rgb, "gt": ax_gt, "pr": ax_pr, "row_names": row_names}


AXIS_CLEAR_PT = 1.0  # an axis label keeps this far inside its own view (it never overhangs the strip)


def axis_labels(L: dict, fcL: dict) -> Tuple[Sequence[str], Sequence[str]]:
    """(view names, bearings) under the last row: the gallery's own short forms where it has them (en), else
    fig1's (``fig_case`` labels, D6)."""
    return (L.get("axis_views") or fcL["views"]), (L.get("axis_bearings") or fcL["axis"])


def draw_bearing_axis(page: fc.Page, x: float, y_top: float, L: dict, fcL: dict) -> List[str]:
    """View names over their bearings under a strip, seams ticked; returns warnings (a label wider than its view).

    Every label is centred on its view and must fit inside it (``AXIS_CLEAR_PT``
    to spare): a 90 deg view is only 37 pt wide in a gallery tile, so the
    English front view reads "Front (input)" / "0°" (``axis_labels``); the
    front view's name is in ink, the others grey."""
    h = AXIS_H
    ax = page.ax(x, y_top, W_TILE, h)
    ax.set_xlim(0, 360)
    ax.set_ylim(h * 72.0, 0)
    ax.axis("off")
    for s in (0, 90, 180, 270, 360):
        ax.plot([s, s], [0.0, 2.4], color=style.AXIS, lw=0.5, clip_on=False, solid_capstyle="butt")
    views, bearings = axis_labels(L, fcL)
    view_pt = 90.0 * PPD
    warnings = []
    for v, (name, bearing) in enumerate(zip(views, bearings)):
        xc = 45.0 + 90.0 * v
        ax.text(xc, 3.4, name, ha="center", va="top", fontsize=FS["axis"], clip_on=False,
                color=style.INK if v == 0 else style.INK_2)
        ax.text(xc, 3.4 + FS["axis"] * 1.22, bearing, ha="center", va="top", fontsize=FS["axis"], clip_on=False,
                color=style.INK_2)
        for t in (name, bearing):
            w = cd.text_width_pt(page.fig, t, FS["axis"])
            if w > view_pt - 2 * AXIS_CLEAR_PT:
                warnings.append(f"axis label {t!r} is {w:.1f} pt wide, its view {view_pt:.1f} pt")
    return warnings


def ordinal(p: int, L: dict) -> str:
    """"10th", "1st", "22nd" (en); the bare number (zh, whose labels say "分位")."""
    if L is not LABELS["en"]:
        return str(p)
    return f"{p}" + ("th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th"))


def draw_column_headers(page: fc.Page, percentiles: Sequence[int], L: dict) -> None:
    """Ranking rule on one line; below it the column titles sit on an arrow from lower to higher error."""
    fig = page.fig
    pcts = " / ".join(ordinal(p, L) for p in percentiles)
    page.text(X_GRID, 0.065, L["ranking"].format(pcts=pcts), ha="left", va="center", fontsize=FS["rank"],
              color=style.INK_2)
    y_col = 0.18
    total = (FIG_W - R_MARGIN - X_GRID) * 72.0
    ax = page.ax(X_GRID, y_col - 0.05, FIG_W - R_MARGIN - X_GRID, 0.10)
    ax.set_xlim(0, total)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    w_b = cd.text_width_pt(fig, L["better"], FS["rank"])
    w_w = cd.text_width_pt(fig, L["worse"], FS["rank"])
    ax.text(0, 0, L["better"], ha="left", va="center", fontsize=FS["rank"], color=style.MUTED)
    ax.text(total, 0, L["worse"], ha="right", va="center", fontsize=FS["rank"], color=style.MUTED)
    ax.annotate("", xy=(total - w_w - 3.0, 0), xytext=(w_b + 3.0, 0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=style.AXIS, lw=0.7, mutation_scale=6.0, shrinkA=0,
                                shrinkB=0))
    for c, pct in enumerate(percentiles):
        xc = (c * (W_TILE + COL_GAP) + W_TILE / 2) * 72.0
        name = L["percentile"].get(pct) or L["percentile_other"].format(p=ordinal(pct, L))
        ax.text(xc, 0, name, ha="center", va="center", fontsize=FS["col"], zorder=3,
                bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="none"),
                **_text_kw(name, style.INK, "bold", FS["col"]))


def draw_legend2(page: fc.Page, y_top: float, L: dict) -> None:
    """Second legend line: a neutral swatch framed like the front view, and which current view the model is given."""
    ax = page.ax(0.0, y_top, FIG_W, LEGEND2_H)
    w_pt, h_pt = FIG_W * 72.0, LEGEND2_H * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.axis("off")
    y = h_pt * 0.55
    sw, sh = 16.0, 6.4
    x = 1.0
    ax.add_patch(Rectangle((x, y - sh / 2), sw, sh, fc="#d9d6ce", ec=style.INK, lw=1.0, zorder=2))
    ax.text(x + sw + 3.0, y, L["legend_input"], ha="left", va="center", fontsize=FS["legend"], color=style.INK)


def window_caption(tiers: Sequence[str], wins: Sequence[Tuple[float, float]], lang: str) -> str:
    """D4: the elevation window of each tier row, stated in the caption."""
    W = WINDOW_TEXT[lang]
    wins = [tuple(w) for w in wins]
    if len(set(wins)) == 1:
        return W["one"].format(lo=cd.fmt_deg(wins[0][0]), hi=cd.fmt_deg(wins[0][1]))
    parts = [W["tier"].format(lo=cd.fmt_deg(w[0]), hi=cd.fmt_deg(w[1]), t=t) for t, w in zip(tiers, wins)]
    return W["many"].format(list=_join_list(parts, lang) if lang == "en" else W["join"].join(parts))


def caption_flags(rows: Sequence[TierRow]) -> dict:
    """What the drawn tiles actually contain, for the caption's conditional clauses (``marks_caption``)."""
    f = dict(share_hits=False, share_miss=False, miss=False, leader=False, below=False, conn=False,
             pred_none=False, notes=set(), fp=False, fp_tagged=False, stagger=False, letters=False,
             letters_out=False)
    for tr in rows:
        for t in tr.tiles:
            if t.error is not None or t.row is None:
                continue
            f["letters"] |= any(d.get("mode") != "omitted" for d in t.letters)
            f["letters_out"] |= any(d.get("mode") == "omitted" for d in t.letters)
            for m in t.marks:
                f["stagger"] |= abs(m["y"] - m.get("y_peak", m["y"])) * PPD > STAGGER_SAY_PT
                if len(m["slots"]) > 1:
                    f["share_miss" if m.get("miss") else "share_hits"] = True
            if t.row.misses(ARM):
                f["miss"] = True
            for lab in (t.placed.get("labels") or {}).values():
                f["leader"] |= lab["leader"] is not None
                f["below"] |= bool(lab["below"])
            f["conn"] |= bool(t.connectors)
            f["fp"] |= bool(t.fp_tags or t.fp_untagged)
            f["fp_tagged"] |= bool(t.fp_tags)
            for it in t.items:
                if it["kind"] == "predicted_none":
                    f["pred_none"] = True
                else:
                    f["notes"].add(it["kind"])
    return f


def letters_caption(f: dict, lang: str) -> str:
    """The clause naming the disc's sector letters (with the letters-left-out note when a tile left one out)."""
    if not (f.get("letters") or f.get("letters_out")):
        return ""
    name, out = LETTERS_TEXT[lang]
    return name + (out if f.get("letters_out") else "")


def marks_caption(f: dict, lang: str) -> str:
    """The caption's sentences on the x marks, misses, ticks and notes: only what the figure draws (D1, D2, D5)."""
    M = MARKS_TEXT[lang]
    share = [M["share_hits"].format(merge=cd.MERGE_DEG)] if f["share_hits"] else []
    share += [M["share_miss"]] if f["share_miss"] else []
    out = M["x"].format(share=M["share_open"] + M["share_sep"].join(share) + M["share_close"] if share else "",
                        stagger=M["stagger"] if f.get("stagger") else "")
    if f["miss"]:
        moved = M["moved"].format(below=M["below"] if f["below"] else "") if f["leader"] else ""
        out += M["miss"].format(moved=moved, conn=M["conn"] if f["conn"] else "")
        out += M["pred_none"] if f["pred_none"] else ""
        out += M["count"]
    else:
        out += M["no_miss"]
    out += M["ticks"]
    kinds = f["notes"]
    if kinds:
        parts = []
        if "at_robot" in kinds:
            parts.append(M["k_at_prev"] if "previous" in kinds else M["k_at"])
        elif "previous" in kinds:
            parts.append(M["k_prev"])
        if "not_visible" in kinds:
            parts.append(M["k_out"])
        plain, after_comma = M["k_join"]
        text = parts[0]
        for p in parts[1:]:
            text += (after_comma if ("," in text or "，" in text) else plain) + p
        fp = M["fp"].format(tagged=M["fp_tagged"] if f["fp_tagged"] else "") if f["fp"] else ""
        out += M["notes"].format(kinds=text, fp=fp)
    return out


def page_height(rows: Sequence[TierRow]) -> float:
    n = max(len(rows), 1)
    return TOP_H + sum(tr.height for tr in rows) + (n - 1) * TIER_GAP + AXIS_H + LEGEND_H + LEGEND2_H


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def make_gallery_figure(cases_json, dumps_root=None, topdown_root=None, clip_root_override=None,
                        out_stem="gallery", lang: str = "en", tiers: Optional[Sequence[str]] = None,
                        metrics=None, variant: Optional[str] = None) -> dict:
    """Render one gallery from a cases.json (path or dict).

    ``variant`` "main" (tiers C, D, E) or "supp" (A-E) sets ``tiers`` and the
    height limit; ``tiers`` alone draws those tiers (with no height limit);
    neither: the main-text gallery (D8: a caller that names neither, such as
    make_all's fig2 job, gets tiers C, D, E, not all five).  ``metrics``:
    metrics.json (dict, file or its directory) for the scene counts and the
    caption's per-tier joint PCK@8; default the cases' ``metrics_dir``, else
    the pre-registered scene counts and no such sentence.
    Returns {"files", "tiers", "missing_tiers", "tiles", "size_in",
    "windows", "warnings", "height_limit_in", "notes_dropped", "min_font_pt",
    "caption_flags"}.
    ``warnings`` is the one list a caller needs (D5): layout problems (a miss
    badge touching another mark, a disc label close to text, a crowded
    legend, text under 5.5 pt, a figure over its height limit) and data
    problems (a dump that does not reproduce cases.json, a placeholder tile),
    each tile's tagged "tier C P90: ..." (the tiles also carry their own as
    ``tile_warnings`` / ``tile_problems``).  A widened elevation window (D4)
    is in ``windows`` and a miss lane under a row in the tile's
    ``miss_lane_below``: the rules working, not warnings.  ``notes_dropped``
    is always empty (a slot without a badge or a note raises, D5).
    """
    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    if variant is None and tiers is None:
        variant = "main"  # D8: the gallery of the main text (tiers C, D, E) unless asked for tiers or "supp"
    if variant is not None:
        if variant not in VARIANT_TIERS:
            raise ValueError(f"variant {variant!r}: one of {sorted(VARIANT_TIERS)}")
        tiers = VARIANT_TIERS[variant]
    L = LABELS[lang]
    fcL = fc.labels_for(lang)
    cases = load_cases(cases_json)
    present, missing, percentiles = gallery_rows(cases, tiers)
    if not present:
        raise ValueError(f"cases.json has no gallery picks for tiers {tiers or TIER_ORDER}")
    metrics_d = load_metrics(cases, metrics)
    n_scenes = scene_counts(cases, metrics_d)

    # ---- data
    grid: List[Tuple[str, dict, List[Tile]]] = []
    for tier, g in present:
        tiles, seen = [], {}
        by_pct = {int(pk["percentile"]): pk for pk in g["picks"]}
        for pct in percentiles:
            pk = by_pct.get(pct)
            if pk is None:
                t = Tile(tier=tier, percentile=pct, pick={}, error="no pick for this percentile")
            else:
                t = prepare_tile(tier, pk, dumps_root, topdown_root, clip_root_override)
                key = pk.get("clip_key")
                if key in seen:
                    t.same_as = seen[key]
                else:
                    seen[key] = pct
            tiles.append(t)
        grid.append((tier, g, tiles))

    # ---- plan every tier row before the page exists (heights depend on windows, notes, miss lanes)
    fig_m = plt.figure(figsize=(FIG_W, 2.0))
    rows = [plan_tier(fig_m, tier, g, tiles, L, fcL, lang) for tier, g, tiles in grid]
    plt.close(fig_m)
    # a widened elevation window (D4) and a miss lane under a prediction row are the rules working, not
    # problems: they are reported under "windows" and in each tile's "miss_lane_below", not as warnings
    warnings: List[str] = []

    height = page_height(rows)
    limit = MAX_HEIGHT_IN.get(variant) if variant else None
    if limit is not None and height > limit + 1e-6:
        warnings.append(f"figure is {height:.2f} in tall, over the {limit:g} in limit of the {variant} gallery")
    fig = plt.figure(figsize=(FIG_W, height))
    page = fc.Page(fig, height)
    rend = fig.canvas.get_renderer()
    draw_column_headers(page, percentiles, L)
    y = TOP_H
    layout = []
    for n, tr in enumerate(rows):
        layout.append({"tier": tr.tier, "y_in": round(y, 4), "height_in": round(tr.height, 4),
                       "body_in": round(tr.body_h, 4), "heat_in": round(tr.heat_h, 4), "under_in": round(tr.under_h, 4),
                       "window": list(tr.win)})
        if n > 0:  # hairline between tiers
            rule = page.ax(0.0, y - TIER_GAP / 2, FIG_W, 0.001)
            rule.axis("off")
            rule.axhline(0.5, color=style.GRID, lw=0.6)
        tier_arts = draw_tier_label(page, y, tr, n_scenes.get(tr.tier), L)
        y_body = y + HEAD_H
        y_lane = y_body + tr.body_h + BODY_PAD
        last = n == len(rows) - 1
        for c, t in enumerate(tr.tiles):
            x = X_GRID + c * (W_TILE + COL_GAP)
            if t.error is not None or t.row is None:
                draw_missing_tile(page, x, y, tr.height, L["missing_tile"].format(why=t.error))
                if last:
                    warnings += draw_bearing_axis(page, x, y + tr.height, L, fcL)
                continue
            head = page.text(x, y + HEAD_Y, t.head, ha="left", va="center", fontsize=FS["head"],
                             color=style.INK)
            w_head = head.get_window_extent(rend).width / fig.dpi
            if w_head > W_TILE + 0.01:
                t.warnings.append(f"header line runs {w_head - W_TILE:.2f} in past the tile")
            text_arts = draw_text_column(page, x, y_body, t, tr.body_h)
            badges, axes = draw_strip(page, x, y_lane, t, tr, L, first_col=(c == 0), fcL=fcL)
            if c == 0 and axes["row_names"]:  # the tier label (with its elevation lines) stays above the row names
                gap, what, _ = min_gap_pt(tier_arts, [text_box(a, rend) for a in axes["row_names"]], rend, fig.dpi)
                layout[-1]["label_to_row_names_pt"] = round(gap, 2)
                if gap < 2.0:
                    warnings.append(f"tier {tr.tier}: its label ({what!r}) is {gap:.1f} pt from the row names")
            badge_slots = [k for g in t.row.groups for k in g]
            note_slots = [k for it in t.items for k in it["slots"]]
            in_notes = [k for it in t.items if it["kind"] == "predicted_none" for k in it["slots"]]
            dd.check_accounting(t.row, ARM, badge_slots, note_slots, t.numbered, in_notes)  # D5: raises
            # ---- the disc: centred in the body, as large as its rim badges allow beside the text around it
            side = DISC
            ax_in = page.ax(x + DISC_X, y_body + (tr.body_h - side) / 2, side, side)
            if t.level is None:
                ax_in.axis("off")
                ax_in.text(0.5, 0.5, "—", ha="center", va="center", transform=ax_in.transAxes, color=style.MUTED)
            else:
                keep = [head] + text_arts + (tier_arts if c == 0 else [])
                keep = [a for a in keep + badges if a.get_text().strip()]
                boxes = [text_box(a, rend) for a in keep]
                rf, dx, ok = fc.fit_inset(ax_in, t.row, boxes)
                if not ok:
                    t.warnings.append("a rim badge of the local map touches nearby text")
                # a displaced sector letter may use the body's free room around the disc, never the text column
                # (and the top half of the lane below it, clear of the lane badges: room for a "B" under the rim)
                bounds = page.bbox(x - 0.01, y_body - 0.02, TEXT_DX - 0.025 + 0.01,
                                   tr.body_h + 0.02 + BODY_PAD + LETTER_LANE_FRAC * LANE_H)
                n_txt = len(ax_in.texts)
                # the letters also keep LETTER_CLEAR_PT clear of the header, text column, tier label and lane
                # badges (the shared keep-out pads each box by cd.LETTER_KEEPOUT_PT: the boxes are shrunk by the
                # difference, else "F" could not sit between the header and the rim and slid onto the scale bar)
                shrink = (cd.LETTER_KEEPOUT_PT - (LETTER_CLEAR_PT - 0.8)) * fig.dpi / 72.0  # + 0.8 pt in cd
                t.letters = fc.draw_inset(ax_in, t.level, t.dump, t.row, show_arrow=(n == 0 and c == 0), L=fcL,
                                          letters="outward", radius_frac=rf, letter_bounds=bounds,
                                          letter_keepout=[b.padded(-shrink) for b in boxes]) or []
                # the disc's own texts (rim badges, scale bar, sector letters) against the header, text column,
                # tier label and lane badges around it
                t.disc_gap_pt, what, i = min_gap_pt(ax_in.texts[n_txt:], boxes, rend, fig.dpi)
                t.disc_gap_what = f"{what!r} vs {keep[i].get_text()!r}" if i >= 0 else ""
                if t.disc_gap_pt < 1.0:
                    t.warnings.append(f"a label of the local map is {t.disc_gap_pt:.1f} pt from nearby text "
                                      f"({t.disc_gap_what})")
            if last:
                warnings += draw_bearing_axis(page, x, y_lane + LANE_H + RGB_H + 2 * ROW_GAP + 2 * tr.heat_h
                                              + tr.under_h, L, fcL)
        y += tr.height + TIER_GAP
    y_leg = y - TIER_GAP + AXIS_H
    warnings += fc.draw_legend(page, y_leg - LEGEND_UP, ARM, fcL)
    draw_legend2(page, y_leg + LEGEND_H, L)

    # nothing smaller than 5.5 pt at print size (the page is drawn at its print size)
    from matplotlib.text import Text

    sizes = [a.get_fontsize() for a in fig.findobj(Text) if a.get_visible() and a.get_text().strip()]
    min_font = float(min(sizes)) if sizes else float("nan")
    if min_font < 5.5 - 1e-6:
        warnings.append(f"smallest text is {min_font:.2f} pt (< 5.5 pt)")

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)

    shown = [tr.tier for tr in rows]
    tiers_text = TIERS_TEXT[lang]
    listed = [tiers_text[t] for t in shown]
    if lang == "en" and len(listed) > 1:
        joined = ", ".join(listed[:-1]) + (", and " if len(listed) > 2 else " and ") + listed[-1]
    else:
        joined = "、".join(listed) if lang == "zh" else ", ".join(listed)
    miss = [t for t in missing if t in TIER_ORDER]
    missing_text = ""
    if miss:
        sep = ", " if lang == "en" else "、"
        missing_text = MISSING_TEXT[lang].format(s="s" if len(miss) > 1 else "", tiers=sep.join(miss))
    training = (" " if lang == "en" else "").join(TRAINING_TEXT[lang][t] for t in shown)
    routes_note = ({"en": " (routes for E)", "zh": "（E 层为路线数）"}[lang] if "E" in shown else "")
    flags = caption_flags(rows)
    caption = CAPTION[lang].format(tiers_text=joined, routes_note=routes_note,
                                   totals=totals_caption(tier_pck8(metrics_d, shown), lang),
                                   elev_window=window_caption(shown, [tr.win for tr in rows], lang),
                                   el_rgb=EL_RGB, marks=marks_caption(flags, lang), training=training,
                                   missing=missing_text, letters=letters_caption(flags, lang))
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption + "\n", encoding="utf-8")
    files.append(cap_path)

    # every tile's warnings and data problems also go into the one top-level list (D5: the caller reads
    # ``warnings``; the tiles keep them as ``tile_warnings`` / ``tile_problems`` so nothing is counted twice)
    for tr in rows:
        for t in tr.tiles:
            tag = f"tier {tr.tier} P{t.percentile}"
            if t.error is not None:
                warnings.append(f"{tag}: drawn as a placeholder: {t.error}")
            warnings += [f"{tag}: {p}" for p in t.problems] + [f"{tag}: {w}" for w in t.warnings]

    warnings = list(dict.fromkeys(warnings))  # the axis under each column reports the same label once
    stats = []
    for tr in rows:
        for t in tr.tiles:
            entry = {"tier": tr.tier, "percentile": t.percentile, "clip_key": t.pick.get("clip_key"),
                     "episode_id": t.pick.get("episode_id"),
                     "episode_median_err": t.pick.get("vo_bearing_err_median"), "error": t.error,
                     "tile_problems": list(t.problems), "tile_warnings": list(t.warnings)}
            if t.row is not None:
                entry.update(row=t.row.index, frame=t.row.frame, frame_label=cd.frame_label(
                    t.row.frame, t.dump.frame_count, "en"), frame_rule="median error closest to the episode's (D8)",
                    span_deg=round(t.span_deg, 2), window=list(tr.win), prediction=t.row.summary(ARM),
                    floor=t.row.summary("floor"), misses=[k + 1 for k in t.row.misses(ARM)],
                    numbered_on_row=[k + 1 for k in t.numbered],
                    notes=[{"kind": it["kind"], "slots": [k + 1 for k in it["slots"]]} for it in t.items],
                    miss_lane_below=t.below, connectors=len(t.connectors), connectors_dropped=t.connectors_dropped,
                    disc_label_gap_pt=round(t.disc_gap_pt, 2), disc_label_gap=t.disc_gap_what,
                    sector_letters={d["name"]: d["mode"] for d in t.letters},
                    marks=[{"slots": [k + 1 for k in m["slots"]], "miss": bool(m["miss"]), "x": round(m["x"], 2),
                            "y": round(m["y"], 2), "y_peak": round(m["y_peak"], 2)} for m in t.marks],
                    marks_touching=t.marks_touching,
                    predicted_visible_note_tags=[{"slots": [k + 1 for k in g["slots"]], "label": g["label"],
                                                  "x": round(g["x"], 2), "y": round(g["y"], 2),
                                                  "x_peak": round(g["x_peak"], 2)} for g in t.fp_tags],
                    predicted_visible_note_untagged=[k + 1 for k in t.fp_untagged],
                    episode={"median": round(t.episode.median, 3), "hits": t.episode.hits, "n": t.episode.n,
                             "n_frames": t.episode.n_frames})
            stats.append(entry)
    return {"files": [str(f) for f in files], "tiers": shown, "missing_tiers": miss, "tiles": stats,
            "size_in": (FIG_W, height), "windows": {tr.tier: list(tr.win) for tr in rows}, "warnings": warnings,
            "height_limit_in": limit, "notes_dropped": [], "min_font_pt": min_font, "layout": layout,
            "caption_flags": {k: (sorted(v) if isinstance(v, set) else v) for k, v in flags.items()}}


def make_gallery_set(cases_json, out_dir, langs: Sequence[str] = ("en", "zh"), dumps_root=None, topdown_root=None,
                     clip_root_override=None, metrics=None, stems=("fig2_gallery_main", "figS_gallery_all")) -> dict:
    """Both galleries (D8) in every language: ``<out_dir>/<stem>[_zh].*``; returns {variant: {lang: result}}."""
    out: Dict[str, Dict[str, dict]] = {}
    for variant, stem in zip(("main", "supp"), stems):
        for lang in langs:
            name = stem if lang == "en" else f"{stem}_{lang}"
            out.setdefault(variant, {})[lang] = make_gallery_figure(
                cases_json, dumps_root=dumps_root, topdown_root=topdown_root, clip_root_override=clip_root_override,
                out_stem=str(Path(out_dir) / name), lang=lang, metrics=metrics, variant=variant)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True, help="cases.json written by select_cases.py")
    ap.add_argument("--variant", default=None, choices=["main", "supp", "both"],
                    help="main: tiers C, D, E; supp: A-E; both: the two, as <out>_main* and <out>_supp*")
    ap.add_argument("--tiers", default=None, help="comma-separated tiers to draw, e.g. C,D,E (without --variant)")
    ap.add_argument("--dumps-root", default=None, help="dump root <root>/<tier>/<scene>/<clip>.npz "
                                                       "(default: the npz_path recorded in cases.json)")
    ap.add_argument("--topdown-root", default=None, help="top-down map root (default $EXP18_ROOT/topdown)")
    ap.add_argument("--clip-root", default=None, help="local copy of the clips: <root>/<scene>/<clip>/chunks")
    ap.add_argument("--metrics", default=None, help="metrics.json or its directory (scene counts; default: the "
                                                    "cases' metrics_dir)")
    ap.add_argument("--out", default="gallery", help="output stem (writes .pdf, .png, _caption.txt)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    args = ap.parse_args(argv)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()] if args.tiers else None
    jobs = [(args.variant, args.out)] if args.variant != "both" else [("main", args.out + "_main"),
                                                                       ("supp", args.out + "_supp")]
    for variant, stem in jobs:
        res = make_gallery_figure(args.cases, dumps_root=args.dumps_root, topdown_root=args.topdown_root,
                                  clip_root_override=args.clip_root, out_stem=stem, lang=args.lang,
                                  tiers=None if variant else tiers, metrics=args.metrics, variant=variant)
        for f in res["files"]:
            print(f)
        for s in res["tiles"]:
            print(s)
        print("tiers", res["tiers"], "missing", res["missing_tiers"], "windows", res["windows"])
        print("size_in", tuple(round(v, 3) for v in res["size_in"]), "limit", res["height_limit_in"])
        for w in res["warnings"]:
            print("warning:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
