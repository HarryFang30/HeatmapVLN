#!/usr/bin/env python3
"""EXP-19 behaviour figure: what the model saw, predicted and decided at key moments of its own closed-loop rerun.

One page per episode, 7.0 in wide.  Each page comes in an English and a
Chinese version, and each version is written as a PDF (300 dpi, TrueType
fonts), a PNG (400 dpi) and a caption text file.

Layout::

  header: category · instruction (full, wrapped) · rerun outcome
  +------------+  column heads: decision image | frames given to System2
  | route map  |  K1  step s   System2 "↓" → "393 181"          executed [↑][↑][←][↑]
  |            |  [decision image] [1][2]...[8][now]   <- history fronts as fed, and the current front
  | legend     |  360° strip: muted surroundings + predicted history affordance map + true past directions
  | data flow  |  360° strip: predicted future affordance map (4 time bins) + System1 path
  +------------+  K2 ... K4

What is drawn at each key moment was fixed before any rerun (README EXP-19
"关键时刻").  The data comes only from the figure bundle
(``scripts/exp19/figures/bundle.py``) and the EXP-18 top-down maps.  The
captions describe each drawn key moment by the rule branch [F] recorded for it
(``KEY_RULES``), so an episode with fewer than four ready calls, or a fallback
branch, is described as it was chosen.

The main figure is split in two, the successes (T1-T3) and the failures
(F1, F2), each with its own legend and data flow.  At 7 in wide, one figure
with all five cases would be about 9 in tall and would have to be scaled below
the 5 pt text floor.  Height budgets: a main figure <= 8.0 in, a page
<= 8.5 in (a double-column text block less its caption); a figure over budget
is reported in the manifest and on stderr.

Figure policy (EXP-18 and EXP-19 ledgers):

* No poses, VO, odometry or relative-pose values.  No heading arrows on the
  map.  Every map mark is a recorded simulator position.
* Both heatmaps are called "affordance map" (predicted / ground truth).  The
  wording never says the model "understands" or "remembers" anything.
* Only the framed front part of the 360° strips (the deployed camera's 79°
  view) was given to the model.  The rest is muted and marked as not given to
  the model (display only).
* The data flow is drawn as it runs.  The future affordance map and the
  System1 path are both decoded from Z̃, and the future map is computed after
  the path.  There is no arrow from the future map to the actions.
* The captions' claims about H1, H2 and H3 come only from ``claim_sentences``,
  a function of the verdicts in ``metrics.json``, which apply the
  pre-registered wording rules.  A page is never hand-edited to add a claim.
* A batch that ``metrics.json`` marks invalid (a validity gate failed: 整批作废)
  is not drawn unless ``--allow-void``, and then every page and caption is
  stamped "VOID BATCH".

Usage (repo root on PYTHONPATH)::

  python -m scripts.exp19.figures.fig_behavior --records <EXP>/records --metrics <EXP>/metrics/metrics.json \\
      --out-dir <EXP>/figures [--lang en zh] [--topdown-root DIR] [--main | --main-only] [--allow-void]
  python -m scripts.exp19.figures.fig_behavior --bundle <x_bundle.json> --no-verdicts --out-dir /tmp/fig
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp18.topdown.topdown_io import load_topdown
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import panels as pn

import matplotlib  # noqa: E402  (configured in setup())
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

MANIFEST_SCHEMA = "exp19-figures-manifest-v1"

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); every string on the page comes from here
# --------------------------------------------------------------------------- #
CATEGORY_NAMES = {
    "en": {"T1": "multi-room, multi-turn", "T2": "cross-floor", "T3": "long range",
           "F1": "reached the goal area, stopped in the wrong place", "F2": "wandering (hit the step limit)"},
    "zh": {"T1": "多房间多转弯", "T2": "跨楼层", "T3": "长程", "F1": "到过目标区却停错", "F2": "徘徊（撞上步数上限）"},
}

LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "title": "{cat} · {name}",
        "ids": "rerun · scene {scene} · episode {ep}",
        "main_case": "main case",
        "rank": "candidate {r} of {cat}",
        "instruction": "“{text}”",
        "dist": "{ne:.1f} m from the goal",
        "dist_missing": "at an unrecorded distance from the goal",
        "outcome_success": "Rerun: success, stopped {dist} after {steps} steps.",
        "outcome_stop": "Rerun: failure, stopped {dist} after {steps} steps{os}.",
        "outcome_cap": "Rerun: failure, hit the {steps}-step limit {dist}{os}.",
        "outcome_other": "Rerun: failure, ended without STOP after {steps} steps, {dist}{os}.",
        "outcome_os": " (it had been within the success radius earlier)",
        "outcome_missing": "Rerun outcome not recorded.",
        "predicate_fails": "The rerun no longer meets the {cat} definition; shown anyway, as pre-registered.",
        "synthetic": "SYNTHETIC BUNDLE: layout test, not a result",
        "void": "VOID BATCH, not evidence (a pre-registered validity condition failed): {reasons}",
        "route": "Route of the rerun",
        "start": "start",
        "radius": "3 m",
        "floor_note": "Route changes floor; the map shows the floor of most steps, dotted = on another floor.",
        "floor_note_nomap": "Route changes floor (height range over 1 m); all steps are drawn on one plate.",
        "no_map": "no top-down map for this scene",
        "no_keys": ("This rerun has no ready System2 call (a call where System2 returned a pixel goal and the history "
                    "head, bridge and future head ran), so there is no key moment and no affordance map to show."),
        "col_decision": "Decision image",
        "col_frames": "Frames given to System2: past frames (1 = oldest) and now; number below = step",
        "col_strips": ("Below each row, the 360° around the robot (centre = straight ahead), re-rendered at the "
                       "recorded position: only the framed front part, the camera's 79° view, was given to the model; "
                       "L, B and R are not given to the model (display only)."),
        "step": "step {s}",
        "system2": "System2",
        "then": "→",
        "executed": "executed",
        "lookdown": "look-down image",
        "front": "front image",
        "now": "now",
        "sectors": ("F", "L", "R", "B"),
        "sector_notes": ("model's view", "display only"),
        "hist_row": "predicted history affordance map",
        "fut_row": "predicted future affordance map",
        "axis": ("180°", "+90° left", "0° ahead", "−90° right", "−180°"),
        "legend_title": "Legend",
        "legend": {
            "route": "executed route (rerun)",
            "ref": "reference path",
            "start_goal": "start · goal · success radius",
            "key": "key moment",
            "goal_px": "System2 pixel goal",
            "path": "System1 mean path",
            "slot": "past frame given to System2 (1 = oldest)",
            "hist": "predicted history affordance map",
            "gt": "true direction of a past frame",
            "edge": "direction beyond ±15° (stairs), on the strip edge: blue = true, orange = predicted",
            "fut": "predicted future affordance map, waypoints 1–8 … 25–32 (darker = later)",
            "actions": "executed actions: forward 0.25 m, turn 15°, stop",
        },
        "strip_note": ("360° strips: centre = straight ahead, re-rendered at the recorded position. Only the framed "
                       "front part (the camera's 79° view) was given to the model; L, B and R are not given to the "
                       "model (display only)."),
        "flow_title": "Data flow at a key moment",
        "flow": {
            "s2": "System2", "text": "text goal first", "z": "$Z$", "zt": r"$\tilde{Z}$",
            "hh": "History head", "mem": "$M$", "bridge": "Bridge", "s1": "System1", "fh": "Future head",
            "hist": "history\naffordance map", "path": "path → actions", "fut": "future\naffordance map",
            "frames": "past + current fronts",
        },
    },
    "zh": {
        "title": "{cat} · {name}",
        "ids": "复跑 · 场景 {scene} · 第 {ep} 集",
        "main_case": "主图案例",
        "rank": "{cat} 第 {r} 个候选",
        "instruction": "“{text}”",
        "dist": "距目标 {ne:.1f} m",
        "dist_missing": "距目标距离未记录",
        "outcome_success": "复跑结果：成功，共 {steps} 步，停下时{dist}。",
        "outcome_stop": "复跑结果：失败，共 {steps} 步，停下时{dist}{os}。",
        "outcome_cap": "复跑结果：失败，撞上 {steps} 步上限，{dist}{os}。",
        "outcome_other": "复跑结果：失败，未经 STOP 结束，共 {steps} 步，{dist}{os}。",
        "outcome_os": "（此前曾进入成功半径）",
        "outcome_missing": "复跑结局未记录。",
        "predicate_fails": "复跑已不满足 {cat} 的定义；按预注册照常出图。",
        "synthetic": "合成数据包：仅用于排版测试，不是结果",
        "void": "整批作废，不是证据（预注册的有效性条件未满足）：{reasons}",
        "route": "复跑路线",
        "start": "起点",
        "radius": "3 m",
        "floor_note": "路线跨楼层；俯视图为多数步所在楼层，点线 = 在另一层。",
        "floor_note_nomap": "路线跨楼层（高差超过 1 m）；所有步画在同一底板上。",
        "no_map": "该场景没有俯视图",
        "no_keys": "本次复跑没有就绪调用（慢系统发了像素目标，历史头、桥、未来头都运行了的调用），因此没有关键时刻，也没有 affordance map 可画。",
        "col_decision": "决策所用图",
        "col_frames": "送入慢系统的帧：历史帧（1 = 最早）与当前帧；下方数字 = 步号",
        "col_strips": "每行下方为机器人周围 360°（中央 = 正前方），在记录位置重渲染：只有加框的前方部分（相机 79° 视场）输入了模型；左、后、右未输入模型（仅展示）。",
        "step": "第 {s} 步",
        "system2": "慢系统",
        "then": "→",
        "executed": "执行",
        "lookdown": "俯视帧",
        "front": "前视帧",
        "now": "当前",
        "sectors": ("前", "左", "右", "后"),
        "sector_notes": ("模型视野", "仅展示"),
        "hist_row": "预测历史 affordance map",
        "fut_row": "预测未来 affordance map",
        "axis": ("180°", "+90° 左", "0° 正前", "−90° 右", "−180°"),
        "legend_title": "图例",
        "legend": {
            "route": "执行路线（复跑）",
            "ref": "参考路径",
            "start_goal": "起点 · 目标 · 成功半径",
            "key": "关键时刻",
            "goal_px": "慢系统像素目标",
            "path": "快系统均值路径",
            "slot": "送入慢系统的历史帧（1 = 最早）",
            "hist": "预测历史 affordance map",
            "gt": "历史帧的真实方向",
            "edge": "超出 ±15° 的方向（楼梯）画在条带边缘：蓝 = 真实，橙 = 预测",
            "fut": "预测未来 affordance map，路点 1–8 … 25–32（越深越晚）",
            "actions": "执行动作：前进 0.25 m、转 15°、停止",
        },
        "strip_note": "360° 条带：中央 = 正前方，在记录位置重渲染。只有加框的前方部分（相机 79° 视场）输入了模型；左、后、右未输入模型（仅展示）。",
        "flow_title": "关键时刻的数据流",
        "flow": {
            "s2": "慢系统", "text": "先出文本目标", "z": "$Z$", "zt": r"$\tilde{Z}$",
            "hh": "历史头", "mem": "$M$", "bridge": "桥", "s1": "快系统", "fh": "未来头",
            "hist": "历史\naffordance map", "path": "路径 → 动作", "fut": "未来\naffordance map",
            "frames": "历史帧 + 当前帧",
        },
    },
}

# How each key moment was chosen, per rule branch (bundle.KEY_BRANCHES, as scripts/exp19/keysteps.py tags them).
KEY_RULES = {
    "en": {
        "K1_first": "the first ready call",
        "K2_turn": "the other ready call whose executed action chunk has the largest |net turn| (at least 30°)",
        "K2_fallback": ("the ready call at position ⌊(n−1)/3⌋ of the n ready calls, counted from 0 (no other executed "
                        "chunk turns by 30° or more)"),
        "K3_two_thirds": "the other ready call whose step is closest to 2/3 of the episode's steps",
        "K3_f1_closest": "the last other ready call at or before the step closest to the goal",
        "K3_f1_fallback_after": ("the first other ready call after the step closest to the goal (there is none at or "
                                 "before it; this fallback is not in the pre-registration)"),
        "K4_last": "the last ready call",
        "K4_shifted": "the latest ready call not already chosen (the last one is already a key moment)",
        "all_lt4": ("The rerun has only {n} ready calls, fewer than four, so all of them are drawn: {labels} in call "
                    "order."),
        "all_lt4_one": "The rerun has only one ready call, drawn as K1.",
        "join": "; ", "first": "{label} is {rule}", "next": "{label} {rule}", "end": ".",
    },
    "zh": {
        "K1_first": "第一个就绪调用",
        "K2_turn": "其余就绪调用中实际执行动作块净转角绝对值最大者（≥ 30°）",
        "K2_fallback": "就绪调用序列第 ⌊(n−1)/3⌋ 个（n 为就绪调用数，从 0 计；其余执行动作块净转角绝对值都不到 30°）",
        "K3_two_thirds": "其余就绪调用中步号最接近全集步数 2/3 者",
        "K3_f1_closest": "距目标最近那一步及其之前的最后一个其余就绪调用",
        "K3_f1_fallback_after": "距目标最近那一步之后的第一个其余就绪调用（该步及之前没有其余就绪调用；此兜底不在预注册中）",
        "K4_last": "最后一个就绪调用",
        "K4_shifted": "尚未被选的最晚一个就绪调用（最后一个已被选为关键时刻）",
        "all_lt4": "本次复跑只有 {n} 个就绪调用（不足 4 个），全部画出，{labels} 按调用序。",
        "all_lt4_one": "本次复跑只有 1 个就绪调用，画为 K1。",
        "join": "；", "first": "{label} 为{rule}", "next": "{label} 为{rule}", "end": "。",
    },
}

CAPTION = {
    "en": (
        "Closed-loop rerun of R2R val_unseen episode {ep} (scene {scene}; category {cat}, {cat_name}): what the model "
        "saw, predicted and decided at {n_moments}. Key moments are ready System2 calls, i.e. calls where System2 "
        "returned a pixel goal and the history head, bridge and future head ran. {key_rules} Left: the rerun's "
        "executed route (black), the reference path (grey, dashed), start (open circle), goal (star) with its 3 m "
        "success radius, and the key moments. Each row, left to right: the image System2 decided on (the look-down "
        "image after its “↓” turn, otherwise the front image) with its pixel goal (ring) and the System1 mean path "
        "(dots); the frames given to System2 (past frames numbered 1 = oldest, then the current front). Below, two "
        "strips of the 360° around the robot, centre = straight ahead and the robot's left to the left, re-rendered "
        "at the recorded position: only the framed front part, the deployed camera's 79° field of view, was given to "
        "the model; the muted left (L), back (B) and right (R) are shown for reference only and are not given to the "
        "model. The first strip is overlaid with the predicted history affordance map (orange; each slot's map "
        "divided by its own peak and multiplied by its predicted visibility, maximum over slots) and the true "
        "directions of the past frames (blue circles); the second shows the predicted future affordance map for the "
        "four time bins of the System1 path (waypoints 1–8, 9–16, 17–24, 25–32; darker = later) with the System1 "
        "path (dots). Chips: the action chunk executed after the call (↑ forward 0.25 m, ←/→ turn 15°, STOP). Data "
        "flow at a ready call: System2 first gives its decision as text; the history head reads the past and current "
        "fronts and outputs the history affordance map and a memory M, which the bridge adds to System2's latent Z to "
        "give Z̃; System1 decodes Z̃ into the path and the actions; the future head decodes the same Z̃ into the "
        "future affordance map after the path is generated, and the future map does not feed the actions."
    ),
    "zh": (
        "R2R val_unseen 第 {ep} 集（场景 {scene}；类别 {cat}，{cat_name}）的闭环复跑：模型在 {n} 个关键时刻看到了什么、"
        "预测了什么、决定了什么。关键时刻取自就绪调用（慢系统发了像素目标，历史头、桥、未来头都运行了的调用）。{key_rules}"
        "左：复跑的执行路线（黑）、参考路径（灰虚线）、起点（空心圆）、目标（星）及 3 m 成功半径、各关键时刻。"
        "每行从左到右：慢系统据以决策的图（答“↓”后为俯视帧，否则为前视帧），标出像素目标（圆环）与快系统均值路径（点）；"
        "送入慢系统的帧（历史帧编号 1 = 最早，末为当前前视帧）。下方两条为机器人周围 360°（中央 = 正前方，左侧 = 机器人左方），"
        "在记录位置重渲染：只有加框的前方部分（部署相机 79° 视场）输入了模型，压灰的左 / 后 / 右仅作展示、未输入模型。"
        "第一条叠加预测历史 affordance map（橙；每个槽位的图除以自身峰值再乘以其预测可见概率，取各槽位最大值）与历史帧的真实方向（蓝圈）；"
        "第二条为快系统路径四个时段（路点 1–8、9–16、17–24、25–32；越深越晚）的预测未来 affordance map 与快系统路径（点）。"
        "方块：该次调用后实际执行的动作块（↑ 前进 0.25 m，←/→ 转 15°，STOP）。就绪调用的数据流：慢系统先以文本给出决策；"
        "历史头读历史帧与当前帧，输出历史 affordance map 与记忆 M，桥把 M 注入慢系统的隐变量 Z 得 Z̃；快系统以 Z̃ 解码出路径与动作；"
        "未来头以同一个 Z̃ 解码未来 affordance map，在路径生成之后才计算，不回流到动作。"
    ),
}

CAPTION_NO_KEYS = {
    "en": ("Closed-loop rerun of R2R val_unseen episode {ep} (scene {scene}; category {cat}, {cat_name}). "
           "The rerun has no ready System2 call (a call where System2 returned a pixel goal and the history head, "
           "bridge and future head ran), so no key moment is drawn. Left: the rerun's executed route (black), the "
           "reference path (grey, dashed), start (open circle) and goal (star) with its 3 m success radius."),
    "zh": ("R2R val_unseen 第 {ep} 集（场景 {scene}；类别 {cat}，{cat_name}）的闭环复跑。本次复跑没有就绪调用（慢系统发了像素目标，"
           "历史头、桥、未来头都运行了的调用），因此不画关键时刻。左：复跑的执行路线（黑）、参考路径（灰虚线）、起点（空心圆）、"
           "目标（星）及 3 m 成功半径。"),
}

CAPTION_EDGE = {
    "en": ("Directions more than 15° above or below the horizon (stairs) are drawn as triangles on the strip's edge: "
           "open blue = a true past direction, orange = the predicted peak of a slot the head calls visible."),
    "zh": "偏离水平方向超过 15° 的方向（楼梯）画成条带边缘的三角：空心蓝 = 历史帧的真实方向，橙 = 历史头判为可见的槽位的预测峰值。",
}

CAPTION_VOID = {
    "en": "VOID BATCH: a pre-registered validity condition failed ({reasons}); these figures are not evidence.",
    "zh": "整批作废：预注册的有效性条件未满足（{reasons}）；这些图不是证据。",
}

# Rerun fidelity vs the seed-42 main-table log.  "Same outcome" is build_records.compare_outcome's: success, oracle
# success and, when the reference records it, how the episode ended (STOP / step limit) are all equal.
FIDELITY = {
    "en": {"all": "Rerun fidelity: all {total} System2 calls identical to the seed-42 main-table evaluation",
           "all_1": "Rerun fidelity: the only System2 call identical to the seed-42 main-table evaluation",
           "all_2": "Rerun fidelity: both System2 calls identical to the seed-42 main-table evaluation",
           "some": ("Rerun fidelity: {same} of {total} System2 calls identical to the seed-42 main-table evaluation, "
                    "the first difference at the {first} call"),
           "outcome_same": "; same outcome.",
           "outcome_diff": "; different outcome (rerun: {rerun}; main table: {ref}).",
           "period": ".", "success": "success", "failure": "failure",
           "ended": {"stop": "failure ending with STOP", "step_cap": "failure at the step limit",
                     "other": "failure ending without STOP"},
           "os": " after reaching the success radius"},
    "zh": {"all": "复跑保真：全部 {total} 次慢系统调用与主表种子 42 评测逐字相同",
           "some": "复跑保真：{total} 次慢系统调用中 {same} 次与主表种子 42 评测逐字相同，首个分歧在第 {first} 次调用",
           "outcome_same": "；结局相同。", "outcome_diff": "；结局不同（复跑：{rerun}；主表：{ref}）。",
           "period": "。", "success": "成功", "failure": "失败",
           "ended": {"stop": "失败（以 STOP 结束）", "step_cap": "失败（撞上步数上限）", "other": "失败（未经 STOP 结束）"},
           "os": "，曾进入成功半径"},
}


# --------------------------------------------------------------------------- #
# Claims: the ONLY source of H1/H2/H3 statements in captions (pre-registered wording rules)
# --------------------------------------------------------------------------- #
# metrics.json["verdicts"][H]["verdict"] as scripts/exp19/build_records.py writes it.  "missing" (no data) and
# "void" (a validity gate failed: the batch is not evidence) allow no claim at all.
VERDICTS = {"H1": ("support", "partial", "refute", "missing", "void"),
            "H2": ("support", "partial", "refute", "not_measured", "missing", "void"),
            "H3": ("support", "refute", "report_only", "missing", "void")}

CLAIMS = {
    "en": {
        ("H1", "support"): "In the closed-loop reruns the history affordance map agrees with the ground truth "
                           "(PCK@8 {pck8:.3f}, trivial baseline {baseline:.3f}).",
        ("H1", "partial"): "The predicted history affordance map is better than a trivial baseline "
                           "(PCK@8 {pck8:.3f} vs {baseline:.3f}).",
        ("H2", "support"): "The future affordance map agrees with the direction of the subsequent actions.",
        ("H3", "support"): "When the history memory is removed, {pct:.0f}% of the decision points change their "
                           "action chunk.",
        ("H3", "refute"): "The affordance maps are shown as a display of the model's internal state.",
    },
    "zh": {
        ("H1", "support"): "闭环中历史 affordance map 与真值吻合（PCK@8 {pck8:.3f}，平凡基线 {baseline:.3f}）。",
        ("H1", "partial"): "预测的历史 affordance map 好于平凡基线（PCK@8 {pck8:.3f}，基线 {baseline:.3f}）。",
        ("H2", "support"): "未来 affordance map 与随后的动作方向一致。",
        ("H3", "support"): "撤掉历史记忆时 {pct:.0f}% 的决定点动作块改变。",
        ("H3", "refute"): "图中的 affordance map 是模型内部状态的展示。",
    },
}


def verdict_of(hyp: str, entry: dict) -> str:
    """The pre-registered verdict word of one ``metrics.json["verdicts"]`` entry (raises on anything else)."""
    word = entry.get("verdict") if isinstance(entry, dict) else None
    if word not in VERDICTS[hyp]:
        raise ValueError(f"{hyp}: verdict {word!r} is not one of {VERDICTS[hyp]}")
    return word


def _number(hyp: str, entry: dict, key: str) -> float:
    value = entry.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{hyp}: the claim needs a finite {key!r}, got {value!r}")
    return float(value)


def claim_sentences(verdicts: Optional[dict], lang: str) -> List[str]:
    """Caption sentences allowed by the pre-registered wording rules (README EXP-19 "判据").

    PCK@8 is printed to three decimals, so a "partial" 0.796 never reads as the 0.80 bar.

    * H1 support -> agreement, with PCK@8 and the trivial baseline; partial ->
      only "better than a trivial baseline"; refute -> nothing (the page only
      ever says "predicted").
    * H2 -> a sentence only when supported (not measured / partial / refute: nothing).
    * H3 support -> the change rate, as dependence only; refute -> "a display of
      the model's internal state"; report only -> nothing.
    * missing / void -> nothing.

    ``verdicts`` is ``metrics.json["verdicts"]`` of ``build_records.py``: per
    hypothesis a dict with ``verdict`` and the numbers (H1 ``joint_pck8``,
    ``floor_pck8``; H3 ``change_rate`` as a fraction).  An unknown verdict raises;
    ``None`` (no metrics yet) gives no claims.
    """
    if verdicts is None:
        return []
    missing = [hyp for hyp in ("H1", "H2", "H3") if hyp not in verdicts]
    if missing:
        raise ValueError(f"verdicts lack {', '.join(missing)}")
    out = []
    for hyp in ("H1", "H2", "H3"):
        entry = verdicts[hyp]
        verdict = verdict_of(hyp, entry)
        template = CLAIMS[lang].get((hyp, verdict))
        if template is None:
            continue
        values = {}
        if hyp == "H1":
            values = {"pck8": _number(hyp, entry, "joint_pck8"), "baseline": _number(hyp, entry, "floor_pck8")}
        elif hyp == "H3" and verdict == "support":
            values = {"pct": 100.0 * _number(hyp, entry, "change_rate")}
        out.append(template.format(**values))
    return out


def load_metrics(metrics_path):
    """(verdicts, void reasons) of ``metrics.json``; the reasons are empty unless the batch is void.

    ``build_records.py`` writes ``validity = {valid, reasons}``: a failed gate
    (code equivalence, trace neutrality) or missing / unfinished episodes make
    the whole batch void (整批作废), and every verdict is then "void".
    """
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if "verdicts" not in data:
        raise ValueError(f"{metrics_path}: no 'verdicts'")
    validity = data.get("validity")
    if validity is not None:
        void = [] if validity.get("valid") else [str(r) for r in validity.get("reasons") or ["validity.valid is false"]]
    else:  # older metrics: read the gates themselves
        void = [f"{name} gate failed" for name, g in (data.get("gates") or {}).items() if not g.get("pass")]
    return data["verdicts"], void


# --------------------------------------------------------------------------- #
# Captions and header text
# --------------------------------------------------------------------------- #
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def outcome_line(b: bd.Bundle, L: dict) -> str:
    """The rerun's outcome; how it ended as recorded ("other" / "unknown" = ended without a STOP)."""
    o = b.outcome
    if o is None:
        return L["outcome_missing"]
    dist = L["dist"].format(ne=o["ne_m"]) if _finite(o["ne_m"]) else L["dist_missing"]
    if o["success"]:
        return L["outcome_success"].format(dist=dist, steps=o["steps"])
    os_note = L["outcome_os"] if o["oracle_success"] else ""
    key = {"stop": "outcome_stop", "step_cap": "outcome_cap"}.get(o["ended_by"], "outcome_other")
    return L[key].format(dist=dist, steps=o["steps"], os=os_note)


def same_outcome(o: dict, ref: dict) -> bool:
    """``build_records.compare_outcome(...)["same"]``: success, oracle success and, when the reference records it,
    how the episode ended (STOP / step limit) are all equal.  ``ref`` is the eval-log reference's ``final``."""
    success = (float(o["success"]) >= 0.5) == (float(ref["success"]) >= 0.5)
    oracle = (float(o["oracle_success"]) >= 0.5) == (float(ref["os"]) >= 0.5)
    ended = ref.get("ended_by") is None or o["ended_by"] == ref["ended_by"]
    return bool(success and oracle and ended)


def _ending(success: bool, oracle: bool, ended_by: Optional[str], F: dict) -> str:
    if success:
        return F["success"]
    return F["ended"].get(ended_by, F["failure"]) + (F["os"] if oracle else "")


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fidelity_sentence(b: bd.Bundle, lang: str) -> str:
    """Rerun fidelity vs the seed-42 main-table log; empty when [F] had no eval-log reference.

    Calls are counted from 1 here ([F]'s ``first_divergent_call`` is 0-based).
    """
    F = FIDELITY[lang]
    f = b.fidelity
    if f["identical_calls"] is None or f["total_calls"] is None:
        return ""
    if f["first_divergent_call"] is None:
        s = F.get(f"all_{f['total_calls']}", F["all"]).format(total=f["total_calls"])
    else:
        first = int(f["first_divergent_call"]) + 1
        s = F["some"].format(same=f["identical_calls"], total=f["total_calls"],
                             first=_ordinal(first) if lang == "en" else first)
    ref, o = b.meta.get("eval_log_outcome") or {}, b.outcome
    if o is None or ref.get("success") is None or ref.get("os") is None:
        return s + F["period"]
    if same_outcome(o, ref):
        return s + F["outcome_same"]
    return s + F["outcome_diff"].format(
        rerun=_ending(bool(o["success"]), bool(o["oracle_success"]), o["ended_by"], F),
        ref=_ending(float(ref["success"]) >= 0.5, float(ref["os"]) >= 0.5, ref.get("ended_by"), F))


def standard_branch(label: str, category: str) -> str:
    """The branch a key moment takes when the rule's main clause applies."""
    if label == "K3":
        return "K3_f1_closest" if category == "F1" else "K3_two_thirds"
    return {"K1": "K1_first", "K2": "K2_turn", "K4": "K4_last"}[label]


def key_rules_sentence(keys: Sequence[bd.KeyStep], lang: str) -> str:
    """How the drawn key moments were chosen, from the branch [F] recorded for each (not the rule in general)."""
    R = KEY_RULES[lang]
    if any(k.branch == "all_lt4" for k in keys):
        return R["all_lt4_one" if len(keys) == 1 else "all_lt4"].format(n=len(keys),
                                                                          labels=", ".join(k.label for k in keys))
    parts = [R["first" if i == 0 else "next"].format(label=k.label, rule=R[k.branch]) for i, k in enumerate(keys)]
    return R["join"].join(parts) + R["end"]


def void_sentence(void: Optional[Sequence[str]], lang: str) -> str:
    return CAPTION_VOID[lang].format(reasons="; ".join(void)) if void else ""


def caption_text(b: bd.Bundle, lang: str, claims: Sequence[str], edge_marks: bool = False,
                 void: Optional[Sequence[str]] = None) -> str:
    L = LABELS[lang]
    template = CAPTION[lang] if b.keys else CAPTION_NO_KEYS[lang]
    parts = [void_sentence(void, lang),
             template.format(ep=b.episode_id, scene=b.scene_id, cat=b.category,
                             cat_name=CATEGORY_NAMES[lang][b.category], n=len(b.keys),
                             n_moments="1 key moment" if len(b.keys) == 1 else f"{len(b.keys)} key moments",
                             key_rules=key_rules_sentence(b.keys, lang))]
    if edge_marks:
        parts.append(CAPTION_EDGE[lang])
    parts += list(claims)
    parts.append(fidelity_sentence(b, lang))
    if b.predicate_holds_on_rerun is False:
        parts.append(L["predicate_fails"].format(cat=b.category))
    sep = " " if lang == "en" else ""
    return sep.join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Geometry (inches, from the top-left corner)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # 7.0
X_L, W_L = 0.0, 1.56  # episode page, left column: map, legend, data flow
X_R = 1.70
W_R = FIG_W - X_R - 0.01
HDR_PAD = 0.04
COLHDR_H = 0.34
BLOCK_GAP = 0.13
AXIS_H = 0.16
N_FRAMES = bd.NUM_SLOTS + 1  # history slots + now
FS = {"title": 7.6, "head": 6.4, "instr": 6.3, "body": 6.1, "small": 5.7, "tiny": 5.4, "legend": 5.7}
FLOW_H_PT = 120.0
FLOW_WIDE_H_PT = 60.0


@dataclass
class BlockGeom:
    """Placement of one key-moment block: header line, decision image + filmstrip, two 360° strips."""

    x: float
    w: float
    img_h: float = 0.56  # decision image height (shown at 4:3)
    hdr_h: float = 0.17
    dec_label_h: float = 0.10
    film_gap_l: float = 0.12
    frame_gap: float = 0.03
    lane_h: float = 0.095  # slot badges above the frames
    step_h: float = 0.09  # step numbers below the frames
    el_hist: float = 15.0  # history strip +-15 deg: past camera centres on the same floor sit on the horizon
    el_fut: float = 9.0  # future strip +-9 deg: the future labels put waypoints at camera height
    gap_a: float = 0.05
    gap_b: float = 0.035
    chip: float = 7.6
    fs_head: float = 6.4
    fs_small: float = 5.4
    badge_fs: float = cd.BADGE_FS
    row_labels: bool = True

    @property
    def dec_w(self) -> float:
        return self.img_h * 4.0 / 3.0

    @property
    def film_x(self) -> float:
        return self.x + self.dec_w + self.film_gap_l

    @property
    def film_w(self) -> float:
        return self.x + self.w - self.film_x

    @property
    def frame_w(self) -> float:
        return (self.film_w - (N_FRAMES - 1) * self.frame_gap) / N_FRAMES

    @property
    def frame_h(self) -> float:
        return self.frame_w * 3.0 / 4.0

    @property
    def img_row_h(self) -> float:
        return max(self.img_h + self.dec_label_h, self.lane_h + self.frame_h + self.step_h)

    @property
    def h_hist(self) -> float:
        return self.w * 2 * self.el_hist / 360.0  # square degrees

    @property
    def h_fut(self) -> float:
        return self.w * 2 * self.el_fut / 360.0

    @property
    def height(self) -> float:
        return self.hdr_h + self.img_row_h + self.gap_a + self.h_hist + self.gap_b + self.h_fut

    def ring_w(self, px_per_in: float) -> int:
        return max(8, int(round(self.w * px_per_in / 8.0)) * 8)


EPISODE_BLOCK = BlockGeom(X_R, W_R)
PAGE_MAX_H_IN = 8.5  # an episode page: a double-column text block (about 8.9-9.5 in) less a short caption
MAIN_MAX_H_IN = 8.0  # a main figure, with room for its longer caption


def has_edge_marks(keys: Sequence[bd.KeyStep], band: float) -> bool:
    """Whether a history strip of these key moments draws a direction beyond its band (stairs) on the edge."""
    return any(not inside for ks in keys for *_, inside in pn.history_marks(
        bd.gt_history_peaks(ks.hist_gt_peak, ks.hist_mask), bd.pred_history_peaks(ks.hist_pred, ks.hist_none,
                                                                                  ks.hist_mask), band))


def over_budget(height: float, budget: float, what: str) -> List[str]:
    """A warning when a figure is taller than its budget (it would have to be scaled below the 5 pt text floor)."""
    return [f"{what} is {height:.2f} in tall, over the {budget:.1f} in budget"] if height > budget + 1e-6 else []


class Page:
    """Axes placement in inches from the top-left corner."""

    def __init__(self, fig, width: float, height: float):
        self.fig, self.w, self.h = fig, width, height

    def ax(self, x: float, y_top: float, w: float, h: float, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h], **kw)

    def pt_axes(self, x: float, y_top: float, w: float, h: float, zorder: float = 0):
        """Invisible axes in point units (x right, y up from the bottom edge)."""
        ax = self.ax(x, y_top, w, h)
        ax.set_xlim(0, w * 72.0)
        ax.set_ylim(0, h * 72.0)
        ax.axis("off")
        ax.set_zorder(zorder)
        return ax

    def text(self, x: float, y: float, s: str, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

    def rule(self, y: float, x0: float = 0.0, x1: Optional[float] = None) -> None:
        ax = self.ax(x0, y, (x1 if x1 is not None else self.w) - x0, 0.001)
        ax.axis("off")
        ax.axhline(0.5, color=style.AXIS, lw=0.6)


_TOKEN = re.compile(r"[⺀-鿿＀-￯]|[^\s⺀-鿿＀-￯]+|\s+")
_NO_LINE_START = set("，。：；、）」』！？")  # CJK punctuation stays on the line before


def wrap(fig, text: str, fontsize: float, width_pt: float, **kw) -> List[str]:
    """Greedy wrap to ``width_pt`` using rendered widths (CJK wraps per character, Latin per word)."""
    lines: List[str] = []
    cur = ""
    for tok in _TOKEN.findall(text):
        trial = cur + tok
        if (cur.strip() and not tok.isspace() and tok not in _NO_LINE_START
                and cd.text_width_pt(fig, trial.rstrip(), fontsize, **kw) > width_pt):
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur = trial
    if cur.strip():
        lines.append(cur.strip())
    return lines


def setup(lang: str) -> None:
    cd.setup(lang)
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": style.LATIN_FAMILY,
                                "mathtext.it": f"{style.LATIN_FAMILY}:italic",
                                "mathtext.bf": f"{style.LATIN_FAMILY}:bold",
                                "mathtext.cal": style.LATIN_FAMILY, "mathtext.sf": style.LATIN_FAMILY})


# --------------------------------------------------------------------------- #
# Page header
# --------------------------------------------------------------------------- #
def header_lines(fig, b: bd.Bundle, L: dict, void: Optional[Sequence[str]] = None) -> dict:
    """Wrapped instruction, and the red lines under the outcome: void batch, predicate no longer met."""
    instr = wrap(fig, L["instruction"].format(text=" ".join(b.instruction.split())), FS["instr"], FIG_W * 72.0 - 2.0,
                 fontstyle="italic")
    extra = wrap(fig, L["void"].format(reasons="; ".join(void)), FS["body"], FIG_W * 72.0 - 2.0) if void else []
    if b.predicate_holds_on_rerun is False:
        extra.append(L["predicate_fails"].format(cat=b.category))
    return {"instr": instr, "extra": extra}


def header_height(lines: dict) -> float:
    return 0.17 + 0.108 * len(lines["instr"]) + 0.125 * (1 + len(lines["extra"])) + HDR_PAD + 0.05


def draw_header(page: Page, b: bd.Bundle, lines: dict, lang: str, L: dict) -> float:
    y = 0.085
    title = L["title"].format(cat=b.category, name=CATEGORY_NAMES[lang][b.category])
    page.text(0.0, y, title, ha="left", va="center", fontsize=FS["title"], fontweight="bold", color=style.INK)
    right = L["ids"].format(scene=b.scene_id, ep=b.episode_id)
    right += "  ·  " + (L["main_case"] if b.is_main else L["rank"].format(r=b.category_rank + 1, cat=b.category))
    page.text(FIG_W, y, right, ha="right", va="center", fontsize=FS["small"], color=style.MUTED)
    if b.synthetic:
        page.text(FIG_W / 2 + 0.3, y, L["synthetic"], ha="center", va="center", fontsize=FS["small"],
                  fontweight="bold", color="#d03b3b")
    y = 0.17
    for line in lines["instr"]:
        y += 0.054
        page.text(0.0, y, line, ha="left", va="center", fontsize=FS["instr"], color=style.INK, fontstyle="italic")
        y += 0.054
    y += 0.07
    page.text(0.0, y, outcome_line(b, L), ha="left", va="center", fontsize=FS["body"], color=style.INK,
              fontweight="bold")
    for line in lines["extra"]:
        y += 0.125
        page.text(0.0, y, line, ha="left", va="center", fontsize=FS["body"], color="#d03b3b")
    y += 0.07 + HDR_PAD
    page.rule(y)
    return y + 0.05


def draw_no_keys_note(page: Page, x: float, y: float, w: float, L: dict, fs: float = FS["body"]) -> None:
    """In place of the key-moment blocks when the rerun has no ready call."""
    for j, line in enumerate(wrap(page.fig, L["no_keys"], fs, w * 72.0, fontstyle="italic")):
        page.text(x, y + 0.08 + 0.12 * j, line, ha="left", va="center", fontsize=fs, color=style.INK_2,
                  fontstyle="italic")


def draw_column_heads(page: Page, y: float, g: BlockGeom, L: dict) -> None:
    page.text(g.x + g.dec_w / 2, y + 0.07, L["col_decision"], ha="center", va="center", fontsize=FS["small"],
              color=style.INK, fontweight="bold")
    page.text(g.film_x, y + 0.07, L["col_frames"], ha="left", va="center", fontsize=FS["small"], color=style.INK,
              fontweight="bold")
    lines = wrap(page.fig, L["col_strips"], FS["small"], g.w * 72.0, fontstyle="italic")
    if len(lines) > 2:  # COLHDR_H holds two lines; never cut the "not given to the model" sentence short
        raise ValueError(f"col_strips wraps to {len(lines)} lines, COLHDR_H holds 2: shorten it")
    for j, line in enumerate(lines):
        page.text(g.x, y + 0.175 + 0.095 * j, line, ha="left", va="center", fontsize=FS["small"], color=style.INK_2,
                  fontstyle="italic")


# --------------------------------------------------------------------------- #
# One key moment
# --------------------------------------------------------------------------- #
def draw_block(page: Page, y: float, ks: bd.KeyStep, L: dict, g: BlockGeom, last: bool, first: bool = False) -> None:
    """One key moment at ``y`` (inches from the top).  ``first`` tags the strip's sectors (model's view / display
    only); ``last`` puts the bearing axis under the future strip."""
    fig = page.fig
    # ---- header line: K badge, step, System2 text; the executed chunk right-aligned
    hax = page.pt_axes(g.x, y, g.w, g.hdr_h)
    mid = g.hdr_h * 72.0 * 0.5
    cd.key_badge(hax, 7.0, mid, ks.label, fs=g.fs_head - 0.1)
    step = L["step"].format(s=ks.step)
    hax.text(16.0, mid, step, ha="left", va="center", fontsize=g.fs_head, color=style.INK, fontweight="bold")
    x = 16.0 + cd.text_width_pt(fig, step, g.fs_head, fontweight="bold") + 0.9 * g.fs_head
    hax.text(x, mid, L["system2"], ha="left", va="center", fontsize=g.fs_head, color=style.INK_2)
    x += cd.text_width_pt(fig, L["system2"], g.fs_head) + 0.5 * g.fs_head
    s2 = f" {L['then']} ".join(f"“{s}”" for s in ks.system2_texts)
    hax.text(x, mid, s2, ha="left", va="center", fontsize=g.fs_head, color=style.INK)
    w_chips = pn.action_chips(hax, g.w * 72.0 - 1.0, mid, ks.executed_actions, size=g.chip, align="right")
    hax.text(g.w * 72.0 - 1.0 - w_chips - 3.5, mid, L["executed"], ha="right", va="center", fontsize=g.fs_head - 0.3,
             color=style.INK_2)

    # ---- decision image with the System2 pixel goal and the System1 mean path
    y_img = y + g.hdr_h
    dax = page.ax(g.x, y_img, g.dec_w, g.img_h)
    pn.draw_decision_image(dax, ks.decision_rgb, ks.pixel_goal_uv, ks.path_uv)
    page.text(g.x + g.dec_w / 2, y_img + g.img_h + g.dec_label_h * 0.5, L[ks.decision_image], ha="center",
              va="center", fontsize=g.fs_small, color=style.INK_2)

    # ---- filmstrip: the history fronts as given to System2 (slot 1 = oldest), then the current front
    lane = page.pt_axes(g.film_x, y_img, g.film_w, g.img_row_h, zorder=5)
    h_pt = g.img_row_h * 72.0
    K = int(ks.history_count)
    frames = [(j, ks.history_rgb[j]) for j in range(K)] + [(N_FRAMES - 1, ks.front_native)]
    for j, img in frames:
        x0 = g.film_x + j * (g.frame_w + g.frame_gap)
        fax = page.ax(x0, y_img + g.lane_h, g.frame_w, g.frame_h)
        now = j == N_FRAMES - 1
        pn.draw_image(fax, img, frame_color=style.INK if now else style.AXIS, frame_lw=0.8 if now else 0.5)
        xc = (x0 - g.film_x + g.frame_w / 2) * 72.0
        yb = h_pt - g.lane_h * 72.0 * 0.5
        if now:
            cd.key_badge(lane, xc, yb, L["now"], fs=g.badge_fs)
        else:
            cd.history_badge(lane, xc, yb, str(j + 1), j, num=max(K, 2), fs=g.badge_fs)
        step_label = str(ks.step if now else ks.history_steps[j])
        lane.text(xc, h_pt - (g.lane_h + g.frame_h + g.step_h * 0.5) * 72.0, step_label, ha="center", va="center",
                  fontsize=g.fs_small, color=style.INK_2)

    # ---- 360° strips: history (surroundings + predicted map + true directions), future (bins + path)
    y_b = y_img + g.img_row_h + g.gap_a
    bax = page.ax(g.x, y_b, g.w, g.h_hist)
    comp = bd.history_pred_composite(ks.hist_pred, ks.hist_none, ks.hist_mask)
    pn.draw_history_strip(bax, ks.pano_rgb, comp, bd.gt_history_peaks(ks.hist_gt_peak, ks.hist_mask),
                          bd.pred_history_peaks(ks.hist_pred, ks.hist_none, ks.hist_mask), g.el_hist, g.ring_w(340),
                          g.ring_w(272), sector_names=L["sectors"], label=L["hist_row"] if g.row_labels else None,
                          sector_notes=L["sector_notes"] if first else None)
    y_c = y_b + g.h_hist + g.gap_b
    cax = page.ax(g.x, y_c, g.w, g.h_fut)
    pn.draw_future_strip(cax, bd.future_bin_maps(ks.fut_pred), ks.path_cam, g.el_fut, g.ring_w(272),
                         label=L["fut_row"] if g.row_labels else None)
    if last:
        pn.ring_axis(cax, L["axis"], fs=g.fs_small + 0.3)


# --------------------------------------------------------------------------- #
# Legend and data flow
# --------------------------------------------------------------------------- #
LEGEND_KEYS = ("route", "ref", "start_goal", "key", "goal_px", "path", "slot", "hist", "gt", "fut", "actions")


def legend_keys(edge_marks: bool) -> tuple:
    """Legend rows; the edge-triangle row only on figures that draw one."""
    return LEGEND_KEYS[:9] + ("edge",) + LEGEND_KEYS[9:] if edge_marks else LEGEND_KEYS


def draw_legend(page: Page, x: float, y: float, w: float, L: dict, columns: int = 1, title: bool = True,
                keys: Sequence[str] = LEGEND_KEYS) -> float:
    """Legend entries (glyph + wrapped text), filled column by column; returns the height used (inches)."""
    fig = page.fig
    glyph_w, row_gap, fs = 17.0, 2.2, FS["legend"]
    col_w = w * 72.0 / columns
    rows = [(key, wrap(fig, L["legend"][key], fs, col_w - glyph_w - 6.0)) for key in keys]
    line_h = fs * 1.22
    heights = [max(len(ls) * line_h, 8.0) + row_gap for _, ls in rows]
    per_col = math.ceil(len(rows) / columns)
    col_heights = [sum(heights[c * per_col:(c + 1) * per_col]) for c in range(columns)]
    head = 11.0 if title else 0.0
    total_pt = head + max(col_heights)
    ax = page.pt_axes(x, y, w, total_pt / 72.0)
    if title:
        ax.text(0.0, total_pt - 4.0, L["legend_title"], ha="left", va="center", fontsize=FS["body"],
                fontweight="bold", color=style.INK)
    for c in range(columns):
        yy = total_pt - head
        for (key, lines), h in zip(rows[c * per_col:(c + 1) * per_col], heights[c * per_col:(c + 1) * per_col]):
            cy = yy - (h - row_gap) / 2
            x0 = c * col_w
            _legend_glyph(ax, key, x0 + 1.0, cy, glyph_w)
            for j, line in enumerate(lines):
                ax.text(x0 + glyph_w + 3.0, cy + (len(lines) - 1) * line_h / 2 - j * line_h, line, ha="left",
                        va="center", fontsize=fs, color=style.INK)
            yy -= h
    return total_pt / 72.0


def _legend_glyph(ax, key: str, x: float, y: float, w: float) -> None:
    if key == "route":
        ax.plot([x, x + w - 2], [y, y], color=style.INK, lw=0.95, solid_capstyle="round")
    elif key == "ref":
        ax.plot([x, x + w - 2], [y, y], color=style.MUTED, lw=0.9, ls=(0, (3.0, 1.8)))
    elif key == "start_goal":
        ax.plot([x + 2.5], [y], marker="o", ms=3.6, mfc="white", mec=style.INK, mew=0.9)
        ax.plot([x + 10.5], [y], marker="*", ms=6.6, mfc=style.INK, mec="white", mew=0.5)
    elif key == "key":
        cd.key_badge(ax, x + 6.0, y, "K1", fs=5.4)
    elif key == "goal_px":
        pn.goal_marker(ax, x + 6.0, y, size=6.4)
    elif key == "path":
        pn.path_dots(ax, [[x + 1.5 + 3.2 * i, y] for i in range(4)])
    elif key == "slot":
        for j, k in enumerate((0, 7)):
            cd.history_badge(ax, x + 3.6 + j * 8.4, y, str(k + 1), k, fs=pn.MIN_FS)
    elif key == "hist":
        n = 16
        for i in range(n):
            ax.add_patch(Rectangle((x + i * (w - 2) / n, y - 3.0), (w - 2) / n + 0.05, 6.0, lw=0, ec="none",
                                   fc=pn.HIST_CMAP((i + 0.5) / n)))
    elif key == "gt":
        pn.gt_circle(ax, x + 6.0, y)
    elif key == "edge":
        ax.plot([x + 3.5], [y + 0.6], marker="^", ms=4.0, mfc="white", mec=pn.GT_COLOR, mew=1.0)
        ax.plot([x + 11.0], [y - 0.6], marker="v", ms=4.0, mfc=pn.HIST_CMAP(0.95), mec="white", mew=0.5)
    elif key == "fut":
        for b, c in enumerate(pn.FUTURE_BIN_COLORS):
            ax.add_patch(Rectangle((x + b * (w - 2) / 4, y - 3.0), (w - 2) / 4 - 0.6, 6.0, lw=0, fc=c))
    elif key == "actions":
        pn.action_chip(ax, x, y, bd.FORWARD, size=6.4)
        pn.action_chip(ax, x + 7.4, y, bd.LEFT, size=6.4)


class _Flow:
    """Boxes, labels and arrows of the data-flow diagram, in point units."""

    def __init__(self, page: Page, ax, fs: float):
        self.page, self.ax, self.fs = page, ax, fs

    def half(self, text: str) -> float:
        """Half width of the box ``box`` draws around ``text``."""
        return cd.text_width_pt(self.page.fig, text, self.fs) / 2 + 2.6

    def box(self, cx, cy, text, fc="white") -> float:
        hw = self.half(text)
        self.ax.add_patch(FancyBboxPatch((cx - hw, cy - 4.4), 2 * hw, 8.8,
                                         boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=style.INK_2, lw=0.6,
                                         zorder=3))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=self.fs, color=style.INK, zorder=4)
        return hw

    def arrow(self, p, q) -> None:
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=4.6, color=style.INK_2, lw=0.6,
                                          shrinkA=0, shrinkB=0, zorder=2))

    def line(self, xs, ys) -> None:
        self.ax.plot(xs, ys, color=style.INK_2, lw=0.6, zorder=2, solid_joinstyle="miter")

    def label(self, cx, cy, text, ha="center", color=style.INK_2, **kw):
        return self.ax.text(cx, cy, text, ha=ha, va="center", fontsize=self.fs, color=color, zorder=4, **kw)

    def swatch(self, x0, y0, w, colors) -> None:
        for j, c in enumerate(colors):
            self.ax.add_patch(Rectangle((x0 + j * w / len(colors), y0 - 1.2), w / len(colors) - 0.4, 2.4, lw=0,
                                        fc=c))


def draw_dataflow(page: Page, x: float, y: float, w: float, L: dict) -> float:
    """Data-flow diagram for the left column; returns its height (inches).

    It is drawn in the order the deployment runs:

    1. System2 decides first, in text.
    2. The history head reads the frames and outputs the history affordance
       map and the memory M.
    3. The bridge adds M to Z, giving Z̃.
    4. System1 decodes Z̃ into the path and then the actions.
    5. The future head decodes the same Z̃, after that.

    No arrow leaves the future affordance map.
    """
    F = L["flow"]
    h_pt = FLOW_H_PT
    ax = page.pt_axes(x, y, w, h_pt / 72.0)
    W = w * 72.0
    f = _Flow(page, ax, FS["tiny"])
    ax.text(0.0, h_pt - 4.0, L["flow_title"], ha="left", va="center", fontsize=FS["body"], fontweight="bold",
            color=style.INK)
    xl, xr = 0.27 * W, 0.74 * W
    y1, y2, y3, y4, y5 = h_pt - 19.0, h_pt - 45.0, h_pt - 66.0, h_pt - 84.0, h_pt - 101.0  # swatches end at 3 pt
    hw = f.box(xl, y1, F["s2"], fc=pn.CHIP_FILL)  # System2 -> Z: the text decision comes first
    f.arrow((xl + hw, y1), (xr - 4.5, y1))
    f.label((xl + hw + xr - 4.5) / 2, y1 + 4.8, F["text"])
    f.label(xr, y1, F["z"], color=style.INK)
    f.label(xl, y2 + 10.2, F["frames"])
    hw = f.box(xl, y2, F["hh"])
    bw = f.box(xr, y2, F["bridge"])
    f.arrow((xr, y1 - 4.5), (xr, y2 + 4.4))
    f.arrow((xl + hw, y2), (xr - bw, y2))
    f.label((xl + hw + xr - bw) / 2, y2 + 4.6, F["mem"], color=style.INK)
    f.arrow((xl, y2 - 4.4), (xl, y3 + 3.8))
    for j, line in enumerate(F["hist"].split("\n")):
        f.label(xl, y3 - j * 6.6, line, color=style.INK)
    f.swatch(xl - 14.0, y3 - 12.5, 28.0, [pn.HIST_CMAP(0.85)])
    x_s1, x_fh = 0.52 * W, 0.84 * W
    f.line([xr, xr], [y2 - 4.4, y3 + 1.0])
    f.label(xr + 2.5, y3 + 3.2, F["zt"], ha="left", color=style.INK)
    f.line([x_s1, x_fh], [y3 + 1.0, y3 + 1.0])
    f.arrow((x_s1, y3 + 1.0), (x_s1, y4 + 4.4))
    f.arrow((x_fh, y3 + 1.0), (x_fh, y4 + 4.4))
    f.box(x_s1, y4, F["s1"])
    f.box(x_fh, y4, F["fh"])
    f.arrow((x_s1, y4 - 4.4), (x_s1, y5 + 3.8))
    f.label(x_s1, y5, F["path"], color=style.INK)
    for j, line in enumerate(F["fut"].split("\n")):
        f.label(x_fh, y5 - j * 6.6, line, color=style.INK)
    f.swatch(x_fh - 14.0, y5 - 12.5, 28.0, pn.FUTURE_BIN_COLORS)  # no arrow leaves the future map
    return h_pt / 72.0


def draw_dataflow_wide(page: Page, x: float, y: float, w: float, L: dict) -> float:
    """The same data flow as ``draw_dataflow``, laid out wide for the main figure's bottom band.

    Row 1: System2 -> Z -> Bridge <- M <- History head -> history affordance map.
    Rows 2-3: Z~ leaves the bridge downwards to System1 -> path -> actions and to
    the future head -> future affordance map, from which no arrow leaves.
    Returns its height (inches).
    """
    F = L["flow"]
    h_pt = FLOW_WIDE_H_PT
    ax = page.pt_axes(x, y, w, h_pt / 72.0)
    f = _Flow(page, ax, FS["tiny"])
    ax.text(0.0, h_pt - 4.0, L["flow_title"], ha="left", va="center", fontsize=FS["small"], fontweight="bold",
            color=style.INK)
    y1, y2, y3 = h_pt - 25.0, h_pt - 40.0, h_pt - 54.0

    def half(key: str) -> float:
        return f.half(F[key])

    x_s2 = 2.0 + half("s2")
    f.box(x_s2, y1, F["s2"], fc=pn.CHIP_FILL)
    x_z = x_s2 + half("s2") + 72.0
    f.arrow((x_s2 + half("s2"), y1), (x_z - 4.5, y1))
    f.label((x_s2 + half("s2") + x_z - 4.5) / 2, y1 + 5.4, F["text"])
    f.label(x_z, y1, F["z"], color=style.INK)
    x_b = x_z + 20.0 + half("bridge")
    f.box(x_b, y1, F["bridge"])
    f.arrow((x_z + 4.5, y1), (x_b - half("bridge"), y1))
    x_h = x_b + half("bridge") + 56.0 + half("hh")
    f.box(x_h, y1, F["hh"])
    f.arrow((x_h - half("hh"), y1), (x_b + half("bridge"), y1))  # M flows left, into the bridge
    f.label((x_h - half("hh") + x_b + half("bridge")) / 2, y1 + 5.4, F["mem"], color=style.INK)
    f.label(x_h, y1 + 10.0, F["frames"])
    f.arrow((x_h + half("hh"), y1), (x_h + half("hh") + 12.0, y1))
    t = f.label(x_h + half("hh") + 14.0, y1, F["hist"].replace("\n", " "), ha="left", color=style.INK)
    f.swatch(x_h + half("hh") + 18.0 + cd.text_width_pt(page.fig, t.get_text(), FS["tiny"]), y1, 20.0,
             [pn.HIST_CMAP(0.85)])

    f.line([x_b, x_b], [y1 - 4.4, y3])  # Z~ leaves the bridge
    f.label(x_b - 2.5, (y1 - 4.4 + y2) / 2, F["zt"], ha="right", color=style.INK)
    left = x_b + 26.0
    for yy, key, out, colors in ((y2, "s1", "path", None), (y3, "fh", "fut", pn.FUTURE_BIN_COLORS)):
        f.arrow((x_b, yy), (left, yy))
        f.box(left + half(key), yy, F[key])
        x_out = left + 2 * half(key)
        f.arrow((x_out, yy), (x_out + 12.0, yy))
        t = f.label(x_out + 14.0, yy, F[out].replace("\n", " "), ha="left", color=style.INK)
        if colors:  # nothing leaves the future affordance map
            f.swatch(x_out + 18.0 + cd.text_width_pt(page.fig, t.get_text(), FS["tiny"]), yy, 20.0, colors)
    return h_pt / 72.0


# --------------------------------------------------------------------------- #
# Route map column
# --------------------------------------------------------------------------- #
FLOOR_NOTE_MIN_M = 0.5  # horizontal length of the dotted (off-floor) route below which there is nothing to point at


def map_height(b: bd.Bundle, width: float, lo: float, hi: float) -> float:
    xz = np.concatenate([b.xz("route_xz"), b.xz("reference_path_xz"), b.xz("goal_xz")])
    span = xz.max(0) - xz.min(0) + 2 * (float(b.goal_radius_m) + 1.0)
    return float(np.clip(width * span[1] / max(span[0], 1e-6), lo, hi))


def off_level_steps(b: bd.Bundle, topdown) -> Optional[np.ndarray]:
    """Per route step, whether it is on another level than the map's (by the map's level ranges); None, no map."""
    if topdown is None:
        return None
    scene, level = topdown
    return scene.level_index(np.asarray(b.route_y, dtype=np.float64)) != level.index


def draw_map(page: Page, x: float, y: float, w: float, h: float, b: bd.Bundle, topdown, L: dict,
             labels: Optional[Sequence[str]] = None) -> None:
    """Route map; ``topdown`` = (scene, level) from ``resolve_level``, or None for an empty plate."""
    ax = page.ax(x, y, w, h)
    keys = [k for k in b.keys if labels is None or k.label in labels]
    pn.draw_route_map(ax, topdown[1] if topdown is not None else None, b.xz("route_xz"), b.xz("reference_path_xz"),
                      b.start_xz, b.goal_xz, float(b.goal_radius_m), [k.position_xz for k in keys],
                      [k.label for k in keys], L["start"], L["radius"], off_level=off_level_steps(b, topdown))
    if topdown is None:
        ax.text(0.5, 0.03, L["no_map"], transform=ax.transAxes, ha="center", va="bottom", fontsize=FS["tiny"],
                color=style.INK_2, fontstyle="italic")


def floor_note(b: bd.Bundle, topdown, L: dict) -> str:
    """The note under the map when the route changes floor, or "" when there is nothing on the map to point at.

    With a map, the route segments that touch another floor are drawn dotted
    (``panels.draw_route_line``); the note needs them to cover FLOOR_NOTE_MIN_M
    of horizontal route (a turn in place on another floor draws nothing).
    Without a map, the route's height range must exceed 1 m, and the note says
    that every step is drawn alike.
    """
    if topdown is None:
        y = np.asarray(b.route_y, dtype=np.float64)
        return L["floor_note_nomap"] if y.size and float(y.max() - y.min()) > 1.0 else ""
    off, xz = off_level_steps(b, topdown), b.xz("route_xz")
    if len(xz) < 2:
        return ""
    touches = off[1:] | off[:-1]  # segment j (step j -> j+1) is drawn dotted
    dotted = float(np.linalg.norm(np.diff(xz, axis=0), axis=1)[touches].sum())
    return L["floor_note"] if dotted >= FLOOR_NOTE_MIN_M else ""


def floor_note_lines(fig, b: bd.Bundle, topdown, L: dict) -> List[str]:
    note = floor_note(b, topdown, L)
    return wrap(fig, note, FS["tiny"], W_L * 72.0, fontstyle="italic") if note else []


def draw_left_column(page: Page, y: float, b: bd.Bundle, topdown, L: dict, keys: Sequence[str]) -> float:
    """Route map, legend (rows ``keys``) and data flow; returns the bottom (inches)."""
    page.text(X_L, y + 0.05, L["route"], ha="left", va="center", fontsize=FS["body"], fontweight="bold",
              color=style.INK)
    y_map = y + 0.12
    h_map = map_height(b, W_L, 1.4, 2.7)
    draw_map(page, X_L, y_map, W_L, h_map, b, topdown, L)
    y_next = y_map + h_map + 0.06
    for line in floor_note_lines(page.fig, b, topdown, L):
        page.text(X_L, y_next + 0.03, line, ha="left", va="center", fontsize=FS["tiny"], color=style.INK_2,
                  fontstyle="italic")
        y_next += 0.085
    y_next += 0.08
    h_leg = draw_legend(page, X_L, y_next, W_L, L, keys=keys)
    y_next += h_leg + 0.12
    return y_next + draw_dataflow(page, X_L, y_next, W_L, L)


# --------------------------------------------------------------------------- #
# Episode page
# --------------------------------------------------------------------------- #
def resolve_level(b: bd.Bundle, topdown_root=None):
    """(scene, level) of the bundle's top-down map, or None when [F] found no map for the scene."""
    if b.topdown["level_index"] is None:
        return None
    root = topdown_root or b.topdown.get("root")
    scene = load_topdown(b.topdown["scene"], root=root)
    idx = int(b.topdown["level_index"])
    if not 0 <= idx < len(scene.levels):
        raise ValueError(f"{b.ep_key}: topdown level_index {idx} not in 0..{len(scene.levels) - 1}")
    return scene, scene.levels[idx]


def figure_stem(b: bd.Bundle, out_dir) -> Path:
    return Path(out_dir) / b.category / f"{b.category_rank}_{b.ep_key}"


def _save(fig, stem: Path, lang: str, caption: str) -> List[str]:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    files = [stem.parent / f"{stem.name}_{lang}.pdf", stem.parent / f"{stem.name}_{lang}.png"]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)
    cap = stem.parent / f"{stem.name}_caption_{lang}.txt"
    cap.write_text(caption + "\n", encoding="utf-8")
    return [str(f) for f in files + [cap]]


def make_episode_figure(bundle_path, out_dir, lang: str = "en", verdicts: Optional[dict] = None,
                        topdown_root=None, void: Optional[Sequence[str]] = None) -> dict:
    """Render one episode page; returns {"files", "size_in", "claims", "warnings"}.

    ``void``: the reasons ``metrics.json`` gives for a void batch; the page and caption are stamped with them.
    """
    setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    L = LABELS[lang]
    b = bd.load_bundle(bundle_path)
    claims = claim_sentences(verdicts, lang)
    topdown = resolve_level(b, topdown_root)
    g = EPISODE_BLOCK
    edge = has_edge_marks(b.keys, g.el_hist)
    leg_keys = legend_keys(edge)

    fig = plt.figure(figsize=(FIG_W, 10.0))
    lines = header_lines(fig, b, L, void)
    n = len(b.keys)
    right_h = COLHDR_H + n * g.height + (n - 1) * BLOCK_GAP + AXIS_H if n else 0.5
    left_h = (0.12 + map_height(b, W_L, 1.4, 2.7) + 0.14 + 0.085 * len(floor_note_lines(fig, b, topdown, L))
              + draw_legend_height(fig, L, keys=leg_keys) + 0.12 + FLOW_H_PT / 72.0)
    height = header_height(lines) + max(right_h, left_h) + 0.03
    fig.set_size_inches(FIG_W, height)
    page = Page(fig, FIG_W, height)

    y_body = draw_header(page, b, lines, lang, L)
    if not n:
        draw_no_keys_note(page, g.x, y_body, g.w, L)
    else:
        draw_column_heads(page, y_body, g, L)
    y = y_body + COLHDR_H
    for i, ks in enumerate(b.keys):
        draw_block(page, y, ks, L, g, last=(i == n - 1), first=(i == 0))
        y += g.height + BLOCK_GAP
    draw_left_column(page, y_body, b, topdown, L, leg_keys)
    files = _save(fig, figure_stem(b, out_dir), lang, caption_text(b, lang, claims, edge_marks=edge, void=void))
    warnings = list(b.warnings) + over_budget(height, PAGE_MAX_H_IN, f"page [{lang}]")
    return {"files": files, "size_in": (FIG_W, round(height, 3)), "claims": claims, "warnings": warnings}


def draw_legend_height(fig, L: dict, w: float = W_L, columns: int = 1, keys: Sequence[str] = LEGEND_KEYS) -> float:
    """Height ``draw_legend`` will use (inches), measured with the same wrapping."""
    fs = FS["legend"]
    col_w = w * 72.0 / columns
    heights = [max(len(wrap(fig, L["legend"][k], fs, col_w - 23.0)) * fs * 1.22, 8.0) + 2.2 for k in keys]
    per_col = math.ceil(len(heights) / columns)
    return (11.0 + max(sum(heights[c * per_col:(c + 1) * per_col]) for c in range(columns))) / 72.0


# --------------------------------------------------------------------------- #
# Main figures: the is_main case of each category, compact; successes and failures in separate figures
# --------------------------------------------------------------------------- #
MAIN_KEYS = ("K2", "K3")  # the largest-turn moment and the 2/3 (F1: at or before the closest approach) moment
MAIN_GROUPS = (("T", ("T1", "T2", "T3")), ("F", ("F1", "F2")))
MAIN_X_MAP, MAIN_W_MAP = 0.0, 1.22
MAIN_X_KEYS = 1.32
MAIN_KEY_GAP = 0.12
MAIN_CASE_GAP = 0.12
CATEGORY_ORDER = {c: i for i, c in enumerate(bd.CATEGORIES)}

CAPTION_MAIN = {
    "en": ("Closed-loop reruns, one typical {group} episode per category ({cats}): what the model saw, predicted and "
           "decided at {nk} of the four pre-registered key moments of each episode, {labels} (all four are on the "
           "per-episode pages). Key moments are ready System2 calls, where System2 returned a pixel goal and the "
           "history head, bridge and future head ran. {key_rules}{exceptions} Encodings as in the per-episode pages: "
           "decision image with the System2 pixel goal (ring) and the System1 mean path (dots); the frames given to "
           "System2 (past frames 1 = oldest, then now); 360° strips centred straight ahead and re-rendered at the "
           "recorded position, where only the framed front part (the deployed camera's 79° view) was given to the "
           "model and the muted left, back and right are display only, not given to the model, with the predicted "
           "history affordance map (orange) and the true directions of the past frames (blue circles), and the "
           "predicted future affordance map for four time bins (aqua, darker = later) with the System1 path; chips = "
           "executed action chunk. The future affordance map is decoded from the same Z̃ as the System1 path, after "
           "the path, and does not feed the actions. Cases were chosen by the pre-registered rule (per category, the "
           "first candidate in typicality order whose rerun still meets the category definition), not by "
           "appearance."),
    "zh": ("闭环复跑，每类一集典型{group}案例（{cats}）：每集四个预注册关键时刻中的 {nk} 个（{labels}；四个全部见逐集页面）上"
           "模型看到了什么、预测了什么、决定了什么。关键时刻取自就绪调用（慢系统发了像素目标，历史头、桥、未来头都运行了）。"
           "{key_rules}{exceptions}编码同逐集页面：决策所用图，标出慢系统像素目标（圆环）与快系统均值路径（点）；送入慢系统的帧"
           "（历史帧 1 = 最早，末为当前帧）；以正前方为中心、在记录位置重渲染的 360° 条带，只有加框的前方部分（部署相机 79° 视场）"
           "输入了模型，压灰的左 / 后 / 右仅作展示、未输入模型，其上为预测历史 affordance map（橙）与历史帧真实方向（蓝圈），"
           "以及四个时段的预测未来 affordance map（青绿，越深越晚）与快系统路径；方块为实际执行的动作块。未来 affordance map "
           "与快系统路径由同一个 Z̃ 解码、在路径之后计算，不回流到动作。案例按预注册规则选取（每类按典型性排序取第一个复跑后仍满足"
           "类别定义的候选），不按好看挑。"),
}
MAIN_TEXT = {
    "en": {"group": {"T": "successful", "F": "failed"}, "nk": {1: "one", 2: "two", 3: "three", 4: "four"},
           "per_f1": "{rule} (for F1: {f1})", "no_keys": " {cat} has no ready call, so no key moment is shown.",
           "lt4": " {cat} has only {n} ready calls, so its key moments are all of them in call order; {labels} shown.",
           "lt4_one": " {cat} has only one ready call; K1 shown.",
           "branch": " In {cat}, {label} is {rule}."},
    "zh": {"group": {"T": "成功", "F": "失败"}, "nk": {1: "1", 2: "2", 3: "3", 4: "4"},
           "per_f1": "{rule}（F1 为{f1}）", "no_keys": "{cat} 没有就绪调用，不画关键时刻。",
           "lt4": "{cat} 只有 {n} 个就绪调用，其关键时刻即全部就绪调用（按调用序），图中为 {labels}。",
           "lt4_one": "{cat} 只有 1 个就绪调用，图中为 K1。",
           "branch": "{cat} 的 {label} 为{rule}。"},
}


def main_keys_of(b: bd.Bundle, wanted: Sequence[str]) -> List[bd.KeyStep]:
    """The wanted key moments.  An episode with fewer than 4 ready calls (branch "all_lt4") has K1..Kn in call order,
    so its wanted labels are topped up in K order to len(wanted); the caption says so (``main_exceptions``)."""
    chosen = [k for k in b.keys if k.label in wanted]
    for k in b.keys:
        if len(chosen) >= len(wanted):
            break
        if k not in chosen:
            chosen.append(k)
    return sorted(chosen, key=lambda k: k.index)


def main_rules_sentence(keys: Sequence[str], cats: Sequence[str], lang: str) -> str:
    """How the main figure's key moments are chosen when the rule's main clause applies (F1 differs for K3)."""
    R, M = KEY_RULES[lang], MAIN_TEXT[lang]
    parts = []
    for i, label in enumerate(keys):
        rules = list(dict.fromkeys(standard_branch(label, c) for c in cats))
        rule = R[rules[0]] if len(rules) == 1 else M["per_f1"].format(rule=R[standard_branch(label, "T1")],
                                                                     f1=R[standard_branch(label, "F1")])
        parts.append(R["first" if i == 0 else "next"].format(label=label, rule=rule))
    return R["join"].join(parts) + R["end"]


def main_exceptions(b: bd.Bundle, chosen: Sequence[bd.KeyStep], cat: str, lang: str) -> str:
    """What differs from ``main_rules_sentence`` for this case: no ready call, fewer than 4, or a fallback branch."""
    R, M = KEY_RULES[lang], MAIN_TEXT[lang]
    if not b.keys:
        return M["no_keys"].format(cat=cat)
    if any(k.branch == "all_lt4" for k in b.keys):
        return M["lt4_one" if len(b.keys) == 1 else "lt4"].format(cat=cat, n=len(b.keys),
                                                                   labels=", ".join(k.label for k in chosen))
    return "".join(M["branch"].format(cat=cat, label=k.label, rule=R[k.branch]) for k in chosen
                   if k.branch != standard_branch(k.label, cat))


def make_main_figures(bundle_paths: Sequence[Path], out_dir, langs: Sequence[str], verdicts: Optional[dict],
                      topdown_root=None, keys: Sequence[str] = MAIN_KEYS,
                      void: Optional[Sequence[str]] = None) -> dict:
    """One main figure per group of MAIN_GROUPS that has an is_main case: main_<group>_<lang>.*"""
    bundles = [bd.load_bundle(p) for p in bundle_paths]
    out = {"keys": list(keys), "figures": []}
    for group, cats in MAIN_GROUPS:
        members = sorted((b for b in bundles if b.membership(main=True)["category"] in cats),
                         key=lambda b: CATEGORY_ORDER[b.membership(main=True)["category"]])
        if not members:
            continue
        entry = {"group": group, "cases": [{"ep_key": b.ep_key, "category": b.membership(main=True)["category"]}
                                           for b in members], "files": [], "size_in": {}, "warnings": []}
        for lang in langs:
            res = make_main_figure(members, Path(out_dir), group, lang, verdicts, topdown_root, keys, void)
            entry["files"] += res["files"]
            entry["size_in"][lang] = res["size_in"]
            entry["warnings"] += res["warnings"]
        out["figures"].append(entry)
    if not out["figures"]:
        out["note"] = "no is_main bundles"
    return out


def make_main_figure(bundles: Sequence[bd.Bundle], out_dir: Path, group: str, lang: str, verdicts: Optional[dict],
                     topdown_root, keys: Sequence[str], void: Optional[Sequence[str]] = None) -> dict:
    setup(lang)
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    n_cols = len(keys)
    key_w = (FIG_W - MAIN_X_KEYS - (n_cols - 1) * MAIN_KEY_GAP) / n_cols
    geoms = [BlockGeom(MAIN_X_KEYS + c * (key_w + MAIN_KEY_GAP), key_w, img_h=0.42, hdr_h=0.15, dec_label_h=0.085,
                       film_gap_l=0.06, frame_gap=0.02, lane_h=0.1, step_h=0.075, gap_a=0.035, gap_b=0.025,
                       chip=6.6, fs_head=5.8, fs_small=pn.MIN_FS, badge_fs=pn.MIN_FS, row_labels=False)
             for c in range(n_cols)]
    chosen_keys = [main_keys_of(b, keys) for b in bundles]
    edge = has_edge_marks([k for chosen in chosen_keys for k in chosen], geoms[0].el_hist)
    leg_keys = legend_keys(edge)
    fig = plt.figure(figsize=(FIG_W, 10.0))
    text_w = (FIG_W - MAIN_X_KEYS) * 72.0
    heads = [(wrap(fig, outcome_line(b, L), FS["small"], text_w),
              wrap(fig, L["instruction"].format(text=" ".join(b.instruction.split())), FS["tiny"], text_w,
                   fontstyle="italic")) for b in bundles]
    case_h = [0.15 + 0.1 * len(out) + 0.095 * len(instr) + 0.03 + geoms[0].height for out, instr in heads]
    banner = wrap(fig, L["void"].format(reasons="; ".join(void)), FS["small"], FIG_W * 72.0) if void else []
    note = wrap(fig, L["strip_note"], FS["small"], FIG_W * 72.0, fontstyle="italic")
    legend_h = draw_legend_height(fig, L, w=FIG_W, columns=4, keys=leg_keys) - 11.0 / 72.0
    y0 = 0.11 * len(banner) + (0.05 if banner else 0.0)
    height = (y0 + sum(case_h) + MAIN_CASE_GAP * (len(bundles) - 1) + AXIS_H + 0.08 + 0.1 * len(note) + 0.06
              + legend_h + 0.1 + FLOW_WIDE_H_PT / 72.0 + 0.02)
    fig.set_size_inches(FIG_W, height)
    page = Page(fig, FIG_W, height)
    for j, line in enumerate(banner):
        page.text(0.0, 0.06 + 0.11 * j, line, ha="left", va="center", fontsize=FS["small"], fontweight="bold",
                  color="#d03b3b")
    y = y0
    for i, (b, (out_lines, instr_lines), h, chosen) in enumerate(zip(bundles, heads, case_h, chosen_keys)):
        last = i == len(bundles) - 1
        cat = b.membership(main=True)["category"]
        title = L["title"].format(cat=cat, name=CATEGORY_NAMES[lang][cat])
        page.text(0.0, y + 0.07, title, ha="left", va="center", fontsize=FS["head"], fontweight="bold",
                  color=style.INK)
        ids = L["ids"].format(scene=b.scene_id, ep=b.episode_id)
        if b.synthetic:
            ids = L["synthetic"] + "  ·  " + ids
        page.text(FIG_W, y + 0.07, ids, ha="right", va="center", fontsize=FS["tiny"],
                  color="#d03b3b" if b.synthetic else style.MUTED)
        yy = y + 0.15
        for line in out_lines:
            page.text(MAIN_X_KEYS, yy + 0.05, line, ha="left", va="center", fontsize=FS["small"], color=style.INK)
            yy += 0.1
        for line in instr_lines:
            page.text(MAIN_X_KEYS, yy + 0.045, line, ha="left", va="center", fontsize=FS["tiny"], color=style.INK_2,
                      fontstyle="italic")
            yy += 0.095
        yy += 0.03
        for c, (ks, g) in enumerate(zip(chosen, geoms)):
            draw_block(page, yy, ks, L, g, last=last, first=(i == 0 and c == 0))
        if not chosen:
            draw_no_keys_note(page, MAIN_X_KEYS, yy, FIG_W - MAIN_X_KEYS, L, fs=FS["small"])
        y_map, room = y + 0.17, yy + geoms[0].height - (y + 0.17)  # the map may use the text rows' height too
        map_h = min(room, map_height(b, MAIN_W_MAP, 0.7, room))
        draw_map(page, MAIN_X_MAP, y_map + (room - map_h) / 2, MAIN_W_MAP, map_h, b, resolve_level(b, topdown_root),
                 L, labels=[k.label for k in chosen])
        y += h + MAIN_CASE_GAP
        if not last:
            page.rule(y - MAIN_CASE_GAP / 2)
    y += AXIS_H - MAIN_CASE_GAP + 0.08
    page.rule(y - 0.04)
    for j, line in enumerate(note):  # the strips' muted sectors: not given to the model (display only)
        page.text(0.0, y + 0.05 + 0.1 * j, line, ha="left", va="center", fontsize=FS["small"], color=style.INK_2,
                  fontstyle="italic")
    y += 0.1 * len(note) + 0.06
    y += draw_legend(page, 0.0, y, FIG_W, L, columns=4, title=False, keys=leg_keys) + 0.1
    draw_dataflow_wide(page, 0.0, y, FIG_W, L)

    cats = [b.membership(main=True)["category"] for b in bundles]
    M = MAIN_TEXT[lang]
    sep = " " if lang == "en" else ""
    caption = CAPTION_MAIN[lang].format(
        group=M["group"][group], cats=("; " if lang == "en" else "；").join(f"{c} {CATEGORY_NAMES[lang][c]}"
                                                                          for c in cats),
        nk=M["nk"].get(len(keys), str(len(keys))), labels=", ".join(keys),
        key_rules=main_rules_sentence(keys, cats, lang),
        exceptions="".join(main_exceptions(b, chosen, c, lang) for b, chosen, c in zip(bundles, chosen_keys, cats)))
    parts = [void_sentence(void, lang), caption] + ([CAPTION_EDGE[lang]] if edge else [])
    caption = sep.join([p for p in parts if p] + claim_sentences(verdicts, lang))
    files = _save(fig, Path(out_dir) / f"main_{group}", lang, caption)
    return {"files": files, "size_in": (FIG_W, round(height, 3)),
            "warnings": over_budget(height, MAIN_MAX_H_IN, f"main figure {group} [{lang}]")}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", nargs="+", help="figure bundle(s): records/<ep_key>_bundle.json")
    src.add_argument("--records", help="records dir: every *_bundle.json in it")
    ver = ap.add_mutually_exclusive_group(required=True)
    ver.add_argument("--metrics", help="metrics/metrics.json; its 'verdicts' decide the caption claims")
    ver.add_argument("--no-verdicts", action="store_true", help="no claims in captions (layout work, synthetic data)")
    ap.add_argument("--out-dir", required=True, help="figures dir (writes <category>/<rank>_<ep_key>_<lang>.*)")
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=sorted(LABELS))
    ap.add_argument("--topdown-root", default=None,
                    help="top-down map root (default: the one recorded in each bundle)")
    ap.add_argument("--main", action="store_true",
                    help="also compose main/main_{T,F}_<lang>.* from the is_main bundles")
    ap.add_argument("--main-only", action="store_true",
                    help="only the main figures; an existing manifest.json keeps its pages and gets the new 'main'")
    ap.add_argument("--main-keys", default=",".join(MAIN_KEYS), help="key moments per case in the main figures")
    ap.add_argument("--allow-void", action="store_true",
                    help="draw a batch metrics.json marks void (a validity gate failed); every page is stamped")
    args = ap.parse_args(argv)

    paths = [bd.bundle_paths(p)[0] for p in args.bundle] if args.bundle else bd.find_bundles(args.records)
    if not paths:
        print("no bundles found", file=sys.stderr)
        return 2
    verdicts, void = None, []
    if args.metrics:
        verdicts, void = load_metrics(args.metrics)
        if void and not args.allow_void:
            print(f"{args.metrics}: the batch is void ({'; '.join(void)}); no figures drawn. --allow-void draws them "
                  "stamped VOID BATCH.", file=sys.stderr)
            return 3
        if void:
            print(f"WARNING: drawing a VOID batch ({'; '.join(void)}); every page and caption is stamped",
                  file=sys.stderr)
    out_dir = Path(args.out_dir)
    source = None
    if args.metrics:
        source = {"path": str(args.metrics), "sha256": _sha256(args.metrics), "verdicts": verdicts}
    claims = {lang: claim_sentences(verdicts, lang) for lang in args.lang}
    manifest = {"schema": MANIFEST_SCHEMA, "verdicts_source": source, "claims": claims, "void_batch": void or None,
                "figures": [], "main": None}
    for json_path in ([] if args.main_only else paths):
        npz_path = bd.bundle_paths(json_path)[1]
        entry = {"bundle": str(json_path), "bundle_sha256": {"json": _sha256(json_path), "npz": _sha256(npz_path)},
                 "files": [], "size_in": {}, "warnings": []}
        for lang in args.lang:
            res = make_episode_figure(json_path, out_dir, lang=lang, verdicts=verdicts, topdown_root=args.topdown_root,
                                      void=void or None)
            entry["files"] += res["files"]
            entry["size_in"][lang] = res["size_in"]
            entry["warnings"] += [w for w in res["warnings"] if w not in entry["warnings"]]
            print(f"{json_path.name} [{lang}] {res['size_in']} -> {res['files'][1]}")
        for w in entry["warnings"]:
            print(f"  WARNING {json_path.name}: {w}", file=sys.stderr)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        entry.update({k: meta[k] for k in ("ep_key", "category", "category_rank", "is_main",
                                           "predicate_holds_on_rerun")})
        entry["synthetic"] = bool(meta.get("synthetic", False))
        manifest["figures"].append(entry)
    if args.main or args.main_only:
        main_paths = [p for p in paths if bd.is_main_case(json.loads(p.read_text(encoding="utf-8")))]
        keys = tuple(k.strip() for k in args.main_keys.split(",") if k.strip())
        manifest["main"] = make_main_figures(main_paths, out_dir / "main", args.lang, verdicts, args.topdown_root, keys,
                                             void=void or None)
        manifest["main"].update(verdicts_source=source, claims=claims, void_batch=void or None)
        for entry in manifest["main"]["figures"]:
            print(f"main {entry['group']} {entry['size_in']} -> {entry['files'][1]}")
            for w in entry["warnings"]:
                print(f"  WARNING main {entry['group']}: {w}", file=sys.stderr)
    manifest_path = out_dir / "manifest.json"
    if args.main_only and manifest_path.is_file():  # keep the pages' record; replace only the main figures'
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old["main"] = manifest["main"]
        manifest = old
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
