#!/usr/bin/env python3
"""EXP-18 quantitative figure: how accurate the predicted affordance map is, per tier.

Layout (7.0 in wide, about 3.0 in tall, double column; read row by row):

  a  Joint PCK@8 (%) by tier                 b  Bearing error of the predicted peak
     A Training scenes    ◆--      ●-- 90.5     share   ___-------------             median (°)
       22 scenes · n=...                        <= x   /|    - - - - -        prediction  always
     B Held-out scenes    ◆--      ■-- 93.0           / |- -                             behind
       0    25   50   75   100                      0 |5  15    30    45       ●  A   2.1   14.3
                                                  miss > 5°                    ■  B   1.8   12.9
  c  Joint PCK@8 (%) by the view ...         d  Joint PCK@8 (%) by frames since the past position
        Front       n  Right     n  Back      n   tier lines (solid, one marker per tier),
     A  ○ n<100  0.0   ●      82.2  ◆  ●   92.9   always behind (thin dashed grey)
                  11           991         8,948  1-10  11-20  21-40  41-80  >80
     B  ■ far   14.5   ...
  legend: tier key + prediction · always behind · 95% CI · hollow = n < 100

What is plotted (every number comes from ``compute_metrics.py`` outputs):

* a  joint PCK@8 per tier with its 95% CI, both read from ``metrics.json``
  (``arms.vo.joint_pck8`` and, for the baseline, ``arms.floor.joint_pck8``).
  The prediction is the dump's ``vo`` arm = the deployed model's output.
* b  cumulative distribution of the bearing error of the predicted peak over
  the tier's GT-visible past positions (``slots.parquet``,
  ``pred_vo_bearing_err``; an undefined error counts as a miss at every x, as
  in ``bearing_err_le15_share``), x from 0 to 45 deg (ticks every 15 deg, the
  pre-registered 15 deg share); a thin reference line at 5 deg, the threshold
  beyond which the per-episode figures number a predicted peak as a miss.  The
  medians (``bearing_err_median_deg`` in ``metrics.json``) are where each curve
  crosses 50 %; they are listed in the table beside the plot, not marked on the
  curves (the tiers' medians lie within a fraction of a degree of each other).
* c  joint PCK@8 split by the GT view of the past position (``strata.gt_view``);
  each cell prints its value with n under it.  n < 100 cells hollow, grey and
  flagged "n<100" (ledger: report only, no conclusion).  A cell with n >= 100
  that lies ``FAR_BELOW_PT`` or more below every other view of its tier is
  annotated "far below the rest" where the annotation fits.  The
  always-behind baseline is 0 outside the back view by construction, so it is
  drawn only in the back column.
* d  joint PCK@8 by the age of the past position (``strata.age``: frames since
  it was visited; 1-10, 11-20, 21-40, 41-80, >80), or by distance with
  ``panel_d="distance"`` (``strata.distance``: 0-2, 2-5, 5-10, >10 m).  The
  always-behind baseline is one thin dashed grey line per tier without markers
  (a reference, not a series to identify); hollow n < 100 cells carry their n
  where the label fits.

Why age and not distance in d (A/B preliminary numbers, ``metrics_prelim_AB``):
over the cells with n >= 100, the prediction's PCK@8 moves by at most 5.4 pt
across the distance bins (A: 89.3-94.7, B: 91.2-95.4) but by up to 18.4 pt
across the age bins (B: 93.8 at 21-40 frames down to 75.5 beyond 80 frames),
and the always-behind baseline spans 50.5 / 53.4 pt across age (A / B; B falls
from 75.6 to 22.1) against 29.6 / 17.1 pt across distance.  Age therefore shows both where the prediction degrades (the oldest
positions) and why the baseline is not a fair ceiling (it works only for
positions just walked away from).  ``stats["family_spread"]`` recomputes this
comparison on whatever tiers are present, so the choice can be revisited when
C/D/E land.

Confidence intervals: a and the medians of b are the ledger's scene-cluster
bootstrap CIs as stored in ``metrics.json``.  For c and d the figure applies
the SAME resamples (``SceneBootstrap`` with the tier's stream
``SeedSequence(seed, spawn_key=(index in "ABCDE",))``, reps and seed from
``metrics.json``) to each cell; the headline PCK@8 CI is recomputed that way
and must reproduce ``metrics.json`` exactly (``stats["checks"]``).  Cell point
values must equal ``metrics.json`` strata values, or the figure refuses to draw
(mismatched files).

Figure policy (user decision, 2026-09-24): the figure never shows or mentions
poses, odometry or the pose-source ablation; the only prediction drawn is the
deployed model's (the ``vo`` arm).  The GT-pose arm's columns
(``pred_gt_*``) are never read.  Honesty that remains: the constant "always
behind" baseline is drawn next to every prediction it applies to; failures
(e.g. the front view) are shown, not cropped; n < 100 cells are drawn but
flagged; absent tiers are left out and named in the caption.

Colour: every coloured mark on this figure is a prediction, so all of them
are orange, the set-wide prediction colour (blue means ground truth in every
EXP-18 figure and never appears here).  Tiers are told apart inside the orange
family by ``TIER_STYLE``: a light-to-dark ramp A -> E (fixed per tier, so a
tier looks the same whichever tiers are present) plus one marker shape per
tier (never the diamond, which is the baseline).  Every tier is also labelled
directly (row labels in a and c, the key table in b, end labels in d), and
text never wears a tier colour.  Grey = the baseline; dark grey whiskers = CIs,
drawn over the marker with a white halo so an interval narrower than the
marker still shows.  ``style.TIER_COLORS`` is not used.

Coarse intervals: a scene bootstrap over S scenes has only C(2S-1, S) distinct
resamples (35 for S = 4); tiers with fewer than ``FEW_SCENES`` scenes are named
in the caption as having coarse intervals.

Usage (repo root on PYTHONPATH):
  python -m scripts.exp18.figures.fig_metrics --metrics-dir DIR [--slots FILE] [--panel-d age|distance]
      [--reps N] [--lang en|zh] --out <dir/stem>
Writes <stem>.pdf (vector, TrueType fonts embedded), <stem>.png (400 dpi) and
<stem>_caption.txt; prints the stats dict as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import compute_metrics as cm  # noqa: E402
from scripts.exp18 import geometry as geo  # noqa: E402
from scripts.exp18.figures import common_draw as cd  # noqa: E402
from scripts.exp18.figures import style  # noqa: E402

# The prediction drawn is always the deployed model's output (dump arm "vo"); "floor" = always behind.
ARM = "vo"
BASE = "floor"
SLOT_COLUMNS = ("tier", "scene", "gt_visible", "gt_class", "gt_dist_m", "age",
                f"pred_{ARM}_joint8", f"pred_{ARM}_bearing_err", f"pred_{BASE}_joint8", f"pred_{BASE}_bearing_err")
FAMILIES = {  # panel d: strata family in metrics.json -> (slot column, bins, inclusive upper bound)
    "age": ("age", cm.AGE_BINS, True),
    "distance": ("gt_dist_m", cm.DIST_BINS, False),
}
CDF_MAX_DEG = 45.0
POINT_TOL = 1e-9

# --------------------------------------------------------------------------- #
# Labels (lang -> key -> text); every string on the figure comes from here
# --------------------------------------------------------------------------- #
LABELS: Dict[str, Dict[str, object]] = {
    "en": {
        "title_a": "Joint PCK@8 (%) by tier",
        "title_b": "Bearing error of the predicted peak",
        "title_c": "Joint PCK@8 (%) by the view the past position lies in",
        "title_d": {"age": "Joint PCK@8 (%) by frames since the past position",
                    "distance": "Joint PCK@8 (%) by distance to the past position"},
        "tiers": {"A": "Training scenes", "B": "Held-out scenes", "C": "Unseen scenes",
                  "D": "HM3D (cross-dataset)", "E": "Designed routes"},
        "tier_sub": "{s} scenes · n = {n}",
        "views": ("Front", "Right", "Back", "Left"),
        "n_head": "n",
        "flag": "n<100",
        "x_b": "bearing error x (°)",
        "y_b": "share ≤ x (%)",
        "x_d": {"age": "frames since the past position", "distance": "distance to the past position (m)"},
        "y_d": "PCK@8 (%)",
        "bins": {"age": ("1–10", "11–20", "21–40", "41–80", ">80"),
                 "distance": ("0–2", "2–5", "5–10", ">10")},
        "median_head": "median (°)",
        "col_pred": "prediction",
        "col_base": ("always", "behind"),  # a two-line column head
        "ref_b": "miss > 5°",
        "base_short": "always behind",
        "far_below": ("far below", "the rest"),
        "flag_n": "n={n}",
        "legend_pred": "prediction (deployed model), shade and marker = tier",
        "legend_base": "always behind (constant baseline)",
        "legend_ci": "95% CI, scene bootstrap",
        "legend_flag": "hollow: n < 100, not interpreted",
    },
    "zh": {
        "title_a": "各层级的 joint PCK@8（%）",
        "title_b": "预测峰值的方位误差",
        "title_c": "按历史位置所在视角的 joint PCK@8（%）",
        "title_d": {"age": "按历史位置距今帧数的 joint PCK@8（%）",
                    "distance": "按到历史位置距离的 joint PCK@8（%）"},
        "tiers": {"A": "训练场景", "B": "留出场景", "C": "未见场景", "D": "HM3D（跨数据集）", "E": "设计路线"},
        "tier_sub": "{s} 个场景 · n = {n}",
        "views": ("前", "右", "后", "左"),
        "n_head": "n",
        "flag": "n<100",
        "x_b": "方位误差 x（°）",
        "y_b": "占比 ≤ x（%）",
        "x_d": {"age": "历史位置距今帧数", "distance": "到历史位置的距离（m）"},
        "y_d": "PCK@8 (%)",
        "bins": {"age": ("1–10", "11–20", "21–40", "41–80", ">80"),
                 "distance": ("0–2", "2–5", "5–10", ">10")},
        "median_head": "中位数（°）",
        "col_pred": "预测",
        "col_base": ("恒答", "正后方"),
        "ref_b": "偏差 > 5°",
        "base_short": "恒答正后方",
        "far_below": ("远低于", "其他视角"),
        "flag_n": "n={n}",
        "legend_pred": "预测（部署模型），深浅与形状 = 层级",
        "legend_base": "恒答正后方（常数基线）",
        "legend_ci": "95% 置信区间（场景 bootstrap）",
        "legend_flag": "空心：n < 100，不下结论",
    },
}

CAPTION = {
    "en": (
        "Accuracy of the predicted affordance map (the deployed model's output) on {tiers_text}.{missing} Joint "
        "PCK@8: a past position counts as correct when the view predicted to contain it is the right one and the "
        "predicted peak lies within 8 px (of 64) of the ground-truth peak in that view; the denominator is the past "
        "positions visible in some view (n). Orange marks and solid lines: prediction; each tier has its own shade "
        "(light to dark from A to E) and marker shape, and is labelled directly. Grey diamonds and dashed lines: the "
        "constant 'always behind' baseline, which always answers the back view with its peak at the view centre "
        "(pixel 32, 32) and never 'not visible'. Whiskers: 95% confidence intervals from a scene-cluster bootstrap "
        "({reps} resamples of whole scenes, seed {seed}).{coarse} (a) Joint PCK@8 per tier; the prediction's value is "
        "printed on the right, the baseline's beside its mark. (b) Share of visible past positions whose bearing "
        "error, the angle between the direction of the predicted peak and the true direction of the past position, "
        "is at most x (x up to 45°). Each curve crosses 50% at its median; the medians are listed on the right. The "
        "thin vertical line marks 5°, beyond which the per-episode figures number a predicted peak as a miss. "
        "(c) Joint PCK@8 by the view the past position lies in; each cell prints its value with n beneath. The "
        "baseline scores 0 outside the back view by construction and is drawn only there.{far} (d) {panel_d} The "
        "baseline is drawn as thin dashed lines without markers. In (c) and (d) the intervals apply each tier's "
        "bootstrap resamples to the cell. Hollow marks are cells with n < 100 (flagged in (c), labelled with their n "
        "in (d) where the label fits): their values are reported, not interpreted."
    ),
    "zh": (
        "预测 affordance map（部署模型的输出）在{tiers_text}上的准确率。{missing}joint PCK@8：预测所在视角正确、且预测"
        "峰值与该视角真值峰值相距不超过 8 像素（共 64）时，该历史位置记为正确；分母为在某个视角中可见的历史位置数（n）。"
        "橙色标记与实线：预测；每个层级有各自的深浅（A 到 E 由浅到深）和标记形状，并直接标注。灰色菱形与虚线：常数"
        "“恒答正后方”基线，即永远判为后视、峰值在视角中心像素 (32, 32)、从不判“不可见”。误差线：以场景为簇的 "
        "bootstrap 95% 置信区间（整场景重采样 {reps} 次，种子 {seed}）。{coarse}(a) 各层级的 joint PCK@8；预测的数值"
        "标在右侧，基线的数值标在其标记旁。(b) 方位误差（预测峰值方向与历史位置真实方向的夹角）不超过 x 的可见历史位置"
        "占比（x 至 45°）。各曲线与 50% 的交点即其中位数，数值列在右侧。细竖线标出 5°：逐集图中偏差超过 5° 的预测峰值"
        "会被单独编号。(c) 按历史位置所在视角的 joint PCK@8，每格给出数值，其下为 n；基线在后视以外按构造恒为 0，只在"
        "后视中画出。{far}(d) {panel_d}基线画成不带标记的细虚线。(c)(d) 的区间把各层级的 bootstrap 重采样用于每一格。"
        "空心标记为 n < 100 的格子（(c) 中标注 n<100，(d) 中放得下时标出其 n）：只报数，不下结论。"
    ),
}
COARSE_TEXT = {  # tiers with few scenes: their scene bootstrap has few distinct resamples
    "en": {"one": " Tier {tiers} has only {scenes} scenes, so its intervals are coarse (a bootstrap over {scenes} "
                  "scenes has only {distinct} distinct resamples).",
           "many": " Tiers {tiers} have only {scenes} scenes, so their intervals are coarse (a bootstrap over that "
                   "few scenes has only {distinct} distinct resamples)."},
    "zh": {"one": "{tiers} 层只有 {scenes} 个场景，其区间较粗（{scenes} 个场景的整场景重采样只有 {distinct} 种不同组合）。",
           "many": "{tiers} 层分别只有 {scenes} 个场景，其区间较粗（整场景重采样分别只有 {distinct} 种不同组合）。"},
}
FAR_TEXT = {
    "en": " A cell {pt} or more points below every other view of its tier is marked 'far below the rest'.",
    "zh": "比同层级其他所有视角都低至少 {pt} 个百分点的格子标注“远低于其他视角”。",
}
PANEL_D_TEXT = {
    "en": {"age": "Joint PCK@8 by frames elapsed since the past position was visited (the 8 past positions are "
                  "evenly spaced from the first frame of the episode to the previous frame).",
           "distance": "Joint PCK@8 by the straight-line distance from the current position to the past position."},
    "zh": {"age": "按历史位置距今帧数的 joint PCK@8（8 个历史位置在该集第一帧到上一帧之间等间隔取）。",
           "distance": "按当前位置到历史位置直线距离的 joint PCK@8。"},
}
TIERS_TEXT = {
    "en": {"A": "training scenes (A)", "B": "held-out scenes (B)", "C": "unseen scenes (C)",
           "D": "a second dataset, HM3D (D)", "E": "designed out-and-back and loop routes (E)"},
    "zh": {"A": "训练场景（A）", "B": "留出场景（B）", "C": "未见场景（C）", "D": "跨数据集 HM3D（D）",
           "E": "设计路线（去而复返、绕圈，E）"},
}
MISSING_TEXT = {  # absent tiers, by why they are absent
    "en": {"not dumped": " Tier{s} {tiers} not shown: no results yet.",
           "other": " Tier{s} {tiers} not shown: no deployed-model prediction in the metrics file."},
    "zh": {"not dumped": "{tiers} 层暂无结果，未画出。",
           "other": "{tiers} 层的指标文件中没有部署模型的预测，未画出。"},
}

# --------------------------------------------------------------------------- #
# Geometry of the page (inches, from the top-left corner)
# --------------------------------------------------------------------------- #
FIG_W = style.WIDTH_DOUBLE  # 7.0
L_W = 3.40  # left column: a above c (both one row per tier)
R_X = 3.70  # right column: b above d (line charts)
TITLE_H = 0.20
AXIS_H = 0.19  # tick labels under the dot panels
HDR_C = 0.14  # view names over the columns of c
ROW_GAP = 0.10  # between the top and bottom rows
LEGEND_ROW_H = 0.19  # one legend line (the legend wraps onto a second line when it does not fit)
PITCH_MIN, PITCH_MAX = 0.18, 0.50  # tier row pitch in a and c (rows fill the panel height)
PLOT_MIN_H = 0.82  # line charts b, d: plot height at least this
LINE_BOTTOM = 0.31  # tick labels + axis label under b, d
A_TRK_X, A_TRK_W = 1.13, 1.80  # a: value axis
C_X0 = 0.20  # c: first view column (tier letters left of it)
C_COL_GAP = 0.12
C_N_W = 0.25  # c: text column right of each track (value over n)
C_TXT_GAP = 0.03  # c: between a track and its text column
LINE_X = R_X + 0.40  # b, d: left edge of the plot (y tick labels + y label left of it)
LINE_W = 1.90  # b, d: plot width; the gutter right of it holds b's table and d's end labels
FS = {"letter": 7.6, "title": 6.8, "label": 6.4, "sub": 5.7, "tick": 5.9, "value": 6.2, "small": 5.7,
      "legend": 6.0}
DOT_MS = 4.6  # nominal prediction marker size (points; TIER_STYLE scales it per shape)
C_SCALE = 0.84  # c: markers a little smaller, the 0-100 tracks are only about 0.45 in wide
DIA_MS = 3.9  # baseline diamond
CI_LW = 0.65  # thin: the whisker is drawn over the marker and must not hide it
CI_HALO = 0.45  # extra width of the thin white halo under a CI whisker (points)
CAP_PT = 1.4  # half-length of a CI end cap
LINE_LW = 1.15
BASE_LW_D = 0.75  # d: the baseline's thin dashed lines
BASE_DASH = (0, (3.2, 1.8))
REF_DEG = 5.0  # b: reference line at the per-episode figures' numbered-miss threshold
FAR_BELOW_PT = 25.0  # c: annotate a cell this far below every other view of its tier
FEW_SCENES = 10  # tiers with fewer scenes get the "coarse intervals" caption sentence

# Tier identity inside the prediction's orange family (every coloured mark here is a prediction; blue means
# ground truth across the EXP-18 figure set and never appears on this figure).  Shades: a light-to-dark ramp
# in the hue of style.HEAT_CMAP_OPAQUE (the predicted affordance map), CIELAB L* 75 / 60 / 48 / 33 / 21, fixed
# per tier; shapes: one per tier, never the diamond (the baseline).  (shade, marker, size factor) — the factor
# evens out the visual size of the shapes.
TIER_STYLE = {
    "A": ("#f4a57c", "o", 1.00),
    "B": ("#eb6834", "s", 0.88),
    "C": ("#c24a17", "^", 1.10),
    "D": ("#8f2c08", "v", 1.10),
    "E": ("#5e1c05", "p", 1.06),
}


def tier_color(t: str) -> str:
    """The tier's shade of prediction orange (never blue: blue is ground truth in every EXP-18 figure)."""
    return TIER_STYLE.get(t, ("#eb6834", "o", 1.0))[0]


def tier_marker(t: str) -> Tuple[str, float]:
    """(matplotlib marker, size factor) of the tier."""
    _, m, f = TIER_STYLE.get(t, ("#eb6834", "o", 1.0))
    return m, f


def distinct_resamples(n_scenes: int) -> int:
    """Distinct multisets a scene bootstrap can draw from ``n_scenes`` scenes: C(2S-1, S)."""
    from math import comb
    return comb(2 * n_scenes - 1, n_scenes) if n_scenes > 0 else 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_metrics(metrics_dir) -> dict:
    return json.loads((Path(metrics_dir) / "metrics.json").read_text(encoding="utf-8"))


def load_slots(metrics_dir, path=None, columns: Sequence[str] = SLOT_COLUMNS) -> pd.DataFrame:
    """The columns of ``slots.parquet`` (or ``.csv.gz``) the figure needs; never the GT-pose arm's."""
    d = Path(metrics_dir)
    candidates = [Path(path)] if path else [d / "slots.parquet", d / "slots.csv.gz"]
    for c in candidates:
        if c.is_file():
            if c.suffix == ".parquet":
                return pd.read_parquet(c, columns=list(columns))
            return pd.read_csv(c, usecols=list(columns))
    raise FileNotFoundError(f"no slots table among {[str(c) for c in candidates]} (written by compute_metrics.py)")


def usable_tiers(metrics: dict) -> Tuple[List[str], Dict[str, str]]:
    """Tiers with a deployed-model prediction, in A-E order; the rest with the reason they are left out."""
    tiers, skipped = [], {}
    for t in cm.TIER_ORDER:
        b = metrics.get("tiers", {}).get(t) or {}
        if not b.get("present"):
            skipped[t] = "not dumped"
        elif ARM not in (b.get("arms") or {}):
            skipped[t] = "no deployed-model prediction in metrics.json"
        else:
            tiers.append(t)
    return tiers, skipped


def _stat(block: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    lo, hi = (block.get("ci95") or [None, None])[:2]
    return block.get("value"), lo, hi


def _ci(reps: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    lo, hi = cm.ci(reps)
    return lo, hi


def _bins(values: np.ndarray, bins, inclusive: bool) -> np.ndarray:
    """Strata labels exactly as compute_metrics.strata_block assigns them ('' = no bin)."""
    lab = np.full(len(values), "", dtype=object)
    for lo, hi, name in bins:
        sel = (values >= lo) & (values <= hi) if inclusive else (values >= lo) & (values < hi)
        lab[sel] = name
    return lab


def _cell(boot, hit, base_hit, sel, stored: Optional[dict], where: str) -> dict:
    """One stratum cell: n, prediction PCK@8 with bootstrap CI, baseline PCK@8; checked against metrics.json."""
    n = int(sel.sum())
    cell = {"n": n, "allowed": n >= cm.N_MIN, "pred": None, "lo": None, "hi": None, "base": None}
    if n:
        p, reps = boot.ratio(hit & sel, sel)
        lo, hi = _ci(reps)
        cell.update(pred=p, lo=lo, hi=hi, base=float(base_hit[sel].mean()))
    if stored is not None:
        if int(stored.get("n", -1)) != n:
            raise ValueError(f"{where}: n = {n} from slots, {stored.get('n')} in metrics.json (mismatched files?)")
        for key, mine in (("vo", cell["pred"]), ("floor", cell["base"])):
            ref = (stored.get(key) or {}).get("joint_pck8")
            if n and (ref is None or abs(ref - mine) > POINT_TOL):
                raise ValueError(f"{where}: {key} PCK@8 {mine} from slots, {ref} in metrics.json")
    return cell


def compute_figure_data(metrics: dict, slots: pd.DataFrame, panel_d: str = "age", reps: Optional[int] = None,
                        seed: Optional[int] = None) -> dict:
    """Everything the figure draws, per tier (no drawing).  Raises if slots and metrics.json disagree."""
    if panel_d not in FAMILIES:
        raise ValueError(f"panel_d must be one of {sorted(FAMILIES)}")
    bcfg = metrics.get("bootstrap") or {}
    reps = int(reps if reps is not None else bcfg.get("reps", 10000))
    seed = int(seed if seed is not None else bcfg.get("seed", 0))
    same_resamples = reps == int(bcfg.get("reps", -1)) and seed == int(bcfg.get("seed", -1))
    tiers, skipped = usable_tiers(metrics)
    out = {"tiers": [], "skipped": skipped, "reps": reps, "seed": seed, "panel_d": panel_d, "per_tier": {},
           "checks": {}, "family_spread": {}}
    for t in tiers:
        blk = metrics["tiers"][t]
        df = slots[slots["tier"] == t]
        if not len(df):
            skipped[t] = "no rows in the slots table"
            continue
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cm.TIER_ORDER.index(t),)))
        boot = cm.SceneBootstrap(df["scene"].to_numpy(), reps, rng)
        vis = df["gt_visible"].to_numpy().astype(bool)
        hit = df[f"pred_{ARM}_joint8"].to_numpy().astype(bool)
        base_hit = df[f"pred_{BASE}_joint8"].to_numpy().astype(bool)
        arms = blk["arms"]

        # ---- a: headline, as stored; recomputed with the same resamples as a check
        pred = _stat(arms[ARM]["joint_pck8"])
        base = _stat(arms[BASE]["joint_pck8"])
        p, r = boot.ratio(hit & vis, vis)
        lo, hi = _ci(r)
        if abs(p - pred[0]) > POINT_TOL:
            raise ValueError(f"tier {t}: PCK@8 {p} from slots, {pred[0]} in metrics.json (mismatched files?)")
        ci_diff = max(abs(lo - pred[1]), abs(hi - pred[2])) if None not in (lo, hi, pred[1], pred[2]) else None
        out["checks"][t] = {"headline_point_equal": True, "headline_ci_max_abs_diff": ci_diff,
                            "headline_ci_reproduced": bool(same_resamples and ci_diff is not None and ci_diff < 1e-9)}
        if same_resamples and not out["checks"][t]["headline_ci_reproduced"]:
            raise ValueError(f"tier {t}: bootstrap CI {lo, hi} does not reproduce metrics.json {pred[1:]}")

        # ---- b: bearing-error distributions (undefined error = a miss at every x)
        err = df[f"pred_{ARM}_bearing_err"].to_numpy()[vis]
        berr = df[f"pred_{BASE}_bearing_err"].to_numpy()[vis]
        err_sorted = np.sort(np.where(np.isfinite(err), err, np.inf))
        berr_sorted = np.sort(np.where(np.isfinite(berr), berr, np.inf))
        med = _stat(arms[ARM]["bearing_err_median_deg"])
        bmed = _stat(arms[BASE]["bearing_err_median_deg"])
        finite = err[np.isfinite(err)]
        if finite.size and abs(float(np.percentile(finite, 50)) - med[0]) > 1e-9:
            raise ValueError(f"tier {t}: median bearing error does not match metrics.json")
        le15 = float(np.mean(err_sorted <= 15.0)) if err_sorted.size else None
        ref15 = arms[ARM]["bearing_err_le15_share"]["value"]
        if le15 is not None and ref15 is not None and abs(le15 - ref15) > POINT_TOL:
            raise ValueError(f"tier {t}: share <= 15 deg {le15} from slots, {ref15} in metrics.json")

        # ---- c: by GT view; d: by the chosen family (both checked against metrics.json strata)
        strata = blk.get("strata") or {}
        cls = df["gt_class"].to_numpy().astype(np.int64)
        view = {}
        for v, name in enumerate(geo.VIEW_NAMES):
            view[name] = _cell(boot, hit, base_hit, vis & (cls == v + 1), (strata.get("gt_view") or {}).get(name),
                               f"tier {t} view {name}")
        families = {}
        for fam, (col, bins, inclusive) in FAMILIES.items():
            lab = _bins(df[col].to_numpy(), bins, inclusive)
            families[fam] = [(name, _cell(boot, hit, base_hit, vis & (lab == name),
                                          (strata.get(fam) or {}).get(name), f"tier {t} {fam} {name}"))
                             for _, _, name in bins]
        out["tiers"].append(t)
        out["per_tier"][t] = {
            "n": int(vis.sum()), "n_scenes": int(blk.get("n_scenes", boot.S)), "n_clips": blk.get("n_clips"),
            "pred": pred, "base": base, "median": med, "base_median": bmed,
            "err_sorted": err_sorted, "base_err_sorted": berr_sorted, "le15": le15,
            "view": view, "family": families[panel_d], "families": families,
        }
    out["family_spread"] = family_spread(out)
    return out


def family_spread(data: dict) -> dict:
    """Per family: the largest spread (max - min PCK@8, pt) across bins with n >= 100, prediction and baseline."""
    res = {}
    for fam in FAMILIES:
        per = {}
        for t in data["tiers"]:
            cells = [c for _, c in data["per_tier"][t]["families"][fam] if c["allowed"] and c["pred"] is not None]
            if len(cells) >= 2:
                p = [c["pred"] for c in cells]
                b = [c["base"] for c in cells]
                per[t] = {"pred_spread_pt": 100 * (max(p) - min(p)), "base_spread_pt": 100 * (max(b) - min(b))}
        res[fam] = {"per_tier": per,
                    "max_pred_spread_pt": max((v["pred_spread_pt"] for v in per.values()), default=None),
                    "max_base_spread_pt": max((v["base_spread_pt"] for v in per.values()), default=None)}
    return res


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
class Page:
    """Axes and text placement in inches from the top-left corner."""

    def __init__(self, fig, width: float, height: float):
        self.fig, self.w, self.h = fig, width, height

    def ax(self, x: float, y_top: float, w: float, h: float, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h], **kw)

    def text(self, x: float, y: float, s: str, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)


def fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def fmt_n(n: int) -> str:
    """One format for every n on the figure: thousands separator, no abbreviation."""
    return f"{int(n):,}"


def panel_title(page: Page, x: float, y: float, letter: str, title: str) -> None:
    page.text(x, y, letter, ha="left", va="baseline", fontsize=FS["letter"], fontweight="bold", color=style.INK)
    page.text(x + 0.14, y, title, ha="left", va="baseline", fontsize=FS["title"], color=style.INK)


def tier_mark(ax, x, y, t: str, hollow: bool = False, scale: float = 1.0, zorder: float = 5, **kw):
    """The tier's prediction marker (its shape in its shade of orange); hollow = an n < 100 cell."""
    m, f = tier_marker(t)
    ms = DOT_MS * f * scale
    if hollow:
        return ax.plot([x], [y], m, ms=ms - 0.4, mfc="white", mec=style.MUTED, mew=0.8, zorder=zorder,
                       clip_on=False, **kw)
    return ax.plot([x], [y], m, ms=ms, mfc=tier_color(t), mec="white", mew=0.6, zorder=zorder, clip_on=False, **kw)


def mark_radius_pt(t: str, scale: float = 1.0) -> float:
    return DOT_MS * tier_marker(t)[1] * scale / 2.0 + 0.3


def dot(ax, x, y, color, hollow: bool = False, ms: float = DOT_MS, zorder: float = 5, **kw):
    """A plain circle (legend glyphs)."""
    if hollow:
        return ax.plot([x], [y], "o", ms=ms - 0.4, mfc="white", mec=style.MUTED, mew=0.8, zorder=zorder,
                       clip_on=False, **kw)
    return ax.plot([x], [y], "o", ms=ms, mfc=color, mec="white", mew=0.6, zorder=zorder, clip_on=False, **kw)


def diamond(ax, x, y, hollow: bool = False, ms: float = DIA_MS, zorder: float = 4, **kw):
    if hollow:
        return ax.plot([x], [y], "D", ms=ms - 0.4, mfc="white", mec=style.MUTED, mew=0.7, zorder=zorder,
                       clip_on=False, **kw)
    return ax.plot([x], [y], "D", ms=ms, mfc=style.MUTED, mec="white", mew=0.5, zorder=zorder, clip_on=False, **kw)


def whisker(ax, lo, hi, y, color, lw: float = CI_LW, horizontal: bool = True, zorder: float = 7,
            cap_pt: float = CAP_PT, halo: bool = False):
    """CI from lo to hi at position y, with short end caps (``cap_pt`` half-length; axes limits must be final).

    Drawn as ONE thin path over the marker, so an interval narrower than the marker still shows as a small
    I-beam on it instead of hiding behind it (``halo``: a white edge, for glyphs outside the plots).
    """
    if lo is None or hi is None:
        return
    from matplotlib import patheffects as pe

    dx, dy = cd.pts_to_data(ax, cap_pt, cap_pt)
    nan = np.nan
    if horizontal:
        xs = [lo, hi, nan, lo, lo, nan, hi, hi]
        ys = [y, y, nan, y - dy, y + dy, nan, y - dy, y + dy]
    else:
        xs = [y, y, nan, y - dx, y + dx, nan, y - dx, y + dx]
        ys = [lo, hi, nan, lo, lo, nan, hi, hi]
    effects = [pe.withStroke(linewidth=lw + CI_HALO, foreground="white")] if halo else None
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=zorder, clip_on=False, path_effects=effects)


def style_line_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=FS["tick"], colors=style.INK_2, length=2.2, width=0.5, pad=1.8)
    ax.set_facecolor("none")
    ax.grid(True, color=style.GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def row_axes_limits(ax, n: int, height_in: float, pitch: float) -> None:
    """Rows 0..n-1 top to bottom, ``pitch`` inches apart, centred in an axes ``height_in`` tall."""
    units = height_in / pitch
    c = (n - 1) / 2.0
    ax.set_ylim(c + units / 2.0, c - units / 2.0)


def pts_per_unit(ax, axis: str = "x") -> float:
    """Points per data unit along ``axis`` (linear axes; call after the limits are final)."""
    dx, dy = cd.pts_to_data(ax, 1.0, 1.0)
    return 1.0 / (dx if axis == "x" else abs(dy))


def _pct(v):
    return None if v is None else 100 * v


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_panel_a(page: Page, y_top: float, h: float, data: dict, L: dict) -> None:
    tiers = data["tiers"]
    n = len(tiers)
    rows_h = h - TITLE_H - AXIS_H
    pitch = min(PITCH_MAX, rows_h / max(n, 1))
    panel_title(page, 0.0, y_top + 0.13, "a", L["title_a"])
    ax = page.ax(A_TRK_X, y_top + TITLE_H, A_TRK_W, rows_h)
    ax.set_xlim(0, 100)
    row_axes_limits(ax, n, rows_h, pitch)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=FS["tick"], colors=style.INK_2, length=0, pad=2.0)
    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=style.GRID if x else style.AXIS, lw=0.45 if x else 0.6, zorder=0)
    ax.set_facecolor("none")
    px = pts_per_unit(ax, "x")
    for i, t in enumerate(tiers):
        d = data["per_tier"][t]
        (pv, plo, phi), (bv, blo, bhi) = d["pred"], d["base"]
        diamond(ax, 100 * bv, i)
        whisker(ax, _pct(blo), _pct(bhi), i, style.MUTED)
        tier_mark(ax, 100 * pv, i, t)
        whisker(ax, _pct(plo), _pct(phi), i, style.INK_2)
        right = 100 * max(v for v in (pv, phi) if v is not None)
        ax.text(max(100.0, right) + 5.0 / px, i, fmt_pct(pv), ha="left", va="center", fontsize=FS["value"],
                color=style.INK, clip_on=False)
        left = 100 * (blo if blo is not None else bv) - 3.2 / px
        ax.text(left, i, fmt_pct(bv), ha="right", va="center", fontsize=FS["small"], color=style.INK_2,
                clip_on=False)
        # tier label left of the axis: bold letter, name, and "scenes · n" under it
        y_mid = y_top + TITLE_H + rows_h / 2 + (i - (n - 1) / 2) * pitch
        page.text(0.0, y_mid - 0.034, t, ha="left", va="center", fontsize=FS["label"] + 0.4, fontweight="bold",
                  color=style.INK)
        page.text(0.13, y_mid - 0.034, L["tiers"][t], ha="left", va="center", fontsize=FS["label"], color=style.INK)
        page.text(0.13, y_mid + 0.053, L["tier_sub"].format(s=d["n_scenes"], n=fmt_n(d["n"])), ha="left",
                  va="center", fontsize=FS["sub"], color=style.INK_2)


def _far_below(tier_views: Dict[str, dict], name: str) -> bool:
    """Is this view's cell (n >= 100) at least FAR_BELOW_PT below every other view of the tier with n >= 100?"""
    c = tier_views[name]
    if not c["allowed"] or c["pred"] is None:
        return False
    others = [o["pred"] for k, o in tier_views.items() if k != name and o["allowed"] and o["pred"] is not None]
    return bool(others) and 100 * (min(others) - c["pred"]) >= FAR_BELOW_PT


def draw_panel_c(page: Page, y_top: float, h: float, data: dict, L: dict, notes: dict) -> None:
    tiers = data["tiers"]
    n = len(tiers)
    fig = page.fig
    rows_h = h - TITLE_H - HDR_C - AXIS_H
    pitch = min(PITCH_MAX, rows_h / max(n, 1))
    panel_title(page, 0.0, y_top + 0.13, "c", L["title_c"])
    col_w = (L_W - C_X0 - 3 * C_COL_GAP) / 4.0
    trk_w = col_w - C_N_W - C_TXT_GAP
    y_rows = y_top + TITLE_H + HDR_C
    for i, t in enumerate(tiers):
        y_mid = y_rows + rows_h / 2 + (i - (n - 1) / 2) * pitch
        page.text(0.0, y_mid, t, ha="left", va="center", fontsize=FS["label"] + 0.4, fontweight="bold",
                  color=style.INK)
    fs_far = FS["small"] - 0.2
    far_lines = L["far_below"]
    far_w = max(cd.text_width_pt(fig, s, fs_far, fontstyle="italic") for s in far_lines)
    for v, name in enumerate(geo.VIEW_NAMES):
        x0 = C_X0 + v * (col_w + C_COL_GAP)
        page.text(x0 + trk_w / 2, y_rows - 0.06, L["views"][v], ha="center", va="baseline", fontsize=FS["label"],
                  color=style.INK)
        page.text(x0 + col_w, y_rows - 0.06, L["n_head"], ha="right", va="baseline", fontsize=FS["sub"],
                  color=style.INK_2, fontstyle="italic")
        ax = page.ax(x0, y_rows, trk_w, rows_h)
        ax.set_xlim(0, 100)
        row_axes_limits(ax, n, rows_h, pitch)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"])
        ax.tick_params(axis="x", labelsize=FS["tick"], colors=style.INK_2, length=0, pad=2.0)
        for x in (0, 50, 100):
            ax.axvline(x, color=style.GRID if x else style.AXIS, lw=0.45 if x else 0.6, zorder=0)
        ax.set_facecolor("none")
        px = pts_per_unit(ax, "x")
        py = pts_per_unit(ax, "y")
        x_txt = 100 + (C_TXT_GAP + C_N_W) * 72.0 / px  # right edge of the text column
        for i, t in enumerate(tiers):
            views = data["per_tier"][t]["view"]
            c = views[name]
            # text column: the value (ink) over n (grey); both grey for an n < 100 cell
            val = fmt_pct(c["pred"])
            ax.text(x_txt, i - 3.3 / py, val, ha="right", va="center", fontsize=FS["value"],
                    color=style.INK if c["allowed"] else style.MUTED, clip_on=False)
            ax.text(x_txt, i + 3.6 / py, fmt_n(c["n"]), ha="right", va="center", fontsize=FS["small"],
                    color=style.INK_2 if c["allowed"] else style.MUTED, clip_on=False)
            if c["pred"] is None:
                continue
            if v == geo.BACK and c["base"] is not None:
                diamond(ax, 100 * c["base"], i, hollow=not c["allowed"], ms=DIA_MS - 0.3)
            if c["allowed"]:
                tier_mark(ax, 100 * c["pred"], i, t, scale=C_SCALE)
                whisker(ax, _pct(c["lo"]), _pct(c["hi"]), i, style.INK_2)
            else:
                tier_mark(ax, 100 * c["pred"], i, t, hollow=True, scale=C_SCALE)
                # flag on the side of the track away from the dot
                right = c["pred"] < 0.5
                xf = 100 * c["pred"] + (1 if right else -1) * 4.4 / px
                ax.text(xf, i, L["flag"], ha="left" if right else "right", va="center", fontsize=FS["small"],
                        color=style.MUTED, fontstyle="italic", clip_on=False)
            if _far_below(views, name):
                # "far below the other views", right of the cell's whisker, left of the text column, if it fits
                start = 100 * max(v_ for v_ in (c["pred"], c["hi"]) if v_ is not None)
                start_pt = start * px + max(CAP_PT + 1.0, mark_radius_pt(t, C_SCALE)) + 2.0
                txt_w = max(cd.text_width_pt(fig, val, FS["value"]), cd.text_width_pt(fig, fmt_n(c["n"]), FS["small"]))
                end_pt = x_txt * px - txt_w - 2.5
                key = f"{t}:{name}"
                if start_pt + far_w <= end_pt:
                    ax.text(start_pt / px, i, "\n".join(far_lines), ha="left", va="center", fontsize=fs_far,
                            color=style.INK_2, fontstyle="italic", linespacing=1.0, clip_on=False)
                    notes["far_below_drawn"].append(key)
                else:
                    notes["far_below_not_drawn"].append(key)


def _line_axes(page: Page, y_top: float, h: float):
    plot_h = h - TITLE_H - LINE_BOTTOM
    ax = page.ax(LINE_X, y_top + TITLE_H, LINE_W, plot_h)
    style_line_axes(ax)
    return ax, plot_h


def draw_panel_b(page: Page, y_top: float, h: float, data: dict, L: dict) -> None:
    panel_title(page, R_X, y_top + 0.13, "b", L["title_b"])
    ax, plot_h = _line_axes(page, y_top, h)
    ax.set_xlim(0, CDF_MAX_DEG)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 15, 30, 45])  # even steps; 15 deg is the ledger's pre-registered share
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(L["x_b"], fontsize=FS["small"], color=style.INK_2, labelpad=1.5)
    ax.set_ylabel(L["y_b"], fontsize=FS["small"], color=style.INK_2, labelpad=2.0)
    x = np.linspace(0.0, CDF_MAX_DEG, 1801)
    base_curves = []
    # the numbered-miss threshold of the per-episode figures: a thin labelled reference line
    ax.axvline(REF_DEG, color=style.INK_2, lw=0.5, zorder=1.5)
    px = pts_per_unit(ax, "x")
    py = pts_per_unit(ax, "y")
    ax.text(REF_DEG + 2.0 / px, 3.0 / py, L["ref_b"], ha="left", va="bottom", fontsize=FS["small"],
            color=style.INK_2, path_effects=cd.HALO_THIN, zorder=7)
    for t in data["tiers"]:  # all baselines first (under every prediction curve), then the predictions A -> E
        e = data["per_tier"][t]["base_err_sorted"]
        if e.size:
            y = 100.0 * np.searchsorted(e, x, side="right") / e.size
            ax.plot(x, y, color=style.MUTED, lw=0.85, ls=BASE_DASH, zorder=2, solid_capstyle="butt")
            base_curves.append(y)
    for t in data["tiers"]:
        e = data["per_tier"][t]["err_sorted"]
        if e.size:
            y = 100.0 * np.searchsorted(e, x, side="right") / e.size
            ax.plot(x, y, color=tier_color(t), lw=LINE_LW, zorder=3, solid_joinstyle="round")
    # direct label for the dashed group: right-aligned near the axis end, under the lowest curve over its span
    if base_curves:
        x1 = CDF_MAX_DEG - 1.0
        x0 = x1 - cd.text_width_pt(page.fig, L["base_short"], FS["small"]) / px
        span = (x >= x0) & (x <= x1)
        y_low = min(float(c[span].min()) for c in base_curves)
        ax.text(x1, y_low - 2.5 / py, L["base_short"], ha="right", va="top",
                fontsize=FS["small"], color=style.INK_2, path_effects=cd.HALO_THIN, zorder=7)
    draw_median_table(page, y_top + TITLE_H, plot_h, data, L)


def draw_median_table(page: Page, y_top: float, plot_h: float, data: dict, L: dict) -> None:
    """Key table right of b: tier (its line, marker, letter) | prediction median | always-behind median (deg)."""
    fig = page.fig
    x0 = LINE_X + LINE_W + 0.12
    w = FIG_W - x0
    tiers = data["tiers"]
    heads = list(L["col_base"])
    n_lines = 1 + len(heads) + len(tiers)
    line_h = min(0.118, plot_h / (n_lines + 0.4))
    ax = page.ax(x0, y_top, w, plot_h)
    ax.set_xlim(0, w * 72.0)
    ax.set_ylim(plot_h * 72.0, 0)
    ax.axis("off")
    lh = line_h * 72.0
    y = (plot_h * 72.0 - n_lines * lh) / 2 + 0.5 * lh
    col_base = w * 72.0 - 1.0
    wv = cd.text_width_pt(fig, "99.9", FS["value"])
    base_head_w = max(cd.text_width_pt(fig, s, FS["small"]) for s in heads)
    col_pred = col_base - max(base_head_w, wv) - 5.0
    pred_head_w = cd.text_width_pt(fig, L["col_pred"], FS["small"])
    span_l = min(col_pred - pred_head_w, col_pred - wv)
    ax.text((span_l + col_base) / 2, y, L["median_head"], ha="center", va="center", fontsize=FS["small"],
            color=style.INK_2)
    ax.plot([span_l, col_base], [y + 0.55 * lh, y + 0.55 * lh], color=style.AXIS, lw=0.5)
    for j, s in enumerate(heads):
        y += lh
        ax.text(col_base, y, s, ha="right", va="center", fontsize=FS["small"], color=style.INK_2)
        if j == len(heads) - 1:
            ax.text(col_pred, y, L["col_pred"], ha="right", va="center", fontsize=FS["small"], color=style.INK_2)
    for t in tiers:
        y += lh
        d = data["per_tier"][t]
        ax.plot([1.0, 11.0], [y, y], color=tier_color(t), lw=LINE_LW + 0.2, solid_capstyle="butt")
        tier_mark(ax, 6.0, y, t)
        ax.text(14.0, y, t, ha="left", va="center", fontsize=FS["value"], fontweight="bold", color=style.INK)
        mv = d["median"][0]
        bv = d["base_median"][0]
        ax.text(col_pred, y, "—" if mv is None else f"{mv:.1f}", ha="right", va="center", fontsize=FS["value"],
                color=style.INK)
        ax.text(col_base, y, "—" if bv is None else f"{bv:.1f}", ha="right", va="center", fontsize=FS["value"],
                color=style.INK_2)


class Obstacles:
    """Boxes (in points, display space) that a label placed later must avoid."""

    def __init__(self, ax):
        self.ax = ax
        self.boxes: List[Tuple[float, float, float, float]] = []
        self.marks: List[np.ndarray] = []  # centres of point marks (a label must sit nearest its own mark)

    def to_pt(self, x, y) -> np.ndarray:
        return self.ax.transData.transform((x, y)) * 72.0 / self.ax.figure.dpi

    def point(self, x, y, r_pt: float) -> None:
        cx, cy = self.to_pt(x, y)
        self.boxes.append((cx - r_pt, cy - r_pt, cx + r_pt, cy + r_pt))
        self.marks.append(np.array([cx, cy]))

    def segment(self, x0, y0, x1, y1, r_pt: float = 0.9) -> None:
        a, b = self.to_pt(x0, y0), self.to_pt(x1, y1)
        n = max(2, int(np.hypot(*(b - a)) / 1.5) + 1)
        for t in np.linspace(0.0, 1.0, n):
            cx, cy = a + t * (b - a)
            self.boxes.append((cx - r_pt, cy - r_pt, cx + r_pt, cy + r_pt))

    def box(self, b) -> None:
        self.boxes.append(tuple(b))

    def overlap(self, b) -> float:
        total = 0.0
        for o in self.boxes:
            w = min(b[2], o[2]) - max(b[0], o[0])
            h = min(b[3], o[3]) - max(b[1], o[1])
            if w > 0 and h > 0:
                total += w * h
        return total


CLEAR_PT2 = 1.0
_SPOTS = ((0, 1, "center", "bottom"), (0, -1, "center", "top"), (1, 0, "left", "center"),
          (-1, 0, "right", "center"), (0.72, 0.72, "left", "bottom"), (-0.72, 0.72, "right", "bottom"),
          (0.72, -0.72, "left", "top"), (-0.72, -0.72, "right", "top"))


def _box_dist(b, p) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def place_label(ax, obst: Obstacles, x, y, text: str, fs: float, radii=(4.6, 8.0), require_clear: bool = False,
                **text_kw) -> bool:
    """Put ``text`` next to the data point (x, y) where it overlaps the fewest obstacles, inside the axes.

    A spot that sits nearer another point mark than its own is never taken (the
    label would read as belonging to that mark).  ``require_clear``: draw only
    if some spot overlaps (almost) nothing — at most ``CLEAR_PT2`` square points, a
    graze of an obstacle's padding; returns whether the label was drawn.
    """
    fig = ax.figure
    w = cd.text_width_pt(fig, text, fs, **{k: v for k, v in text_kw.items() if k in ("fontstyle", "fontweight")})
    h = 0.92 * fs
    cx, cy = obst.to_pt(x, y)
    bb = ax.get_window_extent()
    frame = np.array([bb.x0, bb.y0, bb.x1, bb.y1]) * 72.0 / fig.dpi
    best = None
    for rank, rr in enumerate(radii):
        for order, (ux, uy, ha, va) in enumerate(_SPOTS):
            ax_, ay = cx + ux * rr, cy + uy * rr
            x0 = ax_ - w / 2 if ha == "center" else (ax_ if ha == "left" else ax_ - w)
            y0 = ay - h / 2 if va == "center" else (ay if va == "bottom" else ay - h)
            b = (x0, y0, x0 + w, y0 + h)
            inside_w = max(0.0, min(b[2], frame[2]) - max(b[0], frame[0]))
            inside_h = max(0.0, min(b[3], frame[3]) - max(b[1], frame[1]))
            outside = w * h - inside_w * inside_h
            own = _box_dist(b, (cx, cy))
            if any(_box_dist(b, m) < own + 1.0 for m in obst.marks if np.hypot(*(m - (cx, cy))) > 0.5):
                continue
            score = (obst.overlap(b) + 2.0 * outside, rank, order)
            if best is None or score < best[0]:
                best = (score, ux * rr, uy * rr, ha, va, b)
    if best is None or (require_clear and best[0][0] > CLEAR_PT2):
        return False
    _, dx, dy, ha, va, b = best
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fs,
                path_effects=cd.HALO_THIN, zorder=8, annotation_clip=False, **text_kw)
    obst.box(b)
    return True


def draw_panel_d(page: Page, y_top: float, h: float, data: dict, L: dict, notes: dict) -> None:
    fam = data["panel_d"]
    panel_title(page, R_X, y_top + 0.13, "d", L["title_d"][fam])
    ax, plot_h = _line_axes(page, y_top, h)
    names = L["bins"][fam]
    nb = len(names)
    ax.set_xlim(-0.45, nb - 0.55)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(nb))
    ax.set_xticklabels(names)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(False, axis="x")
    ax.set_xlabel(L["x_d"][fam], fontsize=FS["small"], color=style.INK_2, labelpad=1.5)
    ax.set_ylabel(L["y_d"], fontsize=FS["small"], color=style.INK_2, labelpad=2.0)
    tiers = data["tiers"]
    nt = len(tiers)
    # tiers side by side within a bin, about one marker apart, so whiskers never sit on top of each other
    step = min(0.15, 0.6 / max(nt - 1, 1)) if nt > 1 else 0.0
    py = pts_per_unit(ax, "y")
    obst = Obstacles(ax)
    ends = []  # (x, y) of each tier's last point, tier
    base_ends = []
    flags = []
    series = []
    for j, t in enumerate(tiers):
        cells = data["per_tier"][t]["family"]
        off = (j - (nt - 1) / 2) * step
        pts = [(b + off, b, c) for b, (_, c) in enumerate(cells) if c["pred"] is not None]
        series.append((t, off, pts))
        # the baseline: one thin dashed grey line per tier at the bin centres, no markers (a reference only);
        # a segment into an n < 100 cell is faded
        for (_, ba, ca), (_, bb_, cb) in zip(pts[:-1], pts[1:]):
            faded = not (ca["allowed"] and cb["allowed"])
            ya, yb = 100 * ca["base"], 100 * cb["base"]
            ax.plot([ba, bb_], [ya, yb], color=style.MUTED, lw=BASE_LW_D, ls=BASE_DASH,
                    alpha=0.45 if faded else 1.0, zorder=2)
            obst.segment(ba, ya, bb_, yb, r_pt=0.7)
        if pts:
            base_ends.append((pts[-1][1], 100 * pts[-1][2]["base"]))
    for t, off, pts in series:  # predictions over every baseline
        col = tier_color(t)
        for (xa, _, ca), (xb, _, cb) in zip(pts[:-1], pts[1:]):
            faded = not (ca["allowed"] and cb["allowed"])
            ya, yb = 100 * ca["pred"], 100 * cb["pred"]
            ax.plot([xa, xb], [ya, yb], color=col, lw=LINE_LW * (0.75 if faded else 1.0),
                    alpha=0.5 if faded else 1.0, zorder=3, solid_capstyle="round")
            obst.segment(xa, ya, xb, yb)
        for xb, _, c in pts:
            yb = 100 * c["pred"]
            if c["allowed"]:
                tier_mark(ax, xb, yb, t, zorder=6)
                if c["lo"] is not None and c["hi"] is not None:
                    whisker(ax, 100 * c["lo"], 100 * c["hi"], xb, style.INK_2, horizontal=False)
                    obst.segment(xb, 100 * c["lo"], xb, 100 * c["hi"], r_pt=CAP_PT + 0.4)
            else:
                tier_mark(ax, xb, yb, t, hollow=True, zorder=6)
                flags.append((xb, yb, c["n"], t))
            obst.point(xb, yb, mark_radius_pt(t))
        if pts:
            ends.append((pts[-1][0], 100 * pts[-1][2]["pred"], t))
    # end labels right of the plot, dodged vertically, thin leaders from each series' last point
    x_lab = nb - 0.55 + 0.08
    items = [(x, y, t) for x, y, t in ends]
    if base_ends:
        items.append((max(x for x, _ in base_ends), float(np.mean([y for _, y in base_ends])), None))
    if items:
        heights = [FS["value"] + 1.2 if t else FS["small"] + 1.2 for _, _, t in items]
        placed = cd.dodge_1d([y * py for _, y, _ in items], heights, 0.0, 100 * py, 0.6)
        dx_pt = 1.0 / pts_per_unit(ax, "x")
        for (x, y, t), yp in zip(items, placed):
            yp /= py
            if abs(yp - y) > 0.5 / py:
                ax.plot([x + 3.2 * dx_pt, x_lab - 1.5 * dx_pt], [y, yp], color=style.AXIS, lw=0.5, clip_on=False,
                        zorder=1)
            if t:
                ax.text(x_lab, yp, t, ha="left", va="center", fontsize=FS["value"], fontweight="bold",
                        color=style.INK, clip_on=False)
            else:
                ax.text(x_lab, yp, L["base_short"], ha="left", va="center", fontsize=FS["small"],
                        color=style.INK_2, clip_on=False)
    # n < 100: the hollow marker (legend) always; its n too where the label fits beside it without covering
    # anything or sitting nearer another tier's mark
    for xb, yb, n_cell, t in flags:
        ok = place_label(ax, obst, xb, yb, L["flag_n"].format(n=fmt_n(n_cell)), FS["small"], radii=(4.6, 6.5),
                         require_clear=True, color=style.MUTED, fontstyle="italic")
        (notes["flags_d_drawn"] if ok else notes["flags_d_not_drawn"]).append(f"{t}:{n_cell}")


# --------------------------------------------------------------------------- #
# Legend
# --------------------------------------------------------------------------- #
LEG_INNER = 3.0  # glyph -> text (points)
LEG_GAP_MIN, LEG_GAP_MAX = 12.0, 22.0  # between entries


def legend_entries(fig, L: dict, tiers: Sequence[str]) -> list:
    """[(draw(ax, x, y), glyph width pt, text, text width pt)]: tier key + prediction, baseline, CI, hollow."""
    key_w = []
    for t in tiers:
        key_w.append(2 * mark_radius_pt(t) + 1.6 + cd.text_width_pt(fig, t, FS["legend"], fontweight="bold"))
    key_gap = 4.0
    pred_w = sum(key_w) + key_gap * (len(tiers) - 1)

    def pred(ax, x, y):
        for t, kw in zip(tiers, key_w):
            r = mark_radius_pt(t)
            tier_mark(ax, x + r, y, t)
            ax.text(x + 2 * r + 1.6, y, t, ha="left", va="center", fontsize=FS["legend"], fontweight="bold",
                    color=style.INK)
            x += kw + key_gap

    def base(ax, x, y):
        ax.plot([x, x + 14.0], [y, y], color=style.MUTED, lw=0.85, ls=BASE_DASH)
        diamond(ax, x + 7.0, y)

    def ci(ax, x, y):
        whisker(ax, x + 0.5, x + 13.5, y, style.INK_2)

    def flag(ax, x, y):
        dot(ax, x + 7.0, y, style.INK_2, hollow=True)

    out = []
    for draw, gw, key in ((pred, pred_w, "legend_pred"), (base, 14.0, "legend_base"), (ci, 14.0, "legend_ci"),
                          (flag, 14.0, "legend_flag")):
        out.append((draw, gw, L[key], cd.text_width_pt(fig, L[key], FS["legend"])))
    return out


def legend_rows(entries: list, width_pt: float) -> List[list]:
    """Greedy line breaking: as many entries per line as fit at the minimum gap."""
    rows, cur, used = [], [], 0.0
    for e in entries:
        w = e[1] + LEG_INNER + e[3]
        if cur and used + LEG_GAP_MIN + w > width_pt:
            rows.append(cur)
            cur, used = [], 0.0
        used += (LEG_GAP_MIN if cur else 0.0) + w
        cur.append(e)
    if cur:
        rows.append(cur)
    return rows


def draw_legend(page: Page, y_top: float, rows: List[list]) -> None:
    """The legend lines across the full width, each centred."""
    h_in = LEGEND_ROW_H * len(rows)
    ax = page.ax(0.0, y_top, FIG_W, h_in)
    w_pt, h_pt = FIG_W * 72.0, h_in * 72.0
    ax.set_xlim(0, w_pt)
    ax.set_ylim(h_pt, 0)
    ax.axis("off")
    for r, row in enumerate(rows):
        y = (r + 0.55) * LEGEND_ROW_H * 72.0
        total = sum(gw + LEG_INNER + tw for _, gw, _, tw in row)
        gap = min(LEG_GAP_MAX, (w_pt - 4.0 - total) / (len(row) - 1)) if len(row) > 1 else 0.0
        x = (w_pt - (total + gap * (len(row) - 1))) / 2.0
        for draw, gw, text, tw in row:
            draw(ax, x, y)
            ax.text(x + gw + LEG_INNER, y, text, ha="left", va="center", fontsize=FS["legend"], color=style.INK)
            x += gw + LEG_INNER + tw + gap


# --------------------------------------------------------------------------- #
# Layout and entry point
# --------------------------------------------------------------------------- #
def layout_heights(n_tiers: int, legend_lines: int = 1) -> Tuple[float, float, float]:
    """(top row, bottom row, total) heights in inches for ``n_tiers`` tier rows."""
    n = max(n_tiers, 1)
    line_min = TITLE_H + PLOT_MIN_H + LINE_BOTTOM
    top = max(TITLE_H + n * PITCH_MIN + AXIS_H, line_min)
    bottom = max(TITLE_H + HDR_C + n * PITCH_MIN + AXIS_H, line_min)
    return top, bottom, top + ROW_GAP + bottom + LEGEND_ROW_H * legend_lines + 0.01


def _join(items: Sequence[str], lang: str) -> str:
    if lang == "zh":
        return "、".join(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def caption_text(data: dict, lang: str, notes: Optional[dict] = None) -> str:
    names = TIERS_TEXT[lang]
    tiers_text = _join([names[t] for t in data["tiers"]], lang)
    missing = ""
    for why in ("not dumped", "other"):
        absent = [t for t in cm.TIER_ORDER if t not in data["tiers"]
                  and (data["skipped"].get(t, "not dumped") == "not dumped") == (why == "not dumped")]
        if absent:
            joined = ", ".join(absent) if lang == "en" else "、".join(absent)
            missing += MISSING_TEXT[lang][why].format(s="s" if len(absent) > 1 else "", tiers=joined)
    reps = f"{data['reps']:,}" if lang == "en" else str(data["reps"])
    few = [t for t in data["tiers"] if 0 < data["per_tier"][t]["n_scenes"] < FEW_SCENES]
    coarse = ""
    if few:
        scenes = [data["per_tier"][t]["n_scenes"] for t in few]
        distinct = [distinct_resamples(s) for s in scenes]
        form = "one" if len(few) == 1 else "many"
        coarse = COARSE_TEXT[lang][form].format(tiers=_join(few, lang), scenes=_join([str(s) for s in scenes], lang),
                                                distinct=_join([f"{d:,}" if lang == "en" else str(d)
                                                                for d in distinct], lang))
    far = ""
    if notes and notes.get("far_below_drawn"):
        far = FAR_TEXT[lang].format(pt=f"{FAR_BELOW_PT:g}")
    return CAPTION[lang].format(tiers_text=tiers_text, missing=missing, reps=reps, seed=data["seed"],
                                panel_d=PANEL_D_TEXT[lang][data["panel_d"]], coarse=coarse, far=far)


def _jsonable(data: dict) -> dict:
    """Stats for the caller / CLI (the sorted error arrays dropped)."""
    out = {k: v for k, v in data.items() if k != "per_tier"}
    out["per_tier"] = {}
    for t, d in data["per_tier"].items():
        out["per_tier"][t] = {
            "n": d["n"], "n_scenes": d["n_scenes"], "n_clips": d["n_clips"],
            "pck8": d["pred"], "baseline_pck8": d["base"],
            "bearing_err_median_deg": d["median"], "baseline_bearing_err_median_deg": d["base_median"],
            "share_le15": d["le15"],
            "view": d["view"], data["panel_d"]: dict(d["family"]),
        }
    return out


def make_metrics_figure(metrics_dir, out_stem="metrics", lang: str = "en", panel_d: str = "age", slots_path=None,
                        reps: Optional[int] = None) -> dict:
    """Render the quantitative figure from a compute_metrics.py output dir.

    Returns {"files": [pdf, png, caption], "stats": {...}, "size_in": (w, h)}.
    ``reps`` overrides the bootstrap size for the c/d cell intervals (default:
    metrics.json's, which also makes the headline CI check exact).
    ``stats["drawn"]`` lists the optional labels that were / were not drawn
    (c: "far below" annotations; d: n labels of n < 100 cells).
    """
    if lang not in LABELS:
        raise ValueError(f"lang must be one of {sorted(LABELS)}")
    metrics = load_metrics(metrics_dir)
    slots = load_slots(metrics_dir, slots_path)
    data = compute_figure_data(metrics, slots, panel_d=panel_d, reps=reps)
    if not data["tiers"]:
        raise ValueError(f"no tier with a deployed-model prediction in {metrics_dir}: {data['skipped']}")

    cd.setup(lang)
    import matplotlib.pyplot as plt  # after setup(): Agg backend, fonts registered

    L = LABELS[lang]
    fig = plt.figure(figsize=(FIG_W, 3.0))
    rows = legend_rows(legend_entries(fig, L, data["tiers"]), FIG_W * 72.0 - 4.0)
    top_h, bot_h, height = layout_heights(len(data["tiers"]), len(rows))
    fig.set_size_inches(FIG_W, height)
    page = Page(fig, FIG_W, height)
    notes = {"far_below_drawn": [], "far_below_not_drawn": [], "flags_d_drawn": [], "flags_d_not_drawn": []}
    y_bot = top_h + ROW_GAP
    draw_panel_a(page, 0.0, top_h, data, L)
    draw_panel_b(page, 0.0, top_h, data, L)
    draw_panel_c(page, y_bot, bot_h, data, L, notes)
    draw_panel_d(page, y_bot, bot_h, data, L, notes)
    draw_legend(page, y_bot + bot_h, rows)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [out.parent / (out.name + ".pdf"), out.parent / (out.name + ".png")]
    fig.savefig(files[0], dpi=300, bbox_inches=None)
    fig.savefig(files[1], dpi=400, bbox_inches=None)
    plt.close(fig)
    cap_path = out.parent / (out.name + "_caption.txt")
    cap_path.write_text(caption_text(data, lang, notes) + "\n", encoding="utf-8")
    files.append(cap_path)
    stats = _jsonable(data)
    stats["drawn"] = notes
    return {"files": [str(f) for f in files], "stats": stats, "size_in": (FIG_W, height)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics-dir", default=os.environ.get("EXP18_METRICS_DIR"),
                    help="compute_metrics.py output dir (metrics.json + slots.parquet); default $EXP18_METRICS_DIR")
    ap.add_argument("--slots", default=None, help="slots table if not <metrics-dir>/slots.parquet")
    ap.add_argument("--panel-d", default="age", choices=sorted(FAMILIES))
    ap.add_argument("--reps", type=int, default=None, help="bootstrap reps for c/d cells (default: metrics.json's)")
    ap.add_argument("--lang", default="en", choices=sorted(LABELS))
    ap.add_argument("--out", default="metrics_figure", help="output stem (writes .pdf, .png, _caption.txt)")
    args = ap.parse_args(argv)
    if not args.metrics_dir:
        ap.error("--metrics-dir (or $EXP18_METRICS_DIR) is required")
    res = make_metrics_figure(args.metrics_dir, out_stem=args.out, lang=args.lang, panel_d=args.panel_d,
                              slots_path=args.slots, reps=args.reps)
    for f in res["files"]:
        print(f)
    print(json.dumps(res["stats"], indent=1, ensure_ascii=False, default=float))
    print("size_in", tuple(round(v, 3) for v in res["size_in"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
