#!/usr/bin/env python3
"""EXP-21 parameter sensitivity, IEEE double column (7.0 in wide): SR / SPL / NE
against K, S and M, with the pre-registered 95% paired-bootstrap CI as error bars.

Small multiples, 3 rows x 3 columns.  One metric per panel and therefore one y axis
per panel: SR, SPL and NE do not share a scale and a twin axis would be unreadable,
so they are faceted instead of overlaid.  Each panel carries two series -- the RTX
4090, which is EXP-21's pre-registered platform, and the Ascend 910B, which repeated
all nine points and is shown as a cross-platform replication, not as a second
measurement of the same thing (the ledger's EXP-22 section 3: the two machines'
absolute numbers are not comparable, only the shape of the response is).

What the error bars are: the 95% CI of the PAIRED difference against that platform's
own default point, from 2000 resamples of the same 500 episodes
(``scripts/tools/paired_closed_loop_bootstrap.py``).  They are drawn around the
point's own value, so a bar that does not cross the dashed default line is the
pre-registered "sensitive" verdict and one that crosses it is "within noise".  A CI
of a difference is not a CI of the level: two points whose bars overlap may still
differ significantly, which is why the verdict is annotated rather than left to the
reader's eye.

The default value of each parameter (K=8, S=32, M=10) is drawn as the dashed
reference line, at its own platform's level, and is not a data point of the sweep --
it is the control every point is paired against.

Usage (repo root on PYTHONPATH)::

    python -m scripts.exp21.figures.fig_sensitivity \\
        --results docs/experiments/exp21-sensitivity-results.json \\
        --out-dir <dir> [--lang en zh] [--platforms 4090 ascend]

Outputs per language: PDF and SVG with live text (``pdf.fonttype`` 42,
``svg.fonttype`` none) so the figure can be adjusted in Illustrator or Inkscape,
PNG at 400 dpi, and a caption ``.txt``.  Fonts follow ``scripts/exp18/figures/style``
(Nimbus Sans; zh: a CJK fallback), so the figure matches EXP-18 and EXP-19.

Policy, as the other paper figures: nothing here is drawn from poses or odometry,
and the caption states the subset bias rather than hiding it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.exp18.figures import style  # noqa: E402

# --- what is swept -----------------------------------------------------------
# (parameter, default value, the three swept values and their point keys)
SWEEPS = (
    ("K", 8, ((5, "k5"), (6, "k6"), (7, "k7"))),
    ("S", 32, ((1, "s1"), (8, "s8"), (64, "s64"))),
    ("M", 10, ((2, "m2"), (5, "m5"), (20, "m20"))),
)
# One metric per row.  `scale` turns the stored fraction into the reported unit.
METRICS = (
    ("success", "SR", 100.0, "%"),
    ("spl", "SPL", 100.0, "%"),
    ("ne", "NE", 1.0, "m"),
)
# Two series, in fixed order, from the house style.  Validated as a categorical pair
# on the figure surface (CVD dE 24.7 protan / 32.7 tritan, normal 33.6, contrast >=3:1).
PLATFORMS = {
    "4090": {"label": {"en": "RTX 4090 (pre-registered)", "zh": "RTX 4090（预注册平台）"},
             "color": style.GT_COLOR, "ink": style.GT_INK, "marker": "o", "dx": -0.085},
    "ascend": {"label": {"en": "Ascend 910B (replication)", "zh": "昇腾 910B（跨平台复现）"},
               "color": style.PRED_COLOR, "ink": style.PRED_INK, "marker": "s", "dx": +0.085},
}

WIDTH = style.WIDTH_DOUBLE
ASPECT = 0.58  # height / width
LW = 1.4       # thin marks
MS = 4.2
CAPSIZE = 2.2

L = {
    "en": {
        "xlabel": {"K": "history frames K", "S": "sampled trajectories S", "M": "denoising steps M"},
        "default": "deployment value",
        "sensitive": "sensitive",
        "noise": "within noise",
        "ylab": {"SR": "SR (%)", "SPL": "SPL (%)", "NE": "NE (m)"},
    },
    "zh": {
        "xlabel": {"K": "历史帧数 K", "S": "采样条数 S", "M": "去噪步数 M"},
        "default": "部署值",
        "sensitive": "敏感",
        "noise": "在噪声内",
        "ylab": {"SR": "SR (%)", "SPL": "SPL (%)", "NE": "NE (m)"},
    },
}


def setup(lang: str) -> None:
    """EXP-18's rcParams, which register the per-language fonts and already pin
    ``pdf.fonttype`` 42, then the three things they do not cover: live SVG text, a
    figure face that matches the panels, and an exact canvas.

    ``savefig.bbox`` must go back to None: EXP-18 sets "tight", which trims the
    canvas and would silently change the 7.0 in column width this figure is laid out
    for."""
    style.apply(lang)
    matplotlib.rcParams.update({
        "svg.fonttype": "none",
        "figure.facecolor": style.SURFACE,
        "savefig.facecolor": style.SURFACE,
        "savefig.bbox": None,
        "axes.titlesize": 7.6,
        "legend.fontsize": 6.6,
    })


def verdict(ci: Sequence[float]) -> bool:
    """Pre-registered rule: the SR CI of the paired difference excluding 0 is
    "sensitive"; otherwise "within noise"."""
    return ci[0] > 0 or ci[1] < 0


def draw(res: dict, platforms: List[str], lang: str) -> plt.Figure:
    t = L[lang]
    fig, axes = plt.subplots(
        len(METRICS), len(SWEEPS), figsize=(WIDTH, WIDTH * ASPECT),
        sharex="col", gridspec_kw=dict(hspace=0.30, wspace=0.26,
                                       left=0.075, right=0.995, top=0.90, bottom=0.115))
    for r, (mkey, mname, scale, unit) in enumerate(METRICS):
        for c, (pname, dflt, pts) in enumerate(SWEEPS):
            ax = axes[r][c]
            ax.grid(True, axis="y", color=style.GRID, linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            xs = list(range(len(pts)))
            for plat in platforms:
                p = PLATFORMS[plat]
                data = res["points"][plat]
                base = res["defaults"][plat][mkey] * scale
                ys, los, his = [], [], []
                for _, key in pts:
                    d = data[key]
                    ys.append(d[mkey]["treatment"] * scale)
                    # the CI is of the difference; centre it on this point's value
                    los.append(abs(d[mkey]["diff"] - d[mkey]["ci95"][0]) * scale)
                    his.append(abs(d[mkey]["ci95"][1] - d[mkey]["diff"]) * scale)
                xo = [x + p["dx"] for x in xs]
                ax.axhline(base, color=p["color"], linewidth=0.7, linestyle=(0, (3, 2)),
                           alpha=0.55, zorder=1)
                ax.errorbar(xo, ys, yerr=[los, his], color=p["color"], ecolor=p["color"],
                            elinewidth=0.8, capsize=CAPSIZE, capthick=0.8,
                            marker=p["marker"], markersize=MS, markeredgecolor=style.SURFACE,
                            markeredgewidth=0.6, linewidth=LW, zorder=3,
                            label=p["label"][lang] if (r == 0 and c == 0) else None)
                # Secondary encoding of the verdict, so it is never colour-alone:
                # an open ring on the SR row marks a point whose CI excludes 0.
                if mkey == "success":
                    for x, (_, key) in zip(xo, pts):
                        if verdict(data[key]["success"]["ci95"]):
                            ax.plot([x], [data[key]["success"]["treatment"] * scale],
                                    marker="o", markersize=MS + 4.0, markerfacecolor="none",
                                    markeredgecolor=p["ink"], markeredgewidth=0.9, zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v, _ in pts])
            ax.set_xlim(-0.45, len(pts) - 0.55)
            if r == len(METRICS) - 1:
                ax.set_xlabel(t["xlabel"][pname])
            if c == 0:
                ax.set_ylabel(t["ylab"][mname])
            if r == 0:
                ax.set_title(f"{pname}  ({t['default']} {dflt})", color=style.INK, pad=4)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(platforms) > 1:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.54, 1.0), handlelength=1.8, columnspacing=1.6)
    return fig


def caption(res: dict, platforms: List[str], lang: str) -> str:
    t = L[lang]
    sub = res["defaults"]["4090"]["success"] * 100
    full = res["full_1839_4090"]["success"] * 100
    bias = sub - full
    lines = []
    sens = {}
    for plat in platforms:
        hit = [f"{p}={v}" for p, dflt, pts in SWEEPS for v, key in pts
               if verdict(res["points"][plat][key]["success"]["ci95"])]
        sens[plat] = hit
    if lang == "en":
        lines.append(
            "Parameter sensitivity of the deployed model. Each panel sweeps one inference-time "
            "parameter (history frames K, sampled trajectories S, denoising steps M) with the other "
            "two held at their deployment values; no retraining. Rows are SR, SPL and NE, one metric "
            "per axis. Error bars are the 95% CI of the paired difference against that platform's own "
            f"default point ({res['points'][platforms[0]]['k5']['n']} episodes, 2000 bootstrap "
            "resamples); the dashed line is that default. An open ring on the SR row marks a point "
            "whose CI excludes zero, the pre-registered test for sensitivity.")
        for plat in platforms:
            name = PLATFORMS[plat]["label"]["en"]
            hit = sens[plat]
            if not hit:
                lines.append(f"On {name}, no point is sensitive: every point's SR CI contains zero.")
            else:
                got = " and ".join([", ".join(hit[:-1]), hit[-1]]) if len(hit) > 1 else hit[0]
                verb = "is" if len(hit) == 1 else "are"
                lines.append(f"On {name}, {got} {verb} sensitive; every other point is within noise.")
        lines.append(
            f"Subset bias, stated as pre-registered: the default point scores {sub:.2f}% SR on these "
            f"500 episodes against {full:.2f}% on the full 1839, i.e. {bias:+.2f} pt, just outside the "
            "declared +/-3 pt band. The subset was not changed. Absolute levels in this figure therefore "
            "sit above the main table; the paired differences are unaffected.")
        lines.append(
            "The two platforms are shown together only to compare the shape of the response. Their "
            "absolute numbers are not comparable and are never pooled.")
    else:
        lines.append(
            "部署模型的参数敏感性。每个面板只扫一个推理期参数（历史帧数 K、采样条数 S、去噪步数 M），"
            "其余两个保持部署值，都不用重训。三行分别是 SR、SPL、NE，每个指标占自己的坐标轴。"
            f"误差条是与**同平台自己的默认点**配对后差值的 95% CI（{res['points'][platforms[0]]['k5']['n']} 集，"
            "2000 次 bootstrap）；虚线是该默认点。SR 行上的空心圈标出 CI 不含 0 的点，即预注册的"
            "「敏感」判据。")
        for plat in platforms:
            name = PLATFORMS[plat]["label"]["zh"]
            hit = sens[plat]
            if not hit:
                lines.append(f"在 {name} 上，没有任何点判为敏感：每个点 SR 的 CI 都含 0。")
            else:
                lines.append(f"在 {name} 上，{'、'.join(hit)} 判为敏感，其余各点都在噪声内。")
        lines.append(
            f"按预注册如实写明子集偏差：默认点在这 500 集上 SR 为 {sub:.2f}%，全量 1839 集为 {full:.2f}%，"
            f"相差 {bias:+.2f} pt，刚刚超出事先声明的 ±3 pt。**子集未作改动。**"
            "所以本图的绝对水平高于主表，但配对差值不受影响。")
        lines.append("两个平台放在一起只为比较响应的形状；它们的绝对数不可比，也从不合并。")
    return "\n\n".join(lines)


def save(fig: plt.Figure, out_dir: Path, lang: str, cap: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"exp21_sensitivity_{lang}"
    files = [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(files[0])
    fig.savefig(files[1])
    fig.savefig(files[2], dpi=400)
    txt = stem.with_suffix(".txt")
    txt.write_text(cap, encoding="utf-8")
    files.append(txt)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("docs/experiments/exp21-sensitivity-results.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=["en", "zh"])
    ap.add_argument("--platforms", nargs="+", default=["4090", "ascend"],
                    choices=["4090", "ascend"])
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    for plat in args.platforms:
        missing = [k for _, _, pts in SWEEPS for _, k in pts if k not in res["points"].get(plat, {})]
        if missing:
            raise SystemExit(f"{plat} is missing points: {missing}")
    for lang in args.lang:
        setup(lang)
        fig = draw(res, args.platforms, lang)
        cap = caption(res, args.platforms, lang)
        for f in save(fig, args.out_dir, lang, cap):
            print(f"wrote {f}")
        plt.close(fig)


if __name__ == "__main__":
    main()
