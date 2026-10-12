#!/usr/bin/env python3
"""EXP-21 parameter sensitivity, IEEE double column (7.0 in wide): SR / SPL / NE
against K, S and M.

Small multiples, 3 rows x 3 columns.  One metric per panel and therefore one y axis
per panel: SR, SPL and NE do not share a scale, so they are faceted rather than
twinned.  One series per panel, so no legend -- the row label names the metric and
the column title names the parameter, which already names the deployment value.

Each ROW shares one y axis.  That is the whole point of the small multiples here:
left to its own limits every panel fills its height and a 1.6 point spread in S
reads as steep as the 6.2 point fall in K.  Sharing the row makes the three
parameters comparable at a glance, which is the question the figure exists to
answer.

The numbers come from a single platform, the same one the ablation and main tables
use, so every panel is internally consistent and the figure carries no platform
annotation.  The sweep was also repeated on a second machine; that run stays in the
results JSON and in the ledger but is not drawn here.  Absolute numbers from two
machines are not comparable, and taking a per-point minimum across them produces a
curve no single run ever produced: with the default minimised as well, six of the
nine points land above it, which reads as a finding and is an artefact of the
splice.

The deployment value of each parameter (K=8, S=32, M=10) is drawn as a point of the
curve, in its place in the ordering, and marked with a ring and a bold tick label.

Usage (repo root on PYTHONPATH)::

    python -m scripts.exp21.figures.fig_sensitivity \\
        --results docs/experiments/exp21-sensitivity-results.json \\
        --out-dir <dir> [--lang en zh] [--platform 4090]

Outputs per language: PDF and SVG with live text (``pdf.fonttype`` 42,
``svg.fonttype`` none) so the figure can be adjusted in Illustrator or Inkscape,
PNG at 400 dpi, and a caption ``.txt``.  Fonts follow ``scripts/exp18/figures/style``
so the figure matches EXP-18 and EXP-19.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.exp18.figures import style  # noqa: E402

# Each sweep is (name, deployment value, points in VALUE order).  The deployment
# value is one of the points, with key None: it is the setting the model ships with,
# so the curve has to pass through it rather than refer to it from the side -- a
# reader needs to see where 8 sits between 7 and nothing, and that S=64 is above
# S=32 rather than merely near a line.
SWEEPS = (
    ("K", 8, ((5, "k5"), (6, "k6"), (7, "k7"), (8, None))),
    ("S", 32, ((1, "s1"), (8, "s8"), (32, None), (64, "s64"))),
    ("M", 10, ((2, "m2"), (5, "m5"), (10, None), (20, "m20"))),
)
METRICS = (
    ("success", "SR", 100.0),
    ("spl", "SPL", 100.0),
    ("ne", "NE", 1.0),
)

WIDTH = style.WIDTH_DOUBLE
ASPECT = 0.56
LINE = style.GT_COLOR
LW = 1.6
MS = 4.6

L = {
    "en": {
        "xlabel": {"K": "history frames K", "S": "sampled trajectories S", "M": "denoising steps M"},
        "deployed": "deployed",
        "ylab": {"SR": "SR (%)", "SPL": "SPL (%)", "NE": "NE (m)"},
    },
    "zh": {
        "xlabel": {"K": "历史帧数 K", "S": "采样条数 S", "M": "去噪步数 M"},
        "deployed": "部署值",
        "ylab": {"SR": "SR (%)", "SPL": "SPL (%)", "NE": "NE (m)"},
    },
}


def setup(lang: str) -> None:
    """EXP-18's rcParams (fonts per language, ``pdf.fonttype`` 42), then live SVG
    text, a matching figure face, and an exact canvas -- EXP-18 sets
    ``savefig.bbox`` to "tight", which would trim the 7.0 in column width."""
    style.apply(lang)
    matplotlib.rcParams.update({
        "svg.fonttype": "none",
        "figure.facecolor": style.SURFACE,
        "savefig.facecolor": style.SURFACE,
        "savefig.bbox": None,
        "axes.titlesize": 7.8,
    })


def draw(res: dict, plat: str, lang: str) -> plt.Figure:
    t = L[lang]
    data = res["points"][plat]
    fig, axes = plt.subplots(
        len(METRICS), len(SWEEPS), figsize=(WIDTH, WIDTH * ASPECT),
        sharex="col", sharey="row", gridspec_kw=dict(hspace=0.26, wspace=0.12,
                                       left=0.072, right=0.995, top=0.93, bottom=0.115))
    for r, (mkey, mname, scale) in enumerate(METRICS):
        for c, (pname, dflt, pts) in enumerate(SWEEPS):
            ax = axes[r][c]
            ax.grid(True, axis="y", color=style.GRID, linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            base = res["defaults"][plat][mkey] * scale
            ys = [(base if k is None else data[k][mkey]["treatment"] * scale) for _, k in pts]
            xs = list(range(len(pts)))
            ax.plot(xs, ys, color=LINE, linewidth=LW, marker="o", markersize=MS,
                    markeredgecolor=style.SURFACE, markeredgewidth=0.7, zorder=3)
            # The deployment value is marked, not coloured differently: a ring over
            # its marker plus a bold tick label, so it is identifiable in print and
            # for a colour-blind reader without a legend.
            di = next(i for i, (_, k) in enumerate(pts) if k is None)
            ax.plot([xs[di]], [ys[di]], marker="o", markersize=MS + 4.2,
                    markerfacecolor="none", markeredgecolor=LINE, markeredgewidth=1.1,
                    zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v, _ in pts])
            for i, lab in enumerate(ax.get_xticklabels()):
                if i == di:
                    lab.set_fontweight("bold")
                    lab.set_color(style.INK)
            # a little slack at both ends so the deployment ring is not clipped by a spine
            ax.set_xlim(-0.5, len(pts) - 0.5)
            if r == len(METRICS) - 1:
                ax.set_xlabel(t["xlabel"][pname])
            if c == 0:
                ax.set_ylabel(t["ylab"][mname])
            if r == 0:
                ax.set_title(f"{pname}  ({t['deployed']} {dflt})", color=style.INK, pad=4)
    return fig


def caption(res: dict, plat: str, lang: str) -> str:
    d = res["defaults"][plat]
    pts = res["points"][plat]

    def sr(k: str) -> float:
        return 100 * pts[k]["success"]["treatment"]

    if lang == "en":
        return (
            "Parameter sensitivity of the deployed model. Each panel sweeps one inference-time "
            "parameter -- history frames K, sampled trajectories S, denoising steps M -- with the "
            "other two held at their deployment values; no retraining. Rows are SR, SPL and NE, one "
            f"metric per axis. All points come from one platform and the same {pts['k5']['n']} "
            "episodes. The ringed point with the bold tick label is the deployment value "
            f"(SR {100 * d['success']:.1f}%). Shortening the history costs the most: SR falls "
            f"to {sr('k5'):.1f}% at K=5 and {sr('k6'):.1f}% at K=6. Cutting the sampled trajectories "
            f"to one ({sr('s1'):.1f}%) or the denoising steps to two ({sr('m2'):.1f}%) costs little, "
            "so compute is best saved on S and M rather than on K.")
    return (
        "部署模型的参数敏感性。每个面板只扫一个推理期参数——历史帧数 K、采样条数 S、去噪步数 M，"
        "其余两个保持部署值，都不用重训。三行分别是 SR、SPL、NE，每个指标占自己的坐标轴。"
        f"所有点出自同一平台、同一组 {pts['k5']['n']} 集；带圈、刻度加粗的那个点是部署值"
        f"（SR {100 * d['success']:.1f}%）。缩短历史的代价最大：K=5 降到 {sr('k5'):.1f}%、"
        f"K=6 降到 {sr('k6'):.1f}%。把采样条数砍到 1（{sr('s1'):.1f}%）或去噪步数砍到 2"
        f"（{sr('m2'):.1f}%）代价都很小，所以要省算力应该省 S 和 M，不要省 K。")


def save(fig: plt.Figure, out_dir: Path, lang: str, cap: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"exp21_sensitivity_{lang}"
    files = [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(files[0])
    fig.savefig(files[1])
    fig.savefig(files[2], dpi=400)
    txt = stem.with_suffix(".txt")
    txt.write_text(cap, encoding="utf-8")
    return files + [txt]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("docs/experiments/exp21-sensitivity-results.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=["en", "zh"])
    ap.add_argument("--platform", default="4090",
                    help="which run to draw; the figure itself carries no platform label")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    # key None is the deployment value, which comes from res["defaults"], not points
    missing = [k for _, _, pts in SWEEPS for _, k in pts
               if k is not None and k not in res["points"].get(args.platform, {})]
    if args.platform not in res.get("defaults", {}):
        raise SystemExit(f"{args.platform} has no default point")
    if missing:
        raise SystemExit(f"{args.platform} is missing points: {missing}")
    for lang in args.lang:
        setup(lang)
        fig = draw(res, args.platform, lang)
        for f in save(fig, args.out_dir, lang, caption(res, args.platform, lang)):
            print(f"wrote {f}")
        plt.close(fig)


if __name__ == "__main__":
    main()
