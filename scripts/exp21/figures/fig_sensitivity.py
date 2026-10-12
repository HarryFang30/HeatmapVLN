#!/usr/bin/env python3
"""EXP-21 parameter sensitivity, IEEE double column (7.0 in wide).

One column per parameter (history frames K, sampled trajectories S, denoising
steps M), two rows: SR and SPL together on top -- both percentages, so one axis
holds both without a second scale -- and NE below, in metres.  Every point carries
its value, so the figure reads without the axis and draws no grid; the deployment
value of each parameter sits in a shaded column with a bold tick label.  The one key
sits in a strip above the panels so it can never cover data.  The parameter values
are evenly spaced on x, not to scale (S runs 1, 8, 32, 64), and the caption says so.

Each ROW shares one y axis.  That is the whole point of the small multiples here:
left to its own limits every panel fills its height and a 1.8 point spread in S
reads as steep as the 6.2 point fall in K.  Sharing the row makes the three
parameters comparable at a glance, which is the question the figure exists to
answer.

The figure draws no error band.  Two were tried.  A per-point +-1 standard error
has the same width everywhere -- every point runs the same 500 episodes at a
success rate of 60-66%, so sqrt(p(1-p)/500) is 2.1-2.2 points at every point -- and
says nothing about the parameters.  The 95% paired interval of the change from the
deployment value does vary, but it made the figure hard to read.  The paired
results stay in the results JSON and the ledger; the report states which settings
they separate from the deployment value.

The numbers come from the 4090, the platform the ablation table uses, so
every panel is internally consistent and the figure carries no platform annotation.
The sweep was also repeated on a second machine; that run stays in the results JSON
and in the ledger but is not drawn here, and the caption's prose is written for the
4090 numbers.  Absolute numbers from two machines are not comparable, and taking a
per-point minimum across them produces a curve no single run ever produced.

Usage (repo root on PYTHONPATH)::

    python -m scripts.exp21.figures.fig_sensitivity \\
        --results docs/experiments/exp21-sensitivity-results.json --out-dir <dir>

Outputs: PDF and SVG with live text (``pdf.fonttype`` 42, ``svg.fonttype`` none) so
the figure can be adjusted in Illustrator or Inkscape, PNG at 400 dpi, and a caption
``.txt``.  Fonts come from the EXP-18 setup (``scripts/exp18/figures/common_draw``)
so the figure matches EXP-18 and EXP-19: Nimbus Sans from
``/usr/share/fonts/opentype/urw-base35`` or ``$EXP18_FONT_DIR``, DejaVu Sans when
neither has it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from scripts.exp18.figures import common_draw, style  # noqa: E402

# Each sweep is (name, axis label, points in VALUE order).  The deployment value is
# one of the points, with key None: it is the setting the model ships with, so the
# curve has to pass through it rather than refer to it from the side.
SWEEPS = (
    ("K", "History frames K", ((5, "k5"), (6, "k6"), (7, "k7"), (8, None))),
    ("S", "Sampled trajectories S", ((1, "s1"), (8, "s8"), (32, None), (64, "s64"))),
    ("M", "Denoising steps M", ((2, "m2"), (5, "m5"), (10, None), (20, "m20"))),
)

# (metric key, legend label, scale, colour, marker, value label above or below the point)
# Colours: the house blue for the headline metric, then green and violet from the
# house categorical hues -- not the orange, which the EXP-18 figures reserve for
# predictions.  Checked with the dataviz palette validator on the figure surface:
# all pairs pass.  The pair that shares a panel, SR blue / SPL green, is at dE 26.5
# (protan) but 7.6 under tritanopia, inside the 6-8 band the validator allows only with
# a second cue -- here the marker shape, circle against square.
SR = ("success", "SR", 100.0, style.GT_COLOR, "o", "above")
SPL = ("spl", "SPL", 100.0, style.TIER_COLORS["D"], "s", "below")
NE = ("ne", "NE", 1.0, style.TIER_COLORS["E"], "D", "above")
ROWS = (
    ((SR, SPL), "SR, SPL (%)", "{:.1f}"),
    ((NE,), "NE (m), lower is better", "{:.2f}"),
)

PLATFORM = "4090"
WIDTH = style.WIDTH_DOUBLE
HEIGHT = 3.5
LEFT, RIGHT, TOP, BOTTOM = 0.075, 0.995, 0.905, 0.115
LW = 1.5
MS = 4.2
VALUE_FS = 5.6
VALUE_GAP = 3.2  # points between a marker's edge and its value label
CLEAR = 1.0  # points a value label keeps from lines, markers, other labels and the frame
# Room a value label needs past its point, in points: marker half, gap, the label (its
# height is ~1.16x the font size in DejaVu Sans, the fallback, and less in Nimbus Sans,
# so 1.3x covers both), the clearance, and half a point of slack.
LABEL_ROOM = MS / 2 + VALUE_GAP + 1.3 * VALUE_FS + CLEAR + 0.5
MARKER_R = 0.71 * MS + 0.3  # the diamond's half-diagonal plus its edge; covers 'o' and 's'
DPI = 400  # labels are measured at the PNG's resolution
DEPLOYED_FILL = "#ecebe5"  # a step between the surface and the grid: visible, never louder than data
DEPLOYED_HALF = 0.38  # half-width of the deployed column, in x units (points are 1 apart)


def setup() -> None:
    """EXP-18's figure setup, then live SVG text, a matching figure face, and an
    exact canvas -- EXP-18's style sets ``savefig.bbox`` to "tight", which would
    trim the 7.0 in column width.  ``common_draw.setup`` rather than
    ``style.apply`` because it re-registers the CFF Nimbus Sans as TrueType, so the
    PDF's Type 42 fonts pass preflight."""
    common_draw.setup("en")
    matplotlib.rcParams.update({
        "figure.dpi": DPI,
        "svg.fonttype": "none",
        "figure.facecolor": style.SURFACE,
        "savefig.facecolor": style.SURFACE,
        "savefig.bbox": None,
    })


def value(res: dict, key, metric: str) -> float:
    if key is None:
        return res["defaults"][PLATFORM][metric]
    return res["points"][PLATFORM][key][metric]["treatment"]


def touches(ax, box, placed) -> bool:
    """Whether a label's rendered box (display units) comes within ``CLEAR`` of any
    line or marker in ``ax``, any label already placed, or the axes frame."""
    px = ax.figure.dpi / 72
    frame = ax.get_window_extent()
    clear = CLEAR * px
    if (box.x0 < frame.x0 + clear or box.x1 > frame.x1 - clear
            or box.y0 < frame.y0 + clear or box.y1 > frame.y1 - clear):
        return True
    if any(box.x0 < o.x1 + clear and o.x0 < box.x1 + clear
           and box.y0 < o.y1 + clear and o.y0 < box.y1 + clear for o in placed):
        return True
    pad = (CLEAR + LW / 2) * px
    x0, x1, y0, y1 = box.x0 - pad, box.x1 + pad, box.y0 - pad, box.y1 + pad
    for line in ax.get_lines():
        pts = ax.transData.transform(line.get_xydata())
        r = MARKER_R * px
        if any(x0 - r <= mx <= x1 + r and y0 - r <= my <= y1 + r for mx, my in pts):
            return True
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            lo, hi = max(x0, min(xa, xb)), min(x1, max(xa, xb))
            if lo > hi:
                continue  # the segment does not reach the box's columns
            at = [ya + (yb - ya) * (x - xa) / (xb - xa) for x in (lo, hi)]
            if min(at) <= y1 and max(at) >= y0:  # steep or shallow, it crosses the box
                return True
    return False


def annotate(ax, x, y, text, side, lean, bold):
    """A value label ``side`` of its point, straight over or under it (``lean`` 0) or
    off a corner of the marker (``lean`` +1 right, -1 left)."""
    up = 1 if side == "above" else -1
    off = MS / 2 + VALUE_GAP  # off a corner too, so the label clears its own marker
    if lean:
        xy, ha = (lean * off, up * off), ("left" if lean > 0 else "right")
    else:
        xy, ha = (0, up * off), "center"
    return ax.annotate(text, (x, y), xytext=xy, textcoords="offset points",
                       ha=ha, va="bottom" if up > 0 else "top",
                       fontsize=VALUE_FS, zorder=4, color=style.INK if bold else style.INK_2,
                       fontweight="bold" if bold else "normal")


def put_label(ax, x, y, text, side, bold, placed, renderer, where):
    """Draw a value label on ``side`` of its point; when that would touch data,
    another label or the frame, the other side (NE at K=7 sits in a V between 4.07
    and 3.77, and the line runs through a label centred above it); when both do --
    a point inside a steep run, with the line just above it on one hand and just
    below on the other -- off a corner of the marker, where the run is not.  Measured
    on the rendered text after the axes limits are final, not estimated.  If nothing
    fits, the label stays on ``side`` and a warning names it, so the figure is still
    written and the one label can be checked by eye."""
    other = "below" if side == "above" else "above"
    for s, lean in ((side, 0), (other, 0), (side, 1), (side, -1), (other, 1), (other, -1)):
        ann = annotate(ax, x, y, text, s, lean, bold)
        box = ann.get_window_extent(renderer)
        if not touches(ax, box, placed):
            placed.append(box)
            return
        ann.remove()
    placed.append(annotate(ax, x, y, text, side, 0, bold).get_window_extent(renderer))
    print(f"WARNING: value label {text} ({where}) is closer than {CLEAR} pt to data, "
          "another label or the frame wherever it goes; check it in the figure")


def draw(res: dict) -> plt.Figure:
    labels = []  # value labels go on after the limits are final: their side depends on them
    fig, axes = plt.subplots(
        len(ROWS), len(SWEEPS), figsize=(WIDTH, HEIGHT), sharex="col", sharey="row",
        gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.10, wspace=0.10,
                         left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM))
    for r, (series, ylabel, fmt) in enumerate(ROWS):
        for c, (pname, xlabel, pts) in enumerate(SWEEPS):
            ax = axes[r][c]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            xs = list(range(len(pts)))
            di = next(i for i, (_, k) in enumerate(pts) if k is None)
            ax.axvspan(di - DEPLOYED_HALF, di + DEPLOYED_HALF, color=DEPLOYED_FILL,
                       linewidth=0, zorder=0)
            for mkey, mname, scale, color, marker, side in series:
                ys = [value(res, k, mkey) * scale for _, k in pts]
                ax.plot(xs, ys, color=color, linewidth=LW, marker=marker, markersize=MS,
                        markeredgecolor=style.SURFACE, markeredgewidth=0.6, zorder=3)
                where = [f"{mname}, {pname}={v}" for v, _ in pts]
                labels.append((ax, xs, ys, di, side, fmt, where))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v, _ in pts])
            # only the bottom row has tick labels (shared x), so this is empty above it
            for i, lab in enumerate(ax.get_xticklabels()):
                if i == di:
                    lab.set_fontweight("bold")
                    lab.set_color(style.INK)
            ax.set_xlim(-0.5, len(pts) - 0.5)
            if r == len(ROWS) - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)
        # Room above the highest point and below the lowest for a value label, sized in
        # points: with the pad p in data units, p / (span + 2p) of the axes height must
        # equal LABEL_ROOM, so p = LABEL_ROOM * span / (height - 2 * LABEL_ROOM).
        lo, hi = axes[r][0].get_ylim()
        height = axes[r][0].get_window_extent().height * 72 / fig.dpi
        pad = LABEL_ROOM * (hi - lo) / (height - 2 * LABEL_ROOM)
        axes[r][0].set_ylim(lo - pad, hi + pad)

    # Values in ink, not in the series colour; which line a value belongs to is told
    # by its side (SR above, SPL below, unless that side would touch something) and
    # its distance.
    renderer = fig.canvas.get_renderer()
    placed = {}
    for ax, xs, ys, di, side, fmt, where in labels:
        for i, (x, y) in enumerate(zip(xs, ys)):
            put_label(ax, x, y, fmt.format(y), side, i == di, placed.setdefault(ax, []),
                      renderer, where[i])

    handles = [Line2D([], [], color=color, linewidth=LW, marker=marker, markersize=MS,
                      markeredgecolor=style.SURFACE, markeredgewidth=0.6)
               for _, _, _, color, marker, _ in (SR, SPL, NE)]
    handles.append(Patch(facecolor=DEPLOYED_FILL, linewidth=0))
    labels = [m[1] for m in (SR, SPL, NE)] + ["Deployed setting"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=((LEFT + RIGHT) / 2, 1.0), handlelength=2.2,
               handletextpad=0.6, columnspacing=2.4, borderaxespad=0.3)
    return fig


def caption(res: dict) -> str:
    pts = res["points"][PLATFORM]

    def sr(k) -> float:
        return 100 * value(res, k, "success")

    def ne(k) -> float:
        return value(res, k, "ne")

    # settings whose SR change from the deployment value is outside its paired interval
    names = {k: f"{p}={v}" for p, _, sweep in SWEEPS for v, k in sweep if k is not None}
    apart = [names[k] for _, _, sweep in SWEEPS for _, k in sweep if k is not None
             and (pts[k]["success"]["ci95"][0] > 0 or pts[k]["success"]["ci95"][1] < 0)]
    n, full = pts["k5"]["n"], res["full_1839_4090"]
    return (
        "Parameter sensitivity of the deployed model. Each column varies one inference-time "
        "parameter -- history frames K, sampled trajectories S, denoising steps M -- with the "
        "other two at their deployment values (shaded column, bold tick); no retraining. "
        "Parameter values are evenly spaced, not to scale. Top: SR and SPL (%); bottom: "
        "navigation error NE (m), lower is better. Every point is measured on the same "
        f"{n} R2R-CE val-unseen episodes and labelled with its value. Shortening the history "
        f"costs the most: SR falls from {sr(None):.1f}% to {sr('k6'):.1f}% at K=6 and "
        f"{sr('k5'):.1f}% at K=5, and NE rises from {ne(None):.2f} m to {ne('k5'):.2f} m. "
        f"Cutting the sampled trajectories to one ({sr('s1'):.1f}%) or the denoising steps to "
        f"two ({sr('m2'):.1f}%) costs less, and only {' and '.join(apart)} differ from the "
        f"deployment value beyond chance (paired bootstrap over the same episodes, 95%), so "
        "compute is better saved on S and M than on K. The episodes are a fixed subset "
        "stratified by start-to-goal geodesic distance; on them the deployed model scores "
        f"{sr(None):.1f}% SR against {100 * full['success']:.1f}% on all {full['n']} "
        "val-unseen episodes, so absolute values sit above the main table; every comparison "
        "between settings is paired on these same episodes.")


def save(fig: plt.Figure, out_dir: Path, cap: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "exp21_sensitivity"
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
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    # key None is the deployment value, which comes from res["defaults"], not points
    missing = [k for _, _, pts in SWEEPS for _, k in pts
               if k is not None and k not in res["points"].get(PLATFORM, {})]
    if PLATFORM not in res.get("defaults", {}):
        raise SystemExit(f"{PLATFORM} has no default point")
    if missing:
        raise SystemExit(f"{PLATFORM} is missing points: {missing}")
    setup()
    fig = draw(res)
    for f in save(fig, args.out_dir, caption(res)):
        print(f"wrote {f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
