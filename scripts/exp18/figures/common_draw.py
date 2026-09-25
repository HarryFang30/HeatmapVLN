"""Drawing primitives shared by the EXP-18 figures (case, gallery, route figures).

Every primitive takes a matplotlib ``Axes`` and draws in its data coordinates;
sizes of glyphs (badges, arrows, markers) are given in points so they print at
the same size whatever the map scale.  Conventions come from
``scripts/exp18/geometry.py`` (bearings) and ``topdown_io.py`` (maps); nothing
is re-derived here.

Visual vocabulary (one meaning per encoding, used by every EXP-18 figure):

* **Blue = ground truth / past positions, and nothing else.**  Numbered
  badges 1..8 are the K = 8 past positions (1 = oldest), filled with
  ``style.history_color`` (light = older); the number, not the shade,
  identifies a slot.  The ground-truth affordance map row is a blue
  sequential ramp (``GT_CMAP``).
* **Orange = prediction, and nothing else.**  ``PRED_CMAP`` (=
  ``style.HEAT_CMAP_OPAQUE``) for the predicted affordance map row; a black x
  with a white halo marks a predicted peak (``peak_marks``).
* **Misses (D1, D2).**  A slot is missed iff it fails joint PCK@8
  (``data.CaseRow.misses``); its number is ink in a small white disc ringed
  in dark orange (``miss_badge``), next to its own x or moved and joined to it
  by a short ink leader (``place_miss_labels`` / ``draw_miss_labels``), with
  an optional dotted connector from its true-bearing tick to the x
  (``draw_miss_connectors``).  Never a blue badge on the orange row.
* **Row names (D3)** "ground truth" / "prediction" sit in a fixed gutter left
  of the strip with a tiny colour key (``gutter_row_label``), in every block.
* **Elevation window (D4)**: ``elevation_window(rows)`` -- +-10 deg, widened
  just enough for every GT-visible and drawn predicted peak of the rows shown,
  at most +-45 deg; ``elevation_text`` words it for the caption.
* **Notes (D5)** are wrapped, never dropped: ``wrap_notes`` lays them out on
  as many lines as needed, ``draw_note`` draws one (blue badge for a slot no
  view shows, orange-ringed badge for a miss the model calls not visible).
* **Frame labels (D6)**: ``frame_label`` -> "frame 35 of 79" /
  "第 35 帧（共 79 帧）" (1-based count).
* **Grey = context.**  Maps and the three current views the model is not
  given are desaturated and lightened; only the front view keeps its colour.
* **Black = the robot** (arrowhead pointing where it faces) and key positions
  (``K1`` badges with a facing arrow).
* Bold Chinese text is drawn with a thin stroke of its own colour
  (``bold_effects``; the CJK font has no bold face).

The surround strip is the heading-centred ring of ``geo.stitch_ring`` rolled so
that it starts at the left edge of the front view (bearing +45 deg) and runs
clockwise seen from above: panels Front | Right | Back | Left, 90 deg each,
strip coordinate ``s = (45 - bearing) mod 360`` (``strip_x``).

``setup()`` installs the repo style and re-registers the CFF (OpenType) Nimbus
Sans faces as TrueType, so that Type 42 embedding in the PDF is a real
TrueType font (``pdffonts`` otherwise reports a font-type mismatch that
publisher preflight can reject).
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18 import geometry as geo
from scripts.exp18.figures import style

import matplotlib  # noqa: E402  (style.apply selects the Agg backend first)
from matplotlib import font_manager  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
GT_CMAP = LinearSegmentedColormap.from_list(
    "exp18_gt_heat", ["#fcfcfb", "#d9e6f7", "#a5c6ef", "#5598e7", "#1c5cab", "#0d2f5c"])
PRED_CMAP = style.HEAT_CMAP_OPAQUE
HEAT_TOP = 0.74  # both heat rows use the ramps up to here: a saturated mid-tone peak, never a dark mud
PEAK_COLOR = style.INK
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
HALO_THIN = [pe.withStroke(linewidth=1.1, foreground="white")]
BADGE_FS = 5.6
STRIP_START_DEG = 45.0  # the strip starts at the front view's left edge
EL_DEFAULT = 10.0  # D4: affordance map rows show +-10 deg of elevation ...
EL_MARGIN = 3.0  # ... widened to include every peak with this much room for its mark ...
EL_MAX = 45.0  # ... up to +-45 deg (a view's vertical field of view)
GT_INK = style.GT_INK
PRED_INK = style.PRED_INK


def setup(lang: str = "en") -> None:
    """``style.apply(lang)`` plus TrueType re-registration of CFF fonts (see module doc).

    For ``lang='zh'`` without a CJK font (``style.CJK_CANDIDATES`` /
    ``$EXP18_CJK_FONT``) the figure still renders, with missing-glyph boxes;
    one notice replaces matplotlib's per-glyph warnings.
    """
    style.apply(lang)
    use_truetype_fonts()
    matplotlib.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0.0, "pdf.compression": 9})
    # name the families explicitly: matplotlib falls back glyph by glyph only across an explicit
    # family list, not through the generic "sans-serif" alias (needed for Latin + CJK labels)
    matplotlib.rcParams["font.family"] = list(matplotlib.rcParams["font.sans-serif"])
    if lang == "zh":
        families = [f for f in matplotlib.rcParams["font.sans-serif"] if f not in (style.LATIN_FAMILY, "DejaVu Sans")]
        if not families:
            import warnings
            print("[exp18 figures] no CJK font found (set EXP18_CJK_FONT); Chinese labels will show as boxes")
            warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font", category=UserWarning)


def _otf_to_ttf(src: Path, dst: Path) -> None:
    """Convert a CFF-flavoured OpenType font to TrueType outlines (fontTools cu2qu, max error 1 unit)."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(str(src))
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    glyf = font["glyf"] = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():
        if hasattr(glyph, "xMin"):
            hmtx[name] = (hmtx[name][0], glyph.xMin)
    maxp = font["maxp"] = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for attr in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                 "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp%d" % os.getpid())
    font.save(str(tmp))
    os.replace(tmp, dst)


def use_truetype_fonts(cache_dir=None) -> List[Path]:
    """Replace registered CFF ``.otf`` faces of the paper font by TrueType copies.

    Copies live in ``cache_dir`` (default ``$EXP18_FONT_CACHE`` or the temp
    dir) and are rebuilt when the source is newer.  Without fontTools the CFF
    faces stay (the figure still renders; only the PDF font type differs).
    """
    fm = font_manager.fontManager
    cff = [e for e in fm.ttflist if e.fname.lower().endswith(".otf") and e.name == style.LATIN_FAMILY]
    if not cff:
        return []
    cache = Path(cache_dir or os.environ.get("EXP18_FONT_CACHE") or Path(tempfile.gettempdir()) / "exp18_ttf_fonts")
    converted = []
    try:
        for entry in cff:
            src = Path(entry.fname)
            dst = cache / (src.stem + ".ttf")
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                _otf_to_ttf(src, dst)
            converted.append(dst)
    except Exception as exc:  # pragma: no cover - fontTools missing or font unreadable
        print(f"[exp18 figures] keeping CFF fonts ({exc})")
        return []
    fm.ttflist = [e for e in fm.ttflist if e not in cff]
    for path in converted:
        fm.addfont(str(path))
    fm._findfont_cached.cache_clear()
    font_manager._get_font.cache_clear()
    return converted


MAP_PLATE = (0.953, 0.949, 0.937)  # tone for "no map here" (outside the building), so maps read as plates


def mute(rgb: np.ndarray, sat: float = 0.25, white: float = 0.45) -> np.ndarray:
    """Desaturate (keep ``sat`` of the chroma) and lighten (mix ``white``) an RGB image -> float [0, 1]."""
    x = np.asarray(rgb, dtype=np.float32)[..., :3]
    if x.max() > 1.5:
        x = x / 255.0
    lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = sat * x + (1.0 - sat) * lum
    return np.clip(white + (1.0 - white) * x, 0.0, 1.0)


def mute_map(rgb: np.ndarray, sat: float, white: float) -> np.ndarray:
    """``mute`` for top-down renders: the empty (pure white) background becomes ``MAP_PLATE``."""
    raw = np.asarray(rgb)[..., :3]
    empty = (raw.min(axis=-1) >= (250 if raw.dtype == np.uint8 else 0.98))
    out = mute(raw, sat=sat, white=white)
    out[empty] = MAP_PLATE
    return out


def mix(color, other, t: float) -> Tuple[float, float, float]:
    a, b = np.asarray(to_rgb(color)), np.asarray(to_rgb(other))
    return tuple((1 - t) * a + t * b)


def history_line_color(k: int, num: int = 8):
    """Lines (rays, guides) of slot k: the badge colour, darkened a little so light slots stay visible."""
    return mix(style.history_color(k, num), style.INK, 0.18)


def pts_to_data(ax, dx_pt: float, dy_pt: float = 0.0) -> Tuple[float, float]:
    """Data-space extent of a (dx, dy) offset in points (linear axes; call after the layout is final)."""
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((dx_pt * ax.figure.dpi / 72.0, dy_pt * ax.figure.dpi / 72.0))
    return x1 - x0, y1 - y0


def text_width_pt(fig, text: str, fontsize: float, **kw) -> float:
    """Rendered width of ``text`` in points."""
    renderer = fig.canvas.get_renderer()
    t = fig.text(0, 0, text, fontsize=fontsize, **kw)
    w = t.get_window_extent(renderer).width * 72.0 / fig.dpi
    t.remove()
    return w


def clean_axes(ax, spines: bool = False, color: str = style.AXIS, lw: float = 0.6) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(spines)
        s.set_edgecolor(color)
        s.set_linewidth(lw)


# --------------------------------------------------------------------------- #
# Glyphs
# --------------------------------------------------------------------------- #
def badge_width_pt(label: str) -> float:
    """Approximate outer width of a history badge (circle for one digit, pill for ranges)."""
    return 7.4 if len(label) == 1 else 5.4 + 3.05 * len(label)


def history_badge(ax, x, y, label: str, k: int, num: int = 8, transform=None, zorder: float = 6, fs: float = BADGE_FS):
    """Numbered history badge filled with the slot's ramp colour (``k`` = 0-based slot)."""
    box = "circle,pad=0.22" if len(label) == 1 else "round,pad=0.24,rounding_size=0.62"
    return ax.text(x, y, label, ha="center", va="center", fontsize=fs, fontweight="bold",
                   color=style.history_text_color(k, num), transform=transform or ax.transData, zorder=zorder,
                   bbox=dict(boxstyle=box, fc=style.history_color(k, num), ec="white", lw=0.55))


def key_badge(ax, x, y, text: str, transform=None, fs: float = 6.2, zorder: float = 8, **kw):
    """Black rounded badge naming a key position (``K1``)."""
    return ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="white",
                   transform=transform or ax.transData, zorder=zorder,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.3", fc=style.INK, ec="white", lw=0.5), **kw)


def robot_glyph(ax, x: float, y: float, forward_xy, size_pt: float = 7.0, zorder: float = 7,
                color: str = style.INK) -> Polygon:
    """Arrowhead-shaped robot at (x, y) pointing along ``forward_xy`` (data space), ``size_pt`` long."""
    trans = ax.transData
    p = trans.transform((x, y))
    f = np.asarray(forward_xy, dtype=float)
    tip = trans.transform((x + f[0] * 1e-3, y + f[1] * 1e-3))
    d = (tip - p) / np.linalg.norm(tip - p)
    n = np.array([-d[1], d[0]])
    s = size_pt * ax.figure.dpi / 72.0
    pts = np.array([p + d * s * 0.62, p - d * s * 0.38 + n * s * 0.38, p - d * s * 0.16,
                    p - d * s * 0.38 - n * s * 0.38])
    poly = Polygon(trans.inverted().transform(pts), closed=True, fc=color, ec="white", lw=0.6, zorder=zorder)
    ax.add_patch(poly)
    return poly


def heading_arrow(ax, p, forward_xy, length: float, color: str = style.INK, lw: float = 0.9, zorder: float = 6):
    """Arrow of ``length`` data units from ``p`` along ``forward_xy``."""
    p = np.asarray(p, dtype=float)
    f = np.asarray(forward_xy, dtype=float)
    f = f / np.linalg.norm(f)
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + f * length), arrowstyle="-|>", mutation_scale=5.5,
                                 color=color, lw=lw, zorder=zorder, shrinkA=0, shrinkB=0))


def scale_bar(ax, x: float, y: float, length: float, label: str, fs: float = 5.8, ha: str = "left",
              color: str = style.INK, zorder: float = 9):
    """Horizontal bar of ``length`` data units starting at x (``ha='right'``: ending at x), label above.

    Returns ``(line, label)`` (see ``inset_from_frame_pt``)."""
    x0 = x if ha == "left" else x - length
    line, = ax.plot([x0, x0 + length], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=zorder,
                    path_effects=HALO_THIN)
    text = ax.annotate(label, (x0 + length / 2, y), xytext=(0, 1.6), textcoords="offset points", ha="center",
                       va="bottom", fontsize=fs, color=color, path_effects=HALO, zorder=zorder)
    return line, text


def inset_from_frame_pt(ax, artists) -> float:
    """Smallest distance (pt) from the drawn extents of ``artists`` to the frame of ``ax`` (negative: outside)."""
    rend = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(rend)
    d = []
    for a in artists:
        bb = a.get_window_extent(rend)
        d += [bb.x0 - box.x0, box.x1 - bb.x1, bb.y0 - box.y0, box.y1 - bb.y1]
    return float(min(d)) * 72.0 / ax.figure.dpi if d else float("nan")


def nice_length(target: float, choices=(0.25, 0.5, 1, 2, 5, 10, 20, 50)) -> float:
    """Largest 'nice' length not exceeding ``target`` (the smallest choice if none does)."""
    ok = [c for c in choices if c <= target]
    return float(ok[-1] if ok else choices[0])


def peak_mark(ax, x: float, y: float, size: float = 4.4, color: str = PEAK_COLOR, zorder: float = 7, **kw):
    """Predicted-peak x with a white halo."""
    return ax.plot([x], [y], marker="x", ms=size, mew=0.95, color=color, zorder=zorder, clip_on=False,
                   path_effects=[pe.withStroke(linewidth=2.3, foreground="white")], **kw)


# --------------------------------------------------------------------------- #
# Top-down maps
# --------------------------------------------------------------------------- #
def fit_limits(xz: np.ndarray, pad: float, aspect_hw: float) -> Tuple[float, float, float, float]:
    """(x0, x1, z0, z1) covering ``xz`` + ``pad`` whose height/width equals ``aspect_hw`` (map units)."""
    xz = np.asarray(xz, dtype=float).reshape(-1, 2)
    x0, x1 = xz[:, 0].min() - pad, xz[:, 0].max() + pad
    z0, z1 = xz[:, 1].min() - pad, xz[:, 1].max() + pad
    if (z1 - z0) / (x1 - x0) > aspect_hw:
        c, half = (x0 + x1) / 2, (z1 - z0) / aspect_hw / 2
        x0, x1 = c - half, c + half
    else:
        c, half = (z0 + z1) / 2, (x1 - x0) * aspect_hw / 2
        z0, z1 = c - half, c + half
    return x0, x1, z0, z1


def axes_aspect_hw(ax) -> float:
    box = ax.get_position()
    w_in, h_in = ax.figure.get_size_inches()
    return (box.height * h_in) / (box.width * w_in)


def draw_topdown(ax, level, limits=None, sat: float = 0.22, white: float = 0.5) -> None:
    """Muted top-down map in world metres (+x right, +z down), clipped to ``limits`` = (x0, x1, z0, z1)."""
    ax.imshow(mute_map(level.rgb(), sat=sat, white=white), extent=level.extent, interpolation="bilinear", zorder=0)
    if limits is not None:
        x0, x1, z0, z1 = limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(z1, z0)
    ax.set_autoscale_on(False)
    ax.set_facecolor(MAP_PLATE)


def draw_route(ax, xz: np.ndarray, color: str = style.INK_2, lw: float = 1.0, start_label: Optional[str] = "start",
               zorder: float = 3) -> None:
    """Route polyline with an open start circle (and its label, placed away from the route)."""
    xz = np.asarray(xz, dtype=float)
    ax.plot(xz[:, 0], xz[:, 1], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    ax.plot(*xz[0], marker="o", ms=4.0, mfc="white", mec=color, mew=0.9, zorder=zorder + 1)
    if start_label:
        d = xz[min(3, len(xz) - 1)] - xz[0]
        d = -d / (np.linalg.norm(d) + 1e-9)  # label on the side away from where the route goes
        ha = "right" if d[0] < -0.35 else ("left" if d[0] > 0.35 else "center")
        va = "top" if d[1] > 0.35 else ("bottom" if d[1] < -0.35 else "center")  # +z is down on the map
        ax.annotate(start_label, xz[0], xytext=(d[0] * 4.0, -d[1] * 4.0), textcoords="offset points", ha=ha,
                    va=va, fontsize=5.8, color=color, path_effects=HALO, zorder=zorder + 1)


# --------------------------------------------------------------------------- #
# Route legs (designed routes: outbound vs return)
# --------------------------------------------------------------------------- #
LEG_OUT_LS = "-"  # outbound leg: solid
LEG_BACK_LS = (0, (2.4, 1.4))  # return leg: dashed
LEG_COLOR = style.INK_2


def _thin_display(P: np.ndarray, min_px: float) -> np.ndarray:
    """Drop display points closer than ``min_px`` to the previous kept one (the last point is always kept)."""
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) >= min_px:
            keep.append(i)
    if keep[-1] != len(P) - 1:
        if len(keep) > 1:
            keep[-1] = len(P) - 1
        elif np.hypot(*(P[-1] - P[0])) > 0:
            keep.append(len(P) - 1)
    return P[keep]


def offset_polyline(ax, xy, offset_pt: float, min_step_pt: float = 0.6) -> np.ndarray:
    """Polyline (data coords) shifted ``offset_pt`` to the right of its direction of travel as drawn.

    Offsets are taken in display space (so they print at the same size at
    any map scale) with mitred joints (miter length capped at 2x); points
    closer than ``min_step_pt`` are merged first, so turning on the spot adds
    no spikes.  Two legs travelling the same corridor in opposite directions
    therefore separate into two parallel lanes ("keep right").
    """
    T = ax.transData
    px = ax.figure.dpi / 72.0
    P = _thin_display(T.transform(np.asarray(xy, dtype=float).reshape(-1, 2)), min_step_pt * px)
    if len(P) < 2:
        return T.inverted().transform(P)
    seg = np.diff(P, axis=0)
    u = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    nseg = np.stack([u[:, 1], -u[:, 0]], axis=1)  # right-hand normal (display y is up)
    n = np.zeros_like(P)
    n[0], n[-1] = nseg[0], nseg[-1]
    if len(P) > 2:
        m = nseg[:-1] + nseg[1:]
        ln = np.linalg.norm(m, axis=1, keepdims=True)
        m = np.where(ln > 1e-6, m / np.maximum(ln, 1e-12), nseg[1:])
        cos = np.sum(m * nseg[1:], axis=1, keepdims=True)
        n[1:-1] = m / np.maximum(cos, 0.5)
    return T.inverted().transform(P + n * offset_pt * px)


def chevrons(ax, xy, spacing_pt: float = 26.0, size_pt: float = 1.7, color=LEG_COLOR, lw: float = 0.7,
             end_pt: float = 7.0, zorder: float = 3.2, clip_path=None) -> int:
    """Open chevrons (">") along a polyline (data coords) pointing along its direction; returns how many.

    Evenly spaced ``spacing_pt`` apart and centred on the polyline, none within
    ``end_pt`` of either end; one in the middle when the line is shorter.
    """
    T = ax.transData
    px = ax.figure.dpi / 72.0
    P = _thin_display(T.transform(np.asarray(xy, dtype=float).reshape(-1, 2)), 0.3 * px)
    if len(P) < 2:
        return 0
    seg = np.diff(P, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ln)])
    total = s[-1] / px  # points
    if total < 2 * size_pt + 2:
        return 0
    n = int((total - 2 * end_pt) // spacing_pt) + 1 if total > 2 * end_pt else 1
    at = total / 2 + (np.arange(n) - (n - 1) / 2) * spacing_pt
    a, b = size_pt * px, size_pt * 0.95 * px
    for x in at:
        j = int(np.clip(np.searchsorted(s, x * px) - 1, 0, len(seg) - 1))
        d = seg[j] / ln[j]
        p = P[j] + d * (x * px - s[j])
        nn = np.array([-d[1], d[0]])
        pts = T.inverted().transform(np.array([p - a * d + b * nn, p + a * d, p - a * d - b * nn]))
        line, = ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="miter",
                        zorder=zorder)
        if clip_path is not None:
            line.set_clip_path(clip_path)
    return n


def draw_route_legs(ax, xy, split: int, color=LEG_COLOR, lw: float = 1.0, alpha: float = 1.0,
                    offset_pt: float = 1.3, chevron_spacing_pt: Optional[float] = 26.0, zorder: float = 3,
                    clip_path=None) -> dict:
    """Route ``xy`` [T, 2] (data coords) as two legs split at index ``split``.

    Outbound ``xy[:split + 1]`` solid, return ``xy[split:]`` dashed; each leg is
    shifted ``offset_pt`` to its own right-hand side (``offset_polyline``) so a
    return that retraces the outbound path stays visible beside it; chevrons
    every ``chevron_spacing_pt`` (None: no chevrons) give the direction of
    travel.  Call after the axes' limits and position are final.  Returns
    {"out": [n, 2] or None, "back": [n, 2] or None} (the drawn, offset lines).
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    split = int(np.clip(split, 0, len(xy) - 1))
    out = {"out": None, "back": None}
    for key, part, ls in (("out", xy[:split + 1], LEG_OUT_LS), ("back", xy[split:], LEG_BACK_LS)):
        if len(part) < 2 or np.ptp(part, axis=0).max() <= 0:
            continue
        line_xy = offset_polyline(ax, part, offset_pt)
        line, = ax.plot(line_xy[:, 0], line_xy[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
                        solid_capstyle="round", solid_joinstyle="round", dash_capstyle="butt")
        if clip_path is not None:
            line.set_clip_path(clip_path)
        if chevron_spacing_pt:
            chevrons(ax, line_xy, spacing_pt=chevron_spacing_pt, color=color, zorder=zorder + 0.2,
                     clip_path=clip_path)
        out[key] = line_xy
    return out


def turn_marker(ax, x: float, y: float, size: float = 3.6, zorder: float = 6.5):
    """White diamond with an ink edge: where the route turns back."""
    return ax.plot([x], [y], marker="D", ms=size, mfc="white", mec=style.INK, mew=0.8, zorder=zorder)


def place_near(ax, anchors_xy, obstacles_xy, limits, boxes_pt, radii_pt=(2.5, 4.5, 7.0, 10.0, 14.0, 19.0, 25.0),
               n_dir: int = 16, margin_pt: float = 1.0, fixed_boxes=(),
               clear_pt: float = 4.5, order: Optional[Sequence[int]] = None, away_from=None,
               away_weight: float = 0.0) -> List[Tuple[float, float, float, float]]:
    """Greedy label placement beside points: returns (x, y, gap_pt, score) per anchor, in anchor order.

    Label i is a box of ``boxes_pt[i]`` = (w, h) points whose edge lies ``r``
    points from its anchor (``r`` in ``radii_pt``) in one of ``n_dir``
    directions, so a wide label never covers its own anchor.  It
    maximises the clearance (points; ``clear_pt`` counts as clear enough) to
    ``obstacles_xy`` (route points, markers), to every anchor, to the labels
    placed before it and to ``fixed_boxes`` ((cx, cy, w, h) in data coords /
    points), staying inside ``limits`` = (x0, x1, z0, z1); each point of gap
    costs 0.12, so the nearest clear spot wins.  ``order``: the order in
    which labels are placed (default: as given); ``score`` is each label's
    clearance minus its gap cost, for comparing orders.  ``away_from`` (a data
    point, e.g. the route's centroid) with ``away_weight`` > 0 adds
    ``away_weight * cos`` of the angle between the placement direction and
    the direction from that point to the anchor, so labels fan outward.
    """
    T = ax.transData
    px = ax.figure.dpi / 72.0
    anchors = T.transform(np.asarray(anchors_xy, dtype=float).reshape(-1, 2)) / px
    obst = (T.transform(np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)) / px if len(obstacles_xy)
            else np.zeros((0, 2)))
    x0, x1, z0, z1 = limits
    corners = T.transform(np.array([[x0, z0], [x1, z1]])) / px
    lo, hi = corners.min(0), corners.max(0)
    placed = [(T.transform((cx, cy)) / px, w, h) for cx, cy, w, h in fixed_boxes]
    away = None if away_from is None or away_weight <= 0 else T.transform(np.asarray(away_from, dtype=float)) / px
    out = {}
    for i in (range(len(anchors)) if order is None else order):
        a = anchors[i]
        w, h = boxes_pt[i]
        best, best_score = None, -np.inf
        for r in radii_pt:
            for k in range(n_dir):
                ang = 2 * math.pi * k / n_dir
                d = np.array([math.cos(ang), math.sin(ang)])
                reach = min(w / 2 / (abs(d[0]) + 1e-9), h / 2 / (abs(d[1]) + 1e-9))  # centre-to-edge along d
                c = a + (r + reach) * d
                if not (lo[0] + w / 2 + margin_pt <= c[0] <= hi[0] - w / 2 - margin_pt
                        and lo[1] + h / 2 + margin_pt <= c[1] <= hi[1] - h / 2 - margin_pt):
                    continue
                dx = np.maximum(np.abs(obst[:, 0] - c[0]) - w / 2, 0.0) if len(obst) else np.array([10.0])
                dy = np.maximum(np.abs(obst[:, 1] - c[1]) - h / 2, 0.0) if len(obst) else np.array([0.0])
                ax_ = np.maximum(np.abs(anchors[:, 0] - c[0]) - w / 2, 0.0)
                ay_ = np.maximum(np.abs(anchors[:, 1] - c[1]) - h / 2, 0.0)
                clear = [float(np.min(np.hypot(dx, dy))), float(np.min(np.hypot(ax_, ay_)))]
                for q, qw, qh in placed:  # gap between two boxes (negative when they overlap)
                    clear.append(max(abs(q[0] - c[0]) - (w + qw) / 2, abs(q[1] - c[1]) - (h + qh) / 2))
                score = min(min(clear), clear_pt) - 0.12 * r
                if away is not None:
                    u = a - away
                    nu = float(np.hypot(*u))
                    if nu > 1e-6:
                        score += away_weight * float(np.dot(d, u / nu))
                if score > best_score:
                    best, best_score = (c, r), score
        if best is None:
            best, best_score = (a + np.array([radii_pt[0] + w / 2, 0.0]), radii_pt[0]), -np.inf
        placed.append((best[0], w, h))
        xy = T.inverted().transform(best[0] * px)
        out[i] = (float(xy[0]), float(xy[1]), float(best[1]), float(best_score))
    return [out[i] for i in sorted(out)]


# --------------------------------------------------------------------------- #
# Heading-up local disc (map inset)
# --------------------------------------------------------------------------- #
def bearing_to_xy(bearing_deg, radius=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Heading-up plane (x right, y forward) point at ``bearing`` (left-positive) and ``radius``."""
    b = np.radians(np.asarray(bearing_deg, dtype=float))
    return -np.sin(b) * radius, np.cos(b) * radius


def dodge_1d(targets, widths, lo: float, hi: float, gap: float, periodic: bool = False) -> np.ndarray:
    """Centres near ``targets`` for intervals of ``widths`` without overlap (cluster-and-spread).

    ``periodic``: coordinates wrap at ``hi - lo`` (for angles around a disc).
    """
    t = np.asarray(targets, dtype=float)
    w = np.asarray(widths, dtype=float)
    n = len(t)
    if n == 0:
        return t
    period = hi - lo
    order = np.argsort(t, kind="stable")
    if periodic:  # start the sweep in the widest empty gap so no cluster straddles the cut
        ts = t[order]
        gaps = np.diff(np.concatenate([ts, ts[:1] + period]))
        cut = (int(np.argmax(gaps)) + 1) % n
        order = np.roll(order, -cut)
        base = t[order[0]]
        tt = {int(j): base + ((t[j] - base) % period) for j in order}
    else:
        tt = {int(j): t[j] for j in order}
    clusters = [[int(j)] for j in order]

    def place(cl):
        total = sum(w[j] for j in cl) + gap * (len(cl) - 1)
        centre = float(np.mean([tt[j] for j in cl]))
        start = centre - total / 2
        if not periodic:
            start = float(np.clip(start, lo, hi - total))
        pos, x = {}, start
        for j in cl:
            pos[j] = x + w[j] / 2
            x += w[j] + gap
        return pos, start, start + total

    changed = True
    while changed:
        changed = False
        spans = [place(c) for c in clusters]
        for a in range(len(clusters) - 1):
            if spans[a][2] + gap > spans[a + 1][1]:
                clusters[a] = clusters[a] + clusters[a + 1]
                del clusters[a + 1]
                changed = True
                break
    out = np.zeros(n)
    for c in clusters:
        for j, x in place(c)[0].items():
            out[j] = lo + ((x - lo) % period) if periodic else x
    return out


def draw_local_disc(ax, level, center_xz, forward_xz, half_m: float, past_xz=None, radius_frac: float = 0.74,
                    letters=("F", "R", "B", "L"), sat: float = 0.25, white: float = 0.4, out_px: int = 640,
                    past_split: Optional[int] = None):
    """Round heading-up map around the robot; its rim is the bearing ring the strip unrolls.

    The axes spans +-``half_m / radius_frac`` metres; the disc (radius
    ``half_m``) holds the map, the route so far, and dashed view seams at
    bearings +-45 / +-135 deg.  Returns the :class:`EgoCrop` (world ->
    heading-up metres via ``world_to_local``).  Sector letters and badges go
    outside the disc (``disc_rim_labels``).

    ``past_split``: index into ``past_xz`` where the route turned back; the
    route so far is then drawn as two legs (``draw_route_legs``: outbound
    solid, return dashed, each on its own right-hand side).  ``None`` (the
    default) draws the single grey line of the case figure.
    """
    lim = half_m / radius_frac
    img = (mute_map(level.rgb(), sat=sat, white=white) * 255).astype(np.uint8)
    plate = tuple(int(round(255 * c)) for c in MAP_PLATE)
    crop = level.heading_up_crop(img, center_xz, forward_xz=forward_xz, half_size_m=half_m, out_px=out_px, fill=plate)
    disc = Circle((0, 0), half_m, transform=ax.transData)
    im = ax.imshow(crop.image, extent=crop.extent, interpolation="bilinear", zorder=0)
    im.set_clip_path(disc)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    clean_axes(ax)
    ax.patch.set_visible(False)
    if past_xz is not None and len(past_xz) > 1 and past_split is not None:
        a, b = crop.world_to_local(past_xz[:, 0], past_xz[:, 1])
        draw_route_legs(ax, np.stack([a, b], axis=1), past_split, lw=0.8, alpha=0.8, offset_pt=1.0,
                        chevron_spacing_pt=None, zorder=2, clip_path=Circle((0, 0), half_m, transform=ax.transData))
    elif past_xz is not None and len(past_xz) > 1:
        a, b = crop.world_to_local(past_xz[:, 0], past_xz[:, 1])
        line, = ax.plot(a, b, color=style.INK_2, lw=0.8, alpha=0.75, solid_capstyle="round", zorder=2)
        line.set_clip_path(Circle((0, 0), half_m, transform=ax.transData))
    for b in (45.0, -45.0, 135.0, -135.0):
        x, y = bearing_to_xy(b, half_m)
        ax.plot([0, x], [0, y], color=style.INK_2, lw=0.45, ls=(0, (2.0, 1.6)), alpha=0.8, zorder=1.5)
    ax.add_patch(Circle((0, 0), half_m, fill=False, ec=style.MUTED, lw=0.6, zorder=3))
    return crop


def _badge_support(label: str, u) -> float:
    """Half extent (pt) of a history badge along unit direction ``u``: a disc for one digit, else its box."""
    if len(label) == 1:
        return badge_width_pt(label) / 2
    w, h = badge_width_pt(label), 7.4
    return abs(u[0]) * w / 2 + abs(u[1]) * h / 2


def disc_rim_layout(ax, half_m: float, groups: Sequence[Sequence[int]], bearings: Sequence[float],
                    gap_pt: float = 1.0, offset_pt: float = 5.2):
    """Where ``disc_rim_labels`` puts its badges, without drawing: (labels, theta, placed, widths).

    ``theta`` / ``placed``: true and dodged plot angles (degrees), ``widths``:
    angular width of each badge on the ring.  Badges are dodged in the order of
    their bearings (ties: slot order), so their leaders never cross; a pill
    badge ("1–3") is measured by its real extent along the rim (its width
    beside the disc, its height above or below it).
    """
    per_pt = pts_to_data(ax, 1.0)[0]
    r_pt = half_m / per_pt
    labels = [f"{g[0] + 1}" if len(g) == 1 else f"{g[0] + 1}–{g[-1] + 1}" for g in groups]
    theta = np.array([90.0 + b for b in bearings])  # heading-up: bearing 0 = up, left-positive = counter-clockwise
    widths = []
    for lab, t in zip(labels, theta):
        u = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
        tang = np.array([-u[1], u[0]])
        ring = r_pt + (offset_pt - 3.7) + _badge_support(lab, u)
        widths.append(math.degrees(2 * _badge_support(lab, tang) / ring))
    widths = np.array(widths)
    ring0 = r_pt + offset_pt
    # bearing -180 and +180 are one angle: normalise before sorting; ties in slot order
    order_key = np.mod(theta, 360.0) + 1e-6 * np.array([g[0] for g in groups], dtype=float)
    placed = dodge_1d(order_key, widths, 0.0, 360.0, math.degrees(gap_pt / ring0), periodic=True)
    return labels, theta, placed, widths


def disc_rim_labels(ax, half_m: float, groups: Sequence[Sequence[int]], bearings: Sequence[float],
                    num: int = 8, gap_pt: float = 1.0, offset_pt: float = 5.2) -> List[Tuple[float, float]]:
    """Numbered badges just outside the disc rim at each group's bearing, dodged along the rim in bearing order.

    Returns ``(theta, half_width)`` in degrees (plot angle, 0 = +x, counter-
    clockwise) of every placed badge.  A thin leader joins a dodged badge to
    its true bearing on the rim.  A badge's centre sits ``offset_pt - 3.7`` pt
    plus its own half extent outside the rim, so a pill beside the disc keeps
    the same clearance as a disc badge.
    """
    if not groups:
        return []
    per_pt = pts_to_data(ax, 1.0)[0]
    r_pt = half_m / per_pt  # disc radius in points
    labels, theta, placed, widths = disc_rim_layout(ax, half_m, groups, bearings, gap_pt=gap_pt, offset_pt=offset_pt)
    out = []
    for g, lab, t, p, w in zip(groups, labels, theta, placed, widths):
        k = g[0]
        u_t = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
        u_p = np.array([math.cos(math.radians(p)), math.sin(math.radians(p))])
        ring_pt = r_pt + (offset_pt - 3.7) + _badge_support(lab, u_p)
        c = u_p * ring_pt * per_pt
        if abs(((p - t + 180) % 360) - 180) > 0.8:
            rim, mid = u_t * half_m, u_t * (r_pt + 1.8) * per_pt
            ax.plot([rim[0], mid[0], c[0]], [rim[1], mid[1], c[1]], color=history_line_color(k, num), lw=0.5,
                    zorder=5, solid_joinstyle="round", clip_on=False)
        history_badge(ax, c[0], c[1], lab, k, num=num, zorder=6)
        out.append((float(p % 360.0), float(w / 2)))
    return out


def _rendered_obstacles(ax, skip=()) -> list:
    """Display-space extents of what is drawn on ``ax`` outside the plot's clip: badges (bbox texts), other
    texts, clip-free or haloed lines (leaders, arrows, scale bars) and clip-free patches."""
    rend = ax.figure.canvas.get_renderer()
    out = []
    for t in ax.texts:
        if t in skip or not t.get_text().strip():
            continue
        if t.get_bbox_patch() is not None:
            t.update_bbox_position_size(rend)
            out.append(t.get_bbox_patch().get_window_extent(rend))
        else:
            out.append(t.get_window_extent(rend))
    for ln in ax.lines:
        if (not ln.get_clip_on() or ln.get_path_effects()) and len(ln.get_xdata()) > 1:
            xy = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            for i in range(len(xy) - 1):  # per segment: a bent leader is not its bounding box
                seg = xy[i:i + 2]
                n = max(2, int(np.hypot(*(seg[1] - seg[0])) / 3.0))
                for q in np.linspace(seg[0], seg[1], n):
                    out.append(Bbox.from_bounds(q[0] - 0.4, q[1] - 0.4, 0.8, 0.8))
    for pa in ax.patches:
        if not pa.get_clip_on():
            out.append(pa.get_window_extent(rend))
    return out


def _inner_obstacles(ax) -> list:
    """Display-space points (as tiny boxes) of every line and marker drawn on ``ax`` except the map image:
    rays, past-position dots, the route, view seams, the robot."""
    out = []
    for ln in ax.lines:
        xd, yd = np.asarray(ln.get_xdata(), dtype=float), np.asarray(ln.get_ydata(), dtype=float)
        if not len(xd):
            continue
        xy = ax.transData.transform(np.column_stack([xd, yd]))
        ms = float(ln.get_markersize()) * ax.figure.dpi / 72.0 if ln.get_marker() not in (None, "None", "") else 0.0
        if len(xy) == 1 or ms > 0:
            for q in xy:
                h = max(ms / 2, 0.8)
                out.append(Bbox.from_bounds(q[0] - h, q[1] - h, 2 * h, 2 * h))
        for i in range(len(xy) - 1):
            seg = xy[i:i + 2]
            n = max(2, int(np.hypot(*(seg[1] - seg[0])) / 2.5))
            for q in np.linspace(seg[0], seg[1], n):
                out.append(Bbox.from_bounds(q[0] - 0.5, q[1] - 0.5, 1.0, 1.0))
    for pa in ax.patches:
        if isinstance(pa, Polygon):
            out.append(pa.get_window_extent(ax.figure.canvas.get_renderer()))
    return out


def disc_sector_letters(ax, half_m: float, occupied: Sequence[Tuple[float, float]] = (), offset_pt: float = 5.2,
                        names=("F", "R", "B", "L"), fs: float = 6.0, letter_pt: float = 4.6,
                        displaced: str = "outward", rays: Sequence[float] = (), obstacles: str = "auto",
                        bounds=None) -> None:
    """Sector letters at the sector centres on the badge ring outside the rim, never under a badge.

    Call after the badges, leaders, arrow and scale bar are drawn: with
    ``obstacles="auto"`` (default) every letter is tested against their real
    drawn extents (a pill badge beside the disc reaches far out radially).  A
    blocked letter first slides along the ring to the free spot nearest its
    sector centre within +-38 deg (it stays in its own 90 deg sector); then
    (``displaced="outward"`` or ``"slide"``) it moves radially outward past
    the badges at its sector centre, up to 14 pt, while that stays clear and
    inside ``bounds`` (a display-space box, default the axes: pass the free
    room around the disc to let a blocked F or B sit just beyond the badges);
    else it goes just inside the rim at the angle within +-38 deg of the
    centre nearest to it that keeps clear of ``rays`` (plot angles of the
    blue rays, degrees) and of the drawn dots and badges.
    ``displaced="inside"`` skips the outward step.  ``obstacles="angles"``:
    the old angular test against ``occupied`` only.
    """
    per_pt = pts_to_data(ax, 1.0)[0]
    px = ax.figure.dpi / 72.0
    r_pt = half_m / per_pt
    ring_pt = r_pt + offset_pt
    inner_pt = r_pt - letter_pt / 2 - 2.4
    steps = sorted(np.arange(-38.0, 38.01, 1.0), key=lambda v: (abs(v), v))
    obst = _rendered_obstacles(ax) if obstacles == "auto" else []
    half_letter = math.degrees((letter_pt / 2 + 0.8) / ring_pt)
    ax_box = ax.get_window_extent() if bounds is None else bounds

    def ang_free(th):
        return all(abs((th - t + 180) % 360 - 180) > w + half_letter for t, w in occupied)

    def inside_angle(th):
        if not len(rays):
            return th
        need = math.degrees((letter_pt / 2 + 1.4) / max(inner_pt, 1.0))
        dist = {d: min(abs((th + d - r + 180) % 360 - 180) for r in rays) for d in steps}
        ok = [d for d in steps if dist[d] >= need]
        return th + (ok[0] if ok else max(steps, key=lambda d: dist[d]))

    rend = ax.figure.canvas.get_renderer()
    for name, centre_bearing in zip(names, geo.VIEW_YAWS_DEG):
        col = style.INK if name == names[0] else style.MUTED
        t = ax.text(0, 0, name, ha="center", va="center", fontsize=fs, fontweight="bold", color=col, zorder=6,
                    path_effects=bold_effects(name, col, halo=1.1) if has_cjk(name) else HALO_THIN)
        centre = 90.0 + centre_bearing

        def put(th, radius_pt):
            t.set_position((radius_pt * per_pt * math.cos(math.radians(th)),
                            radius_pt * per_pt * math.sin(math.radians(th))))

        def free():
            bb = t.get_window_extent(rend)
            if obstacles != "auto":
                th = math.degrees(math.atan2(*t.get_position()[::-1]))
                return ang_free(th)
            if bb.x0 < ax_box.x0 - 1.0 or bb.x1 > ax_box.x1 + 1.0 or bb.y0 < ax_box.y0 - 1.0 or bb.y1 > ax_box.y1 + 1.0:
                return False
            return not any(bb.x0 - 0.8 * px < o.x1 and o.x0 < bb.x1 + 0.8 * px and bb.y0 - 0.8 * px < o.y1
                           and o.y0 < bb.y1 + 0.8 * px for o in obst)

        done = False
        for d in ([0.0] if displaced == "inside" else steps):
            put(centre + d, ring_pt)
            if free():
                done = True
                break
        if not done and displaced in ("outward", "slide"):
            for extra in np.arange(1.0, 14.01, 1.0):
                put(centre, ring_pt + extra)
                if free():
                    done = True
                    break
        if not done and obstacles == "auto":  # just inside the rim, clear of the rays, dots and badges
            inner_obst = obst + _inner_obstacles(ax)
            for rad in (inner_pt, inner_pt - 3.0, inner_pt - 6.0):
                for d in steps:
                    put(centre + d, rad)
                    bb = t.get_window_extent(rend)
                    if not any(bb.x0 - 0.6 * px < o.x1 and o.x0 < bb.x1 + 0.6 * px and bb.y0 - 0.6 * px < o.y1
                               and o.y0 < bb.y1 + 0.6 * px for o in inner_obst):
                        done = True
                        break
                if done:
                    break
        if not done:
            put(inside_angle(centre), inner_pt)
        if obstacles == "auto":
            obst.append(t.get_window_extent(rend))


def disc_direction_arrow(ax, half_m: float, start_bearing: float = STRIP_START_DEG, sweep_deg: float = 34.0,
                         offset_pt: float = 3.4, color: str = style.INK_2) -> None:
    """Curved arrow outside the rim from ``start_bearing`` clockwise: where and how the strip unrolls."""
    r = half_m + pts_to_data(ax, offset_pt)[0]
    th = np.radians(np.linspace(90.0 + start_bearing, 90.0 + start_bearing - sweep_deg, 24))
    xs, ys = r * np.cos(th), r * np.sin(th)
    ax.plot(xs[:-3], ys[:-3], color=color, lw=0.7, zorder=5, solid_capstyle="butt", clip_on=False)
    ax.add_patch(FancyArrowPatch((xs[-4], ys[-4]), (xs[-1], ys[-1]), arrowstyle="-|>", mutation_scale=4.6,
                                 color=color, lw=0.7, shrinkA=0, shrinkB=0, zorder=5, clip_on=False))


# --------------------------------------------------------------------------- #
# Surround strip
# --------------------------------------------------------------------------- #
def strip_x(bearing):
    """Strip coordinate in degrees [0, 360): 0 = front view's left edge, increasing clockwise."""
    return np.mod(STRIP_START_DEG - np.asarray(bearing, dtype=float), 360.0)


def ring_to_strip(ring: np.ndarray) -> np.ndarray:
    """Heading-centred ring (+180 deg at the left edge) -> F R B L strip, by an exact column roll."""
    width = ring.shape[1]
    shift = (180.0 - STRIP_START_DEG) * width / 360.0
    if abs(shift - round(shift)) > 1e-9:
        raise ValueError("ring width must make the roll exact (use a multiple of 8)")
    return np.roll(ring, -int(round(shift)), axis=1)


def elev_window(elev) -> Tuple[float, float]:
    """(lo, hi) degrees of an elevation window given as a half-width (``8`` -> (-8, 8)) or a (lo, hi) pair."""
    if np.ndim(elev) == 0:
        e = float(elev)
        return -e, e
    lo, hi = elev
    return float(lo), float(hi)


def elevation_window(rows, arm: str = "vo", default: float = None, margin: float = None, cap: float = None,
                     ) -> Tuple[float, float]:
    """The one elevation-window rule of every EXP-18 figure (orchestrator decision D4).

    ``(lo, hi)`` degrees for the affordance map rows of one figure block showing
    ``rows`` (``data.CaseRow``): ``+-EL_DEFAULT`` (10 deg), widened on each side
    just enough (``EL_MARGIN`` so the x mark fits, whole degrees) to include
    every GT-visible slot's ground-truth peak and every drawn predicted peak
    (GT-visible slots with P(not visible) <= 0.5, ``CaseRow.peak_slots``), at
    most ``+-EL_MAX`` (45 deg, the views' own vertical field of view).  State it
    in the caption with ``elevation_text``.
    """
    default = EL_DEFAULT if default is None else float(default)
    margin = EL_MARGIN if margin is None else float(margin)
    cap = EL_MAX if cap is None else float(cap)
    els: List[float] = []
    for r in rows:
        if r.gt_peak_elev is not None:
            els += [float(v) for v in np.asarray(r.gt_peak_elev)[r.visible]]
        p = r.arms[arm]
        els += [float(p.peak_elev[k]) for k in r.peak_slots(arm)]
    els = [v for v in els if np.isfinite(v)]
    lo, hi = -default, default
    if els:
        lo = min(lo, float(math.floor(min(els) - margin)))
        hi = max(hi, float(math.ceil(max(els) + margin)))
    return max(lo, -cap), min(hi, cap)


def fmt_deg(v: float, signed: bool = True) -> str:
    """"−12°" / "+10°" / "0°" (a real minus sign)."""
    v = int(round(float(v)))
    if v == 0 or not signed:
        return f"{abs(v) if not signed else 0}°"
    return f"{'+' if v > 0 else '−'}{abs(v)}°"


ELEV_TEXT = {"en": "rows show elevation {lo} to {hi}", "zh": "各行显示仰角 {lo} 至 {hi}"}


def elevation_text(win, lang: str = "en") -> str:
    """Caption wording of an elevation window: "rows show elevation −10° to +10°"."""
    lo, hi = elev_window(win)
    return ELEV_TEXT["zh" if lang == "zh" else "en"].format(lo=fmt_deg(lo), hi=fmt_deg(hi))


def strip_height_in(width_in: float, elev) -> float:
    """Height (in) of a strip row of ``width_in`` over window ``elev`` (square degrees)."""
    lo, hi = elev_window(elev)
    return width_in * (hi - lo) / 360.0


def rgb_strip(views: np.ndarray, width: int, elev, observed=(geo.FRONT,), sat: float = 0.12,
              white: float = 0.5) -> np.ndarray:
    """[h, width, 3] float strip of the four views over ``elev`` (half-width or (lo, hi)); others muted."""
    lo, hi = elev_window(elev)
    h = int(round(width * (hi - lo) / 360.0))
    ring, _ = geo.stitch_ring_rgb(views, width=width, height=h, elev_top=hi, elev_bottom=lo)
    strip = ring_to_strip(ring).astype(np.float32) / 255.0
    out = mute(strip, sat=sat, white=white)
    q = width // 4
    for v in observed:  # strip panels are F, R, B, L in view order
        out[:, v * q:(v + 1) * q] = strip[:, v * q:(v + 1) * q]
    return out


def heat_strip(maps: np.ndarray, width: int, elev) -> np.ndarray:
    """[h, width] strip of four 64x64 label-convention maps over ``elev`` (0 outside every view)."""
    lo, hi = elev_window(elev)
    h = int(round(width * (hi - lo) / 360.0))
    ring, _ = geo.stitch_ring(maps, width=width, height=h, elev_top=hi, elev_bottom=lo, fill=0.0)
    return ring_to_strip(np.clip(ring, 0.0, 1.0))


def setup_strip_axes(ax, elev) -> None:
    lo, hi = elev_window(elev)
    ax.set_xlim(0, 360)
    ax.set_ylim(lo, hi)
    ax.set_autoscale_on(False)
    clean_axes(ax)


def draw_rgb_row(ax, strip: np.ndarray, elev, frame_views=(geo.FRONT,)) -> None:
    lo, hi = elev_window(elev)
    ax.imshow(strip, extent=(0, 360, lo, hi), aspect="auto", interpolation="bilinear", zorder=0)
    setup_strip_axes(ax, elev)
    for s in (90, 180, 270):
        ax.axvline(s, color="white", lw=0.9, zorder=2)
    for v in frame_views:
        ax.add_patch(Rectangle((v * 90, lo), 90, hi - lo, fill=False, ec=style.INK, lw=1.0, zorder=5,
                               clip_on=False))


def draw_heat_row(ax, strip: np.ndarray, elev, cmap, top: float = HEAT_TOP) -> None:
    """Heat strip coloured linearly by value (0..1 -> the first ``top`` of ``cmap``) on a framed row."""
    lo, hi = elev_window(elev)
    ax.imshow(cmap(top * np.clip(strip, 0, 1)), extent=(0, 360, lo, hi), aspect="auto",
              interpolation="bilinear", zorder=0)
    setup_strip_axes(ax, elev)
    for s in (90, 180, 270):
        ax.axvline(s, color=style.GRID, lw=0.6, zorder=1)
    ax.add_patch(Rectangle((0, lo), 360, hi - lo, fill=False, ec=style.AXIS, lw=0.5, zorder=4, clip_on=False))


def quietest_panel(*strips: np.ndarray, empty: float = 0.05) -> int:
    """Strip panel for in-row labels: the first (F, R, B, L order) empty in every strip, else the quietest."""
    q = strips[0].shape[1] // 4
    peaks = [max(float(s[:, v * q:(v + 1) * q].max()) for s in strips) for v in range(4)]
    for v in range(4):
        if peaks[v] < empty:
            return v
    return int(np.argmin(peaks))


LABEL_PILL = dict(boxstyle="round,pad=0.22,rounding_size=0.55", fc="white", ec="none", alpha=0.86)


def fixed_label_panel(strips: Sequence[np.ndarray], width_deg: float, prefer: int = 0, max_heat: float = 0.35,
                      start_deg: float = 1.0) -> int:
    """One strip panel for the in-row labels of a whole figure (so they never change sides).

    ``strips``: every heat strip the labels will sit on (all rows of all
    blocks).  The label covers ``[v * 90 + start_deg, + width_deg]`` of panel
    ``v``; the preferred panel (default the front view, the left end of each
    row) is kept while the heat under the label stays below ``max_heat`` in
    every strip (a backing pill keeps faint heat readable); otherwise the
    panel with the least heat under the label, in F, R, B, L order on ties.
    """
    heat = []
    for v in range(4):
        m = 0.0
        for s in strips:
            q = s.shape[1] / 360.0
            a = int((v * 90 + start_deg) * q)
            b = max(int((v * 90 + start_deg + width_deg) * q), a + 1)
            m = max(m, float(s[:, a:b].max()))
        heat.append(m)
    if heat[prefer] < max_heat:
        return prefer
    return int(np.argmin(heat))


def row_label(ax, x: float, text: str, fs: float, color: str = style.INK_2, zorder: float = 6.2):
    """Row name ("ground truth") at strip coordinate ``x`` (left end), on a white pill so heat under it stays readable."""
    return ax.text(x, 0, text, ha="left", va="center", fontsize=fs, color=color, zorder=zorder, bbox=dict(LABEL_PILL))


def route_frame(xz: np.ndarray, split: Optional[int], aspect_hw: float, pad: float):
    """Map rotation for a route panel: (centre_xz, forward_xz) of the heading-up crop that shows the route largest.

    Only quarter turns (walls stay axis-aligned): the one whose padded route
    box fits an axes of height/width ``aspect_hw`` at the largest scale; of
    the two with that scale, the one with the start below ``xz[split]`` (the
    outbound leg goes up the page; ``split=None``: below the last point);
    the unrotated map on a full tie.  A top-down map has no preferred
    orientation, so this only changes the scale.
    """
    from scripts.exp18.topdown.topdown_io import EgoCrop

    xz = np.asarray(xz, dtype=float).reshape(-1, 2)
    split = len(xz) - 1 if split is None else int(split)
    centre = (xz.min(0) + xz.max(0)) / 2.0
    best = None
    for n, fwd in enumerate(((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))):
        crop = EgoCrop(centre, fwd, 1.0, 2)
        a, b = crop.world_to_local(xz[:, 0], xz[:, 1])
        scale = min(1.0 / (np.ptp(a) + 2 * pad), aspect_hw / (np.ptp(b) + 2 * pad))
        key = (round(float(scale), 6), bool(b[0] <= b[split] + 1e-6), -n)
        if best is None or key > best[0]:
            best = (key, fwd)
    return centre, np.asarray(best[1])


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def uncross(anchors, centres, max_pass: int = 6) -> List[int]:
    """Permutation of ``centres`` (label positions) so that no two anchor->label leaders cross.

    Swaps the labels of any crossing pair until none cross (or ``max_pass``
    sweeps); returns ``perm`` with label ``i`` placed at ``centres[perm[i]]``.
    """
    A = [np.asarray(a, dtype=float) for a in anchors]
    C = [np.asarray(c, dtype=float) for c in centres]
    perm = list(range(len(A)))
    for _ in range(max_pass):
        changed = False
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                if _segments_cross(A[i], C[perm[i]], A[j], C[perm[j]]):
                    perm[i], perm[j] = perm[j], perm[i]
                    changed = True
        if not changed:
            break
    return perm


def cluster_1d(values: Sequence[float], tol: float) -> List[List[int]]:
    """Clusters of indices whose values span at most ``tol`` (sweep in sorted order; no chaining)."""
    v = np.asarray(values, dtype=float)
    order = [int(i) for i in np.argsort(v)]
    out: List[List[int]] = []
    for i in order:
        if out and v[i] - v[out[-1][0]] <= tol:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def azimuth_axis(ax, labels: Sequence[str], fs: float = 6.0) -> None:
    """Degree labels under a strip row: panel centres labelled, view seams ticked."""
    ax.set_xticks([45, 135, 225, 315])
    ax.set_xticklabels(labels, fontsize=fs, color=style.INK_2)
    ax.set_xticks([0, 90, 180, 270, 360], minor=True)
    ax.tick_params(axis="x", which="major", length=0, pad=3.0)
    ax.tick_params(axis="x", which="minor", length=2.4, width=0.5, color=style.AXIS)


# --------------------------------------------------------------------------- #
# Frame numbers (D6): one wording in every figure
# --------------------------------------------------------------------------- #
FRAME_LABEL = {"en": "frame {n} of {T}", "zh": "第 {n} 帧（共 {T} 帧）"}


def frame_label(frame_id: int, frame_count: int, lang: str = "en") -> str:
    """"frame 35 of 79": a 1-based count (the dump's 0-based frame id + 1) out of the clip's frame count."""
    return FRAME_LABEL["zh" if lang == "zh" else "en"].format(n=int(frame_id) + 1, T=int(frame_count))


# --------------------------------------------------------------------------- #
# Bold Chinese text: the CJK font has no bold face, so bold is drawn as a thin stroke of the text's own colour
# --------------------------------------------------------------------------- #
def has_cjk(text: str) -> bool:
    """True when ``text`` holds CJK ideographs or full-width forms."""
    return any("　" <= ch <= "鿿" or "＀" <= ch <= "￯" for ch in str(text))


def bold_effects(text: str, color, halo: Optional[float] = None, stroke_pt: float = 0.45):
    """``path_effects`` for bold ``text``: with CJK characters a ``stroke_pt`` stroke in ``color`` (fake bold,
    since the CJK font has no bold face), under an optional white ``halo`` of that line width; ``None`` when
    neither applies (Latin bold comes from the font)."""
    effects = []
    if halo:
        effects.append(pe.Stroke(linewidth=halo, foreground="white"))
    if has_cjk(text):
        effects.append(pe.Stroke(linewidth=stroke_pt, foreground=color))
    return effects + [pe.Normal()] if effects else None


# --------------------------------------------------------------------------- #
# Row names in a fixed left gutter (D3)
# --------------------------------------------------------------------------- #
ROW_LABEL_FS = 5.8


def gutter_row_label(ax_row, text: str, kind: str, fs: float = ROW_LABEL_FS, gap_pt: float = 4.0,
                     key_w_pt: float = 19.0, key_h_pt: float = 2.3, zorder: float = 6):
    """Row name ("ground truth" / "prediction") right-aligned in the gutter left of ``ax_row``, outside the strip.

    ``kind`` "gt" (blue text, blue ramp key) or "pred" (orange).  The name sits
    just above the row's middle and a tiny colour ramp key (``key_w_pt`` x
    ``key_h_pt``) just below it, both ending ``gap_pt`` left of the row.  The
    same place in every block of every figure, whatever the maps show.  Bold
    (a CJK name gets ``bold_effects``).  Returns ``(text, key_axes)`` so a
    caller can keep other marks clear of them.
    """
    fig = ax_row.figure
    box = ax_row.get_position()
    W, H = fig.get_size_inches() * 72.0
    x_right = box.x0 - gap_pt / W
    yc = (box.y0 + box.y1) / 2.0
    color, cmap = (GT_INK, GT_CMAP) if kind == "gt" else (PRED_INK, PRED_CMAP)
    t = fig.text(x_right, yc + 1.9 / H, text, ha="right", va="bottom", fontsize=fs, color=color, fontweight="bold",
                 zorder=zorder, path_effects=bold_effects(text, color, stroke_pt=0.065 * fs))
    key = fig.add_axes([x_right - key_w_pt / W, yc - (1.0 + key_h_pt) / H, key_w_pt / W, key_h_pt / H])
    key.imshow(cmap(HEAT_TOP * np.linspace(0.0, 1.0, 64))[None], aspect="auto", extent=(0, 1, 0, 1),
               interpolation="bilinear")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")
    return t, key


def gutter_width_pt(fig, texts: Sequence[str], fs: float = ROW_LABEL_FS, gap_pt: float = 4.0) -> float:
    """Gutter width (pt) the row names need (bold ``fs``) plus ``gap_pt``."""
    return max(text_width_pt(fig, t, fs, fontweight="bold") for t in texts) + gap_pt


# --------------------------------------------------------------------------- #
# Notes (D5): never dropped -- wrapped onto as many lines as they need
# --------------------------------------------------------------------------- #
NOTE_FS = 5.7
NOTE_BADGE_GAP_PT = 1.4  # between a note's badge and its text
NOTE_SEP_PT = 10.0  # between two notes on one line


def note_width_pt(fig, item: dict, fs: float = NOTE_FS) -> float:
    """Width (pt) of a note: its badge (``item["label"]``, may be empty) + gap + text."""
    w = text_width_pt(fig, item["text"], fs)
    if item.get("label"):
        w += badge_width_pt(item["label"]) + NOTE_BADGE_GAP_PT
    return w


def split_note(fig, item: dict, max_pt: float, fs: float = NOTE_FS) -> List[dict]:
    """A note wider than ``max_pt``: cut its text at " · " (then at spaces) so each part fits; the parts after the
    first carry no badge (``continuation``).  Nothing is dropped."""
    parts = item["text"].split(" · ")
    out, cur = [], dict(item, text=parts[0])
    for part in parts[1:]:
        trial = dict(cur, text=cur["text"] + " · " + part)
        if note_width_pt(fig, trial, fs) <= max_pt:
            cur = trial
        else:
            out.append(cur)
            cur = dict(item, text="· " + part, label="", continuation=True)
    out.append(cur)
    final = []
    for it in out:  # a part still too wide (one long clause): break at spaces
        if note_width_pt(fig, it, fs) <= max_pt or " " not in it["text"]:
            final.append(it)
            continue
        words, line = it["text"].split(" "), ""
        first = True
        for w in words:
            trial = w if not line else line + " " + w
            probe = dict(it, text=trial) if first else dict(it, text=trial, label="")
            if line and note_width_pt(fig, probe, fs) > max_pt:
                final.append(dict(it, text=line) if first else dict(it, text=line, label="", continuation=True))
                first, line = False, w
            else:
                line = trial
        final.append(dict(it, text=line) if first else dict(it, text=line, label="", continuation=True))
    return final


def wrap_notes(fig, items: Sequence[dict], line_pt: float, first_pt: Optional[float] = None,
               fs: float = NOTE_FS, sep_pt: float = NOTE_SEP_PT):
    """Lay notes out left to right: ``(first, lines)`` with every note placed (D5).

    ``first``: the notes that fit, in order, into ``first_pt`` points (a line
    shared with other marks, e.g. the space left of a block's lane badges;
    ``None``: no such line); ``lines``: the rest on full lines of ``line_pt``
    points, as many as they need.  Each entry is ``(x_pt, item)``, x from the
    line's left end.  A note wider than a whole line is split
    (``split_note``) first.
    """
    flat: List[dict] = []
    for it in items:
        flat += split_note(fig, it, line_pt, fs) if note_width_pt(fig, it, fs) > line_pt else [dict(it)]
    first: list = []
    lines: list = []
    x = 0.0
    j = 0
    if first_pt is not None:
        while j < len(flat) and x + note_width_pt(fig, flat[j], fs) <= first_pt:
            first.append((x, flat[j]))
            x += note_width_pt(fig, flat[j], fs) + sep_pt
            j += 1
    cur: list = []
    x = 0.0
    for it in flat[j:]:
        w = note_width_pt(fig, it, fs)
        if cur and x + w > line_pt:
            lines.append(cur)
            cur, x = [], 0.0
        cur.append((x, it))
        x += w + sep_pt
    if cur:
        lines.append(cur)
    return first, lines


def draw_note(ax, x: float, y: float, item: dict, per_pt: float, fs: float = NOTE_FS, color: str = style.INK_2,
              num: int = 8) -> None:
    """One note at data (x, y), left end: its badge -- a blue past-position badge (``item["style"]`` "hist") or an
    orange-ringed miss badge ("miss") -- then its text.  ``per_pt``: data units per point along x."""
    if item.get("label"):
        bw = badge_width_pt(item["label"]) * per_pt
        if item.get("style") == "miss":
            miss_badge(ax, x + bw / 2, y, item["label"])
        else:
            history_badge(ax, x + bw / 2, y, item["label"], item["slots"][0], num=num)
        x += bw + NOTE_BADGE_GAP_PT * per_pt
    ax.text(x, y, item["text"], ha="left", va="center", fontsize=fs, color=color)


# --------------------------------------------------------------------------- #
# Predicted peaks and numbered misses (D1, D2)
# --------------------------------------------------------------------------- #
MARK_PT = 4.4  # size of the predicted-peak x
MERGE_DEG = 4.0  # hits closer than this share one x
STAGGER_PT = 2.6  # vertical offset of an x that would touch its neighbour
MISS_BH = 3.9  # half height of a numbered badge (pt)
LABEL_CLEAR_PT = 3.5  # a badge without a leader keeps at least this far from any other x (2 pt with one)
BELOW_LANE_PT = 8.6  # centre of a badge in the lane under a prediction row (the true-bearing ticks take 1..4 pt)
BELOW_LANE_H_PT = BELOW_LANE_PT + MISS_BH + 0.8  # room that lane needs under the row
MISS_RING = "#b53f12"  # the miss badge's ring: a dark step of the prediction orange (reads on the orange map)
LEADER_COLOR = style.INK  # a moved badge's leader: ink with a white halo (an orange line vanishes on the map)


def miss_badge(ax, x, y, label: str, transform=None, zorder: float = 8, fs: float = BADGE_FS):
    """A missed slot's number (D2): ink digits in a small white disc ringed in the prediction orange.

    Same box as ``history_badge`` (``badge_width_pt`` holds); never blue, so it
    cannot be read as a ground-truth past position.
    """
    box = "circle,pad=0.22" if len(label) == 1 else "round,pad=0.24,rounding_size=0.62"
    kw = dict(ha="center", va="center", fontsize=fs, fontweight="bold", transform=transform or ax.transData,
              clip_on=False)
    # a white outline under the ring keeps it visible on the orange map
    ax.text(x, y, label, color=(1, 1, 1, 0), zorder=zorder - 0.01,
            bbox=dict(boxstyle=box, fc="white", ec="white", lw=2.4), **kw)
    return ax.text(x, y, label, color=style.INK, zorder=zorder,
                   bbox=dict(boxstyle=box, fc="white", ec=MISS_RING, lw=0.95), **kw)


def slots_label(slots: Sequence[int]) -> str:
    """1-based slot list as runs: [0, 1, 2, 5] -> "1–3, 6"."""
    runs, out = sorted(int(k) for k in slots), []
    for k in runs:
        if out and k == out[-1][-1] + 1:
            out[-1].append(k)
        else:
            out.append([k])
    return ", ".join(str(r[0] + 1) if len(r) == 1 else f"{r[0] + 1}–{r[-1] + 1}" for r in out)


def _cluster_2d(pts: Sequence[Tuple[float, float]], tol: float) -> List[List[int]]:
    """Greedy clusters (sweep in x): a point joins the first cluster whose centroid is within ``tol`` in both
    coordinates."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    out: List[List[int]] = []
    for i in order:
        for cl in out:
            cx = float(np.mean([pts[j][0] for j in cl]))
            cy = float(np.mean([pts[j][1] for j in cl]))
            if abs(pts[i][0] - cx) <= tol and abs(pts[i][1] - cy) <= tol:
                cl.append(i)
                break
        else:
            out.append([i])
    return out


def peak_marks(row, arm: str, elev, ppd: float, merge_deg: float = MERGE_DEG, mark_pt: float = MARK_PT,
               stagger_pt: float = STAGGER_PT) -> List[dict]:
    """The x marks of one prediction row, in strip degrees (the rows are square degrees, ``ppd`` pt per degree).

    One x per GT-visible slot with a predicted peak (``CaseRow.peak_slots``):
    hits within ``merge_deg`` of each other (in bearing and elevation) share
    one x; misses (D1) never share an x with a hit, and misses whose peaks lie
    within one mark width of each other (they would print as one blot) share
    one x and one numbered badge ("1–3").  Marks that would still touch are
    staggered vertically by ``stagger_pt``; an x never crosses the row's
    top or bottom edge, and reaches at most 1 pt past a strip end (a peak
    right at the end moves inward by < 1.2 deg, under one map pixel;
    ``x_peak`` keeps the exact place).  Slots with no ground-truth view get no x (their map still shows in
    the row; their notes give P(not visible)).  Returns [{"x", "y", "y_peak",
    "slots", "miss", "slot", "label"}] sorted by x; ``slot`` is the one slot of
    a single-slot mark (else None, kept for older callers); ``label`` the miss
    badge text (``slots_label``).
    """
    lo, hi = elev_window(elev)
    p = row.arms[arm]
    peaks = row.peak_slots(arm)
    missed = set(row.misses(arm))
    marks = []
    for group, is_miss, tol in (([k for k in peaks if k not in missed], False, merge_deg),
                                ([k for k in peaks if k in missed], True, 0.8 * mark_pt / ppd)):
        pts = [(float(strip_x(p.peak_bearing[k])), float(p.peak_elev[k])) for k in group]
        for cl in _cluster_2d(pts, tol):
            ks = sorted(group[i] for i in cl)
            x = float(np.mean([pts[i][0] for i in cl]))
            y = float(np.mean([pts[i][1] for i in cl]))
            marks.append({"x": x, "y": y, "y_peak": y, "slots": ks, "miss": is_miss,
                          "slot": ks[0] if (is_miss and len(ks) == 1) else None,
                          "label": slots_label(ks) if is_miss else ""})
    x_edge = max(mark_pt / 2 - 1.2, 0.0) / ppd  # an x may reach 1 pt past a strip end, never into the row names
    for m in marks:
        m["x_peak"] = m["x"]
        m["x"] = float(np.clip(m["x"], x_edge, 360.0 - x_edge))  # moves a peak at a strip end by < 1.2 deg
    marks.sort(key=lambda m: (m["x"], m["y"]))
    sign = 1.0
    off = [0.0] * len(marks)
    for j in range(1, len(marks)):
        a, b = marks[j - 1], marks[j]
        if ((b["x"] - a["x"]) * ppd < mark_pt + 3.4 and abs(b["y"] - a["y"]) * ppd < mark_pt + 1.0):
            if off[j - 1] == 0.0:
                off[j - 1] = sign * stagger_pt
            off[j] = -np.sign(off[j - 1]) * stagger_pt
            sign = -sign
    edge = (mark_pt / 2 + 1.0) / ppd
    for m, o in zip(marks, off):
        m["y"] = float(np.clip(m["y"] + o / ppd, lo + edge, hi - edge))
    return marks


def place_miss_labels(marks: Sequence[dict], ppd: float, elev, heat=None, below: bool = False,
                      mark_pt: float = MARK_PT, lines: Sequence[Sequence[Tuple[float, float]]] = ()) -> dict:
    """Where each numbered miss's badge goes (pure layout, strip degrees; D2).

    ``marks``: ``peak_marks`` output; one badge per miss mark.  A
    badge sits beside its own x, or slides along one of the lanes inside the
    row (every 2 pt between the frame's inner edges), or -- with ``below`` --
    in the lane ``BELOW_LANE_PT`` under the row.  It never overlaps another x
    or badge, its leader (drawn when it is not level with its x or has slid
    away) never runs through another x or badge, and a badge without a leader
    is clearly nearer its own x than any other (``LABEL_CLEAR_PT``).  Among the
    rest: nearest its x, inside the row before the lane below, then least heat
    under it (``heat(x0, x1, y0, y1)`` in degrees, optional).  ``lines``:
    polylines (strip degrees) a badge should not sit on (the dotted
    connectors, ``miss_connectors``).  Returns
    ``{"labels": {mark index: {"cx", "cy", "label", "slots", "leader": ((x0, y0),
    (x1, y1)) or None, "below": bool}}, "clean": bool}`` -- ``clean`` False when some badge could
    not avoid an overlap or a crossing (then call again with ``below=True``).
    """
    lo, hi = elev_window(elev)
    Ylo, Yhi, W = lo * ppd, hi * ppd, 360.0 * ppd
    half = mark_pt / 2 + 0.8
    bh = MISS_BH
    boxes = [(m["x"] * ppd - half, m["x"] * ppd + half, m["y"] * ppd - half, m["y"] * ppd + half) for m in marks]
    in_lo, in_hi = Ylo + bh + 0.9, Yhi - bh - 0.9
    grid = list(np.arange(in_lo, in_hi + 1e-6, 2.0)) if in_hi >= in_lo else [(Ylo + Yhi) / 2]
    below_y = Ylo - BELOW_LANE_PT

    def gap(b, lo_, hi_, ylo, yhi):
        b0, b1, c0, c1 = b
        return math.hypot(max(0.0, b0 - hi_, lo_ - b1), max(0.0, c0 - yhi, ylo - c1))

    def overlap(b, lo_, hi_, ylo, yhi):
        b0, b1, c0, c1 = b
        return max(0.0, min(hi_, b1) - max(lo_, b0)) * max(0.0, min(yhi, c1) - max(ylo, c0))

    def crosses(p0, p1, skip):
        ts = np.linspace(0.0, 1.0, 20)
        qx, qy = p0[0] + ts * (p1[0] - p0[0]), p0[1] + ts * (p1[1] - p0[1])
        return any(np.any((qx > b0 + 0.3) & (qx < b1 - 0.3) & (qy > c0 + 0.3) & (qy < c1 - 0.3))
                   for j, (b0, b1, c0, c1) in enumerate(boxes) if j != skip)

    line_pts = []
    for ln in lines:
        q = np.asarray(ln, dtype=float) * ppd
        for a_, b_ in zip(q[:-1], q[1:]):
            n_ = max(2, int(np.hypot(*(b_ - a_)) / 1.0))
            line_pts.append(np.linspace(a_, b_, n_))
    line_pts = np.concatenate(line_pts) if line_pts else np.zeros((0, 2))

    def on_line(lo_, hi_, ylo, yhi):
        if not len(line_pts):
            return False
        return bool(np.any((line_pts[:, 0] > lo_ - 0.4) & (line_pts[:, 0] < hi_ + 0.4)
                           & (line_pts[:, 1] > ylo - 0.4) & (line_pts[:, 1] < yhi + 0.4)))

    out, clean = {}, True
    for j_own, m in sorted([(j, m) for j, m in enumerate(marks) if m.get("miss", m.get("slot") is not None)],
                           key=lambda jm: jm[1]["x"]):
        label = m.get("label") or slots_label(m["slots"])
        bw = badge_width_pt(label)
        xp, yp = m["x"] * ppd, m["y"] * ppd
        others = [j for j in range(len(boxes)) if j != j_own]
        lanes = [(float(np.clip(yp, in_lo, in_hi)) if in_hi >= in_lo else grid[0], False)]
        lanes += [(float(y), False) for y in grid]
        if below:
            lanes.append((below_y, True))
        best = None
        for ly, is_below in lanes:
            cands = []
            for shift in np.arange(0.0, 96.0, 1.5):
                for sgn in (1.0, -1.0):
                    a = xp + sgn * (half + 0.6 + shift)
                    lo_, hi_ = (a, a + bw) if sgn > 0 else (a - bw, a)
                    cands.append((lo_, hi_, sgn, shift))
            if abs(ly - yp) >= half + bh + 0.4:  # straight above / below its x
                cands.append((xp - bw / 2, xp + bw / 2, 0.0, 0.0))
            for lo_, hi_, sgn, shift in cands:
                if lo_ < 0.5 or hi_ > W - 0.5:
                    continue
                ylo, yhi = ly - bh, ly + bh
                hard = sum(overlap(boxes[j], lo_, hi_, ylo, yhi) for j in others)
                if sgn == 0.0:
                    hard += overlap(boxes[j_own], lo_, hi_, ylo, yhi)
                cx = (lo_ + hi_) / 2
                d = np.array([cx - xp, ly - yp])
                dist = float(np.hypot(*d))
                u = d / max(dist, 1e-9)
                p0 = (xp + u[0] * half * 0.85, yp + u[1] * half * 0.85)
                rb = bh if bw <= 7.5 else min(bw / 2 / (abs(u[0]) + 1e-9), bh / (abs(u[1]) + 1e-9))
                p1 = (cx - u[0] * (rb + 0.2), ly - u[1] * (rb + 0.2))
                leader = (shift > 0 or abs(ly - yp) > 1.5 or is_below or sgn == 0.0) and math.hypot(
                    p1[0] - p0[0], p1[1] - p0[1]) >= 1.2
                cross = leader and crosses(p0, p1, j_own)
                near = min([gap(boxes[j], lo_, hi_, ylo, yhi) for j in others], default=99.0)
                ambiguous = near < (LABEL_CLEAR_PT if not leader else 2.0)  # nearer another x than its own
                soft = (shift + 0.8 * abs(ly - yp) + (2.0 if leader else 0.0)
                        + 2.0 * max(0.0, LABEL_CLEAR_PT - near) + (8.0 if is_below else 0.0))
                h = heat(lo_ / ppd, hi_ / ppd, ylo / ppd, yhi / ppd) if heat is not None else 0.0
                cost = (hard > 0.01, cross, ambiguous, on_line(lo_, hi_, ylo, yhi), hard, soft, h)
                if best is None or cost < best[0]:
                    best = (cost, cx, ly, p0, p1, leader, is_below)
        cost, cx, ly, p0, p1, leader, is_below = best
        if cost[0] or cost[1]:
            clean = False
        out[j_own] = {"cx": cx / ppd, "cy": ly / ppd, "below": bool(is_below), "label": label,
                      "slots": list(m["slots"]),
                      "leader": ((p0[0] / ppd, p0[1] / ppd), (p1[0] / ppd, p1[1] / ppd)) if leader else None}
        boxes.append((cx - bw / 2, cx + bw / 2, ly - bh, ly + bh))
    return {"labels": out, "clean": clean}


def heat_lookup(strip: np.ndarray, elev):
    """``heat(x0, x1, y0, y1)`` (strip degrees) summing ``strip`` under a box, for ``place_miss_labels``."""
    lo, hi = elev_window(elev)
    h, w = strip.shape[:2]

    def heat(x0, x1, y0, y1):
        c0, c1 = int(max(0, x0) * w / 360.0), int(min(360, x1) * w / 360.0)
        r0, r1 = int((hi - min(hi, y1)) * h / (hi - lo)), int((hi - max(lo, y0)) * h / (hi - lo))
        return float(strip[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)].sum())

    return heat


def draw_peak_marks(ax, marks: Sequence[dict], size: float = MARK_PT) -> None:
    for m in marks:
        peak_mark(ax, m["x"], m["y"], size=size)


CONNECTOR_MAX_DEG = 45.0  # a miss farther than this from its true bearing gets no dotted connector


def miss_connectors(row, marks: Sequence[dict], elev, ppd: float, mark_pt: float = MARK_PT,
                    max_deg: float = CONNECTOR_MAX_DEG) -> List[list]:
    """Per numbered miss: [(x0, y0), (x1, y1)] in strip degrees, from its true-bearing tick (the row's bottom
    edge) to the edge of its own x -- none when the x sits on the tick, or lies more than ``max_deg`` along the
    strip from it (a line across whole views would cross other marks; the numbers still pair the x with its
    tick and lane badge)."""
    lo, _ = elev_window(elev)
    out = []
    for m in marks:
        if not m.get("miss", m.get("slot") is not None):
            continue
        for k in m["slots"]:
            x0, y0 = float(strip_x(row.gt_bearing[k])), lo
            if abs(m["x"] - x0) > max_deg:
                continue
            d = np.array([(m["x"] - x0) * ppd, (m["y"] - y0) * ppd])
            n = float(np.hypot(*d))
            if n < mark_pt / 2 + 1.5:
                continue
            end = np.array([m["x"], m["y"]]) - d / n * (mark_pt / 2 + 0.9) / ppd
            out.append([(x0, y0), (float(end[0]), float(end[1]))])
    return out


def draw_miss_connectors(ax, row, marks: Sequence[dict], elev, ppd: float, mark_pt: float = MARK_PT) -> None:
    """Thin dotted line from each miss's true-bearing tick (the row's bottom edge) to its own x (D2), when they
    lie within ``CONNECTOR_MAX_DEG`` of each other.

    Drawn over the x marks (under badges and leaders), so a connector that
    passes another x visibly runs on to its own."""
    for (x0, y0), (x1, y1) in miss_connectors(row, marks, elev, ppd, mark_pt):
        ax.plot([x0, x1], [y0, y1], color=style.INK_2, lw=0.55, ls=(0, (0.9, 1.1)), zorder=7.2,
                clip_on=False, solid_capstyle="round", dash_capstyle="round")


def draw_miss_labels(ax, placed: dict) -> List[int]:
    """Leaders (ink, white halo) and ``miss_badge``s of ``place_miss_labels``; returns the slots numbered."""
    done = []
    for _, lab in sorted(placed["labels"].items()):
        if lab["leader"] is not None:
            (x0, y0), (x1, y1) = lab["leader"]
            ax.plot([x0, x1], [y0, y1], color=LEADER_COLOR, lw=0.6, zorder=7.6, clip_on=False, solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        miss_badge(ax, lab["cx"], lab["cy"], lab["label"], zorder=8)
        done += [int(k) for k in lab["slots"]]
    return done
