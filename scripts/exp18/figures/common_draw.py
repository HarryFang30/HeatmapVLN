"""Drawing primitives shared by the EXP-18 figures (case, gallery, route figures).

Every primitive takes a matplotlib ``Axes`` and draws in its data coordinates;
sizes of glyphs (badges, arrows, markers) are given in points so they print at
the same size whatever the map scale.  Conventions come from
``scripts/exp18/geometry.py`` (bearings) and ``topdown_io.py`` (maps); nothing
is re-derived here.

Visual vocabulary (one meaning per encoding, used by every EXP-18 figure):

* **Blue = ground truth.**  Numbered badges 1..8 are the K = 8 past camera
  positions (1 = oldest), filled with ``style.history_color`` (light = older);
  the number, not the shade, identifies a slot.  The ground-truth heat row is a
  blue sequential ramp (``GT_CMAP``).
* **Orange = prediction.**  ``PRED_CMAP`` (= ``style.HEAT_CMAP_OPAQUE``) for the
  predicted heat row; a black x with a white halo marks a predicted peak.
* **Grey = context.**  Maps and the three surround views the model never sees
  are desaturated and lightened; only the front view keeps its colour.
* **Black = the robot** (triangle pointing along the heading) and key positions
  (``K1`` badges with a heading arrow).

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
              color: str = style.INK, zorder: float = 9) -> None:
    """Horizontal bar of ``length`` data units starting at x (``ha='right'``: ending at x), label above."""
    x0 = x if ha == "left" else x - length
    ax.plot([x0, x0 + length], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=zorder,
            path_effects=HALO_THIN)
    ax.annotate(label, (x0 + length / 2, y), xytext=(0, 1.6), textcoords="offset points", ha="center",
                va="bottom", fontsize=fs, color=color, path_effects=HALO, zorder=zorder)


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
    order = np.argsort(t)
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
                    letters=("F", "R", "B", "L"), sat: float = 0.25, white: float = 0.4, out_px: int = 640):
    """Round heading-up map around the robot; its rim is the bearing ring the strip unrolls.

    The axes spans +-``half_m / radius_frac`` metres; the disc (radius
    ``half_m``) holds the map, the route so far, and dashed view seams at
    bearings +-45 / +-135 deg.  Returns the :class:`EgoCrop` (world ->
    heading-up metres via ``world_to_local``).  Sector letters and badges go
    outside the disc (``disc_rim_labels``).
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
    if past_xz is not None and len(past_xz) > 1:
        a, b = crop.world_to_local(past_xz[:, 0], past_xz[:, 1])
        line, = ax.plot(a, b, color=style.INK_2, lw=0.8, alpha=0.75, solid_capstyle="round", zorder=2)
        line.set_clip_path(Circle((0, 0), half_m, transform=ax.transData))
    for b in (45.0, -45.0, 135.0, -135.0):
        x, y = bearing_to_xy(b, half_m)
        ax.plot([0, x], [0, y], color=style.INK_2, lw=0.45, ls=(0, (2.0, 1.6)), alpha=0.8, zorder=1.5)
    ax.add_patch(Circle((0, 0), half_m, fill=False, ec=style.MUTED, lw=0.6, zorder=3))
    return crop


def disc_rim_labels(ax, half_m: float, groups: Sequence[Sequence[int]], bearings: Sequence[float],
                    num: int = 8, gap_pt: float = 1.0, offset_pt: float = 5.2) -> List[Tuple[float, float]]:
    """Numbered badges just outside the disc rim at each group's bearing, dodged along the rim.

    Returns ``(theta, half_width)`` in degrees (plot angle, 0 = +x, counter-
    clockwise) of every placed badge, for ``disc_sector_letters`` to avoid.  A
    thin leader joins a dodged badge to its true bearing on the rim.
    """
    if not groups:
        return []
    per_pt = pts_to_data(ax, 1.0)[0]
    r_pt = half_m / per_pt  # disc radius in points
    ring_pt = r_pt + offset_pt
    labels = [f"{g[0] + 1}" if len(g) == 1 else f"{g[0] + 1}–{g[-1] + 1}" for g in groups]
    widths = np.array([math.degrees(badge_width_pt(lab) / ring_pt) for lab in labels])
    theta = np.array([90.0 + b for b in bearings])  # heading-up: bearing 0 = up, left-positive = counter-clockwise
    placed = dodge_1d(theta, widths, 0.0, 360.0, math.degrees(gap_pt / ring_pt), periodic=True)
    out = []
    for g, lab, t, p, w in zip(groups, labels, theta, placed, widths):
        k = g[0]
        u_t = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
        u_p = np.array([math.cos(math.radians(p)), math.sin(math.radians(p))])
        c = u_p * ring_pt * per_pt
        if abs(((p - t + 180) % 360) - 180) > 0.8:
            rim, mid = u_t * half_m, u_t * (r_pt + 1.8) * per_pt
            ax.plot([rim[0], mid[0], c[0]], [rim[1], mid[1], c[1]], color=history_line_color(k, num), lw=0.5,
                    zorder=5, solid_joinstyle="round", clip_on=False)
        history_badge(ax, c[0], c[1], lab, k, num=num, zorder=6)
        out.append((float(p % 360.0), float(w / 2)))
    return out


def disc_sector_letters(ax, half_m: float, occupied: Sequence[Tuple[float, float]] = (), offset_pt: float = 5.2,
                        names=("F", "R", "B", "L"), fs: float = 6.0, letter_pt: float = 4.6) -> None:
    """Sector letters at the sector centres on the badge ring outside the rim.

    ``occupied``: ``(theta, half_width)`` degrees from ``disc_rim_labels``;
    where a badge takes the letter's place the letter moves radially outward,
    past the badges, so it always marks the centre of its sector.
    """
    per_pt = pts_to_data(ax, 1.0)[0]
    ring_pt = half_m / per_pt + offset_pt
    half_letter = math.degrees((letter_pt / 2 + 0.8) / ring_pt)
    for name, centre_bearing in zip(names, geo.VIEW_YAWS_DEG):
        theta = 90.0 + centre_bearing
        free = all(abs((theta - t + 180) % 360 - 180) > w + half_letter for t, w in occupied)
        radius = ring_pt if free else ring_pt + 3.7 + letter_pt / 2 + 1.0
        x = radius * per_pt * math.cos(math.radians(theta))
        y = radius * per_pt * math.sin(math.radians(theta))
        ax.text(x, y, name, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=style.INK if name == names[0] else style.MUTED, zorder=6, path_effects=HALO_THIN)


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


def rgb_strip(views: np.ndarray, width: int, elev: float, observed=(geo.FRONT,), sat: float = 0.12,
              white: float = 0.5) -> np.ndarray:
    """[h, width, 3] float strip of the four views; views not in ``observed`` are muted."""
    h = int(round(width * 2 * elev / 360.0))
    ring, _ = geo.stitch_ring_rgb(views, width=width, height=h, elev_top=elev, elev_bottom=-elev)
    strip = ring_to_strip(ring).astype(np.float32) / 255.0
    out = mute(strip, sat=sat, white=white)
    q = width // 4
    for v in observed:  # strip panels are F, R, B, L in view order
        out[:, v * q:(v + 1) * q] = strip[:, v * q:(v + 1) * q]
    return out


def heat_strip(maps: np.ndarray, width: int, elev: float) -> np.ndarray:
    """[h, width] strip of four 64x64 label-convention maps (0 outside every view)."""
    h = int(round(width * 2 * elev / 360.0))
    ring, _ = geo.stitch_ring(maps, width=width, height=h, elev_top=elev, elev_bottom=-elev, fill=0.0)
    return ring_to_strip(np.clip(ring, 0.0, 1.0))


def setup_strip_axes(ax, elev: float) -> None:
    ax.set_xlim(0, 360)
    ax.set_ylim(-elev, elev)
    ax.set_autoscale_on(False)
    clean_axes(ax)


def draw_rgb_row(ax, strip: np.ndarray, elev: float, frame_views=(geo.FRONT,)) -> None:
    ax.imshow(strip, extent=(0, 360, -elev, elev), aspect="auto", interpolation="bilinear", zorder=0)
    setup_strip_axes(ax, elev)
    for s in (90, 180, 270):
        ax.axvline(s, color="white", lw=0.9, zorder=2)
    for v in frame_views:
        ax.add_patch(Rectangle((v * 90, -elev), 90, 2 * elev, fill=False, ec=style.INK, lw=1.0, zorder=5,
                               clip_on=False))


def draw_heat_row(ax, strip: np.ndarray, elev: float, cmap, top: float = HEAT_TOP) -> None:
    """Heat strip coloured linearly by value (0..1 -> the first ``top`` of ``cmap``) on a framed row."""
    ax.imshow(cmap(top * np.clip(strip, 0, 1)), extent=(0, 360, -elev, elev), aspect="auto",
              interpolation="bilinear", zorder=0)
    setup_strip_axes(ax, elev)
    for s in (90, 180, 270):
        ax.axvline(s, color=style.GRID, lw=0.6, zorder=1)
    ax.add_patch(Rectangle((0, -elev), 360, 2 * elev, fill=False, ec=style.AXIS, lw=0.5, zorder=4, clip_on=False))


def quietest_panel(*strips: np.ndarray, empty: float = 0.05) -> int:
    """Strip panel for in-row labels: the first (F, R, B, L order) empty in every strip, else the quietest."""
    q = strips[0].shape[1] // 4
    peaks = [max(float(s[:, v * q:(v + 1) * q].max()) for s in strips) for v in range(4)]
    for v in range(4):
        if peaks[v] < empty:
            return v
    return int(np.argmin(peaks))


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
