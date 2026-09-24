"""Drawing primitives of the EXP-19 behaviour figure, on top of the EXP-18 toolkit.

Everything shared with EXP-18 is imported, not copied: fonts, palette and
glyphs come from ``scripts/exp18/figures/{style,common_draw}.py``, ring
geometry from ``scripts/exp18/geometry.py`` and maps from
``scripts/exp18/topdown/topdown_io.py``.  This module adds only what EXP-19 needs.

**Colour roles.**  Each colour has one meaning, and orange and blue mean the
same as in EXP-18.

* **Orange** is the predicted history affordance map.  It uses
  ``style.HEAT_CMAP``, a ramp from transparent to orange, laid over imagery.
* **Blue** is the ground truth: the true directions of the past frames, drawn
  as open circles in palette blue ``#2a78d6``.  The history-slot badges use the
  EXP-18 blue slot ramp, because they name the same past frames.
* **Aqua** is the predicted future affordance map.  It is a four-step ordinal
  ramp, one step per time bin, where darker means later: ``FUTURE_BIN_COLORS``.
  The ramp passes the dataviz validator (``--ordinal --mode light``: monotone,
  every adjacent dL >= 0.06, light end at 2.10:1 on #fcfcfb, hue spread 8 deg).
  The trio blue / orange / aqua (#2a78d6, #eb6834, #1baf7a) passes all-pairs:
  worst CVD dE 9.2, normal-vision dE 24.0.  Aqua sits at 2.74:1 contrast, so
  every aqua mark is also labelled in the legend and on its strip.
* **Ink** marks the decisions: the System1 mean path (dots with a white rim),
  the System2 pixel goal (ring with a white halo), action chips, route and
  key-moment badges.
* **Grey** is context: the muted map, and the surroundings outside the
  deployed camera's view, which are never given to the model.

**The 360-degree strips** are the heading-centred ring of ``geo.stitch_ring``,
not rolled.  The x axis is bearing, running from +180 at the left edge (behind,
turning left) through +90 (left) and 0 (straight ahead, centre) to -90 (right)
and -180 at the right edge.  Every mark is plotted at its bearing, with
``xlim = (180, -180)``.  Rows are elevation in square degrees, in a fixed band
(the history strip +-15 deg, the future strip +-9 deg).  A direction outside
the band (stairs) is drawn as a triangle on the band's edge, pointing up or
down, never squashed onto the edge as if it were there.

The re-rendered views are HFOV 90, like the label grid.  The deployed camera
sees HFOV 79, so only the framed +-39.5 deg of the front sector is in colour:
that is the part of the strip the model was given (re-rendered at the recorded
position, not the image it received).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from scripts.exp18 import geometry as geo
from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp19.figures import bundle as bd

from matplotlib import patheffects as pe  # noqa: E402  (cd.setup selects the backend and fonts)
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

GT_COLOR = "#2a78d6"  # palette blue: ground truth
HIST_CMAP = style.HEAT_CMAP  # transparent -> orange, laid over the muted surroundings
FUTURE_BIN_COLORS = ("#4cc594", "#1caf7a", "#008d5a", "#006b3c")  # aqua, light = waypoints 1-8 ... dark = 25-32
FUTURE_ALPHA = 0.92
CHIP_FILL = "#f0efec"
MODEL_VIEW_HALF_DEG = 39.5  # deployed front camera: 640x480, HFOV 79 (the re-renders' F view is HFOV 90)
MIN_FS = 5.0  # smallest text on a page printed at 100 %
SECTORS = ((90.0, 1), (0.0, 0), (-90.0, 2), (157.5, 3), (-157.5, 3))  # (bearing, index into the sector names)


# --------------------------------------------------------------------------- #
# Rings
# --------------------------------------------------------------------------- #
def ring_height(width: int, elev: float) -> int:
    return int(round(width * 2 * elev / 360.0))


def front_columns(width: int) -> np.ndarray:
    """Ring columns inside the deployed camera's horizontal field of view."""
    az = geo.ring_column_azimuths(width)
    return (az > -MODEL_VIEW_HALF_DEG) & (az <= MODEL_VIEW_HALF_DEG)


def ring_rgb(views: np.ndarray, width: int, elev: float, sat: float = 0.12, white: float = 0.5) -> np.ndarray:
    """[h, width, 3] float ring of the four re-rendered views: the camera's view in colour, the rest muted."""
    ring, _ = geo.stitch_ring_rgb(views, width=width, height=ring_height(width, elev), elev_top=elev, elev_bottom=-elev)
    full = ring.astype(np.float32) / 255.0
    out = cd.mute(full, sat=sat, white=white)
    front = front_columns(width)
    out[:, front] = full[:, front]
    return out


def ring_heat(maps: np.ndarray, width: int, elev: float) -> np.ndarray:
    """[h, width] ring of four 64x64 label-convention maps, 0 outside every view."""
    ring, _ = geo.stitch_ring(maps, width=width, height=ring_height(width, elev), elev_top=elev,
                              elev_bottom=-elev, fill=0.0)
    return np.clip(ring, 0.0, 1.0)


def setup_ring_axes(ax, elev: float) -> None:
    ax.set_xlim(180.0, -180.0)
    ax.set_ylim(-elev, elev)
    ax.set_autoscale_on(False)
    cd.clean_axes(ax)


def show_ring(ax, img: np.ndarray, elev: float, zorder: float = 0) -> None:
    ax.imshow(img, extent=geo.ring_extent(elev, -elev), aspect="auto", interpolation="bilinear", zorder=zorder)


def seams(ax, elev: float, color, lw: float) -> None:
    for b in (135.0, 45.0, -45.0, -135.0):
        ax.plot([b, b], [-elev, elev], color=color, lw=lw, zorder=2, solid_capstyle="butt")


def front_frame(ax, elev: float, lw: float = 1.0) -> None:
    """Frame around the deployed camera's view (its vertical field, +-31.7 deg, covers the whole band)."""
    ax.add_patch(Rectangle((-MODEL_VIEW_HALF_DEG, -elev), 2 * MODEL_VIEW_HALF_DEG, 2 * elev, fill=False, ec=style.INK,
                           lw=lw, zorder=5, clip_on=False))


def sector_letters(ax, elev: float, names: Sequence[str], fs: float = 5.6,
                   notes: Optional[Sequence[str]] = None) -> None:
    """Sector letters at the top of the strip.  The front letter is ink and the others muted; back appears at both ends.

    ``notes`` = (front note, side note): short italic tags after the letters, e.g.
    "model's view" after F and "display only" after L and R.  A note that would
    run past the camera frame (F) or the next seam (L, R) on a narrow strip is
    left out; the figure's strip note says the same in full.
    """
    y = elev - cd.pts_to_data(ax, 0.0, fs * 0.72)[1]
    gap = cd.pts_to_data(ax, fs * 0.45, 0.0)[0]
    ends = {0: -MODEL_VIEW_HALF_DEG, 1: 45.0, 2: -135.0}  # where each note must end (xlim runs +180 -> -180)
    for bearing, idx in SECTORS:
        ax.text(bearing, y, names[idx], ha="center", va="center", fontsize=fs, fontweight="bold",
                color=style.INK if idx == 0 else style.INK_2, zorder=6, path_effects=cd.HALO_THIN)
        if notes and idx in ends:  # F, L, R; the back sector is split across the two ends
            note = notes[0 if idx == 0 else 1]
            half = cd.pts_to_data(ax, cd.text_width_pt(ax.figure, names[idx], fs, fontweight="bold") / 2, 0.0)[0]
            width = cd.pts_to_data(ax, cd.text_width_pt(ax.figure, note, fs - 0.3, fontstyle="italic") + 2.0, 0.0)[0]
            if bearing + half + gap + width < ends[idx]:
                continue
            ax.text(bearing + half + gap, y, note, ha="left", va="center", fontsize=fs - 0.3, fontstyle="italic",
                    color=style.INK if idx == 0 else style.INK_2, zorder=6, path_effects=cd.HALO_THIN)


def row_label(ax, start: float, y: float, text: str, fs: float) -> None:
    """Row label from the start bearing of a quiet sector, shifted left when it would run past the -180 edge.

    x is placed in axes fractions (xlim runs +180 -> -180), so the label's
    rendered width can be kept inside the strip at either end.
    """
    fs = max(fs, MIN_FS)
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    w = cd.text_width_pt(ax.figure, text, fs) / width_pt
    pad = 2.5 / width_pt
    x = min(max((180.0 - start) / 360.0 + pad, pad), 1.0 - pad - w)
    ax.text(x, y, text, transform=blended_transform_factory(ax.transAxes, ax.transData), ha="left", va="center",
            fontsize=fs, color=style.INK_2, zorder=6, path_effects=cd.HALO)


def quietest_sector(*rings: np.ndarray) -> float:
    """Start bearing of the side sector (L, R, back-left, back-right) with the least heat: where a row label goes."""
    width = rings[0].shape[1]
    az = geo.ring_column_azimuths(width)
    spans = {135.0: (az > 45) & (az <= 135), -45.0: (az > -135) & (az <= -45), 180.0: az > 135, -135.0: az <= -135}
    heat = {start: max(float(r[:, cols].max()) if cols.any() else 0.0 for r in rings) for start, cols in spans.items()}
    return min(heat, key=lambda s: (round(heat[s], 2), -s))


def pixel_to_ring(view: int, row: float, col: float) -> Tuple[float, float]:
    """(bearing, elevation) of a label-grid pixel."""
    b, e = geo.pixel_to_bearing_elev(view, col, row)
    return float(b), float(e)


# --------------------------------------------------------------------------- #
# History strip: muted surroundings + predicted history affordance map + GT circles
# --------------------------------------------------------------------------- #
def _both_ends(bearing: float) -> Tuple[float, float]:
    """A bearing and its copy one turn away: a mark at the +-180 deg edge shows its halves at both ends."""
    return bearing, (bearing - 360.0 if bearing > 0 else bearing + 360.0)


def gt_circle(ax, bearing: float, elev: float, ms: float = 4.6) -> None:
    """True direction of a past frame: open blue circle with a white halo, clipped to the strip."""
    for b in _both_ends(bearing):
        ax.plot([b], [elev], marker="o", ms=ms, mfc="none", mec=GT_COLOR, mew=1.05, zorder=7, clip_on=True,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def edge_mark(ax, bearing: float, up: bool, band: float, kind: str, ms: float = 4.0) -> None:
    """A direction beyond the band: a triangle on the top (up) or bottom edge pointing out of the band.

    ``kind`` "gt" = open blue (a true past direction), "pred" = filled orange (a visible slot's predicted peak).
    """
    y = band - cd.pts_to_data(ax, 0.0, 2.6)[1] if up else -band + cd.pts_to_data(ax, 0.0, 2.6)[1]
    style_kw = (dict(mfc="white", mec=GT_COLOR, mew=1.0) if kind == "gt" else
                dict(mfc=HIST_CMAP(0.95), mec="white", mew=0.5))
    for b in _both_ends(bearing):
        ax.plot([b], [y], marker="^" if up else "v", ms=ms, zorder=7.5, clip_on=True, **style_kw)


def history_marks(gt_peaks, pred_peaks, band: float) -> list:
    """(kind, bearing, elevation, inside the band) of the marks on a history strip.

    Every true past direction is a mark; a visible slot's predicted peak is a
    mark only when it lies beyond the band (inside it, the orange map shows it).
    """
    out = []
    for kind, peaks in (("gt", gt_peaks), ("pred", pred_peaks)):
        for _, view, row, col in peaks:
            b, e = pixel_to_ring(view, row, col)
            if kind == "gt" or abs(e) > band:
                out.append((kind, b, e, abs(e) <= band))
    return out


def draw_history_strip(ax, views: np.ndarray, composite: np.ndarray, gt_peaks, pred_peaks, elev: float,
                       rgb_width: int, heat_width: int, sector_names: Sequence[str], label: Optional[str] = None,
                       fs_label: float = 5.6, sector_notes: Optional[Sequence[str]] = None) -> None:
    rgb = ring_rgb(views, rgb_width, elev)
    heat = ring_heat(composite, heat_width, elev)
    show_ring(ax, rgb, elev, zorder=0)
    show_ring(ax, HIST_CMAP(heat), elev, zorder=1)
    setup_ring_axes(ax, elev)
    seams(ax, elev, "white", 0.9)
    front_frame(ax, elev)
    for kind, b, e, inside in history_marks(gt_peaks, pred_peaks, elev):
        if inside:
            gt_circle(ax, b, e)
        else:
            edge_mark(ax, b, e > 0, elev, kind)
    sector_letters(ax, elev, sector_names, notes=sector_notes)
    if label:
        row_label(ax, quietest_sector(heat), -elev + cd.pts_to_data(ax, 0, 3.6)[1], label, fs_label)


# --------------------------------------------------------------------------- #
# Future strip: four time bins in one aqua ramp + System1 path
# --------------------------------------------------------------------------- #
def future_ring_rgb(bins: np.ndarray, width: int, elev: float, surface=style.SURFACE,
                    colors: Sequence[str] = FUTURE_BIN_COLORS, alpha: float = FUTURE_ALPHA) -> np.ndarray:
    """[h, width, 3]: bins painted early -> late over the plate, opacity = map value (later bins on top)."""
    out = np.ones((ring_height(width, elev), width, 3), dtype=np.float64) * np.asarray(to_rgb(surface))
    for b in range(bins.shape[0]):
        a = alpha * ring_heat(bins[b], width, elev)[..., None]
        out = out * (1.0 - a) + np.asarray(to_rgb(colors[b])) * a
    return out


def draw_future_strip(ax, bins: np.ndarray, path_cam: np.ndarray, elev: float, width: int,
                      label: Optional[str] = None, fs_label: float = 5.6) -> None:
    img = future_ring_rgb(bins, width, elev)
    show_ring(ax, img, elev)
    setup_ring_axes(ax, elev)
    seams(ax, elev, style.GRID, 0.6)
    ax.add_patch(Rectangle((180.0, -elev), -360.0, 2 * elev, fill=False, ec=style.AXIS, lw=0.5, zorder=4,
                           clip_on=False))
    strip_path(ax, path_cam, elev)
    if label:
        row_label(ax, quietest_sector(ring_heat(bins.max(0), width, elev)), 0.0, label, fs_label)


def strip_path(ax, path_cam: np.ndarray, elev: float) -> None:
    """System1 mean path on a strip at its bearings (camera-height placement of the future labels).

    Ink dots on a white underlay line, no rim per dot: a straight path piles its
    33 waypoints near 0 deg, and rimmed dots would fray into rings there.  The
    future labels put every waypoint at camera height, so the path is on the
    horizon row and never leaves the band.
    """
    bearing, e, _ = bd.path_directions(path_cam)
    if not bearing.size:
        return
    e = np.clip(e, -elev, elev)
    breaks = np.nonzero(np.abs(np.diff(bearing)) > 180.0)[0] + 1  # never draw across the +-180 seam
    for bs, es in zip(np.split(bearing, breaks), np.split(e, breaks)):
        ax.plot(bs, es, color="white", lw=2.6, solid_capstyle="round", zorder=5.5)
        ax.plot(bs, es, ls="none", marker="o", ms=1.9, mfc=style.INK, mec="none", zorder=6)


def ring_axis(ax, labels: Sequence[str], fs: float = 5.8) -> None:
    """Bearing labels under a strip: +180 (behind), +90 (left), 0 (ahead), -90 (right), -180 (behind)."""
    ax.set_xticks([180.0, 90.0, 0.0, -90.0, -180.0])
    ax.set_xticklabels(labels, fontsize=fs, color=style.INK_2)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.tick_params(axis="x", which="major", length=2.4, width=0.5, color=style.AXIS, pad=2.0)


# --------------------------------------------------------------------------- #
# Images: decision image, filmstrip frames
# --------------------------------------------------------------------------- #
def draw_image(ax, img: np.ndarray, frame_color=style.AXIS, frame_lw: float = 0.5) -> Tuple[int, int]:
    """Image filling the axes box (display aspect set by the axes; pixel (u, v) centres at data (u, v))."""
    h, w = img.shape[:2]
    ax.imshow(img, extent=(-0.5, w - 0.5, h - 0.5, -0.5), aspect="auto", interpolation="bilinear", zorder=0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_autoscale_on(False)
    cd.clean_axes(ax, spines=True, color=frame_color, lw=frame_lw)
    return h, w


def goal_marker(ax, u: float, v: float, size: float = 7.0, zorder: float = 8, clip_on: bool = False):
    """System2 pixel goal: an ink ring with a white halo and a centre dot."""
    ax.plot([u], [v], marker="o", ms=size, mfc="none", mec=style.INK, mew=1.15, zorder=zorder, clip_on=clip_on,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.plot([u], [v], marker="o", ms=1.5, mfc=style.INK, mec="none", zorder=zorder, clip_on=clip_on)


def path_dots(ax, uv: np.ndarray, ms: float = 2.1, zorder: float = 7, **kw):
    """System1 mean path: ink dots with a thin white rim."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(uv):
        ax.plot(uv[:, 0], uv[:, 1], ls="none", marker="o", ms=ms, mfc=style.INK, mec="white", mew=0.4,
                zorder=zorder, **kw)


def draw_decision_image(ax, img: np.ndarray, goal_uv, path_uv) -> None:
    """Decision image, System1 path dots on it, and the System2 pixel goal when it lies on the image."""
    h, w = draw_image(ax, img, frame_color=style.INK, frame_lw=0.8)
    uv = np.asarray(path_uv, dtype=np.float64).reshape(-1, 2)
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)
    path_dots(ax, uv[inside])
    if goal_uv is not None and bd.goal_inside(goal_uv, img.shape):  # outside = a convention bug, flagged by [F]
        goal_marker(ax, float(goal_uv[0]), float(goal_uv[1]), clip_on=True)  # a goal at the edge stays on the image


# --------------------------------------------------------------------------- #
# Action chips (axes in point units)
# --------------------------------------------------------------------------- #
def stop_fs(size: float) -> float:
    return max(size * 0.62, MIN_FS)


def chip_width(action: int, size: float) -> float:
    return max(size * 2.3, stop_fs(size) * 3.1) if action == bd.STOP else size


def action_chip(ax, x: float, y: float, action: int, size: float = 7.6, zorder: float = 6) -> float:
    """One executed action at (x = left edge, y = centre) in points: forward/turn arrow or STOP.  Returns its width."""
    w = chip_width(action, size)
    if action == bd.STOP:
        ax.add_patch(FancyBboxPatch((x, y - size / 2), w, size, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=style.INK, ec="none", zorder=zorder))
        ax.text(x + w / 2, y, "STOP", ha="center", va="center", fontsize=stop_fs(size), fontweight="bold",
                color="white", zorder=zorder + 1)
        return w
    ax.add_patch(FancyBboxPatch((x, y - size / 2), w, size, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=CHIP_FILL, ec=style.AXIS, lw=0.5, zorder=zorder))
    c = np.array([x + w / 2, y])
    d = {bd.FORWARD: (0.0, 1.0), bd.LEFT: (-1.0, 0.0), bd.RIGHT: (1.0, 0.0)}[int(action)]
    d = np.asarray(d) * size * 0.34
    ax.add_patch(FancyArrowPatch(tuple(c - d), tuple(c + d), arrowstyle="-|>", mutation_scale=size * 0.62,
                                 color=style.INK, lw=0.85, shrinkA=0, shrinkB=0, zorder=zorder + 1))
    return w


def action_chips(ax, x: float, y: float, actions: Sequence[int], size: float = 7.6, gap: float = 1.6,
                 align: str = "left") -> float:
    """Chips of an action chunk in point units; ``align='right'`` puts its right edge at x.  Returns the width."""
    widths = [chip_width(a, size) for a in actions]
    total = sum(widths) + gap * max(len(widths) - 1, 0)
    x0 = x - total if align == "right" else x
    for a, w in zip(actions, widths):
        action_chip(ax, x0, y, int(a), size=size)
        x0 += w + gap
    return total


# --------------------------------------------------------------------------- #
# Top-down route map
# --------------------------------------------------------------------------- #
def map_limits(xz_sets: Sequence[np.ndarray], goal_xz, goal_radius: float, aspect_hw: float, pad: float = 0.6):
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in xz_sets if len(p)]
    g = np.asarray(goal_xz, dtype=np.float64)
    pts.append(g + goal_radius * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]))
    return cd.fit_limits(np.concatenate(pts), pad=pad, aspect_hw=aspect_hw)


BADGE_HALF_PT = (8.0, 5.0)  # half width / height of a "K1" badge at 5.8 pt, with a little air


def _badge_spot(p: np.ndarray, points: np.ndarray, boxes: np.ndarray, limits, per_pt: float) -> np.ndarray:
    """Centre of a key badge near p (leader 11-23 pt) with the most room, in points.

    ``points`` are marks the badge must not cover (route, reference path, goal,
    key dots); ``boxes`` are centres of badges and labels already placed, which it
    must clear by a whole badge.  A longer leader costs a little; leaving the map
    costs everything.  Clustered key positions (short or wandering routes) fan out.
    """
    hw, hh = BADGE_HALF_PT
    best, best_score = p, -np.inf
    for dist in (11.0, 17.0, 23.0):
        for ang in np.radians(np.arange(0.0, 360.0, 30.0) + 15.0):
            q = p + np.array([math.cos(ang), math.sin(ang)]) * dist * per_pt
            inside = (limits[0] + hw * per_pt < q[0] < limits[1] - hw * per_pt
                      and limits[2] + hh * per_pt < q[1] < limits[3] - hh * per_pt)
            room = float(np.min(np.linalg.norm(points - q, axis=1))) / per_pt
            if len(boxes):  # separation of two badge-sized boxes (< 0: they overlap)
                gap = np.abs(np.asarray(boxes) - q) / per_pt - np.array([2 * hw, 2 * hh])
                room = min(room, 6.0 + float(np.min(np.max(gap, axis=1))))
            score = room - 0.15 * dist - (0.0 if inside else 1e3)
            if score > best_score:
                best, best_score = q, score
    return best


def _scale_bar_spot(limits, per_pt: float, length: float, taken: np.ndarray) -> Tuple[float, float]:
    """Left end of the scale bar: the map corner farthest from every mark (bottom-left on ties)."""
    x0, x1, z0, z1 = limits  # z1 is the bottom edge (+z points down on the map)
    m, label_h = 5.0 * per_pt, 9.0 * per_pt
    best, best_d = None, -1.0
    for x, z in ((x0 + m, z1 - m), (x1 - m - length, z1 - m), (x0 + m, z0 + m + label_h),
                 (x1 - m - length, z0 + m + label_h)):
        lo, hi = np.array([x, z - label_h]), np.array([x + length, z])  # bar with its label above
        d = float(np.linalg.norm(np.maximum(np.maximum(lo - taken, taken - hi), 0.0), axis=1).min())
        if d > best_d + 1e-9:
            best, best_d = (x, z), d
    return best


def draw_plate(ax, limits) -> None:
    """Empty map plate (no top-down map for the scene), same framing as ``cd.draw_topdown``."""
    x0, x1, z0, z1 = limits
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    ax.set_autoscale_on(False)
    ax.set_facecolor(cd.MAP_PLATE)


def draw_route_line(ax, route_xz: np.ndarray, off_level: Optional[np.ndarray], start_label: str) -> None:
    """Executed route in ink; steps on another floor than the map's are dotted grey (they are not on this map)."""
    cd.draw_route(ax, route_xz, color=style.INK, lw=0.95, start_label=start_label, zorder=3)
    if off_level is None or not np.any(off_level):
        return
    off = np.asarray(off_level, dtype=bool)
    seg = off[1:] | off[:-1]  # segment j (steps j -> j+1) touches another floor
    edges = np.diff(np.concatenate([[0], seg.astype(np.int8), [0]]))
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):  # segments a..b-1 = steps a..b
        pts = route_xz[a:b + 1]
        ax.plot(pts[:, 0], pts[:, 1], color=cd.MAP_PLATE, lw=1.6, zorder=3.2, solid_capstyle="butt")
        ax.plot(pts[:, 0], pts[:, 1], color=style.MUTED, lw=0.9, ls=(0, (1.0, 1.4)), zorder=3.3,
                dash_capstyle="round")


def draw_route_map(ax, level, route_xz, ref_xz, start_xz, goal_xz, goal_radius: float, key_xz, key_labels,
                   start_label: str, radius_label: str, off_level: Optional[np.ndarray] = None) -> None:
    """Muted top-down map, reference path (grey dashed), executed route (ink), start, goal + success radius, K badges.

    No heading arrows and nothing derived from a pose estimate: every mark is a
    recorded simulator position.  ``level`` None draws an empty plate (no map for
    the scene); ``off_level`` marks route steps on another floor than ``level``.
    """
    route_xz = np.asarray(route_xz, dtype=np.float64).reshape(-1, 2)
    ref_xz = np.asarray(ref_xz, dtype=np.float64).reshape(-1, 2)
    key_xz = np.asarray(key_xz, dtype=np.float64).reshape(-1, 2)
    limits = map_limits([route_xz, ref_xz], goal_xz, goal_radius, cd.axes_aspect_hw(ax), pad=1.0)
    if level is None:
        draw_plate(ax, limits)
    else:
        cd.draw_topdown(ax, level, limits, sat=0.2, white=0.56)
    cd.clean_axes(ax, spines=True)
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    gx, gz = float(goal_xz[0]), float(goal_xz[1])
    ax.add_patch(Circle((gx, gz), goal_radius, fc=cd.mix(style.INK, "white", 0.93), ec=style.INK_2, lw=0.6,
                        ls=(0, (2.2, 1.6)), alpha=0.9, zorder=1))
    ax.annotate(radius_label, (gx, gz - goal_radius), xytext=(0, 1.2), textcoords="offset points", ha="center",
                va="bottom", fontsize=5.4, color=style.INK_2, path_effects=cd.HALO, zorder=2)
    ax.plot(ref_xz[:, 0], ref_xz[:, 1], color=style.MUTED, lw=0.9, ls=(0, (3.0, 1.8)), zorder=2.5,
            solid_capstyle="round")
    draw_route_line(ax, route_xz, off_level, start_label)
    ax.plot([gx], [gz], marker="*", ms=8.0, mfc=style.INK, mec="white", mew=0.6, zorder=6)
    points = np.concatenate([route_xz, ref_xz, [[gx, gz]], key_xz])
    away = route_xz[0] - route_xz[min(3, len(route_xz) - 1)]  # cd.draw_route puts "start" on this side
    away = away / (np.linalg.norm(away) + 1e-9)
    boxes = [route_xz[0] + away * 12.0 * per_pt,  # the "start" label
             np.array([gx, gz - goal_radius - 4.0 * per_pt])]  # the radius label
    for p, lab in zip(key_xz, key_labels):
        ax.plot(*p, marker="o", ms=3.2, mfc=style.INK, mec="white", mew=0.5, zorder=7)
        q = _badge_spot(p, points, np.array(boxes), limits, per_pt)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=style.INK, lw=0.5, zorder=6.5)
        cd.key_badge(ax, q[0], q[1], lab, fs=5.8)
        boxes.append(q)
    bar = cd.nice_length(0.35 * (limits[1] - limits[0]))
    cd.scale_bar(ax, *_scale_bar_spot(limits, per_pt, bar, np.concatenate([points, boxes])), bar, f"{bar:g} m")
