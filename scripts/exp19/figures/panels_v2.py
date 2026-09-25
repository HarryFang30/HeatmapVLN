"""Drawing primitives of the EXP-19 figures v2 (paper-grade redesign of ``panels.py``).

The v1 primitives stay as they are (``panels.py``); v2 reuses their geometry
(rings, pixel -> bearing, badge placement on the map) and changes only how the
marks look:

* **Heat is a faint field**, never an opaque blob: one hue, opacity
  proportional to the value, at most ``HEAT_ALPHA`` over the imagery.
* **Markers carry the exact information**, and they are small:

  - predicted peak of a history slot the head calls visible
    (1 - P(none) >= 0.5): a filled dark-orange dot, ``PRED_MS`` pt, thin white ring;
  - true direction of a past frame: a hollow blue circle, ``GT_MS`` pt, 0.7 pt
    stroke, no halo (how close a dot sits to a circle depends on the panel's
    scale, so the legend defines the two marks separately, not a "match");
  - System1 mean path: small ink dots, every ``PATH_EVERY``-th waypoint on the
    image; on the 360-degree strips only waypoints at least ``PATH_SEP_PT`` apart;
  - System2 pixel goal: an ink ring with a centre dot.

Colour roles (one meaning each, the same in every panel and figure):

* orange ``HIST_COLOR`` = predicted history affordance map (field), darker
  ``HIST_PEAK_COLOR`` = its per-slot peaks;
* blue ``GT_COLOR`` = ground truth (true direction of a past frame; the
  slot badges of the history frames use a light blue ramp, since they name the
  same past frames);
* teal ``FUT_BIN_COLORS`` = predicted future affordance map, one step per time
  bin, darker = later, ending at ``FUT_COLOR``;
* ink = decisions (pixel goal, System1 path, executed actions, route, key badges);
* greys (``scripts/exp18/figures/style.py``) = context.

360-degree strips use the heading-centred ring of ``geo.stitch_ring`` in square
degrees: x = bearing from +180 at the left edge (behind, via the left) through
+90 (left), 0 (ahead) and -90 (right) to -180 at the right edge.  The history
strip spans +-45 deg of elevation (the 64 x 256 geometry of the four label
views), the future strip +-30 deg.  On the history strip every mark sits at its
own bearing and elevation; a grey hairline joins a past frame's predicted peak
to its true direction when the two are apart, so a dot inside a circle always
means a hit and a miss never looks nested.  The strips run
``STRIP_PAD`` deg past +-180 (the same bearings repeated), so a mark right
behind the robot can appear at both ends; a mark is drawn only where it fits
whole.  Hairlines mark the seams between the views and the +-180 wrap.  Only the
framed front +-39.5 deg (the deployed camera's 79 deg view) is in colour: that
is the part of the re-rendered surroundings the model was given; the rest is
muted, not given to the model (display only).

Display smoothing: the 64 x 64 maps of both heads carry the decoder's
4-pixel grid pattern (about 5.6 deg at a view's centre), which a ring sampled
at the strip's resolution turns into a checker texture.  Every strip's heat
field is therefore drawn after a Gaussian blur of ``SMOOTH_DEG`` deg on the
equiangular ring (circular in bearing; ``smooth_ring_image``), rescaled so that
its maximum is the map's own maximum (peaks keep their height; the dots still
mark the raw argmax).  The timeline's 1-D rings use ``smooth_rings`` with their
own sigma (``timeline_panel.TL_SMOOTH_DEG``).  Only the drawing is smoothed; no
stored value changes, and every caption says so (``fig_v2.smooth_sentence``).

Every text is at least ``MIN_FS`` pt at the printed size (7.0 in wide).
Nothing here draws a pose, a heading or an odometry value.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.exp18 import geometry as geo
from scripts.exp18.figures import common_draw as cd
from scripts.exp18.figures import style
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import panels as pn

import matplotlib  # noqa: E402  (cd.setup selects the backend and fonts)
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib import transforms as mtransforms  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette, marks, type sizes
# --------------------------------------------------------------------------- #
HIST_COLOR = "#d9591f"  # predicted history affordance map (field)
HIST_PEAK_COLOR = "#a8400f"  # its per-slot peaks (dots)
GT_COLOR = "#2f6fd0"  # true direction of a past frame
FUT_COLOR = "#16a07a"  # predicted future affordance map, last time bin
FUT_BIN_COLORS = ("#8fd4bd", "#5cc1a0", "#33b08b", FUT_COLOR)  # waypoints 1-8, 9-16, 17-24, 25-32 (darker = later)
SLOT_RAMP = LinearSegmentedColormap.from_list("exp19v2_slots", ["#dce8f8", "#9dbfee", "#5a93e0", GT_COLOR])
NOMAP_FILL = "#ecebe6"  # a call without an affordance map (System2 answered with turns / STOP)
HATCH_COLOR = "#d3d2c9"  # warm-up period (no affordance map yet): sparse light hatch
HATCH, HATCH_LW = "////", 0.3
CHIP_FILL = "#f0efec"
TURN_COLOR = style.MUTED  # executed turns on the timeline (context, not a decision mark)

HEAT_ALPHA = 0.55  # heat field over imagery: opacity = value x this
TL_HEAT_ALPHA = 0.62  # heat field on the timeline's plain plate
FUT_ALPHA = 0.62
PRED_MS, PRED_RING = 2.2, 0.4
GT_MS, GT_MEW = 3.4, 0.7
PATH_MS, PATH_EVERY = 1.5, 3
PATH_SEP_PT = 2.6  # System 1 waypoints on a 360-degree strip: at least this far apart (else they merge into a bar)
STRIP_PAD = 8.0  # deg the 360-degree strips run past +-180 (wrapped copy)
STOP_MS = 3.3  # where the rerun ended: small filled ink square on the route map
GOAL_MS = 5.6
HAIR = 0.35  # hairlines (seams, axes)
HIST_ELEV = 45.0
FUT_ELEV = 30.0
MODEL_VIEW_HALF_DEG = pn.MODEL_VIEW_HALF_DEG

MIN_FS = 6.0
FS = {"title": 7.5, "panel": 6.8, "head": 6.6, "body": 6.3, "small": 6.0}
SMOOTH_DEG = 2.0  # display blur of the strips' heat fields (sigma, deg of bearing / elevation)


# --------------------------------------------------------------------------- #
# Display smoothing of the heat fields
# --------------------------------------------------------------------------- #
def gaussian_taps(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """(offsets, weights) of a normalised Gaussian of ``sigma`` samples, cut at 3 sigma (a single tap at 0 when
    sigma is 0)."""
    if sigma <= 0:
        return np.zeros(1, dtype=int), np.ones(1)
    r = int(math.ceil(3.0 * sigma))
    k = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    return k, w / w.sum()


def _blur_axis(a: np.ndarray, sigma: float, axis: int, wrap: bool) -> np.ndarray:
    """Gaussian blur of ``a`` along ``axis``: circular (``wrap``) or with the edge values repeated."""
    k, w = gaussian_taps(sigma)
    if k.size == 1:
        return a.copy()
    r = int(k.max())
    n = a.shape[axis]
    idx = np.arange(-r, n + r)
    idx = np.mod(idx, n) if wrap else np.clip(idx, 0, n - 1)
    ext = np.take(a, idx, axis=axis)
    out = np.zeros_like(a, dtype=np.float64)
    for off, wt in zip(k, w):
        out += wt * np.take(ext, np.arange(r + off, r + off + n), axis=axis)
    return out


def keep_peak(smoothed: np.ndarray, raw: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    """``smoothed`` rescaled so that its maximum over ``axes`` equals that of ``raw`` (a blur never lowers a peak on
    the figure)."""
    lo = np.max(smoothed, axis=axes, keepdims=True)
    hi = np.max(raw, axis=axes, keepdims=True)
    scale = np.where(lo > 1e-9, hi / np.maximum(lo, 1e-9), 1.0)
    return np.clip(smoothed * scale, 0.0, None)


def smooth_rings(rings: np.ndarray, sigma_deg: float = SMOOTH_DEG) -> np.ndarray:
    """[..., n] rings (bin i centred on +180 - (i + 0.5) 360 / n), each blurred circularly along the bearing by
    ``sigma_deg`` and rescaled to keep its own maximum (display only)."""
    raw = np.clip(np.nan_to_num(np.asarray(rings, dtype=np.float64)), 0.0, None)
    n = raw.shape[-1]
    return keep_peak(_blur_axis(raw, sigma_deg * n / 360.0, raw.ndim - 1, True), raw, (raw.ndim - 1,))


def smooth_ring_image(img: np.ndarray, sigma_deg: float = SMOOTH_DEG) -> np.ndarray:
    """[h, w] equiangular ring image (w columns over 360 deg, square degrees) blurred by ``sigma_deg`` in bearing
    (circular) and elevation (edges repeated), rescaled to keep its maximum (display only)."""
    raw = np.clip(np.nan_to_num(np.asarray(img, dtype=np.float64)), 0.0, None)
    sigma = sigma_deg * raw.shape[1] / 360.0
    out = _blur_axis(_blur_axis(raw, sigma, 1, True), sigma, 0, False)
    return keep_peak(out, raw, (0, 1))


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #
def heat_rgba(values: np.ndarray, color, alpha_max: float) -> np.ndarray:
    """[..., 4] RGBA: one hue, opacity = clip(value, 0, 1) x alpha_max (a faint field, never opaque)."""
    v = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64)), 0.0, 1.0)
    out = np.empty(v.shape + (4,), dtype=np.float64)
    out[..., :3] = np.asarray(to_rgb(color))
    out[..., 3] = v * alpha_max
    return out


def bins_rgba(bins: np.ndarray, colors: Sequence[str] = FUT_BIN_COLORS, alpha_max: float = FUT_ALPHA) -> np.ndarray:
    """[..., 4] RGBA of time bins [B, ...] painted early -> late (later bins on top), opacity = value x alpha_max.

    "Over" compositing of the bins, returned straight (not premultiplied), so the
    result can itself be laid over any backdrop.
    """
    bins = np.clip(np.nan_to_num(np.asarray(bins, dtype=np.float64)), 0.0, 1.0)
    col = np.zeros(bins.shape[1:] + (3,))
    acc = np.zeros(bins.shape[1:])
    for b in range(bins.shape[0]):
        a = bins[b] * alpha_max
        col = col * (1.0 - a[..., None]) + np.asarray(to_rgb(colors[b])) * a[..., None]
        acc = acc * (1.0 - a) + a
    out = np.zeros(bins.shape[1:] + (4,))
    nz = acc > 1e-9
    out[nz, :3] = col[nz] / acc[nz, None]
    out[..., 3] = acc
    return out


def slot_color(k: int, num: int = bd.NUM_SLOTS):
    return SLOT_RAMP(0.0 if num <= 1 else k / (num - 1))


def slot_text_color(k: int, num: int = bd.NUM_SLOTS) -> str:
    return style.INK if k < num * 0.6 else "white"


# --------------------------------------------------------------------------- #
# Marks
# --------------------------------------------------------------------------- #
def pred_dot(ax, x, y, scale: float = 1.0, zorder: float = 8, clip_on: bool = True, alpha: float = 1.0):
    """Predicted peak of a visible history slot: small filled dark-orange dot with a thin white ring."""
    ax.plot(np.atleast_1d(x), np.atleast_1d(y), ls="none", marker="o", ms=PRED_MS * scale, mfc=HIST_PEAK_COLOR,
            mec="white", mew=PRED_RING * min(scale, 1.0), zorder=zorder, clip_on=clip_on, alpha=alpha)


def gt_ring(ax, x, y, scale: float = 1.0, zorder: float = 7.5, clip_on: bool = True):
    """True direction of a past frame: small hollow blue circle, no halo."""
    ax.plot(np.atleast_1d(x), np.atleast_1d(y), ls="none", marker="o", ms=GT_MS * scale, mfc="none", mec=GT_COLOR,
            mew=GT_MEW * max(min(scale, 1.0), 0.75), zorder=zorder, clip_on=clip_on)


def path_marks(ax, x, y, ms: float = PATH_MS, zorder: float = 8, rim: bool = True, clip_on: bool = True):
    """System1 path: small ink dots (a hairline white rim keeps them visible on dark floors)."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if x.size:
        ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=style.INK, mec="white" if rim else "none",
                mew=0.2 if rim else 0.0, zorder=zorder, clip_on=clip_on)


def goal_ring(ax, u: float, v: float, size: float = GOAL_MS, zorder: float = 9, clip_on: bool = False):
    """System2 pixel goal: ink ring with a centre dot (thin white halo for dark images)."""
    ax.plot([u], [v], marker="o", ms=size, mfc="none", mec=style.INK, mew=0.9, zorder=zorder, clip_on=clip_on,
            path_effects=[pe.withStroke(linewidth=1.7, foreground="white")])
    ax.plot([u], [v], marker="o", ms=1.3, mfc=style.INK, mec="white", mew=0.25, zorder=zorder, clip_on=clip_on)


def subsample_path(n: int, every: int = PATH_EVERY) -> np.ndarray:
    """Indices of every ``every``-th waypoint, always including the last one."""
    idx = list(range(0, n, every))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx, dtype=int)


def both_ends(bearing: float, margin: float = 6.0) -> List[float]:
    """A bearing, and its copy one turn away when it sits within ``margin`` deg of the +-180 edge."""
    out = [bearing]
    if bearing > 180.0 - margin:
        out.append(bearing - 360.0)
    elif bearing < -180.0 + margin:
        out.append(bearing + 360.0)
    return out


def edge_triangle(ax, bearing: float, up: bool, band: float, kind: str, ms: float = 3.2) -> None:
    """A direction beyond the strip's elevation band: a small triangle on its edge, pointing out of the band."""
    y = band - cd.pts_to_data(ax, 0.0, 2.0)[1] if up else -band + cd.pts_to_data(ax, 0.0, 2.0)[1]
    kw = (dict(mfc="none", mec=GT_COLOR, mew=GT_MEW) if kind == "gt" else
          dict(mfc=HIST_PEAK_COLOR, mec="white", mew=PRED_RING))
    for b in both_ends(bearing, margin=STRIP_PAD):
        ax.plot([b], [y], marker="^" if up else "v", ms=ms, zorder=8, clip_on=True, **kw)


# --------------------------------------------------------------------------- #
# 360-degree strips
# --------------------------------------------------------------------------- #
def strip_height(width: float, elev: float) -> float:
    """Height (same unit as width) of a square-degree strip spanning +-elev."""
    return width * 2.0 * elev / 360.0


def muted_ring(views: np.ndarray, width: int, elev: float, sat: float = 0.08, white: float = 0.6) -> np.ndarray:
    """[h, width, 3] ring of the four re-rendered views: the camera's +-39.5 deg in colour, the rest muted.

    Ring pixels outside every view's field (above / below the seams) are the plain
    surface colour, not black.
    """
    ring, valid = geo.stitch_ring_rgb(views, width=width, height=pn.ring_height(width, elev), elev_top=elev,
                                      elev_bottom=-elev)
    full = ring.astype(np.float32) / 255.0
    out = cd.mute(full, sat=sat, white=white)
    front = pn.front_columns(width)
    out[:, front] = full[:, front]
    out[~np.asarray(valid, dtype=bool)] = np.asarray(to_rgb(style.SURFACE), dtype=np.float32)
    return out


def setup_strip(ax, elev: float, pad: float = STRIP_PAD) -> None:
    """Strip axes: x = bearing from +180 + pad (left) to -180 - pad (right), y = elevation."""
    ax.set_xlim(180.0 + pad, -180.0 - pad)
    ax.set_ylim(-elev, elev)
    ax.set_autoscale_on(False)
    cd.clean_axes(ax)


def wrap_columns(img: np.ndarray, pad: float = STRIP_PAD) -> Tuple[np.ndarray, float]:
    """A 360-degree ring image [h, w, ...] extended by ``pad`` deg of wrapped columns at both ends.

    Returns (image, the pad actually used in degrees: a whole number of columns)."""
    w = img.shape[1]
    n = int(round(pad * w / 360.0))
    if n <= 0:
        return img, 0.0
    return np.concatenate([img[:, -n:], img, img[:, :n]], axis=1), n * 360.0 / w


def show_strip(ax, img: np.ndarray, elev: float, zorder: float = 0, pad: float = STRIP_PAD) -> None:
    """``imshow`` of a ring image (+180 at column 0) on strip axes, wrapped ``pad`` deg past +-180."""
    wide, p = wrap_columns(img, pad)
    ax.imshow(wide, extent=(180.0 + p, -180.0 - p, -elev, elev), aspect="auto", interpolation="bilinear",
              zorder=zorder)


def strip_seams(ax, elev: float, color=style.GRID, lw: float = HAIR) -> None:
    """Seams between the four views and the +-180 wrap of a padded strip."""
    for b in (180.0, 135.0, 45.0, -45.0, -135.0, -180.0):
        ax.plot([b, b], [-elev, elev], color=color, lw=lw, zorder=2, solid_capstyle="butt")


def camera_frame(ax, elev: float, lw: float = 0.6) -> None:
    """Frame of the deployed camera's view (+-39.5 deg): the only part of the strip given to the model."""
    ax.add_patch(Rectangle((-MODEL_VIEW_HALF_DEG, -elev), 2 * MODEL_VIEW_HALF_DEG, 2 * elev, fill=False,
                           ec=style.INK, lw=lw, zorder=6, clip_on=False))


def history_strip_marks(ks: bd.KeyStep) -> Tuple[list, list]:
    """(gt, pred) marks of a key moment's history strip: lists of (slot, bearing, elevation).

    gt: every real slot visible in some view in the ground truth; pred: the
    argmax of every real slot whose 1 - P(none) >= 0.5.
    """
    gt = [(k, *pn.pixel_to_ring(v, r, c)) for k, v, r, c in bd.gt_history_peaks(ks.hist_gt_peak, ks.hist_mask)]
    conf = 1.0 - np.asarray(ks.hist_none, dtype=np.float64)
    pred = []
    for k in np.nonzero(np.asarray(ks.hist_mask, dtype=bool) & (conf >= 0.5))[0]:
        v, r, c = np.unravel_index(int(np.argmax(ks.hist_pred[k])), np.asarray(ks.hist_pred[k]).shape)
        pred.append((int(k), *pn.pixel_to_ring(int(v), float(r), float(c))))
    return gt, pred


def inside_copies(ax, bearing: float, radius_pt: float, tol_pt: float = 0.3) -> List[float]:
    """The strip positions of a bearing (itself and, near +-180, its copy one turn away) at which a mark of
    ``radius_pt`` is drawn whole inside the strip's padded x range; the one closest to the middle when none is."""
    lim = abs(ax.get_xlim()[0])
    r = (radius_pt - tol_pt) * abs(cd.pts_to_data(ax, 1.0)[0])
    cands = [bearing, bearing - 360.0, bearing + 360.0]
    whole = [c for c in cands if abs(c) + r <= lim]
    return whole or [min(cands, key=abs)]


PAIR_LW = 0.45  # hairline joining a past frame's predicted peak to its true direction on a history strip
PAIR_COLOR = style.INK_2


def pair_segment(pred_b: float, pred_e: float, gt_b: float, gt_e: float) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the hairline from a predicted peak to the true direction of the same past frame, taking
    the short way round the ring (the true bearing unwrapped next to the predicted one)."""
    return pred_b, pred_e, pred_b + float(geo.wrap_deg(gt_b - pred_b)), gt_e


def draw_history_strip(ax, ks: bd.KeyStep, width_px: int, heat_px: int, elev: float = HIST_ELEV) -> dict:
    """Muted surroundings, faint predicted history field (display-smoothed, ``smooth_ring_image``), and every past
    frame's marks at their own bearing AND elevation (the 64 x 256 geometry of the four label views,
    ``pn.pixel_to_ring``): a small dark-orange dot at the
    predicted peak, a small hollow blue circle at the true direction, and a grey hairline joining the two marks of
    the same past frame when they are apart (the error of that frame; a hit is a dot inside its circle, a miss
    never looks nested).  A mark beyond the strip's elevation band is a triangle on its edge (no hairline).  A
    mark near +-180 is drawn at every end where it fits whole; hairlines are drawn at both ends, clipped.
    Returns {"edge": marks drawn as edge triangles, "pairs": hairlines drawn, "scale": 1.0}.
    """
    composite = bd.history_pred_composite(ks.hist_pred, ks.hist_none, ks.hist_mask)
    rgb = muted_ring(ks.pano_rgb, width_px, elev)
    heat = smooth_ring_image(pn.ring_heat(composite, heat_px, elev))
    show_strip(ax, rgb, elev, zorder=0)
    show_strip(ax, heat_rgba(heat, HIST_COLOR, HEAT_ALPHA), elev, zorder=1)
    setup_strip(ax, elev)
    strip_seams(ax, elev, color="white", lw=HAIR)
    camera_frame(ax, elev)
    gt, pred = history_strip_marks(ks)
    kx, ky = abs(cd.pts_to_data(ax, 1.0, 0.0)[0]), abs(cd.pts_to_data(ax, 0.0, 1.0)[1])  # deg per pt
    true_of = {int(k): (float(b), float(e)) for k, b, e in gt}
    pairs = 0
    for k, b, e in pred:
        if int(k) not in true_of:
            continue
        gb, ge = true_of[int(k)]
        if abs(e) > elev or abs(ge) > elev:
            continue
        x0, y0, x1, y1 = pair_segment(float(b), float(e), gb, ge)
        if math.hypot((x1 - x0) / kx, (y1 - y0) / ky) <= GT_MS / 2:  # the dot sits inside its circle
            continue
        for shift in (-360.0, 0.0, 360.0):
            ax.plot([x0 + shift, x1 + shift], [y0, y1], color=PAIR_COLOR, lw=PAIR_LW, zorder=7,
                    solid_capstyle="butt", clip_on=True)
        pairs += 1
    edge = 0
    for kind, marks in (("gt", gt), ("pred", pred)):
        radius = (GT_MS + GT_MEW) / 2 if kind == "gt" else (PRED_MS + PRED_RING) / 2
        for k, b, e in marks:
            if abs(e) > elev:
                edge_triangle(ax, b, e > 0, elev, kind)
                edge += 1
                continue
            for bb in inside_copies(ax, b, radius):
                (gt_ring if kind == "gt" else pred_dot)(ax, bb, e)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(style.AXIS)
        s.set_linewidth(HAIR)
    return {"edge": edge, "pairs": pairs, "scale": 1.0}


def draw_future_strip(ax, ks: bd.KeyStep, width_px: int, elev: float = FUT_ELEV) -> None:
    """Faint teal field per time bin (later on top, each bin display-smoothed, ``smooth_ring_image``) on a plain
    plate, and the System1 path as ink dots."""
    bins = bd.future_bin_maps(ks.fut_pred)
    rings = np.stack([smooth_ring_image(pn.ring_heat(bins[b], width_px, elev)) for b in range(bins.shape[0])])
    ax.set_facecolor(style.SURFACE)
    show_strip(ax, bins_rgba(rings), elev, zorder=1)
    setup_strip(ax, elev)
    strip_seams(ax, elev)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(style.AXIS)
        s.set_linewidth(HAIR)
    bearing, e, _ = bd.path_directions(ks.path_cam)
    if bearing.size:
        ee = np.clip(e, -elev, elev)
        idx = thin_marks(ax, bearing, ee, PATH_SEP_PT)
        xy = [(bb, float(ee[i])) for i in idx for bb in both_ends(float(bearing[i]), margin=STRIP_PAD)]
        path_marks(ax, [q[0] for q in xy], [q[1] for q in xy], ms=PATH_MS, rim=False)


def thin_marks(ax, x: np.ndarray, y: np.ndarray, min_sep_pt: float) -> List[int]:
    """Indices of marks at least ``min_sep_pt`` apart on the axes, walking back from the last one (always kept).

    The System 1 path's 33 waypoints span only a few degrees of bearing on a
    360-degree strip; drawn every one they merge into a bar."""
    n = len(x)
    if n == 0:
        return []
    kx, ky = cd.pts_to_data(ax, 1.0, 0.0)[0], cd.pts_to_data(ax, 0.0, 1.0)[1]
    pts = np.stack([np.asarray(x, dtype=np.float64) / kx, np.asarray(y, dtype=np.float64) / ky], axis=1)
    keep = [n - 1]
    for i in range(n - 2, -1, -1):
        if np.hypot(*(pts[i] - pts[keep[-1]])) >= min_sep_pt:
            keep.append(i)
    return sorted(keep)


STRIP_TICK_LEN, STRIP_TICK_PAD = 1.8, 1.2  # strip bearing ticks (pt)
STRIP_LABEL_AIR = 3.0  # neighbouring bearing labels under a strip keep at least this much air (pt) ...
STRIP_LABEL_AIR_EVEN = 4.0  # ... and, where labels must move to keep it, up to this much, evenly spread


def _monotone_fit(q: Sequence[float], lo: float, hi: float) -> List[float]:
    """Least-squares non-decreasing fit of ``q`` (pool adjacent violators) clipped to [lo, hi]: the projection of
    ``q`` onto {lo <= y_1 <= ... <= y_n <= hi}."""
    blocks: List[List[float]] = []  # [mean, count]
    for v in q:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    out: List[float] = []
    for m, n in blocks:
        out += [min(max(m, lo), hi)] * int(n)
    return out


def strip_label_layout(pos: Sequence[float], widths: Sequence[float], width_pt: float,
                       air: float = STRIP_LABEL_AIR, air_even: float = STRIP_LABEL_AIR_EVEN
                       ) -> Optional[List[float]]:
    """Left edges (pt from the strip's left edge) of the five bearing labels behind / left / ahead / right /
    behind (tick positions ``pos``, label ``widths``), every neighbour at least ``air`` apart, or None when they
    do not fit.

    Tried in order: the two "behind" labels starting (ending) at their +-180 ticks, then at the strip's own edges
    (``STRIP_PAD`` deg beyond +-180), with left / ahead / right centred on their ticks; then, at the strip's edges,
    left / ahead / right moved as little as possible (least squares, order kept) to leave the largest air from
    ``air_even`` down to ``air`` (0.25 pt steps) -- "right" towards the middle, "left" mirrored, "ahead" along
    when both squeeze it -- as long as every label still spans its own tick."""
    w = list(widths)
    pref = [p - wi / 2 for p, wi in zip(pos[1:4], w[1:4])]  # the middle labels centred on their ticks

    def fits(xs, a, b, gap):
        chain = [a] + list(xs) + [b]
        return all(chain[i + 1] - (chain[i] + w[i]) >= gap - 1e-6 for i in range(4))

    for a, b in ((pos[0], pos[4] - w[4]), (0.0, width_pt - w[4])):  # the edge labels' left edges
        if fits(pref, a, b, air):
            return [a] + pref + [b]
    a, b = 0.0, width_pt - w[4]
    steps = max(int(round((air_even - air) / 0.25)), 0)
    for gap in (air + 0.25 * i for i in range(steps, -1, -1)):
        off = [0.0, w[1] + gap, w[1] + w[2] + 2 * gap]  # x_i = y_i + off_i turns the gaps into y_1 <= y_2 <= y_3
        ys = _monotone_fit([p - o for p, o in zip(pref, off)], a + w[0] + gap, b - gap - w[3] - off[2])
        xs = [y + o for y, o in zip(ys, off)]
        if fits(xs, a, b, gap) and all(x - 1e-6 <= p <= x + wi + 1e-6 for x, p, wi in zip(xs, pos[1:4], w[1:4])):
            return [a] + xs + [b]
    return None


def strip_ticks(ax, labels: Sequence[str], fs: float = FS["small"]) -> bool:
    """Bearing ticks under a strip: +180 / +90 / 0 / -90 / -180 = behind / left / ahead / right / behind.

    The tick marks sit at those five bearings.  The labels keep ``STRIP_LABEL_AIR`` (3 pt) between neighbours
    (``strip_label_layout``): the two "behind" labels aligned inwards from their +-180 ticks, or from the strip's
    own edges on a narrow strip, and on the narrowest (the overview's) left / ahead / right moved inwards as
    little as keeps up to ``STRIP_LABEL_AIR_EVEN`` between all five, each still over its tick.  Only where even that does not fit are both edge labels left out (left / ahead /
    right), so the two ends always match.  The font stays at ``fs`` (>= 6 pt).  Returns whether the edges are
    labelled."""
    ticks = [180.0, 90.0, 0.0, -90.0, -180.0]
    labels = list(labels)
    per_pt = abs(cd.pts_to_data(ax, 1.0)[0])  # deg per point
    x0, x1 = ax.get_xlim()
    widths = [cd.text_width_pt(ax.figure, t, fs) for t in labels]
    pos = [(x0 - b) / per_pt for b in ticks]  # points from the strip's left edge
    width_pt = (x0 - x1) / per_pt
    lefts = strip_label_layout(pos, widths, width_pt)
    edges = lefts is not None
    if edges:  # label anchors in data x: the edge labels by their outer end, the others by their centre
        label_x = ([x0 - lefts[0] * per_pt] + [x0 - (l_ + w_ / 2) * per_pt for l_, w_ in zip(lefts[1:4], widths[1:4])]
                   + [x0 - (lefts[4] + widths[4]) * per_pt])
    else:
        label_x, labels = ticks[1:-1], labels[1:-1]
    # the marks are minor ticks at the five bearings; the labels are major ticks without a mark, so a label can
    # sit off its bearing (an edge label at the strip's edge) while its mark stays put (minor marks are kept)
    ax.xaxis.remove_overlapping_locs = False
    ax.set_xticks(ticks, minor=True)
    ax.set_xticks(label_x)
    ax.set_xticklabels(labels, fontsize=fs, color=style.INK_2)
    if edges:
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
    ax.tick_params(axis="x", which="minor", length=STRIP_TICK_LEN, width=HAIR, color=style.AXIS, labelbottom=False)
    ax.tick_params(axis="x", which="major", length=0.0, width=HAIR, color=style.AXIS,
                   pad=STRIP_TICK_LEN + STRIP_TICK_PAD)
    return edges


# --------------------------------------------------------------------------- #
# Decision image and history frames
# --------------------------------------------------------------------------- #
def draw_decision_image(ax, ks: bd.KeyStep, tag: Optional[str] = None, fs: float = FS["small"],
                        badge: Optional[str] = None) -> None:
    """Decision image, System1 path (every 3rd waypoint on the image) and the System2 pixel goal; ``tag`` in the
    top-left corner, after a K ``badge`` when given."""
    img = ks.decision_rgb
    h, w = pn.draw_image(ax, img, frame_color=style.INK, frame_lw=0.5)
    uv = np.asarray(ks.path_uv, dtype=np.float64).reshape(-1, 2)
    if len(uv):
        uv = uv[subsample_path(len(uv))]
        inside = (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)
        path_marks(ax, uv[inside, 0], uv[inside, 1])
    goal = ks.pixel_goal_uv
    if goal is not None and bd.goal_inside(goal, img.shape):
        goal_ring(ax, float(goal[0]), float(goal[1]), clip_on=True)
    x_pt, y_pt = TAG_X0_PT, -3.0 - fs * 0.62
    if badge:
        bw = cd.text_width_pt(ax.figure, badge, MIN_FS, fontweight="bold") + 2 * 0.22 * MIN_FS
        cd.key_badge(ax, 0.0, 1.0, badge, fs=MIN_FS, zorder=11,
                     transform=mtransforms.offset_copy(ax.transAxes, fig=ax.figure, x=x_pt + bw / 2, y=y_pt,
                                                       units="points"))
        x_pt += bw + TAG_GAP_PT
    if tag:
        ax.text(0.0, 1.0, tag, transform=mtransforms.offset_copy(ax.transAxes, fig=ax.figure, x=x_pt + 1.0, y=y_pt,
                                                                  units="points"),
                ha="left", va="center", fontsize=fs, color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=(0, 0, 0, 0.55), ec="none"))


TAG_X0_PT, TAG_GAP_PT = 2.5, 2.2  # decision-image corner: badge from 2.5 pt, the tag 2.2 pt after it

SLOT_PAD = 0.08  # circle padding of a slot badge (x font size); the overview's thumbnails use SLOT_PAD_SMALL
SLOT_PAD_SMALL = 0.03


def step_tag(ax, x: float, y: float, text: str, fs: float = MIN_FS):
    """The step of a history frame, bottom-right corner of its thumbnail (x, y = that corner, point units)."""
    return ax.text(x - 1.2, y + 1.2, text, ha="right", va="bottom", fontsize=fs, color="white", zorder=9,
                   bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.2", fc=(0, 0, 0, 0.58), ec="none"))


def slot_badge(ax, x: float, y: float, k: int, fs: float = FS["small"], num: int = bd.NUM_SLOTS,
               pad: float = SLOT_PAD):
    """Slot number of a history frame (1 = oldest) in the light blue slot ramp."""
    return ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=fs, fontweight="bold",
                   color=slot_text_color(k, num), zorder=9,
                   bbox=dict(boxstyle=f"circle,pad={pad:g}", fc=slot_color(k, num), ec="white", lw=0.35))


# --------------------------------------------------------------------------- #
# Executed actions
# --------------------------------------------------------------------------- #
def chip_width(action: int, size: float) -> float:
    return max(size * 2.4, MIN_FS * 3.0) if action == bd.STOP else size


def action_chip(ax, x: float, y: float, action: int, size: float = 8.0, zorder: float = 6) -> float:
    """One executed action at (x = left edge, y = centre), point units: forward/turn arrow or STOP (6 pt)."""
    w = chip_width(action, size)
    if action == bd.STOP:
        ax.add_patch(FancyBboxPatch((x, y - size / 2), w, size, boxstyle="round,pad=0,rounding_size=1.3",
                                    fc=style.INK, ec="none", zorder=zorder))
        ax.text(x + w / 2, y, "STOP", ha="center", va="center", fontsize=MIN_FS, fontweight="bold",
                color="white", zorder=zorder + 1)
        return w
    ax.add_patch(FancyBboxPatch((x, y - size / 2), w, size, boxstyle="round,pad=0,rounding_size=1.3",
                                fc=CHIP_FILL, ec=style.AXIS, lw=0.4, zorder=zorder))
    c = np.array([x + w / 2, y])
    d = np.asarray({bd.FORWARD: (0.0, 1.0), bd.LEFT: (-1.0, 0.0), bd.RIGHT: (1.0, 0.0)}[int(action)]) * size * 0.32
    ax.add_patch(FancyArrowPatch(tuple(c - d), tuple(c + d), arrowstyle="-|>", mutation_scale=size * 0.58,
                                 color=style.INK, lw=0.75, shrinkA=0, shrinkB=0, zorder=zorder + 1))
    return w


def action_chips(ax, x: float, y: float, actions: Sequence[int], size: float = 8.0, gap: float = 1.5) -> float:
    """Chips of an action chunk from x (left edge), point units; returns the total width."""
    x0 = x
    for a in actions:
        x0 += action_chip(ax, x0, y, int(a), size=size) + gap
    return x0 - x - (gap if len(actions) else 0.0)


# --------------------------------------------------------------------------- #
# Top-down route map (v1 geometry, v2 type sizes)
# --------------------------------------------------------------------------- #
BADGE_HALF = pn.BADGE_HALF_PT  # half width / height of a K badge (pt), with a little air
CLUSTER_PT = 21.6  # key positions closer than 0.3 in fan their badges out evenly around the cluster


def draw_route_map(ax, level, route_xz, ref_xz, start_xz, goal_xz, goal_radius: float, key_xz, key_labels,
                   start_label: str, radius_label: str, off_level: Optional[np.ndarray] = None,
                   fs: float = MIN_FS, stop_mark: bool = True) -> dict:
    """Muted top-down map, reference path (grey dashed), executed route (ink), start, goal + success radius,
    where the rerun ended (small ink square) and the K badges.

    Badges go where they have room; key positions that cluster within 0.3 in
    fan their badges out evenly around the cluster, in the cluster's own angular
    order, so no leader crosses another.  The start label is placed last, clear
    of the badges and leaders; when no spot is clear it is left out (the open
    circle is in the legend).  Every text is ``fs`` >= 6 pt.  No heading arrow,
    nothing from a pose estimate: every mark is a recorded simulator position.
    Returns {"stop_drawn", "start_label"}.
    """
    route_xz = np.asarray(route_xz, dtype=np.float64).reshape(-1, 2)
    ref_xz = np.asarray(ref_xz, dtype=np.float64).reshape(-1, 2)
    key_xz = np.asarray(key_xz, dtype=np.float64).reshape(-1, 2)
    limits = pn.map_limits([route_xz, ref_xz], goal_xz, goal_radius, cd.axes_aspect_hw(ax), pad=0.9)
    if level is None:
        pn.draw_plate(ax, limits)
    else:
        cd.draw_topdown(ax, level, limits, sat=0.18, white=0.6)
    cd.clean_axes(ax, spines=True, lw=HAIR)
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    gx, gz = float(goal_xz[0]), float(goal_xz[1])
    ax.add_patch(Circle((gx, gz), goal_radius, fc=cd.mix(style.INK, "white", 0.94), ec=style.INK_2, lw=0.5,
                        ls=(0, (2.2, 1.6)), alpha=0.9, zorder=1))
    ax.annotate(radius_label, (gx, gz - goal_radius), xytext=(0, 1.0), textcoords="offset points", ha="center",
                va="bottom", fontsize=fs, color=style.INK_2, path_effects=cd.HALO_THIN, zorder=2)
    radius_box = _box((gx, gz - goal_radius - (1.0 + fs * 0.6) * per_pt), cd.text_width_pt(ax.figure, radius_label,
                                                                                           fs) / 2 + 1.0,
                      fs * 0.6 + 0.5, per_pt)
    ax.plot(ref_xz[:, 0], ref_xz[:, 1], color=style.MUTED, lw=0.8, ls=(0, (3.0, 1.8)), zorder=2.5)
    pn.draw_route_line(ax, route_xz, off_level, None)
    ax.plot([gx], [gz], marker="*", ms=7.0, mfc=style.INK, mec="white", mew=0.5, zorder=6)
    end = route_xz[-1]
    stop_drawn = bool(stop_mark and stop_apart(end, goal_xz, per_pt))
    if stop_drawn:
        ax.plot([end[0]], [end[1]], marker="s", ms=STOP_MS, mfc=style.INK, mec="white", mew=0.45, zorder=7.2)
    points = np.concatenate([route_xz, ref_xz, [[gx, gz]], key_xz]) if len(key_xz) else \
        np.concatenate([route_xz, ref_xz, [[gx, gz]]])
    dense = _densify(np.concatenate([route_xz, ref_xz]), 2.0 * per_pt)
    for p in key_xz:
        ax.plot(*p, marker="o", ms=2.6, mfc=style.INK, mec="white", mew=0.4, zorder=7)
    spots = badge_spots(key_xz, np.concatenate([dense, points]), [radius_box], limits, per_pt)
    boxes = [radius_box]
    leaders = []
    for p, q, lab in zip(key_xz, spots, key_labels):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=style.INK, lw=0.4, zorder=6.5, gid=f"leader:{lab}")
        cd.key_badge(ax, q[0], q[1], lab, fs=fs)
        boxes.append(_box(q, BADGE_HALF[0], BADGE_HALF[1], per_pt))
        leaders.append((p, q))
    # start label last: clear of the badges and their leaders, else left out
    d = route_xz[min(3, len(route_xz) - 1)] - route_xz[0]
    d = -d / (np.linalg.norm(d) + 1e-9)
    hard = [_densify(np.array([p, q]), 1.0 * per_pt) for p, q in leaders]
    for b in boxes[1:]:
        hard.append(_box_points(b, per_pt))
    hard_pts = np.concatenate(hard) if hard else np.zeros((0, 2))
    avoid = _densify(np.concatenate([route_xz, ref_xz, key_xz]) if len(key_xz) else
                     np.concatenate([route_xz, ref_xz]), 2.0 * per_pt)
    placed = place_label(route_xz[0], d, cd.text_width_pt(ax.figure, start_label, fs), fs, limits, per_pt, avoid,
                         hard=hard_pts)
    if placed is not None:
        (ox, oy), ha, va = placed
        ax.annotate(start_label, route_xz[0], xytext=(ox, oy), textcoords="offset points", ha=ha, va=va,
                    fontsize=fs, color=style.INK, path_effects=cd.HALO_THIN, zorder=5)
    taken = np.concatenate([points] + [_box_points(b, per_pt) for b in boxes])
    bar = cd.nice_length(0.3 * (limits[1] - limits[0]))
    cd.scale_bar(ax, *pn._scale_bar_spot(limits, per_pt, bar, taken), bar, f"{bar:g} m", fs=fs)
    return {"stop_drawn": bool(stop_drawn), "start_label": placed is not None}


def stop_apart(end_xz, goal_xz, per_pt: float, min_pt: float = 5.5) -> bool:
    """The end-of-rerun square is drawn only where the goal star does not cover it."""
    return bool(float(np.hypot(end_xz[0] - goal_xz[0], end_xz[1] - goal_xz[1])) / per_pt > min_pt)


def stop_shown(route_xz, ref_xz, goal_xz, goal_radius: float, w_in: float, h_in: float) -> bool:
    """Whether ``draw_route_map`` on axes w_in x h_in will draw the end-of-rerun square (for the legend, before
    the page is laid out)."""
    route_xz = np.asarray(route_xz, dtype=np.float64).reshape(-1, 2)
    limits = pn.map_limits([route_xz, np.asarray(ref_xz, dtype=np.float64).reshape(-1, 2)], goal_xz, goal_radius,
                           h_in / w_in, pad=0.9)
    return stop_apart(route_xz[-1], goal_xz, (limits[1] - limits[0]) / (w_in * 72.0))


def _box(c, hw_pt: float, hh_pt: float, per_pt: float) -> np.ndarray:
    """(x0, z0, x1, z1) of a box of half size (hw, hh) pt around c (map units)."""
    return np.array([c[0] - hw_pt * per_pt, c[1] - hh_pt * per_pt, c[0] + hw_pt * per_pt, c[1] + hh_pt * per_pt])


def _box_points(b: np.ndarray, per_pt: float, step_pt: float = 1.5) -> np.ndarray:
    xs = np.arange(b[0], b[2] + 1e-9, step_pt * per_pt)
    zs = np.arange(b[1], b[3] + 1e-9, step_pt * per_pt)
    return np.stack(np.meshgrid(xs, zs), -1).reshape(-1, 2)


def _seg_hits_box(p, q, b: np.ndarray, n: int = 24) -> bool:
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p)[None, :] * (1 - t) + np.asarray(q)[None, :] * t
    return bool(np.any((pts[:, 0] > b[0]) & (pts[:, 0] < b[2]) & (pts[:, 1] > b[1]) & (pts[:, 1] < b[3])))


def _segs_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def clusters(key_xz: np.ndarray, per_pt: float, within_pt: float = CLUSTER_PT) -> List[List[int]]:
    """Groups of key positions linked by distances below ``within_pt`` (single linkage)."""
    n = len(key_xz)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(key_xz[i] - key_xz[j]) / per_pt < within_pt:
                parent[find(j)] = find(i)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _config_score(ps, qs, points, fixed_boxes, limits, per_pt, dists) -> float:
    hw, hh = BADGE_HALF
    score = 0.0
    boxes = list(fixed_boxes)
    for p, q, dist in zip(ps, qs, dists):
        inside = (limits[0] + hw * per_pt < q[0] < limits[1] - hw * per_pt
                  and limits[2] + hh * per_pt < q[1] < limits[3] - hh * per_pt)
        if not inside:
            return -np.inf
        room = float(np.min(np.linalg.norm(points - q, axis=1))) / per_pt if len(points) else 99.0
        for b in boxes:  # separation from badges / labels already placed (< 0: overlap)
            gap = max(b[0] - (q[0] + hw * per_pt), (q[0] - hw * per_pt) - b[2],
                      b[1] - (q[1] + hh * per_pt), (q[1] - hh * per_pt) - b[3]) / per_pt
            room = min(room, 6.0 + gap)
        score += min(room, 12.0) - 0.15 * dist
        boxes.append(_box(q, hw, hh, per_pt))
    for i, (p, q) in enumerate(zip(ps, qs)):  # a leader must not run through a badge or cross another leader
        for j, b in enumerate(boxes):
            if j != len(fixed_boxes) + i and _seg_hits_box(p, q, b):
                score -= 40.0
        for p2_, q2_ in list(zip(ps, qs))[i + 1:]:
            if _segs_cross(p, q, p2_, q2_):
                score -= 40.0
    return score


def badge_spots(key_xz: np.ndarray, points: np.ndarray, fixed_boxes: Sequence[np.ndarray], limits,
                per_pt: float) -> List[np.ndarray]:
    """Badge centres for every key position (map units).

    A lone key position: the spot with the most room 11-23 pt away (as v1).  A
    cluster (within 0.3 in): the badges fan out at even angles around the
    cluster's centre, the whole fan rotated and sized for the most room, each
    badge on the side of its own position (its angular order around the
    centre), so leaders neither cross each other nor run through a badge.
    """
    key_xz = np.asarray(key_xz, dtype=np.float64).reshape(-1, 2)
    spots: List[Optional[np.ndarray]] = [None] * len(key_xz)
    boxes = list(fixed_boxes)
    for group in sorted(clusters(key_xz, per_pt), key=len, reverse=True):
        ps = key_xz[group]
        best, best_s = None, -np.inf
        if len(group) == 1:
            p = ps[0]
            for dist in (11.0, 17.0, 23.0):
                for ang in np.radians(np.arange(0.0, 360.0, 30.0) + 15.0):
                    q = p + np.array([math.cos(ang), math.sin(ang)]) * dist * per_pt
                    s_ = _config_score([p], [q], points, boxes, limits, per_pt, [dist])
                    if s_ > best_s:
                        best, best_s = [q], s_
        else:
            c = ps.mean(0)
            own = np.arctan2(ps[:, 1] - c[1], ps[:, 0] - c[0])
            order = np.argsort(own)
            n = len(group)
            for step in sorted({360.0 / n, 60.0, 45.0}, reverse=True):
                if step * (n - 1) >= 360.0 - 1e-6 and step != 360.0 / n:
                    continue
                for base in np.arange(0.0, 360.0, 15.0):
                    angs = np.radians(base + step * np.arange(n))
                    for dist in (16.0, 21.0, 27.0):
                        qs = [None] * n
                        for rank, i in enumerate(order):
                            qs[i] = c + np.array([math.cos(angs[rank]), math.sin(angs[rank])]) * dist * per_pt
                        s_ = _config_score(list(ps), qs, points, boxes, limits, per_pt, [dist] * n)
                        if s_ > best_s:
                            best, best_s = qs, s_
        if best is None:  # nothing fits on the map: badges on their positions
            best = list(ps)
        for i, q in zip(group, best):
            spots[i] = q
            boxes.append(_box(q, BADGE_HALF[0], BADGE_HALF[1], per_pt))
    return spots


def _densify(points: np.ndarray, step: float) -> np.ndarray:
    """Polyline vertices plus points every ``step`` along each segment (for label collision tests)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = [pts[:1]] if len(pts) else []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.linalg.norm(b - a) / max(step, 1e-9)), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out) if out else pts


def place_label(p, d, width_pt: float, fs: float, limits, per_pt: float, avoid: np.ndarray,
                hard: Optional[np.ndarray] = None):
    """((dx, dy) points offset, ha, va) of a label next to point p (map units, +z down).

    Candidates: the preferred side ``d`` (away from the route), the other compass
    sides, and above / below the point aligned left or right.  A label must stay
    on the map; among those, the one covering the fewest route points wins, then
    the one closest to ``d``.  With ``hard`` points (badges, leaders), a candidate
    covering any of them is out; None when no candidate is left.
    """
    x0, x1, z0, z1 = limits
    w, h, off = width_pt * per_pt, fs * 1.15 * per_pt, 4.0
    cands = []
    for a in np.radians(np.arange(0, 360, 45)):
        c = np.array([math.cos(a), math.sin(a)])
        ha = "right" if c[0] < -0.35 else ("left" if c[0] > 0.35 else "center")
        va = "top" if c[1] > 0.35 else ("bottom" if c[1] < -0.35 else "center")
        cands.append(((c[0] * off, -c[1] * off), ha, va, float(np.dot(c, d))))
    for dz, va in ((-1.0, "bottom"), (1.0, "top")):
        for ha in ("left", "right"):
            cands.append(((-2.0 if ha == "left" else 2.0, -dz * off), ha, va, -0.2))
    best, best_key = None, None
    for (ox, oy), ha, va, pref in cands:
        cx, cz = p[0] + ox * per_pt, p[1] - oy * per_pt
        left = cx - (w if ha == "right" else (0.0 if ha == "left" else w / 2))
        top = cz - (h if va == "bottom" else (0.0 if va == "top" else h / 2))
        inside = x0 + per_pt <= left and left + w <= x1 - per_pt and z0 + per_pt <= top and top + h <= z1 - per_pt

        def count(pts):
            return int(np.sum((pts[:, 0] >= left) & (pts[:, 0] <= left + w) & (pts[:, 1] >= top)
                              & (pts[:, 1] <= top + h))) if pts is not None and len(pts) else 0
        if hard is not None and (not inside or count(hard)):
            continue
        key = (not inside, count(avoid), -pref)
        if best_key is None or key < best_key:
            best, best_key = ((ox, oy), ha, va), key
    return best


# --------------------------------------------------------------------------- #
# Legend glyphs (point-unit axes)
# --------------------------------------------------------------------------- #
def legend_glyph(ax, key: str, x: float, y: float, w: float = 14.0) -> None:
    """Glyph of a legend entry, drawn in ``w`` points from x, centred on y."""
    if key == "hist":
        n = 14
        for i in range(n):
            ax.add_patch(Rectangle((x + i * w / n, y - 2.8), w / n + 0.05, 5.6, lw=0, ec="none",
                                   fc=cd.mix("white", HIST_COLOR, HEAT_ALPHA * (i + 0.5) / n)))
    elif key == "pred":
        pred_dot(ax, x + w / 2, y, clip_on=False)
    elif key == "gt":
        gt_ring(ax, x + w / 2, y, clip_on=False)
    elif key == "hit":
        gt_ring(ax, x + w / 2, y, clip_on=False)
        pred_dot(ax, x + w / 2, y, clip_on=False)
    elif key == "fut":
        for b, c in enumerate(FUT_BIN_COLORS):
            ax.add_patch(Rectangle((x + b * w / 4, y - 2.8), w / 4 - 0.4, 5.6, lw=0,
                                   fc=cd.mix("white", c, FUT_ALPHA)))
    elif key == "path":
        path_marks(ax, [x + 1.5 + 3.4 * i for i in range(4)], [y] * 4, clip_on=False)
    elif key == "goal":
        goal_ring(ax, x + w / 2, y)
    elif key == "actions":
        action_chip(ax, x, y, bd.FORWARD, size=6.4)
        action_chip(ax, x + 7.4, y, bd.LEFT, size=6.4)
    elif key == "route":
        ax.plot([x, x + w * 0.45], [y, y], color=style.INK, lw=0.95, solid_capstyle="round")
        ax.plot([x + w * 0.55, x + w], [y, y], color=style.MUTED, lw=0.8, ls=(0, (2.4, 1.4)))
    elif key == "start_goal":
        ax.plot([x + 2.5], [y], marker="o", ms=3.6, mfc="white", mec=style.INK, mew=0.8)
        ax.plot([x + 10.0], [y], marker="*", ms=6.0, mfc=style.INK, mec="white", mew=0.4)
    elif key == "stop":
        ax.plot([x, x + w / 2], [y, y], color=style.INK, lw=0.95, solid_capstyle="round")
        ax.plot([x + w / 2], [y], marker="s", ms=STOP_MS, mfc=style.INK, mec="white", mew=0.45)
    elif key == "key":
        cd.key_badge(ax, x + w / 2, y, "K1", fs=MIN_FS)
    elif key == "warmup":
        with matplotlib.rc_context({"hatch.linewidth": HATCH_LW}):
            ax.add_patch(Rectangle((x, y - 3.0), w, 6.0, fc=style.SURFACE, ec=HATCH_COLOR, lw=0.0, hatch=HATCH))
        ax.add_patch(Rectangle((x, y - 3.0), w, 6.0, fill=False, ec=style.AXIS, lw=HAIR))
    elif key == "nomap":
        ax.add_patch(Rectangle((x, y - 3.0), w, 6.0, fc=NOMAP_FILL, ec=style.AXIS, lw=HAIR))
    elif key == "turns":
        ax.plot([x, x + w], [y, y], color=style.GRID, lw=HAIR)
        for i, up in enumerate((True, True, False, True, False)):
            ax.add_patch(Rectangle((x + 1.0 + i * 2.8, y), 1.6, 3.2 if up else -3.2, fc=TURN_COLOR, ec="none"))
    elif key == "keyline":
        ax.plot([x + w / 2, x + w / 2], [y - 3.4, y + 3.4], color=style.INK, lw=0.45, solid_capstyle="butt")
    elif key == "frame":  # a strip in miniature: muted sides, the framed front in colour
        ax.add_patch(Rectangle((x, y - 3.0), w, 6.0, fc="#e4e3de", ec=style.AXIS, lw=HAIR))
        ax.add_patch(Rectangle((x + w * 0.36, y - 3.0), w * 0.28, 6.0, fc="#b7a58c", ec=style.INK, lw=0.6))
    elif key in ("slots", "slots_some", "slots_ov"):  # a timeline column in miniature: frames 1 .. 8 left to right
        ax.add_patch(Rectangle((x, y - 3.4), w, 6.8, fc=cd.mix("white", HIST_COLOR, 0.22), ec=style.AXIS, lw=HAIR))
        for i, (dx, dy) in enumerate(((0.2, -1.9), (0.5, 0.0), (0.8, 1.9))):
            gt_ring(ax, x + w * dx, y + dy, scale=0.72, clip_on=False)
            pred_dot(ax, x + w * dx, y + dy, scale=0.72, clip_on=False)
    elif key in ("slots_stacked", "slots_stacked_ov"):  # a call's marks stacked at its column's centre
        ax.add_patch(Rectangle((x + w * 0.3, y - 3.4), w * 0.4, 6.8, fc=cd.mix("white", HIST_COLOR, 0.22),
                               ec=style.AXIS, lw=HAIR))
        for dy in (-1.9, 0.0, 1.9):
            gt_ring(ax, x + w / 2, y + dy, scale=0.72, clip_on=False)
            pred_dot(ax, x + w / 2, y + dy, scale=0.72, clip_on=False)
    elif key == "pair":  # a past frame's predicted peak joined to its true direction (a strip in miniature)
        ax.plot([x + 3.0, x + w - 3.0], [y - 1.4, y + 1.4], color=PAIR_COLOR, lw=PAIR_LW, solid_capstyle="butt")
        pred_dot(ax, x + 3.0, y - 1.4, clip_on=False)
        gt_ring(ax, x + w - 3.0, y + 1.4, clip_on=False)
    elif key == "frames":  # a history-frame thumbnail with its slot badge
        ax.add_patch(Rectangle((x + 1.0, y - 3.4), w - 1.0, 6.8, fc="#cdd5dd", ec=style.AXIS, lw=HAIR))
        slot_badge(ax, x + 4.2, y + 0.4, 0, fs=MIN_FS, pad=SLOT_PAD_SMALL)
    else:
        raise KeyError(key)


def route_extent(route_xz, ref_xz, goal_xz, goal_radius: float, w_in: float, h_in: float) -> Tuple[float, float]:
    """(largest distance of the executed route from its start in m, the route's bounding box as a fraction of the
    route map's shorter side) for a route map ``w_in`` x ``h_in`` (the limits ``draw_route_map`` uses)."""
    route_xz = np.asarray(route_xz, dtype=np.float64).reshape(-1, 2)
    limits = pn.map_limits([route_xz, np.asarray(ref_xz, dtype=np.float64).reshape(-1, 2)], goal_xz, goal_radius,
                           h_in / w_in, pad=0.9)
    side = min(limits[1] - limits[0], limits[3] - limits[2])
    box = float((route_xz.max(0) - route_xz.min(0)).max()) if len(route_xz) else 0.0
    far = float(np.hypot(*(route_xz - route_xz[0]).T).max()) if len(route_xz) else 0.0
    return far, box / max(side, 1e-9)


ROUTE_SMALL_FRAC = 0.10  # a route whose bounding box is under this fraction of its map gets a note in words


def route_note_m(route_xz, ref_xz, goal_xz, goal_radius: float, w_in: float, h_in: float) -> Optional[float]:
    """For a route that would hide under the start ring and the badges (bounding box < ROUTE_SMALL_FRAC of the
    map): its largest distance from the start, rounded up to 0.1 m (for "stays within d m of the start");
    else None."""
    far, frac = route_extent(route_xz, ref_xz, goal_xz, goal_radius, w_in, h_in)
    if frac >= ROUTE_SMALL_FRAC:
        return None
    return math.ceil(far * 10.0 - 1e-9) / 10.0
