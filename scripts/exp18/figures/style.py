"""Paper figure style for EXP-18: fonts, sizes, and the validated palette.

Colour roles (one meaning per hue across every EXP-18 figure; validated with
the dataviz palette checker on the light surface #fcfcfb):

* **Blue (#2a78d6 family) = ground truth / past positions only.**  History
  index k = 1..K is a single-hue blue ramp, oldest light -> newest dark.  Eight
  steps of one hue cannot all be told apart, so every history marker also
  carries its number; the ramp only adds a sense of "older vs newer".
* **Orange (#eb6834 family) = prediction only.**  The predicted affordance map
  is a single-hue orange ramp; ``PRED_INK`` rings a missed slot's number.
* Tiers A-E are never told apart by blue vs orange.  A chart whose marks are
  all predictions (the quantitative figure) draws every tier in the orange
  family: ``TIER_PRED_SHADES`` (an ordinal light -> dark ramp, passes the
  checker's ``--ordinal`` gates) together with ``TIER_MARKERS`` and direct
  labels; the constant always-behind guess is grey (``FLOOR_COLOR``).
* ``TIER_COLORS`` holds categorical tier hues outside the blue / orange
  families (aqua, yellow, magenta, green, violet; adjacent CVD dE >= 9.1).  The
  paper figures do not use them; they exist for non-prediction tier keys only
  and never for prediction marks.  Three sit below 3:1 contrast, so anything
  drawn in them is direct-labelled.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical tier hues, outside the ground-truth blue and prediction orange families (see the module
# doc: not for prediction marks; unused by the paper figures).
TIER_COLORS = {
    "A": "#1baf7a",  # aqua
    "B": "#eda100",  # yellow
    "C": "#e87ba4",  # magenta
    "D": "#008300",  # green
    "E": "#4a3aa7",  # violet
}
# Prediction marks per tier: one orange hue, light -> dark A..E (ordinal ramp, light end 2.41:1 on SURFACE).
TIER_PRED_SHADES = {
    "A": "#f08a55",
    "B": "#eb6834",
    "C": "#c94f1e",
    "D": "#9a3810",
    "E": "#6b2408",
}
TIER_MARKERS = {"A": "o", "B": "s", "C": "^", "D": "v", "E": "P"}  # never "D" (the diamond marks the baseline)
FLOOR_COLOR = "#c3c2b7"

GT_COLOR = "#2a78d6"  # ground truth / past positions
GT_INK = "#1c5cab"  # dark step of the ground-truth ramp (text, carets, keys)
PRED_COLOR = "#eb6834"  # prediction
PRED_INK = "#c94f1e"  # dark step of the prediction ramp (miss rings, leaders, text keys)

# Blue ramp 250 -> 700 from the reference palette.
_HISTORY_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
HISTORY_CMAP = LinearSegmentedColormap.from_list("exp18_history", _HISTORY_RAMP)

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "exp18_heat",
    [to_rgba("#eb6834", 0.0), to_rgba("#f08a55", 0.55), to_rgba("#eb6834", 0.85), to_rgba("#b53f12", 0.95), to_rgba("#7a2406", 1.0)],
)
# Opaque sequential version for standalone heat panels (no imagery beneath).
HEAT_CMAP_OPAQUE = LinearSegmentedColormap.from_list(
    "exp18_heat_opaque", ["#fcfcfb", "#fbd9c6", "#f4a57c", "#eb6834", "#b53f12", "#5e1c05"]
)

FONT_DIRS = (
    "/usr/share/fonts/opentype/urw-base35",
    os.environ.get("EXP18_FONT_DIR", ""),
)
LATIN_FAMILY = "Nimbus Sans"
CJK_CANDIDATES = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    os.environ.get("EXP18_CJK_FONT", ""),
)

# Double-column width (IEEE / most CV venues) and single-column width, inches.
WIDTH_DOUBLE = 7.0
WIDTH_SINGLE = 3.4


def history_color(k: int, num: int = 8):
    """Color of history slot k (0 = oldest) among ``num`` slots."""
    return HISTORY_CMAP(0.0 if num <= 1 else k / (num - 1))


def history_text_color(k: int, num: int = 8) -> str:
    return INK if k < num / 2 else "white"


def _register_fonts(lang: str) -> list:
    families = []
    for directory in FONT_DIRS:
        if not directory or not Path(directory).is_dir():
            continue
        for path in sorted(Path(directory).glob("NimbusSans-*.otf")):
            font_manager.fontManager.addfont(str(path))
    if any(f.name == LATIN_FAMILY for f in font_manager.fontManager.ttflist):
        families.append(LATIN_FAMILY)
    if lang == "zh":
        for candidate in CJK_CANDIDATES:
            if candidate and Path(candidate).is_file():
                font_manager.fontManager.addfont(candidate)
                families.append(font_manager.FontProperties(fname=candidate).get_name())
                break
    families.append("DejaVu Sans")
    return families


def apply(lang: str = "en") -> None:
    """Install the EXP-18 rcParams (call once before any figure)."""
    matplotlib.use("Agg")
    families = _register_fonts(lang)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.linewidth": 0.6,
            "axes.facecolor": SURFACE,
            "figure.facecolor": "white",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.unicode_minus": True,
        }
    )


def save(fig, stem: Path, formats=("pdf", "png")) -> list:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        fig.savefig(target)
        written.append(target)
    return written
