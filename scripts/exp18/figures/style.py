"""Paper figure style for EXP-18: fonts, sizes, and the validated palette.

Color roles (validated with the dataviz palette checker, light surface #fcfcfb):

* tiers A–E: categorical slots 1–5 in fixed order (blue, orange, aqua, yellow,
  magenta).  Three slots sit below 3:1 contrast, so tiers are always direct-
  labelled on an axis and never identified by color alone.
* pose arms: same tier color, filled marker = VO (deployed), hollow = GT pose.
* history index k = 1..K: a single-hue blue ramp, oldest light -> newest dark.
  Eight steps of one hue cannot all be told apart, so every history marker also
  carries its number; the ramp only adds a sense of "older vs newer".
* predicted heat: a single-hue orange ramp from transparent to deep red-orange,
  so it never collides with the blue history markers.
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

TIER_COLORS = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#1baf7a",
    "D": "#eda100",
    "E": "#e87ba4",
}
FLOOR_COLOR = "#c3c2b7"

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
