"""Matplotlib font configuration helpers for VLN visualizations.

This module centralizes Matplotlib font setup so that Chinese instructions and
other CJK glyphs render correctly in generated figures. It attempts to
register any project-provided fonts and then selects the first available font
from a list of CJK-friendly families. If none of the preferred fonts are
installed, Matplotlib falls back to its defaults and we log a warning so that
users know how to fix the issue (for example by installing Noto Sans CJK).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from matplotlib import font_manager, rcParams

logger = logging.getLogger(__name__)

# Preferred font families ordered from highest to lowest priority.
PREFERRED_CJK_FONTS: Sequence[str] = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "Source Han Sans CN",
    "Source Han Sans SC",
    "Source Han Sans",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Microsoft YaHei",
    "SimHei",
    "Sarasa Gothic SC",
    "Arial Unicode MS",
    "DejaVu Sans",
)

# Candidate directories that may contain additional fonts. Users can drop
# project-specific fonts into ./fonts to make them available without touching
# the system image.
DEFAULT_FONT_SEARCH_DIRS: Sequence[Path] = (
    Path(__file__).resolve().parents[2] / "fonts",
    Path.home() / ".fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
)

_configured: bool = False
_selected_font: Optional[str] = None


def _discover_font_files(font_dirs: Iterable[Path]) -> List[Path]:
    font_files: List[Path] = []
    for font_dir in font_dirs:
        try:
            if not font_dir.exists():
                continue
            for pattern in ("*.ttf", "*.otf", "*.ttc"):
                font_files.extend(font_dir.rglob(pattern))
        except (PermissionError, OSError) as exc:
            logger.debug("Skipping font directory %s: %s", font_dir, exc)
    return font_files


def _register_additional_fonts(extra_font_dirs: Optional[Iterable[Path]] = None) -> int:
    font_dirs = list(DEFAULT_FONT_SEARCH_DIRS)
    if extra_font_dirs:
        for extra_dir in extra_font_dirs:
            try:
                path = Path(extra_dir).expanduser().resolve()
            except RuntimeError:
                # Resolve may fail on some virtual paths; fall back to raw path.
                path = Path(extra_dir)
            font_dirs.append(path)

    registered = 0
    for font_path in _discover_font_files(font_dirs):
        try:
            font_manager.fontManager.addfont(str(font_path))
            registered += 1
        except Exception as exc:  # Matplotlib addfont can raise various errors
            logger.debug("Failed to register font %s: %s", font_path, exc)
    if registered:
        try:
            font_manager._rebuild()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Matplotlib font cache rebuild failed: %s", exc)
    return registered


def configure_matplotlib_fonts(extra_font_dirs: Optional[Iterable[Path]] = None) -> Optional[str]:
    """Ensure Matplotlib can render CJK glyphs and return the chosen font.

    The first available font from :data:`PREFERRED_CJK_FONTS` is used for the
    global ``sans-serif`` family. When no preferred fonts are found, we emit a
    warning so that users can install `fonts-noto-cjk` (or similar) and still
    allow Matplotlib to continue with its default fallback.
    """
    global _configured, _selected_font

    if _configured:
        return _selected_font

    registered = _register_additional_fonts(extra_font_dirs)
    if registered:
        logger.info("Registered %d additional Matplotlib font(s)", registered)

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in PREFERRED_CJK_FONTS:
        if font_name in available_fonts:
            current_family = list(rcParams.get("font.sans-serif", []))
            if font_name not in current_family:
                rcParams["font.sans-serif"] = [font_name, *current_family]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            _selected_font = font_name
            logger.info("Configured Matplotlib to use '%s' for sans-serif text", font_name)
            break
    else:
        logger.warning(
            "No preferred CJK fonts available. Chinese text may render as blanks; "
            "install a Unicode font such as 'Noto Sans CJK SC' or drop it into ./fonts/."
        )

    _configured = True
    return _selected_font


__all__ = [
    "configure_matplotlib_fonts",
    "PREFERRED_CJK_FONTS",
]
