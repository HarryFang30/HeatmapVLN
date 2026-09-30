#!/usr/bin/env python3
"""EXP-18: render every paper figure from the metrics and the pre-registered cases, check the set, write a manifest.

Inputs (all overridable): ``<EXP_ROOT>/metrics/metrics.json`` (+ the ``slots``,
``episodes`` and ``rows`` tables) written by ``compute_metrics.py`` and
``<EXP_ROOT>/metrics/cases.json`` written by ``select_cases.py``; the dumps,
top-down maps and clips those point to.

Outputs in ``--out-dir`` (default ``<EXP_ROOT>/figures``), each figure in every
``--langs`` language (English files carry no suffix, the others ``_<lang>``):

  fig1_main_case             main case (fig_case): main-figure candidate ``--main-index`` (default 1)
                             of the pre-registered tier-C candidates, drawn at its key rows
  fig2_gallery               main-text gallery (fig_gallery, variant "main"): tiers C, D, E
  fig3_metrics               quantitative figure (fig_metrics)
  fig4_route_<pattern>       designed-route figures (fig_routes): out_and_back, loop (tier-E picks only)
  supp/candidate<k>_<scene>_<clip>   every pre-registered main-figure candidate (fig_case), titled
                             "Main-figure candidate k of n" with |episode PCK@8 - tier median|
  supp/figS_gallery_all      supplementary gallery (fig_gallery, variant "supp"): tiers A-E
  supp/anim_main_case        animation of the fig1 episode (fig_anim): .mp4 + .gif
  manifest.json              every file with its sha256, the code version, the sha256 of the inputs,
                             the main case, the conventions, the checks and their findings
  README.md                  the figures with their captions and the check results

Main-figure candidates (orchestrator decision D9).  cases.json ``main_figure``
lists the pre-registered tier-C candidates (>= 4 scored frames, path >= 8 m, a
frame whose ground-truth bearings span >= 90 deg; the top 5 by |episode PCK@8 -
tier median|).  All of them are drawn as supplementary figures, in rank order,
as pre-registered (no re-ranking, no de-duplication).  R2R gives each path
several instructions, so two candidates can be one path: the candidates sharing
a reference path (``trajectory_id`` from the dump metadata, else a hash of the
reference path) say so in their captions (and fig1 / the animation, when the
main case is one of them).  fig1 is candidate ``--main-index``, chosen after the
candidates are rendered; its caption says it is one of the pre-registered
candidates, all shown in the supplement.  No fallback: without a tier-C
candidate list fig1, the candidates and the animation are not drawn.

Checks (``lint`` in the manifest; every error fails the run, exit 1):
  * Figure policy (user decision): patterns ``POLICY_PATTERNS`` (pose / odometry
    wording and the retired names of the affordance map) are errors in every
    string a figure holds, the captions, the modules' label tables, README.md and
    manifest.json; ``WARN_PATTERNS`` (localisation claims) are warnings.  Figure
    text is gathered twice: every ``Text`` artist of a figure walked just before
    it is drawn (``Figure.draw``: savefig, and the animation's canvas draws) and
    every string actually drawn (``Text.draw``, animation blits included).  Only
    a pattern id and where it was found are stored, never the words.
  * Notes (D5): a module warning of the dropped-note / unaccounted-slot type, a
    non-empty ``notes_dropped`` field, or an unaccounted slot (the modules raise)
    fails that figure.
  * Numbers (D1): every drawn row's numbered misses equal the slots table's joint
    PCK@8 failures (tier + clip + row), and its hits / visible count equal the
    tables'; the gallery's frame per tile follows D8.
  * Elevation window (D4): every block's window is ``common_draw.elevation_window``
    of the rows it shows (recomputed from the dumps), and every caption of a figure
    with affordance-map rows states it.
  * Set consistency (D6): one frame-label format per language ("frame N of T",
    "第 N 帧（共 T 帧）"), one qualifier per view label, one wording of the 0 deg
    label, fig1's episode not a gallery tile.
  * Parentheses (D6): zh text uses full-width （） around or after Chinese text
    (figure text and captions; half-width only for a panel letter "(a)",
    coordinates and non-Chinese content); en text never full-width.
  * Colour (D7): no colour in the figure modules reads as the ground-truth blue or
    the prediction orange (hue within 20 deg) outside a ground-truth / prediction
    role.
  * Text below ``MIN_FONT_PT`` at print size (static figures).

Usage (repo root on PYTHONPATH; ``scripts/exp18/run_figures.sh`` wraps it):
  python -m scripts.exp18.figures.make_all [--exp-root DIR] [--metrics-dir DIR] [--cases FILE]
      [--out-dir DIR] [--langs en,zh] [--main-index 1] [--only fig1,supp,fig2,fig3,fig4,anim]
      [--dumps-root DIR] [--topdown-root DIR] [--clip-root DIR] [--anim-size 1920x1080] [--clean] [--draft]
Exit status: 0 when every attempted figure rendered and every check passed; 1 when a figure failed (an
exception, a dropped note or an unaccounted slot) or a check found an error; 2 on bad arguments.
``--draft`` (development) records check errors but does not let them set the exit status (a failed figure
still does).
"""
from __future__ import annotations

import argparse
import ast
import colorsys
import contextlib
import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402

SCHEMA = "exp18-figures-v3"
LANGS = ("en", "zh")
GROUPS = ("fig1", "supp", "fig2", "fig3", "fig4", "anim")  # --only
TIER_ORDER = "ABCDE"
MAIN_TIER = "C"  # main-figure candidates: unseen scenes (pre-registered)
ROUTE_PATTERNS = ("out_and_back", "loop")
GALLERY_STEMS = {"main": ("fig2", "fig2_gallery", "fig2_gallery", ""),  # variant -> (group, id, stem, subdir)
                 "supp": ("supp_gallery", "supp_gallery_all", "figS_gallery_all", "supp")}

TITLES = {
    "fig1": "Main case: predicted affordance map vs ground truth at key positions",
    "fig2": "Gallery (main text): unseen scenes, HM3D and designed routes",
    "fig3": "Accuracy of the predicted affordance map per tier",
    "fig4": "Designed routes: out-and-back and loop",
    "supp": "Supplement: every pre-registered main-figure candidate",
    "supp_gallery": "Supplement: gallery of all five tiers",
    "anim": "Supplement: animation of the main case",
}
PATTERN_TITLES = {"out_and_back": "out-and-back", "loop": "loop"}

# --------------------------------------------------------------------------- #
# Figure policy and the set's conventions
# --------------------------------------------------------------------------- #
POLICY_ID = "exp18-figure-policy-v3"
# (id, regex), case-insensitive.  Errors anywhere in figure text, captions, label tables, README.md and
# manifest.json.  Only the ids are ever written out (README.md, manifest.json), never the words.
POLICY_PATTERNS = (
    ("P1", r"pose|VO\b|odometr|AMB3R|SLAM|odom|位姿|姿态|里程|heat ?row|heatmap|热力"),  # orchestrator's list
    ("P2", r"(?:\bGT|真值)[- ]?(?:arms?|臂)"),  # the second prediction arm
    ("P3", r"\bheat[- ](?:rows?|maps?|strips?|lines?)\b|热图"),  # other spellings of the retired map names
)
WARN_PATTERNS = (("W1", r"locali[sz]|定位"),)  # localisation claims: warnings, for manual review
# The project's own name (repository and workspace paths in README.md / manifest.json) is not figure wording:
# it is masked before the scan, so only that exact token is exempt.
PROJECT_NAME_RE = re.compile(r"HeatmapVLN", re.I)
_POLICY_C = [(i, re.compile(p, re.I)) for i, p in POLICY_PATTERNS]
_WARN_C = [(i, re.compile(p, re.I)) for i, p in WARN_PATTERNS]
POLICY_SHA256 = hashlib.sha256(json.dumps([POLICY_PATTERNS, WARN_PATTERNS], ensure_ascii=False)
                               .encode("utf-8")).hexdigest()
# Written to README.md / manifest.json: neutral on purpose (they must not repeat the words they stand for).
PATTERN_NOTES = {"P1": "orchestrator word list", "P2": "second-arm wording", "P3": "other spellings of the retired "
                 "map names", "W1": "claims of where the robot is (warning)"}

GT_BLUE, PRED_ORANGE = "#2a78d6", "#eb6834"
COLOUR_ROLES = {  # role -> (reference colour, name tokens that make the colour legitimate)
    "ground truth": (GT_BLUE, {"gt", "truth", "ground", "hist", "history"}),
    "prediction": (PRED_ORANGE, {"pred", "prediction", "predicted", "heat", "miss"}),  # miss rings are orange (D2)
}
HUE_TOL_DEG, MIN_SAT, MIN_VAL = 20.0, 0.25, 0.2
MIN_FONT_PT = 5.5  # nothing smaller at print size (static figures are drawn at their print size)
FONT_TOL_PT = 0.05
ZH_PARENS = ("zh: full-width （） around or after Chinese text, in figure text and captions (half-width only for a "
             "panel letter (a), coordinates and non-Chinese content); en: never full-width")
COLOUR_MODULES = ("style", "common_draw", "fig_case", "fig_gallery", "fig_metrics", "fig_routes", "fig_anim")
LABEL_MODULES = ("data", "common_draw", "fig_case", "fig_gallery", "fig_metrics", "fig_routes", "fig_anim")
ROW_MODULES = ("fig_case", "fig_gallery", "fig_anim")  # modules that draw affordance-map rows themselves

CJK = r"㐀-鿿豈-﫿"
_CJK_RE = re.compile(f"[{CJK}]")
_HALF_PAREN_RE = re.compile(r"\(([^()]*)\)")
_PANEL_RE = re.compile(r"[a-h]")
_TUPLE_RE = re.compile(r"[-−]?\d+(?:\.\d+)?(?:\s*,\s*[-−]?\d+(?:\.\d+)?)+")  # coordinates (32, 32)
FRAME_FORMATS = {  # most specific first; each match is removed before the next pattern runs
    "en": (("frame N of T", r"\bframe\s+\d+\s+of\s+\d+"), ("frame N / T", r"\bframe\s+\d+\s*/\s*\d+"),
           ("frame N", r"\bframe\s+\d+\b(?![.,]\d|\s*°|°)")),  # not "this frame 1.3°" (an error value)
    "zh": (("第 N 帧（共 T 帧）", r"第\s*\d+\s*帧\s*（\s*共\s*\d+\s*帧\s*）"), ("第 N / T 帧", r"第\s*\d+\s*/\s*\d+\s*帧"),
           ("第 N 帧 / T", r"第\s*\d+\s*帧\s*/\s*\d+"), ("第 N 帧 (共 T 帧)", r"第\s*\d+\s*帧\s*\(\s*共\s*\d+\s*帧\s*\)"),
           ("第 N 帧", r"第\s*\d+\s*帧")),
}
FRAME_FORMAT_D6 = {"en": "frame N of T", "zh": "第 N 帧（共 T 帧）"}
VIEW_LABEL_RE = {"en": re.compile(r"^(Front|Right|Back|Left)\s*·\s*(.+)$"),
                 "zh": re.compile(r"^(前|右|后|左)\s*·\s*(.+)$")}
AHEAD_RE = re.compile(r"^0°\s*[(（]\s*(.+?)\s*[)）]$")
ELEV_WORD = {"en": "elevation", "zh": "仰角"}  # a caption stating a window names it ...
ELEV_RANGE = {"en": "{lo} to {hi}", "zh": "{lo} 至 {hi}"}  # ... and spells every window drawn like this

# Module warnings that mean a slot is not accounted for in the figure (D5): these fail the figure.
DEFECT_RE = re.compile(r"neither a badge nor a note|unaccounted|no badge (?:or|nor) (?:a )?note|no room for the note"
                       r"|notes? (?:were |was )?(?:dropped|not drawn|left out|omitted)|dropped notes?", re.I)
DROP_KEYS = ("notes_dropped", "dropped_notes", "notes_not_drawn")
_QUIET_WARNINGS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning, ResourceWarning, ImportWarning)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _raise(msg: str):
    raise RuntimeError(msg)


def lang_stem(name: str, lang: str) -> str:
    return name if lang == "en" else f"{name}_{lang}"


@contextlib.contextmanager
def python_warnings(raised: List[str]):
    """Inside the block: keep the Python warnings a figure module raises (not deprecation noise) in ``raised``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for w in rec:
                if issubclass(w.category, _QUIET_WARNINGS):
                    continue
                text = f"{w.category.__name__}: {w.message}"
                if text not in raised:
                    raised.append(text)
                    print(f"[make_all]   {text} ({Path(w.filename).name}:{w.lineno})", flush=True)


class TextSpy:
    """Gathers figure text while ``current`` / ``walked`` are set.

    * ``walked``: every ``Text`` artist of a figure (visible or not, tick labels and annotations included),
      collected by walking the figure just before it is drawn (``Figure.draw``: every savefig, and each
      canvas draw of the animation).
    * ``current``: every string actually drawn (``Text.draw``, which the animation's blits also pass
      through) -> the smallest font size (pt) it was drawn at.
    """

    def __init__(self):
        self.current: Optional[Dict[str, float]] = None
        self.walked: Optional[Set[str]] = None
        self._orig: Optional[tuple] = None

    def __enter__(self):
        from matplotlib.figure import Figure
        from matplotlib.text import Text

        orig_text, orig_fig = Text.draw, Figure.draw
        self._orig = (orig_text, orig_fig)
        spy = self

        @functools.wraps(orig_text)
        def text_draw(text, renderer, *args, **kwargs):
            rec = spy.current
            if rec is not None:
                try:
                    s = text.get_text()
                    if text.get_visible() and s and s.strip():
                        fs = float(text.get_fontsize())
                        if fs < rec.get(s, float("inf")):
                            rec[s] = fs
                except Exception:  # recording must never break a figure
                    pass
            return orig_text(text, renderer, *args, **kwargs)

        @functools.wraps(orig_fig)
        def figure_draw(fig, renderer, *args, **kwargs):
            seen = spy.walked
            if seen is not None:
                try:
                    for t in fig.findobj(Text):
                        s = t.get_text()
                        if s and s.strip():
                            seen.add(s)
                except Exception:
                    pass
            return orig_fig(fig, renderer, *args, **kwargs)

        Text.draw = text_draw
        Figure.draw = figure_draw
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            from matplotlib.figure import Figure
            from matplotlib.text import Text

            Text.draw, Figure.draw = self._orig
        return False


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def input_record(path: Optional[Path]) -> dict:
    if path is None:
        return {"path": None, "sha256": None}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "sha256": None, "missing": True}
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def table_record(stem: Path) -> dict:
    for suffix in (".parquet", ".csv.gz"):
        if stem.with_suffix(suffix).is_file():
            return input_record(stem.with_suffix(suffix))
    return {"path": str(stem.with_suffix(".parquet")), "sha256": None, "missing": True}


def resolve_git_sha(repo: Path) -> dict:
    """EXP18_GIT_SHA, else ``<repo>/.exp18_git_sha`` (staged archives have no .git), else ``git rev-parse``."""
    sha = os.environ.get("EXP18_GIT_SHA", "").strip()
    if sha:
        return {"sha": sha, "source": "EXP18_GIT_SHA"}
    marker = repo / ".exp18_git_sha"
    if marker.is_file():
        return {"sha": marker.read_text().strip(), "source": str(marker)}
    try:
        git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True)
        status = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, timeout=20, check=True)
        return {"sha": head.stdout.strip(), "dirty": bool(status.stdout.strip()), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "source": "none (no EXP18_GIT_SHA, no .exp18_git_sha, no git)"}


def code_digest(repo: Path) -> dict:
    """sha256 over the EXP-18 Python sources (identifies the code even from a dirty tree or an archive)."""
    root = repo / "scripts" / "exp18"
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        h.update(str(p.relative_to(repo)).encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return {"sha256": h.hexdigest(), "files": len(files), "root": str(root.relative_to(repo))}


_KEY_DROP = {"gt", "gt_pck8", "gt_bearing_err_median", "gt_pck8_row", "gt_pck4"}  # stays in metrics.json
_KEY_RENAME = {"vo": "prediction", "floor": "always_behind"}


def public(obj):
    """Manifest view of figure-module stats: the prediction is ``prediction``, the second arm left out."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k = str(k)
            if k in _KEY_DROP:
                continue
            k = _KEY_RENAME.get(k, k)
            k = re.sub(r"(^|_)vo(_|$)", r"\1pred\2", k)
            k = re.sub(r"(^|_)floor(_|$)", r"\1always_behind\2", k)
            out[k] = public(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [public(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"\bVO arm\b|\bVO\b|'vo'", "prediction", obj)
    if isinstance(obj, Path):
        return str(obj)
    if type(obj).__module__ == "numpy":  # numpy scalars and arrays
        return public(obj.tolist())
    if isinstance(obj, float) and obj != obj:  # NaN -> null (strict JSON)
        return None
    return obj


PICK_FIELDS = ("tier", "scene", "clip", "clip_key", "episode_id", "npz_path", "rank", "percentile", "n_scored_rows",
               "n_visible_slots", "path_length_m", "max_row_span_deg", "vo_pck8", "floor_pck8",
               "vo_bearing_err_median", "abs_diff_from_tier_median", "path_key", "same_path_as")


def pick_summary(pick: dict) -> dict:
    return public({k: pick[k] for k in PICK_FIELDS if k in pick})


def feedback(res) -> Tuple[List[str], List[str]]:
    """(warnings, defects) a figure module returned: any ``warnings`` / ``problems`` list and any
    ``notes_dropped`` / ``dropped_notes`` / ``notes_not_drawn`` field, at any depth, tagged with the
    enclosing entry's tier / percentile / key / frame.  Defects: every dropped note, and every warning of the
    dropped-note / unaccounted-slot type (``DEFECT_RE``)."""
    warns: List[str] = []
    defects: List[str] = []

    def tag(d: dict) -> str:
        parts = [f"{k} {d[k]}" for k in ("tier", "percentile", "pattern", "key", "frame")
                 if isinstance(d.get(k), (str, int)) and not isinstance(d.get(k), bool)]
        return " ".join(parts)

    def walk(obj):
        if isinstance(obj, dict):
            t = tag(obj)
            for k, v in obj.items():
                if k in ("warnings", "problems") and isinstance(v, (list, tuple)):
                    warns.extend(f"{t}: {x}" if t else str(x) for x in v if x)
                elif k in DROP_KEYS:
                    if isinstance(v, (list, tuple)):
                        defects.extend(f"{t}: note not drawn: {x}" if t else f"note not drawn: {x}"
                                       for x in v if x is not None)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool) and v:
                        defects.append(f"{t}: {int(v)} note(s) not drawn" if t else f"{int(v)} note(s) not drawn")
                else:
                    walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(res)
    warns = list(dict.fromkeys(warns))
    defects += [w for w in warns if DEFECT_RE.search(w)]
    return warns, list(dict.fromkeys(defects))


def edit_caption(files: Iterable[str], lang: str, prefix: str = "", suffix: str = "") -> None:
    """Put ``prefix`` in front of / ``suffix`` after the caption a figure module wrote (``*_caption.txt``); zh
    sentences are joined without a space."""
    if not (prefix or suffix):
        return
    sep = "" if lang == "zh" else " "
    for f in files:
        if str(f).endswith("_caption.txt") and Path(f).is_file():
            body = Path(f).read_text(encoding="utf-8").strip()
            Path(f).write_text(sep.join(s.strip() for s in (prefix, body, suffix) if s) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    group: str
    fig_id: str
    lang: str
    status: str  # ok | skipped | failed
    files: List[str] = field(default_factory=list)  # absolute paths
    reason: str = ""
    error: str = ""
    details: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    defects: List[str] = field(default_factory=list)  # dropped notes / unaccounted slots (fail the figure)
    texts: Dict[str, float] = field(default_factory=dict)  # every string the figure drew -> smallest pt
    walked: Set[str] = field(default_factory=set)  # every Text artist's string, walked before drawing
    blocks: List[dict] = field(default_factory=list)  # affordance-map blocks drawn: rows and windows (checks)
    seconds: float = 0.0


@dataclass
class Context:
    exp_root: Path
    metrics_dir: Path
    cases_path: Path
    out_dir: Path
    langs: List[str]
    groups: List[str]
    main_index: int
    dumps_root: Optional[Path]
    topdown_root: Path
    clip_root: Optional[Path]
    anim_size: Tuple[int, int]
    draft: bool = False
    cases: Optional[dict] = None
    metrics: Optional[dict] = None
    entries: List[Entry] = field(default_factory=list)
    main_case: dict = field(default_factory=dict)
    dumps_used: Dict[str, str] = field(default_factory=dict)  # npz path -> role
    spy: TextSpy = field(default_factory=TextSpy)
    _tables: Optional[dict] = None

    def run(self, group: str, fig_id: str, lang: str, fn: Callable[[], Tuple[List[str], dict]]) -> Entry:
        t0 = time.time()
        print(f"[make_all] {fig_id} ({lang}) ...", flush=True)
        raised: List[str] = []
        texts: Dict[str, float] = {}
        walked: Set[str] = set()
        self.spy.current, self.spy.walked = texts, walked
        try:
            with python_warnings(raised):
                files, details = fn()
            e = Entry(group, fig_id, lang, "ok", files=[str(f) for f in files], details=details)
        except Exception as exc:  # one figure failing must not stop the others
            tb = traceback.format_exc().strip().splitlines()
            e = Entry(group, fig_id, lang, "failed", error=f"{type(exc).__name__}: {exc}", details={"traceback": tb[-12:]})
            print(f"[make_all] {fig_id} ({lang}) FAILED: {e.error}\n" + "\n".join(tb[-12:]), flush=True)
        finally:
            self.spy.current = self.spy.walked = None
        e.seconds = round(time.time() - t0, 1)
        e.texts, e.walked = texts, walked
        own = list(e.details.pop("warnings", []))
        fb_warn, fb_defects = e.details.pop("_feedback", ([], []))
        e.blocks = e.details.pop("_blocks", [])
        e.warnings = list(dict.fromkeys(public(w) for w in own + fb_warn + raised))
        e.defects = list(dict.fromkeys(public(w) for w in fb_defects + [w for w in own + raised if DEFECT_RE.search(w)]))
        if e.status == "ok" and e.defects:
            # a slot left without a badge or a note is unaccounted for in the figure: the figure is not usable
            e.status = "failed"
            e.error = (f"{len(e.defects)} slot(s) unaccounted for (dropped note / no badge): " + "; ".join(e.defects[:3])
                       + (" ..." if len(e.defects) > 3 else ""))
        self.entries.append(e)
        print(f"[make_all] {fig_id} ({lang}) {e.status} in {e.seconds:.1f} s"
              + (f" ({e.error})" if e.status == "failed" and e.defects else ""), flush=True)
        return e

    def skip(self, group: str, fig_id: str, reason: str, langs: Optional[List[str]] = None) -> None:
        for lang in langs or self.langs:
            self.entries.append(Entry(group, fig_id, lang, "skipped", reason=reason))
        print(f"[make_all] {fig_id}: skipped ({reason})", flush=True)

    def fail(self, group: str, fig_id: str, reason: str) -> None:
        for lang in self.langs:
            self.entries.append(Entry(group, fig_id, lang, "failed", error=reason))
        print(f"[make_all] {fig_id} FAILED: {reason}", flush=True)

    def stem(self, name: str, lang: str, sub: str = "") -> Path:
        d = self.out_dir / sub if sub else self.out_dir
        return d / lang_stem(name, lang)

    def tables(self) -> Optional[dict]:
        """{"slots", "rows", "episodes"} written by compute_metrics.py, or None when one is missing."""
        if self._tables is None:
            try:
                from scripts.exp18.compute_metrics import read_table

                self._tables = {name: read_table(self.metrics_dir / name) for name in ("slots", "rows", "episodes")}
            except Exception as exc:  # no tables: the number checks report it
                print(f"[make_all] no slots/rows/episodes tables in {self.metrics_dir}: {type(exc).__name__}: {exc}")
                self._tables = {}
        return self._tables or None


# --------------------------------------------------------------------------- #
# Main case: the pre-registered candidates, which one is fig1, which rows
# --------------------------------------------------------------------------- #
def resolve_dump(ctx: Context, pick: dict) -> Path:
    from scripts.exp18.figures import fig_routes as fr

    return Path(fr.resolve_dump(pick, str(ctx.dumps_root) if ctx.dumps_root else None))


def path_key(ctx: Context, pick: dict) -> str:
    """The episode's reference path: ``<scene>:trajectory <id>`` from the dump metadata, else a hash of
    the reference path, else the clip itself (then nothing is matched with it)."""
    import numpy as np

    scene = pick.get("scene") or str(pick.get("clip_key", "")).split("/")[0]
    try:
        with np.load(resolve_dump(ctx, pick), allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    except Exception:
        meta = {}
    traj = meta.get("trajectory_id")
    if traj not in (None, ""):
        return f"{scene}:trajectory {traj}"
    ref = meta.get("reference_path")
    if ref:
        pts = np.round(np.asarray(ref, dtype=float).reshape(-1, 3), 2)
        return f"{scene}:path {hashlib.sha1(pts.tobytes()).hexdigest()[:12]}"
    return f"{pick.get('tier')}:{pick.get('clip_key')}"


def gallery_keys(cases: dict, tier: str) -> List[str]:
    g = (cases.get("gallery") or {}).get(tier) or {}
    return [p.get("clip_key") for p in g.get("picks") or []]


def choose_main_case(ctx: Context) -> dict:
    """The pre-registered candidates (all of them, rank order) and the fig1 episode (candidate ``main_index``)."""
    mf = (ctx.cases or {}).get("main_figure") or {}
    if mf.get("status") != "ok" or not mf.get("selected"):
        return {"source": None, "reason": f"cases.json main_figure status '{mf.get('status', 'absent')}' with "
                                          f"{len(mf.get('selected') or [])} candidates"}
    tier = mf.get("tier")
    if tier != MAIN_TIER:
        return {"source": None, "error": True,
                "reason": f"cases.json main-figure candidates are tier {tier!r}; the main figure is drawn only from "
                          f"tier {MAIN_TIER} (unseen scenes)"}
    selected = sorted(mf["selected"], key=lambda s: int(s.get("rank", 0)))
    bad = [c.get("clip_key") for c in selected if c.get("tier") != MAIN_TIER]
    if bad:
        return {"source": None, "error": True, "reason": f"candidates not from tier {MAIN_TIER}: {bad}"}
    keys = [path_key(ctx, c) for c in selected]
    cands = []
    for c, k in zip(selected, keys):
        twins = [int(o["rank"]) for o, ko in zip(selected, keys) if ko == k and o is not c]
        cands.append(dict(c, path_key=k, same_path_as=twins))
    choice = {"source": "main_candidates", "tier": tier, "candidates_source": "cases.json main_figure (pre-registered)",
              "tier_median": mf.get("tier_median_episode_vo_pck8"), "n_meeting_criteria": mf.get("n_candidates"),
              "n_episodes": mf.get("n_episodes"), "criteria": mf.get("criteria") or {},
              "candidates": cands, "n_candidates": len(cands),
              "shared_paths": sorted({tuple(sorted([int(c["rank"])] + c["same_path_as"])) for c in cands
                                      if c["same_path_as"]})}
    if not 1 <= ctx.main_index <= len(cands):
        choice.update(source=None, error=True,
                      reason=f"--main-index {ctx.main_index}: there are {len(cands)} main-figure candidates")
        return choice
    pick = cands[ctx.main_index - 1]
    choice.update(pick=pick, rank=int(pick["rank"]),
                  note=f"candidate {ctx.main_index} of {len(cands)} pre-registered tier-{tier} main-figure candidates "
                       f"(--main-index {ctx.main_index})")
    return choice


def case_rows(dump, pick: dict) -> Tuple[List[int], str]:
    """Dump rows to draw: the pick's pre-registered key rows (checked against the dump), else fig_case's rule."""
    from scripts.exp18.figures import data as dd
    from scripts.exp18.figures import fig_case as fc

    key_rows = pick.get("key_rows") or []
    if not key_rows:
        return dd.key_rows(dump, fc.ARM), "fig_case rule: first scored frame, widest ground-truth bearing span, last frame"
    t = dump.arrays["current_frame_ids"]
    rows = []
    for k in key_rows:
        i = int(k["row"])
        if not 0 <= i < dump.n_rows or int(t[i]) != int(k["t"]):
            raise ValueError(f"{dump.path}: key row {k} does not match the dump (frame ids {t.tolist()})")
        rows.append(i)
    if len(set(rows)) == len(rows):
        return rows, "pre-registered key rows (first scored frame, widest ground-truth bearing span, last frame)"
    alt = dd.key_rows(dump, fc.ARM)
    return alt, (f"pre-registered key rows {rows} coincide (widest span at the first or last frame); the middle "
                 f"scored frame replaces the duplicate (fig_case rule, stated in the caption): {alt}")


def prepare_main_case(ctx: Context) -> None:
    if ctx.cases is None:
        ctx.main_case = {"source": None, "reason": "no cases.json"}
        return
    try:
        choice = ctx.main_case = choose_main_case(ctx)
    except Exception as exc:  # fig1, the candidates and the animation fail with the reason
        traceback.print_exc()
        ctx.main_case = {"source": None, "error": True, "candidates": [],
                         "reason": f"choosing the main case failed: {type(exc).__name__}: {exc}"}
        return
    if choice.get("source") is None:
        return
    from scripts.exp18.figures import data as dd

    pick = choice["pick"]
    try:
        npz = resolve_dump(ctx, pick)
        dump = dd.load_dump(npz)
        rows, rule = case_rows(dump, pick)
    except Exception as exc:  # recorded; fig1 and the animation fail with it
        choice["prepare_error"] = f"{type(exc).__name__}: {exc}"
        return
    choice.update(npz=str(npz), rows=rows, rows_rule=rule,
                  frames=[int(dump.arrays["current_frame_ids"][i]) for i in rows])
    ctx.dumps_used[str(npz)] = "main case"


# Titles / caption sentences of the candidates and the main case (en, zh; zh full-width （）, D6).
CAND_TITLE = {"en": "Main-figure candidate {k} of {n}", "zh": "主图候选 {k}（共 {n} 个）"}
# |episode PCK@8 - tier median| in percentage points (cases.json stores fractions: 0.0157 -> "1.6 points")
CAND_NOTE = {"en": "(|episode PCK@8 − tier median| = {dpp:.1f} points)",
             "zh": "|该集 PCK@8 − 该层中位数| = {dpp:.1f} 个百分点"}
CAND_NOTE_PATH = {"en": " · same R2R path as candidate {others}", "zh": " · 与候选 {others} 为同一条 R2R 路径"}
CAND_CAPTION = {
    "en": "Supplementary figure: main-figure candidate {k} of {n} (episode PCK@8 {ep:.1f}%, tier median {med:.1f}%: "
          "|episode PCK@8 − tier median| = {dpp:.1f} points). ",
    "zh": "补充图：主图候选 {k}（共 {n} 个；该集 PCK@8 {ep:.1f}%，该层中位数 {med:.1f}%，|该集 PCK@8 − 该层中位数| = "
          "{dpp:.1f} 个百分点）。",
}
CRITERIA = {
    "en": "The {n} pre-registered candidates are the {tier_name} episodes whose episode PCK@8 is closest to the tier "
          "median, among those with ≥ {min_rows} scored frames, a path ≥ {min_path:g} m and a frame whose past "
          "positions span ≥ {min_span:g}° of bearing.",
    "zh": "{n} 个预注册候选是{tier_name}中满足 ≥ {min_rows} 个评分帧、路径 ≥ {min_path:g} m、且至少一帧的历史位置方位跨度 "
          "≥ {min_span:g}° 的集里，该集 PCK@8 最接近该层中位数的 {n} 集。",
}
SAME_PATH = {
    "en": "Candidates {ranks} are {count} instructions for one R2R path (episodes {eps}): the route and the images "
          "are the same, and the predictions differ slightly.",
    "zh": "候选 {ranks} 是同一条 R2R 路径的{count}条指令（第 {eps} 集）：路线与图像相同，预测略有差异。",
}
MAIN_NOTE = {
    "en": "The episode is one of the {n} pre-registered main-figure candidates, all {n} shown in the supplement.",
    "zh": "该集是 {n} 个预注册主图候选之一，全部 {n} 个见补充材料。",
}
MAIN_TWIN = {
    "en": "Episode {ep} (supplementary candidate {k}) is another instruction for the same R2R path.",
    "zh": "第 {ep} 集（补充材料候选 {k}）是同一条 R2R 路径的另一条指令。",
}
COUNT_WORD = {"en": {2: "two", 3: "three", 4: "four", 5: "five"}, "zh": {2: "两", 3: "三", 4: "四", 5: "五"}}
TIER_PLURAL = {"en": {"A": "training-scene", "B": "held-out-scene", "C": "unseen-scene", "D": "HM3D",
                      "E": "designed-route"},
               "zh": {"A": "训练场景", "B": "留出场景", "C": "未见场景（C 层）", "D": "HM3D 场景", "E": "设计路线"}}


def _join(items: List[str], lang: str) -> str:
    if lang == "zh":
        return items[0] if len(items) == 1 else "、".join(items[:-1]) + " 与 " + items[-1]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def same_path_sentence(mc: dict, pick: dict, lang: str) -> str:
    """"Candidates 2 and 4 are two instructions for one R2R path (episodes 1528 and 1529)." or ""."""
    if not pick.get("same_path_as"):
        return ""
    ranks = sorted([int(pick["rank"])] + [int(r) for r in pick["same_path_as"]])
    by_rank = {int(c["rank"]): c for c in mc.get("candidates") or []}
    eps = [str(by_rank[r].get("episode_id")) for r in ranks if r in by_rank]
    sep = "、" if lang == "zh" else None
    return SAME_PATH[lang].format(ranks=_join([str(r) for r in ranks], lang),
                                  count=COUNT_WORD[lang].get(len(ranks), str(len(ranks))),
                                  eps=sep.join(eps) if sep else _join(eps, lang))


def criteria_sentence(mc: dict, lang: str) -> str:
    cr = mc.get("criteria") or {}
    return CRITERIA[lang].format(n=mc.get("n_candidates"), tier_name=TIER_PLURAL[lang].get(mc.get("tier"), mc.get("tier")),
                                 min_rows=int(cr.get("min_rows", 4)), min_path=float(cr.get("min_path_m", 8.0)),
                                 min_span=float(cr.get("min_span_deg", 90.0)))


def candidate_label(mc: dict, pick: dict, lang: str) -> dict:
    """Title, title note and caption prefix of a supplementary candidate figure (D9)."""
    k, n = int(pick.get("rank", 0)), int(mc.get("n_candidates", 0))
    dpp = 100.0 * float(pick.get("abs_diff_from_tier_median") or 0.0)  # percentage points
    med = 100.0 * float(mc.get("tier_median") or 0.0)
    ep = 100.0 * float(pick.get("vo_pck8") or 0.0)
    note = CAND_NOTE[lang].format(dpp=dpp)
    if pick.get("same_path_as"):
        note += CAND_NOTE_PATH[lang].format(others=_join([str(r) for r in pick["same_path_as"]], lang))
    cap = CAND_CAPTION[lang].format(k=k, n=n, dpp=dpp, med=med, ep=ep) + criteria_sentence(mc, lang)
    sp = same_path_sentence(mc, pick, lang)
    if sp:
        cap += ("" if lang == "zh" else " ") + sp
    return {"title": CAND_TITLE[lang].format(k=k, n=n), "title_note": note, "caption": cap}


def main_case_suffix(mc: dict, lang: str) -> str:
    """Sentences appended to the captions of fig1 and the animation: one of the candidates; a twin path."""
    pick = mc["pick"]
    parts = [MAIN_NOTE[lang].format(n=mc.get("n_candidates"))]
    by_rank = {int(c["rank"]): c for c in mc.get("candidates") or []}
    for r in pick.get("same_path_as") or []:
        parts.append(MAIN_TWIN[lang].format(ep=by_rank[int(r)].get("episode_id"), k=int(r)))
    return ("" if lang == "zh" else " ").join(parts)


# --------------------------------------------------------------------------- #
# Figure jobs
# --------------------------------------------------------------------------- #
def job_case(ctx: Context, pick: dict, rows: List[int], stem: Path, lang: str, rule: str,
             label: Optional[dict] = None, caption_suffix: str = ""):
    from scripts.exp18.figures import fig_case as fc

    npz = resolve_dump(ctx, pick)
    opts = fc.CaseOptions(title=label["title"] if label else None, title_note=label["title_note"] if label else None)
    res = fc.make_case_figure(str(npz), rows=rows, topdown_root=str(ctx.topdown_root),
                              clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                              out_stem=str(stem), lang=lang, options=opts)
    edit_caption(res["files"], lang, prefix=label["caption"] if label else "", suffix=caption_suffix)
    details = {"case": pick_summary(pick), "rows": res["rows"], "rows_rule": rule,
               "frames": [s["frame"] for s in res["stats"]], "stats": public(res["stats"]),
               "windows": res.get("windows"), "layout_decisions": public(res.get("layout")),
               "size_in": [round(v, 3) for v in res["size_in"]], "_feedback": feedback(res),
               "_blocks": [{"npz": str(npz), "tier": pick.get("tier"), "clip_key": pick.get("clip_key"),
                            "rows": [int(s["row"])], "window": list(s["window"]), "stats": [public(s)]}
                           for s in res["stats"]]}
    return res["files"], details


def run_fig1(ctx: Context) -> None:
    mc = ctx.main_case
    if mc.get("source") is None:
        if mc.get("error"):
            ctx.fail("fig1", "fig1_main_case", mc["reason"])
        else:
            ctx.skip("fig1", "fig1_main_case", mc.get("reason", "no main case"))
        return
    for lang in ctx.langs:
        if mc.get("prepare_error"):
            ctx.run("fig1", "fig1_main_case", lang, lambda: _raise(mc["prepare_error"]))
            continue
        ctx.run("fig1", "fig1_main_case", lang,
                lambda lang=lang: job_case(ctx, mc["pick"], mc["rows"], ctx.stem("fig1_main_case", lang), lang,
                                           mc["rows_rule"], caption_suffix=main_case_suffix(mc, lang)))


def run_supp(ctx: Context) -> None:
    mc = ctx.main_case
    cands = mc.get("candidates") or []
    if not cands:
        reason = "no main-figure candidates" + (f": {mc['reason']}" if mc.get("reason") else "")
        if mc.get("error"):
            ctx.fail("supp", "supp_candidates", reason)
        else:
            ctx.skip("supp", "supp_candidates", reason)
        return
    from scripts.exp18.figures import data as dd

    for c in cands:
        rank = int(c["rank"])
        name = f"candidate{rank}_{c.get('scene', 'scene')}_{c.get('clip', 'clip')}"
        fig_id = f"supp_candidate{rank}"
        prep: dict = {}
        try:
            npz = resolve_dump(ctx, c)
            prep["rows"], prep["rule"] = case_rows(dd.load_dump(npz), c)
            ctx.dumps_used.setdefault(str(npz), f"main-figure candidate {rank}")
        except Exception as exc:
            prep["error"] = f"{type(exc).__name__}: {exc}"
        for lang in ctx.langs:
            if "error" in prep:
                ctx.run("supp", fig_id, lang, lambda: _raise(prep["error"]))
                continue
            ctx.run("supp", fig_id, lang,
                    lambda lang=lang, c=c, name=name, prep=prep: job_case(
                        ctx, c, prep["rows"], ctx.stem(name, lang, "supp"), lang, prep["rule"],
                        label=candidate_label(mc, c, lang)))


def run_fig2(ctx: Context) -> None:
    """The main-text gallery (tiers C, D, E) and the supplementary gallery (A-E), D8."""
    if ctx.cases is None:
        for variant, (group, fig_id, _, _) in GALLERY_STEMS.items():
            ctx.skip(group, fig_id, "no cases.json")
        return
    from scripts.exp18.figures import fig_gallery as fg

    gal = ctx.cases.get("gallery") or {}
    for t in TIER_ORDER:
        for p in (gal.get(t) or {}).get("picks") or []:
            try:
                ctx.dumps_used.setdefault(str(resolve_dump(ctx, p)), f"gallery {t} P{p.get('percentile')}")
            except FileNotFoundError:
                pass  # the gallery draws a placeholder tile and reports it

    def job(variant: str, lang: str):
        group, fig_id, stem, sub = GALLERY_STEMS[variant]
        res = fg.make_gallery_figure(ctx.cases, dumps_root=str(ctx.dumps_root) if ctx.dumps_root else None,
                                     topdown_root=str(ctx.topdown_root),
                                     clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                                     out_stem=str(ctx.stem(stem, lang, sub)), lang=lang, metrics=str(ctx.metrics_dir),
                                     variant=variant)
        tiles = res.get("tiles") or []
        missing = [f"tile {t['tier']} P{t['percentile']} drawn as a placeholder: {t['error']}"
                   for t in tiles if t.get("error")]
        blocks = []
        picks = {(t, p.get("clip_key")): p for t in TIER_ORDER for p in (gal.get(t) or {}).get("picks") or []}
        for tier, win in (res.get("windows") or {}).items():
            tt = [t for t in tiles if t.get("tier") == tier and t.get("row") is not None]
            blocks.append({"tier": tier, "window": list(win),
                           "tiles": [{"npz": str(resolve_dump(ctx, picks[(tier, t["clip_key"])])), "tier": tier,
                                      "clip_key": t["clip_key"], "row": int(t["row"]), "stats": public(t)}
                                     for t in tt]})
        details = {"variant": variant, "tiers": res.get("tiers"), "missing_tiers": res.get("missing_tiers"),
                   "tiles": public(tiles), "windows": res.get("windows"),
                   "size_in": [round(v, 3) for v in res["size_in"]], "height_limit_in": res.get("height_limit_in"),
                   "min_font_pt": res.get("min_font_pt"), "caption_flags": res.get("caption_flags"),
                   "warnings": [public(w) for w in missing], "_feedback": feedback(res), "_blocks": blocks}
        return res["files"], details

    for variant in ("main", "supp"):
        group, fig_id, _, _ = GALLERY_STEMS[variant]
        present = [t for t in fg.VARIANT_TIERS[variant]
                   if (gal.get(t) or {}).get("status") == "ok" and (gal[t].get("picks") or [])]
        if not present:
            ctx.skip(group, fig_id, f"cases.json has no gallery picks for tiers {', '.join(fg.VARIANT_TIERS[variant])}")
            continue
        for lang in ctx.langs:
            ctx.run(group, fig_id, lang, lambda variant=variant, lang=lang: job(variant, lang))


def run_fig3(ctx: Context) -> None:
    if ctx.metrics is None:
        ctx.skip("fig3", "fig3_metrics", f"no metrics.json in {ctx.metrics_dir}")
        return
    from scripts.exp18.figures import fig_metrics as fm

    tiers, skipped = fm.usable_tiers(ctx.metrics)
    if not tiers:
        ctx.skip("fig3", "fig3_metrics", f"no tier with a deployed-model prediction in metrics.json ({skipped})")
        return

    def job(lang: str):
        res = fm.make_metrics_figure(str(ctx.metrics_dir), out_stem=str(ctx.stem("fig3_metrics", lang)), lang=lang)
        st = res.get("stats") or {}
        details = {"tiers": st.get("tiers"), "skipped_tiers": st.get("skipped"), "panel_d": st.get("panel_d"),
                   "bootstrap": {"reps": st.get("reps"), "seed": st.get("seed")},
                   "per_tier": public(st.get("per_tier")), "checks": public(st.get("checks")),
                   "size_in": [round(v, 3) for v in res["size_in"]], "_feedback": feedback(res)}
        return res["files"], details

    for lang in ctx.langs:
        ctx.run("fig3", "fig3_metrics", lang, lambda lang=lang: job(lang))


def run_fig4(ctx: Context) -> None:
    if ctx.cases is None:
        for p in ROUTE_PATTERNS:
            ctx.skip("fig4", f"fig4_route_{p}", "no cases.json")
        return
    pf = ctx.cases.get("pattern_figure") or {}
    ok = []
    for p in ROUTE_PATTERNS:
        pick = pf.get(p)
        if not (isinstance(pick, dict) and pick.get("status") == "ok"):
            status = pick.get("status") if isinstance(pick, dict) else "absent"
            ctx.skip("fig4", f"fig4_route_{p}", f"no {PATTERN_TITLES[p]} pick in cases.json (status {status})")
        elif str(pick.get("tier")) != "E":  # refuse: the route figures come from designed routes only
            ctx.fail("fig4", f"fig4_route_{p}", f"the {PATTERN_TITLES[p]} pick {pick.get('clip_key')} is tier "
                                                f"{pick.get('tier')!r}; route figures are drawn only from tier E")
        else:
            ok.append(p)
            try:
                ctx.dumps_used.setdefault(str(resolve_dump(ctx, pick)), f"route {p}")
            except FileNotFoundError:
                pass
    if not ok:
        return
    import tempfile

    from scripts.exp18.figures import fig_routes as fr

    def job(p: str, lang: str):
        # fig_routes reads its picks from a cases.json file; hand it one holding only pattern p, so a pattern that
        # fails (a dump missing, key rows not matching) does not take the other one with it
        one = dict(ctx.cases, pattern_figure={p: pf[p]})
        with tempfile.TemporaryDirectory(prefix="exp18_fig4_") as tmp:
            cases_p = Path(tmp) / f"cases_{p}.json"
            cases_p.write_text(json.dumps(one, ensure_ascii=False), encoding="utf-8")
            res = fr.make_route_figures(str(cases_p), dumps_root=str(ctx.dumps_root) if ctx.dumps_root else None,
                                        topdown_root=str(ctx.topdown_root),
                                        clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                                        out_dir=str(ctx.out_dir), lang=lang,
                                        metrics_json=str(ctx.metrics_dir / "metrics.json"))
        if p not in res["figures"]:
            raise RuntimeError(f"fig_routes drew no {p} figure: {res['skipped'].get(p, 'no reason given')}")
        r = res["figures"][p]
        files = []
        for f in r["files"]:  # route_<pattern>[_zh].* -> fig4_route_<pattern>[_zh].*
            src = Path(f)
            dst = src.with_name("fig4_" + src.name)
            os.replace(src, dst)
            files.append(str(dst))
        npz = str(r.get("dump") or resolve_dump(ctx, pf[p]))
        details = {"case": pick_summary(pf[p]), "rows": r.get("rows"), "roles": r.get("roles"),
                   "split_frame": r.get("split_frame"), "split_rule": r.get("split_rule"),
                   "front_view_totals": public(r.get("tally_totals")), "same_spot": r.get("same_spot"),
                   "stats": public(r.get("stats")), "windows": r.get("windows"),
                   "size_in": [round(v, 3) for v in r["size_in"]], "_feedback": feedback(r),
                   "_blocks": [{"npz": npz, "tier": "E", "clip_key": pf[p].get("clip_key"), "rows": [int(s["row"])],
                                "window": list(w), "stats": [public(s)]}
                               for s, w in zip(r.get("stats") or [], r.get("windows") or [])]}
        return files, details

    for p in ok:
        for lang in ctx.langs:
            ctx.run("fig4", f"fig4_route_{p}", lang, lambda p=p, lang=lang: job(p, lang))


def run_anim(ctx: Context) -> None:
    mc = ctx.main_case
    if mc.get("source") is None:
        if mc.get("error"):
            ctx.fail("anim", "supp_anim_main_case", mc["reason"])
        else:
            ctx.skip("anim", "supp_anim_main_case", mc.get("reason", "no main case"))
        return

    def job(lang: str):
        if mc.get("prepare_error"):
            raise RuntimeError(mc["prepare_error"])
        from scripts.exp18.figures import fig_anim as fa

        res = fa.make_animation(mc["npz"], out_stem=str(ctx.stem("anim_main_case", lang, "supp")),
                                topdown_root=str(ctx.topdown_root),
                                clip_root_override=str(ctx.clip_root) if ctx.clip_root else None, lang=lang,
                                size=ctx.anim_size)
        edit_caption(res["files"], lang, suffix=main_case_suffix(mc, lang))
        stats = res.get("stats") or []
        details = {"case": pick_summary(mc["pick"]), "rows": res.get("rows"), "video_frames": res.get("frames"),
                   "duration_s": round(res.get("duration_s", 0.0), 2), "size_px": list(res.get("size", ())),
                   "gif_size": res.get("gif_size"), "fps": res.get("fps"), "encoder": res.get("encoder"),
                   "render_s": round(res.get("render_s", 0.0), 1), "caption_flags": res.get("caption_flags"),
                   "queries": [public({k: s.get(k) for k in ("row", "frame", "frame_label", "vo", "floor", "misses",
                                                             "elevation_window", "vertical_scale")}) for s in stats],
                   "_feedback": feedback(res),
                   "_blocks": [{"npz": mc["npz"], "tier": mc["pick"].get("tier"), "clip_key": mc["pick"].get("clip_key"),
                                "rows": [int(s["row"])], "window": list(s["elevation_window"]), "stats": [public(s)]}
                               for s in stats]}
        return res["files"], details

    for lang in ctx.langs:
        ctx.run("anim", "supp_anim_main_case", lang, lambda lang=lang: job(lang))


# --------------------------------------------------------------------------- #
# Checks: policy, parentheses, set consistency, numbers, windows, colour, font size
# --------------------------------------------------------------------------- #
def _finding(level: str, rule: str, where: str, **extra) -> dict:
    d = {"level": level, "rule": rule, "where": where}
    d.update({k: v for k, v in extra.items() if v not in (None, "")})
    return d


def _masked(s: str, m: re.Match, pad: int = 30) -> str:
    """Context of a policy match with the match itself masked (printed to the log, never stored)."""
    return (s[max(0, m.start() - pad): m.start()] + "▇" * max(1, len(m.group(0))) + s[m.end(): m.end() + pad]
            ).replace("\n", " ")


def policy_findings(s: str, where: str, log: bool = True) -> List[dict]:
    """Policy errors (P*) and warnings (W*) in ``s``: pattern id and place only."""
    out = []
    s = PROJECT_NAME_RE.sub("<project>", s)
    for rules, level in ((_POLICY_C, "error"), (_WARN_C, "warning")):
        for pid, rx in rules:
            m = rx.search(s)
            if m:
                out.append(_finding(level, "policy", where, pattern=pid))
                if log:
                    print(f"[make_all]   policy {level} {pid} in {where}: …{_masked(s, m)}…", flush=True)
    return out


def paren_findings(s: str, lang: str, where: str) -> List[dict]:
    """The parentheses convention (D6, ``ZH_PARENS``) in drawn text and captions."""
    out = []
    if lang != "zh":
        full = re.search("[（）]", s)
        if full:
            out.append(_finding("error", "en text with a full-width parenthesis", where, match=full.group(0),
                                context=s[max(0, full.start() - 20): full.end() + 20].replace("\n", " ")))
        return out
    for m in _HALF_PAREN_RE.finditer(s):
        inner = m.group(1).strip()
        if _PANEL_RE.fullmatch(inner) or _TUPLE_RE.fullmatch(inner):
            continue  # panel letter (a), coordinates (32, 32)
        before = s[:m.start()].rstrip()
        if _CJK_RE.search(inner) or (before and _CJK_RE.match(before[-1])):
            out.append(_finding("error", "zh text: use full-width （）", where, match=m.group(0)[:40],
                                context=s[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")))
    return out


def _label_strings(obj, path: str):
    """(string, path) for every string in a label table."""
    if isinstance(obj, str):
        yield obj, path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _label_strings(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _label_strings(v, f"{path}[{i}]")


def _import_fig_module(name: str):
    import importlib

    return importlib.import_module(f"scripts.exp18.figures.{name}")


def label_table_findings() -> Tuple[List[dict], int]:
    """Policy patterns in every module-level label table (``LABELS``, ``CAPTION*``, ``*_TEXT``, ``*NAMES*``, ...),
    including strings of figures not drawn this run."""
    out, n = [], 0
    for mod_name in LABEL_MODULES:
        try:
            mod = _import_fig_module(mod_name)
        except Exception as exc:
            out.append(_finding("error", "cannot import a figure module for the checks", mod_name,
                                context=f"{type(exc).__name__}: {exc}"))
            continue
        for name, val in vars(mod).items():
            if name.startswith("_") or not isinstance(val, (dict, tuple, list)):
                continue
            if not re.search(r"LABEL|CAPTION|TEXT|NAMES|TITLE|^L[A-Z]?$|^LR$", name):
                continue
            for s, path in _label_strings(val, f"{mod_name}.{name}"):
                n += 1
                out += [f for f in policy_findings(s, path) if f["level"] == "error"]
    return out, n


def _hex_rgb(c) -> Optional[Tuple[float, float, float]]:
    if not isinstance(c, str) or not re.fullmatch(r"\s*#[0-9a-fA-F]{6}\s*", c):
        return None
    c = c.strip()
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def colour_role(c) -> Optional[str]:
    """'ground truth' / 'prediction' when colour ``c`` reads as the ground-truth blue / prediction orange."""
    rgb = _hex_rgb(c)
    if rgb is None:
        return None
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    if s < MIN_SAT or v < MIN_VAL:
        return None
    for role, (ref, _) in COLOUR_ROLES.items():
        rh = colorsys.rgb_to_hsv(*_hex_rgb(ref))[0]
        d = abs(h - rh) * 360.0
        if min(d, 360.0 - d) <= HUE_TOL_DEG:
            return role
    return None


def _tokens(name: str) -> set:
    parts = re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", str(name))
    return {p.lower() for p in parts if p}


def _context_names(node, parents: dict) -> List[str]:
    """Names around a literal: its dict key, keyword, call, assignment target(s), enclosing def."""
    names, n = [], node
    while n in parents:
        p = parents[n]
        if isinstance(p, ast.Dict):
            for k, v in zip(p.keys, p.values):
                if v is n and isinstance(k, ast.Constant):
                    names.append(str(k.value))
        elif isinstance(p, ast.keyword) and p.arg:
            names.append(p.arg)
        elif isinstance(p, ast.Call):
            names.append(ast.unparse(p.func))
        elif isinstance(p, (ast.Assign, ast.AnnAssign)):
            for t in (p.targets if isinstance(p, ast.Assign) else [p.target]):
                names.append(ast.unparse(t))
        elif isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(p.name)
            break
        n = p
    return names


def colour_findings() -> List[dict]:
    """Colours that read as the ground-truth blue or the prediction orange, used for anything else (D7)."""
    out, reported = [], set()  # (module, colour) already reported as a literal
    for mod_name in COLOUR_MODULES:
        try:
            mod = _import_fig_module(mod_name)
        except Exception as exc:
            out.append(_finding("error", "cannot import a figure module for the checks", mod_name,
                                context=f"{type(exc).__name__}: {exc}"))
            continue
        path = Path(mod.__file__)
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            role = colour_role(node.value)
            if role is None:
                continue
            names = _context_names(node, parents)
            tokens = set().union(*(_tokens(x) for x in names)) if names else set()
            line = lines[node.lineno - 1]
            marker = re.search(r"colou?r-role:\s*(ground truth|prediction)", line)
            if tokens & COLOUR_ROLES[role][1] or (marker and marker.group(1) == role):
                continue
            reported.add((mod_name, node.value.strip().lower()))
            out.append(_finding("error", f"colour reads as the {role} colour but is used for something else",
                                f"{mod_name}.py:{node.lineno}", match=node.value,
                                context=f"{' / '.join(names[:4]) or 'module level'}: {line.strip()[:100]}"))
        # categorical palettes built at runtime (e.g. from constants)
        for name, val in vars(mod).items():
            if not re.search(r"colou?rs?$|palette|shades?$", name, re.I):
                continue
            items = val.items() if isinstance(val, dict) else enumerate(val) if isinstance(val, (list, tuple)) else ()
            for key, c in items:
                role = colour_role(c)
                if role is None or (_tokens(str(key)) | _tokens(name)) & COLOUR_ROLES[role][1]:
                    continue
                if (mod_name, str(c).strip().lower()) in reported:
                    continue  # the literal is already reported
                out.append(_finding("error", f"categorical palette reuses the {role} colour",
                                    f"{mod_name}.{name}[{key!r}]", match=str(c)))
    return out


def frame_formats(s: str, lang: str) -> List[str]:
    found = []
    for line in s.splitlines():
        for fmt, rx in FRAME_FORMATS.get(lang, ()):
            line, n = re.subn(rx, " ", line, flags=re.I)
            if n:
                found.append(fmt)
    return found


def elevation_rule() -> dict:
    """The one elevation-window rule (D4) as the shared layer defines it, and the modules' defaults."""
    from scripts.exp18.figures import common_draw as cd

    out = {"rule": "common_draw.elevation_window", "default_deg": float(cd.EL_DEFAULT), "max_deg": float(cd.EL_MAX),
           "module_defaults": {}}
    for mod_name in ROW_MODULES:
        try:
            mod = _import_fig_module(mod_name)
        except Exception:
            continue
        v = getattr(mod, "EL_HEAT", None)
        out["module_defaults"][mod_name] = float(v) if v is not None else None
        out.setdefault("calls_rule", {})[mod_name] = "elevation_window(" in Path(mod.__file__).read_text("utf-8")
    return out


def set_findings(ctx: Context) -> Tuple[List[dict], dict]:
    """Consistency across the set, from the text the figures drew; plus the conventions recorded."""
    out: List[dict] = []
    frames: Dict[str, Dict[str, set]] = {}
    views: Dict[Tuple[str, str], Dict[str, set]] = {}
    ahead: Dict[str, Dict[str, set]] = {}
    for e in ctx.entries:
        if e.status == "skipped":
            continue
        for s in e.texts:
            for fmt in frame_formats(s, e.lang):
                frames.setdefault(e.lang, {}).setdefault(fmt, set()).add(e.fig_id)
            for line in s.splitlines():
                line = line.strip()
                m = VIEW_LABEL_RE.get(e.lang, VIEW_LABEL_RE["en"]).match(line)
                if m:
                    views.setdefault((e.lang, m.group(1)), {}).setdefault(m.group(2).strip(), set()).add(e.fig_id)
                m = AHEAD_RE.match(line)
                if m:
                    ahead.setdefault(e.lang, {}).setdefault(m.group(1), set()).add(e.fig_id)
    for lang, fmts in frames.items():
        other = {f: v for f, v in fmts.items() if f != FRAME_FORMAT_D6.get(lang)}
        if other:
            out.append(_finding("error", f"frame labels not in the D6 format '{FRAME_FORMAT_D6.get(lang)}'",
                                f"all figures ({lang})",
                                context="; ".join(f"'{f}' in {', '.join(sorted(v))}" for f, v in sorted(other.items()))))
    for (lang, view), quals in views.items():
        if len(quals) > 1:
            out.append(_finding("error", "one view is labelled in more than one way", f"all figures ({lang})",
                                context="; ".join(f"'{view} · {q}' in {', '.join(sorted(v))}"
                                                  for q, v in sorted(quals.items()))))
    for lang, quals in ahead.items():
        if len(quals) > 1:
            out.append(_finding("error", "the 0° label is worded in more than one way", f"all figures ({lang})",
                                context="; ".join(f"'0° ({q})' in {', '.join(sorted(v))}" for q, v in sorted(quals.items()))))
    el = elevation_rule()
    wrong = {m: v for m, v in el["module_defaults"].items() if v is not None and v != el["default_deg"]}
    if wrong or el["default_deg"] != 10.0 or el["max_deg"] != 45.0:
        out.append(_finding("error", "elevation-window defaults differ from D4 (±10°, up to ±45°)",
                            "common_draw / " + ", ".join(sorted(wrong)),
                            context=f"shared ±{el['default_deg']:g}° up to ±{el['max_deg']:g}°; "
                                    + "; ".join(f"{k} ±{v:g}°" for k, v in sorted(wrong.items()))))
    no_call = [m for m, ok in (el.get("calls_rule") or {}).items() if not ok]
    if no_call:
        out.append(_finding("error", "module draws affordance-map rows without common_draw.elevation_window",
                            ", ".join(no_call)))
    # fig1 must not repeat a gallery tile
    mc = ctx.main_case
    drawn = {e.group for e in ctx.entries if e.status != "skipped"}
    if mc.get("pick") and "fig1" in drawn and ({"fig2", "supp_gallery"} & drawn) and ctx.cases:
        t, key = mc["pick"].get("tier"), mc["pick"].get("clip_key")
        if key in gallery_keys(ctx.cases, t):
            out.append(_finding("error", "fig1 shows an episode that is also a gallery tile", "fig1_main_case, galleries",
                                context=f"tier {t} {key}; pick another candidate with --main-index"))
    conventions = {
        "ground_truth_colour": GT_BLUE, "prediction_colour": PRED_ORANGE,
        "prediction_shown": "the deployed model's output",
        "miss_rule": "joint PCK@8 failure (D1)",
        "zh_parentheses": ZH_PARENS,
        "min_font_pt": MIN_FONT_PT,
        "elevation_window": el,
        "frame_label_formats": {lang: sorted(f) for lang, f in frames.items()},
        "view_labels": {f"{lang}:{view}": sorted(q) for (lang, view), q in sorted(views.items())},
        "zero_bearing_label": {lang: sorted(q) for lang, q in ahead.items()},
    }
    return out, conventions


def _truth(tabs: dict) -> Callable[[str, str, int], Optional[dict]]:
    """(tier, clip_key, row) -> {"misses": [1-based slots], "n", "hits", "pck8_row"} from the tables."""
    slots, rows = tabs["slots"], tabs["rows"]
    grp = {(t, c, int(r)): g for (t, c, r), g in slots.groupby(["tier", "clip_key", "row"])}
    rrow = {(t, c, int(r)): v for t, c, r, v in zip(rows["tier"], rows["clip_key"], rows["row"], rows["vo_pck8_row"])}

    def get(tier: str, clip_key: str, row: int) -> Optional[dict]:
        g = grp.get((tier, clip_key, int(row)))
        if g is None:
            return None
        g = g[g["gt_visible"].astype(bool)]
        ok = g["pred_vo_joint8"].astype(bool).values
        return {"misses": sorted(int(k) + 1 for k, h in zip(g["k"], ok) if not h), "n": int(len(g)),
                "hits": int(ok.sum()), "pck8_row": rrow.get((tier, clip_key, int(row)))}

    return get


def _pred_summary(stats: dict) -> Optional[dict]:
    for k in ("prediction", "vo"):
        if isinstance(stats.get(k), dict):
            return stats[k]
    return None


def number_findings(ctx: Context) -> Tuple[List[dict], dict]:
    """D1 against the tables for every drawn row, D4 against ``elevation_window`` for every block, D8 for the
    gallery tiles."""
    out: List[dict] = []
    summary = {"rows_checked": 0, "blocks_checked": 0, "tiles_checked_d8": 0}
    tabs = ctx.tables()
    drawn = [e for e in ctx.entries if e.status == "ok" and e.blocks]
    if not drawn:
        return out, summary
    if tabs is None:
        out.append(_finding("error", "cannot check the drawn numbers: no slots/rows/episodes tables",
                            str(ctx.metrics_dir)))
        return out, summary
    truth = _truth(tabs)
    from scripts.exp18.figures import common_draw as cd
    from scripts.exp18.figures import data as dd

    dumps: Dict[str, object] = {}

    def case_row(npz: str, i: int):
        if npz not in dumps:
            dumps[npz] = dd.load_dump(npz)
        return dd.case_row(dumps[npz], int(i))

    for e in drawn:
        where = f"{e.fig_id} ({e.lang})"
        for b in e.blocks:
            members = b["tiles"] if "tiles" in b else [dict(b, row=r, stats=s) for r, s in zip(b["rows"], b["stats"])]
            for m in members:
                summary["rows_checked"] += 1
                tv = truth(m["tier"], m["clip_key"], m["row"])
                st = m["stats"]
                ps = _pred_summary(st) or {}
                tag = f"{m['tier']} {m['clip_key']} row {m['row']}"
                if tv is None:
                    out.append(_finding("error", "D1: drawn row not in the slots table", where, context=tag))
                    continue
                drawn_m = sorted(int(k) for k in st.get("misses") or [])
                if drawn_m != tv["misses"]:
                    out.append(_finding("error", "D1: numbered misses differ from the slots table", where,
                                        context=f"{tag}: drawn {drawn_m}, table {tv['misses']}"))
                if (ps.get("n"), ps.get("hits")) != (tv["n"], tv["hits"]):
                    out.append(_finding("error", "D1: hits / visible differ from the slots table", where,
                                        context=f"{tag}: drawn {ps.get('hits')}/{ps.get('n')}, table "
                                                f"{tv['hits']}/{tv['n']}"))
                if tv["pck8_row"] is not None and tv["n"] and abs(tv["hits"] / tv["n"] - float(tv["pck8_row"])) > 1e-9:
                    out.append(_finding("error", "D1: rows table PCK@8 differs from the slots table", where, context=tag))
                if len(drawn_m) != tv["n"] - tv["hits"]:
                    out.append(_finding("error", "D1: numbered misses != visible - hits", where, context=tag))
            # D4: the block's window is the shared rule applied to the rows it shows
            try:
                recs = [case_row(m["npz"], m["row"]) for m in members]
                want = tuple(float(v) for v in cd.elevation_window(recs))
                got = tuple(float(v) for v in b["window"])
                summary["blocks_checked"] += 1
                if max(abs(a - c) for a, c in zip(want, got)) > 1e-6:
                    out.append(_finding("error", "D4: elevation window is not elevation_window(rows shown)", where,
                                        context=f"{b.get('tier')} rows {[m['row'] for m in members]}: drawn {got}, "
                                                f"rule {want}"))
            except Exception as exc:
                out.append(_finding("error", "D4: cannot recompute an elevation window", where,
                                    context=f"{type(exc).__name__}: {exc}"))
        cap = caption_of(e)
        if cap is not None and cap.is_file():
            text = cap.read_text(encoding="utf-8")
            wins = sorted({tuple(float(v) for v in b["window"]) for b in e.blocks})
            missing = [w for w in wins if ELEV_RANGE[e.lang].format(lo=cd.fmt_deg(w[0]), hi=cd.fmt_deg(w[1])) not in text]
            if ELEV_WORD[e.lang] not in text or missing:
                out.append(_finding("error", "D4: caption does not state every elevation window drawn", where,
                                    context=", ".join(f"{cd.fmt_deg(a)}..{cd.fmt_deg(b)}" for a, b in missing)))
            else:
                summary["captions_stating_windows"] = summary.get("captions_stating_windows", 0) + 1
        if e.group in ("fig2", "supp_gallery"):
            out += gallery_frame_findings(tabs, e, summary)
    return out, summary


def gallery_frame_findings(tabs: dict, e: Entry, summary: dict) -> List[dict]:
    """D8: each tile shows the scored frame whose median bearing error is closest to the episode's median
    (ties: wider ground-truth bearing span, then the earlier frame)."""
    out = []
    slots, rows, eps = tabs["slots"], tabs["rows"], tabs["episodes"]
    for b in e.blocks:
        for m in b["tiles"]:
            tier, ck = m["tier"], m["clip_key"]
            s = slots[(slots["tier"] == tier) & (slots["clip_key"] == ck) & slots["gt_visible"].astype(bool)
                      & slots["vo_row"].astype(bool)]
            err = s[s["pred_vo_bearing_err"].notna()]
            if err.empty:
                continue
            ep_med = float(err["pred_vo_bearing_err"].median())
            ep_tab = eps[(eps["tier"] == tier) & (eps["clip_key"] == ck)]["vo_bearing_err_median"]
            r = rows[(rows["tier"] == tier) & (rows["clip_key"] == ck)].set_index("row")
            best = None
            for row, g in err.groupby("row"):
                span = float(r.loc[row, "gt_bearing_span_deg"]) if row in r.index else 0.0
                key = (round(abs(float(g["pred_vo_bearing_err"].median()) - ep_med), 9), -round(span, 9), int(row))
                best = key if best is None or key < best else best
            summary["tiles_checked_d8"] += 1
            if best is not None and best[2] != int(m["row"]):
                out.append(_finding("error", "D8: gallery tile is not the frame closest to the episode median",
                                    f"{e.fig_id} ({e.lang})", context=f"{tier} {ck}: drawn row {m['row']}, rule row "
                                                                      f"{best[2]}"))
            if len(ep_tab) and abs(float(ep_tab.iloc[0]) - ep_med) > 1e-6:
                out.append(_finding("error", "D8: episode median differs from the episodes table",
                                    f"{e.fig_id} ({e.lang})", context=f"{tier} {ck}"))
    return out


def run_checks(ctx: Context, readme: Path) -> dict:
    """Every check except the manifest scan (done when the manifest is written)."""
    findings: List[dict] = []
    checked = {"figures": 0, "strings_drawn": 0, "text_artists_walked": 0, "captions": 0, "label_strings": 0,
               "readme": 0}
    for e in ctx.entries:
        if e.status == "skipped":
            continue
        where = f"{e.fig_id} ({e.lang})"
        strings = set(e.texts) | e.walked
        if strings:
            checked["figures"] += 1
            checked["strings_drawn"] += len(e.texts)
            checked["text_artists_walked"] += len(e.walked)
        for s in sorted(strings):
            findings += policy_findings(s, f"{where} figure text")
        for s in e.texts:
            findings += paren_findings(s, e.lang, f"{where} figure text")
        if e.group != "anim" and e.texts:
            small = sorted((fs, s) for s, fs in e.texts.items() if fs < MIN_FONT_PT - FONT_TOL_PT)
            if small:
                findings.append(_finding("error", f"text below {MIN_FONT_PT} pt at print size", where,
                                         context="; ".join(f"{fs:.2f} pt '{s[:30]}'" for fs, s in small[:6])
                                         + (f" (+{len(small) - 6} more)" if len(small) > 6 else "")))
        cap = caption_of(e)
        if cap is not None and cap.is_file():
            checked["captions"] += 1
            text = cap.read_text(encoding="utf-8")
            findings += policy_findings(text, f"{where} caption") + paren_findings(text, e.lang, f"{where} caption")
    lab, n = label_table_findings()
    checked["label_strings"] = n
    findings += lab
    findings += colour_findings()
    sf, conventions = set_findings(ctx)
    findings += sf
    nf, numbers = number_findings(ctx)
    findings += nf
    if readme.is_file():
        checked["readme"] = 1
        findings += policy_findings(readme.read_text(encoding="utf-8"), "README.md")
    # one finding per (level, rule, where, pattern, match, context)
    uniq = list({tuple(str(f.get(k, "")) for k in ("level", "rule", "where", "pattern", "match", "context")): f
                 for f in findings}.values())
    return {"errors": [f for f in uniq if f["level"] == "error"], "warnings": [f for f in uniq if f["level"] == "warning"],
            "checked": checked, "numbers": numbers, "conventions": conventions}


def manifest_policy_hits(manifest: dict) -> List[dict]:
    """Policy patterns in the manifest itself (tracebacks and error texts are diagnostics and skipped)."""
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in ("traceback", "error", "lint", "policy_check")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    text = json.dumps(strip(manifest), ensure_ascii=False, default=str)
    return [f for f in policy_findings(text, "manifest.json") if f["level"] == "error"]


# --------------------------------------------------------------------------- #
# Fonts, manifest, README
# --------------------------------------------------------------------------- #
def font_report(langs: List[str]) -> dict:
    """Which families the figures resolved per language (a blank container may lack the paper fonts)."""
    out = {}
    try:
        import matplotlib

        from scripts.exp18.figures import common_draw as cd
        from scripts.exp18.figures import style
    except Exception as exc:  # matplotlib missing: every figure job fails and says why
        return {"error": f"{type(exc).__name__}: {exc}"}
    for lang in langs:
        cd.setup(lang)
        fams = list(matplotlib.rcParams["font.sans-serif"])
        rep = {"families": fams, "latin_paper_font": style.LATIN_FAMILY in fams}
        if lang == "zh":
            rep["cjk_font"] = any(f not in (style.LATIN_FAMILY, "DejaVu Sans") for f in fams)
        out[lang] = rep
        if not rep["latin_paper_font"]:
            print(f"[make_all] WARNING ({lang}): {style.LATIN_FAMILY} not found (set EXP18_FONT_DIR); "
                  f"figures fall back to {fams[-1]}")
        if lang == "zh" and not rep["cjk_font"]:
            print("[make_all] WARNING (zh): no CJK font (set EXP18_CJK_FONT); Chinese text will show as boxes")
    return out


def file_records(ctx: Context) -> None:
    for e in ctx.entries:
        recs = []
        for f in e.files:
            p = Path(f)
            rel = os.path.relpath(p, ctx.out_dir)
            if p.is_file():
                recs.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size})
            else:
                recs.append({"path": rel, "sha256": None, "missing": True})
        e.details["_files"] = recs


def stale_files(ctx: Context, produced: set) -> List[str]:
    keep = {"manifest.json", "README.md"}
    out = []
    for p in sorted(ctx.out_dir.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(ctx.out_dir).parts):
            rel = str(p.relative_to(ctx.out_dir))
            if rel not in produced and rel not in keep:
                out.append(rel)
    return out


def clean_previous(out_dir: Path) -> List[str]:
    """Delete the files the previous manifest.json listed (never anything else)."""
    mpath = out_dir / "manifest.json"
    if not mpath.is_file():
        return []
    try:
        prev = json.loads(mpath.read_text(encoding="utf-8"))
    except ValueError:
        return []
    removed = []
    root = out_dir.resolve()
    for fig in prev.get("figures", []):
        for rec in fig.get("files", []):
            p = (out_dir / rec.get("path", "")).resolve()
            if p.is_file() and root in p.parents:
                p.unlink()
                removed.append(rec["path"])
    return removed


def caption_of(e: Entry) -> Optional[Path]:
    for f in e.files:
        if f.endswith("_caption.txt"):
            return Path(f)
    return None


SECTION_ORDER = ("fig1", "fig2", "fig3", "fig4", "supp", "supp_gallery", "anim")


def _sub_order(fig_id: str) -> tuple:
    """Within a section: route patterns in ROUTE_PATTERNS order, candidates by rank."""
    for n, p in enumerate(ROUTE_PATTERNS):
        if fig_id.endswith(p):
            return (n, fig_id)
    m = re.search(r"candidate(\d+)$", fig_id)
    return (int(m.group(1)) if m else 0, fig_id)


def figure_order(e: Entry) -> tuple:
    return (SECTION_ORDER.index(e.group) if e.group in SECTION_ORDER else 99, _sub_order(e.fig_id),
            LANGS.index(e.lang) if e.lang in LANGS else 9)


def _lint_line(f: dict) -> str:
    """README / log line for a check finding; policy findings name only the pattern id (never the words)."""
    if f["rule"] == "policy":
        return f"- policy pattern {f['pattern']} ({PATTERN_NOTES.get(f['pattern'], '')}) — {f['where']}"
    shown = f" «{f['match']}»" if f.get("match") else ""
    ctx_ = f": {f['context']}" if f.get("context") else ""
    return f"- {f['rule']} — {f['where']}{shown}{ctx_}"


def write_readme(ctx: Context, manifest: dict) -> Path:
    L: List[str] = []
    gi = manifest["code"]["git"]
    sha = gi.get("sha", "unknown") + ("-dirty" if gi.get("dirty") else "")
    L.append("# EXP-18 figures\n")
    L.append(f"Rendered {manifest['created_utc'][:19].replace('T', ' ')} UTC by "
             f"`scripts/exp18/figures/make_all.py` (code `{sha}`, EXP-18 sources sha256 "
             f"`{manifest['code']['digest']['sha256'][:16]}…`). Languages: {', '.join(ctx.langs)}."
             + (" **Draft run** (check errors do not fail it)." if ctx.draft else "") + "\n")
    inp = manifest["inputs"]
    L.append("**Inputs.**\n")
    for key, label in (("metrics_json", "metrics"), ("cases_json", "cases"), ("slots", "slots table"),
                       ("episodes", "episodes table"), ("rows", "rows table")):
        r = inp[key]
        h = f"sha256 `{r['sha256'][:16]}…`" if r.get("sha256") else "**missing**"
        L.append(f"- {label}: `{r['path']}` ({h})")
    tiers = manifest.get("tiers", {})
    if tiers:
        L.append(f"- tiers with a prediction in metrics.json: {', '.join(tiers.get('with_prediction') or []) or 'none'}"
                 f"; not available: {', '.join(tiers.get('not_available') or []) or 'none'}")
    L.append("")
    mc = manifest["main_case"]
    if mc.get("source") == "main_candidates":
        c = mc["case"]
        shared = "; ".join("candidates " + " and ".join(str(r) for r in g) + " share one R2R path"
                           for g in mc.get("shared_paths") or [])
        L.append(f"**Main case (fig1, animation).** {mc['note']}: tier {c.get('tier')} `{c.get('clip_key')}`, episode "
                 f"{c.get('episode_id')}, frames {[f + 1 for f in mc.get('frames') or []]} (1-based; "
                 f"{mc.get('rows_rule')}). All {mc['n_candidates']} pre-registered candidates are drawn in `supp/`"
                 + (f" ({shared})" if shared else "") + ". Re-render with `--main-index k` (or `MAIN_INDEX=k`) to make "
                 "candidate k fig1.\n")
    else:
        L.append(f"> **Main case: none.** {mc.get('reason', '')}\n")

    lint = manifest.get("lint", {})
    errs, warns = lint.get("errors", []), lint.get("warnings", [])
    ch = lint.get("checked", {})
    L.append("## Checks\n")
    L.append(f"Policy (id `{POLICY_ID}`, patterns {', '.join(i for i, _ in POLICY_PATTERNS)} = errors, "
             f"{', '.join(i for i, _ in WARN_PATTERNS)} = warnings): "
             f"{len(manifest['policy_check']['hits'])} error hit(s), {len(manifest['policy_check']['warnings'])} "
             f"warning(s) in {ch.get('strings_drawn', 0)} strings drawn and {ch.get('text_artists_walked', 0)} "
             f"text artists walked in {ch.get('figures', 0)} figures, {ch.get('captions', 0)} captions, "
             f"{ch.get('label_strings', 0)} label-table strings, README.md and manifest.json. Numbers (D1 / D4 / D8): "
             f"{lint.get('numbers', {}).get('rows_checked', 0)} drawn rows against the slots table, "
             f"{lint.get('numbers', {}).get('blocks_checked', 0)} elevation windows recomputed, "
             f"{lint.get('numbers', {}).get('tiles_checked_d8', 0)} gallery frames re-selected. Check errors: "
             f"**{len(errs)}**. Slots unaccounted for (dropped notes): **{manifest['counts']['defects']}**.\n")
    if errs:
        L.append("**Errors** (each fails the run):\n")
        L += [_lint_line(f) for f in errs]
        L.append("")
    if warns:
        L.append("**Warnings** (for manual review):\n")
        L += [_lint_line(f) for f in warns]
        L.append("")
    conv = lint.get("conventions", {})
    if conv:
        el = conv.get("elevation_window", {})
        L.append("**Conventions.** Ground truth blue `" + conv["ground_truth_colour"] + "`, prediction orange `"
                 + conv["prediction_colour"] + "`; misses: " + conv.get("miss_rule", "") + "; parentheses: "
                 + conv["zh_parentheses"] + f"; text ≥ {conv['min_font_pt']} pt; affordance-map rows: "
                 + f"{el.get('rule')} (±{el.get('default_deg', 0):g}°, widened up to ±{el.get('max_deg', 0):g}°, stated "
                 + "in each caption); frame labels: "
                 + "; ".join(f"{k}: {', '.join(v)}" for k, v in conv.get("frame_label_formats", {}).items()) + ".\n")

    L.append("## Contents\n")
    L.append("| Figure | Language | Status | Files | Size |")
    L.append("|---|---|---|---|---|")
    ents = sorted(ctx.entries, key=figure_order)
    rows: List[list] = []  # entries that did not render and share a reason share a row
    for e in ents:
        files = ", ".join(f"`{os.path.relpath(f, ctx.out_dir)}`" for f in e.files if not f.endswith("_caption.txt"))
        size = e.details.get("size_in")
        size_txt = f"{size[0]:.2f} × {size[1]:.2f} in" if size else (
            f"{e.details['size_px'][0]}×{e.details['size_px'][1]} px, {e.details.get('duration_s')} s"
            if e.details.get("size_px") else "")
        status = e.status if e.status == "ok" else f"**{e.status}**: {e.reason or e.error}"
        if e.status != "ok" and rows and rows[-1][0] == e.fig_id and rows[-1][2] == status:
            rows[-1][1] += f", {e.lang}"
            continue
        rows.append([e.fig_id, e.lang, status, files, size_txt])
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("")

    warn: Dict[str, List[str]] = {}  # text -> "fig_id (lang)" it applies to
    for e in ents:
        for w in e.warnings:
            warn.setdefault(w, []).append(f"{e.fig_id} ({e.lang})")
    if warn:
        L.append("**Layout notes the figure modules returned** (not errors).\n")
        for w, where in warn.items():
            L.append(f"- {w} [{', '.join(where)}]")
        L.append("")
    if manifest.get("stale_files"):
        L.append("**Files in this directory not produced by this run** (left over; `--clean` removes the files a "
                 "previous manifest listed):\n")
        for f in manifest["stale_files"]:
            L.append(f"- `{f}`")
        L.append("")

    sections = [("fig1", "fig1_main_case"), ("fig2", "fig2_gallery"), ("fig3", "fig3_metrics")]
    sections += [("fig4", f"fig4_route_{p}") for p in ROUTE_PATTERNS]
    cand_ids = sorted({e.fig_id for e in ctx.entries if e.group == "supp"}, key=_sub_order)
    sections += [("supp", fid) for fid in cand_ids]
    sections += [("supp_gallery", "supp_gallery_all"), ("anim", "supp_anim_main_case")]
    for group, fid in sections:
        es = [e for e in ctx.entries if e.fig_id == fid]
        if not es:
            continue
        title = TITLES[group]
        if group == "fig4":
            title = f"Designed route, {PATTERN_TITLES[fid.rsplit('route_', 1)[1]]}"
        elif group == "supp" and fid != "supp_candidates":
            c = next((e.details["case"] for e in es if e.details.get("case")), {})
            title = (f"Main-figure candidate {_sub_order(fid)[0]} of {mc.get('n_candidates', '?')}"
                     + (f": `{c['clip_key']}`, episode {c.get('episode_id')}" if c.get("clip_key") else "")
                     + (f", |episode PCK@8 − tier median| = {100 * c['abs_diff_from_tier_median']:.1f} points"
                        if c.get("abs_diff_from_tier_median") is not None else "")
                     + (f" (same R2R path as candidate {', '.join(str(r) for r in c['same_path_as'])})"
                        if c.get("same_path_as") else "")
                     + (" — **fig1**" if mc.get("rank") == _sub_order(fid)[0] else ""))
        L.append(f"## {fid}: {title}\n")
        for e in sorted(es, key=lambda x: LANGS.index(x.lang) if x.lang in LANGS else 9):
            if e.status == "skipped" or (e.status == "failed" and not e.files):
                L.append(f"- {e.lang}: **{e.status}**: {e.reason or e.error}")
                continue
            if e.status == "failed":
                L.append(f"- {e.lang}: **failed**: {e.error}\n")
            imgs = [f for f in e.files if f.endswith(".png") or f.endswith(".gif")]
            if imgs:
                rel = os.path.relpath(imgs[0], ctx.out_dir)
                L.append(f"![{fid} {e.lang}]({rel})\n")
            cap = caption_of(e)
            if cap is not None and cap.is_file():
                head = "Caption" if e.lang == "en" else f"Caption ({e.lang})"
                L.append(f"**{head}.** {cap.read_text(encoding='utf-8').strip()}\n")
        L.append("")
    path = ctx.out_dir / "README.md"
    path.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    return path


def build_manifest(ctx: Context, started: float, git: dict, digest: dict, fonts: dict, removed: List[str]) -> dict:
    file_records(ctx)
    produced = {r["path"] for e in ctx.entries for r in e.details.get("_files", [])}
    mc = ctx.main_case
    main_case = {k: public(mc.get(k)) for k in ("source", "note", "reason", "tier", "rank", "n_candidates",
                                                 "candidates_source", "tier_median", "n_meeting_criteria", "n_episodes",
                                                 "criteria", "shared_paths", "npz", "rows", "frames", "rows_rule",
                                                 "prepare_error")
                 if mc.get(k) is not None}
    if mc.get("pick"):
        main_case["case"] = pick_summary(mc["pick"])
        main_case["main_index"] = ctx.main_index
    if mc.get("candidates"):
        main_case["candidates"] = [pick_summary(c) for c in mc["candidates"]]
    tiers = {}
    if ctx.metrics is not None:
        try:
            from scripts.exp18.figures import fig_metrics as fm

            with_pred, skipped = fm.usable_tiers(ctx.metrics)
            tiers = {"with_prediction": with_pred, "not_available": sorted(skipped), "reasons": skipped}
        except Exception as exc:
            tiers = {"error": f"{type(exc).__name__}: {exc}"}
    figures = []
    for e in sorted(ctx.entries, key=figure_order):
        d = dict(e.details)
        recs = d.pop("_files", [])
        cap = caption_of(e)
        static = [fs for s, fs in e.texts.items()] if e.group != "anim" else []
        figures.append({"id": e.fig_id, "group": e.group, "title": TITLES.get(e.group, ""), "lang": e.lang,
                        "status": e.status, "reason": e.reason or None, "error": e.error or None,
                        "files": recs, "caption_file": os.path.relpath(cap, ctx.out_dir) if cap else None,
                        "seconds": e.seconds, "warnings": e.warnings, "unaccounted_slots": e.defects,
                        "strings_drawn": len(e.texts), "text_artists_walked": len(e.walked),
                        "min_font_pt": round(min(static), 2) if static else None, "details": d})
    counts = {s: sum(1 for e in ctx.entries if e.status == s) for s in ("ok", "skipped", "failed")}
    counts["defects"] = sum(len(e.defects) for e in ctx.entries)
    dumps = {p: dict(input_record(Path(p)), role=role) for p, role in sorted(ctx.dumps_used.items())}
    return {
        "schema": SCHEMA,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - started, 1),
        "code": {"git": git, "digest": digest, "python": platform.python_version(),
                 "matplotlib": _version("matplotlib"), "numpy": _version("numpy")},
        "args": {"exp_root": str(ctx.exp_root), "metrics_dir": str(ctx.metrics_dir), "cases": str(ctx.cases_path),
                 "out_dir": str(ctx.out_dir), "langs": ctx.langs, "only": ctx.groups, "main_index": ctx.main_index,
                 "dumps_root": str(ctx.dumps_root) if ctx.dumps_root else None,
                 "topdown_root": str(ctx.topdown_root), "clip_root": str(ctx.clip_root) if ctx.clip_root else None,
                 "anim_size": list(ctx.anim_size), "draft": ctx.draft},
        "inputs": {"metrics_json": input_record(ctx.metrics_dir / "metrics.json"),
                   "cases_json": input_record(ctx.cases_path),
                   "slots": table_record(ctx.metrics_dir / "slots"),
                   "episodes": table_record(ctx.metrics_dir / "episodes"),
                   "rows": table_record(ctx.metrics_dir / "rows"),
                   "cases_created_utc": (ctx.cases or {}).get("created_utc"),
                   "cases_metrics_dir": (ctx.cases or {}).get("metrics_dir"),
                   "cases_tiers_present": (ctx.cases or {}).get("tiers_present"),
                   "dumps": dumps},
        "tiers": tiers,
        "fonts": fonts,
        "main_case": main_case,
        "counts": counts,
        "figures": figures,
        "removed_by_clean": removed,
        "stale_files": stale_files(ctx, produced),
    }


def _version(mod: str) -> Optional[str]:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp-root", type=Path, default=common.EXP_ROOT, help="default $EXP18_ROOT (common.EXP_ROOT)")
    ap.add_argument("--metrics-dir", type=Path, default=None, help="default <exp-root>/metrics")
    ap.add_argument("--cases", type=Path, default=None, help="default <metrics-dir>/cases.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <exp-root>/figures")
    ap.add_argument("--langs", default="en,zh", help="comma list of en, zh")
    ap.add_argument("--main-index", type=int, default=1,
                    help="pre-registered main-figure candidate (1-based rank) drawn as fig1 and animated")
    ap.add_argument("--only", default=",".join(GROUPS), help=f"comma list of {', '.join(GROUPS)} "
                                                              f"(fig2 draws both galleries)")
    ap.add_argument("--dumps-root", type=Path, default=None,
                    help="<root>/<tier>/<scene>/<clip>.npz (default: the npz_path recorded in cases.json)")
    ap.add_argument("--topdown-root", type=Path, default=None, help="default <exp-root>/topdown")
    ap.add_argument("--clip-root", type=Path, default=None, help="local copy of the clips (default: as recorded)")
    ap.add_argument("--anim-size", default="1920x1080", help="animation WxH, 16:9")
    ap.add_argument("--clean", action="store_true", help="first delete the files the previous manifest.json listed")
    ap.add_argument("--draft", action="store_true",
                    help="development run: check errors are recorded but do not fail the run")
    args = ap.parse_args(argv)
    args.langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    bad = [s for s in args.langs if s not in LANGS]
    if bad or not args.langs:
        ap.error(f"--langs {bad or args.langs}: choose from {LANGS}")
    args.only = [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in args.only if s not in GROUPS]
    if bad or not args.only:
        ap.error(f"--only {bad or args.only}: choose from {GROUPS}")
    try:
        w, h = (int(v) for v in args.anim_size.lower().split("x"))
    except ValueError:
        ap.error(f"--anim-size {args.anim_size}: WxH")
    args.anim_size = (w, h)
    if args.main_index < 1:
        ap.error("--main-index is 1-based")
    return args


def main(argv=None) -> int:
    started = time.time()
    args = parse_args(argv)
    exp_root = Path(args.exp_root)
    metrics_dir = Path(args.metrics_dir or exp_root / "metrics")
    ctx = Context(exp_root=exp_root, metrics_dir=metrics_dir, cases_path=Path(args.cases or metrics_dir / "cases.json"),
                  out_dir=Path(args.out_dir or exp_root / "figures"), langs=args.langs,
                  groups=[g for g in GROUPS if g in args.only], main_index=args.main_index,
                  dumps_root=args.dumps_root, topdown_root=Path(args.topdown_root or exp_root / "topdown"),
                  clip_root=args.clip_root, anim_size=args.anim_size, draft=args.draft)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    (ctx.out_dir / "supp").mkdir(exist_ok=True)
    removed = clean_previous(ctx.out_dir) if args.clean else []
    if removed:
        print(f"[make_all] --clean removed {len(removed)} files of the previous run")
    if ctx.cases_path.is_file():
        ctx.cases = json.loads(ctx.cases_path.read_text(encoding="utf-8"))
    else:
        print(f"[make_all] no cases.json at {ctx.cases_path} (run scripts/exp18/select_cases.py): fig1, fig2, fig4, "
              f"supp and anim are skipped")
    if (metrics_dir / "metrics.json").is_file():
        ctx.metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    else:
        print(f"[make_all] no metrics.json in {metrics_dir}: fig3 is skipped")
    git = resolve_git_sha(SOURCE_ROOT)
    digest = code_digest(SOURCE_ROOT)
    fonts = font_report(ctx.langs)
    print(f"[make_all] code {git.get('sha')} ({git.get('source')}), out {ctx.out_dir}, langs {ctx.langs}, "
          f"figures {ctx.groups}" + (" [draft]" if ctx.draft else ""), flush=True)

    if {"fig1", "anim", "supp"} & set(ctx.groups):
        prepare_main_case(ctx)
        mc = ctx.main_case
        print(f"[make_all] main case: {mc.get('note') or mc.get('reason')}"
              + (f" -> {mc['pick'].get('clip_key')} rows {mc.get('rows')}" if mc.get("pick") else ""), flush=True)
        for g in mc.get("shared_paths") or []:
            print(f"[make_all] candidates {g} share one R2R path (their captions say so)", flush=True)
    steps = {"fig1": run_fig1, "supp": run_supp, "fig2": run_fig2, "fig3": run_fig3, "fig4": run_fig4,
             "anim": run_anim}
    with ctx.spy:
        for g in ("fig1", "fig2", "fig3", "fig4", "supp", "anim"):
            if g in ctx.groups:
                steps[g](ctx)

    manifest = build_manifest(ctx, started, git, digest, fonts, removed)
    manifest["policy_check"] = {"pattern_id": POLICY_ID, "pattern_sha256": POLICY_SHA256,
                                "patterns": {i: PATTERN_NOTES[i] for i, _ in POLICY_PATTERNS + WARN_PATTERNS},
                                "hits": [], "warnings": []}
    manifest["lint"] = {"errors": [], "warnings": [], "checked": {}, "numbers": {}, "conventions": {}}
    readme = write_readme(ctx, manifest)  # first pass: the checks read the README ...
    lint = run_checks(ctx, readme)
    lint["errors"] += manifest_policy_hits(manifest)
    lint["checked"]["manifest"] = 1
    manifest["lint"] = lint
    manifest["policy_check"]["hits"] = [f for f in lint["errors"] if f["rule"] == "policy"]
    manifest["policy_check"]["warnings"] = [f for f in lint["warnings"] if f["rule"] == "policy"]
    manifest["policy_check"]["checked"] = lint["checked"]
    readme = write_readme(ctx, manifest)  # ... and the second pass reports them
    for f in policy_findings(readme.read_text(encoding="utf-8"), "README.md"):
        if f["level"] == "error" and f not in lint["errors"]:
            lint["errors"].append(f)
            manifest["policy_check"]["hits"].append(f)
    (ctx.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n",
                                               encoding="utf-8")

    c = manifest["counts"]
    errs = lint["errors"]
    print(f"[make_all] {c['ok']} ok, {c['skipped']} skipped, {c['failed']} failed, {c['defects']} unaccounted slots, "
          f"{len(errs)} check errors, {len(lint['warnings'])} check warnings, in {manifest['elapsed_s']:.0f} s; "
          f"wrote {ctx.out_dir / 'manifest.json'} and {readme}")
    for e in sorted(ctx.entries, key=figure_order):
        if e.status != "ok":
            print(f"[make_all]   {e.status:7s} {e.fig_id} ({e.lang}): {e.reason or e.error}")
    for f in errs:
        print(f"[make_all]   CHECK {_lint_line(f)[2:]}")
    for f in lint["warnings"]:
        print(f"[make_all]   CHECK WARNING {_lint_line(f)[2:]}")
    if manifest["stale_files"]:
        print(f"[make_all]   {len(manifest['stale_files'])} files in {ctx.out_dir} not produced by this run")
    failed = any(e.status == "failed" for e in ctx.entries)
    if ctx.draft and errs and not failed:
        print("[make_all] draft run: check errors recorded, exit status 0")
        return 0
    return 1 if (failed or errs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
