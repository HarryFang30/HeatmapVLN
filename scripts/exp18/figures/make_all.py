#!/usr/bin/env python3
"""EXP-18: render every paper figure from the metrics and the pre-registered cases, check the set, write a manifest.

Inputs (all overridable): ``<EXP_ROOT>/metrics/metrics.json`` (+ ``slots``,
``episodes`` and ``rows`` tables) written by ``compute_metrics.py`` and
``<EXP_ROOT>/metrics/cases.json`` written by ``select_cases.py``; the dumps,
top-down maps and clips those point to.

Outputs in ``--out-dir`` (default ``<EXP_ROOT>/figures``), each figure in every
``--langs`` language (English files carry no suffix, the others ``_<lang>``):

  fig1_main_case             main case (fig_case): main-figure candidate ``--main-index``
                             (default 1), drawn at its pre-registered key rows
  fig2_gallery               gallery (fig_gallery): 10th / 50th / 90th percentile episode per tier
  fig3_metrics               quantitative figure (fig_metrics)
  fig4_route_<pattern>       designed-route figures (fig_routes): out_and_back, loop
  supp/candidate<r>_<scene>_<clip>   every main-figure candidate (fig_case), r = its rank, titled
                             "candidate r of n" with its rank statistic
  supp/anim_main_case        animation of the fig1 episode (fig_anim): .mp4 + .gif
  manifest.json              every file with its sha256, the code version, the sha256 of the
                             inputs, the main case and how it was chosen, the conventions the
                             set follows, the checks and their findings, skipped / failed figures
  README.md                  the figures with their captions and the check results

Main-figure candidates.  cases.json lists the pre-registered tier-C candidates
(>= 4 scored rows, path >= 8 m, a row whose ground-truth bearings span >= 90 deg;
ranked by |episode PCK@8 - tier median|, top 5).  R2R gives each path 3
instructions, so several candidates can be one trajectory drawn twice; the
candidates keep one episode per reference path (``(scene, trajectory_id)`` from
the dump's metadata, else a hash of its reference path) and the next-ranked
episode fills the slot.  When cases.json still holds such duplicates, the list
is re-ranked here from the metrics tables with ``select_cases.main_figure``
(the pre-registered rule itself) plus that step, and the manifest says so.

Main case fallback (tier C not scored yet, or no candidate met the criteria):
the same rule applied to tier B (then A, D, E: the first tier with a
candidate), one episode per reference path, the tier's gallery picks
excluded so fig1 never repeats a fig2 tile.  fig1, the animation and the
supplementary candidates are drawn from that list; captions, manifest and
README.md call it a stand-in.  Without the episodes/rows tables there is no
fallback and fig1 is skipped.

Checks (``lint`` in the manifest; every error fails the run).
  * Figure policy (user decision, 2026-09-24): no pose, pose source, odometry
    or pose-arm wording anywhere (pattern stored as an id + sha256 only).
    Scanned: the text every figure actually draws (a hook on
    ``matplotlib.text.Text.draw`` records each string drawn while a job runs,
    animation frames included), the captions, the modules' label tables
    (strings of figures not drawn this run included), README.md and
    manifest.json.  Localisation claims and "heading" wording are listed for
    manual review, not errors.
  * Terminology: the maps are "affordance map"; "heat row", "heatmap",
    "热力行" and the like are errors in figure text, captions and label tables.
  * Set consistency: one frame-label format per language, one qualifier per
    view label ("Front · ..."), one wording of the 0 deg label, one heat-row
    elevation window across fig_case / fig_gallery / fig_anim (recorded under
    ``conventions``), fig1's episode not in the gallery.
  * Parentheses (``ZH_PARENS``): zh figure text half-width () (Droid Sans
    Fallback prints full-width ones with wide gaps); zh captions full-width （）
    around or after Chinese text; en text never full-width.
  * Colour: no colour literal or categorical palette in the figure modules
    reuses the ground-truth blue or the prediction orange (hue within 20 deg,
    saturation >= 0.25) except in a ground-truth / prediction role (the
    enclosing name says gt / truth / history resp. pred / heat, or the line
    carries ``# colour-role: ground truth`` / ``prediction``).
  * Text below ``MIN_FONT_PT`` at print size (static figures).
  * Notes: a figure that leaves a slot's note out ("no room") fails its job.

Warnings.  A job's warnings are gathered from (1) what the figure module
returns (any ``warnings`` / ``problems`` list, any ``notes_dropped`` /
``dropped_notes`` / ``notes_not_drawn`` field, at any depth), (2) Python
warnings raised while it runs (``warnings.catch_warnings(record=True)``), and
(3) lines it prints with a ``[fig_...]`` prefix (kept for modules that still
print).

Usage (repo root on PYTHONPATH; ``scripts/exp18/run_figures.sh`` wraps it):
  python -m scripts.exp18.figures.make_all [--exp-root DIR] [--metrics-dir DIR] [--cases FILE]
      [--out-dir DIR] [--langs en,zh] [--main-index 1] [--only fig1,supp,fig2,fig3,fig4,anim]
      [--dumps-root DIR] [--topdown-root DIR] [--clip-root DIR] [--anim-size 1920x1080] [--clean]
      [--case-layout revised|approved] [--case-options merge_notes,clamp_peaks,letters=slide] [--draft]
fig1 and the candidates use fig_case's revised layout (``CaseOptions.revised()``: notes wrap
instead of being dropped, misses numbered in a lane under the row, fixed row names, ...);
``--case-layout approved`` draws the approved layout instead, and ``--case-options`` overrides
single CaseOptions fields.  A stand-in fig1 carries a banner line saying so; each candidate a
title "candidate k of n" with its rank statistic.
Exit status: 0 when every attempted figure rendered and every check passed; 1 when a figure
failed (an exception, or a note left out) or a check found an error; 2 on bad arguments.
``--draft`` (development) records check errors and left-out notes but does not let them set
the exit status.
"""
from __future__ import annotations

import argparse
import ast
import colorsys
import contextlib
import datetime as _dt
import functools
import hashlib
import io
import json
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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402

SCHEMA = "heatmapvln-exp18-figures-v2"
LANGS = ("en", "zh")
GROUPS = ("fig1", "supp", "fig2", "fig3", "fig4", "anim")
TIER_ORDER = "ABCDE"
FALLBACK_TIERS = ("B", "A", "D", "E")  # main-figure rule applied to the first of these with a candidate
ROUTE_PATTERNS = ("out_and_back", "loop")

TITLES = {
    "fig1": "Main case: predicted affordance map vs ground truth at key positions",
    "fig2": "Gallery: episodes at the 10th / 50th / 90th percentile of each tier",
    "fig3": "Accuracy of the predicted affordance map per tier",
    "fig4": "Designed routes: out-and-back and loop",
    "supp": "Supplement: every main-figure candidate",
    "anim": "Supplement: animation of the main case",
}
PATTERN_TITLES = {"out_and_back": "out-and-back", "loop": "loop"}

# --------------------------------------------------------------------------- #
# Figure policy and the set's conventions
# --------------------------------------------------------------------------- #
POLICY_ID = "exp18-figure-policy-v2"
# (rule, regex): words kept off every figure, caption, label table, README and manifest (error).
# Rule labels are neutral on purpose: they are printed into README.md and manifest.json.
POLICY_RULES = (
    ("forbidden word, group 1", r"\bposes?\b|位姿|姿态"),
    ("forbidden word, group 2", r"odom|\bVO\b|里程|\bAMB3R\b|\bSLAM\b"),
    ("forbidden word, group 3", r"\bGT[- ]?pose|(?:\bGT|真值)[- ]?(?:arms?|臂)"),
)
# Listed for manual review (not errors): localisation claims, pose-like wording.
REVIEW_RULES = (
    ("localisation claim", r"locali[sz]|定位"),
    ("direction-of-travel wording", r"\bheading\b|朝向"),
)
# The maps are called "affordance map" (user decision): error in figure text, captions, label tables.
TERM_RULES = (
    ("name the maps 'affordance map'",
     r"\bheat(?:[- ]?(?:rows?|maps?|strips?|ramps?|lines?))?\b|\bheatmaps?\b|热力|热图"),
)
POLICY_RE = re.compile("|".join(p for _, p in POLICY_RULES), re.I)
_POLICY_C = [(n, re.compile(p, re.I)) for n, p in POLICY_RULES]
_REVIEW_C = [(n, re.compile(p, re.I)) for n, p in REVIEW_RULES]
_TERM_C = [(n, re.compile(p, re.I)) for n, p in TERM_RULES]
POLICY_SHA256 = hashlib.sha256(json.dumps([POLICY_RULES, REVIEW_RULES, TERM_RULES], ensure_ascii=False)
                               .encode("utf-8")).hexdigest()

GT_BLUE, PRED_ORANGE = "#2a78d6", "#eb6834"
COLOUR_ROLES = {  # role -> (reference colour, name tokens that make the colour legitimate)
    "ground truth": (GT_BLUE, {"gt", "truth", "ground", "hist", "history"}),
    "prediction": (PRED_ORANGE, {"pred", "prediction", "predicted", "heat"}),
}
HUE_TOL_DEG, MIN_SAT, MIN_VAL = 20.0, 0.25, 0.2
MIN_FONT_PT = 5.5  # nothing smaller at print size (static figures are drawn at their print size)
FONT_TOL_PT = 0.05
EL_HEAT_MODULES = ("fig_case", "fig_gallery", "fig_anim")  # modules that draw the affordance-map rows
ZH_PARENS = ("zh figure text: half-width () (the CJK font prints full-width ones with wide gaps); zh captions: "
             "full-width （） around or after Chinese text, half-width only for panel letters (a), coordinates and "
             "non-Chinese content; en: never full-width")
COLOUR_MODULES = ("style", "common_draw", "fig_case", "fig_gallery", "fig_metrics", "fig_routes", "fig_anim")
LABEL_MODULES = ("data", "fig_case", "fig_gallery", "fig_metrics", "fig_routes", "fig_anim")

CJK = r"㐀-鿿豈-﫿"
_CJK_RE = re.compile(f"[{CJK}]")
_HALF_PAREN_RE = re.compile(r"\(([^()]*)\)")
_PANEL_RE = re.compile(r"[a-h]")
_TUPLE_RE = re.compile(r"[-−]?\d+(?:\.\d+)?(?:\s*,\s*[-−]?\d+(?:\.\d+)?)+")  # coordinates (32, 32)
FRAME_FORMATS = {  # most specific first; each match is removed before the next pattern runs
    "en": (("frame N of T", r"\bframe\s+\d+\s+of\s+\d+"), ("frame N / T", r"\bframe\s+\d+\s*/\s*\d+"),
           ("frame N", r"\bframe\s+\d+")),
    "zh": (("第 N / T 帧", r"第\s*\d+\s*/\s*\d+\s*帧"), ("第 N 帧 / T", r"第\s*\d+\s*帧\s*/\s*\d+"),
           ("第 N 帧", r"第\s*\d+\s*帧")),
}
VIEW_LABEL_RE = {"en": re.compile(r"^(Front|Right|Back|Left)\s*·\s*(.+)$"),
                 "zh": re.compile(r"^(前|右|后|左)\s*·\s*(.+)$")}
AHEAD_RE = re.compile(r"^0°\s*[(（]\s*(.+?)\s*[)）]$")

# Text that must not be dropped: the figure modules report a note they could not place like this.
DROPPED_RE = re.compile(r"no room for the note|notes? (?:dropped|not drawn|left out)|dropped notes?", re.I)
DROP_KEYS = ("notes_dropped", "dropped_notes", "notes_not_drawn")
DIAG_RE = re.compile(r"^\[(fig_\w+|exp18 figures)\]\s*")
_QUIET_WARNINGS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning, ResourceWarning, ImportWarning)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _raise(msg: str):
    raise RuntimeError(msg)


def lang_stem(name: str, lang: str) -> str:
    return name if lang == "en" else f"{name}_{lang}"


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, s):
        self.stream.write(s)
        self.parts.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()


@contextlib.contextmanager
def diagnostics(printed: List[str], raised: List[str]):
    """Inside the block: keep the figure modules' ``[fig_...]`` lines in ``printed`` and the Python
    warnings raised (not deprecation noise) in ``raised``, echoing both."""
    tee = _Tee(sys.stdout)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(tee):
                yield
        finally:
            for line in "".join(tee.parts).splitlines():
                line = line.strip()
                if DIAG_RE.match(line) and line not in printed:
                    printed.append(line)
            for w in rec:
                if issubclass(w.category, _QUIET_WARNINGS):
                    continue
                text = f"{w.category.__name__}: {w.message}"
                if text not in raised:
                    raised.append(text)
                    print(f"[make_all]   {text} ({Path(w.filename).name}:{w.lineno})", flush=True)


class TextSpy:
    """Records every string matplotlib draws while ``current`` is set: savefig, canvas.draw and the
    animation's draw_artist all pass through ``Text.draw`` (tick labels and annotations included).
    ``current`` maps each string to the smallest font size (pt) it was drawn at."""

    def __init__(self):
        self.current: Optional[Dict[str, float]] = None
        self._orig = None

    def __enter__(self):
        from matplotlib.text import Text

        orig = self._orig = Text.draw
        spy = self

        @functools.wraps(orig)
        def draw(text, renderer, *args, **kwargs):
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
            return orig(text, renderer, *args, **kwargs)

        Text.draw = draw
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            from matplotlib.text import Text

            Text.draw = self._orig
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
    """Manifest view of figure-module stats: the prediction is ``pred``, the second arm left out."""
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
               "vo_bearing_err_median", "abs_diff_from_tier_median", "path_key", "rank_before_dedupe")


def pick_summary(pick: dict) -> dict:
    return public({k: pick[k] for k in PICK_FIELDS if k in pick})


def feedback(res) -> Tuple[List[str], List[str], bool]:
    """Warnings and left-out notes a figure module returned: any ``warnings`` / ``problems`` list and any
    ``notes_dropped`` / ``dropped_notes`` / ``notes_not_drawn`` field, at any depth, tagged with the
    enclosing entry's tier / percentile / key / frame.  Third value: whether a drop field was present."""
    warns: List[str] = []
    dropped: List[str] = []
    has_drop_field = False

    def tag(d: dict) -> str:
        parts = [f"{k} {d[k]}" for k in ("tier", "percentile", "pattern", "key", "frame")
                 if isinstance(d.get(k), (str, int)) and not isinstance(d.get(k), bool)]
        return " ".join(parts)

    def walk(obj):
        nonlocal has_drop_field
        if isinstance(obj, dict):
            t = tag(obj)
            for k, v in obj.items():
                if k in ("warnings", "problems") and isinstance(v, (list, tuple)):
                    warns.extend(f"{t}: {x}" if t else str(x) for x in v if x)
                elif k in DROP_KEYS:
                    has_drop_field = True
                    if isinstance(v, (list, tuple)):
                        dropped.extend(f"{t}: {x}" if t else str(x) for x in v if x is not None)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool) and v:
                        dropped.append(f"{t}: {int(v)} note(s) not drawn" if t else f"{int(v)} note(s) not drawn")
                else:
                    walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(res)
    return warns, dropped, has_drop_field


def prefix_caption(files: Iterable[str], text: str) -> None:
    """Put ``text`` in front of the caption a figure module wrote (the ``*_caption.txt`` among ``files``)."""
    for f in files:
        if str(f).endswith("_caption.txt") and Path(f).is_file():
            body = Path(f).read_text(encoding="utf-8").strip()
            Path(f).write_text(f"{text} {body}\n", encoding="utf-8")


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
    dropped_notes: List[str] = field(default_factory=list)
    texts: Dict[str, float] = field(default_factory=dict)  # every string the figure drew -> smallest pt
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
    case_options: Dict[str, object] = field(default_factory=dict)  # fig_case.CaseOptions fields (fig1, supp)
    case_layout: str = "revised"  # fig_case layout of fig1 and the candidates: revised | approved
    draft: bool = False
    cases: Optional[dict] = None
    metrics: Optional[dict] = None
    entries: List[Entry] = field(default_factory=list)
    main_case: dict = field(default_factory=dict)
    dumps_used: Dict[str, str] = field(default_factory=dict)  # npz path -> role
    spy: TextSpy = field(default_factory=TextSpy)
    _tables: Optional[tuple] = None

    def run(self, group: str, fig_id: str, lang: str, fn: Callable[[], Tuple[List[str], dict]]) -> Entry:
        t0 = time.time()
        print(f"[make_all] {fig_id} ({lang}) ...", flush=True)
        printed: List[str] = []
        raised: List[str] = []
        texts: Dict[str, float] = {}
        self.spy.current = texts
        try:
            with diagnostics(printed, raised):
                files, details = fn()
            e = Entry(group, fig_id, lang, "ok", files=[str(f) for f in files], details=details)
        except Exception as exc:  # one figure failing must not stop the others
            tb = traceback.format_exc().strip().splitlines()
            e = Entry(group, fig_id, lang, "failed", error=f"{type(exc).__name__}: {exc}", details={"traceback": tb[-12:]})
            print(f"[make_all] {fig_id} ({lang}) FAILED: {e.error}\n" + "\n".join(tb[-12:]), flush=True)
        finally:
            self.spy.current = None
        e.seconds = round(time.time() - t0, 1)
        e.texts = texts
        own = list(e.details.pop("warnings", []))
        fb_warn, fb_drop, has_drop_field = e.details.pop("_feedback", ([], [], False))
        # printed lines that repeat a returned message are not listed twice
        tails = [w.split(": ", 1)[-1].strip() for w in fb_warn + fb_drop]
        printed = [p for p in printed if not any(t and p.endswith(t) for t in tails)]
        e.warnings = list(dict.fromkeys(public(w) for w in own + fb_warn + fb_drop + raised + printed))
        e.dropped_notes = [public(w) for w in (fb_drop if has_drop_field else
                                               [w for w in raised + printed if DROPPED_RE.search(w)])]
        if e.status == "ok" and e.dropped_notes:
            # a slot whose note is missing is unaccounted for in the figure: the figure is not usable
            e.status = "failed"
            e.error = (f"{len(e.dropped_notes)} note(s) left out of the figure: " + "; ".join(e.dropped_notes[:3])
                       + (" ..." if len(e.dropped_notes) > 3 else ""))
        self.entries.append(e)
        print(f"[make_all] {fig_id} ({lang}) {e.status} in {e.seconds:.1f} s"
              + (f" ({e.error})" if e.status == "failed" and e.dropped_notes else ""), flush=True)
        return e

    def skip(self, group: str, fig_id: str, reason: str, langs: Optional[List[str]] = None) -> None:
        for lang in langs or self.langs:
            self.entries.append(Entry(group, fig_id, lang, "skipped", reason=reason))
        print(f"[make_all] {fig_id}: skipped ({reason})", flush=True)

    def stem(self, name: str, lang: str, sub: str = "") -> Path:
        d = self.out_dir / sub if sub else self.out_dir
        return d / lang_stem(name, lang)

    def tables(self):
        """(episodes, rows) written by compute_metrics.py, or None when either is missing."""
        if self._tables is None:
            try:
                from scripts.exp18.compute_metrics import read_table

                eps = read_table(self.metrics_dir / "episodes")
                rows = read_table(self.metrics_dir / "rows")
                self._tables = (eps, rows)
            except Exception as exc:  # no tables: no re-ranking, no fallback
                print(f"[make_all] no episodes/rows tables in {self.metrics_dir}: {type(exc).__name__}: {exc}")
                self._tables = (None,)
        return None if self._tables[0] is None else self._tables


# --------------------------------------------------------------------------- #
# Main case: which episodes are candidates, which one is fig1, which rows
# --------------------------------------------------------------------------- #
def resolve_dump(ctx: Context, pick: dict) -> Path:
    from scripts.exp18.figures import fig_routes as fr

    return Path(fr.resolve_dump(pick, str(ctx.dumps_root) if ctx.dumps_root else None))


def path_key(ctx: Context, pick: dict) -> str:
    """The episode's reference path: ``<scene>:trajectory <id>`` from the dump metadata, else a hash of
    the reference path, else the clip itself (then nothing can be merged with it)."""
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


def rank_candidates(ctx: Context, tier: str, exclude: Iterable[str] = ()) -> Optional[dict]:
    """Main-figure candidates of ``tier`` by the pre-registered rule (``select_cases.main_figure``), one
    episode per reference path, episodes in ``exclude`` (clip keys) left out, top ``MAIN['top']``.
    None without the metrics tables."""
    tabs = ctx.tables()
    if tabs is None:
        return None
    from scripts.exp18 import select_cases as sc

    eps, rows = tabs
    saved = sc.MAIN
    sc.MAIN = dict(saved, tier=tier, top=10 ** 9)  # the whole ranking, same criteria / ties / key rows
    try:
        res = sc.main_figure(eps, rows)
    finally:
        sc.MAIN = saved
    out = {"tier": tier, "status": res.get("status"), "reason": res.get("reason"),
           "tier_median": res.get("tier_median_episode_vo_pck8"), "n_episodes": res.get("n_episodes"),
           "n_meeting_criteria": res.get("n_candidates"), "criteria": dict(saved, tier=tier),
           "selected": [], "skipped_same_path": [], "skipped_excluded": []}
    exclude, seen = set(exclude), {}
    for c in res.get("selected") or []:
        if len(out["selected"]) >= saved["top"]:
            break
        if c.get("clip_key") in exclude:
            out["skipped_excluded"].append(c["clip_key"])
            continue
        key = path_key(ctx, c)
        if key in seen:
            out["skipped_same_path"].append({"clip_key": c["clip_key"], "episode_id": c.get("episode_id"),
                                             "same_path_as": seen[key]})
            continue
        seen[key] = c["clip_key"]
        out["selected"].append(dict(c, rank_before_dedupe=c["rank"], rank=len(out["selected"]) + 1, path_key=key))
    return out


def gallery_keys(cases: dict, tier: str) -> List[str]:
    g = (cases.get("gallery") or {}).get(tier) or {}
    return [p.get("clip_key") for p in g.get("picks") or []]


def choose_main_case(ctx: Context) -> dict:
    """The candidates and the fig1 episode (candidate ``main_index``, 1-based)."""
    cases = ctx.cases
    mf = cases.get("main_figure") or {}
    selected = sorted(mf.get("selected") or [], key=lambda s: int(s.get("rank", 0)))
    choice: dict
    if mf.get("status") == "ok" and selected:
        tier = mf.get("tier", "C")
        choice = {"source": "main_candidates", "tier": tier, "candidates_source": "cases.json",
                  "tier_median": mf.get("tier_median_episode_vo_pck8"), "n_meeting_criteria": mf.get("n_candidates")}
        keys = [path_key(ctx, c) for c in selected]
        if len(set(keys)) < len(keys):
            ranked = rank_candidates(ctx, tier)
            if ranked and ranked["selected"]:
                choice.update(candidates_source="re-ranked by make_all: pre-registered rule, one episode per "
                                                "reference path (cases.json repeated a path)",
                              skipped_same_path=ranked["skipped_same_path"], tier_median=ranked["tier_median"])
                selected = ranked["selected"]
            else:
                choice["path_duplicates"] = [k for k in keys if keys.count(k) > 1]
                selected = [dict(c, path_key=k) for c, k in zip(selected, keys)]
        else:
            selected = [dict(c, path_key=k) for c, k in zip(selected, keys)]
    else:
        why = (f"cases.json lists no main-figure candidate: {mf.get('n_candidates', 0)} of "
               f"{mf.get('n_episodes', '?')} tier-{mf.get('tier', 'C')} episodes met the criteria"
               if mf.get("status") == "ok" else
               f"cases.json main_figure status '{mf.get('status', 'absent')}' (tier C not scored yet)")
        tried = []
        for tier in FALLBACK_TIERS:
            ranked = rank_candidates(ctx, tier, exclude=gallery_keys(cases, tier))
            if ranked is None:
                return {"source": None, "reason": why + "; no episodes/rows tables in the metrics directory, so no "
                                                        "fallback (the gallery picks are not reused)", "candidates": []}
            if ranked["selected"]:
                break
            tried.append(f"{tier}: {ranked.get('n_meeting_criteria') or 0} meeting the criteria")
        else:
            return {"source": None, "reason": why + f"; no fallback tier has a candidate ({', '.join(tried)})",
                    "candidates": []}
        selected = ranked["selected"]
        choice = {"source": "fallback_candidates", "tier": tier, "reason": why,
                  "candidates_source": f"main-figure rule applied to tier {tier}, one episode per reference path, "
                                       f"the tier-{tier} gallery picks excluded",
                  "tier_median": ranked["tier_median"], "n_meeting_criteria": ranked["n_meeting_criteria"],
                  "skipped_same_path": ranked["skipped_same_path"], "skipped_gallery": ranked["skipped_excluded"]}
    choice.update(candidates=selected, n_candidates=len(selected))
    if not 1 <= ctx.main_index <= len(selected):
        choice.update(source=None, error=True,
                      reason=f"--main-index {ctx.main_index}: there are {len(selected)} main-figure candidates")
        return choice
    pick = selected[ctx.main_index - 1]
    choice.update(pick=pick, rank=int(pick.get("rank", ctx.main_index)))
    if choice["source"] == "main_candidates":
        choice["note"] = (f"candidate {ctx.main_index} of {len(selected)} pre-registered tier-{choice['tier']} "
                          f"main-figure candidates")
    else:
        choice["note"] = (f"STAND-IN: {choice['reason']}; fig1, the animation and the candidates come from the "
                          f"main-figure rule applied to tier {choice['tier']} (gallery picks excluded, one episode "
                          f"per reference path); fig1 is candidate {ctx.main_index} of {len(selected)}")
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
    return alt, (f"key rows {rows} coincide; fig_case rule instead (a middle scored frame replaces the "
                 f"duplicate): {alt}")


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


# Captions / titles saying where a candidate stands (en, zh).
CAND_TITLE = {"en": "Main-figure candidate {k} of {n}", "zh": "主图候选 {k}(共 {n} 个)"}  # drawn: half-width
CAND_NOTE = {"en": "|episode PCK@8 − tier median| = {d:.3f}{extra}", "zh": "|该集 PCK@8 − 该层中位数| = {d:.3f}{extra}"}
CAND_NOTE_STANDIN = {"en": "  ·  stand-in from {tier_name}", "zh": "  ·  取自{tier_name}的替代"}
CAND_CAPTION = {
    "en": "Main-figure candidate {k} of {n} ({tier_name}), ranked by |episode PCK@8 − tier median| = {d:.3f} "
          "(tier median {med:.3f}); candidates have ≥ 4 scored frames, a path ≥ 8 m and a frame whose past positions "
          "span ≥ 90° of bearing, one episode per reference path.",
    "zh": "主图候选 {k}（共 {n} 个，{tier_name}），按 |该集 PCK@8 − 该层中位数| = {d:.3f} 升序排列（该层中位数 {med:.3f}）；"
          "候选须有 ≥ 4 个评分帧、路径 ≥ 8 m、且至少一帧的历史位置方位跨度 ≥ 90°，每条参考路径只取一集。",
}
STANDIN_BANNER = {  # drawn above fig1 when it is a stand-in (fig_case CaseOptions.banner)
    "en": "Stand-in until the unseen scenes are scored: main-figure rule applied to {tier_name}, candidate {k} of {n}, "
          "gallery episodes excluded",
    "zh": "未见场景评分之前的替代：主图规则用于{tier_name}，候选 {k}/{n}，已排除画廊所用的集",
}
STANDIN_CAPTION = {
    "en": "[Stand-in until the unseen scenes are scored: the main-figure rule applied to {tier_name}, the gallery "
          "episodes excluded.]",
    "zh": "【未见场景评分之前的替代：主图规则用于{tier_name}，已排除画廊所用的集。】",
}


TIER_PLURAL_EN = {"A": "training scenes", "B": "held-out scenes", "C": "unseen scenes",
                  "D": "HM3D scenes (cross-dataset)", "E": "designed routes"}


def _tier_name(tier: str, lang: str) -> str:
    """Tier as a plural noun phrase ("held-out scenes" / "留出场景")."""
    if lang != "zh":
        return TIER_PLURAL_EN.get(tier, f"tier {tier}")
    from scripts.exp18.figures import data as dd

    return getattr(dd, "TIER_NAMES_ZH", {}).get(tier, f"{tier} 层")


def candidate_label(mc: dict, pick: dict, lang: str) -> dict:
    """Title, title note and caption prefix of a supplementary candidate figure."""
    k, n = int(pick.get("rank", 0)), int(mc.get("n_candidates", 0))
    d = float(pick.get("abs_diff_from_tier_median") or 0.0)
    med = float(mc.get("tier_median") or 0.0)
    tname = _tier_name(mc.get("tier", "C"), lang)
    stand_in = mc.get("source") == "fallback_candidates"
    extra = CAND_NOTE_STANDIN[lang].format(tier_name=tname) if stand_in else ""
    cap = CAND_CAPTION[lang].format(k=k, n=n, d=d, med=med, tier_name=tname)
    if stand_in:
        cap = STANDIN_CAPTION[lang].format(tier_name=tname) + " " + cap
    return {"title": CAND_TITLE[lang].format(k=k, n=n), "title_note": CAND_NOTE[lang].format(d=d, extra=extra),
            "caption": cap}


# --------------------------------------------------------------------------- #
# Figure jobs
# --------------------------------------------------------------------------- #
def case_options(ctx: Context, **extra):
    """fig_case.CaseOptions for fig1 / a candidate: the revised layout (unless --case-layout approved),
    --case-options on top, then ``extra`` (title, banner, ...)."""
    from scripts.exp18.figures import fig_case as fc

    kw = dict(ctx.case_options)
    kw.update({k: v for k, v in extra.items() if v is not None})
    if ctx.case_layout == "revised":
        if not hasattr(fc.CaseOptions, "revised"):
            raise RuntimeError("fig_case has no CaseOptions.revised(); use --case-layout approved")
        return fc.CaseOptions.revised(**kw)
    return fc.CaseOptions(**kw) if kw else None


def job_case(ctx: Context, pick: dict, rows: List[int], stem: Path, lang: str, rule: str,
             label: Optional[dict] = None, caption_prefix: str = "", banner: Optional[str] = None):
    from scripts.exp18.figures import fig_case as fc

    npz = resolve_dump(ctx, pick)
    opts = case_options(ctx, title=label["title"] if label else None,
                        title_note=label["title_note"] if label else None, banner=banner)
    res = fc.make_case_figure(str(npz), rows=rows, topdown_root=str(ctx.topdown_root),
                              clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                              out_stem=str(stem), lang=lang, options=opts)
    prefix = " ".join(s for s in (caption_prefix, label["caption"] if label else "") if s)
    if prefix:
        prefix_caption(res["files"], prefix)
    details = {"case": pick_summary(pick), "rows": res["rows"], "rows_rule": rule,
               "frames": [s["frame"] for s in res["stats"]], "stats": public(res["stats"]),
               "layout": ctx.case_layout, "layout_decisions": public(res.get("layout")),
               "size_in": [round(v, 3) for v in res["size_in"]], "_feedback": feedback(res)}
    return res["files"], details


def standin_prefix(mc: dict, lang: str) -> str:
    if mc.get("source") != "fallback_candidates":
        return ""
    return STANDIN_CAPTION[lang].format(tier_name=_tier_name(mc["tier"], lang))


def standin_banner(mc: dict, lang: str) -> Optional[str]:
    if mc.get("source") != "fallback_candidates":
        return None
    return STANDIN_BANNER[lang].format(tier_name=_tier_name(mc["tier"], lang), k=mc.get("rank"),
                                       n=mc.get("n_candidates"))


def run_fig1(ctx: Context) -> None:
    mc = ctx.main_case
    if mc.get("source") is None:
        if mc.get("error"):
            for lang in ctx.langs:
                ctx.entries.append(Entry("fig1", "fig1_main_case", lang, "failed", error=mc["reason"]))
            print(f"[make_all] fig1_main_case FAILED: {mc['reason']}")
        else:
            ctx.skip("fig1", "fig1_main_case", mc.get("reason", "no main case"))
        return
    for lang in ctx.langs:
        if mc.get("prepare_error"):
            ctx.run("fig1", "fig1_main_case", lang, lambda: _raise(mc["prepare_error"]))
            continue
        e = ctx.run("fig1", "fig1_main_case", lang,
                    lambda lang=lang: job_case(ctx, mc["pick"], mc["rows"], ctx.stem("fig1_main_case", lang), lang,
                                               mc["rows_rule"], caption_prefix=standin_prefix(mc, lang),
                                               banner=standin_banner(mc, lang)))
        if mc["source"] == "fallback_candidates":
            e.warnings.append(mc["note"])


def run_supp(ctx: Context) -> None:
    mc = ctx.main_case
    cands = mc.get("candidates") or []
    if not cands:
        reason = "no main-figure candidates" + (f": {mc['reason']}" if mc.get("reason") else "")
        ctx.skip("supp", "supp_candidates", reason)
        return
    from scripts.exp18.figures import data as dd

    for c in cands:
        rank = int(c.get("rank", cands.index(c) + 1))
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
                    lambda lang=lang, c=c, name=name: job_case(ctx, c, prep["rows"], ctx.stem(name, lang, "supp"),
                                                               lang, prep["rule"], label=candidate_label(mc, c, lang)))


def gallery_tiers(cases: dict) -> Tuple[List[str], List[str]]:
    gal = cases.get("gallery") or {}
    present = [t for t in TIER_ORDER if (gal.get(t) or {}).get("status") == "ok" and (gal[t].get("picks") or [])]
    return present, [t for t in TIER_ORDER if t not in present]


def run_fig2(ctx: Context) -> None:
    if ctx.cases is None:
        ctx.skip("fig2", "fig2_gallery", "no cases.json")
        return
    present, missing = gallery_tiers(ctx.cases)
    if not present:
        ctx.skip("fig2", "fig2_gallery", "cases.json has no gallery picks for any tier")
        return
    for t in present:
        for p in ctx.cases["gallery"][t]["picks"]:
            try:
                ctx.dumps_used.setdefault(str(resolve_dump(ctx, p)), f"gallery {t} P{p.get('percentile')}")
            except FileNotFoundError:
                pass  # the gallery draws a placeholder tile and reports it

    def job(lang: str):
        from scripts.exp18.figures import fig_gallery as fg

        res = fg.make_gallery_figure(ctx.cases, dumps_root=str(ctx.dumps_root) if ctx.dumps_root else None,
                                     topdown_root=str(ctx.topdown_root),
                                     clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                                     out_stem=str(ctx.stem("fig2_gallery", lang)), lang=lang)
        warnings_ = [f"tile {t['tier']} P{t['percentile']} drawn as a placeholder: {t['error']}"
                     for t in res.get("tiles", []) if t.get("error")]
        details = {"tiers": res.get("tiers"), "missing_tiers": res.get("missing_tiers"),
                   "tiles": public(res.get("tiles")), "size_in": [round(v, 3) for v in res["size_in"]],
                   "warnings": [public(w) for w in warnings_], "_feedback": feedback(res)}
        return res["files"], details

    for lang in ctx.langs:
        ctx.run("fig2", "fig2_gallery", lang, lambda lang=lang: job(lang))


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
    ok = [p for p in ROUTE_PATTERNS if isinstance(pf.get(p), dict) and pf[p].get("status") == "ok"]
    for p in ROUTE_PATTERNS:
        if p not in ok:
            status = pf[p].get("status") if isinstance(pf.get(p), dict) else "absent"
            ctx.skip("fig4", f"fig4_route_{p}", f"no tier-E {PATTERN_TITLES[p]} pick in cases.json (status {status}; "
                                                f"tiers present: {', '.join(ctx.cases.get('tiers_present') or [])})")
    if not ok:
        return
    for p in ok:
        try:
            ctx.dumps_used.setdefault(str(resolve_dump(ctx, pf[p])), f"route {p}")
        except FileNotFoundError:
            pass

    import tempfile

    from scripts.exp18.figures import fig_routes as fr

    def job(p: str, lang: str):
        # fig_routes reads its picks from a cases.json file; hand it one holding only pattern p, so a
        # pattern that fails (a dump missing, key rows not matching) does not take the other one with it
        one = dict(ctx.cases, pattern_figure={p: pf[p]})
        with tempfile.TemporaryDirectory(prefix="exp18_fig4_") as tmp:
            cases_p = Path(tmp) / f"cases_{p}.json"
            cases_p.write_text(json.dumps(one, ensure_ascii=False), encoding="utf-8")
            res = fr.make_route_figures(str(cases_p), dumps_root=str(ctx.dumps_root) if ctx.dumps_root else None,
                                        topdown_root=str(ctx.topdown_root),
                                        clip_root_override=str(ctx.clip_root) if ctx.clip_root else None,
                                        out_dir=str(ctx.out_dir), lang=lang)
        if p not in res["figures"]:
            raise RuntimeError(f"fig_routes drew no {p} figure: {res['skipped'].get(p, 'no reason given')}")
        r = res["figures"][p]
        files = []
        for f in r["files"]:  # route_<pattern>[_zh].* -> fig4_route_<pattern>[_zh].*
            src = Path(f)
            dst = src.with_name("fig4_" + src.name)
            os.replace(src, dst)
            files.append(str(dst))
        details = {"case": pick_summary(pf[p]), "rows": r.get("rows"), "roles": r.get("roles"),
                   "split_frame": r.get("split_frame"), "split_rule": r.get("split_rule"),
                   "stand_in": r.get("stand_in"), "front_view_totals": public(r.get("tally_totals")),
                   "stats": public(r.get("stats")), "size_in": [round(v, 3) for v in r["size_in"]],
                   "_feedback": feedback(r)}
        if r.get("stand_in"):
            details["warnings"] = ["development stand-in: the pick is not a tier-E dump"]
        return files, details

    for p in ok:
        for lang in ctx.langs:
            ctx.run("fig4", f"fig4_route_{p}", lang, lambda p=p, lang=lang: job(p, lang))
    # fig_routes announces the patterns left out of the one-pattern file; those are not warnings
    for e in ctx.entries:
        if e.group == "fig4":
            e.warnings = [w for w in e.warnings if not re.match(r"\[fig_routes\] \w+: skipped", w)]


def run_anim(ctx: Context) -> None:
    mc = ctx.main_case
    if mc.get("source") is None:
        if mc.get("error"):
            for lang in ctx.langs:
                ctx.entries.append(Entry("anim", "supp_anim_main_case", lang, "failed", error=mc["reason"]))
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
        prefix = standin_prefix(mc, lang)
        if prefix:
            prefix_caption(res["files"], prefix)
        details = {"case": pick_summary(mc["pick"]), "rows": res.get("rows"), "video_frames": res.get("frames"),
                   "duration_s": round(res.get("duration_s", 0.0), 2), "size_px": list(res.get("size", ())),
                   "fps": res.get("fps"), "encoder": res.get("encoder"), "render_s": round(res.get("render_s", 0.0), 1),
                   "_feedback": feedback(res)}
        if mc["source"] == "fallback_candidates":
            details["warnings"] = [mc["note"]]
        return res["files"], details

    for lang in ctx.langs:
        ctx.run("anim", "supp_anim_main_case", lang, lambda lang=lang: job(lang))


# --------------------------------------------------------------------------- #
# Checks: policy, terminology, zh parentheses, set consistency, colour, font size
# --------------------------------------------------------------------------- #
def _finding(level: str, rule: str, where: str, match: str = "", context: str = "", **extra) -> dict:
    d = {"level": level, "rule": rule, "where": where}
    if match:
        d["match"] = match
    if context:
        d["context"] = context
    d.update(extra)
    return d


def _ctx(s: str, m: re.Match, pad: int = 40) -> str:
    return s[max(0, m.start() - pad): m.end() + pad].replace("\n", " ")


def word_findings(s: str, where: str, terms: bool = True, review: bool = True) -> List[dict]:
    out = []
    for rules, level, on in ((_POLICY_C, "error", True), (_TERM_C, "error", terms), (_REVIEW_C, "review", review)):
        if not on:
            continue
        for name, rx in rules:
            for m in rx.finditer(s):
                out.append(_finding(level, name if level != "error" or rules is _TERM_C else f"policy: {name}",
                                    where, m.group(0), _ctx(s, m)))
    return out


def paren_findings(s: str, lang: str, where: str, kind: str) -> List[dict]:
    """The parentheses convention (``ZH_PARENS``); ``kind`` is "figure" (drawn text) or "caption"."""
    out = []
    full = re.search("[（）]", s)
    if lang != "zh" or kind == "figure":
        if full:
            rule = ("zh figure text: use half-width ()" if lang == "zh" else "en text with a full-width parenthesis")
            out.append(_finding("error", rule, where, full.group(0), _ctx(s, full, 20)))
        return out
    for m in _HALF_PAREN_RE.finditer(s):  # zh caption
        inner = m.group(1).strip()
        if _PANEL_RE.fullmatch(inner) or _TUPLE_RE.fullmatch(inner):
            continue  # panel letter (a), coordinates (32, 32)
        before = s[:m.start()].rstrip()
        if _CJK_RE.search(inner) or (before and _CJK_RE.match(before[-1])):
            out.append(_finding("error", "zh caption: use full-width （）", where, m.group(0), _ctx(s, m, 20)))
    return out


def text_findings(s: str, lang: str, where: str, kind: str) -> List[dict]:
    return word_findings(s, where) + paren_findings(s, lang, where, kind)


def _label_strings(obj, lang: Optional[str], path: str):
    """(string, lang, path) for every string in a label table (lang from an 'en'/'zh' key on the way down)."""
    if isinstance(obj, str):
        yield obj, lang, path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _label_strings(v, k if k in LANGS else lang, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _label_strings(v, lang, f"{path}[{i}]")


def _import_fig_module(name: str):
    import importlib

    return importlib.import_module(f"scripts.exp18.figures.{name}")


def label_table_findings() -> Tuple[List[dict], int]:
    """Policy words and terminology in every module-level label table (``LABELS``, ``CAPTION*``,
    ``*_TEXT``, ``*NAMES*``, ...), including strings of figures not drawn this run.  (Parentheses are
    checked on what is drawn and on the captions: label tables also hold a layout's legacy wording.)"""
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
            for s, _lang, path in _label_strings(val, None, f"{mod_name}.{name}"):
                n += 1
                out += word_findings(s, path, review=False)
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
    """Colours that read as the ground-truth blue or the prediction orange, used for anything else."""
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
                                f"{mod_name}.py:{node.lineno}", node.value,
                                f"{' / '.join(names[:4]) or 'module level'}: {line.strip()[:100]}"))
        # categorical palettes built at runtime (e.g. from constants)
        for name, val in vars(mod).items():
            if not re.search(r"colou?rs?$|palette", name, re.I):
                continue
            items = val.items() if isinstance(val, dict) else enumerate(val) if isinstance(val, (list, tuple)) else ()
            for key, c in items:
                role = colour_role(c)
                if role is None or (_tokens(str(key)) | _tokens(name)) & COLOUR_ROLES[role][1]:
                    continue
                if (mod_name, str(c).strip().lower()) in reported:
                    continue  # the literal is already reported
                out.append(_finding("error", f"categorical palette reuses the {role} colour",
                                    f"{mod_name}.{name}[{key!r}]", str(c)))
    return out


def heat_windows() -> Dict[str, Optional[float]]:
    out = {}
    for mod_name in EL_HEAT_MODULES:
        try:
            v = getattr(_import_fig_module(mod_name), "EL_HEAT", None)
            out[mod_name] = float(v) if v is not None else None
        except Exception:
            out[mod_name] = None
    try:
        from scripts.exp18.figures import common_draw as cd

        for name in ("EL_HEAT", "HEAT_EL_DEG", "HEAT_ELEV_DEG"):
            if hasattr(cd, name):
                out[f"common_draw.{name}"] = float(getattr(cd, name))
    except Exception:
        pass
    return out


def frame_formats(s: str, lang: str) -> List[str]:
    found = []
    for line in s.splitlines():
        for fmt, rx in FRAME_FORMATS.get(lang, ()):
            line, n = re.subn(rx, " ", line, flags=re.I)
            if n:
                found.append(fmt)
    return found


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
        if len(fmts) > 1:
            out.append(_finding("error", "frame labels use more than one format", f"all figures ({lang})",
                                context="; ".join(f"'{f}' in {', '.join(sorted(v))}" for f, v in sorted(fmts.items()))))
    for (lang, view), quals in views.items():
        if len(quals) > 1:
            out.append(_finding("error", "one view is labelled in more than one way", f"all figures ({lang})",
                                context="; ".join(f"'{view} · {q}' in {', '.join(sorted(v))}"
                                                  for q, v in sorted(quals.items()))))
    for lang, quals in ahead.items():
        if len(quals) > 1:
            out.append(_finding("error", "the 0° label is worded in more than one way", f"all figures ({lang})",
                                context="; ".join(f"'0° ({q})' in {', '.join(sorted(v))}" for q, v in sorted(quals.items()))))
    windows = heat_windows()
    known = {k: v for k, v in windows.items() if v is not None}
    if len(set(known.values())) > 1:
        out.append(_finding("error", "the affordance-map rows span different elevation windows",
                            ", ".join(sorted(known)),
                            context="; ".join(f"{k} ±{v:g}°" for k, v in sorted(known.items()))))
    # fig1 must not repeat a gallery tile
    mc = ctx.main_case
    drawn = {e.group for e in ctx.entries if e.status != "skipped"}
    if mc.get("pick") and {"fig1", "fig2"} <= drawn and ctx.cases:
        t, key = mc["pick"].get("tier"), mc["pick"].get("clip_key")
        if key in gallery_keys(ctx.cases, t):
            out.append(_finding("error", "fig1 shows an episode that is also a gallery tile", "fig1_main_case, fig2_gallery",
                                context=f"tier {t} {key}; pick another candidate with --main-index"))
    if mc.get("path_duplicates"):
        out.append(_finding("error", "main-figure candidates repeat a reference path (no metrics tables to re-rank)",
                            "supp", context=", ".join(sorted(set(mc["path_duplicates"])))))
    conventions = {
        "ground_truth_colour": GT_BLUE, "prediction_colour": PRED_ORANGE,
        "prediction_shown": "the deployed model's output",
        "zh_parentheses": ZH_PARENS,
        "min_font_pt": MIN_FONT_PT,
        "heat_row_elevation_deg": windows,
        "frame_label_formats": {lang: sorted(f) for lang, f in frames.items()},
        "view_labels": {f"{lang}:{view}": sorted(q) for (lang, view), q in sorted(views.items())},
        "zero_bearing_label": {lang: sorted(q) for lang, q in ahead.items()},
    }
    return out, conventions


def run_checks(ctx: Context, readme: Path) -> dict:
    """Every check except the manifest scan (done when the manifest is written)."""
    findings: List[dict] = []
    checked = {"figure_texts": 0, "strings_drawn": 0, "captions": 0, "label_strings": 0, "readme": 0}
    for e in ctx.entries:
        if e.status == "skipped":
            continue
        where = f"{e.fig_id} ({e.lang})"
        if e.texts:
            checked["figure_texts"] += 1
            checked["strings_drawn"] += len(e.texts)
        for s in e.texts:
            findings += text_findings(s, e.lang, f"{where} figure text", "figure")
        if e.group != "anim" and e.texts:
            small = sorted((fs, s) for s, fs in e.texts.items() if fs < MIN_FONT_PT - FONT_TOL_PT)
            if small:
                findings.append(_finding("error", f"text below {MIN_FONT_PT} pt at print size", where,
                                         context="; ".join(f"{fs:.2f} pt '{s[:30]}'" for fs, s in small[:6])
                                         + (f" (+{len(small) - 6} more)" if len(small) > 6 else "")))
        cap = caption_of(e)
        if cap is not None and cap.is_file():
            checked["captions"] += 1
            findings += text_findings(cap.read_text(encoding="utf-8"), e.lang, f"{where} caption", "caption")
    lab, n = label_table_findings()
    checked["label_strings"] = n
    findings += lab
    findings += colour_findings()
    sf, conventions = set_findings(ctx)
    findings += sf
    if readme.is_file():
        checked["readme"] = 1
        findings += word_findings(readme.read_text(encoding="utf-8"), "README.md", terms=False, review=False)
    # one finding per (rule, where, match)
    uniq = list({(f["level"], f["rule"], f["where"], f.get("match", ""), f.get("context", "")): f
                 for f in findings}.values())
    return {"errors": [f for f in uniq if f["level"] == "error"], "review": [f for f in uniq if f["level"] == "review"],
            "checked": checked, "conventions": conventions}


def manifest_policy_hits(manifest: dict) -> List[dict]:
    """Policy words in the manifest itself (tracebacks and error texts are diagnostics and skipped)."""
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in ("traceback", "error", "lint", "policy_check")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    text = json.dumps(strip(manifest), ensure_ascii=False, default=str)
    return [f for f in word_findings(text, "manifest.json", terms=False, review=False)]


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


SECTION_ORDER = ("fig1", "fig2", "fig3", "fig4", "supp", "anim")


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
    """README line for a check finding; policy words themselves are not repeated (they are in the manifest)."""
    shown = "" if f["rule"].startswith("policy") else (f" «{f['match']}»" if f.get("match") else "")
    ctx_ = "" if f["rule"].startswith("policy") or f.get("match") else (f": {f['context']}" if f.get("context") else "")
    return f"- {f['rule']} — {f['where']}{shown}{ctx_}"


def write_readme(ctx: Context, manifest: dict) -> Path:
    L: List[str] = []
    gi = manifest["code"]["git"]
    sha = gi.get("sha", "unknown") + ("-dirty" if gi.get("dirty") else "")
    L.append("# EXP-18 figures\n")
    L.append(f"Rendered {manifest['created_utc'][:19].replace('T', ' ')} UTC by "
             f"`scripts/exp18/figures/make_all.py` (code `{sha}`, EXP-18 sources sha256 "
             f"`{manifest['code']['digest']['sha256'][:16]}…`). Languages: {', '.join(ctx.langs)}."
             + (" **Draft run** (check errors and left-out notes do not fail it)." if ctx.draft else "") + "\n")
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
        L.append(f"**Main case (fig1, animation).** {mc['note']}: {c.get('tier')} `{c.get('clip_key')}`, episode "
                 f"{c.get('episode_id')}, frames {mc.get('frames')} ({mc.get('rows_rule')}). Candidates: "
                 f"{mc.get('candidates_source')}; all {mc['n_candidates']} are in `supp/`.\n")
    elif mc.get("source") == "fallback_candidates":
        c = mc["case"]
        L.append(f"> **Main case (fig1, animation): STAND-IN.** {mc['reason']}. fig1, the animation and `supp/` use "
                 f"the main-figure rule applied to tier {mc['tier']} (one episode per reference path, the tier-"
                 f"{mc['tier']} gallery picks excluded, so fig1 repeats no gallery tile): candidate "
                 f"{mc.get('main_index')} of {mc['n_candidates']}, `{c.get('clip_key')}`, episode "
                 f"{c.get('episode_id')}, frames {mc.get('frames')}. Re-run once tier C is scored.\n")
    else:
        L.append(f"> **Main case: none.** {mc.get('reason', '')}\n")

    lint = manifest.get("lint", {})
    errs, rev = lint.get("errors", []), lint.get("review", [])
    L.append("## Checks\n")
    L.append(f"Policy words (id `{POLICY_ID}`): {len(manifest['policy_check']['hits'])} hit(s) in "
             f"{lint.get('checked', {}).get('strings_drawn', 0)} strings drawn by "
             f"{lint.get('checked', {}).get('figure_texts', 0)} figures, {lint.get('checked', {}).get('captions', 0)} "
             f"captions, {lint.get('checked', {}).get('label_strings', 0)} label-table strings, README.md and "
             f"manifest.json. Check errors: **{len(errs)}**. Items for manual review: {len(rev)}. "
             f"Notes left out of figures: **{manifest['counts']['notes_not_drawn']}**.\n")
    if errs:
        L.append("**Errors** (each fails the run):\n")
        L += [_lint_line(f) for f in errs]
        L.append("")
    if rev:
        byrule: Dict[str, List[str]] = {}
        for f in rev:
            byrule.setdefault(f"{f['rule']} «{f.get('match', '')}»", []).append(f["where"])
        L.append("**For manual review** (not errors):\n")
        for k, where in byrule.items():
            L.append(f"- {k}: {len(where)} place(s), e.g. {', '.join(where[:3])}")
        L.append("")
    conv = lint.get("conventions", {})
    if conv:
        win = conv.get("heat_row_elevation_deg", {})
        L.append("**Conventions.** Ground truth blue `" + conv["ground_truth_colour"] + "`, prediction orange `"
                 + conv["prediction_colour"] + "`; zh parentheses: " + conv["zh_parentheses"]
                 + f"; text ≥ {conv['min_font_pt']} pt; affordance-map rows ±elevation: "
                 + ", ".join(f"{k} {v:g}°" if v is not None else f"{k} n/a" for k, v in win.items())
                 + "; frame labels: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in conv.get("frame_label_formats", {}).items())
                 + ".\n")

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
    warn.pop(mc.get("note"), None)  # the main-case note is the banner above
    if warn:
        L.append("**Warnings.**\n")
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
    sections += [("supp", fid) for fid in cand_ids] + [("anim", "supp_anim_main_case")]
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
                     + (f": `{c['clip_key']}`" if c.get("clip_key") else "")
                     + (f", |episode PCK@8 − tier median| = {c['abs_diff_from_tier_median']:.3f}"
                        if c.get("abs_diff_from_tier_median") is not None else ""))
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
                                                 "candidates_source", "tier_median", "n_meeting_criteria",
                                                 "skipped_same_path", "skipped_gallery", "path_duplicates", "npz",
                                                 "rows", "frames", "rows_rule", "prepare_error")
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
                        "seconds": e.seconds, "warnings": e.warnings, "notes_dropped": e.dropped_notes,
                        "strings_drawn": len(e.texts), "min_font_pt": round(min(static), 2) if static else None,
                        "details": d})
    counts = {s: sum(1 for e in ctx.entries if e.status == s) for s in ("ok", "skipped", "failed")}
    # a note a figure had no room for leaves that slot unaccounted for in the figure; its job failed
    counts["notes_not_drawn"] = sum(len(e.dropped_notes) for e in ctx.entries)
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
                 "anim_size": list(ctx.anim_size), "case_layout": ctx.case_layout,
                 "case_options": ctx.case_options or None, "draft": ctx.draft},
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
CASE_FLAGS = ("merge_notes", "clamp_peaks")
CASE_LETTERS = ("outward", "slide", "inside")


def parse_case_options(text: str) -> Dict[str, object]:
    """``"merge_notes,clamp_peaks,letters=slide"`` -> fig_case.CaseOptions keyword arguments."""
    out: Dict[str, object] = {}
    for item in (s.strip() for s in text.split(",")):
        if not item:
            continue
        key, _, value = item.partition("=")
        if key in CASE_FLAGS and not value:
            out[key] = True
        elif key == "letters" and value in CASE_LETTERS:
            out[key] = value
        else:
            raise ValueError(f"--case-options {item!r}: use {', '.join(CASE_FLAGS)} or letters={'|'.join(CASE_LETTERS)}")
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp-root", type=Path, default=common.EXP_ROOT, help="default $EXP18_ROOT (common.EXP_ROOT)")
    ap.add_argument("--metrics-dir", type=Path, default=None, help="default <exp-root>/metrics")
    ap.add_argument("--cases", type=Path, default=None, help="default <metrics-dir>/cases.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <exp-root>/figures")
    ap.add_argument("--langs", default="en,zh", help="comma list of en, zh")
    ap.add_argument("--main-index", type=int, default=1, help="main-figure candidate (1-based rank) drawn as fig1")
    ap.add_argument("--only", default=",".join(GROUPS), help=f"comma list of {', '.join(GROUPS)}")
    ap.add_argument("--dumps-root", type=Path, default=None,
                    help="<root>/<tier>/<scene>/<clip>.npz (default: the npz_path recorded in cases.json)")
    ap.add_argument("--topdown-root", type=Path, default=None, help="default <exp-root>/topdown")
    ap.add_argument("--clip-root", type=Path, default=None, help="local copy of the clips (default: as recorded)")
    ap.add_argument("--anim-size", default="1920x1080", help="animation WxH, 16:9")
    ap.add_argument("--clean", action="store_true", help="first delete the files the previous manifest.json listed")
    ap.add_argument("--case-layout", default="revised", choices=("revised", "approved"),
                    help="fig_case layout of fig1 and the candidates (default revised: CaseOptions.revised())")
    ap.add_argument("--case-options", default="",
                    help="single fig_case.CaseOptions fields on top of the layout, comma list of merge_notes, "
                         "clamp_peaks, letters=outward|slide|inside")
    ap.add_argument("--draft", action="store_true",
                    help="development run: check errors and left-out notes are recorded but do not fail the run")
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
    try:
        args.case_options = parse_case_options(args.case_options)
    except ValueError as exc:
        ap.error(str(exc))
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
                  clip_root=args.clip_root, anim_size=args.anim_size, case_options=args.case_options,
                  case_layout=args.case_layout, draft=args.draft)
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
    steps = {"fig1": run_fig1, "supp": run_supp, "fig2": run_fig2, "fig3": run_fig3, "fig4": run_fig4,
             "anim": run_anim}
    with ctx.spy:
        for g in ("fig1", "fig2", "fig3", "fig4", "supp", "anim"):
            if g in ctx.groups:
                steps[g](ctx)

    manifest = build_manifest(ctx, started, git, digest, fonts, removed)
    manifest["policy_check"] = {"pattern_id": POLICY_ID, "pattern_sha256": POLICY_SHA256, "hits": []}
    manifest["lint"] = {"errors": [], "review": [], "checked": {}, "conventions": {}}
    readme = write_readme(ctx, manifest)  # first pass: the checks read the README ...
    lint = run_checks(ctx, readme)
    lint["errors"] += manifest_policy_hits(manifest)
    lint["checked"]["manifest"] = 1
    manifest["lint"] = lint
    manifest["policy_check"]["hits"] = [f for f in lint["errors"] if f["rule"].startswith("policy")]
    manifest["policy_check"]["checked"] = lint["checked"]
    readme = write_readme(ctx, manifest)  # ... and the second pass reports them
    for f in word_findings(readme.read_text(encoding="utf-8"), "README.md", terms=False, review=False):
        if f not in lint["errors"]:
            lint["errors"].append(f)
            manifest["policy_check"]["hits"].append(f)
    (ctx.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n",
                                               encoding="utf-8")

    c = manifest["counts"]
    errs = lint["errors"]
    print(f"[make_all] {c['ok']} ok, {c['skipped']} skipped, {c['failed']} failed, {c['notes_not_drawn']} notes not "
          f"drawn, {len(errs)} check errors, {len(lint['review'])} for review, in {manifest['elapsed_s']:.0f} s; "
          f"wrote {ctx.out_dir / 'manifest.json'} and {readme}")
    for e in sorted(ctx.entries, key=figure_order):
        if e.status != "ok":
            print(f"[make_all]   {e.status:7s} {e.fig_id} ({e.lang}): {e.reason or e.error}")
    for w in [(e.fig_id, e.lang, w) for e in ctx.entries for w in e.warnings]:
        print(f"[make_all]   warning {w[0]} ({w[1]}): {w[2]}")
    for f in errs:
        print(f"[make_all]   CHECK {_lint_line(f)[2:]}" + (f" :: {f['context']}" if f.get("context") and f.get("match")
                                                           and not f["rule"].startswith("policy") else ""))
    if manifest["stale_files"]:
        print(f"[make_all]   {len(manifest['stale_files'])} files in {ctx.out_dir} not produced by this run")
    crashed = any(e.status == "failed" and not e.dropped_notes for e in ctx.entries)
    defects = bool(errs) or any(e.dropped_notes for e in ctx.entries)
    if ctx.draft and defects and not crashed:
        print("[make_all] draft run: check errors / left-out notes recorded, exit status 0")
        return 0
    return 1 if (crashed or defects) else 0


if __name__ == "__main__":
    raise SystemExit(main())
