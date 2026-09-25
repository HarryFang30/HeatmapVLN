#!/usr/bin/env python3
"""EXP-19 figures v2 [A]: animations of the main-case reruns -- how the affordance maps evolve ONLINE.

One MP4 (H.264, yuv420p, 12 fps, 1920 x 1426) and one light GIF preview (<= 900 px
wide, every 2nd frame, same duration) per main case and language, written to
``<out-dir>/<category>_<ep_key>_{zh,en}.{mp4,gif}`` plus ``<out-dir>/manifest.json``.

One frame per executed step t = 0 .. episode_steps (the recorded states; the
camera image is ``front_<t>.jpg``, the native frame the robot saw at step t).
Frames of steps where System 2 was called with a trajectory answer, and of the
key moments K1-K4, are held a little longer (``--hold-call`` / ``--hold-key``),
so the viewer can read the decision; the video stays at 12 fps.

Frame layout (7.0 x 5.2 in, drawn in the static figure's point sizes and
rendered at 1920 px wide, so every mark has the static figure's proportions)::

  header: category · instruction · "step t / N" (K badge at key moments)
  camera frame (4:3)          | route so far       | 360° affordance maps of the latest call
    + inset: the decision      |  (top-down map)    |   history strip (orange field, dots, circles)
      image of the latest call |  current position  |   future strip (teal field, System 1 path)
      (look-down after "↓"),   |  = plain ink dot   |   legend of the marks
      pixel goal +             |  (4.5 pt, white    |
      System 1 path            |  halo, no ring)    |
  System 2 output · executed action of this step
  online timeline (the static figure's encoding), painted up to the end of step t, playhead there
  legend of the timeline's context marks and of the past frames' order (with the stride on a long rerun);
  bearing / strip note, display-smoothing note and data-flow note

The drawing reuses ``panels_v2`` (strips, decision image, marks, legend glyphs)
and ``timeline_panel`` (raster, spans, key lines, badges, turn track, marker
positions, compressed spans) so the encodings are the static figure's (the
history panel with up = left, behind in the middle): a call's dots and circles
are spread across its column by slot (slot 1 = oldest at the left; frames 1, 4,
8 in narrow columns; stacked at the column's centre on long reruns), its System 1
endpoint dot sits at the column's centre.  Online, a call's field is painted one
step per frame up to step t + 1, and each of its marks appears, at its static
position (it never moves), on the first frame whose painted field reaches the
mark's x -- nothing is drawn over the unpainted timeline right of the field.
The column is the call's own executed action chunk, known when the call
returns, so the marks say nothing about later calls.  The fields are drawn
display-smoothed as on the static figure (``timeline_panel.display_hist_rings``
/ ``display_fut_rings``; the frame's note says so).  The playhead is a dashed ink
line at the painted frontier, x of step t + 1 (the end of the step being
executed, ``playhead_step``), so nothing -- field, marks, System 1 endpoint,
turn bar -- is ever right of it; the K lines (solid hairlines) stay at their
calls' steps, one step left of the playhead on a key moment's frame.  A small
triangle and a "now" tag sit under the step axis (no tinted band: grey on the
timeline means "System 2 answered with turns or STOP").  A long rerun's stacked
marks carry no in-panel note: the legend's timeline entry names the stride.  The
camera frame carries the pixel goal and System 1 path of the latest READY call
only (a warm-up call ran without the history heads; its output is in the status
line, marked as a warm-up call).

Data per call are rebuilt from the run exactly as ``build_records.build_bundle``
builds a key moment (trace npz, renders, GT states), for every trajectory call;
a check (on by default, ``--no-check`` skips it) compares them with the v1 bundles
(key calls) and with the timeline rows (every ready call), stores the result in
the manifest and makes the exit code 4 on a mismatch.

Policy (the ledger's): no pose, heading, VO or odometry is drawn -- the map
shows the recorded route, the reference path, start, goal + 3 m circle and the
current position as a plain dot; both heatmaps are "affordance map"; only the
framed front +-39.5 deg of a strip was given to the model (the rest is muted,
display only); the future map and the System 1 path are both decoded from Z~
and nothing is drawn from the future map to the actions.

Usage (dev machine, envs/qwen25, CPU only; ffmpeg from /opt/conda/bin)::

  PYTHONDONTWRITEBYTECODE=1 <python> -m scripts.exp19.figures.animate_v2 \\
      --records <EXP>/records --timelines <EXP>/records_v2 --out-dir <EXP>/figures_v2/anim
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import geometry as geo  # noqa: E402
from scripts.exp18.figures import common_draw as cd  # noqa: E402
from scripts.exp18.figures import style  # noqa: E402
from scripts.exp19 import build_records as br  # noqa: E402
from scripts.exp19 import build_timeline as bt  # noqa: E402
from scripts.exp19 import gt  # noqa: E402
from scripts.exp19.figures import bundle as bd  # noqa: E402
from scripts.exp19.figures import fig_behavior as fb  # noqa: E402
from scripts.exp19.figures import fig_v2 as fv  # noqa: E402
from scripts.exp19.figures import panels as pn  # noqa: E402
from scripts.exp19.figures import panels_v2 as p2  # noqa: E402
from scripts.exp19.figures import timeline_panel as tp  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

MANIFEST_SCHEMA = "exp19-animations-v2-manifest-v1"
DEFAULT_FFMPEG = "/opt/conda/bin/ffmpeg"

# --------------------------------------------------------------------------- #
# Geometry of a frame (inches; the static figure's point sizes)
# --------------------------------------------------------------------------- #
WIDTH_PX, HEIGHT_PX = 1920, 1426
W = 7.0
H = W * HEIGHT_PX / WIDTH_PX  # 5.2 in
DPI = WIDTH_PX / W
FPS = 12
GIF_WIDTH = 900
FS = p2.FS
ROW = 0.1  # legend row (in)

RIGHT = W - 0.03  # right margin: nothing is drawn past it
CAM = dict(x=0.0, y=0.55, w=2.52, h=1.89)
INSET = dict(w=0.84, h=0.63, m=0.045)  # about a third of the camera frame's width
MAP = dict(x=2.66, y=0.55, w=1.84, h=1.89)
STRIP = dict(x=4.64, w=RIGHT - 4.64, title=0.47, src=0.585, cap1=0.705, hist=0.77, cap2=1.43, fut=1.495,
             legend=2.04)
TLG = dict(x=0.30, w=RIGHT - 0.30, title=2.89, badges=2.97, badges_h=0.11, hist_h=0.69, gap=0.125, fut_h=0.37,
           turn_gap=0.02, turn_h=0.10, legend=4.60, legend2=4.70, notes=4.75)  # notes: 4 lines fit above H
TITLE_Y = 0.47
STATUS_Y = (2.505, 2.605)

HOLD_CALL = 4  # extra frames on a trajectory call's step (its decision image appears in the camera inset)
HOLD_KEY = 8  # extra frames on a key moment (on top of HOLD_CALL)
HOLD_START = 12
HOLD_END = 24
MAX_SECONDS = 60.0  # HOLD_CALL shrinks (down to 0) so that a long rerun stays below this

# --------------------------------------------------------------------------- #
# Text (the shared names come from fig_v2.LABELS)
# --------------------------------------------------------------------------- #
ALABELS: Dict[str, Dict[str, object]] = {
    "zh": {
        "cam_title": "机器人看到的画面",
        "cam_tag": "前视相机",
        "lookdown_tag": "俯视帧",
        "front_tag": "前视帧",
        "map_title": "复跑路线（到此刻）",
        "strip_title": "此刻的 360° affordance map",
        "strip_src": "第 {s} 步慢系统调用时预测，保持到下一次调用",
        "strip_warmup": "预热期：尚无 affordance map",
        "strip_native": "第 {s} 步慢系统直接给出转向，没有 affordance map",
        "strip_stop": "第 {s} 步慢系统输出 STOP，没有 affordance map",
        "strip_other": "第 {s} 步的慢系统调用没有 affordance map",
        "strip_stale": "灰显的是第 {r} 步的图",
        "step_now": "第 {t} 步",
        "of_steps": "  / 共 {n} 步",
        "s2": "慢系统（第 {s} 步调用）：{texts}",
        "s2_warm": "慢系统（第 {s} 步，预热期调用）：{texts}",
        "s2_native": "（直接给出转向）",
        "s2_none": "慢系统：尚未调用",
        "exec": "本步执行：",
        "act": {bd.STOP: "停止", bd.FORWARD: "前进 0.25 m", bd.LEFT: "左转 15°", bd.RIGHT: "右转 15°"},
        "exec_end": "本步执行：—（复跑已结束）",
        "pos": "此刻位置",
        "now": "此刻",
        "keyline": "关键时刻",
        "tl_title": "在线运行（整次复跑）",
        "legend_turns": "执行的转向（上 = 左转，下 = 右转）",
        "legend_nomap": "慢系统直接给出转向或停止（无 affordance map）",
        "legend_frame": "黑框 = 输入模型的 79° 视野；压灰部分未输入模型",
        "strip_note": ("方位：0° = 正前方，左为正，±180° = 正后方；时间线各子图向上 = 向左，历史时间线自上而下为前、左、后（居中）、"
                       "右、前。条带为在记录位置重渲染的 360° 环视，以正前方居中，历史标记画在各自的方位与仰角处，两端各重复 8°"
                       "（正后方的标记可能在两端各出现一次）。"),
        "flow_note": ["数据流：快系统路径与预测未来 affordance map 都由", fv.ZT, "解码，", fv.ZT,
                      "为桥接把历史认知头的概括向量 M 注入慢系统隐变量 Z 所得；未来图不回流到动作。"],
    },
    "en": {
        "cam_title": "What the robot sees",
        "cam_tag": "front camera",
        "lookdown_tag": "look-down",
        "front_tag": "front",
        "map_title": "Route so far",
        "strip_title": "360° affordance maps at this moment",
        "strip_src": "from the System 2 call at step {s}, held until the next call",
        "strip_warmup": "warm-up: no affordance map yet",
        "strip_native": "System 2 gave turns directly at step {s}: no affordance map",
        "strip_stop": "System 2 said STOP at step {s}: no affordance map",
        "strip_other": "the System 2 call at step {s} has no affordance map",
        "strip_stale": "greyed: the maps of step {r}",
        "step_now": "step {t}",
        "of_steps": "  / {n} steps",
        "s2": "System 2 (call at step {s}): {texts}",
        "s2_warm": "System 2 (warm-up call, step {s}): {texts}",
        "s2_native": " (turns given directly)",
        "s2_none": "System 2: not called yet",
        "exec": "Executed now:",
        "act": {bd.STOP: "STOP", bd.FORWARD: "forward 0.25 m", bd.LEFT: "turn left 15°", bd.RIGHT: "turn right 15°"},
        "exec_end": "Executed now: — (the rerun has ended)",
        "pos": "current position",
        "now": "now",
        "keyline": "key moment",
        "tl_title": "Online, over the whole rerun",
        "legend_turns": "turns (up = left, down = right)",
        "legend_nomap": "System 2 gave turns or STOP (no affordance map)",
        "legend_frame": "framed: the model's 79° view; muted: not its input",
        "strip_note": ("Bearing: 0° = ahead, left +, ±180° = behind; up = left on every timeline panel (history: "
                       "ahead, left, behind, right, ahead from top to bottom). Strips: the 360° surroundings re-rendered "
                       "at the recorded position, centred on ahead, history marks at their own elevation, 8° repeated "
                       "past ±180° (a mark right behind can show at both ends)."),
        "flow_note": ["Data flow: the System 1 path and the future map are both decoded from ", fv.ZT,
                      " = Z + bridge(Z, M), System 2's latent Z corrected from the history memory M; the future map "
                      "does not feed the actions."],
    },
}
STRIP_LEGEND = ("hist", "pred", "gt", "pair", "frame", "fut", "path", "goal")  # texts: fig_v2.LABELS legend
MAP_LEGEND = ("route", "start_goal", "pos")  # "pos": ALABELS
BOTTOM_LEGEND = ("warmup", "nomap", "turns", "keyline", "now")  # the timeline's context marks
BOTTOM_LEGEND2 = ("slots",)  # which past frame a mark belongs to: fig_v2.SLOT_KEY[plan mode] (fig_v2.LABELS legend)


# --------------------------------------------------------------------------- #
# Data: one System 2 call, one case
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One System 2 call of the rerun (every kind)."""

    index: int
    step: int
    kind: str
    state: str  # ready / warmup / native_actions / stop (build_timeline.call_state)
    texts: List[str]  # System 2's outputs in order ([first turn, final] when the look-down turn ran)
    ks: Optional[bd.KeyStep] = None  # trajectory calls: decision image, pixel goal, path (+ the maps when ready)
    key: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"


@dataclass
class Case:
    ep_key: str
    category: str
    bundle: bd.Bundle
    tl: tp.Timeline
    calls: List[Call]
    steps_dir: Path
    front_names: Dict[int, str]
    n_steps: int
    route_xz: np.ndarray  # [n_steps + 1, 2]: recorded position at each state step
    off_level: Optional[np.ndarray]
    topdown: Optional[tuple]
    map_rgb: Optional[np.ndarray]
    hist_rgba: np.ndarray  # [R, 360, 4] the timeline's history field per ready call (static encoding, smoothed)
    fut_rgba: np.ndarray  # [R, 360, 4] its future field per ready call (smoothed)
    inputs: Dict[str, dict] = field(default_factory=dict)
    checks: Dict[str, object] = field(default_factory=dict)

    def key_steps(self) -> Dict[str, int]:
        return {str(lab): int(s) for lab, s in zip(self.tl.a["key_labels"], self.tl.a["key_step"])}

    def latest(self, t: int, ready_only: bool = False) -> Optional[Call]:
        """The last call (in call order) made at or before step t."""
        out = None
        for c in self.calls:
            if c.step <= t and (c.ready or not ready_only):
                out = c
        return out

    def call_at(self, t: int) -> Optional[Call]:
        """The last call made exactly at step t, if any."""
        out = None
        for c in self.calls:
            if c.step == t:
                out = c
        return out

    def step_action(self, t: int) -> int:
        sa = np.asarray(self.tl.a.get("step_action", np.zeros(0)))
        return int(sa[t]) if 0 <= t < len(sa) else -1


def _texts(resp: dict) -> List[str]:
    first = (resp.get("native_first_output") or "").strip()
    final = (resp.get("llm_output") or "").strip()
    return [x for x in ([final] if (not first or first == final) else [first, final]) if x]


def call_keystep(ep: "br.Episode", c: dict, convention: str, ready: bool) -> Optional[bd.KeyStep]:
    """The KeyStep of one trajectory call, built as build_records.build_bundle builds a key moment.

    Every trajectory call gets the decision image (look-down after a "↓" turn,
    else the 384x384 front), the pixel goal and the System 1 path; a ready call
    also gets the four re-rendered views of its step, the history head's maps,
    P(none), the slot mask, the GT peaks of the past frames (label code, H1
    visibility) and the future head's maps.
    """
    resp = c.get("response") or {}
    if c.get("_npz") is None:
        return None
    step = int(c["current_capture_step"])
    decision = "lookdown" if int(resp.get("native_lookdown_turns") or 0) >= 1 else "front"
    arrays: Dict[str, np.ndarray] = {}
    with np.load(c["_npz"], allow_pickle=False) as z:
        jpeg = "jpeg__lookdown" if decision == "lookdown" else "jpeg__current__front"
        arrays["decision_rgb"] = br.decode_jpeg(z[jpeg])
        path = br.selected_path(c, z)
        if ready:
            hist_steps = [int(s) for s in (c.get("history_capture_steps") or [])]
            arrays["hist_pred"] = br.npz_array(z, "hist_heatmaps_gated").astype(np.float32)
            arrays["hist_none"] = br.npz_array(z, "hist_none_probability").astype(np.float32)
            arrays["hist_mask"] = (br.npz_array(z, "hist_mask").astype(bool) if "hist_mask" in z.files
                                   else np.arange(bd.NUM_SLOTS) < len(hist_steps))
            arrays["fut_pred"] = br.npz_array(z, "fut_heatmaps_gated").astype(np.float32)
    if path is None:
        return None
    wh = gt.DECISION_IMAGES[decision]["wh"]
    uv = gt.project_to_decision_image(gt.path_camera_points(path, gt.DECISION_IMAGES[decision]["pitch_deg"]), wh)
    kept = np.nonzero(np.isfinite(uv[:, 0]))[0]
    pg = resp.get("pixel_goal")
    arrays["path_cam"] = gt.path_camera_points(path).astype(np.float32)
    if ready:
        pano, depth, _ = ep.render(step)
        hist_steps = [int(s) for s in (c.get("history_capture_steps") or [])]
        gmap, gvis = gt.history_labels([ep.cam(s) for s in hist_steps], ep.cam(step), depth)
        gmap, gvis, _ = gt.pad_slots(gmap, gvis)
        cls, grow, gcol = gt.gt_view_class_and_peak(gmap, gvis)
        peak = np.stack([cls - 1, grow, gcol], axis=-1).astype(np.float32)
        peak[cls == 0] = -1
        arrays["pano_rgb"] = np.asarray(pano, dtype=np.uint8)
        arrays["hist_gt_peak"] = peak
    meta = {"label": None, "call_index": int(c["system2_call_index"]), "step": step, "decision_image": decision,
            "system2_first_output": resp.get("native_first_output"), "system2_output": resp.get("llm_output"),
            "pixel_goal_uv": list(gt.pixel_goal_uv(pg, convention)) if pg is not None else None,
            "path_uv": uv[kept].round(2).tolist()}
    return bd.KeyStep(-1, meta, arrays)


def _sha256(path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_case(ep_key: str, found: dict, records: Path, timelines: Path, renders: Path,
              topdown_root=None) -> Case:
    bundle = bd.load_bundle(records / f"{ep_key}_bundle.json")
    tl = tp.load_timeline(tp.timeline_path_for(timelines, ep_key), record_path=records / f"{ep_key}.json")
    ep = br.Episode(ep_key, found[ep_key], renders)
    convention = bundle.meta["conventions"]["pixel_goal"]["convention"]
    key_by_call = {int(ci): str(lab) for lab, ci in zip(tl.a["key_labels"], tl.a["key_call_index"])}
    calls: List[Call] = []
    for c in ep.trace["calls"]:
        resp = c.get("response") or {}
        kind = str(resp.get("kind"))
        ready = kind == "trajectory" and resp.get("ppa_applied") is True
        call = Call(int(c["system2_call_index"]), int(c["current_capture_step"]), kind, bt.call_state(kind, ready),
                    _texts(resp), key=key_by_call.get(int(c["system2_call_index"])))
        if kind == "trajectory":
            call.ks = call_keystep(ep, c, convention, ready)
            if call.ks is not None:
                call.ks.meta["label"] = call.key
        calls.append(call)
    # the timeline was built from the same trace: same calls, steps and states
    if [c.step for c in calls] != [int(s) for s in tl.a["calls_step"]] or \
            [c.state for c in calls] != [str(s) for s in tl.a["calls_state"]]:
        raise ValueError(f"{ep_key}: the trace's calls differ from the timeline's (rebuild records_v2)")
    n_steps = tl.steps
    route_steps = [int(s) for s in bundle.meta.get("route_steps") or range(len(bundle.meta["route_xz"]))]
    if route_steps != list(range(n_steps + 1)):
        raise ValueError(f"{ep_key}: route_steps are not 0..{n_steps}")
    front_names = {s: str(ep.state(s).get("front_jpg") or f"front_{s:04d}.jpg") for s in range(n_steps + 1)}
    missing = [s for s, n in front_names.items() if not (ep.steps_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{ep_key}: no front image for steps {missing[:5]}...")
    topdown = fb.resolve_level(bundle, topdown_root)
    map_rgb = cd.mute_map(topdown[1].rgb(), sat=0.18, white=0.6) if topdown is not None else None
    R = tl.R
    hist_rgba = p2.heat_rgba(tp.display_hist_rings(tl), p2.HIST_COLOR, p2.TL_HEAT_ALPHA) if R else \
        np.zeros((0, tp.N_BEARING, 4))
    fut = tp.display_fut_rings(tl) if R else np.zeros((0, tp.NUM_BINS, tp.N_BEARING))
    fut_rgba = np.stack([p2.bins_rgba(fut[r]) for r in range(R)]) if R else np.zeros((0, tp.N_BEARING, 4))
    inputs = {"bundle_json": records / f"{ep_key}_bundle.json", "bundle_npz": records / f"{ep_key}_bundle.npz",
              "record": records / f"{ep_key}.json", "timeline_json": tp.timeline_path_for(timelines, ep_key),
              "timeline_npz": tp.timeline_paths(tp.timeline_path_for(timelines, ep_key))[1],
              "renders": renders / f"{ep_key}.npz", "steps_jsonl": ep.steps_dir / "steps.jsonl"}
    case = Case(ep_key, bundle.membership(main=True)["category"], bundle, tl, calls, ep.steps_dir, front_names,
                n_steps, bundle.xz("route_xz"), fb.off_level_steps(bundle, topdown), topdown, map_rgb, hist_rgba,
                fut_rgba, inputs={k: {"path": str(v), "sha256": _sha256(v)} for k, v in inputs.items()})
    case.checks["warnings"] = list(bundle.warnings) + list(tl.warnings) + tp.check_against_bundle(tl, bundle)
    return case


# --------------------------------------------------------------------------- #
# Checks: the rebuilt calls against the v1 bundle and the timeline
# --------------------------------------------------------------------------- #
def _maxdiff(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max()) if a.size else 0.0


def check_case(case: Case) -> dict:
    """(a) every key call rebuilt here equals the v1 bundle's key moment; (b) every ready call's maps equal
    the timeline row the static figure draws (rings exactly, GT visibility exactly, GT bearing within the
    label pixel's rounding)."""
    by_index = {c.index: c for c in case.calls}
    a_rows = []
    for k in case.bundle.keys:
        c = by_index.get(int(k.call_index))
        ks = c.ks if c is not None else None
        if ks is None:
            a_rows.append({"label": k.label, "missing": True})
            continue
        row = {"label": k.label, "call_index": int(k.call_index)}
        for name in ("decision_rgb", "pano_rgb", "hist_pred", "hist_none", "hist_mask", "hist_gt_peak", "fut_pred",
                     "path_cam"):
            row[name] = _maxdiff(getattr(ks, name), getattr(k, name))
        row["path_uv"] = _maxdiff(ks.path_uv, k.path_uv)
        row["pixel_goal_uv"] = _maxdiff(ks.pixel_goal_uv or [], k.pixel_goal_uv or [])
        a_rows.append(row)
    a_max = max([max(v for n, v in r.items() if n not in ("label", "call_index")) for r in a_rows
                 if not r.get("missing")] + [0.0])
    tl = case.tl
    ring_d, fut_d, vis_mis, gt_d, n_ready = 0.0, 0.0, 0, 0.0, 0
    for r, ci in enumerate(tl.a["call_index"]):
        ks = by_index[int(ci)].ks
        n_ready += 1
        ring_d = max(ring_d, _maxdiff(bt.hist_ring(ks.hist_pred, ks.hist_none, ks.hist_mask), tl.a["hist_ring"][r]))
        fut_d = max(fut_d, _maxdiff(bt.fut_rings(ks.fut_pred), tl.a["fut_ring"][r]))
        vis = (np.asarray(ks.hist_gt_peak)[:, 0] >= 0) & np.asarray(ks.hist_mask, dtype=bool)
        vis_mis += int(np.sum(vis != np.asarray(tl.a["hist_gt_visible"][r], dtype=bool)))
        for k in np.nonzero(vis)[0]:
            v, row_, col = ks.hist_gt_peak[k]
            b, _ = geo.pixel_to_bearing_elev(int(v), float(col), float(row_))
            gt_d = max(gt_d, float(geo.circular_abs_diff(b, float(tl.a["hist_gt_bearing"][r][k]))))
    out = {"key_calls_vs_bundle": {"rows": a_rows, "max_abs_diff": a_max,
                                   "pass": bool(a_rows) and all(not r.get("missing") for r in a_rows)
                                   and a_max <= 0.01},
           "ready_calls_vs_timeline": {"n_ready": n_ready, "hist_ring_max_abs_diff": ring_d,
                                       "fut_ring_max_abs_diff": fut_d, "gt_visible_mismatch": vis_mis,
                                       "gt_bearing_vs_label_pixel_max_deg": gt_d,
                                       "pass": ring_d == 0.0 and fut_d == 0.0 and vis_mis == 0 and gt_d <= 2.0}}
    out["pass"] = out["key_calls_vs_bundle"]["pass"] and out["ready_calls_vs_timeline"]["pass"]
    return out


# --------------------------------------------------------------------------- #
# The online timeline, revealed up to step t
# --------------------------------------------------------------------------- #
R_ARRAYS = ("call_index", "step", "next_step", "hist_ring", "hist_pred_peak_bearing", "hist_pred_conf",
            "hist_gt_bearing", "hist_gt_visible", "fut_ring", "s1_path_bearing", "s1_path_dist", "hist_steps",
            "hist_mask", "exec_actions")
K_ARRAYS = ("key_labels", "key_call_index", "key_step")


class TimelineUpTo(tp.Timeline):
    """The timeline as known at step t: calls made up to t, every column's FIELD cut at t + 1 (the end of the step
    being executed, where the playhead is drawn: ``playhead_step``), so nothing right of the playhead is painted
    -- the field grows one step per frame.

    A call's marks keep their static positions and never move: its column (step .. next call) is its own
    executed action chunk, known when the call returns (``next_step - step`` equals the chunk's length for every
    ready call), so slot k sits at (k + 0.5) / 8 of that span and the System 1 endpoint dot at its centre.
    ``column_end`` (the painted end) is what grows; a mark is shown once the painted field reaches its x
    (``revealed_history_marks`` / ``revealed_future_marks``)."""

    def __init__(self, full: tp.Timeline, t: int):
        now = int(t) + 1
        a = dict(full.a)
        keep = np.asarray(full.a["step"]) <= t
        for name in R_ARRAYS:
            if name in a:
                a[name] = np.asarray(a[name])[keep]
        a["column_end"] = np.minimum(np.asarray(a["next_step"]), now)  # the painted part; next_step stays static
        ckeep = np.asarray(full.a["calls_step"]) <= t
        for name in list(a):
            if name.startswith("calls_"):
                a[name] = np.asarray(a[name])[ckeep]
        kkeep = np.asarray(full.a["key_step"]) <= t
        for name in K_ARRAYS:
            a[name] = np.asarray(a[name])[kkeep]
        if "step_action" in a:
            sa = np.array(a["step_action"])
            sa[now:] = -1
            a["step_action"] = sa
        pose = None if full.pose_ready is None else np.asarray(full.pose_ready)[ckeep]
        super().__init__(full.meta, a, full.path, pose, [], x0=full.x0, wblock=full.wblock,
                         blocks=list(full.blocks))
        self.now = now
        self._warm_end = full.warmup_end()
        self._steps_full = full.steps

    def warmup_end(self) -> int:
        return self._warm_end

    def spans(self):
        out = []
        for kind, s0, s1 in super().spans():
            s1 = min(float(s1), float(self.now))
            if s1 > s0:
                out.append((kind, s0, s1))
        return out


def playhead_step(tlt: TimelineUpTo) -> float:
    """Where the playhead goes: the painted frontier, step t + 1 (the end of the step being executed; the rerun's
    last step on its final frame).  Every column's painted end (``column_end``), no-map span, turn bar and
    revealed mark lies at or left of it."""
    return float(min(tlt.now, tlt._steps_full))


def painted_end_x(tlt: TimelineUpTo) -> np.ndarray:
    """[R] x where each call's painted field ends (its ``column_end``: the next call, or step t + 1)."""
    return np.atleast_1d(np.asarray(tlt.x_of(np.asarray(tlt.a["column_end"], dtype=np.float64)), dtype=np.float64))


def painted_max_x(tlt: TimelineUpTo, plan: tp.MarkPlan) -> float:
    """The right-most x of the data drawn on the timeline at step t: painted columns, warm-up / no-map spans, turn
    bars, revealed dots, circles and System 1 endpoints (the frame audit checks it against the playhead)."""
    xs = [float(tlt.xlim()[0])]
    if tlt.R:
        xs.append(float(np.max(painted_end_x(tlt))))
    xs += [float(tlt.x_of(s1)) for _, _, s1 in tlt.spans()]
    sa = np.asarray(tlt.a.get("step_action", np.zeros(0)))[: tlt.steps]
    turns = np.nonzero(np.isin(sa, (bd.LEFT, bd.RIGHT)))[0]
    if turns.size:
        xs.append(float(tlt.x_of(float(turns.max()) + 1.0)))
    g, p = revealed_history_marks(tlt, plan)
    xs += [x for x, _ in g + p] + list(revealed_future_marks(tlt, plan)[0])
    return max(xs)


def revealed_history_marks(tlt: TimelineUpTo, plan: tp.MarkPlan) -> Tuple[list, list]:
    """((x, y) circles, (x, y) dots) of the calls made so far, at their static positions (``plan``, the full
    timeline's), each only once the call's painted field reaches its x: nothing right of the painted field."""
    end = painted_end_x(tlt)
    gt_marks, pred_marks = [], []
    for r in (r for r in plan.rows if r < tlt.R):
        g, p = tp.history_marks(tlt, [r], plan=plan)
        gt_marks += [m for m in g if m[0] <= end[r] + 1e-9]
        pred_marks += [m for m in p if m[0] <= end[r] + 1e-9]
    return gt_marks, pred_marks


def revealed_future_marks(tlt: TimelineUpTo, plan: tp.MarkPlan) -> Tuple[list, list]:
    """(xs, ys) of the System 1 endpoint dots (column centres) whose call's painted field has reached them."""
    end = painted_end_x(tlt)
    xs, ys = [], []
    for r in (r for r in plan.rows if r < tlt.R):
        x_, y_ = tp.future_marks(tlt, [r])
        for x, y in zip(x_, y_):
            if x <= end[r] + 1e-9:
                xs.append(x)
                ys.append(y)
    return xs, ys


def draw_tl_history(ax, tlt: TimelineUpTo, case: Case, labels, plan: tp.MarkPlan, warmup_label: Optional[str],
                    warmup_range: Optional[str] = None) -> None:
    """The history panel as known at step t, on the static figure's axis (up = left, behind in the middle); the
    marks of every call made so far, laid out by the full timeline's ``plan`` (the static figure's) and revealed
    by the painted field (``revealed_history_marks``).  Nothing is written over the data (a stacked plan's stride
    is in the legend)."""
    ylim = tp.hist_ylim()
    tp._setup_axes(ax, tlt, ylim)
    tp._span_patches(ax, tlt, ylim)
    tp._hairlines(ax, tlt, tp.HIST_TICKS)
    if tlt.R:
        tp.hist_field(ax, tlt, case.hist_rgba[:tlt.R])
    gt_marks, pred_marks = revealed_history_marks(tlt, plan)
    tp.draw_marks(ax, gt_marks, pred_marks, plan.scale)
    tp._key_lines(ax, tlt, ylim)
    tp.hist_ticks(ax, labels)
    ax.set_xticks([])
    if warmup_label and tlt.now >= tlt.warmup_end():
        tp.draw_warmup_label(ax, tlt, warmup_label, warmup_range, y=180.0)


def draw_tl_future(ax, tlt: TimelineUpTo, case: Case, labels, plan: tp.MarkPlan) -> None:
    ylim = tp.bearing_ylim()
    tp._setup_axes(ax, tlt, ylim)
    tp._span_patches(ax, tlt, ylim)
    tp._hairlines(ax, tlt, (180.0, 90.0, 0.0, -90.0, -180.0))
    if tlt.R:
        tp.fut_field(ax, tlt, case.fut_rgba[:tlt.R])
    xs, ys = revealed_future_marks(tlt, plan)
    if xs:
        p2.path_marks(ax, xs, ys, ms=1.9 * max(plan.scale, 0.75), rim=True)
    tp._key_lines(ax, tlt, ylim)
    tp.break_mark(ax, tlt)
    tp.fut_ticks(ax, labels)  # left / ahead / right, as on the static figure
    ax.set_xticks([])


NOW_LW = 0.95  # the playhead: a dashed line, thicker than the K lines' solid hairline (KEY_LW 0.45)
NOW_DASH = (2.4, 1.4)
NOW_MS = 3.6  # the playhead's triangle under the step axis (pt)


def now_tag_span(fig, x: float, per: float, x_end: float, text: str) -> Tuple[float, float, str]:
    """(x_lo, x_hi, side) of the playhead's triangle + "now" tag under the step axis, in data x: the tag to the
    right of the triangle, or to its left where the axis ends too soon (``per``: points per x unit)."""
    tw = cd.text_width_pt(fig, text, p2.MIN_FS, fontweight="bold")
    half = NOW_MS / 2 + 0.3
    if (x_end - x) * per >= half + 1.0 + tw + 1.0:
        return x - half / per, x + (half + 1.0 + tw) / per, "right"
    return x - (half + 1.0 + tw) / per, x + half / per, "left"


def playhead(axes: Sequence, x: float, axis_ax=None, text: str = "", side: str = "right") -> None:
    """The current step: a dashed ink line (under the marks) through the panels -- solid ink hairlines are the key
    moments -- and, under the step axis of ``axis_ax``, a small ink triangle with its tip on the axis line and the
    tag ``text`` ("now") on its ``side``.  No tinted band (a grey band on the timeline means "System 2 answered
    with turns or STOP")."""
    for ax in axes:
        ax.plot([x, x], list(ax.get_ylim()), color=style.INK, lw=NOW_LW, zorder=7.0, solid_capstyle="butt",
                dashes=NOW_DASH, dash_capstyle="butt")
    if axis_ax is not None:
        from matplotlib import transforms as mtransforms

        tr = mtransforms.offset_copy(axis_ax.get_xaxis_transform(), fig=axis_ax.figure, x=0.0, y=-NOW_MS * 0.5,
                                     units="points")
        axis_ax.plot([x], [0.0], marker="^", ms=NOW_MS, mfc=style.INK, mec="none", transform=tr, zorder=10,
                     clip_on=False)
        if text:
            dx = NOW_MS / 2 + 1.0
            axis_ax.annotate(text, (x, 0.0), xycoords=("data", "axes fraction"),
                             xytext=(dx if side == "right" else -dx, -(tp.TICK_LEN + tp.TICK_PAD)),
                             textcoords="offset points", ha="left" if side == "right" else "right", va="top",
                             fontsize=p2.MIN_FS, fontweight="bold", color=style.INK, annotation_clip=False, zorder=10)


def clear_ticks_near(ax, x_lo: float, x_hi: float, air_pt: float = 1.5) -> int:
    """Drop the step tick labels that would touch the span [x_lo, x_hi] (data x: the playhead's triangle and tag);
    returns how many were dropped.  A kept label keeps its alignment (the axis' end label stays right-aligned)."""
    fig = ax.figure
    per = tp.per_step_pt_axes(ax)
    texts = list(ax.get_xticklabels())
    ticks = list(ax.get_xticks())
    keep = []
    for t, text in zip(ticks, texts):
        w = cd.text_width_pt(fig, text.get_text(), p2.MIN_FS) / per
        lo, hi = (t - w, t) if text.get_ha() == "right" else (t - w / 2, t + w / 2)
        if hi + air_pt / per <= x_lo or lo - air_pt / per >= x_hi:
            keep.append((t, text.get_text(), text.get_ha()))
    if len(keep) != len(ticks):
        ax.set_xticks([k[0] for k in keep])
        ax.set_xticklabels([k[1] for k in keep])
        for (t, lab, ha), text in zip(keep, ax.get_xticklabels()):
            text.set_ha(ha)
    return len(ticks) - len(keep)


# --------------------------------------------------------------------------- #
# Frame panels
# --------------------------------------------------------------------------- #
def _front(case: Case, t: int) -> np.ndarray:
    from PIL import Image

    with Image.open(case.steps_dir / case.front_names[t]) as im:
        return np.asarray(im.convert("RGB"))


def _tag(ax, text: str, x: float = 0.025, y: float = 0.955, fs: float = FS["small"]) -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=fs, color="white", zorder=10,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=(0, 0, 0, 0.55), ec="none"))


def draw_camera(page: fb.Page, case: Case, t: int, A: dict, L: dict) -> None:
    x, y, w, h = CAM["x"], CAM["y"], CAM["w"], CAM["h"]
    ax = page.ax(x, y, w, h)
    img = _front(case, t)
    pn.draw_image(ax, img, frame_color=style.INK, frame_lw=0.5)
    _tag(ax, A["cam_tag"])
    latest = case.latest(t)  # the inset (pixel goal + System 1 path) only for a ready call, as the spec asks: a
    # warm-up call ran without the history heads and the bridge, so the data-flow note does not describe it
    if latest is not None and latest.ready and latest.ks is not None and t < case.n_steps:
        iw_, ih_, m = INSET["w"], INSET["h"], INSET["m"]
        iax = page.ax(x + w - m - iw_, y + m, iw_, ih_)
        tag = A["lookdown_tag"] if latest.ks.decision_image == "lookdown" else A["front_tag"]
        p2.draw_decision_image(iax, latest.ks, tag=tag)
        for s in iax.spines.values():
            s.set_edgecolor("white")
            s.set_linewidth(1.1)
    if t >= case.n_steps:
        lines = fb.wrap(page.fig, fv.outcome_line(case.bundle, L), FS["body"], (w - 0.14) * 72.0, fontweight="bold")
        ax.text(0.5, 0.05, "\n".join(lines), transform=ax.transAxes, ha="center", va="bottom", multialignment="center",
                fontsize=FS["body"], fontweight="bold", color=style.INK, zorder=12, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3", fc=(1, 1, 1, 0.88), ec="none"))


def s2_status(case: Case, t: int, A: dict, L: dict, width_pt: float, fig) -> str:
    c = case.latest(t)
    if c is None:
        return A["s2_none"]
    texts = L["then"].join(f"“{x}”" for x in c.texts)
    if c.kind == "native_actions":
        texts += A["s2_native"]
    line = (A["s2_warm"] if c.state == "warmup" else A["s2"]).format(s=c.step, texts=texts)
    if c.texts and fv.PIXEL_GOAL_TEXT.match(str(c.texts[-1])):
        longer = line + L["goal_suffix"]
        if cd.text_width_pt(fig, longer, FS["body"]) <= width_pt:
            return longer
    return line


def draw_status(page: fb.Page, case: Case, t: int, A: dict, L: dict) -> None:
    fig = page.fig
    w = CAM["w"] + 0.1
    page.text(CAM["x"], STATUS_Y[0], s2_status(case, t, A, L, w * 72.0, fig), ha="left", va="center",
              fontsize=FS["body"], color=style.INK)
    act = case.step_action(t) if t < case.n_steps else -1
    if act < 0:
        page.text(CAM["x"], STATUS_Y[1], A["exec_end"], ha="left", va="center", fontsize=FS["body"],
                  color=style.INK_2)
        return
    ax = page.pt_axes(CAM["x"], STATUS_Y[1] - 0.06, w, 0.12)
    cy = 0.06 * 72.0
    ax.text(0.0, cy, A["exec"], ha="left", va="center", fontsize=FS["body"], color=style.INK_2)
    xx = cd.text_width_pt(fig, A["exec"], FS["body"]) + 3.0
    xx += p2.action_chip(ax, xx, cy, act, size=7.6) + 3.5
    if act != bd.STOP:
        ax.text(xx, cy, A["act"][act], ha="left", va="center", fontsize=FS["body"], color=style.INK)


def _map_limits(case: Case, ax):
    b = case.bundle
    return pn.map_limits([case.route_xz, b.xz("reference_path_xz")], b.goal_xz, float(b.goal_radius_m),
                         cd.axes_aspect_hw(ax), pad=0.9)


def draw_map(page: fb.Page, case: Case, t: int, A: dict, L: dict) -> None:
    """Route so far (ink), reference path (grey dashed), start, goal + 3 m circle, K badges once reached, and the
    current position as a plain ink dot (no heading)."""
    b = case.bundle
    ax = page.ax(MAP["x"], MAP["y"], MAP["w"], MAP["h"])
    limits = _map_limits(case, ax)
    x0, x1, z0, z1 = limits
    if case.map_rgb is not None:
        ax.imshow(case.map_rgb, extent=case.topdown[1].extent, interpolation="bilinear", zorder=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(z1, z0)
        ax.set_autoscale_on(False)
        ax.set_facecolor(cd.MAP_PLATE)
    else:
        pn.draw_plate(ax, limits)
        ax.text(0.5, 0.03, L["no_map"], transform=ax.transAxes, ha="center", va="bottom", fontsize=p2.MIN_FS,
                color=style.INK_2, fontstyle="italic")
    cd.clean_axes(ax, spines=True, lw=p2.HAIR)
    per_pt = cd.pts_to_data(ax, 1.0)[0]
    route = case.route_xz
    ref = b.xz("reference_path_xz")
    gx, gz = float(b.goal_xz[0]), float(b.goal_xz[1])
    radius = float(b.goal_radius_m)
    ax.add_patch(Circle((gx, gz), radius, fc=cd.mix(style.INK, "white", 0.94), ec=style.INK_2, lw=0.5,
                        ls=(0, (2.2, 1.6)), alpha=0.9, zorder=1))
    ax.annotate(L["radius"], (gx, gz - radius), xytext=(0, 1.0), textcoords="offset points", ha="center",
                va="bottom", fontsize=p2.MIN_FS, color=style.INK_2, path_effects=cd.HALO_THIN, zorder=2)
    ax.plot(ref[:, 0], ref[:, 1], color=style.MUTED, lw=0.8, ls=(0, (3.0, 1.8)), zorder=2.5)
    upto = route[: t + 1]
    pn.draw_route_line(ax, upto, None if case.off_level is None else case.off_level[: t + 1], None)
    keys = [(k.label, np.asarray(k.position_xz, dtype=np.float64), int(k.step)) for k in b.keys]
    key_xz = np.asarray([k[1] for k in keys]).reshape(-1, 2)
    d = route[min(3, len(route) - 1)] - route[0]
    d = -d / (np.linalg.norm(d) + 1e-9)
    avoid = np.concatenate([route, ref, key_xz]) if len(key_xz) else np.concatenate([route, ref])
    (ox, oy), ha, va = p2.place_label(route[0], d, cd.text_width_pt(ax.figure, L["start"], p2.MIN_FS), p2.MIN_FS,
                                      limits, per_pt, p2._densify(avoid, 2.0 * per_pt))
    ax.annotate(L["start"], route[0], xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=p2.MIN_FS,
                color=style.INK, path_effects=cd.HALO_THIN, zorder=5)
    ax.plot([gx], [gz], marker="*", ms=7.0, mfc=style.INK, mec="white", mew=0.5, zorder=6)
    points = np.concatenate([route, ref, [[gx, gz]], key_xz]) if len(key_xz) else \
        np.concatenate([route, ref, [[gx, gz]]])
    boxes = [route[0] + d * 12.0 * per_pt, np.array([gx, gz - radius - 4.0 * per_pt])]
    for lab, p, s in keys:  # badge spots from every key moment, so a badge never moves once shown
        q = pn._badge_spot(p, points, np.array(boxes), limits, per_pt)
        boxes.append(q)
        if s <= t:
            ax.plot(*p, marker="o", ms=2.6, mfc=style.INK, mec="white", mew=0.4, zorder=7)
            ax.plot([p[0], q[0]], [p[1], q[1]], color=style.INK, lw=0.4, zorder=6.5)
            cd.key_badge(ax, q[0], q[1], lab, fs=p2.MIN_FS)
    bar = cd.nice_length(0.3 * (x1 - x0))
    cd.scale_bar(ax, *pn._scale_bar_spot(limits, per_pt, bar, np.concatenate([points, boxes])), bar, f"{bar:g} m",
                 fs=p2.MIN_FS)
    pos = route[min(t, len(route) - 1)]
    position_mark(ax, pos[0], pos[1])


POS_MS, POS_HALO = 4.5, 0.6  # current position: a plain filled ink dot (pt) with a thin white halo (pt)


def position_mark(ax, x: float, y: float, zorder: float = 9) -> None:
    """The current position: a plain filled ink dot ``POS_MS`` pt across (larger than the 2.6 pt key-moment
    anchors, unlike the pixel goal's ring) with a ``POS_HALO`` pt white halo -- no ring, no heading."""
    ax.plot([x], [y], marker="o", ms=POS_MS + 2 * POS_HALO, mfc="white", mec="none", zorder=zorder, clip_on=False)
    ax.plot([x], [y], marker="o", ms=POS_MS, mfc=style.INK, mec="none", zorder=zorder + 0.1, clip_on=False)


def _grey_out(ax) -> None:
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="white", ec="none", alpha=0.62, zorder=15))


def _note(ax, text: str, fs: float = FS["body"]) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=fs, color=style.INK,
            zorder=20, bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3", fc="white", ec=style.AXIS, lw=0.4))


def _empty_strip(ax, elev: float) -> None:
    ax.set_facecolor(style.SURFACE)
    p2.setup_strip(ax, elev)
    p2.strip_seams(ax, elev)
    p2.camera_frame(ax, elev)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(style.AXIS)
        s.set_linewidth(p2.HAIR)


def strip_state(case: Case, t: int) -> Tuple[Optional[Call], Optional[Call]]:
    """(call whose maps are shown, latest call when it has no maps); both None before the first call."""
    latest = case.latest(t)
    if latest is None:
        return None, None
    if latest.ready:
        return latest, None
    return case.latest(t, ready_only=True), latest


def draw_strips(page: fb.Page, case: Case, t: int, A: dict, L: dict) -> None:
    fig = page.fig
    x, w = STRIP["x"], STRIP["w"]
    shown, nomap = strip_state(case, t)
    # source line (which call the maps come from)
    if shown is not None and nomap is None:
        src = A["strip_src"].format(s=shown.step)
    elif shown is None:
        src = ""
    else:
        src = A["strip_stale"].format(r=shown.step)
    if src:
        sax = page.pt_axes(x, STRIP["src"] - 0.05, w, 0.1)
        xx = 0.0
        if shown is not None and shown.key and nomap is None:
            bw = cd.text_width_pt(fig, shown.key, p2.MIN_FS, fontweight="bold")
            cd.key_badge(sax, bw / 2 + 2.0, 0.05 * 72.0, shown.key, fs=p2.MIN_FS)
            xx = bw + 7.0
        sax.text(xx, 0.05 * 72.0, src, ha="left", va="center", fontsize=FS["small"], color=style.INK_2)
    page.text(x, STRIP["cap1"], L["cap_hist"], ha="left", va="center", fontsize=FS["small"], color=style.INK_2)
    page.text(x, STRIP["cap2"], L["cap_fut"], ha="left", va="center", fontsize=FS["small"], color=style.INK_2)
    hh, fh = p2.strip_height(w, p2.HIST_ELEV), p2.strip_height(w, p2.FUT_ELEV)
    hax = page.ax(x, STRIP["hist"], w, hh)
    fax = page.ax(x, STRIP["fut"], w, fh)
    px = int(round(w * DPI / 8.0)) * 8
    if shown is None:
        _empty_strip(hax, p2.HIST_ELEV)
        _empty_strip(fax, p2.FUT_ELEV)
    else:
        p2.draw_history_strip(hax, shown.ks, px, px)
        p2.draw_future_strip(fax, shown.ks, px)
    p2.strip_ticks(fax, L["axis"])
    if nomap is not None or shown is None:
        if shown is not None:
            _grey_out(hax)
            _grey_out(fax)
        if (shown is None and t < case.tl.warmup_end()) or nomap is None:
            text = A["strip_warmup"]
        elif nomap.state == "native_actions":
            text = A["strip_native"].format(s=nomap.step)
        elif nomap.state == "stop":
            text = A["strip_stop"].format(s=nomap.step)
        else:
            text = A["strip_other"].format(s=nomap.step)
        _note(hax, text)


def legend_lines(fig, text: str, fs: float, width_pt: float) -> List[str]:
    """A legend text in lines of ``width_pt``: one line when it fits, else broken after its semicolon when both
    parts fit, else wrapped -- narrower when that leaves a single word on the last line."""
    if cd.text_width_pt(fig, text, fs) <= width_pt:
        return [text]
    parts = fv.split_at_semicolon(text)
    if parts and all(cd.text_width_pt(fig, p, fs) <= width_pt for p in parts):
        return parts
    lines = fb.wrap(fig, text, fs, width_pt)
    w = width_pt
    while len(lines) > 1 and " " not in lines[-1].strip() and w > width_pt * 0.6:
        w -= 4.0
        narrower = fb.wrap(fig, text, fs, w)
        if len(narrower) > len(lines):
            break
        lines = narrower
    return lines


def legend_block(page: fb.Page, x: float, y: float, w: float, items: Sequence[Tuple[str, str]],
                 fs: float = FS["small"]) -> float:
    """Legend entries one per row (a long text wraps under itself, ``legend_lines``); returns the height (in)."""
    fig = page.fig
    rows = []
    for key, text in items:
        lines = legend_lines(fig, text, fs, w * 72.0 - 17.0)
        rows.append((key, lines))
    n = sum(len(lines) for _, lines in rows)
    h = ROW * n
    ax = page.pt_axes(x, y, w, h)
    i = 0
    for key, lines in rows:
        yy = h * 72.0 - (i + 0.5) * ROW * 72.0
        _glyph(ax, key, 0.0, yy)
        for j, line in enumerate(lines):
            ax.text(17.0, yy - j * ROW * 72.0, line, ha="left", va="center", fontsize=fs, color=style.INK)
        i += len(lines)
    return h


def _glyph(ax, key: str, x: float, y: float, w: float = 14.0) -> None:
    if key == "pos":
        position_mark(ax, x + w / 2, y)
    elif key == "now":
        ax.plot([x + w / 2, x + w / 2], [y - 1.6, y + 3.8], color=style.INK, lw=NOW_LW, dashes=NOW_DASH,
                dash_capstyle="butt")
        ax.plot([x + w / 2], [y - 1.6 - NOW_MS * 0.5], marker="^", ms=NOW_MS, mfc=style.INK, mec="none")
    else:
        p2.legend_glyph(ax, key, x, y, w)


def legend_row(page: fb.Page, x: float, y: float, w: float, items: Sequence[Tuple[str, str]],
               gap: float = 14.0, fs: float = FS["small"]) -> None:
    """One legend row spread over ``w`` (the gaps shrink to fit, down to 5 pt)."""
    fig = page.fig
    ax = page.pt_axes(x, y - ROW / 2, w, ROW)
    used = sum(17.0 + cd.text_width_pt(fig, text, fs) for _, text in items)
    gap = max(5.0, min(gap, (w * 72.0 - used) / max(len(items) - 1, 1)))
    xx = 0.0
    for key, text in items:
        _glyph(ax, key, xx, ROW * 36.0)
        ax.text(xx + 17.0, ROW * 36.0, text, ha="left", va="center", fontsize=fs, color=style.INK)
        xx += 17.0 + cd.text_width_pt(fig, text, fs) + gap


def draw_timeline(page: fb.Page, case: Case, t: int, A: dict, L: dict) -> dict:
    x, w = TLG["x"], TLG["w"]
    y = TLG["badges"]
    bax = page.pt_axes(x, y, w, TLG["badges_h"], zorder=6)
    y += TLG["badges_h"]
    hax = page.ax(x, y, w, TLG["hist_h"])
    y += TLG["hist_h"] + TLG["gap"]
    fax = page.ax(x, y, w, TLG["fut_h"])
    y += TLG["fut_h"] + TLG["turn_gap"]
    tax = page.ax(x, y, w, TLG["turn_h"])
    full = case.tl
    tp.compress_axis(full, w * 72.0, tp.WARM_BLOCK_PT["anim"])
    plan = tp.mark_plan(full, hax)  # the static figure's layout of the whole rerun's marks: nothing moves online
    tlt = TimelineUpTo(full, t)
    draw_tl_history(hax, tlt, case, L["y_hist"], plan, L["warmup"], L["warmup_range"])
    draw_tl_future(fax, tlt, case, L["y_fut"], plan)
    tp.draw_turn_track(tax, tlt, L["turns"])
    tp.step_axis(tax, tlt, L["x_steps"])
    tp.key_badges(bax, hax, tlt)
    xn = float(full.x_of(playhead_step(tlt)))  # the painted frontier (t + 1): nothing drawn right of it
    lo, hi, side = now_tag_span(page.fig, xn, tp.per_step_pt_axes(tax), full.xlim()[1], A["now"])
    playhead((hax, fax, tax), xn, tax, A["now"], side)
    clear_ticks_near(tax, lo, hi)
    title_at(page, 0.0, TLG["title"], A["tl_title"])
    page.text(x, TLG["badges"] + TLG["badges_h"] + TLG["hist_h"] + TLG["gap"] * 0.5, L["panel_b_fut"], ha="left",
              va="center", fontsize=FS["small"], color=style.INK_2)
    return {"marker_scale": plan.scale, "marker_stride": plan.stride, "mark_mode": plan.mode, "playhead_x": xn,
            "painted_max_x": painted_max_x(tlt, plan)}


def draw_notes(page: fb.Page, y: float, A: dict, lang: str) -> int:
    """The bearing / strip note, the display-smoothing note (``fig_v2.smooth_sentence``, as in the captions) and
    the data-flow note as one paragraph across the frame from ``y`` (in); returns the line count.  Four lines fit
    above the frame's bottom edge; a longer paragraph runs past it and the text audit reports it."""
    fig = page.fig
    w_pt = RIGHT * 72.0
    sep = "" if lang == "zh" else " "
    head = A["strip_note"] + sep + fv.smooth_sentence(lang) + sep + A["flow_note"][0]
    lines = fv.rich_wrap(fig, [head] + list(A["flow_note"][1:]), FS["small"], w_pt)
    h = fv.LINE * len(lines)
    ax = page.pt_axes(0.0, y, RIGHT, h)
    for i, runs in enumerate(lines):
        fv.rich_line(ax, 0.0, h * 72.0 - (i + 0.5) * fv.LINE * 72.0, runs, FS["small"])
    return len(lines)


def title_at(page: fb.Page, x: float, y_center: float, text: str) -> None:
    """A panel title (fig_v2.panel_title: 6.8 pt bold) centred on ``y_center``."""
    fv.panel_title(page, x, y_center - (0.12 - 4.3 / 72.0), None, text)


def draw_header(page: fb.Page, case: Case, t: int, lang: str, A: dict, L: dict) -> None:
    fig = page.fig
    b = case.bundle
    title = L["title"].format(cat=case.category, name=fb.CATEGORY_NAMES[lang][case.category])
    page.text(0.0, 0.10, title, ha="left", va="center", fontsize=FS["title"], fontweight="bold", color=style.INK)
    tw = cd.text_width_pt(fig, title, FS["title"], fontweight="bold") / 72.0
    page.text(tw + 0.12, 0.10, L["ids"].format(scene=b.scene_id, ep=b.episode_id), ha="left", va="center",
              fontsize=FS["small"], color=style.MUTED)
    tail = A["of_steps"].format(n=case.n_steps)
    now = A["step_now"].format(t=t)
    tail_w = cd.text_width_pt(fig, tail, FS["body"]) / 72.0
    now_w = cd.text_width_pt(fig, now, FS["title"], fontweight="bold") / 72.0
    page.text(RIGHT, 0.10, tail, ha="right", va="center", fontsize=FS["body"], color=style.INK_2)
    page.text(RIGHT - tail_w, 0.10, now, ha="right", va="center", fontsize=FS["title"], fontweight="bold",
              color=style.INK)
    here = case.call_at(t)
    if here is not None and here.key and here.ready:
        kax = page.pt_axes(RIGHT - tail_w - now_w - 0.32, 0.03, 0.3, 0.14)
        cd.key_badge(kax, 0.3 * 72.0 - 10.0, 0.07 * 72.0, here.key, fs=FS["small"])
    lines = fv.wrap_ink(fig, L["instruction"].format(text=" ".join(b.instruction.split())), FS["body"],
                        RIGHT * 72.0, margin_pt=0.0, fontstyle="italic")  # rendered ink within RIGHT
    if len(lines) > 2:
        last = lines[1]
        while last and fv.ink_extent_pt(last + "…”", FS["body"], fontstyle="italic")[1] > RIGHT * 72.0:
            last = last[:-1]
        lines = [lines[0], last.rstrip() + "…”"]
    for j, line in enumerate(lines):
        page.text(0.0, 0.215 + fv.LINE * j, line, ha="left", va="center", fontsize=FS["body"], color=style.INK,
                  fontstyle="italic")
    rule(page, 0.40)


def rule(page: fb.Page, y: float) -> None:
    """A hairline across the frame at ``y`` (in)."""
    page.fig.add_artist(Line2D([0.0, RIGHT / W], [1 - y / H] * 2, transform=page.fig.transFigure, color=style.AXIS,
                               lw=0.5))


def draw_frame(case: Case, t: int, lang: str):
    """The frame of step t (0 .. episode_steps) as a matplotlib figure (WIDTH_PX x HEIGHT_PX at ``DPI``)."""
    import matplotlib.pyplot as plt

    A, L = ALABELS[lang], fv.LABELS[lang]
    fig = plt.figure(figsize=(W, H))
    page = fb.Page(fig, W, H)
    draw_header(page, case, t, lang, A, L)
    for xx, key in ((CAM["x"], "cam_title"), (MAP["x"], "map_title"), (STRIP["x"], "strip_title")):
        title_at(page, xx, TITLE_Y, A[key])
    draw_camera(page, case, t, A, L)
    draw_status(page, case, t, A, L)
    draw_map(page, case, t, A, L)
    draw_strips(page, case, t, A, L)
    leg = L["legend"]
    strip_texts = {**leg, "frame": A["legend_frame"]}
    legend_block(page, STRIP["x"], STRIP["legend"], STRIP["w"], [(k, strip_texts[k]) for k in STRIP_LEGEND])
    texts = {**leg, "pos": A["pos"], "nomap": A["legend_nomap"], "turns": A["legend_turns"], "keyline": A["keyline"],
             "now": A["now"]}
    map_items = [(k, texts[k]) for k in MAP_LEGEND]
    my = STATUS_Y[0] - ROW / 2
    ax = page.pt_axes(MAP["x"], my, MAP["w"], ROW * len(map_items))
    for i, (key, text) in enumerate(map_items):
        yy = ROW * 72.0 * (len(map_items) - i - 0.5)
        _glyph(ax, key, 0.0, yy)
        ax.text(17.0, yy, text, ha="left", va="center", fontsize=FS["small"], color=style.INK)
    info = draw_timeline(page, case, t, A, L)
    legend_row(page, 0.0, TLG["legend"], RIGHT, [(k, texts[k]) for k in BOTTOM_LEGEND])
    slot = fv.SLOT_KEY[info["mark_mode"]]  # what the timeline marks (on a long rerun: stacked, one call in k)
    legend_row(page, 0.0, TLG["legend2"], RIGHT, [(slot, fv.slot_legend_text(L, info["mark_mode"],
                                                                             info["marker_stride"]))])
    draw_notes(page, TLG["notes"], A, lang)
    return fig, info


# --------------------------------------------------------------------------- #
# Frame plan, rendering, encoding
# --------------------------------------------------------------------------- #
def frame_plan(case: Case, fps: int = FPS, hold_call: int = HOLD_CALL, hold_key: int = HOLD_KEY,
               hold_start: int = HOLD_START, hold_end: int = HOLD_END, max_seconds: float = MAX_SECONDS):
    """(sequence of steps, one per output frame; the hold_call actually used).

    Every step 0 .. episode_steps appears once; a trajectory call's step is repeated
    ``hold_call`` more times (shrunk so the video stays under ``max_seconds``), a key
    moment ``hold_key`` more, the first and the last step ``hold_start`` / ``hold_end`` more.
    """
    n = case.n_steps
    call_steps = sorted({c.step for c in case.calls if c.ks is not None and c.step <= n})
    key_steps = set(case.key_steps().values())
    base = (n + 1) + hold_start + hold_end + hold_key * len(key_steps)
    budget = int(max_seconds * fps) - base
    h = hold_call if not call_steps else max(0, min(hold_call, budget // len(call_steps)))
    seq = []
    for t in range(n + 1):
        rep = 1 + (h if t in call_steps else 0) + (hold_key if t in key_steps else 0)
        rep += hold_start if t == 0 else 0
        rep += hold_end if t == n else 0
        seq += [t] * rep
    return seq, h


_CASE: Optional[Case] = None
_LANG: Optional[str] = None
_OUT: Optional[Path] = None
_AUDIT = False


def audit(fig) -> dict:
    """Text checks of a frame (fig_v2's): overlapping texts, texts off the frame, the smallest font (pt)."""
    return {"overlaps": fv.text_overlaps(fig), "outside": fv.texts_outside(fig), "min_font_pt": fv.min_font_size(fig)}


def _render_one(t: int) -> Tuple[int, float, dict]:
    import matplotlib.pyplot as plt

    t0 = time.time()
    fig, info = draw_frame(_CASE, t, _LANG)
    if _AUDIT:
        info = dict(info, **audit(fig))
    fig.savefig(_OUT / f"{t:05d}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    return t, time.time() - t0, info


def render_frames(case: Case, lang: str, out: Path, steps: Sequence[int], workers: int,
                  check_text: bool = False) -> dict:
    """PNG of every step in ``steps`` into ``out`` (parallel, fork start method: the case is inherited)."""
    global _CASE, _LANG, _OUT, _AUDIT
    fv.setup(lang)
    _CASE, _LANG, _OUT, _AUDIT = case, lang, out, check_text
    out.mkdir(parents=True, exist_ok=True)
    times, infos = [], {}
    if workers <= 1:
        for t in steps:
            _, dt, info = _render_one(t)
            times.append(dt)
            infos[t] = info
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for t, dt, info in pool.imap_unordered(_render_one, list(steps), chunksize=2):
                times.append(dt)
                infos[t] = info
    out = {"n": len(times), "sec_per_frame_median": float(np.median(times)) if times else None,
           "marker_stride": max((i["marker_stride"] for i in infos.values()), default=1),
           "mark_mode": next((i["mark_mode"] for i in infos.values()), None)}
    out["frames_with_data_past_playhead"] = [t for t, i in sorted(infos.items())
                                             if i["painted_max_x"] > i["playhead_x"] + 1e-6]
    if check_text:
        out["text_overlaps"] = {t: i["overlaps"] for t, i in sorted(infos.items()) if i["overlaps"]}
        out["texts_outside"] = {t: i["outside"] for t, i in sorted(infos.items()) if i["outside"]}
        out["min_font_pt"] = min(i["min_font_pt"] for i in infos.values()) if infos else None
    return out


def _run(cmd: List[str]) -> None:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({res.returncode}):\n{res.stdout[-3000:]}")


def _sequence_dir(frames: Path, seq: Sequence[int], name: str) -> Path:
    d = frames / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, t in enumerate(seq):
        os.symlink(frames / f"{t:05d}.png", d / f"{i:05d}.png")
    return d


LONG_RUN_STEPS = 200  # GIFs of longer reruns: every 4th frame, 64 colours (the F2 GIFs were over 10 MB)


def gif_settings(n_steps: int) -> Tuple[int, int]:
    """(every n-th video frame, palette size) of the GIF preview."""
    return (4, 64) if n_steps > LONG_RUN_STEPS else (2, 256)


def encode(frames: Path, seq: Sequence[int], mp4: Path, gif: Path, ffmpeg: str, fps: int = FPS,
           crf: int = 18, gif_width: int = GIF_WIDTH, gif_every: int = 2, gif_colors: int = 256) -> None:
    """MP4: H.264 yuv420p at ``fps``, one output frame per entry of ``seq``.  GIF: every ``gif_every``-th of
    those frames at fps / gif_every (same duration), ``gif_width`` px wide, one ``gif_colors`` palette per video."""
    mp4.parent.mkdir(parents=True, exist_ok=True)
    d = _sequence_dir(frames, seq, "seq_mp4")
    tmp = mp4.with_suffix(".tmp.mp4")
    _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", str(d / "%05d.png"),
          "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
          "-movflags", "+faststart", str(tmp)])
    os.replace(tmp, mp4)
    g = _sequence_dir(frames, seq[::gif_every], "seq_gif")
    tmp = gif.with_suffix(".tmp.gif")
    vf = (f"scale={gif_width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={gif_colors}:stats_mode=full[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{fps / gif_every:g}", "-i",
          str(g / "%05d.png"), "-vf", vf, "-loop", "0", str(tmp)])
    os.replace(tmp, gif)


def probe(path: Path, ffmpeg: str) -> dict:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    res = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,codec_name,pix_fmt,r_frame_rate,nb_frames:format=duration",
                          "-of", "json", str(path)], stdout=subprocess.PIPE, text=True)
    try:
        info = json.loads(res.stdout)
        s = info["streams"][0]
        return {"width": s.get("width"), "height": s.get("height"), "codec": s.get("codec_name"),
                "pix_fmt": s.get("pix_fmt"), "frame_rate": s.get("r_frame_rate"), "frames": s.get("nb_frames"),
                "duration_s": float(info["format"]["duration"]), "bytes": path.stat().st_size}
    except (KeyError, IndexError, ValueError):
        return {"bytes": path.stat().st_size if path.is_file() else None}


def preview_steps(case: Case) -> Dict[str, int]:
    """Four representative steps: early warm-up, a turn, the middle, the end."""
    tl = case.tl
    warm = case.tl.warmup_end()
    out = {"warmup": max(0, min(warm - 1, warm // 2))}
    k2 = next((k for k in case.bundle.keys if k.label == "K2"), None)
    if k2 is not None and k2.branch == "K2_turn":
        out["turn"] = int(k2.step)
    else:  # the ready call right after the largest executed turn between two ready calls
        steps = [int(s) for s in tl.a["step"]]
        sa = np.asarray(tl.a.get("step_action", np.zeros(0)))
        best, best_turn = (steps[0] if steps else warm), -1.0
        for s0, s1 in zip(steps, steps[1:]):
            turn = abs(bd.net_turn_deg([int(a) for a in sa[s0:s1] if a >= 0]))
            if turn > best_turn:
                best, best_turn = s1, turn
        out["turn"] = best
    mid = case.n_steps // 2
    ready = [int(s) for s in tl.a["step"] if abs(int(s) - out["turn"]) >= 6] or [int(s) for s in tl.a["step"]]
    out["mid"] = min(ready, key=lambda s: abs(s - mid)) + 1 if ready else mid  # one step after a call
    out["end"] = case.n_steps
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main_cases(metrics: Optional[Path], records: Path) -> List[Tuple[str, str]]:
    """(category, ep_key) of the main cases in category order: metrics.json main_cases, else the bundles' is_main."""
    if metrics is not None and metrics.is_file():
        mc = json.loads(metrics.read_text(encoding="utf-8")).get("main_cases") or {}
        if mc:
            return [(c, mc[c]) for c in bd.CATEGORIES if c in mc]
    out = []
    for p in bd.find_bundles(records):
        meta = json.loads(p.read_text(encoding="utf-8"))
        for m in [{"category": meta["category"], "is_main": meta["is_main"]}] + list(meta.get("memberships") or []):
            if m.get("is_main"):
                out.append((m["category"], meta["ep_key"]))
    return sorted(set(out), key=lambda x: bd.CATEGORIES.index(x[0]))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True, type=Path, help="<EXP>/records (v1 bundles + episode records)")
    ap.add_argument("--timelines", required=True, type=Path, help="<EXP>/records_v2 (build_timeline.py)")
    ap.add_argument("--out-dir", required=True, type=Path, help="<EXP>/figures_v2/anim (never the v1 figures dir)")
    ap.add_argument("--exp-root", type=Path, default=None, help="default: the parent of --records")
    ap.add_argument("--run", default=None, help="runs/<run> (default: the timelines' run, else 'main')")
    ap.add_argument("--renders", type=Path, default=None, help="default <exp-root>/renders")
    ap.add_argument("--metrics", type=Path, default=None, help="default <exp-root>/metrics/metrics.json (main_cases)")
    ap.add_argument("--topdown-root", default=None)
    ap.add_argument("--lang", nargs="+", default=["zh", "en"], choices=sorted(ALABELS))
    ap.add_argument("--only", nargs="*", default=None, help="ep_keys (default: the main cases)")
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="PNG frames, on local disk (default: a new dir under the system temp dir, removed after)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--preview-dir", type=Path, default=None, help="JPEG copies of 4 representative frames per case")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="draw only these steps into --preview-dir (layout work; no video)")
    ap.add_argument("--preview-only", action="store_true",
                    help="draw only the 4 representative steps of each case into --preview-dir (no video)")
    ap.add_argument("--check", dest="check", action="store_true", default=True,
                    help="(default) compare the rebuilt calls with the bundles and timelines; exit 4 on a mismatch")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--ffmpeg", default=os.environ.get("EXP19_FFMPEG", DEFAULT_FFMPEG))
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--hold-call", type=int, default=HOLD_CALL)
    ap.add_argument("--hold-key", type=int, default=HOLD_KEY)
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = ap.parse_args(argv)
    args.exp_root = args.exp_root or args.records.resolve().parent
    args.renders = args.renders or args.exp_root / "renders"
    args.metrics = args.metrics or args.exp_root / "metrics" / "metrics.json"
    if args.out_dir.resolve().name == "figures" or "figures" in [p.name for p in args.out_dir.resolve().parents][:1]:
        ap.error("refusing to write into the v1 figures dir")
    for bad in ("records", "metrics"):
        if args.out_dir.resolve() == (args.exp_root / bad).resolve():
            ap.error(f"refusing to write into {bad}/")
    return args


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _jpeg(src: Path, dst: Path, width: int = 1800) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst, quality=90)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    cases = main_cases(args.metrics, args.records)
    if args.only:
        cases = [c for c in cases if c[1] in set(args.only)] or [(json.loads(
            (args.records / f"{k}_bundle.json").read_text("utf-8"))["category"], k) for k in args.only]
    if not cases:
        print("no main cases found", file=sys.stderr)
        return 2
    run = args.run
    if run is None:
        meta = json.loads(tp.timeline_path_for(args.timelines, cases[0][1]).read_text("utf-8"))
        run = meta.get("run") or "main"
    found = br.discover_run(args.exp_root / "runs" / run)
    do_video = args.steps is None and not args.preview_only
    if do_video and not Path(args.ffmpeg).is_file():
        print(f"ffmpeg not found at {args.ffmpeg} (set --ffmpeg or EXP19_FFMPEG)", file=sys.stderr)
        return 2
    frames_root = args.frames_dir or Path(tempfile.gettempdir()) / f"exp19_anim_frames_{os.getpid()}"
    manifest_path = args.out_dir / "manifest.json"
    manifest = {"schema": MANIFEST_SCHEMA, "run": run, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "code_sha256": {n: _sha256(Path(__file__).with_name(n)) for n in
                                ("animate_v2.py", "fig_v2.py", "panels_v2.py", "timeline_panel.py", "panels.py",
                                 "bundle.py")},
                "frame": {"width_px": WIDTH_PX, "height_px": HEIGHT_PX, "design_in": [W, H], "dpi": DPI},
                "videos": []}
    if do_video and manifest_path.is_file() and args.only:
        old = json.loads(manifest_path.read_text("utf-8"))
        keep = {k for k in args.only}
        manifest["videos"] = [v for v in old.get("videos", []) if v.get("ep_key") not in keep]
    status = 0
    for cat, ep_key in cases:
        log(f"{cat} {ep_key}: loading")
        case = load_case(ep_key, found, args.records, args.timelines, args.renders, args.topdown_root)
        if args.check:
            case.checks.update(check_case(case))
            log(f"  check: {'PASS' if case.checks['pass'] else 'FAIL'} "
                f"(key calls max diff {case.checks['key_calls_vs_bundle']['max_abs_diff']:.3g}; timeline rings "
                f"{case.checks['ready_calls_vs_timeline']['hist_ring_max_abs_diff']:.3g} / "
                f"{case.checks['ready_calls_vs_timeline']['fut_ring_max_abs_diff']:.3g}; GT bearing "
                f"{case.checks['ready_calls_vs_timeline']['gt_bearing_vs_label_pixel_max_deg']:.2f} deg)")
            if not case.checks["pass"]:
                status = 4
        seq, hold = frame_plan(case, args.fps, args.hold_call, args.hold_key, max_seconds=args.max_seconds)
        picks = preview_steps(case)
        for lang in args.lang:
            stem = f"{cat}_{ep_key}_{lang}"
            if not do_video:
                pdir = args.preview_dir or args.out_dir / "preview"
                want = {f"t{t:03d}": t for t in sorted(set(args.steps or []))} or picks
                out = pdir / "frames" / stem
                info = render_frames(case, lang, out, sorted(set(want.values())), args.workers, check_text=True)
                log(f"  {stem}: smallest font {info['min_font_pt']} pt; overlaps {info['text_overlaps'] or 'none'}; "
                    f"outside {info['texts_outside'] or 'none'}; data right of the playhead "
                    f"{info['frames_with_data_past_playhead'] or 'none'}")
                for name, t in want.items():
                    _jpeg(out / f"{t:05d}.png", pdir / (f"{stem}_{name}_t{t:03d}.jpg" if name in picks
                                                        else f"{stem}_t{t:03d}.jpg"))
                log(f"  {stem}: steps {want} -> {pdir}")
                continue
            frames = frames_root / stem
            if frames.exists():
                shutil.rmtree(frames)
            info = render_frames(case, lang, frames, range(case.n_steps + 1), args.workers, check_text=True)
            log(f"  {stem}: {info['n']} frames drawn ({info['sec_per_frame_median']:.2f} s/frame median); smallest "
                f"font {info['min_font_pt']} pt; frames with overlapping texts {len(info['text_overlaps'])}, with "
                f"texts off the frame {len(info['texts_outside'])}, with data right of the playhead "
                f"{len(info['frames_with_data_past_playhead'])}")
            if info["text_overlaps"] or info["texts_outside"] or info["frames_with_data_past_playhead"]:
                status = status or 6
            from PIL import Image

            with Image.open(frames / "00000.png") as im:
                size = im.size
            if size != (WIDTH_PX, HEIGHT_PX):
                raise RuntimeError(f"{stem}: frame is {size}, expected {(WIDTH_PX, HEIGHT_PX)}")
            mp4 = args.out_dir / f"{stem}.mp4"
            gif = args.out_dir / f"{stem}.gif"
            gif_every, gif_colors = gif_settings(case.n_steps)
            encode(frames, seq, mp4, gif, args.ffmpeg, fps=args.fps, crf=args.crf, gif_every=gif_every,
                   gif_colors=gif_colors)
            pm, pg = probe(mp4, args.ffmpeg), probe(gif, args.ffmpeg)
            log(f"  {stem}: mp4 {pm.get('width')}x{pm.get('height')} {pm.get('duration_s')} s "
                f"{pm['bytes'] / 1e6:.2f} MB; gif {pg.get('width')}x{pg.get('height')} {pg['bytes'] / 1e6:.2f} MB")
            if args.preview_dir is not None:
                for name, t in picks.items():
                    _jpeg(frames / f"{t:05d}.png", args.preview_dir / f"{stem}_{name}_t{t:03d}.jpg")
            manifest["videos"].append({
                "ep_key": ep_key, "category": cat, "lang": lang, "mp4": str(mp4), "gif": str(gif),
                "mp4_probe": pm, "gif_probe": pg, "fps": args.fps, "gif_fps": args.fps / gif_every,
                "gif_every": gif_every, "gif_colors": gif_colors,
                "frames": len(seq), "steps_drawn": case.n_steps + 1, "hold_call_frames": hold,
                "hold_key_frames": args.hold_key, "hold_start_frames": HOLD_START, "hold_end_frames": HOLD_END,
                "marker_stride": info["marker_stride"], "preview_steps": picks,
                "text_audit": {"min_font_pt": info["min_font_pt"], "frames_with_overlaps": info["text_overlaps"],
                               "frames_with_texts_outside": info["texts_outside"]},
                "frames_with_data_past_playhead": info["frames_with_data_past_playhead"],
                "timeline_marks_at": ("as on the static figure (history panel: up = left, ahead / left / behind / "
                                      "right / ahead from top to bottom): a call's dots and circles spread across its "
                                      "column by slot (frames 1, 4, 8 where a column is too narrow for 8; stacked at "
                                      "the column's centre on long reruns, see mark_mode), the System 1 endpoint at "
                                      "the column's centre; online, a call's field grows one step per frame (painted "
                                      "up to step t + 1) and each mark appears at its static position (its column is "
                                      "the call's own executed action chunk; it never moves) on the first frame whose "
                                      "painted field reaches its x; fields display-smoothed along the bearing (sigma "
                                      f"{tp.TL_SMOOTH_DEG:g} deg on the timeline, {p2.SMOOTH_DEG:g} deg on the strips, "
                                      "each ring keeps its maximum; the frame's note says so); playhead = dashed ink "
                                      "line at the painted frontier (step t + 1, the end of the step being executed; "
                                      "K lines stay at their calls' steps) with a 'now' tag under the step axis; a "
                                      "stacked plan's stride is in the legend (no in-panel note); current position on "
                                      "the map = "
                                      f"plain {POS_MS:g} pt ink dot with a {POS_HALO:g} pt white halo"),
                "mark_mode": info.get("mark_mode"),
                "sec_per_frame_median": info["sec_per_frame_median"], "inputs": case.inputs,
                "checks": case.checks})
            if not args.keep_frames:
                shutil.rmtree(frames)
        if do_video:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=_json_default)
                                     + "\n", encoding="utf-8")
    if do_video:
        if not args.keep_frames and frames_root.exists() and not any(frames_root.iterdir()):
            frames_root.rmdir()
        log(f"wrote {manifest_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
