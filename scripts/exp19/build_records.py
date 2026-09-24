#!/usr/bin/env python3
"""EXP-19 [F]: per-episode records, ground truth, pre-registered metrics and verdicts, figure bundles.

Joins, for every rerun episode of one run:
  [A] cases/candidates.json (schema exp19-candidates-v1) and
      cases/eval_log_reference/<ep_key>.json (exp19-eval-log-ref-v1)
  [B] runs/<run>/gpu<j>/trace/<ep_key>/call_<idx:03d>.{json,npz} (exp19-call-trace-v1)
  [C] runs/<run>/gpu<j>/steps/<ep_key>/steps.jsonl + front_<step:04d>.jpg (exp19-step-trace-v1)
  [E] renders/<ep_key>.npz (steps, rgb [N,4,256,256,3], depth_front [N,256,256] m, c2w_front)
and writes
  records/<ep_key>.json                every call with its per-call quantities, fidelity, key steps
  records/<ep_key>_bundle.{json,npz}   figure bundle (schema exp19-figure-bundle-v1), the only input of [G]
  metrics/metrics.json, summary.md, calls.jsonl (one row per call)

What is computed (docs/experiments/README.md EXP-19 指标 / 判据 / 有效性条件 / 关键时刻; the ledger
text is copied verbatim into metrics.json "definitions" / "criteria"):
* ready call = response kind "trajectory" and ppa_applied True (history head, bridge, future head ran).
* H1: joint PCK@8 with scripts/training/validate.py::_HeatmapJointMetricAccumulator semantics,
  pred = history head ``heatmaps`` (sigmoid) + ``visibility`` logits, GT = the label code on the
  recorded GT states (scripts/exp19/gt.py), slots = the request's history_capture_steps.
  Per-slot records are scored with scripts/exp18/compute_metrics.validator_metrics; with
  --self-check (default) the same tensors also go through the real accumulator and must agree
  bit for bit (otherwise the H1 verdict is "void" and the exit code is 4).  Constant
  straight-behind floor (class back, peak (32, 32), never none).
  95% CI: episode-cluster bootstrap, 10000 reps, seed 0 (exp18 SceneBootstrap with episodes as clusters).
* H2: per ready call and time bin, reference view5 = view of the bin's last waypoint of System1's
  selected mean path rendered by the future-label code; predicted view5 = argmax of
  future_visibility_probability, none if its max < 0.5.  Headline = agreement on bins whose
  reference is non-front; all-bins agreement and the executed-path reference are reported only.
* H3: share of ready calls whose counterfactual action chunk (Z instead of Z~, same noise, same
  post-processing incl. anti-deadlock; [B] diagnostic) differs from the deployed chunk; endpoint
  shift median / P90 of the two mean paths.  The rate is over every ready call: if any lacks its
  counterfactual the verdict is "missing" (the rate over the rest is kept as a descriptive number).
* Fidelity: first divergent call and identical calls vs the eval log; outcome same = success,
  oracle success and ended_by equal (step count reported separately).
* Gates (validity conditions): code equivalence = call 0 identical to the seed-42 eval log
  (System2 text AND action chunk) in every episode; trace neutrality = actions_match on 100% of
  trajectory calls, where a call the client made (episode_end vlm_calls, action records) without
  a trace counts as unverified; plus every candidate episode present.  If any fails, every
  verdict is "void" (the computed one is kept as verdict_if_valid) and the exit code is 3.
* Camera poses: gt.camera_c2w of the recorded body state, checked against the renders'
  c2w_front and against the deployed RGB sensor pose [C] records at every state (a position
  mismatch stops the build).  Every episode needs its [E] renders.
* Key moments: scripts/exp19/keysteps.py (pure function, unit tested).
* Pixel goal: which response field is the column is resolved from the data (gt.py
  resolve_pixel_goal_convention: base-rate-robust sign statistics of the column vs the System1
  endpoint's lateral offset); ambiguous -> stop with the evidence, unless forced by
  --pixel-goal-convention (recorded as forced).

Usage (dev machine, envs/qwen25, CPU only):
  cd <src> && PYTHONDONTWRITEBYTECODE=1 <qwen25 python> -m scripts.exp19.build_records --run <run>
Env fallbacks: EXP19_ROOT, EXP19_RUN, EXP19_TOPDOWN_ROOT, EXP19_BOOTSTRAP_REPS.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import geometry as geo  # noqa: E402
from scripts.exp18.compute_metrics import SceneBootstrap, _clean, _json_default, stat, validator_metrics  # noqa: E402
from scripts.exp19 import gt  # noqa: E402
from scripts.exp19.keysteps import closest_approach_step, net_turn_deg, select_key_steps  # noqa: E402

WORKSPACE = Path(os.environ.get("EXP19_WORKSPACE", "/mnt/afs/liwenhao/agent/370910109"))
EXP_ROOT = Path(os.environ.get("EXP19_ROOT", str(WORKSPACE / "model" / "exp19_behavior_viz")))
TOPDOWN_ROOT = Path(os.environ.get("EXP19_TOPDOWN_ROOT",
                                   str(WORKSPACE / "model" / "exp18_first_person_viz" / "topdown")))

SCHEMA_METRICS = "exp19-metrics-v1"
SCHEMA_RECORD = "exp19-episode-record-v1"
SCHEMA_BUNDLE = "exp19-figure-bundle-v1"
SCHEMA_CANDIDATES = "exp19-candidates-v1"
SCHEMA_EVAL_REF = "exp19-eval-log-ref-v1"
SCHEMA_TRACE = "exp19-call-trace-v1"
SCHEMA_STEPS = "exp19-step-trace-v1"
CATEGORY_ORDER = ("T1", "T2", "T3", "F1", "F2")
STEP_CAP = 500
C2W_TOL = 1e-3  # renders' c2w_front vs the state-derived camera pose, max abs element
FLOOR_CLASS = 1 + geo.BACK
H1 = {"pck8_min": 0.80, "gain_min": 0.20, "gain_ci_low_min": 0.10, "refute_gain_below": 0.05,
      "refute_pck8_below": 0.50}
H2 = {"n_min": 50, "agreement_min": 0.60, "gain_min": 0.20, "refute_gain_below": 0.05}
H3 = {"support_min": 0.10, "refute_below": 0.02}
VERDICT_ZH = {"support": "支持", "partial": "部分", "refute": "否定", "not_measured": "没测出来",
              "report_only": "只报数", "missing": "缺数据", "void": "作废"}
# Ledger text, verbatim (docs/experiments/README.md EXP-19).
DEFINITIONS = {
    "ready_call": "只从就绪调用里取——kind = trajectory 且 ppa_applied（慢系统发了像素目标，历史头、桥、未来头都跑了）",
    "H1": "joint PCK@8，scripts/training/validate.py 的 _HeatmapJointMetricAccumulator 口径（pred = 历史头 heatmaps + "
          "visibility logits，分母 = GT 可见槽位），汇总 15 集全部就绪调用。平凡基线 = 常数正后方（视角恒判后视、峰值 (32, 32)、"
          "从不判 none，同 EXP-18）。95% CI = 以集为簇的 bootstrap，10000 次、种子 0（场景 ≤ 11 个且极不均匀，以集为簇并如实标注）。"
          "描述性分层：类别、GT 视角。",
    "H2": "同一次调用的快系统均值路径按未来头标签口径（action_deltas_to_camera_points + render_future_trajectory_heatmaps）"
          "渲染成 4 时段 × 4 视角作参照，参照视角 = 该时段末路点所在视角（view5）。预测视角 = 该时段 future_visibility_probability "
          "的 argmax，最大值 < 0.5 判 none。指标 = 参照可见时段上的视角一致率；平凡基线 = 常数\"前\"。"
          "主读数只取参照为非前视的时段；全部时段的一致率只报数（常数\"前\"在那里天然占优，不判）。"
          "另报一个描述量：以复跑实际走过的路径（真值位置）作参照的同一指标。",
    "H3": "每个就绪调用的动作块改变率 = 反事实动作块（Z 代替 Z̃、同噪声、同后处理，含 anti-deadlock）≠ 部署动作块的调用占比；"
          "另报两条均值路径终点位移的中位数与 P90（m）。",
    "code_equivalence_gate": "复跑的第一次调用（慢系统文本 + 动作块）必须与主表种子 42 日志逐字相同，15 集全过才继续；"
                             "任何一集不过 → 停，查代码漂移。",
    "fidelity": "每集报与主表种子 42 client_N.log 的首个分歧调用序号、结局是否相同；整体报逐调用完全相同的集数。",
    "trace_neutrality_gate": "逐调用\"重算动作块 = 响应动作块\"必须 100% 成立，否则追踪读到的不是部署用的那条路径，整批作废。",
    "main_case": "每类取排序中第一个复跑后仍满足该类谓词的候选（谓词只看复跑自身的 success / os / 是否 STOP 结束 / 是否撞上限），不按好看挑。",
}
CRITERIA = {
    "H1": "支持 = PCK@8 ≥ 0.80、比平凡基线高 ≥ 20pt、且该差的 95% CI 下界 ≥ 10pt；否定 = 比基线高 < 5pt 或 PCK@8 < 0.50；"
          "其余 = 部分。支持 → 图注可写\"闭环中历史 affordance map 与真值吻合（PCK@8 X，平凡基线 Y）\"；"
          "部分 → 只许写\"好于平凡基线\"；否定 → 图里不许出现\"理解 / 记住了来路\"一类措辞，只能叫\"预测\"。",
    "H2": "参照非前视的时段 n < 50 → 没测出来（只报数，图注不提一致性）；否则 支持 = 一致率 ≥ 0.60 且比常数\"前\"高 ≥ 20pt；"
          "否定 = 比常数\"前\"高 < 5pt；其余 = 部分。只有支持时图注才可写\"未来 affordance map 与随后的动作方向一致\"。",
    "H3": "改变率 ≥ 10% → 支持依赖，图注可写\"撤掉历史记忆时 X% 的决定点动作块改变\"（只写依赖）；"
          "改变率 < 2% → 否定，图注与正文不得暗示 affordance map 在这些时刻驱动了动作，只能写成\"模型内部状态的展示\"；"
          "其余 → 只报数，图注不提。",
}
NPZ_SHAPES = {  # contract shapes; a leading singleton batch dim is dropped
    "hist_heatmaps_gated": (8, 4, 64, 64), "hist_heatmaps": (8, 4, 64, 64), "hist_visibility_logits": (8, 4),
    "hist_none_probability": (8,), "hist_mask": (8,), "fut_heatmaps_gated": (4, 4, 64, 64),
    "fut_visibility_probability": (4, 4), "selected_path_xy": (33, 2), "cf_selected_path_xy": (33, 2),
    "hist_view_peak_yx": (8, 4, 2),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def npz_array(z, name: str) -> np.ndarray:
    a = np.asarray(z[name])
    want = NPZ_SHAPES.get(name)
    if want is None:
        return a
    while a.ndim > len(want) and a.shape[0] == 1:
        a = a[0]
    if a.shape != want:
        raise ValueError(f"{name}: shape {a.shape}, expected {want}")
    return a


def selected_path(c: dict, z) -> np.ndarray | None:
    """System1's selected mean path [33, 2] of one traced call: the npz copy, else the JSON copy."""
    if "selected_path_xy" in z.files:
        return npz_array(z, "selected_path_xy").astype(np.float64)
    if c.get("selected_path_xy") is not None:
        return npz_array({"selected_path_xy": np.asarray(c["selected_path_xy"], dtype=np.float64)}, "selected_path_xy")
    return None


def decode_jpeg(data) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(np.asarray(data, dtype=np.uint8).tobytes())).convert("RGB"))


def load_candidates(path: Path) -> dict:
    """ep_key -> memberships [{category, rank, candidate}] in category order, plus the ordered lists."""
    data = read_json(path)
    if data.get("schema") != SCHEMA_CANDIDATES:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA_CANDIDATES}")
    members, ordered = {}, {}
    for cat in CATEGORY_ORDER:
        cands = sorted((data.get("categories") or {}).get(cat, {}).get("candidates", []), key=lambda c: int(c["rank"]))
        ordered[cat] = [c["ep_key"] for c in cands]
        for c in cands:
            members.setdefault(c["ep_key"], []).append({"category": cat, "rank": int(c["rank"]), "candidate": c})
    return {"data": data, "members": members, "ordered": ordered}


def discover_run(run_dir: Path) -> dict:
    """ep_key -> {gpu, trace_dir, steps_dir} over runs/<run>/gpu*/."""
    found = {}
    for gpu_dir in sorted(p for p in run_dir.glob("gpu*") if p.is_dir()):
        for steps_file in sorted(gpu_dir.glob("steps/*/steps.jsonl")):
            ep_key = steps_file.parent.name
            if ep_key in found:
                raise ValueError(f"{ep_key} appears under both {found[ep_key]['gpu']} and {gpu_dir.name}")
            found[ep_key] = {"gpu": gpu_dir.name, "steps_dir": steps_file.parent,
                             "trace_dir": gpu_dir / "trace" / ep_key}
    return found


def load_steps(steps_dir: Path) -> dict:
    recs = read_jsonl(steps_dir / "steps.jsonl")
    starts = [r for r in recs if r.get("type") == "episode_start"]
    if len(starts) != 1 or starts[0].get("schema") != SCHEMA_STEPS:
        raise ValueError(f"{steps_dir}: need exactly one episode_start with schema {SCHEMA_STEPS}")
    states, duplicates = {}, 0
    for r in recs:
        if r.get("type") == "state":
            if int(r["step"]) in states:
                duplicates += 1
            else:
                states[int(r["step"])] = r
    ends = [r for r in recs if r.get("type") == "episode_end"]
    return {"start": starts[0], "end": ends[-1] if ends else None, "states": states,
            "actions": [r for r in recs if r.get("type") == "action"], "duplicate_states": duplicates}


def load_trace(trace_dir: Path) -> dict:
    calls = []
    for path in sorted(trace_dir.glob("call_*.json")):
        c = read_json(path)
        if c.get("schema") != SCHEMA_TRACE:
            raise ValueError(f"{path}: schema {c.get('schema')!r}, expected {SCHEMA_TRACE}")
        npz = path.with_suffix(".npz")
        c["_npz"] = npz if npz.is_file() else None
        calls.append(c)
    calls.sort(key=lambda c: int(c["system2_call_index"]))
    idx = [int(c["system2_call_index"]) for c in calls]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{trace_dir}: duplicate system2_call_index")
    err_file = trace_dir / "trace_errors.jsonl"
    return {"calls": calls, "trace_errors": read_jsonl(err_file) if err_file.is_file() else []}


def load_renders(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: every episode needs [E] renders (scripts/exp19/render_views.py) "
                                "for its GT depth and 360-degree backdrop")
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in ("steps", "rgb", "depth_front", "c2w_front")}
    out["index"] = {int(s): i for i, s in enumerate(out["steps"])}
    return out


# --------------------------------------------------------------------------- #
# Per-episode join
# --------------------------------------------------------------------------- #
class Episode:
    def __init__(self, ep_key: str, paths: dict, renders_dir: Path):
        self.ep_key = ep_key
        self.gpu = paths["gpu"]
        self.steps_dir = Path(paths["steps_dir"])
        self.trace_dir = Path(paths["trace_dir"])
        self.steps = load_steps(self.steps_dir)
        self.trace = load_trace(self.trace_dir)
        self.renders = load_renders(renders_dir / f"{ep_key}.npz")
        self.start = self.steps["start"]
        self.scene_id, self.episode_id = str(self.start["scene_id"]), int(self.start["episode_id"])
        self.goal = np.asarray(self.start["goal_position"], dtype=np.float64)
        self._cam = {}
        # Calls the client made: 0 .. vlm_calls - 1 (episode_end), every index an action names, and every
        # index below a traced one.
        traced = {int(c["system2_call_index"]) for c in self.trace["calls"]}
        made = set(int(a["system2_call_index"]) for a in self.steps["actions"] if a.get("system2_call_index") is not None)
        end = self.steps["end"]
        counted = end is not None and end.get("vlm_calls") is not None
        if counted:
            made |= set(range(int(end["vlm_calls"])))
        self.traces_beyond_client_count = sorted(traced - made) if counted else []
        made |= set(range(max(traced) + 1)) if traced else set()
        self.missing_call_indices = sorted(made - traced)

    def state(self, step: int) -> dict:
        if step not in self.steps["states"]:
            raise KeyError(f"{self.ep_key}: no state record for step {step}")
        return self.steps["states"][step]

    def cam(self, step: int) -> np.ndarray:
        if step not in self._cam:
            s = self.state(step)
            self._cam[step] = gt.camera_c2w(s["position"], s["rotation_wxyz"])
        return self._cam[step]

    def position(self, step: int) -> np.ndarray:
        return np.asarray(self.state(step)["position"], dtype=np.float64)

    def dist_to_goal(self, step: int) -> float:
        return float(np.linalg.norm(self.position(step) - self.goal))

    def outcome(self) -> dict | None:
        end = self.steps["end"]
        if end is None:
            return None
        m = end.get("metrics") or {}
        ended_by = end.get("ended_by")
        if not ended_by:
            last = self.steps["actions"][-1]["action"] if self.steps["actions"] else None
            ended_by = "step_cap" if int(end["steps"]) >= STEP_CAP else ("stop" if last == 0 else "unknown")
        return {"success": float(m.get("success", float("nan"))), "oracle_success": float(m.get("oracle_success", float("nan"))),
                "ne_m": float(m.get("distance_to_goal", float("nan"))), "spl": float(m.get("spl", float("nan"))),
                "steps": int(end["steps"]), "ended_by": str(ended_by)}

    def sensor_pose_error(self) -> dict:
        """max |camera_c2w(state) - the deployed RGB sensor's own pose ([C] rgb_sensor_*)| over all states."""
        pos, rot, n = 0.0, 0.0, 0
        for step, s in self.steps["states"].items():
            if s.get("rgb_sensor_position") is None or s.get("rgb_sensor_rotation_wxyz") is None:
                continue
            T = self.cam(step)
            pos = max(pos, float(np.abs(np.asarray(s["rgb_sensor_position"], dtype=np.float64) - T[:3, 3]).max()))
            rot = max(rot, float(np.abs(gt.quat_wxyz_to_rot(s["rgb_sensor_rotation_wxyz"]) - T[:3, :3]).max()))
            n += 1
        return {"n_states": n, "max_abs_position_m": pos if n else None, "max_abs_rotation": rot if n else None}

    def render(self, step: int):
        if step not in self.renders["index"]:
            raise KeyError(f"{self.ep_key}: no re-render at step {step}")
        i = self.renders["index"][step]
        return self.renders["rgb"][i], self.renders["depth_front"][i], self.renders["c2w_front"][i]


def executed_by_call(calls: list, actions: list) -> tuple:
    """call_index -> executed action codes, from the action records' system2_call_index.

    [C] writes system2_call_index = null for stops no call produced (auto-stop, max-System2
    stop); those belong to no call.  Only a record without the field falls back to the step
    range (the latest traced call at or before its step); the range is also a cross-check.
    """
    order = sorted((int(c["current_capture_step"]), int(c["system2_call_index"])) for c in calls)
    out = {int(c["system2_call_index"]): [] for c in calls}
    inferred = unattributed = mismatched = 0
    for a in actions:
        by_range = None
        for step, idx in order:  # the latest call at or before this action's step made its chunk
            if step <= int(a["step_before"]):
                by_range = idx
        if "system2_call_index" not in a:
            idx, inferred = by_range, inferred + 1
        else:
            idx = a["system2_call_index"]
            unattributed += idx is None
            mismatched += idx is not None and by_range is not None and int(idx) != by_range
        if idx is not None:
            out.setdefault(int(idx), []).append(int(a["action"]))
    return out, {"actions_without_call_index_field": inferred, "actions_of_no_call": unattributed,
                 "call_index_vs_step_range_mismatch": mismatched}


def view_peaks(maps: np.ndarray) -> np.ndarray:
    """Per-view first-occurrence argmax [..., 4, H, W] -> [..., 4, 2] (row, col), like torch.argmax."""
    row, col = geo.argmax_pixel(maps)
    return np.stack([row, col], axis=-1).astype(np.int64)


def score_slots(pred_logits: np.ndarray, pred_peaks: np.ndarray, gt_cls: np.ndarray, gt_row: np.ndarray,
                gt_col: np.ndarray) -> dict:
    """validate.py prediction side per slot: 5-way class, and the argmax (row, col) in the GT view."""
    view5 = np.concatenate([np.zeros_like(pred_logits[..., :1]), pred_logits], axis=-1).argmax(-1)
    tv = np.clip(gt_cls - 1, 0, 3)
    yx = np.take_along_axis(pred_peaks, tv[:, None, None], axis=1)[:, 0]
    sq = (yx[:, 0] - gt_row) ** 2 + (yx[:, 1] - gt_col) ** 2
    return {"view5": view5.astype(np.int64), "row": yx[:, 0].astype(np.int64), "col": yx[:, 1].astype(np.int64),
            "sq": sq.astype(np.int64)}


def _call_key(c: dict) -> tuple:
    return (int(c["step"]), str(c["kind"]), str(c.get("vlm_output") or "").strip(),
            [int(a) for a in (c.get("actions") or [])])


def compare_calls(rerun: list, ref: list) -> dict:
    """Call-by-call equality of (step, kind, System2 text, action chunk) against the eval-log reference.

    Both sides are matched by call index (rerun: system2_call_index; reference: call_index), so a
    missing trace counts as a difference instead of shifting every later call.
    """
    a_by = {int(c["call_index"]): _call_key(c) for c in rerun}
    b_by = {int(c["call_index"]): _call_key(c) for c in ref}
    n = max(list(a_by) + list(b_by), default=-1) + 1
    per_call = [i in a_by and a_by.get(i) == b_by.get(i) for i in range(n)]
    first = next((i for i, same in enumerate(per_call) if not same), None)
    return {"first_divergent_call": first, "identical_calls": int(sum(per_call)),
            "identical_prefix_calls": first if first is not None else n,
            "total_calls": len(rerun), "reference_calls": len(ref), "all_identical": first is None,
            "per_call_identical": per_call}


def compare_outcome(outcome: dict | None, ref_final: dict | None) -> dict | None:
    """Rerun outcome vs the seed-42 eval log (fidelity: 结局是否相同).

    "same" = success, oracle success and, when the reference records it, how the episode ended
    (stop / step_cap, part of the F1 / F2 predicates).  The step count is compared separately and
    is descriptive only: the same outcome can take a different number of steps.
    """
    if outcome is None or ref_final is None:
        return None
    success = (outcome["success"] >= 0.5) == (float(ref_final["success"]) >= 0.5)
    oracle = (outcome["oracle_success"] >= 0.5) == (float(ref_final["os"]) >= 0.5)
    ended = outcome["ended_by"] == ref_final["ended_by"] if ref_final.get("ended_by") is not None else None
    steps = outcome["steps"] == int(ref_final["steps"]) if ref_final.get("steps") is not None else None
    return {"same": bool(success and oracle and ended is not False), "success_equal": success,
            "oracle_success_equal": oracle, "ended_by_equal": ended, "steps_equal": steps,
            "rerun": {"ended_by": outcome["ended_by"], "steps": outcome["steps"]},
            "eval_log": {"ended_by": ref_final.get("ended_by"), "steps": ref_final.get("steps")}}


def rerun_predicate(category: str, outcome: dict | None):
    """Category predicate on the rerun's own outcome (success / os / STOP-ended / step cap only)."""
    if outcome is None:
        return None
    s, os_, ended = outcome["success"] >= 0.5, outcome["oracle_success"] >= 0.5, outcome["ended_by"]
    if category in ("T1", "T2", "T3"):
        return bool(s)
    if category == "F1":
        return bool(os_ and not s and ended == "stop")
    if category == "F2":
        return bool(not s and ended == "step_cap")
    raise ValueError(category)


def analyse_episode(ep: Episode, category: str, eval_ref: dict | None, accs: dict | None) -> dict:
    """Everything per episode except the pixel-goal convention (needs all episodes) and the bundle."""
    calls = ep.trace["calls"]
    warnings = []
    if ep.missing_call_indices:
        warnings.append(f"calls without a trace: {ep.missing_call_indices}")
    if ep.traces_beyond_client_count:
        warnings.append(f"traces of calls the client did not count: {ep.traces_beyond_client_count}")
    if ep.trace["trace_errors"]:
        warnings.append(f"{len(ep.trace['trace_errors'])} trace_errors.jsonl rows")
    for c in calls:  # [B]'s own consistency checks (captured tensors = the ones the deployment passed on)
        for w in c.get("trace_warnings") or []:
            warnings.append(f"call {c['system2_call_index']} trace warning: {w}")
        for e in (c.get("diagnostic") or {}).get("errors") or []:
            warnings.append(f"call {c['system2_call_index']} diagnostic {e.get('diagnostic')} failed")
    if ep.steps["duplicate_states"]:
        warnings.append(f"{ep.steps['duplicate_states']} duplicate state records (first kept)")
    executed, exec_check = executed_by_call(calls, ep.steps["actions"])
    outcome = ep.outcome()
    steps_sorted = sorted(ep.steps["states"])

    # camera poses derived from the body states vs the renders and vs the deployed sensor itself.
    # Positions feed every label (history slots, executed path); orientation only matters at the
    # call steps, which the renders cover.
    errs = [float(np.abs(ep.renders["c2w_front"][i] - ep.cam(int(s))).max())
            for i, s in enumerate(ep.renders["steps"]) if int(s) in ep.steps["states"]]
    c2w_err = max(errs) if errs else None
    if c2w_err is not None and c2w_err > C2W_TOL:
        raise ValueError(f"{ep.ep_key}: renders c2w_front differs from the state pose by {c2w_err:.4g} > {C2W_TOL}")
    sensor = ep.sensor_pose_error()
    if sensor["n_states"] == 0:
        warnings.append("no rgb_sensor_* in the state records: camera pose not cross-checked against the sensor")
    elif sensor["max_abs_position_m"] > C2W_TOL:
        raise ValueError(f"{ep.ep_key}: camera position from the body state differs from the deployed RGB sensor "
                         f"by {sensor['max_abs_position_m']:.4g} m > {C2W_TOL}")
    elif sensor["max_abs_rotation"] > C2W_TOL:
        warnings.append(f"deployed RGB sensor orientation differs from the level body camera by "
                        f"{sensor['max_abs_rotation']:.4g} at some state")

    rows, slot_rows, bin_rows = [], [], []
    ref_calls = (eval_ref or {}).get("calls") or []
    for c in calls:
        idx = int(c["system2_call_index"])
        resp = c.get("response") or {}
        step = int(c["current_capture_step"])
        kind = resp.get("kind")
        ready = kind == "trajectory" and resp.get("ppa_applied") is True
        acts = [int(a) for a in (resp.get("actions") or [])]
        ex = executed.get(idx, [])
        diag = (c.get("diagnostic") or {}).get("counterfactual_no_memory")
        replay = (c.get("diagnostic") or {}).get("replay_same_plan")
        hist_steps = [int(s) for s in (c.get("history_capture_steps") or [])]
        row = {
            "ep_key": ep.ep_key, "scene_id": ep.scene_id, "episode_id": ep.episode_id, "category": category,
            "call_index": idx, "step": step, "kind": kind, "ready": ready, "ppa_applied": resp.get("ppa_applied"),
            "pose_ready": c.get("pose_ready"), "trajectory_path": c.get("trajectory_path"),
            "vlm_output": resp.get("llm_output"), "first_output": resp.get("native_first_output"),
            "lookdown_turns": resp.get("native_lookdown_turns"), "pixel_goal_field": resp.get("pixel_goal"),
            "response_actions": acts, "executed_actions": ex, "executed_net_turn_deg": net_turn_deg(ex),
            "recomputed_actions": c.get("recomputed_actions"), "actions_match": c.get("actions_match"),
            "actions_match_recheck": (c.get("recomputed_actions") == acts) if c.get("recomputed_actions") is not None else None,
            "anti_deadlock": resp.get("anti_deadlock"), "history_steps": hist_steps, "n_history": len(hist_steps),
            "dist_to_goal_m": ep.dist_to_goal(step) if step in ep.steps["states"] else None,
            "cf_actions": diag.get("actions") if diag else None,
            "cf_changed": (list(diag.get("actions") or []) != acts) if diag else None,
            "endpoint_shift_m": diag.get("endpoint_shift_m") if diag else None,
            "replay_actions_equal": replay.get("actions_equal") if replay else None,
            "replay_endpoint_shift_m": replay.get("endpoint_shift_m") if replay else None,
            "selected_path_end_xy": None,
        }
        ref = next((rc for rc in ref_calls if int(rc["call_index"]) == idx), None)
        row["ref_call"] = ({k: ref.get(k) for k in ("step", "kind", "vlm_output", "actions", "traj_goal")}
                           if ref is not None else None)
        if ready:
            if c["_npz"] is None:
                raise ValueError(f"{ep.ep_key} call {idx}: ready call without an npz")
            _ready_call(ep, c, row, slot_rows, bin_rows, accs, steps_sorted)
        rows.append(row)

    rerun_for_cmp = [{"call_index": r["call_index"], "step": r["step"], "kind": r["kind"], "vlm_output": r["vlm_output"],
                      "actions": r["response_actions"]} for r in rows]
    fidelity = compare_calls(rerun_for_cmp, ref_calls) if eval_ref is not None else None
    for r in rows:
        per_call = (fidelity or {}).get("per_call_identical", [])
        r["identical_to_ref"] = per_call[r["call_index"]] if r["call_index"] < len(per_call) else None
    code_eq = None
    if eval_ref is not None:
        a = next((r for r in rows if r["call_index"] == 0), None)
        b = next((c for c in ref_calls if int(c["call_index"]) == 0), None)
        if a is None or b is None:
            code_eq = {"pass": False, "reason": "call 0 missing in the " + ("rerun trace" if a is None else "eval-log reference")}
        else:  # the gate is System2 text AND action chunk, verbatim
            code_eq = {"pass": (str(a["vlm_output"] or "").strip() == str(b.get("vlm_output") or "").strip()
                                and a["response_actions"] == [int(x) for x in (b.get("actions") or [])]),
                       "rerun": {"vlm_output": a["vlm_output"], "actions": a["response_actions"]},
                       "reference": {"vlm_output": b.get("vlm_output"), "actions": b.get("actions")}}
    outcome_cmp = compare_outcome(outcome, (eval_ref or {}).get("final"))

    ready_rows = [r for r in rows if r["ready"]]
    closest = None
    if category == "F1" and steps_sorted:
        closest = closest_approach_step(steps_sorted, [ep.dist_to_goal(s) for s in steps_sorted])
    keys = select_key_steps(ready_rows, category=category, episode_steps=outcome["steps"] if outcome else
                            (steps_sorted[-1] + 1 if steps_sorted else 0), closest_step=closest) if ready_rows else []
    if not keys:
        warnings.append("no ready call in the rerun: the bundle has no key steps")
    by_call = {k["call_index"]: k["label"] for k in keys}
    for r in rows:
        r["key_label"] = by_call.get(r["call_index"])
    return {"rows": rows, "slot_rows": slot_rows, "bin_rows": bin_rows, "outcome": outcome, "fidelity": fidelity,
            "code_equivalence": code_eq, "outcome_comparison": outcome_cmp,
            "outcome_same_as_eval_log": outcome_cmp["same"] if outcome_cmp else None, "key_steps": keys,
            "closest_approach_step": closest,
            "checks": {"renders_c2w_max_abs_err": c2w_err, "sensor_pose": sensor, **exec_check},
            "warnings": warnings}


def _ready_call(ep, c, row, slot_rows, bin_rows, accs, steps_sorted) -> None:
    idx, step = row["call_index"], row["step"]
    with np.load(c["_npz"], allow_pickle=False) as z:
        need = [n for n in ("hist_visibility_logits", "hist_heatmaps", "hist_none_probability",
                            "fut_visibility_probability") if n not in z.files]
        if need:
            raise ValueError(f"{ep.ep_key} call {idx}: ready call whose npz lacks {need}")
        logits = npz_array(z, "hist_visibility_logits").astype(np.float32)
        # [B] stores the per-view argmax of the f32 sigmoid maps; the f16 copy can tie near 1.
        exact = "hist_view_peak_yx" in z.files
        peaks = (npz_array(z, "hist_view_peak_yx").astype(np.int64) if exact
                 else view_peaks(npz_array(z, "hist_heatmaps").astype(np.float32)))
        none_p = npz_array(z, "hist_none_probability").astype(np.float64)
        mask = npz_array(z, "hist_mask").astype(bool) if "hist_mask" in z.files else None
        fut_vis = npz_array(z, "fut_visibility_probability").astype(np.float64)
        path = selected_path(c, z)
    if path is None:  # fatal like the other required ready-call arrays: H2, the convention and the figure need it
        raise ValueError(f"{ep.ep_key} call {idx}: ready call without selected_path_xy (neither npz nor json); [B] "
                         "leaves it empty only when it captured no System1 trajectory, so actions_match fails too")
    k_hist = row["n_history"]
    if mask is None:
        mask = np.arange(gt.NUM_SLOTS) < k_hist
    elif int(mask.sum()) != k_hist or not mask[:k_hist].all():
        raise ValueError(f"{ep.ep_key} call {idx}: hist_mask {mask.astype(int).tolist()} vs {k_hist} history steps")
    row["selected_path_end_xy"] = path[-1].tolist()

    # ---- H1: GT on the recorded states, scored like the accumulator -------------
    _, depth, _ = ep.render(step)
    gmap, gvis = gt.history_labels([ep.cam(s) for s in row["history_steps"]], ep.cam(step), depth)
    gmap, gvis, _ = gt.pad_slots(gmap, gvis)
    cls, grow, gcol = gt.gt_view_class_and_peak(gmap, gvis)
    model = score_slots(logits, peaks, cls, grow, gcol)
    floor_sq = (grow - geo.FLOOR_PEAK_YX[0]) ** 2 + (gcol - geo.FLOOR_PEAK_YX[1]) ** 2
    n_vis = n_j8 = n_f8 = 0
    for k in np.nonzero(mask)[0]:
        vis = bool(cls[k] > 0)
        j8 = vis and model["view5"][k] == cls[k] and model["sq"][k] <= 64
        f8 = vis and cls[k] == FLOOR_CLASS and floor_sq[k] <= 64
        n_vis, n_j8, n_f8 = n_vis + vis, n_j8 + j8, n_f8 + f8
        slot_rows.append({
            "ep_key": ep.ep_key, "category": row["category"], "call_index": idx, "step": step, "slot": int(k),
            "hist_step": row["history_steps"][k], "age": step - row["history_steps"][k],
            "gt_visible": vis, "gt_class": int(cls[k]), "gt_peak_row": int(grow[k]), "gt_peak_col": int(gcol[k]),
            "gt_vis_front": bool(gvis[k, 0] > 0.5), "gt_vis_right": bool(gvis[k, 1] > 0.5),
            "gt_vis_back": bool(gvis[k, 2] > 0.5), "gt_vis_left": bool(gvis[k, 3] > 0.5),
            "pred_model_view5": int(model["view5"][k]), "pred_model_row": int(model["row"][k]),
            "pred_model_col": int(model["col"][k]), "pred_model_none_prob": float(none_p[k]),
            "pred_model_sq_err": int(model["sq"][k]) if vis else -1,
            "pred_model_joint4": bool(vis and model["view5"][k] == cls[k] and model["sq"][k] <= 16),
            "pred_model_joint8": bool(j8),
            "pred_floor_view5": FLOOR_CLASS, "pred_floor_sq_err": int(floor_sq[k]) if vis else -1,
            "pred_floor_joint4": bool(vis and cls[k] == FLOOR_CLASS and floor_sq[k] <= 16),
            "pred_floor_joint8": bool(f8),
        })
    row.update(h1_n_valid=int(mask.sum()), h1_n_visible=n_vis, h1_joint8=int(n_j8), h1_floor_joint8=int(n_f8),
               h1_pck8=n_j8 / n_vis if n_vis else None, h1_pred_peaks="f32_exact" if exact else "f16_maps")
    if accs is not None:
        _update_accumulators(accs, logits, peaks, gvis, gmap, mask)

    # ---- H2: System1-path and executed-path references vs predicted view --------
    pred5 = gt.predicted_view5(fut_vis)
    ref_s1 = gt.future_reference_from_path(path).view5
    later = [s for s in steps_sorted if s > step]
    ref_ex_t = gt.future_reference_from_poses(ep.cam(step), [ep.cam(s) for s in later])
    ref_ex = ref_ex_t.view5 if ref_ex_t is not None else None
    for b in range(4):
        bin_rows.append({"ep_key": ep.ep_key, "category": row["category"], "call_index": idx, "step": step, "bin": b,
                         "pred5": int(pred5[b]), "pred_max_prob": float(fut_vis[b].max()), "ref5_s1": int(ref_s1[b]),
                         "ref5_exec": int(ref_ex[b]) if ref_ex is not None else -1})
    row.update(h2_pred5=pred5.tolist(), h2_ref5_s1=ref_s1.tolist(),
               h2_ref5_exec=ref_ex.tolist() if ref_ex is not None else None,
               h2_pred_max_prob=fut_vis.max(-1).round(4).tolist())


def _one_hot(peaks: np.ndarray) -> np.ndarray:
    """[8, 4, 2] per-view peaks -> [8, 4, 64, 64] maps whose per-view argmax is that peak."""
    maps = np.zeros(peaks.shape[:-1] + gt.HM_SIZE, dtype=np.float32)
    k, v = np.indices(peaks.shape[:-1])
    maps[k, v, peaks[..., 0], peaks[..., 1]] = 1.0
    return maps


def _update_accumulators(accs, logits, peaks, gvis, gmap, mask) -> None:
    """Feed validate.py's accumulator the same slots (maps rebuilt from the peaks it would argmax)."""
    import torch
    t = torch.from_numpy
    accs["model"].update(pred_visibility_logits=t(logits[None]), pred_heatmaps=t(_one_hot(peaks)[None]),
                         gt_visibility=t(gvis[None]), gt_heatmaps=t(gmap[None]), history_mask=t(mask[None]))
    floor_logits = np.full((1, 8, 4), -10.0, dtype=np.float32)
    floor_logits[..., geo.BACK] = 10.0
    floor_maps = _one_hot(np.broadcast_to(np.asarray(geo.FLOOR_PEAK_YX), (8, 4, 2)))
    accs["floor"].update(pred_visibility_logits=t(floor_logits), pred_heatmaps=t(floor_maps[None]),
                         gt_visibility=t(gvis[None]), gt_heatmaps=t(gmap[None]), history_mask=t(mask[None]))


# --------------------------------------------------------------------------- #
# Metrics and verdicts
# --------------------------------------------------------------------------- #
def _verdict(name: str, **fields) -> dict:
    return {"verdict": name, "label_zh": VERDICT_ZH[name], **fields}


def _r(x: float) -> float:
    """Compare against a threshold at 1e-12: 0.85 - 0.65 must count as the 20pt it is."""
    return round(float(x), 12)


def h1_decision(pck8, gain, gain_ci_low) -> str:
    """CRITERIA["H1"]: support = PCK@8 >= 0.80 and gain >= 20pt and gain CI low >= 10pt;
    refute = gain < 5pt or PCK@8 < 0.50; otherwise partial."""
    if pck8 is None or gain is None or gain_ci_low is None:
        return "missing"
    pck8, gain, low = _r(pck8), _r(gain), _r(gain_ci_low)
    if pck8 >= H1["pck8_min"] and gain >= H1["gain_min"] and low >= H1["gain_ci_low_min"]:
        return "support"
    if gain < H1["refute_gain_below"] or pck8 < H1["refute_pck8_below"]:
        return "refute"
    return "partial"


def h2_decision(n_nonfront: int, agreement, gain) -> str:
    """CRITERIA["H2"]: n < 50 non-front reference bins -> not measured; support = agreement >= 0.60 and
    gain over constant front >= 20pt; refute = gain < 5pt; otherwise partial."""
    if n_nonfront < H2["n_min"]:
        return "not_measured"
    agreement, gain = _r(agreement), _r(gain)
    if agreement >= H2["agreement_min"] and gain >= H2["gain_min"]:
        return "support"
    if gain < H2["refute_gain_below"]:
        return "refute"
    return "partial"


def h3_decision(change_rate, n_missing: int = 0) -> str:
    """CRITERIA["H3"]: change rate >= 10% -> support (dependence); < 2% -> refute; otherwise report only.

    The rate is over every ready call (每个就绪调用): with any ready call lacking its counterfactual it is
    undefined -> missing (a rate over the rest would describe an undeclared subset).
    """
    if change_rate is None or n_missing:
        return "missing"
    rate = _r(change_rate)
    if rate >= H3["support_min"]:
        return "support"
    if rate < H3["refute_below"]:
        return "refute"
    return "report_only"


def h1_block(slots: pd.DataFrame, reps: int, seed: int) -> dict:
    if not len(slots):
        return {"n_valid_slots": 0, "n_visible_slots": 0, "verdict": _verdict("missing", reason="no scored slots")}
    boot = SceneBootstrap(slots["ep_key"].to_numpy(), reps, np.random.default_rng(seed))  # clusters = episodes
    vis = slots["gt_visible"].to_numpy()
    out = {"n_valid_slots": int(len(slots)), "n_visible_slots": int(vis.sum()), "n_episodes": int(boot.S),
           "n_ready_calls": int(slots.groupby(["ep_key", "call_index"]).ngroups)}
    pr = {}
    for arm in ("model", "floor"):
        for key, col in (("joint_pck8", "joint8"), ("joint_pck4", "joint4")):
            p, r = boot.ratio(slots[f"pred_{arm}_{col}"].to_numpy() & vis, vis)
            out.setdefault(arm, {})[key] = stat(p, r)
            pr[(arm, key)] = (p, r)
        out[arm]["validator"] = validator_metrics(slots, arm)
    (pm, rm), (pf, rf) = pr[("model", "joint_pck8")], pr[("floor", "joint_pck8")]
    out["gain_over_floor_pck8"] = stat(pm - pf, rm - rf)
    strata = {}
    for fam, labels, order in (
            ("category", slots["category"].to_numpy(), CATEGORY_ORDER),
            ("gt_view", np.asarray(geo.VIEW_CLASS_NAMES, dtype=object)[slots["gt_class"].to_numpy()], geo.VIEW_NAMES)):
        cells = {}
        for name in order:
            sel = vis & (labels == name)
            n = int(sel.sum())
            cells[name] = {"n": n,
                           "model_pck8": float(slots["pred_model_joint8"].to_numpy()[sel].mean()) if n else None,
                           "floor_pck8": float(slots["pred_floor_joint8"].to_numpy()[sel].mean()) if n else None}
        strata[fam] = cells
    out["strata"] = strata
    g = out["gain_over_floor_pck8"]
    p8, gain, lo = out["model"]["joint_pck8"]["value"], g["value"], g["ci95"][0]
    fields = {"joint_pck8": p8, "floor_pck8": out["floor"]["joint_pck8"]["value"], "gain": gain, "gain_ci95_low": lo,
              "criterion": CRITERIA["H1"], "thresholds": H1}
    decision = h1_decision(p8, gain, lo)
    out["verdict"] = (_verdict(decision, reason="metric undefined", **fields) if decision == "missing"
                      else _verdict(decision, **fields))
    return out


def _agreement(df: pd.DataFrame, ref: str, sel: np.ndarray, boot) -> dict:
    agree = (df["pred5"].to_numpy() == df[ref].to_numpy()) & sel
    front = (df[ref].to_numpy() == 1) & sel
    p, r = boot.ratio(agree, sel)
    pf, rf = boot.ratio(front, sel)
    return {"n": int(sel.sum()), "agreement": stat(p, r), "constant_front": stat(pf, rf), "gain_over_front": stat(p - pf, r - rf)}


def h2_block(bins: pd.DataFrame, reps: int, seed: int) -> dict:
    if not len(bins):
        return {"n_bins": 0, "verdict": _verdict("missing", reason="no ready calls")}
    boot = SceneBootstrap(bins["ep_key"].to_numpy(), reps, np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,))))
    out = {"n_bins": int(len(bins)), "n_ready_calls": int(bins.groupby(["ep_key", "call_index"]).ngroups)}
    for ref in ("ref5_s1", "ref5_exec"):
        r = bins[ref].to_numpy()
        name = "system1_path" if ref == "ref5_s1" else "executed_path"
        out[name] = {"nonfront": _agreement(bins, ref, r >= 2, boot), "all_visible": _agreement(bins, ref, r >= 1, boot),
                     "reference_view5_counts": {geo.VIEW_CLASS_NAMES[v]: int((r == v).sum()) for v in range(5)},
                     "reference_missing": int((r < 0).sum())}
        sel = r >= 2
        out[name]["nonfront"]["confusion_ref_x_pred"] = {
            geo.VIEW_CLASS_NAMES[a]: {geo.VIEW_CLASS_NAMES[b]: int(((r == a) & (bins["pred5"].to_numpy() == b) & sel).sum())
                                      for b in range(5)} for a in range(2, 5)}
    out["pred_view5_counts"] = {geo.VIEW_CLASS_NAMES[v]: int((bins["pred5"].to_numpy() == v).sum()) for v in range(5)}
    out["by_category_nonfront_s1"] = {}
    for cat in CATEGORY_ORDER:
        sel = (bins["category"].to_numpy() == cat) & (bins["ref5_s1"].to_numpy() >= 2)
        n = int(sel.sum())
        out["by_category_nonfront_s1"][cat] = {
            "n": n, "agreement": float((bins["pred5"].to_numpy() == bins["ref5_s1"].to_numpy())[sel].mean()) if n else None}
    head = out["system1_path"]["nonfront"]
    n, a, g = head["n"], head["agreement"]["value"], head["gain_over_front"]["value"]
    fields = {"n_nonfront_bins": n, "agreement_nonfront": a, "constant_front_nonfront": head["constant_front"]["value"],
              "gain": g, "criterion": CRITERIA["H2"], "thresholds": H2}
    decision = h2_decision(n, a, g)
    out["verdict"] = (_verdict(decision, reason=f"n = {n} < {H2['n_min']}", **fields) if decision == "not_measured"
                      else _verdict(decision, **fields))
    return out


def h3_block(rows: list, reps: int, seed: int) -> dict:
    ready = [r for r in rows if r["ready"]]
    have = [r for r in ready if r["cf_changed"] is not None]
    lacking = [[r["ep_key"], r["call_index"]] for r in ready if r["cf_changed"] is None]
    out = {"n_ready_calls": len(ready), "n_with_counterfactual": len(have), "n_missing_counterfactual": len(lacking),
           "missing_counterfactual_calls": lacking}
    fields = {"n": len(have), "n_ready_calls": len(ready), "n_missing_counterfactual": len(lacking),
              "missing_counterfactual_calls": lacking, "criterion": CRITERIA["H3"], "thresholds": H3}
    if not have:
        out["verdict"] = _verdict("missing", reason="no counterfactual on any ready call", change_rate=None, **fields)
        return out
    boot = SceneBootstrap(np.asarray([r["ep_key"] for r in have]), reps,
                          np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,))))
    changed = np.asarray([bool(r["cf_changed"]) for r in have])
    p, rr = boot.ratio(changed, np.ones(len(have), bool))
    out["change_rate"] = stat(p, rr)
    out["n_changed"] = int(changed.sum())
    shift = np.asarray([r["endpoint_shift_m"] for r in have if r["endpoint_shift_m"] is not None], dtype=np.float64)
    out["endpoint_shift_m"] = {"n": int(shift.size), "median": float(np.median(shift)) if shift.size else None,
                               "p90": float(np.percentile(shift, 90)) if shift.size else None,
                               "max": float(shift.max()) if shift.size else None}
    rep = [r for r in ready if r["replay_actions_equal"] is not None]
    out["replay_noise_floor"] = {
        "meaning": "same plan_z, same noise, same post-processing ([B] replay_same_plan): an action change here is "
                   "device non-determinism, the floor under the change rate (descriptive, not in the criterion)",
        "n": len(rep), "n_changed": sum(not r["replay_actions_equal"] for r in rep),
        "rate": sum(not r["replay_actions_equal"] for r in rep) / len(rep) if rep else None}
    out["by_category"] = {cat: {"n": sum(r["category"] == cat for r in have),
                                "changed": sum(bool(r["cf_changed"]) for r in have if r["category"] == cat)}
                          for cat in CATEGORY_ORDER}
    decision = h3_decision(p, len(lacking))
    if decision == "missing":  # the subset rate stays descriptive, under a name no caption reads
        out["verdict"] = _verdict(decision, reason=f"{len(lacking)} of {len(ready)} ready calls have no counterfactual; "
                                  "the change rate is defined over every ready call",
                                  change_rate=None, change_rate_on_subset=p, verdict_on_subset=h3_decision(p), **fields)
    else:
        out["verdict"] = _verdict(decision, change_rate=p, **fields)
    return out


def final_verdicts(blocks: dict, reasons: list, sc: dict | None) -> dict:
    """metrics.json["verdicts"]: each block's verdict, "void" (computed one kept as verdict_if_valid) when a
    validity condition failed; H1 also when its numpy scoring disagreed with validate.py's accumulator."""
    verdicts = {}
    for name, block in blocks.items():
        vd = dict(block["verdict"])
        vd.setdefault("criterion", CRITERIA[name])
        void = list(reasons)
        if name == "H1" and sc is not None and not sc["ok"]:
            void.append("self-check failed: numpy H1 scoring differs from validate.py's _HeatmapJointMetricAccumulator")
        if void:
            vd.update(verdict_if_valid=vd["verdict"], verdict="void", label_zh=VERDICT_ZH["void"], void_reasons=void)
        verdicts[name] = vd
    return verdicts


def self_check(accs: dict, slots: pd.DataFrame) -> dict:
    report = {"ok": True, "checks": []}
    for arm in ("model", "floor"):
        ref = accs[arm].compute()
        mine = validator_metrics(slots, arm)
        mism = {k: [mine.get(k), ref[k]] for k in ref if mine.get(k) != ref[k]}
        report["ok"] &= not mism
        report["checks"].append({"arm": arm, "validator_joint_pck8": ref["val_heatmap_joint_pck8"],
                                 "numpy_joint_pck8": mine["val_heatmap_joint_pck8"], "bit_exact": not mism,
                                 "mismatches": mism})
    return report


# --------------------------------------------------------------------------- #
# Figure bundle
# --------------------------------------------------------------------------- #
def topdown_info(scene_id: str, route_y: np.ndarray, root: Path) -> dict:
    info = {"root": str(root), "scene": scene_id, "level_index": None, "levels_visited": []}
    try:
        from scripts.exp18.topdown import topdown_io
        td = topdown_io.load_topdown(scene_id, root)
    except Exception as exc:  # noqa: BLE001 - no map for this scene: the bundle says so, [G] decides
        info["note"] = f"{type(exc).__name__}: {exc}"
        return info
    levels = np.atleast_1d(td.level_index(route_y)) if len(route_y) else np.zeros(0, np.int64)
    if levels.size:
        counts = Counter(int(v) for v in levels)
        info["level_index"] = max(counts, key=lambda k: (counts[k], -k))  # the level most steps are on
        info["levels_visited"] = sorted(counts)
        info["route_level_index"] = [int(v) for v in levels]  # per route_xz entry (cross-floor episodes)
    return info


def build_bundle(ep: Episode, res: dict, memberships: list, main_flags: dict, conv: dict, topdown_root: Path):
    """(bundle json, bundle npz arrays) for one episode."""
    rows = {r["call_index"]: r for r in res["rows"]}
    calls = {int(c["system2_call_index"]): c for c in ep.trace["calls"]}
    steps_sorted = sorted(ep.steps["states"])
    route = np.asarray([ep.position(s) for s in steps_sorted], dtype=np.float64).reshape(-1, 3)
    ref_path = np.asarray(ep.start.get("reference_path") or [], dtype=np.float64).reshape(-1, 3)
    primary = memberships[0]
    outcome = res["outcome"]
    arrays, keys = {}, []
    for i, k in enumerate(res["key_steps"]):
        c, row = calls[k["call_index"]], rows[k["call_index"]]
        step = row["step"]
        cam = ep.cam(step)
        resp = c.get("response") or {}
        decision = "lookdown" if int(resp.get("native_lookdown_turns") or 0) >= 1 else "front"
        with np.load(c["_npz"], allow_pickle=False) as z:
            dec = decode_jpeg(z["jpeg__lookdown" if decision == "lookdown" else "jpeg__current__front"])
            hist_rgb = [decode_jpeg(z[f"jpeg__history__{j}__front"]) for j in range(row["n_history"])]
            gated = npz_array(z, "hist_heatmaps_gated").astype(np.float32)
            none_p = npz_array(z, "hist_none_probability").astype(np.float32)
            mask = npz_array(z, "hist_mask").astype(bool) if "hist_mask" in z.files else np.arange(8) < row["n_history"]
            fut = npz_array(z, "fut_heatmaps_gated").astype(np.float32)
            fut_vis = npz_array(z, "fut_visibility_probability").astype(np.float32)
            path = selected_path(c, z)  # never None: _ready_call stopped the build otherwise
        pano, depth, _ = ep.render(step)
        gmap, gvis = gt.history_labels([ep.cam(s) for s in row["history_steps"]], cam, depth)
        gmap, gvis, _ = gt.pad_slots(gmap, gvis)
        cls, grow, gcol = gt.gt_view_class_and_peak(gmap, gvis)
        peak = np.stack([cls - 1, grow, gcol], axis=-1).astype(np.float32)
        peak[cls == 0] = -1
        front_jpg = ep.steps_dir / str(ep.state(step).get("front_jpg") or f"front_{step:04d}.jpg")
        front = decode_jpeg(np.frombuffer(front_jpg.read_bytes(), np.uint8))  # [C] writes one per state
        wh = gt.DECISION_IMAGES[decision]["wh"]
        uv = gt.project_to_decision_image(gt.path_camera_points(path, gt.DECISION_IMAGES[decision]["pitch_deg"]), wh)
        kept = np.nonzero(np.isfinite(uv[:, 0]))[0]
        pg = resp.get("pixel_goal")
        p = f"k{i}_"
        arrays.update({
            p + "decision_rgb": dec, p + "front_native": front,
            p + "history_rgb": np.stack(hist_rgb) if hist_rgb else np.zeros((0,) + gt.VLM_IMAGE_WH[::-1] + (3,), np.uint8),
            p + "pano_rgb": np.asarray(pano, dtype=np.uint8), p + "hist_pred": gated, p + "hist_none": none_p,
            p + "hist_mask": mask, p + "hist_gt": gmap, p + "hist_gt_vis": gvis, p + "hist_gt_peak": peak,
            p + "fut_pred": fut, p + "fut_vis": fut_vis,
            p + "path_cam": gt.path_camera_points(path).astype(np.float32),
            p + "path_xz_world": gt.path_world_xz(path, cam).astype(np.float32),
        })
        keys.append({
            "label": k["label"], "rule": k["rule"], "branch": k["branch"], "ready_index": k["ready_index"],
            "call_index": k["call_index"], "step": step, "npz_prefix": p,
            "position_xz": ep.position(step)[[0, 2]].tolist(), "dist_to_goal_m": row["dist_to_goal_m"],
            "decision_image": decision, "decision_image_wh": list(wh),
            "system2_first_output": resp.get("native_first_output"), "system2_output": resp.get("llm_output"),
            "pixel_goal_raw": pg,
            "pixel_goal_uv": list(gt.pixel_goal_uv(pg, conv["convention"])) if pg is not None else None,
            "path_uv": uv[kept].round(2).tolist(), "path_uv_index": kept.tolist(),
            "executed_actions": row["executed_actions"], "response_actions": row["response_actions"],
            "cf_actions": row["cf_actions"], "cf_changed": row["cf_changed"],
            "history_steps": row["history_steps"], "history_count": row["n_history"],
            "h1_call": {"n_valid": row.get("h1_n_valid"), "n_visible": row.get("h1_n_visible"), "pck8": row.get("h1_pck8"),
                        "floor_joint8": row.get("h1_floor_joint8")},
            "h2_call": {"pred_view5": row.get("h2_pred5"), "ref_view5_system1": row.get("h2_ref5_s1"),
                        "ref_view5_executed": row.get("h2_ref5_exec"), "pred_max_prob": row.get("h2_pred_max_prob")},
        })
    fid = res["fidelity"] or {}
    bundle = {
        "schema": SCHEMA_BUNDLE, "scene_id": ep.scene_id, "episode_id": ep.episode_id, "ep_key": ep.ep_key,
        "category": primary["category"], "category_rank": primary["rank"],
        "is_main": bool(main_flags.get(primary["category"]) == ep.ep_key),
        "predicate_holds_on_rerun": rerun_predicate(primary["category"], outcome),
        "memberships": [{"category": m["category"], "rank": m["rank"],
                         "is_main": bool(main_flags.get(m["category"]) == ep.ep_key),
                         "predicate_holds_on_rerun": rerun_predicate(m["category"], outcome)} for m in memberships],
        "instruction": ep.start.get("instruction"),
        "outcome": ({"success": outcome["success"] >= 0.5, "oracle_success": outcome["oracle_success"] >= 0.5,
                     "ne_m": outcome["ne_m"], "steps": outcome["steps"], "ended_by": outcome["ended_by"]}
                    if outcome else None),
        "eval_log_outcome": res.get("eval_log_final"),
        "fidelity": {"first_divergent_call": fid.get("first_divergent_call"), "identical_calls": fid.get("identical_calls"),
                     "total_calls": fid.get("total_calls"), "reference_calls": fid.get("reference_calls")},
        "topdown": topdown_info(ep.scene_id, route[:, 1], topdown_root),
        "route_steps": steps_sorted, "route_xz": route[:, [0, 2]].tolist(), "route_y": route[:, 1].tolist(),
        "reference_path_xz": ref_path[:, [0, 2]].tolist(),
        "start_xz": np.asarray(ep.start["start_position"], dtype=np.float64)[[0, 2]].tolist(),
        "goal_xz": ep.goal[[0, 2]].tolist(), "goal_radius_m": float(ep.start.get("goal_radius", 3.0)),
        "key_steps": keys,
        "conventions": {
            "view_order": list(geo.VIEW_NAMES),
            "image_uv": "pixel index coordinates: pixel (col i, row j) is centred at (i, j) (imshow default); u = column",
            "pixel_goal": {"convention": conv["convention"], "forced": conv.get("forced", False),
                           "meaning": gt.PIXEL_GOAL_CONVENTIONS[conv["convention"]]},
            "hist_pred": "history head heatmaps_gated [8 slots, 4 views, 64, 64]; hist_none = none_probability",
            "hist_gt_peak": "(view, row, col) of the GT slot in validate.py semantics; -1 when not visible",
            "fut_pred": "future head future_heatmaps_gated [4 bins (waypoints 1-8, 9-16, 17-24, 25-32), 4 views, 64, 64]",
            "path_cam": "System1 selected mean path, current front camera coords (x right, y up, -z forward), floor",
            "decision_rgb": "lookdown 640x480 or the 384x384 front, decoded from the bytes the server received",
        },
    }
    return bundle, arrays


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _fmt(x, pct=False, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def _fmt_stat(s, pct=True, nd=1):
    if not s or s.get("value") is None:
        return "—"
    lo, hi = s["ci95"]
    return f"{_fmt(s['value'], pct, nd)}" + (f" [{_fmt(lo, pct, nd)}, {_fmt(hi, pct, nd)}]" if lo is not None else "")


def summary_md(m: dict) -> str:
    L = ["# EXP-19 metrics summary", "",
         f"Generated {m['created_utc']} by `scripts/exp19/build_records.py` from run `{m['run']}` "
         f"(bootstrap {m['bootstrap']['reps']} reps, seed {m['bootstrap']['seed']}, episode clusters). "
         "Percentages ×100; brackets are 95% CIs.", ""]
    v = m["validity"]
    L += ["## Validity", "", f"- valid: **{v['valid']}**" + (f" — {'; '.join(v['reasons'])}" if v["reasons"] else ""),
          f"- code equivalence (call 0 identical to the seed-42 log): **{m['gates']['code_equivalence']['pass']}** "
          f"({m['gates']['code_equivalence']['n_pass']}/{m['gates']['code_equivalence']['n_episodes']})",
          f"- trace neutrality (actions_match on trajectory calls): **{m['gates']['trace_neutrality']['pass']}** "
          f"({m['gates']['trace_neutrality']['n_match']}/{m['gates']['trace_neutrality']['n_trajectory_calls']})",
          f"- episodes present: {len(m['coverage']['present'])}/{len(m['coverage']['expected'])}"
          + (f", missing {m['coverage']['missing']}" if m["coverage"]["missing"] else ""), ""]
    L += ["## Verdicts", ""]
    for h in ("H1", "H2", "H3"):
        vd = m["verdicts"][h]
        extra = (f" (if valid: {vd['verdict_if_valid']}; {'; '.join(vd['void_reasons'])})" if vd["verdict"] == "void"
                 else f" — {vd['reason']}" if vd.get("reason") else "")
        L.append(f"- **{h}**: {vd['label_zh']} / {vd['verdict']}{extra}")
    h1, h2, h3 = m["H1"], m["H2"], m["H3"]
    L += ["", "## H1 history affordance map vs GT (joint PCK@8, ready calls)", ""]
    if h1.get("n_visible_slots"):
        L += ["| GT-visible slots | ready calls | episodes | model PCK@8 | floor PCK@8 | gain (pt) | model PCK@4 |",
              "|---|---|---|---|---|---|---|",
              f"| {h1['n_visible_slots']} | {h1['n_ready_calls']} | {h1['n_episodes']} | {_fmt_stat(h1['model']['joint_pck8'])} | "
              f"{_fmt_stat(h1['floor']['joint_pck8'])} | {_fmt_stat(h1['gain_over_floor_pck8'])} | {_fmt_stat(h1['model']['joint_pck4'])} |",
              "", "| stratum | cell | n | model PCK@8 | floor PCK@8 |", "|---|---|---|---|---|"]
        for fam, cells in h1["strata"].items():
            for name, c in cells.items():
                L.append(f"| {fam} | {name} | {c['n']} | {_fmt(c['model_pck8'], True)} | {_fmt(c['floor_pck8'], True)} |")
    else:
        L.append("no scored slots")
    L += ["", "## H2 future affordance map view vs System1 path (per time bin)", ""]
    if h2.get("n_bins"):
        L += ["| reference | bins | agreement | constant front | gain (pt) |", "|---|---|---|---|---|"]
        for ref in ("system1_path", "executed_path"):
            for part in ("nonfront", "all_visible"):
                b = h2[ref][part]
                L.append(f"| {ref} / {part} | {b['n']} | {_fmt_stat(b['agreement'])} | {_fmt_stat(b['constant_front'])} | "
                         f"{_fmt_stat(b['gain_over_front'])} |")
        L += ["", f"predicted view5 counts: {h2['pred_view5_counts']}; System1 reference counts: "
                  f"{h2['system1_path']['reference_view5_counts']}"]
    else:
        L.append("no ready calls")
    L += ["", "## H3 counterfactual action chunk (Z instead of Z~)", ""]
    if h3.get("change_rate"):
        es = h3["endpoint_shift_m"]
        L += [f"change rate {_fmt_stat(h3['change_rate'])} ({h3['n_changed']}/{h3['n_with_counterfactual']} ready calls; "
              f"{h3['n_missing_counterfactual']} without counterfactual); endpoint shift median {_fmt(es['median'], nd=3)} m, "
              f"P90 {_fmt(es['p90'], nd=3)} m; replay with the deployed plan (device-noise floor) changed "
              f"{h3['replay_noise_floor']['n_changed']}/{h3['replay_noise_floor']['n']}"]
    else:
        L.append("no counterfactuals")
    pc = m["pixel_goal_convention"]
    L += ["", "## Pixel-goal convention (resolved from the traces)", "",
          f"chosen: **{pc.get('convention')}**{' (FORCED)' if pc.get('forced') else ''} — "
          f"{gt.PIXEL_GOAL_CONVENTIONS.get(pc.get('convention'), '')}; informative calls {pc.get('n_informative')}/"
          f"{pc.get('n_calls')} ({pc.get('n_left')} left, {pc.get('n_right')} right)"]
    for name, c in (pc.get("conventions") or {}).items():
        L.append(f"- {name}: sign correlation {_fmt(c['sign_correlation'], nd=3)}, balanced sign agreement "
                 f"{_fmt(c['balanced_sign_agreement'], nd=3)}, range violations {c['range_violations']} "
                 f"(plain agreement {_fmt(c['sign_agreement'], nd=3)}, pearson vs lateral "
                 f"{_fmt(c['pearson_left_offset_vs_lateral'], nd=3)})")
    L += ["", "## Episodes", "",
          "| ep_key | category (rank) | main | predicate on rerun | success / os / steps / ended | calls (ready) | "
          "first divergent call | identical calls | outcome = log | steps = log | call 0 identical | key steps (call@step) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in m["episodes"]:
        o = e["outcome"] or {}
        L.append(f"| {e['ep_key']} | {e['category']} ({e['category_rank']}) | {e['is_main']} | {e['predicate_holds_on_rerun']} | "
                 f"{o.get('success')} / {o.get('oracle_success')} / {o.get('steps')} / {o.get('ended_by')} | "
                 f"{e['n_calls']} ({e['n_ready']}) | {e['first_divergent_call']} | {e['identical_calls']} | "
                 f"{e['outcome_same_as_eval_log']} | {e['steps_same_as_eval_log']} | {e['call0_identical']} | "
                 + ", ".join(f"{k['label']} {k['call_index']}@{k['step']}" for k in e["key_steps"]) + " |")
    L += ["", f"Main cases: {m['main_cases']}", ""]
    if m.get("self_check"):
        L += ["## Self-check vs validate.py", "", f"bit-exact: **{m['self_check']['ok']}**", ""]
        for c in m["self_check"]["checks"]:
            L.append(f"- {c['arm']}: validator {c['validator_joint_pck8']!r}, numpy {c['numpy_joint_pck8']!r}")
    if m["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in m["warnings"]]
    return "\n".join(L) + "\n"


def git_sha() -> str | None:
    marker = SOURCE_ROOT / ".exp19_git_sha"
    if marker.is_file():
        return marker.read_text().strip()
    try:
        return subprocess.run(["git", "-c", f"safe.directory={SOURCE_ROOT}", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp-root", type=Path, default=EXP_ROOT)
    p.add_argument("--run", default=env("EXP19_RUN"), help="runs/<run> to read (required)")
    p.add_argument("--candidates", type=Path, default=None, help="default <exp-root>/cases/candidates.json")
    p.add_argument("--eval-log-ref-dir", type=Path, default=None, help="default <exp-root>/cases/eval_log_reference")
    p.add_argument("--renders-dir", type=Path, default=None, help="default <exp-root>/renders")
    p.add_argument("--records-dir", type=Path, default=None, help="default <exp-root>/records")
    p.add_argument("--metrics-dir", type=Path, default=None, help="default <exp-root>/metrics")
    p.add_argument("--topdown-root", type=Path, default=TOPDOWN_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=int(env("EXP19_BOOTSTRAP_REPS", "10000")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pixel-goal-convention", choices=("auto", "field_uv", "field_vu"), default="auto",
                   help="auto = resolve from the traces and stop if ambiguous; a forced value is recorded as forced")
    p.add_argument("--self-check", action=argparse.BooleanOptionalAction, default=True,
                   help="also score H1 with validate.py's accumulator (imports torch + the training stack)")
    p.add_argument("--allow-invalid", action="store_true", help="exit 0 even when a validity gate fails")
    args = p.parse_args(argv)
    if not args.run:
        p.error("--run (or EXP19_RUN) is required")
    root = args.exp_root
    args.candidates = args.candidates or root / "cases" / "candidates.json"
    args.eval_log_ref_dir = args.eval_log_ref_dir or root / "cases" / "eval_log_reference"
    args.renders_dir = args.renders_dir or root / "renders"
    args.records_dir = args.records_dir or root / "records"
    args.metrics_dir = args.metrics_dir or root / "metrics"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    cands = load_candidates(args.candidates)
    run_dir = args.exp_root / "runs" / args.run
    found = discover_run(run_dir)
    expected = sorted(cands["members"])
    present = [k for k in expected if k in found]
    not_cands = sorted(set(found) - set(cands["members"]))
    warnings = [f"{k}: in the run but not a candidate, skipped" for k in not_cands]
    if not (run_dir / "DONE").is_file():
        warnings.append(f"{run_dir / 'DONE'} missing: the run may still be going or have died")
    log(f"run {run_dir}: {len(found)} episodes, {len(present)}/{len(expected)} candidates present")

    accs = None
    if args.self_check:
        import torch

        from scripts.training.validate import _HeatmapJointMetricAccumulator as Acc
        accs = {arm: Acc(heatmap_size=(64, 64), device=torch.device("cpu")) for arm in ("model", "floor")}

    episodes, results = {}, {}
    for ep_key in present:
        members = cands["members"][ep_key]
        ep = Episode(ep_key, found[ep_key], args.renders_dir)
        ref_path = args.eval_log_ref_dir / f"{ep_key}.json"
        eval_ref = read_json(ref_path) if ref_path.is_file() else None
        if eval_ref is not None and eval_ref.get("schema") != SCHEMA_EVAL_REF:
            raise ValueError(f"{ref_path}: schema {eval_ref.get('schema')!r}")
        res = analyse_episode(ep, members[0]["category"], eval_ref, accs)
        res["eval_log_final"] = (eval_ref or {}).get("final")
        if eval_ref is None:
            res["warnings"].append("no eval-log reference")
        episodes[ep_key], results[ep_key] = ep, res
        warnings += [f"{ep_key}: {w}" for w in res["warnings"]]
        log(f"{ep_key}: {len(res['rows'])} calls, {sum(r['ready'] for r in res['rows'])} ready, "
            f"key steps {[(k['label'], k['call_index']) for k in res['key_steps']]}")

    rows = [r for res in results.values() for r in res["rows"]]
    slots = pd.DataFrame([s for res in results.values() for s in res["slot_rows"]])
    bins = pd.DataFrame([b for res in results.values() for b in res["bin_rows"]])

    # ---- pixel-goal convention from every ready call --------------------------
    samples = [{"pixel_goal": r["pixel_goal_field"], "lateral_m": r["selected_path_end_xy"][1],
                "image_wh": gt.DECISION_IMAGES["lookdown" if int(r["lookdown_turns"] or 0) >= 1 else "front"]["wh"]}
               for r in rows if r["ready"] and r["pixel_goal_field"] is not None and r["selected_path_end_xy"] is not None]
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.pixel_goal_convention == "auto":
        try:
            conv = gt.resolve_pixel_goal_convention(samples)
        except gt.PixelGoalConventionError as exc:
            write_json(args.metrics_dir / "pixel_goal_convention_evidence.json", exc.evidence)
            raise SystemExit(f"STOP: {exc}. Evidence in {args.metrics_dir / 'pixel_goal_convention_evidence.json'}; "
                             "rerun with --pixel-goal-convention only after reading it.")
    else:
        conv = gt.pixel_goal_evidence(samples)
        auto = None
        try:
            auto = gt.resolve_pixel_goal_convention(samples)["convention"]
        except gt.PixelGoalConventionError as exc:
            conv["ambiguous_reasons"] = exc.evidence["ambiguous_reasons"]
        conv.update(convention=args.pixel_goal_convention, forced=True, auto_resolution=auto)
        warnings.append(f"pixel-goal convention FORCED to {args.pixel_goal_convention} (auto: {auto})")
    log(f"pixel-goal convention: {conv['convention']} (forced={conv.get('forced')}) "
        + ", ".join(f"{k}: sign corr {c['sign_correlation']}, balanced agreement {c['balanced_sign_agreement']}, "
                    f"out-of-image {c['range_violations']}" for k, c in conv["conventions"].items()))

    # ---- gates, predicates, main cases ----------------------------------------
    ce = {k: results[k]["code_equivalence"] for k in present}
    code_gate = {"n_episodes": len(present), "n_pass": sum(bool(v and v["pass"]) for v in ce.values()),
                 "failures": {k: (v or "no eval-log reference") for k, v in ce.items() if not (v and v["pass"])}}
    code_gate["pass"] = bool(present) and code_gate["n_pass"] == len(present)
    traj = [r for r in rows if r["kind"] == "trajectory"]
    neutral = {"n_trajectory_calls": len(traj), "n_match": sum(r["actions_match"] is True for r in traj),
               "failures": [(r["ep_key"], r["call_index"], r["actions_match"]) for r in traj if r["actions_match"] is not True],
               "recheck_disagreements": [(r["ep_key"], r["call_index"]) for r in traj
                                         if r["actions_match_recheck"] is not None and r["actions_match_recheck"] != r["actions_match"]],
               # "逐调用" = every call the client made: a call without a trace cannot be shown to match
               "calls_without_trace": {k: episodes[k].missing_call_indices for k in present
                                       if episodes[k].missing_call_indices}}
    neutral["pass"] = (bool(traj) and neutral["n_match"] == len(traj) and not neutral["recheck_disagreements"]
                       and not neutral["calls_without_trace"])
    missing = [k for k in expected if k not in found]
    reasons = []
    if not code_gate["pass"]:
        reasons.append(f"code-equivalence gate failed ({code_gate['n_pass']}/{code_gate['n_episodes']})")
    if not neutral["pass"]:
        reasons.append(f"trace-neutrality gate failed ({neutral['n_match']}/{neutral['n_trajectory_calls']} trajectory "
                       f"calls match, calls without a trace: {neutral['calls_without_trace'] or 'none'})")
    if missing:
        reasons.append(f"{len(missing)} candidate episodes not in the run: {missing}")
    incomplete = [k for k in present if results[k]["outcome"] is None]
    if incomplete:
        reasons.append(f"{len(incomplete)} episodes without episode_end (unfinished): {incomplete}")

    main_cases = {}
    for cat in CATEGORY_ORDER:
        main_cases[cat] = None
        for ep_key in cands["ordered"][cat]:
            if ep_key not in results:
                warnings.append(f"main case of {cat} undetermined: rank of {ep_key} not rerun")
                break
            if rerun_predicate(cat, results[ep_key]["outcome"]):
                main_cases[cat] = ep_key
                break

    # ---- metrics -----------------------------------------------------------------
    h1 = h1_block(slots, args.bootstrap_reps, args.seed)
    h2 = h2_block(bins, args.bootstrap_reps, args.seed)
    h3 = h3_block(rows, args.bootstrap_reps, args.seed)
    if h3.get("n_missing_counterfactual"):
        warnings.append(f"H3 missing: {h3['n_missing_counterfactual']} ready calls without a counterfactual "
                        f"{h3['missing_counterfactual_calls']}")
    sc = self_check(accs, slots) if accs is not None and len(slots) else None
    verdicts = final_verdicts({"H1": h1, "H2": h2, "H3": h3}, reasons, sc)

    ep_summ = []
    for ep_key in present:
        res, prim = results[ep_key], cands["members"][ep_key][0]
        fid = res["fidelity"] or {}
        ep_summ.append({"ep_key": ep_key, "gpu": episodes[ep_key].gpu, "category": prim["category"],
                        "category_rank": prim["rank"], "is_main": main_cases.get(prim["category"]) == ep_key,
                        "predicate_holds_on_rerun": rerun_predicate(prim["category"], res["outcome"]),
                        "outcome": res["outcome"], "eval_log_final": res["eval_log_final"],
                        "n_calls": len(res["rows"]), "n_ready": sum(r["ready"] for r in res["rows"]),
                        "first_divergent_call": fid.get("first_divergent_call"), "identical_calls": fid.get("identical_calls"),
                        "all_calls_identical": fid.get("all_identical"),
                        "outcome_same_as_eval_log": res["outcome_same_as_eval_log"],
                        "steps_same_as_eval_log": (res["outcome_comparison"] or {}).get("steps_equal"),
                        "call0_identical": (res["code_equivalence"] or {}).get("pass"),
                        "key_steps": res["key_steps"],
                        "checks": res["checks"]})
    metrics = {
        "schema": SCHEMA_METRICS, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "git_sha": git_sha(),
        "run": args.run, "inputs": {"exp_root": str(args.exp_root), "run_dir": str(run_dir), "candidates": str(args.candidates),
                                    "eval_log_reference": str(args.eval_log_ref_dir), "renders": str(args.renders_dir)},
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.seed, "cluster": "episode (ep_key)",
                      "streams": "H1 default_rng(seed); H2 SeedSequence(seed, spawn_key=(2,)); H3 spawn_key=(3,)"},
        "definitions": DEFINITIONS, "criteria": CRITERIA,
        "coverage": {"expected": expected, "present": present, "missing": missing, "unfinished": incomplete,
                     "not_candidates": not_cands},
        "gates": {"code_equivalence": code_gate, "trace_neutrality": neutral},
        "validity": {"valid": not reasons, "reasons": reasons},
        "pixel_goal_convention": conv, "H1": h1, "H2": h2, "H3": h3, "verdicts": verdicts,
        "fidelity": {"n_episodes": len(present),
                     "n_all_calls_identical": sum(bool(e["all_calls_identical"]) for e in ep_summ),
                     "n_outcome_same": sum(bool(e["outcome_same_as_eval_log"]) for e in ep_summ),
                     "n_steps_same": sum(bool(e["steps_same_as_eval_log"]) for e in ep_summ),
                     "outcome_same_means": "success, oracle success and ended_by (stop / step_cap; when the reference "
                                           "records it) all equal; the step count is compared separately (descriptive)"},
        "main_cases": main_cases, "episodes": ep_summ, "self_check": sc, "warnings": warnings,
    }

    # ---- write -------------------------------------------------------------------
    args.records_dir.mkdir(parents=True, exist_ok=True)
    for ep_key in present:
        ep, res = episodes[ep_key], results[ep_key]
        bundle, arrays = build_bundle(ep, res, cands["members"][ep_key], main_cases, conv, args.topdown_root)
        write_json(args.records_dir / f"{ep_key}_bundle.json", bundle)
        np.savez_compressed(args.records_dir / f"{ep_key}_bundle.npz", **arrays)
        write_json(args.records_dir / f"{ep_key}.json", {
            "schema": SCHEMA_RECORD, "ep_key": ep_key, "scene_id": ep.scene_id, "episode_id": ep.episode_id,
            "gpu": ep.gpu, "memberships": cands["members"][ep_key],
            "outcome": res["outcome"], "eval_log_final": res["eval_log_final"], "fidelity": res["fidelity"],
            "code_equivalence": res["code_equivalence"], "outcome_same_as_eval_log": res["outcome_same_as_eval_log"],
            "outcome_comparison": res["outcome_comparison"], "closest_approach_step": res["closest_approach_step"], "key_steps": res["key_steps"],
            "checks": res["checks"], "trace_missing_call_indices": ep.missing_call_indices,
            "trace_errors": ep.trace["trace_errors"], "warnings": res["warnings"], "calls": res["rows"]})
    with open(args.metrics_dir / "calls.jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(_clean(r), ensure_ascii=False, default=_json_default) + "\n")
    write_json(args.metrics_dir / "metrics.json", metrics)
    (args.metrics_dir / "summary.md").write_text(summary_md(metrics), encoding="utf-8")
    log(f"wrote {args.metrics_dir}/metrics.json, summary.md, calls.jsonl and {len(present)} records + bundles "
        f"in {args.records_dir}")
    log("verdicts: " + ", ".join(f"{h} {v['verdict']}" + (f" (if valid: {v['verdict_if_valid']})" if "verdict_if_valid" in v else "")
                                 for h, v in verdicts.items()))
    if sc is not None and not sc["ok"]:
        log("SELF-CHECK FAILED: numpy H1 differs from validate.py's accumulator")
        return 4
    if reasons:
        log("INVALID: " + "; ".join(reasons))
        return 0 if args.allow_invalid else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
