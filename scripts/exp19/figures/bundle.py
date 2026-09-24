"""EXP-19 figure bundle (schema ``exp19-figure-bundle-v1``): validation, loading, and what the page draws.

``scripts/exp19/build_records.py`` ([F]) writes one bundle per episode,
``records/<ep_key>_bundle.json`` + ``records/<ep_key>_bundle.npz``.  The figure
code reads nothing else except the EXP-18 top-down maps.  This module is
numpy only, so the schema check runs where matplotlib is missing.

Conventions (they come from the contract and are checked here, not re-derived):

* World: Habitat, y up.  The top-down map plots world (x, z) directly.
* Affordance maps ``[..., 4, 64, 64]`` use view order front, right, back, left
  (yaw 0 / -90 / 180 / +90) and the label pixel convention of
  ``scripts/exp18/geometry.py``: index i sits at coordinate i, and fx = cx = 32.
* ``pixel_goal_uv`` and ``path_uv`` are (column, row) pixels in the decision
  image exactly as stored in ``k{i}_decision_rgb``.
* ``path_cam`` is the System1 mean path in the current front-camera frame
  (x right, y up, camera looks along -z), at floor height (y = -1.25 m).
* Actions: 0 STOP, 1 forward 0.25 m, 2 turn left 15 deg, 3 turn right 15 deg.
* NPZ keys of key step i are ``k{i}_<name>``, where i is the 0-based position in
  ``key_steps``.  A 1-based bundle is also accepted, with a warning.

``validate_bundle`` returns schema errors (a page cannot be drawn from such a
bundle).  ``bundle_warnings`` returns what [F] may legitimately leave open or
what is inconsistent but drawable (no top-down map, no rerun outcome, no
eval-log reference, a pixel goal outside its image); pages are drawn and the
warnings go to the figure manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "exp19-figure-bundle-v1"
CATEGORIES = ("T1", "T2", "T3", "F1", "F2")
NUM_SLOTS = 8  # history slots of the History Head
NUM_BINS = 4  # future time bins: waypoints 1-8, 9-16, 17-24, 25-32 of the System1 path
NUM_VIEWS = 4
HM_SIZE = 64
PATH_POINTS = 33
CAMERA_HEIGHT_M = 1.25
DECISION_IMAGES = ("lookdown", "front")
# Rule branch of a key moment, as scripts/exp19/keysteps.py tags it; the captions describe each drawn moment by it.
# "all_lt4": fewer than 4 ready calls, all drawn as K1.. in call order.  The fallbacks K2_fallback,
# K3_f1_fallback_after and K4_shifted are the rule's own "otherwise" branches.
KEY_BRANCHES = ("all_lt4", "K1_first", "K2_turn", "K2_fallback", "K3_two_thirds", "K3_f1_closest",
                "K3_f1_fallback_after", "K4_last", "K4_shifted")
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (STOP, FORWARD, LEFT, RIGHT)

_NONE = type(None)
_NUM = (int, float)

TOP_FIELDS = {
    "schema": str,
    "scene_id": str,
    "episode_id": int,
    "ep_key": str,
    "category": str,
    "category_rank": int,
    "is_main": bool,
    "predicate_holds_on_rerun": (bool, _NONE),
    "instruction": str,
    "outcome": (dict, _NONE),
    "eval_log_outcome": (dict, _NONE),
    "fidelity": dict,
    "topdown": dict,
    "route_xz": list,
    "route_y": list,
    "reference_path_xz": list,
    "start_xz": list,
    "goal_xz": list,
    "goal_radius_m": _NUM,
    "key_steps": list,
}
OUTCOME_FIELDS = {"success": (bool, int, float), "oracle_success": (bool, int, float), "ne_m": _NUM,
                  "steps": int, "ended_by": str}
FIDELITY_FIELDS = {"first_divergent_call": (int, _NONE), "identical_calls": (int, _NONE), "total_calls": (int, _NONE)}
TOPDOWN_FIELDS = {"root": (str, _NONE), "scene": str, "level_index": (int, _NONE)}  # None: no map for the scene
KEY_FIELDS = {
    "label": str,
    "rule": str,
    "branch": str,
    "call_index": int,
    "step": int,
    "position_xz": list,
    "decision_image": str,
    "system2_first_output": (str, _NONE),
    "system2_output": (str, _NONE),
    "pixel_goal_uv": (list, _NONE),
    "path_uv": list,
    "executed_actions": list,
    "response_actions": (list, _NONE),
    "cf_actions": (list, _NONE),
    "cf_changed": (bool, _NONE),
    "history_steps": list,
    "history_count": int,
    "h1_call": dict,
    "h2_call": dict,
}
# name -> (dtype kind, shape); None = any size, "K" = history_count of the key step
KEY_ARRAYS = {
    "decision_rgb": ("uint8", (None, None, 3)),
    "front_native": ("uint8", (None, None, 3)),
    "history_rgb": ("uint8", ("K", None, None, 3)),
    "pano_rgb": ("uint8", (NUM_VIEWS, None, None, 3)),
    "hist_pred": ("float", (NUM_SLOTS, NUM_VIEWS, HM_SIZE, HM_SIZE)),
    "hist_none": ("float", (NUM_SLOTS,)),
    "hist_mask": ("bool", (NUM_SLOTS,)),
    "hist_gt": ("float", (NUM_SLOTS, NUM_VIEWS, HM_SIZE, HM_SIZE)),
    "hist_gt_vis": ("float", (NUM_SLOTS, NUM_VIEWS)),
    "hist_gt_peak": ("float", (NUM_SLOTS, 3)),
    "fut_pred": ("float", (NUM_BINS, NUM_VIEWS, HM_SIZE, HM_SIZE)),
    "fut_vis": ("float", (NUM_BINS, NUM_VIEWS)),
    "path_cam": ("float", (PATH_POINTS, 3)),
    "path_xz_world": ("float", (PATH_POINTS, 2)),
}


class BundleError(ValueError):
    """A bundle that does not follow ``exp19-figure-bundle-v1`` (message lists every problem)."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _type_ok(value, types) -> bool:
    types = types if isinstance(types, tuple) else (types,)
    if isinstance(value, bool) and bool not in types:
        return False  # bool is an int subclass; an int field must not take True/False
    return isinstance(value, types)


def _check_fields(obj: dict, spec: dict, where: str, errors: List[str]) -> None:
    for name, types in spec.items():
        if name not in obj:
            errors.append(f"{where}: missing field {name!r}")
        elif not _type_ok(obj[name], types):
            errors.append(f"{where}.{name}: {type(obj[name]).__name__} not allowed")


def _xy_list(value, where: str, errors: List[str], n: Optional[int] = None) -> None:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f"{where}: not numeric")
        return
    if a.ndim != 2 or a.shape[1] != 2 or (n is not None and a.shape[0] != n):
        errors.append(f"{where}: shape {a.shape}, expected [{n if n is not None else 'N'}, 2]")
    elif not np.isfinite(a).all():
        errors.append(f"{where}: non-finite values")


def key_prefix_base(meta: dict, files: Sequence[str]) -> int:
    """0 if key step i is stored as ``k{i}_*`` (the contract), 1 for ``k{i+1}_*``, -1 if neither covers them."""
    n = len(meta.get("key_steps") or [])
    names = set(files)
    for base in (0, 1):
        if all(f"k{i + base}_decision_rgb" in names for i in range(n)):
            return base
    return -1


def _check_array(a: np.ndarray, kind: str, shape: tuple, k: int, where: str, errors: List[str]) -> None:
    if kind == "uint8" and a.dtype != np.uint8:
        errors.append(f"{where}: dtype {a.dtype}, expected uint8")
    elif kind == "bool" and a.dtype != np.bool_:
        errors.append(f"{where}: dtype {a.dtype}, expected bool")
    elif kind == "float" and not np.issubdtype(a.dtype, np.floating):
        errors.append(f"{where}: dtype {a.dtype}, expected float")
    want = tuple(k if s == "K" else s for s in shape)
    if a.ndim != len(want) or any(w is not None and w != d for w, d in zip(want, a.shape)):
        errors.append(f"{where}: shape {a.shape}, expected {tuple('*' if w is None else w for w in want)}")
    elif kind == "float" and not np.isfinite(a).all():
        errors.append(f"{where}: non-finite values")


def validate_bundle(meta: dict, arrays: Dict[str, np.ndarray]) -> List[str]:
    """Every schema problem of a bundle (empty list = valid)."""
    errors: List[str] = []
    _check_fields(meta, TOP_FIELDS, "bundle", errors)
    if errors:
        return errors
    if meta["schema"] != SCHEMA:
        errors.append(f"bundle.schema {meta['schema']!r}, expected {SCHEMA!r}")
    if meta["category"] not in CATEGORIES:
        errors.append(f"bundle.category {meta['category']!r} not in {CATEGORIES}")
    expected_key = f"{meta['scene_id']}_{int(meta['episode_id']):04d}"
    if meta["ep_key"] != expected_key:
        errors.append(f"bundle.ep_key {meta['ep_key']!r}, expected {expected_key!r}")
    if meta["outcome"] is not None:
        _check_fields(meta["outcome"], OUTCOME_FIELDS, "outcome", errors)
    _check_fields(meta["fidelity"], FIDELITY_FIELDS, "fidelity", errors)
    _check_fields(meta["topdown"], TOPDOWN_FIELDS, "topdown", errors)
    _xy_list(meta["route_xz"], "route_xz", errors)
    if len(meta["route_y"]) != len(meta["route_xz"]):
        errors.append(f"route_y has {len(meta['route_y'])} entries, route_xz {len(meta['route_xz'])}")
    _xy_list(meta["reference_path_xz"], "reference_path_xz", errors)
    _xy_list([meta["start_xz"]], "start_xz", errors, 1)
    _xy_list([meta["goal_xz"]], "goal_xz", errors, 1)

    keys = meta["key_steps"]
    if len(keys) > 4:  # 0 is legal: a rerun without a ready call has no key moment
        errors.append(f"key_steps: {len(keys)} entries, expected 0..4")
    base = key_prefix_base(meta, arrays.keys())
    if base < 0:
        errors.append("npz: no k{i}_decision_rgb for every key step (0-based i)")
    labels = []
    for i, ks in enumerate(keys):
        where = f"key_steps[{i}]"
        if not isinstance(ks, dict):
            errors.append(f"{where}: not an object")
            continue
        n_before = len(errors)
        _check_fields(ks, KEY_FIELDS, where, errors)
        if len(errors) > n_before:
            continue
        labels.append(ks["label"])
        if ks["decision_image"] not in DECISION_IMAGES:
            errors.append(f"{where}.decision_image {ks['decision_image']!r} not in {DECISION_IMAGES}")
        if ks["branch"] not in KEY_BRANCHES:
            errors.append(f"{where}.branch {ks['branch']!r} not in {KEY_BRANCHES}")
        _xy_list([ks["position_xz"]], f"{where}.position_xz", errors, 1)
        if ks["path_uv"]:
            _xy_list(ks["path_uv"], f"{where}.path_uv", errors)
        for name in ("executed_actions", "response_actions"):
            if any((not _type_ok(a, int)) or a not in ACTIONS for a in ks[name] or []):
                errors.append(f"{where}.{name}: {ks[name]} (actions are 0..3)")
        k = ks["history_count"]
        if not 0 <= k <= NUM_SLOTS or len(ks["history_steps"]) != k:
            errors.append(f"{where}: history_count {k} vs {len(ks['history_steps'])} history_steps")
        if base < 0:
            continue
        for name, (kind, shape) in KEY_ARRAYS.items():
            key = f"k{i + base}_{name}"
            if key not in arrays:
                errors.append(f"npz: missing {key}")
                continue
            _check_array(np.asarray(arrays[key]), kind, shape, k, f"npz {key}", errors)
        pano_key = f"k{i + base}_pano_rgb"
        if pano_key in arrays and np.asarray(arrays[pano_key]).ndim == 4 \
                and np.asarray(arrays[pano_key]).shape[1] != np.asarray(arrays[pano_key]).shape[2]:
            errors.append(f"npz {pano_key}: views must be square (HFOV 90 re-renders)")
        uv = ks["pixel_goal_uv"]
        if uv is not None and not (len(uv) == 2 and all(_type_ok(c, _NUM) for c in uv)):
            errors.append(f"{where}.pixel_goal_uv: {uv!r} is not [u, v]")
    if len(set(labels)) != len(labels):
        errors.append(f"key_steps: duplicate labels {labels}")
    return errors


def goal_inside(uv, image_shape) -> bool:
    """Whether a (column, row) pixel goal lies on the image (pixel centres at integer coordinates)."""
    h, w = image_shape[:2]
    return -0.5 <= float(uv[0]) <= w - 0.5 and -0.5 <= float(uv[1]) <= h - 0.5


def bundle_warnings(meta: dict, arrays: Dict[str, np.ndarray]) -> List[str]:
    """Drawable but noteworthy: open fields [F] may leave, and inconsistencies the page shows as they are.

    Call only on a valid bundle.
    """
    out = []
    base = key_prefix_base(meta, arrays.keys())
    if base == 1:
        out.append("npz key steps are 1-based (contract: 0-based k{i})")
    if meta["topdown"]["level_index"] is None:
        out.append("no top-down map: route drawn on a plain plate" + (f" ({meta['topdown']['note']})"
                                                                     if meta["topdown"].get("note") else ""))
    if meta["outcome"] is None:
        out.append("no rerun outcome (steps.jsonl has no episode_end)")
    if meta["fidelity"]["identical_calls"] is None:
        out.append("no eval-log reference: fidelity sentence omitted")
    if not meta["key_steps"]:
        out.append("no ready call: page shows the route only")
    for i, ks in enumerate(meta["key_steps"]):
        mask = np.asarray(arrays[f"k{i + base}_hist_mask"])
        if int(mask.sum()) != ks["history_count"]:
            out.append(f"{ks['label']}: hist_mask has {int(mask.sum())} slots, history_count {ks['history_count']}")
        uv = ks["pixel_goal_uv"]
        dec = np.asarray(arrays[f"k{i + base}_decision_rgb"])
        if uv is not None and not goal_inside(uv, dec.shape):  # e.g. a swapped (row, col) on the 640x480 look-down
            out.append(f"{ks['label']}: pixel_goal_uv {uv} outside the {dec.shape[1]}x{dec.shape[0]} decision image "
                       "(not drawn)")
        if ks["system2_output"] is None:
            out.append(f"{ks['label']}: no System2 output text")
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class KeyStep:
    """One key moment: the bundle's JSON entry plus its arrays (NPZ prefix stripped)."""

    index: int
    meta: dict
    arrays: Dict[str, np.ndarray]

    def __getattr__(self, name):
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        meta = self.__dict__.get("meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    @property
    def system2_texts(self) -> List[str]:
        """System2's outputs in order: [final] or [first turn, final] when the look-down turn ran."""
        first = (self.meta["system2_first_output"] or "").strip()
        final = (self.meta["system2_output"] or "").strip()
        return [t for t in ([final] if (not first or first == final) else [first, final]) if t]


@dataclass
class Bundle:
    path: Path
    meta: dict
    keys: List[KeyStep]
    warnings: List[str] = field(default_factory=list)

    def __getattr__(self, name):
        meta = self.__dict__.get("meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def xz(self, name: str) -> np.ndarray:
        return np.asarray(self.meta[name], dtype=np.float64).reshape(-1, 2)

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    def membership(self, main: bool = False) -> dict:
        """{category, rank, is_main, predicate_holds_on_rerun} the page is filed under.

        [F] lists every category an episode was picked for (``memberships``, primary
        first); the pre-registered 15 have no overlap, but if one existed the main
        figure uses the membership that made it a main case.
        """
        own = {"category": self.meta["category"], "rank": self.meta["category_rank"], "is_main": self.meta["is_main"],
               "predicate_holds_on_rerun": self.meta["predicate_holds_on_rerun"]}
        if main:
            for m in self.meta.get("memberships") or []:
                if m.get("is_main"):
                    return {**own, **m}
        return own

    @property
    def is_main_any(self) -> bool:
        return is_main_case(self.meta)


def is_main_case(meta: dict) -> bool:
    """Whether the bundle is a main-figure case under its primary category or any other membership."""
    return bool(meta.get("is_main")) or any(m.get("is_main") for m in meta.get("memberships") or [])


def bundle_paths(path) -> Tuple[Path, Path]:
    """(json, npz) of a bundle given either file or the common stem ``records/<ep_key>_bundle``."""
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".json", ".npz") else p
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def load_bundle(path, validate: bool = True) -> Bundle:
    json_path, npz_path = bundle_paths(path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    if validate:
        errors = validate_bundle(meta, arrays)
        if errors:
            raise BundleError(f"{json_path}: " + "; ".join(errors))
    warnings = bundle_warnings(meta, arrays) if validate else []
    base = key_prefix_base(meta, arrays.keys())
    keys = []
    for i, ks in enumerate(meta["key_steps"]):
        prefix = f"k{i + max(base, 0)}_"
        keys.append(KeyStep(i, ks, {n[len(prefix):]: a for n, a in arrays.items() if n.startswith(prefix)}))
    return Bundle(json_path, meta, keys, warnings)


def find_bundles(records_dir) -> List[Path]:
    return sorted(Path(records_dir).glob("*_bundle.json"))


# --------------------------------------------------------------------------- #
# What the strips show (plain arrays; the figure only colours them)
# --------------------------------------------------------------------------- #
def history_pred_composite(pred: np.ndarray, none_p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[4, 64, 64]: max over real slots of (slot map / its own peak) x (1 - P(none)).

    Same recipe as EXP-18 ``data.pred_composite``: each slot counts equally
    whatever its spatial spread, and a slot the head calls "not visible" fades.
    """
    pred = np.asarray(pred, dtype=np.float64)
    peak = pred.reshape(pred.shape[0], -1).max(1)
    visible = 1.0 - np.asarray(none_p, dtype=np.float64)
    norm = pred / np.maximum(peak, 1e-12)[:, None, None, None] * visible[:, None, None, None]
    norm = np.where(np.asarray(mask, dtype=bool)[:, None, None, None], norm, 0.0)
    return np.clip(norm.max(0), 0.0, 1.0)


def gt_history_peaks(peak: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, float, float]]:
    """(slot, view, row, col) of every real slot that is visible in some view in the ground truth."""
    out = []
    for k, (view, row, col) in enumerate(np.asarray(peak, dtype=np.float64)):
        if bool(mask[k]) and view >= 0:
            out.append((k, int(round(view)), float(row), float(col)))
    return out


def pred_history_peaks(pred: np.ndarray, none_p: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, float, float]]:
    """(slot, view, row, col) of the predicted peak of every real slot the head calls visible (P(none) < 0.5)."""
    out = []
    for k in np.nonzero(np.asarray(mask, dtype=bool) & (np.asarray(none_p, dtype=np.float64) < 0.5))[0]:
        view, row, col = np.unravel_index(int(np.argmax(pred[k])), np.asarray(pred[k]).shape)
        out.append((int(k), int(view), float(row), float(col)))
    return out


def future_bin_maps(fut_pred: np.ndarray) -> np.ndarray:
    """[4 bins, 4 views, 64, 64] in [0, 1]: the future head's gated maps as returned (peak = view confidence)."""
    return np.clip(np.asarray(fut_pred, dtype=np.float64), 0.0, 1.0)


def path_directions(path_cam: np.ndarray, camera_height: float = CAMERA_HEIGHT_M, min_dist: float = 0.05):
    """(bearing, elevation) in degrees of the System1 path, the way the future labels place it.

    The future labels put every waypoint at camera height (``action_deltas_to_camera_points``
    with zero relative height), so a flat path sits on the horizon row.  Bearing is
    atan2(left, forward), left-positive.  Points closer than ``min_dist`` (the robot's
    own position) have no direction and are dropped.
    """
    p = np.asarray(path_cam, dtype=np.float64).reshape(-1, 3)
    fwd, left, up = -p[:, 2], -p[:, 0], p[:, 1] + camera_height
    dist = np.hypot(fwd, left)
    keep = dist >= min_dist
    bearing = np.degrees(np.arctan2(left[keep], fwd[keep]))
    elev = np.degrees(np.arctan2(up[keep], dist[keep]))
    return bearing, elev, np.nonzero(keep)[0]


def net_turn_deg(actions: Sequence[int]) -> float:
    """Net turn of an action chunk, left-positive (left +15, right -15)."""
    return 15.0 * sum(1 if a == LEFT else -1 if a == RIGHT else 0 for a in actions)
