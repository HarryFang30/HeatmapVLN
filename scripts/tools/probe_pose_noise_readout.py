#!/usr/bin/env python3
"""EXP-15: does the geometry readout survive pose error of deployment magnitude?

EXP-13-A found the decision information System2 lacks is geometry: an 80-dim
hand-built vector reads the oracle's recovery direction at 0.5227 against
System2's own 0.3633.  That vector is built from ``history_rel_poses``, which in
the sealed DAgger collection are **Habitat ground truth**.  At deployment they
come from AMB3R visual odometry, and EXP-04 measured that domain shift pushing
the heatmap head's pck8 from 0.88 to 0.66.  If the geometry advantage is a
ground-truth artefact, every downstream arm inherits the problem, so this asks
the question before any GPU hour is spent on one.

Everything runs on the features EXP-13-A already cached, on CPU, using the same
fitting recipe as ``fit_recovery_readout`` (imported, not reimplemented) so the
numbers are directly comparable to that experiment's table.

**What this can and cannot conclude.**  Synthetic noise is not VO error: real
odometry error is correlated and drifts, and it corrupts the *images'* apparent
geometry too.  So a collapse here is decisive (it will only be worse in the real
domain) while survival is merely "not refuted" and still requires the AMB3R
backfill that 13-C already demands.

**The sweep is deliberately biased against geometry.**  Perturbing the poses
degrades the ``geometry`` arm, but ``memory`` (``M_t``) and ``system2``
(``traj_hidden``) are replayed from cache and cannot degrade -- recomputing them
would need the frozen head on a GPU.  In the real VO domain ``M_t`` degrades
too, and EXP-03 measured how much it depends on pose (zeroing pose costs
42.08pt).  So: geometry staying ahead is strong evidence; geometry falling
behind is *not* evidence that the memory would be better at deployment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = "heatmapvln-exp15-pose-noise-readout-v1"
# Arms whose features are a function of the poses; the rest are replayed from
# cache and are refit once, as fixed reference lines.
POSE_DEPENDENT = ("geometry", "system2_geometry")


def _load_readout_tool() -> types.ModuleType:
    path = PROJECT_ROOT / "scripts/tools/fit_recovery_readout.py"
    spec = importlib.util.spec_from_file_location("_exp13_fit_recovery_readout", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def perturb_rel_poses(
    rel_poses: np.ndarray,
    *,
    translation_m: float,
    rotation_deg: float,
    ages: np.ndarray | None,
    drift: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add pose error to ``[N, K, 4]`` = (forward_m, left_m, cos yaw, sin yaw).

    Translation gets additive Gaussian metres.  Yaw is perturbed by *rotating*
    the (cos, sin) pair rather than adding noise to each component, so the unit
    norm is preserved exactly: a readout must not be able to detect the
    corruption itself instead of the pose error it stands for.

    ``drift=True`` scales each slot's sigma by ``sqrt(1 + age_steps)``, because
    odometry error accumulates with distance travelled, so an older history slot
    is placed less reliably than a recent one.  ``drift=False`` applies the same
    sigma everywhere.
    """
    if rel_poses.ndim != 3 or rel_poses.shape[-1] != 4:
        raise ValueError(f"rel_poses must be [N,K,4], got {rel_poses.shape}")
    out = np.array(rel_poses, dtype=np.float32, copy=True)
    n, k, _ = out.shape

    if drift:
        if ages is None:
            raise ValueError("drift=True needs history_age_steps")
        scale = np.sqrt(1.0 + np.asarray(ages, dtype=np.float32).reshape(n, k))
    else:
        scale = np.ones((n, k), dtype=np.float32)

    if translation_m > 0:
        sigma = translation_m * scale
        out[:, :, 0] += rng.normal(0.0, 1.0, size=(n, k)).astype(np.float32) * sigma
        out[:, :, 1] += rng.normal(0.0, 1.0, size=(n, k)).astype(np.float32) * sigma

    if rotation_deg > 0:
        delta = np.deg2rad(rotation_deg) * scale
        delta = delta * rng.normal(0.0, 1.0, size=(n, k)).astype(np.float32)
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        cos_y, sin_y = out[:, :, 2].copy(), out[:, :, 3].copy()
        out[:, :, 2] = cos_y * cos_d - sin_y * sin_d
        out[:, :, 3] = sin_y * cos_d + cos_y * sin_d
    return out


def _fit_and_score(
    tool: types.ModuleType,
    features: np.ndarray,
    oracle: np.ndarray,
    source_type: np.ndarray,
    tags: np.ndarray,
    selectors: dict[str, np.ndarray],
    decays: list[float],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    """One arm, the same recipe as EXP-13-A: pick decay on dev, report val."""
    train_sel, dev_sel, val_sel = selectors["train"], selectors["dev"], selectors["val"]
    mean = features[train_sel].mean(axis=0, keepdims=True)
    std = features[train_sel].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = ((features - mean) / std).astype(np.float32)

    best = None
    for decay in decays:
        model = tool.fit_linear(
            normalized[train_sel],
            oracle[train_sel],
            decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
        )
        dev = tool.score(
            tool.predict(model, normalized[dev_sel]),
            oracle[dev_sel],
            source_type[dev_sel],
            tags[dev_sel],
        )
        value = dev["hard_macro_accuracy"] or 0.0
        if best is None or value > best["dev"]:
            best = {"dev": value, "decay": decay, "model": model}
    assert best is not None
    val = tool.score(
        tool.predict(best["model"], normalized[val_sel]),
        oracle[val_sel],
        source_type[val_sel],
        tags[val_sel],
    )
    return {"selected_weight_decay": best["decay"], "dev_hard_macro_accuracy": best["dev"], "val": val}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="EXP-13-A features_merged.npz")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--translation-m",
        default="0,0.05,0.1,0.2,0.4,0.8",
        help="translation sigma per level, metres (paired with --rotation-deg)",
    )
    parser.add_argument(
        "--rotation-deg",
        default="0,2.5,5,10,20,40",
        help="yaw sigma per level, degrees (paired with --translation-m)",
    )
    parser.add_argument("--noise-model", choices=("iid", "drift"), default="drift")
    parser.add_argument("--seeds", default="42,1337,7", help="one refit per seed per level")
    parser.add_argument("--dev-pct", type=int, default=15)
    parser.add_argument("--val-pct", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decays", default="1e-4,1e-3,1e-2,1e-1,1.0")
    parser.add_argument(
        "--arms",
        default="geometry",
        help="pose-dependent arms to sweep; the reference arms are always fit once",
    )
    parser.add_argument(
        "--reference-arms",
        default="system2,memory",
        help="pose-independent arms, fit once as fixed lines",
    )
    args = parser.parse_args()

    tool = _load_readout_tool()

    translations = [float(v) for v in args.translation_m.split(",") if v.strip()]
    rotations = [float(v) for v in args.rotation_deg.split(",") if v.strip()]
    if len(translations) != len(rotations):
        raise SystemExit(
            f"--translation-m and --rotation-deg must pair up: "
            f"{len(translations)} vs {len(rotations)}"
        )
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    decays = [float(v) for v in args.weight_decays.split(",") if v.strip()]
    sweep_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    reference_arms = [name.strip() for name in args.reference_arms.split(",") if name.strip()]
    unknown = sorted(set(sweep_arms) - set(POSE_DEPENDENT))
    if unknown:
        raise SystemExit(
            f"these arms do not depend on the poses, so sweeping them measures "
            f"nothing but refit noise: {unknown}"
        )

    raw = np.load(args.features, allow_pickle=False)
    data = {key: raw[key] for key in raw.files}
    oracle = np.asarray(data["oracle_view"], dtype=np.int64)
    source_type = np.asarray(data["source_type"])
    tags = np.asarray(data["tags"])
    scenes = np.asarray(data["scene_id"])
    ages = np.asarray(data["history_age_steps"], dtype=np.float32)
    clean_poses = np.asarray(data["history_rel_poses"], dtype=np.float32)

    defined = oracle >= 0
    split = tool.scene_split(scenes, args.dev_pct, args.val_pct)
    selectors = {
        "train": defined & (split == 0),
        "dev": defined & (split == 1),
        "val": defined & (split == 2),
    }
    for name, selector in selectors.items():
        if not selector.any():
            raise SystemExit(f"{name} split is empty")
    print(
        f"states: train={int(selectors['train'].sum())} dev={int(selectors['dev'].sum())} "
        f"val={int(selectors['val'].sum())} | noise={args.noise_model} "
        f"levels={len(translations)} seeds={len(seeds)}",
        flush=True,
    )

    # Reference lines: pose-independent, so one fit each at the first seed.
    references: dict[str, Any] = {}
    base_arms = tool.build_arms(data)
    for arm in reference_arms:
        if arm not in base_arms:
            raise SystemExit(f"unknown reference arm {arm!r}")
        result = _fit_and_score(
            tool, base_arms[arm], oracle, source_type, tags, selectors, decays, args, seeds[0]
        )
        references[arm] = result
        print(
            f"[reference] {arm:16s} recovery_nonfront={result['val']['recovery_nonfront_recall']} "
            f"macro={result['val']['hard_macro_accuracy']} "
            f"normal_fa={result['val']['normal_false_alarm']}",
            flush=True,
        )

    levels: list[dict[str, Any]] = []
    for translation, rotation in zip(translations, rotations):
        per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in sweep_arms}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            noisy = dict(data)
            noisy["history_rel_poses"] = perturb_rel_poses(
                clean_poses,
                translation_m=translation,
                rotation_deg=rotation,
                ages=ages,
                drift=(args.noise_model == "drift"),
                rng=rng,
            )
            arms = tool.build_arms(noisy)
            for arm in sweep_arms:
                per_arm[arm].append(
                    _fit_and_score(
                        tool, arms[arm], oracle, source_type, tags, selectors, decays, args, seed
                    )
                )
        entry: dict[str, Any] = {
            "translation_m": translation,
            "rotation_deg": rotation,
            "arms": {},
        }
        for arm, runs in per_arm.items():
            summary: dict[str, Any] = {"seeds": [r["val"] for r in runs],
                                       "selected_weight_decays": [r["selected_weight_decay"] for r in runs]}
            for metric in ("recovery_nonfront_recall", "hard_macro_accuracy", "normal_false_alarm"):
                values = [r["val"][metric] for r in runs if r["val"][metric] is not None]
                summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
                summary[f"{metric}_std"] = float(np.std(values)) if values else None
                summary[f"{metric}_min"] = float(np.min(values)) if values else None
            entry["arms"][arm] = summary
            print(
                f"t={translation:<5g}m r={rotation:<5g}deg {arm:16s} "
                f"recovery_nonfront={summary['recovery_nonfront_recall_mean']:.4f}"
                f"+-{summary['recovery_nonfront_recall_std']:.4f} "
                f"macro={summary['hard_macro_accuracy_mean']:.4f} "
                f"normal_fa={summary['normal_false_alarm_mean']:.4f}",
                flush=True,
            )
        levels.append(entry)

    report = {
        "schema": SCHEMA,
        "features": str(args.features),
        "noise_model": args.noise_model,
        "recipe": {
            "readout": "multinomial logistic regression, inverse-frequency class weights",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decays": decays,
            "seeds": seeds,
            "note": "identical to EXP-13-A's fit_recovery_readout recipe (imported)",
        },
        "split": {
            "dev_pct": args.dev_pct,
            "val_pct": args.val_pct,
            "states_train": int(selectors["train"].sum()),
            "states_dev": int(selectors["dev"].sum()),
            "states_val": int(selectors["val"].sum()),
        },
        "reference_arms": references,
        "levels": levels,
        "bias_note": (
            "Only the pose-dependent arms are perturbed; memory and system2 are "
            "replayed from cache and cannot degrade, so the sweep is biased "
            "against geometry. Geometry staying ahead is strong; geometry "
            "falling behind is not evidence that memory is better under VO."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
