#!/usr/bin/env python3
"""EXP-13 B: can any readout name the oracle direction, and from which features?

Consumes the cache written by ``cache_recovery_decision_features.py`` and fits
one linear readout per feature arm to predict the oracle's four-way direction.
Everything here is CPU-only and deterministic.

The comparison that matters is ``system2_memory`` minus ``system2``: System2's
own summary is the null ("the VLM already knows where to go, it just decodes
'front'"), and ``M_t`` earns its place in the decision layer only by beating it.
``geometry`` is the control that keeps the answer honest -- EXP-02 and EXP-04
showed ``M_t`` is close to a deterministic function of the relative poses, so if
raw pose does just as well, the finding is "System2 lacks geometry", not
"the learned memory carries something extra".

Scene-disjoint by construction: scenes are hashed once and split into
train/dev/val.  ``dev`` exists only to pick each arm's weight decay, so the
higher-dimensional arms cannot win or lose on regularisation luck; every number
that is reported comes from ``val``, which no fitting decision ever touched.

Readouts use inverse-frequency class weights.  Without them every arm collapses
onto the majority class and reproduces EXP-12's constant "front" predictor,
which measures the label prior rather than the features.  The unweighted
constant-front baseline is reported alongside for exactly that reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

VIEWS = ("front", "right", "back", "left")
SCHEMA = "heatmapvln-exp13-readout-v1"
RECOVERY_TAGS = ("wrong_branch", "off_route")


def scene_split(scene_ids: np.ndarray, dev_pct: int, val_pct: int) -> np.ndarray:
    """Assign 0=train, 1=dev, 2=val by a stable hash of the scene id."""
    if not 0 < dev_pct < 100 or not 0 < val_pct < 100 or dev_pct + val_pct >= 100:
        raise SystemExit("dev/val percentages must be positive and leave a train split")
    bucket_of: dict[str, int] = {}
    for scene in sorted(set(scene_ids.tolist())):
        digest = hashlib.md5(str(scene).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if bucket < val_pct:
            bucket_of[scene] = 2
        elif bucket < val_pct + dev_pct:
            bucket_of[scene] = 1
        else:
            bucket_of[scene] = 0
    return np.asarray([bucket_of[str(scene)] for scene in scene_ids.tolist()], dtype=np.int8)


def build_arms(data: Any) -> dict[str, np.ndarray]:
    """Flatten the cached tensors into one feature matrix per arm."""
    states = int(data["oracle_view"].shape[0])

    def flat(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float32).reshape(states, -1)

    traj = flat("traj_hidden")
    plan = flat("plan_z0")
    mask = np.asarray(data["history_memory_mask"], dtype=np.float32).reshape(states, -1)
    memory = flat("history_memory") * np.repeat(
        mask, flat("history_memory").shape[1] // mask.shape[1], axis=1
    )
    geometry = np.concatenate(
        [
            flat("history_rel_poses"),
            flat("history_visibility"),
            np.asarray(data["history_age_steps"], dtype=np.float32).reshape(states, -1),
            mask,
        ],
        axis=1,
    )
    memory_block = np.concatenate([memory, mask], axis=1)
    return {
        "system2": traj,
        "system2_memory": np.concatenate([traj, memory_block], axis=1),
        "memory": memory_block,
        "geometry": geometry,
        "system2_geometry": np.concatenate([traj, geometry], axis=1),
        "plan_z0": plan,
        "plan_z0_memory": np.concatenate([plan, memory_block], axis=1),
    }


def fit_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weight_decay: float,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Any:
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y).long()
    counts = torch.bincount(y, minlength=len(VIEWS)).float()
    weights = torch.where(counts > 0, counts.sum() / (len(VIEWS) * counts.clamp_min(1.0)), counts)
    model = torch.nn.Linear(x.shape[1], len(VIEWS))
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x[index]), y[index]).backward()
            optimizer.step()
    return model


def predict(model: Any, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(features)).argmax(dim=1).numpy().astype(np.int64)


def score(
    predicted: np.ndarray,
    oracle: np.ndarray,
    source_type: np.ndarray,
    tags: np.ndarray,
) -> dict[str, Any]:
    """The three pre-registered readings plus the slices behind them."""
    hard = source_type == "dagger_hard"
    normal = source_type == "dagger_normal"
    recovery = np.asarray(
        [any(tag in str(value).split("|") for tag in RECOVERY_TAGS) for value in tags.tolist()]
    )
    nonfront = oracle > 0

    def accuracy(selector: np.ndarray) -> float | None:
        if not selector.any():
            return None
        return float((predicted[selector] == oracle[selector]).mean())

    macro_parts = {
        VIEWS[view]: accuracy(hard & (oracle == view)) for view in range(len(VIEWS))
    }
    present = [value for value in macro_parts.values() if value is not None]
    recovery_nonfront = hard & recovery & nonfront
    normal_front = normal & (oracle == 0)
    return {
        "recovery_nonfront_recall": accuracy(recovery_nonfront),
        "recovery_nonfront_states": int(recovery_nonfront.sum()),
        "hard_macro_accuracy": float(np.mean(present)) if present else None,
        "hard_per_view_accuracy": macro_parts,
        "hard_top1_accuracy": accuracy(hard),
        "hard_states": int(hard.sum()),
        "normal_false_alarm": (
            float((predicted[normal_front] > 0).mean()) if normal_front.any() else None
        ),
        "normal_front_states": int(normal_front.sum()),
        "prediction_distribution": {
            VIEWS[view]: int((predicted[hard] == view).sum()) for view in range(len(VIEWS))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="merged cache .npz")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dev-pct", type=int, default=15, help="percent of scenes used to pick weight decay")
    parser.add_argument("--val-pct", type=int, default=25, help="percent of scenes reserved for reporting")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-decays",
        default="1e-4,1e-3,1e-2,1e-1,1.0",
        help="grid searched on dev, never on val",
    )
    parser.add_argument("--arms", default="", help="comma-separated subset; empty means all")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    oracle = np.asarray(data["oracle_view"], dtype=np.int64)
    native = np.asarray(data["native_view"], dtype=np.int64)
    source_type = np.asarray(data["source_type"])
    tags = np.asarray(data["tags"])
    scenes = np.asarray(data["scene_id"])

    defined = oracle >= 0
    split = scene_split(scenes, args.dev_pct, args.val_pct)
    arms = build_arms(data)
    requested = [name.strip() for name in args.arms.split(",") if name.strip()]
    if requested:
        missing = sorted(set(requested) - set(arms))
        if missing:
            raise SystemExit(f"unknown arms: {missing}")
        arms = {name: arms[name] for name in requested}

    decays = [float(value) for value in args.weight_decays.split(",") if value.strip()]
    train_sel = defined & (split == 0)
    dev_sel = defined & (split == 1)
    val_sel = defined & (split == 2)
    for name, selector in (("train", train_sel), ("dev", dev_sel), ("val", val_sel)):
        if not selector.any():
            raise SystemExit(f"{name} split is empty; adjust --dev-pct/--val-pct")

    results: dict[str, Any] = {}
    for arm, features in arms.items():
        mean = features[train_sel].mean(axis=0, keepdims=True)
        std = features[train_sel].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        normalized = ((features - mean) / std).astype(np.float32)
        best = None
        for decay in decays:
            model = fit_linear(
                normalized[train_sel],
                oracle[train_sel],
                decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
            dev_score = score(
                predict(model, normalized[dev_sel]),
                oracle[dev_sel],
                source_type[dev_sel],
                tags[dev_sel],
            )
            selection_value = dev_score["hard_macro_accuracy"] or 0.0
            if best is None or selection_value > best["dev_hard_macro_accuracy"]:
                best = {
                    "weight_decay": decay,
                    "dev_hard_macro_accuracy": selection_value,
                    "model": model,
                }
        assert best is not None
        val_score = score(
            predict(best["model"], normalized[val_sel]),
            oracle[val_sel],
            source_type[val_sel],
            tags[val_sel],
        )
        results[arm] = {
            "feature_dim": int(features.shape[1]),
            "selected_weight_decay": best["weight_decay"],
            "dev_hard_macro_accuracy": best["dev_hard_macro_accuracy"],
            "val": val_score,
        }
        print(
            f"{arm:20s} dim={features.shape[1]:6d} wd={best['weight_decay']:<6g} "
            f"val macro={val_score['hard_macro_accuracy']} "
            f"recovery_nonfront={val_score['recovery_nonfront_recall']} "
            f"normal_fa={val_score['normal_false_alarm']}",
            flush=True,
        )

    constant_front = np.zeros(int(val_sel.sum()), dtype=np.int64)
    baselines = {
        "constant_front": score(
            constant_front, oracle[val_sel], source_type[val_sel], tags[val_sel]
        ),
    }
    native_sel = val_sel & (native >= 0)
    if native_sel.any():
        baselines["native_system2_proposal"] = score(
            native[native_sel], oracle[native_sel], source_type[native_sel], tags[native_sel]
        )

    deltas = {}
    if "system2" in results and "system2_memory" in results:
        for metric in ("recovery_nonfront_recall", "hard_macro_accuracy"):
            base = results["system2"]["val"][metric]
            with_memory = results["system2_memory"]["val"][metric]
            deltas[metric] = (
                None if base is None or with_memory is None else (with_memory - base) * 100.0
            )

    report = {
        "schema": SCHEMA,
        "features": str(args.features),
        "split": {
            "dev_pct": args.dev_pct,
            "val_pct": args.val_pct,
            "scenes_total": int(len(set(scenes.tolist()))),
            "scenes_train": int(len(set(scenes[split == 0].tolist()))),
            "scenes_dev": int(len(set(scenes[split == 1].tolist()))),
            "scenes_val": int(len(set(scenes[split == 2].tolist()))),
            "states_train": int(train_sel.sum()),
            "states_dev": int(dev_sel.sum()),
            "states_val": int(val_sel.sum()),
            "states_undefined_oracle": int((~defined).sum()),
        },
        "recipe": {
            "readout": "multinomial logistic regression, inverse-frequency class weights",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decays": decays,
            "seed": args.seed,
        },
        "arms": {name: value for name, value in results.items()},
        "baselines": baselines,
        "memory_minus_system2_pt": deltas,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"memory_minus_system2_pt": deltas, "baselines": baselines}, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
