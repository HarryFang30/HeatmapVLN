#!/usr/bin/env python3
"""EXP-16: is route progress already readable, and can an "arrived" readout pay at episode level?

Pure CPU.  Consumes the EXP-13-A feature cache and the sealed DAgger records; writes one JSON.

Labels come only from the sealed records (no re-simulation):

* ``progress_bin`` = floor(4 * route_progress_m / reference_path_length), clipped to 0..3,
  where the reference-path length is summed per episode from the R2R annotation;
* ``arrived``      = oracle.terminal and oracle.travelled_m <= --arrive-m (default 2.0 m;
  deliberately looser than the 1 m relabel horizon and inside the 3 m success radius).

Arms: ``age_mask`` (16), ``geo_pose`` (32), ``geo_pose_age_mask`` (48), ``plan_z0`` (3072),
``system2`` = PCA of ``traj_hidden`` fitted on train scenes only at 64/256/1024 dims (the
dimension is picked on dev), and ``memory_LEAKED`` (2056) which is reported but never
selected: it is a frozen-head output and the head saw 6 of the 17 held-out scenes.
``history_visibility`` is a head output too and is not used anywhere here.

Scene split is the same md5 rule as ``fit_recovery_readout.scene_split``: <25 val,
25..39 dev, else train.  Readouts use inverse-frequency class weights, weight decay chosen on dev.

Step-3a net-benefit curve (dev): for a threshold sweep, ``p`` = false-stop rate on non-arrived
dev states (all, and normal-bucket only), ``r`` = arrived recall,
``F`` = mean over the native val_unseen episodes of 1-(1-p)^trajectory_calls,
``net_refined`` = BONUS*r - F*SR_NATIVE (every false stop assumed to land outside the 3 m circle),
``net_conservative`` = BONUS*r - F (every false stop assumed to kill a would-be success).
The dev-chosen operating point of the refined peak is then read once on val.

Modes: ``--dry`` builds labels and the join only and prints counts (no metric);
``--smoke N`` scans only the first N episode tars.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "heatmapvln-exp16-progress-readout-v1"
BONUS = 0.0892      # OS - SR of the current method, protocol seed 42 (164 / 1839)
SR_NATIVE = 0.6248  # native closed-loop SR, protocol seed 42
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
PCA_DIMS = (64, 256, 1024)
STATE_FA_GRID = (0.005, 0.002, 0.001, 0.0007, 0.0005)


def scene_bucket(scene: str) -> int:
    return int(hashlib.md5(str(scene).encode("utf-8")).hexdigest()[:8], 16) % 100


def split_code(scene: str, val_pct: int = 25, dev_pct: int = 15) -> int:
    """0 = train, 1 = dev, 2 = val (identical to fit_recovery_readout.scene_split)."""
    bucket = scene_bucket(scene)
    if bucket < val_pct:
        return 2
    if bucket < val_pct + dev_pct:
        return 1
    return 0


def progress_bin(route_progress_m: float, path_length_m: float, bins: int = 4) -> int | None:
    if not (np.isfinite(route_progress_m) and np.isfinite(path_length_m)) or path_length_m <= 0:
        return None
    frac = min(1.0, max(0.0, float(route_progress_m) / float(path_length_m)))
    return int(min(bins - 1, np.floor(frac * bins)))


def arrived_flag(oracle: dict[str, Any] | None, arrive_m: float) -> bool:
    if not isinstance(oracle, dict) or not bool(oracle.get("terminal")):
        return False
    try:
        travelled = float(oracle.get("travelled_m"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(travelled) and 0.0 <= travelled <= float(arrive_m))


def episode_false_stop_rate(state_fa: float, calls: np.ndarray) -> float:
    """Mean over episodes of P(at least one false stop) given a per-call false-stop rate."""
    p = float(min(1.0, max(0.0, state_fa)))
    return float(np.mean(1.0 - (1.0 - p) ** np.asarray(calls, dtype=np.float64)))


def net_benefit(recall: float, episode_false_stop: float, *, refined: bool) -> float:
    cost = episode_false_stop * (SR_NATIVE if refined else 1.0)
    return float(BONUS * recall - cost)


def polyline_length(points: Any) -> float:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def scan_sealed_records(collection_root: Path, limit: int) -> tuple[dict[str, dict[str, Any]], int, int]:
    tars = sorted(glob.glob(str(collection_root / "shard_*/episodes/*/episode.tar")))
    if limit > 0:
        tars = tars[:limit]
    rows: dict[str, dict[str, Any]] = {}
    bad = 0
    for path in tars:
        try:
            with tarfile.open(path) as archive:
                member = next((m for m in archive.getmembers() if m.name.endswith("samples.jsonl")), None)
                if member is None:
                    bad += 1
                    continue
                for line in archive.extractfile(member).read().decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    oracle = row.get("oracle") if isinstance(row.get("oracle"), dict) else {}
                    rows[str(row["key"])] = {
                        "scene": str(row["scene_id"]),
                        "episode_id": str(row["episode_id"]),
                        "route_progress_m": float(row.get("route_progress_m", np.nan)),
                        "oracle": oracle,
                        "source": str(row.get("source_type", "")),
                    }
        except Exception:  # noqa: BLE001 - a corrupt tar is counted, never silently skipped
            bad += 1
    return rows, len(tars), bad


def reference_lengths(train_json: Path) -> dict[str, float]:
    with gzip.open(train_json, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    return {str(e["episode_id"]): polyline_length(e["reference_path"]) for e in episodes}


def trajectory_calls(progress_jsonl: Path) -> np.ndarray:
    calls = [
        int(json.loads(line)["trajectory_calls"])
        for line in progress_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.asarray(calls, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection-root", type=Path, required=True, help="sealed DAgger collection (shard_*/episodes/*/episode.tar)")
    parser.add_argument("--features", type=Path, required=True, help="EXP-13-A features_merged.npz")
    parser.add_argument("--train-json", type=Path, required=True, help="R2R_VLNCE train.json.gz (reference paths)")
    parser.add_argument("--native-progress", type=Path, required=True, help="native val_unseen merged/progress.jsonl (trajectory_calls)")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--arrive-m", type=float, default=2.0)
    parser.add_argument("--dry", action="store_true", help="labels + join only, no readout")
    parser.add_argument("--smoke", type=int, default=0, help="scan only the first N tars")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, n_tars, bad = scan_sealed_records(args.collection_root, args.smoke)
    lengths = reference_lengths(args.train_json)
    data = np.load(args.features, allow_pickle=False)
    keys = [str(k) for k in data["sample_key"].tolist()]
    joined = [i for i, k in enumerate(keys) if k in rows]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        "arrive_m": args.arrive_m,
        "bonus_os_minus_sr": BONUS,
        "sr_native": SR_NATIVE,
        "tars_scanned": n_tars,
        "tars_bad": bad,
        "sealed_states": len(rows),
        "cache_states": len(keys),
        "joined": len(joined),
    }
    if not joined:
        print(json.dumps(report, indent=1))
        return

    idx = np.asarray(joined)
    meta = [rows[keys[i]] for i in idx]
    scenes = np.asarray([m["scene"] for m in meta])
    split = np.asarray([split_code(s) for s in scenes], dtype=np.int8)
    pbin_list = [progress_bin(m["route_progress_m"], lengths.get(m["episode_id"], np.nan)) for m in meta]
    ok = np.asarray([b is not None for b in pbin_list])
    pbin = np.asarray([-1 if b is None else b for b in pbin_list], dtype=np.int64)
    arrived = np.asarray([arrived_flag(m["oracle"], args.arrive_m) for m in meta])
    source = np.asarray([m["source"] for m in meta])
    calls = trajectory_calls(args.native_progress)
    report.update(
        {
            "usable_states": int(ok.sum()),
            "split_counts": {"train": int((split == 0).sum()), "dev": int((split == 1).sum()), "val": int((split == 2).sum())},
            "progress_bin_dist": {str(b): int((pbin[ok] == b).sum()) for b in range(4)},
            "arrived_count": int(arrived.sum()),
            "arrived_frac": float(arrived.mean()),
            "arrived_by_source": {s: int((arrived & (source == s)).sum()) for s in ("dagger_hard", "dagger_normal")},
            "native_trajectory_calls": {
                "episodes": int(calls.size),
                "mean": float(calls.mean()),
                "median": float(np.median(calls)),
                "p90": float(np.percentile(calls, 90)),
            },
        }
    )
    if args.dry:
        print(json.dumps(report, indent=1))
        return

    import torch

    torch.manual_seed(args.seed)
    states = int(data["oracle_view"].shape[0])

    def flat(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float32).reshape(states, -1)[idx]

    mask = np.asarray(data["history_memory_mask"], dtype=np.float32).reshape(states, -1)[idx]
    age = flat("history_age_steps")
    pose = flat("history_rel_poses")
    memory = flat("history_memory") * np.repeat(mask, flat("history_memory").shape[1] // mask.shape[1], axis=1)
    arms: dict[str, np.ndarray] = {
        "age_mask": np.concatenate([age, mask], axis=1),
        "geo_pose": pose,
        "geo_pose_age_mask": np.concatenate([pose, age, mask], axis=1),
        "plan_z0": flat("plan_z0"),
        "memory_LEAKED": np.concatenate([memory, mask], axis=1),
    }
    train = (split == 0) & ok
    dev = (split == 1) & ok
    val = (split == 2) & ok
    traj = flat("traj_hidden")
    mu = traj[train].mean(axis=0, keepdims=True)
    _, _, components = torch.pca_lowrank(torch.from_numpy(traj[train] - mu), q=max(PCA_DIMS), center=False)
    for q in PCA_DIMS:
        arms[f"system2_pca{q}"] = (traj - mu) @ components[:, :q].numpy()

    def standardize(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return (x - ref.mean(axis=0, keepdims=True)) / (ref.std(axis=0, keepdims=True) + 1e-6)

    def fit(x: np.ndarray, y: np.ndarray, wd: float, classes: int, epochs: int = 12, bs: int = 256, lr: float = 1e-3):
        torch.manual_seed(args.seed)
        X, Y = torch.from_numpy(x), torch.from_numpy(y).long()
        counts = torch.bincount(Y, minlength=classes).float()
        weights = torch.where(counts > 0, counts.sum() / (classes * counts.clamp_min(1.0)), counts)
        model = torch.nn.Linear(x.shape[1], classes)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        generator = torch.Generator().manual_seed(args.seed)
        for _ in range(epochs):
            order = torch.randperm(X.shape[0], generator=generator)
            for start in range(0, X.shape[0], bs):
                index = order[start : start + bs]
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(X[index]), Y[index]).backward()
                optimizer.step()
        return model

    def probabilities(model, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()

    def auc(pos: np.ndarray, neg: np.ndarray) -> float:
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
        return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def macro(pred: np.ndarray, truth: np.ndarray, classes: int) -> float:
        parts = [(pred[truth == c] == c).mean() for c in range(classes) if (truth == c).any()]
        return float(np.mean(parts)) if parts else float("nan")

    y_bin = pbin
    y_arr = arrived.astype(np.int64)
    results: dict[str, Any] = {"progress": {}, "arrived": {}, "net_benefit": {}}
    for name, feature in arms.items():
        xs = standardize(feature, feature[train])
        # progress bins
        best = None
        for wd in WEIGHT_DECAYS:
            model = fit(xs[train], y_bin[train], wd, 4)
            score = macro(probabilities(model, xs[dev]).argmax(axis=1), y_bin[dev], 4)
            if best is None or score > best[0]:
                best = (score, wd, model)
        pred = probabilities(best[2], xs[val]).argmax(axis=1)
        results["progress"][name] = {
            "dim": int(feature.shape[1]),
            "weight_decay": best[1],
            "dev_macro": round(best[0], 4),
            "val_macro": round(macro(pred, y_bin[val], 4), 4),
            "val_top1": round(float((pred == y_bin[val]).mean()), 4),
        }
        # arrived
        best = None
        for wd in WEIGHT_DECAYS:
            model = fit(xs[train], y_arr[train], wd, 2)
            s = probabilities(model, xs[dev])[:, 1]
            score = auc(s[y_arr[dev] == 1], s[y_arr[dev] == 0])
            if best is None or score > best[0]:
                best = (score, wd, model)
        model = best[2]
        s_dev, s_val = probabilities(model, xs[dev])[:, 1], probabilities(model, xs[val])[:, 1]
        pos_dev, neg_dev = s_dev[y_arr[dev] == 1], s_dev[y_arr[dev] == 0]
        neg_dev_normal = s_dev[(y_arr[dev] == 0) & (source[dev] == "dagger_normal")]
        if len(neg_dev_normal) == 0:
            neg_dev_normal = neg_dev
        results["arrived"][name] = {
            "weight_decay": best[1],
            "dev_auc": round(best[0], 4),
            "val_auc": round(auc(s_val[y_arr[val] == 1], s_val[y_arr[val] == 0]), 4),
            "dev_recall_at_state_fa": {
                str(fa): round(float((pos_dev > np.quantile(neg_dev, 1 - fa)).mean()), 3) for fa in STATE_FA_GRID
            },
        }
        # net-benefit curve on dev
        curve = []
        for fa in np.geomspace(1e-4, 5e-2, 40):
            threshold = np.quantile(neg_dev, 1 - fa)
            recall = float((pos_dev > threshold).mean())
            p_all = float((neg_dev > threshold).mean())
            p_normal = float((neg_dev_normal > threshold).mean())
            f_all = episode_false_stop_rate(p_all, calls)
            f_normal = episode_false_stop_rate(p_normal, calls)
            curve.append(
                {
                    "state_fa_all": round(p_all, 5),
                    "state_fa_normal": round(p_normal, 5),
                    "recall": round(recall, 3),
                    "episode_false_stop_all": round(f_all, 4),
                    "episode_false_stop_normal": round(f_normal, 4),
                    "net_conservative_normal": round(net_benefit(recall, f_normal, refined=False), 4),
                    "net_refined_normal": round(net_benefit(recall, f_normal, refined=True), 4),
                    "net_refined_all": round(net_benefit(recall, f_all, refined=True), 4),
                }
            )
        peak_normal = max(curve, key=lambda c: c["net_refined_normal"])
        peak_all = max(curve, key=lambda c: c["net_refined_all"])
        threshold = np.quantile(neg_dev, 1 - max(peak_normal["state_fa_all"], 1e-6))
        val_recall = float((s_val[y_arr[val] == 1] > threshold).mean())
        val_neg_normal = s_val[(y_arr[val] == 0) & (source[val] == "dagger_normal")]
        val_p_normal = float((val_neg_normal > threshold).mean()) if len(val_neg_normal) else float("nan")
        results["net_benefit"][name] = {
            "peak_refined_normal": peak_normal,
            "peak_refined_all": peak_all,
            "val_at_dev_peak": {
                "recall": round(val_recall, 3),
                "state_fa_normal": round(val_p_normal, 5),
                "episode_false_stop_normal": round(episode_false_stop_rate(val_p_normal, calls), 4) if np.isfinite(val_p_normal) else None,
                "net_refined_normal": round(net_benefit(val_recall, episode_false_stop_rate(val_p_normal, calls), refined=True), 4) if np.isfinite(val_p_normal) else None,
            },
            "curve": curve,
        }
        print(f"[exp16] {name} done", file=sys.stderr, flush=True)

    clean = {k: v for k, v in results["arrived"].items() if not k.endswith("_LEAKED")}
    best_clean = max(clean.items(), key=lambda kv: kv[1]["dev_auc"])[0]
    results["gate_3a"] = {
        "best_clean_arm_by_dev_auc": best_clean,
        "peak_refined_normal": results["net_benefit"][best_clean]["peak_refined_normal"]["net_refined_normal"],
        "peak_refined_all": results["net_benefit"][best_clean]["peak_refined_all"]["net_refined_all"],
        "rule": "enter step 3 iff peak_refined_normal >= +0.010 and peak_refined_all > 0",
    }
    report["results"] = results
    text = json.dumps(report, indent=1)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
