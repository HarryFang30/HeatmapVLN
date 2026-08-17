#!/usr/bin/env python3
"""Inventory the reference-only data used to train heatmap-conditioned System1.

The tool deliberately does not copy images or materialize predicted heatmaps.  It
validates the existing expert and heatmap clip stores, inventories a future
DAgger root, and writes two small JSON control-plane files:

* ``inventory.json``: physical availability, size, and clip completeness.
* ``training_sources.json``: logical 50/20/30 sampling sources for training.

Heatmaps are expected to be generated online by a frozen heatmap predictor from
the referenced RGB/pose history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FJL_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl")
DEFAULT_EXPERT_ROOT = FJL_ROOT / "r2r_paronamic_data" / "train"
DEFAULT_HEATMAP_ROOT = FJL_ROOT / "data" / "heatmap_randomwalk_train_v1"
DEFAULT_OUTPUT_ROOT = FJL_ROOT / "data" / "heatmap_system1_training_v1"
DEFAULT_DAGGER_ROOT = DEFAULT_OUTPUT_ROOT / "dagger"
DEFAULT_DAGGER_BUDGET_BYTES = 300_000_000_000

MIXTURE = {
    "expert": 0.50,
    "dagger_normal": 0.20,
    "dagger_hard": 0.30,
}

EXPERT_VIEWS = ("front", "right", "back", "left", "front_down")
EXPERT_DEPTH_VIEWS = ("front", "front_down")
HEATMAP_VIEWS = ("front", "right", "back", "left")
HEATMAP_DEPTH_VIEWS = HEATMAP_VIEWS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024.0 or unit == "PiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _under_allowed_root(path: Path) -> bool:
    try:
        return os.path.commonpath((str(FJL_ROOT), str(path.absolute()))) == str(FJL_ROOT)
    except ValueError:
        return False


def _require_allowed_path(path: Path, name: str) -> Path:
    path = path.expanduser().absolute()
    if not _under_allowed_root(path):
        raise ValueError(f"{name} must stay under {FJL_ROOT}, got {path}")
    return path


def _tree_size_bytes(root: Path) -> tuple[int, list[str]]:
    if not root.exists():
        return 0, []
    total = 0
    errors: list[str] = []
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                total += path.stat(follow_symlinks=False).st_size
            except OSError as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return total, errors


def _clip_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    if not root.is_dir():
        return
    direct = sorted(path for path in root.glob("clip_*") if path.is_dir())
    for clip in direct:
        yield ".", clip
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        for clip in sorted(path for path in scene.glob("clip_*") if path.is_dir()):
            yield scene.name, clip


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object, got {type(value).__name__}")
    return value


def _array_first_dim(path: Path) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim < 1:
        raise ValueError(f"expected at least one dimension, got shape={array.shape}")
    return int(array.shape[0])


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _validate_chunks(
    clip_dir: Path,
    *,
    expected_frames: int,
    rgb_pose_views: tuple[str, ...],
    depth_views: tuple[str, ...],
) -> list[str]:
    reasons: list[str] = []
    chunks_dir = clip_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz")) if chunks_dir.is_dir() else []
    if not chunk_files:
        return ["missing_chunks"]

    expected_keys = {"frame_ids"}
    for view in rgb_pose_views:
        expected_keys.update((f"rgb_{view}", f"pose_{view}"))
    for view in depth_views:
        expected_keys.add(f"depth_{view}")

    seen: Counter[int] = Counter()
    for chunk_path in chunk_files:
        try:
            with np.load(chunk_path, allow_pickle=False) as chunk:
                missing_keys = sorted(expected_keys - set(chunk.files))
                if missing_keys:
                    _append_reason(
                        reasons,
                        f"{chunk_path.name}:missing_keys:{','.join(missing_keys)}",
                    )
                    continue
                frame_ids = np.asarray(chunk["frame_ids"], dtype=np.int64).reshape(-1)
        except Exception as exc:
            _append_reason(
                reasons,
                f"{chunk_path.name}:unreadable:{type(exc).__name__}:{exc}",
            )
            continue
        for frame_id in frame_ids.tolist():
            seen[int(frame_id)] += 1

    if expected_frames > 0:
        missing = sorted(set(range(expected_frames)) - set(seen))
        outside = sorted(frame_id for frame_id in seen if frame_id < 0 or frame_id >= expected_frames)
        duplicates = sorted(frame_id for frame_id, count in seen.items() if count > 1)
        if missing:
            _append_reason(
                reasons,
                f"chunk_frame_ids_missing:count={len(missing)}:first={missing[:8]}",
            )
        if outside:
            _append_reason(
                reasons,
                f"chunk_frame_ids_out_of_range:count={len(outside)}:first={outside[:8]}",
            )
        if duplicates:
            _append_reason(
                reasons,
                f"chunk_frame_ids_duplicated:count={len(duplicates)}:first={duplicates[:8]}",
            )
    return reasons


def _audit_clip(clip_dir: Path, *, kind: str, validate_chunks: bool) -> tuple[int, list[str]]:
    reasons: list[str] = []
    required_files = ["meta.json", "intrinsics.json", "trajectory_3d.npy"]
    if kind == "expert":
        required_files.extend(("actions.npy", "discrete_actions.npy"))
    for name in required_files:
        path = clip_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            _append_reason(reasons, f"missing_or_empty:{name}")

    meta: dict[str, Any] = {}
    num_frames = 0
    if (clip_dir / "meta.json").is_file():
        try:
            meta = _read_json(clip_dir / "meta.json")
            num_frames = int(meta.get("num_frames", 0))
            if num_frames < 4:
                _append_reason(reasons, f"num_frames_too_small:{num_frames}")
        except Exception as exc:
            _append_reason(reasons, f"invalid_meta:{type(exc).__name__}:{exc}")

    if (clip_dir / "intrinsics.json").is_file():
        try:
            _read_json(clip_dir / "intrinsics.json")
        except Exception as exc:
            _append_reason(reasons, f"invalid_intrinsics:{type(exc).__name__}:{exc}")

    array_names = ["trajectory_3d.npy"]
    if kind == "expert":
        array_names.extend(("actions.npy", "discrete_actions.npy"))
    for name in array_names:
        path = clip_dir / name
        if not path.is_file() or num_frames <= 0:
            continue
        try:
            length = _array_first_dim(path)
            if length != num_frames:
                _append_reason(reasons, f"length_mismatch:{name}:{length}!={num_frames}")
        except Exception as exc:
            _append_reason(reasons, f"invalid_array:{name}:{type(exc).__name__}:{exc}")

    if validate_chunks and num_frames > 0:
        if kind == "expert":
            views, depth_views = EXPERT_VIEWS, EXPERT_DEPTH_VIEWS
        else:
            views, depth_views = HEATMAP_VIEWS, HEATMAP_DEPTH_VIEWS
        reasons.extend(
            _validate_chunks(
                clip_dir,
                expected_frames=num_frames,
                rgb_pose_views=views,
                depth_views=depth_views,
            )
        )
    elif not (clip_dir / "chunks").is_dir() or not any((clip_dir / "chunks").glob("chunk_*.npz")):
        _append_reason(reasons, "missing_chunks")

    return num_frames, reasons


def _audit_clip_store(root: Path, *, kind: str, validate_chunks: bool) -> dict[str, Any]:
    size_bytes, stat_errors = _tree_size_bytes(root)
    clip_count = 0
    complete_count = 0
    total_frames = 0
    complete_frames = 0
    scene_counts: Counter[str] = Counter()
    incomplete: list[dict[str, Any]] = []

    for scene, clip_dir in _clip_dirs(root):
        clip_count += 1
        scene_counts[scene] += 1
        num_frames, reasons = _audit_clip(
            clip_dir,
            kind=kind,
            validate_chunks=validate_chunks,
        )
        total_frames += max(num_frames, 0)
        if reasons:
            incomplete.append(
                {
                    "relative_clip": str(clip_dir.relative_to(root)),
                    "num_frames_from_meta": num_frames,
                    "reasons": reasons,
                }
            )
        else:
            complete_count += 1
            complete_frames += num_frames
        if clip_count % 1000 == 0:
            print(
                f"[{kind}] audited {clip_count} clips; incomplete={len(incomplete)}",
                file=sys.stderr,
                flush=True,
            )

    return {
        "root": str(root),
        "exists": root.is_dir(),
        "reference_only": True,
        "scene_count": len(scene_counts),
        "scene_clip_counts": dict(sorted(scene_counts.items())),
        "clip_count": clip_count,
        "complete_clip_count": complete_count,
        "incomplete_clip_count": len(incomplete),
        "total_frames_from_meta": total_frames,
        "complete_frames_from_meta": complete_frames,
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "stat_errors": stat_errors,
        "incomplete_clips": incomplete,
        "chunk_frame_ids_validated": validate_chunks,
    }


def _count_jsonl_records(root: Path) -> tuple[int, list[str]]:
    count = 0
    errors: list[str] = []
    if not root.is_dir():
        return count, errors
    for path in sorted(root.rglob("*.jsonl")):
        try:
            with path.open("rb") as handle:
                count += sum(1 for line in handle if line.strip())
        except OSError as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return count, errors


def _audit_dagger_category(root: Path) -> dict[str, Any]:
    size_bytes, stat_errors = _tree_size_bytes(root)
    records, record_errors = _count_jsonl_records(root)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "status": "ready" if root.is_dir() and records > 0 else "planned",
        "jsonl_record_count": records,
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "errors": stat_errors + record_errors,
    }


def _audit_dagger_root(root: Path, budget_bytes: int) -> dict[str, Any]:
    size_bytes, stat_errors = _tree_size_bytes(root)
    normal = _audit_dagger_category(root / "normal")
    hard = _audit_dagger_category(root / "hard")
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "hard_limit_bytes": budget_bytes,
        "hard_limit_human": _human_bytes(budget_bytes),
        "remaining_bytes": max(budget_bytes - size_bytes, 0),
        "within_budget": size_bytes <= budget_bytes,
        "stat_errors": stat_errors,
        "normal": normal,
        "hard": hard,
        "raw_collection_root": str(root / "raw"),
    }


def _dagger_source_summary(root: Path, source_type: str, count: int) -> dict[str, Any]:
    return {
        "root": str(root),
        "container": "episodes/<episode_key>/episode.tar",
        "member": "samples.jsonl",
        "source_type_filter": source_type,
        "status": "ready" if count > 0 else "planned",
        "sample_count": int(count),
        "jsonl_record_count": int(count),
    }


def _audit_dagger_root_v2(root: Path, budget_bytes: int) -> dict[str, Any]:
    size_bytes, stat_errors = _tree_size_bytes(root)
    errors = list(stat_errors)
    source_counts: Counter[str] = Counter()
    committed_episodes = 0
    committed_frames = 0
    committed_samples = 0
    incomplete_episodes: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None

    manifest_path = root / "collection_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _read_json(manifest_path)
        except Exception as exc:
            errors.append(f"{manifest_path}: {type(exc).__name__}: {exc}")

    episodes_root = root / "episodes"
    if episodes_root.is_dir():
        for episode_dir in sorted(path for path in episodes_root.iterdir() if path.is_dir()):
            commit_path = episode_dir / "commit.json"
            tar_path = episode_dir / "episode.tar"
            reasons: list[str] = []
            commit: dict[str, Any] = {}
            if not commit_path.is_file():
                reasons.append("missing_commit_json")
            else:
                try:
                    commit = _read_json(commit_path)
                except Exception as exc:
                    reasons.append(f"invalid_commit:{type(exc).__name__}:{exc}")
            if not tar_path.is_file():
                reasons.append("missing_episode_tar")
            elif commit:
                expected_bytes = int(commit.get("tar_bytes", -1))
                if expected_bytes != tar_path.stat().st_size:
                    reasons.append(
                        f"tar_size_mismatch:{tar_path.stat().st_size}!={expected_bytes}"
                    )

            local_counts: Counter[str] = Counter()
            if not reasons:
                try:
                    with tarfile.open(tar_path, mode="r:") as archive:
                        member = archive.getmember("samples.jsonl")
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise RuntimeError("samples.jsonl is not a regular tar member")
                        for raw_line in handle:
                            if not raw_line.strip():
                                continue
                            sample = json.loads(raw_line)
                            source_type = str(sample.get("source_type") or "")
                            if source_type not in {"dagger_normal", "dagger_hard"}:
                                raise ValueError(f"invalid source_type {source_type!r}")
                            local_counts[source_type] += 1
                except Exception as exc:
                    reasons.append(f"invalid_samples:{type(exc).__name__}:{exc}")

            if reasons:
                incomplete_episodes.append(
                    {"episode_key": episode_dir.name, "reasons": reasons}
                )
                continue
            recorded_samples = int(commit.get("sample_count", -1))
            actual_samples = sum(local_counts.values())
            if recorded_samples != actual_samples:
                incomplete_episodes.append(
                    {
                        "episode_key": episode_dir.name,
                        "reasons": [
                            f"sample_count_mismatch:{actual_samples}!={recorded_samples}"
                        ],
                    }
                )
                continue
            committed_episodes += 1
            committed_samples += actual_samples
            committed_frames += int(commit.get("frame_count", 0))
            source_counts.update(local_counts)

    staging_root = root / ".staging"
    incomplete_staging = (
        sorted(path.name for path in staging_root.iterdir())
        if staging_root.is_dir()
        else []
    )
    normal = _dagger_source_summary(
        root, "dagger_normal", source_counts["dagger_normal"]
    )
    hard = _dagger_source_summary(root, "dagger_hard", source_counts["dagger_hard"])
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "collection_manifest_exists": manifest is not None,
        "collection_schema": None if manifest is None else manifest.get("schema"),
        "collection_fingerprint": None if manifest is None else manifest.get("fingerprint"),
        "committed_episode_count": committed_episodes,
        "committed_frame_count": committed_frames,
        "committed_sample_count": committed_samples,
        "incomplete_episode_count": len(incomplete_episodes),
        "incomplete_episodes": incomplete_episodes,
        "incomplete_staging": incomplete_staging,
        "size_bytes": size_bytes,
        "size_human": _human_bytes(size_bytes),
        "hard_limit_bytes": budget_bytes,
        "hard_limit_human": _human_bytes(budget_bytes),
        "remaining_bytes": max(budget_bytes - size_bytes, 0),
        "within_budget": size_bytes <= budget_bytes,
        "errors": errors,
        "normal": normal,
        "hard": hard,
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _training_sources(
    *,
    output_root: Path,
    expert: dict[str, Any],
    heatmap: dict[str, Any],
    dagger: dict[str, Any],
) -> dict[str, Any]:
    expert_status = "ready" if expert["complete_clip_count"] > 0 else "missing"
    normal = dagger["normal"]
    hard = dagger["hard"]
    return {
        "schema_version": "heatmap-system1-training-sources-v1",
        "generated_at": _utc_now(),
        "dataset_root": str(output_root),
        "mixture_basis": "per_training_sample",
        "mixture": MIXTURE,
        "storage_policy": {
            "copy_existing_images": False,
            "copy_existing_heatmap_data": False,
            "persist_predicted_heatmaps": False,
            "heatmap_generation": "online_frozen_predictor_eval_no_grad_detach",
            "expert_and_heatmap_roots_are_references": True,
        },
        "sources": [
            {
                "name": "expert",
                "role": "normal_navigation_and_native_retention",
                "root": expert["root"],
                "weight": MIXTURE["expert"],
                "status": expert_status,
                "clip_glob": "*/clip_*",
                "complete_clip_count": expert["complete_clip_count"],
                "exclude_incomplete_clips": True,
                "incomplete_clip_count": expert["incomplete_clip_count"],
                "incomplete_clip_inventory": "inventory.json#/sources/expert/incomplete_clips",
            },
            {
                "name": "dagger_normal",
                "role": "on_policy_states_where_adapter_should_not_overcorrect",
                "root": normal["root"],
                "weight": MIXTURE["dagger_normal"],
                "status": normal["status"],
                "jsonl_record_count": normal["jsonl_record_count"],
            },
            {
                "name": "dagger_hard",
                "role": "on_policy_deviation_loop_and_recovery_states",
                "root": hard["root"],
                "weight": MIXTURE["dagger_hard"],
                "status": hard["status"],
                "jsonl_record_count": hard["jsonl_record_count"],
                "recommended_loss_weight": 2.0,
            },
        ],
        "online_heatmap_source": {
            "root": heatmap["root"],
            "status": "ready" if heatmap["complete_clip_count"] > 0 else "missing",
            "purpose": "train_or_validate_heatmap_predictor_only; not a navigation-mixture pool",
            "complete_clip_count": heatmap["complete_clip_count"],
        },
        "training_ready": bool(
            expert_status == "ready"
            and normal["status"] == "ready"
            and hard["status"] == "ready"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--heatmap-root", type=Path, default=DEFAULT_HEATMAP_ROOT)
    parser.add_argument("--dagger-root", type=Path, default=DEFAULT_DAGGER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dagger-budget-bytes", type=int, default=DEFAULT_DAGGER_BUDGET_BYTES)
    parser.add_argument(
        "--skip-chunk-validation",
        action="store_true",
        help="Only check that chunk files exist; do not verify frame_ids and modality keys.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expert_root = _require_allowed_path(args.expert_root, "expert root")
    heatmap_root = _require_allowed_path(args.heatmap_root, "heatmap root")
    dagger_root = _require_allowed_path(args.dagger_root, "DAgger root")
    output_root = _require_allowed_path(args.output_root, "output root")
    if args.dagger_budget_bytes <= 0:
        raise ValueError("--dagger-budget-bytes must be positive")

    validate_chunks = not args.skip_chunk_validation
    print(f"Auditing expert store: {expert_root}", file=sys.stderr, flush=True)
    expert = _audit_clip_store(expert_root, kind="expert", validate_chunks=validate_chunks)
    print(f"Auditing heatmap store: {heatmap_root}", file=sys.stderr, flush=True)
    heatmap = _audit_clip_store(heatmap_root, kind="heatmap", validate_chunks=validate_chunks)
    print(f"Auditing DAgger store: {dagger_root}", file=sys.stderr, flush=True)
    dagger = _audit_dagger_root_v2(dagger_root, args.dagger_budget_bytes)

    inventory = {
        "schema_version": "heatmap-system1-data-inventory-v1",
        "generated_at": _utc_now(),
        "allowed_root": str(FJL_ROOT),
        "purpose": "train heatmap-to-System1 injection while freezing native System1 and heatmap predictor",
        "logical_mixture": MIXTURE,
        "storage_policy": {
            "reference_existing_data": True,
            "images_copied": False,
            "heatmaps_persisted": False,
            "online_heatmap_generation": True,
        },
        "sources": {
            "expert": expert,
            "heatmap_predictor_data": heatmap,
            "dagger": dagger,
        },
    }
    sources = _training_sources(
        output_root=output_root,
        expert=expert,
        heatmap=heatmap,
        dagger=dagger,
    )

    inventory_path = output_root / "inventory.json"
    sources_path = output_root / "training_sources.json"
    _atomic_write_json(inventory_path, inventory)
    _atomic_write_json(sources_path, sources)
    print(f"Wrote {inventory_path}")
    print(f"Wrote {sources_path}")
    print(
        "Expert clips: "
        f"{expert['complete_clip_count']}/{expert['clip_count']} complete; "
        f"Heatmap clips: {heatmap['complete_clip_count']}/{heatmap['clip_count']} complete; "
        f"DAgger bytes: {dagger['size_bytes']}/{args.dagger_budget_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
