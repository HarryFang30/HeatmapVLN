#!/usr/bin/env python3
"""Create a small leaf-symlink-only view for distributed training smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "heatmapvln-amb3r-pose-adapt-smoke-view-v1"


def _under(path: Path, allowed_root: Path) -> Path:
    path = path.expanduser().resolve()
    allowed_root = allowed_root.expanduser().resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Path must stay below {allowed_root}: {path}") from exc
    return path


def flat_split(scene_name: str) -> str:
    value = int(hashlib.md5(scene_name.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if value < 10 else "train"


def valid_clips(scene_dir: Path) -> list[Path]:
    return [
        clip
        for clip in sorted(scene_dir.glob("clip_*"))
        if clip.is_dir()
        and (clip / "meta.json").is_file()
        and any((clip / "chunks").glob("chunk_*.npz"))
    ]


def select_clips(source_root: Path, num_train_clips: int) -> tuple[list[Path], Path]:
    train: list[Path] | None = None
    val: Path | None = None
    for scene in sorted(path for path in source_root.iterdir() if path.is_dir()):
        clips = valid_clips(scene)
        if (
            flat_split(scene.name) == "train"
            and train is None
            and len(clips) >= num_train_clips
        ):
            # Keep train clips in one scene so split identity is unambiguous.
            train = clips[:num_train_clips]
        if flat_split(scene.name) == "val" and val is None and clips:
            # Validation is disabled, but train.py constructs the val dataset
            # before applying that flag. One metadata-only val clip keeps this
            # construction path valid; no AMB3R cache is generated for it.
            val = clips[0]
    if train is None or val is None:
        raise RuntimeError(
            f"Unable to select {num_train_clips} same-scene train clips and one val clip"
        )
    return train, val


def _link_leaf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Unexpected existing smoke-view leaf: {destination}")
    destination.symlink_to(source.resolve())


def materialize_clip(source: Path, data_root: Path) -> Path:
    destination = data_root / source.parent.name / source.name
    (destination / "chunks").mkdir(parents=True, exist_ok=True)
    for directory in (data_root, destination.parent, destination, destination / "chunks"):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Smoke-view hierarchy must use real directories: {directory}")
    for leaf in sorted(source.iterdir()):
        if leaf.is_file():
            _link_leaf(leaf, destination / leaf.name)
    for chunk in sorted((source / "chunks").glob("chunk_*.npz")):
        _link_leaf(chunk, destination / "chunks" / chunk.name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--num-train-clips", type=int, default=2)
    parser.add_argument("--allowed-root", default="/mnt/afs/liwenhao/agent/370910109")
    args = parser.parse_args()
    if args.num_train_clips < 1:
        raise ValueError("--num-train-clips must be positive")

    allowed = Path(args.allowed_root).expanduser().resolve(strict=True)
    source = _under(Path(args.source_root), allowed)
    smoke_root = _under(Path(args.smoke_root), allowed)
    if not source.is_dir():
        raise FileNotFoundError(source)
    smoke_root.mkdir(parents=True, exist_ok=True)
    data_root = smoke_root / "data"
    cache_root = smoke_root / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    for directory in (smoke_root, data_root, cache_root):
        if directory.is_symlink():
            raise RuntimeError(f"Smoke root/data/cache cannot be symlinks: {directory}")

    train_sources, val_source = select_clips(source, args.num_train_clips)
    train_views = [materialize_clip(clip, data_root) for clip in train_sources]
    val_view = materialize_clip(val_source, data_root)
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "smoke_root": str(smoke_root),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "train_clips": [
            f"{path.parent.name}/{path.name}" for path in train_views
        ],
        "num_train_clips": len(train_views),
        "val_construction_only_clip": f"{val_view.parent.name}/{val_view.name}",
        "train_scene_split": flat_split(train_views[0].parent.name),
        "val_scene_split": flat_split(val_view.parent.name),
        "directory_symlinks": False,
        "leaf_file_symlinks": True,
        "checkpoint_hash_locking": False,
    }
    manifest = smoke_root / "view.json"
    temporary = manifest.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
