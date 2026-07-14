#!/usr/bin/env python3
"""Build a deterministic, scene-balanced symlink view of an SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MANIFEST_NAME = "balanced_sft_view_manifest.json"
MANIFEST_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--total-clips", type=int, default=500)
    return parser.parse_args()


def _identity_hash(identities: list[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def enumerate_source_clips(source_root: Path, split: str) -> dict[str, list[Path]]:
    source_split = source_root / split
    if not source_split.is_dir():
        raise FileNotFoundError(f"Source split directory not found: {source_split}")

    scene_dirs = sorted(path for path in source_split.iterdir() if path.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories found in: {source_split}")

    clips_by_scene: dict[str, list[Path]] = {}
    for scene_dir in scene_dirs:
        clips = sorted(
            path.resolve(strict=True)
            for path in scene_dir.iterdir()
            if path.is_dir() and path.name.startswith("clip_")
        )
        if not clips:
            raise FileNotFoundError(f"Scene has no clip_* directories: {scene_dir}")
        clips_by_scene[scene_dir.name] = clips
    return clips_by_scene


def scene_round_robin_selection(
    clips_by_scene: dict[str, list[Path]],
    total_clips: int,
) -> list[tuple[str, Path]]:
    if total_clips <= 0:
        raise ValueError(f"total_clips must be positive, got {total_clips}")
    available = sum(len(clips) for clips in clips_by_scene.values())
    if total_clips > available:
        raise ValueError(
            f"Requested {total_clips} clips, but source only has {available}"
        )

    scenes = sorted(clips_by_scene)
    cursors = {scene: 0 for scene in scenes}
    selected: list[tuple[str, Path]] = []
    while len(selected) < total_clips:
        progressed = False
        for scene in scenes:
            cursor = cursors[scene]
            clips = clips_by_scene[scene]
            if cursor >= len(clips):
                continue
            selected.append((scene, clips[cursor]))
            cursors[scene] += 1
            progressed = True
            if len(selected) == total_clips:
                break
        if not progressed:
            raise RuntimeError(
                f"Scene round-robin exhausted at {len(selected)}/{total_clips} clips"
            )
    return selected


def expected_manifest(
    *,
    source_root: Path,
    output_root: Path,
    split: str,
    total_clips: int,
    selected: list[tuple[str, Path]],
    source_scene_count: int,
    source_clip_count: int,
) -> dict[str, Any]:
    identities = [f"{split}/{scene}/{clip.name}" for scene, clip in selected]
    per_scene = Counter(scene for scene, _clip in selected)
    scenes = sorted(per_scene)
    return {
        "manifest_version": MANIFEST_VERSION,
        "selection_algorithm": "sorted_scene_round_robin_v1",
        "source_root": str(source_root),
        "source_split": str(source_root / split),
        "output_root": str(output_root),
        "split": split,
        "requested_total_clips": total_clips,
        "total_clips": len(selected),
        "scene_count": len(scenes),
        "scenes": scenes,
        "per_scene_counts": {
            scene: int(per_scene[scene])
            for scene in scenes
        },
        "source_scene_count": source_scene_count,
        "source_clip_count": source_clip_count,
        "selected_clip_identities": identities,
        "selected_clip_identity_sha256": _identity_hash(identities),
        "selected_clips": [
            {
                "identity": identity,
                "source_resolved": str(clip),
            }
            for identity, (_scene, clip) in zip(identities, selected, strict=True)
        ],
    }


def _verify_existing_view(output_root: Path, expected: dict[str, Any]) -> None:
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Output root already contains data but has no {MANIFEST_NAME}: {output_root}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        actual_manifest = json.load(handle)
    if actual_manifest != expected:
        raise RuntimeError(
            f"Existing balanced-view manifest does not match requested selection: {manifest_path}"
        )

    expected_root_entries = {MANIFEST_NAME, str(expected["split"])}
    actual_root_entries = {path.name for path in output_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise RuntimeError(
            "Existing balanced-view root contains unexpected or missing content: "
            f"expected={sorted(expected_root_entries)} actual={sorted(actual_root_entries)}"
        )

    expected_links = {
        item["identity"]: Path(item["source_resolved"])
        for item in expected["selected_clips"]
    }
    split_dir = output_root / str(expected["split"])
    actual_links: dict[str, Path] = {}
    scene_entries = sorted(split_dir.iterdir()) if split_dir.is_dir() else []
    if any(not path.is_dir() for path in scene_entries):
        raise RuntimeError(f"Balanced-view split contains non-directory content: {split_dir}")
    actual_scenes = {path.name for path in scene_entries}
    if actual_scenes != set(expected["scenes"]):
        raise RuntimeError(
            "Existing balanced-view scene set differs from manifest: "
            f"expected={expected['scenes']} actual={sorted(actual_scenes)}"
        )
    for scene_dir in scene_entries:
        for clip_path in sorted(scene_dir.iterdir()):
            identity = f"{expected['split']}/{scene_dir.name}/{clip_path.name}"
            actual_links[identity] = clip_path
    if set(actual_links) != set(expected_links):
        missing = sorted(set(expected_links) - set(actual_links))
        unexpected = sorted(set(actual_links) - set(expected_links))
        raise RuntimeError(
            "Existing balanced-view hierarchy differs from manifest: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for identity, link_path in actual_links.items():
        if not link_path.is_symlink():
            raise RuntimeError(f"Balanced-view clip is not a symlink: {link_path}")
        resolved = link_path.resolve(strict=True)
        if resolved != expected_links[identity]:
            raise RuntimeError(
                f"Balanced-view symlink mismatch for {identity}: "
                f"expected={expected_links[identity]} actual={resolved}"
            )


def build_balanced_sft_view(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split: str = "train",
    total_clips: int = 500,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve(strict=True)
    output_input = Path(output_root).expanduser()
    if output_input.is_symlink():
        raise RuntimeError(f"Output root itself must not be a symlink: {output_input}")
    output = output_input.resolve(strict=False)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            f"Source and output roots must be isolated: source={source} output={output}"
        )

    clips_by_scene = enumerate_source_clips(source, split)
    selected = scene_round_robin_selection(clips_by_scene, total_clips)
    manifest = expected_manifest(
        source_root=source,
        output_root=output,
        split=split,
        total_clips=total_clips,
        selected=selected,
        source_scene_count=len(clips_by_scene),
        source_clip_count=sum(len(clips) for clips in clips_by_scene.values()),
    )

    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"Output root exists and is not a directory: {output}")
        if any(output.iterdir()):
            _verify_existing_view(output, manifest)
            return manifest
    else:
        output.mkdir(parents=True)

    for scene, clip in selected:
        destination = output / split / scene / clip.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(clip, target_is_directory=True)

    manifest_path = output / MANIFEST_NAME
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    temporary.replace(manifest_path)
    _verify_existing_view(output, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_balanced_sft_view(
        args.source_root,
        args.output_root,
        split=args.split,
        total_clips=args.total_clips,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
