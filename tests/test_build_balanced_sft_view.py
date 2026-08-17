from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.tools.build_balanced_sft_view import (
    MANIFEST_NAME,
    build_balanced_sft_view,
    scene_round_robin_selection,
)
from scripts.tools.train_heatmap_joint_pilot import sft_dataset_contract


def _source_tree(root: Path) -> Path:
    source = root / "source"
    for scene, count in (("scene_b", 3), ("scene_a", 3), ("scene_c", 2)):
        for index in range(count):
            (source / "train" / scene / f"clip_{index:06d}").mkdir(parents=True)
    return source


def test_scene_round_robin_selection_is_deterministic_and_exact(tmp_path):
    source = _source_tree(tmp_path)
    clips = {
        scene.name: sorted(path.resolve() for path in scene.glob("clip_*"))
        for scene in (source / "train").iterdir()
    }
    selected = scene_round_robin_selection(clips, 5)
    assert [(scene, clip.name) for scene, clip in selected] == [
        ("scene_a", "clip_000000"),
        ("scene_b", "clip_000000"),
        ("scene_c", "clip_000000"),
        ("scene_a", "clip_000001"),
        ("scene_b", "clip_000001"),
    ]
    with pytest.raises(ValueError, match="only has 8"):
        scene_round_robin_selection(clips, 9)


def test_build_balanced_view_creates_verified_idempotent_symlinks(tmp_path):
    source = _source_tree(tmp_path)
    output = tmp_path / "balanced"
    first = build_balanced_sft_view(source, output, total_clips=5)
    second = build_balanced_sft_view(source, output, total_clips=5)
    assert first == second
    assert first["total_clips"] == 5
    assert first["scene_count"] == 3
    assert first["per_scene_counts"] == {"scene_a": 2, "scene_b": 2, "scene_c": 1}
    assert len(first["selected_clip_identity_sha256"]) == 64

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == first
    for selected in first["selected_clips"]:
        link = output / selected["identity"]
        assert link.is_symlink()
        assert link.resolve(strict=True) == Path(selected["source_resolved"])

    dataset = SimpleNamespace(
        root=output,
        clips=sorted((output / "train").glob("*/clip_*")),
    )
    contract = sft_dataset_contract(dataset)
    assert contract["clip_count"] == 5
    assert contract["scene_count"] == 3
    assert contract["balanced_view_manifest"]["source_root"] == str(source.resolve())


def test_existing_manifest_or_symlink_mismatch_is_refused(tmp_path):
    source = _source_tree(tmp_path)
    output = tmp_path / "balanced"
    manifest = build_balanced_sft_view(source, output, total_clips=5)
    manifest_path = output / MANIFEST_NAME

    corrupted = dict(manifest)
    corrupted["selected_clip_identity_sha256"] = "bad"
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest does not match"):
        build_balanced_sft_view(source, output, total_clips=5)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    extra = output / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected or missing content"):
        build_balanced_sft_view(source, output, total_clips=5)
    extra.unlink()

    first = output / manifest["selected_clips"][0]["identity"]
    first.unlink()
    first.symlink_to(source / "train" / "scene_b" / "clip_000002", target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink mismatch"):
        build_balanced_sft_view(source, output, total_clips=5)


def test_balanced_view_refuses_nonempty_unmanaged_or_nested_output(tmp_path):
    source = _source_tree(tmp_path)
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    (unmanaged / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="has no"):
        build_balanced_sft_view(source, unmanaged, total_clips=2)
    with pytest.raises(ValueError, match="must be isolated"):
        build_balanced_sft_view(source, source / "derived", total_clips=2)
