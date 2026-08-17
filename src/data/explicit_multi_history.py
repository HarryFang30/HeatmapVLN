"""Strict manifest-backed history selection for multi-history diagnostics.

The normal :class:`VLNSlidingWindowDataset` sampling path is intentionally
left untouched.  This module provides an opt-in subclass which replaces only
the history frame indices for records named by a Task-3.6 manifest.  The base
dataset still performs RGB/depth loading, augmentation, pose conversion, and
heatmap construction, so a diagnostic cannot accidentally drift onto a
different preprocessing path.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sliding_window_dataset import VLNSlidingWindowDataset

MULTI_HISTORY_SCHEMA = "task36a_multi_history_selection_v1"


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value with a single canonical representation."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_identity(record: dict[str, Any]) -> str:
    history = ",".join(str(int(frame)) for frame in record["history_frames"])
    return f"{record['relative_clip']}:current={int(record['current_frame'])}:history={history}"


def verify_selection_record(record: dict[str, Any], *, expected_k: int | None = None) -> None:
    """Validate identity, ordering, and per-record hashes before any I/O."""
    required = {
        "sample_id",
        "relative_clip",
        "scene",
        "current_frame",
        "history_frames",
        "canonical_history_frames",
        "slot_permutation",
        "model_inputs",
        "loader_alignment",
        "label_metadata",
        "label_metadata_sha256",
        "record_sha256",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Multi-history record is missing fields: {missing}")

    history = [int(frame) for frame in record["history_frames"]]
    canonical = [int(frame) for frame in record["canonical_history_frames"]]
    permutation = [int(index) for index in record["slot_permutation"]]
    current = int(record["current_frame"])
    if expected_k is not None and len(history) != expected_k:
        raise ValueError(f"Expected K={expected_k}, got {len(history)} for {record['sample_id']}")
    if not history or len(history) != len(canonical) or len(history) != len(permutation):
        raise ValueError(f"Inconsistent history lengths for {record['sample_id']}")
    if len(set(history)) != len(history) or any(frame < 0 or frame >= current for frame in history):
        raise ValueError(f"History frames must be unique and precede current for {record['sample_id']}")
    if sorted(permutation) != list(range(len(history))):
        raise ValueError(f"slot_permutation is not a permutation for {record['sample_id']}")
    if history != [canonical[index] for index in permutation]:
        raise ValueError(f"slot_permutation does not reproduce history_frames for {record['sample_id']}")
    if str(record["sample_id"]) != record_identity(record):
        raise ValueError(f"sample_id does not match exact frames for {record['sample_id']}")

    expected_model_inputs = {
        "current_rgb": "current_rgb_panorama",
        "history_rgb": "ordered_history_rgb_observations",
    }
    if record["model_inputs"] != expected_model_inputs:
        raise ValueError(f"model_inputs are not RGB-only for {record['sample_id']}")
    alignment = record["loader_alignment"]
    current_alignment = alignment.get("current", {})
    history_alignment = alignment.get("history", [])
    if (
        str(current_alignment.get("relative_clip")) != str(record["relative_clip"])
        or int(current_alignment.get("frame_index", -1)) != current
        or [int(item.get("frame_index", -1)) for item in history_alignment] != history
        or [int(item.get("loader_position", -1)) for item in history_alignment]
        != list(range(len(history)))
        or [int(item.get("canonical_index", -1)) for item in history_alignment]
        != permutation
        or any(
            str(item.get("relative_clip")) != str(record["relative_clip"])
            for item in history_alignment
        )
        or alignment.get("usage") != "loader_and_label_alignment_only_never_model_input"
    ):
        raise ValueError(f"loader_alignment does not match exact frames for {record['sample_id']}")
    label_anchors = record["label_metadata"].get("anchors", [])
    if (
        [int(item.get("history_frame", -1)) for item in label_anchors] != history
        or [int(item.get("slot", -1)) for item in label_anchors] != list(range(len(history)))
    ):
        raise ValueError(f"label anchors do not match exact frames for {record['sample_id']}")
    actual_label_hash = canonical_sha256(record["label_metadata"])
    if str(record["label_metadata_sha256"]) != actual_label_hash:
        raise ValueError(
            f"label_metadata_sha256 mismatch for {record['sample_id']}: "
            f"expected={record['label_metadata_sha256']} actual={actual_label_hash}"
        )

    unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
    actual_hash = canonical_sha256(unhashed)
    if str(record["record_sha256"]) != actual_hash:
        raise ValueError(
            f"record_sha256 mismatch for {record['sample_id']}: "
            f"expected={record['record_sha256']} actual={actual_hash}"
        )


def load_multi_history_records(
    manifest_path: str | Path,
    split: str,
    *,
    expected_source_inventory_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and strictly verify one split of a Task-3.6 manifest."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MULTI_HISTORY_SCHEMA:
        raise ValueError(
            f"Unsupported multi-history schema: {manifest.get('schema_version')!r}"
        )
    expected_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str):
        raise ValueError("Manifest is missing manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_manifest_hash = canonical_sha256(unhashed)
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError(
            "manifest_sha256 mismatch: "
            f"expected={expected_manifest_hash} actual={actual_manifest_hash}"
        )

    inventory = manifest.get("source_inventory_contract")
    if not isinstance(inventory, dict):
        raise ValueError("Manifest is missing source_inventory_contract")
    inventory_records = inventory.get("records")
    if not isinstance(inventory_records, list):
        raise ValueError("source_inventory_contract.records must be a list")
    ordered_inventory = sorted(
        inventory_records,
        key=lambda record: str(record["relative_clip"]),
    )
    inventory_rows = sorted(
        (
            f"{record['relative_clip']}\t{record['scene_id']}\t{record['episode_id']}\t"
            f"{record['num_frames']}\t{record['seed']}"
        )
        for record in ordered_inventory
    )
    inventory_payload = "\n".join(inventory_rows) + ("\n" if inventory_rows else "")
    actual_inventory_hash = hashlib.sha256(inventory_payload.encode()).hexdigest()
    if inventory.get("inventory_sha256") != actual_inventory_hash:
        raise ValueError(
            "source inventory hash mismatch: "
            f"expected={inventory.get('inventory_sha256')} actual={actual_inventory_hash}"
        )
    if int(inventory.get("clips", -1)) != len(ordered_inventory):
        raise ValueError("source inventory clip count mismatch")
    if expected_source_inventory_sha256 is not None:
        expected_inventory = expected_source_inventory_sha256.strip().lower()
        if actual_inventory_hash != expected_inventory:
            raise ValueError(
                "source inventory does not match requested snapshot: "
                f"expected={expected_inventory} actual={actual_inventory_hash}"
            )

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"Split {split!r} is absent from {path}")
    split_payload = splits[split]
    records = split_payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Split {split!r} records must be a list")
    expected_k = int(manifest["selection_parameters"]["num_history"])
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Split {split!r} contains a non-object record")
        verify_selection_record(record, expected_k=expected_k)

    identities = [str(record["sample_id"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Split {split!r} contains duplicate sample identities")
    expected_hash = split_payload["selection_manifest"]["record_identity_sha256"]
    actual_hash = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(
            f"Split {split!r} identity hash mismatch: expected={expected_hash} actual={actual_hash}"
        )
    selection = split_payload["selection_manifest"]
    if int(selection.get("sample_count", -1)) != len(records):
        raise ValueError(f"Split {split!r} sample_count does not match records")
    record_hashes = [str(record["record_sha256"]) for record in records]
    ordered_record_hash = hashlib.sha256("\n".join(record_hashes).encode()).hexdigest()
    if selection.get("ordered_record_sha256") != ordered_record_hash:
        raise ValueError(f"Split {split!r} ordered_record_sha256 mismatch")
    if selection.get("records_sha256") != canonical_sha256(records):
        raise ValueError(f"Split {split!r} records_sha256 mismatch")
    scenes = sorted({str(record["scene"]) for record in records})
    if selection.get("scenes") != scenes:
        raise ValueError(f"Split {split!r} scene list mismatch")
    return records, manifest


def verify_runtime_targets(
    sample: dict[str, Any],
    record: dict[str, Any],
    runtime_history_frames: Sequence[int],
) -> None:
    """Fail if recomputed visibility/peaks drift from manifest labels."""
    if "gt_visibility" not in sample or "heatmap" not in sample:
        raise RuntimeError(f"Runtime sample is missing heatmap labels for {record['sample_id']}")
    visibility = torch.as_tensor(sample["gt_visibility"]).detach().cpu()
    heatmaps = torch.as_tensor(sample["heatmap"]).detach().cpu()
    if visibility.ndim != 2 or heatmaps.ndim != 4:
        raise RuntimeError(
            f"Runtime label rank mismatch for {record['sample_id']}: "
            f"visibility={tuple(visibility.shape)} heatmap={tuple(heatmaps.shape)}"
        )
    if tuple(heatmaps.shape[:2]) != tuple(visibility.shape):
        raise RuntimeError(f"Runtime visibility/heatmap shape mismatch for {record['sample_id']}")
    if int(visibility.shape[0]) != len(runtime_history_frames):
        raise RuntimeError(f"Runtime K mismatch for {record['sample_id']}")

    labels_by_frame = {
        int(anchor["history_frame"]): anchor
        for anchor in record["label_metadata"]["anchors"]
    }
    view_count = int(visibility.shape[1])
    for runtime_slot, frame in enumerate(runtime_history_frames):
        if int(frame) not in labels_by_frame:
            raise RuntimeError(
                f"Runtime frame {frame} is absent from label metadata for {record['sample_id']}"
            )
        anchor = labels_by_frame[int(frame)]
        target_views = anchor.get("target_views", [])
        if len(target_views) != view_count:
            raise RuntimeError(f"Manifest view count mismatch for {record['sample_id']}")
        for view_index, expected in enumerate(target_views):
            actual_visible = bool(float(visibility[runtime_slot, view_index].item()) > 0.5)
            expected_visible = bool(expected["visible"])
            if actual_visible != expected_visible:
                raise RuntimeError(
                    "Runtime visibility drift: "
                    f"sample={record['sample_id']} frame={frame} view={expected['view']} "
                    f"expected={expected_visible} actual={actual_visible}"
                )
            if not expected_visible:
                continue
            heatmap = heatmaps[runtime_slot, view_index]
            width = int(heatmap.shape[-1])
            flat_index = int(heatmap.reshape(-1).argmax().item())
            actual_x = flat_index % width
            actual_y = flat_index // width
            if actual_x != int(expected["x"]) or actual_y != int(expected["y"]):
                raise RuntimeError(
                    "Runtime heatmap peak drift: "
                    f"sample={record['sample_id']} frame={frame} view={expected['view']} "
                    f"expected=({expected['x']},{expected['y']}) "
                    f"actual=({actual_x},{actual_y})"
                )


class ExplicitMultiHistoryDataset(VLNSlidingWindowDataset):
    """Opt-in dataset that reuses the base pipeline with exact manifest anchors.

    Each DataLoader worker owns a separate dataset instance, and base
    ``__getitem__`` is synchronous.  A short-lived active-history guard is
    therefore sufficient to override ``_sample_history_indices`` without
    mutating global state or the default dataset class.  Pose tensors are
    removed from returned samples by default; ``drop_pose_inputs=False`` is an
    explicit opt-in reserved for geometry baselines.
    """

    def __init__(
        self,
        *args: Any,
        selection_records: Sequence[dict[str, Any]],
        slot_seed: int = 42,
        reshuffle_slots_each_epoch: bool = False,
        drop_pose_inputs: bool = True,
        verify_runtime_labels: bool = True,
        **kwargs: Any,
    ) -> None:
        records = [dict(record) for record in selection_records]
        if not records:
            raise ValueError("Explicit multi-history selection must not be empty")
        expected_k = len(records[0].get("history_frames", []))
        if expected_k <= 0:
            raise ValueError("Explicit multi-history selection has K=0")
        for record in records:
            verify_selection_record(record, expected_k=expected_k)

        kwargs["num_history_sample"] = expected_k
        kwargs["clip_level_sampling"] = False
        super().__init__(*args, **kwargs)

        clip_lookup: dict[str, int] = {}
        for clip_index, clip_path in enumerate(self.clips):
            try:
                relative = Path(clip_path).relative_to(self.root).as_posix()
            except ValueError:
                relative = Path(clip_path).as_posix()
            clip_lookup[relative] = clip_index

        explicit_index: list[tuple[int, int]] = []
        canonical_frames: list[tuple[int, ...]] = []
        initial_frames: list[tuple[int, ...]] = []
        identities: list[str] = []
        for record in records:
            relative_clip = str(record["relative_clip"])
            if relative_clip not in clip_lookup:
                raise ValueError(f"Manifest clip is absent from dataset root: {relative_clip}")
            clip_index = clip_lookup[relative_clip]
            current = int(record["current_frame"])
            meta = self._load_meta(clip_index)
            frame_count = int(meta["num_frames"])
            history = tuple(int(frame) for frame in record["history_frames"])
            if current >= frame_count or any(frame >= frame_count for frame in history):
                raise ValueError(
                    f"Manifest frame exceeds clip length={frame_count}: {record['sample_id']}"
                )
            if Path(self.clips[clip_index]).parent.name != str(record["scene"]):
                raise ValueError(f"Manifest scene mismatch for {record['sample_id']}")
            explicit_index.append((clip_index, current))
            canonical_frames.append(
                tuple(int(frame) for frame in record["canonical_history_frames"])
            )
            initial_frames.append(history)
            identities.append(str(record["sample_id"]))

        self.sample_index = explicit_index
        self._explicit_canonical_frames = canonical_frames
        self._explicit_initial_frames = initial_frames
        self._explicit_history_frames = list(initial_frames)
        self._explicit_identities = identities
        self._explicit_records = records
        self._explicit_slot_seed = int(slot_seed)
        self._explicit_reshuffle_slots_each_epoch = bool(reshuffle_slots_each_epoch)
        self._explicit_drop_pose_inputs = bool(drop_pose_inputs)
        self._explicit_verify_runtime_labels = bool(verify_runtime_labels)
        self._active_explicit_history: tuple[int, ...] | None = None
        self._active_explicit_current: int | None = None
        self._explicit_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Optionally reshuffle slots while keeping each anchor set invariant."""
        self._explicit_epoch = int(epoch)
        if not self._explicit_reshuffle_slots_each_epoch:
            self._explicit_history_frames = list(self._explicit_initial_frames)
            return
        shuffled: list[tuple[int, ...]] = []
        for identity, canonical in zip(
            self._explicit_identities,
            self._explicit_canonical_frames,
            strict=True,
        ):
            seed_material = (
                f"{self._explicit_slot_seed}\n{self._explicit_epoch}\n{identity}"
            ).encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
            permutation = list(range(len(canonical)))
            rng.shuffle(permutation)
            shuffled.append(tuple(canonical[index] for index in permutation))
        self._explicit_history_frames = shuffled

    def _sample_history_indices(self, start: int, end: int, num_samples: int) -> np.ndarray:
        if self._active_explicit_history is None:
            return super()._sample_history_indices(start, end, num_samples)
        if self._active_explicit_current != int(end):
            raise RuntimeError(
                "Explicit history override current-frame mismatch: "
                f"active={self._active_explicit_current} requested={end}"
            )
        if len(self._active_explicit_history) != int(num_samples):
            raise RuntimeError(
                "Explicit history override K mismatch: "
                f"active={len(self._active_explicit_history)} requested={num_samples}"
            )
        return np.asarray(self._active_explicit_history, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._active_explicit_history is not None:
            raise RuntimeError("ExplicitMultiHistoryDataset does not support re-entrant __getitem__")
        _clip_index, current = self.sample_index[index]
        exact_history = self._explicit_history_frames[index]
        self._active_explicit_history = exact_history
        self._active_explicit_current = int(current)
        try:
            sample = super().__getitem__(index)
        finally:
            self._active_explicit_history = None
            self._active_explicit_current = None
        if getattr(self, "_explicit_verify_runtime_labels", True):
            verify_runtime_targets(sample, self._explicit_records[index], exact_history)
        if getattr(self, "_explicit_drop_pose_inputs", True):
            for key in ("history_rel_poses", "history_poses", "current_pose"):
                sample.pop(key, None)
        return sample
