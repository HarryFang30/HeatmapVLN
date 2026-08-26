#!/usr/bin/env python3
"""Fail-closed validator for a HeatmapVLN trajectory-DAgger collection.

Expected collection layout::

    collection_manifest.json
    collection_progress.jsonl
    episodes/<episode_key>/commit.json
    episodes/<episode_key>/episode.tar

The validator never extracts tar files. It checks path containment, the
manifest/progress/commit chain, archive hashes and safe members, every JPEG,
and the native trajectory tensor shape [sample_count, 32, 3]. The collection
and every referenced path must remain below /mnt/afs/liwenhao/agent/370910109.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import uuid
import warnings
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image


FJL_ROOT = Path("/mnt/afs/liwenhao/agent/370910109")
HARD_CAPACITY_BYTES = 300_000_000_000
MANIFEST_SCHEMA = "heatmapvln-trajectory-dagger-collection-v1"
COMMIT_SCHEMA = "heatmapvln-trajectory-dagger-episode-commit-v1"
SAMPLE_SCHEMA = "heatmapvln-trajectory-dagger-sample-v1"
SEAL_SUMMARY_SCHEMA = "heatmapvln-trajectory-dagger-seal-summary-v1"
CARDINAL_VIEWS = ("front", "right", "back", "left")
REQUIRED_TAR_MEMBERS = {
    "episode.json",
    "frames.jsonl",
    "samples.jsonl",
    "arrays/trajectories.npy",
    "arrays/oracle_future_poses.npy",
    "arrays/oracle_future_offsets.npy",
}
EPISODE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
FRAME_JPEG_RE = re.compile(r"^frames/([0-9]{6})_(front|right|back|left)\.jpg$")
LOOKDOWN_JPEG_RE = re.compile(r"^lookdown/([0-9]{6})\.jpg$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ValidationError(RuntimeError):
    """Raised on the first invalid or unsafe collection condition."""


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_json_loads(text: str, context: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"invalid JSON in {context}: {exc}") from exc


def _resolve_roots(raw_collection: str) -> tuple[Path, Path]:
    try:
        fjl_root = FJL_ROOT.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"FJL root is unavailable: {FJL_ROOT}: {exc}") from exc
    candidate = Path(raw_collection).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        collection = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"collection root does not exist: {candidate}: {exc}") from exc
    if not collection.is_dir():
        raise ValidationError(f"collection root is not a directory: {collection}")
    if collection == fjl_root or not _is_within(collection, fjl_root):
        raise ValidationError(f"collection root must be strictly below FJL root: {collection}")
    return fjl_root, collection


def _require_child(path: Path, collection: Path, kind: str) -> Path:
    if path.is_symlink():
        raise ValidationError(f"{kind} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"missing {kind}: {path}: {exc}") from exc
    if not _is_within(resolved, collection):
        raise ValidationError(f"{kind} escapes collection root: {resolved}")
    if not resolved.is_file():
        raise ValidationError(f"{kind} is not a regular file: {resolved}")
    return resolved


def _scan_tree(collection: Path, limit: int) -> tuple[int, dict[str, tuple[int, int, int]], set[Path]]:
    total = 0
    snapshot: dict[str, tuple[int, int, int]] = {}
    files: set[Path] = set()
    stack = [collection]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValidationError(f"cannot scan collection directory {directory}: {exc}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValidationError(f"cannot stat collection entry {entry_path}: {exc}") from exc
            if entry.is_symlink():
                raise ValidationError(f"symlinks are forbidden in a collection: {entry_path}")
            relative = entry_path.relative_to(collection).as_posix()
            if entry.is_dir(follow_symlinks=False):
                snapshot[relative + "/"] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                stack.append(entry_path)
            elif entry.is_file(follow_symlinks=False):
                total += stat.st_size
                if total > limit:
                    raise ValidationError(
                        f"collection size {total} exceeds capacity limit {limit} bytes"
                    )
                snapshot[relative] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                files.add(entry_path)
            else:
                raise ValidationError(f"non-regular collection entry is forbidden: {entry_path}")
    return total, snapshot, files


def _read_text(path: Path, context: str, max_bytes: int) -> str:
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValidationError(f"{context} has invalid size {size} bytes: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {context} {path}: {exc}") from exc


def _load_json_file(path: Path, context: str, max_bytes: int = 16 * 1024 * 1024) -> Any:
    return _strict_json_loads(_read_text(path, context, max_bytes), context)


def _load_jsonl_text(text: str, context: str) -> list[dict[str, Any]]:
    if not text.endswith("\n"):
        raise ValidationError(f"{context} lacks a final newline (possibly a partial commit)")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValidationError(f"blank line in {context} at line {line_number}")
        row = _strict_json_loads(line, f"{context}:{line_number}")
        if not isinstance(row, dict):
            raise ValidationError(f"{context}:{line_number} must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValidationError(f"{context} contains no committed episodes")
    return rows


def _positive_int(value: Any, context: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{context} must be an integer, got {value!r}")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValidationError(f"{context} must be >= {minimum}, got {value}")
    return value


def _safe_episode_key(value: Any, context: str) -> str:
    if not isinstance(value, str) or not EPISODE_KEY_RE.fullmatch(value):
        raise ValidationError(f"{context} is not a safe episode key: {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_tar_member_name(member: tarfile.TarInfo, context: str) -> str:
    name = member.name
    if not name or "\\" in name or name.startswith("/"):
        raise ValidationError(f"unsafe tar member name in {context}: {name!r}")
    pure = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValidationError(f"unsafe tar member name in {context}: {name!r}")
    canonical = pure.as_posix()
    comparable = name.rstrip("/") if member.isdir() else name
    if comparable != canonical:
        raise ValidationError(f"non-canonical tar member name in {context}: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValidationError(f"links/devices are forbidden in {context}: {name}")
    return canonical


def _read_tar_member(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    name: str,
    context: str,
    max_bytes: int,
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise ValidationError(f"missing regular tar member {name} in {context}")
    if member.size <= 0 or member.size > max_bytes:
        raise ValidationError(
            f"tar member {name} in {context} has invalid size {member.size} bytes"
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValidationError(f"cannot read tar member {name} in {context}")
    try:
        data = extracted.read(max_bytes + 1)
    except OSError as exc:
        raise ValidationError(f"cannot read tar member {name} in {context}: {exc}") from exc
    if len(data) != member.size or len(data) > max_bytes:
        raise ValidationError(f"short or oversized tar member {name} in {context}")
    return data


def _load_tar_jsonl(data: bytes, context: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"{context} is not UTF-8: {exc}") from exc
    return _load_jsonl_text(text, context)


def _decode_utf8(data: bytes, context: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"{context} is not UTF-8: {exc}") from exc


def _load_npy(data: bytes, context: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise ValidationError(f"invalid NPY payload in {context}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"{context} must contain one ndarray, not an NPZ archive")
    if value.dtype.hasobject or value.dtype.kind not in "biuf":
        raise ValidationError(f"{context} has unsupported dtype {value.dtype}")
    return value


def _validate_finite(array: np.ndarray, context: str) -> None:
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise ValidationError(f"{context} is not numeric: {exc}") from exc
    if not finite:
        raise ValidationError(f"{context} contains NaN or infinity")


def _validate_jpeg(data: bytes, context: str) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValidationError(f"{context} is not a JPEG bitstream")
                width, height = image.size
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"JPEG decode failed for {context}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValidationError(f"JPEG has invalid dimensions for {context}: {width}x{height}")
    return width, height


def _validate_manifest(manifest: Any, max_bytes: int) -> tuple[str, int, bool]:
    if not isinstance(manifest, dict):
        raise ValidationError("collection_manifest.json must be a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationError(
            f"manifest schema must be {MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    fingerprint = manifest.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not SHA256_RE.fullmatch(fingerprint)
        or fingerprint != fingerprint.lower()
    ):
        raise ValidationError("manifest fingerprint must be a lowercase SHA-256 hex string")
    if not isinstance(manifest.get("contract"), dict) or not manifest["contract"]:
        raise ValidationError("manifest contract must be a non-empty object")
    if not isinstance(manifest.get("ready"), bool):
        raise ValidationError("manifest ready must be boolean")
    ready = manifest["ready"]
    required_keys = {"schema", "contract", "capacity", "fingerprint", "created_at", "ready"}
    if ready:
        required_keys |= {"sealed_at", "summary"}
    if set(manifest) != required_keys:
        raise ValidationError(
            "manifest fields are not canonical; "
            f"missing={sorted(required_keys - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - required_keys)}"
        )
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValidationError("manifest created_at must be a non-empty ISO-8601 timestamp")
    try:
        parsed_created_at = dt.datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValidationError("manifest created_at is not ISO-8601") from exc
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
        raise ValidationError("manifest created_at must include a timezone")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, dict) or set(capacity) != {
        "hard_capacity_bytes",
        "commit_ceiling_bytes",
    }:
        raise ValidationError("manifest capacity must contain exactly the two capacity limits")
    hard_capacity = _positive_int(
        capacity.get("hard_capacity_bytes"), "manifest capacity.hard_capacity_bytes"
    )
    commit_ceiling = _positive_int(
        capacity.get("commit_ceiling_bytes"), "manifest capacity.commit_ceiling_bytes"
    )
    if hard_capacity > HARD_CAPACITY_BYTES:
        raise ValidationError(
            f"manifest hard capacity {hard_capacity} exceeds 300 GB ({HARD_CAPACITY_BYTES})"
        )
    if commit_ceiling >= hard_capacity:
        raise ValidationError("manifest commit ceiling must be below its hard capacity")

    identity = {
        "schema": manifest["schema"],
        "contract": manifest["contract"],
        "capacity": manifest["capacity"],
    }
    expected_fingerprint = _sha256_bytes(_canonical_json_bytes(identity))
    if fingerprint.lower() != expected_fingerprint:
        raise ValidationError(
            "manifest fingerprint does not match canonical schema/contract/capacity identity"
        )

    if ready:
        sealed_at = manifest.get("sealed_at")
        if not isinstance(sealed_at, str) or not sealed_at:
            raise ValidationError("sealed manifest requires a non-empty sealed_at")
        try:
            parsed_sealed_at = dt.datetime.fromisoformat(sealed_at)
        except ValueError as exc:
            raise ValidationError("sealed manifest sealed_at is not ISO-8601") from exc
        if parsed_sealed_at.tzinfo is None or parsed_sealed_at.utcoffset() is None:
            raise ValidationError("sealed manifest sealed_at must include a timezone")
        summary = manifest.get("summary")
        if not isinstance(summary, dict) or summary.get("schema") != SEAL_SUMMARY_SCHEMA:
            raise ValidationError("sealed manifest requires a valid summary object")
    elif "sealed_at" in manifest or "summary" in manifest:
        raise ValidationError("unsealed manifest may not contain sealed_at or summary")

    effective_limit = min(max_bytes, hard_capacity, HARD_CAPACITY_BYTES)
    return fingerprint, effective_limit, ready


def _validate_commit(marker: dict[str, Any], fingerprint: str, context: str) -> tuple[str, int, int]:
    if marker.get("schema") != COMMIT_SCHEMA:
        raise ValidationError(f"{context} has invalid schema {marker.get('schema')!r}")
    episode_key = _safe_episode_key(marker.get("episode_key"), f"{context}.episode_key")
    if marker.get("manifest_fingerprint") != fingerprint:
        raise ValidationError(f"{context} manifest_fingerprint does not match manifest")
    if marker.get("tar_file") != "episode.tar":
        raise ValidationError(f"{context}.tar_file must be exactly 'episode.tar'")
    tar_sha256 = marker.get("tar_sha256")
    if not isinstance(tar_sha256, str) or not SHA256_RE.fullmatch(tar_sha256):
        raise ValidationError(f"{context}.tar_sha256 must be a SHA-256 hex string")
    _positive_int(marker.get("tar_bytes"), f"{context}.tar_bytes")
    sample_count = _positive_int(marker.get("sample_count"), f"{context}.sample_count")
    frame_count = _positive_int(marker.get("frame_count"), f"{context}.frame_count")
    sample_keys = marker.get("sample_keys")
    if (
        not isinstance(sample_keys, list)
        or len(sample_keys) != sample_count
        or any(not isinstance(key, str) or not key for key in sample_keys)
        or len(set(sample_keys)) != len(sample_keys)
    ):
        raise ValidationError(f"{context}.sample_keys must contain {sample_count} unique strings")
    return episode_key, sample_count, frame_count


def _validate_episode_archive(
    tar_path: Path,
    marker: dict[str, Any],
    episode_key: str,
    sample_count: int,
    frame_count: int,
    fingerprint: str,
    capacity_limit: int,
    expected_view_size: tuple[int, int] | None = None,
    expected_lookdown_size: tuple[int, int] | None = None,
    native_policy_contract: dict[str, Any] | None = None,
) -> dict[str, int]:
    context = f"episode {episode_key}"
    actual_tar_bytes = tar_path.stat().st_size
    if actual_tar_bytes != marker["tar_bytes"]:
        raise ValidationError(
            f"{context} tar size mismatch: commit={marker['tar_bytes']} actual={actual_tar_bytes}"
        )
    actual_sha256 = _sha256_file(tar_path)
    if actual_sha256.lower() != marker["tar_sha256"].lower():
        raise ValidationError(f"{context} tar SHA-256 mismatch")

    try:
        archive = tarfile.open(tar_path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise ValidationError(f"cannot open uncompressed tar for {context}: {exc}") from exc

    with archive:
        members: dict[str, tarfile.TarInfo] = {}
        payload_bytes = 0
        for member in archive.getmembers():
            name = _safe_tar_member_name(member, context)
            if name in members:
                raise ValidationError(f"duplicate tar member in {context}: {name}")
            members[name] = member
            if member.isfile():
                payload_bytes += member.size
                if payload_bytes > capacity_limit:
                    raise ValidationError(f"unpacked payload for {context} exceeds capacity limit")
        missing = REQUIRED_TAR_MEMBERS - set(members)
        if missing:
            raise ValidationError(f"{context} tar lacks required members: {sorted(missing)}")

        episode_data = _strict_json_loads(
            _decode_utf8(
                _read_tar_member(
                    archive, members, "episode.json", context, max_bytes=16 * 1024 * 1024
                ),
                f"{context}/episode.json",
            ),
            f"{context}/episode.json",
        )
        if not isinstance(episode_data, dict):
            raise ValidationError(f"{context}/episode.json must be an object")
        if episode_data.get("schema") != MANIFEST_SCHEMA:
            raise ValidationError(f"{context}/episode.json has invalid schema")
        if episode_data.get("episode_key") != episode_key:
            raise ValidationError(f"{context}/episode.json episode_key mismatch")
        if episode_data.get("manifest_fingerprint") != fingerprint:
            raise ValidationError(f"{context}/episode.json manifest fingerprint mismatch")

        frames = _load_tar_jsonl(
            _read_tar_member(
                archive, members, "frames.jsonl", context, max_bytes=256 * 1024 * 1024
            ),
            f"{context}/frames.jsonl",
        )
        samples = _load_tar_jsonl(
            _read_tar_member(
                archive, members, "samples.jsonl", context, max_bytes=256 * 1024 * 1024
            ),
            f"{context}/samples.jsonl",
        )
        if len(frames) != frame_count:
            raise ValidationError(
                f"{context} frame count mismatch: commit={frame_count}, rows={len(frames)}"
            )
        if len(samples) != sample_count:
            raise ValidationError(
                f"{context} sample count mismatch: commit={sample_count}, rows={len(samples)}"
            )

        frame_ids: set[int] = set()
        lookdown_frame_ids: set[int] = set()
        expected_jpegs: set[str] = set()
        for row_number, frame in enumerate(frames, start=1):
            prefix = f"{context}/frames.jsonl:{row_number}"
            required_fields = {
                "frame_id",
                "primitive_step",
                "system2_call_index",
                "pose",
                "views",
                "lookdown",
            }
            missing_fields = required_fields - set(frame)
            if missing_fields:
                raise ValidationError(f"{prefix} lacks fields: {sorted(missing_fields)}")
            frame_id = _positive_int(
                frame.get("frame_id"), f"{prefix}.frame_id", True
            )
            if frame_id in frame_ids:
                raise ValidationError(f"duplicate frame_id {frame_id} in {context}")
            if frame_id > 999999:
                raise ValidationError(f"frame_id exceeds six-digit tar naming contract: {frame_id}")
            frame_ids.add(frame_id)
            _positive_int(frame.get("primitive_step"), f"{prefix}.primitive_step", True)
            system2_call_index = frame.get("system2_call_index")
            if system2_call_index is not None:
                _positive_int(system2_call_index, f"{prefix}.system2_call_index", True)
            try:
                pose = np.asarray(frame.get("pose"), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{prefix}.pose is not numeric: {exc}") from exc
            if pose.shape != (4, 4):
                raise ValidationError(f"{prefix}.pose must have shape [4,4], got {pose.shape}")
            _validate_finite(pose, f"{prefix}.pose")
            views = frame.get("views")
            if not isinstance(views, dict) or set(views) != set(CARDINAL_VIEWS):
                raise ValidationError(
                    f"{prefix}.views must map exactly {list(CARDINAL_VIEWS)}"
                )
            for view in CARDINAL_VIEWS:
                expected_path = f"frames/{frame_id:06d}_{view}.jpg"
                if views.get(view) != expected_path:
                    raise ValidationError(
                        f"{prefix}.views[{view!r}] must be exactly {expected_path!r}"
                    )
                expected_jpegs.add(expected_path)
            lookdown = frame.get("lookdown")
            if lookdown is not None:
                expected_lookdown = f"lookdown/{frame_id:06d}.jpg"
                if lookdown != expected_lookdown:
                    raise ValidationError(
                        f"{prefix}.lookdown must be null or exactly {expected_lookdown!r}"
                    )
                expected_jpegs.add(expected_lookdown)
                lookdown_frame_ids.add(frame_id)

        expected_members = REQUIRED_TAR_MEMBERS | expected_jpegs
        actual_members = set(members)
        if actual_members != expected_members:
            missing_members = sorted(expected_members - actual_members)
            unexpected_members = sorted(actual_members - expected_members)
            raise ValidationError(
                f"{context} tar member set is not official; "
                f"missing={missing_members[:5]}, unexpected={unexpected_members[:5]}"
            )
        actual_jpegs = expected_jpegs
        for name in sorted(expected_jpegs):
            jpeg_data = _read_tar_member(
                archive, members, name, context, max_bytes=32 * 1024 * 1024
            )
            actual_size = _validate_jpeg(jpeg_data, f"{context}/{name}")
            expected_size = (
                expected_lookdown_size
                if LOOKDOWN_JPEG_RE.fullmatch(name)
                else expected_view_size
            )
            if expected_size is not None and actual_size != expected_size:
                raise ValidationError(
                    f"{context}/{name} dimensions {actual_size} "
                    f"do not match collection contract {expected_size}"
                )

        trajectories = _load_npy(
            _read_tar_member(
                archive,
                members,
                "arrays/trajectories.npy",
                context,
                max_bytes=512 * 1024 * 1024,
            ),
            f"{context}/arrays/trajectories.npy",
        )
        if trajectories.shape != (sample_count, 32, 3):
            raise ValidationError(
                f"{context} trajectories shape must be ({sample_count}, 32, 3), "
                f"got {trajectories.shape}"
            )
        _validate_finite(trajectories, f"{context} trajectories")

        future_poses = _load_npy(
            _read_tar_member(
                archive,
                members,
                "arrays/oracle_future_poses.npy",
                context,
                max_bytes=512 * 1024 * 1024,
            ),
            f"{context}/arrays/oracle_future_poses.npy",
        )
        if future_poses.ndim != 3 or future_poses.shape[1:] != (4, 4):
            raise ValidationError(
                f"{context} oracle_future_poses shape must be [N,4,4], got {future_poses.shape}"
            )
        _validate_finite(future_poses, f"{context} oracle_future_poses")

        future_offsets = _load_npy(
            _read_tar_member(
                archive,
                members,
                "arrays/oracle_future_offsets.npy",
                context,
                max_bytes=64 * 1024 * 1024,
            ),
            f"{context}/arrays/oracle_future_offsets.npy",
        )
        if future_offsets.shape != (sample_count + 1,) or future_offsets.dtype.kind not in "iu":
            raise ValidationError(
                f"{context} oracle_future_offsets must be integer [{sample_count + 1}], "
                f"got shape={future_offsets.shape} dtype={future_offsets.dtype}"
            )
        offsets = [int(value) for value in future_offsets]
        if offsets[0] != 0 or offsets[-1] != len(future_poses):
            raise ValidationError(f"{context} oracle future offsets do not span all future poses")
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValidationError(f"{context} oracle future offsets are not monotonic")

        sample_keys: list[str] = []
        trajectory_indices: set[int] = set()
        for row_number, sample in enumerate(samples, start=1):
            prefix = f"{context}/samples.jsonl:{row_number}"
            if sample.get("schema") != SAMPLE_SCHEMA:
                raise ValidationError(f"{prefix}.schema must be {SAMPLE_SCHEMA!r}")
            sample_key = sample.get("key")
            if not isinstance(sample_key, str) or not sample_key:
                raise ValidationError(f"{prefix}.key must be a non-empty string")
            sample_keys.append(sample_key)
            if sample.get("native_kind") != "trajectory":
                raise ValidationError(f"{prefix}.native_kind must be 'trajectory'")
            if sample.get("source_type") not in {"dagger_normal", "dagger_hard"}:
                raise ValidationError(f"{prefix}.source_type must be dagger_normal or dagger_hard")
            if native_policy_contract is not None:
                native = sample.get("native")
                if not isinstance(native, dict):
                    raise ValidationError(
                        f"{prefix}.native must be an object for native InternNav"
                    )
                native_expected = {
                    "policy_backend": "internnav_native",
                    "policy_fingerprint": native_policy_contract[
                        "policy_fingerprint"
                    ],
                    "native_protocol": native_policy_contract[
                        "native_protocol"
                    ],
                    "native_front_only": True,
                    "native_checkpoint_only": True,
                    "system2_source": "internnav_native",
                    "system1_source": "internnav_native_nextdit_async",
                    "trajectory_x_sign": 1.0,
                    "trajectory_heading_alignment": "none",
                }
                mismatches = {
                    key: {"expected": value, "actual": native.get(key)}
                    for key, value in native_expected.items()
                    if native.get(key) != value
                }
                if mismatches:
                    raise ValidationError(
                        f"{prefix}.native policy provenance mismatch: {mismatches}"
                    )
                lookdown_turns = native.get("native_lookdown_turns")
                if (
                    isinstance(lookdown_turns, bool)
                    or not isinstance(lookdown_turns, int)
                    or lookdown_turns not in (0, 1)
                ):
                    raise ValidationError(
                        f"{prefix}.native.native_lookdown_turns is invalid"
                    )
            current_frame_id = _positive_int(
                sample.get("current_frame_id"), f"{prefix}.current_frame_id", True
            )
            if current_frame_id not in frame_ids:
                raise ValidationError(f"{prefix} references unknown current_frame_id")
            if (
                native_policy_contract is not None
                and current_frame_id not in lookdown_frame_ids
            ):
                raise ValidationError(
                    f"{prefix} native current frame {current_frame_id} "
                    "must contain a lookdown observation"
                )
            history_ids = sample.get("history_frame_ids")
            history_mask = sample.get("history_valid_mask")
            history_ages = sample.get("history_age_steps")
            if not all(isinstance(value, list) for value in (history_ids, history_mask, history_ages)):
                raise ValidationError(f"{prefix} history fields must all be arrays")
            if not (len(history_ids) == len(history_mask) == len(history_ages)):
                raise ValidationError(f"{prefix} history arrays have inconsistent lengths")
            for history_index, (frame_id, valid, age) in enumerate(
                zip(history_ids, history_mask, history_ages)
            ):
                if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                    raise ValidationError(f"{prefix} history_frame_ids[{history_index}] is invalid")
                if valid not in (False, True, 0, 1):
                    raise ValidationError(f"{prefix} history_valid_mask[{history_index}] is invalid")
                if isinstance(age, bool) or not isinstance(age, int) or age < 0:
                    raise ValidationError(f"{prefix} history_age_steps[{history_index}] is invalid")
                if bool(valid) and frame_id not in frame_ids:
                    raise ValidationError(f"{prefix} valid history frame does not exist: {frame_id}")
            trajectory_index = _positive_int(
                sample.get("trajectory_index"), f"{prefix}.trajectory_index", True
            )
            if trajectory_index >= sample_count or trajectory_index in trajectory_indices:
                raise ValidationError(f"{prefix}.trajectory_index is duplicate or out of range")
            trajectory_indices.add(trajectory_index)
            expected_start = offsets[trajectory_index]
            expected_end = offsets[trajectory_index + 1]
            if sample.get("future_pose_start") != expected_start:
                raise ValidationError(f"{prefix}.future_pose_start disagrees with offsets")
            if sample.get("future_pose_end") != expected_end:
                raise ValidationError(f"{prefix}.future_pose_end disagrees with offsets")

        if len(set(sample_keys)) != sample_count:
            raise ValidationError(f"{context} samples contain duplicate key values")
        if sorted(sample_keys) != marker["sample_keys"]:
            raise ValidationError(f"{context} sample_keys disagree with commit marker")
        if trajectory_indices != set(range(sample_count)):
            raise ValidationError(f"{context} trajectory_index values are not a full permutation")

    return {
        "samples": sample_count,
        "frames": frame_count,
        "jpegs": len(actual_jpegs),
        "tar_payload_bytes": payload_bytes,
    }


def _resolve_control_root(raw_control: str, fjl_root: Path, collection: Path) -> Path:
    candidate = Path(raw_control).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        control = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"control root does not exist: {candidate}: {exc}") from exc
    if not control.is_dir() or control == fjl_root or not _is_within(control, fjl_root):
        raise ValidationError(f"control root must be a directory strictly below FJL root: {control}")
    if (
        control == collection
        or _is_within(control, collection)
        or _is_within(collection, control)
    ):
        raise ValidationError("control root and collection root must be disjoint")
    return control


def _require_external_file(path_value: Any, fjl_root: Path, context: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValidationError(f"{context} must be a non-empty absolute path")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        raise ValidationError(f"{context} must be absolute: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"{context} does not exist: {candidate}: {exc}") from exc
    if resolved == fjl_root or not _is_within(resolved, fjl_root) or not resolved.is_file():
        raise ValidationError(f"{context} must be a regular file below FJL root: {resolved}")
    return resolved


def _checked_sha256(path: Path, expected: Any, context: str) -> str:
    if (
        not isinstance(expected, str)
        or not SHA256_RE.fullmatch(expected)
        or expected != expected.lower()
    ):
        raise ValidationError(f"{context} must be a lowercase SHA-256 hex string")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValidationError(f"{context} mismatch for {path}")
    return actual


def _read_gzip_json(path: Path, context: str, max_bytes: int = 2 * 1024 * 1024 * 1024) -> Any:
    try:
        with gzip.open(path, "rb") as handle:
            payload = handle.read(max_bytes + 1)
    except (OSError, gzip.BadGzipFile) as exc:
        raise ValidationError(f"cannot read gzip {context} {path}: {exc}") from exc
    if not payload or len(payload) > max_bytes:
        raise ValidationError(f"{context} has invalid decompressed size")
    return _strict_json_loads(_decode_utf8(payload, context), context)


def _episode_rows(payload: Any, context: str) -> list[dict[str, Any]]:
    rows = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValidationError(f"{context} must contain a non-empty episodes array")
    if any(not isinstance(row, dict) for row in rows):
        raise ValidationError(f"{context} episodes must all be objects")
    return rows


def _scene_key(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context}.scene_id must be a non-empty string")
    normalized = value.replace("\\", "/").rstrip("/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        raise ValidationError(f"{context}.scene_id is invalid")
    scene = parts[-2] if parts[-1].endswith(".glb") and len(parts) >= 2 else Path(parts[-1]).stem
    if not scene or not re.fullmatch(r"[A-Za-z0-9_.-]+", scene):
        raise ValidationError(f"{context}.scene_id cannot form a stable episode key: {value!r}")
    return scene


def _cohort_keys(
    manifest: dict[str, Any],
    fjl_root: Path,
) -> tuple[list[tuple[str, int]], int]:
    contract = manifest["contract"]
    round_id = _positive_int(contract.get("round_id"), "contract.round_id", True)
    data_path = _require_external_file(contract.get("data_path"), fjl_root, "contract.data_path")
    _checked_sha256(data_path, contract.get("data_sha256"), "contract.data_sha256")
    episode_cohort = contract.get("episode_cohort")
    if not isinstance(episode_cohort, dict) or set(episode_cohort) != {
        "path",
        "sha256",
        "max_episodes",
    }:
        raise ValidationError("contract.episode_cohort must contain path, sha256, max_episodes")
    max_episodes = episode_cohort.get("max_episodes")
    if max_episodes is not None:
        _positive_int(max_episodes, "contract.episode_cohort.max_episodes")

    cohort_path_value = episode_cohort.get("path")
    if cohort_path_value is None:
        if episode_cohort.get("sha256") is not None:
            raise ValidationError("contract.episode_cohort.sha256 must be null when path is null")
        rows = _episode_rows(_read_gzip_json(data_path, "contract.data_path"), "dataset")
    else:
        cohort_path = _require_external_file(
            cohort_path_value, fjl_root, "contract.episode_cohort.path"
        )
        _checked_sha256(
            cohort_path,
            episode_cohort.get("sha256"),
            "contract.episode_cohort.sha256",
        )
        rows = _episode_rows(
            _load_json_file(cohort_path, "contract.episode_cohort.path", 512 * 1024 * 1024),
            "episode cohort",
        )

    keys: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        context = f"cohort episode {index}"
        scene = _scene_key(row.get("scene_id"), context)
        episode_id = _positive_int(row.get("episode_id"), f"{context}.episode_id", True)
        keys.append((scene, episode_id))
    if len(set(keys)) != len(keys):
        raise ValidationError("expected episode cohort contains duplicate scene/episode keys")
    return keys, round_id


def _habitat_scene_grouped_cohort_order(
    keys: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Mirror Habitat's deterministic GROUP_BY_SCENE iteration order."""
    grouped: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(key)
    return [key for scene_keys in grouped.values() for key in scene_keys]


def _validate_control(
    control: Path,
    collection: Path,
    fjl_root: Path,
    manifest: dict[str, Any],
    commits: dict[str, dict[str, Any]],
    totals: dict[str, int],
) -> dict[str, Any]:
    progress_path = _require_child(control / "progress.json", control, "control progress")
    result_path = _require_child(control / "result.json", control, "control result")
    expected, round_id = _cohort_keys(manifest, fjl_root)
    expected_iteration = _habitat_scene_grouped_cohort_order(expected)
    progress_text = _read_text(
        progress_path, "control progress.json", max_bytes=2 * 1024 * 1024 * 1024
    )
    progress_rows = _load_jsonl_text(progress_text, "control progress.json")
    if len(progress_rows) != len(expected):
        raise ValidationError(
            f"control progress is incomplete: expected={len(expected)}, got={len(progress_rows)}"
        )

    seen_keys: set[str] = set()
    for index, (row, (expected_scene, expected_episode_id)) in enumerate(
        zip(progress_rows, expected_iteration), start=1
    ):
        context = f"control progress.json:{index}"
        scene = _scene_key(row.get("scene_id"), context)
        episode_id = _positive_int(row.get("episode_id"), f"{context}.episode_id", True)
        if (scene, episode_id) != (expected_scene, expected_episode_id):
            raise ValidationError(
                f"{context} is out of Habitat scene-grouped cohort order: "
                f"expected={(expected_scene, expected_episode_id)!r}, got={(scene, episode_id)!r}"
            )
        if row.get("collect_trajectory_dagger") is not True:
            raise ValidationError(f"{context}.collect_trajectory_dagger must be true")
        stable_key = f"round{round_id:02d}_{expected_scene}_{expected_episode_id:06d}"
        _safe_episode_key(stable_key, f"{context} expected stable episode key")
        if row.get("trajectory_dagger_episode_key") != stable_key:
            raise ValidationError(f"{context} has the wrong stable DAgger episode key")
        if stable_key in seen_keys:
            raise ValidationError(f"{context} duplicates stable episode key {stable_key}")
        seen_keys.add(stable_key)

        committed_flag = row.get("trajectory_dagger_committed")
        if not isinstance(committed_flag, bool):
            raise ValidationError(f"{context}.trajectory_dagger_committed must be boolean")
        disk_commit = commits.get(stable_key)
        control_commit = row.get("trajectory_dagger_commit")
        if committed_flag:
            if disk_commit is None or not isinstance(control_commit, dict):
                raise ValidationError(f"{context} claims a missing DAgger commit")
            expected_commit = {
                "tar_sha256": disk_commit["tar_sha256"],
                "tar_bytes": disk_commit["tar_bytes"],
                "sample_count": disk_commit["sample_count"],
                "frame_count": disk_commit["frame_count"],
            }
            if control_commit != expected_commit:
                raise ValidationError(f"{context} DAgger commit summary disagrees with collection")
        elif "trajectory_dagger_commit" in row:
            raise ValidationError(f"{context} is uncommitted but contains a commit summary")

    if not set(commits).issubset(seen_keys):
        raise ValidationError("collection contains commits outside the exact processed cohort")

    result = _load_json_file(result_path, "control result.json")
    if not isinstance(result, dict):
        raise ValidationError("control result.json must be an object")
    total_episodes = _positive_int(
        result.get("total_episodes"), "control result.total_episodes", True
    )
    if total_episodes != len(expected):
        raise ValidationError(
            f"control result total_episodes mismatch: expected={len(expected)}, got={total_episodes}"
        )

    return {
        "schema": SEAL_SUMMARY_SCHEMA,
        "expected_episodes": len(expected),
        "processed_episodes": len(progress_rows),
        "committed_episodes": len(commits),
        "no_sample_episodes": len(expected) - len(commits),
        "samples": totals["samples"],
        "frames": totals["frames"],
        "jpegs": totals["jpegs"],
        "tar_payload_bytes": totals["tar_payload_bytes"],
        "control_progress_sha256": _sha256_file(progress_path),
        "control_result_sha256": _sha256_file(result_path),
    }


def _validate_sealed_summary(
    summary: Any,
    commits: dict[str, dict[str, Any]],
    totals: dict[str, int],
    expected: dict[str, Any] | None,
) -> None:
    required = {
        "schema",
        "expected_episodes",
        "processed_episodes",
        "committed_episodes",
        "no_sample_episodes",
        "samples",
        "frames",
        "jpegs",
        "tar_payload_bytes",
        "control_progress_sha256",
        "control_result_sha256",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValidationError("sealed summary fields are not canonical")
    if summary.get("schema") != SEAL_SUMMARY_SCHEMA:
        raise ValidationError("sealed summary schema is invalid")
    for name in (
        "expected_episodes",
        "processed_episodes",
        "committed_episodes",
        "no_sample_episodes",
        "samples",
        "frames",
        "jpegs",
        "tar_payload_bytes",
    ):
        _positive_int(summary.get(name), f"sealed summary.{name}", True)
    for name in ("control_progress_sha256", "control_result_sha256"):
        value = summary.get(name)
        if (
            not isinstance(value, str)
            or not SHA256_RE.fullmatch(value)
            or value != value.lower()
        ):
            raise ValidationError(f"sealed summary.{name} must be lowercase SHA-256")
    if summary["expected_episodes"] != summary["processed_episodes"]:
        raise ValidationError("sealed summary does not represent a fully processed cohort")
    if summary["committed_episodes"] != len(commits):
        raise ValidationError("sealed summary committed episode count disagrees with collection")
    if summary["no_sample_episodes"] != summary["processed_episodes"] - len(commits):
        raise ValidationError("sealed summary no-sample episode count is inconsistent")
    for name in ("samples", "frames", "jpegs", "tar_payload_bytes"):
        if summary[name] != totals[name]:
            raise ValidationError(f"sealed summary {name} disagrees with collection")
    if expected is not None and summary != expected:
        raise ValidationError("sealed summary disagrees with current control artifacts")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_seal(
    *,
    collection: Path,
    control: Path,
    fjl_root: Path,
    original_manifest: dict[str, Any],
    original_snapshot: dict[str, tuple[int, int, int]],
    effective_limit: int,
    commits: dict[str, dict[str, Any]],
    totals: dict[str, int],
    summary: dict[str, Any],
) -> bool:
    lock_path = _require_child(collection / ".capacity.lock", collection, "capacity lock")
    lock_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, lock_flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current_bytes, current_snapshot, _ = _scan_tree(collection, effective_limit)
        if current_snapshot != original_snapshot:
            raise ValidationError("collection changed before the seal lock was acquired")
        live_manifest_path = _require_child(
            collection / "collection_manifest.json", collection, "collection manifest"
        )
        live_manifest = _load_json_file(live_manifest_path, "collection_manifest.json")
        _validate_manifest(live_manifest, effective_limit)
        if live_manifest != original_manifest:
            raise ValidationError("collection manifest changed before sealing")
        live_summary = _validate_control(
            control, collection, fjl_root, live_manifest, commits, totals
        )
        if live_summary != summary:
            raise ValidationError("control artifacts changed before sealing")
        if live_manifest["ready"]:
            _validate_sealed_summary(live_manifest["summary"], commits, totals, live_summary)
            return False

        sealed_manifest = {
            **live_manifest,
            "ready": True,
            "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "summary": live_summary,
        }
        payload = _canonical_json_bytes(sealed_manifest, newline=True)
        old_manifest_bytes = live_manifest_path.stat().st_size
        final_bytes = current_bytes - old_manifest_bytes + len(payload)
        peak_bytes = current_bytes + len(payload)
        if final_bytes > effective_limit or peak_bytes > effective_limit:
            raise ValidationError(
                f"sealed manifest would exceed capacity: final={final_bytes}, peak={peak_bytes}"
            )
        temporary = collection / (
            f".collection_manifest.seal.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            temp_descriptor = os.open(temporary, flags, 0o644)
            try:
                with os.fdopen(temp_descriptor, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(temp_descriptor)
            os.replace(temporary, live_manifest_path)
            _fsync_directory(collection)
        finally:
            if temporary.exists():
                temporary.unlink()
        return True
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-root",
        required=True,
        help="trajectory-DAgger collection directory strictly below FJL_ROOT",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=HARD_CAPACITY_BYTES,
        help="additional validation ceiling; cannot exceed 300,000,000,000",
    )
    parser.add_argument(
        "--control-root",
        help="evaluator output containing exact progress.json and result.json",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help=(
            "atomically mark ready=true only after the complete contract cohort "
            "is proven by --control-root"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.seal and not args.control_root:
            raise ValidationError("--seal requires --control-root")
        requested_limit = _positive_int(args.max_bytes, "--max-bytes")
        if requested_limit > HARD_CAPACITY_BYTES:
            raise ValidationError(
                f"--max-bytes cannot exceed 300 GB ({HARD_CAPACITY_BYTES}), got {requested_limit}"
            )
        fjl_root, collection = _resolve_roots(args.collection_root)
        control = (
            _resolve_control_root(args.control_root, fjl_root, collection)
            if args.control_root
            else None
        )
        initial_bytes, initial_snapshot, physical_files = _scan_tree(
            collection, requested_limit
        )

        manifest_path = _require_child(
            collection / "collection_manifest.json", collection, "collection manifest"
        )
        manifest = _load_json_file(manifest_path, "collection_manifest.json")
        fingerprint, effective_limit, ready = _validate_manifest(manifest, requested_limit)
        if initial_bytes > effective_limit:
            raise ValidationError(
                f"collection size {initial_bytes} exceeds effective limit {effective_limit}"
            )

        contract = manifest["contract"]
        observation_contract = contract.get("observation")
        expected_view_size = None
        expected_lookdown_size = None
        expected_system1_lookdown_size = None
        if isinstance(observation_contract, dict):
            raw_view_size = observation_contract.get("vlm_image_size")
            raw_lookdown_size = observation_contract.get("lookdown_image_size")
            raw_system1_lookdown_size = observation_contract.get(
                "system1_lookdown_image_size"
            )
            if isinstance(raw_view_size, list) and len(raw_view_size) == 2:
                expected_view_size = tuple(
                    _positive_int(value, "contract observation.vlm_image_size")
                    for value in raw_view_size
                )
            if isinstance(raw_lookdown_size, list) and len(raw_lookdown_size) == 2:
                expected_lookdown_size = tuple(
                    _positive_int(value, "contract observation.lookdown_image_size")
                    for value in raw_lookdown_size
                )
            if (
                isinstance(raw_system1_lookdown_size, list)
                and len(raw_system1_lookdown_size) == 2
            ):
                expected_system1_lookdown_size = tuple(
                    _positive_int(
                        value,
                        "contract observation.system1_lookdown_image_size",
                    )
                    for value in raw_system1_lookdown_size
                )
        native_policy_contract = None
        if contract.get("rpc_policy_mode") == "internnav_native":
            policy_fingerprint = contract.get("rpc_policy_fingerprint")
            native_protocol = contract.get("native_protocol")
            if (
                not isinstance(policy_fingerprint, str)
                or not re.fullmatch(
                    r"internnav-native-v1:[0-9a-f]{64}",
                    policy_fingerprint,
                )
            ):
                raise ValidationError(
                    "native contract has an invalid rpc_policy_fingerprint"
                )
            if native_protocol != (
                "internnav-native-joint-front-history-lookdown-v1"
            ):
                raise ValidationError(
                    "native contract has an invalid native_protocol"
                )
            if expected_view_size != (384, 384):
                raise ValidationError(
                    "native InternNav contract requires 384x384 panoramic views"
                )
            if expected_lookdown_size != (640, 480):
                raise ValidationError(
                    "native InternNav contract requires a 640x480 lookdown image"
                )
            if expected_system1_lookdown_size != (224, 224):
                raise ValidationError(
                    "native InternNav contract requires "
                    "system1_lookdown_image_size=[224,224]"
                )
            native_policy_contract = {
                "policy_fingerprint": policy_fingerprint,
                "native_protocol": native_protocol,
            }

        progress_path = collection / "collection_progress.jsonl"
        if progress_path.exists():
            progress_path = _require_child(progress_path, collection, "progress ledger")
            progress_size = progress_path.stat().st_size
            if progress_size == 0:
                progress_rows = []
            else:
                progress_rows = _load_jsonl_text(
                    _read_text(
                        progress_path,
                        "collection_progress.jsonl",
                        max_bytes=2 * 1024 * 1024 * 1024,
                    ),
                    "collection_progress.jsonl",
                )
        else:
            progress_rows = []
        episodes_root = collection / "episodes"
        if not episodes_root.exists() or not episodes_root.is_dir() or episodes_root.is_symlink():
            raise ValidationError(f"missing safe episodes directory: {episodes_root}")
        staging_root = collection / ".staging"
        if not staging_root.exists() or not staging_root.is_dir() or staging_root.is_symlink():
            raise ValidationError(f"missing safe staging directory: {staging_root}")
        try:
            staging_entries = list(staging_root.iterdir())
        except OSError as exc:
            raise ValidationError(f"cannot inspect staging directory: {exc}") from exc
        if staging_entries:
            raise ValidationError(
                f"collection has incomplete staging entries: {[p.name for p in staging_entries[:5]]}"
            )

        seen_episode_keys: set[str] = set()
        commits: dict[str, dict[str, Any]] = {}
        declared_tars: set[Path] = set()
        totals = {"samples": 0, "frames": 0, "jpegs": 0, "tar_payload_bytes": 0}
        for row_number, marker in enumerate(progress_rows, start=1):
            context = f"collection_progress.jsonl:{row_number}"
            episode_key, sample_count, frame_count = _validate_commit(
                marker, fingerprint, context
            )
            if episode_key in seen_episode_keys:
                raise ValidationError(f"duplicate episode_key in progress ledger: {episode_key}")
            seen_episode_keys.add(episode_key)
            commits[episode_key] = marker

            episode_dir = episodes_root / episode_key
            try:
                resolved_episode_dir = episode_dir.resolve(strict=True)
            except OSError as exc:
                raise ValidationError(f"missing episode directory {episode_dir}: {exc}") from exc
            if not resolved_episode_dir.is_dir() or not _is_within(resolved_episode_dir, collection):
                raise ValidationError(f"unsafe episode directory: {resolved_episode_dir}")
            commit_path = _require_child(
                resolved_episode_dir / "commit.json", collection, f"{context} commit"
            )
            tar_path = _require_child(
                resolved_episode_dir / "episode.tar", collection, f"{context} tar"
            )
            if tar_path.suffix != ".tar":
                raise ValidationError(f"episode archive must be an uncompressed .tar: {tar_path}")
            declared_tars.add(tar_path)

            disk_commit = _load_json_file(commit_path, f"{episode_key}/commit.json")
            if disk_commit != marker:
                raise ValidationError(
                    f"{episode_key}/commit.json does not exactly match its progress ledger row"
                )
            archive_stats = _validate_episode_archive(
                tar_path,
                marker,
                episode_key,
                sample_count,
                frame_count,
                fingerprint,
                effective_limit,
                expected_view_size=expected_view_size,
                expected_lookdown_size=expected_lookdown_size,
                native_policy_contract=native_policy_contract,
            )
            for key, value in archive_stats.items():
                totals[key] += value

        physical_episode_dirs = {
            entry.name
            for entry in episodes_root.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        }
        if physical_episode_dirs != seen_episode_keys:
            missing = sorted(seen_episode_keys - physical_episode_dirs)
            undeclared = sorted(physical_episode_dirs - seen_episode_keys)
            raise ValidationError(
                f"episodes directory disagrees with ledger; missing={missing}, undeclared={undeclared}"
            )
        physical_tars = {path for path in physical_files if path.suffix == ".tar"}
        if physical_tars != declared_tars:
            raise ValidationError("physical .tar files do not exactly match committed episode tars")

        final_bytes, final_snapshot, _ = _scan_tree(collection, effective_limit)
        if final_snapshot != initial_snapshot:
            raise ValidationError("collection changed while validation was running")
        if final_bytes != initial_bytes:
            raise ValidationError("collection size changed while validation was running")

        control_summary = None
        if control is not None:
            control_summary = _validate_control(
                control, collection, fjl_root, manifest, commits, totals
            )
        if ready:
            _validate_sealed_summary(
                manifest["summary"], commits, totals, control_summary
            )

        sealed_now = False
        if args.seal:
            assert control is not None
            assert control_summary is not None
            sealed_now = _atomic_seal(
                collection=collection,
                control=control,
                fjl_root=fjl_root,
                original_manifest=manifest,
                original_snapshot=final_snapshot,
                effective_limit=effective_limit,
                commits=commits,
                totals=totals,
                summary=control_summary,
            )
            ready = True
            manifest = _load_json_file(manifest_path, "collection_manifest.json")
            _, effective_limit, ready = _validate_manifest(manifest, requested_limit)
            _validate_sealed_summary(
                manifest["summary"], commits, totals, control_summary
            )
            initial_bytes, _, _ = _scan_tree(collection, effective_limit)

    except (ValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "collection_root": str(collection),
                "manifest_ready": ready,
                "capacity_bytes": initial_bytes,
                "capacity_limit_bytes": effective_limit,
                "episodes": len(seen_episode_keys),
                "sealed_now": sealed_now,
                **totals,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
