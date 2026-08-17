"""Lazy, read-only data plane for trajectory-DAgger episode tars.

Construction indexes only committed JSON metadata. JPEGs and trajectory arrays
are decoded on demand in DataLoader workers. Heatmaps are never read, generated,
or persisted here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .trajectory_utils import compute_history_rel_poses


FJL_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl")
COLLECTION_SCHEMA = "heatmapvln-trajectory-dagger-collection-v1"
COMMIT_SCHEMA = "heatmapvln-trajectory-dagger-episode-commit-v1"
SAMPLE_SCHEMA = "heatmapvln-trajectory-dagger-sample-v1"
VIEW_NAMES = ("front", "right", "back", "left")
SOURCE_NAMES = ("expert", "dagger_normal", "dagger_hard")
DEFAULT_MIXTURE_PROFILE = "expert50_normal20_hard30"
MIXTURE_PROFILES = {
    DEFAULT_MIXTURE_PROFILE: {
        "expert": 0.50,
        "dagger_normal": 0.20,
        "dagger_hard": 0.30,
    },
    "expert60_hard40": {
        "expert": 0.60,
        "dagger_normal": 0.00,
        "dagger_hard": 0.40,
    },
}
DEFAULT_MIXTURE_WEIGHTS = dict(
    MIXTURE_PROFILES[DEFAULT_MIXTURE_PROFILE]
)
HISTORY_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)
NATIVE_POLICY_MODE = "internnav_native"
NATIVE_PROTOCOL = "internnav-native-joint-front-history-lookdown-v1"
NATIVE_POLICY_FINGERPRINT_RE = re.compile(
    r"^internnav-native-v1:[0-9a-f]{64}$"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_bytes(data: bytes, context: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{context} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError(f"{context} lacks a final newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{context}:{line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{context}:{line_number} is not an object")
        rows.append(value)
    return rows


def _safe_tar_members(
    archive: tarfile.TarFile,
    context: str,
) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = member.name
        pure = PurePosixPath(name)
        canonical = pure.as_posix()
        comparable = name.rstrip("/") if member.isdir() else name
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in pure.parts)
            or comparable != canonical
        ):
            raise ValueError(f"unsafe tar member in {context}: {name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"non-regular tar member in {context}: {name}")
        if name in members:
            raise ValueError(f"duplicate tar member in {context}: {name}")
        members[name] = member
    return members


def _read_tar_member(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    name: str,
    *,
    context: str,
    max_bytes: int,
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise ValueError(f"missing regular member {name} in {context}")
    if member.size <= 0 or member.size > max_bytes:
        raise ValueError(f"invalid member size for {context}/{name}: {member.size}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot read member {name} in {context}")
    data = handle.read(max_bytes + 1)
    if len(data) != member.size or len(data) > max_bytes:
        raise ValueError(f"short or oversized member {name} in {context}")
    return data


def _as_pose(value: Any, context: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{context} must be a finite [4,4] pose")
    if not np.allclose(
        pose[3],
        np.array([0, 0, 0, 1], dtype=np.float32),
        atol=1e-5,
    ):
        raise ValueError(f"{context} has an invalid homogeneous row")
    return pose


def _normalize_roots(
    roots: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[Path]:
    values = [roots] if isinstance(roots, (str, os.PathLike)) else list(roots)
    if not values:
        raise ValueError("collection_roots may not be empty")
    return [Path(value).expanduser() for value in values]


def _require_collection_root(path: Path, allowed_root: Path) -> Path:
    allowed = allowed_root.resolve(strict=True)
    candidate = path.absolute()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"collection root must stay below {allowed}: {candidate}"
        ) from exc
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"collection root resolves outside {allowed}: {resolved}"
        ) from exc
    if resolved == allowed or not resolved.is_dir() or path.is_symlink():
        raise ValueError(f"invalid collection root: {resolved}")
    return resolved


@dataclass(frozen=True)
class DaggerSampleRecord:
    collection_root: Path
    episode_key: str
    tar_path: Path
    sample_key: str
    source_type: str
    trajectory_index: int
    episode_sample_count: int
    sample: dict[str, Any]


class TrajectoryDaggerDataset(Dataset):
    """Read one or more sealed trajectory-DAgger collections.

    The allow_unsealed_debug flag is the only escape hatch for manifest.ready
    not being true. It is intended for smoke diagnostics, never normal training.
    """

    _is_panoramic = True
    single_view_rgb_input = False
    dynamic_sampling_enabled = False

    def __init__(
        self,
        collection_roots: (
            str
            | os.PathLike[str]
            | Sequence[str | os.PathLike[str]]
        ),
        *,
        allow_unsealed_debug: bool = False,
        source_types: Iterable[str] | None = None,
        num_history: int = 8,
        image_size: tuple[int, int] | None = None,
        verify_tar_sha256: bool = False,
        require_lookdown: bool = False,
        expected_policy_mode: str | None = None,
        expected_policy_fingerprint: str | None = None,
        allowed_root: str | os.PathLike[str] = FJL_ROOT,
    ) -> None:
        if isinstance(num_history, bool) or int(num_history) <= 0:
            raise ValueError("num_history must be a positive integer")
        self.num_history = int(num_history)
        self.allow_unsealed_debug = bool(allow_unsealed_debug)
        self.verify_tar_sha256 = bool(verify_tar_sha256)
        if expected_policy_mode is None:
            normalized_policy_mode = None
        elif (
            not isinstance(expected_policy_mode, str)
            or not expected_policy_mode.strip()
        ):
            raise ValueError(
                "expected_policy_mode must be None or a non-empty string"
            )
        else:
            normalized_policy_mode = expected_policy_mode.strip()
        if expected_policy_fingerprint is None:
            normalized_policy_fingerprint = None
        elif (
            not isinstance(expected_policy_fingerprint, str)
            or not expected_policy_fingerprint.strip()
        ):
            raise ValueError(
                "expected_policy_fingerprint must be None or a non-empty string"
            )
        else:
            normalized_policy_fingerprint = (
                expected_policy_fingerprint.strip()
            )
        if (
            normalized_policy_fingerprint is not None
            and normalized_policy_mode is None
        ):
            raise ValueError(
                "expected_policy_fingerprint requires expected_policy_mode"
            )
        if normalized_policy_mode == NATIVE_POLICY_MODE:
            if (
                normalized_policy_fingerprint is None
                or not NATIVE_POLICY_FINGERPRINT_RE.fullmatch(
                    normalized_policy_fingerprint
                )
            ):
                raise ValueError(
                    "internnav_native requires an exact "
                    "internnav-native-v1:<64 lowercase hex> fingerprint"
                )
        self.expected_policy_mode = normalized_policy_mode
        self.expected_policy_fingerprint = normalized_policy_fingerprint
        self.require_lookdown = bool(require_lookdown) or (
            normalized_policy_mode == NATIVE_POLICY_MODE
        )
        self.allowed_root = Path(allowed_root)
        requested = (
            frozenset(("dagger_normal", "dagger_hard"))
            if source_types is None
            else frozenset(str(value) for value in source_types)
        )
        unknown = requested - {"dagger_normal", "dagger_hard"}
        if unknown:
            raise ValueError(f"unsupported DAgger source_types: {sorted(unknown)}")
        self.requested_sources = requested
        self.collection_roots = tuple(
            _require_collection_root(path, self.allowed_root)
            for path in _normalize_roots(collection_roots)
        )
        if len(set(self.collection_roots)) != len(self.collection_roots):
            raise ValueError("collection_roots contains duplicates")

        self._records: list[DaggerSampleRecord] = []
        self._source_indices: dict[str, list[int]] = {
            "dagger_normal": [],
            "dagger_hard": [],
        }
        self.manifests: list[dict[str, Any]] = []
        self._manifest_image_size: tuple[int, int] | None = None
        self._manifest_lookdown_image_size: tuple[int, int] | None = None
        self._manifest_system1_lookdown_image_size: (
            tuple[int, int] | None
        ) = None
        seen_sample_keys: set[str] = set()
        for root in self.collection_roots:
            self._index_collection(root, seen_sample_keys)

        if image_size is None:
            if self._manifest_image_size is None:
                raise ValueError("collection manifests do not define vlm_image_size")
            self.image_size = self._manifest_image_size
        else:
            self.image_size = (int(image_size[0]), int(image_size[1]))
            if min(self.image_size) <= 0:
                raise ValueError("image_size values must be positive")
        if (
            self._manifest_lookdown_image_size is None
            or self._manifest_system1_lookdown_image_size is None
        ):
            raise ValueError(
                "collection manifests do not define both lookdown output sizes"
            )
        # Keep the native System-2 lookdown observation independent from the
        # System-1 DINO/NextDiT image. Both are derived from the same JPEG, but
        # they intentionally have different spatial contracts.
        self.lookdown_image_size = self._manifest_lookdown_image_size
        self.system2_lookdown_image_size = self.lookdown_image_size
        self.system1_lookdown_image_size = (
            self._manifest_system1_lookdown_image_size
        )
        # Backwards-compatible alias: before the two contracts were separated,
        # lookdown_output_size denoted the System-1 resize.
        self.lookdown_output_size = self.system1_lookdown_image_size
        self.source_counts = {
            name: len(indices) for name, indices in self._source_indices.items()
        }
        self.hm_size = (64, 64)

    def _index_collection(
        self,
        root: Path,
        seen_sample_keys: set[str],
    ) -> None:
        manifest_path = root / "collection_manifest.json"
        ledger_path = root / "collection_progress.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != COLLECTION_SCHEMA
        ):
            raise ValueError(f"invalid collection manifest schema: {manifest_path}")
        if manifest.get("ready") is not True and not self.allow_unsealed_debug:
            raise RuntimeError(
                f"collection is not sealed (manifest.ready is not true): {root}; "
                "set allow_unsealed_debug=True only for explicit diagnostics"
            )
        fingerprint = str(manifest.get("fingerprint") or "")
        identity = {
            key: manifest.get(key)
            for key in ("schema", "contract", "capacity")
        }
        expected = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
        if fingerprint != expected:
            raise ValueError(f"collection manifest fingerprint mismatch: {root}")

        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"collection contract is missing: {root}")
        if (
            self.expected_policy_mode is not None
            and contract.get("rpc_policy_mode")
            != self.expected_policy_mode
        ):
            raise ValueError(
                "collection policy mode mismatch: "
                f"expected={self.expected_policy_mode!r} "
                f"actual={contract.get('rpc_policy_mode')!r} "
                f"root={root}"
            )
        if (
            self.expected_policy_fingerprint is not None
            and contract.get("rpc_policy_fingerprint")
            != self.expected_policy_fingerprint
        ):
            raise ValueError(
                "collection policy fingerprint mismatch: "
                f"expected={self.expected_policy_fingerprint!r} "
                f"actual={contract.get('rpc_policy_fingerprint')!r} "
                f"root={root}"
            )
        if (
            self.expected_policy_mode == NATIVE_POLICY_MODE
            and contract.get("native_protocol") != NATIVE_PROTOCOL
        ):
            raise ValueError(
                "collection native protocol mismatch: "
                f"expected={NATIVE_PROTOCOL!r} "
                f"actual={contract.get('native_protocol')!r} "
                f"root={root}"
            )
        observation = contract.get("observation")
        target = contract.get("target")
        if not isinstance(observation, dict) or not isinstance(target, dict):
            raise ValueError(
                f"collection lacks observation/target contract: {root}"
            )
        if tuple(observation.get("view_order") or ()) != VIEW_NAMES:
            raise ValueError(f"collection view order is incompatible: {root}")
        if int(observation.get("num_history", -1)) != self.num_history:
            raise ValueError(
                f"collection num_history is incompatible: "
                f"{observation.get('num_history')} != {self.num_history}"
            )
        manifest_size = tuple(
            int(value)
            for value in observation.get("vlm_image_size") or ()
        )
        if len(manifest_size) != 2 or min(manifest_size) <= 0:
            raise ValueError(f"invalid vlm_image_size in {root}")
        if (
            self._manifest_image_size is not None
            and self._manifest_image_size != manifest_size
        ):
            raise ValueError("collection roots have incompatible image sizes")
        self._manifest_image_size = manifest_size

        lookdown_image_size = tuple(
            int(value)
            for value in observation.get("lookdown_image_size") or ()
        )
        if (
            len(lookdown_image_size) != 2
            or min(lookdown_image_size) <= 0
        ):
            raise ValueError(f"invalid lookdown_image_size in {root}")

        system1_size_value = observation.get(
            "system1_lookdown_image_size"
        )
        if (
            contract.get("rpc_policy_mode") == NATIVE_POLICY_MODE
            and system1_size_value is None
        ):
            raise ValueError(
                "internnav_native collection must define "
                f"system1_lookdown_image_size: {root}"
            )
        system1_lookdown_image_size = tuple(
            int(value)
            for value in (
                system1_size_value
                if system1_size_value is not None
                else lookdown_image_size
            )
        )
        if (
            len(system1_lookdown_image_size) != 2
            or min(system1_lookdown_image_size) <= 0
        ):
            raise ValueError(
                f"invalid system1_lookdown_image_size in {root}"
            )
        if contract.get("rpc_policy_mode") == NATIVE_POLICY_MODE:
            if lookdown_image_size != (640, 480):
                raise ValueError(
                    "internnav_native lookdown_image_size must be "
                    f"(640, 480), got {lookdown_image_size} in {root}"
                )
            if system1_lookdown_image_size != (224, 224):
                raise ValueError(
                    "internnav_native system1_lookdown_image_size must be "
                    f"(224, 224), got {system1_lookdown_image_size} in {root}"
                )
        if (
            self._manifest_lookdown_image_size is not None
            and self._manifest_lookdown_image_size != lookdown_image_size
        ):
            raise ValueError(
                "collection roots have incompatible System-2 lookdown sizes"
            )
        if (
            self._manifest_system1_lookdown_image_size is not None
            and self._manifest_system1_lookdown_image_size
            != system1_lookdown_image_size
        ):
            raise ValueError(
                "collection roots have incompatible System-1 lookdown sizes"
            )
        self._manifest_lookdown_image_size = lookdown_image_size
        self._manifest_system1_lookdown_image_size = (
            system1_lookdown_image_size
        )
        if int(target.get("predict_horizon", -1)) != 32:
            raise ValueError(f"collection predict_horizon must be 32: {root}")
        if str(target.get("camera_forward_axis")) != "-z":
            raise ValueError(f"collection camera convention must be -z: {root}")

        ledger_rows = _read_jsonl_bytes(
            ledger_path.read_bytes(),
            f"{root}/collection_progress.jsonl",
        )
        if not ledger_rows:
            raise ValueError(
                f"sealed collection contains no committed episodes: {root}"
            )
        staging = root / ".staging"
        if staging.is_dir() and any(staging.iterdir()):
            raise RuntimeError(
                f"collection has incomplete staging entries: {staging}"
            )

        declared: set[str] = set()
        for marker in ledger_rows:
            if marker.get("schema") != COMMIT_SCHEMA:
                raise ValueError(f"invalid commit schema in {ledger_path}")
            episode_key = str(marker.get("episode_key") or "")
            if (
                not episode_key
                or "/" in episode_key
                or "\\" in episode_key
                or episode_key in declared
            ):
                raise ValueError(
                    f"invalid or duplicate episode key: {episode_key!r}"
                )
            declared.add(episode_key)
            if marker.get("manifest_fingerprint") != fingerprint:
                raise ValueError(
                    f"commit fingerprint mismatch: {episode_key}"
                )
            episode_dir = root / "episodes" / episode_key
            if episode_dir.is_symlink() or not episode_dir.is_dir():
                raise ValueError(
                    f"unsafe or missing episode directory: {episode_dir}"
                )
            commit_path = episode_dir / "commit.json"
            tar_path = episode_dir / "episode.tar"
            if commit_path.is_symlink() or tar_path.is_symlink():
                raise ValueError(
                    f"symlinked episode artifact: {episode_dir}"
                )
            disk_commit = json.loads(
                commit_path.read_text(encoding="utf-8")
            )
            if disk_commit != marker:
                raise ValueError(
                    f"commit marker disagrees with ledger: {episode_key}"
                )
            if tar_path.stat().st_size != int(marker.get("tar_bytes", -1)):
                raise ValueError(
                    f"tar size disagrees with commit: {episode_key}"
                )
            if (
                self.verify_tar_sha256
                and _sha256_file(tar_path) != marker.get("tar_sha256")
            ):
                raise ValueError(
                    f"tar SHA256 disagrees with commit: {episode_key}"
                )

            with tarfile.open(tar_path, mode="r:") as archive:
                members = _safe_tar_members(archive, episode_key)
                samples = _read_jsonl_bytes(
                    _read_tar_member(
                        archive,
                        members,
                        "samples.jsonl",
                        context=episode_key,
                        max_bytes=256 * 1024 * 1024,
                    ),
                    f"{episode_key}/samples.jsonl",
                )
            sample_count = int(marker.get("sample_count", -1))
            if len(samples) != sample_count:
                raise ValueError(
                    f"sample count disagrees with commit: {episode_key}"
                )
            keys: list[str] = []
            trajectory_indices: set[int] = set()
            for sample in samples:
                if sample.get("schema") != SAMPLE_SCHEMA:
                    raise ValueError(
                        f"invalid sample schema in {episode_key}"
                    )
                sample_key = sample.get("key")
                source_type = sample.get("source_type")
                trajectory_index = sample.get("trajectory_index")
                if not isinstance(sample_key, str) or not sample_key:
                    raise ValueError(
                        f"sample key is missing in {episode_key}"
                    )
                if sample_key in seen_sample_keys:
                    raise ValueError(
                        f"duplicate sample key across collections: {sample_key}"
                    )
                seen_sample_keys.add(sample_key)
                keys.append(sample_key)
                if source_type not in {
                    "dagger_normal",
                    "dagger_hard",
                }:
                    raise ValueError(
                        f"invalid source_type for {sample_key}: "
                        f"{source_type!r}"
                    )
                if self.expected_policy_mode == NATIVE_POLICY_MODE:
                    native = sample.get("native")
                    if not isinstance(native, dict):
                        raise ValueError(
                            f"native provenance is missing for {sample_key}"
                        )
                    native_expected = {
                        "policy_backend": NATIVE_POLICY_MODE,
                        "policy_fingerprint": (
                            self.expected_policy_fingerprint
                        ),
                        "native_protocol": NATIVE_PROTOCOL,
                        "native_front_only": True,
                        "native_checkpoint_only": True,
                        "system2_source": "internnav_native",
                        "system1_source": (
                            "internnav_native_nextdit_async"
                        ),
                        "trajectory_x_sign": 1.0,
                        "trajectory_heading_alignment": "none",
                    }
                    mismatches = {
                        key: {
                            "expected": expected,
                            "actual": native.get(key),
                        }
                        for key, expected in native_expected.items()
                        if native.get(key) != expected
                    }
                    lookdown_turns = native.get(
                        "native_lookdown_turns"
                    )
                    if (
                        isinstance(lookdown_turns, bool)
                        or not isinstance(lookdown_turns, int)
                        or lookdown_turns not in (0, 1)
                    ):
                        mismatches["native_lookdown_turns"] = {
                            "expected": "0 or 1",
                            "actual": lookdown_turns,
                        }
                    if mismatches:
                        raise ValueError(
                            "native sample policy provenance mismatch for "
                            f"{sample_key}: {mismatches}"
                        )
                if (
                    isinstance(trajectory_index, bool)
                    or not isinstance(trajectory_index, int)
                    or trajectory_index < 0
                    or trajectory_index >= sample_count
                    or trajectory_index in trajectory_indices
                ):
                    raise ValueError(
                        f"invalid trajectory_index for {sample_key}"
                    )
                trajectory_indices.add(trajectory_index)
                if source_type not in self.requested_sources:
                    continue
                dataset_index = len(self._records)
                self._records.append(
                    DaggerSampleRecord(
                        collection_root=root,
                        episode_key=episode_key,
                        tar_path=tar_path,
                        sample_key=sample_key,
                        source_type=str(source_type),
                        trajectory_index=trajectory_index,
                        episode_sample_count=sample_count,
                        sample=sample,
                    )
                )
                self._source_indices[str(source_type)].append(
                    dataset_index
                )
            if sorted(keys) != marker.get("sample_keys"):
                raise ValueError(
                    f"sample keys disagree with commit: {episode_key}"
                )
            if trajectory_indices != set(range(sample_count)):
                raise ValueError(
                    f"trajectory indices are incomplete: {episode_key}"
                )

        episodes_root = root / "episodes"
        physical = {
            path.name
            for path in episodes_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        }
        if physical != declared:
            raise ValueError(
                f"physical episodes disagree with ledger: {root}"
            )
        self.manifests.append(manifest)

    @property
    def source_indices(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(indices)
            for name, indices in self._source_indices.items()
        }

    @property
    def sample_keys(self) -> tuple[str, ...]:
        return tuple(record.sample_key for record in self._records)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _decode_jpeg(
        data: bytes,
        size: tuple[int, int],
        context: str,
        *,
        resample_filter: Any | None = None,
    ) -> torch.Tensor:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError(f"{context} is not JPEG")
                image = image.convert("RGB")
                if image.size != size:
                    resampling = getattr(Image, "Resampling", Image)
                    image = image.resize(
                        size,
                        resampling.BILINEAR
                        if resample_filter is None
                        else resample_filter,
                    )
                array = np.asarray(image, dtype=np.uint8).copy()
        except Exception as exc:
            raise ValueError(
                f"failed to decode {context}: {exc}"
            ) from exc
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self._records[index]
        sample = record.sample
        context = f"{record.episode_key}/{record.sample_key}"

        with tarfile.open(record.tar_path, mode="r:") as archive:
            members = _safe_tar_members(
                archive,
                record.episode_key,
            )
            frames = _read_jsonl_bytes(
                _read_tar_member(
                    archive,
                    members,
                    "frames.jsonl",
                    context=record.episode_key,
                    max_bytes=256 * 1024 * 1024,
                ),
                f"{record.episode_key}/frames.jsonl",
            )
            frame_by_id: dict[int, dict[str, Any]] = {}
            for frame in frames:
                frame_id = frame.get("frame_id")
                if (
                    isinstance(frame_id, bool)
                    or not isinstance(frame_id, int)
                    or frame_id < 0
                    or frame_id in frame_by_id
                ):
                    raise ValueError(
                        f"invalid frame id in {record.episode_key}"
                    )
                frame_by_id[frame_id] = frame

            trajectory_data = _read_tar_member(
                archive,
                members,
                "arrays/trajectories.npy",
                context=record.episode_key,
                max_bytes=512 * 1024 * 1024,
            )
            trajectories = np.load(
                io.BytesIO(trajectory_data),
                allow_pickle=False,
            )
            expected_shape = (
                record.episode_sample_count,
                32,
                3,
            )
            if (
                trajectories.shape != expected_shape
                or not np.isfinite(trajectories).all()
            ):
                raise ValueError(
                    f"invalid trajectory array for "
                    f"{record.episode_key}: "
                    f"{trajectories.shape} != {expected_shape}"
                )

            def load_frame(
                frame_id: int,
            ) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
                frame = frame_by_id.get(frame_id)
                if frame is None:
                    raise ValueError(
                        f"{context} references missing frame {frame_id}"
                    )
                views = frame.get("views")
                if (
                    not isinstance(views, dict)
                    or set(views) != set(VIEW_NAMES)
                ):
                    raise ValueError(
                        f"{context} frame {frame_id} has invalid views"
                    )
                tensors = []
                for view in VIEW_NAMES:
                    name = views[view]
                    expected_name = (
                        f"frames/{frame_id:06d}_{view}.jpg"
                    )
                    if name != expected_name:
                        raise ValueError(
                            f"{context} has non-canonical "
                            f"view path {name!r}"
                        )
                    data = _read_tar_member(
                        archive,
                        members,
                        name,
                        context=record.episode_key,
                        max_bytes=32 * 1024 * 1024,
                    )
                    tensors.append(
                        self._decode_jpeg(
                            data,
                            self.image_size,
                            f"{context}/{name}",
                        )
                    )
                return (
                    torch.stack(tensors, dim=0),
                    _as_pose(
                        frame.get("pose"),
                        f"{context}/frame{frame_id}.pose",
                    ),
                    frame,
                )

            current_frame_id = sample.get("current_frame_id")
            if (
                isinstance(current_frame_id, bool)
                or not isinstance(current_frame_id, int)
            ):
                raise ValueError(
                    f"{context} has invalid current_frame_id"
                )
            current_views, current_pose, current_row = load_frame(
                current_frame_id
            )
            if "current_camera_pose" in sample:
                sample_camera_pose = _as_pose(
                    sample["current_camera_pose"],
                    f"{context}.current_camera_pose",
                )
                if not np.allclose(
                    sample_camera_pose,
                    current_pose,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"{context} current camera pose "
                        "disagrees with frame"
                    )
            current_agent_pose = _as_pose(
                sample.get("current_agent_pose", current_pose),
                f"{context}.current_agent_pose",
            )

            history_ids = sample.get("history_frame_ids")
            history_mask = sample.get("history_valid_mask")
            history_ages = sample.get("history_age_steps")
            if not all(
                isinstance(value, list)
                for value in (
                    history_ids,
                    history_mask,
                    history_ages,
                )
            ):
                raise ValueError(
                    f"{context} history fields must be arrays"
                )
            if not (
                len(history_ids)
                == len(history_mask)
                == len(history_ages)
            ):
                raise ValueError(
                    f"{context} history arrays have "
                    "inconsistent lengths"
                )
            if len(history_ids) > self.num_history:
                raise ValueError(
                    f"{context} has more than "
                    f"{self.num_history} histories"
                )

            history_panoramas = torch.zeros(
                self.num_history,
                len(VIEW_NAMES),
                3,
                self.image_size[1],
                self.image_size[0],
                dtype=torch.float32,
            )
            history_poses = np.repeat(
                np.eye(4, dtype=np.float32)[None],
                self.num_history,
                axis=0,
            )
            output_ids = torch.full(
                (self.num_history,),
                -1,
                dtype=torch.long,
            )
            output_mask = torch.zeros(
                self.num_history,
                dtype=torch.bool,
            )
            output_ages = torch.zeros(
                self.num_history,
                dtype=torch.long,
            )
            valid_positions: list[int] = []
            valid_poses: list[np.ndarray] = []
            for position, (frame_id, valid, age) in enumerate(
                zip(history_ids, history_mask, history_ages)
            ):
                if (
                    isinstance(frame_id, bool)
                    or not isinstance(frame_id, int)
                    or valid not in (False, True, 0, 1)
                    or isinstance(age, bool)
                    or not isinstance(age, int)
                    or age < 0
                ):
                    raise ValueError(
                        f"{context} has an invalid history entry"
                    )
                output_ids[position] = frame_id
                output_mask[position] = bool(valid)
                output_ages[position] = age
                if not bool(valid):
                    continue
                panorama, pose, _ = load_frame(frame_id)
                history_panoramas[position] = panorama
                history_poses[position] = pose
                valid_positions.append(position)
                valid_poses.append(pose)

            history_rel_poses = np.zeros(
                (self.num_history, 4),
                dtype=np.float32,
            )
            if valid_poses:
                valid_rel = compute_history_rel_poses(
                    valid_poses,
                    current_pose,
                    camera_forward_axis="-z",
                )
                history_rel_poses[
                    np.asarray(valid_positions, dtype=np.int64)
                ] = valid_rel

            lookdown_name = current_row.get("lookdown")
            if lookdown_name is None:
                if self.require_lookdown:
                    raise ValueError(
                        f"{context} has no lookdown observation"
                    )
                front_name = current_row["views"]["front"]
                front_data = _read_tar_member(
                    archive,
                    members,
                    front_name,
                    context=record.episode_key,
                    max_bytes=32 * 1024 * 1024,
                )
                lookdown_data = front_data
                lookdown_context = (
                    f"{context}/{front_name}:legacy-lookdown-fallback"
                )
            else:
                expected_lookdown = (
                    f"lookdown/{current_frame_id:06d}.jpg"
                )
                if lookdown_name != expected_lookdown:
                    raise ValueError(
                        f"{context} has a non-canonical "
                        "lookdown path"
                    )
                lookdown_data = _read_tar_member(
                    archive,
                    members,
                    lookdown_name,
                    context=record.episode_key,
                    max_bytes=32 * 1024 * 1024,
                )
                lookdown_context = f"{context}/{lookdown_name}"

            bicubic = getattr(Image, "Resampling", Image).BICUBIC
            lookdown = self._decode_jpeg(
                lookdown_data,
                self.system2_lookdown_image_size,
                f"{lookdown_context}:system2",
                resample_filter=bicubic,
            )
            system1_lookdown = self._decode_jpeg(
                lookdown_data,
                self.system1_lookdown_image_size,
                f"{lookdown_context}:system1",
                resample_filter=bicubic,
            )

        trajectory = torch.from_numpy(
            np.asarray(
                trajectories[record.trajectory_index],
                dtype=np.float32,
            ).copy()
        )
        native = (
            sample.get("native")
            if isinstance(sample.get("native"), dict)
            else {}
        )
        native_actions = native.get("actions")
        if not isinstance(native_actions, list):
            native_actions = []
        first_action = (
            int(native_actions[0])
            if (
                native_actions
                and int(native_actions[0]) in (0, 1, 2, 3)
            )
            else 1
        )
        pano_goal_view = native.get("pano_goal_view")
        pixel_goal = native.get("pixel_goal")
        if (
            pano_goal_view is not None
            and pano_goal_view not in VIEW_NAMES
        ):
            raise ValueError(
                f"{context} has invalid native pano_goal_view"
            )
        if pixel_goal is not None:
            if (
                not isinstance(pixel_goal, list)
                or len(pixel_goal) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in pixel_goal
                )
            ):
                raise ValueError(
                    f"{context} has invalid native pixel_goal"
                )
            pixel_goal = [
                int(pixel_goal[0]),
                int(pixel_goal[1]),
            ]

        lookdown_hwc = (
            system1_lookdown.permute(1, 2, 0).contiguous()
        )
        result: dict[str, Any] = {
            "sample_key": record.sample_key,
            "source_type": record.source_type,
            "collection_root": str(record.collection_root),
            "episode_key": record.episode_key,
            "scene_id": sample.get("scene_id"),
            "episode_id": sample.get("episode_id"),
            "text": str(sample.get("instruction") or ""),
            "instruction": str(
                sample.get("instruction") or ""
            ),
            "current_views": current_views,
            "current_frame": current_views[0],
            "history_panoramas": history_panoramas,
            "history_frames": history_panoramas[:, 0],
            "current_pose": torch.from_numpy(
                current_pose.copy()
            ),
            "current_camera_pose": torch.from_numpy(
                current_pose.copy()
            ),
            "current_agent_pose": torch.from_numpy(
                current_agent_pose.copy()
            ),
            "history_poses": torch.from_numpy(
                history_poses
            ),
            "history_rel_poses": torch.from_numpy(
                history_rel_poses
            ),
            "history_frame_ids": output_ids,
            "history_valid_mask": output_mask,
            "history_mask": output_mask.float(),
            "history_age_steps": output_ages,
            "heatmap_direction_order": VIEW_NAMES,
            "history_pose_convention": (
                HISTORY_POSE_CONVENTION
            ),
            "trajectory": trajectory,
            "trajectory_valid": float(
                sample.get("trajectory_valid", 1.0)
            ),
            "lookdown_frame": lookdown,
            "traj_images": torch.stack(
                (lookdown_hwc, lookdown_hwc),
                dim=0,
            ),
            "action": torch.zeros(
                2,
                dtype=torch.float32,
            ),
            "action_valid": 0.0,
            "discrete_action": first_action,
            "is_stop": float(first_action == 0),
            "progress": 0.0,
            "native": native,
            "oracle": sample.get("oracle"),
            "candidate_signals": sample.get(
                "candidate_signals"
            ),
            "failure_tags": sample.get(
                "failure_tags",
                [],
            ),
        }
        if pano_goal_view is not None:
            result["pano_view_id"] = pano_goal_view
        if pixel_goal is not None:
            result["pixel_goal"] = pixel_goal
            result["pano_pixel_goal"] = pixel_goal
            result["pano_sample_kind"] = "pixel"
        return result


def trajectory_dagger_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack the fixed DAgger batch contract without heatmaps."""
    if not batch:
        raise ValueError("cannot collate an empty DAgger batch")
    tensor_keys = (
        "current_views",
        "current_frame",
        "history_panoramas",
        "history_frames",
        "current_pose",
        "current_camera_pose",
        "current_agent_pose",
        "history_poses",
        "history_rel_poses",
        "history_frame_ids",
        "history_valid_mask",
        "history_mask",
        "history_age_steps",
        "trajectory",
        "lookdown_frame",
        "traj_images",
        "action",
    )
    result = {
        key: torch.stack(
            [sample[key] for sample in batch],
            dim=0,
        )
        for key in tensor_keys
    }
    result.update(
        {
            "trajectory_valid": torch.tensor(
                [
                    sample["trajectory_valid"]
                    for sample in batch
                ],
                dtype=torch.float32,
            ),
            "action_valid": torch.tensor(
                [sample["action_valid"] for sample in batch],
                dtype=torch.float32,
            ),
            "discrete_action": torch.tensor(
                [
                    sample["discrete_action"]
                    for sample in batch
                ],
                dtype=torch.long,
            ),
            "is_stop": torch.tensor(
                [sample["is_stop"] for sample in batch],
                dtype=torch.float32,
            ),
            "progress": torch.tensor(
                [sample["progress"] for sample in batch],
                dtype=torch.float32,
            ),
            "sample_key": [
                sample["sample_key"] for sample in batch
            ],
            "source_type": [
                sample["source_type"] for sample in batch
            ],
            "text": [sample["text"] for sample in batch],
            "instruction": [
                sample["instruction"] for sample in batch
            ],
            "episode_key": [
                sample["episode_key"] for sample in batch
            ],
            "collection_root": [
                sample["collection_root"] for sample in batch
            ],
            "heatmap_direction_order": VIEW_NAMES,
            "history_pose_convention": (
                HISTORY_POSE_CONVENTION
            ),
            "native": [sample["native"] for sample in batch],
            "oracle": [sample["oracle"] for sample in batch],
            "candidate_signals": [
                sample["candidate_signals"]
                for sample in batch
            ],
            "failure_tags": [
                sample["failure_tags"] for sample in batch
            ],
        }
    )
    for key in (
        "pano_view_id",
        "pixel_goal",
        "pano_pixel_goal",
        "pano_sample_kind",
    ):
        if all(key in sample for sample in batch):
            result[key] = [
                sample[key] for sample in batch
            ]
    return result


class IndexedSourceDataset(Dataset):
    """A source-labelled index view over a map-style dataset."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        source_type: str,
    ) -> None:
        if source_type not in SOURCE_NAMES:
            raise ValueError(
                f"invalid source type: {source_type}"
            )
        self.dataset = dataset
        self._dataset_length = len(dataset)
        self.indices = tuple(
            int(index) for index in indices
        )
        self.source_type = source_type
        # This class is an index view, not a change in the observation
        # contract. Preserve the source capabilities so the enclosing
        # mixture can select the correct collator and sampling behaviour.
        self._is_panoramic = bool(
            getattr(dataset, "_is_panoramic", False)
        )
        self.single_view_rgb_input = bool(
            getattr(dataset, "single_view_rgb_input", False)
        )
        self.dynamic_sampling_enabled = bool(
            getattr(dataset, "dynamic_sampling_enabled", False)
        )
        if any(
            index < 0 or index >= len(dataset)
            for index in self.indices
        ):
            raise IndexError(
                "source subset contains an out-of-range index"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if not callable(setter):
            return
        setter(int(epoch))
        if len(self.dataset) != self._dataset_length:
            raise RuntimeError(
                "indexed source dataset length changed in set_epoch; "
                "rebuild its index view and mixture"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        local_index = self.indices[index]
        value = self.dataset[local_index]
        if not isinstance(value, dict):
            raise TypeError(
                "mixture source datasets must return dictionaries"
            )
        result = dict(value)
        existing = result.get("source_type")
        if (
            existing is not None
            and existing != self.source_type
        ):
            raise RuntimeError(
                f"source dataset returned {existing!r}, "
                f"expected {self.source_type!r}"
            )
        result["source_type"] = self.source_type
        result.setdefault(
            "sample_key",
            f"{self.source_type}:{local_index:012d}",
        )
        return result


class SourceMixtureDataset(Dataset):
    """Concatenate named sources and expose stable source indices."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
    ) -> None:
        unknown = set(datasets) - set(SOURCE_NAMES)
        if unknown:
            raise ValueError(
                f"unknown mixture sources: {sorted(unknown)}"
            )
        if not datasets:
            raise ValueError(
                "mixture datasets may not be empty"
            )
        self.datasets = {
            name: datasets[name]
            for name in SOURCE_NAMES
            if name in datasets
        }
        self._layout: list[tuple[str, int, int]] = []
        self._source_indices: dict[
            str,
            tuple[int, ...],
        ] = {}
        offset = 0
        for name, dataset in self.datasets.items():
            length = len(dataset)
            self._layout.append(
                (name, offset, offset + length)
            )
            self._source_indices[name] = tuple(
                range(offset, offset + length)
            )
            offset += length
        self._length = offset
        self._is_panoramic = all(
            bool(getattr(dataset, "_is_panoramic", False))
            for dataset in self.datasets.values()
        )
        self.single_view_rgb_input = any(
            bool(
                getattr(
                    dataset,
                    "single_view_rgb_input",
                    False,
                )
            )
            for dataset in self.datasets.values()
        )
        self.dynamic_sampling_enabled = any(
            bool(
                getattr(
                    dataset,
                    "dynamic_sampling_enabled",
                    False,
                )
            )
            for dataset in self.datasets.values()
        )

    @property
    def source_indices(self) -> dict[str, tuple[int, ...]]:
        return dict(self._source_indices)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for source, start, end in self._layout:
            if start <= index < end:
                value = self.datasets[source][index - start]
                if not isinstance(value, dict):
                    raise TypeError(
                        "mixture source datasets must "
                        "return dictionaries"
                    )
                result = dict(value)
                existing = result.get("source_type")
                if (
                    existing is not None
                    and existing != source
                ):
                    raise RuntimeError(
                        f"source dataset returned "
                        f"{existing!r}, expected {source!r}"
                    )
                result["source_type"] = source
                result.setdefault(
                    "sample_key",
                    f"{source}:{index - start:012d}",
                )
                return result
        raise AssertionError(
            "mixture index layout is inconsistent"
        )

    def set_epoch(self, epoch: int) -> None:
        old_lengths = {
            name: len(dataset)
            for name, dataset in self.datasets.items()
        }
        for dataset in self.datasets.values():
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(int(epoch))
        new_lengths = {
            name: len(dataset)
            for name, dataset in self.datasets.items()
        }
        if new_lengths != old_lengths:
            raise RuntimeError(
                "source dataset length changed in set_epoch; "
                "rebuild mixture and sampler"
            )


def build_expert_dagger_mixture(
    expert_dataset: Dataset,
    dagger_dataset: TrajectoryDaggerDataset,
) -> SourceMixtureDataset:
    """Build expert/normal/hard views without copying source data."""
    indices = dagger_dataset.source_indices
    return SourceMixtureDataset(
        {
            "expert": IndexedSourceDataset(
                expert_dataset,
                tuple(range(len(expert_dataset))),
                "expert",
            ),
            "dagger_normal": IndexedSourceDataset(
                dagger_dataset,
                indices["dagger_normal"],
                "dagger_normal",
            ),
            "dagger_hard": IndexedSourceDataset(
                dagger_dataset,
                indices["dagger_hard"],
                "dagger_hard",
            ),
        }
    )


def _stable_hash_int(*parts: Any) -> int:
    payload = "\x1f".join(
        str(part) for part in parts
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )


class DeterministicMixtureSampler(Sampler[int]):
    """Weighted epoch plans with deterministic DDP striding.

    Every rank builds the same global plan and consumes
    plan[rank::num_replicas]. Recreating the sampler with the same seed and
    epoch reproduces the exact order for skip-based mid-epoch resume.
    """

    def __init__(
        self,
        dataset: SourceMixtureDataset,
        *,
        weights: Mapping[str, float] | None = None,
        profile: str | None = None,
        epoch_size: int | None = None,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = True,
    ) -> None:
        if not hasattr(dataset, "source_indices"):
            raise TypeError(
                "dataset must expose source_indices"
            )
        self.dataset = dataset
        if weights is None:
            selected_profile = (
                profile or DEFAULT_MIXTURE_PROFILE
            )
            if selected_profile not in MIXTURE_PROFILES:
                raise ValueError(
                    "unknown mixture profile: "
                    f"{selected_profile!r}"
                )
            raw = dict(MIXTURE_PROFILES[selected_profile])
            self.profile = selected_profile
        else:
            if profile is not None:
                raise ValueError(
                    "pass either weights or profile, not both"
                )
            raw = dict(weights)
            self.profile = "custom"
        unknown = set(raw) - set(SOURCE_NAMES)
        if unknown:
            raise ValueError(
                f"unknown mixture weights: {sorted(unknown)}"
            )
        self.weights: dict[str, float] = {}
        for source in SOURCE_NAMES:
            value = float(raw.get(source, 0.0))
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"invalid mixture weight for "
                    f"{source}: {value}"
                )
            self.weights[source] = value
        total_weight = sum(self.weights.values())
        if total_weight <= 0.0:
            raise ValueError(
                "at least one mixture weight must be positive"
            )
        self.weights = {
            source: value / total_weight
            for source, value in self.weights.items()
        }
        pools = dataset.source_indices
        for source, weight in self.weights.items():
            if weight > 0.0 and not pools.get(source):
                raise ValueError(
                    "positive mixture weight requires "
                    f"non-empty source {source!r}"
                )

        requested_size = (
            len(dataset)
            if epoch_size is None
            else int(epoch_size)
        )
        if requested_size <= 0:
            raise ValueError(
                "epoch_size must be positive"
            )
        if int(num_replicas) <= 0:
            raise ValueError(
                "num_replicas must be positive"
            )
        if (
            int(rank) < 0
            or int(rank) >= int(num_replicas)
        ):
            raise ValueError(
                "rank must be in [0, num_replicas)"
            )
        self.requested_epoch_size = requested_size
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        if self.drop_last:
            self.global_epoch_size = (
                requested_size // self.num_replicas
            ) * self.num_replicas
        else:
            self.global_epoch_size = (
                (
                    requested_size
                    + self.num_replicas
                    - 1
                )
                // self.num_replicas
            ) * self.num_replicas
        if self.global_epoch_size <= 0:
            raise ValueError(
                "epoch_size is smaller than num_replicas "
                "with drop_last=True"
            )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(
                "epoch must be non-negative"
            )
        self.epoch = int(epoch)

    def _allocate_source_counts(self) -> dict[str, int]:
        exact = {
            source: (
                self.global_epoch_size
                * self.weights[source]
            )
            for source in SOURCE_NAMES
        }
        counts = {
            source: int(math.floor(value))
            for source, value in exact.items()
        }
        remaining = (
            self.global_epoch_size
            - sum(counts.values())
        )
        priority = sorted(
            SOURCE_NAMES,
            key=lambda source: (
                -(exact[source] - counts[source]),
                SOURCE_NAMES.index(source),
            ),
        )
        for source in priority[:remaining]:
            counts[source] += 1
        return counts

    def source_counts_for_epoch(self) -> dict[str, int]:
        return self._allocate_source_counts()

    def global_plan(self) -> tuple[int, ...]:
        pools = self.dataset.source_indices
        plan: list[int] = []
        for source, count in (
            self._allocate_source_counts().items()
        ):
            if count <= 0:
                continue
            pool = tuple(
                int(index)
                for index in pools[source]
            )
            drawn: list[int] = []
            cycle = 0
            while len(drawn) < count:
                order = sorted(
                    pool,
                    key=lambda index: _stable_hash_int(
                        self.seed,
                        self.epoch,
                        source,
                        cycle,
                        index,
                    ),
                )
                take = min(
                    count - len(drawn),
                    len(order),
                )
                drawn.extend(order[:take])
                cycle += 1
            plan.extend(drawn)
        decorated = list(enumerate(plan))
        decorated.sort(
            key=lambda pair: _stable_hash_int(
                self.seed,
                self.epoch,
                "global",
                pair[0],
                pair[1],
            )
        )
        output = tuple(
            index for _, index in decorated
        )
        if len(output) != self.global_epoch_size:
            raise AssertionError(
                "mixture sampler built an invalid global plan"
            )
        return output

    def __iter__(self) -> Iterator[int]:
        return iter(
            self.global_plan()[
                self.rank :: self.num_replicas
            ]
        )

    def __len__(self) -> int:
        return (
            self.global_epoch_size
            // self.num_replicas
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "heatmapvln-deterministic-"
                "mixture-sampler-v1"
            ),
            "epoch": self.epoch,
            "seed": self.seed,
            "requested_epoch_size": (
                self.requested_epoch_size
            ),
            "global_epoch_size": self.global_epoch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
            "profile": self.profile,
            "weights": dict(self.weights),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        expected = self.state_dict()
        for key in (
            "schema",
            "seed",
            "requested_epoch_size",
            "global_epoch_size",
            "num_replicas",
            "drop_last",
            "profile",
            "weights",
        ):
            if state.get(key) != expected[key]:
                raise ValueError(
                    "mixture sampler state mismatch "
                    f"for {key}"
                )
        self.set_epoch(int(state.get("epoch", 0)))
