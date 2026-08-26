#!/usr/bin/env python3
"""Finalize a complete set of sealed native-InternNav DAgger shards.

This fail-closed aggregate validator proves that cohort shards cover the full
R2R train split exactly once, keep canonical routes within one shard, and map
one-to-one to sealed native InternNav collections. It then writes the
authoritative multi-root training manifest without copying images or heatmaps.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import build_r2r_train_dagger_cohort as cohort
from scripts.tools import build_r2r_train_dagger_shards as shard_builder
from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset


FJL_ROOT = Path("/mnt/afs/liwenhao/agent/370910109")
OUTPUT_SCHEMA = "heatmapvln-trajectory-dagger-training-roots-v1"
COLLECTION_SCHEMA = "heatmapvln-trajectory-dagger-collection-v1"
WRAPPER_SCHEMA = "heatmap-system1-trajectory-dagger-wrapper-v4"
NATIVE_MODE = "internnav_native"
NATIVE_PROTOCOL = "internnav-native-joint-front-history-lookdown-v1"
FINGERPRINT_RE = re.compile(r"internnav-native-v1:[0-9a-f]{64}")
ABSOLUTE_MAX_BYTES = 300_000_000_000
DEFAULT_EXPECTED_EPISODES = 10_819
DEFAULT_EXPECTED_NUM_SHARDS = 8
DEEP_VALIDATOR = (
    REPO_ROOT
    / "scripts/tools/validate_trajectory_dagger_collection.py"
)
TreeEntrySnapshot = tuple[int, int, int, int, int, str]
TreeSnapshot = dict[str, TreeEntrySnapshot]


class FinalizeError(RuntimeError):
    """Raised when the aggregate collection cannot be trusted."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _root() -> Path:
    try:
        root = FJL_ROOT.resolve(strict=True)
    except OSError as exc:
        raise FinalizeError(f"FJL root is unavailable: {exc}") from exc
    if not root.is_dir():
        raise FinalizeError(f"FJL root is not a directory: {root}")
    return root


def _resolve_existing(raw: str, root: Path, label: str, *, directory: bool) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FinalizeError(f"{label} does not exist: {candidate}: {exc}") from exc
    if not _is_within(resolved, root):
        raise FinalizeError(f"{label} escapes FJL root: {resolved}")
    if candidate.is_symlink():
        raise FinalizeError(f"{label} may not be a symlink: {candidate}")
    if directory and not resolved.is_dir():
        raise FinalizeError(f"{label} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise FinalizeError(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_output(raw: str, root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise FinalizeError(f"invalid output path: {candidate}")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise FinalizeError(f"output parent does not exist: {candidate.parent}") from exc
    resolved = parent / candidate.name
    if not _is_within(resolved, root):
        raise FinalizeError(f"output escapes FJL root: {resolved}")
    if resolved.is_symlink() or (resolved.exists() and not resolved.is_file()):
        raise FinalizeError(f"invalid output target: {resolved}")
    return resolved


def _load_json(path: Path, label: str, max_bytes: int = 64 * 1024 * 1024) -> Any:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise FinalizeError(f"cannot stat {label}: {path}: {exc}") from exc
    if size <= 0 or size > max_bytes:
        raise FinalizeError(f"{label} has invalid size {size}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"cannot read {label}: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_regular_file_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FinalizeError(
            f"cannot safely open collection file {path}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FinalizeError(
            f"cannot hash collection file {path}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise FinalizeError(f"{label} has invalid expected SHA256")
    actual = _sha256(path)
    if actual != expected:
        raise FinalizeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def _tree_size(root: Path) -> int:
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise FinalizeError(f"symlinked directory in collection: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise FinalizeError(f"unsafe file in collection: {path}")
            total += path.stat().st_size
    return total


def _tree_snapshot(
    root: Path,
    *,
    max_bytes: int = ABSOLUTE_MAX_BYTES,
) -> tuple[int, TreeSnapshot]:
    total = 0
    snapshot: TreeSnapshot = {}
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda entry: entry.name,
            )
        except OSError as exc:
            raise FinalizeError(
                f"cannot scan collection directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise FinalizeError(
                    f"cannot stat collection entry {path}: {exc}"
                ) from exc
            if entry.is_symlink():
                raise FinalizeError(
                    f"symlinks are forbidden in a collection: {path}"
                )
            relative = path.relative_to(root).as_posix()
            metadata = (
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_ino,
            )
            if entry.is_dir(follow_symlinks=False):
                snapshot[relative + "/"] = (*metadata, "")
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                total += stat.st_size
                if total > max_bytes:
                    raise FinalizeError(
                        "collection exceeds snapshot limit: "
                        f"{total} > {max_bytes}"
                    )
                content_sha256 = _sha256_regular_file_no_follow(path)
                try:
                    after = os.stat(path, follow_symlinks=False)
                except OSError as exc:
                    raise FinalizeError(
                        f"cannot restat collection file {path}: {exc}"
                    ) from exc
                after_metadata = (
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_ino,
                )
                if after_metadata != metadata:
                    raise FinalizeError(
                        f"collection file changed while hashing: {path}"
                    )
                snapshot[relative] = (*metadata, content_sha256)
            else:
                raise FinalizeError(
                    f"non-regular collection entry is forbidden: {path}"
                )
    return total, snapshot


def _assert_tree_unchanged(
    root: Path,
    expected: TreeSnapshot,
    *,
    max_bytes: int,
    label: str,
) -> int:
    total, current = _tree_snapshot(root, max_bytes=max_bytes)
    if current != expected:
        raise FinalizeError(f"{label} changed during finalization")
    return total


def _contract_invariant(contract: dict[str, Any]) -> bytes:
    episode_cohort = contract.get("episode_cohort")
    if not isinstance(episode_cohort, dict) or set(episode_cohort) != {
        "path",
        "sha256",
        "max_episodes",
    }:
        raise FinalizeError(
            "collection contract episode_cohort is not canonical"
        )
    invariant = dict(contract)
    invariant["episode_cohort"] = {
        "max_episodes": episode_cohort.get("max_episodes"),
    }
    return _canonical_json_bytes(invariant)


def _run_deep_validator(
    collection_root: Path,
    control_root: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if not DEEP_VALIDATOR.is_file() or DEEP_VALIDATOR.is_symlink():
        raise FinalizeError(
            f"deep collection validator is unavailable: {DEEP_VALIDATOR}"
        )
    command = [
        sys.executable,
        str(DEEP_VALIDATOR),
        "--collection-root",
        str(collection_root),
        "--control-root",
        str(control_root),
        "--max-bytes",
        str(max_bytes),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise FinalizeError(
            f"cannot execute deep validator for {collection_root}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FinalizeError(
            "deep validator rejected collection "
            f"{collection_root}: {detail[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise FinalizeError(
            f"deep validator produced no result for {collection_root}"
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise FinalizeError(
            f"deep validator returned invalid JSON for {collection_root}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise FinalizeError(
            f"deep validator returned an invalid result for {collection_root}"
        )
    return payload


def _episode_key(raw: dict[str, Any], label: str) -> tuple[str, int]:
    try:
        scene = cohort._scene_name(raw.get("scene_id"))
        episode_id = cohort._episode_id(raw.get("episode_id"))
    except cohort.CohortError as exc:
        raise FinalizeError(f"{label}: {exc}") from exc
    return scene, episode_id


def _validate_plan(
    plan_path: Path,
    *,
    expected_episode_count: int = DEFAULT_EXPECTED_EPISODES,
    expected_num_shards: int = DEFAULT_EXPECTED_NUM_SHARDS,
) -> dict[str, Any]:
    plan = _load_json(plan_path, "shard plan")
    if not isinstance(plan, dict) or plan.get("schema") != shard_builder.PLAN_SCHEMA:
        raise FinalizeError("wrong shard plan schema")
    if plan.get("split") != "train" or plan.get("route_grouped") is not True:
        raise FinalizeError("shard plan must be route-grouped R2R train")
    if plan.get("partition_strategy") != shard_builder.PARTITION_STRATEGY:
        raise FinalizeError("unexpected shard partition strategy")
    num_shards = plan.get("num_shards")
    if num_shards != expected_num_shards:
        raise FinalizeError(
            f"exactly {expected_num_shards} shards are required, got {num_shards!r}"
        )
    if plan.get("selected_episode_count") != expected_episode_count:
        raise FinalizeError(f"expected {expected_episode_count} selected episodes")
    seed = plan.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FinalizeError("shard plan seed must be an integer")

    dataset_info = plan.get("dataset")
    if not isinstance(dataset_info, dict):
        raise FinalizeError("plan.dataset must be an object")
    dataset_path = _resolve_existing(
        str(dataset_info.get("path")),
        _root(),
        "plan dataset",
        directory=False,
    )
    _require_sha(dataset_path, dataset_info.get("sha256"), "plan dataset")
    canonical = cohort._load_episodes(dataset_path)
    canonical_by_key = {
        cohort._episode_key(episode): episode for episode in canonical
    }
    canonical_keys = set(canonical_by_key)
    canonical_routes = {cohort._route_key(episode) for episode in canonical}
    if len(canonical_keys) != expected_episode_count:
        raise FinalizeError(
            f"canonical dataset has {len(canonical_keys)} episodes, "
            f"expected {expected_episode_count}"
        )
    if dataset_info.get("episode_count") != len(canonical_keys):
        raise FinalizeError("plan dataset episode count mismatch")
    if dataset_info.get("route_count") != len(canonical_routes):
        raise FinalizeError("plan dataset route count mismatch")
    if dataset_info.get("scene_count") != len(
        {episode["scene_id"] for episode in canonical}
    ):
        raise FinalizeError("plan dataset scene count mismatch")
    if plan.get("selected_route_count") != len(canonical_routes):
        raise FinalizeError("plan selected route count mismatch")
    if plan.get("episode_key_sha256") != shard_builder._key_digest(canonical_keys):
        raise FinalizeError("plan global episode digest mismatch")
    if plan.get("route_key_sha256") != shard_builder._key_digest(canonical_routes):
        raise FinalizeError("plan global route digest mismatch")

    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != num_shards:
        raise FinalizeError(
            f"plan must contain exactly {expected_num_shards} shard entries"
        )

    seen_episodes: set[tuple[str, int]] = set()
    route_owners: dict[tuple[str, int], int] = {}
    shard_audits: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_shards):
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise FinalizeError(f"invalid plan shard entry {index}")
        filename = entry.get("file")
        if not isinstance(filename, str) or not re.fullmatch(
            rf"shard_{index:02d}\.json", filename
        ):
            raise FinalizeError(f"invalid shard filename at index {index}")
        cohort_path = _resolve_existing(
            str(plan_path.parent / filename),
            _root(),
            f"shard {index} cohort",
            directory=False,
        )
        if cohort_path.parent != plan_path.parent:
            raise FinalizeError(f"shard {index} cohort escapes plan directory")
        cohort_sha = _require_sha(
            cohort_path,
            entry.get("sha256"),
            f"shard {index} cohort",
        )
        payload = _load_json(cohort_path, f"shard {index} cohort")
        rows = payload.get("episodes") if isinstance(payload, dict) else None
        if (
            not isinstance(rows, list)
            or payload.get("split") != "train"
            or payload.get("count") != len(rows)
            or len(rows) != entry.get("episode_count")
        ):
            raise FinalizeError(f"shard {index} cohort count contract mismatch")

        shard_keys: set[tuple[str, int]] = set()
        shard_routes: set[tuple[str, int]] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise FinalizeError(f"shard {index} episode {row_index} is invalid")
            key = _episode_key(row, f"shard {index} episode {row_index}")
            canonical_episode = canonical_by_key.get(key)
            if canonical_episode is None:
                raise FinalizeError(f"shard {index} contains unknown episode {key}")
            if row.get("instruction") != canonical_episode["instruction"]:
                raise FinalizeError(
                    f"shard {index} instruction disagrees for episode {key}"
                )
            if key in shard_keys or key in seen_episodes:
                raise FinalizeError(f"duplicate episode across shards: {key}")
            shard_keys.add(key)
            seen_episodes.add(key)
            route = cohort._route_key(canonical_episode)
            owner = route_owners.setdefault(route, index)
            if owner != index:
                raise FinalizeError(
                    f"canonical route {route} crosses shards {owner}/{index}"
                )
            shard_routes.add(route)

        if len(shard_routes) != entry.get("route_count"):
            raise FinalizeError(f"shard {index} route count mismatch")
        if entry.get("scene_count") != len(
            {scene for scene, _ in shard_keys}
        ):
            raise FinalizeError(f"shard {index} scene count mismatch")
        if entry.get("episode_key_sha256") != shard_builder._key_digest(shard_keys):
            raise FinalizeError(f"shard {index} episode digest mismatch")
        if entry.get("route_key_sha256") != shard_builder._key_digest(shard_routes):
            raise FinalizeError(f"shard {index} route digest mismatch")
        shard_audits.append(
            {
                "index": index,
                "cohort_path": cohort_path,
                "cohort_sha256": cohort_sha,
                "episode_count": len(shard_keys),
                "route_count": len(shard_routes),
            }
        )

    if seen_episodes != canonical_keys:
        raise FinalizeError("shard episode union is not the full R2R train split")
    if set(route_owners) != canonical_routes:
        raise FinalizeError("shard route union is incomplete")
    try:
        expected_shards = shard_builder._route_grouped_shards(
            canonical,
            num_shards=num_shards,
            seed=seed,
        )
        expected_artifacts = shard_builder._expected_files(
            dataset_path,
            canonical,
            expected_shards,
            seed=seed,
        )
    except shard_builder.ShardError as exc:
        raise FinalizeError(
            f"cannot reconstruct deterministic shard plan: {exc}"
        ) from exc
    if plan_path.read_bytes() != expected_artifacts["plan.json"]:
        raise FinalizeError(
            "shard plan does not match canonical deterministic reconstruction"
        )
    for audit in shard_audits:
        filename = f"shard_{audit['index']:02d}.json"
        if audit["cohort_path"].read_bytes() != expected_artifacts[filename]:
            raise FinalizeError(
                f"shard {audit['index']} does not match deterministic reconstruction"
            )
    return {
        "plan": plan,
        "plan_sha256": _sha256(plan_path),
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_info["sha256"],
        "episode_count": len(canonical_keys),
        "route_count": len(canonical_routes),
        "scene_count": len({key[0] for key in canonical_keys}),
        "shards": shard_audits,
    }


def _validate_collection(
    shard: dict[str, Any],
    collection_base: Path,
    control_base: Path,
    plan_audit: dict[str, Any],
    *,
    deep_max_bytes: int,
) -> dict[str, Any]:
    index = shard["index"]
    name = f"shard_{index:02d}"
    collection_root = _resolve_existing(
        str(collection_base / name),
        _root(),
        f"{name} collection",
        directory=True,
    )
    control_root = _resolve_existing(
        str(control_base / name),
        _root(),
        f"{name} control",
        directory=True,
    )
    manifest_path = _resolve_existing(
        str(collection_root / "collection_manifest.json"),
        _root(),
        f"{name} collection manifest",
        directory=False,
    )
    wrapper_path = _resolve_existing(
        str(control_root / "collection_wrapper_manifest.json"),
        _root(),
        f"{name} wrapper manifest",
        directory=False,
    )
    progress_path = _resolve_existing(
        str(control_root / "progress.json"),
        _root(),
        f"{name} control progress",
        directory=False,
    )
    result_path = _resolve_existing(
        str(control_root / "result.json"),
        _root(),
        f"{name} control result",
        directory=False,
    )

    actual_bytes, collection_snapshot = _tree_snapshot(
        collection_root,
        max_bytes=deep_max_bytes,
    )
    critical_hashes = {
        "manifest": _sha256(manifest_path),
        "wrapper": _sha256(wrapper_path),
        "progress": _sha256(progress_path),
        "result": _sha256(result_path),
    }

    manifest = _load_json(manifest_path, f"{name} collection manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != COLLECTION_SCHEMA
        or manifest.get("ready") is not True
    ):
        raise FinalizeError(f"{name} collection is not sealed")
    contract = manifest.get("contract")
    summary = manifest.get("summary")
    if not isinstance(contract, dict) or not isinstance(summary, dict):
        raise FinalizeError(f"{name} manifest is missing contract/summary")
    fingerprint = contract.get("rpc_policy_fingerprint")
    if (
        contract.get("rpc_policy_mode") != NATIVE_MODE
        or contract.get("native_protocol") != NATIVE_PROTOCOL
        or not isinstance(fingerprint, str)
        or FINGERPRINT_RE.fullmatch(fingerprint) is None
        or contract.get("policy_fingerprint") != fingerprint
    ):
        raise FinalizeError(f"{name} is not a native InternNav collection")
    if (
        contract.get("data_path") != str(plan_audit["dataset_path"])
        or contract.get("data_sha256") != plan_audit["dataset_sha256"]
    ):
        raise FinalizeError(f"{name} canonical dataset contract mismatch")
    cohort_contract = contract.get("episode_cohort")
    if not isinstance(cohort_contract, dict):
        raise FinalizeError(f"{name} is missing episode cohort contract")
    if (
        cohort_contract.get("path") != str(shard["cohort_path"])
        or cohort_contract.get("sha256") != shard["cohort_sha256"]
    ):
        raise FinalizeError(f"{name} cohort contract mismatch")
    expected_count = shard["episode_count"]
    if (
        summary.get("expected_episodes") != expected_count
        or summary.get("processed_episodes") != expected_count
    ):
        raise FinalizeError(f"{name} sealed episode counts disagree with cohort")
    committed = summary.get("committed_episodes")
    no_sample = summary.get("no_sample_episodes")
    if (
        isinstance(committed, bool)
        or not isinstance(committed, int)
        or committed < 0
        or isinstance(no_sample, bool)
        or not isinstance(no_sample, int)
        or no_sample < 0
        or committed + no_sample != expected_count
    ):
        raise FinalizeError(
            f"{name} committed/no-sample counts do not cover the cohort"
        )
    if summary.get("control_progress_sha256") != _sha256(progress_path):
        raise FinalizeError(f"{name} control progress digest mismatch")
    if summary.get("control_result_sha256") != _sha256(result_path):
        raise FinalizeError(f"{name} control result digest mismatch")

    wrapper = _load_json(wrapper_path, f"{name} wrapper manifest")
    verified = wrapper.get("verified_policy") if isinstance(wrapper, dict) else None
    requested = (
        wrapper.get("identity", {}).get("requested_policy")
        if isinstance(wrapper, dict)
        else None
    )
    if (
        wrapper.get("schema") != WRAPPER_SCHEMA
        or wrapper.get("verification_status") != "sealed_native_verified"
        or not isinstance(verified, dict)
        or verified.get("rpc_policy_mode") != NATIVE_MODE
        or verified.get("native_protocol") != NATIVE_PROTOCOL
        or verified.get("policy_fingerprint") != fingerprint
        or not isinstance(requested, dict)
        or requested.get("system2") != "internnav_native_qwen"
        or requested.get("system1") != "internnav_native_nextdit_async"
        or requested.get("external_checkpoint") is not False
        or requested.get("lora") is not False
        or requested.get("adapter") is not False
    ):
        raise FinalizeError(f"{name} wrapper native provenance mismatch")

    result = _load_json(result_path, f"{name} control result")
    if (
        not isinstance(result, dict)
        or result.get("total_episodes") != expected_count
        or result.get("rpc_policy_mode") != NATIVE_MODE
        or result.get("rpc_policy_fingerprint") != fingerprint
        or result.get("native_protocol") != NATIVE_PROTOCOL
    ):
        raise FinalizeError(f"{name} result policy/count mismatch")

    contract_invariant = _contract_invariant(contract)
    deep_result = _run_deep_validator(
        collection_root,
        control_root,
        max_bytes=deep_max_bytes,
    )
    expected_deep = {
        "collection_root": str(collection_root),
        "manifest_ready": True,
        "episodes": committed,
        "samples": summary.get("samples"),
        "frames": summary.get("frames"),
        "jpegs": summary.get("jpegs"),
        "tar_payload_bytes": summary.get("tar_payload_bytes"),
        "capacity_bytes": actual_bytes,
        "sealed_now": False,
    }
    mismatches = {
        field: {
            "expected": expected,
            "actual": deep_result.get(field),
        }
        for field, expected in expected_deep.items()
        if deep_result.get(field) != expected
    }
    if mismatches:
        raise FinalizeError(
            f"{name} deep validator summary mismatch: {mismatches}"
        )
    _assert_tree_unchanged(
        collection_root,
        collection_snapshot,
        max_bytes=deep_max_bytes,
        label=f"{name} collection",
    )
    current_critical_hashes = {
        "manifest": _sha256(manifest_path),
        "wrapper": _sha256(wrapper_path),
        "progress": _sha256(progress_path),
        "result": _sha256(result_path),
    }
    if current_critical_hashes != critical_hashes:
        raise FinalizeError(
            f"{name} control or manifest changed during deep validation"
        )

    return {
        **shard,
        "collection_root": collection_root,
        "control_root": control_root,
        "collection_manifest_sha256": critical_hashes["manifest"],
        "control_progress_sha256": critical_hashes["progress"],
        "control_result_sha256": critical_hashes["result"],
        "fingerprint": fingerprint,
        "samples": summary.get("samples"),
        "frames": summary.get("frames"),
        "jpegs": summary.get("jpegs"),
        "tar_payload_bytes": summary.get("tar_payload_bytes"),
        "actual_bytes": actual_bytes,
        "contract_invariant": contract_invariant,
        "contract_invariant_sha256": hashlib.sha256(
            contract_invariant
        ).hexdigest(),
        "collection_snapshot": collection_snapshot,
        "critical_paths": {
            "manifest": manifest_path,
            "wrapper": wrapper_path,
            "progress": progress_path,
            "result": result_path,
        },
        "critical_hashes": critical_hashes,
    }


def _assert_collection_audit_unchanged(
    audit: dict[str, Any],
    *,
    max_bytes: int,
) -> None:
    name = f"shard_{audit['index']:02d}"
    current_bytes = _assert_tree_unchanged(
        audit["collection_root"],
        audit["collection_snapshot"],
        max_bytes=max_bytes,
        label=f"{name} collection",
    )
    if current_bytes != audit["actual_bytes"]:
        raise FinalizeError(f"{name} collection size changed")
    current_hashes = {
        key: _sha256(path)
        for key, path in audit["critical_paths"].items()
    }
    if current_hashes != audit["critical_hashes"]:
        raise FinalizeError(
            f"{name} control or manifest changed during finalization"
        )


def _require_single_contract_invariant(
    audits: list[dict[str, Any]],
) -> str:
    invariants = {item["contract_invariant"] for item in audits}
    if len(invariants) != 1:
        raise FinalizeError(
            "collection contract invariants differ across shards"
        )
    return hashlib.sha256(next(iter(invariants))).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_idempotent(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise FinalizeError(f"refusing to overwrite different manifest: {path}")
        return False
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except FileExistsError as exc:
        raise FinalizeError(f"output appeared concurrently: {path}") from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--collection-base", required=True)
    parser.add_argument("--control-base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-bytes", type=int, default=ABSOLUTE_MAX_BYTES)
    parser.add_argument(
        "--expected-num-shards",
        type=int,
        default=DEFAULT_EXPECTED_NUM_SHARDS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root = _root()
        if args.max_bytes <= 0 or args.max_bytes > ABSOLUTE_MAX_BYTES:
            raise FinalizeError(
                f"max-bytes must be in [1, {ABSOLUTE_MAX_BYTES}]"
            )
        if not 1 <= args.expected_num_shards <= DEFAULT_EXPECTED_EPISODES:
            raise FinalizeError(
                "expected-num-shards must be in "
                f"[1, {DEFAULT_EXPECTED_EPISODES}]"
            )
        plan_path = _resolve_existing(args.plan, root, "shard plan", directory=False)
        collection_base = _resolve_existing(
            args.collection_base, root, "collection base", directory=True
        )
        control_base = _resolve_existing(
            args.control_base, root, "control base", directory=True
        )
        output_path = _resolve_output(args.output, root)
        plan_audit = _validate_plan(
            plan_path,
            expected_num_shards=args.expected_num_shards,
        )
        collection_audits = [
            _validate_collection(
                shard,
                collection_base,
                control_base,
                plan_audit,
                deep_max_bytes=args.max_bytes,
            )
            for shard in plan_audit["shards"]
        ]
        fingerprints = {item["fingerprint"] for item in collection_audits}
        if len(fingerprints) != 1:
            raise FinalizeError(
                f"native policy fingerprints differ across shards: {fingerprints}"
            )
        fingerprint = next(iter(fingerprints))
        contract_invariant_sha256 = _require_single_contract_invariant(
            collection_audits
        )
        total_bytes = sum(item["actual_bytes"] for item in collection_audits)
        if total_bytes > args.max_bytes:
            raise FinalizeError(
                f"aggregate collection exceeds budget: {total_bytes} > {args.max_bytes}"
            )

        collection_roots = [
            str(item["collection_root"]) for item in collection_audits
        ]
        per_shard_datasets = [
            TrajectoryDaggerDataset(
                item["collection_root"],
                expected_policy_mode=NATIVE_MODE,
                expected_policy_fingerprint=fingerprint,
            )
            for item in collection_audits
        ]
        combined = TrajectoryDaggerDataset(
            collection_roots,
            expected_policy_mode=NATIVE_MODE,
            expected_policy_fingerprint=fingerprint,
        )
        expected_samples = sum(item["samples"] for item in collection_audits)
        if len(combined) != expected_samples:
            raise FinalizeError(
                f"combined sample count mismatch: {len(combined)} != {expected_samples}"
            )
        if sum(len(dataset) for dataset in per_shard_datasets) != len(combined):
            raise FinalizeError("multi-root dataset contains cross-shard duplicates")

        output = {
            "schema": OUTPUT_SCHEMA,
            "ready": True,
            "dataset": {
                "path": str(plan_audit["dataset_path"]),
                "sha256": plan_audit["dataset_sha256"],
                "episodes": plan_audit["episode_count"],
                "routes": plan_audit["route_count"],
                "scenes": plan_audit["scene_count"],
            },
            "partition": {
                "plan_path": str(plan_path),
                "plan_sha256": plan_audit["plan_sha256"],
                "seed": plan_audit["plan"]["seed"],
                "num_shards": plan_audit["plan"]["num_shards"],
                "unit": "canonical_route",
                "strategy": plan_audit["plan"]["partition_strategy"],
            },
            "policy": {
                "mode": NATIVE_MODE,
                "fingerprint": fingerprint,
                "protocol": NATIVE_PROTOCOL,
                "system2": "internnav_native_qwen",
                "system1": "internnav_native_nextdit_async",
                "external_checkpoint": False,
                "lora": False,
                "adapter": False,
                "collection_contract_invariant_sha256": (
                    contract_invariant_sha256
                ),
            },
            "global_capacity": {
                "limit_bytes": args.max_bytes,
                "actual_bytes": total_bytes,
            },
            "collection_roots": collection_roots,
            "sample_count": len(combined),
            "source_counts": combined.source_counts,
            "training_mixture": {
                "expert": 0.5,
                "dagger_normal": 0.2,
                "dagger_hard": 0.3,
                "basis": "per_training_sample",
            },
            "storage_policy": {
                "copy_existing_images": False,
                "persist_predicted_heatmaps": False,
                "online_heatmap_generation": True,
            },
            "shards": [
                {
                    "index": item["index"],
                    "cohort": {
                        "path": str(item["cohort_path"]),
                        "sha256": item["cohort_sha256"],
                        "episodes": item["episode_count"],
                        "routes": item["route_count"],
                    },
                    "collection_root": str(item["collection_root"]),
                    "collection_manifest_sha256": item[
                        "collection_manifest_sha256"
                    ],
                    "control_root": str(item["control_root"]),
                    "control_progress_sha256": item["control_progress_sha256"],
                    "control_result_sha256": item["control_result_sha256"],
                    "samples": item["samples"],
                    "source_counts": dataset.source_counts,
                    "frames": item["frames"],
                    "jpegs": item["jpegs"],
                    "tar_payload_bytes": item["tar_payload_bytes"],
                    "actual_bytes": item["actual_bytes"],
                }
                for item, dataset in zip(
                    collection_audits,
                    per_shard_datasets,
                    strict=True,
                )
            ],
        }
        if _sha256(plan_path) != plan_audit["plan_sha256"]:
            raise FinalizeError("shard plan changed during finalization")
        for shard in plan_audit["shards"]:
            if _sha256(shard["cohort_path"]) != shard["cohort_sha256"]:
                raise FinalizeError(
                    f"shard {shard['index']} cohort changed during finalization"
                )
        for audit in collection_audits:
            _assert_collection_audit_unchanged(
                audit,
                max_bytes=args.max_bytes,
            )
        created = _write_idempotent(output_path, _json_bytes(output))
    except (
        FinalizeError,
        cohort.CohortError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "created" if created else "verified_existing",
                "output": str(output_path),
                "episodes": output["dataset"]["episodes"],
                "routes": output["dataset"]["routes"],
                "samples": output["sample_count"],
                "source_counts": output["source_counts"],
                "actual_bytes": output["global_capacity"]["actual_bytes"],
                "policy_fingerprint": output["policy"]["fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
