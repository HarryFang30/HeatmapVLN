"""Pinned released-InternNav dependency contract for formal control training.

The formal launcher is the sole process that hashes the large model shards.
After that one-time verification it exports a small, immutable environment
contract.  Every torchrun rank validates those scalar values and records the
same closure in the runtime config; ranks deliberately do not re-hash model
files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any


NATIVE_DEPENDENCY_SCHEMA = "native-internnav-checkpoint-v1"
NATIVE_MODEL_PATH = "/mnt/afs/lixiaoou/intern/fjl/InternNav-Model"
NATIVE_MODEL_MANIFEST_PATH = (
    "/mnt/afs/lixiaoou/intern/fjl/evaluation_plans/"
    "internnav_native_r2r_val_unseen_8gpu_20260802/manifests/"
    "internnav_model.sha256"
)
NATIVE_MODEL_MANIFEST_SHA256 = (
    "f37a6df2e0703e38c34ccdba89c861bb8490ad3a36201bc1ec24a7509bf56581"
)
NATIVE_MODEL_FILE_COUNT = 14
RUNTIME_KEY = "native_internnav_dependency"

ENV_SCHEMA = "HEATMAPVLN_NATIVE_DEPENDENCY_SCHEMA"
ENV_MODEL_PATH = "HEATMAPVLN_NATIVE_MODEL_PATH"
ENV_MANIFEST_PATH = "HEATMAPVLN_NATIVE_MODEL_MANIFEST_PATH"
ENV_MANIFEST_SHA256 = "HEATMAPVLN_NATIVE_MODEL_MANIFEST_SHA256"
ENV_FILE_COUNT = "HEATMAPVLN_NATIVE_MODEL_FILE_COUNT"
ENV_VERIFIED = "HEATMAPVLN_NATIVE_MODEL_VERIFIED"


class NativeInternNavDependencyError(RuntimeError):
    """The released native model closure is missing, mutable, or unverified."""


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": NATIVE_DEPENDENCY_SCHEMA,
        "model_path": NATIVE_MODEL_PATH,
        "manifest_path": NATIVE_MODEL_MANIFEST_PATH,
        "manifest_sha256": NATIVE_MODEL_MANIFEST_SHA256,
        "file_count": NATIVE_MODEL_FILE_COUNT,
        "verified": True,
    }


def validate_native_internnav_dependency_contract(
    value: Any,
    *,
    expected_model_path: str | None = None,
    name: str = RUNTIME_KEY,
) -> dict[str, Any]:
    """Return a canonical copy only for the exact released-model closure."""

    if not isinstance(value, Mapping):
        raise NativeInternNavDependencyError(f"{name} must be a mapping")
    expected = _expected_contract()
    actual_keys = set(value)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise NativeInternNavDependencyError(
            f"{name} fields differ from the locked closure: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
        or type(value.get(key)) is not type(expected_value)
    }
    if expected_model_path is not None and expected_model_path != NATIVE_MODEL_PATH:
        mismatches["config_model_path"] = {
            "expected": NATIVE_MODEL_PATH,
            "actual": expected_model_path,
        }
    if mismatches:
        raise NativeInternNavDependencyError(
            f"{name} differs from the locked released InternNav closure: "
            f"{mismatches}"
        )
    return dict(expected)


def inject_native_internnav_dependency_from_env(
    cfg: MutableMapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inject the launcher-verified closure without hashing on torchrun ranks."""

    env = os.environ if environ is None else environ
    required = (
        ENV_SCHEMA,
        ENV_MODEL_PATH,
        ENV_MANIFEST_PATH,
        ENV_MANIFEST_SHA256,
        ENV_FILE_COUNT,
        ENV_VERIFIED,
    )
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise NativeInternNavDependencyError(
            "formal heatmap-control training requires the launcher's verified "
            f"native-model environment contract; missing={missing}"
        )
    try:
        file_count = int(env[ENV_FILE_COUNT], 10)
    except (TypeError, ValueError) as exc:
        raise NativeInternNavDependencyError(
            f"{ENV_FILE_COUNT} must be a base-10 integer"
        ) from exc
    verified_raw = env[ENV_VERIFIED]
    if verified_raw != "1":
        raise NativeInternNavDependencyError(
            f"{ENV_VERIFIED} must be exactly '1' after launcher verification"
        )
    candidate = {
        "schema": env[ENV_SCHEMA],
        "model_path": env[ENV_MODEL_PATH],
        "manifest_path": env[ENV_MANIFEST_PATH],
        "manifest_sha256": env[ENV_MANIFEST_SHA256],
        "file_count": file_count,
        "verified": True,
    }

    try:
        model = cfg["model"]
        llm_path = model["llm"]["model_path"]
        nextdit_path = model["action_head"]["nextdit"]["internnav_model_path"]
    except (KeyError, TypeError) as exc:
        raise NativeInternNavDependencyError(
            "config lacks the native InternNav LLM/System1 model paths"
        ) from exc
    if llm_path != nextdit_path:
        raise NativeInternNavDependencyError(
            "config LLM and NextDiT do not share one native InternNav model path"
        )
    contract = validate_native_internnav_dependency_contract(
        candidate,
        expected_model_path=llm_path,
        name="launcher native-model environment contract",
    )
    runtime = cfg.setdefault("runtime", {})
    if not isinstance(runtime, MutableMapping):
        raise NativeInternNavDependencyError("config.runtime must be mutable mapping")
    existing = runtime.get(RUNTIME_KEY)
    if existing is not None:
        validated_existing = validate_native_internnav_dependency_contract(
            existing,
            expected_model_path=llm_path,
            name=f"config.runtime.{RUNTIME_KEY}",
        )
        if validated_existing != contract:
            raise NativeInternNavDependencyError(
                "config native dependency may not be replaced by launcher injection"
            )
    runtime[RUNTIME_KEY] = contract
    return contract


__all__ = [
    "ENV_FILE_COUNT",
    "ENV_MANIFEST_PATH",
    "ENV_MANIFEST_SHA256",
    "ENV_MODEL_PATH",
    "ENV_SCHEMA",
    "ENV_VERIFIED",
    "NATIVE_DEPENDENCY_SCHEMA",
    "NATIVE_MODEL_FILE_COUNT",
    "NATIVE_MODEL_MANIFEST_PATH",
    "NATIVE_MODEL_MANIFEST_SHA256",
    "NATIVE_MODEL_PATH",
    "NativeInternNavDependencyError",
    "RUNTIME_KEY",
    "inject_native_internnav_dependency_from_env",
    "validate_native_internnav_dependency_contract",
]
