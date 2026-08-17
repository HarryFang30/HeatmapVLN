"""Tests for the launcher-to-torchrun native model closure contract."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.training.checkpoint import CheckpointManager
from scripts.training.native_internnav_dependency import (
    ENV_FILE_COUNT,
    ENV_MANIFEST_PATH,
    ENV_MANIFEST_SHA256,
    ENV_MODEL_PATH,
    ENV_SCHEMA,
    ENV_VERIFIED,
    NATIVE_DEPENDENCY_SCHEMA,
    NATIVE_MODEL_FILE_COUNT,
    NATIVE_MODEL_MANIFEST_PATH,
    NATIVE_MODEL_MANIFEST_SHA256,
    NATIVE_MODEL_PATH,
    NativeInternNavDependencyError,
    inject_native_internnav_dependency_from_env,
)


def _config() -> dict:
    return {
        "model": {
            "llm": {"model_path": NATIVE_MODEL_PATH},
            "action_head": {
                "nextdit": {"internnav_model_path": NATIVE_MODEL_PATH}
            },
        },
        "runtime": {},
    }


def _environment() -> dict[str, str]:
    return {
        ENV_SCHEMA: NATIVE_DEPENDENCY_SCHEMA,
        ENV_MODEL_PATH: NATIVE_MODEL_PATH,
        ENV_MANIFEST_PATH: NATIVE_MODEL_MANIFEST_PATH,
        ENV_MANIFEST_SHA256: NATIVE_MODEL_MANIFEST_SHA256,
        ENV_FILE_COUNT: str(NATIVE_MODEL_FILE_COUNT),
        ENV_VERIFIED: "1",
    }


def test_verified_launcher_contract_is_injected_into_runtime() -> None:
    cfg = _config()
    contract = inject_native_internnav_dependency_from_env(
        cfg, environ=_environment()
    )
    assert cfg["runtime"]["native_internnav_dependency"] == contract
    assert contract == {
        "schema": NATIVE_DEPENDENCY_SCHEMA,
        "model_path": NATIVE_MODEL_PATH,
        "manifest_path": NATIVE_MODEL_MANIFEST_PATH,
        "manifest_sha256": NATIVE_MODEL_MANIFEST_SHA256,
        "file_count": 14,
        "verified": True,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (ENV_MANIFEST_SHA256, "a" * 64),
        (ENV_FILE_COUNT, "13"),
        (ENV_VERIFIED, "true"),
        (ENV_MODEL_PATH, "/models/not-released-internnav"),
    ],
)
def test_injection_rejects_unlocked_environment_contract(
    key: str,
    value: str,
) -> None:
    env = _environment()
    env[key] = value
    with pytest.raises(
        NativeInternNavDependencyError,
        match="locked released|must be exactly",
    ):
        inject_native_internnav_dependency_from_env(_config(), environ=env)


def test_injection_rejects_missing_contract_and_config_path_drift() -> None:
    with pytest.raises(NativeInternNavDependencyError, match="missing"):
        inject_native_internnav_dependency_from_env(_config(), environ={})

    cfg = _config()
    cfg["model"]["action_head"]["nextdit"]["internnav_model_path"] = (
        "/models/drifted-system1"
    )
    with pytest.raises(NativeInternNavDependencyError, match="do not share"):
        inject_native_internnav_dependency_from_env(cfg, environ=_environment())


def test_injection_does_not_overwrite_tampered_runtime_closure() -> None:
    cfg = _config()
    inject_native_internnav_dependency_from_env(cfg, environ=_environment())
    tampered = copy.deepcopy(cfg)
    tampered["runtime"]["native_internnav_dependency"]["verified"] = False
    with pytest.raises(NativeInternNavDependencyError, match="released InternNav"):
        inject_native_internnav_dependency_from_env(
            tampered, environ=_environment()
        )


def test_checkpoint_manager_persists_native_dependency_closure(
    tmp_path: Path,
) -> None:
    class _Stateful:
        @staticmethod
        def state_dict() -> dict:
            return {}

    cfg = _config()
    expected = inject_native_internnav_dependency_from_env(
        cfg, environ=_environment()
    )
    manager = CheckpointManager(str(tmp_path / "checkpoints"), max_ckpts=1)
    path = manager.save(
        nn.Linear(2, 2),
        _Stateful(),
        _Stateful(),
        epoch=1,
        stage_idx=0,
        stage_name="native-closure-test",
        metrics={"val_loss": 1.0},
        cfg=cfg,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["config"]["runtime"][
        "native_internnav_dependency"
    ] == expected
