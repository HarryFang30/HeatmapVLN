"""Pure contract tests for the candidate RPC pose-provider boundary.

The RPC entrypoint imports the full InternNav/Qwen serving stack.  These tests
extract only its pure metadata functions so contract checks stay CPU-only and
do not construct either model.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "rpc_candidate_support_server.py"
)
PURE_FUNCTIONS = {
    "_validate_capture_metadata",
    "_validate_gt_pose_metadata",
    "_strict_nonnegative_int",
    "_validate_amb3r_pose_metadata",
    "validate_history_metadata",
    "_heatmap_control_input_ready",
}
PURE_CONSTANTS = {
    "CONTROL_PROTO_VERSION",
    "GT_POSE_PROVIDER",
    "AMB3R_POSE_PROVIDER",
}


def _load_contract_namespace() -> dict[str, Any]:
    parsed = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))
    selected: list[ast.stmt] = []
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if names & PURE_CONSTANTS:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in PURE_FUNCTIONS:
                selected.append(node)
    def compute_history_rel_poses(
        history_poses: list[np.ndarray],
        current_pose: np.ndarray,
        *,
        camera_forward_axis: str,
    ) -> np.ndarray:
        del current_pose
        assert camera_forward_axis == "-z"
        return np.tile(
            np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            (len(history_poses), 1),
        )

    namespace = {
        "Any": Any,
        "np": np,
        "compute_history_rel_poses": compute_history_rel_poses,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
    assert PURE_FUNCTIONS <= namespace.keys()
    return namespace


CONTRACT = _load_contract_namespace()
validate_history_metadata = CONTRACT["validate_history_metadata"]
heatmap_control_input_ready = CONTRACT["_heatmap_control_input_ready"]


def _capture_fields(num_history: int = 2) -> dict[str, Any]:
    if num_history == 0:
        history_steps: list[int] = []
    else:
        history_steps = list(range(2, 2 + num_history))
    current = 8
    return {
        "control_proto_v": "heatmap-control-eval-v1",
        "current_capture_step": current,
        "history_capture_steps": history_steps,
        "history_age_steps": [current - step for step in history_steps],
    }


def _gt_payload(*, explicit: bool) -> dict[str, Any]:
    current = np.eye(4, dtype=np.float32)
    current[0, 3] = 1.0
    history = np.eye(4, dtype=np.float32)
    payload = {
        **_capture_fields(1),
        "current_c2w": current.tolist(),
        "history_c2w": [history.tolist()],
    }
    if explicit:
        payload["pose_provider"] = "habitat_gt_c2w"
    return payload


def _vo_payload(*, ready: bool = True) -> dict[str, Any]:
    payload = {
        **_capture_fields(2),
        "pose_provider": "amb3r_vo_da3",
        "pose_ready": ready,
        "vo_current_frame_id": 12,
        "vo_history_frame_ids": [3, 9],
        "vo_provider_phase": "stateful_backend" if ready else "direct_warmup",
        "vo_trajectory_revision": 4,
    }
    if ready:
        payload["history_rel_poses"] = [
            [1.25, -0.5, 1.0, 0.0],
            [0.1, 0.2, 0.0, 1.0],
        ]
    return payload


def test_legacy_and_explicit_gt_paths_are_geometry_identical() -> None:
    legacy = validate_history_metadata(_gt_payload(explicit=False), 1)
    explicit = validate_history_metadata(_gt_payload(explicit=True), 1)

    assert legacy["pose_provider"] == "habitat_gt_c2w"
    assert legacy["pose_provider_explicit"] is False
    assert explicit["pose_provider_explicit"] is True
    assert legacy["pose_ready"] is True
    np.testing.assert_array_equal(
        legacy["history_rel_poses"], explicit["history_rel_poses"]
    )
    np.testing.assert_array_equal(legacy["current_c2w"], explicit["current_c2w"])
    np.testing.assert_array_equal(legacy["history_c2w"], explicit["history_c2w"])


def test_ready_amb3r_accepts_only_external_relative_pose_contract() -> None:
    metadata = validate_history_metadata(_vo_payload(ready=True), 2)

    assert metadata["pose_provider"] == "amb3r_vo_da3"
    assert metadata["pose_ready"] is True
    assert metadata["history_rel_poses"].shape == (2, 4)
    assert "current_c2w" not in metadata
    assert "history_c2w" not in metadata
    assert heatmap_control_input_ready("on", 2, metadata) is True


def test_amb3r_c2w_leak_and_nonunit_yaw_fail_closed() -> None:
    leaked = _vo_payload(ready=True)
    leaked["current_c2w"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="must not contain privileged c2w"):
        validate_history_metadata(leaked, 2)

    malformed = _vo_payload(ready=True)
    malformed["history_rel_poses"][0][2:] = [0.2, 0.2]
    with pytest.raises(ValueError, match="unit \\(cos,sin\\)"):
        validate_history_metadata(malformed, 2)


def test_amb3r_warmup_skips_control_without_blocking_native_path() -> None:
    metadata = validate_history_metadata(_vo_payload(ready=False), 2)

    assert metadata["pose_ready"] is False
    assert metadata["history_rel_poses"].shape == (0, 4)
    assert heatmap_control_input_ready("on", 2, metadata) is False

    ambiguous = _vo_payload(ready=False)
    ambiguous["history_rel_poses"] = [[0.0, 0.0, 1.0, 0.0]] * 2
    with pytest.raises(ValueError, match="must omit history_rel_poses"):
        validate_history_metadata(ambiguous, 2)


def test_amb3r_frame_alignment_is_strict() -> None:
    payload = _vo_payload(ready=True)
    payload["vo_history_frame_ids"] = [9, 3]
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_history_metadata(payload, 2)

    repeated_current = _vo_payload(ready=True)
    repeated_current["vo_history_frame_ids"] = [12, 12]
    repeated_current["history_rel_poses"] = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    metadata = validate_history_metadata(repeated_current, 2)
    assert metadata["vo_history_frame_ids"].tolist() == [12, 12]

    future = _vo_payload(ready=True)
    future["vo_history_frame_ids"] = [3, 13]
    with pytest.raises(ValueError, match="no later"):
        validate_history_metadata(future, 2)

    payload = _vo_payload(ready=True)
    payload["vo_history_frame_ids"] = [3]
    with pytest.raises(ValueError, match="does not match num_history"):
        validate_history_metadata(payload, 2)
