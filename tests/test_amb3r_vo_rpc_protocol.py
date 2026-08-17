from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "vo" / "rpc_protocol.py"
)
_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "_test_amb3r_vo_rpc_protocol", _PROTOCOL_PATH
)
assert _PROTOCOL_SPEC is not None and _PROTOCOL_SPEC.loader is not None
_protocol = importlib.util.module_from_spec(_PROTOCOL_SPEC)
sys.modules[_PROTOCOL_SPEC.name] = _protocol
_PROTOCOL_SPEC.loader.exec_module(_protocol)

AMB3R_VO_FRONT_BLOB_NAME = _protocol.AMB3R_VO_FRONT_BLOB_NAME
AMB3R_VO_POSE_PROVIDER = _protocol.AMB3R_VO_POSE_PROVIDER
AMB3R_VO_RPC_PROTOCOL_VERSION = _protocol.AMB3R_VO_RPC_PROTOCOL_VERSION
VOFrameLedger = _protocol.VOFrameLedger
model_pose_fields_from_query = _protocol.model_pose_fields_from_query
native_front_rgb = _protocol.native_front_rgb
unique_past_vo_records = _protocol.unique_past_vo_records


def test_client_protocol_constants_match_vo_server_contract() -> None:
    server_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "amb3r_vo"
        / "rpc_amb3r_vo_server.py"
    )
    spec = importlib.util.spec_from_file_location("_amb3r_vo_rpc_server", server_path)
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    assert AMB3R_VO_RPC_PROTOCOL_VERSION == server.AMB3R_VO_RPC_PROTOCOL_VERSION
    assert AMB3R_VO_FRONT_BLOB_NAME == server.RGB_FRONT_BLOB_NAME == "rgb_front"


def _query_response(*, ready: bool = True) -> dict:
    result = {
        "ok": True,
        "proto_v": AMB3R_VO_RPC_PROTOCOL_VERSION,
        "session_id": "scene/episode",
        "current_frame_id": 2,
        "history_frame_ids": [0, 1],
        "ready": ready,
        "provider_phase": "stateful_backend" if ready else "warmup",
        "trajectory_revision": 3 if ready else 0,
        "pose_provider": AMB3R_VO_POSE_PROVIDER,
    }
    if ready:
        result["history_rel_poses"] = [
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.1, 0.2, 0.98],
        ]
    return result


def test_frame_ledger_deduplicates_repeated_capture_steps() -> None:
    ledger = VOFrameLedger()
    ledger.reset("scene/episode")

    assert ledger.register_capture_step(0) == (0, True)
    assert ledger.register_capture_step(0) == (0, False)
    assert ledger.register_capture_step(1) == (1, True)
    assert ledger.frame_id_for_step(0) == 0
    assert ledger.frame_count == 2

    ledger.register_capture_step(3)
    with pytest.raises(ValueError, match="monotonic"):
        ledger.register_capture_step(2)


def test_native_front_rgb_keeps_native_spatial_shape() -> None:
    rgba = np.zeros((123, 257, 4), dtype=np.uint8)
    rgba[..., 0] = 17
    result = native_front_rgb(rgba)
    assert result.shape == (123, 257, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert np.all(result[..., 0] == 17)


def test_prompt_records_are_unique_and_strictly_past() -> None:
    records = [
        {"vo_frame_id": 0, "tag": "first"},
        {"vo_frame_id": 0, "tag": "duplicate-replan"},
        {"vo_frame_id": 1, "tag": "past"},
        {"vo_frame_id": 2, "tag": "same-as-current"},
    ]
    selected = unique_past_vo_records(records, current_frame_id=2)
    assert [item["tag"] for item in selected] == ["first", "past"]


def test_ready_query_becomes_strict_top_level_model_fields() -> None:
    fields = model_pose_fields_from_query(
        _query_response(),
        session_id="scene/episode",
        current_frame_id=2,
        history_frame_ids=[0, 1],
    )
    relative = fields.pop("history_rel_poses")
    assert fields == {
        "pose_provider": AMB3R_VO_POSE_PROVIDER,
        "pose_ready": True,
        "vo_current_frame_id": 2,
        "vo_history_frame_ids": [0, 1],
        "vo_provider_phase": "stateful_backend",
        "vo_trajectory_revision": 3,
    }
    np.testing.assert_allclose(
        relative,
        [[1.0, 0.0, 0.0, 1.0], [0.5, 0.1, 0.2, 0.98]],
        rtol=0.0,
        atol=1e-6,
    )


def test_not_ready_query_does_not_forge_zero_pose_tokens() -> None:
    fields = model_pose_fields_from_query(
        _query_response(ready=False),
        session_id="scene/episode",
        current_frame_id=2,
        history_frame_ids=[0, 1],
    )
    assert fields["pose_ready"] is False
    assert "history_rel_poses" not in fields

    malformed = _query_response(ready=False)
    malformed["history_rel_poses"] = [[0.0, 0.0, 0.0, 0.0]] * 2
    with pytest.raises(ValueError, match="not-ready"):
        model_pose_fields_from_query(
            malformed,
            session_id="scene/episode",
            current_frame_id=2,
            history_frame_ids=[0, 1],
        )


def test_query_response_must_echo_exact_selected_frame_ids() -> None:
    response = _query_response()
    response["history_frame_ids"] = [1, 0]
    with pytest.raises(RuntimeError, match="identity mismatch"):
        model_pose_fields_from_query(
            response,
            session_id="scene/episode",
            current_frame_id=2,
            history_frame_ids=[0, 1],
        )


def test_query_contract_preserves_duplicate_and_current_prompt_slots() -> None:
    response = _query_response()
    response["history_frame_ids"] = [1, 2, 2]
    response["history_rel_poses"] = [
        [0.5, 0.1, 0.2, 0.98],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    fields = model_pose_fields_from_query(
        response,
        session_id="scene/episode",
        current_frame_id=2,
        history_frame_ids=[1, 2, 2],
    )
    assert fields["vo_history_frame_ids"] == [1, 2, 2]
    assert len(fields["history_rel_poses"]) == 3

    with pytest.raises(ValueError, match="non-decreasing"):
        model_pose_fields_from_query(
            response,
            session_id="scene/episode",
            current_frame_id=2,
            history_frame_ids=[2, 1],
        )
