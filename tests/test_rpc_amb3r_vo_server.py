from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scripts.amb3r_vo.rpc_amb3r_vo_server import (
    AMB3R_VO_RPC_PROTOCOL_VERSION,
    AMB3RVORPCApplication,
)


@dataclass
class MockBlob:
    name: str = "rgb_front"
    data: bytes = b"mock-jpeg"
    mime_type: str = "image/jpeg"
    height: int = 3
    width: int = 5


class MockQuery:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_payload(self) -> dict:
        return dict(self.payload)


class MockSession:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset(self, session_id: str, *, max_frames: int) -> dict:
        self.calls.append(("reset", session_id, max_frames))
        return {
            "schema": "mock-reset-v1",
            "session_id": session_id,
            "max_frames": max_frames,
        }

    def ingest(
        self,
        session_id: str,
        *,
        frame_id: int,
        frame_rgb: np.ndarray,
        capture_step: int,
    ) -> dict:
        self.calls.append(
            (
                "ingest",
                session_id,
                frame_id,
                capture_step,
                frame_rgb.copy(),
            )
        )
        return {
            "schema": "mock-ingest-v1",
            "session_id": session_id,
            "frame_id": frame_id,
        }

    def query(
        self,
        session_id: str,
        *,
        current_frame_id: int,
        history_frame_ids: list[int],
        translation_scale: float,
    ) -> MockQuery:
        self.calls.append(
            (
                "query",
                session_id,
                current_frame_id,
                tuple(history_frame_ids),
                translation_scale,
            )
        )
        return MockQuery(
            {
                "schema": "heatmapvln-amb3r-online-query-v1",
                "session_id": session_id,
                "current_frame_id": current_frame_id,
                "history_frame_ids": list(history_frame_ids),
                "history_rel_poses": [[1.0, 2.0, 0.0, 1.0]],
                "ready": True,
                "provider_phase": "mock_backend",
                "trajectory_revision": 2,
                "pose_provider": "amb3r_vo_da3",
            }
        )


def _decoder(data: bytes) -> np.ndarray:
    assert data == b"mock-jpeg"
    return np.arange(45, dtype=np.uint8).reshape(3, 5, 3)


def _application(
    session: MockSession | None = None,
    *,
    translation_scale: float = 1.75,
) -> tuple[AMB3RVORPCApplication, MockSession]:
    mock = session or MockSession()
    return (
        AMB3RVORPCApplication(
            mock,
            jpeg_decoder=_decoder,
            translation_scale=translation_scale,
            max_frames_limit=100,
        ),
        mock,
    )


def test_mock_full_rpc_sequence_decodes_jpeg_and_returns_relative_poses() -> None:
    application, session = _application()

    reset = application.dispatch(
        "reset_episode",
        {"session_id": "scene/episode-7", "max_frames": 40},
        [],
    )
    ingest = application.dispatch(
        "ingest_frame",
        {
            "session_id": "scene/episode-7",
            "frame_id": 0,
            "capture_step": 0,
        },
        [MockBlob()],
    )
    query = application.dispatch(
        "query_relative_poses",
        {
            "session_id": "scene/episode-7",
            "current_frame_id": 1,
            "history_frame_ids": [0],
        },
        [],
    )

    assert reset["ok"] is True
    assert ingest["proto_v"] == AMB3R_VO_RPC_PROTOCOL_VERSION
    assert query["history_rel_poses"] == [[1.0, 2.0, 0.0, 1.0]]
    assert query["pose_provider"] == "amb3r_vo_da3"
    assert application.requests_processed == 3
    assert session.calls[0] == ("reset", "scene/episode-7", 40)
    assert session.calls[1][:4] == ("ingest", "scene/episode-7", 0, 0)
    np.testing.assert_array_equal(session.calls[1][4], _decoder(b"mock-jpeg"))
    assert session.calls[2] == (
        "query",
        "scene/episode-7",
        1,
        (0,),
        1.75,
    )


@pytest.mark.parametrize(
    ("blob", "match"),
    [
        (MockBlob(name="front"), "named 'rgb_front'"),
        (MockBlob(mime_type="image/png"), "mime_type='image/jpeg'"),
        (MockBlob(data=b""), "non-empty bytes"),
        (MockBlob(height=4), "height does not match"),
        (MockBlob(width=4), "width does not match"),
    ],
)
def test_ingest_rejects_malformed_jpeg_blob(blob: MockBlob, match: str) -> None:
    application, _ = _application()
    with pytest.raises(ValueError, match=match):
        application.dispatch(
            "ingest_frame",
            {"session_id": "ep", "frame_id": 0, "capture_step": 0},
            [blob],
        )
    assert application.requests_processed == 0


def test_protocol_rejects_gt_and_client_side_scale_inputs() -> None:
    application, session = _application()
    with pytest.raises(ValueError, match="unsupported fields: gt_pose"):
        application.dispatch(
            "ingest_frame",
            {
                "session_id": "ep",
                "frame_id": 0,
                "capture_step": 0,
                "gt_pose": [1, 0, 0, 0],
            },
            [MockBlob()],
        )
    with pytest.raises(ValueError, match="unsupported fields: translation_scale"):
        application.dispatch(
            "query_relative_poses",
            {
                "session_id": "ep",
                "current_frame_id": 1,
                "history_frame_ids": [0],
                "translation_scale": 9.0,
            },
            [],
        )
    assert session.calls == []


def test_protocol_is_strict_about_methods_types_blobs_and_limits() -> None:
    application, _ = _application()
    with pytest.raises(ValueError, match="Unsupported method"):
        application.dispatch("infer", {}, [])
    with pytest.raises(ValueError, match="max_frames must be <= 100"):
        application.dispatch(
            "reset_episode",
            {"session_id": "ep", "max_frames": 101},
            [],
        )
    with pytest.raises(ValueError, match="frame_id must be an integer"):
        application.dispatch(
            "ingest_frame",
            {"session_id": "ep", "frame_id": 0.0, "capture_step": 0},
            [MockBlob()],
        )
    with pytest.raises(ValueError, match=r"history_frame_ids\[0\]"):
        application.dispatch(
            "query_relative_poses",
            {
                "session_id": "ep",
                "current_frame_id": 1,
                "history_frame_ids": [0.0],
            },
            [],
        )
    with pytest.raises(ValueError, match="does not accept binary blobs"):
        application.dispatch(
            "query_relative_poses",
            {
                "session_id": "ep",
                "current_frame_id": 1,
                "history_frame_ids": [0],
            },
            [MockBlob()],
        )
