"""Client-side episode state for the independent online AMB3R RPC service."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .rpc_protocol import (
    AMB3R_VO_FRONT_BLOB_NAME,
    AMB3R_VO_INGEST_METHOD,
    AMB3R_VO_QUERY_METHOD,
    AMB3R_VO_RESET_METHOD,
    VOFrameLedger,
    model_pose_fields_from_query,
    native_front_rgb,
    validate_ingest_response,
    validate_reset_response,
    vo_rpc_payload,
)


def sample_unique_past_indices(
    frame_ids: list[int],
    *,
    current_frame_id: int,
    max_history: int,
) -> list[int]:
    """Select endpoint-like linspace history from unique strictly past frames."""
    current = int(current_frame_id)
    limit = int(max_history)
    if limit < 0:
        raise ValueError("max_history must be non-negative")
    eligible: list[int] = []
    seen: set[int] = set()
    previous = -1
    for index, raw in enumerate(frame_ids):
        frame_id = int(raw)
        if frame_id < previous:
            raise ValueError("VO frame IDs must be chronological")
        previous = frame_id
        if frame_id >= current or frame_id in seen:
            continue
        seen.add(frame_id)
        eligible.append(index)
    if limit == 0 or not eligible:
        return []
    if len(eligible) <= limit:
        return eligible
    positions = np.linspace(0, len(eligible) - 1, limit, dtype=np.int64)
    return [eligible[int(position)] for position in positions]


class OnlineVORPCBridge:
    """Map Habitat capture steps to one causal AMB3R session."""

    def __init__(
        self,
        client: Any,
        *,
        jpeg_quality: int = 95,
        jpeg_encoder: Callable[..., bytes] | None = None,
    ) -> None:
        self.client = client
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("AMB3R VO JPEG quality must be in [1,100]")
        self.jpeg_encoder = jpeg_encoder
        self.ledger = VOFrameLedger()

    @property
    def session_id(self) -> str:
        if self.ledger.session_id is None:
            raise RuntimeError("AMB3R VO bridge has not reset an episode")
        return self.ledger.session_id

    def _infer(
        self,
        method: str,
        payload: dict[str, Any],
        blobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = self.client.infer_json(method, payload, blobs)
        if result is None:
            raise RuntimeError(f"AMB3R VO RPC returned no response for {method}")
        response, response_blobs = result
        if response_blobs:
            raise RuntimeError(
                f"AMB3R VO RPC {method} unexpectedly returned binary blobs"
            )
        if not isinstance(response, dict):
            raise TypeError(f"AMB3R VO RPC {method} response must be a JSON object")
        return response

    def reset_episode(self, session_id: str, *, max_frames: int) -> None:
        response = self._infer(
            AMB3R_VO_RESET_METHOD,
            vo_rpc_payload(session_id=str(session_id), max_frames=int(max_frames)),
            [],
        )
        validate_reset_response(response, session_id=str(session_id))
        self.ledger.reset(str(session_id))

    def ingest_rgb(self, frame_rgb: Any, *, capture_step: int) -> int:
        frame_id, is_new = self.ledger.register_capture_step(int(capture_step))
        if not is_new:
            return frame_id
        rgb = native_front_rgb(frame_rgb)
        encoder = self.jpeg_encoder
        if encoder is None:
            from vla_rpc.core.image import encode_rgb_to_jpeg

            encoder = encode_rgb_to_jpeg
        blob = {
            "name": AMB3R_VO_FRONT_BLOB_NAME,
            "data": encoder(rgb, quality=self.jpeg_quality),
            "mime_type": "image/jpeg",
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
        }
        response = self._infer(
            AMB3R_VO_INGEST_METHOD,
            vo_rpc_payload(
                session_id=self.session_id,
                frame_id=frame_id,
                capture_step=int(capture_step),
            ),
            [blob],
        )
        validate_ingest_response(
            response,
            session_id=self.session_id,
            frame_id=frame_id,
            capture_step=int(capture_step),
        )
        return frame_id

    def query_model_pose_fields(
        self,
        *,
        current_frame_id: int,
        history_frame_ids: list[int],
    ) -> dict[str, Any]:
        current = int(current_frame_id)
        history = [int(value) for value in history_frame_ids]
        response = self._infer(
            AMB3R_VO_QUERY_METHOD,
            vo_rpc_payload(
                session_id=self.session_id,
                current_frame_id=current,
                history_frame_ids=history,
            ),
            [],
        )
        return model_pose_fields_from_query(
            response,
            session_id=self.session_id,
            current_frame_id=current,
            history_frame_ids=history,
        )


__all__ = ["OnlineVORPCBridge", "sample_unique_past_indices"]
