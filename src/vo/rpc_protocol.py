"""Pure protocol helpers for the independent online AMB3R-VO service.

The Habitat process owns image capture and frame IDs, while a separate GPU
process owns mutable AMB3R state.  Keeping this contract independent of gRPC,
Habitat, Torch, and AMB3R makes the causality-critical bookkeeping cheap to
test.  No checkpoint hash or file lock is part of this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


AMB3R_VO_RPC_PROTOCOL_VERSION = "heatmapvln-amb3r-vo-json-v1"
AMB3R_VO_RESET_METHOD = "reset_episode"
AMB3R_VO_INGEST_METHOD = "ingest_frame"
AMB3R_VO_QUERY_METHOD = "query_relative_poses"
AMB3R_VO_FRONT_BLOB_NAME = "rgb_front"
AMB3R_VO_POSE_PROVIDER = "amb3r_vo_da3"
HABITAT_GT_POSE_PROVIDER = "habitat_gt_c2w"


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass
class VOFrameLedger:
    """Assign one contiguous VO frame ID to each unique capture step.

    Replanning can capture the same simulator state more than once.  The
    ledger deliberately maps those repeated captures to the existing frame
    instead of ingesting a duplicated image into VO.
    """

    session_id: str | None = None
    _step_to_frame: dict[int, int] = field(default_factory=dict)
    _last_capture_step: int | None = None

    def reset(self, session_id: str) -> None:
        identifier = str(session_id).strip()
        if not identifier:
            raise ValueError("session_id must be non-empty")
        self.session_id = identifier
        self._step_to_frame.clear()
        self._last_capture_step = None

    @property
    def frame_count(self) -> int:
        return len(self._step_to_frame)

    def register_capture_step(self, capture_step: int) -> tuple[int, bool]:
        if self.session_id is None:
            raise RuntimeError("VO frame ledger has not been reset")
        step = _integer(capture_step, name="capture_step")
        existing = self._step_to_frame.get(step)
        if existing is not None:
            return existing, False
        if self._last_capture_step is not None and step < self._last_capture_step:
            raise ValueError(
                "new VO capture steps must be monotonic: "
                f"last={self._last_capture_step}, got={step}"
            )
        frame_id = len(self._step_to_frame)
        self._step_to_frame[step] = frame_id
        self._last_capture_step = step
        return frame_id, True

    def frame_id_for_step(self, capture_step: int) -> int:
        step = _integer(capture_step, name="capture_step")
        try:
            return self._step_to_frame[step]
        except KeyError as exc:
            raise KeyError(f"capture_step {step} has not been ingested") from exc


def native_front_rgb(value: Any) -> np.ndarray:
    """Normalize a native sensor array without resizing or spatial warping."""

    frame = np.asarray(value)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            "front RGB must have shape [H,W,3/4] or [H,W], "
            f"got {frame.shape}"
        )
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError(f"front RGB must be numeric, got {frame.dtype}")
    if np.issubdtype(frame.dtype, np.floating):
        if not np.isfinite(frame).all():
            raise ValueError("front RGB contains non-finite values")
        # Habitat RGB is normally uint8.  Supporting [0,1] floats here keeps
        # the boundary strict while avoiding an accidental all-black cast.
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def unique_past_vo_records(
    records: Sequence[Mapping[str, Any]],
    *,
    current_frame_id: int,
) -> list[Mapping[str, Any]]:
    """Return one prompt record per past VO frame in chronological order."""

    current = _integer(current_frame_id, name="current_frame_id")
    selected: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    previous = -1
    for record in records:
        if "vo_frame_id" not in record:
            raise ValueError("VO-enabled history record is missing vo_frame_id")
        frame_id = _integer(record["vo_frame_id"], name="vo_frame_id")
        if frame_id < previous:
            raise ValueError("VO history records must be chronological")
        previous = frame_id
        if frame_id >= current or frame_id in seen:
            continue
        seen.add(frame_id)
        selected.append(record)
    return selected


def vo_rpc_payload(**fields: Any) -> dict[str, Any]:
    """Build a strict VO request without injecting unsupported fields.

    The protocol discriminator is returned as ``proto_v`` by the service and
    exposed through ``ServerInfo``.  Each method intentionally accepts only
    its documented semantic fields.
    """

    return dict(fields)


def _validate_common_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError("VO RPC response must be a JSON object")
    if response.get("ok") is not True:
        raise RuntimeError(f"VO RPC server error: {response!r}")
    if response.get("proto_v") != AMB3R_VO_RPC_PROTOCOL_VERSION:
        raise RuntimeError(
            "VO RPC protocol mismatch: "
            f"server={response.get('proto_v')!r}, "
            f"expected={AMB3R_VO_RPC_PROTOCOL_VERSION!r}"
        )
    return response


def validate_reset_response(response: Any, *, session_id: str) -> dict[str, Any]:
    result = _validate_common_response(response)
    if result.get("session_id") != str(session_id):
        raise RuntimeError(
            "VO reset session mismatch: "
            f"server={result.get('session_id')!r}, expected={session_id!r}"
        )
    return result


def validate_ingest_response(
    response: Any,
    *,
    session_id: str,
    frame_id: int,
    capture_step: int,
) -> dict[str, Any]:
    result = _validate_common_response(response)
    expected = {
        "session_id": str(session_id),
        "frame_id": _integer(frame_id, name="frame_id"),
        "capture_step": _integer(capture_step, name="capture_step"),
    }
    actual = {key: result.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"VO ingest acknowledgement mismatch: actual={actual!r}, "
            f"expected={expected!r}"
        )
    return result


def model_pose_fields_from_query(
    response: Any,
    *,
    session_id: str,
    current_frame_id: int,
    history_frame_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate a VO query and build fields for ``plan_panoramic``.

    ``history_rel_poses`` is intentionally absent while ``pose_ready`` is
    false.  The model RPC can then bypass heatmap pose control explicitly;
    zero-filled poses must never masquerade as a valid stationary history.
    """

    result = _validate_common_response(response)
    current = _integer(current_frame_id, name="current_frame_id")
    history = [
        _integer(value, name="history_frame_id") for value in history_frame_ids
    ]
    if any(frame_id > current for frame_id in history):
        raise ValueError(
            "VO history frame IDs must be no later than current_frame_id"
        )
    if any(right < left for left, right in zip(history, history[1:])):
        raise ValueError(
            "VO history frame IDs must be chronological (non-decreasing)"
        )
    expected_ids = {
        "session_id": str(session_id),
        "current_frame_id": current,
        "history_frame_ids": history,
    }
    actual_ids = {key: result.get(key) for key in expected_ids}
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"VO query identity mismatch: actual={actual_ids!r}, "
            f"expected={expected_ids!r}"
        )
    if result.get("pose_provider") != AMB3R_VO_POSE_PROVIDER:
        raise RuntimeError(
            f"Unexpected VO pose provider: {result.get('pose_provider')!r}"
        )
    ready = result.get("ready")
    if not isinstance(ready, bool):
        raise TypeError("VO query ready must be boolean")
    phase = result.get("provider_phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError("VO query provider_phase must be a non-empty string")
    revision = _integer(
        result.get("trajectory_revision"), name="trajectory_revision"
    )

    model_fields: dict[str, Any] = {
        "pose_provider": AMB3R_VO_POSE_PROVIDER,
        "pose_ready": ready,
        "vo_current_frame_id": current,
        "vo_history_frame_ids": history,
        "vo_provider_phase": phase,
        "vo_trajectory_revision": revision,
    }
    raw_relative = result.get("history_rel_poses")
    if not ready:
        if raw_relative is not None:
            array = np.asarray(raw_relative)
            if array.size != 0:
                raise ValueError(
                    "not-ready VO query must not expose usable history_rel_poses"
                )
        return model_fields

    relative = np.asarray(raw_relative, dtype=np.float32)
    expected_shape = (len(history), 4)
    if relative.shape != expected_shape or not np.isfinite(relative).all():
        raise ValueError(
            "ready VO history_rel_poses must be finite with shape "
            f"{expected_shape}, got {relative.shape}"
        )
    model_fields["history_rel_poses"] = relative.tolist()
    return model_fields


__all__ = [
    "AMB3R_VO_FRONT_BLOB_NAME",
    "AMB3R_VO_INGEST_METHOD",
    "AMB3R_VO_POSE_PROVIDER",
    "AMB3R_VO_QUERY_METHOD",
    "AMB3R_VO_RESET_METHOD",
    "AMB3R_VO_RPC_PROTOCOL_VERSION",
    "HABITAT_GT_POSE_PROVIDER",
    "VOFrameLedger",
    "model_pose_fields_from_query",
    "native_front_rgb",
    "unique_past_vo_records",
    "validate_ingest_response",
    "validate_reset_response",
    "vo_rpc_payload",
]
