"""Scripted, GPU-free stand-ins for the model server and the AMB3R VO server.

They speak the same ``infer_json`` / ``health_check`` / ``get_server_info`` interface as a
vla_rpc ``VLAClient``, check every request the way the real servers do, and answer from a
script.  Uses: robot plumbing dry runs (``NavAgent(FakeModelServer(), FakeVOServer(),
jpeg_encoder=pil_jpeg_encoder)`` where vla_rpc is not installed), the NavAgent tests, and the
request-level comparison with the Habitat client (scripts/deploy/nav_agent_habitat_check.py
--fake-servers).  They never produce a real action.

The VO stand-in runs the real request dispatcher (AMB3RVORPCApplication) around a session that
follows OnlineAMB3RSession's bookkeeping (contiguous frames, map at 20 frames, flush every 8
or on query, revision counter, phases) with made-up poses.
"""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image

from scripts.amb3r_vo.rpc_amb3r_vo_server import AMB3RVORPCApplication
from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_FIELD,
    validate_rpc_sampling_metadata,
)
from src.deploy.nav_agent import (
    AMB3R_VO_MODEL_VERSION,
    AMB3R_VO_POSE_PROVIDER,
    AMB3R_VO_RPC_PROTOCOL_VERSION,
    LOOKDOWN_IMAGE_SIZE,
    PLAN_METHOD,
    PPA_ONLINE_AMB3R_CAPABILITY,
    VIEWS,
    VLM_IMAGE_SIZE,
    _vo_protocol,
)

Recorder = Callable[[dict[str, Any]], None]


def decode_jpeg(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def pil_jpeg_encoder(rgb: np.ndarray, quality: int = 85) -> bytes:
    """JPEG encoder for machines without vla_rpc/OpenCV.  Not byte-compatible with vla_rpc's."""
    out = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out, format="JPEG", quality=int(quality))
    return out.getvalue()


def request_record(target: str, method: str, payload: dict[str, Any], blobs: Sequence[Any]) -> dict[str, Any]:
    """A request as it goes on the wire: the JSON text vla_rpc sends and each blob's digest."""

    def field(blob, name, default=None):
        return blob.get(name, default) if isinstance(blob, dict) else getattr(blob, name, default)

    return {
        "target": target,
        "method": method,
        "payload": json.dumps(payload, ensure_ascii=False),
        "blobs": [
            {
                "name": str(field(blob, "name", "")),
                "mime_type": str(field(blob, "mime_type", "")),
                "height": int(field(blob, "height", 0) or 0),
                "width": int(field(blob, "width", 0) or 0),
                "bytes": len(field(blob, "data", b"")),
                "sha256": hashlib.sha256(field(blob, "data", b"")).hexdigest(),
            }
            for blob in blobs
        ],
    }


class _FakeServer:
    target = ""

    def __init__(self, recorder: Optional[Recorder] = None) -> None:
        self.recorder = recorder
        self.requests = 0

    def connect(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def infer_json(self, method: str, payload: dict[str, Any], blobs: Optional[list] = None):
        blobs = list(blobs or [])
        if self.recorder is not None:
            self.recorder(request_record(self.target, method, payload, blobs))
        self.requests += 1
        # The real transport is JSON text both ways.
        request = json.loads(json.dumps(payload, ensure_ascii=False))
        response = self._answer(method, request, blobs)
        return json.loads(json.dumps(response, ensure_ascii=False)), []

    def _answer(self, method: str, payload: dict[str, Any], blobs: list) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------- VO
class _FakeVOSession:
    """OnlineAMB3RSession's observable bookkeeping (src/vo/online_amb3r.py) without AMB3R."""

    def __init__(self, *, map_init_window: int = 20, map_every: int = 8, max_history: int = 8) -> None:
        self.map_init_window = int(map_init_window)
        self.map_every = int(map_every)
        self.max_history = int(max_history)
        self._session_id: Optional[str] = None

    def reset(self, session_id: str, *, max_frames: int) -> dict[str, Any]:
        self._session_id = session_id
        self._max_frames = int(max_frames)
        self._steps: list[int] = []
        self._shape: Optional[tuple] = None
        self._last_mapped: Optional[int] = None
        self._revision = 0
        return {
            "schema": "heatmapvln-amb3r-online-reset-v1",
            "session_id": session_id,
            "max_frames": self._max_frames,
            "map_init_window": self.map_init_window,
            "map_every": self.map_every,
        }

    @property
    def trajectory_revision(self) -> int:
        return self._revision

    def _require(self, session_id: str) -> str:
        if self._session_id is None or session_id != self._session_id:
            raise ValueError(f"stale or unknown session {session_id!r} (active {self._session_id!r})")
        return session_id

    def _flush(self) -> None:
        newest = len(self._steps) - 1
        if self._last_mapped is not None and newest > self._last_mapped:
            self._last_mapped = newest
            self._revision += 1

    def ingest(self, session_id: str, *, frame_id: int, frame_rgb: np.ndarray, capture_step: int) -> dict[str, Any]:
        identifier = self._require(session_id)
        if frame_id != len(self._steps):
            raise ValueError(f"frame_id must be strictly contiguous: expected {len(self._steps)}, got {frame_id}")
        if len(self._steps) >= self._max_frames:
            raise RuntimeError(f"AMB3R session exceeded max_frames={self._max_frames}")
        if self._steps and capture_step < self._steps[-1]:
            raise ValueError("capture_step must be monotonic non-decreasing")
        if self._shape is not None and frame_rgb.shape != self._shape:
            raise ValueError(f"frame shape changed from {self._shape} to {frame_rgb.shape}")
        self._shape = frame_rgb.shape
        self._steps.append(int(capture_step))
        if len(self._steps) == self.map_init_window:
            self._last_mapped = len(self._steps) - 1
            self._revision += 1
        elif self._last_mapped is not None and len(self._steps) - 1 - self._last_mapped >= self.map_every:
            self._flush()
        return {
            "schema": "heatmapvln-amb3r-online-ingest-v1",
            "session_id": identifier,
            "frame_id": int(frame_id),
            "capture_step": int(capture_step),
            "frame_count": len(self._steps),
            "map_initialized": self._last_mapped is not None,
            "last_mapped_frame_id": self._last_mapped,
            "trajectory_revision": self._revision,
        }

    def query(self, session_id: str, *, current_frame_id: int, history_frame_ids, translation_scale: float = 1.0):
        identifier = self._require(session_id)
        latest = len(self._steps) - 1
        if current_frame_id != latest:
            raise ValueError(f"queries must target the latest frame: latest={latest}, requested={current_frame_id}")
        history = [int(value) for value in history_frame_ids]
        if any(not 0 <= h <= current_frame_id for h in history) or history != sorted(history):
            raise ValueError(f"bad history_frame_ids {history}")
        if len(history) > self.max_history:
            raise ValueError(f"at most {self.max_history} history frames")
        ready = bool(history) and len(self._steps) >= self.map_init_window
        if ready:
            self._flush()
            phase = "stateful_backend"
            # Made up, but shaped like the real thing: [forward_m, left_m, cos(yaw), sin(yaw)].
            relative = [[-0.25 * (current_frame_id - h), 0.0, 1.0, 0.0] for h in history]
        else:
            phase = "insufficient_history" if not history else "map_warmup"
            relative = []
        payload = {
            "schema": "heatmapvln-amb3r-online-query-v1",
            "session_id": identifier,
            "current_frame_id": int(current_frame_id),
            "history_frame_ids": history,
            "history_rel_poses": relative,
            "ready": ready,
            "provider_phase": phase,
            "frame_count": len(self._steps),
            "trajectory_revision": self._revision,
            "last_mapped_frame_id": self._last_mapped,
            "pose_provider": AMB3R_VO_POSE_PROVIDER,
        }
        return SimpleNamespace(to_payload=lambda: payload)


class FakeVOServer(_FakeServer):
    target = "vo"

    def __init__(self, recorder: Optional[Recorder] = None, *, map_init_window: int = 20, timing: bool = False) -> None:
        super().__init__(recorder)
        self.session = _FakeVOSession(map_init_window=map_init_window)
        self.application = AMB3RVORPCApplication(self.session, jpeg_decoder=decode_jpeg)
        self.timing = timing

    def get_server_info(self) -> Any:
        return SimpleNamespace(
            version=AMB3R_VO_RPC_PROTOCOL_VERSION,
            model_version=AMB3R_VO_MODEL_VERSION,
            supported_formats=["json+jpeg"],
        )

    def _answer(self, method: str, payload: dict[str, Any], blobs: list) -> dict[str, Any]:
        response = self.application.dispatch(method, payload, blobs)
        if self.timing:
            response["timing_ms"] = {"total": 0.0}
        return response


# ---------------------------------------------------------------------------- model
def default_script(call_index: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Cycles through every chunk shape the deployed server produces, then STOPs at call 12.

    Covers: an arrow chunk ending in STOP (replan mid-chunk), a full trajectory chunk, LOOKDOWN
    at the head of a chunk, LOOKDOWN followed by a queued STOP (calls 3 and 6; the replans after
    them, calls 4 and 7, start from a level re-capture), LOOKDOWN last (call 5; call 6, the
    first pose-ready call, starts from the tilted frame), the one-action anti-deadlock chunk,
    four turns.  With NavAgent: 13 calls, 35 steps, 5 LOOKDOWNs.
    """
    if call_index >= 12:
        return {"kind": "stop", "llm_output": "STOP"}
    pattern = [
        {"kind": "native_actions", "llm_output": "←←←", "actions": [2, 2, 2, 0]},
        {"kind": "trajectory", "llm_output": "301 225", "pixel_goal": [225, 301], "actions": [1, 1, 1, 3]},
        {"kind": "native_actions", "llm_output": "↓→", "actions": [5, 3, 0, 0]},
        {"kind": "native_actions", "llm_output": "→↓", "actions": [3, 5, 0, 0]},
        {"kind": "trajectory", "llm_output": "238 293", "pixel_goal": [293, 238], "actions": [1, 1, 2, 1]},
        {"kind": "native_actions", "llm_output": "↑↑↑↓", "actions": [1, 1, 1, 5]},
        {"kind": "native_actions", "llm_output": "↓", "actions": [5, 0, 0, 0]},
        {"kind": "trajectory", "llm_output": "180 176", "pixel_goal": [176, 180], "actions": [2], "anti_deadlock": True},
        {"kind": "native_actions", "llm_output": "→→→→", "actions": [3, 3, 3, 3]},
    ]
    return dict(pattern[call_index % len(pattern)])


class FakeModelServer(_FakeServer):
    """Answers plan_panoramic like rpc_model_server.py's native two-turn path, from a script.

    ``script(call_index, payload)`` returns {"kind", "llm_output", "actions"?, "pixel_goal"?,
    "anti_deadlock"?}.  ``timing=True`` adds a ``timing_ms`` field as a timed server would.
    """

    target = "model"

    def __init__(
        self,
        recorder: Optional[Recorder] = None,
        *,
        script: Callable[[int, dict[str, Any]], dict[str, Any]] = default_script,
        timing: bool = False,
    ) -> None:
        super().__init__(recorder)
        self.script = script
        self.timing = timing

    def get_server_info(self) -> Any:
        return SimpleNamespace(
            version=HEATMAPVLN_RPC_PROTOCOL_VERSION,
            model_version="fake-ppa",
            supported_formats=["json+jpeg", PPA_ONLINE_AMB3R_CAPABILITY],
        )

    def _check_request(self, payload: dict[str, Any], blobs: list) -> dict[str, Any]:
        sampling = validate_rpc_sampling_metadata(payload.get(HEATMAPVLN_RPC_SAMPLING_FIELD), require_deterministic=True)
        count = int(payload["num_history"])
        pose = _vo_protocol.validate_model_pose_fields(payload, num_history=count)
        expected = {
            "vlm_image_size": list(VLM_IMAGE_SIZE),
            "traj_image_size": [224, 224],
            "system1_coord_order": "generated",
            "trajectory_selection": "mean",
            "trajectory_x_sign": 1.0,
            "trajectory_heading_alignment": "none",
            "require_deterministic_sampling": True,
            "phase": "joint",
        }
        wrong = {k: payload.get(k) for k, v in expected.items() if payload.get(k) != v}
        if wrong or not str(payload.get("instruction") or ""):
            raise ValueError(f"unexpected plan request fields: {wrong}")
        names = [f"current/{v}" for v in VIEWS]
        names += [f"history/{i}/{v}" for i in range(count) for v in VIEWS]
        names.append("lookdown")
        got = [blob["name"] if isinstance(blob, dict) else blob.name for blob in blobs]
        if got != names:
            raise ValueError(f"blob names {got} != {names}")
        for blob in blobs:
            size = LOOKDOWN_IMAGE_SIZE if blob["name"] == "lookdown" else VLM_IMAGE_SIZE
            rgb = decode_jpeg(blob["data"])
            if blob["mime_type"] != "image/jpeg" or rgb.shape != (size[1], size[0], 3):
                raise ValueError(f"blob {blob['name']} is {blob['mime_type']} {rgb.shape}, expected {size}")
            if (blob["height"], blob["width"]) != (size[1], size[0]):
                raise ValueError(f"blob {blob['name']} metadata {(blob['height'], blob['width'])}")
        return {"sampling": sampling, "pose": pose}

    def _answer(self, method: str, payload: dict[str, Any], blobs: list) -> dict[str, Any]:
        if method != PLAN_METHOD:
            raise ValueError(f"Unsupported method: {method}")
        checked = self._check_request(payload, blobs)
        ready = bool(checked["pose"]["pose_ready"])
        entry = self.script(int(checked["sampling"]["system2_call_index"]), payload)
        kind = entry["kind"]
        response: dict[str, Any] = {
            "ok": True,
            "proto_v": HEATMAPVLN_RPC_PROTOCOL_VERSION,
            "llm_output": entry["llm_output"],
            "system2_source": "model",
            "oracle_system2": None,
            "actions": [],
            "terminal": False,
            "kind": "unknown",
            "phase": "joint",
            "ppa_runtime": PPA_ONLINE_AMB3R_CAPABILITY,
            "pose_provider": AMB3R_VO_POSE_PROVIDER,
            "pose_ready": ready,
            "vo_provider_phase": checked["pose"]["vo_provider_phase"],
            "vo_trajectory_revision": int(checked["pose"]["vo_trajectory_revision"]),
            "ppa_applied": False,
            "native_first_output": entry["llm_output"],
            "native_lookdown_turns": 0,
            "native_front_only": True,
            HEATMAPVLN_RPC_SAMPLING_FIELD: checked["sampling"],
        }
        if kind in ("stop", "fallback_stop"):
            response.update({"kind": kind, "terminal": True, "actions": [0]})
        elif kind == "native_actions":
            response.update({"kind": kind, "actions": list(entry["actions"])})
        elif kind == "trajectory":
            response.update(
                {
                    "pixel_goal": list(entry["pixel_goal"]),
                    "pano_goal_view": "front",
                    "kind": "trajectory",
                    "actions": list(entry["actions"]),
                    "trajectory_summary": "traj_goal=(1.00,0.00), direct=1.00, path_len=1.00",
                    "trajectory_x_sign": 1.0,
                    "trajectory_heading_alignment": "none",
                    "trajectory_target_heading_deg": None,
                    "ppa_applied": ready,
                }
            )
            if not ready:
                response["ppa_skip_reason"] = "amb3r_map_warmup"
            if entry.get("anti_deadlock"):
                response["anti_deadlock"] = True
        else:
            raise ValueError(f"script returned unknown kind {kind!r}")
        if self.timing:
            response["timing_ms"] = {"total": 0.0}
        return response


__all__ = [
    "FakeModelServer",
    "FakeVOServer",
    "decode_jpeg",
    "default_script",
    "pil_jpeg_encoder",
    "request_record",
]
