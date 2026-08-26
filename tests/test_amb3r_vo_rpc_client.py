from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VO_PACKAGE = types.ModuleType("src.vo")
VO_PACKAGE.__path__ = [str(ROOT / "src" / "vo")]
sys.modules.setdefault("src.vo", VO_PACKAGE)

for name in ("rpc_protocol", "rpc_client"):
    path = ROOT / "src" / "vo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"src.vo.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

protocol = sys.modules["src.vo.rpc_protocol"]
OnlineVORPCBridge = sys.modules["src.vo.rpc_client"].OnlineVORPCBridge
sample_unique_past_indices = sys.modules[
    "src.vo.rpc_client"
].sample_unique_past_indices


class _FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def infer_json(self, method, payload, blobs):
        self.calls.append((method, json.loads(json.dumps(payload)), blobs))
        common = {
            "ok": True,
            "proto_v": protocol.AMB3R_VO_RPC_PROTOCOL_VERSION,
            "session_id": payload["session_id"],
        }
        if method == protocol.AMB3R_VO_RESET_METHOD:
            return ({**common, "max_frames": payload["max_frames"]}, [])
        if method == protocol.AMB3R_VO_INGEST_METHOD:
            return (
                {
                    **common,
                    "frame_id": payload["frame_id"],
                    "capture_step": payload["capture_step"],
                },
                [],
            )
        if method == protocol.AMB3R_VO_QUERY_METHOD:
            history = payload["history_frame_ids"]
            return (
                {
                    **common,
                    "current_frame_id": payload["current_frame_id"],
                    "history_frame_ids": history,
                    "ready": True,
                    "provider_phase": "stateful_backend",
                    "trajectory_revision": 1,
                    "pose_provider": protocol.AMB3R_VO_POSE_PROVIDER,
                    "history_rel_poses": [[0.0, 0.0, 1.0, 0.0] for _ in history],
                },
                [],
            )
        raise AssertionError(method)


def test_bridge_deduplicates_capture_step_and_queries_exact_ids() -> None:
    client = _FakeClient()
    bridge = OnlineVORPCBridge(
        client,
        jpeg_encoder=lambda _rgb, quality: f"jpeg-{quality}".encode(),
    )
    bridge.reset_episode("scene/1", max_frames=32)
    rgb = np.zeros((8, 10, 4), dtype=np.uint8)
    assert bridge.ingest_rgb(rgb, capture_step=0) == 0
    assert bridge.ingest_rgb(rgb, capture_step=0) == 0
    assert bridge.ingest_rgb(rgb, capture_step=1) == 1
    ingest_calls = [call for call in client.calls if call[0] == protocol.AMB3R_VO_INGEST_METHOD]
    assert len(ingest_calls) == 2
    assert ingest_calls[0][2][0]["height"] == 8
    assert ingest_calls[0][2][0]["width"] == 10

    fields = bridge.query_model_pose_fields(
        current_frame_id=1,
        history_frame_ids=[0],
    )
    assert fields["pose_ready"] is True
    assert fields["vo_history_frame_ids"] == [0]
    assert fields["history_rel_poses"] == [[0.0, 0.0, 1.0, 0.0]]


def test_history_sampler_removes_replans_and_current_then_spans_route() -> None:
    indices = sample_unique_past_indices(
        [0, 0, 1, 2, 3, 3, 4, 5],
        current_frame_id=5,
        max_history=3,
    )
    assert indices == [0, 3, 6]
