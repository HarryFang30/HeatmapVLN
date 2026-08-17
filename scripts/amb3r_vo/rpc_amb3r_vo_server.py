#!/usr/bin/env python3
"""Single-session AMB3R-VO RPC server for online HeatmapVLN inference.

The transport deliberately follows the existing HeatmapVLN ``vla_rpc``
JSON-plus-binary-blob convention, but uses a separate protocol version because
this process serves poses rather than navigation actions.  Its mutable state is
strictly serialized by one gRPC worker.  The public request surface contains no
Habitat/GT pose field and no per-episode scale fitting input.

Methods
-------
``reset_episode``
    JSON: ``session_id`` and ``max_frames``.  No blobs.
``ingest_frame``
    JSON: ``session_id``, contiguous ``frame_id`` and monotonic
    ``capture_step``.  Exactly one ``image/jpeg`` blob named ``rgb_front``.
``query_relative_poses``
    JSON: ``session_id``, latest ``current_frame_id`` and past-only
    ``history_frame_ids``.  No blobs.  Returns the existing ``[K,4]`` heatmap
    trajectory representation.

The request dispatcher is dependency-light and independently testable.  Torch,
AMB3R, gRPC and the generated VLA protobuf modules are imported only while
constructing or serving the real runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
from pathlib import Path
from typing import Any, Protocol

import numpy as np


LOGGER = logging.getLogger("heatmapvln-amb3r-vo-rpc-server")

AMB3R_VO_RPC_PROTOCOL_VERSION = "heatmapvln-amb3r-vo-json-v1"
AMB3R_VO_RPC_MODEL_VERSION = "amb3r-vo-da3-online"
RGB_FRONT_BLOB_NAME = "rgb_front"

_METHOD_RESET = "reset_episode"
_METHOD_INGEST = "ingest_frame"
_METHOD_QUERY = "query_relative_poses"
_SUPPORTED_METHODS = (_METHOD_RESET, _METHOD_INGEST, _METHOD_QUERY)


class OnlineAMB3RSessionLike(Protocol):
    """Structural contract used by the dependency-free dispatcher."""

    def reset(self, session_id: str, *, max_frames: int) -> dict[str, Any]: ...

    def ingest(
        self,
        session_id: str,
        *,
        frame_id: int,
        frame_rgb: np.ndarray,
        capture_step: int,
    ) -> dict[str, Any]: ...

    def query(
        self,
        session_id: str,
        *,
        current_frame_id: int,
        history_frame_ids: Sequence[int],
        translation_scale: float = 1.0,
    ) -> Any: ...


def _strict_payload(
    payload: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("JSON payload must be an object")
    allowed = required | (optional or set())
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("JSON payload is missing fields: " + ", ".join(missing))
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(
            "JSON payload has unsupported fields: " + ", ".join(unexpected)
        )
    return payload


def _session_id(payload: Mapping[str, Any]) -> str:
    value = payload["session_id"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session_id must be a non-empty string")
    return value.strip()


def _strict_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def _history_ids(payload: Mapping[str, Any]) -> list[int]:
    value = payload["history_frame_ids"]
    if not isinstance(value, list):
        raise ValueError("history_frame_ids must be a JSON list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"history_frame_ids[{index}] must be an integer"
            )
        result.append(item)
    return result


def _blob_field(blob: Any, field: str, default: Any = None) -> Any:
    if isinstance(blob, Mapping):
        return blob.get(field, default)
    return getattr(blob, field, default)


class AMB3RVORPCApplication:
    """Validated method dispatcher around one mutable online VO session."""

    def __init__(
        self,
        session: OnlineAMB3RSessionLike,
        *,
        jpeg_decoder: Callable[[bytes], np.ndarray],
        translation_scale: float = 1.0,
        max_frames_limit: int = 4096,
    ) -> None:
        if not np.isfinite(translation_scale) or float(translation_scale) <= 0.0:
            raise ValueError("translation_scale must be a finite positive scalar")
        if isinstance(max_frames_limit, bool) or int(max_frames_limit) < 1:
            raise ValueError("max_frames_limit must be a positive integer")
        self.session = session
        self.jpeg_decoder = jpeg_decoder
        self.translation_scale = float(translation_scale)
        self.max_frames_limit = int(max_frames_limit)
        self.requests_processed = 0

    @staticmethod
    def _response(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "proto_v": AMB3R_VO_RPC_PROTOCOL_VERSION,
            **dict(result),
        }

    def dispatch(
        self,
        method: str,
        payload: Any,
        blobs: Sequence[Any],
    ) -> dict[str, Any]:
        """Execute one validated request.

        Calls are serialized by the server's one-worker executor.  This method
        itself intentionally has no lock or hidden retry semantics.
        """

        if method == _METHOD_RESET:
            output = self._reset(payload, blobs)
        elif method == _METHOD_INGEST:
            output = self._ingest(payload, blobs)
        elif method == _METHOD_QUERY:
            output = self._query(payload, blobs)
        else:
            raise ValueError(
                f"Unsupported method {method!r}; expected one of "
                + ", ".join(_SUPPORTED_METHODS)
            )
        self.requests_processed += 1
        return self._response(output)

    def _reset(self, payload: Any, blobs: Sequence[Any]) -> dict[str, Any]:
        values = _strict_payload(payload, required={"session_id", "max_frames"})
        if blobs:
            raise ValueError("reset_episode does not accept binary blobs")
        identifier = _session_id(values)
        max_frames = _strict_int(
            values,
            "max_frames",
            minimum=1,
            maximum=self.max_frames_limit,
        )
        return self.session.reset(identifier, max_frames=max_frames)

    def _decode_front_jpeg(self, blobs: Sequence[Any]) -> np.ndarray:
        if len(blobs) != 1:
            raise ValueError(
                "ingest_frame requires exactly one rgb_front JPEG blob"
            )
        blob = blobs[0]
        name = _blob_field(blob, "name")
        mime_type = _blob_field(blob, "mime_type")
        data = _blob_field(blob, "data")
        if name != RGB_FRONT_BLOB_NAME:
            raise ValueError(
                f"ingest_frame blob must be named {RGB_FRONT_BLOB_NAME!r}, "
                f"got {name!r}"
            )
        if mime_type != "image/jpeg":
            raise ValueError(
                f"ingest_frame blob must have mime_type='image/jpeg', got {mime_type!r}"
            )
        if not isinstance(data, bytes) or not data:
            raise ValueError("ingest_frame JPEG blob data must be non-empty bytes")

        rgb = np.asarray(self.jpeg_decoder(data))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                "Decoded rgb_front must be uint8 [H,W,3], "
                f"got dtype={rgb.dtype} shape={rgb.shape}"
            )
        expected_height = int(_blob_field(blob, "height", 0) or 0)
        expected_width = int(_blob_field(blob, "width", 0) or 0)
        if expected_height > 0 and expected_height != int(rgb.shape[0]):
            raise ValueError(
                "Decoded JPEG height does not match blob metadata: "
                f"decoded={rgb.shape[0]} metadata={expected_height}"
            )
        if expected_width > 0 and expected_width != int(rgb.shape[1]):
            raise ValueError(
                "Decoded JPEG width does not match blob metadata: "
                f"decoded={rgb.shape[1]} metadata={expected_width}"
            )
        return np.ascontiguousarray(rgb)

    def _ingest(self, payload: Any, blobs: Sequence[Any]) -> dict[str, Any]:
        values = _strict_payload(
            payload,
            required={"session_id", "frame_id", "capture_step"},
        )
        identifier = _session_id(values)
        frame_id = _strict_int(values, "frame_id", minimum=0)
        capture_step = _strict_int(values, "capture_step", minimum=0)
        frame_rgb = self._decode_front_jpeg(blobs)
        return self.session.ingest(
            identifier,
            frame_id=frame_id,
            frame_rgb=frame_rgb,
            capture_step=capture_step,
        )

    def _query(self, payload: Any, blobs: Sequence[Any]) -> dict[str, Any]:
        values = _strict_payload(
            payload,
            required={"session_id", "current_frame_id", "history_frame_ids"},
        )
        if blobs:
            raise ValueError("query_relative_poses does not accept binary blobs")
        result = self.session.query(
            _session_id(values),
            current_frame_id=_strict_int(
                values,
                "current_frame_id",
                minimum=0,
            ),
            history_frame_ids=_history_ids(values),
            # This is one deployment-wide calibration constant supplied only
            # when starting the server.  The RPC client cannot fit or override
            # it from GT for an episode.
            translation_scale=self.translation_scale,
        )
        payload_result = result.to_payload()
        if not isinstance(payload_result, Mapping):
            raise TypeError("Online AMB3R query result must provide a JSON object")
        required_result_fields = {
            "ready",
            "history_rel_poses",
            "provider_phase",
            "trajectory_revision",
            "current_frame_id",
            "history_frame_ids",
        }
        missing = sorted(required_result_fields - set(payload_result))
        if missing:
            raise RuntimeError(
                "Online AMB3R query result is missing fields: "
                + ", ".join(missing)
            )
        return dict(payload_result)


def _build_real_application(args: argparse.Namespace) -> AMB3RVORPCApplication:
    project_root = Path(args.repo).expanduser().resolve(strict=True)
    amb3r_root = Path(args.amb3r_root).expanduser().resolve(strict=True)
    checkpoint = Path(args.da3_checkpoint).expanduser().resolve(strict=True)
    cfg_path = (
        Path(args.cfg_path).expanduser().resolve(strict=True)
        if args.cfg_path
        else (amb3r_root / "slam" / "slam_config.yaml").resolve(strict=True)
    )
    sys.path[:0] = [
        str(project_root),
        str(amb3r_root),
        str(amb3r_root / "thirdparty"),
    ]

    from amb3r.model_zoo import load_model
    from src.vo.online_amb3r import build_online_amb3r_session
    from vla_rpc.core.image import decode_jpeg_to_rgb

    model = load_model("da3", ckpt_path=str(checkpoint))
    session = build_online_amb3r_session(
        model,
        cfg_path=cfg_path,
        device=args.device,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
        max_history=args.max_history,
        resolution=tuple(args.resolution),
    )
    return AMB3RVORPCApplication(
        session,
        jpeg_decoder=decode_jpeg_to_rgb,
        translation_scale=args.translation_scale,
        max_frames_limit=args.max_frames_limit,
    )


def _serve(args: argparse.Namespace, application: AMB3RVORPCApplication) -> int:
    import grpc
    from vla_rpc.proto import vla_pb2, vla_pb2_grpc

    class AMB3RVOServicer(vla_pb2_grpc.VLAServicer):
        def InferJSON(
            self,
            request: Any,
            context: Any,
        ) -> Any:
            try:
                payload = (
                    json.loads(request.json_payload)
                    if request.json_payload
                    else {}
                )
                output = application.dispatch(
                    request.method,
                    payload,
                    request.blobs,
                )
                return vla_pb2.JSONResponse(
                    ts=request.ts,
                    json_payload=json.dumps(output, ensure_ascii=False),
                    model_v=AMB3R_VO_RPC_MODEL_VERSION,
                )
            except Exception as exc:
                LOGGER.exception("InferJSON failed")
                context.set_details(str(exc))
                context.set_code(grpc.StatusCode.INTERNAL)
                return vla_pb2.JSONResponse(
                    ts=request.ts,
                    json_payload=json.dumps(
                        {"ok": False, "error": str(exc)},
                        ensure_ascii=False,
                    ),
                    model_v=AMB3R_VO_RPC_MODEL_VERSION,
                )

        def HealthCheck(self, request: Any, context: Any) -> Any:
            return vla_pb2.HealthCheckResponse(
                status=vla_pb2.HealthCheckResponse.SERVING,
                message="AMB3R-VO online pose server is running",
                version=AMB3R_VO_RPC_PROTOCOL_VERSION,
                requests_processed=application.requests_processed,
            )

        def GetServerInfo(self, request: Any, context: Any) -> Any:
            return vla_pb2.ServerInfo(
                version=AMB3R_VO_RPC_PROTOCOL_VERSION,
                model_version=AMB3R_VO_RPC_MODEL_VERSION,
                max_batch_size=1,
                supported_formats=["json+jpeg"],
            )

    # One worker is part of the correctness contract: the AMB3R map and
    # OnlineAMB3RSession state machine are mutable and session-scoped.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
        ],
    )
    vla_pb2_grpc.add_VLAServicer_to_server(AMB3RVOServicer(), server)
    address = f"{args.host}:{args.port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Could not bind AMB3R-VO RPC server to {address}")
    server.start()
    LOGGER.info(
        "AMB3R-VO RPC server listening on %s protocol=%s worker_count=1",
        address,
        AMB3R_VO_RPC_PROTOCOL_VERSION,
    )

    def _shutdown(_signum: int, _frame: Any) -> None:
        LOGGER.info("Stopping AMB3R-VO RPC server")
        server.stop(grace=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve causal AMB3R-VO relative poses over vla_rpc",
    )
    parser.add_argument("--repo", required=True, help="HeatmapVLN repository")
    parser.add_argument("--amb3r-root", required=True)
    parser.add_argument("--da3-checkpoint", required=True)
    parser.add_argument("--cfg-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50081)
    parser.add_argument("--map-init-window", type=int, default=20)
    parser.add_argument("--map-every", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(518, 392),
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=(
            "One train-calibrated deployment constant; never fitted from an "
            "evaluation episode or accepted over RPC"
        ),
    )
    parser.add_argument("--max-frames-limit", type=int, default=4096)
    parser.add_argument("--max-message-mb", type=int, default=32)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)
    for name in (
        "map_init_window",
        "map_every",
        "max_history",
        "max_frames_limit",
        "max_message_mb",
        "port",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(args.resolution) != 2 or any(int(value) < 1 for value in args.resolution):
        parser.error("--resolution values must be positive")
    if not np.isfinite(args.translation_scale) or args.translation_scale <= 0.0:
        parser.error("--translation-scale must be finite and positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    application = _build_real_application(args)
    return _serve(args, application)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMB3R_VO_RPC_MODEL_VERSION",
    "AMB3R_VO_RPC_PROTOCOL_VERSION",
    "AMB3RVORPCApplication",
    "RGB_FRONT_BLOB_NAME",
    "parse_args",
]
