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

Opt-in timing (``--timing`` or ``HEATMAPVLN_TIMING=1``) adds ``timing_ms`` to
every response: ``jpeg_decode``; ``ingest``, ``ingest_map_init`` or
``ingest_map_update`` (by what the call did to the map); ``query`` or
``query_map_update``; ``total`` (the dispatch).  Milliseconds, each stage
bounded by CUDA synchronisation of ``--device``.  On CUDA it also adds
``cuda_memory_mib`` (peak allocated / reserved during the call, whole-card use
after it).  Off, responses are unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
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


class _TimingOff:
    """Stand-in for a disabled ``src.utils.latency.StageTimer``: every call is a no-op.

    That module is imported only when timing is on, because ``src/`` joins
    ``sys.path`` late (``--repo``) and the dispatcher stays dependency-light.
    """

    enabled = False

    def stage(self, _name: str) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def rename(self, _old: str, _new: str) -> None:
        return None


_TIMING_OFF = _TimingOff()


def _map_event_suffix(before: Any, after: Any) -> str:
    """Name what a call did to the AMB3R map from the session's trajectory revision."""

    if not isinstance(before, int) or not isinstance(after, int) or after == before:
        return ""
    return "_map_init" if before == 0 else "_map_update"


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
        timing: bool = False,
        timing_device: Any = None,
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
        # Opt-in latency timing (HEATMAPVLN_TIMING=1 or --timing): per-stage
        # timing_ms (and CUDA memory of timing_device) in each response.  Off,
        # no clock is read and no field added.
        self.timing = bool(timing)
        self.timing_device = timing_device
        self._latency = None
        if self.timing:
            from src.utils import latency

            self._latency = latency

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

        timer = _TIMING_OFF
        if self.timing:
            timer = self._latency.StageTimer(enabled=True, device=self.timing_device)
            if self.timing_device is not None:
                self._latency.reset_accel_peak(self.timing_device)
        with timer.stage("total"):
            if method == _METHOD_RESET:
                output = self._reset(payload, blobs)
            elif method == _METHOD_INGEST:
                output = self._ingest(payload, blobs, timer)
            elif method == _METHOD_QUERY:
                output = self._query(payload, blobs, timer)
            else:
                raise ValueError(
                    f"Unsupported method {method!r}; expected one of "
                    + ", ".join(_SUPPORTED_METHODS)
                )
        self.requests_processed += 1
        response = self._response(output)
        if timer.enabled:
            response["timing_ms"] = timer.as_dict()
            if self.timing_device is not None:
                memory = self._latency.accel_memory_mib(self.timing_device)
                if memory is not None:
                    response["cuda_memory_mib"] = memory
        return response

    def _trajectory_revision(self) -> Any:
        return getattr(self.session, "trajectory_revision", None)

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

    def _ingest(
        self,
        payload: Any,
        blobs: Sequence[Any],
        timer: Any = _TIMING_OFF,
    ) -> dict[str, Any]:
        values = _strict_payload(
            payload,
            required={"session_id", "frame_id", "capture_step"},
        )
        identifier = _session_id(values)
        frame_id = _strict_int(values, "frame_id", minimum=0)
        capture_step = _strict_int(values, "capture_step", minimum=0)
        with timer.stage("jpeg_decode"):
            frame_rgb = self._decode_front_jpeg(blobs)
        revision = self._trajectory_revision() if timer.enabled else None
        # Timed as ingest, ingest_map_init or ingest_map_update by what it did to the map.
        with timer.stage("ingest"):
            result = self.session.ingest(
                identifier,
                frame_id=frame_id,
                frame_rgb=frame_rgb,
                capture_step=capture_step,
            )
        if timer.enabled:
            suffix = _map_event_suffix(revision, self._trajectory_revision())
            timer.rename("ingest", "ingest" + suffix)
        return result

    def _query(
        self,
        payload: Any,
        blobs: Sequence[Any],
        timer: Any = _TIMING_OFF,
    ) -> dict[str, Any]:
        values = _strict_payload(
            payload,
            required={"session_id", "current_frame_id", "history_frame_ids"},
        )
        if blobs:
            raise ValueError("query_relative_poses does not accept binary blobs")
        revision = self._trajectory_revision() if timer.enabled else None
        # Timed as query or query_map_update (the query first mapped the pending tail).
        with timer.stage("query"):
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
        if timer.enabled:
            suffix = _map_event_suffix(revision, self._trajectory_revision())
            timer.rename("query", "query" + suffix)
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


def _prepare_accelerator(device: str) -> None:
    """Bring up ``device`` and refuse to serve on a device that is not really there.

    The DA3 forward is the whole cost of this server, so a device that silently
    resolves elsewhere (or to CPU) would still answer every request, just with
    different numbers and a hundredfold slower.  Checked at startup instead.
    """
    import torch

    kind = torch.device(device).type
    if kind == "cpu":
        LOGGER.warning("VO server runs on CPU by explicit --device cpu")
        return
    if kind == "npu":
        try:
            import torch_npu  # noqa: F401  (registers torch.npu)
        except ImportError as exc:
            raise RuntimeError(f"--device {device} but torch_npu cannot be imported") from exc
    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available():
        raise RuntimeError(f"--device {device} but the {kind} backend is not available here")
    index = torch.device(device).index or 0
    if index >= backend.device_count():
        raise RuntimeError(
            f"--device {device} but only {backend.device_count()} {kind} device(s) are visible"
        )
    backend.set_device(index)
    # DA3's global attention runs over ~20 views x 1037 tokens.  Unchunked, the
    # reference CUDA kernel materialises a 20.6 GiB bf16 score matrix, which is why
    # the certified launcher sets a query chunk.  Serving without the variable set
    # would silently take the unchunked path.
    if kind == "npu":
        # torch's fused encoder-layer kernel has no NPU implementation and falls back
        # to the CPU (measured 365x slower than the ordinary path), so take the
        # ordinary one.  Logged because the two differ in summation order.
        mha = getattr(torch.backends, "mha", None)
        if mha is not None and hasattr(mha, "set_fastpath_enabled"):
            mha.set_fastpath_enabled(False)
            LOGGER.info("Fused MHA fastpath disabled (no NPU kernel; it falls back to the CPU)")
        # Make CANN build its op-compile toolchain now: inside the loaded, serving
        # process that initialisation forks a helper and has been seen to fail, which
        # would kill the first real request instead of the startup.  No RNG, no model
        # state, so it cannot change the poses served.
        import torch.nn.functional as functional

        with torch.no_grad():
            image = torch.zeros(1, 3, 28, 28, device=device, dtype=torch.bfloat16)
            kernel = torch.zeros(8, 3, 14, 14, device=device, dtype=torch.bfloat16)
            functional.conv2d(image, kernel, stride=14)
        backend.synchronize(device)
        LOGGER.info("NPU op toolchain ready (conv2d in bf16 on %s)", device)
    chunk = os.environ.get("DA3_SDPA_QUERY_CHUNK_SIZE", "")
    # These are this process's INPUTS, echoed for the record.  bf16 here is
    # is_bf16_supported(), which says the card can do bf16, not that any forward
    # used it; the chunk size is what this server was handed, not what DA3 made of
    # it.  _report_da3_attention below asks DA3 itself.
    LOGGER.info(
        "VO server device: %s (%s %s, bf16=%s, DA3_SDPA_QUERY_CHUNK_SIZE=%s, "
        "DA3_DISABLE_XFORMERS=%s)",
        device,
        kind,
        torch.__version__,
        getattr(backend, "is_bf16_supported", lambda: "unknown")(),
        chunk or "unset",
        os.environ.get("DA3_DISABLE_XFORMERS", "unset"),
    )


def _report_da3_attention() -> None:
    """What DA3 itself makes of the chunked-attention settings.

    The launcher used to prove the chunked path by exporting
    DA3_SDPA_QUERY_CHUNK_SIZE, having this server echo it back, and grepping its own
    export: a check that could not fail whatever DA3 did with the variable, and the
    deploy doc quoted it as proof the certified path was taken.  This asks the other
    side instead -- DA3's own parser, and the attention function the dinov2 layers
    actually bound -- so the line can be absent or wrong, which is the point of it.

    Still not proof that a given forward chunked: that also depends on the query
    length (memory_bounded_scaled_dot_product_attention falls through to plain SDPA
    when query_length <= chunk_size).  Read-only and import-only: no device work, no
    RNG, so it cannot move a served number on either platform.
    """
    try:
        from depth_anything_3.model.dinov2.layers import attention as dinov2_attention
        from depth_anything_3.model.utils.memory_bounded_attention import (
            _configured_query_chunk_size,
            memory_bounded_scaled_dot_product_attention,
        )

        bound = getattr(dinov2_attention, "memory_bounded_scaled_dot_product_attention", None)
        LOGGER.info(
            "DA3 attention: query_chunk=%s (parsed by DA3), memory_bounded=%s, xformers_disabled=%s",
            _configured_query_chunk_size(),
            bound is memory_bounded_scaled_dot_product_attention,
            os.environ.get("DA3_DISABLE_XFORMERS", "") == "1",
        )
    except Exception as exc:  # noqa: BLE001 - evidence must not be able to stop a server
        LOGGER.warning("could not read DA3's attention configuration (%s: %s)", type(exc).__name__, exc)


def _npu_poison_after_failure(device: str, exc: BaseException) -> str | None:
    """After a failed request, whether the NPU itself can still compute.

    An Ascend AICPU timeout (ACL error 507017) mid-request leaves the process alive
    and listening: HealthCheck, GetServerInfo, reset_episode and every ingest keep
    succeeding, and only the next DA3 mapping forward fails.  So the device is
    probed rather than the exception read: src/vo/online_amb3r.py raises
    ValueError/RuntimeError for bad requests and numerical failures too, and those
    leave the stream healthy, so the synchronize returns and nothing is recorded.

    Returns the original failure as a short reason when the device is gone.
    """
    import torch

    try:
        torch.npu.synchronize(device)
    except Exception as sync_exc:
        LOGGER.error(
            "NPU device unusable after a failed request; HealthCheck now reports "
            "NOT_SERVING (synchronize: %s: %s)",
            type(sync_exc).__name__,
            sync_exc,
        )
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


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

    _prepare_accelerator(args.device)

    # DA3 directly, not model_zoo.load_model("da3"): load_model takes no device and
    # would place the backbone with DA3's own default .to("cuda"), ignoring --device.
    from amb3r.model_zoo import DA3
    from src.utils.latency import timing_enabled
    from src.vo.online_amb3r import build_online_amb3r_session
    from vla_rpc.core.image import decode_jpeg_to_rgb

    model = DA3(device=args.device, ckpt_path=str(checkpoint))
    # After the model is built, so the import order of the DA3 package is the one the
    # certified runs had.
    _report_da3_attention()
    session = build_online_amb3r_session(
        model,
        cfg_path=cfg_path,
        device=args.device,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
        max_history=args.max_history,
        resolution=tuple(args.resolution),
        rng_seed=args.rng_seed,
    )
    return AMB3RVORPCApplication(
        session,
        jpeg_decoder=decode_jpeg_to_rgb,
        translation_scale=args.translation_scale,
        max_frames_limit=args.max_frames_limit,
        timing=bool(args.timing) or timing_enabled(),
        timing_device=args.device,
    )


def _serve(args: argparse.Namespace, application: AMB3RVORPCApplication) -> int:
    import grpc
    import torch
    from vla_rpc.proto import vla_pb2, vla_pb2_grpc

    device_type = torch.device(args.device).type
    rng_seed = "none" if args.rng_seed is None else str(args.rng_seed)
    # The Ascend slots reuse the ports of the 4090's own local servers, and the
    # model version is a constant, so without these a client cannot tell which
    # machine (or which seed and timing setting) answered.  Appended after the
    # existing format, and only with --server-instance.
    instance_formats = (
        [
            f"heatmapvln-instance:{args.server_instance}",
            f"heatmapvln-device:{device_type}",
            f"heatmapvln-timing:{int(application.timing)}",
            f"heatmapvln-vo-rng-seed:{rng_seed}",
        ]
        if args.server_instance
        else []
    )

    class AMB3RVOServicer(vla_pb2_grpc.VLAServicer):
        def __init__(self) -> None:
            # Set once, on NPU only, when a failed request left the device unable
            # to synchronize; from then on HealthCheck reports NOT_SERVING with it.
            self.device_poison: str | None = None

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
                if self.device_poison is None and device_type == "npu":
                    self.device_poison = _npu_poison_after_failure(args.device, exc)
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
            if self.device_poison is not None:
                return vla_pb2.HealthCheckResponse(
                    status=vla_pb2.HealthCheckResponse.NOT_SERVING,
                    message=self.device_poison,
                    version=AMB3R_VO_RPC_PROTOCOL_VERSION,
                    requests_processed=application.requests_processed,
                )
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
                supported_formats=["json+jpeg", *instance_formats],
            )

    options = [
        ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
        ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
    ]
    # The client pings every 30 s (vla_rpc sync_client sets grpc.keepalive_time_ms
    # 30000).  A gRPC server defaults to tolerating two pings per five minutes on a
    # channel carrying no data, and then sends GOAWAY/ENHANCE_YOUR_CALM: measured
    # twice on the 910B, the VO channel sits idle for minutes while the slower model
    # call runs, the client's keepalive pings trip that limit, and the shard dies on
    # "AMB3R VO RPC returned no response for ingest_frame" with the client having no
    # RPC retry.  Accept the pings the client actually sends.  Transport only: no
    # request, response or number changes, on either platform.
    options += [
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.min_ping_interval_without_data_ms", 20000),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    if args.server_instance:
        # gRPC binds with SO_REUSEPORT by default on Linux, so a second server
        # started on an occupied port binds too and the kernel splits connections
        # between the two.  A named instance must own its port or fail to start.
        options.append(("grpc.so_reuseport", 0))
    # One worker is part of the correctness contract: the AMB3R map and
    # OnlineAMB3RSession state machine are mutable and session-scoped.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=options,
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
    # Otherwise visible only in the process arguments; log-only, so no reply changes.
    LOGGER.info("VO device RNG seed per reset_episode: %s", rng_seed)
    if application.timing:
        LOGGER.info("Latency timing enabled: responses carry timing_ms")

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
        "--rng-seed",
        type=int,
        default=None,
        help=(
            "Re-seed the device RNG at every reset_episode. The DA3 forward subsamples "
            "from the device's default generator without seeding it; on CUDA that was "
            "reproducible because the default seed is a constant, on Ascend it is not. "
            "Unset keeps the CUDA reference behaviour."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Same as HEATMAPVLN_TIMING=1: add per-stage timing_ms and CUDA "
            "memory to every response (stages synchronise --device; poses unchanged)"
        ),
    )
    parser.add_argument(
        "--server-instance",
        default="",
        help=(
            "Token naming this server process. When set, GetServerInfo adds "
            "heatmapvln-instance/-device/-timing/-vo-rng-seed entries to "
            "supported_formats and the port is bound without SO_REUSEPORT, so a "
            "client can refuse a different server that happens to listen on the "
            "same port. Empty changes nothing."
        ),
    )
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
