#!/usr/bin/env python3
"""Audited DAgger RPC facade around the locked native InternNav server.

The model runtime, prompt state machine, generate_latents, generate_traj and
traj_to_actions remain owned by the native server. This facade only gives that
existing joint policy a distinct RPC method and adds fail-closed policy
provenance to ServerInfo and every response.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import rpc_internnav_native_server as native_server


LOGGER = logging.getLogger("internnav-native-dagger-facade")
RPC_METHOD = "plan_native_internnav"
NATIVE_PROTOCOL = "internnav-native-joint-front-history-lookdown-v1"
POLICY_BACKEND = "internnav_native"
_FINGERPRINT_RE = re.compile(r"^internnav-native-v1:[0-9a-f]{64}$")


def _required_fingerprint() -> str:
    fingerprint = os.environ.get("INTERNNAV_NATIVE_POLICY_FINGERPRINT", "").strip()
    if not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise RuntimeError(
            "INTERNNAV_NATIVE_POLICY_FINGERPRINT must match "
            "'internnav-native-v1:<64 lowercase hex>'"
        )
    return fingerprint


POLICY_FINGERPRINT = _required_fingerprint()


_original_plan = native_server.InternNavNativeRuntime.plan_panoramic


def _plan_native(
    self: Any,
    payload: dict[str, Any],
    blobs: Any,
) -> dict[str, Any]:
    expected = {
        "policy_backend": POLICY_BACKEND,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "native_protocol": NATIVE_PROTOCOL,
        "phase": "joint",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native DAgger RPC request contract mismatch: {mismatches}")

    response = _original_plan(self, payload, blobs)
    response.update(
        {
            "phase": "joint",
            "policy_backend": POLICY_BACKEND,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "native_protocol": NATIVE_PROTOCOL,
            "system1_source": "internnav_native_nextdit_async",
        }
    )
    return response


native_server.InternNavNativeRuntime.plan_panoramic = _plan_native


_original_infer_json = native_server.InternNavNativeServicer.InferJSON


def _infer_native(self: Any, request: Any, context: Any) -> Any:
    if request.method != RPC_METHOD:
        message = (
            f"Audited native InternNav server accepts only method {RPC_METHOD!r}, "
            f"got {request.method!r}"
        )
        context.set_details(message)
        context.set_code(native_server.grpc.StatusCode.INVALID_ARGUMENT)
        return native_server.vla_pb2.JSONResponse(
            ts=request.ts,
            json_payload=json.dumps({"ok": False, "error": message}),
        )

    forwarded = native_server.vla_pb2.JSONRequest()
    forwarded.CopyFrom(request)
    forwarded.method = "plan_panoramic"
    return _original_infer_json(self, forwarded, context)


native_server.InternNavNativeServicer.InferJSON = _infer_native


_original_server_info = native_server.InternNavNativeServicer.GetServerInfo


def _get_native_server_info(self: Any, request: Any, context: Any) -> Any:
    response = _original_server_info(self, request, context)
    response.model_version = f"internnav-native-r2r:{POLICY_FINGERPRINT}"
    if NATIVE_PROTOCOL not in response.supported_formats:
        response.supported_formats.append(NATIVE_PROTOCOL)
    return response


native_server.InternNavNativeServicer.GetServerInfo = _get_native_server_info


def main() -> int:
    print(
        "[internnav-native-dagger-facade] "
        f"method={RPC_METHOD} protocol={NATIVE_PROTOCOL} "
        f"fingerprint={POLICY_FINGERPRINT}",
        flush=True,
    )
    LOGGER.info(
        "Audited native DAgger facade active: method=%s protocol=%s fingerprint=%s",
        RPC_METHOD,
        NATIVE_PROTOCOL,
        POLICY_FINGERPRINT,
    )
    return native_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
