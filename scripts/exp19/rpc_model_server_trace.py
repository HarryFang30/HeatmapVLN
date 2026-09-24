#!/usr/bin/env python3
"""EXP-19: the deployed model RPC server, plus a read-only per-call trace.

Launch it exactly like ``scripts/evaluation/rpc_model_server.py`` (same CLI,
same environment).  It imports that module, wraps existing functions at
class/module level (``install()``) and calls the module's unchanged ``main()``;
the deployed server file stays byte-identical.  Every wrapper calls the
original with the original arguments, keeps *references* to what went in and
came out, and returns the original object.  Nothing is copied, cast or moved
off the device until ``_plan_panoramic_native`` has returned its final response
dict, and that dict goes back to the servicer as the same, unchanged object.

Wrapped (``uninstall()`` restores the originals):
  HeatmapVLNRuntime._plan_panoramic_native   opens the per-call context; after
                                              the original returns: trace,
                                              diagnostics, write
  HeatmapVLNRuntime._build_ppa_past_output    History Head output dict
  PastPlanActionChain.form_plan               (plan_z0, plan_z, diagnostics) and
                                              its history_memory / mask
  PastPlanActionChain.decode_future           Future Head output dict
  NextDiTActionHead.get_trajectory_from_projected
                                              System1 trajectory on the bridge
                                              path, its traj_images / generator
  rpc_model_server._trajectory_from_condition System1 trajectory during AMB3R
                                              warm-up
  HeatmapVLNRuntime.__init__                  refuses to serve when a wrapper
                                              does not reach the loaded model,
                                              or the model is not the deployed
                                              ppa-online-amb3r arm
Captures happen only inside an open (thread-local) call context, so the
wrappers are inert everywhere else, including in the diagnostics below.

Environment:
  EXP19_TRACE_DIR          required; the server refuses to start without it
  EXP19_TRACE_DIAGNOSTICS  "1" (default) runs the diagnostics, "0" skips them

Output per call: ``<EXP19_TRACE_DIR>/<ep_key>/call_<system2_call_index:03d>``,
``ep_key = <scene_id>_<episode_id:04d>`` from the request's deterministic
sampling key.  The npz is written first and the json last (both atomically),
so a json marks a complete call.

json (schema ``exp19-call-trace-v1``)
  identity    scene_id, episode_id, ep_key, system2_call_index, per_call_seed,
              protocol_seed
  join keys   current_capture_step, history_capture_steps, history_age_steps,
              vo_current_frame_id, vo_history_frame_ids, pose_ready,
              vo_provider_phase, instruction (copied from the request)
  request     the payload as received (no blobs); blob_names: every blob name
  response    the final response dict
  trajectory_path  "ppa" (bridge ran), "warmup" (native System1 before the
              first AMB3R map), null (System2 answered arrows / STOP)
  has_past_output, has_future_output
  selected_path_xy  [33,2] the System1 path the actions were cut from, robot
              frame at call time (x forward, y left, metres), recomputed with
              the deployment's own functions (traj_to_actions internals)
  recomputed_actions, recomputed_anti_deadlock, actions_match (vs response)
  checks      identity checks that the captured tensors are the ones the
              deployment passed on (plan_z -> System1 and Future Head, History
              Head output -> bridge and Future Head) and per-wrapper call counts
  trace_warnings   inconsistencies between the captures and the response
  diagnostic  {diagnostic_only: true, enabled, skipped_reason,
              bridge_attention_available, bridge_attention_check,
              counterfactual_no_memory, replay_same_plan, errors}
  shapes, dtypes (every npz array), npz {file, sha256, bytes}, timing_s
  (model_call, capture, diagnostics, npz_compress, npz_write: the json is
  written after the npz, so it carries its own call's npz write time),
  provenance (server module path, source tree + its .exp19_git_sha, pid, host)

npz (an array is present only when the module producing it ran; views F,R,B,L)
  jpeg__current__{front,right,back,left}, jpeg__history__<i>__front (i <
    num_history), jpeg__lookdown    the JPEG bytes received, uint8 1-D
  hist_heatmaps_gated [8,4,64,64] f16, hist_heatmaps [8,4,64,64] f16 (sigmoid),
  hist_view_peak_yx [8,4,2] int16  per-view argmax (y, x) of the f32
    ``heatmaps`` (what validate.py's joint PCK scores; exact, unlike an argmax
    of the f16 copy, whose near-1 values tie)
  hist_visibility_logits [8,4] f32, hist_none_probability [8] f32,
  hist_mask [8] bool, hist_memory [8,256] f32 (M, the bridge's keys/values)
  plan_z0, plan_z, delta_z [4,768] f32, delta_token_ratio [4] f32
  fut_heatmaps_gated [4,4,64,64] f16, fut_heatmaps [4,4,64,64] f16,
  fut_visibility_probability [4,4] f32
  trajectory_raw [32,32,3] f32 (NextDiT output as returned),
  selected_path_xy [33,2] f64
  diagnostics: bridge_attention [heads,4,8] f32, cf_trajectory_raw,
  cf_selected_path_xy

Diagnostics (``diagnostic_only``; only on calls where the bridge ran; strictly
after the response is final, under ``torch.no_grad()``; they never touch the
deployed generator, the response or any module state):
  (a) bridge attention: query / key_value recomputed exactly as
      ``PastToPlanBridge.forward`` does (fp32, both LayerNorms, the same
      key_padding_mask, autocast off), then ``cross_attention(...,
      need_weights=True, average_attn_weights=False)``.  The recomputed output
      after the same trust-region cap must reproduce the deployed ``delta_z``
      (max abs difference recorded); the weights are kept only if it does.
  (b) counterfactual without memory: the same NextDiT call with plan_z0 in
      place of plan_z, the same traj_images and a fresh generator seeded with
      the same per_call_seed, then the deployment's post-processing
      (selection, x_sign, target heading, _finalize_local_actions, STOP->LEFT).
  (c) replay: (b) with the deployed plan_z.  Same noise and inputs, so any
      difference to the deployed trajectory is device non-determinism: the
      floor below which a counterfactual action change means nothing.

Errors raised while tracing are caught, appended with traceback to
``<ep_key>/trace_errors.jsonl`` (``trace_errors.jsonl`` at the root when the
call cannot be named) and logged; the response is returned unchanged.  When
the full record of a named call cannot be built or written, a fallback
``call_<idx>.json`` (same schema) still keeps the call joinable: identity, join
keys, request, response, ``trace_failed: true``, ``traceback``, ``npz: null``
(written best effort; its own failure is only logged).
Exceptions of the original call propagate untouched.  A call 0 arriving for an
episode that already has files (a restarted episode) first moves the old
directory to ``_superseded/<ep_key>__<n>``.

Example (production goes through scripts/exp19/run_rerun.sh):
  EXP19_TRACE_DIR=/tmp/exp19_trace CUDA_VISIBLE_DEVICES=0 \\
    $W/envs/qwen25/bin/python -u scripts/exp19/rpc_model_server_trace.py \\
    --config configs/ppa_action_refine_v2_8gpu.yaml --checkpoint <best.pth> \\
    --gpu_id 0 --port 52640 --workers 1 \\
    --require_deterministic_sampling --require_ppa_online_amb3r
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Like the deployed server: this source tree first, so an archived copy never
# picks up another checkout's scripts/ or src/ from PYTHONPATH.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from scripts.evaluation import rpc_model_server as S  # noqa: E402
from src.models.action.nextdit_action_head import NextDiTActionHead  # noqa: E402
from src.models.past_plan_action import PastPlanActionChain  # noqa: E402

SCHEMA = "exp19-call-trace-v1"
TRACE_DIR_ENV = "EXP19_TRACE_DIR"
DIAGNOSTICS_ENV = "EXP19_TRACE_DIAGNOSTICS"
GIT_SHA_FILE = ".exp19_git_sha"
# The deployed delta_z is the fp32 capped attention cast to the Plan dtype
# (bf16: up to 2^-8 relative rounding), and need_weights=True takes the explicit
# softmax path instead of SDPA, so the recompute matches to bf16 rounding.
ATTENTION_RTOL = 1e-2
ATTENTION_ATOL_FRACTION = 1e-3  # of max |delta_z|
LOGGER = logging.getLogger("exp19-trace")
# Copied from the request into every call json (also the fallback one).
JOIN_KEYS = (
    "current_capture_step",
    "history_capture_steps",
    "history_age_steps",
    "vo_current_frame_id",
    "vo_history_frame_ids",
    "pose_ready",
    "vo_provider_phase",
    "instruction",
)

_LOCAL = threading.local()
_TRACER: CallTracer | None = None
_PATCHES: list[tuple[Any, str, Any]] = []


# ---------------------------------------------------------------------------
# Per-call context and wrappers
# ---------------------------------------------------------------------------


@dataclass
class _Capture:
    signature: inspect.Signature
    args: tuple
    kwargs: dict
    result: Any

    def arguments(self) -> dict[str, Any]:
        """Arguments by parameter name (``self`` included for methods)."""
        return dict(self.signature.bind(*self.args, **self.kwargs).arguments)


@dataclass
class _CallContext:
    request: dict | None = None
    captures: dict[str, list[_Capture]] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    def last(self, slot: str) -> _Capture | None:
        items = self.captures.get(slot)
        return items[-1] if items else None


def _active_context() -> _CallContext | None:
    return getattr(_LOCAL, "context", None)


def _capturing(slot: str) -> Callable[[Callable], Callable]:
    def make(original: Callable) -> Callable:
        signature = inspect.signature(original)

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            context = _active_context()
            if context is not None:
                context.captures.setdefault(slot, []).append(
                    _Capture(signature, args, kwargs, result)
                )
            return result

        return wrapper

    return make


def _make_plan_wrapper(original: Callable) -> Callable:
    def _plan_panoramic_native(self, payload, blobs):
        context = _CallContext()
        try:
            context.request = json.loads(json.dumps(payload))
        except Exception:  # the snapshot must never block the call
            context.request = None
        previous = _active_context()
        _LOCAL.context = context
        started = time.perf_counter()
        try:
            response = original(self, payload, blobs)
        finally:
            _LOCAL.context = previous
        context.timing["model_call"] = time.perf_counter() - started
        tracer = _TRACER
        if tracer is not None:
            try:
                tracer.trace_call(self, payload, blobs, response, context)
            except Exception:
                tracer.log_error(
                    _identity_or_none(payload, response),
                    "trace_call",
                    traceback.format_exc(),
                )
        return response

    return _plan_panoramic_native


def _make_init_wrapper(original: Callable) -> Callable:
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        verify_patch_reach(self)

    return __init__


def _is_wrapped(function: Any) -> bool:
    return getattr(function, "_exp19_original", None) is not None


def _targets() -> tuple[tuple[Any, str, Callable[[Callable], Callable]], ...]:
    return (
        (S.HeatmapVLNRuntime, "_plan_panoramic_native", _make_plan_wrapper),
        (S.HeatmapVLNRuntime, "__init__", _make_init_wrapper),
        (S.HeatmapVLNRuntime, "_build_ppa_past_output", _capturing("past_output")),
        (PastPlanActionChain, "form_plan", _capturing("form_plan")),
        (PastPlanActionChain, "decode_future", _capturing("decode_future")),
        (NextDiTActionHead, "get_trajectory_from_projected", _capturing("projected")),
        (S, "_trajectory_from_condition", _capturing("warmup")),
    )


def install(trace_dir: str | Path, *, diagnostics: bool = True) -> CallTracer:
    """Wrap the targets in place and route traces to ``trace_dir``."""

    global _TRACER
    if _PATCHES:
        raise RuntimeError("EXP-19 trace wrappers are already installed")
    server_file = Path(S.__file__).resolve()
    if not server_file.is_relative_to(SOURCE_ROOT):
        raise RuntimeError(
            f"rpc_model_server was imported from {server_file}, outside this "
            f"source tree {SOURCE_ROOT}; the trace would wrap another checkout"
        )
    tracer = CallTracer(Path(trace_dir), diagnostics=diagnostics)
    try:
        for owner, name, make in _targets():
            original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
            if _is_wrapped(original):
                raise RuntimeError(f"{owner.__name__}.{name} is already wrapped")
            wrapper = functools.wraps(original)(make(original))
            wrapper._exp19_original = original
            setattr(owner, name, wrapper)
            _PATCHES.append((owner, name, original))
    except Exception:
        uninstall()
        raise
    _TRACER = tracer
    return tracer


def uninstall() -> None:
    global _TRACER
    while _PATCHES:
        owner, name, original = _PATCHES.pop()
        setattr(owner, name, original)
    _TRACER = None


def verify_patch_reach(runtime) -> None:
    """Fail closed when a wrapper cannot see what the loaded model runs."""

    model = runtime.model
    head = getattr(model, "nextdit_action_head", None)
    chain = getattr(model, "past_plan_action", None)
    problems = []
    if head is not None and not _is_wrapped(type(head).get_trajectory_from_projected):
        problems.append(f"{type(head).__qualname__}.get_trajectory_from_projected")
    if chain is not None:
        for name in ("form_plan", "decode_future"):
            if not _is_wrapped(getattr(type(chain), name)):
                problems.append(f"{type(chain).__qualname__}.{name}")
    for name in ("_plan_panoramic_native", "_build_ppa_past_output"):
        if not _is_wrapped(getattr(type(runtime), name)):
            problems.append(f"{type(runtime).__qualname__}.{name}")
    if not _is_wrapped(S._trajectory_from_condition):
        problems.append("rpc_model_server._trajectory_from_condition")
    # Only the deployed arm is traced: the EXP-17 cognition arm never reaches
    # _plan_panoramic_native, and Postprocess mirrors only the Stage-0-disabled
    # action post-processing.
    if not getattr(runtime, "ppa_online_amb3r", False):
        problems.append("runtime is not the ppa-online-amb3r arm (--require_ppa_online_amb3r)")
    elif head is None or chain is None:
        problems.append("PPA runtime without a NextDiT head and Past->Plan->Action chain")
    if getattr(runtime, "system2_cognition_arm", False):
        problems.append("--system2_cognition_arm bypasses _plan_panoramic_native")
    if getattr(runtime, "ppa_stage0_action_arm", "disabled") != "disabled":
        problems.append(f"--ppa_stage0_action_arm={runtime.ppa_stage0_action_arm}")
    if problems:
        raise RuntimeError(
            "EXP-19 trace cannot follow the loaded model: " + ", ".join(problems)
        )


# ---------------------------------------------------------------------------
# Deployment post-processing (trajectory -> action chunk), keeping the path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Postprocess:
    """``traj_to_actions`` -> ``_finalize_local_actions`` -> STOP->LEFT, as deployed."""

    num_sample_trajs: int
    action_scale: float
    selection: str = "mean"
    x_sign: float = 1.0
    target_heading_deg: float | None = None

    @classmethod
    def for_call(cls, runtime, request: dict, response: dict) -> Postprocess:
        # The response echoes the exact target heading the server used (None
        # unless trajectory_heading_alignment == "pano_pixel").
        heading = response.get("trajectory_target_heading_deg")
        return cls(
            num_sample_trajs=int(runtime.num_sample_trajs),
            action_scale=runtime.action_scale,
            selection=str(request.get("trajectory_selection", "mean")),
            x_sign=float(request.get("trajectory_x_sign", 1.0)),
            target_heading_deg=None if heading is None else float(heading),
        )

    def __call__(self, trajectory: torch.Tensor) -> tuple[np.ndarray, list[int], bool]:
        """Return (selected path [T+1,2] metres, actions, anti_deadlock)."""

        # Line for line S.traj_to_actions, which discards the selected path.
        if self.x_sign not in (-1.0, 1.0):
            raise ValueError(f"trajectory_x_sign must be -1 or 1, got {self.x_sign}")
        trajs = trajectory[: self.num_sample_trajs].float().detach().cpu().numpy().copy()
        trajs[:, :, :2] /= self.action_scale
        trajs[:, :, 0] *= self.x_sign
        all_trajectory = S.reconstruct_xy_from_delta(trajs)
        path, _selected_idx = S.select_trajectory_xy(all_trajectory, self.selection)
        if self.target_heading_deg is not None:
            path, _rotation_deg = S.align_trajectory_endpoint_heading(
                path, target_angle_deg=float(self.target_heading_deg)
            )
        actions = S._trajectory_to_discrete_actions_close_to_goal(path)
        # _plan_panoramic_native with ppa_stage0_action_arm == "disabled".
        actions = S._finalize_local_actions(actions if actions else [S.ActionCode.STOP])
        anti_deadlock = bool(actions) and actions[0] == S.ActionCode.STOP
        if anti_deadlock:
            actions = [S.ActionCode.LEFT]
        return np.asarray(path, dtype=np.float64), [int(a) for a in actions], anti_deadlock


# ---------------------------------------------------------------------------
# Diagnostic (a): bridge attention
# ---------------------------------------------------------------------------


def recompute_bridge_attention(
    bridge,
    plan_z0: torch.Tensor,
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    delta_z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Re-run ``PastToPlanBridge.forward``'s attention, returning its weights.

    Returns ``(weights [B_valid, heads, Q, N], check)`` or None when no sample
    has memory (the deployed forward then bypasses attention entirely).
    """

    device = bridge.cross_attention.out_proj.weight.device
    with torch.no_grad(), torch.autocast(device_type=plan_z0.device.type, enabled=False):
        mask = memory_mask.to(device=device)
        idx = mask.any(dim=1).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        z0 = plan_z0.index_select(0, idx).to(dtype=torch.float32)
        query = bridge.plan_norm(z0)
        key_value = bridge.memory_norm(memory.index_select(0, idx).to(dtype=torch.float32))
        output, weights = bridge.cross_attention(
            query,
            key_value,
            key_value,
            key_padding_mask=~mask.index_select(0, idx),
            need_weights=True,
            average_attn_weights=False,
        )
        if bridge.max_delta_ratio is not None:
            scale = torch.clamp(
                bridge.max_delta_ratio
                * z0.norm(dim=-1, keepdim=True)
                / output.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                max=1.0,
            )
            output = output * scale
        deployed = delta_z.index_select(0, idx)
        deployed32 = deployed.float()
        max_delta = float(deployed32.abs().max())
        atol = ATTENTION_ATOL_FRACTION * max_delta
        check = {
            "max_abs_diff_vs_delta_z": float((output - deployed32).abs().max()),
            "max_abs_delta_z": max_delta,
            "rtol": ATTENTION_RTOL,
            "atol": atol,
            "reproduces_delta_z": bool(
                torch.allclose(output, deployed32, rtol=ATTENTION_RTOL, atol=atol)
            ),
            "bitwise_equal_after_cast": bool(
                torch.equal(output.to(dtype=deployed.dtype), deployed)
            ),
            "capped": bridge.max_delta_ratio is not None,
            "samples_with_memory": [int(i) for i in idx.tolist()],
        }
    return weights, check


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


def _to_numpy(tensor: torch.Tensor, dtype) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(dtype, copy=False)


def _single(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got shape {tuple(tensor.shape)}")
    return tensor[0]


def _view_peak_yx(heatmaps: np.ndarray) -> np.ndarray:
    """Per-slot, per-view argmax (y, x) of [K,V,H,W]; first maximum, like torch."""

    height, width = heatmaps.shape[-2:]
    flat = heatmaps.reshape(*heatmaps.shape[:-2], height * width).argmax(axis=-1)
    return np.stack([flat // width, flat % width], axis=-1).astype(np.int16)


def _jpeg_arrays(blobs) -> tuple[list[str], dict[str, np.ndarray]]:
    """Raw JPEG bytes: current four views, history fronts, lookdown."""

    names, arrays = [], {}
    for blob in blobs:
        names.append(str(blob.name))
        parts = str(blob.name).split("/")
        keep = (
            (len(parts) == 2 and parts[0] == "current")
            or (len(parts) == 3 and parts[0] == "history" and parts[2] == "front")
            or parts == ["lookdown"]
        )
        if keep:
            arrays["jpeg__" + "__".join(parts)] = np.frombuffer(blob.data, dtype=np.uint8)
    return names, arrays


def _call_identity(payload: dict, response: dict) -> dict[str, Any]:
    field_name = S.HEATMAPVLN_RPC_SAMPLING_FIELD
    # The response echoes the server-validated key; the payload is the fallback.
    sampling = response.get(field_name)
    if not isinstance(sampling, dict):
        sampling = payload.get(field_name)
    if not isinstance(sampling, dict):
        raise ValueError(
            f"request carries no {field_name!r} key, so the call cannot be named"
        )
    scene_id = str(sampling["scene_id"])
    episode_id = int(sampling["episode_id"])
    seed = sampling.get("per_call_seed")
    protocol_seed = sampling.get("protocol_seed")
    return {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "ep_key": f"{scene_id}_{episode_id:04d}",
        "system2_call_index": int(sampling["system2_call_index"]),
        "per_call_seed": None if seed is None else int(seed),
        "protocol_seed": None if protocol_seed is None else int(protocol_seed),
    }


def _identity_or_none(payload, response) -> dict[str, Any] | None:
    try:
        return _call_identity(payload, response)
    except Exception:
        return None


def _read_git_sha() -> str | None:
    try:
        return (SOURCE_ROOT / GIT_SHA_FILE).read_text().strip() or None
    except OSError:
        return None


def _plain_json(value: Any) -> Any:
    """A JSON round trip that never fails on a stray non-JSON value (repr instead)."""
    return json.loads(json.dumps(value, default=repr))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


@dataclass
class _Resolved:
    """What one call's captures mean, after the original returned."""

    trajectory_path: str | None = None
    trajectory: torch.Tensor | None = None
    selected_path_xy: np.ndarray | None = None
    postprocess: Postprocess | None = None
    plan: _Capture | None = None
    projected: _Capture | None = None


class CallTracer:
    def __init__(self, trace_dir: Path, *, diagnostics: bool):
        self.trace_dir = Path(trace_dir)
        self.diagnostics = bool(diagnostics)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "server_module": str(Path(S.__file__).resolve()),
            "source_root": str(SOURCE_ROOT),
            "source_git_sha": _read_git_sha(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }

    # -- entry point ---------------------------------------------------------

    def trace_call(self, runtime, payload, blobs, response, context: _CallContext) -> None:
        started = time.perf_counter()
        identity = _call_identity(payload, response)
        ep_dir = self._episode_dir(identity["ep_key"], identity["system2_call_index"])
        try:
            record, arrays, resolved = self._build(runtime, payload, blobs, response, context, identity)
            captured = time.perf_counter()
            record["diagnostic"] = self._diagnostics(response, resolved, arrays, identity)
            diagnosed = time.perf_counter()
            record["shapes"] = {name: list(array.shape) for name, array in arrays.items()}
            record["dtypes"] = {name: str(array.dtype) for name, array in arrays.items()}
            record["timing_s"] = {
                **context.timing,
                "capture": captured - started,
                "diagnostics": diagnosed - captured,
            }
            record["provenance"] = self._provenance()
            self._write(ep_dir, identity["system2_call_index"], record, arrays)
        except Exception:
            # The caller logs the error; the call must stay joinable to [C] anyway.
            self._write_fallback(ep_dir, identity, payload, response, context, traceback.format_exc())
            raise

    def log_error(self, identity: dict | None, where: str, trace_text: str) -> None:
        ep_key = identity.get("ep_key") if identity else None
        call_index = identity.get("system2_call_index") if identity else None
        LOGGER.error(
            "EXP-19 trace error in %s (ep=%s call=%s); the response is unaffected\n%s",
            where,
            ep_key,
            call_index,
            trace_text,
        )
        try:
            directory = self.trace_dir / ep_key if ep_key else self.trace_dir
            directory.mkdir(parents=True, exist_ok=True)
            entry = {
                "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "where": where,
                "ep_key": ep_key,
                "system2_call_index": call_index,
                "traceback": trace_text,
            }
            with open(directory / "trace_errors.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.exception("EXP-19 could not append to trace_errors.jsonl")

    # -- pieces --------------------------------------------------------------

    def _episode_dir(self, ep_key: str, call_index: int) -> Path:
        ep_dir = self.trace_dir / ep_key
        if call_index == 0 and ep_dir.is_dir() and any(ep_dir.iterdir()):
            archive = self.trace_dir / "_superseded"
            archive.mkdir(exist_ok=True)
            n = 1
            while (archive / f"{ep_key}__{n}").exists():
                n += 1
            ep_dir.rename(archive / f"{ep_key}__{n}")
            LOGGER.warning(
                "EXP-19 episode %s restarted at call 0; earlier trace moved to %s",
                ep_key,
                archive / f"{ep_key}__{n}",
            )
        ep_dir.mkdir(parents=True, exist_ok=True)
        return ep_dir

    def _build(self, runtime, payload, blobs, response, context, identity):
        request = context.request if context.request is not None else payload
        blob_names, arrays = _jpeg_arrays(blobs)
        past = context.last("past_output")
        plan = context.last("form_plan")
        future = context.last("decode_future")
        projected = context.last("projected")
        warmup = context.last("warmup")
        warnings = []
        if context.request is None:
            warnings.append("request payload could not be snapshotted as JSON")

        if past is not None and past.result is not None:
            out = past.result
            heatmaps = _to_numpy(_single(out["heatmaps"]), np.float32)
            arrays["hist_heatmaps_gated"] = _to_numpy(_single(out["heatmaps_gated"]), np.float16)
            arrays["hist_heatmaps"] = heatmaps.astype(np.float16)
            arrays["hist_view_peak_yx"] = _view_peak_yx(heatmaps)
            arrays["hist_visibility_logits"] = _to_numpy(_single(out["visibility"]), np.float32)
            if "none_probability" in out:
                arrays["hist_none_probability"] = _to_numpy(
                    _single(out["none_probability"]), np.float32
                )
            arrays["hist_mask"] = _single(out["history_mask"]).detach().cpu().numpy().astype(bool)
            arrays["hist_memory"] = _to_numpy(_single(out["history_memory"]), np.float32)
        has_past = "hist_heatmaps" in arrays

        if plan is not None:
            plan_z0, plan_z, bridge_diagnostics = plan.result
            arrays["plan_z0"] = _to_numpy(_single(plan_z0), np.float32)
            arrays["plan_z"] = _to_numpy(_single(plan_z), np.float32)
            arrays["delta_z"] = _to_numpy(_single(bridge_diagnostics["delta_z"]), np.float32)
            arrays["delta_token_ratio"] = _to_numpy(
                _single(bridge_diagnostics["delta_token_ratio"]), np.float32
            )
        if future is not None:
            out = future.result
            arrays["fut_heatmaps_gated"] = _to_numpy(_single(out["future_heatmaps_gated"]), np.float16)
            arrays["fut_heatmaps"] = _to_numpy(_single(out["future_heatmaps"]), np.float16)
            arrays["fut_visibility_probability"] = _to_numpy(
                _single(out["future_visibility_probability"]), np.float32
            )

        resolved = _Resolved(plan=plan, projected=projected)
        if plan is not None and projected is not None:
            resolved.trajectory_path = "ppa"
            resolved.trajectory = projected.result
        elif warmup is not None:
            resolved.trajectory_path = "warmup"
            resolved.trajectory = warmup.result
        ppa_applied = response.get("ppa_applied")
        if (resolved.trajectory_path == "ppa") != (ppa_applied is True):
            warnings.append(
                f"trajectory_path={resolved.trajectory_path!r} but response ppa_applied={ppa_applied!r}"
            )

        checks: dict[str, Any] = {
            "calls": {slot: len(items) for slot, items in sorted(context.captures.items())}
        }
        if plan is not None:
            plan_z = plan.result[1]
            if projected is not None:
                checks["system1_input_is_plan_z"] = projected.arguments()["traj_cond"] is plan_z
            if future is not None:
                checks["future_input_is_plan_z"] = future.arguments()["plan_z"] is plan_z
            if past is not None and past.result is not None:
                checks["bridge_memory_is_history_memory"] = (
                    plan.kwargs.get("history_memory") is past.result["history_memory"]
                )
        if future is not None and past is not None:
            checks["future_past_output_is_history_output"] = (
                future.arguments()["past_output"] is past.result
            )
        for name, value in checks.items():
            if value is False:
                warnings.append(f"check {name} failed")

        response_actions = [int(a) for a in response.get("actions") or []]
        recomputed_actions = anti_deadlock = actions_match = None
        if resolved.trajectory is not None:
            resolved.postprocess = Postprocess.for_call(runtime, request, response)
            path_xy, recomputed_actions, anti_deadlock = resolved.postprocess(resolved.trajectory)
            resolved.selected_path_xy = path_xy
            arrays["trajectory_raw"] = _to_numpy(resolved.trajectory, np.float32)
            arrays["selected_path_xy"] = path_xy
            actions_match = recomputed_actions == response_actions
            if not actions_match:
                warnings.append(
                    f"recomputed actions {recomputed_actions} != response actions {response_actions}"
                )
        for warning in warnings:
            LOGGER.warning("EXP-19 %s call %s: %s", identity["ep_key"], identity["system2_call_index"], warning)

        record = {
            "schema": SCHEMA,
            **identity,
            **{key: request.get(key) for key in JOIN_KEYS},
            "request": context.request,
            "blob_names": blob_names,
            "response": json.loads(json.dumps(response)),
            "trajectory_path": resolved.trajectory_path,
            "has_past_output": has_past,
            "has_future_output": future is not None,
            "selected_path_xy": None if resolved.selected_path_xy is None else resolved.selected_path_xy.tolist(),
            "recomputed_actions": recomputed_actions,
            "recomputed_anti_deadlock": anti_deadlock,
            "actions_match": actions_match,
            "checks": checks,
            "trace_warnings": warnings,
        }
        return record, arrays, resolved

    def _diagnostics(self, response, resolved: _Resolved, arrays, identity) -> dict[str, Any]:
        result: dict[str, Any] = {
            "diagnostic_only": True,
            "enabled": self.diagnostics,
            "skipped_reason": None,
            "bridge_attention_available": False,
            "bridge_attention_check": None,
            "counterfactual_no_memory": None,
            "replay_same_plan": None,
            "errors": [],
        }
        if not self.diagnostics:
            result["skipped_reason"] = f"{DIAGNOSTICS_ENV}=0"
            return result
        if resolved.trajectory_path != "ppa":
            result["skipped_reason"] = "the bridge did not run on this call"
            return result

        plan_z0, plan_z, bridge_diagnostics = resolved.plan.result
        with torch.no_grad():
            try:
                chain = resolved.plan.arguments()["self"]
                recomputed = recompute_bridge_attention(
                    chain.bridge,
                    plan_z0,
                    resolved.plan.kwargs["history_memory"],
                    resolved.plan.kwargs["history_memory_mask"],
                    bridge_diagnostics["delta_z"],
                )
                if recomputed is not None:
                    weights, check = recomputed
                    result["bridge_attention_check"] = check
                    if check["reproduces_delta_z"] and check["samples_with_memory"] == [0]:
                        arrays["bridge_attention"] = _to_numpy(weights[0], np.float32)
                        result["bridge_attention_available"] = True
            except Exception:
                self._diagnostic_error(result, identity, "bridge_attention")

            seed = identity["per_call_seed"]
            arguments = resolved.projected.arguments()
            deployed_generator = arguments.get("generator")
            if seed is None or deployed_generator is None:
                result["skipped_reason"] = (
                    "no per_call_seed / deployed generator: the noise cannot be shared"
                )
                return result
            head = arguments["self"]
            # Everything the deployment passed except the Plan and the generator.
            passthrough = {
                name: value
                for name, value in arguments.items()
                if name not in ("self", "traj_cond", "generator")
            }

            def resample(plan: torch.Tensor) -> torch.Tensor:
                # A fresh generator on the deployed generator's device, seeded
                # like the server seeds it; the deployed one is never touched.
                generator = torch.Generator(device=deployed_generator.device)
                generator.manual_seed(int(seed))
                return head.get_trajectory_from_projected(plan, generator=generator, **passthrough)

            response_actions = [int(a) for a in response.get("actions") or []]
            deployed_end = resolved.selected_path_xy[-1]
            try:
                cf_trajectory = resample(plan_z0)
                cf_path, cf_actions, cf_anti_deadlock = resolved.postprocess(cf_trajectory)
                arrays["cf_trajectory_raw"] = _to_numpy(cf_trajectory, np.float32)
                arrays["cf_selected_path_xy"] = cf_path
                result["counterfactual_no_memory"] = {
                    "plan": "plan_z0",
                    "generator_seed": int(seed),
                    "actions": cf_actions,
                    "anti_deadlock": cf_anti_deadlock,
                    "actions_changed": cf_actions != response_actions,
                    "endpoint_shift_m": float(np.linalg.norm(cf_path[-1] - deployed_end)),
                    "selected_path_xy": cf_path.tolist(),
                }
            except Exception:
                self._diagnostic_error(result, identity, "counterfactual_no_memory")
            try:
                replay = resample(plan_z)
                replay_path, replay_actions, _anti_deadlock = resolved.postprocess(replay)
                deployed = resolved.trajectory
                result["replay_same_plan"] = {
                    "plan": "plan_z",
                    "generator_seed": int(seed),
                    "bitwise_equal": bool(torch.equal(replay, deployed)),
                    "max_abs_diff_raw": float((replay.float() - deployed.float()).abs().max()),
                    "actions": replay_actions,
                    "actions_equal": replay_actions == response_actions,
                    "endpoint_shift_m": float(np.linalg.norm(replay_path[-1] - deployed_end)),
                }
            except Exception:
                self._diagnostic_error(result, identity, "replay_same_plan")
        return result

    def _diagnostic_error(self, result: dict, identity: dict, name: str) -> None:
        trace_text = traceback.format_exc()
        result["errors"].append({"diagnostic": name, "traceback": trace_text})
        self.log_error(identity, f"diagnostic.{name}", trace_text)

    def _provenance(self) -> dict[str, Any]:
        return {
            **self.provenance,
            "wall_time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }

    def _write(self, ep_dir: Path, call_index: int, record: dict, arrays: dict) -> None:
        stem = f"call_{call_index:03d}"
        started = time.perf_counter()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        data = buffer.getvalue()
        compressed = time.perf_counter()
        record["npz"] = {
            "file": f"{stem}.npz",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        _atomic_write_bytes(ep_dir / f"{stem}.npz", data)
        # The json goes last, so it carries this call's own npz write time.
        timing = record.setdefault("timing_s", {})
        timing["npz_compress"] = compressed - started
        timing["npz_write"] = time.perf_counter() - compressed
        text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
        _atomic_write_bytes(ep_dir / f"{stem}.json", text.encode("utf-8"))

    def _write_fallback(self, ep_dir: Path, identity: dict, payload, response, context, trace_text: str) -> None:
        """Best effort: a minimal ``call_<idx>.json`` for a call whose full record failed; never raises."""
        try:
            request = context.request if context.request is not None else _plain_json(payload)
            record = {
                "schema": SCHEMA,
                **identity,
                **{key: request.get(key) for key in JOIN_KEYS},
                "request": request,
                "response": _plain_json(response),
                "trace_failed": True,
                "traceback": trace_text,
                "npz": None,
                "timing_s": dict(context.timing),
                "provenance": self._provenance(),
            }
            text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
            _atomic_write_bytes(ep_dir / f"call_{identity['system2_call_index']:03d}.json", text.encode("utf-8"))
        except Exception:
            LOGGER.exception(
                "EXP-19 could not write the fallback trace of %s call %s",
                identity.get("ep_key"),
                identity.get("system2_call_index"),
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def trace_config_from_env(environ) -> tuple[Path, bool]:
    raw_dir = str(environ.get(TRACE_DIR_ENV, "") or "").strip()
    if not raw_dir:
        raise ValueError(f"{TRACE_DIR_ENV} is not set; refusing to start an untraced EXP-19 server")
    raw_diagnostics = str(environ.get(DIAGNOSTICS_ENV, "1")).strip()
    if raw_diagnostics not in ("0", "1"):
        raise ValueError(f"{DIAGNOSTICS_ENV} must be 0 or 1, got {raw_diagnostics!r}")
    return Path(raw_dir).expanduser().resolve(), raw_diagnostics == "1"


def main() -> int:
    try:
        trace_dir, diagnostics = trace_config_from_env(os.environ)
    except ValueError as exc:
        print(f"[exp19-trace] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    install(trace_dir, diagnostics=diagnostics)
    print(
        f"[exp19-trace] tracing every plan_panoramic call to {trace_dir} "
        f"(diagnostics {'on' if diagnostics else 'off'}); server module {S.__file__}",
        file=sys.stderr,
        flush=True,
    )
    return S.main()


if __name__ == "__main__":
    raise SystemExit(main())
