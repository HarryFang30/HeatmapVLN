"""diffusers' RMSNorm picks its kernel by library availability, not by tensor device.

``diffusers.models.normalization.RMSNorm.forward`` calls ``torch_npu.npu_rms_norm``
whenever ``is_torch_npu_available()`` is true.  It never looks at where the tensor
actually lives, and ``npu::npu_rms_norm`` has no CPU kernel, so on an Ascend host every
CPU-tensor norm raises::

    NotImplementedError: Could not run 'npu::npu_rms_norm' with arguments from the
    'CPU' backend

That is every CPU forward of the NextDiT stack, the test suite included -- the norms
are reached through ``LuminaRMSNormZero`` and ``LuminaLayerNormContinuous``, both of
which delegate to an ``RMSNorm`` instance.

The failure does not stay local.  When the raise happens inside a non-reentrant
``torch.utils.checkpoint`` backward (``nextdit_traj`` checkpoints its layers while
training) it escapes with the checkpoint's saved-tensor hook still installed, and the
*next* ``backward()`` anywhere in the process trips the leaked recompute and dies with
the same error -- in an unrelated test, with no NextDiT in it at all.  On the 910B that
turned 4 real failures in ``tests/test_heatmap_nextdit_control.py`` into 8, the extra 4
landing in ``tests/test_heatmap_raw_logit_loss.py``, which passes on its own.

So the branch is re-gated on the tensor's device.  An NPU tensor still takes the exact
path diffusers would have taken; only tensors that cannot use the kernel at all take
the arithmetic fallback, and for those the current behaviour is not a slower result but
no result.

Nothing is installed unless ``torch_npu`` is actually present: off Ascend,
``is_torch_npu_available()`` is already false and diffusers already runs the arithmetic
path, so the gate would be pure indirection.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

_GATE_MARKER = "_heatmapvln_device_gated"
_NATIVE_MARKER = "_heatmapvln_native_forward"


def _arithmetic_rms_norm(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """diffusers' own non-NPU branch, restated.

    Kept a line-for-line mirror of the ``else`` in
    ``diffusers.models.normalization.RMSNorm.forward`` so the fallback cannot drift
    into a *different* normalisation than the one the NPU kernel implements.
    """
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + module.eps)

    if module.weight is not None:
        # convert into half-precision if necessary
        if module.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(module.weight.dtype)
        hidden_states = hidden_states * module.weight
        if module.bias is not None:
            hidden_states = hidden_states + module.bias
    else:
        hidden_states = hidden_states.to(input_dtype)

    return hidden_states


def install_diffusers_rmsnorm_device_gate(
    logger: logging.Logger | None = None,
) -> bool:
    """Route non-NPU tensors in diffusers' RMSNorm to the arithmetic path.

    Returns True if the gate was installed by this call, False if it was unnecessary
    (no ``torch_npu``) or already in place.  Idempotent: safe to call from every module
    that needs it.
    """
    from diffusers.models import normalization

    rms_norm = getattr(normalization, "RMSNorm", None)
    if rms_norm is None:  # pragma: no cover - diffusers restructured
        return False

    if getattr(rms_norm.forward, _GATE_MARKER, False):
        return False

    # Off Ascend diffusers already takes the arithmetic branch; adding a wrapper there
    # would change nothing except the traceback.
    if not normalization.is_torch_npu_available():
        return False

    native_forward = rms_norm.forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type == "npu":
            return native_forward(self, hidden_states)
        return _arithmetic_rms_norm(self, hidden_states)

    setattr(forward, _GATE_MARKER, True)
    # The implementation this replaced, so a caller can tell what it is wrapping and
    # put it back; the gate is a class-level mutation and the only other handle on the
    # original is this closure.
    setattr(forward, _NATIVE_MARKER, native_forward)
    forward.__doc__ = native_forward.__doc__
    rms_norm.forward = forward

    if logger is not None:
        logger.info(
            "diffusers RMSNorm re-gated on tensor device: npu_rms_norm is now used "
            "only for NPU tensors"
        )
    return True


def native_rmsnorm_forward() -> Any | None:
    """The implementation the gate replaced, or None if no gate is installed."""
    from diffusers.models import normalization

    rms_norm = getattr(normalization, "RMSNorm", None)
    if rms_norm is None:  # pragma: no cover - diffusers restructured
        return None
    return getattr(rms_norm.forward, _NATIVE_MARKER, None)


__all__ = [
    "install_diffusers_rmsnorm_device_gate",
    "native_rmsnorm_forward",
]
