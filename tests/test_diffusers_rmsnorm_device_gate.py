"""The device gate on diffusers' RMSNorm.

Every test here restores ``RMSNorm.forward`` on the way out.  The gate is a class-level
mutation, and a leaked one would be exactly the cross-file failure it exists to stop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.action.nextdit.diffusers_npu_compat import (
    _arithmetic_rms_norm,
    install_diffusers_rmsnorm_device_gate,
    native_rmsnorm_forward,
)


@pytest.fixture()
def normalization():
    """diffusers' normalization module, with RMSNorm.forward put back afterwards.

    Starts each test from the unwrapped implementation, so the outcome does not depend
    on whether an earlier test in the session already imported ``nextdit_traj`` and
    installed the gate.
    """
    from diffusers.models import normalization as module

    installed = module.RMSNorm.forward
    native = native_rmsnorm_forward()
    if native is not None:
        module.RMSNorm.forward = native
    try:
        yield module
    finally:
        module.RMSNorm.forward = installed


def _rms_norm(dim: int = 8, **kwargs):
    from diffusers.models.normalization import RMSNorm

    return RMSNorm(dim, eps=1e-6, **kwargs)


def test_gate_is_not_installed_without_torch_npu(normalization, monkeypatch):
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: False)
    before = normalization.RMSNorm.forward

    assert install_diffusers_rmsnorm_device_gate() is False
    # Off Ascend diffusers already runs the arithmetic branch; wrapping it would only
    # change the traceback.
    assert normalization.RMSNorm.forward is before


def test_gate_keeps_the_kernel_for_npu_tensors_and_spares_everything_else(
    normalization, monkeypatch
):
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: True)
    native_calls = []

    def fake_native(self, hidden_states):
        native_calls.append(hidden_states)
        return "kernel"

    monkeypatch.setattr(normalization.RMSNorm, "forward", fake_native)
    assert install_diffusers_rmsnorm_device_gate() is True

    module = _rms_norm()
    npu_tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert normalization.RMSNorm.forward(module, npu_tensor) == "kernel"
    assert native_calls == [npu_tensor]

    # A CPU tensor never reaches npu_rms_norm, which has no CPU kernel at all.
    cpu_tensor = torch.randn(2, 3, 8)
    out = normalization.RMSNorm.forward(module, cpu_tensor)
    assert isinstance(out, torch.Tensor)
    assert native_calls == [npu_tensor]


def test_gate_is_idempotent(normalization, monkeypatch):
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: True)

    assert install_diffusers_rmsnorm_device_gate() is True
    gated = normalization.RMSNorm.forward
    assert install_diffusers_rmsnorm_device_gate() is False
    assert normalization.RMSNorm.forward is gated


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"elementwise_affine": False},
        {"bias": True},
    ],
)
def test_fallback_is_bitwise_equal_to_diffusers_own_branch(
    normalization, monkeypatch, kwargs
):
    """The mirror must not drift into a different normalisation than the kernel's.

    With the probe forced false, diffusers' unwrapped forward *is* the arithmetic
    branch, so it is the reference the fallback has to reproduce exactly.
    """
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: False)
    module = _rms_norm(**kwargs)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_()

    hidden_states = torch.randn(2, 5, 8)
    expected = normalization.RMSNorm.forward(module, hidden_states)
    actual = _arithmetic_rms_norm(module, hidden_states)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_importing_nextdit_installs_the_gate_when_torch_npu_is_present():
    """The stack is unusable on CPU until the gate is in, so importing it must do it."""
    import src.models.action.nextdit.nextdit_traj  # noqa: F401
    from diffusers.models import normalization as module

    gated = getattr(module.RMSNorm.forward, "_heatmapvln_device_gated", False)
    assert gated is bool(module.is_torch_npu_available())
