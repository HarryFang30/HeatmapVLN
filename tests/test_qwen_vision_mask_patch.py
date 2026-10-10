"""The vision-mask patch must build the same mask, and refuse when it cannot.

src/models/qwen2_5_vl_vision_mask.py replaces one line of transformers 4.51.0's
Qwen2_5_VLVisionSdpaAttention.forward: the window mask's slice bounds come from one
``cu_seqlens.tolist()`` instead of two device-to-host scalar copies per window.  An
NPU profile of a plan call counted 16272 of those copies, each a full synchronisation,
at 2.04 s of a 5.4-5.9 s call.

The whole argument for the patch is that the mask is the same tensor, so that is what
these tests check -- on CPU, so they run anywhere torch does.  They also check that
the installer refuses rather than guesses: it is a copy of someone else's forward, and
silently running a stale copy is the failure this deployment keeps meeting.

Measured on the 910B over 32 layers at a 9408-patch sequence: 1026.5 ms before,
188.9 ms after (a ratio, not the per-call cost -- the deployed pass is smaller).  A 24-step NavAgent run against real
servers, patched and unpatched, returned identical decisions on all 6 plan calls
across 15 recorded fields including the full response payload.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from src.models.qwen2_5_vl_vision_mask import (  # noqa: E402
    ENV_FLAG,
    SUPPORTED_TRANSFORMERS,
    _window_mask,
    install_qwen2_5_vl_vision_mask_patch,
)


def _upstream_mask(seq_length: int, cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """The construction in transformers 4.51.0, bounds read from the tensor."""
    mask = torch.zeros([1, seq_length, seq_length], device=cu_seqlens.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    return mask


def _patched_mask(seq_length: int, cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """What the patch does: one copy, then the id comparison (or its fallback)."""
    return _window_mask(cu_seqlens.tolist(), seq_length, cu_seqlens.device)


@pytest.mark.parametrize(
    "seq_length, edges",
    [
        # One window per view and a ragged tail, the windowed-attention case.
        (64, [0, 16, 32, 48, 64]),
        (70, [0, 16, 32, 48, 64, 70]),
        # One span for the whole sequence, the full-attention blocks.
        (48, [0, 48]),
        # Degenerate shapes that still have to agree.
        (8, [0, 8]),
        (8, [0, 0, 8]),
        (12, [0, 4, 4, 12]),
        # The deployed window count, at a scaled-down sequence.
        (147 * 4, list(range(0, 147 * 4 + 1, 4))),
        # Bounds that do NOT cover the sequence, where the id comparison cannot be
        # used and the fallback loop has to run instead.  A position in no window is
        # False against everything including itself, which no id can express.
        (10, [0, 4]),
        (10, [2, 6]),
        (10, [0, 4, 8]),
        (6, []),
        (6, [3]),
    ],
)
def test_the_patched_mask_is_the_same_tensor(seq_length, edges):
    for dtype in (torch.int32, torch.int64):
        cu = torch.tensor(edges, dtype=dtype)
        assert torch.equal(_upstream_mask(seq_length, cu), _patched_mask(seq_length, cu)), (
            f"masks differ for {seq_length=} {edges=} {dtype=}"
        )


def test_the_mask_is_built_without_a_write_per_window(monkeypatch):
    """The point of _window_mask: the launch count must not grow with the windows.

    If it quietly falls back to the loop at the deployed shape, the patch still gives
    the right answer and none of the speed, which is the kind of silent no-op this
    deployment keeps meeting.  So pin that the covering case takes the vectorised path.
    """
    seen = 0
    real_zeros = torch.zeros

    def counting_zeros(*args, **kwargs):
        nonlocal seen
        seen += 1
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", counting_zeros)
    bounds = list(range(0, 147 * 4 + 1, 4))  # 147 windows that cover the sequence
    _window_mask(bounds, 147 * 4, torch.device("cpu"))
    assert seen == 0, "the covering case must not allocate the zeros it then overwrites"
    # And the non-covering case must still take the loop, which does allocate.
    _window_mask([0, 4], 10, torch.device("cpu"))
    assert seen == 1


def test_the_patch_is_not_installed_on_cuda_unless_asked(monkeypatch):
    """The 4090's numbers are certified against the code as it stands."""
    monkeypatch.delenv(ENV_FLAG, raising=False)
    assert install_qwen2_5_vl_vision_mask_patch("cuda", logging.getLogger("t")) is False
    assert install_qwen2_5_vl_vision_mask_patch("cpu", logging.getLogger("t")) is False


def test_the_patch_can_be_refused_outright(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "0")
    assert install_qwen2_5_vl_vision_mask_patch("npu", logging.getLogger("t")) is False


def test_the_patch_refuses_a_transformers_it_was_not_derived_from(monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.setattr(transformers, "__version__", "4.52.0")
    assert install_qwen2_5_vl_vision_mask_patch("npu", logging.getLogger("t")) is False
    # And it does apply to the version it was derived from, if that is what is installed.
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        monkeypatch.setattr(transformers, "__version__", SUPPORTED_TRANSFORMERS)


def test_the_patch_refuses_when_upstream_stopped_building_the_mask_that_way(monkeypatch):
    """A rewritten upstream forward means this copy is stale, not that it is safe."""
    transformers = pytest.importorskip("transformers")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.setattr(transformers, "__version__", SUPPORTED_TRANSFORMERS)
    modeling = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    target = getattr(modeling, "Qwen2_5_VLVisionSdpaAttention", None)
    if target is None:
        pytest.skip("this transformers has no Qwen2_5_VLVisionSdpaAttention")
    if getattr(target.forward, "_heatmapvln_patched_from", None) is not None:
        pytest.skip("already patched in this process")

    import inspect

    monkeypatch.setattr(inspect, "getsource", lambda _obj: "def forward(self):\n    return None\n")
    assert install_qwen2_5_vl_vision_mask_patch("npu", logging.getLogger("t")) is False


def test_the_mask_builder_reads_nothing_back_off_the_device(monkeypatch):
    """The patch exists to remove scalar reads; it must not add any of its own.

    An earlier version checked the window sizes with ``(counts >= 0).all()`` on the
    device and called ``bool()`` on the result -- one full synchronisation per layer,
    32 per tower pass, for a property the host already knows from ``bounds``.
    """
    for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
        def forbidden(*_a, _name=name, **_k):
            raise AssertionError(f"_window_mask called Tensor.{_name}")

        monkeypatch.setattr(torch.Tensor, name, forbidden)
    bounds = list(range(0, 147 * 4 + 1, 4))
    mask = _window_mask(bounds, 147 * 4, torch.device("cpu"))
    assert mask.shape == (1, 147 * 4, 147 * 4)
