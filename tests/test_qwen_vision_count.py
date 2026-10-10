"""The vision-tower pass counter must count, cost nothing, and change nothing.

src/models/qwen2_5_vl_vision_count.py wraps Qwen2.5-VL's vision tower so one plan call
can report how many times it ran it and on what shapes.  That number decides which of
the remaining latency fixes is worth doing, and every estimate of it so far has been
arithmetic over a profile that -- by its own kernel counts -- was taken on a call that
ran the tower once.  So the counter has to be trustworthy in three ways:

  * it counts every pass, and reports the shapes that tell repeated sets apart;
  * it does not read device data, because on this platform one scalar read off the
    accelerator is a full synchronisation and the instrument would distort what it
    measures;
  * it passes the tower's arguments through untouched, and refuses to install at all
    when the signature it was written against has changed -- a wrapper that silently
    dropped an argument would corrupt every pass it counted.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from src.models.qwen2_5_vl_vision_count import (  # noqa: E402
    ENV_FLAG,
    REUSE_FLAG,
    SUPPORTED_TRANSFORMERS,
    VERIFY_FLAG,
    _patched_forward,
    install_qwen2_5_vl_vision_count,
    record_scope,
    refuse_reuse_on_adapted_tower,
    request_scope,
    serve_scope,
    stats,
)

GRID = torch.tensor([[1, 2, 4]])


class _Tower:
    """Stands in for the tower: records what it was called with."""

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def forward(self, hidden_states, grid_thw):
        self.seen.append((hidden_states, grid_thw))
        return hidden_states.sum(dim=-1, keepdim=True)

    @property
    def seen_count(self) -> int:
        """How many passes actually reached the tower, as opposed to being counted."""
        return len(self.seen)


def _wired():
    tower = _Tower()
    forward = _patched_forward(_Tower.forward)
    return tower, (lambda px, grid: forward(tower, px, grid))


def _pixels(seed=0, n=8):
    return torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + seed


# ------------------------------------------------------------------------- counting
def test_every_pass_is_counted_with_its_shape():
    tower, call = _wired()
    with request_scope():
        call(_pixels(n=7056), GRID)
        call(_pixels(n=8620), torch.tensor([[1, 2, 4], [1, 2, 4]]))
        call(_pixels(n=7056), GRID)
        assert stats() == {
            "vision_tower_calls": 3,
            # The repeated set is visible as a repeated shape, which is what makes
            # "ran the same images again" distinguishable from "ran a bigger set".
            "vision_tower_shapes": ["7056x1", "8620x2", "7056x1"],
            # Reuse is off here, so every pass says so rather than going unexplained.
            "vision_tower_reuse": ["off", "off", "off"],
        }
    assert len(tower.seen) == 3


def test_the_counter_does_not_touch_the_result_or_the_arguments():
    tower, call = _wired()
    px, grid = _pixels(), GRID
    with request_scope():
        out = call(px, grid)
    assert tower.seen == [(px, grid)], "the wrapper must pass both arguments straight through"
    assert tower.seen[0][0] is px and tower.seen[0][1] is grid
    assert torch.equal(out, px.sum(dim=-1, keepdim=True))


def test_the_counter_reads_no_device_data(monkeypatch):
    """A scalar read off the accelerator is a full sync; the instrument must not do one."""
    for name in ("item", "tolist", "cpu", "numpy", "equal"):
        target = torch if name == "equal" else torch.Tensor

        def forbidden(*_a, _name=name, **_k):
            raise AssertionError(f"the counter called torch.{_name}")

        monkeypatch.setattr(target, name, forbidden)
    _, call = _wired()
    with request_scope():
        call(_pixels(), GRID)
        assert stats()["vision_tower_calls"] == 1


def test_an_unreadable_shape_is_recorded_rather_than_raised():
    """The counter is instrumentation: it must never be the thing that fails a call."""
    forward = _patched_forward(lambda self, hidden_states, grid_thw: "answer")
    with request_scope():
        assert forward(object(), object(), object()) == "answer"
        assert stats() == {
            "vision_tower_calls": 1,
            "vision_tower_shapes": ["?"],
            "vision_tower_reuse": ["off"],
        }


# ---------------------------------------------------------------------------- scope
def test_counts_do_not_leak_between_requests():
    _, call = _wired()
    with request_scope():
        call(_pixels(), GRID)
        call(_pixels(), GRID)
        assert stats()["vision_tower_calls"] == 2
    with request_scope():
        call(_pixels(), GRID)
        assert stats()["vision_tower_calls"] == 1
    assert stats() == {}


def test_outside_a_request_it_is_a_pass_through():
    tower, call = _wired()
    assert torch.equal(call(_pixels(), GRID), _pixels().sum(dim=-1, keepdim=True))
    assert len(tower.seen) == 1
    assert stats() == {}


def test_a_nested_scope_restores_the_outer_one():
    _, call = _wired()
    with request_scope():
        call(_pixels(), GRID)
        with request_scope():
            call(_pixels(), GRID)
            assert stats()["vision_tower_calls"] == 1
        assert stats()["vision_tower_calls"] == 1


# ------------------------------------------------------------------------- refusals
def test_it_refuses_a_transformers_it_was_not_written_against(monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.setattr(transformers, "__version__", "4.52.0")
    assert install_qwen2_5_vl_vision_count("npu", logging.getLogger("t")) is False


def test_it_can_be_refused_outright(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "0")
    assert install_qwen2_5_vl_vision_count("npu", logging.getLogger("t")) is False


def test_it_is_not_installed_off_npu_unless_asked(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    assert install_qwen2_5_vl_vision_count("cuda", logging.getLogger("t")) is False
    assert install_qwen2_5_vl_vision_count("cpu", logging.getLogger("t")) is False


def test_it_refuses_a_forward_whose_arguments_changed(monkeypatch):
    """Dropping an argument silently would corrupt every pass, so refuse instead."""
    transformers = pytest.importorskip("transformers")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.setattr(transformers, "__version__", SUPPORTED_TRANSFORMERS)
    modeling = pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    target = getattr(modeling, "Qwen2_5_VisionTransformerPretrainedModel", None)
    if target is None:
        pytest.skip("this transformers has no Qwen2_5_VisionTransformerPretrainedModel")
    if getattr(target.forward, "_heatmapvln_patched_from", None) is not None:
        pytest.skip("already patched in this process")

    def extra(self, hidden_states, grid_thw, something_new=None):
        return hidden_states

    monkeypatch.setattr(target, "forward", extra)
    assert install_qwen2_5_vl_vision_count("npu", logging.getLogger("t")) is False


# ============================================================ the pass-4 reuse
# Only generate_latents' tower pass is ever served, and only because its ViT-block
# captures are dead.  These tests push on the two things that would make that unsafe:
# serving outside the one place that asked for it, and calling two different inputs
# "the same".


def _reuse(monkeypatch, *, verify=False):
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(REUSE_FLAG, "1")
    if verify:
        monkeypatch.setenv(VERIFY_FLAG, "1")
    else:
        monkeypatch.delenv(VERIFY_FLAG, raising=False)
    tower = _Tower()
    forward = _patched_forward(_Tower.forward)
    return tower, (lambda px, grid: forward(tower, px, grid))


def test_the_recorded_pass_is_served_to_the_serving_call(monkeypatch):
    """The deployed shape: turn 1 records, turn 2 records, generate_latents is served."""
    tower, call = _reuse(monkeypatch)
    nine, ten = _pixels(n=7056), _pixels(n=8620, seed=1)
    with request_scope():
        with record_scope():
            call(nine, GRID)          # pass 1
            call(ten, GRID)           # pass 2, overwrites the entry
        call(nine, GRID)              # pass 3, unscoped: must really run
        with serve_scope():
            served = call(ten, GRID)  # pass 4, the same input as pass 2
        recorded = stats()
    assert tower.seen_count == 3, "pass 4 must not have reached the tower"
    assert torch.equal(served, ten.sum(dim=-1, keepdim=True))
    assert recorded["vision_tower_reuse"] == ["record", "record", "unscoped", "hit"]
    assert recorded["vision_tower_calls"] == 4


def test_pass_three_is_never_served(monkeypatch):
    """Its captures ARE read, so it has to run even when the pixels match a record."""
    tower, call = _reuse(monkeypatch)
    nine = _pixels(n=7056)
    with request_scope():
        with record_scope():
            call(nine, GRID)
        call(nine, GRID)  # same pixels, but outside serve_scope
        assert tower.seen_count == 2
        assert stats()["vision_tower_reuse"] == ["record", "unscoped"]


def test_nothing_is_served_before_anything_is_recorded(monkeypatch):
    tower, call = _reuse(monkeypatch)
    with request_scope():
        with serve_scope():
            call(_pixels(), GRID)
        assert tower.seen_count == 1
        assert stats()["vision_tower_reuse"] == ["miss:nothing-recorded"]


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda px, grid: (px.clone().add_(1e-3), grid), "miss:different-pixels"),
        (lambda px, grid: (px, torch.tensor([[1, 4, 2]])), "miss:different-grid"),
        (lambda px, grid: (_pixels(n=9), grid), "miss:different-pixels"),
    ],
)
def test_a_different_input_is_not_served(monkeypatch, mutate, reason):
    tower, call = _reuse(monkeypatch)
    px = _pixels()
    with request_scope():
        with record_scope():
            call(px, GRID)
        with serve_scope():
            call(*mutate(px, GRID))
        assert tower.seen_count == 2
        assert stats()["vision_tower_reuse"][-1] == reason


def test_writing_the_recorded_input_in_place_drops_the_cache(monkeypatch):
    """The recorded input is held by reference, and on the deployed path the serving
    call passes the very same object -- so torch.equal alone would compare a tensor
    with itself and prove nothing.  The version counters are what close that."""
    tower, call = _reuse(monkeypatch)
    px = _pixels()
    with request_scope():
        with record_scope():
            call(px, GRID)
        px.add_(1.0)  # someone writes the input the cache is keyed on
        with serve_scope():
            call(px, GRID)
        assert tower.seen_count == 2, "a mutated key must not be served"
        assert stats()["vision_tower_reuse"][-1] == "miss:recorded-tensor-was-written"


def test_writing_the_recorded_output_in_place_drops_the_cache(monkeypatch):
    tower, call = _reuse(monkeypatch)
    px = _pixels()
    with request_scope():
        with record_scope():
            out = call(px, GRID)
        out.add_(1.0)  # the recording caller mutates what it was handed
        with serve_scope():
            served = call(px, GRID)
        # The cache kept its own copy, so the mutation cannot have reached it.
        assert torch.equal(served, px.sum(dim=-1, keepdim=True))
        assert stats()["vision_tower_reuse"][-1] == "hit"


def test_the_comparison_is_on_bytes_not_on_float_equality(monkeypatch):
    """NaN must equal itself here, and -0.0 must not equal 0.0: the promise is bits."""
    _, call = _reuse(monkeypatch)
    nan = torch.full((4, 4), float("nan"))
    with request_scope():
        with record_scope():
            call(nan, GRID)
        with serve_scope():
            call(nan.clone(), GRID)
        assert stats()["vision_tower_reuse"][-1] == "hit", "equal bits must hit"
    zeros, negzeros = torch.zeros(4, 4), torch.full((4, 4), -0.0)
    with request_scope():
        with record_scope():
            call(zeros, GRID)
        with serve_scope():
            call(negzeros, GRID)
        assert stats()["vision_tower_reuse"][-1] == "miss:different-pixels"


def test_the_cache_does_not_outlive_the_request(monkeypatch):
    tower, call = _reuse(monkeypatch)
    px = _pixels()
    with request_scope():
        with record_scope():
            call(px, GRID)
    with request_scope():
        with serve_scope():
            call(px, GRID)
        assert tower.seen_count == 2, "a hit across requests means the cache outlived one"


def test_reuse_is_off_unless_asked_for(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.delenv(REUSE_FLAG, raising=False)
    tower = _Tower()
    forward = _patched_forward(_Tower.forward)
    px = _pixels()
    with request_scope():
        with record_scope():
            forward(tower, px, GRID)
        with serve_scope():
            forward(tower, px, GRID)
        assert tower.seen_count == 2
        assert stats()["vision_tower_reuse"] == ["off", "off"]


def test_verify_recomputes_and_reports_agreement(monkeypatch):
    tower, call = _reuse(monkeypatch, verify=True)
    px = _pixels()
    with request_scope():
        with record_scope():
            call(px, GRID)
        with serve_scope():
            call(px, GRID)
        assert tower.seen_count == 2, "verify must actually recompute"
        assert stats()["vision_tower_reuse"][-1] == "verified"


def test_verify_serves_the_recomputed_pass_when_the_bits_differ(monkeypatch, caplog):
    """If the tower were not deterministic, the request must not be served from cache."""
    drift = {"n": 0}

    def drifting(self, hidden_states, grid_thw):
        drift["n"] += 1
        return hidden_states.sum(dim=-1, keepdim=True) + drift["n"]

    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(REUSE_FLAG, "1")
    monkeypatch.setenv(VERIFY_FLAG, "1")
    forward = _patched_forward(drifting)
    tower, px = _Tower(), _pixels()
    with caplog.at_level(logging.ERROR), request_scope():
        with record_scope():
            forward(tower, px, GRID)      # n=1, caches sum+1
        with serve_scope():
            served = forward(tower, px, GRID)  # recomputes n=2, differs
        recorded = stats()
    assert recorded["vision_tower_reuse"][-1] == "mismatch"
    assert torch.equal(served, px.sum(dim=-1, keepdim=True) + 2)
    assert "different bits" in caplog.text


# --------------------------------------------------- the adapter precondition
class _LoraLinear(torch.nn.Module):
    """What PEFT injects: a wrapper that keeps the original module as base_layer."""

    def __init__(self) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(2, 2)


def test_a_tower_with_no_adapter_keeps_the_reuse(monkeypatch):
    """The check that matters most, because it is the one that fired wrongly.

    transformers' PreTrainedModel carries disable_adapters() whether or not an adapter
    was ever loaded, so an earlier version of this check tested for the method and
    refused on every real model -- the reuse was silently off on the whole deployment,
    and only the log line gave it away.
    """
    monkeypatch.setenv(REUSE_FLAG, "1")
    plain = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    plain.disable_adapters = lambda: None  # the API, with nothing attached
    plain.active_adapters = lambda: []
    assert refuse_reuse_on_adapted_tower(plain, logging.getLogger("t")) is True


@pytest.mark.parametrize("attach", ["injected-layer", "peft-config", "peft-marker"])
def test_reuse_is_refused_on_a_tower_carrying_an_adapter(monkeypatch, attach):
    """The same pixels give different embeddings with an adapter on, so the pixels are
    not a complete key.  Nothing targets the tower today; refuse if that changes."""
    monkeypatch.setenv(REUSE_FLAG, "1")
    adapted = torch.nn.Sequential(torch.nn.Linear(2, 2))
    if attach == "injected-layer":
        adapted = torch.nn.Sequential(_LoraLinear())
    elif attach == "peft-config":
        adapted.peft_config = {"default": object()}
    else:
        adapted._hf_peft_config_loaded = True
    assert refuse_reuse_on_adapted_tower(adapted, logging.getLogger("t")) is False
    import os as _os

    assert _os.environ[REUSE_FLAG] == "0", "the refusal has to actually turn it off"


def test_the_adapter_check_says_nothing_when_the_reuse_is_already_off(monkeypatch):
    monkeypatch.delenv(REUSE_FLAG, raising=False)
    assert refuse_reuse_on_adapted_tower(torch.nn.Sequential(_LoraLinear())) is False


def test_an_unwalkable_tower_refuses_rather_than_raising(monkeypatch):
    """The check runs as a precondition after model load; it must not be what fails.

    And it must not report "no adapter" about something it could not look inside --
    that would be a silent claim of safety, which is the failure mode this whole
    module is written against.
    """
    monkeypatch.setenv(REUSE_FLAG, "1")
    from types import SimpleNamespace

    assert refuse_reuse_on_adapted_tower(SimpleNamespace(), logging.getLogger("t")) is False
    import os as _os

    assert _os.environ[REUSE_FLAG] == "0"
