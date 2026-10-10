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
    SUPPORTED_TRANSFORMERS,
    _patched_forward,
    install_qwen2_5_vl_vision_count,
    request_scope,
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
        assert stats() == {"vision_tower_calls": 1, "vision_tower_shapes": ["?"]}


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
