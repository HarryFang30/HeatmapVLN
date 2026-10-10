"""The vision tower's passes in one plan call: count them, and reuse the dead one.

A plan call can run Qwen2.5-VL's vision tower four times.  The server counts them, so
the number is in the response rather than in an estimate; measured on the 910B, every
steady-state ``trajectory+ppa`` call gives the same sequence of
``"<patches>x<images>"``::

    ['7056x9', '8620x10', '7056x9', '8620x10']

  pass 1  ``system2_turn1_generate``'s prefill        9 images  (history fronts + current)
  pass 2  ``system2_turn2_generate``'s prefill       10 images  (the same 9 + the look-down)
  pass 3  the PPA History Head                       9 images
  pass 4  ``generate_latents``                       10 images

Against the stage times that is about 9.2 s of an 11.0 s call inside the tower, about
4.6 s of it recomputing images the process has already encoded.

**Only pass 4 is reused here, and only because its side effects are provably dead.**
The tower is observed through forward hooks on ViT blocks 7/15/23/31, registered at load
by ``NativeSingleViewFeatureExtractor`` -- so returning a cached tensor skips every
block inside the tower, and with it those captures.  That is safe for pass 4 and for
nothing else, because ``_vit_captures`` is read in exactly three places, all inside
``native_single_view_feature_extractor.py`` (:170, :174, :202), all reached from
``extract_from_pixels``, which calls ``self.clear()`` (:146) before it runs the tower and
raises "visual hooks did not fire" (:170-172) if the captures are then missing.  A reader
therefore never sees a stale capture: it either sees the ones its own pass produced, or
it raises.  Pass 3 is that reader, and it is never served from the cache.

Reuse is bounded on both ends by the caller rather than inferred:

    with record_scope():   # around the System 2 generates, which produce the output
        ...
    with serve_scope():    # around generate_latents, the one call that may be served
        ...

Outside ``serve_scope`` nothing is ever served, so a future caller cannot pick up the
reuse by accident -- it has to ask.

A hit requires the *same* input, established by identity or by a byte-exact comparison,
never by a hash and never by shape: wrong embeddings returned for the right-looking
pixels would be the worst instance of the failure this deployment keeps meeting.  The
recorded tensors' ``_version`` counters are checked too, because the recorded input is
held by reference and is usually the very object the serving call passes -- so
``torch.equal`` alone would be comparing a tensor with itself and proving nothing.

``HEATMAPVLN_QWEN_VISION_REUSE=1`` enables the reuse (counting is always on, and
``HEATMAPVLN_QWEN_VISION_COUNT=0`` refuses the whole wrapper).
``HEATMAPVLN_QWEN_VISION_REUSE_VERIFY=1`` makes every hit also recompute the pass and
check ``torch.equal`` against the cached tensor: slower than no cache at all, and the
point is to prove bit-identity on the real machine rather than argue it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Optional

ENV_FLAG = "HEATMAPVLN_QWEN_VISION_COUNT"
REUSE_FLAG = "HEATMAPVLN_QWEN_VISION_REUSE"
VERIFY_FLAG = "HEATMAPVLN_QWEN_VISION_REUSE_VERIFY"
SUPPORTED_TRANSFORMERS = "4.51.0"

_STATE = threading.local()
LOGGER = logging.getLogger(__name__)


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip() == "1"


def _scope() -> Optional[dict[str, Any]]:
    return getattr(_STATE, "scope", None)


@contextlib.contextmanager
def request_scope():
    """One set of counters and one cache per request; nothing survives the reply."""
    previous = _scope()
    scope: dict[str, Any] = {
        "calls": 0,
        "shapes": [],
        "reuse": [],
        "entry": None,
        "recording": 0,
        "serving": 0,
    }
    _STATE.scope = scope
    try:
        yield scope
    finally:
        scope["entry"] = None
        _STATE.scope = previous


@contextlib.contextmanager
def record_scope():
    """Tower passes inside here may be remembered for a later ``serve_scope``."""
    scope = _scope()
    if scope is None:
        yield
        return
    scope["recording"] += 1
    try:
        yield
    finally:
        scope["recording"] -= 1


@contextlib.contextmanager
def serve_scope():
    """Tower passes inside here may be answered from ``record_scope``'s output."""
    scope = _scope()
    if scope is None:
        yield
        return
    scope["serving"] += 1
    try:
        yield
    finally:
        scope["serving"] -= 1


def stats() -> dict[str, Any]:
    """This request's tower passes, for the response's ``vision_tower`` field.

    ``vision_tower_shapes`` is one ``"<patches>x<images>"`` per pass in call order, which
    is what distinguishes "ran the same images again" from "ran a bigger set".
    ``vision_tower_reuse`` is one outcome per pass: ``record``, ``hit``, ``verified``,
    ``mismatch``, or ``miss:<reason>``.
    """
    scope = _scope()
    if scope is None:
        return {}
    return {
        "vision_tower_calls": scope["calls"],
        "vision_tower_shapes": list(scope["shapes"]),
        "vision_tower_reuse": list(scope["reuse"]),
    }


def _int_view(tensor: Any) -> Any:
    """A same-width integer view, so the comparison is on bytes.

    ``torch.equal`` on floats says two NaNs differ and that -0.0 equals 0.0.  Neither is
    what "the same input" should mean for a cache whose promise is bit-identity.
    """
    import torch

    widths = {
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }
    target = widths.get(tensor.dtype)
    return tensor.view(target) if target is not None else tensor


def _same_tensor(recorded: Any, incoming: Any) -> bool:
    import torch

    if recorded is incoming:
        return True
    if recorded is None or incoming is None:
        return False
    if (
        recorded.shape != incoming.shape
        or recorded.dtype != incoming.dtype
        or recorded.device != incoming.device
        or recorded.stride() != incoming.stride()
    ):
        return False
    try:
        return bool(torch.equal(_int_view(recorded), _int_view(incoming)))
    except RuntimeError:  # a dtype with no integer view of the same width
        return bool(torch.equal(recorded, incoming))


def _entry_unmutated(entry: dict[str, Any]) -> bool:
    """Nothing we hold by reference has been written in place since it was recorded."""
    return (
        entry["pixels"]._version == entry["pixels_version"]
        and entry["grid"]._version == entry["grid_version"]
        and entry["out"]._version == entry["out_version"]
    )


def _record(scope: dict[str, Any], tower: Any, pixels: Any, grid: Any, out: Any) -> None:
    scope["entry"] = {
        "tower": tower,
        "pixels": pixels,
        "pixels_version": pixels._version,
        "grid": grid,
        "grid_version": grid._version,
        "grid_host": tuple(grid.flatten().tolist()),
        # A private copy: the tensor handed to the recording caller must not be the one
        # a later hit returns, or that caller mutating it in place would corrupt this.
        "out": out.detach().clone(),
        "out_version": None,
    }
    scope["entry"]["out_version"] = scope["entry"]["out"]._version


def _lookup(scope: dict[str, Any], tower: Any, pixels: Any, grid: Any) -> tuple[Any, str]:
    entry = scope.get("entry")
    if entry is None:
        return None, "miss:nothing-recorded"
    if entry["tower"] is not tower:
        return None, "miss:other-tower"
    if not _entry_unmutated(entry):
        scope["entry"] = None
        return None, "miss:recorded-tensor-was-written"
    if tuple(grid.flatten().tolist()) != entry["grid_host"]:
        return None, "miss:different-grid"
    if not _same_tensor(entry["pixels"], pixels):
        return None, "miss:different-pixels"
    return entry, "hit"


def _patched_forward(original: Any):
    def forward(self, hidden_states: Any, grid_thw: Any) -> Any:
        import torch

        scope = _scope()
        if scope is None:  # outside a request: warm-up, startup checks, tests
            return original(self, hidden_states, grid_thw)
        scope["calls"] += 1
        try:
            scope["shapes"].append(f"{int(hidden_states.shape[0])}x{int(grid_thw.shape[0])}")
        except (AttributeError, IndexError, TypeError):
            scope["shapes"].append("?")

        if not _on(REUSE_FLAG):
            scope["reuse"].append("off")
            return original(self, hidden_states, grid_thw)

        if scope["serving"] > 0:
            entry, why = _lookup(scope, self, hidden_states, grid_thw)
            if entry is not None:
                if _on(VERIFY_FLAG):
                    fresh = original(self, hidden_states, grid_thw)
                    if bool(torch.equal(fresh, entry["out"])):
                        scope["reuse"].append("verified")
                        return entry["out"].clone()
                    scope["reuse"].append("mismatch")
                    LOGGER.error(
                        "vision tower reuse returned different bits than recomputing the "
                        "pass; serving the recomputed one and dropping the cache"
                    )
                    scope["entry"] = None
                    return fresh
                scope["reuse"].append("hit")
                return entry["out"].clone()
            scope["reuse"].append(why)
            return original(self, hidden_states, grid_thw)

        out = original(self, hidden_states, grid_thw)
        if scope["recording"] > 0:
            try:
                _record(scope, self, hidden_states, grid_thw, out)
                scope["reuse"].append("record")
            except (AttributeError, RuntimeError, TypeError) as exc:
                scope["entry"] = None
                scope["reuse"].append("record-failed")
                LOGGER.warning("vision tower pass not recorded: %s", exc)
        else:
            scope["reuse"].append("unscoped")
        return out

    forward.__doc__ = (
        "Qwen2_5_VisionTransformerPretrainedModel.forward, counted per request and, "
        "inside serve_scope, answered from record_scope's output when the input is the "
        f"same bytes (HeatmapVLN wrapper for transformers {SUPPORTED_TRANSFORMERS})."
    )
    forward._heatmapvln_patched_from = original
    return forward


def _tower_carries_adapters(tower: Any) -> Optional[str]:
    """A LoRA/PEFT layer anywhere in the tower makes the pixels an incomplete key.

    The same pixels through the same tower give different embeddings depending on
    whether an adapter is enabled, and the server does toggle adapters.  Nothing targets
    the vision tower today, so this is a refusal rather than a key extension: if that
    changes, the reuse must be re-derived, not silently applied.

    This looks for an adapter that is *attached*, not for the API that manages them:
    transformers' ``PreTrainedModel`` carries ``disable_adapters`` whether or not any
    adapter was ever loaded, so testing for the method refuses on every model -- which
    is how the first version of this check silently turned the reuse off everywhere.
    """
    # Model-level markers, set by transformers' PEFT integration when one is loaded.
    if getattr(tower, "_hf_peft_config_loaded", False):
        return "_hf_peft_config_loaded is True"
    if getattr(tower, "peft_config", None):
        return f"peft_config is {sorted(getattr(tower, 'peft_config'))}"
    # Injected layers: PEFT wraps the original module and keeps it as base_layer.
    walk = getattr(tower, "named_modules", None)
    if not callable(walk):
        # Not a module tree at all -- a test stub, or a shape this was not written
        # for.  Say so rather than raising from a precondition check, and rather
        # than reporting "no adapter" about something it could not look inside.
        return f"not an nn.Module ({type(tower).__name__}), cannot be checked"
    for name, module in walk():
        if "lora" in type(module).__name__.lower() or hasattr(module, "base_layer"):
            return f"{name or '<root>'} is {type(module).__name__}"
    return None


def install_qwen2_5_vl_vision_count(
    device_type: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Install the pass counter (and the reuse, if asked for).  Returns whether it went in.

    Refuses rather than guesses, for the same reason as the vision mask patch: it wraps
    someone else's forward, and the signature it passes through is pinned to the version
    it was written against.
    """
    log = logger or LOGGER
    requested = os.environ.get(ENV_FLAG, "").strip()
    if requested == "0":
        log.info("Qwen2.5-VL vision tower pass counter refused by %s=0", ENV_FLAG)
        return False
    if requested != "1" and device_type != "npu":
        return False

    import transformers

    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        log.warning(
            "Qwen2.5-VL vision tower pass counter not installed: it wraps transformers "
            "%s's tower forward and this is %s",
            SUPPORTED_TRANSFORMERS,
            transformers.__version__,
        )
        return False

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling

    target = getattr(modeling, "Qwen2_5_VisionTransformerPretrainedModel", None)
    if target is None:
        log.warning(
            "Qwen2.5-VL vision tower pass counter not installed: "
            "Qwen2_5_VisionTransformerPretrainedModel is gone"
        )
        return False
    if getattr(target.forward, "_heatmapvln_patched_from", None) is not None:
        return True

    import inspect

    try:
        params = list(inspect.signature(target.forward).parameters)
    except (TypeError, ValueError):  # pragma: no cover - a C-implemented forward
        params = []
    if params != ["self", "hidden_states", "grid_thw"]:
        log.warning(
            "Qwen2.5-VL vision tower pass counter not installed: the tower's forward "
            "takes %s, not (self, hidden_states, grid_thw), so this wrapper would drop "
            "arguments",
            params,
        )
        return False

    target.forward = _patched_forward(target.forward)
    log.info(
        "Qwen2.5-VL vision tower passes counted per request; reuse %s (%s), verify %s",
        "on" if _on(REUSE_FLAG) else "off",
        REUSE_FLAG,
        "on" if _on(VERIFY_FLAG) else "off",
    )
    return True


def refuse_reuse_on_adapted_tower(tower: Any, logger: Optional[logging.Logger] = None) -> bool:
    """Turn the reuse off for this process if the built tower carries adapters.

    Called once, after the model is loaded, because the tower does not exist at install
    time.  Returns whether the reuse is still on.
    """
    log = logger or LOGGER
    if not _on(REUSE_FLAG):
        return False
    found = _tower_carries_adapters(tower)
    if found is None:
        return True
    os.environ[REUSE_FLAG] = "0"
    log.warning(
        "Qwen2.5-VL vision tower reuse disabled: the tower carries an adapter (%s), so "
        "the pixels are not a complete cache key",
        found,
    )
    return False
