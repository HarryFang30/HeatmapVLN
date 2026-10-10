"""Count the vision-tower passes in one plan call, and what they were run on.

A plan call can run Qwen2.5-VL's vision tower more than once.  The sites, for a call
that reaches System 1:

  1. ``system2_turn1_generate``'s prefill -- the history fronts plus the current front.
  2. ``system2_turn2_generate``'s prefill, when turn 1 asks to look down -- the same
     images again, plus the look-down.
  3. the PPA History Head (``_build_ppa_past_output``) -- the image processor run again
     over history, any black padding, and the current front.
  4. ``generate_latents`` -- handed ``inputs["pixel_values"]``, the tensor turn 2 built.

How many of those actually fire, and on what shapes, decides whether any of the obvious
fixes is worth doing -- and every estimate of it so far has been arithmetic over a
profile that, by its own kernel counts, was taken on a call that ran the tower *once*.
So this counts them, per request, and reports the counts in the response's
``timing_ms`` next to the stage times.  Nothing here changes what the server computes.

**Why this does not also reuse a pass.**  Passes 1 and 4 can be given byte-identical
input, and the tower is deterministic at eval, so returning a cached tensor would be
bit-identical -- as a value.  It would not be side-effect-identical: the PPA head reads
the tower through forward hooks on ViT blocks 7/15/23/31
(``native_single_view_feature_extractor.py`` registers them at load and raises "visual
hooks did not fire" if they do not), so skipping a tower call skips the captures of
every block inside it.  Pass 3 would therefore fail loudly, and pass 4 would leave the
captures holding pass 3's values for anything that reads them later.  Reuse has to be
done at the call site, by the code that knows which captures its caller needs, not
behind the tower's back.  That is a separate change; this module exists so it can be
decided on measurements instead of on an estimate.

Per-image reuse -- encoding each image once and splicing -- is a different change
again, and is *not* bit-identical: the tower's attention is block-diagonal per image,
so the values are the same mathematically, but the rows are reduced over a sequence of
a different length and the last bits move.  It needs its own certification run.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Optional

ENV_FLAG = "HEATMAPVLN_QWEN_VISION_COUNT"
SUPPORTED_TRANSFORMERS = "4.51.0"

_STATE = threading.local()
LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def request_scope():
    """One set of counters per request; nothing survives the reply."""
    previous = getattr(_STATE, "scope", None)
    scope: dict[str, Any] = {"calls": 0, "shapes": []}
    _STATE.scope = scope
    try:
        yield scope
    finally:
        _STATE.scope = previous


def stats() -> dict[str, Any]:
    """This request's tower passes, for the response's ``timing_ms``.

    ``vision_tower_shapes`` is one ``"<patches>x<images>"`` per pass in call order,
    which is what distinguishes "ran the same images again" from "ran a bigger set".
    """
    scope = getattr(_STATE, "scope", None)
    if scope is None:
        return {}
    return {
        "vision_tower_calls": scope["calls"],
        "vision_tower_shapes": list(scope["shapes"]),
    }


def _patched_forward(original: Any):
    def forward(self, hidden_states: Any, grid_thw: Any) -> Any:
        scope = getattr(_STATE, "scope", None)
        if scope is not None:
            scope["calls"] += 1
            # Shapes only: no device reads, so counting cannot itself cost a
            # synchronisation on a platform where one costs about 120 us.
            try:
                patches = int(hidden_states.shape[0])
                images = int(grid_thw.shape[0])
                scope["shapes"].append(f"{patches}x{images}")
            except (AttributeError, IndexError, TypeError):
                scope["shapes"].append("?")
        return original(self, hidden_states, grid_thw)

    forward.__doc__ = (
        "Qwen2_5_VisionTransformerPretrainedModel.forward, counted per request "
        f"(HeatmapVLN wrapper for transformers {SUPPORTED_TRANSFORMERS})."
    )
    forward._heatmapvln_patched_from = original
    return forward


def install_qwen2_5_vl_vision_count(
    device_type: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Install the pass counter.  Returns whether it went in.

    Refuses rather than guesses, for the same reason as the vision mask patch: it wraps
    someone else's forward, and the signature it passes through is pinned to the
    version it was written against.
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
    log.info("Qwen2.5-VL vision tower passes counted per request (%s=0 to refuse)", ENV_FLAG)
    return True
