"""Read the vision window bounds once instead of once per window per layer.

transformers 4.51.0 builds Qwen2.5-VL's vision window mask like this
(``Qwen2_5_VLVisionSdpaAttention.forward``, modeling_qwen2_5_vl.py:300-302)::

    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

``cu_seqlens`` is a tensor on the accelerator, so every slice bound is a one-element
device-to-host copy, and every one of those is a full synchronisation: the host waits
for the device to drain before it can learn the number it needs to write the next
slice.  An NPU profile of one steady-state plan call on the 910B counted **16272** of
them at this line -- 96% of all 17001 scalar reads in the call -- costing 2.04 s of
device time out of a 5.4-5.9 s call, and leaving the device idle for about half of it.

Reading the whole vector once with ``.tolist()`` cannot change the mask: the same
integers become the same slice bounds.  Measured on the 910B at the deployed shapes
(12 views, 9408 patches, 147 windows), over 32 layers:

    device-tensor bounds   1026.5 ms
    host ints (.tolist())   188.9 ms      and torch.equal(old, new) is True

This is why it is worth doing on a platform where a synchronisation costs about
120 us; on CUDA the same 16272 reads cost roughly a tenth of that, which is most of
the reason the 910B looked two to four times slower per call.

Installed on NPU only.  Not because it would be wrong on CUDA -- the mask is provably
the same tensor -- but because the 4090's numbers are certified against the code as it
stands, and there is no reason to touch that path without a run to back it.  The 910B
has no A0 yet, so there is nothing there to invalidate.  Set
HEATMAPVLN_QWEN_VISION_MASK_PATCH=1 to install it anyway, or =0 to refuse.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

ENV_FLAG = "HEATMAPVLN_QWEN_VISION_MASK_PATCH"
SUPPORTED_TRANSFORMERS = "4.51.0"


def _patched_forward(original: Any):
    """``forward`` with the bounds read once, and otherwise the upstream body."""

    def forward(
        self,
        hidden_states: "Any",
        cu_seqlens: "Any",
        rotary_pos_emb: Optional["Any"] = None,
        position_embeddings: Optional[Tuple["Any", "Any"]] = None,
    ) -> "Any":
        import torch
        import torch.nn.functional as F
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # The one change: the bounds come back in a single copy rather than one per
        # slice.  Same integers, so the same mask -- see this module's docstring.
        bounds = cu_seqlens.tolist()
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(bounds)):
            attention_mask[..., bounds[i - 1] : bounds[i], bounds[i - 1] : bounds[i]] = True

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    forward.__doc__ = (
        "Qwen2_5_VLVisionSdpaAttention.forward with the window bounds read in one copy "
        f"instead of two per window (HeatmapVLN patch of transformers {SUPPORTED_TRANSFORMERS})."
    )
    forward._heatmapvln_patched_from = original
    return forward


def install_qwen2_5_vl_vision_mask_patch(
    device_type: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Install the patch when it applies.  Returns whether it was installed.

    Refuses rather than guesses: a different transformers version may have changed the
    body this copy is based on, and silently running a stale copy of someone else's
    forward is exactly the kind of thing this deployment keeps getting bitten by.
    """
    log = logger or logging.getLogger(__name__)
    requested = os.environ.get(ENV_FLAG, "").strip()
    if requested == "0":
        log.info("Qwen2.5-VL vision mask patch refused by %s=0", ENV_FLAG)
        return False
    if requested != "1" and device_type != "npu":
        return False

    import transformers

    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        log.warning(
            "Qwen2.5-VL vision mask patch not installed: it is a copy of transformers %s's "
            "forward and this is %s",
            SUPPORTED_TRANSFORMERS,
            transformers.__version__,
        )
        return False

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling

    target = getattr(modeling, "Qwen2_5_VLVisionSdpaAttention", None)
    if target is None:
        log.warning("Qwen2.5-VL vision mask patch not installed: Qwen2_5_VLVisionSdpaAttention is gone")
        return False
    if getattr(target.forward, "_heatmapvln_patched_from", None) is not None:
        return True

    source = __import__("inspect").getsource(target.forward)
    # The body this copy was taken from.  If upstream changed the mask construction,
    # the patch must be re-derived rather than applied blind.
    if "cu_seqlens[i - 1] : cu_seqlens[i]" not in source:
        log.warning(
            "Qwen2.5-VL vision mask patch not installed: the upstream forward no longer "
            "builds the mask from cu_seqlens elements, so this copy is stale"
        )
        return False

    target.forward = _patched_forward(target.forward)
    log.info(
        "Qwen2.5-VL vision mask patch installed: window bounds read once per attention "
        "call instead of twice per window (16272 device-to-host scalar reads per plan "
        "call on the 910B, 2.04 s)"
    )
    return True
