#!/usr/bin/env python3
"""EXP-23 component C: LoRA SFT of the deployed System2 on on-policy DAgger records.

The adapter is trained on the exact inputs the server builds
(``src/data/s2_sft_dataset.py``).  It is attached to the same
``Qwen2_5_VLForConditionalGeneration`` the server loads, so the server can wrap
its backbone with PEFT and load the adapter by name.  It touches only the
language model (q/k/v/o/gate/up/down in all layers); the vision tower and
everything System1 reads stay frozen.  System1's Z0 is taken with the adapter
disabled, so its inputs do not move.

Modes::

    # train (torchrun; one process per card)
    torchrun --nproc_per_node 8 scripts/train_system2_lora.py --device npu \\
        --record-root R1 [--record-root R2 ...] --label-root L1 [...] \\
        --model-path InternNav_Model --output-dir OUT

    # gate G0: adapters off, replay N recorded calls through the server's greedy
    # two-turn decode, report the exact-match rate
    torchrun ... scripts/train_system2_lora.py --parity-check 2000 --record-root R1 --output-dir OUT

``--max-steps 0`` saves the freshly initialised adapter (B = 0) and stops.  This
is the no-op adapter the server's byte-identity test needs.

``--kl-beta B`` (EXP-34) adds B * KL(p_base || p_theta) on the supervised tokens of
samples tagged ``kl: true``, p_base being the same model with the adapter off;
``--keep-stream`` never deletes stream shards, so a resume reads them again.

``--objective dpo`` (EXP-37) trains on a pair stream (``pairs: true``): per pair,
s(y) = the mean log-prob of y's supervised tokens, and the loss is AVERT's
-log sigmoid(beta * [(s_theta(c) - s_ref(c)) - (s_theta(r) - s_ref(r))]) - lambda * s_theta(c),
the reference being the same model with the adapter off; a step is the mean over
its global batch of pairs.  No DDP wrapper there: the LoRA gradients are summed
over ranks by hand once per optimizer step (:func:`allreduce_mean_grads`).

Outputs (``--output-dir``): ``adapter/`` (``adapter_config.json`` +
``adapter_model.safetensors``), ``manifest.json``, ``train_log.jsonl``, and
``checkpoints/step_*/`` (adapter + optimizer state, for ``--resume``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_AGREE_WEIGHT,
    DEFAULT_DISAGREE_WEIGHT,
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_MAX_CALLS_PER_EPISODE,
    DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    IGNORE_INDEX,
    RecordCache,
    S2SFTDataset,
    WeightedDistributedSampler,
    collate_sft,
    decision_class,
    holdout_scenes,
    index_calls,
    inference_messages,
    select_training_refs,
    summarize_refs,
)
from src.data.s2_stream import StreamShardDataset, collate_pairs, is_pair_stream, read_manifest  # noqa: E402
from src.models.heatmap.native_internnav_exact import (  # noqa: E402
    append_native_lookdown_turn,
    native_requests_lookdown,
)

LOGGER = logging.getLogger("train_system2_lora")

ADAPTER_SCHEMA = "heatmapvln-s2-lora-adapter-v1"
PARITY_SCHEMA = "heatmapvln-s2-parity-v1"
# Language-model linear layers only.  Matched with re.fullmatch against module names
# of Qwen2_5_VLForConditionalGeneration (transformers 4.51: model.layers.N.*).  The
# vision tower's blocks also have mlp.{gate,up,down}_proj, so a plain name list would
# put adapters on the tower that the History Head reads.
LORA_TARGET_PATTERN = r"model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
LORA_MODULES_PER_LAYER = 7
# rpc_model_server._plan_panoramic_native: both turns decode exactly like this.
NATIVE_GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "do_sample": False,
    "use_cache": True,
    "return_dict_in_generate": True,
}
G0_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    device: torch.device
    backend: str
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def _accelerator(device: torch.device):
    return None if device.type == "cpu" else getattr(torch, device.type, None)


def _detect_device_kind() -> str:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("No NPU or CUDA device found: running on CPU")
    return "cpu"


def disable_mha_fastpath() -> None:
    """The fused encoder-layer kernel has no NPU implementation and falls back to the CPU.

    Same switch as ``rpc_model_server._disable_fused_mha_fastpath``; set on every
    device so training and serving run the same attention path.
    """
    backend = getattr(torch.backends, "mha", None)
    if backend is not None and hasattr(backend, "set_fastpath_enabled"):
        backend.set_fastpath_enabled(False)


def _warm_up_npu(device: torch.device) -> None:
    """Build CANN's op-compile toolchain now, as the server does before loading."""
    with torch.no_grad():
        image = torch.zeros(1, 3, 28, 28, device=device, dtype=torch.bfloat16)
        kernel = torch.zeros(8, 3, 14, 14, device=device, dtype=torch.bfloat16)
        F.conv2d(image, kernel, stride=14)
        square = torch.zeros(8, 8, device=device, dtype=torch.bfloat16)
        square @ square
    torch.npu.synchronize(device)


def init_runtime(kind: str = "auto", *, timeout_min: int = 60) -> Runtime:
    """Device and process group from torchrun's environment (one process per card)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    kind = _detect_device_kind() if kind == "auto" else kind
    if kind == "npu":
        try:
            # Registers the npu device type.  Never torch_npu.contrib.transfer_to_npu
            # (deploy_ascend_910b.md §7 ⑨).
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("--device npu but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("--device npu but torch.npu.is_available() is False")
        torch.npu.set_device(local)
        device, backend = torch.device("npu", local), "hccl"
    elif kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")
        torch.cuda.set_device(local)
        device, backend = torch.device("cuda", local), "nccl"
    elif kind == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError(f"unsupported --device {kind!r}")
    disable_mha_fastpath()
    if device.type == "npu":
        from src.models.qwen2_5_vl_vision_mask import install_qwen2_5_vl_vision_mask_patch

        # The server installs it on NPU before building the model; same here, so
        # the vision tower runs the same code in training, G0 and serving.
        install_qwen2_5_vl_vision_mask_patch("npu", LOGGER)
        _warm_up_npu(device)
    if world > 1:
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world, timeout=timedelta(minutes=timeout_min)
        )
    return Runtime(device=device, backend=backend, rank=rank, local_rank=local, world_size=world)


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    accelerator = _accelerator(device)
    if accelerator is not None and hasattr(accelerator, "manual_seed_all"):
        # The NPU's default generator seed is not a constant (deploy_ascend_910b.md §7 ⑤).
        accelerator.manual_seed_all(seed)


def memory_stats_gib(device: torch.device) -> dict[str, float]:
    accelerator = _accelerator(device)
    if accelerator is None or not hasattr(accelerator, "max_memory_allocated"):
        return {}
    return {
        "max_allocated_gib": round(accelerator.max_memory_allocated(device) / 2**30, 2),
        "max_reserved_gib": round(accelerator.max_memory_reserved(device) / 2**30, 2),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_system2(model_path: str | Path, device: torch.device):
    """System2 exactly as the server loads it: ``Qwen2_5VLIntegration._load_model``.

    That path remaps the InternVLA-N1 checkpoint config to
    ``Qwen2_5_VLForConditionalGeneration`` (System1's tensors are not part of it),
    loads bf16 with SDPA attention as ``configs/ppa_action_refine_v2_8gpu.yaml``
    asks, and builds the processor from the same directory.  The model comes back
    in eval mode with every parameter frozen.
    """
    from src.models.qwen2_5_vl.integration import Qwen2_5VLConfig, Qwen2_5VLIntegration

    integration = Qwen2_5VLIntegration(
        Qwen2_5VLConfig(
            model_path=str(model_path),
            device=str(device),
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            use_lora=False,
            gradient_checkpointing=False,
        )
    )
    integration._load_model()
    if integration.config.attn_implementation != "sdpa":
        raise RuntimeError(f"System2 loaded with {integration.config.attn_implementation}, not sdpa")
    return integration.model, integration.processor


def lora_config(r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=LORA_TARGET_PATTERN,
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_wrapped_modules(peft_model: torch.nn.Module) -> list[str]:
    """Base-model names of the modules carrying a LoRA adapter."""
    names = []
    for name, module in peft_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if isinstance(lora_a, torch.nn.ModuleDict) and len(lora_a) > 0:
            names.append(re.sub(r"^base_model\.model\.", "", name))
    return sorted(names)


def check_lora_targets(peft_model: torch.nn.Module, num_layers: int, *, trainable: bool = True) -> dict[str, Any]:
    """Refuse an adapter that reaches past the language model, or misses part of it."""
    wrapped = lora_wrapped_modules(peft_model)
    stray = [name for name in wrapped if not re.fullmatch(LORA_TARGET_PATTERN, name)]
    if stray or len(wrapped) != num_layers * LORA_MODULES_PER_LAYER:
        raise RuntimeError(
            f"LoRA wraps {len(wrapped)} modules (want {num_layers} x {LORA_MODULES_PER_LAYER}); "
            f"outside the language model: {stray[:5]}"
        )
    layers = {int(name.split(".")[2]) for name in wrapped}
    if layers != set(range(num_layers)):
        raise RuntimeError(f"LoRA covers layers {sorted(layers)}, not 0..{num_layers - 1}")
    params = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    if trainable:
        bad = [n for n, p in params if "lora_" not in n or p.dtype != torch.float32]
        if bad or not params:
            raise RuntimeError(f"trainable parameters other than fp32 LoRA weights: {bad[:5]}")
    elif params:
        raise RuntimeError(f"a loaded adapter must be frozen; trainable: {[n for n, _ in params][:5]}")
    return {
        "wrapped_modules": len(wrapped),
        "layers": num_layers,
        "trainable_params": int(sum(p.numel() for _, p in params)),
    }


def attach_new_lora(model, *, r: int, alpha: int, dropout: float, gradient_checkpointing: bool):
    from peft import get_peft_model

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Frozen embeddings would otherwise hand checkpointed layers inputs that do
        # not require grad, and the loss would come back without a graph.
        model.enable_input_require_grads()
    # autocast_adapter_dtype (the default) keeps the adapter weights in fp32 on a
    # bf16 base.
    return get_peft_model(model, lora_config(r, alpha, dropout))


def load_adapter(model, adapter_dir: str | Path):
    """Attach a saved adapter the way the server will: ``PeftModel.from_pretrained``."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    check_lora_targets(peft_model, int(model.config.num_hidden_layers), trainable=False)
    return peft_model


def adapter_off(model):
    """Context with the adapter disabled (a plain model has none to disable)."""
    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Greedy decision, as served
# ---------------------------------------------------------------------------


def _served_inputs(processor, messages: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    if "video_grid_thw" in inputs:
        raise RuntimeError("System2 prompts carry images only")
    return {key: value.to(device) for key, value in inputs.items()}


@torch.no_grad()
def greedy_native_decision(
    model,
    processor,
    messages: list[dict],
    images: list,
    lookdown,
    device: torch.device,
) -> tuple[str, ...]:
    """``_plan_panoramic_native``'s System2 part: turn 1, and turn 2 after a lookdown request."""
    inputs = _served_inputs(processor, messages, device)
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, **NATIVE_GENERATE_KWARGS).sequences
    first = processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    if not native_requests_lookdown(first):
        return (first,)
    messages, _ = append_native_lookdown_turn(messages, images, first, lookdown)
    inputs = _served_inputs(processor, messages, device)
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, **NATIVE_GENERATE_KWARGS).sequences
    return (first, processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Loss and schedule
# ---------------------------------------------------------------------------


def per_sample_token_mean_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean next-token cross-entropy over each row's supervised tokens, in fp32.

    Only supervised positions are upcast (a full fp32 copy of [L, 152064] logits
    would be about 1.4 GB per sample).  Every row must have at least one
    supervised token.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise RuntimeError("a training row has no supervised token")
    rows = mask.nonzero(as_tuple=True)[0]
    token_loss = F.cross_entropy(shift_logits[mask].float(), shift_labels[mask], reduction="none")
    sums = torch.zeros(labels.shape[0], dtype=torch.float32, device=logits.device).index_add(0, rows, token_loss)
    return sums / counts.to(torch.float32), counts


def micro_loss(per_sample: torch.Tensor, counts: torch.Tensor, mode: str, samples_per_rank_step: int,
               token_scale: torch.Tensor | None = None, *, kl_sum: torch.Tensor | None = None,
               kl_beta: float = 0.0) -> torch.Tensor:
    """One micro-batch's share of the step loss.  ``sample``: every sample weighs the same.  ``token``: every
    supervised token of the global batch weighs the same (InternNav's HF Trainer); ``token_scale`` is
    world_size / the global batch's supervised-token count, the world size undoing DDP's mean over ranks.
    ``kl_sum`` (token mode only) is the micro-batch's summed KL; the loss is then
    (sum CE + kl_beta * sum KL) * token_scale, one denominator for both (ledger EXP-34 E)."""
    if mode == "token":
        total = (per_sample * counts).sum()
        if kl_sum is not None:
            total = total + kl_beta * kl_sum
        return total * token_scale
    if kl_sum is not None:
        raise ValueError("the KL term shares the token-mode denominator; it needs loss normalization 'token'")
    return per_sample.sum() / samples_per_rank_step


def kl_rows(batch: dict[str, Any]) -> torch.Tensor:
    """Each row's ``kl`` tag as a bool tensor.  With ``--kl-beta`` > 0 every sample must carry one."""
    rows = int(batch["labels"].shape[0])
    tags = list(batch.get("kl") or [None] * rows)
    keys = list(batch.get("keys") or [None] * rows)
    missing = [key for key, tag in zip(keys, tags) if not isinstance(tag, bool)]
    if missing:
        raise RuntimeError(
            f"--kl-beta > 0 needs a true/false kl tag on every sample; {len(missing)} in this micro-batch "
            f"have none, e.g. {missing[:3]}"
        )
    return torch.tensor(tags, dtype=torch.bool)


def kl_token_mask(labels: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """[B, L-1]: the supervised next-token positions of the rows tagged ``kl: true``."""
    return (labels[:, 1:] != IGNORE_INDEX) & rows.to(labels.device)[:, None]


def base_supervised_logits(peft_model, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """p_base's logits at ``mask`` ([N, V]): the same peft model, adapter disabled, no graph.

    A plain full forward (vision tower included), not a shared-embedding shortcut.  Only
    the masked rows are kept, so the [L, V] logits go before the training forward runs.
    """
    with torch.no_grad(), adapter_off(peft_model):
        logits = peft_model(**inputs, use_cache=False).logits
        return logits[:, :-1, :][mask]


def supervised_kl_sum(logits: torch.Tensor, base_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over the masked positions of KL(p_base || p_theta) = sum_v p_base (log p_base - log p_theta), fp32."""
    log_theta = F.log_softmax(logits[:, :-1, :][mask].float(), dim=-1)
    log_base = F.log_softmax(base_logits.float(), dim=-1)
    return (log_base.exp() * (log_base - log_theta)).sum()


def micro_batch_loss(
    ddp_model,
    peft_model,
    batch: dict[str, Any],
    device: torch.device,
    *,
    mode: str,
    samples_per_rank_step: int,
    token_scale: torch.Tensor | None,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """(loss to backward, per-sample CE, CE token counts, summed KL or None, KL token count).

    ``kl_beta`` 0 is the CE-only trainer exactly: no adapter-off forward, the same loss
    expression.  A micro-batch with no ``kl: true`` row skips the adapter-off forward
    and adds an exact zero.
    """
    inputs = _to_device(batch, device)
    labels = batch["labels"].to(device, non_blocking=True)
    kl_sum, kl_tokens = None, 0
    if kl_beta > 0:
        mask = kl_token_mask(batch["labels"], kl_rows(batch))
        kl_tokens = int(mask.sum())
        mask = mask.to(device)
        base = base_supervised_logits(peft_model, inputs, mask) if kl_tokens else None
    logits = ddp_model(**inputs, use_cache=False).logits
    per_sample, counts = per_sample_token_mean_loss(logits, labels)
    if kl_beta > 0:
        kl_sum = supervised_kl_sum(logits, base, mask) if kl_tokens else torch.zeros((), dtype=torch.float32, device=device)
    loss = micro_loss(per_sample, counts, mode, samples_per_rank_step, token_scale, kl_sum=kl_sum, kl_beta=kl_beta)
    return loss, per_sample, counts, kl_sum, kl_tokens


# ---------------------------------------------------------------------------
# EXP-37: preference pairs (DPO)
# ---------------------------------------------------------------------------


def sequence_scores(model, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(s [B], supervised-token counts [B]): s(y) = mean log-prob of each row's supervised tokens.

    Exactly ``-per_sample_token_mean_loss``, i.e. the tokens the SFT cross-entropy
    supervises (every assistant turn, ``<|im_end|>`` included); only those positions
    are upcast, and the [L, V] logits go when this returns.
    """
    inputs = _to_device(batch, device)
    labels = batch["labels"].to(device, non_blocking=True)
    logits = model(**inputs, use_cache=False).logits
    per_sample, counts = per_sample_token_mean_loss(logits, labels)
    return -per_sample, counts


def reference_scores(peft_model, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """s_ref: the same peft model with the adapter disabled, no graph (as ``base_supervised_logits``)."""
    with torch.no_grad(), adapter_off(peft_model):
        return sequence_scores(peft_model, batch, device)


def dpo_loss(s_pol_c: torch.Tensor, s_pol_r: torch.Tensor, s_ref_c: torch.Tensor, s_ref_r: torch.Tensor, *,
             beta: float, nll_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(per-pair loss, its DPO part, the margin), AVERT eq. 3-5 on length-normalised scores.

    margin = (s_pol(c) - s_ref(c)) - (s_pol(r) - s_ref(r)); loss = -log sigmoid(beta * margin) - nll_weight * s_pol(c).
    With the adapter at B = 0 every margin is exactly 0 and the loss is log 2 - nll_weight * s_pol(c).
    """
    margin = (s_pol_c - s_ref_c) - (s_pol_r - s_ref_r)
    dpo = -F.logsigmoid(beta * margin)
    return dpo - nll_weight * s_pol_c, dpo, margin


def dpo_micro_batch_loss(
    model,
    peft_model,
    batch: dict[str, Any],
    device: torch.device,
    *,
    beta: float,
    nll_weight: float,
    pairs_per_rank_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """(loss to backward, detached per-pair values) for one micro-batch of pairs (``collate_pairs``).

    Four single-row forwards: the reference (adapter off, no graph) on chosen then rejected,
    then the policy (``model``, with graph) on chosen then rejected.  The loss is the pairs'
    sum over ``pairs_per_rank_step``, so a rank's accumulated gradient is the mean over its
    pairs of the step and :func:`allreduce_mean_grads` makes it the global batch's mean.
    """
    chosen, rejected = batch["chosen"], batch["rejected"]
    s_ref_c, _ = reference_scores(peft_model, chosen, device)
    s_ref_r, _ = reference_scores(peft_model, rejected, device)
    s_pol_c, counts_c = sequence_scores(model, chosen, device)
    s_pol_r, counts_r = sequence_scores(model, rejected, device)
    loss, dpo, margin = dpo_loss(s_pol_c, s_pol_r, s_ref_c, s_ref_r, beta=beta, nll_weight=nll_weight)
    values = {
        "loss": loss, "dpo": dpo, "margin": margin, "s_pol_c": s_pol_c, "s_pol_r": s_pol_r,
        "s_ref_c": s_ref_c, "s_ref_r": s_ref_r, "tokens_c": counts_c, "tokens_r": counts_r,
    }
    return loss.sum() / pairs_per_rank_step, {k: v.detach() for k, v in values.items()}


# Sums over a step's pairs, all-reduced once per step (fp32, as the SFT totals).
DPO_TOTALS = ("loss", "dpo", "nll", "correct", "margin", "s_pol_c", "s_pol_r", "s_ref_c", "s_ref_r", "pairs",
              "tokens_c", "tokens_r")


def dpo_step_totals(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """One micro-batch's contribution to :data:`DPO_TOTALS`."""
    sums = {
        "loss": values["loss"].sum(), "dpo": values["dpo"].sum(), "nll": -values["s_pol_c"].sum(),
        "correct": (values["margin"] > 0).sum(), "margin": values["margin"].sum(),
        "s_pol_c": values["s_pol_c"].sum(), "s_pol_r": values["s_pol_r"].sum(),
        "s_ref_c": values["s_ref_c"].sum(), "s_ref_r": values["s_ref_r"].sum(),
        "pairs": torch.tensor(values["margin"].numel()), "tokens_c": values["tokens_c"].sum(),
        "tokens_r": values["tokens_r"].sum(),
    }
    device = values["margin"].device
    return torch.stack([sums[name].to(device=device, dtype=torch.float32) for name in DPO_TOTALS])


def dpo_log_fields(totals: torch.Tensor, margin_abs_max: float, *, nll_weight: float) -> dict[str, Any]:
    """The step's means over its pairs, for ``train_log.jsonl``.

    ``zero_margin_loss`` = log 2 - nll_weight * mean s_pol(c), what the loss is when every margin
    is 0: at step 1 (B = 0) ``loss`` must equal it and ``margin_abs_max`` must be 0.
    """
    t = dict(zip(DPO_TOTALS, (float(v) for v in totals.tolist())))
    n = t["pairs"]
    mean = lambda name: round(t[name] / n, 6)  # noqa: E731
    return {
        "loss": mean("loss"), "dpo_loss": mean("dpo"), "nll": mean("nll"), "reward_acc": mean("correct"),
        "margin_mean": mean("margin"), "margin_abs_max": round(margin_abs_max, 6),
        "s_pol_c": mean("s_pol_c"), "s_pol_r": mean("s_pol_r"), "s_ref_c": mean("s_ref_c"),
        "s_ref_r": mean("s_ref_r"), "chosen_logratio": round((t["s_pol_c"] - t["s_ref_c"]) / n, 6),
        "rejected_logratio": round((t["s_pol_r"] - t["s_ref_r"]) / n, 6),
        "zero_margin_loss": round(math.log(2.0) + nll_weight * t["nll"] / n, 6),
        "pairs": int(n), "tokens_c": int(t["tokens_c"]), "tokens_r": int(t["tokens_r"]),
    }


def broadcast_trainable(params: Sequence[torch.Tensor], rt: "Runtime") -> None:
    """Rank 0's LoRA weights on every rank, the broadcast DDP makes when it wraps a model."""
    if rt.distributed:
        for p in params:
            dist.broadcast(p.data, 0)


def allreduce_mean_grads(params: Sequence[torch.Tensor], world_size: int, *, bucket_numel: int = 1 << 25) -> None:
    """Each gradient := its mean over ranks (DDP's average), in flat fp32 buckets, once per optimizer step.

    Every rank must call it with the same parameter list; a parameter no pair reached gets a
    zero gradient so the collectives line up.
    """
    if world_size <= 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    def flush(bucket: list[torch.Tensor]) -> None:
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat)
        flat.div_(world_size)
        offset = 0
        for p in bucket:
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n

    bucket: list[torch.Tensor] = []
    size = 0
    for p in params:
        bucket.append(p)
        size += p.grad.numel()
        if size >= bucket_numel:
            flush(bucket)
            bucket, size = [], 0
    if bucket:
        flush(bucket)


def warmup_cosine(total_steps: int, warmup_ratio: float):
    """LR multiplier: linear warmup over ``ceil(ratio * total)`` steps, then cosine to 0."""
    warmup = max(1, math.ceil(float(warmup_ratio) * total_steps)) if warmup_ratio > 0 else 0

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def batch_plan(global_batch: int, micro_batch: int, world_size: int) -> int:
    """Accumulation steps that make ``micro_batch * world_size * accum == global_batch``."""
    per_step = int(micro_batch) * int(world_size)
    if global_batch % per_step:
        raise ValueError(
            f"global batch {global_batch} is not a multiple of micro batch {micro_batch} x {world_size} ranks"
        )
    return global_batch // per_step


# ---------------------------------------------------------------------------
# Checkpoints, provenance, rank shards
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_dirs(output_dir: Path) -> list[Path]:
    """Completed checkpoints (they carry ``COMPLETE``), oldest first."""
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    found = [p for p in root.glob("step_*") if (p / "COMPLETE").is_file() and p.name[5:].isdigit()]
    return sorted(found, key=lambda p: int(p.name[5:]))


def save_checkpoint(
    output_dir: Path,
    peft_model,
    optimizer,
    scheduler,
    *,
    step: int,
    run_key: str,
    keep: int,
) -> Path:
    path = output_dir / "checkpoints" / f"step_{step:06d}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    peft_model.save_pretrained(str(path / "adapter"), safe_serialization=True)
    torch.save(
        {"step": step, "run_key": run_key, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        path / "trainer_state.pt",
    )
    (path / "COMPLETE").write_text(f"{step}\n", encoding="utf-8")
    for old in checkpoint_dirs(output_dir)[: -max(1, keep)]:
        shutil.rmtree(old, ignore_errors=True)
    return path


def restore_checkpoint(path: Path, peft_model, optimizer, scheduler, *, run_key: str) -> int:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    if state.get("run_key") != run_key:
        raise RuntimeError(
            f"{path} belongs to a different run (data, batch plan or hyperparameters changed); "
            "start a new --output-dir instead of resuming into this one"
        )
    weights = load_file(str(path / "adapter" / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(peft_model, weights)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{path}: adapter tensors the model does not have: {unexpected[:5]}")
    lora_params = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    if len(weights) != len(lora_params):
        raise RuntimeError(f"{path}: {len(weights)} adapter tensors for {len(lora_params)} LoRA parameters")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def provenance() -> dict[str, Any]:
    from scripts.training.manifest import _capture_git_state
    from scripts.training.source_fingerprint import compute_source_fingerprint

    git = _capture_git_state(REPO_ROOT)
    return {
        "repo": str(REPO_ROOT),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("is_dirty"),
        "source_fingerprint": compute_source_fingerprint(REPO_ROOT)["fingerprint"],
    }


def base_model_identity(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    identity: dict[str, Any] = {"path": str(root.resolve())}
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        if (root / name).is_file():
            identity[f"{name}_sha256"] = sha256_file(root / name)
    return identity


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def shard_rows(rows: Sequence[Any], rt: Runtime) -> list[Any]:
    return list(rows[rt.rank :: rt.world_size])


def gather_rank_rows(rows: list[dict], rt: Runtime, shard_dir: Path, tag: str) -> list[dict] | None:
    """Every rank writes its rows to ``shard_dir``; rank 0 reads all of them back.

    Files rather than a collective: the per-call rows are text and the HCCL object
    collectives have never been exercised on this box.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{tag}.rank{rt.rank:02d}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    rt.barrier()
    if not rt.is_main:
        return None
    merged: list[dict] = []
    for rank in range(rt.world_size):
        part = shard_dir / f"{tag}.rank{rank:02d}.jsonl"
        merged.extend(json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line)
    return merged


def shared_index(rt: Runtime, cache_path: Path, build):
    """Run ``build`` on rank 0 only and hand its result to the other ranks.

    Indexing opens every record's tar on the shared NFS; eight ranks doing it at
    once is eight times the reads for the same answer.
    """
    if not rt.distributed:
        return build()
    if rt.is_main:
        result = build()
        tmp = cache_path.with_name(f".{cache_path.name}.tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(result, handle)
        os.replace(tmp, cache_path)
    rt.barrier()
    if rt.is_main:
        return result
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def resolve_holdout(args: argparse.Namespace) -> tuple[str, ...]:
    if args.holdout_scenes:
        return tuple(sorted(s.strip() for s in args.holdout_scenes.split(",") if s.strip()))
    return holdout_scenes(args.holdout_count, salt=args.holdout_salt)


def _stable_order(keys: Iterable[str], seed: int) -> list[str]:
    return sorted(keys, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode("utf-8")).hexdigest())


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _log(rt: Runtime, log_path: Path | None, payload: dict[str, Any]) -> None:
    if not rt.is_main:
        return
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return {name: batch[name].to(device, non_blocking=True) for name in names if name in batch}


def train(args: argparse.Namespace, rt: Runtime) -> int:
    output_dir = Path(args.output_dir)
    if (output_dir / "manifest.json").is_file():
        raise RuntimeError(f"{output_dir} already holds a finished adapter")
    if checkpoint_dirs(output_dir) and not args.resume:
        raise RuntimeError(f"{output_dir} has checkpoints; pass --resume or use a new --output-dir")
    if rt.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    log_path = output_dir / "train_log.jsonl"
    started = time.time()

    holdout = resolve_holdout(args)
    stream = read_manifest(args.stream_dir) if args.stream_dir else None
    if stream is not None:
        return train_from_stream(args, rt, stream, output_dir=output_dir, log_path=log_path, started=started)

    def build():
        refs, index_stats = index_calls(
            args.record_root, args.label_root, split="train", holdout=holdout, require_labels=True
        )
        selected, select_stats = select_training_refs(
            refs,
            max_calls_per_episode=args.max_calls_per_episode or None,
            agree_weight=args.agree_weight,
            disagree_weight=args.disagree_weight,
            seed=args.seed,
            max_stop_corrections_per_episode=args.max_stop_corrections_per_episode,
        )
        return selected, index_stats, select_stats

    selected, index_stats, select_stats = shared_index(rt, output_dir / "train_index.pkl", build)
    if args.limit_samples:
        selected = selected[: args.limit_samples]
    if not selected:
        raise RuntimeError(f"no training calls (index {dict(index_stats)}, selection {dict(select_stats)})")
    keys_sha256 = hashlib.sha256("\n".join(r.key + "|" + "\x1f".join(r.turns) for r in selected).encode()).hexdigest()

    accum = batch_plan(args.global_batch, args.micro_batch, rt.world_size)
    steps_per_epoch = max(1, len(selected) // args.global_batch)
    total_steps = int(args.max_steps) if args.max_steps is not None else math.ceil(args.epochs * steps_per_epoch)
    hyper = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_pattern": LORA_TARGET_PATTERN,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "schedule": "linear_warmup_cosine",
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "global_batch": args.global_batch,
        "micro_batch": args.micro_batch,
        "grad_accum": accum,
        "world_size": rt.world_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "max_calls_per_episode": args.max_calls_per_episode,
        "max_stop_corrections_per_episode": args.max_stop_corrections_per_episode,
        "agree_weight": args.agree_weight,
        "disagree_weight": args.disagree_weight,
        "seed": args.seed,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
    }
    if args.loss_normalization != "sample":  # absent for the default, so older runs keep their run key
        hyper["loss_normalization"] = args.loss_normalization
    if args.kl_beta:  # likewise absent at 0
        hyper["kl_beta"] = args.kl_beta
    run_key = hashlib.sha256(json.dumps({"hyper": hyper, "keys": keys_sha256}, sort_keys=True).encode()).hexdigest()
    _log(rt, log_path, {
        "event": "data",
        "holdout_scenes": list(holdout),
        "index": dict(index_stats),
        "selection": dict(select_stats),
        "selected": summarize_refs(selected),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "grad_accum": accum,
    })

    def make_loader(processor, collate, epoch: int, offset: int, step: int):
        dataset = S2SFTDataset(
            selected,
            processor,
            default_vlm_image_size=tuple(args.default_vlm_image_size),
        )
        sampler = WeightedDistributedSampler(
            dataset.weights,
            steps_per_epoch * args.global_batch,
            rank=rt.rank,
            world_size=rt.world_size,
            seed=args.seed,
        )
        sampler.set_epoch(epoch, start=offset * args.micro_batch * accum)
        return DataLoader(
            dataset,
            batch_size=args.micro_batch,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            drop_last=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    return _fit(
        args, rt,
        output_dir=output_dir, log_path=log_path, started=started, holdout=holdout,
        hyper=hyper, run_key=run_key, total_steps=total_steps, steps_per_epoch=steps_per_epoch,
        accum=accum, make_loader=make_loader,
        data_manifest={
            "record_roots": [str(Path(r).resolve()) for r in args.record_root],
            "label_roots": [str(Path(r).resolve()) for r in (args.label_root or [])],
            "index": dict(index_stats),
            "selection": dict(select_stats),
            "selected": summarize_refs(selected),
            "selected_keys_sha256": keys_sha256,
            "limit_samples": args.limit_samples,
        },
    )


def stream_start_shard(step: int, global_batch: int, shard_size: int, num_workers: int) -> int:
    """First shard to read when (re)starting at optimizer ``step``.

    DataLoader workers read shards round-robin, ``num_workers`` at a time, so at any
    step the consumed samples are spread over the current round.  Reading restarts at
    the start of that round, and ``stream_resume_skip`` drops what was already trained.
    """
    workers = max(1, int(num_workers))
    return (step * global_batch // (shard_size * workers)) * workers


def stream_resume_skip(step: int, global_batch: int, shard_size: int, start_shard: int, rows_per_micro: int) -> int:
    """Micro-batches a rank drops after opening the stream at ``start_shard``, to land on ``step``.

    From the start of a workers' round the DataLoader's order is fixed by the worker
    count alone, so dropping the ``step * global_batch - start_shard * shard_size``
    samples trained before the checkpoint gives the uninterrupted run's batches exactly;
    without it the round's head is trained twice and the stream's tail never.
    ``rows_per_micro`` is world_size * micro_batch.
    """
    reread = int(step) * int(global_batch) - int(start_shard) * int(shard_size)
    if reread < 0 or reread % int(rows_per_micro):
        raise ValueError(f"step {step} does not lie after shard {start_shard} on a micro-batch boundary")
    return reread // int(rows_per_micro)


class _SkipHead:
    """``loader`` iterated from its ``skip``-th batch on (a resume inside a workers' round)."""

    def __init__(self, loader: DataLoader, skip: int):
        self.loader, self.skip = loader, int(skip)

    def __iter__(self):
        batches = iter(self.loader)
        for done in range(self.skip):
            if next(batches, None) is None:
                raise RuntimeError(f"the stream ran out after {done} of the {self.skip} micro-batches a resume skips")
        return batches


def prepare_stream_resume(stream_dir: Path, start_shard: int, rt: Runtime, log_path: Path) -> None:
    """Empty the buffer before a resumed reader starts, and say where the pusher must restart.

    The pusher keeps the buffer at its cap with shards far ahead of the reader, and
    the shards between the checkpoint and the crash were read and deleted.  Unless the
    buffer is emptied, the receiver answers BUSY to every re-push of those shards and
    the readers wait for them forever.  The marker files go too: a re-read shard must
    collect a fresh set.  (A first start does not clear anything: the pusher may
    already be filling the buffer, out of order across its ports.)
    """
    if rt.is_main:
        removed = 0
        for path in list(stream_dir.glob("stream-*.tar")) + list(stream_dir.glob("stream-*.tar.done.r*")):
            path.unlink(missing_ok=True)
            removed += 1
        (stream_dir / "NEED_PUSH_FROM").write_text(f"{start_shard}\n", encoding="utf-8")
        _log(rt, log_path, {
            "event": "stream_resume",
            "start_shard": start_shard,
            "buffer_files_removed": removed,
            "action": f"stop the running pusher, then: stream_transfer.py push ... --start-shard {start_shard}",
        })
        LOGGER.warning("stream resume: buffer cleared; restart the pusher with --start-shard %d", start_shard)
    rt.barrier()


def clear_stream_markers(stream_dir: Path, start_shard: int, rt: Runtime, log_path: Path) -> None:
    """``--keep-stream`` resume: every shard is still in the buffer, so only the done markers go.

    Nothing is deleted or re-pushed; the readers start again at ``start_shard``.
    """
    if rt.is_main:
        removed = 0
        for path in list(stream_dir.glob("stream-*.tar.done.r*")):
            path.unlink(missing_ok=True)
            removed += 1
        _log(rt, log_path, {"event": "stream_resume", "start_shard": start_shard, "keep_stream": True,
                            "markers_removed": removed})
    rt.barrier()


def open_stream(args: argparse.Namespace, rt: Runtime, stream: dict[str, Any], processor, *, step: int,
                log_path: Path) -> StreamShardDataset:
    """The rank's reader from optimizer ``step`` on; a resume either keeps the stream or empties the buffer."""
    shard_size = int(stream["shard_size"])
    start_shard = stream_start_shard(step, args.global_batch, shard_size, args.num_workers)
    if step and args.keep_stream:
        clear_stream_markers(Path(args.stream_dir), start_shard, rt, log_path)
    elif step:
        prepare_stream_resume(Path(args.stream_dir), start_shard, rt, log_path)
    return StreamShardDataset(
        args.stream_dir,
        processor,
        rank=rt.rank,
        world_size=rt.world_size,
        start_shard=start_shard,
        num_shards=int(stream["num_shards"]),
        shard_size=shard_size,
        wait_timeout_s=args.stream_wait_timeout,
        delete_consumed=not args.keep_stream,
        pairs=args.objective == "dpo",
    )


def stream_loader(args: argparse.Namespace, rt: Runtime, stream: dict[str, Any], processor, collate, *, step: int,
                  log_path: Path):
    """The rank's micro-batches from optimizer ``step`` on, the same ones an uninterrupted run reads."""
    dataset = open_stream(args, rt, stream, processor, step=step, log_path=log_path)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        collate_fn=collate,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    skip = stream_resume_skip(step, args.global_batch, int(stream["shard_size"]), dataset.start_shard,
                              rt.world_size * args.micro_batch)
    if not skip:
        return loader
    _log(rt, log_path, {"event": "stream_resume_skip", "step": step, "start_shard": dataset.start_shard,
                        "skip_micro_batches": skip})
    return _SkipHead(loader, skip)


def train_from_stream(
    args: argparse.Namespace,
    rt: Runtime,
    stream: dict[str, Any],
    *,
    output_dir: Path,
    log_path: Path,
    started: float,
) -> int:
    """Train from the shard stream the 4090 packed (design §4): one pass, in its order."""
    shard_size = int(stream["shard_size"])
    if args.limit_samples:
        raise RuntimeError("--limit-samples does not apply to a stream; pack a smaller stream or pass --max-steps")
    pairs = is_pair_stream(stream)
    if args.objective == "dpo" and not pairs:
        raise RuntimeError(f"--objective dpo needs a pair stream (pairs: true); {args.stream_dir} is an SFT stream")
    if args.objective != "dpo" and pairs:
        raise RuntimeError(f"{args.stream_dir} is a pair stream (pairs: true); it trains only with --objective dpo")
    if int(stream["global_batch"]) != args.global_batch:
        raise RuntimeError(f"stream was packed for global batch {stream['global_batch']}, not {args.global_batch}")
    if shard_size % (args.global_batch) or args.global_batch % rt.world_size:
        raise RuntimeError(f"shard size {shard_size} / global batch {args.global_batch} / world {rt.world_size} do not divide")
    holdout = tuple(stream["holdout"]["scenes"])
    if tuple(resolve_holdout(args)) != holdout:
        raise RuntimeError(f"stream held out {holdout}, the trainer would hold out {tuple(resolve_holdout(args))}")
    accum = batch_plan(args.global_batch, args.micro_batch, rt.world_size)
    steps_per_epoch = int(stream["total_samples"]) // args.global_batch
    total_steps = int(args.max_steps) if args.max_steps is not None else steps_per_epoch
    if total_steps > steps_per_epoch:
        raise RuntimeError(f"the stream holds {steps_per_epoch} steps, {total_steps} were asked for")
    stream_sha256 = hashlib.sha256("".join(s["sha256"] for s in stream["shards"]).encode()).hexdigest()
    hyper = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_pattern": LORA_TARGET_PATTERN,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "schedule": "linear_warmup_cosine",
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "global_batch": args.global_batch,
        "micro_batch": args.micro_batch,
        "grad_accum": accum,
        "world_size": rt.world_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "seed": args.seed,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "stream": {k: stream[k] for k in ("seed", "shard_size", "num_shards", "total_samples", "replay_fraction", "epochs")},
    }
    if args.loss_normalization != "sample":
        hyper["loss_normalization"] = args.loss_normalization
    if args.kl_beta:
        hyper["kl_beta"] = args.kl_beta
    if args.keep_stream:  # the worker count fixes the read order a resume skips into
        hyper["num_workers"] = args.num_workers
    if args.objective != "sft":  # absent for SFT, so older runs keep their run key
        hyper.update(objective=args.objective, dpo_beta=args.dpo_beta, dpo_nll_weight=args.dpo_nll_weight)
    run_key = hashlib.sha256(json.dumps({"hyper": hyper, "stream": stream_sha256}, sort_keys=True).encode()).hexdigest()
    _log(rt, log_path, {"event": "data", "stream_dir": args.stream_dir, "holdout_scenes": list(holdout),
                        "selection": stream["selection"], "drawn": stream["drawn"], "total_steps": total_steps})

    def make_loader(processor, collate, epoch: int, offset: int, step: int):
        return stream_loader(args, rt, stream, processor, collate, step=step, log_path=log_path)

    data_manifest = {
        "stream_dir": str(Path(args.stream_dir).resolve()),
        "stream_sha256": stream_sha256,
        "record_roots": stream["record_roots"],
        "label_roots": stream["label_roots"],
        "replay_root": stream.get("replay_root", ""),
        "selection": stream["selection"],
        "drawn": stream["drawn"],
    }
    if pairs:
        data_manifest.update(pairs=True, stream_schema=stream["schema"])
    return _fit(
        args, rt,
        output_dir=output_dir, log_path=log_path, started=started, holdout=holdout,
        hyper=hyper, run_key=run_key, total_steps=total_steps, steps_per_epoch=steps_per_epoch,
        accum=accum, make_loader=make_loader, data_manifest=data_manifest,
    )


def _fit(
    args: argparse.Namespace,
    rt: Runtime,
    *,
    output_dir: Path,
    log_path: Path,
    started: float,
    holdout: Sequence[str],
    hyper: dict[str, Any],
    run_key: str,
    total_steps: int,
    steps_per_epoch: int,
    accum: int,
    make_loader,
    data_manifest: dict[str, Any],
) -> int:
    """Model, LoRA, optimiser, the step loop and the adapter, for either data source.

    ``make_loader(processor, collate, epoch, offset, step)`` returns the DataLoader
    that yields the micro-batches of ``epoch`` from optimizer step ``offset`` on.
    """
    seed_everything(args.seed, rt.device)  # same LoRA init on every rank
    model, processor = load_system2(args.model_path, rt.device)
    num_layers = int(model.config.num_hidden_layers)
    peft_model = attach_new_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    lora_info = check_lora_targets(peft_model, num_layers)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, args.warmup_ratio))

    step = 0
    if args.resume:
        found = checkpoint_dirs(output_dir)
        if found:
            step = restore_checkpoint(found[-1], peft_model, optimizer, scheduler, run_key=run_key)
            _log(rt, log_path, {"event": "resume", "checkpoint": str(found[-1]), "step": step})

    dpo = args.objective == "dpo"
    ddp_model = peft_model
    if rt.distributed and dpo:
        # Four forwards (two with a graph) per backward: no DDP reducer, the gradients are
        # averaged by hand once per step (allreduce_mean_grads), DDP's start broadcast too.
        broadcast_trainable(trainable, rt)
    elif rt.distributed:
        from torch.nn.parallel import DistributedDataParallel

        # Every rank loads the same frozen weights from the same files, so only the
        # LoRA tensors need the initial broadcast and the gradient buckets.
        frozen = [n for n, p in peft_model.named_parameters() if not p.requires_grad]
        frozen += [n for n, _ in peft_model.named_buffers()]
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(peft_model, frozen)
        ddp_model = DistributedDataParallel(
            peft_model,
            device_ids=None if rt.device.type == "cpu" else [rt.device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    seed_everything(args.seed + 1009 * rt.rank + 7919 * step, rt.device)  # LoRA dropout differs per rank

    collate = partial(collate_pairs if dpo else collate_sft, pad_token_id=int(processor.tokenizer.pad_token_id))

    accelerator = _accelerator(rt.device)
    if accelerator is not None and hasattr(accelerator, "reset_peak_memory_stats"):
        accelerator.reset_peak_memory_stats(rt.device)
    last_loss = None
    window_start = time.time()
    window_steps = 0
    peft_model.train()
    while step < total_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        loader = make_loader(processor, collate, epoch, offset, step)
        batches = iter(loader)
        for _ in range(offset, steps_per_epoch):
            if step >= total_steps:
                break
            # CE loss sum, samples, supervised (CE) tokens, KL sum, KL tokens; fp32 because HCCL's fp64
            # support is not a given
            totals = torch.zeros(5, dtype=torch.float32, device=rt.device)
            if dpo:
                dpo_totals = torch.zeros(len(DPO_TOTALS), dtype=torch.float32, device=rt.device)
                margin_abs_max = torch.zeros((), dtype=torch.float32, device=rt.device)
            micro_batches = []
            for micro in range(accum):
                try:
                    micro_batches.append(next(batches))
                except StopIteration:
                    # Every rank holds the same number of samples, so they all get here at
                    # the same step: fail loudly rather than step on a partial batch.
                    raise RuntimeError(f"training data ran out at step {step}, micro-batch {micro}") from None
            if args.loss_normalization == "token":
                # InternNav's HF Trainer averages over tokens, so a 2-token STOP answer weighs less than a
                # 10-token pixel answer; DDP averages over ranks, hence world_size / global token count.
                window_tokens = torch.tensor(
                    float(sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in micro_batches)),
                    dtype=torch.float32, device=rt.device)
                if rt.distributed:
                    dist.all_reduce(window_tokens)
                token_scale = rt.world_size / window_tokens
            for micro, batch in enumerate(micro_batches):
                if dpo:
                    loss, values = dpo_micro_batch_loss(
                        peft_model, peft_model, batch, rt.device,
                        beta=args.dpo_beta, nll_weight=args.dpo_nll_weight,
                        pairs_per_rank_step=args.micro_batch * accum,
                    )
                    loss.backward()
                    dpo_totals += dpo_step_totals(values)
                    margin_abs_max = torch.maximum(margin_abs_max, values["margin"].abs().max().float())
                    continue
                sync = micro == accum - 1 or not rt.distributed
                with contextlib.nullcontext() if sync else ddp_model.no_sync():
                    loss, per_sample, counts, kl_sum, kl_tokens = micro_batch_loss(
                        ddp_model, peft_model, batch, rt.device,
                        mode=args.loss_normalization,
                        samples_per_rank_step=args.micro_batch * accum,
                        token_scale=token_scale if args.loss_normalization == "token" else None,
                        kl_beta=args.kl_beta,
                    )
                    loss.backward()
                totals[0] += per_sample.detach().sum()
                totals[1] += float(per_sample.shape[0])
                totals[2] += counts.sum().float()
                if kl_sum is not None:
                    totals[3] += kl_sum.detach()
                    totals[4] += float(kl_tokens)
            if dpo:
                allreduce_mean_grads(trainable, rt.world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_steps += 1
            if dpo:
                if rt.distributed:
                    dist.all_reduce(dpo_totals)
                    dist.all_reduce(margin_abs_max, op=dist.ReduceOp.MAX)
                dpo_fields = dpo_log_fields(dpo_totals, float(margin_abs_max), nll_weight=args.dpo_nll_weight)
                last_loss = float(dpo_totals[DPO_TOTALS.index("loss")] / dpo_totals[DPO_TOTALS.index("pairs")])
            else:
                if rt.distributed:
                    dist.all_reduce(totals)
                last_loss = float(totals[0] / totals[1])
            if (step % args.log_every == 0 or step == total_steps) and dpo:
                if accelerator is not None and hasattr(accelerator, "synchronize"):
                    accelerator.synchronize(rt.device)
                elapsed = time.time() - window_start
                _log(rt, log_path, {
                    "event": "step",
                    "step": step,
                    "epoch": epoch,
                    **dpo_fields,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": round(float(grad_norm), 6),
                    "s_per_step": round(elapsed / window_steps, 3),
                    "samples_per_s": round(window_steps * args.global_batch / max(elapsed, 1e-9), 3),
                    **memory_stats_gib(rt.device),
                })
                window_start, window_steps = time.time(), 0
            elif step % args.log_every == 0 or step == total_steps:
                if accelerator is not None and hasattr(accelerator, "synchronize"):
                    accelerator.synchronize(rt.device)
                elapsed = time.time() - window_start
                _log(rt, log_path, {
                    "event": "step",
                    "step": step,
                    "epoch": epoch,
                    "loss": round(last_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": round(float(grad_norm), 6),
                    "supervised_tokens": int(totals[2]),
                    "ce_tokens": int(totals[2]),
                    "kl_tokens": int(totals[4]),
                    "kl_mean": round(float(totals[3] / totals[4]), 6) if float(totals[4]) > 0 else None,
                    "s_per_step": round(elapsed / window_steps, 3),
                    "samples_per_s": round(window_steps * args.global_batch / max(elapsed, 1e-9), 3),
                    **memory_stats_gib(rt.device),
                })
                window_start, window_steps = time.time(), 0
            if args.save_every and step % args.save_every == 0 and step < total_steps:
                if rt.is_main:
                    save_checkpoint(output_dir, peft_model, optimizer, scheduler, step=step, run_key=run_key, keep=args.keep_checkpoints)
                rt.barrier()
        del batches, loader

    if rt.is_main:
        adapter_dir = output_dir / "adapter"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        peft_model.save_pretrained(str(adapter_dir), safe_serialization=True)
        manifest = {
            "schema": ADAPTER_SCHEMA,
            "created_unix": time.time(),
            "provenance": provenance(),
            "base_model": base_model_identity(args.model_path),
            "data": data_manifest,
            "holdout": {"scenes": list(holdout), "salt": args.holdout_salt, "count": len(holdout)},
            "hyperparameters": hyper,
            "lora": lora_info,
            "training": {
                "steps": step,
                "final_loss": last_loss,
                "wall_s": round(time.time() - started, 1),
                "device": str(rt.device.type),
                "backend": rt.backend,
                "run_key": run_key,
                **memory_stats_gib(rt.device),
            },
            "adapter": {
                "dir": "adapter",
                "files": {p.name: sha256_file(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()},
            },
        }
        write_json(output_dir / "manifest.json", manifest)
        _log(rt, log_path, {"event": "done", "adapter": str(adapter_dir), "steps": step, "final_loss": last_loss})
    rt.barrier()
    return 0


# ---------------------------------------------------------------------------
# Gate G0
# ---------------------------------------------------------------------------


def parity_check(args: argparse.Namespace, rt: Runtime) -> int:
    output_dir = Path(args.output_dir)
    if rt.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    holdout = resolve_holdout(args)
    refs, index_stats = shared_index(
        rt,
        output_dir / "parity_index.pkl",
        lambda: index_calls(
            args.record_root, args.label_root, split=args.parity_split, holdout=holdout, require_labels=False
        ),
    )
    by_uid = {ref.uid: ref for ref in refs}
    # G0 is an alignment test: replay only calls this platform answered (record_device).
    device = rt.device.type if args.parity_device == "auto" else args.parity_device
    eligible = {k: r for k, r in by_uid.items() if device == "any" or r.record_device == device}
    picks = [eligible[k] for k in _stable_order(eligible, args.seed)[: args.parity_check]]
    if not picks:
        raise RuntimeError(f"no recorded calls from device {device!r} to replay ({len(by_uid)} calls in total)")
    model, processor = load_system2(args.model_path, rt.device)
    if args.adapter:
        model = load_adapter(model, args.adapter)
    model.eval()
    records = RecordCache(8)
    rows = []
    for ref in shard_rows(picks, rt):
        record, calls = records.get(ref.record_path)
        call = calls[ref.call_index]
        messages, images, lookdown = inference_messages(
            record, call, default_vlm_image_size=tuple(args.default_vlm_image_size)
        )
        started = time.time()
        with adapter_off(model):
            replay = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
        rows.append({
            "key": ref.key,
            "uid": ref.uid,
            "scene_id": ref.scene_id,
            "kind": call.get("kind"),
            "recorded": list(ref.recorded),
            "replay": list(replay),
            "turn1_match": replay[0] == ref.recorded[0],
            "exact": tuple(replay) == tuple(ref.recorded),
            "class_match": decision_class(replay[-1]) == decision_class(ref.recorded[-1]),
            "seconds": round(time.time() - started, 3),
        })
    merged = gather_rank_rows(rows, rt, output_dir / "parity_shards", "parity")
    if rt.is_main:
        report = summarize_parity(merged, index_stats=dict(index_stats), split=args.parity_split, holdout=holdout,
                                  record_device=device)
        report.update({"adapter": str(args.adapter) if args.adapter else None, "provenance": provenance(),
                       "base_model": base_model_identity(args.model_path), "device": rt.device.type,
                       "world_size": rt.world_size})
        write_json(output_dir / "parity_g0.json", report)
        with open(output_dir / "parity_g0.calls.jsonl", "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        print(json.dumps({k: report[k] for k in ("n", "exact_rate", "turn1_rate", "class_rate", "g0_pass")}), flush=True)
    rt.barrier()
    return 0


def summarize_parity(rows: Sequence[dict], **extra: Any) -> dict[str, Any]:
    n = len(rows)

    def rate(flag: str, subset: Sequence[dict] = rows) -> float | None:
        return round(sum(bool(r[flag]) for r in subset) / len(subset), 6) if subset else None

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(r.get("kind")) for r in rows}):
        subset = [r for r in rows if str(r.get("kind")) == kind]
        by_kind[kind] = {"n": len(subset), "exact_rate": rate("exact", subset)}
    two_turn = [r for r in rows if len(r["recorded"]) == 2]
    exact = rate("exact")
    return {
        "schema": PARITY_SCHEMA,
        "n": n,
        "exact_rate": exact,
        "turn1_rate": rate("turn1_match"),
        "class_rate": rate("class_match"),
        "two_turn": {"n": len(two_turn), "exact_rate": rate("exact", two_turn)},
        "by_kind": by_kind,
        "g0_threshold": G0_THRESHOLD,
        "g0_pass": bool(exact is not None and exact >= G0_THRESHOLD),
        "mismatches": [r for r in rows if not r["exact"]][:50],
        **extra,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-root", action="append", default=None, help="s2 record root (repeatable)")
    parser.add_argument("--label-root", action="append", default=None,
                        help="labels root, one per --record-root or a single one for all")
    parser.add_argument("--model-path", required=True, help="InternNav_Model directory (DualVLN weights)")
    parser.add_argument("--holdout-scenes", default="", help="comma list; overrides --holdout-count/--holdout-salt")
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-salt", default=HOLDOUT_SALT)
    parser.add_argument("--default-vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE),
                        help="used only for records whose episode.json has no vlm_image_size")
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--parity-check", type=int, default=0, metavar="N",
                        help="gate G0 instead of training: replay N recorded calls with adapters off")
    parser.add_argument("--parity-split", choices=("holdout", "train", "all"), default="holdout")
    parser.add_argument("--parity-device", default="auto",
                        help="replay only calls recorded on this device: auto (= this run's), npu, cuda, any")
    parser.add_argument("--adapter", default="", help="parity check: load this adapter and disable it")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-normalization", choices=("sample", "token"), default="sample",
                        help="sample: every sample weighs the same (EXP-23..26 S1); token: every supervised token "
                             "of the global batch weighs the same, as in InternNav's System2 SFT")
    parser.add_argument("--kl-beta", type=float, default=0.0,
                        help="EXP-34: add BETA * KL(p_base || p_theta) on the supervised tokens of samples tagged "
                             "kl: true (p_base = adapter off), same denominator as the token-mean CE; "
                             "0 = the CE-only trainer exactly")
    parser.add_argument("--objective", choices=("sft", "dpo"), default="sft",
                        help="sft: cross-entropy on the stream's targets; dpo (EXP-37): AVERT's preference loss on "
                             "a pair stream (pairs: true), reference = the same model with the adapter off")
    parser.add_argument("--dpo-beta", type=float, default=0.5, help="--objective dpo: beta on the score margin")
    parser.add_argument("--dpo-nll-weight", type=float, default=0.1,
                        help="--objective dpo: lambda, the weight of -s_theta(chosen) added to every pair's loss")
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0, help="in weighted draws of len(selected) calls")
    parser.add_argument("--max-steps", type=int, default=None, help="optimizer steps; overrides --epochs (0 = save the init adapter)")
    parser.add_argument("--max-calls-per-episode", type=int, default=DEFAULT_MAX_CALLS_PER_EPISODE)
    parser.add_argument("--max-stop-corrections-per-episode", type=int, default=DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE)
    parser.add_argument("--agree-weight", type=float, default=DEFAULT_AGREE_WEIGHT)
    parser.add_argument("--disagree-weight", type=float, default=DEFAULT_DISAGREE_WEIGHT)
    parser.add_argument("--limit-samples", type=int, default=0, help="smoke runs only")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100, help="optimizer steps between checkpoints (0 = never)")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --output-dir")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--stream-dir", default="",
                        help="train from the 4090-packed shard stream in this buffer dir (design §4) instead of records")
    parser.add_argument("--stream-wait-timeout", type=float, default=6 * 3600.0,
                        help="seconds a reader waits for the next shard before failing")
    parser.add_argument("--keep-stream", action="store_true",
                        help="never delete stream shards; a --resume then re-reads from its start shard instead "
                             "of emptying the buffer for a re-push (EXP-34: the whole stream sits on /cache)")
    args = parser.parse_args(argv)
    if not args.stream_dir and not args.record_root:
        parser.error("--record-root is required unless --stream-dir is given")
    if args.stream_dir and args.parity_check:
        parser.error("--parity-check reads records, not a stream")
    if args.keep_stream and not args.stream_dir:
        parser.error("--keep-stream needs --stream-dir")
    if args.kl_beta < 0:
        parser.error("--kl-beta must be >= 0")
    if args.kl_beta > 0 and args.loss_normalization != "token":
        parser.error("--kl-beta shares the token-mean denominator: pass --loss-normalization token")
    if args.objective == "dpo":
        if not args.stream_dir:
            parser.error("--objective dpo reads a pair stream: pass --stream-dir")
        if args.kl_beta:
            parser.error("--objective dpo has its own reference term; --kl-beta must be 0")
        if args.loss_normalization != "sample":
            parser.error("--objective dpo averages over pairs; --loss-normalization must stay 'sample'")
        if args.micro_batch != 1:
            parser.error("--objective dpo runs one pair per micro-batch: pass --micro-batch 1")
        if not args.dpo_beta > 0 or args.dpo_nll_weight < 0:
            parser.error("--dpo-beta must be > 0 and --dpo-nll-weight >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    from scripts.training.source_fingerprint import (
        SOURCE_FINGERPRINT_ENV,
        check_pinned_source_fingerprint,
        compute_source_fingerprint,
    )

    if os.environ.get(SOURCE_FINGERPRINT_ENV):
        # The launcher pinned the tree; refuse to train code that moved since.
        check_pinned_source_fingerprint(compute_source_fingerprint(REPO_ROOT)["fingerprint"])
    rt = init_runtime(args.device)
    try:
        if args.parity_check:
            return parity_check(args, rt)
        return train(args, rt)
    finally:
        if rt.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
