"""Past -> Plan -> Action latent modules.

The modules in this file deliberately live outside the released InternNav
System1 and the existing Heatmap Head.  This keeps both pretrained state-dict
contracts intact:

* :class:`PastToPlanBridge` changes only the four projected TRAJ tokens and is
  an exact identity at initialization.
* :class:`FutureTrajectoryHeatmapHead` owns the Plan-token adapter and a small
  Future-specific Fine output projection.  It reuses the existing Past Head's
  view/coarse/Fine-trunk modules by receiving them as forward arguments rather
  than registering aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.func import functional_call


PLAN_TIME_BINS = 4
PLAN_TOKEN_DIM = 768
MEMORY_TOKEN_DIM = 256
MAP_TOKEN_DIM = 256
MAP_VIEWS = 4
MAP_COARSE_SIZE = 8
MAP_SIZE = 64
MAP_DIRECTION_ORDER = ("front", "right", "back", "left")


class PastPlanActionContractError(ValueError):
    """Raised when an input violates the audited latent contract."""


@dataclass(frozen=True)
class Stage0EquivalenceReport:
    """Exact, same-process native-equivalence audit result."""

    plan_equal: bool
    fused_condition_equal: bool
    raw_trajectory_equal: bool
    trajectory_shape: tuple[int, ...]


@dataclass(frozen=True)
class Stage0TreatmentEquivalenceReport:
    """Exact Stage-0 audit through deployment action post-processing."""

    plan_equal: bool
    fused_condition_equal: bool
    raw_trajectory_equal: bool
    trajectory_shape: tuple[int, ...]
    treatment_spec_equal: bool
    treatment_spec: dict[str, object]


def _module_floating_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return a shared decoder's floating dtype without registering it."""

    for value in chain(module.parameters(), module.buffers()):
        if value.is_floating_point():
            return value.dtype
    return fallback


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    for value in chain(module.parameters(), module.buffers()):
        return value.device
    return fallback


class PastToPlanBridge(nn.Module):
    """One zero-initialized Memory -> Plan cross-attention residual.

    ``plan_z0`` is the output of the frozen native ``cond_projector`` and
    ``memory`` is the compact transformed history token emitted by the existing
    Trajectory-Guided Attention block.  The output projection is exactly zero
    initialized, so finite inputs satisfy ``torch.equal(output, plan_z0)`` at
    step zero.
    """

    def __init__(
        self,
        *,
        plan_dim: int = PLAN_TOKEN_DIM,
        memory_dim: int = MEMORY_TOKEN_DIM,
        num_heads: int = 8,
        max_delta_ratio: float | None = None,
    ) -> None:
        super().__init__()
        if plan_dim <= 0 or memory_dim <= 0 or num_heads <= 0:
            raise ValueError("bridge dimensions and num_heads must be positive")
        if plan_dim % num_heads:
            raise ValueError("plan_dim must be divisible by num_heads")
        if max_delta_ratio is not None and not 0.0 < float(max_delta_ratio) <= 1.0:
            raise ValueError("max_delta_ratio must be in (0, 1] or None")
        self.plan_dim = int(plan_dim)
        self.memory_dim = int(memory_dim)
        # Hard trust region: per-token ||delta|| may never exceed this fraction
        # of the native ||plan_z0|| token norm, in training AND deployment.  A
        # soft penalty loses the argmin race against the action loss (measured:
        # unconstrained refinement drifted to per-element delta RMS ~0.7 for a
        # <=4% teacher-forced gain and collapsed closed-loop SR), so the frozen
        # NextDiT must be protected by construction, not by loss weighting.
        self.max_delta_ratio = None if max_delta_ratio is None else float(max_delta_ratio)
        self.plan_norm = nn.LayerNorm(plan_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=plan_dim,
            num_heads=num_heads,
            kdim=memory_dim,
            vdim=memory_dim,
            dropout=0.0,
            batch_first=True,
        )
        # W_o=0 is the sole action-path initialization guarantee.
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(
        self,
        plan_z0: torch.Tensor,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if plan_z0.ndim != 3 or plan_z0.shape[-1] != self.plan_dim:
            raise PastPlanActionContractError(
                f"plan_z0 must be [B,Q,{self.plan_dim}], got {tuple(plan_z0.shape)}"
            )
        if not plan_z0.is_floating_point() or not torch.isfinite(plan_z0).all():
            raise PastPlanActionContractError(
                "plan_z0 must be finite floating point"
            )

        if memory is None:
            if memory_mask is not None:
                raise PastPlanActionContractError("memory_mask requires memory")
            result = plan_z0
            diagnostics = {
                "delta_z": torch.zeros_like(plan_z0),
                "delta_token_ratio": torch.zeros(
                    plan_z0.shape[:2],
                    dtype=torch.float32,
                    device=plan_z0.device,
                ),
                "sample_has_memory": torch.zeros(
                    plan_z0.shape[0], dtype=torch.bool, device=plan_z0.device
                ),
            }
            return (result, diagnostics) if return_diagnostics else result

        if (
            memory.ndim != 3
            or memory.shape[0] != plan_z0.shape[0]
            or memory.shape[-1] != self.memory_dim
        ):
            raise PastPlanActionContractError(
                f"memory must be [B,N,{self.memory_dim}], got {tuple(memory.shape)}"
            )
        if memory_mask is None or memory_mask.dtype != torch.bool:
            raise PastPlanActionContractError("memory_mask must be bool [B,N]")
        if tuple(memory_mask.shape) != tuple(memory.shape[:2]):
            raise PastPlanActionContractError(
                "memory_mask shape must match memory's first two dimensions"
            )
        if not memory.is_floating_point() or not torch.isfinite(memory).all():
            raise PastPlanActionContractError("memory must be finite floating point")
        bridge_device = self.cross_attention.out_proj.weight.device
        if self.cross_attention.out_proj.weight.dtype != torch.float32:
            raise PastPlanActionContractError(
                "Past-to-Plan bridge parameters must remain FP32"
            )
        if plan_z0.device != bridge_device or memory.device != bridge_device:
            raise PastPlanActionContractError(
                "plan_z0, memory, and the bridge parameters must share one device"
            )
        memory_mask = memory_mask.to(device=bridge_device)

        # MHA cannot consume an all-masked key row.  Compute only the valid
        # subset and leave every other sample as a hard identity bypass.
        sample_has_memory = memory_mask.any(dim=1)
        delta = torch.zeros_like(plan_z0)
        if bool(sample_has_memory.any()):
            idx = sample_has_memory.nonzero(as_tuple=False).flatten()
            # Global training autocast must not silently turn this small,
            # stability-critical residual branch into BF16.  The native Plan
            # dtype is restored only at the residual boundary.
            with torch.autocast(device_type=plan_z0.device.type, enabled=False):
                query = self.plan_norm(
                    plan_z0.index_select(0, idx).to(dtype=torch.float32)
                )
                key_value = self.memory_norm(
                    memory.index_select(0, idx).to(dtype=torch.float32)
                )
                key_padding_mask = ~memory_mask.index_select(0, idx)
                attention, _ = self.cross_attention(
                    query,
                    key_value,
                    key_value,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                if self.max_delta_ratio is not None:
                    # Per-token norm cap relative to the native Plan token.  An
                    # exact-zero delta stays exact zero (scale caps at 1), so
                    # the zero-bridge bitwise-identity guarantee is preserved.
                    z0_subset = plan_z0.index_select(0, idx).to(dtype=torch.float32)
                    delta_norm = attention.norm(dim=-1, keepdim=True)
                    z0_norm = z0_subset.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(
                        self.max_delta_ratio * z0_norm / delta_norm.clamp_min(1e-12),
                        max=1.0,
                    )
                    attention = attention * scale
            delta.index_copy_(0, idx, attention.to(dtype=plan_z0.dtype))
        result = plan_z0 + delta
        with torch.no_grad():
            delta_token_ratio = (
                delta.float().norm(dim=-1)
                / plan_z0.float().norm(dim=-1).clamp_min(1e-12)
            )
        diagnostics = {
            "delta_z": delta,
            "delta_token_ratio": delta_token_ratio,
            "sample_has_memory": sample_has_memory,
        }
        return (result, diagnostics) if return_diagnostics else result


class FutureTrajectoryHeatmapHead(nn.Module):
    """Decode four shared Plan tokens into four temporal trajectory maps.

    The module owns the Plan-to-map adapter, future type/horizon embeddings,
    and Future-specific Fine output projection.  It intentionally receives
    the existing Past Head's decoder modules during ``forward``. Consequently
    there is one shared view/coarse/Fine refinement trunk but no duplicate
    registration or checkpoint alias.
    """

    def __init__(
        self,
        *,
        plan_dim: int = PLAN_TOKEN_DIM,
        map_dim: int = MAP_TOKEN_DIM,
        num_time_bins: int = PLAN_TIME_BINS,
    ) -> None:
        super().__init__()
        if num_time_bins != 4:
            raise ValueError("v1 requires exactly four Plan time bins")
        self.plan_dim = int(plan_dim)
        self.map_dim = int(map_dim)
        self.num_time_bins = int(num_time_bins)
        self.plan_projection = nn.Linear(plan_dim, map_dim)
        self.future_type_embedding = nn.Parameter(torch.zeros(1, 1, map_dim))
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, num_time_bins, map_dim) * 0.02
        )
        self.spatial_queries = nn.Parameter(
            torch.randn(1, 1, MAP_VIEWS * MAP_COARSE_SIZE**2, map_dim) * 0.02
        )
        self.summary_norm = nn.LayerNorm(map_dim)
        self.spatial_norm = nn.LayerNorm(map_dim)
        # Past and Future share the Fine decoder's spatial refinement trunk,
        # but retain separate 64->1 readouts because point-history and future
        # tube logits have materially different output distributions.
        self.fine_output_projection = nn.Conv2d(
            64, 1, kernel_size=3, padding=1
        )

    def forward(
        self,
        plan_z: torch.Tensor,
        *,
        panoramic_vit_features: torch.Tensor,
        shared_heatmap_head: nn.Module,
        shared_visibility_head: nn.Module,
        shared_fine_decoder: nn.Module,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if plan_z.ndim != 3 or tuple(plan_z.shape[1:]) != (
            self.num_time_bins,
            self.plan_dim,
        ):
            raise PastPlanActionContractError(
                f"plan_z must be [B,4,{self.plan_dim}], got {tuple(plan_z.shape)}"
            )
        if not plan_z.is_floating_point() or not torch.isfinite(plan_z).all():
            raise PastPlanActionContractError("plan_z must be finite floating point")
        batch_size = plan_z.shape[0]
        expected_vit = (batch_size, MAP_VIEWS, self.map_dim, 16, 16)
        if tuple(panoramic_vit_features.shape) != expected_vit:
            raise PastPlanActionContractError(
                f"panoramic_vit_features must be {expected_vit}, got "
                f"{tuple(panoramic_vit_features.shape)}"
            )
        if (
            not panoramic_vit_features.is_floating_point()
            or not torch.isfinite(panoramic_vit_features).all()
        ):
            raise PastPlanActionContractError(
                "panoramic_vit_features must be finite floating point"
            )
        adapter_device = self.plan_projection.weight.device
        if plan_z.device != adapter_device or panoramic_vit_features.device != adapter_device:
            raise PastPlanActionContractError(
                "plan_z, panoramic_vit_features, and Future Head must share one device"
            )
        shared_modules = (
            shared_heatmap_head,
            shared_visibility_head,
            shared_fine_decoder,
        )
        if any(
            _module_device(module, adapter_device) != adapter_device
            for module in shared_modules
        ):
            raise PastPlanActionContractError(
                "all shared decoder modules must be on the Future Head device"
            )
        if time_mask is None:
            time_mask = torch.ones(
                batch_size,
                self.num_time_bins,
                dtype=torch.bool,
                device=plan_z.device,
            )
        if time_mask.dtype != torch.bool or tuple(time_mask.shape) != (
            batch_size,
            self.num_time_bins,
        ):
            raise PastPlanActionContractError("time_mask must be bool [B,4]")
        time_mask = time_mask.to(device=adapter_device)

        adapter_dtype = self.plan_projection.weight.dtype
        summary = self.plan_projection(plan_z.to(dtype=adapter_dtype))
        summary = self.summary_norm(
            summary + self.future_type_embedding + self.horizon_embedding
        )
        spatial = self.spatial_norm(summary.unsqueeze(2) + self.spatial_queries)
        coarse_dtype = _module_floating_dtype(shared_heatmap_head, spatial.dtype)
        visibility_dtype = _module_floating_dtype(
            shared_visibility_head, summary.dtype
        )
        coarse_logits = shared_heatmap_head(
            spatial.to(dtype=coarse_dtype)
        ).squeeze(-1).reshape(
            batch_size,
            self.num_time_bins,
            MAP_VIEWS,
            MAP_COARSE_SIZE,
            MAP_COARSE_SIZE,
        )
        visibility = shared_visibility_head(summary.to(dtype=visibility_dtype))
        fine_dtype = _module_floating_dtype(shared_fine_decoder, coarse_logits.dtype)
        heatmaps, heatmap_logits = _decode_with_shared_fine_trunk(
            shared_fine_decoder=shared_fine_decoder,
            future_output_projection=self.fine_output_projection,
            vit_fused=panoramic_vit_features.to(dtype=fine_dtype),
            coarse_heatmap=coarse_logits.to(dtype=fine_dtype),
            spatial_out=spatial.to(dtype=fine_dtype),
        )

        heatmap_mask = time_mask.to(dtype=heatmaps.dtype)
        visibility_logits = visibility * time_mask.to(
            dtype=visibility.dtype
        ).unsqueeze(-1)
        # Future tubes may legitimately cross a view boundary, so confidence
        # is four independent visibility probabilities rather than the Past
        # head's mutually exclusive five-way view gate.  This is display-only:
        # losses continue to consume raw spatial/visibility logits.
        visibility_probability = visibility.sigmoid() * time_mask.to(
            dtype=visibility.dtype
        ).unsqueeze(-1)
        spatial_mask = heatmap_mask[:, :, None, None, None]
        coarse_mask = time_mask.to(dtype=coarse_logits.dtype)[
            :, :, None, None, None
        ]
        heatmaps_gated = _confidence_gated_future_heatmaps(
            heatmap_logits,
            visibility_probability,
            time_mask,
        )
        return {
            "future_visibility": visibility_logits,
            "future_visibility_logits": visibility_logits,
            "future_visibility_probability": visibility_probability,
            "future_heatmaps": heatmaps * spatial_mask,
            "future_heatmaps_gated": heatmaps_gated,
            "future_heatmap_logits": heatmap_logits * spatial_mask,
            "future_coarse_heatmap": coarse_logits * coarse_mask,
            "future_time_mask": time_mask,
            "future_plan_summary": summary
            * time_mask.to(dtype=summary.dtype).unsqueeze(-1),
            "future_spatial_out": spatial
            * time_mask.to(dtype=spatial.dtype)[:, :, None, None],
            "future_heatmap_direction_order": MAP_DIRECTION_ORDER,
            "future_time_ranges": ((1, 8), (9, 16), (17, 24), (25, 32)),
        }


def _confidence_gated_future_heatmaps(
    heatmap_logits: torch.Tensor,
    visibility_probability: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Build display maps whose peak brightness is view confidence.

    Spatial shape comes directly from the logits learned by the spatial CE:
    ``exp(logit - per_view_max)`` is softmax normalized by its maximum.  It
    preserves the learned distance-dependent tube shape, has unit peak, and is
    exactly invariant to an arbitrary additive logit offset.  Independent
    sigmoid visibility then supplies the displayed peak brightness.

    This helper is display-only: raw maps/logits and all losses are unchanged.
    Masked horizons are hard-zeroed even if their logits are nonzero.  A map
    with no finite spatial maximum (for example an all-``-inf`` empty map) is
    also rendered as hard zero without producing NaNs.
    """

    if heatmap_logits.ndim != 5:
        raise PastPlanActionContractError(
            "future heatmap logits must be [B,T,V,H,W]"
        )
    if tuple(visibility_probability.shape) != tuple(heatmap_logits.shape[:3]):
        raise PastPlanActionContractError(
            "future visibility probability must be [B,T,V]"
        )
    if time_mask.dtype != torch.bool or tuple(time_mask.shape) != tuple(
        heatmap_logits.shape[:2]
    ):
        raise PastPlanActionContractError("future time mask must be bool [B,T]")
    if not heatmap_logits.is_floating_point():
        raise PastPlanActionContractError(
            "future heatmap logits must be floating point"
        )

    spatial_max = heatmap_logits.amax(dim=(-2, -1), keepdim=True)
    nonempty = torch.isfinite(spatial_max)
    safe_logits = torch.where(
        nonempty,
        heatmap_logits,
        torch.zeros_like(heatmap_logits),
    )
    safe_max = torch.where(nonempty, spatial_max, torch.zeros_like(spatial_max))
    normalized_shape = torch.exp(safe_logits - safe_max) * nonempty.to(
        dtype=heatmap_logits.dtype
    )
    confidence = visibility_probability.to(
        device=heatmap_logits.device,
        dtype=heatmap_logits.dtype,
    )[..., None, None]
    valid_time = time_mask.to(device=heatmap_logits.device)[..., None, None, None]
    return normalized_shape * confidence * valid_time


def _decode_with_shared_fine_trunk(
    *,
    shared_fine_decoder: nn.Module,
    future_output_projection: nn.Conv2d,
    vit_fused: torch.Tensor,
    coarse_heatmap: torch.Tensor,
    spatial_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Past Fine trunk with a Future-specific final projection.

    ``torch.func.functional_call`` replaces only ``refine.4`` for this call.
    The shared decoder remains registered exactly once under the Past Head;
    its refinement trunk receives gradients, while its Past output projection
    does not receive gradients from Future supervision.
    """

    named_parameters = dict(shared_fine_decoder.named_parameters())
    expected_weight = named_parameters.get("refine.4.weight")
    expected_bias = named_parameters.get("refine.4.bias")
    if expected_weight is None or expected_bias is None:
        raise PastPlanActionContractError(
            "shared Fine decoder must expose refine.4 weight/bias"
        )
    if tuple(expected_weight.shape) != tuple(future_output_projection.weight.shape):
        raise PastPlanActionContractError(
            "Future output projection does not match the shared Fine readout"
        )
    if tuple(expected_bias.shape) != tuple(future_output_projection.bias.shape):
        raise PastPlanActionContractError(
            "Future output bias does not match the shared Fine readout"
        )
    replacements = {
        "refine.4.weight": future_output_projection.weight.to(
            device=expected_weight.device, dtype=expected_weight.dtype
        ),
        "refine.4.bias": future_output_projection.bias.to(
            device=expected_bias.device, dtype=expected_bias.dtype
        ),
    }
    result = functional_call(
        shared_fine_decoder,
        replacements,
        args=(),
        kwargs={
            "vit_fused": vit_fused,
            "coarse_heatmap": coarse_heatmap,
            "spatial_out": spatial_out,
            "return_logits": True,
        },
        strict=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise PastPlanActionContractError(
            "shared Fine decoder must return (probabilities, logits)"
        )
    return result


class PastPlanActionChain(nn.Module):
    """Own the only two new learned components in the directed chain."""

    def __init__(
        self,
        *,
        plan_dim: int = PLAN_TOKEN_DIM,
        memory_dim: int = MEMORY_TOKEN_DIM,
        bridge_heads: int = 8,
        max_delta_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.bridge = PastToPlanBridge(
            plan_dim=plan_dim,
            memory_dim=memory_dim,
            num_heads=bridge_heads,
            max_delta_ratio=max_delta_ratio,
        )
        self.future_head = FutureTrajectoryHeatmapHead(plan_dim=plan_dim)

    def form_plan(
        self,
        traj_hidden_states: torch.Tensor,
        *,
        frozen_cond_projector: nn.Module,
        history_memory: torch.Tensor,
        history_memory_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Project native TRAJ states once and apply the sole memory bridge."""

        if any(parameter.requires_grad for parameter in frozen_cond_projector.parameters()):
            raise PastPlanActionContractError("native cond_projector must remain frozen")
        z0 = frozen_cond_projector(traj_hidden_states)
        bridge_result = self.bridge(
            z0,
            history_memory,
            history_memory_mask,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            z, diagnostics = bridge_result
            return z0, z, diagnostics
        return z0, bridge_result

    def decode_future(
        self,
        plan_z: torch.Tensor,
        *,
        past_output: dict[str, torch.Tensor],
        past_head: nn.Module,
        time_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return decode_future_from_shared_past_head(
            self.future_head,
            plan_z,
            past_output,
            past_head,
            time_mask=time_mask,
        )


def compute_shared_plan_action_losses(
    *,
    action_head: nn.Module,
    plan_z0: torch.Tensor,
    plan_z: torch.Tensor,
    gt_trajectory: torch.Tensor,
    trajectory_valid: torch.Tensor | None,
    traj_images: torch.Tensor | None,
    preserve_weight: float = 0.5,
    delta_weight: float = 0.01,
    delta_relative: bool = False,
    advantage_reference_mse: float | None = None,
    advantage_max_weight: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Compute action and native-preservation losses with shared randomness.

    This is intentionally not two calls to ``compute_loss``: those would draw
    different noise/timesteps and make the preservation target meaningless.

    ``delta_relative`` reports the delta penalty as the scale-free per-token
    ratio ``||delta||^2 / ||plan_z0||^2`` instead of the absolute per-element
    mean square.  ``advantage_reference_mse`` enables advantage weighting: each
    sample's action loss is scaled by ``clamp(native_mse / reference, max=
    advantage_max_weight)`` under shared noise, so the bridge is pushed only
    where the frozen native System1 is actually wrong.  The bulk of the native
    residual is irreducible multimodality; uniform weighting turns that floor
    into pure drift pressure on the Plan tokens.
    """

    if preserve_weight < 0.0 or delta_weight < 0.0:
        raise ValueError("preserve_weight and delta_weight must be non-negative")
    if advantage_reference_mse is not None:
        reference = float(advantage_reference_mse)
        if not torch.isfinite(torch.tensor(reference)) or reference <= 0.0:
            raise ValueError("advantage_reference_mse must be a positive finite value")
        if float(advantage_max_weight) < 1.0:
            raise ValueError("advantage_max_weight must be >= 1")
    required = (
        "_expand_sequence_training_inputs",
        "sample_flow_matching_inputs",
        "predict_velocity_from_projected",
        "masked_velocity_mse",
    )
    if any(not hasattr(action_head, name) for name in required):
        raise PastPlanActionContractError("action_head lacks projected flow APIs")
    if plan_z0.ndim != 3 or plan_z.shape != plan_z0.shape:
        raise PastPlanActionContractError(
            "plan_z0 and plan_z must have the same [B,Q,D] shape"
        )
    if plan_z0.device != plan_z.device or plan_z0.dtype != plan_z.dtype:
        raise PastPlanActionContractError(
            "plan_z0 and plan_z must share dtype and device"
        )
    if not plan_z.is_floating_point():
        raise PastPlanActionContractError("Plan tokens must be floating point")
    if action_head.training:
        raise PastPlanActionContractError(
            "frozen native action_head must remain in eval mode"
        )
    if any(parameter.requires_grad for parameter in action_head.parameters()):
        raise PastPlanActionContractError(
            "native action_head parameters must remain frozen"
        )

    # InternNav training can attach N current images/targets to one System2
    # Plan.  Reuse the released expansion rule once, then repeat the native
    # reference Plan in exactly the same batch-major/frame-minor order.
    expanded_z, expanded_gt, expanded_images, expanded_valid = (
        action_head._expand_sequence_training_inputs(
            plan_z,
            gt_trajectory,
            traj_images,
            trajectory_valid,
        )
    )
    if expanded_z.shape[0] == plan_z.shape[0]:
        expanded_z0 = plan_z0
    else:
        if expanded_z.shape[0] % plan_z.shape[0] != 0:
            raise PastPlanActionContractError(
                "expanded action batch is not divisible by Plan batch"
            )
        repeats = expanded_z.shape[0] // plan_z.shape[0]
        expanded_z0 = (
            plan_z0.unsqueeze(1)
            .expand(-1, repeats, -1, -1)
            .reshape(expanded_z.shape)
        )
    if expanded_z.ndim != 3 or expanded_z.shape != expanded_z0.shape:
        raise PastPlanActionContractError(
            "native sequence expansion produced incompatible Plan tensors"
        )
    if expanded_gt.ndim != 3 or expanded_gt.shape[0] != expanded_z.shape[0]:
        raise PastPlanActionContractError(
            "native sequence expansion produced incompatible trajectories"
        )

    noisy, timesteps, target_velocity = action_head.sample_flow_matching_inputs(
        expanded_gt
    )
    velocity = action_head.predict_velocity_from_projected(
        expanded_z,
        noisy,
        timesteps,
        traj_images=expanded_images,
    )
    with torch.no_grad():
        native_velocity = action_head.predict_velocity_from_projected(
            expanded_z0,
            noisy,
            timesteps,
            traj_images=expanded_images,
        )
    advantage_weights = None
    if advantage_reference_mse is not None:
        with torch.no_grad():
            native_per_sample = (
                (native_velocity.float() - target_velocity.float())
                .square()
                .mean(dim=(1, 2))
            )
            advantage_weights = torch.clamp(
                native_per_sample / float(advantage_reference_mse),
                max=float(advantage_max_weight),
            )
    if advantage_weights is None:
        action_loss = action_head.masked_velocity_mse(
            velocity,
            target_velocity,
            trajectory_valid=expanded_valid,
        )
    else:
        # Weighted numerator over the UNWEIGHTED valid denominator: with all
        # weights at 1 this is exactly masked_velocity_mse, and a per-rank
        # batch of one still sees its weight (a weighted mean would cancel it).
        per_sample = (
            (velocity.float() - target_velocity.float()).square().mean(dim=(1, 2))
        )
        valid = (
            torch.ones_like(per_sample)
            if expanded_valid is None
            else expanded_valid.float()
        )
        denominator = valid.sum()
        if denominator <= 0:
            action_loss = per_sample.sum() * 0.0
        else:
            action_loss = (per_sample * valid * advantage_weights).sum() / denominator
    preserve_loss = action_head.masked_velocity_mse(
        velocity,
        native_velocity,
        trajectory_valid=expanded_valid,
    )
    if delta_relative:
        delta_sq = (plan_z.float() - plan_z0.float()).square().sum(dim=-1)
        z0_sq = plan_z0.float().square().sum(dim=-1).clamp_min(1e-12)
        delta_loss = (delta_sq / z0_sq).mean()
    else:
        delta_loss = (plan_z.float() - plan_z0.float()).square().mean()
    total = (
        action_loss
        + float(preserve_weight) * preserve_loss
        + float(delta_weight) * delta_loss
    )
    result = {
        "total": total,
        "action": action_loss,
        "preserve": preserve_loss,
        "delta_z_l2": delta_loss,
        "velocity": velocity,
        "native_velocity": native_velocity,
        "shared_noisy_trajectory": noisy,
        "shared_timesteps": timesteps,
    }
    if advantage_weights is not None:
        result["advantage_weight_mean"] = advantage_weights.mean()
    return result


def decode_future_from_shared_past_head(
    future_head: FutureTrajectoryHeatmapHead,
    plan_z: torch.Tensor,
    past_output: dict[str, torch.Tensor],
    past_head: nn.Module,
    *,
    time_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Checked convenience wrapper that avoids registering shared modules."""

    panoramic_vit = past_output.get("panoramic_vit_features")
    if panoramic_vit is None:
        raise PastPlanActionContractError(
            "past_output lacks panoramic_vit_features; request memory tokens"
        )
    coarse = getattr(past_head, "coarse", None)
    fine = getattr(past_head, "fine", None)
    if coarse is None or fine is None:
        raise PastPlanActionContractError("past_head lacks coarse/fine decoder")
    return future_head(
        plan_z,
        panoramic_vit_features=panoramic_vit,
        shared_heatmap_head=coarse.heatmap_head,
        shared_visibility_head=coarse.vis_head,
        shared_fine_decoder=fine,
        time_mask=time_mask,
    )


@torch.no_grad()
def verify_stage0_native_equivalence(
    *,
    action_head: nn.Module,
    plan_z0: torch.Tensor,
    plan_z: torch.Tensor,
    traj_images: torch.Tensor | None,
    initial_noise: torch.Tensor,
    old_heatmap_control_enabled: bool = False,
    pano_latent_adapter_enabled: bool = False,
) -> Stage0EquivalenceReport:
    """Prove zero-bridge treatment is bitwise native under shared noise.

    This deliberately checks the projected condition before running the full
    frozen sampler.  It must run in one process, on one model instance, with
    explicit ``initial_noise``; reseeding two independent calls is not an
    acceptable substitute.
    """

    if old_heatmap_control_enabled or pano_latent_adapter_enabled:
        raise PastPlanActionContractError(
            "Stage 0 requires all legacy action-control adapters disabled"
        )
    if action_head.training:
        raise PastPlanActionContractError("Stage 0 action_head must be in eval mode")
    if any(parameter.requires_grad for parameter in action_head.parameters()):
        raise PastPlanActionContractError("Stage 0 action_head must be frozen")
    if not hasattr(action_head, "_fuse_projected_conditions") or not hasattr(
        action_head, "get_trajectory_from_projected"
    ):
        raise PastPlanActionContractError(
            "action_head lacks projected inference APIs"
        )
    if plan_z0.ndim != 3 or plan_z.shape != plan_z0.shape:
        raise PastPlanActionContractError(
            "Stage 0 Plan tensors must have the same [B,Q,D] shape"
        )
    if initial_noise.ndim != 3 or not initial_noise.is_floating_point():
        raise PastPlanActionContractError(
            "initial_noise must be floating [B*num_samples,T,A]"
        )
    if not torch.isfinite(initial_noise).all():
        raise PastPlanActionContractError("initial_noise contains non-finite values")
    if not torch.equal(plan_z, plan_z0):
        raise PastPlanActionContractError("zero bridge did not preserve Plan tokens")

    fused_native = action_head._fuse_projected_conditions(plan_z0, traj_images)
    fused_treatment = action_head._fuse_projected_conditions(plan_z, traj_images)
    if not torch.equal(fused_native, fused_treatment):
        raise PastPlanActionContractError(
            "fused NextDiT condition differs under zero bridge"
        )

    native = action_head.get_trajectory_from_projected(
        plan_z0,
        traj_images=traj_images,
        initial_noise=initial_noise.clone(),
    )
    treatment = action_head.get_trajectory_from_projected(
        plan_z,
        traj_images=traj_images,
        initial_noise=initial_noise.clone(),
    )
    if not torch.equal(native, treatment):
        max_error = float((native.float() - treatment.float()).abs().max())
        raise PastPlanActionContractError(
            "raw trajectories differ under zero bridge; "
            f"max_abs_error={max_error:.9g}"
        )
    return Stage0EquivalenceReport(
        plan_equal=True,
        fused_condition_equal=True,
        raw_trajectory_equal=True,
        trajectory_shape=tuple(int(v) for v in native.shape),
    )


@torch.no_grad()
def verify_stage0_treatment_equivalence(
    *,
    action_head: nn.Module,
    plan_z0: torch.Tensor,
    plan_z: torch.Tensor,
    traj_images: torch.Tensor | None,
    initial_noise: torch.Tensor,
    postprocess_config: object,
    old_heatmap_control_enabled: bool = False,
    pano_latent_adapter_enabled: bool = False,
) -> Stage0TreatmentEquivalenceReport:
    """Extend the bitwise sampler audit through the exact local action queue.

    The two sampler calls share one explicit noise tensor.  Their raw outputs
    are then independently passed through selection, metric scaling, optional
    heading alignment, discretization, STOP padding, local-replan semantics,
    and first-STOP anti-deadlock handling.  Training must not start unless the
    resulting :class:`TreatmentSpec` values are exactly equal.
    """

    from src.models.action.treatment_spec import (
        TrajectoryPostprocessConfig,
        assert_exact_treatment_spec_equal,
        build_treatment_spec,
    )

    if not isinstance(postprocess_config, TrajectoryPostprocessConfig):
        raise PastPlanActionContractError(
            "postprocess_config must be TrajectoryPostprocessConfig"
        )
    if old_heatmap_control_enabled or pano_latent_adapter_enabled:
        raise PastPlanActionContractError(
            "Stage 0 requires all legacy action-control adapters disabled"
        )
    if action_head.training:
        raise PastPlanActionContractError("Stage 0 action_head must be in eval mode")
    if any(parameter.requires_grad for parameter in action_head.parameters()):
        raise PastPlanActionContractError("Stage 0 action_head must be frozen")
    if not hasattr(action_head, "_fuse_projected_conditions") or not hasattr(
        action_head, "get_trajectory_from_projected"
    ):
        raise PastPlanActionContractError(
            "action_head lacks projected inference APIs"
        )
    if plan_z0.ndim != 3 or plan_z.shape != plan_z0.shape:
        raise PastPlanActionContractError(
            "Stage 0 Plan tensors must have the same [B,Q,D] shape"
        )
    if initial_noise.ndim != 3 or not initial_noise.is_floating_point():
        raise PastPlanActionContractError(
            "initial_noise must be floating [B*num_samples,T,A]"
        )
    if not torch.isfinite(initial_noise).all():
        raise PastPlanActionContractError("initial_noise contains non-finite values")
    if not torch.equal(plan_z, plan_z0):
        raise PastPlanActionContractError("zero bridge did not preserve Plan tokens")

    fused_native = action_head._fuse_projected_conditions(plan_z0, traj_images)
    fused_treatment = action_head._fuse_projected_conditions(plan_z, traj_images)
    if not torch.equal(fused_native, fused_treatment):
        raise PastPlanActionContractError(
            "fused NextDiT condition differs under zero bridge"
        )
    native = action_head.get_trajectory_from_projected(
        plan_z0,
        traj_images=traj_images,
        initial_noise=initial_noise.clone(),
    )
    treatment = action_head.get_trajectory_from_projected(
        plan_z,
        traj_images=traj_images,
        initial_noise=initial_noise.clone(),
    )
    if not torch.equal(native, treatment):
        max_error = float((native.float() - treatment.float()).abs().max())
        raise PastPlanActionContractError(
            "raw trajectories differ under zero bridge; "
            f"max_abs_error={max_error:.9g}"
        )
    native_spec = build_treatment_spec(native, postprocess_config)
    treatment_spec = build_treatment_spec(treatment, postprocess_config)
    assert_exact_treatment_spec_equal(native_spec, treatment_spec)
    return Stage0TreatmentEquivalenceReport(
        plan_equal=True,
        fused_condition_equal=True,
        raw_trajectory_equal=True,
        trajectory_shape=tuple(int(v) for v in native.shape),
        treatment_spec_equal=True,
        treatment_spec=treatment_spec.to_dict(),
    )
