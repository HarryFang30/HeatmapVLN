"""Pure-vision feature extraction from frozen native InternNav Qwen.

The worker collator preprocesses separate still images in this exact order::

    sample 0: history[0], ..., history[K0 - 1], current
    sample 1: history[0], ..., history[K1 - 1], current
    ...

It returns only ``pixel_values`` and ``image_grid_thw``.  This extractor calls
the native ``qwen.visual`` module directly, hooks intermediate ViT blocks, and
splits both pre-merge and final merged visual tokens back into occurrences.
It never runs the language model and needs no prompt, tokenizer, input ids,
video tokens, text anchors, LoRA, or panoramic adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeSingleViewFeatures:
    """Detached tensors from one frozen native visual-encoder forward.

    Attributes:
        current_vit: hooked ViT layer -> ``[B,C_vit,16,16]``.
        current_merged: final Qwen visual tokens -> ``[B,C_out,8,8]``.
        history_vit: hooked ViT layer -> ``[B,K,C_vit,16,16]``.
        history_merged: final visual tokens -> ``[B,K,C_out,8,8]``.
        history_queries: spatial mean of each history's final visual tokens,
            ``[B,K,C_out]``.  For InternNav-Model, ``C_out == 3584``.
        history_mask: real histories in the padded ``K`` dimension, ``[B,K]``.
    """

    current_vit: dict[int, torch.Tensor]
    current_merged: torch.Tensor
    history_vit: dict[int, torch.Tensor]
    history_merged: torch.Tensor
    history_queries: torch.Tensor
    history_mask: torch.Tensor


@dataclass(frozen=True)
class _Occurrence:
    global_index: int
    grid_t: int
    grid_h: int
    grid_w: int
    vit_start: int
    vit_end: int
    merged_start: int
    merged_end: int
    reverse_window_indices: torch.Tensor | None


class NativeSingleViewFeatureExtractor:
    """Extract current/history still-image features from frozen ``qwen.visual``.

    Args:
        model: Native InternNav model containing Qwen ``visual.blocks``.
        vit_layer_indices: Intermediate visual blocks used by the heatmap head.
        spatial_merge_size: Native Qwen spatial merge (2 for InternNav-Model).
        vit_output_spatial: Fixed heatmap ViT raster size.
        merged_output_spatial: Fixed heatmap final-visual raster size.
        require_frozen_backbone: Fail if any native model tensor is trainable.
        reject_lora: Fail if the checked hierarchy contains LoRA/PEFT names.
        scope_checks_to_visual: Apply ``reject_lora`` and
            ``require_frozen_backbone`` to the visual tower alone instead of
            the whole model, for stages that train the language model.
        restore_vit_spatial_layout: Undo Qwen window packing using the visual
            module's own ``get_window_index`` implementation.

    ``extract_from_pixels`` always runs the visual encoder under
    ``torch.no_grad`` and hooks always detach.  These are independent safety
    layers; optimizer construction must still whitelist the heatmap head.
    """

    def __init__(
        self,
        model,
        vit_layer_indices: Sequence[int],
        *,
        spatial_merge_size: int = 2,
        vit_output_spatial: int = 16,
        merged_output_spatial: int = 8,
        require_frozen_backbone: bool = True,
        reject_lora: bool = True,
        scope_checks_to_visual: bool = False,
        restore_vit_spatial_layout: bool = True,
    ) -> None:
        if not vit_layer_indices:
            raise ValueError("vit_layer_indices must not be empty")
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if vit_output_spatial <= 0 or merged_output_spatial <= 0:
            raise ValueError("output spatial sizes must be positive")

        self.model = model
        self.vit_layer_indices = tuple(int(index) for index in vit_layer_indices)
        self.spatial_merge_size = int(spatial_merge_size)
        self.vit_output_spatial = int(vit_output_spatial)
        self.merged_output_spatial = int(merged_output_spatial)
        self.restore_vit_spatial_layout = bool(restore_vit_spatial_layout)

        self._visual = self._get_visual_module(model)
        # What this extractor promises is that the *visual tower it reads* is
        # the released one, untouched and frozen -- that is the distribution
        # the Past Head was trained on.  By default the promise is enforced
        # over the whole model, which is right for stages that keep all of
        # Qwen frozen.  A stage that adapts the language model (EXP-13's
        # System2 memory arm) narrows it to the visual tower instead: LoRA on
        # text layers cannot reach these features, and pretending otherwise
        # would mean either giving up the contract or giving up the arm.
        checked = self._visual if scope_checks_to_visual else model
        if reject_lora:
            self._assert_no_lora(checked)
        if require_frozen_backbone:
            self._assert_frozen(checked)

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._vit_captures: dict[int, torch.Tensor] = {}
        for layer_index in self.vit_layer_indices:
            if layer_index < 0 or layer_index >= len(self._visual.blocks):
                raise IndexError(f"ViT layer {layer_index} is outside [0, {len(self._visual.blocks)})")
            self._handles.append(self._visual.blocks[layer_index].register_forward_hook(self._vit_hook(layer_index)))

    def extract_from_pixels(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_histories: Sequence[int] | torch.Tensor,
    ) -> NativeSingleViewFeatures:
        """Run frozen Qwen vision and return padded occurrence features.

        ``image_grid_thw`` rows and ``pixel_values`` patches must already be
        flattened in sample-major ``history...,current`` order.  Exactly
        ``sum(num_histories + 1)`` grid rows are required, so a leaked
        right/back/left or look-down image fails immediately.
        """

        self.clear()
        history_counts = self._normalize_history_counts(num_histories)
        sample_plan = self._build_occurrence_plan(image_grid_thw, history_counts)

        visual_dtype = getattr(self._visual, "dtype", None)
        if visual_dtype is None:
            first_parameter = next(self._visual.parameters(), None)
            visual_dtype = first_parameter.dtype if first_parameter is not None else pixel_values.dtype
        # ``pipeline.train()`` is recursive.  Reassert deterministic inference
        # mode at the narrow frozen-vision boundary even if a caller forgot to
        # restore the backbone after switching the trainable head to train().
        self._visual.eval()
        with torch.no_grad():
            merged_tokens = self._visual(
                pixel_values.to(dtype=visual_dtype),
                grid_thw=image_grid_thw,
            )
        if isinstance(merged_tokens, (tuple, list)):
            merged_tokens = merged_tokens[0]
        if not torch.is_tensor(merged_tokens) or merged_tokens.ndim != 2:
            shape = tuple(merged_tokens.shape) if torch.is_tensor(merged_tokens) else type(merged_tokens).__name__
            raise RuntimeError(f"qwen.visual must return [merged_tokens,C], got {shape}")
        merged_tokens = merged_tokens.detach()

        missing_layers = sorted(set(self.vit_layer_indices) - set(self._vit_captures))
        if missing_layers:
            raise RuntimeError(f"visual hooks did not fire for ViT layers {missing_layers}")
        expected_raw = sum(occurrence.vit_end - occurrence.vit_start for sample in sample_plan for occurrence in sample)
        for layer_index, hidden in self._vit_captures.items():
            if int(hidden.shape[0]) != expected_raw:
                raise RuntimeError(
                    f"ViT layer {layer_index} captured {int(hidden.shape[0])} patches, expected {expected_raw}"
                )
        expected_merged = sum(
            occurrence.merged_end - occurrence.merged_start for sample in sample_plan for occurrence in sample
        )
        if int(merged_tokens.shape[0]) != expected_merged:
            raise RuntimeError(
                f"qwen.visual returned {int(merged_tokens.shape[0])} merged tokens, expected {expected_merged}"
            )

        batch_size = len(sample_plan)
        max_history = max(history_counts, default=0)
        history_mask = torch.zeros(
            batch_size,
            max_history,
            dtype=torch.bool,
            device=merged_tokens.device,
        )
        for batch_index, history_count in enumerate(history_counts):
            history_mask[batch_index, :history_count] = True

        current_vit: dict[int, torch.Tensor] = {}
        history_vit: dict[int, torch.Tensor] = {}
        for layer_index in self.vit_layer_indices:
            sample_maps = [
                [self._restore_vit_map(self._vit_captures[layer_index], occurrence) for occurrence in sample]
                for sample in sample_plan
            ]
            current_vit[layer_index] = torch.stack([maps[-1] for maps in sample_maps], dim=0)
            history_vit[layer_index] = self._pad_history_maps(
                [maps[:-1] for maps in sample_maps],
                template=current_vit[layer_index][0],
                max_history=max_history,
            )

        merged_maps = [
            [self._restore_merged_map(merged_tokens, occurrence) for occurrence in sample] for sample in sample_plan
        ]
        current_merged = torch.stack([maps[-1] for maps in merged_maps], dim=0)
        history_merged = self._pad_history_maps(
            [maps[:-1] for maps in merged_maps],
            template=current_merged[0],
            max_history=max_history,
        )
        # Query semantics are explicitly the per-image mean of *native visual
        # merger tokens*, before any decoder-only spatial resize. This keeps
        # the 3584-d input shape of coarse.proj_history while avoiding the old
        # text-anchor semantics.
        query_dim = int(merged_tokens.shape[-1])
        history_queries = merged_tokens.new_zeros((batch_size, max_history, query_dim))
        for batch_index, sample in enumerate(sample_plan):
            for history_index, occurrence in enumerate(sample[:-1]):
                history_queries[batch_index, history_index] = merged_tokens[
                    occurrence.merged_start : occurrence.merged_end
                ].mean(dim=0)

        return NativeSingleViewFeatures(
            current_vit=current_vit,
            current_merged=current_merged,
            history_vit=history_vit,
            history_merged=history_merged,
            history_queries=history_queries,
            history_mask=history_mask,
        )

    @staticmethod
    def _normalize_history_counts(num_histories: Sequence[int] | torch.Tensor) -> list[int]:
        values = num_histories.detach().cpu().tolist() if torch.is_tensor(num_histories) else list(num_histories)
        result = [int(value) for value in values]
        if any(value < 0 for value in result):
            raise ValueError(f"num_histories must be non-negative, got {result}")
        if not result:
            raise ValueError("num_histories must contain one entry per sample")
        return result

    def _build_occurrence_plan(
        self,
        image_grid_thw: torch.Tensor,
        history_counts: Sequence[int],
    ) -> list[list[_Occurrence]]:
        if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
            raise ValueError(f"image_grid_thw must be [num_images,3], got {tuple(image_grid_thw.shape)}")
        expected_images = sum(count + 1 for count in history_counts)
        if int(image_grid_thw.shape[0]) != expected_images:
            raise RuntimeError(
                "single-view image count mismatch: image_grid_thw has "
                f"{int(image_grid_thw.shape[0])} rows, but num_histories requires {expected_images} "
                "(history images plus exactly one current image per sample)"
            )

        raw_prefix = [0]
        merged_prefix = [0]
        grid_rows = []
        for global_index, row in enumerate(image_grid_thw.detach().cpu().tolist()):
            grid_t, grid_h, grid_w = (int(value) for value in row)
            if grid_t != 1:
                raise RuntimeError(
                    f"image {global_index} has grid_t={grid_t}; native single-view heatmap requires "
                    "separate still images, not video grids"
                )
            if min(grid_h, grid_w) <= 0:
                raise RuntimeError(f"image {global_index} has invalid grid {(grid_t, grid_h, grid_w)}")
            if grid_h % self.spatial_merge_size or grid_w % self.spatial_merge_size:
                raise RuntimeError(
                    f"image {global_index} grid {(grid_t, grid_h, grid_w)} is not divisible by "
                    f"merge={self.spatial_merge_size}"
                )
            grid_rows.append((grid_t, grid_h, grid_w))
            raw_prefix.append(raw_prefix[-1] + grid_t * grid_h * grid_w)
            merged_prefix.append(
                merged_prefix[-1] + grid_t * (grid_h // self.spatial_merge_size) * (grid_w // self.spatial_merge_size)
            )

        reverse_indices = self._build_reverse_window_indices(image_grid_thw)

        result: list[list[_Occurrence]] = []
        global_index = 0
        for history_count in history_counts:
            sample = []
            for _local_index in range(history_count + 1):
                grid_t, grid_h, grid_w = grid_rows[global_index]
                sample.append(
                    _Occurrence(
                        global_index=global_index,
                        grid_t=grid_t,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        vit_start=raw_prefix[global_index],
                        vit_end=raw_prefix[global_index + 1],
                        merged_start=merged_prefix[global_index],
                        merged_end=merged_prefix[global_index + 1],
                        reverse_window_indices=reverse_indices[global_index],
                    )
                )
                global_index += 1
            result.append(sample)
        return result

    def _build_reverse_window_indices(self, image_grid_thw: torch.Tensor) -> list[torch.Tensor | None]:
        if not self.restore_vit_spatial_layout:
            return [None] * int(image_grid_thw.shape[0])
        get_window_index = getattr(self._visual, "get_window_index", None)
        if not callable(get_window_index):
            raise RuntimeError("Qwen visual has no get_window_index; cannot restore ViT raster layout")
        result = get_window_index(image_grid_thw)
        window_index = result[0] if isinstance(result, (tuple, list)) else result
        if not torch.is_tensor(window_index):
            raise RuntimeError("visual.get_window_index returned a non-tensor")
        window_index = window_index.detach().reshape(-1).to(dtype=torch.long)

        group_counts = [
            int(t) * (int(h) // self.spatial_merge_size) * (int(w) // self.spatial_merge_size)
            for t, h, w in image_grid_thw.detach().cpu().tolist()
        ]
        if int(window_index.numel()) != sum(group_counts):
            raise RuntimeError(f"window index has {int(window_index.numel())} groups, expected {sum(group_counts)}")

        layouts: list[torch.Tensor | None] = []
        group_offset = 0
        for global_index, group_count in enumerate(group_counts):
            local = window_index[group_offset : group_offset + group_count] - group_offset
            expected = torch.arange(group_count, device=local.device, dtype=local.dtype)
            if not torch.equal(torch.sort(local).values, expected):
                raise RuntimeError(
                    f"Qwen window packing crosses image boundary at occurrence {global_index}; "
                    "visual coordinates are ambiguous"
                )
            layouts.append(torch.argsort(local))
            group_offset += group_count
        return layouts

    def _vit_hook(self, layer_index: int):
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 2:
                shape = tuple(hidden.shape) if torch.is_tensor(hidden) else type(hidden).__name__
                raise RuntimeError(f"ViT layer {layer_index} hook expected [total_patches,C], got {shape}")
            self._vit_captures[layer_index] = hidden.detach()

        return hook

    def _restore_vit_map(self, hidden: torch.Tensor, occurrence: _Occurrence) -> torch.Tensor:
        tokens = hidden[occurrence.vit_start : occurrence.vit_end]
        expected = occurrence.grid_t * occurrence.grid_h * occurrence.grid_w
        if int(tokens.shape[0]) != expected:
            raise RuntimeError(
                f"ViT image {occurrence.global_index} has {int(tokens.shape[0])} tokens, expected {expected}"
            )
        if occurrence.reverse_window_indices is not None:
            merge = self.spatial_merge_size
            merge_area = merge**2
            grouped = tokens.reshape(expected // merge_area, merge_area, tokens.shape[-1])
            grouped = grouped.index_select(0, occurrence.reverse_window_indices.to(tokens.device))
            raster = grouped.reshape(
                occurrence.grid_t,
                occurrence.grid_h // merge,
                occurrence.grid_w // merge,
                merge,
                merge,
                tokens.shape[-1],
            )
            raster = raster.permute(0, 1, 3, 2, 4, 5).reshape(
                occurrence.grid_t,
                occurrence.grid_h,
                occurrence.grid_w,
                tokens.shape[-1],
            )
        else:
            raster = tokens.reshape(
                occurrence.grid_t,
                occurrence.grid_h,
                occurrence.grid_w,
                tokens.shape[-1],
            )
        return self._resize_chw(raster[0].permute(2, 0, 1), self.vit_output_spatial)

    def _restore_merged_map(self, merged_tokens: torch.Tensor, occurrence: _Occurrence) -> torch.Tensor:
        tokens = merged_tokens[occurrence.merged_start : occurrence.merged_end]
        height = occurrence.grid_h // self.spatial_merge_size
        width = occurrence.grid_w // self.spatial_merge_size
        expected = occurrence.grid_t * height * width
        if occurrence.grid_t != 1 or int(tokens.shape[0]) != expected:
            raise RuntimeError(
                f"merged visual image {occurrence.global_index} cannot form a still-image raster: "
                f"tokens={int(tokens.shape[0])}, grid="
                f"{(occurrence.grid_t, occurrence.grid_h, occurrence.grid_w)}"
            )
        raster = tokens.reshape(height, width, tokens.shape[-1]).permute(2, 0, 1)
        return self._resize_chw(raster, self.merged_output_spatial)

    @staticmethod
    def _resize_chw(raster: torch.Tensor, spatial_size: int) -> torch.Tensor:
        if raster.shape[-2:] == (spatial_size, spatial_size):
            return raster
        dtype = raster.dtype
        return (
            F.interpolate(
                raster.unsqueeze(0).float(),
                size=(spatial_size, spatial_size),
                mode="bilinear",
                align_corners=False,
            )
            .to(dtype=dtype)
            .squeeze(0)
        )

    @staticmethod
    def _pad_history_maps(
        samples: Sequence[Sequence[torch.Tensor]],
        *,
        template: torch.Tensor,
        max_history: int,
    ) -> torch.Tensor:
        result = template.new_zeros((len(samples), max_history, *template.shape))
        for batch_index, sample in enumerate(samples):
            if sample:
                result[batch_index, : len(sample)] = torch.stack(list(sample), dim=0)
        return result

    @staticmethod
    def _assert_frozen(model) -> None:
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if trainable:
            preview = ", ".join(trainable[:8])
            suffix = " ..." if len(trainable) > 8 else ""
            raise RuntimeError(
                "Native InternNav must be fully frozen before heatmap feature extraction; "
                f"found {len(trainable)} trainable parameters: {preview}{suffix}"
            )

    @staticmethod
    def _assert_no_lora(model) -> None:
        suspicious = sorted(
            {name for name, _parameter in model.named_parameters() if "lora" in name.lower()}
            | {
                name
                for name, module in model.named_modules()
                if "lora" in name.lower() or "lora" in module.__class__.__name__.lower()
            }
        )
        if suspicious:
            preview = ", ".join(suspicious[:8])
            suffix = " ..." if len(suspicious) > 8 else ""
            raise RuntimeError(f"LoRA/PEFT modules are forbidden in native single-view heatmap mode: {preview}{suffix}")

    def clear(self) -> None:
        self._vit_captures = {}

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self) -> NativeSingleViewFeatureExtractor:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.remove_hooks()

    @staticmethod
    def _candidate_nodes(model) -> Iterable[object]:
        seen: set[int] = set()
        queue = [model]
        while queue:
            node = queue.pop(0)
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            for attribute in ("base_model", "model"):
                child = getattr(node, attribute, None)
                if child is not None:
                    queue.append(child)

    @classmethod
    def _get_visual_module(cls, model):
        for node in cls._candidate_nodes(model):
            visual = getattr(node, "visual", None)
            if visual is not None and hasattr(visual, "blocks"):
                return visual
        raise RuntimeError("cannot locate native Qwen visual.blocks in model hierarchy")
