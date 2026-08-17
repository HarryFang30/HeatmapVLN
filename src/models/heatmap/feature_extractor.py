"""
Feature Extractor for HeatmapVLN
================================

Registers forward hooks on the Qwen2.5-VL backbone to capture:

1. ViT intermediate-layer features (16x16 per image, pre-merge)
2. LLM multi-layer hidden states (8x8 per image, post-merge)
3. Text-anchor hidden states used as history queries

Important:

- History queries come from the deepest *hooked* LLM layer
  (i.e. ``max(llm_layer_indices)``), not necessarily the model's final layer.
- Current default InternNav config uses Qwen2.5-VL with LLM hook layers
  ``[6, 13, 20]`` and ViT hook layers ``[7, 15, 23, 31]``.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Qwen2.5-VL token-layout constants
TOKENS_PER_IMAGE_VIT = 256  # 16x16 pre-merge
TOKENS_PER_IMAGE_LLM = 64  # 8x8 post-merge (after 2x2 spatial merge)
VIT_SPATIAL = 16
LLM_SPATIAL = 8

HISTORY_QUERY_TEXT_ANCHOR = "text_anchor"
HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1 = "history_visual_equal_view_mean_v1"
HISTORY_QUERY_SOURCES = frozenset(
    {
        HISTORY_QUERY_TEXT_ANCHOR,
        HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1,
    }
)


@dataclass(frozen=True)
class _ViTSpatialLayout:
    """Per-image metadata needed to undo Qwen's visual token packing."""

    grid_t: int
    grid_h: int
    grid_w: int
    reverse_window_indices: torch.Tensor


class FeatureExtractor:
    """
    Hook-based feature extractor for a frozen Qwen backbone base model.

    After the model forward pass, call ``extract()`` to retrieve
    grouped features for downstream coarse / fine localisation heads.

    Args:
        model:             VLM backbone model instance.
        vit_layer_indices: ViT block indices to hook (e.g. [6, 12, 18, 24]).
        llm_layer_indices: LLM layer indices to hook.  Should be
                           **full_attention** layers (e.g. [7, 15, 23]).
        spatial_merge_size: backbone spatial merge factor (default 2).
        restore_vit_spatial_layout: undo Qwen visual window/merge packing.
                           Defaults to ``False`` for legacy-checkpoint
                           compatibility.
    """

    def __init__(
        self,
        model,
        vit_layer_indices: list[int],
        llm_layer_indices: list[int] | None = None,
        spatial_merge_size: int = 2,
        detach_features: bool = True,
        history_query_source: str = HISTORY_QUERY_TEXT_ANCHOR,
        restore_vit_spatial_layout: bool = False,
    ):
        if llm_layer_indices is None:
            llm_layer_indices = [7, 15, 23]

        self.vit_features: dict[int, torch.Tensor] = {}
        self.llm_hidden_states: dict[int, torch.Tensor | None] = {}
        self.vit_layer_indices = list(vit_layer_indices)
        self.llm_layer_indices = sorted(llm_layer_indices)
        self.spatial_merge_size = spatial_merge_size
        self.detach_features = detach_features
        self.restore_vit_spatial_layout = bool(restore_vit_spatial_layout)
        self.history_query_source = str(history_query_source).strip().lower()
        if self.history_query_source not in HISTORY_QUERY_SOURCES:
            raise ValueError(
                f"history_query_source must be one of {sorted(HISTORY_QUERY_SOURCES)}, got {history_query_source!r}"
            )
        self._handles: list = []
        self._batch_capture_plan = None
        self._captured_batch_vit: dict[int, list[dict[int, torch.Tensor]]] = {}
        self._captured_batch_llm: dict[int, list[dict[int, torch.Tensor]]] = {}
        self._captured_batch_queries: list[list[torch.Tensor]] | None = None
        self._capture_suspend_depth = 0
        self._llm_resize_logged = False
        self._vit_resize_logged = False

        visual = self._get_visual_module(model)
        self._vit_window_index_fn = getattr(visual, "get_window_index", None)
        num_blocks = len(visual.blocks)
        for idx in self.vit_layer_indices:
            if idx >= num_blocks:
                logger.warning(
                    "ViT hook block %d out of range (max %d), skipping",
                    idx,
                    num_blocks - 1,
                )
                continue
            h = visual.blocks[idx].register_forward_hook(self._make_vit_hook(idx))
            self._handles.append(h)

        llm_layers = self._get_llm_layers(model)
        for layer_idx in self.llm_layer_indices:
            if layer_idx < len(llm_layers):
                h = llm_layers[layer_idx].register_forward_hook(
                    self._make_llm_hook(layer_idx),
                )
                self._handles.append(h)
            else:
                logger.warning(
                    "LLM hook layer %d out of range (max %d)",
                    layer_idx,
                    len(llm_layers) - 1,
                )

        logger.info(
            "FeatureExtractor: ViT hooks %s, LLM hooks %s, detach=%s, "
            "history_query_source=%s, restore_vit_spatial_layout=%s",
            self.vit_layer_indices,
            self.llm_layer_indices,
            self.detach_features,
            self.history_query_source,
            self.restore_vit_spatial_layout,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _maybe_detach(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach() if self.detach_features else tensor

    def _build_vit_spatial_layouts(
        self,
        image_grid_thw: torch.Tensor | None,
        *,
        expected_images: int | None = None,
    ) -> list[_ViTSpatialLayout]:
        """Build exact inverse layouts from Qwen's own window-index routine.

        Qwen packs every ``spatial_merge_size x spatial_merge_size`` patch
        group contiguously, then reorders those groups by ``window_index``
        before executing any visual block.  Block hooks therefore observe
        window-ordered, merge-packed tokens rather than a raster grid.
        """

        if image_grid_thw is None:
            raise RuntimeError(
                "image_grid_thw is required to restore Qwen visual-block features"
            )
        if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
            raise RuntimeError(
                "image_grid_thw must have shape [N,3] to restore Qwen "
                f"visual-block features, got {tuple(image_grid_thw.shape)}"
            )
        num_images = int(image_grid_thw.shape[0])
        if expected_images is not None and num_images != expected_images:
            raise RuntimeError(
                "Qwen visual-grid/image count mismatch: "
                f"grid has {num_images} rows, expected {expected_images}"
            )
        if num_images == 0:
            return []
        if not callable(self._vit_window_index_fn):
            raise RuntimeError(
                "Qwen visual module does not expose get_window_index; "
                "cannot safely restore visual-block token coordinates"
            )

        merge_size = int(self.spatial_merge_size)
        if merge_size <= 0:
            raise RuntimeError(
                f"spatial_merge_size must be positive, got {merge_size}"
            )

        grid_rows: list[tuple[int, int, int]] = []
        group_counts: list[int] = []
        for image_idx, row in enumerate(image_grid_thw.detach().cpu().tolist()):
            grid_t, grid_h, grid_w = (int(value) for value in row)
            if min(grid_t, grid_h, grid_w) <= 0:
                raise RuntimeError(
                    f"image_grid_thw row {image_idx} has non-positive grid "
                    f"{(grid_t, grid_h, grid_w)}"
                )
            if grid_h % merge_size or grid_w % merge_size:
                raise RuntimeError(
                    f"image_grid_thw row {image_idx} grid "
                    f"{(grid_t, grid_h, grid_w)} is not divisible by "
                    f"spatial_merge_size={merge_size}"
                )
            grid_rows.append((grid_t, grid_h, grid_w))
            group_counts.append(
                grid_t
                * (grid_h // merge_size)
                * (grid_w // merge_size)
            )

        window_result = self._vit_window_index_fn(image_grid_thw)
        window_index = (
            window_result[0]
            if isinstance(window_result, (tuple, list))
            else window_result
        )
        if not torch.is_tensor(window_index):
            raise RuntimeError(
                "Qwen visual get_window_index returned a non-tensor index"
            )
        window_index = window_index.detach().reshape(-1).to(dtype=torch.long)
        expected_groups = sum(group_counts)
        if int(window_index.numel()) != expected_groups:
            raise RuntimeError(
                "Qwen visual window index has unexpected length: "
                f"got {int(window_index.numel())}, expected {expected_groups}"
            )

        layouts: list[_ViTSpatialLayout] = []
        group_offset = 0
        for image_idx, ((grid_t, grid_h, grid_w), group_count) in enumerate(
            zip(grid_rows, group_counts)
        ):
            local_window_index = (
                window_index[group_offset : group_offset + group_count]
                - group_offset
            )
            expected_local = torch.arange(
                group_count,
                device=local_window_index.device,
                dtype=local_window_index.dtype,
            )
            if not torch.equal(
                torch.sort(local_window_index).values,
                expected_local,
            ):
                raise RuntimeError(
                    "Qwen visual window index does not preserve a contiguous "
                    f"per-image token segment for image {image_idx}"
                )
            layouts.append(
                _ViTSpatialLayout(
                    grid_t=grid_t,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    reverse_window_indices=torch.argsort(local_window_index),
                )
            )
            group_offset += group_count
        return layouts

    def _restore_vit_tokens(
        self,
        tokens: torch.Tensor,
        layout: _ViTSpatialLayout,
        *,
        layer_idx: int,
        image_idx: int,
    ) -> torch.Tensor:
        """Undo Qwen window order and merge packing into ``[T,H,W,C]``."""

        if tokens.ndim != 2:
            raise RuntimeError(
                f"ViT layer {layer_idx} image {image_idx} must have token "
                f"shape [N,C], got {tuple(tokens.shape)}"
            )
        expected_tokens = layout.grid_t * layout.grid_h * layout.grid_w
        if int(tokens.shape[0]) != expected_tokens:
            raise RuntimeError(
                f"ViT layer {layer_idx} image {image_idx} produced "
                f"{int(tokens.shape[0])} tokens, but grid "
                f"{(layout.grid_t, layout.grid_h, layout.grid_w)} requires "
                f"{expected_tokens}"
            )

        merge_size = int(self.spatial_merge_size)
        merge_area = merge_size**2
        num_groups = expected_tokens // merge_area
        grouped = tokens.reshape(num_groups, merge_area, tokens.shape[-1])
        reverse_indices = layout.reverse_window_indices.to(
            device=tokens.device,
            dtype=torch.long,
        )
        grouped = grouped.index_select(0, reverse_indices)

        raster = grouped.reshape(
            layout.grid_t,
            layout.grid_h // merge_size,
            layout.grid_w // merge_size,
            merge_size,
            merge_size,
            tokens.shape[-1],
        )
        raster = raster.permute(0, 1, 3, 2, 4, 5).reshape(
            layout.grid_t,
            layout.grid_h,
            layout.grid_w,
            tokens.shape[-1],
        )
        return raster

    def _validate_history_visual_occurrences(
        self,
        image_token_positions: dict[int, tuple[int, int]],
        text_anchor_positions: dict[int, int],
        image_grid_thw: torch.Tensor | None,
        *,
        tag: str,
    ) -> list[list[tuple[int, int]]]:
        """Validate and return ``[history][view]`` layer-token ranges.

        The heatmap prompt has four current-image occurrences followed by four
        image occurrences for each history.  The numbered text anchor for a
        history must occur after that history's four images and before the next
        history's images.  This method deliberately derives no identity from
        the anchor; it uses the anchor only as a fail-closed layout boundary.
        """

        history_keys = set(text_anchor_positions)
        expected_history_keys = set(range(len(text_anchor_positions)))
        if history_keys != expected_history_keys:
            raise RuntimeError(
                f"{tag}: history anchors must have dense keys "
                f"{sorted(expected_history_keys)}, got {sorted(history_keys)}"
            )
        num_histories = len(text_anchor_positions)
        expected_occurrences = 4 + 4 * num_histories
        expected_keys = set(range(expected_occurrences))
        actual_keys = set(image_token_positions)
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"{tag}: visual-query occurrence mapping requires exactly "
                f"{expected_occurrences} image occurrences with keys "
                f"{sorted(expected_keys)}, got {sorted(actual_keys)}"
            )

        if image_grid_thw is None:
            raise RuntimeError(f"{tag}: image_grid_thw is required to validate visual-query spans")
        if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
            raise RuntimeError(f"{tag}: image_grid_thw must have shape [N,3], got {tuple(image_grid_thw.shape)}")
        if int(image_grid_thw.shape[0]) != expected_occurrences:
            raise RuntimeError(
                f"{tag}: expected {expected_occurrences} image-grid rows, got {int(image_grid_thw.shape[0])}"
            )
        if self.spatial_merge_size <= 0:
            raise RuntimeError(f"{tag}: spatial_merge_size must be positive, got {self.spatial_merge_size}")

        previous_end = -1
        checked_ranges: dict[int, tuple[int, int]] = {}
        grid_rows = image_grid_thw.detach().cpu().tolist()
        merge_area = int(self.spatial_merge_size) ** 2
        for occurrence in range(expected_occurrences):
            raw_range = image_token_positions[occurrence]
            if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
                raise RuntimeError(f"{tag}: image occurrence {occurrence} has invalid span {raw_range!r}")
            start, end = int(raw_range[0]), int(raw_range[1])
            if start < 0 or end <= start:
                raise RuntimeError(f"{tag}: image occurrence {occurrence} has empty/invalid span ({start}, {end})")
            if start < previous_end:
                raise RuntimeError(
                    f"{tag}: image occurrence {occurrence} starts at {start} before "
                    f"the preceding occurrence ends at {previous_end}"
                )

            t, h, w = (int(value) for value in grid_rows[occurrence])
            if min(t, h, w) <= 0:
                raise RuntimeError(f"{tag}: image occurrence {occurrence} has non-positive grid {(t, h, w)}")
            premerge_tokens = t * h * w
            if premerge_tokens % merge_area:
                raise RuntimeError(
                    f"{tag}: image occurrence {occurrence} grid {(t, h, w)} is not divisible by merge area {merge_area}"
                )
            expected_tokens = premerge_tokens // merge_area
            actual_tokens = end - start
            if actual_tokens != expected_tokens:
                raise RuntimeError(
                    f"{tag}: image occurrence {occurrence} span has {actual_tokens} "
                    f"tokens, but grid {(t, h, w)} and merge={self.spatial_merge_size} "
                    f"require {expected_tokens}"
                )
            side = int(actual_tokens**0.5)
            if side * side != actual_tokens:
                raise RuntimeError(
                    f"{tag}: image occurrence {occurrence} has non-square LLM span of {actual_tokens} tokens"
                )
            checked_ranges[occurrence] = (start, end)
            previous_end = end

        history_ranges: list[list[tuple[int, int]]] = []
        previous_anchor = -1
        for history_idx in range(num_histories):
            first_occurrence = 4 + history_idx * 4
            views = [checked_ranges[first_occurrence + view_idx] for view_idx in range(4)]
            anchor_position = int(text_anchor_positions[history_idx])
            if anchor_position < views[-1][1]:
                raise RuntimeError(
                    f"{tag}: history {history_idx} anchor at {anchor_position} must "
                    f"follow all four visual spans ending at {views[-1][1]}"
                )
            if previous_anchor >= views[0][0]:
                raise RuntimeError(
                    f"{tag}: history {history_idx} images start at {views[0][0]} "
                    f"before the preceding anchor at {previous_anchor}"
                )
            if history_idx + 1 < num_histories:
                next_start = checked_ranges[first_occurrence + 4][0]
                if anchor_position >= next_start:
                    raise RuntimeError(
                        f"{tag}: history {history_idx} anchor at {anchor_position} "
                        f"must precede the next history images at {next_start}"
                    )
            history_ranges.append(views)
            previous_anchor = anchor_position
        return history_ranges

    def _pool_history_visual_queries(
        self,
        hidden_row: torch.Tensor,
        history_ranges: list[list[tuple[int, int]]],
        *,
        tag: str,
    ) -> list[torch.Tensor]:
        """Equal-view mean pool history visual tokens without breaking autograd."""

        if hidden_row.ndim != 2:
            raise RuntimeError(f"{tag}: hooked LLM row must have shape [S,C], got {tuple(hidden_row.shape)}")
        queries: list[torch.Tensor] = []
        sequence_length = int(hidden_row.shape[0])
        for history_idx, view_ranges in enumerate(history_ranges):
            if len(view_ranges) != 4:
                raise RuntimeError(f"{tag}: history {history_idx} must contain four visual spans")
            view_means = []
            for view_idx, (start, end) in enumerate(view_ranges):
                if start < 0 or end <= start or end > sequence_length:
                    raise RuntimeError(
                        f"{tag}: history {history_idx} view {view_idx} span "
                        f"({start}, {end}) is outside sequence length {sequence_length}"
                    )
                view_means.append(hidden_row[start:end, :].mean(dim=0))
            # Each view contributes exactly one quarter even if processor
            # resizing ever gives views different token counts.
            query = torch.stack(view_means, dim=0).mean(dim=0)
            queries.append(self._maybe_detach(query))
        return queries

    def _make_vit_hook(self, idx: int):
        def hook(_module, _input, output):
            if self._capture_suspend_depth > 0:
                return
            if self._batch_capture_plan is not None:
                captured = []
                for view_ranges in self._batch_capture_plan["vit_ranges_batch"]:
                    sample_views = {
                        view_idx: self._maybe_detach(output[start:end])
                        for view_idx, (start, end) in view_ranges.items()
                    }
                    captured.append(sample_views)
                self._captured_batch_vit[idx] = captured
            else:
                self.vit_features[idx] = self._maybe_detach(output)

        return hook

    def _make_llm_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            if self._capture_suspend_depth > 0:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if self._batch_capture_plan is not None:
                captured = []
                for batch_idx, image_positions in enumerate(self._batch_capture_plan["image_token_positions_batch"]):
                    sample_views = {
                        view_idx: self._maybe_detach(hidden[batch_idx, start:end, :])
                        for view_idx, (start, end) in image_positions.items()
                        if view_idx < 4
                    }
                    captured.append(sample_views)
                self._captured_batch_llm[layer_idx] = captured

                if layer_idx == self._batch_capture_plan["deepest_layer"]:
                    if self.history_query_source == HISTORY_QUERY_TEXT_ANCHOR:
                        self._captured_batch_queries = []
                        for batch_idx, anchors in enumerate(self._batch_capture_plan["text_anchor_positions_batch"]):
                            sample_queries = [
                                self._maybe_detach(hidden[batch_idx, anchors[hist_idx], :])
                                for hist_idx in range(len(anchors))
                            ]
                            self._captured_batch_queries.append(sample_queries)
                    else:
                        history_ranges_batch = self._batch_capture_plan.get("history_visual_ranges_batch")
                        if history_ranges_batch is None:
                            raise RuntimeError("Visual history-query source has no validated occurrence plan")
                        if hidden.ndim != 3 or int(hidden.shape[0]) != len(history_ranges_batch):
                            raise RuntimeError(
                                "Hooked LLM hidden-state batch does not match the visual-query "
                                f"capture plan: hidden={tuple(hidden.shape)} "
                                f"plan_batch={len(history_ranges_batch)}"
                            )
                        self._captured_batch_queries = [
                            self._pool_history_visual_queries(
                                hidden[batch_idx],
                                history_ranges,
                                tag=f"compact-batch[{batch_idx}]-layer[{layer_idx}]",
                            )
                            for batch_idx, history_ranges in enumerate(history_ranges_batch)
                        ]
            else:
                self.llm_hidden_states[layer_idx] = self._maybe_detach(hidden)

        return hook

    def clear(self):
        """Reset captured features before each forward pass."""
        self.vit_features = {}
        self.llm_hidden_states = {}
        self._batch_capture_plan = None
        self._captured_batch_vit = {}
        self._captured_batch_llm = {}
        self._captured_batch_queries = None

    @contextmanager
    def suspend_capture(self) -> Iterator[None]:
        """Temporarily disable all feature hooks without removing them.

        A shared Qwen backbone may run an unrelated forward (for example, an
        LM-rehearsal batch) while the heatmap head remains materialized.  The
        registered hooks must stay installed for the next heatmap batch, but
        they must not retain tensors from that unrelated computation graph.

        Suspension is depth-counted so nested callers remain suspended until
        the outermost context exits.  Cached tensors are cleared on every
        entry and exit, including exceptional exits.
        """
        self._capture_suspend_depth += 1
        self.clear()
        try:
            yield
        finally:
            self.clear()
            self._capture_suspend_depth -= 1

    def prepare_batch_capture(
        self,
        image_token_positions_batch: list[dict[int, tuple[int, int]]],
        text_anchor_positions_batch: list[dict[int, int]],
        image_grid_thw: torch.Tensor | None = None,
    ) -> None:
        """Prepare compact token capture plan for batched panoramic forward."""
        sample_image_counts = [len(pos) for pos in image_token_positions_batch]
        sample_offsets: list[int] = []
        running = 0
        for count in sample_image_counts:
            sample_offsets.append(running)
            running += count

        vit_layouts = (
            self._build_vit_spatial_layouts(
                image_grid_thw,
                expected_images=running,
            )
            if self.vit_layer_indices and self.restore_vit_spatial_layout
            else []
        )
        if image_grid_thw is not None:
            per_image_sizes = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        else:
            per_image_sizes = [TOKENS_PER_IMAGE_VIT] * running

        history_visual_ranges_batch = None
        if self.history_query_source == HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1:
            if len(image_token_positions_batch) != len(text_anchor_positions_batch):
                raise RuntimeError(
                    "Image-position and text-anchor capture batches have different sizes: "
                    f"{len(image_token_positions_batch)} vs {len(text_anchor_positions_batch)}"
                )
            if image_grid_thw is None:
                raise RuntimeError("Visual history-query source requires image_grid_thw in compact capture")
            history_visual_ranges_batch = []
            for batch_idx, (positions, anchors) in enumerate(
                zip(image_token_positions_batch, text_anchor_positions_batch)
            ):
                offset = sample_offsets[batch_idx]
                count = sample_image_counts[batch_idx]
                history_visual_ranges_batch.append(
                    self._validate_history_visual_occurrences(
                        positions,
                        anchors,
                        image_grid_thw[offset : offset + count],
                        tag=f"compact-batch[{batch_idx}]",
                    )
                )
            if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[0]) != running:
                raise RuntimeError(
                    "Compact visual-query image-grid count does not match image occurrences: "
                    f"grid={tuple(image_grid_thw.shape)} occurrences={running}"
                )

        prefix = [0]
        for size in per_image_sizes:
            prefix.append(prefix[-1] + int(size))

        vit_ranges_batch: list[dict[int, tuple[int, int]]] = []
        vit_layouts_batch: list[dict[int, _ViTSpatialLayout]] = []
        for batch_idx, image_offset in enumerate(sample_offsets):
            if sample_image_counts[batch_idx] < 4:
                raise RuntimeError(
                    f"Compact batch item {batch_idx} has "
                    f"{sample_image_counts[batch_idx]} images; four current "
                    "panoramic views are required"
                )
            sample_views = {}
            sample_layouts = {}
            for view_idx in range(4):
                global_img_idx = image_offset + view_idx
                start = prefix[global_img_idx]
                end = prefix[global_img_idx + 1]
                sample_views[view_idx] = (start, end)
                if vit_layouts:
                    sample_layouts[view_idx] = vit_layouts[global_img_idx]
            vit_ranges_batch.append(sample_views)
            vit_layouts_batch.append(sample_layouts)

        self._batch_capture_plan = {
            "image_token_positions_batch": image_token_positions_batch,
            "text_anchor_positions_batch": text_anchor_positions_batch,
            "vit_ranges_batch": vit_ranges_batch,
            "vit_layouts_batch": vit_layouts_batch,
            "deepest_layer": max(self.llm_layer_indices),
            "history_visual_ranges_batch": history_visual_ranges_batch,
        }
        self._captured_batch_vit = {}
        self._captured_batch_llm = {}
        self._captured_batch_queries = None

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        image_token_positions: dict[int, tuple[int, int]],
        text_anchor_positions: dict[int, int],
        image_grid_thw: torch.Tensor | None = None,
    ):
        """
        Group captured features by image / history position.

        Args:
            image_token_positions: ``{img_idx: (start, end)}`` — each image's
                vision-token span in the LLM sequence.
            text_anchor_positions: ``{hist_idx: token_position}`` — last token
                of each "Historical observation N ..." annotation.
            image_grid_thw: ``(num_images, 3)`` from Qwen processor.

        Returns:
            current_vit:  ``{view_idx: {layer: (16,16,C_vit)}}``
            current_llm:  ``{view_idx: {layer: (8,8,C_llm)}}``
            history_queries: list of ``(C_llm,)`` tensors (from deepest hooked layer)
            history_llm_views: list of ``{view_idx: (8,8,C_llm)}`` (same hooked layer)
        """
        self._validate_llm_layers_captured()

        deepest_layer = max(self.llm_layer_indices)
        hidden_deepest = self.llm_hidden_states.get(deepest_layer)
        if hidden_deepest is None:
            raise RuntimeError(f"LLM hidden states for layer {deepest_layer} not captured. Did you run model forward?")

        n_hist = len(text_anchor_positions)
        vit_layouts = (
            self._build_vit_spatial_layouts(
                image_grid_thw,
                expected_images=len(image_token_positions),
            )
            if self.vit_layer_indices and self.restore_vit_spatial_layout
            else []
        )

        # --- current 4 views: multi-layer LLM features (8x8) ---
        current_llm: dict[int, dict[int, torch.Tensor]] = {}
        for view_idx in range(4):
            current_llm[view_idx] = {}
            if view_idx not in image_token_positions:
                raise RuntimeError(
                    f"Current panoramic view {view_idx} missing from image_token_positions. "
                    "Prompt/image layout and image token spans are inconsistent."
                )
            start, end = image_token_positions[view_idx]
            for layer_idx in self.llm_layer_indices:
                hidden = self.llm_hidden_states.get(layer_idx)
                if hidden is None:
                    continue
                tokens = hidden[0, start:end, :]  # (n_tokens, C_llm)
                current_llm[view_idx][layer_idx] = self._reshape_llm_tokens(tokens, layer_idx, view_idx, "current")

        # --- current 4 views: ViT features (16x16, multi-layer) ---
        current_vit: dict[int, dict[int, torch.Tensor]] = {}
        for view_idx in range(4):
            current_vit[view_idx] = {}
            for layer_idx in self.vit_layer_indices:
                vit_out = self.vit_features.get(layer_idx)
                if vit_out is None:
                    continue
                vit_tokens = self._get_vit_for_image(
                    vit_out,
                    view_idx,
                    image_grid_thw,
                )
                if vit_tokens is not None:
                    current_vit[view_idx][layer_idx] = self._reshape_vit_tokens(
                        vit_tokens,
                        layer_idx,
                        view_idx,
                        layout=vit_layouts[view_idx] if vit_layouts else None,
                    )

        # --- history query vectors (from deepest hooked LLM layer) ---
        if self.history_query_source == HISTORY_QUERY_TEXT_ANCHOR:
            history_queries: list[torch.Tensor] = []
            for hist_idx in range(n_hist):
                pos = text_anchor_positions[hist_idx]
                q = hidden_deepest[0, pos, :]  # (C_llm,)
                history_queries.append(q)
        else:
            history_ranges = self._validate_history_visual_occurrences(
                image_token_positions,
                text_anchor_positions,
                image_grid_thw,
                tag="single-extract",
            )
            history_queries = self._pool_history_visual_queries(
                hidden_deepest[0],
                history_ranges,
                tag=f"single-extract-layer[{deepest_layer}]",
            )

        # --- history LLM visual features (same hooked layer, for ablation) ---
        history_llm_views: list[dict[int, torch.Tensor]] = []
        for hist_idx in range(n_hist):
            views: dict[int, torch.Tensor] = {}
            for v in range(4):
                img_idx = 4 + hist_idx * 4 + v
                if img_idx not in image_token_positions:
                    continue
                start, end = image_token_positions[img_idx]
                tokens = hidden_deepest[0, start:end, :]
                views[v] = self._reshape_llm_tokens(tokens, deepest_layer, img_idx, f"history[{hist_idx}]")
            history_llm_views.append(views)

        return current_vit, current_llm, history_queries, history_llm_views

    def extract_batch(
        self,
        image_token_positions_batch: list[dict[int, tuple[int, int]]],
        text_anchor_positions_batch: list[dict[int, int]],
        image_grid_thw: torch.Tensor | None = None,
    ) -> list[tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], list[torch.Tensor]]]:
        """Batched variant of ``extract()`` for panoramic single-chain inputs."""
        if self._batch_capture_plan is not None:
            return self._extract_batch_compact()

        self._validate_llm_layers_captured()

        deepest_layer = max(self.llm_layer_indices)
        hidden_deepest = self.llm_hidden_states.get(deepest_layer)
        if hidden_deepest is None:
            raise RuntimeError(f"LLM hidden states for layer {deepest_layer} not captured. Did you run model forward?")

        sample_image_counts = [len(pos) for pos in image_token_positions_batch]
        sample_image_offsets: list[int] = []
        running = 0
        for count in sample_image_counts:
            sample_image_offsets.append(running)
            running += count

        vit_layouts = (
            self._build_vit_spatial_layouts(
                image_grid_thw,
                expected_images=running,
            )
            if self.vit_layer_indices and self.restore_vit_spatial_layout
            else []
        )

        if self.history_query_source == HISTORY_QUERY_VISUAL_EQUAL_VIEW_MEAN_V1:
            if len(image_token_positions_batch) != len(text_anchor_positions_batch):
                raise RuntimeError(
                    "Image-position and text-anchor extraction batches have different sizes: "
                    f"{len(image_token_positions_batch)} vs {len(text_anchor_positions_batch)}"
                )
            if image_grid_thw is None:
                raise RuntimeError("Visual history-query source requires image_grid_thw in batched extraction")
            if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[0]) != running:
                raise RuntimeError(
                    "Batched visual-query image-grid count does not match image occurrences: "
                    f"grid={tuple(image_grid_thw.shape)} occurrences={running}"
                )
            if hidden_deepest.ndim != 3 or int(hidden_deepest.shape[0]) != len(image_token_positions_batch):
                raise RuntimeError(
                    "Batched visual-query hidden states do not match capture inputs: "
                    f"hidden={tuple(hidden_deepest.shape)} "
                    f"input_batch={len(image_token_positions_batch)}"
                )

        extracted = []
        for batch_idx, image_token_positions in enumerate(image_token_positions_batch):
            n_hist = len(text_anchor_positions_batch[batch_idx])

            current_llm: dict[int, dict[int, torch.Tensor]] = {}
            for view_idx in range(4):
                current_llm[view_idx] = {}
                if view_idx not in image_token_positions:
                    raise RuntimeError(
                        f"Current panoramic view {view_idx} missing from image_token_positions "
                        f"for batch item {batch_idx}."
                    )
                start, end = image_token_positions[view_idx]
                for layer_idx in self.llm_layer_indices:
                    hidden = self.llm_hidden_states.get(layer_idx)
                    if hidden is None:
                        continue
                    tokens = hidden[batch_idx, start:end, :]
                    current_llm[view_idx][layer_idx] = self._reshape_llm_tokens(
                        tokens, layer_idx, view_idx, f"batch[{batch_idx}]-current"
                    )

            current_vit: dict[int, dict[int, torch.Tensor]] = {}
            image_offset = sample_image_offsets[batch_idx]
            for view_idx in range(4):
                current_vit[view_idx] = {}
                global_img_idx = image_offset + view_idx
                for layer_idx in self.vit_layer_indices:
                    vit_out = self.vit_features.get(layer_idx)
                    if vit_out is None:
                        continue
                    vit_tokens = self._get_vit_for_image(vit_out, global_img_idx, image_grid_thw)
                    if vit_tokens is not None:
                        current_vit[view_idx][layer_idx] = self._reshape_vit_tokens(
                            vit_tokens,
                            layer_idx,
                            view_idx,
                            layout=vit_layouts[global_img_idx] if vit_layouts else None,
                        )

            if self.history_query_source == HISTORY_QUERY_TEXT_ANCHOR:
                history_queries: list[torch.Tensor] = []
                for hist_idx in range(n_hist):
                    pos = text_anchor_positions_batch[batch_idx][hist_idx]
                    q = hidden_deepest[batch_idx, pos, :]
                    history_queries.append(q)
            else:
                if image_grid_thw is None:
                    raise RuntimeError("Visual history-query source requires image_grid_thw in batched extraction")
                image_count = sample_image_counts[batch_idx]
                history_ranges = self._validate_history_visual_occurrences(
                    image_token_positions,
                    text_anchor_positions_batch[batch_idx],
                    image_grid_thw[image_offset : image_offset + image_count],
                    tag=f"batch-extract[{batch_idx}]",
                )
                history_queries = self._pool_history_visual_queries(
                    hidden_deepest[batch_idx],
                    history_ranges,
                    tag=f"batch-extract[{batch_idx}]-layer[{deepest_layer}]",
                )

            extracted.append((current_vit, current_llm, history_queries))

        return extracted

    def _extract_batch_compact(
        self,
    ) -> list[tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], list[torch.Tensor]]]:
        self._validate_llm_layers_captured()

        if self._captured_batch_queries is None:
            raise RuntimeError("Deepest-layer history queries were not captured in compact batch mode.")

        batch_size = len(self._captured_batch_queries)
        vit_layouts_batch = self._batch_capture_plan.get("vit_layouts_batch", [])
        if self.restore_vit_spatial_layout and len(vit_layouts_batch) != batch_size:
            raise RuntimeError(
                "Compact ViT capture is missing spatial-layout metadata"
            )
        extracted = []
        for batch_idx in range(batch_size):
            current_llm: dict[int, dict[int, torch.Tensor]] = {view_idx: {} for view_idx in range(4)}
            for layer_idx in self.llm_layer_indices:
                layer_samples = self._captured_batch_llm.get(layer_idx)
                if layer_samples is None:
                    continue
                for view_idx, tokens in layer_samples[batch_idx].items():
                    current_llm[view_idx][layer_idx] = self._reshape_llm_tokens(
                        tokens, layer_idx, view_idx, f"batch[{batch_idx}]-current"
                    )

            current_vit: dict[int, dict[int, torch.Tensor]] = {view_idx: {} for view_idx in range(4)}
            for layer_idx in self.vit_layer_indices:
                layer_samples = self._captured_batch_vit.get(layer_idx)
                if layer_samples is None:
                    continue
                for view_idx, vit_tokens in layer_samples[batch_idx].items():
                    current_vit[view_idx][layer_idx] = self._reshape_vit_tokens(
                        vit_tokens,
                        layer_idx,
                        view_idx,
                        layout=(
                            vit_layouts_batch[batch_idx][view_idx]
                            if self.restore_vit_spatial_layout
                            else None
                        ),
                    )

            extracted.append((current_vit, current_llm, self._captured_batch_queries[batch_idx]))

        return extracted

    def extract_batch_compact_tensors(
        self,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[list[torch.Tensor]]]:
        self._validate_llm_layers_captured()

        if self._captured_batch_queries is None:
            raise RuntimeError("Deepest-layer history queries were not captured in compact batch mode.")

        batch_size = len(self._captured_batch_queries)
        vit_layouts_batch = self._batch_capture_plan.get("vit_layouts_batch", [])
        if self.restore_vit_spatial_layout and len(vit_layouts_batch) != batch_size:
            raise RuntimeError(
                "Compact ViT capture is missing spatial-layout metadata"
            )
        llm_tensors: dict[int, torch.Tensor] = {}
        vit_tensors: dict[int, torch.Tensor] = {}

        for layer_idx in self.llm_layer_indices:
            layer_samples = self._captured_batch_llm.get(layer_idx)
            if layer_samples is None:
                continue
            batch_views = []
            for batch_idx in range(batch_size):
                view_tensors = []
                for view_idx in range(4):
                    tokens = layer_samples[batch_idx][view_idx]
                    view_tensors.append(
                        self._reshape_llm_tokens(tokens, layer_idx, view_idx, f"batch[{batch_idx}]-current")
                    )
                batch_views.append(torch.stack(view_tensors, dim=0))
            llm_tensors[layer_idx] = torch.stack(batch_views, dim=0)

        for layer_idx in self.vit_layer_indices:
            layer_samples = self._captured_batch_vit.get(layer_idx)
            if layer_samples is None:
                continue
            batch_views = []
            for batch_idx in range(batch_size):
                view_tensors = []
                for view_idx in range(4):
                    vit_tokens = layer_samples[batch_idx][view_idx]
                    view_tensors.append(
                        self._reshape_vit_tokens(
                            vit_tokens,
                            layer_idx,
                            view_idx,
                            layout=(
                                vit_layouts_batch[batch_idx][view_idx]
                                if self.restore_vit_spatial_layout
                                else None
                            ),
                        )
                    )
                batch_views.append(torch.stack(view_tensors, dim=0))
            vit_tensors[layer_idx] = torch.stack(batch_views, dim=0)

        return vit_tensors, llm_tensors, self._captured_batch_queries

    def _validate_llm_layers_captured(self) -> None:
        if self._batch_capture_plan is not None:
            missing_layers = [
                layer_idx for layer_idx in self.llm_layer_indices if self._captured_batch_llm.get(layer_idx) is None
            ]
        else:
            missing_layers = [
                layer_idx for layer_idx in self.llm_layer_indices if self.llm_hidden_states.get(layer_idx) is None
            ]
        if missing_layers:
            raise RuntimeError(
                "Missing hooked LLM hidden states for layers "
                f"{missing_layers}. Requested layers={self.llm_layer_indices}. "
                "This usually means the hook indices no longer match the Qwen model layout."
            )

    @staticmethod
    def _resolve_llm_spatial_shape(
        num_tokens: int,
        layer_idx: int,
        image_idx: int,
        tag: str,
    ) -> tuple[int, int]:
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            raise RuntimeError(
                f"LLM layer {layer_idx} image {image_idx} ({tag}) produced {num_tokens} tokens, "
                "which is not a square grid and cannot be reshaped into 8x8-style spatial features."
            )
        return side, side

    def _reshape_llm_tokens(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        image_idx: int,
        tag: str,
    ) -> torch.Tensor:
        """Reshape square LLM image tokens and resize to the expected 8x8 grid."""
        side, _ = self._resolve_llm_spatial_shape(tokens.shape[0], layer_idx, image_idx, tag)
        feat = tokens.reshape(side, side, -1)
        if side == LLM_SPATIAL:
            return feat

        if not self._llm_resize_logged:
            logger.warning(
                "LLM spatial grid is %dx%d instead of %dx%d; resizing hooked image features to the expected shape",
                side,
                side,
                LLM_SPATIAL,
                LLM_SPATIAL,
            )
            self._llm_resize_logged = True

        feat = feat.permute(2, 0, 1).unsqueeze(0)
        feat = F.interpolate(
            feat,
            size=(LLM_SPATIAL, LLM_SPATIAL),
            mode="bilinear",
            align_corners=False,
        )
        return feat.squeeze(0).permute(1, 2, 0)

    def _reshape_vit_tokens(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        image_idx: int,
        *,
        layout: _ViTSpatialLayout | None = None,
    ) -> torch.Tensor:
        """Restore/reshape ViT tokens and resize to the expected 16x16 grid."""
        if self.restore_vit_spatial_layout:
            if layout is None:
                raise RuntimeError(
                    "ViT spatial-layout restoration is enabled but no layout "
                    f"was provided for layer {layer_idx} image {image_idx}"
                )
            raster = self._restore_vit_tokens(
                tokens,
                layout,
                layer_idx=layer_idx,
                image_idx=image_idx,
            )
            if layout.grid_t != 1:
                raise RuntimeError(
                    "Heatmap ViT features require image grid_t=1, got "
                    f"{layout.grid_t} for image {image_idx}"
                )
            feat = raster[0]
            spatial_h, spatial_w = layout.grid_h, layout.grid_w
        else:
            # Legacy compatibility: historical checkpoints were trained with
            # a direct square reshape of block-hook output.
            num_tokens = tokens.shape[0]
            side = int(num_tokens**0.5)
            if side * side != num_tokens:
                raise RuntimeError(
                    f"ViT layer {layer_idx} image {image_idx} produced {num_tokens} tokens, "
                    "which is not a square grid and cannot be reshaped into 16x16-style spatial features."
                )
            feat = tokens.reshape(side, side, -1)
            spatial_h = spatial_w = side

        if spatial_h == VIT_SPATIAL and spatial_w == VIT_SPATIAL:
            return feat

        if not self._vit_resize_logged:
            logger.warning(
                "ViT spatial grid is %dx%d instead of %dx%d; resizing hooked image features to the expected shape",
                spatial_h,
                spatial_w,
                VIT_SPATIAL,
                VIT_SPATIAL,
            )
            self._vit_resize_logged = True

        feat = feat.permute(2, 0, 1).unsqueeze(0)
        feat = F.interpolate(
            feat,
            size=(VIT_SPATIAL, VIT_SPATIAL),
            mode="bilinear",
            align_corners=False,
        )
        return feat.squeeze(0).permute(1, 2, 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_vit_for_image(
        self,
        vit_layer_output: torch.Tensor,
        img_idx: int,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Extract the pre-merge ViT tokens for a single image."""
        if image_grid_thw is not None:
            per_image_sizes = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
            if img_idx >= len(per_image_sizes):
                return None
            start = int(sum(per_image_sizes[:img_idx]))
            end = start + int(per_image_sizes[img_idx])
        else:
            tokens_per_image = TOKENS_PER_IMAGE_VIT
            start = img_idx * tokens_per_image
            end = start + tokens_per_image
        if end > vit_layer_output.shape[0]:
            return None
        return vit_layer_output[start:end]

    @staticmethod
    def _get_visual_module(model):
        """Resolve the vision module regardless of wrapping."""
        candidates = [model]
        if hasattr(model, "base_model"):
            candidates.append(model.base_model)
        if hasattr(model, "model"):
            candidates.append(model.model)

        for node in candidates:
            if hasattr(node, "visual"):
                return node.visual
            if hasattr(node, "model") and hasattr(node.model, "visual"):
                return node.model.visual
        raise RuntimeError("Cannot locate vision model in model hierarchy")

    @staticmethod
    def _get_llm_layers(model):
        """Resolve the list of LLM transformer layers.

        Walks through common Qwen2.5-VL wrapping patterns, checking each
        candidate node for a ``language_model.layers`` or ``layers``
        attribute.
        """
        candidates = [model]
        if hasattr(model, "base_model"):
            candidates.append(model.base_model)
        if hasattr(model, "model"):
            candidates.append(model.model)

        for node in candidates:
            if hasattr(node, "language_model") and hasattr(node.language_model, "layers"):
                return node.language_model.layers
            if hasattr(node, "model"):
                inner = node.model
                if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                    return inner.language_model.layers
                if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                    return inner.model.layers
                if hasattr(inner, "layers"):
                    return inner.layers
            if hasattr(node, "layers"):
                return node.layers

        raise RuntimeError("Cannot locate LLM layers in model hierarchy")
