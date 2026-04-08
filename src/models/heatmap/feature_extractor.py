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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Qwen2.5-VL token-layout constants
TOKENS_PER_IMAGE_VIT = 256   # 16x16 pre-merge
TOKENS_PER_IMAGE_LLM = 64    # 8x8 post-merge (after 2x2 spatial merge)
VIT_SPATIAL = 16
LLM_SPATIAL = 8


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
    """

    def __init__(
        self,
        model,
        vit_layer_indices: List[int],
        llm_layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 2,
    ):
        if llm_layer_indices is None:
            llm_layer_indices = [7, 15, 23]

        self.vit_features: Dict[int, torch.Tensor] = {}
        self.llm_hidden_states: Dict[int, Optional[torch.Tensor]] = {}
        self.vit_layer_indices = list(vit_layer_indices)
        self.llm_layer_indices = sorted(llm_layer_indices)
        self.spatial_merge_size = spatial_merge_size
        self._handles: list = []
        self._batch_capture_plan = None
        self._captured_batch_vit: Dict[int, List[Dict[int, torch.Tensor]]] = {}
        self._captured_batch_llm: Dict[int, List[Dict[int, torch.Tensor]]] = {}
        self._captured_batch_queries: Optional[List[List[torch.Tensor]]] = None
        self._llm_resize_logged = False
        self._vit_resize_logged = False

        visual = self._get_visual_module(model)
        num_blocks = len(visual.blocks)
        for idx in self.vit_layer_indices:
            if idx >= num_blocks:
                logger.warning(
                    "ViT hook block %d out of range (max %d), skipping",
                    idx, num_blocks - 1,
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
                    layer_idx, len(llm_layers) - 1,
                )

        logger.info(
            "FeatureExtractor: ViT hooks %s, LLM hooks %s",
            self.vit_layer_indices, self.llm_layer_indices,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _make_vit_hook(self, idx: int):
        def hook(_module, _input, output):
            if self._batch_capture_plan is not None:
                captured = []
                for view_ranges in self._batch_capture_plan["vit_ranges_batch"]:
                    sample_views = {
                        view_idx: output[start:end].detach()
                        for view_idx, (start, end) in view_ranges.items()
                    }
                    captured.append(sample_views)
                self._captured_batch_vit[idx] = captured
            else:
                self.vit_features[idx] = output.detach()
        return hook

    def _make_llm_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self._batch_capture_plan is not None:
                captured = []
                for batch_idx, image_positions in enumerate(self._batch_capture_plan["image_token_positions_batch"]):
                    sample_views = {
                        view_idx: hidden[batch_idx, start:end, :].detach()
                        for view_idx, (start, end) in image_positions.items()
                        if view_idx < 4
                    }
                    captured.append(sample_views)
                self._captured_batch_llm[layer_idx] = captured

                if layer_idx == self._batch_capture_plan["deepest_layer"]:
                    self._captured_batch_queries = []
                    for batch_idx, anchors in enumerate(self._batch_capture_plan["text_anchor_positions_batch"]):
                        sample_queries = [
                            hidden[batch_idx, anchors[hist_idx], :].detach()
                            for hist_idx in range(len(anchors))
                        ]
                        self._captured_batch_queries.append(sample_queries)
            else:
                self.llm_hidden_states[layer_idx] = hidden.detach()
        return hook

    def clear(self):
        """Reset captured features before each forward pass."""
        self.vit_features = {}
        self.llm_hidden_states = {}
        self._batch_capture_plan = None
        self._captured_batch_vit = {}
        self._captured_batch_llm = {}
        self._captured_batch_queries = None

    def prepare_batch_capture(
        self,
        image_token_positions_batch: List[Dict[int, Tuple[int, int]]],
        text_anchor_positions_batch: List[Dict[int, int]],
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> None:
        """Prepare compact token capture plan for batched panoramic forward."""
        sample_image_counts = [len(pos) for pos in image_token_positions_batch]
        sample_offsets: List[int] = []
        running = 0
        for count in sample_image_counts:
            sample_offsets.append(running)
            running += count

        if image_grid_thw is not None:
            per_image_sizes = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            ).tolist()
        else:
            per_image_sizes = [TOKENS_PER_IMAGE_VIT] * running

        prefix = [0]
        for size in per_image_sizes:
            prefix.append(prefix[-1] + int(size))

        vit_ranges_batch: List[Dict[int, Tuple[int, int]]] = []
        for image_offset in sample_offsets:
            sample_views = {}
            for view_idx in range(4):
                global_img_idx = image_offset + view_idx
                start = prefix[global_img_idx]
                end = prefix[global_img_idx + 1]
                sample_views[view_idx] = (start, end)
            vit_ranges_batch.append(sample_views)

        self._batch_capture_plan = {
            "image_token_positions_batch": image_token_positions_batch,
            "text_anchor_positions_batch": text_anchor_positions_batch,
            "vit_ranges_batch": vit_ranges_batch,
            "deepest_layer": max(self.llm_layer_indices),
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
        image_token_positions: Dict[int, Tuple[int, int]],
        text_anchor_positions: Dict[int, int],
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """
        Group captured features by image / history position.

        Args:
            image_token_positions: ``{img_idx: (start, end)}`` — each image's
                vision-token span in the LLM sequence.
            text_anchor_positions: ``{hist_idx: token_position}`` — last token
                of each "历史位置X..." annotation.
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
            raise RuntimeError(
                f"LLM hidden states for layer {deepest_layer} not captured. "
                "Did you run model forward?"
            )

        n_hist = len(text_anchor_positions)

        # --- current 4 views: multi-layer LLM features (8x8) ---
        current_llm: Dict[int, Dict[int, torch.Tensor]] = {}
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
                current_llm[view_idx][layer_idx] = self._reshape_llm_tokens(
                    tokens, layer_idx, view_idx, "current"
                )

        # --- current 4 views: ViT features (16x16, multi-layer) ---
        current_vit: Dict[int, Dict[int, torch.Tensor]] = {}
        for view_idx in range(4):
            current_vit[view_idx] = {}
            for layer_idx in self.vit_layer_indices:
                vit_out = self.vit_features.get(layer_idx)
                if vit_out is None:
                    continue
                vit_tokens = self._get_vit_for_image(
                    vit_out, view_idx, image_grid_thw,
                )
                if vit_tokens is not None:
                    current_vit[view_idx][layer_idx] = self._reshape_vit_tokens(vit_tokens, layer_idx, view_idx)

        # --- history query vectors (from deepest hooked LLM layer) ---
        history_queries: List[torch.Tensor] = []
        for hist_idx in range(n_hist):
            pos = text_anchor_positions[hist_idx]
            q = hidden_deepest[0, pos, :]  # (C_llm,)
            history_queries.append(q)

        # --- history LLM visual features (same hooked layer, for ablation) ---
        history_llm_views: List[Dict[int, torch.Tensor]] = []
        for hist_idx in range(n_hist):
            views: Dict[int, torch.Tensor] = {}
            for v in range(4):
                img_idx = 4 + hist_idx * 4 + v
                if img_idx not in image_token_positions:
                    continue
                start, end = image_token_positions[img_idx]
                tokens = hidden_deepest[0, start:end, :]
                views[v] = self._reshape_llm_tokens(
                    tokens, deepest_layer, img_idx, f"history[{hist_idx}]"
                )
            history_llm_views.append(views)

        return current_vit, current_llm, history_queries, history_llm_views

    def extract_batch(
        self,
        image_token_positions_batch: List[Dict[int, Tuple[int, int]]],
        text_anchor_positions_batch: List[Dict[int, int]],
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> List[Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]], List[torch.Tensor]]]:
        """Batched variant of ``extract()`` for panoramic single-chain inputs."""
        if self._batch_capture_plan is not None:
            return self._extract_batch_compact()

        self._validate_llm_layers_captured()

        deepest_layer = max(self.llm_layer_indices)
        hidden_deepest = self.llm_hidden_states.get(deepest_layer)
        if hidden_deepest is None:
            raise RuntimeError(
                f"LLM hidden states for layer {deepest_layer} not captured. "
                "Did you run model forward?"
            )

        sample_image_counts = [len(pos) for pos in image_token_positions_batch]
        sample_image_offsets: List[int] = []
        running = 0
        for count in sample_image_counts:
            sample_image_offsets.append(running)
            running += count

        extracted = []
        for batch_idx, image_token_positions in enumerate(image_token_positions_batch):
            n_hist = len(text_anchor_positions_batch[batch_idx])

            current_llm: Dict[int, Dict[int, torch.Tensor]] = {}
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

            current_vit: Dict[int, Dict[int, torch.Tensor]] = {}
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
                            vit_tokens, layer_idx, view_idx
                        )

            history_queries: List[torch.Tensor] = []
            for hist_idx in range(n_hist):
                pos = text_anchor_positions_batch[batch_idx][hist_idx]
                q = hidden_deepest[batch_idx, pos, :]
                history_queries.append(q)

            extracted.append((current_vit, current_llm, history_queries))

        return extracted

    def _extract_batch_compact(
        self,
    ) -> List[Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]], List[torch.Tensor]]]:
        self._validate_llm_layers_captured()

        if self._captured_batch_queries is None:
            raise RuntimeError("Deepest-layer history queries were not captured in compact batch mode.")

        batch_size = len(self._captured_batch_queries)
        extracted = []
        for batch_idx in range(batch_size):
            current_llm: Dict[int, Dict[int, torch.Tensor]] = {view_idx: {} for view_idx in range(4)}
            for layer_idx in self.llm_layer_indices:
                layer_samples = self._captured_batch_llm.get(layer_idx)
                if layer_samples is None:
                    continue
                for view_idx, tokens in layer_samples[batch_idx].items():
                    current_llm[view_idx][layer_idx] = self._reshape_llm_tokens(
                        tokens, layer_idx, view_idx, f"batch[{batch_idx}]-current"
                    )

            current_vit: Dict[int, Dict[int, torch.Tensor]] = {view_idx: {} for view_idx in range(4)}
            for layer_idx in self.vit_layer_indices:
                layer_samples = self._captured_batch_vit.get(layer_idx)
                if layer_samples is None:
                    continue
                for view_idx, vit_tokens in layer_samples[batch_idx].items():
                    current_vit[view_idx][layer_idx] = self._reshape_vit_tokens(
                        vit_tokens, layer_idx, view_idx
                    )

            extracted.append((current_vit, current_llm, self._captured_batch_queries[batch_idx]))

        return extracted

    def extract_batch_compact_tensors(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[List[torch.Tensor]]]:
        self._validate_llm_layers_captured()

        if self._captured_batch_queries is None:
            raise RuntimeError("Deepest-layer history queries were not captured in compact batch mode.")

        batch_size = len(self._captured_batch_queries)
        llm_tensors: Dict[int, torch.Tensor] = {}
        vit_tensors: Dict[int, torch.Tensor] = {}

        for layer_idx in self.llm_layer_indices:
            layer_samples = self._captured_batch_llm.get(layer_idx)
            if layer_samples is None:
                continue
            batch_views = []
            for batch_idx in range(batch_size):
                view_tensors = []
                for view_idx in range(4):
                    tokens = layer_samples[batch_idx][view_idx]
                    view_tensors.append(self._reshape_llm_tokens(
                        tokens, layer_idx, view_idx, f"batch[{batch_idx}]-current"
                    ))
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
                    view_tensors.append(self._reshape_vit_tokens(vit_tokens, layer_idx, view_idx))
                batch_views.append(torch.stack(view_tensors, dim=0))
            vit_tensors[layer_idx] = torch.stack(batch_views, dim=0)

        return vit_tensors, llm_tensors, self._captured_batch_queries

    def _validate_llm_layers_captured(self) -> None:
        if self._batch_capture_plan is not None:
            missing_layers = [
                layer_idx for layer_idx in self.llm_layer_indices
                if self._captured_batch_llm.get(layer_idx) is None
            ]
        else:
            missing_layers = [
                layer_idx for layer_idx in self.llm_layer_indices
                if self.llm_hidden_states.get(layer_idx) is None
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
    ) -> Tuple[int, int]:
        side = int(num_tokens ** 0.5)
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
    ) -> torch.Tensor:
        """Reshape square ViT tokens and resize to the expected 16x16 grid."""
        num_tokens = tokens.shape[0]
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError(
                f"ViT layer {layer_idx} image {image_idx} produced {num_tokens} tokens, "
                "which is not a square grid and cannot be reshaped into 16x16-style spatial features."
            )

        feat = tokens.reshape(side, side, -1)
        if side == VIT_SPATIAL:
            return feat

        if not self._vit_resize_logged:
            logger.warning(
                "ViT spatial grid is %dx%d instead of %dx%d; resizing hooked image features to the expected shape",
                side,
                side,
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
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Extract the pre-merge ViT tokens for a single image."""
        if image_grid_thw is not None:
            per_image_sizes = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            ).tolist()
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
