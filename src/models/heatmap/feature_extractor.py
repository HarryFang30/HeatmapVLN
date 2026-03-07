"""
Feature Extractor for HeatmapVLN
==================================

Registers forward hooks on Qwen3.5-9B to capture:
  1. ViT intermediate-layer features (16x16 per image, pre-merge)
  2. LLM multi-layer hidden states (8x8 per image, post-merge)
  3. Text token hidden states (query vectors for each history position)

Qwen3.5 uses alternating linear_attention and full_attention layers
(full_attention_interval=4).  We hook only **full_attention** layers
(e.g. 7, 15, 23) because they have global cross-token interaction,
producing spatially richer 8x8 visual features.

Reference: HeatmapVLN设计文档 Section 4
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Qwen3.5-9B constants
TOKENS_PER_IMAGE_VIT = 256   # 16x16 pre-merge
TOKENS_PER_IMAGE_LLM = 64    # 8x8 post-merge (after 2x2 spatial merge)
VIT_SPATIAL = 16
LLM_SPATIAL = 8


class FeatureExtractor:
    """
    Hook-based feature extractor for a frozen Qwen3.5-9B.

    After the model forward pass, call ``extract()`` to retrieve
    grouped features for downstream coarse / fine localisation heads.

    Args:
        model:             Qwen3.5 model instance.
        vit_layer_indices: ViT block indices to hook (e.g. [6, 12, 18, 24]).
        llm_layer_indices: LLM layer indices to hook.  Should be
                           **full_attention** layers (e.g. [7, 15, 23]).
        spatial_merge_size: Qwen3.5 spatial merge factor (default 2).
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
            self.vit_features[idx] = output.detach()
        return hook

    def _make_llm_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                self.llm_hidden_states[layer_idx] = output[0].detach()
            else:
                self.llm_hidden_states[layer_idx] = output.detach()
        return hook

    def clear(self):
        """Reset captured features before each forward pass."""
        self.vit_features = {}
        self.llm_hidden_states = {}

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
            history_queries: list of ``(C_llm,)`` tensors (from deepest layer)
            history_llm_views: list of ``{view_idx: (8,8,C_llm)}`` (deepest)
        """
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
            start, end = image_token_positions[view_idx]
            for layer_idx in self.llm_layer_indices:
                hidden = self.llm_hidden_states.get(layer_idx)
                if hidden is None:
                    continue
                tokens = hidden[0, start:end, :]  # (n_tokens, C_llm)
                n = tokens.shape[0]
                h = w = int(n ** 0.5)
                current_llm[view_idx][layer_idx] = tokens.reshape(h, w, -1)

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
                    h = w = int(vit_tokens.shape[0] ** 0.5)
                    current_vit[view_idx][layer_idx] = vit_tokens.reshape(h, w, -1)

        # --- history query vectors (from deepest LLM layer) ---
        history_queries: List[torch.Tensor] = []
        for hist_idx in range(n_hist):
            pos = text_anchor_positions[hist_idx]
            q = hidden_deepest[0, pos, :]  # (C_llm,)
            history_queries.append(q)

        # --- history LLM visual features (deepest layer, for ablation) ---
        history_llm_views: List[Dict[int, torch.Tensor]] = []
        for hist_idx in range(n_hist):
            views: Dict[int, torch.Tensor] = {}
            for v in range(4):
                img_idx = 4 + hist_idx * 4 + v
                if img_idx not in image_token_positions:
                    continue
                start, end = image_token_positions[img_idx]
                tokens = hidden_deepest[0, start:end, :]
                n = tokens.shape[0]
                h = w = int(n ** 0.5)
                views[v] = tokens.reshape(h, w, -1)
            history_llm_views.append(views)

        return current_vit, current_llm, history_queries, history_llm_views

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
        """Resolve the Qwen3_5VisionModel regardless of wrapping."""
        m = model
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model"):
            inner = m.model
            if hasattr(inner, "visual"):
                return inner.visual
            if hasattr(inner, "model") and hasattr(inner.model, "visual"):
                return inner.model.visual
        if hasattr(m, "visual"):
            return m.visual
        raise RuntimeError("Cannot locate vision model in model hierarchy")

    @staticmethod
    def _get_llm_layers(model):
        """Resolve the list of LLM transformer layers."""
        m = model
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model"):
            inner = m.model
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return inner.model.layers
            if hasattr(inner, "layers"):
                return inner.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        raise RuntimeError("Cannot locate LLM layers in model hierarchy")
