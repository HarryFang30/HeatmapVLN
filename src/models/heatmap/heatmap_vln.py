"""
HeatmapVLN — Complete Model Assembly
=======================================

Frozen:    Qwen3.5-9B  (~9B parameters)
Trainable: DPTLiteFusion + FineLocalization  (~2M parameters, 0.02%)

Data flow:
    Multi-image + text input  →  Qwen3.5 forward (frozen)
    →  ViT intermediate features (16x16)
       + LLM intermediate features (8x8)
       + text hidden states
    →  Coarse localisation (zero params):
         text query × current views  →  visibility + 8x8 coarse heatmap
    →  Fine localisation (trainable):
         ViT features + coarse heatmap + text query  →  64x64 fine heatmap

Reference: HeatmapVLN设计文档 Section 7
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .input_constructor import construct_input, find_text_anchor_positions
from .feature_extractor import FeatureExtractor
from .coarse_localization import CoarseLocalization
from .dpt_lite_fusion import DPTLiteFusion
from .fine_localization import FineLocalization

logger = logging.getLogger(__name__)


class HeatmapVLN(nn.Module):
    """
    HeatmapVLN complete model.

    Args:
        qwen_model:        Qwen3.5-9B model instance (will be frozen).
        processor:         Qwen3.5 processor / tokenizer.
        c_vit:             ViT hidden dimension (1152 for Qwen3.5).
        c_llm:             LLM hidden dimension (4096 for Qwen3.5 7B).
        c_fused:           Fused feature dimension for DPT / fine head.
        vit_layer_indices: ViT block indices to hook.
        llm_layer_idx:     LLM layer index to hook.
    """

    def __init__(
        self,
        qwen_model,
        processor,
        c_vit: int = 1152,
        c_llm: int = 4096,
        c_fused: int = 256,
        vit_layer_indices: Optional[List[int]] = None,
        llm_layer_idx: int = 24,
    ):
        super().__init__()

        if vit_layer_indices is None:
            vit_layer_indices = [6, 12, 18, 24]

        self.qwen = qwen_model
        self.processor = processor
        self.c_vit = c_vit
        self.c_llm = c_llm
        self.c_fused = c_fused
        self.vit_layer_indices = vit_layer_indices
        self.llm_layer_idx = llm_layer_idx

        # Freeze Qwen3.5
        for param in self.qwen.parameters():
            param.requires_grad = False

        # Feature extractor (hooks, no parameters)
        self.feat_extractor = FeatureExtractor(
            self.qwen, vit_layer_indices, llm_layer_idx,
        )

        # Coarse localisation (zero parameters)
        self.coarse = CoarseLocalization()

        # Trainable modules
        n_vit_layers = len(vit_layer_indices)
        self.dpt_fusion = DPTLiteFusion(c_vit, c_fused, n_vit_layers)
        self.fine = FineLocalization(c_fused, c_llm)

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "HeatmapVLN: c_vit=%d, c_llm=%d, c_fused=%d, "
            "vit_layers=%s, llm_layer=%d, trainable=%s",
            c_vit, c_llm, c_fused,
            vit_layer_indices, llm_layer_idx,
            f"{trainable:,}",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        current_views: Dict[str, object],
        history_panoramas: List[Dict[str, object]],
        instruction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward pass.

        Args:
            current_views:     dict ``{'front': img, 'right': img, 'back': img, 'left': img}``
            history_panoramas: list of dicts with same structure.
            instruction:       optional navigation instruction.

        Returns:
            dict with keys:
                ``visibility``:  ``(N_hist, 4)``
                ``heatmaps``:    ``(N_hist, 4, 64, 64)``
        """
        N_hist = len(history_panoramas)
        device = next(self.fine.parameters()).device

        # === Step 1: construct multi-image input with text annotations ===
        messages = construct_input(
            current_views,
            history_panoramas,
            instruction=instruction,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Locate image and text-anchor positions
        image_positions = self._find_image_positions(inputs)
        text_anchors = find_text_anchor_positions(
            inputs["input_ids"], self.processor.tokenizer,
        )

        # === Step 2: Qwen3.5 forward (frozen, no grad) ===
        self.feat_extractor.clear()

        # Expand video_grid_thw if needed (same fix as integration.py)
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            vgt = inputs["video_grid_thw"]
            if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                inputs["video_grid_thw"] = torch.repeat_interleave(
                    vgt, vgt[:, 0], dim=0,
                )
                inputs["video_grid_thw"][:, 0] = 1

        with torch.no_grad():
            self.qwen(**inputs, output_hidden_states=False, return_dict=True)

        # === Step 3: extract grouped features ===
        image_grid_thw = inputs.get("image_grid_thw")
        current_vit, current_llm, history_queries, _ = (
            self.feat_extractor.extract(
                image_positions, text_anchors, image_grid_thw,
            )
        )

        # === Step 4: coarse localisation (zero params) ===
        coarse_results = self.coarse(current_llm, history_queries)

        # === Step 5: ViT feature fusion (trainable) ===
        fused_vit: Dict[int, torch.Tensor] = {}
        for view_idx in range(4):
            multi_layer = []
            for layer_idx in self.vit_layer_indices:
                feat = current_vit[view_idx].get(layer_idx)
                if feat is None:
                    continue
                # (H, W, C_vit) -> (1, C_vit, H, W)
                feat = feat.permute(2, 0, 1).unsqueeze(0).to(device)
                multi_layer.append(feat)
            if multi_layer:
                fused_vit[view_idx] = self.dpt_fusion(multi_layer)  # (1, C_fused, H, W)

        # === Step 6: fine localisation (trainable) ===
        all_visibility = []
        all_heatmaps = []

        for hist_idx in range(N_hist):
            coarse = coarse_results[hist_idx]
            vis = coarse["visibility"]            # (4,)
            query = history_queries[hist_idx]      # (C_llm,)

            all_visibility.append(vis)

            view_heatmaps = []
            for view_idx in range(4):
                if view_idx in fused_vit:
                    fine_hm = self.fine(
                        vit_fused=fused_vit[view_idx],
                        coarse_heatmap=coarse["coarse_heatmap"][view_idx],
                        query_vector=query,
                    )  # (64, 64)
                else:
                    fine_hm = torch.zeros(64, 64, device=device)

                gated_hm = fine_hm * torch.sigmoid(vis[view_idx])
                view_heatmaps.append(gated_hm)

            all_heatmaps.append(torch.stack(view_heatmaps))  # (4, 64, 64)

        return {
            "visibility": torch.stack(all_visibility),   # (N_hist, 4)
            "heatmaps": torch.stack(all_heatmaps),       # (N_hist, 4, 64, 64)
        }

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _find_image_positions(
        self, inputs: Dict[str, torch.Tensor],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Find start/end positions of each image's vision tokens in the LLM
        input sequence.

        Qwen3.5 uses ``<|image_pad|>`` (ID 248056) tokens as placeholders.
        Each contiguous block of image_pad tokens corresponds to one image.
        """
        IMAGE_PAD_ID = 248056
        input_ids = inputs["input_ids"].squeeze().tolist()

        positions: Dict[int, Tuple[int, int]] = {}
        img_idx = 0
        i = 0
        n = len(input_ids)

        while i < n:
            if input_ids[i] == IMAGE_PAD_ID:
                start = i
                while i < n and input_ids[i] == IMAGE_PAD_ID:
                    i += 1
                positions[img_idx] = (start, i)
                img_idx += 1
            else:
                i += 1

        return positions
