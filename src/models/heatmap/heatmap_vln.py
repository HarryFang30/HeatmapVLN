"""
HeatmapVLN — Complete Model Assembly
=======================================

Frozen:    Qwen3.5-9B  (~9B parameters)
Trainable: DPTLiteFusion (ViT) + DPTLiteFusion (LLM)
           + CoarseLocalization (query_proj + vis_head)
           + FineLocalization

Data flow:
    Multi-image + text input  →  Qwen3.5 forward (frozen)
    →  ViT intermediate features (16x16, multi-layer)
       + LLM intermediate features (8x8, multi-layer from full_attention)
       + text hidden states (deepest layer)
    →  DPT-Lite fusion for ViT features   →  (16x16, C_fused)
    →  DPT-Lite fusion for LLM features   →  (8x8,  C_fused)
    →  Coarse localisation (query_proj + cosine sim):
         query_proj(text) × fused_llm  →  visibility + 8x8 coarse heatmap
    →  Fine localisation (trainable):
         fused_vit + coarse heatmap + text query  →  64x64 fine heatmap

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
        qwen_model:         Qwen3.5-9B model instance (will be frozen).
        processor:          Qwen3.5 processor / tokenizer.
        c_vit:              ViT hidden dimension (1152 for Qwen3.5).
        c_llm:              LLM hidden dimension (4096 for Qwen3.5).
        c_fused:            Fused feature dimension for DPT / fine head.
        vit_layer_indices:  ViT block indices to hook.
        llm_layer_indices:  LLM layer indices to hook (full_attention layers).
    """

    def __init__(
        self,
        qwen_model,
        processor,
        c_vit: int = 1152,
        c_llm: int = 4096,
        c_fused: int = 256,
        vit_layer_indices: Optional[List[int]] = None,
        llm_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        if vit_layer_indices is None:
            vit_layer_indices = [6, 12, 18, 24]
        if llm_layer_indices is None:
            llm_layer_indices = [7, 15, 23]

        self.qwen = qwen_model
        self.processor = processor
        self.c_vit = c_vit
        self.c_llm = c_llm
        self.c_fused = c_fused
        self.vit_layer_indices = vit_layer_indices
        self.llm_layer_indices = llm_layer_indices

        # Freeze Qwen3.5
        for param in self.qwen.parameters():
            param.requires_grad = False

        # Feature extractor (hooks, no parameters)
        self.feat_extractor = FeatureExtractor(
            self.qwen, vit_layer_indices, llm_layer_indices,
        )

        # DPT-Lite fusion for ViT 16x16 multi-layer features
        n_vit_layers = len(vit_layer_indices)
        self.vit_dpt_fusion = DPTLiteFusion(c_vit, c_fused, n_vit_layers)

        # DPT-Lite fusion for LLM 8x8 multi-layer features
        n_llm_layers = len(llm_layer_indices)
        self.llm_dpt_fusion = DPTLiteFusion(c_llm, c_fused, n_llm_layers)

        # Coarse localisation (query_proj + vis_head)
        self.coarse = CoarseLocalization(c_llm=c_llm, c_fused=c_fused)

        # Fine localisation head
        self.fine = FineLocalization(c_fused, c_llm)

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "HeatmapVLN: c_vit=%d, c_llm=%d, c_fused=%d, "
            "vit_layers=%s, llm_layers=%s, trainable=%s",
            c_vit, c_llm, c_fused,
            vit_layer_indices, llm_layer_indices,
            f"{trainable:,}",
        )

    # ------------------------------------------------------------------
    # Qwen input / decode helpers
    # ------------------------------------------------------------------

    def prepare_qwen_inputs(
        self,
        current_views: Dict[str, object],
        history_panoramas: List[Dict[str, object]],
        instruction: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Build processor inputs for the panoramic single-chain forward."""
        if device is None:
            device = next(self.fine.parameters()).device

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
        self._normalize_multimodal_inputs(inputs)
        return inputs, len(history_panoramas)

    @staticmethod
    def _normalize_multimodal_inputs(inputs: Dict[str, torch.Tensor]) -> None:
        """Normalize processor outputs to match Qwen's multimodal expectations."""
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            vgt = inputs["video_grid_thw"]
            if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                inputs["video_grid_thw"] = torch.repeat_interleave(
                    vgt, vgt[:, 0], dim=0,
                )
                inputs["video_grid_thw"][:, 0] = 1

    def decode_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        num_history: int,
    ) -> Dict[str, torch.Tensor]:
        """Decode heatmaps from the most recent hooked Qwen forward."""
        device = next(self.fine.parameters()).device

        image_positions = self._find_image_positions(inputs)
        text_anchors = find_text_anchor_positions(
            inputs["input_ids"],
            self.processor.tokenizer,
            num_history=num_history,
        )

        image_grid_thw = inputs.get("image_grid_thw")
        current_vit, current_llm, history_queries, _ = self.feat_extractor.extract(
            image_positions, text_anchors, image_grid_thw,
        )

        if len(history_queries) != num_history:
            raise RuntimeError(
                f"Expected {num_history} history queries, got {len(history_queries)}"
            )

        return self._decode_features(
            current_vit=current_vit,
            current_llm=current_llm,
            history_queries=history_queries,
            num_history=num_history,
            device=device,
        )

    def _decode_features(
        self,
        current_vit: Dict[int, Dict[int, torch.Tensor]],
        current_llm: Dict[int, Dict[int, torch.Tensor]],
        history_queries: List[torch.Tensor],
        num_history: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Run coarse-to-fine decoding from pre-extracted features."""

        # ---- Fuse multi-layer ViT features per view (16x16) ----
        fused_vit: Dict[int, torch.Tensor] = {}
        for view_idx in range(4):
            multi_layer = []
            template_feat = None
            for layer_idx in self.vit_layer_indices:
                feat = current_vit[view_idx].get(layer_idx)
                if feat is None:
                    continue
                feat = feat.permute(2, 0, 1).unsqueeze(0).to(device)
                template_feat = feat
                multi_layer.append(feat)
            if template_feat is not None and len(multi_layer) != len(self.vit_layer_indices):
                multi_layer = []
                for layer_idx in self.vit_layer_indices:
                    feat = current_vit[view_idx].get(layer_idx)
                    if feat is None:
                        multi_layer.append(torch.zeros_like(template_feat))
                        continue
                    feat = feat.permute(2, 0, 1).unsqueeze(0).to(device)
                    multi_layer.append(feat)
            if multi_layer:
                fused_vit[view_idx] = self.vit_dpt_fusion(multi_layer)

        # ---- Fuse multi-layer LLM features per view (8x8) ----
        fused_llm: Dict[int, torch.Tensor] = {}
        for view_idx in range(4):
            multi_layer = []
            template_feat = None
            for layer_idx in self.llm_layer_indices:
                feat = current_llm[view_idx].get(layer_idx)
                if feat is None:
                    continue
                feat = feat.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C_llm, H, W)
                template_feat = feat
                multi_layer.append(feat)
            if template_feat is not None and len(multi_layer) != len(self.llm_layer_indices):
                multi_layer = []
                for layer_idx in self.llm_layer_indices:
                    feat = current_llm[view_idx].get(layer_idx)
                    if feat is None:
                        multi_layer.append(torch.zeros_like(template_feat))
                        continue
                    feat = feat.permute(2, 0, 1).unsqueeze(0).to(device)
                    multi_layer.append(feat)
            if multi_layer:
                fused = self.llm_dpt_fusion(multi_layer)         # (1, C_fused, 8, 8)
                # reshape back to (H, W, C_fused) for CoarseLocalization
                fused_llm[view_idx] = fused.squeeze(0).permute(1, 2, 0)

        # ---- Coarse localisation ----
        coarse_results = self.coarse(fused_llm, history_queries)

        if num_history == 0:
            return {
                "visibility": torch.empty(0, 4, device=device),
                "heatmaps": torch.empty(0, 4, 64, 64, device=device),
            }

        # ---- Fine localisation ----
        all_visibility = []
        all_heatmaps = []
        for hist_idx in range(num_history):
            coarse = coarse_results[hist_idx]
            vis = coarse["visibility"]
            query = history_queries[hist_idx]
            all_visibility.append(vis)

            view_heatmaps = []
            for view_idx in range(4):
                if view_idx in fused_vit:
                    fine_hm = self.fine(
                        vit_fused=fused_vit[view_idx],
                        coarse_heatmap=coarse["coarse_heatmap"][view_idx],
                        query_vector=query,
                    )
                else:
                    fine_hm = torch.zeros(64, 64, device=device)

                if self.training:
                    view_heatmaps.append(fine_hm)
                else:
                    view_heatmaps.append(fine_hm * torch.sigmoid(vis[view_idx]))

            all_heatmaps.append(torch.stack(view_heatmaps))

        return {
            "visibility": torch.stack(all_visibility),
            "heatmaps": torch.stack(all_heatmaps),
        }

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
        device = next(self.fine.parameters()).device
        inputs, num_history = self.prepare_qwen_inputs(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=instruction,
            device=device,
        )

        # === Step 2: Qwen3.5 forward (frozen, no grad) ===
        self.feat_extractor.clear()

        with torch.no_grad():
            self.qwen(**inputs, output_hidden_states=False, return_dict=True)

        return self.decode_from_inputs(inputs, num_history)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _find_image_positions(
        self, inputs: Dict[str, torch.Tensor],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Find start/end positions of each image's vision tokens in the LLM
        input sequence.

        Uses the tokenizer's ``<|image_pad|>`` token ID dynamically so that
        we are robust to tokenizer changes.
        """
        if not hasattr(self, '_image_pad_id'):
            tokenizer = self.processor.tokenizer
            pad_token = "<|image_pad|>"
            self._image_pad_id = tokenizer.convert_tokens_to_ids(pad_token)
            if self._image_pad_id is None:
                logger.warning(
                    "Could not resolve %s from tokenizer, falling back to 248056",
                    pad_token,
                )
                self._image_pad_id = 248056

        image_pad_id = self._image_pad_id
        input_ids = inputs["input_ids"].squeeze().tolist()

        positions: Dict[int, Tuple[int, int]] = {}
        img_idx = 0
        i = 0
        n = len(input_ids)

        while i < n:
            if input_ids[i] == image_pad_id:
                start = i
                while i < n and input_ids[i] == image_pad_id:
                    i += 1
                positions[img_idx] = (start, i)
                img_idx += 1
            else:
                i += 1

        return positions
