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
from typing import Any, Dict, List, Optional, Tuple, Union

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
        self._logged_llm_feature_stats = False

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

    def prepare_qwen_inputs_batch(
        self,
        current_views: Union[torch.Tensor, List[Dict[str, object]]],
        history_panoramas: Union[torch.Tensor, List[List[Dict[str, object]]]],
        instruction: Optional[Union[str, List[Optional[str]]]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """Build one batched processor input for panoramic batch forward."""
        if device is None:
            device = next(self.fine.parameters()).device

        if torch.is_tensor(current_views):
            if current_views.dim() != 5:
                raise ValueError(f"Expected current_views [B, 4, C, H, W], got {tuple(current_views.shape)}")
            current_views_list = [
                self._views_tensor_to_dict(current_views[b])
                for b in range(current_views.shape[0])
            ]
        else:
            current_views_list = current_views

        if torch.is_tensor(history_panoramas):
            if history_panoramas.dim() != 6:
                raise ValueError(
                    f"Expected history_panoramas [B, N, 4, C, H, W], got {tuple(history_panoramas.shape)}"
                )
            history_panoramas_list = [
                self._history_tensor_to_list(history_panoramas[b])
                for b in range(history_panoramas.shape[0])
            ]
        else:
            history_panoramas_list = history_panoramas

        batch_size = len(current_views_list)
        if isinstance(instruction, list):
            instructions = list(instruction)
            if len(instructions) != batch_size:
                raise ValueError(
                    f"Instruction batch size mismatch: got {len(instructions)} for {batch_size} samples"
                )
        else:
            instructions = [instruction] * batch_size

        messages_batch = [
            construct_input(
                current_views=current_views_list[b],
                history_panoramas=history_panoramas_list[b],
                instruction=instructions[b],
            )
            for b in range(batch_size)
        ]
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self._normalize_multimodal_inputs(inputs)
        return inputs, [len(panos) for panos in history_panoramas_list]

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
        self._validate_and_log_current_llm(current_llm)

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

    def decode_from_inputs_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        num_histories: List[int],
        image_positions_batch: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        text_anchors_batch: Optional[List[Dict[int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode batched panoramic inputs from one shared Qwen forward."""
        device = next(self.fine.parameters()).device
        input_ids = inputs["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError(f"Expected batched input_ids [B, S], got {tuple(input_ids.shape)}")

        if image_positions_batch is None:
            image_positions_batch = [
                self._find_image_positions_from_ids(input_ids[b])
                for b in range(input_ids.shape[0])
            ]
        if text_anchors_batch is None:
            text_anchors_batch = [
                find_text_anchor_positions(
                    input_ids[b:b + 1],
                    self.processor.tokenizer,
                    num_history=num_histories[b],
                )
                for b in range(input_ids.shape[0])
            ]

        extracted = self.feat_extractor.extract_batch(
            image_token_positions_batch=image_positions_batch,
            text_anchor_positions_batch=text_anchors_batch,
            image_grid_thw=inputs.get("image_grid_thw"),
        )

        all_visibility = []
        all_heatmaps = []
        for batch_idx, (current_vit, current_llm, history_queries) in enumerate(extracted):
            self._validate_and_log_current_llm(current_llm)
            expected_hist = num_histories[batch_idx]
            if len(history_queries) != expected_hist:
                raise RuntimeError(
                    f"Expected {expected_hist} history queries for batch item {batch_idx}, got {len(history_queries)}"
                )
            decoded = self._decode_features(
                current_vit=current_vit,
                current_llm=current_llm,
                history_queries=history_queries,
                num_history=expected_hist,
                device=device,
            )
            all_visibility.append(decoded["visibility"])
            all_heatmaps.append(decoded["heatmaps"])

        return {
            "visibility": torch.stack(all_visibility, dim=0),
            "heatmaps": torch.stack(all_heatmaps, dim=0),
        }

    def _decode_features(
        self,
        current_vit: Dict[int, Dict[int, torch.Tensor]],
        current_llm: Dict[int, Dict[int, torch.Tensor]],
        history_queries: List[torch.Tensor],
        num_history: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Run coarse-to-fine decoding from pre-extracted features."""
        if num_history == 0:
            return {
                "visibility": torch.empty(0, 4, device=device),
                "heatmaps": torch.empty(0, 4, 64, 64, device=device),
            }
        fused_vit = self._fuse_view_features_batched(
            current_vit, self.vit_layer_indices, self.vit_dpt_fusion, device, output_layout="nchw",
        )
        fused_llm = self._fuse_view_features_batched(
            current_llm, self.llm_layer_indices, self.llm_dpt_fusion, device, output_layout="hwc",
        )
        history_queries_tensor = torch.stack(history_queries, dim=0)

        coarse_results = self.coarse.forward_batched(fused_llm, history_queries_tensor)
        all_visibility = coarse_results["visibility"]
        all_heatmaps = self.fine.forward_batched(
            vit_fused=fused_vit,
            coarse_heatmap=coarse_results["coarse_heatmap"],
            query_vector=history_queries_tensor,
        )
        if not self.training:
            all_heatmaps = all_heatmaps * torch.sigmoid(all_visibility).unsqueeze(-1).unsqueeze(-1)

        return {"visibility": all_visibility, "heatmaps": all_heatmaps}

    @staticmethod
    def _find_first_feature(
        feature_map: Dict[int, Dict[int, torch.Tensor]],
        layer_indices: List[int],
    ) -> Optional[torch.Tensor]:
        for view_idx in range(4):
            for layer_idx in layer_indices:
                feat = feature_map.get(view_idx, {}).get(layer_idx)
                if feat is not None:
                    return feat
        return None

    def _fuse_view_features_batched(
        self,
        feature_map: Dict[int, Dict[int, torch.Tensor]],
        layer_indices: List[int],
        fusion_module: nn.Module,
        device: torch.device,
        output_layout: str,
    ) -> torch.Tensor:
        template = self._find_first_feature(feature_map, layer_indices)
        if template is None:
            raise RuntimeError("No features available for batched DPT fusion.")

        per_layer_batches = []
        for layer_idx in layer_indices:
            view_tensors = []
            for view_idx in range(4):
                feat = feature_map.get(view_idx, {}).get(layer_idx)
                feat_tensor = torch.zeros_like(template) if feat is None else feat
                view_tensors.append(feat_tensor.permute(2, 0, 1).unsqueeze(0).to(device))
            per_layer_batches.append(torch.cat(view_tensors, dim=0))

        fused = fusion_module(per_layer_batches)
        if output_layout == "nchw":
            return fused
        if output_layout == "hwc":
            return fused.permute(0, 2, 3, 1)
        raise ValueError(f"Unsupported output_layout: {output_layout}")

    def _validate_and_log_current_llm(
        self,
        current_llm: Dict[int, Dict[int, torch.Tensor]],
    ) -> None:
        stats_lines = []
        for layer_idx in self.llm_layer_indices:
            layer_feats = []
            missing_views = []
            for view_idx in range(4):
                feat = current_llm.get(view_idx, {}).get(layer_idx)
                if feat is None:
                    missing_views.append(view_idx)
                    continue
                if feat.shape[:2] != (8, 8):
                    raise RuntimeError(
                        f"LLM layer {layer_idx} view {view_idx} has spatial shape {tuple(feat.shape[:2])}, expected (8, 8)."
                    )
                if not torch.isfinite(feat).all():
                    raise RuntimeError(f"LLM layer {layer_idx} view {view_idx} contains non-finite values.")
                layer_feats.append(feat)

            if missing_views:
                raise RuntimeError(
                    f"LLM layer {layer_idx} missing current-view features for views {missing_views}. "
                    "Expected all 4 panoramic views to be present."
                )

            stacked = torch.stack(layer_feats, dim=0)
            abs_mean = float(stacked.abs().mean().item())
            abs_max = float(stacked.abs().max().item())
            std = float(stacked.std().item())
            if abs_max <= 1e-8:
                raise RuntimeError(
                    f"LLM layer {layer_idx} 8x8 features are effectively all-zero after hooking (abs_max={abs_max:.3e})."
                )
            stats_lines.append(
                f"L{layer_idx + 1}: shape={tuple(stacked.shape[1:])}, abs_mean={abs_mean:.3e}, abs_max={abs_max:.3e}, std={std:.3e}"
            )

        if not self._logged_llm_feature_stats and stats_lines:
            logger.info("Captured LLM 8x8 multi-layer features:\n  %s", "\n  ".join(stats_lines))
            self._logged_llm_feature_stats = True

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

        return self._find_image_positions_from_ids(inputs["input_ids"].squeeze().tolist())

    def _find_image_positions_from_ids(
        self, input_ids: Union[torch.Tensor, List[int]],
    ) -> Dict[int, Tuple[int, int]]:
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

        if torch.is_tensor(input_ids):
            input_ids = input_ids.squeeze().tolist()

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

    @staticmethod
    def _views_tensor_to_dict(views: torch.Tensor) -> Dict[str, Any]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(("front", "right", "back", "left"))}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> List[Dict[str, Any]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(
                f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}"
            )
        return [
            self._views_tensor_to_dict(history_panoramas[idx])
            for idx in range(history_panoramas.shape[0])
        ]
