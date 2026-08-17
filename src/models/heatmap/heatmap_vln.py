"""
HeatmapVLN — Current heatmap branch assembly
============================================

Current default implementation:

- Backbone: Qwen2.5-VL integration
- Backbone weights: frozen
- LoRA: optional, enabled by default in InternNav config
- Trainable heatmap modules:
  - DPTLiteFusion (ViT)
  - DPTLiteFusion (LLM)
  - TrajectoryGuidedAttention or CoarseLocalization
  - FineLocalization

Actual default data flow:

    Multi-image + text input
    → Qwen forward (base frozen, optional LoRA)
    → ViT intermediate features (16x16, multi-layer)
      + LLM intermediate features (8x8, multi-layer)
      + text-anchor hidden states (deepest hooked LLM layer)
    → ViT DPT-Lite fusion → (16x16, C_fused)
    → LLM DPT-Lite fusion → (8x8, C_fused)
    → Coarse stage
      - default: TrajectoryGuidedAttention(history_query + rel_pose + spatial tokens)
      - fallback: CoarseLocalization
    → Fine localisation
      - current default: [vit_fused, spatial_out_up, coarse_attn] → 64x64 heatmap
"""

import logging
import time
from typing import Any, Union

import torch
import torch.nn as nn

from .coarse_localization import CoarseLocalization
from .dpt_lite_fusion import DPTLiteFusion
from .feature_extractor import FeatureExtractor
from .fine_localization import FineLocalization
from .input_constructor import construct_input, find_text_anchor_positions
from .pose_free_matching import PoseFreeHistoryMatcher, pad_history_queries
from .trajectory_attention import TrajectoryGuidedAttention

logger = logging.getLogger(__name__)


class HeatmapVLN(nn.Module):
    """
    HeatmapVLN complete model.

    Args:
        qwen_model:         Qwen model instance (base weights will be frozen).
        processor:          Qwen2.5-VL processor / tokenizer.
        c_vit:              ViT hidden dimension (1280 for InternNav Qwen2.5-VL).
        c_llm:              LLM hidden dimension (3584 for InternNav Qwen2.5-VL).
        c_fused:            Fused feature dimension for DPT / fine head.
        vit_layer_indices:  ViT block indices to hook.
        llm_layer_indices:  LLM layer indices to hook (full_attention layers).
    """

    def __init__(
        self,
        qwen_model,
        processor,
        c_vit: int = 1280,
        c_llm: int = 3584,
        c_fused: int = 256,
        vit_layer_indices: list[int] | None = None,
        llm_layer_indices: list[int] | None = None,
        spatial_merge_size: int = 2,
        enable_runtime_timing: bool = False,
        trajectory_config: dict[str, Any] | None = None,
        heatmap_trains_backbone: bool = False,
        decoder_mode: str = "legacy",
        pose_free_config: dict[str, Any] | None = None,
        coarse_logit_residual: bool = False,
        restore_vit_spatial_layout: bool = False,
        joint_panorama_inference: bool = False,
    ):
        super().__init__()

        if vit_layer_indices is None:
            vit_layer_indices = [7, 15, 23, 31]
        if llm_layer_indices is None:
            llm_layer_indices = [6, 13, 20]

        self.qwen = qwen_model
        self.processor = processor
        self.c_vit = c_vit
        self.c_llm = c_llm
        self.c_fused = c_fused
        self.vit_layer_indices = vit_layer_indices
        self.llm_layer_indices = llm_layer_indices
        self.enable_runtime_timing = enable_runtime_timing
        self.heatmap_trains_backbone = heatmap_trains_backbone
        self.restore_vit_spatial_layout = bool(restore_vit_spatial_layout)
        self.joint_panorama_inference = bool(joint_panorama_inference)
        self.decoder_mode = str(decoder_mode).strip().lower()
        if self.decoder_mode not in {"legacy", "pose_free_matcher"}:
            raise ValueError(f"decoder_mode must be 'legacy' or 'pose_free_matcher', got {decoder_mode!r}")
        if self.decoder_mode == "pose_free_matcher":
            # The pose-free matcher consumes only deepest-layer LLM patches.
            # Retaining hooked ViT tensors would keep a large, completely
            # unused autograd graph alive when LoRA training is enabled.
            self.vit_layer_indices = []
        self._logged_llm_feature_stats = False

        traj_cfg = trajectory_config or {}
        pose_cfg = pose_free_config or {}
        allowed_pose_free_keys = {
            "match_dim",
            "heatmap_size",
            "visibility_hidden_dim",
            "logit_temperature",
            "history_query_source",
        }
        unknown_pose_free_keys = set(pose_cfg) - allowed_pose_free_keys
        if unknown_pose_free_keys:
            raise ValueError(f"Unknown pose_free configuration keys: {sorted(unknown_pose_free_keys)}")
        self.history_query_source = str(pose_cfg.get("history_query_source", "text_anchor")).strip().lower()
        self.enable_trajectory = traj_cfg.get("enable", False)
        if self.decoder_mode == "pose_free_matcher" and self.enable_trajectory:
            raise ValueError(
                "pose_free_matcher cannot be combined with trajectory.enable=true; "
                "exact relative pose is forbidden in the diagnostic branch"
            )

        # Freeze backbone base weights. Optional LoRA, if present on the passed
        # model, remains trainable unless the outer training policy freezes it.
        for param in self.qwen.parameters():
            param.requires_grad = False

        # Feature extractor (hooks, no parameters).
        # When heatmap_trains_backbone=True, hooks retain the computation graph
        # so that heatmap loss gradients flow back through the backbone (LoRA).
        self.feat_extractor = FeatureExtractor(
            self.qwen,
            self.vit_layer_indices,
            llm_layer_indices,
            spatial_merge_size=spatial_merge_size,
            detach_features=not heatmap_trains_backbone,
            history_query_source=self.history_query_source,
            restore_vit_spatial_layout=self.restore_vit_spatial_layout,
        )

        self.pose_free_matcher: PoseFreeHistoryMatcher | None = None
        if self.decoder_mode == "pose_free_matcher":
            self.vit_dpt_fusion = None
            self.llm_dpt_fusion = None
            self.coarse = None
            self.fine = None
            self.pose_free_matcher = PoseFreeHistoryMatcher(
                current_dim=c_llm,
                query_dim=c_llm,
                match_dim=pose_cfg.get("match_dim", 64),
                heatmap_size=tuple(pose_cfg.get("heatmap_size", (64, 64))),
                visibility_hidden_dim=pose_cfg.get("visibility_hidden_dim", 16),
                logit_temperature=pose_cfg.get("logit_temperature", 10.0),
            )
            logger.info("HeatmapVLN: using pose-free shared history-query x current-patch matcher")
        else:
            # DPT-Lite fusion for ViT 16x16 multi-layer features
            n_vit_layers = len(vit_layer_indices)
            self.vit_dpt_fusion = DPTLiteFusion(c_vit, c_fused, n_vit_layers)

            # DPT-Lite fusion for LLM 8x8 multi-layer features
            n_llm_layers = len(llm_layer_indices)
            self.llm_dpt_fusion = DPTLiteFusion(c_llm, c_fused, n_llm_layers)

            # Coarse localisation
            if self.enable_trajectory:
                self.coarse = TrajectoryGuidedAttention(
                    c_llm=c_llm,
                    c_fused=c_fused,
                    num_freqs=traj_cfg.get("num_freqs", 16),
                    d_attn=traj_cfg.get("d_attn", c_fused),
                    num_heads=traj_cfg.get("num_heads", 4),
                    num_layers=traj_cfg.get("num_layers", 1),
                    max_spatial_range=traj_cfg.get("max_spatial_range", 10.0),
                )
                logger.info("HeatmapVLN: using TrajectoryGuidedAttention (replacing CoarseLocalization)")
            else:
                self.coarse = CoarseLocalization(c_llm=c_llm, c_fused=c_fused)

            # Fine localisation head (no longer needs c_llm — uses spatial_out from coarse)
            self.fine = FineLocalization(
                c_fused,
                coarse_logit_residual=coarse_logit_residual,
            )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "HeatmapVLN: c_vit=%d, c_llm=%d, c_fused=%d, "
            "vit_layers=%s, llm_layers=%s, enable_trajectory=%s, "
            "heatmap_trains_backbone=%s, decoder_mode=%s, history_query_source=%s, "
            "restore_vit_spatial_layout=%s, joint_panorama_inference=%s, trainable=%s",
            c_vit,
            c_llm,
            c_fused,
            self.vit_layer_indices,
            llm_layer_indices,
            self.enable_trajectory,
            self.heatmap_trains_backbone,
            self.decoder_mode,
            self.history_query_source,
            self.restore_vit_spatial_layout,
            self.joint_panorama_inference,
            f"{trainable:,}",
        )

    # ------------------------------------------------------------------
    # Qwen input / decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_for_timing(device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _decoder_device(self) -> torch.device:
        if self.decoder_mode == "pose_free_matcher":
            if self.pose_free_matcher is None:
                raise RuntimeError("pose_free_matcher decoder was not initialized")
            return next(self.pose_free_matcher.parameters()).device
        if self.fine is None:
            raise RuntimeError("legacy fine decoder was not initialized")
        return next(self.fine.parameters()).device

    def _reject_pose_for_pose_free(
        self,
        history_rel_poses: torch.Tensor | None,
    ) -> None:
        if self.decoder_mode == "pose_free_matcher" and history_rel_poses is not None:
            raise ValueError(
                "pose_free_matcher received non-None history_rel_poses; "
                "the visual-grounding diagnostic fails closed when exact pose is supplied"
            )

    def prepare_qwen_inputs(
        self,
        current_views: dict[str, object],
        history_panoramas: list[dict[str, object]],
        instruction: str | None = None,
        device: torch.device | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Build processor inputs for the panoramic single-chain forward."""
        if device is None:
            device = self._decoder_device()

        messages = construct_input(
            current_views,
            history_panoramas,
            instruction=instruction,
            heatmap_layout=True,
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
        current_views: Union[torch.Tensor, list[dict[str, object]]],
        history_panoramas: Union[torch.Tensor, list[list[dict[str, object]]]],
        instruction: Union[str, list[str | None]] | None = None,
        device: torch.device | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Build one batched processor input for panoramic batch forward."""
        if device is None:
            device = self._decoder_device()

        if torch.is_tensor(current_views):
            if current_views.dim() != 5:
                raise ValueError(f"Expected current_views [B, 4, C, H, W], got {tuple(current_views.shape)}")
            current_views_list = [self._views_tensor_to_dict(current_views[b]) for b in range(current_views.shape[0])]
        else:
            current_views_list = current_views

        if torch.is_tensor(history_panoramas):
            if history_panoramas.dim() != 6:
                raise ValueError(f"Expected history_panoramas [B, N, 4, C, H, W], got {tuple(history_panoramas.shape)}")
            history_panoramas_list = [
                self._history_tensor_to_list(history_panoramas[b]) for b in range(history_panoramas.shape[0])
            ]
        else:
            history_panoramas_list = history_panoramas

        batch_size = len(current_views_list)
        if isinstance(instruction, list):
            instructions = list(instruction)
            if len(instructions) != batch_size:
                raise ValueError(f"Instruction batch size mismatch: got {len(instructions)} for {batch_size} samples")
        else:
            instructions = [instruction] * batch_size

        messages_batch = [
            construct_input(
                current_views=current_views_list[b],
                history_panoramas=history_panoramas_list[b],
                instruction=instructions[b],
                heatmap_layout=True,
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
    def _normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]) -> None:
        """Normalize processor outputs to match Qwen's multimodal expectations."""
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            vgt = inputs["video_grid_thw"]
            if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                inputs["video_grid_thw"] = torch.repeat_interleave(
                    vgt,
                    vgt[:, 0],
                    dim=0,
                )
                inputs["video_grid_thw"][:, 0] = 1

    def decode_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        num_history: int,
        history_rel_poses: torch.Tensor | None = None,
        *,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode heatmaps from the most recent hooked Qwen forward."""
        device = self._decoder_device()

        image_positions = self._find_image_positions(inputs)
        text_anchors = find_text_anchor_positions(
            inputs["input_ids"],
            self.processor.tokenizer,
            num_history=num_history,
        )

        image_grid_thw = inputs.get("image_grid_thw")
        current_vit, current_llm, history_queries, _ = self.feat_extractor.extract(
            image_positions,
            text_anchors,
            image_grid_thw,
        )
        self._validate_and_log_current_llm(current_llm)

        if len(history_queries) != num_history:
            raise RuntimeError(f"Expected {num_history} history queries, got {len(history_queries)}")

        return self._decode_features(
            current_vit=current_vit,
            current_llm=current_llm,
            history_queries=history_queries,
            num_history=num_history,
            device=device,
            history_rel_poses=history_rel_poses,
            return_memory_tokens=return_memory_tokens,
        )

    def decode_from_inputs_batch(
        self,
        inputs: dict[str, torch.Tensor],
        num_histories: list[int],
        image_positions_batch: list[dict[int, tuple[int, int]]] | None = None,
        text_anchors_batch: list[dict[int, int]] | None = None,
        history_rel_poses: torch.Tensor | None = None,
        *,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode batched panoramic inputs from one shared Qwen forward."""
        device = self._decoder_device()
        input_ids = inputs["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError(f"Expected batched input_ids [B, S], got {tuple(input_ids.shape)}")

        if image_positions_batch is None:
            image_positions_batch = [
                self._find_image_positions_from_ids(input_ids[b]) for b in range(input_ids.shape[0])
            ]
        if text_anchors_batch is None:
            text_anchors_batch = [
                find_text_anchor_positions(
                    input_ids[b : b + 1],
                    self.processor.tokenizer,
                    num_history=num_histories[b],
                )
                for b in range(input_ids.shape[0])
            ]

        if self.feat_extractor._batch_capture_plan is not None:
            vit_tensors, llm_tensors, history_queries_batch = self.feat_extractor.extract_batch_compact_tensors()
            return self._decode_feature_tensors_batch(
                vit_layer_tensors=vit_tensors,
                llm_layer_tensors=llm_tensors,
                history_queries_batch=history_queries_batch,
                num_histories=num_histories,
                device=device,
                history_rel_poses=history_rel_poses,
                return_memory_tokens=return_memory_tokens,
            )

        extracted = self.feat_extractor.extract_batch(
            image_token_positions_batch=image_positions_batch,
            text_anchor_positions_batch=text_anchors_batch,
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        return self._decode_features_batch(
            extracted=extracted,
            num_histories=num_histories,
            device=device,
            history_rel_poses=history_rel_poses,
            return_memory_tokens=return_memory_tokens,
        )

    def _decode_features(
        self,
        current_vit: dict[int, dict[int, torch.Tensor]],
        current_llm: dict[int, dict[int, torch.Tensor]],
        history_queries: list[torch.Tensor],
        num_history: int,
        device: torch.device,
        history_rel_poses: torch.Tensor | None = None,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run coarse-to-fine decoding from pre-extracted features."""
        self._reject_pose_for_pose_free(history_rel_poses)
        self._validate_memory_request(return_memory_tokens)
        if self.decoder_mode == "pose_free_matcher":
            deepest_layer = max(self.llm_layer_indices)
            try:
                current_patches = torch.stack(
                    [current_llm[view_idx][deepest_layer] for view_idx in range(4)],
                    dim=0,
                ).unsqueeze(0)
            except KeyError as exc:
                raise RuntimeError(
                    f"Pose-free decoder requires current LLM patches from layer {deepest_layer} for all four views"
                ) from exc
            if history_queries:
                history_queries_tensor = torch.stack(history_queries, dim=0).unsqueeze(0)
            else:
                history_queries_tensor = current_patches.new_empty((1, 0, self.c_llm))
            history_mask = torch.ones(
                (1, num_history),
                dtype=torch.bool,
                device=current_patches.device,
            )
            result = self._run_pose_free_matcher(
                current_patches=current_patches,
                history_queries=history_queries_tensor,
                history_mask=history_mask,
            )
            return {
                key: value.squeeze(0) if torch.is_tensor(value) and value.shape[:1] == (1,) else value
                for key, value in result.items()
            }
        if num_history == 0:
            result = {
                "visibility": torch.empty(0, 4, device=device),
                "heatmaps": torch.empty(0, 4, 64, 64, device=device),
                "heatmap_logits": torch.empty(0, 4, 64, 64, device=device),
            }
            if return_memory_tokens:
                fused_vit = self._fuse_view_features_batched(
                    current_vit,
                    self.vit_layer_indices,
                    self.vit_dpt_fusion,
                    device,
                    output_layout="nchw",
                )
                result.update(
                    history_memory=fused_vit.new_empty((1, 0, self.c_fused)),
                    history_memory_mask=torch.empty(
                        1, 0, dtype=torch.bool, device=device
                    ),
                    history_spatial_memory=fused_vit.new_empty(
                        (1, 0, 4 * 8 * 8, self.c_fused)
                    ),
                    panoramic_vit_features=fused_vit.unsqueeze(0),
                )
            return result
        fused_vit = self._fuse_view_features_batched(
            current_vit,
            self.vit_layer_indices,
            self.vit_dpt_fusion,
            device,
            output_layout="nchw",
        )
        fused_llm = self._fuse_view_features_batched(
            current_llm,
            self.llm_layer_indices,
            self.llm_dpt_fusion,
            device,
            output_layout="hwc",
        )
        history_queries_tensor = torch.stack(history_queries, dim=0)

        if self.enable_trajectory:
            coarse_results = self.coarse(
                fused_llm,
                history_queries_tensor,
                history_rel_poses=history_rel_poses,
            )
        else:
            coarse_results = self.coarse(fused_llm, history_queries_tensor)
        all_visibility = coarse_results["visibility"]
        all_heatmaps, all_heatmap_logits = self.fine(
            vit_fused=fused_vit,
            coarse_heatmap=coarse_results["coarse_heatmap"],
            spatial_out=coarse_results["spatial_out"],
            return_logits=True,
        )
        result = {
            "visibility": all_visibility,
            "heatmaps": all_heatmaps,
            "heatmap_logits": all_heatmap_logits,
        }
        if return_memory_tokens:
            result.update(
                history_memory=coarse_results["history_memory"].unsqueeze(0),
                history_memory_mask=torch.ones(
                    1, num_history, dtype=torch.bool, device=device
                ),
                history_spatial_memory=coarse_results["spatial_out"].unsqueeze(0),
                panoramic_vit_features=fused_vit.unsqueeze(0),
            )
        if not self.training:
            result.update(self._build_inference_heatmaps(result))
        return result

    def _decode_features_batch(
        self,
        extracted: list[
            tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], list[torch.Tensor]]
        ],
        num_histories: list[int],
        device: torch.device,
        history_rel_poses: torch.Tensor | None = None,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._reject_pose_for_pose_free(history_rel_poses)
        self._validate_memory_request(return_memory_tokens)
        batch_size = len(extracted)
        if batch_size == 0:
            result = {
                "visibility": torch.empty(0, 0, 4, device=device),
                "heatmaps": torch.empty(0, 0, 4, 64, 64, device=device),
                "heatmap_logits": torch.empty(0, 0, 4, 64, 64, device=device),
            }
            if return_memory_tokens:
                result.update(
                    history_memory=torch.empty(
                        0, 0, self.c_fused, device=device
                    ),
                    history_memory_mask=torch.empty(
                        0, 0, dtype=torch.bool, device=device
                    ),
                    history_spatial_memory=torch.empty(
                        0, 0, 4 * 8 * 8, self.c_fused, device=device
                    ),
                    panoramic_vit_features=torch.empty(
                        0, 4, self.c_fused, 16, 16, device=device
                    ),
                )
            return result

        max_hist = max(num_histories) if num_histories else 0
        if max_hist == 0:
            result = {
                "visibility": torch.empty(batch_size, 0, 4, device=device),
                "heatmaps": torch.empty(batch_size, 0, 4, 64, 64, device=device),
                "heatmap_logits": torch.empty(batch_size, 0, 4, 64, 64, device=device),
            }
            if return_memory_tokens:
                fused_vit = self._fuse_view_features_multi_batch(
                    feature_maps=[sample[0] for sample in extracted],
                    layer_indices=self.vit_layer_indices,
                    fusion_module=self.vit_dpt_fusion,
                    device=device,
                    output_layout="nchw",
                )
                result.update(
                    history_memory=fused_vit.new_empty(
                        (batch_size, 0, self.c_fused)
                    ),
                    history_memory_mask=torch.empty(
                        batch_size, 0, dtype=torch.bool, device=device
                    ),
                    history_spatial_memory=fused_vit.new_empty(
                        (batch_size, 0, 4 * 8 * 8, self.c_fused)
                    ),
                    panoramic_vit_features=fused_vit,
                )
            return result

        if self.decoder_mode == "pose_free_matcher":
            deepest_layer = max(self.llm_layer_indices)
            current_patch_samples = []
            history_queries_batch = []
            for batch_idx, (_current_vit, current_llm, history_queries) in enumerate(extracted):
                self._validate_and_log_current_llm(current_llm)
                expected_hist = num_histories[batch_idx]
                if len(history_queries) != expected_hist:
                    raise RuntimeError(
                        f"Expected {expected_hist} history queries for batch item {batch_idx}, "
                        f"got {len(history_queries)}"
                    )
                try:
                    current_patch_samples.append(
                        torch.stack(
                            [current_llm[view_idx][deepest_layer] for view_idx in range(4)],
                            dim=0,
                        )
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        f"Pose-free decoder requires current LLM patches from layer {deepest_layer} "
                        f"for all four views in batch item {batch_idx}"
                    ) from exc
                history_queries_batch.append(history_queries)

            history_queries_tensor, history_mask = pad_history_queries(
                history_queries_batch,
                device=device,
            )
            return self._run_pose_free_matcher(
                current_patches=torch.stack(current_patch_samples, dim=0).to(device),
                history_queries=history_queries_tensor,
                history_mask=history_mask,
            )

        current_vit_batch = []
        current_llm_batch = []
        first_query = next(
            (
                query
                for _current_vit, _current_llm, queries in extracted
                for query in queries[:1]
            ),
            None,
        )
        if first_query is None:
            raise RuntimeError(
                "num_histories reports history but no history query was captured"
            )
        history_queries_tensor = torch.zeros(
            batch_size,
            max_hist,
            first_query.shape[-1],
            device=device,
            dtype=first_query.dtype,
        )
        history_mask = torch.zeros(
            batch_size, max_hist, device=device, dtype=torch.bool
        )

        for batch_idx, (current_vit, current_llm, history_queries) in enumerate(extracted):
            self._validate_and_log_current_llm(current_llm)
            expected_hist = num_histories[batch_idx]
            if len(history_queries) != expected_hist:
                raise RuntimeError(
                    f"Expected {expected_hist} history queries for batch item {batch_idx}, got {len(history_queries)}"
                )
            current_vit_batch.append(current_vit)
            current_llm_batch.append(current_llm)

            if history_queries:
                query_stack = torch.stack(history_queries, dim=0).to(device=device, dtype=history_queries_tensor.dtype)
                history_queries_tensor[batch_idx, :expected_hist] = query_stack
                history_mask[batch_idx, :expected_hist] = True

        fused_vit = self._fuse_view_features_multi_batch(
            feature_maps=current_vit_batch,
            layer_indices=self.vit_layer_indices,
            fusion_module=self.vit_dpt_fusion,
            device=device,
            output_layout="nchw",
        )
        fused_llm = self._fuse_view_features_multi_batch(
            feature_maps=current_llm_batch,
            layer_indices=self.llm_layer_indices,
            fusion_module=self.llm_dpt_fusion,
            device=device,
            output_layout="hwc",
        )

        if self.enable_trajectory:
            coarse_results = self.coarse(
                fused_llm,
                history_queries_tensor,
                history_rel_poses=history_rel_poses,
            )
        else:
            coarse_results = self.coarse(fused_llm, history_queries_tensor)
        all_visibility = coarse_results["visibility"]
        all_heatmaps, all_heatmap_logits = self.fine(
            vit_fused=fused_vit,
            coarse_heatmap=coarse_results["coarse_heatmap"],
            spatial_out=coarse_results["spatial_out"],
            return_logits=True,
        )

        history_mask_f = history_mask.to(all_visibility.dtype)
        all_visibility = all_visibility * history_mask_f.unsqueeze(-1)
        all_heatmaps = all_heatmaps * history_mask_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        all_heatmap_logits = (
            all_heatmap_logits
            * history_mask_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        result = {
            "visibility": all_visibility,
            "heatmaps": all_heatmaps,
            "heatmap_logits": all_heatmap_logits,
        }
        if return_memory_tokens:
            result.update(
                history_memory=coarse_results["history_memory"]
                * history_mask_f.unsqueeze(-1),
                history_memory_mask=history_mask,
                history_spatial_memory=coarse_results["spatial_out"]
                * history_mask_f[:, :, None, None],
                panoramic_vit_features=fused_vit,
            )
        if not self.training:
            result.update(self._build_inference_heatmaps(result))
        return result

    def _decode_feature_tensors_batch(
        self,
        vit_layer_tensors: dict[int, torch.Tensor],
        llm_layer_tensors: dict[int, torch.Tensor],
        history_queries_batch: list[list[torch.Tensor]],
        num_histories: list[int],
        device: torch.device,
        history_rel_poses: torch.Tensor | None = None,
        return_memory_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._reject_pose_for_pose_free(history_rel_poses)
        self._validate_memory_request(return_memory_tokens)
        timings: dict[str, float] = {}
        batch_size = len(history_queries_batch)
        if batch_size == 0:
            result = {
                "visibility": torch.empty(0, 0, 4, device=device),
                "heatmaps": torch.empty(0, 0, 4, 64, 64, device=device),
                "heatmap_logits": torch.empty(0, 0, 4, 64, 64, device=device),
            }
            if return_memory_tokens:
                result.update(
                    history_memory=torch.empty(
                        0, 0, self.c_fused, device=device
                    ),
                    history_memory_mask=torch.empty(
                        0, 0, dtype=torch.bool, device=device
                    ),
                    history_spatial_memory=torch.empty(
                        0, 0, 4 * 8 * 8, self.c_fused, device=device
                    ),
                    panoramic_vit_features=torch.empty(
                        0, 4, self.c_fused, 16, 16, device=device
                    ),
                )
            return result

        max_hist = max(num_histories) if num_histories else 0
        if max_hist == 0:
            result = {
                "visibility": torch.empty(batch_size, 0, 4, device=device),
                "heatmaps": torch.empty(batch_size, 0, 4, 64, 64, device=device),
                "heatmap_logits": torch.empty(batch_size, 0, 4, 64, 64, device=device),
            }
            if return_memory_tokens:
                fused_vit = self._fuse_layer_tensor_batch(
                    layer_tensors=vit_layer_tensors,
                    layer_indices=self.vit_layer_indices,
                    fusion_module=self.vit_dpt_fusion,
                    output_layout="nchw",
                )
                result.update(
                    history_memory=fused_vit.new_empty(
                        (batch_size, 0, self.c_fused)
                    ),
                    history_memory_mask=torch.empty(
                        batch_size, 0, dtype=torch.bool, device=device
                    ),
                    history_spatial_memory=fused_vit.new_empty(
                        (batch_size, 0, 4 * 8 * 8, self.c_fused)
                    ),
                    panoramic_vit_features=fused_vit,
                )
            return result

        self._validate_and_log_current_llm_layer_tensors(llm_layer_tensors)

        if self.decoder_mode == "pose_free_matcher":
            for batch_idx, history_queries in enumerate(history_queries_batch):
                expected_hist = num_histories[batch_idx]
                if len(history_queries) != expected_hist:
                    raise RuntimeError(
                        f"Expected {expected_hist} history queries for batch item {batch_idx}, "
                        f"got {len(history_queries)}"
                    )
            history_queries_tensor, history_mask = pad_history_queries(
                history_queries_batch,
                device=device,
            )
            deepest_layer = max(self.llm_layer_indices)
            current_patches = llm_layer_tensors.get(deepest_layer)
            if current_patches is None:
                raise RuntimeError(f"Pose-free decoder requires current LLM patches from layer {deepest_layer}")
            if self.enable_runtime_timing:
                self._sync_for_timing(device)
                t_pose_free0 = time.perf_counter()
            result = self._run_pose_free_matcher(
                current_patches=current_patches.to(device),
                history_queries=history_queries_tensor,
                history_mask=history_mask,
            )
            if self.enable_runtime_timing:
                self._sync_for_timing(device)
                result["timings"] = {
                    "decode_pose_free_matcher_s": time.perf_counter() - t_pose_free0,
                }
            return result

        first_query = next((queries[0] for queries in history_queries_batch if queries), None)
        if first_query is None:
            raise RuntimeError("No history queries found in compact batched decode.")

        history_queries_tensor = torch.zeros(
            batch_size,
            max_hist,
            first_query.shape[-1],
            device=device,
            dtype=first_query.dtype,
        )
        history_mask = torch.zeros(batch_size, max_hist, device=device, dtype=torch.bool)
        for batch_idx, history_queries in enumerate(history_queries_batch):
            expected_hist = num_histories[batch_idx]
            if len(history_queries) != expected_hist:
                raise RuntimeError(
                    f"Expected {expected_hist} history queries for batch item {batch_idx}, got {len(history_queries)}"
                )
            if history_queries:
                query_stack = torch.stack(history_queries, dim=0).to(device=device, dtype=history_queries_tensor.dtype)
                history_queries_tensor[batch_idx, :expected_hist] = query_stack
                history_mask[batch_idx, :expected_hist] = True

        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            t_vit0 = time.perf_counter()
        fused_vit = self._fuse_layer_tensor_batch(
            layer_tensors=vit_layer_tensors,
            layer_indices=self.vit_layer_indices,
            fusion_module=self.vit_dpt_fusion,
            output_layout="nchw",
        )
        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            timings["decode_vit_fusion_s"] = time.perf_counter() - t_vit0

        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            t_llm0 = time.perf_counter()
        fused_llm = self._fuse_layer_tensor_batch(
            layer_tensors=llm_layer_tensors,
            layer_indices=self.llm_layer_indices,
            fusion_module=self.llm_dpt_fusion,
            output_layout="hwc",
        )
        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            timings["decode_llm_fusion_s"] = time.perf_counter() - t_llm0

        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            t_coarse0 = time.perf_counter()
        if self.enable_trajectory:
            coarse_results = self.coarse(
                fused_llm,
                history_queries_tensor,
                history_rel_poses=history_rel_poses,
            )
        else:
            coarse_results = self.coarse(fused_llm, history_queries_tensor)
        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            timings["decode_coarse_s"] = time.perf_counter() - t_coarse0
        all_visibility = coarse_results["visibility"]

        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            t_fine0 = time.perf_counter()
        all_heatmaps, all_heatmap_logits = self.fine(
            vit_fused=fused_vit,
            coarse_heatmap=coarse_results["coarse_heatmap"],
            spatial_out=coarse_results["spatial_out"],
            return_logits=True,
        )
        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            timings["decode_fine_s"] = time.perf_counter() - t_fine0

        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            t_post0 = time.perf_counter()
        history_mask_f = history_mask.to(all_visibility.dtype)
        all_visibility = all_visibility * history_mask_f.unsqueeze(-1)
        all_heatmaps = all_heatmaps * history_mask_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        all_heatmap_logits = (
            all_heatmap_logits
            * history_mask_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        if self.enable_runtime_timing:
            self._sync_for_timing(device)
            timings["decode_post_s"] = time.perf_counter() - t_post0

        result = {
            "visibility": all_visibility,
            "heatmaps": all_heatmaps,
            "heatmap_logits": all_heatmap_logits,
        }
        if return_memory_tokens:
            result.update(
                history_memory=coarse_results["history_memory"]
                * history_mask_f.unsqueeze(-1),
                history_memory_mask=history_mask,
                history_spatial_memory=coarse_results["spatial_out"]
                * history_mask_f[:, :, None, None],
                panoramic_vit_features=fused_vit,
            )
        if not self.training:
            result.update(self._build_inference_heatmaps(result))
        if self.enable_runtime_timing:
            result["timings"] = timings
        return result

    def _run_pose_free_matcher(
        self,
        current_patches: torch.Tensor,
        history_queries: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.pose_free_matcher is None:
            raise RuntimeError("pose_free_matcher decoder was not initialized")
        result = self.pose_free_matcher(
            current_patches=current_patches,
            history_queries=history_queries,
            history_mask=history_mask,
        )
        if not self.training:
            result.update(self._build_inference_heatmaps(result))
        return result

    def _validate_memory_request(self, return_memory_tokens: bool) -> None:
        if return_memory_tokens and (
            self.decoder_mode != "legacy" or not self.enable_trajectory
        ):
            raise RuntimeError(
                "Past -> Plan memory requires the trajectory-guided legacy decoder"
            )

    def _build_inference_heatmaps(
        self,
        result: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build operational probabilities from the same raw logits as training."""
        heatmap_logits = result.get("heatmap_logits")
        if heatmap_logits is None:
            raise RuntimeError(
                "Heatmap inference requires raw heatmap_logits; the decoder "
                "returned only sigmoid probabilities"
            )
        visibility = result["visibility"]
        if self.joint_panorama_inference:
            heatmaps_gated, none_probability = self._joint_panorama_probabilities(
                heatmap_logits,
                visibility,
            )
            return {
                "heatmaps_gated": heatmaps_gated.to(result["heatmaps"].dtype),
                "none_probability": none_probability,
            }
        return {
            "heatmaps_gated": self._gated_softmax_heatmaps(
                result["heatmaps"],
                visibility,
                heatmap_logits=heatmap_logits,
            )
        }

    @staticmethod
    def _joint_panorama_probabilities(
        heatmap_logits: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized ``4*H*W + none`` hierarchical probabilities."""
        logits = heatmap_logits.float()
        height, width = logits.shape[-2:]
        spatial_probability = torch.softmax(
            logits.reshape(*logits.shape[:-2], height * width),
            dim=-1,
        ).reshape_as(logits)

        view_logits = visibility.float()
        none_logit = torch.zeros(
            *view_logits.shape[:-1],
            1,
            device=view_logits.device,
            dtype=view_logits.dtype,
        )
        view_none_probability = torch.softmax(
            torch.cat((none_logit, view_logits), dim=-1),
            dim=-1,
        )
        view_probability = view_none_probability[..., 1:]
        while view_probability.dim() < spatial_probability.dim():
            view_probability = view_probability.unsqueeze(-1)
        joint_probability = spatial_probability * view_probability
        return joint_probability, view_none_probability[..., 0]

    @staticmethod
    def _gated_softmax_heatmaps(
        heatmaps: torch.Tensor,
        visibility: torch.Tensor,
        *,
        heatmap_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce inference-time probability heatmaps.

        1. Convert sigmoid outputs back to logits.
        2. Apply spatial softmax over each view's H×W pixels to get a
           probability distribution — semantically aligned with the
           Softmax CE training objective.
        3. Gate by sigmoid(visibility) to suppress invisible views.

        Returns tensor with same shape as *heatmaps*.
        """
        # Legacy callers can omit logits; corrected training/inference passes
        # the decoder's raw output and avoids sigmoid -> clamp -> logit.
        _H, _W = heatmaps.shape[-2], heatmaps.shape[-1]
        logits = (
            heatmap_logits.float()
            if heatmap_logits is not None
            else torch.logit(heatmaps.float().clamp(1e-6, 1 - 1e-6))
        )
        if logits.shape != heatmaps.shape:
            raise ValueError(
                "heatmap_logits/heatmaps shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(heatmaps.shape)}"
            )
        probs = torch.softmax(logits.reshape(*logits.shape[:-2], -1), dim=-1)
        probs = probs.reshape_as(heatmaps).to(heatmaps.dtype)
        # Visibility gate
        vis_gate = torch.sigmoid(visibility)
        while vis_gate.dim() < probs.dim():
            vis_gate = vis_gate.unsqueeze(-1)
        return probs * vis_gate

    @staticmethod
    def _find_first_feature(
        feature_map: dict[int, dict[int, torch.Tensor]],
        layer_indices: list[int],
    ) -> torch.Tensor | None:
        for view_idx in range(4):
            for layer_idx in layer_indices:
                feat = feature_map.get(view_idx, {}).get(layer_idx)
                if feat is not None:
                    return feat
        return None

    def _find_first_feature_multi_batch(
        self,
        feature_maps: list[dict[int, dict[int, torch.Tensor]]],
        layer_indices: list[int],
    ) -> torch.Tensor | None:
        for feature_map in feature_maps:
            feat = self._find_first_feature(feature_map, layer_indices)
            if feat is not None:
                return feat
        return None

    def _fuse_view_features_batched(
        self,
        feature_map: dict[int, dict[int, torch.Tensor]],
        layer_indices: list[int],
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

    def _fuse_view_features_multi_batch(
        self,
        feature_maps: list[dict[int, dict[int, torch.Tensor]]],
        layer_indices: list[int],
        fusion_module: nn.Module,
        device: torch.device,
        output_layout: str,
    ) -> torch.Tensor:
        template = self._find_first_feature_multi_batch(feature_maps, layer_indices)
        if template is None:
            raise RuntimeError("No features available for multi-sample batched DPT fusion.")

        per_layer_batches = []
        for layer_idx in layer_indices:
            view_tensors = []
            for feature_map in feature_maps:
                for view_idx in range(4):
                    feat = feature_map.get(view_idx, {}).get(layer_idx)
                    feat_tensor = torch.zeros_like(template) if feat is None else feat
                    view_tensors.append(feat_tensor.permute(2, 0, 1).unsqueeze(0).to(device))
            per_layer_batches.append(torch.cat(view_tensors, dim=0))

        fused = fusion_module(per_layer_batches)
        batch_size = len(feature_maps)
        fused = fused.reshape(batch_size, 4, fused.shape[1], fused.shape[2], fused.shape[3])
        if output_layout == "nchw":
            return fused
        if output_layout == "hwc":
            return fused.permute(0, 1, 3, 4, 2)
        raise ValueError(f"Unsupported output_layout: {output_layout}")

    def _fuse_layer_tensor_batch(
        self,
        layer_tensors: dict[int, torch.Tensor],
        layer_indices: list[int],
        fusion_module: nn.Module,
        output_layout: str,
    ) -> torch.Tensor:
        per_layer_batches = []
        batch_size = None
        for layer_idx in layer_indices:
            layer_tensor = layer_tensors.get(layer_idx)
            if layer_tensor is None:
                raise RuntimeError(f"Missing batched tensor for layer {layer_idx}.")
            if layer_tensor.dim() != 5:
                raise RuntimeError(
                    f"Expected layer tensor [B, 4, H, W, C] for layer {layer_idx}, got {tuple(layer_tensor.shape)}"
                )
            if batch_size is None:
                batch_size = layer_tensor.shape[0]
            per_layer_batches.append(
                layer_tensor.permute(0, 1, 4, 2, 3).reshape(
                    layer_tensor.shape[0] * layer_tensor.shape[1],
                    layer_tensor.shape[4],
                    layer_tensor.shape[2],
                    layer_tensor.shape[3],
                )
            )

        fused = fusion_module(per_layer_batches)
        if batch_size is None:
            raise RuntimeError("No layer tensors available for batched DPT fusion.")
        fused = fused.reshape(batch_size, 4, fused.shape[1], fused.shape[2], fused.shape[3])
        if output_layout == "nchw":
            return fused
        if output_layout == "hwc":
            return fused.permute(0, 1, 3, 4, 2)
        raise ValueError(f"Unsupported output_layout: {output_layout}")

    def _validate_and_log_current_llm(
        self,
        current_llm: dict[int, dict[int, torch.Tensor]],
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
            # abs_max is always needed for the safety check.
            abs_max = float(stacked.abs().max().item())
            if abs_max <= 1e-8:
                raise RuntimeError(
                    f"LLM layer {layer_idx} 8x8 features are effectively all-zero after hooking (abs_max={abs_max:.3e})."
                )
            # Only compute abs_mean and std when logging — these incur CUDA syncs.
            if not self._logged_llm_feature_stats:
                abs_mean = float(stacked.abs().mean().item())
                std = float(stacked.std().item())
                stats_lines.append(
                    f"L{layer_idx + 1}: shape={tuple(stacked.shape[1:])}, abs_mean={abs_mean:.3e}, abs_max={abs_max:.3e}, std={std:.3e}"
                )

        if not self._logged_llm_feature_stats and stats_lines:
            logger.info("Captured LLM 8x8 multi-layer features:\n  %s", "\n  ".join(stats_lines))
            self._logged_llm_feature_stats = True

    def _validate_and_log_current_llm_layer_tensors(
        self,
        llm_layer_tensors: dict[int, torch.Tensor],
    ) -> None:
        stats_lines = []
        for layer_idx in self.llm_layer_indices:
            feats = llm_layer_tensors.get(layer_idx)
            if feats is None:
                raise RuntimeError(f"LLM layer {layer_idx} missing from compact batched tensor decode.")
            if feats.shape[1:4] != (4, 8, 8):
                raise RuntimeError(
                    f"LLM layer {layer_idx} batched features have shape {tuple(feats.shape)}, expected [B, 4, 8, 8, C]."
                )
            if not torch.isfinite(feats).all():
                raise RuntimeError(f"LLM layer {layer_idx} batched features contain non-finite values.")
            # abs_max is always needed for the safety check.
            abs_max = float(feats.abs().max().item())
            if abs_max <= 1e-8:
                raise RuntimeError(
                    f"LLM layer {layer_idx} batched 8x8 features are effectively all-zero after hooking (abs_max={abs_max:.3e})."
                )
            # Only compute abs_mean and std when logging — these incur CUDA syncs.
            if not self._logged_llm_feature_stats:
                abs_mean = float(feats.abs().mean().item())
                std = float(feats.std().item())
                stats_lines.append(
                    f"L{layer_idx + 1}: shape={tuple(feats.shape[1:])}, abs_mean={abs_mean:.3e}, abs_max={abs_max:.3e}, std={std:.3e}"
                )

        if not self._logged_llm_feature_stats and stats_lines:
            logger.info("Captured LLM 8x8 multi-layer features:\n  %s", "\n  ".join(stats_lines))
            self._logged_llm_feature_stats = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        current_views: dict[str, object],
        history_panoramas: list[dict[str, object]],
        instruction: str | None = None,
    ) -> dict[str, torch.Tensor]:
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
        device = self._decoder_device()
        inputs, num_history = self.prepare_qwen_inputs(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=instruction,
            device=device,
        )

        # === Step 2: Qwen2.5-VL forward (frozen, no grad) ===
        self.feat_extractor.clear()

        with torch.no_grad():
            self.qwen(**inputs, output_hidden_states=False, return_dict=True)

        return self.decode_from_inputs(inputs, num_history)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _find_image_positions(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[int, tuple[int, int]]:
        """
        Find start/end positions of each image's vision tokens in the LLM
        input sequence.

        Uses the tokenizer's ``<|image_pad|>`` token ID dynamically so that
        we are robust to tokenizer changes.
        """
        if not hasattr(self, "_image_pad_id"):
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
        self,
        input_ids: Union[torch.Tensor, list[int]],
    ) -> dict[int, tuple[int, int]]:
        if not hasattr(self, "_image_pad_id"):
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

        positions: dict[int, tuple[int, int]] = {}
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
    def _views_tensor_to_dict(views: torch.Tensor) -> dict[str, Any]:
        if views.dim() != 4 or views.shape[0] != 4:
            raise ValueError(f"Expected views tensor [4, C, H, W], got {tuple(views.shape)}")
        return {name: views[idx] for idx, name in enumerate(("front", "right", "back", "left"))}

    def _history_tensor_to_list(self, history_panoramas: torch.Tensor) -> list[dict[str, Any]]:
        if history_panoramas.dim() != 5 or history_panoramas.shape[1] != 4:
            raise ValueError(f"Expected history panoramas [N, 4, C, H, W], got {tuple(history_panoramas.shape)}")
        return [self._views_tensor_to_dict(history_panoramas[idx]) for idx in range(history_panoramas.shape[0])]
