"""
Direct Heatmap Head - Plan C (v2)
=================================

Single-pass FPN decoder for direct heatmap prediction.

Key improvements over v1:
    1. Global confidence gate: condition-driven bias that starts at -5
       (sigmoid(-5)≈0.007), forcing the model to actively "turn on" pixels.
       Provides an easy mechanism for suppressing output on negative samples.
    2. BCEWithLogitsLoss: replaces sigmoid+MSE. Numerically stable,
       provides strong gradients even at extreme values (no vanishing
       gradient plateau).
    3. Focal weighting: down-weights easy pixels, focuses learning on
       hard examples (peak boundaries, difficult negatives).

Architecture:
    Input:
        - LLM tokens: (B, seq_len, llm_dim)
        - Observation: (B, 3, H, W)

    Processing:
        1. MultiModalConditionEncoder → cond, seq_cond, spatial_features
        2. FPN Decoder: spatial_features + cross-attention → logits
        3. Global confidence gate: cond → scalar bias added to logits
        4. sigmoid(logits) → heatmap in [0, 1]

    Loss:
        - Focal BCEWithLogitsLoss (spatially + sample weighted)
        - Dice loss (shape matching)
        - Peak location loss (soft-argmax position matching)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion.image_encoder import MultiModalConditionEncoder
from .diffusion.unet2d import (
    CrossAttention2D,
    ConditionalResidualBlock2D,
    Upsample2D,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DirectHeatmapConfig:
    """Configuration for DirectHeatmapHead."""

    # ==================== Condition Encoding (shared with diffusion) ==========
    llm_dim: int = 2048
    image_channels: int = 3
    cond_dim: int = 512
    image_size: Tuple[int, int] = (224, 224)
    llm_hidden_dim: int = 1024
    llm_pool_method: str = 'attention'
    llm_pool_num_heads: int = 4
    image_encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    image_encoder_use_pretrained: bool = True
    use_image_encoder: bool = True
    use_sequence_conditioning: bool = True
    seq_cross_attn_heads: int = 8
    seq_cross_attn_head_dim: int = 64
    dropout: float = 0.1

    # ==================== Heatmap Output ======================================
    heatmap_size: Tuple[int, int] = (64, 64)

    # ==================== FPN Decoder =========================================
    hidden_dim: int = 128
    num_decoder_blocks: int = 3
    decoder_num_heads: int = 8
    decoder_head_dim: int = 16

    # ==================== Loss Weights ========================================
    lambda_dice: float = 0.5
    lambda_peak: float = 1.0
    peak_spatial_weight: float = 10.0
    negative_sample_weight: float = 1.0
    positive_sample_boost: float = 3.0
    focal_gamma: float = 2.0

    # ==================== Direction Embedding =================================
    num_directions: int = 4

    # ==================== Initialization ======================================
    init_bias: float = -5.0

    def __post_init__(self):
        assert self.use_image_encoder, (
            "DirectHeatmapHead requires use_image_encoder=True "
            "(spatial features are fundamental to the FPN decoder)"
        )
        assert self.image_encoder_use_pretrained, (
            "DirectHeatmapHead requires image_encoder_use_pretrained=True "
            "(ResNet provides the multi-scale spatial backbone)"
        )


# ============================================================================
# FPN Decoder Block
# ============================================================================

class FPNDecoderBlock(nn.Module):
    """
    Single FPN decoder level: upsample + skip fusion + cross-attention + FiLM conv.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.upsample = Upsample2D(hidden_dim)
        self.fuse_conv = ConditionalResidualBlock2D(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        self.cross_attn = CrossAttention2D(
            channels=hidden_dim,
            cond_dim=cond_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
        seq_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = x + skip
        x = self.cross_attn(x, seq_cond)
        x = self.fuse_conv(x, cond)
        return x


# ============================================================================
# Direct Heatmap Head
# ============================================================================

class DirectHeatmapHead(nn.Module):
    """
    Direct prediction heatmap head with FPN decoder.

    Generates heatmaps in a single forward pass (no diffusion).
    Same interface as DiffusionHeatmapHead for drop-in replacement.
    """

    def __init__(self, config: DirectHeatmapConfig):
        super().__init__()

        self.config = config
        self.heatmap_size = config.heatmap_size

        self._training_step_counter = 0
        self._diag_interval = 100

        # ==================== Condition Encoder (shared) ======================
        self.condition_encoder = MultiModalConditionEncoder(
            llm_dim=config.llm_dim,
            image_channels=config.image_channels,
            cond_dim=config.cond_dim,
            image_encoder_channels=config.image_encoder_channels,
            llm_hidden_dim=config.llm_hidden_dim,
            pool_method=config.llm_pool_method,
            pool_num_heads=config.llm_pool_num_heads,
            image_size=config.image_size,
            dropout=config.dropout,
            use_image_encoder=config.use_image_encoder,
            use_sequence_conditioning=config.use_sequence_conditioning,
            image_encoder_use_pretrained=config.image_encoder_use_pretrained,
        )

        # ==================== Direction Embedding =============================
        self.direction_embedding = nn.Embedding(config.num_directions, config.cond_dim)
        nn.init.zeros_(self.direction_embedding.weight)
        logger.info(
            "DirectionEmbedding: %d directions x %d dim (zero-init)",
            config.num_directions, config.cond_dim,
        )

        # ==================== Spatial Feature Projectors ======================
        from .diffusion.image_encoder import ResNetImageConditionEncoder
        resnet_channels = list(ResNetImageConditionEncoder.CHANNELS)

        self.spatial_projectors = nn.ModuleList()
        for ch in resnet_channels:
            proj = nn.Sequential(
                nn.Conv2d(ch, config.hidden_dim, 1),
                nn.GroupNorm(8, config.hidden_dim),
                nn.SiLU(),
            )
            self.spatial_projectors.append(proj)

        # ==================== FPN Decoder Blocks ==============================
        self.decoder_blocks = nn.ModuleList()
        for _ in range(config.num_decoder_blocks):
            block = FPNDecoderBlock(
                hidden_dim=config.hidden_dim,
                cond_dim=config.cond_dim,
                num_heads=config.decoder_num_heads,
                head_dim=config.decoder_head_dim,
                dropout=config.dropout,
            )
            self.decoder_blocks.append(block)

        # ==================== Output Head =====================================
        self.output_conv = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim // 2, 1, 1),
        )
        # Zero-init the final conv so spatial logits start at 0
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)

        # ==================== Global Confidence Gate ==========================
        # Condition-dependent bias added to ALL spatial logits.
        # Initialized to config.init_bias (default -5) so sigmoid ≈ 0.007.
        # The model must learn to raise this bias for positive samples.
        self.global_gate = nn.Sequential(
            nn.Linear(config.cond_dim, config.cond_dim // 4),
            nn.SiLU(),
            nn.Linear(config.cond_dim // 4, 1),
        )
        nn.init.zeros_(self.global_gate[0].weight)
        nn.init.zeros_(self.global_gate[0].bias)
        nn.init.zeros_(self.global_gate[2].weight)
        nn.init.constant_(self.global_gate[2].bias, config.init_bias)

        # ==================== Loss Config =====================================
        self.peak_spatial_weight = config.peak_spatial_weight
        self.negative_sample_weight = config.negative_sample_weight
        self.positive_sample_boost = config.positive_sample_boost
        self.lambda_dice = config.lambda_dice
        self.lambda_peak = config.lambda_peak
        self.focal_gamma = config.focal_gamma

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DirectHeatmapHead v2: heatmap=%s, cond=%d, hidden=%d, "
            "blocks=%d, init_bias=%.1f, focal_gamma=%.1f, params=%s",
            config.heatmap_size, config.cond_dim, config.hidden_dim,
            config.num_decoder_blocks, config.init_bias, config.focal_gamma,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        llm_tokens: torch.Tensor,
        observation: torch.Tensor,
        gt_heatmap: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        skip_inference: bool = False,
        direction_indices: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if llm_tokens.dim() > 3:
            B = llm_tokens.shape[0]
            D = llm_tokens.shape[-1]
            llm_tokens = llm_tokens.reshape(B, -1, D)

        cond, seq_cond, spatial_features = self.condition_encoder.forward_with_spatial(
            llm_tokens, observation
        )

        if direction_indices is not None:
            dir_emb = self.direction_embedding(direction_indices)
            cond = cond + dir_emb
            if seq_cond is not None:
                seq_cond = seq_cond + dir_emb.unsqueeze(1)

        logits = self._decode(cond, seq_cond, spatial_features)
        heatmap = torch.sigmoid(logits)  # (B, Hm, Wm)

        if gt_heatmap is not None and return_loss:
            return self._compute_loss(logits, heatmap, gt_heatmap, cond)

        if return_loss:
            return {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }

        return heatmap

    # ------------------------------------------------------------------
    # FPN Decode (returns raw logits)
    # ------------------------------------------------------------------

    def _decode(
        self,
        cond: torch.Tensor,
        seq_cond: Optional[torch.Tensor],
        spatial_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            (B, Hm, Wm) raw logits (before sigmoid).
        """
        cross_cond = seq_cond if seq_cond is not None else cond

        projected = [proj(feat) for proj, feat in zip(self.spatial_projectors, spatial_features)]

        x = projected[-1]

        for i, block in enumerate(self.decoder_blocks):
            skip_idx = len(projected) - 2 - i
            skip = projected[skip_idx]
            x = block(x, skip, cond, cross_cond)

        Hm, Wm = self.heatmap_size
        if x.shape[-2:] != (Hm, Wm):
            x = F.interpolate(x, size=(Hm, Wm), mode='bilinear', align_corners=False)

        spatial_logits = self.output_conv(x).squeeze(1)  # (B, Hm, Wm)

        # Global gate: condition-dependent bias shifts all spatial logits
        gate_bias = self.global_gate(cond).squeeze(-1)  # (B,)
        logits = spatial_logits + gate_bias.unsqueeze(-1).unsqueeze(-1)

        return logits

    # ------------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: torch.Tensor,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = logits.device
        B = logits.shape[0]

        # Prepare GT: (B, Hm, Wm)
        gt = gt_heatmap
        if gt.dim() == 4:
            gt = gt.squeeze(1)

        Hm, Wm = self.heatmap_size
        if gt.shape[-2:] != (Hm, Wm):
            gt = F.interpolate(gt.unsqueeze(1), size=(Hm, Wm),
                               mode='bilinear', align_corners=False).squeeze(1)

        # ==================== Sample-level weighting ==========================
        with torch.no_grad():
            sample_max = gt.flatten(1).max(dim=1).values  # (B,)
            is_negative = (sample_max < 0.01).float()
            is_positive = 1.0 - is_negative
            sample_weight = (
                self.positive_sample_boost * is_positive
                + self.negative_sample_weight * is_negative
            )  # (B,)

        # ==================== Spatial importance weighting ====================
        with torch.no_grad():
            spatial_weight = 1.0 + gt * (self.peak_spatial_weight - 1.0)  # (B, H, W)

        # ==================== 1. Focal BCEWithLogitsLoss ======================
        bce_unreduced = F.binary_cross_entropy_with_logits(
            logits, gt, reduction='none'
        )  # (B, H, W)

        # Focal weighting: down-weight easy pixels, focus on hard ones
        with torch.no_grad():
            pt = pred_heatmap * gt + (1 - pred_heatmap) * (1 - gt)
            focal_weight = (1 - pt) ** self.focal_gamma

        weighted_bce = (
            bce_unreduced
            * focal_weight
            * spatial_weight
            * sample_weight.view(-1, 1, 1)
        ).mean()

        # ==================== 2. Dice Loss ====================================
        pred_4d = pred_heatmap.unsqueeze(1)
        gt_4d = gt.unsqueeze(1)
        dice_loss = self._dice_loss(pred_4d, gt_4d, sample_weight.view(-1, 1, 1, 1))

        # ==================== 3. Peak Location Loss ===========================
        peak_loss, peak_loss_val = self._peak_location_loss(
            pred_4d, gt_4d, is_negative
        )

        # ==================== Total Loss ======================================
        total_loss = (
            weighted_bce
            + self.lambda_dice * dice_loss
            + self.lambda_peak * peak_loss
        )

        # ==================== Diagnostics =====================================
        self._training_step_counter += 1

        with torch.no_grad():
            pred_max = pred_heatmap.max().item()
            pred_mean = pred_heatmap.mean().item()
            raw_bce = bce_unreduced.mean().item()

            peak_mask = (gt > 0.1).float()
            bg_mask = 1.0 - peak_mask
            peak_px = peak_mask.sum()
            bg_px = bg_mask.sum()

            per_pixel_mse = (pred_heatmap - gt) ** 2
            mse_peak = (per_pixel_mse * peak_mask).sum().item() / (peak_px.item() + 1e-8)
            mse_bg = (per_pixel_mse * bg_mask).sum().item() / (bg_px.item() + 1e-8)

            gate_val = self.global_gate(cond).squeeze(-1).mean().item()

            neg_pred_mean = 0.0
            pos_pred_mean = 0.0
            n_neg = is_negative.sum().item()
            n_pos = is_positive.sum().item()
            if n_neg > 0:
                neg_pred_mean = pred_heatmap[is_negative.bool()].mean().item()
            if n_pos > 0:
                pos_pred_mean = pred_heatmap[is_positive.bool()].mean().item()

        return {
            'loss': total_loss,
            'heatmap': pred_heatmap.detach(),
            'direct_mse': per_pixel_mse.mean().item(),
            'direct_bce': raw_bce,
            'direct_dice_loss': dice_loss.item(),
            'direct_peak_loss': peak_loss_val,
            'direct_mse_peak': mse_peak,
            'direct_mse_bg': mse_bg,
            'direct_pred_max': pred_max,
            'direct_pred_mean': pred_mean,
            'direct_gate_bias': gate_val,
            'direct_neg_pred_mean': neg_pred_mean,
            'direct_pos_pred_mean': pos_pred_mean,
        }

    # ------------------------------------------------------------------
    # Dice Loss
    # ------------------------------------------------------------------

    def _dice_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        gt_flat = gt.flatten(1)

        intersection = (pred_flat * gt_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        gt_sum = gt_flat.sum(dim=1)

        dice = 1.0 - (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)

        weight = sample_weight.view(pred.shape[0])
        return (dice * weight).mean()

    # ------------------------------------------------------------------
    # Peak Location Loss
    # ------------------------------------------------------------------

    def _peak_location_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        is_negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        device = pred.device
        dtype = pred.dtype
        B, _, H, W = pred.shape
        diag = (H ** 2 + W ** 2) ** 0.5

        pos_mask = ~is_negative.bool()
        if not pos_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=False), 0.0

        pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
        pos_gt = gt[pos_indices]
        pos_pred = pred[pos_indices]

        with torch.no_grad():
            peaks_per_sample = self._find_gt_peaks(pos_gt)

        sample_ids = []
        peak_ys = []
        peak_xs = []

        for i, peaks in enumerate(peaks_per_sample):
            if len(peaks) == 0:
                continue
            for j in range(len(peaks)):
                sample_ids.append(i)
                peak_ys.append(peaks[j, 0].float())
                peak_xs.append(peaks[j, 1].float())

        if len(sample_ids) == 0:
            return torch.tensor(0.0, device=device, requires_grad=False), 0.0

        K = len(sample_ids)
        sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
        peak_ys_t = torch.stack(peak_ys)
        peak_xs_t = torch.stack(peak_xs)

        window_sigma = 5.0
        y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

        dy = y_grid - peak_ys_t.view(K, 1, 1)
        dx = x_grid - peak_xs_t.view(K, 1, 1)
        windows = torch.exp(-(dy ** 2 + dx ** 2) / (2 * window_sigma ** 2))

        pred_for_peaks = pos_pred[sample_ids_t, 0]

        windowed_pred = pred_for_peaks * windows
        flat = windowed_pred.view(K, -1)
        weights = F.softmax(flat / 0.1, dim=-1).view(K, H, W)

        pred_peak_y = (weights * y_grid.expand(K, H, W)).sum(dim=[1, 2])
        pred_peak_x = (weights * x_grid.expand(K, H, W)).sum(dim=[1, 2])

        dists = ((pred_peak_y - peak_ys_t) ** 2 + (pred_peak_x - peak_xs_t) ** 2).sqrt()
        peak_dist_loss = (dists / diag).mean()

        return peak_dist_loss, peak_dist_loss.item()

    # ------------------------------------------------------------------
    # GT Peak Detection (NMS)
    # ------------------------------------------------------------------

    def _find_gt_peaks(
        self,
        gt_heatmap: torch.Tensor,
        min_value: float = 0.1,
        nms_kernel: int = 5,
    ) -> list:
        B, _, H, W = gt_heatmap.shape
        pad = nms_kernel // 2

        gt_padded = F.pad(gt_heatmap, [pad] * 4, mode='replicate')
        local_max = F.max_pool2d(gt_padded, kernel_size=nms_kernel, stride=1, padding=0)

        is_peak = (gt_heatmap == local_max) & (gt_heatmap > min_value)
        is_peak = is_peak.squeeze(1)

        peaks_per_sample = []
        for i in range(B):
            peak_coords = is_peak[i].nonzero(as_tuple=False)
            if len(peak_coords) > 8:
                gt_vals = gt_heatmap[i, 0, peak_coords[:, 0], peak_coords[:, 1]]
                top_k_idx = gt_vals.topk(8).indices
                peak_coords = peak_coords[top_k_idx]
            peaks_per_sample.append(peak_coords)

        return peaks_per_sample
