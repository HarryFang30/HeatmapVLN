"""
Direct Heatmap Head - Plan C
============================

Replaces the diffusion-based heatmap generation with a single-pass FPN decoder.
Uses the same MultiModalConditionEncoder for condition encoding, but generates
the heatmap directly via an FPN-style decoder with CrossAttention.

Advantages over diffusion:
    - Zero train-inference gap (same forward pass for both)
    - Direct optimization of heatmap quality (not epsilon MSE)
    - 50x faster inference (one forward vs 50-step DDIM)
    - Simpler training (no noise schedule, no Min-SNR, no CFG)

Architecture:
    Input:
        - LLM tokens: (B, seq_len, llm_dim)
        - Observation: (B, 3, H, W)

    Processing:
        1. MultiModalConditionEncoder: LLM + image -> cond, seq_cond, spatial_features
        2. FPN Decoder: spatial_features + cross-attention with seq_cond -> heatmap
        3. Direct loss on heatmap (MSE + Dice + Peak Location)

    Output:
        - Heatmap: (B, Hm, Wm) in [0, 1]
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
    negative_sample_weight: float = 0.3
    positive_sample_boost: float = 3.0

    # ==================== Direction Embedding =================================
    num_directions: int = 4

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
        """
        Args:
            x: (B, hidden_dim, H, W) current features
            skip: (B, hidden_dim, 2H, 2W) projected skip features from ResNet
            cond: (B, cond_dim) global conditioning for FiLM
            seq_cond: (B, seq_len, cond_dim) sequence conditioning for cross-attn
        """
        x = self.upsample(x)
        # Align spatial dims (skip may differ by 1 pixel due to stride rounding)
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

        # Training step counter for periodic diagnostics
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
        # ResNet-18 channels: [64, 128, 256, 512]
        from .diffusion.image_encoder import ResNetImageConditionEncoder
        resnet_channels = list(ResNetImageConditionEncoder.CHANNELS)  # [64, 128, 256, 512]

        self.spatial_projectors = nn.ModuleList()
        for ch in resnet_channels:
            proj = nn.Sequential(
                nn.Conv2d(ch, config.hidden_dim, 1),
                nn.GroupNorm(8, config.hidden_dim),
                nn.SiLU(),
            )
            self.spatial_projectors.append(proj)

        # ==================== FPN Decoder Blocks ==============================
        # 3 blocks: 7→14, 14→28, 28→56
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
        self.output_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(config.hidden_dim // 2, 1, 1),
        )

        # ==================== Loss Config =====================================
        self.peak_spatial_weight = config.peak_spatial_weight
        self.negative_sample_weight = config.negative_sample_weight
        self.positive_sample_boost = config.positive_sample_boost
        self.lambda_dice = config.lambda_dice
        self.lambda_peak = config.lambda_peak

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DirectHeatmapHead initialized: "
            "heatmap_size=%s, cond_dim=%d, hidden_dim=%d, "
            "decoder_blocks=%d, params=%s",
            config.heatmap_size, config.cond_dim, config.hidden_dim,
            config.num_decoder_blocks, f"{total_params:,}",
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
        """
        Same interface as DiffusionHeatmapHead.forward().
        """
        # 1. Flatten LLM tokens
        if llm_tokens.dim() > 3:
            B = llm_tokens.shape[0]
            D = llm_tokens.shape[-1]
            llm_tokens = llm_tokens.reshape(B, -1, D)

        # 2. Condition encoding (always with spatial features)
        cond, seq_cond, spatial_features = self.condition_encoder.forward_with_spatial(
            llm_tokens, observation
        )

        # 3. Direction embedding
        if direction_indices is not None:
            dir_emb = self.direction_embedding(direction_indices)
            cond = cond + dir_emb
            if seq_cond is not None:
                seq_cond = seq_cond + dir_emb.unsqueeze(1)

        # 4. Decode heatmap
        heatmap = self._decode(cond, seq_cond, spatial_features)

        # 5. Training mode
        if gt_heatmap is not None and return_loss:
            return self._compute_loss(heatmap, gt_heatmap, cond)

        if return_loss:
            return {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }

        return heatmap

    # ------------------------------------------------------------------
    # FPN Decode
    # ------------------------------------------------------------------

    def _decode(
        self,
        cond: torch.Tensor,
        seq_cond: Optional[torch.Tensor],
        spatial_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        FPN decode: deepest ResNet feature → upsample with skip connections → heatmap.

        Args:
            cond: (B, cond_dim)
            seq_cond: (B, seq_len, cond_dim) or None
            spatial_features: list of 4 ResNet feature maps
                [0] layer1: (B, 64, 56, 56)
                [1] layer2: (B, 128, 28, 28)
                [2] layer3: (B, 256, 14, 14)
                [3] layer4: (B, 512, 7, 7)

        Returns:
            (B, Hm, Wm) heatmap in [0, 1]
        """
        cross_cond = seq_cond if seq_cond is not None else cond

        # Project all spatial features to hidden_dim
        projected = [proj(feat) for proj, feat in zip(self.spatial_projectors, spatial_features)]

        # Start from deepest (index 3)
        x = projected[-1]  # (B, hidden_dim, 7, 7)

        # Decode through 3 blocks: 7→14→28→56
        for i, block in enumerate(self.decoder_blocks):
            skip_idx = len(projected) - 2 - i  # 2, 1, 0
            skip = projected[skip_idx]
            x = block(x, skip, cond, cross_cond)

        # Resize to target heatmap size and predict
        Hm, Wm = self.heatmap_size
        if x.shape[-2:] != (Hm, Wm):
            x = F.interpolate(x, size=(Hm, Wm), mode='bilinear', align_corners=False)

        logits = self.output_head(x)  # (B, 1, Hm, Wm)
        heatmap = torch.sigmoid(logits).squeeze(1)  # (B, Hm, Wm)

        return heatmap

    # ------------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: spatial-weighted MSE + Dice + Peak Location.

        All losses operate directly on the heatmap space — no epsilon indirection.
        """
        device = pred_heatmap.device
        B = pred_heatmap.shape[0]

        # Prepare GT: (B, Hm, Wm) -> (B, 1, Hm, Wm)
        if gt_heatmap.dim() == 3:
            gt_4d = gt_heatmap.unsqueeze(1)
        else:
            gt_4d = gt_heatmap

        Hm, Wm = self.heatmap_size
        if gt_4d.shape[-2:] != (Hm, Wm):
            gt_4d = F.interpolate(gt_4d, size=(Hm, Wm), mode='bilinear', align_corners=False)

        pred_4d = pred_heatmap.unsqueeze(1) if pred_heatmap.dim() == 3 else pred_heatmap

        # ==================== Sample-level weighting ==========================
        with torch.no_grad():
            sample_max = gt_4d.flatten(1).max(dim=1).values  # (B,)
            is_negative = (sample_max < 0.01).float()
            is_positive = 1.0 - is_negative
            sample_weight = (
                self.positive_sample_boost * is_positive
                + self.negative_sample_weight * is_negative
            ).view(-1, 1, 1, 1)

        # ==================== Spatial importance weighting ====================
        with torch.no_grad():
            spatial_weight = 1.0 + gt_4d * (self.peak_spatial_weight - 1.0)

        # ==================== 1. Spatial-weighted MSE =========================
        per_pixel_mse = (pred_4d - gt_4d) ** 2
        mse_loss = (spatial_weight * sample_weight * per_pixel_mse).mean()

        # ==================== 2. Dice Loss ====================================
        dice_loss = self._dice_loss(pred_4d, gt_4d, sample_weight)

        # ==================== 3. Peak Location Loss ===========================
        peak_loss, peak_loss_val = self._peak_location_loss(pred_4d, gt_4d, is_negative)

        # ==================== Total Loss ======================================
        total_loss = mse_loss + self.lambda_dice * dice_loss + self.lambda_peak * peak_loss

        # ==================== Diagnostics (no grad) ===========================
        self._training_step_counter += 1

        with torch.no_grad():
            raw_mse = per_pixel_mse.mean().item()
            pred_max = pred_4d.max().item()
            pred_mean = pred_4d.mean().item()

            peak_mask = (gt_4d > 0.1).float()
            bg_mask = 1.0 - peak_mask
            peak_px = peak_mask.sum()
            bg_px = bg_mask.sum()
            mse_peak = (per_pixel_mse * peak_mask).sum().item() / (peak_px.item() + 1e-8)
            mse_bg = (per_pixel_mse * bg_mask).sum().item() / (bg_px.item() + 1e-8)

        return {
            'loss': total_loss,
            'heatmap': pred_heatmap.detach(),
            'direct_mse': raw_mse,
            'direct_dice_loss': dice_loss.item(),
            'direct_peak_loss': peak_loss_val,
            'direct_mse_peak': mse_peak,
            'direct_mse_bg': mse_bg,
            'direct_pred_max': pred_max,
            'direct_pred_mean': pred_mean,
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
        """
        Soft Dice loss for shape matching.

        Args:
            pred: (B, 1, H, W) predicted heatmap in [0, 1]
            gt: (B, 1, H, W) ground truth heatmap in [0, 1]
            sample_weight: (B, 1, 1, 1) per-sample weight
        """
        # Per-sample dice
        pred_flat = pred.flatten(1)  # (B, H*W)
        gt_flat = gt.flatten(1)      # (B, H*W)

        intersection = (pred_flat * gt_flat).sum(dim=1)  # (B,)
        pred_sum = pred_flat.sum(dim=1)
        gt_sum = gt_flat.sum(dim=1)

        dice = 1.0 - (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)  # (B,)

        # Apply sample weight
        weight = sample_weight.view(B := pred.shape[0])
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
        """
        Multi-peak location loss via soft-argmax.

        Directly compares predicted vs GT peak positions using differentiable
        soft-argmax within Gaussian attention windows around GT peaks.
        """
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

        # NMS peak detection on GT
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

        # Gaussian attention windows
        window_sigma = 5.0
        y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

        dy = y_grid - peak_ys_t.view(K, 1, 1)
        dx = x_grid - peak_xs_t.view(K, 1, 1)
        windows = torch.exp(-(dy ** 2 + dx ** 2) / (2 * window_sigma ** 2))

        # Get prediction values at peak locations
        pred_for_peaks = pos_pred[sample_ids_t, 0]  # (K, H, W)

        # Windowed soft-argmax
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
        """NMS-based peak detection on GT heatmap. Returns list of (num_peaks, 2) tensors."""
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
