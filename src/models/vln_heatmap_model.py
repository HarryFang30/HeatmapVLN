"""
VLNHeatmapModel: Main model assembly for heatmap training.

This module implements the complete model as specified in train.md section 4.
It integrates:
- Backbone (Qwen vision backbone)
- Temporal aggregation (mean/GRU)
- MultiHeatmapHead
- GaussianRenderer
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union
import logging

# Import our components
from .heatmap.multi_head import MultiHeatmapHead
from .heatmap.renderer import GaussianRenderer

logger = logging.getLogger(__name__)


class VLNHeatmapModel(nn.Module):
    """
    Complete VLN heatmap model as specified in train.md.

    Architecture:
        frames -> backbone -> temporal_aggregation -> head -> renderer -> probs
        [B,T,3,H,W] -> [B,T,D] -> [B,D] -> [B,K,Hm,Wm] -> [B,K,Hm,Wm]
    """

    def __init__(
        self,
        k_heatmaps: int,
        hm_size: Tuple[int, int] = (64, 64),
        vision_dim: int = 1024,
        agg: str = 'mean',
        use_lora: bool = False,
        lora_rank: int = 16,
        models_root: str = './models'
    ):
        """
        Initialize VLNHeatmapModel.

        Args:
            k_heatmaps: Number of heatmaps to generate (K)
            hm_size: Heatmap spatial size (Hm, Wm)
            vision_dim: Vision backbone output dimension
            agg: Temporal aggregation method ('mean' or 'gru')
            use_lora: Whether to use LoRA fine-tuning
            lora_rank: LoRA rank
            models_root: Path to model weights
        """
        super().__init__()

        self.k_heatmaps = k_heatmaps
        self.hm_size = hm_size
        self.vision_dim = vision_dim
        self.agg = agg
        self.use_lora = use_lora
        self.models_root = models_root

        # Initialize components
        self._build_backbone()
        self._build_temporal_aggregation()
        self._build_head_and_renderer()

        logger.info(f"VLNHeatmapModel initialized: K={k_heatmaps}, hm_size={hm_size}, agg={agg}")

    def _build_backbone(self):
        """Build vision backbone (placeholder - needs real implementation)"""
        # NOTE: This is a placeholder implementation
        # Real implementation should use build_qwen_vision_backbone from your existing code
        logger.warning("Using placeholder backbone - replace with real Qwen vision backbone")

        # Placeholder: simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, self.vision_dim)
        )

        # TODO: Replace with real implementation:
        # self.backbone = build_qwen_vision_backbone(
        #     models_root=self.models_root,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="eager"
        # )
        #
        # if self.use_lora:
        #     inject_lora(
        #         self.backbone,
        #         rank=self.lora_rank,
        #         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        #     )

    def _build_temporal_aggregation(self):
        """Build temporal aggregation layer"""
        if self.agg == 'gru':
            self.temporal = nn.GRU(
                self.vision_dim,
                self.vision_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        else:  # mean aggregation
            self.temporal = None

    def _build_head_and_renderer(self):
        """Build heatmap head and renderer"""
        fused_dim = self.vision_dim

        # Multi-heatmap head
        self.head = MultiHeatmapHead(
            in_dim=fused_dim,
            k_heatmaps=self.k_heatmaps,
            hm_size=self.hm_size
        )

        # Gaussian renderer
        self.renderer = GaussianRenderer(hm_size=self.hm_size)

    def forward(self, frames: torch.Tensor, text: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.

        Args:
            frames: Input frames [B, T, 3, H, W]
            text: Optional text instruction (not used in current implementation)

        Returns:
            Tuple of (probs, aux_outputs):
            - probs: Probability distributions [B, K, Hm, Wm]
            - aux_outputs: Dictionary with auxiliary outputs
        """
        B, T, C, H, W = frames.shape

        # Step 1: Extract features from all frames
        # Reshape to [B*T, C, H, W] for backbone processing
        frames_flat = frames.view(B * T, C, H, W)
        feats_flat = self.backbone(frames_flat)  # [B*T, vision_dim]

        # Reshape back to [B, T, vision_dim]
        feats = feats_flat.view(B, T, self.vision_dim)

        # Step 2: Temporal aggregation
        if self.temporal is not None:
            # GRU aggregation
            feats, _ = self.temporal(feats)  # [B, T, vision_dim]
            # Use final timestep
            fused = feats[:, -1]  # [B, vision_dim]
        else:
            # Mean aggregation
            fused = feats.mean(dim=1)  # [B, vision_dim]

        # Step 3: Generate heatmap logits
        logits = self.head(fused)  # [B, K, Hm, Wm]

        # Step 4: Render to probability distributions
        probs = self.renderer(logits)  # [B, K, Hm, Wm]

        # Auxiliary outputs
        aux_outputs = {
            "logits": logits,
            "features": feats,
            "fused_features": fused,
            "renderer_params": self.renderer.get_parameters_info()
        }

        return probs, aux_outputs

    def update_hm_size(self, new_hm_size: Tuple[int, int]):
        """
        Update heatmap size for curriculum learning.

        Args:
            new_hm_size: New heatmap size (Hm, Wm)
        """
        logger.info(f"Updating heatmap size from {self.hm_size} to {new_hm_size}")

        self.hm_size = new_hm_size
        self.head.update_hm_size(new_hm_size)
        self.renderer.update_hm_size(new_hm_size)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone parameters.

        Args:
            freeze: Whether to freeze backbone
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        if freeze:
            logger.info("Backbone frozen")
        else:
            logger.info("Backbone unfrozen")

    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get count of trainable parameters by component.

        Returns:
            Dict with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone": count_params(self.backbone),
            "temporal": count_params(self.temporal) if self.temporal else 0,
            "head": count_params(self.head),
            "renderer": count_params(self.renderer),
            "total": count_params(self)
        }

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return (f'k_heatmaps={self.k_heatmaps}, hm_size={self.hm_size}, '
                f'vision_dim={self.vision_dim}, agg={self.agg}')


def build_qwen_vision_backbone(models_root: str = './models', **kwargs):
    """
    Placeholder for real Qwen vision backbone builder.

    TODO: Replace with actual implementation that loads Qwen2.5-VL vision encoder
    from your existing codebase.

    Args:
        models_root: Path to model weights
        **kwargs: Additional arguments

    Returns:
        Vision backbone model
    """
    logger.warning("build_qwen_vision_backbone is a placeholder - implement with real Qwen backbone")

    # Return simple placeholder backbone
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, kwargs.get('vision_dim', 1024))
    )


def inject_lora(model, rank: int, target_modules: list):
    """
    Placeholder for LoRA injection.

    TODO: Implement with real LoRA library (e.g., peft)

    Args:
        model: Model to inject LoRA into
        rank: LoRA rank
        target_modules: Target module names
    """
    logger.warning("inject_lora is a placeholder - implement with real LoRA library")
    pass


# Test and example usage
if __name__ == "__main__":
    # Test VLNHeatmapModel
    batch_size = 2
    num_frames = 8
    k_heatmaps = 4
    hm_size = (64, 64)
    frame_size = (224, 224)

    # Create model
    model = VLNHeatmapModel(
        k_heatmaps=k_heatmaps,
        hm_size=hm_size,
        vision_dim=1024,
        agg='mean'
    )

    # Create test input
    frames = torch.randn(batch_size, num_frames, 3, *frame_size)

    print(f"Input frames shape: {frames.shape}")

    # Forward pass
    probs, aux_outputs = model(frames)

    print(f"Output probs shape: {probs.shape}")
    print(f"Auxiliary outputs keys: {list(aux_outputs.keys())}")

    # Check normalization
    probs_sum = probs.sum(dim=(-1, -2))
    print(f"Probs normalization check:")
    print(f"  Mean sum: {probs_sum.mean().item():.6f}")
    print(f"  Min sum: {probs_sum.min().item():.6f}")
    print(f"  Max sum: {probs_sum.max().item():.6f}")

    # Test parameter counting
    param_counts = model.get_trainable_parameters()
    print(f"\nTrainable parameters:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")

    # Test curriculum learning
    print(f"\nTesting curriculum learning:")
    print(f"Original hm_size: {model.hm_size}")

    new_size = (128, 128)
    model.update_hm_size(new_size)
    print(f"Updated hm_size: {model.hm_size}")

    # Test with new size
    probs_new, _ = model(frames)
    print(f"New output shape: {probs_new.shape}")

    # Test backbone freezing
    print(f"\nTesting backbone freezing:")
    model.freeze_backbone(True)
    frozen_params = model.get_trainable_parameters()
    print(f"Parameters after freezing backbone: {frozen_params['backbone']}")

    model.freeze_backbone(False)
    unfrozen_params = model.get_trainable_parameters()
    print(f"Parameters after unfreezing backbone: {unfrozen_params['backbone']}")

    print("\nVLNHeatmapModel test completed successfully!")