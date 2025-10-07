"""
VLNHeatmapModel: Main model assembly for heatmap training.

This module implements the complete model with ResNet backbone for reliable learning.
Architecture: frames → ResNet encoder → temporal aggregation → head → (optional renderer)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Dict, Any, Tuple, Optional
import logging

# Import our components
from .heatmap.multi_head import MultiHeatmapHead
from .heatmap.renderer import GaussianRenderer

logger = logging.getLogger(__name__)


class FrameEncoder(nn.Module):
    """
    Frame encoder using pretrained ResNet backbone.

    Uses ImageNet-pretrained ResNet18/34 for reliable feature extraction.
    """

    def __init__(self, out_dim: int = 1024, arch: str = "resnet18"):
        """
        Initialize frame encoder.

        Args:
            out_dim: Output feature dimension
            arch: Backbone architecture ('resnet18' or 'resnet34')
        """
        super().__init__()

        # Load pretrained ResNet
        if arch == "resnet34":
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        else:  # resnet18
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512

        # Remove final FC layer → output [B, 512, 1, 1]
        self.backbone = nn.Sequential(*list(m.children())[:-1])

        # Project to target dimension
        self.proj = nn.Linear(feat_dim, out_dim)

        # ImageNet normalization (register as buffer, not parameter)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        logger.info(f"FrameEncoder initialized: {arch}, out_dim={out_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input frames [B*T, 3, H, W] in range [0, 1]

        Returns:
            torch.Tensor: Features [B*T, out_dim]
        """
        # Normalize to ImageNet stats
        x = (x - self.mean) / self.std

        # Extract features
        f = self.backbone(x).flatten(1)  # [B*T, 512]

        # Project
        return self.proj(f)  # [B*T, out_dim]


class VLNHeatmapModel(nn.Module):
    """
    Complete VLN heatmap model with ResNet backbone.

    Architecture:
        frames → ResNet encoder → temporal_agg → head → logits
                                                      ↓ (optional)
                                                   renderer → probs
    """

    def __init__(
        self,
        k_heatmaps: int,
        hm_size: Tuple[int, int] = (64, 64),
        vision_dim: int = 1024,
        agg: str = 'mean',
        backbone: str = 'resnet18',
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
            backbone: Backbone architecture ('resnet18' or 'resnet34')
            use_lora: Whether to use LoRA fine-tuning (not used for ResNet)
            lora_rank: LoRA rank (not used for ResNet)
            models_root: Path to model weights (not used for ResNet)
        """
        super().__init__()

        self.k_heatmaps = k_heatmaps
        self.hm_size = hm_size
        self.vision_dim = vision_dim
        self.agg = agg
        self.backbone_type = backbone

        # Frame encoder with pretrained ResNet
        self.encoder = FrameEncoder(out_dim=vision_dim, arch=backbone)

        # Temporal aggregation
        if agg == 'gru':
            self.temporal = nn.GRU(
                vision_dim,
                vision_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.temporal = None

        # Multi-heatmap head
        self.head = MultiHeatmapHead(
            in_dim=vision_dim,
            k_heatmaps=k_heatmaps,
            hm_size=hm_size
        )

        # Gaussian renderer (for evaluation/visualization only)
        self.renderer = GaussianRenderer(hm_size=hm_size)

        logger.info(f"VLNHeatmapModel initialized: K={k_heatmaps}, hm_size={hm_size}, "
                   f"backbone={backbone}, agg={agg}")

    def forward(self, frames: torch.Tensor, return_logits: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.

        Args:
            frames: Input frames [B, T, 3, H, W]
            return_logits: If True, return raw logits; if False, return probabilities

        Returns:
            Tuple of (output, aux_dict):
            - output: Logits [B, K, Hm, Wm] if return_logits=True,
                     else probabilities [B, K, Hm, Wm]
            - aux_dict: Dictionary with auxiliary outputs
        """
        B, T, C, H, W = frames.shape

        # Step 1: Encode all frames
        x = frames.view(B * T, C, H, W)
        feat = self.encoder(x).view(B, T, self.vision_dim)  # [B, T, vision_dim]

        # Step 2: Temporal aggregation
        if self.temporal is not None:
            # GRU aggregation
            feat, _ = self.temporal(feat)  # [B, T, vision_dim]
            fused = feat[:, -1]  # Use final timestep [B, vision_dim]
        else:
            # Mean aggregation
            fused = feat.mean(dim=1)  # [B, vision_dim]

        # Step 3: Generate heatmap logits
        logits = self.head(fused)  # [B, K, Hm, Wm]

        # Auxiliary outputs
        aux_outputs = {
            "logits": logits,
            "features": feat,
            "fused_features": fused
        }

        # Return based on mode
        if return_logits:
            # Training mode: return raw logits
            return logits, aux_outputs
        else:
            # Evaluation mode: apply renderer
            probs = self.renderer(logits)  # [B, K, Hm, Wm]
            aux_outputs["renderer_params"] = self.renderer.get_parameters_info()
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
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

        if freeze:
            logger.info("Encoder (ResNet) frozen")
        else:
            logger.info("Encoder (ResNet) unfrozen")

    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get count of trainable parameters by component.

        Returns:
            Dict with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "encoder": count_params(self.encoder),
            "temporal": count_params(self.temporal) if self.temporal else 0,
            "head": count_params(self.head),
            "renderer": count_params(self.renderer),
            "total": count_params(self)
        }

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return (f'k_heatmaps={self.k_heatmaps}, hm_size={self.hm_size}, '
                f'backbone={self.backbone_type}, agg={self.agg}')


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
        agg='mean',
        backbone='resnet18'
    )

    # Create test input
    frames = torch.randn(batch_size, num_frames, 3, *frame_size)

    print(f"Input frames shape: {frames.shape}")

    # Forward pass (training mode)
    logits, aux_outputs = model(frames, return_logits=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Logits std: {logits.std():.3f}")

    # Forward pass (evaluation mode)
    probs, aux_outputs_eval = model(frames, return_logits=False)
    print(f"\nOutput probs shape: {probs.shape}")

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

    print("\nVLNHeatmapModel test completed successfully!")