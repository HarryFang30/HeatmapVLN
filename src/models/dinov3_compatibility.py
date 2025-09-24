"""
DINOv3 Compatibility Layer for Spatial-MLLM Integration
======================================================

This module provides compatibility layers and adapters to integrate DINOv3
with the Spatial-MLLM architecture, handling version differences and 
ensuring proper feature dimension alignment.

Key Compatibility Issues Addressed:
1. DINOv2 vs DINOv3 feature dimension differences
2. Token format changes between versions  
3. Attention mechanism updates
4. Position encoding differences (absolute vs RoPE)
5. Output format standardization for downstream components

Architecture Integration:
- DINOv3 processes N_k selected keyframes (2D encoder path)
- Features are aligned with VGGT 3D features for fusion
- Compatible with Qwen2.5-VL token requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from .dinov3.vision_transformer import DinoVisionTransformer, vit_base, vit_large, vit_giant
from .dinov3.hub import load_safetensors_weights, load_local_dinov3, dinov3_vit7b16
from ..utils.path_utils import resolve_model_path

logger = logging.getLogger(__name__)


@dataclass
class DINOv3CompatConfig:
    """Configuration for DINOv3 compatibility layer."""
    # Model selection
    model_size: str = "base"  # Options: base, large, giant
    patch_size: int = 14  # DINOv3 default patch size
    img_size: int = 518  # DINOv3 default image size
    
    # Compatibility settings
    target_embed_dim: int = 1024  # Target embedding dimension for VGGT compatibility
    enable_feature_adaptation: bool = True  # Enable dimension adaptation
    preserve_spatial_tokens: bool = True  # Keep spatial patch tokens
    
    # Performance settings
    use_checkpoint: bool = False  # Gradient checkpointing
    compile_model: bool = False  # torch.compile optimization
    
    # Integration settings
    align_with_vggt: bool = True  # Align feature dims with VGGT
    enable_multi_scale: bool = False  # Multi-scale feature extraction
    
    # Device settings
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class DINOv3CompatibilityLayer(nn.Module):
    """
    Compatibility layer for integrating DINOv3 with Spatial-MLLM pipeline.
    
    This layer handles:
    - Feature dimension adaptation
    - Token format standardization
    - Multi-scale feature extraction (optional)
    - Integration with VGGT features
    - Qwen2.5-VL compatibility
    """
    
    def __init__(self, config: DINOv3CompatConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Load DINOv3 model
        self.dinov3_model = self._create_dinov3_model()
        
        # Feature adaptation layers
        if config.enable_feature_adaptation:
            self.feature_adapter = self._create_feature_adapter()
        else:
            self.feature_adapter = nn.Identity()
            
        # Multi-scale extraction (optional)
        if config.enable_multi_scale:
            self.multi_scale_extractor = self._create_multi_scale_extractor()
        else:
            self.multi_scale_extractor = None
            
        # VGGT alignment layer
        if config.align_with_vggt:
            self.vggt_aligner = self._create_vggt_aligner()
        else:
            self.vggt_aligner = nn.Identity()
            
        # Move to device and set precision
        self.to(device=self.device, dtype=config.dtype)
        
        if config.compile_model:
            self.dinov3_model = torch.compile(self.dinov3_model)
            
    def _create_dinov3_model(self) -> DinoVisionTransformer:
        """Create and initialize DINOv3 model."""

        model_path = "./models/dinov3"

        # Resolve model path to absolute path
        try:
            resolved_path = resolve_model_path(model_path, "DINOv3")
            model_path = str(resolved_path)
            logger.info(f"Resolved DINOv3 path to: {model_path}")
        except FileNotFoundError as e:
            logger.warning(f"Could not resolve DINOv3 path: {e}, using original: {model_path}")

        try:
            # Use our working load_local_dinov3 function
            logger.info(f"Loading DINOv3 from local safetensors: {model_path}")
            model = load_local_dinov3(model_path)
            logger.info("Successfully loaded DINOv3 model using load_local_dinov3()")
            return model

        except Exception as e:
            logger.warning(f"Failed to load with load_local_dinov3: {e}")

            # Fallback: create model and load weights manually
            try:
                logger.info("Attempting fallback loading with dinov3_vit7b16()")
                model = dinov3_vit7b16(
                    pretrained=False,
                    patch_size=self.config.patch_size,
                    img_size=self.config.img_size,
                    use_checkpoint=self.config.use_checkpoint,
                )

                # Try to load safetensors weights
                model = load_safetensors_weights(model, model_path)
                logger.info("Successfully loaded DINOv3 model using fallback method")
                return model

            except Exception as fallback_error:
                logger.error(f"All loading methods failed. Fallback error: {fallback_error}")
                logger.warning("Creating uninitialized DINOv3 model")

                # Last resort: create uninitialized model
                model = dinov3_vit7b16(
                    pretrained=False,
                    patch_size=self.config.patch_size,
                    img_size=self.config.img_size,
                    use_checkpoint=self.config.use_checkpoint,
                )
                return model
    
    def _create_feature_adapter(self) -> nn.Module:
        """Create feature adaptation layer for dimension compatibility."""

        # DINOv3 embedding dimensions (updated with 7B model)
        embed_dims = {
            "base": 768,
            "large": 1024,
            "giant": 1536,
            "7b": 4096  # Your local model
        }

        # Determine current dimension based on actual model
        try:
            # First, try to get from actual model (most reliable)
            current_dim = self.dinov3_model.embed_dim
        except:
            # Fallback to config mapping
            if self.config.model_size in embed_dims:
                current_dim = embed_dims[self.config.model_size]
            else:
                # Default to 7B model size since that's what we're loading
                current_dim = 4096

        target_dim = self.config.target_embed_dim

        if current_dim == target_dim:
            return nn.Identity()
        else:
            # Linear projection with normalization
            return nn.Sequential(
                nn.LayerNorm(current_dim),
                nn.Linear(current_dim, target_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
    
    def _create_multi_scale_extractor(self) -> nn.Module:
        """Create multi-scale feature extraction module."""

        embed_dims = {
            "base": 768,
            "large": 1024,
            "giant": 1536,
            "7b": 4096  # Your local model
        }

        # Determine current dimension based on actual model
        try:
            # First, try to get from actual model (most reliable)
            current_dim = self.dinov3_model.embed_dim
        except:
            # Fallback to config mapping
            if self.config.model_size in embed_dims:
                current_dim = embed_dims[self.config.model_size]
            else:
                # Default to 7B model size since that's what we're loading
                current_dim = 4096
        
        # Multi-scale pooling for different spatial resolutions
        return nn.ModuleDict({
            'global_pool': nn.AdaptiveAvgPool2d(1),
            'patch_pool': nn.AdaptiveAvgPool2d(7),
            'fine_pool': nn.AdaptiveAvgPool2d(14),
            'projection': nn.Linear(current_dim * 3, current_dim)
        })
    
    def _create_vggt_aligner(self) -> nn.Module:
        """Create alignment layer for VGGT feature compatibility."""
        
        # Ensure features are compatible with VGGT 2*embed_dim format
        target_dim = self.config.target_embed_dim
        
        return nn.Sequential(
            nn.Linear(target_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Linear(target_dim * 2, target_dim * 2)
        )
    
    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process images through DINOv3 with compatibility adaptations.
        
        Args:
            images: Input images [B, N_k, C, H, W] or [B*N_k, C, H, W]
            return_attention: Return attention maps
            return_features: Return intermediate features
            
        Returns:
            Dictionary containing:
                - 'patch_tokens': Spatial patch tokens [B, N_k, N_patches, D]
                - 'cls_tokens': Class tokens [B, N_k, D]
                - 'adapted_features': Dimension-adapted features
                - 'vggt_aligned_features': VGGT-compatible features
                - 'attention_maps': Attention visualizations (if requested)
        """
        
        # Handle input dimensions
        batch_size, num_frames = None, None
        if len(images.shape) == 5:  # [B, N_k, C, H, W]
            batch_size, num_frames = images.shape[:2]
            images = images.view(-1, *images.shape[2:])  # [B*N_k, C, H, W]
        elif len(images.shape) == 4:  # [B*N_k, C, H, W]
            pass  # Already correct format
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")
            
        # Ensure correct device and dtype
        images = images.to(device=self.device, dtype=self.config.dtype)
        
        # Forward through DINOv3
        with torch.amp.autocast('cuda', enabled=True, dtype=self.config.dtype):
            features = self.dinov3_model.forward_features(images)
            
        # Extract tokens
        cls_tokens = features["x_norm_clstoken"]  # [B*N_k, D]
        patch_tokens = features["x_norm_patchtokens"]  # [B*N_k, N_patches, D]
        
        # Apply feature adaptation
        adapted_cls = self.feature_adapter(cls_tokens)
        adapted_patches = self.feature_adapter(patch_tokens)
        
        # Multi-scale extraction (if enabled)
        if self.multi_scale_extractor is not None:
            multi_scale_features = self._extract_multi_scale_features(patch_tokens)
            adapted_cls = adapted_cls + multi_scale_features
        
        # VGGT alignment
        vggt_aligned_cls = self.vggt_aligner(adapted_cls)
        vggt_aligned_patches = self.vggt_aligner(adapted_patches)
        
        # Reshape back to frame structure if needed
        if batch_size is not None and num_frames is not None:
            adapted_cls = adapted_cls.view(batch_size, num_frames, -1)
            adapted_patches = adapted_patches.view(batch_size, num_frames, *adapted_patches.shape[1:])
            vggt_aligned_cls = vggt_aligned_cls.view(batch_size, num_frames, -1)
            vggt_aligned_patches = vggt_aligned_patches.view(batch_size, num_frames, *vggt_aligned_patches.shape[1:])
        
        # Prepare output
        output = {
            'cls_tokens': adapted_cls,
            'patch_tokens': adapted_patches,
            'adapted_features': adapted_patches,  # For backward compatibility
            'vggt_aligned_features': vggt_aligned_patches,
            'vggt_aligned_cls': vggt_aligned_cls
        }
        
        # Add attention maps if requested
        if return_attention:
            output['attention_maps'] = self._extract_attention_maps(images)
            
        return output
    
    def _extract_multi_scale_features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale spatial features."""
        
        B, N, D = patch_tokens.shape
        
        # Determine spatial dimensions from number of patches
        spatial_size = int(N ** 0.5)
        if spatial_size * spatial_size != N:
            logger.warning(f"Non-square spatial layout: {N} patches")
            spatial_size = int(N ** 0.5)
        
        # Reshape to spatial layout
        spatial_tokens = patch_tokens.view(B, spatial_size, spatial_size, D)
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # Multi-scale pooling
        global_feat = self.multi_scale_extractor['global_pool'](spatial_tokens).squeeze(-1).squeeze(-1)  # [B, D]
        patch_feat = self.multi_scale_extractor['patch_pool'](spatial_tokens).mean(dim=[-1, -2])  # [B, D]
        fine_feat = self.multi_scale_extractor['fine_pool'](spatial_tokens).mean(dim=[-1, -2])  # [B, D]
        
        # Combine scales
        multi_scale = torch.cat([global_feat, patch_feat, fine_feat], dim=-1)  # [B, 3*D]
        combined = self.multi_scale_extractor['projection'](multi_scale)  # [B, D]
        
        return combined
    
    def _extract_attention_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization."""
        
        # This is a simplified implementation
        # Full implementation would require modifying the forward pass
        # to return attention weights from each layer
        
        return {
            'last_layer_attention': torch.zeros(
                images.shape[0], self.dinov3_model.blocks[-1].attn.num_heads,
                self.dinov3_model.patch_embed.num_patches + 1,
                self.dinov3_model.patch_embed.num_patches + 1,
                device=images.device
            )
        }
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get output feature dimensions for downstream components."""

        embed_dims = {
            "base": 768,
            "large": 1024,
            "giant": 1536,
            "7b": 4096  # Your local model
        }

        # Get actual original dimension
        try:
            original_dim = self.dinov3_model.embed_dim
        except:
            original_dim = embed_dims.get(self.config.model_size, 4096)

        adapted_dim = self.config.target_embed_dim
        vggt_aligned_dim = adapted_dim * 2 if self.config.align_with_vggt else adapted_dim

        # Get number of patches safely
        try:
            num_patches = self.dinov3_model.patch_embed.num_patches
        except:
            # Calculate from image size and patch size
            patches_per_side = self.config.img_size // self.config.patch_size
            num_patches = patches_per_side * patches_per_side

        return {
            'original_dim': original_dim,
            'adapted_dim': adapted_dim,
            'vggt_aligned_dim': vggt_aligned_dim,
            'patch_tokens_shape': f"[B, N_k, {num_patches}, {adapted_dim}]",
            'cls_tokens_shape': f"[B, N_k, {adapted_dim}]"
        }


def create_dinov3_compatibility_layer(
    model_size: str = "large",
    patch_size: int = 14,
    img_size: int = 518,
    target_embed_dim: int = 1024,
    align_with_vggt: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> DINOv3CompatibilityLayer:
    """
    Factory function to create DINOv3 compatibility layer.
    
    Args:
        model_size: DINOv3 model size ("base", "large", "giant")
        patch_size: Patch size for vision transformer
        img_size: Input image size  
        target_embed_dim: Target embedding dimension for compatibility
        align_with_vggt: Enable VGGT feature alignment
        device: Computing device
        dtype: Model precision
        
    Returns:
        Configured DINOv3CompatibilityLayer
    """
    config = DINOv3CompatConfig(
        model_size=model_size,
        patch_size=patch_size,
        img_size=img_size,
        target_embed_dim=target_embed_dim,
        align_with_vggt=align_with_vggt,
        device=device,
        dtype=dtype
    )
    
    return DINOv3CompatibilityLayer(config)


# Testing and validation
if __name__ == "__main__":
    # Test compatibility layer creation
    compat_layer = create_dinov3_compatibility_layer(
        model_size="base",  # Use smaller model for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("DINOv3 Compatibility Layer created successfully!")
    print(f"Feature dimensions: {compat_layer.get_feature_dimensions()}")
    
    # Test forward pass
    batch_size, num_keyframes = 1, 4
    test_images = torch.randn(batch_size, num_keyframes, 3, 518, 518)
    
    with torch.no_grad():
        output = compat_layer(test_images)
        
    print(f"Output keys: {list(output.keys())}")
    for key, tensor in output.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{key}: {tensor.shape}")