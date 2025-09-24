"""
Advanced Feature Fusion for Dual-Encoder Architecture
====================================================

This module implements sophisticated feature fusion mechanisms for combining
VGGT 3D features with DINOv3 2D features in the VLN pipeline.

Architecture Integration:
- VGGT (3D encoder): Processes all N_m frames → Geometry features
- DINOv3 (2D encoder): Processes selected N_k keyframes → Semantic features  
- Feature Fusion: Combines 3D geometry + 2D semantics → Rich spatial understanding

Key Fusion Strategies:
1. Cross-Modal Attention Fusion
2. Graph-based Spatial Fusion  
3. Hierarchical Feature Pyramid Fusion
4. Adaptive Weighted Fusion
5. Transformer-based Deep Fusion

The goal is to create rich spatial-semantic representations that leverage
both geometric understanding from VGGT and semantic understanding from DINOv3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion mechanisms."""
    # Input dimensions
    vggt_feature_dim: int = 2048  # VGGT outputs 2*embed_dim (2*1024)
    dinov3_feature_dim: int = 2048  # DINOv3 aligned features (2*1024)
    output_dim: int = 2048  # Fused feature dimension
    
    # Fusion strategy
    fusion_method: str = "cross_attention"  # Options: concatenate, cross_attention, graph, hierarchical, adaptive
    
    # Cross-attention parameters
    num_attention_heads: int = 16
    attention_dropout: float = 0.1
    
    # Graph fusion parameters
    num_graph_layers: int = 3
    graph_hidden_dim: int = 1024
    
    # Hierarchical fusion parameters
    pyramid_levels: int = 4
    level_dims: List[int] = None
    
    # Adaptive fusion parameters
    num_experts: int = 8  # For mixture of experts
    temperature: float = 1.0  # For attention temperature
    
    # Performance settings
    use_gradient_checkpointing: bool = False
    enable_layer_norm: bool = True
    residual_connections: bool = True
    
    def __post_init__(self):
        if self.level_dims is None:
            # Default pyramid dimensions
            self.level_dims = [self.output_dim // (2**i) for i in range(self.pyramid_levels)]


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion between VGGT 3D and DINOv3 2D features.
    
    This fusion mechanism allows 3D geometric features to attend to 2D semantic 
    features and vice versa, creating rich spatial-semantic representations.
    """
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        
        # Feature projection to common dimension
        self.vggt_proj = nn.Linear(config.vggt_feature_dim, config.output_dim)
        self.dinov3_proj = nn.Linear(config.dinov3_feature_dim, config.output_dim)
        
        # Cross-attention: 3D → 2D
        self.cross_attn_3d_to_2d = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Cross-attention: 2D → 3D  
        self.cross_attn_2d_to_3d = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Self-attention for fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        if config.enable_layer_norm:
            self.norm_3d = nn.LayerNorm(config.output_dim)
            self.norm_2d = nn.LayerNorm(config.output_dim) 
            self.norm_fused = nn.LayerNorm(config.output_dim)
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.output_dim * 2, config.output_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.output_dim * 4, config.output_dim),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        vggt_features: torch.Tensor,  # [B, N_k, N_patches, C_3d]
        dinov3_features: torch.Tensor,  # [B, N_k, N_patches, C_2d]
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform cross-modal attention fusion.
        
        Args:
            vggt_features: 3D geometry features from VGGT
            dinov3_features: 2D semantic features from DINOv3
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing fused features and attention maps
        """
        
        batch_size, num_keyframes, num_patches = vggt_features.shape[:3]
        
        # Project to common dimension
        vggt_proj = self.vggt_proj(vggt_features)  # [B, N_k, N_patches, D]
        dinov3_proj = self.dinov3_proj(dinov3_features)  # [B, N_k, N_patches, D]
        
        # Reshape for attention: [B*N_k, N_patches, D]
        vggt_flat = vggt_proj.view(batch_size * num_keyframes, num_patches, -1)
        dinov3_flat = dinov3_proj.view(batch_size * num_keyframes, num_patches, -1)
        
        # Cross-attention: 3D features attend to 2D features
        enhanced_3d, attn_3d_to_2d = self.cross_attn_3d_to_2d(
            query=vggt_flat,
            key=dinov3_flat, 
            value=dinov3_flat,
            attn_mask=attention_mask,
            need_weights=True
        )
        
        # Cross-attention: 2D features attend to 3D features
        enhanced_2d, attn_2d_to_3d = self.cross_attn_2d_to_3d(
            query=dinov3_flat,
            key=vggt_flat,
            value=vggt_flat,
            attn_mask=attention_mask,
            need_weights=True
        )
        
        # Apply layer normalization and residual connections
        if hasattr(self, 'norm_3d'):
            enhanced_3d = self.norm_3d(enhanced_3d + vggt_flat)
            enhanced_2d = self.norm_2d(enhanced_2d + dinov3_flat)
        
        # Concatenate enhanced features
        concatenated = torch.cat([enhanced_3d, enhanced_2d], dim=-1)
        
        # Final fusion through MLP
        fused_features = self.fusion_mlp(concatenated)
        
        # Self-attention on fused features for further refinement
        refined_fused, self_attn = self.self_attention(
            fused_features, fused_features, fused_features, need_weights=True
        )
        
        if hasattr(self, 'norm_fused'):
            refined_fused = self.norm_fused(refined_fused + fused_features)
        
        # Reshape back: [B, N_k, N_patches, D]
        final_features = refined_fused.view(batch_size, num_keyframes, num_patches, -1)
        
        return {
            'fused_features': final_features,
            'enhanced_3d': enhanced_3d.view(batch_size, num_keyframes, num_patches, -1),
            'enhanced_2d': enhanced_2d.view(batch_size, num_keyframes, num_patches, -1),
            'attention_3d_to_2d': attn_3d_to_2d,
            'attention_2d_to_3d': attn_2d_to_3d,
            'self_attention': self_attn
        }


class GraphSpatialFusion(nn.Module):
    """
    Graph-based spatial fusion leveraging spatial relationships between patches.
    
    This approach treats image patches as graph nodes and models their spatial
    relationships explicitly for enhanced fusion.
    """
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        
        # Feature projections
        self.vggt_proj = nn.Linear(config.vggt_feature_dim, config.graph_hidden_dim)
        self.dinov3_proj = nn.Linear(config.dinov3_feature_dim, config.graph_hidden_dim)
        
        # Graph convolutional layers
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(config.graph_hidden_dim, config.graph_hidden_dim)
            for _ in range(config.num_graph_layers)
        ])
        
        # Final output projection
        self.output_proj = nn.Linear(config.graph_hidden_dim * 2, config.output_dim)
        
        # Spatial positional encoding for patches
        self.spatial_pos_encoding = SpatialPositionalEncoding(config.graph_hidden_dim)
        
    def forward(
        self,
        vggt_features: torch.Tensor,
        dinov3_features: torch.Tensor,
        patch_coordinates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform graph-based spatial fusion.
        
        Args:
            vggt_features: 3D geometry features
            dinov3_features: 2D semantic features  
            patch_coordinates: Optional spatial coordinates of patches
            
        Returns:
            Dictionary containing fused features
        """
        
        batch_size, num_keyframes, num_patches = vggt_features.shape[:3]
        
        # Project features
        vggt_proj = self.vggt_proj(vggt_features)
        dinov3_proj = self.dinov3_proj(dinov3_features)
        
        # Add spatial positional encoding
        if patch_coordinates is not None:
            spatial_encoding = self.spatial_pos_encoding(patch_coordinates)
            vggt_proj = vggt_proj + spatial_encoding
            dinov3_proj = dinov3_proj + spatial_encoding
        
        # Create adjacency matrix based on spatial proximity
        adjacency_matrix = self._create_spatial_adjacency(num_patches)
        adjacency_matrix = adjacency_matrix.to(vggt_features.device)
        
        # Apply graph convolutions
        vggt_graph = vggt_proj
        dinov3_graph = dinov3_proj
        
        for graph_layer in self.graph_layers:
            vggt_graph = graph_layer(vggt_graph, adjacency_matrix)
            dinov3_graph = graph_layer(dinov3_graph, adjacency_matrix)
        
        # Combine graph-processed features
        combined = torch.cat([vggt_graph, dinov3_graph], dim=-1)
        
        # Final projection
        fused_features = self.output_proj(combined)
        
        return {
            'fused_features': fused_features,
            'vggt_graph': vggt_graph,
            'dinov3_graph': dinov3_graph,
            'adjacency_matrix': adjacency_matrix
        }
    
    def _create_spatial_adjacency(self, num_patches: int) -> torch.Tensor:
        """Create spatial adjacency matrix for patches."""
        
        # Assume square patch layout
        patch_size = int(math.sqrt(num_patches))
        adjacency = torch.zeros(num_patches, num_patches)
        
        for i in range(patch_size):
            for j in range(patch_size):
                current = i * patch_size + j
                
                # Connect to spatial neighbors
                neighbors = []
                if i > 0: neighbors.append((i-1) * patch_size + j)  # Up
                if i < patch_size-1: neighbors.append((i+1) * patch_size + j)  # Down  
                if j > 0: neighbors.append(i * patch_size + (j-1))  # Left
                if j < patch_size-1: neighbors.append(i * patch_size + (j+1))  # Right
                
                for neighbor in neighbors:
                    adjacency[current, neighbor] = 1.0
        
        # Add self-connections
        adjacency.fill_diagonal_(1.0)
        
        # Normalize adjacency matrix
        degree = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (degree + 1e-8)
        
        return adjacency


class GraphConvLayer(nn.Module):
    """Graph convolution layer for spatial feature processing."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply graph convolution.
        
        Args:
            x: Node features [B, N_k, N_patches, D]
            adjacency: Adjacency matrix [N_patches, N_patches]
            
        Returns:
            Updated node features [B, N_k, N_patches, D]
        """
        
        # Apply linear transformation
        h = self.linear(x)
        
        # Graph convolution: aggregate neighbor features
        h = torch.matmul(adjacency.unsqueeze(0).unsqueeze(0), h)
        
        # Apply activation and normalization
        h = self.activation(h)
        h = self.layer_norm(h)
        
        return h + x if x.shape[-1] == h.shape[-1] else h


class SpatialPositionalEncoding(nn.Module):
    """Spatial positional encoding for patch coordinates."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)  # 2D coordinates → embedding
        
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial positional encoding.
        
        Args:
            coordinates: Patch coordinates [B, N_k, N_patches, 2]
            
        Returns:
            Positional encoding [B, N_k, N_patches, dim]
        """
        return self.proj(coordinates)


class AdaptiveWeightedFusion(nn.Module):
    """
    Adaptive weighted fusion that learns optimal combination weights.
    
    This approach uses attention mechanisms to dynamically weight the 
    contribution of 3D and 2D features based on content.
    """
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        
        # Feature projections
        self.vggt_proj = nn.Linear(config.vggt_feature_dim, config.output_dim)
        self.dinov3_proj = nn.Linear(config.dinov3_feature_dim, config.output_dim)
        
        # Adaptive weight prediction
        self.weight_predictor = nn.Sequential(
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.output_dim, 2),  # Weights for 3D and 2D
            nn.Softmax(dim=-1)
        )
        
        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.output_dim * 2, config.output_dim)
        )
        
    def forward(
        self,
        vggt_features: torch.Tensor,
        dinov3_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform adaptive weighted fusion.
        
        Args:
            vggt_features: 3D geometry features
            dinov3_features: 2D semantic features
            
        Returns:
            Dictionary containing fused features and weights
        """
        
        # Project to common dimension
        vggt_proj = self.vggt_proj(vggt_features)
        dinov3_proj = self.dinov3_proj(dinov3_features)
        
        # Predict adaptive weights
        combined = torch.cat([vggt_proj, dinov3_proj], dim=-1)
        weights = self.weight_predictor(combined)  # [B, N_k, N_patches, 2]
        
        # Apply adaptive weights
        weight_3d = weights[..., 0:1]  # [B, N_k, N_patches, 1]
        weight_2d = weights[..., 1:2]  # [B, N_k, N_patches, 1]
        
        weighted_features = weight_3d * vggt_proj + weight_2d * dinov3_proj
        
        # Enhance fused features
        enhanced_features = self.feature_enhancer(weighted_features)
        
        return {
            'fused_features': enhanced_features + weighted_features,  # Residual connection
            'fusion_weights_3d': weight_3d,
            'fusion_weights_2d': weight_2d,
            'weighted_features': weighted_features
        }


class FeatureFusionFactory:
    """Factory for creating different types of feature fusion mechanisms."""
    
    @staticmethod
    def create_fusion(
        fusion_method: str,
        config: FeatureFusionConfig
    ) -> nn.Module:
        """
        Create feature fusion module based on specified method.
        
        Args:
            fusion_method: Type of fusion ("cross_attention", "graph", "adaptive", etc.)
            config: Fusion configuration
            
        Returns:
            Configured fusion module
        """
        
        if fusion_method == "cross_attention":
            return CrossModalAttentionFusion(config)
        elif fusion_method == "graph":
            return GraphSpatialFusion(config)
        elif fusion_method == "adaptive":
            return AdaptiveWeightedFusion(config)
        elif fusion_method == "concatenate":
            # Simple concatenation baseline
            return SimpleConcatenateFusion(config)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")


class SimpleConcatenateFusion(nn.Module):
    """Simple concatenation fusion baseline."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.vggt_feature_dim + config.dinov3_feature_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        vggt_features: torch.Tensor,
        dinov3_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Simple concatenation and projection."""
        
        concatenated = torch.cat([vggt_features, dinov3_features], dim=-1)
        fused = self.projection(concatenated)
        
        return {'fused_features': fused}


def create_feature_fusion(
    fusion_method: str = "cross_attention",
    vggt_feature_dim: int = 2048,
    dinov3_feature_dim: int = 2048,
    output_dim: int = 2048,
    num_attention_heads: int = 16
) -> nn.Module:
    """
    Factory function to create feature fusion module.
    
    Args:
        fusion_method: Type of fusion mechanism
        vggt_feature_dim: VGGT feature dimension
        dinov3_feature_dim: DINOv3 feature dimension  
        output_dim: Output feature dimension
        num_attention_heads: Number of attention heads
        
    Returns:
        Configured feature fusion module
    """
    
    config = FeatureFusionConfig(
        vggt_feature_dim=vggt_feature_dim,
        dinov3_feature_dim=dinov3_feature_dim,
        output_dim=output_dim,
        fusion_method=fusion_method,
        num_attention_heads=num_attention_heads
    )
    
    return FeatureFusionFactory.create_fusion(fusion_method, config)


# Example usage and testing
if __name__ == "__main__":
    # Create different fusion mechanisms
    fusion_methods = ["cross_attention", "graph", "adaptive", "concatenate"]
    
    for method in fusion_methods:
        print(f"\nTesting {method} fusion:")
        
        fusion = create_feature_fusion(fusion_method=method)
        
        # Test data
        batch_size, num_keyframes, num_patches = 1, 4, 1369  # 37*37 patches
        vggt_features = torch.randn(batch_size, num_keyframes, num_patches, 2048)
        dinov3_features = torch.randn(batch_size, num_keyframes, num_patches, 2048)
        
        with torch.no_grad():
            result = fusion(vggt_features, dinov3_features)
        
        print(f"  Output shape: {result['fused_features'].shape}")
        print(f"  Output keys: {list(result.keys())}")
    
    print("\nFeature fusion mechanisms created successfully!")