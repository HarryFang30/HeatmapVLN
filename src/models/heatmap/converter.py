# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# Adapted from BridgeVLA LLM token to heatmap conversion logic

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple, List, Dict

from .upsampling import ConvexUpSample


class LLMToHeatmapConverter(nn.Module):
    """
    LLM Token to First-Person Inter-Frame Heatmap Converter
    
    This module converts LLM token outputs to a single 2D heatmap that shows 
    spatial relationships between video frames from a first-person perspective:
    1. Extracting spatial tokens from LLM hidden states
    2. Fusing cross-frame spatial information 
    3. Generating a single first-person heatmap showing where content from other frames appears
    
    This enables understanding of inter-frame spatial connections and demonstrates
    the model's 3D spatial reasoning capabilities through 2D visualization.
    """
    
    def __init__(
        self, 
        vlm_dim: int = 2048, 
        patch_size: int = 16, 
        target_size: int = 224,
        up_kernel: int = 3,
        mask_scale: float = 0.1,
        with_bn: bool = False,
        enable_inter_frame_fusion: bool = True
    ):
        """
        Initialize the LLM to Heatmap converter.
        
        Args:
            vlm_dim (int): VLM hidden dimension (default: 2048)
            patch_size (int): Patch size for spatial reconstruction (default: 16)
            target_size (int): Target heatmap size (default: 224)
            up_kernel (int): Upsampling kernel size (default: 3)
            mask_scale (float): Mask scale for upsampling (default: 0.1)
            with_bn (bool): Whether to use batch normalization (default: False)
            enable_inter_frame_fusion (bool): Enable inter-frame spatial fusion (default: True)
        """
        super().__init__()
        self.vlm_dim = vlm_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.up_ratio = target_size // patch_size  # e.g., 224 // 16 = 14
        self.enable_inter_frame_fusion = enable_inter_frame_fusion
        
        # ConvexUpSample module for generating heatmaps
        self.upsampler = ConvexUpSample(
            in_dim=vlm_dim,
            out_dim=1,  # Single channel heatmap output
            up_ratio=self.up_ratio,
            up_kernel=up_kernel,
            mask_scale=mask_scale,
            with_bn=with_bn
        )
        
        # Inter-frame spatial fusion module
        if enable_inter_frame_fusion:
            self.inter_frame_attention = nn.MultiheadAttention(
                embed_dim=vlm_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(vlm_dim)
        
    def extract_vision_tokens(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        num_image_tokens: int = 256,
        num_views: int = 1
    ) -> torch.Tensor:
        """
        Extract vision-related tokens from LLM hidden states.
        
        Args:
            hidden_states (torch.Tensor): LLM output [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            num_image_tokens (int): Number of image tokens per view (default: 256 = 16x16)
            num_views (int): Number of camera views (default: 1)
            
        Returns:
            torch.Tensor: Extracted image tokens [batch_size, num_tokens, hidden_dim]
        """
        batch_size = hidden_states.shape[0]
        total_image_tokens = num_image_tokens * num_views
        image_tokens = []
        
        for i in range(batch_size):
            # Get valid tokens for current sample
            current_mask = attention_mask[i]
            current_output = hidden_states[i]
            
            # Find non-zero token indices
            non_zero_indices = torch.nonzero(current_mask != 0, as_tuple=True)[0]
            non_zero_output = current_output[non_zero_indices]
            
            # Extract the first total_image_tokens (corresponding to image patches)
            if non_zero_output.shape[0] >= total_image_tokens:
                vision_tokens = non_zero_output[:total_image_tokens]
            else:
                # Pad with zeros if insufficient tokens
                vision_tokens = torch.zeros(total_image_tokens, self.vlm_dim, 
                                         device=hidden_states.device, dtype=hidden_states.dtype)
                vision_tokens[:non_zero_output.shape[0]] = non_zero_output
            
            image_tokens.append(vision_tokens)
        
        return torch.stack(image_tokens)  # [batch_size, total_tokens, hidden_dim]
    
    def tokens_to_spatial_features(
        self, 
        image_tokens: torch.Tensor, 
        num_views: int = 1
    ) -> torch.Tensor:
        """
        Reconstruct 1D token sequence into 2D spatial feature maps.
        
        Args:
            image_tokens (torch.Tensor): Token features [batch_size, num_tokens, hidden_dim]
            num_views (int): Number of camera views (default: 1)
            
        Returns:
            torch.Tensor: Spatial features [batch_size*num_views, hidden_dim, patch_size, patch_size]
        """
        batch_size = image_tokens.shape[0]
        
        # Reshape token sequence to spatial layout
        # [batch, num_views*256, dim] -> [batch, dim, num_views, 16, 16]
        x = rearrange(
            image_tokens, 
            'b (c h w) d -> b d c h w', 
            c=num_views, 
            h=self.patch_size, 
            w=self.patch_size
        )
        
        # Merge batch and view dimensions for processing
        # [batch, dim, num_views, 16, 16] -> [batch*num_views, dim, 16, 16]
        x = x.transpose(1, 2).contiguous().view(
            batch_size * num_views, 
            self.vlm_dim, 
            self.patch_size, 
            self.patch_size
        )
        
        return x
    
    def generate_heatmap(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Generate heatmaps from spatial features using ConvexUpSample.
        
        Args:
            spatial_features (torch.Tensor): Features [batch*views, hidden_dim, 16, 16]
            
        Returns:
            torch.Tensor: Generated heatmaps [batch*views, 1, target_size, target_size]
        """
        # Ensure float32 for stable computation
        spatial_features = spatial_features.to(torch.float32)
        
        # Generate heatmaps using learned upsampling
        heatmaps = self.upsampler(spatial_features)
        
        return heatmaps
    
    def generate_single_inter_frame_heatmap(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_idx: int = 0
    ) -> torch.Tensor:
        """
        Generate a single first-person inter-frame heatmap showing spatial relationships.
        
        This method creates one heatmap from the current frame's perspective that displays
        where content from other frames would appear spatially, demonstrating the model's
        understanding of 3D spatial relationships across video frames.
        
        Args:
            hidden_states (torch.Tensor): LLM output [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            current_frame_idx (int): Index of the current reference frame (default: 0)
            
        Returns:
            torch.Tensor: Single inter-frame heatmap [batch_size, 1, target_size, target_size]
        """
        batch_size = hidden_states.shape[0]
        
        # 1. Extract all vision tokens from LLM output
        image_tokens = self.extract_vision_tokens(hidden_states, attention_mask, num_views=1)
        
        # 2. Apply inter-frame spatial fusion if enabled
        if self.enable_inter_frame_fusion and image_tokens.shape[1] > 256:
            # Assume tokens are organized as [current_frame_tokens, other_frame_tokens, ...]
            
            # Reshape to separate frames: assume 256 tokens per frame
            tokens_per_frame = 256
            num_frames = image_tokens.shape[1] // tokens_per_frame
            
            if num_frames > 1:
                # Reshape: [B, N*256, D] -> [B, N, 256, D]
                frame_tokens = image_tokens.view(batch_size, num_frames, tokens_per_frame, self.vlm_dim)
                
                # Current frame as query, other frames as keys/values
                current_tokens = frame_tokens[:, current_frame_idx:current_frame_idx+1]  # [B, 1, 256, D]
                other_tokens = frame_tokens  # [B, N, 256, D]
                
                # Flatten for attention: [B, 256, D] and [B, N*256, D]
                current_flat = current_tokens.view(batch_size, tokens_per_frame, self.vlm_dim)
                other_flat = other_tokens.view(batch_size, num_frames * tokens_per_frame, self.vlm_dim)
                
                # Apply cross-frame attention to understand spatial relationships
                attended_tokens, attention_weights = self.inter_frame_attention(
                    query=current_flat,
                    key=other_flat, 
                    value=other_flat
                )
                
                # Normalize and use attended tokens
                fused_tokens = self.fusion_norm(attended_tokens + current_flat)
                
                # Use fused tokens for heatmap generation
                image_tokens = fused_tokens  # [B, 256, D]
            else:
                # Single frame case - use original tokens
                image_tokens = image_tokens.view(batch_size, -1, self.vlm_dim)
        else:
            # No inter-frame fusion - use first 256 tokens
            image_tokens = image_tokens[:, :256]  # [B, 256, D]
        
        # 3. Reconstruct tokens into spatial features
        spatial_features = self.tokens_to_spatial_features(image_tokens, num_views=1)
        
        # 4. Generate single heatmap
        heatmap = self.generate_heatmap(spatial_features)
        
        # 5. Ensure output is [B, 1, H, W] format
        if heatmap.dim() == 4 and heatmap.shape[1] == 1:
            # Already correct format: [B, 1, H, W]
            return heatmap
        elif heatmap.dim() == 3:
            # Add channel dimension: [B, H, W] -> [B, 1, H, W]
            return heatmap.unsqueeze(1)
        else:
            # Reshape to correct format
            return heatmap.view(batch_size, 1, self.target_size, self.target_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        num_views: int = 1
    ) -> torch.Tensor:
        """
        Complete forward pass: LLM output -> heatmaps.
        
        Args:
            hidden_states (torch.Tensor): LLM output [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            num_views (int): Number of camera views (default: 1)
            
        Returns:
            torch.Tensor: Generated heatmaps [batch_size, num_views, target_size, target_size]
        """
        batch_size = hidden_states.shape[0]
        
        # 1. Extract vision tokens from LLM output
        image_tokens = self.extract_vision_tokens(
            hidden_states, attention_mask, num_views=num_views
        )
        
        # 2. Reconstruct tokens into spatial features
        spatial_features = self.tokens_to_spatial_features(image_tokens, num_views)
        
        # 3. Generate heatmaps via upsampling
        heatmaps = self.generate_heatmap(spatial_features)
        
        # 4. Reshape to multi-view format
        heatmaps = heatmaps.view(batch_size, num_views, self.target_size, self.target_size)
        
        return heatmaps
    
    def forward_with_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        num_views: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns intermediate spatial features.

        Args:
            hidden_states (torch.Tensor): LLM output [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            num_views (int): Number of camera views (default: 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Generated heatmaps [batch_size, num_views, target_size, target_size]
                - Spatial features [batch_size*num_views, hidden_dim, patch_size, patch_size]
        """
        batch_size = hidden_states.shape[0]

        # Extract and process tokens
        image_tokens = self.extract_vision_tokens(
            hidden_states, attention_mask, num_views=num_views
        )
        spatial_features = self.tokens_to_spatial_features(image_tokens, num_views)

        # Generate heatmaps
        heatmaps = self.generate_heatmap(spatial_features)
        heatmaps = heatmaps.view(batch_size, num_views, self.target_size, self.target_size)

        return heatmaps, spatial_features

    def generate_frame_indexed_heatmaps(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        keyframe_indices: List[int],
        current_obs_frame: int = None
    ) -> Dict[int, torch.Tensor]:
        """
        Generate frame-indexed heatmaps showing spatial relationships between keyframes and current observation.

        This method creates separate heatmaps for each keyframe, where each heatmap shows where that specific
        keyframe's content would appear in the current observation viewpoint. The heatmaps are indexed by
        their original video frame numbers from VGGT space-aware sampling.

        Args:
            hidden_states (torch.Tensor): LLM output [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            keyframe_indices (List[int]): Original frame numbers from VGGT sampling (e.g., [5, 17, 23, 34, 51, 78, 89, 95])
            current_obs_frame (int, optional): Original frame number of current observation (e.g., 100 or 89)

        Returns:
            Dict[int, torch.Tensor]: Dictionary mapping original frame indices to heatmaps
                {
                    5: tensor[batch_size, target_size, target_size],   # Frame 5's content in current view
                    17: tensor[batch_size, target_size, target_size],  # Frame 17's content in current view
                    23: tensor[batch_size, target_size, target_size],  # Frame 23's content in current view
                    ...
                }
        """
        batch_size = hidden_states.shape[0]
        num_keyframes = len(keyframe_indices)

        if num_keyframes == 0:
            raise ValueError("keyframe_indices cannot be empty")

        # Validate inputs
        if current_obs_frame is not None and current_obs_frame < 0:
            raise ValueError(f"current_obs_frame must be non-negative, got {current_obs_frame}")

        for idx in keyframe_indices:
            if idx < 0:
                raise ValueError(f"All keyframe indices must be non-negative, got {idx}")

        # Extract vision tokens for all keyframes
        # Assume LLM processes tokens in the same order as keyframe_indices
        total_image_tokens = 256 * num_keyframes  # 256 tokens per frame
        image_tokens = self.extract_vision_tokens(
            hidden_states, attention_mask,
            num_image_tokens=total_image_tokens,
            num_views=1
        )

        # Generate frame-specific heatmaps
        frame_heatmaps = {}
        tokens_per_frame = 256

        for i, original_frame_idx in enumerate(keyframe_indices):
            # Extract tokens for this specific frame
            start_idx = i * tokens_per_frame
            end_idx = (i + 1) * tokens_per_frame

            if end_idx > image_tokens.shape[1]:
                # Handle edge case: insufficient tokens
                remaining_tokens = image_tokens.shape[1] - start_idx
                if remaining_tokens <= 0:
                    # Create zero heatmap for missing frame
                    frame_heatmaps[original_frame_idx] = torch.zeros(
                        batch_size, self.target_size, self.target_size,
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    continue
                else:
                    # Use available tokens and pad
                    frame_tokens = torch.zeros(batch_size, tokens_per_frame, self.vlm_dim,
                                             device=hidden_states.device, dtype=hidden_states.dtype)
                    frame_tokens[:, :remaining_tokens] = image_tokens[:, start_idx:image_tokens.shape[1]]
            else:
                frame_tokens = image_tokens[:, start_idx:end_idx]  # [B, 256, D]

            # Convert tokens to spatial features for this frame
            spatial_features = self.tokens_to_spatial_features(frame_tokens, num_views=1)

            # Generate heatmap for this frame
            frame_heatmap = self.generate_heatmap(spatial_features)  # [B, 1, H, W]

            # Store with original frame index as key
            frame_heatmaps[original_frame_idx] = frame_heatmap.squeeze(1)  # [B, H, W]

        return frame_heatmaps


if __name__ == "__main__":
    print("Testing LLMToHeatmapConverter...")

    # Create converter
    converter = LLMToHeatmapConverter(vlm_dim=2048, patch_size=16, target_size=224)

    # Mock LLM output
    batch_size = 2
    seq_len = 300  # Text + image tokens
    hidden_dim = 2048
    num_views = 3

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 250:] = 0  # Simulate padding

    print(f"Input hidden_states shape: {hidden_states.shape}")
    print(f"Input attention_mask shape: {attention_mask.shape}")

    # Test forward pass
    with torch.no_grad():
        heatmaps = converter(hidden_states, attention_mask, num_views=num_views)

    print(f"Output heatmaps shape: {heatmaps.shape}")
    print(f"Expected shape: {(batch_size, num_views, 224, 224)}")
    print(f"Shape matches: {heatmaps.shape == (batch_size, num_views, 224, 224)}")
    print(f"Heatmap value range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

    # Test forward with features
    with torch.no_grad():
        heatmaps, features = converter.forward_with_features(
            hidden_states, attention_mask, num_views=num_views
        )

    print(f"Spatial features shape: {features.shape}")
    print(f"Expected features shape: {(batch_size * num_views, 2048, 16, 16)}")

    # Test new frame-indexed heatmap generation
    print("\n--- Testing Frame-Indexed Heatmap Generation ---")

    # Mock VGGT keyframe selection result
    keyframe_indices = [5, 17, 23, 34, 51, 78, 89, 95]  # Original frame numbers from VGGT
    current_obs_frame = 100  # Current observation frame

    print(f"Keyframe indices: {keyframe_indices}")
    print(f"Current observation frame: {current_obs_frame}")

    # Test frame-indexed heatmap generation
    with torch.no_grad():
        frame_heatmaps = converter.generate_frame_indexed_heatmaps(
            hidden_states, attention_mask, keyframe_indices, current_obs_frame
        )

    print(f"Generated {len(frame_heatmaps)} frame-indexed heatmaps")
    print("Frame heatmap shapes:")
    for frame_idx, heatmap in frame_heatmaps.items():
        print(f"  Frame {frame_idx}: {heatmap.shape}")

    # Validate output format
    expected_frames = set(keyframe_indices)
    actual_frames = set(frame_heatmaps.keys())
    print(f"Expected frame indices: {expected_frames}")
    print(f"Actual frame indices: {actual_frames}")
    print(f"Frame indices match: {expected_frames == actual_frames}")

    # Validate tensor shapes
    all_shapes_correct = True
    for frame_idx, heatmap in frame_heatmaps.items():
        expected_shape = (batch_size, 224, 224)
        if heatmap.shape != expected_shape:
            print(f"ERROR: Frame {frame_idx} has incorrect shape {heatmap.shape}, expected {expected_shape}")
            all_shapes_correct = False

    if all_shapes_correct:
        print("✓ All heatmap shapes are correct")
    else:
        print("✗ Some heatmap shapes are incorrect")

    print("LLMToHeatmapConverter test completed successfully!")