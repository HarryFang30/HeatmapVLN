# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# Adapted from BridgeVLA: https://github.com/NVlabs/RVT/blob/master/rvt/mvt/raft_utils.py
# Therefore, the code is also under the NVIDIA Source Code License

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexUpSample(nn.Module):
    """
    Learned convex upsampling module for generating high-resolution heatmaps.
    
    This module takes low-resolution feature maps (e.g., 16x16) and upsamples them
    to high-resolution heatmaps (e.g., 224x224) using learned convex combinations.
    
    Key features:
    - Learned upsampling masks for smooth interpolation
    - Convex combination ensures stable gradients
    - Configurable upsampling ratio and kernel size
    """

    def __init__(
        self, in_dim, out_dim, up_ratio, up_kernel=3, mask_scale=0.1, with_bn=False
    ):
        """
        Initialize the ConvexUpSample module.

        Args:
            in_dim (int): Input feature dimension
            out_dim (int): Output feature dimension  
            up_ratio (int): Upsampling ratio (e.g., 14 for 16->224)
            up_kernel (int): Kernel size for convex combination (default: 3)
            mask_scale (float): Scale factor for upsampling masks (default: 0.1)
            with_bn (bool): Whether to use batch normalization (default: False)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        self.with_bn = with_bn

        assert (self.up_kernel % 2) == 1, "up_kernel must be odd"

        if with_bn:
            self.net_out_bn1 = nn.BatchNorm2d(2 * in_dim)
            self.net_out_bn2 = nn.BatchNorm2d(2 * in_dim)

        # Network for generating output features
        self.net_out = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, out_dim, 3, padding=1),
        )

        # Network for generating upsampling masks
        mask_dim = (self.up_ratio**2) * (self.up_kernel**2)
        self.net_mask = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, mask_dim, 1, padding=0),
        )

    def forward(self, x):
        """
        Forward pass of the ConvexUpSample module.

        Args:
            x (torch.Tensor): Input tensor of shape (bs, in_dim, h, w)

        Returns:
            torch.Tensor: Upsampled output of shape (bs, out_dim, h*up_ratio, w*up_ratio)
        """
        bs, c, h, w = x.shape
        assert c == self.in_dim, f"Expected {self.in_dim} input channels, got {c}"

        # Generate low resolution output features
        if self.with_bn:
            out_low = self.net_out[0](x)
            out_low = self.net_out_bn1(out_low)
            out_low = self.net_out[1](out_low)
            out_low = self.net_out[2](out_low)
            out_low = self.net_out_bn2(out_low)
            out_low = self.net_out[3](out_low)
            out_low = self.net_out[4](out_low)
        else:
            out_low = self.net_out(x)

        # Generate upsampling masks
        mask = self.mask_scale * self.net_mask(x)
        mask = mask.view(bs, 1, self.up_kernel**2, self.up_ratio, self.up_ratio, h, w)
        mask = torch.softmax(mask, dim=2)  # Convex combination weights

        # Extract patches using unfold operation
        out = F.unfold(
            out_low,
            kernel_size=[self.up_kernel, self.up_kernel],
            padding=self.up_kernel // 2,
        )
        out = out.view(bs, self.out_dim, self.up_kernel**2, 1, 1, h, w)

        # Apply convex combination
        out = torch.sum(out * mask, dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(bs, self.out_dim, h * self.up_ratio, w * self.up_ratio)

        return out


if __name__ == "__main__":
    # Test the ConvexUpSample module
    print("Testing ConvexUpSample module...")
    
    # Create module
    upsampler = ConvexUpSample(in_dim=2048, out_dim=1, up_ratio=14)
    
    # Test input: 16x16 feature map
    x = torch.randn(4, 2048, 16, 16)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        y = upsampler(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Expected shape: {(4, 1, 224, 224)}")
    print(f"Shape matches: {y.shape == (4, 1, 224, 224)}")
    print("ConvexUpSample test completed successfully!")