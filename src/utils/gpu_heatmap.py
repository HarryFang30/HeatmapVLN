"""
GPU-accelerated heatmap computation using PyTorch

This module provides vectorized, GPU-compatible heatmap computation
that replaces the CPU-bound NumPy implementation for significant speedup.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GPUHeatmapComputer:
    """
    GPU-accelerated history heatmap computation.
    
    Key optimizations:
    1. Vectorized matrix operations (process all history poses in parallel)
    2. GPU tensor operations (can run on CUDA)
    3. Batched Gaussian rendering
    
    Usage:
        computer = GPUHeatmapComputer(hm_size=(64, 64), device='cuda')
        heatmaps = computer.compute_batch(
            history_poses,  # [B, K, 4, 4]
            current_poses,  # [B, 4, 4]
            current_depths, # [B, H, W] or None
            intrinsics,     # [B, 3, 3] or None
        )
    """
    
    def __init__(
        self,
        hm_size: Tuple[int, int] = (64, 64),
        img_size: Tuple[int, int] = (640, 480),
        device: str = 'cuda',
        max_visible_distance: float = 15.0,
        occlusion_tolerance: float = 0.5,
        depth_min: float = 0.0,
        depth_max: float = 10.0,
        min_sigma: float = 1.5,
        max_sigma: float = 6.0,
        object_size_3d: float = 0.5,
        use_distance_decay: bool = True,
        distance_decay_ref: float = 5.0,
        min_peak_value: float = 0.3,
    ):
        self.hm_size = hm_size  # (H, W)
        self.img_size = img_size  # (W, H)
        self.device = device
        self.max_visible_distance = max_visible_distance
        self.occlusion_tolerance = occlusion_tolerance
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.object_size_3d = object_size_3d
        self.use_distance_decay = use_distance_decay
        self.distance_decay_ref = distance_decay_ref
        self.min_peak_value = min_peak_value
        
        # Pre-compute default intrinsics (HFOV=90°)
        img_w, img_h = img_size
        hfov_rad = math.radians(90.0)
        fx = img_w / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        self.default_K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)
        
        # Pre-compute heatmap coordinate grid
        Hm, Wm = hm_size
        y_coords = torch.arange(Hm, dtype=torch.float32, device=device)
        x_coords = torch.arange(Wm, dtype=torch.float32, device=device)
        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Shape: [Hm, Wm]
    
    def compute_batch(
        self,
        history_poses: torch.Tensor,      # [B, K, 4, 4]
        current_poses: torch.Tensor,      # [B, 4, 4]
        current_depths: Optional[torch.Tensor] = None,  # [B, Hd, Wd]
        intrinsics: Optional[torch.Tensor] = None,      # [B, 3, 3]
        depth_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute heatmaps for a batch of samples.
        
        Args:
            history_poses: History frame poses [B, K, 4, 4]
            current_poses: Current frame poses [B, 4, 4]
            current_depths: Depth maps for occlusion [B, Hd, Wd], optional
            intrinsics: Camera intrinsics [B, 3, 3], optional (uses default if None)
            depth_normalized: Whether depth is normalized to [0, 1]
        
        Returns:
            heatmaps: [B, Hm, Wm] normalized heatmaps
        """
        B, K, _, _ = history_poses.shape
        Hm, Wm = self.hm_size
        device = history_poses.device
        
        # Use default intrinsics if not provided
        if intrinsics is None:
            K_mat = self.default_K.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
            img_w, img_h = self.img_size
        else:
            K_mat = intrinsics.to(device)
            # 从 intrinsics 推断图像尺寸 (cx ≈ img_w/2, cy ≈ img_h/2)
            # 取 batch 中第一个样本的值（假设 batch 内一致）
            cx = K_mat[0, 0, 2].item()
            cy = K_mat[0, 1, 2].item()
            img_w = int(cx * 2)
            img_h = int(cy * 2)
        
        # Extract fx for sigma computation
        fx = K_mat[:, 0, 0]  # [B]
        
        # Compute inverse of current poses: T_current_inv [B, 4, 4]
        T_current_inv = torch.linalg.inv(current_poses)  # [B, 4, 4]
        
        # Extract history camera centers in world coordinates
        # history_poses[..., :3, 3] gives translation [B, K, 3]
        hist_centers_world = torch.cat([
            history_poses[..., :3, 3],  # [B, K, 3]
            torch.ones(B, K, 1, device=device, dtype=history_poses.dtype)  # [B, K, 1]
        ], dim=-1)  # [B, K, 4]
        
        # Transform to current camera coordinates
        # T_current_inv @ hist_center = [B, 4, 4] @ [B, K, 4, 1] -> [B, K, 4]
        p_cam = torch.einsum('bij,bkj->bki', T_current_inv, hist_centers_world)  # [B, K, 4]
        
        # Extract x, y, z in camera coordinates
        x_cam = p_cam[..., 0]  # [B, K]
        y_cam = p_cam[..., 1]  # [B, K]
        z_cam = p_cam[..., 2]  # [B, K]
        
        # Compute distances
        distances = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)  # [B, K]
        
        # Habitat camera: X right, Y up, -Z forward
        # Points in front of camera have z < 0; convert to positive depth
        z_depth = -z_cam  # [B, K] positive depth values
        
        # Validity masks
        valid_distance = (distances > 1e-4) & (distances < self.max_visible_distance)
        valid_in_front = z_depth > 0.1  # at least 0.1m in front
        
        # Pinhole projection with Habitat conventions
        fx_batch = K_mat[:, 0, 0].unsqueeze(1)  # [B, 1]
        fy_batch = K_mat[:, 1, 1].unsqueeze(1)  # [B, 1]
        cx_batch = K_mat[:, 0, 2].unsqueeze(1)  # [B, 1]
        cy_batch = K_mat[:, 1, 2].unsqueeze(1)  # [B, 1]
        
        z_safe = torch.where(z_depth > 0.1, z_depth, torch.ones_like(z_depth))
        
        u = fx_batch * x_cam / z_safe + cx_batch           # [B, K]
        v = fy_batch * (-y_cam) / z_safe + cy_batch        # [B, K] Y axis flipped
        
        # Check if projected points are within image bounds
        valid_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        
        # Combine validity masks
        valid = valid_distance & valid_in_front & valid_bounds  # [B, K]
        
        # Occlusion detection (if depth provided)
        if current_depths is not None:
            depth_h, depth_w = current_depths.shape[1], current_depths.shape[2]
            
            u_d = u * (depth_w / img_w)
            v_d = v * (depth_h / img_h)
            
            u_d = torch.clamp(u_d, 0, depth_w - 1).long()
            v_d = torch.clamp(v_d, 0, depth_h - 1).long()
            
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
            observed_depth = current_depths[batch_idx, v_d, u_d]  # [B, K]
            
            if depth_normalized:
                observed_depth = self.depth_min + observed_depth * (self.depth_max - self.depth_min)
            
            valid_depth = (observed_depth > 0) & torch.isfinite(observed_depth)
            not_occluded = observed_depth >= (z_depth - self.occlusion_tolerance)
            
            valid = valid & valid_depth & not_occluded
        
        # Adaptive sigma (in heatmap coordinate space)
        sigma = (self.object_size_3d * fx_batch / z_safe) * (Wm / img_w)  # [B, K]
        sigma = torch.clamp(sigma, self.min_sigma, self.max_sigma)
        
        # Distance decay: near points have peak ~1.0, far points decay to min_peak_value
        if self.use_distance_decay:
            decay = 1.0 / (1.0 + distances / self.distance_decay_ref)
            peak_values = self.min_peak_value + (1.0 - self.min_peak_value) * decay  # [B, K]
        else:
            peak_values = torch.ones_like(distances)
        
        # Convert to heatmap coordinates
        u_hm = u * Wm / img_w  # [B, K]
        v_hm = v * Hm / img_h  # [B, K]
        
        # Render Gaussians (max merge, with per-point peak values)
        heatmaps = self._render_gaussians_batch(u_hm, v_hm, sigma, valid, peak_values)
        
        return heatmaps
    
    def _render_gaussians_batch(
        self,
        u_hm: torch.Tensor,       # [B, K]
        v_hm: torch.Tensor,       # [B, K]
        sigma: torch.Tensor,      # [B, K]
        valid: torch.Tensor,      # [B, K]
        peak_values: torch.Tensor, # [B, K]
    ) -> torch.Tensor:
        """
        Render batch of Gaussian heatmaps using max merge.

        Each Gaussian is scaled by its peak_value (from distance decay),
        then combined via element-wise max to avoid saturation in
        overlapping regions.
        """
        B, K = u_hm.shape
        Hm, Wm = self.hm_size
        device = u_hm.device
        
        grid_x = self.grid_x.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, Hm, Wm]
        grid_y = self.grid_y.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, Hm, Wm]
        
        u_hm = u_hm.unsqueeze(-1).unsqueeze(-1)           # [B, K, 1, 1]
        v_hm = v_hm.unsqueeze(-1).unsqueeze(-1)           # [B, K, 1, 1]
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)          # [B, K, 1, 1]
        valid = valid.unsqueeze(-1).unsqueeze(-1).float()  # [B, K, 1, 1]
        peak_values = peak_values.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        
        dx = grid_x - u_hm  # [B, K, Hm, Wm]
        dy = grid_y - v_hm  # [B, K, Hm, Wm]
        
        sigma_safe = torch.where(sigma > 1e-6, sigma, torch.ones_like(sigma))
        
        gaussians = peak_values * torch.exp(-(dx**2 + dy**2) / (2 * sigma_safe**2))
        gaussians = gaussians * valid  # [B, K, Hm, Wm]
        
        # Max merge over K history points (consistent with CPU version)
        heatmaps = gaussians.max(dim=1)[0]  # [B, Hm, Wm]
        
        return heatmaps
    
    def compute_single(
        self,
        history_poses: torch.Tensor,     # [K, 4, 4]
        current_pose: torch.Tensor,      # [4, 4]
        current_depth: Optional[torch.Tensor] = None,  # [Hd, Wd]
        intrinsics: Optional[torch.Tensor] = None,     # [3, 3]
        depth_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute heatmap for a single sample.
        
        Convenience wrapper around compute_batch.
        """
        # Add batch dimension
        history_poses = history_poses.unsqueeze(0)  # [1, K, 4, 4]
        current_pose = current_pose.unsqueeze(0)    # [1, 4, 4]
        
        if current_depth is not None:
            current_depth = current_depth.unsqueeze(0)  # [1, Hd, Wd]
        
        if intrinsics is not None:
            intrinsics = intrinsics.unsqueeze(0)  # [1, 3, 3]
        
        heatmap = self.compute_batch(
            history_poses, current_pose, current_depth, intrinsics, depth_normalized
        )
        
        return heatmap.squeeze(0)  # [Hm, Wm]


def create_gpu_heatmap_computer(
    hm_size: Tuple[int, int] = (64, 64),
    img_size: Tuple[int, int] = (640, 480),
    device: str = 'cuda',
    **kwargs
) -> GPUHeatmapComputer:
    """Factory function to create GPUHeatmapComputer."""
    return GPUHeatmapComputer(
        hm_size=hm_size,
        img_size=img_size,
        device=device,
        **kwargs
    )


# ==================== Integration helpers ====================

def convert_poses_for_gpu(
    history_poses: list,  # List of [4, 4] numpy arrays
    current_pose,         # [4, 4] numpy array
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy poses to GPU tensors.
    
    Args:
        history_poses: List of K numpy arrays, each [4, 4]
        current_pose: Numpy array [4, 4]
        device: Target device
    
    Returns:
        history_poses_tensor: [K, 4, 4]
        current_pose_tensor: [4, 4]
    """
    import numpy as np
    
    history_poses_np = np.stack(history_poses, axis=0)  # [K, 4, 4]
    history_poses_tensor = torch.from_numpy(history_poses_np).float().to(device)
    current_pose_tensor = torch.from_numpy(np.array(current_pose)).float().to(device)
    
    return history_poses_tensor, current_pose_tensor
