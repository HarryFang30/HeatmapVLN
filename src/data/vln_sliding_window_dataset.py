"""
VLN Sliding Window Dataset
===========================

将一段视频序列通过滑动窗口扩展为多个训练样本。
每个样本由"历史帧 + 当前帧"构成，输出热力图（当前帧中历史帧的位置）和下一步动作。

核心思路：
- 一段 T 帧的视频，可以生成 T - min_history 个训练样本
- 样本 i: 历史帧 [0, i-1]，当前帧 i，动作 i → i+1

Returns:
    {
        "history_frames": [K, 3, H, W],    # K 帧历史
        "current_frame": [3, H, W],        # 当前观测
        "heatmap": [Hm, Wm],               # 历史帧在当前帧的位置
        "action": [2],                     # 下一步动作 (dx, dy)
        "action_valid": float,             # 是否有有效动作 (0 or 1)
        "text": str,                       # 指令
    }
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import logging
import random

logger = logging.getLogger(__name__)

# ==================== 数据增强工具 ====================

class ColorJitterAugmentation:
    """
    颜色抖动增强 - 不影响几何关系，安全用于VLN任务
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: [H, W, 3] RGB image, uint8
        Returns:
            augmented image
        """
        if random.random() > self.p:
            return image

        image = image.astype(np.float32)

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = image * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation (convert to HSV)
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv = cv2.cvtColor(image.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Hue shift
        if self.hue > 0:
            shift = random.uniform(-self.hue, self.hue) * 180  # OpenCV hue range is 0-180
            hsv = cv2.cvtColor(image.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return np.clip(image, 0, 255).astype(np.uint8)


class GaussianNoiseAugmentation:
    """
    高斯噪声增强 - 增加模型对噪声的鲁棒性
    """

    def __init__(self, std: float = 10.0, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        noise = np.random.normal(0, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


# ==================== 热力图计算工具函数 (Pinhole 投影) ====================

def project_point_pinhole(
    p_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int
) -> Optional[Tuple[float, float, float]]:
    """
    将相机坐标系下的3D点投影到Pinhole图像坐标
    
    Args:
        p_cam: [x, y, z] 或 [x, y, z, 1] 相机坐标系下的点
        K: 3x3 相机内参矩阵
        width: 图像宽度
        height: 图像高度
    
    Returns:
        (u, v, z_depth) 像素坐标和深度，或 None 如果点在相机后方或太近
    
    Note:
        Habitat 相机坐标系：X 右，Y 上，-Z 前
        因此相机前方是 z < 0
    """
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    
    # 相机前方是 -Z 方向，所以 z < 0 才是在相机前方
    if z >= -0.1:  # 在相机后方或太近
        return None
    
    # 转换深度为正值
    z_depth = -z
    
    # Pinhole 投影
    # 注意 Y 轴方向：相机 Y 向上，图像 v 向下
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x / z_depth + cx
    v = fy * (-y) / z_depth + cy  # Y 轴翻转
    
    # 检查是否在图像范围内
    if not (0 <= u < width and 0 <= v < height):
        return None
    
    return float(u), float(v), float(z_depth)


def compute_adaptive_sigma_pinhole(
    z_depth: float,
    fx: float,
    object_size_3d: float = 0.5,
    heatmap_width: int = 64,
    img_width: int = 640,
    min_sigma: float = 0.8,
    max_sigma: float = 6.0
) -> float:
    """
    计算 Pinhole 投影下的自适应 sigma
    
    透视投影：物体在图像中的大小 = object_size * fx / z_depth
    sigma 约为投影大小的 1/3
    
    Args:
        z_depth: 点到相机的深度（米）
        fx: 相机焦距（像素）
        object_size_3d: 3D 物体大小（米）
        heatmap_width: 热力图宽度
        img_width: 原始图像宽度
        min_sigma: 最小 sigma
        max_sigma: 最大 sigma
    
    Returns:
        sigma 值
    """
    if z_depth <= 0.1:
        return float(max_sigma)
    
    # 在原图中的投影大小（像素）
    projected_size_img = object_size_3d * fx / z_depth
    
    # 转换到热力图坐标系
    scale = heatmap_width / img_width
    projected_size_hm = projected_size_img * scale
    
    # sigma 约为投影大小的 1/3
    sigma = projected_size_hm / 3.0
    sigma = np.clip(sigma, min_sigma, max_sigma)
    
    return float(sigma)


def draw_gaussian_point(
    heatmap: np.ndarray,
    center: Tuple[float, float],
    sigma: float,
) -> None:
    """
    在热力图上绘制高斯点（简化版，不使用 NeRF ripple）
    
    Args:
        heatmap: [H, W] 热力图数组
        center: (u, v) 中心坐标
        sigma: 高斯标准差
    """
    H, W = heatmap.shape
    u, v = center

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x_min = max(0, int(np.floor(u - radius)))
    x_max = min(W, int(np.ceil(u + radius)))
    y_min = max(0, int(np.floor(v - radius)))
    y_max = min(H, int(np.ceil(v + radius)))

    if x_min >= x_max or y_min >= y_max:
        return

    xs = np.arange(x_min, x_max, dtype=np.float32)
    ys = np.arange(y_min, y_max, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)

    # 高斯包络
    blob = np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    
    # 累加到热力图
    roi = heatmap[y_min:y_max, x_min:x_max]
    np.add(roi, blob, out=roi)


def compute_history_heatmap(
    history_poses: List[np.ndarray],
    current_pose: np.ndarray,
    current_depth: Optional[np.ndarray],
    hm_size: Tuple[int, int] = (64, 64),
    img_size: Tuple[int, int] = (640, 480),
    K: Optional[np.ndarray] = None,
    depth_normalize: bool = True,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    occlusion_tolerance: float = 0.5,
    max_visible_distance: float = 15.0,
) -> Tuple[np.ndarray, int]:
    """
    计算当前帧中历史帧相机位置的热力图 (Pinhole 投影版本)
    
    Args:
        history_poses: 历史帧的 4x4 位姿矩阵列表
        current_pose: 当前帧的 4x4 位姿矩阵
        current_depth: 当前帧的深度图（用于遮挡检测），可选
        hm_size: 热力图尺寸 (H, W)
        img_size: 原始图像尺寸 (W, H)
        K: 3x3 相机内参矩阵，如果为 None 则根据 img_size 和默认 HFOV=90° 计算
        depth_normalize: 深度是否归一化到 [0, 1]
        depth_min/max: 深度范围
        occlusion_tolerance: 遮挡容差（米）
        max_visible_distance: 最大可见距离（米）
    
    Returns:
        heatmap: [Hm, Wm] 归一化热力图
        visibility_count: 可见的历史帧数量
    """
    Hm, Wm = hm_size
    img_w, img_h = img_size
    
    heatmap = np.zeros((Hm, Wm), dtype=np.float32)
    visibility_count = 0
    
    # 如果没有提供内参，使用默认值（HFOV=90°）
    if K is None:
        hfov_rad = math.radians(90.0)
        fx = img_w / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    else:
        K = np.array(K, dtype=np.float32)
    
    fx = K[0, 0]
    
    # 当前帧位姿的逆（用于将历史帧位置转换到当前帧坐标系）
    T_current = np.array(current_pose, dtype=np.float32)
    T_current_inv = np.linalg.inv(T_current)
    
    # 处理深度图
    depth_plane = None
    if current_depth is not None:
        if current_depth.ndim == 3 and current_depth.shape[-1] == 1:
            depth_plane = current_depth[:, :, 0]
        else:
            depth_plane = current_depth
    
    for hist_pose in history_poses:
        T_hist = np.array(hist_pose, dtype=np.float32)
        
        # 历史帧相机中心（世界坐标系）
        hist_center_world = np.array([
            T_hist[0, 3], T_hist[1, 3], T_hist[2, 3], 1.0
        ], dtype=np.float32)
        
        # 转换到当前帧相机坐标系
        p_cam = T_current_inv @ hist_center_world
        
        # 计算距离
        distance = float(np.linalg.norm(p_cam[:3]))
        if distance < 1e-4 or distance > max_visible_distance:
            continue
        
        # 投影到图像坐标 (Pinhole)
        projection = project_point_pinhole(p_cam, K, img_w, img_h)
        if projection is None:
            continue
        u, v, z_depth = projection
        
        # 遮挡检测
        if depth_plane is not None:
            depth_h, depth_w = depth_plane.shape
            u_d = u * (depth_w / img_w)
            v_d = v * (depth_h / img_h)
            u_int = int(np.clip(u_d, 0, depth_w - 1))
            v_int = int(np.clip(v_d, 0, depth_h - 1))
            observed_depth = float(depth_plane[v_int, u_int])
            
            if depth_normalize:
                observed_depth = depth_min + observed_depth * (depth_max - depth_min)
            
            if not np.isfinite(observed_depth) or observed_depth <= 0:
                continue
            if observed_depth < z_depth - occlusion_tolerance:
                continue  # 被遮挡
        
        # 计算自适应 sigma (Pinhole 版本)
        sigma = compute_adaptive_sigma_pinhole(
            z_depth=z_depth,
            fx=fx,
            object_size_3d=0.5,
            heatmap_width=Wm,
            img_width=img_w,
            min_sigma=0.8,
            max_sigma=6.0
        )
        
        # 转换到热力图坐标
        u_hm = u * Wm / img_w
        v_hm = v * Hm / img_h
        
        # 绘制高斯点
        draw_gaussian_point(heatmap, (u_hm, v_hm), sigma)
        visibility_count += 1
    
    # 归一化
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    
    return heatmap, visibility_count


# ==================== 主数据集类 ====================

class VLNSlidingWindowDataset(Dataset):
    """
    滑动窗口数据集：一段视频生成多个训练样本
    
    将一段 T 帧的视频序列扩展为 T - min_history 个独立训练样本。
    每个样本由"历史帧 + 当前帧"构成。
    
    动作语义（重要）：
    - actions.npy: action[i] = 从 frame[i] 到 frame[i+1] 的 agent-local 2D 位移 (dx, dy)
    - discrete_actions.npy: discrete_action[i] = 从 frame[i] 到 frame[i+1] 的离散动作
      (0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT)
    - 最后一帧 action[T-1] = (0, 0) 且 discrete_action[T-1] = STOP，因为没有后续帧
    
    Args:
        root: 数据集根目录
        split: 数据集划分 ('train', 'val')
        min_history: 最小历史帧数（默认 5）
        num_history_sample: 从历史中采样的帧数（默认 8）
        image_size: 图像尺寸 (W, H)
        hm_size: 热力图尺寸 (W, H)
        load_depth: 是否加载深度图用于遮挡检测
        cache_poses: 是否缓存位姿数据
    
    Returns:
        {
            "history_frames": [K, 3, H, W],    # K 帧历史
            "current_frame": [3, H, W],        # 当前观测
            "heatmap": [Hm, Wm],               # 历史帧在当前帧的位置
            "action": [2],                     # 下一步动作 (dx, dy)
            "action_valid": float,             # 是否有有效动作 (0 or 1)
            "discrete_action": int,            # 离散动作 (0-3)
            "is_stop": float,                  # 是否为STOP动作 (0 or 1)
            "text": str,                       # 指令
        }
    """
    
    def __init__(
        self,
        root: str,
        split: str,
        min_history: int = 5,
        num_history_sample: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        hm_size: Tuple[int, int] = (64, 64),
        load_depth: bool = True,
        cache_poses: bool = True,
        sample_stride: int = 1,  # 采样步长：每隔 N 帧采样一次，1 表示不跳过
        enable_augmentation: bool = True,  # 是否启用数据增强
        # Clip-level 采样策略（解决样本高度相关性问题）
        samples_per_clip: int = 2,  # 每个 clip 每 epoch 采样的样本数
        clip_level_sampling: bool = True,  # 是否启用 clip-level 采样
        # 随机子序列采样（大幅增加数据多样性）
        random_subsequence: bool = False,  # 是否启用随机子序列采样
        min_subsequence_length: int = 30,  # 最小子序列长度
        subsequence_samples_per_clip: int = 3,  # 每个 clip 生成的子序列数量
        # GPU 热力图计算（将热力图计算延迟到 GPU）
        defer_heatmap_to_gpu: bool = False,  # 是否返回 poses 而非预计算的热力图
    ):
        self.root = Path(root).expanduser()
        self.split = split
        self.min_history = min_history
        self.num_history_sample = num_history_sample
        self.image_size = image_size  # (W, H)
        self.hm_size = hm_size        # (W, H)
        self.load_depth = load_depth
        self.cache_poses = cache_poses
        self.sample_stride = max(1, sample_stride)  # 采样步长
        
        # Clip-level 采样配置
        self.samples_per_clip = samples_per_clip
        self.clip_level_sampling = clip_level_sampling
        self._epoch = 0  # 当前 epoch，用于随机采样
        self._rng = np.random.RandomState(42)  # 可重复的随机数生成器
        
        # 随机子序列采样配置
        self.random_subsequence = random_subsequence and (split == 'train')  # 仅训练集启用
        self.min_subsequence_length = min_subsequence_length
        self.subsequence_samples_per_clip = subsequence_samples_per_clip
        
        # GPU 热力图计算配置
        self.defer_heatmap_to_gpu = defer_heatmap_to_gpu
        if defer_heatmap_to_gpu:
            logger.info("GPU heatmap computation enabled: returning poses instead of heatmaps")

        # 数据增强 (仅训练集启用)
        self.enable_augmentation = enable_augmentation and (split == 'train')
        if self.enable_augmentation:
            self.color_jitter = ColorJitterAugmentation(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5
            )
            self.gaussian_noise = GaussianNoiseAugmentation(std=8.0, p=0.3)
            logger.info("Data augmentation enabled: ColorJitter + GaussianNoise")
        
        # 枚举所有 clips
        self.clips = self._enumerate_clips()
        
        # 预计算每个 clip 的有效帧范围（用于 clip-level 采样）
        self._clip_valid_frames = {}  # clip_idx -> list of valid frame indices
        self._precompute_valid_frames()
        
        # 预计算样本索引
        self.sample_index = []  # [(clip_idx, current_frame_idx), ...]
        self._poses_cache = {}  # clip_idx -> poses
        self._meta_cache = {}   # clip_idx -> meta
        
        self._build_sample_index()
        
        if self.random_subsequence:
            sampling_mode = "random-subsequence"
        elif self.clip_level_sampling:
            sampling_mode = "clip-level"
        else:
            sampling_mode = "sliding-window"
        logger.info(
            f"VLNSlidingWindowDataset initialized: {len(self.clips)} clips, "
            f"{len(self.sample_index)} samples, mode={sampling_mode}, "
            f"samples_per_clip={self.samples_per_clip}, min_history={min_history}"
        )
    
    def _enumerate_clips(self) -> List[Path]:
        """枚举所有 clip 目录
        
        支持两种目录结构：
        1. root/split/scene/clip_xxx（标准结构）
        2. root/scene/clip_xxx（patrol_data 结构，无 split 层）
        """
        split_dir = self.root / self.split
        
        # 检查是否存在 split 目录
        if split_dir.exists():
            search_dir = split_dir
        else:
            # 尝试直接使用 root 目录（patrol_data 格式）
            logger.info(f"Split directory {split_dir} not found, using root directory directly")
            search_dir = self.root
        
        if not search_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {search_dir}")
        
        clips = []
        scene_dirs = [d for d in search_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            clip_dirs = sorted([
                d for d in scene_dir.iterdir()
                if d.is_dir() and d.name.startswith('clip_')
            ])
            clips.extend(clip_dirs)
        
        if len(clips) == 0:
            raise FileNotFoundError(f"No clips found in {search_dir}")
        
        logger.info(f"Found {len(clips)} clips in {len(scene_dirs)} scenes")
        return clips
    
    def _precompute_valid_frames(self):
        """预计算每个 clip 的有效帧索引列表（用于 clip-level 采样）
        
        有效帧 = 有足够历史帧的帧 (frame_idx >= min_history)
        """
        self._clip_valid_frames = {}
        
        for clip_idx, clip_dir in enumerate(self.clips):
            try:
                meta_file = clip_dir / "meta.json"
                if not meta_file.exists():
                    continue
                    
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                T = meta["num_frames"]
                # 有效帧：从 min_history 到 T-1
                valid_frames = list(range(self.min_history, T))
                
                if len(valid_frames) > 0:
                    self._clip_valid_frames[clip_idx] = valid_frames
                    
            except Exception as e:
                logger.warning(f"Failed to precompute valid frames for clip {clip_dir}: {e}")
                continue
        
        logger.info(f"Precomputed valid frames for {len(self._clip_valid_frames)} clips")
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch，触发 clip-level 重新采样
        
        每个 epoch 从每个 clip 随机选择不同的样本，
        确保模型看到不同的训练数据，减少过拟合。
        
        Args:
            epoch: 当前 epoch 编号
        """
        if not self.clip_level_sampling:
            return
            
        self._epoch = epoch
        self._rng = np.random.RandomState(42 + epoch)  # 每个 epoch 不同的随机种子
        self._build_sample_index()
        logger.info(f"[Epoch {epoch}] Resampled {len(self.sample_index)} samples from {len(self.clips)} clips")
    
    def _load_meta(self, clip_idx: int) -> Dict:
        """加载并缓存 clip 元数据"""
        if clip_idx in self._meta_cache:
            return self._meta_cache[clip_idx]
        
        clip_dir = self.clips[clip_idx]
        meta_file = clip_dir / "meta.json"
        
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        self._meta_cache[clip_idx] = meta
        return meta
    
    def _load_poses(self, clip_idx: int) -> List[np.ndarray]:
        """加载并缓存位姿数据"""
        if self.cache_poses and clip_idx in self._poses_cache:
            return self._poses_cache[clip_idx]
        
        clip_dir = self.clips[clip_idx]
        poses_file = clip_dir / "poses.json"
        
        if not poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {poses_file}")
        
        with open(poses_file, 'r') as f:
            poses_list = json.load(f)
        
        poses = [np.array(p, dtype=np.float32) for p in poses_list]
        
        if self.cache_poses:
            self._poses_cache[clip_idx] = poses
        
        return poses
    
    def _build_sample_index(self):
        """预计算所有样本的全局索引
        
        三种采样模式：
        1. 随机子序列采样（新增，推荐）：每个 clip 生成多个随机子序列
           - 大幅增加数据多样性
           - 同一个 100 帧 clip 可以生成无数种子序列组合
        
        2. Clip-level 采样：每个 clip 每 epoch 随机选择 N 个样本
           - 解决样本高度相关性问题
           - 每个 epoch 看到不同的样本组合
           - 通过 set_epoch() 触发重新采样
        
        3. 滑动窗口采样（传统）：使用 sample_stride 控制采样密度
           - stride=1: 每帧都作为样本
           - stride=5: 每隔 5 帧采样一次
        """
        self.sample_index = []
        # 存储每个样本的子序列范围：{idx: (subseq_start, subseq_end)}
        self._sample_subsequence_range = {}
        
        if self.random_subsequence:
            # ========== 随机子序列采样（新增） ==========
            total_subsequences = 0
            total_samples = 0
            
            for clip_idx in self._clip_valid_frames:
                valid_frames = self._clip_valid_frames[clip_idx]
                if len(valid_frames) == 0:
                    continue
                
                clip_start = valid_frames[0] - self.min_history  # 原始起始帧
                clip_end = valid_frames[-1] + 1  # 原始结束帧（不含）
                clip_length = clip_end - clip_start
                
                # 如果 clip 太短，不做子序列采样
                if clip_length < self.min_subsequence_length:
                    # 退化为普通采样
                    for frame_idx in valid_frames:
                        sample_idx = len(self.sample_index)
                        self.sample_index.append((clip_idx, frame_idx))
                        self._sample_subsequence_range[sample_idx] = (clip_start, clip_end)
                        total_samples += 1
                    continue
                
                # 生成多个随机子序列
                for _ in range(self.subsequence_samples_per_clip):
                    # 随机选择子序列长度
                    max_subseq_len = clip_length
                    min_subseq_len = self.min_subsequence_length
                    subseq_length = self._rng.randint(min_subseq_len, max_subseq_len + 1)
                    
                    # 随机选择子序列起始位置
                    max_start = clip_end - subseq_length
                    subseq_start = self._rng.randint(clip_start, max_start + 1)
                    subseq_end = subseq_start + subseq_length
                    
                    # 在子序列范围内采样
                    subseq_valid_start = subseq_start + self.min_history
                    subseq_valid_end = subseq_end
                    
                    if subseq_valid_end <= subseq_valid_start:
                        continue
                    
                    # 每个子序列采样 samples_per_clip 个样本
                    subseq_valid_frames = list(range(subseq_valid_start, subseq_valid_end))
                    num_samples = min(self.samples_per_clip, len(subseq_valid_frames))
                    
                    if num_samples > 0:
                        sampled_frames = self._rng.choice(
                            subseq_valid_frames,
                            size=num_samples,
                            replace=False
                        )
                        for frame_idx in sampled_frames:
                            sample_idx = len(self.sample_index)
                            self.sample_index.append((clip_idx, frame_idx))
                            self._sample_subsequence_range[sample_idx] = (subseq_start, subseq_end)
                            total_samples += 1
                    
                    total_subsequences += 1
            
            # 打乱样本顺序（需要同时打乱 sample_index 和 _sample_subsequence_range）
            indices = list(range(len(self.sample_index)))
            self._rng.shuffle(indices)
            self.sample_index = [self.sample_index[i] for i in indices]
            new_range = {}
            for new_idx, old_idx in enumerate(indices):
                new_range[new_idx] = self._sample_subsequence_range[old_idx]
            self._sample_subsequence_range = new_range
            
            logger.info(
                f"Built random subsequence sample index: {len(self.sample_index)} samples "
                f"from {total_subsequences} subsequences, {len(self._clip_valid_frames)} clips, "
                f"min_subseq_len={self.min_subsequence_length}, epoch={self._epoch}"
            )
        
        elif self.clip_level_sampling:
            # ========== Clip-level 采样 ==========
            stop_samples = 0
            normal_samples = 0
            
            for clip_idx in self._clip_valid_frames:
                valid_frames = self._clip_valid_frames[clip_idx]
                if len(valid_frames) == 0:
                    continue
                
                # 最后一帧是 STOP 帧，需要特殊处理
                last_frame = valid_frames[-1]
                non_stop_frames = valid_frames[:-1] if len(valid_frames) > 1 else []
                
                # 1. 始终包含 STOP 帧（最后一帧）
                self.sample_index.append((clip_idx, last_frame))
                stop_samples += 1
                
                # 2. 从非 STOP 帧中随机采样 (samples_per_clip - 1) 个
                num_normal_samples = min(self.samples_per_clip - 1, len(non_stop_frames))
                if num_normal_samples > 0:
                    sampled_indices = self._rng.choice(
                        non_stop_frames, 
                        size=num_normal_samples, 
                        replace=False
                    )
                    for frame_idx in sampled_indices:
                        self.sample_index.append((clip_idx, frame_idx))
                        normal_samples += 1
            
            # 打乱样本顺序
            self._rng.shuffle(self.sample_index)
            
            logger.info(
                f"Built clip-level sample index: {len(self.sample_index)} samples "
                f"({stop_samples} STOP + {normal_samples} normal) from {len(self._clip_valid_frames)} clips, "
                f"epoch={self._epoch}"
            )
        else:
            # ========== 滑动窗口采样（传统模式） ==========
            stop_frames_added = 0
            
            for clip_idx, clip_dir in enumerate(self.clips):
                try:
                    meta = self._load_meta(clip_idx)
                    T = meta["num_frames"]
                    
                    # 每个 clip 可生成 T - min_history 个样本
                    for t in range(self.min_history, T, self.sample_stride):
                        self.sample_index.append((clip_idx, t))
                    
                    # 确保最后一帧（STOP）被采样
                    last_frame = T - 1
                    if last_frame >= self.min_history:
                        if (last_frame - self.min_history) % self.sample_stride != 0:
                            self.sample_index.append((clip_idx, last_frame))
                            stop_frames_added += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to index clip {clip_dir}: {e}")
                    continue
            
            logger.info(
                f"Built sliding-window sample index: {len(self.sample_index)} samples from {len(self.clips)} clips "
                f"(stride={self.sample_stride}, added {stop_frames_added} STOP frames)"
            )
    
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def _sample_history_indices(self, start: int, end: int, num_samples: int) -> np.ndarray:
        """
        从 [start, end) 范围内均匀采样 num_samples 个索引
        
        Args:
            start: 起始索引（包含）
            end: 结束索引（不包含）
            num_samples: 采样数量，-1 表示使用全部帧
        
        Returns:
            采样的索引数组
        """
        available = end - start
        if available <= 0:
            return np.array([], dtype=int)
        
        # num_samples == -1 表示使用全部帧
        if num_samples == -1 or available <= num_samples:
            # 返回所有可用帧
            return np.arange(start, end, dtype=int)
        else:
            # 均匀采样
            indices = np.linspace(start, end - 1, num_samples, dtype=int)
            return indices
    
    def _load_frame(self, clip_dir: Path, frame_idx: int, apply_augmentation: bool = True) -> torch.Tensor:
        """加载单帧图像"""
        # 支持 .jpg 和 .png 两种格式（patrol_data 使用 .jpg）
        rgb_path = clip_dir / "rgb" / f"{frame_idx:06d}.jpg"
        if not rgb_path.exists():
            rgb_path = clip_dir / "rgb" / f"{frame_idx:06d}.png"

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB file not found: {clip_dir / 'rgb' / f'{frame_idx:06d}.{{jpg,png}}'}")

        image = cv2.imread(str(rgb_path))
        if image is None:
            raise ValueError(f"Failed to load image: {rgb_path}")

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        target_w, target_h = self.image_size
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))

        # 应用数据增强 (仅训练集)
        if apply_augmentation and self.enable_augmentation:
            image = self.color_jitter(image)
            image = self.gaussian_noise(image)

        # 转换为 tensor 并归一化
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        return image_tensor
    
    def _load_frames(self, clip_dir: Path, frame_indices: np.ndarray) -> torch.Tensor:
        """加载多帧图像"""
        frames = []
        for idx in frame_indices:
            frame = self._load_frame(clip_dir, idx)
            frames.append(frame)
        
        return torch.stack(frames, dim=0)  # [K, C, H, W]
    
    def _load_depth(self, clip_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
        """加载深度图"""
        if not self.load_depth:
            return None
        
        depth_path = clip_dir / "depth" / f"{frame_idx:06d}.npy"
        
        if not depth_path.exists():
            return None
        
        try:
            depth = np.load(depth_path)
            return depth
        except Exception as e:
            logger.warning(f"Failed to load depth: {depth_path}: {e}")
            return None
    
    def _load_actions(self, clip_dir: Path) -> Optional[np.ndarray]:
        """加载连续动作数据 (dx, dy)"""
        actions_path = clip_dir / "actions.npy"
        
        if not actions_path.exists():
            return None
        
        try:
            actions = np.load(actions_path)
            return actions
        except Exception as e:
            logger.warning(f"Failed to load actions: {actions_path}: {e}")
            return None
    
    def _load_discrete_actions(self, clip_dir: Path) -> Optional[np.ndarray]:
        """加载离散动作数据 (STOP=0, FORWARD=1, LEFT=2, RIGHT=3)"""
        discrete_path = clip_dir / "discrete_actions.npy"
        
        if not discrete_path.exists():
            return None
        
        try:
            discrete_actions = np.load(discrete_path)
            return discrete_actions
        except Exception as e:
            logger.warning(f"Failed to load discrete actions: {discrete_path}: {e}")
            return None
    
    def compute_action_stats(self, margin: float = 0.1) -> Tuple[List[float], List[float]]:
        """
        遍历数据集计算 action 的 min/max 统计值
        
        Args:
            margin: 安全余量百分比，默认 10%
            
        Returns:
            (min_vals, max_vals): 每个维度的最小值和最大值列表
        """
        logger.info(f"Computing action statistics from {len(self.clips)} clips...")
        
        all_actions = []
        valid_clips = 0
        
        for clip_dir in self.clips:
            actions = self._load_actions(clip_dir)
            if actions is not None and len(actions) > 0:
                all_actions.append(actions)
                valid_clips += 1
        
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            
            # 计算 min/max
            raw_min = all_actions.min(axis=0)
            raw_max = all_actions.max(axis=0)
            range_size = raw_max - raw_min
            
            # 添加安全余量
            min_val = raw_min - margin * range_size
            max_val = raw_max + margin * range_size
            
            logger.info(
                f"Action stats computed from {valid_clips} clips, {len(all_actions)} actions:\n"
                f"  Raw range: min={raw_min.tolist()}, max={raw_max.tolist()}\n"
                f"  With {margin*100:.0f}% margin: min={min_val.tolist()}, max={max_val.tolist()}"
            )
            
            return min_val.tolist(), max_val.tolist()
        
        # Fallback 默认值
        logger.warning("No actions found in dataset, using default action stats")
        return [-0.17, -0.03], [0.19, 0.31]
    
    def get_action_valid_ratio(self) -> float:
        """
        计算数据集中有效动作的比例
        
        Returns:
            有效动作占总样本的比例 (0.0 - 1.0)
        """
        valid_count = 0
        total_count = len(self.sample_index)
        
        for clip_idx, current_t in self.sample_index:
            clip_dir = self.clips[clip_idx]
            meta = self._load_meta(clip_idx)
            T = meta["num_frames"]
            actions = self._load_actions(clip_dir)
            
            if actions is not None and current_t < T - 1:
                valid_count += 1
        
        ratio = valid_count / max(total_count, 1)
        logger.info(f"Action valid ratio: {valid_count}/{total_count} = {ratio*100:.1f}%")
        return ratio
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        """
        加载一个训练样本
        
        Args:
            idx: 样本索引
        
        Returns:
            {
                "history_frames": [K, 3, H, W],
                "current_frame": [3, H, W],
                "heatmap": [Hm, Wm],
                "action": [2],
                "action_valid": float,
                "text": str,
            }
        """
        clip_idx, current_t = self.sample_index[idx]
        clip_dir = self.clips[clip_idx]
        
        try:
            # 1. 加载元数据
            meta = self._load_meta(clip_idx)
            T = meta["num_frames"]
            text = meta.get("instruction", "")
            
            # 2. 采样历史帧索引
            history_indices = self._sample_history_indices(0, current_t, self.num_history_sample)
            
            # 3. 加载历史帧
            history_frames = self._load_frames(clip_dir, history_indices)
            
            # 4. 加载当前帧
            current_frame = self._load_frame(clip_dir, current_t)
            
            # 5. 加载位姿
            poses = self._load_poses(clip_idx)
            history_poses = [poses[i] for i in history_indices]
            current_pose = poses[current_t]
            
            # 6. 加载当前帧深度（用于遮挡检测）
            current_depth = self._load_depth(clip_dir, current_t)
            
            # 7. 获取相机内参
            intrinsics_path = clip_dir / "intrinsics.json"
            K = None
            if intrinsics_path.exists():
                with open(intrinsics_path) as f:
                    intrinsics = json.load(f)
                img_size = (intrinsics["width"], intrinsics["height"])
                if "K" in intrinsics:
                    K = np.array(intrinsics["K"], dtype=np.float32)
            else:
                img_size = (640, 480)  # 默认 Pinhole 图像尺寸
            
            # 7b. 计算热力图或返回 poses（GPU 延迟计算模式）
            hm_w, hm_h = self.hm_size
            if self.defer_heatmap_to_gpu:
                # GPU 模式：返回 poses，在训练循环中用 GPU 计算热力图
                history_poses_np = np.stack(history_poses, axis=0)  # [K, 4, 4]
                current_pose_np = np.array(current_pose)            # [4, 4]
                history_poses_tensor = torch.from_numpy(history_poses_np).float()
                current_pose_tensor = torch.from_numpy(current_pose_np).float()
                
                # 深度图转 tensor（如果有）
                if current_depth is not None:
                    depth_tensor = torch.from_numpy(current_depth).float()
                else:
                    # placeholder: 使用与热力图相同尺寸，避免 stack 时形状不匹配
                    depth_tensor = torch.zeros(hm_h, hm_w)
                
                # 内参转 tensor
                if K is not None:
                    intrinsics_tensor = torch.from_numpy(K).float()
                else:
                    intrinsics_tensor = torch.zeros(3, 3)  # 使用默认值的标志
                
                # 返回占位热力图（实际计算在 GPU 上完成）
                heatmap_tensor = torch.zeros(hm_h, hm_w)
                has_depth = current_depth is not None
                has_intrinsics = K is not None
            else:
                # CPU 模式：直接计算热力图
                heatmap, visibility = compute_history_heatmap(
                    history_poses=history_poses,
                    current_pose=current_pose,
                    current_depth=current_depth,
                    hm_size=(hm_h, hm_w),
                    img_size=img_size,
                    K=K,
                )
                heatmap_tensor = torch.from_numpy(heatmap).float()
                # GPU 模式所需的 placeholder
                history_poses_tensor = torch.zeros(1)
                current_pose_tensor = torch.zeros(1)
                depth_tensor = torch.zeros(1)
                intrinsics_tensor = torch.zeros(1)
                has_depth = False
                has_intrinsics = False
            
            # 8. 加载连续动作
            actions = self._load_actions(clip_dir)
            if actions is not None and current_t < len(actions):
                # 动作语义（来自 collect.py）：
                # actions[i] = 从 frame[i] 到 frame[i+1] 的 agent-local 2D 位移 (dx, dy)
                # 因此对于 current_t 帧，应该加载 actions[current_t]
                action = actions[current_t]
                # 🔧 修复：对于最后一帧，如果是 STOP 动作，action_valid 应该为 1
                # 因为 STOP 是一个有效的决策，不应该被 mask 掉
                if current_t == T - 1:
                    # 最后一帧：检查是否是 STOP 动作
                    discrete_actions = self._load_discrete_actions(clip_dir)
                    is_last_frame_stop = (discrete_actions is not None and 
                                          current_t < len(discrete_actions) and 
                                          int(discrete_actions[current_t]) == 0)
                    action_valid = 1.0 if is_last_frame_stop else 0.0
                else:
                    action_valid = 1.0
            else:
                action = np.zeros(2, dtype=np.float32)
                action_valid = 0.0
            
            action_tensor = torch.from_numpy(action.astype(np.float32))
            
            # 9. 加载离散动作（用于 Stop Prediction）
            discrete_actions = self._load_discrete_actions(clip_dir)
            if discrete_actions is not None and current_t < len(discrete_actions):
                # 动作语义：discrete_actions[i] = 从 frame[i] 到 frame[i+1] 的离散动作
                # 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
                discrete_action = int(discrete_actions[current_t])
                # is_stop: 1 if STOP action, 0 otherwise
                is_stop = 1.0 if discrete_action == 0 else 0.0
            else:
                discrete_action = 1  # Default to FORWARD
                is_stop = 0.0
            
            result = {
                "history_frames": history_frames,      # [K, 3, H, W]
                "current_frame": current_frame,        # [3, H, W]
                "heatmap": heatmap_tensor,             # [Hm, Wm]
                "action": action_tensor,               # [2]
                "action_valid": action_valid,          # float
                "discrete_action": discrete_action,    # int (0-3)
                "is_stop": is_stop,                    # float (0 or 1)
                "text": text,                          # str
            }
            
            # GPU 热力图计算所需的额外字段
            if self.defer_heatmap_to_gpu:
                result["history_poses"] = history_poses_tensor  # [K, 4, 4]
                result["current_pose"] = current_pose_tensor    # [4, 4]
                result["current_depth"] = depth_tensor          # [Hd, Wd] or [1]
                result["intrinsics"] = intrinsics_tensor        # [3, 3] or [1]
                result["has_depth"] = has_depth                 # bool
                result["has_intrinsics"] = has_intrinsics       # bool
                result["img_size"] = img_size                   # (W, H) tuple
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} (clip {clip_idx}, t={current_t}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self) -> Dict[str, Union[torch.Tensor, str, float, int]]:
        """生成虚拟样本（用于错误处理）"""
        target_w, target_h = self.image_size
        hm_w, hm_h = self.hm_size
        K = self.num_history_sample
        
        result = {
            "history_frames": torch.zeros(K, 3, target_h, target_w),
            "current_frame": torch.zeros(3, target_h, target_w),
            "heatmap": torch.zeros(hm_h, hm_w),
            "action": torch.zeros(2),
            "action_valid": 0.0,
            "discrete_action": 1,  # Default to FORWARD
            "is_stop": 0.0,
            "text": "",
        }
        
        # GPU 热力图计算所需的额外字段
        if self.defer_heatmap_to_gpu:
            result["history_poses"] = torch.zeros(K, 4, 4)
            result["current_pose"] = torch.zeros(4, 4)
            result["current_depth"] = torch.zeros(1)
            result["intrinsics"] = torch.zeros(3, 3)
            result["has_depth"] = False
            result["has_intrinsics"] = False
            result["img_size"] = (640, 480)
        
        return result


def create_sliding_window_dataloader(
    root: str,
    split: str,
    min_history: int = 5,
    num_history_sample: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    hm_size: Tuple[int, int] = (64, 64),
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    创建滑动窗口数据集的 DataLoader
    
    Args:
        root: 数据集根目录
        split: train/val
        min_history: 最小历史帧数
        num_history_sample: 采样的历史帧数
        image_size: 图像尺寸 (W, H)
        hm_size: 热力图尺寸 (W, H)
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否锁页内存
        drop_last: 是否丢弃最后不完整批次
    
    Returns:
        DataLoader
    """
    dataset = VLNSlidingWindowDataset(
        root=root,
        split=split,
        min_history=min_history,
        num_history_sample=num_history_sample,
        image_size=image_size,
        hm_size=hm_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# ==================== 轨迹处理工具函数（参考 InternNav）====================

def get_trajectory_relative_to_frame(extrinsics: np.ndarray, camera_deg: float = 0) -> np.ndarray:
    """
    计算相对于参考帧的轨迹位姿 (x, y, yaw)
    
    参考 InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py
    
    Args:
        extrinsics: 4x4 外参矩阵序列 [T_world2camera], shape: (n, 4, 4)
        camera_deg: 相机俯仰角度
    
    Returns:
        relative_xyyaw: 相对于参考帧的位姿序列 (x, y, yaw), shape: (n, 3)
    """
    # T_world2camera
    # 坐标变换矩阵
    T_camera2robot = np.array([
        [[0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ])
    
    T_robot2camera = np.array([
        [[0.0, 0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ])
    
    # 应用相机俯仰角度变换
    if camera_deg is not None and camera_deg != 0:
        camera_rad = np.radians(camera_deg)
        T_deg = np.array([
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
             [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ], dtype=np.float32)
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera[0])[np.newaxis]
    
    # 转换到机器人坐标系
    extrinsics_robot = np.matmul(extrinsics, T_camera2robot[0])
    
    # 获取参考帧的变换矩阵并计算其逆
    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)
    
    # 计算所有帧相对于参考帧的变换
    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)
    
    # 提取相对位姿
    relative_translations = relative_to_ref[:, :2, 3]  # (x, y)
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])
    
    relative_xyyaw = np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)
    
    return relative_xyyaw


def smooth_and_resample_trajectory(points: np.ndarray, sample_length: int = 25, interval: float = 0.1) -> np.ndarray:
    """
    对轨迹进行平滑和重采样
    
    参考 InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py
    
    Args:
        points: 2D 轨迹点, shape: (n, 2)
        sample_length: 采样长度
        interval: 采样间隔（米）
    
    Returns:
        resampled: 重采样后的轨迹点, shape: (sample_length, 2)
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        logger.warning("scipy not available, using linear interpolation")
        # Fallback to linear interpolation
        if len(points) == 0:
            return np.zeros((sample_length, 2))
        if len(points) == 1:
            return np.tile(points[0], (sample_length, 1))
        indices = np.linspace(0, len(points) - 1, sample_length)
        return np.array([points[int(i)] for i in indices])
    
    total_distance = sample_length * interval
    
    if len(points) == 0:
        return np.zeros((sample_length, 2))
    
    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))
    
    # 计算原始轨迹的累积距离
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    
    # 使用三次样条插值进行平滑
    if len(points) > 3:
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])
        
        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))
        
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances
    
    target_distances = np.linspace(0, total_distance, sample_length)
    resampled = np.zeros((sample_length, 2))
    
    for i, target_dist in enumerate(target_distances):
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue
        
        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1
        segment_idx = max(0, min(segment_idx, len(smooth_cumulative_distances) - 2))
        
        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]
        
        if end_dist > start_dist:
            t = (target_dist - start_dist) / (end_dist - start_dist)
        else:
            t = 0
        
        resampled[i] = smoothed_points[segment_idx] + t * (
            smoothed_points[min(segment_idx + 1, len(smoothed_points) - 1)] - smoothed_points[segment_idx]
        )
    
    return resampled


def xy_to_delta_xyt(xy_actions: np.ndarray) -> np.ndarray:
    """
    计算 (dx, dy, delta_yaw)
    
    参考 InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py
    
    Args:
        xy_actions: 绝对位置序列, shape: (N, 2)
    
    Returns:
        delta_xyt: 增量动作, shape: (N-1, 3)
    """
    if len(xy_actions) < 2:
        return np.zeros((max(0, len(xy_actions) - 1), 3), dtype=np.float32)
    
    vectors = np.diff(xy_actions, axis=0)  # [N-1, 2]
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])  # [N-1]
    
    if len(yaw) < 2:
        delta_yaw = yaw.copy() if len(yaw) > 0 else np.array([0.0])
    else:
        delta_yaw = np.diff(yaw)
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        delta_yaw = np.concatenate([[yaw[0]], delta_yaw])
    
    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt.astype(np.float32)


def interpolate_and_resample_trajectory(
    absolute_trajectories: np.ndarray, 
    predict_step_num: int = 24,
    action_scale: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    插值和重采样轨迹
    
    参考 InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py
    
    Args:
        absolute_trajectories: 绝对轨迹 (x, y, yaw), shape: (N, 3)
        predict_step_num: 预测步数
        action_scale: 动作放大倍数
    
    Returns:
        resampled_trajectories: 重采样后的绝对轨迹, shape: (predict_step_num+1, 2)
        resampled_relative_poses: 重采样后的相对位姿, shape: (predict_step_num, 3)
    """
    start_point = np.array([[0.0, 0.0]])
    
    traj = absolute_trajectories[..., :2]
    
    # 过滤有效步（距离平方 > 0.05）
    if len(traj) > 1:
        steps = traj[1:] - traj[:-1]
        steps_sq = (steps**2).sum(axis=-1)
        mask = steps_sq > 0.05
        
        filtered_traj = traj[1:][mask]
        filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)
    else:
        filtered_traj = start_point
    
    resampled_trajectories = smooth_and_resample_trajectory(
        filtered_traj, sample_length=predict_step_num + 1
    )
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)
    
    # 放大动作（参考 InternNav）
    resampled_relative_poses[:, 0:2] *= action_scale
    
    return resampled_trajectories, resampled_relative_poses


def apply_trajectory_augmentation(
    trajectory: np.ndarray,
    rotation_range: float = 0.3,  # 旋转范围（弧度）
    scale_range: Tuple[float, float] = (0.8, 1.2),  # 缩放范围
    p: float = 0.5,  # 应用概率
) -> np.ndarray:
    """
    轨迹增强：随机旋转和缩放
    
    Args:
        trajectory: 轨迹 (N, 3) - (dx, dy, delta_yaw)
        rotation_range: 旋转范围
        scale_range: 缩放范围
        p: 应用概率
    
    Returns:
        augmented: 增强后的轨迹
    """
    if random.random() > p:
        return trajectory
    
    augmented = trajectory.copy()
    
    # 随机旋转
    if random.random() > 0.5:
        angle = random.uniform(-rotation_range, rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # 旋转 dx, dy
        dx = augmented[:, 0].copy()
        dy = augmented[:, 1].copy()
        augmented[:, 0] = dx * cos_a - dy * sin_a
        augmented[:, 1] = dx * sin_a + dy * cos_a
        
        # 调整 yaw
        augmented[:, 2] += angle
        augmented[:, 2] = (augmented[:, 2] + np.pi) % (2 * np.pi) - np.pi
    
    # 随机缩放
    if random.random() > 0.5:
        scale = random.uniform(*scale_range)
        augmented[:, 0:2] *= scale
    
    return augmented


# ==================== 轨迹数据集（支持 24 步预测）====================

class VLNTrajectoryDataset(VLNSlidingWindowDataset):
    """
    轨迹数据集：支持多步轨迹预测
    
    基于 VLNSlidingWindowDataset，增加以下功能：
    1. 24 步轨迹预测（参考 InternNav）
    2. 3D 动作表示 (dx, dy, delta_yaw) + 4x 放大
    3. 轨迹增强
    4. Progress 预测
    
    Args:
        predict_horizon: 预测步数（默认 24）
        action_scale: 动作放大倍数（默认 4.0）
        enable_trajectory_augmentation: 是否启用轨迹增强
        其他参数继承自 VLNSlidingWindowDataset
    
    Returns:
        {
            "history_frames": [K, 3, H, W],    # K 帧历史
            "current_frame": [3, H, W],        # 当前观测
            "heatmap": [Hm, Wm],               # 历史帧在当前帧的位置
            "trajectory": [predict_horizon, 3], # 24 步轨迹 (dx, dy, yaw)
            "trajectory_valid": float,          # 轨迹是否有效
            "progress": float,                  # 任务完成进度 (0-1)
            "text": str,                        # 指令
        }
    """
    
    def __init__(
        self,
        root: str,
        split: str,
        min_history: int = 5,
        num_history_sample: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        hm_size: Tuple[int, int] = (64, 64),
        load_depth: bool = True,
        cache_poses: bool = True,
        sample_stride: int = 1,
        enable_augmentation: bool = True,
        samples_per_clip: int = 2,
        clip_level_sampling: bool = True,
        # 随机子序列采样配置
        random_subsequence: bool = False,
        min_subsequence_length: int = 30,
        subsequence_samples_per_clip: int = 3,
        # 轨迹预测相关配置
        predict_horizon: int = 24,
        action_scale: float = 4.0,
        enable_trajectory_augmentation: bool = True,
        # FGR2R 子指令配置
        fgr2r_subinstr_path: Optional[str] = None,
        use_subinstruction: bool = False,
    ):
        super().__init__(
            root=root,
            split=split,
            min_history=min_history,
            num_history_sample=num_history_sample,
            image_size=image_size,
            hm_size=hm_size,
            load_depth=load_depth,
            cache_poses=cache_poses,
            sample_stride=sample_stride,
            enable_augmentation=enable_augmentation,
            samples_per_clip=samples_per_clip,
            clip_level_sampling=clip_level_sampling,
            random_subsequence=random_subsequence,
            min_subsequence_length=min_subsequence_length,
            subsequence_samples_per_clip=subsequence_samples_per_clip,
        )
        
        self.predict_horizon = predict_horizon
        self.action_scale = action_scale
        self.enable_trajectory_augmentation = enable_trajectory_augmentation and (split == 'train')
        
        # 加载 FGR2R 子指令映射表
        self.use_subinstruction = use_subinstruction
        self._fgr2r_mapping = {}
        if use_subinstruction:
            self._load_fgr2r_mapping(fgr2r_subinstr_path)
        
        logger.info(
            f"VLNTrajectoryDataset initialized: predict_horizon={predict_horizon}, "
            f"action_scale={action_scale}, trajectory_aug={self.enable_trajectory_augmentation}, "
            f"random_subseq={self.random_subsequence}, use_subinstr={self.use_subinstruction}"
        )
    
    def _load_fgr2r_mapping(self, fgr2r_path: Optional[str] = None):
        """加载 FGR2R 子指令映射表（支持 .json 和 .json.gz 格式）"""
        import gzip
        
        if fgr2r_path is None:
            # 默认路径：项目目录下的 data/fgr2r/subinstr_mapping.json.gz
            fgr2r_path = Path(__file__).parent.parent.parent / "data/fgr2r/subinstr_mapping.json.gz"
        else:
            fgr2r_path = Path(fgr2r_path)
            # 如果是相对路径，相对于项目根目录
            if not fgr2r_path.is_absolute():
                fgr2r_path = Path(__file__).parent.parent.parent / fgr2r_path
        
        # 检查是否存在 .gz 版本
        if not fgr2r_path.exists() and not str(fgr2r_path).endswith('.gz'):
            gz_path = Path(str(fgr2r_path) + '.gz')
            if gz_path.exists():
                fgr2r_path = gz_path
        
        try:
            if str(fgr2r_path).endswith('.gz'):
                with gzip.open(fgr2r_path, 'rt', encoding='utf-8') as f:
                    mapping = json.load(f)
            else:
                with open(fgr2r_path, 'r') as f:
                    mapping = json.load(f)
            # 转换 key 为 int（JSON 键是字符串）
            self._fgr2r_mapping = {int(k): v for k, v in mapping.items()}
            logger.info(f"Loaded FGR2R subinstruction mapping: {len(self._fgr2r_mapping)} trajectories from {fgr2r_path}")
        except FileNotFoundError:
            logger.warning(f"FGR2R mapping file not found: {fgr2r_path}")
            self._fgr2r_mapping = {}
            self.use_subinstruction = False
        except Exception as e:
            logger.warning(f"Failed to load FGR2R mapping: {e}")
            self._fgr2r_mapping = {}
            self.use_subinstruction = False
    
    def _get_subinstruction(
        self, 
        trajectory_id: int, 
        num_frames: int,
        current_t: int, 
        subseq_start: int, 
        subseq_end: int,
        original_instruction: str,
        instr_idx: int = 0,  # 使用第几条原始指令
    ) -> str:
        """
        根据帧范围获取对应的子指令
        
        Args:
            trajectory_id: 轨迹 ID（对应 FGR2R 的 path_id）
            num_frames: 总帧数
            current_t: 当前帧索引
            subseq_start: 子序列起始帧
            subseq_end: 子序列结束帧
            original_instruction: 原始完整指令（备用）
            instr_idx: 使用第几条原始指令（默认第 0 条）
        
        Returns:
            对应的子指令文本
        """
        if not self.use_subinstruction or trajectory_id not in self._fgr2r_mapping:
            return original_instruction
        
        fgr2r_item = self._fgr2r_mapping[trajectory_id]
        num_viewpoints = fgr2r_item['num_viewpoints']
        instructions = fgr2r_item['instructions']
        
        if num_viewpoints < 2 or not instructions:
            return original_instruction
        
        # 计算帧到 viewpoint 的映射
        # viewpoint 之间均匀分布帧
        frames_per_segment = num_frames / (num_viewpoints - 1)
        
        # 将帧范围转换为 viewpoint 范围（1-based）
        # subseq_start 对应的 viewpoint（向下取整 + 1）
        vp_start = int(subseq_start / frames_per_segment) + 1
        # subseq_end 对应的 viewpoint（向上取整 + 1）
        vp_end = min(int((subseq_end - 1) / frames_per_segment) + 2, num_viewpoints + 1)
        
        # 选择指令（如果请求的索引超出范围，使用第 0 条）
        if instr_idx >= len(instructions):
            instr_idx = 0
        instr_data = instructions[instr_idx]
        sub_instructions = instr_data['sub_instructions']
        
        if not sub_instructions:
            return instr_data['original']
        
        # 收集与帧范围有重叠的子指令
        matching_subs = []
        for sub in sub_instructions:
            sub_vp_start, sub_vp_end = sub['viewpoint_range']
            # 检查是否有重叠
            if sub_vp_start < vp_end and sub_vp_end > vp_start:
                matching_subs.append(sub['text'])
        
        if matching_subs:
            # 将匹配的子指令连接起来
            return ' '.join(matching_subs)
        else:
            # 没有匹配的子指令，返回第一个子指令
            return sub_instructions[0]['text'] if sub_instructions else original_instruction
    
    def _compute_trajectory(
        self, 
        poses: List[np.ndarray], 
        current_t: int, 
        subseq_end: int,
        subseq_start: int = 0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        从位姿计算轨迹
        
        Args:
            poses: 所有帧的位姿列表
            current_t: 当前帧索引
            subseq_end: 子序列结束帧（不含）
            subseq_start: 子序列起始帧（用于计算 progress）
        
        Returns:
            trajectory: (predict_horizon, 3) 轨迹
            trajectory_valid: 轨迹是否有效
            progress: 任务完成进度 (0-1)，基于子序列范围
        """
        # 计算 progress（对齐 InternNav 的定义）
        # InternNav: stop_progress = (np.arange(total_steps) + 1) / total_steps
        # 即 progress = (step + 1) / total_steps，从 1/N 开始，到 N/N=1 结束
        subseq_length = subseq_end - subseq_start
        relative_pos = current_t - subseq_start
        progress = float(relative_pos + 1) / max(subseq_length, 1)
        
        # 获取从当前帧到子序列结束的位姿
        # ⚠️ 重要：必须使用 subseq_end 而不是整个 clip 结束
        # 否则 progress 和轨迹会不一致（progress=1 但轨迹非零）
        future_poses = poses[current_t:subseq_end]
        
        if len(future_poses) < 2:
            # 最后一帧或没有足够的未来帧 - 返回零轨迹（静止/STOP）
            # 注意：对于 STOP 帧，零轨迹是有意义的（表示静止不动）
            # 因此标记为有效 (trajectory_valid=1.0)，让模型学习 "停止时输出零轨迹"
            return np.zeros((self.predict_horizon, 3), dtype=np.float32), 1.0, progress
        
        # 转换为 numpy 数组
        future_poses_np = np.array(future_poses, dtype=np.float32)
        
        # 计算相对于当前帧的轨迹
        try:
            relative_xyyaw = get_trajectory_relative_to_frame(future_poses_np, camera_deg=0)
            
            # 插值和重采样到 predict_horizon 步
            _, resampled_poses = interpolate_and_resample_trajectory(
                relative_xyyaw,
                predict_step_num=self.predict_horizon,
                action_scale=self.action_scale,
            )
            
            # 确保形状正确
            if len(resampled_poses) < self.predict_horizon:
                # 填充
                pad_size = self.predict_horizon - len(resampled_poses)
                resampled_poses = np.concatenate([
                    resampled_poses,
                    np.zeros((pad_size, 3), dtype=np.float32)
                ], axis=0)
            else:
                resampled_poses = resampled_poses[:self.predict_horizon]
            
            trajectory_valid = 1.0
            
        except Exception as e:
            logger.warning(f"Failed to compute trajectory: {e}")
            resampled_poses = np.zeros((self.predict_horizon, 3), dtype=np.float32)
            trajectory_valid = 0.0
        
        return resampled_poses.astype(np.float32), trajectory_valid, progress
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        """
        加载一个训练样本（带轨迹）
        """
        clip_idx, current_t = self.sample_index[idx]
        clip_dir = self.clips[clip_idx]
        
        try:
            # 1. 加载元数据
            meta = self._load_meta(clip_idx)
            T = meta["num_frames"]
            original_text = meta.get("instruction", "")
            trajectory_id = int(meta.get("trajectory_id", 0))
            
            # 获取子序列范围（如果启用了随机子序列采样）
            if self.random_subsequence and idx in self._sample_subsequence_range:
                subseq_start, subseq_end = self._sample_subsequence_range[idx]
            else:
                subseq_start, subseq_end = 0, T
            
            # 1.5 获取子指令（如果启用了 FGR2R 子指令）
            if self.use_subinstruction:
                text = self._get_subinstruction(
                    trajectory_id=trajectory_id,
                    num_frames=T,
                    current_t=current_t,
                    subseq_start=subseq_start,
                    subseq_end=subseq_end,
                    original_instruction=original_text,
                )
            else:
                text = original_text
            
            # 2. 采样历史帧索引（使用子序列范围）
            history_indices = self._sample_history_indices(subseq_start, current_t, self.num_history_sample)
            
            # 3. 加载历史帧
            history_frames = self._load_frames(clip_dir, history_indices)
            
            # 4. 加载当前帧
            current_frame = self._load_frame(clip_dir, current_t)
            
            # 5. 加载位姿
            poses = self._load_poses(clip_idx)
            history_poses = [poses[i] for i in history_indices]
            current_pose = poses[current_t]
            
            # 6. 加载当前帧深度（用于遮挡检测）
            current_depth = self._load_depth(clip_dir, current_t)
            
            # 7. 计算热力图
            intrinsics_path = clip_dir / "intrinsics.json"
            K = None
            if intrinsics_path.exists():
                with open(intrinsics_path) as f:
                    intrinsics = json.load(f)
                img_size = (intrinsics["width"], intrinsics["height"])
                if "K" in intrinsics:
                    K = np.array(intrinsics["K"], dtype=np.float32)
            else:
                img_size = (640, 480)  # 默认 Pinhole 图像尺寸
            
            hm_w, hm_h = self.hm_size
            heatmap, visibility = compute_history_heatmap(
                history_poses=history_poses,
                current_pose=current_pose,
                current_depth=current_depth,
                hm_size=(hm_h, hm_w),
                img_size=img_size,
                K=K,
            )
            heatmap_tensor = torch.from_numpy(heatmap).float()
            
            # 8. 计算轨迹（使用子序列范围计算 progress）
            trajectory, trajectory_valid, progress = self._compute_trajectory(
                poses, current_t, subseq_end, subseq_start
            )
            
            # 9. 应用轨迹增强
            if self.enable_trajectory_augmentation and trajectory_valid > 0:
                trajectory = apply_trajectory_augmentation(trajectory, p=0.5)
            
            trajectory_tensor = torch.from_numpy(trajectory).float()
            
            # 10. 保留旧的 action 接口用于兼容
            actions = self._load_actions(clip_dir)
            if actions is not None and current_t < len(actions):
                action = actions[current_t]
            else:
                action = np.zeros(2, dtype=np.float32)
            
            # 🔧 修复：让 action_valid 与 trajectory_valid 保持一致
            # 这样 tensorboard 记录的 action_valid_ratio 能正确反映有效样本比例
            action_valid = trajectory_valid
            
            action_tensor = torch.from_numpy(action.astype(np.float32))
            
            # 11. 离散动作
            discrete_actions = self._load_discrete_actions(clip_dir)
            if discrete_actions is not None and current_t < len(discrete_actions):
                discrete_action = int(discrete_actions[current_t])
                is_stop = 1.0 if discrete_action == 0 else 0.0
            else:
                discrete_action = 1
                is_stop = 0.0
            
            return {
                "history_frames": history_frames,        # [K, 3, H, W]
                "current_frame": current_frame,          # [3, H, W]
                "heatmap": heatmap_tensor,               # [Hm, Wm]
                "trajectory": trajectory_tensor,         # [predict_horizon, 3]
                "trajectory_valid": trajectory_valid,    # float
                "progress": progress,                    # float (0-1)
                # 兼容旧接口
                "action": action_tensor,                 # [2]
                "action_valid": action_valid,            # float
                "discrete_action": discrete_action,      # int
                "is_stop": is_stop,                      # float
                "text": text,                            # str
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} (clip {clip_idx}, t={current_t}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_dummy_sample_trajectory()
    
    def _get_dummy_sample_trajectory(self) -> Dict[str, Union[torch.Tensor, str, float, int]]:
        """生成虚拟样本（用于错误处理）"""
        base_sample = self._get_dummy_sample()
        base_sample["trajectory"] = torch.zeros(self.predict_horizon, 3)
        base_sample["trajectory_valid"] = 0.0
        base_sample["progress"] = 0.0
        return base_sample


def create_trajectory_dataloader(
    root: str,
    split: str,
    min_history: int = 5,
    num_history_sample: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    hm_size: Tuple[int, int] = (64, 64),
    predict_horizon: int = 24,
    action_scale: float = 4.0,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    创建轨迹数据集的 DataLoader
    """
    dataset = VLNTrajectoryDataset(
        root=root,
        split=split,
        min_history=min_history,
        num_history_sample=num_history_sample,
        image_size=image_size,
        hm_size=hm_size,
        predict_horizon=predict_horizon,
        action_scale=action_scale,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

