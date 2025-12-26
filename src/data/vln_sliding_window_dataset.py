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

logger = logging.getLogger(__name__)


# ==================== 热力图计算工具函数 ====================

def project_point_equirect(p_cam: np.ndarray, width: int, height: int) -> Optional[Tuple[float, float]]:
    """
    将相机坐标系下的3D点投影到Equirectangular图像坐标
    
    Args:
        p_cam: [x, y, z] 或 [x, y, z, 1] 相机坐标系下的点
        width: 图像宽度
        height: 图像高度
    
    Returns:
        (u, v) 像素坐标，或 None 如果点太近
    """
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-6:
        return None

    phi = math.atan2(x, -z)  # 水平角
    theta = math.asin(np.clip(y / r, -1.0, 1.0))  # 垂直角

    u = (phi + math.pi) / (2.0 * math.pi) * width
    v = (0.5 - (theta / math.pi)) * height

    u = u % width
    v = np.clip(v, 0.0, height - 1e-6)
    return float(u), float(v)


def compute_adaptive_sigma(
    distance: float,
    object_size_3d: float = 0.3,
    heatmap_width: int = 64,
    min_sigma: float = 0.5,
    max_sigma: float = 5.0
) -> float:
    """计算Equirectangular坐标系下的自适应sigma"""
    if distance <= 1e-4:
        return float(max_sigma)

    angular_radius = math.atan2(object_size_3d, distance)
    pixels_per_rad = heatmap_width / (2.0 * math.pi)
    projected_radius_heatmap = angular_radius * pixels_per_rad
    sigma = projected_radius_heatmap / 3.0
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
    img_size: Tuple[int, int] = (512, 256),
    depth_normalize: bool = True,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    occlusion_tolerance: float = 0.25,
    max_visible_distance: float = 15.0,
) -> Tuple[np.ndarray, int]:
    """
    计算当前帧中历史帧相机位置的热力图
    
    Args:
        history_poses: 历史帧的 4x4 位姿矩阵列表
        current_pose: 当前帧的 4x4 位姿矩阵
        current_depth: 当前帧的深度图（用于遮挡检测），可选
        hm_size: 热力图尺寸 (H, W)
        img_size: 原始图像尺寸 (W, H)
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
        
        # 投影到图像坐标
        projection = project_point_equirect(p_cam, img_w, img_h)
        if projection is None:
            continue
        u, v = projection
        
        if not (0.0 <= v < img_h):
            continue
        
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
            if observed_depth < distance - occlusion_tolerance:
                continue  # 被遮挡
        
        # 计算自适应 sigma
        sigma = compute_adaptive_sigma(
            distance=distance,
            object_size_3d=0.5,
            heatmap_width=Wm,
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
    ):
        self.root = Path(root)
        self.split = split
        self.min_history = min_history
        self.num_history_sample = num_history_sample
        self.image_size = image_size  # (W, H)
        self.hm_size = hm_size        # (W, H)
        self.load_depth = load_depth
        self.cache_poses = cache_poses
        self.sample_stride = max(1, sample_stride)  # 采样步长
        
        # 枚举所有 clips
        self.clips = self._enumerate_clips()
        
        # 预计算样本索引
        self.sample_index = []  # [(clip_idx, current_frame_idx), ...]
        self._poses_cache = {}  # clip_idx -> poses
        self._meta_cache = {}   # clip_idx -> meta
        
        self._build_sample_index()
        
        logger.info(
            f"VLNSlidingWindowDataset initialized: {len(self.clips)} clips, "
            f"{len(self.sample_index)} samples, min_history={min_history}, "
            f"num_history_sample={num_history_sample}, sample_stride={self.sample_stride}"
        )
    
    def _enumerate_clips(self) -> List[Path]:
        """枚举所有 clip 目录"""
        split_dir = self.root / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        clips = []
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            clip_dirs = sorted([
                d for d in scene_dir.iterdir()
                if d.is_dir() and d.name.startswith('clip_')
            ])
            clips.extend(clip_dirs)
        
        if len(clips) == 0:
            raise FileNotFoundError(f"No clips found in {split_dir}")
        
        logger.info(f"Found {len(clips)} clips in {len(scene_dirs)} scenes")
        return clips
    
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
        
        使用 sample_stride 控制采样密度：
        - stride=1: 每帧都作为样本（默认）
        - stride=5: 每隔 5 帧采样一次（样本数减少 5 倍）
        """
        self.sample_index = []
        
        for clip_idx, clip_dir in enumerate(self.clips):
            try:
                meta = self._load_meta(clip_idx)
                T = meta["num_frames"]
                
                # 每个 clip 可生成 T - min_history 个样本
                # 当前帧从 min_history 开始，到 T-1 结束
                # 使用 sample_stride 跳过部分帧
                for t in range(self.min_history, T, self.sample_stride):
                    self.sample_index.append((clip_idx, t))
                    
            except Exception as e:
                logger.warning(f"Failed to index clip {clip_dir}: {e}")
                continue
        
        logger.info(
            f"Built sample index: {len(self.sample_index)} samples from {len(self.clips)} clips "
            f"(stride={self.sample_stride})"
        )
    
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def _sample_history_indices(self, start: int, end: int, num_samples: int) -> np.ndarray:
        """
        从 [start, end) 范围内均匀采样 num_samples 个索引
        
        Args:
            start: 起始索引（包含）
            end: 结束索引（不包含）
            num_samples: 采样数量
        
        Returns:
            采样的索引数组
        """
        available = end - start
        if available <= 0:
            return np.array([], dtype=int)
        
        if available <= num_samples:
            # 帧数不够，返回所有可用帧
            return np.arange(start, end, dtype=int)
        else:
            # 均匀采样
            indices = np.linspace(start, end - 1, num_samples, dtype=int)
            return indices
    
    def _load_frame(self, clip_dir: Path, frame_idx: int) -> torch.Tensor:
        """加载单帧图像"""
        rgb_path = clip_dir / "rgb" / f"{frame_idx:06d}.png"
        
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        
        image = cv2.imread(str(rgb_path))
        if image is None:
            raise ValueError(f"Failed to load image: {rgb_path}")
        
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        target_w, target_h = self.image_size
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))
        
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
            
            # 7. 计算热力图
            # 获取原始图像尺寸（从 intrinsics 或使用默认值）
            intrinsics_path = clip_dir / "intrinsics.json"
            if intrinsics_path.exists():
                with open(intrinsics_path) as f:
                    intrinsics = json.load(f)
                img_size = (intrinsics["width"], intrinsics["height"])
            else:
                img_size = (512, 256)  # 默认全景图尺寸
            
            hm_w, hm_h = self.hm_size
            heatmap, visibility = compute_history_heatmap(
                history_poses=history_poses,
                current_pose=current_pose,
                current_depth=current_depth,
                hm_size=(hm_h, hm_w),
                img_size=img_size,
            )
            heatmap_tensor = torch.from_numpy(heatmap).float()
            
            # 8. 加载连续动作
            actions = self._load_actions(clip_dir)
            if actions is not None and current_t < T - 1:
                # 动作定义：从 current_t 到 current_t+1 的位移
                # actions[t] 存储的是从 t-1 到 t 的位移，所以取 actions[current_t + 1]
                action = actions[current_t + 1] if current_t + 1 < len(actions) else np.zeros(2)
                action_valid = 1.0
            else:
                action = np.zeros(2, dtype=np.float32)
                action_valid = 0.0
            
            action_tensor = torch.from_numpy(action.astype(np.float32))
            
            # 9. 加载离散动作（用于 Stop Prediction）
            discrete_actions = self._load_discrete_actions(clip_dir)
            if discrete_actions is not None and current_t < len(discrete_actions):
                discrete_action = int(discrete_actions[current_t])
                # is_stop: 1 if STOP action, 0 otherwise
                is_stop = 1.0 if discrete_action == 0 else 0.0
            else:
                discrete_action = 1  # Default to FORWARD
                is_stop = 0.0
            
            return {
                "history_frames": history_frames,      # [K, 3, H, W]
                "current_frame": current_frame,        # [3, H, W]
                "heatmap": heatmap_tensor,             # [Hm, Wm]
                "action": action_tensor,               # [2]
                "action_valid": action_valid,          # float
                "discrete_action": discrete_action,    # int (0-3)
                "is_stop": is_stop,                    # float (0 or 1)
                "text": text,                          # str
            }
            
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
        
        return {
            "history_frames": torch.zeros(K, 3, target_h, target_w),
            "current_frame": torch.zeros(3, target_h, target_w),
            "heatmap": torch.zeros(hm_h, hm_w),
            "action": torch.zeros(2),
            "action_valid": 0.0,
            "discrete_action": 1,  # Default to FORWARD
            "is_stop": 0.0,
            "text": "",
        }


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

